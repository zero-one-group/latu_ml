defmodule Latu.ML.Result do
  @moduledoc """
  Read what an `MlCommand` answered.

  Latu latches the whole `MlCommandResult` onto the finished execution rather than picking an
  arm, because which arm is set is part of the answer. This is where the picking happens.

  Clauses match maps rather than named messages, as `Latu.Result.Literal` does: the shape is
  what matters, and a pattern is not a call into the protocol layer.
  """

  alias Latu.Error
  alias Latu.ML.Linalg
  alias Latu.Result.Literal
  alias Latu.Result.UDT

  @doc """
  The object reference a `Fit` put in the ML cache.

  A `Fit` populates `obj_ref` and nothing else — `uid`, `params` and `warning_message` all come
  back nil — which is why a model's uid is the client's own, not the server's.
  """
  @spec object_ref(struct()) :: {:ok, Latu.ML.Plan.object_ref()} | {:error, Error.t()}
  def object_ref(execution)

  def object_ref(%{ml_command_result: %{result_type: result_type}})
      when elem(result_type, 0) == :operator_info do
    case elem(result_type, 1) do
      %{type: {:obj_ref, ref}} -> {:ok, ref.id}
      _info -> {:error, protocol_error("a Fit answered with no object reference")}
    end
  end

  def object_ref(execution), do: {:error, unexpected(execution, "an object reference")}

  @doc """
  The value an attribute `Fetch` answered with.

  A `Vector` or `Matrix` arrives as a UDT-typed struct literal and comes back as an
  `Nx.Tensor` or a `Latu.ML.SparseVector`; everything else is whatever
  `Latu.Result.Literal.value/1` makes of it.
  """
  @spec param(struct()) :: {:ok, term()} | {:error, Error.t()}
  def param(execution)

  def param(%{ml_command_result: %{result_type: {:param, literal}}}), do: decode(literal)

  def param(execution), do: {:error, unexpected(execution, "a param")}

  @typedoc """
  A loaded operator, as a `Read` described it.

  `ref` is set for a model and nil for everything else; `name` is the reverse. `params` are
  the values the saved metadata carried, in the shape `Latu.ML.Plan.params/1` takes them.
  """
  @type operator_info :: %{
          ref: String.t() | nil,
          name: String.t() | nil,
          uid: String.t() | nil,
          params: [Latu.ML.Plan.param()]
        }

  @doc """
  What a `Read` answered with: a loaded operator, as data.

  The one answer the server fills in. A `Fit` populates `obj_ref` and nothing else — a fitted
  model's uid and params are the client's own, copied from the estimator — where this carries
  `uid` and `params` too, because both came off the saved metadata and no client could have
  known them.

  A **model** carries a cache reference as well: reading one registers it in the session's ML
  cache exactly as a fit does, and it is the caller's to release. An estimator, transformer or
  evaluator carries its class name instead, and nothing is cached.

  Each param is typed from the arm the server chose rather than from the registry, so a class
  the registry never heard of loads as readily as one it knows. The two agree everywhere they
  both apply: PySpark re-types a loaded param through its declared converter, and every
  disagreement that could produce — a `:float` param, an `:int` past 32 bits — collapses to
  the same literal on the way back out.
  """
  @spec operator_info(struct()) :: {:ok, operator_info()} | {:error, Error.t()}
  def operator_info(execution)

  def operator_info(%{ml_command_result: %{result_type: result_type}})
      when elem(result_type, 0) == :operator_info do
    info = elem(result_type, 1)

    with {:ok, params} <- params(info.params) do
      {:ok, %{ref: ref(info.type), name: name(info.type), uid: info.uid, params: params}}
    end
  end

  def operator_info(execution), do: {:error, unexpected(execution, "an operator")}

  # A model is answered by reference and everything else by name; the oneof is which.
  defp ref({:obj_ref, ref}), do: ref.id
  defp ref(_other), do: nil

  defp name({:name, name}), do: name
  defp name(_other), do: nil

  # Sorted by wire name: the params are a map on the wire and a map has no order, so anything
  # downstream that prints or diffs them would otherwise vary between two identical loads.
  defp params(nil), do: {:ok, []}

  defp params(%{params: params}) do
    params
    |> Enum.sort_by(&elem(&1, 0))
    |> Enum.reduce_while({:ok, []}, fn {wire, literal}, {:ok, done} ->
      case param_entry(wire, literal) do
        {:ok, entry} -> {:cont, {:ok, [entry | done]}}
        {:error, error} -> {:halt, {:error, error}}
      end
    end)
    |> case do
      {:ok, entries} -> {:ok, Enum.reverse(entries)}
      {:error, error} -> {:error, error}
    end
  end

  defp param_entry(wire, literal) do
    with {:ok, type} <- param_type(wire, literal),
         {:ok, value} <- decode(literal) do
      {:ok, {wire, type, value}}
    end
  end

  # The type a param comes back as, read off the literal the server sent. A param is a bag of
  # literals on the wire with no declared type beside it, so the arm is the only thing that
  # says what one is — and it is enough, because every arm below rebuilds as itself.
  defp param_type(_wire, %{literal_type: {:integer, _}}), do: {:ok, :int}
  defp param_type(_wire, %{literal_type: {:long, _}}), do: {:ok, :long}
  defp param_type(_wire, %{literal_type: {:boolean, _}}), do: {:ok, :boolean}
  defp param_type(_wire, %{literal_type: {:string, _}}), do: {:ok, :string}

  # Elixir has no 32-bit float and `Latu.ML.Plan.literal/2` writes `:float` and `:double` to
  # the same arm, so the two collapse here as they do everywhere else in this package.
  defp param_type(_wire, %{literal_type: {kind, _}}) when kind in [:float, :double] do
    {:ok, :double}
  end

  defp param_type(wire, %{literal_type: {:array, array}} = literal) do
    case element_type(array.element_type) || element_type(array_type(literal.data_type)) do
      nil -> {:error, unreadable(wire, "an array whose element type it does not know")}
      type -> {:ok, {:list, type}}
    end
  end

  # A specialized array is six flat lists under one arm. It rebuilds as an ordinary `array`
  # literal rather than as itself — the server reads both, and one encoder is better than a
  # second that exists only for values that came from a load.
  defp param_type(wire, %{literal_type: {:specialized_array, %{value_type: value_type}}}) do
    case value_type do
      {:ints, _} -> {:ok, {:list, :int}}
      {:longs, _} -> {:ok, {:list, :long}}
      {:floats, _} -> {:ok, {:list, :double}}
      {:doubles, _} -> {:ok, {:list, :double}}
      {:bools, _} -> {:ok, {:list, :boolean}}
      {:strings, _} -> {:ok, {:list, :string}}
      _other -> {:error, unreadable(wire, "a specialized array of a kind it does not know")}
    end
  end

  defp param_type(wire, %{literal_type: {:struct, %{struct_type: struct_type}}} = literal) do
    case udt_class(literal.data_type) || udt_class(struct_type) do
      "org.apache.spark.ml.linalg.VectorUDT" -> {:ok, :vector}
      "org.apache.spark.ml.linalg.MatrixUDT" -> {:ok, :matrix}
      nil -> {:error, unreadable(wire, "a struct that names no type")}
      class -> {:error, unreadable(wire, "a #{class}, which is not a Vector or a Matrix")}
    end
  end

  defp param_type(wire, %{literal_type: {kind, _}}) do
    {:error, unreadable(wire, "the literal arm #{inspect(kind)}")}
  end

  defp param_type(wire, _literal), do: {:error, unreadable(wire, "a literal with nothing set")}

  defp element_type(%{kind: {:string, _}}), do: :string
  defp element_type(%{kind: {:boolean, _}}), do: :boolean
  defp element_type(%{kind: {:integer, _}}), do: :int
  defp element_type(%{kind: {:long, _}}), do: :long
  defp element_type(%{kind: {kind, _}}) when kind in [:float, :double], do: :double
  defp element_type(_other), do: nil

  defp array_type(%{kind: {:array, array}}), do: array.element_type
  defp array_type(_other), do: nil

  defp udt_class(%{kind: {:udt, udt}}), do: udt.jvm_class
  defp udt_class(_other), do: nil

  defp unreadable(wire, what) do
    protocol_error("the param #{wire} came back as #{what}")
  end

  @doc """
  The cache id a `Fetch` answered with, where the answer was another cached object.

  `MlCommandResult`'s third arm, and a bare string: asking a model for its `summary` gets the
  id of the summary rather than the summary itself, because a summary is a server-side object
  like the model is. `Latu.ML.summary/1` does not send this — it composes the reference from
  the model's own, as PySpark does — so this is here for the paths that do, `evaluate` first
  among them.
  """
  @spec summary(struct()) :: {:ok, String.t()} | {:error, Error.t()}
  def summary(execution)

  def summary(%{ml_command_result: %{result_type: {:summary, id}}}) when is_binary(id) do
    {:ok, id}
  end

  def summary(execution), do: {:error, unexpected(execution, "a summary")}

  @doc """
  The cache ids a `Fetch` answered with, where the answer was a *list* of cached objects.

  `MlCommandResult`'s `operator_info` arm again, but the server has packed several ids into one
  `ObjectRef` by joining them with commas — `MLHandler` does `ids.mkString(",")` and PySpark
  splits them apart. There is no repeated field to read instead, so this is the format rather
  than a convention.

  Each id is a **separate cache entry**: the server registered every sub-model on its way out,
  so a forest of a hundred trees is a hundred entries the caller now owns.
  """
  @spec operators(struct()) :: {:ok, [Latu.ML.Plan.object_ref()]} | {:error, Error.t()}
  def operators(execution)

  def operators(%{ml_command_result: %{result_type: result_type}})
      when elem(result_type, 0) == :operator_info do
    case elem(result_type, 1) do
      %{type: {:obj_ref, ref}} -> {:ok, String.split(ref.id, ",", trim: true)}
      _info -> {:error, protocol_error("a list of operators answered with no object reference")}
    end
  end

  def operators(execution), do: {:error, unexpected(execution, "a list of operators")}

  defp decode(literal) do
    case Literal.value(literal) do
      {:ok, %UDT{} = udt} -> Linalg.from_udt(udt)
      {:ok, value} -> {:ok, value}
      {:error, error} -> {:error, error}
    end
  end

  defp unexpected(%{ml_command_result: nil}, wanted) do
    protocol_error(
      "the server ran the command and answered with no ML result, where #{wanted} belongs"
    )
  end

  defp unexpected(%{ml_command_result: %{result_type: {arm, _}}}, wanted) do
    protocol_error("the server answered with #{arm} where #{wanted} belongs")
  end

  defp unexpected(_execution, wanted) do
    protocol_error("no #{wanted} in the server's answer")
  end

  defp protocol_error(message), do: %Error{kind: :protocol, message: message}
end
