defmodule Latu.ML.Plan do
  @moduledoc """
  ML commands and relations, built as protos.

  The layer below `Latu.ML`, and the twin of `Latu.Plan`: it takes values, hands back a
  `Plan` or a `Relation`, does no IO and touches no session.

  Spark models ML as two shapes over one set of messages. A `Fit`, a `Delete`, an attribute
  whose answer is a literal — those are **commands**, run by `Latu.Client.execute_command/3`.
  A `Transform`, and an attribute whose answer is a DataFrame — those are **relations**, lazy
  like any other, and nothing reaches the server until an action.

  `Fetch` is one message shared by both: `fetch_command/2` and `fetch_relation/3` build the
  same `Fetch` and differ only in the wrapper.

  Every `Relation` here comes from `Latu.Plan.relation/1`, which is the one place a `plan_id`
  is assigned. A second allocator out of tree would be a second sequence.
  """

  alias Latu.Protocol.Spark.Connect, as: Proto

  @typedoc "A server-side object in the session's ML cache, as `Fit` handed it back."
  @type object_ref :: String.t()

  # These five are the protocol's own types under names this package can talk about. Latu's
  # `Latu.Plan` does the same, for the same reason: the generated modules are `@moduledoc
  # false`, so a spec naming one directly is a link ExDoc cannot resolve, and a caller reading
  # `command()` learns more than one reading `Proto.Plan.t()` anyway.

  @typedoc "A command plan, ready for `Latu.Client.execute_command/3`."
  @type command :: Proto.Plan.t()

  @typedoc "A relation, lazy until something acts on it."
  @type relation :: Proto.Relation.t()

  @typedoc "An ML operator: a JVM class, a uid, and what kind of thing it is."
  @type operator :: Proto.MlOperator.t()

  @typedoc "The params a caller set, keyed by Spark's own spelling."
  @type params :: Proto.MlParams.t()

  @typedoc "A single value on the wire."
  @type literal :: Proto.Expression.Literal.t()

  @typedoc "One link in a `Fetch` chain."
  @type method :: Proto.Fetch.Method.t()

  @typedoc "A writer's options, passed through to Spark's own `MLWriter`."
  @type options :: %{optional(String.t()) => String.t()}

  @typedoc "What an `MlOperator` is, in the server's own vocabulary."
  @type kind :: :estimator | :transformer | :evaluator | :model

  @typedoc "A scalar param type, and the only kind of element a list param may hold."
  @type scalar_type :: :int | :long | :float | :double | :boolean | :string

  @typedoc """
  A param's declared type, as the registry reports it.

  `:int` and `:long` build the same literal — see `literal/2`. A list holds scalars, or lists
  of them: 4.2.0 has one nested param (`Bucketizer.splitsArray`, a list of lists of doubles)
  and one nested helper argument (`stringIndexerModelFromLabelsArray`, a list of lists of
  strings). Deeper than that is refused rather than built blind.
  """
  @type param_type ::
          scalar_type()
          | :vector
          | :matrix
          | {:list, scalar_type() | {:list, scalar_type()}}
          | {:either, [scalar_type() | :vector | :matrix]}

  @typedoc "One param, ready for `params/1`: its wire name, its declared type, and the value."
  @type param :: {String.t(), param_type(), term()}

  # 32-bit bounds. Spark's own, and the reason an integer's wire type is a matter of magnitude.
  @int32 -2_147_483_648..2_147_483_647

  # The id the server resolves to its `ConnectHelper` singleton. Not a cache entry and not a
  # uuid: a `Fetch` naming it reaches an object that was never registered and cannot be
  # deleted. PySpark spells it the same way, in `ML_CONNECT_HELPER_ID`.
  @helper_object "______ML_CONNECT_HELPER______"

  @operator_types [
    estimator: :OPERATOR_TYPE_ESTIMATOR,
    transformer: :OPERATOR_TYPE_TRANSFORMER,
    evaluator: :OPERATOR_TYPE_EVALUATOR,
    model: :OPERATOR_TYPE_MODEL
  ]

  # =============================================
  # Operators and params
  # =============================================

  @doc """
  An `MlOperator`: the JVM class, a client-assigned uid, and what kind of thing it is.

  The class name is the operator's identity on the wire — there is no registry lookup on the
  server, only a `ServiceLoader` over what its classpath offers.
  """
  @spec operator(String.t(), String.t(), kind()) :: operator()
  def operator(class, uid, kind) when is_binary(class) and is_binary(uid) do
    %Proto.MlOperator{
      name: class,
      uid: uid,
      type: lookup(@operator_types, kind, "operator kind")
    }
  end

  @doc """
  Params as the server wants them: a map of wire name to literal, carrying **only what the
  caller set**.

  Defaults are never sent. PySpark serialises its `_paramMap`, which holds set values alone,
  and the server fills the rest from the operator's own defaults — so a param absent here and
  a param sent with its default value are different requests, and only the first is right.
  """
  @spec params([param()]) :: params()
  def params(entries) when is_list(entries) do
    %Proto.MlParams{
      params: Map.new(entries, fn {wire, type, value} -> {wire, literal(value, type)} end)
    }
  end

  @doc """
  Coerce a value to its param's declared type, then to a literal.

  Two stages, both PySpark's. The declared type decides the **kind** — a double param takes
  `1` and sends `1.0`, because `TypeConverters.toFloat` ran first. Magnitude then decides the
  **wire type** for integers, exactly as `Latu.Plan.lit/1` does: `:int` and `:long` are the
  same path, and a value past 32 bits becomes a `long` whichever one declared it.

  A refusal names the declared type, never the range: `reg_param: :nope` is wrong here,
  `reg_param: -1.0` is wrong on the server, and Spark's `ParamValidators` says so better.
  """
  @spec literal(term(), param_type()) :: literal()
  def literal(value, type)

  def literal(values, {:list, element}) when is_list(values) do
    # Only the deprecated `element_type`, and no outer `data_type` — which is what PySpark
    # writes and the server reads, the same asymmetry as the UDT `struct_type` in
    # `Latu.ML.Linalg`. Setting `data_type` as well is accepted by the server but is not what
    # the oracle produces, and byte-identical is the whole point of the goldens.
    %Proto.Expression.Literal{
      literal_type:
        {:array,
         %Proto.Expression.Literal.Array{
           element_type: data_type(element),
           # Fixed by the declared element type, not by each value's magnitude: one array
           # literal carries one element type, so the elements cannot disagree.
           elements: Enum.map(values, &element(&1, element))
         }}
    }
  end

  def literal(value, {:list, _element}) do
    raise ArgumentError, "a list param takes a list, not #{inspect(value)}"
  end

  def literal(value, type) when type in [:vector, :matrix] do
    Latu.ML.Linalg.to_literal(value)
  end

  # One attribute in 4.2.0 declares a union — `Word2VecModel.findSynonyms(word)` takes a term
  # or the vector to look near. The value's own shape picks the member, in the order PySpark
  # declared them; a value no member accepts is refused naming all of them.
  def literal(value, {:either, members}) when is_list(members) do
    case Enum.find(members, &accepts?(value, &1)) do
      nil ->
        raise ArgumentError,
              "#{inspect(value)} is not a valid value for any of #{inspect(members)}"

      member ->
        literal(value, member)
    end
  end

  def literal(value, type) do
    %Proto.Expression.Literal{literal_type: scalar(value, type)}
  end

  # One element of an array literal. A nested list is an array literal of its own — the only
  # place a `literal/2` arm recurses — and everything else is a scalar whose wire type is the
  # **declared** one rather than its magnitude: one array literal carries one element type, so
  # two elements either side of 32 bits cannot disagree.
  defp element(values, {:list, _inner} = nested), do: literal(values, nested)

  defp element(value, type) do
    %Proto.Expression.Literal{literal_type: fixed(value, type)}
  end

  defp fixed(value, :int), do: {:integer, whole(value, :int)}
  defp fixed(value, :long), do: {:long, whole(value, :long)}
  defp fixed(value, type), do: scalar(value, type)

  # The scalar arms. Magnitude picks, as PySpark's `_from_value` does.
  defp scalar(value, type) when type in [:int, :long], do: integer(whole(value, type), type)
  defp scalar(value, type) when type in [:float, :double], do: {:double, double(value, type)}
  defp scalar(value, :boolean) when is_boolean(value), do: {:boolean, value}
  defp scalar(value, :string) when is_binary(value), do: {:string, value}

  defp scalar(value, :string) when is_atom(value) and not is_nil(value) do
    {:string, to_string(value)}
  end

  defp scalar(value, type), do: wrong_kind(value, type)

  defp integer(value, _type) when value in @int32, do: {:integer, value}
  defp integer(value, _type), do: {:long, value}

  defp whole(value, _type) when is_integer(value), do: value

  # PySpark's `toInt` takes a float with nothing after the point and refuses the rest.
  defp whole(value, type) when is_float(value) do
    if value == Float.round(value), do: trunc(value), else: wrong_kind(value, type)
  end

  defp whole(value, type), do: wrong_kind(value, type)

  defp double(value, _type) when is_float(value), do: value
  defp double(value, _type) when is_integer(value), do: value * 1.0
  defp double(value, type), do: wrong_kind(value, type)

  defp wrong_kind(value, type) do
    raise ArgumentError, "#{inspect(value)} is not a valid #{inspect(type)} value"
  end

  # Which member of an `{:either, …}` a value belongs to. Shape, not coercion: the point is to
  # choose the arm, and the arm's own clause then refuses anything it cannot build. Kept
  # deliberately loose — `1` is a fine `:double` — so that choosing never rejects a value the
  # chosen member would have accepted.
  defp accepts?(value, :string), do: is_binary(value) or (is_atom(value) and not is_nil(value))
  defp accepts?(value, :boolean), do: is_boolean(value)
  defp accepts?(value, type) when type in [:int, :long], do: is_number(value)
  defp accepts?(value, type) when type in [:float, :double], do: is_number(value)
  defp accepts?(value, {:list, _element}), do: is_list(value)

  defp accepts?(value, type) when type in [:vector, :matrix] do
    is_struct(value, Latu.ML.SparseVector) or is_struct(value, Nx.Tensor)
  end

  defp accepts?(_value, _type), do: false

  # Spark's default collation, which PySpark's `StringType()` carries and the wire shows.
  defp data_type(:string) do
    %Proto.DataType{kind: {:string, %Proto.DataType.String{collation: "UTF8_BINARY"}}}
  end

  defp data_type(:boolean), do: %Proto.DataType{kind: {:boolean, %Proto.DataType.Boolean{}}}
  defp data_type(:int), do: %Proto.DataType{kind: {:integer, %Proto.DataType.Integer{}}}
  defp data_type(:long), do: %Proto.DataType{kind: {:long, %Proto.DataType.Long{}}}

  defp data_type(type) when type in [:float, :double] do
    %Proto.DataType{kind: {:double, %Proto.DataType.Double{}}}
  end

  # `ArrayType(StringType())`, PySpark's `containsNull=True` default and all.
  defp data_type({:list, element}) do
    %Proto.DataType{
      kind: {:array, %Proto.DataType.Array{element_type: data_type(element), contains_null: true}}
    }
  end

  defp data_type(type) do
    raise ArgumentError,
          "no array element type for #{inspect(type)}, expected a scalar or a list of them"
  end

  # =============================================
  # Commands
  # =============================================

  @doc "Fit an estimator on a dataset. The result is a handle, not a model."
  @spec fit(operator(), params(), relation()) :: command()
  def fit(
        %Proto.MlOperator{} = estimator,
        %Proto.MlParams{} = params,
        %Proto.Relation{} = data
      ) do
    command({:fit, %Proto.MlCommand.Fit{estimator: estimator, params: params, dataset: data}})
  end

  @doc """
  Score a frame with an evaluator. One number comes back.

  The twin of `fit/3`, and the same three fields: `MlCommand.Evaluate` differs from
  `MlCommand.Fit` only in naming its operator `evaluator` and in what the result carries.
  A metric comes back on the `param` arm, where a `Fit` answers with a handle.
  """
  @spec evaluate(operator(), params(), relation()) :: command()
  def evaluate(
        %Proto.MlOperator{} = evaluator,
        %Proto.MlParams{} = params,
        %Proto.Relation{} = data
      ) do
    command(
      {:evaluate, %Proto.MlCommand.Evaluate{evaluator: evaluator, params: params, dataset: data}}
    )
  end

  @doc """
  Write an operator or a fitted model to a path, in Spark's own on-disk format.

  One command with two arms, and which one is used says where the thing being saved lives. A
  **model** is named by its cache reference: the server already holds it, and the params here
  are laid over a copy of it before it is written. An **estimator, transformer or evaluator**
  is named by its class and uid instead, because the server holds nothing for one — it builds
  the instance from this message and writes that.

  `should_overwrite` is set on every write, `false` included. PySpark sets it unconditionally
  and the field carries explicit presence, so leaving it unset where the answer is `false`
  would be a different message from the one the goldens compare against.

  `options` are the writer's, passed through to Spark's own `MLWriter` — what a format understands
  is the format's business, and nothing here checks them.
  """
  @spec write(object_ref() | operator(), params(), String.t(), boolean(), options()) :: command()
  def write(target, params, path, overwrite, options)

  def write(ref, %Proto.MlParams{} = params, path, overwrite, options) when is_binary(ref) do
    write_arm({:obj_ref, object_ref(ref)}, params, path, overwrite, options)
  end

  def write(
        %Proto.MlOperator{} = operator,
        %Proto.MlParams{} = params,
        path,
        overwrite,
        options
      ) do
    write_arm({:operator, operator}, params, path, overwrite, options)
  end

  @doc """
  Load an operator or a model from a path.

  The `MlOperator` here names the **class to read**, not an instance: there is no uid to send,
  because the uid is in the saved metadata and comes back in the answer. `OPERATOR_TYPE_MODEL`
  is what puts the result in the session's ML cache; the other three kinds are answered as
  data, with nothing cached and nothing to release.
  """
  @spec read(operator(), String.t()) :: command()
  def read(%Proto.MlOperator{} = operator, path) when is_binary(path) do
    command({:read, %Proto.MlCommand.Read{operator: operator, path: path}})
  end

  @doc """
  Call a method on the server's own helper object.

  `ConnectHelper` is in the same allowlist a model's attributes are in, and is reached by the
  same `Fetch` — but it is an attribute of nothing. There is no model, no `Fit` behind it and
  nothing cached: the object is named by the literal id below, which the server resolves to a
  singleton rather than to a cache entry.

  Its arguments are positional and every one is sent, where a `Fit`'s params are a map of only
  what the caller set. Three of the eleven answer with a **model reference** all the same —
  they build one server-side out of what you pass — and five answer with a DataFrame, which is
  `helper_relation/2`.
  """
  @spec helper(String.t() | atom(), [literal() | relation()]) :: command()
  def helper(method, args \\ []) do
    command({:fetch, fetch(@helper_object, [method(method, args)])})
  end

  @doc """
  See `helper/2`. The five whose answer is a DataFrame, so a relation rather than a command.
  """
  @spec helper_relation(String.t() | atom(), [literal() | relation()]) :: relation()
  def helper_relation(method, args \\ []) do
    ml_relation({:fetch, fetch(@helper_object, [method(method, args)])}, nil)
  end

  @doc """
  Read an attribute whose answer is a literal — a coefficient, an intercept, a vector.

  `methods` is a chain: `model.summary.weightedPrecision` is one `Fetch` with two of them.
  """
  @spec fetch_command(object_ref(), [method()]) :: command()
  def fetch_command(ref, methods) when is_binary(ref) and is_list(methods) do
    command({:fetch, fetch(ref, methods)})
  end

  @doc """
  Rebuild a summary the cache has dropped, from the frame its model was fitted on.

  The recovery half of the summary path. A summary is not offloaded like a model — it is
  dropped, and the next `Fetch` answers `CONNECT_ML.MODEL_SUMMARY_LOST` — so a client that
  wants one to survive a quiet session has to be able to make it again. `dataset` is not
  always the frame the caller passed to `Latu.ML.fit/2`: three of the ten models with a summary
  want
  the model's own transform of it instead, and `priv/ml_attributes.exs` says which.
  """
  @spec create_summary(object_ref(), relation()) :: command()
  def create_summary(ref, %Proto.Relation{} = dataset) when is_binary(ref) do
    command(
      {:create_summary,
       %Proto.MlCommand.CreateSummary{model_ref: object_ref(ref), dataset: dataset}}
    )
  end

  @doc "Release cached objects. The server answers with the refs it dropped, not silence."
  @spec delete([object_ref()]) :: command()
  def delete(refs) when is_list(refs) do
    command({:delete, %Proto.MlCommand.Delete{obj_refs: Enum.map(refs, &object_ref/1)}})
  end

  @doc """
  What the server thinks one model weighs, in bytes.

  Spark's own `Model.estimatedSize`, and the number the cache's budgets are spent against —
  `maxModelSize` refuses a fit above it, `maxStorageSize` refuses one that would take the
  session past it. So this is the one measurement that explains a refusal, and it is a
  *server* estimate: nothing client-side can compute it.
  """
  @spec get_model_size(object_ref()) :: command()
  def get_model_size(ref) when is_binary(ref) do
    command({:get_model_size, %Proto.MlCommand.GetModelSize{model_ref: object_ref(ref)}})
  end

  @doc """
  Everything the session's ML cache is holding, one JSON string per object.

  Takes no arguments — the cache is the session's, and the session is what carries the
  command. `MLCache.getInfo` renders `id`, `class` and `size` per entry.
  """
  @spec get_cache_info() :: command()
  def get_cache_info do
    command({:get_cache_info, %Proto.MlCommand.GetCacheInfo{}})
  end

  @doc """
  Drop everything the session's ML cache holds, and answer with how many objects that was.

  The blunt instrument: `MLCache.clear` empties the map and the offload directory both, so
  every model and summary in the session is gone at once and every `%Latu.ML.Model{}` a
  caller still holds is now a reference to nothing.
  """
  @spec clean_cache() :: command()
  def clean_cache do
    command({:clean_cache, %Proto.MlCommand.CleanCache{}})
  end

  # =============================================
  # Relations
  # =============================================

  @doc """
  Apply a transformer or a fitted model to a frame.

  Pure plan building: the relation carries the operator and the input, and `Latu.schema/1`
  answers from it without executing. Takes an `object_ref` for a model, or an `MlOperator`
  for a transformer that needs no fitting.
  """
  @spec transform(object_ref() | operator(), relation(), params()) :: relation()
  def transform(operator, input, params)

  def transform(ref, %Proto.Relation{} = input, %Proto.MlParams{} = params) when is_binary(ref) do
    transform_arm({:obj_ref, object_ref(ref)}, input, params)
  end

  def transform(
        %Proto.MlOperator{} = operator,
        %Proto.Relation{} = input,
        %Proto.MlParams{} = params
      ) do
    transform_arm({:transformer, operator}, input, params)
  end

  @doc """
  Read an attribute whose answer is a DataFrame — `summary.roc` and its kind.

  `dataset` is the training frame, carried so the server can rebuild a summary it has evicted;
  it rides `MlRelation`'s own field, beside the `Fetch` rather than inside it.
  """
  @spec fetch_relation(object_ref(), [method()], relation() | nil) :: relation()
  def fetch_relation(ref, methods, dataset \\ nil) when is_binary(ref) and is_list(methods) do
    ml_relation({:fetch, fetch(ref, methods)}, dataset)
  end

  @doc "One link in a `Fetch` chain. Arguments are literals or relations."
  @spec method(String.t() | atom(), [literal() | relation()]) :: method()
  def method(name, args \\ []) when is_list(args) do
    %Proto.Fetch.Method{method: name(name, "method"), args: Enum.map(args, &argument/1)}
  end

  # =============================================
  # Wrappers
  # =============================================

  defp command(arm) do
    ml = %Proto.MlCommand{command: arm}

    Latu.Plan.new(%Proto.Command{command_type: {:ml_command, ml}})
  end

  defp write_arm(type, %Proto.MlParams{} = params, path, overwrite, options)
       when is_binary(path) and is_boolean(overwrite) and is_map(options) do
    command(
      {:write,
       %Proto.MlCommand.Write{
         type: type,
         params: params,
         path: path,
         should_overwrite: overwrite,
         options: options
       }}
    )
  end

  defp transform_arm(operator, input, params) do
    ml_relation(
      {:transform, %Proto.MlRelation.Transform{operator: operator, input: input, params: params}},
      nil
    )
  end

  defp ml_relation(ml_type, dataset) do
    Latu.Plan.relation(
      {:ml_relation, %Proto.MlRelation{ml_type: ml_type, model_summary_dataset: dataset}}
    )
  end

  defp fetch(ref, methods) do
    %Proto.Fetch{obj_ref: object_ref(ref), methods: methods}
  end

  defp object_ref(id) when is_binary(id), do: %Proto.ObjectRef{id: id}

  defp argument(%Proto.Expression.Literal{} = literal) do
    %Proto.Fetch.Method.Args{args_type: {:param, literal}}
  end

  defp argument(%Proto.Relation{} = relation) do
    %Proto.Fetch.Method.Args{args_type: {:input, relation}}
  end

  defp argument(other) do
    raise ArgumentError,
          "a fetch argument is a literal or a relation, not #{inspect(other)}"
  end

  defp name(value, _what) when is_binary(value), do: value
  defp name(value, _what) when is_atom(value) and not is_nil(value), do: to_string(value)

  defp name(value, what) do
    raise ArgumentError, "a #{what} is a string or an atom, not #{inspect(value)}"
  end

  # Spark's enums as atoms, with a message that lists the spellings this package accepts.
  defp lookup(table, key, what) do
    Keyword.get(table, key) ||
      raise ArgumentError,
            "unknown #{what} #{inspect(key)}, expected one of #{inspect(Keyword.keys(table))}"
  end
end
