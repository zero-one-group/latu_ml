defmodule Latu.ML.Transformer do
  @moduledoc """
  A transformer that needs no fitting — `VectorAssembler` and its kind.

  `Latu.ML.transform/2` applies one, and because a `Transform` is a relation and not a command,
  applying it reaches no server: the frame it hands back is lazy like any other.
  """

  alias Latu.ML.Internal
  alias Latu.ML.Param

  @enforce_keys [:class, :uid]
  defstruct [:class, :uid, params: [], known: []]

  @type t :: %__MODULE__{
          class: String.t(),
          uid: String.t(),
          params: [Latu.ML.Plan.param()],
          known: [Param.t()]
        }

  @doc """
  A transformer the registry does not know, by its JVM class name.

  See `Latu.ML.Estimator.new/2`; the same caveats apply.

  ## Options

    * `:params` — the params to send, already in wire form. Defaults to `[]`.
    * `:uid` — the uid to send. Defaults to a generated one in PySpark's shape.
  """
  @spec new(String.t(), keyword()) :: t()
  def new(class, opts \\ []) when is_binary(class) do
    opts = Keyword.validate!(opts, params: [], uid: nil)

    %__MODULE__{class: class, uid: opts[:uid] || Internal.uid(class), params: opts[:params]}
  end
end
