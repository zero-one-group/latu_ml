defmodule Latu.ML.Evaluator do
  @moduledoc """
  A metric, as inert data: what to measure and on which columns.

  The third kind of operator beside `Latu.ML.Estimator` and `Latu.ML.Transformer`, and the
  simplest — it is never fitted and never transforms, it just scores a frame. Spark has six of
  them, all under `Latu.ML.Evaluation`.

      evaluator = Latu.ML.Evaluation.binary_classification_evaluator(metric_name: "areaUnderROC")

  `Latu.ML.evaluate/2` is what runs one, and it answers with the metric itself rather than with
  a handle: an evaluator caches nothing on the server. The frame it scores is one a model has
  already transformed — an evaluator reads `prediction_col` and `label_col`, not features.
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
  An evaluator the registry does not know, by its JVM class name.

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
