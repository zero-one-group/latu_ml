defmodule Latu.ML.Pipeline do
  @moduledoc """
  A sequence of stages, fitted as one.

  **Not a server operator.** Spark has no `Fit` for a pipeline: PySpark loops over the stages in
  Python and so does this package, which is why `Latu.ML.pipeline/1` needs no session and why a
  pipeline has no cache entry of its own. What `Latu.ML.fit/2` hands back is a
  `Latu.ML.PipelineModel` holding one entry per estimator it fitted.

      assembler = Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)
      lr = Classification.logistic_regression(max_iter: 10)

      {:ok, model} = ML.fit(ML.pipeline([assembler, lr]), training)
      scored = ML.transform(model, test)
      :ok = ML.delete(model)

  A stage is an estimator, a transformer, a model already fitted, or another pipeline. The fold
  is PySpark's exactly, and its one surprise is worth knowing: **stages after the last estimator
  are carried but never applied during the fit**. There is nothing left to learn from them, so
  the frame stops being transformed at that point.
  """

  alias Latu.ML.Estimator
  alias Latu.ML.Model
  alias Latu.ML.PipelineModel
  alias Latu.ML.Transformer

  @typedoc "Anything a pipeline may hold. A `Latu.ML.PipelineModel` is a transformer too."
  @type stage :: Estimator.t() | Transformer.t() | Model.t() | t() | PipelineModel.t()

  @enforce_keys [:uid]
  defstruct [:uid, stages: []]

  @type t :: %__MODULE__{uid: String.t(), stages: [stage()]}
end
