defmodule Latu.ML.PipelineModel do
  @moduledoc """
  A fitted pipeline: its stages, in order, with every estimator replaced by the model it fitted.

  **It owns those models.** Each one is an entry in the session's ML cache, so
  `Latu.ML.delete/1` on a pipeline model releases every model inside it — including the ones in
  a nested pipeline — and nothing else does. `Latu.ML.with_model/3` takes a pipeline and brackets
  the whole thing.

  `Latu.ML.transform/2` composes the stages lazily: the frame it hands back carries every stage's
  relation and reaches no server until you collect it. There is no server-side pipeline, so there
  is nothing to release but the models.

  `session` is where those models live, and it is what `Latu.ML.save/3` writes through.
  """

  alias Latu.ML.Pipeline

  @enforce_keys [:uid, :session]
  defstruct [:uid, :session, stages: []]

  @type t :: %__MODULE__{
          uid: String.t(),
          session: Latu.Session.t(),
          stages: [Pipeline.stage()]
        }
end
