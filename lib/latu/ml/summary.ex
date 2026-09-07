defmodule Latu.ML.Summary do
  @moduledoc """
  A model's training summary: a second server-side object, and the one the cache really drops.

  A model is offloaded to disk under memory pressure and comes back; a summary is *dropped*,
  and the fetch that finds it gone answers `CONNECT_ML.MODEL_SUMMARY_LOST`. That is not a
  failure the caller caused and not one they can prevent, so this package recovers from it the
  way PySpark does — it sends `CreateSummary` with the frame the model was fitted on and asks
  again. `dataset` is that frame, and carrying it is the whole reason this struct exists rather
  than a bare reference.

  **The reference is the model's own.** Spark has no separate cache key for a summary: it is
  reached by asking the model for one, so a `Fetch` names the model and puts `summary` at the
  head of its method chain. PySpark spells that as the string `"<model_ref>.summary"` and
  splits it apart again on the way out; here the parts are just kept apart.

      summary = Latu.ML.summary(model)
      {:ok, auc} = Latu.ML.Classification.BinaryLogisticRegressionSummary.area_under_roc(summary)
      roc = Latu.ML.Classification.BinaryLogisticRegressionSummary.roc(summary)

  `class` is nil for `LogisticRegressionModel` and `RandomForestClassificationModel`, whose
  summary class depends on how many classes the *fitted* model found — `_summaryCls` branches
  on `numClasses`, and no client knows the answer before it asks. `candidates` is what PySpark
  can return, both of them, in its own order; `class` is set only where that list has one
  entry, so a class check is made where one can be and skipped where it cannot.

  The candidates are what an argument's type is looked up in, and the two never disagree:
  4.2.0's binary summary is a *subclass* of its multiclass one, so every attribute they share
  has the same signature and the binary one only adds. `Latu.ML.attributes/1` asks a narrower
  question — what is allowed whichever class this turns out to be — and so reports what they
  share; name `Latu.ML.Classification.BinaryLogisticRegressionTrainingSummary` to see the rest.
  """

  @enforce_keys [:ref, :session]
  defstruct [:ref, :session, :class, :candidates, :dataset, methods: ["summary"]]

  @type t :: %__MODULE__{
          ref: Latu.ML.Plan.object_ref(),
          session: Latu.Session.t(),
          class: String.t() | nil,
          candidates: [String.t()],
          dataset: term(),
          methods: [String.t()]
        }
end
