defmodule Latu.ML.CrossValidatorModel do
  @moduledoc """
  What a cross-validation found: the winning model, and the score of every param map.

  `best_model` is **refit on the whole frame** with the winning param map — it is not one of
  the fold models, which each saw only a fraction of the rows. That is PySpark's behaviour and
  the reason a search costs one fit more than the grid suggests.

  `avg_metrics` and `std_metrics` are the mean and population standard deviation across folds,
  one per param map, in the order `param_maps` gives them. `Latu.ML.best_index/1` says which
  one won and `Latu.ML.larger_better?/1` says why.

  `sub_models` is `nil` unless the search was asked to `collect_sub_models: true`, in which
  case it is a list of `num_folds` lists, one model per param map. **Those are yours to
  delete**; `Latu.ML.delete/1` on this struct releases them along with `best_model`.
  """

  alias Latu.ML.CrossValidator
  alias Latu.ML.Model
  alias Latu.ML.PipelineModel

  @typedoc "A fitted stage: a bare model, or a pipeline model where a pipeline was searched."
  @type fitted :: Model.t() | PipelineModel.t()

  @enforce_keys [:uid, :session, :best_model, :avg_metrics, :validator]
  defstruct [
    :uid,
    :session,
    :best_model,
    :avg_metrics,
    :std_metrics,
    :sub_models,
    :validator
  ]

  @type t :: %__MODULE__{
          uid: String.t(),
          session: Latu.Session.t(),
          best_model: fitted(),
          avg_metrics: [float()],
          std_metrics: [float()] | nil,
          sub_models: [[fitted()]] | nil,
          validator: CrossValidator.t()
        }
end
