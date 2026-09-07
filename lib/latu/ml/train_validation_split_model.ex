defmodule Latu.ML.TrainValidationSplitModel do
  @moduledoc """
  What a train/validation split found: the winning model, and one metric per param map.

  `validation_metrics` is a single score per param map rather than a mean over folds, which is
  the only real difference from `Latu.ML.CrossValidatorModel` — there is nothing to average, so
  there is no standard deviation either.

  `best_model` is refit on the whole frame, and `sub_models` is `nil` unless the search was
  asked to `collect_sub_models: true`. `Latu.ML.delete/1` releases everything this holds.
  """

  alias Latu.ML.CrossValidatorModel
  alias Latu.ML.TrainValidationSplit

  @enforce_keys [:uid, :session, :best_model, :validation_metrics, :validator]
  defstruct [:uid, :session, :best_model, :validation_metrics, :sub_models, :validator]

  @type t :: %__MODULE__{
          uid: String.t(),
          session: Latu.Session.t(),
          best_model: CrossValidatorModel.fitted(),
          validation_metrics: [float()],
          sub_models: [CrossValidatorModel.fitted()] | nil,
          validator: TrainValidationSplit.t()
        }
end
