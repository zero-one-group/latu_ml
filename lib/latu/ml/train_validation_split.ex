defmodule Latu.ML.TrainValidationSplit do
  @moduledoc """
  One split instead of k folds: fit every param map once, and keep the best.

  `Latu.ML.CrossValidator` with `num_folds: 1` is not this — a cross-validation scores every
  param map k times and averages. This scores each one exactly once, against a single held-out
  slice, and is what you reach for when a fit is expensive enough that k of them is not worth
  the variance reduction.

      tvs =
        ML.train_validation_split(
          estimator: lr,
          param_maps: ML.param_grid(lr, reg_param: [0.1, 0.01]),
          evaluator: Evaluation.binary_classification_evaluator(),
          train_ratio: 0.75
        )

      {:ok, model} = ML.fit(tvs, training)

  The split is cut the same way a fold is — `rand(seed)` and a range — so `train_ratio: 0.75`
  means rows whose draw is under 0.75, not exactly three quarters of them. Pass `seed:` for a
  split that is the same twice.
  """

  alias Latu.ML.CrossValidator
  alias Latu.ML.Evaluator

  @enforce_keys [:uid, :estimator, :param_maps, :evaluator]
  defstruct [
    :uid,
    :estimator,
    :param_maps,
    :evaluator,
    :seed,
    train_ratio: 0.75,
    collect_sub_models: false
  ]

  @type t :: %__MODULE__{
          uid: String.t(),
          estimator: CrossValidator.searchable(),
          param_maps: [Latu.ML.param_map()],
          evaluator: Evaluator.t(),
          seed: integer() | nil,
          train_ratio: float(),
          collect_sub_models: boolean()
        }
end
