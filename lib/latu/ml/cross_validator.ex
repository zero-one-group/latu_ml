defmodule Latu.ML.CrossValidator do
  @moduledoc """
  K-fold cross-validation: fit every param map on every fold, and keep the best.

  **Not a server operator.** Spark has no `Fit` for a validator — `MlCommand.Fit` takes an
  `MlOperator` and the Connect server exposes none for `CrossValidator` — so the search is a
  fold this package runs, as it is a loop PySpark runs. What reaches the server is one `Fit`,
  one `Transform` and one `Evaluate` per fold per param map, and one last `Fit` for the winner.

      grid = ML.param_grid(lr, reg_param: [0.1, 0.01], max_iter: [10, 100])

      cv =
        ML.cross_validator(
          estimator: lr,
          param_maps: grid,
          evaluator: Evaluation.binary_classification_evaluator(),
          num_folds: 3
        )

      {:ok, model} = ML.fit(cv, training)

  ## What it costs

  `num_folds * length(param_maps)` fits, plus one. Each is a cache entry on the server while it
  is being scored, and `Latu.ML.fit/2` deletes it as soon as its metric is read — so a 3-by-4
  search holds one model at a time rather than twelve. `collect_sub_models: true` keeps them
  all instead, and they become yours to `Latu.ML.delete/1`.

  ## The folds

  A fold is cut by ranking rows on `rand(seed)` and taking a range of it, which is Spark's own
  way and **not** a `random_split`. Pass `seed:` for a cut that is the same twice. `fold_col:`
  names a column that already holds a fold number instead, in which case `seed` and `num_folds`
  only have to agree with it.
  """

  alias Latu.ML.Estimator
  alias Latu.ML.Evaluator
  alias Latu.ML.Pipeline

  @typedoc "What is being searched: one estimator, or a pipeline with one inside it."
  @type searchable :: Estimator.t() | Pipeline.t()

  @enforce_keys [:uid, :estimator, :param_maps, :evaluator]
  defstruct [
    :uid,
    :estimator,
    :param_maps,
    :evaluator,
    :seed,
    :fold_col,
    num_folds: 3,
    collect_sub_models: false
  ]

  @type t :: %__MODULE__{
          uid: String.t(),
          estimator: searchable(),
          param_maps: [Latu.ML.param_map()],
          evaluator: Evaluator.t(),
          seed: integer() | nil,
          fold_col: String.t() | nil,
          num_folds: pos_integer(),
          collect_sub_models: boolean()
        }
end
