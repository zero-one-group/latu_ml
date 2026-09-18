defmodule Latu.ML.Review3RegressionsTest do
  use ExUnit.Case, async: true

  @moduletag :integration

  # The 2026-09-18 third review. Each test is one finding about search/tuning ownership under the
  # deterministic, eager cache release this package does in place of PySpark's GC-driven cleanup.
  # Cache size is read per session, so these run concurrently: each test connects its own. The
  # frames are 40 rows with a fixed split seed, so no fold is ever empty — an empty fold fails in
  # Spark ("Nothing has been added to this summarizer") long before ownership is exercised.

  @url System.get_env("SPARK_REMOTE", "sc://localhost:15003")

  alias Latu.ML
  alias Latu.ML.Evaluation
  alias Latu.ML.Feature
  alias Latu.ML.Regression

  setup do
    session = Latu.connect!(@url)
    on_exit(fn -> Latu.disconnect(session, release: true) end)
    %{session: session}
  end

  defp features(session) do
    base =
      Latu.sql!(session, """
      SELECT
        CAST(id AS DOUBLE) AS x1,
        CAST(id % 7 AS DOUBLE) AS x2,
        CAST(2 * id + (id % 5) AS DOUBLE) AS label
      FROM range(40)
      """)

    assembler = Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)
    ML.transform(assembler, base)
  end

  # A frame a model can still transform is proof its server object is alive; a released one fails
  # at the collect with CACHE_INVALID.
  defp works?(model, frame) do
    match?(
      {:ok, [_ | _]},
      model |> ML.transform(frame) |> Latu.select([:label]) |> Latu.collect()
    )
  end

  # F1 (P1): a search fits around a pre-fitted stage the caller supplied and must never release it.
  test "a search leaves a caller-supplied pre-fitted stage alone", %{session: session} do
    features = features(session)
    scaler = ML.fit!(Feature.standard_scaler(input_col: :features, output_col: :scaled), features)

    lr = Regression.linear_regression(features_col: :scaled)
    pipeline = ML.pipeline([scaler, lr])
    grid = ML.param_grid(lr, max_iter: [5])

    tvs =
      ML.train_validation_split(
        estimator: pipeline,
        param_maps: grid,
        evaluator: Evaluation.regression_evaluator(),
        seed: 1
      )

    assert {:ok, searched} = ML.fit(tvs, features)
    assert works?(searched.best_model, features)
    assert works?(scaler, features)

    # Deleting the fitted search frees its acquisitions but not the caller's stage.
    assert :ok = ML.delete(searched)
    assert works?(scaler, features)
    assert :ok = ML.delete(scaler)
  end

  # F3 (P2): an evaluator that raises rather than returns (an invalid param kind) must not strand
  # the candidate it had already fitted, and must re-raise rather than swallow.
  test "a raising evaluator during a search strands nothing", %{session: session} do
    features = features(session)
    {:ok, before} = ML.cache_info(session)

    lr = Regression.linear_regression(features_col: :features)
    grid = ML.param_grid(lr, max_iter: [5])

    tvs =
      ML.train_validation_split(
        estimator: lr,
        param_maps: grid,
        evaluator: Evaluation.regression_evaluator(metric_name: 123),
        seed: 1
      )

    assert catch_error(ML.fit(tvs, features))

    {:ok, after_raise} = ML.cache_info(session)
    assert length(after_raise) == length(before)
  end

  # F4 (P2): a legitimately non-finite fold metric (R^2 on a constant target) is a named error,
  # not an ArithmeticError, and its collected sub-models are released.
  test "a non-finite fold metric is a named error and leaks no sub-models", %{session: session} do
    base =
      Latu.sql!(session, """
      SELECT
        CAST(1.0 AS DOUBLE) AS label,
        CAST(id AS DOUBLE) AS x1,
        CAST(id % 7 AS DOUBLE) AS x2
      FROM range(20)
      """)

    features =
      ML.transform(Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features), base)

    {:ok, before} = ML.cache_info(session)

    lr = Regression.linear_regression(features_col: :features)
    grid = ML.param_grid(lr, max_iter: [5])

    cv =
      ML.cross_validator(
        estimator: lr,
        param_maps: grid,
        evaluator: Evaluation.regression_evaluator(metric_name: "r2"),
        num_folds: 2,
        collect_sub_models: true,
        seed: 1
      )

    assert {:error, error} = ML.fit(cv, features)
    assert Exception.message(error) =~ "not finite"

    {:ok, after_fit} = ML.cache_info(session)
    assert length(after_fit) == length(before)
  end

  # F5 (P2): a loaded search owns everything it read — the winner, any sub-models, and the
  # estimator's own stages — so delete clears the cache, and a failed load does too.
  test "a loaded search owns the estimator it read", %{session: session} do
    features = features(session)

    path =
      "/tmp/latu_ml_test/review3-#{System.system_time(:second)}-#{System.unique_integer([:positive])}"

    scaler = ML.fit!(Feature.standard_scaler(input_col: :features, output_col: :scaled), features)
    lr = Regression.linear_regression(features_col: :scaled)
    pipeline = ML.pipeline([scaler, lr])
    grid = ML.param_grid(lr, max_iter: [5])

    tvs =
      ML.train_validation_split(
        estimator: pipeline,
        param_maps: grid,
        evaluator: Evaluation.regression_evaluator(),
        collect_sub_models: true,
        seed: 1
      )

    assert {:ok, searched} = ML.fit(tvs, features)
    assert :ok = ML.save(searched, path, sub_models: false)
    assert :ok = ML.delete(searched)
    assert :ok = ML.delete(scaler)

    {:ok, baseline} = ML.cache_info(session)

    # A successful load, then delete, returns to the baseline: the estimator's stage is owned too.
    assert {:ok, loaded} = ML.load(session, :train_validation_split_model, path)
    assert :ok = ML.delete(loaded)
    {:ok, after_delete} = ML.cache_info(session)
    assert length(after_delete) == length(baseline)

    # A failed load (asking for sub-models a `sub_models: false` save never wrote) keeps nothing.
    assert {:error, _} = ML.load(session, :train_validation_split_model, path, sub_models: true)
    {:ok, after_failed} = ML.cache_info(session)
    assert length(after_failed) == length(baseline)
  end
end
