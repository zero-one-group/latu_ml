defmodule Latu.ML.Review4RegressionsTest do
  use ExUnit.Case, async: true

  @moduletag :integration

  # The 2026-09-19 fourth review: three P2s, one shape. A model enters the server cache at one
  # step and the cleanup that knows about it sat at the next step, or one level up. Each test
  # fails a step right after an acquisition and reads the cache size back, per session, so these
  # run concurrently. `docs/decisions.md`, 2026-09-19.

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

  defp cache_size(session) do
    {:ok, entries} = ML.cache_info(session)
    length(entries)
  end

  # F1 (P2): a raise while fitting a later candidate gives back the sub-models this fold has kept.
  # The control is the same search with a returned error, which already cleaned up.
  test "a raise while fitting a later candidate strands no kept sub-model", %{session: session} do
    features = features(session)
    before = cache_size(session)
    lr = Regression.linear_regression(features_col: :features)

    search = fn grid ->
      ML.train_validation_split(
        estimator: lr,
        param_maps: grid,
        evaluator: Evaluation.regression_evaluator(),
        collect_sub_models: true,
        seed: 1
      )
    end

    # Built outside the assertion: the grid accepts the string and encoding refuses it, after the
    # first candidate has fitted and been kept. A grid that refused it here would fail loudly.
    raising = ML.param_grid(lr, max_iter: [2, "bad"])
    assert_raise ArgumentError, fn -> ML.fit(search.(raising), features) end
    assert cache_size(session) == before

    # `-1` is Spark's own range error, returned rather than raised.
    returned = ML.param_grid(lr, max_iter: [2, -1])
    assert {:error, %Latu.Error{}} = ML.fit(search.(returned), features)
    assert cache_size(session) == before
  end

  # F2 (P2): a saved search is read estimator, evaluator, winner, sub-models, in that order, and
  # the estimator's own stages are cached from the first step. A refusal at any later step gives
  # them back. Review 3 covered the sub-models step; this spoils the winner, then the evaluator.
  test "a load that fails after the estimator keeps nothing", %{session: session} do
    features = features(session)

    path =
      "/tmp/latu_ml_test/review4-#{System.system_time(:second)}-#{System.unique_integer([:positive])}"

    scaler = ML.fit!(Feature.standard_scaler(input_col: :features, output_col: :scaled), features)
    lr = Regression.linear_regression(features_col: :scaled)

    tvs =
      ML.train_validation_split(
        estimator: ML.pipeline([scaler, lr]),
        param_maps: ML.param_grid(lr, max_iter: [5]),
        evaluator: Evaluation.regression_evaluator(),
        seed: 1
      )

    assert {:ok, searched} = ML.fit(tvs, features)
    assert :ok = ML.save(searched, path)
    assert :ok = ML.delete(searched)
    assert :ok = ML.delete(scaler)
    baseline = cache_size(session)

    for part <- ["bestModel", "evaluator"] do
      spoil(session, Path.join(path, part))
      assert {:error, %Latu.Error{}} = ML.load(session, :train_validation_split_model, path)
      assert cache_size(session) == baseline
    end
  end

  # Overwrite a part's metadata with JSON naming no class, so reading that part is refused before
  # anything in it is cached. A one-row text write, as `Layout.write_metadata/6` does it.
  defp spoil(session, part) do
    {:ok, frame} = Latu.create_dataframe(session, [value: ["{}"]], schema: "value STRING")

    :ok =
      frame
      |> Latu.coalesce(1)
      |> Latu.write(format: "text", path: Path.join(part, "metadata"), mode: :overwrite)
  end

  # F3 (P2): a nested pipeline's models exist before the parent has recorded them. A raise while
  # transforming its output for the next stage gives them back.
  test "a nested pipeline that raises after fitting strands nothing", %{session: session} do
    base =
      Latu.sql!(session, """
      SELECT
        CAST(id AS DOUBLE) AS x1,
        CAST(id % 3 AS STRING) AS cat,
        CAST(2 * id AS DOUBLE) AS label
      FROM range(40)
      """)

    before = cache_size(session)

    # The inner fit carries its trailing transformer unapplied, so it succeeds and caches a
    # StringIndexerModel. The outer pipeline then applies the inner model on the way to `lr`,
    # which is where `statement: 123` is refused at encoding. Built outside the assertion, so a
    # constructor that refused it would fail loudly.
    inner =
      ML.pipeline([
        Feature.string_indexer(input_col: :cat, output_col: :cat_index),
        Feature.sql_transformer(statement: 123)
      ])

    outer = ML.pipeline([inner, Regression.linear_regression(features_col: :features)])

    assert_raise ArgumentError, fn -> ML.fit(outer, base) end
    assert cache_size(session) == before
  end
end
