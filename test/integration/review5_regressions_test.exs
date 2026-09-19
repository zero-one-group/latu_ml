defmodule Latu.ML.Review5RegressionsTest do
  use ExUnit.Case, async: true

  @moduletag :integration

  # The 2026-09-19 fifth review: four P2s. Three are the round-four shape at sites the sweep
  # missed (a helper constructor, the shared traversal, a search result's assembly); one is
  # arithmetic, a standard deviation that overflowed where its answer was representable. Each
  # test reads the cache size back per session, so these run concurrently. `docs/decisions.md`,
  # 2026-09-19.

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

  defp fresh_path(tag) do
    "/tmp/latu_ml_test/review5-#{tag}-#{System.system_time(:second)}-" <>
      Integer.to_string(System.unique_integer([:positive]))
  end

  # Rewrite a saved directory's `metadata` in place, as a one-row text write, so a test can
  # spoil one key and nothing else without touching the server's filesystem.
  defp rewrite_metadata(session, path, fun) do
    metadata_path = Path.join(path, "metadata")
    frame = Latu.read(session, format: "text", path: metadata_path)
    {:ok, %{value: json}} = Latu.first(frame)
    rewritten = json |> JSON.decode!() |> fun.() |> JSON.encode!()
    {:ok, frame} = Latu.create_dataframe(session, [value: [rewritten]], schema: "value STRING")

    :ok =
      frame
      |> Latu.coalesce(1)
      |> Latu.write(format: "text", path: metadata_path, mode: :overwrite)
  end

  # F1 (P2): a helper constructor checks its options before the send, so a refused option leaves
  # no model behind that the caller has no handle to.
  test "a helper constructor refused an option acquires nothing", %{session: session} do
    before = cache_size(session)

    assert_raise ArgumentError, fn ->
      Feature.StringIndexerModel.from_labels(session, ["a", "b"],
        input_col: :category,
        output_col: :indexed,
        typo: true
      )
    end

    assert cache_size(session) == before
  end

  # F2 (P2): a saved pipeline whose stageUids holds a non-string is refused by name before any
  # stage is read, rather than reading the first and raising on the second's path.
  test "a saved pipeline with a null stage uid is refused, and nothing is read", %{session: s} do
    base =
      Latu.sql!(s, """
      SELECT CAST(id % 3 AS STRING) AS cat, CAST(id % 2 AS STRING) AS cat2 FROM range(40)
      """)

    pipeline =
      ML.pipeline([
        Feature.string_indexer(input_col: :cat, output_col: :cat_index),
        Feature.string_indexer(input_col: :cat2, output_col: :cat2_index)
      ])

    path = fresh_path("pipeline")
    assert {:ok, fitted} = ML.fit(pipeline, base)
    assert :ok = ML.save(fitted, path)
    assert :ok = ML.delete(fitted)
    baseline = cache_size(s)

    rewrite_metadata(s, path, fn metadata ->
      update_in(metadata, ["paramMap", "stageUids"], fn [first, _second] -> [first, nil] end)
    end)

    assert {:error, %Latu.Error{} = error} = ML.load(s, :pipeline_model, path)
    assert Exception.message(error) =~ "stageUids"
    assert cache_size(s) == baseline
  end

  # F3 (P2): a saved search missing the metrics its result is built from is refused by name
  # before anything is read, rather than reading the winner and sub-models and raising while
  # assembling them.
  test "a saved search without its metrics is refused, and nothing is read", %{session: s} do
    features = features(s)
    lr = Regression.linear_regression(features_col: :features)

    tvs =
      ML.train_validation_split(
        estimator: lr,
        param_maps: ML.param_grid(lr, max_iter: [2, 5]),
        evaluator: Evaluation.regression_evaluator(),
        collect_sub_models: true,
        seed: 1
      )

    path = fresh_path("search")
    assert {:ok, searched} = ML.fit(tvs, features)
    assert :ok = ML.save(searched, path, sub_models: true)
    assert :ok = ML.delete(searched)
    baseline = cache_size(s)

    rewrite_metadata(s, path, &Map.delete(&1, "validationMetrics"))

    assert {:error, %Latu.Error{} = error} =
             ML.load(s, :train_validation_split_model, path, sub_models: true)

    assert Exception.message(error) =~ "validationMetrics"
    assert cache_size(s) == baseline
  end

  # F4 (P2): finite fold metrics of a large magnitude have a finite mean and standard deviation,
  # where squaring a deviation used to overflow, raise after the refit, and strand every model.
  test "fold metrics around 1.0e160 aggregate, and the search owns its models", %{session: s} do
    frame =
      Latu.sql!(s, """
      SELECT
        CAST(id % 2 AS INT) AS fold,
        CAST(id % 3 AS STRING) AS cat,
        CAST(CASE WHEN id % 2 = 0 THEN 1.0e160 ELSE 2.0e160 END AS DOUBLE) AS label
      FROM range(40)
      """)

    before = cache_size(s)

    # A StringIndexer stands in for a model: its index is a double, so a regression evaluator
    # scores it, and MAE against these labels is the label itself. Fold 0's rows are all 1.0e160
    # and fold 1's all 2.0e160, so the two fold metrics are exactly those.
    indexer = Feature.string_indexer(input_col: :cat, output_col: :prediction)

    cv =
      ML.cross_validator(
        estimator: indexer,
        param_maps: ML.param_grid(indexer, handle_invalid: ["keep"]),
        evaluator: Evaluation.regression_evaluator(metric_name: "mae"),
        num_folds: 2,
        fold_col: :fold,
        collect_sub_models: true
      )

    assert {:ok, searched} = ML.fit(cv, frame)
    assert [average] = searched.avg_metrics
    assert [deviation] = searched.std_metrics
    assert_in_delta average / 1.5e160, 1.0, 1.0e-9
    assert_in_delta deviation / 5.0e159, 1.0, 1.0e-9

    assert :ok = ML.delete(searched)
    assert cache_size(s) == before
  end
end
