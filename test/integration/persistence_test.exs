defmodule Latu.ML.PersistenceTest do
  use ExUnit.Case, async: true

  @moduletag :integration

  # ML4's claim against a live server: a model written to the driver's disk, read back into a
  # *fresh* cache reference, scored, and evaluated. What is deliberately not proved here is
  # that PySpark reads what this wrote — that needs both clients and one shared path, and it is
  # the interop dance in `dev/README.md`.
  #
  # Every path is on the **server**: the write happens where the session is, so these are the
  # driver container's `/tmp` and nothing on this machine has to exist.

  @url System.get_env("SPARK_REMOTE", "sc://localhost:15003")

  alias Latu.ML
  alias Latu.ML.Classification
  alias Latu.ML.Evaluation
  alias Latu.ML.Feature

  setup do
    session = Latu.connect!(@url)
    on_exit(fn -> Latu.disconnect(session, release: true) end)

    training =
      Latu.sql!(session, """
      SELECT * FROM VALUES (0.0, 0.0, 1.1), (0.0, 0.1, 1.0), (1.0, 2.0, 0.1), (1.0, 2.1, 0.2)
      AS t(label, x1, x2)
      """)

    assembler = Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)

    %{session: session, features: ML.transform(assembler, training), path: path()}
  end

  # Unique per test, because nothing cleans these up and two runs of the suite must not race
  # for one directory. The monotonic integer is enough: one server, one clock.
  defp path do
    "/tmp/latu_ml_test/#{System.system_time(:second)}-#{System.unique_integer([:positive])}"
  end

  defp predictions(frame) do
    {:ok, rows} = frame |> Latu.select([:label, :prediction]) |> Latu.collect()

    Enum.map(rows, & &1.prediction)
  end

  test "a fitted model survives a save and a load", context do
    %{session: session, features: features, path: path} = context
    lr = Classification.logistic_regression(max_iter: 5, reg_param: 0.01)

    assert {:ok, fitted} = ML.fit(lr, features)
    assert :ok = ML.save(fitted, path)

    assert {:ok, loaded} = ML.load(session, :logistic_regression_model, path)

    # A load caches a model of its own: a second entry, a different reference, and the caller
    # owns both.
    assert is_binary(loaded.ref)
    assert loaded.ref != fitted.ref
    assert loaded.class == "org.apache.spark.ml.classification.LogisticRegressionModel"

    # The uid is the saved one, which is the one the estimator generated — a Fit answers with
    # no uid at all, so this is the only command that gives one back.
    assert loaded.uid == lr.uid

    # And it scores the same rows the same way, which is the whole point of a saved model.
    assert predictions(ML.transform(loaded, features)) ==
             predictions(ML.transform(fitted, features))

    assert {:ok, coefficients} = ML.attribute(loaded, :coefficients)
    assert Nx.shape(coefficients) == {2}

    assert :ok = ML.delete([fitted, loaded])
  end

  test "the params come back with it", context do
    %{session: session, features: features, path: path} = context
    lr = Classification.logistic_regression(max_iter: 5, reg_param: 0.01, features_col: :features)

    assert {:ok, fitted} = ML.fit(lr, features)
    assert :ok = ML.save(fitted, path)
    assert {:ok, loaded} = ML.load(session, :logistic_regression_model, path)

    # A saved model carries every param it has a value for, not just the ones the caller set,
    # so this is a superset of the estimator's — and the values that were set are in it.
    values = Map.new(loaded.params, fn {wire, _type, value} -> {wire, value} end)

    assert values["maxIter"] == 5
    assert values["regParam"] == 0.01
    assert values["featuresCol"] == "features"

    assert :ok = ML.delete([fitted, loaded])
  end

  test "an unfitted estimator round-trips, and fits afterwards", context do
    %{session: session, features: features, path: path} = context
    lr = Classification.logistic_regression(max_iter: 5, reg_param: 0.01)

    assert :ok = ML.save(lr, path, session: session)
    assert {:ok, loaded} = ML.load(session, :logistic_regression, path)

    # Nothing was cached: the server answered with data and there is nothing to release.
    assert loaded.__struct__ == Latu.ML.Estimator
    assert loaded.class == lr.class
    assert loaded.uid == lr.uid
    assert loaded.model_class == "org.apache.spark.ml.classification.LogisticRegressionModel"

    assert {:ok, model} = ML.fit(loaded, features)
    assert :ok = ML.delete(model)
  end

  test "a write refuses a path that exists, until it is told to overwrite", context do
    %{features: features, path: path} = context
    lr = Classification.logistic_regression(max_iter: 1)

    assert {:ok, model} = ML.fit(lr, features)
    assert :ok = ML.save(model, path)

    assert {:error, error} = ML.save(model, path)
    assert error.message =~ path

    assert :ok = ML.save(model, path, overwrite: true)
    assert :ok = ML.delete(model)
  end

  test "a loaded model has no summary, and neither side pretends otherwise", context do
    %{session: session, features: features, path: path} = context
    lr = Classification.logistic_regression(max_iter: 5)

    assert {:ok, fitted} = ML.fit(lr, features)
    assert :ok = ML.save(fitted, path)
    assert {:ok, loaded} = ML.load(session, :logistic_regression_model, path)

    # The fitted model has one.
    assert {:ok, iterations} = ML.attribute(ML.summary(fitted), :totalIterations)
    assert is_integer(iterations)

    # The loaded one does not: a model's save format does not carry its summary. There is no
    # training frame to rebuild from either, so `attribute/2` reports rather than retrying —
    # a retry with no dataset is what this test exists to keep out.
    assert {:error, error} = ML.attribute(ML.summary(loaded), :totalIterations)
    assert ML.error_kind(error) == :summary_lost
    assert ML.hint(error) =~ "Latu.ML.load/3"

    assert :ok = ML.delete([fitted, loaded])
  end

  test "an evaluator scores a frame", context do
    %{features: features} = context
    lr = Classification.logistic_regression(max_iter: 5, reg_param: 0.01)

    assert {:ok, model} = ML.fit(lr, features)
    scored = ML.transform(model, features)

    assert {:ok, auc} = ML.evaluate(Evaluation.binary_classification_evaluator(), scored)
    assert is_float(auc)
    assert auc >= 0.0 and auc <= 1.0

    # The metric is the evaluator's own param, so a second metric is a second evaluator and
    # not a second verb.
    pr = Evaluation.binary_classification_evaluator(metric_name: "areaUnderPR")

    assert {:ok, area} = ML.evaluate(pr, scored)
    assert is_float(area)

    assert :ok = ML.delete(model)
  end
end
