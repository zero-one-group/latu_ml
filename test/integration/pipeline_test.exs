defmodule Latu.ML.PipelineTest do
  use ExUnit.Case, async: true

  @moduletag :integration

  # ML5a against a live server. A pipeline is the first thing in this package with **no
  # server-side object at all**: no `Fit`, no cache entry of its own, no `Read`. The fold is
  # client-side and so is the directory it saves into, which is why the interop dance in
  # `dev/README.md` matters more here than anywhere else — nothing but PySpark reading what this
  # wrote can say the format is right.

  @url System.get_env("SPARK_REMOTE", "sc://localhost:15003")

  alias Latu.ML
  alias Latu.ML.Classification
  alias Latu.ML.Feature
  alias Latu.ML.PipelineModel

  setup do
    session = Latu.connect!(@url)
    on_exit(fn -> Latu.disconnect(session, release: true) end)

    training =
      Latu.sql!(session, """
      SELECT * FROM VALUES (0.0, 0.0, 1.1), (0.0, 0.1, 1.0), (1.0, 2.0, 0.1), (1.0, 2.1, 0.2)
      AS t(label, x1, x2)
      """)

    %{session: session, training: training, path: path()}
  end

  defp path do
    "/tmp/latu_ml_test/#{System.system_time(:second)}-#{System.unique_integer([:positive])}"
  end

  defp pipeline do
    ML.pipeline([
      Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features),
      Feature.standard_scaler(input_col: :features, output_col: :scaled),
      Classification.logistic_regression(features_col: :scaled, max_iter: 5)
    ])
  end

  defp predictions(frame) do
    {:ok, rows} = frame |> Latu.select([:label, :prediction]) |> Latu.collect()

    Enum.map(rows, & &1.prediction)
  end

  test "a fit does one action per estimator, and the model owns them", context do
    %{session: session, training: training} = context

    {:ok, before} = ML.cache_info(session)
    assert {:ok, model} = ML.fit(pipeline(), training)

    # Two estimators in that pipeline, so two cache entries — and the transformer in front of
    # them is carried, not cached.
    {:ok, during} = ML.cache_info(session)
    assert length(during) - length(before) == 2

    assert %PipelineModel{} = model
    assert length(model.stages) == 3
    assert [%Latu.ML.Transformer{}, %Latu.ML.Model{}, %Latu.ML.Model{}] = model.stages

    scored = ML.transform(model, training)
    assert {:ok, rows} = scored |> Latu.select([:label, :prediction]) |> Latu.collect()
    assert length(rows) == 4

    # One delete for the whole thing.
    assert :ok = ML.delete(model)
    {:ok, after_delete} = ML.cache_info(session)
    assert length(after_delete) == length(before)
  end

  test "the fold stops transforming after the last estimator", context do
    %{training: training} = context

    # A transformer *after* the last estimator is carried but never applied during the fit —
    # PySpark's rule, and the one thing about the fold that surprises. It still applies when the
    # pipeline model transforms.
    trailing = Feature.binarizer(input_col: :prediction, output_col: :flag, threshold: 0.5)

    stages =
      ML.pipeline([
        Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features),
        Classification.logistic_regression(max_iter: 5),
        trailing
      ])

    assert {:ok, model} = ML.fit(stages, training)
    assert List.last(model.stages) == trailing

    scored = ML.transform(model, training)
    assert {:ok, [row | _]} = scored |> Latu.select([:flag]) |> Latu.collect()
    assert is_float(row.flag)

    assert :ok = ML.delete(model)
  end

  test "save and load, and the loaded one scores the same rows", context do
    %{session: session, training: training, path: path} = context

    assert {:ok, fitted} = ML.fit(pipeline(), training)
    assert :ok = ML.save(fitted, path)

    assert {:ok, loaded} = ML.load(session, :pipeline_model, path)

    assert %PipelineModel{} = loaded
    assert loaded.uid == fitted.uid
    assert length(loaded.stages) == 3

    # Every stage came back as what it was, in order.
    assert Enum.map(loaded.stages, & &1.__struct__) ==
             [Latu.ML.Transformer, Latu.ML.Model, Latu.ML.Model]

    assert Enum.map(loaded.stages, & &1.uid) == Enum.map(fitted.stages, & &1.uid)

    assert predictions(ML.transform(loaded, training)) ==
             predictions(ML.transform(fitted, training))

    assert :ok = ML.delete(fitted)
    assert :ok = ML.delete(loaded)
  end

  test "an unfitted pipeline round-trips too, and fits afterwards", context do
    %{session: session, training: training, path: path} = context
    unfitted = pipeline()

    assert :ok = ML.save(unfitted, path, session: session)
    assert {:ok, loaded} = ML.load(session, :pipeline, path)

    assert loaded.uid == unfitted.uid
    assert Enum.map(loaded.stages, & &1.uid) == Enum.map(unfitted.stages, & &1.uid)

    assert {:ok, model} = ML.fit(loaded, training)
    assert :ok = ML.delete(model)
  end

  test "a second save needs overwrite, and clearing the directory is a helper call", context do
    %{training: training, path: path} = context

    assert {:ok, model} = ML.fit(pipeline(), training)
    assert :ok = ML.save(model, path)

    # Without it the stage writes refuse: the directory is already there.
    assert {:error, _error} = ML.save(model, path)

    # With it, `handleOverwrite` clears the whole directory before anything is written — which
    # is what `MlCommand.Write`'s own flag cannot do for a tree.
    assert :ok = ML.save(model, path, overwrite: true)
    assert :ok = ML.delete(model)
  end

  test "a nested pipeline is a stage like any other", context do
    %{session: session, training: training, path: path} = context

    inner = ML.pipeline([Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)])
    outer = ML.pipeline([inner, Classification.logistic_regression(max_iter: 5)])

    assert {:ok, model} = ML.fit(outer, training)
    assert [%PipelineModel{}, %Latu.ML.Model{}] = model.stages

    assert :ok = ML.save(model, path)
    assert {:ok, loaded} = ML.load(session, :pipeline_model, path)
    assert [%PipelineModel{}, %Latu.ML.Model{}] = loaded.stages

    assert predictions(ML.transform(loaded, training)) ==
             predictions(ML.transform(model, training))

    assert :ok = ML.delete(model)
    assert :ok = ML.delete(loaded)
  end

  test "with_model brackets a whole pipeline", context do
    %{session: session, training: training} = context
    {:ok, before} = ML.cache_info(session)

    assert {:ok, _rows} =
             ML.with_model(pipeline(), training, fn model ->
               model |> ML.transform(training) |> Latu.select([:label]) |> Latu.collect()
             end)

    {:ok, after_bracket} = ML.cache_info(session)
    assert length(after_bracket) == length(before)
  end
end
