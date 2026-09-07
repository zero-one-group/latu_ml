# Runnable example. With a server up:
#
#     docker compose up -d spark-connect
#     mix run dev/example.exs
#
# Or, to keep the bindings and poke at them:
#
#     iex -S mix
#     iex> import_file "dev/example.exs"
#
# Latu's `dev/example.exs`, for the ML surface. Deliberately does not disconnect, so the REPL
# keeps a live `session` — and with it a live ML cache. Every model below is deleted when it is
# done with, which is what the bracket is for, and the last section empties whatever is left.

alias Latu.ML
alias Latu.ML.Classification.{BinaryLogisticRegressionSummary, LogisticRegressionModel}
alias Latu.ML.{Classification, Clustering, Evaluation, Feature, Functions, Regression, Stat}

session = Latu.connect!("sc://localhost:15002")

IO.puts("connected to Spark #{Latu.spark_version!(session)}")

# =============================================
# The registry, which needs no server at all
# =============================================
#
# Every operator, param and attribute is generated from three committed files. `h` on any
# constructor opens with the status the probe gave it, so tab completion is the real surface
# and this is only what it looks like from a script.

IO.inspect(length(ML.operators()), label: "operators in the registry")
IO.inspect(length(ML.operators(kind: :estimator)), label: "estimators")

ML.operators(group: :clustering, kind: :estimator)
|> Enum.map(& &1.name)
|> IO.inspect(label: "clusterers")

ML.params(:logistic_regression)
|> Enum.filter(&(&1.name in [:max_iter, :reg_param, :family]))
|> Enum.map(&{&1.name, &1.type, &1.default})
|> IO.inspect(label: "three of logistic_regression's params")

# The allowlist for a fitted model: what the **server** will let a `Fetch` ask for, not what
# PySpark exposes. Asking for anything else is `CONNECT_ML.ATTRIBUTE_NOT_ALLOWED`.
ML.attributes(LogisticRegressionModel)
|> Enum.map(& &1.name)
|> Enum.take(8)
|> IO.inspect(label: "eight of LogisticRegressionModel's attributes")

# =============================================
# Features
# =============================================

training =
  Latu.sql!(session, """
  SELECT CAST(label AS DOUBLE) AS label, CAST(x1 AS DOUBLE) AS x1, CAST(x2 AS DOUBLE) AS x2
  FROM VALUES
    (0.0, 0.0, 1.1), (0.0, 0.1, 1.0), (1.0, 2.0, 0.1), (1.0, 2.1, 0.2),
    (0.0, 0.2, 1.3), (1.0, 2.2, 0.3)
  AS t(label, x1, x2)
  """)

assembler = Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)

# `transform/2` is a **relation**: nothing has run yet, and the schema answers without executing.
features = ML.transform(assembler, training)

features |> Latu.schema!() |> Enum.map(& &1.name) |> IO.inspect(label: "assembled columns")

# =============================================
# Fit, and read the parameters back as tensors
# =============================================

{:ok, model} = ML.fit(Classification.logistic_regression(max_iter: 10, reg_param: 0.01), features)

{:ok, coefficients} = ML.attribute(model, :coefficients)
{:ok, intercept} = ML.attribute(model, :intercept)

IO.inspect(Nx.to_flat_list(coefficients), label: "coefficients")
IO.inspect(intercept, label: "intercept")

# The generated accessor is the same fetch with a name and a doc — `h
# LogisticRegressionModel.coefficients` says what Spark says about it.
{:ok, ^coefficients} = LogisticRegressionModel.coefficients(model)

# A training summary is another server-side object, reached lazily off the model. The generated
# accessor is the route to reach for: it carries the class, and a model of the wrong one is
# refused before the round trip.
summary = ML.summary(model)
{:ok, auc} = BinaryLogisticRegressionSummary.area_under_roc(summary)
IO.inspect(auc, label: "area under ROC, from the training summary")

# The generic verb underneath takes **either spelling** — the snake_case name `ML.attributes/1`
# advertises, or the camelCase one the server's allowlist holds.
{:ok, ^auc} = ML.attribute(summary, :area_under_roc)
{:ok, ^auc} = ML.attribute(summary, "areaUnderROC")

# =============================================
# Reading a Vector column
# =============================================
#
# `Latu.collect/2` refuses one: Spark describes it as a UDT and declines to say what it
# serialises as. Two ways through, and they answer different questions.

{:error, refused} = Latu.collect(features)
IO.puts("collect refuses it: #{refused.message}")

# `to_nx/2` reads the Arrow bytes, where the type *is* stated, and gives one {rows, width}
# tensor. No copy for a single batch.
{:ok, %{"features" => x}} = Latu.to_nx(features, columns: ["features"])
IO.inspect(Nx.shape(x), label: "features as a tensor")

# `vector_to_array` converts server-side instead, which is what you want when the destination
# is Explorer rather than a tensor.
{:ok, frame} =
  features
  |> Latu.select([:label, v: Functions.vector_to_array(:features)])
  |> Latu.to_explorer()

IO.inspect(Explorer.DataFrame.n_rows(frame), label: "rows through Explorer")

# =============================================
# Scoring, and evaluating
# =============================================

scored = ML.transform(model, features)

scored |> Latu.select([:label, :prediction]) |> Latu.show()

{:ok, metric} = ML.evaluate(Evaluation.binary_classification_evaluator(), scored)
IO.inspect(metric, label: "areaUnderROC on the scored frame")

# =============================================
# Persistence, and what a loaded model does not have
# =============================================

path = "/tmp/latu_ml_example/#{System.unique_integer([:positive])}"

:ok = ML.save(model, path, overwrite: true)
{:ok, loaded} = ML.load(session, :logistic_regression_model, path)

{:ok, same} = ML.attribute(loaded, :coefficients)
coefficients
|> Nx.equal(same)
|> Nx.to_flat_list()
|> IO.inspect(label: "coefficients survive a round trip")

# A loaded model has no training summary and no frame to rebuild one from — it never saw the
# data. `hint/1` names both ways of arriving here.
{:error, no_summary} = BinaryLogisticRegressionSummary.area_under_roc(ML.summary(loaded))
IO.puts("a loaded model's summary: #{ML.error_kind(no_summary)}")

:ok = ML.delete([model, loaded])

# =============================================
# The bracket
# =============================================
#
# A model is a server-side handle, so it has a lifetime. `with_model/3` deletes in an `after`,
# which is the shape to reach for when the handle does not need to outlive the expression.

{:ok, trained_auc} =
  ML.with_model(Classification.logistic_regression(max_iter: 5), features, fn m ->
    ML.evaluate(Evaluation.binary_classification_evaluator(), ML.transform(m, features))
  end)

IO.inspect(trained_auc, label: "fitted, evaluated and released in one expression")

# =============================================
# Pipelines and tuning, both client-side
# =============================================
#
# Spark has no `Fit` for either: PySpark folds the stages itself and so does this, which is why
# the on-disk format is the thing that has to match rather than the wire.

pipeline = ML.pipeline([assembler, Classification.logistic_regression(max_iter: 5)])
{:ok, fitted} = ML.fit(pipeline, training)

pipeline_scored = ML.transform(fitted, training)
pipeline_scored |> Latu.select([:label, :prediction]) |> Latu.show(num_rows: 3)

lr = Classification.logistic_regression()
grid = ML.param_grid(lr, max_iter: [5, 20], reg_param: [0.0, 0.1])
IO.inspect(length(grid), label: "param maps in the grid")

search =
  ML.cross_validator(
    estimator: lr,
    param_maps: grid,
    evaluator: Evaluation.binary_classification_evaluator(),
    num_folds: 2,
    seed: 1
  )

{:ok, best} = ML.fit(search, features)
IO.inspect(ML.best_index(best), label: "winning param map")

# One `Delete` for both, composites and all: the pipeline model owns its stage models and the
# search result owns its winner, and `delete/1` flattens each before it sends anything.
:ok = ML.delete([fitted, best])

# =============================================
# Statistics that run where the data is
# =============================================

Stat.correlation(features, :features) |> Latu.schema!() |> IO.inspect(label: "correlation schema")

# =============================================
# The cache
# =============================================
#
# A model is never silently evicted: over budget the **fit** is refused rather than something
# already held being dropped. So the whole lifetime story is the bracket and an explicit delete.

{:ok, k} = ML.fit(Clustering.k_means(k: 2), features)
{:ok, bytes} = ML.model_size(k)
IO.inspect(bytes, label: "one model's size in the cache")

{:ok, entries} = ML.cache_info(session)
IO.inspect(length(entries), label: "entries this session holds")

# `clean_cache/1` answers with how many entries it freed, not with `:ok` — the count is the
# interesting part, and an empty cache is a legitimate zero.
{:ok, freed} = ML.clean_cache(session)
IO.inspect(freed, label: "entries the clean freed")

{:ok, []} = ML.cache_info(session)
IO.puts("cache emptied")

# A regressor, so the example leaves one live handle to poke at in the REPL. `Latu.disconnect/2`
# with `release: true` is what ends the session, and with it everything above.
{:ok, linear} = ML.fit(Regression.linear_regression(max_iter: 5), features)

linear
