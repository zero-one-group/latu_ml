# Cookbook

One recipe per thing people actually do. The [quick start](quick-start.md) is the tour; this is
the page you come back to.

Every `elixir` snippet here is executed by `mix check.all`
(`test/integration/guides_test.exs`), in order, sharing one set of bindings and one session. The
pattern matches are the assertions. Every model fitted below is released before the page ends.

Where a recipe departs from PySpark, [`docs/deviations.md`](../deviations.md) has the reason.

## Setting up

```elixir
alias Latu.ML
alias Latu.ML.{Classification, Clustering, Evaluation, Feature, Functions, Regression}

{:ok, session} = Latu.connect("sc://localhost:15003")
```

One frame carries the page. It has a binary `label` for the classifiers, a continuous `y` for the
regression, and two features. Twelve rows, so that a two-fold search later on has something to
cut:

```elixir
data =
  Latu.sql!(session, """
  SELECT CAST(label AS DOUBLE) AS label, CAST(y AS DOUBLE) AS y,
         CAST(x1 AS DOUBLE) AS x1, CAST(x2 AS DOUBLE) AS x2
  FROM VALUES
    (0.0, 1.0, 0.0, 1.1), (0.0, 1.2, 0.1, 1.0), (0.0, 1.4, 0.2, 1.3),
    (0.0, 1.6, 0.3, 1.2), (0.0, 1.8, 0.4, 1.1), (0.0, 2.0, 0.5, 1.4),
    (1.0, 4.0, 2.0, 0.1), (1.0, 4.2, 2.1, 0.2), (1.0, 4.4, 2.2, 0.3),
    (1.0, 4.6, 2.3, 0.1), (1.0, 4.8, 2.4, 0.2), (1.0, 5.0, 2.5, 0.4)
  AS t(label, y, x1, x2)
  """)

12 = Latu.count!(data)
```

**Every estimator wants one `Vector` column.** `VectorAssembler` builds it, and applying it is a
lazy builder. The frame below has reached no server:

```elixir
assembler = Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)
features = ML.transform(assembler, data)

["label", "y", "x1", "x2", "features"] = Latu.columns!(features)
```

## A classifier, end to end

The shape every recipe follows: build the estimator, fit it, read something off it, score with
it, release it.

```elixir
alias Latu.ML.Classification.{BinaryLogisticRegressionSummary, LogisticRegressionModel}

lr = Classification.logistic_regression(max_iter: 10, reg_param: 0.01)
{:ok, model} = ML.fit(lr, features)

{:ok, coefficients} = LogisticRegressionModel.coefficients(model)
{:ok, intercept} = LogisticRegressionModel.intercept(model)

{2} = Nx.shape(coefficients)
true = is_float(intercept)
```

The training summary is a second server-side object, reached lazily off the model:

```elixir
summary = ML.summary(model)
{:ok, auc} = BinaryLogisticRegressionSummary.area_under_roc(summary)
{:ok, accuracy} = BinaryLogisticRegressionSummary.accuracy(summary)

true = auc >= 0.0 and auc <= 1.0
true = accuracy >= 0.0 and accuracy <= 1.0
```

Scoring is a lazy builder too, so a scored frame composes like any other. **Select away the
`Vector` columns before collecting**: `features`, `rawPrediction` and `probability` are UDTs the
decoder refuses:

```elixir
scored = ML.transform(model, features)

{:ok, rows} = scored |> Latu.select([:label, :prediction]) |> Latu.collect()

12 = length(rows)
true = Enum.all?(rows, &(&1.prediction in [0.0, 1.0]))
```

An evaluator is inert data, so a second metric is a second value rather than an override at the
call site:

```elixir
{:ok, evaluated} = ML.evaluate(Evaluation.binary_classification_evaluator(), scored)

true = evaluated >= 0.0 and evaluated <= 1.0

:ok = ML.delete(model)
```

## A regression, and what its summary knows

`label_col:` points an estimator at a different target column; the features column is
`features` by default, which is why nothing above had to name it.

```elixir
alias Latu.ML.Regression.{LinearRegressionModel, LinearRegressionTrainingSummary}

{:ok, linear} = ML.fit(Regression.linear_regression(max_iter: 10, label_col: :y), features)

{:ok, weights} = LinearRegressionModel.coefficients(linear)
{2} = Nx.shape(weights)
```

A regression summary is where the diagnostics live. Some of it is values and some of it is
frames. `r2` is a number and arrives with `{:ok, _}`; `residuals` is a `Latu.DataFrame` and has
run nothing yet:

```elixir
fit_summary = ML.summary(linear)

{:ok, r2} = LinearRegressionTrainingSummary.r2(fit_summary)
{:ok, rmse} = LinearRegressionTrainingSummary.root_mean_squared_error(fit_summary)
{:ok, iterations} = LinearRegressionTrainingSummary.total_iterations(fit_summary)

true = r2 >= 0.0 and r2 <= 1.0
true = rmse >= 0.0
true = is_integer(iterations)

residuals = LinearRegressionTrainingSummary.residuals(fit_summary)
["residuals"] = Latu.columns!(residuals)

:ok = ML.delete(linear)
```

That split runs through the whole package: `Latu.ML.attribute/2` for a value, and
`Latu.ML.attribute_frame/2` for a frame, each refusing the other's names. The generated
accessors pick for you.

## A decision tree, and what a tree will not give you

A tree fits like anything else, and the interesting attributes are about its shape:

```elixir
alias Latu.ML.Classification.DecisionTreeClassificationModel

{:ok, tree} = ML.fit(Classification.decision_tree_classifier(max_depth: 3, seed: 1), features)

{:ok, depth} = DecisionTreeClassificationModel.depth(tree)
{:ok, nodes} = DecisionTreeClassificationModel.num_nodes(tree)

true = depth >= 1
true = nodes >= 3
```

`feature_importances` is the one attribute on this page that does **not** come back as a tensor.
Spark builds it with `Vectors.sparse`, because a tree usually ignores some of its features, and
this package never densifies for you. A vector that is sparse is generally sparse on purpose:

```elixir
{:ok, importances} = DecisionTreeClassificationModel.feature_importances(tree)

%Latu.ML.SparseVector{size: 2} = importances

{2} = importances |> Latu.ML.SparseVector.to_dense() |> Nx.shape()
```

`Latu.ML.SparseVector.to_dense/1` is the caller's decision, and `Nx` is where it lands.

**What you cannot get is the tree itself.** The allowlist gives a tree its depth, its node count,
its feature importances, the four `predict*` methods and `to_debug_string`. The `predict*` methods
are one round trip per row, so not a scoring path. The debug string has no compatibility promise.
There is no node table and no thresholds as data:

```elixir
{:ok, dump} = DecisionTreeClassificationModel.to_debug_string(tree)

true = is_binary(dump)
```

So a tree is scored where it lives, with `Latu.ML.transform/2`. The parameters-out seam in
[With Nx and Scholar](with-scholar-and-nx.md) ends at the model families whose parameters *are*
the model.

An ensemble adds one shape worth knowing: `trees` answers with a list of **cache entries**, one
per tree, and they are yours to give back.

```elixir
alias Latu.ML.Classification.RandomForestClassificationModel

forest = Classification.random_forest_classifier(num_trees: 3, max_depth: 3, seed: 1)
{:ok, rf} = ML.fit(forest, features)

{:ok, n} = RandomForestClassificationModel.get_num_trees(rf)
{:ok, trees} = RandomForestClassificationModel.trees(rf)

3 = n
3 = length(trees)

:ok = ML.delete(trees)
:ok = ML.delete([tree, rf])
```

`Latu.ML.delete/1` takes a list, and a mixed one: models, pipeline models and search results in
the same call become a single `Delete`.

## Clustering, with no labels to speak of

Unsupervised is the same shape with the label column absent:

```elixir
alias Latu.ML.Clustering.{KMeansModel, KMeansSummary}

{:ok, kmeans} = ML.fit(Clustering.k_means(k: 2, seed: 1), features)

{:ok, centres} = KMeansModel.cluster_center_matrix(kmeans)
{2, 2} = Nx.shape(centres)
```

A `Matrix` comes back row-major whatever Spark's `isTransposed` flag says. A storage flag is not
a shape. The summary carries the sizes and the cost:

```elixir
clustering = ML.summary(kmeans)

{:ok, sizes} = KMeansSummary.cluster_sizes(clustering)
{:ok, cost} = KMeansSummary.training_cost(clustering)

2 = length(sizes)
12 = Enum.sum(sizes)
true = cost >= 0.0

:ok = ML.delete(kmeans)
```

## A pipeline, saved and read back

`Pipeline` has no `Fit` on the wire, so both PySpark and this package fold over the stages
themselves. Which means the thing that has to match is the **on-disk format**, not the wire.

```elixir
scaler = Feature.standard_scaler(input_col: :features, output_col: :scaled, with_mean: true)

pipeline =
  ML.pipeline([
    assembler,
    scaler,
    Classification.logistic_regression(max_iter: 10, features_col: :scaled)
  ])

{:ok, fitted} = ML.fit(pipeline, data)

%Latu.ML.PipelineModel{stages: [_assembler, _scaler, _lr]} = fitted
```

The fitted pipeline transforms in one call, composing each stage's relation lazily:

```elixir
piped = ML.transform(fitted, data)

{:ok, piped_rows} = piped |> Latu.select([:label, :prediction]) |> Latu.collect()
12 = length(piped_rows)
```

Saving writes Spark's own directory layout to a path **on the server**. The write happens where
the session is, so a bare path is the driver's disk. Anything a second machine has to read wants a
URL the cluster's filesystem understands:

```elixir
path = "/tmp/latu_ml_cookbook/#{System.unique_integer([:positive])}"

:ok = ML.save(fitted, path, overwrite: true)
{:ok, reloaded} = ML.load(session, :pipeline_model, path)

%Latu.ML.PipelineModel{stages: [_, _, _]} = reloaded
```

A loaded model holds a **fresh cache reference**. Reading one costs a cache entry exactly as
fitting one does. It carries no training frame, so it has no summary and nothing to rebuild one
from.

```elixir
:ok = ML.delete([fitted, reloaded])
```

Two things about that directory before you rely on it. Each stage underneath is written by the
*server*, so every stage is in Scala's own format and loads anywhere. The wrapper is not: its
metadata says `class: "pyspark.ml.pipeline.Pipeline"`, which Scala's reader refuses. It refuses
PySpark's own saved pipelines too, for the same one key.

## Searching a grid

`ML.param_grid/2` builds the grid as data. Each entry carries the **uid** of the operator it
sets, because the thing searched is usually a pipeline and a param belongs to one stage of it:

```elixir
tuned = Classification.logistic_regression(max_iter: 10)
grid = ML.param_grid(tuned, reg_param: [0.0, 0.1])

2 = length(grid)
```

The search itself is client code on both sides: folds cut with `rand(seed)` and range filters,
exactly as `CrossValidator._kFold` does it. Pass a `seed:`. Without one the draw differs between
runs and the search is not repeatable:

```elixir
search =
  ML.cross_validator(
    estimator: tuned,
    param_maps: grid,
    evaluator: Evaluation.binary_classification_evaluator(),
    num_folds: 2,
    seed: 1
  )

{:ok, best} = ML.fit(search, features)

2 = length(best.avg_metrics)
true = ML.best_index(best) in [0, 1]
```

The winner is refit on the whole frame and lives in `best_model`; the fold models are released
as their metrics are read, unless you asked for them with `collect_sub_models: true`.

```elixir
%Latu.ML.Model{} = best.best_model

:ok = ML.delete(best)
```

`Latu.ML.larger_better?/1` decides argmax from argmin, and it is pure. The server's allowlist has
no `isLargerBetter`, so the registry carries PySpark's own client-side overrides.

```elixir
true = ML.larger_better?(Evaluation.binary_classification_evaluator())
false = ML.larger_better?(Evaluation.regression_evaluator(metric_name: "rmse"))
```

## Embeddings from the BEAM, clustered on the cluster

[With Nx and Scholar](with-scholar-and-nx.md) is about numbers coming *down*. This is the other
direction: a vector computed on the BEAM going *up*, to become the `features` column an
estimator wants. Spark Connect ships no Elixir function to the executors, so the embedding model
runs on one BEAM and only its output crosses.

Four short documents, carrying their ids, because a vector is worth nothing beside the wrong row:

```elixir
docs =
  Latu.sql!(session, """
  SELECT * FROM VALUES
    (1, 'the cat sat on the mat'), (2, 'a kitten slept on a rug'),
    (3, 'quarterly revenue rose'), (4, 'annual earnings increased')
  AS t(id, text)
  """)

{:ok, texts} = docs |> Latu.sort(:id) |> Latu.collect()

4 = length(texts)
ids = Enum.map(texts, & &1.id)
```

In your own pipeline the next step is a serving, and the shape it has to produce is the only
thing the rest of this recipe cares about: one list of floats per row, all the same width.

> **Not executed.** Bumblebee is not a dependency of this package.

```elixir
repo = {:hf, "sentence-transformers/all-MiniLM-L6-v2"}

{:ok, model} = Bumblebee.load_model(repo)
{:ok, tokenizer} = Bumblebee.load_tokenizer(repo)

serving = Bumblebee.Text.text_embedding(model, tokenizer, output_pool: :mean_pooling)

vectors =
  texts
  |> Enum.map(& &1.text)
  |> then(&Nx.Serving.run(serving, &1))
  |> Enum.map(&Nx.to_flat_list(&1.embedding))
```

Here it is four fixed vectors of width four, so that what follows asserts on Spark's half rather
than on a model's:

```elixir
vectors = [
  [0.10, 0.11, 0.90, 0.91],
  [0.12, 0.09, 0.88, 0.93],
  [0.91, 0.89, 0.10, 0.12],
  [0.88, 0.92, 0.11, 0.09]
]
```

### The way up is not `create_dataframe/3`

One fence, so it does not cost you an afternoon. Note where the refusal lands: the bytes ship,
and the frame comes back looking ordinary, because the server does not read the Arrow schema
until the first action.

```elixir
with_list = Latu.create_dataframe!(session, id: [1], embedding: [[0.1, 0.2]])

{:error, refused} = Latu.dtypes(with_list)
true = refused.message =~ "Unsupported arrow type LargeList"
```

Explorer writes Arrow through Polars, which gives every list 64-bit offsets, which is Arrow's
`LargeList`. Spark's `ArrowUtils.fromArrowField` has an arm for `List` and one for `ListView`
and none for `LargeList`. It *does* have one for `LargeUtf8`, which is why a string column goes
up without complaint and this one does not. A `schema:` naming `ARRAY<DOUBLE>` does not save it
either: the schema is a cast applied after the server has already read the Arrow schema. PySpark
never meets this, because pyarrow writes 32-bit offsets.

### The way up is Parquet

Parquet's list is Parquet's own, so Polars' offsets never reach Spark's Arrow reader.
`Latu.copy_to_fs/3` puts the bytes on the cluster's filesystem and `Latu.read/2` reads them back:

```elixir
frame = Explorer.DataFrame.new(id: ids, embedding: vectors)
parquet = "/tmp/latu_ml_cookbook/#{System.unique_integer([:positive])}.parquet"

:ok = Latu.set_conf(session, "spark.sql.artifact.copyFromLocalToFs.allowDestLocal", "true")
:ok = Latu.copy_to_fs(session, parquet, Explorer.DataFrame.dump_parquet!(frame))

uploaded = Latu.read(session, format: "parquet", path: parquet)

[{"id", "bigint"}, {"embedding", "array<double>"}] = Latu.dtypes!(uploaded)
```

That conf is a single-machine wart rather than part of the recipe: `copy_to_fs/3` refuses a
destination on the driver's own local disk unless the session allows it, and a cluster whose
default filesystem is not local needs nothing. `Latu.copy_to_fs/3` has the whole of it.

### And then it is an ordinary features column

`Latu.ML.Functions.array_to_vector/1` is the one step MLlib needs. It is an
`org.apache.spark.ml` function rather than a SQL one, which is why it lives here and not in
`Latu.Functions`:

```elixir
features_up = Latu.select(uploaded, [:id, features: Functions.array_to_vector(:embedding)])

{:ok, clusters} = ML.fit(Clustering.k_means(k: 2, seed: 1), features_up)
{:ok, centres} = KMeansModel.cluster_center_matrix(clusters)

{2, 4} = Nx.shape(centres)
```

Two centres of four dimensions, the embedding's own width. Scoring joins back to the text by id,
which is the only thing tying a vector to what it came from:

```elixir
{:ok, labelled} =
  clusters
  |> ML.transform(features_up)
  |> Latu.select([:id, :prediction])
  |> Latu.join(docs, on: :id)
  |> Latu.sort(:id)
  |> Latu.collect()

[
  %{id: 1, prediction: cats, text: "the cat sat on the mat"},
  %{id: 2, prediction: cats},
  %{id: 3, prediction: money},
  %{id: 4, prediction: money}
] = labelled

true = cats != money

:ok = ML.delete(clusters)
```

### When there is no filesystem to write to

The vector can ride up as a string and be rebuilt server-side. Lossless, since
`Float.to_string/1` round-trips, and it needs no conf and no file. It also carries every digit
as text and builds the whole relation inside the plan, so it is the small-data route rather than
the one to reach for first:

```elixir
as_text =
  Latu.create_dataframe!(session,
    id: ids,
    emb: Enum.map(vectors, &Enum.map_join(&1, ",", fn f -> Float.to_string(f) end))
  )

rebuilt =
  Latu.sql!(session, "SELECT id, CAST(split(emb, ',') AS ARRAY<DOUBLE>) AS embedding FROM v",
    views: [v: as_text]
  )

[{"id", "bigint"}, {"embedding", "array<double>"}] = Latu.dtypes!(rebuilt)
```

## Cleaning up after yourself

A model is a server-side resource with no finalizer behind it. Three things help.

`Latu.ML.with_model/3` is the bracket: acquire, use, release in an `after`. It is the right shape
whenever the handle does not have to outlive the expression:

```elixir
{:ok, bracketed} =
  ML.with_model(Classification.logistic_regression(max_iter: 5), features, fn m ->
    ML.evaluate(Evaluation.binary_classification_evaluator(), ML.transform(m, features))
  end)

true = is_float(bracketed)
```

`Latu.ML.cache_info/1` says what the session is actually holding, and `Latu.ML.model_size/1`
what one entry costs. Every recipe above deleted what it fitted:

```elixir
{:ok, entries} = ML.cache_info(session)

[] = entries
```

`Latu.ML.clean_cache/1` empties the lot, and answers with how many entries it freed rather than a
bare `:ok`. An empty cache is a legitimate zero:

```elixir
{:ok, 0} = ML.clean_cache(session)
```

Ending the session ends everything in it, which is the backstop rather than the plan:

```elixir
Latu.disconnect(session, release: true)
```

## Where to go next

  * [Coming from PySpark ML](from-pyspark-ml.md). The five differences, and what a model *is*.
  * [With Nx and Scholar](with-scholar-and-nx.md). The seam back onto the BEAM, and where it ends.
  * [`docs/deviations.md`](../deviations.md). The spelling table, and the reason for each row.
