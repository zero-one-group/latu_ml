# Quick start

Connect, fit one model, read it, score with it, and give it back. Ten minutes, one frame, no
files on disk.

Every `elixir` snippet on this page is executed by `mix check.all`
(`test/integration/guides_test.exs`), in order, sharing one set of bindings. The pattern matches
are the assertions. `{2} = Nx.shape(coefficients)` is documentation and a test at once.

This assumes Latu itself is familiar; if it is not,
[Latu's quick start](https://hexdocs.pm/latu/quick-start.html) comes first.

## Connecting, and the aliases

`Latu.ML` is called qualified. The operator groups are aliased because you name one every time
you build something, and an accessor module is aliased when you use it more than once:

```elixir
alias Latu.ML
alias Latu.ML.Classification.BinaryLogisticRegressionSummary
alias Latu.ML.{Classification, Evaluation, Feature}

{:ok, session} = Latu.connect("sc://localhost:15003")
```

Nothing was started. A session is Latu's plain struct, and this package adds no process to it.

## A frame to learn from

MLlib is DataFrame in, DataFrame out. Six rows are enough to show every shape on this page:

```elixir
training =
  Latu.sql!(session, """
  SELECT CAST(label AS DOUBLE) AS label, CAST(x1 AS DOUBLE) AS x1, CAST(x2 AS DOUBLE) AS x2
  FROM VALUES
    (0.0, 0.0, 1.1), (0.0, 0.1, 1.0), (0.0, 0.2, 1.3),
    (1.0, 2.0, 0.1), (1.0, 2.1, 0.2), (1.0, 2.2, 0.3)
  AS t(label, x1, x2)
  """)

6 = Latu.count!(training)
```

## Assembling features

Every MLlib estimator wants its features in **one** `Vector` column. `VectorAssembler` builds
one, and applying it is a lazy builder: nothing has reached the server yet.

```elixir
assembler = Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)
features = ML.transform(assembler, training)

["label", "x1", "x2", "features"] = Latu.columns!(features)
```

`Latu.columns!/1` answered without running anything. A `Transform` is a relation, so the schema
comes back from an analyse rather than an execution.

## Fitting

`ML.fit/2` is an action, and what comes back is a `Latu.ML.Model`: a reference into the
session's ML cache, not a value you hold.

```elixir
lr = Classification.logistic_regression(max_iter: 10, reg_param: 0.01)
{:ok, model} = ML.fit(lr, features)

%Latu.ML.Model{class: "org.apache.spark.ml.classification.LogisticRegressionModel"} = model
```

Params are Elixir-spelled, and only what you set is sent. `max_iter:` is `maxIter` on the wire,
and the server fills in every default you left alone. A value of the wrong **kind** is refused
here. A value out of **range** is Spark's own `ParamValidators` to refuse.

## Reading the model

What you may ask a model for is the server's own allowlist. There is a generated accessor per
allowlisted attribute, on a module per class. Reach for the accessor: it carries the class, the
wire name and Spark's own docs, so tab completion is the discovery surface.

```elixir
alias Latu.ML.Classification.LogisticRegressionModel

{:ok, coefficients} = LogisticRegressionModel.coefficients(model)
{:ok, intercept} = LogisticRegressionModel.intercept(model)

{2} = Nx.shape(coefficients)
true = is_float(intercept)
```

A `Vector` comes back as an `Nx.Tensor`; a sparse one comes back as a `Latu.ML.SparseVector`,
never silently densified. `ML.attribute/2` is the generic verb underneath, and it takes either
spelling: the snake_case name the registry advertises, or the camelCase one the allowlist holds:

```elixir
{:ok, ^intercept} = ML.attribute(model, :intercept)
{:ok, ^intercept} = ML.attribute(model, "intercept")
```

## The training summary

Ten of the 43 model classes record one. It is a second server-side object, reached lazily off the
model. Spark has no separate cache key for it, so nothing is sent until you ask it something:

```elixir
summary = ML.summary(model)
{:ok, area} = BinaryLogisticRegressionSummary.area_under_roc(summary)

true = area >= 0.0 and area <= 1.0
```

Some of a summary's attributes are frames rather than values, and those are lazy too:

```elixir
roc = BinaryLogisticRegressionSummary.roc(summary)

["FPR", "TPR"] = Latu.columns!(roc)
```

## Scoring, and evaluating

`ML.transform/2` on a fitted model is the same lazy builder as on a transformer, so a scored
frame is a `Latu.DataFrame` you can go on composing with:

```elixir
scored = ML.transform(model, features)

{:ok, rows} = scored |> Latu.select([:label, :prediction]) |> Latu.collect()

6 = length(rows)
```

Note the `select`. **A `Vector` column cannot be collected.** On 4.2.0 the server describes
`features`, `rawPrediction` and `probability` as UDTs with no SQL type. Latu's dtype guard refuses
them rather than guessing a layout. `Latu.to_nx/2` reads the Arrow bytes, where the type *is*
stated:

```elixir
{:ok, %{"features" => x}} = Latu.to_nx(features, columns: ["features"])

{6, 2} = Nx.shape(x)
```

An evaluator is inert data, so a second metric is a second value rather than an override at the
call site:

```elixir
{:ok, auc} = ML.evaluate(Evaluation.binary_classification_evaluator(), scored)

true = auc >= 0.0 and auc <= 1.0
```

## Giving the model back

The BEAM has no finalizer, so the cache entry is yours to release. `ML.delete/1` is the explicit
form:

```elixir
:ok = ML.delete(model)
{:ok, []} = ML.cache_info(session)
```

When the handle does not need to outlive the expression, `ML.with_model/3` is the shape to
reach for. It deletes in an `after`, so a raise in the body still releases:

```elixir
{:ok, bracketed} =
  ML.with_model(Classification.logistic_regression(max_iter: 5), features, fn m ->
    ML.evaluate(Evaluation.binary_classification_evaluator(), ML.transform(m, features))
  end)

true = is_float(bracketed)
{:ok, []} = ML.cache_info(session)
```

## The registry, which needs no server at all

Every operator, param and attribute is data in the package, and queryable:

```elixir
true = length(ML.operators()) > 100
[_ | _] = ML.operators(group: :clustering, kind: :estimator)
[_ | _] = ML.params(:logistic_regression)
[_ | _] = ML.attributes(LogisticRegressionModel)
```

Then hand the session back:

```elixir
Latu.disconnect(session, release: true)
```

## Where to go next

  * [Cookbook](cookbook.md). A recipe per model family, plus pipelines, grid search and cleaning
    up.
  * [Coming from PySpark ML](from-pyspark-ml.md). The five differences, if MLlib is already
    familiar.
  * [With Nx and Scholar](with-scholar-and-nx.md). The seam back onto the BEAM, and where it ends.
  * [`usage-rules.md`](../../usage-rules.md). The rules that are not guessable from the function
    names.
