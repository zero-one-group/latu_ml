# Coming from PySpark ML

`latu_ml` sends the same `MlCommand`s PySpark sends, so what you know about MLlib transfers
whole: the same operators, the same params, the same on-disk format. What changes is what a
*model* is in your program.

Five differences account for nearly everything. The spelling — `maxIter` to `max_iter`, and
twenty-five more rows — is [`docs/deviations.md`](../deviations.md), which also carries the
reason for each.

Every `elixir` snippet here is executed by `mix check.all`
(`test/integration/guides_test.exs`).

```elixir
alias Latu.ML
alias Latu.ML.Classification.BinaryLogisticRegressionSummary
alias Latu.ML.{Classification, Feature, Functions}

session = Latu.connect!("sc://localhost:15003")

training =
  Latu.sql!(session, """
  SELECT CAST(label AS DOUBLE) AS label, CAST(x1 AS DOUBLE) AS x1, CAST(x2 AS DOUBLE) AS x2
  FROM VALUES
    (0.0, 0.0, 1.1), (0.0, 0.1, 1.0), (1.0, 2.0, 0.1), (1.0, 2.1, 0.2)
  AS t(label, x1, x2)
  """)

assembler = Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)
features = ML.transform(assembler, training)
```

## 1. A model is not a value

In PySpark a fitted model is an object in your process. On Connect it holds a reference to a
server-side one, and a refcount frees it in `__del__` when the last name goes out of scope — so
the lifetime is invisible and you never think about it.

Here it is visible on purpose. `ML.fit/2` hands back a `%Latu.ML.Model{}`, which is a reference
into a per-session cache: inert data on the BEAM, and something real on the server that stays
real until you say otherwise. The BEAM has no finalizers, so there is nothing to hook — and
faking one would need a process or a NIF resource, neither of which this package holds.

So the model has to be released, and the shape for that is a bracket:

```elixir
{:ok, bracketed} =
  ML.with_model(Classification.logistic_regression(max_iter: 5), features, fn model ->
    ML.attribute(model, :intercept)
  end)

true = is_float(bracketed)
```

`with_model/3` deletes in an `after`, so a raise in the body still releases. When the handle has
to outlive the expression, `fit/2` and `delete/1` are the pair — and `delete/1` on a
`PipelineModel` or a search result empties the whole tree, because those own the models inside
them:

```elixir
{:ok, model} = ML.fit(Classification.logistic_regression(max_iter: 10), features)
{:ok, coefficients} = ML.attribute(model, :coefficients)

{2} = Nx.shape(coefficients)
```

What you do *not* have to worry about is eviction. A model is offloaded to disk under memory
pressure and comes back transparently; over budget it is the **fit** that is refused rather than
something already held being dropped. A `%Latu.ML.Model{}` does not go stale underneath you.

## 2. An attribute is an action, and the server has an allowlist

`model.coefficients` is a property in PySpark that raises on failure. It is a round trip either
way — but here it is spelled as one:

```elixir
{:ok, intercept} = ML.attribute(model, :intercept)
true = is_float(intercept)
```

The `{:ok, _}` is Latu's convention for anything that reaches the server, and there is a `!`
twin. What is new is that **the server decides what you may ask for**. `MLUtils` holds an
allowlist per class, and anything outside it is `CONNECT_ML.ATTRIBUTE_NOT_ALLOWED` — so the set
is not "whatever PySpark exposes":

```elixir
allowed = ML.attributes(Classification.LogisticRegressionModel)

true = :coefficients in Enum.map(allowed, & &1.name)
true = length(allowed) > 5
```

**Reach for the generated accessor first.** There is one function per allowlisted attribute, on
a module per model class, carrying the class and the wire name and Spark's own docs — so
`h LogisticRegressionModel.coefficients` tells you what Spark says, and tab completion is the
discovery surface. `ML.attribute/2` underneath takes the **wire** name, which is why
`:coefficients` works and `:area_under_roc` would not; the accessor is what translates.

## 3. A Vector column does not come back as a vector

This is the one that surprises. `df.collect()` in PySpark hands you `DenseVector` objects,
because PySpark has a Python class for them. Latu has no such thing, and more to the point Spark
does not describe the column well enough to build one: it says "UDT" and declines to say what it
serialises as.

```elixir
scored = ML.transform(model, features)

{:error, refused} = Latu.collect(scored)
true = refused.message =~ "UDT that does not say its SQL type"
```

Two ways through, and they answer different questions. `Latu.to_nx/2` reads the Arrow bytes,
where the type *is* stated, and gives you one `{rows, width}` tensor:

```elixir
{:ok, %{"features" => x}} = Latu.to_nx(features, columns: ["features"])

{4, 2} = Nx.shape(x)
```

`Functions.vector_to_array/2` is `pyspark.ml.functions.vector_to_array`, and converts on the
server so the column arrives as an ordinary `array<double>` any reader takes:

```elixir
{:ok, rows} =
  scored
  |> Latu.select([:prediction, v: Functions.vector_to_array(:features)])
  |> Latu.collect()

4 = length(rows)
[_, _] = hd(rows).v
```

`select` or `drop` the Vector columns is the third way, and often the right one — a prediction
column is a double and was never the problem.

## 4. A summary is a second object, and the cache drops it

`model.summary` is a property in PySpark. Here it is a lazy handle to another server-side
object, reached by asking the model — Spark has no separate cache key for one:

```elixir
summary = ML.summary(model)
{:ok, area} = BinaryLogisticRegressionSummary.area_under_roc(summary)

true = is_float(area) and area >= 0.0 and area <= 1.0
```

The part worth knowing: a model is *offloaded* under memory pressure, but a summary is
**dropped** — a model's save format does not carry one, so the model that comes back from disk
has none. The fetch then throws `CONNECT_ML.MODEL_SUMMARY_LOST`, and this package does what
PySpark does: rebuilds the summary from the training frame and asks again, once.

Except where there is no frame. A model from `load/3` never saw the data, so nothing can be
rebuilt and the server's refusal stands — which is the one case the recovery must not fire on.

```elixir
:ok = ML.delete(model)
```

## 5. Pipelines and tuning are client code on both sides

`Pipeline`, `CrossValidator` and `TrainValidationSplit` have no `Fit` on the wire. PySpark loops
over the stages itself, cuts folds with `rand(seed)` and range filters, and writes the layout by
hand — and so does this. The two are the same shape:

```elixir
%Latu.ML.Pipeline{stages: [_assembler, _lr]} =
  ML.pipeline([assembler, Classification.logistic_regression(max_iter: 5)])

grid = ML.param_grid(Classification.logistic_regression(), reg_param: [0.0, 0.1])
2 = length(grid)
```

Because both sides are client code, **the wire is not what has to match — the format is**. A
grid entry here carries the uid of the operator it sets, since the thing searched is usually a
pipeline and a param belongs to one stage of it. `dev/README.md` has the interop dance: PySpark
reads what this wrote, and the reverse, for six different directories.

## Discovery: the registry, not the docs

PySpark's answer to "what params does this take" is its documentation. Here it is in the package,
generated from PySpark's own classes and the server's own Scala, and queryable:

```elixir
true = length(ML.operators()) > 100
[_ | _] = ML.operators(kind: :evaluator)
[_ | _] = ML.params(:logistic_regression)
```

Every constructor's `@doc` opens with whether a live server has actually run it — `:probed`,
`:built` or `:missing` — so `h Classification.logistic_regression` tells you what is verified
rather than what is claimed.

## Things deliberately not here

Anything closure-shaped, because there is no way to ship an Elixir function to a Python worker:
`predict_batch_udf`, and `xgboost.spark`, whose training is Python code even on Connect. The
RDD-era `spark.mllib`. Building a model from parameters you hold — a model exists only by a fit
or a read, and there is no wire path for the other direction. `OneVsRest`, which shares nothing
with the validators but the word *meta-algorithm*. And the deprecated torch-based
`pyspark.ml.connect` package from 3.4 and 3.5, which is not what any of this mirrors.

`Summarizer` is reachable but not wrapped: its `aggregate_metrics` resolves, and reading one
metric out of the struct it answers with needs field access Latu does not build yet.

## Where to go next

[With Nx and Scholar](with-scholar-and-nx.md) is the other direction — getting numbers back onto
the BEAM, and where that stops working. [`docs/deviations.md`](../deviations.md) is the spelling
table and the reasons.

```elixir
Latu.disconnect(session, release: true)
```
