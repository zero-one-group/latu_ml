<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/zero-one-group/latu_ml/main/assets/latu-ml-lockup-dark.svg">
    <img alt="Latu ML" src="https://raw.githubusercontent.com/zero-one-group/latu_ml/main/assets/latu-ml-lockup.svg" width="420">
  </picture>
</p>

Spark MLlib from Elixir, over [Spark
Connect](https://spark.apache.org/docs/latest/spark-connect-overview.html), as a companion
package to [Latu](https://hexdocs.pm/latu).

The algorithms run on the cluster, where the data is. The Elixir side holds plans and handles.

## Install

```elixir
def deps do
  [
    {:latu, "~> 0.3"},
    {:latu_ml, "~> 0.1"}
  ]
end
```

Requires Elixir ~> 1.20 and a Spark **4.2.0** Connect server — the same one Latu targets, and
the only one either package claims.

## A server to talk to

Point at your cluster's `sc://` URL, or run one locally:

```bash
docker run -p 15002:15002 apache/spark:4.2.0 \
  /opt/spark/sbin/start-connect-server.sh --packages org.apache.spark:spark-connect_2.13:4.2.0
```

## The lines at the top of your file

`Latu.ML` is called qualified, the way `Latu` is. The operator groups are aliased because you
name one every time you build something, and they are PySpark's own grouping:

```elixir
alias Latu.ML
alias Latu.ML.{Classification, Clustering, Evaluation, Feature, Regression}
```

## A first model

```elixir
{:ok, session} = Latu.connect("sc://localhost:15002")

training =
  Latu.sql!(session, """
  SELECT CAST(label AS DOUBLE) AS label, CAST(x1 AS DOUBLE) AS x1, CAST(x2 AS DOUBLE) AS x2
  FROM VALUES (0.0, 0.0, 1.1), (0.0, 0.1, 1.0), (1.0, 2.0, 0.1), (1.0, 2.1, 0.2)
  AS t(label, x1, x2)
  """)

assembler = Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)
features = ML.transform(assembler, training)

{:ok, auc} =
  ML.with_model(Classification.logistic_regression(max_iter: 10), features, fn model ->
    ML.evaluate(Evaluation.binary_classification_evaluator(), ML.transform(model, features))
  end)
```

Four things are happening there, and three of them are the whole package. `ML.transform/2` is a
**lazy builder** — it hands back a `Latu.DataFrame` and reaches no server. `ML.fit/2`, which
`ML.with_model/3` calls, is an **action**, and what it hands back is a reference into a
server-side cache rather than a value. And `ML.with_model/3` deletes that reference in an
`after`, because the BEAM has no finalizer to do it for you.

This page shows the shape of the API. The
[quick start](https://hexdocs.pm/latu_ml/quick-start.html) shows results, and every line of it
is executed by the test suite — which is why the numbers there can be trusted and the ones in a
README cannot.

## What is in it

**Every operator the 4.2.0 server exposes**, generated rather than transcribed: 111 rows
carrying 991 params, extracted from PySpark's own classes and the server's own Scala. That is
68 constructors — 42 estimators, 18 transformers, 6 evaluators and the 2 operators that are
neither — and 623 accessors across 64 model and summary modules.

**The verbs.** `fit`, `transform`, `evaluate`, `attribute`, `attribute_frame`, `summary`,
`save`, `load`, `delete`, `with_model`, `model_size`, `cache_info`, `clean_cache`. Actions
return `{:ok, _} | {:error, %Latu.Error{}}` with `!` twins; builders return a struct or a
`Latu.DataFrame`.

**Attributes off a fitted model**, as `Nx` tensors, `Latu.ML.SparseVector`s or lazy frames —
`coefficients`, `feature_importances`, cluster centres, a training summary's ROC curve. What
you may ask for is the server's own allowlist, and the generated accessor per class is the
discovery surface: `h Latu.ML.Regression.LinearRegressionTrainingSummary.r2` prints Spark's
own prose.

**Pipelines and tuning**, client-side on both sides of the wire because Spark exposes no `Fit`
for either: `Latu.ML.pipeline/1`, `Latu.ML.param_grid/2`, `Latu.ML.cross_validator/1` and
`Latu.ML.train_validation_split/1`, all saveable and loadable in Spark's own on-disk format.

**The `ConnectHelper` route** for the things that are neither fitted nor applied:
`Latu.ML.Stat`'s three statistical tests, `Latu.ML.assign_clusters/2`,
`Latu.ML.find_frequent_sequential_patterns/2`, and the model constructors that build a
`StringIndexerModel` from labels or a `CountVectorizerModel` from a vocabulary.

**Two column functions**, `Latu.ML.Functions.vector_to_array/2` and
`Latu.ML.Functions.array_to_vector/1` — the way to read a `Vector` column, which
`Latu.collect/2` refuses because Spark describes it as a UDT and declines to say what it
serialises as.

**Not here.** Anything closure-shaped — `predict_batch_udf`, and `xgboost.spark`, whose
training is Python code shipped to Python workers even on Connect. The RDD-era `spark.mllib`.
Building a model out of parameters you already hold: a model exists only by a fit or a read,
and there is no wire path for the other direction. `OneVsRest`. And the deprecated torch-based
`pyspark.ml.connect` package from 3.4 and 3.5, which is not what any of this mirrors.

## What this is

Remote MLlib, not a local one. Two things are load-bearing:

1. **A model is not a value.** It lives on the server, in a per-session cache that offloads to
   disk under memory pressure and is gone when the session ends. `Latu.ML.with_model/3`
   brackets one; `Latu.ML.fit/2` and `Latu.ML.delete/1` are the pair for when the handle has to
   outlive an expression. Nothing here holds a process, so nothing can release it for you.

2. **The surface is generated from two extractors and checked by a probe.** Constructors,
   params, accessors and their docs come from `priv/ml_operators.exs` and
   `priv/ml_attributes.exs`, which are read off PySpark's classes and the server's `MLUtils`
   rather than typed by hand. Every constructor's `@doc` says whether a live server has
   actually run it — `:probed` or `:built` — so `h` tells you what is verified rather than what
   is claimed.

`Scholar` runs classical ML on one node, in memory, and hands you a model you can inspect and
ship. This runs MLlib on a cluster and hands you a reference. Reach for Scholar when the
training set fits a node; reach for this when it does not, or when the features already live in
a lakehouse. [With Nx and Scholar](https://hexdocs.pm/latu_ml/with-scholar-and-nx.html) is
that seam, measured.

## Status

**0.1.0, and pre-1.0 in the way that matters**: a minor version may rename or remove, and
`CHANGELOG.md` carries the migration in one line when it does.

All 66 runnable operators are `:probed` — fitted, applied or evaluated against a live 4.2.0
server, with every allowlisted attribute answered — and so are the eleven `ConnectHelper`
methods. No operator is `:missing`. Six saved directories cross over between PySpark and this
package in both directions.

## Where to go next

  * [Quick start](https://hexdocs.pm/latu_ml/quick-start.html)
    — connect, fit, read, score, release; every line of it executed
  * [Cookbook](https://hexdocs.pm/latu_ml/cookbook.html)
    — a recipe per model family, and the ones for pipelines, grid search and cleaning up
  * [Coming from PySpark ML](https://hexdocs.pm/latu_ml/from-pyspark-ml.html)
    — the five differences that matter, and what a model *is* here
  * [With Nx and Scholar](https://hexdocs.pm/latu_ml/with-scholar-and-nx.html)
    — getting numbers back onto the BEAM, where that is worth it, and where it stops working
  * [`docs/cheatsheet.cheatmd`](https://hexdocs.pm/latu_ml/cheatsheet.html)
    — every verb on one page
  * [`usage-rules.md`](https://hexdocs.pm/latu_ml/usage-rules.html)
    — the short set of rules that are not guessable from the function names, in the
    [`usage_rules`](https://github.com/ash-project/usage_rules) convention, so an agent can sync
    it into its context
  * [`docs/deviations.md`](https://hexdocs.pm/latu_ml/deviations.html)
    — every place the API departs from `pyspark.ml`, and why
  * [`CONTRIBUTING.md`](https://hexdocs.pm/latu_ml/contributing.html)
    — the servers, the two extractors, the oracle and the probe

Latu's own [docs](https://hexdocs.pm/latu) cover the DataFrame API this is built on; its
[usage rules](https://hexdocs.pm/latu/usage-rules.html) are worth having in context alongside
these.

## Development

`docker-compose.yml` brings up the two Spark servers; `CONTRIBUTING.md` has the rest.

```bash
docker compose up -d --wait
mix check      # format, compile, offline tests
mix check.all  # the above plus integration, against sc://localhost:15003
```

## How this was built

**Claude (Anthropic) wrote the overwhelming majority of the code, the tests and the
documentation.** The maintainers set the scope, made the design calls, ran every gate and
reviewed the result — the same division as Latu's, stated for the same reason.

What holds it up is the apparatus rather than the review: two extractors that reproduce the
committed registries byte for byte, a PySpark oracle that captures the protobuf PySpark *would*
have sent, a live probe that has run every operator, and `docs/decisions.md`, which records the
reasoning behind every non-obvious choice.

## License

Apache License 2.0. See the `LICENSE` file.
