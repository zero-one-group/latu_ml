# Changelog

`latu_ml` follows [Semantic Versioning](https://semver.org). Before 1.0, a minor version may
rename or remove; each such change is listed here with the migration in one line.

## 0.1.0 — 2026-09-07

First release, against Spark **4.2.0**, and a companion to `latu ~> 0.3`.

Spark MLlib from Elixir over Spark Connect: `fit`, `transform`, `evaluate`, model and summary
attributes, `save`/`load` in Spark's own on-disk format, the model cache and its bracket, the
`ConnectHelper` route, pipelines and grid search. The operator surface is **generated** — 111
rows carrying 991 params, extracted from PySpark's own classes and the server's own
`ALLOWED_ATTRIBUTES` rather than transcribed — giving 68 constructors and 623 accessors across
64 model and summary modules. All 66 runnable operators have been run against a live server.

Every place the API departs from `pyspark.ml` is in `docs/deviations.md`, with why. What
follows is what each milestone added, oldest last.

`Latu.ML.Functions.vector_to_array/2` and `array_to_vector/1`: the way to read a `Vector`
column, which `Latu.collect/2` refuses. Spark keeps both in an *internal* function registry
rather than the builtin one, so they are unreachable from SQL and reachable over Connect — the
wire form is a plain `UnresolvedFunction` with `is_internal` left unset. `docs/deviations.md`
has the whole of it. `Summarizer` uses the same route and is not wrapped yet: reading a metric
out of the struct `aggregate_metrics` answers with needs field access Latu does not build.

`Latu.ML.evaluate/2` runs an evaluator; `Latu.ML.save/3` and `Latu.ML.load/3` write a model or
an unfitted operator to Spark's own on-disk format and read one back. All six evaluators are
`:probed` as a result, and `dev/README.md` has the interop dance that shows PySpark reading what
this wrote and the reverse.

The `ConnectHelper` route: `Latu.ML.Stat`'s three statistical tests, `Latu.ML.assign_clusters/2`
and `find_frequent_sequential_patterns/2` for the two operators that are neither fitted nor
applied, `from_labels/3` and its siblings generated onto their model modules,
`Latu.ML.Feature.load_default_stop_words/2`, and `Latu.ML.helper/3` under all of them. The
registry grew a fifth operator kind and a helpers section, both extracted; `Latu.ML.Plan` grew
nested lists, which also closes `Bucketizer.splitsArray`.

Pipelines: `Latu.ML.pipeline/1` builds one, `Latu.ML.fit/2` folds a frame through its stages,
and `save/3` and `load/3` carry both a fitted and an unfitted one in Spark's own directory
layout. There is no wire form for a pipeline in any client, so the fold is this package's own —
`dev/probe_ml.exs`'s pipelines section checks it lands where fitting the stages by hand does,
and `--interop` has PySpark reading a pipeline Latu wrote and the reverse. The layout is
PySpark's, which Scala cannot read; `docs/deviations.md` says so plainly and `docs/decisions.md`
has the one key it turns on.

Tuning: `Latu.ML.param_grid/1,2` builds a grid as data, `cross_validator/1` and
`train_validation_split/1` search one, and `fit/2` runs the search — folds cut with
`rand(seed)` and a range as Spark does it, one fit at a time, the winner refit on the whole
frame. A sub-model is released as its metric is read, so a five-by-six search holds one cache
entry rather than thirty; `collect_sub_models: true` keeps them and `delete/1` gives them back.
`larger_better?/1` and `best_index/1` are public, read off the six client-side `isLargerBetter`
overrides the registry now extracts. `dev/pyspark_oracle.py --interop` prints PySpark's metrics
for the same grid, seed and rows, which is the only check either side has.

`save/3` and `load/4` carry a search too, in Spark's own layout: `metadata` with the
validator's params and its encoded grid, `estimator/` and `evaluator/`, and for a fitted one
`bestModel/` and the metrics. `sub_models: true` writes the fold models and asks for them back
— **a plain load leaves them on disk**, which is where this deviates from PySpark and why is in
`docs/deviations.md`. PySpark reads a search this wrote and the reverse; `dev/README.md` has
the dance.

Documentation, to Latu's standard: a [quick start](docs/guides/quick-start.md) and a
[cookbook](docs/guides/cookbook.md) — a recipe per model family, plus pipelines, grid search and
cleaning up — both executed by `test/integration/guides_test.exs` like the two "coming from"
guides. `usage-rules.md` ships for agents in the `usage_rules` convention, and
`docs/cheatsheet.cheatmd` puts every hand-written verb on one page with
`test/latu/cheatsheet_test.exs` failing on either kind of drift. The README gained an install
section, a server to point at and links to all of it; `CONTRIBUTING.md` carries the extractors,
the oracle, the probe and the release checklist.
