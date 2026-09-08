# Changelog

`latu_ml` follows [Semantic Versioning](https://semver.org). Before 1.0, a minor version may
rename or remove; each such change is listed here with the migration in one line.

## 0.1.0 — 2026-09-07

First release, against Spark **4.2.0**, and a companion to `latu ~> 0.3`.

Spark MLlib from Elixir over Spark Connect: `fit`, `transform`, `evaluate`, model and summary
attributes, `save`/`load` in Spark's own on-disk format, the model cache and its bracket, the
`ConnectHelper` route, pipelines and grid search. The operator surface is **generated**: 111
rows carrying 991 params, extracted from PySpark's own classes and the server's own
`ALLOWED_ATTRIBUTES` rather than transcribed. That gives 68 constructors and 623 accessors
across 64 model and summary modules. All 66 runnable operators have been run against a
live server.

Every place the API departs from `pyspark.ml` is in `docs/deviations.md`, with why.

- **Operators and models.** `Latu.ML.fit/2` and `transform/2`, `attribute/2` and
  `attribute_frame/2` for what a fitted model knows, and a generated accessor module per model
  and summary class. A model is a handle into a server-side cache, not a value. `with_model/3`
  brackets one and `delete/1` gives it back.
- **Reading a `Vector` column.** `Latu.ML.Functions.vector_to_array/2` and `array_to_vector/1`,
  which is what `Latu.collect/2` refuses. Spark keeps both in an internal function registry
  rather than the builtin one, so they are unreachable from SQL and reachable over Connect. The
  wire form is a plain `UnresolvedFunction` with `is_internal` left unset. `Summarizer` takes
  the same route and is not wrapped: reading one metric out of the struct it answers with needs
  field access Latu does not build.
- **Evaluation and persistence.** `Latu.ML.evaluate/2` runs an evaluator, and `Latu.ML.save/3`
  and `Latu.ML.load/3` carry a model or an unfitted operator through Spark's own on-disk
  format. All six evaluators are `:probed` as a result. PySpark reads what this writes and the
  reverse; `dev/README.md` has the interop dance.
- **The `ConnectHelper` route.** `Latu.ML.Stat`'s three statistical tests,
  `Latu.ML.assign_clusters/2` and `find_frequent_sequential_patterns/2` for the two operators
  that are neither fitted nor applied, `from_labels/3` and its siblings generated onto their
  model modules, `Latu.ML.Feature.load_default_stop_words/2`, and `Latu.ML.helper/3` under all
  of them.
- **Pipelines and tuning.** `Latu.ML.pipeline/1` builds a pipeline and `Latu.ML.param_grid/1,2`
  builds a grid as data; `cross_validator/1` and `train_validation_split/1` search one. None of
  them has a wire form in any client, so the folds are this package's own, cut with `rand(seed)`
  and a range as Spark does it. A sub-model is released as its metric is read, so a five-by-six
  search holds one cache entry rather than thirty. `collect_sub_models: true` keeps them.
  `larger_better?/1` and `best_index/1` are public. A saved search carries its encoded grid, and
  `sub_models: true` writes the fold models and asks for them back. **A plain load leaves them
  on disk**, which is where this deviates from PySpark.
- **Documentation.** A [quick start](docs/guides/quick-start.md) and a
  [cookbook](docs/guides/cookbook.md), a recipe per model family, both executed by
  `test/integration/guides_test.exs` like the two "coming from" guides. `usage-rules.md` ships
  for agents in the `usage_rules` convention, and `docs/cheatsheet.cheatmd` puts every
  hand-written verb on one page with `test/latu/cheatsheet_test.exs` failing on either kind of
  drift. `CONTRIBUTING.md` carries the extractors, the oracle, the probe and the release
  checklist.
