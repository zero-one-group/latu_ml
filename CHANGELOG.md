# Changelog

`latu_ml` follows [Semantic Versioning](https://semver.org). Before 1.0, a minor version may
rename or remove; each such change is listed here with the migration in one line.

## 0.3.0 — 2026-09-11

**`Summarizer`.** `Latu.ML.Stat.summary/3` is `aggregate_metrics` over a `Vector` column with
the metrics as a list and the weight as an option; `Latu.ML.Stat.mean/2` and nine siblings read
one metric out. Roadmap D14, open since `aggregate_metrics` was found to resolve: what it waited
for was `Latu.Column.get_field/2`, which is Latu 0.7.0.

**The `latu` requirement is `~> 0.7`**, the Latu this was built against and the one `Summarizer`
needs. **Elixir 1.18 is the floor**, down from 1.20, as Latu's is; CI compiles and tests the
floor as its own job.

**`Latu.ML` is 600 lines shorter.** `save/3` and `load/4` and everything under them moved to a
module of their own, and the five helpers that decide who owns a cached model to another; the
facade keeps the verbs and the docstrings. No change to what is called or what comes back.

**An error the status cut at 2048 characters arrives whole**, and an older Spark's two refusals
of a 4.2 client name their cause. Both are Latu 0.7.0's and apply here through it.

## 0.2.0 — 2026-09-08

No API change. One dependency change, which is what makes this a minor.

**The `latu` requirement is now `~> 0.4`**, raised from `~> 0.3`. Nothing here needs a Latu 0.4
feature. What the floor buys is honesty: it names the Latu this package is developed and tested
against rather than the oldest one that happens to work. Note what it does **not** buy. A
two-component `~>` caps at the next major, so `~> 0.4` still resolves any Latu below 1.0.0,
exactly as `~> 0.3` did. Capping a minor would need `~> 0.4.0`, and that is deliberately not
what is here: it would mean a release of this package for every Latu minor. If you are below
Latu 0.4.0, upgrade it alongside this.

**Three docstrings were unreadable on hexdocs.** `Latu.ML.attribute/2`, `summary/1` and
`operator/1` each carried a duplicated first line with a stray heredoc opener between the two
copies. It is a legal escape inside a docstring, so it compiled, formatted and passed
`mix docs --warnings-as-errors` without complaint.

**`Latu.ML.load/4` could orphan a cache entry.** Loading a search reads its estimator before its
evaluator, and a failure on the second left the first in the server's cache with nothing owning
it. Fixing that turned up the larger half. The code that decides what to give back had no
clause for an *unfitted* pipeline, whose stages may include a model that is already fitted, so
such a pipeline was invisible to every release path and not only to this one.

**The documentation was rewritten.** The README, the four guides and `usage-rules.md` are the
same facts in a different voice, about a fifth shorter. The README now says on its first screen
what shape MLlib is: DataFrame in, DataFrame out, with no `fit(x, y)` on tensors.

Internals, with no surface change: the load and save folds share two helpers, so the rule that
an error path gives back what it cached lives in one place instead of five; a failed second
persist in a grid search no longer leaves the first one persisted; and the save layout's three
parallel tables over the same six kinds became one.

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
