# Changelog

`latu_ml` follows [Semantic Versioning](https://semver.org). Before 1.0, a minor version may
rename or remove; each such change is listed here with the migration in one line.

## 0.4.2 — 2026-09-19

Fixes from a fifth review. Three are 0.4.1's shape at sites its sweep missed, one is arithmetic.
No API changes.

**A helper constructor refused an option acquires nothing.** A `from_labels/3` call with an
unknown option raised the right `ArgumentError` after the server had already built the model,
leaving one cached with no handle to delete it by. The options are checked before the send.

**A saved pipeline with a non-string stage uid is refused by name.** A `null` in `stageUids` read
the first stage into the cache and raised building the second's path. It is refused before
anything is read, naming the entry. The shared loader traversal also gives back what it has
collected on a raise, as it did on a returned error.

**A saved search without its metrics is refused by name.** Missing `validationMetrics` (or
`avgMetrics`) read the winner and every sub-model and raised `KeyError` assembling the result,
stranding the sub-models. It is refused before anything is read, and the sub-models are held
through the assembly.

**Fold metrics of any finite magnitude aggregate.** Squaring a deviation of `5.0e159` overflowed a
float, so a cross-validation over metrics around `1.0e160` raised `ArithmeticError` after
refitting the winner and stranded every fitted model. `avg_metrics` and `std_metrics` are now
computed over the metrics divided by a power of two, exact for ordinary values and finite wherever
the answer is (`np.std` answers `inf` there; `docs/deviations.md`), and the search holds what it
fitted through aggregation and assembly.

## 0.4.1 — 2026-09-19

Fixes from a fourth review, all one shape: a model entered the server cache at one step and the
cleanup that knew about it sat at the next step, or one level up. No API changes.

**A raise while fitting a later search candidate strands nothing.** With
`collect_sub_models: true`, a param kind refused at encoding on the second candidate
(`max_iter: [2, "bad"]`) used to leave the first candidate's model cached. The fold's kept
sub-models are released on the way out; the raise is still yours.

**A load that fails after the estimator keeps nothing.** Loading a saved search reads the
estimator first, and a pipeline estimator can carry a fitted stage. A winner whose directory would
not read left that stage cached; now every refusal after the estimator, the winner included, gives
it back.

**A nested pipeline that raises after fitting strands nothing.** An inner pipeline's models exist
before the outer fit has recorded them, so a transformer it carried whose param kind is refused at
encoding (`sql_transformer(statement: 123)`) stranded an inner model. The stage that fitted them
now gives them back.

## 0.4.0 — 2026-09-18

Fixes from a third review. The `latu` floor moves to `~> 0.8`, which is what makes this a minor; one
behaviour changed, with the migration below.

**The `latu` requirement is now `~> 0.8`.** `delete/1` and the search now identify a session by
server, user and id together — `Latu.Session.identity/1`, new in Latu 0.8.0 — so that is the floor.
A two-component `~>` still caps at the next major, so any Latu 0.8+ resolves; if you are below
0.8.0, upgrade it alongside this.

**Tuning no longer deletes a pre-fitted stage you supplied.** Put a model you already fitted into a
pipeline and tune that pipeline, and each discarded candidate used to delete every model in the
fitted pipeline — the stage you passed in included — so the winner's refit failed and your own model
was gone. A search now tracks what each candidate fit apart from what the caller carried in, and
releases only the former: while searching, while unwinding an error, and on the winner's refit.

**A raise during a search strands nothing.** An evaluator that raises rather than returns — a param
kind refused at encoding — used to leak the candidate it had already fitted and leave the fold's two
frames persisted. The fit is released and the frames unpersisted on a raise as on a returned error,
and the original exception is re-raised.

**A non-finite fold metric is a named error, not a crash.** A legitimately non-finite score — R² on
a constant target, which the protocol carries as `:nan` — used to reach fold-metric aggregation and
raise `ArithmeticError`. It is a `Latu.Error` now, and any collected sub-models are released.
Migration: match `{:error, %Latu.Error{}}` from `fit/2` where an `ArithmeticError` would have
escaped before.

**A loaded search owns the estimator it read.** Loading a saved search caches its estimator, its
winner and any sub-models, but `delete/1` and the load-failure paths freed the winner and sub-models
and forgot the estimator's own stages. A search result now records the models it owns — a loaded one
owns everything it read, a fitted one owns its acquisitions but not a stage you carried in — and
`delete/1` frees exactly those.

**`delete/1` batches by full session, not the id alone.** 0.3.1 sent each reference to the session
that made it but grouped by `session_id`, so two servers handed the same explicit id were batched
onto one connection and the second's model survived. It groups by full session identity now — the
ML side of the same fix in Latu 0.8.0's two-input verbs.

## 0.3.1 — 2026-09-18

Bug fixes from a follow-up review. No API changes.

**A failed pipeline fit releases only what it fitted.** A pipeline may carry a model the caller
already fitted, and a later stage failing used to delete it. Each stage's newly cached models are
now tracked apart from carried ones, and only the former are released, on a returned error and on a
raise alike.

**Cross-validation with a string `fold_col` works.** The fold column name was compared as a string
literal, so a `fold_col` like `"fold"` made Spark try to cast the word to a number. It is a column
reference now. The k-fold and `TrainValidationSplit` random draw column is a binary rather than a
fresh atom per validator, so building validators no longer grows the atom table.

**`delete/1` deletes across sessions.** It sent every reference through the first model's session
and reported success, so models from other sessions survived. It now groups references by session
and sends one delete to each.

**A Bucketizer's `splits_array` loads.** The nested `array<array<double>>` param saved, but the
loaded-parameter decoder knew only scalar element types; it recurses now.

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
