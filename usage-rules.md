# Using Latu ML

`latu_ml` is Spark MLlib over Spark Connect: the algorithms run on the cluster and the Elixir
side holds plans and handles. This file is the short set of rules that are not guessable from
the function names. It follows the `usage_rules` convention, so a consuming project can sync it
into an agent's context.

**Latu's own [`usage-rules.md`](https://hexdocs.pm/latu/usage-rules.html) applies here in full**
— coercion, the `!` twins, errors, session config. What follows is the delta. Every deliberate
departure from `pyspark.ml` is in `docs/deviations.md`, with why.

## The three shapes

Latu has two; this has three, and the third is the one that surprises.

**A builder takes a struct and returns a struct.** Pure, no IO. `Latu.ML.pipeline/1`,
`Latu.ML.param_grid/2`, `Latu.ML.cross_validator/1`, and every generated constructor —
`Latu.ML.Classification.logistic_regression/1` and its sixty-seven siblings.

**An action runs something** and returns `{:ok, value} | {:error, %Latu.Error{}}`, with a `!`
twin. `Latu.ML.fit/2`, `Latu.ML.evaluate/2`, `Latu.ML.attribute/2`, `Latu.ML.save/3`,
`Latu.ML.load/4`, `Latu.ML.delete/1`, `Latu.ML.model_size/1`, `Latu.ML.cache_info/1`,
`Latu.ML.clean_cache/1`.

**A relation builder returns a `Latu.DataFrame` and reaches no server.** `Latu.ML.transform/2`
is one — applying a transformer or a fitted model builds a plan, nothing more, and
`Latu.schema/1` answers on it without executing. So is `Latu.ML.attribute_frame/2` and every
generated accessor for a DataFrame-valued attribute, and so are `Latu.ML.Stat`'s three tests and
`Latu.ML.assign_clusters/2`.

    features = Latu.ML.transform(assembler, training)   # nothing has run
    {:ok, model} = Latu.ML.fit(lr, features)            # one round trip
    scored = Latu.ML.transform(model, features)         # still nothing

## A model is a server-side resource, and it is yours to release

`Latu.ML.fit/2` hands back a `Latu.ML.Model`: a reference into a per-session cache, not a value.
The BEAM has no finalizer, so nothing here can free it for you.

    # the bracket — deletes in an `after`, so a raise in the body still releases
    {:ok, auc} =
      Latu.ML.with_model(lr, training, fn model ->
        Latu.ML.evaluate(evaluator, Latu.ML.transform(model, test))
      end)

    # the pair, for when the handle outlives the expression
    {:ok, model} = Latu.ML.fit(lr, training)
    :ok = Latu.ML.delete(model)

`Latu.ML.delete/1` takes a list, and a mixed one. A `Latu.ML.PipelineModel` or a search result
owns the models inside it, so deleting one empties the tree in a single `Delete`.

Two things that also spend cache, and are easy to miss: `Latu.ML.load/4` on a model registers a
**fresh** entry, exactly as a fit does; and a `trees` accessor answers with one cache entry per
tree. `Latu.ML.cache_info/1` is how you find out what you are holding.

What you do **not** have to guard against is eviction. A model is offloaded to disk under
memory pressure and comes back transparently; over budget it is the *fit* that is refused
rather than something already held being dropped.

## Params: snake_case, only what you set, kind not range

A constructor takes a keyword list in Elixir's spelling — `max_iter:` is `maxIter` on the wire.
Only the params you set are sent; the server fills in every default. **A documented default is
never sent**, because a param you did not set and a param sent with its default value are
different requests.

Client-side validation refuses the **kind** and never the range. `reg_param: :bad` is this
package's error; `reg_param: -1.0` is Spark's own `ParamValidators`' error, and it is a better
one. A column is an atom, as everywhere in Latu.

`Latu.ML.params/1` prints the table for an operator, with each param's type and default.

## Attributes: the server has an allowlist, and there are two verbs

`MLUtils` on the server holds a list of the methods a fetch may invoke on each model and summary
class; anything else is `CONNECT_ML.ATTRIBUTE_NOT_ALLOWED`. So the set is not "whatever PySpark
exposes", and `Latu.ML.attributes/1` is what it actually is.

**Reach for the generated accessor first.** There is one function per allowlisted attribute, on
a module per class — `Latu.ML.Regression.LinearRegressionModel.coefficients/1` — carrying the
class, the wire name and Spark's own docs. It checks the class before the round trip; the
generic verbs underneath do not.

Which generic verb depends on how the attribute answers, and they refuse each other's names:

  * `Latu.ML.attribute/2` for a value — `{:ok, term}`, an action. A `Vector` or `Matrix` comes
    back as an `Nx.Tensor`, or a `Latu.ML.SparseVector` where densifying would be this
    package's decision rather than yours. A `Matrix` is always row-major.
  * `Latu.ML.attribute_frame/2` for a DataFrame-valued one — a lazy `Latu.DataFrame`.

Either verb takes either spelling: the snake_case name `Latu.ML.attributes/1` advertises, or the
camelCase one the allowlist holds.

## Summaries

Ten of the 43 model classes record a training summary. `Latu.ML.summary/1` is a **lazy
builder** — Spark reaches a summary through the model that owns it, so the reference is composed
locally and nothing is sent until you ask it for something. It raises for a class that has none,
naming the ten that do.

The cache **drops** a summary rather than offloading it, so a fetch can answer
`CONNECT_ML.MODEL_SUMMARY_LOST` through no fault of yours. `Latu.ML.attribute/2` recovers from
that on its own, from the frame the model was fitted on. **Except for a loaded model**, which
never saw the data: there the server's refusal reaches you, and `Latu.ML.hint/1` says why.

## A `Vector` column cannot be collected

On Spark 4.2.0 the server describes `features`, `rawPrediction` and `probability` as UDTs with
no SQL type, so Latu's dtype guard refuses them rather than guessing a layout. This is how
Connect describes the type, not something a fit does — a column you build yourself with
`Latu.ML.Functions.array_to_vector/1` is refused the same way.

Three routes through, and the destination picks one:

  * **`Latu.to_nx/2`** reads the Arrow bytes, where the type *is* stated, and gives one
    `{rows, width}` `f64` tensor. Use it when the destination is a tensor.
  * **`Latu.ML.Functions.vector_to_array/2`** converts server-side, so the column arrives as an
    ordinary `array<double>` any reader takes. Use it when the destination is an
    `Explorer.DataFrame`.
  * **`select` or `drop` the column**, which is often the answer — a `prediction` column is a
    double and was never the problem.

## Pipelines and tuning are client code

Spark exposes no `Fit` for `Pipeline`, `CrossValidator` or `TrainValidationSplit`, so both
PySpark and this package loop over the stages themselves. Which means **the wire is not what has
to match — the on-disk format is**.

Two consequences worth knowing before you build one. A `Latu.ML.param_grid/2` entry carries the
**uid** of the operator it sets, because the thing searched is usually a pipeline and a param
belongs to one stage of it; a grid whose params reach nothing is refused before the first fit.
And a search is only repeatable with a `seed:` — folds are cut with `rand(seed)` and range
filters, and without one the draw differs between runs.

`Latu.ML.larger_better?/1` is what turns a metric into an argmax or an argmin. It is pure: the
server's allowlist has no `isLargerBetter`, so the registry carries PySpark's own client-side
overrides.

## Persistence: the path is the server's

`Latu.ML.save/3` writes where the **session** is, so a bare path is the driver's disk. Anything
a second machine has to read wants a URL the cluster's filesystem understands.

`Latu.ML.load/4` **names** what is at the path rather than discovering it — a `Read` has to say
which class to load before the server will look. Three spellings: an operator name from the
registry, a generated model module, or `{class, kind}` for a class this package does not know.

A bare model's directory is written entirely by the server, so any Spark client reads it. A
**pipeline's** wrapper is written by the client in PySpark's layout, which Scala's reader
refuses — as it refuses PySpark's own, for the same one key.

## Errors

A `%Latu.Error{}` carries Spark's own `error_class`. **Match on it, never on the message**:
`CONNECT_ML.CACHE_INVALID`'s text calls the missing object a "Summary object" and blames a
15-minute idle eviction, which is wrong for a model you deleted a moment ago.

`Latu.ML.error_kind/1` maps the five `CONNECT_ML` classes to atoms, and `Latu.ML.hint/1` turns
one into a sentence naming the fix.

## The registry is the reference

Everything the package knows about operators, params and attributes is data, queryable without a
server:

    Latu.ML.operators(kind: :estimator, group: :classification, status: :probed)
    Latu.ML.operator(:logistic_regression)
    Latu.ML.params(:logistic_regression)
    Latu.ML.attributes(Latu.ML.Classification.LogisticRegressionModel)

`status:` is `:probed` for an operator a live server has actually run with every allowlisted
attribute answered, `:built` for one generated from PySpark's param table and not yet exercised,
`:missing` for one PySpark names and the server does not load. Every constructor's `@doc` opens
with its status, so `h` tells you what is verified rather than what is claimed.

An unknown filter key raises rather than matching nothing. A model has no constructor to look
up, because a model exists only by a fit or a read.

## Things this deliberately does not do

  * **No `predict/2` on tensors.** MLlib is DataFrame in, DataFrame out, with a `features`
    `Vector` column. `Latu.ML.transform/2` is the scoring path; the per-row `predict` attribute
    is a round trip a row and is not one.
  * **No building a model from parameters you hold.** There is no wire path — a model exists
    only by `Latu.ML.fit/2` or `Latu.ML.load/4`. Nothing goes the other way, including from
    Scholar.
  * **No closures on the cluster.** `predict_batch_udf` and `xgboost.spark` ship Python to
    Python workers; there is no equivalent for the BEAM.
  * **No parallel grid search yet.** PySpark runs the grid on a thread pool; whether several
    processes can drive one session's channel at once is unmeasured here, so the option is not
    offered rather than offered untested.
  * **No process, and no supervision tree entry**, exactly as Latu promises. The one thing that
    outlives a call is the server-side cache entry, which is why `Latu.ML.delete/1` exists.
