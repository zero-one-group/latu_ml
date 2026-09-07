# dev

Tooling. None of it ships.

## `check_offline.exs`

    elixir dev/check_offline.exs

What a container with no mix, hex or server can still check: the pure layers — param
translation, literal coercion, the command and relation builders, and the struct-to-proto
mapping the facade shares with the goldens. Compiles them against `standins/protocol.exs`,
which is a set of structs with the right fields and no protobuf behaviour.

Not a substitute for `mix check`. Anything needing Nx (`Latu.ML.Linalg`) or Latu itself
(`Latu.ML`, `Latu.ML.Result`) is out of reach here and covered by `mix test`.

When a field is added to a message the plan layer uses, add it to `standins/protocol.exs` too.

## `extract_ml_operators.py`

    python dev/extract_ml_operators.py            # summarise
    python dev/extract_ml_operators.py --write    # rewrite priv/ml_operators.exs
    python dev/extract_ml_operators.py --check    # exit 1 if that file is out of date

Every operator PySpark exposes, with its JVM class, the model class a fit produces, and its
params — snake_case name, wire name, declared type, default and doc — read off the live
`pyspark.ml` classes. 111 operators, 991 params on 4.2.0, including the two whose one method is
a helper call rather than a fit — `PowerIterationClustering` and `PrefixSpan`, kind `:helper`,
caught by a property (a concrete `JavaParams` operator that is none of the four kinds) rather
than by name. Nothing is transcribed, so a Spark
bump is a re-run and a diff; a public operator that is neither a row nor excluded by name stops
it, which is what makes an upgrade's new estimators impossible to miss.

It never reaches a server, and must not: `SPARK_CONNECT_MODE_ENABLED` is what makes
`_new_java_obj` hand back the class name instead of dialling a JVM, and `StopWordsRemover`'s
`invoke_helper_attr` is stubbed. Two of its params have no default a client can know as a
result; three other markers cover defaults that are not literals. See the script's own header.

## `extract_ml_allowlist.py`

    python dev/extract_ml_allowlist.py --write    # rewrite priv/ml_attributes.exs
    python dev/extract_ml_allowlist.py --scala ~/MLUtils.scala --write     # offline

`ALLOWED_ATTRIBUTES` from the server's own `MLUtils.scala`, fetched by tag, plus what it means
for each concrete class: the server validates a `Fetch` by walking `cls.isInstance(obj)`, so an
object's real allowlist is the union up its hierarchy, and that hierarchy is resolved through
PySpark's mirror of it. 61 entries, 65 classes, 669 attributes on 4.2.0.

**The union is the one claim here a mirror cannot settle** — `dev/probe_ml.exs` is what
confirms it against a server. The rest is read: `:returns` comes from PySpark's own
`try_remote_attribute_relation` decorator and return annotations, `:arity` from its signatures.

It also writes the **helpers** section: the `ConnectHelper` object, which is in that allowlist
and is an attribute of nothing. Its names are the Scala's; its argument types are only at
PySpark's call sites, because a helper has no class and so no signature — so this walks for
`invoke_helper_attr` and `invoke_helper_relation` calls and resolves each argument expression
against the enclosing function's annotations, that class's param table, or an explicit
`(value, DataType)` tuple. The two sources are checked against each other: a method PySpark
calls that the server does not allow, or one it allows that no call site types, stops the run.
`handleOverwrite` is excused by name, because PySpark reaches it through `_call_java`.

Needs a network for the Scala file; `--scala` takes a local copy instead.

## `pyspark_oracle.py`

    export SPARK_REMOTE='sc://localhost:15003'
    python dev/pyspark_oracle.py --generate
    python dev/pyspark_oracle.py --show ml_fit_logistic_regression

Writes `test/wire/<name>.{bin,txtpb}`: the protos PySpark builds for the same operations this
package builds. **Run it twice — the second run must leave the tree clean.** A diff on the
second run means something is non-deterministic, and a map field serialised in process order is
the usual cause.

ML commands are only built on the way to being sent, so this monkeypatches
`client.execute_command` to capture the command and raise. Nothing is sent, and nothing about
how PySpark assembles a command is re-implemented here.

**Latu must be checked out beside this repo**, or `LATU_DEV` set to its `dev/` directory: the
plan-id normaliser is imported from Latu's oracle rather than copied, because it already has a
twin in `Latu.Plan.normalize_ids/1` and a third copy is how the three drift.

It needs PySpark 4.2.0, which Latu's `dev/.venv` has. The **servers** are this repo's own
`docker-compose.yml` — same ports as Latu's, so run one repo's at a time.

**One fixture per operator family** — a `Fit` for a regressor, a clusterer, a feature
estimator, a frequent-pattern miner and a recommender, plus an `Evaluate`. A generated
constructor is only as good as the class name and param spellings the registry handed it, and
those are per operator; the vertical slice only ever proved `LogisticRegression`. The command
is captured before it is sent, so none of them needs a frame of the right shape.

**And one per arm of `Write` and `Read`**: a model is written by reference and an unfitted
operator by class and uid, and a `Read` names a class with no uid at all. `should_overwrite` is
set even when false, because PySpark sets it unconditionally and the field carries explicit
presence — a message that leaves it out is a different message.

### `--interop`

    python dev/pyspark_oracle.py --interop

Not a fixture: this one really does reach the server. Each side fits, saves under
`/tmp/latu_ml_interop/` **on the server**, and reads back what the other left there — a bare
model, a fitted pipeline, and an unfitted pipeline. The Elixir half is
`mix run dev/probe_ml.exs -- --interop`, and each side compares the other's coefficients
against its own fit, so nothing passes between them but the saved files. Run them in this
order:

    python dev/pyspark_oracle.py --interop        # writes PySpark's, finds no Latu model yet
    mix run dev/probe_ml.exs -- --interop         # writes Latu's, reads PySpark's
    python dev/pyspark_oracle.py --interop        # now reads Latu's

Six directories cross over now: a bare model, a fitted pipeline, an unfitted pipeline, a fitted
search, an unfitted search, and PySpark's own of each. A model's is written by the **server**,
so both clients get it right for free; everything else is client-written and is what these two
scripts exist to check.

Both sides must talk to the **same server**, because both the saved models and the fits live
there: `SPARK_REMOTE=sc://localhost:15003` for the oracle is the profile the probe defaults to.
A run that says DIFFER is a format two clients read differently, which is the only thing this
is about.

The pipeline legs matter more than the model one. A model's directory is written by the
**server**, so both clients get it right for free; a pipeline's `metadata` and `stages/NN_uid/`
are laid out by the **client**, in Python on one side and Elixir on the other. Nothing offline
can check that layout, so this is the only place ML5a's claim is tested at all.

## `probe_ml.exs`

    docker compose up -d spark-connect
    mix run dev/probe_ml.exs                 # report, change nothing
    mix run dev/probe_ml.exs -- --write      # rewrite priv/ml_probe.exs, then `mix compile`
    mix run dev/probe_ml.exs -- --recovery   # also watch a dropped summary come back (~80 s)
    mix run dev/probe_ml.exs -- --interop    # also read the model PySpark saved

`probe_dtypes.exs`'s job, for ML: which of the 111 registered operators the server actually
loads, and which of the allowlisted attributes actually answer. The extractors read PySpark and
the server's own Scala, and both can only be wrong in the same way PySpark is; **the server is
the only oracle for behaviour**, and this asks it.

What it passes is what the docs may call supported. `Latu.ML.Registry` reads
`priv/ml_probe.exs`, and every constructor's `@doc` opens with the status found there —
`:probed`, `:built` with the reason kept, or `:missing` where the server would not load the
class at all. **`priv/ml_probe.exs` is the one registry file that may be absent**: without it
every operator is `:built`, which is what a package with no probe result is entitled to claim.

A `:built` reason is far more often the probe's own four-row fixture being the wrong shape for
an operator than the operator being broken; the table at the top of the script is the first
place to look. It validates itself against the registry before reaching for a server, so a
stale entry stops the run rather than turning into a reason that reads like Spark's.

It also runs the whole `ConnectHelper` route — all eleven methods, in their three shapes —
which is the only thing that says the server accepts the arguments PySpark's call sites claim
they take. `handleOverwrite` is the odd one: PySpark reaches it through `_call_java` rather
than a helper call, so it is pinned by hand in the extractor, and a pipeline saved over an
existing directory is what exercises it.

Models are reached through the estimators that fit them, and not run directly. Evaluators are
run: `Latu.ML.evaluate/2` sends an `Evaluate` as of ML4, and the three fixture frames the six
need — scalars, a Vector with a cluster beside it, arrays — are written rather than fitted, so
an evaluator's status says nothing about the classifier that would ordinarily be in front of
it.

It also saves a model, loads it back, and checks that a loaded one has neither its summary nor
a frame to rebuild one from; and it fits a pipeline, checks the fold lands exactly where
fitting the same stages by hand does, and round-trips both a fitted and an unfitted one. With
`--interop` it reads what PySpark saved as well; see `pyspark_oracle.py --interop` above for
the order to run the two in.

Finally it runs a grid search, on **its own twelve-row fixture** rather than the four-row one.
That is deliberate and worth keeping: on four rows every param map scored identically, and
three ties make index 0 the answer whether or not the search works at all. The fixture is a
linear regression scored by `rmse` because `rmse` varies with `reg_param`, because it is the
*minimised* branch of `larger_better?` (the default evaluator only ever exercises the other
one), and the grid is ordered so the winner sits in the middle — a best index of 0 is what a
broken argmin returns too. `the grid actually discriminates` and `the winner is in the middle`
are the two lines that keep it honest. `pyspark_oracle.py --interop` prints the same metrics
from PySpark for the same rows, seed and grid; they should agree to the last digit.

Then it saves that search and reads it back, which is ML5c: a search's directory is laid out
entirely by the client, so `--interop` is the only thing that says whether it is Spark's own
shape. Watch the grid in particular — a wrong `parent` uid loads without complaint and simply
leaves the search with nothing to vary.

## `probe_eviction.exs`

    docker compose up -d spark-connect
    mix run dev/probe_eviction.exs

ML3 reconnaissance, run early because it is cheap and the answers change ML3's design. What the
ML cache does under a budget it cannot meet, and what a client sees — offloaded, gone, or
refused at `fit`.

Runs against `:15002` and sets the memory-control confs to absurd values on **its own session**,
which the server scopes per session; it releases that session at the end. Do not point it at a
server other tests are using.

The first answer it needs is whether the cache re-reads a conf mid-session at all. If it does
not, every later step passes for the wrong reason — the script says so rather than assuming.

Answered 2026-09-05: it does, the confs are session-scoped, and a model is offloaded rather than
evicted. `test/integration/ml_cache_test.exs` holds what was worth keeping.

## `probe_reattach_fit.exs`

    docker compose up -d spark-reattach
    mix run dev/probe_reattach_fit.exs

    ROWS=50000000 MAX_ITER=200 mix run dev/probe_reattach_fit.exs   # if it finishes too fast

A `Fit` long enough that `spark-reattach`'s five-second `senderMaxStreamDuration` cuts the
stream under it, forcing the client to reattach and collect the answer on a second stream. What
is under test is **Latu's** `ml_command_result` latch, which is guarded so a replay cannot
clobber the first result — proved as a state machine in Latu's `execution_test.exs`, never
against a server, because nothing before this package could send an `MlCommand`.

Deliberately a probe and not a test: it costs seconds, the property belongs to Latu, and a gate
that slow would tax every run of a suite that finishes in about one. It prints **INCONCLUSIVE**
rather than passing quietly if the fit finishes inside five seconds.

## `probe_ml_functions.exs`

    docker compose up -d spark-connect
    mix run dev/probe_ml_functions.exs

ML6 reconnaissance. Roadmap §1 carried `vector_to_array` and `array_to_vector` as uncertain
because Latu's `probe_ml_server.exs` asked the server for them in **SQL text** and got
`UNRESOLVED_ROUTINE`. Reading 4.2.0's source says that question could not have been answered any
other way: mllib registers those two and `aggregate_metrics` into `FunctionRegistry.internal`,
which the SQL parser does not search, through a `SparkSessionExtensionsProvider` named in its
`META-INF/services` and loaded off the classpath (SPARK-49086). The Connect planner reaches that
registry only for an `UnresolvedFunction` whose `is_internal` is **unset** — which is exactly
what `pyspark/ml/connect/functions.py` sends, and exactly what `Latu.Plan.fun/3` builds.

So this asks the wire what the SQL route could not: both dtypes, the arity and string-literal
rules the Scala builders enforce, `is_internal` set true and false beside unset,
`aggregate_metrics` (which is `Summarizer`'s whole Connect route, and in no section of the
roadmap), and what the resulting `array<double>` column is worth to `collect/2`, `to_explorer/2`
and `to_arrow/2`.

That last one is the ML6 decision it exists for. If an array column collects, `Latu.to_nx/2` is
an optimisation with a measured cost rather than the only way out of a Vector column, and §2.5
and §4 seam 1 both have to say so.

## `example.exs`

    docker compose up -d spark-connect
    mix run dev/example.exs

    # or, to keep the bindings
    iex -S mix
    iex> import_file "dev/example.exs"

Latu's `dev/example.exs`, for the ML surface: the registry without a server, assemble and fit,
attributes as tensors, a training summary, both ways of reading a `Vector` column, evaluate,
save and load, the bracket, a pipeline, a grid search, one statistical test, and the cache
verbs. A smoke test that is also the fastest way to see what the package does.

It **does not disconnect**, so a REPL keeps a live `session` to poke at — and with it a live ML
cache. Every model it fits is deleted when it is done with; the cache section empties whatever
is left, and `Latu.disconnect(session, release: true)` ends the rest.

The *documented* snippets live in `docs/guides/`, where the fences are executed by
`test/integration/guides_test.exs`. This is not in any gate: it is a script to run by hand.
