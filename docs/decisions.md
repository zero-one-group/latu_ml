# Decisions

Why `latu_ml` is the way it is. One entry per decision, with its reason — not the story of
reaching it, which is the commit. A reversed decision is a new entry plus one line on the old
one. This file does not ship: it is not an ExDoc extra and not in the Hex package, so a `@doc`
carries its own one-sentence reason and may point here as a tail.

Latu's own `docs/decisions.md` holds everything this package inherits. Only the delta is here.

## 2026-09-05 — A transformer is its own struct, not an estimator with a flag

`latu-ml-roadmap.md` §3.2 listed `Estimator`, `Model`, `Summary`, `Evaluator`, `Pipeline` and
`Tuning`, and no transformer — but `VectorAssembler` is one, and `fit/2` must refuse it.
Folding it into `Estimator` with a `kind` field would make `Estimator.new/2`, the documented
escape hatch for third-party JVM classes, mean something other than what it says.

So: `%Latu.ML.Transformer{}` beside `%Latu.ML.Estimator{}`, same shape, and `transform/2` takes
a `Transformer` or a `Model`.

## 2026-09-05 — Params coerce twice: declared kind, then magnitude

`MlParams` is untyped on the wire, so the registry is the only thing between `reg_param: 1` and
a server error. PySpark does it in two stages and so does this package, because the server was
built against PySpark's output:

1. The param's **declared type** picks the kind — `TypeConverters.toFloat` turns `1` into `1.0`
   before it is ever a literal.
2. **Magnitude** then picks the wire type, exactly as `LiteralExpression._from_value` does: an
   integer becomes `integer` or `long` by size, a float always `double`. `:int` and `:long` are
   therefore one path for a scalar.

Inside an array the declared element type wins instead: one array literal carries one element
type, so two values either side of 32 bits cannot disagree.

This is why `Latu.ML.Plan.literal/2` exists rather than reusing `Latu.Plan.lit/1`, which returns
an `Expression` (not the bare `Literal` a param needs) and infers from the Elixir value alone —
it would send `{:integer, 1}` where the server wants a double. Not a gap in Latu.

Client-side validation refuses the **kind** and never the range: `reg_param: -1.0` is Spark's
`ParamValidators` to refuse, with a better message than this package could write.

## 2026-09-05 — Write the deprecated type field, not the new one

Spark 4.1 deprecated `Literal.Array.element_type` and `Literal.Struct.struct_type` in favour of
`Literal.data_type`. PySpark still writes the deprecated field and leaves `data_type` unset, and
the server reads what PySpark writes — so this package does the same, for arrays as well as for
the UDT structs in `Latu.ML.Linalg`. One rule, not two special cases.

Setting `data_type` **as well** is accepted: the M15 probe hand-built an `inputCols` literal that
way and `VectorAssembler` resolved. It is still wrong here, because byte-identical to PySpark is
the property the goldens exist to hold, and a field PySpark does not send is a difference nobody
would have noticed until a server stopped tolerating it.

Found by `ml_fit_logistic_regression` and `ml_transform_assembler` on their first run, together
with the other half of the same fix: a string element type carries `collation: "UTF8_BINARY"`,
PySpark's `StringType()` default, where an unset field would have left `""`.

## 2026-09-05 — `Latu.ML.Result` is a third module allowed to name the protos

The roadmap allowed `Plan` and `Linalg`. Reading `ml_command_result` is decoding, not building;
putting it in `Plan` would make a pure builder read results, which is the split Latu draws
between `Latu.Plan` and `Latu.Result`. The layering test's allowlist is `{plan,linalg,result}.ex`.

## 2026-09-05 — Public specs name `Latu.ML.Plan`'s type aliases, not the protos

`@type command :: Proto.Plan.t()` and four siblings. Latu's `Latu.Plan` does the same. Two
reasons, and the second is the load-bearing one: a caller reading `command()` learns more than
one reading `Proto.Plan.t()`; and the generated modules are `@moduledoc false`, so a spec naming
one directly is a link ExDoc cannot resolve and warns about once per spec. With the aliases,
`Latu.ML.Plan` is the only module `mix.exs` has to exempt.

## 2026-09-05 — The struct-to-proto mapping lives in `Latu.ML.Internal`

A `Fit` is a command, so unlike a relation there is no lazy form a test can inspect without
sending it. If `Latu.ML.fit/2` built its own plan inline, a golden test could only re-state that
construction — and would then pass whatever the facade did. `Internal.fit_plan/2` and its three
siblings are what the facade calls and what the goldens assert on, so the two cannot drift.

`Internal` is `@moduledoc false` and names no protocol type: its specs use `Latu.ML.Plan`'s
aliases, which keeps the layering rule intact.

## 2026-09-05 — The bracket ships with the first release, not with ML3

`with_model/3` was scheduled for ML3. A model is a server-side resource with no finalizer behind
it, and shipping `fit/2` without the bracket would mean the first example anyone copies leaks
one. Moved forward.

## 2026-09-05 — The oracle intercepts rather than re-implements

Latu's oracle evaluates an expression and reads its plan. ML has no expressions: PySpark only
builds a `Fit` on its way to sending it. So `dev/pyspark_oracle.py` monkeypatches
`client.execute_command` to capture the command and raise, then calls PySpark's own
`estimator.fit(df)`. Nothing about how PySpark assembles a command is re-implemented here, which
is the whole point of having an oracle.

`Latu.ML.Linalg`'s layouts were read off `pyspark/ml/connect/serialize.py` rather than guessed —
the deprecated `struct_type` field that PySpark still writes and the server still reads,
`udt.type = "udt"`, the type flag as a **byte** literal, and the null placeholders at elements
1–2 of a Vector and 3–4 of a Matrix.

## 2026-09-05 — The normalisers are imported from Latu's oracle, not copied

`normalize_plan_ids` already has a twin in `Latu.Plan.normalize_ids/1`, and Latu's own progress
notes list those two as an open thread. A third implementation of one rule is how they drift, so
`dev/pyspark_oracle.py` loads Latu's by path. Cost: Latu must be checked out beside this repo,
or `LATU_DEV` set. Stated in the script's docstring and in `dev/README.md`.

## 2026-09-05 — Fixtures pin every uid

PySpark generates a random uid per operator and so does this package. A golden cannot have
either, so `dev/pyspark_oracle.py` and `test/support/wire.exs` share three pinned constants and
the tests override the generated uid. Nothing about uid generation is under test as a result;
`dev/check_offline.exs` covers its shape instead.

## 2026-09-05 — The registries are read, never transcribed, and the reader fails loudly

The objection that kept MLlib out of Latu was that a hundred operators' params would be
hand-transcribed and nothing would notice drift. `dev/extract_ml_operators.py` and
`dev/extract_ml_allowlist.py` are the answer, and both hold `extract_functions.py`'s rule: a
public operator class that is neither a row nor excluded **by name** stops the run, as does an
unknown type converter, a param name two wire names both spell, and a `classOf` entry the Scala
parser could not read. Silence is not an outcome either script can produce.

`priv/ml_operators.exs` and `priv/ml_attributes.exs` are committed rather than generated at
build time: the package must compile without PySpark, a network, or a server, and the diff at a
re-vendor is the drift check.

## 2026-09-05 — A default is documentation, and four of them are not literals

Defaults never reach the wire (`Latu.ML.Plan.params/1`), so the only thing asked of the value in
the registry is that it read true and not move between runs. Four do not survive that:

- `{:uid_suffix, "__output"}` — 54 output columns default to the operator's own random uid.
- `:server_supplied` — `StopWordsRemover`'s locale and stop-word list; its `__init__` asks a
  server for both, which is why `invoke_helper_attr` is stubbed at all.
- `:varies` — 33 `seed`s. PySpark's default is `hash(type(self).__name__)` and Python salts
  string hashing per process, so it is a different number every run. The server's own default
  is the JVM hash of the *Scala* class name, a third number, so there is nothing true to write.
- `:nan` — `Imputer.missingValue`, which has no Elixir float literal.

The extractor finds the last two by property rather than by name: a default that differs between
two instances, or that equals `hash(class name)`.

## 2026-09-05 — An attribute answers in one of three shapes, not two

The roadmap split attribute access two ways — `attribute/2` for a literal, `attribute_frame/2`
for a DataFrame — off a relation-valued flag. There is a third: `summary` and `evaluate` answer
with **another cached object**, on `MlCommandResult`'s `summary` arm, and the client turns its
ref into `"<model_ref>.summary"` and fetches through it. 17 of 669 attributes are this shape.

So `:returns` is `:value | :frame | :summary`, and each model row carries the summary class its
`summary` yields. Two models — `LogisticRegressionModel` and `RandomForestClassificationModel` —
decide that class from the fitted model's `numClasses`; ~~recorded `:model_dependent`~~ recorded
as both candidates, for the reason in the 2026-09-06 entry below. The summary path is ML3's; the
registry stops at naming it.

## 2026-09-05 — Generated files are emitted already formatted

Both extractors render what `mix format` would render, down to the 98-column wrap, the digit
grouping in long integers and the trailing comma's width. `mix check` runs
`format --check-formatted`, so a generator that needs a formatting pass afterwards makes every
regeneration a two-step dance with a diff in between — and the second `--check` run, the one
that proves the extractor is deterministic, would fail for a reason that is not drift.

## 2026-09-06 — `fpm` is `Latu.ML.FPM`, and it is the only group module that is a decision

Six of the seven generated group modules are PySpark's own module name in CamelCase.
`pyspark.ml.fpm` is not a word: `Latu.ML.Fpm` reads like a typo, `Latu.ML.FrequentPatterns`
invents a name Spark does not use, and `Latu.ML.Association` names the output rather than the
algorithm. So the abbreviation stays an abbreviation, the way `SQLTransformer` does. Elixir is
fine with an all-caps segment; it is only unusual to look at, and it is unusual in Spark too.

Closes D11's sibling, D12 in `latu-ml-roadmap.md`.

## 2026-09-06 — An accessor is generated only where nothing has to be guessed

144 of the 669 allowlisted attributes get a generated function. The other 525 are not omissions
of taste, and each has a different reason:

- **They take arguments** (`predict`, `evaluate`, `approxNearestNeighbors`). A `Fetch` method
  can carry literals and relations, but nothing in this package builds one yet, and an
  `attribute/3` that types its arguments is ML3's design rather than a line of codegen.
- **They answer with a summary** — 17 of them, on `MlCommandResult`'s third arm. There is no
  `%Latu.ML.Summary{}` yet, so an accessor would return a bare ref whose type changes in ML3.
- **They belong to a summary class** — 22 classes and 370 attributes. Same reason from the other
  end: a summary cannot be obtained, so a module of accessors for one is dead code.
- **`toString`**, which the allowlist grants on all 65 classes through `Identifiable`. Defining
  it needs `import Kernel, except: [to_string: 1]` in every generated module, and it tells the
  caller less than the uid the model already carries.

All of them stay reachable through `Latu.ML.attribute/2` where they take no arguments, and each
generated module's `@moduledoc` says which of its class's attributes are missing and why.

## 2026-09-06 — `attribute_frame/2` ships with the accessors, not with ML3

Seven of the 144 generated accessors answer with a DataFrame — `itemFactors`, `surrogateDF`,
`gaussiansDF` and their kind. The registry already knows which, from PySpark's own
`try_remote_attribute_relation` decorator, and `Latu.ML.Plan.fetch_relation/3` has been there
since ML1. Generating the value-valued half and leaving the frame-valued half unreachable would
have been a surface with a hole in it for one milestone, for no gain.

`model_summary_dataset` stays nil on those relations: it exists to rebuild an evicted *summary*
from its training frame, and a model is never evicted.

## 2026-09-06 — `Plan.evaluate/3` ships with the goldens, `Latu.ML.evaluate/2` does not

ML2 generates six evaluator constructors, and a constructor whose wire form nothing checks is a
constructor nobody has read. `MlCommand.Evaluate` is `MlCommand.Fit` with one field renamed, so
the plan builder is six lines and `ml_evaluate_regression` pins it against PySpark's own bytes.

The verb stays ML4's, where the *result* decoding lives. What ML4 inherits is a wire form that
has already been diffed rather than one to get right at the same time as the answer.

## 2026-09-06 — Generated docs are folded here, not written pre-folded

Every `@doc` and `@moduledoc` this package generates is built as one long string and wrapped to
92 columns by `Latu.ML.Generate.wrap/1`. The alternative — heredocs in the generator with the
line breaks written in — folds on the *generator's* line lengths rather than the generated
text's, which is how `h Latu.ML.Feature.pca` ends up ragged. Param bullet lists are left
unwrapped: a folded bullet needs a continuation indent, and a long param doc reads better on one
line than badly broken.

## 2026-09-06 — `priv/` ships in the Hex package

The registries are read at compile time, so a package without them does not build. Added to
`mix.exs`'s `files`; the two files are 480 KB together, which is the cost of not transcribing
981 params by hand.

## 2026-09-06 — `priv/ml_probe.exs` is optional, and the other two are not

The operator and attribute registries are read off PySpark and Spark's own Scala: they are
always available, always the same, and the package cannot build without them. The probe result
is what a *server* said, and no server is guaranteed to be there. So `Latu.ML.Registry` reads
it when it exists and shrugs when it does not, and everything is `:built` — which is precisely
the claim a package with no probe result has earned.

The reverse would be worse in both directions: a required probe file makes a checkout unbuildable
without Docker, and a probe result baked into the extractors' output would mix "what PySpark
says" with "what a server did" in one file, which is the distinction the whole milestone rests
on.

## 2026-09-06 — A `:built` reason goes in the operator's own docs, sanitised

The status line is generated per constructor, so `h Latu.ML.Feature.min_hash_lsh` says what the
probe saw rather than sending the reader to a table. Two consequences worth writing down.

The reason is a *server's* error text landing in Markdown that ExDoc builds with
`warnings_as_errors`, so backticks in it are replaced before it goes in — an unbalanced code
span in a generated doc would fail the build for a reason nobody could see.

And the line says what a reason usually means: the probe's four-row fixture being the wrong
shape for that operator, not the operator being broken. A generated doc that overstated a
refusal would be worse than no status line at all.

## 2026-09-06 — There are four answer shapes, and the fourth was measured, not read

`dev/probe_ml.exs`'s first run fitted 51 of 60 operators and found two things no amount of
reading PySpark would have.

**`TreeEnsembleModel.trees` answers on `operator_info`**, not on `param`: the server hands back
references to the sub-models, which PySpark splits on a comma and wraps. So `:returns` gains
`:operators` beside `:value`, `:frame` and `:summary` — four attributes on 4.2.0, all of them
`trees`. `dev/extract_ml_allowlist.py` reads it from PySpark's own return annotation
(`List[SomethingModel]`) now that the probe has said what to look for, so it stays extracted
rather than pinned; the generator leaves those four alone, for the same reason it leaves
summaries alone — there is no struct to hold one.

**`Latu.ML.Linalg` could not read an empty Matrix.** `NaiveBayesModel.sigma` is 0x0 on a
multinomial model, and `Nx.tensor([])` refuses to guess a shape from nothing. Nor can it be
given one: `Nx` has no zero-sized dimension at all. So an empty Vector or Matrix decodes to
`nil` — an answer, not a failure — with a doctest pinning it and a line in `docs/deviations.md`.

Both are the argument for the probe existing, and neither was a guess that could have been
checked offline.

## 2026-09-06 — The probe's fixture casts every numeric column

`VALUES (0.0)` is a `DECIMAL(2,1)`. `VectorAssembler` and every predictor cast it silently on
the way in, so 51 operators passed straight over it; `Binarizer`, which reads a raw column,
answered "Data type DecimalType(2,1) is not supported". A fixture whose types are nearly right
is worse than one that is obviously wrong, because it fails for nine operators and passes for
fifty-one. Everything numeric is cast explicitly now.

## 2026-09-06 — A summary keeps the model's reference, not a dotted string

Spark has no cache key for a training summary: it is reached by asking the model for one, so a
`Fetch` names the model and puts `summary` at the head of its method chain. PySpark carries that
as the string `"<model_ref>.summary"` in the slot where a model keeps its `RemoteModelRef`, and
`_extract_id_methods` splits it apart again on the way to the wire.

That string never reaches the wire, and Elixir has a struct where PySpark has one slot, so
`%Latu.ML.Summary{}` keeps `ref` and `methods` apart. Same bytes, one fewer parse, and no way
to build a reference by concatenation that the splitter then has to trust.

## 2026-09-06 — Which frame rebuilds a summary is per model class, and extracted

`CreateSummary` takes the frame the model was fitted on — except that
`HasTrainingSummary._summary_dataset` defaults to `self.transform(train_dataset)` and **seven of
the ten** models that have a summary override it to return the raw frame instead. Only the three
clustering models use the default.

Sending the wrong one would fail nowhere except under eviction, which is the hardest kind of bug
to see, so `dev/extract_ml_allowlist.py` reads the override off each class and refuses to write
if any override's body is not the single line it expects today. `priv/ml_attributes.exs` records
it as `:summary_dataset`.

## 2026-09-06 — `summary` gets a generated accessor and `evaluate` does not

Both answer with a summary, and the generator excludes `:summary`-shaped attributes. `summary`
is the exception because **nothing is sent**: the reference is composed from the model's own, so
`Latu.ML.summary/1` is a lazy builder and the generated `summary/1` is a class check in front of
it. `evaluate` takes a dataset, so it waits with the other 21 argument-taking attributes.

## 2026-09-06 — `MODEL_SUMMARY_LOST` is the one error this package handles rather than reports

Latu's rule is that a server's refusal reaches the caller: it has the better message, and
swallowing it hides what happened. A dropped summary is the exception. It is not the caller's
doing, not something they can prevent, and the recovery is exactly what PySpark does — send
`CreateSummary` with the frame the struct carries, and ask once more.

A fifth `CONNECT_ML` class, incidentally: ML1 found four, and this is not one of them.

Only the *command* arm needs it. A summary's frame-valued attributes ride a relation carrying
`model_summary_dataset`, and the server rebuilds from that while it plans.

## 2026-09-06 — `error_class` carries the dotted name, and the probe is why we know

`%Latu.Error{}`'s `error_class` for a lost summary is `"CONNECT_ML.MODEL_SUMMARY_LOST"`, not
`"MODEL_SUMMARY_LOST"`. This document, `latu-ml-roadmap.md` and Spark's own docs all write these
classes bare — `CACHE_INVALID`, `ATTRIBUTE_NOT_ALLOWED` — so a matcher written from any of them
would silently never fire. `dev/probe_ml.exs` prints it; that is where the spelling came from,
and ML3's error mapping should read it off a run rather than off prose.

## 2026-09-06 — `offloadingTimeout` is in minutes, and it is read once

~~An earlier entry here blamed a race between `CreateSummary` and the retry.~~ Wrong; the real
reason was read off `MLCache.scala` and `Connect.scala` after a second run, and it is worth two
paragraphs because it corrects something already written down.

**The conf is `.timeConf(TimeUnit.MINUTES)`.** `"1s"` does not mean one second — it truncates to
**0**, and `MLCache` passes that straight to Guava as `expireAfterAccess(0, MINUTES)`. Nothing is
ever held in memory. Every command reloads the model from disk, and a model's save format does
not carry its summary (`MLCache` writes that to a sibling path), so `hasSummary` is false again
before the next command arrives. No `CreateSummary` can outlive that, and PySpark's recovery
would fail exactly the same way.

**And `cachedModel` is built once**, a `val` whose `expireAfterAccess` and `maximumWeight` come
from `offloadingTimeout` and `maxInMemorySize` at construction. Changing either mid-session does
nothing. **This corrects ML1's finding** that the memory controls are read live: that is true of
`maxModelSize` and `maxStorageSize`, which `checkModelSize` reads per call, and false of the
other two. ML1's eviction probe only appeared to set `maxInMemorySize` because it did so on a
fresh session, before the cache existed.

`dev/probe_ml.exs -- --recovery` therefore uses one minute — the smallest value that is not
degenerate — and waits it out. The same run measures the other half of ML1's eviction finding
from the summary's side: the offload that costs a *model* nothing (25 ms cold) costs its
*summary* everything.

## 2026-09-06 — The three cache verbs take a session, except the one that takes a model

`GetModelSize` names a model, so `model_size/1` takes one. `GetCacheInfo` and `CleanCache` name
nothing — the cache is the session's — so `cache_info/1` and `clean_cache/1` take a session, and
that is the first place in this package where a verb does. It reads oddly beside `fit/2` and
`attribute/2`, which get their session from the frame or the model, and the alternative reads
worse: passing a model to ask what *else* the cache holds implies the answer is about that model.

`clean_cache/1` is documented as the blunt instrument it is. `MLCache.clear` empties the map and
the offload directory together, so every `%Latu.ML.Model{}` in the session becomes a reference to
nothing at once. `delete/1` and `with_model/3` stay the ordinary way to give a model back.

## 2026-09-06 — `cache_info/1` decodes the server's JSON rather than handing it over

`MLCache.getInfo` renders each entry as a JSON string — `{"id": ..., "class": ..., "size": ...}` —
and the wire carries an array of them. Handing a caller a list of strings to parse would be the
easy thing and the wrong one: the shape is fixed by the server, so the parse belongs here, once,
where a shape that changes becomes `{:error, %Latu.Error{kind: :protocol}}` instead of a surprise
in someone's pipeline. Elixir's own `JSON` does it, so this costs no dependency.

`size` is `0` for a summary always, and `0` for a model where the session has memory control off:
`MLCache` only measures what it is going to budget against. Documented rather than hidden, because
a caller comparing sizes needs to know which zeros are real.

## 2026-09-06 — ML1's `GetCacheInfo` `[]` was an empty cache, and the summary path is closed

Both measured by `dev/probe_ml.exs`, and both replace a question this document was holding open.

**`GetCacheInfo` works.** Asked on the session that holds the model, it answers with one entry
whose `size` is the same byte count `model_size/1` reports. So the empty list ML1 recorded was an
empty cache — a different session, or no fit yet — and not a defect in the walk or in the decode.
The probe prints the raw `MlCommandResult` beside the decoded list so that a future `[]` can be
told apart from a decode that lost its elements without a second reading session.

**A summary costs no cache entry.** `cache_info/1` does not grow after one is read. `Fetch`
registers a `Summary` only where the method chain *ends* on one, and Latu's chains always end on
the attribute — `["summary", "totalIterations"]`, never `["summary"]` alone. So the only ML cache
entries this package creates are models, and `clean_cache/1`'s count is a count of models.

**`CONNECT_ML.MODEL_SUMMARY_LOST` recovers.** After a real offload — `offloadingTimeout` at `1m`
on a fresh session, then a 70-second wait — the model answered and the summary did not, and
`Latu.ML.attribute/2` put it back with `CreateSummary` and answered `2` where it had answered `2`
before the wait. That is ML1's eviction finding seen from the other side: the offload that costs a
model nothing costs its summary everything, and the retry is what makes the difference invisible
to a caller.

The probe asks the summary **twice**, once down `Latu.Client` with no retry and once through the
facade. Without the first line an `ok` on the second proves nothing — a summary that was never
dropped answers exactly like one the retry put back.

`CACHE_INVALID`'s message calls a deleted *model* a "Summary object", confirmed here on the model
path too. Upstream finding #11.

## 2026-09-06 — An argument's type is recorded, not asked of the caller and not guessed

A hundred allowlisted attributes take arguments, and what each wants is not on the wire:
`predict` a `Vector`, `recommendForAllUsers` an `int`, `evaluate` a whole DataFrame — three
different arms of `Fetch.Method.Args`. Three ways to settle that, and only one is any good.

Asking the caller (`attribute(model, "predict", [{:vector, v}])`) makes every call site restate
something fixed, and gets it wrong quietly. Guessing from the Elixir value is worse: `1` is a
valid `int` and a valid `double`, and the server answers a wrong choice with a `MethodNotFound`
that names neither. So `dev/extract_ml_allowlist.py` reads PySpark's own annotations —
every one of the hundred has one — and `priv/ml_attributes.exs` records `:args` beside `:arity`.
`Latu.ML.attribute/3` builds the literals from that, and refuses a wrong count or a wrong kind
before anything is sent.

The extractor refuses in both directions rather than defaulting: a signature that disagrees with
the arity read off the `_call_java` call site means the two sources have drifted, and an
annotation `ARG_TYPES` does not map means a shape the plan layer has never built. Both are a
reading to do by hand.

`{:either, [:string, :vector]}` is a new `param_type`, and 4.2.0 has exactly one —
`Word2VecModel.findSynonyms(word)` takes a term or the vector to look near. The value's own
shape picks the member, in PySpark's declared order; `Latu.ML.Plan.accepts?/2` is deliberately
loose so that choosing an arm never rejects a value that arm would have taken.

Three regression models — `LinearRegressionModel`, `DecisionTreeRegressionModel`,
`FMRegressionModel` — do not parameterise their generic base while every sibling does, so
`predict(value: T)` has no `T` to resolve. Pinned in `TYPEVAR_PINS` with the reason, so a release
that fixes it shows up as three lines that can go, rather than as a default that was quietly
right. Filed as an upstream finding.

## 2026-09-06 — `attribute/2` is `attribute/3` with no arguments, and now says so

`Latu.ML.attribute(model, "predict")` used to send a no-argument `Fetch` and let the server
refuse it. It now raises client-side, naming what `predict` wants: `predict takes 1 arguments —
value (:vector)`. The registry knows, so the round trip buys nothing.

The escape hatch is unchanged where the registry does not know: a model built by
`Latu.ML.Estimator.new/2` with no class has nothing to look arguments up in, so nothing is
checked and the server's own refusal is the only one — which is exactly what that hatch promises.

## 2026-09-06 — A model-dependent summary records both candidates, not `:model_dependent`

`_summaryCls` branches on the *fitted* model for two classes:

    def _summaryCls(self) -> type:
        if self.numClasses <= 2:
            return BinaryLogisticRegressionTrainingSummary
        return LogisticRegressionTrainingSummary

ML2 recorded that as the atom `:model_dependent` — true, and useless. `dev/probe_ml.exs` found
the cost the moment `attribute/3` needed argument types: a `%Latu.ML.Summary{}` with no class has
nothing to look them up in, so `fMeasureByLabel` on a `LogisticRegressionModel`'s summary could
not be asked at all. `Latu.ML.attributes/1` had the same hole and had been left as a known gap.

So `dev/extract_ml_allowlist.py` reads the branches instead of calling the property, and
`:summary_classes` is a **list** in PySpark's own order — one entry where it commits, two where
it does not. `%Latu.ML.Summary{}` keeps them as `candidates`, sets `class` only when there is
one, and looks names up across both. `attributes/1` answers with their union.

The candidates cannot disagree on a *signature*, and that is a property of 4.2.0 rather than an
assumption: the binary summary is a **subclass** of the multiclass one, so a shared attribute has
one signature and the binary one only adds. ~~The union is therefore what `attributes/1`
reports.~~ It is not — see the entry below, which the probe forced a day later.

A branch returning anything but a bare class name stops the extractor. That is the same rule
`_summary_dataset` follows, for the same reason: the shape is narrow today and reading it is
cheap, so a widening should be a decision rather than a silent default.

## 2026-09-06 — `LDA` fits two classes, and the model says so the same way a summary does

The probe found the second half of the previous entry's problem one run later. `_create_model`
branches for exactly one estimator:

    def _create_model(self, java_model):
        if self.getOptimizer() == "em":
            return DistributedLDAModel(java_model)
        else:
            return LocalLDAModel(java_model)

A `Fit` answers with an object reference and nothing else, so nothing in the answer says which
came back — and `describeTopics` on an LDA model had no class to look its argument type up in.

Same shape, same fix. `dev/extract_ml_operators.py` records `model_classes` as a list in the
source's own order; `Latu.ML.Registry` collapses a one-element list back to `model_class`, so
the forty-one estimators that fit one class are untouched and so is everything downstream —
`Latu.ML.Generate`, the goldens, the tests. `LDA`'s `model_class` stays nil and both go in
`model_candidates`, which `fit/2` copies onto the model as `candidates`.

The three places that resolve a class now take any holder that carries candidates rather than
matching `%Latu.ML.Summary{}`, so a model and a summary are looked up the same way.

**This one is knowable and is still not claimed.** The branch reads `getOptimizer()`, a param
the *caller* set, so a client could work out which class it will get. Doing that would mean
interpreting a Python `if` in the extractor and keeping the interpretation in step with
upstream — a lot of machinery to turn "either of these two" into "this one", when both answer
every attribute the caller can reach through the candidates anyway. If a use appears that needs
the exact class, the estimator's own `optimizer` is right there to read.

## 2026-09-06 — Listing candidates is the intersection; looking a type up in them is the union

Two entries above claimed the union of a holder's candidate classes was what `attributes/1`
should report, on the grounds that 4.2.0's binary summary is a subclass of its multiclass one.
True of the summaries and **false of `LDA`**, which the probe said within one run: `LocalLDAModel`
and `DistributedLDAModel` are siblings under `LDAModel`, so their union is no real class's
allowlist, and four of its names — `getCheckpointFiles`, `logPrior`, `toLocal`,
`trainingLogLikelihood` — answer `CONNECT_ML.ATTRIBUTE_NOT_ALLOWED` on the model a default fit
actually returns. `lda` went from `:probed` to `:built` on a claim this package had made up.

The fix is not one answer but two, because there are two questions:

  * **`Latu.ML.attributes/1` lists what the server will let you fetch.** A promise, so it is the
    **intersection** — what every class the holder could be allows. Naming a class you know you
    have (`Latu.ML.attributes(Latu.ML.Clustering.DistributedLDAModel)`) gives its own wider list,
    which is the right way to ask a narrower question.
  * **`Latu.ML.Internal.arguments!/3` looks up the type of a name the caller already chose.** No
    promise is being made about what exists, so it is the **union**: a name means the same thing
    in whichever candidate has it, and a name only the other branch has is refused by the server
    with a message that names the class.

The general rule, worth keeping: a listing must not over-promise, and a lookup must not
under-serve. Merging them into one set was the mistake.

`dev/probe_ml.exs` therefore probes only the shared attributes for a class the fit decides, and
its header says so. The four `DistributedLDAModel`-only names stay unprobed until the fixture
fits an `em` LDA as well.

## 2026-09-06 — The argument-takers get accessors too, at `1 + length(args)`

The last thing the registry was extended for. `Latu.ML.Generate` no longer filters on
`arity == 0`: an attribute is generated when it answers with a value or a frame **and** the
registry knows its arguments' types, which after the previous entry is all hundred of them.
Accessors go from 520 to 613 — every argument-taker except the seven `evaluate`s, whose answer
is another cached object with no struct yet.

    Latu.ML.Classification.LogisticRegressionModel.predict(model, Nx.tensor([1.0, 0.0]))
    Latu.ML.Recommendation.ALSModel.recommend_for_all_users(model, 5)
    Latu.ML.Feature.BucketedRandomProjectionLSHModel
      .approx_nearest_neighbors(model, df, key, 3, "distance")

Each carries a `@spec` per argument and an `## Arguments` section naming them, because a spec
alone does not say which of two `int`s is which. The specs are as wide as `Latu.ML.Plan.literal/2`
really is — a `:double` argument takes `1` as readily as `1.0`, a `:string` one takes an atom —
so a spec never refuses what the code accepts.

**No default arguments, though PySpark has six.** `describeTopics(maxTermsPerTopic=10)` and its
kind default client-side and always send the value, so a generated `describe_topics/2` requires
it. Adding the defaults later is additive — a lower arity, not a changed one — so it can wait for
a reason better than symmetry with Python.

The "allowed but not here" line in each module's docs loses its argument bucket and keeps a
defensive one for an argument-taker whose types nothing could read. There are none in 4.2.0; the
filter says so rather than assuming it, which is the same rule the extractors follow.

## 2026-09-06 — `trees` answers with cache entries, so it hands back models

`:operators` was the one answer shape with no home. Two facts settle its design, and both are
read off the server rather than guessed:

  * **`MLHandler` registers every sub-model on the way out** — `a.map(m => mlCache.register(m))`
    — and joins their ids with commas into one `ObjectRef`, because `MlOperatorInfo` has a
    single ref field and no repeated one. So a fetch of `trees` is not a read, it is an
    allocation: a hundred-tree forest becomes a hundred cache entries the caller now owns.
  * **The element class is in PySpark's annotation**, the same `List[X]` that identified the
    shape in the first place. It is emphatically not guessable from the owner: a **GBT
    classifier's** trees are `DecisionTreeRegressionModel`s, because gradient boosting fits
    regression trees to residuals whatever it is classifying, while a random forest
    classifier's are `DecisionTreeClassificationModel`s. `priv/ml_attributes.exs` records it as
    `:element_class`.

So `trees/1` answers `{:ok, [%Latu.ML.Model{}]}` and its docs say the list is spent cache.
`Latu.ML.delete/1` grew a list clause to match — `MlCommand.Delete` takes repeated refs, and
giving a forest back one tree at a time would be a hundred round trips for no reason.

## 2026-09-06 — Five error classes get atoms, and a second opinion beside Spark's message

Latu's rule is that a server's refusal reaches the caller intact, and it still does: nothing
here rewrites `%Latu.Error{}`. What ML adds is two functions.

`Latu.ML.error_kind/1` maps the five `CONNECT_ML` classes to atoms. The reason it exists rather
than a note saying "compare `error_class`" is that the string is a trap: `error_class` carries
the **dotted** name and Spark's own documentation, this package's roadmap and every entry above
write these bare. A matcher built from the prose compares against a string the server never
sends, and fails as "that error never happens".

`Latu.ML.hint/1` is a second opinion, for the five whose text is actively misleading.
`CACHE_INVALID` calls the missing object a *Summary object* even for a model, blames a
fifteen-minute idle eviction even for a model deleted a moment earlier, and recommends
`model.evaluate(dataset)` — all three measured false — and every one of the five opens "Generic
Spark Connect ML error". The hint says what to do; the server's own message is still there.

## 2026-09-06 — `Inspect` for the ML structs, and deliberately no `Kino.Render`

The default struct inspect prints a training plan and a param list that are pages long and say
nothing. `#Latu.ML.Model<KMeansModel "4b8aa52d…">` is what a reader wants: which operator, and
enough of the ref to tell two apart. An estimator shows only the params the caller **set**,
which is exactly what a `Fit` will send.

A model whose class the fit decides shows every class it could be —
`#Latu.ML.Model<DistributedLDAModel | LocalLDAModel "…">` — rather than picking one, which
would be a lie, or neither, which would be unhelpful.

**No `Kino.Render` impl**, which is Latu's own precedent for a `Latu.Session` and a
`Latu.GroupedData`: Kino falls back to `inspect/2`, and a handle has no HTML worth rendering.
What does render is what a model *produces* — `transform/2` and `attribute_frame/2` hand back a
`Latu.DataFrame`, and Latu renders that already. This is a decision not to add a dependency
surface, not an omission.

## 2026-09-06 — Three goldens for an argument-carrying `Fetch`, one per arm

`dev/probe_ml.exs` proves the server *accepts* the literals this package builds for the hundred
argument-taking attributes. Only a golden proves they are the bytes PySpark builds, which is the
standard every other part of the wire format is held to — and because each argument's type comes
from `priv/ml_attributes.exs`, a matching fixture is also the registry's types being right on the
wire rather than merely being accepted.

Three, chosen so that no two make the same claim:

  * `ml_fetch_predict` — a `Vector`, the widest literal the plan layer builds, on the command arm.
  * `ml_fetch_f_measure_by_label` — a double, and one that arrives *after* a chain: the object is
    the model, `summary` leads, and only the last method carries an argument.
  * `ml_relation_recommend_for_user_subset` — a DataFrame **and** an int, so one argument list
    holds both `Fetch.Method.Args` arms. Nothing else in the file pins the `input` arm at all.

## 2026-09-06 — Fetching `trees` twice costs twice

The probe measured it: three trees on a three-tree forest took the cache from 1 entry to 4, and
asking again took it to 7. `MLHandler` registers the sub-models on **every** `Fetch` and dedupes
nothing, so `trees` is not a value a caller may read in a loop — it is an allocation, and each
one is theirs to give back.

Found by a bug in the probe itself, which asked twice (once through the accessor, once through
`attribute/2`) and deleted one batch. The numbers looked like a server surprise and were not.
The section now asks once, then asks a second time *deliberately* and prints the cost, because
the non-dedupe is the finding rather than the accident.

## 2026-09-06 — `dev/check_offline.exs` silences eight warnings, by name

Five modules it does not compile — `Latu.ML`, `Latu.ML.Result`, `Latu.ML.Linalg`,
`Latu.Result.Literal` — plus `Latu.ML.Internal`, which it does compile but *after* the three
operator structs that call its `uid/1`. Eight warnings per run, none of them news, and between
them enough noise to hide the one warning that would matter.

`Code.put_compiler_option(:no_warn_undefined, [...])` with the five named, rather than turning
warnings off: anything not on the list still warns, which is the whole point of running it.

## 2026-09-06 — `save/3` implies the session where a struct carries one, and asks where it cannot

A model knows its session: it is a reference into that session's cache and means nothing
anywhere else. An estimator, a transformer and an evaluator are inert data with no session
behind them at all, and `load/3` has nothing *but* a session. So the session arrives three
different ways, and each is the only honest one for its case:

    ML.save(model, path, overwrite: true)          # the model's own
    ML.save(lr, path, session: session)            # nothing else carries one
    ML.load(session, :logistic_regression, path)    # nothing else to carry it

Passing `session:` with a model is refused by `Keyword.validate!` rather than ignored — a second
session beside a model's own is not a redundancy, it is a way to write to the wrong server —
and omitting it for an unfitted operator names what to add. The alternative, a session first
everywhere, would read uniform and make that mistake possible; the alternative of making it an
option on `load/3` too would dress a required argument as an optional one.

## 2026-09-06 — A loaded param is typed from its own arm, not looked up in the registry

`MlParams` is a bag of literals with no declared types beside it, so `Latu.ML.Result` reads each
param's type off the arm the server chose. The obvious alternative is the registry's declared
type for that class, which is what PySpark effectively does — `_set` runs each value through its
param's `typeConverter`.

They cannot disagree in a way that reaches the wire. Every arm rebuilds as itself: `:float` and
`:double` are one arm here as they are everywhere else in this package, an `:int` past 32 bits
becomes a `long` by magnitude whichever type declared it, and a Vector is a Vector. What the arm
buys over the registry is the case the registry has no answer for — a class from
`Latu.ML.Estimator.new/2`, or a third-party model — which loads exactly as readily as a class
this package knows.

Two arms are read but not written back as themselves. A `specialized_array` rebuilds as an
ordinary `array` literal, because the server reads both and a second encoder that exists only
for values that came from a load is a second encoder to keep right. And anything else — a
timestamp, a map, a struct that is not a Vector or a Matrix — is refused **naming the param**,
because the alternative is a model that loads quietly and sends something else on its next
transform.

## 2026-09-06 — A `Write` answers with an empty result, and Latu drops it on the floor

`MLHandler` answers a `Write` with `proto.MlCommandResult.newBuilder().build()` — a message with
no arm set. Latu's `ml_command_result` latch is guarded `when not is_nil(result.result_type)`, so
that message never reaches `%Execution{}` and `save/3` sees `ml_command_result: nil`.

Nothing is lost: there is nothing in it. A clean execution is the whole answer, which is what
PySpark relies on too — `RemoteMLWriter.saveInstance` ignores the return of `execute_command`
entirely. So `save/3` matches on the execution succeeding and answers `:ok`.

Worth writing down because it is a **seam finding rather than a bug**: Latu's guard was written
when every ML result had an arm, and `Write` is the proof that an arm-less `MlCommandResult` is
a real thing a server sends. If a later command ever answers empty *and* meaningfully — an
acknowledgement a caller has to see — that guard is what would have to be relaxed, and it is a
Latu release rather than anything this package can do.

## 2026-09-06 — `load/3` names what is at the path, and a summary module is refused by name

Spark's saved metadata carries the class, but a `Read` has to name the class *before* the server
will look, so there is no version of this where the caller does not say what they expect. Three
spellings, one per way a caller knows what they saved: an operator name from the registry, a
generated model module, or `{class, kind}` for something this package never heard of.

A generated **summary** module is an atom that looks exactly like a model module and is the near
miss worth catching — nothing saves a summary, and a caller who tries has misread which object
they have. It is refused by name rather than sent, because the server's own message for it names
a class the caller never mentioned.

`Latu.ML.attributes/1` already accepted a module, a struct or a class string, so a module
spelling here is the surface being consistent rather than an invention.

## 2026-09-06 — A loaded model has no summary, and the recovery knows not to try

A model's save format does not carry its training summary — `MLCache` writes one to a sibling
path when it offloads, and a `Read` does not go looking. So a loaded model answers
`CONNECT_ML.MODEL_SUMMARY_LOST` for its summary, exactly as an offloaded one does, and the two
are indistinguishable from the outside.

They are not the same, though, and the difference is what a client can do about it. An offloaded
model's summary is rebuilt from the frame it was fitted on. A loaded model has no such frame and
never did, so `Latu.ML.attribute/2` reports the server's refusal rather than sending a
`CreateSummary` with a nil dataset — a guard on the retry clause, and the reason `Latu.ML.hint/1`
for `:summary_lost` now names `load/3` as one of the two ways to arrive there.

`Latu.ML.summary/1` still builds one, because it is a lazy builder and a class check: refusing at
build time would mean this package deciding, on its own, that a class it can see has no summary
it can reach. The server says that, and says it in a class this package maps to an atom.

## 2026-09-06 — The six evaluators are probed on written frames, not fitted ones

`dev/probe_ml.exs` fits or applies every runnable operator; an evaluator is the one kind that
reads *predictions* rather than features, and nothing else in the probe produces a scored frame.

Fitting a classifier to make one would have made every evaluator's status a statement about the
classifier in front of it — a `:built` on `RegressionEvaluator` that really meant that
`LinearRegression` misbehaved. So three frames are written with `sql/2`: scalars (`label`,
`prediction`, `rawPrediction`), a Vector with a cluster beside it, and arrays of doubles.
Between them the six are covered, and each one's status is about itself.

`rawPrediction` is a double rather than a Vector, which `BinaryClassificationEvaluator` accepts
and which is one fewer moving part in a fixture.

## 2026-09-06 — Interop is a dance between two scripts, not a test

"PySpark can read what this wrote, and this can read what PySpark wrote" needs two clients, one
server and one directory, and no test runner owns all three. So it is
`python dev/pyspark_oracle.py --interop` and `mix run dev/probe_ml.exs -- --interop`, run in the
order `dev/README.md` gives, each writing its own model and reading the other's.

What crosses between them is **only the saved model**. Each side compares the coefficients it
reads out of the other's file against the coefficients of a model it fitted itself, on the same
rows with the same params against the same server — so the two fits are the same computation and
agreeing exactly is a real expectation rather than a tolerance. A difference is then one thing
only: a format two clients read differently.

Neither half stats the filesystem to find out whether the other has run: the load is the test,
and its refusal says so plainly enough. That keeps both halves independent of where the server
keeps its `/tmp`.

## 2026-09-06 — The helper object's names are Scala's and its argument types are call sites'

`ConnectHelper` is in `ALLOWED_ATTRIBUTES` like every model class, and is an attribute of
nothing: a singleton the server resolves by the literal id `______ML_CONNECT_HELPER______`.
That makes it the one part of the surface where the two sources this package reads have to be
joined rather than merely read.

The **names** are the server's, from the Scala. The **arguments** are only at PySpark's call
sites — `invoke_helper_attr("stringIndexerModelFromLabels", model.uid, (list(labels),
ArrayType(StringType())))` — because a helper has no Python class and so no signature to
inspect. So `dev/extract_ml_allowlist.py` walks for those calls and resolves each argument
expression: a bare name against the enclosing function's annotation, a `self.getK()` against
that class's own param table, an explicit `(value, DataType)` tuple by reading the `DataType`,
and an `IfExp` by taking the interesting arm.

Six expression shapes cover all ten methods, and anything else stops the run naming the call
site. The alternative — an eleven-row table written by hand — is what `TYPEVAR_PINS` is, and it
would have been defensible; what decided it is that the two sources can now be **checked against
each other**. A method PySpark calls that the allowlist does not have, or a method the server
allows that no call site types, is a problem the extractor reports rather than a gap nobody
sees. `handleOverwrite` is the one method with no `invoke_helper_*` call site — PySpark reaches
it through `_call_java` — so it is excused by name, with its reason, in
`HELPERS_WITHOUT_CALL_SITE`.

## 2026-09-06 — A fifth operator kind, read off a property rather than named

`PowerIterationClustering` and `PrefixSpan` carry params and a JVM class like an estimator, and
are neither estimator, transformer, model nor evaluator: what they do is one method on the
helper object. `dev/extract_ml_operators.py` skipped them silently, so their ten params were in
no registry at all.

They are now kind `:helper`, and the rule is a property rather than a list of names: a concrete
`JavaParams` operator that is none of the four. On 4.2.0 that catches exactly those two, checked
before the rule was written. A third in a later release becomes a row in the committed diff
instead of vanishing, which is the same bargain every other extractor rule makes.

`%Latu.ML.Helper{}` is their struct, and it has **no uid** — unlike every other operator — for
the plain reason that nothing on the wire carries one: a helper call sends params as positional
arguments and no `MlOperator` at all.

## 2026-09-06 — A helper's arguments are positional, so the defaults are the client's problem

`MlParams` is a map carrying only what the caller set, and the server fills the rest from the
operator's own defaults. A helper call has no such thing: its arguments are positional, so
**every one is sent every time**, and something has to decide what an unset param becomes.

Three sources, in order, and none of them invented: the caller's value; then the param's own
default from `priv/ml_operators.exs`; then the fallback PySpark's call site records —
`self.getWeightCol() if self.isDefined(self.weightCol) else ""`, which the extractor now reads
as `unset: ""` rather than dropping. `weightCol` is the only argument on 4.2.0 that needs the
third, and reading it beat inventing a rule about empty strings.

A param with none of the three is refused by name. The four marker defaults — `:varies`,
`:server_supplied`, `:nan` and `{:uid_suffix, _}` — count as none, because sending a marker
would be worse than refusing.

## 2026-09-06 — The three model-building helpers are generated onto their model modules

`StringIndexerModel.from_labels` and its two siblings build a model server-side out of a list
the caller already holds. PySpark has them as classmethods on the model class, and the answer
to where they go here was settled before the work: mirror that.

Which forced them to be **generated**. A group module like `Latu.ML.Feature` is hand-written
around a `constructors(:feature)` call, so a function can sit beside the generated ones — but a
model's accessor module is emitted whole by `accessors/1`, so nothing can be added to it by
hand. That is why the helper rows carry `:function`, PySpark's own name snaked: the generator
needs a name, and a name is exactly the kind of thing this package reads rather than invents.
A name a bare `def` could not take stops the compile.

The uid is the seam worth knowing. It is the **client's** — the call's first declared argument —
and sending it is how the model the server registers and the `%Latu.ML.Model{}` the caller holds
agree on one. A generated constructor makes it; the generic `Latu.ML.helper/3` does not, and
takes it like every other declared argument. An escape hatch that quietly supplied one would be
an escape hatch that surprises.

## 2026-09-06 — `Latu.ML.Stat` is hand-written, because a flat module cannot have two `test`s

The other seven helper methods are named on the Elixir side rather than generated, and the
three statistical tests say why: PySpark spells them `ChiSquareTest.test`, `Correlation.corr`
and `KolmogorovSmirnovTest.test`, three classes with two `test`s between them. Flattened into
one module — which is the right home, since none of them is about an operator — the names have
to be decided, and `chi_square_test`, `correlation` and `kolmogorov_smirnov_test` are the
decision.

The other two named by hand are `StopWordsRemover`'s, and they go on `Latu.ML.Feature` for the
reason PySpark puts them on the class: they answer what that transformer's two
`:server_supplied` defaults are. Which leaves `Latu.ML.helper/3` and `helper_frame/2`
underneath all of them, in exactly the relationship `attribute/3` has to the generated
accessors.

## 2026-09-06 — Nested lists, and the two places that need them

`Latu.ML.Plan.literal/2` refused a list of lists from ML1 until now, on the grounds that
`Bucketizer.splitsArray` was the only param that needed one and no golden said what PySpark
sent for it. `stringIndexerModelFromLabelsArray` is the second place, and between them they
answer the question from both directions: the helper types its argument **explicitly** at the
call site (`ArrayType(ArrayType(StringType()))`) and the param leaves PySpark to **infer** it
from the value.

Both goldens are in `test/wire/`, and the two paths write the same bytes — an array literal
whose `element_type` is an `ArrayType` with `containsNull` set, each element an array literal
carrying its own element type. So one recursive arm in `literal/2` serves both, and the param
that ML1 documented as refused is refused no longer.

Elixir has no float infinity, so the `splitsArray` fixture uses finite bounds. `Bucketizer`
only asks that splits increase; PySpark's own docs use `-inf` and `+inf`, and a client on the
BEAM cannot send those at all — worth knowing before someone reports it as a bug.

## 2026-09-06 — A pipeline is folded here, because there is no wire form for one

`MlCommand` has `Fit`, and `Fit` takes an `MlOperator`. There is no operator for a pipeline:
Spark's `Pipeline` is a JVM class the Connect server never exposes, so a pipeline cannot be
fitted over the wire by any client. PySpark loops in Python; Latu folds in Elixir. Both send
one `Fit` per estimator stage and one `Transform` per stage that has to feed the next.

So `Latu.ML.pipeline/1` reaches no server and needs no session — it is a builder, and the
`%Latu.ML.Pipeline{}` it returns is inert data like every other operator struct here. The
session arrives with the frame, at `fit/2`.

Two things fall out of the fold that are worth stating, because nothing offline can check
either. The **last estimator's own output is never computed**: nothing after it is fitted, so
transforming through it would be a round trip for a frame no one reads. And a fit that fails
partway has already cached models on the server that nothing asked for and nothing else will
give back, so the stages fitted before it are deleted — in one `Delete` — before the error is
returned. A failure logs rather than replaces the error the caller wants.

`dev/probe_ml.exs`'s pipelines section checks the fold lands exactly where fitting the same
stages by hand does. That is the only place it can be checked at all.

## 2026-09-06 — The saved layout is PySpark's, and Scala cannot read it

A pipeline's directory is written by the **client**: `metadata` and `stages/NN_uid/`, with the
stage directories themselves written by the server through `MlCommand.Write`. So the layout
around the stages is a format this package chooses, and there are two to choose from.

Read side by side, they differ by **one key**:

| | PySpark writes | Scala writes |
|---|---|---|
| `class` | `pyspark.ml.pipeline.Pipeline` | `org.apache.spark.ml.Pipeline` |
| `paramMap.stageUids` | same | same |
| `paramMap.language` | `"Python"` | absent |
| `uid`, `timestamp`, `sparkVersion` | same | same |
| `stages/NN_uid/` padding | `zfill(len(str(n)))` | `s"%0${n.toString.length}d"` — identical |

And the readers are not symmetric. PySpark 4.2's `PipelineReader.load` consumes exactly
`metadata["uid"]` and `metadata["paramMap"]["stageUids"]` — it never looks at `class` or
`language` at all. Scala's `DefaultParamsReader.loadMetadata` does
`require(className == expectedClassName, ...)`. So the PySpark name is readable by PySpark
only, and the Scala name would be readable by both.

Latu writes PySpark's. Not because it is the better format — by that table it is the worse one
— but because it is the one this package can *check*. `dev/pyspark_oracle.py --interop` reads
what Latu wrote with a client that is already in the loop; claiming the Scala name would mean a
JVM, a Spark distribution and a `Pipeline.load` smoke test in CI to say anything true about it,
and no one has asked for Scala to read these. The gap is one string in `Latu.ML.Layout`, so the
day someone does ask, the change is a constant and the work is the harness.

`docs/deviations.md` states the limitation plainly, because a caller who saves a pipeline and
hands the path to a Scala job should find out here rather than there.

## 2026-09-06 — A Latu-written metadata file says so, in a key both readers ignore

`Latu.ML.Layout.metadata/4` adds a top-level `latuMl` key beside Spark's six. Two reasons: a
directory that turns up in a bug report can be attributed without guessing, and a reader that
one day cares which client wrote a file has something to read.

It is inert on both sides, checked rather than assumed. PySpark's `DefaultParamsReader` returns
the parsed JSON as a plain dict and reads named keys out of it. Scala's `parseMetadata` extracts
its six fields with json4s path expressions into a `Metadata` case class, keeping the whole
`JValue` alongside; neither validates the key set. So the key costs a reader nothing and is
visible to anyone who opens the file, which is the whole point of putting it there rather than
in a sidecar.

It is **not** a version marker and nothing branches on it. A file Latu wrote and a file PySpark
wrote are the same format; this says who held the pen.

## 2026-09-06 — `isLargerBetter` is extracted from six client-side overrides

`MLUtils.scala`'s allowlist has no `isLargerBetter`, so a `Fetch` cannot ask for it and the
probe cannot settle it. PySpark works around exactly this: its six evaluators each override the
method **client-side** — `BinaryClassificationEvaluator` returns `True`, `RegressionEvaluator`
returns `metric in ["r2", "var"]`, and so on — precisely so the answer exists without the JVM.

An override is a Python expression, so it is read rather than transcribed.
`dev/extract_ml_operators.py` parses each one and accepts exactly three shapes, raising on
anything else rather than falling back on the base class's `return True`. That fallback is the
tempting one and it is wrong: a default here does not fail, it silently reports the worst model
in a grid as the best one.

The value lands on the evaluator's registry row as `true`, `{:only, metrics}` or
`{:except, metrics}`, and `Latu.ML.larger_better?/1` resolves it against the evaluator's own
`metric_name`, set or default. Public, because a caller comparing metrics by hand should not
have to remember that `rmse` is minimised and `r2` maximised.

## 2026-09-06 — A grid entry carries the uid of the operator it sets

`Latu.ML.param_grid/2` could have returned plain keyword lists. It returns
`{uid, param_name, value}` triples instead, and the uid is the whole point: the thing being
searched is usually a **pipeline**, and the param belongs to one stage of it. PySpark has the
same problem and the same answer — its `Param` objects carry a `parent` uid, and the saved
format keys each entry on `parent` and `name`.

So `Internal.apply_params/2` walks a pipeline and reaches the stage that owns each setting,
leaving the others alone. The cost is that a grid reads badly in `iex`; callers build one with
`param_grid/2` and hand it straight to a validator, so nobody should have to look inside.

`Internal.covered!/1` refuses a grid whose params reach nothing, **before the first fit**.
PySpark makes this check only when a validator is *saved*, which is late enough to be useless:
a search over the wrong estimator runs to completion, every param map scores identically, and
the first one "wins". A search you never persist never finds out at all.

## 2026-09-06 — A sub-model is released as its metric is read

A 5-fold search over a 6-point grid fits thirty models. PySpark drops the Python references and
lets the JVM collect them; the BEAM has no finalizer, so something has to say when they end.

They end immediately. Each sub-model is deleted the moment the evaluator has scored it, so the
search holds one cache entry at a time rather than thirty, and what survives is the winner —
refit on the whole frame, because the fold models each saw a fraction of the rows.
`collect_sub_models: true` keeps them instead, and they become the caller's:
`Latu.ML.delete/1` on the returned model releases them along with `best_model`, in one
`Delete`. `dev/probe_ml.exs` counts cache entries before and after, so the claim is a number
rather than a comment.

Same shape as the pipeline fold and `load_stages`: anything that fails partway gives back what
it already cached before returning the error.

## 2026-09-06 — The search is serial, and says so rather than claiming otherwise

PySpark runs the grid on a `ThreadPool` sized by its `parallelism` param. Latu fits one at a
time.

Not because concurrency is wrong, but because the claim would be unmeasured. Whether several
processes can safely drive one session's channel at once is not something this package has
tested, and Latu's own suite found once that the server serialises much of the work anyway —
so the speedup is unknown as well as the safety. `max_concurrency:` gets built when a probe
says what it buys, and `docs/deviations.md` states the difference in the meantime.

Population standard deviation, not sample: `np.std` defaults to `ddof=0` and PySpark takes the
default. `dev/pyspark_oracle.py --interop` prints PySpark's `stdMetrics` for the same grid,
seed and rows, which is the only thing that could have caught the difference.

## 2026-09-06 — The tuning fixture is chosen so that a broken search fails

The first version of the probe's tuning section searched a logistic regression on the four-row
fixture and printed `[0.75, 0.75, 0.75]` — matching PySpark exactly, and meaning nothing. Every
param map scored identically, so three ties made index 0 the answer whether or not the grid was
ever applied, the fold cut worked, or `larger_better?` was right.

The fixture is now twelve rows and a **linear regression scored by `rmse`**, for three reasons
that are worth keeping if it is ever changed again:

  * `rmse` varies continuously with `reg_param`, and the grid spans `[0.5, 0.0, 5.0]` — wide
    enough that the three metrics genuinely differ. `the grid actually discriminates` asserts
    exactly that, so it cannot go degenerate again quietly.
  * `rmse` is **minimised**, so the argmin branch of `larger_better?` runs. The old fixture
    only ever exercised the maximising path, which is also the default — the branch most
    likely to be wrong was the one never tested.
  * The grid is ordered so the winner sits in the **middle**. A best index of 0 is what a
    broken argmin returns and what "take the first" returns; 1 is an answer only a working
    comparison produces.

## 2026-09-06 — A saved grid's value is JSON inside JSON

`estimatorParamMaps` is a list of lists of `{parent, name, value, isJson}`. `parent` is the uid
of the operator whose param it is, `name` is the **wire** spelling — and `value` is a JSON
*string*, not a JSON value. PySpark writes `jsonParam["value"] = json.dumps(v)` into a document
that is itself JSON, and reads it back with `json.loads`, so `0.5` is stored as `"0.5"` and a
file with the number in it does not load.

Nothing about that is inferable from the format; it is only visible with the writer and the
reader side by side. Pinned in `test/latu/ml_test.exs` with the reasoning attached, because the
obvious "fix" — writing the number — is a silent interop break.

`isJson: false` marks a value that was itself an operator, written to its own `epm_*`
directory. No param in this registry takes an operator (`OneVsRest`, the one that does, is in
neither extractor), so both writing and reading that shape are refused by name rather than
half-implemented.

## 2026-09-06 — A fitted search carries fewer params than an unfitted one

`DefaultParamsReader.getAndSetParams` calls `getParam(name)` for **every** key in `paramMap`
and raises on one the instance does not have. `CrossValidator` has `collectSubModels`;
`CrossValidatorModel` does not. So writing that key into a saved *model* makes PySpark's
`CrossValidatorModel.load` raise — and nothing on this side would ever have noticed, because
Latu's own reader ignores unknown keys.

PySpark never trips on it because `extractJsonParams` extracts from whichever instance is in
front of it. A client building the map itself has to know the difference, so
`Latu.ML.Layout.search_params/3` takes the kind and drops the key for the fitted ones.

Found by reading `CrossValidatorModel().params` rather than by a failure, which is the only
reason it is not an interop bug in this milestone.

## 2026-09-06 — Loading a search leaves its sub-models on disk

PySpark's reader loads every persisted sub-model. A five-fold search over a six-point grid is
then thirty `Read` commands and thirty cache entries, on a `load/4` whose caller may only have
wanted the winner — and on the BEAM those thirty have no finalizer behind them.

So `load/4` reads `bestModel` and stops. `sub_models: true` asks for the rest, and then they
are the caller's, released by `delete/1` along with the winner. The directory is unchanged
either way, so a second call gets them.

Same reasoning as ML5b, where a sub-model nobody asked for goes back as soon as it is scored:
this package does not cache things on the caller's behalf without being asked. A deviation, and
`docs/deviations.md` says so.

`sub_models: true` on a search that was saved without them is a refusal naming the metadata key
rather than a nil, because the alternative is a caller wondering which of the two ends lost
them.

## 2026-09-06 — `sub_models:` is an option of its own, not an entry in `options:`

PySpark decides whether to write sub-models from the writer's option map —
`writer.optionMap["persistsubmodels"]`, lowercased on the way in by `MLWriter.option`. Latu's
`options:` means the options Spark's own writer reads, and this package deliberately does not
lowercase what a caller wrote (`docs/deviations.md`). Inheriting the wart would have given one
keyword two meanings.

`sub_models: true` instead, defaulting to whether the search collected any — which is PySpark's
default, reached differently. The metadata key written is still `persistSubModels`, so PySpark
reads it.

## 2026-09-07 — `attribute/2` takes either spelling of an attribute name

`Latu.ML.attributes/1` lists snake_case names because that is what every other name in this
package is, and the server's allowlist is camelCase because that is what `MLUtils` holds. Taking
only the wire name made the table a list of things a caller could read and not ask for, and
`dev/example.exs` found it the way a user would: `attribute(summary, :area_under_roc)` answering
`CONNECT_ML.ATTRIBUTE_NOT_ALLOWED` for a name the package had just advertised.

`Latu.ML.Internal.wire_name/2` resolves it, in this order: a name already on the class's
allowlist is taken as written, so a wire name can never mean something else; then the rows' snake
names; then **through unchanged**, which is what keeps a class the registry does not know
reachable at all — the server's own refusal is the better error there than a guess.

The generated accessors stay the documented route: they carry the class as well as the name, so a
model of the wrong one is refused before the round trip. This is the layer underneath them.

## 2026-09-07 — `delete/1` takes a mixed list, and refuses anything else by name

Its clauses were a model, each composite, and a list of *models* — so `delete([fitted, best])`,
which is what a script that fitted a pipeline and a search naturally writes, matched nothing and
raised a `FunctionClauseError` printing two fitted models. `CLAUDE.md` promises a refusal that
names the fix.

Now every shape funnels into one list clause: a composite is flattened by the same walk it always
was, so a mixed list is one `Delete` for the whole tree. Inside a composite a stage that holds
nothing on the server is skipped, as before. **At the top level the contract is strict** — the
four shapes, or an `ArgumentError` naming them — because a silent `:ok` for an argument this
cannot release is the worse answer, and `cached_models/1`'s leniency would have given exactly
that.

## 2026-09-07 — the ExDoc gate is Latu's two test files, adapted twice

`test/latu/docs_test.exs` and `test/latu/examples_test.exs` are ports: docs coverage, sidebar
groups, extras and links; then every markdown page accounted for, every block parsed, every call
and reference resolved at its arity. `mix docs --warnings-as-errors` was already in `check.all`
and checks neither.

Two adaptations, both because most of this surface is generated. `:groups_for_modules` carries a
**regex** for the 64 accessor modules, so "is it placed" is membership *or* a match — and a
pattern matching nothing is itself an error, since a dead pattern hides a module as surely as a
missing entry. And there is **no hand-maintained alias map**: Latu keeps one because it has two
aliases, while the docs here alias a dozen generated modules, so an alias head resolves by unique
suffix against the app's own module list, with an ambiguous suffix raising rather than silently
picking one.

The bare-reference check deliberately covers `@moduledoc false` modules too. ExDoc renders
nothing for them, so a bare span there cannot break a page — but it still tells the next reader a
function exists at that arity, and every one this found on its first run was a cross-module
reference written bare.

## 2026-09-07 — third-party operators: probed, then deferred on the evidence

`latu-ml-roadmap.md`'s D11 asked where hand-written registries for XGBoost4J and friends should
live. `dev/probe_ext.exs` asked the prior question first — whether they work at all — and the
answer moves the decision rather than settling it. Nothing under `lib/` changed to find out:
the script drives `Latu.ML.Estimator.new/2`, which has been the documented escape hatch since
ML1, and posts its own artifacts.

**The mechanism works.** The server discovers ML operators through a `ServiceLoader` over
`META-INF/services/org.apache.spark.ml.Estimator`, which no third-party library ships — so a jar
carrying nothing but that file is the gate, and 347 bytes of it is enough. Posted from the client
under `AddArtifacts`'s `jars/` prefix it is discovered per session, with no restart and no
administrator, and LinkedIn's isolation-forest then fits, transforms, reports `model_size`, saves
and releases from Elixir against a live 4.2.0 server.

**Three bounds, all structural rather than incidental.** A session artifact reaches the driver's
classloader and not the executors', so a library whose `fit` runs a Spark job over its own
classes fails inside that job (`ClassNotFound`, or a task result that will not deserialise) —
library jars belong on the cluster classpath and only the shim comes from the client. The shim
may name **estimators only**: a third-party model class has no public no-arg constructor, one
unresolvable provider fails the whole `ServiceLoader` enumeration, and the server runs that
enumeration on *every* transform — so naming a model class breaks `VectorAssembler` for the
session, and an artifact cannot be withdrawn from a session either. MLlib's own models survive in
that file only because each carries a `private[ml] def this()`, package-private in Scala and
public in bytecode. And `Read` needs the model class in that same file, so a third-party model
cannot be loaded back server-side at all; `load/3`'s `{class, kind}` spelling answers
`Unsupported read`, and the artefact is read locally instead.

**And XGBoost4J-Spark, the reason to want any of this, does not work on 4.2.0.** It trains, then
cannot write its metadata: `CustomGeneralParam.jsonEncode` calls
`org.json4s.Extraction.decompose`, whose json4s 3.x signature Spark 4.x dropped, and
`SparkParams` defaults `customObj` and `customEval` to `null`, which Spark's writer encodes on
every save — so it fires whatever the caller passed. Default memory control saves a model as it
registers it, so every fit fails, and the trained model is left orphaned in the ML cache because
`MLCache.register` caches before it saves. One file upstream; nothing here can route around it.

So **D11 is deferred with evidence rather than left open, and nothing ships in 0.1.0.** The one
library proven end to end is one nobody asked for; the one people ask for is blocked upstream; and
a hand-written registry with no oracle is the very objection that kept ML out of Latu in the first
place (`latu`'s `docs/decisions.md`, 2026-09-02). It should not re-enter as the weakest corner of
a package whose whole claim is that nothing is transcribed and nothing unprobed is called
supported — `:probed` here would have meant "on one laptop, with these jar versions, on this
Spark", which is a different and weaker word than it means everywhere else in this repo.

`dev/probe_ext.exs` stays as the worked example and the record of how it was asked; `dev/` is not
in the Hex file list, so it ships nowhere. When ext support is built it belongs here as
`Latu.ML.Ext.<Library>`, opt-in, with its own registry and status files and out of every core
count — a separate package would make `Latu.ML.Generate` public compile-time API to solve a
release-cadence problem that does not exist until there are two libraries. Revisit when XGBoost
rebuilds against Spark 4.x, or when someone wants CatBoost (the only remaining candidate built
against a 4.x line, which is what separated isolation-forest's success from XGBoost's failure).

## 2026-09-11 — `Summarizer` is `Stat.summary/3` and ten shortcuts, over `get_field/2`

D14 waited on field access, and Latu 0.7.0's `Latu.Column.get_field/2` is it. PySpark's route
on Connect is `aggregate_metrics(array(metric names), features, weight or lit(1.0))`, a struct
keyed by metric, then `getField`. `Latu.ML.Stat.summary/3` is that one call with the metrics as
a list of atoms and `weight:` as an option; the ten shortcuts (`mean/2` through `norm_l1/2`) are
`get_field/2` over a one-metric summary, which is exactly what PySpark's `Summarizer.mean`
builds, minus the alias PySpark adds and Latu leaves to the projection. The wire is pinned by
three goldens built from `Summarizer.metrics(...).summary(...)` and `.getField`, not from the
shortcut, for that reason.

**Metric names are atoms in Elixir spelling, refused client-side against the ten.** The server
would answer `FAILED_FUNCTION_CALL` for an unknown name; the set is closed and known, so the
typo is caught where it is made. The struct's fields keep Spark's spelling (`numNonzeros`,
`normL2`), because the server names them and a caller reading the struct directly should see
what the server wrote.

**`max` and `min` collide with `Kernel`'s.** Spark's names win, and the module imports `Kernel`
without those two arities; nothing in it needed them. A rename would have been a deviation with
no reason behind it but the collision.

## 2026-09-11 — `Latu.ML.Persistence` and `Latu.ML.Cache`, the facade's first split

`lib/latu/ml.ex` was 2150 lines. `save/3`, `load/4` and their 38 private helpers were 600 of
them, called nothing else in the file, and were called by nothing else in the file. They are now
`Latu.ML.Persistence`, `@moduledoc false`; the four verbs stay on `Latu.ML` with their
docstrings, because `h Latu.ML.save` is where the contract is read.

The seam was measured before the move and had grown since the punch list scoped it: the region
needed `traverse/2`, `each_ok/2`, `collected/1` and `release/1` from the facade, the four
functions that apply the rule that a `Read` caches so every error path owns what it read. Those,
with `cached_models/1` they rest on, are `Latu.ML.Cache`, also hidden, shared by both. Making
them public on the facade instead would have put five internals on the cheatsheet, whose test
counts every export of `Latu.ML`.

**The layering rule widened by one module.** "Only the facade calls Latu's transport" now admits
`Latu.ML.Persistence`, since a `Write` and a `Read` are the RPCs those two verbs are; the test
says so beside the regex. The proto rule is unchanged: neither new module names
`Latu.Protocol`.

`dev/check_offline.exs` reads all three files against themselves now, with a floor per file.
The facade's floor came down from 100 definitions to 60, which is what it has after the move.

## 2026-09-11 — The floor is 1.18, the pin is `~> 0.7`

Both follow Latu's own decisions of the same day. Nothing here needs a stdlib newer than 1.18
(a scan of every stdlib call found nothing past 1.14), and the `googleapis` dependency under
Latu asks for 1.18 anyway. The `latu` pin names the minor this package was built and tested
against, which settles the question 0.2.0 left open in the changelog: the pin says what was
tested, not the oldest thing that happens to work, and `Summarizer` makes 0.7.0 a hard
requirement in any case.
