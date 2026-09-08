# Conventions

`latu_ml` is Spark MLlib from Elixir, over Spark Connect, as a companion package to
[Latu](https://hexdocs.pm/latu). Latu owns the session, the transport and the plan; this package
owns the ML commands and relations built on top of them.

Design docs, roadmap and progress notes live in the Latu project on claude.ai, not in the repo —
`latu-ml-roadmap.md` is the plan this follows. Per-decision rationale goes in
`docs/decisions.md`, which does not ship; every place the public API departs from PySpark goes in
`docs/deviations.md`, which does. Only the delta from Latu's own two files belongs in either.

**Latu's `CLAUDE.md` applies here in full.** What follows is the delta, plus the rules that are
load-bearing enough to restate.

## What this package may assume of Latu

Latu **0.4.0**, pinned `~> 0.4`, exposes exactly five things for this package, and no
more:

- `Latu.Plan.new/1` wraps a `Relation` or a `Command` as a `Plan`.
- `Latu.Plan.relation/1` wraps a `rel_type` arm as a `Relation` with a fresh `plan_id`.
  **The one allocator. Never build a `%Proto.Relation{}` here.**
- `Latu.Client.execute_command/3` returns `{:ok, %Execution{ml_command_result: ...}}`.
- `Latu.Result.Literal.value/1`, which decodes a UDT-typed struct literal to
  `%Latu.Result.UDT{class, elements}` and asserts nothing about the elements. Asserting is this
  package's job.
- `Latu.Plan.normalize_ids/1`, which the golden harness uses. It is the twin of
  `normalize_plan_ids` in Latu's oracle; `dev/pyspark_oracle.py` imports that rather than
  copying it, so there are two implementations of the rule and not three.

Everything else in Latu that looks useful is either public API meant for users (`Latu.collect/2`,
`Latu.schema/1`, `Latu.to_nx/2`, `%Latu.DataFrame{}`, `%Latu.Error{}`) or private — reach for
those as a user would, from tests and docs, not from `lib/`. `Latu.Plan.lit/1` builds an
`Expression`, not the bare `Literal` `MlParams` needs, and its type inference is the wrong rule
anyway — see "Params" below. Do not reach for a `@doc false` function; if something is genuinely
missing, that is a Latu release, not a workaround here.

## Layering

```text
Latu.ML                          facade: fit, transform, attribute, delete
Latu.ML.{Estimator, Transformer, Model, SparseVector}   inert structs
Latu.ML.{Feature, Classification}                        constructors (generated from ML2)
Latu.ML.Linalg                   Vector/Matrix <-> Nx
Latu.ML.Plan                     MlCommand / MlRelation protos
```

**Only `Latu.ML.Plan` and `Latu.ML.Linalg` may name `Latu.Protocol.*`. Nothing here calls
`GRPC`.** `test/latu/ml/layering_test.exs` reads it off the BEAM import tables, so it sees calls
and not mentions.

## Params

`MlParams` is `map<string, Expression.Literal>` carrying **only what the caller set** — never
defaults; the server fills those in. Two stages, and PySpark does both:

1. **Coerce to the param's declared kind.** `reg_param: 1` is a double param, so it becomes
   `1.0`. This is `TypeConverters.toFloat` and friends, and it is why Latu's own literal
   inference cannot be reused: it would send `{:integer, 1}` and the server would refuse.
2. **Let magnitude pick the wire type**, exactly as Latu's `lit/1` does — an integer becomes
   `integer` or `long` by size, a float always `double`.

Client-side validation refuses the **kind**, never the range. `reg_param: :bad` is this
package's error; `reg_param: -1.0` is Spark's `ParamValidators`' error, and it is a better one.

## What is the oracle for what

- **Wire shape**: `dev/pyspark_oracle.py`, which monkeypatches `client.execute_command` to
  capture the command and raise. The proto PySpark *would* have sent is the fixture; never
  re-implement how PySpark builds one. `test/latu/ml/wire_test.exs` asserts against them
  through `Latu.ML.Internal`, which is what the facade calls — so a golden cannot pass by
  restating the implementation.
- **What the server accepts and answers**: the server. `dev/probe_ml.exs`, and Latu's
  `dev/probe_ml_server.exs`, which is where this package's first `Fit` came from.
- **Values a model returns**: nothing. Not this package's claim, and never was.

Anything only a live server can settle is named in the handover as the assertion that would have
to change.

## Restated from Latu, because they bite here

- **Functions and values, not magic.** No macro DSLs. Constructors are generated *functions*
  with docs, because tab completion is the discoverability surface.
- **Naming: Spark > Elixir > Scholar/Nx > scikit-learn.** `fit`, `transform`, `evaluate` are
  Spark's and Scholar's alike. No invented `predict/2`.
- **A refusal names the fix.** An unknown param prints the operator's params; a disallowed
  attribute prints that class's allowlist; a wrong kind prints the declared type.
- Lazy builders take a struct and return a struct. Actions return
  `{:ok, _} | {:error, %Latu.Error{}}` with `!` twins — `twins_test.exs` reads the specs.
- Max line width 98, `mix format` clean. `@moduledoc` on public modules, `@moduledoc false` on
  internals.
- **Every example is executed**: doctests for anything pure, guide fences for anything needing a
  server.
- Source files carry as little prose as possible; explanation goes in Markdown.
- `mix check` before committing; `mix check.all` when the servers are up. This repo's own
  `docker-compose.yml` runs both; integration uses the reattach profile (`:15003`). Latu's
  compose claims the same ports — one repo's servers at a time.
- Never `head` a grep you are using to decide something is finished.

## Working agreement

- The device shell has no docker, elixir, mix or network, and cannot delete files. Claude writes
  files; the maintainer runs them.
- Don't run `git status` from that shell — it leaves a `.git/index.lock` it cannot remove.
- One roadmap milestone at a time; sanity check at each boundary.
