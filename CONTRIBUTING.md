# Contributing

**[Latu's `CONTRIBUTING.md`](https://hexdocs.pm/latu/contributing.html) applies here in full.**
Same conventions, same rhythm, same gates. What follows is the delta, which is mostly the two
extractors and the probe.

Requires Elixir ~> 1.20, Docker, and Python 3 for the plan oracle and the extractors.

```bash
mix deps.get
docker compose up -d --wait                                  # both Spark servers

mix check      # format + warnings-as-errors + offline tests
mix check.all  # the same, plus the integration suite and `mix docs`
```

Both halt on the first failure and exit non-zero. Integration here uses the **reattach** profile
on `:15003`; `mix check` needs no server at all. Latu's `docker-compose.yml` claims the same two
ports and carries confs only its own tests use, so bring up one repo's servers at a time — the
container names differ (`latu-ml-spark-*`) so that `docker ps` says which.

The oracle is the one thing that still wants Latu beside this repo: `dev/pyspark_oracle.py`
imports Latu's plan-id normaliser rather than copying it, because a third copy of that rule is
how the three drift. `dev/README.md` has the detail, including `LATU_DEV`.

Conventions — naming, the shapes a builder and an action return, how errors are worded, how a
verb is documented — live in `CLAUDE.md`, and Latu's applies underneath it. Per-decision
rationale is `docs/decisions.md`, a dated log that does not ship. Every place the public API
departs from `pyspark.ml` is [`docs/deviations.md`](docs/deviations.md), which does.

## What is generated, and from what

Most of this package's surface is not written by hand. Three committed files in `priv/` carry
it, all stamped with the Spark version they came from — a mismatch fails the compile:

  * **`priv/ml_operators.exs`** — every class under `pyspark.ml`: its params, their converters,
    their defaults, their docs, and the model class each estimator fits. Written by
    `dev/extract_ml_operators.py`.
  * **`priv/ml_attributes.exs`** — the server's own `ALLOWED_ATTRIBUTES` from `MLUtils.scala`,
    with each concrete class's union resolved and each attribute marked with how it answers and
    what arguments it takes. Written by `dev/extract_ml_allowlist.py`.
  * **`priv/ml_probe.exs`** — what a live server actually did with each operator. The one file
    that may be absent: the package compiles without it and everything is then `:built`, which
    is what a package with no probe result has earned.

`Latu.ML.Generate` reads all three at compile time and defines the constructors, the accessor
modules and their docs. **Do not hand-edit any of the three.** Regenerate:

```bash
python3 dev/extract_ml_operators.py --pyspark <tree> --write
python3 dev/extract_ml_allowlist.py --write
```

`--check` instead of `--write` reproduces them and fails on a diff; that is the drift check to
run at a re-vendor, and running `--write` twice is what proves determinism.

## What is the oracle for what

  * **Wire shape** — `dev/pyspark_oracle.py`, which monkeypatches `client.execute_command` to
    capture the `Command` and raise. The proto PySpark *would* have sent is the fixture; nothing
    about how PySpark builds one is re-implemented. `test/latu/ml/wire_test.exs` asserts against
    those fixtures through `Latu.ML.Internal`, which is what the facade calls — so a golden
    cannot pass by restating the implementation.
  * **What the server accepts and answers** — the server. `dev/probe_ml.exs` fits, applies or
    evaluates every registered operator and fetches every allowlisted attribute;
    `dev/probe_ml_functions.exs`, `dev/probe_eviction.exs` and `dev/probe_ext.exs` cover the
    corners. A probe result records **how the question was asked**, not only the answer.
  * **Values a model returns** — nothing. Not this package's claim, and never was.
  * **The on-disk format** — round-tripping with PySpark. `dev/README.md` has the interop dance:
    two clients, one server, one directory, each reading what the other wrote.

`dev/example.exs` walks the whole surface with a server up (`mix run dev/example.exs`). It is a
smoke test, not the source of the README's snippets.

## Documentation is executed

An example that is not executed is a lie with a shelf life, so every example is run one of three
ways — doctests for anything pure, `elixir` fences in `docs/guides/*.md` for anything needing a
server, and `test/latu/examples_test.exs` for the rest, which cannot run an example but proves
every function it names exists at the arity shown.

Five of the tests are checks on the repo rather than on the package, and are worth knowing about
before you trip one:

  * `test/latu/docs_test.exs` — every public function in a documented module has a `@doc`; every
    documented module has a `groups_for_modules` entry, by name **or** by the regex that covers
    the 64 generated ones; every guide is in `:extras` and its Guides group; and every relative
    link resolves *and* to the file ExDoc will pick, which matches by basename alone.
  * `test/latu/examples_test.exs` — every code block in every docstring and prose doc parses,
    and every qualified call and `Mod.fun/arity` span resolves at that arity. It also insists
    that every markdown file in the repo is checked, run as a guide, or excused there by name.
  * `test/latu/cheatsheet_test.exs` — every hand-written verb is on the cheatsheet, and the
    cheatsheet names no verb that has gone.
  * `test/latu/ml/twins_test.exs` — every action has a `!` twin of the same arity, read off the
    specs.
  * `test/latu/ml/layering_test.exs` — only `Latu.ML.Plan`, `Latu.ML.Linalg` and
    `Latu.ML.Result` name `Latu.Protocol.*`, and nothing calls `GRPC`. Read off the BEAM import
    tables, so it sees calls rather than mentions.

A guide's fences run in order, threading one binding. A fence that **cannot** run against the
test server is preceded by a visible line beginning

    > **Not executed.**

with the reason on the same line; `guides_test.exs` asserts the whole skipped set, so a new one
fails until somebody writes down why. **A guide that fits a model deletes it** — these run async
beside every other integration test, and a page that leaks a cache entry is a page that makes
another test's `cache_info/1` wrong.

## Adding a page

Four hand-maintained lists, and a check on each:

  1. `mix.exs` `:extras`, and `:groups_for_extras` if it is a guide or reference.
  2. `test/latu/docs_test.exs`'s `@prose_extras`, unless it is a guide.
  3. `test/latu/examples_test.exs`'s `@extras`, or its `@excused` with a reason.
  4. A link from the README's "Where to go next", if a reader would want it.

And one trap that is not a list: **naming a `@moduledoc false` module in prose needs a
`skip_code_autolink_to` entry in `mix.exs`.** `mix docs` runs with `warnings_as_errors`, so a
page that explains an internal — as this one explains the generator and the golden harness —
fails the build until the module is named there. `examples_test.exs` still resolves the
reference, because it reads real exports rather than what is documented.

## What Latu owes this package

Exactly five things, listed in `CLAUDE.md`. If something else is genuinely missing, that is a
**Latu release** — a new version, a `CHANGELOG.md` entry, a `~> ` bump here — not a workaround
in `lib/`. Reaching for a `@doc false` function in Latu is not the fix.

## CI

One job, `ci`, on `ubuntu-24.04`: `docker compose up -d`, then the BEAM from `.tool-versions`
while the JVMs warm, then `mix deps.get --check-locked`, a forced `MIX_ENV=test` compile, `up -d
--wait`, and `mix check.all`. Secret-free, and every command is one you run locally. On failure
it dumps the last 200 lines of both Spark logs, because a red integration test is usually a
server that did not come up.

Two differences from Latu's, both subtractions: no `mix fixtures` (this suite builds its frames
from SQL `VALUES`, so there are no data files) and no venv (the goldens under `test/wire/` are
committed, so CI never runs the oracle).

`main` requires the `ci` job. Changes go branch → PR → merge; required approvals is 0.

## Release checklist

Maintainer only, and Latu's is the model. Four things are specific to this package, and the
first three are pre-flight for the **first** release — **in this order**, because each unblocks
the next:

- [ ] the repository goes **public**
  - it is private until then, as Latu's was. Nothing below works while it is not: a
    `raw.githubusercontent.com` image 404s, a hexdocs "view source" link 404s, and Latu cannot
    link here.
- [ ] the README's relative links and its lockup image become absolute
  - `docs/` and `assets/` are not in the Hex tarball, so on **hex.pm** a relative link 404s and
    the logo does not render. Latu's README is absolute for exactly this reason; this one is
    relative while there is nothing to point at.
- [ ] `mix.exs`'s `@source_url` and the `:links` map name a repository that exists
- [ ] Latu's three mentions of MLlib being a separate package become links
  - Latu's `README.md`, `usage-rules.md` and `docs/guides/from-pyspark.md`. That is a Latu
    release of its own, so it is the last step rather than the first.

Then Latu's own list, unchanged: `CHANGELOG.md` folded and dated, `@version` matching, the
README's install snippet naming the new minor, `mix check.all` green with both servers up,
`mix hex.build` read, working tree clean, `git tag v<version>` pushed **before**
`mix hex.publish`, and the versioned URLs checked within the hour.
