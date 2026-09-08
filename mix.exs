defmodule Latu.ML.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/zero-one-group/latu_ml"

  def project do
    [
      app: :latu_ml,
      version: @version,
      elixir: "~> 1.20",
      start_permanent: Mix.env() == :prod,
      # Latu's reason: any file the incremental compiler touches is held to it, rather than
      # forcing a full recompile per run with `--force --warnings-as-errors`.
      elixirc_options: elixirc_options(Mix.env()),
      # test/support is loaded by test_helper.exs, not by the test loader.
      test_ignore_filters: [&String.starts_with?(&1, "test/support/")],
      deps: deps(),
      aliases: aliases(),
      description: "Spark MLlib from Elixir, over Spark Connect. A companion package to Latu.",
      package: package(),
      docs: docs(),
      source_url: @source_url
    ]
  end

  defp elixirc_options(:prod), do: []
  defp elixirc_options(_env), do: [warnings_as_errors: true]

  # `check` and `check.all` run `mix test`, which refuses to run outside the test env.
  def cli do
    [preferred_envs: [check: :test, "check.all": :test]]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:latu, "~> 0.3"},
      # Hard, not optional: a fitted model's attributes come back as tensors and there is no
      # useful surface here without them. Roadmap section 6.
      {:nx, "~> 0.13"},
      {:ex_doc, "~> 0.36", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
      licenses: ["Apache-2.0"],
      links: %{
        "GitHub" => @source_url,
        "Changelog" => "https://hexdocs.pm/latu_ml/changelog.html",
        "Latu" => "https://hexdocs.pm/latu",
        "MLlib" => "https://spark.apache.org/docs/latest/ml-guide.html"
      },
      # `priv/` is not optional: the registries are read at compile time, so a package without
      # them does not build. `usage-rules.md` ships because tooling reads it from the dep;
      # `assets/` does not, because nothing in the package reads it.
      files: ~w(lib priv .formatter.exs mix.exs README.md LICENSE CHANGELOG.md
               usage-rules.md)
    ]
  end

  defp docs do
    [
      name: "Latu ML",
      main: "readme",
      # The README opens with the lockup image, not an `# Latu ML` heading, so ExDoc is told the
      # title. The tile is the sidebar logo and the favicon; assets/README.md has the system.
      logo: "assets/latu-ml-avatar.svg",
      favicon: "assets/latu-ml-avatar.svg",
      assets: %{"assets" => "assets"},
      # Sidebar order. The guides run from tour to reference; `test/latu/docs_test.exs` checks
      # that this list and `docs/guides/` agree, in both directions.
      extras: [
        {"README.md", title: "Latu ML"},
        "docs/guides/quick-start.md",
        "docs/guides/cookbook.md",
        "docs/guides/from-pyspark-ml.md",
        "docs/guides/with-scholar-and-nx.md",
        "docs/cheatsheet.cheatmd",
        "usage-rules.md",
        "docs/deviations.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md"
      ],
      # Guides are matched by directory rather than listed, so the wildcard in `docs_test.exs`
      # is the list. Reference is named: its order is the sidebar's and there is no directory
      # that means "reference".
      groups_for_extras: [
        Guides: ~r/docs\/guides\//,
        Reference: ["docs/cheatsheet.cheatmd", "usage-rules.md", "docs/deviations.md"]
      ],
      # Two knobs, not interchangeable, and Latu's `mix.exs` explains why: prose references
      # consult `skip_code_autolink_to` with the **term**; typespecs consult only
      # `skip_undefined_reference_warnings_on`, by module — never the term. So a spec naming a
      # generated protocol type can only be silenced per module.
      #
      # `Latu.Protocol.*` and `Latu.Client` are `@moduledoc false` in Latu, on purpose. Public
      # specs name `Latu.ML.Plan`'s own type aliases instead, so the only place a protocol type
      # is still written out is where those aliases are *defined*.
      #
      # The `@moduledoc false` modules **this package's own docs name** are here too, and they
      # are named on purpose: `CONTRIBUTING.md` explains the generator and the golden harness,
      # which means writing `Latu.ML.Generate` and `Latu.ML.Internal` down. ExDoc has no page to
      # link and says so once per mention. A prose reference is checked by
      # `test/latu/examples_test.exs` either way, which resolves against real exports rather
      # than against what is documented.
      skip_code_autolink_to: fn term ->
        String.starts_with?(term, "Latu.Protocol.") or String.starts_with?(term, "Latu.Client") or
          String.starts_with?(term, "Latu.ML.Generate") or
          String.starts_with?(term, "Latu.ML.Internal")
      end,
      # What this costs, stated plainly: an undefined *type* in a spec on this one module would
      # not be reported here. `elixirc_options`' `warnings_as_errors` still catches it — an
      # unknown type in a `@spec` is a compile warning.
      skip_undefined_reference_warnings_on: fn id -> id == "Latu.ML.Plan" end,
      warnings_as_errors: true,
      groups_for_modules: [
        # `Latu.ML.Stat` and `Latu.ML.Functions` are verb modules too, in PySpark's own
        # grouping rather than on the facade: three statistical tests that run where the data
        # is, and the two column functions that convert a `Vector` to an array and back.
        Verbs: [Latu.ML, Latu.ML.Functions, Latu.ML.Stat],
        # The seven generated constructor modules, one per PySpark group.
        Operators: [
          Latu.ML.Classification,
          Latu.ML.Clustering,
          Latu.ML.Evaluation,
          Latu.ML.Feature,
          Latu.ML.FPM,
          Latu.ML.Recommendation,
          Latu.ML.Regression
        ],
        # Sixty-four generated modules: one per fitted-model class (42) and one per summary
        # class (22), the two counts `dev/check_offline.exs` asserts.
        # Matched by shape rather than listed: they are generated, so a list here would be a
        # second place to keep in step with `priv/ml_attributes.exs`. The pattern is
        # deliberately narrow — a group module itself has no second dot after `Latu.ML.`.
        "Model and summary attributes": [~r/^Latu\.ML\.\w+\.\w+$/],
        # Inert data a verb takes or returns. You read these; you rarely name one.
        "Plans and carriers": [
          Latu.ML.Estimator,
          Latu.ML.Transformer,
          Latu.ML.Evaluator,
          Latu.ML.Helper,
          Latu.ML.CrossValidator,
          Latu.ML.CrossValidatorModel,
          Latu.ML.Model,
          Latu.ML.Pipeline,
          Latu.ML.PipelineModel,
          Latu.ML.Summary,
          Latu.ML.TrainValidationSplit,
          Latu.ML.TrainValidationSplitModel,
          Latu.ML.Operator,
          Latu.ML.Param,
          Latu.ML.SparseVector,
          Latu.ML.Plan
        ],
        Results: [Latu.ML.Linalg, Latu.ML.Result]
      ],
      source_url: @source_url,
      source_ref: "v#{@version}"
    ]
  end

  defp aliases do
    [
      # `--warnings-as-errors` because `mix test` compiles test files itself, outside
      # `elixirc_options`; without it a warning in test/ is the one kind CI lets through.
      check: [
        "format --check-formatted",
        "compile",
        "test --warnings-as-errors"
      ],
      # `mix docs` is spawned through `env` because this alias resolves under `:test` (the
      # `preferred_envs` above) and `ex_doc` is `only: :dev`.
      "check.all": [
        "format --check-formatted",
        "compile",
        "test --include integration --warnings-as-errors",
        "cmd env MIX_ENV=dev mix docs"
      ]
    ]
  end
end
