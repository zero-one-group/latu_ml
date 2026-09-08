defmodule Latu.ML.Registry do
  @moduledoc false
  # What this package knows about MLlib, loaded once at compile time.
  #
  # Two committed files under `priv/`, both written by an extractor and neither hand-edited:
  #
  #   * `ml_operators.exs` — every operator PySpark exposes, with its JVM class, the model class
  #     a fit produces, and each param's name, wire spelling, declared type, default and doc.
  #   * `ml_attributes.exs` — the server's own `ALLOWED_ATTRIBUTES`, plus the union each
  #     concrete class inherits, how each attribute answers, and the `ConnectHelper` methods,
  #     which are in that allowlist but are attributes of nothing.
  #
  # `Latu.ML.Generate` reads this at *its* compile time to write the constructors and the
  # accessor modules; `Latu.ML.operators/1` and friends read it at run time. Nothing here
  # touches a session, a server or a proto — it is a data file with an index on it.
  #
  # Both files are `@external_resource`, so editing one recompiles what depends on it.

  alias Latu.ML.Operator
  alias Latu.ML.Param

  # The one home for the Spark version this package targets. Both registries stamp the version
  # they were extracted from, and a mismatch stops the compile rather than generating a surface
  # for a Spark nobody is running — the same guard Latu puts on its missing `function_docs.exs`.
  @spark "4.2.0"

  # PySpark's module grouping, as Elixir module names. Kept here rather than derived, because
  # one of them is a decision: `fpm` is Spark's abbreviation for frequent-pattern mining and
  # `Latu.ML.Fpm` reads like a typo, so it stays an acronym (`docs/decisions.md`, D12).
  @groups %{
    classification: Latu.ML.Classification,
    clustering: Latu.ML.Clustering,
    evaluation: Latu.ML.Evaluation,
    feature: Latu.ML.Feature,
    fpm: Latu.ML.FPM,
    recommendation: Latu.ML.Recommendation,
    regression: Latu.ML.Regression
  }

  @operators_path Path.expand("../../../priv/ml_operators.exs", __DIR__)
  @attributes_path Path.expand("../../../priv/ml_attributes.exs", __DIR__)

  # The third file is the only optional one, and deliberately: it is what a *server* said, and
  # the package has to build without one. Absent, every operator is `:built` — which is exactly
  # what a package with no probe result is entitled to claim.
  @probe_path Path.expand("../../../priv/ml_probe.exs", __DIR__)

  @external_resource @operators_path
  @external_resource @attributes_path
  @external_resource @probe_path

  read = fn path ->
    {%{spark: spark} = registry, _binding} = Code.eval_file(path)

    if spark != @spark do
      raise CompileError,
        description:
          "#{Path.basename(path)} was extracted from Spark #{spark}, and this package targets " <>
            "#{@spark}. Re-run its extractor, or move @spark in lib/latu/ml/registry.ex."
    end

    registry
  end

  attributes = read.(@attributes_path)
  probe = if File.exists?(@probe_path), do: read.(@probe_path).operators, else: %{}

  operators =
    for row <- read.(@operators_path).operators do
      {status, reason} =
        case Map.get(probe, row.name) do
          nil -> {:built, nil}
          result -> {result.status, result.reason}
        end

      %Operator{
        name: row.name,
        python: row.python,
        group: row.group,
        kind: row.kind,
        class: row.class,
        # A one-element list collapses to the single name every consumer already expects; `LDA`
        # is the one estimator that fits two classes, and its `model_class` stays nil so that
        # nothing claims to know which. Both are kept in `model_classes` either way.
        model_class:
          case Map.get(row, :model_classes) do
            [model_class] -> model_class
            _many_or_none -> nil
          end,
        model_classes: Map.get(row, :model_classes, []),
        constructor:
          if row.kind != :model do
            {Map.fetch!(@groups, row.group), row.name, 1}
          end,
        doc: Map.get(row, :doc),
        # Evaluator rows only; nil everywhere else, which is what `larger_better?/1` refuses on.
        larger_better: Map.get(row, :larger_better),
        spark: @spark,
        status: status,
        reason: reason,
        params:
          for param <- row.params do
            %Param{
              name: param.name,
              wire: param.wire,
              type: param.type,
              default: param.default,
              doc: param.doc
            }
          end
      }
    end

  # The attribute rows say which JVM class they belong to but not which group; the class name
  # says it already — `org.apache.spark.ml.<group>.<Name>` — so the group is read off the
  # package rather than stored twice.
  classes =
    for row <- attributes.classes do
      group = row.class |> String.split(".") |> Enum.at(-2) |> String.to_atom()

      Map.merge(row, %{
        group: group,
        module: Module.concat(Map.fetch!(@groups, group), row.python)
      })
    end

  # The `ConnectHelper` methods. `:python` is the call site each was read from — "Class.method"
  # — and both halves are load-bearing: the class links a helper to the operator whose method
  # it is, and the method name is what a generated constructor is called after.
  helpers =
    for row <- attributes.helpers do
      [owner, _function] = String.split(row.python, ".", parts: 2)

      Map.merge(row, %{owner: owner, args: row.args || []})
    end

  @operators Enum.sort_by(operators, & &1.name)
  @by_name Map.new(operators, &{&1.name, &1})
  @classes Enum.sort_by(classes, & &1.class)
  @by_class Map.new(classes, &{&1.class, &1})
  @by_operator_class Map.new(operators, &{&1.class, &1})
  @by_module Map.new(classes, &{&1.module, &1})
  @helpers Enum.sort_by(helpers, & &1.name)
  @by_helper Map.new(helpers, &{&1.name, &1})
  @helpers_by_owner Enum.group_by(helpers, & &1.owner)
  @helpers_by_model Enum.group_by(Enum.filter(helpers, & &1.model_class), & &1.model_class)

  @doc "The Spark version both registries were extracted from, and the only one they describe."
  @spec spark() :: String.t()
  def spark, do: @spark

  @doc "Every operator, sorted by name."
  @spec operators() :: [Operator.t()]
  def operators, do: @operators

  @doc "The Elixir module each PySpark group generates into."
  @spec groups() :: %{atom() => module()}
  def groups, do: @groups

  @doc "One operator by name."
  @spec fetch(atom()) :: {:ok, Operator.t()} | :error
  def fetch(name), do: Map.fetch(@by_name, name)

  @doc "One operator by name, or a refusal that lists the ones there are."
  @spec fetch!(atom()) :: Operator.t()
  def fetch!(name) do
    case fetch(name) do
      {:ok, operator} -> operator
      :error -> raise ArgumentError, "no operator #{inspect(name)}. " <> nearest(name)
    end
  end

  @doc """
  Every concrete model and summary class that has allowlisted attributes.

  Each is a map carrying its JVM `:class`, the `:module` of generated accessors, `:kind`,
  `:from` (the allowlist entries it inherits) and `:attributes`.
  """
  @spec classes() :: [map()]
  def classes, do: @classes

  @doc """
  The registry row for a JVM class name, an accessor module, or nil.

  Answers nil rather than raising for anything else: a summary whose class the server decides
  carries none, and a caller asking about a class this registry never heard of is asking a
  reasonable question with a plain answer.
  """
  @spec class(String.t() | module() | term()) :: map() | nil
  def class(name) when is_binary(name), do: Map.get(@by_class, name)
  def class(module) when is_atom(module), do: Map.get(@by_module, module)
  def class(_other), do: nil

  @doc """
  The operator row for a JVM class name, or nil.

  The reverse of the usual lookup, and what a helper needs: `stringIndexerModelFromLabels`
  builds a `StringIndexerModel` and the class name is all the answer carries, so the params a
  caller may then set are looked up by it.
  """
  @spec operator_of_class(String.t()) :: Operator.t() | nil
  def operator_of_class(class) when is_binary(class), do: Map.get(@by_operator_class, class)

  @doc """
  Every method on the server's own `ConnectHelper` object, sorted by name.

  Each is a map carrying `:name`, the `:method` that goes on the wire, how it `:returns`
  (`:value`, `:frame` or `:model`), the `:model_class` a `:model` one builds, its `:args` named
  and typed, and the `:python` call site all of that was read from.
  """
  @spec helpers() :: [map()]
  def helpers, do: @helpers

  @doc "One helper method by name, or nil."
  @spec helper(atom()) :: map() | nil
  def helper(name) when is_atom(name), do: Map.get(@by_helper, name)

  @doc "See `helper/1`. Raises on a miss, naming the ones there are."
  @spec helper!(atom()) :: map()
  def helper!(name) when is_atom(name) do
    helper(name) ||
      raise ArgumentError,
            "no helper method #{inspect(name)}. Latu.ML.helpers/0 lists the " <>
              "#{length(@helpers)} there are."
  end

  @doc """
  The helper methods that build a model of one JVM class.

  Three on 4.2.0, over two classes. They are what a generated model module's constructors come
  from: a `StringIndexerModel` can be had from a fit or from a list of labels, and only the
  first of those is a `Fit`.
  """
  @spec helpers_making(String.t()) :: [map()]
  def helpers_making(class) when is_binary(class), do: Map.get(@helpers_by_model, class, [])

  @doc """
  The helper methods that belong to one PySpark class, by its short name.

  How an operator finds its own: `PowerIterationClustering` neither fits nor transforms, and
  what it does instead is a method on the helper object.
  """
  @spec helpers_of(String.t()) :: [map()]
  def helpers_of(owner) when is_binary(owner), do: Map.get(@helpers_by_owner, owner, [])

  # A miss names the near ones rather than all 111: an unknown operator is nearly always a
  # spelling, and a wall of names is what a caller reads past.
  defp nearest(name) do
    text = to_string(name)

    case Enum.filter(Map.keys(@by_name), &(String.jaro_distance(to_string(&1), text) > 0.8)) do
      [] -> "Latu.ML.operators/0 lists them all."
      near -> "Did you mean one of #{inspect(Enum.sort(near))}?"
    end
  end
end
