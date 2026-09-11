defmodule Latu.ML.Stat do
  @moduledoc """
  `pyspark.ml.stat`: three tests that run where the data is, and `Summarizer`.

  The tests are chi-square, correlation and Kolmogorov-Smirnov. None of them fits anything and
  none caches anything — each is one method on the server's own helper object, and each hands
  back a lazy `Latu.DataFrame` with a row or two in it. Nothing reaches the server until you
  collect.

  `summary/3` and its ten shortcuts are `Summarizer`: aggregate statistics over a `Vector`
  column, as column expressions for `Latu.agg/2`.

  Most of this reads a `features` column, which is a `Vector` — and a Vector column cannot
  be collected on 4.2.0 (`docs/deviations.md`). The tests' *results* are ordinary columns, so
  that only bites where you collect the frame you passed in rather than the one you got back.
  A summary's vector-valued metrics are Vectors too; `Latu.ML.Functions.vector_to_array/2`
  reads them.

  `Latu.ML.helper/3` is the layer underneath the tests, and `Latu.ML.helpers/0` the table they
  are generated from — except that these three are not generated: a flat module cannot have two
  functions called `test`, so they are named here.
  """

  # `max/2` and `min/2` are Spark's names for two of the ten metrics, and `Kernel`'s for two
  # unrelated functions; nothing in this module needs the latter.
  import Kernel, except: [max: 2, min: 2]

  alias Latu.Column
  alias Latu.DataFrame
  alias Latu.Functions, as: F

  @typedoc "A column, as anywhere in Latu."
  @type column :: atom() | String.t()

  @typedoc "One of `Summarizer`'s ten metrics, in Elixir spelling."
  @type metric ::
          :mean
          | :sum
          | :variance
          | :std
          | :count
          | :num_nonzeros
          | :max
          | :min
          | :norm_l2
          | :norm_l1

  # Spark's spellings, which are also the struct's field names. Measured from PySpark's
  # `Summarizer.metrics` docstring and `dev/probe_ml_functions.exs`.
  @metrics [
    mean: "mean",
    sum: "sum",
    variance: "variance",
    std: "std",
    count: "count",
    num_nonzeros: "numNonzeros",
    max: "max",
    min: "min",
    norm_l2: "normL2",
    norm_l1: "normL1"
  ]

  @doc """
  Pearson's independence test for every feature against a label.

  A lazy builder. `features_col` is a `Vector` column and `label_col` any numeric one; both are
  treated as categorical, one contingency table per feature.

  With `flatten: false` — the default — the answer is a single row of `pValues`,
  `degreesOfFreedom` and `statistics`, one entry per feature. With `flatten: true` it is one
  row per feature, carrying `featureIndex`, `pValue`, `degreesOfFreedom` and `statistic`.

      Latu.ML.Stat.chi_square_test(features, :features, :label, true) |> Latu.collect()
  """
  @spec chi_square_test(DataFrame.t(), column(), column(), boolean()) :: DataFrame.t()
  def chi_square_test(%DataFrame{} = data, features_col, label_col, flatten \\ false) do
    Latu.ML.helper_frame(:chi_square_test, [data, features_col, label_col, flatten])
  end

  @doc """
  The correlation matrix of a column of `Vector`s.

  A lazy builder. `method` is `"pearson"` or `"spearman"`; the answer is one row and one
  column, named for the method and the column, holding a `Matrix`.

  Spearman is a rank correlation and needs a sort per column, so cache the input before asking
  for one if the lineage is expensive to recompute.

      Latu.ML.Stat.correlation(features, :features)
  """
  @spec correlation(DataFrame.t(), column(), String.t()) :: DataFrame.t()
  def correlation(%DataFrame{} = data, column, method \\ "pearson") do
    Latu.ML.helper_frame(:correlation, [data, column, method])
  end

  @doc """
  A one-sample, two-sided Kolmogorov-Smirnov test against a theoretical distribution.

  A lazy builder. `dist_name` is `"norm"` — the only one Spark supports — and `params` are that
  distribution's, mean then standard deviation. The answer is one row of `pValue` and
  `statistic`.

      Latu.ML.Stat.kolmogorov_smirnov_test(sample, :value, "norm", [0.0, 1.0])

  PySpark takes those parameters as varargs (`test(df, col, "norm", 0.0, 1.0)`); here they are
  a list, because a trailing list reads better in Elixir than a trailing splat.
  """
  @spec kolmogorov_smirnov_test(DataFrame.t(), column(), String.t(), [number()]) :: DataFrame.t()
  def kolmogorov_smirnov_test(%DataFrame{} = data, sample_col, dist_name, params) do
    Latu.ML.helper_frame(:kolmogorov_smirnov_test, [data, sample_col, dist_name, params])
  end

  # =============================================
  # Summarizer
  # =============================================

  @doc """
  Summary statistics over a `Vector` column, as one struct. Spark's `Summarizer`.

      alias Latu.ML.Stat

      Latu.agg(scored, s: Stat.summary(:features, [:mean, :count]))

  An aggregate expression, so it belongs in `Latu.agg/2`. The struct carries one field per
  metric under Spark's own spelling (`numNonzeros`, `normL2`), and `Latu.Column.get_field/2`
  reads one out; `mean/2` and its nine siblings do that for a single metric in one call.

  Every metric but `:count` is a `Vector`, which `Latu.collect/2` refuses. Read it through
  `Latu.ML.Functions.vector_to_array/2` or `Latu.to_nx/2`:

      Latu.agg(scored, mean: Functions.vector_to_array(Stat.mean(:features)))

  The wire is `aggregate_metrics`, a function in Spark's internal registry with the metric
  names as an array of literals; an unknown name is refused here, naming the ten.

  ## Options

    * `:weight` — a column, or a number, weighting each row. Defaults to `1.0`, as PySpark's
      does.
  """
  @spec summary(column(), [metric()], keyword()) :: Latu.Plan.expression()
  def summary(features, metrics, opts \\ []) when is_list(metrics) do
    opts = Keyword.validate!(opts, weight: 1.0)
    names = Enum.map(metrics, &metric!/1)
    weight = to_column(opts[:weight])

    Column.fun("aggregate_metrics", [F.array(names), to_column(features), weight])
  end

  for {name, spark} <- @metrics do
    @doc """
    `Summarizer`'s `#{spark}` of a `Vector` column, as one expression.

    `summary/3` for the one metric, with its `#{spark}` field read out. Same `:weight` option.
    """
    @spec unquote(name)(column(), keyword()) :: Latu.Plan.expression()
    def unquote(name)(features, opts \\ []) do
      Column.get_field(summary(features, [unquote(name)], opts), unquote(spark))
    end
  end

  defp metric!(metric) do
    with true <- is_atom(metric), {:ok, name} <- Keyword.fetch(@metrics, metric) do
      name
    else
      _ ->
        raise ArgumentError,
              "unknown metric #{inspect(metric)}, expected one of " <>
                inspect(Keyword.keys(@metrics))
    end
  end

  # A binary names a column here, as everywhere in this package; a number is the literal it is.
  defp to_column(name) when is_binary(name), do: Column.col(name)
  defp to_column(other), do: other
end
