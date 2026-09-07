defmodule Latu.ML.Stat do
  @moduledoc """
  Statistical tests that run where the data is: chi-square, correlation, Kolmogorov-Smirnov.

  `pyspark.ml.stat`'s three, in its own grouping. None of them fits anything and none caches
  anything — each is one method on the server's own helper object, and each hands back a lazy
  `Latu.DataFrame` with a row or two in it. Nothing reaches the server until you collect.

  Two of the three read a `features` column, which is a `Vector` — and a Vector column cannot
  be collected on 4.2.0 (`docs/deviations.md`). Their *results* are ordinary columns, so this
  only bites where you collect the frame you passed in rather than the one you got back.

  `Latu.ML.helper/3` is the layer underneath, and `Latu.ML.helpers/0` the table these are
  generated from — except that these three are not generated: a flat module cannot have two
  functions called `test`, so they are named here.
  """

  alias Latu.DataFrame

  @typedoc "A column, as anywhere in Latu."
  @type column :: atom() | String.t()

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
end
