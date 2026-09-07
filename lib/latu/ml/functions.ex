defmodule Latu.ML.Functions do
  @moduledoc """
  Converting between a `Vector` column and an ordinary array of numbers.

  `pyspark.ml.functions`' two, and the way to read a `Vector` column: `Latu.collect/2` and
  `Latu.to_explorer/2` both refuse one, because Spark describes it as a UDT that does not say
  its SQL type (`docs/deviations.md`).

      alias Latu.ML.Functions

      scored = Latu.ML.transform(model, features)

      scored
      |> Latu.select([:prediction, features: Functions.vector_to_array(:features)])
      |> Latu.collect()

  Both are lazy: each builds a column expression and reaches no server.

  Spark keeps these in an **internal** function registry rather than the builtin one, so they
  are unreachable from SQL — `Latu.sql/3` on `SELECT vector_to_array(…)` answers
  `UNRESOLVED_ROUTINE` on any 4.2.0 server, and that is by design rather than a gap. Over
  Connect they resolve, which is what these two build.

  Both are UDFs on the server rather than compiled expressions, so they cost a pass over the
  rows, and `vector_to_array` densifies a sparse `Vector` rather than refusing it.
  """

  alias Latu.Column

  @dtypes ~w(float64 float32)

  @typedoc """
  A column.

  An atom or a string names one; anything else is used as the expression it is, so a built
  column such as `Latu.Column.fun/3`'s result goes straight through.
  """
  @type column :: atom() | String.t() | Latu.Plan.expression()

  @doc """
  A `Vector` column as an array of numbers.

  `dtype` is `"float64"` — the default — for an `array<double>`, or `"float32"` for an
  `array<float>`. Dense and sparse vectors alike; a sparse one comes back dense.

      Functions.vector_to_array(:features)
      Functions.vector_to_array(:features, "float32")

  The dtype is checked here rather than on the server, which would answer
  `INVALID_PARAMETER_VALUE.DTYPE` a round trip later.
  """
  @spec vector_to_array(column(), String.t()) :: Latu.Plan.expression()
  def vector_to_array(column, dtype \\ "float64") do
    Column.fun("vector_to_array", [to_column(column), dtype!(dtype)])
  end

  @doc """
  An array of numbers as a dense `Vector` column.

  The input is an `array<double>`; the answer is a `Vector` every ML operator accepts, and one
  `Latu.collect/2` refuses like any other.

      Functions.array_to_vector(:embedding)

  `VectorAssembler` is the other way to build one, and the better one where the features are
  already separate columns — it takes them without a round trip through an array.
  """
  @spec array_to_vector(column()) :: Latu.Plan.expression()
  def array_to_vector(column) do
    Column.fun("array_to_vector", [to_column(column)])
  end

  @doc "The dtypes `vector_to_array/2` takes, which are the two Spark registers."
  @spec dtypes() :: [String.t()]
  def dtypes, do: @dtypes

  defp dtype!(dtype) when dtype in @dtypes, do: dtype

  defp dtype!(other) do
    raise ArgumentError,
          "vector_to_array's dtype is " <>
            Enum.map_join(@dtypes, " or ", &inspect/1) <> ", not #{inspect(other)}"
  end

  # A binary names a column here, as it does everywhere this package asks for one.
  # `Latu.Plan.to_expr/1` would read it as a string literal instead.
  defp to_column(name) when is_binary(name), do: Column.col(name)
  defp to_column(other), do: other
end
