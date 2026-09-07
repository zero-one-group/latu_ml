defmodule Latu.ML.FunctionsTest do
  use ExUnit.Case, async: true

  # Goldens, as everywhere else in this package: what `Latu.ML.Functions` builds against what
  # `pyspark.ml.functions` builds for the same call. These are the only fixtures that are plain
  # expressions rather than ML commands — the functions are not ML operators, they are names in
  # Spark's internal function registry.

  alias Latu.ML.Functions
  alias Latu.ML.Wire

  describe "vector_to_array" do
    test "the default dtype" do
      Wire.base()
      |> Latu.select(v: Functions.vector_to_array(:features))
      |> Wire.assert_wire("ml_functions_vector_to_array")
    end

    test "float32 is a different literal, not a different function" do
      Wire.base()
      |> Latu.select(v: Functions.vector_to_array(:features, "float32"))
      |> Wire.assert_wire("ml_functions_vector_to_array_float32")
    end

    test "a string names a column, where Latu's expression rule would make it a literal" do
      assert Functions.vector_to_array("features") == Functions.vector_to_array(:features)
    end

    test "a built expression goes through as it stands" do
      built = Latu.Column.col("features")

      assert Functions.vector_to_array(built) == Functions.vector_to_array(:features)
    end

    test "an unregistered dtype is refused here, naming both" do
      assert_raise ArgumentError, ~r/"float64" or "float32", not "float16"/, fn ->
        apply(Functions, :vector_to_array, [:features, "float16"])
      end
    end

    test "a dtype that is not a string is refused the same way" do
      assert_raise ArgumentError, ~r/not :float64/, fn ->
        apply(Functions, :vector_to_array, [:features, :float64])
      end
    end
  end

  describe "array_to_vector" do
    test "one argument, no dtype" do
      Wire.base()
      |> Latu.select(v: Functions.array_to_vector(:a))
      |> Wire.assert_wire("ml_functions_array_to_vector")
    end
  end

  test "dtypes/0 is the set the refusal names" do
    assert Functions.dtypes() == ["float64", "float32"]
  end
end
