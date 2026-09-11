defmodule Latu.ML.StatTest do
  use ExUnit.Case, async: true

  # `Summarizer`, offline: the wire shape against PySpark's, and the surface rules. The three
  # statistical tests are helper calls and live in `helper_test.exs`, against a server.

  alias Latu.ML.Stat
  alias Latu.ML.Wire

  describe "summary/3" do
    test "the metrics travel as an array of literals, the weight as 1.0 when absent" do
      Wire.base()
      |> Latu.select(v: Stat.summary(:features, [:mean, :count]))
      |> Wire.assert_wire("ml_stat_summary")
    end

    test "a weight column goes where the literal would" do
      Wire.base()
      |> Latu.select(v: Stat.summary(:features, [:norm_l2], weight: :w))
      |> Wire.assert_wire("ml_stat_summary_weighted")
    end

    test "a string names a column, as everywhere in this package" do
      assert Stat.summary("features", [:mean]) == Stat.summary(:features, [:mean])
    end

    test "an unknown metric is refused here, naming the ten" do
      assert_raise ArgumentError, ~r/unknown metric :median, expected one of \[:mean/, fn ->
        apply(Stat, :summary, [:features, [:median]])
      end

      assert_raise ArgumentError, ~r/unknown metric "mean"/, fn ->
        apply(Stat, :summary, [:features, ["mean"]])
      end
    end
  end

  describe "the ten shortcuts" do
    test "read one field out of a one-metric summary" do
      Wire.base()
      |> Latu.select(v: Stat.mean(:features))
      |> Wire.assert_wire("ml_stat_summary_field")
    end

    test "carry Spark's spelling of the metric, not Elixir's" do
      {:unresolved_extract_value, node} = Stat.num_nonzeros(:features).expr_type

      assert %{expr_type: {:literal, %{literal_type: {:string, "numNonzeros"}}}} = node.extraction
    end

    test "and Kernel's max/2 and min/2 are not in the way" do
      assert Stat.max(:features, weight: :w) == Stat.max(:features, weight: :w)
      assert %{expr_type: {:unresolved_extract_value, _}} = Stat.min(:features)
    end
  end
end
