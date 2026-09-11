defmodule Latu.ML.HelperTest do
  use ExUnit.Case, async: true

  @moduletag :integration

  # ML4b against a live server: the `ConnectHelper` route, which is the one object in the
  # server's attribute allowlist that is an attribute of nothing. No model, no cache entry, and
  # arguments that are positional and complete.
  #
  # Every method here is one PySpark reaches through `invoke_helper_attr` or
  # `invoke_helper_relation`, and the argument types come from those call sites — so what this
  # proves that `wire_test.exs` cannot is that the server accepts what the call sites said.

  @url System.get_env("SPARK_REMOTE", "sc://localhost:15003")

  alias Latu.ML
  alias Latu.ML.Clustering
  alias Latu.ML.Feature
  alias Latu.ML.FPM
  alias Latu.ML.Stat

  setup do
    session = Latu.connect!(@url)
    on_exit(fn -> Latu.disconnect(session, release: true) end)

    labelled =
      Latu.sql!(session, """
      SELECT
        CAST(label AS DOUBLE) AS label,
        CAST(x1 AS DOUBLE) AS x1,
        CAST(x2 AS DOUBLE) AS x2,
        category
      FROM VALUES
        (0.0, 0.0, 1.1, 'a'), (0.0, 0.1, 1.0, 'b'), (1.0, 2.0, 0.1, 'a'), (1.0, 2.1, 0.2, 'c')
      AS t(label, x1, x2, category)
      """)

    assembler = Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)

    %{session: session, labelled: labelled, features: ML.transform(assembler, labelled)}
  end

  describe "values" do
    test "the locale a StopWordsRemover would default to", %{session: session} do
      assert {:ok, locale} = Feature.default_locale(session)
      assert is_binary(locale)
    end

    test "and the stop words for a language", %{session: session} do
      assert {:ok, words} = Feature.load_default_stop_words(session, "german")
      assert is_list(words)
      assert "aber" in words

      # The pair is what `stop_words_remover/1` cannot default: both are the server's to say,
      # which is why the registry records those two params as `:server_supplied`.
      remover =
        Feature.stop_words_remover(input_col: :words, output_col: :kept, stop_words: words)

      assert {"stopWords", {:list, :string}, ^words} =
               List.keyfind(remover.params, "stopWords", 0)
    end

    test "an unknown language is the server's refusal", %{session: session} do
      assert {:error, error} = Feature.load_default_stop_words(session, "klingon")
      assert error.kind == :rpc
    end
  end

  describe "models the server builds out of what you send" do
    test "a StringIndexerModel from labels, which then transforms", context do
      %{session: session, labelled: labelled} = context

      assert {:ok, model} =
               Feature.StringIndexerModel.from_labels(session, ["a", "b", "c"],
                 input_col: :category,
                 output_col: :category_index
               )

      # A cache entry like a fit's, and the uid is the one the client sent with the call.
      assert is_binary(model.ref)
      assert model.uid =~ ~r/^StringIndexerModel_[0-9a-f]{12}$/
      assert model.class == "org.apache.spark.ml.feature.StringIndexerModel"

      indexed = ML.transform(model, labelled)

      assert {:ok, rows} = indexed |> Latu.select([:category, :category_index]) |> Latu.collect()
      assert Enum.map(rows, & &1.category_index) == [0.0, 1.0, 0.0, 2.0]

      # The labels it was built from are readable back off the model, which is the point of
      # building one this way.
      assert {:ok, ["a", "b", "c"]} = ML.attribute(model, :labels)
      assert :ok = ML.delete(model)
    end

    test "a CountVectorizerModel from a vocabulary", %{session: session} do
      assert {:ok, model} =
               Feature.CountVectorizerModel.from_vocabulary(session, ["a", "b"],
                 input_col: :words,
                 output_col: :counts
               )

      assert {:ok, ["a", "b"]} = ML.attribute(model, :vocabulary)
      assert :ok = ML.delete(model)
    end

    # The one nested list in the surface: a list of lists of strings, typed explicitly at
    # PySpark's own call site because a value cannot say what an empty list holds.
    test "and one from arrays of labels", %{session: session} do
      assert {:ok, model} =
               Feature.StringIndexerModel.from_arrays_of_labels(session, [["a", "b"], ["c"]],
                 input_cols: [:one, :two],
                 output_cols: [:one_index, :two_index]
               )

      assert is_binary(model.ref)
      assert :ok = ML.delete(model)
    end
  end

  describe "the statistical tests" do
    test "chi-square, flattened so the answer is collectable", %{features: features} do
      flat = Stat.chi_square_test(features, :features, :label, true)

      assert {:ok, rows} = Latu.collect(flat)
      assert length(rows) == 2
      assert Enum.all?(rows, &is_float(&1.pValue))
    end

    # Unflattened it answers with Vectors, which is the column Latu cannot decode — so the
    # schema is as far as this can go, and that is a real round trip through the planner.
    test "and unflattened it answers with Vectors", %{features: features} do
      assert {:ok, schema} = Latu.schema(Stat.chi_square_test(features, :features, :label))
      assert Enum.map(schema, & &1.name) == ["pValues", "degreesOfFreedom", "statistics"]
    end

    test "correlation, whose one column is a Matrix", %{features: features} do
      assert {:ok, schema} = Latu.schema(Stat.correlation(features, :features))
      assert length(schema) == 1
    end

    test "Kolmogorov-Smirnov, with its parameters as a list", %{labelled: labelled} do
      result = Stat.kolmogorov_smirnov_test(labelled, :x1, "norm", [0.0, 1.0])

      assert {:ok, [row]} = Latu.collect(result)
      assert is_float(row.pValue)
      assert is_float(row.statistic)
    end
  end

  describe "Summarizer" do
    alias Latu.Column
    alias Latu.ML.Functions

    # x1 is [0.0, 0.1, 2.0, 2.1] and x2 [1.1, 1.0, 0.1, 0.2] in the setup above.
    test "one metric at a time, read back through vector_to_array", %{features: features} do
      {:ok, [row]} =
        features
        |> Latu.agg(
          count: Stat.count(:features),
          mean: Functions.vector_to_array(Stat.mean(:features))
        )
        |> Latu.collect()

      assert row.count == 4
      assert [x1, x2] = row.mean
      assert_in_delta x1, 1.05, 1.0e-9
      assert_in_delta x2, 0.6, 1.0e-9
    end

    test "several metrics in one struct, with a weight column", %{features: features} do
      {:ok, [row]} =
        features
        |> Latu.with_columns(w: 2.0)
        |> Latu.agg(s: Stat.summary(:features, [:count, :max], weight: :w))
        |> Latu.select(
          count: Column.get_field(:s, :count),
          max: Functions.vector_to_array(Column.get_field(:s, :max))
        )
        |> Latu.collect()

      assert row.count == 4
      assert [x1, x2] = row.max
      assert_in_delta x1, 2.1, 1.0e-9
      assert_in_delta x2, 1.1, 1.0e-9
    end
  end

  describe "the two operators that are neither fitted nor applied" do
    test "power iteration clustering assigns a cluster per vertex", %{session: session} do
      edges =
        Latu.sql!(session, """
        SELECT CAST(src AS LONG) AS src, CAST(dst AS LONG) AS dst, CAST(w AS DOUBLE) AS weight
        FROM VALUES (0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0), (4, 5, 1.0)
        AS t(src, dst, w)
        """)

      pic = Clustering.power_iteration_clustering(k: 2, max_iter: 10)
      assigned = ML.assign_clusters(pic, edges)

      assert {:ok, rows} = Latu.collect(assigned)
      assert length(rows) == 6
      assert Enum.all?(rows, &is_integer(&1.cluster))
    end

    test "prefix span finds frequent sequences", %{session: session} do
      sequences =
        Latu.sql!(session, """
        SELECT array(array(CAST(a AS INT)), array(CAST(b AS INT))) AS sequence
        FROM VALUES (1, 2), (1, 2), (1, 3), (2, 3)
        AS t(a, b)
        """)

      span = FPM.prefix_span(min_support: 0.5, max_pattern_length: 3)
      patterns = ML.find_frequent_sequential_patterns(span, sequences)

      assert {:ok, rows} = patterns |> Latu.select([:freq]) |> Latu.collect()
      assert Enum.all?(rows, &is_integer(&1.freq))
    end
  end

  describe "the generic verbs underneath" do
    test "helper/3 sends every declared argument, uid included", %{session: session} do
      uid = "StringIndexerModel_deadbeef0000"

      assert {:ok, model} =
               ML.helper(session, :string_indexer_model_from_labels, [uid, ["x", "y"]])

      assert model.uid == uid
      assert :ok = ML.delete(model)
    end

    test "and a wrong count is refused before anything is sent", %{session: session} do
      assert_raise ArgumentError, ~r/takes 2 arguments — uid \(:string\), labels/, fn ->
        ML.helper(session, :string_indexer_model_from_labels, [["x"]])
      end
    end
  end
end
