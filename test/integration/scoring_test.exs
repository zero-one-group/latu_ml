defmodule Latu.ML.ScoringTest do
  use ExUnit.Case, async: true

  @moduletag :integration

  # Roadmap section 4, seam 2: "train at scale, predict on the BEAM". For the model families
  # where the parameters *are* the model, a fit on the cluster hands back tensors and the
  # scoring is a few lines of `defn` — no round trip, no model server, no export format.
  #
  # This is the claim under test, and the only honest way to test it is against **Spark's own
  # predictions**: both sides use the same coefficients, so a mismatch is this package's
  # arithmetic and nothing else. The guide lifts this example; it is a test first so the guide
  # cannot say something the server disagrees with.

  @url System.get_env("SPARK_REMOTE", "sc://localhost:15003")

  alias Latu.ML
  alias Latu.ML.Feature
  alias Latu.ML.Functions
  alias Latu.ML.Regression

  defmodule Linear do
    @moduledoc false
    import Nx.Defn

    # The whole of it. A linear model is a dot product and an intercept.
    defn predict(features, coefficients, intercept) do
      Nx.dot(features, coefficients) + intercept
    end
  end

  setup do
    session = Latu.connect!(@url)
    on_exit(fn -> Latu.disconnect(session, release: true) end)

    # Six rows, no two alike, so no two predictions are alike either — see the check below.
    training =
      Latu.sql!(session, """
      SELECT CAST(y AS DOUBLE) AS label, CAST(x1 AS DOUBLE) AS x1, CAST(x2 AS DOUBLE) AS x2
      FROM VALUES
        (7.0, 1.0, 1.0), (12.0, 2.0, 2.0), (17.0, 3.0, 3.0),
        (10.0, 1.0, 2.0), (14.0, 2.0, 3.0), (9.0, 3.0, 0.5)
      AS t(y, x1, x2)
      """)

    assembler = Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)

    %{session: session, features: ML.transform(assembler, training)}
  end

  test "a linear model fitted on Spark scores identically on the BEAM", %{features: features} do
    {:ok, model} = ML.fit(Regression.linear_regression(max_iter: 20), features)

    # The parameters, as tensors. This is the seam: nothing else crosses.
    {:ok, coefficients} = ML.attribute(model, :coefficients)
    {:ok, intercept} = ML.attribute(model, :intercept)

    assert Nx.shape(coefficients) == {2}
    assert Nx.type(coefficients) == {:f, 64}
    assert is_float(intercept)

    # **A bare Elixir float entering `defn` becomes an f32 tensor** — `Nx.tensor(2.0)` defaults
    # to f32 — and an f32 intercept costs about 1e-7 of precision. `Latu.ML.Linalg` builds the
    # coefficients as f64, so only the intercept needs saying; this assertion is what caught it,
    # and it is the first thing the guide has to warn about.
    intercept = Nx.tensor(intercept, type: :f64)
    assert Nx.type(intercept) == {:f, 64}

    # Spark's own predictions, beside the feature rows they were computed from, in one collect
    # so the two cannot fall out of order. `vector_to_array` is what makes a Vector column
    # readable at all — `Latu.collect/2` refuses one (`docs/deviations.md`).
    {:ok, rows} =
      model
      |> ML.transform(features)
      |> Latu.select([:prediction, x: Functions.vector_to_array(:features)])
      |> Latu.collect()

    spark = Enum.map(rows, & &1.prediction)

    # A check whose expected value equals the broken value is not a check: if every prediction
    # were the same, this would pass with the coefficients ignored entirely.
    assert length(Enum.uniq(spark)) == length(spark)
    assert length(spark) == 6

    ours =
      rows
      |> Enum.map(& &1.x)
      |> Nx.tensor(type: :f64)
      |> Linear.predict(coefficients, intercept)
      |> Nx.to_flat_list()

    assert length(ours) == length(spark)

    for {mine, theirs} <- Enum.zip(ours, spark) do
      assert_in_delta mine, theirs, 1.0e-10
    end

    :ok = ML.delete(model)
  end

  test "the parameters outlive the model, which the cache does not", %{features: features} do
    {:ok, model} = ML.fit(Regression.linear_regression(max_iter: 20), features)
    {:ok, coefficients} = ML.attribute(model, :coefficients)
    {:ok, intercept} = ML.attribute(model, :intercept)
    intercept = Nx.tensor(intercept, type: :f64)

    :ok = ML.delete(model)

    # The handle is gone — this is the whole reason seam 2 is worth having. The tensors are
    # ordinary BEAM terms and score just as well with no session at all.
    assert {:error, error} = ML.attribute(model, :coefficients)
    assert error.error_class == "CONNECT_ML.CACHE_INVALID"

    scored =
      [[1.0, 1.0], [2.0, 2.0]]
      |> Nx.tensor(type: :f64)
      |> Linear.predict(coefficients, intercept)

    assert Nx.shape(scored) == {2}
    assert Nx.to_flat_list(scored) |> Enum.uniq() |> length() == 2
  end
end
