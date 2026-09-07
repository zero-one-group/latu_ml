defmodule Latu.ML.VerticalSliceTest do
  use ExUnit.Case, async: true

  @moduletag :integration

  # ML1's slice, end to end against a live server: assemble, fit, transform, collect a
  # prediction, read a coefficient as a tensor, delete. Latu's compose file provides the
  # servers; this runs on the reattach profile, which is the integration default.
  #
  # What is deliberately NOT proved here: a `Fit` long enough to cross the profile's
  # `senderMaxStreamDuration` and force a reattach. Four rows finish in well under five
  # seconds, so the latch's replay guard is still untested — see the ML1 handover.

  @url System.get_env("SPARK_REMOTE", "sc://localhost:15003")

  alias Latu.ML
  alias Latu.ML.Classification
  alias Latu.ML.Feature
  alias Latu.ML.Functions

  setup do
    session = Latu.connect!(@url)
    on_exit(fn -> Latu.disconnect(session, release: true) end)

    training =
      Latu.sql!(session, """
      SELECT * FROM VALUES (0.0, 0.0, 1.1), (0.0, 0.1, 1.0), (1.0, 2.0, 0.1), (1.0, 2.1, 0.2)
      AS t(label, x1, x2)
      """)

    assembler = Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)

    %{session: session, features: ML.transform(assembler, training)}
  end

  test "a transform is lazy: the schema answers without executing", %{features: features} do
    assert {:ok, schema} = Latu.schema(features)
    assert Enum.map(schema, & &1.name) == ["label", "x1", "x2", "features"]
  end

  test "fit, predict, read a coefficient, delete", %{features: features} do
    lr = Classification.logistic_regression(max_iter: 5, reg_param: 0.01)

    assert {:ok, model} = ML.fit(lr, features)
    assert is_binary(model.ref)
    assert model.uid == lr.uid

    # Every Vector column is dropped before collecting — see the test below for why.
    scored = ML.transform(model, features)

    assert {:ok, rows} = scored |> Latu.select([:label, :prediction]) |> Latu.collect()
    assert length(rows) == 4
    assert Enum.all?(rows, &is_float(&1.prediction))

    # A VectorUDT literal, through Latu's decoder and this package's Linalg.
    assert {:ok, coefficients} = ML.attribute(model, :coefficients)
    assert Nx.shape(coefficients) == {2}
    assert Nx.type(coefficients) == {:f, 64}

    assert {:ok, intercept} = ML.attribute(model, :intercept)
    assert is_float(intercept)

    assert :ok = ML.delete(model)

    # The cache key is gone, and the server says so rather than answering with a stale model.
    #
    # Match the **class**, not the message. A deleted ref answers `CONNECT_ML.CACHE_INVALID`,
    # where M15's probe on a ref that never existed got a bare `INTERNAL_ERROR` — so the two
    # cases do differ, which is what ML3 needed to know. The message is no use to match on:
    # Spark calls the missing thing a "Summary object" and suggests `model.evaluate(dataset)`
    # whatever kind of object actually went missing, and this was a model.
    assert {:error, error} = ML.attribute(model, :coefficients)
    assert error.error_class == "CONNECT_ML.CACHE_INVALID"
  end

  test "with_model deletes even when the body raises", %{features: features} do
    lr = Classification.logistic_regression(max_iter: 1)
    parent = self()

    assert_raise RuntimeError, "boom", fn ->
      ML.with_model(lr, features, fn model ->
        send(parent, {:ref, model.ref})
        raise "boom"
      end)
    end

    assert_received {:ref, _ref}
  end

  # Measured 2026-09-05, and it contradicts what the roadmap assumed: on 4.2.0 the server
  # describes a `features` column as a UDT **without** a `sql_type`, so Latu's dtype guard
  # refuses it rather than guessing a layout. `Latu.ML.Functions.vector_to_array/2` is the way
  # through and `Latu.to_arrow/2` the other; `Latu.to_nx/2` is ML6's. This test is the finding:
  # if Spark ever populates `sql_type`, it goes red and we learn.
  test "a Vector column cannot be collected, and Latu says exactly why", %{features: features} do
    assert {:error, error} = Latu.collect(features)
    assert error.kind == :decode
    assert error.message =~ "column features is a UDT that does not say its SQL type"
  end

  # The other half of that finding, measured 2026-09-06: the column is unreadable, and this is
  # how it is read. A gate rather than probe output, because it is Spark behaviour and can move.
  #
  # The SQL leg is the surprising one, and the reason the roadmap was wrong about this for five
  # milestones: `vector_to_array` lives in Spark's *internal* function registry, which the
  # parser never searches, so a name that works over the wire cannot be spelled in SQL at all.
  # `docs/deviations.md`.
  test "vector_to_array reads what collect refuses", %{session: session, features: features} do
    assert {:ok, rows} =
             features
             |> Latu.select(v: Functions.vector_to_array(:features))
             |> Latu.collect()

    assert Enum.sort(Enum.map(rows, & &1.v)) ==
             Enum.sort([[0.0, 1.1], [0.1, 1.0], [2.0, 0.1], [2.1, 0.2]])

    assert {:error, error} = Latu.sql(session, "SELECT vector_to_array(array(1.0), 'float64')")
    assert error.error_class == "UNRESOLVED_ROUTINE"
  end

  test "an unknown attribute is the server's refusal, naming its class", %{features: features} do
    lr = Classification.logistic_regression(max_iter: 1)

    result =
      ML.with_model(lr, features, fn model ->
        ML.attribute(model, :notAnAttribute)
      end)

    assert {:error, error} = result
    assert error.kind == :rpc

    # A deliberate probe rather than a settled fact: roadmap section 2.4 reads
    # `CONNECT_ML.ATTRIBUTE_NOT_ALLOWED` off `MLUtils.scala`, and nothing has confirmed it over
    # the wire. If this goes red the failure prints the class the server really sent, which is
    # the cheapest way to find out.
    assert error.error_class == "CONNECT_ML.ATTRIBUTE_NOT_ALLOWED"
  end

  # The wire shape is pinned offline (`test/latu/ml/wire_test.exs`); what only a server can say
  # is that the resolved name is one its allowlist actually answers to. Both spellings of the
  # same attribute must give the same number, and `dev/example.exs` reaching for the snake one
  # is what turned this from a nicety into a bug report.
  test "a snake_case attribute name is one the server answers", %{features: features} do
    lr = Classification.logistic_regression(max_iter: 5)

    result =
      ML.with_model(lr, features, fn model ->
        summary = ML.summary(model)

        with {:ok, snake} <- ML.attribute(summary, :area_under_roc),
             {:ok, wire} <- ML.attribute(summary, "areaUnderROC"),
             do: {:ok, {snake, wire}}
      end)

    assert {:ok, {snake, wire}} = result
    assert is_float(snake)
    assert snake == wire
  end
end
