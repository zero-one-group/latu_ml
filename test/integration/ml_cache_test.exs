defmodule Latu.ML.CacheTest do
  @moduledoc false
  use ExUnit.Case, async: true

  @moduletag :integration

  # What the server's ML cache does under a budget it cannot meet. Measured by
  # `dev/probe_eviction.exs` on 2026-09-05; this is the half worth keeping as a gate, because
  # ML3's error mapping is built on it and a Spark upgrade could move any of it.
  #
  # The memory-control confs are **session-scoped and read live** — the probe set them
  # mid-session and the very next fit obeyed, while a fresh session still saw the defaults. So
  # these tests can run concurrently against the shared server: each has its own session, and
  # nothing here leaks to another.

  alias Latu.ML
  alias Latu.ML.Classification
  alias Latu.ML.Feature

  @url System.get_env("SPARK_REMOTE", "sc://localhost:15003")
  @prefix "spark.connect.session.connectML.mlCache.memoryControl"

  setup do
    session = Latu.connect!(@url)
    on_exit(fn -> Latu.disconnect(session, release: true) end)

    base =
      Latu.sql!(session, """
      SELECT * FROM VALUES (0.0, 0.0, 1.1), (0.0, 0.1, 1.0), (1.0, 2.0, 0.1), (1.0, 2.1, 0.2)
      AS t(label, x1, x2)
      """)

    assembler = Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)

    %{session: session, features: ML.transform(assembler, base)}
  end

  # The bang twin on purpose: a budget that silently failed to apply would make every test in
  # this module pass for the wrong reason, which is the one failure mode they exist to rule out.
  defp budget(session, suffix, value) do
    Latu.set_conf!(session, "#{@prefix}.#{suffix}", value)
  end

  defp fit(features), do: ML.fit(Classification.logistic_regression(max_iter: 2), features)

  # The finding that matters most for ML3: a model pushed out of memory is **not gone**. The
  # server writes it to driver-local disk and a fetch still answers, with nothing the client can
  # see beyond latency. So a `%Latu.ML.Model{}` does not go stale under memory pressure — only
  # an explicit delete or the end of the session ends one.
  test "a model over maxInMemorySize is offloaded, not lost", ctx do
    budget(ctx.session, "maxInMemorySize", "1b")

    assert {:ok, model} = fit(ctx.features)
    assert {:ok, coefficients} = ML.attribute(model, :coefficients)
    assert Nx.shape(coefficients) == {2}

    # And again, to prove the first fetch did not consume it.
    assert {:ok, ^coefficients} = ML.attribute(model, :coefficients)
    assert :ok = ML.delete(model)
  end

  test "a model over maxModelSize is refused at the fit, by its own class", ctx do
    budget(ctx.session, "maxModelSize", "1b")

    assert {:error, error} = fit(ctx.features)
    assert error.error_class == "CONNECT_ML.MODEL_SIZE_OVERFLOW_EXCEPTION"
  end

  test "a session over maxStorageSize is refused at the fit, by a different class", ctx do
    budget(ctx.session, "maxStorageSize", "1b")

    assert {:error, error} = fit(ctx.features)
    assert error.error_class == "CONNECT_ML.ML_CACHE_SIZE_OVERFLOW_EXCEPTION"
  end

  # The confs are the whole reason the three tests above can be `async: true`. If this ever goes
  # red they are no longer isolated, and the rest of this module is suspect.
  test "a budget set on one session does not reach another", ctx do
    budget(ctx.session, "maxModelSize", "1b")

    other = Latu.connect!(@url)
    on_exit(fn -> Latu.disconnect(other, release: true) end)

    assert {:ok, "1073741824b"} = Latu.conf(other, "#{@prefix}.maxModelSize")
  end
end
