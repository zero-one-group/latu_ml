# Probe: a `Fit` long enough that the server cuts the stream under it.
#
#     docker compose up -d spark-reattach
#     mix run dev/probe_reattach_fit.exs
#
# The last thing ML1 named and could not settle. `spark-reattach` runs with
# `senderMaxStreamDuration=5s`: the server ends the response stream after five seconds of
# sending and the client must reattach to collect the rest. A `Fit` is one `ExecutePlan` whose
# only payload is the final result, so a fit that outlasts five seconds exercises the path where
# a client receives progress, an EOF without `ResultComplete`, and then the real answer on a
# second stream.
#
# What is under test is **Latu's**, not this package's: `ml_command_result` is latched with a
# `%{ml_command_result: nil}` guard so the first one wins and a replay after a reattach cannot
# clobber it (Latu M15.1). `execution_test.exs` proves that as a state machine; nothing has ever
# proved it against a server, because nothing before `latu_ml` could send an `MlCommand` at all.
#
# Deliberately not a test. It costs seconds, the property belongs to Latu, and a gate that slow
# would tax every run of a suite that finishes in about one.

alias Latu.ML
alias Latu.ML.Classification
alias Latu.ML.Feature

defmodule ProbeReattach do
  def say(label, value) do
    IO.puts("  " <> String.pad_trailing(label, 40) <> to_string(value))
  end

  def heading(text) do
    IO.puts("\n" <> text <> "\n" <> String.duplicate("-", String.length(text)))
  end

  @doc """
  A frame deep enough to make the fit slow, from Spark's own `range` table function.

  SQL rather than the verbs, because the shape here is incidental and a plan built from one
  string cannot be wrong about a function name.

  The label is a **threshold on x2**, deliberately. The first version of this probe used
  `id % 2`, which is not a linear function of either feature — the fit converged to
  `[0.0, 0.0]`, correctly, and a zero vector is poor evidence that the model handed back after
  a reattach is the one that was fitted. A threshold is linearly learnable, so the coefficients
  come back distinctive and the last step of this probe means something.

  No fixture files and no `create_dataframe` — a big `range` keeps the *request* small and the
  work on the server, which is what has to be slow.
  """
  def features(session, rows) do
    base =
      Latu.sql!(session, """
      SELECT CAST(id % 7 > 3 AS DOUBLE) AS label,
             CAST(id AS DOUBLE)         AS x1,
             CAST(id % 7 AS DOUBLE)     AS x2
      FROM range(0, #{rows})
      """)

    assembler = Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)

    ML.transform(assembler, base)
  end

  def timed(fun) do
    {micros, result} = :timer.tc(fun)

    {div(micros, 1000), result}
  end

  def describe({:ok, %ML.Model{} = model}), do: "ok — obj_ref " <> model.ref
  def describe({:ok, value}), do: "ok — " <> inspect(value, printable_limit: 60, limit: 3)

  def describe({:error, %Latu.Error{} = error}) do
    "REFUSED " <> inspect(error.error_class) <> " — " <> String.slice(error.message, 0, 100)
  end

  def describe(other), do: inspect(other, printable_limit: 80)
end

# The reattach profile is the whole point; :15002 would prove nothing.
url = System.get_env("SPARK_REMOTE", "sc://localhost:15003")
rows = String.to_integer(System.get_env("ROWS", "20000000"))
iterations = String.to_integer(System.get_env("MAX_ITER", "100"))

session = Latu.connect!(url)

IO.puts("Spark #{Latu.spark_version!(session)} — long-fit reattach probe (#{url})")

ProbeReattach.heading("1. The server's stream duration")

ProbeReattach.say(
  "senderMaxStreamDuration",
  case Latu.conf(session, "spark.connect.execute.reattachable.senderMaxStreamDuration") do
    {:ok, nil} -> "(unset — is this really spark-reattach?)"
    {:ok, value} -> value
    {:error, error} -> "refused: " <> String.slice(error.message, 0, 50)
  end
)

ProbeReattach.heading("2. A fit that should outlast it")

ProbeReattach.say("rows", rows)
ProbeReattach.say("max_iter", iterations)

features = ProbeReattach.features(session, rows)
estimator = Classification.logistic_regression(max_iter: iterations, reg_param: 0.001)

{fit_ms, fitted} = ProbeReattach.timed(fn -> ML.fit(estimator, features) end)

ProbeReattach.say("fit", "#{fit_ms} ms — " <> ProbeReattach.describe(fitted))

cond do
  match?({:error, _}, fitted) ->
    ProbeReattach.say("verdict", "the fit failed; nothing about the latch was learned")

  fit_ms < 5_000 ->
    ProbeReattach.say(
      "verdict",
      "INCONCLUSIVE — finished in under 5 s, so the stream was never cut. " <>
        "Raise ROWS or MAX_ITER and run again."
    )

  true ->
    ProbeReattach.say("verdict", "the fit outlasted the stream, so a reattach happened")
end

ProbeReattach.heading("3. Is the latched result the real one?")

case fitted do
  {:ok, model} ->
    # The assertion that matters. A replay that clobbered the latch would leave a ref naming
    # nothing, or naming an object whose attributes do not answer.
    {ms, attribute} = ProbeReattach.timed(fn -> ML.attribute(model, :coefficients) end)

    ProbeReattach.say("fetch coefficients", "#{ms} ms — " <> ProbeReattach.describe(attribute))

    # A zero vector would mean the fit learned nothing, which makes this step weak evidence
    # rather than wrong. The label is linearly learnable precisely so this reads non-zero.
    ProbeReattach.say(
      "learned something?",
      case attribute do
        {:ok, tensor} ->
          if Nx.to_flat_list(tensor) == [0.0, 0.0] do
            "NO — all-zero coefficients, so the fetch proves little. Check the label."
          else
            "yes — " <> inspect(Nx.shape(tensor))
          end

        _other ->
          "n/a"
      end
    )

    ProbeReattach.say("delete", ProbeReattach.describe(ML.delete(model)))

  _other ->
    ProbeReattach.say("fetch coefficients", "skipped, no model")
end

IO.puts("")
Latu.disconnect(session, release: true)
