# Probe: what the ML cache does when it runs out of room, and what a client sees.
#
#     docker compose up -d spark-connect
#     mix run dev/probe_eviction.exs
#
# Reconnaissance for ML3, run during ML1 because it is cheap now and would otherwise be ML3's
# first day. The roadmap treats "evicted" as one state. It is at least three, and they are not
# interchangeable:
#
#   * **offloaded** — the model is over `maxInMemorySize` or idle past `offloadingTimeout`, so
#     the server wrote it to driver-local disk. Still there. A fetch should work and cost more.
#   * **gone** — the object gets dropped rather than offloaded, and a fetch is an error. This is
#     what a *summary* does past the timeout: the CACHE_INVALID message says so in as many
#     words, and it is why PySpark keeps the training frame to rebuild one.
#   * **refused** — the model never entered the cache, because it is over `maxModelSize` or the
#     session is over `maxStorageSize`. The error arrives at `fit`, not at the fetch.
#
# The load-bearing question comes first, because every other answer here depends on it: **does
# the cache re-read a conf mid-session?** `isModifiable` says the *key* is settable per session
# (M15 probe); it says nothing about whether the cache looks again after it has started. If it
# does not, every step below passes for the wrong reason.
#
# Each step reports and continues, so one refusal does not cost the others.

alias Latu.ML
alias Latu.ML.Classification
alias Latu.ML.Feature
alias Latu.Protocol.Spark.Connect, as: Proto

defmodule ProbeEviction do
  @prefix "spark.connect.session.connectML.mlCache.memoryControl"

  def say(label, value) do
    IO.puts("  " <> String.pad_trailing(label, 44) <> to_string(value))
  end

  def heading(text) do
    IO.puts("\n" <> text <> "\n" <> String.duplicate("-", String.length(text)))
  end

  def key(suffix), do: "#{@prefix}.#{suffix}"

  def conf(session, suffix) do
    case Latu.conf(session, key(suffix)) do
      {:ok, nil} -> "(unset)"
      {:ok, value} -> value
      {:error, error} -> "refused: " <> String.slice(error.message, 0, 50)
    end
  end

  def set(session, suffix, value) do
    case Latu.set_conf(session, key(suffix), value) do
      :ok -> :ok
      {:error, error} -> {:error, String.slice(error.message, 0, 70)}
    end
  end

  def describe({:ok, value}), do: "ok — " <> inspect(value, printable_limit: 60, limit: 3)

  def describe({:error, %Latu.Error{} = error}) do
    "REFUSED " <> inspect(error.error_class) <> " — " <> String.slice(error.message, 0, 100)
  end

  def describe(:ok), do: "ok"
  def describe(other), do: inspect(other, printable_limit: 80)

  @doc "Milliseconds a function took, beside its result."
  def timed(fun) do
    {micros, result} = :timer.tc(fun)

    {div(micros, 1000), result}
  end

  # Not on `Latu.ML.Plan` yet — cache control is ML3's surface. A probe may reach for the
  # protos directly; the library may not.
  def model_size(session, ref) do
    command =
      {:get_model_size,
       %Proto.MlCommand.GetModelSize{model_ref: %Proto.ObjectRef{id: ref}}}

    send_ml(session, command)
  end

  def cache_info(session) do
    send_ml(session, {:get_cache_info, %Proto.MlCommand.GetCacheInfo{}})
  end

  defp send_ml(session, command) do
    ml = %Proto.MlCommand{command: command}
    plan = Latu.Plan.new(%Proto.Command{command_type: {:ml_command, ml}})

    with {:ok, execution} <- Latu.Client.execute_command(session, plan) do
      Latu.ML.Result.param(execution)
    end
  end

  def features(session) do
    base =
      Latu.sql!(session, """
      SELECT * FROM VALUES (0.0, 0.0, 1.1), (0.0, 0.1, 1.0), (1.0, 2.0, 0.1), (1.0, 2.1, 0.2)
      AS t(label, x1, x2)
      """)

    assembler = Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)

    ML.transform(assembler, base)
  end

  def fit(session) do
    ML.fit(Classification.logistic_regression(max_iter: 2), features(session))
  end
end

url = System.get_env("SPARK_REMOTE", "sc://localhost:15002")
session = Latu.connect!(url)
IO.puts("Spark #{Latu.spark_version!(session)} — ML cache eviction probe (#{url})")

suffixes = ~w(enabled maxInMemorySize offloadingTimeout maxModelSize maxStorageSize)

# =============================================
# 1. Does a conf set mid-session stick, and is it read back?
# =============================================

ProbeEviction.heading("1. The confs as the session starts")

for suffix <- suffixes, do: ProbeEviction.say(suffix, ProbeEviction.conf(session, suffix))

ProbeEviction.heading("2. Setting one mid-session: does it stick?")

before = ProbeEviction.conf(session, "maxInMemorySize")
result = ProbeEviction.set(session, "maxInMemorySize", "1b")

ProbeEviction.say("set maxInMemorySize to 1b", ProbeEviction.describe(result))
ProbeEviction.say("read back", ProbeEviction.conf(session, "maxInMemorySize"))
ProbeEviction.say("was", before)

# =============================================
# 3. A model under a 1-byte memory budget: offloaded, or refused?
# =============================================

ProbeEviction.heading("3. Fit with maxInMemorySize=1b — offloaded, or refused?")

{fit_ms, fitted} = ProbeEviction.timed(fn -> ProbeEviction.fit(session) end)
ProbeEviction.say("fit", "#{fit_ms} ms — " <> ProbeEviction.describe(fitted))

case fitted do
  {:ok, model} ->
    size = ProbeEviction.model_size(session, model.ref)
    ProbeEviction.say("model size", ProbeEviction.describe(size))

    {ms, attribute} = ProbeEviction.timed(fn -> ML.attribute(model, :coefficients) end)
    ProbeEviction.say("fetch coefficients", "#{ms} ms — " <> ProbeEviction.describe(attribute))

    {ms2, again} = ProbeEviction.timed(fn -> ML.attribute(model, :coefficients) end)
    ProbeEviction.say("fetch again", "#{ms2} ms — " <> ProbeEviction.describe(again))

    ProbeEviction.say("cache info", ProbeEviction.describe(ProbeEviction.cache_info(session)))
    ProbeEviction.say("delete", ProbeEviction.describe(ML.delete(model)))

  _other ->
    ProbeEviction.say("fetch coefficients", "skipped, no model")
end

# =============================================
# 4. Idle past offloadingTimeout
# =============================================

ProbeEviction.heading("4. Fit, then idle past offloadingTimeout")

timeout = ProbeEviction.set(session, "offloadingTimeout", "1s")

ProbeEviction.say("set offloadingTimeout to 1s", ProbeEviction.describe(timeout))
ProbeEviction.say("read back", ProbeEviction.conf(session, "offloadingTimeout"))

case ProbeEviction.fit(session) do
  {:ok, model} ->
    Process.sleep(4_000)

    {ms, attribute} = ProbeEviction.timed(fn -> ML.attribute(model, :coefficients) end)
    ProbeEviction.say("fetch after 4 s idle", "#{ms} ms — " <> ProbeEviction.describe(attribute))
    ProbeEviction.say("delete", ProbeEviction.describe(ML.delete(model)))

  error ->
    ProbeEviction.say("fit", ProbeEviction.describe(error))
end

# =============================================
# 5. A model larger than maxModelSize, and a session over maxStorageSize
# =============================================

ProbeEviction.heading("5. Budgets that should refuse the fit outright")

restored = ProbeEviction.set(session, "offloadingTimeout", "900000ms")
ProbeEviction.say("restore offloadingTimeout", ProbeEviction.describe(restored))

small_model = ProbeEviction.set(session, "maxModelSize", "1b")
ProbeEviction.say("set maxModelSize to 1b", ProbeEviction.describe(small_model))
ProbeEviction.say("fit", ProbeEviction.describe(ProbeEviction.fit(session)))

big_model = ProbeEviction.set(session, "maxModelSize", "1g")
ProbeEviction.say("set maxModelSize to 1g", ProbeEviction.describe(big_model))

small_store = ProbeEviction.set(session, "maxStorageSize", "1b")
ProbeEviction.say("set maxStorageSize to 1b", ProbeEviction.describe(small_store))
ProbeEviction.say("fit", ProbeEviction.describe(ProbeEviction.fit(session)))

# =============================================
# 6. Whether any of it survives a fresh session
# =============================================

ProbeEviction.heading("6. A fresh session, to see what was session-scoped")

fresh = Latu.connect!(url)
for suffix <- suffixes, do: ProbeEviction.say(suffix, ProbeEviction.conf(fresh, suffix))
Latu.disconnect(fresh, release: true)

IO.puts("")
Latu.disconnect(session, release: true)
