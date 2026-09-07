# Probe: third-party JVM operators through a services shim, against a live 4.2.0 server.
#
#     bash dev/fetch_jars.sh                                    # once: tmp/jars/*.jar
#     docker compose up -d --wait spark-connect
#     mix run dev/probe_ext.exs                                 # upload route: shim + jars
#     mix run dev/probe_ext.exs -- --offload                    # also force an offload (~80 s)
#     mix run dev/probe_ext.exs -- --second-session             # also load the native lib twice
#     mix run dev/probe_ext.exs -- --poison                     # also a shim naming no real class
#
#     docker compose -f dev/docker-compose.ext.yml up -d --wait
#     SPARK_REMOTE=sc://localhost:15004 mix run dev/probe_ext.exs -- --mounted
#
# `latu-ml-third-party.md` (the project) is the scoping this probes, and §1 there is the reading
# of the server's own source that every question below comes from. Nothing in `lib/` changes for
# this; the script talks to the stub itself where Latu has no public verb yet, and if what it
# measures holds, the script is the design for `Latu.add_jar/2`.
#
# ## The questions, in the order they are asked
#
#   1. **Does a services jar uploaded through `AddArtifacts` under `jars/` reach the ML
#      `ServiceLoader`?** The server builds a per-session classloader from session jars and runs
#      every plan under it; `MLUtils.loadOperators` reads that loader on each call. If so, a
#      third-party operator needs no server restart and no administrator. `--mounted` asks the
#      other half: jars an admin placed with `--jars`, and only the shim from the client — and
#      first, a fit with **no** shim, to show the services file is the gate.
#   2. **Do jars built for Spark 3.5 (XGBoost) and 4.1 (isolation-forest) run on 4.2.0?** The
#      precedent is SPARK-52259, which broke XGBoost4J on 4.0.0 with a `NoSuchMethodError` at the
#      first fit. That is where it would show here too.
#   3. **Which of the six inherited attributes answer**, and which of XGBoost's throw by design.
#   4. **Does memory control's save-at-fit survive a third-party writer?** `MLCache.register`
#      writes every model to driver disk as it is registered. isolation-forest writes its nodes
#      as Avro through a DataFrame write, so its fit needs `spark-avro` on the server and is the
#      case for a writer that does not know about local saving mode. Asked with memory control
#      on, then off in a session of its own.
#   5. **Does a model come back from an offload?** `--offload`: `offloadingTimeout=1m` before the
#      first fit, a fit, seventy seconds, a fetch and a transform — the `loadFromLocal` path
#      through a third-party reader.
#   6. **Does a second session loading the same native library work?** `--second-session`.
#   7. **What does a shim naming a class no jar carries do to the session?** `--poison`. The
#      reading of `ServiceLoader.stream()` says `Provider.type()` throws for it, which would take
#      every ML operator in the session down with it — and nothing removes an artifact from a
#      session once it is there.
#
# Every step is reported, none is asserted: a refusal is a finding. Save paths are on the
# **server's** filesystem; the one artefact read back — XGBoost's JSON booster — comes through
# `binaryFile`, and a copy lands in `tmp/xgb_booster.json` for an EXGBoost attempt later.
#
# ## First run, 2026-09-07 — the finding that shaped the shim
#
# The upload route reaches the loader: the `--mounted` no-shim fit said `Unsupported ML operator`,
# and uploading the shim changed the error, so the services file is the gate and a client can
# post it. But the first shim listed model classes under the Transformer service, and every fit
# died at its `VectorAssembler` with `IsolationForestModel ... Unable to get public no-arg
# constructor` (question 4's mechanism, found the hard way): a model class has no no-arg ctor,
# and one unresolvable provider fails the whole `ServiceLoader` enumeration the server runs on
# every transform. The shim now lists **estimators only** (see `@libraries`).
#
# ## Second run, 2026-09-07 — the estimator-only shim, and the executor wall
#
# The assembler resolves now, and both fits reach real training and fail *there*, not in
# discovery: isolation-forest with `ClassNotFound ... MutableURLClassLoader` (an executor task
# classloader), XGBoost with `StreamCorruptedException: invalid type code: AC` (a task result
# deserialized under the wrong classloader). Same cause: the driver's `ServiceLoader` found and
# built the estimator, but these libraries launch **distributed Spark jobs** whose task closures
# reference their own classes, and a session-uploaded `addJar` does not reliably reach the task
# classloader. So client upload discovers and instantiates on the driver; a distributed trainer
# also needs its jar on the executors.
#
# ## Mounted run, 2026-09-07 — the mechanism works, and XGBoost is the one that does not
#
# With the jars on `--jars` (driver *and* executors) and the shim from the client,
# **isolation-forest runs end to end**: fit, transform to real scores, `toString`, `model_size`,
# `save`, `delete`, with `numFeatures` refused (`ATTRIBUTE_NOT_ALLOWED`, a plain `Model`) and
# `load` refused (`Unsupported read`) exactly as the allowlist and the missing services entry
# predict. So the division of labour is: **library jars on the cluster, the 347-byte shim from
# the client** — and the shim is still needed even when the jars are mounted, because a library
# jar carries no services file.
#
# **XGBoost4J-Spark 3.4.0 does not work on 4.2.0**, and the cause is exact:
# `NoSuchMethodError` for `org.json4s.Extraction.decompose` returning `JsonAST$JValue` — the
# json4s 3.x signature Spark 4.x dropped. `CustomGeneralParam.jsonEncode` calls it, and
# `SparkParams` defaults `customObj` and `customEval` to `null`, which Spark's metadata writer
# encodes on every save. It therefore fires on *any* save, whatever the caller passed. Training
# itself completed: the run left two orphaned cache entries, one per failed fit, which only exist
# because `MLCache.register` caches the model and saves it afterwards. Sections 4b (memory
# control off) and 5 (offload) are what this run added to confirm that and to test the one
# limitation the mechanism implies but nothing has measured.

alias Latu.ML
alias Latu.ML.Feature

defmodule ProbeExt do
  @moduledoc false

  alias Latu.Protocol.Spark.Connect, as: Proto
  alias Latu.Protocol.Spark.Connect.SparkConnectService.Stub

  # 32 KiB — PySpark's chunk size, and Latu's for `cache/` artifacts.
  @chunk 32 * 1024

  @if_estimator "com.linkedin.relevance.isolationforest.IsolationForest"
  @if_model "com.linkedin.relevance.isolationforest.IsolationForestModel"
  @xgb "ml.dmlc.xgboost4j.scala.spark"

  # What each library's shim names, and where. **Only estimators go in a services file**
  # (measured 2026-09-07, first run). A model class must NOT: it has no public no-arg
  # constructor, and the server runs a `ServiceLoader` enumeration over the Transformer service
  # on *every* transform — `VectorAssembler` included — that throws for the whole service the
  # moment it meets a provider it cannot resolve. Listing models there
  # poisons the Transformer loader for the session, and the first run died at the assembler with
  # `IsolationForestModel ... Unable to get public no-arg constructor` before any fit reached its
  # estimator. MLlib's own models sit in that file unharmed because each carries a
  # `private[ml] def this()` — package-private in Scala, public in bytecode.
  #
  # A fitted model transforms from its cache reference (`ModelAttributeHelper`, no loader), so
  # `models` here is only the `model_class` the struct carries and the `{class, :model}` a
  # server-side `load` would need — a load that cannot work for these classes, which is the
  # "read the artefact locally" seam rather than a gap. `transformers` is for real, non-model
  # Transformer operators; these two libraries have none.
  @libraries %{
    "isolation-forest" => %{estimators: [@if_estimator], transformers: [], models: [@if_model]},
    "xgboost4j-spark" => %{
      estimators:
        Enum.map(~w(XGBoostClassifier XGBoostRegressor XGBoostRanker), &"#{@xgb}.#{&1}"),
      transformers: [],
      models:
        Enum.map(
          ~w(XGBoostClassificationModel XGBoostRegressionModel XGBoostRankerModel),
          &"#{@xgb}.#{&1}"
        )
    }
  }

  def libraries, do: @libraries
  def if_estimator, do: @if_estimator
  def if_model, do: @if_model
  def xgb(class), do: "#{@xgb}.#{class}"

  # ---------------------------------------------------------------------------------------
  # The shim, and getting a jar onto the server
  # ---------------------------------------------------------------------------------------

  @doc "A services jar for the named libraries: a zip of one or two text files, no JDK."
  def shim(libraries) when is_list(libraries) do
    taken = Map.take(@libraries, libraries)

    shim_from(
      Enum.flat_map(taken, fn {_, lib} -> lib.estimators end),
      Enum.flat_map(taken, fn {_, lib} -> lib.transformers end)
    )
  end

  def shim_from(estimators, transformers) do
    # Only write a services file that has a name in it — an empty file is harmless, but omitting
    # the Transformer file entirely leaves MLlib's own Transformer loader untouched.
    entries =
      [
        {~c"META-INF/services/org.apache.spark.ml.Estimator", estimators},
        {~c"META-INF/services/org.apache.spark.ml.Transformer", transformers}
      ]
      |> Enum.reject(fn {_path, names} -> names == [] end)
      |> Enum.map(fn {path, names} -> {path, lines(names)} end)

    {:ok, {_name, bytes}} = :zip.create(~c"shim.jar", entries, [:memory])
    bytes
  end

  defp lines(names), do: Enum.map_join(names, "", &(&1 <> "\n"))

  @doc """
  Upload one jar as a session artifact named `jars/<name>`.

  The request sequence is `Latu.Client.artifact_requests/2`'s with the prefix changed — a small
  blob rides a `Batch`, a large one is `BeginChunkedArtifact` plus 32 KiB chunks — sent on the
  stub directly, because Latu's own path is hard-wired to `cache/`. `is_crc_successful` per
  artifact is the server's answer; anything else is the gRPC error.
  """
  def add_jar(session, name, bytes) when is_binary(name) and is_binary(bytes) do
    stream = Stub.add_artifacts(session.channel, timeout: session.timeout)

    result =
      session
      |> requests("jars/" <> name, bytes)
      |> Enum.reduce(stream, &GRPC.Stub.send_request(&2, &1))
      |> GRPC.Stub.end_stream()
      |> GRPC.Stub.recv(timeout: session.timeout)

    case result do
      {:ok, %Proto.AddArtifactsResponse{artifacts: summaries}} ->
        case Enum.reject(summaries, & &1.is_crc_successful) do
          [] -> {:ok, Enum.map(summaries, & &1.name)}
          bad -> {:error, "crc failed for " <> Enum.map_join(bad, ", ", & &1.name)}
        end

      {:error, %GRPC.RPCError{status: status, message: message}} ->
        {:error, "grpc #{status} — " <> first_line(message)}

      other ->
        {:error, inspect(other)}
    end
  end

  defp requests(session, name, bytes) when byte_size(bytes) <= @chunk do
    single = %Proto.AddArtifactsRequest.SingleChunkArtifact{name: name, data: chunk(bytes)}
    [request(session, {:batch, %Proto.AddArtifactsRequest.Batch{artifacts: [single]}})]
  end

  defp requests(session, name, bytes) do
    [first | rest] = chunks(bytes)

    begin = %Proto.AddArtifactsRequest.BeginChunkedArtifact{
      name: name,
      total_bytes: byte_size(bytes),
      num_chunks: 1 + length(rest),
      initial_chunk: chunk(first)
    }

    [
      request(session, {:begin_chunk, begin})
      | Enum.map(rest, &request(session, {:chunk, chunk(&1)}))
    ]
  end

  defp chunk(data),
    do: %Proto.AddArtifactsRequest.ArtifactChunk{data: data, crc: :erlang.crc32(data)}

  defp chunks(bytes) when byte_size(bytes) <= @chunk, do: [bytes]
  defp chunks(<<head::binary-size(@chunk), rest::binary>>), do: [head | chunks(rest)]

  defp request(session, payload) do
    %Proto.AddArtifactsRequest{
      session_id: session.session_id,
      client_observed_server_side_session_id: session.server_session_id,
      user_context: %Proto.UserContext{user_id: session.user_id, user_name: session.user_name},
      client_type: session.client_type,
      payload: payload
    }
  end

  @doc "What `LIST JARS` shows — every jar `sparkContext.addJar` has been given."
  def list_jars(session) do
    with {:ok, frame} <- Latu.sql(session, "LIST JARS"),
         {:ok, rows} <- Latu.collect(frame, keys: :strings) do
      {:ok, Enum.map(rows, &Map.get(&1, "Results"))}
    end
  end

  @doc "The library jars in `tmp/jars`, keyed by the library each belongs to."
  def local_jars(dir \\ "tmp/jars") do
    for path <- Path.wildcard(Path.join(dir, "*.jar")),
        base = Path.basename(path),
        library =
          Enum.find(Map.keys(@libraries) ++ ["spark-avro"], &String.starts_with?(base, &1)),
        do: {library, base, path}
  end

  # ---------------------------------------------------------------------------------------
  # The fixture and the verbs
  # ---------------------------------------------------------------------------------------

  @doc "probe_ml.exs's four rows: a `label` and a `features` Vector under Spark's default names."
  def frames(session) do
    base =
      Latu.sql!(session, """
      SELECT
        CAST(label AS DOUBLE) AS label,
        CAST(x1 AS DOUBLE) AS x1,
        CAST(x2 AS DOUBLE) AS x2
      FROM VALUES (0.0, 0.0, 1.1), (0.0, 0.1, 1.0), (1.0, 2.0, 0.1), (1.0, 2.1, 0.2)
      AS t(label, x1, x2)
      """)

    assembler = Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)
    %{base: base, features: ML.transform(assembler, base)}
  end

  @doc "Rows of the named columns, sorted by `x1` so two transforms of one plan compare."
  def rows(frame, columns) do
    with {:ok, rows} <- frame |> Latu.select([:x1 | columns]) |> Latu.collect() do
      {:ok, rows |> Enum.sort_by(& &1.x1) |> Enum.map(&Map.take(&1, columns))}
    end
  end

  @doc """
  A `Fetch` with Vector arguments, hand-built.

  `Latu.ML.attribute/3` types its arguments from the registry, which has no row for a class it
  does not know; the plan underneath takes literals, so this builds them with `Linalg` and sends
  the same command the facade would.
  """
  def fetch(model, name, tensors) do
    plan = ML.Internal.fetch_plan(model, name, Enum.map(tensors, &ML.Linalg.to_literal/1))

    with {:ok, execution} <- Latu.Client.execute_command(model.session, plan) do
      ML.Result.param(execution)
    end
  end

  @doc "The six attributes the allowlist grants by inheritance, asked one by one."
  def inherited(model, vector) do
    say("toString", outcome(ML.attribute(model, "toString")))
    say("numFeatures", outcome(ML.attribute(model, "numFeatures")))
    say("numClasses", outcome(ML.attribute(model, "numClasses")))
    say("predict(v)", outcome(fetch(model, "predict", [vector])))
    say("predictRaw(v)", outcome(fetch(model, "predictRaw", [vector])))
    say("predictProbability(v)", outcome(fetch(model, "predictProbability", [vector])))
  end

  @doc "Save, load back by `{class, :model}`, transform both, compare, release the loaded one."
  def round_trip(model, class, frame, path, columns, options \\ %{}) do
    say("save", outcome(ML.save(model, path, overwrite: true, options: options)))

    case ML.load(model.session, {class, :model}, path) do
      {:ok, loaded} ->
        say("load {class, :model}", "ok — " <> loaded.ref)
        mine = rows(ML.transform(model, frame), columns)
        theirs = rows(ML.transform(loaded, frame), columns)
        say("loaded model scores the same", inspect(mine == theirs) <> " — " <> outcome(theirs))
        say("delete loaded", outcome(ML.delete(loaded)))

      {:error, _} = error ->
        say("load {class, :model}", outcome(error))
    end
  end

  def cache_count(session) do
    case ML.cache_info(session) do
      {:ok, entries} -> length(entries)
      {:error, _} -> :unknown
    end
  end

  # ---------------------------------------------------------------------------------------
  # Reporting
  # ---------------------------------------------------------------------------------------

  def section(title), do: IO.puts("\n" <> title)
  def say(label, detail), do: IO.puts("  " <> String.pad_trailing(label, 36) <> detail)

  def outcome(:ok), do: "ok"
  def outcome({:ok, value}), do: "ok — " <> inspect(value, limit: 8, printable_limit: 120)

  # A refusal this package made itself carries no `error_class` — that field is Spark's.
  def outcome({:error, %Latu.Error{error_class: nil, kind: kind, message: message}}),
    do: "#{inspect(kind)} — " <> first_line(message)

  def outcome({:error, %Latu.Error{error_class: class, message: message}}),
    do: "#{inspect(class)} — " <> first_line(message)

  def outcome({:error, text}) when is_binary(text), do: "error — " <> first_line(text)
  def outcome({:error, other}), do: "error — " <> inspect(other, limit: 6)
  def outcome(other), do: inspect(other, limit: 8)

  def first_line(text) when is_binary(text) do
    text |> String.split("\n", parts: 2) |> hd() |> String.slice(0, 180)
  end

  def first_line(other), do: inspect(other)

  @doc "Run a step that may raise as well as fail, and report either the same way."
  def attempt(fun) do
    fun.()
  rescue
    error -> {:error, Exception.message(error)}
  end
end

# =========================================================================================
# Setup
# =========================================================================================

url = System.get_env("SPARK_REMOTE", "sc://localhost:15002")
mounted? = "--mounted" in System.argv()
mc_key = "spark.connect.session.connectML.mlCache.memoryControl.enabled"

session = Latu.connect!(url)
spark = Latu.spark_version!(session)

route = if mounted?, do: "mounted", else: "upload"
IO.puts("Spark #{spark} — third-party operator probe (#{url}, #{route} route)")

frames = ProbeExt.frames(session)
vector = Nx.tensor([2.0, 0.1])

jars = ProbeExt.local_jars()
present = jars |> Enum.map(&elem(&1, 0)) |> Enum.uniq()

if not mounted? and present == [] do
  IO.puts("\nno jars in tmp/jars — run `bash dev/fetch_jars.sh` first")
  System.halt(1)
end

# Which libraries the shim may name: on the mounted route every jar the compose file carries,
# on the upload route only the ones found locally, since each is uploaded before use.
shim_for = if mounted?, do: Map.keys(ProbeExt.libraries()), else: present -- ["spark-avro"]

# =========================================================================================
# 1. The artifact route
# =========================================================================================

ProbeExt.section("the artifact route")

ProbeExt.say("jars before", ProbeExt.outcome(ProbeExt.list_jars(session)))

if mounted? do
  # The gate, shown from the wrong side: the jar is on the classpath, the services file is not.
  bare = ML.Estimator.new(ProbeExt.if_estimator(), params: [{"numEstimators", :int, 5}])
  ProbeExt.say("fit with no shim at all", ProbeExt.outcome(ML.fit(bare, frames.features)))
end

shim = ProbeExt.shim(shim_for)
ProbeExt.say("shim bytes for #{inspect(shim_for)}", inspect(byte_size(shim)))

ProbeExt.say(
  "upload shim",
  ProbeExt.outcome(ProbeExt.add_jar(session, "latu-ext-shim.jar", shim))
)

unless mounted? do
  for {library, base, path} <- jars do
    bytes = File.read!(path)
    {micros, result} = :timer.tc(fn -> ProbeExt.add_jar(session, base, bytes) end)
    size = Float.round(byte_size(bytes) / 1_048_576, 1)

    ProbeExt.say(
      "upload #{library} (#{size} MB)",
      ProbeExt.outcome(result) <> " in #{div(micros, 1000)} ms"
    )
  end
end

ProbeExt.say("jars after", ProbeExt.outcome(ProbeExt.list_jars(session)))

ProbeExt.say(
  "re-upload the shim",
  ProbeExt.outcome(ProbeExt.add_jar(session, "latu-ext-shim.jar", shim))
)

# =========================================================================================
# 2. isolation-forest — the mechanism, with nothing native in the way
# =========================================================================================

if "isolation-forest" in shim_for do
  ProbeExt.section("isolation-forest — memory control on (the default)")

  # `maxSamples` at or below 1.0 is a fraction; four rows need all four.
  forest =
    ML.Estimator.new(ProbeExt.if_estimator(),
      params: [{"numEstimators", :int, 10}, {"maxSamples", :double, 1.0}],
      model_class: ProbeExt.if_model()
    )

  key = "spark.connect.session.connectML.mlCache.memoryControl.enabled"
  ProbeExt.say(key, ProbeExt.outcome(Latu.conf(session, key)))
  before = ProbeExt.cache_count(session)

  case ML.fit(forest, frames.features) do
    {:ok, model} ->
      ProbeExt.say("fit", "ok — " <> model.ref)
      ProbeExt.say("cache entries", "#{before} -> #{ProbeExt.cache_count(session)}")

      scored = ML.transform(model, frames.features)

      ProbeExt.say(
        "transform, collected",
        ProbeExt.outcome(ProbeExt.rows(scored, [:outlierScore, :predictedLabel]))
      )

      # A plain `Model`: `Identifiable.toString` is the one thing the allowlist grants it.
      ProbeExt.say("toString", ProbeExt.outcome(ML.attribute(model, "toString")))
      ProbeExt.say("numFeatures", ProbeExt.outcome(ML.attribute(model, "numFeatures")))
      ProbeExt.say("model_size", ProbeExt.outcome(ML.model_size(model)))

      path = "/tmp/latu_ext/forest"
      ProbeExt.round_trip(model, ProbeExt.if_model(), frames.features, path, [:predictedLabel])
      ProbeExt.say("delete", ProbeExt.outcome(ML.delete(model)))

    error ->
      # Under memory control the fit ends in `MLCache.register`, which saves the model at once;
      # a writer needing what the server lacks — `spark-avro` — fails here, not in training.
      ProbeExt.say("fit", ProbeExt.outcome(error))
  end

  ProbeExt.section("isolation-forest — memory control off, in a session of its own")

  side = Latu.connect!(url)
  # A session's shim names only the libraries whose jars that session has — a shim naming an
  # estimator whose class is absent fails `loadOperators(Estimator)` for the whole service.
  if_shim = ProbeExt.shim(["isolation-forest"])
  ProbeExt.say("set #{key} = false", ProbeExt.outcome(Latu.set_conf(side, key, "false")))

  ProbeExt.say(
    "upload shim",
    ProbeExt.outcome(ProbeExt.add_jar(side, "latu-ext-shim.jar", if_shim))
  )

  unless mounted? do
    for {"isolation-forest", base, path} <- jars do
      ProbeExt.say(
        "upload #{base}",
        ProbeExt.outcome(ProbeExt.add_jar(side, base, File.read!(path)))
      )
    end
  end

  side_frames = ProbeExt.frames(side)

  case ML.fit(forest, side_frames.features) do
    {:ok, model} ->
      ProbeExt.say("fit", "ok — " <> model.ref)
      scored = ML.transform(model, side_frames.features)

      ProbeExt.say(
        "transform, collected",
        ProbeExt.outcome(ProbeExt.rows(scored, [:outlierScore, :predictedLabel]))
      )

      ProbeExt.say(
        "save (needs spark-avro)",
        ProbeExt.outcome(ML.save(model, "/tmp/latu_ext/forest_off", overwrite: true))
      )

      ProbeExt.say("delete", ProbeExt.outcome(ML.delete(model)))

    error ->
      ProbeExt.say("fit", ProbeExt.outcome(error))
  end

  Latu.disconnect(side, release: true)
end

# =========================================================================================
# 3. XGBoost — classifier
# =========================================================================================

if "xgboost4j-spark" in shim_for do
  ProbeExt.section("XGBoost classifier")

  # Spark-side params are camelCase, native ones snake_case, both as the `Param` name.
  xgb =
    ML.Estimator.new(ProbeExt.xgb("XGBoostClassifier"),
      params: [{"numRound", :int, 5}, {"numWorkers", :int, 1}, {"max_depth", :int, 2}],
      model_class: ProbeExt.xgb("XGBoostClassificationModel")
    )

  before = ProbeExt.cache_count(session)
  {micros, fitted} = :timer.tc(fn -> ML.fit(xgb, frames.features) end)

  case fitted do
    {:ok, model} ->
      ProbeExt.say("fit", "ok — #{model.ref} in #{div(micros, 1000)} ms")
      ProbeExt.say("cache entries", "#{before} -> #{ProbeExt.cache_count(session)}")

      scored = ML.transform(model, frames.features)
      ProbeExt.say("schema of the scored frame", ProbeExt.outcome(Latu.columns(scored)))
      collected = ProbeExt.rows(scored, [:label, :prediction])
      ProbeExt.say("transform, collected", ProbeExt.outcome(collected))

      # A `Transform` lays the estimator's params over a copy of the model, as PySpark does. If
      # that is what refused, the same transform with none re-sent says so.
      with {:error, _} <- collected do
        bare = ML.transform(%{model | params: []}, frames.features)

        ProbeExt.say(
          "transform, no params re-sent",
          ProbeExt.outcome(ProbeExt.rows(bare, [:label, :prediction]))
        )
      end

      auc = ML.Evaluation.binary_classification_evaluator()
      ProbeExt.say("evaluate, areaUnderROC", ProbeExt.outcome(ML.evaluate(auc, scored)))

      # The six by inheritance. XGBoost's `predictRaw` throws by design, and `predict` and
      # `predictProbability` route through it; `numClasses` is a plain val.
      ProbeExt.inherited(model, vector)
      ProbeExt.say("model_size", ProbeExt.outcome(ML.model_size(model)))

      # The model at rest, as JSON, read back through Spark rather than a shared mount.
      path = "/tmp/latu_ext/xgb"

      ProbeExt.round_trip(
        model,
        ProbeExt.xgb("XGBoostClassificationModel"),
        frames.features,
        path,
        [:prediction],
        %{"format" => "json"}
      )

      booster =
        with {:ok, rows} <-
               Latu.read(session, format: "binaryFile", path: path <> "/data/model")
               |> Latu.select([:content])
               |> Latu.collect(),
             [%{content: bytes}] <- rows do
          File.mkdir_p!("tmp")
          File.write!("tmp/xgb_booster.json", bytes)
          {:ok, bytes}
        else
          other -> {:error, inspect(other, limit: 4)}
        end

      ProbeExt.say(
        "data/model read back",
        ProbeExt.outcome(with {:ok, b} <- booster, do: {:ok, byte_size(b)})
      )

      facts =
        ProbeExt.attempt(fn ->
          with {:ok, bytes} <- booster do
            json = JSON.decode!(bytes)
            learner = json["learner"]
            trees = get_in(learner, ["gradient_booster", "model", "trees"]) || []

            {:ok,
             %{
               version: json["version"],
               num_feature: get_in(learner, ["learner_model_param", "num_feature"]),
               num_class: get_in(learner, ["learner_model_param", "num_class"]),
               objective: get_in(learner, ["objective", "name"]),
               trees: length(trees)
             }}
          end
        end)

      ProbeExt.say("the JSON booster says", ProbeExt.outcome(facts))
      ProbeExt.say("delete", ProbeExt.outcome(ML.delete(model)))

    error ->
      ProbeExt.say("fit", ProbeExt.outcome(error) <> " after #{div(micros, 1000)} ms")
  end

  # =======================================================================================
  # 4. XGBoost — regressor, and `predict(Vector)` against the frame's own prediction
  # =======================================================================================

  ProbeExt.section("XGBoost regressor")

  reg =
    ML.Estimator.new(ProbeExt.xgb("XGBoostRegressor"),
      params: [{"numRound", :int, 5}, {"numWorkers", :int, 1}],
      model_class: ProbeExt.xgb("XGBoostRegressionModel")
    )

  case ML.fit(reg, frames.features) do
    {:ok, model} ->
      ProbeExt.say("fit", "ok — " <> model.ref)
      scored = ML.transform(model, frames.features)

      ProbeExt.say(
        "transform, collected",
        ProbeExt.outcome(ProbeExt.rows(scored, [:label, :prediction]))
      )

      # The third fixture row is [2.0, 0.1]; `predict` on that vector should agree with its row.
      ProbeExt.say(
        "predict([2.0, 0.1])",
        ProbeExt.outcome(ProbeExt.fetch(model, "predict", [vector]))
      )

      ProbeExt.say("numFeatures", ProbeExt.outcome(ML.attribute(model, "numFeatures")))
      ProbeExt.say("delete", ProbeExt.outcome(ML.delete(model)))

    error ->
      ProbeExt.say("fit", ProbeExt.outcome(error))
  end

  # =======================================================================================
  # 4b. XGBoost with memory control off — does it train, or only fail to persist?
  # =======================================================================================

  ProbeExt.section("XGBoost — memory control off, in a session of its own")

  # The mounted run's diagnosis, put to the test. Under the default, `MLCache.register` saves a
  # model the moment it registers it, and XGBoost's metadata writer cannot on 4.2.0:
  # `CustomGeneralParam.jsonEncode` calls `Extraction.decompose`, whose json4s 3.x signature
  # Spark 4.x dropped, and `SparkParams` defaults `customObj` and `customEval` to `null` — which
  # Spark's writer encodes on every save, so it fires whatever params the caller passed. The fit
  # therefore fails *after* training, and the two orphaned cache entries the mounted run counted
  # are the models it left behind. With the save-at-registration off, training should stand on
  # its own; `save/3` should still refuse, and with the same json4s message.
  xgb_side = Latu.connect!(url)
  side_shim = ProbeExt.shim(["xgboost4j-spark"])

  ProbeExt.say(
    "set #{mc_key} = false",
    ProbeExt.outcome(Latu.set_conf(xgb_side, mc_key, "false"))
  )

  ProbeExt.say(
    "upload shim",
    ProbeExt.outcome(ProbeExt.add_jar(xgb_side, "latu-ext-shim.jar", side_shim))
  )

  unless mounted? do
    for {"xgboost4j-spark", base, path} <- jars do
      ProbeExt.say(
        "upload #{base}",
        ProbeExt.outcome(ProbeExt.add_jar(xgb_side, base, File.read!(path)))
      )
    end
  end

  xgb_frames = ProbeExt.frames(xgb_side)

  case ML.fit(xgb, xgb_frames.features) do
    {:ok, model} ->
      ProbeExt.say("fit", "ok — " <> model.ref)
      scored = ML.transform(model, xgb_frames.features)

      ProbeExt.say(
        "transform, collected",
        ProbeExt.outcome(ProbeExt.rows(scored, [:label, :prediction]))
      )

      auc = ML.Evaluation.binary_classification_evaluator()
      ProbeExt.say("evaluate, areaUnderROC", ProbeExt.outcome(ML.evaluate(auc, scored)))
      ProbeExt.inherited(model, vector)

      # The same json4s break, now on the caller's own save rather than the cache's.
      ProbeExt.say(
        "save (expected: json4s)",
        ProbeExt.outcome(ML.save(model, "/tmp/latu_ext/xgb_off", overwrite: true))
      )

      ProbeExt.say("delete", ProbeExt.outcome(ML.delete(model)))

    error ->
      ProbeExt.say("fit", ProbeExt.outcome(error))
  end

  ProbeExt.say("cache entries left here", inspect(ProbeExt.cache_count(xgb_side)))
  Latu.disconnect(xgb_side, release: true)
end

# =========================================================================================
# 5. Offload — can a third-party model come back?
# =========================================================================================

# isolation-forest is the subject, not XGBoost: an offload is a `saveToLocal` and a
# `loadFromLocal`, and XGBoost cannot write its metadata on 4.2.0 at all. The mechanism says
# this should **fail**, and it is the most consequential thing left to measure:
# `MLCache.get` on an offloaded entry calls `MLUtils.loadTransformer(..., loadFromLocal = true)`,
# which looks the model class up in the **Transformer `ServiceLoader` map** — the one place a
# third-party model may not be listed, because listing it poisons the loader for everything.
# If so, a third-party model under default memory control is *transient*: it works until it is
# offloaded (15 minutes idle, or memory pressure) and is then unreachable through no fault of
# the caller. That would make memory control off, or a short-lived session, the documented way
# to use one — and it is the sharpest caveat any guide would owe.
if "--offload" in System.argv() and "isolation-forest" in shim_for do
  ProbeExt.section("offload — about 80 s")

  offload = Latu.connect!(url)
  timeout_key = "spark.connect.session.connectML.mlCache.memoryControl.offloadingTimeout"

  # Read once, when the cache is first touched: set it before the first fit, never under `1m`.
  off_shim = ProbeExt.shim(["isolation-forest"])

  ProbeExt.say(
    "set offloadingTimeout = 1m",
    ProbeExt.outcome(Latu.set_conf(offload, timeout_key, "1m"))
  )

  ProbeExt.say(
    "upload shim",
    ProbeExt.outcome(ProbeExt.add_jar(offload, "latu-ext-shim.jar", off_shim))
  )

  unless mounted? do
    for {"isolation-forest", base, path} <- jars do
      ProbeExt.say(
        "upload #{base}",
        ProbeExt.outcome(ProbeExt.add_jar(offload, base, File.read!(path)))
      )
    end
  end

  offload_frames = ProbeExt.frames(offload)

  forest =
    ML.Estimator.new(ProbeExt.if_estimator(),
      params: [{"numEstimators", :int, 10}, {"maxSamples", :double, 1.0}],
      model_class: ProbeExt.if_model()
    )

  case ML.fit(forest, offload_frames.features) do
    {:ok, model} ->
      ProbeExt.say("fit", "ok — " <> model.ref)
      ProbeExt.say("toString, straight away", ProbeExt.outcome(ML.attribute(model, "toString")))

      IO.puts("  waiting out the minute...")
      Process.sleep(70_000)
      {micros, back} = :timer.tc(fn -> ML.attribute(model, "toString") end)

      ProbeExt.say(
        "toString, after the offload",
        ProbeExt.outcome(back) <> " in #{div(micros, 1000)} ms"
      )

      scored = ML.transform(model, offload_frames.features)

      ProbeExt.say(
        "transform, after the offload",
        ProbeExt.outcome(ProbeExt.rows(scored, [:predictedLabel]))
      )

      ProbeExt.say("cache_info", ProbeExt.outcome(ML.cache_info(offload)))
      ProbeExt.say("delete", ProbeExt.outcome(ML.delete(model)))

    error ->
      ProbeExt.say("fit", ProbeExt.outcome(error))
  end

  Latu.disconnect(offload, release: true)
else
  IO.puts("\n(offload not run; pass --offload, it takes about 80 s)")
end

# =========================================================================================
# 6. A second session, the same native library
# =========================================================================================

if "--second-session" in System.argv() and "xgboost4j-spark" in shim_for do
  ProbeExt.section("second session — the native library loaded twice")

  second = Latu.connect!(url)
  xgb_shim = ProbeExt.shim(["xgboost4j-spark"])

  # Memory control off, or the fit fails at its metadata save and says nothing about the
  # native library — which is the only thing this section is asking about.
  ProbeExt.say("set #{mc_key} = false", ProbeExt.outcome(Latu.set_conf(second, mc_key, "false")))

  ProbeExt.say(
    "upload shim",
    ProbeExt.outcome(ProbeExt.add_jar(second, "latu-ext-shim.jar", xgb_shim))
  )

  unless mounted? do
    for {"xgboost4j-spark", base, path} <- jars do
      ProbeExt.say(
        "upload #{base}",
        ProbeExt.outcome(ProbeExt.add_jar(second, base, File.read!(path)))
      )
    end
  end

  xgb =
    ML.Estimator.new(ProbeExt.xgb("XGBoostClassifier"),
      params: [{"numRound", :int, 5}, {"numWorkers", :int, 1}]
    )

  case ML.fit(xgb, ProbeExt.frames(second).features) do
    {:ok, model} ->
      ProbeExt.say("fit", "ok — " <> model.ref)
      ProbeExt.say("delete", ProbeExt.outcome(ML.delete(model)))

    error ->
      ProbeExt.say("fit", ProbeExt.outcome(error))
  end

  Latu.disconnect(second, release: true)
else
  IO.puts("\n(second session not run; pass --second-session)")
end

# =========================================================================================
# 7. A shim naming a class no jar carries
# =========================================================================================

if "--poison" in System.argv() do
  ProbeExt.section("poison — a services entry for a class that does not exist")

  poisoned = Latu.connect!(url)
  bad = ProbeExt.shim_from(["latu.ext.NoSuchEstimator"], [])
  lr = Latu.ML.Classification.logistic_regression(max_iter: 2)
  poisoned_frames = ProbeExt.frames(poisoned)

  ProbeExt.say(
    "logistic_regression, before",
    ProbeExt.outcome(with {:ok, m} <- ML.fit(lr, poisoned_frames.features), do: ML.delete(m))
  )

  ProbeExt.say(
    "upload the poison shim",
    ProbeExt.outcome(ProbeExt.add_jar(poisoned, "latu-ext-poison.jar", bad))
  )

  # The reading: `Provider.type()` throws `ServiceConfigurationError` for the missing class and
  # `loadOperators` propagates it, so *MLlib's own* estimator should now fail in this session.
  ProbeExt.say(
    "logistic_regression, after",
    ProbeExt.outcome(with {:ok, m} <- ML.fit(lr, poisoned_frames.features), do: ML.delete(m))
  )

  ProbeExt.say("a plain query still answers", ProbeExt.outcome(Latu.count(poisoned_frames.base)))

  Latu.disconnect(poisoned, release: true)
else
  IO.puts("\n(poison not run; pass --poison)")
end

# =========================================================================================
# Accounting
# =========================================================================================

ProbeExt.section("accounting")

# Not hygiene — a measurement. `MLCache.register` puts a model in the cache and *then* saves it,
# so a fit whose save throws (every XGBoost fit under default memory control) leaves a model
# behind that the client never got a ref to and can therefore never delete. The mounted run
# counted two, one per failed fit. `clean_cache/1` is the only way out short of ending the
# session, and what it answers with is how many were orphaned.
left = ProbeExt.cache_count(session)
ProbeExt.say("cache entries left in the main session", inspect(left))

if left != 0 do
  ProbeExt.say("orphaned by a failed fit's save", ProbeExt.outcome(ML.clean_cache(session)))
end

Latu.disconnect(session, release: true)
