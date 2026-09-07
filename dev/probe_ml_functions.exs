# Probe: the three functions mllib registers in Spark's *internal* function registry, and
# whether Latu reaches them over the wire.
#
#     docker compose up -d spark-connect
#     mix run dev/probe_ml_functions.exs
#
# ## The question
#
# Roadmap §1 carries `vector_to_array` and `array_to_vector` as uncertain on the strength of one
# measurement: Latu's `dev/probe_ml_server.exs` asked for them in **SQL text** and the server
# answered `UNRESOLVED_ROUTINE` over `[system.builtin, …]`. Reading 4.2.0's own source says that
# measurement could not have answered anything else, and that the proto route is a different
# path:
#
#   * `mllib/src/main/scala/org/apache/spark/sql/ml/InternalFunctionRegistration.scala`
#     registers `vector_to_array`, `array_to_vector` and `aggregate_metrics` into
#     `FunctionRegistry.internal` — not the builtin registry SQL text is parsed against, which
#     is why the SQL route cannot see them however well they work. It reaches a session as a
#     `SparkSessionExtensionsProvider` named in mllib's `META-INF/services`, loaded off the
#     classpath by default (SPARK-49086).
#   * `SparkConnectPlanner.transformUnresolvedFunction` reroutes a name into that registry when
#     `is_internal` is **unset** and the name is registered there — a back-compat path kept for
#     exactly these, and what PySpark rides: `pyspark/ml/connect/functions.py` sends a plain
#     `_invoke_function("vector_to_array", col, lit(dtype))` and sets no `is_internal`.
#
# `Latu.Plan.fun/3` builds that message already, so if the chain holds, all three are reachable
# today with no change to Latu and none to this package. This asks the server whether it holds.
#
# The builders are in the Scala above, and they are strict: `vector_to_array` takes exactly two
# arguments and the second must be a **string literal** `"float64"` or `"float32"`;
# `array_to_vector` takes exactly one; `aggregate_metrics` takes an array of metric-name
# literals, the features column, and a weight column. All three are UDFs rather than codegen'd
# expressions, so a sparse vector densifies on the way out.
#
# ## Where it goes red
#
#   * The whole chain, if the proto route answers `UNRESOLVED_ROUTINE` as well. Then the
#     extension is not on this image's classpath and §1 was right for the wrong reason.
#   * **Unset is not false.** `is_internal: false` explicitly set is a different wire message
#     from the field being absent, and only the absent one takes the back-compat path. If both
#     resolve, `hasIsInternal` is not doing what the planner's source says, and Latu never needs
#     an `internal:` option.
#   * **§2.5, from the other side.** `array_to_vector` builds a Vector column out of a plan the
#     client wrote rather than out of a fit. If its schema carries a `sql_type` where a fitted
#     `features` column does not, the UDT gap is narrower than the roadmap says and `to_nx/2`'s
#     capability argument shrinks with it.
#   * **What `collect/2` does with `array<double>`.** If the round trip out of a Vector column
#     lands in ordinary lists, `to_nx/2` is an optimisation again rather than the only route —
#     which is the ML6 decision this probe is really for.

alias Latu.Column
alias Latu.ML
alias Latu.ML.Feature
alias Latu.Protocol.Spark.Connect, as: Proto

defmodule ProbeFns do
  @moduledoc false

  def first_line(message) do
    message |> String.split("\n") |> hd() |> String.slice(0, 100)
  end

  # `:limit` is a budget shared across nested collections, not a per-collection cap — at 6 it
  # hid the all-zero rows this probe exists to show.
  def describe({:ok, value}), do: "ok — " <> inspect(value, limit: 50, printable_limit: 400)

  def describe({:error, %Latu.Error{} = error}) do
    "#{inspect(error.error_class)} — " <> first_line(error.message)
  end

  # A client-side refusal, which is a different thing from the server's and says so.
  def describe({:raised, message}), do: "raised here — " <> first_line(message)

  def describe(other), do: inspect(other)

  @doc """
  A function call with `is_internal` set by hand, which `Latu.Column.fun/3` cannot spell — it
  always leaves the field absent, which is the point of the comparison.
  """
  def fun_with_internal(name, arguments, internal) do
    %Proto.Expression{
      expr_type:
        {:unresolved_function,
         %Proto.Expression.UnresolvedFunction{
           function_name: name,
           arguments: Enum.map(arguments, &Latu.Plan.to_expr/1),
           is_distinct: false,
           is_internal: internal
         }}
    }
  end

  @doc """
  Run something that answers `{:ok, _}` or `{:error, _}` and might instead raise — Latu's dtype
  guard does, and a probe that fell over on the first refusal would report nothing after it.
  """
  def safely(fun) do
    fun.()
  rescue
    error -> {:raised, Exception.message(error)}
  end

  @doc "Select one expression and execute, so a lazy relation cannot pass for a resolved name."
  def collect_one(df, expression) do
    safely(fn -> df |> Latu.select(v: expression) |> Latu.collect() end)
  end

  @doc "`describe/1` of `safely/1`, which is most of the lines below."
  def try_say(fun), do: describe(safely(fun))
end

say = fn label, detail ->
  IO.puts("  " <> String.pad_trailing(label, 44) <> detail)
end

url = System.get_env("SPARK_REMOTE", "sc://localhost:15002")
session = Latu.connect!(url)
spark = Latu.spark_version!(session)

IO.puts("Spark #{spark} — ML internal functions probe (#{url})\n")

# =============================================
# The fixture
# =============================================
#
# One frame with a real fitted-shape `features` column — `VectorAssembler` over two doubles, the
# same way `probe_ml.exs` builds one — and one with a plain `array<double>` to go the other way.
# Rows three and four are all zero, which is where `VectorAssembler` emits a `SparseVector`:
# both `vector_to_array` UDFs have a sparse branch of their own, so those two rows coming back
# as `[0.0, 0.0]` rather than as an error or a short list is the check on it.

base =
  Latu.sql!(session, """
  SELECT CAST(x1 AS DOUBLE) AS x1, CAST(x2 AS DOUBLE) AS x2
  FROM VALUES (1.5, 2.5), (0.5, 3.5), (0.0, 0.0), (0.0, 0.0) AS t(x1, x2)
  """)

assembler = Feature.vector_assembler(input_cols: [:x1, :x2], output_col: :features)
features = ML.transform(assembler, base)

arrays =
  Latu.sql!(session, """
  SELECT CAST(a AS ARRAY<DOUBLE>) AS a FROM VALUES
    (array(1.5, 2.5)), (array(0.5, 3.5)), (array(0.0, 0.0))
  AS t(a)
  """)

# =============================================
# 1. The SQL-text route, re-run
# =============================================
#
# Kept so the correction to §1 carries its own evidence: this is the measurement the roadmap
# had, and it is expected to stay `UNRESOLVED_ROUTINE` even if everything below resolves. An
# internal function is not reachable by name from the parser, and that is by design, not a gap.

IO.puts("1. the SQL-text route (the measurement §1 had)")

for {name, call} <- [
      {"vector_to_array", "vector_to_array(array(1.0, 2.0), 'float64')"},
      {"array_to_vector", "array_to_vector(array(1.0, 2.0))"},
      {"aggregate_metrics", "aggregate_metrics(array('mean'), array(1.0), 1.0)"}
    ] do
  answer =
    ProbeFns.try_say(fn ->
      with {:ok, df} <- Latu.sql(session, "SELECT #{call} AS v"), do: Latu.collect(df)
    end)

  say.(name <> " in SQL text", answer)
end

# =============================================
# 2. vector_to_array over the proto wire
# =============================================
#
# `Latu.Column.fun/3` with no `is_internal`, which is byte for byte what PySpark's Connect
# client sends. A string argument becomes a literal through `Latu.Plan.to_expr/1` — an atom
# would have become a column reference, which is the mistake the Scala builder's third clause
# exists to catch, so both are asked.

IO.puts("\n2. vector_to_array over the proto wire")

v2a = fn args -> Column.fun("vector_to_array", args) end
on_features = fn expression -> ProbeFns.describe(ProbeFns.collect_one(features, expression)) end

say.("float64, collected", on_features.(v2a.([:features, "float64"])))
say.("float32, collected", on_features.(v2a.([:features, "float32"])))
say.("one argument (wants two)", on_features.(v2a.([:features])))
say.("dtype as a column, not a literal", on_features.(v2a.([:features, :x1])))
say.("a dtype nobody registered", on_features.(v2a.([:features, "float16"])))

as_array = Latu.select(features, v: v2a.([:features, "float64"]))
as_floats = Latu.select(features, v: v2a.([:features, "float32"]))

say.("float64's column type", ProbeFns.try_say(fn -> Latu.schema(as_array) end))
say.("float32's column type", ProbeFns.try_say(fn -> Latu.schema(as_floats) end))

# Rows three and four are all zero, which is where `VectorAssembler` emits a `SparseVector` and
# where each UDF takes a branch of its own. Printed as bare values rather than left inside four
# rows of maps, because that is the whole check.
values = fn df ->
  ProbeFns.try_say(fn ->
    with {:ok, rows} <- Latu.collect(df), do: {:ok, Enum.map(rows, & &1.v)}
  end)
end

say.("every row's values, float64", values.(as_array))
say.("every row's values, float32", values.(as_floats))

# =============================================
# 3. array_to_vector, and what the Vector column says about itself
# =============================================
#
# The §2.5 question from the other side. `collect/2` is expected to refuse the built Vector
# column exactly as it refuses a fitted `features` — and if it does not, that is the finding.

IO.puts("\n3. array_to_vector")

a2v = fn args -> Column.fun("array_to_vector", args) end
built = Latu.select(arrays, v: a2v.([:a]))

say.("the built column's type", ProbeFns.try_say(fn -> Latu.schema(built) end))
say.("collected", ProbeFns.try_say(fn -> Latu.collect(built) end))

say.(
  "two arguments (wants one)",
  ProbeFns.describe(ProbeFns.collect_one(arrays, a2v.([:a, :a])))
)

# Is it a Vector the ML side accepts, or only something shaped like one? A scaler is the
# cheapest operator that reads a Vector column and refuses anything else.
scaler = Feature.standard_scaler(input_col: :v, output_col: :scaled)
say.("fitted by a StandardScaler", ProbeFns.try_say(fn -> ML.fit(scaler, built) end))

# The round trip, which is the shape `Latu.ML.Functions` would document if it exists.
round_trip = v2a.([a2v.([v2a.([:features, "float64"])]), "float64"])
say.("vector_to_array back to vector", on_features.(round_trip))

# =============================================
# 4. Unset is not false
# =============================================
#
# The planner reads `hasIsInternal` before it reads the value: absent means "look me up in the
# internal registry if you know the name", `false` means "do not". Latu's `fun/3` can only send
# absent, so what this settles is whether an `internal:` option would ever be worth adding.

IO.puts("\n4. is_internal, set by hand")

for {label, internal} <- [{"unset (what fun/3 sends)", nil}, {"true", true}, {"false", false}] do
  call = ProbeFns.fun_with_internal("vector_to_array", [:features, "float64"], internal)
  say.(label, on_features.(call))
end

# =============================================
# 5. aggregate_metrics — the third one, and a whole PySpark class
# =============================================
#
# Not in §1 at all: `Summarizer` is `pyspark.ml.stat`'s, and over Connect it is this one
# function — `aggregate_metrics(array(<names>), features, weight)` returning a struct keyed by
# metric name. Beyond ML6's scope, but it is registered by the same file as the other two, so
# one run here decides whether `Latu.ML.Stat` has a fourth member waiting.

IO.puts("\n5. aggregate_metrics (Summarizer's route)")

summarize = fn names ->
  Column.fun("aggregate_metrics", [Column.fun("array", names), :features, 1.0])
end

say.("mean, max, count over features", on_features.(summarize.(["mean", "max", "count"])))
say.("a metric nobody registered", on_features.(summarize.(["nonsense"])))

# The vector-valued metrics come back as a struct of UDTs, which the dtype guard refuses for the
# same reason a `features` column is refused; the scalar ones do not. Getting at either needs
# field access, and **Latu has no builder for `UnresolvedExtractValue`** — `Latu.Column.expr/1`
# over a named column is the only way in today. Whether that is enough decides how `Summarizer`
# would be shaped, if it is shaped at all (D13).
summarized = Latu.select(features, m: summarize.(["mean", "count"]))

say.(
  "count, out of the struct by expr",
  ProbeFns.try_say(fn -> Latu.collect(Latu.select(summarized, c: Column.expr("m.count"))) end)
)

say.(
  "mean, out of the struct and to an array",
  ProbeFns.try_say(fn ->
    Latu.collect(Latu.select(summarized, mean: v2a.([Column.expr("m.mean"), "float64"])))
  end)
)

# =============================================
# 6. What the array column is worth downstream
# =============================================
#
# The ML6 decision. If `array<double>` reaches Explorer and Nx through the paths Latu already
# has, `to_nx/2` is an optimisation with a measured cost, not the only way out of a Vector
# column — and §4 seam 1 has to say so.

IO.puts("\n6. the array column downstream")

batches = fn ->
  with {:ok, batches} <- Latu.to_arrow(as_array) do
    bytes = batches |> Enum.map(&byte_size/1) |> Enum.sum()
    {:ok, "#{length(batches)} batch(es), #{bytes} bytes"}
  end
end

say.("to_explorer", ProbeFns.try_say(fn -> Latu.to_explorer(as_array) end))
say.("to_arrow", ProbeFns.try_say(fn -> batches.() end))

Latu.disconnect(session, release: true)
