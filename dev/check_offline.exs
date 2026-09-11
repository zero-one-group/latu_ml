# What the container can check without mix, hex or a server.
#
#     elixir dev/check_offline.exs
#
# Latu's `dev/check_offline.exs`, in miniature. The pure layers — param translation, literal
# coercion, the command and relation builders — compile against `dev/standins/protocol.exs`
# and run here, so a session with no Elixir toolchain still gets a real answer rather than a
# parse check. Everything that needs Nx (`Latu.ML.Linalg`) or Latu itself (`Latu.ML`,
# `Latu.ML.Result`) is out of reach and is covered by `mix test` instead.
#
# A stand-in carries no protobuf behaviour: it is a struct with the right fields and nothing
# else. When a field is added to a message this file uses, add it here too.

Code.require_file(Path.join(__DIR__, "standins/protocol.exs"))

src = Path.expand("..", __DIR__)
sources = [
  "lib/latu/ml/param.ex",
  "lib/latu/ml/plan.ex",
  "lib/latu/ml/estimator.ex",
  "lib/latu/ml/transformer.ex",
  "lib/latu/ml/evaluator.ex",
  "lib/latu/ml/helper.ex",
  "lib/latu/ml/pipeline.ex",
  "lib/latu/ml/pipeline_model.ex",
  "lib/latu/ml/cross_validator.ex",
  "lib/latu/ml/cross_validator_model.ex",
  "lib/latu/ml/train_validation_split.ex",
  "lib/latu/ml/train_validation_split_model.ex",
  "lib/latu/ml/layout.ex",
  "lib/latu/ml/model.ex",
  "lib/latu/ml/summary.ex",
  "lib/latu/ml/operator.ex",
  "lib/latu/ml/registry.ex",
  "lib/latu/ml/generate.ex",
  "lib/latu/ml/internal.ex",
  "lib/latu/ml/inspect.ex",
  "lib/latu/ml/result.ex",
  "lib/latu/ml/classification.ex",
  "lib/latu/ml/clustering.ex",
  "lib/latu/ml/evaluation.ex",
  "lib/latu/ml/feature.ex",
  "lib/latu/ml/fpm.ex",
  "lib/latu/ml/recommendation.ex",
  "lib/latu/ml/regression.ex",
  "lib/latu/ml/stat.ex"
]

# Eight warnings, all of them the same shape and none of them news: a module this file does not
# compile, or one it compiles later in the list. Left on and they drown the one warning that
# would matter, so they are silenced **by name** rather than by turning warnings off — anything
# not on this list still warns, which is the point.
#
#   * `Latu.ML`, `Latu.ML.Result`, `Latu.ML.Linalg` — reached from the modules below, and not
#     compiled here because they need Latu itself or Nx.
#   * `Latu.Result.Literal` — Latu's, same reason.
#   * `Latu.ML.Internal` — compiled here, but *after* the three operator structs that call its
#     `uid/1`. A forward reference, not a missing module.
#   * `Latu` and `JSON` — `Latu.ML.Layout` needs both, and neither is here: Latu is the
#     dependency this cannot compile, and `JSON` is Elixir 1.18's, where the container has 1.14.
#     So the format's *paths and classes* are checked below and its JSON is not; that is a unit
#     test's job, on a toolchain that has the module.
Code.put_compiler_option(:no_warn_undefined, [
  JSON,
  Latu,
  Latu.ML,
  Latu.ML.Internal,
  Latu.ML.Linalg,
  Latu.ML.Result,
  Latu.Result.Literal
])

for f <- sources, do: Code.compile_file(Path.join(src, f))

alias Latu.ML.Plan
alias Latu.ML.Internal
alias Latu.Protocol.Spark.Connect, as: Proto

failures = :counters.new(1, [])

pass = fn label -> IO.puts("  ok   #{label}") end

fail = fn label, detail ->
  :counters.add(failures, 1, 1)
  IO.puts("  FAIL #{label}: #{detail}")
end

check = fn label, actual, expected ->
  if actual == expected do
    pass.(label)
  else
    fail.(label, "got #{inspect(actual)}, wanted #{inspect(expected)}")
  end
end

raises = fn label, fun, pattern ->
  try do
    fun.()
    fail.(label, "did not raise")
  rescue
    e in ArgumentError ->
      if Regex.match?(pattern, Exception.message(e)) do
        pass.(label)
      else
        fail.(label, "message was #{inspect(Exception.message(e))}")
      end
  end
end

IO.puts("\nliteral/2")
check.("int fits 32", Plan.literal(10, :int).literal_type, {:integer, 10})
check.("int past 32", Plan.literal(3_000_000_000, :int).literal_type, {:long, 3_000_000_000})
check.("long small is integer, as PySpark", Plan.literal(10, :long).literal_type, {:integer, 10})
check.("double from integer", Plan.literal(1, :double).literal_type, {:double, 1.0})
check.("float is double", Plan.literal(1.5, :float).literal_type, {:double, 1.5})
check.("int from whole float", Plan.literal(4.0, :int).literal_type, {:integer, 4})
check.("string from atom", Plan.literal(:features, :string).literal_type, {:string, "features"})
check.("boolean", Plan.literal(true, :boolean).literal_type, {:boolean, true})
raises.("fractional float is not an int", fn -> Plan.literal(4.5, :int) end, ~r/not a valid :int/)
raises.("integer is not a boolean", fn -> Plan.literal(1, :boolean) end, ~r/not a valid :boolean/)

raises.(
  "bare value is not a list",
  fn -> Plan.literal(:x, {:list, :string}) end,
  ~r/takes a list/
)

IO.puts("\narrays")
lit = Plan.literal([:x1, "x2"], {:list, :string})
{:array, array} = lit.literal_type
{:string, string_type} = array.element_type.kind

check.("no outer data_type, as PySpark writes it", lit.data_type, nil)
check.("strings carry Spark's default collation", string_type.collation, "UTF8_BINARY")

check.("elements", Enum.map(array.elements, & &1.literal_type), [
  {:string, "x1"},
  {:string, "x2"}
])

{:array, mixed} = Plan.literal([1, 3_000_000_000], {:list, :long}).literal_type

check.(
  "array element type is declared, not magnitude",
  Enum.map(mixed.elements, &(&1.literal_type |> elem(0))),
  [:long, :long]
)

IO.puts("\nparams/1 and Internal.params!/2")
params = Plan.params([{"maxIter", :int, 10}, {"regParam", :double, 0.01}])
check.("wire keys", params.params |> Map.keys() |> Enum.sort(), ["maxIter", "regParam"])
check.("empty is a map", Plan.params([]).params, %{})

known = [
  struct(Latu.ML.Param, %{name: :max_iter, wire: "maxIter", type: :int}),
  struct(Latu.ML.Param, %{name: :reg_param, wire: "regParam", type: :double})
]

check.("params! translates", Internal.params!([max_iter: 3], known), [{"maxIter", :int, 3}])

raises.(
  "unknown param lists the set",
  fn -> Internal.params!([max_iters: 3], known) end,
  ~r/unknown param :max_iters, expected one of \[:max_iter, :reg_param\]/
)

uid = Internal.uid("org.apache.spark.ml.classification.LogisticRegression")
check.("uid shape", Regex.match?(~r/^LogisticRegression_[0-9a-f]{12}$/, uid), true)

IO.puts("\noperator/3")
op = Plan.operator("org.example.LR", "LR_1", :estimator)
check.("kind", op.type, :OPERATOR_TYPE_ESTIMATOR)

raises.(
  "unknown kind",
  fn -> Plan.operator("c", "u", :regressor) end,
  ~r/expected one of \[:estimator, :transformer, :evaluator, :model\]/
)

IO.puts("\ncommands and relations")
input = Latu.Plan.relation({:range, :stand_in})
{:command, cmd} = Plan.fit(op, params, input).op_type
{:ml_command, ml} = cmd.command_type
{:fit, fit} = ml.command
check.("fit dataset", fit.dataset, input)
check.("fit estimator", fit.estimator, op)

{:command, del} = Plan.delete(["m-1", "m-2"]).op_type
{:ml_command, ml_del} = del.command_type
{:delete, delete} = ml_del.command
check.("delete refs", Enum.map(delete.obj_refs, & &1.id), ["m-1", "m-2"])

t1 = Plan.transform("m-1", input, Plan.params([]))
t2 = Plan.transform("m-1", input, Plan.params([]))
check.("fresh plan_id per relation", t1.common.plan_id != t2.common.plan_id, true)
{:ml_relation, rel} = t1.rel_type
{:transform, transform} = rel.ml_type
check.("obj_ref arm", transform.operator |> elem(1) |> Map.get(:id), "m-1")

t3 = Plan.transform(op, input, Plan.params([]))
{:ml_relation, rel3} = t3.rel_type
check.("transformer arm", rel3.ml_type |> elem(1) |> Map.get(:operator) |> elem(0), :transformer)

methods = [Plan.method(:summary), Plan.method("roc")]
{:command, fc} = Plan.fetch_command("m-1", methods).op_type
{:ml_command, mlf} = fc.command_type
{:fetch, from_command} = mlf.command
{:ml_relation, relf} = Plan.fetch_relation("m-1", methods).rel_type
{:fetch, from_relation} = relf.ml_type
check.("one Fetch, two wrappers", from_command, from_relation)
check.("method chain", Enum.map(from_command.methods, & &1.method), ["summary", "roc"])

{:ml_relation, rels} = Plan.fetch_relation("m-1", methods, input).rel_type
check.("summary dataset sits beside the fetch", rels.model_summary_dataset, input)

m = Plan.method(:predict, [Plan.literal(1, :int), input])
check.("args", Enum.map(m.args, &(&1.args_type |> elem(0))), [:param, :input])
raises.("bad arg", fn -> Plan.method(:p, [:nope]) end, ~r/literal or a relation/)

IO.puts("\nInternal, the struct-to-proto mapping the facade and the goldens share")

# `struct/2` rather than a literal: this file is compiled before `Code.compile_file` has
# defined these modules, so `%Latu.ML.Estimator{}` cannot expand here.
estimator =
  struct(Latu.ML.Estimator, %{
    class: "org.apache.spark.ml.classification.LogisticRegression",
    uid: "LR_fixture",
    params: [{"maxIter", :int, 7}]
  })

{:command, fit_cmd} = Internal.fit_plan(estimator, input).op_type
{:ml_command, fit_ml} = fit_cmd.command_type
{:fit, built} = fit_ml.command
check.("fit_plan names the estimator", built.estimator.name, estimator.class)
check.("fit_plan carries the uid", built.estimator.uid, "LR_fixture")
check.("fit_plan sends only what was set", Map.keys(built.params.params), ["maxIter"])
check.("fit_plan takes the dataset as given", built.dataset, input)

transformer =
  struct(Latu.ML.Transformer, %{
    class: "org.apache.spark.ml.feature.VectorAssembler",
    uid: "VA_fixture",
    params: [{"outputCol", :string, :features}]
  })

{:ml_relation, t_rel} = Internal.transform_relation(transformer, input).rel_type
{:transform, t_built} = t_rel.ml_type
check.("transform_relation uses the transformer arm", elem(t_built.operator, 0), :transformer)

model = struct(Latu.ML.Model, %{ref: "m-1", params: [{"threshold", :double, 0.7}]})
{:ml_relation, m_rel} = Internal.transform_relation(model, input).rel_type
{:transform, m_built} = m_rel.ml_type
check.("transform_relation uses the obj_ref arm", elem(m_built.operator, 0), :obj_ref)

check.(
  "a model resends its params, as PySpark does",
  m_built.params.params["threshold"].literal_type,
  {:double, 0.7}
)

{:command, fetch_cmd} = Internal.fetch_plan(model, :coefficients).op_type
{:ml_command, fetch_ml} = fetch_cmd.command_type
{:fetch, fetched} = fetch_ml.command
check.("fetch_plan names the ref", fetched.obj_ref.id, "m-1")
check.("fetch_plan sends one method", Enum.map(fetched.methods, & &1.method), ["coefficients"])

{:command, del_cmd} = Internal.delete_plan(model).op_type
{:ml_command, del_ml} = del_cmd.command_type
{:delete, deleted} = del_ml.command
check.("delete_plan names the ref", Enum.map(deleted.obj_refs, & &1.id), ["m-1"])

IO.puts("\nthe registry, and what it generates")

alias Latu.ML.Registry

check.("both files agree on the Spark version", Registry.spark(), "4.2.0")
check.("operators", length(Registry.operators()), 111)

kinds =
  Registry.operators()
  |> Enum.frequencies_by(& &1.kind)
  |> Enum.sort()

# The fifth kind, and the two that have it: neither `PowerIterationClustering` nor
# `PrefixSpan` is fitted or applied, and each has one method on the server's helper object.
check.("by kind", kinds,
  estimator: 42,
  evaluator: 6,
  helper: 2,
  model: 43,
  transformer: 18
)
check.("params", Enum.sum(Enum.map(Registry.operators(), &length(&1.params))), 991)
check.("classes with allowlisted attributes", length(Registry.classes()), 65)

check.(
  "attributes",
  Enum.sum(Enum.map(Registry.classes(), &length(&1.attributes))),
  669
)

# One generated function per non-model operator, in the module the registry names. Counted off
# the compiled modules rather than off the registry, so this fails if generation silently
# skipped a group — and matched against the operator names, because a group module may also
# carry a hand-written function (`Latu.ML.Feature.default_locale/1` is one).
operator_names = MapSet.new(Registry.operators(), & &1.name)

constructors =
  Registry.groups()
  |> Map.values()
  |> Enum.map(fn module ->
    Enum.count(
      module.__info__(:functions),
      fn {name, arity} -> arity == 1 and MapSet.member?(operator_names, name) end
    )
  end)
  |> Enum.sum()

check.("constructors", constructors, Enum.count(Registry.operators(), &(&1.kind != :model)))

modules_of = fn kind ->
  Registry.classes()
  |> Enum.filter(&(&1.kind == kind))
  |> Enum.map(& &1.module)
  |> Enum.filter(&Code.ensure_loaded?/1)
end

accessor_modules = modules_of.(:model) ++ modules_of.(:summary)

check.("model accessor modules", length(modules_of.(:model)), 42)
check.("summary accessor modules", length(modules_of.(:summary)), 22)

# 520 zero-argument names, the 93 that take arguments (all hundred but the seven
# `evaluate`s), and the four `trees`.
check.(
  "accessors and the three constructors, at every arity",
  Enum.sum(Enum.map(accessor_modules, &(length(&1.__info__(:functions)) - 1))),
  623
)

# Ten of the 43 model classes recorded a summary while fitting, and only those get a
# `summary/1`. The other 33 raise from `Latu.ML.summary/1` naming the ten.
check.(
  "models with a summary",
  Enum.count(modules_of.(:model), &function_exported?(&1, :summary, 1)),
  10
)

# `trees` is the fourth answer shape, found by the probe rather than read off PySpark: the
# server answers it on `operator_info` with refs to sub-models, each of them a cache entry it
# registered on the way out.
generated_names =
  for module <- accessor_modules,
      {name, 1} <- module.__info__(:functions),
      into: MapSet.new(),
      do: name

operator_shaped =
  for row <- Registry.classes(),
      attribute <- row.attributes,
      attribute.returns == :operators,
      do: attribute

check.(
  "there are four attributes that answer with sub-model refs, all `trees`",
  Enum.map(operator_shaped, & &1.wire),
  ~w(trees trees trees trees)
)

check.(
  "and each one has an accessor now",
  Enum.reject(operator_shaped, &MapSet.member?(generated_names, &1.name)),
  []
)

# Not guessable, and the reason `:element_class` is read rather than derived: gradient boosting
# fits **regression** trees to residuals whatever it is classifying, so a GBT classifier's trees
# are not classification trees.
check.(
  "the element class is read, not guessed from the owner",
  Map.new(
    for row <- Registry.classes(),
        attribute <- row.attributes,
        attribute.returns == :operators,
        do: {row.python, attribute.element_class |> String.split(".") |> List.last()}
  ),
  %{
    "GBTClassificationModel" => "DecisionTreeRegressionModel",
    "GBTRegressionModel" => "DecisionTreeRegressionModel",
    "RandomForestClassificationModel" => "DecisionTreeClassificationModel",
    "RandomForestRegressionModel" => "DecisionTreeRegressionModel"
  }
)

# Every constructor the registry advertises actually exists, at the arity it advertises. The
# registry's `:constructor` is what `apply/3` and `h` are pointed at, so a wrong one is a
# broken promise rather than a cosmetic slip.
missing =
  for %{constructor: {module, name, arity}} <- Registry.operators(),
      Code.ensure_loaded!(module),
      not function_exported?(module, name, arity),
      do: "#{inspect(module)}.#{name}/#{arity}"

check.("every advertised constructor exists", missing, [])

# `priv/ml_probe.exs` is the one registry file that may be absent, and the package has to
# build without it. Absent, everything is `:built` with no reason; present, the statuses are
# whatever a server said.
statuses = Registry.operators() |> Enum.map(& &1.status) |> Enum.uniq()

check.("every status is one of the three", statuses -- [:probed, :built, :missing], [])

check.(
  "a probed operator keeps no reason",
  Enum.filter(Registry.operators(), &(&1.status == :probed and &1.reason != nil)),
  []
)

lr = Latu.ML.Classification.logistic_regression(max_iter: 10, reg_param: 0.01)
check.("generated constructor builds an estimator", lr.__struct__, Latu.ML.Estimator)
check.("with the JVM class", lr.class, "org.apache.spark.ml.classification.LogisticRegression")

check.(
  "and the model class a Fit does not answer with",
  lr.model_class,
  "org.apache.spark.ml.classification.LogisticRegressionModel"
)

check.("sending only what was set", lr.params, [
  {"maxIter", :int, 10},
  {"regParam", :double, 0.01}
])

# `LDA` is the one estimator that fits two classes — `LocalLDAModel` or `DistributedLDAModel`,
# on its `optimizer` — so it claims neither and keeps both. `dev/probe_ml.exs` found this the
# same way it found the summary case: an attribute with arguments had nothing to look them up in.
lda = Latu.ML.Clustering.lda(k: 2)
check.("an estimator that fits two classes claims neither", lda.model_class, nil)

check.("and keeps both, in PySpark's own order", lda.model_candidates, [
  "org.apache.spark.ml.clustering.DistributedLDAModel",
  "org.apache.spark.ml.clustering.LocalLDAModel"
])

lda_model = struct(Latu.ML.Model, %{ref: "m-4", session: :stand_in, candidates: lda.model_candidates})

check.(
  "and an LDA model finds an argument's type in them",
  Internal.arguments!(lda_model, "describeTopics", [2]),
  [Plan.literal(2, :int)]
)

# The probe measured this: `LocalLDAModel` and `DistributedLDAModel` are siblings, so their
# union is no real class's allowlist and four of its names refuse on a `LocalLDAModel`. What
# `attributes/1` may promise is what they share.
local = Latu.ML.Registry.class("org.apache.spark.ml.clustering.LocalLDAModel").attributes
distributed = Latu.ML.Registry.class("org.apache.spark.ml.clustering.DistributedLDAModel").attributes
shared = MapSet.intersection(MapSet.new(local, & &1.wire), MapSet.new(distributed, & &1.wire))

check.(
  "the candidates really do differ, which is why this matters",
  MapSet.size(shared) < length(distributed),
  true
)

# `Latu.ML.attributes/1` itself is out of reach here (it lives on `Latu.ML`), so this pins the
# rule it implements rather than the function.
check.(
  "and `to_local` is one of the names only one candidate allows",
  MapSet.member?(shared, "toLocal"),
  false
)

check.(
  "a generated transformer",
  Latu.ML.Feature.vector_assembler(input_cols: [:x1]).__struct__,
  Latu.ML.Transformer
)

check.(
  "a generated evaluator",
  Latu.ML.Evaluation.regression_evaluator(metric_name: "rmse").__struct__,
  Latu.ML.Evaluator
)

raises.(
  "an unknown param still lists the operator's own",
  fn -> Latu.ML.Classification.logistic_regression(max_iters: 10) end,
  ~r/unknown param :max_iters, expected one of \[:aggregation_depth/
)

raises.(
  "an unknown operator names the near spellings",
  fn -> Registry.fetch!(:logistic_regresion) end,
  ~r/no operator :logistic_regresion. Did you mean one of .*:logistic_regression/
)

IO.puts("\naccessors")

alias Latu.ML.Clustering.KMeansModel

check.("carry their class", KMeansModel.class(), "org.apache.spark.ml.clustering.KMeansModel")

check.(
  "one function per allowlisted attribute this package can send",
  KMeansModel.__info__(:functions) |> Keyword.keys() |> Enum.sort(),
  [:class, :cluster_center_matrix, :has_summary, :num_features, :predict, :summary]
)

check.(
  "and an argument-taker is generated at 1 + its arguments",
  KMeansModel.__info__(:functions)[:predict],
  2
)

kmeans = struct(Latu.ML.Model, %{ref: "m-1", class: "org.apache.spark.ml.clustering.KMeansModel"})

raises.(
  "and refuse a model of another class, naming the module that would take it",
  fn ->
    Latu.ML.Internal.attribute(
      kmeans,
      "org.apache.spark.ml.classification.LogisticRegressionModel",
      "coefficients"
    )
  end,
  ~r/coefficients is an attribute of .*LogisticRegressionModel.*Try Latu.ML.Clustering.KMeans/
)

{:ml_relation, frame_fetch} = Internal.fetch_relation(kmeans, "clusterCenterMatrix").rel_type
{:fetch, fetched_frame} = frame_fetch.ml_type
check.("a frame attribute rides a relation", fetched_frame.obj_ref.id, "m-1")
check.("with no summary dataset", frame_fetch.model_summary_dataset, nil)

IO.puts("\nsummaries")

# A summary has no cache key of its own: the Fetch names the *model* and puts `summary` at the
# head of the chain. That is PySpark's dotted `"<model_ref>.summary"`, kept as parts.
lr_model =
  struct(Latu.ML.Model, %{
    ref: "m-1",
    class: "org.apache.spark.ml.classification.LogisticRegressionModel",
    dataset: input
  })

lr_summary = Internal.summary(lr_model)
check.("a summary names the model", lr_summary.ref, "m-1")
check.("and leads with `summary`", lr_summary.methods, ["summary"])

check.(
  "a class the server decides is left unclaimed",
  lr_summary.class,
  nil
)

check.(
  "but both candidates are kept, in PySpark's own order",
  lr_summary.candidates,
  [
    "org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary",
    "org.apache.spark.ml.classification.LogisticRegressionTrainingSummary"
  ]
)

# What `dev/probe_ml.exs` found the hard way: without the candidates there is nothing to look
# an argument's type up in, and a summary of a model-dependent class could not be asked at all.
check.(
  "and an argument's type is found in them",
  Internal.arguments!(lr_summary, "fMeasureByLabel", [1.0]),
  [Plan.literal(1.0, :double)]
)

raises.(
  "a name in neither candidate says so",
  fn -> Internal.arguments!(lr_summary, "nonesuch", [1.0]) end,
  ~r/None of \S+BinaryLogisticRegressionTrainingSummary or \S+ has it/
)

check.("and the raw training frame is what rebuilds it", lr_summary.dataset, input)

# The three clustering models want the model's own transform of the training frame instead,
# which is `HasTrainingSummary`'s default and the reason the registry records which.
kmeans_summary =
  Internal.summary(
    struct(Latu.ML.Model, %{
      ref: "k-1",
      class: "org.apache.spark.ml.clustering.KMeansModel",
      dataset: input
    })
  )

check.("KMeans knows its summary class", kmeans_summary.class,
  "org.apache.spark.ml.clustering.KMeansSummary")

check.(
  "and is rebuilt from its own transform",
  match?({:ml_relation, _}, kmeans_summary.dataset.rel_type),
  true
)

raises.(
  "a model with no summary refuses, naming the ones that have one",
  fn ->
    Internal.summary(struct(Latu.ML.Model, %{ref: "p", class: "org.apache.spark.ml.feature.PCAModel"}))
  end,
  ~r/has no training summary. The ones that do: .*LogisticRegressionModel/
)

{:command, sfc} = Internal.fetch_plan(lr_summary, "areaUnderROC").op_type
{:ml_command, sml} = sfc.command_type
{:fetch, summary_fetch} = sml.command
check.("a summary fetch names the model", summary_fetch.obj_ref.id, "m-1")

check.(
  "and chains through `summary`",
  Enum.map(summary_fetch.methods, & &1.method),
  ["summary", "areaUnderROC"]
)

{:ml_relation, sfr} = Internal.fetch_relation(lr_summary, "roc").rel_type
{:fetch, summary_frame} = sfr.ml_type
check.("a summary frame chains the same way", Enum.map(summary_frame.methods, & &1.method), [
  "summary",
  "roc"
])

check.(
  "and carries the frame that rebuilds it, which a model's own never does",
  sfr.model_summary_dataset,
  input
)

{:command, csc} = Internal.create_summary_plan(lr_summary).op_type
{:ml_command, csml} = csc.command_type
{:create_summary, created} = csml.command
check.("CreateSummary names the model", created.model_ref.id, "m-1")
check.("and carries the frame", created.dataset, input)

IO.puts("\nattributes that take arguments")

# Nothing here builds a Vector literal: that is `Latu.ML.Linalg`, which needs Nx, which this
# file is built to run without (see the header). The types are still *checked* against vectors
# below, because a refusal happens before anything is built.

als =
  struct(Latu.ML.Model, %{
    ref: "m-1",
    session: :stand_in,
    class: "org.apache.spark.ml.recommendation.ALSModel"
  })

last_method = fn holder, name, values ->
  {:command, command} =
    holder
    |> Internal.fetch_plan(name, Internal.arguments!(holder, name, values))
    |> Map.get(:op_type)
    |> then(fn {:command, c} -> {:command, c} end)

  {:ml_command, ml_command} = command.command_type
  {:fetch, fetched} = ml_command.command

  List.last(fetched.methods)
end

users = last_method.(als, "recommendForAllUsers", [5])
check.("recommendForAllUsers names its method", users.method, "recommendForAllUsers")
check.("and carries one argument", length(users.args), 1)
check.("an int argument rides the literal arm", hd(users.args).args_type, {:param, Plan.literal(5, :int)})

# A bare map, not `struct/2`: Latu itself is not compiled here, and `Internal.argument/2`
# checks with `is_struct/2`, which reads the field rather than loading the module.
frame = %{__struct__: Latu.DataFrame, session: :stand_in, plan: input}
subset = last_method.(als, "recommendForItemSubset", [frame, 5])
check.("a frame and an int, in the declared order", length(subset.args), 2)
check.("the frame rides the relation arm", hd(subset.args).args_type, {:input, input})

# `Word2VecModel.findSynonyms(word)` is 4.2.0's only union-typed argument. The value's shape
# picks the member; the vector half of that choice is the probe's to confirm.
w2v =
  struct(Latu.ML.Model, %{
    ref: "m-2",
    session: :stand_in,
    class: "org.apache.spark.ml.feature.Word2VecModel"
  })

synonyms = last_method.(w2v, "findSynonyms", ["king", 3])
check.("a union argument takes the string arm for a string", hd(synonyms.args).args_type, {:param, Plan.literal("king", :string)})

lr_model =
  struct(Latu.ML.Model, %{
    ref: "m-3",
    session: :stand_in,
    class: "org.apache.spark.ml.classification.LogisticRegressionModel"
  })

raises.(
  "a wrong count names the arguments and their types",
  fn -> Internal.arguments!(lr_model, "predict", []) end,
  ~r/predict takes 1 arguments — value \(:vector\)/
)

raises.(
  "a frame argument that is not a frame",
  fn -> Internal.arguments!(als, "recommendForItemSubset", [7, 5]) end,
  ~r/dataset is a DataFrame argument/
)

raises.(
  "and a zero-argument attribute says where to go instead",
  fn -> Internal.arguments!(lr_model, "intercept", [1.0]) end,
  ~r/takes no arguments; use Latu.ML.attribute\/2/
)

check.("no arguments is still no arguments", Internal.arguments!(lr_model, "intercept", []), [])

IO.puts("\nerror classes")

# `Latu.ML.error_kind/1` and `hint/1` live on `Latu.ML`, which needs Latu itself and is out of
# reach here. What can be checked without it is the thing most likely to be wrong: that the
# table is keyed on the **dotted** spelling the server actually sends, not the bare one Spark's
# own documentation and this package's roadmap both write.
kinds =
  Regex.scan(~r/"(CONNECT_ML\.[A-Z_.]+)" => :(\w+)/, File.read!(Path.join(src, "lib/latu/ml.ex")))

check.(
  "four classes are keyed by their dotted spelling",
  Enum.map(kinds, &Enum.at(&1, 1)) |> Enum.sort(),
  [
    "CONNECT_ML.ATTRIBUTE_NOT_ALLOWED",
    "CONNECT_ML.CACHE_INVALID",
    "CONNECT_ML.ML_CACHE_SIZE_OVERFLOW_EXCEPTION",
    "CONNECT_ML.MODEL_SIZE_OVERFLOW_EXCEPTION"
  ]
)

check.(
  "and the fifth comes from the constant the recovery already matched on",
  File.read!(Path.join(src, "lib/latu/ml.ex")) =~ "@summary_lost => :summary_lost",
  true
)

check.(
  "no class is written bare anywhere in the table",
  Enum.any?(kinds, fn [_whole, class, _atom] -> not String.starts_with?(class, "CONNECT_ML.") end),
  false
)

IO.puts("\nsub-model refs")

# The server joins the ids with commas — `MLHandler` does `ids.mkString(",")` — because
# `MlOperatorInfo` has one `ObjectRef` and no repeated field to put several in.
joined =
  struct(Proto.MlCommandResult, %{
    result_type:
      {:operator_info,
       struct(Proto.MlCommandResult.MlOperatorInfo, %{
         type: {:obj_ref, struct(Proto.ObjectRef, %{id: "t-1,t-2,t-3"})}
       })}
  })

check.(
  "several ids arrive as one comma-joined ref",
  Latu.ML.Result.operators(%{ml_command_result: joined}),
  {:ok, ["t-1", "t-2", "t-3"]}
)

forest =
  struct(Latu.ML.Model, %{
    ref: "m-5",
    session: :stand_in,
    class: "org.apache.spark.ml.classification.RandomForestClassificationModel"
  })

{:ok, trees} = Internal.operators(forest, "trees", %{ml_command_result: joined})

check.("and become models, one per id", Enum.map(trees, & &1.ref), ["t-1", "t-2", "t-3"])

check.(
  "each carrying the element class, not the owner's",
  trees |> hd() |> Map.get(:class),
  "org.apache.spark.ml.classification.DecisionTreeClassificationModel"
)

check.("and the session they were reached through", trees |> hd() |> Map.get(:session), :stand_in)

IO.puts("\ninspect")

check.(
  "a model leads with its class, then a short ref",
  inspect(lr_model),
  ~s(#Latu.ML.Model<LogisticRegressionModel "m-3">)
)

check.(
  "a model whose class the fit decides shows both",
  inspect(lda_model),
  ~s(#Latu.ML.Model<DistributedLDAModel | LocalLDAModel "m-4">)
)

check.(
  "a summary too",
  inspect(lr_summary),
  ~s(#Latu.ML.Summary<BinaryLogisticRegressionTrainingSummary | ) <>
    ~s(LogisticRegressionTrainingSummary "m-1">)
)

check.(
  "an estimator shows only the params that were set",
  inspect(Latu.ML.Classification.logistic_regression(max_iter: 10)),
  ~s(#Latu.ML.Estimator<LogisticRegression [maxIter: 10]>)
)

check.(
  "and nothing where nothing was set",
  inspect(Latu.ML.Classification.logistic_regression()),
  "#Latu.ML.Estimator<LogisticRegression>"
)

# A uuid is abbreviated at its first group, which is what a real ref looks like.
check.(
  "a real cache id is cut short",
  inspect(struct(Latu.ML.Model, %{ref: "4b8aa52d-0d65-48e0-864a-3c1d76a5df4a", session: nil})),
  ~s{#Latu.ML.Model<? "4b8aa52d…">}
)

IO.puts("\nthe cache verbs")

arm = fn plan ->
  {:command, command} = plan.op_type
  {:ml_command, ml_command} = command.command_type
  ml_command.command
end

{:get_model_size, sized} = arm.(Plan.get_model_size("m-1"))
check.("GetModelSize names the model", sized.model_ref.id, "m-1")

# `struct/2` and `__struct__`, not a literal, for the reason given above `estimator`.
{:get_cache_info, info} = arm.(Plan.get_cache_info())
check.("GetCacheInfo is the empty message", info, struct(Proto.MlCommand.GetCacheInfo))

{:clean_cache, cleaned} = arm.(Plan.clean_cache())
check.("CleanCache is the empty message", cleaned, struct(Proto.MlCommand.CleanCache))

IO.puts("\npipelines")

alias Latu.ML.Layout

# A pipeline has no server-side anything: no `Fit`, no cache entry, no `Read`. What it has is a
# directory this package writes, and these are the parts of that format that are pure.
check.("the class the metadata is written under", Layout.class(:pipeline_model),
  "pyspark.ml.pipeline.PipelineModel"
)

check.("and back the other way", Layout.kind("pyspark.ml.pipeline.Pipeline"), :pipeline)
check.("a class that is not one of ours", Layout.kind("org.apache.spark.ml.feature.PCA"), nil)

# `getStagePath`: the index is padded to the width of the largest one, so a nine-stage pipeline
# and a ten-stage one name their first stage differently.
check.("a stage path pads to the count's width", Layout.stage_path("/tmp/p", 0, "PCA_1", 9),
  "/tmp/p/stages/0_PCA_1"
)

check.("...and to two digits at ten", Layout.stage_path("/tmp/p", 3, "PCA_1", 10),
  "/tmp/p/stages/03_PCA_1"
)

check.(
  "which is what makes a saved pipeline readable by PySpark at all",
  Layout.stage_path("s3://bucket/model", 11, "LogisticRegression_ab", 12),
  "s3://bucket/model/stages/11_LogisticRegression_ab"
)

check.(
  "load/3 takes the names that name no operator",
  Internal.loadable!(:pipeline_model),
  {"pyspark.ml.pipeline.PipelineModel", :pipeline_model, nil}
)

check.("and lists them", Enum.sort(Internal.meta_algorithms()), [
  :cross_validator,
  :cross_validator_model,
  :pipeline,
  :pipeline_model,
  :train_validation_split,
  :train_validation_split_model
])

# Two shapes of directory now, both read by their `class`. ML5a could assume every known class
# here was a pipeline; a validator's is not, and `shape/1` is what says so.
check.("a pipeline lays out stages", Layout.shape(:pipeline_model), :stages)
check.("and a validator lays out a search", Layout.shape(:cross_validator_model), :search)

check.(
  "a search names its parts the way PySpark does",
  Enum.map([:estimator, :evaluator, :best_model], &Layout.part_path("/p", &1)),
  ["/p/estimator", "/p/evaluator", "/p/bestModel"]
)

# A fold index where there are folds, and none where there is one split.
check.("a cross-validated sub-model sits under its fold", Layout.sub_model_path("/p", 2, 5),
  "/p/subModels/fold2/5"
)

check.("and a split one sits directly", Layout.sub_model_path("/p", nil, 5), "/p/subModels/5")

# The grid encoding needs `JSON`, which this container's Elixir does not have — it is in
# `no_warn_undefined` above for exactly that reason. Those checks are in `test/latu/ml_test.exs`
# beside the metadata ones, which are absent here for the same reason.

IO.puts("\nthe helper object")

# `ConnectHelper` is in the same allowlist a model's attributes are in and is an attribute of
# nothing: a singleton the server resolves by name. Its names come from that Scala and its
# argument types from PySpark's call sites, which is the only place they exist.
helpers = Registry.helpers()

check.("methods", length(helpers), 11)

check.(
  "by shape",
  helpers |> Enum.frequencies_by(& &1.returns) |> Enum.sort(),
  frame: 5, model: 3, value: 3
)

# Ten are read off PySpark's call sites; `handleOverwrite` has none, so its arguments are pinned
# in the extractor and the row says so. A pipeline is what needed it.
check.(
  "and one of them was read by hand, marked as such",
  Enum.filter(helpers, & &1.read_by_hand) |> Enum.map(& &1.name),
  [:handle_overwrite]
)

check.(
  "every argument is typed",
  Enum.filter(helpers, fn h -> Enum.any?(h.args, &(&1.type == nil)) end),
  []
)

# The five that answer with a DataFrame all take it first, which is what lets `helper_frame/2`
# find a session without being handed one.
check.(
  "a frame-valued helper takes its frame first",
  Enum.all?(Enum.filter(helpers, &(&1.returns == :frame)), &(hd(&1.args).type == :frame)),
  true
)

chi = Registry.helper!(:chi_square_test)

# A helper answering with a frame is a *relation*, so it unwraps one layer differently from
# every command above: no `op_type`, because nothing has wrapped it in a plan yet.
ml_arm = fn relation ->
  {:ml_relation, ml_relation} = relation.rel_type
  ml_relation.ml_type
end

# `struct/2` rather than a literal, for the reason given above `estimator`: a script is
# compiled before it runs, so a struct literal cannot see a module the script requires.
frame_stand_in = struct(Latu.DataFrame, %{session: :stand_in, plan: input})
{:fetch, fetch} =
  ml_arm.(Internal.helper_relation_plan(chi, [frame_stand_in, :features, :label, true]))

check.(
  "named by the literal id, not a cache entry",
  fetch.obj_ref.id,
  "______ML_CONNECT_HELPER______"
)

check.(
  "one method, the server's own spelling",
  Enum.map(fetch.methods, & &1.method),
  ["chiSquareTest"]
)

check.(
  "the frame rides the relation arm and the rest are literals",
  Enum.map(hd(fetch.methods).args, &elem(&1.args_type, 0)),
  [:input, :param, :param, :param]
)

raises.(
  "a wrong count names the arguments and their types",
  fn -> Internal.helper_arguments!(chi, [frame_stand_in]) end,
  ~r/chiSquareTest takes 4 arguments — dataset \(:frame\), features_col/
)

# A nested list, which `literal/2` refused until ML4b. Two things need it: this helper, and
# `Bucketizer.splitsArray`, the one nested param in 4.2.0.
nested = Plan.literal([["a", "b"], ["c"]], {:list, {:list, :string}})
{:array, outer} = nested.literal_type
{:array, element} = outer.element_type.kind

check.(
  "an array of arrays declares an array element type",
  element.element_type.kind |> elem(0),
  :string
)
check.("PySpark's containsNull default", element.contains_null, true)

check.(
  "and each element is an array literal of its own",
  Enum.map(outer.elements, fn element ->
    element.literal_type |> elem(1) |> Map.get(:elements) |> length()
  end),
  [2, 1]
)

check.(
  "which is what Bucketizer.splitsArray needs too",
  Plan.literal([[0.0, 1.0]], {:list, {:list, :double}}).literal_type |> elem(0),
  :array
)

# A helper's arguments are positional, so every one is sent. The value is the caller's, then
# the param's default, then the fallback PySpark's own call site records — and `weightCol` is
# the only one on 4.2.0 that needs the third.
pic = Latu.ML.Clustering.power_iteration_clustering(k: 3)
row = Registry.helper!(:power_iteration_clustering_assign_clusters)

check.("a helper operator is its own struct", pic.__struct__, Latu.ML.Helper)
check.("and carries no uid, because nothing sends one", Map.has_key?(pic, :uid), false)

check.(
  "set, then default, then the call site's fallback",
  Internal.operator_values!(row, pic, :frame_stand_in),
  [:frame_stand_in, 3, 20, "random", "src", "dst", ""]
)

# The three that build a model server-side get a generated constructor on the model's own
# module, because that is the class they make.
check.(
  "from_labels is generated onto StringIndexerModel",
  function_exported?(Latu.ML.Feature.StringIndexerModel, :from_labels, 3),
  true
)

check.(
  "and takes a session, the labels, and params for afterwards",
  Latu.ML.Feature.StringIndexerModel.__info__(:functions)
  |> Enum.filter(&(elem(&1, 0) == :from_labels)),
  [from_labels: 2, from_labels: 3]
)

check.(
  "the nested one too",
  function_exported?(Latu.ML.Feature.StringIndexerModel, :from_arrays_of_labels, 3),
  true
)

check.(
  "and the vocabulary one, on its own class",
  function_exported?(Latu.ML.Feature.CountVectorizerModel, :from_vocabulary, 3),
  true
)

IO.puts("\nsave and load")

saved_model =
  struct(Latu.ML.Model, %{
    ref: "m-6",
    session: :stand_in,
    class: "org.apache.spark.ml.classification.LogisticRegressionModel",
    params: [{"maxIter", :int, 10}]
  })

{:write, model_write} = arm.(Internal.save_plan(saved_model, "/tmp/lr", false, %{}))

check.(
  "a model is written by reference",
  model_write.type,
  {:obj_ref, struct(Proto.ObjectRef, %{id: "m-6"})}
)
check.("carrying its params", Map.keys(model_write.params.params), ["maxIter"])
check.("and the path", model_write.path, "/tmp/lr")

# Explicit presence: PySpark sets this on every write, so `false` is sent rather than left out.
check.("should_overwrite is set even when false", model_write.should_overwrite, false)
check.("options default to none", model_write.options, %{})

{:write, operator_write} =
  arm.(Internal.save_plan(lr, "/tmp/lr-unfitted", true, %{"a" => "b"}))

{:operator, written} = operator_write.type
check.("an unfitted estimator is written by class", written.name, lr.class)
check.("with its uid, which the server has no other way to learn", written.uid, lr.uid)
check.("and its kind", written.type, :OPERATOR_TYPE_ESTIMATOR)
check.("overwrite goes through", operator_write.should_overwrite, true)
check.("and so do the writer's options", operator_write.options, %{"a" => "b"})

{:read, read} = arm.(Internal.load_plan(saved_model.class, :model, "/tmp/lr"))

check.("a Read names the class", read.operator.name, saved_model.class)
check.("as a model, which is what caches it", read.operator.type, :OPERATOR_TYPE_MODEL)
check.("with no uid: the saved metadata has it", read.operator.uid, "")
check.("and the path", read.path, "/tmp/lr")

check.(
  "an operator name resolves to its class and kind",
  Internal.loadable!(:logistic_regression),
  {lr.class, :estimator, Registry.fetch!(:logistic_regression)}
)

{class, kind, _row} = Internal.loadable!(:logistic_regression_model)
check.("and so does a model's own name", {class, kind}, {saved_model.class, :model})

check.(
  "a generated model module means the class it accesses",
  Internal.loadable!(Latu.ML.Classification.LogisticRegressionModel),
  {saved_model.class, :model, nil}
)

check.(
  "and a class this package never heard of is named outright",
  Internal.loadable!({"com.example.Thing", :transformer}),
  {"com.example.Thing", :transformer, nil}
)

raises.(
  "a summary module is refused, since nothing saves one",
  fn -> Internal.loadable!(Latu.ML.Classification.BinaryLogisticRegressionSummary) end,
  ~r/is a training summary, and a summary is not saved or loaded/
)

raises.(
  "and so is anything else, naming the three spellings",
  fn -> Internal.loadable!("org.apache.spark.ml.classification.LogisticRegression") end,
  ~r/names nothing to load/
)

info = %{
  ref: "m-7",
  name: nil,
  uid: "LogisticRegressionModel_abc",
  params: [{"maxIter", :int, 5}]
}
loaded = Internal.loaded(:stand_in, Internal.loadable!(:logistic_regression_model), info)

check.("a loaded model holds the reference the read registered", loaded.ref, "m-7")
check.("the uid the metadata carried, not a generated one", loaded.uid, info.uid)
check.("and the params it was saved with", loaded.params, info.params)

# The consequence that is easy to miss: a saved model does not carry its summary, so a loaded
# one has neither a summary on the server nor a frame to rebuild one from.
check.("and no training frame at all", loaded.dataset, nil)

loaded_estimator =
  Internal.loaded(:stand_in, Internal.loadable!(:logistic_regression), %{info | ref: nil})

check.("a loaded estimator caches nothing", Map.has_key?(loaded_estimator, :ref), false)
check.("and knows what it fits", loaded_estimator.model_class, saved_model.class)
check.("and its param table, so a later set can be checked", loaded_estimator.known != [], true)

check.(
  "a summary of a loaded model has no frame to rebuild from",
  Internal.summary(loaded).dataset,
  nil
)


# `load/4` dispatches on `loadable!/1`, and **two different things** answer with nil in the
# third slot: a meta-algorithm, which this package reads itself, and a `{class, kind}` pair,
# which the server reads. A clause matching nil alone routes every operator stage into the
# meta-algorithm path — which is what happened, and only the integration tests noticed.
check.(
  "a meta-algorithm answers with no registry row",
  Internal.loadable!(:cross_validator_model),
  {"pyspark.ml.tuning.CrossValidatorModel", :cross_validator_model, nil}
)

check.(
  "and so does a {class, kind} pair, which is why nil alone cannot be the test",
  Internal.loadable!({"org.apache.spark.ml.feature.VectorAssembler", :transformer}),
  {"org.apache.spark.ml.feature.VectorAssembler", :transformer, nil}
)

check.(
  "the kinds that are directories this package wrote",
  Enum.sort(Internal.meta_algorithms()) -- [:transformer, :estimator, :model, :evaluator],
  Enum.sort(Internal.meta_algorithms())
)


# The row is what carries the param table, and a nil row is a real answer — `{class, kind}`
# gives one. So a load path that resolves an operator and then passes `{class, kind}` onward
# throws the table away, which is what `load_operator_stage` did from ML5a until ML5c needed
# to look a wire name up in a loaded stage.
check.(
  "a loaded operator with its row knows its params",
  Internal.loaded(nil, {"org.apache.spark.ml.regression.LinearRegression", :estimator,
    Registry.fetch!(:linear_regression)}, %{uid: "LR_1", ref: nil, params: []}).known != [],
  true
)

check.(
  "and without one it knows none, which is why the row has to be threaded",
  Internal.loaded(nil, {"org.apache.spark.ml.regression.LinearRegression", :estimator, nil},
    %{uid: "LR_1", ref: nil, params: []}).known,
  []
)

IO.puts("\nthe facade against itself")

# `lib/latu/ml.ex` is the one file this script cannot compile: it calls `Latu.Client`, which is
# the whole point of the layering. So nothing here has ever checked the facade for a call to a
# function that does not exist — and twice now one has reached `mix compile` instead.
#
# This is not a compiler. It reads the file's own AST, collects what it defines and what it
# calls unqualified, and reports a call whose *name* the module defines at some arity but not
# at that one — plus a call to a name it never defines at all. Kernel and imported names are
# excluded by only considering names the module itself mentions in a definition.

# Since 2026-09-11 the same file split three ways: `Latu.ML.Persistence` (save and load) and
# `Latu.ML.Cache` (what owns a cached model) call `Latu.ML` and each other, so all three are
# read here, each against itself.
lint = fn path, floor ->
  {:ok, facade_ast} = Code.string_to_quoted(File.read!(Path.join(src, path)))

  # Pipes first: `x |> f()` is `f/0` in the AST and `f/1` in fact.
  {piped, _} =
    Macro.prewalk(facade_ast, nil, fn
      {:|>, _meta, [left, {name, meta, args}]}, acc when is_atom(name) ->
        {{name, meta, [left | args || []]}, acc}

      node, acc ->
        {node, acc}
    end)

  arities = fn args ->
    optional = Enum.count(args, &match?({:\\, _, _}, &1))
    Enum.to_list((length(args) - optional)..length(args))
  end

  {_, defined} =
    Macro.prewalk(piped, MapSet.new(), fn
      {kind, _, [{:when, _, [{name, _, args} | _]} | _]} = node, acc
      when kind in [:def, :defp] and is_list(args) ->
        {node, Enum.reduce(arities.(args), acc, &MapSet.put(&2, {name, &1}))}

      {kind, _, [{name, _, args} | _]} = node, acc
      when kind in [:def, :defp] and is_list(args) ->
        {node, Enum.reduce(arities.(args), acc, &MapSet.put(&2, {name, &1}))}

      {:defdelegate, _, [{name, _, args} | _]} = node, acc when is_list(args) ->
        {node, MapSet.put(acc, {name, length(args)})}

      node, acc ->
        {node, acc}
    end)

  {_, calls} =
    Macro.prewalk(piped, [], fn
      {name, meta, args} = node, acc when is_atom(name) and is_list(args) ->
        {node, [{name, length(args), meta[:line]} | acc]}

      node, acc ->
        {node, acc}
    end)

  known = MapSet.new(defined, fn {name, _arity} -> name end)

  unresolved =
    calls
    |> Enum.filter(fn {name, arity, _line} ->
      MapSet.member?(known, name) and not MapSet.member?(defined, {name, arity})
    end)
    |> Enum.uniq_by(fn {name, arity, _line} -> {name, arity} end)
    |> Enum.sort()

  check.("#{path} defines what it calls", unresolved, [])
  check.("and it is a surface worth the check", MapSet.size(defined) > floor, true)
end

lint.("lib/latu/ml.ex", 60)
lint.("lib/latu/ml/persistence.ex", 30)
lint.("lib/latu/ml/cache.ex", 5)

IO.puts("\ntuning")

alias Latu.ML.Evaluation

# `isLargerBetter` is not on the server's allowlist, so no `Fetch` can ask for it and no probe
# can settle it. What settles it is that PySpark's six evaluators each override the method
# client-side — an expression the extractor reads. These checks are the whole guarantee that
# the reading was right, because a wrong answer here does not fail loudly: it quietly calls the
# worst model in a grid the best one.

evaluators = Enum.filter(Registry.operators(), &(&1.kind == :evaluator))

check.("every evaluator carries one", Enum.count(evaluators, &(&1.larger_better != nil)), 6)

check.(
  "and nothing else does",
  Enum.count(Registry.operators(), &(&1.kind != :evaluator and &1.larger_better != nil)),
  0
)

default_metric = fn operator ->
  Enum.find_value(operator.params, fn param ->
    if param.name == :metric_name, do: param.default
  end)
end

# Every metric the extraction actually names, resolved through the public function. The list is
# derived from the registry rather than written out here, so a metric moving between the two
# lists upstream changes what is checked instead of slipping past a fixed table.
for operator <- evaluators do
  {module, name, 1} = operator.constructor

  named =
    case operator.larger_better do
      true -> []
      {_shape, metrics} -> metrics
    end

  for metric <- [default_metric.(operator) | named] do
    expected =
      case operator.larger_better do
        true -> true
        {:only, metrics} -> metric in metrics
        {:except, metrics} -> metric not in metrics
      end

    check.(
      "#{name}, #{metric}",
      Internal.larger_better?(apply(module, name, [[metric_name: metric]])),
      expected
    )
  end
end

# A metric never set at all has to resolve the way the set default does. The registry's
# `:default` is the fallback, and this is the only thing that says so.
check.(
  "an unset metric_name falls back to the default",
  Internal.larger_better?(Evaluation.regression_evaluator()),
  Internal.larger_better?(Evaluation.regression_evaluator(metric_name: "rmse"))
)

# Five answers read straight off PySpark's overrides and pinned by hand. If these and the
# derivation above ever disagree, the derivation is the one that moved.
for {module, name, metric, expected} <- [
      {Evaluation, :regression_evaluator, "rmse", false},
      {Evaluation, :regression_evaluator, "r2", true},
      {Evaluation, :multiclass_classification_evaluator, "logLoss", false},
      {Evaluation, :multiclass_classification_evaluator, "accuracy", true},
      {Evaluation, :clustering_evaluator, "silhouette", true}
    ] do
  check.(
    "pinned: #{name}, #{metric}",
    Internal.larger_better?(apply(module, name, [[metric_name: metric]])),
    expected
  )
end

# `param_grid` is a Cartesian product and nothing else — no session, no server — so all of it
# is checkable here. The ordering claim is the one that could quietly matter: the best index is
# the *first* maximum, so two param maps that tie are separated by nothing but their position.

lr_grid = Latu.ML.Classification.logistic_regression()

grid = Internal.param_grid([{lr_grid, [reg_param: [0.1, 0.01], max_iter: [10, 100]]}])

check.("a two-by-two grid has four entries", length(grid), 4)

check.(
  "the last param varies fastest, as itertools.product does it",
  Enum.map(grid, fn map -> Enum.map(map, fn {_uid, name, value} -> {name, value} end) end),
  [
    [reg_param: 0.1, max_iter: 10],
    [reg_param: 0.1, max_iter: 100],
    [reg_param: 0.01, max_iter: 10],
    [reg_param: 0.01, max_iter: 100]
  ]
)

check.(
  "every entry names the operator it belongs to",
  grid |> List.flatten() |> Enum.map(fn {uid, _name, _value} -> uid end) |> Enum.uniq(),
  [lr_grid.uid]
)

# Two operators, one product: this is what tuning a pipeline stage alongside a feature step
# looks like, and it is why a grid entry carries a uid at all.
assembler_grid = Latu.ML.Feature.vector_assembler()

both =
  Internal.param_grid([
    {assembler_grid, [handle_invalid: ["skip", "keep"]]},
    {lr_grid, [max_iter: [10, 100]]}
  ])

check.("a grid across two operators is still one product", length(both), 4)

check.(
  "and each entry carries both uids, in the order the pairs were given",
  both |> hd() |> Enum.map(fn {uid, _name, _value} -> uid end),
  [assembler_grid.uid, lr_grid.uid]
)

# A typo is refused now rather than after ten fits, with the operator's own params named.
check.(
  "an unknown param is refused against the operator's own table",
  try do
    Internal.param_grid([{lr_grid, [reg_parm: [0.1]]}])
  rescue
    error in ArgumentError -> error.message =~ "reg_param"
  end,
  true
)

check.(
  "and so is an empty grid, which would silently try nothing",
  try do
    Internal.param_grid([{lr_grid, [reg_param: []]}])
  rescue
    error in ArgumentError -> error.message =~ "is empty"
  end,
  true
)

IO.puts("")

count = :counters.get(failures, 1)

if count == 0 do
  IO.puts("offline check: everything the container can reach is green\n")
else
  IO.puts("offline check: #{count} failure(s)\n")
  System.halt(1)
end
