# Stand-ins for the generated protos, for `dev/check_offline.exs`.
#
# `:protobuf` needs a toolchain the container does not have, so the plan layer is compiled
# against structs with the right fields and no behaviour. Nothing here is used by the library
# or by `mix test`.

defmodule Latu.Protocol.Spark.Connect do
  @moduledoc false
end

# Not protos: Latu's own two, and what `Latu.ML.Result` needs to compile here at all. `UDT` is
# the struct it pattern-matches a Vector literal on; `Error` is what it builds when the server
# answers in a shape the protocol does not allow.
defmodule Latu.Result.UDT do
  @moduledoc false
  defstruct [:class, elements: []]
end

defmodule Latu.Error do
  @moduledoc false
  defexception [:kind, :message, :error_class]
end

# A lazy frame, for the one argument arm that is a relation rather than a literal: a helper
# call and an argument-taking attribute both put one on `Fetch.Method.Args`'s `input`.
defmodule Latu.DataFrame do
  @moduledoc false
  defstruct [:session, :plan]
end

for {mod, fields} <- [
      {MlOperator, [:name, :uid, :type]},
      {MlParams, [params: %{}]},
      {MlCommand, [:command]},
      {MlCommandResult, [:result_type]},
      {Command, [:command_type]},
      {Plan, [:op_type]},
      {Relation, [:rel_type, :common]},
      {RelationCommon, [:plan_id]},
      {MlRelation, [:ml_type, :model_summary_dataset]},
      {Fetch, [:obj_ref, methods: []]},
      {ObjectRef, [:id]},
      {DataType, [:kind]},
      {Ints, [values: []]},
      {Doubles, [values: []]}
    ] do
  name = Module.concat(Latu.Protocol.Spark.Connect, mod)
  Module.create(
    name,
    quote(do: defstruct(unquote(Macro.escape(fields)))),
    Macro.Env.location(__ENV__)
  )
end

for {parent, mod, fields} <- [
      {MlCommand, Fit, [:estimator, :params, :dataset]},
      {MlCommand, Evaluate, [:evaluator, :params, :dataset]},
      {MlCommand, CreateSummary, [:model_ref, :dataset]},
      {MlCommand, Delete, [obj_refs: []]},
      {MlCommand, Write, [:type, :params, :path, :should_overwrite, options: %{}]},
      {MlCommand, Read, [:operator, :path]},
      {MlCommand, GetModelSize, [:model_ref]},
      {MlCommand, GetCacheInfo, []},
      {MlCommand, CleanCache, []},
      {MlRelation, Transform, [:operator, :input, :params]},
      {MlCommandResult, MlOperatorInfo, [:type, :uid, :params, :warning_message]},
      {Fetch, Method, [:method, args: []]},
      {DataType, Array, [:element_type, :contains_null]},
      {DataType, String, [collation: "", type_variation_reference: 0]},
      {DataType, Boolean, []},
      {DataType, Integer, []},
      {DataType, Long, []},
      {DataType, Double, []},
      {DataType, UDT, [:type, :jvm_class]},
      {DataType, NULL, []},
      {Expression, Literal, [:data_type, :literal_type]}
    ] do
  name = Module.concat([Latu.Protocol.Spark.Connect, parent, mod])
  Module.create(
    name,
    quote(do: defstruct(unquote(Macro.escape(fields)))),
    Macro.Env.location(__ENV__)
  )
end

Module.create(
  Module.concat([Latu.Protocol.Spark.Connect, Fetch, Method, Args]),
  quote(do: defstruct([:args_type])),
  Macro.Env.location(__ENV__)
)

for {mod, fields} <- [
      {Array, [:element_type, elements: []]},
      {Struct, [:struct_type, elements: []]},
      {SpecializedArray, [:value_type]}
    ] do
  name = Module.concat([Latu.Protocol.Spark.Connect, Expression, Literal, mod])
  Module.create(
    name,
    quote(do: defstruct(unquote(Macro.escape(fields)))),
    Macro.Env.location(__ENV__)
  )
end

defmodule Latu.Plan do
  @moduledoc false
  alias Latu.Protocol.Spark.Connect, as: Proto

  def new(%Proto.Relation{} = relation), do: %Proto.Plan{op_type: {:root, relation}}
  def new(%Proto.Command{} = command), do: %Proto.Plan{op_type: {:command, command}}

  def relation(rel_type) do
    %Proto.Relation{rel_type: rel_type, common: %Proto.RelationCommon{plan_id: plan_id()}}
  end

  def plan_id, do: System.unique_integer([:positive, :monotonic])
end

defmodule Latu.ML.Linalg do
  @moduledoc false
  alias Latu.Protocol.Spark.Connect, as: Proto

  def to_literal(_value) do
    %Proto.Expression.Literal{literal_type: {:struct, :stand_in}}
  end
end
