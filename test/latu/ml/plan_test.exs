defmodule Latu.ML.PlanTest do
  use ExUnit.Case, async: true

  alias Latu.ML.Plan
  alias Latu.Protocol.Spark.Connect, as: Proto

  defp input, do: Latu.Plan.range(0, 4, 1)

  defp ml_command(%Proto.Plan{op_type: {:command, command}}) do
    {:ml_command, %Proto.MlCommand{command: arm}} = command.command_type

    arm
  end

  defp ml_relation(%Proto.Relation{rel_type: {:ml_relation, relation}}), do: relation

  describe "literal/2 coerces to the declared kind, then to a wire type by magnitude" do
    test "an int param stays an integer while it fits 32 bits" do
      assert Plan.literal(10, :int).literal_type == {:integer, 10}
      assert Plan.literal(3_000_000_000, :int).literal_type == {:long, 3_000_000_000}
    end

    # PySpark's `_from_value` picks by magnitude whatever the param declared, so :int and :long
    # are one path. The declared type still decides the *kind*.
    test "a long param is the same path" do
      assert Plan.literal(10, :long).literal_type == {:integer, 10}
    end

    test "a double param takes an integer and sends a double" do
      assert Plan.literal(1, :double).literal_type == {:double, 1.0}
      assert Plan.literal(0.01, :double).literal_type == {:double, 0.01}
    end

    test "a float param is a double on the wire, because Elixir has no 32-bit float" do
      assert Plan.literal(1.5, :float).literal_type == {:double, 1.5}
    end

    test "an int param takes a float with nothing after the point" do
      assert Plan.literal(4.0, :int).literal_type == {:integer, 4}
    end

    test "and refuses one with something after it, naming the type" do
      assert_raise ArgumentError, "4.5 is not a valid :int value", fn ->
        Plan.literal(4.5, :int)
      end
    end

    test "a string param takes an atom, because columns are atoms here" do
      assert Plan.literal(:features, :string).literal_type == {:string, "features"}
      assert Plan.literal("features", :string).literal_type == {:string, "features"}
    end

    test "a boolean param takes only a boolean" do
      assert Plan.literal(true, :boolean).literal_type == {:boolean, true}

      assert_raise ArgumentError, ~r/not a valid :boolean value/, fn ->
        Plan.literal(1, :boolean)
      end
    end

    # PySpark writes the deprecated `element_type` and leaves the outer `data_type` unset. The
    # golden in wire_test.exs is what settled this; keep both in step.
    test "a list param names its element type once, on the array" do
      literal = Plan.literal([:x1, "x2"], {:list, :string})

      assert literal.data_type == nil
      assert {:array, array} = literal.literal_type
      assert {:string, %Proto.DataType.String{collation: "UTF8_BINARY"}} = array.element_type.kind
      assert Enum.map(array.elements, & &1.literal_type) == [{:string, "x1"}, {:string, "x2"}]
    end

    test "a list param refuses a bare value" do
      assert_raise ArgumentError, ~r/takes a list/, fn -> Plan.literal(:x1, {:list, :string}) end
    end

    test "a vector param goes through Linalg" do
      literal = Plan.literal(Nx.tensor([1.0], type: :f64), :vector)

      assert {:struct, _} = literal.literal_type
    end
  end

  describe "params/1" do
    test "keys by the wire name, and carries only what was given" do
      params = Plan.params([{"maxIter", :int, 10}, {"regParam", :double, 0.01}])

      assert Map.keys(params.params) |> Enum.sort() == ["maxIter", "regParam"]
      assert params.params["maxIter"].literal_type == {:integer, 10}
    end

    test "an empty list is an empty map, not a nil" do
      assert Plan.params([]).params == %{}
    end
  end

  describe "operator/3" do
    test "names the class and the kind the server expects" do
      operator = Plan.operator("org.example.Thing", "Thing_abc", :estimator)

      assert operator.name == "org.example.Thing"
      assert operator.uid == "Thing_abc"
      assert operator.type == :OPERATOR_TYPE_ESTIMATOR
    end

    test "an unknown kind lists the ones there are" do
      assert_raise ArgumentError, ~r/expected one of \[:estimator, :transformer/, fn ->
        Plan.operator("org.example.Thing", "uid", :regressor)
      end
    end
  end

  describe "commands" do
    test "fit carries the estimator, the params and the dataset" do
      operator = Plan.operator("org.example.LR", "LR_1", :estimator)
      params = Plan.params([{"maxIter", :int, 1}])
      input = input()

      assert {:fit, fit} = ml_command(Plan.fit(operator, params, input))
      assert fit.estimator == operator
      assert fit.params == params
      assert fit.dataset == input
    end

    test "fetch as a command carries the ref and the method chain" do
      assert {:fetch, fetch} =
               ml_command(Plan.fetch_command("m-1", [Plan.method(:summary), Plan.method("roc")]))

      assert fetch.obj_ref.id == "m-1"
      assert Enum.map(fetch.methods, & &1.method) == ["summary", "roc"]
    end

    test "delete names every ref" do
      assert {:delete, delete} = ml_command(Plan.delete(["m-1", "m-2"]))
      assert Enum.map(delete.obj_refs, & &1.id) == ["m-1", "m-2"]
    end
  end

  describe "relations" do
    test "transform by a model reference" do
      input = input()
      relation = ml_relation(Plan.transform("m-1", input, Plan.params([])))

      assert {:transform, transform} = relation.ml_type
      assert {:obj_ref, ref} = transform.operator
      assert ref.id == "m-1"
      assert transform.input == input
    end

    test "transform by an unfitted transformer" do
      operator = Plan.operator("org.example.VA", "VA_1", :transformer)
      relation = ml_relation(Plan.transform(operator, input(), Plan.params([])))

      assert {:transform, transform} = relation.ml_type
      assert transform.operator == {:transformer, operator}
    end

    # The whole reason `Latu.Plan.relation/1` was made public: one allocator, one sequence.
    test "every relation gets its own plan_id from Latu's allocator" do
      first = Plan.transform("m-1", input(), Plan.params([]))
      second = Plan.transform("m-1", input(), Plan.params([]))

      assert first.common.plan_id != second.common.plan_id
      assert is_integer(first.common.plan_id)
    end

    test "fetch as a relation is the same Fetch, differently wrapped" do
      methods = [Plan.method(:summary), Plan.method(:roc)]

      {:fetch, from_command} = ml_command(Plan.fetch_command("m-1", methods))
      {:fetch, from_relation} = ml_relation(Plan.fetch_relation("m-1", methods)).ml_type

      assert from_command == from_relation
    end

    test "a summary relation carries the training frame beside the fetch, not inside it" do
      input = input()
      relation = ml_relation(Plan.fetch_relation("m-1", [Plan.method(:roc)], input))

      assert relation.model_summary_dataset == input
      assert {:fetch, _} = relation.ml_type
    end
  end

  describe "method/2" do
    test "arguments are literals or relations, and nothing else" do
      literal = Plan.literal(1, :int)
      input = input()
      method = Plan.method(:predict, [literal, input])

      assert Enum.map(method.args, & &1.args_type) == [{:param, literal}, {:input, input}]

      assert_raise ArgumentError, ~r/literal or a relation/, fn ->
        Plan.method(:predict, [:nope])
      end
    end
  end
end
