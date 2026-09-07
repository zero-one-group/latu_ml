defmodule Latu.ML.LayeringTest do
  use ExUnit.Case, async: true

  # Latu's layering test, narrowed to this package. The load-bearing rule is the last one: if
  # only the plan, linalg and decode layers know the protos exist, everything above them is
  # testable with no server and no wire — which is why every test but the vertical slice runs
  # offline.
  #
  # Two instruments, because neither is enough alone:
  #
  #   * BEAM import tables see **calls**, so a module named only in a docstring does not count.
  #     They are blind to struct literals and patterns, which compile away — and building a
  #     proto is almost always a struct literal, so this instrument cannot police the protos.
  #   * Source text sees **mentions**, which is exactly what a struct literal is. Coarser, and
  #     the only thing that catches `%Proto.MlOperator{}` appearing where it should not.

  @grpc ~r/^GRPC\b/
  @transport ~r/^Latu\.Client\b/
  @facade ~r/^Latu\.ML$/

  @proto_layers ~w(plan.ex linalg.ex result.ex)

  test "nothing here calls gRPC" do
    for module <- modules() do
      offenders = module |> remote_calls() |> Enum.filter(&(&1 =~ @grpc))

      assert offenders == [],
             "#{inspect(module)} calls #{inspect(offenders)}. Latu owns the transport; " <>
               "route it through Latu.Client."
    end
  end

  test "only the facade calls Latu's transport" do
    for module <- modules(), not (inspect(module) =~ @facade) do
      offenders = module |> remote_calls() |> Enum.filter(&(&1 =~ @transport))

      assert offenders == [],
             "#{inspect(module)} calls #{inspect(offenders)}. An action belongs on Latu.ML, " <>
               "or widen the layer."
    end
  end

  test "the plan layer never reaches the transport" do
    offenders =
      Latu.ML.Plan
      |> remote_calls()
      |> Enum.filter(&(&1 =~ @transport or &1 =~ @grpc))

    assert offenders == [],
           "Latu.ML.Plan calls #{inspect(offenders)}. Plan building is pure: take values in, " <>
             "hand a proto back, and let the layer above do the RPC."
  end

  # The reason `Latu.Plan.relation/1` was made public in Latu 0.2.0. A second wrapper out of
  # tree would be a second plan_id sequence.
  test "the plan layer takes its relations from Latu's one allocator" do
    assert Enum.any?(remote_calls(Latu.ML.Plan), &(&1 =~ ~r/^Latu\.Plan$/)),
           "Latu.ML.Plan does not call Latu.Plan — has it grown its own plan_id?"
  end

  test "only the plan, linalg and decode layers name the protos" do
    offenders =
      for {file, source} <- sources(),
          Path.basename(file) not in @proto_layers,
          source =~ ~r/Latu\.Protocol\b/,
          do: file

    assert offenders == [],
           "#{inspect(offenders)} name Latu.Protocol. Keep the protos below the API layer, " <>
             "or add the file to @proto_layers with a reason."
  end

  defp modules, do: Application.spec(:latu_ml, :modules)

  defp sources do
    "lib/**/*.ex" |> Path.wildcard() |> Map.new(&{&1, File.read!(&1)})
  end

  defp remote_calls(module) do
    case :beam_lib.chunks(:code.which(module), [:imports]) do
      {:ok, {_module, [imports: imports]}} ->
        imports |> Enum.map(&(&1 |> elem(0) |> inspect())) |> Enum.uniq()

      _other ->
        flunk("cannot read imports for #{inspect(module)}")
    end
  end
end
