defmodule Latu.ML.TwinsTest do
  use ExUnit.Case, async: true

  # Latu's convention, and this package inherits it: an action that can return `{:error, _}` has
  # a `!` twin that raises. Read off the compiled specs rather than a list someone remembers to
  # update. Copied from Latu's `twins_test.exs`; keep them in step.
  # The generated accessor modules are deliberately not here: an accessor answers `{:ok, _}`
  # and has no `!` twin, which is a decision (`docs/decisions.md`) rather than an omission.
  @modules [Latu.ML, Latu.ML.Linalg, Latu.ML.Feature, Latu.ML.Stat]

  @exempt []

  test "every function whose spec can return an error has a ! twin" do
    missing =
      for module <- @modules,
          {{name, arity}, clauses} <- specs(module),
          not bang?(name),
          not hidden?(module, name),
          {module, name, arity} not in @exempt,
          Enum.any?(clauses, &returns_error?/1),
          not function_exported?(module, :"#{name}!", arity),
          do: "#{inspect(module)}.#{name}/#{arity}"

    assert missing == [], """
    These can fail and have no ! twin:

    #{Enum.map_join(missing, "\n", &"  #{&1}")}

    Either add the twin or, if raising makes no sense for it, add it to @exempt with the reason.
    """
  end

  test "and every ! twin has the function it is a twin of" do
    orphans =
      for module <- @modules,
          {{name, arity}, _clauses} <- specs(module),
          bang?(name),
          not function_exported?(module, unbang(name), arity),
          do: "#{inspect(module)}.#{name}/#{arity}"

    assert orphans == []
  end

  # `Code.ensure_loaded!/1` is load-bearing: `function_exported?/3` answers false for a module
  # that is merely compiled, and neither `fetch_specs/1` nor `fetch_docs/1` loads one.
  defp specs(module) do
    Code.ensure_loaded!(module)
    {:ok, specs} = Code.Typespec.fetch_specs(module)

    specs
  end

  defp bang?(name), do: String.ends_with?(Atom.to_string(name), "!")

  defp hidden?(module, name) do
    {:docs_v1, _, _, _, _, _, docs} = Code.fetch_docs(module)

    Enum.any?(docs, fn
      {{:function, ^name, _arity}, _anno, _sig, :hidden, _meta} -> true
      _entry -> false
    end)
  end

  defp unbang(name), do: name |> Atom.to_string() |> String.trim_trailing("!") |> String.to_atom()

  defp returns_error?({:type, _, :bounded_fun, [fun, _constraints]}), do: returns_error?(fun)
  defp returns_error?({:type, _, :fun, [_args, return]}), do: error_tuple?(return)
  defp returns_error?(_clause), do: false

  defp error_tuple?({:type, _, :union, members}), do: Enum.any?(members, &error_tuple?/1)
  defp error_tuple?({:type, _, :tuple, [{:atom, _, :error} | _]}), do: true
  defp error_tuple?(_type), do: false
end
