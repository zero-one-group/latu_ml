defmodule Latu.ML.CheatsheetTest do
  use ExUnit.Case, async: true

  # `docs/cheatsheet.cheatmd` is hand-written, and Latu's is generated — so where Latu's test
  # asserts the page has not drifted from its generator, this one has to be the generator's
  # replacement: it says the page and the modules still agree, in both directions.
  #
  # Why hand-written at all. Latu's cheatsheet covers ~500 functions and could not be typed;
  # this one covers **56** — 52 across the five wholly hand-written modules, plus the four
  # helper verbs stranded on `Latu.ML.Feature` — because everything else in this package is
  # generated and the registry is already its page (`Latu.ML.operators/1` and friends). A page
  # that size with a coverage check is cheaper than a generator, and reads better.
  #
  # `test/latu/examples_test.exs` resolves `Mod.fun/arity` spans across every extra, but its
  # regex needs one arity and a closing backtick — so `Latu.ML.attribute/2,3`, which is how a
  # cheatsheet writes a function with an optional argument, is invisible to it. That form is
  # exactly what this file parses.

  # The modules whose whole public surface has to be on the page. Every other module here is
  # generated in part or in whole — `Latu.ML.Feature` carries 57 constructors beside its two
  # helper verbs — so listing one would be asking the page to be the registry. Entries from a
  # generated module still have to *resolve*; they are just not counted for coverage.
  @covered [Latu.ML, Latu.ML.Stat, Latu.ML.Functions, Latu.ML.SparseVector, Latu.ML.Linalg]

  # Not surface: the struct and module callbacks the compiler adds.
  @callbacks [:__struct__, :__info__, :module_info]

  @page "docs/cheatsheet.cheatmd"

  # ``| `Latu.ML.attribute/2,3` | … **+ !** |`` — a qualified name whose arities are a list.
  @entry ~r/`(Latu(?:\.[A-Z][A-Za-z0-9_]*)*)\.([a-z_][A-Za-z0-9_]*[?!]?)\/([\d,]+)`/

  test "the page is there, and is a page" do
    assert File.exists?(@page)
    assert length(entries()) > 40
  end

  test "every function the cheatsheet names is real, at every arity it shows" do
    missing =
      for {module, name, arity} <- Enum.uniq(entries()),
          not exported?(module, name, arity),
          do: "#{inspect(module)}.#{name}/#{arity} — #{arities(module, name)}"

    assert missing == []
  end

  test "and every hand-written verb is on it" do
    listed = MapSet.new(entries())

    absent =
      for module <- @covered,
          {name, arity} <- exports(module),
          not MapSet.member?(listed, {module, name, arity}),
          do: "#{inspect(module)}.#{name}/#{arity}"

    assert absent == [],
           "not on #{@page}: " <> Enum.join(absent, ", ")
  end

  # A default argument makes one `def` several exports, so a row spells its arities out —
  # `save/2,3` — and every one of them has to be real and every one of them counts for
  # coverage. Writing only the highest would let a default disappear without anything noticing.
  defp entries do
    for line <- @page |> File.read!() |> String.split("\n"),
        [_all, module, name, arities] <- Regex.scan(@entry, line),
        arity <- String.split(arities, ","),
        entry <- twinned(line, Module.concat([module]), name, String.to_integer(arity)),
        do: entry
  end

  # A row marked `+ !` stands for the raising twin at the same arities, which is what the
  # page's own legend promises a reader — so the twin is checked without being written out.
  defp twinned(line, module, name, arity) do
    if String.contains?(line, "**+ !**") do
      [{module, String.to_atom(name), arity}, {module, String.to_atom(name <> "!"), arity}]
    else
      [{module, String.to_atom(name), arity}]
    end
  end

  defp exports(module) do
    Code.ensure_loaded!(module)

    for {name, arity} <- module.__info__(:functions),
        name not in @callbacks,
        not String.starts_with?(Atom.to_string(name), "_"),
        do: {name, arity}
  end

  defp exported?(module, name, arity) do
    Code.ensure_loaded?(module) and {name, arity} in module.__info__(:functions)
  end

  defp arities(module, name) do
    if Code.ensure_loaded?(module) do
      case for {^name, arity} <- module.__info__(:functions), do: arity do
        [] -> "no #{name} in #{inspect(module)}"
        found -> "#{name} takes #{Enum.join(found, " or ")}"
      end
    else
      "#{inspect(module)} is not a module"
    end
  end
end
