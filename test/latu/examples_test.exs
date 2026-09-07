defmodule Latu.ML.ExamplesTest do
  use ExUnit.Case, async: true

  # Latu's `test/latu/examples_test.exs`, for this package. Every code example in the docs,
  # checked against the real API without a server.
  #
  # Three tiers verify the examples here as they do in Latu: doctests run the pure ones, the
  # guides run under `test/integration/guides_test.exs`, and this is the third. It cannot run an
  # example, but it proves every function an example names exists at the arity it is called with
  # — the whole class of rot a doc acquires by sitting still while the API moves.
  #
  # Reads source rather than `Code.fetch_docs/1`, as Latu's does, so the two files stay the same
  # shape. What that misses here is the **generated** docs: `Latu.ML.Generate` builds a
  # `@moduledoc` and a `@doc` per constructor and accessor at compile time from
  # `priv/ml_operators.exs` and `priv/ml_attributes.exs`, and none of it is a literal in
  # `lib/`. Those docs are Spark's own prose, with no Elixir in them, and `dev/check_offline.exs`
  # is what checks the registry they come from.

  # What has to be kept in sync by hand, and what keeps it honest:
  #
  #   * `@extras` — the prose that is checked. A new page joins neither this nor the guides by
  #     itself, so "every markdown doc is accounted for" below fails until someone decides.
  #   * `@excused` — the pages deliberately not checked, each with the reason.
  #   * `@not_elixir` — the first-line shapes that are output rather than code. A new one shows
  #     up as a parse failure naming the block, which is the prompt to decide, not a silent miss.
  #   * the corpus floor — raise it as the docs grow.
  #
  # There is deliberately **no alias map**. Latu keeps one because it has two aliases; the docs
  # here alias a dozen generated modules, so `module_for/1` resolves an alias against the app's
  # own module list instead (see "Resolution").

  @extras [
    "README.md",
    "CHANGELOG.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    "docs/cheatsheet.cheatmd",
    "usage-rules.md",
    "docs/deviations.md"
  ]

  @excused %{
    "docs/decisions.md" =>
      "a dated maintainer's log: its snippets record what the API was when a decision was made",
    "dev/README.md" =>
      "contributor notes: its code is shell, Python and the interop dance, not Latu examples",
    "assets/README.md" => "the identity assets: palette, fonts and placement rules"
  }

  # A block that is not Elixir, by its first line. Everything else must parse — a block nobody
  # can parse is reported, never skipped, or the opt-out is where a broken example hides.
  @not_elixir ["$ ", "+-", "|", "> ", "root", "sc://", "mix ", "docker ", "#=>", "** ("]

  # The corpus is `@extras` plus `lib/`, and a page joins it by being named. So a doc nobody
  # checks is exactly how this file rots — quietly, while every assertion above stays green.
  test "every markdown doc is checked here, run as a guide, or excused by name" do
    accounted = @extras ++ Path.wildcard("docs/guides/*.md") ++ Map.keys(@excused)

    assert markdown() -- accounted == [],
           "neither checked, nor a guide, nor excused: " <> inspect(markdown() -- accounted)

    assert Map.keys(@excused) -- markdown() == [],
           "@excused names a file that is gone: " <> inspect(Map.keys(@excused) -- markdown())
  end

  # A floor, not a target. Every check below passes in silence on an empty corpus, so if the
  # extractor breaks — a markdown shape it stops recognising, a wildcard matching nothing —
  # this is what notices. Raise it when the docs grow; never lower it to make a run pass.
  #
  # Measured 2026-09-07: 78 Elixir blocks, 11 doctest blocks, 181 references.
  test "the corpus is still there" do
    kinds = Enum.frequencies_by(blocks(), fn {_source, _line, block} -> kind(block) end)

    assert kinds[:elixir] >= 105, "Elixir blocks fell to #{kinds[:elixir]}"
    assert kinds[:doctest] >= 10, "doctest blocks fell to #{kinds[:doctest]}"
    assert length(references()) >= 240, "references fell to #{length(references())}"
  end

  test "every code block in the docs parses as Elixir" do
    problems =
      for {source, line, block} <- blocks(),
          kind(block) == :elixir,
          {:error, reason} <- [Code.string_to_quoted(joined(block))] do
        "#{source}:#{line}: does not parse (#{trim(reason)}) — starts: #{hd(block)}"
      end

    assert problems == []
  end

  test "every qualified call in an example exists with that arity" do
    problems =
      for {source, line, block} <- blocks(),
          kind(block) == :elixir,
          {:ok, ast} <- [Code.string_to_quoted(joined(block))],
          {module, name, arity, offset} <- calls(ast),
          not exported?(module, name, arity) do
        "#{source}:#{line + offset}: #{inspect(module)}.#{name}/#{arity}" <>
          " — #{arities(module, name)}"
      end

    assert problems == []
  end

  # **Inline code in prose gets neither of the checks above.** `blocks/0` reads fenced and
  # indented blocks and `references/0` reads `Mod.fun/arity` mentions, so a
  # `Latu.ML.attribute(model, :coefficients)` written in a sentence or a table cell is invisible
  # to both — and `docs/deviations.md` is a table of exactly those.
  test "every qualified call written inline in prose exists with that arity" do
    problems =
      for {source, line, code} <- inline_calls(),
          {:ok, ast} <- [Code.string_to_quoted(code)],
          {module, name, arity, _offset} <- calls(ast),
          not exported?(module, name, arity) do
        "#{source}:#{line}: `#{inspect(module)}.#{name}/#{arity}`" <>
          " — #{arities(module, name)}"
      end

    assert problems == []
  end

  test "every `Module.fun/arity` reference in the docs resolves" do
    problems =
      for {source, line, module, name, arity} <- references(),
          not exported?(module, name, arity) do
        "#{source}:#{line}: `#{inspect(module)}.#{name}/#{arity}` — #{arities(module, name)}"
      end

    assert problems == []
  end

  # A doctest whose expected value is a struct is asserting on `Inspect`, which Latu's
  # `CLAUDE.md` forbids: it breaks on a harmless change to a renderer. This package has its own
  # `Latu.ML.Inspect`, so the temptation is live.
  test "a doctest asserts data, not an inspected struct" do
    problems =
      for {source, line, block} <- blocks(),
          kind(block) == :doctest,
          {expected, offset} <- expectations(block),
          String.starts_with?(expected, ["#Latu", "%Latu"]) do
        "#{source}:#{line + offset}: expected value renders a struct — #{expected}"
      end

    assert problems == []
  end

  test "a doctest line has examples to run, and examples have a doctest line to run them" do
    declared = declared_doctests()
    documented = documented_doctests()

    assert declared -- documented == [],
           "these `doctest` lines run nothing — no `iex>` in the module's own docs: " <>
             inspect(declared -- documented)

    assert documented -- declared == [],
           "these modules have `iex>` examples no `doctest` line runs: " <>
             inspect(documented -- declared)
  end

  # =============================================
  # Sources
  # =============================================

  defp lib_sources, do: "lib/**/*.ex" |> Path.wildcard() |> Enum.sort()

  # Every markdown doc the repo *owns*. Walked rather than listed, so a page in a directory
  # nobody thought of still has to be accounted for. What is not ours is what `.gitignore`
  # already says is not ours: `_build/` and `deps/` are Mix's, `doc/` appears the moment anyone
  # runs `mix docs`, and `tmp/` is scratch — an unzipped asset drop and the working notes of a
  # pruning pass have both landed there. A **denylist** on purpose: an allowlist of the places
  # docs live would silently stop checking a directory nobody added to it, which is the rot this
  # file exists to catch.
  @not_ours ["deps/", "_build/", "doc/", "tmp/"]

  defp markdown do
    "**/*.md"
    |> Path.wildcard()
    |> Enum.reject(&String.starts_with?(&1, @not_ours))
    |> Enum.sort()
  end

  # `{line, text}` for every `@doc`/`@moduledoc` carrying a string. An interpolated heredoc
  # keeps its literal segments; a `@doc unquote(built)` in `Latu.ML.Generate` has none, so it
  # contributes nothing rather than something wrong.
  defp docstrings(path) do
    {:ok, ast} = path |> File.read!() |> Code.string_to_quoted()
    {_ast, found} = Macro.prewalk(ast, [], &docstring/2)

    Enum.reverse(found)
  end

  defp docstring({:@, _meta, [{tag, meta, [doc]}]} = node, acc)
       when tag in [:doc, :moduledoc] do
    case doc_text(doc) do
      nil -> {node, acc}
      text -> {node, [{meta[:line], text} | acc]}
    end
  end

  defp docstring(node, acc), do: {node, acc}

  defp doc_text(doc) when is_binary(doc), do: doc

  defp doc_text({:<<>>, _meta, parts}) do
    parts |> Enum.filter(&is_binary/1) |> Enum.join(" ... ")
  end

  defp doc_text(_other), do: nil

  # `{source, line, lines}` for every code block in every doc. Lines are 1-based and
  # approximate for a docstring: the `@doc` line plus the offset inside the heredoc.
  defp blocks do
    from_lib =
      for path <- lib_sources(),
          {line, text} <- docstrings(path),
          {offset, block} <- code_blocks(text),
          do: {path, line + offset + 1, block}

    # **Guides included, even though the integration tier runs them.** Running a fence is
    # strictly stronger than parsing it, so this is redundant for an executed one — and it is
    # the only check an illustrative fence gets at all. Rather than teach this file which
    # fences `guides_test.exs` skips and risk the two definitions drifting apart, it checks
    # every guide fence and lets the redundancy pay for the guarantee: no fence in
    # `docs/guides/` is unchecked by both.
    from_prose =
      for path <- @extras ++ Path.wildcard("docs/guides/*.md"),
          {offset, block} <- code_blocks(File.read!(path)),
          do: {path, offset + 1, block}

    from_lib ++ from_prose
  end

  # =============================================
  # Markdown
  # =============================================

  defp code_blocks(text) do
    text |> String.split("\n") |> Enum.with_index() |> gather([], true)
  end

  defp gather([], acc, _after_blank), do: Enum.reverse(acc)

  defp gather([{line, index} | rest], acc, after_blank) do
    cond do
      fence?(line) ->
        {body, tail} = Enum.split_while(rest, fn {l, _i} -> not fence?(l) end)
        block = if elixir_fence?(line), do: Enum.map(body, &elem(&1, 0)), else: []
        gather(Enum.drop(tail, 1), keep(block, index + 1, acc), true)

      # Markdown needs a blank line before an indented code block; requiring one keeps an
      # indented continuation of a bullet out of the corpus. Most examples in this package's
      # docstrings are indented rather than fenced, so this branch carries the bulk of them.
      #
      # A blank line inside such a block does not end it, in markdown — but it does separate
      # one example from the next, and ExUnit reads two `iex>` runs either side of a blank as
      # two doctests. Splitting there is what stops a docstring whose first example is a
      # doctest from hiding every example after it.
      after_blank and indented?(line) ->
        {body, tail} =
          Enum.split_while([{line, index} | rest], fn {l, _i} -> indented?(l) or blank?(l) end)

        gather(tail, segments(body) ++ acc, true)

      true ->
        gather(rest, acc, blank?(line))
    end
  end

  defp keep([], _index, acc), do: acc
  defp keep(block, index, acc), do: [{index, dedent(block)} | acc]

  # One indexed block in, its blank-separated runs out, each keeping its own line number.
  defp segments(body) do
    body
    |> Enum.chunk_by(fn {line, _index} -> blank?(line) end)
    |> Enum.reject(fn [{line, _index} | _rest] -> blank?(line) end)
    |> Enum.reduce([], fn [{_line, index} | _rest] = run, acc ->
      keep(Enum.map(run, &elem(&1, 0)), index, acc)
    end)
  end

  defp fence?(line), do: String.starts_with?(String.trim_leading(line), "```")

  # A fence declares its own language, so trust it: a ```sh or ```text fence is illustration.
  # An untagged one is code, which is what makes a missing tag show up here rather than pass.
  defp elixir_fence?(line) do
    tag = line |> String.trim() |> String.trim_leading("`")

    tag in ["", "elixir"]
  end

  defp indented?(line), do: Regex.match?(~r/^ {4,}\S/, line)
  defp blank?(line), do: String.trim(line) == ""

  defp dedent(lines) do
    margin =
      lines
      |> Enum.reject(&blank?/1)
      |> Enum.map(fn line -> String.length(line) - String.length(String.trim_leading(line)) end)
      |> Enum.min(fn -> 0 end)

    Enum.map(lines, fn line -> String.slice(line, margin..-1//1) || "" end)
  end

  defp joined(block), do: Enum.join(block, "\n")

  defp kind([first | _rest] = block) do
    cond do
      String.starts_with?(first, "iex>") -> :doctest
      Enum.any?(@not_elixir, &String.starts_with?(first, &1)) -> :prose
      Enum.all?(block, &blank?/1) -> :prose
      true -> :elixir
    end
  end

  defp expectations(block) do
    block
    |> Enum.with_index()
    |> Enum.reject(fn {line, _i} ->
      blank?(line) or String.starts_with?(line, "iex>") or String.starts_with?(line, "...>")
    end)
    |> Enum.map(fn {line, i} -> {String.trim(line), i} end)
  end

  # =============================================
  # Resolution
  # =============================================

  defp calls(ast) do
    {_ast, found} = ast |> depipe() |> uncapture() |> Macro.prewalk([], &collect/2)

    Enum.reverse(found)
  end

  # `Code.string_to_quoted/1` does not expand pipes, so `x |> Latu.limit(2)` leaves the call
  # node holding ONE argument. Rewriting `|>` first is what makes the arity real.
  defp depipe(ast) do
    Macro.prewalk(ast, fn
      {:|>, _meta, [left, {call, meta, args}]} when is_list(args) -> {call, meta, [left | args]}
      node -> node
    end)
  end

  # `&Latu.ML.delete!/1` is a call node with no arguments and the arity beside it. Rewriting it
  # into a call of that arity is what keeps a capture in the corpus rather than reading as `/0`.
  defp uncapture(ast) do
    Macro.prewalk(ast, fn
      {:&, _meta, [{:/, _s, [{{:., _d, _t} = target, meta, []}, arity]}]}
      when is_integer(arity) ->
        {target, meta, List.duplicate(nil, arity)}

      node ->
        node
    end)
  end

  # `alias Latu.ML.{Classification, Feature}` is a call node too — name `:{}` on `Latu.ML`,
  # with the aliases as arguments. It is an alias, not a call, and every example here opens with
  # one. Latu's copy of this file has no such clause because Latu's docs alias one module at a
  # time.
  defp collect({{:., _dot, [{:__aliases__, _a, _parts}, :{}]}, _meta, _args} = node, acc) do
    {node, acc}
  end

  defp collect({{:., _dot, [{:__aliases__, _a, parts}, name]}, meta, args} = node, acc)
       when is_atom(name) and is_list(args) do
    case module_for(parts) do
      nil -> {node, acc}
      module -> {node, [{module, name, length(args), meta[:line] - 1} | acc]}
    end
  end

  defp collect(node, acc), do: {node, acc}

  # An example writes `ML.fit(...)` and `Classification.logistic_regression(...)`, not the full
  # name, and the alias that made it so is at the top of the fence rather than in this file. So:
  # a `Latu.`-headed name is taken as written, and anything else is resolved by **unique suffix**
  # against the app's own modules — `[:ML]` is `Latu.ML`, `[:Feature]` is `Latu.ML.Feature`,
  # `[:BinaryLogisticRegressionSummary]` is the one generated module with that last segment.
  #
  # Two properties worth stating. A name that matches nothing — `Nx`, `Explorer.DataFrame`, or a
  # `defmodule Linear` a guide defines for itself — resolves to nil and is skipped, exactly as a
  # non-Latu module is in Latu's own copy. And an **ambiguous** suffix is reported by the caller
  # rather than skipped, because silently picking one of two would be the wrong kind of green.
  defp module_for([head | _rest] = parts) do
    if head == :Latu do
      Module.concat(parts)
    else
      case suffix_matches(parts) do
        [module] -> module
        [] -> nil
        many -> raise "#{Enum.join(parts, ".")} is ambiguous: #{inspect(many)}"
      end
    end
  end

  defp suffix_matches(parts) do
    wanted = Enum.map(parts, &Atom.to_string/1)

    for module <- Application.spec(:latu_ml, :modules),
        split = Module.split(module),
        length(split) >= length(wanted),
        Enum.take(split, -length(wanted)) == wanted,
        do: module
  end

  defp exported?(module, name, arity) do
    Code.ensure_loaded?(module) and
      ({name, arity} in module.__info__(:functions) or
         {name, arity} in module.__info__(:macros))
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

  # =============================================
  # References
  # =============================================

  @qualified ~r/`([A-Z][A-Za-z0-9_.]*)\.([a-z_][A-Za-z0-9_]*[?!]?)\/(\d+)`/
  @bare ~r/`([a-z_][A-Za-z0-9_]*[?!]?)\/(\d+)`/

  # A bare `fun/2` in a module's own docs is what ExDoc autolinks against that module, falling
  # back to `Kernel`. In an extra or a guide it autolinks against nothing, so only the
  # qualified form is checked there — and a guide's *code* is checked by running it instead.
  #
  # **`@moduledoc false` modules are checked too, and that is not an oversight.** ExDoc renders
  # nothing for them, so a bare span there autolinks nowhere and cannot break a page — but it
  # still tells the next reader that a function exists at that arity. Every one this found on
  # its first run was a cross-module reference written bare: `Latu.ML.load/4` and
  # `Latu.ML.fit/2` inside `Latu.ML.Internal`, `Latu.first/2` inside `Latu.ML.Layout`. Qualify
  # them, rather than narrowing this to the rendered modules.
  defp references do
    from_lib =
      for path <- lib_sources(),
          {doc_line, text} <- docstrings(path),
          {line, offset} <- text |> String.split("\n") |> Enum.with_index(),
          {module, name, arity} <- qualified(line) ++ bare(line, own_module(path)),
          do: {path, doc_line + offset + 1, module, name, arity}

    from_prose =
      for path <- @extras ++ Path.wildcard("docs/guides/*.md"),
          {line, offset} <- path |> File.read!() |> String.split("\n") |> Enum.with_index(1),
          {module, name, arity} <- qualified(line),
          do: {path, offset, module, name, arity}

    from_lib ++ from_prose
  end

  # A backticked span naming a `Latu.` call: the `(` is what tells a call from a reference,
  # since `Latu.ML.fit/2` parses as a division and would otherwise read as a zero-arity call —
  # the references test above is what checks those. A span that does not parse is skipped rather
  # than reported: prose is full of code-ish fragments, and `blocks/0` is where a parse failure
  # is a defect.
  defp inline_calls do
    for path <- @extras ++ Path.wildcard("docs/guides/*.md"),
        {line, number} <- path |> File.read!() |> String.split("\n") |> Enum.with_index(1),
        [_span, code] <- Regex.scan(~r/`(Latu\.[^`]+)`/, line),
        String.contains?(code, "("),
        do: {path, number, code}
  end

  defp qualified(line) do
    for [_all, module, name, arity] <- Regex.scan(@qualified, line),
        module = Module.concat([module]),
        match?(["Latu" | _rest], Module.split(module)),
        "Protocol" not in Module.split(module),
        do: {module, String.to_atom(name), String.to_integer(arity)}
  end

  defp bare(line, module) do
    for [_all, name, arity] <- Regex.scan(@bare, line),
        name = String.to_atom(name),
        arity = String.to_integer(arity),
        not exported?(Kernel, name, arity),
        do: {module, name, arity}
  end

  # `lib/latu/ml/plan.ex` is `Latu.ML.Plan`, and `Macro.camelize` would say `Ml`. Two
  # abbreviations stay abbreviations here, `ML` and `FPM` (`docs/decisions.md`, 2026-09-06).
  defp own_module(path) do
    path
    |> Path.rootname()
    |> Path.split()
    |> Enum.drop(2)
    |> Enum.map(fn
      "ml" -> "ML"
      "fpm" -> "FPM"
      segment -> Macro.camelize(segment)
    end)
    |> then(&Module.concat(["Latu" | &1]))
  end

  # =============================================
  # Doctest bookkeeping
  # =============================================

  defp declared_doctests do
    for path <- Path.wildcard("test/**/*.exs"),
        [_line, module] <- Regex.scan(~r/^\s*doctest\s+([A-Z][\w.]*)/m, File.read!(path)),
        do: Module.concat([module])
  end

  defp documented_doctests do
    for path <- lib_sources(),
        {_line, text} <- docstrings(path),
        String.contains?(text, "iex>"),
        uniq: true,
        do: own_module(path)
  end

  defp trim(reason), do: reason |> inspect() |> String.slice(0, 120)
end
