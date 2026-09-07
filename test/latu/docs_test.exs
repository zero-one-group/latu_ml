defmodule Latu.ML.DocsTest do
  use ExUnit.Case, async: true

  # Latu's `test/latu/docs_test.exs`, for this package. Documentation as a test, not a habit —
  # and it earns its keep more here than there, because **most of this surface is generated**:
  # 68 constructors and 64 accessor modules come out of `Latu.ML.Generate`, so a gap is never
  # one forgotten function. It is a shape, and a shape repeats sixty-two times.
  #
  # Three of the four checks below read `Mix.Project.config()`, so this file is `mix`-only —
  # `dev/check_offline.exs` cannot run it.

  test "every public function in a documented module has a doc" do
    undocumented =
      Enum.flat_map(modules(), fn module ->
        case Code.fetch_docs(module) do
          {:docs_v1, _, _, _, moduledoc, _, docs} when moduledoc not in [:hidden, :none] ->
            for {{:function, name, arity}, _line, _sig, :none, _meta} <- docs,
                do: "#{inspect(module)}.#{name}/#{arity}"

          _other ->
            []
        end
      end)

    assert undocumented == []
  end

  test "and every module says whether it is public, rather than leaving it open" do
    silent =
      for module <- modules(),
          match?({:docs_v1, _, _, _, :none, _, _}, Code.fetch_docs(module)),
          do: inspect(module)

    assert silent == []
  end

  # `:groups_for_modules` is hand-maintained, so a new public module would land in ExDoc's
  # default group and nobody would see it.
  #
  # **One entry is a regex**, and deliberately: the 64 generated model and summary modules are
  # matched by shape rather than listed, because a list would be a second place to keep in step
  # with `priv/ml_attributes.exs`. So "is it placed" is membership *or* a match — and a regex
  # that matches nothing is checked too, since a pattern that has stopped matching is exactly as
  # invisible as a missing entry.
  test "every documented module has a place in the docs sidebar, and every place is real" do
    entries =
      Mix.Project.config()
      |> Keyword.fetch!(:docs)
      |> Keyword.fetch!(:groups_for_modules)
      |> Enum.flat_map(fn {_group, matchers} -> matchers end)

    {patterns, listed} = Enum.split_with(entries, &is_struct(&1, Regex))

    documented = Enum.sort(documented_modules())

    unplaced =
      for module <- documented,
          module not in listed,
          not Enum.any?(patterns, &Regex.match?(&1, inspect(module))),
          do: module

    assert unplaced == [], "not in any `groups_for_modules`: " <> inspect(unplaced)

    assert listed -- documented == [],
           "grouped but not a documented module: " <> inspect(listed -- documented)

    for pattern <- patterns do
      assert Enum.any?(documented, &Regex.match?(pattern, inspect(&1))),
             "`groups_for_modules` has #{inspect(pattern)}, which matches no documented module"
    end
  end

  # `:extras` is hand-maintained too, and a new guide is otherwise invisible three ways at once:
  # `test/integration/guides_test.exs` runs its fences, `Latu.ML.ExamplesTest` counts it as a
  # guide, and `mix docs` renders no page for it. A check with a hand-maintained input list
  # needs a check on the list.
  #
  # Prose extras are named individually because their order is the sidebar's. Guides are a
  # directory, and `:groups_for_extras` groups them with a **regex**, so the wildcard is the
  # list: any file in `docs/guides/` has to appear in `:extras` and be matched by that regex,
  # and the regex must not reach anything else.
  @prose_extras [
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "docs/cheatsheet.cheatmd",
    "usage-rules.md",
    "docs/deviations.md"
  ]

  test "every guide is an extra, in the Guides group, and every extra is a real file" do
    docs = Mix.Project.config() |> Keyword.fetch!(:docs)
    # An extra is a path, or `{path, opts}` when ExDoc needs a title it cannot read from the file.
    extras = docs |> Keyword.fetch!(:extras) |> Enum.map(&extra_path/1)
    guides = "docs/guides/*.md" |> Path.wildcard() |> Enum.sort()
    pattern = docs |> Keyword.fetch!(:groups_for_extras) |> Keyword.fetch!(:Guides)

    assert guides != [], "docs/guides/ is empty; the wildcard would make this pass vacuously"

    assert guides -- extras == [],
           "a guide that renders nowhere: " <> inspect(guides -- extras)

    ungrouped = Enum.reject(guides, &Regex.match?(pattern, &1))

    assert ungrouped == [],
           "a guide the Guides pattern does not match: " <> inspect(ungrouped)

    overreach = Enum.filter(@prose_extras, &Regex.match?(pattern, &1))

    assert overreach == [],
           "the Guides pattern also matches prose: " <> inspect(overreach)

    assert Enum.sort(extras) == Enum.sort(guides ++ @prose_extras),
           "extras and the files on disk disagree: " <>
             inspect(Enum.sort(extras) -- Enum.sort(guides ++ @prose_extras)) <>
             " / " <> inspect(Enum.sort(guides ++ @prose_extras) -- Enum.sort(extras))

    for extra <- extras do
      assert File.exists?(extra), "`:extras` names #{extra}, which does not exist"
    end
  end

  defp extra_path({path, _opts}), do: to_string(path)
  defp extra_path(path), do: path

  # **ExDoc resolves an extra link by basename alone** (`config.extras[Path.basename(path)]`), so
  # a path that is wrong in a clone renders fine on the docs site, and a basename belonging to a
  # *different* extra retargets silently. Both halves are checked: does the path resolve on disk
  # relative to the file it is in (GitHub's question), and which extra will ExDoc pick (it must
  # be the file the path names). A link whose basename is no extra at all 404s on the published
  # site, so there are none — repo-only files are plain code spans.
  #
  # This package links **out** as well, to `hexdocs.pm/latu/...` for anything Latu owns. Those
  # are absolute and skipped here; `mix docs`' own `warnings_as_errors` is what catches a broken
  # autolink to a Latu module.
  @exdoc_extensions [".md", ".livemd", ".cheatmd", ".txt", ""]

  test "every relative link in the docs resolves, and to the file it names" do
    by_basename = Map.new(extras(), &{Path.basename(&1), &1})

    problems =
      for {source, line, text, href} <- links(),
          problem <- link_problem(source, href, by_basename),
          do: "#{source}:#{line}: [#{text}](#{href}) — #{problem}"

    assert problems == []
  end

  defp link_problem(source, href, by_basename) do
    path = href |> String.split("#") |> hd()
    target = path |> Path.expand(Path.dirname(source)) |> rel()
    resolved = by_basename[Path.basename(path)]

    cond do
      path == "" -> []
      not File.exists?(target) -> ["no such file: #{target}"]
      Path.extname(path) not in @exdoc_extensions -> []
      resolved == target -> []
      resolved != nil -> ["ExDoc matches the basename, so this renders as a link to #{resolved}"]
      true -> ["no extra has this basename, so it stays a relative href and 404s in `mix docs`"]
    end
  end

  @link ~r/\[([^\]]*)\]\(([^)\s]+)\)/

  defp links do
    for source <- extras(),
        {line, number} <- source |> File.read!() |> String.split("\n") |> Enum.with_index(1),
        [_all, text, href] <- Regex.scan(@link, line),
        not String.starts_with?(href, ["http://", "https://", "#", "mailto:"]),
        do: {source, number, text, href}
  end

  defp extras do
    Mix.Project.config()
    |> Keyword.fetch!(:docs)
    |> Keyword.fetch!(:extras)
    |> Enum.map(&extra_path/1)
  end

  defp rel(absolute), do: Path.relative_to_cwd(absolute)

  # The whole app, generated modules included — they are the majority and the point.
  defp modules, do: Application.spec(:latu_ml, :modules)

  defp documented_modules do
    for module <- modules(),
        match?(
          {:docs_v1, _, _, _, doc, _, _} when doc not in [:hidden, :none],
          Code.fetch_docs(module)
        ),
        do: module
  end
end
