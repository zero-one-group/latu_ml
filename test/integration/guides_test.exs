defmodule Latu.ML.GuidesTest do
  use ExUnit.Case, async: true

  # A guide is a Markdown page whose `elixir` fences are a script, and this runs them.
  #
  # The fences of one guide run in order, threading one binding and one environment
  # (`Code.eval_quoted_with_env/3`; `Code.eval_string/3` hands back no environment, so an alias
  # would not survive from one fence to the next). A guide asserts by matching —
  # `{6, 2} = Nx.shape(t)` is documentation and a test at once — and only `elixir` fences run.
  #
  # A fence the test server cannot run is preceded by a line beginning `> **Not executed.**` and
  # the runner skips it. A visible blockquote, not an HTML comment: the page tells the reader
  # every snippet is executed, so the exceptions have to be legible on the page.
  #
  # Ported from Latu's `test/integration/guides_test.exs`. Kept a copy rather than shared:
  # the two repos have different servers, different skip lists and different budgets, and the
  # thing worth sharing is thirty lines of fence parsing.
  #
  # Needs a Spark Connect server on :15003 — docker compose up -d.
  @moduletag :integration
  @moduletag :capture_log

  # A guide here fits models, and a fit is seconds of JVM before it is anything else. Latu's
  # budget is 180s for pages that only query; this is the same rule with fits allowed for.
  # **Not a licence to grow a guide** — a page that needs longer is a page nobody finishes.
  @moduletag timeout: 300_000

  @guides "docs/guides/*.md" |> Path.wildcard() |> Enum.sort()

  # Every skipped fence, by guide and reason. An opt-out nobody counts is an opt-out that
  # spreads, so a new one fails here until somebody writes down why.
  @illustrative [
    {"docs/guides/cookbook.md", "Bumblebee is not a dependency of this package."}
  ]

  # A wildcard that matched nothing would define no tests and pass in silence, which is the one
  # failure mode a runner like this cannot report on its own. Read the directory again rather
  # than asserting on `@guides`, which is a literal by the time the assertion runs.
  test "every guide has a test" do
    guides = "docs/guides/*.md" |> Path.wildcard() |> Enum.sort()

    assert guides != []
    assert guides == @guides
  end

  test "every fence that is not executed is named, with its reason" do
    skipped =
      for guide <- @guides,
          {_line, _code, {:skipped, reason}} <- fences(File.read!(guide)),
          do: {guide, reason}

    assert Enum.sort(skipped) == Enum.sort(@illustrative)
  end

  for guide <- @guides do
    test "#{guide}" do
      run(unquote(guide))
    end
  end

  # A guide must not leave state on the server: these run async beside every other integration
  # test. A page that fits a model deletes it, and a page that opens a session releases it.
  defp run(guide) do
    fences = fences(File.read!(guide))

    assert fences != [], "#{guide} has no `elixir` fences to run"

    runnable = for {line, code, :runs} <- fences, do: {line, code}

    assert runnable != [],
           "#{guide}'s fences are all marked not-executed; it is prose, not a guide"

    ExUnit.CaptureIO.capture_io(fn ->
      Enum.reduce(runnable, {[], Code.env_for_eval([])}, &eval(&1, &2, guide))
    end)
  end

  defp eval({line, code}, {binding, env}, guide) do
    quoted = Code.string_to_quoted!(code, file: guide, line: line)
    {_value, binding, env} = Code.eval_quoted_with_env(quoted, binding, env)
    {binding, env}
  rescue
    error ->
      flunk("""
      #{guide}:#{line} raised #{inspect(error.__struct__)}

      #{Exception.message(error)}

      #{code |> String.split("\n") |> hd()}
      """)
  end

  # `{line, code, :runs | {:skipped, reason}}` for every ```elixir fence, the line being the
  # fence's first line of code.
  defp fences(text) do
    text |> String.split("\n") |> Enum.with_index(1) |> gather([], :runs)
  end

  defp gather([], acc, _pending), do: Enum.reverse(acc)

  defp gather([{line, number} | rest], acc, pending) do
    cond do
      String.trim(line) == "```elixir" ->
        {body, tail} = Enum.split_while(rest, fn {l, _n} -> String.trim(l) != "```" end)
        code = body |> Enum.map(&elem(&1, 0)) |> Enum.join("\n")

        gather(Enum.drop(tail, 1), [{number + 1, code, pending} | acc], :runs)

      marker = marker(line) ->
        gather(rest, acc, marker)

      # A blank line between the marker and its fence is fine; anything else clears it, so a
      # marker cannot leak onto a fence further down the page.
      String.trim(line) == "" ->
        gather(rest, acc, pending)

      true ->
        gather(rest, acc, :runs)
    end
  end

  # `> **Not executed.** <reason>` — the reason is what the accounting test carries, so it has
  # to be on the same line and non-empty.
  defp marker(line) do
    case Regex.run(~r/^>\s+\*\*Not executed\.\*\*\s+(\S.*)$/, String.trim_trailing(line)) do
      [_all, reason] -> {:skipped, String.trim(reason)}
      nil -> nil
    end
  end
end
