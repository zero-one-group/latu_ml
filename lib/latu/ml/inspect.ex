# How the ML structs show in `iex` and in a Livebook cell. Latu's own `Latu.DataFrame` does the
# same thing for the same reason: the default struct inspect prints a plan or a param list that
# is pages long and says nothing, and the one fact a reader wants is which operator this is.
#
# There is deliberately **no `Kino.Render` impl** for any of them, which is Latu's precedent for
# a `Latu.Session` and a `Latu.GroupedData`: Kino falls back to `inspect/2`, and a handle has no
# HTML worth rendering. What does render is what a model *produces* — `Latu.ML.transform/2` and
# `attribute_frame/2` hand back a `Latu.DataFrame`, which Latu already renders.

defimpl Inspect, for: Latu.ML.Model do
  import Inspect.Algebra

  def inspect(model, opts) do
    concat(["#Latu.ML.Model<", Latu.ML.Inspect.body(model, opts), ">"])
  end
end

defimpl Inspect, for: Latu.ML.Summary do
  import Inspect.Algebra

  def inspect(summary, opts) do
    concat(["#Latu.ML.Summary<", Latu.ML.Inspect.body(summary, opts), ">"])
  end
end

defimpl Inspect, for: Latu.ML.Estimator do
  import Inspect.Algebra

  def inspect(estimator, opts) do
    concat(["#Latu.ML.Estimator<", Latu.ML.Inspect.operator(estimator, opts), ">"])
  end
end

defimpl Inspect, for: Latu.ML.Transformer do
  import Inspect.Algebra

  def inspect(transformer, opts) do
    concat(["#Latu.ML.Transformer<", Latu.ML.Inspect.operator(transformer, opts), ">"])
  end
end

defimpl Inspect, for: Latu.ML.Evaluator do
  import Inspect.Algebra

  def inspect(evaluator, opts) do
    concat(["#Latu.ML.Evaluator<", Latu.ML.Inspect.operator(evaluator, opts), ">"])
  end
end

defmodule Latu.ML.Inspect do
  @moduledoc false
  # The shared bodies. A module rather than five copies, and `@moduledoc false` because nobody
  # calls this on purpose.

  import Inspect.Algebra

  @doc """
  A handle: the short class name, then the reference, truncated the way a uuid should be.

  A model with no class shows its candidates instead — an `LDA` fit is a `LocalLDAModel` or a
  `DistributedLDAModel` and nothing client-side knows which, so showing one of them would be a
  lie and showing neither would be unhelpful.
  """
  def body(holder, opts) do
    concat([short(holder), " ", to_doc(abbreviate(holder.ref), opts)])
  end

  @doc "An unfitted operator: the short class name, then only the params the caller set."
  def operator(operator, opts) do
    set = for {wire, _type, value} <- operator.params, do: {String.to_atom(wire), value}

    case set do
      [] -> short(operator)
      params -> concat([short(operator), " ", to_doc(params, opts)])
    end
  end

  defp short(%{class: class}) when is_binary(class), do: last(class)

  defp short(%{candidates: [_ | _] = candidates}) do
    Enum.map_join(candidates, " | ", &last/1)
  end

  defp short(_holder), do: "?"

  defp last(class), do: class |> String.split(".") |> List.last()

  # A cache id is a uuid, and its first group is enough to tell two apart in a session while
  # staying short enough that the class is still the first thing read. Anything already short
  # is shown whole — the server assigns uuids, but a hand-built model may carry anything, and
  # cutting `"m-3"` down to `"m…"` would lose the only part that identified it.
  @keep 12

  defp abbreviate(ref) when is_binary(ref) do
    if String.length(ref) <= @keep do
      ref
    else
      ref |> String.split("-") |> hd() |> String.slice(0, @keep) |> Kernel.<>("…")
    end
  end

  defp abbreviate(other), do: other
end
