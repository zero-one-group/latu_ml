defmodule Latu.ML.Helper do
  @moduledoc """
  An operator that is neither fitted nor applied: its work is one method on the server's own
  helper object.

  Two of them on Spark 4.2.0 — `PowerIterationClustering` and `PrefixSpan` — and they are
  operators in every way but the one that matters to the wire. They carry a JVM class and a
  param table like an estimator does, and PySpark models them as classes; but there is no
  `Fit`, no `Transform` and nothing cached, because what they do is a `Fetch` on
  `ConnectHelper` with their params as **positional arguments**.

      pic = Latu.ML.Clustering.power_iteration_clustering(k: 2, max_iter: 10)
      clusters = Latu.ML.assign_clusters(pic, edges)

  That positional call is the whole difference from every other operator here, and it has one
  consequence worth knowing: an `MlParams` map carries only what the caller set and the server
  fills the rest, where a helper's arguments are all sent every time — so this package supplies
  the defaults from the registry, and refuses rather than guessing where a param has none that
  can be written down.

  No uid: nothing on the wire carries one, because nothing on the wire is an operator.
  """

  alias Latu.ML.Param

  @enforce_keys [:class]
  defstruct [:class, params: [], known: []]

  @type t :: %__MODULE__{
          class: String.t(),
          params: [Latu.ML.Plan.param()],
          known: [Param.t()]
        }
end
