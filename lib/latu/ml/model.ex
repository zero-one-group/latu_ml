defmodule Latu.ML.Model do
  @moduledoc """
  A fitted model: a reference into the session's ML cache, and the context needed to use it.

  **Not a value.** The model itself lives on the server, in a cache that offloads it to disk
  when memory is tight and evicts it when the session ends — so a `%Latu.ML.Model{}` can name
  something that is no longer there, and every verb that touches one can fail for that reason.
  `Latu.ML.with_model/3` is the bracket; `Latu.ML.delete/1` is the explicit release. There is no
  finalizer, because BEAM terms have none.

  `dataset` is the plan the model was fitted on, kept so a summary the server has evicted can
  be rebuilt from it. It is nil for a model from `Latu.ML.load/3`, which was not fitted here —
  and which has no summary on the server either, because a model's save format does not carry
  one. `uid` is the client's, copied from the estimator: a `Fit` answers with an `obj_ref` and
  nothing else. A loaded model's is the one the saved metadata held, which is the only command
  that gives a uid back.

  `class` is nil for a model whose estimator can fit more than one — `LDA` returns a
  `LocalLDAModel` or a `DistributedLDAModel` depending on its `optimizer`, and a `Fit` does not
  say which. `candidates` is then both of them, and is what an attribute is looked up in;
  nothing claims a class it cannot know.
  """

  @enforce_keys [:ref, :session]
  defstruct [:ref, :session, :class, :uid, :dataset, params: [], candidates: []]

  @type t :: %__MODULE__{
          ref: Latu.ML.Plan.object_ref(),
          session: Latu.Session.t(),
          class: String.t() | nil,
          candidates: [String.t()],
          uid: String.t() | nil,
          dataset: term(),
          params: [Latu.ML.Plan.param()]
        }
end
