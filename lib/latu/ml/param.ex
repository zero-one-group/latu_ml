defmodule Latu.ML.Param do
  @moduledoc """
  One parameter an operator accepts.

  Read off PySpark's own param tables by `dev/extract_ml_operators.py`, so drift is something a
  re-run notices rather than something a reader has to. `Latu.ML.params/1` hands the table for
  one operator back.

  `name` is the snake_case atom callers pass; `wire` is Spark's own camelCase spelling, and the
  key the server reads. `default` is documentation only — a default is never sent, because a
  param the caller did not set and a param sent with its default value are different requests.
  """

  @typedoc """
  A param's default, where there is one to state.

  Most are ordinary values. Four kinds are not, and each says why rather than inventing a
  number: `nil` is no default at all, `{:uid_suffix, s}` is the operator's own random uid with
  `s` on the end, `:server_supplied` is a value only a running JVM knows, `:varies` is one that
  is not the same twice, and `:nan` is NaN, which has no Elixir literal.
  """
  @type default ::
          term() | nil | {:uid_suffix, String.t()} | :server_supplied | :varies | :nan

  @enforce_keys [:name, :wire, :type]
  defstruct [:name, :wire, :type, :default, :doc]

  @type t :: %__MODULE__{
          name: atom(),
          wire: String.t(),
          type: Latu.ML.Plan.param_type(),
          default: default(),
          doc: String.t() | nil
        }
end
