defmodule Latu.ML.Operator do
  @moduledoc """
  One operator the registry knows, as inert data.

  The registry is where `latu_ml` keeps what it knows about MLlib: 111 operators on Spark
  4.2.0, read off PySpark's own param tables rather than transcribed. `Latu.ML.operators/1`
  hands these back so you can look, filter and count from `iex` rather than from the docs —
  the same table the constructors are generated from.

      iex> lr = Latu.ML.operator!(:logistic_regression)
      iex> lr.class
      "org.apache.spark.ml.classification.LogisticRegression"
      iex> lr.constructor
      {Latu.ML.Classification, :logistic_regression, 1}

  `constructor` is the generated function that builds one — `{module, name, arity}`, so
  `apply/3` works and so `h Latu.ML.Classification.logistic_regression` is one copy away.
  """

  alias Latu.ML.Param

  @typedoc """
  What an operator is.

  A `:model` has no constructor: it exists only by `Latu.ML.fit/2` or a load. It is in the
  registry anyway, because its params are what `Latu.ML.transform/2` resends and its class is
  what an accessor module checks against.

  A `:helper` is neither fitted nor applied: its one method reaches the server through the
  `ConnectHelper` object. Two on 4.2.0 — `PowerIterationClustering` and `PrefixSpan`.
  """
  @type kind :: :estimator | :transformer | :evaluator | :model | :helper

  @typedoc """
  Whether a bigger metric is a better one, for an evaluator. `nil` for everything else.

  `true` where every metric the evaluator offers is maximised, or `{:only, metrics}` /
  `{:except, metrics}` where it depends on which one is asked for. `Latu.ML.larger_better?/1`
  resolves it against an evaluator's own `metric_name`; a tuning fold picks its best model by
  the answer.

  Read off the six client-side `isLargerBetter` overrides, because the server's allowlist has
  no such method and a `Fetch` cannot ask for it.
  """
  @type larger_better :: true | {:only, [String.t()]} | {:except, [String.t()]} | nil

  @typedoc """
  How far this operator has been taken.

    * `:probed` — fitted or applied against a live server on the Spark version named here, and
      every allowlisted attribute answered. "Supported" in these docs means this.
    * `:built` — generated from PySpark's tables, and not working under the probe. `:reason`
      says what the probe saw, and is nil where the probe has not run at all.
    * `:missing` — PySpark names it, the server does not load it.
  """
  @type status :: :probed | :built | :missing

  @enforce_keys [:name, :python, :group, :kind, :class, :spark]
  defstruct [
    :name,
    :python,
    :group,
    :kind,
    :class,
    :model_class,
    :constructor,
    :doc,
    :reason,
    :spark,
    :larger_better,
    model_classes: [],
    params: [],
    status: :built
  ]

  @type t :: %__MODULE__{
          name: atom(),
          python: String.t(),
          group: atom(),
          kind: kind(),
          class: String.t(),
          model_class: String.t() | nil,
          model_classes: [String.t()],
          constructor: {module(), atom(), arity()} | nil,
          doc: String.t() | nil,
          reason: String.t() | nil,
          spark: String.t(),
          larger_better: larger_better(),
          params: [Param.t()],
          status: status()
        }
end
