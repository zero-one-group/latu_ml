defmodule Latu.ML.Estimator do
  @moduledoc """
  An unfitted estimator: inert data, no session, no server-side anything.

  `Latu.ML.fit/2` is what turns one into a `Latu.ML.Model`, and the frame it is fitted on is
  where the session comes from.
  """

  alias Latu.ML.Internal
  alias Latu.ML.Param

  @enforce_keys [:class, :uid]
  defstruct [:class, :uid, :model_class, params: [], known: [], model_candidates: []]

  @type t :: %__MODULE__{
          class: String.t(),
          uid: String.t(),
          model_class: String.t() | nil,
          model_candidates: [String.t()],
          params: [Latu.ML.Plan.param()],
          known: [Param.t()]
        }

  @doc """
  An estimator the registry does not know, by its JVM class name.

  The escape hatch for anything on the server's classpath that MLlib did not put there. Nothing
  checks the param names — there is no table to check them against — so the server's error is
  the only one you get, and `model_class` is yours to supply if you want the model's attributes
  to resolve later.

  ## Options

    * `:params` — the params to send, already in wire form: `{name, type, value}` per param,
      where `name` is Spark's own spelling. Defaults to `[]`.
    * `:model_class` — the JVM class of the model a fit produces. Defaults to `nil`, which
      leaves `Latu.ML.Model`'s `:class` unset.
    * `:uid` — the uid to send. Defaults to a generated one in PySpark's shape.

  ## Example

      Latu.ML.Estimator.new("ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier",
        params: [{"numRound", :int, 100}, {"max_depth", :int, 6}])

  A param's name is whatever the library declared it, and that is not always one convention:
  XGBoost4J spells its own params camelCase and the native booster's snake_case, and the server
  resolves each exactly as written.
  """
  @spec new(String.t(), keyword()) :: t()
  def new(class, opts \\ []) when is_binary(class) do
    opts = Keyword.validate!(opts, params: [], model_class: nil, uid: nil)

    %__MODULE__{
      class: class,
      uid: opts[:uid] || Internal.uid(class),
      model_class: opts[:model_class],
      params: opts[:params]
    }
  end
end
