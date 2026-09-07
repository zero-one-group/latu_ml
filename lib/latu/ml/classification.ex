defmodule Latu.ML.Classification do
  @moduledoc """
  Classifiers, and the models they fit.

  Everything here predicts a label. The estimators take a `features` Vector column and a
  `label` column and answer with a model whose attributes live in this module's namespace —
  `Latu.ML.Classification.LogisticRegressionModel` and its kind.

  Every constructor here is generated from PySpark 4.2.0's own param table, and every
  accessor module from the server's own attribute allowlist. Nothing in this file is
  hand-written but the words you are reading — see `Latu.ML.operators/1` for the table it all
  comes from.
  """

  require Latu.ML.Generate

  Latu.ML.Generate.constructors(:classification)
end

require Latu.ML.Generate

Latu.ML.Generate.accessors(:classification)
