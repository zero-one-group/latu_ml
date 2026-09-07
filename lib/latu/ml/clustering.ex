defmodule Latu.ML.Clustering do
  @moduledoc """
  Clustering estimators, and the models they fit.

  Unsupervised: a `features` column goes in and a cluster assignment comes out, with the
  cluster centres reachable as tensors from the model's own module.

  Every constructor here is generated from PySpark 4.2.0's own param table, and every
  accessor module from the server's own attribute allowlist. Nothing in this file is
  hand-written but the words you are reading — see `Latu.ML.operators/1` for the table it all
  comes from.
  """

  require Latu.ML.Generate

  Latu.ML.Generate.constructors(:clustering)
end

require Latu.ML.Generate

Latu.ML.Generate.accessors(:clustering)
