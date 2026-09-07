defmodule Latu.ML.Regression do
  @moduledoc """
  Regressors, and the models they fit.

  Everything here predicts a number. For the linear families the fitted coefficients *are* the
  model, so reading them back as tensors and scoring on the BEAM with `Nx` is a real option —
  see the model modules in this namespace.

  Every constructor here is generated from PySpark 4.2.0's own param table, and every
  accessor module from the server's own attribute allowlist. Nothing in this file is
  hand-written but the words you are reading — see `Latu.ML.operators/1` for the table it all
  comes from.
  """

  require Latu.ML.Generate

  Latu.ML.Generate.constructors(:regression)
end

require Latu.ML.Generate

Latu.ML.Generate.accessors(:regression)
