defmodule Latu.ML.Evaluation do
  @moduledoc """
  Evaluators: one metric over a scored frame.

  An evaluator is neither fitted nor applied. It reads the columns a model wrote and answers
  with one number, and `metric_name:` is what picks which number.

  Every constructor here is generated from PySpark 4.2.0's own param table, and every
  accessor module from the server's own attribute allowlist. Nothing in this file is
  hand-written but the words you are reading — see `Latu.ML.operators/1` for the table it all
  comes from.
  """

  require Latu.ML.Generate

  Latu.ML.Generate.constructors(:evaluation)
end

require Latu.ML.Generate

Latu.ML.Generate.accessors(:evaluation)
