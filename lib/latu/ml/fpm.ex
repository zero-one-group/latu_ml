defmodule Latu.ML.FPM do
  @moduledoc """
  Frequent pattern mining.

  Spark's own abbreviation, kept as one: `fpm` in PySpark, `FPM` here. Two operators, and both
  answer with frames rather than predictions — the frequent itemsets and the association rules
  a transaction log implies.

  Every constructor here is generated from PySpark 4.2.0's own param table, and every
  accessor module from the server's own attribute allowlist. Nothing in this file is
  hand-written but the words you are reading — see `Latu.ML.operators/1` for the table it all
  comes from.
  """

  require Latu.ML.Generate

  Latu.ML.Generate.constructors(:fpm)
end

require Latu.ML.Generate

Latu.ML.Generate.accessors(:fpm)
