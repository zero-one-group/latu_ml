defmodule Latu.ML.Recommendation do
  @moduledoc """
  Collaborative filtering.

  One operator: alternating least squares over a user/item/rating frame. Its model holds the
  learned factors, and recommends in bulk rather than one row at a time — which is why most of
  `Latu.ML.Recommendation.ALSModel` answers with frames.

  Every constructor here is generated from PySpark 4.2.0's own param table, and every
  accessor module from the server's own attribute allowlist. Nothing in this file is
  hand-written but the words you are reading — see `Latu.ML.operators/1` for the table it all
  comes from.
  """

  require Latu.ML.Generate

  Latu.ML.Generate.constructors(:recommendation)
end

require Latu.ML.Generate

Latu.ML.Generate.accessors(:recommendation)
