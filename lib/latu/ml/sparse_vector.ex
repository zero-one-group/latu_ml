defmodule Latu.ML.SparseVector do
  @moduledoc """
  A Spark sparse vector, kept sparse.

  `Nx` has no sparse tensor, so densifying is the only way to make one — and a sparse vector
  is usually sparse for a reason. This struct is what `Latu.ML.Linalg.from_udt/1` hands back
  instead, and `to_dense/1` is the caller's decision, never Latu ML's.

  `indices` are ascending and zero-based, and `values` runs parallel to them.
  """

  @enforce_keys [:size, :indices, :values]
  defstruct [:size, :indices, :values]

  @type t :: %__MODULE__{
          size: non_neg_integer(),
          indices: [non_neg_integer()],
          values: [float()]
        }

  @doc """
  Fill in the zeros: an `{size}` `f64` tensor.

      iex> vector = %Latu.ML.SparseVector{size: 4, indices: [1, 3], values: [2.0, 4.0]}
      iex> Nx.to_flat_list(Latu.ML.SparseVector.to_dense(vector))
      [0.0, 2.0, 0.0, 4.0]
  """
  @spec to_dense(t()) :: Nx.Tensor.t()
  def to_dense(%__MODULE__{indices: []} = vector), do: zeros(vector.size)

  def to_dense(%__MODULE__{} = vector) do
    Nx.indexed_put(
      zeros(vector.size),
      Nx.reshape(Nx.tensor(vector.indices), {length(vector.indices), 1}),
      Nx.tensor(vector.values, type: :f64)
    )
  end

  defp zeros(size), do: Nx.broadcast(Nx.tensor(0.0, type: :f64), {size})
end
