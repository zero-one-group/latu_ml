defmodule Latu.ML.Linalg do
  @moduledoc """
  Spark's `Vector` and `Matrix`, both ways.

  Spark serialises these as struct literals typed by a JVM class rather than by field names,
  so Latu hands them back as `Latu.Result.UDT` — the class, and the elements in the order that
  class defines them, with nothing asserted about how many. Asserting is this module's job, and
  the layouts are PySpark's `pyspark/ml/connect/serialize.py`, element for element.

  A dense vector becomes an `Nx.Tensor`; a sparse one becomes a `Latu.ML.SparseVector`, because
  `Nx` has no sparse tensor and quietly densifying a vector that is sparse on purpose is not a
  decision this package makes. A matrix always comes back **row-major**, whatever
  `isTransposed` said on the wire — that flag is a storage detail, not a shape.
  """

  alias Latu.Error
  alias Latu.ML.SparseVector
  alias Latu.Protocol.Spark.Connect, as: Proto
  alias Latu.Result.UDT

  @vector "org.apache.spark.ml.linalg.VectorUDT"
  @matrix "org.apache.spark.ml.linalg.MatrixUDT"

  @typedoc """
  What a `Vector` or `Matrix` literal decodes to.

  `nil` where the server sent an empty one. That is not a failure and not a missing value: some
  attributes are genuinely empty on some models, and `Nx` cannot represent a tensor with a
  zero-sized dimension.
  """
  @type decoded :: Nx.Tensor.t() | SparseVector.t() | nil

  # =============================================
  # Decoding
  # =============================================

  @doc """
  Turn a `Latu.Result.UDT` into a tensor or a `Latu.ML.SparseVector`.

      iex> udt = %Latu.Result.UDT{class: "org.apache.spark.ml.linalg.VectorUDT",
      ...>                        elements: [1, nil, nil, [0.5, -1.5]]}
      iex> {:ok, tensor} = Latu.ML.Linalg.from_udt(udt)
      iex> Nx.to_flat_list(tensor)
      [0.5, -1.5]

      iex> udt = %Latu.Result.UDT{class: "org.apache.spark.ml.linalg.VectorUDT",
      ...>                        elements: [0, 4, [1, 3], [2.0, 4.0]]}
      iex> Latu.ML.Linalg.from_udt(udt)
      {:ok, %Latu.ML.SparseVector{size: 4, indices: [1, 3], values: [2.0, 4.0]}}

  An empty answer is an answer, and the server does send them — `NaiveBayesModel.sigma` is a
  0x0 Matrix on a multinomial model, because sigma belongs to the gaussian one. `Nx` has no
  empty tensor at all (a dimension must be a positive integer), so this is `nil` rather than a
  shape nothing can hold:

      iex> udt = %Latu.Result.UDT{class: "org.apache.spark.ml.linalg.MatrixUDT",
      ...>                        elements: [1, 0, 0, nil, nil, [], true]}
      iex> Latu.ML.Linalg.from_udt(udt)
      {:ok, nil}
  """
  @spec from_udt(UDT.t()) :: {:ok, decoded()} | {:error, Error.t()}
  def from_udt(%UDT{class: @vector, elements: elements}) when length(elements) == 4 do
    case elements do
      [1, _size, _indices, values] -> dense_vector(values)
      [0, size, indices, values] -> sparse_vector(size, indices, values)
      [type | _] -> decode_error("unknown Vector type #{inspect(type)}, expected 0 or 1")
    end
  end

  def from_udt(%UDT{class: @matrix, elements: elements}) when length(elements) == 7 do
    case elements do
      [1, rows, cols, _ptrs, _row_indices, values, transposed?] ->
        dense_matrix(rows, cols, values, transposed?)

      [0, _rows, _cols, _ptrs, _row_indices, _values, _transposed?] ->
        decode_error(
          "a sparse Matrix has no Nx equivalent; save the model and read the artefact, " <>
            "or ask for a dense attribute"
        )

      [type | _] ->
        decode_error("unknown Matrix type #{inspect(type)}, expected 0 or 1")
    end
  end

  def from_udt(%UDT{class: class, elements: elements}) when class in [@vector, @matrix] do
    decode_error(
      "a #{short(class)} literal has #{expected(class)} elements, and the server sent " <>
        "#{length(elements)}"
    )
  end

  def from_udt(%UDT{class: nil}) do
    decode_error("a Python UDT names no JVM class; Latu ML reads Vector and Matrix only")
  end

  def from_udt(%UDT{class: class}) do
    decode_error("#{class} is not a Vector or a Matrix")
  end

  @doc "See `from_udt/1`. Raises instead of returning an error."
  @spec from_udt!(UDT.t()) :: decoded()
  def from_udt!(%UDT{} = udt) do
    case from_udt(udt) do
      {:ok, decoded} -> decoded
      {:error, error} -> raise error
    end
  end

  defp dense_vector([]), do: {:ok, nil}
  defp dense_vector(values) when is_list(values), do: {:ok, Nx.tensor(values, type: :f64)}

  defp dense_vector(other) do
    decode_error("a Vector's values are a list, not #{inspect(other)}")
  end

  defp sparse_vector(size, indices, values)
       when is_integer(size) and is_list(indices) and is_list(values) do
    if length(indices) == length(values) do
      {:ok, %SparseVector{size: size, indices: indices, values: values}}
    else
      decode_error("a sparse Vector has #{length(indices)} indices and #{length(values)} values")
    end
  end

  defp sparse_vector(size, _indices, _values) do
    decode_error("a sparse Vector's size is an integer, not #{inspect(size)}")
  end

  # Spark stores a dense matrix column-major unless `isTransposed` says the values are already
  # row-major. Either way the caller gets rows.
  defp dense_matrix(rows, cols, values, transposed?)
       when is_integer(rows) and is_integer(cols) and is_list(values) do
    if length(values) == rows * cols do
      if values == [] do
        {:ok, nil}
      else
        dense_matrix_of(rows, cols, values, transposed?)
      end
    else
      decode_error(
        "a #{rows}x#{cols} Matrix needs #{rows * cols} values, and the server sent " <>
          "#{length(values)}"
      )
    end
  end

  defp dense_matrix(rows, cols, _values, _transposed?) do
    decode_error("a Matrix's shape is two integers, not #{inspect({rows, cols})}")
  end

  defp dense_matrix_of(rows, cols, values, transposed?) do
    tensor = Nx.tensor(values, type: :f64)

    if transposed? do
      {:ok, Nx.reshape(tensor, {rows, cols})}
    else
      {:ok, tensor |> Nx.reshape({cols, rows}) |> Nx.transpose()}
    end
  end

  defp expected(@vector), do: 4
  defp expected(@matrix), do: 7

  defp short(@vector), do: "Vector"
  defp short(@matrix), do: "Matrix"

  defp decode_error(message), do: {:error, %Error{kind: :decode, message: message}}

  # =============================================
  # Encoding
  # =============================================

  @doc """
  Turn a tensor or a `Latu.ML.SparseVector` into the literal Spark expects.

  A rank-1 tensor is a dense vector, a rank-2 tensor a dense matrix sent row-major — which is
  what `isTransposed: true` means on the wire, and why it is set. Element order and the null
  placeholders are PySpark's `serialize_param`, so the oracle can diff this byte for byte.

  Raises rather than returning an error: this is a value on its way out, and a caller who hands
  over a rank-3 tensor has a bug, not a runtime condition.
  """
  @spec to_literal(Nx.Tensor.t() | SparseVector.t()) :: Latu.ML.Plan.literal()
  def to_literal(%SparseVector{} = vector) do
    udt(@vector, [
      byte(0),
      integer(vector.size),
      ints(vector.indices),
      doubles(vector.values)
    ])
  end

  def to_literal(tensor) do
    case Nx.shape(tensor) do
      {_size} ->
        udt(@vector, [byte(1), null(), null(), doubles(Nx.to_flat_list(tensor))])

      {rows, cols} ->
        udt(@matrix, [
          byte(1),
          integer(rows),
          integer(cols),
          null(),
          null(),
          doubles(Nx.to_flat_list(tensor)),
          boolean(true)
        ])

      shape ->
        raise ArgumentError,
              "a Spark Vector is rank 1 and a Matrix is rank 2, and this tensor is " <>
                inspect(shape)
    end
  end

  # The deprecated `struct_type` field, not `data_type`: PySpark still writes it there, the
  # server still reads it, and a golden diff against PySpark is the point.
  defp udt(class, elements) do
    %Proto.Expression.Literal{
      literal_type:
        {:struct,
         %Proto.Expression.Literal.Struct{
           struct_type: %Proto.DataType{
             kind: {:udt, %Proto.DataType.UDT{type: "udt", jvm_class: class}}
           },
           elements: elements
         }}
    }
  end

  defp byte(value), do: %Proto.Expression.Literal{literal_type: {:byte, value}}
  defp integer(value), do: %Proto.Expression.Literal{literal_type: {:integer, value}}
  defp boolean(value), do: %Proto.Expression.Literal{literal_type: {:boolean, value}}

  defp null do
    %Proto.Expression.Literal{
      literal_type: {:null, %Proto.DataType{kind: {:null, %Proto.DataType.NULL{}}}}
    }
  end

  defp ints(values), do: specialized({:ints, %Proto.Ints{values: values}})
  defp doubles(values), do: specialized({:doubles, %Proto.Doubles{values: values}})

  defp specialized(value_type) do
    %Proto.Expression.Literal{
      literal_type:
        {:specialized_array, %Proto.Expression.Literal.SpecializedArray{value_type: value_type}}
    }
  end
end
