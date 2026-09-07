defmodule Latu.ML.LinalgTest do
  use ExUnit.Case, async: true

  doctest Latu.ML.Linalg
  # `SparseVector.to_dense/1`'s example lives here rather than in a file of its own: it is one
  # function, and densifying is this module's subject.
  doctest Latu.ML.SparseVector

  alias Latu.ML.Linalg
  alias Latu.ML.SparseVector
  alias Latu.Result.UDT

  @vector "org.apache.spark.ml.linalg.VectorUDT"
  @matrix "org.apache.spark.ml.linalg.MatrixUDT"

  defp vector(elements), do: %UDT{class: @vector, elements: elements}
  defp matrix(elements), do: %UDT{class: @matrix, elements: elements}

  defp round_trip(value) do
    {:ok, udt} = value |> Linalg.to_literal() |> Latu.Result.Literal.value()

    Linalg.from_udt(udt)
  end

  describe "decoding a vector" do
    test "a dense vector is a tensor" do
      assert {:ok, tensor} = Linalg.from_udt(vector([1, nil, nil, [1.0, 2.0, 3.0]]))
      assert Nx.to_flat_list(tensor) == [1.0, 2.0, 3.0]
      assert Nx.type(tensor) == {:f, 64}
    end

    test "a sparse vector stays sparse" do
      assert {:ok, sparse} = Linalg.from_udt(vector([0, 5, [0, 4], [1.0, 2.0]]))
      assert sparse == %SparseVector{size: 5, indices: [0, 4], values: [1.0, 2.0]}
    end

    test "indices and values that disagree are refused" do
      assert {:error, error} = Linalg.from_udt(vector([0, 5, [0, 4], [1.0]]))
      assert error.message =~ "2 indices and 1 values"
    end

    test "an unknown type flag is refused, naming what was expected" do
      assert {:error, error} = Linalg.from_udt(vector([7, nil, nil, []]))
      assert error.message =~ "expected 0 or 1"
    end

    test "the element count is this package's assertion, not Latu's" do
      assert {:error, error} = Linalg.from_udt(vector([1, nil, nil]))
      assert error.message =~ "Vector literal has 4 elements"
      assert error.message =~ "server sent 3"
    end
  end

  describe "decoding a matrix" do
    # Spark stores column-major unless isTransposed says otherwise. Both give the caller rows.
    test "column-major values come back row-major" do
      column_major = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]

      assert {:ok, tensor} = Linalg.from_udt(matrix([1, 2, 3, nil, nil, column_major, false]))
      assert Nx.to_list(tensor) == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    end

    test "row-major values come back unchanged" do
      row_major = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

      assert {:ok, tensor} = Linalg.from_udt(matrix([1, 2, 3, nil, nil, row_major, true]))
      assert Nx.to_list(tensor) == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    end

    test "a shape that does not match the values is refused" do
      assert {:error, error} = Linalg.from_udt(matrix([1, 2, 3, nil, nil, [1.0], true]))
      assert error.message =~ "needs 6 values"
    end

    test "a sparse matrix is refused, naming the way out" do
      sparse = matrix([0, 2, 2, [0, 1, 2], [0, 1], [1.0, 2.0], false])

      assert {:error, error} = Linalg.from_udt(sparse)
      assert error.message =~ "no Nx equivalent"
    end
  end

  describe "decoding anything else" do
    test "a Python UDT is refused" do
      assert {:error, error} = Linalg.from_udt(%UDT{class: nil, elements: []})
      assert error.message =~ "Python UDT"
    end

    test "another JVM class is refused by name" do
      assert {:error, error} = Linalg.from_udt(%UDT{class: "com.example.ThingUDT", elements: []})
      assert error.message =~ "com.example.ThingUDT"
    end

    # Through `apply/3`: 1.20's type checker can prove the direct call raises, and
    # warnings-as-errors fails the build on that conclusion. Latu's `local_data_test.exs` has
    # the same pattern for the same reason.
    test "from_udt! raises the same error" do
      udt = %UDT{class: nil, elements: []}

      assert_raise Latu.Error, ~r/Python UDT/, fn -> apply(Linalg, :from_udt!, [udt]) end
    end
  end

  # The round trip is the real assertion: this package encodes, and *Latu's own decoder* reads
  # it back. If the struct_type field or the element order were wrong, this is where it shows.
  describe "encoding, read back through Latu" do
    test "a dense vector" do
      assert {:ok, tensor} = round_trip(Nx.tensor([1.5, -2.5], type: :f64))
      assert Nx.to_flat_list(tensor) == [1.5, -2.5]
    end

    test "a sparse vector" do
      sparse = %SparseVector{size: 4, indices: [1, 3], values: [2.0, 4.0]}

      assert round_trip(sparse) == {:ok, sparse}
    end

    test "a matrix" do
      assert {:ok, tensor} = round_trip(Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f64))
      assert Nx.to_list(tensor) == [[1.0, 2.0], [3.0, 4.0]]
    end

    # A square matrix cannot distinguish numRows from numCols, nor a transposed flat order from
    # a straight one — both are the identity on 2x2. A wide one does, so the encoder's element
    # order is actually under test here rather than only in the decoder above.
    test "a non-square matrix" do
      wide = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f64)

      assert {:ok, tensor} = round_trip(wide)
      assert Nx.to_list(tensor) == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    end

    test "the dense vector layout is PySpark's, element for element" do
      literal = Linalg.to_literal(Nx.tensor([1.0], type: :f64))
      {:struct, struct} = literal.literal_type

      assert {:udt, udt} = struct.struct_type.kind
      assert udt.jvm_class == "org.apache.spark.ml.linalg.VectorUDT"
      assert udt.type == "udt"

      assert [type, size, indices, values] = struct.elements
      assert type.literal_type == {:byte, 1}
      assert {:null, _} = size.literal_type
      assert {:null, _} = indices.literal_type
      assert {:specialized_array, array} = values.literal_type
      assert array.value_type == {:doubles, %Latu.Protocol.Spark.Connect.Doubles{values: [1.0]}}
    end
  end

  test "a rank-3 tensor is a bug, and says so" do
    assert_raise ArgumentError, ~r/rank 1 and a Matrix is rank 2/, fn ->
      Linalg.to_literal(Nx.broadcast(0.0, {2, 2, 2}))
    end
  end
end
