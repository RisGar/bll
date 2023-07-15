defmodule ElixirML.Matrix do
  alias ElixirML.NIFs
  alias ElixirML.Matrix
  alias ElixirML.Layer

  defstruct [:rows, :cols, :nums]

  @spec rand(integer | {integer, integer}) :: %Matrix{}
  @doc ~S"Generates an $(m \times n)$ or $(1 \times n)$ sized matrix filled with random numbers"
  def rand({rows, cols}) when is_integer(rows) and is_integer(cols),
    do: NIFs.mat_rand(rows, cols)

  def rand(cols) when is_integer(cols),
    do: NIFs.mat_rand(1, cols)

  @spec fill(integer | {integer, integer}, float | nonempty_list) :: %Matrix{}
  @doc ~S"Fills an $(m \times n)$ or $(1 \times n)$ sized matrix filled with the specified value(s)"
  def fill({rows, cols}, value) when is_integer(rows) and is_integer(cols) and is_float(value),
    do: NIFs.mat_fill(rows, cols, value)

  def fill(cols, value) when is_integer(cols) and is_float(value),
    do: NIFs.mat_fill(1, cols, value)

  def fill({rows, cols}, values) when is_integer(rows) and is_integer(cols) and is_list(values),
    do: NIFs.mat_fill_vals(rows, cols, values)

  def fill(cols, values) when is_integer(cols) and is_list(values),
    do: NIFs.mat_fill_vals(1, cols, values)

  @spec activate(%Matrix{}, Layer.activation()) :: %Matrix{}
  @doc ~S"Applies an activation function to every value in a matrix"
  def activate(%Matrix{} = mat, act) when is_atom(act) do
    case act do
      :sigmoid -> NIFs.mat_sig(mat)
      :relu -> NIFs.mat_relu(mat)
      # :softmax -> NIFs.mat_softmax(mat)
      _ -> raise "Invalid activation function"
    end
  end

  @spec sum(%Matrix{}, %Matrix{}) :: %Matrix{}
  @doc ~S"Performs matrix addition on two matrices of equal dimensions"
  def sum(%Matrix{} = a, %Matrix{} = b),
    do: NIFs.mat_sum(a, b)

  @spec prod(%Matrix{}, %Matrix{}) :: %Matrix{}
  @doc ~S"Performs a matrix-matrix multiplication using cgemm"
  def prod(%Matrix{} = a, %Matrix{} = b),
    do: NIFs.mat_prod(a, b)

  @spec shuffle(%Matrix{}, %Matrix{}) :: nonempty_list(%Matrix{})
  @doc ~S"Shuffles the first dimensiion (rows) of a matrix"
  def shuffle(%Matrix{} = a, %Matrix{} = b), do: NIFs.mat_shuffle_rows(a, b)

  @spec batch(%Matrix{}, integer) :: nonempty_list(%Matrix{})
  @doc ~S"Seperates the matrix into batches of the specified size"
  def batch(%Matrix{} = mat, batch_size) when is_integer(batch_size),
    do: NIFs.mat_batch(mat, batch_size)
end
