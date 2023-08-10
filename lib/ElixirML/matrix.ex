defmodule ElixirML.Matrix do
  alias ElixirML.NIFs
  alias ElixirML.Matrix
  alias ElixirML.Layer

  @enforce_keys [:rows, :cols, :nums]
  defstruct [:rows, :cols, :nums]

  @type t :: %__MODULE__{
          rows: pos_integer,
          cols: pos_integer,
          nums: ElixirML.mat_elems()
        }

  @spec rand(pos_integer | {pos_integer, pos_integer}) :: Matrix.t()
  @doc ~S"Generates an $(m \times n)$ or $(1 \times n)$ sized matrix filled with random numbers"
  def rand({rows, cols}), do: NIFs.mat_rand(rows, cols)
  def rand(cols), do: NIFs.mat_rand(1, cols)

  @spec fill(pos_integer | {pos_integer, pos_integer}, float | ElixirML.mat_elems()) :: Matrix.t()
  @doc ~S"Fills an $(m \times n)$ or $(1 \times n)$ sized matrix filled with the specified value(s)"
  def fill({rows, cols}, value) when is_float(value), do: NIFs.mat_fill(rows, cols, value)
  def fill(cols, value) when is_float(value), do: NIFs.mat_fill(1, cols, value)
  def fill({rows, cols}, values) when is_list(values), do: NIFs.mat_fill_vals(rows, cols, values)
  def fill(cols, values) when is_list(values), do: NIFs.mat_fill_vals(1, cols, values)

  @spec activate(Matrix.t(), Layer.activation()) :: Matrix.t()
  @doc ~S"Applies an activation function to every value in a matrix"
  def activate(mat, act) when is_atom(act) do
    case act do
      :sigmoid -> NIFs.mat_sig(mat)
      :relu -> NIFs.mat_relu(mat)
      # :softmax -> NIFs.mat_softmax(mat)
      _ -> raise "Invalid activation function"
    end
  end

  @spec sum(Matrix.t(), Matrix.t()) :: Matrix.t()
  @doc ~S"Performs matrix addition on two matrices of equal dimensions"
  def sum(a, b),
    do: NIFs.mat_sum(a, b)

  @spec prod(Matrix.t(), Matrix.t()) :: Matrix.t()
  @doc ~S"Performs a matrix-matrix multiplication using cgemm"
  def prod(a, b),
    do: NIFs.mat_prod(a, b)

  @spec shuffle(Matrix.t(), Matrix.t()) :: ElixirML.Network.matricies()
  @doc ~S"Shuffles the first dimensiion (rows) of a matrix"
  def shuffle(a, b), do: NIFs.mat_shuffle_rows(a, b)

  @spec batch(Matrix.t(), pos_integer) :: ElixirML.matrices()
  @doc ~S"Seperates the matrix into batches of the specified size"
  def batch(mat, batch_size) when is_integer(batch_size),
    do: NIFs.mat_batch(mat, batch_size)
end
