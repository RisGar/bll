defmodule ElixirML.NIFs do
  use Rustler, otp_app: :elixir_ml, crate: "elixirml_nifs"

  # When your NIF is loaded, it will override these functions.
  def mat_rand(_rows, _cols), do: :erlang.nif_error(:nif_not_loaded)
  def mat_fill(_rows, _cols, _value), do: :erlang.nif_error(:nif_not_loaded)
  def mat_fill_vals(_rows, _cols, _values), do: :erlang.nif_error(:nif_not_loaded)
  def mat_softmax(_mat), do: :erlang.nif_error(:nif_not_loaded)
  def mat_sig(_mat), do: :erlang.nif_error(:nif_not_loaded)
  def mat_relu(_mat), do: :erlang.nif_error(:nif_not_loaded)
  def mat_sum(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def mat_prod(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def mat_shuffle_rows(_a, _b), do: :erlang.nif_error(:nif_not_loaded)
  def mat_batch(_mat, _batch_size), do: :erlang.nif_error(:nif_not_loaded)

  def mnist_load(), do: :erlang.nif_error(:nif_not_loaded)
end
