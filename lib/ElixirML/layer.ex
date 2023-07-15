defmodule ElixirML.Layer do
  @enforce_keys [:type]
  # Types: input, linear (feedforward), activation
  defstruct [:type, :size, :activation]

  @type activation :: :relu | :sigmoid | :softmax
end
