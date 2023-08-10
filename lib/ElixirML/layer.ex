defmodule ElixirML.Layer do
  @enforce_keys [:layer_type]
  defstruct [:layer_type, :size, :activation]

  @type t :: %__MODULE__{
          layer_type: layer_type,
          size: pos_integer,
          activation: activation
        }

  @type layer_type :: :dense | :input | :activation
  @type activation :: :relu | :sigmoid | :softmax
end
