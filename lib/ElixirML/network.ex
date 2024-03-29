defmodule ElixirML.Network do
  alias ElixirML.Network
  alias ElixirML.Matrix
  alias ElixirML.Layer
  import ExUnit.Assertions

  @enforce_keys [:layers, :loss, :optimiser]
  defstruct [:layers, :loss, :optimiser, :weights, :biases]

  @type t :: %__MODULE__{
          layers: nonempty_list(Layer.t()),
          loss: loss,
          optimiser: optimiser,
          weights: ElixirML.matrices(),
          biases: ElixirML.matrices()
        }

  @type loss :: :cross_entropy | :quadratic
  @type optimiser :: :adamw

  @spec each_batch(
          Network.t(),
          ElixirML.matricies(),
          ElixirML.matricies(),
          pos_integer,
          non_neg_integer
        ) :: Network.t()
  defp each_batch(network, batched_images, batched_labels, batch_nums, i \\ 0)

  defp each_batch(network, _, _, batch_nums, i) when i == batch_nums,
    do: network

  defp each_batch(network, batched_images, batched_labels, batch_nums, i) do
    images = Enum.at(batched_images, i)
    labels = Enum.at(batched_labels, i)

    network = network

    # n = 0

    # MNIST.print(
    #   Enum.slice(images.nums, (n * 28 * 28)..((n + 1) * 28 * 28 - 1)),
    #   Enum.slice(labels.nums, (n * 10)..((n + 1) * 10 - 1)),
    #   [
    #     "T-shirt/top",
    #     "Trouser",
    #     "Pullover",
    #     "Dress",
    #     "Coat",
    #     "Sandal",
    #     "Shirt",
    #     "Sneaker",
    #     "Bag",
    #     "Ankle boot"
    #   ]
    # )

    each_batch(network, batched_images, batched_labels, batch_nums, i + 1)
  end

  @spec each_epoch(Network.t(), Matrix.t(), Matrix.t(), pos_integer, non_neg_integer) ::
          Network.t()
  defp each_epoch(network, images, labels, epochs, batch_size, i \\ 0)

  defp each_epoch(network, _, _, epochs, _, i) when i == epochs,
    do: network

  defp each_epoch(network, images, labels, epochs, batch_size, i) do
    [images, labels] = Matrix.shuffle(images, labels)

    batched_images = Matrix.batch(images, batch_size)
    batched_labels = Matrix.batch(labels, batch_size)

    network = each_batch(network, batched_images, batched_labels, length(batched_images))

    IO.puts("Epoch #{i + 1}")

    each_epoch(network, images, labels, epochs, batch_size, i + 1)
  end

  alias ElixirML.Network

  @spec train(Network.t(), Matrix.t(), Matrix.t(), pos_integer, pos_integer) :: Network.t()
  def train(network, images, labels, epochs, batch_size) do
    IO.puts("Training...")

    each_epoch(network, images, labels, epochs, batch_size)
  end
end
