defmodule Mix.Tasks.Fashion do
  @moduledoc "Example implementation of a network aiming to replicate a XOR gate"
  @shortdoc "Recognise items in FashionMNIST dataset"

  use Mix.Task
  alias ElixirML.MNIST
  alias ElixirML.Layer
  alias ElixirML.Network

  @impl Mix.Task
  def run(_args) do
    image_size = 28 * 28

    mnist =
      [train_images, _test_images, train_labels, _test_labels] =
      MNIST.load()

    MNIST.save(mnist)

    layers = [
      %Layer{layer_type: :input, size: image_size},
      %Layer{layer_type: :linear, size: 128},
      %Layer{layer_type: :activation, activation: :relu},
      %Layer{layer_type: :linear, size: 128},
      %Layer{layer_type: :activation, activation: :relu},
      %Layer{layer_type: :linear, size: 10},
      %Layer{layer_type: :activation, activation: :softmax}
    ]

    Network.init(
      layers,
      :cross_entropy,
      :adamw
    )
    |> Network.train(
      # input
      train_images,
      # target
      train_labels,
      # epochs
      30,
      # batch size
      10
    )
    # |> feedforward()
    |> IO.inspect()

    :ok
  end
end
