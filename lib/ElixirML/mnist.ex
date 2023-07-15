defmodule ElixirML.MNIST do
  alias ElixirML.NIFs
  alias ElixirML.Matrix

  @filename "priv/data/mnist.bin"

  @spec load :: nonempty_list(%Matrix{})
  def load do
    case File.read(@filename) do
      {:ok, binary} -> :erlang.binary_to_term(binary)
      {:error, _} -> NIFs.mnist_load()
    end
  end

  @spec save(nonempty_list(%Matrix{})) :: nil
  def save(data) do
    case File.read(@filename) do
      {:ok, _} ->
        nil

      {:error, _} ->
        File.mkdir_p!(Path.dirname(@filename))
        File.write(@filename, :erlang.term_to_binary(data))
        nil
    end
  end

  @doc ~S"Prints the image to the terminal in 24 colours"
  @spec print(nonempty_list(float), nonempty_list(float), nonempty_list(String.t())) :: nil
  def print(image, label, label_list)
      when length(image) == 28 * 28 and length(label) == 10 and length(label_list) == 10 do
    for i <- 1..(28 * 28) do
      case Enum.at(image, i - 1) do
        n when n >= 23 / 24 -> IO.write(IO.ANSI.color_background(255) <> "  ")
        n when n >= 22 / 24 -> IO.write(IO.ANSI.color_background(254) <> "  ")
        n when n >= 21 / 24 -> IO.write(IO.ANSI.color_background(253) <> "  ")
        n when n >= 20 / 24 -> IO.write(IO.ANSI.color_background(252) <> "  ")
        n when n >= 19 / 24 -> IO.write(IO.ANSI.color_background(251) <> "  ")
        n when n >= 18 / 24 -> IO.write(IO.ANSI.color_background(250) <> "  ")
        n when n >= 17 / 24 -> IO.write(IO.ANSI.color_background(249) <> "  ")
        n when n >= 16 / 24 -> IO.write(IO.ANSI.color_background(248) <> "  ")
        n when n >= 15 / 24 -> IO.write(IO.ANSI.color_background(247) <> "  ")
        n when n >= 14 / 24 -> IO.write(IO.ANSI.color_background(246) <> "  ")
        n when n >= 13 / 24 -> IO.write(IO.ANSI.color_background(245) <> "  ")
        n when n >= 12 / 24 -> IO.write(IO.ANSI.color_background(244) <> "  ")
        n when n >= 11 / 24 -> IO.write(IO.ANSI.color_background(243) <> "  ")
        n when n >= 10 / 24 -> IO.write(IO.ANSI.color_background(242) <> "  ")
        n when n >= 09 / 24 -> IO.write(IO.ANSI.color_background(241) <> "  ")
        n when n >= 08 / 24 -> IO.write(IO.ANSI.color_background(240) <> "  ")
        n when n >= 07 / 24 -> IO.write(IO.ANSI.color_background(239) <> "  ")
        n when n >= 06 / 24 -> IO.write(IO.ANSI.color_background(238) <> "  ")
        n when n >= 05 / 24 -> IO.write(IO.ANSI.color_background(237) <> "  ")
        n when n >= 04 / 24 -> IO.write(IO.ANSI.color_background(236) <> "  ")
        n when n >= 03 / 24 -> IO.write(IO.ANSI.color_background(235) <> "  ")
        n when n >= 02 / 24 -> IO.write(IO.ANSI.color_background(234) <> "  ")
        n when n >= 01 / 24 -> IO.write(IO.ANSI.color_background(233) <> "  ")
        n when n >= 00 / 24 -> IO.write(IO.ANSI.color_background(232) <> "  ")
      end

      if rem(i, 28) == 0, do: IO.write(IO.ANSI.reset() <> "\n")
    end

    label_list
    |> Enum.at(Enum.find_index(label, &(&1 == 1)))
    |> IO.puts()

    nil
  end
end
