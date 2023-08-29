defmodule ElixirML.MixProject do
  use Mix.Project

  def project do
    [
      app: :elixir_ml,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      compilers: Mix.compilers(),

      # Docs
      name: "ElixirML",
      source_url: "https://github.com/RisGar/elixir_ml",
      docs: [
        main: "ElixirML",
        extras: ["README.md"],
        before_closing_body_tag: &before_closing_body_tag/1
      ]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:ex_doc, "~> 0.29.4", only: :dev, runtime: false},
      {:rustler, "~> 0.29.1"}
    ]
  end

  defp before_closing_body_tag(:html) do
    """
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.7/dist/katex.min.css" integrity="sha384-3UiQGuEI4TTMaFmGIZumfRPtfKQ3trwQE2JgosJxCnGmQpL/lJdjpcHkaaFwHlcI" crossorigin="anonymous">
    <script type="module">
    import renderMathInElement from "https://cdn.jsdelivr.net/npm/katex@0.16.7/dist/contrib/auto-render.mjs";
    renderMathInElement(document.body, {
      delimiters: [
        {left: '$$', right: '$$', display: true},
        {left: '$', right: '$', display: false},
      ],
      throwOnError : false
    });
    </script>
    """
  end

  defp before_closing_body_tag(_), do: ""
end
