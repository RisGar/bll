# ElixirML

Deep learning in Elixir using [Erlang NIFs](https://www.erlang.org/doc/tutorial/nif.html) through the [rustler](https://github.com/rusterlium/rustler) bindings in [Rust](https://www.rust-lang.org/).

Uses the MacOS BLAS implementation through [vecLib](https://developer.apple.com/documentation/accelerate/veclib) for processing.

## Requirements

- MacOS version supporting Accelerate.framework (10.3+)
- XCode Command Line Tools (`xcode-install`)
- Elixir
- Rust

## Installation

```console
$ git clone https://github.com/RisGar/elixir_ml
$ cd elixir_ml
$ ./datasets.sh
$ mix deps.get
```

## Examples

Run any of the provided mix tasks in the [mix tasks folder](lib/mix/tasks/) with `mix <task_name>`. The tasks can also be found through the `mix help` command.
