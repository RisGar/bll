# BLL

Deep learning in Rust (mostly[^1]) without a ML library.

## Requirements

- MacOS version supporting Accelerate.framework (10.3+)
- XCode Command Line Tools (`xcode-install`)
- Rust

## Installation

```console
$ git clone https://github.com/RisGar/elixir_ml
$ cd bll
$ ./datasets.sh
$ cargo run --release minist
```

## Examples

The list of tasks can be found through the `cargo run -- --help` command.

[^1]: Uses the MacOS BLAS implementation through [vecLib](https://developer.apple.com/documentation/accelerate/veclib) for matrix multiplication.
