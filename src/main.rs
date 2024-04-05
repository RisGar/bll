use clap::{Parser, ValueEnum};

mod layer;
mod matrix;
mod mnist;
mod network;
mod rowvector;
mod tasks;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum Task {
  /// Fashion MNIST Dataset
  Mnist,
  /// XOR Network
  Xor,
  /// Test Script
  Test,
}

/// BLL
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
  /// Task to execute
  #[arg(value_enum)]
  task: Task,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let args = Args::parse();
  match args.task {
    Task::Mnist => tasks::mnist::run(),
    Task::Test => tasks::test::run(),
    Task::Xor => tasks::xor::run(),
  }
}
