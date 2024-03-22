use clap::{Parser, ValueEnum};

mod layer;
mod loss;
mod matrix;
mod mnist;
mod network;
mod optimiser;
mod tasks;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum Task {
  /// Fashion MNIST Dataset
  Mnist,
  /// Test Script
  Test,
}

/// Informatik BLL
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
  /// Task to execute
  #[arg(value_enum)]
  task: Task,
}

fn main() {
  let args = Args::parse();
  match args.task {
    Task::Mnist => tasks::mnist::run(),
    Task::Test => tasks::test::run(),
  }
}
