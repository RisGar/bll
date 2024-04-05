use serde::{Deserialize, Serialize};

use crate::matrix::Matrix;
use crate::rowvector::RowVector;

#[derive(Debug, Serialize, Deserialize, Copy, Clone)]
pub struct Layer(pub LayerType, pub usize);

impl Layer {
  pub fn layer_type(&self) -> &LayerType {
    &self.0
  }
  pub fn size(&self) -> usize {
    self.1
  }

  pub fn activate(self, input: &mut Matrix, targets: &Matrix) -> Option<RowVector> {
    match self.layer_type() {
      LayerType::Input => unreachable!("Input layer cannot be activated"),
      LayerType::Hidden => {
        relu(&mut input.nums);
        None
      }
      LayerType::Output => Some(softmax_loss(input, targets)),
    }
  }
  pub fn deactivate(self, n: &mut Matrix, input_or_targets: &Matrix) {
    match self.layer_type() {
      LayerType::Input => unreachable!("Input layer cannot be activated"),
      LayerType::Hidden => relu_prime(n, input_or_targets),
      LayerType::Output => softmax_loss_prime(n, input_or_targets),
    }
  }

  #[allow(clippy::too_many_arguments)]
  pub fn optimise(
    self,
    weights: &mut Matrix,
    biases: &mut RowVector,
    d_weights: &Matrix,
    d_biases: &RowVector,
    weights_momentum: &mut Matrix,
    biases_momentum: &mut RowVector,
    learning_rate: f32,
    momentum: f32,
  ) {
    assert_eq!(weights.rows, d_weights.rows);
    assert_eq!(weights.cols, d_weights.cols);
    assert_eq!(biases.cols, d_biases.cols);

    sgd(
      weights,
      biases,
      d_weights.clone(),
      d_biases.clone(),
      weights_momentum,
      biases_momentum,
      learning_rate,
      momentum,
    )
  }
}

#[derive(PartialEq, Eq, Debug, Serialize, Deserialize, Clone, Copy)]
pub enum LayerType {
  Input,
  Hidden,
  Output,
}

// relu becomes unleaky when factor is 0.0

fn relu(n: &mut [f32]) {
  n.iter_mut().for_each(|e| {
    if *e <= 0.0 {
      *e *= 0.0;
    }
  })
}

fn relu_prime(mat: &mut Matrix, inputs: &Matrix) {
  // Zero where original input was negative or zero
  inputs
    .nums
    .iter()
    .enumerate()
    .filter(|(_, e)| **e <= 0.0)
    .for_each(|(i, _)| {
      mat.nums[i] = 0.0;
    })
}

/// Row-wise softmax
fn softmax(mat: &mut Matrix) {
  for i in 0..mat.rows {
    let mut n = mat.nums[i * mat.cols..(i + 1) * mat.cols].to_vec();
    // Subtract max value so that all values are <= 0 and the exp function doesn't overflow
    let max = n.iter().cloned().fold(0.0, f32::max);
    n.iter_mut().for_each(|e| *e -= max);

    n.iter_mut().for_each(|e| *e = f32::exp(*e));

    let m: f32 = n.iter().sum();
    n.iter_mut().for_each(|e| *e /= m);

    for (j, e) in n.iter().enumerate() {
      mat.nums[i * mat.cols + j] = *e;
    }
  }
}

/// Categorical cross-entropy over a batch of targets and predictions, returns a row vector containing each batches error
fn categorical_cross_entropy(predictions: &Matrix, targets: &Matrix) -> RowVector {
  let mut input = predictions.clone();

  // clamp predictions so that we don't ln(0)
  input.clamp(1e-7, 1.0 - 1e-7);

  input.hadamard(targets);

  let mut output = input.row_sum();

  output.nums.iter_mut().for_each(|e| *e = -e.ln());

  output
}

/// Softmax activation function, returns cateogorical cross entropy loss
fn softmax_loss(input: &mut Matrix, targets: &Matrix) -> RowVector {
  assert_eq!(targets.rows, input.rows);
  assert_eq!(targets.cols, input.cols);

  softmax(input);
  categorical_cross_entropy(input, targets)
}

fn softmax_loss_prime(input: &mut Matrix, targets: &Matrix) {
  input.subtract(targets);

  // Normalise gradient
  input.scale(1.0 / input.rows as f32);
}

/// Stochastic gradient descent with momentum
#[allow(clippy::too_many_arguments)]
fn sgd(
  weights: &mut Matrix,
  biases: &mut RowVector,
  mut d_weights: Matrix,
  mut d_biases: RowVector,
  weights_momentum: &mut Matrix,
  biases_momentum: &mut RowVector,
  learning_rate: f32,
  momentum: f32,
) {
  d_weights.scale(learning_rate);
  weights_momentum.scale(momentum);
  weights_momentum.subtract(&d_weights);
  weights.add(weights_momentum);

  d_biases.scale(learning_rate);
  biases_momentum.scale(momentum);
  biases_momentum.subtract(&d_biases);
  biases.add(biases_momentum);
}
