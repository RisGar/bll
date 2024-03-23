use crate::matrix::{Matrix, RowVector};

#[derive(Debug, Copy, Clone)]
pub enum Optimiser {
  Sgd,
  // AdaGrad,
  // RMSProp,
  // Adam,
}

// TODO Learning rate decay: Implement in SGD or in Network (Do I need it for Adam etc. too?)

impl Optimiser {
  pub fn optimise(
    self,
    weights: &mut Matrix,
    biases: &mut RowVector,
    d_weights: &Matrix,
    d_biases: &RowVector,
    learning_rate: f32,
  ) {
    match self {
      Optimiser::Sgd => sgd(
        weights,
        biases,
        d_weights.clone(),
        d_biases.clone(),
        learning_rate,
      ),
    }
  }
}

// TODO: momentum

fn sgd(
  weights: &mut Matrix,
  biases: &mut RowVector,
  mut d_weights: Matrix,
  mut d_biases: RowVector,
  learning_rate: f32,
) {
  Matrix::scale(&mut d_weights, -learning_rate);
  RowVector::scale(&mut d_biases, -learning_rate);
  Matrix::add(weights, &d_weights);
  RowVector::add(biases, &d_biases);
}
