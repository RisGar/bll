use crate::matrix::Matrix;

#[derive(Debug)]
pub enum Loss {
  CrossEntropy,
}

impl Loss {
  pub fn loss(&self, targets: Matrix, predictions: Matrix) -> f32 {
    assert_eq!(targets.rows, predictions.rows);
    assert_eq!(targets.cols, predictions.cols);

    match self {
      Loss::CrossEntropy => categorical_cross_entropy(targets.nums, predictions.nums),
    }
  }
  pub fn backwards(&self, predictions: &mut Matrix, targets: &Matrix) {
    assert_eq!(predictions.rows, targets.rows);
    assert_eq!(predictions.cols, targets.cols);

    match self {
      Loss::CrossEntropy => cross_entropy_backward(&mut predictions.nums, &targets.nums),
    }
  }
}

fn categorical_cross_entropy(targets: Vec<f32>, predictions: Vec<f32>) -> f32 {
  // TODO: fix function: loop should use LABELS not prediction length
  let mut loss = 0.0;
  for i in 0..predictions.len() {
    loss += targets[i] * predictions[i].ln();
  }
  -loss
}

fn cross_entropy_backward(predictions: &mut Vec<f32>, targets: &[f32]) {
  let len = predictions.len() as f32;
  for i in 0..predictions.len() {
    predictions[i] = -(targets[i] / predictions[i]);
    predictions[i] /= len;
  }
}
