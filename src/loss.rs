#[derive(Debug)]
pub enum Loss {
  CrossEntropy,
  Quadratic,
}

impl Loss {
  pub fn loss(&self, targets: Vec<f32>, predictions: Vec<f32>) -> f32 {
    assert_eq!(targets.len(), predictions.len());

    match self {
      Loss::CrossEntropy => categorical_cross_entropy(targets, predictions),
      Loss::Quadratic => mean_squared_error(targets, predictions),
    }
  }
}

fn categorical_cross_entropy(targets: Vec<f32>, predictions: Vec<f32>) -> f32 {
  let mut loss = 0.0;
  for i in 0..targets.len() {
    loss += targets[i] * predictions[i].ln();
  }
  -loss
}

fn cross_entropy_backward(targets: Vec<f32>, predictions: Vec<f32>) -> Vec<f32> {
  let mut d_inputs = vec![0.0; targets.len()];
  let len = targets.len() as f32;
  for i in 0..targets.len() {
    d_inputs[i] = -(targets[i] / predictions[i]);
    d_inputs[i] /= len;
  }
  d_inputs
}

fn mean_squared_error(targets: Vec<f32>, predictions: Vec<f32>) -> f32 {
  let mut loss = 0.0;
  for i in 0..targets.len() {
    loss += (targets[i] - predictions[i]).powi(2);
  }
  loss / targets.len() as f32
}
