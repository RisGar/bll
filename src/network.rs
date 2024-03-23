use crate::{
  layer::{Layer, LayerType},
  loss::Loss,
  matrix::{Matrix, RowVector},
  optimiser::Optimiser,
};

#[derive(Debug)]
pub struct Network {
  pub layers: Box<[Layer]>,
  pub loss: Loss,
  pub optimiser: Optimiser,
  pub weights: Vec<Matrix>,
  pub biases: Vec<RowVector>,
  pub d_weights: Vec<Matrix>,
  pub d_biases: Vec<RowVector>,
  pub learning_rate: f32,
}

impl Network {
  pub fn new(layers: Box<[Layer]>, loss: Loss, optimiser: Optimiser, learning_rate: f32) -> Self {
    // Atleast an input and an output layer
    assert!(layers.len() >= 2);
    assert!(layers[0].0 == LayerType::Input);

    let layer_sizes: Vec<usize> = layers.iter().map(|layer| layer.size()).collect();

    let weights: Vec<Matrix> = layer_sizes
      .iter()
      .zip(layer_sizes.iter().skip(1))
      .map(|(&rows, &cols)| Matrix::random(rows, cols))
      .collect();

    let biases: Vec<RowVector> = layer_sizes
      .iter()
      .skip(1)
      .map(|&cols| RowVector::random(cols))
      .collect();

    Self {
      layers,
      loss,
      optimiser,
      d_weights: weights.clone(),
      d_biases: biases.clone(),
      weights,
      biases,
      learning_rate,
    }
  }

  // Fordward pass
  pub fn feedforward(&self, mut activations: Matrix) -> Matrix {
    for i in 0..self.layers.len() - 1 {
      // println!("Layer: {:#?}", self.layers[i]);
      // println!("Activation: {:#?}", activations);
      // println!("Weights: {:#?}", self.weights[i]);

      activations = Matrix::multiply(&activations, &self.weights[i]);
      Matrix::add_row_vector(&mut activations, &self.biases[i]);

      if let &Some(activation) = self.layers[i + 1].activation() {
        activation.activate(&mut activations);
      }
    }

    activations.clone()
  }

  // Backward pass with derivatives to descend gradient
  fn backpropagate(&self, predictions: &mut Matrix, targets: &Matrix) {
    self.loss.backwards(predictions, targets);

    for i in 0..self.layers.len() - 1 {
      if let &Some(activation) = self.layers[i + 1].activation() {
        activation.backwards(predictions);
      }

      // TODO backwards dense
    }
  }

  fn optimise(&mut self) {
    for i in 0..self.layers.len() - 1 {
      self.optimiser.optimise(
        &mut self.weights[i],
        &mut self.biases[i],
        &self.d_weights[i],
        &self.d_biases[i],
        self.learning_rate,
      )
    }
  }

  fn accuracy(&self, predictions: &Matrix, targets: &Matrix) {
    assert_eq!(predictions.rows, targets.rows);
    assert_eq!(predictions.cols, targets.cols);

    let prediction = predictions
      .nums
      .iter()
      .enumerate()
      .max_by(|(_, a), (_, b)| a.total_cmp(b))
      .map(|(index, _)| index);

    let target = targets
      .nums
      .iter()
      .enumerate()
      .max_by(|(_, a), (_, b)| a.total_cmp(b))
      .map(|(index, _)| index);
  }

  pub fn learn(&mut self, activations: &Matrix, targets: &Matrix, epochs: usize) {
    for epoch in 0..epochs {
      let activations = activations.clone();
      let mut predictions = self.feedforward(activations);

      // TODO calculate accuracy
      println!("Epoch: {}", epoch);

      self.backpropagate(&mut predictions, targets);

      self.optimise();
    }
  }

  pub fn validate(&self, activations: &Matrix, targets: &Matrix) {}
}
