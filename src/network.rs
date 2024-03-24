use std::{fs::File, io::BufWriter, path::Path};

use bincode::serialize_into;
use serde::{Deserialize, Serialize};

use crate::{
  layer::{Layer, LayerType},
  matrix::{Matrix, RowVector},
  optimiser::Optimiser,
};

#[derive(Debug, Serialize, Deserialize)]
pub struct State {
  pub layers: Box<[Layer]>,
  pub weights: Vec<Matrix>,
  pub biases: Vec<RowVector>,
}

#[derive(Debug)]
pub struct Network {
  pub layers: Box<[Layer]>,
  pub optimiser: Optimiser,
  pub inputs: Vec<Matrix>,
  pub weights: Vec<Matrix>,
  pub biases: Vec<RowVector>,
  pub d_weights: Vec<Matrix>,
  pub d_biases: Vec<RowVector>,
  pub learning_rate: f32,
}

impl Network {
  pub fn new(layers: Box<[Layer]>, optimiser: Optimiser, learning_rate: f32) -> Self {
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
      optimiser,
      inputs: Vec::with_capacity(layers.len() - 1),
      layers,
      d_weights: weights.clone(),
      d_biases: biases.clone(),
      weights,
      biases,
      learning_rate,
    }
  }

  // Fordward pass
  pub fn feedforward(&mut self, activations: &mut Matrix, targets: &Matrix) -> RowVector {
    for i in 0..self.layers.len() - 1 {
      self.inputs[i] = activations.clone();

      *activations = Matrix::multiply(activations, &self.weights[i]);
      Matrix::add_row_vector(activations, &self.biases[i]);

      if let Some(loss) = self.layers[i + 1].activate(activations, targets) {
        return loss;
      }
    }

    unreachable!("Network should always have an output layer");
  }

  // Backward pass with derivatives to descend gradient
  fn backpropagate(&mut self, predictions: &mut Matrix, targets: &Matrix) {
    for i in (0..self.layers.len() - 1).rev() {
      self.layers[i + 1].deactivate(predictions, targets);

      self.d_weights[i] = Matrix::multiply(&self.inputs[i].clone().T(), predictions);

      // Column sum
      self.d_biases[i] = predictions.clone().T().row_sum();

      // d_predictions
      *predictions = Matrix::multiply(predictions, &self.weights[i].clone().T());
    }
  }

  fn optimise(&mut self) {
    for i in 0..self.layers.len() - 1 {
      self.layers[i + 1].optimise(
        &mut self.weights[i],
        &mut self.biases[i],
        &self.d_weights[i],
        &self.d_biases[i],
        self.learning_rate,
      )
    }
  }

  // TODO: actual accuracy instead of loss
  fn accuracy(&self, predictions: &Matrix, targets: &Matrix) -> f32 {
    assert_eq!(predictions.rows, targets.rows);
    assert_eq!(predictions.cols, targets.cols);

    let mut temp_predictions = predictions.clone();

    temp_predictions.hadamard(targets);

    let output = predictions.row_sum();

    let mut accuracy: f32 = 0.0;
    for i in 0..output.cols {
      accuracy += output.nums[i];
    }
    accuracy /= output.cols as f32;

    accuracy
  }

  pub fn learn(&mut self, activations: &Matrix, targets: &Matrix, epochs: usize) {
    for epoch in 0..epochs {
      // Matrix of values passed through the network
      let mut predictions = activations.clone();
      let loss = self.feedforward(&mut predictions, targets);

      // TODO calculate accuracy
      println!("Epoch: {}, Loss: {:#?}", epoch, loss);

      self.backpropagate(&mut predictions, targets);

      // TODO: set d_weights and d_biases

      self.optimise();
    }
  }

  pub fn save(self, path: &impl AsRef<Path>) -> Result<(), bincode::Error> {
    let state = State {
      layers: self.layers,
      weights: self.weights,
      biases: self.biases,
    };

    let file = File::create(path).unwrap();
    let writer = BufWriter::new(file);
    println!("Saving to file...");

    serialize_into(writer, &state)?;

    // into_writer(matrices, writer).unwrap();
    // serialize_into(file, matrices).unwrap();
    println!("Saved to path {}", path.as_ref().display());

    Ok(())
  }

  pub fn validate(&self, activations: &Matrix, targets: &Matrix) {}
}
