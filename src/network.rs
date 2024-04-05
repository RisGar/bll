use std::{fs::File, io::BufWriter, path::Path};

use bincode::serialize_into;
use serde::{Deserialize, Serialize};

use crate::{
  layer::{Layer, LayerType},
  matrix::Matrix,
  rowvector::RowVector,
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

  // --- Feedforward ---
  pub weights: Vec<Matrix>,
  pub biases: Vec<RowVector>,

  // --- Backpropagation ---
  pub activation_inputs: Vec<Matrix>,
  pub layer_inputs: Vec<Matrix>,
  pub d_weights: Vec<Matrix>,
  pub d_biases: Vec<RowVector>,

  // --- Optimisation --
  // Learning rate
  pub learning_rate_initial: f32,
  pub learning_rate: f32,
  pub learning_rate_decay: f32,

  // Momentum
  pub momentum: f32,
  pub weight_momentum: Vec<Matrix>,
  pub bias_momentum: Vec<RowVector>,
}

impl Network {
  pub fn new(
    layers: Box<[Layer]>,
    learning_rate: f32,
    learning_rate_decay: f32,
    momentum: f32,
  ) -> Self {
    // Atleast an input and an output layer
    assert!(layers.len() >= 2);
    assert!(layers[0].0 == LayerType::Input);

    let layer_sizes: Vec<usize> = layers.iter().map(|layer| layer.size()).collect();

    // Weights with random values and with zeros
    let weights: Vec<Matrix> = layer_sizes
      .iter()
      .zip(layer_sizes.iter().skip(1))
      .map(|(&rows, &cols)| Matrix::random(rows, cols))
      .collect();
    let null_weights: Vec<Matrix> = layer_sizes
      .iter()
      .zip(layer_sizes.iter().skip(1))
      .map(|(&rows, &cols)| Matrix::fill_scalar(rows, cols, 0.0))
      .collect();

    // Biases with zeros
    let biases: Vec<RowVector> = layer_sizes
      .iter()
      .skip(1)
      .map(|&cols| RowVector::fill_scalar(cols, 0.0))
      .collect();

    let empty_inputs = vec![Matrix::placeholder(); layers.len() - 1];
    let empty_weights = vec![Matrix::placeholder(); weights.len()];
    let empty_biases = vec![RowVector::placeholder(); biases.len()];

    Self {
      layers,
      activation_inputs: empty_inputs.clone(),
      layer_inputs: empty_inputs,
      d_weights: empty_weights.clone(),
      weight_momentum: null_weights,
      d_biases: empty_biases.clone(),
      bias_momentum: biases.clone(),
      weights,
      biases,
      learning_rate,
      learning_rate_initial: learning_rate,
      learning_rate_decay,
      momentum,
    }
  }

  pub fn from_state(
    state: State,
    learning_rate: f32,
    learning_rate_decay: f32,
    momentum: f32,
  ) -> Self {
    // Atleast an input and an output layer
    assert!(state.layers.len() >= 2);
    assert!(state.layers[0].0 == LayerType::Input);

    let empty_inputs = vec![Matrix::placeholder(); state.layers.len() - 1];
    let empty_weights = vec![Matrix::placeholder(); state.weights.len()];
    let empty_biases = vec![RowVector::placeholder(); state.biases.len()];

    Self {
      layers: state.layers,
      activation_inputs: empty_inputs.clone(),
      layer_inputs: empty_inputs,
      d_weights: empty_weights.clone(),
      weight_momentum: empty_weights,
      d_biases: empty_biases.clone(),
      bias_momentum: empty_biases,
      weights: state.weights,
      biases: state.biases,
      learning_rate,
      learning_rate_initial: learning_rate,
      learning_rate_decay,
      momentum,
    }
  }

  // Fordward pass
  fn feedforward(&mut self, activations: &mut Matrix, targets: &Matrix) -> RowVector {
    for i in 0..self.layers.len() - 1 {
      self.layer_inputs[i] = activations.clone();

      // w*x+b
      *activations = Matrix::multiply(activations, &self.weights[i]);
      Matrix::add_row_vector(activations, &self.biases[i]);

      self.activation_inputs[i] = activations.clone();

      if let Some(loss) = self.layers[i + 1].activate(activations, targets) {
        return loss;
      }
    }

    unreachable!("Network should always have an output layer");
  }

  fn input_layer(&self) -> &Layer {
    self.layers.first().unwrap()
  }
  fn output_layer(&self) -> &Layer {
    self.layers.last().unwrap()
  }

  /// Backward pass with derivatives to descend gradient
  fn backpropagate(&mut self, predictions: &mut Matrix, targets: &Matrix) {
    for i in (0..self.layers.len() - 1).rev() {
      match self.layers[i + 1].layer_type() {
        LayerType::Output => {
          self.layers[i + 1].deactivate(predictions, targets);
        }
        LayerType::Hidden => {
          self.layers[i + 1].deactivate(predictions, &self.activation_inputs[i]);
        }
        LayerType::Input => unreachable!("Input layer cannot be (de)activated"),
      }

      self.d_weights[i] = Matrix::multiply(&self.layer_inputs[i].clone().T(), predictions);

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
        &mut self.weight_momentum[i],
        &mut self.bias_momentum[i],
        self.learning_rate,
        self.momentum,
      )
    }
  }

  // TODO: actual accuracy instead of loss
  fn accuracy(&self, predictions: &Matrix, targets: &Matrix) -> f32 {
    assert_eq!(predictions.rows, targets.rows);
    assert_eq!(predictions.cols, targets.cols);

    let mut max_predictions: Vec<usize> = Vec::with_capacity(predictions.rows);
    for row in 0..predictions.rows {
      let cur = &predictions.nums[row * predictions.cols..(row + 1) * predictions.cols];

      let index_of_max = cur
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index)
        .expect("Could not read target");

      max_predictions.push(index_of_max);
    }

    let mut target_indexes: Vec<usize> = Vec::with_capacity(targets.rows);
    for row in 0..targets.rows {
      let cur = &targets.nums[row * targets.cols..(row + 1) * targets.cols];

      let index_of_max = cur
        .iter()
        .enumerate()
        .find(|(_, x)| **x == 1.0)
        .map(|(index, _)| index)
        .expect("Could not read label");

      target_indexes.push(index_of_max);
    }

    println!("max predictions: {:?}", max_predictions);
    println!("target indexes: {:?}", target_indexes);

    let mut correct_predictions: usize = 0;
    for i in 0..predictions.rows {
      if max_predictions[i] == target_indexes[i] {
        correct_predictions += 1;
      }
    }

    println!("correct predictions: {}", correct_predictions);

    correct_predictions as f32 / predictions.rows as f32
  }

  pub fn learn(
    &mut self,
    activations: &Matrix,
    targets: &Matrix,
    epochs: usize,
    batch_size: usize,
    print_every: usize,
  ) {
    assert_eq!(activations.rows, targets.rows);
    assert_eq!(targets.cols, self.output_layer().size());
    assert_eq!(activations.cols, self.input_layer().size());

    // Ceiling division to allow for a remaining batch at the end
    let steps = activations.rows.div_ceil(batch_size);

    println!("steps: {}", steps);

    // Matrix::shuffle_rows(&mut activations, &mut targets);

    let activation_batches = activations.batch(batch_size, steps);
    let target_batches = targets.batch(batch_size, steps);

    for epoch in 0..epochs {
      println!("Epoch: {}", epoch + 1);
      for step in 0..steps {
        // Matrix of values passed through the network
        let mut predictions = activation_batches[step].clone();
        let targets = &target_batches[step];

        let loss = self.feedforward(&mut predictions, targets);
        let loss = loss.nums.iter().sum::<f32>() / loss.nums.len() as f32;

        // TODO calculate accuracy

        self.backpropagate(&mut predictions, targets);

        // learning rate decay
        self.learning_rate = (1.0
          / (1.0 + self.learning_rate_decay * (epoch * steps + step) as f32))
          * self.learning_rate_initial;

        self.optimise();

        if step % print_every == 0 {
          println!(
            "Step: {}, Loss: {}, Learning rate: {}",
            step + 1,
            loss,
            self.learning_rate
          );
        }
      }
    }
  }

  pub fn save(self, path: &impl AsRef<Path>) -> Result<(), bincode::Error> {
    let state = State {
      layers: self.layers,
      weights: self.weights,
      biases: self.biases,
    };

    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    println!("Saving to file...");

    serialize_into(writer, &state)?;

    println!("Saved to path {}", path.as_ref().display());

    Ok(())
  }

  pub fn validate(&mut self, activations: &Matrix, targets: &Matrix) {
    let mut predictions = activations.clone();

    let loss = self.feedforward(&mut predictions, targets);
    let loss = loss.nums.iter().sum::<f32>() / loss.nums.len() as f32;

    println!("predictions: {:#?}", predictions);
    println!("targets: {:#?}", targets);

    let accuracy = self.accuracy(&predictions, targets);

    println!("Accuracy: {}, Loss: {}", accuracy, loss);
  }
}
