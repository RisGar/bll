use crate::{
  layer::{Layer, LayerType},
  matrix::Matrix,
};

#[derive(Debug)]
pub struct Network {
  pub layers: Box<[Layer]>,
  pub loss: Loss,
  pub optimiser: Optimiser,
  pub weights: Vec<Matrix>,
  pub biases: Vec<Matrix>,
}

impl Network {
  pub fn new(layers: Box<[Layer]>, loss: Loss, optimiser: Optimiser) -> Self {
    // Wir brauchen mindestens einen Input Layer, einen Activation Layer und einen Output Layer
    assert!(layers.len() > 2);
    assert!(layers[0].0 == LayerType::Input);

    let layer_sizes: Vec<usize> = layers.iter().map(|layer| layer.size()).collect();

    let weights: Vec<Matrix> = layer_sizes
      .iter()
      .zip(layer_sizes.iter().skip(1))
      .map(|(rows, cols): (&usize, &usize)| Matrix::random(*rows, *cols))
      .collect();

    let biases: Vec<Matrix> = layer_sizes
      .iter()
      .skip(1)
      .map(|cols: &usize| Matrix::random(1, *cols))
      .collect();

    Self {
      layers,
      loss,
      optimiser,
      weights,
      biases,
    }
  }

  pub fn feedforward(&self, mut activations: Matrix) -> Matrix {
    for i in 0..self.layers.len() - 1 {
      activations = Matrix::multiply(&activations, &self.weights[i]);
      activations = Matrix::add(&activations, &self.biases[i]);

      if let Some(activation) = self.layers[i + 1].activation() {
        activations = activations.activate(*activation);
      }
    }

    activations
  }
}

#[derive(Debug)]
pub enum Loss {
  CrossEntropy,
  Quadratic,
}

#[derive(Debug)]
pub enum Optimiser {
  AdamW,
}
