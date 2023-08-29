use crate::{layer::Layer, matrix::Matrix};

#[derive(Debug)]
pub struct Network {
  pub layers: Vec<Layer>,
  pub loss: Loss,
  pub optimiser: Optimiser,
  pub weights: Vec<Matrix>,
  pub biases: Vec<Matrix>,
}

impl Network {
  pub fn new(layers: Vec<Layer>, loss: Loss, optimiser: Optimiser) -> Self {
    // Wir brauchen mindestens einen Input Layer, einen Activation Layer und einen Output Layer
    assert!(layers.len() > 2);

    // Wir filtern die Layer nach "echten" Layern, also nicht Activation Layern, um einen Vektor mit den größen
    // aller Layer zu erhalten
    let mut network_size: Vec<usize> = vec![];
    for layer in layers.iter() {
      match layer {
        Layer::Number(_, size) => network_size.push(*size),
        Layer::Activation(_) => (),
      }
    }

    let weights: Vec<Matrix> = network_size
      .iter()
      .zip(network_size.iter().skip(1))
      .map(|(rows, cols): (&usize, &usize)| Matrix::random(*rows, *cols))
      .collect();

    let biases: Vec<Matrix> = network_size
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
