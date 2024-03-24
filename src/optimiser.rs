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
}

// TODO: momentum
