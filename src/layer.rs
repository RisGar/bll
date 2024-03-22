#[derive(Debug)]
pub struct Layer(pub LayerType, pub usize, pub Option<ActivationType>);

impl Layer {
  pub fn layer_type(&self) -> &LayerType {
    &self.0
  }
  pub fn size(&self) -> usize {
    self.1
  }
  pub fn activation(&self) -> &Option<ActivationType> {
    &self.2
  }
}

#[derive(PartialEq, Eq, Debug)]
pub enum LayerType {
  Input,
  Dense,
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum ActivationType {
  Relu,
  Sigmoid,
  Softmax,
}

impl ActivationType {
  pub fn activate(self, n: &mut [f32]) {
    match self {
      ActivationType::Relu => leaky_relu(n),
      ActivationType::Sigmoid => sigmoid(n),
      ActivationType::Softmax => softmax(n),
    }
  }
  // TODO
  pub fn derivative(self, n: &mut [f32]) {
    match self {
      ActivationType::Relu => leaky_relu(n),
      ActivationType::Sigmoid => sigmoid(n),
      ActivationType::Softmax => softmax(n),
    }
  }
}

fn sigmoid(n: &mut [f32]) {
  n.iter_mut().for_each(|e| {
    *e = 1.0 / (1.0 + f32::exp(-(*e)));
  })
}

// relu becomes unleaky when factor is 0.0
const RELU_FACTOR: f32 = 0.01;

fn leaky_relu(n: &mut [f32]) {
  n.iter_mut().for_each(|e| {
    if *e <= 0.0 {
      *e *= RELU_FACTOR;
    }
  })
}

fn leaky_relu_prime(n: &mut [f32]) {
  n.iter_mut().for_each(|e| {
    if *e <= 0.0 {
      *e = RELU_FACTOR;
    } else {
      *e = 1.0;
    }
  })
}

fn softmax(n: &mut [f32]) {
  n.iter_mut().for_each(|e| *e = f32::exp(*e));
  let m: f32 = n.iter().sum();
  n.iter_mut().for_each(|e| *e /= m);
}

const LINEAR_FACTOR: f32 = 1.0;

fn linear(n: &mut [f32]) {
  n.iter_mut().for_each(|e| *e *= LINEAR_FACTOR);
}

fn linear_prime(n: &mut [f32]) {
  n.iter_mut().for_each(|e| *e = LINEAR_FACTOR);
}

// fn sigmoid_primef(n: f32) -> f32 {
//   sigmoidf(n) * (1.0 - sigmoidf(n))
// }

// fn leaky_relu_primef(n: f32) -> f32 {
//   if n > 0.0 {
//     1.0
//   } else {
//     RELU_FACTOR
//   }
// }
