use crate::matrix::Matrix;

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
  Softmax,
}

impl ActivationType {
  pub fn activate(self, n: &mut Matrix) {
    match self {
      ActivationType::Relu => leaky_relu(&mut n.nums),
      ActivationType::Softmax => softmax(n),
    }
  }
  // TODO
  pub fn backwards(self, n: &mut Matrix) {
    match self {
      ActivationType::Relu => leaky_relu_prime(&mut n.nums),
      ActivationType::Softmax => softmax_prime(&mut n.nums),
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

// Row-wise softmax
pub fn softmax(mat: &mut Matrix) {
  for i in 0..mat.rows {
    let mut n = mat.nums[i * mat.cols..(i + 1) * mat.cols].to_vec();
    // Subtract max value so that all values are <= 0 and the exp function doesn't overflow
    let max = n.iter().cloned().fold(0.0, f32::max);
    n.iter_mut().for_each(|e| *e -= max);

    n.iter_mut().for_each(|e| *e = f32::exp(*e));

    let m: f32 = n.iter().sum();
    n.iter_mut().for_each(|e| *e /= m);

    for (j, e) in n.iter().enumerate() {
      mat.nums[i * mat.cols + j] = *e;
    }
  }
}

fn softmax_prime(n: &mut [f32]) {
  todo!("softmax_prime");
}
