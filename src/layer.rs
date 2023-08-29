#[derive(Debug)]
pub enum Layer {
  Number(LayerType, usize),
  Activation(ActivationType),
}

#[derive(PartialEq, Eq, Debug)]
pub enum LayerType {
  Dense,
}

#[derive(PartialEq, Eq, Debug)]
pub enum ActivationType {
  Relu,
  Sigmoid,
  Softmax,
}
