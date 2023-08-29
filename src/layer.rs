#[derive(Debug)]
pub enum Layer {
  Number(NumberLayer),
  Activation(ActivationLayer),
}

#[derive(PartialEq, Eq, Debug)]
pub struct NumberLayer {
  pub layer_type: LayerType,
  pub size: usize,
}

#[derive(PartialEq, Eq, Debug)]
pub struct ActivationLayer {
  pub activation: ActivationType,
}

#[derive(PartialEq, Eq, Debug)]
pub enum LayerType {
  Dense,
  Input,
}

#[derive(PartialEq, Eq, Debug)]
pub enum ActivationType {
  Relu,
  Sigmoid,
  Softmax,
}
