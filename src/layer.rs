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
