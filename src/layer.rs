#[derive(Debug)]
pub struct Layer(pub LayerType, pub usize, pub Option<ActivationType>);

#[derive(PartialEq, Eq, Debug)]
pub enum LayerType {
  Input,
  Dense,
}

#[derive(PartialEq, Eq, Debug)]
pub enum ActivationType {
  Relu,
  Sigmoid,
  Softmax,
}
