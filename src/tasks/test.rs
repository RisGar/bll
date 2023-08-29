use crate::{
  layer::{ActivationType, Layer, LayerType},
  network::{Loss, Network, Optimiser},
};

pub fn run() {
  let layers = vec![
    Layer::Number(LayerType::Dense, 12),
    Layer::Activation(ActivationType::Relu),
    Layer::Number(LayerType::Dense, 4),
    Layer::Activation(ActivationType::Relu),
    Layer::Number(LayerType::Dense, 2),
    Layer::Activation(ActivationType::Relu),
  ];

  let network = Network::new(layers, Loss::CrossEntropy, Optimiser::AdamW);
  println!("Network: {:#?}", network);
}
