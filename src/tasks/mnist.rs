use crate::{
  layer::{ActivationType, Layer, LayerType},
  matrix::Matrix,
  mnist,
  network::{Loss, Network, Optimiser},
};

pub fn run() {
  let mnist: [Matrix; 4] = mnist::load(&"./datasets/fashionmnist/fashion.bin".to_owned());

  const IMAGE_SIZE: usize = 28 * 28;

  let layers = vec![
    Layer(LayerType::Input, IMAGE_SIZE, None),
    Layer(LayerType::Dense, 128, Some(ActivationType::Relu)),
    Layer(LayerType::Dense, 24, Some(ActivationType::Relu)),
    Layer(LayerType::Dense, 10, Some(ActivationType::Sigmoid)),
  ]
  .into_boxed_slice();

  Network::new(layers, Loss::CrossEntropy, Optimiser::AdamW);
}
