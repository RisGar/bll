use crate::{
  layer::{ActivationType, Layer, LayerType},
  loss::Loss,
  matrix::Matrix,
  mnist,
  network::Network,
  optimiser::Optimiser,
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

  Network::new(layers, Loss::CrossEntropy, Optimiser::Adam);

  mnist::print(
    &mnist[0].nums[3 * 28 * 28..4 * 28 * 28],
    &mnist[2].nums[3 * 10..4 * 10],
    &[
      "T-shirt/top",
      "Trouser",
      "Pullover",
      "Dress",
      "Coat",
      "Sandal",
      "Shirt",
      "Sneaker",
      "Bag",
      "Ankle boot",
    ],
  )
}
