use crate::{
  layer::{softmax, ActivationType, Layer, LayerType},
  loss::Loss,
  matrix::Matrix,
  mnist,
  network::Network,
  optimiser::Optimiser,
};

pub fn run() {
  let mnist: [Matrix; 4] = mnist::load(&"./datasets/fashionmnist/fashion.bin".to_owned());

  const IMAGE_SIZE: usize = 28 * 28;
  const LABEL_SIZE: usize = 10;

  let layers = vec![
    Layer(LayerType::Input, IMAGE_SIZE, None),
    Layer(LayerType::Dense, 128, Some(ActivationType::Relu)),
    Layer(LayerType::Dense, 24, Some(ActivationType::Relu)),
    Layer(LayerType::Dense, LABEL_SIZE, Some(ActivationType::Softmax)),
  ]
  .into_boxed_slice();

  let mut network = Network::new(layers, Loss::CrossEntropy, Optimiser::Sgd, 1.0);

  let image_1 = Matrix {
    nums: mnist[0].nums[3 * IMAGE_SIZE..5 * IMAGE_SIZE].to_vec(),
    rows: 2,
    cols: IMAGE_SIZE,
  };
  let target_1 = Matrix {
    nums: mnist[2].nums[3 * LABEL_SIZE..5 * LABEL_SIZE].to_vec(),
    rows: 2,
    cols: LABEL_SIZE,
  };

  let output = network.feedforward(image_1.clone());
  println!("Predictions: {:#?}", output);

  network.learn(&image_1, &target_1, 1000);
  // println!("Predictions: {:#?}", output);

  mnist::print(
    &mnist[0].nums[3 * IMAGE_SIZE..4 * IMAGE_SIZE],
    &mnist[2].nums[3 * LABEL_SIZE..4 * LABEL_SIZE],
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
