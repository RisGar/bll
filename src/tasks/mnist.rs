use crate::{
  layer::{Layer, LayerType},
  matrix::Matrix,
  mnist,
  network::Network,
  optimiser::Optimiser,
};

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
  let mnist: [Matrix; 4] = mnist::load(&"./datasets/fashionmnist/fashion.bin".to_owned());

  const IMAGE_SIZE: usize = 28 * 28;
  const LABEL_SIZE: usize = 10;

  let layers = vec![
    Layer(LayerType::Input, IMAGE_SIZE),
    Layer(LayerType::Hidden, 128),
    Layer(LayerType::Hidden, 24),
    Layer(LayerType::Output, LABEL_SIZE),
  ]
  .into_boxed_slice();

  let mut network = Network::new(layers, Optimiser::Sgd, 1.0);

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

  network.learn(&image_1, &target_1, 1000);
  // println!("Predictions: {:#?}", output);

  network.save(&"./datasets/fashionmnist/fashion.bin".to_owned())?;

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
  );

  Ok(())
}
