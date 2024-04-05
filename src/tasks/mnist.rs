use crate::{
  layer::{Layer, LayerType},
  mnist::Mnist,
  network::Network,
};

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
  let mnist = Mnist::load(&"./datasets/fashionmnist/fashion.bin".to_owned())?;

  const IMAGE_SIZE: usize = 28 * 28;
  const LABEL_SIZE: usize = 10;

  let layers = vec![
    Layer(LayerType::Input, IMAGE_SIZE),
    Layer(LayerType::Hidden, 128),
    Layer(LayerType::Hidden, 64),
    Layer(LayerType::Hidden, 32),
    Layer(LayerType::Output, LABEL_SIZE),
  ]
  .into_boxed_slice();

  let mut network = Network::new(layers, 0.1, 1e-4, 0.5);

  // let images = Matrix {
  //   rows: 1000,
  //   cols: IMAGE_SIZE,
  //   nums: mnist
  //     .train_images
  //     .nums
  //     .iter()
  //     .take(1000 * IMAGE_SIZE)
  //     .copied()
  //     .collect(),
  // };

  // let targets = Matrix {
  //   rows: 1000,
  //   cols: LABEL_SIZE,
  //   nums: mnist
  //     .train_labels
  //     .nums
  //     .iter()
  //     .take(1000 * LABEL_SIZE)
  //     .copied()
  //     .collect(),
  // };

  network.learn(&mnist.train_images, &mnist.train_labels, 10000, 1000, 10);
  network.validate(&mnist.test_images, &mnist.test_labels);
  // println!("Predictions: {:#?}", output);

  // network.save(&"./state/mnist.bin".to_owned())?;

  // mnist::print(
  //   &mnist[0].nums[3 * IMAGE_SIZE..4 * IMAGE_SIZE],
  //   &mnist[2].nums[3 * LABEL_SIZE..4 * LABEL_SIZE],
  //   &[
  //     "T-shirt/top",
  //     "Trouser",
  //     "Pullover",
  //     "Dress",
  //     "Coat",
  //     "Sandal",
  //     "Shirt",
  //     "Sneaker",
  //     "Bag",
  //     "Ankle boot",
  //   ],
  // );

  Ok(())
}
