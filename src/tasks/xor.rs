use crate::{
  layer::{Layer, LayerType},
  matrix::Matrix,
  network::Network,
};

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
  const IMAGE_SIZE: usize = 2;
  const LABEL_SIZE: usize = 2;

  let layers = vec![
    Layer(LayerType::Input, IMAGE_SIZE),
    Layer(LayerType::Hidden, 8),
    Layer(LayerType::Output, LABEL_SIZE),
  ]
  .into_boxed_slice();

  let input = Matrix {
    #[rustfmt::skip]
    //         A    B
    nums: vec![0.0, 0.0,
               0.0, 1.0,
               1.0, 0.0,
               1.0, 1.0],
    rows: 4,
    cols: 2,
  };
  let target = Matrix {
    #[rustfmt::skip]
    //         T    F
    nums: vec![0.0, 1.0,
               1.0, 0.0,
               1.0, 0.0,
               0.0, 1.0],
    rows: 4,
    cols: 2,
  };

  let mut network = Network::new(layers, 1.0, 1e-3, 0.5);
  // let mut network = Network::fixed_weights(layers, weights, 1.0);

  network.learn(&input, &target, 1000, 2, 1);
  network.validate(&input, &target);

  Ok(())
}
