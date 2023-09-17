use crate::{
  layer::{ActivationType, Layer, LayerType},
  matrix::{self, Matrix},
  mnist::load,
  network::{Loss, Network, Optimiser},
};

pub fn run() {
  let mut a = Matrix::fill_vector(
    4,
    4,
    vec![
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ],
  );

  let mut b = Matrix::fill_scalar(4, 4, 2.5);
  println!("Matrix a: {:#?}", a);
  println!("Matrix b: {:#?}", b);

  let c = Matrix::add(&a, &b);
  println!("Matrix c: {:#?}", c);

  let c_2 = matrix::metal::add(&a, &b);
  println!("Matrix c_2: {:#?}", c_2);

  let d = Matrix::multiply(&a, &b);
  println!("Matrix d: {:#?}", d);

  let e = a.clone().activate(ActivationType::Sigmoid);
  println!("Matrix e: {:#?}", e);

  Matrix::shuffle_rows(&mut a, &mut b);
  println!("Matrix a & b: {:#?}", (a, b));

  let mnist = load(&"./datasets/fashionmnist/fashion.bin".to_owned());

  let layers = vec![
    Layer(LayerType::Input, 12, None),
    Layer(LayerType::Dense, 4, Some(ActivationType::Relu)),
    Layer(LayerType::Dense, 2, Some(ActivationType::Sigmoid)),
  ]
  .into_boxed_slice();

  let network = Network::new(layers, Loss::CrossEntropy, Optimiser::AdamW);
  println!("Network: {:#?}", network);
}
