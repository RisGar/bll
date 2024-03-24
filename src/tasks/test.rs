use crate::matrix::{self, Matrix};

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
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

  Matrix::add(&mut a, &b);
  println!("Addition: {:#?}", a);

  let c_2 = matrix::metal::add(&a, &b);
  println!("Metal addition: {:#?}", c_2);

  let d = Matrix::multiply(&a, &b);
  println!("Multiplication {:#?}", d);

  let f = a.clone().T();
  println!("Transpose: {:#?}", f);

  Matrix::shuffle_rows(&mut a, &mut b);
  println!("Shuffle rows: {:#?}", (a, b));

  Ok(())
}
