use std::error::Error;

use crate::matrix::Matrix;

pub fn run() -> Result<(), Box<dyn Error>> {
  let a = Matrix {
    rows: 17,
    cols: 1,
    nums: vec![
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
    ],
  };

  let z = a.batch(5, 4);
  println!("z: {:#?}", z);

  Ok(())
}
