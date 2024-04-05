use std::{
  ffi::{c_float, c_int},
  fmt,
  num::TryFromIntError,
};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct RowVector {
  pub cols: usize,
  pub nums: Vec<f32>,
}

impl RowVector {
  pub fn new(cols: usize, nums: Vec<f32>) -> RowVector {
    assert_eq!(nums.len(), cols);
    RowVector { cols, nums }
  }

  pub fn placeholder() -> RowVector {
    RowVector {
      cols: 0,
      nums: vec![],
    }
  }

  pub fn fill_scalar(cols: usize, value: f32) -> RowVector {
    RowVector {
      cols,
      nums: vec![value; cols],
    }
  }

  pub fn add(&mut self, b: &RowVector) {
    assert!(self.cols == b.cols);

    self.nums = (0..self.cols).map(|i| self.nums[i] + b.nums[i]).collect();
  }

  pub fn subtract(&mut self, b: &RowVector) {
    assert!(self.cols == b.cols);

    self.nums = (0..self.cols).map(|i| self.nums[i] - b.nums[i]).collect();
  }

  pub fn scale(&mut self, s: f32) {
    extern "C" {
      fn cblas_sscal(N: c_int, alpha: c_float, X: *mut c_float, incX: c_int);
    }
    unsafe {
      cblas_sscal(self.nums.len() as i32, s, self.nums.as_mut_ptr(), 1);
    }
  }
}

impl fmt::Debug for RowVector {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let max_width = self
      .nums
      .iter()
      .map(|n| n.to_string().len())
      .max()
      .expect("Empty row vector");

    let mut numbers = String::new();

    for (i, e) in self.nums.iter().enumerate() {
      if (i + 1) % self.cols == 0 && (i + 1) != self.nums.len() {
        numbers += &format!(" {: >1$.10}\n ", e, max_width);
      } else if (i + 1) == self.nums.len() {
        numbers += &format!(" {: >1$.10} ", e, max_width);
      } else {
        numbers += &format!(" {: >1$.10}", e, max_width);
      }
    }

    write!(f, "Matrix(1x{}):\n[{}]", self.cols, numbers)?;
    Ok(())
  }
}
