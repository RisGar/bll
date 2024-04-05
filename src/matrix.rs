use crate::rowvector::RowVector;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::{
  ffi::{c_float, c_int},
  fmt,
};

#[derive(Serialize, Deserialize, Clone)]
pub struct Matrix {
  pub rows: usize,
  pub cols: usize,
  pub nums: Vec<f32>,
}

impl Matrix {
  // Creation

  pub fn placeholder() -> Matrix {
    Matrix {
      rows: 0,
      cols: 0,
      nums: vec![],
    }
  }

  pub fn new(rows: usize, cols: usize, vals: Vec<f32>) -> Matrix {
    assert_eq!(vals.len(), rows * cols);
    Matrix {
      rows,
      cols,
      nums: vals,
    }
  }

  pub fn random(rows: usize, cols: usize) -> Matrix {
    let normal = StandardNormal.sample_iter(rand::thread_rng());
    let mut mat = Matrix {
      rows,
      cols,
      nums: normal.take(rows * cols).collect(),
    };

    mat.scale(0.01);

    mat
  }

  pub fn fill_scalar(rows: usize, cols: usize, value: f32) -> Matrix {
    Matrix {
      rows,
      cols,
      nums: vec![value; rows * cols],
    }
  }

  // --- Unary functions ---
  /// Sum of each row in form of a row vector
  pub fn row_sum(&self) -> RowVector {
    RowVector {
      cols: self.rows,
      nums: self
        .nums
        .chunks(self.cols)
        .map(|c| c.iter().sum())
        .collect(),
    }
  }

  pub fn transpose(self) -> Matrix {
    let mut mat = Matrix {
      rows: self.cols,
      cols: self.rows,
      nums: vec![0.0; self.rows * self.cols],
    };

    for i in 0..mat.rows {
      for j in 0..mat.cols {
        mat.nums[i * mat.cols + j] = self.nums[j * self.cols + i];
      }
    }

    mat
  }

  #[allow(non_snake_case)]
  pub fn T(self) -> Matrix {
    self.transpose()
  }

  // --- Binary functions ---
  pub fn scale(&mut self, s: f32) {
    extern "C" {
      fn cblas_sscal(N: c_int, alpha: c_float, X: *mut c_float, incX: c_int);
    }
    unsafe {
      cblas_sscal(self.nums.len() as i32, s, self.nums.as_mut_ptr(), 1);
    }
  }

  /// Performs elementwise addition on two matrices
  pub fn add(&mut self, b: &Matrix) {
    assert_eq!(self.rows, b.rows);
    assert_eq!(self.cols, b.cols);

    self.nums = (0..self.nums.len())
      .map(|i| self.nums[i] + b.nums[i])
      .collect();
  }

  /// Performs elementwise subtraction on two matrices
  pub fn subtract(&mut self, b: &Matrix) {
    assert_eq!(self.rows, b.rows);
    assert_eq!(self.cols, b.cols);

    self.nums = (0..self.nums.len())
      .map(|i| self.nums[i] - b.nums[i])
      .collect();
  }

  /// Add the same row vector to all rows
  pub fn add_row_vector(&mut self, b: &RowVector) {
    assert!(self.cols == b.cols);

    for i in 0..self.rows {
      for j in 0..self.cols {
        self.nums[i * self.cols + j] += b.nums[j];
      }
    }
  }

  /// Hadamard product of two matrices
  pub fn hadamard(&mut self, b: &Matrix) {
    assert!(self.rows == b.rows);
    assert!(self.cols == b.cols);

    self.nums = (0..self.nums.len())
      .map(|i| self.nums[i] * b.nums[i])
      .collect();
  }

  pub fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    assert!(a.cols == b.rows);

    let mut mat = Matrix {
      rows: a.rows,
      cols: b.cols,
      nums: vec![0.0; a.rows * b.cols],
    };

    #[repr(C)]
    #[allow(dead_code)]
    pub enum CblasLayout {
      RowMajor = 101,
      ColMajor = 102,
    }

    #[repr(C)]
    #[allow(dead_code)]
    pub enum CblasTranspose {
      NoTrans = 111,
      Trans = 112,
      ConjTrans = 113,
    }

    extern "C" {
      fn cblas_sgemm(
        layout: CblasLayout,
        transa: CblasTranspose,
        transb: CblasTranspose,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        b: *const c_float,
        ldb: c_int,
        beta: c_float,
        c: *mut c_float,
        ldc: c_int,
      );
    }

    unsafe {
      cblas_sgemm(
        CblasLayout::RowMajor,
        CblasTranspose::NoTrans,
        CblasTranspose::NoTrans,
        a.rows as i32,
        b.cols as i32,
        a.cols as i32,
        1.0,
        a.nums.as_ptr(),
        a.cols as i32,
        b.nums.as_ptr(),
        b.cols as i32,
        0.0,
        mat.nums.as_mut_ptr(),
        mat.cols as i32,
      );
    }

    mat
  }

  /// Shuffle two matrices in the same order
  pub fn shuffle_rows(&mut self, b: &mut Matrix) {
    assert!(self.rows == b.rows);

    // Fisher-Yates shuffle
    for i in 0..self.rows {
      let mut rng = rand::thread_rng();
      let j: usize = i + rng.gen::<usize>() % (self.rows - i);

      // Swap rows, keep columns intact
      for k in 0..self.cols {
        self.nums.swap(i * self.cols + k, j * self.cols + k);
      }
      for k in 0..b.cols {
        b.nums.swap(i * b.cols + k, j * b.cols + k);
      }
    }
  }

  // --- Ternary functions ---
  pub fn clamp(&mut self, min: f32, max: f32) {
    for i in 0..self.nums.len() {
      self.nums[i] = self.nums[i].clamp(min, max);
    }
  }

  pub fn batch(&self, batch_size: usize, batch_amount: usize) -> Vec<Matrix> {
    let remainder: usize = self.rows % batch_size;

    let mut res = Vec::<Matrix>::with_capacity(batch_amount);

    for i in 0..batch_amount {
      let current_batch_size = if i == batch_amount - 1 && remainder != 0 {
        remainder
      } else {
        batch_size
      };

      let mat = Matrix {
        rows: current_batch_size,
        cols: self.cols,
        nums: self.nums
          [i * current_batch_size * self.cols..(i + 1) * current_batch_size * self.cols]
          .into(),
      };

      res.push(mat)
    }

    res
  }
}

impl fmt::Debug for Matrix {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let max_width = self
      .nums
      .iter()
      .map(|n| n.to_string().len())
      .max()
      .expect("Empty matrix");

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

    write!(f, "Matrix({}x{}):\n[{}]", self.rows, self.cols, numbers)?;
    Ok(())
  }
}
