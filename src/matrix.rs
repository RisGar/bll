use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::{
  ffi::{c_float, c_int},
  fmt,
};

pub mod metal;

#[derive(Serialize, Deserialize, Clone)]
pub struct Matrix {
  pub rows: usize,
  pub cols: usize,
  pub nums: Vec<f32>,
}

impl Matrix {
  // Creation
  pub fn random(rows: usize, cols: usize) -> Matrix {
    let normal = StandardNormal.sample_iter(rand::thread_rng());
    Matrix {
      rows,
      cols,
      nums: normal.take(rows * cols).collect(),
    }
  }

  pub fn fill_scalar(rows: usize, cols: usize, value: f32) -> Matrix {
    Matrix {
      rows,
      cols,
      nums: vec![value; rows * cols],
    }
  }

  pub fn fill_vector(rows: usize, cols: usize, vals: Vec<f32>) -> Matrix {
    assert!(vals.len() == rows * cols);
    Matrix {
      rows,
      cols,
      nums: vals,
    }
  }

  // Unary functions
  // pub fn activate(&mut self, activation: ActivationType) {
  //   activation.activate(&mut self.nums);
  // }

  pub fn scale(&mut self, s: f32) {
    for i in 0..self.nums.len() {
      self.nums[i] *= s;
    }
  }

  pub fn add(&mut self, b: &Matrix) {
    assert_eq!(self.rows, b.rows);
    assert_eq!(self.cols, b.cols);

    self.nums = (0..self.nums.len())
      .map(|i| self.nums[i] + b.nums[i])
      .collect();
  }
  pub fn subtract(&mut self, b: &Matrix) {
    assert!(self.rows == b.rows);
    assert!(self.cols == b.cols);

    self.nums = (0..self.nums.len())
      .map(|i| self.nums[i] - b.nums[i])
      .collect();
  }

  pub fn add_row_vector(&mut self, b: &RowVector) {
    // assert!(self.rows == b.rows);
    assert!(self.cols == b.cols);

    for i in 0..self.rows {
      for j in 0..self.cols {
        self.nums[i * self.cols + j] += b.nums[j];
      }
    }

    // self.nums = (0..self.rows * self.cols)
    //   .map(|i| self.nums[i] + b.nums[i])
    //   .collect();
  }

  /// Hadamard product of two matrices
  pub fn hadamard(&mut self, b: &Matrix) {
    assert!(self.rows == b.rows);
    assert!(self.cols == b.cols);

    self.nums = (0..self.nums.len())
      .map(|i| self.nums[i] * b.nums[i])
      .collect();
  }

  /// Sum of each row in form of a row vector
  pub fn row_sum(&self) -> RowVector {
    RowVector {
      cols: self.cols,
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

    for i in 0..self.rows {
      for j in 0..self.cols {
        mat.nums[j * self.cols + i] = self.nums[i * self.cols + j];
      }
    }

    mat
  }

  #[allow(non_snake_case)]
  pub fn T(self) -> Matrix {
    self.transpose()
  }

  // Binary functions
  // pub fn add2(a: &Matrix, b: &Matrix) -> Matrix {
  //   assert!(a.rows == b.rows);
  //   assert!(a.cols == b.cols);

  //   Matrix {
  //     rows: a.rows,
  //     cols: b.cols,
  //     nums: (0..a.rows * a.cols)
  //       .map(|i| a.nums[i] + b.nums[i])
  //       .collect(),
  //   }
  // }

  pub fn multiply(a: &Matrix, b: &Matrix) -> Matrix {
    assert!(a.cols == b.rows);
    let mat_size = a.rows * b.cols;

    let mut mat = Matrix {
      rows: a.rows,
      cols: b.cols,
      nums: vec![0.0; mat_size],
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
        a.cols as i32, // stride
        b.nums.as_ptr(),
        b.cols as i32, // stride
        0.0,
        mat.nums.as_mut_ptr(),
        mat.cols as i32, // stride
      );
    }

    mat
  }

  pub fn shuffle_rows(a: &mut Matrix, b: &mut Matrix) {
    assert!(a.rows == b.rows);

    // Fisher-Yates shuffle
    for i in 0..a.rows {
      let mut rng = rand::thread_rng();
      let j: usize = i + rng.gen::<usize>() % (a.rows - i);

      // unsafe { ptr::swap_nonoverlapping(a.nums[i * a.cols], a.nums[j * a.cols], count) }

      for k in 0..a.cols {
        a.nums.swap(i * a.cols + k, j * a.cols + k);
      }
      for k in 0..b.cols {
        b.nums.swap(i * b.cols + k, j * b.cols + k);
      }
    }
  }

  pub fn batch(mat: Matrix, batch_size: usize) -> Vec<Matrix> {
    assert!(mat.rows % batch_size == 0);
    let batch_amount: usize = mat.rows / batch_size;

    let mut res = Vec::<Matrix>::with_capacity(batch_amount);
    for i in 0..batch_amount {
      let mat = Matrix {
        rows: batch_size,
        cols: mat.cols,
        nums: mat.nums[i * batch_size * mat.cols..(i + 1) * batch_size * mat.cols].into(),
      };

      res.push(mat)
    }

    res
  }
}

impl fmt::Debug for Matrix {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let max_width = 3; //self.nums.iter().map(|n| n.to_string().len()).max().unwrap()));

    let mut numbers = String::new();

    for (i, e) in self.nums.iter().enumerate() {
      if (i + 1) % self.cols == 0 && (i + 1) != self.nums.len() {
        numbers += &format!(" {: >1$.3}\n ", e, max_width);
      } else if (i + 1) == self.nums.len() {
        numbers += &format!(" {: >1$.3} ", e, max_width);
      } else {
        numbers += &format!(" {: >1$.3}", e, max_width);
      }
    }

    write!(f, "Matrix({}x{}):\n[{}]", self.rows, self.cols, numbers)
  }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct RowVector {
  pub cols: usize,
  pub nums: Vec<f32>,
}

impl RowVector {
  pub fn random(cols: usize) -> RowVector {
    let normal = StandardNormal.sample_iter(rand::thread_rng());
    RowVector {
      cols,
      nums: normal.take(cols).collect(),
    }
  }

  pub fn add(&mut self, b: &RowVector) {
    assert!(self.cols == b.cols);

    self.nums = (0..self.cols).map(|i| self.nums[i] + b.nums[i]).collect();
  }

  pub fn scale(&mut self, s: f32) {
    for i in 0..self.nums.len() {
      self.nums[i] *= s;
    }
  }
}

impl fmt::Debug for RowVector {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let max_width = self.nums.iter().map(|n| n.to_string().len()).max().unwrap();

    let mut numbers = String::new();

    for (i, e) in self.nums.iter().enumerate() {
      if (i + 1) % self.cols == 0 && (i + 1) != self.nums.len() {
        numbers += &format!(" {: >1$}\n ", e, max_width);
      } else if (i + 1) == self.nums.len() {
        numbers += &format!(" {: >1$} ", e, max_width);
      } else {
        numbers += &format!(" {: >1$}", e, max_width);
      }
    }

    write!(f, "Matrix(1x{}):\n[{}]", self.cols, numbers)
  }
}
