use crate::layer::ActivationType;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use std::ffi::{c_float, c_int};

pub mod metal;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Matrix {
  pub rows: usize,
  pub cols: usize,
  pub nums: Vec<f32>,
}

impl Matrix {
  pub fn random(rows: usize, cols: usize) -> Matrix {
    let mat_size = rows * cols;

    let mut mat = Matrix {
      rows,
      cols,
      nums: Vec::<f32>::with_capacity(mat_size),
    };

    let mut normal = StandardNormal.sample_iter(rand::thread_rng());

    for _ in 0..mat_size {
      mat.nums.push(normal.next().unwrap());
    }

    mat
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

  pub fn activate(mat: &mut Matrix, activation: &ActivationType) -> Matrix {
    let func: fn(f32) -> f32 = match activation {
      ActivationType::Relu => leaky_reluf,
      ActivationType::Sigmoid => sigmoidf,
      ActivationType::Softmax => todo!("implement softmax"),
    };

    for n in mat.nums.iter_mut() {
      *n = func(*n);
    }

    mat.to_owned()
  }

  pub fn add(a: &Matrix, b: &Matrix) -> Matrix {
    assert!(a.rows == b.rows);
    assert!(a.cols == b.cols);
    let mat_size = a.rows * a.cols;

    let mut mat = Matrix {
      rows: a.rows,
      cols: b.cols,
      nums: Vec::<f32>::with_capacity(mat_size),
    };

    for i in 0..mat_size {
      mat.nums.push(a.nums[i] + b.nums[i]);
    }

    mat
  }

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
    for i in 0..a.rows {
      let mut rng = rand::thread_rng();
      let j: usize = i + rng.gen::<usize>() % (a.rows - i);

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
        nums: Vec::from_iter(
          mat.nums[i * batch_size * mat.cols..(i + 1) * batch_size * mat.cols]
            .iter()
            .cloned(),
        ),
      };

      res.push(mat)
    }

    res
  }
}

fn sigmoidf(n: f32) -> f32 {
  1.0 / (1.0 + f32::exp(-n))
}

// fn sigmoid_primef(n: f32) -> f32 {
//   sigmoidf(n) * (1.0 - sigmoidf(n))
// }

// relu becomes unleaky when factor is 0.0
const RELU_FACTOR: f32 = 0.01;

fn leaky_reluf(n: f32) -> f32 {
  if n > 0.0 {
    n
  } else {
    RELU_FACTOR * n
  }
}

// fn leaky_relu_primef(n: f32) -> f32 {
//   if n > 0.0 {
//     1.0
//   } else {
//     RELU_FACTOR
//   }
// }
