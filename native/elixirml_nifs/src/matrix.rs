use libc::c_void;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

#[derive(rustler::NifStruct, Debug)]
#[module = "ElixirML.Matrix"]
pub struct Matrix {
  pub rows: usize,
  pub cols: usize,
  pub nums: Vec<f64>,
}

fn sigmoidf(n: f64) -> f64 {
  1.0 / (1.0 + f64::exp(-n))
}

// fn sigmoid_primef(n: f64) -> f64 {
//   sigmoidf(n) * (1.0 - sigmoidf(n))
// }

const RELU_FACTOR: f64 = 0.01;

fn leaky_reluf(n: f64) -> f64 {
  if n > 0.0 {
    n
  } else {
    RELU_FACTOR * n
  }
}

// fn leaky_relu_primef(n: f64) -> f64 {
//   if n > 0.0 {
//     1.0
//   } else {
//     RELU_FACTOR
//   }
// }

pub fn random(rows: usize, cols: usize) -> Matrix {
  let mat_size = rows * cols;

  let mut mat = Matrix {
    rows,
    cols,
    nums: Vec::<f64>::with_capacity(mat_size),
  };

  let mut normal = StandardNormal.sample_iter(rand::thread_rng());

  for _ in 0..mat_size {
    mat.nums.push(normal.next().unwrap());
  }

  mat
}

pub fn fill(rows: usize, cols: usize, value: f64) -> Matrix {
  Matrix {
    rows,
    cols,
    nums: vec![value; rows * cols],
  }
}

pub fn fill_vals(rows: usize, cols: usize, vals: Vec<f64>) -> Matrix {
  assert!(vals.len() == rows * cols);
  Matrix {
    rows,
    cols,
    nums: vals,
  }
}

// TODO: Implement softmax
pub fn softmax(mut mat: Matrix) -> Matrix {
  for n in mat.nums.iter_mut() {
    *n = sigmoidf(*n);
  }

  mat
}

pub fn sig(mut mat: Matrix) -> Matrix {
  for n in mat.nums.iter_mut() {
    *n = sigmoidf(*n);
  }

  mat
}

pub fn relu(mut mat: Matrix) -> Matrix {
  for n in mat.nums.iter_mut() {
    *n = leaky_reluf(*n);
  }

  mat
}

pub fn sum(a: Matrix, b: Matrix) -> Matrix {
  assert!(a.rows == b.rows);
  assert!(a.cols == b.cols);
  let mat_size = a.rows * a.cols;

  let mut mat = Matrix {
    rows: a.rows,
    cols: b.cols,
    nums: Vec::<f64>::with_capacity(mat_size),
  };

  for i in 0..mat_size {
    mat.nums.push(a.nums[i] + b.nums[i]);
  }

  mat
}

pub fn prod(a: Matrix, b: Matrix) -> Matrix {
  assert!(a.cols == b.rows);
  let mat_size = a.rows * b.cols;

  let mut mat = Matrix {
    rows: a.rows,
    cols: b.cols,
    nums: vec![0.0; mat_size],
  };

  extern "C" {
    fn cblas_dgemm(
      Order: u32,
      TransA: u32,
      TransB: u32,
      M: i32,
      N: i32,
      K: i32,
      alpha: f64,
      A: *const c_void,
      lda: i32,
      B: *const c_void,
      ldb: i32,
      beta: f64,
      C: *mut c_void,
      ldc: i32,
    );
  }

  unsafe {
    cblas_dgemm(
      101,
      111,
      111,
      a.rows as i32,
      b.cols as i32,
      a.cols as i32,
      1.0,
      a.nums.as_ptr() as *const c_void,
      a.cols as i32, // stride
      b.nums.as_ptr() as *const c_void,
      b.cols as i32, // stride
      0.0,
      mat.nums.as_mut_ptr() as *mut c_void,
      mat.cols as i32, // stride
    );
  }

  mat
}

pub fn shuffle_rows(mut a: Matrix, mut b: Matrix) -> Vec<Matrix> {
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

  vec![a, b]
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
