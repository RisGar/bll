mod matrix;
mod mnist;

use matrix::Matrix;

#[rustler::nif]
fn mat_rand(rows: u32, cols: u32) -> Matrix {
  matrix::random(rows as usize, cols as usize)
}
#[rustler::nif]
fn mat_fill(rows: u32, cols: u32, value: f64) -> Matrix {
  matrix::fill(rows as usize, cols as usize, value)
}
#[rustler::nif]
fn mat_fill_vals(rows: u32, cols: u32, values: Vec<f64>) -> Matrix {
  matrix::fill_vals(rows as usize, cols as usize, values)
}
#[rustler::nif]
fn mat_softmax(mat: Matrix) -> Matrix {
  matrix::softmax(mat)
}
#[rustler::nif]
fn mat_sig(mat: Matrix) -> Matrix {
  matrix::sig(mat)
}
#[rustler::nif]
fn mat_relu(mat: Matrix) -> Matrix {
  matrix::relu(mat)
}
#[rustler::nif]
fn mat_sum(a: Matrix, b: Matrix) -> Matrix {
  matrix::sum(a, b)
}
#[rustler::nif]
fn mat_prod(a: Matrix, b: Matrix) -> Matrix {
  matrix::prod(a, b)
}
#[rustler::nif(schedule = "DirtyCpu")]
fn mat_shuffle_rows(a: Matrix, b: Matrix) -> Vec<Matrix> {
  matrix::shuffle_rows(a, b)
}
#[rustler::nif(schedule = "DirtyCpu")]
fn mat_batch(mat: Matrix, batch_size: usize) -> Vec<Matrix> {
  matrix::batch(mat, batch_size)
}
#[rustler::nif(schedule = "DirtyIo")]
fn mnist_load() -> Vec<Matrix> {
  mnist::load()
}

rustler::init!(
  "Elixir.ElixirML.NIFs",
  [
    mat_rand,
    mat_fill,
    mat_fill_vals,
    mat_softmax,
    mat_sig,
    mat_relu,
    mat_sum,
    mat_prod,
    mat_shuffle_rows,
    mat_batch,
    mnist_load
  ]
);
