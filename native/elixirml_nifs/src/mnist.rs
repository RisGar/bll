use crate::matrix::Matrix;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::mem::size_of;

const TRAIN_SIZE: usize = 60000;
const TEST_SIZE: usize = 10000;

const IMAGE_SIZE: usize = 28 * 28;
const LABEL_SIZE: usize = 10;

const HEADER_ELEMENT_SIZE: u64 = size_of::<u32>() as u64; // How many bytes each header element takes up
const IMAGE_HEADER_SIZE: u64 = 4 * HEADER_ELEMENT_SIZE; // How much space the header elements take up for image files
const LABEL_HEADER_SIZE: u64 = 2 * HEADER_ELEMENT_SIZE; // How much space the header elements take up for label files

fn load_train_images() -> Matrix {
  let mat_size = TRAIN_SIZE * IMAGE_SIZE;
  let mut mat = Matrix {
    rows: TRAIN_SIZE,
    cols: IMAGE_SIZE,
    nums: Vec::<f64>::with_capacity(mat_size),
  };

  let mut data = File::open("datasets/fashionmnist/train-images-idx3-ubyte").unwrap();
  data.seek(SeekFrom::Start(IMAGE_HEADER_SIZE)).unwrap();

  let mut buffer = Vec::<u8>::new();
  data.read_to_end(&mut buffer).unwrap();

  for e in buffer.into_iter() {
    mat.nums.push(e as f64 / 255.0);
  }

  mat
}

fn load_test_images() -> Matrix {
  let mat_size = TEST_SIZE * IMAGE_SIZE;
  let mut mat = Matrix {
    rows: TEST_SIZE,
    cols: IMAGE_SIZE,
    nums: Vec::<f64>::with_capacity(mat_size),
  };

  let mut data = File::open("datasets/fashionmnist/t10k-images-idx3-ubyte").unwrap();
  data.seek(SeekFrom::Start(IMAGE_HEADER_SIZE)).unwrap();

  let mut buffer = Vec::<u8>::new();
  data.read_to_end(&mut buffer).unwrap();

  for e in buffer.into_iter() {
    mat.nums.push(e as f64 / 255.0);
  }

  mat
}

fn load_train_labels() -> Matrix {
  let mat_size = TRAIN_SIZE * LABEL_SIZE;
  let mut mat = Matrix {
    rows: TRAIN_SIZE,
    cols: LABEL_SIZE,
    nums: vec![0.0; mat_size],
  };

  let mut data = File::open("datasets/fashionmnist/train-labels-idx1-ubyte").unwrap();
  data.seek(SeekFrom::Start(LABEL_HEADER_SIZE)).unwrap();

  let mut buffer = Vec::<u8>::new();
  data.read_to_end(&mut buffer).unwrap();

  for (i, e) in buffer.into_iter().enumerate() {
    // mat.nums.push(buffer[i] as f64 / 255.0);
    mat.nums[i * 10 + e as usize] = 1.0;
  }

  mat
}

fn load_test_labels() -> Matrix {
  let mat_size = TEST_SIZE * LABEL_SIZE;
  let mut mat = Matrix {
    rows: TEST_SIZE,
    cols: LABEL_SIZE,
    nums: vec![0.0; mat_size],
  };

  let mut data = File::open("datasets/fashionmnist/t10k-labels-idx1-ubyte").unwrap();
  data.seek(SeekFrom::Start(LABEL_HEADER_SIZE)).unwrap();

  let mut buffer = Vec::<u8>::new();
  data.read_to_end(&mut buffer).unwrap();

  for (i, e) in buffer.into_iter().enumerate() {
    // mat.nums.push(buffer[i] as f64 / 255.0);
    mat.nums[i * 10 + e as usize] = 1.0;
  }

  mat
}

pub fn load() -> Vec<Matrix> {
  vec![
    load_train_images(),
    load_test_images(),
    load_train_labels(),
    load_test_labels(),
  ]
}
