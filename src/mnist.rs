use crate::matrix::Matrix;
use ciborium::{from_reader, into_writer};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom};
use std::mem::size_of;
use std::path::Path;

const LABEL_SIZE: usize = 10;

const HEADER_ELEMENT_SIZE: u64 = size_of::<u32>() as u64; // How many bytes each header element takes up
const IMAGE_HEADER_SIZE: u64 = 4 * HEADER_ELEMENT_SIZE; // How much space the header elements take up for image files
const LABEL_HEADER_SIZE: u64 = 2 * HEADER_ELEMENT_SIZE; // How much space the header elements take up for label files

fn load_train_images() -> Matrix {
  let mut data = File::open("datasets/fashionmnist/train-images-idx3-ubyte").unwrap();

  let mut header_buffer: Vec<u8> = vec![0; IMAGE_HEADER_SIZE as usize];
  data.read_exact(&mut header_buffer).unwrap();

  let magic_num = i32::from_be_bytes(header_buffer[0..4].try_into().unwrap());
  assert!(magic_num == 2051);

  let rows = i32::from_be_bytes(header_buffer[4..8].try_into().unwrap());
  let cols = i32::from_be_bytes(header_buffer[8..12].try_into().unwrap())
    * i32::from_be_bytes(header_buffer[12..16].try_into().unwrap());

  let mut mat = Matrix {
    rows: rows as usize,
    cols: cols as usize,
    nums: Vec::<f32>::with_capacity((rows * cols) as usize),
  };

  data.seek(SeekFrom::Start(IMAGE_HEADER_SIZE)).unwrap();

  let mut buffer = Vec::<u8>::new();
  data.read_to_end(&mut buffer).unwrap();

  for e in buffer.into_iter() {
    mat.nums.push(e as f32 / 255.0);
  }

  mat
}

fn load_test_images() -> Matrix {
  let mut data = File::open("datasets/fashionmnist/t10k-images-idx3-ubyte").unwrap();

  let mut header_buffer: Vec<u8> = vec![0; IMAGE_HEADER_SIZE as usize];
  data.read_exact(&mut header_buffer).unwrap();

  // Check that the magic number is correct
  let magic_num = i32::from_be_bytes(header_buffer[0..4].try_into().unwrap());
  assert!(magic_num == 2051);

  // Number of images
  let rows = i32::from_be_bytes(header_buffer[4..8].try_into().unwrap());
  // Both dimensions of the image
  let cols = i32::from_be_bytes(header_buffer[8..12].try_into().unwrap())
    * i32::from_be_bytes(header_buffer[12..16].try_into().unwrap());

  let mut mat = Matrix {
    rows: rows as usize,
    cols: cols as usize,
    nums: Vec::<f32>::with_capacity((rows * cols) as usize),
  };

  data.seek(SeekFrom::Start(IMAGE_HEADER_SIZE)).unwrap();

  let mut buffer = Vec::<u8>::new();
  data.read_to_end(&mut buffer).unwrap();

  for e in buffer.into_iter() {
    mat.nums.push(e as f32 / 255.0);
  }

  mat
}

fn load_train_labels() -> Matrix {
  let mut data = File::open("datasets/fashionmnist/train-labels-idx1-ubyte").unwrap();

  let mut header_buffer: Vec<u8> = vec![0; LABEL_HEADER_SIZE as usize];
  data.read_exact(&mut header_buffer).unwrap();

  // Check that the magic number is correct
  let magic_num = i32::from_be_bytes(header_buffer[0..4].try_into().unwrap());
  assert!(magic_num == 2049);

  // Number of labels
  let rows = i32::from_be_bytes(header_buffer[4..8].try_into().unwrap());

  let mut mat = Matrix {
    rows: rows as usize,
    cols: LABEL_SIZE,
    nums: vec![0.0; rows as usize * LABEL_SIZE],
  };

  data.seek(SeekFrom::Start(LABEL_HEADER_SIZE)).unwrap();

  let mut buffer = Vec::<u8>::new();
  data.read_to_end(&mut buffer).unwrap();

  for (i, e) in buffer.into_iter().enumerate() {
    mat.nums[i * LABEL_SIZE + e as usize] = 1.0;
  }

  mat
}

fn load_test_labels() -> Matrix {
  let mut data = File::open("datasets/fashionmnist/t10k-labels-idx1-ubyte").unwrap();

  let mut header_buffer: Vec<u8> = vec![0; LABEL_HEADER_SIZE as usize];
  data.read_exact(&mut header_buffer).unwrap();

  // Check that the magic number is correct
  let magic_num = i32::from_be_bytes(header_buffer[0..4].try_into().unwrap());
  assert!(magic_num == 2049);

  // Number of labels
  let rows = i32::from_be_bytes(header_buffer[4..8].try_into().unwrap());

  let mut mat = Matrix {
    rows: rows as usize,
    cols: LABEL_SIZE,
    nums: vec![0.0; rows as usize * LABEL_SIZE],
  };

  data.seek(SeekFrom::Start(LABEL_HEADER_SIZE)).unwrap();

  let mut buffer = Vec::<u8>::new();
  data.read_to_end(&mut buffer).unwrap();

  for (i, e) in buffer.into_iter().enumerate() {
    mat.nums[i * LABEL_SIZE + e as usize] = 1.0;
  }

  mat
}

fn save_to_file(matrices: &[Matrix; 4], path: &impl AsRef<Path>) {
  let file = File::create(path).unwrap();
  let writer = BufWriter::new(file);
  println!("Saving to file...");
  into_writer(matrices, writer).unwrap();
  // serialize_into(file, matrices).unwrap();
  println!("Saved to path {}", path.as_ref().display());
}

pub fn load(path: &impl AsRef<Path>) -> [Matrix; 4] {
  match File::open(path) {
    Ok(file) => {
      let reader = BufReader::new(file);
      from_reader(reader).unwrap()
    } // deserialize_from(&file).unwrap(),
    Err(_) => {
      println!("Loading from scratch...");
      let matrices = [
        load_train_images(),
        load_test_images(),
        load_train_labels(),
        load_test_labels(),
      ];
      save_to_file(&matrices, path);
      matrices
    }
  }
}
