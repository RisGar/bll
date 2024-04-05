use crate::matrix::Matrix;
use ansi_term::Colour::Fixed;
use ansi_term::Style;
use bincode::{deserialize_from, serialize_into};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom};
use std::mem::size_of;
use std::path::Path;

const LABEL_SIZE: usize = 10;

const HEADER_ELEMENT_SIZE: u64 = size_of::<u32>() as u64; // How many bytes each header element takes up
const IMAGE_HEADER_SIZE: u64 = 4 * HEADER_ELEMENT_SIZE; // How much space the header elements take up for image files
const LABEL_HEADER_SIZE: u64 = 2 * HEADER_ELEMENT_SIZE; // How much space the header elements take up for label files

fn load_image(path: &str) -> Matrix {
  let mut data = File::open(path).unwrap();

  let mut header_buffer: Vec<u8> = vec![0; IMAGE_HEADER_SIZE as usize];
  data.read_exact(&mut header_buffer).unwrap();

  // Check that the magic number is correct
  let magic_num = i32::from_be_bytes(header_buffer[0..4].try_into().unwrap());
  assert!(magic_num == 2051);

  data.seek(SeekFrom::Start(IMAGE_HEADER_SIZE)).unwrap();

  let mut buffer = Vec::<u8>::new();
  data.read_to_end(&mut buffer).unwrap();

  Matrix {
    // Number of images
    rows: i32::from_be_bytes(header_buffer[4..8].try_into().unwrap()) as usize,
    // Both dimensions of the image
    cols: i32::from_be_bytes(header_buffer[8..12].try_into().unwrap()) as usize
      * i32::from_be_bytes(header_buffer[12..16].try_into().unwrap()) as usize,
    nums: buffer.iter().map(|e| *e as f32 / 255.0).collect(),
  }
}

fn load_label(path: &str) -> Matrix {
  let mut data = File::open(path).unwrap();

  let mut header_buffer: Vec<u8> = vec![0; LABEL_HEADER_SIZE as usize];
  data.read_exact(&mut header_buffer).unwrap();

  // Check that the magic number is correct
  let magic_num = i32::from_be_bytes(header_buffer[0..4].try_into().unwrap());
  assert!(magic_num == 2049);

  data.seek(SeekFrom::Start(LABEL_HEADER_SIZE)).unwrap();

  let mut buffer = Vec::<u8>::new();
  data.read_to_end(&mut buffer).unwrap();

  // for (i, e) in buffer.into_iter().enumerate() {
  //   mat.nums[i * LABEL_SIZE + e as usize] = 1.0;
  // }

  Matrix {
    // Number of labels
    rows: i32::from_be_bytes(header_buffer[4..8].try_into().unwrap()) as usize,
    cols: LABEL_SIZE,
    nums: buffer.iter().flat_map(vectorise).collect(),
  }
}

fn vectorise(num: &u8) -> Vec<f32> {
  let mut vec = vec![0.0; 10];
  vec[*num as usize] = 1.0;
  vec
}

fn save_to_file(matrices: &Mnist, path: &impl AsRef<Path>) -> Result<(), bincode::Error> {
  let file = File::create(path).unwrap();
  let writer = BufWriter::new(file);

  println!("Saving to file...");
  serialize_into(writer, matrices)?;
  println!("Saved to path {}", path.as_ref().display());

  Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Mnist {
  pub train_images: Matrix,
  pub train_labels: Matrix,
  pub test_images: Matrix,
  pub test_labels: Matrix,
}

impl Mnist {
  pub fn load(path: &impl AsRef<Path>) -> bincode::Result<Mnist> {
    match File::open(path) {
      Ok(file) => {
        let reader = BufReader::new(file);

        deserialize_from(reader)
      }
      Err(_) => {
        println!("Loading from scratch...");

        let matrices = Mnist {
          train_images: load_image("datasets/fashionmnist/train-images-idx3-ubyte"),
          test_images: load_image("datasets/fashionmnist/t10k-images-idx3-ubyte"),
          train_labels: load_label("datasets/fashionmnist/train-labels-idx1-ubyte"),
          test_labels: load_label("datasets/fashionmnist/t10k-labels-idx1-ubyte"),
        };
        save_to_file(&matrices, path)?;
        Ok(matrices)
      }
    }
  }

  pub fn print(image: &[f32], label: &[f32], label_list: &[&str]) {
    assert_eq!(image.len(), 28 * 28);
    assert_eq!(label.len(), 10);
    assert_eq!(label_list.len(), 10);

    for (i, e) in image.iter().enumerate() {
      let style = Style::new().on(Fixed((e * 24.0) as u8 + 232));
      print!("{}", style.paint("  "));
      if (i + 1) % 28 == 0 {
        println!();
      }
    }

    println!(
      "{}",
      label_list[label
        .iter()
        .position(|&e| e == 1.0)
        .expect("Label is empty.")]
    );
  }
}
