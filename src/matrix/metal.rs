use std::{
  ffi::{c_float, c_void},
  mem::size_of,
  path::PathBuf,
  slice,
};

use metal::{
  ComputePipelineDescriptor, ComputePipelineState, Device, MTLResourceOptions, MTLSize, NSUInteger,
};

use super::Matrix;

pub fn add(a: &Matrix, b: &Matrix) -> Matrix {
  assert!(a.rows == b.rows);
  assert!(a.cols == b.cols);
  let mat_size = a.rows * a.cols;

  let device = Device::system_default().unwrap();
  let command_queue = device.new_command_queue();
  let command_buffer = command_queue.new_command_buffer();
  let command_encoder = command_buffer.new_compute_command_encoder();

  let pipeline_state = create_pipeline_state(&device);
  command_encoder.set_compute_pipeline_state(&pipeline_state);

  let buffer_a = device.new_buffer_with_data(
    a.nums.as_ptr() as *const c_void,
    (a.nums.len() * size_of::<c_float>()) as NSUInteger,
    MTLResourceOptions::StorageModeShared,
  );
  let buffer_b = device.new_buffer_with_data(
    b.nums.as_ptr() as *const c_void,
    (b.nums.len() * size_of::<c_float>()) as NSUInteger,
    MTLResourceOptions::StorageModeShared,
  );
  let buffer_sum = device.new_buffer(
    (a.nums.len() * size_of::<c_float>()) as NSUInteger,
    MTLResourceOptions::StorageModeShared,
  );

  command_encoder.set_buffer(0, Some(&buffer_a), 0);
  command_encoder.set_buffer(1, Some(&buffer_b), 0);
  command_encoder.set_buffer(2, Some(&buffer_sum), 0);

  let num_threads = pipeline_state.thread_execution_width();

  let thread_group_count = MTLSize {
    width: ((mat_size as NSUInteger + num_threads) / num_threads),
    height: 1,
    depth: 1,
  };

  let thread_group_size = MTLSize {
    width: num_threads,
    height: 1,
    depth: 1,
  };

  command_encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
  command_encoder.end_encoding();

  command_buffer.commit();
  command_buffer.wait_until_completed();

  // mat.nums = unsafe { slice::from_raw_parts(*ptr, mat_size).to_vec() };

  Matrix {
    rows: a.rows,
    cols: b.cols,
    nums: unsafe {
      slice::from_raw_parts::<c_float>(buffer_sum.contents() as *const _, mat_size).to_vec()
    },
  }
}

fn create_pipeline_state(device: &Device) -> ComputePipelineState {
  let library_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/matrix/main.metallib");
  let library = device.new_library_with_file(library_path).unwrap();
  let kernel = library.get_function("add_matrices", None).unwrap();

  let pipeline_state_descriptor = ComputePipelineDescriptor::new();
  pipeline_state_descriptor.set_compute_function(Some(&kernel));

  device
    .new_compute_pipeline_state_with_function(pipeline_state_descriptor.compute_function().unwrap())
    .unwrap()
}
