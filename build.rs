fn main() {
  println!("cargo:rustc-link-arg=-DACCELERATE_NEW_LAPACK");
  // println!("cargo:rustc-link-arg=-DACCELERATE_LAPACK_ILP64");
  println!("cargo:rustc-link-lib=framework=Accelerate");
}
