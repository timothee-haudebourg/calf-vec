#![feature(min_const_generics)]
#![feature(untagged_unions)]
#![feature(vec_into_raw_parts)]

pub mod generic;
pub mod string;
pub mod wide;
#[cfg(target_pointer_width = "64")]
pub mod lean;
#[cfg(not(target_pointer_width = "64"))]
pub mod lean {
	/// Re-exports `wide` for non-64-bit targets
	pub use super::wide::*;
}

pub use wide::{
	SmallCowVec,
	SmallCowString
};
