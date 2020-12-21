//! This crate provides the
//! [`CalfVec`](https://docs.rs/calf-vec/latest/calf_vec/generic/struct.CalfVec.html)
//! data structure for small copy-on-write arrays.
//! As long as the data is not written to, it is only borrowed.
//! When owned, the data is stored on the stack as long as it is small enough.
//! Data is only moved on the heap as a last resort.
//! This is basically the intersection between
//! [`SmallVec`](https://crates.io/crates/smallvec) and
//! [`Cow`](https://doc.rust-lang.org/std/borrow/enum.Cow.html) (`Small` + `Cow` = `Calf`).
//! Additionally this crate provides a
//! [`CalfString`](https://docs.rs/calf-vec/latest/calf_vec/string/struct.CalfString.html)
//! for small copy-on-write strings
//! based on `CalfVec`.
//!
//! ## Basic usage
//!
//! A `CalfVec` either borrows or owns its data.
//! You can start by creating a `CalfVec` from a slice.
//! It will only be copied when the `CalfVec` is modified.
//! ```rust
//! use calf_vec::CalfVec;
//!
//! let slice = &[1, 2, 3];
//! let mut calf: CalfVec<'_, u8, 32> = CalfVec::borrowed(slice); // at this point, data is only borrowed.
//! calf[0]; // => 1
//! calf[0] = 4; // because it is modified, the data is copied here.
//! assert_eq!(calf, [4, 2, 3])
//! ```
//!
//! A `CalfVec` can also be directly created to own its data:
//! ```rust
//! # use calf_vec::CalfVec;
//! let mut owned: CalfVec<'_, u8, 32> = CalfVec::owned(vec![1, 2, 3]);
//! ```
//! Here, since the owned buffer's capacity is smaller than 32 (given as parameter),
//! it is stored on the stack.
//! It will be moved on the heap only when necessary:
//! ```rust
//! # use calf_vec::CalfVec;
//! # let mut owned: CalfVec<'_, u8, 32> = CalfVec::owned(vec![1, 2, 3]);
//! owned.push(4);
//! owned.push(5);
//! // ...
//! owned.push(31);
//! owned.push(32); // <- here the buffer's capacity now exceeds the given limit (32).
//!                 //    it is hence moved on the heap, transparently.
//! ```
#![feature(allocator_api)]
#![feature(int_bits_const)]
#![feature(maybe_uninit_extra)]
#![feature(min_const_generics)]
#![feature(shrink_to)]
#![feature(slice_partition_dedup)]
#![feature(slice_ptr_len)]
#![feature(specialization)]
#![feature(try_reserve)]
#![feature(untagged_unions)]
#![feature(vec_into_raw_parts)]

pub mod raw;
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
	CalfVec,
	CalfString
};
