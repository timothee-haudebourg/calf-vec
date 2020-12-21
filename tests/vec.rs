#![feature(allocator_api)]

use std::{
	ptr::NonNull,
	alloc::{
		Allocator,
		Global,
		AllocError,
		Layout
	},
	cell::Cell
};

const N: usize = 2;

#[test]
fn allocator_param() {
	use calf_vec::{
		wide::Meta,
		generic::CalfVec
	};

	// Writing a test of integration between third-party
	// allocators and `CalfVec` is a little tricky because the `CalfVec`
	// API does not expose fallible allocation methods, so we
	// cannot check what happens when allocator is exhausted
	// (beyond detecting a panic).
	//
	// Instead, this just checks that the `CalfVec` methods do at
	// least go through the Allocator API when it reserves
	// storage.

	// A dumb allocator that consumes a fixed amount of fuel
	// before allocation attempts start failing.
	struct BoundedAlloc {
		fuel: Cell<usize>,
	}
	unsafe impl Allocator for BoundedAlloc {
		fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
			let size = layout.size();
			if size > self.fuel.get() {
				return Err(AllocError);
			}
			match Global.allocate(layout) {
				ok @ Ok(_) => {
					self.fuel.set(self.fuel.get() - size);
					ok
				}
				err @ Err(_) => err,
			}
		}
		unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
			unsafe { Global.deallocate(ptr, layout) }
		}
	}

	let a = BoundedAlloc { fuel: Cell::new(500) };
	let mut v: CalfVec<Meta, u8, _, N> = CalfVec::with_capacity_in(50, a);
	v.resize(50, 0);
	assert_eq!(v.allocator().fuel.get(), 450);
	v.reserve(150); // (causes a realloc, thus using 50 + 150 = 200 units of fuel)
	assert_eq!(v.allocator().fuel.get(), 250);
}

#[test]
fn reserve_does_not_overallocate() {
	use calf_vec::CalfVec;

	{
		let mut v: CalfVec<u32, N> = CalfVec::new();
		// First, `reserve` allocates like `reserve_exact`.
		v.reserve(9);
		assert_eq!(Some(9), v.capacity());
	}

	{
		let mut v: CalfVec<u32, N> = CalfVec::new();
		v.reserve(7);
		assert_eq!(Some(7), v.capacity());
		// 97 is more than double of 7, so `reserve` should work
		// like `reserve_exact`.
		v.resize(7, 0);
		v.reserve(90);
		assert_eq!(Some(97), v.capacity());
	}

	{
		let mut v: CalfVec<u32, N> = CalfVec::new();
		v.reserve(12);
		assert_eq!(Some(12), v.capacity());
		v.resize(12, 0);
		v.reserve(3);
		// 3 is less than half of 12, so `reserve` must grow
		// exponentially. At the time of writing this test grow
		// factor is 2, so new capacity is 24, however, grow factor
		// of 1.5 is OK too. Hence `>= 18` in assert.
		assert!(v.capacity().unwrap() >= 12 + 12 / 2);
	}
}
