use std::{
	alloc::{
		AllocRef,
		Global
	},
	marker::PhantomData,
	mem::ManuallyDrop,
	ptr,
	ops::{
		Deref,
		DerefMut
	},
	borrow::Cow,
	fmt,
	cmp
};

/// Metadata representing the length and capacity of the array.
///
/// This crate provides two implementation of this trait:
/// [`wide::Meta`](crate::wide::Meta) stores the length and capacity with two `usize`.
/// Then the maximum size/capacity depends on the bit-depth of the plateform.
/// For 64-bit plateforms, this crate also provides [`lean::Meta`](crate::lean::Meta) that stores both the length
/// and capacity on a single `usize`. As a result, the maximum size/capacity is [`std::u32::MAX`].
pub trait Meta: Copy {
	/// Maximum size/capacity of the array using this metadata format.
	const MAX_LENGTH: usize;

	/// Create a new metadata from an array's length and capacity (if any).
	fn new(len: usize, capacity: Option<usize>) -> Self;

	/// Get the length of the array.
	fn len(&self) -> usize;

	/// Get the capacity of the buffer, if any.
	///
	/// The capacity is only defined on owned buffers.
	fn capacity(&self) -> Option<usize>;

	/// Set the new length of the array.
	fn set_len(&mut self, len: usize);

	/// Set the new capacity of the buffer.
	fn set_capacity(&mut self, capacity: Option<usize>);
}

/// Inner data storage.
///
/// We use an union here since the actual type depends on the where the data is stored.
/// If the data is owned and on the stack, then the relevent field is `stack`.
/// If the data is borrowed or spilled, the the relevent field is `ptr`.
union Data<T, const N: usize> {
	/// Data stored on the stack.
	stack: ManuallyDrop<[T; N]>,

	/// Pointer to the data (either borrowed, or owned on the heap).
	ptr: *mut T
}

impl<T, const N: usize> Data<T, N> {
	#[inline]
	fn new() -> Data<T, N> {
		Data {
			ptr: ptr::null_mut()
		}
	}

	#[inline]
	unsafe fn drop_with<M: Meta, A: AllocRef>(&mut self, meta: M, alloc: &A) {
		match meta.capacity() {
			Some(capacity) => {
				let len = meta.len();
				if capacity <= N {
					// stacked
					ptr::drop_in_place(&mut (*self.stack)[0..len]);
				} else {
					// spilled
					Vec::from_raw_parts_in(self.ptr, len, capacity, alloc);
				}
			},
			None => ()
		}
	}
}

/// Contiguous growable array type that is either borrowed, stack allocated or heap allocated.
///
/// This type behaves just like a `Vec<T>` but with a few more optimizations.
/// Just like [`Cow`](std::borrow::Cow), the data can be simply borrowed as long as it is not accessed
/// mutably.
/// Otherwise just like [`SmallVec`](https://crates.io/crates/smallvec) the data is stored on the
/// stack as long as the buffer's capacity does not exceed a given capacity
/// (given as type parameter `N`).
/// If this capacity is exceeded, then the data is stored on the heap.
///
/// The maximum capacity of a `CalfVec<T>` array depends on the metadata format used
/// which is given as type parameter `M`, implementing the [`Meta`] trait.
/// By default the `wide::Meta` is used, which behaves just like `Vec`.
/// In this case, the maximum capacity is `std::usize::MAX`.
///
/// # Examples
///
/// ```
/// # use calf_vec::CalfVec;
/// let slice = &[1, 2, 3];
/// let mut calf: CalfVec<'_, u8, 32> = CalfVec::borrowed(slice); // at this point, data is only borrowed.
/// calf[0]; // => 1
/// calf[0] = 4; // because it is modified, the data is copied here.
/// assert_eq!(calf, [4, 2, 3])
/// ```
///
/// A `CalfVec` can also be directly created to own its data:
/// ```
/// # use calf_vec::CalfVec;
/// let owned: CalfVec<'_, u8, 32> = CalfVec::owned(vec![1, 2, 3]);
/// ```
pub struct CalfVec<'a, M: Meta, T, A: AllocRef, const N: usize> {
	/// Metadata storing the length and capacity of the array.
	meta: M,

	/// The actual data (or a pointer to the actual data).
	data: Data<T, N>,

	/// Allocator.
	alloc: A,

	/// Remembers the lifetime of the data if it is borrowed.
	lifetime: PhantomData<&'a T>
}

impl<'a, M: Meta, T, A: AllocRef, const N: usize> Drop for CalfVec<'a, M, T, A, N> {
	fn drop(&mut self) {
		unsafe {
			self.data.drop_with(self.meta, &self.alloc)
		}
	}
}

impl<'a, M: Meta, T, const N: usize> CalfVec<'a, M, T, Global, N> {
	/// Creates a new empty `CalfVec`.
	/// 
	/// The vector will not allocate until more than `N` elements are pushed onto it.
	// TODO make this function `const` as soon as the `const_fn` feature allows it.
	#[inline]
	pub fn new() -> CalfVec<'a, M, T, Global, N> {
		CalfVec {
			meta: M::new(0, Some(N)),
			data: Data::new(),
			alloc: Global,
			lifetime: PhantomData
		}
	}

	/// Creates a new empty `CalfVec` with a particular capacity.
	///
	/// The actual capacity of the created `CalfVec` will be at least `N`.
	#[inline]
	pub fn with_capacity(capacity: usize) -> CalfVec<'a, M, T, Global, N> {
		if capacity <= N {
			CalfVec::new()
		} else {
			let vec = Vec::with_capacity(capacity);
			let (ptr, _, actual_capacity) = vec.into_raw_parts();

			CalfVec {
				meta: M::new(0, Some(actual_capacity)),
				data: Data { ptr },
				alloc: Global,
				lifetime: PhantomData
			}
		}
	}

	/// Create a new `CalfVec` from borrowed data.
	///
	/// The input's data is not copied until it is accessed mutably.
	///
	/// # Example
	/// ```
	/// # use calf_vec::CalfVec;
	/// let slice = &[1, 2, 3];
	/// let mut calf: CalfVec<'_, u8, 32> = CalfVec::borrowed(slice); // at this point, data is only borrowed.
	/// calf[0]; // => 1
	/// calf[0] = 4; // because it is modified, the data is copied here.
	/// assert_eq!(calf, [4, 2, 3])
	/// ```
	#[inline]
	pub fn borrowed<B: AsRef<[T]> + ?Sized>(borrowed: &'a B) -> CalfVec<'a, M, T, Global, N> {
		let slice = borrowed.as_ref();

		CalfVec {
			meta: M::new(slice.len(), None),
			data: Data { ptr: slice.as_ptr() as *mut T },
			alloc: Global,
			lifetime: PhantomData
		}
	}
}

impl<'a, M: Meta, T, A: AllocRef, const N: usize> CalfVec<'a, M, T, A, N> {
	/// Constructs a new, empty `CalfVec<M, T, A, N>`.
	///
	/// The vector will not allocate until more than `N` elements are pushed onto it.
	///
	/// # Examples
	///
	/// ```
	/// #![feature(allocator_api)]
	///
	/// use std::alloc::System;
	///
	/// # #[allow(unused_mut)]
	/// let mut vec: Vec<i32, _> = Vec::new_in(System);
	/// ```
	#[inline]
	pub fn new_in(alloc: A) -> CalfVec<'a, M, T, A, N> {
		CalfVec {
			meta: M::new(0, Some(N)),
			data: Data::new(),
			alloc,
			lifetime: PhantomData
		}
	}

	/// Create a new `CalfVec` from owned data.
	///
	/// The input is consumed and stored either on the stack if it does not exceed the
	/// capacity parameter `N`, or on the heap otherwise.
	#[inline]
	pub fn owned<O: Into<Vec<T, A>>>(owned: O) -> CalfVec<'a, M, T, A, N> where A: Clone {
		let vec = owned.into();
		let alloc = vec.alloc_ref().clone();
		let (ptr, len, capacity) = vec.into_raw_parts();
		if capacity <= N {
			// put on stack
			unsafe {
				let mut data = Data { ptr: ptr::null_mut() };
				std::ptr::copy_nonoverlapping(ptr, (*data.stack).as_mut_ptr(), len);
				Vec::from_raw_parts(ptr, 0, capacity); // destroy the original vec without touching its content.

				CalfVec {
					meta: M::new(len, Some(N)),
					data,
					alloc,
					lifetime: PhantomData
				}
			}
		} else {
			// put on heap
			CalfVec {
				meta: M::new(len, Some(capacity)),
				data: Data { ptr },
				alloc,
				lifetime: PhantomData
			}
		}
	}

	/// Returns a reference to the underlying allocator.
	#[inline]
	pub fn alloc_ref(&self) -> &A {
		&self.alloc
	}

	/// Try to convert this `CalfVec` into a borrowed slice.
	///
	/// Returns `Ok(slice)` if the data is borrowed, and `Err(self)` otherwise.
	///
	/// This is a cost-free operation.
	#[inline]
	pub fn try_into_slice(self) -> Result<&'a [T], Self> {
		match self.capacity() {
			Some(_) => Err(self),
			None => unsafe {
				Ok(std::slice::from_raw_parts(self.as_ptr(), self.len()))
			}
		}
	}

	// /// Try to convert this `CalfVec` into a fixed size array.
	// ///
	// /// Returns `Ok(slice)` if the data is owned and stored on the stack,
	// /// and `Err(self)` otherwise.
	// ///
	// /// Note that some elements may be uninitialized if the `CalfVec` length is smaller than `N`.
	// #[inline]
	// pub fn try_into_array(self) -> Result<[std::mem::MaybeUninit<T>; N], Self> {
	// 	match self.capacity() {
	// 		Some(capacity) if capacity <= N => unsafe {
	// 			let mut data = Data {
	// 				ptr: ptr::null_mut()
	// 			};
	//
	// 			std::mem::swap(&mut data, &mut self.data);
	// 			std::mem::forget(self); // there is nothing left to drop in `self`, we can forget it.
	// 			Ok(std::mem::transmute(data.stack))
	// 		},
	// 		_ => Err(self)
	// 	}
	// }

	/// Try to convert this `CalfVec` into `Vec`.
	///
	/// Returns `Ok(vec)` if the data is owned and on the heap, and `Err(self)` otherwise.
	///
	/// This is a cost-free operation.
	#[inline]
	pub fn try_into_vec(self) -> Result<Vec<T, A>, Self> {
		match self.capacity() {
			Some(capacity) if capacity > N => unsafe {
				let alloc = ptr::read(&self.alloc); // this is safe because `self.alloc` is never used ever after.
				let ptr = self.data.ptr;
				let len = self.len();
				std::mem::forget(self); // there is nothing left to drop in `self`, we can forget it.
				Ok(Vec::from_raw_parts_in(ptr, len, capacity, alloc))
			},
			_ => Err(self)
		}
	}

	/// Convert this `CalfVec` into `Vec`.
	///
	/// If the data is borrowed it will be cloned.
	/// If the data is owned on the stack, it will be moved on the heap.
	/// If the data is owned on the heap, then this is a cost-free operation.
	#[inline]
	pub fn into_vec(mut self) -> Vec<T, A> where T: Clone {
		unsafe {
			let capacity = self.own();
			let len = self.len();
			let alloc = ptr::read(&self.alloc); // this is safe because `self.alloc` is never used ever after.
			let vec = if capacity <= N {
				let src = (*self.data.stack).as_mut_ptr();
				let mut vec = Vec::with_capacity_in(len, alloc);
				std::ptr::copy_nonoverlapping(src, vec.as_mut_ptr(), len);
				vec
			} else {
				let ptr = self.data.ptr;
				Vec::from_raw_parts_in(ptr, len, capacity, alloc)
			};
			std::mem::forget(self); // there is nothing left to drop in `self`, we can forget it.
			vec
		}
	}

	/// Returns a raw pointer to the vector's buffer.
	///
	/// The caller must ensure that the vector outlives the pointer this
	/// function returns, or else it will end up pointing to garbage.
	/// Modifying the vector may cause its buffer to be reallocated,
	/// which would also make any pointers to it invalid.
	///
	/// The caller must also ensure that the memory the pointer (non-transitively) points to
	/// is never written to (except inside an `UnsafeCell`) using this pointer or any pointer
	/// derived from it. If you need to mutate the contents of the slice, use [`as_mut_ptr`](#as_mut_ptr).
	#[inline]
	pub fn as_ptr(&self) -> *const T {
		unsafe {
			match self.capacity() {
				Some(capacity) => {
					if capacity <= N {
						(*self.data.stack).as_ptr()
					} else {
						self.data.ptr
					}
				},
				None => self.data.ptr
			}
		}
	}

	/// Extracts a slice containing the entire vector.
	///
	/// Equivalent to `&s[..]`.
	#[inline]
	pub fn as_slice(&self) -> &[T] {
		unsafe {
			std::slice::from_raw_parts(self.as_ptr(), self.len())
		}
	}

	/// Returns true if the data is owned, i.e. if `to_mut` would be a no-op.
	#[inline]
	pub fn is_owned(&self) -> bool {
		self.meta.capacity().is_some()
	}

	/// Returns true if the data is borrowed, i.e. if `to_mut` would require additional work.
	#[inline]
	pub fn is_borrowed(&self) -> bool {
		self.meta.capacity().is_none()
	}

	/// Returns the length of the array.
	#[inline]
	pub fn len(&self) -> usize {
		self.meta.len()
	}

	/// Returns the capacity of the owned buffer, or `None` if the data is only borrowed.
	#[inline]
	pub fn capacity(&self) -> Option<usize> {
		self.meta.capacity()
	}
}

impl<'a, M: Meta, T, A: AllocRef, const N: usize> CalfVec<'a, M, T, A, N> where T: Clone {
	#[inline]
	pub fn own(&mut self) -> usize {
		match self.capacity() {
			Some(capacity) => capacity,
			None => unsafe { // copy time!
				let len = self.len();
				let slice = std::slice::from_raw_parts(self.data.ptr, len);

				let capacity = if len <= N {
					// clone on stack
					&mut (*self.data.stack)[0..len].clone_from_slice(slice); // FIXME
					N
				} else {
					// clone on heap
					let (ptr, _, capacity) = self.as_slice().to_vec_in(self.alloc).into_raw_parts();
					self.data.ptr = ptr;
					capacity
				};

				self.meta.set_capacity(Some(capacity));
				capacity
			}
		}
	}

	/// Returns an unsafe mutable pointer to the vector's buffer.
	///
	/// The caller must ensure that the vector outlives the pointer this
	/// function returns, or else it will end up pointing to garbage.
	/// Modifying the vector may cause its buffer to be reallocated,
	/// which would also make any pointers to it invalid.
	#[inline]
	pub fn as_mut_ptr(&mut self) -> *mut T {
		let capacity = self.own();
		unsafe {
			if capacity <= N {
				(*self.data.stack).as_mut_ptr()
			} else {
				self.data.ptr
			}
		}
	}

	/// Extracts a mutable slice of the entire vector.
	///
	/// Equivalent to `&mut s[..]`.
	#[inline]
	pub fn as_mut_slice(&mut self) -> &mut [T] {
		self.own();
		unsafe {
			std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len())
		}
	}

	#[inline]
	pub(crate) unsafe fn set_len(&mut self, len: usize) {
		self.meta.set_len(len)
	}

	/// Shortens the vector, keeping the first `len` elements and dropping
	/// the rest.
	///
	/// If `len` is greater than the vector's current length, this has no
	/// effect.
	///
	/// The [`drain`] method can emulate `truncate`, but causes the excess
	/// elements to be returned instead of dropped.
	///
	/// Note that this method has no effect on the allocated capacity
	/// of the vector.
	#[inline]
	pub fn truncate(&mut self, len: usize) {
		self.own();
		unsafe {
			if len > self.len() {
				return;
			}

			let remaining_len = self.len() - len;
			let s = ptr::slice_from_raw_parts_mut(self.as_mut_ptr().add(len), remaining_len);
			self.set_len(len);
			ptr::drop_in_place(s);
		}
	}

	/// Reserves capacity for at least `additional` more elements to be inserted
	/// in the given `CalfVec<T>`. The collection may reserve more space to avoid
	/// frequent reallocations. After calling `reserve`, capacity will be
	/// greater than or equal to `self.len() + additional`. Does nothing if
	/// capacity is already sufficient.
	///
	/// # Panics
	///
	/// Panics if the new capacity exceeds `M::MAX_LENGTH` bytes.
	pub fn reserve(&mut self, additional: usize) {
		let capacity = self.own();
		unsafe {
			let mut vec = if capacity <= N {
				self.spill()
			} else {
				Vec::from_raw_parts_in(self.data.ptr, self.len(), capacity, self.alloc)
			};

			vec.reserve(additional);
			let (ptr, _, capacity) = vec.into_raw_parts();
			self.data.ptr = ptr;
			self.meta.set_capacity(Some(capacity));
		}
	}

	/// Reserves the minimum capacity for exactly `additional` more elements to
	/// be inserted in the given `Vec<T>`. After calling `reserve_exact`,
	/// capacity will be greater than or equal to `self.len() + additional`.
	/// Does nothing if the capacity is already sufficient.
	///
	/// Note that the allocator may give the collection more space than it
	/// requests. Therefore, capacity can not be relied upon to be precisely
	/// minimal. Prefer `reserve` if future insertions are expected.
	///
	/// # Panics
	///
	/// Panics if the new capacity overflows `usize`.
	pub fn reserve_exact(&mut self, additional: usize) {
		let capacity = self.own();
		unsafe {
			let mut vec = if capacity <= N {
				self.spill()
			} else {
				Vec::from_raw_parts_in(self.data.ptr, self.len(), capacity, self.alloc)
			};

			vec.reserve_exact(additional);
			let (ptr, _, capacity) = vec.into_raw_parts();
			self.data.ptr = ptr;
			self.meta.set_capacity(Some(capacity));
		}
	}

	/// Move the data on the stack.
	///
	/// The data must already be owned, and on the stack.
	/// 
	/// Returns a `Vec` holding the data.
	/// The returned `Vec` **must not be dropped**.
	#[inline]
	unsafe fn spill(&mut self) -> Vec<T, A> {
		let mut data = Data {
			ptr: ptr::null_mut()
		};

		std::mem::swap(&mut data, &mut self.data);

		let boxed_slice: Box<[T], A> = Box::new_in(ManuallyDrop::into_inner(data.stack), self.alloc);
		let mut vec = boxed_slice.into_vec();

		self.data.ptr = vec.as_mut_ptr();

		vec
	}

	/// Shrinks the capacity of the vector with a lower bound.
	///
	/// The capacity will remain at least as large as `N`, the length
	/// and the supplied value.
	///
	/// If the resulting capacity is equal to `N`, the data will be placed on the stack if it
	/// is not already.
	///
	/// This function has no effect if the data is borrowed.
	///
	/// # Panics
	///
	/// Panics if the current capacity is smaller than the supplied
	/// minimum capacity.
	pub fn shrink_to(&mut self, min_capacity: usize) where A: Clone {
		// FIXME remove the `Clone` bound in `A` by not using `Vec::from_raw_parts_in`.
		match self.capacity() {
			Some(capacity) => unsafe {
				assert!(capacity < min_capacity);
				let len = self.len();
				let new_capacity = cmp::max(len, min_capacity);

				if new_capacity != capacity {
					if new_capacity <= N {
						if capacity > N {
							// put back on the stack.
							let ptr = self.data.ptr;
							ptr::copy_nonoverlapping(ptr, (*self.data.stack).as_mut_ptr(), len);
							Vec::from_raw_parts_in(ptr, 0, capacity, self.alloc.clone()); // drop the vec without touching its content.
							self.meta.set_capacity(Some(N));
						}
					} else {
						let mut vec = Vec::from_raw_parts_in(self.data.ptr, len, capacity, self.alloc.clone());
						vec.shrink_to(new_capacity);
						let (ptr, _, actual_new_capacity) = vec.into_raw_parts();
						self.data.ptr = ptr;
						self.meta.set_capacity(Some(actual_new_capacity));
					}
				}
			},
			None => ()
		}
	}

	/// Shrinks the capacity of the vector as much as possible.
	///
	/// It will drop down as close as possible to the length but the allocator
	/// may still inform the vector that there is space for a few more elements.
	#[inline]
	pub fn shrink_to_fit(&mut self) where A: Clone {
		// FIXME remove the `Clone` bound in `A` by not using `Vec::from_raw_parts_in` in `shrink_to`.
		self.shrink_to(self.len());
	}

	/// Inserts an element at position `index` within the vector, shifting all
	/// elements after it to the right.
	///
	/// # Panics
	///
	/// Panics if `index > len`.
	pub fn insert(&mut self, index: usize, element: T) {
		let len = self.len();
		if index > len {
			panic!("insertion index (is {}) should be <= len (which is {})", index, len);
		}

		let capacity = self.own();

		// space for the new element
		if len == capacity {
			self.reserve(1);
		}

		unsafe {
			// infallible
			// The spot to put the new value
			{
				let p = self.as_mut_ptr().add(index);
				// Shift everything over to make space. (Duplicating the
				// `index`th element into two consecutive places.)
				ptr::copy(p, p.offset(1), len - index);
				// Write it in, overwriting the first copy of the `index`th
				// element.
				ptr::write(p, element);
			}
			self.set_len(len + 1);
		}
	}

	/// Removes and returns the element at position `index` within the vector,
	/// shifting all elements after it to the left.
	///
	/// # Panics
	///
	/// Panics if `index` is out of bounds.
	pub fn remove(&mut self, index: usize) -> T {
		let len = self.len();
		if index >= len {
			panic!("removal index (is {}) should be < len (is {})", index, len);
		}

		self.own();

		unsafe {
			// infallible
			let ret;
			{
				// the place we are taking from.
				let ptr = self.as_mut_ptr().add(index);
				// copy it out, unsafely having a copy of the value on
				// the stack and in the vector at the same time.
				ret = ptr::read(ptr);

				// Shift everything down to fill in that spot.
				ptr::copy(ptr.offset(1), ptr, len - index - 1);
			}
			self.set_len(len - 1);
			ret
		}
	}

	/// Moves all the elements of `other` into `Self`, leaving `other` empty.
	///
	/// # Panics
	///
	/// Panics if the number of elements in the vector overflows.
	#[inline]
	pub fn append(&mut self, other: &mut Vec<T>) {
		unsafe {
			self.append_elements(other.as_slice() as _);
			other.set_len(0);
		}
	}

	/// Appends elements to `Self` from other buffer.
	#[inline]
	unsafe fn append_elements(&mut self, other: *const [T]) {
		let count = (*other).len();
		self.reserve(count);
		let len = self.len();
		ptr::copy_nonoverlapping(other as *const T, self.as_mut_ptr().add(len), count);
		self.set_len(len + count);
	}

	/// Clones and appends all elements in a slice to the `Vec`.
	///
	/// Iterates over the slice `other`, clones each element, and then appends
	/// it to this `CalfVec`. The `other` vector is traversed in-order.
	///
	/// Note that this function is same as [`extend`](#extend) except that it is
	/// specialized to work with slices instead. If and when Rust gets
	/// specialization this function will likely be deprecated (but still
	/// available).
	#[inline]
	pub fn extend_from_slice(&mut self, other: &[T]) {
		self.extend(other.iter().cloned())
	}

	/// Clears the vector, removing all values.
	///
	/// Note that this method has no effect on the allocated capacity
	/// of the vector.
	#[inline]
	pub fn clear(&mut self) {
		self.truncate(0)
	}

	/// Appends an element to the back of a collection.
	///
	/// # Panics
	///
	/// Panics if the new capacity exceeds `M::MAX_LENGTH` bytes.
	#[inline]
	pub fn push(&mut self, value: T) {
		let capacity = self.own();

		unsafe {
			if self.len() == capacity {
				self.reserve(1);
			}

			let end = self.as_mut_ptr().add(self.len());
			ptr::write(end, value);
			self.set_len(self.len()+1);
		}
	}

	/// Removes the last element from a vector and returns it, or [`None`] if it
	/// is empty.
	#[inline]
	pub fn pop(&mut self) -> Option<T> {
		if self.len() == 0 {
			None
		} else {
			self.own();
			unsafe {
				self.set_len(self.len()-1);
				Some(ptr::read(self.as_ptr().add(self.len())))
			}
		}
	}

	/// Removes all but the first of consecutive elements in the vector satisfying a given equality
	/// relation.
	///
	/// The `same_bucket` function is passed references to two elements from the vector and
	/// must determine if the elements compare equal. The elements are passed in opposite order
	/// from their order in the slice, so if `same_bucket(a, b)` returns `true`, `a` is removed.
	///
	/// If the vector is sorted, this removes all duplicates.
	pub fn dedup_by<F>(&mut self, same_bucket: F) where F: FnMut(&mut T, &mut T) -> bool {
		self.own();
		let len = {
			let (dedup, _) = self.as_mut_slice().partition_dedup_by(same_bucket);
			dedup.len()
		};
		self.truncate(len);
	}

	/// Removes all but the first of consecutive elements in the vector that resolve to the same
	/// key.
	///
	/// If the vector is sorted, this removes all duplicates.
	#[inline]
	pub fn dedup_by_key<F, K>(&mut self, mut key: F) where F: FnMut(&mut T) -> K, K: PartialEq {
		self.dedup_by(|a, b| key(a) == key(b))
	}

	/// Removes consecutive repeated elements in the vector according to the
	/// [`PartialEq`] trait implementation.
	///
	/// If the vector is sorted, this removes all duplicates.
	#[inline]
	pub fn dedup(&mut self) where T: PartialEq {
		self.dedup_by(|a, b| a == b)
	}
}

impl<'a, M: Meta, T: Clone, A: AllocRef + Clone, const N: usize> Clone for CalfVec<'a, M, T, A, N> {
	fn clone(&self) -> CalfVec<'a, M, T, A, N> {
		CalfVec::owned(self)
	}
}

impl<'v, 'a, M: Meta, T: Clone, A: AllocRef + Clone, const N: usize> From<&'v CalfVec<'a, M, T, A, N>> for Vec<T, A> {
	fn from(vec: &'v CalfVec<'a, M, T, A, N>) -> Vec<T, A> {
		vec.to_vec_in(vec.alloc_ref().clone())
	}
}

impl<'a, M: Meta, T: Clone, A: AllocRef, const N: usize> From<CalfVec<'a, M, T, A, N>> for Vec<T, A> {
	fn from(vec: CalfVec<'a, M, T, A, N>) -> Vec<T, A> {
		vec.into_vec()
	}
}

unsafe impl<'a, M: Meta + Send, T: Sync, A: AllocRef + Send, const N: usize> Send for CalfVec<'a, M, T, A, N> {}
unsafe impl<'a, M: Meta + Sync, T: Sync, A: AllocRef, const N: usize> Sync for CalfVec<'a, M, T, A, N> {}

impl<'a, M: Meta, T, A: AllocRef, const N: usize> Deref for CalfVec<'a, M, T, A, N> {
	type Target = [T];

	#[inline]
	fn deref(&self) -> &[T] {
		self.as_slice()
	}
}

impl<'a, M: Meta, T, A: AllocRef, const N: usize> DerefMut for CalfVec<'a, M, T, A, N> where T: Clone {
	#[inline]
	fn deref_mut(&mut self) -> &mut [T] {
		self.as_mut_slice()
	}
}

impl<'v, 'a, M: Meta, T, A: AllocRef, const N: usize> IntoIterator for &'v CalfVec<'a, M, T, A, N> {
	type Item = &'v T;
	type IntoIter = std::slice::Iter<'v, T>;

	fn into_iter(self) -> Self::IntoIter {
		self.as_slice().into_iter()
	}
}

impl<'v, 'a, M: Meta, T, A: AllocRef, const N: usize> IntoIterator for &'v mut CalfVec<'a, M, T, A, N> where T: Clone {
	type Item = &'v mut T;
	type IntoIter = std::slice::IterMut<'v, T>;

	fn into_iter(self) -> Self::IntoIter {
		self.as_mut_slice().into_iter()
	}
}

pub union IntoIterData<T, A: AllocRef, const N: usize> {
	stack: ManuallyDrop<[T; N]>,
	vec: ManuallyDrop<std::vec::IntoIter<T, A>>
}

pub struct IntoIter<M: Meta, T, A: AllocRef, const N: usize> {
	meta: M,
	offset: usize,
	data: IntoIterData<T, A, N>
}

impl<M: Meta, T, A: AllocRef, const N: usize> Iterator for IntoIter<M, T, A, N> {
	type Item = T;

	fn next(&mut self) -> Option<T> {
		unsafe {
			let capacity = self.meta.capacity().unwrap();
			let item = if capacity <= N {
				let i = self.offset;
				if i < self.meta.len() {
					self.offset += 1;
					Some(ptr::read(self.data.stack.as_ptr().add(i)))
				} else {
					None
				}
			} else {
				(*self.data.vec).next()
			};

			item
		}
	}
}

impl<M: Meta, T, A: AllocRef, const N: usize> Drop for IntoIter<M, T, A, N> {
	fn drop(&mut self) {
		unsafe {
			let capacity = self.meta.capacity().unwrap();
			if capacity <= N {
				ptr::drop_in_place(&mut (*self.data.stack)[self.offset..self.meta.len()]); // only drop remaining elements.
			} else {
				ManuallyDrop::drop(&mut self.data.vec)
			}
		}
	}
}

impl<'a, M: Meta, T, A: AllocRef, const N: usize> IntoIterator for CalfVec<'a, M, T, A, N> where T: Clone {
	type Item = T;
	type IntoIter = IntoIter<M, T, A, N>;

	fn into_iter(mut self) -> Self::IntoIter {
		unsafe {
			let capacity = self.own();

			let alloc = ptr::read(&self.alloc); // this is safe because `self.alloc` is not used ever after.
			let meta = self.meta;
			let mut data = Data {
				ptr: ptr::null_mut()
			};
			std::mem::swap(&mut data, &mut self.data);
			std::mem::forget(self); // there is nothing left to drop in `self`, we can forget it.

			let into_iter_data = if capacity <= N {
				IntoIterData {
					stack: data.stack
				}
			} else {
				let vec = Vec::from_raw_parts_in(data.ptr, meta.len(), capacity, alloc);
				IntoIterData {
					vec: ManuallyDrop::new(vec.into_iter())
				}
			};

			IntoIter {
				meta,
				offset: 0,
				data: into_iter_data
			}
		}
	}
}

impl<'a, M: Meta, T, A: AllocRef, const N: usize> Extend<T> for CalfVec<'a, M, T, A, N> where T: Clone {
	#[inline]
	fn extend<I: IntoIterator<Item = T>>(&mut self, iterator: I) {
		let mut iterator = iterator.into_iter();
		while let Some(element) = iterator.next() {
			let len = self.len();
			if len == self.own() {
				let (lower, _) = iterator.size_hint();
				self.reserve(lower.saturating_add(1));
			}
			unsafe {
				ptr::write(self.as_mut_ptr().add(len), element);
				// NB can't overflow since we would have had to alloc the address space
				self.set_len(len + 1);
			}
		}
	}

	// #[inline]
	// fn extend_one(&mut self, item: T) {
	// 	self.push(item);
	// }
	//
	// #[inline]
	// fn extend_reserve(&mut self, additional: usize) {
	// 	self.reserve(additional);
	// }
}

impl<'a, M: Meta, T: fmt::Debug, A: AllocRef, const N: usize> fmt::Debug for CalfVec<'a, M, T, A, N> {
	#[inline]
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		fmt::Debug::fmt(&**self, f)
	}
}

impl<'a, M: Meta, T, A: AllocRef, const N: usize> AsRef<CalfVec<'a, M, T, A, N>> for CalfVec<'a, M, T, A, N> {
	#[inline]
	fn as_ref(&self) -> &CalfVec<'a, M, T, A, N> {
		self
	}
}

impl<'a, M: Meta, T, A: AllocRef, const N: usize> AsMut<CalfVec<'a, M, T, A, N>> for CalfVec<'a, M, T, A, N> {
	#[inline]
	fn as_mut(&mut self) -> &mut CalfVec<'a, M, T, A, N> {
		self
	}
}

impl<'a, M: Meta, T, A: AllocRef, const N: usize> AsRef<[T]> for CalfVec<'a, M, T, A, N> {
	#[inline]
	fn as_ref(&self) -> &[T] {
		self
	}
}

impl<'a, M: Meta, T, A: AllocRef, const N: usize> AsMut<[T]> for CalfVec<'a, M, T, A, N> where T: Clone {
	#[inline]
	fn as_mut(&mut self) -> &mut [T] {
		self
	}
}

impl<'a, M: Meta, T, A: AllocRef + Clone, const N: usize> From<Vec<T, A>> for CalfVec<'a, M, T, A, N> {
	#[inline]
	fn from(v: Vec<T, A>) -> CalfVec<'a, M, T, A, N> {
		CalfVec::owned(v)
	}
}

impl<'a, M: Meta, T, const N: usize> From<&'a [T]> for CalfVec<'a, M, T, Global, N> {
	#[inline]
	fn from(s: &'a [T]) -> CalfVec<'a, M, T, Global, N> {
		CalfVec::borrowed(s)
	}
}

impl<'a, M: Meta, T, const N: usize> From<Cow<'a, [T]>> for CalfVec<'a, M, T, Global, N> where T: Clone {
	#[inline]
	fn from(c: Cow<'a, [T]>) -> CalfVec<'a, M, T, Global, N> {
		match c {
			Cow::Borrowed(s) => s.into(),
			Cow::Owned(v) => v.into()
		}
	}
}

macro_rules! impl_slice_eq1 {
	([$($vars:tt)*] $lhs:ty, $rhs:ty $(where $ty:ty: $bound:ident)?) => {
		impl<$($vars)*> PartialEq<$rhs> for $lhs where T: PartialEq<U>, $($ty: $bound)? {
			#[inline]
			fn eq(&self, other: &$rhs) -> bool { self[..] == other[..] }
			#[inline]
			fn ne(&self, other: &$rhs) -> bool { self[..] != other[..] }
		}
	}
}

impl_slice_eq1! { ['a, 'b, T, U, O: Meta, P: Meta, A: AllocRef, B: AllocRef, const N: usize, const M: usize] CalfVec<'a, O, T, A, N>, CalfVec<'b, P, U, B, M> }
impl_slice_eq1! { ['a, T, U, M: Meta, A: AllocRef, B: AllocRef, const N: usize] CalfVec<'a, M, T, A, N>, Vec<U, B> }
impl_slice_eq1! { ['b, T, U, M: Meta, A: AllocRef, B: AllocRef, const N: usize] Vec<T, A>, CalfVec<'b, M, U, B, N> }
impl_slice_eq1! { ['a, T, U, M: Meta, A: AllocRef, const N: usize] CalfVec<'a, M, T, A, N>, &[U] }
impl_slice_eq1! { ['a, T, U, M: Meta, A: AllocRef, const N: usize] CalfVec<'a, M, T, A, N>, &mut [U] }
impl_slice_eq1! { ['b, T, U, M: Meta, A: AllocRef, const N: usize] &[T], CalfVec<'b, M, U, A, N> }
impl_slice_eq1! { ['b, T, U, M: Meta, A: AllocRef, const N: usize] &mut [T], CalfVec<'b, M, U, A, N> }
impl_slice_eq1! { ['a, T, U, M: Meta, A: AllocRef, const N: usize] CalfVec<'a, M, T, A, N>, Cow<'_, [U]> where U: Clone }
impl_slice_eq1! { ['b, T, U, M: Meta, A: AllocRef, const N: usize] Cow<'_, [T]>, CalfVec<'b, M, U, A, N> where T: Clone }
impl_slice_eq1! { ['a, T, U, M: Meta, A: AllocRef, const N: usize, const O: usize] CalfVec<'a, M, T, A, N>, [U; O] }
impl_slice_eq1! { ['a, T, U, M: Meta, A: AllocRef, const N: usize, const O: usize] CalfVec<'a, M, T, A, N>, &[U; O] }
impl_slice_eq1! { ['b, T, U, M: Meta, A: AllocRef, const N: usize, const O: usize] [T; O], CalfVec<'b, M, U, A, N> }
impl_slice_eq1! { ['b, T, U, M: Meta, A: AllocRef, const N: usize, const O: usize] &[T; O], CalfVec<'b, M, U, A, N> }

impl<'a, M: Meta, T: Eq, A: AllocRef, const N: usize> Eq for CalfVec<'a, M, T, A, N> {}

pub trait ToCalfVec {
	fn to_calf_vec<M: Meta, A: AllocRef, const N: usize>(s: &[Self], alloc: A) -> CalfVec<M, Self, A, N> where Self: Sized;
}

impl<T: Clone> ToCalfVec for T {
	#[inline]
	default fn to_calf_vec<M: Meta, A: AllocRef, const N: usize>(s: &[Self], alloc: A) -> CalfVec<M, Self, A, N> {
		// struct DropGuard<'a, T, A: AllocRef> {
		// 	vec: &'a mut Vec<T, A>,
		// 	num_init: usize,
		// }
		// impl<'a, T, A: AllocRef> Drop for DropGuard<'a, T, A> {
		// 	#[inline]
		// 	fn drop(&mut self) {
		// 		// SAFETY:
		// 		// items were marked initialized in the loop below
		// 		unsafe {
		// 			self.vec.set_len(self.num_init);
		// 		}
		// 	}
		// }
		// let mut vec = Vec::with_capacity_in(s.len(), alloc);
		// let mut guard = DropGuard { vec: &mut vec, num_init: 0 };
		// let slots = guard.vec.spare_capacity_mut();
		// // .take(slots.len()) is necessary for LLVM to remove bounds checks
		// // and has better codegen than zip.
		// for (i, b) in s.iter().enumerate().take(slots.len()) {
		// 	guard.num_init = i;
		// 	slots[i].write(b.clone());
		// }
		// core::mem::forget(guard);
		// // SAFETY:
		// // the vec was allocated and initialized above to at least this length.
		// unsafe {
		// 	vec.set_len(s.len());
		// }
		// vec
		panic!("TODO")
	}
}

impl<T: Copy> ToCalfVec for T {
	#[inline]
	fn to_calf_vec<M: Meta, A: AllocRef, const N: usize>(s: &[Self], alloc: A) -> CalfVec<M, Self, A, N> {
		// let mut v = Vec::with_capacity_in(s.len(), alloc);
		// // SAFETY:
		// // allocated above with the capacity of `s`, and initialize to `s.len()` in
		// // ptr::copy_to_non_overlapping below.
		// unsafe {
		// 	s.as_ptr().copy_to_nonoverlapping(v.as_mut_ptr(), s.len());
		// 	v.set_len(s.len());
		// }
		// v
		panic!("TODO")
	}
}