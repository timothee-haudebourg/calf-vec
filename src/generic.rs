use std::{
	marker::PhantomData,
	mem::ManuallyDrop,
	ptr,
	ops::{
		Deref,
		DerefMut
	}
};

/// Metadata representing the length and capacity of the array.
///
/// This crate provides two implementation of this trait:
/// [`wide::Meta`] stores the length and capacity with two `usize`.
/// Then the maximum size/capacity depends on the bit-depth of the plateform.
/// For 64-bit plateforms, this crate also provides [`lean::Meta`] that stores both the length
/// and capacity on a single `usize`. As a result, the maximum size/capacity is [`std::u32::MAX`].
pub trait Meta {
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

	/// Pointer to the data (aither borrowed, or owned on the heap).
	ptr: *mut T
}

/// Contiguous growable array type that is either borrowed, stack allocated or heap allocated.
///
/// This type behaves just like a `Vec<T>` but with a few more optimizations.
/// Just like [`std::borrow::Cow`], the data can be simply borrowed as long as it is not accessed
/// mutably.
/// Otherwise just like [`SmallVec`](https://crates.io/crates/smallvec) the data is stored on the
/// stack as long as the buffer's capacity does not exceed a given capacity
/// (given as type parameter `N`).
/// If this capacity is exceeded, then the data is stored on the heap.
///
/// The maximum capacity of a `CalfVec<T>` array depends on the metadata format used
/// which is given as type parameter `M`, implementing the [`Meta`] trait.
/// By default the `wide::Meta` is used, which behaves just `Vec`.
/// In this case, the maximum capacity is `std::usize::MAX`.
///
/// # Examples
///
/// ```
/// # use calf_vec::CalfVec;
/// let slice = &[1, 2, 3];
/// let mut calf = CalfVec::borrowed(slice); // at this point, data is only borrowed.
/// calf[0]; // -> 1
///
/// calf[0] = 4; // because it is modified, the data is first copied.
/// println!("{:?}", calf); // prints "[4, 2, 3]"
/// ```
///
/// A `CalfVec` can also be directly created to own its data:
/// ```
/// # use calf_vec::CalfVec;
/// let owned = CalfVec::owned(vec![1, 2, 3]);
/// ```
pub struct CalfVec<'a, M: Meta, T, const N: usize> {
	/// Metadata storing the length and capacity of the array.
	meta: M,

	/// The actual data (or a pointer to the actual data).
	data: Data<T, N>,

	/// Remembers the lifetime of the data if it is borrowed.
	lifetime: PhantomData<&'a T>
}

impl<'a, M: Meta, T, const N: usize> Drop for CalfVec<'a, M, T, N> {
	fn drop(&mut self) {
		match self.capacity() {
			Some(capacity) => unsafe {
				let len = self.len();
				if capacity <= N {
					// stacked
					ptr::drop_in_place(&mut (*self.data.stack)[0..len]);
				} else {
					// spilled
					Vec::from_raw_parts(self.data.ptr, len, capacity);
				}
			},
			None => ()
		}
	}
}

impl<'a, M: Meta, T, const N: usize> CalfVec<'a, M, T, N> {
	/// Create a new `CalfVec` from borrowed data.
	///
	/// The input's data is not copied until it is accessed mutably.
	///
	/// # Example
	/// ```
	/// # use calf_vec::CalfVec;
	/// let slice = &[1, 2, 3];
	/// let mut calf = CalfVec::borrowed(slice); // at this point, data is only borrowed.
	/// calf[0]; // -> 1
	///
	/// calf[0] = 4; // because it is modified, the data is first copied.
	/// println!("{:?}", calf); // prints "[4, 2, 3]"
	/// ```
	#[inline]
	pub fn borrowed<B: AsRef<[T]>>(borrowed: &'a B) -> CalfVec<'a, M, T, N> {
		let slice = borrowed.as_ref();

		CalfVec {
			meta: M::new(slice.len(), None),
			data: Data { ptr: slice.as_ptr() as *mut T },
			lifetime: PhantomData
		}
	}

	/// Create a new `CalfVec` from owned data.
	///
	/// The input is consumed and stored either on the stack if it does not exceed the
	/// capacity parameter `N`, or on the heap otherwise.
	#[inline]
	pub fn owned<O: Into<Vec<T>>>(owned: O) -> CalfVec<'a, M, T, N> {
		let vec = owned.into();
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
					lifetime: PhantomData
				}
			}
		} else {
			// put on heap
			CalfVec {
				meta: M::new(len, Some(capacity)),
				data: Data { ptr },
				lifetime: PhantomData
			}
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
	/// derived from it. If you need to mutate the contents of the slice, use [`as_mut_ptr`].
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

impl<'a, M: Meta, T, const N: usize> CalfVec<'a, M, T, N> where T: Clone {
	/// Acquires a mutable reference to the owned form of the data.
	///
	/// Clones the data if it is not already owned.
	#[inline]
	pub fn to_mut<'v>(&'v mut self) -> CalfVecMut<'v, 'a, M, T, N> {
		if self.is_borrowed() {
			// copy time!
			unsafe {
				let len = self.len();
				let slice = std::slice::from_raw_parts(self.data.ptr, len);
				if len <= N {
					// clone on stack
					&mut (*self.data.stack)[0..len].clone_from_slice(slice);
					self.meta.set_capacity(Some(N));
				} else {
					// clone on heap
					let (ptr, _, capacity) = self.as_slice().to_vec().into_raw_parts();
					self.data.ptr = ptr;
					self.meta.set_capacity(Some(capacity));
				}
			}
		}

		CalfVecMut {
			vec: self
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
		self.to_mut().as_mut_ptr()
	}

	/// Extracts a mutable slice of the entire vector.
	///
	/// Equivalent to `&mut s[..]`.
	#[inline]
	pub fn as_mut_slice(&mut self) -> &mut [T] {
		self.to_mut().into_mut_slice()
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
		self.to_mut().truncate(len)
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
	#[inline]
	pub fn reserve(&mut self, additional: usize) {
		self.to_mut().reserve(additional)
	}

	/// Moves all the elements of `other` into `Self`, leaving `other` empty.
	///
	/// # Panics
	///
	/// Panics if the number of elements in the vector overflows.
	#[inline]
	pub fn append(&mut self, other: &mut Vec<T>) {
		self.to_mut().append(other)
	}

	/// Clones and appends all elements in a slice to the `Vec`.
	///
	/// Iterates over the slice `other`, clones each element, and then appends
	/// it to this `CalfVec`. The `other` vector is traversed in-order.
	///
	/// Note that this function is same as [`extend`] except that it is
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
		self.to_mut().clear()
	}

	/// Appends an element to the back of a collection.
	///
	/// # Panics
	///
	/// Panics if the new capacity exceeds `M::MAX_LENGTH` bytes.
	#[inline]
	pub fn push(&mut self, value: T) {
		self.to_mut().push(value)
	}

	/// Removes the last element from a vector and returns it, or [`None`] if it
	/// is empty.
	#[inline]
	pub fn pop(&mut self) -> Option<T> {
		self.to_mut().pop()
	}
}

unsafe impl<'a, M: Meta + Send, T: Sync, const N: usize> Send for CalfVec<'a, M, T, N> {}
unsafe impl<'a, M: Meta + Sync, T: Sync, const N: usize> Sync for CalfVec<'a, M, T, N> {}

impl<'a, M: Meta, T, const N: usize> Deref for CalfVec<'a, M, T, N> {
	type Target = [T];

	#[inline]
	fn deref(&self) -> &[T] {
		self.as_slice()
	}
}

impl<'a, M: Meta, T, const N: usize> DerefMut for CalfVec<'a, M, T, N> where T: Clone {
	#[inline]
	fn deref_mut(&mut self) -> &mut [T] {
		self.as_mut_slice()
	}
}

impl<'a, M: Meta, T, const N: usize> Extend<T> for CalfVec<'a, M, T, N> where T: Clone {
	#[inline]
	fn extend<I: IntoIterator<Item = T>>(&mut self, iterator: I) {
		self.to_mut().extend(iterator)
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

pub struct CalfVecMut<'v, 'a, M: Meta, T, const N: usize> {
	vec: &'v mut CalfVec<'a, M, T, N>
}

impl<'v, 'a, M: Meta, T, const N: usize> CalfVecMut<'v, 'a, M, T, N> {
	/// Returns the size of the array.
	#[inline]
	pub fn len(&self) -> usize {
		self.vec.len()
	}

	/// Returns `true` if the vector contains no elements.
	#[inline]
	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}

	/// Returns the capacity of the buffer.
	#[inline]
	pub fn capacity(&self) -> usize {
		self.vec.capacity().unwrap()
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
	/// derived from it. If you need to mutate the contents of the slice, use [`as_mut_ptr`].
	#[inline]
	pub fn as_ptr(&self) -> *const T {
		unsafe {
			if self.capacity() <= N {
				(*self.vec.data.stack).as_ptr()
			} else {
				self.vec.data.ptr
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

	/// Returns an unsafe mutable pointer to the vector's buffer.
	///
	/// The caller must ensure that the vector outlives the pointer this
	/// function returns, or else it will end up pointing to garbage.
	/// Modifying the vector may cause its buffer to be reallocated,
	/// which would also make any pointers to it invalid.
	#[inline]
	pub fn as_mut_ptr(&mut self) -> *mut T {
		unsafe {
			if self.capacity() <= N {
				(*self.vec.data.stack).as_mut_ptr()
			} else {
				self.vec.data.ptr
			}
		}
	}

	/// Convert this handle and to a mutable slice of the entire vector.
	#[inline]
	pub fn into_mut_slice(mut self) -> &'v mut [T] {
		unsafe {
			std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len())
		}
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
		unsafe {
			if len > self.len() {
				return;
			}

			let remaining_len = self.len() - len;
			let s = ptr::slice_from_raw_parts_mut(self.as_mut_ptr().add(len), remaining_len);
			self.vec.meta.set_len(len);
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
	#[inline]
	pub fn reserve(&mut self, additional: usize) {
		unsafe {
			let mut vec = if self.capacity() <= N {
				// time to spill!

				let mut data = Data {
					ptr: ptr::null_mut()
				};

				std::mem::swap(&mut data, &mut self.vec.data);

				let boxed_slice: Box<[T]> = Box::new(ManuallyDrop::into_inner(data.stack));
				let mut vec = boxed_slice.into_vec();

				self.vec.data.ptr = vec.as_mut_ptr();

				vec
			} else {
				Vec::from_raw_parts(self.vec.data.ptr, self.len(), self.capacity())
			};

			vec.reserve(additional);
			let (ptr, _, capacity) = vec.into_raw_parts();
			self.vec.data.ptr = ptr;
			self.vec.meta.set_capacity(Some(capacity));
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
		self.vec.meta.set_len(len + count);
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
		unsafe {
			if self.len() == self.capacity() {
				self.reserve(1);
			}

			let end = self.as_mut_ptr().add(self.len());
			ptr::write(end, value);
			self.vec.meta.set_len(self.len()+1);
		}
	}

	/// Removes the last element from a vector and returns it, or [`None`] if it
	/// is empty.
	#[inline]
	pub fn pop(&mut self) -> Option<T> {
		if self.len() == 0 {
			None
		} else {
			unsafe {
				self.vec.meta.set_len(self.len()-1);
				Some(ptr::read(self.as_ptr().add(self.len())))
			}
		}
	}
}

impl<'v, 'a, M: Meta, T, const N: usize> Deref for CalfVecMut<'v, 'a, M, T, N> {
	type Target = [T];

	#[inline]
	fn deref(&self) -> &[T] {
		self.vec.deref()
	}
}

impl<'v, 'a, M: Meta, T, const N: usize> DerefMut for CalfVecMut<'v, 'a, M, T, N> where T: Clone {
	#[inline]
	fn deref_mut(&mut self) -> &mut [T] {
		self.vec.deref_mut()
	}
}

impl<'v, 'a, M: Meta, T, const N: usize> Extend<T> for CalfVecMut<'v, 'a, M, T, N> {
	#[inline]
	fn extend<I: IntoIterator<Item = T>>(&mut self, iterator: I) {
		let mut iterator = iterator.into_iter();
		while let Some(element) = iterator.next() {
			let len = self.len();
			if len == self.capacity() {
				let (lower, _) = iterator.size_hint();
				self.reserve(lower.saturating_add(1));
			}
			unsafe {
				ptr::write(self.as_mut_ptr().add(len), element);
				// NB can't overflow since we would have had to alloc the address space
				self.vec.meta.set_len(len + 1);
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
