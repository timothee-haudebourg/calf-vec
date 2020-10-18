use std::{
	marker::PhantomData,
	mem::ManuallyDrop,
	ptr,
	ops::{
		Deref,
		DerefMut
	}
};

pub trait Meta {
	const MAX_LENGTH: usize;

	fn new(len: usize, capacity: Option<usize>) -> Self;

	fn len(&self) -> usize;

	fn capacity(&self) -> Option<usize>;

	fn set_len(&mut self, len: usize);

	fn set_capacity(&mut self, capacity: Option<usize>);
}

pub union Data<T, const N: usize> {
	stack: ManuallyDrop<[T; N]>,
	ptr: *mut T
}

pub struct CalfVec<'a, M: Meta, T, const N: usize> {
	meta: M,
	data: Data<T, N>,
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
	pub fn borrowed<B: AsRef<[T]>>(borrowed: &'a B) -> CalfVec<'a, M, T, N> {
		let slice = borrowed.as_ref();

		CalfVec {
			meta: M::new(slice.len(), None),
			data: Data { ptr: slice.as_ptr() as *mut T },
			lifetime: PhantomData
		}
	}

	pub fn owned<O: Into<Vec<T>>>(owned: O) -> CalfVec<'a, M, T, N> {
		let vec = owned.into();
		if vec.capacity() <= N {
			// put on stack
			panic!("TODO")
		} else {
			// put on heap
			let (ptr, len, capacity) = vec.into_raw_parts();
			CalfVec {
				meta: M::new(len, Some(capacity)),
				data: Data { ptr },
				lifetime: PhantomData
			}
		}
	}

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

	pub fn is_owned(&self) -> bool {
		self.meta.capacity().is_some()
	}

	pub fn is_borrowed(&self) -> bool {
		self.meta.capacity().is_none()
	}

	pub fn len(&self) -> usize {
		self.meta.len()
	}

	pub fn capacity(&self) -> Option<usize> {
		self.meta.capacity()
	}

	pub fn as_slice(&self) -> &[T] {
		unsafe {
			match self.capacity() {
				Some(capacity) => {
					if capacity <= N {
						(*self.data.stack).as_ref()
					} else {
						std::slice::from_raw_parts(self.data.ptr, self.len())
					}
				},
				None => {
					std::slice::from_raw_parts(self.data.ptr, self.len())
				}
			}
		}
	}
}

impl<'a, M: Meta, T, const N: usize> CalfVec<'a, M, T, N> where T: Clone {
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

	#[inline]
	pub fn as_mut_ptr(&mut self) -> *mut T {
		self.to_mut().as_mut_ptr()
	}

	#[inline]
	pub fn truncate(&mut self, len: usize) {
		self.to_mut().truncate(len)
	}

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
	/// it to this `Vec`. The `other` vector is traversed in-order.
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

	fn deref(&self) -> &[T] {
		self.as_slice()
	}
}

impl<'a, M: Meta, T, const N: usize> DerefMut for CalfVec<'a, M, T, N> {
	fn deref_mut(&mut self) -> &mut [T] {
		unsafe {
			match self.capacity() {
				Some(capacity) => {
					if capacity <= N {
						(*self.data.stack).as_mut()
					} else {
						std::slice::from_raw_parts_mut(self.data.ptr, self.len())
					}
				},
				None => {
					std::slice::from_raw_parts_mut(self.data.ptr, self.len())
				}
			}
		}
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
	#[inline]
	pub fn len(&self) -> usize {
		self.vec.len()
	}

	/// Returns `true` if the vector contains no elements.
	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}

	#[inline]
	pub fn capacity(&self) -> usize {
		self.vec.capacity().unwrap()
	}

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

	fn deref(&self) -> &[T] {
		self.vec.deref()
	}
}

impl<'v, 'a, M: Meta, T, const N: usize> DerefMut for CalfVecMut<'v, 'a, M, T, N> {
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
