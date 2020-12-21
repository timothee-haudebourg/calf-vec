use core::{
	alloc::{
		Allocator,
		Layout,
		LayoutError
	},
	marker::PhantomData,
	mem::{
		self,
		ManuallyDrop,
		MaybeUninit
	},
	ptr::{
		self,
		NonNull
	},
	ops::{
		Deref,
		DerefMut
	},
	fmt,
	cmp
};
use std::{
	alloc::{
		Global,
		handle_alloc_error
	},
	collections::TryReserveError,
	borrow::Cow
};
use crate::raw::{
	self,
	RawCalfVec
};

/// Metadata representing the length and capacity of the array.
///
/// This crate provides two implementation of this trait:
/// [`wide::Meta`](crate::wide::Meta) stores the length and capacity with two `usize`.
/// Then the maximum size/capacity depends on the bit-depth of the plateform.
/// For 64-bit plateforms, this crate also provides [`lean::Meta`](crate::lean::Meta) that stores both the length
/// and capacity on a single `usize`. As a result, the maximum size/capacity is [`std::u32::MAX`].
pub unsafe trait Meta: raw::Meta {
	/// Create a new metadata from an array's length and capacity (if any).
	fn new(len: usize, capacity: Option<usize>) -> Self;

	/// Get the length of the array.
	fn len(&self) -> usize;
	/// Set the new length of the array.
	fn set_len(&mut self, len: usize);
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
pub struct CalfVec<'a, M: Meta, T, A: Allocator, const N: usize> {
	/// Raw vec.
	raw: RawCalfVec<M, T, A, N>,

	/// Remembers the lifetime of the data if it is borrowed.
	lifetime: PhantomData<&'a T>
}

impl<'a, M: Meta, T, A: Allocator, const N: usize> Drop for CalfVec<'a, M, T, A, N> {
	fn drop(&mut self) {
		match self.capacity() {
			Some(_) => unsafe {
				let len = self.raw.meta().len();

				// drop every element.
				ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.raw.owned_as_mut_ptr(), len));
			},
			None => ()
		}

		// NOTE: deallocation is handled by `RawCalfVec`.
	}
}

impl<'a, M: Meta, T, const N: usize> CalfVec<'a, M, T, Global, N> {
	/// Creates a new empty `CalfVec`.
	/// 
	/// The vector will not allocate until more than `N` elements are pushed onto it.
	// TODO make this function `const` as soon as the `const_fn` feature allows it.
	#[inline]
	pub fn new() -> Self {
		Self::new_in(Global)
	}

	/// Creates a new empty `CalfVec` with a particular capacity.
	///
	/// The actual capacity of the created `CalfVec` will be at least `N`.
	#[inline]
	pub fn with_capacity(capacity: usize) -> Self {
		Self::with_capacity_in(capacity, Global)
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
	pub fn borrowed<B: AsRef<[T]> + ?Sized>(borrowed: &'a B) -> Self {
		Self::borrowed_in(borrowed, Global)
	}
}

impl<'a, M: Meta, T, A: Allocator, const N: usize> CalfVec<'a, M, T, A, N> {
	/// Constructs a new, empty `CalfVec<M, T, A, N>`.
	///
	/// The vector will not allocate until more than `N` elements are pushed onto it.
	#[inline]
	pub fn new_in(alloc: A) -> CalfVec<'a, M, T, A, N> {
		CalfVec {
			raw: RawCalfVec::new_in(alloc),
			lifetime: PhantomData
		}
	}

	/// Like `with_capacity`, but parameterized over the choice of
	/// allocator for the returned `RawVec`.
	#[inline]
	pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
		CalfVec {
			raw: RawCalfVec::with_capacity_in(capacity, alloc),
			lifetime: PhantomData
		}
	}

	/// Like `with_capacity_zeroed`, but parameterized over the choice
	/// of allocator for the returned `RawVec`.
	#[inline]
	pub fn with_capacity_zeroed_in(capacity: usize, alloc: A) -> Self {
		CalfVec {
			raw: RawCalfVec::with_capacity_zeroed_in(capacity, alloc),
			lifetime: PhantomData
		}
	}

	#[inline]
	pub fn borrowed_in<B: AsRef<[T]> + ?Sized>(borrowed: &'a B, alloc: A) -> Self {
		let slice = borrowed.as_ref();

		CalfVec {
			raw: RawCalfVec::borrowed_in(unsafe { NonNull::new_unchecked(slice.as_ptr() as *mut T) }, alloc),
			lifetime: PhantomData
		}
	}

	/// Create a new `CalfVec` from owned data.
	///
	/// The input is consumed and stored either on the stack if it does not exceed the
	/// capacity parameter `N`, or on the heap otherwise.
	#[inline]
	pub fn owned<O: Into<Vec<T, A>>>(owned: O) -> CalfVec<'a, M, T, A, N> {
		let vec = owned.into();
		let (ptr, len, capacity, alloc) = vec.into_raw_parts_with_alloc();

		let raw = unsafe {
			if capacity <= N {
				// put on stack.
				let mut raw = RawCalfVec::with_capacity_in(N, alloc);
				std::ptr::copy_nonoverlapping(ptr, raw.owned_as_mut_ptr(), len);

				// deallocate heap buffer.
				let layout = Layout::array::<T>(capacity).unwrap();
				raw.allocator().deallocate(NonNull::new_unchecked(ptr).cast(), layout);

				raw
			} else {
				// put on heap.
				RawCalfVec::spilled_in(NonNull::new_unchecked(ptr), capacity, alloc)
			}
		};

		CalfVec {
			raw,
			lifetime: PhantomData
		}
	}

	/// Returns a reference to the underlying allocator.
	#[inline]
	pub fn allocator(&self) -> &A {
		self.raw.allocator()
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

	#[inline]
	pub fn into_raw(self) -> RawCalfVec<M, T, A, N> {
		let raw = unsafe { ptr::read(&self.raw) };
		std::mem::forget(self);
		raw
	}

	#[inline]
	pub fn into_raw_parts_with_alloc(self) -> (M, raw::Data<T, N>, A) {
		self.into_raw().into_raw_parts_with_alloc()
	}

	/// Try to convert this `CalfVec` into `Vec`.
	///
	/// Returns `Ok(vec)` if the data is owned and on the heap, and `Err(self)` otherwise.
	///
	/// This is a cost-free operation.
	#[inline]
	pub fn try_into_vec(self) -> Result<Vec<T, A>, Self> {
		match self.capacity() {
			Some(capacity) if capacity > N => unsafe {
				let (meta, data, alloc) = self.into_raw_parts_with_alloc();
				Ok(Vec::from_raw_parts_in(data.ptr.as_ptr(), meta.len(), capacity, alloc))
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
			let (meta, data, alloc) = self.into_raw_parts_with_alloc();
			let len = meta.len();

			let vec = if capacity <= N {
				let src = (*data.stack).as_ptr() as *const T;

				let mut vec = Vec::with_capacity_in(len, alloc);
				std::ptr::copy_nonoverlapping(src, vec.as_mut_ptr(), len);
				vec
			} else {
				Vec::from_raw_parts_in(data.ptr.as_ptr(), len, capacity, alloc)
			};
			
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
		self.raw.as_ptr()
	}

	/// Returns an unsafe mutable pointer to the owned vector's buffer.
	///
	/// Same as [`as_mut_ptr`] but the caller must ensure that the data is owned by the vector.
	#[inline]
	pub unsafe fn owned_as_mut_ptr(&mut self) -> *mut T {
		self.raw.owned_as_mut_ptr()
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
		self.capacity().is_some()
	}

	/// Returns true if the data is borrowed, i.e. if `to_mut` would require additional work.
	#[inline]
	pub fn is_borrowed(&self) -> bool {
		self.capacity().is_none()
	}

	/// Returns `true` if the data is owned and stored on the heap,
	/// `false` if it is borrowed or stored on the stack.
	#[inline]
	pub fn is_spilled(&self) -> bool {
		match self.capacity() {
			Some(capacity) => capacity > N,
			None => false
		}
	}

	/// Returns the length of the array.
	#[inline]
	pub fn len(&self) -> usize {
		self.raw.meta().len()
	}

	#[inline]
	pub unsafe fn set_len(&mut self, len: usize) {
		self.raw.meta_mut().set_len(len)
	}

	/// Returns the capacity of the owned buffer, or `None` if the data is only borrowed.
	#[inline]
	pub fn capacity(&self) -> Option<usize> {
		self.raw.capacity()
	}

	/// Returns the remaining spare capacity of the vector as a slice of
	/// `MaybeUninit<T>`.
	///
	/// The returned slice can be used to fill the vector with data (e.g. by
	/// reading from a file) before marking the data as initialized using the
	/// [`set_len`] method.
	///
	/// [`set_len`]: CalfVec::set_len
	#[inline]
	pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
		let len = self.len();
		match self.capacity() {
			Some(capacity) => unsafe {
				let ptr = self.owned_as_mut_ptr() as *mut MaybeUninit<T>;
				std::slice::from_raw_parts_mut(ptr.add(len), capacity - len)
			},
			None => {
				&mut []
			}
		}
	}
}

impl<'a, M: Meta, T, A: Allocator, const N: usize> CalfVec<'a, M, T, A, N> where T: Clone {
	#[inline]
	pub fn own(&mut self) -> usize {
		match self.capacity() {
			Some(capacity) => capacity,
			None => unsafe { // copy time!
				let len = self.len();
				let slice = std::slice::from_raw_parts(self.raw.as_ptr(), len);

				let alloc = ptr::read(self.allocator()); // this is safe because `self.alloc` is never used ever after.
				let mut vec = T::to_calf_vec(slice, alloc);
				std::mem::swap(&mut vec, self);
				std::mem::forget(vec);

				self.capacity().unwrap()
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
		unsafe { self.raw.owned_as_mut_ptr_with_capacity(capacity) }
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

	/// The same as `reserve`, but returns on errors instead of panicking or aborting.
	pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
		if let Some(dst) = self.raw.try_reserve(self.len(), additional)? {
			panic!("TODO") // copy to buffer.
		}

		Ok(())
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
		handle_reserve(self.try_reserve(additional))
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
		handle_reserve(self.try_reserve_exact(additional))
	}

	/// The same as `reserve`, but returns on errors instead of panicking or aborting.
	pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
		if let Some(dst) = self.raw.try_reserve_exact(self.len(), additional)? {
			panic!("TODO") // copy to buffer.
		}

		Ok(())
	}

	/// The same as `shrink_to`, but returns on errors instead of panicking or aborting.
	pub fn try_shrink_to(&mut self, min_capacity: usize) -> Result<(), TryReserveError> {
		self.raw.try_shrink_to(min_capacity)
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
	pub fn shrink_to(&mut self, min_capacity: usize) {
		self.raw.shrink_to(min_capacity)
	}

	/// Shrinks the capacity of the vector as much as possible.
	///
	/// It will drop down as close as possible to the length but the allocator
	/// may still inform the vector that there is space for a few more elements.
	#[inline]
	pub fn shrink_to_fit(&mut self) {
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

	/// Resizes the `CalfVec` in-place so that `len` is equal to `new_len`.
	///
	/// If `new_len` is greater than `len`, the `CalfVec` is extended by the
	/// difference, with each additional slot filled with `value`.
	/// If `new_len` is less than `len`, the `CalfVec` is simply truncated.
	///
	/// This method requires `T` to implement [`Clone`],
	/// in order to be able to clone the passed value.
	/// If you need more flexibility (or want to rely on [`Default`] instead of
	/// [`Clone`]), use [`CalfVec::resize_with`].
	pub fn resize(&mut self, new_len: usize, value: T) {
		let len = self.len();

		if new_len > len {
			self.extend_with(new_len - len, ExtendElement(value))
		} else {
			self.truncate(new_len);
		}
	}

	/// Clones and appends all elements in a slice to the `CalfVec`.
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

impl<'a, M: Meta, T, A: Allocator, const N: usize> CalfVec<'a, M, T, A, N> where T: Clone {
	/// Extend the vector by `n` values, using the given generator.
	fn extend_with<E: ExtendWith<T>>(&mut self, n: usize, mut value: E) {
		self.reserve(n);

		unsafe {
			let mut ptr = self.as_mut_ptr().add(self.len());
			// Use SetLenOnDrop to work around bug where compiler
			// may not realize the store through `ptr` through self.set_len()
			// don't alias.
			let mut local_len = SetLenOnDrop::new(self.raw.meta_mut());

			// Write all elements except the last one
			for _ in 1..n {
				ptr::write(ptr, value.next());
				ptr = ptr.offset(1);
				// Increment the length in every step in case next() panics
				local_len.increment_len(1);
			}

			if n > 0 {
				// We can write the last element directly without cloning needlessly
				ptr::write(ptr, value.last());
				local_len.increment_len(1);
			}

			// len set by scope guard
		}
	}
}

// Set the length of the vec when the `SetLenOnDrop` value goes out of scope.
//
// The idea is: The length field in SetLenOnDrop is a local variable
// that the optimizer will see does not alias with any stores through the Vec's data
// pointer. This is a workaround for alias analysis issue #32155
struct SetLenOnDrop<'a, M: Meta> {
	meta: &'a mut M,
	local_len: usize,
}

impl<'a, M: Meta> SetLenOnDrop<'a, M> {
	#[inline]
	fn new(meta: &'a mut M) -> Self {
		SetLenOnDrop { local_len: meta.len(), meta }
	}

	#[inline]
	fn increment_len(&mut self, increment: usize) {
		self.local_len += increment;
	}
}

impl<M: Meta> Drop for SetLenOnDrop<'_, M> {
	#[inline]
	fn drop(&mut self) {
		self.meta.set_len(self.local_len);
	}
}

// This code generalizes `extend_with_{element,default}`.
trait ExtendWith<T> {
	fn next(&mut self) -> T;
	fn last(self) -> T;
}

struct ExtendElement<T>(T);
impl<T: Clone> ExtendWith<T> for ExtendElement<T> {
	fn next(&mut self) -> T {
		self.0.clone()
	}
	fn last(self) -> T {
		self.0
	}
}

struct ExtendDefault;
impl<T: Default> ExtendWith<T> for ExtendDefault {
	fn next(&mut self) -> T {
		Default::default()
	}
	fn last(self) -> T {
		Default::default()
	}
}

struct ExtendFunc<F>(F);
impl<T, F: FnMut() -> T> ExtendWith<T> for ExtendFunc<F> {
	fn next(&mut self) -> T {
		(self.0)()
	}
	fn last(mut self) -> T {
		(self.0)()
	}
}

impl<'a, M: Meta, T: Clone, A: Allocator + Clone, const N: usize> Clone for CalfVec<'a, M, T, A, N> {
	fn clone(&self) -> CalfVec<'a, M, T, A, N> {
		CalfVec::owned(self)
	}
}

impl<'v, 'a, M: Meta, T: Clone, A: Allocator + Clone, const N: usize> From<&'v CalfVec<'a, M, T, A, N>> for Vec<T, A> {
	fn from(vec: &'v CalfVec<'a, M, T, A, N>) -> Vec<T, A> {
		vec.to_vec_in(vec.allocator().clone())
	}
}

impl<'a, M: Meta, T: Clone, A: Allocator, const N: usize> From<CalfVec<'a, M, T, A, N>> for Vec<T, A> {
	fn from(vec: CalfVec<'a, M, T, A, N>) -> Vec<T, A> {
		vec.into_vec()
	}
}

unsafe impl<'a, M: Meta + Send, T: Sync, A: Allocator + Send, const N: usize> Send for CalfVec<'a, M, T, A, N> {}
unsafe impl<'a, M: Meta + Sync, T: Sync, A: Allocator, const N: usize> Sync for CalfVec<'a, M, T, A, N> {}

impl<'a, M: Meta, T, A: Allocator, const N: usize> Deref for CalfVec<'a, M, T, A, N> {
	type Target = [T];

	#[inline]
	fn deref(&self) -> &[T] {
		self.as_slice()
	}
}

impl<'a, M: Meta, T, A: Allocator, const N: usize> DerefMut for CalfVec<'a, M, T, A, N> where T: Clone {
	#[inline]
	fn deref_mut(&mut self) -> &mut [T] {
		self.as_mut_slice()
	}
}

impl<'v, 'a, M: Meta, T, A: Allocator, const N: usize> IntoIterator for &'v CalfVec<'a, M, T, A, N> {
	type Item = &'v T;
	type IntoIter = std::slice::Iter<'v, T>;

	fn into_iter(self) -> Self::IntoIter {
		self.as_slice().into_iter()
	}
}

impl<'v, 'a, M: Meta, T, A: Allocator, const N: usize> IntoIterator for &'v mut CalfVec<'a, M, T, A, N> where T: Clone {
	type Item = &'v mut T;
	type IntoIter = std::slice::IterMut<'v, T>;

	fn into_iter(self) -> Self::IntoIter {
		self.as_mut_slice().into_iter()
	}
}

pub union IntoIterData<T, A: Allocator, const N: usize> {
	stack: ManuallyDrop<[MaybeUninit<T>; N]>,
	vec: ManuallyDrop<std::vec::IntoIter<T, A>>
}

pub struct IntoIter<M: Meta, T, A: Allocator, const N: usize> {
	meta: M,
	offset: usize,
	data: IntoIterData<T, A, N>
}

impl<M: Meta, T, A: Allocator, const N: usize> Iterator for IntoIter<M, T, A, N> {
	type Item = T;

	fn next(&mut self) -> Option<T> {
		unsafe {
			let capacity = self.meta.capacity().unwrap();
			let item = if capacity <= N {
				let i = self.offset;
				if i < self.meta.len() {
					self.offset += 1;
					Some(self.data.stack[i].assume_init_read())
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

impl<M: Meta, T, A: Allocator, const N: usize> Drop for IntoIter<M, T, A, N> {
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

impl<'a, M: Meta, T, A: Allocator, const N: usize> IntoIterator for CalfVec<'a, M, T, A, N> where T: Clone {
	type Item = T;
	type IntoIter = IntoIter<M, T, A, N>;

	fn into_iter(mut self) -> Self::IntoIter {
		unsafe {
			let capacity = self.own();
			let (meta, data, alloc) = self.into_raw_parts_with_alloc();

			let into_iter_data = if capacity <= N {
				IntoIterData {
					stack: data.stack
				}
			} else {
				let vec = Vec::from_raw_parts_in(data.ptr.as_ptr(), meta.len(), capacity, alloc);
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

impl<'a, M: Meta, T, A: Allocator, const N: usize> Extend<T> for CalfVec<'a, M, T, A, N> where T: Clone {
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

impl<'a, M: Meta, T: fmt::Debug, A: Allocator, const N: usize> fmt::Debug for CalfVec<'a, M, T, A, N> {
	#[inline]
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		fmt::Debug::fmt(&**self, f)
	}
}

impl<'a, M: Meta, T, A: Allocator, const N: usize> AsRef<CalfVec<'a, M, T, A, N>> for CalfVec<'a, M, T, A, N> {
	#[inline]
	fn as_ref(&self) -> &CalfVec<'a, M, T, A, N> {
		self
	}
}

impl<'a, M: Meta, T, A: Allocator, const N: usize> AsMut<CalfVec<'a, M, T, A, N>> for CalfVec<'a, M, T, A, N> {
	#[inline]
	fn as_mut(&mut self) -> &mut CalfVec<'a, M, T, A, N> {
		self
	}
}

impl<'a, M: Meta, T, A: Allocator, const N: usize> AsRef<[T]> for CalfVec<'a, M, T, A, N> {
	#[inline]
	fn as_ref(&self) -> &[T] {
		self
	}
}

impl<'a, M: Meta, T, A: Allocator, const N: usize> AsMut<[T]> for CalfVec<'a, M, T, A, N> where T: Clone {
	#[inline]
	fn as_mut(&mut self) -> &mut [T] {
		self
	}
}

impl<'a, M: Meta, T, A: Allocator + Clone, const N: usize> From<Vec<T, A>> for CalfVec<'a, M, T, A, N> {
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

impl_slice_eq1! { ['a, 'b, T, U, O: Meta, P: Meta, A: Allocator, B: Allocator, const N: usize, const M: usize] CalfVec<'a, O, T, A, N>, CalfVec<'b, P, U, B, M> }
impl_slice_eq1! { ['a, T, U, M: Meta, A: Allocator, B: Allocator, const N: usize] CalfVec<'a, M, T, A, N>, Vec<U, B> }
impl_slice_eq1! { ['b, T, U, M: Meta, A: Allocator, B: Allocator, const N: usize] Vec<T, A>, CalfVec<'b, M, U, B, N> }
impl_slice_eq1! { ['a, T, U, M: Meta, A: Allocator, const N: usize] CalfVec<'a, M, T, A, N>, &[U] }
impl_slice_eq1! { ['a, T, U, M: Meta, A: Allocator, const N: usize] CalfVec<'a, M, T, A, N>, &mut [U] }
impl_slice_eq1! { ['b, T, U, M: Meta, A: Allocator, const N: usize] &[T], CalfVec<'b, M, U, A, N> }
impl_slice_eq1! { ['b, T, U, M: Meta, A: Allocator, const N: usize] &mut [T], CalfVec<'b, M, U, A, N> }
impl_slice_eq1! { ['a, T, U, M: Meta, A: Allocator, const N: usize] CalfVec<'a, M, T, A, N>, Cow<'_, [U]> where U: Clone }
impl_slice_eq1! { ['b, T, U, M: Meta, A: Allocator, const N: usize] Cow<'_, [T]>, CalfVec<'b, M, U, A, N> where T: Clone }
impl_slice_eq1! { ['a, T, U, M: Meta, A: Allocator, const N: usize, const O: usize] CalfVec<'a, M, T, A, N>, [U; O] }
impl_slice_eq1! { ['a, T, U, M: Meta, A: Allocator, const N: usize, const O: usize] CalfVec<'a, M, T, A, N>, &[U; O] }
impl_slice_eq1! { ['b, T, U, M: Meta, A: Allocator, const N: usize, const O: usize] [T; O], CalfVec<'b, M, U, A, N> }
impl_slice_eq1! { ['b, T, U, M: Meta, A: Allocator, const N: usize, const O: usize] &[T; O], CalfVec<'b, M, U, A, N> }

impl<'a, M: Meta, T: Eq, A: Allocator, const N: usize> Eq for CalfVec<'a, M, T, A, N> {}

// Central function for reserve error handling.
#[inline]
fn handle_reserve(result: Result<(), TryReserveError>) {
	match result {
		Err(TryReserveError::CapacityOverflow) => capacity_overflow(),
		Err(TryReserveError::AllocError { layout, .. }) => handle_alloc_error(layout),
		Ok(()) => { /* yay */ }
	}
}

// One central function responsible for reporting capacity overflows. This'll
// ensure that the code generation related to these panics is minimal as there's
// only one location which panics rather than a bunch throughout the module.
fn capacity_overflow() -> ! {
	panic!("capacity overflow");
}

pub trait ToCalfVec {
	/// Convert the input slice into a `CalfVec` with at least the given capacity.
	fn to_calf_vec_with_capacity<'t, M: Meta, A: Allocator, const N: usize>(s: &[Self], capacity: usize, alloc: A) -> CalfVec<'t, M, Self, A, N> where Self: Sized;

	#[inline]
	fn to_calf_vec<'t, M: Meta, A: Allocator, const N: usize>(s: &[Self], alloc: A) -> CalfVec<'t, M, Self, A, N> where Self: Sized {
		Self::to_calf_vec_with_capacity(s, s.len(), alloc)
	}
}

impl<T: Clone> ToCalfVec for T {
	#[inline]
	default fn to_calf_vec_with_capacity<'t, M: Meta, A: Allocator, const N: usize>(s: &[Self], capacity: usize, alloc: A) -> CalfVec<'t, M, Self, A, N> {
		struct DropGuard<'a, 'b, M: Meta, T, A: Allocator, const N: usize> {
			vec: &'a mut CalfVec<'b, M, T, A, N>,
			num_init: usize,
		}

		impl<'a, 'b, M: Meta, T, A: Allocator, const N: usize> Drop for DropGuard<'a, 'b, M, T, A, N> {
			#[inline]
			fn drop(&mut self) {
				// SAFETY:
				// items were marked initialized in the loop below
				unsafe {
					self.vec.set_len(self.num_init);
				}
			}
		}

		let capacity = cmp::max(capacity, s.len());
		let mut vec = CalfVec::with_capacity_in(capacity, alloc);
		let mut guard = DropGuard { vec: &mut vec, num_init: 0 };

		let slots = guard.vec.spare_capacity_mut();

		// .take(slots.len()) is necessary for LLVM to remove bounds checks
		// and has better codegen than zip.
		for (i, b) in s.iter().enumerate().take(slots.len()) {
			guard.num_init = i;
			slots[i].write(b.clone());
		}

		core::mem::forget(guard);

		// SAFETY:
		// the vec was allocated and initialized above to at least this length.
		unsafe {
			vec.set_len(s.len());
		}

		vec
	}
}

impl<T: Copy> ToCalfVec for T {
	#[inline]
	fn to_calf_vec_with_capacity<'t, M: Meta, A: Allocator, const N: usize>(s: &[Self], capacity: usize, alloc: A) -> CalfVec<'t, M, Self, A, N> {
		let capacity = cmp::max(capacity, s.len());
		let mut v = CalfVec::with_capacity_in(capacity, alloc);

		// SAFETY:
		// allocated above with the capacity of `s`, and initialize to `s.len()` in
		// ptr::copy_to_non_overlapping below.
		unsafe {
			s.as_ptr().copy_to_nonoverlapping(v.as_mut_ptr(), s.len());
			v.set_len(s.len());
		}

		v
	}
}