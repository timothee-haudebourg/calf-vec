use core::{
	alloc::{
		Allocator,
		Layout,
		LayoutError
	},
	mem::{
		self,
		ManuallyDrop,
		MaybeUninit
	},
	ptr::{
		self,
		NonNull
	},
	cmp
};
use std::{
	alloc::{
		Global,
		handle_alloc_error
	},
	collections::TryReserveError
};

/// Metadata representing the capacity of a `RawCalfVec`.
pub unsafe trait Meta: Copy {
	/// Maximum size/capacity of the array using this metadata format.
	const MAX_LENGTH: usize;

	/// Create a new metadata from an array's length and capacity (if any).
	fn with_capacity(capacity: Option<usize>) -> Self;

	/// Get the capacity of the buffer, if any.
	///
	/// The capacity is only defined on owned buffers.
	fn capacity(&self) -> Option<usize>;

	/// Set the new capacity of the buffer.
	fn set_capacity(&mut self, capacity: Option<usize>);
}

enum AllocInit {
	/// The contents of the new memory are uninitialized.
	Uninitialized,
	/// The new memory is guaranteed to be zeroed.
	Zeroed,
}

/// Inner data storage.
///
/// We use an union here since the actual type depends on the where the data is stored.
/// If the data is owned and on the stack, then the relevent field is `stack`.
/// If the data is borrowed or spilled, the the relevent field is `ptr`.
pub union Data<T, const N: usize> {
	/// Data stored on the stack.
	pub stack: ManuallyDrop<[MaybeUninit<T>; N]>,

	/// Pointer to the data (either borrowed, or owned on the heap).
	pub ptr: NonNull<T>
}

impl<T, const N: usize> Data<T, N> {
	#[inline]
	fn new(init: AllocInit) -> Data<T, N> {
		match init {
			AllocInit::Uninitialized => {
				Data {
					stack: ManuallyDrop::new(
						// SAFETY: An uninitialized `[MaybeUninit<_>; N]` is valid.
						unsafe { MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init() }
					)
				}
			},
			AllocInit::Zeroed => {
				Data {
					stack: ManuallyDrop::new(
						// SAFETY: An uninitialized `[MaybeUninit<_>; N]` is valid.
						unsafe { MaybeUninit::<[MaybeUninit<T>; N]>::zeroed().assume_init() }
					)
				}
			}
		}
	}

	#[inline]
	unsafe fn drop_with<M: Meta, A: Allocator>(&mut self, meta: M, alloc: &A) {
		match meta.capacity() {
			Some(capacity) => {
				if capacity <= N { // stacked
				} else { // spilled
					// free memory.
					let align = std::mem::align_of::<T>();
					let size = std::mem::size_of::<T>() * capacity;
					let layout = std::alloc::Layout::from_size_align_unchecked(size, align);
					alloc.deallocate(self.ptr.cast(), layout)
				}
			},
			None => () // borrowed
		}
	}
}

/// A low-level utility for more ergonomically managing a "calf" buffer.
/// 
/// This type does not in anyway inspect the memory that it manages. When dropped it *will*
/// free its memory, but it *won't* try to drop its contents. It is up to the user of `RawCalfVec`
/// to handle the actual things *stored* inside of a `RawCalfVec`.
pub struct RawCalfVec<M: Meta, T, A: Allocator, const N: usize> {
	/// Metadata storing the length and capacity of the array.
	meta: M,

	/// The actual data (or a pointer to the actual data).
	data: Data<T, N>,

	/// Allocator.
	alloc: A
}

impl<M: Meta, T, A: Allocator, const N: usize> Drop for RawCalfVec<M, T, A, N> {
	fn drop(&mut self) {
		unsafe {
			self.data.drop_with(self.meta, &self.alloc)
		}
	}
}

impl<M: Meta, T, const N: usize> RawCalfVec<M, T, Global, N> {
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
	pub fn borrowed(ptr: NonNull<T>) -> Self {
		Self::borrowed_in(ptr, Global)
	}
}

impl<M: Meta, T, A: Allocator, const N: usize> RawCalfVec<M, T, A, N> {
	#[inline]
	pub fn into_raw_parts(self) -> (M, Data<T, N>) {
		let meta = self.meta;
		let data = unsafe { ptr::read(&self.data) };
		mem::forget(self);
		(meta, data)
	}

	#[inline]
	pub fn into_raw_parts_with_alloc(self) -> (M, Data<T, N>, A) {
		let meta = self.meta;
		let data = unsafe { ptr::read(&self.data) };
		let alloc = unsafe { ptr::read(&self.alloc) };
		mem::forget(self);
		(meta, data, alloc)
	}

	/// Initialize a new `CalfVec` of capacity `N` on the stack.
	fn init_in(init: AllocInit, alloc: A) -> Self {
		RawCalfVec {
			meta: M::with_capacity(Some(N)),
			data: Data::new(init),
			alloc
		}
	}

	/// Constructs a new, empty `CalfVec<M, T, A, N>`.
	///
	/// The vector will not allocate until more than `N` elements are pushed onto it.
	#[inline]
	pub fn new_in(alloc: A) -> Self {
		RawCalfVec {
			meta: M::with_capacity(Some(N)),
			data: Data::new(AllocInit::Uninitialized),
			alloc
		}
	}

	fn allocate_in(capacity: usize, init: AllocInit, alloc: A) -> Self {
		if mem::size_of::<T>() == 0 {
			Self::new_in(alloc)
		} else {
			let meta = M::with_capacity(Some(capacity));

			let layout = match Layout::array::<T>(capacity) {
				Ok(layout) => layout,
				Err(_) => capacity_overflow(),
			};
			match alloc_guard(layout.size()) {
				Ok(_) => {}
				Err(_) => capacity_overflow(),
			}
			let result = match init {
				AllocInit::Uninitialized => alloc.allocate(layout),
				AllocInit::Zeroed => alloc.allocate_zeroed(layout),
			};
			let ptr = match result {
				Ok(ptr) => ptr,
				Err(_) => handle_alloc_error(layout),
			};

			Self {
				meta,
				data: Data { ptr: ptr.cast() },
				alloc
			}
		}
	}

	/// Like `with_capacity`, but parameterized over the choice of
	/// allocator for the returned `RawVec`.
	#[inline]
	pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
		if capacity <= N {
			Self::init_in(AllocInit::Uninitialized, alloc)
		} else {
			Self::allocate_in(capacity, AllocInit::Uninitialized, alloc)
		}
	}

	/// Like `with_capacity_zeroed`, but parameterized over the choice
	/// of allocator for the returned `RawVec`.
	#[inline]
	pub fn with_capacity_zeroed_in(capacity: usize, alloc: A) -> Self {
		if capacity <= N {
			Self::init_in(AllocInit::Zeroed, alloc)
		} else {
			Self::allocate_in(capacity, AllocInit::Zeroed, alloc)
		}
	}

	#[inline]
	pub fn borrowed_in(ptr: NonNull<T>, alloc: A) -> Self {
		RawCalfVec {
			meta: M::with_capacity(None),
			data: Data { ptr },
			alloc
		}
	}

	/// The capacity must be greater than `N`.
	#[inline]
	pub fn spilled_in(ptr: NonNull<T>, capacity: usize, alloc: A) -> Self {
		assert!(capacity > N);
		RawCalfVec {
			meta: M::with_capacity(Some(capacity)),
			data: Data { ptr },
			alloc
		}
	}

	#[inline]
	pub fn meta(&self) -> &M {
		&self.meta
	}

	#[inline]
	pub fn meta_mut(&mut self) -> &mut M {
		&mut self.meta
	}

	/// Returns a reference to the underlying allocator.
	#[inline]
	pub fn allocator(&self) -> &A {
		&self.alloc
	}

	/// Returns the current allocated memory and layout.
	///
	/// Returns `None` if the data is borrowed, on the stack,
	/// or is the size of `T` is 0.
	fn current_memory(&self) -> Option<(NonNull<u8>, Layout)> {
		match self.capacity() {
			None => None,
			Some(capacity) => {
				if capacity <= N {
					None
				} else {
					// We have an allocated chunk of memory, so we can bypass runtime
					// checks to get our current layout.
					unsafe {
						let align = mem::align_of::<T>();
						let size = mem::size_of::<T>() * capacity;
						let layout = Layout::from_size_align_unchecked(size, align);
						Some((self.data.ptr.cast().into(), layout))
					}
				}
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
	/// derived from it. If you need to mutate the contents of the slice, use [`as_mut_ptr`](#as_mut_ptr).
	#[inline]
	pub fn as_ptr(&self) -> *const T {
		unsafe {
			match self.capacity() {
				Some(capacity) => {
					if capacity <= N {
						(*self.data.stack).as_ptr() as *const T
					} else {
						self.data.ptr.as_ptr()
					}
				},
				None => self.data.ptr.as_ptr()
			}
		}
	}

	/// Returns an unsafe mutable pointer to the owned vector's buffer.
	///
	/// Same as [`as_mut_ptr`] but the caller must ensure that the data is owned by the vector.
	#[inline]
	pub unsafe fn owned_as_mut_ptr(&mut self) -> *mut T {
		self.owned_as_mut_ptr_with_capacity(self.capacity().unwrap())
	}

	/// Same as [`owned_as_mut_ptr`] but with a given capacity.
	/// 
	/// Useful when you already know the actual capacity of the vector.
	/// 
	/// ## Safety
	/// 
	/// The given capacity must be equal to the vector's capacity.
	#[inline]
	pub unsafe fn owned_as_mut_ptr_with_capacity(&mut self, capacity: usize) -> *mut T {
		if capacity <= N {
			(*self.data.stack).as_mut_ptr() as *mut T
		} else {
			self.data.ptr.as_ptr()
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

	/// Returns the capacity of the owned buffer, or `None` if the data is only borrowed.
	#[inline]
	pub fn capacity(&self) -> Option<usize> {
		if mem::size_of::<T>() == 0 {
			Some(usize::MAX)
		} else {
			self.meta.capacity()
		}
	}
}

impl<M: Meta, T, A: Allocator, const N: usize> RawCalfVec<M, T, A, N> where T: Clone {
	/// The same as `reserve`, but returns on errors instead of panicking or aborting.
	pub fn try_reserve(&mut self, len: usize, additional: usize) -> Result<Option<*mut T>, TryReserveError> {
		unsafe {
			if self.needs_to_grow(len, additional) {
				match self.capacity() {
					Some(capacity) => {
						if capacity <= N { // on stack
							self.spill_amortized(len, additional)?
						} else { // spilled
							// Safe because we just checked that the data is spilled.
							self.grow_amortized(len, additional)?
						}

						Ok(None)
					},
					None => { // borrowed
						Ok(Some(self.prepare_amortized(len, additional)?))
					}
				}
			} else {
				Ok(None)
			}
		}
	}

	/// Reserves capacity for at least `additional` more elements to be inserted
	/// in the given `CalfVec<T>`. The collection may reserve more space to avoid
	/// frequent reallocations. After calling `reserve`, capacity will be
	/// greater than or equal to `self.len() + additional`. Does nothing if
	/// capacity is already sufficient.
	/// 
	/// Returns `Some(ptr)` if the data was borrowed,
	/// in which case the caller is responsible for cloning the data into the newly allocated memory space at `ptr`.
	/// Returns `None` if the data was owned or had already enough capacity,
	/// in which case nothing is expected from the caller.
	///
	/// # Panics
	///
	/// Panics if the new capacity exceeds `M::MAX_LENGTH` bytes.
	pub fn reserve(&mut self, len: usize, additional: usize) -> Option<*mut T> {
		handle_reserve(self.try_reserve(len, additional))
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
	pub fn reserve_exact(&mut self, len: usize, additional: usize) -> Option<*mut T> {
		handle_reserve(self.try_reserve_exact(len, additional))
	}

	/// The same as `reserve`, but returns on errors instead of panicking or aborting.
	pub fn try_reserve_exact(&mut self, len: usize, additional: usize) -> Result<Option<*mut T>, TryReserveError> {
		unsafe {
			if self.needs_to_grow(len, additional) {
				match self.capacity() {
					Some(capacity) => {
						if capacity <= N { // on stack
							self.spill_exact(len, additional)?
						} else { // spilled
							// Safe because we just checked that the data is spilled.
							self.grow_exact(len, additional)?
						}

						Ok(None)
					},
					None => { // borrowed
						self.prepare_exact(len, additional)?;
						Ok(Some(self.data.ptr.as_ptr()))
					}
				}
			} else {
				Ok(None)
			}
		}
	}

	/// The same as `shrink_to`, but returns on errors instead of panicking or aborting.
	pub fn try_shrink_to(&mut self, new_capacity: usize) -> Result<(), TryReserveError> {
		match self.capacity() {
			Some(capacity) => unsafe {
				assert!(new_capacity <= capacity);

				if capacity <= N { // stacked.
					Ok(()) // nothing to do.
				} else { // spilled.
					if new_capacity <= N { // put back on stack.
						// move to stack.
						let ptr = self.data.ptr;
						let src = ptr.as_ptr();
						let dst = (*self.data.stack).as_mut_ptr() as *mut T;
						ptr::copy_nonoverlapping(src, dst, new_capacity);
						self.meta.set_capacity(Some(N));

						// free allocated memory.
						let align = std::mem::align_of::<T>();
						let size = std::mem::size_of::<T>() * capacity;
						let layout = std::alloc::Layout::from_size_align_unchecked(size, align);
						self.alloc.deallocate(ptr.cast(), layout);

						Ok(())
					} else { // shrink allocated memory.
						self.shrink(new_capacity)
					}
				}
			},
			None => {
				Ok(())
			}
		}
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
		handle_reserve(self.try_shrink_to(min_capacity))
	}
}

impl<M: Meta, T, A: Allocator, const N: usize> RawCalfVec<M, T, A, N> where T: Clone {
	/// Returns if the buffer needs to grow to fulfill the needed extra capacity.
	/// Mainly used to make inlining reserve-calls possible without inlining `grow`.
	fn needs_to_grow(&self, len: usize, additional: usize) -> bool {
		match self.capacity() {
			Some(capacity) => {
				additional > capacity.wrapping_sub(len)
			},
			None => true
		}
	}

	fn capacity_from_bytes(excess: usize) -> usize {
		debug_assert_ne!(mem::size_of::<T>(), 0);
		excess / mem::size_of::<T>()
	}

	fn set_ptr(&mut self, ptr: NonNull<[u8]>) {
		self.data.ptr = ptr.cast();
		self.meta.set_capacity(Some(Self::capacity_from_bytes(ptr.len())));
	}

	/// Prepare enough memory to store `len + additional` elements either on the stack or on the heap.
	pub unsafe fn prepare_amortized(&mut self, len: usize, additional: usize) -> Result<*mut T, TryReserveError> {
		debug_assert!(self.is_borrowed());

		let required_capacity = len.checked_add(additional).ok_or(TryReserveError::CapacityOverflow)?;
		let capacity = cmp::max(len * 2, required_capacity);

		let ptr = if capacity <= N {
			self.meta.set_capacity(Some(N));
			(&mut self.data.stack).as_mut_ptr() as *mut T
		} else {
			let new_layout = Layout::array::<T>(capacity);

			// `finish_grow` is non-generic over `T`.
			let ptr = finish_grow(new_layout, self.current_memory(), &mut self.alloc)?;
			self.set_ptr(ptr);
			self.data.ptr.as_ptr()
		};

		Ok(ptr)
	}

	/// Prepare enough memory to store `len + additional` elements either on the stack or on the heap.
	pub unsafe fn prepare_exact(&mut self, len: usize, additional: usize) -> Result<*mut T, TryReserveError> {
		debug_assert!(self.is_borrowed());

		let capacity = len.checked_add(additional).ok_or(TryReserveError::CapacityOverflow)?;

		let ptr = if capacity <= N {
			self.meta.set_capacity(Some(N));
			(&mut self.data.stack).as_mut_ptr() as *mut T
		} else {
			let new_layout = Layout::array::<T>(capacity);

			// `finish_grow` is non-generic over `T`.
			let ptr = finish_grow(new_layout, self.current_memory(), &mut self.alloc)?;
			self.set_ptr(ptr);
			self.data.ptr.as_ptr()
		};

		Ok(ptr)
	}

	/// Spill the data stored on the stack into the heap with some additional capacity.
	/// 
	/// ## Safety
	/// 
	/// The caller must ensure that the data is indeed stored in the stack.
	unsafe fn spill_amortized(&mut self, len: usize, additional: usize) -> Result<(), TryReserveError> {
		debug_assert!(self.is_owned() && !self.is_spilled());

		let required_capacity = len.checked_add(additional).ok_or(TryReserveError::CapacityOverflow)?;
		let capacity = cmp::max(len * 2, cmp::max(required_capacity, N+1));

		let layout = match Layout::array::<T>(capacity) {
			Ok(layout) => layout,
			Err(_) => capacity_overflow(),
		};
		match alloc_guard(layout.size()) {
			Ok(_) => {}
			Err(_) => capacity_overflow(),
		}
		let result = self.alloc.allocate(layout);
		let dst = match result {
			Ok(ptr) => ptr,
			Err(_) => handle_alloc_error(layout),
		};

		self.data.ptr.as_ptr().copy_to_nonoverlapping(dst.as_ptr().cast(), N);

		self.set_ptr(dst);

		Ok(())
	}

	unsafe fn spill_exact(&mut self, len: usize, additional: usize) -> Result<(), TryReserveError> {
		debug_assert!(self.is_owned() && !self.is_spilled());

		let capacity = cmp::max(len.checked_add(additional).ok_or(TryReserveError::CapacityOverflow)?, N+1);

		let layout = match Layout::array::<T>(capacity) {
			Ok(layout) => layout,
			Err(_) => capacity_overflow(),
		};
		match alloc_guard(layout.size()) {
			Ok(_) => {}
			Err(_) => capacity_overflow(),
		}
		let result = self.alloc.allocate(layout);
		let dst = match result {
			Ok(ptr) => ptr,
			Err(_) => handle_alloc_error(layout),
		};

		self.data.ptr.as_ptr().copy_to_nonoverlapping(dst.as_ptr().cast(), N);

		self.set_ptr(dst);

		Ok(())
	}

	/// Grow the memory allocated on the heap.
	/// 
	/// ## Safety
	/// 
	/// The caller must ensure that the data is owned and stored on the heap (spilled).
	unsafe fn grow_amortized(&mut self, len: usize, additional: usize) -> Result<(), TryReserveError> {
		// This is ensured by the calling contexts.
		debug_assert!(additional > 0);
		debug_assert!(self.is_spilled());

		if mem::size_of::<T>() == 0 {
			// Since we return a capacity of `usize::MAX` when `elem_size` is
			// 0, getting to here necessarily means the `RawVec` is overfull.
			return Err(TryReserveError::CapacityOverflow);
		}

		// Nothing we can really do about these checks, sadly.
		let required_cap = len.checked_add(additional).ok_or(TryReserveError::CapacityOverflow)?;

		match self.capacity() {
			Some(capacity) => {
				// This guarantees exponential growth. The doubling cannot overflow
				// because `cap <= isize::MAX` and the type of `cap` is `usize`.
				let cap = cmp::max(capacity * 2, required_cap);

				// Tiny Vecs are dumb. Skip to:
				// - 8 if the element size is 1, because any heap allocators is likely
				//   to round up a request of less than 8 bytes to at least 8 bytes.
				// - 4 if elements are moderate-sized (<= 1 KiB).
				// - 1 otherwise, to avoid wasting too much space for very short Vecs.
				// Note that `min_non_zero_cap` is computed statically.
				let elem_size = mem::size_of::<T>();
				let min_non_zero_cap = if elem_size == 1 {
					8
				} else if elem_size <= 1024 {
					4
				} else {
					1
				};
				let cap = cmp::max(min_non_zero_cap, cap);

				let new_layout = Layout::array::<T>(cap);

				// `finish_grow` is non-generic over `T`.
				let ptr = finish_grow(new_layout, self.current_memory(), &mut self.alloc)?;
				self.set_ptr(ptr);
				Ok(())
			},
			None => {
				panic!("cannot grow a borrowed slice")
			}
		}
	}

	// The constraints on this method are much the same as those on
	// `grow_amortized`, but this method is usually instantiated less often so
	// it's less critical.
	unsafe fn grow_exact(&mut self, len: usize, additional: usize) -> Result<(), TryReserveError> {
		debug_assert!(self.is_spilled());

		if mem::size_of::<T>() == 0 {
			// Since we return a capacity of `usize::MAX` when the type size is
			// 0, getting to here necessarily means the `RawVec` is overfull.
			return Err(TryReserveError::CapacityOverflow);
		}

		let cap = len.checked_add(additional).ok_or(TryReserveError::CapacityOverflow)?;
		let new_layout = Layout::array::<T>(cap);

		// `finish_grow` is non-generic over `T`.
		let ptr = finish_grow(new_layout, self.current_memory(), &mut self.alloc)?;
		self.set_ptr(ptr);
		Ok(())
	}

	unsafe fn shrink(&mut self, new_capacity: usize) -> Result<(), TryReserveError> {
		match self.capacity() {
			Some(capacity) => {
				assert!(new_capacity <= capacity, "Tried to shrink to a larger capacity");

				let (ptr, layout) = if let Some(mem) = self.current_memory() {
					mem
				} else {
					return Ok(())
				};

				let new_size = new_capacity * mem::size_of::<T>();

				let ptr = {
					let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
					self.alloc.shrink(ptr, layout, new_layout).map_err(|_| TryReserveError::AllocError {
						layout: new_layout,
						non_exhaustive: (),
					})?
				};

				self.set_ptr(ptr);
				Ok(())
			},
			None => {
				panic!("Tried to shrink borrowed data")
			}
		}
	}
}

// This function is outside `CalfVec` to minimize compile times. See the comment
// above `RawVec::grow_amortized` for details. (The `A` parameter isn't
// significant, because the number of different `A` types seen in practice is
// much smaller than the number of `T` types.)
#[inline(never)]
fn finish_grow<A>(new_layout: Result<Layout, LayoutError>, current_memory: Option<(NonNull<u8>, Layout)>, alloc: &mut A) -> Result<NonNull<[u8]>, TryReserveError> where A: Allocator {
	// Check for the error here to minimize the size of `CalfVec::grow_*`.
	let new_layout = new_layout.map_err(|_| TryReserveError::CapacityOverflow)?;

	alloc_guard(new_layout.size())?;

	let memory = if let Some((ptr, old_layout)) = current_memory {
		debug_assert_eq!(old_layout.align(), new_layout.align());
		unsafe {
			// The allocator checks for alignment equality
			// intrinsics::assume(old_layout.align() == new_layout.align()); // TODO is thre a way to keep this optimisation outside of the compiler?
			alloc.grow(ptr, old_layout, new_layout)
		}
	} else {
		alloc.allocate(new_layout)
	};

	memory.map_err(|_| TryReserveError::AllocError { layout: new_layout, non_exhaustive: () })
}

// Central function for reserve error handling.
#[inline]
fn handle_reserve<T>(result: Result<T, TryReserveError>) -> T {
	match result {
		Err(TryReserveError::CapacityOverflow) => capacity_overflow(),
		Err(TryReserveError::AllocError { layout, .. }) => handle_alloc_error(layout),
		Ok(t) => t
	}
}

// We need to guarantee the following:
// * We don't ever allocate `> isize::MAX` byte-size objects.
// * We don't overflow `usize::MAX` and actually allocate too little.
//
// On 64-bit we just need to check for overflow since trying to allocate
// `> isize::MAX` bytes will surely fail. On 32-bit and 16-bit we need to add
// an extra guard for this in case we're running on a platform which can use
// all 4GB in user-space, e.g., PAE or x32.
#[inline]
fn alloc_guard(alloc_size: usize) -> Result<(), TryReserveError> {
	if usize::BITS < 64 && alloc_size > isize::MAX as usize {
		Err(TryReserveError::CapacityOverflow)
	} else {
		Ok(())
	}
}

// One central function responsible for reporting capacity overflows. This'll
// ensure that the code generation related to these panics is minimal as there's
// only one location which panics rather than a bunch throughout the module.
fn capacity_overflow() -> ! {
	panic!("capacity overflow");
}