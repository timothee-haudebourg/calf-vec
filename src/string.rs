use std::{
	alloc::{
		AllocRef,
		Global
	},
	str,
	ptr,
	fmt,
	ops::{
		Deref,
		DerefMut
	},
	borrow::Cow
};
use crate::generic::{
	Meta,
	CalfVec
};

/// A possible error value when converting a `CalfString` from a UTF-8 byte vector.
///
/// This type is the error type for the [`from_utf8`] method on [`CalfString`]. It
/// is designed in such a way to carefully avoid reallocations: the
/// [`into_bytes`] method will give back the byte vector that was used in the
/// conversion attempt.
///
/// [`from_utf8`]: CalfString::from_utf8
/// [`into_bytes`]: FromUtf8Error::into_bytes
///
/// The [`Utf8Error`] type provided by [`std::str`] represents an error that may
/// occur when converting a slice of [`u8`]s to a [`&str`]. In this sense, it's
/// an analogue to `FromUtf8Error`, and you can get one from a `FromUtf8Error`
/// through the [`utf8_error`] method.
///
/// [`Utf8Error`]: core::str::Utf8Error
/// [`std::str`]: core::str
/// [`&str`]: prim@str
/// [`utf8_error`]: Self::utf8_error
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FromUtf8Error<'a, M: Meta, A: AllocRef, const N: usize> {
	bytes: CalfVec<'a, M, u8, A, N>,
	error: str::Utf8Error
}

impl<'a, M: Meta, A: AllocRef, const N: usize> FromUtf8Error<'a, M, A, N> {
	/// Returns a slice of [`u8`]s bytes that were attempted to convert to a `CalfString`.
	pub fn as_bytes(&self) -> &[u8] {
		&self.bytes
	}

	/// Returns the bytes that were attempted to convert to a `CalfString`.
	///
	/// This method is carefully constructed to avoid allocation. It will
	/// consume the error, moving out the bytes, so that a copy of the bytes
	/// does not need to be made.
	pub fn into_bytes(self) -> CalfVec<'a, M, u8, A, N> {
		self.bytes
	}

	/// Fetch a `Utf8Error` to get more details about the conversion failure.
	///
	/// The [`Utf8Error`] type provided by [`std::str`] represents an error that may
	/// occur when converting a slice of [`u8`]s to a [`&str`]. In this sense, it's
	/// an analogue to `FromUtf8Error`. See its documentation for more details
	/// on using it.
	///
	/// [`Utf8Error`]: std::str::Utf8Error
	/// [`std::str`]: core::str
	/// [`&str`]: prim@str
	pub fn utf8_error(&self) -> str::Utf8Error {
		self.error
	}
}

type FromUtf16Error = std::string::FromUtf16Error;

pub struct CalfString<'a, M: Meta, A: AllocRef, const N: usize> {
	/// Internal bytes buffer.
	vec: CalfVec<'a, M, u8, A, N>
}

impl<'a, M: Meta, const N: usize> CalfString<'a, M, Global, N> {
	/// Creates a new empty `String`.
	// TODO make this function `const` as soon as the `const_fn` feature allows it.
	#[inline]
	pub fn new() -> CalfString<'a, M, Global, N> {
		CalfString { vec: CalfVec::new() }
	}

	/// Creates a new empty `CalfString` with a particular capacity.
	#[inline]
	pub fn with_capacity(capacity: usize) -> CalfString<'a, M, Global, N> {
		CalfString { vec: CalfVec::with_capacity(capacity) }
	}

	/// Converts a vector of bytes to a `CalfString` without checking that the
	/// string contains valid UTF-8.
	///
	/// See the safe version, [`from_utf8`], for more details.
	///
	/// [`from_utf8`]: CalfString::from_utf8
	///
	/// # Safety
	///
	/// This function is unsafe because it does not check that the bytes passed
	/// to it are valid UTF-8. If this constraint is violated, it may cause
	/// memory unsafety issues with future users of the `CalfString`.
	#[inline]
	pub unsafe fn from_utf8_unchecked<B: Into<CalfVec<'a, M, u8, Global, N>>>(bytes: B) -> CalfString<'a, M, Global, N> {
		CalfString { vec: bytes.into() }
	}

	/// Converts a vector of bytes to a `CalfString`.
	///
	/// A string is made of bytes ([`u8`]), and a vector of bytes is made of bytes,
	/// so this function converts between the
	/// two. Not all byte slices are valid strings, however: `CalfString`
	/// requires that it is valid UTF-8. `from_utf8()` checks to ensure that
	/// the bytes are valid UTF-8, and then does the conversion.
	///
	/// If you are sure that the byte slice is valid UTF-8, and you don't want
	/// to incur the overhead of the validity check, there is an unsafe version
	/// of this function, [`from_utf8_unchecked`], which has the same behavior
	/// but skips the check.
	///
	/// This method will take care to not copy the vector, for efficiency's
	/// sake.
	///
	/// The inverse of this method is [`into_bytes`].
	///
	/// [`from_utf8_unchecked`]: CalfString::from_utf8_unchecked
	/// [`into_bytes`]: CalfString::into_bytes
	///
	/// # Errors
	///
	/// Returns [`Err`] if the slice is not UTF-8 with a description as to why the
	/// provided bytes are not UTF-8. The vector you moved in is also included.
	#[inline]
	pub fn from_utf8<B: Into<CalfVec<'a, M, u8, Global, N>>>(bytes: B) -> Result<CalfString<'a, M, Global, N>, FromUtf8Error<'a, M, Global, N>> {
		let vec = bytes.into();
		match str::from_utf8(&vec) {
			Ok(..) => Ok(CalfString { vec }),
			Err(e) => Err(FromUtf8Error { bytes: vec, error: e }),
		}
	}

	/// Converts a slice of bytes to a string, including invalid characters.
	///
	/// Strings are made of bytes ([`u8`]), and a slice of bytes
	/// ([`&[u8]`][byteslice]) is made of bytes, so this function converts
	/// between the two. Not all byte slices are valid strings, however: strings
	/// are required to be valid UTF-8. During this conversion,
	/// `from_utf8_lossy()` will replace any invalid UTF-8 sequences with
	/// [`U+FFFD REPLACEMENT CHARACTER`][U+FFFD], which looks like this: �
	///
	/// [byteslice]: ../../std/primitive.slice.html
	/// [U+FFFD]: core::char::REPLACEMENT_CHARACTER
	///
	/// If you are sure that the byte slice is valid UTF-8, and you don't want
	/// to incur the overhead of the conversion, there is an unsafe version
	/// of this function, [`from_utf8_unchecked`], which has the same behavior
	/// but skips the checks.
	///
	/// [`from_utf8_unchecked`]: CalfString::from_utf8_unchecked
	#[inline]
	pub fn from_utf8_lossy(v: &'a [u8]) -> CalfString<'a, M, Global, N> {
		String::from_utf8_lossy(v).into()
	}

	/// Decode a UTF-16 encoded vector `v` into a `CalfString`, returning [`Err`]
	/// if `v` contains any invalid data.
	#[inline]
	pub fn from_utf16(v: &[u16]) -> Result<CalfString<'a, M, Global, N>, FromUtf16Error> {
		let str = String::from_utf16(v)?;
		Ok(str.into())
	}

	/// Decode a UTF-16 encoded slice `v` into a `String`, replacing
	/// invalid data with [the replacement character (`U+FFFD`)][U+FFFD].
	///
	/// [`from_utf8_lossy`]: CalfString::from_utf8_lossy
	/// [U+FFFD]: core::char::REPLACEMENT_CHARACTER
	#[inline]
	pub fn from_utf16_lossy(v: &[u16]) -> CalfString<'a, M, Global, N> {
		String::from_utf16_lossy(v).into()
	}
}

impl<'a, M: Meta, A: AllocRef, const N: usize> CalfString<'a, M, A, N> {
	/// Returns this `String`'s size, in bytes.
	#[inline]
	pub fn len(&self) -> usize {
		self.vec.len()
	}

	/// Returns this `String`'s capacity, in bytes.
	#[inline]
	pub fn capacity(&self) -> Option<usize> {
		self.vec.capacity()
	}

	/// Converts a `CalfString` into a [`CalfVec`] byte vector.
	///
	/// This consumes the `CalfString`, so we do not need to copy its contents.
	#[inline]
	pub fn into_bytes(self) -> CalfVec<'a, M, u8, A, N> {
		self.vec
	}

	/// Truncates this `CalfString`, removing all contents.
	///
	/// While this means the `CalfString` will have a length of zero, it does not
	/// touch its capacity.
	#[inline]
	pub fn clear(&mut self) {
		self.vec.clear()
	}

	/// Ensures that this `CalfString`'s capacity is at least `additional` bytes
	/// larger than its length.
	///
	/// The capacity may be increased by more than `additional` bytes if it
	/// chooses, to prevent frequent reallocations.
	///
	/// If you do not want this "at least" behavior, see the [`reserve_exact`]
	/// method.
	///
	/// # Panics
	///
	/// Panics if the new capacity overflows [`usize`].
	///
	/// [`reserve_exact`]: CalfString::reserve_exact
	#[inline]
	pub fn reserve(&mut self, additional: usize) {
		self.vec.reserve(additional)
	}

	/// Ensures that this `CalfString`'s capacity is `additional` bytes
	/// larger than its length.
	///
	/// Consider using the [`reserve`] method unless you absolutely know
	/// better than the allocator.
	///
	/// [`reserve`]: CalfString::reserve
	///
	/// # Panics
	///
	/// Panics if the new capacity overflows `usize`.
	#[inline]
	pub fn reserve_exact(&mut self, additional: usize) {
		self.vec.reserve_exact(additional)
	}

	/// Shrinks the capacity of this `CalfString` to match its length.
	#[inline]
	pub fn shrink_to_fit(&mut self) where A: Clone {
		// TODO remove the `Clone` bound in `A` by not using `Vec::from_raw_parts_in` in `CalfVec::shrink_to`.
		self.vec.shrink_to_fit()
	}

	/// Shrinks the capacity of this `CalfString` with a lower bound.
	///
	/// The capacity will remain at least as large as both the length
	/// and the supplied value.
	///
	/// Panics if the current capacity is smaller than the supplied
	/// minimum capacity.
	#[inline]
	pub fn shrink_to(&mut self, min_capacity: usize) where A: Clone {
		// TODO remove the `Clone` bound in `A` by not using `Vec::from_raw_parts_in` in `CalfVec::shrink_to`.
		self.vec.shrink_to(min_capacity)
	}

	unsafe fn insert_bytes(&mut self, idx: usize, bytes: &[u8]) {
		let len = self.len();
		let amt = bytes.len();
		self.vec.reserve(amt);

		ptr::copy(self.vec.as_ptr().add(idx), self.vec.as_mut_ptr().add(idx + amt), len - idx);
		ptr::copy(bytes.as_ptr(), self.vec.as_mut_ptr().add(idx), amt);
		self.vec.set_len(len + amt);
	}

	/// Inserts a character into this `CalfString` at a byte position.
	///
	/// This is an *O*(*n*) operation as it requires copying every element in the
	/// buffer.
	///
	/// # Panics
	///
	/// Panics if `idx` is larger than the `CalfString`'s length, or if it does not
	/// lie on a [`char`] boundary.
	#[inline]
	pub fn insert(&mut self, idx: usize, ch: char) {
		assert!(self.is_char_boundary(idx));
		let mut bits = [0; 4];
		let bits = ch.encode_utf8(&mut bits).as_bytes();

		unsafe {
			self.insert_bytes(idx, bits);
		}
	}

	/// Inserts a string slice into this `CalfString` at a byte position.
	///
	/// This is an *O*(*n*) operation as it requires copying every element in the
	/// buffer.
	///
	/// # Panics
	///
	/// Panics if `idx` is larger than the `CalfString`'s length, or if it does not
	/// lie on a [`char`] boundary.
	#[inline]
	pub fn insert_str(&mut self, idx: usize, string: &str) {
		assert!(self.is_char_boundary(idx));

		unsafe {
			self.insert_bytes(idx, string.as_bytes());
		}
	}

	/// Appends the given [`char`] to the end of this `CalfString`.
	#[inline]
	pub fn push(&mut self, ch: char) {
		match ch.len_utf8() {
			1 => self.vec.push(ch as u8),
			_ => self.vec.extend_from_slice(ch.encode_utf8(&mut [0; 4]).as_bytes()),
		}
	}

	/// Appends a given string slice onto the end of this `CalfString`.
	#[inline]
	pub fn push_str(&mut self, string: &str) {
		self.vec.extend_from_slice(string.as_bytes())
	}

	/// Removes a [`char`] from this `CalfString` at a byte position and returns it.
	///
	/// This is an *O*(*n*) operation, as it requires copying every element in the
	/// buffer.
	///
	/// # Panics
	///
	/// Panics if `idx` is larger than or equal to the `CalfString`'s length,
	/// or if it does not lie on a [`char`] boundary.
	#[inline]
	pub fn remove(&mut self, idx: usize) -> char {
		let ch = match self[idx..].chars().next() {
			Some(ch) => ch,
			None => panic!("cannot remove a char from the end of a string"),
		};

		let next = idx + ch.len_utf8();
		let len = self.len();
		unsafe {
			ptr::copy(self.vec.as_ptr().add(next), self.vec.as_mut_ptr().add(idx), len - next);
			self.vec.set_len(len - (next - idx));
		}
		ch
	}

	/// Removes the last character from the string buffer and returns it.
	///
	/// Returns [`None`] if this `CalfString` is empty.
	#[inline]
	pub fn pop(&mut self) -> Option<char> {
		let ch = self.chars().rev().next()?;
		let newlen = self.len() - ch.len_utf8();
		unsafe {
			self.vec.set_len(newlen);
		}
		Some(ch)
	}

	/// Retains only the characters specified by the predicate.
	///
	/// In other words, remove all characters `c` such that `f(c)` returns `false`.
	/// This method operates in place, visiting each character exactly once in the
	/// original order, and preserves the order of the retained characters.
	#[inline]
	pub fn retain<F>(&mut self, mut f: F) where F: FnMut(char) -> bool {
		let len = self.len();
		let mut del_bytes = 0;
		let mut idx = 0;

		while idx < len {
			let ch = unsafe { self.get_unchecked(idx..len).chars().next().unwrap() };
			let ch_len = ch.len_utf8();

			if !f(ch) {
				del_bytes += ch_len;
			} else if del_bytes > 0 {
				unsafe {
					ptr::copy(
						self.vec.as_ptr().add(idx),
						self.vec.as_mut_ptr().add(idx - del_bytes),
						ch_len,
					);
				}
			}

			// Point idx to the next char
			idx += ch_len;
		}

		if del_bytes > 0 {
			unsafe {
				self.vec.set_len(len - del_bytes);
			}
		}
	}

	// /// Removes the specified range in the string,
	// /// and replaces it with the given string.
	// /// The given string doesn't need to be the same length as the range.
	// ///
	// /// # Panics
	// ///
	// /// Panics if the starting point or end point do not lie on a [`char`]
	// /// boundary, or if they're out of bounds.
	// pub fn replace_range<R>(&mut self, range: R, replace_with: &str) where R: std::ops::RangeBounds<usize> {
	// 	// Memory safety
	// 	//
	// 	// Replace_range does not have the memory safety issues of a vector Splice.
	// 	// of the vector version. The data is just plain bytes.
	// 	use std::ops::Bound;
	//
	// 	match range.start_bound() {
	// 		Bound::Included(&n) => assert!(self.is_char_boundary(n)),
	// 		Bound::Excluded(&n) => assert!(self.is_char_boundary(n + 1)),
	// 		Bound::Unbounded => {}
	// 	};
	// 	match range.end_bound() {
	// 		Bound::Included(&n) => assert!(self.is_char_boundary(n + 1)),
	// 		Bound::Excluded(&n) => assert!(self.is_char_boundary(n)),
	// 		Bound::Unbounded => {}
	// 	};
	//
	// 	self.vec.splice(range, replace_with.bytes());
	// }

	/// Returns a byte slice of this `CalfString`'s contents.
	///
	/// The inverse of this method is [`from_utf8`].
	///
	/// [`from_utf8`]: CalfString::from_utf8
	#[inline]
	pub fn as_bytes(&self) -> &[u8] {
		&self.vec
	}

	/// Extracts a string slice containing the entire `CalfString`.
	#[inline]
	pub fn as_str(&self) -> &str {
		self
	}

	/// Converts a `CalfString` into a mutable string slice.
	#[inline]
	pub fn as_mut_str(&mut self) -> &mut str {
		self
	}

	/// Returns a mutable reference to the contents of this `CalfString`.
	///
	/// # Safety
	///
	/// This function is unsafe because it does not check that the bytes passed
	/// to it are valid UTF-8. If this constraint is violated, it may cause
	/// memory unsafety issues with future users of the `CalfString`, as the rest of
	/// the standard library assumes that `CalfString`s are valid UTF-8.
	#[inline]
	pub unsafe fn as_mut_vec(&mut self) -> &mut CalfVec<'a, M, u8, A, N> {
		&mut self.vec
	}

	/// Shortens this `CalfString` to the specified length.
	///
	/// If `new_len` is greater than the string's current length, this has no
	/// effect.
	///
	/// Note that this method has no effect on the allocated capacity
	/// of the string
	///
	/// # Panics
	///
	/// Panics if `new_len` does not lie on a [`char`] boundary.
	#[inline]
	pub fn truncate(&mut self, new_len: usize) {
		if new_len <= self.len() {
			assert!(self.is_char_boundary(new_len));
			self.vec.truncate(new_len)
		}
	}
}

impl<'a, M: Meta, A: AllocRef, const N: usize> Deref for CalfString<'a, M, A, N> {
	type Target = str;

	#[inline]
	fn deref(&self) -> &str {
		unsafe {
			std::str::from_utf8_unchecked(&self.vec)
		}
	}
}

impl<'a, M: Meta, A: AllocRef, const N: usize> DerefMut for CalfString<'a, M, A, N> {
	#[inline]
	fn deref_mut(&mut self) -> &mut str {
		unsafe {
			std::str::from_utf8_unchecked_mut(&mut self.vec)
		}
	}
}

impl<'a, M: Meta, const N: usize> From<&'a str> for CalfString<'a, M, Global, N> {
	#[inline]
	fn from(s: &'a str) -> CalfString<'a, M, Global, N> {
		CalfString {
			vec: s.as_bytes().into()
		}
	}
}

impl<'a, M: Meta, const N: usize> From<String> for CalfString<'a, M, Global, N> {
	#[inline]
	fn from(s: String) -> CalfString<'a, M, Global, N> {
		CalfString {
			vec: s.into_bytes().into()
		}
	}
}

impl<'a, M: Meta, const N: usize> From<Cow<'a, str>> for CalfString<'a, M, Global, N> {
	#[inline]
	fn from(c: Cow<'a, str>) -> CalfString<'a, M, Global, N> {
		match c {
			Cow::Borrowed(s) => s.into(),
			Cow::Owned(s) => s.into()
		}
	}
}

impl<'a, M: Meta, const N: usize> std::str::FromStr for CalfString<'a, M, Global, N> {
	type Err = std::convert::Infallible;

	fn from_str(s: &str) -> Result<CalfString<'a, M, Global, N>, std::convert::Infallible> {
		Ok(String::from_str(s).unwrap().into())
	}
}

impl<'a, M: Meta, A: AllocRef, const N: usize> PartialEq<str> for CalfString<'a, M, A, N> {
	#[inline]
	fn eq(&self, other: &str) -> bool {
		self.as_str() == other
	}
}

impl<'a, 'b, M: Meta, A: AllocRef, const N: usize> PartialEq<&'b str> for CalfString<'a, M, A, N> {
	#[inline]
	fn eq(&self, other: &&'b str) -> bool {
		self.as_str() == *other
	}
}

impl<'a, M: Meta, A: AllocRef, const N: usize> PartialEq<String> for CalfString<'a, M, A, N> {
	#[inline]
	fn eq(&self, other: &String) -> bool {
		self.as_str() == other
	}
}

impl<'a, M: Meta, A: AllocRef, const N: usize> fmt::Display for CalfString<'a, M, A, N> {
	#[inline]
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		fmt::Display::fmt(&**self, f)
	}
}

impl<'a, M: Meta, A: AllocRef, const N: usize> fmt::Debug for CalfString<'a, M, A, N> {
	#[inline]
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		fmt::Debug::fmt(&**self, f)
	}
}
