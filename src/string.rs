use std::{
	str,
	ops::{
		Deref,
		DerefMut
	}
};
use crate::generic::{
	Meta,
	CalfVec
};

pub struct FromUtf8Error<'a, M: Meta, const N: usize> {
	bytes: CalfVec<'a, M, u8, N>,
	error: str::Utf8Error
}

impl<'a, M: Meta, const N: usize> FromUtf8Error<'a, M, N> {
	pub fn bytes(&self) -> &[u8] {
		&self.bytes
	}

	pub fn utf8_error(&self) -> str::Utf8Error {
		self.error
	}
}

pub struct CalfString<'a, M: Meta, const N: usize> {
	vec: CalfVec<'a, M, u8, N>
}

impl<'a, M: Meta, const N: usize> CalfString<'a, M, N> {
	/// Converts a vector of bytes to a `CalfString`.
	///
	/// A string ([`String`]) is made of bytes ([`u8`]), and a vector of bytes
	/// ([`Vec<u8>`]) is made of bytes, so this function converts between the
	/// two. Not all byte slices are valid `String`s, however: `String`
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
	/// # Errors
	///
	/// Returns [`Err`] if the slice is not UTF-8 with a description as to why the
	/// provided bytes are not UTF-8. The vector you moved in is also included.
	#[inline]
	pub fn from_utf8<V: Into<CalfVec<'a, M, u8, N>>>(vec: V) -> Result<CalfString<'a, M, N>, FromUtf8Error<'a, M, N>> {
		let vec = vec.into();
		match str::from_utf8(&vec) {
			Ok(..) => Ok(CalfString { vec }),
			Err(e) => Err(FromUtf8Error { bytes: vec, error: e }),
		}
	}

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

	/// Appends the given [`char`] to the end of this `CalfString`.
	#[inline]
	pub fn push(&mut self, ch: char) {
		match ch.len_utf8() {
			1 => self.vec.push(ch as u8),
			_ => self.vec.extend_from_slice(ch.encode_utf8(&mut [0; 4]).as_bytes()),
		}
	}

	/// Returns a byte slice of this `String`'s contents.
	///
	/// The inverse of this method is [`from_utf8`].
	///
	/// [`from_utf8`]: String::from_utf8
	#[inline]
	pub fn as_bytes(&self) -> &[u8] {
		&self.vec
	}

	/// Shortens this `String` to the specified length.
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

impl<'a, M: Meta, const N: usize> Deref for CalfString<'a, M, N> {
	type Target = str;

	#[inline]
	fn deref(&self) -> &str {
		unsafe {
			std::str::from_utf8_unchecked(&self.vec)
		}
	}
}

impl<'a, M: Meta, const N: usize> DerefMut for CalfString<'a, M, N> {
	#[inline]
	fn deref_mut(&mut self) -> &mut str {
		unsafe {
			std::str::from_utf8_unchecked_mut(&mut self.vec)
		}
	}
}
