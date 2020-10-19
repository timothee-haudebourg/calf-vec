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
pub struct FromUtf8Error<'a, M: Meta, const N: usize> {
	bytes: CalfVec<'a, M, u8, N>,
	error: str::Utf8Error
}

impl<'a, M: Meta, const N: usize> FromUtf8Error<'a, M, N> {
	/// Returns a slice of [`u8`]s bytes that were attempted to convert to a `CalfString`.
	pub fn as_bytes(&self) -> &[u8] {
		&self.bytes
	}

	/// Returns the bytes that were attempted to convert to a `CalfString`.
	///
	/// This method is carefully constructed to avoid allocation. It will
	/// consume the error, moving out the bytes, so that a copy of the bytes
	/// does not need to be made.
	pub fn into_bytes(self) -> CalfVec<'a, M, u8, N> {
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

pub struct CalfString<'a, M: Meta, const N: usize> {
	/// Internal bytes buffer.
	vec: CalfVec<'a, M, u8, N>
}

impl<'a, M: Meta, const N: usize> CalfString<'a, M, N> {
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
