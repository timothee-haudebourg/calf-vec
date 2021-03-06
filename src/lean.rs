#[derive(Clone, Copy, Debug)]
pub struct Meta {
	data: usize
}

const CAP_MASK: usize = std::u32::MAX as usize;
const LEN_MASK: usize = (std::u32::MAX as usize) << 32;

unsafe impl crate::raw::Meta for Meta {
	const MAX_LENGTH: usize = std::u32::MAX as usize;

	#[inline]
	fn with_capacity(capacity: Option<usize>) -> Self {
		Meta {
			data: match capacity {
				Some(capacity) => {
					assert!(capacity <= Self::MAX_LENGTH);
					capacity
				},
				None => 0
			}
		}
	}

	#[inline]
	fn capacity(&self) -> Option<usize> {
		let capacity = self.data & CAP_MASK;
		if capacity == 0 {
			None
		} else {
			Some(capacity)
		}
	}

	#[inline]
	fn set_capacity(&mut self, capacity: Option<usize>) {
		self.data = match capacity {
			Some(capacity) => {
				assert!(capacity <= Self::MAX_LENGTH);
				(self.data & LEN_MASK) | capacity
			},
			None => {
				self.data & LEN_MASK
			}
		}
	}
}

unsafe impl crate::generic::Meta for Meta {
	#[inline]
	fn new(len: usize, capacity: Option<usize>) -> Self {
		assert!(len <= <Self as crate::raw::Meta>::MAX_LENGTH);

		Meta {
			data: len << 32 | match capacity {
				Some(capacity) => {
					assert!(capacity <= <Self as crate::raw::Meta>::MAX_LENGTH);
					capacity
				},
				None => 0
			}
		}
	}

	#[inline]
	fn len(&self) -> usize {
		self.data >> 32
	}

	#[inline]
	fn set_len(&mut self, len: usize) {
		assert!(len <= <Self as crate::raw::Meta>::MAX_LENGTH);
		self.data = (len << 32) | (self.data & CAP_MASK)
	}
}

pub type CalfVec<'a, T, const N: usize> = crate::generic::CalfVec<'a, Meta, T, std::alloc::Global, N>;
pub type CalfString<'a, const N: usize> = crate::string::CalfString<'a, Meta, std::alloc::Global, N>;
