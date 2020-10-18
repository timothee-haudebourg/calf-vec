pub struct Meta {
	data: usize
}

const CAP_MASK: usize = std::u32::MAX as usize;
const LEN_MASK: usize = (std::u32::MAX as usize) << 32;

impl crate::generic::Meta for Meta {
	const MAX_LENGTH: usize = std::u32::MAX as usize;

	fn new(len: usize, capacity: Option<usize>) -> Self {
		assert!(len <= Self::MAX_LENGTH);

		Meta {
			data: len << 32 | match capacity {
				Some(capacity) => {
					assert!(capacity <= Self::MAX_LENGTH);
					capacity
				},
				None => 0
			}
		}
	}

	fn len(&self) -> usize {
		self.data >> 32
	}

	fn capacity(&self) -> Option<usize> {
		let capacity = self.data & CAP_MASK;
		if capacity == 0 {
			None
		} else {
			Some(capacity)
		}
	}

	fn set_len(&mut self, len: usize) {
		assert!(len <= Self::MAX_LENGTH);
		self.data = (len << 32) | (self.data & CAP_MASK)
	}

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

pub type CalfVec<'a, T, const N: usize> = crate::generic::CalfVec<'a, Meta, T, N>;
pub type CalfString<'a, const N: usize> = crate::string::CalfString<'a, Meta, N>;
