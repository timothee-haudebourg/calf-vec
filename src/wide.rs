pub struct Meta {
	len: usize,
	capacity: usize
}

impl crate::generic::Meta for Meta {
	const MAX_LENGTH: usize = std::usize::MAX;

	fn new(len: usize, capacity: Option<usize>) -> Self {
		Meta {
			len,
			capacity: match capacity {
				Some(capacity) => capacity,
				None => 0
			}
		}
	}

	fn len(&self) -> usize {
		self.len
	}

	fn capacity(&self) -> Option<usize> {
		if self.capacity == 0 {
			None
		} else {
			Some(self.capacity)
		}
	}

	fn set_len(&mut self, len: usize) {
		self.len = len
	}

	fn set_capacity(&mut self, capacity: Option<usize>) {
		self.capacity = match capacity {
			Some(capacity) => capacity,
			None => 0
		}
	}
}

pub type SmallCowVec<'a, T, const N: usize> = crate::generic::SmallCowVec<'a, Meta, T, N>;
pub type SmallCowString<'a, const N: usize> = crate::string::SmallCowString<'a, Meta, N>;
