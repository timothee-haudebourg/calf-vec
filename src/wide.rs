#[derive(Clone, Copy, Debug)]
pub struct Meta {
	len: usize,
	capacity: usize
}

impl crate::generic::Meta for Meta {
	const MAX_LENGTH: usize = std::usize::MAX;

	#[inline]
	fn new(len: usize, capacity: Option<usize>) -> Self {
		Meta {
			len,
			capacity: match capacity {
				Some(capacity) => capacity,
				None => 0
			}
		}
	}

	#[inline]
	fn len(&self) -> usize {
		self.len
	}

	#[inline]
	fn capacity(&self) -> Option<usize> {
		if self.capacity == 0 {
			None
		} else {
			Some(self.capacity)
		}
	}

	#[inline]
	fn set_len(&mut self, len: usize) {
		self.len = len
	}

	#[inline]
	fn set_capacity(&mut self, capacity: Option<usize>) {
		self.capacity = match capacity {
			Some(capacity) => capacity,
			None => 0
		}
	}
}

pub type CalfVec<'a, T, const N: usize> = crate::generic::CalfVec<'a, Meta, T, std::alloc::Global, N>;
pub type CalfString<'a, const N: usize> = crate::string::CalfString<'a, Meta, std::alloc::Global, N>;
