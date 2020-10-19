# Small copy-on-write arrays for Rust

<table><tr>
	<td><a href="https://docs.rs/calf-vec">Documentation</a></td>
	<td><a href="https://crates.io/crates/calf-vec">Crate informations</a></td>
	<td><a href="https://github.com/timothee-haudebourg/calf-vec">Repository</a></td>
</tr></table>

This crate provides the
[`CalfVec`](https://docs.rs/calf-vec/latest/calf_vec/generic/struct.CalfVec.html)
data structure for small copy-on-write arrays.
As long as the data is not written to, it is only borrowed.
When owned, the data is stored on the stack as long as it is small enough.
Data is only moved on the heap as a last resort.
This is basically the intersection between
[`SmallVec`](https://crates.io/crates/smallvec) and
[`Cow`](https://doc.rust-lang.org/std/borrow/enum.Cow.html) (`Small` + `Cow` = `Calf`).
Additionally this crate provides a
[`CalfString`](https://docs.rs/calf-vec/latest/calf_vec/string/struct.CalfString.html)
for small copy-on-write strings
based on `CalfVec`.

## Basic usage

A `CalfVec` either borrows or owns its data.
You can start by creating a `CalfVec` from a slice.
It will only be copied when the `CalfVec` is modified.
```rust
use calf_vec::CalfVec;

let slice = &[1, 2, 3];
let mut calf: CalfVec<'_, u8, 32> = CalfVec::borrowed(slice); // at this point, data is only borrowed.
calf[0]; // => 1
calf[0] = 4; // because it is modified, the data is copied here.
assert_eq!(calf, [4, 2, 3])
```

A `CalfVec` can also be directly created to own its data:
```rust
let mut owned: CalfVec<'_, u8, 32> = CalfVec::owned(vec![1, 2, 3]);
```
Here, since the owned buffer's capacity is smaller than 32 (given as parameter),
it is stored on the stack.
It will be moved on the heap only when necessary:
```rust
owned.push(4);
owned.push(5);
// ...
owned.push(31);
owned.push(32); // <- here the buffer's capacity now exceeds the given limit (32).
                //    it is hence moved on the heap, transparently.
```

## License

Licensed under either of

 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
