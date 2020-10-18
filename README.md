# Small copy-on-write arrays for Rust

<table><tr>
	<td><a href="https://docs.rs/calf-vec">Documentation</a></td>
	<td><a href="https://crates.io/crates/calf-vec">Crate informations</a></td>
	<td><a href="https://github.com/timothee-haudebourg/calf-vec">Repository</a></td>
</tr></table>

This crate provides the `CalfVec` data structure for small copy-on-write arrays.
As long as the data is not written to, it is only borrowed.
When owned, the data is stored on the stack as long as it is small enough.
Data is only moved on the heap as a last resort.
This is basically the intersection between
[`SmallVec`](https://crates.io/crates/smallvec) and
[`Cow`](https://doc.rust-lang.org/std/borrow/enum.Cow.html) ("small cow" = "calf").
Additionally this crate provides a `CalfString` for small copy-on-write strings
based on `CalfVec`.

## Basic usage

TODO

## License

Licensed under either of

 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
