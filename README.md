# Web-RWKV-FFI

Simple FFI for [`web-rwkv`](https://github.com/cryscan/web-rwkv).

## APIs

The FFI exports the following APIs:

```rust
pub struct Sampler {
    pub temp: f32,
    pub top_p: f32,
    pub top_k: usize,
}

/// Initialize logger and RNG. Call this once before everything.
pub fn init(seed: u64);
/// Load a runtime.
pub fn load(model: *const c_char);
/// Clear the model state.
pub fn clear_state();
/// Generate the next token prediction given the input tokens and a sampler.
pub fn infer(tokens: *const u16, len: usize, sampler: Sampler) -> u16;
```
