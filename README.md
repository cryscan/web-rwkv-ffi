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

pub struct ModelOutput {
    pub len: usize,
    pub data: *mut f32,
}

pub struct ModelInfoOutput {
    pub version: usize,
    pub num_layer: usize,
    pub num_hidden: usize,
    pub num_emb: usize,
    pub num_vocab: usize,
    pub num_head: usize,
}

/// Initialize logger and RNG. Call this once before everything.
pub fn init(seed: u64);
/// Set the RNG seed.
pub fn seed(seed: u64);
/// Load a runtime.
pub fn load(model: *const c_char, quant: usize, quant_nf4: usize);
/// Clear the model state.
pub fn clear_state();
/// Generate the next token prediction given the input tokens and a sampler.
pub fn infer(tokens: *const u16, len: usize, sampler: Sampler) -> u16;
/// Compute the model's raw output (next token prediction only) given the input tokens.
pub fn infer_raw_last(tokens: *const u16, len: usize) -> ModelOutput;
/// Compute the model's raw output (predictions of all tokens) given the input tokens.
pub fn infer_raw_full(tokens: *const u16, len: usize) -> ModelOutput;
/// Delete the model output vector created by the infer functions.
pub fn free_raw(output: ModelOutput);
// Returns the model info.
pub fn get_model_info() -> ModelInfoOutput;
```
