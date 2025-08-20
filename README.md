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

pub struct StateRaw {
    pub len: usize,
    pub data: *mut f32,
}

/// Initialize logger and RNG. Call this once before everything.
pub fn init(seed: u64);
/// Set the RNG seed.
pub fn seed(seed: u64);
/// Load a runtime.
pub fn load(model: *const c_char, quant: usize, quant_nf4: usize, quant_sf4: usize, fp16: bool);
/// Load a prefab model.
pub fn load_prefab(model: *const c_char, fp16: bool);
/// Load a model with rescale.
pub fn load_with_rescale(model: *const c_char, quant: usize, quant_nf4: usize, quant_sf4: usize, rescale: usize, fp16: bool);
/// Load an extended model (for Othello and other demos).
pub fn load_extended(model: *const c_char, quant: usize, quant_nf4: usize, quant_sf4: usize, fp16: bool);
/// Clear the model state.
pub fn clear_state();
/// Get the model state.
pub fn get_state() -> StateRaw;
/// Set the model state.
pub fn set_state(data: StateRaw);
/// Free the model state.
pub fn free_state(state: StateRaw);
/// Generate the next token prediction given the input tokens and a sampler.
pub fn infer(tokens: *const u32, len: usize, sampler: Sampler) -> u32;
/// Compute the model's raw output (next token prediction only) given the input tokens.
pub fn infer_raw_last(tokens: *const u32, len: usize) -> ModelOutput;
/// Compute the model's raw output (predictions of all tokens) given the input tokens.
pub fn infer_raw_full(tokens: *const u32, len: usize) -> ModelOutput;
/// Delete the model output vector created by the infer functions.
pub fn free_raw(output: ModelOutput);
// Returns the model info.
pub fn get_model_info() -> ModelInfoOutput;
// Release the model.
pub fn release();
```
