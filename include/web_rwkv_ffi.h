#include "stdint.h"

struct Sampler
{
  float temp;
  float top_p;
  uintptr_t top_k;
};

struct ModelOutput {
  uintptr_t len;
  float *logits;
};

struct ModelInfoOutput {
  uintptr_t version;
  uintptr_t num_layer;
  uintptr_t num_hidden;
  uintptr_t num_emb;
  uintptr_t num_vocab;
  uintptr_t num_head;
};

struct StateRaw {
  uintptr_t len;
  float *state;
};

#ifdef __cplusplus
extern "C" {
#endif
/// Initialize logger and RNG. Call this once before everything.
void init(uint64_t seed);

/// Set the RNG seed.
void seed(uint64_t seed);

/// Load a runtime.
///
/// # Safety
///
/// The caller must ensure that `model` is valid.
void load(const char *model, uintptr_t quant, uintptr_t quant_nf4, uintptr_t quant_sf4, bool fp16);

void load_prefab(const char *model, bool fp16);

void load_extended(const char *model, uintptr_t quant, uintptr_t quant_nf4, uintptr_t quant_sf4, bool fp16);

void load_with_rescale(const char *model, uintptr_t quant, uintptr_t quant_nf4, uintptr_t quant_sf4, uintptr_t rescale, bool fp16);

void release();

/// Clear the model state.
void clear_state();

/// Generate the next token prediction given the input tokens and a sampler.
///
/// # Safety
///
/// The caller must ensure that `tokens` is valid and `len` does not exceed the actual length of `tokens`.
uint32_t infer(const uint32_t *tokens,
               uintptr_t len,
               struct Sampler sampler);

/// Delete the model output vector created by the infer functions.
void free_raw(struct ModelOutput output);

/// Compute the model's raw output (next token prediction only) given the input tokens.
///
/// # Safety
///
/// The caller must ensure that `tokens` is valid and `len` does not exceed the actual length of `tokens`.
struct ModelOutput infer_raw_last(const uint32_t *tokens, uintptr_t len);

/// Compute the model's raw output (predictions of all tokens) given the input tokens.
///
/// # Safety
///
/// The caller must ensure that `tokens` is valid and `len` does not exceed the actual length of `tokens`.
struct ModelOutput infer_raw_all(const uint32_t *tokens, uintptr_t len);

struct ModelInfoOutput get_model_info();

struct StateRaw get_state();

void set_state(struct StateRaw state);

void free_state(struct StateRaw state);

#ifdef __cplusplus
} // extern "C"
#endif