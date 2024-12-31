#ifdef _WIN32
typedef unsigned __int64 uintptr_t;
#else
typedef unsigned long uintptr_t;
#endif
typedef unsigned long long uint64_t;
typedef unsigned short uint16_t;

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
void load(const char *model, uintptr_t quant, uintptr_t quant_nf4);

void load_with_rescale(const char *model, uintptr_t quant, uintptr_t quant_nf4, uintptr_t rescale);

/// Clear the model state.
void clear_state();

/// Generate the next token prediction given the input tokens and a sampler.
///
/// # Safety
///
/// The caller must ensure that `tokens` is valid and `len` does not exceed the actual length of `tokens`.
uint16_t infer(const uint16_t *tokens,
               uintptr_t len,
               struct Sampler sampler);

/// Delete the model output vector created by the infer functions.
void free_raw(struct ModelOutput output);

/// Compute the model's raw output (next token prediction only) given the input tokens.
///
/// # Safety
///
/// The caller must ensure that `tokens` is valid and `len` does not exceed the actual length of `tokens`.
struct ModelOutput infer_raw_last(const uint16_t *tokens, uintptr_t len);

/// Compute the model's raw output (predictions of all tokens) given the input tokens.
///
/// # Safety
///
/// The caller must ensure that `tokens` is valid and `len` does not exceed the actual length of `tokens`.
struct ModelOutput infer_raw_all(const uint16_t *tokens, uintptr_t len);

#ifdef __cplusplus
} // extern "C"
#endif