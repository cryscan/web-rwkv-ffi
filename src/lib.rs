use std::{
    ffi::{c_char, CStr},
    path::Path,
    sync::{Arc, RwLock},
};

use anyhow::Result;
use half::f16;
use serde::{de::DeserializeSeed, Deserialize};
use itertools::Itertools;
use memmap2::Mmap;
use safetensors::SafeTensors;
use tokio::fs::File;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption, Token},
        loader::Loader,
        model::{
            ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant,
            State, Bundle
        },
        softmax::softmax_one,
        v4, v5, v6, v7, TokioRuntime,
    },
    num::Float,
    tensor::{ops::TensorOp, serialization::Seed},
    wgpu,
};
use ops::TensorOpExt;

mod ops;

static RUNTIME: RwLock<Option<WktvRuntime>> = RwLock::new(None);

#[derive(Clone)]
struct WktvRuntime {
    runtime: TokioRuntime<Rnn>,
    info: ModelInfo,
    state: Arc<dyn State + Sync + Send + 'static>,
    context: Context,
    tokio: Arc<tokio::runtime::Runtime>,
}

fn make_hooks_extended_v6<F: Float>(info: &ModelInfo) -> Result<v6::HookMap<F>> {
    let mut hooks = v6::HookMap::new();
    for layer in 0..info.num_layer {
        // add a custom operation before time-mix for each layer
        hooks.insert(
            v6::Hook::PreAttTimeDecayActivate(layer),
            Box::new(move |frame: v6::Frame<F>| {
                let op = TensorOp::ext_v6(&frame.buffer.time_decay, &frame.buffer.att_k)?;
                Ok(TensorOp::List(vec![op]))
            }),
        );
    }
    Ok(hooks)
}

fn make_hooks_extended_v7<F: Float>(info: &ModelInfo) -> Result<v7::HookMap<F>> {
    let mut hooks = v7::HookMap::new();
    for layer in 0..info.num_layer {
        hooks.insert(
            v7::Hook::PostAttAdapt(layer),
            Box::new(move |frame: v7::Frame<F>| {
                let op = TensorOp::affine(&frame.buffer.att_a, 2.0, 0.0)?;
                Ok(TensorOp::List(vec![op]))
            }),
        );
        hooks.insert(
            v7::Hook::PostAttControl(layer),
            Box::new(move |frame: v7::Frame<F>| {
                let op = TensorOp::ext_v7(&frame.buffer.att_w, &frame.buffer.att_a)?;
                Ok(TensorOp::List(vec![op]))
            }),
        );
    }
    Ok(hooks)
}

#[derive(Debug, Deserialize)]
struct Prefab {
    info: ModelInfo,
}

async fn create_context(info: &ModelInfo) -> Result<Context> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let context = ContextBuilder::new(adapter)
        .auto_limits(info)
        .build()
        .await?;
    Ok(context)
}

// async fn load_tokenizer(path: impl AsRef<Path>) -> Result<Tokenizer> {
//     let file = File::open(path).await?;
//     let mut reader = BufReader::new(file);
//     let mut contents = String::new();
//     reader.read_to_string(&mut contents).await?;
//     Ok(Tokenizer::new(&contents)?)
// }

fn load_runtime(
    model: impl AsRef<Path>,
    quant: usize,
    quant_nf4: usize,
    quant_sf4: usize,
    rescale: Option<usize>,
    extended: bool,
    fp16: bool,
) -> Result<WktvRuntime> {
    let tokio = Arc::new(tokio::runtime::Runtime::new()?);
    let _tokio = tokio.clone();

    _tokio.block_on(async move {
        let file = File::open(model).await?;
        let data = unsafe { Mmap::map(&file)? };

        let model = SafeTensors::deserialize(&data)?;
        let info = Loader::info(&model)?;
        log::info!("{:#?}", info);

        let context = create_context(&info).await?;
        log::info!("{:#?}", context.adapter.get_info());

        let quant = (0..quant)
            .map(|layer| (layer, Quant::Int8))
            .chain((0..quant_nf4).map(|layer| (layer, Quant::NF4)))
            .chain((0..quant_sf4).map(|layer| (layer, Quant::SF4)))
            .collect();

        let builder = ModelBuilder::new(&context, model).quant(quant);
        let builder = match rescale {
            Some(rescale) => builder.rescale(rescale),
            None => builder,
        };
        let runtime = match info.version {
            ModelVersion::V4 => {
                if fp16 {
                    let model = builder.build_v4().await?;
                    let bundle = v4::Bundle::<f16>::new(model, 1);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;
                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                } else {
                    let model = builder.build_v4().await?;
                    let bundle = v4::Bundle::<f32>::new(model, 1);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;
                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                }
            }
            ModelVersion::V5 => {
                if fp16 {
                    let model = builder.build_v5().await?;
                    let bundle = v5::Bundle::<f16>::new(model, 1);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;
                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                } else {
                    let model = builder.build_v5().await?;
                    let bundle = v5::Bundle::<f32>::new(model, 1);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;
                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                }
            }
            ModelVersion::V6 => {
                if fp16 {
                    let model = builder.build_v6().await?;
                    let bundle = match extended {
                        true => {
                            let hooks = make_hooks_extended_v6(&info)?;
                            v6::Bundle::<f16>::new_with_hooks(model, 1, hooks)
                        }
                        false => v6::Bundle::<f16>::new(model, 1),
                    };
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;
                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                } else {
                    let model = builder.build_v6().await?;
                    let bundle = match extended {
                        true => {
                            let hooks = make_hooks_extended_v6(&info)?;
                            v6::Bundle::<f32>::new_with_hooks(model, 1, hooks)
                        }
                        false => v6::Bundle::<f32>::new(model, 1),
                    };
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;
                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                }
            }
            ModelVersion::V7 => {
                if fp16 {
                    let model = builder.build_v7().await?;
                    let bundle = match extended {
                        true => {
                            let hooks = make_hooks_extended_v7(&info)?;
                            v7::Bundle::<f16>::new_with_hooks(model, 1, hooks)
                        }
                        false => v7::Bundle::<f16>::new(model, 1),
                    };
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;
                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                } else {
                    let model = builder.build_v7().await?;
                    let bundle = match extended {
                        true => {
                            let hooks = make_hooks_extended_v7(&info)?;
                            v7::Bundle::<f32>::new_with_hooks(model, 1, hooks)
                        }
                        false => v7::Bundle::<f32>::new(model, 1),
                    };
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;
                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                }
            }
        };
        Ok(runtime)
    })
}

fn load_runtime_prefab(model: impl AsRef<Path>, fp16: bool) -> Result<WktvRuntime> {

    let tokio = Arc::new(tokio::runtime::Runtime::new()?);
    let _tokio = tokio.clone();

    _tokio.block_on(async move {
        let file = File::open(model).await?;
        let data = unsafe { Mmap::map(&file)? };

        let Prefab { info } = cbor4ii::serde::from_slice::<Prefab>(&data)?;

        let reader = cbor4ii::core::utils::SliceReader::new(&data);
        let mut deserializer = cbor4ii::serde::Deserializer::new(reader);

        log::info!("{:#?}", info);
        let context = create_context(&info).await?;

        let runtime = match info.version {
            ModelVersion::V4 => {
                if fp16 {
                    let seed: Seed<_, v4::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let bundle = v4::Bundle::<f16>::new(model, 1);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;

                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                } else {
                    let seed: Seed<_, v4::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let bundle = v4::Bundle::<f32>::new(model, 1);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;

                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                }
            }
            ModelVersion::V5 => {
                if fp16 {
                    let seed: Seed<_, v5::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let bundle = v5::Bundle::<f16>::new(model, 1);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;

                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                } else {
                    let seed: Seed<_, v5::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let bundle = v5::Bundle::<f32>::new(model, 1);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;

                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                }
            }
            ModelVersion::V6 => {
                if fp16 {
                    let seed: Seed<_, v6::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let bundle = v6::Bundle::<f16>::new(model, 1);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;

                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                } else {
                    let seed: Seed<_, v6::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let bundle = v6::Bundle::<f32>::new(model, 1);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;

                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                }
            }
            ModelVersion::V7 => {
                if fp16 {
                    let seed: Seed<_, v7::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let bundle = v7::Bundle::<f16>::new(model, 1);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;

                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                } else {
                    let seed: Seed<_, v7::Model> = Seed::new(&context);
                    let model = seed.deserialize(&mut deserializer)?;
                    let bundle = v7::Bundle::<f32>::new(model, 1);
                    let state = Arc::new(bundle.state());
                    let runtime = TokioRuntime::new(bundle).await;

                    WktvRuntime {
                        runtime,
                        info,
                        state,
                        context,
                        tokio,
                    }
                }
            }
        };
        Ok(runtime)
    })
}

/// Initialize logger and RNG. Call this once before everything.
#[no_mangle]
pub extern "C" fn init(seed: u64) {
    let _ = simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .with_module_level("web_rwkv_ffi", log::LevelFilter::Info)
        .init();
    fastrand::seed(seed);
}

/// Set the RNG seed.
#[no_mangle]
pub extern "C" fn seed(seed: u64) {
    fastrand::seed(seed);
}

/// Load a runtime.
///
/// # Safety
///
/// The caller must ensure that `model` is valid.
#[no_mangle]
pub unsafe extern "C" fn load(model: *const c_char, quant: usize, quant_nf4: usize, quant_sf4: usize, fp16: bool) {
    let model = unsafe { CStr::from_ptr(model).to_string_lossy().to_string() };
    match load_runtime(model, quant, quant_nf4, quant_sf4,None, false, fp16) {
        Ok(runtime) => {
            let mut rt = RUNTIME.write().unwrap();
            rt.replace(runtime);
        }
        Err(err) => log::error!("{err}"),
    }
}

#[no_mangle]
pub unsafe extern "C" fn release() {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return;
        };
        runtime
    };
    let _ = runtime.runtime;
    let _ = runtime.context;
    let _ = runtime.tokio;
    let _ = runtime.state;
    let mut rt = RUNTIME.write().unwrap();
    rt.take();
}

/// Load a runtime from prefab.
/// 
/// # Safety
/// 
/// The caller must ensure that `model` is valid.
#[no_mangle]
pub unsafe extern "C" fn load_prefab(model: *const c_char, fp16: bool) {
    let model = unsafe { CStr::from_ptr(model).to_string_lossy().to_string() };
    match load_runtime_prefab(model, fp16) {
        Ok(runtime) => {
            let mut rt = RUNTIME.write().unwrap();
            rt.replace(runtime);
        }
        Err(err) => log::error!("{err}"),
    }
}

/// Load a runtime with `rescale` layers specified.
///
/// # Safety
///
/// The caller must ensure that `model` is valid.
#[no_mangle]
pub unsafe extern "C" fn load_with_rescale(
    model: *const c_char,
    quant: usize,
    quant_nf4: usize,
    quant_sf4: usize,
    rescale: usize,
    fp16: bool,
) {
    let model = unsafe { CStr::from_ptr(model).to_string_lossy().to_string() };
    match load_runtime(model, quant, quant_nf4, quant_sf4, Some(rescale), false, fp16) {
        Ok(runtime) => {
            let mut rt = RUNTIME.write().unwrap();
            rt.replace(runtime);
        }
        Err(err) => log::error!("{err}"),
    }
}

/// Load a runtime with extended hooks.
///
/// # Safety
///
/// The caller must ensure that `model` is valid.
#[no_mangle]
pub unsafe extern "C" fn load_extended(
    model: *const c_char,
    quant: usize,
    quant_nf4: usize,
    quant_sf4: usize,
    fp16: bool,
) {
    let model = unsafe { CStr::from_ptr(model).to_string_lossy().to_string() };
    match load_runtime(model, quant, quant_nf4, quant_sf4, None, true, fp16) {
        Ok(runtime) => {
            let mut rt = RUNTIME.write().unwrap();
            rt.replace(runtime);
        }
        Err(err) => log::error!("{err}"),
    }
}

/// Clear the model state.
#[no_mangle]
pub extern "C" fn clear_state() {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return;
        };
        runtime
    };
    let tensor = runtime.state.init();
    let _ = runtime.state.load(tensor, 0);
}

/// Generate the next token prediction given the input tokens and a sampler.
///
/// # Safety
///
/// The caller must ensure that `tokens` is valid and `len` does not exceed the actual length of `tokens`.
#[no_mangle]
pub unsafe extern "C" fn infer(tokens: *const u32, len: usize, sampler: Sampler) -> u32 {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return 0;
        };
        runtime
    };

    let tokens: Vec<Token> = unsafe { std::slice::from_raw_parts(tokens, len) }.iter().map(|t| Token::Token(*t)).collect();
    if tokens.is_empty() {
        log::error!("input cannot be empty");
        return 0;
    }

    let tokio = runtime.tokio.clone();
    tokio.block_on(async move {
        let context = &runtime.context;
        let mut inference = Some(RnnInput::new(
            vec![RnnInputBatch {
                tokens,
                option: RnnOption::Last,
            }],
            128,
        ));
        let output = loop {
            let input = inference.take().unwrap();
            let (input, output) = match runtime.runtime.infer(input).await {
                Ok(result) => result,
                Err(err) => {
                    log::error!("Inference error: {err}");
                    return 0;
                }
            };
            let output = output[0].0.clone();

            if input.batches[0].tokens.is_empty() {
                if sampler.top_k > 1 {
                    let output = softmax_one(context, output).await.expect("softmax failed");
                    break output.to_vec();
                } else {
                    break output.to_vec();
                }
            }
            inference.replace(input);
        };
        if sampler.top_k > 1 {
            sampler.sample(&output)
        } else {
            output
                .iter()
                .enumerate()
                .max_by(|(_, x), (_, y)| x.total_cmp(y))
                .unwrap()
                .0 as u32
        }

    })
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelOutput {
    pub len: usize,
    pub data: *mut f32,
}

impl ModelOutput {
    pub fn empty() -> ModelOutput {
        ModelOutput::from(vec![])
    }
}

impl From<Vec<f32>> for ModelOutput {
    fn from(value: Vec<f32>) -> Self {
        let mut value = std::mem::ManuallyDrop::new(value);
        let len = value.len();
        let data = value.as_mut_ptr();
        ModelOutput { data, len }
    }
}
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateRaw {
    pub len: usize,
    pub data: *mut f32,
}

impl StateRaw {
    pub fn empty() -> StateRaw {
        StateRaw::from(vec![])
    }
}

impl From<Vec<f32>> for StateRaw {
    fn from(value: Vec<f32>) -> Self {
        let mut value = std::mem::ManuallyDrop::new(value);
        let len = value.len();
        let data = value.as_mut_ptr();
        StateRaw { data, len }
    }
}

/// Get the model state.
#[no_mangle]
pub extern "C" fn get_state() -> StateRaw {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return StateRaw::empty();
        };
        runtime
    };
    let tokio = runtime.tokio.clone();
    let tensor = tokio.block_on(async move {
        runtime.state.back(0).await.map_err(|err| log::error!("{err}"))
    }).unwrap();
    tensor.to_vec().into()
}

/// Free the returned state vector created by the get_state function.
#[no_mangle]
pub extern "C" fn free_state(state: StateRaw) {
    let x = unsafe { std::slice::from_raw_parts_mut(state.data, state.len) };
    let x = x.as_mut_ptr();
    let _ = unsafe { Box::from_raw(x) };
}

/// Set the model state.
#[no_mangle]
pub extern "C" fn set_state(data: StateRaw) {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return;
        };
        runtime
    };
    let tokio = runtime.tokio.clone();
    tokio.block_on(async move {
            let shape = runtime.state.init_shape();
            let state = unsafe { std::slice::from_raw_parts(data.data, data.len) };
            let state: web_rwkv::tensor::Tensor<web_rwkv::tensor::Cpu<f32>, f32> = runtime.context.tensor_from_data(shape, state.to_vec()).unwrap();
            let _ = runtime.state.load(state, 0);
        },
    );
}

/// Delete the model output vector created by the infer functions.
#[no_mangle]
pub extern "C" fn free_raw(output: ModelOutput) {
    let x = unsafe { std::slice::from_raw_parts_mut(output.data, output.len) };
    let x = x.as_mut_ptr();
    let _ = unsafe { Box::from_raw(x) };
}

/// Compute the model's raw output (next token prediction only) given the input tokens.
///
/// # Safety
///
/// The caller must ensure that `tokens` is valid and `len` does not exceed the actual length of `tokens`.
#[no_mangle]
pub unsafe extern "C" fn infer_raw_last(tokens: *const u32, len: usize) -> ModelOutput {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return ModelOutput::empty();
        };
        runtime
    };

    let tokens: Vec<Token> = unsafe { std::slice::from_raw_parts(tokens, len) }.iter().map(|t| Token::Token(*t)).collect();
    if tokens.is_empty() {
        log::error!("input cannot be empty");
        return ModelOutput::empty();
    }

    let tokio = runtime.tokio.clone();
    let output = tokio.block_on(async move {
        let mut inference = Some(RnnInput::new(
            vec![RnnInputBatch {
                tokens: tokens.to_vec(),
                option: RnnOption::Last,
            }],
            128,
        ));
        loop {
            let input = inference.take().unwrap();
            let (input, output) = match runtime.runtime.infer(input).await {
                Ok(result) => result,
                Err(err) => {
                    log::error!("Inference error: {err}");
                    break vec![];
                }
            };
            let output = output[0].0.clone();

            if input.batches[0].tokens.is_empty() {
                break output.to_vec();
            }
            inference.replace(input);
        }
    });

    output.into()
}

/// Compute the model's raw output (predictions of all tokens) given the input tokens.
///
/// # Safety
///
/// The caller must ensure that `tokens` is valid and `len` does not exceed the actual length of `tokens`.
#[no_mangle]
pub unsafe extern "C" fn infer_raw_full(tokens: *const u32, len: usize) -> ModelOutput {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return ModelOutput::empty();
        };
        runtime
    };

    let tokens: Vec<Token> = unsafe { std::slice::from_raw_parts(tokens, len) }.iter().map(|t| Token::Token(*t)).collect();
    if tokens.is_empty() {
        log::error!("input cannot be empty");
        return ModelOutput::empty();
    }

    let tokio = runtime.tokio.clone();
    let output = tokio.block_on(async move {
        let mut inference = Some(RnnInput::new(
            vec![RnnInputBatch {
                tokens: tokens.to_vec(),
                option: RnnOption::Full,
            }],
            128,
        ));
        let mut outputs = vec![];
        loop {
            let input = inference.take().unwrap();
            let (input, output) = match runtime.runtime.infer(input).await {
                Ok(result) => result,
                Err(err) => {
                    log::error!("Inference error: {err}");
                    break;
                }
            };
            let mut output = output[0].0.clone().to_vec();
            outputs.append(&mut output);

            if input.batches[0].tokens.is_empty() {
                break;
            }
            inference.replace(input);
        }
        outputs
    });

    output.into()
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Sampler {
    pub temp: f32,
    pub top_p: f32,
    pub top_k: usize,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            temp: 1.0,
            top_p: 0.5,
            top_k: 128,
        }
    }
}

impl Sampler {
    pub fn sample(&self, probs: &[f32]) -> u32 {
        let sorted: Vec<_> = probs
            .iter()
            .copied()
            .enumerate()
            .sorted_unstable_by(|(_, x), (_, y)| x.total_cmp(y).reverse())
            .take(self.top_k.max(1))
            .scan((0, 0.0, 0.0), |(_, cum, _), (id, x)| {
                if *cum > self.top_p {
                    None
                } else {
                    *cum += x;
                    Some((id, *cum, x))
                }
            })
            .map(|(id, _, x)| (id, x.powf(1.0 / self.temp)))
            .collect();

        let sum: f32 = sorted.iter().map(|(_, x)| x).sum();
        let sorted: Vec<_> = sorted
            .into_iter()
            .map(|(id, x)| (id, x / sum))
            .scan((0, 0.0), |(_, cum), (id, x)| {
                *cum += x;
                Some((id, *cum))
            })
            .collect();

        let rand = fastrand::f32();
        let token = sorted
            .into_iter()
            .find_or_first(|&(_, cum)| rand <= cum)
            .map(|(id, _)| id)
            .unwrap_or_default();
        token as u32
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ModelInfoOutput {
    pub version: usize,
    pub num_layer: usize,
    pub num_hidden: usize,
    pub num_emb: usize,
    pub num_vocab: usize,
    pub num_head: usize,
}

impl Default for ModelInfoOutput {
    fn default() -> Self {
        Self {
            version: 0,
            num_layer: 0,
            num_hidden: 0,
            num_emb: 0,
            num_vocab: 0,
            num_head: 0,
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn get_model_info() -> ModelInfoOutput {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return ModelInfoOutput::default();
        };
        runtime
    };

    let info = runtime.info;
    ModelInfoOutput {
        version: match info.version {
            ModelVersion::V4 => 4,
            ModelVersion::V5 => 5,
            ModelVersion::V6 => 6,
            ModelVersion::V7 => 7,
        },
        num_layer: info.num_layer,
        num_hidden: info.num_hidden,
        num_emb: info.num_emb,
        num_vocab: info.num_vocab,
        num_head: info.num_head,
    }
}
