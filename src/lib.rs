use std::{
    ffi::{c_char, CStr},
    path::Path,
    sync::{Arc, RwLock},
};

use anyhow::Result;
use half::f16;
use itertools::Itertools;
use memmap2::Mmap;
use safetensors::SafeTensors;
use tokio::fs::File;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        infer::{InferInput, InferInputBatch, InferOption, InferOutput},
        loader::Loader,
        model::{
            Build, ContextAutoLimits, ModelBuilder, ModelInfo, ModelRuntime, ModelVersion, Quant,
            State,
        },
        softmax::softmax_one,
        v4, v5, v6, JobRuntime,
    },
    wgpu,
};

static RUNTIME: RwLock<Option<Runtime>> = RwLock::new(None);

#[derive(Clone)]
struct Runtime {
    runtime: JobRuntime<InferInput, InferOutput>,
    state: Arc<dyn State + Sync + Send + 'static>,
    context: Context,
    tokio: Arc<tokio::runtime::Runtime>,
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
    rescale: Option<usize>,
) -> Result<Runtime> {
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
            .collect();

        let builder = ModelBuilder::new(&context, model).quant(quant);
        let builder = match rescale {
            Some(rescale) => builder.rescale(rescale),
            None => builder,
        };
        let runtime = match info.version {
            ModelVersion::V4 => {
                let model = Build::<v4::Model>::build(builder).await?;
                let builder = v4::ModelRuntime::<f16>::new(model, 1);
                let state = Arc::new(builder.state());
                let runtime = JobRuntime::new(builder).await;
                Runtime {
                    runtime,
                    state,
                    context,
                    tokio,
                }
            }
            ModelVersion::V5 => {
                let model = Build::<v5::Model>::build(builder).await?;
                let builder = v5::ModelRuntime::<f16>::new(model, 1);
                let state = Arc::new(builder.state());
                let runtime = JobRuntime::new(builder).await;
                Runtime {
                    runtime,
                    state,
                    context,
                    tokio,
                }
            }
            ModelVersion::V6 => {
                let model = Build::<v6::Model>::build(builder).await?;
                let builder = v6::ModelRuntime::<f16>::new(model, 1);
                let state = Arc::new(builder.state());
                let runtime = JobRuntime::new(builder).await;
                Runtime {
                    runtime,
                    state,
                    context,
                    tokio,
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
pub unsafe extern "C" fn load(model: *const c_char, quant: usize, quant_nf4: usize) {
    let model = unsafe { CStr::from_ptr(model).to_string_lossy().to_string() };
    match load_runtime(model, quant, quant_nf4, None) {
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
    rescale: usize,
) {
    let model = unsafe { CStr::from_ptr(model).to_string_lossy().to_string() };
    match load_runtime(model, quant, quant_nf4, Some(rescale)) {
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
pub unsafe extern "C" fn infer(tokens: *const u16, len: usize, sampler: Sampler) -> u16 {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return 0;
        };
        runtime
    };

    let tokens: &[u16] = unsafe { std::slice::from_raw_parts(tokens, len) };
    if tokens.is_empty() {
        log::error!("input cannot be empty");
        return 0;
    }

    let tokio = runtime.tokio.clone();
    tokio.block_on(async move {
        let context = &runtime.context;
        let mut inference = Some(InferInput::new(
            vec![InferInputBatch {
                tokens: tokens.to_vec(),
                option: InferOption::Last,
            }],
            128,
        ));
        let output = loop {
            let input = inference.take().unwrap();
            let (input, InferOutput(output)) = runtime.runtime.infer(input).await;
            let output = output[0].0.clone();

            if input.batches[0].tokens.is_empty() {
                let output = softmax_one(context, output).await.expect("softmax failed");
                break output.to_vec();
            }
            inference.replace(input);
        };
        sampler.sample(&output)
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

/// Delete the model output vector created by the infer functions.
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
pub unsafe extern "C" fn infer_raw_last(tokens: *const u16, len: usize) -> ModelOutput {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return ModelOutput::empty();
        };
        runtime
    };

    let tokens: &[u16] = unsafe { std::slice::from_raw_parts(tokens, len) };
    if tokens.is_empty() {
        log::error!("input cannot be empty");
        return ModelOutput::empty();
    }

    let tokio = runtime.tokio.clone();
    let output = tokio.block_on(async move {
        let mut inference = Some(InferInput::new(
            vec![InferInputBatch {
                tokens: tokens.to_vec(),
                option: InferOption::Last,
            }],
            128,
        ));
        loop {
            let input = inference.take().unwrap();
            let (input, InferOutput(output)) = runtime.runtime.infer(input).await;
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
pub unsafe extern "C" fn infer_raw_full(tokens: *const u16, len: usize) -> ModelOutput {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return ModelOutput::empty();
        };
        runtime
    };

    let tokens: &[u16] = unsafe { std::slice::from_raw_parts(tokens, len) };
    if tokens.is_empty() {
        log::error!("input cannot be empty");
        return ModelOutput::empty();
    }

    let tokio = runtime.tokio.clone();
    let output = tokio.block_on(async move {
        let mut inference = Some(InferInput::new(
            vec![InferInputBatch {
                tokens: tokens.to_vec(),
                option: InferOption::Full,
            }],
            128,
        ));
        let mut outputs = vec![];
        loop {
            let input = inference.take().unwrap();
            let (input, InferOutput(output)) = runtime.runtime.infer(input).await;
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
    pub fn sample(&self, probs: &[f32]) -> u16 {
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
        token as u16
    }
}
