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
            Build, ContextAutoLimits, ModelBuilder, ModelInfo, ModelRuntime, ModelVersion, State,
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

fn load_runtime(model: impl AsRef<Path>) -> Result<Runtime> {
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

        let builder = ModelBuilder::new(&context, model);
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

/// Initialize global states.
#[no_mangle]
pub extern "C" fn init() {
    let _ = simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Warn)
        .with_module_level("web_rwkv", log::LevelFilter::Info)
        .with_module_level("web_rwkv_ffi", log::LevelFilter::Info)
        .init();
}

/// Load a runtime.
#[no_mangle]
pub extern "C" fn load(model: *const c_char) {
    let model = unsafe { CStr::from_ptr(model).to_string_lossy().to_string() };
    match load_runtime(model) {
        Ok(runtime) => {
            let mut rt = RUNTIME.write().unwrap();
            rt.replace(runtime);
        }
        Err(err) => log::error!("{err}"),
    }
}

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
    let _ = runtime.state.load(0, tensor);
}

#[no_mangle]
pub extern "C" fn infer(tokens: *const u16, len: usize, sampler: Sampler) -> u16 {
    let runtime = {
        let runtime = RUNTIME.read().unwrap();
        let Some(runtime) = runtime.clone() else {
            log::error!("runtime not loaded");
            return 0;
        };
        runtime
    };

    let tokens: &[u16] = unsafe { std::slice::from_raw_parts(tokens, len) };

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
            inference.replace(input);

            if output.size() > 0 {
                let output = softmax_one(context, output).await.expect("softmax failed");
                break output.to_vec();
            }
        };
        sampler.sample(&output)
    })
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Sampler {
    temp: f32,
    top_p: f32,
    top_k: usize,
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
