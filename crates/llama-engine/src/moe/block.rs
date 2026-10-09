//! Bloco esparso que conecta router, cache e executor Q4_0.
//!
//! O bloco recebe uma ativação e os logits produzidos pela camada de router.
//! A projeção do router pode vir do tensor GGUF local, sem mudar o wire entre
//! peers; a atenção continua no runtime que entrega a ativação a esta fronteira.

#[cfg(feature = "cuda")]
use super::cache::DeviceExpertLease;
use super::cache::{ExpertCache, ExpertKey};
#[cfg(feature = "cuda")]
use super::coexecution::{combine_lane_outputs, filter_route, run_overlap, split_route_experts};
#[cfg(feature = "cuda")]
use super::cuda::{CudaHeterogeneousExpert, CudaHeterogeneousTensor, CudaMoeError};
use super::format_heterogeneous::HeterogeneousMoeSource;
#[cfg(feature = "cuda")]
use super::format_heterogeneous::{HeterogeneousTensorSpec, TensorRole};
use super::format_iq_quants::{
    IqQuantError, IqQuantExecuteError, IqQuantExecutor, IqQuantFormat, IqQuantSource,
};
use super::format_k_quants::{
    KQuantError, KQuantExecuteError, KQuantExecutor, KQuantFormat, KQuantSource,
};
use super::format_legacy_quants::{
    LegacyQuantExecuteError, LegacyQuantExecutor, LegacyQuantFormat, LegacyQuantSource,
};
use super::format_low_bits::{LowBitExecuteError, LowBitExecutor, LowBitFormat, LowBitSource};
use super::format_nvfp4::{Nvfp4BackendKind, Nvfp4ExecuteError, Nvfp4Executor, Nvfp4Source};
use super::format_plain::{PlainExecuteError, PlainExecutor, PlainFormat, PlainSource};
use super::format_q4_0::{Q4Error, Q4ExecuteError, Q4_0Executor, Q4_0Source};
use super::router::{
    dispatch_route_parallel, dispatch_top_k, select_top_k, DispatchError, RouteError,
};
#[cfg(feature = "cuda")]
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::sync::Mutex;

#[derive(Debug)]
pub enum MoeBlockError {
    Route(RouteError),
    Executor(String),
    #[cfg(feature = "cuda")]
    CudaNonFinite(String),
}

impl fmt::Display for MoeBlockError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Route(error) => error.fmt(formatter),
            Self::Executor(error) => error.fmt(formatter),
            #[cfg(feature = "cuda")]
            Self::CudaNonFinite(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for MoeBlockError {}

impl From<Q4ExecuteError> for MoeBlockError {
    fn from(error: Q4ExecuteError) -> Self {
        Self::Executor(error.to_string())
    }
}

impl From<Nvfp4ExecuteError> for MoeBlockError {
    fn from(error: Nvfp4ExecuteError) -> Self {
        Self::Executor(error.to_string())
    }
}

impl From<KQuantExecuteError> for MoeBlockError {
    fn from(error: KQuantExecuteError) -> Self {
        match error {
            #[cfg(feature = "cuda")]
            KQuantExecuteError::CudaNonFinite {
                stage,
                index,
                count,
                kind,
            } => {
                Self::CudaNonFinite(format!(
                    "executor CUDA MoE produziu {count} valores não finitos ({kind}) no estágio {stage}, primeiro índice da saída {index}"
                ))
            }
            error => Self::Executor(error.to_string()),
        }
    }
}

impl From<IqQuantExecuteError> for MoeBlockError {
    fn from(error: IqQuantExecuteError) -> Self {
        Self::Executor(error.to_string())
    }
}

impl From<PlainExecuteError> for MoeBlockError {
    fn from(error: PlainExecuteError) -> Self {
        Self::Executor(error.to_string())
    }
}

impl From<LegacyQuantExecuteError> for MoeBlockError {
    fn from(error: LegacyQuantExecuteError) -> Self {
        Self::Executor(error.to_string())
    }
}

impl From<LowBitExecuteError> for MoeBlockError {
    fn from(error: LowBitExecuteError) -> Self {
        Self::Executor(error.to_string())
    }
}

fn dispatch_resolved<E, F>(
    ids: &[u32],
    weights: &[f32],
    execute: F,
) -> Result<Vec<f32>, MoeBlockError>
where
    E: fmt::Display + Send,
    F: Fn(u32) -> Result<Vec<f32>, E> + Sync,
{
    dispatch_route_parallel(ids, weights, execute).map_err(|error| match error {
        DispatchError::Route(error) => MoeBlockError::Route(error),
        DispatchError::Expert(error) => MoeBlockError::Executor(error.to_string()),
    })
}

/// Forward de um bloco MoE Q4_1, Q5_0, Q5_1 ou Q8_0.
pub struct LegacyQuantMoeBlock {
    model: String,
    layer: u32,
    top_k: usize,
    format: LegacyQuantFormat,
    executor: LegacyQuantExecutor,
}

impl LegacyQuantMoeBlock {
    pub fn new(
        cache: Arc<ExpertCache>,
        model: impl Into<String>,
        layer: u32,
        top_k: usize,
        format: LegacyQuantFormat,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            model: model.into(),
            layer,
            top_k,
            format,
            executor: LegacyQuantExecutor::new(cache, format, hidden, intermediate)
                .map_err(|error| error.to_string())?,
        })
    }

    pub fn forward_from_source(
        &self,
        source: &LegacyQuantSource,
        hidden: &[f32],
    ) -> Result<Vec<f32>, MoeBlockError> {
        let logits = source
            .router_logits(self.layer, hidden)
            .map_err(LegacyQuantExecuteError::Source)?;
        let model = &self.model;
        let layer = self.layer;
        let format = self.format.name();
        let output = dispatch_top_k(&logits, self.top_k, |expert| {
            let key = ExpertKey::new(model, layer, expert, format);
            self.executor.run_from_source(source, key, layer, hidden)
        });
        output.map_err(|error| match error {
            DispatchError::Route(error) => MoeBlockError::Route(error),
            DispatchError::Expert(error) => MoeBlockError::Executor(error.to_string()),
        })
    }

    pub fn forward_from_source_with_route(
        &self,
        source: &LegacyQuantSource,
        hidden: &[f32],
        ids: &[u32],
        weights: &[f32],
    ) -> Result<Vec<f32>, MoeBlockError> {
        let model = &self.model;
        let layer = self.layer;
        let format = self.format.name();
        dispatch_resolved(ids, weights, |expert| {
            let key = ExpertKey::new(model, layer, expert, format);
            self.executor.run_from_source(source, key, layer, hidden)
        })
    }
}

/// Forward de um bloco MoE escalar F32, F16 ou BF16 para uma camada GGUF.
pub struct PlainMoeBlock {
    model: String,
    layer: u32,
    top_k: usize,
    format: PlainFormat,
    executor: PlainExecutor,
    #[cfg(feature = "cuda")]
    cuda: Option<super::cuda::PlainCudaExecutor>,
}

/// Forward de um bloco MoE MXFP4, TQ1_0, TQ2_0, Q1_0 ou Q2_0.
pub struct LowBitMoeBlock {
    model: String,
    layer: u32,
    top_k: usize,
    format: LowBitFormat,
    executor: LowBitExecutor,
}

impl LowBitMoeBlock {
    pub fn new(
        cache: Arc<ExpertCache>,
        model: impl Into<String>,
        layer: u32,
        top_k: usize,
        format: LowBitFormat,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            model: model.into(),
            layer,
            top_k,
            format,
            executor: LowBitExecutor::new(cache, format, hidden, intermediate)
                .map_err(|error| error.to_string())?,
        })
    }

    pub fn forward_from_source(
        &self,
        source: &LowBitSource,
        hidden: &[f32],
    ) -> Result<Vec<f32>, MoeBlockError> {
        let logits = source
            .router_logits(self.layer, hidden)
            .map_err(LowBitExecuteError::Source)?;
        let model = &self.model;
        let layer = self.layer;
        let format = self.format.name();
        let output = dispatch_top_k(&logits, self.top_k, |expert| {
            let key = ExpertKey::new(model, layer, expert, format);
            self.executor.run_from_source(source, key, layer, hidden)
        });
        output.map_err(|error| match error {
            DispatchError::Route(error) => MoeBlockError::Route(error),
            DispatchError::Expert(error) => MoeBlockError::Executor(error.to_string()),
        })
    }

    pub fn forward_from_source_with_route(
        &self,
        source: &LowBitSource,
        hidden: &[f32],
        ids: &[u32],
        weights: &[f32],
    ) -> Result<Vec<f32>, MoeBlockError> {
        let model = &self.model;
        let layer = self.layer;
        let format = self.format.name();
        dispatch_resolved(ids, weights, |expert| {
            let key = ExpertKey::new(model, layer, expert, format);
            self.executor.run_from_source(source, key, layer, hidden)
        })
    }
}

impl PlainMoeBlock {
    #[cfg(not(feature = "cuda"))]
    pub fn new(
        cache: Arc<ExpertCache>,
        model: impl Into<String>,
        layer: u32,
        top_k: usize,
        format: PlainFormat,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            model: model.into(),
            layer,
            top_k,
            format,
            executor: PlainExecutor::new(cache, format, hidden, intermediate)
                .map_err(|error| error.to_string())?,
            #[cfg(feature = "cuda")]
            cuda: None,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn new_with_device_cache(
        cache: Arc<ExpertCache>,
        device_cache: Arc<super::cache::DeviceExpertCache>,
        model: impl Into<String>,
        layer: u32,
        top_k: usize,
        format: PlainFormat,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            model: model.into(),
            layer,
            top_k,
            format,
            executor: PlainExecutor::new(cache, format, hidden, intermediate)
                .map_err(|error| error.to_string())?,
            cuda: Some(
                super::cuda::PlainCudaExecutor::new_with_cache(
                    format,
                    hidden,
                    intermediate,
                    device_cache,
                )
                .map_err(|error| error.to_string())?,
            ),
        })
    }

    #[cfg(feature = "cuda")]
    pub fn uses_cuda(&self) -> bool {
        self.cuda.is_some()
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_diagnostics(&self) -> super::cuda::Nvfp4CudaDiagnostics {
        self.cuda
            .as_ref()
            .map_or_else(Default::default, |executor| executor.diagnostics())
    }

    #[cfg(feature = "cuda")]
    pub fn resident_cache_stats(&self) -> super::cache::DeviceCacheStats {
        self.cuda
            .as_ref()
            .map_or_else(super::cache::DeviceCacheStats::default, |executor| {
                executor.resident_cache_stats()
            })
    }

    pub fn forward_from_source(
        &self,
        source: &PlainSource,
        hidden: &[f32],
    ) -> Result<Vec<f32>, MoeBlockError> {
        let logits = source
            .router_logits(self.layer, hidden)
            .map_err(PlainExecuteError::Source)?;
        let model = &self.model;
        let layer = self.layer;
        let format = self.format.name();
        let output = dispatch_top_k(&logits, self.top_k, |expert| {
            let key = ExpertKey::new(model, layer, expert, format);
            self.executor.run_from_source(source, key, layer, hidden)
        });
        output.map_err(|error| match error {
            DispatchError::Route(error) => MoeBlockError::Route(error),
            DispatchError::Expert(error) => MoeBlockError::Executor(error.to_string()),
        })
    }

    pub fn forward_from_source_with_route(
        &self,
        source: &PlainSource,
        hidden: &[f32],
        ids: &[u32],
        weights: &[f32],
    ) -> Result<Vec<f32>, MoeBlockError> {
        let model = &self.model;
        let layer = self.layer;
        let format = self.format.name();
        dispatch_resolved(ids, weights, |expert| {
            let key = ExpertKey::new(model, layer, expert, format);
            self.executor.run_from_source(source, key, layer, hidden)
        })
    }

    #[cfg(feature = "cuda")]
    pub fn forward_batch_from_source_with_route(
        &self,
        source: &PlainSource,
        hidden: &[f32],
        experts: &[u32],
        token_ids: &[u32],
        weights: &[f32],
        token_offsets: &[usize],
        tokens: usize,
    ) -> Result<Vec<f32>, MoeBlockError> {
        let Some(cuda) = self.cuda.as_ref() else {
            return Err(MoeBlockError::Executor(
                "lote CUDA escalar solicitado sem ponte CUDA ativa".into(),
            ));
        };
        if experts.is_empty()
            || experts.len() != token_ids.len()
            || experts.len() != weights.len()
            || tokens == 0
            || token_offsets.len() != tokens + 1
            || token_offsets.first() != Some(&0)
            || token_offsets.last() != Some(&experts.len())
        {
            return Err(MoeBlockError::Executor(
                "lote CUDA escalar tem rotas inválidas".into(),
            ));
        }
        let mut leases = Vec::with_capacity(experts.len());
        for &expert in experts {
            let key = ExpertKey::new(&self.model, self.layer, expert, self.format.name());
            let lease = match cuda.resident_lookup(&key) {
                Some(lease) => lease,
                None => {
                    let payload = source
                        .load_expert(self.layer, expert)
                        .map_err(|error| MoeBlockError::Executor(error.to_string()))?;
                    cuda.resident_admit(key, &payload)
                        .map_err(|error| MoeBlockError::Executor(error.to_string()))?
                }
            };
            leases.push(lease);
        }
        cuda.run_route_batch_resident(&leases, token_ids, weights, token_offsets, hidden, tokens)
            .map_err(|error| MoeBlockError::Executor(error.to_string()))
    }
}

/// Forward de um bloco MoE IQ3_S, IQ2_S ou IQ4_XS para uma camada GGUF.
pub struct IqQuantMoeBlock {
    model: String,
    layer: u32,
    top_k: usize,
    format: IqQuantFormat,
    executor: IqQuantExecutor,
}

impl IqQuantMoeBlock {
    pub fn new(
        cache: Arc<ExpertCache>,
        model: impl Into<String>,
        layer: u32,
        top_k: usize,
        format: IqQuantFormat,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, IqQuantError> {
        Ok(Self {
            model: model.into(),
            layer,
            top_k,
            format,
            executor: IqQuantExecutor::new(cache, format, hidden, intermediate)?,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn new_with_device_cache(
        cache: Arc<ExpertCache>,
        device_cache: Arc<super::cache::DeviceExpertCache>,
        model: impl Into<String>,
        layer: u32,
        top_k: usize,
        format: IqQuantFormat,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, IqQuantError> {
        Ok(Self {
            model: model.into(),
            layer,
            top_k,
            format,
            executor: IqQuantExecutor::new_with_device_cache(
                cache,
                device_cache,
                format,
                hidden,
                intermediate,
            )?,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn uses_cuda(&self) -> bool {
        self.executor.uses_cuda()
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_diagnostics(&self) -> super::cuda::Nvfp4CudaDiagnostics {
        self.executor.cuda_diagnostics()
    }

    #[cfg(feature = "cuda")]
    pub fn resident_cache_stats(&self) -> super::cache::DeviceCacheStats {
        self.executor.resident_cache_stats()
    }

    pub fn forward_from_source(
        &self,
        source: &IqQuantSource,
        hidden: &[f32],
    ) -> Result<Vec<f32>, MoeBlockError> {
        let logits = source
            .router_logits(self.layer, hidden)
            .map_err(IqQuantExecuteError::Source)?;
        let model = &self.model;
        let layer = self.layer;
        let output = dispatch_top_k(&logits, self.top_k, |expert| {
            let key = ExpertKey::new(model, layer, expert, self.format.name());
            self.executor.run_from_source(source, key, layer, hidden)
        });
        output.map_err(|error| match error {
            DispatchError::Route(error) => MoeBlockError::Route(error),
            DispatchError::Expert(error) => MoeBlockError::Executor(error.to_string()),
        })
    }

    pub fn forward_from_source_with_route(
        &self,
        source: &IqQuantSource,
        hidden: &[f32],
        ids: &[u32],
        weights: &[f32],
    ) -> Result<Vec<f32>, MoeBlockError> {
        let model = &self.model;
        let layer = self.layer;
        let format = self.format.name();
        dispatch_resolved(ids, weights, |expert| {
            let key = ExpertKey::new(model, layer, expert, format);
            self.executor.run_from_source(source, key, layer, hidden)
        })
    }

    #[cfg(feature = "cuda")]
    pub fn forward_batch_from_source_with_route(
        &self,
        source: &IqQuantSource,
        hidden: &[f32],
        experts: &[u32],
        token_ids: &[u32],
        weights: &[f32],
        token_offsets: &[usize],
        tokens: usize,
    ) -> Result<Vec<f32>, MoeBlockError> {
        if !self.uses_cuda() {
            return Err(MoeBlockError::Executor(
                "lote CUDA IQ-quant solicitado sem ponte CUDA ativa".into(),
            ));
        }
        self.executor
            .run_route_batch_from_source(
                source,
                &self.model,
                self.layer,
                experts,
                token_ids,
                weights,
                token_offsets,
                hidden,
                tokens,
            )
            .map_err(MoeBlockError::from)
    }
}

/// Forward de um bloco MoE Q4_K ou Q6_K para uma camada GGUF.
pub struct KQuantMoeBlock {
    model: String,
    layer: u32,
    top_k: usize,
    format: KQuantFormat,
    executor: KQuantExecutor,
}

impl KQuantMoeBlock {
    pub fn new(
        cache: Arc<ExpertCache>,
        model: impl Into<String>,
        layer: u32,
        top_k: usize,
        format: KQuantFormat,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, KQuantError> {
        Ok(Self {
            model: model.into(),
            layer,
            top_k,
            format,
            executor: KQuantExecutor::new(cache, format, hidden, intermediate)?,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn new_with_device_cache(
        cache: Arc<ExpertCache>,
        device_cache: Arc<super::cache::DeviceExpertCache>,
        model: impl Into<String>,
        layer: u32,
        top_k: usize,
        format: KQuantFormat,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, KQuantError> {
        Ok(Self {
            model: model.into(),
            layer,
            top_k,
            format,
            executor: KQuantExecutor::new_with_device_cache(
                cache,
                device_cache,
                format,
                hidden,
                intermediate,
            )?,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn uses_cuda(&self) -> bool {
        self.executor.uses_cuda()
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_diagnostics(&self) -> super::cuda::Nvfp4CudaDiagnostics {
        self.executor.cuda_diagnostics()
    }

    #[cfg(feature = "cuda")]
    pub fn resident_cache_stats(&self) -> super::cache::DeviceCacheStats {
        self.executor.resident_cache_stats()
    }

    pub fn forward_from_source(
        &self,
        source: &KQuantSource,
        hidden: &[f32],
    ) -> Result<Vec<f32>, MoeBlockError> {
        let logits = source
            .router_logits(self.layer, hidden)
            .map_err(KQuantExecuteError::Source)?;
        let model = &self.model;
        let layer = self.layer;
        let format = self.format.name();
        let output = dispatch_top_k(&logits, self.top_k, |expert| {
            let key = ExpertKey::new(model, layer, expert, format);
            self.executor.run_from_source(source, key, layer, hidden)
        });
        output.map_err(|error| match error {
            DispatchError::Route(error) => MoeBlockError::Route(error),
            DispatchError::Expert(error) => MoeBlockError::Executor(error.to_string()),
        })
    }

    pub fn forward_from_source_with_route(
        &self,
        source: &KQuantSource,
        hidden: &[f32],
        ids: &[u32],
        weights: &[f32],
    ) -> Result<Vec<f32>, MoeBlockError> {
        #[cfg(feature = "cuda")]
        if self.uses_cuda() {
            return self
                .executor
                .run_route_batch_from_source(
                    source,
                    &self.model,
                    self.layer,
                    ids,
                    &vec![0; ids.len()],
                    weights,
                    &[0, ids.len()],
                    hidden,
                    1,
                )
                .map_err(MoeBlockError::from);
        }
        let model = &self.model;
        let layer = self.layer;
        let format = self.format.name();
        dispatch_resolved(ids, weights, |expert| {
            let key = ExpertKey::new(model, layer, expert, format);
            self.executor.run_from_source(source, key, layer, hidden)
        })
    }

    #[cfg(feature = "cuda")]
    pub fn forward_batch_from_source_with_route(
        &self,
        source: &KQuantSource,
        hidden: &[f32],
        experts: &[u32],
        token_ids: &[u32],
        weights: &[f32],
        token_offsets: &[usize],
        tokens: usize,
    ) -> Result<Vec<f32>, MoeBlockError> {
        if !self.uses_cuda() {
            return Err(MoeBlockError::Executor(
                "lote CUDA K-quant solicitado sem ponte CUDA ativa".into(),
            ));
        }
        self.executor
            .run_route_batch_from_source(
                source,
                &self.model,
                self.layer,
                experts,
                token_ids,
                weights,
                token_offsets,
                hidden,
                tokens,
            )
            .map_err(MoeBlockError::from)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_batch_from_source_with_route_cpu(
        &self,
        source: &KQuantSource,
        experts: &[u32],
        token_ids: &[u32],
        weights: &[f32],
        token_offsets: &[usize],
        hidden: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, MoeBlockError> {
        if experts.is_empty() || experts.len() != token_ids.len() || experts.len() != weights.len()
        {
            return Err(MoeBlockError::Route(RouteError::MismatchedRouteLengths {
                ids: experts.len(),
                weights: weights.len(),
            }));
        }
        if token_offsets.len() != tokens.saturating_add(1)
            || token_offsets.first() != Some(&0)
            || token_offsets.last() != Some(&experts.len())
            || token_offsets.windows(2).any(|pair| pair[0] > pair[1])
            || hidden.len() != tokens.saturating_mul(source.hidden())
        {
            return Err(MoeBlockError::Executor(
                "lote CPU K-quant tem geometria de rota inválida".into(),
            ));
        }
        let mut output = vec![0.0; hidden.len()];
        for token in 0..tokens {
            let token_start = token * source.hidden();
            let token_hidden = &hidden[token_start..token_start + source.hidden()];
            let route_start = token_offsets[token];
            let route_end = token_offsets[token + 1];
            if token_ids[route_start..route_end]
                .iter()
                .any(|&token_id| token_id != token as u32)
            {
                return Err(MoeBlockError::Executor(
                    "lote CPU K-quant não está em ordem token-major".into(),
                ));
            }
            let route_output = dispatch_route_parallel(
                &experts[route_start..route_end],
                &weights[route_start..route_end],
                |expert| {
                    let key = ExpertKey::new(&self.model, self.layer, expert, self.format.name());
                    self.executor
                        .run_from_source_cpu(source, key, self.layer, token_hidden)
                },
            )
            .map_err(|error| match error {
                DispatchError::Route(error) => MoeBlockError::Route(error),
                DispatchError::Expert(error) => MoeBlockError::Executor(error.to_string()),
            })?;
            output[token_start..token_start + source.hidden()].copy_from_slice(&route_output);
        }
        Ok(output)
    }
}

/// Forward de um bloco MoE Q4_0 para uma camada GGUF.
pub struct Q4_0MoeBlock {
    model: String,
    layer: u32,
    top_k: usize,
    executor: Q4_0Executor,
    #[cfg(feature = "cuda")]
    cuda: Option<super::cuda::HeterogeneousCudaExecutor>,
}

impl Q4_0MoeBlock {
    pub fn new(
        cache: Arc<ExpertCache>,
        model: impl Into<String>,
        layer: u32,
        top_k: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, Q4Error> {
        Ok(Self {
            model: model.into(),
            layer,
            top_k,
            executor: Q4_0Executor::new(cache, hidden, intermediate)?,
            #[cfg(feature = "cuda")]
            cuda: None,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn new_with_device_cache(
        cache: Arc<ExpertCache>,
        device_cache: Arc<super::cache::DeviceExpertCache>,
        model: impl Into<String>,
        layer: u32,
        top_k: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            model: model.into(),
            layer,
            top_k,
            executor: Q4_0Executor::new(cache, hidden, intermediate)
                .map_err(|error| error.to_string())?,
            cuda: Some(super::cuda::HeterogeneousCudaExecutor::new_with_cache(
                hidden,
                intermediate,
                device_cache,
            )),
        })
    }

    #[cfg(feature = "cuda")]
    pub fn uses_cuda(&self) -> bool {
        self.cuda.is_some()
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_diagnostics(&self) -> super::cuda::Nvfp4CudaDiagnostics {
        self.cuda.as_ref().map_or_else(
            Default::default,
            super::cuda::HeterogeneousCudaExecutor::diagnostics,
        )
    }

    #[cfg(feature = "cuda")]
    pub fn resident_cache_stats(&self) -> super::cache::DeviceCacheStats {
        self.cuda.as_ref().map_or_else(
            Default::default,
            super::cuda::HeterogeneousCudaExecutor::resident_cache_stats,
        )
    }

    pub fn forward(
        &self,
        source: &Q4_0Source,
        hidden: &[f32],
        router_logits: &[f32],
    ) -> Result<Vec<f32>, MoeBlockError> {
        let model = &self.model;
        let layer = self.layer;
        let output = dispatch_top_k(router_logits, self.top_k, |expert| {
            let key = ExpertKey::new(model, layer, expert, "q4_0");
            self.executor.run_from_source(source, key, layer, hidden)
        });
        output.map_err(|error| match error {
            DispatchError::Route(error) => MoeBlockError::Route(error),
            DispatchError::Expert(error) => MoeBlockError::Executor(error.to_string()),
        })
    }

    pub fn forward_from_source(
        &self,
        source: &Q4_0Source,
        hidden: &[f32],
    ) -> Result<Vec<f32>, MoeBlockError> {
        let logits = source
            .router_logits(self.layer, hidden)
            .map_err(Q4ExecuteError::Source)?;
        self.forward(source, hidden, &logits)
    }

    pub fn forward_from_source_with_route(
        &self,
        source: &Q4_0Source,
        hidden: &[f32],
        ids: &[u32],
        weights: &[f32],
    ) -> Result<Vec<f32>, MoeBlockError> {
        let model = &self.model;
        let layer = self.layer;
        dispatch_resolved(ids, weights, |expert| {
            let key = ExpertKey::new(model, layer, expert, "q4_0");
            self.executor.run_from_source(source, key, layer, hidden)
        })
    }

    #[cfg(feature = "cuda")]
    pub fn forward_batch_from_source_with_route(
        &self,
        source: &Q4_0Source,
        experts: &[u32],
        token_ids: &[u32],
        weights: &[f32],
        token_offsets: &[usize],
        hidden: &[f32],
        tokens: usize,
    ) -> Result<Vec<f32>, MoeBlockError> {
        if experts.is_empty()
            || experts.len() != token_ids.len()
            || experts.len() != weights.len()
            || tokens == 0
            || token_offsets.len() != tokens + 1
            || token_offsets.first() != Some(&0)
            || token_offsets.last() != Some(&experts.len())
            || token_offsets.windows(2).any(|pair| pair[0] > pair[1])
            || token_ids.iter().any(|&token| token as usize >= tokens)
        {
            return Err(MoeBlockError::Executor(
                "lote CUDA Q4_0 tem rotas inválidas".into(),
            ));
        }
        let Some(cuda) = self.cuda.as_ref() else {
            return Err(MoeBlockError::Executor(
                "lote CUDA Q4_0 solicitado sem ponte CUDA ativa".into(),
            ));
        };
        let (gate_up_bytes, down_bytes) =
            super::format_q4_0::section_lengths(source.hidden(), source.intermediate())
                .map_err(|error| MoeBlockError::Executor(error.to_string()))?;
        let mut descriptors = Vec::with_capacity(experts.len());
        let mut leases = Vec::with_capacity(experts.len());
        for &expert in experts {
            let key = ExpertKey::new(&self.model, self.layer, expert, "q4_0");
            let lease = match cuda.resident_lookup(&key) {
                Some(lease) => lease,
                None => {
                    let payload = source
                        .load_expert(self.layer, expert)
                        .map_err(|error| MoeBlockError::Executor(error.to_string()))?;
                    super::format_q4_0::payload_section_lengths(
                        source.hidden(),
                        source.intermediate(),
                        &payload,
                    )
                    .map_err(|error| MoeBlockError::Executor(error.to_string()))?;
                    cuda.resident_admit(key, &payload)
                        .map_err(|error| MoeBlockError::Executor(error.to_string()))?
                }
            };
            let matrix_start = (lease.handle() as *const u8).wrapping_add(8);
            descriptors.push(CudaHeterogeneousExpert {
                gate_up: CudaHeterogeneousTensor {
                    data: matrix_start,
                    bytes: gate_up_bytes,
                    rows: 2 * source.intermediate(),
                    columns: source.hidden(),
                    dtype: 2,
                    ..CudaHeterogeneousTensor::default()
                },
                down: CudaHeterogeneousTensor {
                    data: matrix_start.wrapping_add(gate_up_bytes),
                    bytes: down_bytes,
                    rows: source.hidden(),
                    columns: source.intermediate(),
                    dtype: 2,
                    ..CudaHeterogeneousTensor::default()
                },
                ..CudaHeterogeneousExpert::default()
            });
            leases.push(lease);
        }
        cuda.run_route_batch_resident(
            &descriptors,
            &leases
                .iter()
                .map(super::cache::DeviceExpertLease::device_index)
                .collect::<Vec<_>>(),
            experts,
            token_ids,
            weights,
            token_offsets,
            hidden,
            tokens,
            1,
        )
        .map_err(|error| MoeBlockError::Executor(error.to_string()))
    }
}

/// Forward de um bloco MoE NVFP4 para uma camada GGUF.
pub struct Nvfp4MoeBlock {
    model: String,
    layer: u32,
    top_k: usize,
    executor: Nvfp4Executor,
}

impl Nvfp4MoeBlock {
    pub fn new(
        cache: Arc<ExpertCache>,
        model: impl Into<String>,
        layer: u32,
        top_k: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            model: model.into(),
            layer,
            top_k,
            executor: Nvfp4Executor::new(cache, hidden, intermediate)
                .map_err(|error| error.to_string())?,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn new_cpu(
        cache: Arc<ExpertCache>,
        model: impl Into<String>,
        layer: u32,
        top_k: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            model: model.into(),
            layer,
            top_k,
            executor: Nvfp4Executor::new_cpu(cache, hidden, intermediate)
                .map_err(|error| error.to_string())?,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn new_with_device_cache(
        cache: Arc<ExpertCache>,
        device_cache: Arc<super::cache::DeviceExpertCache>,
        model: impl Into<String>,
        layer: u32,
        top_k: usize,
        hidden: usize,
        intermediate: usize,
        coexecution_profile: Option<super::bandwidth::BandwidthProfile>,
    ) -> Result<Self, String> {
        Ok(Self {
            model: model.into(),
            layer,
            top_k,
            executor: Nvfp4Executor::new_with_device_cache(
                cache,
                device_cache,
                hidden,
                intermediate,
                coexecution_profile,
            )
            .map_err(|error| error.to_string())?,
        })
    }

    pub fn forward_from_source(
        &self,
        source: &Nvfp4Source,
        hidden: &[f32],
    ) -> Result<Vec<f32>, MoeBlockError> {
        let logits = source
            .router_logits(self.layer, hidden)
            .map_err(|error| MoeBlockError::Executor(error.to_string()))?;
        self.forward_from_source_with_logits(source, hidden, &logits)
    }

    pub fn forward_from_source_with_logits(
        &self,
        source: &Nvfp4Source,
        hidden: &[f32],
        logits: &[f32],
    ) -> Result<Vec<f32>, MoeBlockError> {
        let route = select_top_k(logits, self.top_k).map_err(MoeBlockError::Route)?;
        self.forward_from_source_with_route(source, hidden, &route.ids, &route.weights)
    }

    pub fn executor_kind(&self) -> Nvfp4BackendKind {
        self.executor.backend_kind()
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_diagnostics(&self) -> super::cuda::Nvfp4CudaDiagnostics {
        self.executor.cuda_diagnostics()
    }

    #[cfg(feature = "cuda")]
    pub fn host_diagnostics(&self) -> super::format_nvfp4::Nvfp4HostDiagnostics {
        self.executor.host_diagnostics()
    }

    #[cfg(feature = "cuda")]
    pub fn resident_cache_stats(&self) -> super::cache::DeviceCacheStats {
        self.executor.resident_cache_stats()
    }

    pub fn forward_from_source_with_route(
        &self,
        source: &Nvfp4Source,
        hidden: &[f32],
        ids: &[u32],
        weights: &[f32],
    ) -> Result<Vec<f32>, MoeBlockError> {
        if ids.is_empty() {
            return Err(MoeBlockError::Route(RouteError::InvalidWeightSum));
        }
        if ids.len() != weights.len() {
            return Err(MoeBlockError::Route(RouteError::MismatchedRouteLengths {
                ids: ids.len(),
                weights: weights.len(),
            }));
        }
        if weights.iter().any(|weight| !weight.is_finite()) {
            return Err(MoeBlockError::Route(RouteError::InvalidWeightSum));
        }
        #[cfg(feature = "cuda")]
        if matches!(self.executor.backend_kind(), Nvfp4BackendKind::Cuda) {
            let token_ids = vec![0; ids.len()];
            return self
                .executor
                .run_route_batch_from_source(
                    source,
                    &self.model,
                    self.layer,
                    ids,
                    &token_ids,
                    weights,
                    &[0, ids.len()],
                    hidden,
                    1,
                )
                .map_err(MoeBlockError::from);
        }
        let model = &self.model;
        let layer = self.layer;
        dispatch_resolved(ids, weights, |expert| {
            let key = ExpertKey::new(model, layer, expert, "nvfp4");
            self.executor.run_from_source(source, key, layer, hidden)
        })
    }

    #[cfg(feature = "cuda")]
    pub fn forward_batch_from_source_with_route(
        &self,
        source: &Nvfp4Source,
        hidden: &[f32],
        experts: &[u32],
        token_ids: &[u32],
        weights: &[f32],
        token_offsets: &[usize],
        tokens: usize,
    ) -> Result<Vec<f32>, MoeBlockError> {
        if experts.is_empty() {
            return Err(MoeBlockError::Route(RouteError::InvalidWeightSum));
        }
        if experts.len() != token_ids.len() || experts.len() != weights.len() {
            return Err(MoeBlockError::Route(RouteError::MismatchedRouteLengths {
                ids: experts.len(),
                weights: weights.len(),
            }));
        }
        if weights.iter().any(|weight| !weight.is_finite()) {
            return Err(MoeBlockError::Route(RouteError::InvalidWeightSum));
        }
        self.executor
            .run_route_batch_from_source(
                source,
                &self.model,
                self.layer,
                experts,
                token_ids,
                weights,
                token_offsets,
                hidden,
                tokens,
            )
            .map_err(MoeBlockError::from)
    }
}

/// Bloco MoE cujo decoder é escolhido por tensor, para GGUF com dtypes
/// heterogêneos entre gate, up e down.
pub struct HeterogeneousMoeBlock {
    model: String,
    layer: u32,
    top_k: usize,
    cache: Arc<ExpertCache>,
    #[cfg(feature = "cuda")]
    cuda: Option<Arc<super::cuda::HeterogeneousCudaExecutor>>,
    #[cfg(feature = "cuda")]
    coexecution_profile: Option<super::bandwidth::BandwidthProfile>,
    #[cfg(feature = "cuda")]
    host_diagnostics: Mutex<HeterogeneousHostDiagnostics>,
    #[cfg(feature = "cuda")]
    working_set_diagnostics: Mutex<HeterogeneousLayerWorkingSetDiagnostics>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, Default)]
pub struct HeterogeneousHostDiagnostics {
    pub cpu_expert_calls: u64,
    pub coexecution_gpu_fraction: Option<f32>,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HeterogeneousLayerWorkingSetDiagnostics {
    pub layer: u32,
    pub batches: u64,
    pub tokens: u64,
    pub route_experts: u64,
    pub distinct_experts: u64,
    /// Soma dos experts distintos nas rotas de cada token observado.
    /// Dividido por `tokens`, dá a média por token sem guardar cada rota.
    pub distinct_experts_per_token: u64,
    pub max_distinct_experts_per_token: u64,
}

impl HeterogeneousMoeBlock {
    pub fn new(
        cache: Arc<ExpertCache>,
        model: impl Into<String>,
        layer: u32,
        top_k: usize,
    ) -> Self {
        Self {
            model: model.into(),
            layer,
            top_k,
            cache,
            #[cfg(feature = "cuda")]
            cuda: None,
            #[cfg(feature = "cuda")]
            coexecution_profile: None,
            #[cfg(feature = "cuda")]
            host_diagnostics: Mutex::new(HeterogeneousHostDiagnostics::default()),
            #[cfg(feature = "cuda")]
            working_set_diagnostics: Mutex::new(HeterogeneousLayerWorkingSetDiagnostics {
                layer,
                ..HeterogeneousLayerWorkingSetDiagnostics::default()
            }),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn new_with_device_cache(
        cache: Arc<ExpertCache>,
        device_cache: Arc<super::cache::DeviceExpertCache>,
        model: impl Into<String>,
        layer: u32,
        top_k: usize,
        _hidden: usize,
        _intermediate: usize,
    ) -> Self {
        Self {
            model: model.into(),
            layer,
            top_k,
            cache,
            cuda: Some(Arc::new(
                super::cuda::HeterogeneousCudaExecutor::new_with_cache(
                    _hidden,
                    _intermediate,
                    device_cache,
                ),
            )),
            coexecution_profile: None,
            host_diagnostics: Mutex::new(HeterogeneousHostDiagnostics::default()),
            working_set_diagnostics: Mutex::new(HeterogeneousLayerWorkingSetDiagnostics {
                layer,
                ..HeterogeneousLayerWorkingSetDiagnostics::default()
            }),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn new_with_device_executor(
        cache: Arc<ExpertCache>,
        cuda: Arc<super::cuda::HeterogeneousCudaExecutor>,
        model: impl Into<String>,
        layer: u32,
        top_k: usize,
        coexecution_profile: Option<super::bandwidth::BandwidthProfile>,
    ) -> Self {
        Self {
            model: model.into(),
            layer,
            top_k,
            cache,
            cuda: Some(cuda),
            coexecution_profile,
            host_diagnostics: Mutex::new(HeterogeneousHostDiagnostics::default()),
            working_set_diagnostics: Mutex::new(HeterogeneousLayerWorkingSetDiagnostics {
                layer,
                ..HeterogeneousLayerWorkingSetDiagnostics::default()
            }),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn uses_cuda(&self) -> bool {
        self.cuda.is_some()
    }

    #[cfg(feature = "cuda")]
    pub fn cuda_diagnostics(&self) -> super::cuda::Nvfp4CudaDiagnostics {
        self.cuda
            .as_ref()
            .map_or_else(Default::default, |cuda| cuda.diagnostics())
    }

    #[cfg(feature = "cuda")]
    pub fn resident_cache_stats(&self) -> super::cache::DeviceCacheStats {
        self.cuda
            .as_ref()
            .map_or_else(Default::default, |cuda| cuda.resident_cache_stats())
    }

    #[cfg(feature = "cuda")]
    pub fn resident_cache_capacity_bytes_by_device(&self) -> Vec<Option<u64>> {
        self.cuda.as_ref().map_or_else(Vec::new, |cuda| {
            cuda.resident_cache_capacity_bytes_by_device()
        })
    }

    #[cfg(feature = "cuda")]
    pub fn resident_cache_budget_snapshots_by_device(
        &self,
    ) -> Vec<Option<super::admission::CacheBudgetSnapshot>> {
        self.cuda.as_ref().map_or_else(Vec::new, |cuda| {
            cuda.resident_cache_budget_snapshots_by_device()
        })
    }

    #[cfg(feature = "cuda")]
    pub fn host_diagnostics(&self) -> HeterogeneousHostDiagnostics {
        *self
            .host_diagnostics
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    #[cfg(feature = "cuda")]
    pub fn working_set_diagnostics(&self) -> HeterogeneousLayerWorkingSetDiagnostics {
        self.working_set_diagnostics
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    #[cfg(feature = "cuda")]
    fn record_working_set_diagnostics(
        &self,
        experts: &[u32],
        token_offsets: &[usize],
        tokens: usize,
    ) {
        let mut all_experts = HashSet::with_capacity(experts.len());
        let mut distinct_experts_per_token = 0u64;
        let mut max_distinct_experts_per_token = 0u64;
        for window in token_offsets.windows(2).take(tokens) {
            let mut token_experts = HashSet::with_capacity(window[1] - window[0]);
            for &expert in &experts[window[0]..window[1]] {
                all_experts.insert(expert);
                token_experts.insert(expert);
            }
            let distinct = token_experts.len() as u64;
            distinct_experts_per_token = distinct_experts_per_token.saturating_add(distinct);
            max_distinct_experts_per_token = max_distinct_experts_per_token.max(distinct);
        }
        let mut diagnostics = self
            .working_set_diagnostics
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        diagnostics.batches = diagnostics.batches.saturating_add(1);
        diagnostics.tokens = diagnostics.tokens.saturating_add(tokens as u64);
        diagnostics.route_experts = diagnostics
            .route_experts
            .saturating_add(experts.len() as u64);
        diagnostics.distinct_experts = diagnostics
            .distinct_experts
            .saturating_add(all_experts.len() as u64);
        diagnostics.distinct_experts_per_token = diagnostics
            .distinct_experts_per_token
            .saturating_add(distinct_experts_per_token);
        diagnostics.max_distinct_experts_per_token = diagnostics
            .max_distinct_experts_per_token
            .max(max_distinct_experts_per_token);
    }

    pub fn forward_from_source_with_route(
        &self,
        source: &HeterogeneousMoeSource,
        hidden: &[f32],
        ids: &[u32],
        weights: &[f32],
        activation: i32,
    ) -> Result<Vec<f32>, MoeBlockError> {
        if ids.len() != weights.len() {
            return Err(MoeBlockError::Route(RouteError::MismatchedRouteLengths {
                ids: ids.len(),
                weights: weights.len(),
            }));
        }
        if ids.is_empty() || ids.len() > self.top_k {
            return Err(MoeBlockError::Route(RouteError::InvalidTopK {
                top_k: ids.len(),
                experts: self.top_k,
            }));
        }
        source
            .run_from_source_with_route(
                &self.cache,
                &self.model,
                self.layer,
                hidden,
                ids,
                weights,
                activation,
            )
            .map_err(MoeBlockError::Executor)
    }

    pub fn forward_batch_from_source_with_route_cpu(
        &self,
        source: &HeterogeneousMoeSource,
        experts: &[u32],
        token_ids: &[u32],
        weights: &[f32],
        token_offsets: &[usize],
        hidden: &[f32],
        tokens: usize,
        activation: i32,
    ) -> Result<Vec<f32>, MoeBlockError> {
        if experts.is_empty() || experts.len() != token_ids.len() || experts.len() != weights.len()
        {
            return Err(MoeBlockError::Route(RouteError::MismatchedRouteLengths {
                ids: experts.len(),
                weights: weights.len(),
            }));
        }
        if experts.len() > self.top_k.saturating_mul(tokens.max(1)) {
            return Err(MoeBlockError::Route(RouteError::InvalidTopK {
                top_k: experts.len(),
                experts: self.top_k,
            }));
        }
        source
            .run_route_batch_from_source(
                &self.cache,
                &self.model,
                self.layer,
                experts,
                token_ids,
                weights,
                token_offsets,
                hidden,
                tokens,
                activation,
            )
            .map_err(MoeBlockError::Executor)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_batch_from_source_with_route(
        &self,
        source: &HeterogeneousMoeSource,
        experts: &[u32],
        token_ids: &[u32],
        weights: &[f32],
        token_offsets: &[usize],
        hidden: &[f32],
        tokens: usize,
        activation: i32,
    ) -> Result<Vec<f32>, MoeBlockError> {
        if experts.is_empty() || experts.len() != token_ids.len() || experts.len() != weights.len()
        {
            return Err(MoeBlockError::Route(RouteError::MismatchedRouteLengths {
                ids: experts.len(),
                weights: weights.len(),
            }));
        }
        if experts.len() > self.top_k.saturating_mul(tokens.max(1))
            || tokens == 0
            || token_offsets.len() != tokens + 1
            || token_offsets.first() != Some(&0)
            || token_offsets.last() != Some(&experts.len())
            || token_offsets.windows(2).any(|pair| pair[0] > pair[1])
            || token_ids.iter().any(|&token| token as usize >= tokens)
        {
            return Err(MoeBlockError::Route(RouteError::InvalidTopK {
                top_k: experts.len(),
                experts: self.top_k,
            }));
        }
        self.record_working_set_diagnostics(experts, token_offsets, tokens);
        if let Some(profile) = self.coexecution_profile {
            return self.forward_batch_coexecuted_from_source(
                source,
                experts,
                token_ids,
                weights,
                token_offsets,
                hidden,
                tokens,
                activation,
                profile,
            );
        }
        self.forward_batch_cuda_only(
            source,
            experts,
            token_ids,
            weights,
            token_offsets,
            hidden,
            tokens,
            activation,
        )
    }

    #[cfg(feature = "cuda")]
    fn forward_batch_coexecuted_from_source(
        &self,
        source: &HeterogeneousMoeSource,
        experts: &[u32],
        token_ids: &[u32],
        weights: &[f32],
        token_offsets: &[usize],
        hidden: &[f32],
        tokens: usize,
        activation: i32,
        profile: super::bandwidth::BandwidthProfile,
    ) -> Result<Vec<f32>, MoeBlockError> {
        let Some(cuda) = self.cuda.as_ref() else {
            return Err(MoeBlockError::Executor(
                "executor CUDA heterogêneo não foi selecionado".into(),
            ));
        };

        // Experts que já têm todos os seus tensors residentes ficam na GPU.
        // Um expert parcialmente residente não é elegível: admitir apenas os
        // tensors faltantes na lane GPU manteria o custo de H2D e poderia
        // dividir um descriptor entre devices.
        let mut unique_experts = Vec::new();
        for &expert in experts {
            if !unique_experts.contains(&expert) {
                unique_experts.push(expert);
            }
        }
        let specs_by_expert = source
            .expert_tensor_specs_batch(self.layer, &unique_experts)
            .map_err(|error| MoeBlockError::Executor(error.to_string()))?;
        let resident_gpu_experts = unique_experts
            .iter()
            .copied()
            .zip(specs_by_expert)
            .filter_map(|(expert, specs)| {
                let complete = !specs.is_empty()
                    && specs.into_iter().all(|spec| {
                        let key = ExpertKey::new(
                            &self.model,
                            self.layer,
                            expert,
                            format!("heterogeneous:{}:{}", spec.role.as_str(), spec.dtype),
                        );
                        cuda.resident_contains(&key)
                    });
                complete.then_some(expert)
            })
            .collect::<Vec<_>>();
        let split = split_route_experts(experts, &resident_gpu_experts, Some(profile))
            .map_err(|error| MoeBlockError::Executor(error.to_string()))?;
        {
            let mut diagnostics = self
                .host_diagnostics
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            diagnostics.cpu_expert_calls = diagnostics
                .cpu_expert_calls
                .saturating_add(split.cpu_experts.len() as u64);
            diagnostics.coexecution_gpu_fraction = Some(split.gpu_fraction);
        }
        let (cpu_experts, cpu_token_ids, cpu_weights, cpu_offsets) = filter_route(
            experts,
            token_ids,
            weights,
            token_offsets,
            &split.cpu_experts,
            tokens,
        );
        let (gpu_experts, gpu_token_ids, gpu_weights, gpu_offsets) = filter_route(
            experts,
            token_ids,
            weights,
            token_offsets,
            &split.gpu_experts,
            tokens,
        );

        let (cpu_output, gpu_output) = run_overlap(
            || {
                if cpu_experts.is_empty() {
                    Ok::<_, MoeBlockError>(vec![0.0; hidden.len()])
                } else {
                    self.forward_batch_from_source_with_route_cpu(
                        source,
                        &cpu_experts,
                        &cpu_token_ids,
                        &cpu_weights,
                        &cpu_offsets,
                        hidden,
                        tokens,
                        activation,
                    )
                }
            },
            || {
                if gpu_experts.is_empty() {
                    Ok::<_, MoeBlockError>(vec![0.0; hidden.len()])
                } else {
                    self.forward_batch_cuda_only(
                        source,
                        &gpu_experts,
                        &gpu_token_ids,
                        &gpu_weights,
                        &gpu_offsets,
                        hidden,
                        tokens,
                        activation,
                    )
                }
            },
        )
        .map_err(|error| MoeBlockError::Executor(error.to_string()))?;
        combine_lane_outputs(
            &cpu_output,
            &gpu_output,
            tokens,
            hidden.len() / tokens.max(1),
        )
        .map_err(|error| MoeBlockError::Executor(error.to_string()))
    }

    #[cfg(feature = "cuda")]
    fn forward_batch_cuda_only(
        &self,
        source: &HeterogeneousMoeSource,
        experts: &[u32],
        token_ids: &[u32],
        weights: &[f32],
        token_offsets: &[usize],
        hidden: &[f32],
        tokens: usize,
        activation: i32,
    ) -> Result<Vec<f32>, MoeBlockError> {
        if experts.is_empty() || experts.len() != token_ids.len() || experts.len() != weights.len()
        {
            return Err(MoeBlockError::Route(RouteError::MismatchedRouteLengths {
                ids: experts.len(),
                weights: weights.len(),
            }));
        }
        if experts.len() > self.top_k.saturating_mul(tokens.max(1)) {
            return Err(MoeBlockError::Route(RouteError::InvalidTopK {
                top_k: experts.len(),
                experts: self.top_k,
            }));
        }
        let Some(cuda) = self.cuda.as_ref() else {
            return Err(MoeBlockError::Executor(
                "executor CUDA heterogêneo não foi selecionado".into(),
            ));
        };

        // Primeiro resolvemos os hits e planejamos os misses. O caminho
        // anterior carregava cada tensor no mesmo thread, mesmo quando os
        // shards pertenciam a GPUs diferentes. Isso serializava dezenas de
        // GB de H2D e deixava CUDA1 ociosa durante o upload para CUDA0.
        //
        // Mantemos uma fila por device: dentro de um shard o mutex do cache
        // continua serializando admissão, eviction e atualização do LRU, mas
        // uploads para GPUs distintas podem progredir em paralelo.
        let mut resolved_experts = Vec::with_capacity(experts.len());
        let mut pending_by_device: HashMap<u32, Vec<(ExpertKey, HeterogeneousTensorSpec)>> =
            HashMap::new();
        let mut pending_keys = HashSet::new();
        let specs_by_expert = source
            .expert_tensor_specs_batch(self.layer, experts)
            .map_err(|error| MoeBlockError::Executor(error.to_string()))?;
        for (&expert, specs) in experts.iter().zip(specs_by_expert) {
            let mut resolved_specs = Vec::with_capacity(specs.len());
            for spec in specs {
                let key = ExpertKey::new(
                    &self.model,
                    self.layer,
                    expert,
                    format!("heterogeneous:{}:{}", spec.role.as_str(), spec.dtype),
                );
                let lease = cuda.resident_lookup(&key);
                if lease.is_none() && pending_keys.insert(key.clone()) {
                    let device_index = cuda.device_index_for_key(&key);
                    pending_by_device
                        .entry(device_index)
                        .or_default()
                        .push((key.clone(), spec));
                }
                resolved_specs.push((spec, key, lease));
            }
            resolved_experts.push(resolved_specs);
        }

        let admission_results = std::thread::scope(|scope| {
            let handles = pending_by_device
                .into_iter()
                .map(|(device_index, pending)| {
                    scope.spawn(move || {
                        let candidates = pending
                            .into_iter()
                            .map(|(key, spec)| {
                                source
                                    .expert_tensor_slice(spec)
                                    .map(|bytes| (key, bytes))
                                    .map_err(|error| MoeBlockError::Executor(error.to_string()))
                            })
                            .collect::<Result<Vec<_>, MoeBlockError>>()?;
                        cuda.resident_admit_many(device_index, candidates)
                            .map_err(|error| MoeBlockError::Executor(error.to_string()))
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| {
                    handle.join().map_err(|_| {
                        MoeBlockError::Executor(
                            "thread de admissão CUDA MoE entrou em pânico".into(),
                        )
                    })?
                })
                .collect::<Result<Vec<Vec<DeviceExpertLease>>, MoeBlockError>>()
        })?;
        // Os leases de admissão ficam ativos enquanto montamos os descriptors
        // e executamos o kernel. Assim um shard sob pressão não pode expulsar
        // uma entrada recém-carregada antes de ela ser usada.
        let _admitted_leases = admission_results.into_iter().flatten().collect::<Vec<_>>();

        let mut descriptors = Vec::with_capacity(experts.len());
        let mut leases = Vec::new();
        let mut device_indices = Vec::with_capacity(experts.len());
        for (expert, resolved_specs) in experts.iter().copied().zip(resolved_experts) {
            let mut descriptor = CudaHeterogeneousExpert::default();
            let mut expert_device_index = None;
            for (spec, key, lease) in resolved_specs {
                let lease = match lease {
                    Some(lease) => lease,
                    None => cuda.resident_lookup(&key).ok_or_else(|| {
                        MoeBlockError::Executor(format!(
                            "expert heterogêneo {expert} desapareceu do cache após admissão"
                        ))
                    })?,
                };
                let HeterogeneousTensorSpec {
                    role,
                    dtype,
                    rows,
                    columns,
                    scales,
                    ..
                } = spec;
                match expert_device_index {
                    Some(expected) if expected != lease.device_index() => {
                        return Err(MoeBlockError::Executor(format!(
                            "expert heterogêneo {expert} foi particionado entre CUDA{expected} e CUDA{}",
                            lease.device_index()
                        )))
                    }
                    None => expert_device_index = Some(lease.device_index()),
                    _ => {}
                }
                let tensor = CudaHeterogeneousTensor {
                    data: lease.handle() as *const u8,
                    bytes: lease.bytes(),
                    rows,
                    columns,
                    dtype,
                    scale: scales.primary,
                    input_scale: scales.input_primary,
                    secondary_scale: scales.secondary,
                    secondary_input_scale: scales.input_secondary,
                };
                match role {
                    TensorRole::GateUp if descriptor.gate_up.data.is_null() => {
                        descriptor.gate_up = tensor
                    }
                    TensorRole::Gate if descriptor.gate.data.is_null() => descriptor.gate = tensor,
                    TensorRole::Up if descriptor.up.data.is_null() => descriptor.up = tensor,
                    TensorRole::Down if descriptor.down.data.is_null() => descriptor.down = tensor,
                    role => {
                        return Err(MoeBlockError::Executor(format!(
                            "papel de tensor heterogêneo duplicado: {}",
                            role.as_str()
                        )))
                    }
                }
                leases.push(lease);
            }
            let Some(device_index) = expert_device_index else {
                return Err(MoeBlockError::Executor(format!(
                    "expert heterogêneo {expert} não possui tensors"
                )));
            };
            device_indices.push(device_index);
            descriptors.push(descriptor);
        }
        let output = cuda
            .run_route_batch_resident(
                &descriptors,
                &device_indices,
                experts,
                token_ids,
                weights,
                token_offsets,
                hidden,
                tokens,
                activation,
            )
            .map_err(|error| match error {
                CudaMoeError::NonFinite {
                    stage,
                    index,
                    count,
                    kind,
                } => MoeBlockError::CudaNonFinite(format!(
                    "executor CUDA MoE produziu {count} valores não finitos ({kind}) no estágio {stage}, primeiro índice da saída {index}"
                )),
                error => MoeBlockError::Executor(error.to_string()),
            })?;
        drop(leases);
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::super::admission::MemoryAccount;
    use super::super::cache::{ExpertCache, FreeMemoryProbe};
    use super::super::format_q4_0::{tests::write_q4_source, Q4_0Source};
    use super::*;
    use std::io;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct FixedProbe(AtomicU64);

    impl FixedProbe {
        fn new(bytes: u64) -> Arc<Self> {
            Arc::new(Self(AtomicU64::new(bytes)))
        }
    }

    impl FreeMemoryProbe for FixedProbe {
        fn free_bytes(&self) -> io::Result<u64> {
            Ok(self.0.load(Ordering::Relaxed))
        }
    }

    #[test]
    fn block_routes_top_k_and_reuses_the_global_cache() {
        let path = PathBuf::from(std::env::temp_dir())
            .join(format!("pom-ftk-q4-block-{}.gguf", std::process::id()));
        let (hidden, intermediate) = write_q4_source(&path);
        let source = Q4_0Source::open(&path, 2, hidden, intermediate).unwrap();
        let cache = ExpertCache::new(
            MemoryAccount {
                baseline_free_bytes: 100_000,
                compute_bytes: 0,
                weights_bytes: 0,
                fixed_bytes: 0,
                reserve_bytes: 0,
            },
            FixedProbe::new(100_000),
        );
        let block =
            Q4_0MoeBlock::new(cache.clone(), "fixture", 0, 1, hidden, intermediate).unwrap();
        let first = block.forward_from_source(&source, &[1.0; 32]).unwrap();
        let second = block.forward_from_source(&source, &[1.0; 32]).unwrap();
        assert_eq!(first, second);
        drop(source);
        std::fs::remove_file(path).unwrap();
    }
}
