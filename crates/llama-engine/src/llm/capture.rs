use super::*;

#[derive(Default)]
pub(super) struct CaptureSlot {
    pub(super) want: Option<CString>,
    pub(super) buf: Option<Vec<f32>>,
    pub(super) expected_tokens: usize,
    pub(super) max_microbatch_tokens: usize,
    pub(super) allow_physical_chunks: bool,
    pub(super) n_embd: usize,
    pub(super) expected_floats: Option<usize>,
    pub(super) layout: Option<ResidualCaptureLayout>,
    pub(super) error: Option<String>,
    pub(super) last_shape: Option<[i64; 4]>,
    pub(super) last_strides: Option<[usize; 4]>,
    pub(super) last_nbytes: Option<usize>,
    /// Monotonic point at which the scheduler asked whether this tensor needs
    /// a ready callback. The callback itself is still the ownership boundary:
    /// llama.cpp may reuse graph buffers between microbatches, so postponing
    /// the read until `llama_decode` returns would lose earlier residual rows.
    pub(super) capture_requested_at: Option<Instant>,
    /// Time from the scheduler's ask callback until the ready callback. This
    /// includes graph evaluation and scheduler synchronization before capture.
    pub(super) ready_wait_us: u64,
    /// Wall time inside `ggml_backend_tensor_get`. CUDA backends may include
    /// both D2H transfer and stream synchronization in this opaque call.
    pub(super) copy_us: u64,
    pub(super) postprocess_us: u64,
    pub(super) copy_bytes: u64,
    pub(super) copy_calls: u64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct CaptureMetrics {
    pub(super) ready_wait_us: u64,
    pub(super) copy_us: u64,
    pub(super) postprocess_us: u64,
    pub(super) copy_bytes: u64,
    pub(super) copy_calls: u64,
}

impl CaptureSlot {
    pub(super) fn metrics(&self) -> CaptureMetrics {
        CaptureMetrics {
            ready_wait_us: self.ready_wait_us,
            copy_us: self.copy_us,
            postprocess_us: self.postprocess_us,
            copy_bytes: self.copy_bytes,
            copy_calls: self.copy_calls,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct ResidualCaptureLayout {
    pub(super) feature_shape: Vec<usize>,
    pub(super) token_axis: usize,
}

pub fn validate_hidden_tensor(
    values_len: usize,
    shape: &TensorShape,
    n_tokens: usize,
    n_embd: usize,
    n_embd_inp: usize,
    n_embd_out: usize,
    item: &str,
) -> Result<usize> {
    if shape.len() < 2 {
        bail!(
            "{item}: hidden state shape {:?} não suportado, esperado [tokens, n_embd, ...]",
            shape
        );
    }
    if shape[0] != n_tokens {
        bail!(
            "{item}: hidden state declara {} tokens no shape {:?}, esperado {}",
            shape[0],
            shape,
            n_tokens
        );
    }
    if shape[1] != n_embd {
        bail!(
            "{item}: dimensão base {} no hidden state {:?} diverge de n_embd {}",
            shape[1],
            shape,
            n_embd
        );
    }
    let feature_width = shape[1..]
        .iter()
        .try_fold(1usize, |width, &dimension| width.checked_mul(dimension))
        .ok_or_else(|| anyhow::anyhow!("{item}: largura do hidden state excede o limite"))?;
    // A forma 2D continua usando a entrada base. Quando o emissor preserva
    // eixos de fluxo, llama.cpp reporta em n_embd_out a largura física que
    // aparece no residual. A contagem de fluxos da metadata é estrutural e
    // não deve ser multiplicada aqui em paralelo com essa fonte autoritativa.
    let has_non_singleton_extra_axis = shape[2..].iter().any(|&dimension| dimension > 1);
    let expected_width = if !has_non_singleton_extra_axis {
        n_embd_inp
    } else {
        n_embd_out
    };
    if feature_width != expected_width {
        bail!(
            "{item}: hidden state shape {:?} tem largura de entrada {}, esperado {} a partir da metadata",
            shape,
            feature_width,
            expected_width
        );
    }
    let expected_values = shape
        .iter()
        .try_fold(1usize, |values, &dimension| values.checked_mul(dimension))
        .ok_or_else(|| anyhow::anyhow!("{item}: tamanho do hidden state excede o limite"))?;
    if values_len != expected_values {
        bail!(
            "{item}: hidden state tem {} floats, esperado {} para shape {:?}",
            values_len,
            expected_values,
            shape
        );
    }
    Ok(feature_width)
}

#[derive(Debug, Clone, PartialEq)]
pub struct CapturedResidual {
    pub values: Vec<f32>,
    pub shape: TensorShape,
}

#[derive(Default)]
pub(super) struct MoeCallbackTelemetry {
    pub(super) forward_us: u64,
    pub(super) executors: Vec<crate::moe::runtime::MoeExecutorKind>,
}

pub(super) const MOE_ROUTE_DIAGNOSTICS_ENV: &str = "POM_MOE_ROUTE_DIAGNOSTICS";

#[derive(Debug, PartialEq)]
pub(super) struct MoeRouteDiagnosticSample {
    pub(super) slot: usize,
    pub(super) source_offset_elements: usize,
    pub(super) expert: i32,
    pub(super) weight: f32,
}

#[derive(Debug, PartialEq)]
pub(super) struct MoeRouteDiagnostic {
    pub(super) llama_declared_route_elements: usize,
    pub(super) pom_expected_route_elements: usize,
    pub(super) selected_read_elements: usize,
    pub(super) weights_read_elements: usize,
    pub(super) raw_nan_count: usize,
    pub(super) raw_non_finite_count: usize,
    pub(super) token_zero_samples: Vec<MoeRouteDiagnosticSample>,
}

pub(super) fn collect_moe_route_diagnostic(
    selected_experts: &[i32],
    weights: &[f32],
    n_tokens: usize,
    expert_count_used: usize,
) -> MoeRouteDiagnostic {
    let declared_route_elements = n_tokens
        .checked_mul(expert_count_used)
        .unwrap_or(usize::MAX);
    let token_zero_samples = if n_tokens == 0 {
        Vec::new()
    } else {
        (0..expert_count_used.min(4))
            .filter_map(|slot| {
                let source_offset_elements = slot;
                Some(MoeRouteDiagnosticSample {
                    slot,
                    source_offset_elements,
                    expert: *selected_experts.get(source_offset_elements)?,
                    weight: *weights.get(source_offset_elements)?,
                })
            })
            .collect()
    };

    MoeRouteDiagnostic {
        llama_declared_route_elements: declared_route_elements,
        pom_expected_route_elements: declared_route_elements,
        selected_read_elements: selected_experts.len(),
        weights_read_elements: weights.len(),
        raw_nan_count: weights.iter().filter(|weight| weight.is_nan()).count(),
        raw_non_finite_count: weights.iter().filter(|weight| !weight.is_finite()).count(),
        token_zero_samples,
    }
}

pub(super) fn moe_route_diagnostics_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var(MOE_ROUTE_DIAGNOSTICS_ENV).ok().as_deref(),
            Some("1" | "true" | "yes" | "on")
        )
    })
}

pub(super) fn log_moe_route_diagnostics_configuration() {
    static LOGGED: OnceLock<()> = OnceLock::new();
    if LOGGED.set(()).is_err() {
        return;
    }
    let raw = std::env::var(MOE_ROUTE_DIAGNOSTICS_ENV).unwrap_or_else(|_| "<unset>".to_owned());
    let enabled = moe_route_diagnostics_enabled();
    warn!("POM_MOE_ROUTE_DIAGNOSTICS configuration raw={raw:?} enabled={enabled}");
}

pub(super) fn route_diagnostic_should_emit(first_callback: bool, raw_non_finite: bool) -> bool {
    first_callback || raw_non_finite
}

pub(super) fn first_non_finite(values: &[f32]) -> Option<(usize, f32)> {
    values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
}

pub(super) struct MoeCallbackState {
    pub(super) runtime: Arc<crate::moe::runtime::MoeRuntime>,
    pub(super) telemetry: Mutex<MoeCallbackTelemetry>,
    pub(super) error: Mutex<Option<String>>,
    pub(super) route_diagnostic_emitted: AtomicBool,
}

impl MoeCallbackState {
    pub(super) fn begin(&self) {
        *self
            .error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        *self
            .telemetry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = MoeCallbackTelemetry::default();
        self.route_diagnostic_emitted
            .store(false, Ordering::Relaxed);
    }

    pub(super) fn claim_route_diagnostic(&self) -> bool {
        moe_route_diagnostics_enabled()
            && !self.route_diagnostic_emitted.swap(true, Ordering::Relaxed)
    }

    pub(super) fn fail(&self, message: String) {
        let mut error = self
            .error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if error.is_none() {
            *error = Some(message);
        }
    }

    pub(super) fn take(&self) -> Result<MoeCallbackTelemetry> {
        let error = self
            .error
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        if let Some(error) = error {
            bail!("hook MoE do llama.cpp falhou: {error}");
        }
        Ok(std::mem::take(
            &mut *self
                .telemetry
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        ))
    }
}

pub(super) fn log_moe_callback_entry(
    user_data: *mut c_void,
    descriptor: *const sys::llama_moe_ffn_descriptor,
    hidden: *const f32,
    selected_experts: *const i32,
    weights: *const f32,
    n_tokens: usize,
    hidden_width: usize,
    output: *mut f32,
) {
    tracing::warn!(
        "POM_MOE_ROUTE_DIAGNOSTICS callback_entry_default raw_args=1 descriptor_ptr={:p} user_data_ptr={:p} hidden_ptr={:p} selected_ptr={:p} weights_ptr={:p} output_ptr={:p} n_tokens={n_tokens} hidden_width={hidden_width}",
        descriptor,
        user_data,
        hidden,
        selected_experts,
        weights,
        output,
    );
    tracing::warn!(
        target: "moe.route_diagnostic",
        "POM_MOE_ROUTE_DIAGNOSTICS callback_entry raw_args=1 descriptor_ptr={:p} user_data_ptr={:p} hidden_ptr={:p} selected_ptr={:p} weights_ptr={:p} output_ptr={:p} n_tokens={n_tokens} hidden_width={hidden_width}",
        descriptor,
        user_data,
        hidden,
        selected_experts,
        weights,
        output,
    );
}

#[derive(Debug, PartialEq)]
pub(super) struct MoeCallbackExitDiagnostic {
    pub(super) reason: &'static str,
    pub(super) raw_pointers: [usize; 6],
    pub(super) n_tokens: usize,
    pub(super) hidden_width: usize,
}

pub(super) fn build_moe_callback_exit_diagnostic(
    reason: &'static str,
    user_data: *mut c_void,
    descriptor: *const sys::llama_moe_ffn_descriptor,
    hidden: *const f32,
    selected_experts: *const i32,
    weights: *const f32,
    n_tokens: usize,
    hidden_width: usize,
    output: *mut f32,
) -> MoeCallbackExitDiagnostic {
    MoeCallbackExitDiagnostic {
        reason,
        raw_pointers: [
            user_data as usize,
            descriptor as usize,
            hidden as usize,
            selected_experts as usize,
            weights as usize,
            output as usize,
        ],
        n_tokens,
        hidden_width,
    }
}

pub(super) fn log_moe_callback_exit(diagnostic: &MoeCallbackExitDiagnostic) {
    let [user_data, descriptor, hidden, selected_experts, weights, output] =
        diagnostic.raw_pointers;
    tracing::warn!(
        target: "moe.route_diagnostic",
        "POM_MOE_ROUTE_DIAGNOSTICS callback_exit reason={} user_data_ptr=0x{user_data:x} descriptor_ptr=0x{descriptor:x} hidden_ptr=0x{hidden:x} selected_ptr=0x{selected_experts:x} weights_ptr=0x{weights:x} output_ptr=0x{output:x} n_tokens={} hidden_width={} null_args=user_data:{} descriptor:{} hidden:{} selected:{} weights:{} output:{}",
        diagnostic.reason,
        diagnostic.n_tokens,
        diagnostic.hidden_width,
        user_data == 0,
        descriptor == 0,
        hidden == 0,
        selected_experts == 0,
        weights == 0,
        output == 0,
    );
}

pub(super) unsafe extern "C" fn moe_ffn_callback(
    user_data: *mut c_void,
    descriptor: *const sys::llama_moe_ffn_descriptor,
    hidden: *const f32,
    selected_experts: *const i32,
    weights: *const f32,
    n_tokens: usize,
    hidden_width: usize,
    output: *mut f32,
) -> bool {
    let diagnostics_enabled = moe_route_diagnostics_enabled();
    if diagnostics_enabled {
        log_moe_callback_entry(
            user_data,
            descriptor,
            hidden,
            selected_experts,
            weights,
            n_tokens,
            hidden_width,
            output,
        );
    }

    macro_rules! diagnostic_return {
        ($reason:literal) => {{
            if diagnostics_enabled {
                let diagnostic = build_moe_callback_exit_diagnostic(
                    $reason,
                    user_data,
                    descriptor,
                    hidden,
                    selected_experts,
                    weights,
                    n_tokens,
                    hidden_width,
                    output,
                );
                log_moe_callback_exit(&diagnostic);
            }
            return false;
        }};
    }

    if user_data.is_null()
        || descriptor.is_null()
        || hidden.is_null()
        || selected_experts.is_null()
        || weights.is_null()
        || output.is_null()
    {
        diagnostic_return!("null_callback_argument");
    }
    let state = unsafe { &*(user_data as *const MoeCallbackState) };
    let descriptor = unsafe { &*descriptor };
    let used = match usize::try_from(descriptor.expert_count_used) {
        Ok(used) if used > 0 => used,
        _ => {
            state.fail("descriptor MoE sem quantidade válida de experts ativos".to_string());
            diagnostic_return!("invalid_expert_count_used");
        }
    };
    let expected = match n_tokens.checked_mul(hidden_width) {
        Some(expected) => expected,
        None => {
            state.fail("overflow na geometria hidden do hook".to_string());
            diagnostic_return!("hidden_geometry_overflow");
        }
    };
    let route_len = match n_tokens.checked_mul(used) {
        Some(length) => length,
        None => {
            state.fail("overflow na geometria da rota do hook".to_string());
            diagnostic_return!("route_geometry_overflow");
        }
    };
    let hidden = unsafe { std::slice::from_raw_parts(hidden, expected) };
    let selected_experts = unsafe { std::slice::from_raw_parts(selected_experts, route_len) };
    let weights = unsafe { std::slice::from_raw_parts(weights, route_len) };
    let architecture = unsafe {
        CStr::from_ptr(descriptor.architecture.as_ptr())
            .to_string_lossy()
            .into_owned()
    };
    if let Some((index, value)) = first_non_finite(hidden) {
        let kind = if value.is_nan() { "NaN" } else { "infinito" };
        state.fail(format!(
            "entrada hidden do hook MoE da arquitetura {architecture}, camada {}, índice {index} tem valor não finito ({kind}); o forward MoE anterior contaminou esta camada",
            descriptor.layer
        ));
        diagnostic_return!("hidden_input_non_finite");
    }
    let raw_non_finite = weights.iter().any(|weight| !weight.is_finite());
    let emit_route_diagnostic = state.claim_route_diagnostic()
        || (diagnostics_enabled && route_diagnostic_should_emit(false, raw_non_finite));
    if emit_route_diagnostic {
        log_moe_route_input_diagnostic(
            &architecture,
            descriptor,
            hidden,
            selected_experts,
            weights,
            n_tokens,
            hidden_width,
            expected,
            route_len,
            used,
            output,
        );
    }
    let (selected_experts, weights) = match normalize_moe_route(
        &architecture,
        descriptor.layer,
        selected_experts,
        weights,
        n_tokens,
        used,
    ) {
        Ok(route) => route,
        Err(error) => {
            state.fail(error.to_string());
            diagnostic_return!("route_normalization_failed");
        }
    };
    if emit_route_diagnostic {
        log_moe_route_post_diagnostic(descriptor.layer, &selected_experts, &weights, used);
    }
    if descriptor.gate_up_bias.present != 0
        || descriptor.gate_bias.present != 0
        || descriptor.up_bias.present != 0
        || descriptor.down_bias.present != 0
    {
        state.fail(format!(
            "arquitetura {architecture} usa bias de experts não suportado pelo hook"
        ));
        diagnostic_return!("unsupported_expert_bias");
    }
    if descriptor.down.present == 0
        || (descriptor.gate_up.present == 0
            && (descriptor.gate.present == 0 || descriptor.up.present == 0))
    {
        state.fail(format!(
            "arquitetura {architecture} não forneceu metadata completa de gate/up/down"
        ));
        diagnostic_return!("incomplete_expert_metadata");
    }
    let mut metadata_owned = Vec::new();
    for (role, tensor) in [
        (
            crate::moe::format_heterogeneous::TensorRole::GateUp,
            &descriptor.gate_up,
        ),
        (
            crate::moe::format_heterogeneous::TensorRole::Gate,
            &descriptor.gate,
        ),
        (
            crate::moe::format_heterogeneous::TensorRole::Up,
            &descriptor.up,
        ),
        (
            crate::moe::format_heterogeneous::TensorRole::Down,
            &descriptor.down,
        ),
    ] {
        if tensor.present == 0 {
            continue;
        }
        let end = tensor
            .name
            .iter()
            .position(|&character| character == 0)
            .unwrap_or(tensor.name.len());
        let name = tensor.name[..end]
            .iter()
            .map(|&character| u8::try_from(character).map_err(|_| "nome de tensor inválido"))
            .collect::<Result<Vec<_>, _>>();
        let name = match name
            .and_then(|bytes| String::from_utf8(bytes).map_err(|_| "nome de tensor não UTF-8"))
        {
            Ok(name) => name,
            Err(error) => {
                state.fail(format!("arquitetura {architecture}: {error}"));
                diagnostic_return!("invalid_tensor_name");
            }
        };
        let dtype = match u32::try_from(tensor.dtype) {
            Ok(dtype) => dtype,
            Err(_) => {
                state.fail(format!(
                    "arquitetura {architecture}: tensor {name} tem dtype negativo {}",
                    tensor.dtype
                ));
                diagnostic_return!("negative_tensor_dtype");
            }
        };
        metadata_owned.push((role, name, dtype, tensor.n_dims, tensor.ne));
    }
    let metadata = metadata_owned
        .iter()
        .map(|(role, name, dtype, n_dims, dims)| (*role, name.as_str(), *dtype, *n_dims, *dims))
        .collect::<Vec<_>>();
    if let Err(error) = state
        .runtime
        .validate_callback_metadata(descriptor.layer as u32, &metadata)
    {
        state.fail(format!("arquitetura {architecture}: {error}"));
        diagnostic_return!("callback_metadata_validation_failed");
    }
    let expert_count = match usize::try_from(descriptor.expert_count) {
        Ok(count) => count,
        Err(_) => {
            state.fail(format!(
                "arquitetura {architecture}: quantidade de experts excede usize"
            ));
            diagnostic_return!("expert_count_overflow");
        }
    };
    let started = Instant::now();
    let result = state
        .runtime
        .forward_batch_with_route(crate::moe::runtime::MoeRouteBatch {
            layer: descriptor.layer as u32,
            hidden,
            selected_experts: &selected_experts,
            weights: &weights,
            tokens: n_tokens,
            expert_count,
            expert_count_used: used,
            activation: descriptor.activation,
            weight_before_ffn: descriptor.weight_before_ffn != 0,
        });
    let values = match result {
        Ok(values) if values.len() == expected => values,
        Ok(values) => {
            state.fail(format!(
                "hook MoE devolveu {} floats, esperado {expected}",
                values.len()
            ));
            diagnostic_return!("forward_output_geometry_mismatch");
        }
        Err(error) => {
            state.fail(format!("arquitetura {architecture}: {error}"));
            diagnostic_return!("forward_failed");
        }
    };
    if let Some((index, value)) = first_non_finite(&values) {
        let kind = if value.is_nan() { "NaN" } else { "infinito" };
        state.fail(format!(
            "executor MoE da arquitetura {architecture}, camada {}, devolveu saída não finita no índice {index} ({kind}); o resultado não será enviado ao llama.cpp",
            descriptor.layer
        ));
        diagnostic_return!("forward_output_non_finite");
    }
    unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), output, expected) };
    let mut telemetry = state
        .telemetry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    telemetry.forward_us = telemetry
        .forward_us
        .saturating_add(started.elapsed().as_micros() as u64);
    let executor = state.runtime.executor_kind();
    if !telemetry.executors.contains(&executor) {
        telemetry.executors.push(executor);
    }
    true
}

pub(super) fn diagnostic_tensor_name(tensor: &sys::llama_moe_tensor_descriptor) -> String {
    let end = tensor
        .name
        .iter()
        .position(|&character| character == 0)
        .unwrap_or(tensor.name.len());
    let bytes = tensor.name[..end]
        .iter()
        .map(|&character| character as u8)
        .collect::<Vec<_>>();
    String::from_utf8_lossy(&bytes).into_owned()
}

pub(super) fn log_moe_route_input_diagnostic(
    architecture: &str,
    descriptor: &sys::llama_moe_ffn_descriptor,
    hidden: &[f32],
    selected_experts: &[i32],
    weights: &[f32],
    n_tokens: usize,
    hidden_width: usize,
    hidden_elements: usize,
    route_elements: usize,
    expert_count_used: usize,
    output: *mut f32,
) {
    let diagnostic =
        collect_moe_route_diagnostic(selected_experts, weights, n_tokens, expert_count_used);
    let selected_bytes = route_elements.saturating_mul(std::mem::size_of::<i32>());
    let weights_bytes = route_elements.saturating_mul(std::mem::size_of::<f32>());
    let hidden_bytes = hidden_elements.saturating_mul(std::mem::size_of::<f32>());
    let output_bytes = hidden_bytes;

    tracing::warn!(
        target: "moe.route_diagnostic",
        "POM_MOE_ROUTE_DIAGNOSTICS input architecture={architecture} layer={} token=0 selected_ptr={:p} selected_bytes={selected_bytes} weights_ptr={:p} weights_bytes={weights_bytes} hidden_ptr={:p} hidden_bytes={hidden_bytes} output_ptr={:p} output_bytes={output_bytes} hidden_width={hidden_width} n_tokens={n_tokens} llama_declared_expert_count={} llama_declared_expert_count_used={} llama_declared_route_elements={} pom_expected_route_elements={} selected_read_elements={} weights_read_elements={}",
        descriptor.layer,
        selected_experts.as_ptr(),
        weights.as_ptr(),
        hidden.as_ptr(),
        output,
        descriptor.expert_count,
        descriptor.expert_count_used,
        diagnostic.llama_declared_route_elements,
        diagnostic.pom_expected_route_elements,
        diagnostic.selected_read_elements,
        diagnostic.weights_read_elements,
    );
    tracing::warn!(
        target: "moe.route_diagnostic",
        "POM_MOE_ROUTE_DIAGNOSTICS raw route_dtype_assumed=i32 selected_dtype_assumed=i32 weight_dtype_assumed=f32 hidden_dtype_assumed=f32 output_dtype_assumed=f32 raw_has_nan={} raw_nan_count={} raw_has_non_finite={} raw_non_finite_count={} raw_token_zero_slots={:?}",
        diagnostic.raw_nan_count > 0,
        diagnostic.raw_nan_count,
        diagnostic.raw_non_finite_count > 0,
        diagnostic.raw_non_finite_count,
        diagnostic.token_zero_samples,
    );

    for (role, tensor) in [
        ("gate_up", &descriptor.gate_up),
        ("gate", &descriptor.gate),
        ("up", &descriptor.up),
        ("down", &descriptor.down),
        ("gate_up_bias", &descriptor.gate_up_bias),
        ("gate_bias", &descriptor.gate_bias),
        ("up_bias", &descriptor.up_bias),
        ("down_bias", &descriptor.down_bias),
        ("gate_scale", &descriptor.gate_scale),
        ("up_scale", &descriptor.up_scale),
        ("down_scale", &descriptor.down_scale),
    ] {
        tracing::warn!(
            target: "moe.route_diagnostic",
            "POM_MOE_ROUTE_DIAGNOSTICS tensor role={role} present={} dtype={} n_dims={} ne={:?} name={}",
            tensor.present,
            tensor.dtype,
            tensor.n_dims,
            tensor.ne,
            diagnostic_tensor_name(tensor),
        );
    }
}

pub(super) fn log_moe_route_post_diagnostic(
    layer: i32,
    selected_experts: &[i32],
    weights: &[f32],
    expert_count_used: usize,
) {
    let post_nan_count = weights.iter().filter(|weight| weight.is_nan()).count();
    let post_non_finite_count = weights.iter().filter(|weight| !weight.is_finite()).count();
    let token_zero_weights = weights
        .iter()
        .take(expert_count_used.min(4))
        .copied()
        .collect::<Vec<_>>();

    tracing::warn!(
        target: "moe.route_diagnostic",
        "POM_MOE_ROUTE_DIAGNOSTICS after_pom_route_normalization layer={layer} normalized_selected_ptr={:p} normalized_weights_ptr={:p} normalized_selected_elements={} normalized_weight_elements={} normalized_weight_dtype=f32 post_pom_has_nan={} post_pom_nan_count={} post_pom_has_non_finite={} post_pom_non_finite_count={} normalized_token_zero_experts={:?} normalized_token_zero_weights={:?}",
        selected_experts.as_ptr(),
        weights.as_ptr(),
        selected_experts.len(),
        weights.len(),
        post_nan_count > 0,
        post_nan_count,
        post_non_finite_count > 0,
        post_non_finite_count,
        selected_experts.iter().take(expert_count_used.min(4)).copied().collect::<Vec<_>>(),
        token_zero_weights,
    );
}

pub(super) fn normalize_moe_route(
    architecture: &str,
    layer: i32,
    selected_experts: &[i32],
    weights: &[f32],
    n_tokens: usize,
    expert_count_used: usize,
) -> Result<(Vec<i32>, Vec<f32>)> {
    let expected = n_tokens
        .checked_mul(expert_count_used)
        .ok_or_else(|| anyhow::anyhow!("overflow na rota MoE"))?;
    if selected_experts.len() != expected || weights.len() != expected {
        bail!(
            "tensor(es) de rota MoE têm {} ids e {} pesos, esperado {expected}",
            selected_experts.len(),
            weights.len()
        );
    }

    let mut selected = Vec::with_capacity(expected);
    let mut normalized = Vec::with_capacity(expected);
    for source in 0..expected {
        let token = source / expert_count_used;
        let slot = source % expert_count_used;
        let expert = selected_experts[source];
        if expert < 0 {
            bail!(
                "tensor ffn_moe_selected_experts da arquitetura {architecture}, camada {layer}, token {token}, slot {slot} tem expert negativo {expert}"
            );
        }
        let weight = weights[source];
        if !weight.is_finite() {
            let kind = if weight.is_nan() { "NaN" } else { "infinito" };
            bail!(
                "tensor ffn_moe_weights da arquitetura {architecture}, camada {layer}, token {token}, slot {slot}, expert {expert} tem valor não finito ({kind})"
            );
        }
        selected.push(expert);
        normalized.push(weight);
    }
    Ok((selected, normalized))
}

#[cfg(test)]
mod moe_route_tests {
    use super::*;

    #[test]
    fn the_first_non_finite_output_is_identified_before_the_copy() {
        assert!(first_non_finite(&[0.0, 1.0, f32::NAN])
            .is_some_and(|(index, value)| index == 2 && value.is_nan()));
        assert!(first_non_finite(&[0.0, f32::INFINITY])
            .is_some_and(|(index, value)| index == 1 && value == f32::INFINITY));
        assert_eq!(first_non_finite(&[0.0, -1.0]), None);
    }

    #[test]
    fn capture_metrics_separate_wait_copy_and_post_processing() {
        let mut slot = CaptureSlot::default();
        slot.ready_wait_us = 1_468_000;
        slot.copy_us = 1_200;
        slot.postprocess_us = 340;
        slot.copy_bytes = 1_048_576;
        slot.copy_calls = 2;

        assert_eq!(
            slot.metrics(),
            CaptureMetrics {
                ready_wait_us: 1_468_000,
                copy_us: 1_200,
                postprocess_us: 340,
                copy_bytes: 1_048_576,
                copy_calls: 2,
            }
        );
    }

    #[test]
    fn the_llama_cpp_get_rows_route_already_arrives_in_token_major_order() {
        // `ggml_get_rows` writes the [1, used, tokens] result with slot as
        // the fastest-changing dimension. After the C++ reshape to
        // [used, tokens], the callback therefore receives
        // [token0/slot0, token0/slot1, token1/slot0, ...].
        let selected_experts = [10, 20, 11, 21, 12, 22];
        let weights = [0.1, 0.4, 0.2, 0.5, 0.3, 0.6];

        let (selected, normalized) =
            normalize_moe_route("qwen35moe", 7, &selected_experts, &weights, 3, 2).unwrap();

        assert_eq!(selected, selected_experts);
        assert_eq!(normalized, weights);
    }

    #[test]
    fn expert_indices_above_127_preserve_i32() {
        let selected_experts = [132, 129, 38, 253];
        let weights = [0.4, 0.3, 0.2, 0.1];

        let (selected, normalized) =
            normalize_moe_route("qwen35moe", 1, &selected_experts, &weights, 1, 4).unwrap();

        assert_eq!(selected, selected_experts);
        assert_eq!(normalized, weights);
    }

    #[test]
    fn a_non_finite_route_identifies_tensor_token_slot_and_expert() {
        let error = normalize_moe_route("qwen35moe", 4, &[13, 27], &[f32::INFINITY, 0.5], 1, 2)
            .unwrap_err()
            .to_string();

        assert!(error.contains("ffn_moe_weights"));
        assert!(error.contains("camada 4"));
        assert!(error.contains("token 0"));
        assert!(error.contains("slot 0"));
        assert!(error.contains("expert 13"));
        assert!(error.contains("infinito"));
    }

    #[test]
    fn route_diagnostic_records_raw_input_and_token_zero_layout() {
        let selected_experts = [10, 20, 11, 21, 12, 22];
        let weights = [0.1, 0.4, 0.2, 0.5, 0.3, 0.6];

        let diagnostic = collect_moe_route_diagnostic(&selected_experts, &weights, 3, 2);

        assert_eq!(diagnostic.llama_declared_route_elements, 6);
        assert_eq!(diagnostic.pom_expected_route_elements, 6);
        assert_eq!(diagnostic.raw_nan_count, 0);
        assert_eq!(diagnostic.raw_non_finite_count, 0);
        assert_eq!(
            diagnostic.token_zero_samples,
            vec![
                MoeRouteDiagnosticSample {
                    slot: 0,
                    source_offset_elements: 0,
                    expert: 10,
                    weight: 0.1,
                },
                MoeRouteDiagnosticSample {
                    slot: 1,
                    source_offset_elements: 1,
                    expert: 20,
                    weight: 0.4,
                },
            ]
        );
    }

    #[test]
    fn diagnostic_captures_the_first_call_and_a_non_finite_route_in_any_layer() {
        assert!(route_diagnostic_should_emit(true, false));
        assert!(route_diagnostic_should_emit(false, true));
        assert!(!route_diagnostic_should_emit(false, false));
    }

    #[test]
    fn output_diagnostic_preserves_raw_arguments_and_the_return_reason() {
        let diagnostic = build_moe_callback_exit_diagnostic(
            "null_callback_argument",
            0x11usize as *mut c_void,
            0x22usize as *const sys::llama_moe_ffn_descriptor,
            0x33usize as *const f32,
            0x44usize as *const i32,
            0x55usize as *const f32,
            7,
            4096,
            0x66usize as *mut f32,
        );

        assert_eq!(diagnostic.reason, "null_callback_argument");
        assert_eq!(
            diagnostic.raw_pointers,
            [0x11, 0x22, 0x33, 0x44, 0x55, 0x66]
        );
        assert_eq!(diagnostic.n_tokens, 7);
        assert_eq!(diagnostic.hidden_width, 4096);
    }

    #[derive(Clone)]
    struct SharedLogWriter(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for SharedLogWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .map_err(|_| std::io::Error::other("writer mutex poisoned"))?
                .extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SharedLogWriter {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    #[test]
    #[ignore = "prova do callback requer POM_MOE_ROUTE_DIAGNOSTICS=1"]
    fn callback_entry_emits_the_default_warn_and_the_moe_target_on_the_same_sink() {
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .with_ansi(false)
            .with_target(false)
            .with_writer(SharedLogWriter(bytes.clone()))
            .finish();

        tracing::subscriber::with_default(subscriber, || {
            let accepted = unsafe {
                moe_ffn_callback(
                    ptr::null_mut(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    1,
                    4096,
                    ptr::null_mut(),
                )
            };
            assert!(!accepted);
        });

        let output = String::from_utf8(bytes.lock().unwrap().clone()).unwrap();
        assert!(output.contains("callback_entry_default"), "{output}");
        assert!(output.contains("callback_entry raw_args=1"), "{output}");
        assert!(
            output.contains("callback_exit reason=null_callback_argument"),
            "{output}"
        );
    }

    #[test]
    #[ignore = "prova do dispatcher global requer POM_MOE_ROUTE_DIAGNOSTICS=1"]
    fn global_callback_emits_the_diagnostic_on_an_external_thread_and_records_the_gate() {
        std::env::set_var(MOE_ROUTE_DIAGNOSTICS_ENV, "1");
        let bytes = Arc::new(Mutex::new(Vec::new()));
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .with_ansi(false)
            .with_target(false)
            .with_writer(SharedLogWriter(bytes.clone()))
            .init();

        log_moe_route_diagnostics_configuration();
        log_moe_route_diagnostics_configuration();
        let join = std::thread::spawn(|| unsafe {
            moe_ffn_callback(
                ptr::null_mut(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                ptr::null(),
                1,
                4096,
                ptr::null_mut(),
            )
        });
        assert!(!join
            .join()
            .expect("thread do callback não pode entrar em pânico"));

        let output = String::from_utf8(bytes.lock().unwrap().clone()).unwrap();
        assert_eq!(
            output
                .matches("POM_MOE_ROUTE_DIAGNOSTICS configuration")
                .count(),
            1,
            "{output}"
        );
        assert!(
            output.contains("POM_MOE_ROUTE_DIAGNOSTICS configuration"),
            "{output}"
        );
        assert!(output.contains("enabled=true"), "{output}");
        assert!(output.contains("callback_entry_default"), "{output}");
        assert!(output.contains("callback_entry raw_args=1"), "{output}");
        assert!(
            output.contains("callback_exit reason=null_callback_argument"),
            "{output}"
        );
    }
}

pub(super) unsafe extern "C" fn capture_callback(
    t: *mut sys::ggml_tensor,
    ask: bool,
    user_data: *mut c_void,
) -> bool {
    if user_data.is_null() {
        return true;
    }
    // SAFETY: `user_data` é o ponteiro que `LlmEngine::load_with_cache` registrou, e ele
    // aponta para o `Box<Mutex<CaptureSlot>>` do mesmo `LlmEngine` que é dono
    // deste contexto. O Box vive enquanto o contexto viver (ver `Drop`).
    let slot = unsafe { &*(user_data as *const Mutex<CaptureSlot>) };
    let mut slot = match slot.lock() {
        Ok(g) => g,
        Err(_) => return true,
    };
    if slot.want.is_none() {
        return true;
    }

    let name = unsafe { sys::ggml_get_name(t) };
    if name.is_null() {
        return true;
    }
    let matches = slot.want.as_deref().is_some_and(|target| {
        unsafe { CStr::from_ptr(name) }
            .to_str()
            .is_ok_and(|name| capture_target_matches(name, target))
    });

    // The two phases mean different things, and the return value means something
    // different in each - see `ggml_backend_sched_compute_splits`:
    //
    //   ask = true   "will you want this tensor?" `false` lets ggml batch this
    //                node with the following ones, which is what we want for
    //                every node but ours.
    //   ask = false  the tensor is ready. Here `false` does NOT mean "not mine":
    //                it BREAKS the evaluation loop and abandons the rest of the
    //                graph.
    //
    // Returning `ask` for a non-matching tensor got both wrong at once: it
    // claimed every node, forcing them to be computed one at a time, and then
    // returned false on the first one - so the forward pass stopped at the very
    // first node while `llama_decode` still reported success. Nothing after the
    // input embedding was ever computed.
    if ask {
        if matches {
            // Preserve the scheduler split: its ready callback is the only
            // point where this microbatch's graph buffer is stable. The
            // interval includes graph evaluation and scheduler synchronization.
            slot.capture_requested_at = Some(Instant::now());
        }
        return matches;
    }
    if !matches {
        return true; // keep going; this node was simply not ours
    }

    let ready_at = Instant::now();
    if let Some(requested_at) = slot.capture_requested_at.take() {
        slot.ready_wait_us += ready_at.saturating_duration_since(requested_at).as_micros() as u64;
    }
    let tensor_shape = unsafe { (*t).ne };
    let tensor_strides = unsafe { (*t).nb };
    let n_bytes = unsafe { sys::ggml_nbytes(t) } as usize;
    slot.last_shape = Some(tensor_shape);
    slot.last_strides = Some(tensor_strides);
    slot.last_nbytes = Some(n_bytes);
    if n_bytes % std::mem::size_of::<f32>() != 0 {
        if slot.error.is_none() {
            slot.error = Some(format!(
                "residual capture: tensor ocupa {n_bytes} bytes, não é múltiplo de f32"
            ));
        }
        return true;
    }
    let mut floats = vec![0.0f32; n_bytes / std::mem::size_of::<f32>()];
    let copy_started = Instant::now();
    unsafe {
        sys::ggml_backend_tensor_get(t, floats.as_mut_ptr() as *mut c_void, 0, n_bytes);
    }
    slot.copy_us += copy_started.elapsed().as_micros() as u64;
    slot.copy_bytes = slot.copy_bytes.saturating_add(n_bytes as u64);
    slot.copy_calls = slot.copy_calls.saturating_add(1);

    let postprocess_started = Instant::now();
    if let Err(error) = record_capture(&mut slot, tensor_shape, tensor_strides, n_bytes, floats) {
        if slot.error.is_none() {
            slot.error = Some(error.to_string());
        }
    }
    slot.postprocess_us += postprocess_started.elapsed().as_micros() as u64;
    true
}

/// Some graph variants name the residual immediately before the output head
/// `l_last-N`; older graphs and the POM capture API use `l_out-N` for the same
/// boundary. The target remains `l_out-N` so existing callers keep working.
pub(super) fn capture_target_matches(name: &str, target: &CStr) -> bool {
    let Ok(target) = target.to_str() else {
        return false;
    };
    name == target
        || target
            .strip_prefix("l_out-")
            .is_some_and(|layer| name == format!("l_last-{layer}"))
}

/// Junta o `l_out` de mais um microlote ao residual deste decode.
///
/// **Acumula, não substitui.** Um decode não é um disparo: o llama.cpp parte o
/// lote em microlotes de `n_ubatch` e chama a captura uma vez por microlote,
/// cada uma com o `l_out` daquele pedaço.
///
/// Enquanto isto atribuía, o buffer terminava com o **último**
/// microlote e o resto do chunk ia embora em silêncio - com `fit_memory=true`
/// (padrão), que limita `n_ubatch` a 128, um chunk de 512 tokens perdia 3/4 e o
/// peer seguinte recusava a requisição com "got 655360 floats, expected
/// 2621440".
///
/// A ordem sai certa porque `split_simple` do llama.cpp fatia o lote em índices
/// crescentes e contíguos, que é exatamente a ordem das linhas que
/// `BatchLayout` usa para reencontrar cada item. O llama.cpp só escolhe
/// `split_simple` quando o cache tem um stream só
/// (`n_stream = kv_unified ? 1 : n_seq_max`); com vários streams ele usa
/// `split_equal`, que agrupa por sequência e trocaria o residual de cada item
/// sem erro nenhum. **É por isso que `kv_unified` é ligado incondicionalmente**
/// nos dois pontos que criam contexto - ver `kv_seq_budget`. Mexer nisso quebra
/// esta função em silêncio.
pub(super) fn accumulate_capture(slot: &mut CaptureSlot, floats: Vec<f32>) {
    if let Some(expected) = slot.expected_floats {
        let current = slot.buf.as_ref().map_or(0, Vec::len);
        if current >= expected {
            // Gemma4 can evaluate the same named boundary more than once in a
            // decode. Once the requested batch is complete, later callbacks
            // are duplicate evaluations, not additional token rows.
            return;
        }
    }
    match slot.buf.as_mut() {
        Some(accumulated) => accumulated.extend_from_slice(&floats),
        None => slot.buf = Some(floats),
    }
}

pub(super) fn residual_capture_layout(
    tensor_shape: [i64; 4],
    tensor_strides: [usize; 4],
    n_tokens: usize,
    n_embd: usize,
) -> Result<ResidualCaptureLayout> {
    residual_capture_layout_with_token_count(tensor_shape, tensor_strides, Some(n_tokens), n_embd)
}

/// Resolve a residual layout when a multimodal decode has split the physical
/// cells into multiple graph chunks. `mtmd` can present the text chunk first
/// and the image chunk after it, so neither chunk dimension is necessarily the
/// whole decode's `n_tokens`.
fn residual_capture_layout_for_physical_chunk(
    tensor_shape: [i64; 4],
    tensor_strides: [usize; 4],
    n_embd: usize,
) -> Result<ResidualCaptureLayout> {
    residual_capture_layout_with_token_count(tensor_shape, tensor_strides, None, n_embd)
}

fn residual_capture_layout_with_token_count(
    tensor_shape: [i64; 4],
    tensor_strides: [usize; 4],
    expected_tokens: Option<usize>,
    n_embd: usize,
) -> Result<ResidualCaptureLayout> {
    if expected_tokens == Some(0) {
        bail!("residual capture: quantidade de tokens vazia");
    }
    let dims = tensor_shape
        .into_iter()
        .map(|dim| {
            usize::try_from(dim)
                .map_err(|_| anyhow::anyhow!("residual capture: dimensão negativa {dim}"))
        })
        .collect::<Result<Vec<_>>>()?;
    if dims[0] != n_embd {
        bail!(
            "residual capture: dimensão 0 é {}, esperada n_embd {}",
            dims[0],
            n_embd
        );
    }

    let mut saw_token_axis = false;
    let mut saw_unsupported_trailing = false;
    let mut saw_invalid_stride = false;
    for token_axis in 1..dims.len() {
        if expected_tokens.is_some_and(|n_tokens| dims[token_axis] != n_tokens) {
            continue;
        }
        saw_token_axis = true;
        if dims[token_axis + 1..].iter().any(|&dim| dim > 1) {
            saw_unsupported_trailing = true;
            continue;
        }
        let feature_width = dims[..token_axis]
            .iter()
            .try_fold(1usize, |width, &dimension| width.checked_mul(dimension))
            .ok_or_else(|| anyhow::anyhow!("residual capture: largura excede o limite"))?;
        let expected_stride = feature_width
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| anyhow::anyhow!("residual capture: stride excede o limite"))?;
        if tensor_strides[token_axis] != expected_stride {
            saw_invalid_stride = true;
            continue;
        }
        let mut stride = std::mem::size_of::<f32>();
        let mut contiguous = true;
        for axis in 0..=token_axis {
            if tensor_strides[axis] != stride {
                contiguous = false;
                break;
            }
            stride = stride
                .checked_mul(dims[axis])
                .ok_or_else(|| anyhow::anyhow!("residual capture: stride excede o limite"))?;
        }
        if !contiguous {
            saw_invalid_stride = true;
            continue;
        }
        // The scan is intentionally from low to high axes. When more than one
        // axis has the token count (most notably a one-token batch), the
        // lowest axis that satisfies the physical layout wins. This is
        // deterministic and avoids absorbing trailing singleton dimensions
        // into the canonical shape.
        return Ok(ResidualCaptureLayout {
            feature_shape: dims[..token_axis].to_vec(),
            token_axis,
        });
    }

    if saw_unsupported_trailing {
        bail!(
            "residual capture: shape {:?} tem dimensões posteriores ao eixo de tokens não suportadas",
            tensor_shape
        );
    }
    if saw_token_axis && saw_invalid_stride {
        bail!(
            "residual capture: shape {:?} não é contígua no eixo de tokens {}",
            tensor_shape,
            expected_tokens.unwrap_or_default()
        );
    }
    if let Some(n_tokens) = expected_tokens {
        bail!(
            "residual capture: shape {:?} não expõe o número de tokens {}",
            tensor_shape,
            n_tokens
        )
    } else {
        bail!(
            "residual capture: shape {:?} não expõe um eixo de tokens físico",
            tensor_shape
        )
    }
}

fn captured_tokens(slot: &CaptureSlot) -> Result<usize> {
    match &slot.layout {
        Some(layout) => {
            let feature_values = layout
                .feature_shape
                .iter()
                .try_fold(1usize, |values, &dimension| values.checked_mul(dimension))
                .ok_or_else(|| anyhow::anyhow!("residual capture: largura excede o limite"))?;
            if feature_values == 0 {
                bail!("residual capture: largura de features vazia");
            }
            let current = slot.buf.as_ref().map_or(0, Vec::len);
            if current % feature_values != 0 {
                bail!(
                    "residual capture: buffer acumulado de {} floats não fecha a largura de features {}",
                    current,
                    feature_values
                );
            }
            Ok(current / feature_values)
        }
        None => Ok(0),
    }
}

pub(super) fn capture_microbatch_tokens(slot: &CaptureSlot) -> Result<usize> {
    // `expected_tokens` is the batch count supplied by the decode caller. The
    // callback sees only the tensor, so after the first capture the accumulated
    // feature rows tell us how many tokens already arrived; the next physical
    // microbatch is the remaining count capped by `n_ubatch`.
    if slot.expected_tokens == 0 {
        bail!("residual capture: quantidade esperada de tokens vazia");
    }
    if slot.max_microbatch_tokens == 0 {
        bail!("residual capture: limite de microlote vazio");
    }
    let captured_tokens = captured_tokens(slot)?;
    let remaining = slot
        .expected_tokens
        .checked_sub(captured_tokens)
        .ok_or_else(|| {
            anyhow::anyhow!("residual capture: microlotes excederam o batch esperado")
        })?;
    Ok(remaining.min(slot.max_microbatch_tokens))
}

pub(super) fn record_capture(
    slot: &mut CaptureSlot,
    tensor_shape: [i64; 4],
    tensor_strides: [usize; 4],
    n_bytes: usize,
    floats: Vec<f32>,
) -> Result<()> {
    if let Some(expected) = slot.expected_floats {
        let current = slot.buf.as_ref().map_or(0, Vec::len);
        if current >= expected {
            return Ok(());
        }
    }
    let expected_microbatch_tokens = capture_microbatch_tokens(slot)?;
    let captured = captured_tokens(slot)?;
    let remaining = slot.expected_tokens.checked_sub(captured).ok_or_else(|| {
        anyhow::anyhow!("residual capture: microlotes excederam o batch esperado")
    })?;
    let layout = match residual_capture_layout(
        tensor_shape,
        tensor_strides,
        expected_microbatch_tokens,
        slot.n_embd,
    ) {
        Ok(layout) => layout,
        Err(strict_error) if slot.allow_physical_chunks => {
            let layout = residual_capture_layout_for_physical_chunk(
                tensor_shape,
                tensor_strides,
                slot.n_embd,
            )?;
            let physical_tokens = usize::try_from(tensor_shape[layout.token_axis])
                .map_err(|_| anyhow::anyhow!("residual capture: dimensão negativa"))?;
            if physical_tokens == 0 || physical_tokens > remaining {
                return Err(strict_error);
            }
            layout
        }
        Err(strict_error) => return Err(strict_error),
    };
    let microbatch_tokens = usize::try_from(tensor_shape[layout.token_axis])
        .map_err(|_| anyhow::anyhow!("residual capture: dimensão negativa"))?;
    if microbatch_tokens == 0 || microbatch_tokens > remaining {
        bail!(
            "residual capture: microlote físico de {} células excede as {} restantes",
            microbatch_tokens,
            remaining
        );
    }
    let captured_bytes = floats
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| anyhow::anyhow!("residual capture: tamanho excede o limite"))?;
    if captured_bytes != n_bytes {
        bail!(
            "residual capture: tensor shape {:?} declara {} bytes, mas a cópia tem {}",
            tensor_shape,
            n_bytes,
            captured_bytes
        );
    }
    if let Some(existing) = &slot.layout {
        if existing != &layout {
            bail!("residual capture: layout mudou entre microlotes ({existing:?} vs {layout:?})");
        }
    } else {
        let feature_values = layout
            .feature_shape
            .iter()
            .try_fold(1usize, |values, &dimension| values.checked_mul(dimension))
            .ok_or_else(|| anyhow::anyhow!("residual capture: largura excede o limite"))?;
        let expected = slot
            .expected_tokens
            .checked_mul(feature_values)
            .ok_or_else(|| anyhow::anyhow!("residual capture: tamanho esperado excede o limite"))?;
        slot.expected_floats = Some(expected);
        slot.layout = Some(layout);
    }
    let feature_shape = slot
        .layout
        .as_ref()
        .map(|layout| layout.feature_shape.as_slice())
        .ok_or_else(|| anyhow::anyhow!("residual capture: layout não definido"))?;
    let feature_values = feature_shape
        .iter()
        .try_fold(1usize, |values, &dimension| values.checked_mul(dimension))
        .ok_or_else(|| anyhow::anyhow!("residual capture: largura excede o limite"))?;
    if feature_values == 0 {
        bail!("residual capture: largura de features vazia");
    }
    let expected_microbatch = microbatch_tokens
        .checked_mul(feature_values)
        .ok_or_else(|| anyhow::anyhow!("residual capture: tamanho do microlote excede o limite"))?;
    if floats.len() != expected_microbatch {
        bail!(
            "residual capture: microlote tem {} floats, esperado {} para {} tokens e features {:?}",
            floats.len(),
            expected_microbatch,
            microbatch_tokens,
            feature_shape
        );
    }
    let expected = slot.expected_floats.unwrap_or(0);
    let current = slot.buf.as_ref().map_or(0, Vec::len);
    if current >= expected {
        return Ok(());
    }
    let Some(end) = current.checked_add(floats.len()) else {
        bail!("residual capture: tamanho do microlote excede o limite numérico");
    };
    if end > expected {
        bail!(
            "residual capture: microlote excede o batch esperado ({} + {} > {})",
            current,
            floats.len(),
            expected
        );
    }
    accumulate_capture(slot, floats);
    Ok(())
}

pub(super) fn take_captured_residual(slot: &mut CaptureSlot) -> Result<CapturedResidual> {
    if let Some(error) = slot.error.take() {
        bail!(error);
    }
    let expected = slot
        .expected_floats
        .ok_or_else(|| anyhow::anyhow!("residual capture did not observe a tensor"))?;
    let values = slot
        .buf
        .take()
        .ok_or_else(|| anyhow::anyhow!("residual capture did not fire"))?;
    if values.len() != expected {
        bail!(
            "residual capture: got {} floats, expected {}",
            values.len(),
            expected
        );
    }
    let layout = slot
        .layout
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("residual capture did not record a layout"))?;
    let mut shape = Vec::with_capacity(layout.feature_shape.len() + 1);
    shape.push(slot.expected_tokens);
    shape.extend_from_slice(&layout.feature_shape);
    Ok(CapturedResidual { values, shape })
}

// ── Ponte de diagnóstico ─────────────────────────────────────────────────────

/// Chama uma função `llama_rs_*_json` e devolve o JSON já em `serde_json`.
///
/// Toda a família devolve `LLAMA_RS_STATUS_OK` mais uma string de `malloc` que
/// **tem que** voltar por `llama_rs_string_free` - `free` de outra alocação, ou
/// um `CString::from_raw`, é liberar com um alocador que não é o dono. Por isso
/// a liberação mora aqui e não em cada chamador.
///
/// # Safety
/// `f` tem que ser uma dessas funções, com os ponteiros do llama.cpp ainda
/// vivos.
pub(super) unsafe fn json_from_wrapper(
    f: impl FnOnce(*mut *mut std::os::raw::c_char) -> sys::llama_rs_status,
) -> Option<serde_json::Value> {
    let mut raw: *mut std::os::raw::c_char = ptr::null_mut();
    let status = f(&mut raw);
    if status != sys::LLAMA_RS_STATUS_OK || raw.is_null() {
        warn!("diagnóstico do llama.cpp indisponível (status {status})");
        return None;
    }
    let text = CStr::from_ptr(raw).to_string_lossy().into_owned();
    sys::llama_rs_string_free(raw);
    match serde_json::from_str(&text) {
        Ok(v) => Some(v),
        Err(e) => {
            warn!("diagnóstico do llama.cpp veio com JSON inválido: {e}");
            None
        }
    }
}
