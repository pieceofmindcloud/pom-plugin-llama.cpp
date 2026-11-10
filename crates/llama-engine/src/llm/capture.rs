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
