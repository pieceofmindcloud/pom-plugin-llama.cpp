use super::*;

/// Monta peças de token em texto válido entre chamadas.
///
/// O vocabulário devolve bytes crus por token: um token BYTE é um byte só
/// (`llama-vocab.cpp::token_to_piece`, ramo `LLAMA_TOKEN_ATTR_BYTE`) e um
/// token normal pode terminar no meio de um codepoint. Converter cada peça
/// com `from_utf8_lossy` por chamada nascia U+FFFD independente a cada
/// fragmento e rasgava a palavra no delta do chat. O acumulador guarda o
/// rabicho incompleto e só emite caractere completo; `flush` drena o resto
/// no fim do stream.
pub struct Utf8Assembler {
    pub(super) pending: Vec<u8>,
}

impl Utf8Assembler {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Acrescenta os bytes da peça e devolve o texto completo até aqui.
    ///
    /// Sequência incompleta no fim fica pendente para a peça seguinte;
    /// sequência inválida de verdade vira um único U+FFFD na hora, a mesma
    /// semântica de um `from_utf8_lossy` sobre o stream inteiro.
    pub fn feed(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    if valid > 0 {
                        out.push_str(
                            std::str::from_utf8(&self.pending[..valid])
                                .expect("prefixo validado por from_utf8"),
                        );
                    }
                    match e.error_len() {
                        // Incompleta: o caractere pode completar na próxima peça.
                        None => {
                            self.pending.drain(..valid);
                            break;
                        }
                        // Inválida: não vai completar nunca; repõe e segue.
                        Some(n) => {
                            self.pending.drain(..valid + n);
                            out.push('\u{fffd}');
                        }
                    }
                }
            }
        }
        out
    }

    /// Fim do stream: drena o rabicho que não virou caractere completo.
    pub fn flush(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

/// A configuração de um contexto do pool.
///
/// Separada para poder ser conferida sem GPU: é ela que sustenta a promessa de
/// que o pool não custa memória a mais. Como `cfg.n_ctx` é a capacidade total,
/// cada contexto recebe `cfg.n_ctx / n`, e `max_sessions = 1` porque agora cada
/// contexto atende uma sessão.
pub(super) fn per_context_config(cfg: &EngineConfig, n: usize) -> EngineConfig {
    EngineConfig {
        n_ctx: cfg.n_ctx / n.max(1) as u32,
        max_sessions: 1,
        ..cfg.clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchEntry {
    pub item_id: String,
    pub seq_id: u32,
    pub pos: u32,
    pub tokens: usize,
    pub want_logits: bool,
}

impl BatchEntry {
    #[cfg(test)]
    pub(super) fn test(
        item_id: &str,
        seq_id: u32,
        pos: u32,
        tokens: usize,
        want_logits: bool,
    ) -> Self {
        Self {
            item_id: item_id.into(),
            seq_id,
            pos,
            tokens,
            want_logits,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TokenBatchEntry {
    pub descriptor: BatchEntry,
    pub tokens: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct HiddenBatchEntry {
    pub descriptor: BatchEntry,
    pub hidden: Vec<f32>,
    pub shape: TensorShape,
    /// Four M-RoPE coordinates per physical embedding cell, in `[t, y, x, z]`
    /// order. `None` keeps the ordinary one-dimensional position path.
    pub mrope_positions: Option<Vec<[u32; 4]>>,
    /// The token ids of the rows, as the head tokenized them. A model whose
    /// layer-boundary state needs them (`Engine::state_tokens`) gets them
    /// beside the state; `None` sends the state alone.
    pub tokens: Option<Vec<i32>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchRange {
    pub rows: Range<usize>,
    pub logits_index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchLayout {
    pub(super) ranges: HashMap<String, BatchRange>,
    pub(super) total_tokens: usize,
}

impl BatchLayout {
    pub fn new(entries: &[BatchEntry]) -> Result<Self> {
        if entries.is_empty() {
            bail!("batch sem entradas");
        }

        let mut ranges = HashMap::with_capacity(entries.len());
        let mut seq_ids = HashSet::with_capacity(entries.len());
        let mut cursor = 0usize;
        for entry in entries {
            if entry.item_id.is_empty() {
                bail!("batch item sem item_id");
            }
            if ranges.contains_key(&entry.item_id) {
                bail!("item_id duplicado: {}", entry.item_id);
            }
            if !seq_ids.insert(entry.seq_id) {
                bail!("seq_id duplicado: {}", entry.seq_id);
            }
            if entry.tokens == 0 {
                bail!("item {} sem tokens", entry.item_id);
            }
            let end = cursor
                .checked_add(entry.tokens)
                .ok_or_else(|| anyhow::anyhow!("batch excede o limite numérico"))?;
            let logits_index = entry.want_logits.then_some(end - 1);
            ranges.insert(
                entry.item_id.clone(),
                BatchRange {
                    rows: cursor..end,
                    logits_index,
                },
            );
            cursor = end;
        }

        Ok(Self {
            ranges,
            total_tokens: cursor,
        })
    }

    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    pub fn range(&self, item_id: &str) -> Option<&BatchRange> {
        self.ranges.get(item_id)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchDecodeOutput {
    pub layout: BatchLayout,
    pub(super) residual: Option<Vec<f32>>,
    pub(super) residual_shape: TensorShape,
    pub(super) n_embd: usize,
}

impl BatchDecodeOutput {
    #[cfg(test)]
    pub(super) fn test(layout: BatchLayout, residual: Vec<f32>, n_embd: usize) -> Self {
        Self {
            layout,
            residual: Some(residual),
            residual_shape: vec![0, n_embd],
            n_embd,
        }
    }

    pub fn residual_for(&self, item_id: &str, n_embd: usize) -> Result<(Vec<f32>, TensorShape)> {
        let range = self
            .layout
            .range(item_id)
            .ok_or_else(|| anyhow::anyhow!("item_id ausente no resultado: {item_id}"))?;
        let residual = self
            .residual
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("batch não capturou residual"))?;
        if n_embd != self.n_embd {
            bail!(
                "n_embd solicitado {} diverge do batch {}",
                n_embd,
                self.n_embd
            );
        }
        let feature_width = self
            .residual_shape
            .get(1..)
            .ok_or_else(|| anyhow::anyhow!("shape do residual ausente no resultado"))?
            .iter()
            .try_fold(1usize, |width, &dimension| width.checked_mul(dimension))
            .ok_or_else(|| anyhow::anyhow!("largura de residual excede o limite numérico"))?;
        let start = range
            .rows
            .start
            .checked_mul(feature_width)
            .ok_or_else(|| anyhow::anyhow!("offset de residual excede o limite numérico"))?;
        let end = range
            .rows
            .end
            .checked_mul(feature_width)
            .ok_or_else(|| anyhow::anyhow!("fim de residual excede o limite numérico"))?;
        let values = residual
            .get(start..end)
            .ok_or_else(|| anyhow::anyhow!("residual incompleto para item {item_id}"))?
            .to_vec();
        let mut shape = self.residual_shape.clone();
        shape[0] = range.rows.len();
        Ok((values, shape))
    }
}

#[cfg(test)]
mod batch_layout_tests {
    use super::*;

    #[test]
    fn layout_marks_the_global_logits_position_of_each_item() {
        let entries = vec![
            BatchEntry::test("a", 0, 10, 3, true),
            BatchEntry::test("b", 1, 4, 2, true),
            BatchEntry::test("c", 2, 8, 1, false),
        ];
        let layout = BatchLayout::new(&entries).unwrap();

        assert_eq!(layout.total_tokens(), 6);
        assert_eq!(layout.range("a").unwrap().rows, 0..3);
        assert_eq!(layout.range("a").unwrap().logits_index, Some(2));
        assert_eq!(layout.range("b").unwrap().rows, 3..5);
        assert_eq!(layout.range("b").unwrap().logits_index, Some(4));
        assert_eq!(layout.range("c").unwrap().rows, 5..6);
        assert_eq!(layout.range("c").unwrap().logits_index, None);
    }

    #[test]
    fn layout_refuses_a_duplicate_or_tokenless_item() {
        let duplicate = vec![
            BatchEntry::test("a", 0, 0, 1, false),
            BatchEntry::test("a", 1, 0, 1, false),
        ];
        assert!(BatchLayout::new(&duplicate)
            .unwrap_err()
            .to_string()
            .contains("item_id duplicado"));

        let empty = vec![BatchEntry::test("empty", 0, 0, 0, false)];
        assert!(BatchLayout::new(&empty)
            .unwrap_err()
            .to_string()
            .contains("sem tokens"));
    }

    #[test]
    fn batch_output_returns_the_residual_of_the_requested_item_only() {
        let entries = vec![
            BatchEntry::test("a", 0, 0, 2, false),
            BatchEntry::test("b", 1, 0, 1, false),
        ];
        let layout = BatchLayout::new(&entries).unwrap();
        let output = BatchDecodeOutput::test(layout, vec![1.0, 2.0, 3.0], 1);

        assert_eq!(
            output.residual_for("a", 1).unwrap(),
            (vec![1.0, 2.0], vec![2, 1])
        );
        assert_eq!(
            output.residual_for("b", 1).unwrap(),
            (vec![3.0], vec![1, 1])
        );
        assert!(output.residual_for("missing", 1).is_err());
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    fn cfg() -> EngineConfig {
        EngineConfig {
            n_ctx: 120_320,
            mmproj_path: None,
            mtp_draft_path: None,
            offload_kqv: true,
            cache_type_k: KvCacheType::F16,
            cache_type_v: KvCacheType::F16,
            fit_memory: true,
            fit_memory_margin_mb: crate::DEFAULT_FIT_MEMORY_MARGIN_MB,
            n_batch: 512,
            n_ubatch: crate::DEFAULT_N_UBATCH,
            n_gpu_layers: 99,
            n_threads: 0,
            max_sessions: 2,
            need_embeddings: true,
            mtp: false,
            mtp_cpu_fallback_enabled: true,
            mtp_placement: Default::default(),
            moe_experts_in_ram: false,
            moe_cache_mib: 0,
            rs_seq_window: 0,
        }
    }

    #[test]
    fn gpu_kv_failure_fallback_keeps_the_requested_gpu_layers() {
        let cfg = cfg();
        let fallback = LlmEngine::cpu_kv_fallback(&cfg).expect("GPU KV was enabled");

        assert!(!fallback.offload_kqv);
        assert_eq!(fallback.n_gpu_layers, cfg.n_gpu_layers);
        assert_eq!(fallback.n_ctx, cfg.n_ctx);
    }

    #[test]
    fn cpu_kv_configuration_does_not_retry() {
        let mut cfg = cfg();
        cfg.offload_kqv = false;

        assert!(LlmEngine::cpu_kv_fallback(&cfg).is_none());
    }

    #[test]
    fn fit_only_auto_adjusts_the_default_gpu_layer_setting() {
        assert!(fit_can_adjust_layers(&cfg()));

        let mut explicit_layers = cfg();
        explicit_layers.n_gpu_layers = 20;
        assert!(!fit_can_adjust_layers(&explicit_layers));

        let mut disabled = cfg();
        disabled.fit_memory = false;
        assert!(!fit_can_adjust_layers(&disabled));
    }

    /// A regressão de 2026-08-21: prompt vindo de um cliente OpenAI externo
    /// morria com "got 655360 floats, expected 2621440 (512x5120)".
    ///
    /// 655360 = 128 x 5120, e 128 era o microlote máximo do fit da altura - hoje ele
    /// vale 512, por escolha do operador. O chunk de 512 tokens virava quatro
    /// microlotes, a captura substituia a cada um, e sobrava so o ultimo
    /// quarto. Pela UI passava despercebido porque prompt curto cabe num
    /// microlote so - e por isso o teste abaixo usa mais de um, sem depender
    /// do valor concreto da constante.
    #[test]
    fn capture_accumulates_every_microbatch_of_one_decode() {
        let mut slot = CaptureSlot {
            want: None,
            buf: None,
            expected_tokens: 0,
            max_microbatch_tokens: 0,
            allow_physical_chunks: false,
            n_embd: 0,
            expected_floats: None,
            layout: None,
            error: None,
            last_shape: None,
            last_strides: None,
            last_nbytes: None,
            copy_us: 0,
            ..Default::default()
        };

        let n_embd = 4;
        let ubatch = 2;
        let tokens = 6; // tres microlotes, como um chunk maior que `n_ubatch`
        for micro in 0..(tokens / ubatch) {
            let start = (micro * ubatch * n_embd) as f32;
            let chunk: Vec<f32> = (0..(ubatch * n_embd)).map(|i| start + i as f32).collect();
            accumulate_capture(&mut slot, chunk);
        }

        let captured = slot.buf.expect("a captura tem que ter disparado");
        assert_eq!(
            captured.len(),
            tokens * n_embd,
            "o residual do chunk inteiro, nao o do ultimo microlote"
        );
        let expected: Vec<f32> = (0..(tokens * n_embd)).map(|i| i as f32).collect();
        assert_eq!(
            captured, expected,
            "os microlotes entram na ordem em que o split_simple os produz"
        );
    }

    /// Sem isto, o residual de uma requisicao entraria concatenado no da
    /// proxima - o modo de falha que a acumulacao introduziria se o buffer
    /// nao fosse zerado a cada decode.
    #[test]
    fn a_new_decode_starts_from_an_empty_capture() {
        let mut slot = CaptureSlot {
            want: None,
            buf: None,
            expected_tokens: 0,
            max_microbatch_tokens: 0,
            allow_physical_chunks: false,
            n_embd: 0,
            expected_floats: None,
            layout: None,
            error: None,
            last_shape: None,
            last_strides: None,
            last_nbytes: None,
            copy_us: 0,
            ..Default::default()
        };
        accumulate_capture(&mut slot, vec![1.0, 2.0]);
        slot.buf = None; // o que `reset_capture_slot` faz antes de cada decode
        accumulate_capture(&mut slot, vec![3.0, 4.0]);
        assert_eq!(slot.buf.expect("captura"), vec![3.0, 4.0]);
    }

    #[test]
    fn capture_ignores_duplicate_named_boundary_after_expected_rows() {
        let mut slot = CaptureSlot {
            want: None,
            buf: None,
            expected_tokens: 1,
            max_microbatch_tokens: 1,
            allow_physical_chunks: false,
            n_embd: 4,
            expected_floats: Some(4),
            layout: None,
            error: None,
            last_shape: None,
            last_strides: None,
            last_nbytes: None,
            copy_us: 0,
            ..Default::default()
        };

        accumulate_capture(&mut slot, vec![1.0, 2.0, 3.0, 4.0]);
        accumulate_capture(&mut slot, vec![5.0, 6.0, 7.0, 8.0]);

        assert_eq!(slot.buf.as_deref(), Some(&[1.0, 2.0, 3.0, 4.0][..]));
    }

    fn capture_slot_for_test(
        expected_tokens: usize,
        n_embd: usize,
        max_microbatch_tokens: usize,
    ) -> CaptureSlot {
        CaptureSlot {
            want: None,
            buf: None,
            expected_tokens,
            max_microbatch_tokens,
            allow_physical_chunks: false,
            n_embd,
            expected_floats: None,
            layout: None,
            error: None,
            last_shape: None,
            last_strides: None,
            last_nbytes: None,
            copy_us: 0,
            ..Default::default()
        }
    }

    #[test]
    fn residual_capture_without_extra_dimension_keeps_the_current_wire_shape() {
        let mut slot = capture_slot_for_test(2, 4, 4);
        record_capture(
            &mut slot,
            [4, 2, 1, 1],
            [4, 16, 32, 32],
            2 * 4 * 4,
            (0..8).map(|value| value as f32).collect(),
        )
        .unwrap();

        let captured = take_captured_residual(&mut slot).unwrap();
        assert_eq!(captured.shape, vec![2, 4]);
        assert_eq!(
            captured.values,
            (0..8).map(|value| value as f32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn residual_capture_accepts_the_classic_ggml_shape_with_trailing_singletons() {
        let mut slot = capture_slot_for_test(3, 4096, 4096);
        let values = (0..(3 * 4096))
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        record_capture(
            &mut slot,
            [4096, 3, 1, 1],
            [4, 16_384, 49_152, 49_152],
            values.len() * 4,
            values.clone(),
        )
        .unwrap();

        let captured = take_captured_residual(&mut slot).unwrap();
        assert_eq!(captured.shape, vec![3, 4096]);
        assert_eq!(captured.values, values);
    }

    #[test]
    fn residual_capture_accepts_a_model_without_hyperconnection_streams() {
        let mut slot = capture_slot_for_test(4, 8, 8);
        let values = (0..32).map(|value| value as f32).collect::<Vec<_>>();
        record_capture(
            &mut slot,
            [8, 4, 1, 1],
            [4, 32, 128, 128],
            values.len() * 4,
            values.clone(),
        )
        .unwrap();

        let captured = take_captured_residual(&mut slot).unwrap();
        assert_eq!(captured.shape, vec![4, 8]);
        assert_eq!(captured.values, values);
    }

    #[test]
    fn residual_capture_preserves_extra_stream_dimensions() {
        let mut slot = capture_slot_for_test(11, 4096, 4096);
        let values = (0..(11 * 4096 * 4))
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        record_capture(
            &mut slot,
            [4096, 4, 11, 1],
            [4, 16_384, 65_536, 720_896],
            values.len() * 4,
            values.clone(),
        )
        .unwrap();

        let captured = take_captured_residual(&mut slot).unwrap();
        assert_eq!(captured.shape, vec![11, 4096, 4]);
        assert_eq!(captured.values, values);
    }

    #[test]
    fn residual_capture_accumulates_extra_dimensions_in_microbatch_order() {
        let mut slot = capture_slot_for_test(5, 4, 2);
        record_capture(
            &mut slot,
            [4, 2, 2, 1],
            [4, 16, 32, 64],
            2 * 8 * 4,
            (0..16).map(|value| value as f32).collect(),
        )
        .unwrap();
        record_capture(
            &mut slot,
            [4, 2, 2, 1],
            [4, 16, 32, 64],
            2 * 8 * 4,
            (16..32).map(|value| value as f32).collect(),
        )
        .unwrap();
        record_capture(
            &mut slot,
            [4, 2, 1, 1],
            [4, 16, 32, 32],
            8 * 4,
            (32..40).map(|value| value as f32).collect(),
        )
        .unwrap();

        let captured = take_captured_residual(&mut slot).unwrap();
        assert_eq!(captured.shape, vec![5, 4, 2]);
        let expected = (0..40).map(|value| value as f32).collect::<Vec<_>>();
        assert_eq!(captured.values, expected);
    }

    #[test]
    fn residual_capture_names_unsupported_trailing_dimensions() {
        let mut slot = capture_slot_for_test(2, 4, 8);
        let error = record_capture(
            &mut slot,
            [4, 2, 2, 3],
            [4, 16, 32, 64],
            2 * 3 * 8 * 4,
            vec![0.0; 2 * 3 * 8],
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("dimensões posteriores"),
            "erro inesperado: {error}"
        );
    }

    #[test]
    fn residual_capture_preserves_an_internal_singleton_before_tokens() {
        let mut slot = capture_slot_for_test(2, 4, 4);
        record_capture(
            &mut slot,
            [4, 1, 2, 1],
            [4, 16, 16, 32],
            2 * 4 * 4,
            vec![0.0; 2 * 4],
        )
        .unwrap();
        let captured = take_captured_residual(&mut slot).unwrap();
        assert_eq!(captured.shape, vec![2, 4, 1]);
        assert_eq!(captured.values, vec![0.0; 2 * 4]);
    }

    #[test]
    fn residual_capture_does_not_use_n_embd_in_to_find_the_token_axis() {
        // The surrounding model may have n_embd=4 and n_embd_inp=8. The
        // residual is still identified from its physical shape and token
        // count, not from that input-only metadata.
        let mut slot = capture_slot_for_test(3, 4, 3);
        let values = (0..12).map(|value| value as f32).collect::<Vec<_>>();
        record_capture(
            &mut slot,
            [4, 3, 1, 1],
            [4, 16, 48, 48],
            values.len() * 4,
            values.clone(),
        )
        .unwrap();

        let captured = take_captured_residual(&mut slot).unwrap();
        assert_eq!(captured.shape, vec![3, 4]);
        assert_eq!(captured.values, values);
    }

    #[test]
    fn residual_capture_resolves_ambiguous_single_token_axis_deterministically() {
        let mut slot = capture_slot_for_test(1, 4, 1);
        record_capture(
            &mut slot,
            [4, 1, 1, 1],
            [4, 16, 16, 16],
            4 * 4,
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();

        let captured = take_captured_residual(&mut slot).unwrap();
        assert_eq!(captured.shape, vec![1, 4]);
        assert_eq!(captured.values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn fit_mode_caps_compute_microbatch_without_changing_prompt_batch() {
        let cfg = cfg();
        assert_eq!(cfg.n_batch, 512);
        assert_eq!(effective_n_ubatch(&cfg), 512);

        let mut no_fit = cfg;
        no_fit.fit_memory = false;
        assert_eq!(effective_n_ubatch(&no_fit), 512);

        let mut small_batch = no_fit;
        small_batch.n_batch = 64;
        small_batch.fit_memory = true;
        assert_eq!(effective_n_ubatch(&small_batch), 64);
    }

    #[test]
    fn context_batch_params_cap_a_large_mtp_batch_with_fit_enabled() {
        let mut cfg = cfg();
        cfg.mtp = true;
        cfg.n_batch = 4096;

        let mut params = unsafe { sys::llama_context_default_params() };
        apply_context_batch_params(&mut params, &cfg);

        assert_eq!(params.n_batch, 4096);
        // 512 fixo de proposito: e a escolha do operador para o microlote máximo do fit e
        // ja foi revertida para 128 uma vez. Comparar com a constante deixaria
        // o teste passar na mesma; o literal faz esta mudanca ficar vermelha.
        assert_eq!(params.n_ubatch, 512);
    }

    #[test]
    fn threads_default_to_everything_the_process_may_use() {
        let available = std::thread::available_parallelism().unwrap().get() as i32;
        assert_eq!(effective_n_threads(0), available);
        assert_eq!(effective_n_threads(-1), available);
        assert_eq!(effective_n_threads(3), 3);

        let mut params = unsafe { sys::llama_context_default_params() };
        apply_thread_params(&mut params, &cfg());
        assert_eq!(params.n_threads, available);
        assert_eq!(params.n_threads_batch, available);
    }

    #[test]
    fn context_batch_params_follow_the_requested_batch_when_fit_is_disabled() {
        let mut cfg = cfg();
        cfg.mtp = true;
        cfg.fit_memory = false;
        cfg.n_batch = 4096;

        let mut params = unsafe { sys::llama_context_default_params() };
        apply_context_batch_params(&mut params, &cfg);

        assert_eq!(params.n_batch, 4096);
        assert_eq!(params.n_ubatch, 4096);
    }

    #[test]
    fn mtp_configuration_requests_loading_nextn_weights() {
        let mut cfg = cfg();
        assert_eq!(empty_fit_plan(&cfg).mtp_mode, MtpFitMode::Disabled);

        cfg.mtp = true;
        assert_eq!(empty_fit_plan(&cfg).mtp_mode, MtpFitMode::Vram);
    }

    /// O pool nao pode custar memoria a mais: o cache de cada contexto e o
    /// mesmo pedaco que a sessao ja tinha. Numeros da ksia em 2026-08-14, onde
    /// um contexto de 120.320 atendia 2 sessoes de 60.160.
    #[test]
    fn each_context_gets_the_slice_a_session_already_had() {
        let by_context = per_context_config(&cfg(), 2);
        assert_eq!(by_context.n_ctx, 60_160);
        assert_eq!(by_context.n_ctx * 2, cfg().n_ctx, "o total nao cresce");
    }

    /// Cada contexto atende uma sessao; quem divide agora e o pool.
    #[test]
    fn a_pooled_context_serves_one_session() {
        assert_eq!(per_context_config(&cfg(), 4).max_sessions, 1);
    }

    /// O resto da configuracao e do no, nao do contexto, e nao pode divergir
    /// entre irmaos.
    #[test]
    fn everything_else_is_carried_over_unchanged() {
        let p = per_context_config(&cfg(), 3);
        assert_eq!(p.n_batch, 512);
        assert_eq!(p.n_gpu_layers, 99);
        assert!(p.offload_kqv);
        assert!(p.need_embeddings);
    }

    /// Zero contextos nao existe, e dividir por zero menos ainda.
    #[test]
    fn a_pool_of_zero_is_treated_as_one() {
        assert_eq!(per_context_config(&cfg(), 0).n_ctx, cfg().n_ctx);
    }
}
