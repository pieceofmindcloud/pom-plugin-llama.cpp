#[cfg(test)]
use super::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_processor_without_the_compiled_instruction_sets_is_named() {
        let built_with = ["sse3", "avx", "avx2", "fma"];
        let old_cpu = |instruction_set: &str| instruction_set == "sse3";

        assert_eq!(
            missing_instruction_sets(&built_with, old_cpu),
            vec!["avx", "avx2", "fma"]
        );
        assert!(missing_instruction_sets(&built_with, |_| true).is_empty());
        assert!(missing_instruction_sets(&[], |_| false).is_empty());
    }

    #[test]
    fn receiver_accepts_a_hidden_state_with_expanded_streams() {
        let shape = vec![11, 4096, 4];
        let values_len = 11 * 4096 * 4;

        validate_hidden_tensor(values_len, &shape, 11, 4096, 4096, 16_384, "hidden state")
            .expect("a largura expandida do eixo de fluxos deve ser aceita");
    }

    #[test]
    fn receiver_accepts_a_hidden_state_without_streams() {
        let shape = vec![3, 8];

        validate_hidden_tensor(3 * 8, &shape, 3, 8, 8, 32, "hidden state")
            .expect("o modelo sem fluxos deve manter a validação atual");
    }

    #[test]
    fn receiver_keeps_the_singletons_of_a_model_without_streams() {
        let shape = vec![3, 8, 1];

        validate_hidden_tensor(3 * 8, &shape, 3, 8, 8, 32, "hidden state")
            .expect("dimensões singleton não devem virar um eixo de fluxo");
    }

    #[test]
    fn receiver_refuses_an_incompatible_hidden_state_width_naming_the_shape() {
        let shape = vec![11, 4096, 3];

        let error = validate_hidden_tensor(
            11 * 4096 * 3,
            &shape,
            11,
            4096,
            4096,
            16_384,
            "hidden state",
        )
        .expect_err("uma largura de fluxo incompatível deve continuar recusada")
        .to_string();

        assert!(error.contains("[11, 4096, 3]"), "erro inesperado: {error}");
        assert!(
            error.contains("largura de entrada 12288"),
            "erro inesperado: {error}"
        );
    }

    #[test]
    fn the_sender_residual_over_the_wire_is_accepted_by_the_receiver() {
        let mut slot = CaptureSlot {
            want: None,
            buf: None,
            expected_tokens: 11,
            max_microbatch_tokens: 11,
            allow_physical_chunks: false,
            n_embd: 4096,
            expected_floats: None,
            layout: None,
            error: None,
            last_shape: None,
            last_strides: None,
            last_nbytes: None,
            copy_us: 0,
            ..Default::default()
        };
        let values = (0..(11 * 4096 * 4))
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        record_capture(
            &mut slot,
            [4096, 4, 11, 1],
            [4, 16_384, 65_536, 720_896],
            values.len() * 4,
            values,
        )
        .unwrap();
        let emitted = take_captured_residual(&mut slot).unwrap();

        let frame = crate::wire::encode(
            &serde_json::json!({"session_id": "round-trip"}),
            Some((&emitted.values, emitted.shape.clone())),
            crate::wire::WireDtype::F32,
        )
        .unwrap();
        let (_, received) = crate::wire::decode(&frame).unwrap();
        let (received_values, received_shape) = received.unwrap();

        let width = validate_hidden_tensor(
            received_values.len(),
            &received_shape,
            11,
            4096,
            4096,
            16_384,
            "hidden state",
        )
        .expect("o receptor deve aceitar exatamente a forma emitida");
        assert_eq!(received_shape, vec![11, 4096, 4]);
        assert_eq!(width, 16_384);
    }

    #[test]
    fn multimodal_capture_accumulates_microbatches_by_physical_cells() {
        let mut slot = CaptureSlot {
            want: None,
            buf: None,
            expected_tokens: 257,
            max_microbatch_tokens: 512,
            allow_physical_chunks: true,
            n_embd: 2048,
            expected_floats: None,
            layout: None,
            error: None,
            last_shape: None,
            last_strides: None,
            last_nbytes: None,
            copy_us: 0,
            ..Default::default()
        };
        let record = |slot: &mut CaptureSlot, tokens: usize, start: usize| {
            let values = (start..start + tokens * 2048)
                .map(|value| value as f32)
                .collect::<Vec<_>>();
            record_capture(
                slot,
                [2048, tokens as i64, 1, 1],
                [4, 2048 * 4, 2048 * tokens * 4, 2048 * tokens * 4],
                values.len() * 4,
                values,
            )
        };

        record(&mut slot, 39, 0).expect("o microlote textual deve ser capturado");
        record(&mut slot, 218, 39 * 2048)
            .expect("o microlote expandido pela imagem deve ser capturado");

        let captured = take_captured_residual(&mut slot).expect("a sequência física completa");
        assert_eq!(captured.shape, vec![257, 2048]);
        assert_eq!(captured.values.len(), 257 * 2048);
        assert_eq!(captured.values[39 * 2048], (39 * 2048) as f32);
    }

    #[test]
    fn capture_target_accepts_an_alternate_last_layer_alias() {
        let target = CString::new("l_out-19").unwrap();
        assert!(capture_target_matches("l_out-19", &target));
        assert!(capture_target_matches("l_last-19", &target));
        assert!(!capture_target_matches("l_last-18", &target));
        assert!(!capture_target_matches("output-19", &target));
    }

    #[test]
    fn output_width_mismatch_disables_unsafe_generic_embedding_copy() {
        assert!(!generic_embeddings_compatible(true, 4096, 16384));
        assert!(generic_embeddings_compatible(true, 4096, 4096));
        assert!(!generic_embeddings_compatible(false, 4096, 4096));
    }

    const GEMMA4_TEMPLATE_FIXTURE: &str = r#"
        {{- bos_token -}}
        {%- if (enable_thinking is defined and enable_thinking) or messages[0]['role'] in ['system', 'developer'] -%}
            {{- '<|turn>system\n' -}}
            {%- if enable_thinking is defined and enable_thinking -%}
                {{- '<|think|>\n' -}}
            {%- endif -%}
            {{- '<turn|>\n' -}}
        {%- endif -%}
        {%- for message in messages -%}
            {{- '<|turn>' + ('model' if message['role'] == 'assistant' else message['role']) + '\n' -}}
            {{- message['content'] | trim -}}
            {{- '<turn|>\n' -}}
        {%- endfor -%}
        {%- if add_generation_prompt -%}
            {{- '<|turn>model\n' -}}
            {%- if not enable_thinking | default(false) -%}
                {{- '<|channel>thought\n<channel|>' -}}
            {%- endif -%}
        {%- endif -%}
    "#;

    #[test]
    fn profile_sync_defaults_on_and_accepts_explicit_off_values() {
        assert!(profile_backend_sync_value(None));
        assert!(profile_backend_sync_value(Some("1")));
        assert!(!profile_backend_sync_value(Some("0")));
        assert!(!profile_backend_sync_value(Some("false")));
    }

    #[test]
    fn cache_types_map_to_the_llama_context_types() {
        assert_eq!(cache_type_to_ggml(KvCacheType::F16), sys::GGML_TYPE_F16);
        assert_eq!(cache_type_to_ggml(KvCacheType::Q8_0), sys::GGML_TYPE_Q8_0);
        assert_eq!(
            cache_type_to_ggml(KvCacheType::Iq4Nl),
            sys::GGML_TYPE_IQ4_NL
        );
    }

    #[test]
    fn model_cache_retains_the_handle_until_clear() {
        let cache = ModelCache::new();
        let loaded = Arc::new(LoadedModel {
            model: std::ptr::null_mut(),
            vocab: std::ptr::null(),
            n_embd: 1,
            n_vocab: 1,
            mtp_weights_loaded: false,
        });
        cache.put(
            "model.gguf",
            loaded.clone(),
            true,
            7,
            false,
            false,
        );

        let first = cache
            .get("model.gguf", false, false)
            .unwrap()
            .loaded;
        let second = cache
            .get("model.gguf", false, false)
            .unwrap()
            .loaded;
        assert!(Arc::ptr_eq(&first, &second));
        assert!(cache.get("model.gguf", false, true).is_none());
        assert!(cache.get("other.gguf", false, false).is_none());
        assert!(cache.get("model.gguf", true, false).is_none());

        drop((first, second));
        cache.clear();
        assert!(cache.get("model.gguf", false, false).is_none());
        assert_eq!(Arc::strong_count(&loaded), 1);
    }

    #[test]
    fn cache_drops_the_target_after_the_mtp_fallback_to_ram() {
        let cache = ModelCache::new();
        let loaded = Arc::new(LoadedModel {
            model: std::ptr::null_mut(),
            vocab: std::ptr::null(),
            n_embd: 1,
            n_vocab: 1,
            mtp_weights_loaded: false,
        });
        cache.put(
            "model.gguf",
            loaded.clone(),
            true,
            7,
            false,
            true,
        );

        assert!(cache.clear_mtp_fallback());
        assert!(cache.get("model.gguf", true, true).is_none());
        assert!(!cache.clear_mtp_fallback());
        assert_eq!(Arc::strong_count(&loaded), 1);
    }

    #[test]
    fn cache_drops_a_target_without_mtp_before_a_new_attempt() {
        let cache = ModelCache::new();
        let loaded = Arc::new(LoadedModel {
            model: std::ptr::null_mut(),
            vocab: std::ptr::null(),
            n_embd: 1,
            n_vocab: 1,
            mtp_weights_loaded: false,
        });
        cache.put(
            "model.gguf",
            loaded.clone(),
            true,
            7,
            false,
            false,
        );

        assert!(cache.get("model.gguf", true, false).is_none());
        assert!(cache.clear_mtp_fallback());
        assert_eq!(Arc::strong_count(&loaded), 1);
    }

    #[test]
    fn mtp_fallback_selects_a_companion_with_nextn_and_never_a_target_without_weights() {
        let target = LoadedModel {
            model: std::ptr::null_mut(),
            vocab: std::ptr::null(),
            n_embd: 1,
            n_vocab: 1,
            mtp_weights_loaded: false,
        };
        let companion = LoadedModel {
            model: std::ptr::null_mut(),
            vocab: std::ptr::null(),
            n_embd: 1,
            n_vocab: 1,
            mtp_weights_loaded: true,
        };

        let selected = LlmEngine::mtp_model_for_attach(&target, Some(&companion));
        assert!(selected.is_some_and(|model| std::ptr::eq(model, &companion)));
        assert!(LlmEngine::mtp_model_for_attach(&target, None).is_none());
    }

    /// Testes do comportamento vigente do sampler: conjunto de candidatos,
    /// limites determinísticos e sorteio. Não são provas de equivalência com
    /// implementações anteriores, porque top-p e desempates evoluíram.
    mod sampler_behaviour {
        use super::*;

        fn cfg(temperature: f32, top_k: i32, top_p: f32) -> SamplingConfig {
            SamplingConfig {
                temperature,
                top_p,
                top_k,
                repeat_penalty: 1.0,
                repeat_last_n: 0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                min_p: 0.0,
                dry: Default::default(),
                reasoning: Default::default(),
            }
        }

        fn candidate_ids(logits: &[f32], c: &SamplingConfig) -> Vec<i32> {
            sampling_candidates(logits, &[], c)
                .into_iter()
                .map(|(id, _)| id as i32)
                .collect()
        }

        /// Zero temperature is argmax, and argmax is the highest logit.
        #[test]
        fn zero_temperature_is_argmax() {
            let logits = [0.1, 9.0, 0.3, 2.0];
            assert_eq!(sample_token(&logits, &[], &cfg(0.0, 40, 1.0)), 1);
        }

        /// With top-p disabled, top-k has to be the only thing narrowing the
        /// field: no token outside the k highest logits may ever be drawn.
        #[test]
        fn top_k_confines_the_draw_to_the_k_highest_logits() {
            let _sampling = SAMPLER_TEST_LOCK.lock().unwrap();
            // Indices 3 and 1 hold the two highest logits.
            let logits = [0.0, 5.0, 1.0, 9.0, 2.0, 0.5];
            let c = cfg(1.0, 2, 1.0);

            for _ in 0..500 {
                let picked = sample_token(&logits, &[], &c);
                assert!(picked == 1 || picked == 3, "saiu fora do top-2: {picked}");
            }
        }

        #[test]
        fn top_k_does_not_expand_past_k_on_a_tie() {
            let logits = [7.0, 7.0, 7.0, 1.0];
            assert_eq!(candidate_ids(&logits, &cfg(1.0, 1, 1.0)), vec![0]);
        }

        /// The repetition penalty divides positive logits, which can move a
        /// token out of the running entirely.
        #[test]
        fn the_repetition_penalty_can_push_a_token_out_of_the_top_k() {
            let logits = [10.0, 9.0, 0.0];
            let penalised = SamplingConfig {
                repeat_penalty: 4.0,
                repeat_last_n: 4,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                min_p: 0.0,
                dry: Default::default(),
                reasoning: Default::default(),
                ..cfg(0.0, 40, 1.0)
            };
            // Index 0 falls to 2.5 and index 1 wins.
            assert_eq!(sample_token(&logits, &[0], &penalised), 1);
        }

        #[test]
        fn top_p_keeps_the_minimum_normalised_nucleus() {
            let logits = [4.0, 3.0, 2.0, 1.0, -1.0];
            assert_eq!(candidate_ids(&logits, &cfg(1.0, 40, 0.8)), vec![0, 1]);
        }

        #[test]
        fn top_p_near_one_keeps_the_tail_and_small_top_p_does_not() {
            let logits = [4.0, 3.0, 2.0, 1.0, -1.0];
            assert_eq!(
                candidate_ids(&logits, &cfg(1.0, 40, 0.999)),
                vec![0, 1, 2, 3, 4]
            );
            assert_eq!(candidate_ids(&logits, &cfg(1.0, 40, 0.2)), vec![0]);
        }

        #[test]
        fn temperature_changes_the_nucleus_candidates() {
            let logits = [4.0, 3.0, 2.0, 1.0, 0.0];
            assert_eq!(candidate_ids(&logits, &cfg(0.5, 0, 0.9)), vec![0, 1]);
            assert_eq!(candidate_ids(&logits, &cfg(2.0, 0, 0.9)), vec![0, 1, 2, 3]);
        }

        #[test]
        fn non_positive_temperature_remains_deterministic_argmax() {
            let logits = [0.1, 9.0, 0.3, 2.0];
            for temperature in [0.0, -1.0] {
                assert_eq!(sample_token(&logits, &[], &cfg(temperature, 1, 0.01)), 1);
            }
        }

        #[test]
        fn non_positive_temperature_does_not_advance_the_lcg() {
            let _sampling = SAMPLER_TEST_LOCK.lock().unwrap();
            const SEED: u64 = 0x853c49e6748fea9b;
            let stochastic = cfg(1.0, 0, 1.0);
            let deterministic = cfg(0.0, 0, 1.0);
            let logits = [2.0, 0.0];

            *SAMPLER_RNG.lock().unwrap() = SEED;
            let expected = sample_token(&logits, &[], &stochastic);

            *SAMPLER_RNG.lock().unwrap() = SEED;
            assert_eq!(sample_token(&logits, &[], &deterministic), 0);
            assert_eq!(sample_token(&logits, &[], &stochastic), expected);
        }

        #[test]
        fn sampler_lcg_values_are_finite_unit_interval() {
            let mut state = 0x853c49e6748fea9b;
            for _ in 0..1024 {
                let value = next_sampler_rng_value(&mut state);
                assert!(value.is_finite(), "LCG devolveu {value:?}");
                assert!((0.0..1.0).contains(&value), "LCG devolveu {value:?}");
            }
        }

        #[test]
        fn top_k_then_top_p_composes_as_count_then_mass() {
            let logits = [10.0, 9.0, 8.0, 7.0, 6.0];
            assert_eq!(candidate_ids(&logits, &cfg(1.0, 3, 0.95)), vec![0, 1, 2]);
        }

        #[test]
        fn top_p_regression_preserves_multiple_candidates_instead_of_greedy() {
            let logits = [1.0, 8.0, 7.9, 7.8];
            assert_eq!(candidate_ids(&logits, &cfg(1.0, 40, 0.9)), vec![1, 2, 3]);
        }
    }

    /// The two boundaries that belong to the *caller* stay at zero here.
    ///
    /// Reading the logits out of llama.cpp and turning a token id into text are
    /// measured by `infer_server`, on either side of this call. If the sampler
    /// ever started reporting them, a cost would be counted twice.
    /// The capture target and buffer are process-global — the module doc
    /// explains why that is sound in production, where one engine exists per
    /// process. Tests run in parallel threads, so two engines would exist at
    /// once and whichever loaded last would decide what the other captures.
    /// This lock restores the one-engine-at-a-time property the design assumes.
    static ENGINE_LOCK: Mutex<()> = Mutex::new(());

    /// How many layers the tests offload. Real peers run with everything on the
    /// GPU, and the capture behaves differently there, so this has to be a knob.
    fn test_gpu_layers() -> i32 {
        std::env::var("POM_TEST_GPU_LAYERS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    }

    /// Loads the real Qwen3.8 language model and its separate projector through
    /// the same `LlmEngine` path used by a deployed node. The test is opt-in
    /// because the model and `mmproj-BF16.gguf` are large.
    ///
    /// ```text
    /// POM_QWEN4EXP_MODEL=/models/Qwen3.8-Flash-Next-BF16-00001-of-00008.gguf \\
    /// POM_QWEN4EXP_MMPROJ=/models/mmproj-BF16.gguf \\
    /// cargo test -p llama-engine qwen4exp_initializes_vision_from_separate_mmproj -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "requires the real Qwen3.8 GGUF and mmproj-BF16.gguf"]
    fn qwen4exp_initializes_vision_from_separate_mmproj() {
        let _guard = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let model_path = std::env::var("POM_QWEN4EXP_MODEL")
            .expect("POM_QWEN4EXP_MODEL deve apontar para o primeiro shard GGUF");
        let mmproj_path = std::env::var("POM_QWEN4EXP_MMPROJ")
            .expect("POM_QWEN4EXP_MMPROJ deve apontar para mmproj-BF16.gguf");
        let info = crate::gguf::read_info(Path::new(&model_path))
            .expect("o cabeçalho do modelo Qwen3.8 deve ser legível");
        assert_eq!(
            info.architecture.as_deref(),
            Some("qwen4exp"),
            "o GGUF deve usar a chave de arquitetura declarada pelo Hub"
        );

        let cfg = EngineConfig {
            n_ctx: 512,
            offload_kqv: false,
            cache_type_k: KvCacheType::F16,
            cache_type_v: KvCacheType::F16,
            fit_memory: false,
            fit_memory_margin_mb: crate::DEFAULT_FIT_MEMORY_MARGIN_MB,
            n_batch: 512,
            n_ubatch: crate::DEFAULT_N_UBATCH,
            n_gpu_layers: 0,
            n_threads: 2,
            max_sessions: 1,
            need_embeddings: false,
            mtp: false,
            mtp_cpu_fallback_enabled: true,
            mtp_placement: Default::default(),
            moe_experts_in_ram: false,
            moe_cache_mib: 0,
            rs_seq_window: 0,
            mmproj_path: Some(mmproj_path),
            mtp_draft_path: None,
        };
        let engine = LlmEngine::load_with_cache(&model_path, &cfg, &ModelCache::new(), false)
            .expect("o modelo qwen4exp deve carregar com o fork suportado");

        assert_eq!(
            engine.loaded.model_architecture().as_deref(),
            Some("qwen4exp")
        );
        assert!(
            engine.vision_enabled(),
            "o mmproj separado deve inicializar o contexto mtmd"
        );
    }

    fn mtp_attach_refuses_target_without_loaded_nextn(max_sessions: u32) {
        let _guard = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(path) = std::env::var("POM_MTP_MODEL") else {
            eprintln!("skipping: set POM_MTP_MODEL to a .gguf to run this");
            return;
        };
        let target_cfg = EngineConfig {
            n_ctx: 4096,
            offload_kqv: false,
            cache_type_k: KvCacheType::F16,
            cache_type_v: KvCacheType::F16,
            fit_memory: false,
            fit_memory_margin_mb: crate::DEFAULT_FIT_MEMORY_MARGIN_MB,
            n_batch: 4,
            n_ubatch: crate::DEFAULT_N_UBATCH,
            n_gpu_layers: 0,
            n_threads: 2,
            max_sessions,
            need_embeddings: false,
            mtp: false,
            mtp_cpu_fallback_enabled: true,
            mtp_placement: Default::default(),
            moe_experts_in_ram: false,
            moe_cache_mib: 0,
            rs_seq_window: 0,
            mmproj_path: None,
            mtp_draft_path: None,
        };
        let target_plan = empty_fit_plan_with_mtp_mode(&target_cfg, MtpFitMode::Disabled);
        let target_loaded = LoadedModel::load_with_mtp(&path, &target_plan, false)
            .expect("o alvo sem pesos nextn deve carregar");
        let target = LlmEngine::with_model(target_loaded.clone(), &target_cfg)
            .expect("o contexto principal deve carregar");

        let mtp_cfg = EngineConfig {
            mtp: true,
            ..target_cfg
        };
        drop(target);
        let engine = LlmEngine::with_model(target_loaded, &mtp_cfg)
            .expect("o alvo sem nextn deve continuar servindo sem abortar");
        assert!(!engine.mtp_ready());
        assert_eq!(engine.mtp_unavailable_reason(), Some(MTP_NO_NEXTN_REASON));

        let direct = crate::mtp::MtpDrafter::attach(
            &engine.loaded,
            engine.ctx,
            &mtp_cfg,
            engine.n_embd_out,
            engine.n_vocab,
            None,
            false,
        )
        .expect("a guarda interna deve recusar sem abortar");
        assert!(direct.is_none());
    }

    #[test]
    #[ignore = "requires a real Qwen3.5 MTP GGUF"]
    fn mtp_attach_on_target_without_nextn_is_refused_for_one_session() {
        mtp_attach_refuses_target_without_loaded_nextn(1);
    }

    #[test]
    #[ignore = "requires a real Qwen3.5 MTP GGUF"]
    fn mtp_attach_on_target_without_nextn_is_refused_for_two_sessions() {
        mtp_attach_refuses_target_without_loaded_nextn(2);
    }

    /// Sonda ao vivo do caminho de recuo: um modelo irmão CPU-only deve deixar
    /// o contexto MTP na RAM nos próprios buffers. Ela não mede desempenho e
    /// fica ignorada porque requer um GGUF com cabeça MTP e um backend
    /// compilado.
    #[test]
    #[ignore]
    fn mtp_cpu_companion_keeps_nextn_context_in_host_ram() {
        let _guard = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(path) = std::env::var("POM_MTP_MODEL") else {
            eprintln!("skipping: set POM_MTP_MODEL to a .gguf to run this");
            return;
        };
        let cfg = EngineConfig {
            n_ctx: 4096,
            offload_kqv: true,
            cache_type_k: KvCacheType::F16,
            cache_type_v: KvCacheType::F16,
            fit_memory: true,
            fit_memory_margin_mb: crate::DEFAULT_FIT_MEMORY_MARGIN_MB,
            n_batch: 512,
            n_ubatch: crate::DEFAULT_N_UBATCH,
            n_gpu_layers: DEFAULT_GPU_LAYERS,
            n_threads: 0,
            max_sessions: 1,
            need_embeddings: false,
            mtp: true,
            mtp_cpu_fallback_enabled: true,
            mtp_placement: Default::default(),
            moe_experts_in_ram: false,
            moe_cache_mib: 0,
            rs_seq_window: 0,
            mmproj_path: None,
            mtp_draft_path: None,
        };
        let mut target_cfg = cfg.clone();
        target_cfg.fit_memory = false;
        target_cfg.n_gpu_layers = test_gpu_layers();
        let target_plan = empty_fit_plan(&target_cfg);
        let loaded = LoadedModel::load_with_mtp(&path, &target_plan, false)
            .expect("modelo alvo sem pesos MTP carrega");
        let mtp_loaded = LoadedModel::load_mtp_cpu(&path, &target_cfg).expect("modelo MTP CPU");
        let engine = LlmEngine::from_loaded_with_mtp_offload(
            loaded,
            &target_cfg,
            target_plan.applied,
            target_cfg.n_gpu_layers,
            Some(mtp_loaded),
            false,
        )
        .expect("contexto MTP na RAM");

        assert!(engine.mtp_ready(), "o GGUF precisa trazer o bloco nextn");
        assert_eq!(engine.mtp_allocation(), Some("ram"));
    }

    /// `tokenize` sizes its buffer from a call with `n_tokens_max = 0`, whose
    /// documented answer is the NEGATIVE of the count. Reading that as an error
    /// made every prompt come back empty, and the head peer of the pipeline
    /// rejected chunk 0 with "não tem tokens" before the second peer was ever
    /// contacted. Only a real vocab exercises the contract, so the test points
    /// at a GGUF given by the environment and skips when there is none.
    ///
    /// The residual capture has to actually fire, and the tensor has to have one
    /// row per token. This is the regression test for the eval-callback
    /// contract: returning `false` in the compute phase does not mean "not my
    /// tensor", it breaks ggml's evaluation loop and abandons the rest of the
    /// graph. With that inverted, the forward pass stopped at the input
    /// embedding, `llama_decode` still returned success, and the peer answered
    /// "residual capture did not fire".
    ///
    /// The middle-peer path: a hidden state comes in, the residual must come
    /// out. This is the one the pipeline actually breaks on and the one the
    /// token test above cannot reach — `decode_hidden` builds a different batch
    /// (`batch.token = NULL`, data in `batch.embd`) and therefore a different
    /// graph. Set `POM_TEST_GPU_LAYERS=99` to run it with the weights
    /// offloaded, which is how every real peer runs it.
    /// A DeepSeek-V4.1 model run whole and as slices chained through the
    /// layer-boundary state (`decode_tokens`, `take_residual`, `decode_hidden`
    /// with the token ids, as a pipeline's stages do) must pick the same
    /// tokens. CPU only, on a real model:
    ///
    /// ```text
    /// POM_DSV41_WHOLE=<first part of the whole GGUF> \
    /// POM_DSV41_SLICES=<slice 0>,<slice 1>,... \
    /// cargo test -p llama-engine --lib deepseek41_slices_chain_like_the_whole_model -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a real DeepSeek-V4.1 GGUF and its slices"]
    fn deepseek41_slices_chain_like_the_whole_model() {
        let _guard = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let whole = std::env::var("POM_DSV41_WHOLE").expect("POM_DSV41_WHOLE");
        let slices: Vec<String> = std::env::var("POM_DSV41_SLICES")
            .expect("POM_DSV41_SLICES")
            .split(',')
            .map(str::to_owned)
            .collect();
        let steps: usize = std::env::var("POM_DSV41_STEPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6);
        let cfg = |need_embeddings| EngineConfig {
            n_ctx: 256,
            mmproj_path: None,
            mtp_draft_path: None,
            offload_kqv: false,
            cache_type_k: KvCacheType::F16,
            cache_type_v: KvCacheType::F16,
            fit_memory: false,
            fit_memory_margin_mb: crate::DEFAULT_FIT_MEMORY_MARGIN_MB,
            n_batch: 256,
            n_ubatch: crate::DEFAULT_N_UBATCH,
            n_gpu_layers: 0,
            n_threads: std::thread::available_parallelism()
                .map(|n| n.get() as i32)
                .unwrap_or(8),
            max_sessions: 1,
            need_embeddings,
            mtp: false,
            mtp_cpu_fallback_enabled: false,
            mtp_placement: Default::default(),
            moe_experts_in_ram: false,
            moe_cache_mib: 0,
            rs_seq_window: 0,
        };
        let argmax = |logits: &[f32]| {
            logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as i32)
                .unwrap()
        };
        let cosine = |a: &[f32], b: &[f32]| {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            dot / (na * nb)
        };

        let text = |engine: &LlmEngine, tokens: &[i32]| {
            String::from_utf8_lossy(
                &tokens
                    .iter()
                    .flat_map(|t| engine.token_to_piece_bytes(*t, false))
                    .collect::<Vec<_>>(),
            )
            .into_owned()
        };
        // the whole model: greedy tokens and the logits of each step (POM_DSV41_REF caches them in a file)
        let reference = std::env::var("POM_DSV41_REF").ok();
        let cached = reference
            .as_ref()
            .and_then(|path| std::fs::read(path).ok())
            .map(|bytes| {
                let words: Vec<u32> = bytes
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                let (n_prompt, n_steps, n_vocab) =
                    (words[0] as usize, words[1] as usize, words[2] as usize);
                let mut at = 3;
                let prompt: Vec<i32> = words[at..at + n_prompt].iter().map(|w| *w as i32).collect();
                at += n_prompt;
                let tokens: Vec<i32> = words[at..at + n_steps].iter().map(|w| *w as i32).collect();
                at += n_steps;
                let logits: Vec<Vec<f32>> = (0..n_steps)
                    .map(|s| {
                        words[at + s * n_vocab..at + (s + 1) * n_vocab]
                            .iter()
                            .map(|w| f32::from_bits(*w))
                            .collect()
                    })
                    .collect();
                (prompt, tokens, logits)
            });
        let (prompt, whole_tokens, whole_logits) = if let Some(cached) = cached {
            cached
        } else {
            let mut engine =
                LlmEngine::load_with_cache(&whole, &cfg(false), &ModelCache::new(), false)
                    .expect("whole model loads");
            let prompt = engine.tokenize("Oi, teste. Responda em uma frase:", true, true);
            let mut whole_tokens = Vec::new();
            let mut whole_logits = Vec::new();
            engine.decode_tokens(&prompt, 0, 0, false).unwrap();
            for step in 0..steps {
                let logits = engine.logits_ith(-1).to_vec();
                let next = argmax(&logits);
                whole_tokens.push(next);
                whole_logits.push(logits);
                let pos = (prompt.len() + step) as u32;
                engine.decode_tokens(&[next], pos, 0, false).unwrap();
            }
            eprintln!(
                "whole : {:?} {:?}",
                whole_tokens,
                text(&engine, &whole_tokens)
            );
            if let Some(path) = &reference {
                let mut words = vec![
                    prompt.len() as u32,
                    steps as u32,
                    whole_logits[0].len() as u32,
                ];
                words.extend(prompt.iter().map(|t| *t as u32));
                words.extend(whole_tokens.iter().map(|t| *t as u32));
                words.extend(whole_logits.iter().flatten().map(|v| v.to_bits()));
                std::fs::write(
                    path,
                    words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<u8>>(),
                )
                .unwrap();
            }
            (prompt, whole_tokens, whole_logits)
        };
        let steps = steps.min(whole_tokens.len());

        // the slices: tokens into the first, the state on through the others with the token ids
        let mut stages: Vec<LlmEngine> = slices
            .iter()
            .enumerate()
            .map(|(i, path)| {
                LlmEngine::load_with_cache(
                    path,
                    &cfg(i + 1 < slices.len()),
                    &ModelCache::new(),
                    false,
                )
                .expect("slice loads")
            })
            .collect();
        let run = |stages: &mut Vec<LlmEngine>, tokens: &[i32], pos: u32| -> Vec<f32> {
            let n = tokens.len();
            stages[0].decode_tokens(tokens, pos, 0, true).unwrap();
            let mut state = stages[0].take_residual(n).unwrap();
            for k in 1..stages.len() {
                let last = k + 1 == stages.len();
                eprintln!("  stage {k} gets state shape {:?}", state.shape);
                stages[k]
                    .decode_hidden(&state.values, &state.shape, n, pos, 0, !last, Some(tokens))
                    .unwrap();
                if !last {
                    state = stages[k].take_residual(n).unwrap();
                }
            }
            stages.last().unwrap().logits_ith(-1).to_vec()
        };
        let mut chain_tokens = Vec::new();
        let mut logits = run(&mut stages, &prompt, 0);
        for step in 0..steps {
            let next = argmax(&logits);
            eprintln!(
                "step {step}: whole {} chain {next} cosine {:.6}",
                whole_tokens[step],
                cosine(&logits, &whole_logits[step])
            );
            chain_tokens.push(next);
            // follow the whole model's tokens so every step compares the same context
            logits = run(
                &mut stages,
                &[whole_tokens[step]],
                (prompt.len() + step) as u32,
            );
        }
        eprintln!(
            "chain : {:?} {:?}",
            chain_tokens,
            text(stages.last().unwrap(), &chain_tokens)
        );
        assert_eq!(chain_tokens, whole_tokens);
    }

    /// The state a DeepSeek-V4.1 slice hands on matches the whole model's hc
    /// streams at the same layer (`l_out-<last layer>`).
    ///
    /// ```text
    /// POM_DSV41_WHOLE=<whole GGUF> POM_DSV41_SLICE=<slice from layer 0> \
    /// cargo test -p llama-engine --lib deepseek41_first_slice_state_matches_the_whole_model -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "needs a real DeepSeek-V4.1 GGUF and a slice"]
    fn deepseek41_first_slice_state_matches_the_whole_model() {
        let _guard = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let whole = std::env::var("POM_DSV41_WHOLE").expect("POM_DSV41_WHOLE");
        let slice = std::env::var("POM_DSV41_SLICE").expect("POM_DSV41_SLICE");
        let cfg = EngineConfig {
            n_ctx: 256,
            mmproj_path: None,
            mtp_draft_path: None,
            offload_kqv: false,
            cache_type_k: KvCacheType::F16,
            cache_type_v: KvCacheType::F16,
            fit_memory: false,
            fit_memory_margin_mb: crate::DEFAULT_FIT_MEMORY_MARGIN_MB,
            n_batch: 256,
            n_ubatch: crate::DEFAULT_N_UBATCH,
            n_gpu_layers: 0,
            n_threads: std::thread::available_parallelism()
                .map(|n| n.get() as i32)
                .unwrap_or(8),
            max_sessions: 1,
            need_embeddings: true,
            mtp: false,
            mtp_cpu_fallback_enabled: false,
            mtp_placement: Default::default(),
            moe_experts_in_ram: false,
            moe_cache_mib: 0,
            rs_seq_window: 0,
        };
        let mut part = LlmEngine::load_with_cache(&slice, &cfg, &ModelCache::new(), false)
            .expect("slice loads");
        let last = part.n_layer() - 1;
        let prompt = part.tokenize("Oi, teste. Responda em uma frase:", true, true);
        let n = prompt.len();
        part.decode_tokens(&prompt, 0, 0, true).unwrap();
        let state = part.take_residual(n).unwrap();
        drop(part);

        let mut engine = LlmEngine::load_with_cache(&whole, &cfg, &ModelCache::new(), false)
            .expect("whole loads");
        engine.set_capture_layer(Some(last)).unwrap();
        engine.decode_tokens(&prompt, 0, 0, true).unwrap();
        let reference = engine.take_residual(n).unwrap();
        eprintln!(
            "slice state {:?}, whole l_out-{last} {:?}",
            state.shape, reference.shape
        );

        let n_embd = engine.n_embd;
        let streams = reference.values.len() / (n * n_embd);
        let width = state.values.len() / n;
        let mut worst = 0f64;
        for t in 0..n {
            let a = &state.values[t * width..t * width + streams * n_embd];
            let b = &reference.values[t * streams * n_embd..(t + 1) * streams * n_embd];
            let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let cos = dot / (na * nb);
            eprintln!("token {t}: cosine {cos:.6} |slice| {na:.3} |whole| {nb:.3}");
            worst = worst.max(1.0 - cos);
        }
        assert!(
            worst < 1e-3,
            "the slice's hc streams diverge from the whole model's"
        );
    }

    #[test]
    fn residual_capture_fires_for_an_injected_hidden_state() {
        let _guard = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(path) = std::env::var("POM_TEST_GGUF") else {
            eprintln!("skipping: set POM_TEST_GGUF to a .gguf to run this");
            return;
        };
        let cfg = EngineConfig {
            n_ctx: 512,
            mmproj_path: None,
            mtp_draft_path: None,
            offload_kqv: true,
            cache_type_k: KvCacheType::F16,
            cache_type_v: KvCacheType::F16,
            fit_memory: false,
            fit_memory_margin_mb: crate::DEFAULT_FIT_MEMORY_MARGIN_MB,
            n_batch: 512,
            n_ubatch: crate::DEFAULT_N_UBATCH,
            n_gpu_layers: test_gpu_layers(),
            n_threads: 2,
            max_sessions: 1,
            need_embeddings: true,
            mtp: false,
            mtp_cpu_fallback_enabled: true,
            mtp_placement: Default::default(),
            moe_experts_in_ram: false,
            moe_cache_mib: 0,
            rs_seq_window: 0,
        };
        let mut engine = LlmEngine::load_with_cache(&path, &cfg, &ModelCache::new(), false)
            .expect("model loads");

        let n_tokens = 4;
        // Values in the range a real residual carries; zeros would let a graph
        // that never ran look identical to one that did.
        let hidden: Vec<f32> = (0..n_tokens * engine.n_embd)
            .map(|i| ((i % 97) as f32 - 48.0) / 48.0)
            .collect();
        engine
            .decode_hidden(
                &hidden,
                &vec![n_tokens, engine.n_embd],
                n_tokens,
                0,
                0,
                true,
                None,
            )
            .expect("decode of an injected hidden state succeeds");

        let residual = engine
            .take_residual(n_tokens)
            .expect("the capture callback must fire on the hidden-state path too");
        assert_eq!(residual.values.len(), n_tokens * engine.n_embd);
        assert!(
            residual.values.iter().any(|v| *v != 0.0 && v.is_finite()),
            "residual is all zeros or non-finite — the graph did not run"
        );
    }

    #[test]
    fn residual_capture_fires_with_one_row_per_token() {
        let _guard = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(path) = std::env::var("POM_TEST_GGUF") else {
            eprintln!("skipping: set POM_TEST_GGUF to a .gguf to run this");
            return;
        };
        let cfg = EngineConfig {
            n_ctx: 512,
            offload_kqv: true,
            cache_type_k: KvCacheType::F16,
            cache_type_v: KvCacheType::F16,
            fit_memory: false,
            fit_memory_margin_mb: crate::DEFAULT_FIT_MEMORY_MARGIN_MB,
            n_batch: 512,
            n_ubatch: crate::DEFAULT_N_UBATCH,
            n_gpu_layers: test_gpu_layers(),
            n_threads: 2,
            max_sessions: 1,
            need_embeddings: true,
            mtp: false,
            mtp_cpu_fallback_enabled: true,
            mtp_placement: Default::default(),
            moe_experts_in_ram: false,
            moe_cache_mib: 0,
            rs_seq_window: 0,
            mmproj_path: None,
            mtp_draft_path: None,
        };
        let mut engine = LlmEngine::load_with_cache(&path, &cfg, &ModelCache::new(), false)
            .expect("model loads");
        let toks = engine.tokenize("Explique inferencia distribuida em uma frase.", true, true);
        assert!(!toks.is_empty(), "prompt must tokenize");
        engine
            .decode_tokens(&toks, 0, 0, true)
            .expect("decode succeeds");

        let residual = engine
            .take_residual(toks.len())
            .expect("the capture callback must have fired");
        assert_eq!(residual.values.len(), toks.len() * engine.n_embd);
        // An abandoned graph leaves the buffer zeroed; a real residual does not.
        assert!(
            residual.values.iter().any(|v| *v != 0.0 && v.is_finite()),
            "residual is all zeros or non-finite — the graph did not run"
        );
    }

    /// Round-trip de `--slot-save-path`: prefill numa sequência, grava em disco,
    /// apaga a sequência, relê o arquivo para a mesma sequência e confere que o
    /// cabeçalho de tokens volta intacto e que dá para continuar decodificando.
    ///
    /// ```text
    /// POM_TEST_GGUF=<qualquer>.gguf cargo test -p llama-engine \
    ///   state_seq_save_and_load_round_trip -- --nocapture
    /// ```
    #[test]
    fn state_seq_save_and_load_round_trip() {
        let _guard = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(path) = std::env::var("POM_TEST_GGUF") else {
            eprintln!("skipping: set POM_TEST_GGUF to a .gguf to run this");
            return;
        };
        let cfg = EngineConfig {
            n_ctx: 512,
            offload_kqv: true,
            cache_type_k: KvCacheType::F16,
            cache_type_v: KvCacheType::F16,
            fit_memory: false,
            fit_memory_margin_mb: crate::DEFAULT_FIT_MEMORY_MARGIN_MB,
            n_batch: 512,
            n_ubatch: crate::DEFAULT_N_UBATCH,
            n_gpu_layers: test_gpu_layers(),
            n_threads: 2,
            max_sessions: 1,
            need_embeddings: false,
            mtp: false,
            mtp_cpu_fallback_enabled: true,
            mtp_placement: Default::default(),
            moe_experts_in_ram: false,
            moe_cache_mib: 0,
            rs_seq_window: 0,
            mmproj_path: None,
            mtp_draft_path: None,
        };
        let mut engine = LlmEngine::load_with_cache(&path, &cfg, &ModelCache::new(), false)
            .expect("model loads");
        let toks = engine.tokenize("A short note on saving KV cache to disk.", true, true);
        assert!(!toks.is_empty());
        engine
            .decode_tokens(&toks, 0, 0, true)
            .expect("prefill succeeds");

        let file = std::env::temp_dir().join(format!("pom-slot-test-{}.bin", std::process::id()));
        let _ = std::fs::remove_file(&file);
        let written = engine
            .state_seq_save_file(&file, 0, &toks)
            .expect("state_seq_save_file");
        assert!(written > 0, "save wrote no bytes");

        assert!(engine.kv_remove(0, 0, -1), "kv_remove clears the sequence");

        let (read, header) = engine
            .state_seq_load_file(&file, 0, toks.len() + 8)
            .expect("state_seq_load_file");
        assert!(read > 0, "load read no bytes");
        assert_eq!(header, toks, "token header must round-trip verbatim");

        // A sequência restaurada continua decodificável a partir da posição
        // salva: decodifica o próximo token sem "inconsistent sequence positions".
        engine
            .decode_tokens(&[toks[toks.len() - 1]], toks.len() as u32, 0, false)
            .expect("decode continues from the restored position");

        let _ = std::fs::remove_file(&file);
    }

    /// Mudança 6 do spec de 2026-08-12: `node_logits_access_ms` marcava ~4,9 ms
    /// por token no nó da LM head, com `backend_sync_ms` em zero quando o
    /// `llama_synchronize` de diagnóstico está desligado pela configuração. A
    /// hipótese é que esses milissegundos são a GPU terminando o trabalho,
    /// cobrados na etapa errada - o primeiro toque de CPU nos logits é quem
    /// espera.
    ///
    /// Este teste decide isso sem precisar de duas máquinas: decodifica um
    /// token na mesma fatia duas vezes, uma com `POM_PROFILE_SYNC`
    /// desligado e outra com ele ligado, e imprime para onde o tempo foi.
    ///
    /// ```text
    /// POM_TEST_GGUF=<fatia final>.gguf POM_TEST_GPU_LAYERS=99 \
    ///   cargo test -p node --release --features metal logits_access_versus -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore]
    fn logits_access_versus_backend_sync() {
        let _guard = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(path) = std::env::var("POM_TEST_GGUF") else {
            eprintln!("skipping: set POM_TEST_GGUF to a .gguf to run this");
            return;
        };
        let cfg = EngineConfig {
            n_ctx: 512,
            offload_kqv: true,
            cache_type_k: KvCacheType::F16,
            cache_type_v: KvCacheType::F16,
            fit_memory: false,
            fit_memory_margin_mb: crate::DEFAULT_FIT_MEMORY_MARGIN_MB,
            n_batch: 512,
            n_ubatch: crate::DEFAULT_N_UBATCH,
            n_gpu_layers: test_gpu_layers(),
            n_threads: 0,
            max_sessions: 1,
            // O nó da LM head não exporta residual — é o ponto da Mudança 5.
            need_embeddings: false,
            mtp: false,
            mtp_cpu_fallback_enabled: true,
            mtp_placement: Default::default(),
            moe_experts_in_ram: false,
            moe_cache_mib: 0,
            rs_seq_window: 0,
            mmproj_path: None,
            mtp_draft_path: None,
        };
        let mut engine = LlmEngine::load_with_cache(&path, &cfg, &ModelCache::new(), false)
            .expect("model loads");
        let tokens = engine.tokenize(
            "Explique o que e inferencia distribuida em uma frase.",
            true,
            true,
        );

        for sync in [false, true, false, true] {
            std::env::set_var("POM_PROFILE_SYNC", if sync { "1" } else { "0" });
            // Cache limpo entre as rodadas; o `false` aqui seria o cache que
            // não desloca, e este teste não depende disso.
            let _ = engine.kv_remove(0, 0, -1);
            let mut pos = 0u32;
            engine
                .decode_tokens(&tokens, pos, 0, false)
                .expect("prefill decodes");
            let _ = engine.logits_ith(tokens.len() as i32 - 1)[0];
            pos += tokens.len() as u32;

            let mut access_us = 0u64;
            let mut completion_us = 0u64;
            let mut backend_call_us = 0u64;
            let reps = 16;
            let wall_started = Instant::now();
            for _ in 0..reps {
                let t = engine.decode_tokens(&[1i32], pos, 0, false).unwrap();
                let started = Instant::now();
                let logits = engine.logits_ith(0);
                let _touch = logits[0] + logits[logits.len() - 1];
                access_us += started.elapsed().as_micros() as u64;
                completion_us += t.backend_sync_us;
                backend_call_us += t.backend_call_us;
                pos += 1;
            }
            let wall = wall_started.elapsed().as_secs_f64() * 1000.0 / reps as f64;
            println!(
                "PROFILE_SYNC={}: TOTAL {wall:.2} ms/token | enfileirar {:.2} ms | conclusão {:.2} ms | acesso aos logits {:.2} ms",
                sync as u8,
                backend_call_us as f64 / reps as f64 / 1000.0,
                completion_us as f64 / reps as f64 / 1000.0,
                access_us as f64 / reps as f64 / 1000.0,
            );
        }
        std::env::remove_var("POM_PROFILE_SYNC");
    }

    #[test]
    fn tokenize_returns_tokens_for_a_real_vocab() {
        let _guard = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(path) = std::env::var("POM_TEST_GGUF") else {
            eprintln!("skipping: set POM_TEST_GGUF to a .gguf to run this");
            return;
        };
        let cfg = EngineConfig {
            n_ctx: 512,
            offload_kqv: true,
            cache_type_k: KvCacheType::F16,
            cache_type_v: KvCacheType::F16,
            fit_memory: false,
            fit_memory_margin_mb: crate::DEFAULT_FIT_MEMORY_MARGIN_MB,
            n_batch: 512,
            n_ubatch: crate::DEFAULT_N_UBATCH,
            n_gpu_layers: 0,
            n_threads: 2,
            max_sessions: 1,
            need_embeddings: false,
            mtp: false,
            mtp_cpu_fallback_enabled: true,
            mtp_placement: Default::default(),
            moe_experts_in_ram: false,
            moe_cache_mib: 0,
            rs_seq_window: 0,
            mmproj_path: None,
            mtp_draft_path: None,
        };
        let engine = LlmEngine::load_with_cache(&path, &cfg, &ModelCache::new(), false)
            .expect("model loads");
        let tokens = engine.tokenize("oi", true, true);
        assert!(!tokens.is_empty(), "a two-character prompt must tokenize");
        assert!(
            engine
                .tokenize(
                    "Explique o que e inferencia distribuida em uma frase.",
                    true,
                    true
                )
                .len()
                > tokens.len(),
            "a longer prompt must produce more tokens"
        );
    }

    #[test]
    fn chat_template_wraps_the_user_prompt_and_opens_assistant_turn() {
        let _guard = ENGINE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(path) = std::env::var("POM_TEST_GGUF") else {
            eprintln!("skipping: set POM_TEST_GGUF to a .gguf to run this");
            return;
        };
        let cfg = EngineConfig {
            n_ctx: 512,
            offload_kqv: true,
            cache_type_k: KvCacheType::F16,
            cache_type_v: KvCacheType::F16,
            fit_memory: false,
            fit_memory_margin_mb: crate::DEFAULT_FIT_MEMORY_MARGIN_MB,
            n_batch: 512,
            n_ubatch: crate::DEFAULT_N_UBATCH,
            n_gpu_layers: 0,
            n_threads: 2,
            max_sessions: 1,
            need_embeddings: false,
            mtp: false,
            mtp_cpu_fallback_enabled: true,
            mtp_placement: Default::default(),
            moe_experts_in_ram: false,
            moe_cache_mib: 0,
            rs_seq_window: 0,
            mmproj_path: None,
            mtp_draft_path: None,
        };
        let engine = LlmEngine::load_with_cache(&path, &cfg, &ModelCache::new(), false)
            .expect("model loads");
        let formatted = engine
            .apply_chat_template(&[("user", "Oi, teste")], true, false)
            .expect("chat template must render");
        assert!(formatted.contains("Oi, teste"));
        assert!(!formatted.starts_with("Oi, teste"));
        assert!(formatted.to_ascii_lowercase().contains("assistant"));
        assert!(
            engine.tokenize(&formatted, false, true).len()
                > engine.tokenize("Oi, teste", true, true).len()
        );
    }

    #[test]
    fn gemma4_fallback_uses_the_model_turns_and_thinking_token() {
        let formatted = render_gemma4_template(
            GEMMA4_TEMPLATE_FIXTURE,
            "<bos>",
            &[("user", "Oi, teste")],
            true,
            true,
        )
        .unwrap();

        assert!(formatted.starts_with("<bos><|turn>user\n"));
        assert!(formatted.contains("<|turn>user\nOi, teste<turn|>\n"));
        assert!(formatted.ends_with("<|turn>model\n<|channel>thought\n<channel|>"));
        assert!(formatted.contains("<|channel>thought\n<channel|>"));
        assert!(!formatted.contains("<|im_start|>"));
        assert!(!formatted.contains("<|im_end|>"));
        assert!(!formatted.contains("<think>"));
    }

    #[test]
    fn gemma4_fallback_preserves_the_thinking_start_when_enabled() {
        let formatted = render_gemma4_template(
            GEMMA4_TEMPLATE_FIXTURE,
            "<bos>",
            &[("user", "Oi")],
            true,
            false,
        )
        .unwrap();

        assert!(formatted.contains("<|turn>system\n<|think|>\n<turn|>\n"));
        assert!(formatted.ends_with("<|turn>model\n"));
        assert!(!formatted.contains("<|channel>thought\n<channel|>"));
    }

    #[test]
    fn gemma4_renderer_honors_add_generation_prompt_from_the_gguf_template() {
        let formatted = render_gemma4_template(
            GEMMA4_TEMPLATE_FIXTURE,
            "<bos>",
            &[("user", "Oi")],
            false,
            true,
        )
        .unwrap();

        assert_eq!(formatted, "<bos><|turn>user\nOi<turn|>\n");
    }

    #[test]
    fn generation_stop_matches_a_complete_turn_marker() {
        assert!(token_sequence_ends_with(&[12, 106], &[106]));
        assert!(!token_sequence_ends_with(&[12, 105], &[106]));
    }
}

#[cfg(test)]
mod utf8_assembler_tests {
    use super::*;

    /// Um token pode terminar no meio de um codepoint (byte-fallback) e o
    /// pedaço seguinte completa a sequência. Converter cada chamada com
    /// `from_utf8_lossy` nascia U+FFFD no meio da palavra - este é o
    /// contrato de regressão: só caractere completo sai, e o rabinho fica.
    #[test]
    fn assembler_emits_only_complete_characters_and_keeps_the_trailing_bytes() {
        let mut asm = Utf8Assembler::new();
        // "ão" = c3 a3, quebrado entre dois tokens depois de "Jo".
        assert_eq!(asm.feed(b"Jo\xc3"), "Jo");
        assert_eq!(asm.feed(b"\xa3o "), "\u{e3}o ");
        // Emoji de 4 bytes (🧠 = f0 9f a7 a0) quebrado em três feeds.
        assert_eq!(asm.feed(&[0xf0]), "");
        assert_eq!(asm.feed(&[0x9f, 0xa7]), "");
        assert_eq!(asm.feed(&[0xa0, b'!']), "\u{1f9e0}!");
        assert_eq!(asm.flush(), "");
    }

    #[test]
    fn assembler_passes_whole_ascii_in_a_single_call() {
        let mut asm = Utf8Assembler::new();
        assert_eq!(
            asm.feed(b"Ol\xc3\xa1, a\xc3\xa7\xc3\xa3o, cora\xc3\xa7\xc3\xa3o"),
            "Ol\u{e1}, a\u{e7}\u{e3}o, cora\u{e7}\u{e3}o"
        );
    }

    /// Byte inválido de verdade (não é prefixo de nada): vira um U+FFFD na
    /// hora e o stream continua - mesma semântica do from_utf8_lossy global,
    /// sem esperar bytes que nunca vão completar sequência alguma.
    #[test]
    fn assembler_forwards_an_invalid_byte_as_one_fffd_and_goes_on() {
        let mut asm = Utf8Assembler::new();
        assert_eq!(asm.feed(b"a\xffb"), "a\u{fffd}b");
    }

    /// Fim do stream com sequência truncada: o flush drena o resto como um
    /// único replacement, e chamar de novo devolve vazio.
    #[test]
    fn flush_drains_the_truncated_trailing_bytes_once() {
        let mut asm = Utf8Assembler::new();
        assert_eq!(asm.feed(b"ok\xc3"), "ok");
        assert_eq!(asm.flush(), "\u{fffd}");
        assert_eq!(asm.flush(), "");
    }
}
