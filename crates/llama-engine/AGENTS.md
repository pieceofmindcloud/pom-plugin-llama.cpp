# AGENTS.md - crates/llama-engine

## O que é

Todo o código Rust que fala com `llama-cpp-sys-2`, tirado de `crates/node` para
um crate próprio: editar backend/rede/roteamento de `node` não recompila nem
re-typecheck estes módulos (o wrapper `llm/` reúne 250 KB de código). `node` depende deste
crate e **re-exporta** os módulos em `main.rs`, então `crate::llm::…`,
`crate::wire::…`, `crate::gguf::…` etc. continuam resolvendo em `node` sem tocar
os ~40 arquivos que os usam.

O build nativo (o fork do `/llama.cpp`, `cmake`, as features
`cuda`/`metal`/`vulkan`) é o mesmo de antes, agora declarado aqui. Ver
`docs/subsystems/build-and-run.md`.

`build.rs` aceita `CCACHE` como comando wrapper para o `nvcc`; portanto
`CCACHE=sccache` e valido. `CUDAARCHS` seleciona todas as arquiteturas dos
kernels próprios (por exemplo, `86;120` vira dois alvos, com 120 normalizado
para `120a`). As pontes C/C++ compiladas por `cc` tambem herdam o
`RUSTC_WRAPPER` e o helper comum de `scripts/build-env.sh` cobre os dois casos.

## Módulos (`src/`)

- `llm/` - wrapper FFI do llama.cpp dividido por responsabilidade em captura,
  configuração, modelo, engine, batch, amostragem e testes. É a única
  superfície `unsafe` do produto. Alcança o estado da coletiva TP por
  `tp_probe` (see below). `llm/config.rs::init_backend` is the only way the
  product calls `llama_backend_init`: it first refuses a processor that lacks
  an instruction set the linked CPU backend was compiled for
  (`llama_cpp_sys_2::ggml_cpu_built_with`), which would otherwise be a SIGILL
  that takes the node down. In `llm/config.rs`, `llama_max_devices()` is only
  the capacity of vectors passed to C; `fit_device_count` counts physical
  devices. The MTP KV reservation counts the fitted unified context once and
  follows the native `tensor_split` per NextN layer. When llama.cpp leaves the
  split all-zero because no fit adjustment was needed, resolve its default from
  the free-memory snapshot and preserve that explicit split for the real load;
  otherwise the joint fit would always reject MTP on multi-GPU systems. The
  joint fit repeats until reservation and split agree, then falls back to RAM
  if they do not. See
  `docs/subsystems/inference.md`, the fit section.
  Os snapshots nativos usados por `/health` passam pelo
  `wrapper_common.cpp`; eles devem tratar qualquer nome/descrição C nulo antes
  de chegar a `std::string` ou JSON, pois essa falha pode virar
  `strlen(nullptr)` no runtime C do Windows. Antes de cada avaliação do grafo
  com callback MoE, `begin_moe_callback` sinaliza o cancelamento do warmup sem
  aguardar a thread: o caminho quente não fica atrás de uma admissão CUDA
  corrente. O `join` permanece somente na troca explícita de warmup, fora da
  inferência. `DecodeTimings::backend_call_us` mede a chamada de avaliação do
  llama.cpp; `backend_sync_us` mede separadamente a sincronização explícita. A
  primeira não é tempo puro de kernel, e a segunda pode esperar trabalho de
  dispositivo e transferências já enfileirados. `llm/sampling.rs` representa a
  penalidade como overrides esparsos, usa índices `u32` no seletor de `top_k`
  para vocabulários usuais e não materializa um vetor de IDs quando `top_k` não
  restringe a distribuição.
- `mtp.rs` (+ `llama_mtp_shim.cpp`) - predição multi-token pelo bloco `nextn`;
  a lista de arquiteturas MTP aceitas é gerada em build por `build/mtp_arch.rs`.
  O `build.rs` lê o mesmo checkout preparado pelo `llama-cpp-sys-2`, recebido
  pelo metadado Cargo `DEP_LLAMA_LLAMA_SOURCE`; há fallback para
  `LLAMA_SOURCE` e, por padrão, o caminho vendorizado
  `vendor/llama-cpp-sys-2/llama.cpp`, que é materializado como diretório real
  no Windows.
  Quando o modelo alvo usa MoE externo metadata-only, `LlmEngine::set_moe_runtime`
  instala o mesmo callback também no contexto MTP; o acompanhante CPU do recuo
  mantém o caminho nativo porque seus pesos de experts estão presentes.
  `EngineConfig.mtp_draft_path` (+ `LoadedModel::load_mtp_draft`) é uma segunda
  fonte de pesos `nextn`: um GGUF de rascunho separado ("shared MTP module",
  ex. `qwen4exp`/Qwen3.8-Flash-Next) carregado como `LoadedModel` independente e
  usado como o mesmo `companion` do recuo para RAM em `mtp_model_for_attach` -
  os dois nunca coexistem. O rascunho externo depende do alvo exportar
  `h_nextn` do próprio grafo (código por arquitetura no fork,
  `res->t_h_nextn`); `qwen4exp.cpp` ganhou isso em `pieceofmindcloud/llama.cpp`
  (`pom`), gated por
  `cparams.embeddings_nextn && !cparams.embeddings_nextn_masked` para não
  regredir o corte antecipado de linhas da última camada quando MTP não foi
  pedido. Não há checagem estática do lado Rust dessa capacidade - a lista
  gerada em `build/mtp_arch.rs` certifica só quem constrói o grafo
  `DECODER_MTP` (o lado do rascunho); um alvo sem `h_nextn` falha no primeiro
  `observe` e desliga o MTP daquela sessão com aviso, sem abortar. Ver o
  cabeçalho de `mtp.rs` para a mecânica completa e a convenção de nome
  (`catalog::find_mtp_draft_sibling`, prefixo `mtp-`). Para `qwen4exp`, o
  fork (`pieceofmindcloud/llama.cpp`, `pom`) porta o padrão de
  `qwen35.cpp`/`deepseek4.cpp` em `qwen4exp.cpp`: `load_arch_tensors` detecta
  um rascunho nextn-only (`mtp_only`, mesma sondagem de `blk.0.*`) e libera o
  tronco e os tensores de nível de modelo (`hc_head_*`) como
  `TENSOR_NOT_REQUIRED`; a cauda ganha os tensores `layer.nextn.*`
  (`eh_proj`/`enorm`/`hnorm`/`hc_head_*` próprios); e `build_arch_graph`
  despacha para uma `graph_mtp` nova quando `params.gtype ==
  LLM_GRAPH_TYPE_DECODER_MTP`. O `h_nextn` exportado pelo alvo também foi
  corrigido para o estado **largo** (`n_embd*hc`, pré-mixer) - exportar a
  versão já colapsada silenciava a cabeça de rascunho sem travar nada.
  Prova sintética completa (falha antes do porte, contexto MTP reservando
  depois) em `llm/model.rs::mtp_draft_tests` e fixture/gerador em
  `tests/fixtures/mtp/`.
  `LoadedModel::load_mtp_draft` sozinho sempre tenta VRAM
  (`MtpFitMode::Vram`, `DEFAULT_GPU_LAYERS`) sem medir memória livre - ao
  contrário do alvo, cujo `fit_plan` mede a VRAM restante e reduz camadas
  antes de carregar. `load_with_cache` é quem fecha essa lacuna, em duas
  camadas: (1) na **carga dos pesos**, olha `vram_plan.mtp_mode` (a decisão
  que o `fit_plan` do alvo já tomou) e pula direto para
  `LoadedModel::load_mtp_draft_cpu` (mesma carga, `n_gpu_layers=0`) quando
  já é `Ram`, com um recuo reativo para essa mesma função se
  `load_mtp_draft` falhar mesmo partindo de `Vram`; (2) na **criação do
  contexto**, escolhe `offload_kqv=false` de saída quando os pesos já
  ficaram na RAM (pelas razões acima), e mantém o recuo reativo por trás de
  `mtp_initial_context_error` para quando os pesos couberam na VRAM mas o
  buffer de compute do ubatch de pior caso não coube (`op_offload=true`
  deixa o agendador tentar GPU mesmo com pesos em CPU). As duas camadas
  falharam ao vivo no servidor CUDA de teste em rodadas separadas, 2026-09-18: primeiro
  `cudaMalloc failed` na criação do contexto (camada 2, corrigida antes),
  depois `cudaMalloc failed: out of memory` no próprio
  `llama_model_load_from_file` do rascunho, com `mtp_fit_mode=Ram` já
  decidido na linha anterior do log (camada 1). Não verificável neste
  devcontainer: não há GPU aqui, então nenhuma das duas falhas de `cudaMalloc`
  pode ser reproduzida diretamente; a camada 2 se apoia no teste unitário já
  existente para o recuo do embutido
  (`mtp_context_to_ram_turns_off_the_cache_offload` em `mtp.rs`), e a
  camada 1 no espelhamento exato de `load_mtp_cpu` (mesmo padrão, já usado
  em produção para o `nextn` embutido).
- `moe/` - executor MoE próprio (denso + esparso), com kernels CUDA em
  `moe/cuda_kernels.cu` compilados por `build.rs` sob `--features cuda`
  (no alvo MSVC: `-Xcompiler /MD` e `pom_moe_cuda.lib` via `lib.exe`; nos
  demais, `libpom_moe_cuda.a` via `ar`). O
  fallback CPU do router executa os experts selecionados em workers escopados
  e agrega os vetores na ordem original para manter o resultado determinístico;
  Q4_0 e Q5_K também usam a ponte CUDA heterogênea quando `cuda` está
  disponível, e um fallback CPU efetivo fica registrado na telemetria e no log
  de inicialização. O cache CUDA residente é particionado por ordinal (`CUDA0`,
  `CUDA1`, ...), com orçamento e allocator próprios por placa; as rotas são
  agrupadas por shard e executadas em paralelo. O caminho heterogêneo consulta
  primeiro as especificações do mmap e só copia bytes em um miss; seu workspace
  CUDA é um pipeline persistente compartilhado pelas camadas, indexado por
  device e geometria. O agregado do cache publica os nomes dos devices e os
  bytes residentes por shard para a telemetria. A ponte CUDA é código próprio
  deste crate; o fork `vendor/llama-cpp-sys-2/llama.cpp` só fornece o callback e
  os metadados de memória. O caminho uniforme IQ3_XXS (dtype 18) também usa o
  pipeline persistente e kernels cooperativos; IQ e K compartilham buffers e
  o mesmo agrupamento por CUDA<n>, sem `cudaDeviceSynchronize` no hot path.
  Os diagnósticos CUDA são automáticos sempre que um executor MoE CUDA existe;
  `POM_MOE_CUDA_PROFILE` não é necessário nem altera a execução. Quando há
  `<modelo>.cuda-profile.json` com um `BandwidthProfile`, o bloco heterogêneo
  também divide misses entre lanes CPU e GPU e as executa em paralelo, como o
  NVFP4. Sem sidecar, mantém a política exclusivamente CUDA; coexecução só
  usa taxas persistidas e medidas para não inventar uma velocidade de CPU.
  Antes do forward heterogêneo, as specs de toda a rota são construídas em
  uma travessia do inventário da camada e os misses são carregados em uma
  thread por shard;
  cada shard serializa sua admissão pelo mutex, enquanto CUDA0/CUDA1 podem
  fazer H2D em paralelo. O upload usa diretamente a fatia do mmap, sem copiar
  cada tensor para um `Vec` intermediário. Os leases de admissão permanecem
  ativos até o forward terminar. Quando há vários misses no mesmo shard, o
  allocator faz todas as alocações e cópias em um lote e sincroniza o stream
  uma vez; eviction usa `cudaFreeAsync` para não introduzir um wait global por
  tensor. A cópia ainda termina antes de o slice temporário sair de escopo.
  O warmup percorre experts por camada em ordem expert-major, espalhando as
  admissões pelas camadas antes de esgotar o orçamento de cada GPU; isso evita
  aquecer apenas o prefixo do modelo e concentrar misses na primeira
  requisição.
  A telemetria também publica, por camada, o working set observado de experts
  distintos por token e, por shard, os valores brutos de VRAM que formaram o
  orçamento (`baseline_free`, `current_free`, compute, reserva, limites e
  entradas residentes). Isso separa pressão real do orçamento de uma eviction
  indevida de lease ativo e permite auditar uma margem conservadora.
  Por decisão aprovada do operador, o contexto principal com
  `moe_external_executor` usa `n_ubatch=128` para devolver VRAM ao cache
  residente. Essa exceção também reduz o microlote de atenção e KV desse
  contexto, trade-off aceito para o caminho MoE. O caminho denso do llama.cpp
  e o contexto MTP permanecem no valor padrão de `FIT_MAX_UBATCH=512`; o MTP
  usa uma política de parâmetros própria para não herdar a exceção.
  Pares repetidos de expert e token são calculados uma vez e a agregação
  conserva o peso de cada rota original.
  No IQ3_XXS a entrada é quantizada para `block_q8_1` e o decoder nativo
  `vec_dot_iq3_xxs_q8_1` do fork usa `dp4a`. As 32 lanes cobrem quatro
  superblocos por warp, cada uma processando uma fatia q8 e avançando em
  passos de quatro, sem uma árvore de shared memory por linha. Gate e up são
  calculados na mesma warp e os buffers q8 de hidden e ativação ficam
  residentes na pipeline persistente.
  O caminho heterogêneo também usa os `vec_dot_*_q8_1` canônicos para Q4/K/IQ,
  quantiza hidden e ativação em buffers persistentes e executa uma warp por
  linha; isso é importante para GGUF como Mistral-Small, que combina dtypes
  18, 22 e 23 nos papéis gate/up/down.
  O dispatcher q8 heterogêneo escolhe helpers diretos por dtype, sem ponteiro
  de função de device no loop de blocos, permitindo ao nvcc fazer inline das
  primitivas canônicas.
  Cada pipeline heterogêneo mantém um staging host pinned para a saída CUDA:
  o callback do llama.cpp continua recebendo um buffer host comum, mas a
  transferência D2H não paga o staging pageable em cada chamada.
  Com o perfil CUDA, `cuda_device_calls` separa chamadas, experts e tempos por
  ordinal para distinguir bytes residentes de uso efetivo da placa.
  A cobertura CUDA por dtype e os números do servidor CUDA de teste ficam em
  `docs/subsystems/freetoken-moe.md`.
- `collective.rs` - `reduce_sum` + `ReduceBarrier` do tensor parallelism
  (caminho C). O transporte HTTP e a coordenação com o primary ficam em
  `crates/node/src/inference/tp.rs`, que **não** migrou.
- `tp_probe.rs` - ponte fina: `llm.rs` consulta o contador de reduces e o
  motivo de falha da coletiva de dentro do decode; `node` instala as funções
  reais (`tp::install_probe`) no arranque. Sem instalação, `is_active()` é
  `false` (caminho não-TP).
- `wire.rs` - formato binário do estado oculto entre peers. f16 é o padrão;
  f32 está disponível na implantação como formato sem perda de precisão, e
  f8/i4g64/i4g128 são alternativas com perda. I4 usa valores assinados de 4
  bits e uma escala f16 por grupo; o decodificador sempre entrega f32 ao motor.
  `encode_profiled`/`decode_profiled` e as variantes batch medem separadamente
  a conversão f32/f16; esses nanos são subconjuntos do tempo total de
  encode/decode. A identidade de sessão do `BatchItem` é autoritativa ao
  decodificar, para que um relay não reative uma sessão sintética a partir de
  metadata antiga. O caminho quente não copia o tensor à toa:
  `encode_batch_items` escreve direto no buffer que vai para o fio,
  `split_frame` empresta o payload em vez de o copiar, e
  `peek_batch_header`/`validate_batch_bytes` leem só o cabeçalho para quem
  precisa da rota, do `batch_id` ou de uma validação. Detalhe e números em
  `docs/subsystems/inference.md` §2.
- `mtp_draft.rs` - `ready_draft_path(draft, target)`: a shared MTP draft
  without `token_embd.weight` gets a completed copy in `.mtp-drafts/` next to
  the target, with `token_embd`/`output` appended from the target GGUF; a
  complete draft comes back unchanged. Called by `LlmEngine::load_with_cache`
  before the draft load. See `docs/subsystems/inference.md` (shared drafts).
- `gguf.rs` + `multipart.rs` - leitura de cabeçalhos GGUF, incluindo nomes de
  identidade (`general.name`, `general.basename` e `general.base_model.*`),
  `embedding_length` textual e `clip.vision.projection_dim` para associar
  projetores, além de modelos multipart. `read_layout` valida arrays por camada
  usando a tabela unificada de todas as partes físicas. Para `qwen4exp`, mede
  `per_layer_token_embd.weight` apenas nas faixas que contêm uma camada PLE;
  os três arrays de estrutura (`layer_multipliers`, `head_offsets` e
  `head_vocab_sizes`) permanecem completos, enquanto `ple.layers` é filtrado
  e rebaseado pelo fatiador. `kv_budget.rs` - orçamento de bytes do cache KV
  por geometria.
- `kv_cache.rs` - `KvCacheType` (era `node/src/utils/config.rs`; `Config` de
  `node` só o referencia como campo).
- `sysmon.rs` - monitor de CPU/RAM/disco/GPU. A raiz do projeto que ele mede é
  injetada por `node` (`sysmon::set_disk_root`). Cada família de GPU tem a
  própria leitura - `nvidia-smi`, sysfs do `amdgpu` + `/proc/<pid>/fdinfo`, e
  `ioreg -c AGXAccelerator` no Apple Silicon -, e o que uma plataforma não
  informa fica `None` em vez de virar zero. NVIDIA e AMD também publicam
  `power_watts` quando o driver oferece a leitura no mesmo caminho da GPU;
  Apple continua explicitamente sem esse dado. Montagens de infraestrutura do
  macOS e equivalentes (`/boot/efi`, `/boot`, `/efi`) são retiradas de `disks`,
  enquanto volumes de dados permanecem. `disk` é o volume do projeto;
  `disks` é toda partição montada relevante. Detalhe por plataforma em
  `docs/subsystems/operations.md` ("Monitor de carga"). Todo utilitário
  externo sai de `host_command`, que no Windows cria o processo com
  `CREATE_NO_WINDOW`: o `pom.exe` é GUI e, sem a flag, cada `nvidia-smi` da
  amostragem abria uma janela de console.
- `lib.rs` - re-exporta `KvCacheType`, `DEFAULT_FIT_MEMORY_MARGIN_MB` e
  `llama_cpp_sys_2` (para `node` alcançar a FFI crua sem redeclarar a dep).

## O que ficou em `crates/node`

`inference/tp.rs` (a coletiva é rede), `inference/relay.rs` (`ByteTransport`
vive nele), `inference/executor.rs`, `models/splitter.rs` (usa `gguf` e, nos
testes, `llama_engine::llama_cpp_sys_2`), e todo o resto do pipeline
(`infer_server.rs`, `prefill.rs`, `direct.rs`, `runtime.rs`, ...).

O ciclo de vida nativo deve ser encerrado na ordem inversa da criação:
contextos MTP e Vision liberam callbacks antes de llama_free, e o callback
MoE do engine principal é limpo antes de qualquer userdata compartilhado cair.
Essa regra fica implementada nos Drop de LlmEngine e MtpDrafter; alterá-la
exige revisar a fronteira FFI e testar a destruição de gerações compatíveis.

### Visão

`llm::VisionContext` usa o `mtmd` já fornecido pelo llama.cpp para carregar o
GGUF `mmproj` separado e transformar imagens em tokens no prefill. O caminho é
opcional e é validado junto com `NodeConfig.mmproj_path`. Qwen3.8-Flash-Next
continua usando esse contrato sidecar: o fork reconhece a arquitetura GGUF
exata `qwen4exp`, e o teste ignorado
`qwen4exp_initializes_vision_from_separate_mmproj` valida o modelo real e o
`mmproj-BF16.gguf` quando `POM_QWEN4EXP_MODEL` e `POM_QWEN4EXP_MMPROJ` estão
configurados.

`decode_multimodal` devolve `MultimodalDecode`, que separa `n_tokens`
(células físicas escritas no KV) de `n_past` (posição lógica seguinte). Em
M-RoPE as duas divergem por desenho do upstream, e `mrope_positions` carrega a
coordenada `[t, y, x, z]` de cada célula para que o estado oculto atravesse os
hops sem reduzir as duas contagens a uma só - ver
`decode_hidden_with_positions` e `crates/node/src/inference/AGENTS.md`.
Ao criar um bitmap, `decode_multimodal` passa
`mtmd_helper_init_opt_default()` à API do upstream. As opções passaram a ser
explícitas, e esse default preserva o comportamento de imagens sem configurar
parâmetros de vídeo.
A captura de residual acumula microlotes pelo número de células físicas: o
`mtmd` pode entregar o trecho textual e o trecho expandido da imagem em
callbacks separados, enquanto `n_past` continua sendo apenas a posição lógica.
Na captura, `ready_wait_us` mede do callback de consulta do scheduler até o
callback de prontidão e inclui avaliação do grafo e sincronização do scheduler.
`copy_us` mede a chamada `ggml_backend_tensor_get` inteira; no backend CUDA,
ela pode incluir a cópia D2H e a sincronização da stream, portanto não isola
DMA nem espera de GPU. `postprocess_us` mede a conversão e o registro. O campo
legado `residual_capture_d2h_sync_us` soma prontidão e `tensor_get` e não é uma
medida de largura de banda D2H. A stream CUDA e a vida útil do tensor não são
expostas por essa API, então a captura não pode mover a cópia para uma operação
assíncrona com segurança sem mudar a fronteira nativa.

## Defeitos anotados

- DEFEITO ANOTADO: com `--features cuda`, `cargo test -p llama-engine` não
  compila. Quatro testes em `src/moe/cuda.rs` (por volta das linhas 2985, 3593,
  3921 e 4020) chamam `run_route_batch_resident` com 8 argumentos, e a função
  hoje recebe 9 (`device_indices` foi acrescentado). Sem CUDA a suíte compila.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
