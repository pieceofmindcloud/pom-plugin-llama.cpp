# llama-cpp-sys-2

Este diretório contém o wrapper Rust e o script de build do binding nativo.
O código do fork `llama.cpp` fica fora deste diretório e só pode ser alterado
com autorização explícita. O wrapper compila CUDA Graphs desligado por padrão
por segurança de VRAM; `POM_CUDA_GRAPHS=1` é uma opção de build para habilitá-los.

## Preparo automático do `llama.cpp`

`llama.cpp`, neste diretório, é um **symlink versionado** para a raiz do
repositório, e a pasta que ele aponta (`/llama.cpp/`) está no `.gitignore`. O
`build.rs` detecta um symlink quebrado ou a representação textual que o Git
usa no Windows e prepara automaticamente um clone raso da branch
`pom` de `pieceofmindcloud/llama.cpp` antes de caminhar pelos
arquivos nativos. Em Windows, ele remove o symlink ou arquivo-placeholder e
instala o clone como um diretório real em `vendor/llama-cpp-sys-2/llama.cpp`;
isso evita que o `WalkDir` dependa de um reparse point criado pelo checkout.
Em Unix, o clone continua sendo instalado no destino relativo do symlink. O
fork continua fora do histórico deste repositório e o script não altera
arquivos dentro de um checkout já existente.

Se o download automático falhar, o erro informa o comando completo para
preparar `vendor/llama-cpp-sys-2/llama.cpp` manualmente. Também é possível usar
um checkout local sem copiá-lo:

```bash
LLAMA_SOURCE=/caminho/para/llama.cpp cargo build -p node
```

O diretório informado precisa conter `ggml` e, quando explícito, tem
precedência sobre o symlink vendorizado `llama.cpp`. Isso também permite que o
Docker use a fonte da release mesmo se o named context local estiver vazio.
`POM_LLAMA_CPP_REPOSITORY` e `POM_LLAMA_CPP_REF` permitem apontar o preparo
automático para outro remoto ou branch durante testes e desenvolvimento; o
padrão é o fork público e a branch `pom`.

Depois de escolher o checkout, o `build.rs` normaliza um `LLAMA_SOURCE` relativo
para um caminho absoluto e publica o caminho em
`cargo:llama_source`. O Cargo entrega esse valor ao `llama-engine` como
`DEP_LLAMA_LLAMA_SOURCE`, para que a geração da lista MTP leia exatamente a
mesma árvore, inclusive quando o Windows materializa o symlink em
`vendor/llama-cpp-sys-2/llama.cpp`. Esse metadado só expõe o caminho; não copia
nem altera o fork.

Os wrappers C/C++ usam nomes de headers sem prefixo de diretório. O `build.rs`
adiciona explicitamente ao bindgen e às pontes C++ os diretórios `include/`,
`ggml/include/`, `src/`, `common/`, `tools/mtmd/` e `vendor/` da árvore
selecionada. Assim o mesmo contrato funciona no cache externo de CI, no
symlink versionado e em `LLAMA_SOURCE`, inclusive para o wrapper mtmd e os
headers internos.

No Windows, `canonicalize()` pode devolver o prefixo estendido `\\?\`. O
bindgen aceita esse caminho, mas CMake/MSBuild e algumas versões do MSVC não.
Antes de entregar a origem, os diretórios ou os arquivos a `cmake::Config` ou
`cc::Build`, o `build.rs` remove o prefixo; os caminhos chegam aos compiladores
como `D:\...` ou UNC convencionais, inclusive quando a origem veio de
`.ci-cache`.

O `build.rs` força o reconfigure do CMake a cada execução. Isso é intencional:
o fork coleta os arquivos de `src/models` com `file(GLOB ...)`, enquanto Docker
e Cargo podem reutilizar o diretório `target/` e o `CMakeCache.txt`. Sem o
reconfigure, um arquivo de modelo presente no checkout pode continuar fora de
`libllama.a` e só aparecer como símbolo C++ indefinido no link final.

O mesmo caminho é usado em Linux, macOS e Windows. No Windows o checkout pode
materializar o symlink como um arquivo de texto; o `build.rs` o substitui pelo
checkout no caminho que o Cargo lê. Ver `docs/subsystems/build-and-run.md`,
"Release pelo GitHub Actions".

O `build.rs` repassa `CMAKE_C_COMPILER_LAUNCHER`,
`CMAKE_CXX_COMPILER_LAUNCHER` e `CMAKE_CUDA_COMPILER_LAUNCHER` como entradas do
cache CMake e declara os tres valores para o Cargo reexecutar o script quando
o launcher muda. O helper `scripts/build-env.sh` usa `sccache` nesses pontos.
No Docker de desenvolvimento, `POM_LLAMA_CPP_PREBUILT_LIB_DIR` e uma excecao
explícita para Linux: valida e copia as quatro bibliotecas compartilhadas do
core, e só então pula o CMake. Bindgen, wrappers, mtmd e kernels CUDA próprios
do POM continuam compilados contra o source de `LLAMA_SOURCE`; sem essa
variavel o build local permanece no caminho CMake normal. Esse contrato não é
um `LLAMA_CPP_LIB_DIR` genérico e não é usado pelos releases do POM.

NCCL fica opt-in nos builds CUDA. `GGML_CUDA_NCCL=ON` ativa o caminho coletivo
no CMake e tambem faz o `build.rs` propagar `NCCL_LIBRARY` do cache CMake para o
link final do executavel Rust. Sem essa segunda etapa, os arquivos estaticos do
llama.cpp compilam chamadas `nccl*`, mas o link final falha com simbolos
indefinidos. O caminho desativado continua sem dependencia em `libnccl.so`.

Com a feature `vulkan` em macOS, o build tambem publica a busca e o link do
loader `libvulkan.dylib` a partir de `CMAKE_Vulkan_LIBRARY`, `Vulkan_LIBRARY` ou
`VULKAN_SDK`; isso é necessário porque o CMake produz os objetos Vulkan
estáticos e o binário Rust faz o link final.

## `wrapper_common.cpp` - o que é nosso dentro deste diretório

`wrapper_common.{h,cpp}` e `wrapper_utils.h` são código **deste** repositório,
não do fork: é a ponte `extern "C"` para o que o `llama.h` público não expõe. O
`build.rs` os compila com os includes do fork (linhas ~628-635) e o bindgen lê
as declarações do `.h`, então cada função nova aparece em `llama_cpp_sys_2::`
sem mais nenhum passo.

Após a migração upstream para `common_json`,
`llama_rs_json_schema_to_grammar` deve analisar o texto com
`common_json::parse`, não com `nlohmann::ordered_json`, porque a API interna
recebe a abstração própria. `llama_rs_fit_params` passa `nullptr` no novo
parâmetro de modelo extra de `common_fit_params`; isso representa corretamente
que o POM não usa um modelo adicional nessa sondagem.

Duas delas incluem cabeçalhos **internos** do fork (`src/llama-model.h`,
`src/llama-ext.h`), e é bom saber por quê antes de mexer:

| Função | O que responde |
|---|---|
| `llama_rs_model_placement_json` | onde cada camada ficou de verdade, lendo o buffer de cada tensor em `tensors_by_name` mais `dev_layer(il)` |
| `llama_rs_context_memory_json` | a memória do contexto por buffer (`llama_get_memory_breakdown`), cujo campo `context` é o cache KV e `host` distingue RAM de VRAM |
| `llama_rs_context_flash_attn` | o Flash Attention **resolvido**, de `get_cparams()`. O padrão do `llama.h` é `AUTO`, o llama.cpp sonda o backend ao criar o contexto e nada público devolve a decisão; `out_auto` diz se ela veio da sonda ou do pedido |
| `llama_rs_get_devices` | os handles elegíveis na mesma ordem padrão que o llama.cpp usaria |
| `llama_rs_get_device_memory` | o livre e o total atuais dos dispositivos escolhidos pelo posicionamento padrão, na mesma ordem do `common_fit_params` |
| `llama_rs_benchmark_device` | uma pontuação de um grafo GGML genérico, com cache por handle no processo; zero significa velocidade indeterminável |
| `llama_rs_backend_devices_json` | descricao e `device_id` PCI dos dispositivos registrados pelos backends Vulkan, Metal, CUDA e CPU; nomes e descricoes nulos viram texto vazio, `device_id` ausente continua `null` |

Os snapshots de `placement`, `context_memory` e `backend_devices` são
diagnósticos pós-carga e não podem derrubar o processo por metadado incompleto
de um backend. O wrapper converte nomes e descrições C nulos para rótulos locais
antes de criar `std::string` ou JSON; a fronteira é deliberadamente defensiva
porque `strlen(nullptr)` em Windows aparece como `ucrtbase!strlen`, sem panic
Rust. A implementação de `wrapper_common.cpp` mantém essa conversão em helpers
únicos e a auditoria de todos os `wrapper*.cpp/.h` deve ser repetida quando uma
nova API do backend passar a alimentar JSON.

`llama_rs_benchmark_device` usa somente a interface pública de backend do GGML
para executar uma multiplicação matriz-vetor pequena. O POM chama a medição no
máximo uma vez por dispositivo e só usa a ordem quando todos os resultados são
válidos e as diferenças superam o limiar de confiança. Backend sem memória,
inicialização, alocação ou execução compatível devolve zero e conserva a ordem
padrão. O probe não contém tabela de hardware nem nomes de fabricantes.

`llama_rs_fit_params` eleva temporariamente o limite de verbosidade do logger
comum para `TRACE` e o restaura ao sair, porque `common_fit_params` registra a
decisão nesses níveis e não é thread-safe. O fit MTP no POM não usa um wrapper
de sondagem seca: calcula o KV determinístico a partir da metadata GGUF e
deixa a construção do grafo para a tentativa real, que tem recuo recuperável
para a RAM. O `common/fit.cpp` do checkout local captura `free` e `total` antes
de criar seu contexto de medição genérico. Essa é uma divergência local do
fork e precisa acompanhar a branch `pom` usada no build. A
regressão da contabilidade sintética fica em `llama.cpp/tests/test-fit-memory.cpp`
e no alvo CTest `test-fit-memory`.

As funções que devolvem JSON retornam strings de `malloc` que **têm que** voltar por
`llama_rs_string_free`; do lado Rust quem faz isso é `json_from_wrapper` em
`crates/node/src/inference/llm.rs`, e é por isso que a liberação mora lá e não
em cada chamador.

O preço de usar cabeçalho interno é que uma atualização do fork pode mudar
`llama_model` sem aviso. O sintoma é erro de compilação alto e na hora, nunca
resultado errado em silêncio, e o conserto é reler o cabeçalho novo - a
alternativa, derivar a colocação de `n_gpu_layers`, mente sempre que um override
de buffer type entra. Ver `docs/subsystems/inference.md`, "Onde as camadas
ficaram".

## Estado recorrente no Kimi-K3

O checkout nativo também implementa a parte específica dos modelos híbridos.
Kimi-K3 precisa declarar suporte a rollback recorrente e materializar as
`1 + n_rs_seq` janelas de convolução nos grupos de snapshot antes do decode.
Sem isso, `llama_context` reduz `n_rs_seq` a zero e o estado KDA fica
incompatível com o rollback/replay usado pela inferência especulativa. A
regressão executável está em `llama.cpp/tests/test-hybrid-memory-batch.cpp`;
consulte `docs/subsystems/inference.md`, seção 5.3, para a invariável do
batch e do estado.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
