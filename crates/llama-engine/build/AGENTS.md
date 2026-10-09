# build/ - auxiliares do build do node

Esta pasta contém código executado pelo `build.rs`, não pelo binário em
runtime.

- `mtp_arch.rs` lê, em tempo de compilação, a árvore somente leitura do
`llama.cpp`. A fonte vem do metadado `DEP_LLAMA_LLAMA_SOURCE`, publicado pelo
build script da dependência depois de preparar o checkout; há fallback para
`LLAMA_SOURCE` e depois para `vendor/llama-cpp-sys-2/llama.cpp`. Isso é
importante no Windows, onde o symlink versionado é substituído por um diretório
real nesse caminho. O extrator encontra verificações efetivas de
`LLM_GRAPH_TYPE_DECODER_MTP` em `src/models`, liga a classe ao
  `LLM_ARCH_*` pelo dispatch de `llama-model.cpp` e resolve o nome oficial
  em `LLM_ARCH_NAMES`, em `llama-arch.cpp`. O resultado é gravado em
  `OUT_DIR` e incluído por `inference/mtp.rs`.
- A extração falha o build quando o fork está ausente, quando nenhuma
  verificação efetiva é encontrada, quando ela não está no escopo inequívoco de
  `build_arch_graph`, quando o mapeamento é inconsistente ou quando a lista fica
  menor que treze entradas, um piso de segurança contra extração parcial. O fork
  nunca é alterado.

O acoplamento operacional dessa extração ao fork e ao piso de arquiteturas está
registrado em `TODO.md`, item 10.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
