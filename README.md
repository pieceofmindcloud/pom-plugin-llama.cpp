# pom-plugin-llama.cpp

Motor de inferência llama.cpp do POM, em um repositório próprio. O processo
do POM carrega este crate e roda a inferência no mesmo processo e na mesma GPU
(sem IPC no caminho quente).

## Conteúdo

- `crates/llama-engine` - todo o código Rust que fala com `llama-cpp-sys-2`:
  wrapper FFI (`llm`), predição multi-token (`mtp`), executor MoE próprio
  (`moe`, com kernels CUDA em `cuda_kernels.cu`), tensor parallelism
  (`collective`), leitura de cabeçalhos GGUF (`gguf`, `multipart`), orçamento de
  KV (`kv_budget`) e monitor de sistema (`sysmon`).
- `vendor/llama-cpp-sys-2` - o binding `-sys`, vendorado, que compila o
  llama.cpp com CMake e gera o bindgen.

## Fonte do llama.cpp

O build nativo lê o fork do llama.cpp de um checkout na raiz deste repositório:

```sh
ln -s /caminho/para/llama.cpp llama.cpp   # fork pieceofmindcloud, branch pom
```

`vendor/llama-cpp-sys-2/llama.cpp` aponta para `../../llama.cpp`. Também é
possível apontar a variável `LLAMA_SOURCE` para qualquer checkout. O `llama.cpp`
não entra neste repositório (está no `.gitignore`).

## Features

- `cuda`, `vulkan`, `metal` - repassadas ao `llama-cpp-sys-2`.

## Verificação

```sh
cargo check -p llama-engine
cargo test -p llama-engine
```

## Relação com o POM

O POM (`pom-jair`) depende deste crate e re-exporta os módulos em
`crates/node/src/main.rs`. A constante `GGUF_PREFIX` (`pom`) precisa ser igual ao
`slug` de `brand.json` no POM; o POM tem um teste que confere isso.
