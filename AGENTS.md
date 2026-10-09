# AGENTS.md - pom-plugin-llama.cpp

Motor llama.cpp do POM. Código movido de `crates/llama-engine` do `pom-jair`;
o mapa de módulos e as notas de implementação estão em `crates/llama-engine/AGENTS.md`.

- Este crate é carregado **no processo** do POM (crate Rust, não plugin
  `cdylib`). Não introduza IPC no caminho quente da inferência.
- Não há `brand` aqui: o prefixo GGUF é `crate::GGUF_PREFIX`. Ao mudar, mude o
  `slug` do `brand.json` do POM também.
- O fork do llama.cpp não é modificado por este repositório.
- Push direto em `main`, conforme o AGENTS.md do workspace.
