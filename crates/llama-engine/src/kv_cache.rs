//! Tipos de dados aceitos pelo cache KV do llama.cpp desta instalação.
//!
//! Vivia em `crates/node/src/utils/config.rs`; mudou para o `llama-engine`
//! porque o lado FFI (`llm.rs`, `kv_budget.rs`, `sysmon.rs`) precisa dele e o
//! `Config` de `node` só o referencia como campo. `crate::config` de `node`
//! re-exporta `KvCacheType` daqui.

use serde::{Deserialize, Serialize};

/// A lista acompanha `common/arg.cpp::kv_cache_types` do llama.cpp vendorizado
/// e evita deixar uma string inválida chegar ao FFI. `f16` é o padrão upstream.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheType {
    F32,
    F16,
    Bf16,
    Q8_0,
    Q4_0,
    Q4_1,
    Iq4Nl,
    Q5_0,
    Q5_1,
}

impl Default for KvCacheType {
    fn default() -> Self {
        Self::F16
    }
}

impl KvCacheType {
    pub const ALL: [Self; 9] = [
        Self::F32,
        Self::F16,
        Self::Bf16,
        Self::Q8_0,
        Self::Q4_0,
        Self::Q4_1,
        Self::Iq4Nl,
        Self::Q5_0,
        Self::Q5_1,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Iq4Nl => "iq4_nl",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
        }
    }
}
