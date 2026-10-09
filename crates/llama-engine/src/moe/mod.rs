//! Componentes do executor próprio de modelos MoE.
//!
//! O caminho denso não importa este módulo. O ponto de integração do motor
//! próprio é `Q4_0MoeBlock`, que combina cache, loader GGUF, router e executor
//! CPU ou CUDA para o primeiro formato suportado.

pub mod admission;
#[cfg(any(feature = "cuda", test))]
pub mod bandwidth;
pub mod block;
pub mod cache;
#[cfg(any(feature = "cuda", test))]
pub mod coexecution;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod format_heterogeneous;
pub mod format_iq_quants;
pub mod format_k_quants;
pub mod format_legacy_quants;
pub mod format_low_bits;
pub mod format_nvfp4;
pub mod format_plain;
pub mod format_q4_0;
pub mod router;
pub mod runtime;
