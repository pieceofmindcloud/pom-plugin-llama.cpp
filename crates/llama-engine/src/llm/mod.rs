//! Low-level llama.cpp inference engine.
//!
//! Wraps `llama-cpp-sys-2` raw FFI to give the inference server a safe,
//! typed surface. All GPU/CPU calls are here; the rest of the codebase is pure
//! Rust.
//!
//! # Residual capture
//!
//! A non-last peer must hand the *residual stream* (the un-normalised `h` that
//! enters the next block's attention) to its successor. `llama_get_embeddings_ith`
//! returns the post-output_norm value instead, so we install a
//! `ggml_backend_sched_eval_callback` that fires when the last block's output
//! tensor (`l_out-<N>`, or an equivalent last-layer name used by the graph) is
//! computed
//! and copies it before the output head sees it. The graph keeps running after
//! the copy, so a required output head may cost compute without changing the
//! residual already exported to the next peer. If `n_embd_out` differs from
//! `n_embd`, generic llama.cpp embedding extraction stays disabled: that path
//! sizes its backend read from the metadata width, which may exceed the real
//! output tensor. The callback remains the only embedding-like output used by
//! the POM in that case.
//!
//! The callback identifies the token axis from the known token count of the
//! current physical microbatch and the tensor shape. It never uses
//! `n_embd_inp` to find that axis: dimensions before it are features and are
//! preserved in their physical order. Trailing singleton dimensions are
//! ignored, while trailing dimensions greater than one are rejected.
//!
//! The callback is an `extern "C"` fn with no closure environment, so the
//! capture state travels in llama.cpp's own `cb_eval_user_data`: each context
//! owns a `CaptureSlot` and the callback is handed its address. It used to be
//! two process-global `Mutex`es, safe only while a single decode ran at a time;
//! with two contexts decoding in parallel that became a race in which one
//! request would carry the other's residual - no error, just a wrong answer.

pub(crate) use crate::wire::TensorShape;
pub(crate) use crate::KvCacheType;
pub(crate) use anyhow::{bail, Context, Result};
pub(crate) use std::collections::{HashMap, HashSet};
pub(crate) use std::ffi::{c_void, CStr, CString};
pub(crate) use std::ops::Range;
pub(crate) use std::path::Path;
pub(crate) use std::ptr;
pub(crate) use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex, OnceLock,
};
pub(crate) use std::time::Instant;
pub(crate) use tracing::warn;

pub(crate) use crate::device_placement;
pub(crate) use llama_cpp_sys_2 as sys;

// ── Capture state, one per context ────────────────────────────────────────────

#[path = "batch.rs"]
mod batch;
/// Estado da captura de um contexto.
///
/// Já foi global de processo, e o comentário do topo deste arquivo dizia que
/// isso era seguro porque um decode rodava por vez. Deixou de ser verdade no
/// momento em que dois contextos passaram a decodificar em paralelo: as duas
/// requisições escreveriam no mesmo buffer e uma levaria o residual da outra -
/// sem erro nenhum, só resposta errada. Agora cada contexto tem o seu, e o
/// endereço viaja no `cb_eval_user_data` do próprio llama.cpp.

#[path = "capture.rs"]
mod capture;
#[path = "config.rs"]
mod config;
#[path = "engine.rs"]
mod engine;
#[path = "model.rs"]
mod model;
#[path = "sampling.rs"]
mod sampling;
#[path = "tests.rs"]
mod tests;

pub(super) use batch::*;
pub(super) use capture::*;
pub(super) use config::*;
pub(super) use model::*;
pub(super) use sampling::*;

pub use batch::*;
pub use capture::*;
pub use config::*;
pub use model::*;
pub use sampling::*;
