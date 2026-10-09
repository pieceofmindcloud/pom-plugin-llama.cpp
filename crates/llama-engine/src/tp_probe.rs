//! Ponte fina para o estado global do tensor parallelism.
//!
//! `inference/tp.rs` continua em `crates/node`: a coletiva fala com o primary
//! por HTTP/túnel libp2p e não caberia num crate de motor. Mas `llm.rs` precisa
//! consultar o contador de reduces e o motivo de falha da coletiva **de dentro
//! do loop de decode**. `node` instala estas funções no arranque
//! (`tp_probe::install`); sem instalação, `is_active()` é `false` e o resto é
//! no-op — que é exatamente o caminho não-TP.

use std::sync::OnceLock;

/// As funções de `inference/tp.rs` que o `llm` chama. Ponteiros de função
/// simples: o `node` passa os itens `fn` diretamente.
pub struct TpProbe {
    pub is_active: fn() -> bool,
    pub reduce_count: fn() -> u64,
    pub reduces_in_current_decode: fn() -> u64,
    pub take_failure: fn() -> Option<String>,
    pub set_expected_reduces: fn(u64),
}

static PROBE: OnceLock<TpProbe> = OnceLock::new();

/// Chamado uma vez pelo `node` no arranque. Uma segunda chamada é ignorada.
pub fn install(probe: TpProbe) {
    let _ = PROBE.set(probe);
}

pub fn is_active() -> bool {
    PROBE.get().is_some_and(|p| (p.is_active)())
}

pub fn reduce_count() -> u64 {
    PROBE.get().map_or(0, |p| (p.reduce_count)())
}

pub fn reduces_in_current_decode() -> u64 {
    PROBE.get().map_or(0, |p| (p.reduces_in_current_decode)())
}

pub fn take_failure() -> Option<String> {
    PROBE.get().and_then(|p| (p.take_failure)())
}

pub fn set_expected_reduces(expected: u64) {
    if let Some(p) = PROBE.get() {
        (p.set_expected_reduces)(expected);
    }
}
