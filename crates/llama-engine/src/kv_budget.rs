use crate::KvCacheType;
use anyhow::{bail, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvLayerGeometry {
    pub kv_heads: u64,
    pub key_head_dim: u64,
    pub value_head_dim: u64,
}

pub fn kv_bytes_for_layers(
    context_tokens: u64,
    max_sessions: u64,
    layers: &[KvLayerGeometry],
    cache_type_k: KvCacheType,
    cache_type_v: KvCacheType,
) -> Result<u64> {
    if context_tokens == 0 || max_sessions == 0 || layers.is_empty() {
        bail!("geometria KV inválida: contexto, sessões ou camadas é zero");
    }
    let bytes_per_layer = layers.iter().try_fold(0u64, |total, geometry| {
        total
            .checked_add(kv_bytes_per_layer(geometry, cache_type_k, cache_type_v)?)
            .ok_or_else(|| anyhow::anyhow!("overflow ao somar camadas do KV"))
    })?;
    context_tokens
        .checked_mul(max_sessions)
        .and_then(|cells| cells.checked_mul(bytes_per_layer))
        .ok_or_else(|| anyhow::anyhow!("overflow ao calcular bytes totais do KV"))
}

fn kv_bytes_per_layer(
    geometry: &KvLayerGeometry,
    cache_type_k: KvCacheType,
    cache_type_v: KvCacheType,
) -> Result<u64> {
    if geometry.kv_heads == 0 || geometry.key_head_dim == 0 || geometry.value_head_dim == 0 {
        bail!("geometria KV inválida: cabeça ou dimensão é zero");
    }
    let key_elements = geometry
        .kv_heads
        .checked_mul(geometry.key_head_dim)
        .ok_or_else(|| anyhow::anyhow!("overflow ao calcular largura K do KV"))?;
    let value_elements = geometry
        .kv_heads
        .checked_mul(geometry.value_head_dim)
        .ok_or_else(|| anyhow::anyhow!("overflow ao calcular largura V do KV"))?;
    row_bytes(cache_type_k, key_elements)?
        .checked_add(row_bytes(cache_type_v, value_elements)?)
        .ok_or_else(|| anyhow::anyhow!("overflow ao somar linhas K e V do KV"))
}

/// Replica `ggml_row_size` para os tipos de cache aceitos pelo produto.
///
/// Os tipos quantizados só formam uma linha inteira quando a largura fecha o
/// bloco ggml. Recusar aqui evita que o planejador subestime a memória por
/// arredondamento implícito.
fn row_bytes(kind: KvCacheType, elements: u64) -> Result<u64> {
    let (block_bytes, block_elements) = match kind {
        KvCacheType::F32 => (4, 1),
        KvCacheType::F16 | KvCacheType::Bf16 => (2, 1),
        KvCacheType::Q8_0 => (34, 32),
        KvCacheType::Q4_0 | KvCacheType::Iq4Nl => (18, 32),
        KvCacheType::Q4_1 => (20, 32),
        KvCacheType::Q5_0 => (22, 32),
        KvCacheType::Q5_1 => (24, 32),
    };
    if !elements.is_multiple_of(block_elements) {
        bail!("largura KV de {elements} elementos não fecha o bloco de {block_elements} elementos");
    }
    (elements / block_elements)
        .checked_mul(block_bytes)
        .ok_or_else(|| anyhow::anyhow!("overflow ao calcular linha do KV"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::KvCacheType;

    #[test]
    fn computes_kv_with_asymmetric_types_and_sessions() {
        let geometry = KvLayerGeometry {
            kv_heads: 8,
            key_head_dim: 64,
            value_head_dim: 64,
        };
        assert_eq!(
            kv_bytes_for_layers(1024, 2, &[geometry; 4], KvCacheType::F16, KvCacheType::Q8_0,)
                .unwrap(),
            12_845_056
        );
    }

    #[test]
    fn refuses_a_dimension_that_does_not_close_a_quantized_block() {
        let geometry = KvLayerGeometry {
            kv_heads: 1,
            key_head_dim: 33,
            value_head_dim: 32,
        };

        assert!(
            kv_bytes_for_layers(1, 1, &[geometry], KvCacheType::Q8_0, KvCacheType::F16,)
                .unwrap_err()
                .to_string()
                .contains("bloco")
        );
    }

    #[test]
    fn refuses_overflow_instead_of_wrapping_the_planning() {
        let geometry = KvLayerGeometry {
            kv_heads: 8,
            key_head_dim: 128,
            value_head_dim: 128,
        };

        assert!(kv_bytes_for_layers(
            u64::MAX,
            2,
            &[geometry; 2],
            KvCacheType::F16,
            KvCacheType::F16,
        )
        .is_err());
    }

    #[test]
    fn refuses_zero_sessions_instead_of_inventing_one() {
        let geometry = KvLayerGeometry {
            kv_heads: 8,
            key_head_dim: 128,
            value_head_dim: 128,
        };

        assert!(
            kv_bytes_for_layers(1024, 0, &[geometry], KvCacheType::F16, KvCacheType::F16,)
                .unwrap_err()
                .to_string()
                .contains("sessões")
        );
    }
}
