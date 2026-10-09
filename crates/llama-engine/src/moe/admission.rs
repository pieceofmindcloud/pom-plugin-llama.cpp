//! Orçamento puro para o cache de experts.
//!
//! A política recebe leituras de memória feitas pelo processo chamador. Não
//! escolhe uma constante por placa e não consulta CUDA: isso deixa a aritmética
//! testável sem runtime nativo.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryAccount {
    /// Memória livre medida antes de reservar pesos e caches deste executor.
    pub baseline_free_bytes: u64,
    /// Buffer de computação que motivou a margem, para diagnóstico.
    pub compute_bytes: u64,
    /// Peso já residente, contado contra o baseline.
    pub weights_bytes: u64,
    /// Buffers fixos que não entram no LRU.
    pub fixed_bytes: u64,
    /// Margem adicional que deve continuar livre para crescimento de grafos,
    /// fragmentação e alocações transitórias.
    pub reserve_bytes: u64,
}

/// Snapshot dos dois limites usados para admitir bytes no cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheBudgetSnapshot {
    pub baseline_free_bytes: u64,
    pub compute_bytes: u64,
    pub weights_bytes: u64,
    pub fixed_bytes: u64,
    pub reserve_bytes: u64,
    pub current_free_bytes: u64,
    pub resident_bytes: u64,
    pub baseline_limit_bytes: Option<u64>,
    pub dynamic_limit_bytes: Option<u64>,
    pub effective_limit_bytes: Option<u64>,
}

impl CacheBudgetSnapshot {
    #[cfg(any(feature = "cuda", test))]
    pub fn summary(self) -> String {
        fn value(bytes: Option<u64>) -> String {
            bytes.map_or_else(|| "indisponível".to_owned(), |bytes| bytes.to_string())
        }

        format!(
            "baseline={} bytes, compute={} bytes, reserva={} bytes, livre atual={} bytes, residentes={} bytes, limite baseline={} bytes, limite dinâmico={} bytes, limite efetivo={} bytes",
            self.baseline_free_bytes,
            self.compute_bytes,
            self.reserve_bytes,
            self.current_free_bytes,
            self.resident_bytes,
            value(self.baseline_limit_bytes),
            value(self.dynamic_limit_bytes),
            value(self.effective_limit_bytes),
        )
    }
}

/// Margem adicional relativa ao buffer de computação medido no contexto
/// llama.cpp.
///
/// O buffer de computação já foi alocado quando esta política mede o baseline.
/// Portanto a reserva não deve contar o buffer inteiro novamente: conserva 25%
/// dele como margem adicional, mas limita essa margem a 512 MiB para que um
/// microlote e contexto grandes não consumam todo o orçamento do cache.
#[cfg(any(feature = "cuda", test))]
const COMPUTE_HEADROOM_NUMERATOR: u64 = 1;
#[cfg(any(feature = "cuda", test))]
const COMPUTE_HEADROOM_DENOMINATOR: u64 = 4;
#[cfg(any(feature = "cuda", test))]
const COMPUTE_HEADROOM_CAP_BYTES: u64 = 512 * 1024 * 1024;

/// Converte o tamanho observado do compute buffer em reserva de admissão.
#[cfg(any(feature = "cuda", test))]
pub fn device_cache_reserve_bytes(compute_bytes: u64) -> Option<u64> {
    if compute_bytes == 0 {
        return None;
    }
    compute_bytes
        .checked_mul(COMPUTE_HEADROOM_NUMERATOR)?
        .checked_add(COMPUTE_HEADROOM_DENOMINATOR - 1)?
        .checked_div(COMPUTE_HEADROOM_DENOMINATOR)
        .map(|headroom| headroom.min(COMPUTE_HEADROOM_CAP_BYTES))
}

/// Extrai os compute buffers residentes na memória CUDA do relatório do
/// contexto llama.cpp.
///
/// `CUDA_Host` é memória de host e não pode ser usada como reserva de VRAM.
/// Um relatório ausente, incompleto ou sem compute CUDA positivo retorna
/// `None`, para que o chamador recuse o executor residente em vez de assumir
/// zero.
pub fn cuda_compute_bytes_from_llama_memory(memory: &serde_json::Value) -> Option<u64> {
    cuda_compute_bytes_by_device(memory).and_then(|devices| {
        devices
            .into_iter()
            .try_fold(0u64, |total, (_, bytes)| total.checked_add(bytes))
    })
}

/// Extrai os compute buffers CUDA indexados pelo ordinal real da placa.
///
/// O caminho denso do llama.cpp pode espalhar o contexto por `CUDA0`, `CUDA1`
/// e assim por diante. O executor externo precisa manter essa mesma dimensão:
/// um orçamento agregado faria o LRU admitir bytes em uma única placa mesmo
/// quando o relatório já provou que há compute buffers em várias placas.
pub fn cuda_compute_bytes_by_device(memory: &serde_json::Value) -> Option<Vec<(u32, u64)>> {
    let buffers = memory.get("buffers")?.as_array()?;
    let mut totals = std::collections::BTreeMap::<u32, u64>::new();

    for buffer in buffers {
        let name = buffer.get("buffer")?.as_str()?;
        if !name.starts_with("CUDA") || name.ends_with("_Host") {
            continue;
        }
        let device = buffer.get("device")?.as_str()?;
        let Some(device_index) = device
            .strip_prefix("CUDA")
            .and_then(|index| index.parse::<u32>().ok())
        else {
            return None;
        };
        let compute = buffer.get("compute")?.as_u64()?;
        let total = totals.entry(device_index).or_default();
        *total = total.checked_add(compute)?;
    }

    let devices = totals
        .into_iter()
        .filter(|(_, bytes)| *bytes > 0)
        .collect::<Vec<_>>();
    (!devices.is_empty()).then_some(devices)
}

impl MemoryAccount {
    /// Cria o orçamento do cache de device a partir da medição do llama.cpp.
    ///
    /// Os pesos e buffers do contexto já estão refletidos na leitura de VRAM
    /// livre feita depois da criação do contexto. O compute buffer medido fica
    /// registrado para diagnóstico, com a margem adicional definida acima.
    #[cfg(any(feature = "cuda", test))]
    pub fn for_device_cache(baseline_free_bytes: u64, compute_bytes: u64) -> Option<Self> {
        Some(Self {
            baseline_free_bytes,
            compute_bytes,
            weights_bytes: 0,
            fixed_bytes: 0,
            reserve_bytes: device_cache_reserve_bytes(compute_bytes)?,
        })
    }

    pub fn budget_snapshot(
        &self,
        current_free_bytes: u64,
        resident_bytes: u64,
    ) -> CacheBudgetSnapshot {
        let fixed_without_reserve = self.weights_bytes.checked_add(self.fixed_bytes);
        let baseline_limit_bytes = fixed_without_reserve
            .and_then(|fixed| fixed.checked_add(self.reserve_bytes))
            .and_then(|fixed| self.baseline_free_bytes.checked_sub(fixed));
        let dynamic_limit_bytes = current_free_bytes
            .checked_add(resident_bytes)
            .and_then(|available| available.checked_sub(self.reserve_bytes));
        let effective_limit_bytes = baseline_limit_bytes
            .zip(dynamic_limit_bytes)
            .map(|(baseline, dynamic)| baseline.min(dynamic));

        CacheBudgetSnapshot {
            baseline_free_bytes: self.baseline_free_bytes,
            compute_bytes: self.compute_bytes,
            weights_bytes: self.weights_bytes,
            fixed_bytes: self.fixed_bytes,
            reserve_bytes: self.reserve_bytes,
            current_free_bytes,
            resident_bytes,
            baseline_limit_bytes,
            dynamic_limit_bytes,
            effective_limit_bytes,
        }
    }

    /// Limite total de bytes do cache neste instante.
    ///
    /// O primeiro termo é o orçamento estável do processo. O segundo usa uma
    /// nova leitura de memória livre e o cache já residente, protegendo contra
    /// outro processo que tenha consumido memória desde a inicialização.
    pub fn cache_limit_bytes(&self, current_free_bytes: u64, resident_bytes: u64) -> Option<u64> {
        self.budget_snapshot(current_free_bytes, resident_bytes)
            .effective_limit_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_by_the_measured_baseline_minus_weights_and_reserve() {
        let account = MemoryAccount {
            baseline_free_bytes: 1_000,
            compute_bytes: 0,
            weights_bytes: 200,
            fixed_bytes: 100,
            reserve_bytes: 100,
        };

        assert_eq!(account.cache_limit_bytes(900, 0), Some(600));
    }

    #[test]
    fn a_new_measurement_lowers_the_limit_without_evicting_the_existing_cache() {
        let account = MemoryAccount {
            baseline_free_bytes: 1_000,
            compute_bytes: 0,
            weights_bytes: 200,
            fixed_bytes: 100,
            reserve_bytes: 100,
        };

        // A leitura atual informa 250 bytes livres e há 300 bytes residentes.
        // Apenas 450 bytes podem continuar no cache sem consumir a reserva.
        assert_eq!(account.cache_limit_bytes(250, 300), Some(450));
    }

    #[test]
    fn refuses_arithmetic_without_a_budget() {
        let account = MemoryAccount {
            baseline_free_bytes: 100,
            compute_bytes: 0,
            weights_bytes: 80,
            fixed_bytes: 30,
            reserve_bytes: 20,
        };

        assert_eq!(account.cache_limit_bytes(1, 0), None);
    }

    #[test]
    fn refuses_overflow_in_the_budget_sum() {
        let account = MemoryAccount {
            baseline_free_bytes: u64::MAX,
            compute_bytes: 0,
            weights_bytes: u64::MAX,
            fixed_bytes: 1,
            reserve_bytes: 1,
        };

        assert_eq!(account.cache_limit_bytes(0, 0), None);
    }

    #[test]
    fn extracts_cuda_compute_from_the_llama_report_and_ignores_cuda_host() {
        let memory = serde_json::json!({
            "buffers": [
                {"buffer": "CUDA_Host", "device": "CUDA0", "compute": 200},
                {"buffer": "CUDA0", "device": "CUDA0", "compute": 1000},
                {"buffer": "CPU_Mapped", "device": "CPU", "compute": 3000}
            ]
        });

        assert_eq!(cuda_compute_bytes_from_llama_memory(&memory), Some(1000));
    }

    #[test]
    fn extracts_cuda_compute_split_per_device() {
        let memory = serde_json::json!({
            "buffers": [
                {"buffer": "CUDA0", "device": "CUDA0", "compute": 1000},
                {"buffer": "CUDA1", "device": "CUDA1", "compute": 2000},
                {"buffer": "CUDA_Host", "device": "CUDA0", "compute": 3000}
            ]
        });

        assert_eq!(
            cuda_compute_bytes_by_device(&memory),
            Some(vec![(0, 1000), (1, 2000)])
        );
    }

    #[test]
    fn a_report_without_cuda_compute_is_refused() {
        let memory = serde_json::json!({
            "buffers": [
                {"buffer": "CUDA_Host", "device": "CUDA0", "compute": 200},
                {"buffer": "CPU_Mapped", "device": "CPU", "compute": 3000}
            ]
        });

        assert_eq!(cuda_compute_bytes_from_llama_memory(&memory), None);
    }

    #[test]
    fn the_cache_reserve_is_derived_from_the_measured_compute_with_a_margin() {
        assert_eq!(device_cache_reserve_bytes(1_000), Some(250));
        assert_eq!(device_cache_reserve_bytes(1), Some(1));
        assert_eq!(device_cache_reserve_bytes(0), None);
    }

    #[test]
    fn a_large_compute_has_a_bounded_margin_and_does_not_zero_the_cache() {
        let gib = 1024 * 1024 * 1024;
        let compute = 8 * gib;
        let baseline = 10 * gib;
        let account = MemoryAccount::for_device_cache(baseline, compute).unwrap();

        assert_eq!(account.reserve_bytes, 512 * 1024 * 1024);
        assert_eq!(
            account.cache_limit_bytes(baseline, 0),
            Some(baseline - account.reserve_bytes)
        );
    }

    #[test]
    fn a_small_compute_keeps_a_proportional_margin() {
        assert_eq!(device_cache_reserve_bytes(1_000), Some(250));
    }

    #[test]
    fn the_device_cache_account_preserves_the_measured_reserve() {
        let account = MemoryAccount::for_device_cache(10_000, 1_000).unwrap();

        assert_eq!(account.reserve_bytes, 250);
        assert_eq!(account.cache_limit_bytes(10_000, 0), Some(9_750));
    }
}
