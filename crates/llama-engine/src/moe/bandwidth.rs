//! Política de divisão CPU/GPU derivada de banda medida.
//!
//! Este módulo não consulta hardware e não escolhe uma constante de placa. O
//! ensaio injeta as quatro taxas e a política só transforma os números em uma
//! fração determinística dos experts que serão buscados na GPU.

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BandwidthProfile {
    /// GEMV de expert na CPU durante a disputa real com a transferência.
    pub cpu_overlap_bytes_per_sec: f64,
    /// Cópia host para device durante a disputa real com o GEMV.
    pub transfer_overlap_bytes_per_sec: f64,
    /// Fallback para perfis antigos sem ensaio simultâneo.
    pub cpu_bytes_per_sec: f64,
    pub transfer_bytes_per_sec: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandwidthError {
    MissingCpuMeasurement,
    MissingTransferMeasurement,
    InvalidMeasurement,
}

impl fmt::Display for BandwidthError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingCpuMeasurement => formatter.write_str("banda CPU sem medição"),
            Self::MissingTransferMeasurement => {
                formatter.write_str("banda de transferência sem medição")
            }
            Self::InvalidMeasurement => {
                formatter.write_str("perfil de banda não finito ou negativo")
            }
        }
    }
}

impl std::error::Error for BandwidthError {}

#[derive(Debug, Clone, PartialEq)]
pub struct MissSplit {
    pub gpu: Vec<u32>,
    pub cpu: Vec<u32>,
    pub gpu_fraction: f32,
}

impl BandwidthProfile {
    /// Retorna a parcela dos misses que deve atravessar a transferência.
    pub fn gpu_fraction(self) -> Result<f32, BandwidthError> {
        let overlap_cpu = valid(self.cpu_overlap_bytes_per_sec);
        let overlap_transfer = valid(self.transfer_overlap_bytes_per_sec);
        let standalone_cpu = valid(self.cpu_bytes_per_sec);
        let standalone_transfer = valid(self.transfer_bytes_per_sec);

        let (cpu, transfer) = if overlap_cpu && overlap_transfer {
            (
                self.cpu_overlap_bytes_per_sec,
                self.transfer_overlap_bytes_per_sec,
            )
        } else if standalone_cpu && standalone_transfer {
            (self.cpu_bytes_per_sec, self.transfer_bytes_per_sec)
        } else {
            if !overlap_cpu && !standalone_cpu {
                return Err(BandwidthError::MissingCpuMeasurement);
            }
            if !overlap_transfer && !standalone_transfer {
                return Err(BandwidthError::MissingTransferMeasurement);
            }
            return Err(BandwidthError::InvalidMeasurement);
        };

        let denominator = cpu + transfer;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(BandwidthError::InvalidMeasurement);
        }
        Ok((transfer / denominator).clamp(0.0, 1.0) as f32)
    }

    pub fn split_misses(self, misses: &[u32]) -> Result<MissSplit, BandwidthError> {
        let gpu_fraction = self.gpu_fraction()?;
        let gpu_len = (misses.len() as f32 * gpu_fraction).round() as usize;
        let gpu_len = gpu_len.min(misses.len());
        Ok(MissSplit {
            gpu: misses[..gpu_len].to_vec(),
            cpu: misses[gpu_len..].to_vec(),
            gpu_fraction,
        })
    }
}

fn valid(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

#[cfg(test)]
mod tests {
    use super::{BandwidthError, BandwidthProfile};

    fn profile() -> BandwidthProfile {
        BandwidthProfile {
            cpu_overlap_bytes_per_sec: 3.0,
            transfer_overlap_bytes_per_sec: 1.0,
            cpu_bytes_per_sec: 30.0,
            transfer_bytes_per_sec: 10.0,
        }
    }

    #[test]
    fn uses_the_overlapping_band_when_there_is_one() {
        assert!((profile().gpu_fraction().unwrap() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn splits_misses_deterministically() {
        let split = profile().split_misses(&[10, 11, 12, 13]).unwrap();
        assert_eq!(split.gpu, vec![10]);
        assert_eq!(split.cpu, vec![11, 12, 13]);
    }

    #[test]
    fn uses_the_old_profile_fallback() {
        let mut old = profile();
        old.cpu_overlap_bytes_per_sec = 0.0;
        old.transfer_overlap_bytes_per_sec = 0.0;
        assert!((old.gpu_fraction().unwrap() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn an_incomplete_profile_does_not_choose_overlap() {
        let mut incomplete = profile();
        incomplete.cpu_overlap_bytes_per_sec = 0.0;
        incomplete.transfer_overlap_bytes_per_sec = 0.0;
        incomplete.cpu_bytes_per_sec = 0.0;
        assert_eq!(
            incomplete.gpu_fraction(),
            Err(BandwidthError::MissingCpuMeasurement)
        );
    }
}
