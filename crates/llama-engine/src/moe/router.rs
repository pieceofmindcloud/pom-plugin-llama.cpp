//! Roteamento top-k comum ao executor de experts.
//!
//! A seleção é determinística para empates e renormaliza somente os experts
//! selecionados. O executor do formato não conhece a política do router: ele
//! recebe um expert por vez e o bloco soma as saídas com estes pesos.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct ExpertRoute {
    pub ids: Vec<u32>,
    pub weights: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteError {
    EmptyLogits,
    InvalidTopK { top_k: usize, experts: usize },
    NonFiniteLogit { expert: usize },
    InvalidWeightSum,
    MismatchedRouteLengths { ids: usize, weights: usize },
    OutputWidth { expected: usize, actual: usize },
    WorkerPanic,
}

impl fmt::Display for RouteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLogits => write!(formatter, "router sem experts"),
            Self::InvalidTopK { top_k, experts } => {
                write!(formatter, "top_k={top_k} não cabe em {experts} experts")
            }
            Self::NonFiniteLogit { expert } => {
                write!(formatter, "logit não finito no expert {expert}")
            }
            Self::InvalidWeightSum => write!(formatter, "soma de pesos do router inválida"),
            Self::MismatchedRouteLengths { ids, weights } => {
                write!(formatter, "router entregou {ids} ids e {weights} pesos")
            }
            Self::OutputWidth { expected, actual } => {
                write!(
                    formatter,
                    "saída de expert tem {actual}, esperado {expected}"
                )
            }
            Self::WorkerPanic => {
                write!(formatter, "thread de execução do expert entrou em panic")
            }
        }
    }
}

/// Executa exatamente a rota já resolvida por outro grafo.
///
/// O hook do llama.cpp entrega ids e pesos depois de aplicar a sua política de
/// gating, top-k, grupos e normalização. Não se deve reconstruir logits aqui:
/// isso perderia diferenças de arquitetura e poderia selecionar experts
/// diferentes do grafo que foi substituído.
pub fn dispatch_route<E, F>(
    ids: &[u32],
    weights: &[f32],
    mut execute: F,
) -> Result<Vec<f32>, DispatchError<E>>
where
    F: FnMut(u32) -> Result<Vec<f32>, E>,
{
    if ids.is_empty() {
        return Err(DispatchError::Route(RouteError::InvalidWeightSum));
    }
    if ids.len() != weights.len() {
        return Err(DispatchError::Route(RouteError::MismatchedRouteLengths {
            ids: ids.len(),
            weights: weights.len(),
        }));
    }
    if weights.iter().any(|weight| !weight.is_finite()) {
        return Err(DispatchError::Route(RouteError::InvalidWeightSum));
    }

    let mut output = None::<Vec<f32>>;
    for (&expert, &weight) in ids.iter().zip(weights) {
        let values = execute(expert).map_err(DispatchError::Expert)?;
        let target = output.get_or_insert_with(|| vec![0.0; values.len()]);
        if target.len() != values.len() {
            return Err(DispatchError::Route(RouteError::OutputWidth {
                expected: target.len(),
                actual: values.len(),
            }));
        }
        for (slot, value) in target.iter_mut().zip(values) {
            *slot += weight * value;
        }
    }
    output.ok_or(DispatchError::Route(RouteError::InvalidWeightSum))
}

/// Executa em paralelo uma rota já resolvida e agrega na mesma ordem dos ids.
///
/// O callback precisa ser somente leitura e thread-safe. Os resultados são
/// recolhidos antes da soma para que a ordem de ponto flutuante permaneça
/// determinística e um panic de worker vire erro de inferência, não panic do
/// servidor.
pub fn dispatch_route_parallel<E, F>(
    ids: &[u32],
    weights: &[f32],
    execute: F,
) -> Result<Vec<f32>, DispatchError<E>>
where
    E: Send,
    F: Fn(u32) -> Result<Vec<f32>, E> + Sync,
{
    if ids.is_empty() {
        return Err(DispatchError::Route(RouteError::InvalidWeightSum));
    }
    if ids.len() != weights.len() {
        return Err(DispatchError::Route(RouteError::MismatchedRouteLengths {
            ids: ids.len(),
            weights: weights.len(),
        }));
    }
    if weights.iter().any(|weight| !weight.is_finite()) {
        return Err(DispatchError::Route(RouteError::InvalidWeightSum));
    }

    let values = std::thread::scope(|scope| {
        let execute = &execute;
        ids.iter()
            .map(|&expert| scope.spawn(move || execute(expert)))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|worker| match worker.join() {
                Ok(Ok(values)) => Ok(values),
                Ok(Err(error)) => Err(DispatchError::Expert(error)),
                Err(_) => Err(DispatchError::Route(RouteError::WorkerPanic)),
            })
            .collect::<Result<Vec<_>, _>>()
    })?;

    let mut output = None::<Vec<f32>>;
    for (weight, values) in weights.iter().copied().zip(values) {
        let target = output.get_or_insert_with(|| vec![0.0; values.len()]);
        if target.len() != values.len() {
            return Err(DispatchError::Route(RouteError::OutputWidth {
                expected: target.len(),
                actual: values.len(),
            }));
        }
        for (slot, value) in target.iter_mut().zip(values) {
            *slot += weight * value;
        }
    }
    output.ok_or(DispatchError::Route(RouteError::InvalidWeightSum))
}

impl std::error::Error for RouteError {}

#[derive(Debug)]
pub enum DispatchError<E> {
    Route(RouteError),
    Expert(E),
}

impl<E: fmt::Display> fmt::Display for DispatchError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Route(error) => error.fmt(formatter),
            Self::Expert(error) => write!(formatter, "executor de expert falhou: {error}"),
        }
    }
}

impl<E: fmt::Debug + fmt::Display> std::error::Error for DispatchError<E> {}

pub fn select_top_k(logits: &[f32], top_k: usize) -> Result<ExpertRoute, RouteError> {
    if logits.is_empty() {
        return Err(RouteError::EmptyLogits);
    }
    if top_k == 0 || top_k > logits.len() {
        return Err(RouteError::InvalidTopK {
            top_k,
            experts: logits.len(),
        });
    }
    for (expert, logit) in logits.iter().enumerate() {
        if !logit.is_finite() {
            return Err(RouteError::NonFiniteLogit { expert });
        }
    }

    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|left, right| {
        logits[*right]
            .partial_cmp(&logits[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.cmp(right))
    });
    order.truncate(top_k);

    let max_logit = order
        .iter()
        .map(|&expert| logits[expert])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weights: Vec<f32> = order
        .iter()
        .map(|&expert| (logits[expert] - max_logit).exp())
        .collect();
    let sum: f32 = weights.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(RouteError::InvalidWeightSum);
    }
    for weight in &mut weights {
        *weight /= sum;
    }
    Ok(ExpertRoute {
        ids: order.into_iter().map(|expert| expert as u32).collect(),
        weights,
    })
}

/// Roteia uma ativação por top-k e soma os outputs dos experts.
pub fn dispatch_top_k<E, F>(
    logits: &[f32],
    top_k: usize,
    mut execute: F,
) -> Result<Vec<f32>, DispatchError<E>>
where
    F: FnMut(u32) -> Result<Vec<f32>, E>,
{
    let route = select_top_k(logits, top_k).map_err(DispatchError::Route)?;
    let mut output = None::<Vec<f32>>;
    for (expert, weight) in route.ids.into_iter().zip(route.weights) {
        let values = execute(expert).map_err(DispatchError::Expert)?;
        let target = output.get_or_insert_with(|| vec![0.0; values.len()]);
        if target.len() != values.len() {
            return Err(DispatchError::Route(RouteError::OutputWidth {
                expected: target.len(),
                actual: values.len(),
            }));
        }
        for (slot, value) in target.iter_mut().zip(values) {
            *slot += weight * value;
        }
    }
    output.ok_or(DispatchError::Route(RouteError::InvalidWeightSum))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    #[test]
    fn top_k_is_deterministic_on_a_tie() {
        let route = select_top_k(&[2.0, 2.0, 1.0], 2).unwrap();
        assert_eq!(route.ids, vec![0, 1]);
        assert!((route.weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dispatch_sums_the_experts_with_normalized_weights() {
        let output = dispatch_top_k(&[0.0, 2.0, 3.0], 2, |expert| {
            Ok::<_, ()>(vec![expert as f32, 1.0])
        })
        .unwrap();
        assert!(output[0] > 1.0 && output[0] < 2.0);
        assert!((output[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dispatch_refuses_different_widths() {
        let result = dispatch_top_k(&[1.0, 2.0], 2, |expert| {
            Ok::<_, ()>(if expert == 0 {
                vec![1.0]
            } else {
                vec![1.0, 2.0]
            })
        });
        assert!(matches!(
            result,
            Err(DispatchError::Route(RouteError::OutputWidth { .. }))
        ));
    }

    #[test]
    fn router_refuses_a_non_finite_logit() {
        assert!(matches!(
            select_top_k(&[f32::NAN, 1.0], 1),
            Err(RouteError::NonFiniteLogit { expert: 0 })
        ));
    }

    #[test]
    fn dispatch_route_preserves_the_resolved_ids_and_weights() {
        let output = dispatch_route(&[2, 0], &[0.25, 0.75], |expert| {
            Ok::<_, ()>(vec![expert as f32, 1.0])
        })
        .unwrap();
        assert!((output[0] - 0.5).abs() < 1e-6);
        assert!((output[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn dispatch_route_parallel_runs_concurrent_experts_and_preserves_order() {
        let barrier = Arc::new(Barrier::new(2));
        let output = dispatch_route_parallel(&[2, 0], &[0.25, 0.75], {
            let barrier = Arc::clone(&barrier);
            move |expert| {
                barrier.wait();
                Ok::<_, ()>(vec![expert as f32 + 1.0, 1.0])
            }
        })
        .unwrap();

        assert!((output[0] - 1.5).abs() < 1e-6);
        assert!((output[1] - 1.0).abs() < 1e-6);
    }
}
