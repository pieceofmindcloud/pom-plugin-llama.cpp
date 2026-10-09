//! Coletivas de tensor parallelism: all-reduce de ativações entre os ranks.
//!
//! O caminho C (tensor parallelism sem transferir peso) troca ativações entre
//! os nós de um grupo: cada rank produz uma soma parcial da projeção por
//! linhas e o resultado é a soma das parciais. Este módulo tem o núcleo
//! matemático (`reduce_sum`) e a barreira que coordena a soma entre ranks
//! (`ReduceBarrier`), modelada no `ReplyInbox` de `direct.rs`: cada
//! contribuição fica pendurada num `oneshot` até a barreira completar.
//!
//! O transporte HTTP (um `POST /v1/tp/reduce` no servidor de inferência) é
//! uma camada fina por cima, ligada junto com o fork do llama.cpp que chama a
//! coletiva no meio do forward pass.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;

/// Soma de vetores de mesmo tamanho. É o núcleo matemático da coletiva: o que
/// qualquer topologia (centralizada ou anel) calcula no fim. `None` quando os
/// vetores têm tamanhos diferentes ou a lista é vazia.
pub fn reduce_sum(partials: &[Vec<f32>]) -> Option<Vec<f32>> {
    let len = partials.first()?.len();
    if partials.iter().any(|p| p.len() != len) {
        return None;
    }
    let mut acc = vec![0.0f32; len];
    for p in partials {
        for (a, &v) in acc.iter_mut().zip(p) {
            *a += v;
        }
    }
    Some(acc)
}

/// Barreira de redução: acumula os parciais de cada rank e devolve a soma a
/// todos quando o grupo inteiro tiver contribuído.
///
/// O `barrier_id` é sorteado por redução (UUID, como o ticket do
/// `direct.rs`), então nunca há confusão entre duas reduções em voo.
#[derive(Clone, Default)]
pub struct ReduceBarrier {
    inner: Arc<Mutex<HashMap<String, BarrierEntry>>>,
}

struct BarrierEntry {
    n_ranks: u32,
    /// (rank, partial) já recebidos, em ordem de chegada.
    partials: Vec<(u32, Vec<f32>)>,
    /// Cada contribuidor pendura aqui o canal por onde recebe a soma.
    waiters: Vec<oneshot::Sender<Vec<f32>>>,
}

impl ReduceBarrier {
    pub fn new() -> Self {
        Self::default()
    }

    /// Contribui com o parcial deste rank e devolve o receiver da soma.
    ///
    /// Erra se `rank`/`n_ranks` forem inconsistentes, se o rank já tiver
    /// contribuído ou se o tamanho do vetor divergir do dos demais. O receiver
    /// resolve quando o último rank contribuir; quem chamou faz `await` nele.
    pub fn contribute(
        &self,
        barrier_id: &str,
        rank: u32,
        n_ranks: u32,
        data: Vec<f32>,
    ) -> Result<oneshot::Receiver<Vec<f32>>, String> {
        if n_ranks == 0 {
            return Err("n_ranks precisa ser >= 1".into());
        }
        if rank >= n_ranks {
            return Err(format!("rank {rank} fora de um grupo de {n_ranks}"));
        }

        let mut map = self.inner.lock().unwrap();
        let entry = map
            .entry(barrier_id.to_string())
            .or_insert_with(|| BarrierEntry {
                n_ranks,
                partials: Vec::new(),
                waiters: Vec::new(),
            });
        if entry.n_ranks != n_ranks {
            return Err(format!(
                "barreira {barrier_id} já registrada com n_ranks={}",
                entry.n_ranks
            ));
        }
        if entry.partials.iter().any(|(r, _)| *r == rank) {
            return Err(format!("rank {rank} já contribuiu para {barrier_id}"));
        }
        if let Some((_, first)) = entry.partials.first() {
            if first.len() != data.len() {
                return Err(format!(
                    "tamanho {} diverge do esperado {}",
                    data.len(),
                    first.len()
                ));
            }
        }
        entry.partials.push((rank, data));
        let (tx, rx) = oneshot::channel();
        entry.waiters.push(tx);

        if entry.partials.len() as u32 == n_ranks {
            // Soma em ordem de rank, para determinismo.
            let mut sorted = entry.partials.clone();
            sorted.sort_by_key(|(r, _)| *r);
            let sum = reduce_sum(&sorted.iter().map(|(_, d)| d.clone()).collect::<Vec<_>>())
                .expect("tamanhos já validados");
            for w in entry.waiters.drain(..) {
                let _ = w.send(sum.clone());
            }
            map.remove(barrier_id);
        }
        Ok(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_sum_sums_aligned_vectors() {
        assert_eq!(
            reduce_sum(&[vec![1.0, 2.0], vec![3.0, 4.0]]),
            Some(vec![4.0, 6.0])
        );
        assert_eq!(
            reduce_sum(&[vec![1.0, 2.0, 3.0]]),
            Some(vec![1.0, 2.0, 3.0])
        );
        // Tamanho divergente é recusa, não soma pela metade.
        assert_eq!(reduce_sum(&[vec![1.0], vec![2.0, 3.0]]), None);
        assert_eq!(reduce_sum(&[]), None);
    }

    #[tokio::test]
    async fn all_ranks_receive_the_reduced_sum() {
        let barrier = ReduceBarrier::new();
        let partials = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let expected = vec![9.0, 12.0];

        let mut handles = Vec::new();
        for rank in 0..3 {
            let barrier = barrier.clone();
            let data = partials[rank].clone();
            handles.push(tokio::spawn(async move {
                let rx = barrier
                    .contribute("b", rank as u32, 3, data)
                    .expect("contribuição válida");
                rx.await.expect("soma entregue")
            }));
        }
        for h in handles {
            assert_eq!(h.await.expect("task"), expected);
        }
    }

    #[tokio::test]
    async fn a_rank_cannot_contribute_twice_to_the_same_barrier() {
        let barrier = ReduceBarrier::new();
        let _rx0 = barrier.contribute("dup", 0, 2, vec![1.0]).unwrap();
        assert!(barrier.contribute("dup", 0, 2, vec![1.0]).is_err());
    }

    #[tokio::test]
    async fn mismatched_vector_size_is_rejected() {
        let barrier = ReduceBarrier::new();
        let _rx0 = barrier.contribute("size", 0, 2, vec![1.0, 2.0]).unwrap();
        assert!(barrier.contribute("size", 1, 2, vec![3.0]).is_err());
    }
}
