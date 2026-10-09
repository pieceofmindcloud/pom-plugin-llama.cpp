const MIN_CONFIDENT_SPEED_RATIO: f64 = 1.25;

/// Returns a device permutation ordered from the largest memory to the
/// smallest.
///
/// llama.cpp's fit fills the devices in the order it is given: the first one
/// takes layers up to its memory budget and each next one takes what is left.
/// Putting the largest card first uses the most memory of one card and leaves
/// the slack, and any layer that spills to system memory, on a smaller one.
/// With a speed order instead, a 12 GB card was filled to the brim, spilling
/// part of a layer, while a 16 GB card after it kept 3 GB unused.
///
/// Cards of the same size are ordered by a clearly different measured speed,
/// fastest first; otherwise they keep their current relative order. A device
/// that reports no memory makes the result `None`, so callers keep llama.cpp's
/// current order.
pub fn largest_memory_first_order(
    total_bytes: &[u64],
    scores: &[Option<f64>],
) -> Option<Vec<usize>> {
    if total_bytes.len() <= 1 || total_bytes.contains(&0) {
        return None;
    }
    let speed = |index: usize| {
        scores
            .get(index)
            .copied()
            .flatten()
            .filter(|score| score.is_finite() && *score > 0.0)
    };
    let mut order: Vec<usize> = (0..total_bytes.len()).collect();
    // A stable sort: what neither rule separates stays where it was.
    order.sort_by(|&left, &right| {
        total_bytes[right]
            .cmp(&total_bytes[left])
            .then_with(|| match (speed(left), speed(right)) {
                (Some(left), Some(right)) if left / right >= MIN_CONFIDENT_SPEED_RATIO => {
                    std::cmp::Ordering::Less
                }
                (Some(left), Some(right)) if right / left >= MIN_CONFIDENT_SPEED_RATIO => {
                    std::cmp::Ordering::Greater
                }
                _ => std::cmp::Ordering::Equal,
            })
    });
    Some(order)
}

/// Returns a memory order only when all devices use the same backend.
///
/// The llama.cpp scheduler supports an explicit device order for a homogeneous
/// backend, but a mixed backend graph has backend-specific split and copy
/// paths. Keep llama.cpp's default order when backend identity is incomplete or
/// heterogeneous, so a reordering cannot change that graph topology.
pub fn largest_memory_first_order_for_backends(
    total_bytes: &[u64],
    scores: &[Option<f64>],
    backend_names: &[Option<&str>],
) -> Option<Vec<usize>> {
    if total_bytes.len() != backend_names.len()
        || backend_names.len() <= 1
        || backend_names.iter().any(Option::is_none)
        || backend_names.windows(2).any(|pair| pair[0] != pair[1])
    {
        return None;
    }

    largest_memory_first_order(total_bytes, scores)
}

#[cfg(test)]
mod tests {
    use super::{largest_memory_first_order, largest_memory_first_order_for_backends};

    const GIB: u64 = 1 << 30;

    #[test]
    fn one_device_keeps_the_current_order() {
        assert_eq!(
            largest_memory_first_order(&[16 * GIB], &[Some(100.0)]),
            None
        );
    }

    #[test]
    fn the_card_with_more_memory_comes_first() {
        assert_eq!(
            largest_memory_first_order(&[12 * GIB, 16 * GIB], &[]),
            Some(vec![1, 0])
        );
        assert_eq!(
            largest_memory_first_order(&[12 * GIB, 24 * GIB, 16 * GIB], &[]),
            Some(vec![1, 2, 0])
        );
    }

    #[test]
    fn memory_decides_even_when_the_smaller_card_is_the_fast_one() {
        // O dispositivo 1 é o rápido, mas o 0 tem mais memória: é ele que o fit
        // deve encher primeiro.
        assert_eq!(
            largest_memory_first_order(&[16 * GIB, 4 * GIB], &[Some(80.0), Some(400.0)]),
            Some(vec![0, 1])
        );
    }

    #[test]
    fn same_size_cards_are_tie_broken_by_clear_speed() {
        assert_eq!(
            largest_memory_first_order(&[16 * GIB, 16 * GIB], &[Some(100.0), Some(400.0)]),
            Some(vec![1, 0])
        );
        // Diferença pequena é ruído de medição: a ordem atual fica.
        assert_eq!(
            largest_memory_first_order(&[16 * GIB, 16 * GIB], &[Some(100.0), Some(105.0)]),
            Some(vec![0, 1])
        );
        // Sem medição utilizável, idem.
        for scores in [
            vec![Some(400.0), None],
            vec![Some(f64::NAN), Some(400.0)],
            vec![Some(0.0), Some(400.0)],
            vec![],
        ] {
            assert_eq!(
                largest_memory_first_order(&[16 * GIB, 16 * GIB], &scores),
                Some(vec![0, 1])
            );
        }
    }

    #[test]
    fn unknown_memory_keeps_the_current_order() {
        assert_eq!(largest_memory_first_order(&[0, 16 * GIB], &[]), None);
    }

    #[test]
    fn heterogeneous_backends_get_no_explicit_order() {
        let backends = [Some("Vulkan"), Some("CUDA")];
        assert_eq!(
            largest_memory_first_order_for_backends(&[12 * GIB, 16 * GIB], &[], &backends),
            None
        );
        assert_eq!(
            largest_memory_first_order_for_backends(
                &[12 * GIB, 16 * GIB],
                &[],
                &[Some("CUDA"), None]
            ),
            None
        );
    }

    #[test]
    fn a_homogeneous_backend_gets_the_order_by_memory() {
        let backends = [Some("CUDA"), Some("CUDA")];
        assert_eq!(
            largest_memory_first_order_for_backends(
                &[12 * GIB, 16 * GIB],
                &[Some(100.0), Some(400.0)],
                &backends
            ),
            Some(vec![1, 0])
        );
    }
}
