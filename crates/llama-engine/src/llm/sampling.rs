use super::*;

// ── Sampling ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    /// Subtracted once from the logit of every token seen in the window,
    /// however often it appeared (llama.cpp's `presence_penalty`). 0 is off.
    pub presence_penalty: f32,
    /// Subtracted from the logit of a token once per time it appeared in the
    /// window (llama.cpp's `frequency_penalty`). 0 is off.
    pub frequency_penalty: f32,
    /// Drops every candidate less likely than this fraction of the most
    /// likely one, after top-k and top-p (llama.cpp's `min_p`). 0 is off.
    pub min_p: f32,
    /// llama.cpp's DRY sampler: penalizes the token that would extend a
    /// sequence already seen. Off while `multiplier` is 0.
    pub dry: DrySampling,
    /// llama.cpp's reasoning budget: how many tokens a thinking block may
    /// take before its end tag is forced. Off while `tokens` is negative.
    pub reasoning: ReasoningBudget,
}

/// A limit on the tokens of a thinking block. When the block reaches
/// `tokens`, the sampler stops choosing and emits the end tag, token by
/// token, so the model goes on to the answer.
#[derive(Debug, Clone, PartialEq)]
pub struct ReasoningBudget {
    /// -1 is no limit; 0 closes the block at once.
    pub tokens: i32,
    /// The tokens of the tag that opens a thinking block (`<think>`).
    pub start: Vec<i32>,
    /// The tokens of the tag that closes it (`</think>`).
    pub end: Vec<i32>,
}

impl Default for ReasoningBudget {
    fn default() -> Self {
        Self {
            tokens: -1,
            start: Vec::new(),
            end: Vec::new(),
        }
    }
}

/// The texts of the tags that delimit a thinking block.
pub const THINKING_START_TAG: &str = "<think>";
pub const THINKING_END_TAG: &str = "</think>";

/// Index right after the last occurrence of `needle` in `tokens`.
fn after_last(tokens: &[i32], needle: &[i32]) -> Option<usize> {
    if needle.is_empty() || tokens.len() < needle.len() {
        return None;
    }
    (0..=tokens.len() - needle.len())
        .rev()
        .find(|&start| &tokens[start..start + needle.len()] == needle)
        .map(|start| start + needle.len())
}

/// The token the reasoning budget forces next, if the thinking block of this
/// generation used its budget up. `generated` holds the tokens generated so
/// far in this request; `open_at_start` says the prompt itself already opened
/// the block, so no start tag is among them.
pub fn forced_reasoning_token(
    generated: &[i32],
    budget: &ReasoningBudget,
    open_at_start: bool,
) -> Option<i32> {
    if budget.tokens < 0 || budget.end.is_empty() {
        return None;
    }
    let opened = after_last(generated, &budget.start);
    let closed = after_last(generated, &budget.end);
    let origin = match (opened, closed) {
        (Some(opened), Some(closed)) if opened > closed => opened,
        (Some(opened), None) => opened,
        (None, None) if open_at_start => 0,
        // Closed, or never opened: the answer is being written.
        _ => return None,
    };
    if generated.len() - origin < budget.tokens as usize {
        return None;
    }
    // How much of the end tag is already out: it is forced one token at a
    // time, and a tag of several tokens continues where it stopped.
    let written = (1..budget.end.len())
        .rev()
        .find(|&count| generated.ends_with(&budget.end[..count]))
        .unwrap_or(0);
    Some(budget.end[written])
}

/// `sample_token_profiled` under the reasoning budget: a forced token is
/// returned without sampling.
pub fn sample_token_within_budget(
    logits: &[f32],
    recent: &[i32],
    cfg: &SamplingConfig,
    thinking_open_at_start: bool,
) -> (i32, SamplingTimings) {
    match forced_reasoning_token(recent, &cfg.reasoning, thinking_open_at_start) {
        Some(token) => (token, SamplingTimings::default()),
        None => sample_token_profiled(logits, recent, cfg),
    }
}

/// DRY ("don't repeat yourself"). A token that would continue a run of
/// `match` tokens that already appeared loses
/// `multiplier * base^(match - allowed_length)` from its logit, once the run
/// is at least `allowed_length` long: short repeats are free, long ones cost
/// exponentially more.
#[derive(Debug, Clone, PartialEq)]
pub struct DrySampling {
    pub multiplier: f32,
    pub base: f32,
    pub allowed_length: usize,
    /// How many of the most recent tokens are searched; 0 searches all the
    /// history the sampler is given, up to `DRY_MAX_WINDOW`.
    pub penalty_last_n: usize,
    /// Tokens a repeated run may not cross (llama.cpp's sequence breakers:
    /// newline, `:`, `"`, `*`), so the frame of a list or a table is not
    /// taken for a repetition.
    pub breakers: Vec<i32>,
}

impl Default for DrySampling {
    fn default() -> Self {
        Self {
            multiplier: 0.0,
            base: 1.75,
            allowed_length: 2,
            penalty_last_n: 1024,
            breakers: Vec::new(),
        }
    }
}

/// The texts llama.cpp uses as DRY sequence breakers by default.
pub const DRY_SEQUENCE_BREAKERS: [&str; 4] = ["\n", ":", "\"", "*"];
const DRY_MAX_WINDOW: usize = 8192;
/// Longer runs than this cost the same: the penalty is already overwhelming.
const DRY_MAX_MATCH: usize = 50;

/// What DRY takes from the logit of each token that would extend a repeated
/// run, given the tokens so far.
pub(super) fn dry_penalties(
    recent: &[i32],
    dry: &DrySampling,
) -> std::collections::HashMap<i32, f32> {
    let mut penalties = std::collections::HashMap::new();
    if dry.multiplier <= 0.0 || dry.base < 1.0 || recent.len() < 2 {
        return penalties;
    }
    let limit = match dry.penalty_last_n {
        0 => DRY_MAX_WINDOW,
        n => n.min(DRY_MAX_WINDOW),
    };
    let window = &recent[recent.len().saturating_sub(limit)..];
    let last = window.len() - 1;
    // How far back the current run may reach: it stops at a breaker.
    let mut reach = 0;
    while reach < DRY_MAX_MATCH.min(window.len() - 1)
        && !dry.breakers.contains(&window[last - reach])
    {
        reach += 1;
    }
    let allowed = dry.allowed_length.max(1);
    if reach < allowed {
        return penalties;
    }
    let mut longest: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    for end in 0..last {
        if window[end] != window[last] {
            continue;
        }
        // `end` closes an earlier occurrence of the current run: count how
        // many tokens of it match, then the token after it is the one that
        // would repeat.
        let mut matched = 1;
        while matched < reach && matched <= end && window[end - matched] == window[last - matched] {
            matched += 1;
        }
        if matched >= allowed {
            let entry = longest.entry(window[end + 1]).or_insert(0);
            *entry = (*entry).max(matched);
        }
    }
    for (token, matched) in longest {
        penalties.insert(
            token,
            dry.multiplier * dry.base.powi((matched - allowed) as i32),
        );
    }
    penalties
}

impl SamplingConfig {
    /// Whether any penalty changes a logit.
    fn penalizes(&self) -> bool {
        self.repeat_penalty != 1.0
            || self.presence_penalty != 0.0
            || self.frequency_penalty != 0.0
            || self.dry.multiplier > 0.0
    }

    /// The logit of a token that appeared `count` times in the window, in
    /// llama.cpp's order: the repeat penalty scales it, then frequency and
    /// presence are subtracted.
    fn penalized(&self, logit: f32, count: usize) -> f32 {
        let scaled = if logit > 0.0 {
            logit / self.repeat_penalty
        } else {
            logit * self.repeat_penalty
        };
        scaled - count as f32 * self.frequency_penalty - self.presence_penalty
    }
}

/// How often each token appears in the last `repeat_last_n` tokens.
fn window_counts(recent: &[i32], cfg: &SamplingConfig) -> std::collections::HashMap<i32, usize> {
    let window_start = recent.len().saturating_sub(cfg.repeat_last_n);
    let mut counts = std::collections::HashMap::new();
    for &token in &recent[window_start..] {
        *counts.entry(token).or_insert(0) += 1;
    }
    counts
}

/// CPU-only sampling boundaries. GPU/backend completion is measured before
/// this function, so a long CPU sampler cannot hide a pending CUDA stream.
///
/// Measured in **microseconds**. Milliseconds truncated every stage that cost
/// less than 1 ms to zero, which after this rewrite is every stage — the whole
/// sampler now costs less than one of the old millisecond ticks.
/// O acesso aos logits e a conversão do token em texto ficam de fora: quem os
/// mede é o chamador (`infer_server::NodeTimings`), porque acontecem antes e
/// depois desta função. Campo que ninguém preenche vira zero na tela e some da
/// soma sem avisar.
#[derive(Debug, Clone, Default)]
pub struct SamplingTimings {
    /// Kept in the telemetry shape; sparse repetition adjustments avoid this copy.
    pub logits_clone_us: u64,
    pub repetition_penalty_us: u64,
    pub softmax_us: u64,
    pub top_k_us: u64,
    pub top_p_us: u64,
    pub selection_us: u64,
}

/// The state the sampler's generator starts from when a deployment names no
/// seed. Fixed, so a node that restarts and receives the same requests in the
/// same order draws the same tokens.
pub const DEFAULT_SAMPLER_SEED: u64 = 0x853c49e6748fea9b;

pub(super) static SAMPLER_RNG: Mutex<u64> = Mutex::new(DEFAULT_SAMPLER_SEED);

/// Restarts the sampler's generator from `seed`. Called when a deployment is
/// loaded: the sequence then runs on across requests, it is not restarted for
/// each one.
pub fn set_sampler_seed(seed: u64) {
    *SAMPLER_RNG
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = seed;
}

/// The seed in JSON: text, because a browser cannot hold every 64-bit integer
/// in a number. A number is still accepted on the way in.
pub mod seed_text {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(seed: &u64, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&seed.to_string())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u64, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Seed {
            Number(u64),
            Text(String),
        }
        match Seed::deserialize(deserializer)? {
            Seed::Number(seed) => Ok(seed),
            Seed::Text(text) => text.trim().parse::<u64>().map_err(|_| {
                serde::de::Error::custom(
                    "the seed is a whole number from 0 to 18446744073709551615",
                )
            }),
        }
    }

    pub fn default_seed() -> u64 {
        super::DEFAULT_SAMPLER_SEED
    }
}

#[cfg(test)]
pub(super) static SAMPLER_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Probability of each candidate answer at one position, read from the raw
/// logits (temperature 1, no penalty, no truncation).
///
/// Used by typed decisions: instead of sampling, the last peer reports how
/// much of the model's next-token distribution falls on each candidate. A
/// candidate is a group of token ids (for example `"A"` and `" A"`); its
/// probability is the sum of theirs. `mass` is the total probability the
/// candidates cover, so a caller can tell a confident pick among them from a
/// model that wanted to say something else.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeResult {
    pub probabilities: Vec<f32>,
    pub mass: f32,
}

pub fn probe_candidates(logits: &[f32], candidates: &[Vec<i32>]) -> ProbeResult {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if logits.is_empty() || !max.is_finite() {
        return ProbeResult {
            probabilities: vec![0.0; candidates.len()],
            mass: 0.0,
        };
    }
    let total: f64 = logits
        .iter()
        .map(|value| f64::from((value - max).exp()))
        .sum();
    let mut seen = std::collections::HashSet::new();
    let probabilities: Vec<f32> = candidates
        .iter()
        .map(|ids| {
            ids.iter()
                .filter(|id| **id >= 0 && (**id as usize) < logits.len() && seen.insert(**id))
                .map(|id| f64::from((logits[*id as usize] - max).exp()) / total)
                .sum::<f64>() as f32
        })
        .collect();
    let mass = probabilities.iter().sum();
    ProbeResult {
        probabilities,
        mass,
    }
}

/// Temperature + top-k + top-p + repetition-penalty sampling.
///
/// Returns the sampled token id. When `temperature <= 0` this is greedy argmax.
pub fn sample_token_profiled(
    logits: &[f32],
    recent: &[i32],
    cfg: &SamplingConfig,
) -> (i32, SamplingTimings) {
    let mut timings = SamplingTimings::default();
    if logits.is_empty() {
        return (0, timings);
    }
    let kept = sampling_candidates_profiled(logits, recent, cfg, &mut timings);
    if cfg.temperature <= 0.0 {
        let selection_started = Instant::now();
        let token = kept[0].0 as i32;
        timings.selection_us = selection_started.elapsed().as_micros() as u64;
        return (token, timings);
    }
    // ── Multinomial draw ─────────────────────────────────────────────────────
    //
    // The candidates are normalized and in ascending id order, so this is the
    // only place that draws from the distribution.
    let selection_started = Instant::now();
    let rng_val: f32 = {
        // A simple LCG, to avoid pulling in `rand`.
        let mut r = SAMPLER_RNG.lock().unwrap();
        next_sampler_rng_value(&mut r)
    };

    let mut cumsum = 0.0f32;
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, p) in &kept {
        cumsum += *p;
        if rng_val <= cumsum {
            timings.selection_us = selection_started.elapsed().as_micros() as u64;
            return (*i as i32, timings);
        }
        // Rounding can leave `rng_val` just past the final cumulative sum; the
        // most likely token is the answer then, matching the old fallback.
        if *p >= best.1 {
            best = (*i, *p);
        }
    }
    timings.selection_us = selection_started.elapsed().as_micros() as u64;
    (best.0 as i32, timings)
}

pub(super) fn next_sampler_rng_value(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mantissa = ((*state >> 33) as u32) & 0x007f_ffff;
    let bits = mantissa | 0x3f800000;
    f32::from_bits(bits) - 1.0
}

pub(super) fn sampling_candidates_profiled(
    logits: &[f32],
    recent: &[i32],
    cfg: &SamplingConfig,
    timings: &mut SamplingTimings,
) -> Vec<(usize, f32)> {
    let repetition_started = Instant::now();
    let adjustments = repetition_adjustments(logits, recent, cfg);
    timings.repetition_penalty_us = repetition_started.elapsed().as_micros() as u64;

    if cfg.temperature <= 0.0 {
        return vec![(argmax_adjusted(logits, &adjustments) as usize, 1.0)];
    }

    // Positive-temperature softmax preserves logit order. Repeated IDs are
    // adjusted before top-k refinement, and token ID resolves every tie.
    let selects_top_k = cfg.top_k > 0 && (cfg.top_k as usize) < logits.len();
    let mut candidates = if selects_top_k {
        let top_k_started = Instant::now();
        let ids = top_k_ids_adjusted(logits, &adjustments, cfg.top_k as usize);
        timings.top_k_us = top_k_started.elapsed().as_micros() as u64;

        let softmax_started = Instant::now();
        let max_l = ids
            .iter()
            .map(|&id| adjusted_logit_at(logits, &adjustments, id))
            .fold(f32::NEG_INFINITY, f32::max);
        let mut candidates = ids
            .into_iter()
            .map(|id| {
                (
                    id,
                    ((adjusted_logit_at(logits, &adjustments, id) - max_l) / cfg.temperature).exp(),
                )
            })
            .collect::<Vec<_>>();
        normalise_candidates_with_fallback(&mut candidates, || {
            argmax_adjusted(logits, &adjustments) as usize
        });
        timings.softmax_us = softmax_started.elapsed().as_micros() as u64;
        candidates
    } else {
        // With top-k disabled, stream token IDs directly into the probability
        // vector. Materializing a second vocabulary-sized ID list is redundant.
        let softmax_started = Instant::now();
        let max_l = if adjustments.is_empty() {
            logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        } else {
            adjusted_logits(logits, &adjustments)
                .map(|(_, logit)| logit)
                .fold(f32::NEG_INFINITY, f32::max)
        };
        let mut candidates = if adjustments.is_empty() {
            logits
                .iter()
                .copied()
                .enumerate()
                .map(|(id, logit)| (id, ((logit - max_l) / cfg.temperature).exp()))
                .collect::<Vec<_>>()
        } else {
            adjusted_logits(logits, &adjustments)
                .map(|(id, logit)| (id, ((logit - max_l) / cfg.temperature).exp()))
                .collect::<Vec<_>>()
        };
        normalise_candidates_with_fallback(&mut candidates, || {
            argmax_adjusted(logits, &adjustments) as usize
        });
        timings.softmax_us = softmax_started.elapsed().as_micros() as u64;
        candidates
    };

    let top_p_started = Instant::now();
    if cfg.top_p < 1.0 {
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let target = cfg.top_p.max(0.0);
        let mut mass = 0.0;
        let mut keep = 0;
        for (_, probability) in &candidates {
            mass += *probability;
            keep += 1;
            if mass >= target {
                break;
            }
        }
        candidates.truncate(keep);
        candidates.sort_unstable_by_key(|(id, _)| *id);
        normalise_candidates_with_fallback(&mut candidates, || {
            argmax_adjusted(logits, &adjustments) as usize
        });
    }
    if cfg.min_p > 0.0 && candidates.len() > 1 {
        let best = candidates
            .iter()
            .map(|(_, probability)| *probability)
            .fold(0.0_f32, f32::max);
        let floor = best * cfg.min_p.min(1.0);
        candidates.retain(|(_, probability)| *probability >= floor);
        normalise_candidates_with_fallback(&mut candidates, || {
            argmax_adjusted(logits, &adjustments) as usize
        });
    }
    timings.top_p_us = top_p_started.elapsed().as_micros() as u64;
    candidates
}

#[cfg(test)]
pub(super) fn sampling_candidates(
    logits: &[f32],
    recent: &[i32],
    cfg: &SamplingConfig,
) -> Vec<(usize, f32)> {
    sampling_candidates_profiled(logits, recent, cfg, &mut SamplingTimings::default())
}

#[cfg(test)]
pub(super) fn apply_repetition_penalty(logits: &mut [f32], recent: &[i32], cfg: &SamplingConfig) {
    for (id, adjusted) in repetition_adjustments(logits, recent, cfg) {
        logits[id] = adjusted;
    }
}

/// The logits the penalties change, by token id: the window's repeat,
/// frequency and presence penalties, then DRY.
fn repetition_adjustments(
    logits: &[f32],
    recent: &[i32],
    cfg: &SamplingConfig,
) -> Vec<(usize, f32)> {
    if !cfg.penalizes() || recent.is_empty() {
        return Vec::new();
    }

    let windowed =
        cfg.repeat_penalty != 1.0 || cfg.presence_penalty != 0.0 || cfg.frequency_penalty != 0.0;
    let mut adjusted: std::collections::BTreeMap<usize, f32> = std::collections::BTreeMap::new();
    if windowed {
        for (token, count) in window_counts(recent, cfg) {
            if let Some(id) = usize::try_from(token).ok().filter(|&id| id < logits.len()) {
                adjusted.insert(id, cfg.penalized(logits[id], count));
            }
        }
    }
    for (token, penalty) in dry_penalties(recent, &cfg.dry) {
        if let Some(id) = usize::try_from(token).ok().filter(|&id| id < logits.len()) {
            *adjusted.entry(id).or_insert(logits[id]) -= penalty;
        }
    }
    adjusted.into_iter().collect()
}

fn adjusted_logit_at(logits: &[f32], adjustments: &[(usize, f32)], id: usize) -> f32 {
    if adjustments.is_empty() {
        return logits[id];
    }
    match adjustments.binary_search_by_key(&id, |(adjusted_id, _)| *adjusted_id) {
        Ok(index) => adjustments[index].1,
        Err(_) => logits[id],
    }
}

fn adjusted_logits<'a>(
    logits: &'a [f32],
    adjustments: &'a [(usize, f32)],
) -> impl Iterator<Item = (usize, f32)> + 'a {
    let mut adjustment_index = 0;
    logits.iter().copied().enumerate().map(move |(id, logit)| {
        while adjustment_index < adjustments.len() && adjustments[adjustment_index].0 < id {
            adjustment_index += 1;
        }
        if adjustment_index < adjustments.len() && adjustments[adjustment_index].0 == id {
            let adjusted = adjustments[adjustment_index].1;
            adjustment_index += 1;
            (id, adjusted)
        } else {
            (id, logit)
        }
    })
}

fn argmax_adjusted(logits: &[f32], adjustments: &[(usize, f32)]) -> i32 {
    if adjustments.is_empty() {
        return argmax(logits);
    }
    adjusted_logits(logits, adjustments)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(id, _)| id as i32)
        .unwrap_or(0)
}

fn top_k_ids_adjusted(logits: &[f32], adjustments: &[(usize, f32)], k: usize) -> Vec<usize> {
    if adjustments.is_empty() {
        return top_k_ids_compact(logits, k);
    }

    // At most one adjusted token can displace one raw top-k token. Keeping the
    // raw top (k + adjusted_count) and every adjusted ID therefore contains the
    // exact adjusted top-k, including penalties that raise a logit.
    let raw_count = k.saturating_add(adjustments.len()).min(logits.len());
    let mut ids = top_k_ids_compact(logits, raw_count);
    ids.extend(adjustments.iter().map(|(id, _)| *id));
    ids.sort_unstable();
    ids.dedup();

    let mut ranked: Vec<(usize, f32)> = ids
        .into_iter()
        .map(|id| (id, adjusted_logit_at(logits, adjustments, id)))
        .collect();
    ranked.select_nth_unstable_by(k - 1, |a, b| {
        b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
    });
    ranked.truncate(k);
    ranked.sort_unstable_by_key(|(id, _)| *id);
    ranked.into_iter().map(|(id, _)| id).collect()
}

/// Select the highest `k` token indices and return them in ID order.
///
/// A real model vocabulary fits in a `u32`; using that compact scratch vector
/// halves the per-token allocation and memory traffic on the full-vocabulary
/// selection. Keep the wider fallback for slices too large to represent.
pub(super) fn top_k_ids_compact(logits: &[f32], k: usize) -> Vec<usize> {
    if k == 0 || k >= logits.len() {
        return (0..logits.len()).collect();
    }

    if logits.len() <= u32::MAX as usize {
        let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
        ids.select_nth_unstable_by(k - 1, |&a, &b| {
            logits[b as usize]
                .total_cmp(&logits[a as usize])
                .then_with(|| a.cmp(&b))
        });
        ids.truncate(k);
        ids.sort_unstable();
        ids.into_iter().map(|id| id as usize).collect()
    } else {
        let mut ids: Vec<usize> = (0..logits.len()).collect();
        ids.select_nth_unstable_by(k - 1, |&a, &b| {
            logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b))
        });
        ids.truncate(k);
        ids.sort_unstable();
        ids
    }
}

#[cfg(test)]
mod compact_top_k_tests {
    use super::*;

    fn full_sort_reference(logits: &[f32], k: usize) -> Vec<usize> {
        let mut ids: Vec<usize> = (0..logits.len()).collect();
        ids.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b)));
        ids.truncate(k);
        ids.sort_unstable();
        ids
    }

    #[test]
    fn compact_indices_match_full_sort_with_ties_and_total_float_order() {
        let cases = [
            (vec![7.0, 7.0, 7.0, 1.0], 1),
            (vec![3.0, 3.0, 2.0, 2.0, 1.0], 3),
            (
                vec![f32::NAN, 0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY],
                3,
            ),
            ((0..257).map(|i| ((i % 11) as f32) - 5.0).collect(), 40),
        ];

        for (logits, k) in cases {
            assert_eq!(
                top_k_ids_compact(&logits, k),
                full_sort_reference(&logits, k)
            );
        }
    }

    #[test]
    fn compact_indices_match_full_sort_on_a_large_tied_vocabulary() {
        const VOCAB: usize = 151_936;
        let logits: Vec<f32> = (0..VOCAB)
            .map(|i| ((i.wrapping_mul(48_271) % 32_768) as i32 - 16_384) as f32 * 0.125)
            .collect();

        assert_eq!(
            top_k_ids_compact(&logits, 40),
            full_sort_reference(&logits, 40)
        );
    }
}

#[cfg(test)]
mod sampler_profile_regressions {
    use super::*;

    fn cfg(temperature: f32, top_k: i32, top_p: f32) -> SamplingConfig {
        SamplingConfig {
            temperature,
            top_p,
            top_k,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            min_p: 0.0,
            dry: Default::default(),
            reasoning: Default::default(),
        }
    }

    fn candidate_ids(logits: &[f32], recent: &[i32], cfg: &SamplingConfig) -> Vec<usize> {
        sampling_candidates(logits, recent, cfg)
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    fn legacy_top_k_ids(logits: &[f32], top_k: i32) -> Vec<usize> {
        if top_k <= 0 || (top_k as usize) >= logits.len() {
            return (0..logits.len()).collect();
        }
        let k = top_k as usize;
        let mut ids: Vec<usize> = (0..logits.len()).collect();
        ids.select_nth_unstable_by(k - 1, |&a, &b| {
            logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b))
        });
        ids.truncate(k);
        ids.sort_unstable();
        ids
    }

    fn legacy_sampling_candidates_profiled(
        logits: &[f32],
        recent: &[i32],
        cfg: &SamplingConfig,
        timings: &mut SamplingTimings,
    ) -> Vec<(usize, f32)> {
        let clone_started = Instant::now();
        let mut adjusted = logits.to_vec();
        timings.logits_clone_us = clone_started.elapsed().as_micros() as u64;

        let repetition_started = Instant::now();
        apply_repetition_penalty(&mut adjusted, recent, cfg);
        timings.repetition_penalty_us = repetition_started.elapsed().as_micros() as u64;
        if cfg.temperature <= 0.0 {
            return vec![(argmax(&adjusted) as usize, 1.0)];
        }

        let top_k_started = Instant::now();
        let ids = legacy_top_k_ids(&adjusted, cfg.top_k);
        timings.top_k_us = top_k_started.elapsed().as_micros() as u64;
        let softmax_started = Instant::now();
        let max_logit = ids
            .iter()
            .map(|&id| adjusted[id])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut candidates: Vec<(usize, f32)> = ids
            .into_iter()
            .map(|id| (id, ((adjusted[id] - max_logit) / cfg.temperature).exp()))
            .collect();
        normalise_candidates(&mut candidates, &adjusted);
        timings.softmax_us = softmax_started.elapsed().as_micros() as u64;

        let top_p_started = Instant::now();
        if cfg.top_p < 1.0 {
            let mut ranked = candidates.clone();
            ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let target = cfg.top_p.max(0.0);
            let mut mass = 0.0;
            let mut keep = 0;
            for (_, probability) in &ranked {
                mass += *probability;
                keep += 1;
                if mass >= target {
                    break;
                }
            }
            ranked.truncate(keep);
            ranked.sort_unstable_by_key(|(id, _)| *id);
            candidates = ranked;
            normalise_candidates(&mut candidates, &adjusted);
        }
        timings.top_p_us = top_p_started.elapsed().as_micros() as u64;
        candidates
    }

    fn measure_candidates(
        sampler: fn(&[f32], &[i32], &SamplingConfig, &mut SamplingTimings) -> Vec<(usize, f32)>,
        logits: &[f32],
        recent: &[i32],
        cfg: &SamplingConfig,
        rounds: usize,
    ) -> (f64, SamplingTimings) {
        let mut totals = SamplingTimings::default();
        let started = Instant::now();
        for _ in 0..rounds {
            let mut timings = SamplingTimings::default();
            std::hint::black_box(sampler(logits, recent, cfg, &mut timings));
            totals.logits_clone_us += timings.logits_clone_us;
            totals.repetition_penalty_us += timings.repetition_penalty_us;
            totals.softmax_us += timings.softmax_us;
            totals.top_k_us += timings.top_k_us;
            totals.top_p_us += timings.top_p_us;
        }
        (
            started.elapsed().as_secs_f64() * 1_000.0 / rounds as f64,
            totals,
        )
    }

    #[test]
    #[ignore = "benchmark manual em release para comparar o caminho anterior e o atual"]
    fn sampler_cost_profiles_on_a_qwen_vocabulary() {
        const VOCAB: usize = 151_936;
        const ROUNDS: usize = 100;
        let logits: Vec<f32> = (0..VOCAB).map(|i| ((i % 977) as f32) * 0.01).collect();
        let recent: Vec<i32> = (0..64).map(|i| (i * 1979 % VOCAB) as i32).collect();
        let actual_defaults = SamplingConfig {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.15,
            repeat_last_n: 64,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            min_p: 0.0,
            dry: Default::default(),
            reasoning: Default::default(),
        };
        let cases = [
            ("top_k_enabled", cfg(0.7, 40, 1.0), &[][..]),
            ("deployed_defaults", actual_defaults, recent.as_slice()),
            ("top_k_disabled", cfg(0.7, 0, 0.9), &[][..]),
        ];

        for (name, config, recent) in cases {
            assert_eq!(
                legacy_sampling_candidates_profiled(
                    &logits,
                    recent,
                    &config,
                    &mut SamplingTimings::default()
                ),
                sampling_candidates_profiled(
                    &logits,
                    recent,
                    &config,
                    &mut SamplingTimings::default()
                ),
                "candidate distribution changed for {name}"
            );
            let legacy = measure_candidates(
                legacy_sampling_candidates_profiled,
                &logits,
                recent,
                &config,
                ROUNDS,
            );
            let current = measure_candidates(
                sampling_candidates_profiled,
                &logits,
                recent,
                &config,
                ROUNDS,
            );
            let avg = |total: u64| total as f64 / ROUNDS as f64;
            println!(
                "{name}: legacy wall={:.1}us clone={:.1}us repetition={:.1}us softmax={:.1}us top_k={:.1}us top_p={:.1}us; current wall={:.1}us clone={:.1}us repetition={:.1}us softmax={:.1}us top_k={:.1}us top_p={:.1}us",
                legacy.0 * 1_000.0,
                avg(legacy.1.logits_clone_us),
                avg(legacy.1.repetition_penalty_us),
                avg(legacy.1.softmax_us),
                avg(legacy.1.top_k_us),
                avg(legacy.1.top_p_us),
                current.0 * 1_000.0,
                avg(current.1.logits_clone_us),
                avg(current.1.repetition_penalty_us),
                avg(current.1.softmax_us),
                avg(current.1.top_k_us),
                avg(current.1.top_p_us),
            );
        }
    }

    #[test]
    fn sparse_adjustments_preserve_candidates_across_sampling_options() {
        let logits = [4.0, 4.0, 1.0, -2.0, -2.0, 0.5, 3.0];
        let recent = [0, 0, 3, 5, -1, 99];
        let configs = [
            SamplingConfig {
                repeat_penalty: 1.15,
                repeat_last_n: 5,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                min_p: 0.0,
                dry: Default::default(),
                reasoning: Default::default(),
                ..cfg(0.7, 3, 0.9)
            },
            SamplingConfig {
                repeat_penalty: 0.5,
                repeat_last_n: 6,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                min_p: 0.0,
                dry: Default::default(),
                reasoning: Default::default(),
                ..cfg(1.2, 0, 0.5)
            },
            SamplingConfig {
                repeat_penalty: 2.0,
                repeat_last_n: 4,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                min_p: 0.0,
                dry: Default::default(),
                reasoning: Default::default(),
                ..cfg(0.0, 2, 0.1)
            },
            cfg(1.0, 40, 1.0),
        ];

        for config in configs {
            let expected = legacy_sampling_candidates_profiled(
                &logits,
                &recent,
                &config,
                &mut SamplingTimings::default(),
            );
            assert_eq!(
                sampling_candidates(&logits, &recent, &config),
                expected,
                "config={config:?}"
            );
        }
    }

    #[test]
    fn active_repetition_penalty_does_not_clone_the_full_vocabulary() {
        const VOCAB: usize = 151_936;
        let logits: Vec<f32> = (0..VOCAB).map(|i| ((i % 977) as f32) * 0.01).collect();
        let recent: Vec<i32> = (0..64).map(|i| (i * 1979 % VOCAB) as i32).collect();
        let config = SamplingConfig {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.15,
            repeat_last_n: 64,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            min_p: 0.0,
            dry: Default::default(),
            reasoning: Default::default(),
        };

        let expected = legacy_sampling_candidates_profiled(
            &logits,
            &recent,
            &config,
            &mut SamplingTimings::default(),
        );
        assert_eq!(sampling_candidates(&logits, &recent, &config), expected);
        let (_, timings) = sample_token_profiled(&logits, &recent, &config);
        assert_eq!(timings.logits_clone_us, 0);
    }

    #[test]
    fn disabled_top_k_does_not_materialize_a_vocabulary_sized_id_list() {
        const VOCAB: usize = 151_936;
        let logits = vec![0.25; VOCAB];
        let config = cfg(0.7, 0, 1.0);

        let (_, timings) = sample_token_profiled(&logits, &[], &config);
        assert_eq!(timings.top_k_us, 0);
        assert!(timings.softmax_us > 0);
    }

    #[test]
    fn presence_and_frequency_penalties_subtract_from_seen_tokens() {
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repeat_penalty: 1.0,
            repeat_last_n: 4,
            presence_penalty: 1.5,
            frequency_penalty: 0.5,
            min_p: 0.0,
            dry: Default::default(),
            reasoning: Default::default(),
        };
        // Token 9 is outside the window of four; token 1 appears twice in it.
        let recent = [9, 1, 2, 1, 3];
        let mut logits = vec![5.0_f32; 10];
        apply_repetition_penalty(&mut logits, &recent, &cfg);
        assert_eq!(logits[1], 5.0 - 2.0 * 0.5 - 1.5);
        assert_eq!(logits[2], 5.0 - 0.5 - 1.5);
        assert_eq!(logits[3], 5.0 - 0.5 - 1.5);
        assert_eq!((logits[0], logits[9]), (5.0, 5.0));
        // The sparse path the sampler uses gives the same numbers.
        let adjustments = repetition_adjustments(&[5.0; 10], &recent, &cfg);
        assert_eq!(adjustments, vec![(1, 2.5), (2, 3.0), (3, 3.0)]);

        // The repeat penalty scales first, then the others are subtracted.
        let both = SamplingConfig {
            repeat_penalty: 2.0,
            frequency_penalty: 0.0,
            ..cfg.clone()
        };
        assert_eq!(
            repetition_adjustments(&[4.0; 10], &[1], &both),
            vec![(1, 0.5)]
        );
        // All off: nothing is touched.
        let off = SamplingConfig {
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            ..cfg
        };
        assert!(repetition_adjustments(&[5.0; 10], &recent, &off).is_empty());
    }

    #[test]
    fn the_seed_travels_as_text_and_restarts_the_generator() {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct Holder {
            #[serde(with = "seed_text", default = "seed_text::default_seed")]
            seed: u64,
        }
        let holder: Holder = serde_json::from_str("{}").unwrap();
        assert_eq!(holder.seed, DEFAULT_SAMPLER_SEED);
        assert_eq!(
            serde_json::to_string(&holder).unwrap(),
            r#"{"seed":"9600629759793949339"}"#
        );
        let text: Holder = serde_json::from_str(r#"{"seed":" 42 "}"#).unwrap();
        let number: Holder = serde_json::from_str(r#"{"seed":42}"#).unwrap();
        assert_eq!((text.seed, number.seed), (42, 42));
        assert!(serde_json::from_str::<Holder>(r#"{"seed":"-1"}"#).is_err());
        assert!(serde_json::from_str::<Holder>(r#"{"seed":"abc"}"#).is_err());

        let _guard = SAMPLER_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        set_sampler_seed(42);
        let first = next_sampler_rng_value(&mut SAMPLER_RNG.lock().unwrap());
        set_sampler_seed(42);
        let again = next_sampler_rng_value(&mut SAMPLER_RNG.lock().unwrap());
        set_sampler_seed(DEFAULT_SAMPLER_SEED);
        assert_eq!(first, again);
    }

    #[test]
    fn the_reasoning_budget_forces_the_end_of_a_thinking_block() {
        // 100 opens, 200 201 closes (a tag of two tokens).
        let budget = |tokens| ReasoningBudget {
            tokens,
            start: vec![100],
            end: vec![200, 201],
        };
        // Off, or the block still has room.
        assert_eq!(
            forced_reasoning_token(&[100, 1, 2, 3], &budget(-1), false),
            None
        );
        assert_eq!(
            forced_reasoning_token(&[100, 1, 2], &budget(3), false),
            None
        );
        // The budget is used up: the end tag comes out token by token.
        assert_eq!(
            forced_reasoning_token(&[100, 1, 2, 3], &budget(3), false),
            Some(200)
        );
        assert_eq!(
            forced_reasoning_token(&[100, 1, 2, 3, 200], &budget(3), false),
            Some(201)
        );
        // Closed: the answer is free, however long.
        assert_eq!(
            forced_reasoning_token(&[100, 1, 2, 3, 200, 201, 7, 8, 9, 9], &budget(3), false),
            None
        );
        // The prompt opened the block: nothing in the output says so.
        assert_eq!(
            forced_reasoning_token(&[1, 2, 3], &budget(3), true),
            Some(200)
        );
        assert_eq!(forced_reasoning_token(&[1, 2, 3], &budget(3), false), None);
        assert_eq!(
            forced_reasoning_token(&[1, 200, 201, 5, 6, 7, 8], &budget(3), true),
            None
        );
        // Zero closes the block at once.
        assert_eq!(forced_reasoning_token(&[], &budget(0), true), Some(200));
        // A second block in the same answer gets the same budget.
        assert_eq!(
            forced_reasoning_token(&[100, 1, 200, 201, 100, 4, 5, 6], &budget(3), false),
            Some(200)
        );
        // No known end tag: nothing to force.
        let no_tags = ReasoningBudget {
            tokens: 0,
            ..ReasoningBudget::default()
        };
        assert_eq!(forced_reasoning_token(&[1, 2], &no_tags, true), None);

        // The sampler returns the forced token instead of the likeliest one.
        let mut cfg = dry(0.0, &[]);
        cfg.temperature = 0.0;
        cfg.reasoning = budget(1);
        let logits = [9.0_f32, 0.0, 0.0];
        assert_eq!(
            sample_token_within_budget(&logits, &[100, 5], &cfg, false).0,
            200
        );
        assert_eq!(
            sample_token_within_budget(&logits, &[100], &cfg, false).0,
            0
        );
    }

    fn dry(multiplier: f32, breakers: &[i32]) -> SamplingConfig {
        SamplingConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            min_p: 0.0,
            dry: DrySampling {
                multiplier,
                breakers: breakers.to_vec(),
                ..DrySampling::default()
            },
            reasoning: Default::default(),
        }
    }

    #[test]
    fn dry_penalizes_the_token_that_would_extend_a_repeated_run() {
        // "1 2 3 4" appeared, and the text is now at "... 1 2 3": token 4
        // would extend a run of three.
        let recent = [1, 2, 3, 4, 9, 1, 2, 3];
        let penalties = dry_penalties(&recent, &dry(0.8, &[]).dry);
        assert_eq!(penalties.len(), 1);
        // multiplier * base^(match - allowed_length) = 0.8 * 1.75^(3 - 2)
        assert!((penalties[&4] - 0.8 * 1.75).abs() < 1e-6, "{penalties:?}");

        // A run shorter than the allowed length is free.
        assert!(dry_penalties(&[1, 4, 9, 1], &dry(0.8, &[]).dry).is_empty());
        // Exactly the allowed length costs the multiplier.
        let two = dry_penalties(&[1, 2, 4, 9, 1, 2], &dry(0.8, &[]).dry);
        assert!((two[&4] - 0.8).abs() < 1e-6);
        // Off, or nothing to compare with.
        assert!(dry_penalties(&recent, &dry(0.0, &[]).dry).is_empty());
        assert!(dry_penalties(&[1], &dry(0.8, &[]).dry).is_empty());
    }

    #[test]
    fn dry_does_not_carry_a_run_across_a_sequence_breaker() {
        // 7 is a breaker (a newline, say). The run "2 3" after it repeats,
        // but "1 7 2 3" may not count as a run of four.
        let recent = [1, 7, 2, 3, 4, 5, 1, 7, 2, 3];
        let with_breaker = dry_penalties(&recent, &dry(1.0, &[7]).dry);
        assert!((with_breaker[&4] - 1.0).abs() < 1e-6, "{with_breaker:?}");
        let without = dry_penalties(&recent, &dry(1.0, &[]).dry);
        assert!((without[&4] - 1.75_f32.powi(2)).abs() < 1e-5, "{without:?}");
        // Right after a breaker there is no run yet.
        assert!(dry_penalties(&[2, 3, 7, 2, 3, 7], &dry(1.0, &[7]).dry).is_empty());
        // Only the last tokens are searched when a window is set.
        let mut windowed = dry(1.0, &[]);
        windowed.dry.penalty_last_n = 3;
        assert!(dry_penalties(&[1, 2, 3, 4, 9, 1, 2, 3], &windowed.dry).is_empty());
    }

    #[test]
    fn dry_lowers_the_logit_the_sampler_sees() {
        let cfg = dry(0.8, &[]);
        let recent = [1, 2, 3, 4, 9, 1, 2, 3];
        let adjustments = repetition_adjustments(&[5.0; 10], &recent, &cfg);
        assert_eq!(adjustments.len(), 1);
        assert_eq!(adjustments[0].0, 4);
        assert!((adjustments[0].1 - (5.0 - 0.8 * 1.75)).abs() < 1e-5);
        // Together with the window penalties, on the same token.
        let both = SamplingConfig {
            presence_penalty: 1.0,
            repeat_last_n: 8,
            ..cfg
        };
        let token_4 = repetition_adjustments(&[5.0; 10], &recent, &both)
            .into_iter()
            .find(|(id, _)| *id == 4)
            .unwrap();
        assert!((token_4.1 - (5.0 - 1.0 - 0.8 * 1.75)).abs() < 1e-5);
    }

    #[test]
    fn min_p_drops_candidates_far_below_the_most_likely_one() {
        let cfg = SamplingConfig {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            min_p: 0.2,
            dry: Default::default(),
            reasoning: Default::default(),
        };
        // Probabilities about 0.705, 0.259, 0.035: the last is under 20% of
        // the first.
        let logits = [3.0_f32, 2.0, 0.0];
        let kept = sampling_candidates(&logits, &[], &cfg);
        assert_eq!(kept.iter().map(|(id, _)| *id).collect::<Vec<_>>(), [0, 1]);
        let total: f32 = kept.iter().map(|(_, probability)| probability).sum();
        assert!((total - 1.0).abs() < 1e-5);
        let off = SamplingConfig { min_p: 0.0, ..cfg };
        assert_eq!(sampling_candidates(&logits, &[], &off).len(), 3);
    }

    #[test]
    fn repetition_penalty_is_applied_before_top_k() {
        let cfg = SamplingConfig {
            repeat_penalty: 2.0,
            repeat_last_n: 4,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            min_p: 0.0,
            dry: Default::default(),
            reasoning: Default::default(),
            ..cfg(1.0, 2, 1.0)
        };

        assert_eq!(candidate_ids(&[10.0, 9.0, 8.0, 7.0], &[0, 0], &cfg), [1, 2]);
    }

    #[test]
    fn repetition_penalty_below_one_can_raise_a_token_into_top_k() {
        let cfg = SamplingConfig {
            repeat_penalty: 0.5,
            repeat_last_n: 4,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            min_p: 0.0,
            dry: Default::default(),
            reasoning: Default::default(),
            ..cfg(1.0, 2, 1.0)
        };

        assert_eq!(
            candidate_ids(&[9.0, 8.0, 7.0, 6.0, 5.0], &[4], &cfg),
            [0, 4]
        );
    }

    #[test]
    fn disabled_top_k_keeps_the_full_distribution_for_top_p() {
        assert_eq!(
            candidate_ids(&[3.0, 2.0, 1.0], &[], &cfg(1.0, 0, 0.8)),
            [0, 1]
        );
    }

    #[test]
    fn top_k_and_top_p_ties_keep_the_lowest_token_ids() {
        assert_eq!(
            candidate_ids(&[7.0, 7.0, 7.0, 1.0], &[], &cfg(1.0, 1, 1.0)),
            [0]
        );
        assert_eq!(candidate_ids(&[1.0; 4], &[], &cfg(1.0, 0, 0.5)), [0, 1]);
    }
}

#[cfg(test)]
pub(super) fn normalise_candidates(candidates: &mut Vec<(usize, f32)>, adjusted: &[f32]) {
    normalise_candidates_with_fallback(candidates, || argmax(adjusted) as usize);
}

fn normalise_candidates_with_fallback(
    candidates: &mut Vec<(usize, f32)>,
    fallback: impl FnOnce() -> usize,
) {
    let total = candidates
        .iter()
        .map(|(_, probability)| *probability)
        .sum::<f32>();
    if total > 0.0 {
        for (_, probability) in candidates {
            *probability /= total;
        }
    } else {
        candidates.clear();
        candidates.push((fallback(), 1.0));
    }
}

/// Index of the largest value, matching `max_by`'s "last one wins" on ties.
pub(super) fn argmax(values: &[f32]) -> i32 {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

/// Atalho sem a decomposição de tempos. A produção chama
/// `sample_token_profiled` direto, porque alimenta a telemetria por hop; este
/// embrulho existe para os testes que só querem o token.
#[cfg(test)]
pub fn sample_token(logits: &[f32], recent: &[i32], cfg: &SamplingConfig) -> i32 {
    sample_token_profiled(logits, recent, cfg).0
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub(super) fn template_marker<'a>(template: &str, marker: &'a str) -> Result<&'a str> {
    if template.contains(marker) {
        Ok(marker)
    } else {
        bail!("GGUF Gemma4 chat_template não contém o marcador {marker:?}")
    }
}

/// Render the Gemma4 template shape stored in the GGUF.
///
/// llama.cpp's public C helper accepts the Jinja source but has no argument
/// for `enable_thinking`. This renderer only supports the exact control-flow
/// markers it finds in the model's own template and refuses a drifted
/// template instead of substituting a Qwen-like prompt.
pub(super) fn render_gemma4_template(
    template: &str,
    bos: &str,
    messages: &[(&str, &str)],
    add_generation_prompt: bool,
    no_thinking: bool,
) -> Result<String> {
    let turn = template_marker(template, "<|turn>")?;
    let turn_end = template_marker(template, "<turn|>")?;
    let think = template_marker(template, "<|think|>")?;
    let channel = template_marker(template, "<|channel>")?;
    let channel_end = template_marker(template, "<channel|>")?;
    let enable_thinking = !no_thinking;

    let mut out = String::from(bos);
    let mut loop_messages = messages;
    let has_system_block = enable_thinking
        || messages
            .first()
            .is_some_and(|(role, _)| *role == "system" || *role == "developer");
    if has_system_block {
        out.push_str(turn);
        out.push_str("system\n");
        if enable_thinking {
            out.push_str(think);
            out.push('\n');
        }
        if let Some((role, content)) = messages.first() {
            if *role == "system" || *role == "developer" {
                out.push_str(content.trim());
                loop_messages = &messages[1..];
            }
        }
        out.push_str(turn_end);
        out.push('\n');
    }

    for (role, content) in loop_messages {
        let role = if *role == "assistant" { "model" } else { role };
        out.push_str(turn);
        out.push_str(role);
        out.push('\n');
        out.push_str(content.trim());
        out.push_str(turn_end);
        out.push('\n');
    }

    if add_generation_prompt {
        out.push_str(turn);
        out.push_str("model\n");
        if !enable_thinking {
            out.push_str(channel);
            out.push_str("thought\n");
            out.push_str(channel_end);
        }
    }
    Ok(out)
}

pub(super) fn token_sequence_ends_with(tokens: &[i32], suffix: &[i32]) -> bool {
    !suffix.is_empty() && tokens.len() >= suffix.len() && tokens.ends_with(suffix)
}

#[cfg(test)]
mod probe_tests {
    use super::*;

    #[test]
    fn candidates_get_their_share_of_the_full_distribution() {
        // ln(4) vs ln(1): ids 0 and 1 hold 4/7 and 1/7 of the mass.
        let logits = [4f32.ln(), 1f32.ln(), 2f32.ln()];
        let result = probe_candidates(&logits, &[vec![0], vec![1]]);
        assert!((result.probabilities[0] - 4.0 / 7.0).abs() < 1e-5);
        assert!((result.probabilities[1] - 1.0 / 7.0).abs() < 1e-5);
        assert!((result.mass - 5.0 / 7.0).abs() < 1e-5);
    }

    #[test]
    fn spelling_variants_add_up_and_an_id_counts_once() {
        let logits = [0.0f32, 0.0, 0.0, 0.0];
        let result = probe_candidates(&logits, &[vec![0, 1], vec![1, 2], vec![99, -1]]);
        assert_eq!(result.probabilities, vec![0.5, 0.25, 0.0]);
        assert!(probe_candidates(&[], &[vec![0]]).probabilities == vec![0.0]);
    }
}
