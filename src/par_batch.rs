//! Ordered, bounded fan-out of a batch of texts across threads, for
//! [`crate::embedgemma2_tok::Tokenizer::encode_batch_with_prompt`].
//!
//! The batch is cut into contiguous chunks balanced by byte length, one chunk
//! per thread; chunk 0 runs on the calling thread and the others on scoped
//! threads, so nothing outlives the call and a batch too small to be worth a
//! thread (see [`plan`]) never spawns one. Results are joined in chunk order,
//! so the output is the serial output. The error is the serial error too: the
//! first failing text by position, because each chunk stops at its own first
//! failure and a chunk is dropped from the search only when an earlier text
//! has already failed.
//!
//! A worker that panics becomes an `Err`, never a partial batch.

use std::sync::atomic::{AtomicUsize, Ordering};

/// The most threads one batch is ever spread over, whatever was asked for.
pub(crate) const MAX_THREADS: usize = 64;
/// The default thread count is the machine's, up to this.
pub(crate) const DEFAULT_THREAD_CAP: usize = 32;
/// A chunk smaller than this many bytes of text costs more to hand to a
/// thread (tens of microseconds to spawn and join) than the encode saves.
pub(crate) const MIN_CHUNK_BYTES: usize = 8 * 1024;

/// The environment variable that sets the thread count of a batch.
pub(crate) const THREADS_ENV: &str = "TESSL_TOK_THREADS";

/// The thread count `value` (the environment variable, if set) asks for, or
/// the default: `available` capped at [`DEFAULT_THREAD_CAP`]. A value that is
/// not an integer in `1..=`[`MAX_THREADS`] is an error, not a fallback.
pub(crate) fn resolve_threads(value: Option<&str>, available: usize) -> Result<usize, String> {
    match value {
        None => Ok(available.clamp(1, DEFAULT_THREAD_CAP)),
        Some(v) => match v.trim().parse::<usize>() {
            Ok(n) if (1..=MAX_THREADS).contains(&n) => Ok(n),
            _ => Err(format!(
                "{THREADS_ENV}={v:?}: expected an integer from 1 to {MAX_THREADS}"
            )),
        },
    }
}

/// The thread count from the environment and the machine.
pub(crate) fn configured_threads() -> Result<usize, String> {
    let value = match std::env::var(THREADS_ENV) {
        Ok(v) => Some(v),
        Err(std::env::VarError::NotPresent) => None,
        Err(e) => return Err(format!("{THREADS_ENV}: {e}")),
    };
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    resolve_threads(value.as_deref(), available)
}

/// The chunk ranges of `lens` for at most `threads` threads: contiguous,
/// covering every index once, none empty, each worth [`MIN_CHUNK_BYTES`] on
/// average. One range (the whole batch) means no thread is spawned.
pub(crate) fn plan(lens: &[usize], threads: usize) -> Vec<std::ops::Range<usize>> {
    let n = lens.len();
    let total: usize = lens.iter().sum();
    let chunks = threads.clamp(1, MAX_THREADS).min(n).min(total / MIN_CHUNK_BYTES).max(1);
    if chunks == 1 {
        return std::iter::once(0..n).collect();
    }
    let mut ranges = Vec::with_capacity(chunks);
    let mut start = 0;
    let mut remaining = total;
    for k in 0..chunks {
        let left = chunks - k;
        if left == 1 {
            ranges.push(start..n);
            break;
        }
        // Close the chunk once it holds its fair share of what is left, but
        // leave one text for every chunk still to come.
        let share = remaining.div_ceil(left);
        let mut end = start;
        let mut acc = 0;
        while end < n - (left - 1) && (end == start || acc < share) {
            acc += lens[end];
            end += 1;
        }
        ranges.push(start..end);
        remaining -= acc;
        start = end;
    }
    ranges
}

/// What one chunk's worker gave back: its ids, or the position and message of
/// its first failing text; `Err` outside is the worker itself dying.
type Joined = Result<Result<Vec<Vec<u32>>, (usize, String)>, String>;

/// `f(i, texts[i])` for every text, in order, over at most `threads` threads.
/// `f` is the whole of one text's work and carries its own error message; the
/// first failure by position is returned, as a serial loop returns it.
pub(crate) fn map_ordered<F>(texts: &[&str], threads: usize, f: &F) -> Result<Vec<Vec<u32>>, String>
where
    F: Fn(usize, &str) -> Result<Vec<u32>, String> + Sync,
{
    let lens: Vec<usize> = texts.iter().map(|t| t.len()).collect();
    let ranges = plan(&lens, threads);
    if ranges.len() == 1 {
        return texts.iter().enumerate().map(|(i, t)| f(i, t)).collect();
    }
    // The lowest index that has failed so far: a chunk past it stops, since
    // its result could not be the one returned.
    let first_failure = AtomicUsize::new(usize::MAX);
    let run = |range: std::ops::Range<usize>| -> Result<Vec<Vec<u32>>, (usize, String)> {
        let mut out = Vec::with_capacity(range.len());
        for i in range {
            if i > first_failure.load(Ordering::Relaxed) {
                break;
            }
            match f(i, texts[i]) {
                Ok(ids) => out.push(ids),
                Err(e) => {
                    first_failure.fetch_min(i, Ordering::Relaxed);
                    return Err((i, e));
                }
            }
        }
        Ok(out)
    };
    let results: Vec<Joined> = std::thread::scope(|s| {
        let handles: Vec<_> = ranges[1..]
            .iter()
            .map(|r| {
                let r = r.clone();
                let run = &run;
                std::thread::Builder::new().spawn_scoped(s, move || run(r))
            })
            .collect();
        let mut results = Vec::with_capacity(ranges.len());
        results.push(catch(|| run(ranges[0].clone())));
        for h in handles {
            results.push(match h {
                Ok(h) => h.join().map_err(panic_message),
                Err(e) => Err(format!("could not start a batch worker: {e}")),
            });
        }
        results
    });
    let mut out = Vec::with_capacity(texts.len());
    let mut failure: Option<(usize, String)> = None;
    for r in results {
        match r {
            Err(panic) => return Err(format!("a batch worker failed: {panic}")),
            Ok(Err((i, e))) => {
                if failure.as_ref().is_none_or(|(j, _)| i < *j) {
                    failure = Some((i, e));
                }
            }
            Ok(Ok(chunk)) => out.extend(chunk),
        }
    }
    match failure {
        Some((_, e)) => Err(e),
        None => {
            debug_assert_eq!(out.len(), texts.len());
            if out.len() != texts.len() {
                return Err(format!(
                    "a batch worker returned {} of {} texts",
                    out.len(),
                    texts.len()
                ));
            }
            Ok(out)
        }
    }
}

fn catch<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(panic_message)
}

fn panic_message(p: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        format!("panicked: {s}")
    } else if let Some(s) = p.downcast_ref::<String>() {
        format!("panicked: {s}")
    } else {
        "panicked".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_thread_count_is_bounded_and_a_bad_value_is_an_error() {
        assert_eq!(resolve_threads(None, 10).unwrap(), 10);
        assert_eq!(resolve_threads(None, 1000).unwrap(), DEFAULT_THREAD_CAP);
        assert_eq!(resolve_threads(None, 0).unwrap(), 1);
        assert_eq!(resolve_threads(Some("3"), 10).unwrap(), 3);
        assert_eq!(resolve_threads(Some(" 64 "), 2).unwrap(), 64);
        for bad in ["0", "65", "-1", "x", "", "2.5"] {
            let e = resolve_threads(Some(bad), 8).unwrap_err();
            assert!(e.contains(THREADS_ENV) && e.contains("1 to 64"), "{bad:?}: {e}");
        }
    }

    #[test]
    fn plans_are_contiguous_nonempty_bounded_and_balanced_by_bytes() {
        let big = MIN_CHUNK_BYTES;
        // Nothing to split, or nothing worth a thread: one range, no spawn.
        assert_eq!(plan(&[], 8), vec![0..0]);
        assert_eq!(plan(&[big * 100], 8), vec![0..1]);
        assert_eq!(plan(&[1, 2, 3], 8), vec![0..3]);
        assert_eq!(plan(&[big; 8], 1), vec![0..8]);
        // Never more chunks than threads, texts, or MIN_CHUNK_BYTES of text.
        assert_eq!(plan(&[big; 8], 3).len(), 3);
        assert_eq!(plan(&[big; 2], 8).len(), 2);
        assert_eq!(plan(&[big; 8], 1000).len(), 8);
        // Balanced by bytes, not by count: a long text stands alone.
        let mut lens = vec![big * 6];
        lens.extend([big / 2; 12]);
        assert_eq!(plan(&lens, 2), vec![0..1, 1..13]);
        // Skew with empties and a giant: every chunk non-empty and contiguous.
        for threads in 1..=9 {
            let lens = [0, 0, big * 40, 0, 1, big, 0, big * 3, 7, 0, 0, big * 2];
            let ranges = plan(&lens, threads);
            assert!(ranges.len() <= threads.max(1));
            let mut at = 0;
            for r in &ranges {
                assert_eq!(r.start, at);
                assert!(r.end > r.start, "{ranges:?}");
                at = r.end;
            }
            assert_eq!(at, lens.len());
        }
    }

    fn ids_of(i: usize, t: &str) -> Result<Vec<u32>, String> {
        Ok(vec![i as u32, t.len() as u32])
    }

    #[test]
    fn output_is_in_text_order_at_any_thread_count() {
        let texts: Vec<String> = (0..300).map(|i| "x".repeat((i % 7) * 900 + i)).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let serial = map_ordered(&refs, 1, &ids_of).unwrap();
        for threads in [2, 3, 8, MAX_THREADS, 1000] {
            assert_eq!(
                map_ordered(&refs, threads, &ids_of).unwrap(),
                serial,
                "{threads} threads"
            );
        }
        assert_eq!(map_ordered(&[], 8, &ids_of).unwrap(), Vec::<Vec<u32>>::new());
    }

    #[test]
    fn the_first_failure_by_position_is_the_one_returned() {
        let texts: Vec<String> = (0..64).map(|_| "y".repeat(MIN_CHUNK_BYTES)).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let fail_at = |bad: &'static [usize]| {
            move |i: usize, _t: &str| -> Result<Vec<u32>, String> {
                if bad.contains(&i) {
                    Err(format!("text {i}: boom"))
                } else {
                    Ok(vec![i as u32])
                }
            }
        };
        for threads in [1, 2, 5, 16] {
            assert_eq!(
                map_ordered(&refs, threads, &fail_at(&[40, 9, 63])).unwrap_err(),
                "text 9: boom"
            );
            assert_eq!(
                map_ordered(&refs, threads, &fail_at(&[63])).unwrap_err(),
                "text 63: boom"
            );
        }
    }

    #[test]
    fn a_worker_panic_is_an_error_not_a_partial_batch() {
        let texts: Vec<String> = (0..64).map(|_| "z".repeat(MIN_CHUNK_BYTES)).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        for at in [0usize, 31, 63] {
            let f = move |i: usize, _t: &str| -> Result<Vec<u32>, String> {
                assert!(i != at, "planted panic at {i}");
                Ok(vec![i as u32])
            };
            let e = map_ordered(&refs, 4, &f).unwrap_err();
            assert!(
                e.contains("a batch worker failed") && e.contains("planted panic"),
                "{at}: {e}"
            );
        }
    }

    #[test]
    fn a_small_batch_runs_on_the_calling_thread() {
        let me = std::thread::current().id();
        let f = move |_i: usize, _t: &str| -> Result<Vec<u32>, String> {
            assert_eq!(std::thread::current().id(), me, "a worker thread was started");
            Ok(vec![])
        };
        for texts in [vec![], vec!["one"], vec!["a"; 50], vec!["b"; 1000]] {
            map_ordered(&texts, 32, &f).unwrap();
        }
        let big = "c".repeat(MIN_CHUNK_BYTES * 50);
        map_ordered(&[big.as_str()], 32, &f).unwrap();
    }
}
