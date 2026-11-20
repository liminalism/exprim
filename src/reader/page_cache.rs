//! Bounded retention and single-flight compilation, independent of raster workers.

use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

use crate::reader::{CancellationToken, ReadError};

#[derive(Debug)]
pub(crate) struct PageCache<T> {
    capacity: usize,
    /// Bytes the retained entries may weigh together. The most recent entry
    /// is kept whatever it weighs, so rendering one heavy page repeatedly
    /// does not recompile it each time.
    byte_budget: usize,
    weigh: fn(&T) -> usize,
    state: Mutex<State<T>>,
    ready: Condvar,
}

#[derive(Debug)]
struct State<T> {
    /// `(page, value, weight)`, least recently used first.
    pages: VecDeque<(u32, Arc<T>, usize)>,
    bytes: usize,
    compiling: HashSet<u32>,
    generation: u64,
}

impl<T> PageCache<T> {
    /// A cache bounded by entry count alone.
    #[cfg(test)]
    pub(crate) fn new(capacity: usize) -> Self {
        Self::with_byte_budget(capacity, usize::MAX, |_| 0)
    }

    /// A cache bounded by entry count and by the entries' combined weight. A
    /// count alone let eight compiled pages of full-resolution image samples
    /// (26 MB each for 300 dpi colour scans) sit behind a bound that looked
    /// small.
    pub(crate) fn with_byte_budget(
        capacity: usize,
        byte_budget: usize,
        weigh: fn(&T) -> usize,
    ) -> Self {
        Self {
            capacity,
            byte_budget,
            weigh,
            state: Mutex::new(State {
                pages: VecDeque::new(),
                bytes: 0,
                compiling: HashSet::new(),
                generation: 0,
            }),
            ready: Condvar::new(),
        }
    }

    /// Combined weight of the retained entries.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.lock().bytes
    }

    fn lock(&self) -> MutexGuard<'_, State<T>> {
        // A failed compilation must not poison future requests for the document.
        self.state.lock().unwrap_or_else(|error| error.into_inner())
    }

    pub(crate) fn clear(&self) {
        let mut state = self.lock();
        state.pages.clear();
        state.bytes = 0;
        // A compilation that predates this eviction may finish for its caller,
        // but must not refill a cache just cleared for memory pressure.
        state.generation = state.generation.wrapping_add(1);
    }

    pub(crate) fn get_or_compile(
        &self,
        page: u32,
        cancellation: Option<&CancellationToken>,
        compile: impl FnOnce() -> Result<Arc<T>, ReadError>,
    ) -> Result<Arc<T>, ReadError> {
        let generation = {
            let mut state = self.lock();
            loop {
                check_cancelled(page, cancellation)?;
                if let Some(index) = state.pages.iter().position(|(key, _, _)| *key == page)
                    && let Some(entry) = state.pages.remove(index)
                {
                    let result = Arc::clone(&entry.1);
                    state.pages.push_back(entry);
                    return Ok(result);
                }
                if state.compiling.insert(page) {
                    break state.generation;
                }
                // Wait without holding up unrelated pages. A waiting consumer
                // can cancel independently of the consumer doing the work.
                let (next, _) = self
                    .ready
                    .wait_timeout(state, Duration::from_millis(20))
                    .unwrap_or_else(|error| error.into_inner());
                state = next;
            }
        };
        // Also releases waiters on an error or unwind; failures are never cached.
        let _flight = Flight { cache: self, page };
        let result = compile()?;
        let mut state = self.lock();
        check_cancelled(page, cancellation)?;
        if self.capacity > 0 && generation == state.generation {
            let weight = (self.weigh)(&result);
            while !state.pages.is_empty()
                && (state.pages.len() >= self.capacity
                    || state.bytes.saturating_add(weight) > self.byte_budget)
            {
                if let Some((_, _, evicted)) = state.pages.pop_front() {
                    state.bytes = state.bytes.saturating_sub(evicted);
                }
            }
            state.bytes = state.bytes.saturating_add(weight);
            state.pages.push_back((page, Arc::clone(&result), weight));
        }
        Ok(result)
    }
}

struct Flight<'a, T> {
    cache: &'a PageCache<T>,
    page: u32,
}

impl<T> Drop for Flight<'_, T> {
    fn drop(&mut self) {
        self.cache.lock().compiling.remove(&self.page);
        self.cache.ready.notify_all();
    }
}

fn check_cancelled(page: u32, cancellation: Option<&CancellationToken>) -> Result<(), ReadError> {
    if cancellation.is_some_and(CancellationToken::is_cancelled) {
        Err(ReadError::Compile {
            page,
            message: "page compilation cancelled".into(),
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn value(cache: &PageCache<u32>, page: u32) -> Arc<u32> {
        cache
            .get_or_compile(page, None, || Ok(Arc::new(page)))
            .unwrap()
    }

    #[test]
    fn the_byte_budget_evicts_before_the_count_does() {
        // Eight entries allowed, but only 100 bytes; each value weighs itself.
        let cache: PageCache<u32> = PageCache::with_byte_budget(8, 100, |v| *v as usize);
        let weigh = |page: u32, weight: u32| {
            cache
                .get_or_compile(page, None, || Ok(Arc::new(weight)))
                .unwrap()
        };
        let a = weigh(0, 40);
        let b = weigh(1, 40);
        assert_eq!(cache.retained_bytes(), 80);
        // 40 + 40 + 40 > 100: the least recently used goes.
        let c = weigh(2, 40);
        assert_eq!(Arc::strong_count(&a), 1);
        assert_eq!(Arc::strong_count(&b), 2);
        assert_eq!(cache.retained_bytes(), 80);
        // Heavier than the whole budget: kept alone, so it is not recompiled
        // on every use.
        let huge = weigh(3, 500);
        assert_eq!(Arc::strong_count(&b), 1);
        assert_eq!(Arc::strong_count(&c), 1);
        assert_eq!(Arc::strong_count(&huge), 2);
        assert!(Arc::ptr_eq(&huge, &weigh(3, 0)));
        cache.clear();
        assert_eq!(cache.retained_bytes(), 0);
    }

    #[test]
    fn lru_eviction_and_clear_release_cache_ownership() {
        let cache = PageCache::new(2);
        let first = value(&cache, 0);
        let second = value(&cache, 1);
        assert!(Arc::ptr_eq(&first, &value(&cache, 0)));
        let third = value(&cache, 2);
        assert_eq!(Arc::strong_count(&second), 1, "least recently used evicted");
        assert_eq!(Arc::strong_count(&first), 2);
        cache.clear();
        assert_eq!(Arc::strong_count(&first), 1);
        assert_eq!(Arc::strong_count(&third), 1);
        assert!(!Arc::ptr_eq(&first, &value(&cache, 0)));

        let disabled = PageCache::new(0);
        assert!(!Arc::ptr_eq(&value(&disabled, 0), &value(&disabled, 0)));
    }

    #[test]
    fn cancellation_and_clear_during_compilation_prevent_retention() {
        let cache = PageCache::new(2);
        let token = CancellationToken::new();
        let result = cache.get_or_compile(0, Some(&token), || {
            token.cancel();
            Ok(Arc::new(0))
        });
        assert!(result.is_err());
        assert!(cache.lock().pages.is_empty());
        assert!(cache.lock().compiling.is_empty());

        let page = cache
            .get_or_compile(0, None, || {
                cache.clear();
                Ok(Arc::new(0))
            })
            .unwrap();
        assert_eq!(Arc::strong_count(&page), 1);
        assert!(!Arc::ptr_eq(&page, &value(&cache, 0)));
    }

    #[test]
    fn failures_and_unwinds_allow_a_later_retry() {
        let cache = PageCache::<u32>::new(1);
        assert!(
            cache
                .get_or_compile(0, None, || Err(ReadError::Compile {
                    page: 0,
                    message: "test failure".into(),
                }))
                .is_err()
        );
        let result = std::panic::catch_unwind(|| {
            cache.get_or_compile(0, None, || panic!("test compiler unwind"))
        });
        assert!(result.is_err());
        assert_eq!(*value(&cache, 0), 0);
    }

    #[test]
    fn waiting_caller_can_cancel_without_cancelling_the_owner() {
        let cache = Arc::new(PageCache::new(2));
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let owner_cache = Arc::clone(&cache);
        let owner = std::thread::spawn(move || {
            owner_cache
                .get_or_compile(0, None, || {
                    started_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok(Arc::new(0))
                })
                .unwrap()
        });
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        // An unrelated page must make progress while page zero is blocked.
        assert_eq!(*value(&cache, 1), 1);
        let token = CancellationToken::new();
        let waiting_token = token.clone();
        let waiting_cache = Arc::clone(&cache);
        let (waiting_tx, waiting_rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            let result = waiting_cache.get_or_compile(0, Some(&waiting_token), || {
                panic!("waiter must not interpret the page")
            });
            waiting_tx.send(result.is_err()).unwrap();
        });
        token.cancel();
        let cancelled = waiting_rx.recv_timeout(Duration::from_secs(5));
        // Always release the owner, including on assertion failure.
        release_tx.send(()).unwrap();
        let page = owner.join().unwrap();
        waiter.join().unwrap();
        assert!(cancelled.unwrap());
        assert!(Arc::ptr_eq(&page, &value(&cache, 0)));
        assert!(
            cache
                .get_or_compile(0, Some(&token), || {
                    panic!("cancelled cache hit must not compile")
                })
                .is_err()
        );
        assert!(Arc::ptr_eq(&page, &value(&cache, 0)));
    }
}
