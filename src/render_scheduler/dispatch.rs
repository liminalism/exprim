//! Compiled-page hand-off between the compile and render workers, with the
//! byte bound on the reorder backlog.
//!
//! The page-count window (`SchedulerOptions::reorder_window_depth`) bounds how
//! many pages can wait in the reorder buffer, but not how large they are: a
//! slow early page let every other admitted page finish and wait, which at
//! 300 dpi was gigabytes of finished surfaces while the memory budget, which
//! only covers renders in progress, stood mostly idle. This stage adds the
//! byte bound.
//!
//! # The rule
//!
//! A render worker takes the lowest-numbered compiled page. It may start it
//! when any of these holds:
//!
//! - it is the *head*, the page the collector emits next;
//! - no finished page is waiting (`backlog == 0`);
//! - the finished-but-unemitted bytes plus this page's surface fit the limit;
//! - it carries no surface (a compile error), or the pipeline is cancelled.
//!
//! Otherwise the worker waits until an emission frees bytes or advances the
//! head. Only finished pages count, not pages being rendered: those already
//! hold a [`super::MemoryBudget`] permit, and counting them as well would cap
//! render parallelism on large pages even when nothing is out of order.
//! The backlog therefore stays below the limit plus the surfaces of the pages
//! that were already rendering when it was reached, at most one per render
//! worker.
//!
//! # Why it cannot deadlock
//!
//! The head always progresses. Pages reach the compile workers in order from
//! a FIFO channel, so a compile worker can only be holding pages at or after
//! the head; the head itself is inserted even when the ready set is full; it
//! is then the lowest entry and is always takeable; and its render only waits
//! on the memory budget, whose other holders are renders that finish. Emitting
//! the head frees bytes and moves the head on. Waiting happens only here, on
//! a worker that holds no page, never on a worker that holds one.

use std::collections::BTreeMap;
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

/// How long a waiting worker sleeps before re-testing the pipeline's
/// cancellation token, which does not notify the condvar.
const CANCEL_POLL: Duration = Duration::from_millis(25);

/// One compiled page waiting for a render worker.
#[derive(Debug)]
pub(super) struct Ready<T> {
    pub page: u32,
    pub item: T,
    /// Bytes of the output surface this page will hold once rendered.
    pub surface_bytes: u64,
}

#[derive(Debug)]
struct State<T> {
    /// Compiled pages by page number; the lowest is dispatched first.
    ready: BTreeMap<u32, (T, u64)>,
    /// Sequence number of the next page the collector emits.
    head: u64,
    /// Bytes of finished surfaces not yet emitted.
    backlog: u64,
    /// Compile workers still running. When none is left and `ready` is empty,
    /// render workers stop.
    compilers: usize,
    /// Render workers still running. When none is left, compile workers stop
    /// instead of waiting for room that will never come.
    renderers: usize,
    /// Set when the collector stops (normally, or by unwinding out of the
    /// caller's `emit`). Every waiter gives up.
    closed: bool,
}

/// Shared between the compile workers, the render workers and the collector.
#[derive(Debug)]
pub(super) struct Dispatch<T> {
    state: Mutex<State<T>>,
    changed: Condvar,
    /// Compiled pages that may wait for a render worker, the head excepted.
    capacity: usize,
    /// Finished-but-unemitted bytes beyond which no out-of-order page starts.
    backlog_limit: u64,
}

impl<T> Dispatch<T> {
    pub(super) fn new(
        first_page: u64,
        capacity: usize,
        backlog_limit: u64,
        compilers: usize,
        renderers: usize,
    ) -> Self {
        Self {
            state: Mutex::new(State {
                ready: BTreeMap::new(),
                head: first_page,
                backlog: 0,
                compilers,
                renderers,
                closed: false,
            }),
            changed: Condvar::new(),
            capacity: capacity.max(1),
            backlog_limit,
        }
    }

    /// Lock with poison recovery: every mutation below is a plain insert,
    /// remove or saturating arithmetic, none of which can panic halfway, so
    /// the state is valid even after another thread panicked holding it.
    fn lock(&self) -> MutexGuard<'_, State<T>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn wait<'a>(&self, guard: MutexGuard<'a, State<T>>, poll: bool) -> MutexGuard<'a, State<T>> {
        if poll {
            self.changed
                .wait_timeout(guard, CANCEL_POLL)
                .map(|(guard, _)| guard)
                .unwrap_or_else(|poisoned| poisoned.into_inner().0)
        } else {
            self.changed
                .wait(guard)
                .unwrap_or_else(PoisonError::into_inner)
        }
    }

    /// Hand a compiled page to the render workers. Blocks while the ready set
    /// is full, except for the head page, which is always accepted. Returns
    /// `false` when no render worker is left to take it (the pipeline is
    /// shutting down) or the collector has stopped.
    pub(super) fn push(&self, page: u32, item: T, surface_bytes: u64) -> bool {
        let mut state = self.lock();
        loop {
            if state.closed || state.renderers == 0 {
                return false;
            }
            if state.ready.len() < self.capacity || u64::from(page) == state.head {
                break;
            }
            state = self.wait(state, false);
        }
        state.ready.insert(page, (item, surface_bytes));
        drop(state);
        self.changed.notify_all();
        true
    }

    /// The next page a render worker may start, waiting until one is
    /// admissible (see the module documentation). `None` once every compile
    /// worker has finished and nothing is left, or the collector stopped.
    ///
    /// `poll` makes the wait re-test `cancelled` periodically; pass it when
    /// the pipeline has a cancellation token.
    pub(super) fn take(&self, cancelled: &dyn Fn() -> bool, poll: bool) -> Option<Ready<T>> {
        let mut state = self.lock();
        loop {
            if state.closed {
                return None;
            }
            if let Some((&page, &(_, surface_bytes))) = state.ready.first_key_value() {
                let admissible = surface_bytes == 0
                    || u64::from(page) == state.head
                    || state.backlog == 0
                    || state.backlog.saturating_add(surface_bytes) <= self.backlog_limit
                    || cancelled();
                if admissible {
                    let (item, surface_bytes) = state.ready.remove(&page)?;
                    drop(state);
                    // A slot freed: a compile worker may be waiting for it.
                    self.changed.notify_all();
                    return Some(Ready {
                        page,
                        item,
                        surface_bytes,
                    });
                }
            } else if state.compilers == 0 {
                return None;
            }
            state = self.wait(state, poll);
        }
    }

    /// A render finished holding `surface_bytes` (zero for a failed page).
    /// Growth never unblocks anyone, so there is nothing to notify.
    pub(super) fn finished(&self, surface_bytes: u64) {
        let mut state = self.lock();
        state.backlog = state.backlog.saturating_add(surface_bytes);
    }

    /// The collector emitted page `next_head - 1`, which held `surface_bytes`.
    pub(super) fn emitted(&self, next_head: u64, surface_bytes: u64) {
        let mut state = self.lock();
        state.backlog = state.backlog.saturating_sub(surface_bytes);
        state.head = state.head.max(next_head);
        drop(state);
        self.changed.notify_all();
    }

    /// Finished-but-unemitted bytes, for tests.
    #[cfg(test)]
    pub(super) fn backlog(&self) -> u64 {
        self.lock().backlog
    }

    fn worker_exited(&self, compiler: bool) {
        let mut state = self.lock();
        if compiler {
            state.compilers = state.compilers.saturating_sub(1);
        } else {
            state.renderers = state.renderers.saturating_sub(1);
        }
        drop(state);
        self.changed.notify_all();
    }

    fn close(&self) {
        self.lock().closed = true;
        self.changed.notify_all();
    }

    /// Held by a compile worker for its lifetime; dropping it (also by
    /// unwinding) lets the render workers stop once the ready set drains.
    pub(super) fn compiler(&self) -> ExitGuard<'_, T> {
        ExitGuard {
            dispatch: self,
            kind: GuardKind::Compiler,
        }
    }

    /// Held by a render worker for its lifetime.
    pub(super) fn renderer(&self) -> ExitGuard<'_, T> {
        ExitGuard {
            dispatch: self,
            kind: GuardKind::Renderer,
        }
    }

    /// Held by the collector; dropping it releases every waiter, so a panic
    /// in the caller's `emit` unwinds the pipeline instead of hanging it.
    pub(super) fn collector(&self) -> ExitGuard<'_, T> {
        ExitGuard {
            dispatch: self,
            kind: GuardKind::Collector,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum GuardKind {
    Compiler,
    Renderer,
    Collector,
}

/// Records a worker's exit on drop. See [`Dispatch::compiler`].
#[derive(Debug)]
pub(super) struct ExitGuard<'a, T> {
    dispatch: &'a Dispatch<T>,
    kind: GuardKind,
}

impl<T> Drop for ExitGuard<'_, T> {
    fn drop(&mut self) {
        match self.kind {
            GuardKind::Compiler => self.dispatch.worker_exited(true),
            GuardKind::Renderer => self.dispatch.worker_exited(false),
            GuardKind::Collector => self.dispatch.close(),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;

    const NEVER: &dyn Fn() -> bool = &|| false;

    #[test]
    fn head_is_taken_even_when_the_backlog_is_over_the_limit() {
        let d = Dispatch::<&str>::new(0, 4, 10, 1, 1);
        d.finished(100); // e.g. page 1 finished early and waits
        assert!(d.push(2, "two", 8));
        assert!(d.push(0, "zero", 8));
        // Page 0 is the head and is lowest, so it goes first despite the
        // backlog being ten times the limit.
        assert_eq!(d.take(NEVER, false).unwrap().page, 0);
    }

    #[test]
    fn out_of_order_pages_wait_for_backlog_room() {
        let d = std::sync::Arc::new(Dispatch::<u32>::new(0, 4, 10, 1, 1));
        d.finished(8);
        assert!(d.push(3, 3, 8)); // 8 + 8 > 10 and page 3 is not the head
        let taker = {
            let d = d.clone();
            std::thread::spawn(move || d.take(NEVER, false).map(|r| r.page))
        };
        std::thread::sleep(Duration::from_millis(30));
        assert!(
            !taker.is_finished(),
            "page 3 must wait while the backlog is full"
        );
        // Emitting the waiting page frees its bytes; page 3 now fits.
        d.emitted(1, 8);
        assert_eq!(taker.join().unwrap(), Some(3));
        assert_eq!(d.backlog(), 0);
    }

    #[test]
    fn an_empty_backlog_admits_a_page_larger_than_the_limit() {
        let d = Dispatch::<()>::new(0, 4, 10, 1, 1);
        assert!(d.push(5, (), 1_000));
        assert_eq!(d.take(NEVER, false).unwrap().page, 5);
    }

    #[test]
    fn pages_without_a_surface_and_cancelled_pipelines_are_never_held() {
        let d = Dispatch::<()>::new(0, 4, 10, 1, 1);
        d.finished(100);
        assert!(d.push(7, (), 0));
        assert_eq!(d.take(NEVER, false).unwrap().page, 7);
        assert!(d.push(8, (), 50));
        assert_eq!(d.take(&|| true, true).unwrap().page, 8);
    }

    #[test]
    fn the_head_is_accepted_into_a_full_ready_set() {
        let d = Dispatch::<()>::new(0, 1, u64::MAX, 1, 1);
        assert!(d.push(1, (), 1));
        // Capacity one is used up by page 1, but page 0 is the head.
        assert!(d.push(0, (), 1));
    }

    #[test]
    fn workers_stop_when_their_counterparts_are_gone() {
        let d = Dispatch::<()>::new(0, 1, u64::MAX, 1, 1);
        drop(d.compiler());
        assert!(
            d.take(NEVER, false).is_none(),
            "no compiler and nothing ready"
        );
        drop(d.renderer());
        assert!(!d.push(3, (), 1), "nobody left to render it");
    }

    #[test]
    fn a_stopped_collector_releases_waiters() {
        let d = std::sync::Arc::new(Dispatch::<()>::new(0, 4, 0, 1, 1));
        d.finished(1);
        assert!(d.push(2, (), 1));
        let taker = {
            let d = d.clone();
            std::thread::spawn(move || d.take(NEVER, false).is_none())
        };
        std::thread::sleep(Duration::from_millis(20));
        drop(d.collector());
        assert!(taker.join().unwrap());
    }
}
