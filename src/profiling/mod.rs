//! Feature-gated, allocation-tolerant profiling primitives for the renderer.
//!
//! This crate deliberately has no renderer dependencies and no global state.
//! Callers own one [`ProfileReport`] per operation or worker, then merge the
//! reports at a synchronization point.  Renderer crates depend on it only
//! through their `profiling` feature, so production builds do not execute
//! timers or counter updates.

use std::collections::BTreeMap;
use std::time::Duration;

/// A stable, machine-readable profile payload for one operation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProfileReport {
    /// Versioned independently so report readers can reject incompatible rows.
    pub schema_version: u32,
    /// Inclusive named stage durations. Repeated scopes are accumulated.
    pub durations: BTreeMap<&'static str, Duration>,
    /// Named event and work counters. Repeated updates are accumulated.
    pub counters: BTreeMap<&'static str, u64>,
    /// Renderer-owned bytes live when the operation ended.
    pub live_bytes: u64,
    /// Maximum renderer-owned bytes observed during the operation.
    pub peak_live_bytes: u64,
}

impl ProfileReport {
    pub const SCHEMA_VERSION: u32 = 1;

    pub fn new() -> Self {
        Self {
            schema_version: Self::SCHEMA_VERSION,
            ..Self::default()
        }
    }

    pub fn add_duration(&mut self, name: &'static str, duration: Duration) {
        *self.durations.entry(name).or_default() += duration;
    }

    pub fn increment(&mut self, name: &'static str, amount: u64) {
        *self.counters.entry(name).or_default() += amount;
    }

    /// Account for a renderer-owned allocation. `release_bytes` must be
    /// called when it ceases to be live; this intentionally models only
    /// buffers the renderer can name, not the process allocator as a whole.
    pub fn allocate_bytes(&mut self, bytes: u64) {
        self.live_bytes = self.live_bytes.saturating_add(bytes);
        self.peak_live_bytes = self.peak_live_bytes.max(self.live_bytes);
    }

    pub fn release_bytes(&mut self, bytes: u64) {
        self.live_bytes = self.live_bytes.saturating_sub(bytes);
    }

    pub fn merge(&mut self, other: &Self) {
        self.schema_version = self.schema_version.max(other.schema_version);
        for (&name, &duration) in &other.durations {
            self.add_duration(name, duration);
        }
        for (&name, &count) in &other.counters {
            self.increment(name, count);
        }
        self.peak_live_bytes = self.peak_live_bytes.max(other.peak_live_bytes);
        self.live_bytes = self.live_bytes.saturating_add(other.live_bytes);
    }

    /// Render the report's flattened metrics as JSON object fields. Kept here
    /// to avoid making serde a production or toolchain dependency.
    pub fn write_json_fields(&self, out: &mut String) {
        use std::fmt::Write as _;
        let _ = write!(out, "\"profile_schema\":{}", self.schema_version);
        for (&name, duration) in &self.durations {
            let _ = write!(out, ",\"{}\":{}", name, duration.as_nanos());
        }
        for (&name, count) in &self.counters {
            let _ = write!(out, ",\"{}\":{}", name, count);
        }
        let _ = write!(
            out,
            ",\"live_bytes\":{},\"peak_live_bytes\":{}",
            self.live_bytes, self.peak_live_bytes
        );
    }
}

/// A monotonic lap timer for exclusive stage attribution: each [`Lap::lap_ns`]
/// returns the time since the previous lap (or the start) with one clock read,
/// so consecutive buckets cannot double count and their sum is the stage's
/// wall time. Hot paths add the result to a plain integer field and flush to a
/// [`ProfileReport`] once per page, keeping map updates out of per-fill loops.
#[derive(Debug, Clone, Copy)]
pub struct Lap {
    last: std::time::Instant,
}

impl Lap {
    #[inline]
    pub fn start() -> Self {
        Self {
            last: std::time::Instant::now(),
        }
    }

    /// Nanoseconds since the previous lap; restarts the lap.
    #[inline]
    pub fn lap_ns(&mut self) -> u64 {
        let now = std::time::Instant::now();
        let ns = now.duration_since(self.last).as_nanos() as u64;
        self.last = now;
        ns
    }
}

/// Lap an optional timer; an absent timer (detail level off) costs nothing
/// and reports zero.
#[inline]
pub fn lap_opt(lap: &mut Option<Lap>) -> u64 {
    match lap.as_mut() {
        Some(lap) => lap.lap_ns(),
        None => 0,
    }
}

/// How much timing a profiled render collects. Counters are always recorded;
/// the levels differ only in clock reads, which on a page of very many tiny
/// fills are a visible share of the work being measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProfileDetail {
    /// Stage and per-operation timers only (page-level draws, tiling fills).
    #[default]
    Stage,
    /// Also every kernel fill's phases and every nested tiling-cell draw.
    Fill,
    /// Also every coverage-row callback (span dispatch and compositing).
    Row,
}

/// Measured cost of one clock read in nanoseconds (median of a few batches),
/// so a report can state the instrumentation's own share: reads × this.
pub fn clock_read_cost_ns() -> u64 {
    const BATCH: u32 = 2_000;
    let mut samples = [0u64; 9];
    for sample in &mut samples {
        let start = std::time::Instant::now();
        let mut lap = Lap::start();
        let mut sink = 0u64;
        for _ in 0..BATCH {
            sink = sink.wrapping_add(lap.lap_ns());
        }
        std::hint::black_box(sink);
        *sample = start.elapsed().as_nanos() as u64 / u64::from(BATCH);
    }
    samples.sort_unstable();
    samples[samples.len() / 2]
}

impl From<()> for ProfileReport {
    fn from(_: ()) -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;

    #[test]
    fn accumulates_and_merges() {
        let mut a = ProfileReport::new();
        a.add_duration("render.execute", Duration::from_nanos(3));
        a.increment("image.draws", 2);
        a.allocate_bytes(10);
        a.release_bytes(4);
        let mut b = ProfileReport::new();
        b.add_duration("render.execute", Duration::from_nanos(2));
        b.increment("image.draws", 1);
        b.allocate_bytes(20);
        a.merge(&b);
        assert_eq!(a.durations["render.execute"], Duration::from_nanos(5));
        assert_eq!(a.counters["image.draws"], 3);
        assert_eq!(a.peak_live_bytes, 20);
        assert_eq!(a.schema_version, ProfileReport::SCHEMA_VERSION);
    }
}
