//! Windows process helpers for the probe: pinning to a performance core,
//! priority, per-thread cycle and CPU time, and process memory.
//!
//! Measurements on a busy machine cannot trust wall time: a preempted render
//! looks slow. Thread cycle time only advances while the thread is scheduled,
//! so it is the primary cost metric here; wall time is recorded beside it and
//! the ratio shows how contended a sample was. Declared by hand (kernel32 /
//! psapi) so the harness needs no extra dependency. Non-Windows builds get
//! inert fallbacks.

#![allow(unsafe_code)]

/// A point-in-time reading of the calling thread's cost counters.
#[derive(Debug, Clone, Copy, Default)]
pub struct ThreadCost {
    /// CPU cycles charged to this thread (0 when unavailable).
    pub cycles: u64,
    /// CPU cycles charged to the whole process, every thread included. The
    /// renderer hands some work (clip-mask prebuilds, image rows) to a rayon
    /// pool even at one worker thread, where the calling thread only waits,
    /// so this is the cost metric; the thread figure shows the difference.
    pub process_cycles: u64,
    /// Kernel + user time in 100 ns units (coarse: scheduler-tick resolution).
    pub cpu_100ns: u64,
}

/// Process memory high-water marks in bytes.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryInfo {
    pub peak_working_set: u64,
    pub working_set: u64,
    pub peak_commit: u64,
    pub page_faults: u64,
}

/// What `pin` did, for the report's meta row.
#[derive(Debug, Clone, Default)]
pub struct PinReport {
    pub requested: String,
    /// Affinity mask in force afterwards (0 when unknown).
    pub mask: u64,
    /// Logical processors by efficiency class, as `class:count` pairs.
    pub efficiency_classes: String,
    pub priority_high: bool,
    pub note: String,
}

#[cfg(windows)]
mod imp {
    use super::{MemoryInfo, PinReport, ThreadCost};
    use std::ffi::c_void;

    type Handle = *mut c_void;

    #[repr(C)]
    #[derive(Default, Clone, Copy)]
    struct FileTime {
        low: u32,
        high: u32,
    }

    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> Handle;
        fn GetCurrentThread() -> Handle;
        fn SetProcessAffinityMask(process: Handle, mask: usize) -> i32;
        fn GetProcessAffinityMask(process: Handle, process_mask: *mut usize, system_mask: *mut usize) -> i32;
        fn SetPriorityClass(process: Handle, class: u32) -> i32;
        fn QueryThreadCycleTime(thread: Handle, cycles: *mut u64) -> i32;
        fn QueryProcessCycleTime(process: Handle, cycles: *mut u64) -> i32;
        fn GetThreadTimes(
            thread: Handle,
            creation: *mut FileTime,
            exit: *mut FileTime,
            kernel: *mut FileTime,
            user: *mut FileTime,
        ) -> i32;
        fn GetSystemCpuSetInformation(
            information: *mut u8,
            buffer_length: u32,
            returned_length: *mut u32,
            process: Handle,
            flags: u32,
        ) -> i32;
        fn K32GetProcessMemoryInfo(process: Handle, counters: *mut ProcessMemoryCounters, cb: u32) -> i32;
    }

    const HIGH_PRIORITY_CLASS: u32 = 0x0000_0080;

    /// `(logical index, core index, efficiency class)` per logical processor.
    fn cpu_sets() -> Vec<(u8, u8, u8)> {
        let mut needed = 0u32;
        // SAFETY: a null buffer with zero length only reports the needed size.
        unsafe {
            GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut needed, GetCurrentProcess(), 0);
        }
        if needed == 0 {
            return Vec::new();
        }
        let mut buffer = vec![0u8; needed as usize];
        let mut returned = 0u32;
        // SAFETY: `buffer` is `needed` bytes long, as passed.
        let ok = unsafe {
            GetSystemCpuSetInformation(buffer.as_mut_ptr(), needed, &mut returned, GetCurrentProcess(), 0)
        };
        if ok == 0 {
            return Vec::new();
        }
        // SYSTEM_CPU_SET_INFORMATION: Size u32 @0, Type u32 @4, then the CpuSet
        // arm: Id u32 @8, Group u16 @12, LogicalProcessorIndex u8 @14,
        // CoreIndex u8 @15, LastLevelCacheIndex @16, NumaNodeIndex @17,
        // EfficiencyClass u8 @18.
        let mut out = Vec::new();
        let mut offset = 0usize;
        let end = (returned as usize).min(buffer.len());
        while offset + 20 <= end {
            let size = u32::from_le_bytes([
                buffer[offset],
                buffer[offset + 1],
                buffer[offset + 2],
                buffer[offset + 3],
            ]) as usize;
            if size < 20 || offset + size > end {
                break;
            }
            let kind = u32::from_le_bytes([
                buffer[offset + 4],
                buffer[offset + 5],
                buffer[offset + 6],
                buffer[offset + 7],
            ]);
            if kind == 0 {
                out.push((buffer[offset + 14], buffer[offset + 15], buffer[offset + 18]));
            }
            offset += size;
        }
        out
    }

    fn current_mask() -> u64 {
        let (mut process, mut system) = (0usize, 0usize);
        // SAFETY: both out-pointers are valid for the call.
        let ok = unsafe { GetProcessAffinityMask(GetCurrentProcess(), &mut process, &mut system) };
        if ok == 0 { 0 } else { process as u64 }
    }

    pub fn pin(spec: &str, priority_high: bool) -> PinReport {
        let sets = cpu_sets();
        let mut classes: std::collections::BTreeMap<u8, usize> = std::collections::BTreeMap::new();
        for &(_, _, class) in &sets {
            *classes.entry(class).or_default() += 1;
        }
        let mut report = PinReport {
            requested: spec.to_owned(),
            efficiency_classes: classes
                .iter()
                .map(|(class, count)| format!("{class}:{count}"))
                .collect::<Vec<_>>()
                .join(","),
            ..PinReport::default()
        };
        let mask: Option<u64> = if spec == "none" {
            None
        } else if spec == "pcore" {
            // The first hardware thread of the last core in the highest
            // efficiency class: a performance core away from CPU 0's
            // interrupt load.
            let top = sets.iter().map(|s| s.2).max();
            let pick = sets
                .iter()
                .filter(|s| Some(s.2) == top && s.0 == s.1 && s.0 < 64)
                .map(|s| s.0)
                .max();
            match pick {
                Some(index) => Some(1u64 << index),
                None => {
                    report.note = "no cpu-set information; not pinned".into();
                    None
                }
            }
        } else if let Some(n) = spec.strip_prefix("cpu:") {
            n.parse::<u32>().ok().filter(|n| *n < 64).map(|n| 1u64 << n)
        } else if let Some(hex) = spec.strip_prefix("mask:") {
            u64::from_str_radix(hex.trim_start_matches("0x"), 16).ok()
        } else {
            report.note = format!("unrecognised --pin {spec}; not pinned");
            None
        };
        if let Some(mask) = mask.filter(|m| *m != 0) {
            // SAFETY: plain value call on the current process pseudo-handle.
            let ok = unsafe { SetProcessAffinityMask(GetCurrentProcess(), mask as usize) };
            if ok == 0 {
                report.note = "SetProcessAffinityMask failed".into();
            }
        }
        if priority_high {
            // SAFETY: plain value call on the current process pseudo-handle.
            report.priority_high =
                unsafe { SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS) } != 0;
        }
        report.mask = current_mask();
        report
    }

    pub fn thread_cost() -> ThreadCost {
        let mut cycles = 0u64;
        let mut process_cycles = 0u64;
        let (mut creation, mut exit, mut kernel, mut user) =
            (FileTime::default(), FileTime::default(), FileTime::default(), FileTime::default());
        // SAFETY: every out-pointer refers to a live local for the call.
        unsafe {
            let thread = GetCurrentThread();
            if QueryThreadCycleTime(thread, &mut cycles) == 0 {
                cycles = 0;
            }
            if QueryProcessCycleTime(GetCurrentProcess(), &mut process_cycles) == 0 {
                process_cycles = 0;
            }
            if GetThreadTimes(thread, &mut creation, &mut exit, &mut kernel, &mut user) == 0 {
                kernel = FileTime::default();
                user = FileTime::default();
            }
        }
        let wide = |t: FileTime| (u64::from(t.high) << 32) | u64::from(t.low);
        ThreadCost {
            cycles,
            process_cycles,
            cpu_100ns: wide(kernel) + wide(user),
        }
    }

    pub fn memory() -> MemoryInfo {
        let mut counters = ProcessMemoryCounters {
            cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
            ..ProcessMemoryCounters::default()
        };
        // SAFETY: `counters` is a correctly sized, initialised out-structure.
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
        if ok == 0 {
            return MemoryInfo::default();
        }
        MemoryInfo {
            peak_working_set: counters.peak_working_set_size as u64,
            working_set: counters.working_set_size as u64,
            peak_commit: counters.peak_pagefile_usage as u64,
            page_faults: u64::from(counters.page_fault_count),
        }
    }
}

#[cfg(not(windows))]
mod imp {
    use super::{MemoryInfo, PinReport, ThreadCost};

    pub fn pin(spec: &str, _priority_high: bool) -> PinReport {
        PinReport {
            requested: spec.to_owned(),
            note: "pinning is only implemented on Windows".into(),
            ..PinReport::default()
        }
    }

    pub fn thread_cost() -> ThreadCost {
        ThreadCost::default()
    }

    pub fn memory() -> MemoryInfo {
        MemoryInfo::default()
    }
}

/// Pin the whole process (before any worker thread exists) and optionally
/// raise its priority class. `spec`: `pcore`, `cpu:N`, `mask:HEX`, or `none`.
pub fn pin(spec: &str, priority_high: bool) -> PinReport {
    imp::pin(spec, priority_high)
}

/// Cost counters of the calling thread.
pub fn thread_cost() -> ThreadCost {
    imp::thread_cost()
}

/// Process memory counters.
pub fn memory() -> MemoryInfo {
    imp::memory()
}

/// Estimated thread cycles per nanosecond while running, taken as the highest
/// ratio over several short busy windows so a preempted window cannot lower
/// it. Returns 0.0 when cycle counts are unavailable.
pub fn cycles_per_ns() -> f64 {
    let mut best = 0.0f64;
    for _ in 0..12 {
        let cost0 = thread_cost();
        let start = std::time::Instant::now();
        let mut sink = 0u64;
        while start.elapsed().as_micros() < 4_000 {
            for i in 0..2_000u64 {
                sink = sink.wrapping_mul(6364136223846793005).wrapping_add(i);
            }
        }
        std::hint::black_box(sink);
        let ns = start.elapsed().as_nanos() as f64;
        let cycles = thread_cost().cycles.saturating_sub(cost0.cycles) as f64;
        if ns > 0.0 {
            best = best.max(cycles / ns);
        }
    }
    best
}
