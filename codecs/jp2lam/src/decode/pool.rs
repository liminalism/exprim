//! Bounded, thread-safe pool of reusable coefficient and synthesis planes.
//!
//! Large JPEG 2000 decodes allocate multi-megabyte `f32` (irreversible 9/7) or
//! `i32` (reversible 5/3) subband planes per component. Above the system
//! allocator's heap threshold (~128–512 KiB), fresh allocations and drops go
//! directly to OS virtual-memory mappings (`VirtualAlloc`/`VirtualFree`,
//! `mmap`/`munmap`), paying kernel lock, soft-fault, and cross-core TLB
//! shootdown overhead on every image.
//!
//! [`Jp2BufferPool`] parks finished planes under a hard byte ceiling so
//! repeated decodes — whether on one [`super::Jp2Decoder`] or across worker
//! threads sharing a renderer session — reuse resident user-space memory and
//! release it deterministically when the pool or session is cleared or dropped.

use std::sync::Mutex;

/// Default maximum bytes retained in an idle [`Jp2BufferPool`] (128 MiB).
///
/// Sized to hold the three colour components' subband planes (`~110 MB`) plus
/// coarse-level scratch of a 300 dpi letter/A4 scan without retaining
/// per-worker copies across a thread pool.
pub const DEFAULT_DECODE_BUFFER_POOL_BYTES: usize = 128 * 1024 * 1024;

/// Minimum buffer capacity (in 4-byte elements = 64 KiB) eligible for pooling.
/// Smaller buffers stay in the thread-local small-block heap without touching
/// the pool lock or bucket table.
const MIN_POOLED_ELEMENTS: usize = 16 * 1024;

/// Maximum distinct `(kind, capacity_bucket)` classes kept at once so a
/// long-lived pool moving across unrelated image geometries evicts stale sizes
/// even before reaching `cap_bytes`. One odd-dimension 5-level tile uses up to
/// six subband sizes, so twelve covers two distinct geometries simultaneously.
const MAX_POOL_BUCKETS: usize = 12;

enum ParkedPlane {
    F32(Vec<f32>),
    I32(Vec<i32>),
}

impl ParkedPlane {
    #[inline]
    fn byte_capacity(&self) -> usize {
        match self {
            Self::F32(buf) => buf.capacity().saturating_mul(std::mem::size_of::<f32>()),
            Self::I32(buf) => buf.capacity().saturating_mul(std::mem::size_of::<i32>()),
        }
    }

    #[inline]
    fn bucket_key(&self) -> (u8, usize) {
        match self {
            Self::F32(buf) => (0, buf.capacity() >> 10),
            Self::I32(buf) => (1, buf.capacity() >> 10),
        }
    }
}

#[derive(Default)]
struct PoolState {
    /// Oldest first; lookups scan from the back so the most recently used
    /// (warmest in cache/TLB) buffer is reused first.
    planes: Vec<ParkedPlane>,
    total_bytes: usize,
}

/// Thread-safe, byte-capped pool of `f32` and `i32` working planes for
/// [`super::Jp2Decoder`].
pub struct Jp2BufferPool {
    cap_bytes: usize,
    state: Mutex<PoolState>,
}

impl std::fmt::Debug for Jp2BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Jp2BufferPool")
            .field("cap_bytes", &self.cap_bytes)
            .field("idle_bytes", &self.idle_bytes())
            .finish()
    }
}

impl Default for Jp2BufferPool {
    fn default() -> Self {
        Self::new(DEFAULT_DECODE_BUFFER_POOL_BYTES)
    }
}

#[inline]
fn max_reusable_capacity(len: usize) -> usize {
    len.saturating_add(len >> 2)
}

impl Jp2BufferPool {
    /// Create a pool retaining at most `cap_bytes` of idle plane capacity.
    /// Passing `0` disables retention completely.
    pub fn new(cap_bytes: usize) -> Self {
        Self {
            cap_bytes,
            state: Mutex::new(PoolState::default()),
        }
    }

    /// Create a zero-capacity pool that never retains buffers.
    pub fn disabled() -> Self {
        Self::new(0)
    }

    /// Maximum bytes of idle buffer capacity this pool will retain.
    pub fn cap_bytes(&self) -> usize {
        self.cap_bytes
    }

    /// Bytes of buffer capacity currently parked in the pool.
    pub fn idle_bytes(&self) -> usize {
        if self.cap_bytes == 0 {
            return 0;
        }
        self.lock().total_bytes
    }

    /// Drop all parked buffers immediately, returning their memory to the
    /// allocator outside the internal lock.
    pub fn clear(&self) {
        if self.cap_bytes == 0 {
            return;
        }
        let dropped = {
            let mut state = self.lock();
            state.total_bytes = 0;
            std::mem::take(&mut state.planes)
        };
        drop(dropped);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, PoolState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn take_raw_f32(&self, len: usize) -> Option<Vec<f32>> {
        if self.cap_bytes == 0 || len < MIN_POOLED_ELEMENTS {
            return None;
        }
        let max_cap = max_reusable_capacity(len);
        let mut state = self.lock();
        let index = state.planes.iter().rposition(|plane| match plane {
            ParkedPlane::F32(buf) => buf.capacity() >= len && buf.capacity() <= max_cap,
            ParkedPlane::I32(_) => false,
        })?;
        let plane = state.planes.remove(index);
        state.total_bytes = state.total_bytes.saturating_sub(plane.byte_capacity());
        match plane {
            ParkedPlane::F32(buf) => Some(buf),
            ParkedPlane::I32(_) => None,
        }
    }

    fn take_raw_i32(&self, len: usize) -> Option<Vec<i32>> {
        if self.cap_bytes == 0 || len < MIN_POOLED_ELEMENTS {
            return None;
        }
        let max_cap = max_reusable_capacity(len);
        let mut state = self.lock();
        let index = state.planes.iter().rposition(|plane| match plane {
            ParkedPlane::I32(buf) => buf.capacity() >= len && buf.capacity() <= max_cap,
            ParkedPlane::F32(_) => false,
        })?;
        let plane = state.planes.remove(index);
        state.total_bytes = state.total_bytes.saturating_sub(plane.byte_capacity());
        match plane {
            ParkedPlane::I32(buf) => Some(buf),
            ParkedPlane::F32(_) => None,
        }
    }

    /// Take a `Vec<f32>` of length `len` whose elements are all `+0.0`.
    pub(crate) fn take_zeroed_f32(&self, len: usize) -> Vec<f32> {
        if let Some(mut buf) = self.take_raw_f32(len) {
            buf.clear();
            buf.resize(len, 0.0);
            buf
        } else {
            vec![0.0f32; len]
        }
    }

    /// Take a `Vec<f32>` of length `len` when the caller will overwrite every
    /// element `0..len` before reading it.
    pub(crate) fn take_for_overwrite_f32(&self, len: usize) -> Vec<f32> {
        if let Some(mut buf) = self.take_raw_f32(len) {
            if buf.len() >= len {
                buf.truncate(len);
            } else {
                buf.resize(len, 0.0);
            }
            buf
        } else {
            vec![0.0f32; len]
        }
    }

    /// Take a `Vec<i32>` of length `len` whose elements are all `0`.
    pub(crate) fn take_zeroed_i32(&self, len: usize) -> Vec<i32> {
        if let Some(mut buf) = self.take_raw_i32(len) {
            buf.clear();
            buf.resize(len, 0);
            buf
        } else {
            vec![0i32; len]
        }
    }

    /// Take a `Vec<i32>` of length `len` when the caller will overwrite every
    /// element `0..len` before reading it.
    pub(crate) fn take_for_overwrite_i32(&self, len: usize) -> Vec<i32> {
        if let Some(mut buf) = self.take_raw_i32(len) {
            if buf.len() >= len {
                buf.truncate(len);
            } else {
                buf.resize(len, 0);
            }
            buf
        } else {
            vec![0i32; len]
        }
    }

    pub(crate) fn park_f32(&self, buf: Vec<f32>) {
        if self.cap_bytes == 0 || buf.capacity() < MIN_POOLED_ELEMENTS {
            return;
        }
        self.park_plane(ParkedPlane::F32(buf));
    }

    pub(crate) fn park_i32(&self, buf: Vec<i32>) {
        if self.cap_bytes == 0 || buf.capacity() < MIN_POOLED_ELEMENTS {
            return;
        }
        self.park_plane(ParkedPlane::I32(buf));
    }

    fn park_plane(&self, plane: ParkedPlane) {
        let bytes = plane.byte_capacity();
        if bytes == 0 || bytes > self.cap_bytes {
            return;
        }
        let mut evicted: Vec<ParkedPlane> = Vec::new();
        {
            let mut state = self.lock();
            state.planes.push(plane);
            state.total_bytes = state.total_bytes.saturating_add(bytes);

            let mut buckets = [(0u8, 0usize); MAX_POOL_BUCKETS];
            let mut bucket_count = 0usize;
            for item in state.planes.iter().rev() {
                let key = item.bucket_key();
                if !buckets[..bucket_count].contains(&key) && bucket_count < MAX_POOL_BUCKETS {
                    buckets[bucket_count] = key;
                    bucket_count += 1;
                }
            }
            let mut i = 0usize;
            while i < state.planes.len() {
                if !buckets[..bucket_count].contains(&state.planes[i].bucket_key()) {
                    let old = state.planes.remove(i);
                    state.total_bytes = state.total_bytes.saturating_sub(old.byte_capacity());
                    evicted.push(old);
                } else {
                    i += 1;
                }
            }
            while state.total_bytes > self.cap_bytes && !state.planes.is_empty() {
                let old = state.planes.remove(0);
                state.total_bytes = state.total_bytes.saturating_sub(old.byte_capacity());
                evicted.push(old);
            }
        }
        drop(evicted);
    }
}

#[cfg(test)]
mod tests {
    use super::{Jp2BufferPool, MIN_POOLED_ELEMENTS};

    #[test]
    fn zeroed_checkout_scrubs_dirty_f32_and_i32_bits() {
        let pool = Jp2BufferPool::new(4 * 1024 * 1024);
        let len = MIN_POOLED_ELEMENTS * 2;

        let mut dirty_f32 = vec![f32::NAN; len];
        dirty_f32[0] = -0.0;
        dirty_f32[len - 1] = 42.5;
        let ptr_f32 = dirty_f32.as_ptr() as usize;
        pool.park_f32(dirty_f32);
        assert!(pool.idle_bytes() >= len * 4);

        let clean_f32 = pool.take_zeroed_f32(len);
        assert_eq!(
            clean_f32.as_ptr() as usize,
            ptr_f32,
            "reused parked allocation"
        );
        assert_eq!(clean_f32.len(), len);
        assert!(
            clean_f32.iter().all(|v| v.to_bits() == 0),
            "every recycled f32 must be bit-exact +0.0"
        );

        let dirty_i32 = vec![-999_999i32; len];
        let ptr_i32 = dirty_i32.as_ptr() as usize;
        pool.park_i32(dirty_i32);
        let clean_i32 = pool.take_zeroed_i32(len);
        assert_eq!(clean_i32.as_ptr() as usize, ptr_i32);
        assert!(clean_i32.iter().all(|&v| v == 0));
    }

    #[test]
    fn pool_enforces_byte_cap_and_clear() {
        let len = MIN_POOLED_ELEMENTS * 2; // 128 KiB
        let cap = len * 4 * 2; // room for at most 2 buffers
        let pool = Jp2BufferPool::new(cap);

        pool.park_f32(vec![1.0; len]);
        pool.park_f32(vec![2.0; len]);
        pool.park_f32(vec![3.0; len]);
        assert!(pool.idle_bytes() <= cap);

        // Oversized single buffer (> cap) is never retained.
        pool.park_f32(vec![4.0; len * 4]);
        assert!(pool.idle_bytes() <= cap);

        pool.clear();
        assert_eq!(pool.idle_bytes(), 0);
    }

    #[test]
    fn disabled_pool_retains_nothing() {
        let pool = Jp2BufferPool::disabled();
        pool.park_f32(vec![1.0; MIN_POOLED_ELEMENTS * 2]);
        pool.park_i32(vec![1; MIN_POOLED_ELEMENTS * 2]);
        assert_eq!(pool.idle_bytes(), 0);
    }
}
