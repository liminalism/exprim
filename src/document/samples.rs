//! Image sample decoding, shared between concurrent page compiles.
//!
//! A filter-decoded (Flate/LZW/RunLength/unfiltered) image is decoded by the
//! page compiler, not by a backend codec, and every compiled page that draws
//! it carries its samples until its render finishes. Without sharing, a
//! document whose pages all draw one large image held one decoded copy per
//! page in flight. [`ImageSampleCache`] lets every compile that overlaps
//! another reuse that compile's `Arc<[u8]>`: one decode, one buffer.
//!
//! Exactness: a hit returns the bytes a fresh decode would, and charges the
//! context's decode budget the same.
//!
//! - The entry is keyed by the image stream's object id but *matched* on a
//!   [`Fingerprint`] of everything the decoder reads: the raw body (its
//!   source range, or for an `Owned` body — every stream of an encrypted
//!   document, decrypted with its object's key — the bytes themselves) and
//!   the `/Filter` and `/DecodeParms` values after the context resolved their
//!   indirect parts. Resolving those can depend on the worker (a reference
//!   that failed to resolve under an exhausted budget is left as is), so a
//!   compile whose fingerprint differs simply decodes for itself.
//! - A hit charges the bytes the publishing decode charged. When the reader's
//!   remaining budget cannot cover that, it decodes for itself and fails (or
//!   not) exactly as it would have; so a limit outcome never depends on which
//!   page compiled first.
//! - A shared decode resolves the filter parameters before it reads the body
//!   (a hit never reads it). Only an in-bounds body is shared, so the two
//!   orders differ only when a read of the immutable source fails on an I/O
//!   error; everything else goes the unshared way, in the unshared order.
//! - Only successful sample decodes are published. A failed, panicking or
//!   codec-terminated leader publishes nothing; the compiles that waited for
//!   it then decode for themselves.
//!
//! Retention: the cache keeps an entry only while some compile is running
//! (a session, see [`DocumentSnapshot::image_sample_session`]) and drops
//! entries nothing else holds whenever a session begins or ends; when the
//! last session ends it empties. An `Arc<[u8]>` keeps its bytes inline, so a
//! `Weak` would pin the whole allocation just the same; the cache therefore
//! holds a strong reference and prunes by count instead, which costs no more
//! memory and lets a compile revive samples a just-finished page released.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use crate::document::{DocumentError, DocumentSnapshot, ParseContext};
use crate::object::{Dictionary, NameTable, ObjectId, PdfObject, PdfStream, StreamData};
use crate::structure::decode::SharedDecoded;

impl DocumentSnapshot {
    /// Decode an image stream for its samples: the same bytes, errors and
    /// decode-budget charge as [`Self::decode_stream_data_to_codec`], but
    /// returned as shared samples without a full-buffer copy.
    ///
    /// `expected_len` is the caller's estimate of the decoded size (row bytes
    /// × height); it only lets a well-formed Flate/LZW/RunLength stream decode
    /// into one exactly-sized buffer and never changes the result (see
    /// [`crate::structure::decode::decode_stream_to_codec_shared`]). `object`
    /// is the image stream's indirect object id: while an
    /// [`Self::image_sample_session`] is open, concurrent and overlapping
    /// decodes of the same object share one decode and one buffer.
    pub fn decode_image_data(
        &self,
        stream: &PdfStream,
        object: Option<ObjectId>,
        expected_len: Option<usize>,
        ctx: &mut ParseContext,
    ) -> Result<SharedDecoded, DocumentError> {
        let resolver = self.resolver();
        let raw_shared = match &stream.data {
            StreamData::Owned(bytes) => Some(bytes),
            StreamData::InSource { .. } => None,
        };
        let decode = |raw: &[u8], dict: &Dictionary, ctx: &mut ParseContext| {
            resolver.decode_body_shared(raw, raw_shared, dict, ctx, expected_len)
        };
        let cache = self.extension::<ImageSampleCache>();
        let shareable = object.filter(|_| cache.is_open() && self.body_in_bounds(&stream.data));
        let Some(id) = shareable else {
            // Not shared: exactly `decode_stream_data_to_codec`'s effects on
            // `ctx`, in its order — read the body, resolve the filter
            // parameters, decode.
            let raw = resolver.stream_body(&stream.data)?;
            let patched = resolver.effective_filter_dict(&stream.dict, ctx);
            let dict = patched.as_ref().unwrap_or(&stream.dict);
            return decode(&raw, dict, ctx).map(|(decoded, _)| decoded);
        };
        // Shared: the filter parameters are resolved first (the fingerprint
        // needs them) and the body is read only when this compile decodes,
        // so a hit never re-reads a large unfiltered body out of a
        // file-backed source. The order only matters if that read fails,
        // and a read of an in-bounds range of the immutable source fails
        // only on an I/O error, never because of the document.
        let patched = resolver.effective_filter_dict(&stream.dict, ctx);
        let dict = patched.as_ref().unwrap_or(&stream.dict);
        let fingerprint = Fingerprint::new(stream, dict, self.names());
        match ImageSampleCache::acquire(&cache, id) {
            Acquire::Cached(ready)
                if ready.fingerprint.matches(&fingerprint)
                    && ready.charged <= resolver.remaining_decode_budget(ctx) =>
            {
                ctx.decoded_bytes_used += ready.charged;
                Ok(SharedDecoded::Bytes(ready.samples))
            }
            Acquire::Lead(flight) => {
                // An error or panic from here on drops `flight` unpublished,
                // which releases the compiles waiting on it.
                let raw = resolver.stream_body(&stream.data)?;
                let result = decode(&raw, dict, ctx);
                if let Ok((SharedDecoded::Bytes(samples), charged)) = &result {
                    flight.publish(fingerprint, samples, *charged);
                }
                result.map(|(decoded, _)| decoded)
            }
            // A different stream state or a budget too small to take the
            // cached charge (decoding fails exactly as it would have), or no
            // flight to join: decode for this compile alone.
            Acquire::Cached(_) | Acquire::Alone => {
                let raw = resolver.stream_body(&stream.data)?;
                decode(&raw, dict, ctx).map(|(decoded, _)| decoded)
            }
        }
    }

    /// Whether a stream's body lies inside the source, i.e. reading it can
    /// fail only on an I/O error.
    fn body_in_bounds(&self, data: &StreamData) -> bool {
        match data {
            StreamData::InSource { offset, len } => offset
                .checked_add(*len)
                .is_some_and(|end| end <= self.source().len()),
            StreamData::Owned(_) => true,
        }
    }

    /// Open an image-sample sharing session for the length of one page
    /// compile (the page compiler holds one per interpreter). Decodes through
    /// [`Self::decode_image_data`] share only while some session is open, and
    /// the cache empties when the last one closes, so an idle document
    /// retains no samples on the cache's account.
    pub(crate) fn image_sample_session(&self) -> ImageSampleSession {
        let cache = self.extension::<ImageSampleCache>();
        {
            let mut state = cache.lock();
            state.sessions += 1;
            state.prune();
        }
        ImageSampleSession { cache }
    }
}

/// An open sharing session (see [`DocumentSnapshot::image_sample_session`]).
#[derive(Debug)]
pub(crate) struct ImageSampleSession {
    cache: Arc<ImageSampleCache>,
}

impl Drop for ImageSampleSession {
    fn drop(&mut self) {
        let mut state = self.cache.lock();
        state.sessions = state.sessions.saturating_sub(1);
        if state.sessions == 0 {
            // No compile is running, so none is decoding either: nothing in
            // flight to keep.
            state
                .slots
                .retain(|_, slot| matches!(slot, Slot::Decoding(_)));
        } else {
            state.prune();
        }
    }
}

/// The document-scoped cache (an extension slot of the snapshot).
#[derive(Debug, Default)]
pub(crate) struct ImageSampleCache {
    state: Mutex<CacheState>,
    /// Signalled whenever a flight ends, published or not.
    settled: Condvar,
}

#[derive(Debug, Default)]
struct CacheState {
    /// Open sessions; nothing is shared outside one.
    sessions: usize,
    next_flight: u64,
    slots: HashMap<ObjectId, Slot>,
}

impl CacheState {
    /// Drop published entries nothing outside the cache holds any more.
    fn prune(&mut self) {
        self.slots.retain(|_, slot| match slot {
            Slot::Decoding(_) => true,
            Slot::Ready(ready) => Arc::strong_count(&ready.samples) > ready.own_references(),
        });
    }
}

#[derive(Debug)]
enum Slot {
    /// A compile is decoding this object; others wait for it.
    Decoding(u64),
    Ready(Ready),
}

/// A published decode.
#[derive(Debug, Clone)]
struct Ready {
    fingerprint: Arc<Fingerprint>,
    samples: Arc<[u8]>,
    /// Decode-budget bytes the publishing decode charged.
    charged: usize,
}

impl Ready {
    /// References to `samples` this entry itself holds: its own, plus the
    /// fingerprint's when an unfiltered `Owned` body *is* the samples.
    fn own_references(&self) -> usize {
        match &self.fingerprint.body {
            Body::Owned(body) if Arc::ptr_eq(body, &self.samples) => 2,
            _ => 1,
        }
    }
}

enum Acquire {
    /// A published entry; the caller still checks its fingerprint.
    Cached(Ready),
    /// The caller decodes and publishes.
    Lead(Flight),
    /// The caller decodes for itself: no session is open, or the flight it
    /// waited for published nothing.
    Alone,
}

impl ImageSampleCache {
    fn lock(&self) -> MutexGuard<'_, CacheState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Whether some session is open (sharing is on).
    fn is_open(&self) -> bool {
        self.lock().sessions > 0
    }

    /// The published entry for `id`, after waiting out a decode in flight;
    /// or the lead, when there is none.
    fn acquire(cache: &Arc<Self>, id: ObjectId) -> Acquire {
        let mut state = cache.lock();
        if state.sessions == 0 {
            return Acquire::Alone;
        }
        let mut waited = false;
        loop {
            match state.slots.get(&id) {
                Some(Slot::Ready(ready)) => return Acquire::Cached(ready.clone()),
                Some(Slot::Decoding(_)) => {
                    waited = true;
                    state = cache
                        .settled
                        .wait(state)
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                }
                // The flight this compile waited for ended without
                // publishing (it failed, panicked, or hit a codec): decode
                // in parallel with the others it released rather than one
                // after another.
                None if waited => return Acquire::Alone,
                None => {
                    let flight = state.next_flight;
                    state.next_flight = state.next_flight.wrapping_add(1);
                    state.slots.insert(id, Slot::Decoding(flight));
                    return Acquire::Lead(Flight {
                        cache: Arc::clone(cache),
                        id,
                        flight,
                        settled: false,
                    });
                }
            }
        }
    }
}

/// The lead on one object's decode. Dropped unpublished — an error, a codec
/// stream, or a panic unwinding through the decode — it withdraws the slot
/// and wakes the waiters, so none is stranded.
struct Flight {
    cache: Arc<ImageSampleCache>,
    id: ObjectId,
    flight: u64,
    settled: bool,
}

impl Flight {
    fn publish(mut self, fingerprint: Fingerprint, samples: &Arc<[u8]>, charged: usize) {
        let mut state = self.cache.lock();
        if matches!(state.slots.get(&self.id), Some(Slot::Decoding(f)) if *f == self.flight) {
            let ready = Ready {
                fingerprint: Arc::new(fingerprint),
                samples: Arc::clone(samples),
                charged,
            };
            state.slots.insert(self.id, Slot::Ready(ready));
        }
        self.settled = true;
        drop(state);
        self.cache.settled.notify_all();
    }
}

impl Drop for Flight {
    fn drop(&mut self) {
        if self.settled {
            return;
        }
        let mut state = self.cache.lock();
        if matches!(state.slots.get(&self.id), Some(Slot::Decoding(f)) if *f == self.flight) {
            state.slots.remove(&self.id);
        }
        drop(state);
        self.cache.settled.notify_all();
    }
}

/// Everything the decoder reads from an image stream.
#[derive(Debug)]
struct Fingerprint {
    body: Body,
    filter: Option<PdfObject>,
    parms: Option<PdfObject>,
}

#[derive(Debug)]
enum Body {
    /// A range of the (immutable) document source.
    InSource { offset: u64, len: u64 },
    /// Materialised bytes (a decrypted body), compared by content.
    Owned(Arc<[u8]>),
}

impl Fingerprint {
    /// `dict` is the stream's dictionary as decoded: `/Filter` and
    /// `/DecodeParms` already resolved through the context.
    fn new(stream: &PdfStream, dict: &Dictionary, names: &NameTable) -> Self {
        let body = match &stream.data {
            StreamData::InSource { offset, len } => Body::InSource {
                offset: *offset,
                len: *len,
            },
            StreamData::Owned(bytes) => Body::Owned(Arc::clone(bytes)),
        };
        Self {
            body,
            filter: dict.get(names.known.filter).cloned(),
            parms: dict.get(names.known.decode_parms).cloned(),
        }
    }

    fn matches(&self, other: &Self) -> bool {
        let body = match (&self.body, &other.body) {
            (
                Body::InSource { offset, len },
                Body::InSource {
                    offset: other_offset,
                    len: other_len,
                },
            ) => offset == other_offset && len == other_len,
            (Body::Owned(a), Body::Owned(b)) => Arc::ptr_eq(a, b) || a[..] == b[..],
            _ => false,
        };
        body && same_option(self.filter.as_ref(), other.filter.as_ref())
            && same_option(self.parms.as_ref(), other.parms.as_ref())
    }
}

fn same_option(a: Option<&PdfObject>, b: Option<&PdfObject>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => same_value(a, b),
        _ => false,
    }
}

/// Structural equality strict enough that equal values decode identically:
/// reals compare by bit pattern, nested streams only by identity.
fn same_value(a: &PdfObject, b: &PdfObject) -> bool {
    match (a, b) {
        (PdfObject::Null, PdfObject::Null) => true,
        (PdfObject::Boolean(a), PdfObject::Boolean(b)) => a == b,
        (PdfObject::Integer(a), PdfObject::Integer(b)) => a == b,
        (PdfObject::Real(a), PdfObject::Real(b)) => a.to_bits() == b.to_bits(),
        (PdfObject::Name(a), PdfObject::Name(b)) => a == b,
        (PdfObject::String(a), PdfObject::String(b)) => a == b,
        (PdfObject::Array(a), PdfObject::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b.iter()).all(|(a, b)| same_value(a, b))
        }
        (PdfObject::Dictionary(a), PdfObject::Dictionary(b)) => same_dict(a, b),
        (PdfObject::Stream(a), PdfObject::Stream(b)) => Arc::ptr_eq(a, b),
        (PdfObject::Reference(a), PdfObject::Reference(b)) => a == b,
        _ => false,
    }
}

fn same_dict(a: &Dictionary, b: &Dictionary) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|((ka, va), (kb, vb))| ka == kb && same_value(va, vb))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;

    fn ready(samples: &Arc<[u8]>) -> Ready {
        Ready {
            fingerprint: Arc::new(Fingerprint {
                body: Body::InSource { offset: 0, len: 1 },
                filter: None,
                parms: None,
            }),
            samples: Arc::clone(samples),
            charged: samples.len(),
        }
    }

    #[test]
    fn a_failed_leader_wakes_its_waiters_and_publishes_nothing() {
        let cache = Arc::new(ImageSampleCache::default());
        cache.lock().sessions = 1;
        let id = ObjectId::new(5, 0);
        let Acquire::Lead(flight) = ImageSampleCache::acquire(&cache, id) else {
            panic!("first acquire leads");
        };
        let waiter = {
            let cache = Arc::clone(&cache);
            std::thread::spawn(move || match ImageSampleCache::acquire(&cache, id) {
                // Blocked on the flight, released by its failure.
                Acquire::Alone => true,
                // Arrived only after the failure (scheduling): it leads.
                Acquire::Lead(_) => true,
                Acquire::Cached(_) => false,
            })
        };
        // Give the waiter time to block, then fail the flight by unwinding
        // through it.
        std::thread::sleep(std::time::Duration::from_millis(50));
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _flight = flight;
            panic!("leader panics mid-decode");
        }));
        assert!(panicked.is_err());
        assert!(
            waiter.join().unwrap(),
            "the waiter is released and nothing was published"
        );
        assert!(cache.lock().slots.is_empty());
        // The next compile leads afresh.
        assert!(matches!(
            ImageSampleCache::acquire(&cache, id),
            Acquire::Lead(_)
        ));
    }

    #[test]
    fn entries_live_only_while_held_and_while_a_session_is_open() {
        let cache = Arc::new(ImageSampleCache::default());
        let id = ObjectId::new(9, 0);
        let samples: Arc<[u8]> = Arc::from(vec![1u8, 2, 3]);
        {
            let mut state = cache.lock();
            state.sessions = 2;
            state.slots.insert(id, Slot::Ready(ready(&samples)));
            state.prune();
            assert_eq!(state.slots.len(), 1, "held by a page: kept");
        }
        drop(samples);
        {
            let mut state = cache.lock();
            state.prune();
            assert!(state.slots.is_empty(), "held by nothing else: pruned");
        }
        // No session open: nothing is shared or published.
        cache.lock().sessions = 0;
        assert!(matches!(
            ImageSampleCache::acquire(&cache, id),
            Acquire::Alone
        ));
    }

    #[test]
    fn fingerprints_compare_owned_bodies_by_content() {
        let a = Fingerprint {
            body: Body::Owned(Arc::from(vec![1u8, 2, 3])),
            filter: Some(PdfObject::Integer(1)),
            parms: Some(PdfObject::Real(0.5)),
        };
        let b = Fingerprint {
            body: Body::Owned(Arc::from(vec![1u8, 2, 3])),
            filter: Some(PdfObject::Integer(1)),
            parms: Some(PdfObject::Real(0.5)),
        };
        assert!(a.matches(&b));
        let c = Fingerprint {
            body: Body::Owned(Arc::from(vec![1u8, 2, 4])),
            ..b
        };
        assert!(!a.matches(&c));
        let d = Fingerprint {
            body: Body::InSource { offset: 0, len: 3 },
            filter: None,
            parms: None,
        };
        assert!(!a.matches(&d));
    }

    // ── Through the page compiler ───────────────────────────────────────────

    use crate::content::PageCompiler;
    use crate::document::{DocumentLimits, PageIndex};
    use crate::page_ir::{CompiledPage, ImageMask};
    use crate::security::{Cipher, EncryptDict, StandardHandler};
    use crate::source::{OwnedBytesSource, PdfSource};
    use crate::test_support::builder::PdfBuilder;
    use std::io::Write;

    fn zlib(data: &[u8]) -> Vec<u8> {
        let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(data).unwrap();
        enc.finish().unwrap()
    }

    fn pattern(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed) ^ (i >> 5) as u8)
            .collect()
    }

    fn encrypt_fields(u: Vec<u8>) -> EncryptDict {
        EncryptDict {
            v: 2,
            r: 3,
            o: vec![0xAB; 32],
            u,
            ue: vec![],
            oe: vec![],
            p: -44,
            perms: vec![],
            key_bytes: 16,
            encrypt_metadata: true,
            cipher: Cipher::Rc4,
            file_id: vec![0x5A; 16],
        }
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// `pages` pages that all draw the same two images: `Im1`, RGB with a
    /// PNG predictor whose `/DecodeParms` is indirect and a gray `/SMask`;
    /// `Im2`, gray with a stencil `/Mask` stream. Every stream RC4-encrypted
    /// when `encrypt` (so its body resolves `Owned`).
    fn image_doc(w: usize, h: usize, pages: u32, encrypt: bool) -> Vec<u8> {
        let u = StandardHandler::compute_user_entry(&encrypt_fields(vec![]), "");
        let handler = StandardHandler::open(&encrypt_fields(u.clone())).unwrap();
        let body = |n: u32, plain: Vec<u8>| -> Vec<u8> {
            let mut b = plain;
            if encrypt {
                handler.decrypt(n, 0, &mut b); // RC4 is symmetric
            }
            b
        };
        let rgb = pattern(w * h * 3, 1);
        let mut filtered = Vec::new();
        for (r, row) in rgb.chunks(w * 3).enumerate() {
            filtered.push(2); // Up
            for (i, &v) in row.iter().enumerate() {
                let up = if r == 0 { 0 } else { rgb[(r - 1) * w * 3 + i] };
                filtered.push(v.wrapping_sub(up));
            }
        }
        let mut b = PdfBuilder::new();
        b.add_object(1, "<</Type/Catalog/Pages 2 0 R>>");
        let kids: String = (0..pages).map(|i| format!("{} 0 R ", 100 + i)).collect();
        b.add_object(
            2,
            &format!("<</Type/Pages/Kids[{kids}]/Count {pages}/MediaBox[0 0 200 200]>>"),
        );
        if encrypt {
            b.add_object(
                4,
                &format!(
                    "<</Filter/Standard/V 2/R 3/Length 128/P -44/O<{}>/U<{}>>>",
                    hex(&[0xAB; 32]),
                    hex(&u)
                ),
            );
        }
        b.add_stream(
            5,
            "",
            &body(
                5,
                b"q 50 0 0 50 0 0 cm /Im1 Do Q q 50 0 0 50 60 0 cm /Im2 Do Q".to_vec(),
            ),
        );
        b.add_stream(
            10,
            &format!(
                "/Type/XObject/Subtype/Image/Width {w}/Height {h}/BitsPerComponent 8\
                 /ColorSpace/DeviceRGB/Filter/FlateDecode/DecodeParms 13 0 R/SMask 11 0 R"
            ),
            &body(10, zlib(&filtered)),
        );
        b.add_object(13, &format!("<</Predictor 15/Colors 3/Columns {w}>>"));
        b.add_stream(
            11,
            &format!(
                "/Type/XObject/Subtype/Image/Width {w}/Height {h}/BitsPerComponent 8\
                 /ColorSpace/DeviceGray/Filter/FlateDecode"
            ),
            &body(11, zlib(&pattern(w * h, 2))),
        );
        b.add_stream(
            12,
            &format!(
                "/Type/XObject/Subtype/Image/Width {w}/Height {h}/BitsPerComponent 8\
                 /ColorSpace/DeviceGray/Filter/FlateDecode/Mask 14 0 R"
            ),
            &body(12, zlib(&pattern(w * h, 3))),
        );
        b.add_stream(
            14,
            &format!(
                "/Type/XObject/Subtype/Image/ImageMask true/Width {w}/Height {h}\
                 /Filter/FlateDecode"
            ),
            &body(14, zlib(&pattern(w.div_ceil(8) * h, 4))),
        );
        for i in 0..pages {
            b.add_object(
                100 + i,
                "<</Type/Page/Parent 2 0 R/Contents 5 0 R\
                 /Resources<</XObject<</Im1 10 0 R/Im2 12 0 R>>>>>>",
            );
        }
        let id = format!("/ID[<{}><{}>]", hex(&[0x5A; 16]), hex(&[0x5A; 16]));
        if encrypt {
            b.finish_classic_xref(&format!("/Root 1 0 R/Encrypt 4 0 R{id}"));
        } else {
            b.finish_classic_xref(&format!("/Root 1 0 R{id}"));
        }
        b.into_bytes()
    }

    fn open(bytes: &[u8], limits: DocumentLimits) -> DocumentSnapshot {
        let source: Arc<dyn PdfSource> = Arc::new(OwnedBytesSource::new(bytes.to_vec()));
        DocumentSnapshot::open(source, limits).unwrap()
    }

    /// Im1, its soft mask, Im2, its stencil mask — `None` where a decode
    /// failed and the page carries no samples for it.
    fn page_samples(page: &CompiledPage) -> [Option<Arc<[u8]>>; 4] {
        let im1 = page.images.iter().find(|i| i.smask.is_some());
        let im2 = page.images.iter().find(|i| i.mask.is_some());
        [
            im1.and_then(|i| i.samples.clone()),
            im1.and_then(|i| i.smask.as_ref())
                .map(|m| Arc::clone(&m.samples)),
            im2.and_then(|i| i.samples.clone()),
            im2.and_then(|i| match &i.mask {
                Some(ImageMask::Stencil(m)) => Some(Arc::clone(&m.samples)),
                _ => None,
            }),
        ]
    }

    fn compile(snapshot: &DocumentSnapshot, page: u32, ctx: &mut ParseContext) -> CompiledPage {
        PageCompiler::new()
            .compile(snapshot, PageIndex(page), ctx)
            .unwrap()
    }

    /// The bytes an uncached, unhinted decode of object `n` produces.
    fn uncached(snapshot: &DocumentSnapshot, n: u32) -> Vec<u8> {
        let mut ctx = ParseContext::new();
        let object = snapshot
            .objects()
            .resolve(snapshot, ObjectId::new(n, 0), &mut ctx)
            .unwrap();
        let PdfObject::Stream(stream) = &*object else {
            panic!("object {n} is not a stream");
        };
        let (bytes, codec) = snapshot
            .decode_stream_data_to_codec(stream, &mut ctx)
            .unwrap();
        assert!(codec.is_none());
        bytes
    }

    #[test]
    fn concurrent_compiles_share_one_decode_of_each_image() {
        const PAGES: u32 = 6;
        const THREADS: usize = 6;
        for encrypt in [false, true] {
            let bytes = image_doc(64, 48, PAGES, encrypt);
            // The decode-budget charge of one page compiled with no sharing.
            let reference = open(&bytes, DocumentLimits::default());
            let mut ctx = ParseContext::new();
            let plain = compile(&reference, 0, &mut ctx);
            let charged = ctx.decoded_bytes_used;
            assert!(page_samples(&plain).iter().all(Option::is_some));

            let snapshot = open(&bytes, DocumentLimits::default());
            // Held across the run, as overlapping compiles of a scheduler
            // keep it open, so the sharing is deterministic here.
            let session = snapshot.image_sample_session();
            let barrier = std::sync::Barrier::new(THREADS);
            let results: Vec<Vec<(CompiledPage, usize)>> = std::thread::scope(|scope| {
                let workers: Vec<_> = (0..THREADS)
                    .map(|_| {
                        scope.spawn(|| {
                            barrier.wait();
                            (0..PAGES)
                                .map(|page| {
                                    let mut ctx = ParseContext::new();
                                    let compiled = compile(&snapshot, page, &mut ctx);
                                    (compiled, ctx.decoded_bytes_used)
                                })
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect();
                workers.into_iter().map(|w| w.join().unwrap()).collect()
            });
            let first = page_samples(&results[0][0].0);
            let objects = [10, 11, 12, 14];
            for (slot, n) in objects.iter().enumerate() {
                let shared = first[slot].as_ref().unwrap();
                assert_eq!(
                    &shared[..],
                    &uncached(&reference, *n)[..],
                    "object {n} (encrypt {encrypt}): shared bytes = uncached decode"
                );
            }
            for (compiled, used) in results.iter().flatten() {
                let samples = page_samples(compiled);
                for (slot, n) in objects.iter().enumerate() {
                    assert!(
                        Arc::ptr_eq(
                            samples[slot].as_ref().unwrap(),
                            first[slot].as_ref().unwrap()
                        ),
                        "object {n} (encrypt {encrypt}) decoded more than once"
                    );
                }
                assert_eq!(*used, charged, "a hit charges what a decode charges");
            }
            drop(results);
            drop(session);
            assert!(
                snapshot
                    .extension::<ImageSampleCache>()
                    .lock()
                    .slots
                    .is_empty(),
                "the cache empties when the last session closes"
            );
        }
    }

    /// A compile whose remaining decode budget cannot cover a cached decode
    /// must fail (or not) exactly as it would without the cache, and leave
    /// the same budget used.
    #[test]
    fn budget_outcomes_do_not_depend_on_the_cache() {
        let bytes = image_doc(16, 12, 1, false);
        let probe = open(&bytes, DocumentLimits::default());
        let mut ctx = ParseContext::new();
        let _page = compile(&probe, 0, &mut ctx);
        let total = ctx.decoded_bytes_used;

        let limits = DocumentLimits {
            max_decoded_bytes_per_context: total,
            ..Default::default()
        };
        let uncached_doc = open(&bytes, limits.clone());
        let cached_doc = open(&bytes, limits);
        let _session = cached_doc.image_sample_session();
        // Publish every image with a full budget; keep the page alive.
        let mut ctx = ParseContext::new();
        let primed = compile(&cached_doc, 0, &mut ctx);
        let primed_samples = page_samples(&primed);
        assert!(primed_samples.iter().all(Option::is_some));

        let as_bytes = |s: [Option<Arc<[u8]>>; 4]| s.map(|o| o.map(|a| a.to_vec()));
        let mut hits = 0;
        let mut failures = 0;
        for preload in 0..=total + 1 {
            let mut fresh = ParseContext::new();
            fresh.decoded_bytes_used = preload;
            let expected = as_bytes(page_samples(&compile(&uncached_doc, 0, &mut fresh)));
            failures += expected.iter().filter(|s| s.is_none()).count();
            let mut shared = ParseContext::new();
            shared.decoded_bytes_used = preload;
            let page = compile(&cached_doc, 0, &mut shared);
            let samples = page_samples(&page);
            hits += samples
                .iter()
                .zip(&primed_samples)
                .filter(|(s, p)| match (s, p) {
                    (Some(s), Some(p)) => Arc::ptr_eq(s, p),
                    _ => false,
                })
                .count();
            assert_eq!(as_bytes(samples), expected, "preload {preload}");
            assert_eq!(
                shared.decoded_bytes_used, fresh.decoded_bytes_used,
                "preload {preload}"
            );
        }
        assert!(hits > 0, "the cached path was exercised");
        assert!(failures > 0, "the budget-limited path was exercised");
    }
}
