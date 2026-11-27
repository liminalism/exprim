//! Stream filter decoding: the `/Filter` chain.
//!
//! Phase 1 codecs: FlateDecode (+ predictors), ASCIIHexDecode,
//! ASCII85Decode, RunLengthDecode. Image codecs (DCT, JBIG2, CCITT, JPX)
//! are later-phase work and surface as [`DecodeError::UnsupportedFilter`].
//!
//! Decompression is the classic PDF bomb vector, so every byte produced is
//! charged against a caller-supplied [`DecodeBudget`]. The budget is
//! worker-owned (it lives on / is derived from the worker's parse context)
//! — never global.

use std::borrow::Cow;
use std::sync::Arc;

use crate::object::{Dictionary, NameTable, PdfObject};

use crate::structure::predictor::{
    PredictorError, PredictorParms, apply_predictor, geometry, tiff_in_place, unfilter_png_row,
};

/// Decompressed-byte budget for one decode context. Constructed from the
/// worker's remaining allowance; the caller settles `used()` back into its
/// own accounting afterwards.
#[derive(Debug)]
pub struct DecodeBudget {
    remaining: usize,
    used: usize,
}

impl DecodeBudget {
    pub fn new(remaining: usize) -> Self {
        Self { remaining, used: 0 }
    }

    /// Total bytes charged so far.
    pub fn used(&self) -> usize {
        self.used
    }

    /// Charge `n` output bytes; fails when the budget is exhausted.
    pub fn charge(&mut self, n: usize) -> Result<(), DecodeError> {
        if n > self.remaining {
            return Err(DecodeError::BudgetExceeded);
        }
        self.remaining -= n;
        self.used += n;
        Ok(())
    }
}

/// Errors from filter decoding.
#[derive(Debug, Clone, thiserror::Error)]
pub enum DecodeError {
    #[error("unsupported stream filter /{0}")]
    UnsupportedFilter(String),
    #[error("decompressed data exceeds the decoded-bytes budget")]
    BudgetExceeded,
    #[error("corrupt stream data: {0}")]
    Corrupt(&'static str),
    #[error(transparent)]
    Predictor(#[from] PredictorError),
}

/// Apply the full `/Filter` chain of `dict` to `raw`, in order.
///
/// `/Filter` may be a single name or an array; `/DecodeParms` parallels it
/// (single dict, or array with nulls for filters without parameters).
/// Abbreviated inline-image names (`Fl`, `AHx`, `A85`, `RL`) are tolerated
/// here too — they cost nothing and appear in malformed files.
pub fn decode_stream(
    raw: &[u8],
    dict: &Dictionary,
    names: &NameTable,
    budget: &mut DecodeBudget,
) -> Result<Vec<u8>, DecodeError> {
    match decode_stream_inner(raw, dict, names, budget, false, None)? {
        (data, None) => Ok(data.into_vec()),
        (_, Some(codec)) => Err(DecodeError::UnsupportedFilter(codec)),
    }
}

/// Like [`decode_stream`], but stop at the first *image-codec* filter
/// (DCT/JPX/JBIG2/CCITTFax): the general-filter prefix is applied and the
/// still-codec-encoded bytes are returned together with the codec filter's
/// canonical name. `None` means the chain contained no codec filter and the
/// bytes are fully decoded.
pub fn decode_stream_to_codec(
    raw: &[u8],
    dict: &Dictionary,
    names: &NameTable,
    budget: &mut DecodeBudget,
) -> Result<(Vec<u8>, Option<String>), DecodeError> {
    let (data, codec) = decode_stream_inner(raw, dict, names, budget, true, None)?;
    Ok((data.into_vec(), codec))
}

/// What [`decode_stream_to_codec_shared`] produced.
#[derive(Debug)]
pub enum SharedDecoded {
    /// The chain held no image codec: the fully decoded bytes.
    Bytes(Arc<[u8]>),
    /// The chain stopped at an image codec: the still-encoded bytes and the
    /// codec filter's canonical name, exactly as [`decode_stream_to_codec`]
    /// returns them.
    Codec(Vec<u8>, String),
    /// The chain is the image codec alone: its still-encoded bytes are the
    /// stream body `raw` itself, unchanged. Returned instead of [`Self::Codec`]
    /// so a caller holding the body in a mapped source can keep a reference
    /// to it rather than a copy (a scanned book's pages are each a megabyte
    /// of JPEG, and every page compile copied its image).
    CodecUnfiltered(String),
    /// [`Self::CodecUnfiltered`] resolved by
    /// [`crate::document::DocumentSnapshot::decode_image_data`]: the payload as
    /// shared bytes, borrowed from the document's source when it is mapped.
    /// [`decode_stream_to_codec_shared`] itself never returns it.
    CodecShared(crate::page_ir::SharedBytes, String),
}

/// [`decode_stream_to_codec`] for a caller that keeps the result as shared
/// `Arc<[u8]>` samples (image data), producing the same bytes, errors and
/// budget charges without the full-buffer copy of `Arc::from(Vec)`.
///
/// `expected_len` is the caller's estimate of the fully decoded length (an
/// image's row bytes × height). When the last filter of the chain is
/// Flate, LZW or RunLength and its output turns out to be exactly that long —
/// and, for Flate, the deflate stream ends cleanly there — it is decoded
/// straight into one exactly-sized shared buffer, with a PNG or TIFF
/// predictor undone in that buffer row by row. Any other outcome (a shorter,
/// longer, truncated or corrupt stream, a predictor whose geometry does not
/// match) discards that attempt, charges nothing for it, and decodes the
/// ordinary way, so the hint can change only memory use and time, never the
/// result. `raw_shared`, when given, must hold the same bytes as `raw`: an
/// unfiltered body is then handed back without copying it.
pub fn decode_stream_to_codec_shared(
    raw: &[u8],
    raw_shared: Option<&Arc<[u8]>>,
    dict: &Dictionary,
    names: &NameTable,
    budget: &mut DecodeBudget,
    expected_len: Option<usize>,
) -> Result<SharedDecoded, DecodeError> {
    match decode_stream_inner(raw, dict, names, budget, true, expected_len)? {
        (Decoded::Shared(bytes), None) => Ok(SharedDecoded::Bytes(bytes)),
        (Decoded::Cow(Cow::Borrowed(body)), None) => {
            // Only an unfiltered body comes back borrowed: it *is* `raw`.
            let shared = raw_shared
                .filter(|shared| std::ptr::eq(shared.as_ptr(), body.as_ptr()))
                .filter(|shared| shared.len() == body.len())
                .map_or_else(|| Arc::from(body), Arc::clone);
            Ok(SharedDecoded::Bytes(shared))
        }
        (Decoded::Cow(Cow::Owned(bytes)), None) => Ok(SharedDecoded::Bytes(Arc::from(bytes))),
        (Decoded::Cow(Cow::Borrowed(body)), Some(codec))
            if std::ptr::eq(body.as_ptr(), raw.as_ptr()) && body.len() == raw.len() =>
        {
            Ok(SharedDecoded::CodecUnfiltered(codec))
        }
        (data, Some(codec)) => Ok(SharedDecoded::Codec(data.into_vec(), codec)),
    }
}

/// The bytes a filter chain produced: still the source body (unfiltered),
/// an owned intermediate, or an exactly-sized shared buffer.
enum Decoded<'a> {
    Cow(Cow<'a, [u8]>),
    Shared(Arc<[u8]>),
}

impl Decoded<'_> {
    fn into_vec(self) -> Vec<u8> {
        match self {
            Decoded::Cow(data) => data.into_owned(),
            Decoded::Shared(data) => data.to_vec(),
        }
    }
}

fn decode_stream_inner<'a>(
    raw: &'a [u8],
    dict: &Dictionary,
    names: &NameTable,
    budget: &mut DecodeBudget,
    stop_at_codec: bool,
    expected_len: Option<usize>,
) -> Result<(Decoded<'a>, Option<String>), DecodeError> {
    let filter = dict.get(names.known.filter);
    let parms = dict.get(names.known.decode_parms);

    let filters: Vec<PdfObject> = match filter {
        None | Some(PdfObject::Null) => Vec::new(),
        Some(PdfObject::Array(a)) => a.to_vec(),
        Some(other) => vec![other.clone()],
    };
    let parms_list: Vec<Option<PdfObject>> = match parms {
        None | Some(PdfObject::Null) => vec![None; filters.len()],
        Some(PdfObject::Array(a)) => {
            let mut v: Vec<Option<PdfObject>> = a.iter().map(|p| Some(p.clone())).collect();
            v.resize(filters.len(), None);
            v
        }
        Some(other) => {
            let mut v = vec![Some(other.clone())];
            v.resize(filters.len(), None);
            v
        }
    };

    // Materializing an unfiltered body still allocates: charge it, so the
    // per-context budget covers every byte a stream produces, filtered or
    // not (a 2 GiB "uncompressed" stream is as much a bomb as a flate one).
    if filters.is_empty() {
        budget.charge(raw.len())?;
    }
    // Borrow the body rather than copying it: the first filter reads it and
    // replaces `data` outright, so an eager `to_vec` here is a full-stream
    // copy that is discarded on the very next line for every filtered stream
    // in the document. The copy still happens, once, at whichever return
    // actually needs to hand back owned bytes.
    let mut data: Cow<'_, [u8]> = Cow::Borrowed(raw);
    for (i, (f, p)) in filters.iter().zip(parms_list).enumerate() {
        let Some(name_id) = f.as_name() else {
            return Err(DecodeError::Corrupt("/Filter entry is not a name"));
        };
        let name = names.resolve(name_id);
        let parms_dict = p.as_ref().and_then(|p| p.as_dict().cloned());
        // The last filter produces the final bytes: with a length hint, try
        // decoding it straight into its exactly-sized shared buffer. `None`
        // has charged nothing, and the ordinary step below runs as if the
        // attempt never happened.
        if i + 1 == filters.len()
            && let Some(expected) = expected_len
            && let Some(shared) = decode_last_exact(
                name.as_ref(),
                &data,
                parms_dict.as_ref(),
                names,
                budget,
                expected,
            )
        {
            return Ok((Decoded::Shared(shared), None));
        }
        data = Cow::Owned(match name.as_ref() {
            b"FlateDecode" | b"Fl" => {
                let inflated = flate_decode(&data, budget)?;
                apply_parms(inflated, parms_dict.as_ref(), names)?
            }
            b"LZWDecode" | b"LZW" => {
                // /EarlyChange (default 1) lives in the same /DecodeParms dict
                // as the predictor keys; read it before applying the predictor.
                let early_change = lzw_early_change(parms_dict.as_ref(), names);
                let decoded = lzw_decode(&data, budget, early_change)?;
                apply_parms(decoded, parms_dict.as_ref(), names)?
            }
            b"ASCIIHexDecode" | b"AHx" => ascii_hex_decode(&data, budget)?,
            b"ASCII85Decode" | b"A85" => ascii85_decode(&data, budget)?,
            b"RunLengthDecode" | b"RL" => run_length_decode(&data, budget)?,
            codec @ (b"DCTDecode" | b"DCT" | b"JPXDecode" | b"JBIG2Decode" | b"CCITTFaxDecode"
            | b"CCF")
                if stop_at_codec =>
            {
                // Canonicalize the abbreviated inline-image spellings.
                let canonical = match codec {
                    b"DCT" => "DCTDecode",
                    b"CCF" => "CCITTFaxDecode",
                    other => std::str::from_utf8(other).unwrap_or("DCTDecode"),
                };
                // Preceding filters already charged their output; a codec
                // that is the first filter returns the raw body, uncharged
                // so far — charge it like an unfiltered body.
                if i == 0 {
                    budget.charge(data.len())?;
                }
                return Ok((Decoded::Cow(data), Some(canonical.to_owned())));
            }
            other => {
                return Err(DecodeError::UnsupportedFilter(
                    String::from_utf8_lossy(other).into_owned(),
                ));
            }
        });
    }
    Ok((Decoded::Cow(data), None))
}

/// Read predictor parameters out of a `/DecodeParms` dict and apply them.
fn apply_parms(
    data: Vec<u8>,
    parms: Option<&Dictionary>,
    names: &NameTable,
) -> Result<Vec<u8>, DecodeError> {
    let Some(p) = predictor_parms(parms, names) else {
        return Ok(data);
    };
    Ok(apply_predictor(data, &p)?)
}

/// The predictor parameters a `/DecodeParms` dict names; `None` (no
/// predictor) when there is no parms dict.
fn predictor_parms(parms: Option<&Dictionary>, names: &NameTable) -> Option<PredictorParms> {
    let parms = parms?;
    let get_int = |key: &[u8], default: i64| -> i64 {
        names
            .lookup(key)
            .and_then(|id| parms.get(id))
            .and_then(PdfObject::as_int)
            .unwrap_or(default)
    };
    Some(PredictorParms {
        predictor: get_int(b"Predictor", 1),
        colors: get_int(b"Colors", 1),
        bits_per_component: get_int(b"BitsPerComponent", 8),
        columns: get_int(b"Columns", 1),
    })
}

/// `/EarlyChange` (default 1) from an LZW filter's `/DecodeParms`.
fn lzw_early_change(parms: Option<&Dictionary>, names: &NameTable) -> bool {
    parms
        .and_then(|d| names.lookup(b"EarlyChange").and_then(|id| d.get(id)))
        .and_then(PdfObject::as_int)
        .unwrap_or(1)
        != 0
}

// ── Exactly-sized decoding of the last filter ──────────────────────────────
//
// The ordinary path grows a `Vec` geometrically — capacity ends anywhere in
// (S, 2S] with the old and new buffers both alive across every reallocation —
// then un-predicts a PNG stream into a second buffer, and the caller copies
// the result into an `Arc<[u8]>`. For a well-formed image whose decoded size
// is known up front, everything below produces the same bytes in one shared
// buffer of exactly that size. Each attempt either commits (charging the
// budget exactly what the ordinary path would have) or returns `None` having
// charged nothing, so the ordinary path then runs unchanged.

/// Most bytes one input byte can expand to under each filter, used to refuse
/// a hint the input cannot possibly fill before allocating for it: deflate
/// tops out near 1032:1, a 9-bit LZW code emits at most 4096 bytes, and a
/// two-byte RunLength repeat emits 128.
const FLATE_MAX_RATIO: usize = 1100;
const LZW_MAX_RATIO: usize = 4096;
const RUN_LENGTH_MAX_RATIO: usize = 64;
/// Headroom on top of those ratios for headers, checksums and tiny inputs.
const HINT_SLACK: usize = 8 * 1024;

/// How the last filter's output becomes the final bytes.
enum Unpredict {
    /// No predictor (absent, or `/Predictor 1`).
    None,
    /// TIFF predictor 2: same length, undone in place afterwards.
    Tiff(PredictorParms),
    /// PNG predictors 10–15: `rows` rows of `row_len` bytes, each preceded in
    /// the filter output by its one-byte filter tag.
    Png {
        row_len: usize,
        bpp: usize,
        rows: usize,
    },
}

impl Unpredict {
    /// The plan for producing `expected` final bytes, and how many bytes the
    /// filter itself must output for it. `None` when the parameters are ones
    /// the ordinary path rejects, or a PNG geometry `expected` is not a whole
    /// number of rows of.
    fn plan(parms: Option<PredictorParms>, expected: usize) -> Option<(Self, usize)> {
        let Some(p) = parms else {
            return Some((Unpredict::None, expected));
        };
        match p.predictor {
            1 => Some((Unpredict::None, expected)),
            2 => {
                geometry(&p).ok()?;
                Some((Unpredict::Tiff(p), expected))
            }
            10..=15 => {
                let (bpp, row_len) = geometry(&p).ok()?;
                if !expected.is_multiple_of(row_len) {
                    return None;
                }
                let rows = expected / row_len;
                let produced = rows.checked_mul(row_len.checked_add(1)?)?;
                Some((Unpredict::Png { row_len, bpp, rows }, produced))
            }
            _ => None,
        }
    }
}

/// Try to decode the chain's last filter (`filter`, over `input`) straight
/// into an exactly `expected`-byte shared buffer. See the section comment.
fn decode_last_exact(
    filter: &[u8],
    input: &[u8],
    parms: Option<&Dictionary>,
    names: &NameTable,
    budget: &mut DecodeBudget,
    expected: usize,
) -> Option<Arc<[u8]>> {
    if expected == 0 {
        return None;
    }
    let (ratio, plan, produced) = match filter {
        b"FlateDecode" | b"Fl" => {
            let (plan, produced) = Unpredict::plan(predictor_parms(parms, names), expected)?;
            (FLATE_MAX_RATIO, plan, produced)
        }
        b"LZWDecode" | b"LZW" => {
            let (plan, produced) = Unpredict::plan(predictor_parms(parms, names), expected)?;
            // A PNG-predicted LZW stream would need its tag bytes threaded
            // through the code stream; rare enough to leave to the ordinary
            // path.
            if matches!(plan, Unpredict::Png { .. }) {
                return None;
            }
            (LZW_MAX_RATIO, plan, produced)
        }
        // RunLength takes no predictor.
        b"RunLengthDecode" | b"RL" => (RUN_LENGTH_MAX_RATIO, Unpredict::None, expected),
        _ => return None,
    };
    // A hint the budget cannot cover would fail the ordinary way part-way
    // through, charging what it had produced by then; leave that outcome to
    // the ordinary path. A hint the input cannot fill is not worth an
    // allocation of that size.
    if produced > budget.remaining
        || produced > input.len().saturating_mul(ratio).saturating_add(HINT_SLACK)
    {
        return None;
    }
    let mut out = zeroed_shared(expected);
    let buf = Arc::get_mut(&mut out)?;
    let filled = match filter {
        b"FlateDecode" | b"Fl" => inflate_exact(input, &plan, buf),
        b"LZWDecode" | b"LZW" => {
            let mut sink = FixedSink { buf, len: 0 };
            // The budget was checked above and is charged on commit; the
            // decoder's own per-code charges go to a scratch budget the sink
            // bounds long before it could matter.
            let mut scratch = DecodeBudget::new(usize::MAX);
            matches!(
                lzw_decode_into(
                    input,
                    &mut scratch,
                    lzw_early_change(parms, names),
                    &mut sink
                ),
                Ok(true)
            ) && sink.len == expected
        }
        _ => {
            let mut sink = FixedSink { buf, len: 0 };
            let mut scratch = DecodeBudget::new(usize::MAX);
            matches!(run_length_into(input, &mut scratch, &mut sink), Ok(true))
                && sink.len == expected
        }
    };
    if !filled {
        return None;
    }
    if let Unpredict::Tiff(p) = &plan {
        tiff_in_place(Arc::get_mut(&mut out)?, p).ok()?;
    }
    budget.charge(produced).ok()?;
    Some(out)
}

/// A zero-filled shared buffer of `len` bytes, allocated once at its final
/// size (a trusted-length iterator collects straight into the `Arc`).
fn zeroed_shared(len: usize) -> Arc<[u8]> {
    std::iter::repeat_n(0u8, len).collect()
}

/// Inflate `input` into `out` per `plan`, succeeding only when the deflate
/// stream ends cleanly (`StreamEnd`) exactly at the last byte the plan
/// needs. A clean stream's output does not depend on how it was buffered, so
/// that is precisely the case where these bytes and the ordinary path's
/// agree; a truncated, corrupt or longer stream fails here and is decoded
/// the ordinary way (whose salvage can depend on its buffer sizes).
fn inflate_exact(input: &[u8], plan: &Unpredict, out: &mut [u8]) -> bool {
    let mut z = flate2::Decompress::new(true);
    let ended = match *plan {
        Unpredict::None | Unpredict::Tiff(_) => inflate_fill(&mut z, input, out),
        Unpredict::Png { row_len, bpp, rows } => {
            let zero_row = vec![0u8; row_len];
            let mut tag = [0u8; 1];
            let mut ended = Some(false);
            for r in 0..rows {
                // Each row's tag byte goes to scratch, its filtered bytes
                // straight to their final place, where the row is un-filtered
                // against the decoded row above it.
                if inflate_fill(&mut z, input, &mut tag) != Some(false) {
                    return false;
                }
                let (done, rest) = out.split_at_mut(r * row_len);
                let Some(cur) = rest.get_mut(..row_len) else {
                    return false;
                };
                ended = inflate_fill(&mut z, input, cur);
                if ended.is_none() || (ended == Some(true) && r + 1 < rows) {
                    return false;
                }
                let prev: &[u8] = if r == 0 {
                    &zero_row
                } else {
                    &done[(r - 1) * row_len..]
                };
                unfilter_png_row(tag[0], cur, prev, bpp);
            }
            ended
        }
    };
    match ended {
        Some(true) => true,
        Some(false) => inflate_at_end(&mut z, input),
        None => false,
    }
}

/// Inflate into all of `out`. `Some(true)`: filled, and the deflate stream
/// ended with its last byte; `Some(false)`: filled, stream not (yet) ended;
/// `None`: the stream ended, stalled or failed before `out` was full.
fn inflate_fill(z: &mut flate2::Decompress, input: &[u8], out: &mut [u8]) -> Option<bool> {
    let mut pos = 0usize;
    while pos < out.len() {
        let before_in = z.total_in();
        let before_out = z.total_out();
        let rest = input.get(usize::try_from(before_in).ok()?..)?;
        let status = z
            .decompress(rest, &mut out[pos..], flate2::FlushDecompress::None)
            .ok()?;
        let produced = usize::try_from(z.total_out() - before_out).ok()?;
        pos = pos.checked_add(produced)?;
        match status {
            flate2::Status::StreamEnd => return (pos == out.len()).then_some(true),
            _ if produced == 0 && z.total_in() == before_in => return None,
            _ => {}
        }
    }
    Some(false)
}

/// After `out` is full: whether the deflate stream now ends (it may still
/// have its checksum to consume) without producing one more byte.
fn inflate_at_end(z: &mut flate2::Decompress, input: &[u8]) -> bool {
    let mut probe = [0u8; 1];
    loop {
        let before_in = z.total_in();
        let before_out = z.total_out();
        let Some(rest) = usize::try_from(before_in)
            .ok()
            .and_then(|start| input.get(start..))
        else {
            return false;
        };
        let Ok(status) = z.decompress(rest, &mut probe, flate2::FlushDecompress::None) else {
            return false;
        };
        if z.total_out() != before_out {
            return false; // longer than expected
        }
        match status {
            flate2::Status::StreamEnd => return true,
            _ if z.total_in() == before_in => return false, // stalled: truncated
            _ => {}
        }
    }
}

/// Where the LZW and RunLength decoders put their output: a growable `Vec`
/// (the ordinary path) or a fixed buffer that reports overflow.
trait ByteSink {
    /// Append `bytes`; `false` when the sink is full.
    fn put(&mut self, bytes: &[u8]) -> bool;
    /// Append `bytes` in reverse order.
    fn put_rev(&mut self, bytes: &[u8]) -> bool;
    /// Append `count` copies of `byte`.
    fn put_run(&mut self, byte: u8, count: usize) -> bool;
}

impl ByteSink for Vec<u8> {
    fn put(&mut self, bytes: &[u8]) -> bool {
        self.extend_from_slice(bytes);
        true
    }
    fn put_rev(&mut self, bytes: &[u8]) -> bool {
        self.extend(bytes.iter().rev());
        true
    }
    fn put_run(&mut self, byte: u8, count: usize) -> bool {
        self.extend(std::iter::repeat_n(byte, count));
        true
    }
}

/// A fixed-size output buffer and how much of it is written.
struct FixedSink<'a> {
    buf: &'a mut [u8],
    len: usize,
}

impl FixedSink<'_> {
    fn next(&mut self, count: usize) -> Option<&mut [u8]> {
        let end = self.len.checked_add(count)?;
        let dst = self.buf.get_mut(self.len..end)?;
        self.len = end;
        Some(dst)
    }
}

impl ByteSink for FixedSink<'_> {
    fn put(&mut self, bytes: &[u8]) -> bool {
        self.next(bytes.len())
            .map(|dst| dst.copy_from_slice(bytes))
            .is_some()
    }
    fn put_rev(&mut self, bytes: &[u8]) -> bool {
        self.next(bytes.len())
            .map(|dst| {
                for (d, s) in dst.iter_mut().zip(bytes.iter().rev()) {
                    *d = *s;
                }
            })
            .is_some()
    }
    fn put_run(&mut self, byte: u8, count: usize) -> bool {
        self.next(count).map(|dst| dst.fill(byte)).is_some()
    }
}

/// FlateDecode. Tries zlib format first, then raw deflate (headerless
/// streams occur in the wild — PDFium's inflate tolerates them the same
/// way). A corrupt tail after some successful output is tolerated: the
/// bytes produced so far are returned, matching viewer behavior of showing
/// what can be salvaged.
fn flate_decode(data: &[u8], budget: &mut DecodeBudget) -> Result<Vec<u8>, DecodeError> {
    match inflate_with(flate2::Decompress::new(true), data, budget) {
        Ok(out) => Ok(out),
        Err(DecodeError::BudgetExceeded) => Err(DecodeError::BudgetExceeded),
        Err(first_err) => match inflate_with(flate2::Decompress::new(false), data, budget) {
            Ok(out) if !out.is_empty() => Ok(out),
            _ => Err(first_err),
        },
    }
}

fn inflate_with(
    mut decompress: flate2::Decompress,
    data: &[u8],
    budget: &mut DecodeBudget,
) -> Result<Vec<u8>, DecodeError> {
    // Inflate straight into the output vector's spare capacity: a scratch
    // buffer would memcpy every byte twice. The initial guess mirrors PDFium's
    // `EstimateFlateUncompressBufferSize` — compressed size scaled by a typical
    // ratio and capped, so a 4:1 text stream lands in one allocation without a
    // small stream reserving megabytes.
    const MIN_RESERVE: usize = 8 * 1024;
    const MAX_GUESS: usize = 4 * 1024 * 1024;
    let guess = data.len().saturating_mul(4).clamp(MIN_RESERVE, MAX_GUESS);
    let mut out: Vec<u8> = Vec::with_capacity(guess);
    loop {
        if out.len() == out.capacity() {
            out.reserve(out.capacity().max(MIN_RESERVE));
        }
        let before_in = decompress.total_in();
        let before_out = decompress.total_out();
        let input = &data[usize::try_from(before_in)
            .unwrap_or(usize::MAX)
            .min(data.len())..];
        let status = decompress
            .decompress_vec(input, &mut out, flate2::FlushDecompress::None)
            .map_err(|_| {
                // Corrupt tail: keep what we already produced, if anything.
                DecodeError::Corrupt("flate stream is corrupt")
            });
        let produced = usize::try_from(decompress.total_out() - before_out)
            .map_err(|_| DecodeError::Corrupt("flate output overflow"))?;
        budget.charge(produced)?;
        match status {
            Ok(flate2::Status::StreamEnd) => return Ok(out),
            Ok(_) => {}
            Err(e) => {
                if out.is_empty() {
                    return Err(e);
                }
                return Ok(out);
            }
        }
        let consumed = decompress.total_in() - before_in;
        if consumed == 0 && produced == 0 {
            // No forward progress: truncated stream. Tolerate if we
            // produced output (salvage), else corrupt.
            if out.is_empty() {
                return Err(DecodeError::Corrupt("flate stream truncated"));
            }
            return Ok(out);
        }
    }
}

/// ASCIIHexDecode: hex pairs, whitespace ignored, `>` is EOD, odd trailing
/// digit padded with 0. Other bytes are skipped (tolerated superset of the
/// spec, consistent with the lexer's hex-string handling).
fn ascii_hex_decode(data: &[u8], budget: &mut DecodeBudget) -> Result<Vec<u8>, DecodeError> {
    let mut out = Vec::new();
    let mut pending: Option<u8> = None;
    for &b in data {
        if b == b'>' {
            break;
        }
        if let Some(v) = crate::syntax::classify::hex_value(b) {
            match pending.take() {
                Some(hi) => {
                    budget.charge(1)?;
                    out.push((hi << 4) | v);
                }
                None => pending = Some(v),
            }
        }
    }
    if let Some(hi) = pending {
        budget.charge(1)?;
        out.push(hi << 4);
    }
    Ok(out)
}

/// ASCII85Decode per ISO 32000-1 §7.4.3.
fn ascii85_decode(data: &[u8], budget: &mut DecodeBudget) -> Result<Vec<u8>, DecodeError> {
    let mut out = Vec::new();
    let mut group = [0u8; 5];
    let mut n = 0usize;
    let mut i = 0usize;
    // Optional `<~` prefix (Adobe convention) is tolerated.
    if data.starts_with(b"<~") {
        i = 2;
    }
    while i < data.len() {
        let b = data[i];
        i += 1;
        if b == b'~' {
            break; // `~>` EOD
        }
        if crate::syntax::classify::is_whitespace(b) {
            continue;
        }
        if b == b'z' && n == 0 {
            budget.charge(4)?;
            out.extend_from_slice(&[0, 0, 0, 0]);
            continue;
        }
        if !(b'!'..=b'u').contains(&b) {
            return Err(DecodeError::Corrupt("invalid ASCII85 character"));
        }
        group[n] = b - b'!';
        n += 1;
        if n == 5 {
            let mut v: u32 = 0;
            for &d in &group {
                v = v
                    .checked_mul(85)
                    .and_then(|v| v.checked_add(u32::from(d)))
                    .ok_or(DecodeError::Corrupt("ASCII85 group overflow"))?;
            }
            budget.charge(4)?;
            out.extend_from_slice(&v.to_be_bytes());
            n = 0;
        }
    }
    if n == 1 {
        return Err(DecodeError::Corrupt("ASCII85 partial group of one digit"));
    }
    if n > 1 {
        // Pad with 'u' (84) and emit n-1 bytes.
        let mut v: u32 = 0;
        for (k, &digit) in group.iter().enumerate() {
            let d = if k < n { u32::from(digit) } else { 84 };
            v = v
                .checked_mul(85)
                .and_then(|v| v.checked_add(d))
                .ok_or(DecodeError::Corrupt("ASCII85 group overflow"))?;
        }
        budget.charge(n - 1)?;
        out.extend_from_slice(&v.to_be_bytes()[..n - 1]);
    }
    Ok(out)
}

/// RunLengthDecode per ISO 32000-1 §7.4.5.
fn run_length_decode(data: &[u8], budget: &mut DecodeBudget) -> Result<Vec<u8>, DecodeError> {
    let mut out = Vec::new();
    run_length_into(data, budget, &mut out)?;
    Ok(out)
}

/// [`run_length_decode`] into any [`ByteSink`]; `Ok(false)` when the sink
/// filled up first.
fn run_length_into<S: ByteSink>(
    data: &[u8],
    budget: &mut DecodeBudget,
    out: &mut S,
) -> Result<bool, DecodeError> {
    let mut i = 0usize;
    while i < data.len() {
        let l = data[i];
        i += 1;
        match l {
            128 => break, // EOD
            0..=127 => {
                let count = usize::from(l) + 1;
                if i + count > data.len() {
                    return Err(DecodeError::Corrupt("run-length literal overruns data"));
                }
                budget.charge(count)?;
                if !out.put(&data[i..i + count]) {
                    return Ok(false);
                }
                i += count;
            }
            129..=255 => {
                let count = 257 - usize::from(l);
                let Some(&b) = data.get(i) else {
                    return Err(DecodeError::Corrupt("run-length repeat missing byte"));
                };
                i += 1;
                budget.charge(count)?;
                if !out.put_run(b, count) {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

/// LZWDecode per ISO 32000-1 §7.4.4 (the TIFF/PDF variable-width LZW variant).
///
/// Codes are packed most-significant-bit first and grow from 9 to 12 bits as
/// the string table fills. Code 256 clears the table, 257 is end-of-data.
/// `early_change` (PDF `/EarlyChange`, default 1) makes the code width step up
/// one code sooner, matching every Acrobat-produced stream; a `0` stream turns
/// it off. A truncated stream that simply runs out of bits before EOD yields
/// the bytes decoded so far, matching a viewer showing a partial image.
///
/// The table stores each code as `(prefix, suffix)` and strings are rebuilt
/// through a scratch stack, so a pathological stream cannot cost O(n²) copying.
fn lzw_decode(
    data: &[u8],
    budget: &mut DecodeBudget,
    early_change: bool,
) -> Result<Vec<u8>, DecodeError> {
    let mut out = Vec::new();
    lzw_decode_into(data, budget, early_change, &mut out)?;
    Ok(out)
}

/// [`lzw_decode`] into any [`ByteSink`]; `Ok(false)` when the sink filled up
/// first.
fn lzw_decode_into<S: ByteSink>(
    data: &[u8],
    budget: &mut DecodeBudget,
    early_change: bool,
    out: &mut S,
) -> Result<bool, DecodeError> {
    const CLEAR: usize = 256;
    const EOD: usize = 257;
    const FIRST: usize = 258;
    const MAX: usize = 4096;

    let early = usize::from(early_change);
    let mut prefix = [0u16; MAX];
    let mut suffix = [0u8; MAX];
    let mut stack: Vec<u8> = Vec::with_capacity(MAX);

    let mut bit_buf: u32 = 0;
    let mut bit_count: u32 = 0;
    let mut pos = 0usize;
    let mut width: u32 = 9;
    let mut next = FIRST;
    let mut old_code: i32 = -1;
    let mut first_byte: u8 = 0;

    loop {
        // Refill to at least `width` bits, MSB-first. Running dry before EOD is
        // tolerated: emit what we have (truncated-stream salvage).
        while bit_count < width && pos < data.len() {
            bit_buf = (bit_buf << 8) | u32::from(data[pos]);
            pos += 1;
            bit_count += 8;
        }
        if bit_count < width {
            break;
        }
        bit_count -= width;
        let code = ((bit_buf >> bit_count) & ((1 << width) - 1)) as usize;

        if code == EOD {
            break;
        }
        if code == CLEAR {
            width = 9;
            next = FIRST;
            old_code = -1;
            continue;
        }
        if code > next || (code == next && old_code < 0) {
            return Err(DecodeError::Corrupt("LZW code out of range"));
        }

        // Rebuild this code's string onto the stack (reversed).
        stack.clear();
        let mut in_code = if code == next {
            // KwKwK: the not-yet-defined code is prev's string + its first byte.
            stack.push(first_byte);
            old_code as usize
        } else {
            code
        };
        while in_code >= FIRST {
            stack.push(suffix[in_code]);
            in_code = usize::from(prefix[in_code]);
        }
        first_byte = in_code as u8; // in_code is now a root byte (0..=255)
        stack.push(first_byte);

        budget.charge(stack.len())?;
        if !out.put_rev(&stack) {
            return Ok(false);
        }

        // Add prev + first_byte as the next table entry (§7.4.4.2).
        if old_code >= 0 && next < MAX {
            prefix[next] = old_code as u16;
            suffix[next] = first_byte;
            next += 1;
            if width < 12 && next + early >= (1 << width) {
                width += 1;
            }
        }
        old_code = code as i32;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
    use super::*;
    use crate::object::NameTable;
    use std::io::Write;

    fn budget() -> DecodeBudget {
        DecodeBudget::new(1 << 20)
    }

    fn dict_with_filter(names: &NameTable, filter: &[u8]) -> Dictionary {
        Dictionary::from_pairs([(names.known.filter, PdfObject::Name(names.intern(filter)))])
    }

    fn zlib_compress(data: &[u8]) -> Vec<u8> {
        let mut enc = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(data).unwrap();
        enc.finish().unwrap()
    }

    #[test]
    fn no_filter_is_identity_but_still_budgeted() {
        let names = NameTable::new();
        let dict = Dictionary::new();
        let mut b = budget();
        assert_eq!(
            decode_stream(b"plain", &dict, &names, &mut b).unwrap(),
            b"plain"
        );
        assert_eq!(b.used(), 5);
        let mut zero = DecodeBudget::new(0);
        assert!(matches!(
            decode_stream(b"plain", &dict, &names, &mut zero),
            Err(DecodeError::BudgetExceeded)
        ));
    }

    #[test]
    fn flate_roundtrip() {
        let names = NameTable::new();
        let dict = dict_with_filter(&names, b"FlateDecode");
        let raw = zlib_compress(b"the quick brown fox");
        assert_eq!(
            decode_stream(&raw, &dict, &names, &mut budget()).unwrap(),
            b"the quick brown fox"
        );
    }

    #[test]
    fn flate_headerless_raw_deflate_tolerated() {
        let names = NameTable::new();
        let dict = dict_with_filter(&names, b"FlateDecode");
        let mut enc =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        enc.write_all(b"headerless").unwrap();
        let raw = enc.finish().unwrap();
        assert_eq!(
            decode_stream(&raw, &dict, &names, &mut budget()).unwrap(),
            b"headerless"
        );
    }

    #[test]
    fn flate_budget_enforced() {
        let names = NameTable::new();
        let dict = dict_with_filter(&names, b"FlateDecode");
        // 1 MiB of zeros compresses tiny but must still exceed a 1 KiB budget.
        let raw = zlib_compress(&vec![0u8; 1 << 20]);
        let mut small = DecodeBudget::new(1024);
        assert!(matches!(
            decode_stream(&raw, &dict, &names, &mut small),
            Err(DecodeError::BudgetExceeded)
        ));
    }

    #[test]
    fn flate_with_png_predictor() {
        let names = NameTable::new();
        // Up-filtered rows: [1,2,3], [4,5,6] → deltas [1,2,3],[3,3,3].
        let filtered: Vec<u8> = vec![2, 1, 2, 3, 2, 3, 3, 3];
        let raw = zlib_compress(&filtered);
        let parms = Dictionary::from_pairs([
            (names.intern(b"Predictor"), PdfObject::Integer(12)),
            (names.intern(b"Columns"), PdfObject::Integer(3)),
        ]);
        let dict = Dictionary::from_pairs([
            (
                names.known.filter,
                PdfObject::Name(names.intern(b"FlateDecode")),
            ),
            (
                names.known.decode_parms,
                PdfObject::Dictionary(parms.into()),
            ),
        ]);
        assert_eq!(
            decode_stream(&raw, &dict, &names, &mut budget()).unwrap(),
            vec![1, 2, 3, 4, 5, 6]
        );
    }

    #[test]
    fn ascii_hex() {
        let names = NameTable::new();
        let dict = dict_with_filter(&names, b"ASCIIHexDecode");
        assert_eq!(
            decode_stream(b"48 65 6C 6C 6F>", &dict, &names, &mut budget()).unwrap(),
            b"Hello"
        );
        // Odd digit → padded with zero.
        assert_eq!(
            decode_stream(b"7>", &dict, &names, &mut budget()).unwrap(),
            vec![0x70]
        );
    }

    #[test]
    fn ascii85() {
        let names = NameTable::new();
        let dict = dict_with_filter(&names, b"ASCII85Decode");
        // "Man " encodes to 9jqo^ ; 'z' is four zero bytes.
        assert_eq!(
            decode_stream(b"9jqo^~>", &dict, &names, &mut budget()).unwrap(),
            b"Man "
        );
        assert_eq!(
            decode_stream(b"z~>", &dict, &names, &mut budget()).unwrap(),
            vec![0, 0, 0, 0]
        );
        // Partial group: "Man" → 3 bytes from 4 digits.
        assert_eq!(
            decode_stream(b"9jqo~>", &dict, &names, &mut budget()).unwrap(),
            b"Man"
        );
        // Whitespace ignored.
        assert_eq!(
            decode_stream(b"9j qo\n^~>", &dict, &names, &mut budget()).unwrap(),
            b"Man "
        );
    }

    #[test]
    fn run_length() {
        let names = NameTable::new();
        let dict = dict_with_filter(&names, b"RunLengthDecode");
        // literal "AB", repeat 'C' x4 (257-253), EOD.
        assert_eq!(
            decode_stream(
                &[1, b'A', b'B', 253, b'C', 128],
                &dict,
                &names,
                &mut budget()
            )
            .unwrap(),
            b"ABCCCC"
        );
    }

    #[test]
    fn filter_chain_applies_in_order() {
        let names = NameTable::new();
        // Flate-compress, then hex-encode; /Filter [AHx Fl] decodes
        // hex first, flate second.
        let deflated = zlib_compress(b"chained");
        let hex: Vec<u8> = deflated
            .iter()
            .flat_map(|b| format!("{b:02X}").into_bytes())
            .collect();
        let dict = Dictionary::from_pairs([(
            names.known.filter,
            PdfObject::Array(
                vec![
                    PdfObject::Name(names.intern(b"ASCIIHexDecode")),
                    PdfObject::Name(names.intern(b"FlateDecode")),
                ]
                .into(),
            ),
        )]);
        assert_eq!(
            decode_stream(&hex, &dict, &names, &mut budget()).unwrap(),
            b"chained"
        );
    }

    #[test]
    fn unsupported_filter_is_typed_error() {
        let names = NameTable::new();
        let dict = dict_with_filter(&names, b"DCTDecode");
        assert!(matches!(
            decode_stream(b"...", &dict, &names, &mut budget()),
            Err(DecodeError::UnsupportedFilter(f)) if f == "DCTDecode"
        ));
    }

    #[test]
    fn corrupt_flate_with_no_output_errors() {
        let names = NameTable::new();
        let dict = dict_with_filter(&names, b"FlateDecode");
        assert!(matches!(
            decode_stream(b"\xff\xff\xff\xff", &dict, &names, &mut budget()),
            Err(DecodeError::Corrupt(_))
        ));
    }

    #[test]
    fn truncated_flate_salvages_its_inflated_prefix() {
        // PDFium renders what it can from a truncated /FlateDecode stream; we
        // must return the successfully-inflated prefix rather than dropping the
        // whole page. Compress a long payload, then cut the compressed bytes
        // short of the stream end.
        let names = NameTable::new();
        let dict = dict_with_filter(&names, b"FlateDecode");
        let original: Vec<u8> = (0..8192u32)
            .map(|i| b"the quick brown fox "[(i % 20) as usize])
            .collect();
        let full = zlib_compress(&original);
        let truncated = &full[..full.len() * 3 / 4];

        let out = decode_stream(truncated, &dict, &names, &mut budget())
            .expect("a truncated flate stream should salvage its prefix, not error");
        assert!(!out.is_empty(), "expected a salvaged prefix");
        assert!(
            out.len() < original.len(),
            "a truncated stream cannot yield the whole payload"
        );
        assert_eq!(
            &out[..],
            &original[..out.len()],
            "salvaged bytes must be a correct prefix"
        );
    }

    // ── LZWDecode ───────────────────────────────────────────────────────────

    /// An MSB-first bit sink mirroring `lzw_decode`'s reader, for the test
    /// encoder below.
    struct BitWriter {
        out: Vec<u8>,
        buf: u32,
        cnt: u32,
    }
    impl BitWriter {
        fn new() -> Self {
            Self {
                out: Vec::new(),
                buf: 0,
                cnt: 0,
            }
        }
        fn write(&mut self, code: u32, width: u32) {
            self.buf = (self.buf << width) | (code & ((1 << width) - 1));
            self.cnt += width;
            while self.cnt >= 8 {
                self.cnt -= 8;
                self.out.push((self.buf >> self.cnt) as u8);
            }
        }
        fn finish(mut self) -> Vec<u8> {
            if self.cnt > 0 {
                self.out.push((self.buf << (8 - self.cnt)) as u8);
            }
            self.out
        }
    }

    /// An independent LZW encoder (separate logic from the decoder), so a
    /// round trip is a real cross-check rather than a shared bug. Width-step
    /// timing uses the same `next + early` rule as the decoder — the two must
    /// agree on it to interoperate; that they match Adobe for `early = 1` is
    /// what the corpus/pdfium pass validates.
    fn lzw_encode(data: &[u8], early_change: bool) -> Vec<u8> {
        use std::collections::HashMap;
        const CLEAR: u32 = 256;
        const EOD: u32 = 257;
        const FIRST: u32 = 258;
        const MAX: u32 = 4096;
        let early = u32::from(early_change);

        let seed =
            || -> HashMap<Vec<u8>, u32> { (0..256u32).map(|i| (vec![i as u8], i)).collect() };
        let mut dict = seed();
        let mut next = FIRST;
        let mut width = 9u32;
        let mut bw = BitWriter::new();
        bw.write(CLEAR, width);

        let mut w: Vec<u8> = Vec::new();
        for &b in data {
            let mut wc = w.clone();
            wc.push(b);
            if dict.contains_key(&wc) {
                w = wc;
                continue;
            }
            bw.write(dict[&w], width);
            // Freeze the table when full rather than emitting a CLEAR: this
            // mirrors the decoder's `next < MAX` freeze, so the round trip
            // exercises the shared 9→12-bit width-step logic without also
            // depending on encoder/decoder agreement on the CLEAR-reset
            // boundary. (Real Adobe streams do emit CLEAR at table-full;
            // the decoder's handling of that is validated against PDFium on
            // real corpus files, not by this self-round-trip.)
            if next < MAX {
                dict.insert(wc, next);
                next += 1;
                // One entry LATER than the decoder's `>=` bump: the decoder
                // builds its table one entry behind the encoder (it needs
                // the next code to complete an entry), so to emit each code
                // at the width the decoder will read it, the encoder must
                // defer its width step by exactly one assigned code.
                if width < 12 && next + early > (1 << width) {
                    width += 1;
                }
            }
            w = vec![b];
        }
        if !w.is_empty() {
            bw.write(dict[&w], width);
        }
        bw.write(EOD, width);
        bw.finish()
    }

    fn lzw_round_trip(data: &[u8]) {
        for early in [true, false] {
            let encoded = lzw_encode(data, early);
            let decoded = lzw_decode(&encoded, &mut budget(), early).unwrap();
            assert_eq!(
                decoded, data,
                "LZW round trip failed (early_change={early})"
            );
        }
    }

    #[test]
    fn lzw_round_trips_text_and_kwkwk() {
        lzw_round_trip(b"");
        lzw_round_trip(b"A");
        lzw_round_trip(b"TOBEORNOTTOBEORTOBEORNOT"); // the classic worked example
        lzw_round_trip(&[0x41u8; 4096]); // long run -> exercises the KwKwK case
    }

    #[test]
    fn lzw_round_trips_across_width_steps() {
        // A long, high-entropy input forces the table past 512, 1024 and 2048
        // entries, exercising every 9→10→11→12-bit width transition and the
        // table-full reset.
        let mut data = Vec::with_capacity(20_000);
        let mut x: u32 = 0x1234_5678;
        for _ in 0..20_000 {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            data.push((x >> 24) as u8);
        }
        lzw_round_trip(&data);
    }

    #[test]
    fn lzw_through_decode_stream_with_early_change_param() {
        let names = NameTable::new();
        // Default /EarlyChange (1): a bare /Filter /LZWDecode.
        let dict = dict_with_filter(&names, b"LZWDecode");
        let encoded = lzw_encode(b"hello hello hello world", true);
        assert_eq!(
            decode_stream(&encoded, &dict, &names, &mut budget()).unwrap(),
            b"hello hello hello world"
        );

        // /EarlyChange 0 via /DecodeParms must be honored.
        let parms = Dictionary::from_pairs([(names.intern(b"EarlyChange"), PdfObject::Integer(0))]);
        let dict0 = Dictionary::from_pairs([
            (
                names.known.filter,
                PdfObject::Name(names.intern(b"LZWDecode")),
            ),
            (
                names.known.decode_parms,
                PdfObject::Dictionary(std::sync::Arc::new(parms)),
            ),
        ]);
        let encoded0 = lzw_encode(b"hello hello hello world", false);
        assert_eq!(
            decode_stream(&encoded0, &dict0, &names, &mut budget()).unwrap(),
            b"hello hello hello world"
        );
    }

    #[test]
    fn lzw_truncated_stream_salvages_prefix() {
        // Chop the encoded stream before EOD: the decoder returns the bytes it
        // managed to decode rather than erroring the whole page away.
        let encoded = lzw_encode(b"TOBEORNOTTOBEORTOBEORNOT", true);
        let truncated = &encoded[..encoded.len() / 2];
        let out = lzw_decode(truncated, &mut budget(), true).unwrap();
        assert!(!out.is_empty(), "expected a salvaged prefix");
        assert!(b"TOBEORNOTTOBEORTOBEORNOT".starts_with(&out[..out.len().min(4)]));
    }

    #[test]
    fn lzw_budget_is_enforced() {
        let encoded = lzw_encode(&[0x41u8; 1024], true);
        let mut tiny = DecodeBudget::new(16);
        assert!(matches!(
            lzw_decode(&encoded, &mut tiny, true),
            Err(DecodeError::BudgetExceeded)
        ));
    }

    // ── Shared (hinted) decoding against the ordinary path ─────────────────

    /// Deterministic pseudo-random bytes with some structure, so deflate
    /// compresses them and PNG filters have something to undo.
    fn image_like(len: usize, seed: u32) -> Vec<u8> {
        let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..len)
            .map(|i| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // Smooth ramps plus a little noise.
                ((i / 7) as u8).wrapping_add(((x >> 28) as u8) & 3)
            })
            .collect()
    }

    /// PNG-filter `rows` of `row_len` bytes (`bpp` bytes per pixel), cycling
    /// the filter type per row through `tags` (an unknown tag is emitted as
    /// unfiltered, which is how decoders treat it).
    fn png_encode(data: &[u8], row_len: usize, bpp: usize, tags: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let zero = vec![0u8; row_len];
        for (r, row) in data.chunks(row_len).enumerate() {
            let prev = if r == 0 {
                &zero[..]
            } else {
                &data[(r - 1) * row_len..r * row_len]
            };
            let tag = tags[r % tags.len()];
            out.push(tag);
            for i in 0..row.len() {
                let left = if i >= bpp { row[i - bpp] } else { 0 };
                let up = prev[i];
                let up_left = if i >= bpp { prev[i - bpp] } else { 0 };
                let pred = match tag {
                    1 => left,
                    2 => up,
                    3 => ((u16::from(left) + u16::from(up)) / 2) as u8,
                    4 => crate::structure::predictor::paeth(left, up, up_left),
                    _ => 0,
                };
                out.push(row[i].wrapping_sub(pred));
            }
        }
        out
    }

    fn tiff_encode_8bit(data: &[u8], row_len: usize, bpp: usize) -> Vec<u8> {
        let mut out = data.to_vec();
        for row in out.chunks_exact_mut(row_len) {
            for i in (bpp..row.len()).rev() {
                row[i] = row[i].wrapping_sub(row[i - bpp]);
            }
        }
        out
    }

    fn stream_dict(names: &NameTable, filters: &[&[u8]], parms: Option<PdfObject>) -> Dictionary {
        let filter = if filters.len() == 1 {
            PdfObject::Name(names.intern(filters[0]))
        } else {
            PdfObject::Array(
                filters
                    .iter()
                    .map(|f| PdfObject::Name(names.intern(f)))
                    .collect(),
            )
        };
        let mut pairs = vec![(names.known.filter, filter)];
        if let Some(p) = parms {
            pairs.push((names.known.decode_parms, p));
        }
        Dictionary::from_pairs(pairs)
    }

    fn predictor_dict(
        names: &NameTable,
        predictor: i64,
        colors: i64,
        bpc: i64,
        columns: i64,
    ) -> PdfObject {
        PdfObject::Dictionary(Arc::new(Dictionary::from_pairs([
            (names.intern(b"Predictor"), PdfObject::Integer(predictor)),
            (names.intern(b"Colors"), PdfObject::Integer(colors)),
            (names.intern(b"BitsPerComponent"), PdfObject::Integer(bpc)),
            (names.intern(b"Columns"), PdfObject::Integer(columns)),
        ])))
    }

    /// One decode's observable outcome: bytes (or codec payload) or error,
    /// plus the budget it charged.
    type Outcome = (Result<(Vec<u8>, Option<String>), String>, usize);

    fn ordinary(raw: &[u8], dict: &Dictionary, names: &NameTable, budget: usize) -> Outcome {
        let mut b = DecodeBudget::new(budget);
        let r = decode_stream_to_codec(raw, dict, names, &mut b).map_err(|e| format!("{e:?}"));
        (r, b.used())
    }

    fn shared(
        raw: &[u8],
        dict: &Dictionary,
        names: &NameTable,
        budget: usize,
        hint: Option<usize>,
    ) -> Outcome {
        let mut b = DecodeBudget::new(budget);
        let r = decode_stream_to_codec_shared(raw, None, dict, names, &mut b, hint)
            .map(|d| match d {
                SharedDecoded::Bytes(bytes) => (bytes.to_vec(), None),
                SharedDecoded::Codec(bytes, name) => (bytes, Some(name)),
                SharedDecoded::CodecUnfiltered(name) => (raw.to_vec(), Some(name)),
                SharedDecoded::CodecShared(bytes, name) => (bytes.as_slice().to_vec(), Some(name)),
            })
            .map_err(|e| format!("{e:?}"));
        (r, b.used())
    }

    /// The shared path must match the ordinary one for every hint — right,
    /// short, long, absurd — and under budgets around the decode's need.
    fn assert_same_as_ordinary(label: &str, raw: &[u8], dict: &Dictionary, names: &NameTable) {
        let (reference, used) = ordinary(raw, dict, names, 1 << 30);
        let exact = reference
            .as_ref()
            .map(|(bytes, _)| bytes.len())
            .unwrap_or(4096);
        let hints = [
            None,
            Some(exact),
            Some(exact.saturating_sub(1)),
            Some(exact + 1),
            Some(exact / 2),
            Some(exact * 2),
            Some(1),
            Some(1 << 40),
        ];
        let budgets = [1 << 30, used, used.saturating_sub(1), used / 2, 0];
        for budget in budgets {
            let expected = ordinary(raw, dict, names, budget);
            for hint in hints {
                let got = shared(raw, dict, names, budget, hint);
                assert_eq!(
                    got, expected,
                    "{label}: budget {budget}, hint {hint:?} diverged from the ordinary path"
                );
            }
        }
    }

    fn flate_cases() -> Vec<(&'static str, Vec<u8>)> {
        let data = image_like(30_000, 7);
        let full = zlib_compress(&data);
        let mut corrupt_middle = full.clone();
        let mid = corrupt_middle.len() / 2;
        for b in &mut corrupt_middle[mid..mid + 6] {
            *b ^= 0x5a;
        }
        let mut bad_adler = full.clone();
        if let Some(last) = bad_adler.last_mut() {
            *last ^= 0xff;
        }
        let mut overlong_trailer = full.clone();
        overlong_trailer.extend_from_slice(b"garbage after the stream");
        let mut raw_deflate =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        raw_deflate.write_all(&data).unwrap();
        vec![
            ("well-formed", full.clone()),
            ("truncated 3/4", full[..full.len() * 3 / 4].to_vec()),
            ("missing adler", full[..full.len() - 4].to_vec()),
            ("half adler", full[..full.len() - 2].to_vec()),
            ("corrupt middle", corrupt_middle),
            ("bad adler", bad_adler),
            ("trailing garbage", overlong_trailer),
            ("raw deflate", raw_deflate.finish().unwrap()),
            ("not deflate", b"\xff\xff\xff\xff not a stream".to_vec()),
            ("empty", Vec::new()),
        ]
    }

    #[test]
    fn shared_flate_matches_ordinary_without_predictor() {
        let names = NameTable::new();
        let dict = dict_with_filter(&names, b"FlateDecode");
        for (label, raw) in flate_cases() {
            assert_same_as_ordinary(label, &raw, &dict, &names);
        }
    }

    #[test]
    fn shared_flate_matches_ordinary_with_png_predictor() {
        let names = NameTable::new();
        // 3 colours × 8 bits, 50 columns: 150-byte rows, 200 rows.
        let (row_len, bpp) = (150, 3);
        let data = image_like(row_len * 200, 11);
        let filtered = png_encode(&data, row_len, bpp, &[0, 1, 2, 3, 4, 2, 9]);
        let full = zlib_compress(&filtered);
        let dict = stream_dict(
            &names,
            &[b"FlateDecode"],
            Some(predictor_dict(&names, 15, 3, 8, 50)),
        );
        // The well-formed stream decodes to the original samples.
        let (ok, _) = shared(&full, &dict, &names, 1 << 30, Some(data.len()));
        assert_eq!(ok.unwrap().0, data);
        // A partial trailing row: the ordinary path keeps the complete rows.
        let mut ragged = filtered.clone();
        ragged.truncate(filtered.len() - 40);
        let mut cases = vec![
            ("png well-formed", full.clone()),
            ("png ragged tail", zlib_compress(&ragged)),
            ("png truncated", full[..full.len() / 2].to_vec()),
            ("png missing adler", full[..full.len() - 4].to_vec()),
        ];
        let mut corrupt = full.clone();
        let at = corrupt.len() / 3;
        corrupt[at] ^= 0x33;
        cases.push(("png corrupt", corrupt));
        for (label, raw) in cases {
            assert_same_as_ordinary(label, &raw, &dict, &names);
        }
        // Mismatched geometry: /Columns says 49, the hint 150-byte rows.
        let skewed = stream_dict(
            &names,
            &[b"FlateDecode"],
            Some(predictor_dict(&names, 12, 3, 8, 49)),
        );
        assert_same_as_ordinary("png skewed columns", &full, &skewed, &names);
        // A /DecodeParms array parallel to the filter array.
        let arrayed = stream_dict(
            &names,
            &[b"FlateDecode"],
            Some(PdfObject::Array(
                vec![predictor_dict(&names, 15, 3, 8, 50)].into(),
            )),
        );
        assert_same_as_ordinary("png parms array", &full, &arrayed, &names);
    }

    #[test]
    fn shared_flate_matches_ordinary_with_tiff_predictor() {
        let names = NameTable::new();
        let data = image_like(4 * 64 * 40, 3);
        let filtered = tiff_encode_8bit(&data, 4 * 64, 4);
        let full = zlib_compress(&filtered);
        let dict = stream_dict(
            &names,
            &[b"FlateDecode"],
            Some(predictor_dict(&names, 2, 4, 8, 64)),
        );
        let (ok, _) = shared(&full, &dict, &names, 1 << 30, Some(data.len()));
        assert_eq!(ok.unwrap().0, data);
        assert_same_as_ordinary("tiff 8-bit", &full, &dict, &names);
        assert_same_as_ordinary("tiff truncated", &full[..full.len() / 2], &dict, &names);
        // Sub-byte TIFF: 2-bit gray, 37 columns (a padded final byte).
        let packed = image_like(10 * 40, 5);
        let dict2 = stream_dict(
            &names,
            &[b"FlateDecode"],
            Some(predictor_dict(&names, 2, 1, 2, 37)),
        );
        assert_same_as_ordinary("tiff 2-bit", &zlib_compress(&packed), &dict2, &names);
        // An unsupported predictor fails the same way.
        let bad = stream_dict(
            &names,
            &[b"FlateDecode"],
            Some(predictor_dict(&names, 7, 1, 8, 1)),
        );
        assert_same_as_ordinary("predictor 7", &full, &bad, &names);
    }

    #[test]
    fn shared_lzw_and_run_length_match_ordinary() {
        let names = NameTable::new();
        let data = image_like(9_000, 13);
        let lzw = lzw_encode(&data, true);
        let dict = dict_with_filter(&names, b"LZWDecode");
        assert_same_as_ordinary("lzw", &lzw, &dict, &names);
        assert_same_as_ordinary("lzw truncated", &lzw[..lzw.len() / 2], &dict, &names);
        let mut bad = lzw.clone();
        bad[lzw.len() / 3] = 0xff;
        bad[lzw.len() / 3 + 1] = 0xff;
        assert_same_as_ordinary("lzw corrupt", &bad, &dict, &names);
        let tiff = lzw_encode(&tiff_encode_8bit(&data, 90, 3), true);
        let tiff_dict = stream_dict(
            &names,
            &[b"LZWDecode"],
            Some(predictor_dict(&names, 2, 3, 8, 30)),
        );
        assert_same_as_ordinary("lzw tiff", &tiff, &tiff_dict, &names);
        let png = lzw_encode(&png_encode(&data, 90, 3, &[1, 2, 4]), true);
        let png_dict = stream_dict(
            &names,
            &[b"LZWDecode"],
            Some(predictor_dict(&names, 12, 3, 8, 30)),
        );
        assert_same_as_ordinary("lzw png", &png, &png_dict, &names);

        let rl_dict = dict_with_filter(&names, b"RunLengthDecode");
        let rl: Vec<u8> = vec![2, b'a', b'b', b'c', 250, b'z', 0, b'q', 128];
        assert_same_as_ordinary("run length", &rl, &rl_dict, &names);
        assert_same_as_ordinary("run length overrun", &[5, b'a'], &rl_dict, &names);
        assert_same_as_ordinary("run length no eod", &rl[..rl.len() - 1], &rl_dict, &names);
    }

    #[test]
    fn shared_chains_codecs_and_raw_bodies_match_ordinary() {
        let names = NameTable::new();
        let data = image_like(5_000, 17);
        let deflated = zlib_compress(&data);
        let hex: Vec<u8> = deflated
            .iter()
            .flat_map(|b| format!("{b:02X}").into_bytes())
            .collect();
        let chain = stream_dict(&names, &[b"ASCIIHexDecode", b"FlateDecode"], None);
        assert_same_as_ordinary("hex then flate", &hex, &chain, &names);
        let codec = stream_dict(&names, &[b"FlateDecode", b"DCTDecode"], None);
        assert_same_as_ordinary("flate then dct", &deflated, &codec, &names);
        let codec_first = dict_with_filter(&names, b"DCTDecode");
        assert_same_as_ordinary("dct", &data, &codec_first, &names);
        assert_same_as_ordinary("unfiltered", &data, &Dictionary::new(), &names);

        // An unfiltered `Owned` body comes back as that very buffer.
        let body: Arc<[u8]> = Arc::from(data.clone());
        let mut b = budget();
        let out = decode_stream_to_codec_shared(
            &body,
            Some(&body),
            &Dictionary::new(),
            &names,
            &mut b,
            Some(data.len()),
        )
        .unwrap();
        let SharedDecoded::Bytes(out) = out else {
            panic!("no codec in the chain");
        };
        assert!(Arc::ptr_eq(&out, &body));
        assert_eq!(b.used(), data.len());
    }

    /// The exactly-sized path actually commits for the well-formed cases it
    /// exists for, and declines (charging nothing) otherwise.
    #[test]
    fn exact_path_commits_only_on_an_exact_clean_stream() {
        let names = NameTable::new();
        let data = image_like(150 * 40, 23);
        let png = zlib_compress(&png_encode(&data, 150, 3, &[4, 1, 2]));
        let parms = Dictionary::from_pairs([
            (names.intern(b"Predictor"), PdfObject::Integer(15)),
            (names.intern(b"Colors"), PdfObject::Integer(3)),
            (names.intern(b"Columns"), PdfObject::Integer(50)),
        ]);
        let mut b = budget();
        let out = decode_last_exact(
            b"FlateDecode",
            &png,
            Some(&parms),
            &names,
            &mut b,
            data.len(),
        )
        .expect("well-formed PNG-predicted flate commits");
        assert_eq!(&out[..], &data[..]);
        assert_eq!(b.used(), data.len() + 40, "charged the inflated length");
        assert_eq!(Arc::strong_count(&out), 1);

        let plain = zlib_compress(&data);
        for hint in [data.len() - 1, data.len() + 1] {
            let mut b = budget();
            assert!(
                decode_last_exact(b"FlateDecode", &plain, None, &names, &mut b, hint).is_none()
            );
            assert_eq!(b.used(), 0, "a declined attempt charges nothing");
        }
        let mut b = budget();
        assert!(
            decode_last_exact(
                b"FlateDecode",
                &plain[..plain.len() - 4],
                None,
                &names,
                &mut b,
                data.len()
            )
            .is_none(),
            "a stream without its checksum is left to the ordinary path"
        );
        let mut b = budget();
        let lzw = lzw_encode(&data, true);
        assert!(decode_last_exact(b"LZWDecode", &lzw, None, &names, &mut b, data.len()).is_some());
        let mut b = budget();
        assert!(
            decode_last_exact(b"RunLengthDecode", &[1, 7, 8, 128], None, &names, &mut b, 2)
                .is_some()
        );
        // Over budget: declined before allocating.
        let mut tight = DecodeBudget::new(data.len() - 1);
        assert!(
            decode_last_exact(b"FlateDecode", &plain, None, &names, &mut tight, data.len())
                .is_none()
        );
    }
}
