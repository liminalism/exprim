//! Exercise the exported C ABI exactly as a C caller would: raw pointers,
//! status codes, library-owned buffers.

use std::ffi::{CStr, c_int, c_void};
use std::ptr;

use jbig2rs::*;

const WIDTH: u32 = 203;
const HEIGHT: u32 = 97;

/// One byte per pixel, 1 = black: a deterministic page of repeated marks.
fn pixels() -> Vec<u8> {
    (0..WIDTH * HEIGHT)
        .map(|i| {
            let (x, y) = (i % WIDTH, i / WIDTH);
            u8::from((x / 3 + y / 5) % 7 == 0 || (x % 11 == 0 && y % 13 < 6))
        })
        .collect()
}

/// Pack to MSB-first rows `stride` bytes apart, padding bits set to garbage
/// so that the library must ignore them.
fn pack(pixels: &[u8], stride: usize, zero_is_black: bool) -> Vec<u8> {
    let mut out = vec![0xA5u8; stride * HEIGHT as usize];
    for y in 0..HEIGHT as usize {
        for x in 0..WIDTH as usize {
            let bit = (pixels[y * WIDTH as usize + x] != 0) != zero_is_black;
            let byte = &mut out[y * stride + x / 8];
            if bit {
                *byte |= 0x80 >> (x % 8);
            } else {
                *byte &= !(0x80 >> (x % 8));
            }
        }
    }
    out
}

fn packed_image(data: &[u8], stride: usize, polarity: u32) -> Jbig2rsImage {
    Jbig2rsImage {
        width: WIDTH,
        height: HEIGHT,
        stride,
        data: data.as_ptr(),
        format: JBIG2RS_FORMAT_PACKED1,
        polarity,
        threshold: 0,
    }
}

fn options(preset: u32) -> Jbig2rsEncodeOptions {
    let mut options = Jbig2rsEncodeOptions {
        struct_size: 0,
        preset: 0,
        refine: 0,
    };
    unsafe { jbig2rs_encode_options_init(&mut options) };
    options.preset = preset;
    options
}

fn last_error() -> String {
    let message = jbig2rs_last_error_message();
    assert!(!message.is_null(), "a failure must leave a message");
    unsafe { CStr::from_ptr(message) }
        .to_string_lossy()
        .into_owned()
}

fn bytes(buffer: &Jbig2rsBuffer) -> &[u8] {
    unsafe { std::slice::from_raw_parts(buffer.data, buffer.len) }
}

/// Copy a decoded bitmap out as one byte per pixel, via packed rows.
fn unpack(bitmap: *const Jbig2rsBitmap) -> Vec<u8> {
    let (mut width, mut height) = (0u32, 0u32);
    assert_eq!(
        unsafe { jbig2rs_bitmap_size(bitmap, &mut width, &mut height) },
        JBIG2RS_OK
    );
    assert_eq!((width, height), (WIDTH, HEIGHT));
    let stride = (WIDTH as usize).div_ceil(8) + 3;
    let mut rows = vec![0u8; stride * HEIGHT as usize];
    let status = unsafe {
        jbig2rs_bitmap_copy_rows(
            bitmap,
            rows.as_mut_ptr(),
            rows.len(),
            stride,
            JBIG2RS_ONE_IS_BLACK,
        )
    };
    assert_eq!(status, JBIG2RS_OK);
    (0..HEIGHT as usize)
        .flat_map(|y| {
            let rows = &rows;
            (0..WIDTH as usize).map(move |x| (rows[y * stride + x / 8] >> (7 - x % 8)) & 1)
        })
        .collect()
}

fn decode_file_page(file: &[u8]) -> Vec<u8> {
    let mut document = ptr::null_mut();
    let status =
        unsafe { jbig2rs_decode_file(file.as_ptr(), file.len(), ptr::null(), &mut document) };
    assert_eq!(status, JBIG2RS_OK, "{}", last_error());
    assert_eq!(unsafe { jbig2rs_document_page_count(document) }, 1);
    assert!(unsafe { jbig2rs_document_page(document, 1) }.is_null());
    let page = unpack(unsafe { jbig2rs_document_page(document, 0) });
    unsafe { jbig2rs_document_free(document) };
    page
}

#[test]
fn versions_are_reported() {
    let version = unsafe { CStr::from_ptr(jbig2rs_version()) }
        .to_str()
        .unwrap();
    assert!(version.starts_with("0."), "{version}");
    assert_eq!(jbig2rs_abi_version() >> 16, JBIG2RS_ABI_MAJOR);
}

#[test]
fn packed_input_round_trips_through_a_file_in_both_polarities() {
    let source = pixels();
    for (polarity, zero_is_black) in [(JBIG2RS_ONE_IS_BLACK, false), (JBIG2RS_ZERO_IS_BLACK, true)]
    {
        let stride = 32;
        let data = pack(&source, stride, zero_is_black);
        let image = packed_image(&data, stride, polarity);
        let options = options(JBIG2RS_PRESET_GENERIC);
        let mut out = Jbig2rsBuffer {
            data: ptr::null_mut(),
            len: 0,
        };
        let status = unsafe { jbig2rs_encode_file(&image, &options, &mut out) };
        assert_eq!(status, JBIG2RS_OK, "{}", last_error());
        assert!(jbig2rs_last_error_message().is_null());
        assert_eq!(&bytes(&out)[..8], b"\x97JB2\r\n\x1a\n");
        assert_eq!(decode_file_page(bytes(&out)), source);
        unsafe { jbig2rs_buffer_free(&mut out) };
        assert!(out.data.is_null());
        unsafe { jbig2rs_buffer_free(&mut out) }; // a second free is inert
    }
}

#[test]
fn pdf_fragments_decode_against_their_globals() {
    let source = pixels();
    let data = pack(&source, 26, false);
    let image = packed_image(&data, 26, JBIG2RS_ONE_IS_BLACK);
    let empty = || Jbig2rsBuffer {
        data: ptr::null_mut(),
        len: 0,
    };
    let (mut globals_bytes, mut page_bytes) = (empty(), empty());
    // NULL options: the default (symbol) preset.
    let status =
        unsafe { jbig2rs_encode_pdf(&image, ptr::null(), &mut globals_bytes, &mut page_bytes) };
    assert_eq!(status, JBIG2RS_OK, "{}", last_error());
    assert!(page_bytes.len > 0);

    let mut globals = ptr::null_mut();
    if globals_bytes.len > 0 {
        let status = unsafe {
            jbig2rs_globals_decode(
                globals_bytes.data,
                globals_bytes.len,
                ptr::null(),
                &mut globals,
            )
        };
        assert_eq!(status, JBIG2RS_OK, "{}", last_error());
    }
    let mut bitmap = ptr::null_mut();
    let status = unsafe {
        jbig2rs_decode_pdf_page(
            globals,
            page_bytes.data,
            page_bytes.len,
            ptr::null(),
            &mut bitmap,
        )
    };
    assert_eq!(status, JBIG2RS_OK, "{}", last_error());
    // Symbol coding is lossy in general, but these marks are exact repeats.
    assert_eq!(unpack(bitmap), source);
    unsafe {
        jbig2rs_bitmap_free(bitmap);
        jbig2rs_globals_free(globals);
        jbig2rs_buffer_free(&mut globals_bytes);
        jbig2rs_buffer_free(&mut page_bytes);
    }
}

#[test]
fn a_multi_page_encoder_shares_one_dictionary_and_is_spent_after_finishing() {
    let source = pixels();
    let gray: Vec<u8> = source
        .iter()
        .map(|&p| if p != 0 { 10 } else { 240 })
        .collect();
    let image = Jbig2rsImage {
        width: WIDTH,
        height: HEIGHT,
        stride: WIDTH as usize,
        data: gray.as_ptr(),
        format: JBIG2RS_FORMAT_GRAY8,
        polarity: JBIG2RS_ONE_IS_BLACK,
        threshold: 128,
    };
    let options = options(JBIG2RS_PRESET_TEXT);
    let mut encoder = ptr::null_mut();
    assert_eq!(
        unsafe { jbig2rs_encoder_new(&options, &mut encoder) },
        JBIG2RS_OK
    );
    for _ in 0..3 {
        assert_eq!(
            unsafe { jbig2rs_encoder_add_page(encoder, &image) },
            JBIG2RS_OK,
            "{}",
            last_error()
        );
    }
    assert_eq!(unsafe { jbig2rs_encoder_page_count(encoder) }, 3);

    let mut output = ptr::null_mut();
    assert_eq!(
        unsafe { jbig2rs_encoder_finish_pdf(encoder, &mut output) },
        JBIG2RS_OK
    );
    assert_eq!(unsafe { jbig2rs_pdf_output_page_count(output) }, 3);
    let (mut data, mut len) = (ptr::null(), 0usize);
    assert_eq!(
        unsafe { jbig2rs_pdf_output_globals(output, &mut data, &mut len) },
        JBIG2RS_OK
    );
    assert!(
        len > 0,
        "three identical text pages should share a dictionary"
    );
    let mut globals = ptr::null_mut();
    assert_eq!(
        unsafe { jbig2rs_globals_decode(data, len, ptr::null(), &mut globals) },
        JBIG2RS_OK
    );
    for index in 0..3 {
        assert_eq!(
            unsafe { jbig2rs_pdf_output_page(output, index, &mut data, &mut len) },
            JBIG2RS_OK
        );
        let mut bitmap = ptr::null_mut();
        let status =
            unsafe { jbig2rs_decode_pdf_page(globals, data, len, ptr::null(), &mut bitmap) };
        assert_eq!(status, JBIG2RS_OK, "{}", last_error());
        assert_eq!(unpack(bitmap), source);
        unsafe { jbig2rs_bitmap_free(bitmap) };
    }
    assert_eq!(
        unsafe { jbig2rs_pdf_output_page(output, 3, &mut data, &mut len) },
        JBIG2RS_ERR_INVALID_ARGUMENT
    );

    // Spent: every further call is refused, and freeing is still required.
    assert_eq!(
        unsafe { jbig2rs_encoder_add_page(encoder, &image) },
        JBIG2RS_ERR_BAD_STATE
    );
    let mut buffer = Jbig2rsBuffer {
        data: ptr::null_mut(),
        len: 0,
    };
    assert_eq!(
        unsafe { jbig2rs_encoder_finish_file(encoder, &mut buffer) },
        JBIG2RS_ERR_BAD_STATE
    );
    assert!(buffer.data.is_null());
    unsafe {
        jbig2rs_globals_free(globals);
        jbig2rs_pdf_output_free(output);
        jbig2rs_encoder_free(encoder);
    }
}

struct Collected {
    groups: Vec<(usize, usize, usize)>,
    abort_at: Option<usize>,
}

unsafe extern "C" fn collect(user: *mut c_void, group: *const Jbig2rsPdfGroup) -> c_int {
    let collected = unsafe { &mut *user.cast::<Collected>() };
    let output = unsafe { jbig2rs_pdf_group_output(group) };
    collected.groups.push(unsafe {
        (
            jbig2rs_pdf_group_index(group),
            jbig2rs_pdf_group_first_page(group),
            jbig2rs_pdf_output_page_count(output),
        )
    });
    match collected.abort_at {
        Some(index) if index + 1 == collected.groups.len() => 7,
        _ => 0,
    }
}

#[test]
fn a_group_encoder_emits_bounded_groups_and_honours_an_abort() {
    let source = pixels();
    let data = pack(&source, 26, false);
    let image = packed_image(&data, 26, JBIG2RS_ONE_IS_BLACK);
    let options = options(JBIG2RS_PRESET_TEXT);

    let mut collected = Collected {
        groups: Vec::new(),
        abort_at: None,
    };
    let mut encoder = ptr::null_mut();
    let user = (&raw mut collected).cast::<c_void>();
    let status =
        unsafe { jbig2rs_group_encoder_new(&options, 2, 0, Some(collect), user, &mut encoder) };
    assert_eq!(status, JBIG2RS_OK, "{}", last_error());
    for _ in 0..5 {
        assert_eq!(
            unsafe { jbig2rs_group_encoder_add_page(encoder, &image) },
            JBIG2RS_OK
        );
    }
    assert_eq!(unsafe { jbig2rs_group_encoder_finish(encoder) }, JBIG2RS_OK);
    assert_eq!(collected.groups, vec![(0, 0, 2), (1, 2, 2), (2, 4, 1)]);
    assert_eq!(
        unsafe { jbig2rs_group_encoder_finish(encoder) },
        JBIG2RS_ERR_BAD_STATE
    );
    unsafe { jbig2rs_group_encoder_free(encoder) };

    let mut collected = Collected {
        groups: Vec::new(),
        abort_at: Some(0),
    };
    let user = (&raw mut collected).cast::<c_void>();
    let status =
        unsafe { jbig2rs_group_encoder_new(&options, 1, 0, Some(collect), user, &mut encoder) };
    assert_eq!(status, JBIG2RS_OK);
    assert_eq!(
        unsafe { jbig2rs_group_encoder_add_page(encoder, &image) },
        JBIG2RS_ERR_ABORTED
    );
    assert!(last_error().contains("returned 7"));
    assert_eq!(
        unsafe { jbig2rs_group_encoder_add_page(encoder, &image) },
        JBIG2RS_ERR_BAD_STATE
    );
    unsafe { jbig2rs_group_encoder_free(encoder) };
}

#[test]
fn bad_arguments_map_to_their_status_codes() {
    let source = pixels();
    let data = pack(&source, 26, false);
    let mut out = Jbig2rsBuffer {
        data: ptr::null_mut(),
        len: 0,
    };
    let call = |image: &Jbig2rsImage,
                options: *const Jbig2rsEncodeOptions,
                out: &mut Jbig2rsBuffer| unsafe {
        jbig2rs_encode_file(image, options, out)
    };

    let good = packed_image(&data, 26, JBIG2RS_ONE_IS_BLACK);
    assert_eq!(
        unsafe { jbig2rs_encode_file(ptr::null(), ptr::null(), &mut out) },
        JBIG2RS_ERR_NULL_ARGUMENT
    );
    assert_eq!(
        unsafe { jbig2rs_encode_file(&good, ptr::null(), ptr::null_mut()) },
        JBIG2RS_ERR_NULL_ARGUMENT
    );

    let short_stride = Jbig2rsImage { stride: 25, ..good };
    assert_eq!(
        call(&short_stride, ptr::null(), &mut out),
        JBIG2RS_ERR_INVALID_ARGUMENT
    );
    assert!(last_error().contains("stride"));
    let bad_format = Jbig2rsImage { format: 9, ..good };
    assert_eq!(
        call(&bad_format, ptr::null(), &mut out),
        JBIG2RS_ERR_INVALID_ARGUMENT
    );
    let bad_polarity = Jbig2rsImage {
        polarity: 9,
        ..good
    };
    assert_eq!(
        call(&bad_polarity, ptr::null(), &mut out),
        JBIG2RS_ERR_INVALID_ARGUMENT
    );
    let empty = Jbig2rsImage { width: 0, ..good };
    assert_eq!(
        call(&empty, ptr::null(), &mut out),
        JBIG2RS_ERR_INVALID_ARGUMENT
    );
    let no_data = Jbig2rsImage {
        data: ptr::null(),
        ..good
    };
    assert_eq!(
        call(&no_data, ptr::null(), &mut out),
        JBIG2RS_ERR_NULL_ARGUMENT
    );

    let mut bad = options(99);
    assert_eq!(call(&good, &bad, &mut out), JBIG2RS_ERR_INVALID_ARGUMENT);
    bad = options(JBIG2RS_PRESET_GENERIC);
    bad.refine = 1;
    assert_eq!(call(&good, &bad, &mut out), JBIG2RS_ERR_INVALID_CONFIG);
    assert!(last_error().contains("refinement"));
    bad.refine = 2;
    assert_eq!(call(&good, &bad, &mut out), JBIG2RS_ERR_INVALID_ARGUMENT);
    bad.struct_size = 0;
    assert_eq!(call(&good, &bad, &mut out), JBIG2RS_ERR_INVALID_ARGUMENT);

    // An older caller whose struct ends before `refine`: the field is not read.
    let mut older = options(JBIG2RS_PRESET_GENERIC);
    older.refine = 1;
    older.struct_size = 8;
    assert_eq!(
        call(&good, &older, &mut out),
        JBIG2RS_OK,
        "{}",
        last_error()
    );
    unsafe { jbig2rs_buffer_free(&mut out) };

    let garbage = b"not a jbig2 stream at all";
    let mut document = ptr::null_mut();
    let status =
        unsafe { jbig2rs_decode_file(garbage.as_ptr(), garbage.len(), ptr::null(), &mut document) };
    assert_eq!(status, JBIG2RS_ERR_DECODE_MALFORMED);
    assert!(document.is_null());
}

#[test]
fn decode_limits_are_enforced() {
    let source = pixels();
    let data = pack(&source, 26, false);
    let image = packed_image(&data, 26, JBIG2RS_ONE_IS_BLACK);
    let options = options(JBIG2RS_PRESET_GENERIC);
    let mut out = Jbig2rsBuffer {
        data: ptr::null_mut(),
        len: 0,
    };
    assert_eq!(
        unsafe { jbig2rs_encode_file(&image, &options, &mut out) },
        JBIG2RS_OK
    );

    let mut limits = Jbig2rsDecodeOptions {
        struct_size: 0,
        max_page_pixels: 0,
    };
    unsafe { jbig2rs_decode_options_init(&mut limits) };
    limits.max_page_pixels = u64::from(WIDTH * HEIGHT) - 1;
    let mut document = ptr::null_mut();
    let status = unsafe { jbig2rs_decode_file(out.data, out.len, &limits, &mut document) };
    assert_eq!(status, JBIG2RS_ERR_DECODE_LIMIT, "{}", last_error());
    unsafe { jbig2rs_buffer_free(&mut out) };
}

#[test]
fn error_messages_are_thread_local() {
    let message = std::thread::spawn(|| {
        let garbage = b"garbage";
        let mut document = ptr::null_mut();
        let status = unsafe {
            jbig2rs_decode_file(garbage.as_ptr(), garbage.len(), ptr::null(), &mut document)
        };
        assert_ne!(status, JBIG2RS_OK);
        last_error()
    })
    .join()
    .unwrap();
    assert!(!message.is_empty());
    assert!(jbig2rs_last_error_message().is_null());
}
