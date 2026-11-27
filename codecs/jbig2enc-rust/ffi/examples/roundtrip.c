/*
 * roundtrip.c -- encode a packed 1bpp image with jbig2rs, decode it again,
 * and check every pixel survived.
 *
 * Build (after `cargo build -p jbig2enc-ffi --release`):
 *   MSVC:  cl /I include examples\roundtrip.c ..\target\release\jbig2rs.dll.lib
 *   Linux: cc -I include examples/roundtrip.c -L ../target/release -ljbig2rs
 *   macOS: cc -I include examples/roundtrip.c -L ../target/release -ljbig2rs
 * Run with the shared library on the loader path.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jbig2rs.h"

#define WIDTH 300
#define HEIGHT 120

static int check(int status, const char *what) {
    if (status != JBIG2RS_OK) {
        const char *message = jbig2rs_last_error_message();
        fprintf(stderr, "%s failed (%d): %s\n", what, status, message ? message : "?");
        exit(1);
    }
    return status;
}

int main(void) {
    /* A packed, min-is-white (1 = black) page with padded rows, as a TIFF
     * strip or a Leptonica PIX would hold it. */
    const size_t stride = (WIDTH + 7) / 8 + 4;
    unsigned char *page = (unsigned char *)calloc(stride * HEIGHT, 1);
    if (!page) return 1;
    for (int y = 0; y < HEIGHT; y++)
        for (int x = 0; x < WIDTH; x++)
            if ((x / 4 + y / 6) % 5 == 0 || (x % 17 == 0 && y % 11 < 7))
                page[y * stride + x / 8] |= (unsigned char)(0x80 >> (x % 8));

    printf("jbig2rs %s, ABI %u.%u\n", jbig2rs_version(), jbig2rs_abi_version() >> 16,
           jbig2rs_abi_version() & 0xFFFF);

    Jbig2rsImage image = {WIDTH, HEIGHT, stride, page, JBIG2RS_FORMAT_PACKED1,
                          JBIG2RS_ONE_IS_BLACK, 0};
    Jbig2rsEncodeOptions options;
    jbig2rs_encode_options_init(&options);
    options.preset = JBIG2RS_PRESET_LOSSLESS;

    /* A standalone .jb2 file. */
    Jbig2rsBuffer file = {0};
    check(jbig2rs_encode_file(&image, &options, &file), "encode");
    printf("encoded %dx%d (%zu bytes packed) into %zu bytes\n", WIDTH, HEIGHT,
           stride * HEIGHT, file.len);

    /* Decode it, and copy the page out in the same layout it came in. */
    Jbig2rsDocument *document = NULL;
    check(jbig2rs_decode_file(file.data, file.len, NULL, &document), "decode");
    const Jbig2rsBitmap *bitmap = jbig2rs_document_page(document, 0);
    uint32_t width = 0, height = 0;
    check(jbig2rs_bitmap_size(bitmap, &width, &height), "size");
    unsigned char *decoded = (unsigned char *)calloc(stride * HEIGHT, 1);
    if (!decoded) return 1;
    check(jbig2rs_bitmap_copy_rows(bitmap, decoded, stride * HEIGHT, stride, JBIG2RS_ONE_IS_BLACK),
          "copy rows");

    int mismatches = 0;
    for (int y = 0; y < HEIGHT; y++)
        for (int x = 0; x < WIDTH; x++) {
            int a = (page[y * stride + x / 8] >> (7 - x % 8)) & 1;
            int b = (decoded[y * stride + x / 8] >> (7 - x % 8)) & 1;
            mismatches += a != b;
        }
    printf("decoded %ux%u, %d mismatched pixels\n", width, height, mismatches);

    jbig2rs_document_free(document);
    jbig2rs_buffer_free(&file);
    free(decoded);
    free(page);
    return mismatches == 0 && width == WIDTH && height == HEIGHT ? 0 : 1;
}
