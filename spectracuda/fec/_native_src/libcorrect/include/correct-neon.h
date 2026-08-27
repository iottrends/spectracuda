#ifndef CORRECT_NEON_H
#define CORRECT_NEON_H
#include <correct.h>

/* ARM NEON-accelerated Viterbi decode for libcorrect's convolutional
 * code -- NOT part of upstream libcorrect (quiet/libcorrect has a
 * portable build and an x86-only SSE4.1 build, no ARM/NEON equivalent
 * -- see spectracuda/fec/_native.py's own module docstring). Written
 * for spectracuda specifically, reusing upstream's own portable
 * pair_lookup_t/history_buffer/bit_reader-writer machinery unchanged
 * (see src/convolutional/neon/decode.c's own comment for why only the
 * add-compare-select inner loop needed a new implementation) --
 * licensed under spectracuda's own MIT terms (see this repo's LICENSE),
 * not upstream libcorrect's BSD license, since this file has no
 * upstream counterpart to inherit that license from.
 *
 * Mirrors correct-sse.h's shape exactly: these instances should not be
 * used with the non-neon/sse functions, and non-neon instances should
 * not be used with the neon functions.
 */

struct correct_convolutional_neon;
typedef struct correct_convolutional_neon correct_convolutional_neon;

correct_convolutional_neon *correct_convolutional_neon_create(
    size_t rate, size_t order, const correct_convolutional_polynomial_t *poly);

void correct_convolutional_neon_destroy(correct_convolutional_neon *conv);

size_t correct_convolutional_neon_encode_len(correct_convolutional_neon *conv, size_t msg_len);

size_t correct_convolutional_neon_encode(correct_convolutional_neon *conv, const uint8_t *msg,
                                         size_t msg_len, uint8_t *encoded);

ssize_t correct_convolutional_neon_decode(correct_convolutional_neon *conv, const uint8_t *encoded,
                                          size_t num_encoded_bits, uint8_t *msg);

#endif
