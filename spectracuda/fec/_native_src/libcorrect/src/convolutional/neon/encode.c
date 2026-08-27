#include "correct/convolutional/neon/convolutional.h"

/* Same reasoning as sse/encode.c: libcorrect's encoder is a simple
 * shift-register convolution, not the Viterbi add-compare-select
 * bottleneck -- no separate speedup claim here, just call straight
 * through to the identical portable correct_convolutional_encode(). */

size_t correct_convolutional_neon_encode_len(correct_convolutional_neon *conv, size_t msg_len) {
    return correct_convolutional_encode_len(&conv->base_conv, msg_len);
}

size_t correct_convolutional_neon_encode(correct_convolutional_neon *conv, const uint8_t *msg,
                                         size_t msg_len, uint8_t *encoded) {
    return correct_convolutional_encode(&conv->base_conv, msg, msg_len, encoded);
}
