#include "correct/convolutional/neon/convolutional.h"

correct_convolutional_neon *correct_convolutional_neon_create(size_t rate,
                                                               size_t order,
                                                               const polynomial_t *poly) {
    correct_convolutional_neon *conv = malloc(sizeof(correct_convolutional_neon));
    correct_convolutional *init_conv = _correct_convolutional_init(&conv->base_conv, rate, order, poly);
    if (!init_conv) {
        free(conv);
        conv = NULL;
    }
    return conv;
}

void correct_convolutional_neon_destroy(correct_convolutional_neon *conv) {
    // Unlike correct_convolutional_sse_destroy (which must separately
    // oct_lookup_destroy() its own SSE-specific lookup table before
    // tearing down base_conv), there is nothing NEON-specific to free
    // here -- see this file's own header comment: no new per-instance
    // state beyond the portable base_conv exists.
    _correct_convolutional_teardown(&conv->base_conv);
    free(conv);
}
