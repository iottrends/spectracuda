#include "correct/convolutional/convolutional.h"
// BIG HEAPING TODO sort out the include mess (same TODO the SSE header
// left itself -- see sse/convolutional.h)
#include "correct-neon.h"
#include <arm_neon.h>

/* Unlike correct_convolutional_sse (sse/convolutional.h), which adds an
 * SSE-specific oct_lookup_t member because its ACS inner loop uses its
 * own 8-wide packed lookup table, this struct is JUST the portable
 * correct_convolutional -- the NEON inner loop (see neon/decode.c)
 * reuses pair_lookup_t (conv->pair_lookup, already built by the
 * portable _convolutional_decode_init()) completely unchanged. No new
 * per-instance state is needed. */
struct correct_convolutional_neon {
    correct_convolutional base_conv;
};
