#include "correct/convolutional/neon/convolutional.h"

/* NEON-accelerated add-compare-select (ACS) inner loop for conv_v27's
 * Viterbi decode -- NOT vendored from upstream libcorrect (no such file
 * exists there, see correct-neon.h's own header comment for why).
 * Written for spectracuda specifically, as a port of the algorithm in
 * the portable ../decode.c's own convolutional_decode_inner() -- NOT a
 * translation of sse/decode.c's own inner loop (that file's
 * oct_lookup/shuffle-mask approach is a much more aggressive, harder-
 * to-verify-blind design).
 *
 * This is the SECOND attempt at this kernel. The first (4-lane,
 * uint16x4_t) attempt measured ~2.1x SLOWER than the portable build on
 * real Pi 5 hardware, despite being bit-exact correct -- seven small
 * stack arrays' worth of memory round-tripping (four for the gathered
 * inputs, two more for staging the vector results back out to scalar
 * before the final copy into write_errors[]/history[]) around only 4
 * lanes of real vector math. See this project's git history for that
 * version and its measured numbers.
 *
 * This version widens to 8 lanes (uint16x8_t, covering two of the
 * portable loop's outer-loop steps -- 8 base_offsets / 16 successor
 * states -- per NEON sequence) AND removes as much of that round-
 * tripping as the data dependencies allow:
 *
 *   - read_errors[base..base+7] and read_errors[highbase+base..+7] are
 *     already contiguous in memory (base_offset is a plain array
 *     index, not a gather), so low_past_v/high_past_v are loaded
 *     straight off conv->errors->read_errors with vld1q_u16 -- no
 *     staging array at all.
 *
 *   - the distance-pair gather itself (pair_lookup.keys[...] ->
 *     pair_lookup.distances[key]) IS data-dependent, so it must stay a
 *     scalar loop -- ARMv8's base NEON/ASIMD has no general gather
 *     instruction. But rather than manually splitting each 32-bit
 *     concat value into lo/hi halves with scalar shifts/masks into 4
 *     separate arrays (the first attempt's approach), the raw uint32_t
 *     concat values are gathered into ONE small scratch array per side
 *     and then de-interleaved into two uint16x8_t vectors (lo lanes,
 *     hi lanes) with a single vld2q_u16 -- the packed layout
 *     (lo0,hi0,lo1,hi1,...) is exactly what vld2 de-interleaves.
 *
 *   - on the output side, the "successor" (low+2*bo, even) and "plus
 *     one" (low+2*bo+1, odd) results are exactly two interleaved
 *     streams over write_errors[low..low+15] / history[low..low+15],
 *     so they're written back with a single vst2q_u16 (errors) /
 *     vst2_u8 (history, after vmovn_u16 narrowing the 0/1 mask) --
 *     no scalar copy-out loop at all.
 *
 * That leaves exactly two small stack arrays (the raw gathered concat
 * values), which is the least this loop can do given no gather
 * instruction exists -- everything else routes through vector
 * registers only. Falls back to the original 4-lane grouping for any
 * tail that doesn't divide evenly into 8 (not exercised by this
 * project's fixed K=7/order=7 code, where highbase=32 always divides
 * evenly, but kept for any other order).
 *
 * Reuses the portable build's pair_lookup_t (conv->pair_lookup) and
 * history_buffer/error_buffer machinery COMPLETELY UNCHANGED -- only
 * the ACS math itself is replaced by NEON vector add/compare/select.
 *
 * Deliberately kept in plain 16-bit lanes (uint16x8_t), not widened to
 * 32-bit: distance_t is uint16_t, and the portable scalar code computes
 * `low_error`/`high_error` (also distance_t, i.e. implicitly truncating
 * mod 65536 on assignment) -- doing the add directly in uint16x8_t
 * lanes reproduces that exact mod-65536 wraparound with plain NEON
 * integer arithmetic, rather than needing an explicit widen-then-
 * truncate dance to match it.
 *
 * MEASURED, not assumed, on the Pi 5 (see spectracuda/fec/_native.py's
 * neon_available() gate and tests/test_fec_native_acceleration.py's
 * correctness sweep, mirrored for this path -- must pass there before
 * any speed number from this file is trusted).
 */

static inline void convolutional_neon_decode_inner(correct_convolutional *conv, unsigned int sets,
                                                    const uint8_t *soft) {
    shift_register_t highbit = 1 << (conv->order - 1);
    for (unsigned int i = conv->order - 1; i < (sets - conv->order + 1); i++) {
        distance_t *distances = conv->distances;
        if (soft) {
            if (conv->soft_measurement == CORRECT_SOFT_LINEAR) {
                for (unsigned int j = 0; j < 1 << (conv->rate); j++) {
                    distances[j] = metric_soft_distance_linear(j, soft + i * conv->rate, conv->rate);
                }
            } else {
                for (unsigned int j = 0; j < 1 << (conv->rate); j++) {
                    distances[j] = metric_soft_distance_quadratic(j, soft + i * conv->rate, conv->rate);
                }
            }
        } else {
            unsigned int out = bit_reader_read(conv->bit_reader, conv->rate);
            for (unsigned int j = 0; j < 1 << (conv->rate); j++) {
                distances[j] = metric_distance(j, out);
            }
        }
        pair_lookup_t pair_lookup = conv->pair_lookup;
        pair_lookup_fill_distance(pair_lookup, distances);

        unsigned int num_iter = highbit << 1;
        const distance_t *read_errors = conv->errors->read_errors;
        distance_t *write_errors = conv->errors->write_errors;
        uint8_t *history = history_buffer_get_slice(conv->history_buffer);

        shift_register_t highbase = highbit >> 1;
        shift_register_t low = 0, high = highbit, base = 0;

        // -- Main loop: 8 base_offsets (16 successor states) per NEON
        // sequence --
        for (; high + 8 <= num_iter; low += 16, high += 16, base += 8) {
            // Gather this step's 8 base_offset lanes worth of packed
            // distance-pair concat values -- data-dependent indexed
            // loads, stays scalar (see this file's own top comment).
            // This is the one unavoidable round trip through memory:
            // no ARMv8 ASIMD gather instruction exists to vectorize it.
            uint32_t low_concat[8], high_concat[8];
            for (unsigned int bo = 0; bo < 8; bo++) {
                distance_pair_key_t low_key = pair_lookup.keys[base + bo];
                distance_pair_key_t high_key = pair_lookup.keys[highbase + base + bo];
                low_concat[bo] = pair_lookup.distances[low_key];
                high_concat[bo] = pair_lookup.distances[high_key];
            }
            // De-interleave each packed (lo0,hi0,lo1,hi1,...) array into
            // its lo-lane and hi-lane vectors in one instruction.
            uint16x8x2_t low_pair = vld2q_u16((const uint16_t *)low_concat);
            uint16x8x2_t high_pair = vld2q_u16((const uint16_t *)high_concat);
            uint16x8_t low_lo_v = low_pair.val[0], low_hi_v = low_pair.val[1];
            uint16x8_t high_lo_v = high_pair.val[0], high_hi_v = high_pair.val[1];

            // read_errors[base..] / read_errors[highbase+base..] are
            // plain contiguous array slices (base_offset is a real
            // index here, not a gather key) -- load directly, no
            // staging array needed.
            uint16x8_t low_past_v = vld1q_u16(&read_errors[base]);
            uint16x8_t high_past_v = vld1q_u16(&read_errors[highbase + base]);

            // -- "successor" stream (low+0, low+2, ..., low+14) --
            uint16x8_t low_error_v = vaddq_u16(low_lo_v, low_past_v);
            uint16x8_t high_error_v = vaddq_u16(high_lo_v, high_past_v);
            uint16x8_t le_mask = vcleq_u16(low_error_v, high_error_v);
            uint16x8_t error_v = vbslq_u16(le_mask, low_error_v, high_error_v);
            uint16x8_t hist_v = vbslq_u16(le_mask, vdupq_n_u16(0), vdupq_n_u16(1));

            // -- "plus one" stream (low+1, low+3, ..., low+15) -- same
            // shape, using the packed distance pair's hi lanes --
            uint16x8_t low_error2_v = vaddq_u16(low_hi_v, low_past_v);
            uint16x8_t high_error2_v = vaddq_u16(high_hi_v, high_past_v);
            uint16x8_t le_mask2 = vcleq_u16(low_error2_v, high_error2_v);
            uint16x8_t error2_v = vbslq_u16(le_mask2, low_error2_v, high_error2_v);
            uint16x8_t hist2_v = vbslq_u16(le_mask2, vdupq_n_u16(0), vdupq_n_u16(1));

            // The successor/plus-one streams are exactly the even/odd
            // interleaving of write_errors[low..low+15] and
            // history[low..low+15] -- write both back with a single
            // interleaved store each, no scalar copy-out loop.
            uint16x8x2_t error_pair = {{error_v, error2_v}};
            vst2q_u16(&write_errors[low], error_pair);
            uint8x8x2_t hist_pair = {{vmovn_u16(hist_v), vmovn_u16(hist2_v)}};
            vst2_u8(&history[low], hist_pair);
        }

        // -- Tail: any remainder that doesn't divide evenly into 8
        // falls back to the original 4-lane grouping (never exercised
        // for this project's fixed order=7 code, where highbase=32
        // always divides evenly by 8, but kept for generality). --
        for (; high < num_iter; low += 8, high += 8, base += 4) {
            uint16_t low_lo[4], low_hi[4], high_lo[4], high_hi[4];
            uint16_t low_past[4], high_past[4];
            for (unsigned int bo = 0; bo < 4; bo++) {
                distance_pair_key_t low_key = pair_lookup.keys[base + bo];
                distance_pair_key_t high_key = pair_lookup.keys[highbase + base + bo];
                distance_pair_t low_concat = pair_lookup.distances[low_key];
                distance_pair_t high_concat = pair_lookup.distances[high_key];
                low_lo[bo] = (uint16_t)(low_concat & 0xffff);
                low_hi[bo] = (uint16_t)(low_concat >> 16);
                high_lo[bo] = (uint16_t)(high_concat & 0xffff);
                high_hi[bo] = (uint16_t)(high_concat >> 16);
                low_past[bo] = read_errors[base + bo];
                high_past[bo] = read_errors[highbase + base + bo];
            }
            uint16x4_t low_past_v = vld1_u16(low_past);
            uint16x4_t high_past_v = vld1_u16(high_past);

            uint16x4_t low_error_v = vadd_u16(vld1_u16(low_lo), low_past_v);
            uint16x4_t high_error_v = vadd_u16(vld1_u16(high_lo), high_past_v);
            uint16x4_t le_mask = vcle_u16(low_error_v, high_error_v);
            uint16x4_t error_v = vbsl_u16(le_mask, low_error_v, high_error_v);
            uint16x4_t hist_v = vbsl_u16(le_mask, vdup_n_u16(0), vdup_n_u16(1));
            uint16_t error_arr[4], hist_arr[4];
            vst1_u16(error_arr, error_v);
            vst1_u16(hist_arr, hist_v);
            for (unsigned int bo = 0; bo < 4; bo++) {
                shift_register_t successor = low + 2 * bo;
                write_errors[successor] = error_arr[bo];
                history[successor] = (uint8_t)hist_arr[bo];
            }

            uint16x4_t low_error2_v = vadd_u16(vld1_u16(low_hi), low_past_v);
            uint16x4_t high_error2_v = vadd_u16(vld1_u16(high_hi), high_past_v);
            uint16x4_t le_mask2 = vcle_u16(low_error2_v, high_error2_v);
            uint16x4_t error2_v = vbsl_u16(le_mask2, low_error2_v, high_error2_v);
            uint16x4_t hist2_v = vbsl_u16(le_mask2, vdup_n_u16(0), vdup_n_u16(1));
            uint16_t error2_arr[4], hist2_arr[4];
            vst1_u16(error2_arr, error2_v);
            vst1_u16(hist2_arr, hist2_v);
            for (unsigned int bo = 0; bo < 4; bo++) {
                shift_register_t successor = low + 2 * bo + 1;
                write_errors[successor] = error2_arr[bo];
                history[successor] = (uint8_t)hist2_arr[bo];
            }
        }

        history_buffer_process(conv->history_buffer, write_errors, conv->bit_writer);
        error_buffer_swap(conv->errors);
    }
}

static ssize_t _convolutional_neon_decode(correct_convolutional_neon *neon_conv,
                                          size_t num_encoded_bits, size_t num_encoded_bytes,
                                          uint8_t *msg, const soft_t *soft_encoded) {
    correct_convolutional *conv = &neon_conv->base_conv;
    if (!conv->has_init_decode) {
        // Unlike sse/decode.c's own init (which halves this because its
        // ACS math is signed), this stays IDENTICAL to the portable
        // path's own renormalize_interval formula -- this file's ACS
        // math is plain unsigned uint16x4_t arithmetic throughout, the
        // same as portable, so the full distance_max range is safe.
        uint64_t max_error_per_input = conv->rate * soft_max;
        unsigned int renormalize_interval = distance_max / max_error_per_input;
        _convolutional_decode_init(conv, 5 * conv->order, 15 * conv->order, renormalize_interval);
    }

    size_t sets = num_encoded_bits / conv->rate;
    size_t decoded_len_bytes = num_encoded_bytes;
    bit_writer_reconfigure(conv->bit_writer, msg, decoded_len_bytes);

    error_buffer_reset(conv->errors);
    history_buffer_reset(conv->history_buffer);

    // Warmup and tail are UNCHANGED portable functions -- only the bulk
    // "inner" phase (the actual ACS hot loop) is NEON-accelerated here,
    // exactly mirroring how correct_convolutional_sse_decode() reuses
    // these same two portable functions unchanged (see sse/decode.c).
    convolutional_decode_warmup(conv, sets, soft_encoded);
    convolutional_neon_decode_inner(conv, sets, soft_encoded);
    convolutional_decode_tail(conv, sets, soft_encoded);

    history_buffer_flush(conv->history_buffer, conv->bit_writer);

    return bit_writer_length(conv->bit_writer);
}

ssize_t correct_convolutional_neon_decode(correct_convolutional_neon *conv, const uint8_t *encoded,
                                          size_t num_encoded_bits, uint8_t *msg) {
    if (num_encoded_bits % conv->base_conv.rate) {
        return -1;
    }
    size_t num_encoded_bytes =
        (num_encoded_bits % 8) ? (num_encoded_bits / 8 + 1) : (num_encoded_bits / 8);
    bit_reader_reconfigure(conv->base_conv.bit_reader, encoded, num_encoded_bytes);

    return _convolutional_neon_decode(conv, num_encoded_bits, num_encoded_bytes, msg, NULL);
}
