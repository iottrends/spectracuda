# Notes on NVIDIA Aerial / cuPHY's LDPC kernels

Source: `reference/aerial-cuda-accelerated-ran/` (git-ignored local clone,
same "study, don't ship" status as `reference/libcorrect/`/
`reference/liquid-dsp/` -- see `.gitignore`). Cloned from
https://github.com/NVIDIA/aerial-cuda-accelerated-ran, Apache 2.0
licensed (confirmed: `LICENSE` file, `SPDX-License-Identifier: Apache-2.0`
header on every source file checked) -- the repo's own source is freely
usable/studyable; only pulling the prebuilt Aerial *container* from NGC
carries separate license terms (irrelevant here, we're not doing that).

## Why we're looking at this

`fec/ldpc.py`'s decode() -- and this project's own
`examples/benchmark_ldpc_cuda.py` -- showed LDPC on cupy getting only a
~10x speedup over numpy, and even at that, absolute throughput
(~0.35 Mbps at batch=512) was ~25-30x SLOWER than this project's
already-optimized CPU rs_m8+conv_v27 (~8-11 Mbps). cuPHY is NVIDIA's own
production LDPC implementation (part of the Aerial 5G/6G RAN SDK) --
proof that real GPU LDPC throughput is achievable, IF built differently
than "chain generic cupy array ops in a Python loop."

**Important caveat, not to lose sight of**: Aerial itself targets
carrier-grade, multi-cell, datacenter hardware (its own docs list Grace
Hopper MGX / DGX Spark / Dell R750 as supported systems; no mention of
Jetson anywhere across the docs checked). It's not something to adopt
wholesale for a Jetson-based point-to-point drone link -- it's a
reference for HOW real fused LDPC kernels are built, not a drop-in
dependency for this project.

## The two real reasons cuPHY's LDPC decode is fast

Found by reading `cuPHY/src/cuphy/error_correction/ldpc2_kernel.cuh`
(the abstract kernel template -- comments intact even though the body's
commented out, superseded by the concrete instantiations) and
`ldpc2_global.cu` (the real launch/dispatch code):

1. **One kernel launch decodes the ENTIRE batch, all iterations,
   start to finish.** `ldpc2_kernel<...>` loads LLRs once, runs
   `do_first_iteration()` then a plain C++ `for` loop (not a Python
   loop re-entering cupy each time) for the remaining iterations, then
   writes output -- ALL inside one `__global__` function, one launch.
   Our fec/ldpc.py does 50 Python-loop iterations x ~8 cupy calls each
   = ~400 separate kernel launches per decode() call. Each launch pays
   real dispatch overhead; cuPHY pays it once (per batch).

2. **Each codeword gets its own dedicated thread block, with all its
   working data resident in on-chip SHARED memory for the whole
   decode**, not round-tripping through slower GPU global memory every
   iteration:
   ```cpp
   dim3 grdDim(config.num_codewords);  // one CUDA block PER codeword
   dim3 blkDim(config.Z);              // Z threads cooperate on that ONE codeword
   ```
   (`ldpc2_global.cu`, `decode_ldpc2_global_address()`). Every codeword
   in the batch decodes concurrently, each in its own block, each
   entirely in shared memory. Our cupy version instead treats "batch"
   as just another array axis -- every `xp.where`/`xp.argsort`/etc.
   call reads the WHOLE batch from global memory and writes the whole
   batch back to global memory, every single iteration.

Both of these compound: ~400 launches x paying full global-memory
round-trip cost each time, vs. 1 launch x data staying on-chip the
whole time.

## The other big thing cuPHY does that we do NOT need to copy

`ldpc2_global.cu` is ~1000+ lines, almost entirely one giant `switch`
statement picking between HUNDREDS of separately pre-compiled kernel
*variants* -- one per (lifting size Z, base graph, number of parity
rows mb, LLR data type) combination, each a distinct C++ template
instantiation (`launch_glob_all_shared<float, 1, 22, 384, 8, ...>`,
`launch_glob_all_shared<float, 1, 22, 384, 9, ...>`, etc., repeated for
every Z in {64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384} and
every mb from 4 to 46). That's because Aerial needs to serve every
possible 5G NR LDPC configuration a live cell might negotiate with any
UE, at any moment.

This project doesn't need that generality: ONE fixed LDPC variant
(whichever `ldpc_*_r*` is actually configured), ONE realistic batch
size range (~8-30 codewords/frame, from `benchmark_x86_stages_ldpc.py`'s
own measurement). A purpose-built kernel here only needs to be
specialized for that one, fixed (Z, k_bits, n_bits) shape -- not
hundreds of variants.

## What a minimal version of this would need (not yet built)

1. A `__global__` kernel doing steps 1-2 above for exactly ONE
   (Z, k_bits, n_bits) configuration: one thread block per codeword in
   the batch, that block's Z threads holding channel LLRs + messages in
   shared memory, a plain `for` loop over the (max_iterations=50)
   belief-propagation iterations entirely inside the kernel, writing
   decoded bits out at the end.
2. The actual min-sum check-node/variable-node update math already
   exists, correctly, in `fec/ldpc.py`'s `decode()` (the array-op
   version) -- that's the reference to translate into per-thread CUDA
   logic, not something to re-derive from scratch.
3. Compiled via CuPy's `RawKernel` (or `cupy.RawModule` for multiple
   kernels sharing device code), not chained `self.xp` calls -- this is
   the actual "write real CUDA C, not Python" step referenced in chat.
4. Would plug into `LDPCCode` the same way `fec/_native.py`'s
   `NativeConvolutionalSSE`/`NativeConvolutional` plug into
   `ConvolutionalCode` -- an optional, transparently-preferred backend,
   with the existing pure-array-op path kept as the correctness
   reference / fallback for platforms without it.

Not attempted yet -- this is a scoped next step, not started work.
