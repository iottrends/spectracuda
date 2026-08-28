# 2026-08-27 session: NEON Viterbi kernel + RX throughput investigation

**Goal driving this whole session:** the user wants a 10 Mbps TX/RX link.
TX already clears that comfortably (~13-22 Mbps measured). RX does not
(~2.5-3.7 Mbps depending on config) -- this doc is the record of what was
tried, what actually won, what's still open, and exact numbers so the
next session doesn't have to re-derive any of this from scratch.

Machine this was all measured on: a real Raspberry Pi 5 (aarch64,
`Linux ... +rpt-rpi-2712`, 4 cores) -- not WSL2, not emulated. Every
number below is real, not estimated, unless explicitly marked as a
prediction/extrapolation.

Branch: `perf/native-fec-and-ofdm-batching` (this is where all the diffs
below already landed, uncommitted at the time of writing -- see `git
status`/`git diff` before doing anything else, don't assume this doc's
line numbers are still exact).

---

## 1. DONE: 8-lane NEON Viterbi decode kernel (implemented + wired in)

### Backstory (context for why this existed before this session)
- `fec/_native_src/libcorrect/src/convolutional/neon/decode.c` had a
  first-attempt NEON kernel (commit `aa92a4e`) using 4-lane `uint16x4_t`
  vectors, with heavy stack-array round-tripping (gather into scratch
  arrays -> `vld1` -> vector math -> `vst1` -> scalar copy-out).
  Measured **~2.1x SLOWER** than the portable C build (7.92ms vs 3.72ms
  for a ~24000-bit PDU) -- correct (bit-exact) but a real regression, so
  commit `4b8a122` explicitly reverted `viterbi.py`'s dispatch to skip
  NEON (`SSE -> portable`, no NEON), leaving the kernel in place but
  unused, for "a future, tighter kernel."

### What this session built
Rewrote `convolutional_neon_decode_inner()` in that same file: widened
to **8-lane `uint16x8_t`** (covers 2 of the portable loop's outer-loop
steps per NEON sequence) AND eliminated almost all the memory
round-tripping:
- `read_errors[base..base+7]` / `read_errors[highbase+base..+7]` are
  already contiguous -> loaded directly via `vld1q_u16`, no staging
  array at all.
- The distance-pair gather (`pair_lookup.keys[...] ->
  pair_lookup.distances[key]`) is genuinely data-dependent (no ARMv8
  ASIMD gather instruction exists), so it stays scalar -- but instead of
  manually splitting each 32-bit concat value into lo/hi via scalar
  shifts into 4 separate arrays, the raw `uint32_t` values are gathered
  into ONE scratch array per side and de-interleaved into two
  `uint16x8_t` vectors with a single `vld2q_u16`.
- Output: the "successor" (even) and "plus-one" (odd) result streams are
  exactly the two interleaved halves of `write_errors`/`history` over 16
  states at once -> written back with `vst2q_u16` / `vst2_u8` (after
  `vmovn_u16` narrowing history's 0/1 mask), no scalar copy-out loop.
- Kept a scalar/4-lane tail loop for any non-multiple-of-8 remainder
  (never actually hit for this project's fixed K=7/order=7 code, where
  `highbase=32` always divides evenly by 8 -- kept for generality only).
- `pair_lookup_t`/`history_buffer` machinery: untouched, exactly as the
  first attempt already did.

Also wired it into `fec/viterbi.py`'s `ConvolutionalCode.__init__`
dispatch: `SSE -> NEON -> portable -> pure-Python` (previously `SSE ->
portable`, NEON skipped). Updated that method's own explanatory comment
block with the new measured numbers (search for "NEON, second attempt,
DOES" in that file).

### Verification (same two-gate rigor as the original NEON promotion)
1. **Correctness first, unconditionally**: `pytest
   tests/test_fec_native_acceleration.py -k neon -v` -- all 30/30 pass
   bit-exact, INCLUDING `test_convolutional_code_neon_backend_is_active`
   (which failed before this session's dispatch change, by design --
   it asserts NEON is the actual default pick). Full suite
   (`pytest tests/ -q`) -- 901 passed, 1 skipped, no regressions.
2. **Only then, benchmark**: `python examples/benchmark_x86_stages_v3.py
   32000` (standard QPSK/24040-bit-PDU config), "FEC decode -- Viterbi
   (fec1, outer)" line, 3 repeated runs each:

   | Build | Viterbi decode (24040-bit PDU) |
   |---|---|
   | Portable C (baseline) | 3.79 ms (3.7876 / 3.7969 / 3.8001) |
   | First NEON attempt (4-lane, stack round-trips) | 7.92 ms (known, from commit `4b8a122`) |
   | **This session's 8-lane kernel** | **2.38 ms (2.3782 / 2.3824 / 2.3773)** |

   ~1.6x faster than portable, ~3.3x faster than the first attempt.
   Genuinely wins -> wired into default dispatch (unlike the first
   attempt, which correctly was NOT wired in when it lost).

### Files touched
- `spectracuda/fec/_native_src/libcorrect/src/convolutional/neon/decode.c`
  (the kernel rewrite)
- `spectracuda/fec/viterbi.py` (dispatch + comment update)

### If resuming work on this kernel
It's not tapped out -- the tail loop, the warmup/tail phases (still
portable, unchanged), and the traceback/history-buffer overhead are all
unexamined for further NEON gains. Any change here: re-verify bit-exact
via `-k neon` FIRST, only then re-benchmark, same order as above, don't
skip step 1.

---

## 2. DONE: root-caused why RX throughput looked like it "fluctuated" by modem

Not a code change -- a diagnosis, requested by the user after noticing
Viterbi/RS decode times swung a lot across modem choices at the same SDU
bit count, while sync/CFO/OFDM-decode barely moved.

**Root cause**: `Mac.max_segment_bits` (`mac/capacity.py`,
`compute_max_segment_bits()`) is derived from the PHY's actual per-frame
bit capacity (`ofdm.bits_per_ofdm_symbol`), which scales with the
modem's bits/symbol. For this project's standard PHY config
(`fft_size=256, n_pilot=8, n_data=216, cp_len=32, fec=rs_m8,
fec1=conv_v27, crc=crc16`):

| Modem | `max_segment_bits` |
|---|---|
| QPSK | 24008 |
| QAM16 | 48120 |
| QAM64 | 72392 |

So a 30000/32000/36000-bit SDU under **QPSK** splits into **two very
UNEVEN PDUs** (e.g. 32000 bits -> one 24040-bit PDU + one 8024-bit
remainder), while the same SDU under QAM16/QAM64 fits in **one uniform
PDU**. `benchmark_x86_stages_v3.py`'s RX stage breakdown averages
per-frame cost with equal weight across however many frames there are
-- so for QPSK's uneven 2-PDU case, one fast small-PDU decode drags the
reported "per-frame average" down, making it look like decode got
faster, when actually decode cost is a clean, near-linear function of
bits-per-PDU, independent of modem (verified: QAM16 vs QAM64 at
identical PDU sizes agree within ~0.3%; QPSK's one unsplit case, 24000
bits, matches QAM16/64's 24032-bit rows almost exactly, ~3.46-3.49ms).

Full 4-size x 3-modem table with real numbers is in the conversation
transcript, not reproduced here -- rerun `examples/benchmark_x86_stages_v3.py
<bits> <modem>` for any of {24000,30000,32000,36000} x
{qpsk,qam16,qam64} to regenerate it if needed (each run ~1-2s).

**Practical implication**: compare decode throughput across modems (or
across the NEON-kernel win) at matching PDU sizes, not raw SDU-size
rows, or QPSK's uneven-split rows will look artificially fast/slow.

---

## 3. NOT DONE: sync/CFO acceleration (diagnosed only, no code written)

Read `spectracuda/sync/schmidl_cox.py` and `spectracuda/cfo/schmidl_cox.py`
in full. Conclusion: this is NOT the "naive Python loop" situation FEC
was in before its native path existed -- `SchmidlCoxSync.process()` is
already a fully vectorized batch correlation (cumsum-based prefix sums,
no per-offset Python loop), and has already been through prior
optimization passes (the file's own comments document a fixed 5.7x
fancy-indexing regression and a 3x cos/sin-vs-`exp()` fix).

**What's actually costing ~1.1-1.7ms/frame**: the algorithm still does
**~10 separate full-frame numpy passes** per call (`conj`, multiply,
`abs`, two `cumsum`s, several slice-subtracts, divide, `argmax`) over
the whole ~37K-sample frame -- each one a full memory round-trip. Same
*shape* of problem the NEON kernel just fixed (not bad math, just more
memory traffic than the arithmetic needs), but the fix here would look
different: a **Numba-JIT'd single-pass fused kernel** (mirroring the
existing transparent-acceleration pattern in `fec/_numba_crc.py`) doing
`a`, `e`, the running window sums, and the metric in one sliding-window
loop instead of ~10 separate array-wide numpy ops.

**Not built, not measured.** If picking this up:
1. Read `sync/schmidl_cox.py` and `cfo/schmidl_cox.py` again (they're
   short, ~160 and ~95 lines) before writing anything -- the symmetric
   `R(d)` formula deviation from the textbook Schmidl-Cox paper (see
   that file's own module docstring) needs to be preserved exactly.
2. Prototype the fused Numba kernel, verify bit-exact/numerically
   equivalent against the current numpy path first (there should be an
   existing sync test suite -- find it, likely `tests/test_sync_*.py`
   or similar, check before assuming).
3. Only then benchmark via `benchmark_x86_stages_v3.py`'s "sync detect +
   CFO" line, same before/after discipline as the NEON kernel above.
4. See section 4 below (Amdahl's law finding) for why this stage is
   actually HIGHER leverage than it first looks, if going for full RX
   throughput rather than just this stage's own number.

---

## 4. DONE: multi-core FEC decode via threads (`Mac.receive_iq_batch`)

### Design
Added to `spectracuda/mac/mac.py`:
- `Mac._rx_process_only(ofdm, iq_array)` -- pure decode step (just
  `ofdm.rx_process(iq_array)`, or `None` on the same
  ValueError/NotImplementedError `_rx_one_frame` always treated as "no
  frame"). Touches only the given `ofdm` instance, nothing on `self`.
- `Mac._apply_rx_result(result)` -- the stateful half (quality
  bookkeeping via `self.quality.observe()`, delivered/CRC check).
  Deliberately single-threaded-only.
- `Mac._deliver(delivered_bits)` -- shared mode-dependent hand-off to
  `self._impl.receive()`/`receive_data()`, factored out of `receive_iq()`
  so `receive_iq_batch()` uses the identical logic, not a second copy.
- `Mac._rx_one_frame()` refactored to just
  `_apply_rx_result(_rx_process_only(self.ofdm, iq_array))` --
  behavior-preserving, existing tests confirm no regression.
- `Mac._ofdm_replica_pool(n)` -- lazily builds/caches N independent
  `Ofdm` instances from the exact same `ofdm_kwargs` (now stored as
  `self._ofdm_kwargs` in `__init__`, wasn't stored before this session).
- **`Mac.receive_iq_batch(iq_arrays, n_workers=2)`** -- the new public
  method. Round-robins frames across a `ThreadPoolExecutor` of
  `n_workers`, each worker bound to exactly ONE `Ofdm` replica for its
  whole life in that call (so no two concurrently-running tasks ever
  share a replica). After the parallel decode phase, runs
  `_apply_rx_result`/`_deliver` sequentially, in original frame order.

### WHY a pool of replicas, not just `receive_iq()` from N threads
The native Viterbi/RS decoders (`fec/_native.py`) each own ONE
persistent C struct, RESET (not recreated) at the start of every decode
call. Correct for sequential reuse of one `Ofdm`/`ConvolutionalCode`
instance, but two threads calling `decode()` through the SAME instance
concurrently would race on that struct's internal
`history_buffer`/`error_buffer` and silently corrupt both results. This
was proven as an isolated experiment first (giving each thread its own
`ConvolutionalCode()` instance) before writing any of the `Mac`-level
code -- see the conversation transcript for that standalone script if
useful, not preserved as a repo file.

### Verification
- New test file `tests/test_mac_receive_iq_batch.py` (9 tests): output
  of `receive_iq_batch()` is bit-exact against sequential `receive_iq()`
  calls, swept across `n_workers` in {1,2,3,4} and SDU sizes producing
  1/2/3 PDUs (including more workers than frames, and the empty-input
  case). Plus a direct test that `_ofdm_replica_pool` returns genuinely
  distinct objects, never `self.ofdm` itself, and caches/grows correctly.
- Full suite: `pytest tests/ -q` -> 901 passed, 1 skipped. No regressions
  from the `_rx_one_frame` refactor.

### Measured real-world result -- and WHY it's smaller than hoped
| Test | Sequential | 2 workers | Speedup |
|---|---|---|---|
| Isolated Viterbi-only micro-benchmark (separate `ConvolutionalCode` instances, no Mac/Ofdm involved) | 23.7ms/8 decodes | 14.5ms | **1.65x** |
| Full RX pipeline, real SDU (48000 bits QPSK, balanced ~24040/24024-bit 2-PDU split) | 18.5 ms/SDU | 14.8 ms/SDU | **1.25x** |
| Full RX pipeline, 32000 bits QPSK (UNEVEN 24040/8024-bit split) | 13.1 ms/SDU | 11.5 ms/SDU | 1.13x |

The full-pipeline number (1.25x, even with a balanced PDU split) is well
short of the isolated FEC-only test's 1.65x. **Root cause, confirmed
quantitatively, not just asserted**: only the native C-backed stages
(Viterbi + RS decode, via `ctypes`, which releases the GIL for the
duration of the call) actually parallelize across threads. Everything
else in `ofdm.rx_process()` -- sync/CFO, OFDM demod, channel estimation,
equalization, header/resource-grid handling -- is pure Python/numpy and
holds the GIL, so it stays serialized regardless of worker count.

Check: from the earlier stage breakdown, Viterbi+RS is **~44%** of total
RX time. Amdahl's law for a 44%-parallelizable workload on 2 workers
predicts `1/(0.56 + 0.44/2) = 1.28x` -- matches the measured 1.25-1.29x
almost exactly. This is strong evidence the GIL explanation is right,
not a guess.

### Status: opt-in, NOT the default
`receive_iq()` (used by `send_iq()`'s natural counterpart everywhere
else in the codebase, including `benchmark_x86_stages_v3.py`) is
UNCHANGED and still the default, single-frame-at-a-time path.
`receive_iq_batch()` is purely additive -- same discipline as the NEON
kernel: don't make something the default until it's an unconditional
win, and a 1.13-1.29x real-but-modest gain with a known, explained
ceiling isn't that yet.

### If resuming work here
Two independent directions, not mutually exclusive:
1. **Grow the parallelizable fraction**: the sync/CFO Numba-fusion work
   (section 3) would help doubly here if written with `nogil=True` --
   it speeds up that stage directly AND lets more of the pipeline
   actually run concurrently under `receive_iq_batch`, pushing the
   Amdahl ceiling up from ~44% parallelizable toward much more.
2. **Sidestep the GIL entirely**: real multiprocessing (e.g.
   `ProcessPoolExecutor`) instead of threads. Not attempted this
   session -- open questions: whether `Ofdm`/`ConvolutionalCode`
   instances need to be rebuilt per-worker-process (can't just pickle a
   loaded `ctypes.CDLL` handle across a process boundary) vs. built once
   in each persistent worker and reused; IPC/serialization cost of
   passing ~37K-sample complex64 IQ arrays (~300KB each) between
   processes, not yet measured; how this interacts with `backend="cupy"`
   (a worker process doesn't share the parent's CUDA context/device
   memory, would need its own).
3. Whichever is pursued, benchmark `receive_iq_batch` again afterward at
   BOTH the balanced (48000-bit) and unbalanced (32000-bit) SDU sizes
   above, since the QPSK PDU-imbalance issue (section 2) compounds with
   whatever parallel-speedup ceiling the GIL/multiprocessing situation
   allows.

### Files touched
- `spectracuda/mac/mac.py` (stores `_ofdm_kwargs`, adds
  `_rx_process_only`/`_apply_rx_result`/`_deliver`/`_ofdm_replica_pool`/
  `receive_iq_batch`, refactors `_rx_one_frame`/`receive_iq`)
- `tests/test_mac_receive_iq_batch.py` (new)

---

## 5. Where the 10Mbps RX goal actually stands right now

At the standard 32000-bit/QPSK config, end-to-end:
- **TX**: ~13-22 Mbps depending on config -- already well above target,
  not a concern.
- **RX before this session**: ~3.07 Mbps (portable Viterbi).
- **RX after the NEON kernel (section 1)**: ~3.71 Mbps -- a real ~21%
  gain, but the single biggest reason RX still falls short: **Viterbi
  decode alone, even at the new NEON speed, caps single-threaded RX
  around ~6.5 Mbps** (computed: two PDUs for a 32000-bit SDU cost ~4.9ms
  of Viterbi time alone -> 32000 bits / 4.9ms = 6.5 Mbps ceiling, before
  ANY other stage costs anything). So even a hypothetical zero-cost
  everything-else still wouldn't reach 10Mbps on Viterbi's own current
  speed alone, single-threaded.
- **Multi-core decode (section 4)** adds another ~1.13-1.29x on top of
  that, real but capped by the GIL as explained above.

**Net assessment**: 10Mbps RX is reachable but needs combined
improvement across multiple stages/approaches, not one silver bullet:
the NEON kernel (done), sync/CFO fusion (diagnosed, not built),
multi-core decode (built, modest as-is), and either growing the
GIL-releasing fraction of the pipeline or moving to real multiprocessing
for the rest. Recommended next move, per the Amdahl reasoning in section
4: **sync/CFO Numba fusion with `nogil=True`** is probably the
highest-leverage next step, since it both shrinks that stage's own cost
AND unlocks more of the pipeline for the multi-core decode work already
built.

---

## 6. Quick reference: commands used throughout this session

```bash
# activate venv (repo root)
source .venv/bin/activate

# NEON correctness gate (must pass bit-exact BEFORE any benchmark)
python -m pytest tests/test_fec_native_acceleration.py -k neon -v

# full suite (regression check after any change)
python -m pytest tests/ -q

# standard single-config benchmark (SDU bits, then optional modem)
python examples/benchmark_x86_stages_v3.py 32000
python examples/benchmark_x86_stages_v3.py 32000 qam16

# this Pi's core count / architecture (sanity check before any
# multi-core or NEON work)
nproc              # -> 4
uname -m           # -> aarch64
```

No scratch/throwaway scripts from this session were kept in the repo
(they lived under the session's own scratchpad dir and are gone) --
the `tests/test_mac_receive_iq_batch.py` file is the one new permanent
test artifact; everything else provable is reproducible via the
commands above plus the diffs already in the working tree (see `git
diff` on this branch before doing anything else, since none of this was
committed yet as of writing this doc).
