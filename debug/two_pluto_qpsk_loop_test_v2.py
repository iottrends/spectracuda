"""v2 of two_pluto_qpsk_loop_test.py -- fixes a real design flaw in v1,
not just a bug: v1 used a CYCLIC tx buffer (same packet repeated
non-stop on-air, ~100-200+ times per trial with the settings actually
used) purely as a crutch to sidestep not knowing exactly when RX's
snapshot-after-a-sleep would land relative to when TX started. That's
fine for a five-minute bring-up script proving the DSP chain works at
all (see two_pluto_qpsk_bind_test.py), but it's wrong for measuring real
single-shot delivery reliability: it hogs the channel and doesn't
resemble how a real link would ever transmit.

v2: transmit each packet EXACTLY ONCE (tx_cyclic_buffer=False), and make
that safe by having RX already continuously running -- an independent
background thread does nothing but call rx.rx() in a tight loop from
before the first TX to after the last trial, appending every buffer it
gets into a bounded rolling history (a deque of raw chunks, maxlen set
by --history-ms). This thread never stops, never sleeps, and never
waits on decode -- it is the one thing in this script solely
responsible for making sure no IQ samples are ever missed ("mopping").

Decode (mac.receive_iq()) deliberately stays on the MAIN thread, not a
third worker thread -- see this file's own README/session notes for the
reasoning (short version: the reader thread already guarantees no
samples are missed; a decode thread would add real synchronization
complexity, including a genuine hazard already documented elsewhere in
this codebase -- spectracuda/mac/mac.py's Mac.receive_iq_batch docstring:
the native Viterbi/RS decoder holds ONE persistent C struct per Ofdm
instance, reset not recreated per call, so two threads decoding through
the SAME instance concurrently would race and silently corrupt both
results -- for zero benefit at this trial rate, where decode (~300us
measured) is a rounding error against --period).

Per trial: tx.tx(burst) once -> wait a short, fixed margin for the burst
to actually appear in the reader thread's rolling history -> take a
snapshot of that whole rolling window (a plain list-copy under a lock,
concatenated outside the lock) -> mac.receive_iq() on the snapshot ->
compare against the known SDU.

Usage:
    python3 debug/two_pluto_qpsk_loop_test_v2.py \\
        --uri-tx ip:192.168.2.1 --uri-rx ip:192.168.3.1 --n 30 --period 0.1 --fft 256
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from spectracuda.mac import Mac  # noqa: E402


def _import_adi():
    try:
        return __import__("adi")
    except ImportError:
        sys.exit("error: pyadi-iio not installed. Run 'pip install pyadi-iio'.")


# Same two known-good PHY presets as v1's own --fft switch (see that
# file for the fuller writeup of why fft=64 needs n_training_symbols=2).
PHY_KWARGS_BY_FFT = {
    256: dict(
        fft_size=256, n_pilot=8, n_data=216, cp_len=32, modem="qpsk",
        fec="rs_m8", fec1="conv_v27", crc="crc16",
        sync="schmidl_cox", cfo="schmidl_cox",
        channel_estimator="ls", equalizer="mmse",
        backend="numpy",
    ),
    64: dict(
        fft_size=64, n_pilot=6, n_data=44, cp_len=16, modem="qpsk",
        fec="rs_m8", fec1="conv_v27",
        interleaver="block", interleaver_kwargs={"unit_bits": 8},
        crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
        n_training_symbols=2,
        channel_estimator="ls", equalizer="mmse",
        backend="numpy",
    ),
}
SDU_BITS = 2000


def _normalize_from_adc(samples: np.ndarray) -> np.ndarray:
    """Verbatim copy of ~/work/ofdm-hls/sim/pluto_loopback.py's own
    normalize_from_adc() -- see v1's identical copy for why duplicated,
    not imported. Only safe to call on a window where the burst is a
    REASONABLE fraction of the samples (see _crop_around_burst() --
    calling this directly on the mopper's whole multi-hundred-ms rolling
    history is exactly the "sparse burst inside a much larger buffer"
    failure two_pluto_qpsk_bind_test.py's own docstring already warns
    about: the 95th-percentile reference ends up measuring the noise
    floor, not the signal, and wildly over-scales the actual burst."""
    a = np.abs(samples)
    if a.size == 0:
        return samples
    ref = np.percentile(a, 95)
    if ref < 1e-9:
        return samples
    return samples * (0.5 / ref)


def _crop_around_burst(snapshot: np.ndarray, crop_samples: int, block: int = 512) -> np.ndarray:
    """Locate the burst inside a large, mostly-silent rolling-history
    snapshot via a cheap coarse energy scan (block-averaged power), then
    return a crop_samples-wide slice centered on the peak -- restores
    the same burst-to-window dilution ratio the already-proven scripts
    use (two_pluto_qpsk_bind_test.py: ~200000 samples for a ~4288-5440
    sample frame, ratio ~40-47x) instead of handing normalize_from_adc()
    the WHOLE multi-hundred-ms history (ratio ~180x+ for a 200ms/1M
    sample window, empirically confirmed this session to corrupt the
    header via massive over-scaling). Found this session via a direct
    diagnostic: max(|x|) was ~125x the 95th-percentile reference over
    the full history, scaling the real burst to ~60x its intended
    magnitude."""
    if snapshot.size <= crop_samples:
        return snapshot
    n_blocks = len(snapshot) // block
    if n_blocks == 0:
        return snapshot
    powers = np.array([
        np.mean(np.abs(snapshot[i * block:(i + 1) * block]) ** 2)
        for i in range(n_blocks)
    ])
    peak_center = int(np.argmax(powers)) * block + block // 2
    half = crop_samples // 2
    start = max(0, peak_center - half)
    end = min(len(snapshot), start + crop_samples)
    start = max(0, end - crop_samples)  # re-clamp if end hit the array edge
    return snapshot[start:end]


class RxMopper:
    """The one thread in this script responsible for never missing an
    IQ sample. Does nothing but rx.rx() -> append -> repeat, forever,
    from start() until stop(). Readers (the main thread) only ever get
    a SNAPSHOT (a copy) of recent history via history() -- they never
    block this thread and this thread never waits on them."""

    def __init__(self, rx, maxlen_chunks: int) -> None:
        self._rx = rx
        self._chunks: collections.deque = collections.deque(maxlen=maxlen_chunks)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.n_reads = 0  # diagnostic only, read from the main thread after stop()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            buf = np.asarray(self._rx.rx(), dtype="complex64")
            with self._lock:
                self._chunks.append(buf)
            self.n_reads += 1

    def history(self) -> np.ndarray:
        """A snapshot of everything currently in the rolling window,
        oldest-first, concatenated into one array. Cheap lock hold (just
        a list copy); the actual concatenate happens outside the lock so
        the mopper is never blocked waiting on numpy work."""
        with self._lock:
            chunks = list(self._chunks)
        if not chunks:
            return np.zeros(0, dtype="complex64")
        return np.concatenate(chunks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri-tx", required=True)
    ap.add_argument("--uri-rx", required=True)
    ap.add_argument("--freq", type=float, default=2.425e9)
    ap.add_argument("--rate", type=float, default=5e6)
    ap.add_argument("--tx-gain", type=float, default=0.0)
    ap.add_argument("--rx-gain", type=float, default=60.0)
    ap.add_argument("--agc", default="manual")
    ap.add_argument("--rx-samples", type=int, default=4096, help="samples per rx.rx() call the mopper thread makes -- smaller than v1's 40000 for finer-grained rolling history (~0.82ms/chunk @ 5Msps) without adding meaningful per-call overhead")
    ap.add_argument("--history-ms", type=float, default=200.0, help="rolling history depth the mopper thread keeps. Measured directly this session (oneshot_tx_timing_diag.py): a one-shot (non-cyclic) tx.tx() call has ~55-70ms of real DMA/USB dispatch latency before the burst actually appears on-air (MUCH slower than a cyclic buffer, which starts looping immediately once armed) -- default bumped from an initial 50ms, which was silently snapshotting BEFORE the burst had even been transmitted, every trial")
    ap.add_argument("--post-tx-wait-ms", type=float, default=150.0, help="fixed wait after tx.tx() before snapshotting history -- must cover the one-shot TX dispatch latency above (measured ~55-70ms one one run; this default gives real margin, not just a couple of chunk periods) -- NOT a settle/AGC hack (see module docstring)")
    ap.add_argument("--crop-samples", type=int, default=60_000, help="width of the window cropped around the located burst before normalize+decode -- keeps the burst-to-window dilution ratio close to two_pluto_qpsk_bind_test.py's own proven ~40-47x, instead of handing normalize_from_adc() the whole --history-ms rolling snapshot (found this session to badly over-scale a burst that's <1% of the window -- see _crop_around_burst()'s docstring)")
    ap.add_argument("--n", type=int, default=30, help="number of trials")
    ap.add_argument("--period", type=float, default=0.1, help="seconds between trial starts (best-effort)")
    ap.add_argument("--fft", type=int, default=256, choices=sorted(PHY_KWARGS_BY_FFT))
    ap.add_argument("--lead-zeros", type=int, default=None, help="default: 2*(fft_size+cp_len) for the selected --fft")
    ap.add_argument("--trail-zeros", type=int, default=None, help="default: 2*(fft_size+cp_len) for the selected --fft")
    args = ap.parse_args()

    PHY_KWARGS = PHY_KWARGS_BY_FFT[args.fft]
    if args.lead_zeros is None:
        args.lead_zeros = 2 * (PHY_KWARGS["fft_size"] + PHY_KWARGS["cp_len"])
    if args.trail_zeros is None:
        args.trail_zeros = 2 * (PHY_KWARGS["fft_size"] + PHY_KWARGS["cp_len"])

    adi = _import_adi()
    rf_bw = int(max(args.rate * 1.25, 5e6))

    tx = adi.Pluto(uri=args.uri_tx)
    tx.sample_rate = int(args.rate)
    tx.tx_lo = int(args.freq)
    tx.tx_rf_bandwidth = rf_bw
    tx.tx_hardwaregain_chan0 = float(args.tx_gain)
    tx.tx_cyclic_buffer = False  # the actual v1->v2 fix: one-shot playback, not a
    # repeating loop -- see module docstring for why v1's cyclic approach was
    # wrong (hogs the channel, doesn't model real single-shot delivery).

    rx = adi.Pluto(uri=args.uri_rx)
    rx.sample_rate = int(args.rate)
    rx.rx_lo = int(args.freq)
    rx.rx_rf_bandwidth = rf_bw
    rx.gain_control_mode_chan0 = args.agc
    if args.agc == "manual":
        rx.rx_hardwaregain_chan0 = float(args.rx_gain)
    rx.rx_buffer_size = int(args.rx_samples)

    chunk_duration_s = args.rx_samples / args.rate
    maxlen_chunks = max(1, int(args.history_ms / 1000.0 / chunk_duration_s) + 1)
    mopper = RxMopper(rx, maxlen_chunks=maxlen_chunks)
    mopper.start()
    # Let the mopper get a few chunks of real history before trial 0 --
    # otherwise the very first trial's post-tx-wait might race the
    # thread's own startup.
    time.sleep(3 * chunk_duration_s)

    print(f"[loop_v2] uri_tx={args.uri_tx} uri_rx={args.uri_rx} freq={args.freq/1e9:.3f}GHz "
          f"rate={args.rate/1e6:.1f}MSPS tx_gain={args.tx_gain:+.1f}dB rx_gain={args.rx_gain:+.1f}dB "
          f"({args.agc}) fft={args.fft} n={args.n} period={args.period}s "
          f"chunk={args.rx_samples}samples({chunk_duration_s*1e3:.2f}ms) "
          f"history={maxlen_chunks}chunks({maxlen_chunks*chunk_duration_s*1e3:.1f}ms) "
          f"post_tx_wait={args.post_tx_wait_ms}ms")

    n_ok = 0
    n_fail = 0
    rng = np.random.default_rng()
    try:
        for i in range(args.n):
            t_start = time.perf_counter()

            # Fresh Mac per trial -- same reasoning as v1 (a shared
            # ReassemblyBuffer permanently desyncs after one missed
            # decode in a run shorter than window_size, see v1's own
            # comment for the full expected_sn trace that found this).
            mac = Mac(mode="um", ofdm_kwargs=PHY_KWARGS)
            mac.bound = True

            sdu = rng.integers(0, 2, size=SDU_BITS).astype("uint8")
            frames = mac.send_iq(sdu)
            if len(frames) != 1:
                print(f"  [{i:3d}] SKIP -- SDU_BITS produced {len(frames)} frames, expected 1")
                continue
            iq_frame = np.asarray(frames[0])[0].astype(np.complex64)
            burst = np.concatenate([
                np.zeros(args.lead_zeros, dtype=np.complex64),
                iq_frame,
                np.zeros(args.trail_zeros, dtype=np.complex64),
            ])
            peak = np.max(np.abs(burst))
            scaled = burst * (2 ** 13 / peak) if peak > 1e-12 else burst * 0

            tx.tx(scaled.astype(np.complex64))  # ONE-SHOT -- plays once, no
            # tx_destroy_buffer() needed (that's only for stopping a cyclic
            # loop, which this isn't anymore).
            time.sleep(args.post_tx_wait_ms / 1000.0)

            rx_history = mopper.history()
            rx_snapshot = _crop_around_burst(rx_history, args.crop_samples)
            rx_scaled = _normalize_from_adc(rx_snapshot)

            try:
                delivered = mac.receive_iq(rx_scaled[None, :].astype(np.complex64))
            except Exception as e:
                print(f"  [{i:3d}] EXC {type(e).__name__}: {e}")
                n_fail += 1
                delivered = []

            if delivered and np.array_equal(np.asarray(delivered[0]), sdu):
                n_ok += 1
                status = "OK"
            else:
                n_fail += 1
                status = "FAIL"
            p_lin = float(np.mean(np.abs(rx_snapshot) ** 2)) if rx_snapshot.size else 0.0
            rssi_dbfs = 10 * np.log10(p_lin / (2**15 * 2**15) + 1e-300)
            print(f"  [{i:3d}] {status}  rx_dbfs={rssi_dbfs:6.1f}  snapshot={len(rx_snapshot)}samp  "
                  f"running: {n_ok}/{i+1} ok ({100*n_ok/(i+1):.0f}%)")

            elapsed = time.perf_counter() - t_start
            remaining = args.period - elapsed
            if remaining > 0:
                time.sleep(remaining)
    finally:
        mopper.stop()
        print(f"[loop_v2] mopper thread made {mopper.n_reads} rx.rx() calls total "
              f"({mopper.n_reads * args.rx_samples / args.rate:.2f}s of real IQ consumed)")

    print(f"\n=== FINAL: {n_ok}/{args.n} succeeded ({100*n_ok/args.n:.1f}%), {n_fail} failed ===")


if __name__ == "__main__":
    main()
