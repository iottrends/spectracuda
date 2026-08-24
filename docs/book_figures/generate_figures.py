"""Regenerates every plot embedded in the tutorial book (see docs/BOOK.md
pointer / the published artifact). Every figure here is produced by
actually running spectracuda's real classes (Modem, SchmidlCoxSync,
Ofdm, Channel) -- nothing is a hand-drawn illustration standing in for
real output. Where a figure needs internal values a class doesn't
normally return (e.g. Schmidl-Cox's full metric-vs-offset curve, not
just its argmax peak), the exact documented formula is mirrored here and
asserted to agree with the real class's own output on the same signal --
never presented without that check.

Usage: .venv/bin/python docs/book_figures/generate_figures.py
Requires the `docs` extra: pip install -e ".[docs]"  (matplotlib only;
never a runtime dependency of spectracuda itself.)

Output: docs/book_figures/*.png -- consumed by the publish step that
base64-embeds them into the book's HTML (images are captured-instrument-
style renders with their own fixed dark styling, independent of the
book page's light/dark theme -- see the book's own design notes).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from spectracuda.modem.mapper import Modem
from spectracuda.sync.schmidl_cox import SchmidlCoxSync
from spectracuda.pipeline import Ofdm
from spectracuda.sim import Channel

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- "instrument screen" figure styling, used for every plot ----------
# Deliberately fixed regardless of the book page's own light/dark theme --
# every figure is presented as a captured measurement, like a scope/
# spectrum-analyzer screenshot, not page chrome that needs to re-theme.
BG = "#0F1826"
GRID = "#26344A"
TEXT = "#C9D3E0"
TEAL = "#35D0C4"
AMBER = "#F2A94E"
VIOLET = "#9C8CF5"
RED = "#F26B6B"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.9,
    "text.color": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "font.family": "monospace",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "savefig.facecolor": BG,
    "savefig.edgecolor": BG,
})


def _save(fig, name, dpi=140):
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    size_kb = os.path.getsize(path) / 1024
    print(f"  wrote {name}.png  ({size_kb:.0f} KB)")


# =========================================================================
# Figure 1: Schmidl-Cox timing metric vs. sample offset
# =========================================================================
def fig_sync_metric():
    print("fig_sync_metric...")
    fft_size = 256
    sync = SchmidlCoxSync(fft_size=fft_size, backend="numpy")
    preamble = np.asarray(sync.generate_preamble())  # (fft_size,) complex64

    rng = np.random.default_rng(7)
    n_lead = 400  # noise-only samples before the real preamble arrives
    n_tail = 300
    lead = (rng.standard_normal(n_lead) + 1j * rng.standard_normal(n_lead)).astype("complex64") * 0.05
    tail = (rng.standard_normal(n_tail) + 1j * rng.standard_normal(n_tail)).astype("complex64") * 0.05
    payload_noise = (rng.standard_normal(600) + 1j * rng.standard_normal(600)).astype("complex64") * 0.15
    rx = np.concatenate([lead, preamble, payload_noise, tail]).astype("complex64")
    true_start = n_lead

    # Ground truth: what the real block actually detects.
    result = sync.process(rx[None, :])
    detected_start = int(result["start_index"][0])
    detected_peak = float(result["metric"][0])

    # Mirror SchmidlCoxSync.process()'s exact documented formula to get
    # the FULL curve (process() only returns the argmax peak) -- verified
    # against the real block's own output below, not just assumed correct.
    L = fft_size // 2
    a = np.conj(rx[:-L]) * rx[L:]
    b1 = np.abs(rx[:-L]) ** 2
    b2 = np.abs(rx[L:]) ** 2
    a_cum = np.concatenate([[0], np.cumsum(a)])
    b1_cum = np.concatenate([[0], np.cumsum(b1)])
    b2_cum = np.concatenate([[0], np.cumsum(b2)])
    n_candidates = len(rx) - 2 * L + 1
    idx = np.arange(n_candidates)
    p = a_cum[idx + L] - a_cum[idx]
    r1 = b1_cum[idx + L] - b1_cum[idx]
    r2 = b2_cum[idx + L] - b2_cum[idx]
    r = 0.5 * (r1 + r2)
    metric = np.abs(p) ** 2 / (r ** 2 + 1e-12)

    assert int(np.argmax(metric)) == detected_start
    assert abs(float(metric[np.argmax(metric)]) - detected_peak) < 1e-4
    print(f"  verified: mirrored curve's argmax ({int(np.argmax(metric))}) == "
          f"SchmidlCoxSync.process()'s own start_index ({detected_start})")

    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(idx, metric, color=TEAL, linewidth=1.3)
    ax.axvline(true_start, color=TEXT, linestyle=":", linewidth=1, alpha=0.6, label=f"true preamble start (n={true_start})")
    ax.axvline(detected_start, color=AMBER, linestyle="--", linewidth=1.4, label=f"detected start (n={detected_start})")
    ax.set_xlabel("candidate offset d (samples)")
    ax.set_ylabel("timing metric M(d)")
    ax.set_title(f"Schmidl-Cox timing metric — fft_size={fft_size}, peak M={detected_peak:.3f}")
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, loc="upper right", fontsize=9)
    ax.set_xlim(0, len(rx) - 2 * L)
    _save(fig, "sync_metric")


# =========================================================================
# Figure 2: CFO — the rotation, illustrated on real QPSK symbols
# =========================================================================
def fig_cfo_rotation():
    print("fig_cfo_rotation...")
    modem = Modem("qpsk", backend="numpy")
    rng = np.random.default_rng(3)
    n_symbols = 400
    bits = rng.integers(0, 2, size=(1, n_symbols * modem.bits_per_symbol)).astype("uint8")
    symbols = np.asarray(modem.modulate(bits))[0]  # (n_symbols,) complex64, unit-power QPSK

    # Exactly Channel's own CFO formula (sim/channel.py): a per-sample
    # phase ramp exp(j*2*pi*cfo*n/fft_size) -- applied here directly to
    # symbol-rate data purely to make the geometric effect legible
    # (one point per symbol instead of one per IQ sample); Channel itself
    # applies the identical formula sample-by-sample to the real IQ
    # waveform inside the CFO-correction chapter's actual Ofdm run.
    fft_size = 256
    # 3.5 subcarriers, deliberately large so the spiral is visually obvious
    # here -- this panel applies/removes a KNOWN offset directly (no
    # estimator involved), which works at any magnitude. The real
    # SchmidlCoxCFO estimator has a narrower reliable range (empirically,
    # this codebase's implementation decodes correctly up to ~1.0
    # subcarrier of offset at 25 dB SNR and fails past ~1.5 -- see
    # fig_cfo_recovered() below, which uses 0.9 for exactly that reason).
    cfo = 3.5  # subcarriers of offset -- illustrative only, see note above
    n = np.arange(n_symbols)
    phase = np.exp(1j * (2 * np.pi * cfo * n / fft_size))
    rotated = symbols * phase
    corrected = rotated * np.conj(phase)  # exact removal, known offset

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    for ax, data, title, color in (
        (axes[0], rotated, f"uncorrected  (cfo = {cfo} subcarriers)", RED),
        (axes[1], corrected, "corrected  (offset removed)", TEAL),
    ):
        ax.scatter(data.real, data.imag, s=10, alpha=0.55, color=color, edgecolors="none")
        ax.axhline(0, color=GRID, linewidth=0.8)
        ax.axvline(0, color=GRID, linewidth=0.8)
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("I")
    axes[0].set_ylabel("Q")
    fig.suptitle("What an uncorrected carrier frequency offset does to a QPSK constellation",
                 color=TEXT, fontsize=12, fontweight="bold", y=1.02)
    _save(fig, "cfo_rotation")


def _rx_process_with_captured_payload_symbols(ofdm, rx_iq):
    """Runs the REAL ofdm.rx_process(rx_iq) unmodified, but temporarily
    wraps ofdm.equalizer.process() to record every array it returns --
    this is how the actual equalized payload symbols get pulled out for
    plotting, since rx_process() itself only returns hard-decision bits
    downstream of them, not the complex symbols. Not a re-derivation:
    the real pipeline runs exactly as it always does, this only observes
    its own internal call. Header-stage equalizer calls (num_symbols_header
    of them) happen first and are sliced off; the remainder, in order, are
    the real payload-stage equalized symbols. Verified below (in each
    caller) against rx_process()'s own reported EVM before being trusted
    for display."""
    xp = ofdm.xp
    captured = []
    original = ofdm.equalizer.process

    def _wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        captured.append(xp.asarray(out))
        return out

    ofdm.equalizer.process = _wrapped
    try:
        result = ofdm.rx_process(rx_iq)
    finally:
        ofdm.equalizer.process = original

    payload_chunks = captured[ofdm.num_symbols_header:]
    payload_symbols = np.asarray(xp.concatenate(payload_chunks, axis=-1))[0] if payload_chunks else None
    return result, payload_symbols


# =========================================================================
# Figure 3: real Ofdm + Channel(cfo=...) round trip, SchmidlCoxCFO recovers it
# =========================================================================
def fig_cfo_recovered():
    print("fig_cfo_recovered...")
    fft_size = 256
    ofdm = Ofdm(
        fft_size=fft_size, n_pilot=8, n_data=216, cp_len=32, modem="qpsk",
        crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
        channel_estimator="ls", equalizer="mmse",
        n_training_symbols=2, backend="numpy",
    )
    rng = np.random.default_rng(11)
    bits = rng.integers(0, 2, size=(1, 4000)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)

    channel = Channel(snr_db=25.0, cfo=0.9, cfo_fft_size=fft_size, seed=1, backend="numpy")
    rx_iq = channel.process(tx_iq)
    result, payload_symbols = _rx_process_with_captured_payload_symbols(ofdm, rx_iq)
    assert bool(np.asarray(result["frame_found"]))
    assert bool(np.asarray(result["crc_valid"])[0])
    cfo_est = float(np.asarray(result["cfo_estimate"])[0])
    evm_reported = float(np.asarray(result["evm"])[0])

    # Verify the captured symbols are genuinely the ones rx_process()
    # itself used, not a look-alike: recompute EVM the same way
    # framing/stats.py's compute_evm() does, against the SAME hard-
    # decision re-modulation rx_process() uses internally, and require it
    # to match the real reported EVM closely.
    hard_bits = np.asarray(ofdm.modem.demodulate(payload_symbols[None, :]))
    ideal = np.asarray(ofdm.modem.modulate(hard_bits))[0]
    evm_check = float(np.sqrt(np.mean(np.abs(payload_symbols - ideal) ** 2) / np.mean(np.abs(ideal) ** 2)))
    assert abs(evm_check - evm_reported) < 1e-3, (evm_check, evm_reported)
    print(f"  true cfo=0.900, estimated={cfo_est:.3f}, reported EVM={evm_reported:.4f}, "
          f"recomputed from captured symbols={evm_check:.4f} (verified match)")

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.scatter(payload_symbols.real, payload_symbols.imag, s=10, alpha=0.5, color=TEAL, edgecolors="none")
    ax.axhline(0, color=GRID, linewidth=0.8)
    ax.axvline(0, color=GRID, linewidth=0.8)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal")
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title(f"real decoded payload symbols, after SchmidlCoxCFO correction\n"
                 f"estimated cfo={cfo_est:.3f} (true 0.900), EVM={evm_reported:.4f}",
                 fontsize=10.5)
    _save(fig, "cfo_recovered")


# =========================================================================
# Figure 4: channel estimation + equalization, real multipath
# =========================================================================
def fig_channel_equalization():
    print("fig_channel_equalization...")
    fft_size = 256
    ofdm = Ofdm(
        fft_size=fft_size, n_pilot=8, n_data=216, cp_len=32, modem="qpsk",
        crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
        channel_estimator="ls", equalizer="mmse",
        n_training_symbols=2, backend="numpy",
    )
    taps = Channel.random_multipath_taps(4, seed=1)
    channel = Channel(snr_db=26.0, multipath_taps=taps, seed=1, backend="numpy")

    rng = np.random.default_rng(9)
    bits = rng.integers(0, 2, size=(1, 2000)).astype("uint8")
    tx_iq = ofdm.generate_frame(bits)
    # Real, pre-existing edge case, not introduced by this script: multipath
    # delay spread can shift the detected frame start a few samples later
    # than the noiseless case, and generate_frame()'s output has zero
    # built-in trailing margin beyond exactly what a nominal-start decode
    # needs -- a late-detected start then runs the last payload slot past
    # the end of the buffer. Padding trailing silence, exactly as any real
    # captured buffer would have some idle samples after a frame, avoids
    # it here; this is not a fix to the underlying limitation (worth its
    # own investigation, see docs/todo.md #1.11's neighboring territory).
    tx_iq = np.concatenate([tx_iq, np.zeros((1, 64), dtype=tx_iq.dtype)], axis=-1)
    rx_iq = channel.process(tx_iq)
    result, payload_symbols = _rx_process_with_captured_payload_symbols(ofdm, rx_iq)
    assert bool(np.asarray(result["frame_found"]))
    assert bool(np.asarray(result["crc_valid"])[0])

    h_hat = np.asarray(result["channel_estimate"])[0]  # (n_data,) complex, LS/MMSE estimate at data bins
    data_indices = ofdm.grid.data_indices
    H_true_full = np.fft.fft(np.asarray(taps), n=fft_size)
    H_true_data = H_true_full[data_indices]

    order = np.argsort(data_indices)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    axes[0].plot(data_indices[order], np.abs(H_true_data)[order], color=AMBER, linewidth=1.6,
                 label="true |H| (4-tap multipath)", marker=".", markersize=3)
    axes[0].plot(data_indices[order], np.abs(h_hat)[order], color=TEAL, linewidth=1.1, linestyle="--",
                 label="LS channel estimate |Ĥ|", alpha=0.9)
    axes[0].set_xlabel("subcarrier bin")
    axes[0].set_ylabel("|H|")
    axes[0].set_title("channel estimate vs. true response (data subcarriers)", fontsize=10.5)
    axes[0].legend(facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=8.5, loc="best")

    evm = float(np.asarray(result["evm"])[0])
    hard_bits = np.asarray(ofdm.modem.demodulate(payload_symbols[None, :]))
    ideal = np.asarray(ofdm.modem.modulate(hard_bits))[0]
    evm_check = float(np.sqrt(np.mean(np.abs(payload_symbols - ideal) ** 2) / np.mean(np.abs(ideal) ** 2)))
    assert abs(evm_check - evm) < 1e-3, (evm_check, evm)
    print(f"  reported EVM={evm:.4f}, recomputed from captured symbols={evm_check:.4f} (verified match)")
    axes[1].scatter(payload_symbols.real, payload_symbols.imag, s=10, alpha=0.5, color=TEAL, edgecolors="none")
    axes[1].axhline(0, color=GRID, linewidth=0.8)
    axes[1].axvline(0, color=GRID, linewidth=0.8)
    axes[1].set_xlim(-1.6, 1.6)
    axes[1].set_ylim(-1.6, 1.6)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("I")
    axes[1].set_ylabel("Q")
    axes[1].set_title(f"after MMSE equalization\nEVM={evm:.4f}", fontsize=10.5)
    fig.suptitle("4-tap multipath, snr=26 dB — LS estimate + MMSE equalizer", color=TEXT,
                 fontsize=12, fontweight="bold", y=1.02)
    _save(fig, "channel_equalization")


# =========================================================================
# Figure 5: constellation gallery, every modem scheme
# =========================================================================
def fig_constellation_gallery():
    print("fig_constellation_gallery...")
    schemes = ["bpsk", "qpsk", "qam16", "qam64", "qam256"]
    fig, axes = plt.subplots(1, len(schemes), figsize=(4.2 * len(schemes), 4.2))
    for ax, scheme in zip(axes, schemes):
        modem = Modem(scheme, backend="numpy")
        k = modem.bits_per_symbol
        n_points = 1 << k
        all_ints = np.arange(n_points, dtype="int64")
        bits = ((all_ints[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype("uint8")[None, :, :].reshape(1, -1)
        symbols = np.asarray(modem.modulate(bits))[0]
        lim = np.max(np.abs(symbols)) * 1.35
        ax.scatter(symbols.real, symbols.imag, s=26, color=TEAL, edgecolors=AMBER, linewidths=0.6)
        ax.axhline(0, color=GRID, linewidth=0.8)
        ax.axvline(0, color=GRID, linewidth=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(f"{scheme}  ({n_points} points, {k} bit/sym)", fontsize=10.5)
        ax.set_xlabel("I")
    axes[0].set_ylabel("Q")
    fig.suptitle("Modem(scheme) — every supported constellation, unit average power",
                 color=TEXT, fontsize=12, fontweight="bold", y=1.03)
    _save(fig, "constellation_gallery")


if __name__ == "__main__":
    fig_sync_metric()
    fig_cfo_rotation()
    fig_cfo_recovered()
    fig_channel_equalization()
    fig_constellation_gallery()
    print("done.")
