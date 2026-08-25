"""The full 4-Mac/4-Ofdm bidirectional picture, AM mode, decoding through
Ofdm.rx_process() (batch -- whole-frame-buffer decode; see
mac_bidirectional_am_streaming_demo.py for the same scenario through
Ofdm.rx_streaming() instead).

Model (see docs/mac.md, "4-Mac/4-Ofdm bidirectional"):

    HW1.tx_mac  -- owns Ofdm_A   (forward-direction PHY: hw1 -> hw2)
    HW2.rx_mac  -- owns Ofdm_A'  (SAME config as Ofdm_A, separate object)
    HW2.tx_mac  -- owns Ofdm_B   (reverse-direction PHY: hw2 -> hw1)
    HW1.rx_mac  -- owns Ofdm_B'  (SAME config as Ofdm_B, separate object)

4 independently-constructed Mac objects, 4 independently-constructed Ofdm
objects, all mode="am". Fully manual wiring throughout -- no orchestrator
class (see docs/mac.md's earlier design discussion: "it is fully manual
it is responsibility of developer anyway").

The wrinkle AM adds on top of the two-unit/half-duplex UM demos: a STATUS
pdu reporting on traffic received in ONE direction has to physically
travel in the OTHER direction, and is decoded by a DIFFERENT Mac object
than the one whose retransmission buffer it's actually about:

  - hw2_rx_mac (received hw1's DATA over Ofdm_A) builds the STATUS.
  - It travels hw2 -> hw1 physically over Ofdm_B (hw2_tx_mac's transmit
    chain -- the same chain hw2 uses for its OWN reverse-direction DATA).
  - hw1_rx_mac (owns Ofdm_B', the matching receive side) decodes it.
  - hw1_tx_mac (a DIFFERENT Mac object at HW1, the one that actually
    holds the retransmission buffer for the DATA it sent) is the one
    that calls receive_status() on the decoded bits.

Mac.send_iq()/receive_iq()/build_bind_request()/handle_bind_request_iq()
would do all of this, but ONLY EVER hand back/accept IQ arrays -- the
actual PDU bytes crossing the air (bind handshake included) are
invisible from outside. This file deliberately does NOT call those
convenience methods for the parts it wants to show -- it inlines their
exact same logic (see spectracuda/mac/mac.py's own send_iq()/
handle_bind_request_iq() bodies, which this mirrors 1:1) so every PDU,
in both directions, at every stage (BIND_REQUEST, BIND_RESPONSE, DATA,
STATUS), can be printed as hex before it becomes IQ and right after it's
decoded back out of IQ.
"""
from __future__ import annotations

import numpy as np

from spectracuda.backend import default_backend
from spectracuda.mac import Mac
from spectracuda.mac.bind import decode_bind_response, encode_bind_request
from spectracuda.mac.pdu import (
    HEADER_LEN_BITS,
    TYPE_BIND_REQUEST,
    TYPE_BIND_RESPONSE,
    TYPE_DATA,
    TYPE_LINK_QUALITY,
    TYPE_STATUS,
    decode_header,
)
from spectracuda.sim import Channel

_TYPE_NAMES = {
    TYPE_DATA: "DATA", TYPE_STATUS: "STATUS",
    TYPE_BIND_REQUEST: "BIND_REQUEST", TYPE_BIND_RESPONSE: "BIND_RESPONSE",
    TYPE_LINK_QUALITY: "LINK_QUALITY",
}

# Deliberately small capacity (small fft_size/n_data, bpsk) so a modest
# SDU genuinely segments into multiple PDUs -- otherwise a single-PDU
# demo could never show a real dropped-segment retransmission. PHY_A and
# PHY_B are two SEPARATE dict literals (not one shared and reused) to
# underline that they're independent per-direction configs -- they
# happen to be identical here for simplicity; making them genuinely
# different (e.g. different modem=) is already proven not to matter by
# test_real_cross_object_bind_genuinely_rejects_a_mismatched_capacity.
PHY_A = dict(
    fft_size=64, n_pilot=4, n_data=16, cp_len=16, modem="bpsk",
    crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
    n_training_symbols=2, backend=default_backend(),  # "cupy" if a working CUDA runtime is present, else "numpy"
)
PHY_B = dict(
    fft_size=64, n_pilot=4, n_data=16, cp_len=16, modem="bpsk",
    crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
    n_training_symbols=2, backend=default_backend(),  # "cupy" if a working CUDA runtime is present, else "numpy"
)

# snr_db=40, not the 25-30 dB used elsewhere in the mac_*_demo.py files:
# a max-length PDU at this tiny fft_size (many OFDM symbols per frame,
# needed to reach 2000 bits of payload over only 16 data subcarriers)
# runs into the still-open "long-frame reliability" gap (docs/todo.md
# #1.11 -- EVM degrades with frame length, root cause not yet nailed
# down) at more realistic SNRs. Raised here specifically so a real
# encode/channel/decode round trip is what demonstrates the AM
# retransmission logic, not a channel so lossy it would fail regardless
# of AM even being involved.


def _hex(bit_array, max_bytes=None):
    """Pack a 0/1 bit array (as every pdu.py function represents a PDU)
    into hex bytes for display. Truncates long payloads to max_bytes
    (with an explicit "...(N bytes total)" marker) so a several-hundred-
    byte DATA PDU doesn't flood the terminal -- the header (always
    printed separately, decoded) is what actually identifies the frame;
    the hex is there to show real bytes crossed the air, not to be read
    byte-by-byte."""
    bits = np.asarray(bit_array, dtype="uint8")
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype="uint8")])
    raw = np.packbits(bits).tobytes()
    if max_bytes is not None and len(raw) > max_bytes:
        return raw[:max_bytes].hex() + f"...({len(raw)} bytes total)"
    return raw.hex()


def _describe(pdu_bits, max_bytes=16):
    h = decode_header(pdu_bits[:HEADER_LEN_BITS])
    type_name = _TYPE_NAMES.get(h["pdu_type"], h["pdu_type"])
    return f"{type_name} si={h['si']} sn={h['sn']} so={h['so']} hex={_hex(pdu_bits, max_bytes)}"


def _bind(mac_a, mac_b, label, verbose=True):
    """The real 3-call handshake (see docs/mac.md) -- send_iq() requires
    a successful bind() first. Mirrors Mac.build_bind_request()/
    handle_bind_request_iq()/handle_bind_response_iq() exactly (same
    calls, same order, same real accept/reject decision on mac_b's own
    capacity) but keeps the pre-IQ PDU bits around so they can be
    printed -- those methods only ever return/accept IQ."""
    request_pdu = encode_bind_request(mac_a.mode, mac_a.max_segment_bits, mac_a._window_size, mac_a._max_retries)
    if verbose:
        print(f"[{label}] BIND_REQUEST  sent     {_describe(request_pdu)}")
    req_iq = mac_a.ofdm.generate_frame(request_pdu[None, :])  # == mac_a.build_bind_request()

    resp_iq = mac_b.handle_bind_request_iq(req_iq)  # mac_b's REAL decode + accept/reject decision
    assert resp_iq is not None

    # Decode the response ourselves too, purely for display -- rx_process()
    # is stateless (no persistent Ofdm state it mutates), so this doesn't
    # disturb the real handle_bind_response_iq() call below, which redoes
    # the identical decode for real (and is what actually sets mac_a.bound).
    resp_result = mac_a.ofdm.rx_process(resp_iq)
    resp_bits = np.asarray(resp_result["bits"])[0].astype("uint8")
    decision = decode_bind_response(resp_bits)
    if verbose:
        print(f"[{label}] BIND_RESPONSE received {_describe(resp_bits)}  "
              f"-> accepted={decision['accepted']} reason={decision['reason']!r}")

    assert mac_a.handle_bind_response_iq(resp_iq)
    if verbose:
        print(f"[{label}] bound: mac_a.bound={mac_a.bound} mac_b.bound={mac_b.bound}")


def _send(mac, sdu, label, verbose=True):
    """The exact same thing Mac.send_iq() does internally
    (self._impl.transmit() -> per-PDU ofdm.generate_frame()) -- inlined
    here so each PDU's raw bytes can be printed before they're turned
    into IQ, which send_iq() itself has no way to expose."""
    pdus = mac.transmit(sdu)  # PDU-level AmEntity.transmit(), via Mac.__getattr__
    iq_frames = []
    for i, pdu in enumerate(pdus):
        if verbose:
            print(f"[{label}] DATA PDU #{i} sent     {_describe(pdu)}")
        iq_frames.append(mac.ofdm.generate_frame(np.asarray(pdu, dtype="uint8")[None, :]))
    return iq_frames


def _rx_raw_pdu(mac, iq, label, verbose=True):
    """Decode one arrived IQ frame into raw PDU bits (post header/FEC/
    CRC), WITHOUT assuming its pdu_type -- unlike Mac.receive_iq(), which
    for AM mode always calls receive_data() and would raise on a STATUS
    pdu. This is the same rx_process()/CRC/quality-tracking logic
    Mac._rx_one_frame() uses internally, written out explicitly here
    because this file needs to decode BOTH DATA and STATUS frames
    depending on what's physically arriving on a given Ofdm at a given
    moment -- and to print what actually arrived. Returns None if the
    frame didn't arrive usably."""
    result = mac.ofdm.rx_process(np.asarray(iq))
    crc_valid = result["crc_valid"]
    delivered = bool(result["frame_found"]) and (crc_valid is None or bool(np.asarray(crc_valid)[0]))
    evm = result["evm"]
    mac.quality.observe(
        rssi_db=float(np.asarray(result["rssi_db"])[0]),
        evm=None if evm is None else float(np.asarray(evm)[0]),
        delivered=delivered,
    )
    if not delivered:
        if verbose:
            print(f"[{label}] frame NOT delivered (frame_found={bool(result['frame_found'])}, "
                  f"crc_valid={None if crc_valid is None else bool(np.asarray(crc_valid)[0])})")
        return None
    bits = np.asarray(result["bits"])[0].astype("uint8")
    if verbose:
        print(f"[{label}] PDU received     {_describe(bits)}")
    return bits


def run(seed: int = 0, snr_db: float = 40.0, verbose: bool = True):
    if verbose:
        print("=== 4-Mac/4-Ofdm bidirectional AM (batch rx_process) ===\n")

    hw1_tx_mac = Mac(mode="am", ofdm_kwargs=PHY_A)  # owns Ofdm_A
    hw2_rx_mac = Mac(mode="am", ofdm_kwargs=PHY_A)  # owns Ofdm_A'
    hw2_tx_mac = Mac(mode="am", ofdm_kwargs=PHY_B)  # owns Ofdm_B
    hw1_rx_mac = Mac(mode="am", ofdm_kwargs=PHY_B)  # owns Ofdm_B'
    assert len({id(m.ofdm) for m in (hw1_tx_mac, hw2_rx_mac, hw2_tx_mac, hw1_rx_mac)}) == 4

    _bind(hw1_tx_mac, hw2_rx_mac, "bind fwd (hw1->hw2, Ofdm_A)", verbose)
    _bind(hw2_tx_mac, hw1_rx_mac, "bind rev (hw2->hw1, Ofdm_B)", verbose)
    if verbose:
        print(f"\nmax_segment_bits: forward={hw1_tx_mac.max_segment_bits}, "
              f"reverse={hw2_tx_mac.max_segment_bits}")

    channel = Channel(snr_db=snr_db, seed=seed, backend=default_backend())
    rng = np.random.default_rng(seed)
    sdu_fwd = rng.integers(0, 2, size=4800).astype("uint8")  # hw1 -> hw2
    sdu_rev = rng.integers(0, 2, size=1200).astype("uint8")  # hw2 -> hw1

    # ---- forward direction: hw1 -> hw2, with one PDU deliberately lost ----
    fwd_frames = _send(hw1_tx_mac, sdu_fwd, "fwd", verbose)
    assert len(fwd_frames) >= 3, "demo needs real segmentation to mean anything"
    drop_index = len(fwd_frames) // 2
    if verbose:
        print(f"\nforward: {len(sdu_fwd)} bits -> {len(fwd_frames)} PDUs; "
              f"dropping PDU #{drop_index} to force a real NACK+retransmit\n")

    delivered_fwd = []
    for i, iq in enumerate(fwd_frames):
        if i == drop_index:
            if verbose:
                print(f"[fwd] DATA PDU #{i} DROPPED  (never reaches hw2_rx_mac at all)")
            continue  # simulate a lost frame
        bits = _rx_raw_pdu(hw2_rx_mac, channel.process(iq), "fwd", verbose)
        if bits is not None:
            delivered_fwd.extend(hw2_rx_mac.receive_data(bits))
    assert delivered_fwd == []  # genuinely incomplete after round 1 -- reassembly can't finish
    if verbose:
        print(f"\nafter round 1 (1 PDU missing): delivered={len(delivered_fwd)} SDU(s) -- "
              f"confirmed incomplete, as expected\n")

    status_pdu = hw2_rx_mac.build_status()  # PDU-level, hw2_rx_mac's own AmEntity
    if verbose:
        print(f"[fwd status] STATUS sent      {_describe(status_pdu)}  (hw2_rx_mac -- built)")
    status_iq = hw2_tx_mac.ofdm.generate_frame(status_pdu[None, :])  # sent over Ofdm_B
    status_bits = _rx_raw_pdu(hw1_rx_mac, channel.process(status_iq), "fwd status", verbose)  # over Ofdm_B'
    assert status_bits is not None
    retransmits = hw1_tx_mac.receive_status(status_bits)  # hw1_tx_mac's OWN retx buffer
    assert len(retransmits) == 1  # exactly the one dropped PDU, nothing extra
    if verbose:
        print(f"[fwd status] routed hw2_rx_mac --(Ofdm_B)--> hw1_rx_mac --> "
              f"hw1_tx_mac.receive_status(): {len(retransmits)} PDU(s) to retransmit\n")

    for pdu in retransmits:
        if verbose:
            print(f"[fwd retx] DATA PDU resent {_describe(pdu)}")
        iq = hw1_tx_mac.ofdm.generate_frame(pdu[None, :])  # resent over Ofdm_A
        bits = _rx_raw_pdu(hw2_rx_mac, channel.process(iq), "fwd retx", verbose)
        if bits is not None:
            delivered_fwd.extend(hw2_rx_mac.receive_data(bits))

    ok_fwd = len(delivered_fwd) == 1 and np.array_equal(delivered_fwd[0], sdu_fwd)
    if verbose:
        print(f"\nafter retransmission: delivered={len(delivered_fwd)} SDU(s), correct={ok_fwd}\n")

    # ---- reverse direction: hw2 -> hw1, clean (nothing dropped) ----
    rev_frames = _send(hw2_tx_mac, sdu_rev, "rev", verbose)
    delivered_rev = []
    for iq in rev_frames:
        bits = _rx_raw_pdu(hw1_rx_mac, channel.process(iq), "rev", verbose)
        if bits is not None:
            delivered_rev.extend(hw1_rx_mac.receive_data(bits))

    status_pdu2 = hw1_rx_mac.build_status()
    if verbose:
        print(f"\n[rev status] STATUS sent      {_describe(status_pdu2)}  (hw1_rx_mac -- built)")
    status_iq2 = hw1_tx_mac.ofdm.generate_frame(status_pdu2[None, :])  # sent over Ofdm_A
    status_bits2 = _rx_raw_pdu(hw2_rx_mac, channel.process(status_iq2), "rev status", verbose)  # over Ofdm_A'
    assert status_bits2 is not None
    retransmits2 = hw2_tx_mac.receive_status(status_bits2)
    assert retransmits2 == []  # nothing lost -- nothing to resend
    if verbose:
        print(f"[rev status] routed hw1_rx_mac --(Ofdm_A)--> hw2_rx_mac --> "
              f"hw2_tx_mac.receive_status(): {len(retransmits2)} PDU(s) to retransmit")
        print(f"\nreverse: {len(sdu_rev)} bits -> {len(rev_frames)} PDU(s), delivered cleanly")

    ok_rev = len(delivered_rev) == 1 and np.array_equal(delivered_rev[0], sdu_rev)
    if verbose:
        print(f"reverse correct={ok_rev}")
        print(f"\nBoth directions correct, 4 independent Mac/Ofdm objects, "
              f"one real retransmission proven: {ok_fwd and ok_rev}")

    return {
        "n_distinct_ofdm_objects": 4,
        "forward_pdus": len(fwd_frames),
        "forward_retransmits": len(retransmits),
        "forward_correct": ok_fwd,
        "reverse_pdus": len(rev_frames),
        "reverse_retransmits": len(retransmits2),
        "reverse_correct": ok_rev,
    }


if __name__ == "__main__":
    run()
