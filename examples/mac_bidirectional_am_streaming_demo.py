"""The same 4-Mac/4-Ofdm bidirectional AM picture as
mac_bidirectional_am_batch_demo.py, decoding through Ofdm.rx_streaming()
(chunked, arbitrarily-aligned IQ) instead of Ofdm.rx_process() (one-shot,
whole-buffer). See that file's module docstring for the full model, the
STATUS-routing wrinkle, and why the bind handshake and every PDU are
printed by inlining Mac's own send_iq()/handle_bind_request_iq() logic
instead of calling those (IQ-only) convenience methods directly -- this
file only changes HOW each receiving Ofdm consumes its arriving IQ, not
the AM/bind/routing/printing logic itself, same relationship
examples/mac_streaming_demo.py already has to mac_two_units_demo.py /
mac_half_duplex_demo.py.

Every receiving Ofdm here (hw2_rx_mac.ofdm for forward DATA, hw1_rx_mac.ofdm
for reverse DATA AND for the forward-direction STATUS, hw2_rx_mac.ofdm
again for the reverse-direction STATUS) needs its OWN independent
rx_streaming() state machine -- reset_stream() is called once per Ofdm,
not per frame, exactly like mac_streaming_demo.py's precedent.
"""
from __future__ import annotations

import numpy as np

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

# Same PHY choice and same snr_db=40 rationale as the batch demo (see its
# module docstring / PHY_A comment) -- this file isn't where the
# long-frame-reliability gap (docs/todo.md #1.11) is being explored.
PHY_A = dict(
    fft_size=64, n_pilot=4, n_data=16, cp_len=16, modem="bpsk",
    crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
    n_training_symbols=2, backend="numpy",
)
PHY_B = dict(
    fft_size=64, n_pilot=4, n_data=16, cp_len=16, modem="bpsk",
    crc="crc16", sync="schmidl_cox", cfo="schmidl_cox",
    n_training_symbols=2, backend="numpy",
)

CHUNK_SIZE = 64  # arbitrary, same as mac_streaming_demo.py


def _hex(bit_array, max_bytes=None):
    """See mac_bidirectional_am_batch_demo.py's _hex() -- identical,
    duplicated rather than cross-imported (same convention every other
    mac_*_demo.py file already follows for PHY_KWARGS etc.)."""
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
    """The real 3-call handshake -- binding itself still goes through
    rx_process() (batch), not rx_streaming(): it's a one-shot control PDU,
    not the thing this file exists to demonstrate streamed decode of
    (same convention as mac_streaming_demo.py's own _bind()). Mirrors
    Mac.build_bind_request()/handle_bind_request_iq()/
    handle_bind_response_iq() exactly, but keeps the pre-IQ PDU bits
    around to print -- see the batch demo's _bind() for the same
    approach, explained in full there."""
    request_pdu = encode_bind_request(mac_a.mode, mac_a.max_segment_bits, mac_a._window_size, mac_a._max_retries)
    if verbose:
        print(f"[{label}] BIND_REQUEST  sent     {_describe(request_pdu)}")
    req_iq = mac_a.ofdm.generate_frame(request_pdu[None, :])

    resp_iq = mac_b.handle_bind_request_iq(req_iq)  # mac_b's REAL decode + accept/reject decision
    assert resp_iq is not None

    resp_result = mac_a.ofdm.rx_process(resp_iq)  # stateless, purely for display (see batch demo)
    resp_bits = np.asarray(resp_result["bits"])[0].astype("uint8")
    decision = decode_bind_response(resp_bits)
    if verbose:
        print(f"[{label}] BIND_RESPONSE received {_describe(resp_bits)}  "
              f"-> accepted={decision['accepted']} reason={decision['reason']!r}")

    assert mac_a.handle_bind_response_iq(resp_iq)
    if verbose:
        print(f"[{label}] bound: mac_a.bound={mac_a.bound} mac_b.bound={mac_b.bound}")


def _send(mac, sdu, label, verbose=True):
    """The exact same thing Mac.send_iq() does internally -- inlined so
    each PDU's raw bytes can be printed before they're turned into IQ
    (see the batch demo's _send() for the same approach)."""
    pdus = mac.transmit(sdu)
    iq_frames = []
    for i, pdu in enumerate(pdus):
        if verbose:
            print(f"[{label}] DATA PDU #{i} sent     {_describe(pdu)}")
        iq_frames.append(mac.ofdm.generate_frame(np.asarray(pdu, dtype="uint8")[None, :]))
    return iq_frames


def _stream_rx_raw_pdu(ofdm, iq, label, verbose=True, quality=None):
    """Feed one IQ frame through ofdm.rx_streaming() in CHUNK_SIZE pieces
    and return the raw decoded PDU bits (post header/FEC/CRC) once a
    complete frame finishes -- the streaming counterpart of the batch
    demo's _rx_raw_pdu(), used here instead of Mac.receive_iq() because
    the arriving frame's pdu_type isn't known in advance (DATA or
    STATUS) and Mac.receive_iq() assumes DATA for AM mode. Optionally
    feeds a LinkQualityTracker, same as Mac._rx_one_frame() does, and
    prints the decoded PDU, same as the batch demo's version."""
    samples = np.asarray(iq)[0]  # (1, N) -> (N,), generate_frame()'s own batch dim
    for i in range(0, samples.shape[-1], CHUNK_SIZE):
        chunk = samples[i : i + CHUNK_SIZE]
        result = ofdm.rx_streaming(chunk)
        if result is None:
            continue
        crc_valid = result["crc_valid"]
        delivered = crc_valid is None or bool(np.asarray(crc_valid)[0])  # frame_found always True here
        if quality is not None:
            evm = result["evm"]
            quality.observe(
                rssi_db=float(np.asarray(result["rssi_db"])[0]),
                evm=None if evm is None else float(np.asarray(evm)[0]),
                delivered=delivered,
            )
        if not delivered:
            if verbose:
                print(f"[{label}] frame NOT delivered (crc_valid=False, after {i // CHUNK_SIZE + 1} chunks)")
            return None
        bits = np.asarray(result["bits"])[0].astype("uint8")
        if verbose:
            print(f"[{label}] PDU received     {_describe(bits)}  (after {i // CHUNK_SIZE + 1} chunks "
                  f"of {CHUNK_SIZE} samples)")
        return bits
    return None  # ran out of chunks without a complete frame -- shouldn't happen here


def _stream_receive_iq(mac, iq, label, verbose=True):
    """The streaming counterpart of Mac.receive_iq() for AM DATA frames:
    decode via mac.ofdm.rx_streaming() instead of rx_process(), then hand
    off to mac.receive_data() exactly as Mac.receive_iq() does
    internally. Mac itself has no built-in streaming IQ-level method
    (see docs/todo.md #2.5 -- rx_streaming() is Ofdm-level only), so this
    is written out explicitly here, same spirit as mac_streaming_demo.py's
    own _stream_deliver()."""
    bits = _stream_rx_raw_pdu(mac.ofdm, iq, label, verbose, quality=mac.quality)
    if bits is None:
        return []
    return mac.receive_data(bits)


def run(seed: int = 0, snr_db: float = 40.0, verbose: bool = True):
    if verbose:
        print("=== 4-Mac/4-Ofdm bidirectional AM (streaming rx_streaming) ===\n")

    hw1_tx_mac = Mac(mode="am", ofdm_kwargs=PHY_A)  # owns Ofdm_A
    hw2_rx_mac = Mac(mode="am", ofdm_kwargs=PHY_A)  # owns Ofdm_A'
    hw2_tx_mac = Mac(mode="am", ofdm_kwargs=PHY_B)  # owns Ofdm_B
    hw1_rx_mac = Mac(mode="am", ofdm_kwargs=PHY_B)  # owns Ofdm_B'
    assert len({id(m.ofdm) for m in (hw1_tx_mac, hw2_rx_mac, hw2_tx_mac, hw1_rx_mac)}) == 4

    _bind(hw1_tx_mac, hw2_rx_mac, "bind fwd (hw1->hw2, Ofdm_A)", verbose)  # batch, see docstring
    _bind(hw2_tx_mac, hw1_rx_mac, "bind rev (hw2->hw1, Ofdm_B)", verbose)  # batch, see docstring
    if verbose:
        print(f"\nmax_segment_bits: forward={hw1_tx_mac.max_segment_bits}, "
              f"reverse={hw2_tx_mac.max_segment_bits}")

    # Every RECEIVING Ofdm gets its own streaming state, once, up front --
    # hw2_rx_mac.ofdm receives forward DATA; hw1_rx_mac.ofdm receives
    # reverse DATA *and* the forward-direction STATUS (both physically
    # arrive over Ofdm_B'); hw2_rx_mac.ofdm ALSO receives the
    # reverse-direction STATUS (arrives over Ofdm_A', same object as the
    # forward DATA -- one stream, multiple frame types, same as a real
    # receive chain would see).
    hw2_rx_mac.ofdm.reset_stream()
    hw1_rx_mac.ofdm.reset_stream()

    channel = Channel(snr_db=snr_db, seed=seed, backend="numpy")
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
                print(f"[fwd] DATA PDU #{i} DROPPED  (never fed into rx_streaming() at all)")
            continue  # simulate a lost frame
        delivered_fwd.extend(_stream_receive_iq(hw2_rx_mac, channel.process(iq), "fwd", verbose))
    assert delivered_fwd == []
    if verbose:
        print(f"\nafter round 1 (1 PDU missing): delivered={len(delivered_fwd)} SDU(s) -- "
              f"confirmed incomplete, as expected\n")

    status_pdu = hw2_rx_mac.build_status()
    if verbose:
        print(f"[fwd status] STATUS sent      {_describe(status_pdu)}  (hw2_rx_mac -- built)")
    status_iq = hw2_tx_mac.ofdm.generate_frame(status_pdu[None, :])  # sent over Ofdm_B
    status_bits = _stream_rx_raw_pdu(hw1_rx_mac.ofdm, channel.process(status_iq), "fwd status", verbose,
                                      quality=hw1_rx_mac.quality)
    assert status_bits is not None
    retransmits = hw1_tx_mac.receive_status(status_bits)
    assert len(retransmits) == 1
    if verbose:
        print(f"[fwd status] routed hw2_rx_mac --(Ofdm_B, streamed)--> hw1_rx_mac --> "
              f"hw1_tx_mac.receive_status(): {len(retransmits)} PDU(s) to retransmit\n")

    for pdu in retransmits:
        if verbose:
            print(f"[fwd retx] DATA PDU resent {_describe(pdu)}")
        iq = hw1_tx_mac.ofdm.generate_frame(pdu[None, :])  # resent over Ofdm_A
        delivered_fwd.extend(_stream_receive_iq(hw2_rx_mac, channel.process(iq), "fwd retx", verbose))

    ok_fwd = len(delivered_fwd) == 1 and np.array_equal(delivered_fwd[0], sdu_fwd)
    if verbose:
        print(f"\nafter retransmission: delivered={len(delivered_fwd)} SDU(s), correct={ok_fwd}\n")

    # ---- reverse direction: hw2 -> hw1, clean (nothing dropped) ----
    rev_frames = _send(hw2_tx_mac, sdu_rev, "rev", verbose)
    delivered_rev = []
    for iq in rev_frames:
        delivered_rev.extend(_stream_receive_iq(hw1_rx_mac, channel.process(iq), "rev", verbose))

    status_pdu2 = hw1_rx_mac.build_status()
    if verbose:
        print(f"\n[rev status] STATUS sent      {_describe(status_pdu2)}  (hw1_rx_mac -- built)")
    status_iq2 = hw1_tx_mac.ofdm.generate_frame(status_pdu2[None, :])  # sent over Ofdm_A
    status_bits2 = _stream_rx_raw_pdu(hw2_rx_mac.ofdm, channel.process(status_iq2), "rev status", verbose,
                                       quality=hw2_rx_mac.quality)
    assert status_bits2 is not None
    retransmits2 = hw2_tx_mac.receive_status(status_bits2)
    assert retransmits2 == []
    if verbose:
        print(f"[rev status] routed hw1_rx_mac --(Ofdm_A, streamed)--> hw2_rx_mac --> "
              f"hw2_tx_mac.receive_status(): {len(retransmits2)} PDU(s) to retransmit")
        print(f"\nreverse: {len(sdu_rev)} bits -> {len(rev_frames)} PDU(s), delivered cleanly (streamed)")

    ok_rev = len(delivered_rev) == 1 and np.array_equal(delivered_rev[0], sdu_rev)
    if verbose:
        print(f"reverse correct={ok_rev}")
        print(f"\nBoth directions correct via rx_streaming(), 4 independent Mac/Ofdm "
              f"objects, one real retransmission proven: {ok_fwd and ok_rev}")

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
