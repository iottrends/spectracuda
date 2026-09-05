"""Ofdm.reconfigure_tx_scheme(): in-place modem/fec/fec1 change for
adaptive MCS (see its own docstring in spectracuda/pipeline/ofdm.py for
the full "reconfigure, don't rebuild" rationale).

What this suite exists to prove, each independently:
1. A frame generated AFTER a scheme change decodes correctly on a
   default (strict_fec_check=False) receiver Ofdm that was NEVER
   reconfigured -- the header's own mod_scheme/fec0/fec1 fields
   (resolved from the wire, not from self -- see Ofdm's class
   docstring) are what make this work, exactly as it already does for
   two independently-configured Ofdm instances.
1b. The opt-in strict_fec_check=True mode (Ofdm.__init__) is the
   opposite by design: a receiver constructed with it REJECTS any
   decoded fec0/fec1 that doesn't equal its own self.fec/self.fec1 --
   see _decode_header_from_sync()'s own comment for why (constructing a
   codec, LDPC's GF(2) matrix inversion above all, for whatever a false
   sync detection decodes out of pure noise is a real, measured multi-
   hundred-ms-to-multi-second stall on real hardware -- see
   debug/pluto_rx_standalone_test.py's own commit history) -- and
   decodes fine again once the receiver is ALSO reconfigured to the
   same fec/fec1, proving the "both ends must move together" mechanism
   any future fec-negotiating feature would need. This project's own
   adaptive-MCS controller, examples/drone_tui/adaptive_mcs.py, never
   varies fec/fec1 in the first place (see 1. above), so opting into
   strict_fec_check costs it nothing real.
2. Fields NOT related to modem/fec/fec1 are untouched by a reconfigure
   (grid, header_codec, sync object identity, preamble, RX streaming
   buffer) -- this is what makes it safe on a full-duplex Mac's single
   shared Ofdm (see the docstring's "reason 1").
3. Leaving an argument as None leaves that piece alone.
4. Invalid fec/fec1 names raise ValueError, same as __init__ does.
5. The interleaver/fec1 interaction Packetizer.__init__ already enforces
   (interleaver != "none" requires fec1 != "none") surfaces correctly
   through reconfigure_tx_scheme() too, not just at construction time --
   this is the exact hazard examples/drone_tui/adaptive_mcs.py's own
   module docstring documents avoiding by only ever varying modem.
"""
import numpy as np
import pytest

from spectracuda.pipeline import Ofdm

_BASE = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32, crc="crc16",
    sync="schmidl_cox", cfo="schmidl_cox", channel_estimator="ls", equalizer="mmse",
    backend="numpy",
)


def _roundtrip_ok(tx: Ofdm, rx: Ofdm, bits: np.ndarray) -> bool:
    iq = tx.generate_frame(bits[None, :])
    r = rx.rx_process(iq)
    if not r["frame_found"] or r["crc_valid"] is None or not bool(r["crc_valid"][0]):
        return False
    return np.array_equal(r["bits"][0], bits)


def test_reconfigured_frame_decodes_on_an_untouched_receiver():
    tx = Ofdm(modem="qpsk", fec="conv_v27", **_BASE)
    rx = Ofdm(modem="qpsk", fec="conv_v27", **_BASE)  # never reconfigured, default strict_fec_check=False
    bits = np.random.default_rng(1).integers(0, 2, size=800).astype("uint8")

    assert _roundtrip_ok(tx, rx, bits)  # baseline, before any change

    new_bps = tx.reconfigure_tx_scheme(modem="qam16", fec="rs_m8")
    assert tx.modem.scheme == "qam16"
    assert tx.fec == "rs_m8"
    assert new_bps == tx.grid.n_data * tx.modem.bits_per_symbol
    assert _roundtrip_ok(tx, rx, bits)  # untouched rx still decodes it, via the header alone

    # switch again, modem only this time -- fec/fec1 must be left as-is
    tx.reconfigure_tx_scheme(modem="qam64")
    assert tx.fec == "rs_m8"
    assert tx.fec1 == "none"
    assert _roundtrip_ok(tx, rx, bits)


def test_strict_fec_check_rejects_fec_change_on_an_untouched_receiver():
    """See module docstring point 1b."""
    tx = Ofdm(modem="qpsk", fec="conv_v27", **_BASE)
    rx = Ofdm(modem="qpsk", fec="conv_v27", strict_fec_check=True, **_BASE)  # never reconfigured
    bits = np.random.default_rng(2).integers(0, 2, size=800).astype("uint8")

    assert _roundtrip_ok(tx, rx, bits)  # baseline (fec still matches) still works under strict mode

    tx.reconfigure_tx_scheme(fec="rs_m8")
    iq = tx.generate_frame(bits[None, :])
    with pytest.raises(ValueError, match="fec0"):
        rx.rx_process(iq)


def test_strict_fec_check_decodes_once_receiver_is_reconfigured_too():
    """The other half of 1b: both ends moving together works fine."""
    tx = Ofdm(modem="qpsk", fec="conv_v27", **_BASE)
    rx = Ofdm(modem="qpsk", fec="conv_v27", strict_fec_check=True, **_BASE)
    bits = np.random.default_rng(3).integers(0, 2, size=800).astype("uint8")

    tx.reconfigure_tx_scheme(fec="rs_m8")
    rx.reconfigure_tx_scheme(fec="rs_m8")  # the receiver must move too, not just the sender
    assert _roundtrip_ok(tx, rx, bits)


def test_unrelated_state_is_untouched():
    ofdm = Ofdm(modem="qpsk", fec="conv_v27", **_BASE)
    grid, header_codec, sync, preamble = ofdm.grid, ofdm.header_codec, ofdm.sync, ofdm._preamble_time
    ofdm.reset_stream()
    ofdm._stream_buffer = np.zeros((1, 123), dtype="complex64")  # pretend a partial frame is buffered

    ofdm.reconfigure_tx_scheme(modem="qam16", fec="rs_m8")

    assert ofdm.grid is grid
    assert ofdm.header_codec is header_codec
    assert ofdm.sync is sync
    assert ofdm._preamble_time is preamble
    assert ofdm._stream_buffer.shape == (1, 123)  # the whole point: RX streaming state survives a TX-side switch


def test_none_arguments_leave_that_piece_unchanged():
    ofdm = Ofdm(modem="qpsk", fec="conv_v27", fec1="none", **_BASE)
    ofdm.reconfigure_tx_scheme(fec="rs_m8")  # modem/fec1 omitted
    assert ofdm.modem.scheme == "qpsk"
    assert ofdm.fec == "rs_m8"
    assert ofdm.fec1 == "none"


def test_no_arguments_is_a_true_no_op():
    ofdm = Ofdm(modem="qpsk", fec="conv_v27", **_BASE)
    packetizer_before = ofdm.packetizer
    bps_before = ofdm.bits_per_ofdm_symbol
    result = ofdm.reconfigure_tx_scheme()
    assert result == bps_before
    assert ofdm.packetizer is packetizer_before  # not even rebuilt when nothing changed


@pytest.mark.parametrize("kwargs", [{"fec": "not_a_real_scheme"}, {"fec1": "not_a_real_scheme"}])
def test_invalid_fec_name_raises(kwargs):
    ofdm = Ofdm(modem="qpsk", fec="conv_v27", **_BASE)
    with pytest.raises(ValueError):
        ofdm.reconfigure_tx_scheme(**kwargs)


def test_interleaver_requires_fec1_surfaces_through_reconfigure_too():
    """See examples/drone_tui/adaptive_mcs.py's module docstring -- the
    exact hazard that pinned its MCS table to modem-only."""
    ofdm = Ofdm(
        modem="qpsk", fec="rs_m8", fec1="conv_v27",
        interleaver="block", interleaver_kwargs={"unit_bits": 8}, **_BASE
    )
    with pytest.raises(ValueError, match="interleaver"):
        ofdm.reconfigure_tx_scheme(fec1="none")
