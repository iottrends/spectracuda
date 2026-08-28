"""Mac.set_tx_scheme(): the Mac-layer half of adaptive MCS -- calls
Ofdm.reconfigure_tx_scheme() then keeps max_segment_bits (both this
Mac's own record AND whatever the active mode entity's segmentation
actually consults -- TmEntity/UmEntity/AmEntity each keep their OWN copy
passed at construction, see mac.py's own comment on this) in step with
the new PHY capacity. See test_ofdm_reconfigure_tx_scheme.py for the
lower-level Ofdm behavior this builds on.
"""
import numpy as np
import pytest

from spectracuda.mac import Mac

_PHY_KWARGS = dict(
    fft_size=256, n_pilot=8, n_data=216, cp_len=32,
    modem="qpsk", fec="conv_v27", crc="crc16",
    sync="schmidl_cox", cfo="schmidl_cox", channel_estimator="ls", equalizer="mmse",
    backend="numpy",
)


def _bind(a: Mac, b: Mac) -> None:
    req = a.build_bind_request()
    resp = b.handle_bind_request_iq(req)
    assert resp is not None
    assert a.handle_bind_response_iq(resp)
    assert a.bound and b.bound


@pytest.mark.parametrize("mode", ["um", "am"])
def test_set_tx_scheme_updates_max_segment_bits_and_segmenter(mode):
    kwargs = {"window_size": 8} if mode in ("um", "am") else {}
    tx = Mac(mode=mode, ofdm_kwargs=_PHY_KWARGS, **kwargs)
    before = tx.max_segment_bits

    new_cap = tx.set_tx_scheme(modem="qam64")
    assert new_cap > before  # qam64 packs strictly more bits/OFDM-symbol than qpsk
    assert tx.max_segment_bits == new_cap
    segmenter = tx._impl._segmenter if mode == "um" else tx._impl._um._segmenter
    assert segmenter.max_segment_bits == new_cap  # not just Mac's own bookkeeping -- the actual consulted value


def test_set_tx_scheme_tm_mode_updates_directly():
    tm = Mac(mode="tm", ofdm_kwargs=_PHY_KWARGS)
    before = tm.max_segment_bits
    new_cap = tm.set_tx_scheme(modem="qam16")
    assert new_cap > before
    assert tm._impl.max_segment_bits == new_cap


def test_segmentation_actually_changes_on_the_next_send_after_a_switch():
    tx = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    rx = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    _bind(tx, rx)

    sdu = np.random.default_rng(2).integers(0, 2, size=40000).astype("uint8")
    n_pdus_before = len(tx.send_iq(sdu))

    tx.set_tx_scheme(modem="qam64")
    n_pdus_after = len(tx.send_iq(sdu))
    assert n_pdus_after < n_pdus_before  # same SDU, strictly fewer PDUs at the higher-order modem


def test_bit_exact_delivery_across_a_live_scheme_switch():
    """The end-to-end proof: switch scheme, THEN send, and confirm the
    peer -- constructed with the ORIGINAL scheme and never itself
    reconfigured -- still reassembles the SDU bit-exact. This is
    set_tx_scheme()'s whole reason to exist: rx_process() resolves
    mod_scheme/fec from each frame's own decoded header, so the peer
    needs no matching action at all."""
    tx = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    rx = Mac(mode="um", ofdm_kwargs=_PHY_KWARGS)
    _bind(tx, rx)

    tx.set_tx_scheme(modem="qam64")
    sdu = np.random.default_rng(3).integers(0, 2, size=50000).astype("uint8")
    pdus = tx.send_iq(sdu)

    delivered = []
    for pdu_iq in pdus:
        for out in rx.receive_iq(pdu_iq):
            delivered.append(out)
    assert len(delivered) == 1
    assert np.array_equal(delivered[0], sdu)


def test_set_tx_scheme_without_ofdm_raises():
    mac = Mac(mode="um", max_segment_bits=1000)
    with pytest.raises(ValueError, match="ofdm_kwargs"):
        mac.set_tx_scheme(modem="qam16")
