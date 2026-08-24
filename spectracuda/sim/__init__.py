"""Simulation/test utilities -- not part of the real tx/rx PHY chain
(Layers 1-3). Currently: Channel, an impairment simulator mirroring
liquid-dsp's channel_cccf.
"""
from .channel import Channel

__all__ = ["Channel"]
