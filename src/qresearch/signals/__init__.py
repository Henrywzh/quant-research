from .momentum import SignalType, SignalConfig, compute_scores
# Import modules for side-effect registration
from . import momentum  # noqa: F401

from .registry import compute_signal, list_signals, get_signal  # convenience exports


__all__ = ["SignalType", "SignalConfig", "compute_scores"]
