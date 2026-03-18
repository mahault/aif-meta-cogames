"""Discrete active inference agent for CogsGuard.

Provides a POMDP generative model, observation discretizer, and tools
for fitting model parameters from trajectory data.
"""

from .discretizer import (
    Hand,
    ObservationDiscretizer,
    ObsInventory,
    ObsResource,
    ObsStation,
    Phase,
    NUM_HANDS,
    NUM_OBS,
    NUM_PHASES,
    NUM_STATES,
    state_factors,
    state_index,
    state_label,
)
from .generative_model import CogsGuardPOMDP, NUM_ACTIONS

__all__ = [
    "CogsGuardPOMDP",
    "Hand",
    "NUM_ACTIONS",
    "NUM_HANDS",
    "NUM_OBS",
    "NUM_PHASES",
    "NUM_STATES",
    "ObservationDiscretizer",
    "ObsInventory",
    "ObsResource",
    "ObsStation",
    "Phase",
    "state_factors",
    "state_index",
    "state_label",
]
