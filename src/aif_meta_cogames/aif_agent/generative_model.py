"""CogsGuard POMDP generative model for pymdp.

Defines the A (likelihood), B (transition), C (preference), and D (prior)
matrices for the 18-state CogsGuard POMDP, compatible with ``pymdp.Agent``.

State space:
    18 flat states = phase(6) x hand(3)

Observation modalities:
    o_resource(3), o_station(4), o_inventory(3)

Actions:
    5 — noop, north, south, west, east

The model provides three initialisation modes:
    - **default**: hand-crafted matrices encoding economy-chain structure
    - **uniform**: uninformative matrices (for fitting from data)
    - **from_fitted**: load A/B matrices fitted from trajectory data
"""

from pathlib import Path

import numpy as np

from .discretizer import (
    NUM_HANDS,
    NUM_OBS,
    NUM_PHASES,
    NUM_STATES,
    Hand,
    ObsInventory,
    ObsResource,
    ObsStation,
    Phase,
    state_index,
    state_label,
)

NUM_ACTIONS = 5
ACTION_NAMES = ["noop", "north", "south", "west", "east"]


# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------

def build_uniform_A() -> list[np.ndarray]:
    """Uniform observation likelihood (no information)."""
    return [np.ones((n_obs, NUM_STATES)) / n_obs for n_obs in NUM_OBS]


def build_default_A() -> list[np.ndarray]:
    """Hand-crafted observation likelihood encoding economy-chain structure.

    - Inventory observation is near-deterministic (directly reveals hand).
    - Resource and station observations depend on phase.
    """
    A = []

    # -- A[0]: o_resource (3 x 18) --
    a_res = np.full((len(ObsResource), NUM_STATES), 0.1)
    for p in Phase:
        for h in Hand:
            s = state_index(p, h)
            if p == Phase.MINE:
                a_res[ObsResource.AT, s] = 0.7
                a_res[ObsResource.NEAR, s] = 0.2
            elif p == Phase.EXPLORE:
                a_res[ObsResource.NONE, s] = 0.6
                a_res[ObsResource.NEAR, s] = 0.3
            else:
                a_res[ObsResource.NONE, s] = 0.7
                a_res[ObsResource.NEAR, s] = 0.2
            a_res[:, s] /= a_res[:, s].sum()
    A.append(a_res)

    # -- A[1]: o_station (4 x 18) --
    a_sta = np.full((len(ObsStation), NUM_STATES), 0.05)
    for p in Phase:
        for h in Hand:
            s = state_index(p, h)
            if p == Phase.DEPOSIT:
                a_sta[ObsStation.HUB, s] = 0.6
            elif p in (Phase.CRAFT, Phase.GEAR):
                a_sta[ObsStation.CRAFT, s] = 0.6
            elif p == Phase.CAPTURE:
                a_sta[ObsStation.JUNCTION, s] = 0.6
            else:
                a_sta[ObsStation.NONE, s] = 0.7
            a_sta[:, s] /= a_sta[:, s].sum()
    A.append(a_sta)

    # -- A[2]: o_inventory (3 x 18) --
    # Near-deterministic: hand state directly determines observation.
    a_inv = np.full((len(ObsInventory), NUM_STATES), 0.02)
    for p in Phase:
        for h in Hand:
            s = state_index(p, h)
            a_inv[int(h), s] = 0.96
            a_inv[:, s] /= a_inv[:, s].sum()
    A.append(a_inv)

    return A


def build_uniform_B() -> list[np.ndarray]:
    """Identity transition matrix (agent stays in current state)."""
    B = np.zeros((NUM_STATES, NUM_STATES, NUM_ACTIONS))
    for a in range(NUM_ACTIONS):
        B[:, :, a] = np.eye(NUM_STATES)
    return [B]


def build_default_B() -> list[np.ndarray]:
    """Hand-crafted transition matrices encoding economy-chain logic.

    The economy chain flows:
        EXPLORE -> MINE -> DEPOSIT -> CRAFT -> GEAR -> CAPTURE -> EXPLORE

    Transitions are action-independent at this abstraction level
    (movement actions affect spatial position, not the economy phase).
    The B matrix captures the probability of phase/hand changes given
    that the agent is taking steps in the environment.
    """
    B = np.zeros((NUM_STATES, NUM_STATES, NUM_ACTIONS))

    for a in range(NUM_ACTIONS):
        for p in Phase:
            for h in Hand:
                s = state_index(p, h)
                _set_transitions(B, s, p, h, a)

    return [B]


def _set_transitions(B, s, p, h, a):
    """Set transition probabilities for one (state, action) pair."""
    si = state_index  # shorthand

    if p == Phase.EXPLORE and h == Hand.EMPTY:
        B[si(Phase.EXPLORE, Hand.EMPTY), s, a] = 0.7
        B[si(Phase.MINE, Hand.EMPTY), s, a] = 0.3

    elif p == Phase.MINE and h == Hand.EMPTY:
        B[si(Phase.MINE, Hand.EMPTY), s, a] = 0.4
        B[si(Phase.MINE, Hand.HOLDING_RESOURCE), s, a] = 0.3
        B[si(Phase.DEPOSIT, Hand.HOLDING_RESOURCE), s, a] = 0.3

    elif p == Phase.MINE and h == Hand.HOLDING_RESOURCE:
        B[si(Phase.DEPOSIT, Hand.HOLDING_RESOURCE), s, a] = 0.8
        B[si(Phase.MINE, Hand.HOLDING_RESOURCE), s, a] = 0.2

    elif p == Phase.DEPOSIT and h == Hand.HOLDING_RESOURCE:
        B[si(Phase.DEPOSIT, Hand.HOLDING_RESOURCE), s, a] = 0.5
        B[si(Phase.CRAFT, Hand.EMPTY), s, a] = 0.3
        B[si(Phase.DEPOSIT, Hand.EMPTY), s, a] = 0.2

    elif p == Phase.CRAFT and h == Hand.EMPTY:
        B[si(Phase.CRAFT, Hand.EMPTY), s, a] = 0.4
        B[si(Phase.GEAR, Hand.HOLDING_GEAR), s, a] = 0.3
        B[si(Phase.CRAFT, Hand.HOLDING_GEAR), s, a] = 0.3

    elif p in (Phase.GEAR, Phase.CRAFT) and h == Hand.HOLDING_GEAR:
        B[si(Phase.CAPTURE, Hand.HOLDING_GEAR), s, a] = 0.7
        B[si(Phase.GEAR, Hand.HOLDING_GEAR), s, a] = 0.3

    elif p == Phase.CAPTURE and h == Hand.HOLDING_GEAR:
        B[si(Phase.CAPTURE, Hand.HOLDING_GEAR), s, a] = 0.4
        B[si(Phase.EXPLORE, Hand.EMPTY), s, a] = 0.4
        B[si(Phase.CAPTURE, Hand.EMPTY), s, a] = 0.2

    else:
        # Unlikely states — self-transition with drift toward EXPLORE/EMPTY
        B[s, s, a] = 0.6
        B[si(Phase.EXPLORE, Hand.EMPTY), s, a] = 0.4

    # Normalise column
    col_sum = B[:, s, a].sum()
    if col_sum > 0:
        B[:, s, a] /= col_sum


def build_C() -> list[np.ndarray]:
    """Preference vectors (log-preferences over observations).

    Encodes the economy-chain goal: capture junctions.
    Intermediate preferences guide toward resource gathering and gear.
    """
    c_res = np.array([0.0, 0.5, 1.0])          # NONE, NEAR, AT
    c_sta = np.array([0.0, 0.5, 1.0, 3.0])     # NONE, HUB, CRAFT, JUNCTION
    c_inv = np.array([0.0, 1.0, 2.0])           # EMPTY, RESOURCE, GEAR
    return [c_res, c_sta, c_inv]


def build_D() -> list[np.ndarray]:
    """Initial state prior: start in EXPLORE/EMPTY."""
    D = np.full(NUM_STATES, 0.01)
    D[state_index(Phase.EXPLORE, Hand.EMPTY)] = 0.9
    D /= D.sum()
    return [D]


# ---------------------------------------------------------------------------
# POMDP wrapper
# ---------------------------------------------------------------------------

class CogsGuardPOMDP:
    """CogsGuard POMDP generative model.

    Wraps A/B/C/D matrices and provides utilities for creating pymdp agents,
    loading fitted parameters, and inspecting the model.

    Parameters
    ----------
    A : list[np.ndarray] | None
        Observation likelihood matrices. Default: hand-crafted.
    B : list[np.ndarray] | None
        Transition matrices. Default: hand-crafted.
    C : list[np.ndarray] | None
        Preference vectors. Default: economy-chain preferences.
    D : list[np.ndarray] | None
        Initial state prior. Default: EXPLORE/EMPTY.
    """

    def __init__(self, A=None, B=None, C=None, D=None):
        self.A = A if A is not None else build_default_A()
        self.B = B if B is not None else build_default_B()
        self.C = C if C is not None else build_C()
        self.D = D if D is not None else build_D()

    @classmethod
    def uniform(cls) -> "CogsGuardPOMDP":
        """Model with uniform (uninformative) A and B matrices."""
        return cls(A=build_uniform_A(), B=build_uniform_B())

    @classmethod
    def from_fitted(cls, path: str | Path) -> "CogsGuardPOMDP":
        """Load fitted A/B matrices from a ``.npz`` file.

        C and D use defaults (preferences are environment-invariant).
        """
        data = np.load(str(path))
        A = [data[f"A_{i}"] for i in range(len(NUM_OBS))]
        B = [data["B_0"]]
        return cls(A=A, B=B)

    def save(self, path: str | Path):
        """Save A/B matrices to a ``.npz`` file."""
        arrays = {}
        for i, a in enumerate(self.A):
            arrays[f"A_{i}"] = a
        for i, b in enumerate(self.B):
            arrays[f"B_{i}"] = b
        np.savez_compressed(str(path), **arrays)

    def create_agent(self, **kwargs):
        """Create a ``pymdp.Agent`` from this generative model.

        Additional keyword arguments are forwarded to the Agent constructor
        (e.g. ``policy_len``, ``inference_algo``, ``use_states_info_gain``).
        """
        import pymdp

        defaults = {
            "policy_len": 2,
            "inference_algo": "VANILLA",
            "use_states_info_gain": True,
            "use_param_info_gain": False,
        }
        defaults.update(kwargs)
        return pymdp.Agent(A=self.A, B=self.B, C=self.C, D=self.D, **defaults)

    def summary(self) -> str:
        """Human-readable model summary."""
        obs_desc = ", ".join(
            f"{n}({s})" for n, s in zip(
                ["resource", "station", "inventory"], NUM_OBS
            )
        )
        lines = [
            "CogsGuard POMDP",
            f"  States:    {NUM_STATES} (phase={NUM_PHASES} x hand={NUM_HANDS})",
            f"  Obs:       {len(self.A)} modalities — {obs_desc}",
            f"  Actions:   {NUM_ACTIONS} ({', '.join(ACTION_NAMES)})",
            f"  A shapes:  {[a.shape for a in self.A]}",
            f"  B shapes:  {[b.shape for b in self.B]}",
            f"  C shapes:  {[c.shape for c in self.C]}",
            f"  D shapes:  {[d.shape for d in self.D]}",
        ]
        # State labels
        lines.append("  State labels:")
        for s in range(NUM_STATES):
            lines.append(f"    {s:2d}: {state_label(s)}")
        return "\n".join(lines)
