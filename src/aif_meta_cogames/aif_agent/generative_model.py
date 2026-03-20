"""CogsGuard POMDP generative model for pymdp.

Defines the A (likelihood), B (transition), C (preference), and D (prior)
matrices for the 216-state CogsGuard POMDP, compatible with ``pymdp.Agent``.

State space:
    216 flat states = phase(6) x hand(3) x target_mode(3) x role(4)

Observation modalities:
    o_resource(3), o_station(4), o_inventory(3),
    o_contest(3), o_social(4), o_role_signal(2)

Actions:
    13 task-level policies (not primitive movements!)
    NAV_RESOURCE, MINE, NAV_DEPOT, DEPOSIT, NAV_CRAFT, CRAFT,
    NAV_GEAR, ACQUIRE_GEAR, NAV_JUNCTION, CAPTURE, EXPLORE, YIELD, WAIT

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
    NUM_ROLES,
    NUM_STATES,
    NUM_TARGET_MODES,
    NUM_TASK_POLICIES,
    Hand,
    ObsContest,
    ObsInventory,
    ObsResource,
    ObsRoleSignal,
    ObsSocial,
    ObsStation,
    Phase,
    Role,
    TargetMode,
    TaskPolicy,
    state_index,
    state_label,
    TASK_POLICY_NAMES,
)

NUM_ACTIONS = NUM_TASK_POLICIES  # 13
ACTION_NAMES = TASK_POLICY_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_states():
    """Iterate over all (phase, hand, target_mode, role) combinations."""
    for p in Phase:
        for h in Hand:
            for t in TargetMode:
                for r in Role:
                    yield p, h, t, r, state_index(p, h, t, r)


def _normalize_columns(matrix: np.ndarray, axis: int = 0):
    """Normalize columns (axis=0) of a 2D matrix to sum to 1."""
    col_sums = matrix.sum(axis=axis, keepdims=True)
    col_sums = np.where(col_sums == 0, 1.0, col_sums)
    return matrix / col_sums


# ---------------------------------------------------------------------------
# A matrices (observation likelihood)
# ---------------------------------------------------------------------------

def build_uniform_A() -> list[np.ndarray]:
    """Uniform observation likelihood (no information)."""
    return [np.ones((n_obs, NUM_STATES)) / n_obs for n_obs in NUM_OBS]


def build_default_A() -> list[np.ndarray]:
    """Hand-crafted observation likelihood encoding economy-chain structure.

    6 observation modalities x 216 states.
    Each modality primarily depends on one or two state factors.
    """
    A = []

    # -- A[0]: o_resource (3 x 216) — depends on phase --
    a_res = np.full((len(ObsResource), NUM_STATES), 0.1)
    for p, h, t, r, s in _all_states():
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

    # -- A[1]: o_station (4 x 216) — depends on phase --
    a_sta = np.full((len(ObsStation), NUM_STATES), 0.05)
    for p, h, t, r, s in _all_states():
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

    # -- A[2]: o_inventory (3 x 216) — near-deterministic from hand --
    a_inv = np.full((len(ObsInventory), NUM_STATES), 0.02)
    for p, h, t, r, s in _all_states():
        a_inv[int(h), s] = 0.96
        a_inv[:, s] /= a_inv[:, s].sum()
    A.append(a_inv)

    # -- A[3]: o_contest (3 x 216) — near-deterministic from target_mode --
    a_con = np.full((len(ObsContest), NUM_STATES), 0.05)
    for p, h, t, r, s in _all_states():
        a_con[int(t), s] = 0.90
        a_con[:, s] /= a_con[:, s].sum()
    A.append(a_con)

    # -- A[4]: o_social (4 x 216) — weakly informative --
    a_soc = np.full((len(ObsSocial), NUM_STATES), 0.15)
    for p, h, t, r, s in _all_states():
        # Slight bias: capturer/support more likely near allies
        if r in (Role.CAPTURER, Role.SUPPORT):
            a_soc[ObsSocial.ALLY_NEAR, s] = 0.3
        else:
            a_soc[ObsSocial.ALONE, s] = 0.3
        # Contested junctions → enemies likely
        if t == TargetMode.CONTESTED:
            a_soc[ObsSocial.BOTH_NEAR, s] = 0.3
        elif t == TargetMode.LOST:
            a_soc[ObsSocial.ENEMY_NEAR, s] = 0.3
        a_soc[:, s] /= a_soc[:, s].sum()
    A.append(a_soc)

    # -- A[5]: o_role_signal (2 x 216) — depends on role --
    a_role = np.full((len(ObsRoleSignal), NUM_STATES), 0.3)
    for p, h, t, r, s in _all_states():
        # Support role more likely to see same-role allies (flexible)
        if r == Role.SUPPORT:
            a_role[ObsRoleSignal.SAME_ROLE, s] = 0.6
        else:
            a_role[ObsRoleSignal.DIFFERENT_ROLE, s] = 0.6
        a_role[:, s] /= a_role[:, s].sum()
    A.append(a_role)

    return A


# ---------------------------------------------------------------------------
# B matrices (transition model) — ACTION-DEPENDENT
# ---------------------------------------------------------------------------

def build_uniform_B() -> list[np.ndarray]:
    """Identity transition matrix (agent stays in current state)."""
    B = np.zeros((NUM_STATES, NUM_STATES, NUM_ACTIONS))
    for a in range(NUM_ACTIONS):
        B[:, :, a] = np.eye(NUM_STATES)
    return [B]


def build_default_B() -> list[np.ndarray]:
    """Hand-crafted transition matrices with action-dependent dynamics.

    Each of the 13 task-level policies produces distinct state transitions.
    This is the critical fix: primitive movement actions produce action-
    independent B (same transitions for all actions), but task-level
    policies produce action-dependent B (different transitions per task).

    The B matrix is block-sparse: most task policies only affect phase
    and hand, while role self-transitions and target_mode changes are
    task-specific.
    """
    B = np.zeros((NUM_STATES, NUM_STATES, NUM_ACTIONS))

    for p, h, t, r, s in _all_states():
        for a in TaskPolicy:
            _set_task_transitions(B, s, p, h, t, r, int(a))

    return [B]


def _set_task_transitions(B, s, p, h, t, r, a):
    """Set transition probabilities for one (state, task_policy) pair.

    Convention: role never changes within an episode (self-transition).
    Target_mode can change based on capture/contest outcomes.
    Phase and hand change based on economy-chain task policies.
    """
    si = state_index

    if a == TaskPolicy.NAV_RESOURCE:
        # Navigate toward extractor → transition toward MINE phase
        if h == Hand.EMPTY:
            B[si(Phase.MINE, Hand.EMPTY, t, r), s, a] = 0.6
            B[si(Phase.EXPLORE, Hand.EMPTY, t, r), s, a] = 0.3
            B[s, s, a] = 0.1
        else:
            # Can't mine if holding something — mostly self-transition
            B[s, s, a] = 0.8
            B[si(Phase.EXPLORE, h, t, r), s, a] = 0.2

    elif a == TaskPolicy.MINE:
        # Extract resources → hand changes to HOLDING_RESOURCE
        if p == Phase.MINE and h == Hand.EMPTY:
            B[si(Phase.MINE, Hand.HOLDING_RESOURCE, t, r), s, a] = 0.7
            B[si(Phase.MINE, Hand.EMPTY, t, r), s, a] = 0.3
        elif p == Phase.MINE and h == Hand.HOLDING_RESOURCE:
            # Already holding — stay
            B[s, s, a] = 0.9
            B[si(Phase.DEPOSIT, Hand.HOLDING_RESOURCE, t, r), s, a] = 0.1
        else:
            # Not at mine — mostly self-transition
            B[s, s, a] = 0.9
            B[si(Phase.EXPLORE, h, t, r), s, a] = 0.1

    elif a == TaskPolicy.NAV_DEPOT:
        # Navigate toward hub → transition to DEPOSIT phase
        if h == Hand.HOLDING_RESOURCE:
            B[si(Phase.DEPOSIT, Hand.HOLDING_RESOURCE, t, r), s, a] = 0.6
            B[s, s, a] = 0.3
            B[si(p, Hand.HOLDING_RESOURCE, t, r), s, a] = 0.1
        else:
            B[s, s, a] = 0.8
            B[si(Phase.DEPOSIT, h, t, r), s, a] = 0.2

    elif a == TaskPolicy.DEPOSIT:
        # Deposit resources at hub → hand becomes EMPTY
        if p == Phase.DEPOSIT and h == Hand.HOLDING_RESOURCE:
            B[si(Phase.CRAFT, Hand.EMPTY, t, r), s, a] = 0.5
            B[si(Phase.DEPOSIT, Hand.EMPTY, t, r), s, a] = 0.2
            B[si(Phase.DEPOSIT, Hand.HOLDING_RESOURCE, t, r), s, a] = 0.3
        else:
            B[s, s, a] = 0.9
            B[si(Phase.EXPLORE, h, t, r), s, a] = 0.1

    elif a == TaskPolicy.NAV_CRAFT:
        # Navigate toward craft station → transition to CRAFT phase
        if h == Hand.EMPTY:
            B[si(Phase.CRAFT, Hand.EMPTY, t, r), s, a] = 0.5
            B[s, s, a] = 0.4
            B[si(Phase.EXPLORE, Hand.EMPTY, t, r), s, a] = 0.1
        else:
            B[s, s, a] = 0.8
            B[si(Phase.CRAFT, h, t, r), s, a] = 0.2

    elif a == TaskPolicy.CRAFT:
        # Craft gear → hand changes to HOLDING_GEAR
        if p == Phase.CRAFT and h == Hand.EMPTY:
            B[si(Phase.CRAFT, Hand.HOLDING_GEAR, t, r), s, a] = 0.6
            B[si(Phase.CRAFT, Hand.EMPTY, t, r), s, a] = 0.4
        elif p == Phase.CRAFT and h == Hand.HOLDING_GEAR:
            B[s, s, a] = 0.8
            B[si(Phase.GEAR, Hand.HOLDING_GEAR, t, r), s, a] = 0.2
        else:
            B[s, s, a] = 0.9
            B[si(Phase.EXPLORE, h, t, r), s, a] = 0.1

    elif a == TaskPolicy.NAV_GEAR:
        # Navigate to gear pickup → transition to GEAR phase
        if h in (Hand.EMPTY, Hand.HOLDING_GEAR):
            B[si(Phase.GEAR, Hand.HOLDING_GEAR, t, r), s, a] = 0.5
            B[s, s, a] = 0.4
            B[si(Phase.CRAFT, h, t, r), s, a] = 0.1
        else:
            B[s, s, a] = 0.8
            B[si(Phase.GEAR, h, t, r), s, a] = 0.2

    elif a == TaskPolicy.ACQUIRE_GEAR:
        # Pick up gear → hand becomes HOLDING_GEAR
        if p == Phase.GEAR:
            B[si(Phase.GEAR, Hand.HOLDING_GEAR, t, r), s, a] = 0.7
            B[si(Phase.GEAR, h, t, r), s, a] = 0.3
        else:
            B[s, s, a] = 0.9
            B[si(Phase.GEAR, h, t, r), s, a] = 0.1

    elif a == TaskPolicy.NAV_JUNCTION:
        # Navigate toward junction → transition to CAPTURE phase
        # Target_mode may change (approach contested junction)
        if h == Hand.HOLDING_GEAR:
            if t == TargetMode.FREE:
                B[si(Phase.CAPTURE, Hand.HOLDING_GEAR, TargetMode.FREE, r), s, a] = 0.5
                B[si(Phase.CAPTURE, Hand.HOLDING_GEAR, TargetMode.CONTESTED, r), s, a] = 0.1
            elif t == TargetMode.CONTESTED:
                B[si(Phase.CAPTURE, Hand.HOLDING_GEAR, TargetMode.CONTESTED, r), s, a] = 0.5
                B[si(Phase.CAPTURE, Hand.HOLDING_GEAR, TargetMode.FREE, r), s, a] = 0.1
            else:  # LOST
                B[si(Phase.CAPTURE, Hand.HOLDING_GEAR, TargetMode.CONTESTED, r), s, a] = 0.4
                B[si(Phase.CAPTURE, Hand.HOLDING_GEAR, TargetMode.LOST, r), s, a] = 0.2
            B[s, s, a] = 0.4
        else:
            B[s, s, a] = 0.8
            B[si(Phase.CAPTURE, h, t, r), s, a] = 0.2

    elif a == TaskPolicy.CAPTURE:
        # Capture junction → cycle resets to EXPLORE/EMPTY
        # Target_mode may improve (CONTESTED→FREE, LOST→CONTESTED)
        if p == Phase.CAPTURE and h == Hand.HOLDING_GEAR:
            if t == TargetMode.FREE:
                B[si(Phase.EXPLORE, Hand.EMPTY, TargetMode.FREE, r), s, a] = 0.5
                B[si(Phase.CAPTURE, Hand.HOLDING_GEAR, TargetMode.FREE, r), s, a] = 0.5
            elif t == TargetMode.CONTESTED:
                B[si(Phase.EXPLORE, Hand.EMPTY, TargetMode.FREE, r), s, a] = 0.3
                B[si(Phase.CAPTURE, Hand.HOLDING_GEAR, TargetMode.CONTESTED, r), s, a] = 0.5
                B[si(Phase.CAPTURE, Hand.HOLDING_GEAR, TargetMode.LOST, r), s, a] = 0.2
            else:  # LOST
                B[si(Phase.CAPTURE, Hand.HOLDING_GEAR, TargetMode.CONTESTED, r), s, a] = 0.4
                B[si(Phase.CAPTURE, Hand.HOLDING_GEAR, TargetMode.LOST, r), s, a] = 0.5
                B[si(Phase.EXPLORE, Hand.EMPTY, TargetMode.LOST, r), s, a] = 0.1
        else:
            B[s, s, a] = 0.9
            B[si(Phase.EXPLORE, h, t, r), s, a] = 0.1

    elif a == TaskPolicy.EXPLORE:
        # Exploration — self-transition with slow drift
        B[s, s, a] = 0.7
        B[si(Phase.EXPLORE, Hand.EMPTY, t, r), s, a] = 0.2
        B[si(Phase.MINE, Hand.EMPTY, t, r), s, a] = 0.1

    elif a == TaskPolicy.YIELD:
        # Give way — self-transition (stay put, let teammate act)
        B[s, s, a] = 0.9
        B[si(Phase.EXPLORE, h, t, r), s, a] = 0.1

    elif a == TaskPolicy.WAIT:
        # Wait — near-complete self-transition
        B[s, s, a] = 0.95
        B[si(Phase.EXPLORE, h, t, r), s, a] = 0.05

    # Normalise column
    col_sum = B[:, s, a].sum()
    if col_sum > 0:
        B[:, s, a] /= col_sum


# ---------------------------------------------------------------------------
# C vectors (preferences)
# ---------------------------------------------------------------------------

def build_C() -> list[np.ndarray]:
    """Preference vectors (log-preferences over observations).

    Encodes the economy-chain goal: capture junctions.
    Intermediate preferences guide toward resource gathering and gear.
    """
    c_res = np.array([0.0, 0.5, 1.0])           # NONE, NEAR, AT
    c_sta = np.array([0.0, 0.5, 1.0, 3.0])      # NONE, HUB, CRAFT, JUNCTION
    c_inv = np.array([0.0, 1.0, 2.0])            # EMPTY, RESOURCE, GEAR
    c_con = np.array([1.0, -0.5, -2.0])          # FREE, CONTESTED, LOST
    c_soc = np.array([0.0, 0.5, -0.5, 0.0])     # ALONE, ALLY, ENEMY, BOTH
    c_role = np.array([0.3, 0.0])                # SAME_ROLE, DIFFERENT
    return [c_res, c_sta, c_inv, c_con, c_soc, c_role]


# ---------------------------------------------------------------------------
# D vector (initial state prior)
# ---------------------------------------------------------------------------

def build_D() -> list[np.ndarray]:
    """Initial state prior: spread across EXPLORE/EMPTY/FREE + all roles."""
    D = np.full(NUM_STATES, 0.001)
    # Equal prior across roles in EXPLORE/EMPTY/FREE
    for r in Role:
        D[state_index(Phase.EXPLORE, Hand.EMPTY, TargetMode.FREE, r)] = 0.2
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
        Initial state prior. Default: EXPLORE/EMPTY/FREE.
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
        """Create a ``pymdp.agent.Agent`` (JAX) from this generative model.

        Uses the JAX-based Agent from inferactively-pymdp v1.0.
        Numpy A/B/C/D are auto-converted to JAX arrays by the constructor.

        Additional keyword arguments are forwarded to the Agent constructor
        (e.g. ``policy_len``, ``inference_algo``, ``use_states_info_gain``).
        """
        from pymdp.agent import Agent

        defaults = {
            "policy_len": 2,
            "inference_algo": "fpi",
            "use_states_info_gain": True,
            "use_param_info_gain": False,
            "action_selection": "deterministic",
        }
        defaults.update(kwargs)
        return Agent(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            **defaults,
        )

    def summary(self) -> str:
        """Human-readable model summary."""
        obs_names = ["resource", "station", "inventory",
                     "contest", "social", "role_signal"]
        obs_desc = ", ".join(
            f"{n}({s})" for n, s in zip(obs_names, NUM_OBS)
        )
        lines = [
            "CogsGuard POMDP",
            f"  States:    {NUM_STATES} (phase={NUM_PHASES} x hand={NUM_HANDS}"
            f" x target_mode={NUM_TARGET_MODES} x role={NUM_ROLES})",
            f"  Obs:       {len(self.A)} modalities — {obs_desc}",
            f"  Actions:   {NUM_ACTIONS} task-level policies ({', '.join(ACTION_NAMES)})",
            f"  A shapes:  {[a.shape for a in self.A]}",
            f"  B shapes:  {[b.shape for b in self.B]}",
            f"  C shapes:  {[c.shape for c in self.C]}",
            f"  D shapes:  {[d.shape for d in self.D]}",
        ]
        # Sample state labels
        lines.append("  State labels (first 12):")
        for s in range(min(12, NUM_STATES)):
            lines.append(f"    {s:3d}: {state_label(s)}")
        lines.append(f"    ...")
        return "\n".join(lines)
