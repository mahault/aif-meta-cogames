"""CogsGuard POMDP generative model for pymdp 1.0 (factored).

Defines the A (likelihood), B (transition), C (preference), and D (prior)
matrices for the CogsGuard POMDP using a **factored** state representation,
compatible with ``pymdp.agent.Agent`` from inferactively-pymdp v1.0.

State factors:
    factor 0: phase (6)       — economy-chain phase
    factor 1: hand (3)        — what agent is holding
    factor 2: target_mode (3) — junction contest status
    factor 3: role (4)        — agent specialisation

Observation modalities (6):
    o_resource(3), o_station(4), o_inventory(3),
    o_contest(3), o_social(4), o_role_signal(2)

Actions:
    13 task-level policies (shared across all state factors)

Dependencies:
    A_dependencies: [[0], [0], [1], [2], [3, 2], [3]]
    B_dependencies: [[0, 1], [0, 1], [2], [3]]
"""

from pathlib import Path

import numpy as np

from .discretizer import (
    NUM_HANDS,
    NUM_OBS,
    NUM_OPTIONS,
    NUM_PHASES,
    NUM_ROLES,
    NUM_STATES,
    NUM_TARGET_MODES,
    NUM_TASK_POLICIES,
    Hand,
    MacroOption,
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

# Factor counts
NUM_STATE_FACTORS = [NUM_PHASES, NUM_HANDS, NUM_TARGET_MODES, NUM_ROLES]  # [6, 3, 3, 4]

# Dependency structure
A_DEPENDENCIES = [[0], [0], [1], [2], [3, 2], [3]]
B_DEPENDENCIES = [[0, 1], [0, 1], [2], [3]]


# ---------------------------------------------------------------------------
# A matrices (observation likelihood) — factored
# ---------------------------------------------------------------------------

def build_uniform_A() -> list[np.ndarray]:
    """Uniform observation likelihood (no information), factored."""
    A = []
    for m, n_obs in enumerate(NUM_OBS):
        dep_dims = tuple(NUM_STATE_FACTORS[f] for f in A_DEPENDENCIES[m])
        shape = (n_obs,) + dep_dims
        a = np.ones(shape) / n_obs
        A.append(a)
    return A


def build_default_A() -> list[np.ndarray]:
    """Hand-crafted observation likelihood, factored.

    Each modality depends only on its A_dependency factors.
    """
    A = []

    # -- A[0]: o_resource (3 x 6) — depends on phase --
    a_res = np.full((len(ObsResource), NUM_PHASES), 0.05)
    for p in Phase:
        if p == Phase.MINE:
            a_res[ObsResource.AT, p] = 0.85
            a_res[ObsResource.NEAR, p] = 0.10
        elif p == Phase.EXPLORE:
            a_res[ObsResource.NONE, p] = 0.80
            a_res[ObsResource.NEAR, p] = 0.15
        else:
            a_res[ObsResource.NONE, p] = 0.85
            a_res[ObsResource.NEAR, p] = 0.10
        a_res[:, p] /= a_res[:, p].sum()
    A.append(a_res)

    # -- A[1]: o_station (4 x 6) — depends on phase --
    a_sta = np.full((len(ObsStation), NUM_PHASES), 0.03)
    for p in Phase:
        if p == Phase.DEPOSIT:
            a_sta[ObsStation.HUB, p] = 0.85
        elif p in (Phase.CRAFT, Phase.GEAR):
            a_sta[ObsStation.CRAFT, p] = 0.85
        elif p == Phase.CAPTURE:
            a_sta[ObsStation.JUNCTION, p] = 0.85
        else:
            a_sta[ObsStation.NONE, p] = 0.85
        a_sta[:, p] /= a_sta[:, p].sum()
    A.append(a_sta)

    # -- A[2]: o_inventory (3 x 3) — depends on hand, near-deterministic --
    a_inv = np.full((len(ObsInventory), NUM_HANDS), 0.02)
    for h in Hand:
        a_inv[int(h), h] = 0.96
        a_inv[:, h] /= a_inv[:, h].sum()
    A.append(a_inv)

    # -- A[3]: o_contest (3 x 3) — depends on target_mode, near-deterministic --
    a_con = np.full((len(ObsContest), NUM_TARGET_MODES), 0.05)
    for t in TargetMode:
        a_con[int(t), t] = 0.90
        a_con[:, t] /= a_con[:, t].sum()
    A.append(a_con)

    # -- A[4]: o_social (4 x 4 x 3) — depends on role, target_mode --
    # A_dependencies[4] = [3, 2] → shape (4, 4, 3) = (n_social, n_role, n_target)
    a_soc = np.full((len(ObsSocial), NUM_ROLES, NUM_TARGET_MODES), 0.05)
    for r in Role:
        for t in TargetMode:
            if r in (Role.CAPTURER, Role.SUPPORT):
                a_soc[ObsSocial.ALLY_NEAR, r, t] = 0.5
            else:
                a_soc[ObsSocial.ALONE, r, t] = 0.5
            if t == TargetMode.CONTESTED:
                a_soc[ObsSocial.BOTH_NEAR, r, t] = 0.5
            elif t == TargetMode.LOST:
                a_soc[ObsSocial.ENEMY_NEAR, r, t] = 0.5
            a_soc[:, r, t] /= a_soc[:, r, t].sum()
    A.append(a_soc)

    # -- A[5]: o_role_signal (2 x 4) — depends on role --
    a_role = np.full((len(ObsRoleSignal), NUM_ROLES), 0.1)
    for r in Role:
        if r == Role.SUPPORT:
            a_role[ObsRoleSignal.SAME_ROLE, r] = 0.9
        else:
            a_role[ObsRoleSignal.DIFFERENT_ROLE, r] = 0.9
        a_role[:, r] /= a_role[:, r].sum()
    A.append(a_role)

    return A


# ---------------------------------------------------------------------------
# B matrices (transition model) — factored, action-dependent
# ---------------------------------------------------------------------------

def build_uniform_B() -> list[np.ndarray]:
    """Identity transition matrices (agent stays in current state), factored."""
    B = []
    for f, deps in enumerate(B_DEPENDENCIES):
        dep_dims = tuple(NUM_STATE_FACTORS[d] for d in deps)
        shape = (NUM_STATE_FACTORS[f],) + dep_dims + (NUM_ACTIONS,)
        b = np.zeros(shape)
        # Set identity: for each dep combo and action, self-transition = 1.0
        for a in range(NUM_ACTIONS):
            for idx in np.ndindex(*dep_dims):
                # The factor's own current-state index within deps
                own_pos = deps.index(f)
                own_val = idx[own_pos]
                b[(own_val,) + idx + (a,)] = 1.0
        B.append(b)
    return B


def build_default_B() -> list[np.ndarray]:
    """Hand-crafted factored transition matrices.

    Returns [B_phase, B_hand, B_target, B_role] where:
      B_phase:  (6, 6, 3, 13)  P(phase' | phase, hand, action)
      B_hand:   (3, 6, 3, 13)  P(hand'  | phase, hand, action)
      B_target: (3, 3, 13)     P(target' | target, action)
      B_role:   (4, 4, 13)     P(role'  | role, action) — identity
    """
    n_p, n_h, n_t, n_r = NUM_PHASES, NUM_HANDS, NUM_TARGET_MODES, NUM_ROLES
    n_a = NUM_ACTIONS

    B_phase = np.zeros((n_p, n_p, n_h, n_a))
    B_hand = np.zeros((n_h, n_p, n_h, n_a))
    B_target = np.zeros((n_t, n_t, n_a))
    B_role = np.zeros((n_r, n_r, n_a))

    # Role never changes
    for a in range(n_a):
        B_role[:, :, a] = np.eye(n_r)

    # Fill phase/hand transitions for each action
    for a in TaskPolicy:
        _fill_phase_hand(B_phase, B_hand, int(a))
        _fill_target(B_target, int(a))

    # Normalize columns
    _normalize_B_factor(B_phase, dims=(n_p, n_h), axes=2)
    _normalize_B_factor(B_hand, dims=(n_p, n_h), axes=2)
    for t in range(n_t):
        for a in range(n_a):
            s = B_target[:, t, a].sum()
            if s > 0:
                B_target[:, t, a] /= s
            else:
                B_target[t, t, a] = 1.0

    return [B_phase, B_hand, B_target, B_role]


def _normalize_B_factor(B, dims, axes):
    """Normalize B factor columns (first axis sums to 1)."""
    n_a = B.shape[-1]
    for a in range(n_a):
        for idx in np.ndindex(*dims):
            col = B[(slice(None),) + idx + (a,)]
            s = col.sum()
            if s > 0:
                B[(slice(None),) + idx + (a,)] = col / s
            else:
                # Default: self-transition (if factor's own state is in deps)
                B[(idx[0],) + idx + (a,)] = 1.0


def _fill_phase_hand(B_p, B_h, a):
    """Set phase and hand transition probabilities for action a."""
    P = Phase
    H = Hand

    for p in Phase:
        for h in Hand:
            if a == TaskPolicy.NAV_RESOURCE:
                if h == H.EMPTY:
                    B_p[P.MINE, p, h, a] += 0.6
                    B_p[P.EXPLORE, p, h, a] += 0.3
                    B_p[p, p, h, a] += 0.1
                else:
                    B_p[p, p, h, a] += 0.8
                    B_p[P.EXPLORE, p, h, a] += 0.2
                B_h[h, p, h, a] += 1.0

            elif a == TaskPolicy.MINE:
                if p == P.MINE and h == H.EMPTY:
                    B_p[P.MINE, p, h, a] += 1.0
                    B_h[H.HOLDING_RESOURCE, p, h, a] += 0.7
                    B_h[H.EMPTY, p, h, a] += 0.3
                elif p == P.MINE and h == H.HOLDING_RESOURCE:
                    B_p[P.MINE, p, h, a] += 0.9
                    B_p[P.DEPOSIT, p, h, a] += 0.1
                    B_h[h, p, h, a] += 1.0
                else:
                    B_p[p, p, h, a] += 0.9
                    B_p[P.EXPLORE, p, h, a] += 0.1
                    B_h[h, p, h, a] += 1.0

            elif a == TaskPolicy.NAV_DEPOT:
                if h == H.HOLDING_RESOURCE:
                    B_p[P.DEPOSIT, p, h, a] += 0.6
                    B_p[p, p, h, a] += 0.4
                else:
                    B_p[p, p, h, a] += 0.8
                    B_p[P.DEPOSIT, p, h, a] += 0.2
                B_h[h, p, h, a] += 1.0

            elif a == TaskPolicy.DEPOSIT:
                if p == P.DEPOSIT and h == H.HOLDING_RESOURCE:
                    B_p[P.CRAFT, p, h, a] += 0.5
                    B_p[P.DEPOSIT, p, h, a] += 0.5
                    B_h[H.EMPTY, p, h, a] += 0.7
                    B_h[H.HOLDING_RESOURCE, p, h, a] += 0.3
                else:
                    B_p[p, p, h, a] += 0.9
                    B_p[P.EXPLORE, p, h, a] += 0.1
                    B_h[h, p, h, a] += 1.0

            elif a == TaskPolicy.NAV_CRAFT:
                if h == H.EMPTY:
                    B_p[P.CRAFT, p, h, a] += 0.5
                    B_p[p, p, h, a] += 0.4
                    B_p[P.EXPLORE, p, h, a] += 0.1
                else:
                    B_p[p, p, h, a] += 0.8
                    B_p[P.CRAFT, p, h, a] += 0.2
                B_h[h, p, h, a] += 1.0

            elif a == TaskPolicy.CRAFT:
                if p == P.CRAFT and h == H.EMPTY:
                    B_p[P.CRAFT, p, h, a] += 1.0
                    B_h[H.HOLDING_GEAR, p, h, a] += 0.6
                    B_h[H.EMPTY, p, h, a] += 0.4
                elif p == P.CRAFT and h == H.HOLDING_GEAR:
                    B_p[P.CRAFT, p, h, a] += 0.8
                    B_p[P.GEAR, p, h, a] += 0.2
                    B_h[h, p, h, a] += 1.0
                else:
                    B_p[p, p, h, a] += 0.9
                    B_p[P.EXPLORE, p, h, a] += 0.1
                    B_h[h, p, h, a] += 1.0

            elif a == TaskPolicy.NAV_GEAR:
                if h in (H.EMPTY, H.HOLDING_GEAR):
                    B_p[P.GEAR, p, h, a] += 0.5
                    B_p[p, p, h, a] += 0.4
                    B_p[P.CRAFT, p, h, a] += 0.1
                else:
                    B_p[p, p, h, a] += 0.8
                    B_p[P.GEAR, p, h, a] += 0.2
                B_h[h, p, h, a] += 1.0

            elif a == TaskPolicy.ACQUIRE_GEAR:
                if p == P.GEAR:
                    B_p[P.GEAR, p, h, a] += 1.0
                    B_h[H.HOLDING_GEAR, p, h, a] += 0.7
                    B_h[h, p, h, a] += 0.3
                else:
                    B_p[p, p, h, a] += 0.9
                    B_p[P.GEAR, p, h, a] += 0.1
                    B_h[h, p, h, a] += 1.0

            elif a == TaskPolicy.NAV_JUNCTION:
                if h == H.HOLDING_GEAR:
                    B_p[P.CAPTURE, p, h, a] += 0.6
                    B_p[p, p, h, a] += 0.4
                else:
                    B_p[p, p, h, a] += 0.8
                    B_p[P.CAPTURE, p, h, a] += 0.2
                B_h[h, p, h, a] += 1.0

            elif a == TaskPolicy.CAPTURE:
                if p == P.CAPTURE and h == H.HOLDING_GEAR:
                    B_p[P.EXPLORE, p, h, a] += 0.4
                    B_p[P.CAPTURE, p, h, a] += 0.6
                    B_h[H.EMPTY, p, h, a] += 0.4
                    B_h[H.HOLDING_GEAR, p, h, a] += 0.6
                else:
                    B_p[p, p, h, a] += 0.9
                    B_p[P.EXPLORE, p, h, a] += 0.1
                    B_h[h, p, h, a] += 1.0

            elif a == TaskPolicy.EXPLORE:
                B_p[p, p, h, a] += 0.7
                B_p[P.EXPLORE, p, h, a] += 0.2
                B_p[P.MINE, p, h, a] += 0.1
                B_h[h, p, h, a] += 0.8
                B_h[H.EMPTY, p, h, a] += 0.2

            elif a == TaskPolicy.YIELD:
                B_p[p, p, h, a] += 0.9
                B_p[P.EXPLORE, p, h, a] += 0.1
                B_h[h, p, h, a] += 1.0

            elif a == TaskPolicy.WAIT:
                B_p[p, p, h, a] += 0.95
                B_p[P.EXPLORE, p, h, a] += 0.05
                B_h[h, p, h, a] += 1.0


def _fill_target(B_t, a):
    """Set target_mode transition probabilities for action a.

    Target mode changes only for NAV_JUNCTION and CAPTURE.
    """
    T = TargetMode

    if a == TaskPolicy.NAV_JUNCTION:
        # Approaching junction — might discover contest status
        B_t[T.FREE, T.FREE, a] = 0.8
        B_t[T.CONTESTED, T.FREE, a] = 0.2
        B_t[T.FREE, T.CONTESTED, a] = 0.2
        B_t[T.CONTESTED, T.CONTESTED, a] = 0.7
        B_t[T.LOST, T.CONTESTED, a] = 0.1
        B_t[T.CONTESTED, T.LOST, a] = 0.4
        B_t[T.LOST, T.LOST, a] = 0.6

    elif a == TaskPolicy.CAPTURE:
        # Capture attempt — may improve contest status
        B_t[T.FREE, T.FREE, a] = 1.0
        B_t[T.FREE, T.CONTESTED, a] = 0.4
        B_t[T.CONTESTED, T.CONTESTED, a] = 0.4
        B_t[T.LOST, T.CONTESTED, a] = 0.2
        B_t[T.CONTESTED, T.LOST, a] = 0.3
        B_t[T.LOST, T.LOST, a] = 0.7

    else:
        # All other actions: target_mode unchanged (identity)
        for t in T:
            B_t[t, t, a] = 1.0


# ---------------------------------------------------------------------------
# B matrices — option-level (strategic POMDP)
# ---------------------------------------------------------------------------

def build_option_B() -> list[np.ndarray]:
    """Option-level B matrices for the strategic POMDP (5 macro-options).

    Returns [B_phase, B_hand, B_target, B_role] where:
      B_phase:  (6, 6, 3, 5)  P(phase' | phase, hand, option)
      B_hand:   (3, 6, 3, 5)  P(hand'  | phase, hand, option)
      B_target: (3, 3, 5)     P(target' | target, option)
      B_role:   (4, 4, 5)     P(role'  | role, option) — identity

    Each option column models the dominant single-step transition
    when that option is actively executing.
    """
    n_p, n_h, n_t, n_r = NUM_PHASES, NUM_HANDS, NUM_TARGET_MODES, NUM_ROLES
    n_o = NUM_OPTIONS

    B_phase = np.zeros((n_p, n_p, n_h, n_o))
    B_hand = np.zeros((n_h, n_p, n_h, n_o))
    B_target = np.zeros((n_t, n_t, n_o))
    B_role = np.zeros((n_r, n_r, n_o))

    # Role never changes
    for o in range(n_o):
        B_role[:, :, o] = np.eye(n_r)

    P = Phase
    H = Hand
    T = TargetMode
    O = MacroOption

    # --- MINE_CYCLE (option 0) ---
    # Sequences: NAV_RESOURCE → MINE → NAV_DEPOT → DEPOSIT
    for p in Phase:
        for h in Hand:
            a = O.MINE_CYCLE
            if h == H.EMPTY:
                if p == P.EXPLORE:
                    B_phase[P.MINE, p, h, a] += 0.5
                    B_phase[P.EXPLORE, p, h, a] += 0.5
                elif p == P.MINE:
                    B_phase[P.MINE, p, h, a] += 0.8
                    B_phase[P.DEPOSIT, p, h, a] += 0.2
                elif p == P.DEPOSIT:
                    B_phase[P.EXPLORE, p, h, a] += 0.6
                    B_phase[P.MINE, p, h, a] += 0.3
                    B_phase[P.DEPOSIT, p, h, a] += 0.1
                else:
                    B_phase[P.MINE, p, h, a] += 0.4
                    B_phase[p, p, h, a] += 0.6
                B_hand[H.EMPTY, p, h, a] += 0.6
                B_hand[H.HOLDING_RESOURCE, p, h, a] += 0.4
            elif h == H.HOLDING_RESOURCE:
                if p in (P.MINE, P.EXPLORE):
                    B_phase[P.DEPOSIT, p, h, a] += 0.6
                    B_phase[p, p, h, a] += 0.4
                elif p == P.DEPOSIT:
                    B_phase[P.DEPOSIT, p, h, a] += 0.5
                    B_phase[P.EXPLORE, p, h, a] += 0.5
                else:
                    B_phase[P.DEPOSIT, p, h, a] += 0.5
                    B_phase[p, p, h, a] += 0.5
                B_hand[H.HOLDING_RESOURCE, p, h, a] += 0.6
                B_hand[H.EMPTY, p, h, a] += 0.4
            else:  # HOLDING_GEAR — shouldn't be in mine cycle
                B_phase[p, p, h, a] += 0.9
                B_phase[P.EXPLORE, p, h, a] += 0.1
                B_hand[h, p, h, a] += 1.0

    # --- CRAFT_CYCLE (option 1) ---
    # Sequences: NAV_CRAFT → CRAFT → NAV_GEAR → ACQUIRE_GEAR
    for p in Phase:
        for h in Hand:
            a = O.CRAFT_CYCLE
            if h == H.EMPTY:
                if p in (P.EXPLORE, P.DEPOSIT):
                    B_phase[P.CRAFT, p, h, a] += 0.5
                    B_phase[p, p, h, a] += 0.5
                elif p == P.CRAFT:
                    B_phase[P.CRAFT, p, h, a] += 0.7
                    B_phase[P.GEAR, p, h, a] += 0.3
                elif p == P.GEAR:
                    B_phase[P.GEAR, p, h, a] += 0.8
                    B_phase[P.CRAFT, p, h, a] += 0.2
                else:
                    B_phase[P.CRAFT, p, h, a] += 0.4
                    B_phase[p, p, h, a] += 0.6
                B_hand[H.EMPTY, p, h, a] += 0.5
                B_hand[H.HOLDING_GEAR, p, h, a] += 0.5
            elif h == H.HOLDING_GEAR:
                # Goal achieved — craft cycle complete
                B_phase[P.GEAR, p, h, a] += 0.7
                B_phase[p, p, h, a] += 0.3
                B_hand[H.HOLDING_GEAR, p, h, a] += 1.0
            else:  # HOLDING_RESOURCE — deposit first
                B_phase[P.DEPOSIT, p, h, a] += 0.5
                B_phase[p, p, h, a] += 0.5
                B_hand[h, p, h, a] += 0.7
                B_hand[H.EMPTY, p, h, a] += 0.3

    # --- CAPTURE_CYCLE (option 2) ---
    # Sequences: NAV_JUNCTION → CAPTURE (requires gear)
    for p in Phase:
        for h in Hand:
            a = O.CAPTURE_CYCLE
            if h == H.HOLDING_GEAR:
                if p == P.CAPTURE:
                    B_phase[P.CAPTURE, p, h, a] += 0.6
                    B_phase[P.EXPLORE, p, h, a] += 0.4
                else:
                    B_phase[P.CAPTURE, p, h, a] += 0.6
                    B_phase[p, p, h, a] += 0.4
                B_hand[H.HOLDING_GEAR, p, h, a] += 0.6
                B_hand[H.EMPTY, p, h, a] += 0.4
            else:
                # No gear — can't capture effectively
                B_phase[p, p, h, a] += 0.9
                B_phase[P.EXPLORE, p, h, a] += 0.1
                B_hand[h, p, h, a] += 1.0

    # --- EXPLORE (option 3) ---
    for p in Phase:
        for h in Hand:
            a = O.EXPLORE
            B_phase[P.EXPLORE, p, h, a] += 0.6
            B_phase[P.MINE, p, h, a] += 0.2
            B_phase[p, p, h, a] += 0.2
            B_hand[h, p, h, a] += 0.8
            B_hand[H.EMPTY, p, h, a] += 0.2

    # --- DEFEND (option 4) ---
    for p in Phase:
        for h in Hand:
            a = O.DEFEND
            B_phase[P.CAPTURE, p, h, a] += 0.6
            B_phase[p, p, h, a] += 0.4
            B_hand[h, p, h, a] += 1.0

    # Target mode transitions per option
    for o in range(n_o):
        if o in (O.CAPTURE_CYCLE, O.DEFEND):
            B_target[T.FREE, T.FREE, o] = 0.9
            B_target[T.CONTESTED, T.FREE, o] = 0.1
            B_target[T.FREE, T.CONTESTED, o] = 0.4
            B_target[T.CONTESTED, T.CONTESTED, o] = 0.4
            B_target[T.LOST, T.CONTESTED, o] = 0.2
            B_target[T.CONTESTED, T.LOST, o] = 0.3
            B_target[T.LOST, T.LOST, o] = 0.7
        else:
            for t in T:
                B_target[t, t, o] = 1.0

    # Normalize columns
    _normalize_B_factor(B_phase, dims=(n_p, n_h), axes=2)
    _normalize_B_factor(B_hand, dims=(n_p, n_h), axes=2)
    for t in range(n_t):
        for o in range(n_o):
            s = B_target[:, t, o].sum()
            if s > 0:
                B_target[:, t, o] /= s
            else:
                B_target[t, t, o] = 1.0

    return [B_phase, B_hand, B_target, B_role]


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


def build_C_miner() -> list[np.ndarray]:
    """Preferences for miner role — maximize resource gathering and depositing.

    Miners focus on the extractor→hub loop: find resources, mine them,
    deposit at hub, repeat. Avoids junctions and combat.
    """
    c_res = np.array([-1.0, 1.0, 3.0])          # penalize NONE, AT resource best
    c_sta = np.array([-0.5, 2.5, 0.5, 0.0])     # penalize NONE, HUB best
    c_inv = np.array([-1.0, 2.5, 0.0])           # penalize EMPTY strongly
    c_con = np.array([0.5, 0.0, 0.0])            # prefer FREE (avoid combat)
    c_soc = np.array([0.0, 0.5, -0.5, 0.0])     # ALONE, ALLY, ENEMY, BOTH
    c_role = np.array([0.3, 0.0])                # SAME_ROLE, DIFFERENT
    return [c_res, c_sta, c_inv, c_con, c_soc, c_role]


def build_C_aligner() -> list[np.ndarray]:
    """Preferences for aligner role — maximize junction capture.

    Aligners focus on the hub→craft→junction chain: craft gear at stations,
    then navigate to junctions and capture them. Strongly prefers junctions.
    """
    c_res = np.array([-0.5, 0.0, 0.5])          # penalize NONE
    c_sta = np.array([-0.5, 1.0, 2.5, 5.0])     # penalize NONE, JUNCTION strongest
    c_inv = np.array([-0.5, 0.5, 3.5])           # penalize EMPTY, GEAR best
    c_con = np.array([2.5, -1.0, -3.0])          # FREE strong, LOST very bad
    c_soc = np.array([0.0, 1.0, -1.0, 0.0])     # prefer allies, avoid enemies
    c_role = np.array([0.3, 0.0])                # SAME_ROLE, DIFFERENT
    return [c_res, c_sta, c_inv, c_con, c_soc, c_role]


# ---------------------------------------------------------------------------
# D vectors (initial state prior) — factored
# ---------------------------------------------------------------------------

def build_D() -> list[np.ndarray]:
    """Initial state prior, factored over 4 state factors."""
    # Phase prior: peaked on EXPLORE
    d_phase = np.full(NUM_PHASES, 0.02)
    d_phase[Phase.EXPLORE] = 0.9
    d_phase /= d_phase.sum()

    # Hand prior: peaked on EMPTY
    d_hand = np.full(NUM_HANDS, 0.02)
    d_hand[Hand.EMPTY] = 0.96
    d_hand /= d_hand.sum()

    # Target mode prior: peaked on FREE
    d_target = np.full(NUM_TARGET_MODES, 0.05)
    d_target[TargetMode.FREE] = 0.9
    d_target /= d_target.sum()

    # Role prior: uniform (agent doesn't know its role yet)
    d_role = np.ones(NUM_ROLES) / NUM_ROLES

    return [d_phase, d_hand, d_target, d_role]


# ---------------------------------------------------------------------------
# POMDP wrapper
# ---------------------------------------------------------------------------

class CogsGuardPOMDP:
    """CogsGuard POMDP generative model (factored).

    Uses a factored state representation: [phase(6), hand(3), target_mode(3), role(4)]
    with dependency-aware A and B matrices for pymdp 1.0.

    Parameters
    ----------
    A : list[np.ndarray] | None
        Observation likelihood matrices. Default: hand-crafted.
    B : list[np.ndarray] | None
        Transition matrices (4 factors). Default: hand-crafted.
    C : list[np.ndarray] | None
        Preference vectors. Default: economy-chain preferences.
    D : list[np.ndarray] | None
        Initial state prior (4 factors). Default: peaked on EXPLORE/EMPTY/FREE.
    """

    def __init__(self, A=None, B=None, C=None, D=None):
        self.A = A if A is not None else build_default_A()
        self.B = B if B is not None else build_default_B()
        self.C = C if C is not None else build_C()
        self.D = D if D is not None else build_D()

    @classmethod
    def for_role(cls, role: str) -> "CogsGuardPOMDP":
        """Create a role-specialized POMDP with tuned C preferences.

        Parameters
        ----------
        role : str
            "miner" (extractor→hub loop) or "aligner" (craft→junction capture).
            Any other value returns the default (generalist) model.
        """
        if role == "miner":
            D = build_D()
            # Miner: role prior peaked on GATHERER
            D[3] = np.full(NUM_ROLES, 0.02)
            D[3][Role.GATHERER] = 0.94
            D[3] /= D[3].sum()
            return cls(C=build_C_miner(), D=D)
        elif role == "aligner":
            D = build_D()
            # Aligner: role prior peaked on CAPTURER
            D[3] = np.full(NUM_ROLES, 0.02)
            D[3][Role.CAPTURER] = 0.94
            D[3] /= D[3].sum()
            return cls(C=build_C_aligner(), D=D)
        return cls()

    @classmethod
    def uniform(cls) -> "CogsGuardPOMDP":
        """Model with uniform (uninformative) A and identity B matrices."""
        return cls(A=build_uniform_A(), B=build_uniform_B())

    @classmethod
    def from_fitted(cls, path: str | Path) -> "CogsGuardPOMDP":
        """Load fitted A/B matrices from a ``.npz`` file.

        C and D use defaults (preferences are environment-invariant).
        """
        data = np.load(str(path))
        A = [data[f"A_{i}"] for i in range(len(NUM_OBS))]
        B = [data[f"B_{i}"] for i in range(len(B_DEPENDENCIES))]
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

        Uses factored state representation with constrained policies
        (all 4 factors share the same 13-action control space).
        """
        import jax.numpy as jnp
        from pymdp.agent import Agent

        A = [jnp.array(a) for a in self.A]
        B = [jnp.array(b) for b in self.B]
        C = [jnp.array(c) for c in self.C]
        D = [jnp.array(d) for d in self.D]

        # Constrained policies: all 4 factors take the SAME action.
        import itertools
        policy_len = kwargs.pop("policy_len", 2)
        pol_list = [
            [[a] * 4 for a in seq]
            for seq in itertools.product(range(NUM_ACTIONS), repeat=policy_len)
        ]
        policies = jnp.array(pol_list)

        # Build pB (Dirichlet concentration) for online B-learning
        pB_scale = kwargs.pop("pB_scale", 5.0)
        learn_B = kwargs.pop("learn_B", False)  # pop to avoid double-passing; added to defaults below
        pB = None
        if learn_B:
            pB = [jnp.array(b * pB_scale + 0.1) for b in self.B]

        defaults = {
            "A_dependencies": A_DEPENDENCIES,
            "B_dependencies": B_DEPENDENCIES,
            "num_controls": [NUM_ACTIONS, NUM_ACTIONS, NUM_ACTIONS, NUM_ACTIONS],
            "policies": policies,
            "sampling_mode": "full",
            "policy_len": policy_len,
            "inference_algo": "fpi",
            "num_iter": 16,
            "use_utility": True,
            "use_states_info_gain": True,
            "use_param_info_gain": learn_B,
            "learn_B": learn_B,
            "action_selection": "deterministic",
            "gamma": 8.0,
        }
        if pB is not None:
            defaults["pB"] = pB
        defaults.update(kwargs)
        return Agent(A=A, B=B, C=C, D=D, **defaults)

    @staticmethod
    def create_batched_agent(n_agents: int = 8, **kwargs):
        """Create one ``Agent(batch_size=n_agents)`` with per-role C/D.

        Even-indexed agents are miners, odd-indexed are aligners.
        A and B matrices are shared (same model structure).
        C and D are replaced after construction with per-batch values
        using ``eqx.tree_at`` (the Agent constructor always broadcasts
        to batch_size, so we override afterward).
        """
        import equinox as eqx
        import jax.numpy as jnp

        pomdp_miner = CogsGuardPOMDP.for_role("miner")
        pomdp_aligner = CogsGuardPOMDP.for_role("aligner")

        # Create agent with uniform C/D (constructor adds batch dim)
        base = CogsGuardPOMDP()
        agent = base.create_agent(batch_size=n_agents, **kwargs)

        # Build per-batch C: (n_agents, n_obs_m)
        C_batched = []
        for m in range(len(NUM_OBS)):
            per_agent = []
            for i in range(n_agents):
                c = pomdp_miner.C[m] if i % 2 == 0 else pomdp_aligner.C[m]
                per_agent.append(c)
            C_batched.append(jnp.array(np.stack(per_agent)))

        # Build per-batch D: (n_agents, n_states_f)
        D_batched = []
        for f in range(len(NUM_STATE_FACTORS)):
            per_agent = []
            for i in range(n_agents):
                d = pomdp_miner.D[f] if i % 2 == 0 else pomdp_aligner.D[f]
                per_agent.append(d)
            D_batched.append(jnp.array(np.stack(per_agent)))

        # Replace C and D with per-batch versions
        agent = eqx.tree_at(lambda a: a.C, agent, C_batched)
        agent = eqx.tree_at(lambda a: a.D, agent, D_batched)

        return agent

    @staticmethod
    def create_strategic_agent(n_agents: int = 8, **kwargs):
        """Create a strategic POMDP agent with 5 macro-options.

        Same state factors, A, C, D as the tactical agent.
        Different B matrices (5 options instead of 13 task policies).
        25 two-step policies (5²) instead of 169 (13²).

        Even-indexed agents are miners, odd-indexed are aligners
        (per-role C/D via eqx.tree_at).
        """
        import equinox as eqx
        import itertools
        import jax.numpy as jnp
        from pymdp.agent import Agent

        pomdp_miner = CogsGuardPOMDP.for_role("miner")
        pomdp_aligner = CogsGuardPOMDP.for_role("aligner")
        base = CogsGuardPOMDP()

        A = [jnp.array(a) for a in base.A]
        B_option = [jnp.array(b) for b in build_option_B()]
        C = [jnp.array(c) for c in base.C]
        D = [jnp.array(d) for d in base.D]

        # Constrained policies: all 4 factors share the same option.
        policy_len = kwargs.pop("policy_len", 2)
        pol_list = [
            [[a] * 4 for a in seq]
            for seq in itertools.product(range(NUM_OPTIONS), repeat=policy_len)
        ]
        policies = jnp.array(pol_list)

        # B-learning setup
        pB_scale = kwargs.pop("pB_scale", 5.0)
        learn_B = kwargs.pop("learn_B", False)
        pB = None
        if learn_B:
            pB = [jnp.array(b * pB_scale + 0.1) for b in build_option_B()]

        defaults = {
            "A_dependencies": A_DEPENDENCIES,
            "B_dependencies": B_DEPENDENCIES,
            "num_controls": [NUM_OPTIONS, NUM_OPTIONS, NUM_OPTIONS, NUM_OPTIONS],
            "policies": policies,
            "sampling_mode": "full",
            "policy_len": policy_len,
            "inference_algo": "fpi",
            "num_iter": 16,
            "use_utility": True,
            "use_states_info_gain": True,
            "use_param_info_gain": learn_B,
            "learn_B": learn_B,
            "action_selection": "deterministic",
            "gamma": 8.0,
        }
        if pB is not None:
            defaults["pB"] = pB
        defaults.update(kwargs)
        agent = Agent(A=A, B=B_option, C=C, D=D,
                      batch_size=n_agents, **defaults)

        # Per-batch C/D (miner vs aligner)
        C_batched = []
        for m in range(len(NUM_OBS)):
            per_agent = []
            for i in range(n_agents):
                c = pomdp_miner.C[m] if i % 2 == 0 else pomdp_aligner.C[m]
                per_agent.append(c)
            C_batched.append(jnp.array(np.stack(per_agent)))

        D_batched = []
        for f in range(len(NUM_STATE_FACTORS)):
            per_agent = []
            for i in range(n_agents):
                d = pomdp_miner.D[f] if i % 2 == 0 else pomdp_aligner.D[f]
                per_agent.append(d)
            D_batched.append(jnp.array(np.stack(per_agent)))

        agent = eqx.tree_at(lambda a: a.C, agent, C_batched)
        agent = eqx.tree_at(lambda a: a.D, agent, D_batched)

        return agent

    def summary(self) -> str:
        """Human-readable model summary."""
        obs_names = ["resource", "station", "inventory",
                     "contest", "social", "role_signal"]
        obs_desc = ", ".join(
            f"{n}({s})" for n, s in zip(obs_names, NUM_OBS)
        )
        lines = [
            "CogsGuard POMDP (factored)",
            f"  State factors: {NUM_STATE_FACTORS} "
            f"(phase={NUM_PHASES} x hand={NUM_HANDS}"
            f" x target_mode={NUM_TARGET_MODES} x role={NUM_ROLES})"
            f" = {NUM_STATES} total states",
            f"  Obs:       {len(self.A)} modalities — {obs_desc}",
            f"  Actions:   {NUM_ACTIONS} task-level policies ({', '.join(ACTION_NAMES)})",
            f"  A shapes:  {[a.shape for a in self.A]}",
            f"  B shapes:  {[b.shape for b in self.B]}",
            f"  C shapes:  {[c.shape for c in self.C]}",
            f"  D shapes:  {[d.shape for d in self.D]}",
            f"  A_deps:    {A_DEPENDENCIES}",
            f"  B_deps:    {B_DEPENDENCIES}",
        ]
        return "\n".join(lines)
