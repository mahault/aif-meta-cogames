"""CogsGuard POMDP generative model for pymdp 1.0 (factored).

Defines the A (likelihood), B (transition), C (preference), and D (prior)
matrices for the CogsGuard POMDP using a **factored** state representation,
compatible with ``pymdp.agent.Agent`` from inferactively-pymdp v1.0.

State factors:
    factor 0: phase (6)       — economy-chain phase
    factor 1: hand (4)        — what agent is holding
    factor 2: target_mode (3) — junction contest status
    factor 3: role (4)        — agent specialisation

Observation modalities (6):
    o_resource(3), o_station(4), o_inventory(4),
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
        elif p == Phase.MINE:
            # Miners work near extractors which are near hubs —
            # elevated HUB probability makes MINE observationally
            # distinct from EXPLORE on the station modality.
            a_sta[ObsStation.NONE, p] = 0.55
            a_sta[ObsStation.HUB, p] = 0.30
        else:  # EXPLORE
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
      B_phase:  (6, 6, 4, 13)  P(phase' | phase, hand, action)
      B_hand:   (4, 6, 4, 13)  P(hand'  | phase, hand, action)
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
    """Set phase and hand transition probabilities for action a.

    HOLDING_BOTH (gear + resources simultaneously) transitions:
    - DEPOSIT: removes resources, keeps gear → HOLDING_GEAR
    - CAPTURE: consumes gear, keeps resources → HOLDING_RESOURCE
    - Most other actions: hand stays HOLDING_BOTH
    """
    P = Phase
    H = Hand

    for p in Phase:
        for h in Hand:
            if a == TaskPolicy.NAV_RESOURCE:
                if h == H.EMPTY:
                    B_p[P.MINE, p, h, a] += 0.6
                    B_p[P.EXPLORE, p, h, a] += 0.3
                    B_p[p, p, h, a] += 0.1
                elif h == H.HOLDING_BOTH:
                    # Already has resources — should deposit
                    B_p[p, p, h, a] += 0.7
                    B_p[P.DEPOSIT, p, h, a] += 0.3
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
                elif p == P.MINE and h == H.HOLDING_GEAR:
                    # Mining with gear → picks up resource → HOLDING_BOTH
                    B_p[P.MINE, p, h, a] += 1.0
                    B_h[H.HOLDING_BOTH, p, h, a] += 0.7
                    B_h[H.HOLDING_GEAR, p, h, a] += 0.3
                elif p == P.MINE and h == H.HOLDING_BOTH:
                    # Already full — should deposit
                    B_p[P.MINE, p, h, a] += 0.8
                    B_p[P.DEPOSIT, p, h, a] += 0.2
                    B_h[h, p, h, a] += 1.0
                else:
                    B_p[p, p, h, a] += 0.9
                    B_p[P.EXPLORE, p, h, a] += 0.1
                    B_h[h, p, h, a] += 1.0

            elif a == TaskPolicy.NAV_DEPOT:
                if h == H.HOLDING_RESOURCE:
                    B_p[P.DEPOSIT, p, h, a] += 0.6
                    B_p[p, p, h, a] += 0.4
                elif h == H.HOLDING_BOTH:
                    # Has resources to deposit
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
                elif p == P.DEPOSIT and h == H.HOLDING_BOTH:
                    # Deposit resources, keep gear → HOLDING_GEAR
                    B_p[P.MINE, p, h, a] += 0.5
                    B_p[P.DEPOSIT, p, h, a] += 0.5
                    B_h[H.HOLDING_GEAR, p, h, a] += 0.7
                    B_h[H.HOLDING_BOTH, p, h, a] += 0.3
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
                if h in (H.EMPTY, H.HOLDING_GEAR, H.HOLDING_BOTH):
                    B_p[P.GEAR, p, h, a] += 0.5
                    B_p[p, p, h, a] += 0.4
                    B_p[P.CRAFT, p, h, a] += 0.1
                else:
                    B_p[p, p, h, a] += 0.8
                    B_p[P.GEAR, p, h, a] += 0.2
                B_h[h, p, h, a] += 1.0

            elif a == TaskPolicy.ACQUIRE_GEAR:
                if p == P.GEAR and h == H.HOLDING_RESOURCE:
                    # Acquiring gear when holding resources → HOLDING_BOTH
                    B_p[P.GEAR, p, h, a] += 1.0
                    B_h[H.HOLDING_BOTH, p, h, a] += 0.7
                    B_h[H.HOLDING_RESOURCE, p, h, a] += 0.3
                elif p == P.GEAR:
                    B_p[P.GEAR, p, h, a] += 1.0
                    B_h[H.HOLDING_GEAR, p, h, a] += 0.7
                    B_h[h, p, h, a] += 0.3
                else:
                    B_p[p, p, h, a] += 0.9
                    B_p[P.GEAR, p, h, a] += 0.1
                    B_h[h, p, h, a] += 1.0

            elif a == TaskPolicy.NAV_JUNCTION:
                if h in (H.HOLDING_GEAR, H.HOLDING_BOTH):
                    # Has gear → can capture
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
                elif p == P.CAPTURE and h == H.HOLDING_BOTH:
                    # Capture consumes gear, keeps resources → HOLDING_RESOURCE
                    B_p[P.EXPLORE, p, h, a] += 0.4
                    B_p[P.CAPTURE, p, h, a] += 0.6
                    B_h[H.HOLDING_RESOURCE, p, h, a] += 0.4
                    B_h[H.HOLDING_BOTH, p, h, a] += 0.6
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
      B_phase:  (6, 6, 4, 5)  P(phase' | phase, hand, option)
      B_hand:   (4, 6, 4, 5)  P(hand'  | phase, hand, option)
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
            elif h in (H.HOLDING_RESOURCE, H.HOLDING_BOTH):
                # Has resources (with or without gear) → go deposit
                if p in (P.MINE, P.EXPLORE):
                    B_phase[P.DEPOSIT, p, h, a] += 0.6
                    B_phase[p, p, h, a] += 0.4
                elif p == P.DEPOSIT:
                    B_phase[P.DEPOSIT, p, h, a] += 0.5
                    B_phase[P.MINE, p, h, a] += 0.5
                else:
                    B_phase[P.DEPOSIT, p, h, a] += 0.5
                    B_phase[p, p, h, a] += 0.5
                if h == H.HOLDING_BOTH:
                    # Depositing removes resources, keeps gear → HOLDING_GEAR
                    B_hand[H.HOLDING_BOTH, p, h, a] += 0.4
                    B_hand[H.HOLDING_GEAR, p, h, a] += 0.6
                else:
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
            elif h in (H.HOLDING_GEAR, H.HOLDING_BOTH):
                # Goal achieved — craft cycle complete (has gear)
                B_phase[P.GEAR, p, h, a] += 0.7
                B_phase[p, p, h, a] += 0.3
                B_hand[h, p, h, a] += 1.0
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
            elif h == H.HOLDING_BOTH:
                # Has gear (and resources) — can capture
                if p == P.CAPTURE:
                    B_phase[P.CAPTURE, p, h, a] += 0.6
                    B_phase[P.EXPLORE, p, h, a] += 0.4
                else:
                    B_phase[P.CAPTURE, p, h, a] += 0.6
                    B_phase[p, p, h, a] += 0.4
                # Capture consumes gear, resources stay → HOLDING_RESOURCE
                B_hand[H.HOLDING_BOTH, p, h, a] += 0.6
                B_hand[H.HOLDING_RESOURCE, p, h, a] += 0.4
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
    c_inv = np.array([0.0, 1.0, 2.0, 1.5])      # EMPTY, RESOURCE, GEAR, BOTH
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
    c_sta = np.array([-1.0, 4.0, -1.0, -1.0])   # HUB best, penalize craft/junction
    c_inv = np.array([-2.0, 4.0, -2.0, 1.0])    # penalize EMPTY/GEAR; BOTH = has resources, should deposit
    c_con = np.array([0.5, 0.0, 0.0])            # prefer FREE (avoid combat)
    c_soc = np.array([0.0, 0.5, -0.5, 0.0])     # ALONE, ALLY, ENEMY, BOTH
    c_role = np.array([0.3, 0.0])                # SAME_ROLE, DIFFERENT
    return [c_res, c_sta, c_inv, c_con, c_soc, c_role]


def build_C_aligner() -> list[np.ndarray]:
    """Preferences for aligner role — maximize junction capture.

    Aligners focus on the hub→craft→junction chain: craft gear at stations,
    then navigate to junctions and capture them. Strongly prefers junctions.

    Inventory preference order: BOTH > GEAR > RESOURCE > EMPTY.
    HAS_BOTH is the capture-ready state (gear + hearts) and must be the
    peak preference.  Previous c_inv had HAS_GEAR=5.0 > HAS_BOTH=2.0,
    which made the EFE penalize capture (it consumes gear → leaves BOTH).
    """
    c_res = np.array([-0.5, 0.0, 0.5])          # penalize NONE
    c_sta = np.array([-1.0, 0.0, 3.0, 7.0])     # JUNCTION strongest, penalize NONE
    c_inv = np.array([-1.0, 0.0, 2.0, 5.0])     # BOTH peak (capture-ready), GEAR intermediate
    c_con = np.array([2.5, -1.0, -3.0])          # FREE strong, LOST very bad
    c_soc = np.array([0.0, 1.0, -1.0, 0.0])     # prefer allies, avoid enemies
    c_role = np.array([0.3, 0.0])                # SAME_ROLE, DIFFERENT
    return [c_res, c_sta, c_inv, c_con, c_soc, c_role]


def build_C_scout() -> list[np.ndarray]:
    """Preferences for scout role — near-uniform for epistemic dominance.

    When C is approximately uniform, the pragmatic component of EFE
    (D_KL[q(o|pi) || P(o|C)]) approaches zero.  The agent is then driven
    purely by epistemic value (information gain), naturally exploring
    to reduce uncertainty about the world state.

    Scout: +400 HP, +100 energy in cogames.  Explores and shares
    station discoveries via SharedSpatialMemory (Catal et al. 2024).
    """
    c_res = np.array([0.0, 0.1, 0.1])           # near-uniform: no resource preference
    c_sta = np.array([0.0, 0.1, 0.1, 0.1])      # near-uniform: no station preference
    c_inv = np.array([0.0, 0.0, 0.0, 0.0])      # flat: scout doesn't care about inventory
    c_con = np.array([0.1, 0.0, -0.1])           # slight: prefer FREE (avoid combat)
    c_soc = np.array([0.0, 0.0, 0.0, 0.0])      # flat: no social preference
    c_role = np.array([0.1, 0.0])                # slight: prefer same-role neighbors
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

def _agent_role(agent_id: int, n_agents: int = 8) -> str:
    """Role assignment tuned per team size.

    n=8: 4 miners (even), 3 aligners (odd<7), 1 scout (agent 7).
    n=4: 1 miner, 2 aligners, 1 scout — maximise junction output.
    n<=3: even=miner, odd=aligner (no scout).

    Last agent is the dedicated epistemic scout (flat C-vector →
    information gain dominates EFE → natural exploration).
    """
    if n_agents >= 4 and agent_id == n_agents - 1:
        return "scout"
    if n_agents == 4:
        # agent 0=miner, 1=aligner, 2=aligner, 3=scout
        return "aligner" if agent_id in (1, 2) else "miner"
    return "miner" if agent_id % 2 == 0 else "aligner"


class CogsGuardPOMDP:
    """CogsGuard POMDP generative model (factored).

    Uses a factored state representation: [phase(6), hand(4), target_mode(3), role(4)]
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
            "miner" (extractor→hub loop), "aligner" (craft→junction capture),
            or "scout" (epistemic explorer with flat C).
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
        elif role == "scout":
            D = build_D()
            # Scout: role prior peaked on SUPPORT (epistemic explorer)
            D[3] = np.full(NUM_ROLES, 0.02)
            D[3][Role.SUPPORT] = 0.94
            D[3] /= D[3].sum()
            return cls(C=build_C_scout(), D=D)
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

        Role assignment: 4 miners (even), 3 aligners (odd<last), 1 scout (last).
        A and B matrices are shared (same model structure).
        C and D are replaced after construction with per-batch values
        using ``eqx.tree_at`` (the Agent constructor always broadcasts
        to batch_size, so we override afterward).
        """
        import equinox as eqx
        import jax.numpy as jnp

        role_pomdps = {
            "miner": CogsGuardPOMDP.for_role("miner"),
            "aligner": CogsGuardPOMDP.for_role("aligner"),
            "scout": CogsGuardPOMDP.for_role("scout"),
        }

        # Create agent with uniform C/D (constructor adds batch dim)
        base = CogsGuardPOMDP()
        agent = base.create_agent(batch_size=n_agents, **kwargs)

        # Build per-batch C: (n_agents, n_obs_m)
        C_batched = []
        for m in range(len(NUM_OBS)):
            per_agent = []
            for i in range(n_agents):
                role = _agent_role(i, n_agents)
                per_agent.append(role_pomdps[role].C[m])
            C_batched.append(jnp.array(np.stack(per_agent)))

        # Build per-batch D: (n_agents, n_states_f)
        D_batched = []
        for f in range(len(NUM_STATE_FACTORS)):
            per_agent = []
            for i in range(n_agents):
                role = _agent_role(i, n_agents)
                per_agent.append(role_pomdps[role].D[f])
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

        Role assignment: 4 miners (even), 3 aligners (odd<last), 1 scout (last).
        Per-role C/D/E via eqx.tree_at.
        """
        import equinox as eqx
        import itertools
        import jax.numpy as jnp
        from pymdp.agent import Agent

        role_pomdps = {
            "miner": CogsGuardPOMDP.for_role("miner"),
            "aligner": CogsGuardPOMDP.for_role("aligner"),
            "scout": CogsGuardPOMDP.for_role("scout"),
        }
        # Optional: use custom (learned) A matrices instead of hand-crafted
        custom_A = kwargs.pop("custom_A", None)
        base = CogsGuardPOMDP(A=custom_A) if custom_A is not None else CogsGuardPOMDP()

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

        # A-learning setup (online Dirichlet updates)
        learn_A = kwargs.pop("learn_A", False)
        pA = None
        if learn_A:
            pA_scale = kwargs.pop("pA_scale", 5.0)
            pA = [jnp.array(a * pA_scale + 0.1) for a in base.A]

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
            "use_param_info_gain": learn_B or learn_A,
            "learn_B": learn_B,
            "learn_A": learn_A,
            "action_selection": "deterministic",
            "gamma": 8.0,
        }
        if pB is not None:
            defaults["pB"] = pB
        if pA is not None:
            defaults["pA"] = pA
        defaults.update(kwargs)
        agent = Agent(A=A, B=B_option, C=C, D=D,
                      batch_size=n_agents, **defaults)

        # Per-batch C/D (miner / aligner / scout)
        C_batched = []
        for m in range(len(NUM_OBS)):
            per_agent = []
            for i in range(n_agents):
                role = _agent_role(i, n_agents)
                per_agent.append(role_pomdps[role].C[m])
            C_batched.append(jnp.array(np.stack(per_agent)))

        D_batched = []
        for f in range(len(NUM_STATE_FACTORS)):
            per_agent = []
            for i in range(n_agents):
                role = _agent_role(i, n_agents)
                per_agent.append(role_pomdps[role].D[f])
            D_batched.append(jnp.array(np.stack(per_agent)))

        agent = eqx.tree_at(lambda a: a.C, agent, C_batched)
        agent = eqx.tree_at(lambda a: a.D, agent, D_batched)

        # Per-role E-vector (habit prior over option policies).
        # q(π) ∝ σ(-G(π)) · E(π) — biases option selection by role
        # without disabling epistemic drive.
        # 25 policies: [0-4]=MINE first, [5-9]=CRAFT first,
        # [10-14]=CAPTURE first, [15-19]=EXPLORE first, [20-24]=DEFEND first.
        n_policies = policies.shape[0]
        E_miner = np.ones(n_policies)
        E_aligner = np.ones(n_policies)
        E_scout = np.ones(n_policies)

        E_miner[0:5] = 4.0     # MINE first — core role
        E_miner[15:20] = 2.0   # EXPLORE first (find resources)
        E_miner[20:25] = 1.5   # DEFEND — acceptable fallback
        E_miner[5:10] = 0.001  # CRAFT first — blocked (not miner's job)
        E_miner[10:15] = 0.001 # CAPTURE first — blocked (not miner's job)

        E_aligner[5:10] = 4.0    # CRAFT first — core role
        E_aligner[10:15] = 4.0   # CAPTURE first — core role
        E_aligner[20:25] = 2.0   # DEFEND — hold junctions
        E_aligner[15:20] = 1.5   # EXPLORE — find stations
        E_aligner[0:5] = 0.001   # MINE first — blocked (not aligner's job)

        # Scout: epistemic agent — EXPLORE and DEFEND only.
        # Flat C makes epistemic term dominate EFE; E biases toward
        # exploration policies as a precision gate.
        E_scout[15:20] = 4.0     # EXPLORE first — core scout role
        E_scout[20:25] = 2.0     # DEFEND — secondary (hold territory)
        E_scout[0:5] = 0.001     # MINE first — blocked
        E_scout[5:10] = 0.001    # CRAFT first — blocked
        E_scout[10:15] = 0.001   # CAPTURE first — blocked

        E_miner /= E_miner.sum()
        E_aligner /= E_aligner.sum()
        E_scout /= E_scout.sum()

        E_map = {"miner": E_miner, "aligner": E_aligner, "scout": E_scout}
        E_batched = []
        for i in range(n_agents):
            role = _agent_role(i, n_agents)
            E_batched.append(E_map[role])
        E_batched = jnp.array(np.stack(E_batched))

        agent = eqx.tree_at(lambda a: a.E, agent, E_batched)

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


# ============================================================================
# Navigation POMDP (Level 0 — fast timescale)
# ============================================================================

# Navigation dependencies (each factor depends on both factors)
NAV_A_DEPENDENCIES = [[1], [0]]        # obs_range ← target_range; obs_movement ← nav_progress
NAV_B_DEPENDENCIES = [[0, 1], [0, 1]]  # cross-coupled


def _build_nav_A() -> list[np.ndarray]:
    """Build observation likelihood matrices for the navigation POMDP.

    A[0]: P(obs_range | target_range) — shape (4, 4)
    A[1]: P(obs_movement | nav_progress) — shape (4, 4)
    """
    from .discretizer import NUM_NAV_PROGRESS, NUM_TARGET_RANGE

    # A[0]: obs_range depends on target_range — mostly identity with noise
    A_range = np.array([
        # target:  ADJ   NEAR   FAR   NOTGT
        [0.85, 0.10, 0.00, 0.00],   # obs: ADJACENT
        [0.10, 0.80, 0.10, 0.05],   # obs: NEAR
        [0.05, 0.10, 0.80, 0.10],   # obs: FAR
        [0.00, 0.00, 0.10, 0.85],   # obs: NO_TARGET
    ], dtype=np.float64)
    # Normalize columns
    A_range /= A_range.sum(axis=0, keepdims=True)

    # A[1]: obs_movement depends on nav_progress — mostly identity
    A_movement = np.array([
        # progress: APPR  LATR  RETR  BLKD
        [0.85, 0.10, 0.05, 0.05],   # obs: APPROACHING
        [0.10, 0.75, 0.15, 0.05],   # obs: LATERAL
        [0.00, 0.10, 0.75, 0.10],   # obs: RETREATING
        [0.05, 0.05, 0.05, 0.80],   # obs: BLOCKED
    ], dtype=np.float64)
    A_movement /= A_movement.sum(axis=0, keepdims=True)

    return [A_range, A_movement]


def _build_nav_B() -> list[np.ndarray]:
    """Build transition matrices for the navigation POMDP.

    B[0]: P(progress' | progress, range, action) — shape (4, 4, 4, 5)
    B[1]: P(range' | progress, range, action) — shape (4, 4, 4, 5)

    Key asymmetry: TOWARD from BLOCKED → likely stays BLOCKED,
    LEFT/RIGHT from BLOCKED → likely APPROACHING (obstacle avoidance).
    """
    from .discretizer import (
        NavAction, NavProgress, TargetRange,
        NUM_NAV_ACTIONS, NUM_NAV_PROGRESS, NUM_TARGET_RANGE,
    )
    APPR, LATR, RETR, BLKD = 0, 1, 2, 3
    ADJ, NEAR, FAR, NOTGT = 0, 1, 2, 3
    n_prog = NUM_NAV_PROGRESS   # 4
    n_range = NUM_TARGET_RANGE  # 4
    n_act = NUM_NAV_ACTIONS     # 5

    # B[0]: progress transitions — shape (prog', prog, range, action)
    B_prog = np.full((n_prog, n_prog, n_range, n_act), 0.1, dtype=np.float64)

    for rng in range(n_range):
        # --- TOWARD (action 0) ---
        # From non-blocked: usually approach
        for p in range(n_prog):
            B_prog[APPR, p, rng, NavAction.TOWARD] = 0.55
            B_prog[LATR, p, rng, NavAction.TOWARD] = 0.15
            B_prog[RETR, p, rng, NavAction.TOWARD] = 0.05
            B_prog[BLKD, p, rng, NavAction.TOWARD] = 0.25
        # From BLOCKED, TOWARD likely stays BLOCKED (wall ahead)
        B_prog[BLKD, BLKD, rng, NavAction.TOWARD] = 0.60
        B_prog[APPR, BLKD, rng, NavAction.TOWARD] = 0.15
        B_prog[LATR, BLKD, rng, NavAction.TOWARD] = 0.15
        B_prog[RETR, BLKD, rng, NavAction.TOWARD] = 0.10

        # --- LEFT (action 1) --- obstacle avoidance
        for p in range(n_prog):
            B_prog[APPR, p, rng, NavAction.LEFT] = 0.30
            B_prog[LATR, p, rng, NavAction.LEFT] = 0.45
            B_prog[RETR, p, rng, NavAction.LEFT] = 0.10
            B_prog[BLKD, p, rng, NavAction.LEFT] = 0.15
        # From BLOCKED, LEFT likely unblocks
        B_prog[APPR, BLKD, rng, NavAction.LEFT] = 0.35
        B_prog[LATR, BLKD, rng, NavAction.LEFT] = 0.35
        B_prog[RETR, BLKD, rng, NavAction.LEFT] = 0.10
        B_prog[BLKD, BLKD, rng, NavAction.LEFT] = 0.20

        # --- RIGHT (action 2) --- symmetric to LEFT
        B_prog[:, :, rng, NavAction.RIGHT] = B_prog[:, :, rng, NavAction.LEFT]

        # --- AWAY (action 3) ---
        for p in range(n_prog):
            B_prog[APPR, p, rng, NavAction.AWAY] = 0.10
            B_prog[LATR, p, rng, NavAction.AWAY] = 0.15
            B_prog[RETR, p, rng, NavAction.AWAY] = 0.55
            B_prog[BLKD, p, rng, NavAction.AWAY] = 0.20
        # From BLOCKED, AWAY likely unblocks (going opposite)
        B_prog[RETR, BLKD, rng, NavAction.AWAY] = 0.40
        B_prog[LATR, BLKD, rng, NavAction.AWAY] = 0.25
        B_prog[BLKD, BLKD, rng, NavAction.AWAY] = 0.20
        B_prog[APPR, BLKD, rng, NavAction.AWAY] = 0.15

        # --- RANDOM (action 4) --- uniform
        B_prog[:, :, rng, NavAction.RANDOM] = 0.25

    # Normalize: sum over first axis (progress') must = 1
    B_prog /= B_prog.sum(axis=0, keepdims=True)

    # B[1]: range transitions — shape (range', prog, range, action)
    B_range = np.full((n_range, n_prog, n_range, n_act), 0.05, dtype=np.float64)

    for act in range(n_act):
        for p in range(n_prog):
            # --- ADJ stays ADJ mostly (target right there) ---
            B_range[ADJ, p, ADJ, act] = 0.70
            B_range[NEAR, p, ADJ, act] = 0.25
            B_range[FAR, p, ADJ, act] = 0.05
            B_range[NOTGT, p, ADJ, act] = 0.00

            # --- NEAR transitions depend on progress ---
            if p == APPR:
                B_range[ADJ, p, NEAR, act] = 0.40
                B_range[NEAR, p, NEAR, act] = 0.45
                B_range[FAR, p, NEAR, act] = 0.10
                B_range[NOTGT, p, NEAR, act] = 0.05
            elif p == RETR:
                B_range[ADJ, p, NEAR, act] = 0.05
                B_range[NEAR, p, NEAR, act] = 0.35
                B_range[FAR, p, NEAR, act] = 0.50
                B_range[NOTGT, p, NEAR, act] = 0.10
            else:  # LATERAL or BLOCKED
                B_range[ADJ, p, NEAR, act] = 0.10
                B_range[NEAR, p, NEAR, act] = 0.60
                B_range[FAR, p, NEAR, act] = 0.20
                B_range[NOTGT, p, NEAR, act] = 0.10

            # --- FAR transitions ---
            if p == APPR:
                B_range[ADJ, p, FAR, act] = 0.05
                B_range[NEAR, p, FAR, act] = 0.35
                B_range[FAR, p, FAR, act] = 0.50
                B_range[NOTGT, p, FAR, act] = 0.10
            elif p == RETR:
                B_range[ADJ, p, FAR, act] = 0.00
                B_range[NEAR, p, FAR, act] = 0.05
                B_range[FAR, p, FAR, act] = 0.55
                B_range[NOTGT, p, FAR, act] = 0.40
            else:  # LATERAL or BLOCKED
                B_range[ADJ, p, FAR, act] = 0.02
                B_range[NEAR, p, FAR, act] = 0.13
                B_range[FAR, p, FAR, act] = 0.60
                B_range[NOTGT, p, FAR, act] = 0.25

            # --- NO_TARGET mostly stays (until something found) ---
            B_range[ADJ, p, NOTGT, act] = 0.02
            B_range[NEAR, p, NOTGT, act] = 0.08
            B_range[FAR, p, NOTGT, act] = 0.15
            B_range[NOTGT, p, NOTGT, act] = 0.75

    # TOWARD action biases range decrease
    for p in range(n_prog):
        B_range[ADJ, p, NEAR, NavAction.TOWARD] += 0.10
        B_range[NEAR, p, FAR, NavAction.TOWARD] += 0.10
    # AWAY biases range increase
    for p in range(n_prog):
        B_range[FAR, p, NEAR, NavAction.AWAY] += 0.10
        B_range[NOTGT, p, FAR, NavAction.AWAY] += 0.05

    # Normalize
    B_range /= B_range.sum(axis=0, keepdims=True)

    return [B_prog, B_range]


def _build_nav_C() -> list[np.ndarray]:
    """Build preference vectors for the navigation POMDP.

    C[0]: preferences over obs_range — prefer ADJACENT
    C[1]: preferences over obs_movement — prefer APPROACHING
    """
    C_range = np.array([3.0, 1.0, -0.5, -1.0])     # ADJACENT > NEAR > FAR > NO_TARGET
    C_movement = np.array([2.0, 0.0, -1.0, -2.0])   # APPROACHING > LATERAL > RETREATING > BLOCKED
    return [C_range, C_movement]


def _build_nav_D() -> list[np.ndarray]:
    """Build initial state priors for the navigation POMDP.

    D[0]: prior over nav_progress — slightly optimistic
    D[1]: prior over target_range — initially FAR or UNKNOWN
    """
    D_progress = np.array([0.35, 0.30, 0.10, 0.25])
    D_progress /= D_progress.sum()
    D_range = np.array([0.05, 0.15, 0.40, 0.40])
    D_range /= D_range.sum()
    return [D_progress, D_range]


def create_nav_agent(n_agents: int = 8, policy_len: int = 2,
                     learn_B: bool = True, pB_scale: float = 5.0):
    """Create the navigation POMDP agent (Level 0).

    Small 16-state POMDP with 5 relative actions, operating at every step.
    Uses epistemic value (use_states_info_gain) for exploration.

    When ``learn_B=True`` (default), the agent learns transition dynamics
    online via Dirichlet updates on pB.  This is the principled AIF fix
    for stuck-in-a-loop behaviour: after ~5 blocked steps with the same
    action, the posterior B shifts enough to make that action unattractive
    and the agent switches to an alternative.  ``use_param_info_gain``
    adds an epistemic bonus for untried actions, driving natural
    exploration of alternatives.

    Parameters
    ----------
    n_agents : int
        Batch size (number of agents).
    policy_len : int
        Planning horizon (2 = two-step planning for obstacle avoidance).
    learn_B : bool
        Enable online B-matrix learning via Dirichlet updates.
    pB_scale : float
        Concentration scale for the Dirichlet prior on B.
        Lower = faster learning (fewer blocked steps to switch).
    """
    import itertools
    import jax.numpy as jnp
    from pymdp.agent import Agent
    from .discretizer import NUM_NAV_ACTIONS

    A = [jnp.array(a) for a in _build_nav_A()]
    B_raw = _build_nav_B()
    B = [jnp.array(b) for b in B_raw]
    C = [jnp.array(c) for c in _build_nav_C()]
    D = [jnp.array(d) for d in _build_nav_D()]

    # Constrained policies: both factors share the same action
    pol_list = [
        [[a] * 2 for a in seq]
        for seq in itertools.product(range(NUM_NAV_ACTIONS), repeat=policy_len)
    ]
    policies = jnp.array(pol_list)

    # Dirichlet priors for B learning
    pB = None
    if learn_B:
        pB = [jnp.array(b * pB_scale + 0.1) for b in B_raw]

    agent = Agent(
        A=A, B=B, C=C, D=D,
        A_dependencies=NAV_A_DEPENDENCIES,
        B_dependencies=NAV_B_DEPENDENCIES,
        num_controls=[NUM_NAV_ACTIONS, NUM_NAV_ACTIONS],
        policies=policies,
        pB=pB,
        learn_B=learn_B,
        use_param_info_gain=learn_B,
        batch_size=n_agents,
        sampling_mode="full",
        policy_len=policy_len,
        inference_algo="fpi",
        num_iter=8,
        use_utility=True,
        use_states_info_gain=True,
        action_selection="deterministic",
        gamma=8.0,
    )

    return agent
