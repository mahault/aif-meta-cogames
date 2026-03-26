"""Observation discretizer: token obs (200, 3) -> discrete POMDP quantities.

Maps raw uint8 token observations from MettaGrid to the discrete state and
observation spaces used by the CogsGuard POMDP generative model.

State space (216 states):
    phase(6) x hand(3) x target_mode(3) x role(4)

Observation modalities (6):
    o_resource(3):     extractor proximity
    o_station(4):      hub / craft-station / junction proximity
    o_inventory(3):    what the agent is holding
    o_contest(3):      junction contest status
    o_social(4):       nearby agent presence
    o_role_signal(2):  teammate role similarity

Action space (13 task-level policies):
    NAV_RESOURCE, MINE, NAV_DEPOT, DEPOSIT, NAV_CRAFT, CRAFT,
    NAV_GEAR, ACQUIRE_GEAR, NAV_JUNCTION, CAPTURE, EXPLORE, YIELD, WAIT

The discretizer serves two roles:
    1. **State inference** (infer_state): reconstructs the hidden state from
       full token observations. Used for supervised fitting of A/B matrices.
    2. **Observation discretisation** (discretize_obs): produces the partial
       observation an active inference agent would receive.
"""

from enum import IntEnum

import numpy as np


# ---------------------------------------------------------------------------
# State enums
# ---------------------------------------------------------------------------

class Phase(IntEnum):
    """Economy-chain phase the agent is currently in."""
    EXPLORE = 0
    MINE = 1
    DEPOSIT = 2
    CRAFT = 3
    GEAR = 4
    CAPTURE = 5


class Hand(IntEnum):
    """What the agent is holding."""
    EMPTY = 0
    HOLDING_RESOURCE = 1
    HOLDING_GEAR = 2


class TargetMode(IntEnum):
    """Junction contest status visible to the agent."""
    FREE = 0        # No enemy presence at nearest junction
    CONTESTED = 1   # Both teams present at junction
    LOST = 2        # Enemy controls nearest junction


class Role(IntEnum):
    """Agent's specialisation role within the team."""
    GATHERER = 0    # Focuses on resource extraction chain
    CRAFTER = 1     # Focuses on crafting gear
    CAPTURER = 2    # Focuses on junction capture
    SUPPORT = 3     # Flexible / supports teammates


# ---------------------------------------------------------------------------
# Task-level policies (POMDP action space)
# ---------------------------------------------------------------------------

class TaskPolicy(IntEnum):
    """Task-level policies — the POMDP action space.

    Each task policy has distinct B matrix transitions, giving pymdp
    meaningful EFE differences for planning.  A navigator converts the
    selected task policy into primitive movement actions.
    """
    NAV_RESOURCE = 0     # Navigate toward nearest resource extractor
    MINE = 1             # Extract resources at current location
    NAV_DEPOT = 2        # Navigate toward deposit hub
    DEPOSIT = 3          # Deposit resources at hub
    NAV_CRAFT = 4        # Navigate toward craft station
    CRAFT = 5            # Craft gear at craft station
    NAV_GEAR = 6         # Navigate toward gear pickup
    ACQUIRE_GEAR = 7     # Pick up crafted gear
    NAV_JUNCTION = 8     # Navigate toward junction
    CAPTURE = 9          # Capture junction
    EXPLORE = 10         # Random exploration
    YIELD = 11           # Give way to teammate
    WAIT = 12            # Wait / noop


class MacroOption(IntEnum):
    """Macro-level options — the strategic POMDP action space.

    Each option is a temporally-extended reactive policy (state machine)
    that sequences multiple TaskPolicy actions toward a subgoal.
    The strategic POMDP selects among these options; option state machines
    in OptionExecutor handle the task-level sequencing.
    """
    MINE_CYCLE = 0       # NAV_RESOURCE → MINE → NAV_DEPOT → DEPOSIT
    CRAFT_CYCLE = 1      # NAV_CRAFT → CRAFT → NAV_GEAR → ACQUIRE_GEAR
    CAPTURE_CYCLE = 2    # NAV_JUNCTION → CAPTURE (requires gear)
    EXPLORE = 3          # Wander until finding resource / station
    DEFEND = 4           # NAV_JUNCTION → CAPTURE (defend territory)


# ---------------------------------------------------------------------------
# Observation enums
# ---------------------------------------------------------------------------

class ObsResource(IntEnum):
    """Nearest resource extractor observation."""
    NONE = 0
    NEAR = 1
    AT = 2


class ObsStation(IntEnum):
    """Nearest station observation."""
    NONE = 0
    HUB = 1
    CRAFT = 2       # craft/gear stations (c:aligner, c:miner, etc.)
    JUNCTION = 3


class ObsInventory(IntEnum):
    """Agent inventory observation."""
    EMPTY = 0
    HAS_RESOURCE = 1
    HAS_GEAR = 2


class ObsContest(IntEnum):
    """Junction contest status observation."""
    FREE = 0
    CONTESTED = 1
    LOST = 2


class ObsSocial(IntEnum):
    """Nearby agent presence observation."""
    ALONE = 0
    ALLY_NEAR = 1
    ENEMY_NEAR = 2
    BOTH_NEAR = 3


class ObsRoleSignal(IntEnum):
    """Teammate role similarity observation (from vibe tokens)."""
    SAME_ROLE = 0
    DIFFERENT_ROLE = 1


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

NUM_PHASES = len(Phase)             # 6
NUM_HANDS = len(Hand)               # 3
NUM_TARGET_MODES = len(TargetMode)  # 3
NUM_ROLES = len(Role)               # 4
NUM_STATES = NUM_PHASES * NUM_HANDS * NUM_TARGET_MODES * NUM_ROLES  # 216

NUM_TASK_POLICIES = len(TaskPolicy)  # 13
NUM_ACTIONS = NUM_TASK_POLICIES      # 13 (alias for generative model compat)

NUM_OPTIONS = len(MacroOption)       # 5
OPTION_NAMES = [o.name for o in MacroOption]

NUM_OBS = [
    len(ObsResource),     # 3
    len(ObsStation),      # 4
    len(ObsInventory),    # 3
    len(ObsContest),      # 3
    len(ObsSocial),       # 4
    len(ObsRoleSignal),   # 2
]
OBS_MODALITY_NAMES = [
    "o_resource", "o_station", "o_inventory",
    "o_contest", "o_social", "o_role_signal",
]

TASK_POLICY_NAMES = [tp.name for tp in TaskPolicy]


# ---------------------------------------------------------------------------
# Feature and tag constants for CogsGuard
# ---------------------------------------------------------------------------

# Inventory feature names indicating held resources
RESOURCE_INVENTORY = frozenset([
    "inv:carbon", "inv:oxygen", "inv:germanium", "inv:silicon",
])

# Inventory feature names indicating held gear
GEAR_INVENTORY = frozenset([
    "inv:aligner", "inv:scrambler", "inv:miner", "inv:scout",
])

# Standard CogsGuard tag-value -> category mapping.
COGSGUARD_TAG_CATEGORIES: dict[int, str] = {
    11: "craft",       # type:c:aligner
    12: "craft",       # type:c:miner
    13: "craft",       # type:c:scout
    14: "craft",       # type:c:scrambler
    15: "extractor",   # type:carbon_extractor
    16: "extractor",   # type:germanium_extractor
    17: "hub",         # type:hub
    18: "junction",    # type:junction
    19: "extractor",   # type:oxygen_extractor
    20: "extractor",   # type:solar_extractor
    21: "extractor",   # type:silicon_extractor
}

# Reserved location values in MettaGrid token observations
LOC_GLOBAL = 254   # global / self token (inventory, episode info)
LOC_EMPTY = 255    # empty padding token


# ---------------------------------------------------------------------------
# State indexing (4 factors)
# ---------------------------------------------------------------------------

def state_index(phase: int, hand: int, target_mode: int = 0, role: int = 0) -> int:
    """Flat state index from (phase, hand, target_mode, role) factors.

    Index layout: phase * (H * T * R) + hand * (T * R) + target_mode * R + role
    where H=3, T=3, R=4.
    """
    return (
        int(phase) * (NUM_HANDS * NUM_TARGET_MODES * NUM_ROLES)
        + int(hand) * (NUM_TARGET_MODES * NUM_ROLES)
        + int(target_mode) * NUM_ROLES
        + int(role)
    )


def state_factors(flat_idx: int) -> tuple[int, int, int, int]:
    """Recover (phase, hand, target_mode, role) from a flat state index."""
    htr = NUM_HANDS * NUM_TARGET_MODES * NUM_ROLES  # 36
    tr = NUM_TARGET_MODES * NUM_ROLES                # 12
    r = NUM_ROLES                                     # 4

    phase = flat_idx // htr
    remainder = flat_idx % htr
    hand = remainder // tr
    remainder = remainder % tr
    target_mode = remainder // r
    role = remainder % r
    return phase, hand, target_mode, role


def state_label(flat_idx: int) -> str:
    """Human-readable label for a flat state index."""
    p, h, t, r = state_factors(flat_idx)
    return f"{Phase(p).name}/{Hand(h).name}/{TargetMode(t).name}/{Role(r).name}"


# ---------------------------------------------------------------------------
# Task-policy inference (from trajectory state transitions)
# ---------------------------------------------------------------------------

def infer_task_policy(
    phase_t: int,
    hand_t: int,
    phase_next: int,
    hand_next: int,
) -> int:
    """Infer which task-level policy was executing from state transition.

    Maps (phase_t, hand_t) → (phase_next, hand_next) to one of 13 task
    policies.  Used by fit_matrices.py to convert primitive-action
    trajectories into task-level action labels for B matrix fitting.

    This is a heuristic — it infers intent from outcome, not from the
    actual primitive action taken.  The mapping captures the dominant
    economy-chain transitions.
    """
    # Resource acquisition chain
    if phase_next == Phase.MINE and hand_next == Hand.EMPTY:
        if phase_t in (Phase.EXPLORE, Phase.MINE):
            return TaskPolicy.NAV_RESOURCE
    if phase_next == Phase.MINE and hand_next == Hand.HOLDING_RESOURCE:
        return TaskPolicy.MINE

    # Deposit chain
    if phase_next == Phase.DEPOSIT and hand_next == Hand.HOLDING_RESOURCE:
        if phase_t != Phase.DEPOSIT:
            return TaskPolicy.NAV_DEPOT
        return TaskPolicy.DEPOSIT  # still at depot, depositing
    if phase_t == Phase.DEPOSIT and hand_next == Hand.EMPTY:
        return TaskPolicy.DEPOSIT  # successfully deposited

    # Crafting chain
    if phase_next == Phase.CRAFT and hand_next == Hand.EMPTY:
        if phase_t != Phase.CRAFT:
            return TaskPolicy.NAV_CRAFT
    if phase_next == Phase.CRAFT and hand_next == Hand.HOLDING_GEAR:
        return TaskPolicy.CRAFT

    # Gear acquisition
    if phase_next == Phase.GEAR and hand_next == Hand.HOLDING_GEAR:
        if phase_t != Phase.GEAR:
            return TaskPolicy.NAV_GEAR
        return TaskPolicy.ACQUIRE_GEAR
    if phase_t == Phase.GEAR and phase_next == Phase.GEAR:
        return TaskPolicy.ACQUIRE_GEAR

    # Junction capture
    if phase_next == Phase.CAPTURE and hand_next == Hand.HOLDING_GEAR:
        if phase_t != Phase.CAPTURE:
            return TaskPolicy.NAV_JUNCTION
        return TaskPolicy.CAPTURE
    if phase_t == Phase.CAPTURE and phase_next == Phase.EXPLORE:
        return TaskPolicy.CAPTURE  # successful capture, cycle reset

    # Exploration (no phase/hand change, in explore state)
    if phase_t == Phase.EXPLORE and phase_next == Phase.EXPLORE:
        if hand_t == Hand.EMPTY and hand_next == Hand.EMPTY:
            return TaskPolicy.EXPLORE

    # Self-transition (no change) — default to WAIT
    if phase_t == phase_next and hand_t == hand_next:
        return TaskPolicy.WAIT

    # Fallback: classify as EXPLORE
    return TaskPolicy.EXPLORE


# ---------------------------------------------------------------------------
# Discretizer
# ---------------------------------------------------------------------------

class ObservationDiscretizer:
    """Maps token observations (200, 3) uint8 to discrete POMDP quantities.

    Parameters
    ----------
    obs_feature_names : list[str]
        Feature names in order (index = feature_id in token observations).
        Obtained from trajectory metadata: ``metadata["obs_features"]``.
    tag_categories : dict[int, str] | None
        Mapping from tag feature value to category string.
        Categories: ``"extractor"``, ``"hub"``, ``"craft"``, ``"junction"``.
        Defaults to ``COGSGUARD_TAG_CATEGORIES``.
    near_radius : int
        Manhattan distance threshold for NEAR observations (default 3).
    """

    def __init__(
        self,
        obs_feature_names: list[str],
        tag_categories: dict[int, str] | None = None,
        near_radius: int = 3,
    ):
        self._feat_name_to_id = {
            name: i for i, name in enumerate(obs_feature_names)
        }
        self._tag_categories = (
            tag_categories if tag_categories is not None
            else COGSGUARD_TAG_CATEGORIES
        )
        self._near_radius = near_radius

        # Cache feature IDs for inventory items
        self._resource_feat_ids = frozenset(
            self._feat_name_to_id[n]
            for n in RESOURCE_INVENTORY
            if n in self._feat_name_to_id
        )
        self._gear_feat_ids = frozenset(
            self._feat_name_to_id[n]
            for n in GEAR_INVENTORY
            if n in self._feat_name_to_id
        )

        # Tag feature ID
        self._tag_feat_id = self._feat_name_to_id.get("tag")

        # Group feature ID (for contest / social inference)
        self._group_feat_id = self._feat_name_to_id.get("agent:group")

        # Vibe feature ID (for role signal inference)
        self._vibe_feat_id = self._feat_name_to_id.get("vibe")

        # Agent ID feature
        self._agent_id_feat_id = self._feat_name_to_id.get("agent_id")

    # -- Location helpers ---------------------------------------------------

    @staticmethod
    def _loc_to_rowcol(loc: int) -> tuple[int, int]:
        """Decode location byte to (row, col).

        MettaGrid encodes egocentric positions as ``row << 4 | col``
        (4-bit packing, max 16x16 grid).
        """
        return loc >> 4, loc & 0x0F

    @staticmethod
    def _center() -> tuple[int, int]:
        """Center of a 13x13 egocentric grid."""
        return 6, 6

    def _manhattan_from_center(self, loc: int) -> int:
        r, c = self._loc_to_rowcol(loc)
        cr, cc = self._center()
        return abs(r - cr) + abs(c - cc)

    # -- State inference ----------------------------------------------------

    def infer_hand(self, obs: np.ndarray) -> int:
        """Infer Hand state from inventory tokens.

        Checks gear first (higher priority if agent has both gear and
        resource features simultaneously).

        Inventory tokens may use LOC_GLOBAL (254) or the center-cell
        encoding (6<<4|6 = 102) depending on the mettagrid version.
        """
        loc_center = (6 << 4) | 6   # 102
        has_resource = False
        for i in range(obs.shape[0]):
            loc, feat_id, value = int(obs[i, 0]), int(obs[i, 1]), int(obs[i, 2])
            if (loc != LOC_GLOBAL and loc != loc_center) or value == 0:
                continue
            if feat_id in self._gear_feat_ids:
                return Hand.HOLDING_GEAR
            if feat_id in self._resource_feat_ids:
                has_resource = True
        return Hand.HOLDING_RESOURCE if has_resource else Hand.EMPTY

    def infer_phase(self, obs: np.ndarray, hand: int) -> int:
        """Infer Phase from nearby entity tags and hand state.

        Priority: if adjacent (distance <= 1) to a tagged entity, the phase
        matches that entity type. Otherwise, inferred from hand state.
        """
        nearest = self._nearest_tagged(obs)
        cat, dist = nearest if nearest else (None, float("inf"))

        if dist <= 1:
            if cat == "junction":
                return Phase.CAPTURE
            if cat == "craft":
                return Phase.CRAFT if hand == Hand.HOLDING_RESOURCE else Phase.GEAR
            if cat == "hub":
                return Phase.DEPOSIT
            if cat == "extractor":
                return Phase.MINE

        # Not adjacent to a station — infer from hand
        if hand == Hand.HOLDING_RESOURCE:
            return Phase.DEPOSIT
        if hand == Hand.HOLDING_GEAR:
            return Phase.CAPTURE
        return Phase.EXPLORE

    def infer_target_mode(self, obs: np.ndarray) -> int:
        """Infer TargetMode from junction ownership visible in observation.

        Scans for junction tags and checks nearby agent group tokens to
        determine contest status.  Returns FREE if no junction visible.
        """
        if self._tag_feat_id is None:
            return TargetMode.FREE

        junction_loc = None
        best_dist = float("inf")

        # Find nearest junction
        for i in range(obs.shape[0]):
            loc, feat_id, value = int(obs[i, 0]), int(obs[i, 1]), int(obs[i, 2])
            if loc >= LOC_GLOBAL or feat_id != self._tag_feat_id:
                continue
            cat = self._tag_categories.get(value)
            if cat != "junction":
                continue
            dist = self._manhattan_from_center(loc)
            if dist < best_dist:
                best_dist = dist
                junction_loc = loc

        if junction_loc is None:
            return TargetMode.FREE

        # Check for agents near the junction
        has_ally = False
        has_enemy = False
        if self._group_feat_id is not None:
            jr, jc = self._loc_to_rowcol(junction_loc)
            for i in range(obs.shape[0]):
                loc, feat_id, value = int(obs[i, 0]), int(obs[i, 1]), int(obs[i, 2])
                if loc >= LOC_GLOBAL or feat_id != self._group_feat_id:
                    continue
                ar, ac = self._loc_to_rowcol(loc)
                agent_dist = abs(ar - jr) + abs(ac - jc)
                if agent_dist > 3:
                    continue
                # Group 0 = own team (convention), nonzero = enemy
                if value == 0:
                    has_ally = True
                else:
                    has_enemy = True

        if has_enemy and has_ally:
            return TargetMode.CONTESTED
        if has_enemy:
            return TargetMode.LOST
        return TargetMode.FREE

    def infer_role(self, obs: np.ndarray, agent_id: int = 0) -> int:
        """Infer Role from agent_id (round-robin assignment).

        For live inference, role is assigned by agent_id mod 4.
        For trajectory fitting, use ``infer_role_from_history()`` instead.
        """
        return agent_id % NUM_ROLES

    @staticmethod
    def infer_role_from_history(phase_history: np.ndarray) -> int:
        """Infer Role from behavioral pattern over a phase history.

        Parameters
        ----------
        phase_history : np.ndarray
            Array of Phase values over recent timesteps.

        Returns
        -------
        int
            Role enum value based on most frequent phase pattern.
        """
        if len(phase_history) == 0:
            return Role.SUPPORT

        counts = np.bincount(phase_history, minlength=NUM_PHASES)
        # Classify by dominant phase
        dominant = int(np.argmax(counts))

        if dominant in (Phase.MINE, Phase.EXPLORE):
            return Role.GATHERER
        if dominant in (Phase.CRAFT, Phase.GEAR):
            return Role.CRAFTER
        if dominant == Phase.CAPTURE:
            return Role.CAPTURER
        return Role.SUPPORT

    def infer_state(self, obs: np.ndarray, agent_id: int = 0) -> int:
        """Infer flat state index from full token observation.

        Returns an integer in ``[0, NUM_STATES)``.
        """
        hand = self.infer_hand(obs)
        phase = self.infer_phase(obs, hand)
        target_mode = self.infer_target_mode(obs)
        role = self.infer_role(obs, agent_id)
        return state_index(phase, hand, target_mode, role)

    # -- Observation discretisation -----------------------------------------

    def discretize_obs(self, obs: np.ndarray) -> tuple[int, int, int, int, int, int]:
        """Convert token observation to discrete POMDP observation tuple.

        Returns ``(o_resource, o_station, o_inventory, o_contest,
        o_social, o_role_signal)`` where each value is an index into the
        corresponding observation modality.
        """
        o_inv = int(self.infer_hand(obs))  # Hand enum matches ObsInventory
        o_res, o_sta = self._discretize_spatial(obs)
        o_contest = self._discretize_contest(obs)
        o_social = self._discretize_social(obs)
        o_role = self._discretize_role_signal(obs)
        return (o_res, o_sta, o_inv, o_contest, o_social, o_role)

    # -- Batch processing ---------------------------------------------------

    def discretize_trajectory(
        self,
        obs_seq: np.ndarray,
        agent_ids: np.ndarray | None = None,
    ) -> dict:
        """Discretize a full trajectory of observations.

        Parameters
        ----------
        obs_seq : np.ndarray
            Shape ``(T, N, 200, 3)`` uint8 — T timesteps, N agents.
        agent_ids : np.ndarray | None
            Shape ``(N,)`` — agent IDs for role inference.
            Defaults to ``range(N)``.

        Returns
        -------
        dict with:
            ``states``  — ``(T, N)`` int32, inferred flat state indices.
            ``obs``     — ``(T, N, 6)`` int32, discrete observations.
        """
        T, N = obs_seq.shape[:2]
        n_modalities = len(NUM_OBS)
        states = np.zeros((T, N), dtype=np.int32)
        obs_disc = np.zeros((T, N, n_modalities), dtype=np.int32)

        if agent_ids is None:
            agent_ids = np.arange(N)

        for t in range(T):
            for a in range(N):
                aid = int(agent_ids[a])
                states[t, a] = self.infer_state(obs_seq[t, a], agent_id=aid)
                obs_disc[t, a] = self.discretize_obs(obs_seq[t, a])

        return {"states": states, "obs": obs_disc}

    # -- Internal helpers ---------------------------------------------------

    def _nearest_tagged(self, obs: np.ndarray) -> tuple[str, int] | None:
        """Find the nearest tagged entity category and its distance."""
        if self._tag_feat_id is None:
            return None
        best_cat, best_dist = None, float("inf")
        for i in range(obs.shape[0]):
            loc, feat_id, value = int(obs[i, 0]), int(obs[i, 1]), int(obs[i, 2])
            if loc >= LOC_GLOBAL or feat_id != self._tag_feat_id:
                continue
            cat = self._tag_categories.get(value)
            if cat is None:
                continue
            dist = self._manhattan_from_center(loc)
            if dist < best_dist:
                best_dist = dist
                best_cat = cat
        return (best_cat, best_dist) if best_cat is not None else None

    def _discretize_spatial(self, obs: np.ndarray) -> tuple[int, int]:
        """Extract resource and station observations from spatial tokens."""
        best_res_dist = float("inf")
        best_sta_dist = float("inf")
        best_sta_type = ObsStation.NONE

        if self._tag_feat_id is None:
            return int(ObsResource.NONE), int(ObsStation.NONE)

        sta_map = {"hub": ObsStation.HUB, "craft": ObsStation.CRAFT,
                   "junction": ObsStation.JUNCTION}

        for i in range(obs.shape[0]):
            loc, feat_id, value = int(obs[i, 0]), int(obs[i, 1]), int(obs[i, 2])
            if loc >= LOC_GLOBAL or feat_id != self._tag_feat_id:
                continue
            cat = self._tag_categories.get(value)
            if cat is None:
                continue
            dist = self._manhattan_from_center(loc)

            if cat == "extractor" and dist < best_res_dist:
                best_res_dist = dist
            if cat in sta_map and dist < best_sta_dist:
                best_sta_dist = dist
                best_sta_type = sta_map[cat]

        # Threshold into NONE / NEAR / AT
        if best_res_dist <= 1:
            o_res = ObsResource.AT
        elif best_res_dist <= self._near_radius:
            o_res = ObsResource.NEAR
        else:
            o_res = ObsResource.NONE

        if best_sta_dist > self._near_radius:
            best_sta_type = ObsStation.NONE

        return int(o_res), int(best_sta_type)

    def _discretize_contest(self, obs: np.ndarray) -> int:
        """Discretize junction contest status."""
        return int(self.infer_target_mode(obs))

    def _discretize_social(self, obs: np.ndarray) -> int:
        """Discretize nearby agent presence from group tokens."""
        if self._group_feat_id is None:
            return ObsSocial.ALONE

        has_ally = False
        has_enemy = False

        for i in range(obs.shape[0]):
            loc, feat_id, value = int(obs[i, 0]), int(obs[i, 1]), int(obs[i, 2])
            if loc >= LOC_GLOBAL or feat_id != self._group_feat_id:
                continue
            # Skip self (at center)
            dist = self._manhattan_from_center(loc)
            if dist == 0:
                continue
            if dist > self._near_radius:
                continue
            if value == 0:
                has_ally = True
            else:
                has_enemy = True

        if has_ally and has_enemy:
            return ObsSocial.BOTH_NEAR
        if has_ally:
            return ObsSocial.ALLY_NEAR
        if has_enemy:
            return ObsSocial.ENEMY_NEAR
        return ObsSocial.ALONE

    def _discretize_role_signal(self, obs: np.ndarray) -> int:
        """Discretize teammate role similarity from vibe tokens.

        Checks if nearest ally has the same vibe value (proxy for role).
        Returns SAME_ROLE if no allies visible (default / uninformative).
        """
        if self._vibe_feat_id is None:
            return ObsRoleSignal.SAME_ROLE

        own_vibe = None
        nearest_ally_vibe = None
        nearest_ally_dist = float("inf")

        for i in range(obs.shape[0]):
            loc, feat_id, value = int(obs[i, 0]), int(obs[i, 1]), int(obs[i, 2])
            if feat_id != self._vibe_feat_id:
                continue
            dist = self._manhattan_from_center(loc)
            if dist == 0:
                own_vibe = value
                continue
            if loc >= LOC_GLOBAL:
                continue
            # Check if this is an ally (need group check)
            if dist < nearest_ally_dist:
                nearest_ally_dist = dist
                nearest_ally_vibe = value

        if own_vibe is None or nearest_ally_vibe is None:
            return ObsRoleSignal.SAME_ROLE  # uninformative default

        if own_vibe == nearest_ally_vibe:
            return ObsRoleSignal.SAME_ROLE
        return ObsRoleSignal.DIFFERENT_ROLE
