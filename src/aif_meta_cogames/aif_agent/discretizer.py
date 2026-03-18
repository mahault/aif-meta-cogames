"""Observation discretizer: token obs (200, 3) -> discrete POMDP quantities.

Maps raw uint8 token observations from MettaGrid to the discrete state and
observation spaces used by the CogsGuard POMDP generative model.

State space (18 states):
    phase(6) x hand(3)

Observation modalities:
    o_resource(3): extractor proximity
    o_station(4):  hub / craft-station / junction proximity
    o_inventory(3): what the agent is holding

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


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

NUM_PHASES = len(Phase)       # 6
NUM_HANDS = len(Hand)         # 3
NUM_STATES = NUM_PHASES * NUM_HANDS  # 18

NUM_OBS = [len(ObsResource), len(ObsStation), len(ObsInventory)]  # [3, 4, 3]
OBS_MODALITY_NAMES = ["o_resource", "o_station", "o_inventory"]

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
# Tag values are indices into PolicyEnvInterface.tags (alphabetically sorted).
# Verified against cogames 0.18.x.
# To confirm on a live environment, run:
#     pei = PolicyEnvInterface.from_mg_cfg(env_cfg)
#     print(list(enumerate(pei.tags)))
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
# State indexing
# ---------------------------------------------------------------------------

def state_index(phase: int, hand: int) -> int:
    """Flat state index from (phase, hand) factors."""
    return int(phase) * NUM_HANDS + int(hand)


def state_factors(flat_idx: int) -> tuple[int, int]:
    """Recover (phase, hand) from a flat state index."""
    return flat_idx // NUM_HANDS, flat_idx % NUM_HANDS


def state_label(flat_idx: int) -> str:
    """Human-readable label for a flat state index."""
    p, h = state_factors(flat_idx)
    return f"{Phase(p).name}/{Hand(h).name}"


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
        """Infer Hand state from global inventory tokens.

        Checks gear first (higher priority if agent has both gear and
        resource features simultaneously, which shouldn't happen in
        normal play but is handled gracefully).
        """
        has_resource = False
        for i in range(obs.shape[0]):
            loc, feat_id, value = int(obs[i, 0]), int(obs[i, 1]), int(obs[i, 2])
            if loc != LOC_GLOBAL or value == 0:
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

    def infer_state(self, obs: np.ndarray) -> int:
        """Infer flat state index from full token observation.

        Returns an integer in ``[0, NUM_STATES)``.
        """
        hand = self.infer_hand(obs)
        phase = self.infer_phase(obs, hand)
        return state_index(phase, hand)

    # -- Observation discretisation -----------------------------------------

    def discretize_obs(self, obs: np.ndarray) -> tuple[int, int, int]:
        """Convert token observation to discrete POMDP observation tuple.

        Returns ``(o_resource, o_station, o_inventory)`` where each value
        is an index into the corresponding observation modality.
        """
        o_inv = int(self.infer_hand(obs))  # Hand enum matches ObsInventory
        o_res, o_sta = self._discretize_spatial(obs)
        return (o_res, o_sta, o_inv)

    # -- Batch processing ---------------------------------------------------

    def discretize_trajectory(self, obs_seq: np.ndarray) -> dict:
        """Discretize a full trajectory of observations.

        Parameters
        ----------
        obs_seq : np.ndarray
            Shape ``(T, N, 200, 3)`` uint8 — T timesteps, N agents.

        Returns
        -------
        dict with:
            ``states``  — ``(T, N)`` int32, inferred flat state indices.
            ``obs``     — ``(T, N, 3)`` int32, discrete observations.
        """
        T, N = obs_seq.shape[:2]
        states = np.zeros((T, N), dtype=np.int32)
        obs_disc = np.zeros((T, N, len(NUM_OBS)), dtype=np.int32)

        for t in range(T):
            for a in range(N):
                states[t, a] = self.infer_state(obs_seq[t, a])
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
