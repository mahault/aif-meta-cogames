"""Tests for the discrete AIF agent module (216-state, 13 task policies)."""

import numpy as np
import pytest

from aif_meta_cogames.aif_agent import (
    NUM_ACTIONS,
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
    TASK_POLICY_NAMES,
    infer_task_policy,
    state_factors,
    state_index,
    state_label,
)
from aif_meta_cogames.aif_agent.discretizer import (
    COGSGUARD_TAG_CATEGORIES,
    LOC_GLOBAL,
    ObservationDiscretizer,
)
from aif_meta_cogames.aif_agent.generative_model import (
    CogsGuardPOMDP,
    build_default_A,
    build_default_B,
    build_C,
    build_D,
    build_uniform_A,
    build_uniform_B,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Minimal obs_feature_names list for testing.
OBS_FEATURES = [
    "agent:group",      # 0
    "agent:frozen",     # 1
    "episode_pct",      # 2
    "last_action",      # 3
    "last_reward",      # 4
    "goal",             # 5
    "vibe",             # 6
    "tag",              # 7
    "cooldown",         # 8
    "remaining_uses",   # 9
    "lp:east",          # 10
    "lp:west",          # 11
    "lp:north",         # 12
    "lp:south",         # 13
    "agent_id",         # 14
    "inv:energy",       # 15
    "inv:energy:p1",    # 16
    "inv:energy:p2",    # 17
    "inv:heart",        # 18
    "inv:heart:p1",     # 19
    "inv:heart:p2",     # 20
    "inv:hp",           # 21
    "inv:hp:p1",        # 22
    "inv:hp:p2",        # 23
    "inv:solar",        # 24
    "inv:solar:p1",     # 25
    "inv:solar:p2",     # 26
    "inv:oxygen",       # 27
    "inv:oxygen:p1",    # 28
    "inv:oxygen:p2",    # 29
    "inv:carbon",       # 30
    "inv:carbon:p1",    # 31
    "inv:carbon:p2",    # 32
    "inv:germanium",    # 33
    "inv:germanium:p1", # 34
    "inv:germanium:p2", # 35
    "inv:silicon",      # 36
    "inv:silicon:p1",   # 37
    "inv:silicon:p2",   # 38
    "inv:aligner",      # 39
    "inv:aligner:p1",   # 40
    "inv:aligner:p2",   # 41
    "inv:scrambler",    # 42
    "inv:scrambler:p1", # 43
    "inv:scrambler:p2", # 44
    "inv:miner",        # 45
    "inv:miner:p1",     # 46
    "inv:miner:p2",     # 47
    "inv:scout",        # 48
    "inv:scout:p1",     # 49
    "inv:scout:p2",     # 50
]


def _empty_obs() -> np.ndarray:
    """All-empty observation: 200 tokens, all padding."""
    obs = np.full((200, 3), 255, dtype=np.uint8)
    return obs


def _obs_with_inventory(feat_id: int, value: int) -> np.ndarray:
    """Observation with a single global inventory token set."""
    obs = _empty_obs()
    obs[0] = [LOC_GLOBAL, feat_id, value]
    return obs


def _obs_with_tag(tag_value: int, location: int) -> np.ndarray:
    """Observation with a single spatial tag token."""
    obs = _empty_obs()
    obs[0] = [location, 7, tag_value]  # feature_id 7 = tag
    return obs


@pytest.fixture
def disc():
    return ObservationDiscretizer(OBS_FEATURES)


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

class TestDimensions:
    def test_state_dimensions(self):
        assert NUM_PHASES == 6
        assert NUM_HANDS == 3
        assert NUM_TARGET_MODES == 3
        assert NUM_ROLES == 4
        assert NUM_STATES == 216  # 6 * 3 * 3 * 4

    def test_action_dimensions(self):
        assert NUM_TASK_POLICIES == 13
        assert NUM_ACTIONS == 13
        assert len(TASK_POLICY_NAMES) == 13

    def test_obs_dimensions(self):
        assert len(NUM_OBS) == 6
        assert NUM_OBS == [3, 4, 3, 3, 4, 2]


# ---------------------------------------------------------------------------
# State indexing (4 factors)
# ---------------------------------------------------------------------------

class TestStateIndexing:
    def test_roundtrip(self):
        for p in range(NUM_PHASES):
            for h in range(NUM_HANDS):
                for t in range(NUM_TARGET_MODES):
                    for r in range(NUM_ROLES):
                        idx = state_index(p, h, t, r)
                        assert 0 <= idx < NUM_STATES
                        assert state_factors(idx) == (p, h, t, r)

    def test_unique_indices(self):
        indices = set()
        for p in range(NUM_PHASES):
            for h in range(NUM_HANDS):
                for t in range(NUM_TARGET_MODES):
                    for r in range(NUM_ROLES):
                        idx = state_index(p, h, t, r)
                        assert idx not in indices
                        indices.add(idx)
        assert len(indices) == NUM_STATES

    def test_label(self):
        assert state_label(0) == "EXPLORE/EMPTY/FREE/GATHERER"
        idx = state_index(Phase.CAPTURE, Hand.HOLDING_GEAR, TargetMode.CONTESTED, Role.CAPTURER)
        assert state_label(idx) == "CAPTURE/HOLDING_GEAR/CONTESTED/CAPTURER"

    def test_default_target_mode_and_role(self):
        # Backward compat: state_index with just phase, hand defaults to FREE, GATHERER
        idx = state_index(Phase.EXPLORE, Hand.EMPTY)
        p, h, t, r = state_factors(idx)
        assert p == Phase.EXPLORE
        assert h == Hand.EMPTY
        assert t == TargetMode.FREE
        assert r == Role.GATHERER


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestEnums:
    def test_task_policy_values(self):
        assert TaskPolicy.NAV_RESOURCE == 0
        assert TaskPolicy.WAIT == 12
        assert len(TaskPolicy) == 13

    def test_target_mode_values(self):
        assert TargetMode.FREE == 0
        assert TargetMode.CONTESTED == 1
        assert TargetMode.LOST == 2

    def test_role_values(self):
        assert Role.GATHERER == 0
        assert Role.SUPPORT == 3
        assert len(Role) == 4


# ---------------------------------------------------------------------------
# Task-policy inference
# ---------------------------------------------------------------------------

class TestTaskPolicyInference:
    def test_nav_resource(self):
        tp = infer_task_policy(Phase.EXPLORE, Hand.EMPTY, Phase.MINE, Hand.EMPTY)
        assert tp == TaskPolicy.NAV_RESOURCE

    def test_mine(self):
        tp = infer_task_policy(Phase.MINE, Hand.EMPTY, Phase.MINE, Hand.HOLDING_RESOURCE)
        assert tp == TaskPolicy.MINE

    def test_nav_depot(self):
        tp = infer_task_policy(Phase.MINE, Hand.HOLDING_RESOURCE, Phase.DEPOSIT, Hand.HOLDING_RESOURCE)
        assert tp == TaskPolicy.NAV_DEPOT

    def test_deposit(self):
        tp = infer_task_policy(Phase.DEPOSIT, Hand.HOLDING_RESOURCE, Phase.DEPOSIT, Hand.EMPTY)
        assert tp == TaskPolicy.DEPOSIT

    def test_craft(self):
        tp = infer_task_policy(Phase.CRAFT, Hand.EMPTY, Phase.CRAFT, Hand.HOLDING_GEAR)
        assert tp == TaskPolicy.CRAFT

    def test_capture_success(self):
        tp = infer_task_policy(Phase.CAPTURE, Hand.HOLDING_GEAR, Phase.EXPLORE, Hand.EMPTY)
        assert tp == TaskPolicy.CAPTURE

    def test_self_transition_is_wait(self):
        # CRAFT/HOLDING_RESOURCE → same = no matching economy pattern → WAIT
        tp = infer_task_policy(Phase.CRAFT, Hand.HOLDING_RESOURCE,
                               Phase.CRAFT, Hand.HOLDING_RESOURCE)
        assert tp == TaskPolicy.WAIT

    def test_deposit_self_transition_is_deposit(self):
        # DEPOSIT/HOLDING_RESOURCE → same = still depositing (not WAIT)
        tp = infer_task_policy(Phase.DEPOSIT, Hand.HOLDING_RESOURCE,
                               Phase.DEPOSIT, Hand.HOLDING_RESOURCE)
        assert tp == TaskPolicy.DEPOSIT

    def test_explore_self_transition(self):
        tp = infer_task_policy(Phase.EXPLORE, Hand.EMPTY, Phase.EXPLORE, Hand.EMPTY)
        assert tp == TaskPolicy.EXPLORE


# ---------------------------------------------------------------------------
# Discretizer — hand inference
# ---------------------------------------------------------------------------

class TestHandInference:
    def test_empty_obs(self, disc):
        obs = _empty_obs()
        assert disc.infer_hand(obs) == Hand.EMPTY

    def test_holding_resource(self, disc):
        obs = _obs_with_inventory(30, 5)  # inv:carbon = 5
        assert disc.infer_hand(obs) == Hand.HOLDING_RESOURCE

    def test_holding_gear(self, disc):
        obs = _obs_with_inventory(39, 1)  # inv:aligner = 1
        assert disc.infer_hand(obs) == Hand.HOLDING_GEAR

    def test_gear_takes_priority(self, disc):
        obs = _empty_obs()
        obs[0] = [LOC_GLOBAL, 30, 5]   # carbon
        obs[1] = [LOC_GLOBAL, 39, 1]   # aligner gear
        assert disc.infer_hand(obs) == Hand.HOLDING_GEAR

    def test_zero_value_ignored(self, disc):
        obs = _obs_with_inventory(30, 0)  # inv:carbon = 0 (empty)
        assert disc.infer_hand(obs) == Hand.EMPTY


# ---------------------------------------------------------------------------
# Discretizer — phase inference
# ---------------------------------------------------------------------------

class TestPhaseInference:
    def test_explore_when_empty(self, disc):
        obs = _empty_obs()
        assert disc.infer_phase(obs, Hand.EMPTY) == Phase.EXPLORE

    def test_deposit_when_holding_resource(self, disc):
        obs = _empty_obs()
        assert disc.infer_phase(obs, Hand.HOLDING_RESOURCE) == Phase.DEPOSIT

    def test_capture_when_holding_gear(self, disc):
        obs = _empty_obs()
        assert disc.infer_phase(obs, Hand.HOLDING_GEAR) == Phase.CAPTURE

    def test_mine_when_adjacent_to_extractor(self, disc):
        obs = _obs_with_tag(15, 103)  # carbon_extractor adjacent
        assert disc.infer_phase(obs, Hand.EMPTY) == Phase.MINE

    def test_deposit_when_adjacent_to_hub(self, disc):
        obs = _obs_with_tag(17, 102)  # hub at center
        assert disc.infer_phase(obs, Hand.HOLDING_RESOURCE) == Phase.DEPOSIT

    def test_capture_when_adjacent_to_junction(self, disc):
        obs = _obs_with_tag(18, 102)  # junction at center
        assert disc.infer_phase(obs, Hand.HOLDING_GEAR) == Phase.CAPTURE

    def test_craft_when_at_craft_station_with_resource(self, disc):
        obs = _obs_with_tag(11, 102)  # c:aligner at center
        assert disc.infer_phase(obs, Hand.HOLDING_RESOURCE) == Phase.CRAFT

    def test_gear_when_at_craft_station_empty(self, disc):
        obs = _obs_with_tag(11, 102)  # c:aligner at center
        assert disc.infer_phase(obs, Hand.EMPTY) == Phase.GEAR


# ---------------------------------------------------------------------------
# Discretizer — observation discretisation (6 modalities)
# ---------------------------------------------------------------------------

class TestObservationDiscretisation:
    def test_empty_obs_returns_defaults(self, disc):
        obs = _empty_obs()
        result = disc.discretize_obs(obs)
        assert len(result) == 6
        o_res, o_sta, o_inv, o_con, o_soc, o_role = result
        assert o_res == ObsResource.NONE
        assert o_sta == ObsStation.NONE
        assert o_inv == ObsInventory.EMPTY
        assert o_con == ObsContest.FREE
        assert o_soc == ObsSocial.ALONE
        assert o_role == ObsRoleSignal.SAME_ROLE

    def test_inventory_reflects_hand(self, disc):
        obs = _obs_with_inventory(39, 1)  # aligner gear
        result = disc.discretize_obs(obs)
        assert result[2] == ObsInventory.HAS_GEAR

    def test_adjacent_extractor(self, disc):
        obs = _obs_with_tag(15, 103)  # carbon_extractor adjacent
        result = disc.discretize_obs(obs)
        assert result[0] == ObsResource.AT

    def test_near_extractor(self, disc):
        obs = _obs_with_tag(15, 105)  # dist=3 from center
        result = disc.discretize_obs(obs)
        assert result[0] == ObsResource.NEAR

    def test_far_extractor(self, disc):
        obs = _obs_with_tag(15, 34)  # dist=8 from center
        result = disc.discretize_obs(obs)
        assert result[0] == ObsResource.NONE

    def test_adjacent_junction(self, disc):
        obs = _obs_with_tag(18, 102)  # junction at center
        result = disc.discretize_obs(obs)
        assert result[1] == ObsStation.JUNCTION


# ---------------------------------------------------------------------------
# Discretizer — batch processing
# ---------------------------------------------------------------------------

class TestBatchDiscretization:
    def test_trajectory_shape(self, disc):
        obs_seq = np.full((10, 4, 200, 3), 255, dtype=np.uint8)
        result = disc.discretize_trajectory(obs_seq)
        assert result["states"].shape == (10, 4)
        assert result["obs"].shape == (10, 4, 6)  # 6 modalities

    def test_all_empty_trajectory(self, disc):
        obs_seq = np.full((5, 2, 200, 3), 255, dtype=np.uint8)
        result = disc.discretize_trajectory(obs_seq)
        # All should be EXPLORE/EMPTY/FREE/GATHERER (agent_id=0 → role=GATHERER)
        expected_state = state_index(Phase.EXPLORE, Hand.EMPTY, TargetMode.FREE, Role.GATHERER)
        assert np.all(result["states"][:, 0] == expected_state)


# ---------------------------------------------------------------------------
# Generative model — matrix properties
# ---------------------------------------------------------------------------

class TestGenerativeModel:
    def test_A_shapes(self):
        A = build_default_A()
        assert len(A) == 6
        for m, a in enumerate(A):
            assert a.shape == (NUM_OBS[m], NUM_STATES)

    def test_A_normalised(self):
        A = build_default_A()
        for a in A:
            col_sums = a.sum(axis=0)
            np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)

    def test_B_shape(self):
        B = build_default_B()
        assert len(B) == 1
        assert B[0].shape == (NUM_STATES, NUM_STATES, NUM_ACTIONS)

    def test_B_normalised(self):
        B = build_default_B()
        for s in range(NUM_STATES):
            for a in range(NUM_ACTIONS):
                col_sum = B[0][:, s, a].sum()
                assert abs(col_sum - 1.0) < 1e-10, (
                    f"B column for state={state_label(s)}, action={TASK_POLICY_NAMES[a]} "
                    f"sums to {col_sum}"
                )

    def test_B_is_action_dependent(self):
        """B must be action-dependent — different task policies produce different transitions."""
        B = build_default_B()[0]
        s = state_index(Phase.EXPLORE, Hand.EMPTY, TargetMode.FREE, Role.GATHERER)
        # NAV_RESOURCE should have different transitions than WAIT
        nav_col = B[:, s, TaskPolicy.NAV_RESOURCE]
        wait_col = B[:, s, TaskPolicy.WAIT]
        assert not np.allclose(nav_col, wait_col), (
            "B must be action-dependent: NAV_RESOURCE and WAIT should differ"
        )

    def test_C_shapes(self):
        C = build_C()
        assert len(C) == 6
        for m, c in enumerate(C):
            assert c.shape == (NUM_OBS[m],)

    def test_D_shape_and_normalisation(self):
        D = build_D()
        assert len(D) == 1
        assert D[0].shape == (NUM_STATES,)
        np.testing.assert_allclose(D[0].sum(), 1.0, atol=1e-10)

    def test_D_peaks_at_explore_empty_free(self):
        D = build_D()
        # Should have highest mass across EXPLORE/EMPTY/FREE for all roles
        explore_empty_free = [
            state_index(Phase.EXPLORE, Hand.EMPTY, TargetMode.FREE, r)
            for r in Role
        ]
        for s in explore_empty_free:
            assert D[0][s] > D[0].mean(), (
                f"D[{state_label(s)}] should be above average"
            )

    def test_uniform_A_is_uniform(self):
        A = build_uniform_A()
        for m, a in enumerate(A):
            expected_val = 1.0 / NUM_OBS[m]
            np.testing.assert_allclose(a, expected_val, atol=1e-10)


# ---------------------------------------------------------------------------
# POMDP wrapper
# ---------------------------------------------------------------------------

class TestCogsGuardPOMDP:
    def test_default_construction(self):
        model = CogsGuardPOMDP()
        assert len(model.A) == 6
        assert len(model.B) == 1
        assert len(model.C) == 6
        assert len(model.D) == 1

    def test_uniform_construction(self):
        model = CogsGuardPOMDP.uniform()
        for a in model.A:
            col_sums = a.sum(axis=0)
            np.testing.assert_allclose(col_sums, 1.0, atol=1e-10)

    def test_save_and_load(self, tmp_path):
        model = CogsGuardPOMDP()
        save_path = tmp_path / "test_model.npz"
        model.save(save_path)

        loaded = CogsGuardPOMDP.from_fitted(save_path)
        for m in range(6):
            np.testing.assert_array_almost_equal(model.A[m], loaded.A[m])
        np.testing.assert_array_almost_equal(model.B[0], loaded.B[0])

    def test_summary(self):
        model = CogsGuardPOMDP()
        text = model.summary()
        assert "216" in text
        assert "EXPLORE/EMPTY/FREE/GATHERER" in text
        assert "13" in text
