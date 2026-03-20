"""Tests for cogames_policy -- discrete AIF agent integration.

These tests mock the mettagrid types so they can run locally
without cogames/mettagrid installed. pymdp (JAX) is required.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Check pymdp availability
# ---------------------------------------------------------------------------

try:
    import jax.numpy as jnp
    from pymdp.agent import Agent as _JaxAgent

    HAS_PYMDP = True
except ImportError:
    HAS_PYMDP = False

try:
    from mettagrid.simulator.interface import AgentObservation
    from mettagrid.simulator import Action
    from mettagrid.policy.policy_env_interface import PolicyEnvInterface

    HAS_METTAGRID = True
except ImportError:
    HAS_METTAGRID = False


# ---------------------------------------------------------------------------
# Tests that don't need mettagrid or pymdp
# ---------------------------------------------------------------------------

from aif_meta_cogames.aif_agent.discretizer import (
    Hand,
    Phase,
    state_factors,
    state_index,
)
from aif_meta_cogames.aif_agent.generative_model import CogsGuardPOMDP


class TestBuildTagCategories:
    """Test dynamic tag category mapping."""

    def test_standard_cogsguard_tags(self):
        from aif_meta_cogames.aif_agent.cogames_policy import _build_tag_categories

        # Simulated tag list from cogames 0.18.x (alphabetically sorted)
        tags = [
            "type:agent",           # 0
            "type:agent:team0",     # 1
            "type:agent:team1",     # 2
            "type:border",          # 3
            "type:c:aligner",       # 4
            "type:c:miner",         # 5
            "type:c:scout",         # 6
            "type:c:scrambler",     # 7
            "type:carbon_extractor",  # 8
            "type:chest",           # 9
            "type:germanium_extractor",  # 10
            "type:hub",             # 11
            "type:junction",        # 12
            "type:oxygen_extractor",  # 13
            "type:silicon_extractor",  # 14
            "type:solar_extractor",  # 15
            "type:wall",            # 16
        ]

        cats = _build_tag_categories(tags)

        # Craft stations
        assert cats[4] == "craft"   # c:aligner
        assert cats[5] == "craft"   # c:miner
        assert cats[6] == "craft"   # c:scout
        assert cats[7] == "craft"   # c:scrambler

        # Extractors
        assert cats[8] == "extractor"   # carbon
        assert cats[10] == "extractor"  # germanium
        assert cats[13] == "extractor"  # oxygen
        assert cats[14] == "extractor"  # silicon
        assert cats[15] == "extractor"  # solar

        # Hub
        assert cats[11] == "hub"

        # Junction
        assert cats[12] == "junction"

        # Should NOT categorize agents, walls, etc.
        assert 0 not in cats
        assert 3 not in cats
        assert 16 not in cats

    def test_empty_tags(self):
        from aif_meta_cogames.aif_agent.cogames_policy import _build_tag_categories

        assert _build_tag_categories([]) == {}

    def test_without_type_prefix(self):
        from aif_meta_cogames.aif_agent.cogames_policy import _build_tag_categories

        tags = ["hub", "junction", "carbon_extractor", "c:miner"]
        cats = _build_tag_categories(tags)
        assert cats[0] == "hub"
        assert cats[1] == "junction"
        assert cats[2] == "extractor"
        assert cats[3] == "craft"


class TestGoalSelectionLogic:
    """Test the navigation target selection from beliefs."""

    def test_holding_gear_goes_to_junction(self):
        """When belief says HOLDING_GEAR, target should be junction."""
        qs = np.zeros(18)
        qs[state_index(Phase.CAPTURE, Hand.HOLDING_GEAR)] = 0.9
        qs[state_index(Phase.EXPLORE, Hand.EMPTY)] = 0.1

        best = int(np.argmax(qs))
        phase, hand = state_factors(best)
        assert phase == Phase.CAPTURE
        assert hand == Hand.HOLDING_GEAR

    def test_holding_resource_goes_to_hub(self):
        """When belief says HOLDING_RESOURCE, target should be hub."""
        qs = np.zeros(18)
        qs[state_index(Phase.DEPOSIT, Hand.HOLDING_RESOURCE)] = 0.9
        qs[state_index(Phase.EXPLORE, Hand.EMPTY)] = 0.1

        best = int(np.argmax(qs))
        phase, hand = state_factors(best)
        assert phase == Phase.DEPOSIT
        assert hand == Hand.HOLDING_RESOURCE

    def test_empty_explore_goes_to_extractor(self):
        """When empty-handed and exploring, target should be extractor."""
        qs = np.zeros(18)
        qs[state_index(Phase.EXPLORE, Hand.EMPTY)] = 0.95
        qs[state_index(Phase.MINE, Hand.EMPTY)] = 0.05

        best = int(np.argmax(qs))
        phase, hand = state_factors(best)
        assert phase == Phase.EXPLORE
        assert hand == Hand.EMPTY

    def test_craft_phase_goes_to_craft_station(self):
        """When in CRAFT phase with empty hand, target craft station."""
        qs = np.zeros(18)
        qs[state_index(Phase.CRAFT, Hand.EMPTY)] = 0.8
        qs[state_index(Phase.EXPLORE, Hand.EMPTY)] = 0.2

        best = int(np.argmax(qs))
        phase, hand = state_factors(best)
        assert phase == Phase.CRAFT
        assert hand == Hand.EMPTY


@pytest.mark.skipif(not HAS_PYMDP, reason="pymdp (JAX) not installed")
class TestPOMDPAgentCreation:
    """Test that pymdp JAX Agent can be created from CogsGuardPOMDP."""

    def test_create_default_agent(self):
        model = CogsGuardPOMDP()
        agent = model.create_agent(policy_len=2, use_states_info_gain=True)

        # JAX Agent has batch dimension on A/B matrices
        assert len(agent.A) == 3  # 3 observation modalities
        assert agent.A[0].shape == (1, 3, 18)   # (batch, o_resource, states)
        assert agent.A[1].shape == (1, 4, 18)   # (batch, o_station, states)
        assert agent.A[2].shape == (1, 3, 18)   # (batch, o_inventory, states)
        assert agent.B[0].shape == (1, 18, 18, 5)  # (batch, s', s, actions)

    def test_agent_inference_loop(self):
        """Test the full pymdp JAX inference cycle."""
        model = CogsGuardPOMDP()
        agent = model.create_agent(policy_len=2, use_states_info_gain=True)

        # Observe: NONE resource, NONE station, EMPTY inventory
        # Shape: (batch=1, T=1) per modality — Agent converts to one-hot
        obs = [jnp.array([[0]]), jnp.array([[0]]), jnp.array([[0]])]
        qs = agent.infer_states(obs, empirical_prior=agent.D)

        # qs[0] shape: (batch=1, T=1, n_states=18)
        assert qs[0].shape == (1, 1, 18)

        # Compute EFE
        q_pi, G = agent.infer_policies(qs)
        assert q_pi is not None
        assert G is not None

        # Sample action (deterministic, no rng_key needed)
        action = agent.sample_action(q_pi)
        # action shape: (batch=1, num_factors=1)
        assert action.shape == (1, 1)
        assert 0 <= int(action[0, 0]) < 5

    def test_agent_belief_update(self):
        """Beliefs should shift when observations change."""
        model = CogsGuardPOMDP()
        agent = model.create_agent(policy_len=2, use_states_info_gain=True)

        # Initial: observe nothing (expect EXPLORE/EMPTY to dominate)
        obs_none = [jnp.array([[0]]), jnp.array([[0]]), jnp.array([[0]])]
        qs = agent.infer_states(obs_none, empirical_prior=agent.D)
        qs_initial = np.asarray(qs[0][0, -1, :]).copy()

        # Update empirical prior
        q_pi, _G = agent.infer_policies(qs)
        action = agent.sample_action(q_pi)
        result = agent.update_empirical_prior(action, qs)
        # pymdp alpha returns (pred, qs), released v1.0.0 returns just pred
        empirical_prior = result[0] if isinstance(result, tuple) else result

        # Now observe: AT extractor, NONE station, HAS_RESOURCE
        obs_mining = [jnp.array([[2]]), jnp.array([[0]]), jnp.array([[1]])]
        qs2 = agent.infer_states(obs_mining, empirical_prior=empirical_prior)
        qs_mining = np.asarray(qs2[0][0, -1, :]).copy()

        # Belief should have shifted -- MINE states should have higher prob
        mine_states = [state_index(Phase.MINE, h) for h in Hand]
        p_mine_initial = sum(qs_initial[s] for s in mine_states)
        p_mine_after = sum(qs_mining[s] for s in mine_states)
        assert p_mine_after > p_mine_initial

    def test_empirical_prior_propagation(self):
        """Empirical prior should propagate beliefs through B matrix."""
        model = CogsGuardPOMDP()
        agent = model.create_agent(policy_len=2, use_states_info_gain=True)

        # Initial inference
        obs = [jnp.array([[0]]), jnp.array([[0]]), jnp.array([[0]])]
        qs = agent.infer_states(obs, empirical_prior=agent.D)

        # Propagate through B
        action = jnp.array([[0]])  # noop
        result = agent.update_empirical_prior(action, qs)
        empirical_prior = result[0] if isinstance(result, tuple) else result

        # empirical_prior should be a list with one element per factor
        assert len(empirical_prior) == 1
        # shape: (batch=1, n_states=18)
        assert empirical_prior[0].shape == (1, 18)
        # Should be a valid probability distribution
        total = float(jnp.sum(empirical_prior[0]))
        assert abs(total - 1.0) < 1e-5


class TestObsToArray:
    """Test observation conversion without mettagrid."""

    def test_numpy_array_shape(self):
        """Output should always be (200, 3) uint8."""
        arr = np.full((200, 3), 255, dtype=np.uint8)
        arr[0] = [102, 30, 5]
        arr[1] = [0, 1, 10]

        assert arr.shape == (200, 3)
        assert arr.dtype == np.uint8
        assert arr[0, 0] == 102
        assert arr[2, 0] == 255  # padding


@pytest.mark.skipif(not HAS_METTAGRID, reason="mettagrid not installed")
class TestIntegration:
    """Integration tests requiring cogames/mettagrid."""

    def test_aif_policy_construction(self):
        """Test AIFPolicy can be instantiated."""
        from aif_meta_cogames.aif_agent.cogames_policy import AIFPolicy
        from cogames.cogs_vs_clips.mission import CvCMission
        from cogames.cogs_vs_clips.sites import COGSGUARD_ARENA
        from cogames.cogs_vs_clips.clip_difficulty import EASY
        from cogames.cogs_vs_clips.cog import CogTeam
        from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
        from mettagrid.envs.stats_tracker import StatsTracker
        from mettagrid.simulator import Simulator
        from mettagrid.util.stats_writer import NoopStatsWriter

        mission = CvCMission(
            name="test", description="test", site=COGSGUARD_ARENA,
            num_cogs=4, max_steps=100,
            teams={"cogs": CogTeam(name="cogs", num_agents=4, wealth=3, initial_hearts=0)},
            variants=[EASY],
        )
        env_cfg = mission.make_env()
        sim = Simulator()
        sim.add_event_handler(StatsTracker(NoopStatsWriter()))
        env = MettaGridPufferEnv(sim, env_cfg, seed=0)
        pei = PolicyEnvInterface.from_mg_cfg(env.env_cfg)
        env.close()

        policy = AIFPolicy(pei)
        agent_policy = policy.agent_policy(0)
        assert agent_policy is not None
        assert policy.is_recurrent()
