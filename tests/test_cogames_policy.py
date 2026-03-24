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
    Role,
    TargetMode,
)
from aif_meta_cogames.aif_agent.generative_model import (
    CogsGuardPOMDP,
    NUM_STATE_FACTORS,
)


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
    """Test the navigation target selection from factored beliefs."""

    def test_holding_gear_goes_to_junction(self):
        """When belief says HOLDING_GEAR, target should be junction."""
        phase_qs = np.zeros(6)
        phase_qs[Phase.CAPTURE] = 0.9
        phase_qs[Phase.EXPLORE] = 0.1
        hand_qs = np.zeros(3)
        hand_qs[Hand.HOLDING_GEAR] = 1.0

        assert int(np.argmax(phase_qs)) == Phase.CAPTURE
        assert int(np.argmax(hand_qs)) == Hand.HOLDING_GEAR

    def test_holding_resource_goes_to_hub(self):
        """When belief says HOLDING_RESOURCE, target should be hub."""
        phase_qs = np.zeros(6)
        phase_qs[Phase.DEPOSIT] = 0.9
        phase_qs[Phase.EXPLORE] = 0.1
        hand_qs = np.zeros(3)
        hand_qs[Hand.HOLDING_RESOURCE] = 1.0

        assert int(np.argmax(phase_qs)) == Phase.DEPOSIT
        assert int(np.argmax(hand_qs)) == Hand.HOLDING_RESOURCE

    def test_empty_explore_goes_to_extractor(self):
        """When empty-handed and exploring, target should be extractor."""
        phase_qs = np.zeros(6)
        phase_qs[Phase.EXPLORE] = 0.95
        phase_qs[Phase.MINE] = 0.05
        hand_qs = np.zeros(3)
        hand_qs[Hand.EMPTY] = 1.0

        assert int(np.argmax(phase_qs)) == Phase.EXPLORE
        assert int(np.argmax(hand_qs)) == Hand.EMPTY

    def test_craft_phase_goes_to_craft_station(self):
        """When in CRAFT phase with empty hand, target craft station."""
        phase_qs = np.zeros(6)
        phase_qs[Phase.CRAFT] = 0.8
        phase_qs[Phase.EXPLORE] = 0.2
        hand_qs = np.zeros(3)
        hand_qs[Hand.EMPTY] = 1.0

        assert int(np.argmax(phase_qs)) == Phase.CRAFT
        assert int(np.argmax(hand_qs)) == Hand.EMPTY


@pytest.mark.skipif(not HAS_PYMDP, reason="pymdp (JAX) not installed")
class TestPOMDPAgentCreation:
    """Test that pymdp JAX Agent can be created from CogsGuardPOMDP."""

    def test_create_default_agent(self):
        model = CogsGuardPOMDP()
        agent = model.create_agent(policy_len=2, use_states_info_gain=True)

        # JAX Agent adds batch dimension to A/B matrices
        assert len(agent.A) == 6  # 6 observation modalities
        # A shapes: (batch, n_obs, *dep_factor_dims)
        assert agent.A[0].shape == (1, 3, 6)   # (batch, o_resource, phase)
        assert agent.A[1].shape == (1, 4, 6)   # (batch, o_station, phase)
        assert agent.A[2].shape == (1, 3, 3)   # (batch, o_inventory, hand)
        assert agent.A[3].shape == (1, 3, 3)   # (batch, o_contest, target_mode)
        assert agent.A[4].shape == (1, 4, 4, 3)  # (batch, o_social, role, target_mode)
        assert agent.A[5].shape == (1, 2, 4)   # (batch, o_role_signal, role)

        # B shapes: (batch, n_states_f', *dep_dims, n_controls_f)
        assert len(agent.B) == 4  # 4 state factors
        assert agent.B[0].shape == (1, 6, 6, 3, 13)  # phase: (batch, p', p, h, actions)
        assert agent.B[1].shape == (1, 3, 6, 3, 13)  # hand: (batch, h', p, h, actions)
        assert agent.B[2].shape == (1, 3, 3, 13)      # target: (batch, t', t, actions)
        assert agent.B[3].shape == (1, 4, 4, 13)      # role: (batch, r', r, actions)

    def test_agent_inference_loop(self):
        """Test the full pymdp JAX inference cycle."""
        model = CogsGuardPOMDP()
        agent = model.create_agent(policy_len=2, use_states_info_gain=True)

        # Observe: all zeros (NONE/NONE/EMPTY/FREE/ALONE/SAME)
        # Shape: (batch=1,) per modality
        obs = [jnp.array([[0]]) for _ in range(6)]
        qs = agent.infer_states(obs, empirical_prior=agent.D)

        # qs is a list of 4 factors
        assert len(qs) == 4
        # Each qs[f] shape: (batch=1, T=1, n_states_f)
        assert qs[0].shape[0] == 1  # batch
        assert qs[0].shape[-1] == 6  # phase states
        assert qs[1].shape[-1] == 3  # hand states

        # Compute EFE
        q_pi, G = agent.infer_policies(qs)
        assert q_pi is not None
        assert G is not None

        # Sample action
        action = agent.sample_action(q_pi)
        # action shape: (batch=1, num_factors=4)
        assert action.shape == (1, 4)
        # All factors should have same action (constrained policies)
        assert int(action[0, 0]) == int(action[0, 1])
        assert int(action[0, 0]) == int(action[0, 2])
        assert int(action[0, 0]) == int(action[0, 3])
        assert 0 <= int(action[0, 0]) < 13

    def test_agent_belief_update(self):
        """Beliefs should shift when observations change."""
        model = CogsGuardPOMDP()
        agent = model.create_agent(policy_len=2, use_states_info_gain=True)

        # Initial: observe nothing (expect EXPLORE to dominate phase beliefs)
        obs_none = [jnp.array([[0]]) for _ in range(6)]
        qs = agent.infer_states(obs_none, empirical_prior=agent.D)
        phase_initial = np.asarray(qs[0][0, -1]).copy()  # (n_states_f=6,)

        # Update empirical prior
        q_pi, _G = agent.infer_policies(qs)
        action = agent.sample_action(q_pi)
        pred, _ = agent.update_empirical_prior(action, qs)

        # Now observe: AT extractor (o_res=2), NONE station, HAS_RESOURCE (o_inv=1)
        # contest=FREE(0), social=ALONE(0), role=SAME(0)
        obs_mining = [jnp.array([[2]]), jnp.array([[0]]), jnp.array([[1]]),
                      jnp.array([[0]]), jnp.array([[0]]), jnp.array([[0]])]
        qs2 = agent.infer_states(obs_mining, empirical_prior=pred)
        phase_after = np.asarray(qs2[0][0, -1]).copy()  # (n_states_f=6,)

        # MINE phase should have higher probability after seeing AT extractor
        assert phase_after[Phase.MINE] > phase_initial[Phase.MINE]

    def test_empirical_prior_propagation(self):
        """Empirical prior should propagate beliefs through B matrix."""
        model = CogsGuardPOMDP()
        agent = model.create_agent(policy_len=2, use_states_info_gain=True)

        # Initial inference
        obs = [jnp.array([[0]]) for _ in range(6)]
        qs = agent.infer_states(obs, empirical_prior=agent.D)

        # Propagate through B
        action = jnp.array([[0, 0, 0, 0]])  # NAV_RESOURCE for all factors
        pred, _ = agent.update_empirical_prior(action, qs)

        # empirical_prior should be a list with 4 elements (one per factor)
        assert len(pred) == 4
        # Each factor's prior: (batch=1, n_states_f)
        assert pred[0].shape == (1, 6)   # phase
        assert pred[1].shape == (1, 3)   # hand
        assert pred[2].shape == (1, 3)   # target_mode
        assert pred[3].shape == (1, 4)   # role
        # Should be valid probability distributions
        for f_pred in pred:
            total = float(jnp.sum(f_pred))
            assert abs(total - 1.0) < 1e-4


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
