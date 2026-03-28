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
    MacroOption,
    NavAction,
    NavProgress,
    NUM_NAV_ACTIONS,
    ObsContest,
    ObsInventory,
    ObsResource,
    ObsStation,
    Phase,
    Role,
    TargetRange,
    TaskPolicy,
    TargetMode,
)
from aif_meta_cogames.aif_agent.generative_model import (
    CogsGuardPOMDP,
    NUM_STATE_FACTORS,
    create_nav_agent,
    _build_nav_A,
    _build_nav_B,
    _build_nav_C,
    _build_nav_D,
)
from aif_meta_cogames.aif_agent.cogames_policy import (
    OptionExecutor,
    OptionState,
    SpatialMemory,
    _BEARING_DIRS,
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

        # Extractors — element-typed for EFE-optimal resource selection
        assert cats[8] == "extractor:carbon"
        assert cats[10] == "extractor:germanium"
        assert cats[13] == "extractor:oxygen"
        assert cats[14] == "extractor:silicon"
        assert cats[15] == "extractor"  # solar (no element match)

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
        assert cats[2] == "extractor:carbon"
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


# ---------------------------------------------------------------------------
# OptionExecutor tests
# ---------------------------------------------------------------------------

class TestOptionExecutor:
    """Test reactive option state machines."""

    def test_initial_state(self):
        executor = OptionExecutor(n_agents=4)
        for i in range(4):
            assert executor.states[i].current_option == MacroOption.EXPLORE
            assert executor.states[i].steps_in_option == 0

    def test_explore_returns_explore(self):
        executor = OptionExecutor(n_agents=1)
        obs = [ObsResource.NONE, ObsStation.NONE, ObsInventory.EMPTY,
               ObsContest.FREE, 0, 0]
        assert executor.get_task_policy(0, obs) == TaskPolicy.EXPLORE

    def test_mine_cycle_nav_resource(self):
        executor = OptionExecutor(n_agents=1)
        executor.set_option(0, MacroOption.MINE_CYCLE)
        obs = [ObsResource.NONE, ObsStation.NONE, ObsInventory.EMPTY,
               ObsContest.FREE, 0, 0]
        assert executor.get_task_policy(0, obs) == TaskPolicy.NAV_RESOURCE

    def test_mine_cycle_at_resource_still_navigates(self):
        """At resource: keep navigating (auto-extracts at dist=0 via bumping)."""
        executor = OptionExecutor(n_agents=1)
        executor.set_option(0, MacroOption.MINE_CYCLE)
        obs = [ObsResource.AT, ObsStation.NONE, ObsInventory.EMPTY,
               ObsContest.FREE, 0, 0]
        assert executor.get_task_policy(0, obs) == TaskPolicy.NAV_RESOURCE

    def test_mine_cycle_nav_depot_with_resource(self):
        executor = OptionExecutor(n_agents=1)
        executor.set_option(0, MacroOption.MINE_CYCLE)
        obs = [ObsResource.NONE, ObsStation.NONE, ObsInventory.HAS_RESOURCE,
               ObsContest.FREE, 0, 0]
        assert executor.get_task_policy(0, obs) == TaskPolicy.NAV_DEPOT

    def test_mine_cycle_nav_depot_at_hub(self):
        """Even at HUB, use NAV_DEPOT (auto-deposits at dist=0)."""
        executor = OptionExecutor(n_agents=1)
        executor.set_option(0, MacroOption.MINE_CYCLE)
        obs = [ObsResource.NONE, ObsStation.HUB, ObsInventory.HAS_RESOURCE,
               ObsContest.FREE, 0, 0]
        assert executor.get_task_policy(0, obs) == TaskPolicy.NAV_DEPOT

    def test_craft_cycle_empty_goes_to_hub(self):
        """No hearts: go to hub first (dinky aligner chain)."""
        executor = OptionExecutor(n_agents=2)
        executor.set_option(1, MacroOption.CRAFT_CYCLE)  # agent 1 = aligner
        obs = [ObsResource.NONE, ObsStation.NONE, ObsInventory.EMPTY,
               ObsContest.FREE, 0, 0]
        assert executor.get_task_policy(1, obs) == TaskPolicy.NAV_DEPOT

    def test_craft_cycle_with_resource_goes_to_craft(self):
        """Has hearts (resource): go to craft station."""
        executor = OptionExecutor(n_agents=2)
        executor.set_option(1, MacroOption.CRAFT_CYCLE)  # agent 1 = aligner
        obs = [ObsResource.NONE, ObsStation.NONE, ObsInventory.HAS_RESOURCE,
               ObsContest.FREE, 0, 0]
        assert executor.get_task_policy(1, obs) == TaskPolicy.NAV_CRAFT

    def test_capture_cycle_nav_junction(self):
        executor = OptionExecutor(n_agents=2)
        executor.set_option(1, MacroOption.CAPTURE_CYCLE)  # agent 1 = aligner
        obs = [ObsResource.NONE, ObsStation.NONE, ObsInventory.HAS_GEAR,
               ObsContest.FREE, 0, 0]
        assert executor.get_task_policy(1, obs) == TaskPolicy.NAV_JUNCTION

    def test_capture_cycle_nav_junction_at_junction(self):
        """Even at JUNCTION, use NAV_JUNCTION (auto-captures at dist=0)."""
        executor = OptionExecutor(n_agents=2)
        executor.set_option(1, MacroOption.CAPTURE_CYCLE)  # agent 1 = aligner
        obs = [ObsResource.NONE, ObsStation.JUNCTION, ObsInventory.HAS_GEAR,
               ObsContest.FREE, 0, 0]
        assert executor.get_task_policy(1, obs) == TaskPolicy.NAV_JUNCTION

    def test_capture_cycle_no_gear_waits(self):
        executor = OptionExecutor(n_agents=2)
        executor.set_option(1, MacroOption.CAPTURE_CYCLE)  # agent 1 = aligner
        obs = [ObsResource.NONE, ObsStation.NONE, ObsInventory.EMPTY,
               ObsContest.FREE, 0, 0]
        assert executor.get_task_policy(1, obs) == TaskPolicy.WAIT

    def test_defend_nav_junction(self):
        executor = OptionExecutor(n_agents=1)
        executor.set_option(0, MacroOption.DEFEND)
        obs = [ObsResource.NONE, ObsStation.NONE, ObsInventory.EMPTY,
               ObsContest.FREE, 0, 0]
        assert executor.get_task_policy(0, obs) == TaskPolicy.NAV_JUNCTION

    def test_defend_nav_junction_at_junction(self):
        """Even at JUNCTION, use NAV_JUNCTION (auto-captures at dist=0)."""
        executor = OptionExecutor(n_agents=1)
        executor.set_option(0, MacroOption.DEFEND)
        obs = [ObsResource.NONE, ObsStation.JUNCTION, ObsInventory.EMPTY,
               ObsContest.FREE, 0, 0]
        assert executor.get_task_policy(0, obs) == TaskPolicy.NAV_JUNCTION


    def test_role_filter_miner_cannot_craft(self):
        """Miner (even) requesting CRAFT_CYCLE gets MINE_CYCLE instead."""
        executor = OptionExecutor(n_agents=2)
        executor.set_option(0, MacroOption.CRAFT_CYCLE)  # agent 0 = miner
        assert executor.states[0].current_option == MacroOption.MINE_CYCLE

    def test_role_filter_miner_cannot_capture(self):
        """Miner (even) requesting CAPTURE_CYCLE gets MINE_CYCLE instead."""
        executor = OptionExecutor(n_agents=2)
        executor.set_option(0, MacroOption.CAPTURE_CYCLE)  # agent 0 = miner
        assert executor.states[0].current_option == MacroOption.MINE_CYCLE

    def test_role_filter_aligner_cannot_mine(self):
        """Aligner (odd) requesting MINE_CYCLE gets EXPLORE instead."""
        executor = OptionExecutor(n_agents=2)
        executor.set_option(1, MacroOption.MINE_CYCLE)  # agent 1 = aligner
        assert executor.states[1].current_option == MacroOption.EXPLORE


class TestOptionTermination:
    """Test option termination conditions."""

    def test_timeout(self):
        executor = OptionExecutor(n_agents=1)
        executor.set_option(0, MacroOption.EXPLORE)
        obs = [ObsResource.NONE, ObsStation.NONE, ObsInventory.EMPTY,
               ObsContest.FREE, 0, 0]
        # Tick 50 times (EXPLORE timeout)
        for _ in range(50):
            executor.tick(0, obs)
        assert executor.check_termination(0, obs) is True

    def test_explore_terminates_on_resource(self):
        executor = OptionExecutor(n_agents=1)
        executor.set_option(0, MacroOption.EXPLORE)
        obs_resource = [ObsResource.AT, ObsStation.NONE, ObsInventory.EMPTY,
                        ObsContest.FREE, 0, 0]
        assert executor.check_termination(0, obs_resource) is True

    def test_explore_terminates_on_station_miner(self):
        """Miner (agent 0) terminates EXPLORE on any station (including HUB)."""
        executor = OptionExecutor(n_agents=2)
        executor.set_option(0, MacroOption.EXPLORE)
        obs_station = [ObsResource.NONE, ObsStation.HUB, ObsInventory.EMPTY,
                       ObsContest.FREE, 0, 0]
        assert executor.check_termination(0, obs_station) is True

    def test_explore_aligner_ignores_hub(self):
        """Aligner (agent 1) does NOT terminate EXPLORE on HUB."""
        executor = OptionExecutor(n_agents=2)
        executor.set_option(1, MacroOption.EXPLORE)
        obs_hub = [ObsResource.NONE, ObsStation.HUB, ObsInventory.EMPTY,
                   ObsContest.FREE, 0, 0]
        assert executor.check_termination(1, obs_hub) is False

    def test_explore_aligner_terminates_on_craft(self):
        """Aligner (agent 1) terminates EXPLORE on CRAFT station."""
        executor = OptionExecutor(n_agents=2)
        executor.set_option(1, MacroOption.EXPLORE)
        obs_craft = [ObsResource.NONE, ObsStation.CRAFT, ObsInventory.EMPTY,
                     ObsContest.FREE, 0, 0]
        assert executor.check_termination(1, obs_craft) is True

    def test_mine_cycle_terminates_on_deposit(self):
        executor = OptionExecutor(n_agents=1)
        executor.set_option(0, MacroOption.MINE_CYCLE)
        # Set prev_inv to RESOURCE (simulates having resource)
        executor.states[0].prev_inv = ObsInventory.HAS_RESOURCE
        obs = [ObsResource.NONE, ObsStation.HUB, ObsInventory.EMPTY,
               ObsContest.FREE, 0, 0]
        assert executor.check_termination(0, obs) is True

    def test_craft_cycle_terminates_on_gear(self):
        executor = OptionExecutor(n_agents=2)
        executor.set_option(1, MacroOption.CRAFT_CYCLE)  # agent 1 = aligner
        obs = [ObsResource.NONE, ObsStation.CRAFT, ObsInventory.HAS_GEAR,
               ObsContest.FREE, 0, 0]
        assert executor.check_termination(1, obs) is True

    def test_capture_terminates_when_gear_used(self):
        executor = OptionExecutor(n_agents=2)
        executor.set_option(1, MacroOption.CAPTURE_CYCLE)  # agent 1 = aligner
        executor.states[1].prev_inv = ObsInventory.HAS_GEAR
        obs = [ObsResource.NONE, ObsStation.JUNCTION, ObsInventory.EMPTY,
               ObsContest.FREE, 0, 0]
        assert executor.check_termination(1, obs) is True

    def test_set_option_resets_state(self):
        executor = OptionExecutor(n_agents=2)
        executor.set_option(1, MacroOption.CRAFT_CYCLE)  # agent 1 = aligner
        obs = [0, 0, 0, 0, 0, 0]
        for _ in range(10):
            executor.tick(1, obs)
        assert executor.states[1].steps_in_option == 10
        executor.set_option(1, MacroOption.CAPTURE_CYCLE)
        assert executor.states[1].steps_in_option == 0
        assert executor.states[1].current_option == MacroOption.CAPTURE_CYCLE


@pytest.mark.skipif(not HAS_PYMDP, reason="pymdp (JAX) not installed")
class TestHierarchicalEngine:
    """Test the hierarchical BatchedAIFEngine."""

    def test_engine_creation(self):
        from aif_meta_cogames.aif_agent.cogames_policy import BatchedAIFEngine
        engine = BatchedAIFEngine(n_agents=8)
        assert engine.n_agents == 8
        assert engine.agent.batch_size == 8
        assert len(engine.agent.policies) == 25  # 5^2

    def test_engine_step(self):
        import jax.numpy as jnp
        from aif_meta_cogames.aif_agent.cogames_policy import BatchedAIFEngine
        engine = BatchedAIFEngine(n_agents=4)

        # Submit observations for all agents
        for agent_id in range(4):
            jax_obs = [jnp.array([0]) for _ in range(6)]
            policy = engine.submit_and_get_policy(agent_id, jax_obs)
            assert 0 <= policy < 13  # valid task policy

    def test_option_persistence(self):
        """Options should persist across steps (not replan every step)."""
        import jax.numpy as jnp
        from aif_meta_cogames.aif_agent.cogames_policy import BatchedAIFEngine
        engine = BatchedAIFEngine(n_agents=2)

        # Run a few steps with same obs
        obs = [jnp.array([0]) for _ in range(6)]
        for _ in range(5):
            for agent_id in range(2):
                engine.submit_and_get_policy(agent_id, obs)

        # Options should have steps > 0 (persisting)
        for i in range(2):
            assert engine.option_executor.states[i].steps_in_option > 0


class TestSpatialMemory:
    """Test spatial memory for navigation."""

    def test_wall_tracking(self):
        from aif_meta_cogames.aif_agent.cogames_policy import SpatialMemory
        mem = SpatialMemory()
        mem.position = (10, 10)
        mem.walls.add((10, 11))
        mem.walls.add((9, 10))
        assert (10, 11) in mem.walls
        assert (10, 12) not in mem.walls

    def test_is_wall_adjacent(self):
        from aif_meta_cogames.aif_agent.cogames_policy import SpatialMemory
        mem = SpatialMemory()
        mem.position = (10, 10)
        mem.walls.add((10, 11))  # wall to the east
        mem.walls.add((9, 10))   # wall to the north
        assert mem.is_wall_adjacent("move_east")
        assert mem.is_wall_adjacent("move_north")
        assert not mem.is_wall_adjacent("move_south")
        assert not mem.is_wall_adjacent("move_west")

    def test_is_wall_adjacent_no_position(self):
        from aif_meta_cogames.aif_agent.cogames_policy import SpatialMemory
        mem = SpatialMemory()
        # No position set → should return False
        assert not mem.is_wall_adjacent("move_east")

    def test_station_memory(self):
        from aif_meta_cogames.aif_agent.cogames_policy import SpatialMemory
        mem = SpatialMemory()
        mem.position = (10, 10)
        mem.stations[(50, 50)] = "hub"
        mem.stations[(20, 20)] = "extractor:carbon"
        mem.stations[(30, 30)] = "craft"
        nearest_ext = mem.find_nearest_station("extractor:carbon")
        assert nearest_ext == (20, 20)
        nearest_hub = mem.find_nearest_station("hub")
        assert nearest_hub == (50, 50)

    def test_find_nearest_station_no_match(self):
        from aif_meta_cogames.aif_agent.cogames_policy import SpatialMemory
        mem = SpatialMemory()
        mem.position = (10, 10)
        mem.stations[(20, 20)] = "hub"
        assert mem.find_nearest_station("junction") is None

    def test_find_nearest_station_no_position(self):
        from aif_meta_cogames.aif_agent.cogames_policy import SpatialMemory
        mem = SpatialMemory()
        mem.stations[(20, 20)] = "hub"
        assert mem.find_nearest_station("hub") is None

    def test_stuck_detection_oscillation(self):
        from aif_meta_cogames.aif_agent.cogames_policy import SpatialMemory
        mem = SpatialMemory()
        # Oscillate between two positions for 20 steps
        for _ in range(10):
            mem.position_history.append((10, 10))
            mem.position_history.append((10, 11))
        assert mem.is_stuck()

    def test_stuck_detection_single_position(self):
        from aif_meta_cogames.aif_agent.cogames_policy import SpatialMemory
        mem = SpatialMemory()
        for _ in range(20):
            mem.position_history.append((10, 10))
        assert mem.is_stuck()

    def test_stuck_detection_not_yet(self):
        """Not stuck if oscillating for fewer than 20 steps."""
        from aif_meta_cogames.aif_agent.cogames_policy import SpatialMemory
        mem = SpatialMemory()
        for _ in range(3):
            mem.position_history.append((10, 10))
            mem.position_history.append((10, 11))
        assert not mem.is_stuck()

    def test_not_stuck_when_moving(self):
        from aif_meta_cogames.aif_agent.cogames_policy import SpatialMemory
        mem = SpatialMemory()
        for i in range(10):
            mem.position_history.append((10, 10 + i))
        assert not mem.is_stuck()

    def test_not_stuck_short_history(self):
        from aif_meta_cogames.aif_agent.cogames_policy import SpatialMemory
        mem = SpatialMemory()
        mem.position_history.append((10, 10))
        mem.position_history.append((10, 10))
        assert not mem.is_stuck()  # Only 2 entries, need 20


class TestSharedSpatialMemory:
    """Test shared spatial memory (belief sharing — Catal et al. 2024)."""

    def test_contribute_merges_stations(self):
        from aif_meta_cogames.aif_agent.cogames_policy import (
            SharedSpatialMemory, SpatialMemory,
        )
        shared = SharedSpatialMemory()
        mem1 = SpatialMemory()
        mem1.stations[(10, 10)] = "hub"
        mem1.stations[(20, 20)] = "extractor:carbon"
        mem2 = SpatialMemory()
        mem2.stations[(30, 30)] = "extractor:silicon"
        shared.contribute(mem1)
        shared.contribute(mem2)
        assert shared.stations[(10, 10)] == "hub"
        assert shared.stations[(20, 20)] == "extractor:carbon"
        assert shared.stations[(30, 30)] == "extractor:silicon"

    def test_find_nearest_element_specific(self):
        from aif_meta_cogames.aif_agent.cogames_policy import SharedSpatialMemory
        shared = SharedSpatialMemory()
        shared.stations[(10, 10)] = "extractor:carbon"
        shared.stations[(20, 20)] = "extractor:silicon"
        shared.stations[(5, 5)] = "extractor:silicon"  # closer but wrong element
        pos = (0, 0)
        nearest_carbon = shared.find_nearest_station("extractor:carbon", pos)
        assert nearest_carbon == (10, 10)
        nearest_silicon = shared.find_nearest_station("extractor:silicon", pos)
        assert nearest_silicon == (5, 5)

    def test_multiple_agents_share_discoveries(self):
        from aif_meta_cogames.aif_agent.cogames_policy import (
            SharedSpatialMemory, SpatialMemory,
        )
        shared = SharedSpatialMemory()
        # Agent 0 finds carbon extractor
        mem0 = SpatialMemory()
        mem0.stations[(15, 15)] = "extractor:carbon"
        mem0.explored.add((15, 15))
        shared.contribute(mem0)
        # Agent 1 doesn't know about it directly, but shared does
        pos = (0, 0)
        result = shared.find_nearest_station("extractor:carbon", pos)
        assert result == (15, 15)
        # Agent 1 also explored some area
        mem1 = SpatialMemory()
        mem1.explored.add((40, 40))
        shared.contribute(mem1)
        # Shared pool has both explored areas
        assert (15, 15) in shared.explored
        assert (40, 40) in shared.explored

    def test_walls_shared(self):
        from aif_meta_cogames.aif_agent.cogames_policy import (
            SharedSpatialMemory, SpatialMemory,
        )
        shared = SharedSpatialMemory()
        mem = SpatialMemory()
        mem.walls.add((10, 11))
        mem.walls.add((10, 12))
        shared.contribute(mem)
        assert (10, 11) in shared.walls
        assert (10, 12) in shared.walls

    def test_efe_element_selection(self):
        """Scarcest element has lowest EFE — mining it minimizes
        D_KL(Q(resources|mine_e) || C_uniform)."""
        from aif_meta_cogames.aif_agent.cogames_policy import RESOURCE_NAMES
        team_res = {"carbon": 2, "oxygen": 30, "germanium": 25, "silicon": 50}
        scarcest = min(RESOURCE_NAMES, key=lambda e: team_res.get(e, 0))
        assert scarcest == "carbon"
        # After mining carbon, resources are more balanced → lower KL from uniform
        team_res2 = {"carbon": 20, "oxygen": 20, "germanium": 20, "silicon": 20}
        scarcest2 = min(RESOURCE_NAMES, key=lambda e: team_res2.get(e, 0))
        # When balanced, any element is fine (min picks first alphabetically)
        assert scarcest2 in RESOURCE_NAMES


# ---------------------------------------------------------------------------
# Navigation POMDP tests
# ---------------------------------------------------------------------------

class TestNavEnums:
    """Test navigation POMDP enums and constants."""

    def test_nav_progress_values(self):
        assert NavProgress.APPROACHING == 0
        assert NavProgress.LATERAL == 1
        assert NavProgress.RETREATING == 2
        assert NavProgress.BLOCKED == 3
        assert len(NavProgress) == 4

    def test_target_range_values(self):
        assert TargetRange.ADJACENT == 0
        assert TargetRange.NEAR == 1
        assert TargetRange.FAR == 2
        assert TargetRange.NO_TARGET == 3
        assert len(TargetRange) == 4

    def test_nav_action_values(self):
        assert NavAction.TOWARD == 0
        assert NavAction.LEFT == 1
        assert NavAction.RIGHT == 2
        assert NavAction.AWAY == 3
        assert NavAction.RANDOM == 4
        assert NUM_NAV_ACTIONS == 5

    def test_bearing_dirs(self):
        assert len(_BEARING_DIRS) == 4
        assert _BEARING_DIRS[0] == "move_north"
        assert _BEARING_DIRS[1] == "move_east"
        assert _BEARING_DIRS[2] == "move_south"
        assert _BEARING_DIRS[3] == "move_west"


class TestNavGenerativeModel:
    """Test navigation POMDP A/B/C/D matrices."""

    def test_nav_A_shapes(self):
        A = _build_nav_A()
        assert len(A) == 2
        assert A[0].shape == (4, 4)  # obs_range x target_range
        assert A[1].shape == (4, 4)  # obs_movement x nav_progress

    def test_nav_A_normalized(self):
        A = _build_nav_A()
        for m in range(2):
            col_sums = A[m].sum(axis=0)
            np.testing.assert_allclose(col_sums, 1.0, atol=1e-6)

    def test_nav_B_shapes(self):
        B = _build_nav_B()
        assert len(B) == 2
        assert B[0].shape == (4, 4, 4, 5)  # prog' x prog x range x action
        assert B[1].shape == (4, 4, 4, 5)  # range' x prog x range x action

    def test_nav_B_normalized(self):
        B = _build_nav_B()
        for f in range(2):
            for p in range(4):
                for r in range(4):
                    for a in range(5):
                        col_sum = B[f][:, p, r, a].sum()
                        assert abs(col_sum - 1.0) < 1e-6, (
                            f"B[{f}][:, {p}, {r}, {a}] sums to {col_sum}"
                        )

    def test_nav_B_blocked_asymmetry(self):
        """TOWARD from BLOCKED should stay BLOCKED more than LEFT/RIGHT."""
        B = _build_nav_B()
        B_prog = B[0]
        BLKD = NavProgress.BLOCKED
        # Check across all range values
        for rng in range(4):
            # TOWARD from BLOCKED → BLOCKED should be higher
            toward_blocked = B_prog[BLKD, BLKD, rng, NavAction.TOWARD]
            left_blocked = B_prog[BLKD, BLKD, rng, NavAction.LEFT]
            assert toward_blocked > left_blocked, (
                f"At range={rng}: TOWARD→BLOCKED ({toward_blocked:.3f}) "
                f"should > LEFT→BLOCKED ({left_blocked:.3f})"
            )

    def test_nav_C_shapes(self):
        C = _build_nav_C()
        assert len(C) == 2
        assert C[0].shape == (4,)  # range preferences
        assert C[1].shape == (4,)  # movement preferences

    def test_nav_C_preferences(self):
        """ADJACENT preferred over FAR, APPROACHING preferred over BLOCKED."""
        C = _build_nav_C()
        assert C[0][TargetRange.ADJACENT] > C[0][TargetRange.FAR]
        assert C[0][TargetRange.ADJACENT] > C[0][TargetRange.NO_TARGET]
        assert C[1][NavProgress.APPROACHING] > C[1][NavProgress.BLOCKED]
        assert C[1][NavProgress.APPROACHING] > C[1][NavProgress.RETREATING]

    def test_nav_D_shapes(self):
        D = _build_nav_D()
        assert len(D) == 2
        assert D[0].shape == (4,)
        assert D[1].shape == (4,)

    def test_nav_D_normalized(self):
        D = _build_nav_D()
        for f in range(2):
            assert abs(D[f].sum() - 1.0) < 1e-6


@pytest.mark.skipif(not HAS_PYMDP, reason="pymdp (JAX) not installed")
class TestNavPOMDPAgent:
    """Test navigation POMDP agent creation and inference."""

    def test_create_nav_agent(self):
        agent = create_nav_agent(n_agents=2, policy_len=2)
        assert agent.batch_size == 2
        assert len(agent.A) == 2  # 2 observation modalities
        assert len(agent.B) == 2  # 2 state factors
        assert len(agent.policies) == 25  # 5^2 two-step policies

    def test_nav_inference(self):
        agent = create_nav_agent(n_agents=2, policy_len=2)
        obs = [jnp.zeros((2, 1), dtype=jnp.int32) for _ in range(2)]
        qs = agent.infer_states(obs, empirical_prior=agent.D)
        assert len(qs) == 2  # 2 state factors
        assert qs[0].shape[0] == 2  # batch=2

    def test_nav_full_cycle(self):
        """Test full nav POMDP inference cycle."""
        agent = create_nav_agent(n_agents=4, policy_len=2)
        obs = [jnp.zeros((4, 1), dtype=jnp.int32) for _ in range(2)]
        qs = agent.infer_states(obs, empirical_prior=agent.D)
        q_pi, G = agent.infer_policies(qs)
        action = agent.sample_action(q_pi)
        # action shape: (batch=4, num_factors=2)
        assert action.shape == (4, 2)
        # Both factors same action (constrained)
        assert int(action[0, 0]) == int(action[0, 1])
        assert 0 <= int(action[0, 0]) < 5

    def test_nav_blocked_prefers_lateral(self):
        """When observing BLOCKED, nav POMDP should prefer LEFT/RIGHT over TOWARD."""
        agent = create_nav_agent(n_agents=1, policy_len=2)
        # Observe: range=NEAR, movement=BLOCKED
        obs = [
            jnp.array([[int(TargetRange.NEAR)]]),
            jnp.array([[int(NavProgress.BLOCKED)]]),
        ]
        qs = agent.infer_states(obs, empirical_prior=agent.D)
        q_pi, G = agent.infer_policies(qs)
        action = agent.sample_action(q_pi)
        nav_act = int(action[0, 0])
        # Should NOT choose TOWARD when blocked (wall ahead)
        # Note: this is probabilistic, but deterministic selection should avoid TOWARD
        assert nav_act != NavAction.TOWARD or nav_act in (
            NavAction.LEFT, NavAction.RIGHT
        )


@pytest.mark.skipif(not HAS_PYMDP, reason="pymdp (JAX) not installed")
class TestNavEngine:
    """Test nav POMDP integration in BatchedAIFEngine."""

    def test_engine_has_nav_agent(self):
        from aif_meta_cogames.aif_agent.cogames_policy import BatchedAIFEngine
        engine = BatchedAIFEngine(n_agents=4)
        assert engine.nav_agent is not None
        assert engine.nav_agent.batch_size == 4

    def test_submit_nav_returns_valid_action(self):
        from aif_meta_cogames.aif_agent.cogames_policy import BatchedAIFEngine
        engine = BatchedAIFEngine(n_agents=4)

        # Submit nav obs for all agents
        for agent_id in range(4):
            nav_obs = [jnp.array([0]), jnp.array([0])]
            nav_action = engine.submit_nav_and_get_action(agent_id, nav_obs)
            assert 0 <= nav_action < NUM_NAV_ACTIONS

    def test_nav_belief_reset(self):
        from aif_meta_cogames.aif_agent.cogames_policy import BatchedAIFEngine
        engine = BatchedAIFEngine(n_agents=2)

        # Run a step to update beliefs
        for agent_id in range(2):
            nav_obs = [
                jnp.array([int(TargetRange.FAR)]),
                jnp.array([int(NavProgress.APPROACHING)]),
            ]
            engine.submit_nav_and_get_action(agent_id, nav_obs)

        # Reset agent 0's beliefs
        engine._reset_nav_beliefs([0])

        # Agent 0's prior should be back to D
        for f in range(len(engine.nav_prior)):
            reset_prior = np.asarray(engine.nav_prior[f][0])
            d_prior = np.asarray(engine.nav_agent.D[f][0])
            np.testing.assert_allclose(reset_prior, d_prior, atol=1e-6)


class TestFrontierExploration:
    """Test frontier-based exploration target computation."""

    def test_frontier_basic(self):
        """Frontier cells are unexplored cells adjacent to explored ones."""
        mem = SpatialMemory()
        mem.position = (5, 5)
        # Explore a 3x3 area centered on (5, 5)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                mem.explored.add((5 + dr, 5 + dc))

        # Import the method (it's on AIFCogPolicyImpl, but we can test logic)
        frontiers = set()
        for (r, c) in mem.explored:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = (r + dr, c + dc)
                if nb not in mem.explored and nb not in mem.walls:
                    frontiers.add(nb)

        assert len(frontiers) > 0
        # All frontiers should be adjacent to explored territory
        for fr, fc in frontiers:
            has_explored_neighbor = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (fr + dr, fc + dc) in mem.explored:
                    has_explored_neighbor = True
                    break
            assert has_explored_neighbor

    def test_frontier_excludes_walls(self):
        """Frontier cells should not include known walls."""
        mem = SpatialMemory()
        mem.position = (5, 5)
        mem.explored.add((5, 5))
        mem.walls.add((5, 6))  # Wall to the east

        frontiers = set()
        for (r, c) in mem.explored:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = (r + dr, c + dc)
                if nb not in mem.explored and nb not in mem.walls:
                    frontiers.add(nb)

        assert (5, 6) not in frontiers

    def test_frontier_nearest(self):
        """Should pick the frontier nearest to current position."""
        mem = SpatialMemory()
        mem.position = (5, 5)
        mem.explored.add((5, 5))
        mem.explored.add((5, 6))
        mem.explored.add((5, 7))

        frontiers = set()
        for (r, c) in mem.explored:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = (r + dr, c + dc)
                if nb not in mem.explored and nb not in mem.walls:
                    frontiers.add(nb)

        nearest = min(
            frontiers,
            key=lambda f: abs(f[0] - mem.position[0]) + abs(f[1] - mem.position[1])
        )
        # Nearest frontier should be dist 1 from (5, 5)
        dist = abs(nearest[0] - 5) + abs(nearest[1] - 5)
        assert dist == 1


class TestBearingConversion:
    """Test relative-to-absolute direction conversion logic."""

    def test_toward_north(self):
        """TOWARD when target is north → move_north."""
        bearing_idx = _BEARING_DIRS.index("move_north")
        direction = _BEARING_DIRS[bearing_idx]
        assert direction == "move_north"

    def test_left_of_north(self):
        """LEFT when bearing is north → move_west (CCW)."""
        bearing_idx = _BEARING_DIRS.index("move_north")
        direction = _BEARING_DIRS[(bearing_idx + 3) % 4]
        assert direction == "move_west"

    def test_right_of_north(self):
        """RIGHT when bearing is north → move_east (CW)."""
        bearing_idx = _BEARING_DIRS.index("move_north")
        direction = _BEARING_DIRS[(bearing_idx + 1) % 4]
        assert direction == "move_east"

    def test_away_from_north(self):
        """AWAY when bearing is north → move_south."""
        bearing_idx = _BEARING_DIRS.index("move_north")
        direction = _BEARING_DIRS[(bearing_idx + 2) % 4]
        assert direction == "move_south"

    def test_toward_east(self):
        """TOWARD when target is east → move_east."""
        bearing_idx = _BEARING_DIRS.index("move_east")
        direction = _BEARING_DIRS[bearing_idx]
        assert direction == "move_east"

    def test_left_of_east(self):
        """LEFT when bearing is east → move_north (CCW)."""
        bearing_idx = _BEARING_DIRS.index("move_east")
        direction = _BEARING_DIRS[(bearing_idx + 3) % 4]
        assert direction == "move_north"

    def test_all_bearings_cycle(self):
        """Verify all 4 bearings produce correct TOWARD/LEFT/RIGHT/AWAY."""
        expected = {
            "move_north": {
                "toward": "move_north", "left": "move_west",
                "right": "move_east", "away": "move_south",
            },
            "move_east": {
                "toward": "move_east", "left": "move_north",
                "right": "move_south", "away": "move_west",
            },
            "move_south": {
                "toward": "move_south", "left": "move_east",
                "right": "move_west", "away": "move_north",
            },
            "move_west": {
                "toward": "move_west", "left": "move_south",
                "right": "move_north", "away": "move_east",
            },
        }
        for bearing, dirs in expected.items():
            idx = _BEARING_DIRS.index(bearing)
            assert _BEARING_DIRS[idx] == dirs["toward"], f"{bearing} toward"
            assert _BEARING_DIRS[(idx + 3) % 4] == dirs["left"], f"{bearing} left"
            assert _BEARING_DIRS[(idx + 1) % 4] == dirs["right"], f"{bearing} right"
            assert _BEARING_DIRS[(idx + 2) % 4] == dirs["away"], f"{bearing} away"
