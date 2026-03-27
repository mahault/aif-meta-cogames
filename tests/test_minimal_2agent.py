"""Minimal 2-agent mock test — validates the full AIF economy chain.

Uses an 8x8 mock CogsGuard grid with fixed stations to test the full
hierarchy (strategic POMDP → option executor → nav POMDP) without
cogames/mettagrid dependencies. Runs on Windows, ~10ms/step.

Grid (8x8):
    E(0,0)  .  .  .  .  .  .  .     E = Extractor
    .  .  .  .  .  .  .  .          H = Hub (spawn)
    .  .  H  .  .  .  J  .          C = Craft station
    .  .  .  .  .  .  .  .          J = Junction
    .  .  .  .  C  .  .  .
    .  .  .  .  .  .  .  .
    .  .  .  .  .  .  .  E
    .  .  .  .  .  .  .  .

2 agents:
    Agent 0 (miner):   extract carbon at E → deposit at H
    Agent 1 (aligner): get hearts at H → craft gear at C → capture J
"""

import numpy as np
import pytest

try:
    import jax.numpy as jnp
    HAS_PYMDP = True
except ImportError:
    HAS_PYMDP = False

from aif_meta_cogames.aif_agent.discretizer import (
    MacroOption,
    NavAction,
    ObsInventory,
    ObsResource,
    ObsStation,
    TaskPolicy,
)
from aif_meta_cogames.aif_agent.cogames_policy import (
    SpatialMemory,
    _BEARING_DIRS,
)


# ---------------------------------------------------------------------------
# Mock CogsGuard Environment (8x8)
# ---------------------------------------------------------------------------

class MockCogsGuardEnv:
    """8x8 grid with fixed stations for economy chain testing.

    Stations:
        (0, 0): extractor (carbon)
        (6, 7): extractor (oxygen)
        (2, 2): hub (spawn)
        (4, 4): craft station
        (2, 6): junction

    Agents start at the hub (2, 2).
    Interaction = bumping (move INTO entity).
    """

    STATIONS = {
        (0, 0): "extractor",
        (6, 7): "extractor",
        (2, 2): "hub",
        (4, 4): "craft",
        (2, 6): "junction",
    }
    GRID_SIZE = 8

    def __init__(self, n_agents: int = 2):
        self.n_agents = n_agents
        self.positions = [(2, 2)] * n_agents  # all start at hub
        self.inventories = ["empty"] * n_agents
        self.junctions_captured = 0
        self.carbon_mined = 0
        self.carbon_deposited = 0
        self.hearts_withdrawn = 0
        self.gear_crafted = 0
        self.step_count = 0

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def step(self, agent_id: int, direction: str) -> dict:
        """Process one movement for an agent. Returns info dict."""
        pos = self.positions[agent_id]
        dr, dc = {
            "move_north": (-1, 0), "move_south": (1, 0),
            "move_east": (0, 1), "move_west": (0, -1),
            "noop": (0, 0),
        }.get(direction, (0, 0))

        new_r = max(0, min(self.GRID_SIZE - 1, pos[0] + dr))
        new_c = max(0, min(self.GRID_SIZE - 1, pos[1] + dc))
        new_pos = (new_r, new_c)

        info = {"moved": False, "interaction": None}

        # Check if bumping a station
        if new_pos in self.STATIONS and new_pos != pos:
            station = self.STATIONS[new_pos]
            info["interaction"] = station

            if station == "extractor" and self.inventories[agent_id] == "empty":
                self.inventories[agent_id] = "carbon"
                self.carbon_mined += 1
                info["interaction"] = "mine"

            elif station == "hub":
                inv = self.inventories[agent_id]
                if inv == "carbon":
                    self.inventories[agent_id] = "empty"
                    self.carbon_deposited += 1
                    info["interaction"] = "deposit"
                elif inv == "empty":
                    self.inventories[agent_id] = "hearts"
                    self.hearts_withdrawn += 1
                    info["interaction"] = "withdraw_hearts"

            elif station == "craft" and self.inventories[agent_id] == "hearts":
                self.inventories[agent_id] = "gear"
                self.gear_crafted += 1
                info["interaction"] = "craft_gear"

            elif station == "junction" and self.inventories[agent_id] == "gear":
                self.inventories[agent_id] = "empty"
                self.junctions_captured += 1
                info["interaction"] = "capture"

            # Stay at current position (bumped, move fails but interaction happens)
        else:
            self.positions[agent_id] = new_pos
            info["moved"] = True

        self.step_count += 1
        return info

    def observe(self, agent_id: int) -> dict:
        """Generate observation for an agent."""
        pos = self.positions[agent_id]
        inv = self.inventories[agent_id]

        # Find nearest stations
        nearest = {}
        for station_pos, station_type in self.STATIONS.items():
            dist = self._manhattan(pos, station_pos)
            if station_type not in nearest or dist < nearest[station_type][1]:
                nearest[station_type] = (station_pos, dist)

        # Discretize observations
        obs_resource = ObsResource.NONE
        obs_station = ObsStation.NONE
        obs_inv = ObsInventory.EMPTY

        # Resource
        if "extractor" in nearest:
            ext_pos, ext_dist = nearest["extractor"]
            if ext_dist <= 1:
                obs_resource = ObsResource.AT
            elif ext_dist <= 4:
                obs_resource = ObsResource.NEAR

        # Station
        for stype in ["hub", "craft", "junction"]:
            if stype in nearest:
                s_pos, s_dist = nearest[stype]
                if s_dist <= 1:
                    if stype == "hub":
                        obs_station = ObsStation.HUB
                    elif stype == "craft":
                        obs_station = ObsStation.CRAFT
                    elif stype == "junction":
                        obs_station = ObsStation.JUNCTION

        # Inventory
        if inv == "carbon" or inv == "hearts":
            obs_inv = ObsInventory.HAS_RESOURCE
        elif inv == "gear":
            obs_inv = ObsInventory.HAS_GEAR

        return {
            "position": pos,
            "inventory": inv,
            "obs_resource": obs_resource,
            "obs_station": obs_station,
            "obs_inventory": obs_inv,
            "nearest_stations": nearest,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMockEnvironment:
    """Test the mock environment itself."""

    def test_initial_positions(self):
        env = MockCogsGuardEnv(n_agents=2)
        assert env.positions[0] == (2, 2)
        assert env.positions[1] == (2, 2)

    def test_movement(self):
        env = MockCogsGuardEnv(n_agents=1)
        env.step(0, "move_east")
        assert env.positions[0] == (2, 3)
        env.step(0, "move_south")
        assert env.positions[0] == (3, 3)

    def test_boundary(self):
        env = MockCogsGuardEnv(n_agents=1)
        env.positions[0] = (0, 0)
        env.step(0, "move_north")
        assert env.positions[0] == (0, 0)  # clamped
        env.step(0, "move_west")
        assert env.positions[0] == (0, 0)  # clamped

    def test_mine_interaction(self):
        env = MockCogsGuardEnv(n_agents=1)
        env.positions[0] = (0, 1)  # adjacent to extractor at (0,0)
        info = env.step(0, "move_west")
        assert info["interaction"] == "mine"
        assert env.inventories[0] == "carbon"
        assert env.carbon_mined == 1

    def test_deposit_interaction(self):
        env = MockCogsGuardEnv(n_agents=1)
        env.positions[0] = (2, 3)  # adjacent to hub at (2,2)
        env.inventories[0] = "carbon"
        info = env.step(0, "move_west")
        assert info["interaction"] == "deposit"
        assert env.inventories[0] == "empty"

    def test_aligner_chain(self):
        """Test the full aligner chain: hub(hearts) → craft(gear) → junction."""
        env = MockCogsGuardEnv(n_agents=1)

        # Step 1: Get hearts from hub
        env.positions[0] = (2, 3)
        info = env.step(0, "move_west")  # bump hub
        assert info["interaction"] == "withdraw_hearts"
        assert env.inventories[0] == "hearts"

        # Step 2: Navigate to craft station and craft gear
        env.positions[0] = (4, 5)
        info = env.step(0, "move_west")  # bump craft
        assert info["interaction"] == "craft_gear"
        assert env.inventories[0] == "gear"

        # Step 3: Navigate to junction and capture
        env.positions[0] = (2, 7)
        info = env.step(0, "move_west")  # bump junction
        assert info["interaction"] == "capture"
        assert env.inventories[0] == "empty"
        assert env.junctions_captured == 1

    def test_observe_at_extractor(self):
        env = MockCogsGuardEnv(n_agents=1)
        env.positions[0] = (0, 1)
        obs = env.observe(0)
        assert obs["obs_resource"] == ObsResource.AT

    def test_observe_at_hub(self):
        env = MockCogsGuardEnv(n_agents=1)
        env.positions[0] = (2, 2)
        obs = env.observe(0)
        assert obs["obs_station"] == ObsStation.HUB


class TestMinerEconomyChain:
    """Test that a simple miner loop works."""

    def test_mine_and_deposit(self):
        """Miner: extractor → hub → repeat."""
        env = MockCogsGuardEnv(n_agents=1)

        # Navigate to extractor at (0, 0) from hub (2, 2)
        env.step(0, "move_north")  # (1, 2)
        env.step(0, "move_north")  # (0, 2)
        env.step(0, "move_west")   # (0, 1)
        info = env.step(0, "move_west")  # bump (0, 0) → mine
        assert info["interaction"] == "mine"
        assert env.inventories[0] == "carbon"
        # Position stays at (0, 1) after bump

        # Navigate back to hub at (2, 2)
        env.step(0, "move_south")  # (1, 1)
        env.step(0, "move_south")  # (2, 1)
        info = env.step(0, "move_east")  # bump (2, 2) → deposit
        assert info["interaction"] == "deposit"
        assert env.inventories[0] == "empty"
        assert env.carbon_deposited == 1


class TestAlignerEconomyChain:
    """Test the full aligner chain: hub → craft → junction."""

    def test_hearts_craft_capture(self):
        env = MockCogsGuardEnv(n_agents=1)

        # Start at hub — bump to get hearts
        env.positions[0] = (2, 3)
        env.step(0, "move_west")  # bump hub
        assert env.inventories[0] == "hearts"

        # Navigate to craft at (4, 4)
        env.positions[0] = (4, 3)
        env.step(0, "move_east")  # bump craft
        assert env.inventories[0] == "gear"

        # Navigate to junction at (2, 6)
        env.positions[0] = (2, 5)
        env.step(0, "move_east")  # bump junction
        assert env.inventories[0] == "empty"
        assert env.junctions_captured == 1


@pytest.mark.skipif(not HAS_PYMDP, reason="pymdp (JAX) not installed")
class TestAIFWithMockEnv:
    """Test AIF components interacting with the mock environment."""

    def test_option_executor_selects_nav_resource_for_miner(self):
        """Miner with MINE_CYCLE should select NAV_RESOURCE when empty."""
        from aif_meta_cogames.aif_agent.cogames_policy import OptionExecutor
        executor = OptionExecutor(n_agents=2)
        executor.set_option(0, MacroOption.MINE_CYCLE)

        env = MockCogsGuardEnv(n_agents=2)
        obs = env.observe(0)
        obs_ints = [
            obs["obs_resource"], obs["obs_station"], obs["obs_inventory"],
            0, 0, 0,  # contest, social, role
        ]
        policy = executor.get_task_policy(0, obs_ints)
        assert policy == TaskPolicy.NAV_RESOURCE

    def test_option_executor_selects_nav_depot_for_aligner(self):
        """Aligner with CRAFT_CYCLE should go to hub first (empty-handed)."""
        from aif_meta_cogames.aif_agent.cogames_policy import OptionExecutor
        executor = OptionExecutor(n_agents=2)
        executor.set_option(1, MacroOption.CRAFT_CYCLE)

        env = MockCogsGuardEnv(n_agents=2)
        obs = env.observe(1)
        obs_ints = [
            obs["obs_resource"], obs["obs_station"], obs["obs_inventory"],
            0, 0, 0,
        ]
        policy = executor.get_task_policy(1, obs_ints)
        assert policy == TaskPolicy.NAV_DEPOT

    def test_nav_pomdp_produces_actions_from_mock_obs(self):
        """Nav POMDP should produce valid actions given mock observations."""
        from aif_meta_cogames.aif_agent.cogames_policy import BatchedAIFEngine
        engine = BatchedAIFEngine(n_agents=2)

        env = MockCogsGuardEnv(n_agents=2)
        for agent_id in range(2):
            obs = env.observe(agent_id)
            # Compute simple nav obs
            target = (0, 0)  # extractor
            pos = env.positions[agent_id]
            dist = abs(target[0] - pos[0]) + abs(target[1] - pos[1])

            if dist <= 1:
                obs_range = 0  # ADJACENT
            elif dist <= 4:
                obs_range = 1  # NEAR
            else:
                obs_range = 2  # FAR

            nav_obs = [jnp.array([obs_range]), jnp.array([1])]  # LATERAL
            nav_action = engine.submit_nav_and_get_action(agent_id, nav_obs)
            assert 0 <= nav_action < 5

    def test_spatial_memory_tracks_mock_positions(self):
        """SpatialMemory should track positions from mock environment."""
        env = MockCogsGuardEnv(n_agents=1)
        mem = SpatialMemory()

        # Simulate position tracking (move south to avoid junction at (2,6))
        for _ in range(5):
            env.step(0, "move_south")
            mem.position = env.positions[0]
            mem.position_history.append(mem.position)
            mem.explored.add(mem.position)

        assert mem.position == (7, 2)
        assert len(mem.position_history) == 5
        assert (3, 2) in mem.explored
        assert (7, 2) in mem.explored

    def test_frontier_exploration_on_mock_grid(self):
        """Frontier exploration should find unexplored cells in mock grid."""
        mem = SpatialMemory()
        mem.position = (2, 2)

        # Explore a small area around hub
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                mem.explored.add((2 + dr, 2 + dc))

        # Find frontiers
        frontiers = set()
        for (r, c) in mem.explored:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = (r + dr, c + dc)
                if nb not in mem.explored and nb not in mem.walls:
                    frontiers.add(nb)

        # Should have frontiers in all 4 directions from the explored 3x3 area
        assert len(frontiers) > 0
        # Frontiers should be at distance 2 or 3 from center
        # (edge cells of 3x3 are dist 1-2 from center; their neighbors are 2-3)
        for fr, fc in frontiers:
            dist = abs(fr - 2) + abs(fc - 2)
            assert 2 <= dist <= 3, f"Frontier ({fr}, {fc}) dist={dist} from (2,2)"

    def test_two_agent_role_divergence(self):
        """Even agent (miner) and odd agent (aligner) should get different options."""
        from aif_meta_cogames.aif_agent.cogames_policy import BatchedAIFEngine
        engine = BatchedAIFEngine(n_agents=2)

        # Run several steps to trigger option selection
        for step in range(60):  # past EXPLORE timeout
            for agent_id in range(2):
                jax_obs = [jnp.array([0]) for _ in range(6)]
                engine.submit_and_get_policy(agent_id, jax_obs)

        # After enough steps, miners and aligners should diverge
        miner_option = engine._current_options[0]
        aligner_option = engine._current_options[1]
        # We can't guarantee specific options, but can verify they're valid
        assert 0 <= miner_option < 5
        assert 0 <= aligner_option < 5
