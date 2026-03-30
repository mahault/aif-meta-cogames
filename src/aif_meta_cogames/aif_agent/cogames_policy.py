"""Live CogsGuard policy using deep active inference (two nested POMDPs).

Architecture (deep AIF with nested generative models):
- **Level 2**: Strategic POMDP (288 states, 5 macro-options, replans every
  ~30-80 steps). Decides WHAT to do via Expected Free Energy (EFE).
- **Level 1**: OptionExecutor (reactive state machines) maps option + obs ->
  task policy (~0ms).
- **Level 0**: Navigation POMDP (16 states, 5 relative actions, every step).
  Decides HOW to move toward the target. Uses epistemic value (info gain) for
  principled exploration and 2-step planning for obstacle avoidance.

Both POMDPs use JIT-compiled batched inference via pymdp 1.0 (JAX/Equinox).
Agent 0's step triggers batched inference for all 8 agents; agents 1-7 use
cached results (1-step lag).

Implements the cogames MultiAgentPolicy interface:
    cogames eval -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy

Note: mettagrid has no Windows wheel. This module uses lazy imports so that
``_build_tag_categories`` and ``AIFBeliefState`` can be imported standalone
for testing, while the full policy classes require mettagrid at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .discretizer import (
    Hand,
    MacroOption,
    NavAction,
    NavProgress,
    NUM_NAV_ACTIONS,
    NUM_OBS,
    NUM_OPTIONS,
    NUM_TASK_POLICIES,
    OPTION_NAMES,
    ObsContest,
    ObsInventory,
    ObsResource,
    ObsStation,
    ObservationDiscretizer,
    Phase,
    TASK_POLICY_NAMES,
    TargetRange,
    TaskPolicy,
    state_factors,
)
from .generative_model import CogsGuardPOMDP, _agent_role, create_nav_agent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEAR_NAMES = ("aligner", "scrambler", "miner", "scout")
RESOURCE_NAMES = ("carbon", "oxygen", "germanium", "silicon")
WANDER_DIRECTIONS = ("move_east", "move_south", "move_west", "move_north")
WANDER_STEPS = 15

DIRECTION_DELTAS = {
    "move_north": (-1, 0), "move_south": (1, 0),
    "move_east": (0, 1), "move_west": (0, -1),
}
OPPOSITES = {
    "move_north": "move_south", "move_south": "move_north",
    "move_east": "move_west", "move_west": "move_east",
}
# Clockwise bearing order for relative→absolute nav action conversion
_BEARING_DIRS = ("move_north", "move_east", "move_south", "move_west")


# ---------------------------------------------------------------------------
# Spatial memory (persistent world model for navigation)
# ---------------------------------------------------------------------------

class SpatialMemory:
    """Persistent spatial memory for navigation on partially-observed maps.

    Tracks absolute position (from lp:* observation tokens), walls,
    stations, and explored territory.  Enables navigating toward
    remembered targets when they're outside the 13x13 observation window.

    Positions are stored in **spawn-relative** coordinates (per-agent frame).
    The ``hub_offset`` records the hub's position in this frame, enabling
    conversion to a **hub-relative** shared frame via ``to_shared()`` /
    ``from_shared()``.
    """

    def __init__(self):
        self.position: Optional[tuple[int, int]] = None
        self.walls: set[tuple[int, int]] = set()
        self.stations: dict[tuple[int, int], str] = {}
        self.explored: set[tuple[int, int]] = set()
        self.position_history: list[tuple[int, int]] = []
        # Hub position in this agent's spawn-relative frame
        self.hub_offset: Optional[tuple[int, int]] = None

    def update(self, obs: Any, center: tuple[int, int],
               wall_tag_ids: set[int], station_tag_map: dict[int, str]):
        """Update memory from current observation tokens."""
        # 1. Parse lp:* tokens for absolute position (offset from spawn)
        lp_row, lp_col, has_lp = 0, 0, False
        for token in obs.tokens:
            name = token.feature.name
            if name == "lp:south":
                lp_row = int(token.value); has_lp = True
            elif name == "lp:north":
                lp_row = -int(token.value); has_lp = True
            elif name == "lp:east":
                lp_col = int(token.value); has_lp = True
            elif name == "lp:west":
                lp_col = -int(token.value); has_lp = True

        if has_lp:
            self.position = (lp_row, lp_col)
        if self.position is None:
            return

        # Track position history for stuck detection
        self.position_history.append(self.position)
        if len(self.position_history) > 30:
            self.position_history.pop(0)

        # 2. Scan spatial tokens for walls and stations
        cr, cc = center
        for token in obs.tokens:
            if token.feature.name != "tag":
                continue
            loc = token.location
            if loc is None:
                continue
            abs_pos = (self.position[0] + loc[0] - cr,
                       self.position[1] + loc[1] - cc)
            val = token.value
            if val in wall_tag_ids:
                self.walls.add(abs_pos)
            elif val in station_tag_map:
                cat = station_tag_map[val]
                self.stations[abs_pos] = cat
                # Record hub position for coordinate frame conversion
                if cat == "hub" and self.hub_offset is None:
                    self.hub_offset = abs_pos

        # 3. Mark observed area as explored
        for dr in range(-cr, cr + 1):
            for dc in range(-cc, cc + 1):
                self.explored.add((self.position[0] + dr,
                                   self.position[1] + dc))

    def to_shared(self, pos: tuple[int, int]) -> Optional[tuple[int, int]]:
        """Convert spawn-relative position to hub-relative (shared frame)."""
        if self.hub_offset is None:
            return None
        return (pos[0] - self.hub_offset[0], pos[1] - self.hub_offset[1])

    def from_shared(self, shared_pos: tuple[int, int]) -> Optional[tuple[int, int]]:
        """Convert hub-relative (shared frame) position to spawn-relative."""
        if self.hub_offset is None:
            return None
        return (shared_pos[0] + self.hub_offset[0],
                shared_pos[1] + self.hub_offset[1])

    def is_wall_adjacent(self, direction: str) -> bool:
        """Check if moving in direction hits a known wall."""
        if self.position is None:
            return False
        dr, dc = DIRECTION_DELTAS.get(direction, (0, 0))
        return (self.position[0] + dr, self.position[1] + dc) in self.walls

    def find_nearest_station(self, category: str,
                             ref_pos: Optional[tuple[int, int]] = None,
                             max_ref_dist: Optional[int] = None,
                             ) -> Optional[tuple[int, int]]:
        """Find nearest remembered station of given category (absolute pos).

        If *ref_pos* and *max_ref_dist* are given, only consider stations
        within *max_ref_dist* Manhattan distance of *ref_pos*.  This is used
        for junction targeting: only junctions within alignment range of the
        hub (25 tiles) or existing network (15 tiles) are eligible.
        """
        if self.position is None:
            return None
        best, best_dist = None, float("inf")
        for pos, cat in self.stations.items():
            if cat != category:
                continue
            # Proximity filter (e.g. junction within hub radius)
            if ref_pos is not None and max_ref_dist is not None:
                ref_dist = abs(pos[0] - ref_pos[0]) + abs(pos[1] - ref_pos[1])
                if ref_dist > max_ref_dist:
                    continue
            dist = abs(pos[0] - self.position[0]) + abs(pos[1] - self.position[1])
            if dist < best_dist:
                best_dist = dist
                best = pos
        return best

    def is_stuck(self) -> bool:
        """Detect stuck from position history (conservative threshold)."""
        h = self.position_history
        if len(h) < 20:
            return False
        # Oscillating between ≤2 positions for 20 steps
        if len(set(h[-20:])) <= 2:
            return True
        return False


class SharedSpatialMemory:
    """Shared observation pool for multi-agent belief sharing.

    Following Catal, Van de Maele, ..., Albarracin et al. (2024) "Belief
    sharing: a blessing or a curse": agent factual observations (station
    locations) are shared as a social observation modality.  Each agent's
    spatial discoveries propagate to all other agents.

    All positions are stored in **hub-relative** coordinates (hub = origin).
    Each agent converts to/from its own spawn-relative frame via
    ``SpatialMemory.to_shared()`` / ``from_shared()``.

    Shares ground-truth observations (not posterior beliefs) to avoid echo
    chambers.  Agents maintain independent beliefs about what to DO.
    """

    def __init__(self):
        self.stations: dict[tuple[int, int], str] = {}
        self.explored: set[tuple[int, int]] = set()

    def contribute(self, agent_memory: SpatialMemory):
        """Merge one agent's station discoveries into the shared pool.

        Converts from agent's spawn-relative frame to hub-relative.
        Skips contribution if the agent hasn't discovered the hub yet
        (no coordinate frame available).
        """
        if agent_memory.hub_offset is None:
            return  # can't convert without knowing hub position
        for pos, cat in agent_memory.stations.items():
            shared_pos = agent_memory.to_shared(pos)
            if shared_pos is not None:
                self.stations[shared_pos] = cat
        # Share explored cells in hub-relative coords
        for pos in agent_memory.explored:
            shared_pos = agent_memory.to_shared(pos)
            if shared_pos is not None:
                self.explored.add(shared_pos)

    def find_nearest_station(
        self, category: str, position: tuple[int, int],
        max_hub_dist: Optional[int] = None,
    ) -> Optional[tuple[int, int]]:
        """Find nearest station from shared knowledge.

        Both ``position`` and the returned position are in hub-relative
        coordinates.  The caller must convert using ``SpatialMemory``
        ``to_shared()`` / ``from_shared()``.

        If *max_hub_dist* is given, only stations within that Manhattan
        distance of hub origin (0,0) are considered.
        """
        best, best_dist = None, float("inf")
        for pos, cat in self.stations.items():
            if cat != category and not (
                category == "extractor" and cat.startswith("extractor")
            ):
                continue
            # Hub proximity filter (shared coords are hub-relative, hub=(0,0))
            if max_hub_dist is not None:
                hub_dist = abs(pos[0]) + abs(pos[1])
                if hub_dist > max_hub_dist:
                    continue
            dist = abs(pos[0] - position[0]) + abs(pos[1] - position[1])
            if dist < best_dist:
                best_dist = dist
                best = pos
        return best

    def find_least_explored_direction(
        self, position: tuple[int, int], radius: int = 15
    ) -> Optional[tuple[int, int]]:
        """Find target in the least-explored cardinal direction.

        For scouts: navigate toward the area with the most unexplored
        cells (maximum epistemic value / information gain).
        """
        best_target, best_unexplored = None, 0
        for dr, dc in [(-radius, 0), (radius, 0), (0, -radius), (0, radius)]:
            target = (position[0] + dr, position[1] + dc)
            unexplored = sum(
                1 for r in range(-3, 4) for c in range(-3, 4)
                if (target[0] + r, target[1] + c) not in self.explored
            )
            if unexplored > best_unexplored:
                best_unexplored = unexplored
                best_target = target
        return best_target


# ---------------------------------------------------------------------------
# Mettagrid-independent utilities
# ---------------------------------------------------------------------------

def _build_tag_categories(tags: list[str]) -> dict[int, str]:
    """Build tag_value -> category mapping dynamically from tag names.

    More robust than hardcoded indices -- works across cogames versions.
    Does NOT require mettagrid.  Extractors are element-typed
    (e.g. ``"extractor:carbon"``) for EFE-optimal resource selection.
    """
    categories: dict[int, str] = {}
    for i, tag_name in enumerate(tags):
        name = tag_name.removeprefix("type:")
        if "extractor" in name:
            # Element-typed: richer generative model
            matched = False
            for elem in RESOURCE_NAMES:
                if elem in name:
                    categories[i] = f"extractor:{elem}"
                    matched = True
                    break
            if not matched:
                categories[i] = "extractor"  # solar or unknown
        elif name in ("hub",):
            categories[i] = "hub"
        elif name in ("junction",):
            categories[i] = "junction"
        elif name.startswith("c:"):
            categories[i] = "craft"
    return categories


# ---------------------------------------------------------------------------
# State (no mettagrid dependency)
# ---------------------------------------------------------------------------

@dataclass
class AIFBeliefState:
    """Per-agent navigator state for the AIF policy.

    Belief tracking and inference are handled by the shared
    ``BatchedAIFEngine``. This dataclass holds only per-agent
    navigator state (wander direction, step counts, logging).
    """
    step_count: int = 0
    wander_dir: int = 0
    wander_steps: int = WANDER_STEPS
    last_phase: int = 0
    last_hand: int = 0
    last_task_policy: int = TaskPolicy.EXPLORE
    spatial_memory: Optional[SpatialMemory] = None
    # Navigation POMDP state
    last_heading: str = "move_east"
    prev_target_dist: int = -1
    # Gear tracking
    has_role_gear: bool = False


# ---------------------------------------------------------------------------
# Level 2: JIT-compiled strategic inference functions
# ---------------------------------------------------------------------------

def _belief_update(agent, batched_obs, empirical_prior, option_actions):
    """JIT-compilable belief update -- runs every step (~28ms).

    Updates beliefs given new observations and the currently executing option.
    Does NOT replan (no infer_policies).
    """
    qs = agent.infer_states(batched_obs, empirical_prior=empirical_prior)
    pred, _ = agent.update_empirical_prior(option_actions, qs)
    return pred, qs


def _select_option(agent, qs):
    """JIT-compilable option selection -- runs at option termination (~42ms).

    Evaluates EFE over 25 two-step option policies and samples.
    """
    q_pi, _efe = agent.infer_policies(qs)
    sampled = agent.sample_action(q_pi)
    return sampled[:, 0], q_pi


# ---------------------------------------------------------------------------
# Level 0: JIT-compiled navigation POMDP functions
# ---------------------------------------------------------------------------

def _nav_infer(agent, batched_obs, empirical_prior, nav_actions):
    """JIT-compilable nav POMDP: belief update + policy selection.

    Runs every step. ~3-5ms for 16-state POMDP with 25 policies.
    Returns qs for downstream B-learning.
    """
    qs = agent.infer_states(batched_obs, empirical_prior=empirical_prior)
    pred, _ = agent.update_empirical_prior(nav_actions, qs)
    q_pi, _efe = agent.infer_policies(qs)
    sampled = agent.sample_action(q_pi)
    return pred, sampled[:, 0], q_pi, qs


def _nav_learn_B(agent, beliefs_T2, obs_T2, actions_T1):
    """JIT-compilable nav POMDP B-learning via Dirichlet updates.

    After observing a transition (s_{t-1}, a_{t-1}, s_t), updates the
    Dirichlet posterior pB.  The expected B matrices are recomputed so
    that future policy evaluation uses the learned dynamics.
    """
    # Use all positional args: param name is 'outcomes' locally, 'observations' on AWS
    return agent.infer_parameters(
        beliefs_T2, obs_T2, actions_T1, beliefs_T2, 0.0, 1.0,
    )


# ---------------------------------------------------------------------------
# Level 1: Option Executor (reactive state machines)
# ---------------------------------------------------------------------------

@dataclass
class OptionState:
    """Per-agent option execution state."""
    current_option: int = MacroOption.EXPLORE
    steps_in_option: int = 0
    prev_inv: int = int(ObsInventory.EMPTY)
    free_steps: int = 0  # consecutive steps at FREE junction (for DEFEND)


class OptionExecutor:
    """Level 1: Reactive state machines mapping obs -> task policy.

    Each macro-option defines observation-conditional rules that select
    the appropriate task policy without any planning or inference.
    """

    TIMEOUTS = {
        MacroOption.MINE_CYCLE: 80,
        MacroOption.CRAFT_CYCLE: 200,
        MacroOption.CAPTURE_CYCLE: 200,
        MacroOption.EXPLORE: 50,
        MacroOption.DEFEND: 60,
    }

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.states = [OptionState() for _ in range(n_agents)]
        # Roles: 4 miners (even), 3 aligners (odd<7), 1 scout (agent 7)
        self._role = [_agent_role(i, n_agents) for i in range(n_agents)]
        self._is_aligner = [r == "aligner" for r in self._role]
        self._is_scout = [r == "scout" for r in self._role]

    def get_task_policy(self, agent_id: int, obs_ints: list[int]) -> int:
        """Map current option + observation -> task policy."""
        st = self.states[agent_id]
        option = st.current_option

        o_res = obs_ints[0]   # ObsResource
        o_sta = obs_ints[1]   # ObsStation
        o_inv = obs_ints[2]   # ObsInventory

        if option == MacroOption.MINE_CYCLE:
            return self._mine_cycle(o_res, o_sta, o_inv)
        elif option == MacroOption.CRAFT_CYCLE:
            return self._craft_cycle(o_sta, o_inv)
        elif option == MacroOption.CAPTURE_CYCLE:
            return self._capture_cycle(o_sta, o_inv)
        elif option == MacroOption.EXPLORE:
            return TaskPolicy.EXPLORE
        elif option == MacroOption.DEFEND:
            return self._defend(o_sta, o_inv)
        return TaskPolicy.EXPLORE

    def check_termination(self, agent_id: int, obs_ints: list[int]) -> bool:
        """Check if current option should terminate."""
        st = self.states[agent_id]
        option = st.current_option

        # Timeout check
        timeout = self.TIMEOUTS.get(option, 60)
        if st.steps_in_option >= timeout:
            return True

        o_res = obs_ints[0]
        o_sta = obs_ints[1]
        o_inv = obs_ints[2]
        o_contest = obs_ints[3]

        if option == MacroOption.MINE_CYCLE:
            # Deposit complete: had resource (or both), now empty (or gear only)
            if (st.prev_inv in (ObsInventory.HAS_RESOURCE, ObsInventory.HAS_BOTH)
                    and o_inv in (ObsInventory.EMPTY, ObsInventory.HAS_GEAR)):
                return True
        elif option == MacroOption.CRAFT_CYCLE:
            # Gear acquired
            if o_inv in (ObsInventory.HAS_GEAR, ObsInventory.HAS_BOTH):
                return True
        elif option == MacroOption.CAPTURE_CYCLE:
            # Gear used (had gear/both, now no gear)
            if (st.prev_inv in (ObsInventory.HAS_GEAR, ObsInventory.HAS_BOTH)
                    and o_inv not in (ObsInventory.HAS_GEAR, ObsInventory.HAS_BOTH)):
                return True
            # Started without gear -- bail after grace period
            if o_inv not in (ObsInventory.HAS_GEAR, ObsInventory.HAS_BOTH) and st.steps_in_option > 5:
                return True
        elif option == MacroOption.EXPLORE:
            # Scouts: never self-terminate on station (only timeout)
            # — epistemic agent explores continuously
            if self._is_scout[agent_id]:
                pass
            elif self._is_aligner[agent_id]:
                # Aligners: only craft/junction (hub is useless)
                if o_sta >= ObsStation.CRAFT:
                    return True
            else:
                # Miners: any resource or station ends exploration
                if o_res >= ObsResource.AT or o_sta > ObsStation.NONE:
                    return True
        elif option == MacroOption.DEFEND:
            # Junction secured for 10+ steps
            if o_sta == ObsStation.JUNCTION and o_contest == ObsContest.FREE:
                st.free_steps += 1
                if st.free_steps >= 10:
                    return True
            else:
                st.free_steps = 0

        return False

    def set_option(self, agent_id: int, option: int):
        """Activate a new option for this agent.

        Enforces role-based option initiation sets (Options framework):
        miners cannot initiate CRAFT/CAPTURE, aligners cannot initiate MINE.
        If a disallowed option is selected, fall back to role-appropriate default.
        """
        # Role filter: restrict option initiation by role (precision gate)
        if self._is_scout[agent_id]:
            # Scout: only EXPLORE/DEFEND allowed (epistemic initiation set)
            if option not in (MacroOption.EXPLORE, MacroOption.DEFEND):
                option = MacroOption.EXPLORE
        elif self._is_aligner[agent_id]:
            if option == MacroOption.MINE_CYCLE:
                # Redirect to next-best from aligner initiation set
                # (not EXPLORE — that was causing the stuck loop)
                inv = self.states[agent_id].prev_inv
                if inv in (ObsInventory.HAS_GEAR, ObsInventory.HAS_BOTH):
                    option = MacroOption.CAPTURE_CYCLE
                else:
                    option = MacroOption.CRAFT_CYCLE
        else:  # miner
            if option in (MacroOption.CRAFT_CYCLE, MacroOption.CAPTURE_CYCLE):
                option = MacroOption.MINE_CYCLE

        st = self.states[agent_id]
        st.current_option = option
        st.steps_in_option = 0
        st.free_steps = 0

    def tick(self, agent_id: int, obs_ints: list[int]):
        """Advance step counter and update tracking state."""
        st = self.states[agent_id]
        st.steps_in_option += 1
        st.prev_inv = obs_ints[2]

    # ------------------------------------------------------------------
    # Option-specific reactive policies
    # ------------------------------------------------------------------

    @staticmethod
    def _mine_cycle(o_res, o_sta, o_inv):
        """MINE_CYCLE: NAV_RESOURCE until pickup -> NAV_DEPOT until deposit."""
        if o_inv in (ObsInventory.HAS_RESOURCE, ObsInventory.HAS_BOTH):
            return TaskPolicy.NAV_DEPOT  # Navigate to hub (auto-deposits at dist=0)
        # Keep navigating to extractor — auto-extracts at dist=0 (noop via dr==dc==0)
        return TaskPolicy.NAV_RESOURCE

    @staticmethod
    def _craft_cycle(o_sta, o_inv):
        """CRAFT_CYCLE: hub (get hearts) -> role-specific gear station.

        Hub pays element costs; agent just needs to be on the right station.
        NAV_GEAR targets role-specific craft station (c:aligner for aligners,
        c:miner for miners) instead of any craft station.
        """
        if o_inv in (ObsInventory.HAS_GEAR, ObsInventory.HAS_BOTH):
            return TaskPolicy.WAIT  # Already has gear
        if o_inv == ObsInventory.HAS_RESOURCE:
            return TaskPolicy.NAV_GEAR  # Have hearts, go to role-specific station
        return TaskPolicy.NAV_DEPOT  # Need hearts from hub first

    @staticmethod
    def _capture_cycle(o_sta, o_inv):
        """CAPTURE_CYCLE: get hearts if needed, then navigate to junction.

        Junction alignment costs 1 gear + 1 heart. HAS_GEAR alone means
        the agent has gear but no hearts — must visit hub first.
        """
        if o_inv not in (ObsInventory.HAS_GEAR, ObsInventory.HAS_BOTH):
            return TaskPolicy.WAIT  # No gear, bail
        if o_inv == ObsInventory.HAS_GEAR:
            return TaskPolicy.NAV_DEPOT  # Get hearts first
        # HAS_BOTH: has gear + hearts → go capture
        return TaskPolicy.NAV_JUNCTION

    @staticmethod
    def _defend(o_sta, o_inv):
        """DEFEND: go to junction and hold it. Get hearts if has gear only."""
        if o_inv == ObsInventory.HAS_GEAR:
            return TaskPolicy.NAV_DEPOT  # Need hearts for alignment
        return TaskPolicy.NAV_JUNCTION


# ---------------------------------------------------------------------------
# Batched AIF Engine (hierarchical)
# ---------------------------------------------------------------------------

class BatchedAIFEngine:
    """Hierarchical batched active inference engine.

    Level 2: Strategic POMDP with 5 macro-options (25 two-step policies).
             Replans only at option termination (~42ms).
    Level 1: OptionExecutor maps option + obs -> task policy (~0ms).

    Most steps only run belief update (~28ms). Full replanning happens
    every 30-80 steps when an option terminates.
    """

    def __init__(self, n_agents: int = 8, learn_B: bool = False,
                 learn_interval: int = 50, policy_len: int = 2,
                 auto_chain: bool = True, context_E: bool = False):
        self.n_agents = n_agents
        self.learn_B = learn_B
        self.learn_interval = learn_interval
        self.auto_chain = auto_chain
        self.context_E = context_E
        self._step_count = 0

        # Level 2: Strategic agent with 5 macro-options
        self.agent = CogsGuardPOMDP.create_strategic_agent(
            n_agents, learn_B=learn_B, policy_len=policy_len
        )

        # Level 1: Option executor
        self.option_executor = OptionExecutor(n_agents)

        # Shared spatial memory — social observation modality
        # (Catal, Van de Maele, ..., Albarracin et al., 2024)
        self.shared_memory = SharedSpatialMemory()

        # Obs buffer: per-agent, per-modality, shape (1,) = (T=1,)
        self._obs_buffer: list[list[Any]] = [
            [jnp.array([0]) for _ in range(6)]
            for _ in range(n_agents)
        ]

        # Discrete obs for option executor
        self._discrete_obs: list[list[int]] = [[0] * 6 for _ in range(n_agents)]

        # Beliefs managed here
        self.empirical_prior = self.agent.D
        self.qs = None

        # Current option actions: (n_agents, 4) -- same option for all 4 factors
        self._current_options = np.full(n_agents, MacroOption.EXPLORE, dtype=np.int32)
        self._option_actions = jnp.full(
            (n_agents, 4), MacroOption.EXPLORE, dtype=jnp.int32
        )

        # Cached task policies
        self._cached_policies = [int(TaskPolicy.EXPLORE)] * n_agents

        # B-learning buffers
        self._prev_qs = None
        self._prev_obs = None
        self._prev_actions = None

        # Level 0: Navigation POMDP (with B-learning by default)
        self.nav_agent = create_nav_agent(n_agents, policy_len=2,
                                          learn_B=True)
        self._nav_obs_buffer: list[list[Any]] = [
            [jnp.array([0]) for _ in range(2)]
            for _ in range(n_agents)
        ]
        self.nav_prior = self.nav_agent.D
        self._nav_actions = jnp.full(
            (n_agents, 2), NavAction.TOWARD, dtype=jnp.int32
        )
        self._cached_nav_actions = [int(NavAction.TOWARD)] * n_agents

        # Nav B-learning buffers
        self._prev_nav_qs = None
        self._prev_nav_obs = None
        self._prev_nav_actions = None
        # Store initial pB/B for reset on option change
        self._nav_initial_pB = (
            [pb.copy() for pb in self.nav_agent.pB]
            if self.nav_agent.pB is not None else None
        )
        self._nav_initial_B = [b.copy() for b in self.nav_agent.B]

        # JIT-compile all functions
        self._jit_belief_update = eqx.filter_jit(_belief_update)
        self._jit_select_option = eqx.filter_jit(_select_option)
        self._jit_nav_infer = eqx.filter_jit(_nav_infer)
        self._jit_nav_learn_B = eqx.filter_jit(_nav_learn_B)

        # Warmup JIT compilation (avoids timeout on first eval step)
        dummy_obs = [jnp.zeros((n_agents, 1), dtype=jnp.int32) for _ in range(6)]
        dummy_actions = jnp.full((n_agents, 4), 0, dtype=jnp.int32)
        pred, qs = self._jit_belief_update(
            self.agent, dummy_obs, self.empirical_prior, dummy_actions
        )
        self._jit_select_option(self.agent, qs)

        # Warmup nav POMDP JIT (including B-learning)
        dummy_nav_obs = [jnp.zeros((n_agents, 1), dtype=jnp.int32) for _ in range(2)]
        dummy_nav_act = jnp.full((n_agents, 2), 0, dtype=jnp.int32)
        _pred, _act, _qpi, _qs = self._jit_nav_infer(
            self.nav_agent, dummy_nav_obs, self.nav_prior, dummy_nav_act
        )

        # Context-dependent E vectors for principled hierarchical AIF.
        # E(π | context) where context = inventory state.
        # Higher level (inventory) sets the habit prior for the lower level
        # (policy selection), modulating which options are favoured.
        if self.context_E:
            self._base_E = np.array(self.agent.E)  # (n_agents, 25)
            n_pol = self._base_E.shape[1]
            # Aligner contexts: CRAFT-biased (need gear) vs CAPTURE-biased (have gear)
            self._E_aligner_craft = np.ones(n_pol)
            self._E_aligner_craft[5:10] = 6.0    # CRAFT first — strong bias
            self._E_aligner_craft[10:15] = 1.5   # CAPTURE — available but low
            self._E_aligner_craft[15:20] = 1.5   # EXPLORE — find stations
            self._E_aligner_craft[20:25] = 1.0   # DEFEND
            self._E_aligner_craft[0:5] = 0.001   # MINE — blocked
            self._E_aligner_craft /= self._E_aligner_craft.sum()

            self._E_aligner_capture = np.ones(n_pol)
            self._E_aligner_capture[10:15] = 6.0  # CAPTURE first — strong bias
            self._E_aligner_capture[5:10] = 1.5   # CRAFT — fallback
            self._E_aligner_capture[20:25] = 2.0  # DEFEND — hold junctions
            self._E_aligner_capture[15:20] = 1.0  # EXPLORE
            self._E_aligner_capture[0:5] = 0.001  # MINE — blocked
            self._E_aligner_capture /= self._E_aligner_capture.sum()

    def submit_and_get_policy(self, agent_id: int, jax_obs: list) -> int:
        """Store obs for agent_id and return its task policy.

        Agent 0 triggers batched inference for all agents.
        Others return cached policies (1-step lag).
        """
        self._obs_buffer[agent_id] = jax_obs
        self._discrete_obs[agent_id] = [int(o[0]) for o in jax_obs]

        if agent_id == 0:
            self._run_batch()

        return self._cached_policies[agent_id]

    def submit_nav_and_get_action(self, agent_id: int, nav_obs: list) -> int:
        """Store nav obs for agent_id and return its nav action.

        Agent 0 triggers batched nav inference for all agents.
        Others return cached nav actions (1-step lag).
        """
        self._nav_obs_buffer[agent_id] = nav_obs

        if agent_id == 0:
            self._run_nav_batch()

        return self._cached_nav_actions[agent_id]

    def get_beliefs(self, agent_id: int):
        """Return per-agent beliefs (qs) from the batched posterior."""
        if self.qs is None:
            return None
        return [q[agent_id] for q in self.qs]

    # ------------------------------------------------------------------
    # Batch inference (hierarchical)
    # ------------------------------------------------------------------

    def _run_batch(self):
        """Run hierarchical batched inference."""
        # Stack obs: (n_agents, T=1) per modality
        batched_obs = []
        for m in range(6):
            stacked = jnp.stack(
                [self._obs_buffer[i][m] for i in range(self.n_agents)]
            )
            batched_obs.append(stacked)

        # Level 2a: Belief update (every step, ~28ms)
        new_prior, qs = self._jit_belief_update(
            self.agent, batched_obs, self.empirical_prior, self._option_actions
        )
        self.empirical_prior = new_prior
        self.qs = qs

        # Level 1: Check option termination and get task policies
        terminated = []
        for i in range(self.n_agents):
            obs_ints = self._discrete_obs[i]
            if self.option_executor.check_termination(i, obs_ints):
                terminated.append(i)
            self.option_executor.tick(i, obs_ints)

        # Level 2b: Replan for terminated agents (~42ms if any)
        if terminated:
            if self.auto_chain:
                # Aligner auto-chain: CRAFT→CAPTURE→CRAFT loop without
                # replanning.  Pragmatic option composition — see ROADMAP
                # "Context-Dependent Preferences" for principled alternative.
                replan_needed = []
                for i in terminated:
                    if self._auto_chain_aligner(i):
                        pass  # Option already set by auto-chain
                    else:
                        replan_needed.append(i)
            else:
                replan_needed = terminated

            if replan_needed:
                # Context-dependent E: update habit prior based on inventory
                agent = self._apply_context_E() if self.context_E else self.agent
                options, _q_pi = self._jit_select_option(agent, qs)
                for i in replan_needed:
                    new_option = int(options[i])
                    self.option_executor.set_option(i, new_option)

            # Record post-filter options for all terminated agents
            for i in terminated:
                self._current_options[i] = self.option_executor.states[i].current_option
            # Update option actions for next belief update
            self._option_actions = jnp.tile(
                jnp.array(self._current_options)[:, None], (1, 4)
            )
            # Reset nav POMDP beliefs for agents with new options
            self._reset_nav_beliefs(terminated)

        # Level 1: Get task policies from option executor
        for i in range(self.n_agents):
            self._cached_policies[i] = self.option_executor.get_task_policy(
                i, self._discrete_obs[i]
            )

        # Online B-learning
        self._step_count += 1
        if (self.learn_B and self._prev_qs is not None
                and self._step_count % self.learn_interval == 0):
            self._update_B(batched_obs, qs)

        # Periodic logging (every 500 steps)
        if self._step_count % 500 == 0:
            option_names = [OPTION_NAMES[self._current_options[i]]
                           for i in range(self.n_agents)]
            task_names = [TASK_POLICY_NAMES[self._cached_policies[i]]
                         for i in range(self.n_agents)]
            extras = []
            if self.learn_B and hasattr(self.agent, 'pB') and self.agent.pB is not None:
                pB_sum = sum(float(pb.sum()) for pb in self.agent.pB)
                extras.append(f"pB={pB_sum:.0f}")
            opt_steps = [self.option_executor.states[i].steps_in_option
                        for i in range(self.n_agents)]
            extras.append(f"opt_steps={opt_steps}")
            extras.append(f"shared_stations={len(self.shared_memory.stations)}")
            # Count unique shared station categories
            shared_cats: dict[str, int] = {}
            for _p, _c in self.shared_memory.stations.items():
                shared_cats[_c] = shared_cats.get(_c, 0) + 1
            extras.append(f"shared_cats={shared_cats}")
            extra_str = f", {', '.join(extras)}" if extras else ""
            print(f"[AIF step={self._step_count}] options={option_names}")
            print(f"  tasks={task_names}{extra_str}")

        # Store for next B-learning step
        self._prev_qs = qs
        self._prev_obs = batched_obs
        self._prev_actions = self._option_actions

    # ------------------------------------------------------------------
    # Aligner auto-chaining (option composition)
    # ------------------------------------------------------------------

    def _apply_context_E(self):
        """Context-dependent E: update habit prior from inventory state.

        Principled hierarchical AIF: the higher level (inventory observation)
        sets E(π | context) for the lower level (policy selection).

            context = HAS_BOTH | HAS_GEAR  →  E biased toward CAPTURE
            context = EMPTY | HAS_RESOURCE →  E biased toward CRAFT

        Returns an updated agent (equinox functional update — immutable).
        """
        new_E = np.array(self._base_E)  # (n_agents, 25)
        for i in range(self.n_agents):
            if not self.option_executor._is_aligner[i]:
                continue
            inv = self._discrete_obs[i][2]
            if inv in (ObsInventory.HAS_GEAR, ObsInventory.HAS_BOTH):
                new_E[i] = self._E_aligner_capture
            else:
                new_E[i] = self._E_aligner_craft
        return eqx.tree_at(lambda a: a.E, self.agent, jnp.array(new_E))

    def _auto_chain_aligner(self, agent_id: int) -> bool:
        """Auto-chain CRAFT↔CAPTURE for aligners, skipping POMDP replan.

        The aligner economy loop is deterministic:
          CRAFT_CYCLE (hub→gear station→HAS_BOTH) → CAPTURE_CYCLE (junction)
          → CRAFT_CYCLE (restock) → CAPTURE_CYCLE → ...

        Returns True if auto-chaining was applied (option set directly),
        False if the agent should go through normal POMDP replanning.
        """
        if not self.option_executor._is_aligner[agent_id]:
            return False
        st = self.option_executor.states[agent_id]
        current = st.current_option
        inv = self._discrete_obs[agent_id][2]

        if (current == MacroOption.CRAFT_CYCLE
                and inv in (ObsInventory.HAS_GEAR, ObsInventory.HAS_BOTH)):
            # Got gear → go capture
            st.current_option = MacroOption.CAPTURE_CYCLE
            st.steps_in_option = 0
            st.free_steps = 0
            return True

        if current == MacroOption.CAPTURE_CYCLE:
            # Capture done (or timed out) → go craft again
            st.current_option = MacroOption.CRAFT_CYCLE
            st.steps_in_option = 0
            st.free_steps = 0
            return True

        return False

    # ------------------------------------------------------------------
    # Level 0: Nav POMDP batch inference
    # ------------------------------------------------------------------

    def _run_nav_batch(self):
        """Run batched nav POMDP inference + online B-learning."""
        # Stack obs: (n_agents, T=1) per modality
        batched_nav_obs = []
        for m in range(2):
            stacked = jnp.stack(
                [self._nav_obs_buffer[i][m] for i in range(self.n_agents)]
            )
            batched_nav_obs.append(stacked)

        # Nav inference: belief update + policy selection (~3-5ms)
        new_prior, nav_actions, _q_pi, nav_qs = self._jit_nav_infer(
            self.nav_agent, batched_nav_obs, self.nav_prior, self._nav_actions
        )
        self.nav_prior = new_prior

        # Cache actions for each agent
        nav_act_np = np.asarray(nav_actions, dtype=np.int32)
        for i in range(self.n_agents):
            self._cached_nav_actions[i] = int(nav_act_np[i])

        # Online B-learning: update nav transition model from experience
        if self._prev_nav_qs is not None and self.nav_agent.pB is not None:
            beliefs_T2 = [
                jnp.concatenate([self._prev_nav_qs[f], nav_qs[f]], axis=1)
                for f in range(len(nav_qs))
            ]
            obs_T2 = [
                jnp.concatenate(
                    [self._prev_nav_obs[m], batched_nav_obs[m]], axis=1
                )
                for m in range(len(batched_nav_obs))
            ]
            actions_T1 = self._prev_nav_actions[:, None, :]
            self.nav_agent = self._jit_nav_learn_B(
                self.nav_agent, beliefs_T2, obs_T2, actions_T1
            )

        # Store for next B-learning step
        self._prev_nav_qs = nav_qs
        self._prev_nav_obs = batched_nav_obs
        self._prev_nav_actions = self._nav_actions

        # Store actions for next belief update (both factors get same action)
        self._nav_actions = jnp.tile(
            nav_actions[:, None].astype(jnp.int32), (1, 2)
        )

    def _reset_nav_beliefs(self, agent_ids: list[int]):
        """Reset nav POMDP beliefs and learned B for agents that changed options.

        When the strategic POMDP picks a new option the nav target changes,
        so relative actions (LEFT/RIGHT/TOWARD) map to different compass
        directions.  The learned B matrices are no longer valid and must
        be reset to the prior so the agent re-learns which actions work
        in the new spatial context.
        """
        for i in agent_ids:
            for f in range(len(self.nav_prior)):
                self.nav_prior[f] = self.nav_prior[f].at[i].set(
                    self.nav_agent.D[f][i]
                )

        # Reset pB and B to initial priors (Dirichlet concentration resets)
        if self._nav_initial_pB is not None:
            for f in range(len(self._nav_initial_pB)):
                for i in agent_ids:
                    new_pB_f = self.nav_agent.pB[f].at[i].set(
                        self._nav_initial_pB[f][i]
                    )
                    new_B_f = self.nav_agent.B[f].at[i].set(
                        self._nav_initial_B[f][i]
                    )
                    self.nav_agent = eqx.tree_at(
                        lambda a, _f=f: a.pB[_f], self.nav_agent, new_pB_f
                    )
                    self.nav_agent = eqx.tree_at(
                        lambda a, _f=f: a.B[_f], self.nav_agent, new_B_f
                    )

    # ------------------------------------------------------------------
    # B-learning (Dirichlet updates)
    # ------------------------------------------------------------------

    def _update_B(self, curr_obs, curr_qs):
        """Update B matrices via online Dirichlet learning."""
        beliefs_T2 = [
            jnp.concatenate([self._prev_qs[f], curr_qs[f]], axis=1)
            for f in range(len(curr_qs))
        ]
        obs_T2 = [
            jnp.concatenate([self._prev_obs[m], curr_obs[m]], axis=1)
            for m in range(len(curr_obs))
        ]
        actions_T1 = self._prev_actions[:, None, :]

        # Use all positional args: param name differs across pymdp versions
        self.agent = self.agent.infer_parameters(
            beliefs_T2, obs_T2, actions_T1, beliefs_T2, 0.0, 1.0,
        )


# ---------------------------------------------------------------------------
# Per-agent policy implementation (requires mettagrid)
# ---------------------------------------------------------------------------

# Import mettagrid lazily -- only when classes are actually used at runtime.
try:
    from mettagrid.policy.policy import (
        MultiAgentPolicy as _MultiAgentPolicy,
        StatefulAgentPolicy as _StatefulAgentPolicy,
        StatefulPolicyImpl as _StatefulPolicyImpl,
    )
    from mettagrid.policy.policy_env_interface import (
        PolicyEnvInterface as _PolicyEnvInterface,
    )
    from mettagrid.simulator import Action as _Action
    from mettagrid.simulator.interface import AgentObservation as _AgentObservation

    _HAS_METTAGRID = True
except ImportError:
    _HAS_METTAGRID = False

    # Stub base classes so the module can be imported on Windows
    class _StatefulPolicyImpl:  # type: ignore[no-redef]
        pass

    class _MultiAgentPolicy:  # type: ignore[no-redef]
        pass


class AIFCogPolicyImpl(_StatefulPolicyImpl):
    """Discrete active inference agent for a single CogsGuard agent.

    Each step:
    1. Convert AgentObservation -> numpy array -> 6 discrete POMDP observations
    2. Submit obs to BatchedAIFEngine -> get task policy (hierarchical)
    3. Execute selected task policy via navigator (spatial movement)
    """

    def __init__(
        self,
        policy_env_info: _PolicyEnvInterface,
        agent_id: int,
        engine: BatchedAIFEngine,
        discretizer: ObservationDiscretizer,
    ):
        self._agent_id = agent_id
        self._policy_env_info = policy_env_info
        self._engine = engine
        self._discretizer = discretizer

        self._action_names = policy_env_info.action_names
        self._action_name_set = set(self._action_names)
        self._center = (policy_env_info.obs_height // 2, policy_env_info.obs_width // 2)

        # Build tag lookups (same pattern as starter_agent.py)
        self._tag_name_to_id = {name: idx for idx, name in enumerate(policy_env_info.tags)}
        self._extractor_tags = self._resolve_tag_ids(
            [f"{e}_extractor" for e in RESOURCE_NAMES]
        )
        # Per-element extractor tags for EFE-optimal resource selection
        self._element_extractor_tags: dict[str, set[int]] = {
            elem: self._resolve_tag_ids([f"{elem}_extractor"])
            for elem in RESOURCE_NAMES
        }
        self._hub_tags = self._resolve_tag_ids(["hub"])
        self._craft_tags = self._resolve_tag_ids(
            [f"c:{g}" for g in GEAR_NAMES]
        )
        self._junction_tags = self._resolve_tag_ids(["junction"])
        self._heart_source_tags = self._resolve_tag_ids(["hub", "chest"])
        # Per-role gear station tags
        self._miner_gear_tags = self._resolve_tag_ids(["c:miner"])
        self._aligner_gear_tags = self._resolve_tag_ids(["c:aligner"])

        # Spatial memory support: wall tags + station tag map
        self._wall_tags = self._resolve_tag_ids(["wall"])
        self._station_tag_map: dict[int, str] = {}
        # Element-typed extractor categories for richer world model
        for elem in RESOURCE_NAMES:
            for tid in self._element_extractor_tags[elem]:
                self._station_tag_map[tid] = f"extractor:{elem}"
        for tid in self._hub_tags:
            self._station_tag_map[tid] = "hub"
        for tid in self._craft_tags:
            self._station_tag_map[tid] = "craft"
        for tid in self._junction_tags:
            self._station_tag_map[tid] = "junction"

    def _resolve_tag_ids(self, names: list[str]) -> set[int]:
        """Resolve tag names to tag value IDs (handles type: prefix)."""
        tag_ids: set[int] = set()
        for name in names:
            if name in self._tag_name_to_id:
                tag_ids.add(self._tag_name_to_id[name])
            type_name = f"type:{name}"
            if type_name in self._tag_name_to_id:
                tag_ids.add(self._tag_name_to_id[type_name])
        return tag_ids

    def initial_agent_state(self) -> AIFBeliefState:
        """Create initial navigator state (beliefs are in the shared engine)."""
        return AIFBeliefState(
            wander_dir=self._agent_id % len(WANDER_DIRECTIONS),
            spatial_memory=SpatialMemory(),
        )

    def step_with_state(
        self, obs: _AgentObservation, state: AIFBeliefState
    ) -> tuple[_Action, AIFBeliefState]:
        """Run one step of the AIF agent."""
        # 0. Update spatial memory from raw observation
        if state.spatial_memory is None:
            state.spatial_memory = SpatialMemory()
        state.spatial_memory.update(
            obs, self._center, self._wall_tags, self._station_tag_map
        )
        # Contribute discoveries to shared memory (belief sharing)
        self._engine.shared_memory.contribute(state.spatial_memory)

        # 1. Convert AgentObservation -> numpy array
        obs_array = self._obs_to_array(obs)

        # 2. Discretize -> 6-tuple (o_res, o_sta, o_inv, o_contest, o_social, o_role)
        disc_obs = self._discretizer.discretize_obs(obs_array)

        # 3. Convert to jax arrays -- shape (T=1,) per modality
        jax_obs = [jnp.array([int(o)]) for o in disc_obs]

        # 4. Submit to batched engine -> get task policy (hierarchical)
        task_policy = self._engine.submit_and_get_policy(self._agent_id, jax_obs)

        # 5. Execute selected task policy via navigator
        action, state = self._execute_task_policy(task_policy, obs, state)

        # 6. Track for logging -- get beliefs from engine
        beliefs = self._engine.get_beliefs(self._agent_id)
        if beliefs is not None:
            phase_beliefs = np.asarray(beliefs[0][-1])  # last timestep
            hand_beliefs = np.asarray(beliefs[1][-1])
            state.last_phase = int(np.argmax(phase_beliefs))
            state.last_hand = int(np.argmax(hand_beliefs))
        state.last_task_policy = task_policy
        state.step_count += 1

        return action, state

    # ------------------------------------------------------------------
    # Observation conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _obs_to_array(obs: _AgentObservation) -> np.ndarray:
        """Convert AgentObservation tokens to (200, 3) uint8 array."""
        arr = np.full((200, 3), 255, dtype=np.uint8)
        for i, token in enumerate(obs.tokens):
            if i >= 200:
                break
            arr[i] = token.raw_token
        return arr

    # ------------------------------------------------------------------
    # Task-policy execution (nav POMDP)
    # ------------------------------------------------------------------

    def _execute_task_policy(
        self,
        task_policy: int,
        obs: _AgentObservation,
        state: AIFBeliefState,
    ) -> tuple[_Action, AIFBeliefState]:
        """Execute a task-level policy using the navigation POMDP.

        Level 2 (strategic POMDP) selects WHAT to do (task policy).
        Level 0 (nav POMDP) decides HOW to move toward the target.
        """
        # Noop task policies — no movement needed
        if task_policy in (
            TaskPolicy.MINE, TaskPolicy.DEPOSIT, TaskPolicy.CRAFT,
            TaskPolicy.ACQUIRE_GEAR, TaskPolicy.CAPTURE, TaskPolicy.WAIT,
        ):
            return self._action("noop"), state

        # Resolve target position (absolute coordinates)
        target = self._resolve_nav_target(task_policy, obs, state)

        # Compute nav POMDP observations
        obs_range, obs_movement = self._compute_nav_obs(target, state)

        # Submit to nav POMDP engine -> get relative action
        nav_obs = [jnp.array([obs_range]), jnp.array([obs_movement])]
        nav_action = self._engine.submit_nav_and_get_action(
            self._agent_id, nav_obs
        )

        # Convert relative -> absolute direction
        direction = self._relative_to_absolute(nav_action, target, state)

        return self._action(direction), state

    def _resolve_nav_target(
        self,
        task_policy: int,
        obs: _AgentObservation,
        state: AIFBeliefState,
    ) -> Optional[tuple[int, int]]:
        """Resolve task policy to absolute target position.

        For NAV_RESOURCE, uses EFE-optimal element selection: the element
        whose extraction minimally reduces D_KL from the preferred uniform
        team resource distribution = the scarcest element.

        Returns (abs_row, abs_col) or None if no target found.
        """
        mem = state.spatial_memory

        # Map task policy to tag set and station category
        if task_policy == TaskPolicy.NAV_RESOURCE:
            # EFE-optimal element selection (Level 0.5 goal generation):
            # G(e) = D_KL(Q(resources|mine_e) || C_uniform)
            # Scarcest element minimizes G → lowest EFE.
            team_res = self._team_resources(obs)
            if team_res:
                scarcest = min(RESOURCE_NAMES,
                               key=lambda e: team_res.get(e, 0))
                tag_ids = self._element_extractor_tags[scarcest]
                category = f"extractor:{scarcest}"
            else:
                tag_ids, category = self._extractor_tags, "extractor"
        elif task_policy == TaskPolicy.NAV_DEPOT:
            tag_ids, category = self._hub_tags, "hub"
        elif task_policy == TaskPolicy.NAV_CRAFT:
            tag_ids, category = self._craft_tags, "craft"
        elif task_policy == TaskPolicy.NAV_GEAR:
            # Role-specific gear station
            role = _agent_role(self._agent_id, self._engine.n_agents)
            tag_ids = self._aligner_gear_tags if role == "aligner" else self._miner_gear_tags
            category = "craft"  # gear stations are categorized as "craft" in spatial memory
        elif task_policy == TaskPolicy.NAV_JUNCTION:
            # Junction alignment requires proximity: ≤25 from hub or ≤15
            # from existing team network.  Target junctions in range first.
            return self._resolve_junction_target(obs, mem)
        elif task_policy in (TaskPolicy.EXPLORE, TaskPolicy.YIELD):
            if mem is not None:
                return self._get_frontier_target(mem)
            return None
        else:
            return None

        # 1. Try visible target (convert egocentric to absolute)
        target_loc = self._closest_tag_location(obs, tag_ids)
        if target_loc is not None and mem is not None and mem.position is not None:
            abs_r = mem.position[0] + target_loc[0] - self._center[0]
            abs_c = mem.position[1] + target_loc[1] - self._center[1]
            return (abs_r, abs_c)

        # 2. Try own spatial memory
        if mem is not None and mem.position is not None:
            station_pos = mem.find_nearest_station(category)
            if station_pos is not None:
                return station_pos

        # 2b. Try shared spatial memory (belief sharing — Catal et al. 2024)
        #     Shared memory uses hub-relative coords; convert both ways.
        if mem is not None and mem.position is not None:
            shared = self._engine.shared_memory
            shared_pos = mem.to_shared(mem.position)
            if shared_pos is not None:
                station_shared = shared.find_nearest_station(category, shared_pos)
                if station_shared is not None:
                    station_local = mem.from_shared(station_shared)
                    if station_local is not None:
                        return station_local

        # 2c. Fallback for NAV_RESOURCE: try any element's extractor
        if task_policy == TaskPolicy.NAV_RESOURCE and mem is not None and mem.position is not None:
            for elem in RESOURCE_NAMES:
                fb_tags = self._element_extractor_tags[elem]
                fb_loc = self._closest_tag_location(obs, fb_tags)
                if fb_loc is not None:
                    abs_r = mem.position[0] + fb_loc[0] - self._center[0]
                    abs_c = mem.position[1] + fb_loc[1] - self._center[1]
                    return (abs_r, abs_c)
            for elem in RESOURCE_NAMES:
                cat = f"extractor:{elem}"
                pos = mem.find_nearest_station(cat)
                if pos is not None:
                    return pos
                # Shared memory fallback (hub-relative conversion)
                shared_pos = mem.to_shared(mem.position)
                if shared_pos is not None:
                    shared_result = self._engine.shared_memory.find_nearest_station(
                        cat, shared_pos
                    )
                    if shared_result is not None:
                        local_result = mem.from_shared(shared_result)
                        if local_result is not None:
                            return local_result

        # 3. Fall back to frontier exploration
        if mem is not None:
            return self._get_frontier_target(mem)

        return None

    # ------------------------------------------------------------------
    # Inventory parsing (adapted from starter_agent.py)
    # ------------------------------------------------------------------

    def _inventory_amounts(self, obs: _AgentObservation) -> dict[str, int]:
        """Extract inventory amounts from center-tile tokens."""
        items: dict[str, int] = {}
        for token in obs.tokens:
            if token.location != self._center:
                continue
            name = token.feature.name
            if not name.startswith("inv:"):
                continue
            suffix = name[4:]
            if not suffix:
                continue
            item_name, sep, power_str = suffix.rpartition(":p")
            if not sep or not item_name or not power_str.isdigit():
                item_name = suffix
                power = 0
            else:
                power = int(power_str)
            value = int(token.value)
            if value <= 0:
                continue
            base = max(int(token.feature.normalization), 1)
            items[item_name] = items.get(item_name, 0) + value * (base ** power)
        return items

    def _team_resources(self, obs: _AgentObservation) -> dict[str, int]:
        """Read team resource levels from global ``team:*`` obs tokens.

        These are hub-level totals for each element, shared across agents.
        Used for EFE-optimal element selection (mine scarcest).
        """
        resources: dict[str, int] = {}
        for token in obs.tokens:
            name = token.feature.name
            if not name.startswith("team:"):
                continue
            elem = name[5:]
            item_name, sep, power_str = elem.rpartition(":p")
            if sep and item_name and power_str.isdigit():
                power = int(power_str)
                elem = item_name
            else:
                power = 0
            value = int(token.value)
            if value <= 0:
                continue
            base = max(int(token.feature.normalization), 1)
            resources[elem] = resources.get(elem, 0) + value * (base ** power)
        return resources

    # ------------------------------------------------------------------
    # Navigation (nav POMDP support methods)
    # ------------------------------------------------------------------

    def _closest_tag_location(
        self, obs: _AgentObservation, tag_ids: set[int]
    ) -> Optional[tuple[int, int]]:
        """Find closest entity with matching tag (egocentric coords)."""
        if not tag_ids:
            return None
        best_loc: Optional[tuple[int, int]] = None
        best_dist = 999
        for token in obs.tokens:
            if token.feature.name != "tag":
                continue
            if token.value not in tag_ids:
                continue
            loc = token.location
            if loc is None:
                continue
            dist = abs(loc[0] - self._center[0]) + abs(loc[1] - self._center[1])
            if dist < best_dist:
                best_dist = dist
                best_loc = loc
        return best_loc

    def _compute_nav_obs(
        self,
        target: Optional[tuple[int, int]],
        state: AIFBeliefState,
    ) -> tuple[int, int]:
        """Compute nav POMDP observations from current state.

        Returns (obs_range, obs_movement) as integer indices.
        """
        mem = state.spatial_memory

        # Target range
        if target is None or mem is None or mem.position is None:
            target_range = int(TargetRange.NO_TARGET)
            curr_dist = -1
        else:
            curr_dist = (abs(target[0] - mem.position[0])
                         + abs(target[1] - mem.position[1]))
            if curr_dist <= 1:
                target_range = int(TargetRange.ADJACENT)
            elif curr_dist <= 4:
                target_range = int(TargetRange.NEAR)
            else:
                target_range = int(TargetRange.FAR)

        # Nav progress (movement quality relative to target)
        prev_dist = state.prev_target_dist
        if prev_dist < 0 or curr_dist < 0:
            nav_progress = int(NavProgress.LATERAL)
        elif (mem is not None and len(mem.position_history) >= 2
              and mem.position_history[-1] == mem.position_history[-2]):
            nav_progress = int(NavProgress.BLOCKED)
        elif curr_dist < prev_dist:
            nav_progress = int(NavProgress.APPROACHING)
        elif curr_dist > prev_dist:
            nav_progress = int(NavProgress.RETREATING)
        else:
            nav_progress = int(NavProgress.LATERAL)

        # Update for next step
        state.prev_target_dist = curr_dist

        return (target_range, nav_progress)

    def _relative_to_absolute(
        self,
        nav_action: int,
        target: Optional[tuple[int, int]],
        state: AIFBeliefState,
    ) -> str:
        """Convert relative nav action to absolute direction.

        Uses target bearing to determine what TOWARD/LEFT/RIGHT/AWAY mean.
        """
        import random as _random

        mem = state.spatial_memory

        # Compute bearing to target
        if target is not None and mem is not None and mem.position is not None:
            dr = target[0] - mem.position[0]
            dc = target[1] - mem.position[1]

            if dr == 0 and dc == 0:
                bearing_dir = state.last_heading
            elif abs(dr) >= abs(dc):
                bearing_dir = "move_south" if dr > 0 else "move_north"
            else:
                bearing_dir = "move_east" if dc > 0 else "move_west"
        else:
            bearing_dir = state.last_heading

        bearing_idx = _BEARING_DIRS.index(bearing_dir)

        if nav_action == NavAction.TOWARD:
            direction = _BEARING_DIRS[bearing_idx]
        elif nav_action == NavAction.LEFT:
            direction = _BEARING_DIRS[(bearing_idx + 3) % 4]  # CCW
        elif nav_action == NavAction.RIGHT:
            direction = _BEARING_DIRS[(bearing_idx + 1) % 4]  # CW
        elif nav_action == NavAction.AWAY:
            direction = _BEARING_DIRS[(bearing_idx + 2) % 4]
        else:  # RANDOM
            direction = _random.choice(_BEARING_DIRS)

        # Wall safety: skip if wall-adjacent (but not at dist<=1, that's bumping)
        if mem is not None and mem.is_wall_adjacent(direction):
            if target is not None and mem.position is not None:
                dist = (abs(target[0] - mem.position[0])
                        + abs(target[1] - mem.position[1]))
                if dist <= 1:
                    state.last_heading = direction
                    return direction
            # Try alternatives: right, left, away
            for offset in [1, 3, 2]:
                alt = _BEARING_DIRS[(bearing_idx + offset) % 4]
                if not mem.is_wall_adjacent(alt):
                    state.last_heading = alt
                    return alt
            return "noop"

        state.last_heading = direction
        return direction

    # Junction alignment radius constants (from TeamJunctionVariant)
    _HUB_ALIGN_RADIUS = 25
    _NET_ALIGN_RADIUS = 15

    def _resolve_junction_target(
        self,
        obs: _AgentObservation,
        mem: Optional[SpatialMemory],
    ) -> Optional[tuple[int, int]]:
        """Find the nearest junction within alignment range.

        Junction alignment requires the junction to be within 25 tiles of
        the team hub or 15 tiles of an existing network junction.  We target
        junctions satisfying this constraint first, falling back to the
        nearest junction if none are in range (to at least move toward
        junction-dense areas).
        """
        hub_pos = mem.hub_offset if mem is not None else None
        tag_ids = self._junction_tags

        # Helper: check if a position (spawn-relative) is within alignment range
        def _in_range(pos: tuple[int, int]) -> bool:
            if hub_pos is None:
                return True  # Can't filter without hub knowledge
            return (abs(pos[0] - hub_pos[0]) + abs(pos[1] - hub_pos[1])
                    <= self._HUB_ALIGN_RADIUS)

        # 1. Try visible junctions — prefer those in range
        visible_candidates = []
        if mem is not None and mem.position is not None:
            for token in obs.tokens:
                if token.feature.name != "tag":
                    continue
                if token.value not in tag_ids:
                    continue
                loc = token.location
                if loc is None:
                    continue
                abs_r = mem.position[0] + loc[0] - self._center[0]
                abs_c = mem.position[1] + loc[1] - self._center[1]
                visible_candidates.append((abs_r, abs_c))

        if visible_candidates:
            in_range = [p for p in visible_candidates if _in_range(p)]
            candidates = in_range if in_range else visible_candidates
            # Return nearest candidate
            my_pos = mem.position
            return min(candidates, key=lambda p: abs(p[0] - my_pos[0]) + abs(p[1] - my_pos[1]))

        # 2. Try own spatial memory — prefer junctions in hub range
        if mem is not None and mem.position is not None and hub_pos is not None:
            station_pos = mem.find_nearest_station(
                "junction", ref_pos=hub_pos,
                max_ref_dist=self._HUB_ALIGN_RADIUS,
            )
            if station_pos is not None:
                return station_pos

        # 3. Try shared spatial memory (hub-relative coords, hub=(0,0))
        if mem is not None and mem.position is not None:
            shared = self._engine.shared_memory
            shared_pos = mem.to_shared(mem.position)
            if shared_pos is not None:
                station_shared = shared.find_nearest_station(
                    "junction", shared_pos,
                    max_hub_dist=self._HUB_ALIGN_RADIUS,
                )
                if station_shared is not None:
                    station_local = mem.from_shared(station_shared)
                    if station_local is not None:
                        return station_local

        # 4. Fallback: nearest junction regardless of range
        if mem is not None and mem.position is not None:
            station_pos = mem.find_nearest_station("junction")
            if station_pos is not None:
                return station_pos
            shared = self._engine.shared_memory
            shared_pos = mem.to_shared(mem.position)
            if shared_pos is not None:
                station_shared = shared.find_nearest_station("junction", shared_pos)
                if station_shared is not None:
                    station_local = mem.from_shared(station_shared)
                    if station_local is not None:
                        return station_local

        return None

    def _get_frontier_target(
        self, mem: SpatialMemory
    ) -> Optional[tuple[int, int]]:
        """Find nearest unexplored cell adjacent to explored territory."""
        if mem.position is None:
            return None

        frontiers: set[tuple[int, int]] = set()
        for (r, c) in mem.explored:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nb = (r + dr, c + dc)
                if nb not in mem.explored and nb not in mem.walls:
                    frontiers.add(nb)

        if frontiers:
            return min(
                frontiers,
                key=lambda f: (abs(f[0] - mem.position[0])
                               + abs(f[1] - mem.position[1]))
            )

        # Frontier exhausted → shared memory fallback
        return self._get_shared_fallback_target(mem)

    def _get_shared_fallback_target(
        self, mem: SpatialMemory
    ) -> Optional[tuple[int, int]]:
        """Role-appropriate fallback when local frontier is exhausted.

        Uses shared spatial memory (social epistemic inference) to find
        a meaningful target based on role:
        - Scout: least-explored direction (max epistemic value)
        - Miner: nearest extractor from shared knowledge
        - Aligner: nearest craft/junction from shared knowledge
        """
        if mem.position is None:
            return None

        shared = self._engine.shared_memory
        shared_pos = mem.to_shared(mem.position)
        if shared_pos is None:
            return None

        role = _agent_role(self._agent_id, self._engine.n_agents)

        if role == "scout":
            # Max epistemic value direction from shared explored cells
            target_shared = shared.find_least_explored_direction(shared_pos)
            if target_shared is not None:
                return mem.from_shared(target_shared)
        elif role == "miner":
            # Nearest extractor from shared knowledge
            for elem in RESOURCE_NAMES:
                cat = f"extractor:{elem}"
                station = shared.find_nearest_station(cat, shared_pos)
                if station is not None:
                    return mem.from_shared(station)
        else:  # aligner
            # Try junction first, then craft station
            for cat in ("junction", "craft"):
                station = shared.find_nearest_station(cat, shared_pos)
                if station is not None:
                    return mem.from_shared(station)

        return None

    def _action(self, name: str) -> _Action:
        """Create Action, falling back to noop if name is unavailable."""
        if name in self._action_name_set:
            return _Action(name=name)
        return _Action(name="noop")


# ---------------------------------------------------------------------------
# MultiAgentPolicy wrapper (requires mettagrid)
# ---------------------------------------------------------------------------

class AIFPolicy(_MultiAgentPolicy):
    """Deep active inference policy for CogsGuard (two nested POMDPs).

    Level 2: Strategic POMDP (288 states, 5 macro-options, ~42ms replan).
    Level 1: Option state machines map obs -> task policy (~0ms).
    Level 0: Navigation POMDP (16 states, 5 relative actions, ~3-5ms/step).

    All 8 agents share one BatchedAIFEngine (JIT-compiled, batched inference).
    4 miners (even), 3 aligners (odd<7), 1 scout (agent 7).
    """

    short_names = ["aif"]

    def __init__(
        self,
        policy_env_info: _PolicyEnvInterface,
        device: str = "cpu",
        n_agents: int = 8,
        **kwargs: Any,
    ):
        super().__init__(policy_env_info, device=device, **kwargs)

        # Build feature name list ordered by ID
        obs_features = policy_env_info.obs_features
        max_id = max((int(f.id) for f in obs_features), default=0)
        feat_names = [""] * (max_id + 1)
        for f in obs_features:
            feat_names[int(f.id)] = f.name

        # Build tag categories dynamically
        tag_categories = _build_tag_categories(policy_env_info.tags)

        self._discretizer = ObservationDiscretizer(feat_names, tag_categories)

        # Hierarchical engine: strategic POMDP + option state machines
        self._engine = BatchedAIFEngine(
            n_agents=n_agents, learn_B=True,
            policy_len=2, learn_interval=50,
        )
        self._agents: dict[int, _StatefulAgentPolicy] = {}

    def agent_policy(self, agent_id: int) -> _StatefulAgentPolicy:
        if agent_id not in self._agents:
            impl = AIFCogPolicyImpl(
                self._policy_env_info,
                agent_id,
                self._engine,
                self._discretizer,
            )
            self._agents[agent_id] = _StatefulAgentPolicy(
                impl, self._policy_env_info, agent_id=agent_id,
            )
        return self._agents[agent_id]

    def is_recurrent(self) -> bool:
        return True
