"""Live CogsGuard policy using discrete active inference (pymdp JAX).

Architecture:
- **Hierarchical 2-level**: Level 2 (strategic POMDP, 5 macro-options) selects
  macro-options every 30-80 steps. Level 1 (option state machines) maps
  observation -> task policy reactively within each option.
- **BatchedAIFEngine** runs one pymdp Agent(batch_size=8) for all agents,
  JIT-compiled via eqx.filter_jit for ~5-10x speedup over sequential
- **pymdp** (JAX) handles belief tracking (posterior over 216 economy-chain
  states) and macro-option selection via Expected Free Energy (EFE)
- **OptionExecutor** converts macro-options + observations -> task policies
- **Navigator** converts the selected task policy into primitive movement

The POMDP action space is 5 macro-options (not 13 task policies or 5 movements).
This gives 25 two-step policies vs 169 previously, reducing infer_policies from
~280ms to ~42ms. Most steps only run belief update (~28ms), with full replanning
only at option termination (~70ms total).

Uses batched inference: one pymdp Agent(batch_size=8) with per-role C/D
vectors (even=miner, odd=aligner). Agent 0's step triggers batched inference
for all 8 agents; agents 1-7 use cached results (1-step lag).

Implements the cogames MultiAgentPolicy interface so it can be used directly:
    cogames eval -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy

Note: mettagrid has no Windows wheel. This module uses lazy imports so that
``_build_tag_categories`` and ``AIFBeliefState`` can be imported standalone
for testing, while the full policy classes require mettagrid at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .discretizer import (
    Hand,
    MacroOption,
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
    TaskPolicy,
    state_factors,
)
from .generative_model import CogsGuardPOMDP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEAR_NAMES = ("aligner", "scrambler", "miner", "scout")
RESOURCE_NAMES = ("carbon", "oxygen", "germanium", "silicon")
WANDER_DIRECTIONS = ("move_east", "move_south", "move_west", "move_north")
WANDER_STEPS = 8


# ---------------------------------------------------------------------------
# Mettagrid-independent utilities
# ---------------------------------------------------------------------------

def _build_tag_categories(tags: list[str]) -> dict[int, str]:
    """Build tag_value -> category mapping dynamically from tag names.

    More robust than hardcoded indices -- works across cogames versions.
    Does NOT require mettagrid.
    """
    categories: dict[int, str] = {}
    for i, tag_name in enumerate(tags):
        name = tag_name.removeprefix("type:")
        if "extractor" in name:
            categories[i] = "extractor"
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
        MacroOption.CRAFT_CYCLE: 60,
        MacroOption.CAPTURE_CYCLE: 80,
        MacroOption.EXPLORE: 30,
        MacroOption.DEFEND: 60,
    }

    def __init__(self, n_agents: int):
        self.n_agents = n_agents
        self.states = [OptionState() for _ in range(n_agents)]

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
            return self._defend(o_sta)
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
            # Deposit complete: had resource, now empty
            if st.prev_inv == ObsInventory.HAS_RESOURCE and o_inv == ObsInventory.EMPTY:
                return True
        elif option == MacroOption.CRAFT_CYCLE:
            # Gear acquired
            if o_inv == ObsInventory.HAS_GEAR:
                return True
        elif option == MacroOption.CAPTURE_CYCLE:
            # Gear used (had gear, now empty)
            if st.prev_inv == ObsInventory.HAS_GEAR and o_inv != ObsInventory.HAS_GEAR:
                return True
            # Started without gear -- bail after grace period
            if o_inv != ObsInventory.HAS_GEAR and st.steps_in_option > 5:
                return True
        elif option == MacroOption.EXPLORE:
            # Found resource or station
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
        """Activate a new option for this agent."""
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
        """MINE_CYCLE: NAV_RESOURCE -> MINE -> NAV_DEPOT -> DEPOSIT."""
        if o_inv == ObsInventory.EMPTY:
            if o_res >= ObsResource.AT:
                return TaskPolicy.MINE
            return TaskPolicy.NAV_RESOURCE
        elif o_inv == ObsInventory.HAS_RESOURCE:
            if o_sta == ObsStation.HUB:
                return TaskPolicy.DEPOSIT
            return TaskPolicy.NAV_DEPOT
        return TaskPolicy.NAV_RESOURCE

    @staticmethod
    def _craft_cycle(o_sta, o_inv):
        """CRAFT_CYCLE: NAV_CRAFT -> CRAFT -> acquire gear."""
        if o_inv == ObsInventory.HAS_GEAR:
            return TaskPolicy.WAIT
        if o_sta == ObsStation.CRAFT:
            return TaskPolicy.CRAFT
        return TaskPolicy.NAV_CRAFT

    @staticmethod
    def _capture_cycle(o_sta, o_inv):
        """CAPTURE_CYCLE: requires gear, NAV_JUNCTION -> CAPTURE."""
        if o_inv != ObsInventory.HAS_GEAR:
            return TaskPolicy.WAIT
        if o_sta == ObsStation.JUNCTION:
            return TaskPolicy.CAPTURE
        return TaskPolicy.NAV_JUNCTION

    @staticmethod
    def _defend(o_sta):
        """DEFEND: go to junction and hold it."""
        if o_sta == ObsStation.JUNCTION:
            return TaskPolicy.CAPTURE
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
                 learn_interval: int = 50, policy_len: int = 2):
        self.n_agents = n_agents
        self.learn_B = learn_B
        self.learn_interval = learn_interval
        self._step_count = 0

        # Level 2: Strategic agent with 5 macro-options
        self.agent = CogsGuardPOMDP.create_strategic_agent(
            n_agents, learn_B=learn_B, policy_len=policy_len
        )

        # Level 1: Option executor
        self.option_executor = OptionExecutor(n_agents)

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

        # JIT-compile both functions
        self._jit_belief_update = eqx.filter_jit(_belief_update)
        self._jit_select_option = eqx.filter_jit(_select_option)

        # Warmup JIT compilation (avoids timeout on first eval step)
        dummy_obs = [jnp.zeros((n_agents, 1), dtype=jnp.int32) for _ in range(6)]
        dummy_actions = jnp.full((n_agents, 4), 0, dtype=jnp.int32)
        pred, qs = self._jit_belief_update(
            self.agent, dummy_obs, self.empirical_prior, dummy_actions
        )
        self._jit_select_option(self.agent, qs)

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
            options, _q_pi = self._jit_select_option(self.agent, qs)
            for i in terminated:
                new_option = int(options[i])
                self._current_options[i] = new_option
                self.option_executor.set_option(i, new_option)
            # Update option actions for next belief update
            self._option_actions = jnp.tile(
                jnp.array(self._current_options)[:, None], (1, 4)
            )

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
            extra_str = f", {', '.join(extras)}" if extras else ""
            print(f"[AIF step={self._step_count}] options={option_names}")
            print(f"  tasks={task_names}{extra_str}")

        # Store for next B-learning step
        self._prev_qs = qs
        self._prev_obs = batched_obs
        self._prev_actions = self._option_actions

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

        self.agent = self.agent.infer_parameters(
            beliefs_A=beliefs_T2,
            observations=obs_T2,
            actions=actions_T1,
            beliefs_B=beliefs_T2,
            lr_pA=0.0,
            lr_pB=1.0,
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
        self._hub_tags = self._resolve_tag_ids(["hub"])
        self._craft_tags = self._resolve_tag_ids(
            [f"c:{g}" for g in GEAR_NAMES]
        )
        self._junction_tags = self._resolve_tag_ids(["junction"])
        self._heart_source_tags = self._resolve_tag_ids(["hub", "chest"])

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
        )

    def step_with_state(
        self, obs: _AgentObservation, state: AIFBeliefState
    ) -> tuple[_Action, AIFBeliefState]:
        """Run one step of the AIF agent."""
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
    # Task-policy execution (the navigator)
    # ------------------------------------------------------------------

    def _execute_task_policy(
        self,
        task_policy: int,
        obs: _AgentObservation,
        state: AIFBeliefState,
    ) -> tuple[_Action, AIFBeliefState]:
        """Execute a task-level policy by dispatching to the navigator.

        pymdp selects WHAT to do (task policy). This method handles HOW
        to do it (spatial movement toward the right target).
        """
        if task_policy == TaskPolicy.NAV_RESOURCE:
            return self._navigate_to_tags(self._extractor_tags, obs, state)

        elif task_policy == TaskPolicy.MINE:
            # At extractor -- interact (noop to mine)
            return self._action("noop"), state

        elif task_policy == TaskPolicy.NAV_DEPOT:
            return self._navigate_to_tags(self._hub_tags, obs, state)

        elif task_policy == TaskPolicy.DEPOSIT:
            # At hub -- interact (noop to deposit)
            return self._action("noop"), state

        elif task_policy == TaskPolicy.NAV_CRAFT:
            return self._navigate_to_tags(self._craft_tags, obs, state)

        elif task_policy == TaskPolicy.CRAFT:
            # At craft station -- interact (noop to craft)
            return self._action("noop"), state

        elif task_policy == TaskPolicy.NAV_GEAR:
            return self._navigate_to_tags(self._craft_tags, obs, state)

        elif task_policy == TaskPolicy.ACQUIRE_GEAR:
            # At gear station -- interact (noop to pick up)
            return self._action("noop"), state

        elif task_policy == TaskPolicy.NAV_JUNCTION:
            return self._navigate_to_tags(self._junction_tags, obs, state)

        elif task_policy == TaskPolicy.CAPTURE:
            # At junction -- interact (noop to capture)
            return self._action("noop"), state

        elif task_policy == TaskPolicy.EXPLORE:
            return self._wander(state)

        elif task_policy == TaskPolicy.YIELD:
            # Move away from nearest agent (simple: opposite of nearest)
            return self._wander(state)

        elif task_policy == TaskPolicy.WAIT:
            return self._action("noop"), state

        else:
            return self._wander(state)

    def _navigate_to_tags(
        self,
        tag_ids: set[int],
        obs: _AgentObservation,
        state: AIFBeliefState,
    ) -> tuple[_Action, AIFBeliefState]:
        """Navigate toward the closest entity with matching tags."""
        target_loc = self._closest_tag_location(obs, tag_ids)
        return self._move_toward(target_loc, state)

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

    # ------------------------------------------------------------------
    # Navigation (spatial movement)
    # ------------------------------------------------------------------

    def _closest_tag_location(
        self, obs: _AgentObservation, tag_ids: set[int]
    ) -> Optional[tuple[int, int]]:
        """Find closest entity with matching tag."""
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

    def _move_toward(
        self,
        target: Optional[tuple[int, int]],
        state: AIFBeliefState,
    ) -> tuple[_Action, AIFBeliefState]:
        """Move toward target location, or wander if no target visible."""
        if target is None:
            return self._wander(state)

        dr = target[0] - self._center[0]
        dc = target[1] - self._center[1]

        if dr == 0 and dc == 0:
            return self._action("noop"), state

        if abs(dr) >= abs(dc):
            direction = "move_south" if dr > 0 else "move_north"
        else:
            direction = "move_east" if dc > 0 else "move_west"

        return self._action(direction), state

    def _wander(self, state: AIFBeliefState) -> tuple[_Action, AIFBeliefState]:
        """Wander in a rectangular pattern when no target is visible."""
        if state.wander_steps <= 0:
            state.wander_dir = (state.wander_dir + 1) % len(WANDER_DIRECTIONS)
            state.wander_steps = WANDER_STEPS
        direction = WANDER_DIRECTIONS[state.wander_dir]
        state.wander_steps -= 1
        return self._action(direction), state

    def _action(self, name: str) -> _Action:
        """Create Action, falling back to noop if name is unavailable."""
        if name in self._action_name_set:
            return _Action(name=name)
        return _Action(name="noop")


# ---------------------------------------------------------------------------
# MultiAgentPolicy wrapper (requires mettagrid)
# ---------------------------------------------------------------------------

class AIFPolicy(_MultiAgentPolicy):
    """Hierarchical active inference policy for CogsGuard.

    Level 2: 216-state POMDP (phase x hand x target_mode x role) with 5
    macro-options.  Replans at option termination (~every 30-80 steps).
    Level 1: Reactive option state machines map obs -> task policy.
    Level 0: Navigator converts task policy -> primitive movement.

    All 8 agents share one BatchedAIFEngine (JIT-compiled, batched inference).
    Even agents are miners, odd agents are aligners (per-role C/D vectors).
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
