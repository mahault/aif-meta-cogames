"""Live CogsGuard policy using discrete active inference (pymdp JAX).

Architecture:
- **BatchedAIFEngine** runs one pymdp Agent(batch_size=8) for all agents,
  JIT-compiled via eqx.filter_jit for ~5-10x speedup over sequential
- **pymdp** (JAX) handles belief tracking (posterior over 216 economy-chain
  states) and task-level policy selection via Expected Free Energy (EFE)
- **Navigator** converts the selected task policy into primitive movement

The POMDP action space is 13 task-level policies (not 5 primitive movements).
This makes B matrices action-dependent, so pymdp's EFE computation produces
meaningful, non-uniform policy selection.

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
    NUM_OBS,
    NUM_TASK_POLICIES,
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
# Batched AIF Engine (no mettagrid dependency)
# ---------------------------------------------------------------------------

def _batched_step(agent, batched_obs, empirical_prior, e_bias):
    """JIT-compilable batched inference step.

    Runs infer_states → infer_policies → sample_action →
    update_empirical_prior for all agents in one vectorized call.

    e_bias: (n_agents, n_policies) habit prior applied to q_pi.
            All-ones = no bias.
    """
    qs = agent.infer_states(batched_obs, empirical_prior=empirical_prior)
    q_pi, _efe = agent.infer_policies(qs)

    # Apply E-vector habit bias to policy posterior
    q_pi = q_pi * e_bias
    q_pi = q_pi / jnp.sum(q_pi, axis=-1, keepdims=True)

    sampled = agent.sample_action(q_pi)
    task_policies = sampled[:, 0]  # all factors share same action

    # Build pomdp_action: (n_agents, 4) — same action for all 4 factors
    pomdp_action = jnp.tile(sampled[:, :1], (1, 4))
    pred, _ = agent.update_empirical_prior(pomdp_action, qs)

    return task_policies, pred, qs, pomdp_action


class BatchedAIFEngine:
    """Batched active inference engine: 1 Agent(batch_size=N) for all agents.

    Agent 0's step triggers batched inference over all N agents.
    Agents 1-(N-1) return cached results (1-step lag, negligible for POMDP).
    Navigator stays per-agent (cheap, needs current obs).

    Learning mechanisms:
    - **B-learning**: Online Dirichlet updates to transition model (pB)
    - **C-from-reward**: Update C preferences from intrinsic reward signal
    - **E-vector**: Habit prior that reinforces successful task policies
    """

    def __init__(self, n_agents: int = 8, learn_B: bool = False,
                 learn_interval: int = 50, policy_len: int = 2,
                 learn_C: bool = False, learn_E: bool = False,
                 c_learning_rate: float = 0.1, e_learning_rate: float = 0.05,
                 c_update_interval: int = 200):
        self.n_agents = n_agents
        self.learn_B = learn_B
        self.learn_interval = learn_interval
        self.learn_C = learn_C
        self.learn_E = learn_E
        self.c_learning_rate = c_learning_rate
        self.e_learning_rate = e_learning_rate
        self.c_update_interval = c_update_interval
        self._step_count = 0

        self.agent = CogsGuardPOMDP.create_batched_agent(
            n_agents, learn_B=learn_B, policy_len=policy_len
        )

        # Obs buffer: per-agent, per-modality, shape (1,) = (T=1,)
        self._obs_buffer: list[list[Any]] = [
            [jnp.array([0]) for _ in range(6)]
            for _ in range(n_agents)
        ]

        # Beliefs managed here (shared across batch)
        self.empirical_prior = self.agent.D
        self.qs = None

        # Cached task policies from last batch
        self._cached_policies = [int(TaskPolicy.EXPLORE)] * n_agents

        # B-learning buffers
        self._prev_qs = None
        self._prev_obs = None
        self._prev_actions = None

        # C-from-reward and E-vector learning buffers
        if learn_C or learn_E:
            self._prev_discrete_obs = [[0] * 6 for _ in range(n_agents)]
            # Per-role (0=miner, 1=aligner) reward accumulators
            self._c_reward_sum = [np.zeros((2, n)) for n in NUM_OBS]
            self._c_reward_count = [np.zeros((2, n)) for n in NUM_OBS]
            # Per-role action reward tracking for E-vector
            self._e_action_reward_sum = np.zeros((2, NUM_TASK_POLICIES))
            self._e_action_reward_count = np.zeros((2, NUM_TASK_POLICIES))

        # E-vector habit bias: (n_agents, n_policies), all-ones = neutral
        n_policies = len(self.agent.policies)
        self._e_bias = jnp.ones((n_agents, n_policies))

        # JIT-compile the batched step function
        self._jit_step = eqx.filter_jit(_batched_step)

        # Warmup JIT compilation (avoids timeout on first eval step)
        dummy_obs = [jnp.zeros((n_agents, 1), dtype=jnp.int32) for _ in range(6)]
        self._jit_step(self.agent, dummy_obs, self.empirical_prior, self._e_bias)

    def submit_and_get_policy(self, agent_id: int, jax_obs: list) -> int:
        """Store obs for agent_id and return its task policy.

        Agent 0 triggers batched inference for all agents.
        Others return cached policies (1-step lag).
        """
        # Accumulate reward data for C/E learning
        if self.learn_C or self.learn_E:
            curr_obs_ints = [int(o[0]) for o in jax_obs]
            prev_obs_ints = self._prev_discrete_obs[agent_id]
            reward = self._compute_intrinsic_reward(curr_obs_ints, prev_obs_ints)
            self._accumulate_reward(agent_id, curr_obs_ints, reward)
            self._prev_discrete_obs[agent_id] = curr_obs_ints

        self._obs_buffer[agent_id] = jax_obs

        if agent_id == 0:
            self._run_batch()

        return self._cached_policies[agent_id]

    def get_beliefs(self, agent_id: int):
        """Return per-agent beliefs (qs) from the batched posterior."""
        if self.qs is None:
            return None
        # qs[f] shape: (batch, T, n_states_f) → index by agent_id
        return [q[agent_id] for q in self.qs]

    # ------------------------------------------------------------------
    # Intrinsic reward (economy chain progress)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_intrinsic_reward(curr_obs, prev_obs):
        """Compute intrinsic reward from observation transitions.

        Measures economy chain progress:
        - Inventory gain (picked up resource/gear) → high reward
        - Arriving at station → moderate reward
        - Junction contest improvement → highest reward
        - Idle (no obs change) → penalty
        """
        reward = 0.0

        # Inventory change (obs[2]): EMPTY=0, RESOURCE=1, GEAR=2
        inv_delta = curr_obs[2] - prev_obs[2]
        if inv_delta > 0:
            reward += 3.0 * inv_delta  # acquired resource or gear
        elif inv_delta < 0 and curr_obs[1] > 0:
            reward += 2.0  # deposited/used at station (productive)

        # Arrived at station (obs[1]): NONE=0, HUB=1, CRAFT=2, JUNCTION=3
        if curr_obs[1] > 0 and prev_obs[1] == 0:
            reward += 1.0 + curr_obs[1] * 0.5

        # Resource proximity (obs[0]): NONE=0, NEAR=1, AT=2
        if curr_obs[0] > prev_obs[0]:
            reward += 0.5

        # Contest improvement (obs[3]): FREE=0, CONTESTED=1, LOST=2
        if prev_obs[3] > 0 and curr_obs[3] < prev_obs[3]:
            reward += 5.0  # improved junction control
        elif curr_obs[3] > prev_obs[3]:
            reward -= 1.0

        # Idle penalty
        if curr_obs == prev_obs:
            reward -= 0.3

        return reward

    def _accumulate_reward(self, agent_id, obs_vals, reward):
        """Accumulate obs-reward and action-reward data for C/E learning."""
        role_idx = 0 if agent_id % 2 == 0 else 1  # 0=miner, 1=aligner

        # C-from-reward: which obs values correlate with reward
        if self.learn_C:
            for m in range(6):
                v = obs_vals[m]
                if v < NUM_OBS[m]:
                    self._c_reward_sum[m][role_idx, v] += reward
                    self._c_reward_count[m][role_idx, v] += 1

        # E-vector: which prior actions correlate with reward
        if self.learn_E and self._step_count > 0:
            prev_action = self._cached_policies[agent_id]
            if prev_action < NUM_TASK_POLICIES:
                self._e_action_reward_sum[role_idx, prev_action] += reward
                self._e_action_reward_count[role_idx, prev_action] += 1

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def _run_batch(self):
        """Run batched inference over all agents."""
        # Stack obs: (n_agents, T=1) per modality
        batched_obs = []
        for m in range(6):
            stacked = jnp.stack(
                [self._obs_buffer[i][m] for i in range(self.n_agents)]
            )
            batched_obs.append(stacked)

        policies, new_prior, qs, pomdp_action = self._jit_step(
            self.agent, batched_obs, self.empirical_prior, self._e_bias
        )

        self.empirical_prior = new_prior
        self.qs = qs
        self._cached_policies = [int(policies[i]) for i in range(self.n_agents)]

        # Online B-learning
        self._step_count += 1
        if (self.learn_B and self._prev_qs is not None
                and self._step_count % self.learn_interval == 0):
            self._update_B(batched_obs, qs, pomdp_action)

        # C-from-reward and E-vector updates
        if ((self.learn_C or self.learn_E)
                and self._step_count > 0
                and self._step_count % self.c_update_interval == 0):
            if self.learn_C:
                self._update_C_from_reward()
            if self.learn_E:
                self._update_E_from_reward()

        # Periodic logging (every 500 steps)
        if self._step_count % 500 == 0:
            actions = [int(policies[i]) for i in range(self.n_agents)]
            action_names = [TASK_POLICY_NAMES[a] for a in actions]
            extras = []
            if self.learn_B and hasattr(self.agent, 'pB') and self.agent.pB is not None:
                pB_sum = sum(float(pb.sum()) for pb in self.agent.pB)
                extras.append(f"pB={pB_sum:.0f}")
            if self.learn_E:
                e_range = float(self._e_bias.max() - self._e_bias.min())
                extras.append(f"E_range={e_range:.2f}")
            extra_str = f", {', '.join(extras)}" if extras else ""
            print(f"[AIF step={self._step_count}] actions={action_names}{extra_str}")

        # Store for next learning step
        self._prev_qs = qs
        self._prev_obs = batched_obs
        self._prev_actions = pomdp_action

    # ------------------------------------------------------------------
    # B-learning (Dirichlet updates)
    # ------------------------------------------------------------------

    def _update_B(self, curr_obs, curr_qs, curr_actions):
        """Update B matrices via online Dirichlet learning.

        Constructs a T=2 temporal buffer from previous and current beliefs,
        then calls pymdp's infer_parameters to update pB.
        """
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

    # ------------------------------------------------------------------
    # C-from-reward learning
    # ------------------------------------------------------------------

    def _update_C_from_reward(self):
        """Update C preferences from accumulated reward-observation correlations.

        For each modality m and obs value v, computes:
            C_new[m][v] = mean(reward when obs[m] == v)
        Then applies EMA: C = (1-lr) * C_old + lr * C_new
        """
        lr = self.c_learning_rate
        new_C = []

        for m in range(6):
            per_agent = []
            for i in range(self.n_agents):
                role_idx = 0 if i % 2 == 0 else 1
                old_c = np.array(self.agent.C[m][i])
                c_new = np.copy(old_c)

                for v in range(NUM_OBS[m]):
                    count = self._c_reward_count[m][role_idx, v]
                    if count > 5:  # minimum samples for reliable estimate
                        mean_reward = self._c_reward_sum[m][role_idx, v] / count
                        c_new[v] = (1 - lr) * old_c[v] + lr * mean_reward

                per_agent.append(c_new)
            new_C.append(jnp.array(np.stack(per_agent)))

        self.agent = eqx.tree_at(lambda a: a.C, self.agent, new_C)

        # Clear accumulators
        self._c_reward_sum = [np.zeros((2, n)) for n in NUM_OBS]
        self._c_reward_count = [np.zeros((2, n)) for n in NUM_OBS]

    # ------------------------------------------------------------------
    # E-vector habit learning
    # ------------------------------------------------------------------

    def _update_E_from_reward(self):
        """Update E-vector (habit prior) from action-reward correlations.

        Reinforces policies whose first action correlates with high reward.
        E[π] ∝ exp(mean_reward[first_action_of_π])
        Applied as multiplicative bias on q_pi in _batched_step.
        """
        n_policies = len(self.agent.policies)
        new_e = np.ones((self.n_agents, n_policies))

        for i in range(self.n_agents):
            role_idx = 0 if i % 2 == 0 else 1

            # Mean reward per action for this role
            action_values = np.zeros(NUM_TASK_POLICIES)
            for a in range(NUM_TASK_POLICIES):
                count = self._e_action_reward_count[role_idx, a]
                if count > 3:
                    action_values[a] = (
                        self._e_action_reward_sum[role_idx, a] / count
                    )

            # Map action values to policies (by first action)
            for pi_idx in range(n_policies):
                first_action = int(self.agent.policies.policy_arr[pi_idx, 0, 0])
                new_e[i, pi_idx] = np.exp(
                    np.clip(action_values[first_action], -5, 5)
                )

            # Normalize so mean = 1 (neutral baseline)
            new_e[i] /= new_e[i].mean()

        # EMA update
        lr = self.e_learning_rate
        old_e = np.array(self._e_bias)
        updated_e = (1 - lr) * old_e + lr * new_e
        self._e_bias = jnp.array(updated_e)

        # Clear accumulators
        self._e_action_reward_sum = np.zeros((2, NUM_TASK_POLICIES))
        self._e_action_reward_count = np.zeros((2, NUM_TASK_POLICIES))


# ---------------------------------------------------------------------------
# Per-agent policy implementation (requires mettagrid)
# ---------------------------------------------------------------------------

# Import mettagrid lazily — only when classes are actually used at runtime.
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
    2. Submit obs to BatchedAIFEngine → get task policy (JIT-batched)
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

        # 3. Convert to jax arrays — shape (T=1,) per modality
        jax_obs = [jnp.array([int(o)]) for o in disc_obs]

        # 4. Submit to batched engine → get task policy (JIT-compiled)
        task_policy = self._engine.submit_and_get_policy(self._agent_id, jax_obs)

        # 5. Execute selected task policy via navigator
        action, state = self._execute_task_policy(task_policy, obs, state)

        # 6. Track for logging — get beliefs from engine
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
            # At extractor — interact (noop to mine)
            return self._action("noop"), state

        elif task_policy == TaskPolicy.NAV_DEPOT:
            return self._navigate_to_tags(self._hub_tags, obs, state)

        elif task_policy == TaskPolicy.DEPOSIT:
            # At hub — interact (noop to deposit)
            return self._action("noop"), state

        elif task_policy == TaskPolicy.NAV_CRAFT:
            return self._navigate_to_tags(self._craft_tags, obs, state)

        elif task_policy == TaskPolicy.CRAFT:
            # At craft station — interact (noop to craft)
            return self._action("noop"), state

        elif task_policy == TaskPolicy.NAV_GEAR:
            return self._navigate_to_tags(self._craft_tags, obs, state)

        elif task_policy == TaskPolicy.ACQUIRE_GEAR:
            # At gear station — interact (noop to pick up)
            return self._action("noop"), state

        elif task_policy == TaskPolicy.NAV_JUNCTION:
            return self._navigate_to_tags(self._junction_tags, obs, state)

        elif task_policy == TaskPolicy.CAPTURE:
            # At junction — interact (noop to capture)
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
    """Discrete active inference policy for CogsGuard.

    Uses a 216-state POMDP (phase x hand x target_mode x role) with 13
    task-level policies as the action space.  pymdp JAX handles Bayesian
    belief tracking and EFE-driven task policy selection, combined with a
    navigator for spatial movement.

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

        # One shared engine for all agents (JIT-compiled, batched)
        self._engine = BatchedAIFEngine(
            n_agents=n_agents, learn_B=True, learn_C=True, learn_E=True,
            policy_len=2, learn_interval=50,
            c_learning_rate=0.1, e_learning_rate=0.05,
            c_update_interval=200,
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
