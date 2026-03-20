"""Live CogsGuard policy using discrete active inference (pymdp JAX).

Architecture:
- **pymdp** (JAX) handles belief tracking (posterior over 216 economy-chain
  states) and task-level policy selection via Expected Free Energy (EFE)
- **Navigator** converts the selected task policy into primitive movement

The POMDP action space is 13 task-level policies (not 5 primitive movements).
This makes B matrices action-dependent, so pymdp's EFE computation produces
meaningful, non-uniform policy selection.

The JAX Agent (equinox Module) is immutable — beliefs and empirical priors
are managed externally in ``AIFBeliefState``.

Implements the cogames MultiAgentPolicy interface so it can be used directly:
    cogames eval -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy

Or with a fitted model:
    cogames eval -p class=...AIFPolicy,kw.model_path=fitted_pomdp/arena.npz

Note: mettagrid has no Windows wheel. This module uses lazy imports so that
``_build_tag_categories`` and ``AIFBeliefState`` can be imported standalone
for testing, while the full policy classes require mettagrid at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

from .discretizer import (
    Hand,
    ObservationDiscretizer,
    Phase,
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
    """Per-agent state for the AIF policy.

    The JAX pymdp Agent is an immutable equinox Module, so beliefs and
    empirical priors must be managed externally. This dataclass holds:
    - The Agent (immutable, carries A/B/C/D matrices and policies)
    - The current empirical prior (for next infer_states call)
    - The latest posterior beliefs qs (for logging/inspection)
    """
    agent: Any  # pymdp.agent.Agent (JAX, immutable equinox Module)
    empirical_prior: Any = None  # list of jax arrays, shape (batch, n_states_f)
    qs: Any = None  # latest beliefs from infer_states
    step_count: int = 0
    wander_dir: int = 0
    wander_steps: int = WANDER_STEPS
    last_phase: int = 0
    last_hand: int = 0
    last_task_policy: int = TaskPolicy.EXPLORE


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
    2. Update beliefs via pymdp JAX (Bayesian inference over 216 states)
    3. Select task-level policy via EFE (13 task policies)
    4. Execute selected task policy via navigator (spatial movement)
    5. Update empirical prior for next timestep
    """

    def __init__(
        self,
        policy_env_info: _PolicyEnvInterface,
        agent_id: int,
        pomdp: CogsGuardPOMDP,
        discretizer: ObservationDiscretizer,
    ):
        self._agent_id = agent_id
        self._policy_env_info = policy_env_info
        self._pomdp = pomdp
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
        """Create initial state with a fresh pymdp JAX Agent."""
        agent = self._pomdp.create_agent(
            policy_len=2,
            use_states_info_gain=True,
            use_param_info_gain=False,
        )
        return AIFBeliefState(
            agent=agent,
            empirical_prior=agent.D,
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

        # 3. Convert to jax arrays with (batch, T) dims for JAX Agent
        jax_obs = [jnp.array([[int(o)]]) for o in disc_obs]

        # 4. Belief update via pymdp JAX
        qs = state.agent.infer_states(jax_obs, empirical_prior=state.empirical_prior)

        # 5. Policy inference — EFE over 13 task-level policies
        q_pi, efe = state.agent.infer_policies(qs)

        # 6. Select best task policy
        task_policy = int(jnp.argmax(q_pi[0]))

        # 7. Execute selected task policy via navigator
        action, state = self._execute_task_policy(task_policy, obs, state)

        # 8. Update empirical prior with selected task policy
        pomdp_action = jnp.array([[task_policy]])
        result = state.agent.update_empirical_prior(pomdp_action, qs)
        # pymdp alpha returns (pred, qs), v1.0.0 returns just pred
        state.empirical_prior = result[0] if isinstance(result, tuple) else result
        state.qs = qs

        # Track for logging
        beliefs = np.asarray(qs[0][0, -1, :])
        best_state = int(np.argmax(beliefs))
        state.last_phase, state.last_hand, _, _ = state_factors(best_state)
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

    Supports loading fitted A/B matrices via model_path kwarg.
    """

    short_names = ["aif"]

    def __init__(
        self,
        policy_env_info: _PolicyEnvInterface,
        device: str = "cpu",
        model_path: Optional[str] = None,
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

        # Load fitted model if path provided, otherwise hand-crafted
        if model_path:
            self._pomdp = CogsGuardPOMDP.from_fitted(model_path)
        else:
            self._pomdp = CogsGuardPOMDP()

        self._agents: dict[int, _StatefulAgentPolicy] = {}

    def agent_policy(self, agent_id: int) -> _StatefulAgentPolicy:
        if agent_id not in self._agents:
            impl = AIFCogPolicyImpl(
                self._policy_env_info,
                agent_id,
                self._pomdp,
                self._discretizer,
            )
            self._agents[agent_id] = _StatefulAgentPolicy(
                impl, self._policy_env_info, agent_id=agent_id,
            )
        return self._agents[agent_id]

    def is_recurrent(self) -> bool:
        return True
