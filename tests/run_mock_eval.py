"""Run the full AIF hierarchy on the 8x8 mock CogsGuard grid.

Tests whether the two-POMDP hierarchy (strategic + navigation) can
complete the full economy chain without cogames/mettagrid.

The mock uses a **dict-based inventory** (bag of items) that mirrors
real CogsGuard: agents can hold gear AND resources simultaneously.
This is critical for testing the discretizer's HAS_GEAR vs HAS_RESOURCE
priority and the mine_cycle's deposit logic.

Usage:
    python tests/run_mock_eval.py
"""

import sys
sys.path.insert(0, "src")

import jax.numpy as jnp
import numpy as np

from aif_meta_cogames.aif_agent.discretizer import (
    NavAction,
    NavProgress,
    ObsContest,
    ObsInventory,
    ObsResource,
    ObsRoleSignal,
    ObsSocial,
    ObsStation,
    OPTION_NAMES,
    TASK_POLICY_NAMES,
    TargetRange,
    TaskPolicy,
)
from aif_meta_cogames.aif_agent.cogames_policy import (
    BatchedAIFEngine,
    RESOURCE_NAMES,
    SharedSpatialMemory,
    SpatialMemory,
    _BEARING_DIRS,
)
from aif_meta_cogames.aif_agent.generative_model import _agent_role

# ---------------------------------------------------------------------------
# Mock CogsGuard Environment (8x8)
# ---------------------------------------------------------------------------

STATIONS = {
    (0, 0): "extractor:carbon",
    (6, 7): "extractor:silicon",
    (2, 2): "hub",
    (4, 4): "craft",      # c:miner AND c:aligner (simplified)
    (2, 6): "junction",
}
GRID_SIZE = 8

STATION_OBS = {
    "hub": ObsStation.HUB,
    "craft": ObsStation.CRAFT,
    "junction": ObsStation.JUNCTION,
}

# Cargo capacity before deposit (dinky uses 40; mock uses 5 for speed)
CARGO_MAX = 5


class MockAgent:
    """Per-agent state for the mock evaluation.

    Uses dict-based inventory like real CogsGuard: agents can hold
    gear AND resources simultaneously (e.g. {miner: 1, carbon: 5}).
    """

    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.pos = (2, 2)  # start at hub
        self.inventory: dict[str, int] = {}  # bag of items
        self.prev_target_dist = -1
        self.last_heading = "move_east"
        self.prev_pos = (2, 2)
        self.just_deposited = False
        self.spatial_memory = SpatialMemory()
        self.spatial_memory.position = (2, 2)
        self.spatial_memory.explored.add((2, 2))
        # No pre-seeded knowledge: agents must discover stations by exploration

    def has_gear(self) -> bool:
        """Check if agent has role-specific gear."""
        role = _agent_role(self.agent_id, 2)
        role_gear = "aligner" if role == "aligner" else "miner"
        return self.inventory.get(role_gear, 0) > 0

    def has_resources(self) -> bool:
        """Check if agent has any minable resources."""
        return any(self.inventory.get(r, 0) > 0 for r in RESOURCE_NAMES)

    def has_hearts(self) -> bool:
        """Check if agent has hearts."""
        return self.inventory.get("heart", 0) > 0

    def resource_count(self) -> int:
        """Total count of minable resources in inventory."""
        return sum(self.inventory.get(r, 0) for r in RESOURCE_NAMES)


def step_env(agent: MockAgent, direction: str) -> dict:
    """Process one movement step. Returns info dict.

    Inventory is a dict: gear persists through mining/depositing.
    Mining adds resources to the bag. Depositing removes resources
    but keeps gear.
    """
    pos = agent.pos
    dr, dc = {
        "move_north": (-1, 0), "move_south": (1, 0),
        "move_east": (0, 1), "move_west": (0, -1),
        "noop": (0, 0),
    }.get(direction, (0, 0))

    new_r = max(0, min(GRID_SIZE - 1, pos[0] + dr))
    new_c = max(0, min(GRID_SIZE - 1, pos[1] + dc))
    new_pos = (new_r, new_c)

    info = {"moved": False, "interaction": None, "direction": direction}

    if new_pos in STATIONS and new_pos != pos:
        station = STATIONS[new_pos]
        if station.startswith("extractor") and agent.resource_count() < CARGO_MAX:
            # Mine: add resource to bag (keeps gear)
            elem = station.split(":")[1] if ":" in station else "carbon"
            agent.inventory[elem] = agent.inventory.get(elem, 0) + 1
            info["interaction"] = "mine"
            agent.just_deposited = False
        elif station == "hub":
            if agent.has_resources():
                # Deposit: remove resources, keep gear
                for r in RESOURCE_NAMES:
                    agent.inventory.pop(r, None)
                info["interaction"] = "deposit"
                agent.just_deposited = True
            elif not agent.just_deposited and not agent.has_hearts():
                # Withdraw hearts (simplified: always available)
                agent.inventory["heart"] = agent.inventory.get("heart", 0) + 1
                info["interaction"] = "withdraw_hearts"
                agent.just_deposited = False
        elif station == "craft":
            if agent.has_hearts():
                # Craft gear: consume hearts, get role gear
                agent.inventory.pop("heart", None)
                role = _agent_role(agent.agent_id, 2)
                role_gear = "aligner" if role == "aligner" else "miner"
                agent.inventory[role_gear] = 1
                info["interaction"] = "craft_gear"
                agent.just_deposited = False
        elif station == "junction" and agent.has_gear() and agent.has_hearts():
            # Capture: consume gear AND heart (like real CogsGuard)
            role = _agent_role(agent.agent_id, 2)
            role_gear = "aligner" if role == "aligner" else "miner"
            agent.inventory.pop(role_gear, None)
            agent.inventory["heart"] = agent.inventory.get("heart", 0) - 1
            if agent.inventory.get("heart", 0) <= 0:
                agent.inventory.pop("heart", None)
            info["interaction"] = "capture"
            agent.just_deposited = False
        # Stay at current position (bumped)
    elif new_pos != pos:
        agent.pos = new_pos
        info["moved"] = True
        agent.just_deposited = False
    # else: blocked by wall/edge — position unchanged

    # Update spatial memory
    agent.prev_pos = pos
    agent.spatial_memory.position = agent.pos
    agent.spatial_memory.position_history.append(agent.pos)
    if len(agent.spatial_memory.position_history) > 30:
        agent.spatial_memory.position_history.pop(0)
    # Explore nearby cells
    for edr in range(-1, 2):
        for edc in range(-1, 2):
            agent.spatial_memory.explored.add(
                (agent.pos[0] + edr, agent.pos[1] + edc)
            )

    # Discover stations by adjacency (realistic: no pre-seeded knowledge)
    for (sr, sc), stype in STATIONS.items():
        if abs(agent.pos[0] - sr) + abs(agent.pos[1] - sc) <= 2:
            agent.spatial_memory.stations[(sr, sc)] = stype
            if stype == "hub" and agent.spatial_memory.hub_offset is None:
                agent.spatial_memory.hub_offset = (sr, sc)

    return info


def discretize_obs(agent: MockAgent) -> list:
    """Convert agent state to 6 POMDP observations.

    Mirrors the real ObservationDiscretizer: gear takes priority
    over resources in HAS_GEAR / HAS_RESOURCE discretization.
    """
    pos = agent.pos

    # Resource: nearest extractor distance
    ext_positions = [(r, c) for (r, c), s in STATIONS.items() if s.startswith("extractor")]
    min_ext_dist = min(abs(pos[0] - r) + abs(pos[1] - c) for r, c in ext_positions)
    if min_ext_dist <= 1:
        obs_res = ObsResource.AT
    elif min_ext_dist <= 4:
        obs_res = ObsResource.NEAR
    else:
        obs_res = ObsResource.NONE

    # Station: check adjacent stations (dist <= 1)
    obs_sta = ObsStation.NONE
    for (sr, sc), stype in STATIONS.items():
        if stype.startswith("extractor"):
            continue
        dist = abs(pos[0] - sr) + abs(pos[1] - sc)
        if dist <= 1 and stype in STATION_OBS:
            obs_sta = STATION_OBS[stype]

    # Inventory: detect gear + resources held simultaneously
    has_gear = agent.has_gear()
    has_stuff = agent.has_resources() or agent.has_hearts()
    if has_gear and has_stuff:
        obs_inv = ObsInventory.HAS_BOTH
    elif has_gear:
        obs_inv = ObsInventory.HAS_GEAR
    elif has_stuff:
        obs_inv = ObsInventory.HAS_RESOURCE
    else:
        obs_inv = ObsInventory.EMPTY

    return [
        jnp.array([int(obs_res)]),
        jnp.array([int(obs_sta)]),
        jnp.array([int(obs_inv)]),
        jnp.array([int(ObsContest.FREE)]),
        jnp.array([int(ObsSocial.ALONE)]),
        jnp.array([int(ObsRoleSignal.SAME_ROLE)]),
    ]


def resolve_nav_target(agent: MockAgent, task_policy: int,
                       shared_memory: SharedSpatialMemory = None):
    """Resolve task policy to target using spatial memory (realistic).

    Uses agent's discovered stations first, then shared memory fallback,
    then frontier exploration. No direct STATIONS dict access.
    """
    mem = agent.spatial_memory
    pos = agent.pos

    # Map task policy to station category
    if task_policy == TaskPolicy.NAV_RESOURCE:
        category = "extractor"
    elif task_policy == TaskPolicy.NAV_DEPOT:
        category = "hub"
    elif task_policy in (TaskPolicy.NAV_CRAFT, TaskPolicy.NAV_GEAR):
        category = "craft"
    elif task_policy == TaskPolicy.NAV_JUNCTION:
        category = "junction"
    elif task_policy in (TaskPolicy.EXPLORE, TaskPolicy.YIELD):
        target = get_frontier(agent)
        if target is not None:
            return target
        # Frontier exhausted → shared memory fallback
        if shared_memory is not None and mem.hub_offset is not None:
            shared_pos = mem.to_shared(pos)
            if shared_pos is not None:
                target_shared = shared_memory.find_least_explored_direction(shared_pos)
                if target_shared is not None:
                    return mem.from_shared(target_shared)
        return None
    else:
        return None

    # 1. Try own spatial memory
    station = mem.find_nearest_station(category)
    if station is not None:
        return station

    # 2. Try shared memory (hub-relative conversion)
    if shared_memory is not None and mem.hub_offset is not None:
        shared_pos = mem.to_shared(pos)
        if shared_pos is not None:
            station_shared = shared_memory.find_nearest_station(category, shared_pos)
            if station_shared is not None:
                local = mem.from_shared(station_shared)
                if local is not None:
                    return local

    # 3. Fall back to frontier exploration
    return get_frontier(agent)


def get_frontier(agent: MockAgent):
    """Find nearest unexplored cell."""
    mem = agent.spatial_memory
    if mem.position is None:
        return None
    frontiers = set()
    for (r, c) in mem.explored:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (r + dr, c + dc)
            if (nb not in mem.explored and nb not in mem.walls
                    and 0 <= nb[0] < GRID_SIZE and 0 <= nb[1] < GRID_SIZE):
                frontiers.add(nb)
    if not frontiers:
        return None
    return min(frontiers,
               key=lambda f: abs(f[0] - mem.position[0]) + abs(f[1] - mem.position[1]))


def compute_nav_obs(agent: MockAgent, target):
    """Compute nav POMDP observations."""
    if target is None:
        agent.prev_target_dist = -1
        return int(TargetRange.NO_TARGET), int(NavProgress.LATERAL)

    curr_dist = abs(target[0] - agent.pos[0]) + abs(target[1] - agent.pos[1])

    if curr_dist <= 1:
        t_range = int(TargetRange.ADJACENT)
    elif curr_dist <= 4:
        t_range = int(TargetRange.NEAR)
    else:
        t_range = int(TargetRange.FAR)

    prev_dist = agent.prev_target_dist
    if prev_dist < 0:
        progress = int(NavProgress.LATERAL)
    elif agent.pos == agent.prev_pos:
        progress = int(NavProgress.BLOCKED)
    elif curr_dist < prev_dist:
        progress = int(NavProgress.APPROACHING)
    elif curr_dist > prev_dist:
        progress = int(NavProgress.RETREATING)
    else:
        progress = int(NavProgress.LATERAL)

    agent.prev_target_dist = curr_dist
    return t_range, progress


def relative_to_absolute(nav_action: int, target, agent: MockAgent) -> str:
    """Convert relative nav action to absolute direction."""
    import random

    if target is not None:
        dr = target[0] - agent.pos[0]
        dc = target[1] - agent.pos[1]
        if dr == 0 and dc == 0:
            bearing_dir = agent.last_heading
        elif abs(dr) >= abs(dc):
            bearing_dir = "move_south" if dr > 0 else "move_north"
        else:
            bearing_dir = "move_east" if dc > 0 else "move_west"
    else:
        bearing_dir = agent.last_heading

    bearing_idx = _BEARING_DIRS.index(bearing_dir)

    if nav_action == NavAction.TOWARD:
        direction = _BEARING_DIRS[bearing_idx]
    elif nav_action == NavAction.LEFT:
        direction = _BEARING_DIRS[(bearing_idx + 3) % 4]
    elif nav_action == NavAction.RIGHT:
        direction = _BEARING_DIRS[(bearing_idx + 1) % 4]
    elif nav_action == NavAction.AWAY:
        direction = _BEARING_DIRS[(bearing_idx + 2) % 4]
    else:
        direction = random.choice(_BEARING_DIRS)

    agent.last_heading = direction
    return direction


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval(n_agents: int = 2, max_steps: int = 500, verbose: bool = True):
    """Run the full AIF hierarchy on the mock environment."""
    print(f"\n{'='*60}")
    print(f"  AIF Mock Evaluation — {n_agents} agents, {max_steps} steps")
    print(f"  Grid: 8x8 | Stations: extractor:carbon(0,0) hub(2,2)")
    print(f"    craft(4,4) extractor:silicon(6,7) junction(2,6)")
    roles = [_agent_role(i, n_agents) for i in range(n_agents)]
    print(f"  Roles: {roles}")
    print(f"  Cargo capacity: {CARGO_MAX} (inventory is a bag of items)")
    print(f"{'='*60}\n")

    engine = BatchedAIFEngine(n_agents=n_agents)
    agents = [MockAgent(i) for i in range(n_agents)]
    shared_memory = SharedSpatialMemory()

    # Metrics
    metrics = {
        "mine": 0, "deposit": 0, "withdraw_hearts": 0,
        "craft_gear": 0, "capture": 0,
        "moves": 0, "noops": 0, "blocked": 0,
    }
    per_agent_metrics = [{k: 0 for k in metrics} for _ in range(n_agents)]

    for step in range(max_steps):
        for agent_id in range(n_agents):
            agent = agents[agent_id]

            # 1. Discretize observations
            jax_obs = discretize_obs(agent)

            # 2. Strategic POMDP → task policy
            task_policy = engine.submit_and_get_policy(agent_id, jax_obs)

            # 3. Handle noop policies
            if task_policy in (TaskPolicy.MINE, TaskPolicy.DEPOSIT,
                               TaskPolicy.CRAFT, TaskPolicy.ACQUIRE_GEAR,
                               TaskPolicy.CAPTURE, TaskPolicy.WAIT):
                info = step_env(agent, "noop")
                shared_memory.contribute(agent.spatial_memory)
                metrics["noops"] += 1
                per_agent_metrics[agent_id]["noops"] += 1
                continue

            # 4. Resolve nav target (using spatial memory, not direct STATIONS)
            target = resolve_nav_target(agent, task_policy, shared_memory)

            # 5. Compute nav observations
            obs_range, obs_movement = compute_nav_obs(agent, target)
            nav_obs = [jnp.array([obs_range]), jnp.array([obs_movement])]

            # 6. Nav POMDP → relative action
            nav_action = engine.submit_nav_and_get_action(agent_id, nav_obs)

            # 7. Convert to absolute direction
            direction = relative_to_absolute(nav_action, target, agent)

            # 8. Step environment
            info = step_env(agent, direction)

            # 8b. Contribute discoveries to shared memory (belief sharing)
            shared_memory.contribute(agent.spatial_memory)

            # 9. Detailed nav diagnostics
            nav_action_names = ["TOWARD", "LEFT", "RIGHT", "AWAY", "RANDOM"]
            range_names = ["ADJACENT", "NEAR", "FAR", "NO_TARGET"]
            progress_names = ["APPROACHING", "LATERAL", "RETREATING", "BLOCKED"]
            if verbose and (85 <= step <= 105 or (step % 50 == 0 and step > 0)):
                inv_str = str(agent.inventory) if agent.inventory else "{}"
                print(
                    f"  [nav] step={step} agent={agent_id} "
                    f"pos={agent.pos} target={target} "
                    f"range={range_names[obs_range]} progress={progress_names[obs_movement]} "
                    f"nav_act={nav_action_names[nav_action]} -> {direction} "
                    f"moved={info['moved']} inv={inv_str} "
                    f"task={TASK_POLICY_NAMES[task_policy]}"
                )

            # 10. Track metrics
            if info["interaction"]:
                metrics[info["interaction"]] += 1
                per_agent_metrics[agent_id][info["interaction"]] += 1
                if verbose:
                    role = _agent_role(agent_id, n_agents)
                    option = OPTION_NAMES[engine._current_options[agent_id]]
                    inv_str = str(agent.inventory) if agent.inventory else "{}"
                    print(
                        f"  step {step:3d} | agent {agent_id} ({role}) "
                        f"| {info['interaction']:15s} "
                        f"| inv={inv_str:30s} "
                        f"| pos={agent.pos} "
                        f"| option={option} "
                        f"| task={TASK_POLICY_NAMES[task_policy]}"
                    )

            if info["moved"]:
                metrics["moves"] += 1
            elif not info["interaction"]:
                metrics["blocked"] += 1

        # Periodic status
        if verbose and (step + 1) % 100 == 0:
            print(f"\n--- Step {step + 1} ---")
            for i in range(n_agents):
                role = _agent_role(i, n_agents)
                option = OPTION_NAMES[engine._current_options[i]]
                tp = TASK_POLICY_NAMES[engine._cached_policies[i]]
                inv_str = str(agents[i].inventory) if agents[i].inventory else "{}"
                print(
                    f"  Agent {i} ({role}): pos={agents[i].pos} "
                    f"inv={inv_str} option={option} task={tp}"
                )
            print()

    # Final report
    print(f"\n{'='*60}")
    print(f"  RESULTS ({max_steps} steps)")
    print(f"{'='*60}")
    print(f"  Carbon mined:      {metrics['mine']}")
    print(f"  Carbon deposited:  {metrics['deposit']}")
    print(f"  Hearts withdrawn:  {metrics['withdraw_hearts']}")
    print(f"  Gear crafted:      {metrics['craft_gear']}")
    print(f"  Junctions captured:{metrics['capture']}")
    print(f"  Moves:             {metrics['moves']}")
    print(f"  Noops:             {metrics['noops']}")
    print(f"  Blocked:           {metrics['blocked']}")
    print()

    for i in range(n_agents):
        role = "miner" if i % 2 == 0 else "aligner"
        m = per_agent_metrics[i]
        inv_str = str(agents[i].inventory) if agents[i].inventory else "{}"
        print(f"  Agent {i} ({role}): mine={m['mine']} deposit={m['deposit']} "
              f"hearts={m['withdraw_hearts']} gear={m['craft_gear']} "
              f"capture={m['capture']} noops={m['noops']} final_inv={inv_str}")

    chain_complete = metrics["capture"] > 0
    print(f"\n  ECONOMY CHAIN: {'COMPLETE' if chain_complete else 'INCOMPLETE'}")
    if not chain_complete:
        if metrics["craft_gear"] == 0:
            if metrics["withdraw_hearts"] == 0:
                print("  Bottleneck: No hearts withdrawn (aligner never reached hub)")
            else:
                print("  Bottleneck: No gear crafted (aligner never reached craft station)")
        else:
            print("  Bottleneck: No junction captured (aligner has gear but didn't reach junction)")
    print(f"{'='*60}\n")

    return metrics


if __name__ == "__main__":
    run_eval(n_agents=2, max_steps=500, verbose=True)
