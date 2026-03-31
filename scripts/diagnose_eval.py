#!/usr/bin/env python3
"""Diagnostic script: patches AIFCogPolicyImpl to log key decision points.

Run on AWS:
    python scripts/diagnose_eval.py

Logs to stdout every 100 steps for agent 0 (miner) and agent 1 (aligner):
- team_resources() output
- scarcest element selected
- NAV target resolution path (visible / memory / shared / fallback)
- has_role_gear state
- stuck detection
- disc_obs (o_res, o_sta, o_inv, ...)
- spatial memory stats
"""

import sys
import os

# Monkey-patch step_with_state for diagnostics
from aif_meta_cogames.aif_agent import cogames_policy as cp
from aif_meta_cogames.aif_agent.discretizer import TaskPolicy, ObsResource, ObsStation, ObsInventory

_orig_step = cp.AIFCogPolicyImpl.step_with_state
_orig_resolve = cp.AIFCogPolicyImpl._resolve_nav_target
_orig_team_res = cp.AIFCogPolicyImpl._team_resources

# Track per-agent diagnostics
_diag = {}


def _patched_team_resources(self, obs):
    result = _orig_team_res(self, obs)
    aid = self._agent_id
    if aid not in _diag:
        _diag[aid] = {}
    _diag[aid]["team_res"] = dict(result)
    return result


def _patched_resolve(self, task_policy, obs, state):
    aid = self._agent_id
    if aid not in _diag:
        _diag[aid] = {}

    mem = state.spatial_memory

    # Log what the category/tag_ids would be
    if task_policy == TaskPolicy.NAV_RESOURCE:
        team_res = self._team_resources(obs)
        _diag[aid]["team_res_in_resolve"] = dict(team_res)
        if team_res:
            from aif_meta_cogames.aif_agent.cogames_policy import RESOURCE_NAMES
            scarcest = min(RESOURCE_NAMES, key=lambda e: team_res.get(e, 0))
            _diag[aid]["scarcest"] = scarcest
            _diag[aid]["nav_res_path"] = "efe_optimal"
        else:
            _diag[aid]["scarcest"] = "N/A (no team_res)"
            _diag[aid]["nav_res_path"] = "fallback_generic"

    # Call original
    result = _orig_resolve(self, task_policy, obs, state)

    _diag[aid]["nav_target"] = result
    _diag[aid]["task_policy"] = TaskPolicy(task_policy).name

    # Memory stats
    if mem is not None:
        _diag[aid]["mem_stations"] = len(mem.stations)
        _diag[aid]["mem_walls"] = len(mem.walls)
        _diag[aid]["mem_explored"] = len(mem.explored)
        _diag[aid]["mem_pos"] = mem.position
        _diag[aid]["mem_stuck"] = mem.is_stuck()
        # Station categories in memory
        cats = {}
        for pos, cat in mem.stations.items():
            cats[cat] = cats.get(cat, 0) + 1
        _diag[aid]["mem_station_cats"] = cats

    # Shared memory stats
    shared = self._engine.shared_memory
    shared_cats = {}
    for pos, cat in shared.stations.items():
        shared_cats[cat] = shared_cats.get(cat, 0) + 1
    _diag[aid]["shared_station_cats"] = shared_cats
    _diag[aid]["shared_stations"] = len(shared.stations)

    return result


def _patched_step(self, obs, state):
    aid = self._agent_id
    if aid not in _diag:
        _diag[aid] = {}

    # Check what obs tokens are available (scan for team:* tokens)
    if state.step_count == 0 and aid == 0:
        # First step: log ALL feature names we see
        feat_names = set()
        for token in obs.tokens:
            feat_names.add(token.feature.name)
        team_tokens = [n for n in feat_names if n.startswith("team:")]
        inv_tokens = [n for n in feat_names if n.startswith("inv:")]
        print(f"\n[DIAG] Agent {aid} step 0: ALL team:* tokens = {sorted(team_tokens)}")
        print(f"[DIAG] Agent {aid} step 0: ALL inv:* tokens = {sorted(inv_tokens)}")
        print(f"[DIAG] Agent {aid} step 0: total feature names = {len(feat_names)}")

    action, state = _orig_step(self, obs, state)

    # Log every 100 steps for agents 0 and 1
    if state.step_count % 100 == 0 and aid in (0, 1):
        role = "aligner" if aid % 2 == 1 else "miner"
        d = _diag.get(aid, {})
        obs_array = self._obs_to_array(obs)
        disc = self._discretizer.discretize_obs(obs_array)

        print(f"\n[DIAG step={state.step_count}] agent={aid} ({role})")
        print(f"  disc_obs: o_res={ObsResource(disc[0]).name} o_sta={ObsStation(disc[1]).name} "
              f"o_inv={ObsInventory(disc[2]).name}")
        print(f"  has_role_gear={state.has_role_gear}")
        print(f"  task_policy={d.get('task_policy', '?')}")
        print(f"  team_res={d.get('team_res', '?')}")
        print(f"  scarcest={d.get('scarcest', '?')}")
        print(f"  nav_target={d.get('nav_target', '?')}")
        print(f"  nav_res_path={d.get('nav_res_path', '?')}")
        print(f"  mem: pos={d.get('mem_pos')} stations={d.get('mem_stations', 0)} "
              f"walls={d.get('mem_walls', 0)} explored={d.get('mem_explored', 0)} "
              f"stuck={d.get('mem_stuck', False)}")
        print(f"  mem_station_cats={d.get('mem_station_cats', {})}")
        print(f"  shared_stations={d.get('shared_stations', 0)} "
              f"shared_cats={d.get('shared_station_cats', {})}")
        print(f"  action={action.name}")

    return action, state


# Apply patches
cp.AIFCogPolicyImpl.step_with_state = _patched_step
cp.AIFCogPolicyImpl._resolve_nav_target = _patched_resolve
cp.AIFCogPolicyImpl._team_resources = _patched_team_resources

# Now run eval
if __name__ == "__main__":
    import subprocess
    # Run cogames eval with the patched module already loaded
    # We need to use the cogames eval entry point but with our patches active
    # Easiest: import and call directly
    from cogames.evaluate import main as eval_main
    sys.argv = [
        "cogames", "eval",
        "-p", "class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy",
        "-m", "arena",
        "-e", "1",
        "-s", "1000",
    ]
    eval_main()
