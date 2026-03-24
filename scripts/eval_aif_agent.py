#!/usr/bin/env python3
"""Evaluate the discrete AIF agent on CogsGuard.

Runs N episodes and reports hearts, junctions, inventory changes,
and factored belief state distributions.

Usage:
    python scripts/eval_aif_agent.py
    python scripts/eval_aif_agent.py --episodes 20 --agents 8
    python scripts/eval_aif_agent.py --model-path fitted_pomdp/arena.npz
    python scripts/eval_aif_agent.py --mission machina_1
"""

import argparse

import numpy as np

from cogames.cogs_vs_clips.clip_difficulty import EASY
from cogames.cogs_vs_clips.cog import CogTeam
from cogames.cogs_vs_clips.mission import CvCMission
from cogames.cogs_vs_clips.sites import COGSGUARD_ARENA
from mettagrid.envs.mettagrid_puffer_env import MettaGridPufferEnv
from mettagrid.envs.stats_tracker import StatsTracker
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Simulator
from mettagrid.util.stats_writer import NoopStatsWriter

from aif_meta_cogames.aif_agent.cogames_policy import AIFPolicy
from aif_meta_cogames.aif_agent.discretizer import Phase, Hand, TaskPolicy


PHASE_NAMES = [p.name for p in Phase]
HAND_NAMES = [h.name for h in Hand]
TASK_NAMES = [tp.name for tp in TaskPolicy]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate AIF agent on CogsGuard")
    p.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    p.add_argument("--agents", type=int, default=8, help="Agents per episode")
    p.add_argument("--max-steps", type=int, default=500, help="Steps per episode")
    p.add_argument("--model-path", type=str, default=None, help="Path to fitted POMDP .npz")
    p.add_argument("--mission", type=str, default="arena",
                   choices=["arena", "machina_1"],
                   help="Mission type")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_episode(policy_cfg, env_cfg, seed, max_steps, num_agents):
    """Run one episode and collect stats."""
    sim = Simulator()
    sim.add_event_handler(StatsTracker(NoopStatsWriter()))
    env = MettaGridPufferEnv(sim, env_cfg, seed=seed)

    pei = PolicyEnvInterface.from_mg_cfg(env.env_cfg)
    aif_policy = AIFPolicy(pei, model_path=policy_cfg.get("model_path"))

    # Reset env
    obs_raw, _ = env.reset()

    # Initialize agent policies
    agent_policies = []
    for i in range(num_agents):
        ap = aif_policy.agent_policy(i)
        ap.reset()
        agent_policies.append(ap)

    # Factored belief tracking
    phase_counts = np.zeros(len(Phase), dtype=int)
    hand_counts = np.zeros(len(Hand), dtype=int)
    task_counts = np.zeros(len(TaskPolicy), dtype=int)

    for step in range(max_steps):
        # Get observations for each agent
        observations = env.get_observations()

        actions = []
        for agent_id in range(num_agents):
            obs = observations[agent_id]
            action = agent_policies[agent_id].step(obs)
            actions.append(action)

        # Step environment
        env.step_agents(actions)

        # Track factored beliefs
        for agent_id in range(num_agents):
            ap = agent_policies[agent_id]
            state = ap._state
            if state is not None and state.qs is not None:
                phase_counts[state.last_phase] += 1
                hand_counts[state.last_hand] += 1
                task_counts[state.last_task_policy] += 1

    # Collect stats
    stats = env.get_episode_stats() if hasattr(env, 'get_episode_stats') else {}
    env.close()

    return {
        "phase_counts": phase_counts,
        "hand_counts": hand_counts,
        "task_counts": task_counts,
        "stats": stats,
    }


def main():
    args = parse_args()

    role_info = "miner/aligner split (even/odd agent_id)"
    print("=== AIF Agent Evaluation (Factored POMDP) ===")
    print(f"Episodes: {args.episodes}")
    print(f"Agents: {args.agents}")
    print(f"Max steps: {args.max_steps}")
    print(f"Mission: {args.mission}")
    print(f"Model: {'hand-crafted' if args.model_path is None else args.model_path}")
    print(f"Roles: {role_info}")
    print()

    # Create mission
    if args.mission == "machina_1":
        from cogames.cogs_vs_clips.sites import COGSGUARD_MACHINA_1
        site = COGSGUARD_MACHINA_1
    else:
        site = COGSGUARD_ARENA

    mission = CvCMission(
        name="aif_eval",
        description="AIF agent evaluation",
        site=site,
        num_cogs=args.agents,
        max_steps=args.max_steps,
        teams={
            "cogs": CogTeam(
                name="cogs",
                num_agents=args.agents,
                wealth=3,
                initial_hearts=0,
            )
        },
        variants=[EASY],
    )
    env_cfg = mission.make_env()

    policy_cfg = {"model_path": args.model_path}

    # Accumulators
    all_phase = np.zeros(len(Phase), dtype=int)
    all_hand = np.zeros(len(Hand), dtype=int)
    all_task = np.zeros(len(TaskPolicy), dtype=int)

    for ep in range(args.episodes):
        try:
            result = run_episode(
                policy_cfg, env_cfg,
                seed=args.seed + ep,
                max_steps=args.max_steps,
                num_agents=args.agents,
            )
            all_phase += result["phase_counts"]
            all_hand += result["hand_counts"]
            all_task += result["task_counts"]

            stats = result.get("stats", {})
            junctions = stats.get("aligned.junctions", 0)
            hearts = stats.get("hearts", 0)
            print(f"  Episode {ep+1}/{args.episodes}: "
                  f"junctions={junctions}, hearts={hearts}")
        except Exception as e:
            print(f"  Episode {ep+1} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Report
    total = all_phase.sum()
    print(f"\n=== Results ({args.episodes} episodes, {total:,} agent-steps) ===")

    print(f"\nPhase belief distribution:")
    for i, name in enumerate(PHASE_NAMES):
        count = all_phase[i]
        pct = 100.0 * count / max(total, 1)
        if pct > 0.5:
            print(f"  {name:12s}: {pct:5.1f}% ({count:,})")

    print(f"\nHand belief distribution:")
    for i, name in enumerate(HAND_NAMES):
        count = all_hand[i]
        pct = 100.0 * count / max(total, 1)
        print(f"  {name:16s}: {pct:5.1f}% ({count:,})")

    print(f"\nTask policy distribution:")
    for i, name in enumerate(TASK_NAMES):
        count = all_task[i]
        pct = 100.0 * count / max(total, 1)
        if pct > 0.5:
            print(f"  {name:15s}: {pct:5.1f}% ({count:,})")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
