#!/usr/bin/env python3
"""Evaluate the discrete AIF agent on CogsGuard.

Runs N episodes and reports hearts, junctions, inventory changes,
and belief state distributions.

Usage:
    python scripts/eval_aif_agent.py
    python scripts/eval_aif_agent.py --episodes 20 --agents 8
    python scripts/eval_aif_agent.py --model-path fitted_pomdp/arena.npz
"""

import argparse
import sys

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
from aif_meta_cogames.aif_agent.discretizer import Phase, Hand, state_label


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate AIF agent on CogsGuard")
    p.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    p.add_argument("--agents", type=int, default=4, help="Agents per episode")
    p.add_argument("--max-steps", type=int, default=500, help="Steps per episode")
    p.add_argument("--model-path", type=str, default=None, help="Path to fitted POMDP .npz")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_episode(policy, env_cfg, seed, max_steps, num_agents):
    """Run one episode and collect stats."""
    sim = Simulator()
    sim.add_event_handler(StatsTracker(NoopStatsWriter()))
    env = MettaGridPufferEnv(sim, env_cfg, seed=seed)

    pei = PolicyEnvInterface.from_mg_cfg(env.env_cfg)
    aif_policy = AIFPolicy(pei, model_path=policy.get("model_path"))

    # Reset env
    obs_raw, _ = env.reset()

    # Initialize agent policies
    agent_policies = []
    for i in range(num_agents):
        ap = aif_policy.agent_policy(i)
        ap.reset()
        agent_policies.append(ap)

    # Run episode
    belief_counts = np.zeros(18, dtype=int)

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

        # Track beliefs
        for agent_id in range(num_agents):
            ap = agent_policies[agent_id]
            impl = ap._base_policy
            state = ap._state
            if state is not None and hasattr(state, 'agent') and state.agent.qs is not None:
                best = int(np.argmax(state.agent.qs[0]))
                belief_counts[best] += 1

    # Collect stats
    stats = env.get_episode_stats() if hasattr(env, 'get_episode_stats') else {}
    env.close()

    return {
        "belief_counts": belief_counts,
        "stats": stats,
    }


def main():
    args = parse_args()

    print(f"=== AIF Agent Evaluation ===")
    print(f"Episodes: {args.episodes}")
    print(f"Agents: {args.agents}")
    print(f"Max steps: {args.max_steps}")
    print(f"Model: {'hand-crafted' if args.model_path is None else args.model_path}")
    print()

    # Create mission
    mission = CvCMission(
        name="aif_eval",
        description="AIF agent evaluation",
        site=COGSGUARD_ARENA,
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

    # Run episodes
    all_belief_counts = np.zeros(18, dtype=int)

    for ep in range(args.episodes):
        try:
            result = run_episode(
                policy_cfg, env_cfg,
                seed=args.seed + ep,
                max_steps=args.max_steps,
                num_agents=args.agents,
            )
            all_belief_counts += result["belief_counts"]
            print(f"  Episode {ep+1}/{args.episodes} complete")
        except Exception as e:
            print(f"  Episode {ep+1} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Report
    print(f"\n=== Results ({args.episodes} episodes) ===")
    print(f"\nBelief state distribution (across all agents × steps):")
    total = all_belief_counts.sum()
    if total > 0:
        for s in range(18):
            count = all_belief_counts[s]
            pct = 100.0 * count / total
            if pct > 0.5:  # Only show significant states
                print(f"  {state_label(s):25s}: {pct:5.1f}% ({count:,} steps)")

    print(f"\nTotal belief updates: {total:,}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
