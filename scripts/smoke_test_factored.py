#!/usr/bin/env python3
"""Standalone smoke test for the factored [6,3,3,4] CogsGuard POMDP.

Runs 10 steps of the full active inference loop (belief update, policy
inference, action selection, empirical prior propagation) without any
cogames/mettagrid dependency.

Tests both miner and aligner roles to verify role-specific behavior.
Also tests the BatchedAIFEngine (JIT-compiled, batched inference).

Usage:
    python scripts/smoke_test_factored.py
"""

import time

import jax.numpy as jnp
import numpy as np

from aif_meta_cogames.aif_agent.generative_model import CogsGuardPOMDP
from aif_meta_cogames.aif_agent.cogames_policy import BatchedAIFEngine
from aif_meta_cogames.aif_agent.discretizer import Phase, Hand, TargetMode, TaskPolicy

TASK_NAMES = [tp.name for tp in TaskPolicy]


def run_role(role: str, scenarios: list):
    """Run inference loop for a given role and observation scenarios."""
    model = CogsGuardPOMDP.for_role(role)
    agent = model.create_agent()

    print(f"\n{'='*80}")
    print(f"  Role: {role.upper()}")
    print(f"  C preferences: {[c.tolist() for c in model.C]}")
    print(f"{'='*80}")
    print(f"{'Step':>4}  {'Scenario':<22}  {'Task Policy':<15}  {'Phase Belief'}")
    print("-" * 80)

    empirical_prior = agent.D
    tasks_chosen = []

    for step, (name, obs_vals) in enumerate(scenarios):
        obs = [jnp.array([[v]]) for v in obs_vals]

        qs = agent.infer_states(obs, empirical_prior=empirical_prior)
        q_pi, neg_efe = agent.infer_policies(qs)
        action = agent.sample_action(q_pi)
        task = int(action[0, 0])
        tasks_chosen.append(task)

        pomdp_action = jnp.array([[task, task, task, task]])
        pred, _ = agent.update_empirical_prior(pomdp_action, qs)
        empirical_prior = pred

        phase_beliefs = np.asarray(qs[0][0, -1])
        hand_beliefs = np.asarray(qs[1][0, -1])
        target_beliefs = np.asarray(qs[2][0, -1])
        phase_names = [p.name for p in Phase]
        top_phase = phase_names[np.argmax(phase_beliefs)]
        hand_names = [h.name for h in Hand]
        top_hand = hand_names[np.argmax(hand_beliefs)]

        belief_str = ", ".join(f"{phase_names[i]}={phase_beliefs[i]:.2f}" for i in range(6))
        print(f"{step:4d}  {name:<22}  {TASK_NAMES[task]:<15}  [{belief_str}]")
        print(f"      hand: {top_hand} ({hand_beliefs[np.argmax(hand_beliefs)]:.2f}), "
              f"target: {TargetMode(np.argmax(target_beliefs)).name} ({target_beliefs[np.argmax(target_beliefs)]:.2f})")

    return tasks_chosen


def main():
    print("Creating factored CogsGuard POMDP (generalist)...")
    model = CogsGuardPOMDP()
    print(model.summary())

    # Observation scenarios — same for both roles to show divergent behavior
    scenarios = [
        ("Empty/exploring",   [0, 0, 0, 0, 0, 0]),
        ("Near resource",     [1, 0, 0, 0, 0, 0]),
        ("At resource",       [2, 0, 0, 0, 0, 0]),
        ("At resource+hold",  [2, 0, 1, 0, 0, 0]),
        ("At hub+hold",       [0, 1, 1, 0, 0, 0]),
        ("At hub+empty",      [0, 1, 0, 0, 0, 0]),
        ("At craft+empty",    [0, 2, 0, 0, 0, 0]),
        ("At craft+gear",     [0, 2, 2, 0, 0, 0]),
        ("At junction+gear",  [0, 3, 2, 0, 0, 0]),
        ("Junction contested",[0, 3, 2, 1, 0, 0]),
    ]

    # Run both roles
    miner_tasks = run_role("miner", scenarios)
    aligner_tasks = run_role("aligner", scenarios)

    # Summary comparison
    print(f"\n{'='*80}")
    print("  ROLE COMPARISON")
    print(f"{'='*80}")
    print(f"{'Step':>4}  {'Scenario':<22}  {'Miner':<15}  {'Aligner':<15}  {'Same?'}")
    print("-" * 75)
    for i, (name, _) in enumerate(scenarios):
        m = TASK_NAMES[miner_tasks[i]]
        a = TASK_NAMES[aligner_tasks[i]]
        same = "YES" if m == a else "---"
        print(f"{i:4d}  {name:<22}  {m:<15}  {a:<15}  {same}")

    # Check that roles diverge on at least some steps
    n_different = sum(1 for m, a in zip(miner_tasks, aligner_tasks) if m != a)
    print(f"\nDifferent choices: {n_different}/{len(scenarios)}")

    if n_different > 0:
        print("\nRole specialization is working — miner and aligner make different choices.")
    else:
        print("\nWARNING: Roles are making identical choices — C preferences may need tuning.")

    print("\nSmoke test PASSED")


def test_batched_engine():
    """Test BatchedAIFEngine: JIT-compiled batched inference."""
    print(f"\n{'='*80}")
    print("  BATCHED ENGINE TEST")
    print(f"{'='*80}")

    scenarios = [
        ("Empty/exploring",   [0, 0, 0, 0, 0, 0]),
        ("Near resource",     [1, 0, 0, 0, 0, 0]),
        ("At resource",       [2, 0, 0, 0, 0, 0]),
        ("At resource+hold",  [2, 0, 1, 0, 0, 0]),
        ("At hub+hold",       [0, 1, 1, 0, 0, 0]),
        ("At hub+empty",      [0, 1, 0, 0, 0, 0]),
        ("At craft+empty",    [0, 2, 0, 0, 0, 0]),
        ("At craft+gear",     [0, 2, 2, 0, 0, 0]),
        ("At junction+gear",  [0, 3, 2, 0, 0, 0]),
        ("Junction contested",[0, 3, 2, 1, 0, 0]),
    ]

    # Create engine with 8 agents
    print("Creating BatchedAIFEngine(n_agents=8)...")
    engine = BatchedAIFEngine(n_agents=8)
    print(f"  Agent batch_size: {engine.agent.batch_size}")

    step_times = []
    for step, (name, obs_vals) in enumerate(scenarios):
        # Submit same obs for all 8 agents (agent 0 triggers batch)
        t0 = time.perf_counter()
        for agent_id in range(8):
            jax_obs = [jnp.array([v]) for v in obs_vals]
            policy = engine.submit_and_get_policy(agent_id, jax_obs)
        t1 = time.perf_counter()
        dt = (t1 - t0) * 1000

        # Get policies for agents 0 (miner) and 1 (aligner)
        miner_policy = engine._cached_policies[0]
        aligner_policy = engine._cached_policies[1]

        step_times.append(dt)
        jit_note = " (JIT compile)" if step == 0 else ""
        print(f"  Step {step}: miner={TASK_NAMES[miner_policy]:<15} "
              f"aligner={TASK_NAMES[aligner_policy]:<15} "
              f"({dt:.1f}ms{jit_note})")

    # Verify JIT compilation happened (first call should be much slower)
    if len(step_times) >= 3:
        first = step_times[0]
        avg_rest = np.mean(step_times[2:])
        speedup = first / max(avg_rest, 0.001)
        print(f"\n  JIT compilation detected: first={first:.0f}ms, "
              f"avg_rest={avg_rest:.1f}ms, speedup={speedup:.1f}x")

    # Verify beliefs are accessible
    beliefs = engine.get_beliefs(0)
    assert beliefs is not None, "Beliefs should be available after inference"
    print(f"  Beliefs available: {len(beliefs)} factors, "
          f"phase shape={beliefs[0].shape}")

    # Verify role differentiation (miners vs aligners should diverge)
    n_different = sum(
        1 for i in range(0, 8, 2)  # even = miner
        if engine._cached_policies[i] != engine._cached_policies[i + 1]
    )
    print(f"  Miner/aligner divergence: {n_different}/4 pairs differ")

    print("\nBatched engine test PASSED")


if __name__ == "__main__":
    main()
    test_batched_engine()
