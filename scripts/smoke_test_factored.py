#!/usr/bin/env python3
"""Standalone smoke test for the factored [6,3,3,4] CogsGuard POMDP.

Runs 10 steps of the full active inference loop (belief update, policy
inference, action selection, empirical prior propagation) without any
cogames/mettagrid dependency.

Tests both miner and aligner roles to verify role-specific behavior.

Usage:
    python scripts/smoke_test_factored.py
"""

import jax.numpy as jnp
import numpy as np

from aif_meta_cogames.aif_agent.generative_model import CogsGuardPOMDP
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


if __name__ == "__main__":
    main()
