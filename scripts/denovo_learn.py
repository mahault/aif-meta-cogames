#!/usr/bin/env python3
"""Phase B-V: De Novo Learning — Literal (Friston 2025).

Gradient-free approach to learning all POMDP parameters (A, B, C, D) from
trajectory data using Dirichlet accumulation and Bayesian Model Reduction.

Three stages:
1. Dirichlet accumulation: count observation-state and state-transition
   co-occurrences from trajectory beliefs to build concentration parameters.
2. Goal-conditioned C: identify observation patterns preceding rewards
   (junction captures) to set preference vectors.
3. Bayesian Model Reduction (BMR): analytically prune A/B connections
   that don't improve the variational free energy.

References:
    Friston et al. (2025). Gradient-Free De Novo Learning. Entropy 27(9):992.
    Friston et al. (2018). Bayesian Model Reduction. Neural Computation 30(8).

Usage::

    # Learn A, B, C, D from trajectory via Dirichlet accumulation
    python scripts/denovo_learn.py learn \\
        --trajectory /tmp/aif_traj.npz --output /tmp/denovo_params.npz

    # Learn + apply BMR pruning
    python scripts/denovo_learn.py learn \\
        --trajectory /tmp/aif_traj.npz --bmr --output /tmp/denovo_bmr.npz

    # Compare de novo params with gradient-learned params
    python scripts/learn_parameters.py compare \\
        --trajectory /tmp/aif_traj.npz --learned-params /tmp/denovo_params.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aif_meta_cogames.aif_agent.generative_model import (
    A_DEPENDENCIES,
    B_DEPENDENCIES,
    NUM_STATE_FACTORS,
    build_default_A,
    build_option_B,
    build_C,
    build_D,
    _agent_role,
)
from aif_meta_cogames.aif_agent.discretizer import NUM_OBS, NUM_OPTIONS


# ---------------------------------------------------------------------------
# Dirichlet accumulation for A (observation likelihood)
# ---------------------------------------------------------------------------

def accumulate_A(trajectory, n_agents=4, prior_scale=0.1):
    """Accumulate Dirichlet concentration parameters for A matrices.

    For each modality m with dependencies A_DEPENDENCIES[m]:
        alpha_A[m][o, s_deps] += sum_t I(o_m(t) = o) * prod_{d in deps} q(s_d, t)

    Parameters
    ----------
    trajectory : list of dicts
        Each dict has obs (6 x (N, 1)), qs (4 x (N, S_f)).
    n_agents : int
    prior_scale : float
        Dirichlet prior concentration (uniform). Higher = more regularization.

    Returns
    -------
    list of np.ndarray — Dirichlet concentration parameters (same shapes as A).
    """
    n_modalities = len(NUM_OBS)
    # Initialize with uniform prior
    alphas = []
    for m in range(n_modalities):
        dep_dims = tuple(NUM_STATE_FACTORS[f] for f in A_DEPENDENCIES[m])
        shape = (NUM_OBS[m],) + dep_dims
        alphas.append(np.full(shape, prior_scale))

    for record in trajectory:
        for agent_idx in range(n_agents):
            # Get beliefs: flatten to 1-D per factor
            qs = []
            for f in range(len(NUM_STATE_FACTORS)):
                q = np.asarray(record["qs"][f][agent_idx]).ravel()
                qs.append(q[-NUM_STATE_FACTORS[f]:] if len(q) > NUM_STATE_FACTORS[f]
                          else q)

            for m in range(n_modalities):
                obs_val = int(np.asarray(record["obs"][m][agent_idx]).ravel()[0])
                deps = A_DEPENDENCIES[m]

                # Compute outer product of dependency beliefs
                dep_beliefs = qs[deps[0]]
                for d in deps[1:]:
                    dep_beliefs = np.outer(dep_beliefs, qs[d]).ravel()
                dep_beliefs = dep_beliefs.reshape(
                    tuple(NUM_STATE_FACTORS[d] for d in deps))

                # Accumulate: one-hot obs × dep beliefs
                if obs_val < NUM_OBS[m]:
                    alphas[m][obs_val] += dep_beliefs

    return alphas


# ---------------------------------------------------------------------------
# Dirichlet accumulation for B (transition model)
# ---------------------------------------------------------------------------

def accumulate_B(trajectory, n_agents=4, prior_scale=0.1):
    """Accumulate Dirichlet concentration parameters for B matrices.

    For each factor f with dependencies B_DEPENDENCIES[f]:
        alpha_B[f][s', s_deps, a] += sum_t q(s'_f, t+1) * prod_{d in deps} q(s_d, t) * I(option=a)

    Requires consecutive timesteps (t, t+1) in trajectory.

    Parameters
    ----------
    trajectory : list of dicts
    n_agents : int
    prior_scale : float

    Returns
    -------
    list of np.ndarray — Dirichlet concentrations (option-level B shapes).
    """
    n_factors = len(NUM_STATE_FACTORS)
    alphas = []
    for f in range(n_factors):
        dep_dims = tuple(NUM_STATE_FACTORS[d] for d in B_DEPENDENCIES[f])
        shape = (NUM_STATE_FACTORS[f],) + dep_dims + (NUM_OPTIONS,)
        alphas.append(np.full(shape, prior_scale))

    for t in range(len(trajectory) - 1):
        record_t = trajectory[t]
        record_t1 = trajectory[t + 1]

        for agent_idx in range(n_agents):
            option = int(np.asarray(record_t["options"][agent_idx]).ravel()[0])
            if option >= NUM_OPTIONS:
                continue

            # Beliefs at t and t+1
            qs_t = []
            qs_t1 = []
            for f in range(n_factors):
                q = np.asarray(record_t["qs"][f][agent_idx]).ravel()
                qs_t.append(q[-NUM_STATE_FACTORS[f]:] if len(q) > NUM_STATE_FACTORS[f]
                            else q)
                q1 = np.asarray(record_t1["qs"][f][agent_idx]).ravel()
                qs_t1.append(q1[-NUM_STATE_FACTORS[f]:]
                             if len(q1) > NUM_STATE_FACTORS[f] else q1)

            for f in range(n_factors):
                deps = B_DEPENDENCIES[f]
                s_next = qs_t1[f]  # (S'_f,)

                # Dep beliefs at time t
                dep_beliefs = qs_t[deps[0]]
                for d in deps[1:]:
                    dep_beliefs = np.outer(dep_beliefs, qs_t[d]).ravel()
                dep_beliefs = dep_beliefs.reshape(
                    tuple(NUM_STATE_FACTORS[d] for d in deps))

                # Outer product: s_next × dep_beliefs → (S'_f, *dep_dims)
                contrib = np.einsum("i,...->i...", s_next, dep_beliefs)
                alphas[f][..., option] += contrib

    return alphas


# ---------------------------------------------------------------------------
# D accumulation (initial state prior)
# ---------------------------------------------------------------------------

def accumulate_D(trajectory, n_agents=4, prior_scale=0.1):
    """Accumulate Dirichlet parameters for D (initial state prior).

    Uses beliefs at t=0 from the trajectory.

    Returns
    -------
    list of np.ndarray — one (S_f,) Dirichlet per factor.
    """
    n_factors = len(NUM_STATE_FACTORS)
    alphas = [np.full(NUM_STATE_FACTORS[f], prior_scale) for f in range(n_factors)]

    # Use first timestep only
    if len(trajectory) > 0:
        record = trajectory[0]
        for agent_idx in range(n_agents):
            for f in range(n_factors):
                q = np.asarray(record["qs"][f][agent_idx]).ravel()
                q = q[-NUM_STATE_FACTORS[f]:] if len(q) > NUM_STATE_FACTORS[f] else q
                alphas[f] += q

    return alphas


# ---------------------------------------------------------------------------
# C from goal-state identification
# ---------------------------------------------------------------------------

def learn_C_goal_conditioned(trajectory, n_agents=4, reward_key="rewards",
                              window=5):
    """Learn C vectors by identifying observations preceding rewards.

    For each modality m, compute:
        C_m[o] ∝ log( freq(o near reward) / freq(o baseline) + eps )

    This identifies observations that are predictive of reward, providing
    a data-driven C without inverse EFE gradients.

    Parameters
    ----------
    trajectory : list of dicts
    n_agents : int
    reward_key : str
        Key in trajectory records for reward data. If absent, uses
        option changes as a proxy (replanning = achieved subgoal).
    window : int
        Number of steps before a reward event to count as "near reward".

    Returns
    -------
    list of np.ndarray — C vectors (6 modalities).
    """
    n_modalities = len(NUM_OBS)

    # Baseline observation counts
    baseline = [np.zeros(NUM_OBS[m]) + 1e-6 for m in range(n_modalities)]
    # Reward-conditioned observation counts
    reward_counts = [np.zeros(NUM_OBS[m]) + 1e-6 for m in range(n_modalities)]

    # Detect "reward events": option changes (proxy for subgoal completion)
    reward_steps = set()
    for agent_idx in range(n_agents):
        for t in range(len(trajectory) - 1):
            cur_opt = int(np.asarray(trajectory[t]["options"][agent_idx]).ravel()[0])
            nxt_opt = int(np.asarray(trajectory[t + 1]["options"][agent_idx]).ravel()[0])
            if cur_opt != nxt_opt:
                # The window before the option change
                for w in range(max(0, t - window), t + 1):
                    reward_steps.add((w, agent_idx))

    # Count observations
    for t, record in enumerate(trajectory):
        for agent_idx in range(n_agents):
            for m in range(n_modalities):
                obs_val = int(np.asarray(record["obs"][m][agent_idx]).ravel()[0])
                if obs_val < NUM_OBS[m]:
                    baseline[m][obs_val] += 1
                    if (t, agent_idx) in reward_steps:
                        reward_counts[m][obs_val] += 1

    # C = log-ratio (reward-conditioned vs baseline)
    C = []
    for m in range(n_modalities):
        ratio = reward_counts[m] / baseline[m]
        c_m = np.log(ratio + 1e-8)
        # Center (subtract mean) so preferences are relative
        c_m -= c_m.mean()
        C.append(c_m)

    return C


# ---------------------------------------------------------------------------
# Bayesian Model Reduction (BMR)
# ---------------------------------------------------------------------------

def bayesian_model_reduction(alphas, threshold=3.0):
    """Prune Dirichlet parameters using Bayesian Model Reduction.

    For each column of the Dirichlet (one conditional distribution),
    test if simplifying to uniform improves the free energy bound.

    The log Bayes factor for reducing column j from alpha to uniform:
        ln BF = sum_i [lgamma(alpha_i) - lgamma(alpha_reduced_i)]
              + lgamma(sum alpha_reduced) - lgamma(sum alpha)

    If ln BF > threshold, the reduced model is better (prune).

    Parameters
    ----------
    alphas : list of np.ndarray
        Dirichlet concentration parameters.
    threshold : float
        Log Bayes factor threshold for accepting reduction.
        3.0 ≈ strong evidence (Kass & Raftery, 1995).

    Returns
    -------
    pruned : list of np.ndarray — pruned concentration parameters.
    n_pruned : int — number of columns pruned.
    """
    from scipy.special import gammaln

    pruned = [a.copy() for a in alphas]
    total_pruned = 0

    for idx, alpha in enumerate(pruned):
        # Reshape to (first_dim, rest) for column-wise processing
        first_dim = alpha.shape[0]
        rest_shape = alpha.shape[1:]
        if len(rest_shape) == 0:
            continue  # Skip 1-D arrays (D vectors)

        n_cols = int(np.prod(rest_shape))
        alpha_2d = alpha.reshape(first_dim, n_cols)

        # Reduced model: uniform Dirichlet with same total concentration
        for col in range(n_cols):
            a_col = alpha_2d[:, col]
            total = a_col.sum()

            if total < 1.0:
                continue  # Too little evidence to prune

            # Reduced: uniform with same total
            a_reduced = np.full(first_dim, total / first_dim)

            # Log Bayes factor: reduced vs full
            ln_bf = (gammaln(a_reduced.sum()) - gammaln(total)
                     + np.sum(gammaln(a_col) - gammaln(a_reduced)))

            if ln_bf > threshold:
                # Reduced model is better — prune to uniform
                alpha_2d[:, col] = a_reduced
                total_pruned += 1

        pruned[idx] = alpha_2d.reshape(alpha.shape)

    return pruned, total_pruned


def dirichlet_expected_value(alphas):
    """Convert Dirichlet concentrations to expected probability matrices.

    E[Dir(alpha)] = alpha / sum(alpha) along axis 0.
    """
    result = []
    for alpha in alphas:
        total = alpha.sum(axis=0, keepdims=True)
        result.append(alpha / np.maximum(total, 1e-16))
    return result


# ---------------------------------------------------------------------------
# Full de novo pipeline
# ---------------------------------------------------------------------------

def denovo_learn(
    trajectory,
    n_agents=4,
    prior_scale=0.1,
    bmr=False,
    bmr_threshold=3.0,
    verbose=True,
):
    """Full de novo learning pipeline: Dirichlet accumulation + optional BMR.

    Parameters
    ----------
    trajectory : list of dicts
    n_agents : int
    prior_scale : float
    bmr : bool — apply Bayesian Model Reduction after accumulation
    bmr_threshold : float
    verbose : bool

    Returns
    -------
    dict with A, B, C, D, alphas_A, alphas_B, alphas_D, metadata.
    """
    if verbose:
        print(f"[denovo] Accumulating from {len(trajectory)} steps, "
              f"{n_agents} agents, prior_scale={prior_scale}")

    # Stage 1: Dirichlet accumulation
    alphas_A = accumulate_A(trajectory, n_agents, prior_scale)
    alphas_B = accumulate_B(trajectory, n_agents, prior_scale)
    alphas_D = accumulate_D(trajectory, n_agents, prior_scale)

    if verbose:
        for i, a in enumerate(alphas_A):
            print(f"  A[{i}] total evidence: {a.sum() - prior_scale * a.size:.1f}")
        for i, b in enumerate(alphas_B):
            print(f"  B[{i}] total evidence: {b.sum() - prior_scale * b.size:.1f}")

    # Stage 2: BMR pruning (optional)
    n_pruned_A = 0
    n_pruned_B = 0
    if bmr:
        alphas_A, n_pruned_A = bayesian_model_reduction(alphas_A, bmr_threshold)
        alphas_B, n_pruned_B = bayesian_model_reduction(alphas_B, bmr_threshold)
        if verbose:
            print(f"  BMR pruned: {n_pruned_A} A columns, {n_pruned_B} B columns")

    # Stage 3: Expected parameters
    A_learned = dirichlet_expected_value(alphas_A)
    B_learned = dirichlet_expected_value(alphas_B)
    D_learned = dirichlet_expected_value(alphas_D)

    # Stage 4: C from goal-conditioned counting
    C_learned = learn_C_goal_conditioned(trajectory, n_agents)
    if verbose:
        for i, c in enumerate(C_learned):
            print(f"  C[{i}]: {np.array2string(c, precision=2)}")

    return {
        "A": A_learned,
        "B": B_learned,
        "C": C_learned,
        "D": D_learned,
        "alphas_A": alphas_A,
        "alphas_B": alphas_B,
        "alphas_D": alphas_D,
        "metadata": {
            "method": "denovo_dirichlet",
            "prior_scale": prior_scale,
            "bmr": bmr,
            "bmr_threshold": bmr_threshold if bmr else None,
            "n_pruned_A": n_pruned_A,
            "n_pruned_B": n_pruned_B,
            "n_steps": len(trajectory),
            "n_agents": n_agents,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_learn(args):
    """Learn parameters via de novo Dirichlet accumulation."""
    # Import from learn_parameters for trajectory I/O
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from learn_parameters import load_trajectory, save_params

    trajectory = load_trajectory(args.trajectory)
    print(f"[denovo] Loaded trajectory: {len(trajectory)} steps")

    result = denovo_learn(
        trajectory,
        n_agents=args.n_agents,
        prior_scale=args.prior_scale,
        bmr=args.bmr,
        bmr_threshold=args.bmr_threshold,
        verbose=True,
    )

    save_params(
        args.output,
        A=result["A"],
        B=result["B"],
        C=result["C"],
        D=result["D"],
        metadata=result["metadata"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Phase B-V: De Novo Learning (Friston 2025)"
    )
    sub = parser.add_subparsers(dest="command")

    p_learn = sub.add_parser("learn",
                             help="Learn A/B/C/D via Dirichlet accumulation")
    p_learn.add_argument("--trajectory", required=True,
                         help="Path to trajectory .npz")
    p_learn.add_argument("--output", default="/tmp/denovo_params.npz")
    p_learn.add_argument("--n-agents", type=int, default=4)
    p_learn.add_argument("--prior-scale", type=float, default=0.1,
                         help="Dirichlet prior concentration (uniform)")
    p_learn.add_argument("--bmr", action="store_true",
                         help="Apply Bayesian Model Reduction after accumulation")
    p_learn.add_argument("--bmr-threshold", type=float, default=3.0,
                         help="Log Bayes factor threshold for BMR pruning")

    args = parser.parse_args()
    if args.command == "learn":
        cmd_learn(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
