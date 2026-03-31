#!/usr/bin/env python3
"""Phase B: Differentiable parameter learning for AIF agent.

Uses JAX gradients through pymdp's VFE computation to optimize A, B, C, D
matrices from collected trajectory data. The key insight: since pymdp 1.0 is
fully JAX-based, we can compute dVFE/d(theta) and update parameters via
gradient descent.

Two modes:
1. Online Dirichlet: Enable learn_A/learn_B on the pymdp Agent (Bayesian
   conjugate updates during evaluation).
2. Offline gradient: Collect trajectories, then optimize A/B/C via
   jax.grad(calc_vfe) + optax (Adam).

Usage::

    # Step 1: Collect trajectory data from eval
    python scripts/learn_parameters.py collect \\
        --episodes 5 --agents 4 --output /tmp/traj_v10.npz

    # Step 2: Learn parameters from trajectory
    python scripts/learn_parameters.py learn \\
        --trajectory /tmp/traj_v10.npz \\
        --lr 0.001 --steps 200 --output /tmp/learned_params.npz

    # Step 3: Evaluate with learned parameters
    python scripts/learn_parameters.py compare \\
        --default-params /tmp/default_params.npz \\
        --learned-params /tmp/learned_params.npz \\
        --trajectory /tmp/traj_v10.npz

References:
    - Da Costa et al. (2020): Active Inference on Discrete State-Spaces
    - Fountas et al. (2020): Deep Active Inference (differentiable AIF)
    - pymdp 1.0: inferactively-pymdp (JAX backend)
"""

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.nn
import numpy as np

# Ensure the project is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aif_meta_cogames.aif_agent.generative_model import (
    A_DEPENDENCIES,
    B_DEPENDENCIES,
    CogsGuardPOMDP,
    NUM_STATE_FACTORS,
    build_default_A,
    build_default_B,
    build_C,
    build_D,
    _agent_role,
)
from aif_meta_cogames.aif_agent.discretizer import NUM_OBS


# ---------------------------------------------------------------------------
# Softmax parameterization: unconstrained logits <-> valid distributions
# ---------------------------------------------------------------------------

def params_to_logits(matrices: list[np.ndarray]) -> list[jnp.ndarray]:
    """Convert probability matrices to unconstrained logits via log."""
    return [jnp.log(jnp.clip(jnp.array(m, dtype=jnp.float32), 1e-8, 1.0))
            for m in matrices]


def logits_to_params(logits: list[jnp.ndarray]) -> list[jnp.ndarray]:
    """Convert logits back to valid probability distributions via softmax.

    Softmax is applied along axis=0 (the observation/next-state dimension),
    ensuring each column sums to 1.
    """
    return [jax.nn.softmax(l, axis=0) for l in logits]


# ---------------------------------------------------------------------------
# VFE computation over a trajectory
# ---------------------------------------------------------------------------

def _compute_vfe_factored(qs, prior, obs, A):
    """Compute VFE for factored state/observation model.

    VFE = complexity - accuracy
        = KL[q(s) || p(s)] - E_q[ln p(o|s)]

    For factored states with A_dependencies:
    - complexity = sum_f KL[q(s_f) || prior_f]
    - accuracy = sum_m ln [sum_{s_deps} A_m(o_m | s_deps) * prod_{f in deps} q(s_f)]

    All inputs should be unbatched (single agent, single timestep).
    obs: list of (1,) int arrays (categorical obs indices).
    qs: list of (1, S_f) arrays — we squeeze the batch dim.
    prior: list of (1, S_f) arrays.
    A: list of factored A matrices.
    """
    eps = 1e-16

    # Flatten to 1-D: qs may be (1, 1, S_f) or (1, S_f) or (S_f,)
    qs_flat = [q.ravel()[-NUM_STATE_FACTORS[f]:] if q.size > NUM_STATE_FACTORS[f]
               else q.ravel() for f, q in enumerate(qs)]
    prior_flat = [p.ravel()[-NUM_STATE_FACTORS[f]:] if p.size > NUM_STATE_FACTORS[f]
                  else p.ravel() for f, p in enumerate(prior)]

    # Complexity: sum of KL divergences per factor
    complexity = jnp.float32(0.0)
    for f in range(len(qs_flat)):
        q_f = qs_flat[f]
        p_f = prior_flat[f]
        kl_f = jnp.sum(q_f * (jnp.log(q_f + eps) - jnp.log(p_f + eps)))
        complexity = complexity + kl_f

    # Accuracy: sum over modalities
    # Use one-hot indexing (JAX-compatible, no int() needed)
    accuracy = jnp.float32(0.0)
    for m in range(len(A)):
        o_m_idx = obs[m].ravel()[0]  # JAX integer array (no Python int)
        deps = A_DEPENDENCIES[m]
        n_obs_m = A[m].shape[0]

        # One-hot encode observation for JAX-compatible indexing
        one_hot = jax.nn.one_hot(o_m_idx, n_obs_m)  # (n_obs_m,)

        # Compute p(o_m | s_deps) = sum_obs A_m(obs, s_deps) * one_hot(obs)
        # = A_m[o_m, s_deps]  (but done via contraction, not indexing)
        a_weighted = jnp.tensordot(one_hot, A[m], axes=([0], [0]))
        # a_weighted now has shape (*dep_dims)

        # Contract with q factors for each dependency
        likelihood = a_weighted
        for i, f in enumerate(deps):
            likelihood = jnp.tensordot(likelihood, qs_flat[f], axes=([0], [0]))

        # likelihood is now a scalar (all dims contracted)
        accuracy = accuracy + jnp.log(likelihood + eps)

    return complexity - accuracy


def trajectory_vfe(
    A_logits: list[jnp.ndarray],
    B_logits: list[jnp.ndarray],
    trajectory: list[dict],
    agent_idx: int = 0,
) -> jnp.ndarray:
    """Compute total VFE over a trajectory for one agent.

    Parameters
    ----------
    A_logits : list of jnp arrays
        Unconstrained logits for A matrices (will be softmaxed).
    B_logits : list of jnp arrays
        Unconstrained logits for B matrices (will be softmaxed).
    trajectory : list of dicts
        Each dict has keys: obs, qs, prior, actions (from BatchedAIFEngine).
    agent_idx : int
        Which agent's data to use (0-indexed).

    Returns
    -------
    jnp.ndarray
        Scalar total VFE summed over all timesteps.
    """
    A = logits_to_params(A_logits)
    # B logits kept for future use (transition VFE terms)
    _ = B_logits

    total_vfe = jnp.float32(0.0)

    for t, record in enumerate(trajectory):
        # Extract single-agent data: unbatched for VFE computation
        obs_t = [jnp.array(record["obs"][m][agent_idx:agent_idx+1])
                 for m in range(len(record["obs"]))]
        qs_t = [jnp.array(record["qs"][f][agent_idx:agent_idx+1])
                for f in range(len(record["qs"]))]
        prior_t = [jnp.array(record["prior"][f][agent_idx:agent_idx+1])
                   for f in range(len(record["prior"]))]

        vfe_t = _compute_vfe_factored(qs_t, prior_t, obs_t, A)
        total_vfe = total_vfe + vfe_t

    return total_vfe


# ---------------------------------------------------------------------------
# Gradient-based parameter optimization
# ---------------------------------------------------------------------------

def _kl_divergence_logits(logits: list[jnp.ndarray],
                          ref_logits: list[jnp.ndarray]) -> jnp.ndarray:
    """KL divergence between two sets of softmax-parameterized distributions.

    KL[P_ref || P_learned] summed over all modalities and columns.
    This penalizes deviation from the reference (default) parameters.
    """
    eps = 1e-16
    kl = jnp.float32(0.0)
    for l, r in zip(logits, ref_logits):
        p = jax.nn.softmax(l, axis=0)
        q = jax.nn.softmax(r, axis=0)
        kl = kl + jnp.sum(q * (jnp.log(q + eps) - jnp.log(p + eps)))
    return kl


def trajectory_vfe_multi_agent(
    A_logits: list[jnp.ndarray],
    B_logits: list[jnp.ndarray],
    trajectory: list[dict],
    agent_indices: list[int],
) -> jnp.ndarray:
    """Compute total VFE summed over MULTIPLE agents' trajectories.

    Prevents role-biased learning by including all roles in the loss.
    """
    total = jnp.float32(0.0)
    for idx in agent_indices:
        total = total + trajectory_vfe(A_logits, B_logits, trajectory, idx)
    return total / len(agent_indices)


def learn_from_trajectory(
    trajectory: list[dict],
    n_steps: int = 200,
    lr: float = 0.001,
    agent_idx: int = 0,
    verbose: bool = True,
    kl_weight: float = 0.0,
    multi_agent: bool = False,
    n_agents: int = 4,
) -> dict:
    """Learn A, B parameters from a trajectory via gradient descent on VFE.

    Uses softmax parameterization to maintain valid probability distributions.
    Optimizes with Adam via optax.

    Parameters
    ----------
    trajectory : list of dicts
        Trajectory data from BatchedAIFEngine.get_trajectory().
    n_steps : int
        Number of gradient descent steps.
    lr : float
        Learning rate for Adam optimizer.
    agent_idx : int
        Which agent's trajectory to learn from (ignored if multi_agent=True).
    verbose : bool
        Print progress.
    kl_weight : float
        Weight for KL regularization toward default parameters.
        0.0 = no regularization, 0.1 = mild, 1.0 = strong.
    multi_agent : bool
        If True, average VFE across ALL agents (prevents role bias).
    n_agents : int
        Number of agents (used when multi_agent=True).

    Returns
    -------
    dict with keys: A, B, vfe_history, initial_vfe, final_vfe
    """
    try:
        import optax
    except ImportError:
        print("optax not installed. Install with: pip install optax")
        sys.exit(1)

    # Initialize from default hand-crafted matrices
    A_default = build_default_A()
    B_default = build_default_B()

    A_logits = params_to_logits(A_default)
    B_logits = params_to_logits(B_default)

    # Keep reference logits for KL regularization (frozen)
    A_ref_logits = params_to_logits(A_default)
    B_ref_logits = params_to_logits(B_default)

    # Subsample trajectory for efficiency (every 10th step)
    stride = max(1, len(trajectory) // 50)
    traj_sub = trajectory[::stride]

    agent_indices = list(range(n_agents)) if multi_agent else [agent_idx]
    mode_str = f"multi-agent ({n_agents})" if multi_agent else f"agent={agent_idx}"
    reg_str = f", kl_weight={kl_weight}" if kl_weight > 0 else ""
    if verbose:
        print(f"[learn] Using {len(traj_sub)}/{len(trajectory)} steps "
              f"(stride={stride}), {mode_str}{reg_str}")

    # Define loss function
    def loss_fn(A_logits, B_logits):
        if multi_agent:
            vfe = trajectory_vfe_multi_agent(
                A_logits, B_logits, traj_sub, agent_indices)
        else:
            vfe = trajectory_vfe(A_logits, B_logits, traj_sub, agent_idx)
        if kl_weight > 0:
            kl_penalty = (_kl_divergence_logits(A_logits, A_ref_logits)
                          + _kl_divergence_logits(B_logits, B_ref_logits))
            vfe = vfe + kl_weight * kl_penalty
        return vfe

    # Compute initial VFE
    initial_vfe = float(loss_fn(A_logits, B_logits))
    if verbose:
        print(f"[learn] Initial VFE: {initial_vfe:.4f}")

    # Gradient function (no JIT — trajectory data must stay concrete)
    grad_fn = jax.grad(loss_fn, argnums=(0, 1))

    # Adam optimizer
    optimizer = optax.adam(lr)
    opt_state_A = optimizer.init(A_logits)
    opt_state_B = optimizer.init(B_logits)

    vfe_history = [initial_vfe]

    for step in range(n_steps):
        grads_A, grads_B = grad_fn(A_logits, B_logits)

        # Update A logits
        updates_A, opt_state_A = optimizer.update(grads_A, opt_state_A)
        A_logits = [a + u for a, u in zip(A_logits, updates_A)]

        # Update B logits
        updates_B, opt_state_B = optimizer.update(grads_B, opt_state_B)
        B_logits = [b + u for b, u in zip(B_logits, updates_B)]

        if (step + 1) % 20 == 0 or step == 0:
            current_vfe = float(loss_fn(A_logits, B_logits))
            vfe_history.append(current_vfe)
            if verbose:
                pct = (initial_vfe - current_vfe) / abs(initial_vfe) * 100
                print(f"[learn] Step {step+1}/{n_steps}: "
                      f"VFE={current_vfe:.4f} ({pct:+.1f}%)")

    # Final VFE
    final_vfe = float(loss_fn(A_logits, B_logits))
    vfe_history.append(final_vfe)

    # Convert back to probability distributions
    A_learned = [np.asarray(a) for a in logits_to_params(A_logits)]
    B_learned = [np.asarray(b) for b in logits_to_params(B_logits)]

    if verbose:
        pct = (initial_vfe - final_vfe) / abs(initial_vfe) * 100
        print(f"[learn] Final VFE: {final_vfe:.4f} ({pct:+.1f}% from initial)")

    return {
        "A": A_learned,
        "B": B_learned,
        "vfe_history": vfe_history,
        "initial_vfe": initial_vfe,
        "final_vfe": final_vfe,
    }


# ---------------------------------------------------------------------------
# Bayesian model comparison
# ---------------------------------------------------------------------------

def compare_models(
    trajectory: list[dict],
    params_1: dict,
    params_2: dict,
    agent_idx: int = 0,
    label_1: str = "default",
    label_2: str = "learned",
) -> dict:
    """Compare two parameter sets via VFE (approximate log model evidence).

    Lower VFE = better model fit. Bayes factor BF = exp(VFE_1 - VFE_2):
    if BF > 3, strong evidence for model 2.

    Parameters
    ----------
    trajectory : list of dicts
        Trajectory data to evaluate both models on.
    params_1, params_2 : dicts
        Each has keys "A", "B" (lists of numpy arrays).
    agent_idx : int
        Which agent's data to evaluate.
    label_1, label_2 : str
        Names for the two models.

    Returns
    -------
    dict with VFE for each model, Bayes factor, and winner.
    """
    A1_logits = params_to_logits(params_1["A"])
    B1_logits = params_to_logits(params_1["B"])
    A2_logits = params_to_logits(params_2["A"])
    B2_logits = params_to_logits(params_2["B"])

    vfe_1 = float(trajectory_vfe(A1_logits, B1_logits, trajectory, agent_idx))
    vfe_2 = float(trajectory_vfe(A2_logits, B2_logits, trajectory, agent_idx))

    # Bayes factor: BF = exp(VFE_1 - VFE_2)
    # (VFE is negative log evidence, so lower = better)
    log_bf = vfe_1 - vfe_2
    bf = float(np.exp(np.clip(log_bf, -50, 50)))

    winner = label_2 if vfe_2 < vfe_1 else label_1

    print(f"[compare] {label_1} VFE: {vfe_1:.4f}")
    print(f"[compare] {label_2} VFE: {vfe_2:.4f}")
    print(f"[compare] Bayes factor (favoring {label_2}): {bf:.2f}")
    if bf > 3:
        print(f"[compare] Strong evidence for {label_2}")
    elif bf > 1:
        print(f"[compare] Weak evidence for {label_2}")
    else:
        print(f"[compare] Evidence favors {label_1} (BF={1/bf:.2f})")

    return {
        f"vfe_{label_1}": vfe_1,
        f"vfe_{label_2}": vfe_2,
        "log_bayes_factor": log_bf,
        "bayes_factor": bf,
        "winner": winner,
    }


# ---------------------------------------------------------------------------
# Save/load helpers
# ---------------------------------------------------------------------------

def save_params(path: str, A: list, B: list, C: list = None, D: list = None,
                metadata: dict = None):
    """Save POMDP parameters to .npz file."""
    data = {}
    for i, a in enumerate(A):
        data[f"A_{i}"] = np.asarray(a)
    for i, b in enumerate(B):
        data[f"B_{i}"] = np.asarray(b)
    if C is not None:
        for i, c in enumerate(C):
            data[f"C_{i}"] = np.asarray(c)
    if D is not None:
        for i, d in enumerate(D):
            data[f"D_{i}"] = np.asarray(d)
    if metadata:
        data["metadata"] = np.array(json.dumps(metadata))
    np.savez_compressed(path, **data)
    print(f"[save] Parameters saved to {path}")


def load_params(path: str) -> dict:
    """Load POMDP parameters from .npz file."""
    data = np.load(path, allow_pickle=True)
    result = {"A": [], "B": [], "C": [], "D": []}
    for key in sorted(data.files):
        if key.startswith("A_"):
            result["A"].append(data[key])
        elif key.startswith("B_"):
            result["B"].append(data[key])
        elif key.startswith("C_"):
            result["C"].append(data[key])
        elif key.startswith("D_"):
            result["D"].append(data[key])
    return result


def save_trajectory(path: str, trajectory: list[dict]):
    """Save trajectory data to .npz file."""
    data = {}
    for t, record in enumerate(trajectory):
        for m, obs_m in enumerate(record["obs"]):
            data[f"obs_{t}_{m}"] = obs_m
        for f, qs_f in enumerate(record["qs"]):
            data[f"qs_{t}_{f}"] = qs_f
        for f, prior_f in enumerate(record["prior"]):
            data[f"prior_{t}_{f}"] = prior_f
        data[f"actions_{t}"] = record["actions"]
        data[f"options_{t}"] = record["options"]
    data["n_steps"] = np.array(len(trajectory))
    data["n_obs_modalities"] = np.array(len(trajectory[0]["obs"]))
    data["n_state_factors"] = np.array(len(trajectory[0]["qs"]))
    np.savez_compressed(path, **data)
    print(f"[save] Trajectory ({len(trajectory)} steps) saved to {path}")


def load_trajectory(path: str) -> list[dict]:
    """Load trajectory data from .npz file."""
    data = np.load(path, allow_pickle=True)
    n_steps = int(data["n_steps"])
    n_obs = int(data["n_obs_modalities"])
    n_factors = int(data["n_state_factors"])

    trajectory = []
    for t in range(n_steps):
        record = {
            "obs": [data[f"obs_{t}_{m}"] for m in range(n_obs)],
            "qs": [data[f"qs_{t}_{f}"] for f in range(n_factors)],
            "prior": [data[f"prior_{t}_{f}"] for f in range(n_factors)],
            "actions": data[f"actions_{t}"],
            "options": data[f"options_{t}"],
        }
        trajectory.append(record)
    return trajectory


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_learn(args):
    """Learn parameters from a saved trajectory."""
    trajectory = load_trajectory(args.trajectory)
    print(f"[learn] Loaded trajectory: {len(trajectory)} steps")

    result = learn_from_trajectory(
        trajectory,
        n_steps=args.steps,
        lr=args.lr,
        agent_idx=args.agent,
        verbose=True,
        kl_weight=args.kl_weight,
        multi_agent=args.multi_agent,
        n_agents=args.n_agents,
    )

    # Save learned parameters
    save_params(
        args.output,
        A=result["A"],
        B=result["B"],
        C=[np.asarray(c) for c in build_C()],
        D=[np.asarray(d) for d in build_D()],
        metadata={
            "initial_vfe": result["initial_vfe"],
            "final_vfe": result["final_vfe"],
            "vfe_reduction_pct": (result["initial_vfe"] - result["final_vfe"])
                                  / abs(result["initial_vfe"]) * 100,
            "n_steps": args.steps,
            "lr": args.lr,
            "agent_idx": args.agent,
            "kl_weight": args.kl_weight,
            "multi_agent": args.multi_agent,
        },
    )


def cmd_compare(args):
    """Compare default vs learned parameters."""
    trajectory = load_trajectory(args.trajectory)
    params_default = {
        "A": build_default_A(),
        "B": build_default_B(),
    }
    params_learned = load_params(args.learned_params)

    result = compare_models(
        trajectory, params_default, params_learned,
        agent_idx=args.agent,
        label_1="default", label_2="learned",
    )
    print(f"\n[compare] Result: {json.dumps(result, indent=2)}")


def cmd_save_default(args):
    """Save default hand-crafted parameters for comparison."""
    save_params(
        args.output,
        A=build_default_A(),
        B=build_default_B(),
        C=build_C(),
        D=build_D(),
        metadata={"source": "hand-crafted default"},
    )


def main():
    parser = argparse.ArgumentParser(
        description="Phase B: Differentiable parameter learning for AIF agent"
    )
    sub = parser.add_subparsers(dest="command")

    # learn subcommand
    p_learn = sub.add_parser("learn", help="Learn parameters from trajectory")
    p_learn.add_argument("--trajectory", required=True, help="Path to trajectory .npz")
    p_learn.add_argument("--output", default="/tmp/learned_params.npz")
    p_learn.add_argument("--steps", type=int, default=200)
    p_learn.add_argument("--lr", type=float, default=0.001)
    p_learn.add_argument("--agent", type=int, default=0)
    p_learn.add_argument("--kl-weight", type=float, default=0.0,
                         help="KL regularization toward default (0=none, 0.1=mild, 1.0=strong)")
    p_learn.add_argument("--multi-agent", action="store_true",
                         help="Average VFE across ALL agents (prevents role bias)")
    p_learn.add_argument("--n-agents", type=int, default=4,
                         help="Number of agents (for --multi-agent)")

    # compare subcommand
    p_compare = sub.add_parser("compare", help="Compare parameter sets via VFE")
    p_compare.add_argument("--trajectory", required=True)
    p_compare.add_argument("--learned-params", required=True)
    p_compare.add_argument("--agent", type=int, default=0)

    # save-default subcommand
    p_default = sub.add_parser("save-default", help="Save default parameters")
    p_default.add_argument("--output", default="/tmp/default_params.npz")

    args = parser.parse_args()
    if args.command == "learn":
        cmd_learn(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "save-default":
        cmd_save_default(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
