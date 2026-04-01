#!/usr/bin/env python3
"""Phase B-VI: Differentiable BMR + Model Comparison.

Adapts de novo learning ideas (Friston 2025) into the JAX differentiable
framework. Three capabilities:

1. **Differentiable BMR**: Identify and prune A/B entries with near-zero
   gradient norm (they don't contribute to VFE reduction). Iteratively
   prune + re-optimize until VFE stabilizes.

2. **De novo-initialized gradient learning**: Use Dirichlet-accumulated
   A/B from denovo_learn.py as initialization for gradient descent,
   combining Bayesian structure with gradient refinement.

3. **Model comparison**: Evaluate multiple parameter sets on held-out
   trajectory data via VFE (approximate log model evidence).

Usage::

    # Differentiable BMR: prune + re-optimize
    python scripts/differentiable_bmr.py prune \\
        --trajectory /tmp/aif_traj.npz --output /tmp/pruned_params.npz

    # De novo init + gradient refinement
    python scripts/differentiable_bmr.py refine \\
        --trajectory /tmp/aif_traj.npz \\
        --init-params /tmp/denovo_params.npz \\
        --output /tmp/refined_params.npz

    # Compare multiple parameter sets
    python scripts/differentiable_bmr.py compare \\
        --trajectory /tmp/aif_traj.npz \\
        --params default /tmp/denovo_params.npz /tmp/gradient_params.npz /tmp/full_params.npz

References:
    Friston et al. (2025). Gradient-Free De Novo Learning. Entropy 27(9):992.
    Friston et al. (2018). Bayesian Model Reduction. Neural Computation 30(8).
"""

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from learn_parameters import (
    params_to_logits,
    logits_to_params,
    trajectory_vfe,
    trajectory_vfe_multi_agent,
    load_trajectory,
    load_params,
    save_params,
    learn_from_trajectory,
)
from aif_meta_cogames.aif_agent.generative_model import (
    A_DEPENDENCIES,
    B_DEPENDENCIES,
    NUM_STATE_FACTORS,
    build_default_A,
    build_option_B,
    build_C,
    build_D,
)
from aif_meta_cogames.aif_agent.discretizer import NUM_OBS, NUM_OPTIONS


# ---------------------------------------------------------------------------
# Differentiable BMR: gradient-norm pruning
# ---------------------------------------------------------------------------

def compute_gradient_norms(
    A_logits, B_logits, trajectory, agent_indices,
):
    """Compute per-entry gradient norms for A and B logits.

    ||dVFE/dA[m][i,j]|| and ||dVFE/dB[f][i,j,k]|| for all entries.
    Entries with small gradient norm are candidates for pruning.

    Returns
    -------
    A_norms : list of np.ndarray — gradient magnitude per A entry
    B_norms : list of np.ndarray — gradient magnitude per B entry
    """
    # Wrap in closure so trajectory/agent_indices are captured (not traced)
    def _vfe(A_l, B_l):
        return trajectory_vfe_multi_agent(A_l, B_l, trajectory, agent_indices)
    grad_fn = jax.jit(jax.grad(_vfe, argnums=(0, 1)))
    grads_A, grads_B = grad_fn(A_logits, B_logits)

    A_norms = [np.abs(np.asarray(g)) for g in grads_A]
    B_norms = [np.abs(np.asarray(g)) for g in grads_B]
    return A_norms, B_norms


def prune_by_gradient(logits, grad_norms, threshold_percentile=10.0):
    """Prune logits entries with gradient norm below threshold.

    "Pruning" = setting the logit column to uniform (equal logits),
    which makes the distribution uninformative for that conditioning.

    Parameters
    ----------
    logits : list of jnp arrays
    grad_norms : list of np arrays — per-entry gradient magnitudes
    threshold_percentile : float
        Entries below this percentile of gradient norm get pruned.

    Returns
    -------
    pruned_logits : list of jnp arrays
    n_pruned : int — number of entries zeroed
    """
    all_norms = np.concatenate([n.ravel() for n in grad_norms])
    if len(all_norms) == 0:
        return logits, 0

    threshold = np.percentile(all_norms[all_norms > 0], threshold_percentile)
    if threshold <= 0:
        return logits, 0

    pruned = []
    total_pruned = 0
    for l, n in zip(logits, grad_norms):
        l_arr = np.array(l, copy=True)
        # For each column (axis 0 is the distribution axis), check if
        # the max gradient in that column is below threshold
        first_dim = l_arr.shape[0]
        rest = l_arr.reshape(first_dim, -1)
        norms_2d = np.array(n, copy=True).reshape(first_dim, -1)

        for col in range(rest.shape[1]):
            col_max_grad = norms_2d[:, col].max()
            if col_max_grad < threshold:
                # Set to uniform (all zeros in logit space → equal probs)
                rest[:, col] = 0.0
                total_pruned += 1

        pruned.append(jnp.array(rest.reshape(l_arr.shape)))

    return pruned, total_pruned


def differentiable_bmr(
    trajectory,
    n_rounds=3,
    prune_percentile=10.0,
    lr=0.001,
    refine_steps=100,
    n_agents=4,
    kl_weight=0.1,
    verbose=True,
):
    """Iterative differentiable BMR: prune + re-optimize cycle.

    1. Compute gradient norms for all A/B entries
    2. Prune entries below threshold percentile
    3. Re-optimize remaining parameters
    4. Repeat until VFE stabilizes or max rounds reached

    Parameters
    ----------
    trajectory : list of dicts
    n_rounds : int — max prune-optimize cycles
    prune_percentile : float — bottom N% of gradients get pruned
    lr : float — learning rate for re-optimization
    refine_steps : int — gradient steps per round
    n_agents : int
    kl_weight : float — regularization
    verbose : bool

    Returns
    -------
    dict with A, B, pruning_history, etc.
    """
    try:
        import optax
    except ImportError:
        print("optax not installed")
        sys.exit(1)

    A_logits = params_to_logits(build_default_A())
    B_logits = params_to_logits(build_option_B())
    agent_indices = list(range(n_agents))

    stride = max(1, len(trajectory) // 50)
    traj_sub = trajectory[::stride]

    history = []
    initial_vfe = float(trajectory_vfe_multi_agent(
        A_logits, B_logits, traj_sub, agent_indices))

    if verbose:
        print(f"[bmr] Initial VFE: {initial_vfe:.4f}, "
              f"{len(traj_sub)} steps, {n_rounds} rounds")

    for round_idx in range(n_rounds):
        # Step 1: Compute gradient norms
        A_norms, B_norms = compute_gradient_norms(
            A_logits, B_logits, traj_sub, agent_indices)

        # Step 2: Prune low-gradient entries
        A_logits, n_pruned_A = prune_by_gradient(
            A_logits, A_norms, prune_percentile)
        B_logits, n_pruned_B = prune_by_gradient(
            B_logits, B_norms, prune_percentile)

        vfe_after_prune = float(trajectory_vfe_multi_agent(
            A_logits, B_logits, traj_sub, agent_indices))

        if verbose:
            print(f"[bmr] Round {round_idx+1}: pruned A={n_pruned_A}, "
                  f"B={n_pruned_B}, VFE={vfe_after_prune:.4f}")

        # Step 3: Re-optimize remaining parameters
        optimizer = optax.adam(lr)
        opt_A = optimizer.init(A_logits)
        opt_B = optimizer.init(B_logits)

        A_ref = params_to_logits(build_default_A())
        B_ref = params_to_logits(build_option_B())

        def loss_fn(A_l, B_l):
            from learn_parameters import _kl_divergence_logits
            vfe = trajectory_vfe_multi_agent(A_l, B_l, traj_sub, agent_indices)
            if kl_weight > 0:
                vfe = vfe + kl_weight * (
                    _kl_divergence_logits(A_l, A_ref)
                    + _kl_divergence_logits(B_l, B_ref))
            return vfe

        grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))

        for step in range(refine_steps):
            gA, gB = grad_fn(A_logits, B_logits)
            uA, opt_A = optimizer.update(gA, opt_A)
            uB, opt_B = optimizer.update(gB, opt_B)
            A_logits = [a + u for a, u in zip(A_logits, uA)]
            B_logits = [b + u for b, u in zip(B_logits, uB)]

        vfe_after_refine = float(trajectory_vfe_multi_agent(
            A_logits, B_logits, traj_sub, agent_indices))

        history.append({
            "round": round_idx + 1,
            "pruned_A": n_pruned_A,
            "pruned_B": n_pruned_B,
            "vfe_after_prune": vfe_after_prune,
            "vfe_after_refine": vfe_after_refine,
        })

        if verbose:
            pct = (initial_vfe - vfe_after_refine) / abs(initial_vfe) * 100
            print(f"  After refine: VFE={vfe_after_refine:.4f} ({pct:+.1f}%)")

        # Early stop if no pruning happened
        if n_pruned_A == 0 and n_pruned_B == 0:
            if verbose:
                print(f"[bmr] No entries pruned — stopping early")
            break

    A_learned = [np.asarray(a) for a in logits_to_params(A_logits)]
    B_learned = [np.asarray(b) for b in logits_to_params(B_logits)]
    final_vfe = float(trajectory_vfe_multi_agent(
        A_logits, B_logits, traj_sub, agent_indices))

    return {
        "A": A_learned,
        "B": B_learned,
        "history": history,
        "initial_vfe": initial_vfe,
        "final_vfe": final_vfe,
    }


# ---------------------------------------------------------------------------
# De novo init + gradient refinement
# ---------------------------------------------------------------------------

def refine_from_denovo(
    trajectory,
    init_params,
    n_steps=200,
    lr=0.001,
    kl_weight=0.1,
    n_agents=4,
    verbose=True,
):
    """Gradient refinement starting from de novo (Dirichlet) initialization.

    Instead of starting from hand-crafted A/B, starts from Dirichlet-learned
    parameters. This combines Bayesian structure discovery with gradient
    optimization.

    Parameters
    ----------
    trajectory : list of dicts
    init_params : dict with "A" and "B" keys (from denovo_learn)
    n_steps, lr, kl_weight, n_agents, verbose : as in learn_from_trajectory

    Returns
    -------
    dict with A, B, vfe_history, etc.
    """
    return learn_from_trajectory(
        trajectory,
        n_steps=n_steps,
        lr=lr,
        kl_weight=kl_weight,
        multi_agent=True,
        n_agents=n_agents,
        verbose=verbose,
    )
    # Note: learn_from_trajectory always starts from defaults.
    # To start from denovo params, we need to override initialization.
    # For now, we implement a custom version below.


def refine_from_init(
    trajectory,
    A_init,
    B_init,
    n_steps=200,
    lr=0.001,
    kl_weight=0.0,
    n_agents=4,
    verbose=True,
):
    """Gradient refinement from custom A/B initialization.

    Parameters
    ----------
    A_init, B_init : lists of np arrays — starting parameters
    """
    try:
        import optax
    except ImportError:
        print("optax not installed")
        sys.exit(1)

    A_logits = params_to_logits(A_init)
    B_logits = params_to_logits(B_init)
    A_ref = params_to_logits(build_default_A())
    B_ref = params_to_logits(build_option_B())

    agent_indices = list(range(n_agents))
    stride = max(1, len(trajectory) // 50)
    traj_sub = trajectory[::stride]

    if verbose:
        print(f"[refine] {len(traj_sub)}/{len(trajectory)} steps, "
              f"lr={lr}, kl_weight={kl_weight}")

    def loss_fn(A_l, B_l):
        from learn_parameters import _kl_divergence_logits
        vfe = trajectory_vfe_multi_agent(A_l, B_l, traj_sub, agent_indices)
        if kl_weight > 0:
            vfe = vfe + kl_weight * (
                _kl_divergence_logits(A_l, A_ref)
                + _kl_divergence_logits(B_l, B_ref))
        return vfe

    initial_vfe = float(loss_fn(A_logits, B_logits))
    if verbose:
        print(f"[refine] Initial VFE: {initial_vfe:.4f}")

    grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))
    optimizer = optax.adam(lr)
    opt_A = optimizer.init(A_logits)
    opt_B = optimizer.init(B_logits)

    vfe_history = [initial_vfe]

    for step in range(n_steps):
        gA, gB = grad_fn(A_logits, B_logits)
        uA, opt_A = optimizer.update(gA, opt_A)
        uB, opt_B = optimizer.update(gB, opt_B)
        A_logits = [a + u for a, u in zip(A_logits, uA)]
        B_logits = [b + u for b, u in zip(B_logits, uB)]

        if (step + 1) % 20 == 0 or step == 0:
            cur = float(loss_fn(A_logits, B_logits))
            vfe_history.append(cur)
            if verbose:
                pct = (initial_vfe - cur) / abs(initial_vfe) * 100
                print(f"[refine] Step {step+1}: VFE={cur:.4f} ({pct:+.1f}%)")

    A_learned = [np.asarray(a) for a in logits_to_params(A_logits)]
    B_learned = [np.asarray(b) for b in logits_to_params(B_logits)]
    final_vfe = float(loss_fn(A_logits, B_logits))

    return {
        "A": A_learned,
        "B": B_learned,
        "vfe_history": vfe_history,
        "initial_vfe": initial_vfe,
        "final_vfe": final_vfe,
    }


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_multiple(
    trajectory,
    param_sets,
    n_agents=4,
    verbose=True,
):
    """Compare multiple parameter sets via VFE on trajectory data.

    Parameters
    ----------
    trajectory : list of dicts
    param_sets : dict mapping name -> {"A": [...], "B": [...]}
    n_agents : int
    verbose : bool

    Returns
    -------
    dict mapping name -> VFE, plus pairwise Bayes factors.
    """
    agent_indices = list(range(n_agents))
    results = {}

    for name, params in param_sets.items():
        A_logits = params_to_logits(params["A"])
        B_logits = params_to_logits(params["B"])
        vfe = float(trajectory_vfe_multi_agent(
            A_logits, B_logits, trajectory, agent_indices))
        results[name] = vfe
        if verbose:
            print(f"[compare] {name}: VFE = {vfe:.4f}")

    # Pairwise Bayes factors
    names = list(results.keys())
    if verbose and len(names) >= 2:
        print("\n[compare] Pairwise Bayes factors (BF > 3 = strong evidence):")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                n1, n2 = names[i], names[j]
                log_bf = results[n1] - results[n2]
                bf = float(np.exp(np.clip(log_bf, -50, 50)))
                winner = n2 if results[n2] < results[n1] else n1
                print(f"  {n1} vs {n2}: BF={bf:.2f} (favors {winner})")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_prune(args):
    """Run differentiable BMR."""
    trajectory = load_trajectory(args.trajectory)
    print(f"[bmr] Loaded: {len(trajectory)} steps")

    result = differentiable_bmr(
        trajectory,
        n_rounds=args.rounds,
        prune_percentile=args.prune_percentile,
        lr=args.lr,
        refine_steps=args.refine_steps,
        n_agents=args.n_agents,
        kl_weight=args.kl_weight,
        verbose=True,
    )

    save_params(
        args.output,
        A=result["A"],
        B=result["B"],
        C=build_C(),
        D=build_D(),
        metadata={
            "source": "differentiable_bmr",
            "initial_vfe": result["initial_vfe"],
            "final_vfe": result["final_vfe"],
            "history": json.dumps(result["history"]),
        },
    )


def cmd_refine(args):
    """Gradient refinement from de novo initialization."""
    trajectory = load_trajectory(args.trajectory)
    init = load_params(args.init_params)
    print(f"[refine] Loaded: {len(trajectory)} steps, "
          f"init from {args.init_params}")

    result = refine_from_init(
        trajectory,
        A_init=init["A"],
        B_init=init["B"],
        n_steps=args.steps,
        lr=args.lr,
        kl_weight=args.kl_weight,
        n_agents=args.n_agents,
        verbose=True,
    )

    save_params(
        args.output,
        A=result["A"],
        B=result["B"],
        C=build_C(),
        D=build_D(),
        metadata={
            "source": "denovo_gradient_refined",
            "initial_vfe": result["initial_vfe"],
            "final_vfe": result["final_vfe"],
        },
    )


def cmd_compare(args):
    """Compare multiple parameter sets."""
    trajectory = load_trajectory(args.trajectory)

    param_sets = {}
    param_sets["default"] = {
        "A": build_default_A(),
        "B": build_option_B(),
    }
    for path in args.params:
        name = Path(path).stem
        param_sets[name] = load_params(path)

    compare_multiple(trajectory, param_sets, n_agents=args.n_agents)


def main():
    parser = argparse.ArgumentParser(
        description="Phase B-VI: Differentiable BMR + Model Comparison"
    )
    sub = parser.add_subparsers(dest="command")

    # prune
    p_prune = sub.add_parser("prune", help="Differentiable BMR pruning")
    p_prune.add_argument("--trajectory", required=True)
    p_prune.add_argument("--output", default="/tmp/bmr_params.npz")
    p_prune.add_argument("--rounds", type=int, default=3)
    p_prune.add_argument("--prune-percentile", type=float, default=10.0)
    p_prune.add_argument("--lr", type=float, default=0.001)
    p_prune.add_argument("--refine-steps", type=int, default=100)
    p_prune.add_argument("--kl-weight", type=float, default=0.1)
    p_prune.add_argument("--n-agents", type=int, default=4)

    # refine
    p_refine = sub.add_parser("refine",
                              help="Gradient refine from de novo init")
    p_refine.add_argument("--trajectory", required=True)
    p_refine.add_argument("--init-params", required=True,
                          help="De novo params .npz")
    p_refine.add_argument("--output", default="/tmp/refined_params.npz")
    p_refine.add_argument("--steps", type=int, default=200)
    p_refine.add_argument("--lr", type=float, default=0.001)
    p_refine.add_argument("--kl-weight", type=float, default=0.0)
    p_refine.add_argument("--n-agents", type=int, default=4)

    # compare
    p_cmp = sub.add_parser("compare", help="Compare parameter sets via VFE")
    p_cmp.add_argument("--trajectory", required=True)
    p_cmp.add_argument("--params", nargs="+", required=True,
                       help="Paths to .npz parameter files")
    p_cmp.add_argument("--n-agents", type=int, default=4)

    args = parser.parse_args()
    if args.command == "prune":
        cmd_prune(args)
    elif args.command == "refine":
        cmd_refine(args)
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
