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
import itertools
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
    build_option_B,
    build_C,
    build_C_miner,
    build_C_aligner,
    build_C_scout,
    build_D,
    _agent_role,
)
from aif_meta_cogames.aif_agent.discretizer import NUM_OBS, NUM_OPTIONS


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


def _compute_transition_loss_factored(qs_t, qs_t1, option_t, B,
                                       frozen_factors=None):
    """Compute transition prediction loss for factored B matrices.

    L_B = -sum_f sum_{s'} q(s'_f, t+1) * ln P_predicted(s'_f)

    where P_predicted(s'_f) = sum_{s_deps} B_f(s', s_deps, a) * prod_d q(s_d, t)

    This is the VFE complexity/transition term — measures how well B predicts
    the observed state transitions under the current beliefs.

    Parameters
    ----------
    qs_t : list of jnp arrays
        Posterior beliefs at time t (one per factor, may need flattening).
    qs_t1 : list of jnp arrays
        Posterior beliefs at time t+1.
    option_t : jnp scalar
        Macro-option active at time t (0-4).
    B : list of jnp arrays
        Option-level B matrices [B_phase, B_hand, B_target, B_role].
    frozen_factors : set, optional
        Factor indices to skip (default: {3} for role, which is always identity).

    Returns
    -------
    jnp.ndarray
        Scalar transition prediction loss.
    """
    if frozen_factors is None:
        frozen_factors = {3}  # Role never changes

    eps = 1e-16

    # Flatten beliefs to 1-D per factor (handles (1, S_f) or (1, 1, S_f))
    qs_t_flat = [q.ravel()[-NUM_STATE_FACTORS[f]:] if q.size > NUM_STATE_FACTORS[f]
                 else q.ravel() for f, q in enumerate(qs_t)]
    qs_t1_flat = [q.ravel()[-NUM_STATE_FACTORS[f]:] if q.size > NUM_STATE_FACTORS[f]
                  else q.ravel() for f, q in enumerate(qs_t1)]

    loss = jnp.float32(0.0)

    for f in range(len(B)):
        if f in frozen_factors:
            continue

        deps = B_DEPENDENCIES[f]
        B_f = B[f]
        n_options = B_f.shape[-1]

        # Select action slice via one-hot (JAX-compatible, no int indexing)
        action_oh = jax.nn.one_hot(option_t, n_options)
        # Contract action dim: B_f[..., option] -> B_f_a[s', deps...]
        B_f_a = jnp.tensordot(B_f, action_oh, axes=([-1], [0]))

        # Contract with q(s_deps_t) for each dependency factor
        # After action removal, shape is (S'_f, dep_0_dim, dep_1_dim, ...)
        # Each contraction removes axis 1 (the first dep dim)
        predicted = B_f_a
        for d in deps:
            predicted = jnp.tensordot(predicted, qs_t_flat[d], axes=([1], [0]))
        # predicted shape: (S'_f,) — distribution over next state for factor f

        # Cross-entropy: -sum q(s'_f, t+1) * ln P_predicted(s'_f)
        loss_f = -jnp.sum(qs_t1_flat[f] * jnp.log(predicted + eps))
        loss = loss + loss_f

    return loss


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
    B = logits_to_params(B_logits)

    total_vfe = jnp.float32(0.0)

    for t, record in enumerate(trajectory):
        # Extract single-agent data: unbatched for VFE computation
        obs_t = [jnp.array(record["obs"][m][agent_idx:agent_idx+1])
                 for m in range(len(record["obs"]))]
        qs_t = [jnp.array(record["qs"][f][agent_idx:agent_idx+1])
                for f in range(len(record["qs"]))]
        prior_t = [jnp.array(record["prior"][f][agent_idx:agent_idx+1])
                   for f in range(len(record["prior"]))]

        # Accuracy + Complexity (A-dependent)
        vfe_t = _compute_vfe_factored(qs_t, prior_t, obs_t, A)
        total_vfe = total_vfe + vfe_t

        # Transition prediction (B-dependent): consecutive timesteps
        if t + 1 < len(trajectory):
            qs_t1 = [jnp.array(trajectory[t + 1]["qs"][f][agent_idx:agent_idx+1])
                     for f in range(len(record["qs"]))]
            option_t = jnp.array(record["options"][agent_idx])
            trans_loss = _compute_transition_loss_factored(
                qs_t, qs_t1, option_t, B)
            total_vfe = total_vfe + trans_loss

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
    # B uses option-level (5 macro-options) since trajectories store options
    A_default = build_default_A()
    B_default = build_option_B()

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
# EFE helpers for C learning (inverse EFE, Shin et al. 2022)
# ---------------------------------------------------------------------------

def _flatten_qs(qs):
    """Flatten belief arrays to 1-D per factor (handles pymdp batch dims)."""
    return [q.ravel()[-NUM_STATE_FACTORS[f]:] if q.size > NUM_STATE_FACTORS[f]
            else q.ravel() for f, q in enumerate(qs)]


def _compute_expected_state(qs_flat, B, action):
    """Predict next state distribution for each factor given action.

    Parameters
    ----------
    qs_flat : list of (S_f,) arrays — current beliefs per factor
    B : list of option-level B matrices
    action : jnp scalar — macro-option index (0-4)

    Returns
    -------
    list of (S_f,) predicted next-state distributions
    """
    qs_next = []
    for f in range(len(B)):
        deps = B_DEPENDENCIES[f]
        B_f = B[f]
        action_oh = jax.nn.one_hot(action, B_f.shape[-1])
        B_f_a = jnp.tensordot(B_f, action_oh, axes=([-1], [0]))
        predicted = B_f_a
        for d in deps:
            predicted = jnp.tensordot(predicted, qs_flat[d], axes=([1], [0]))
        qs_next.append(predicted)
    return qs_next


def _compute_expected_obs(qs_flat, A):
    """Predict observation distribution for each modality.

    Parameters
    ----------
    qs_flat : list of (S_f,) arrays — state beliefs
    A : list of A matrices

    Returns
    -------
    list of (n_obs_m,) predicted observation distributions
    """
    qo = []
    for m in range(len(A)):
        deps = A_DEPENDENCIES[m]
        predicted = A[m]
        for d in reversed(deps):
            predicted = jnp.tensordot(predicted, qs_flat[d], axes=([-1], [0]))
        qo.append(predicted)
    return qo


def _compute_expected_utility(qo, C):
    """Compute utility: sum_m sum_o q(o_m) * C_m[o]."""
    util = jnp.float32(0.0)
    for qo_m, c_m in zip(qo, C):
        util = util + jnp.sum(qo_m * c_m)
    return util


def _compute_neg_efe_two_step(qs_flat, A, B, C, o1, o2):
    """Compute negative EFE for a single two-step policy (o1, o2).

    neg_G = utility_step1 + utility_step2
    (Epistemic term omitted — C learning targets pragmatic preferences.)
    """
    # Step 1
    qs1 = _compute_expected_state(qs_flat, B, o1)
    qo1 = _compute_expected_obs(qs1, A)
    util1 = _compute_expected_utility(qo1, C)
    # Step 2
    qs2 = _compute_expected_state(qs1, B, o2)
    qo2 = _compute_expected_obs(qs2, A)
    util2 = _compute_expected_utility(qo2, C)
    return util1 + util2


def _compute_q_pi_from_C(qs_flat, A, B, C, E, gamma=8.0):
    """Compute policy posterior q(pi) = softmax(gamma * neg_G + ln E).

    Enumerates all 25 two-step policies from 5 macro-options.

    Returns
    -------
    q_pi : (25,) array — posterior over two-step policies
    neg_G : (25,) array — negative EFE per policy
    """
    eps = 1e-16
    neg_G_list = []
    for o1 in range(NUM_OPTIONS):
        for o2 in range(NUM_OPTIONS):
            neg_G_list.append(
                _compute_neg_efe_two_step(
                    qs_flat, A, B, C,
                    jnp.int32(o1), jnp.int32(o2))
            )
    neg_G = jnp.array(neg_G_list)
    q_pi = jax.nn.softmax(gamma * neg_G + jnp.log(E + eps))
    return q_pi, neg_G


def _efe_policy_loss_single(qs_flat, A, B, C, E, observed_option, gamma=8.0):
    """Inverse EFE loss for one agent at one replan step.

    Loss = -ln p(first_action = observed_option)
    where p marginalizes q(pi) over all two-step policies whose
    first action matches the observed option.

    Parameters
    ----------
    qs_flat : list of (S_f,) — beliefs at replan time
    A, B : generative model matrices
    C : list of (n_obs_m,) — candidate preference vectors
    E : (25,) — habit prior (fixed, per-role)
    observed_option : jnp int — which option was selected (0-4)
    gamma : float — policy precision
    """
    eps = 1e-16
    q_pi, _ = _compute_q_pi_from_C(qs_flat, A, B, C, E, gamma)

    # Marginalize: q_pi is ordered (0,0),(0,1),...,(0,4),(1,0),...,(4,4)
    # p(first=a) = sum_{a2} q_pi[a*5 + a2]
    q_pi_2d = q_pi.reshape(NUM_OPTIONS, NUM_OPTIONS)
    marginal = q_pi_2d.sum(axis=1)  # (5,)
    return -jnp.log(marginal[observed_option] + eps)


def _detect_replan_steps(trajectory, agent_idx):
    """Find timesteps where agent_idx changed option (replanned).

    Returns list of (t, new_option) tuples where t is the timestep
    at which the agent's beliefs were used to select new_option.
    """
    replans = []
    for t in range(len(trajectory) - 1):
        cur = int(trajectory[t]["options"][agent_idx])
        nxt = int(trajectory[t + 1]["options"][agent_idx])
        if cur != nxt:
            replans.append((t + 1, nxt))
    return replans


def efe_policy_loss(
    C_all: list[jnp.ndarray],
    A: list[jnp.ndarray],
    B: list[jnp.ndarray],
    E_by_role: dict,
    trajectory: list[dict],
    agent_indices: list[int],
    n_agents: int = 4,
    gamma: float = 8.0,
) -> jnp.ndarray:
    """Inverse EFE loss for C learning over a full trajectory.

    Sums -ln q_pi(observed_option) at each replan step, per role.

    Parameters
    ----------
    C_all : list of jnp arrays
        Candidate C vectors (6 modalities × 3 roles = 18 arrays).
        Layout: C_all[role_idx * 6 + modality] for role_idx in {0,1,2}.
        Role order: miner=0, aligner=1, scout=2.
    A, B : fixed generative model
    E_by_role : dict mapping role name -> (25,) habit prior
    trajectory : list of dicts
    agent_indices : list of agent indices to include
    n_agents : int — total agents (for role assignment)
    gamma : float — policy precision

    Returns
    -------
    Scalar loss (lower = C better explains observed behavior).
    """
    n_modalities = len(A)
    role_to_idx = {"miner": 0, "aligner": 1, "scout": 2}

    total_loss = jnp.float32(0.0)
    n_replan = 0

    for agent_idx in agent_indices:
        role = _agent_role(agent_idx, n_agents)
        ridx = role_to_idx[role]
        C_role = [C_all[ridx * n_modalities + m] for m in range(n_modalities)]
        E_role = E_by_role[role]

        replans = _detect_replan_steps(trajectory, agent_idx)
        for t, new_option in replans:
            qs_raw = [jnp.array(trajectory[t]["qs"][f][agent_idx])
                      for f in range(len(trajectory[t]["qs"]))]
            qs_flat = _flatten_qs(qs_raw)

            loss_step = _efe_policy_loss_single(
                qs_flat, A, B, C_role, E_role,
                jnp.int32(new_option), gamma)
            total_loss = total_loss + loss_step
            n_replan += 1

    if n_replan > 0:
        return total_loss / n_replan
    return total_loss


def learn_C_from_trajectory(
    trajectory: list[dict],
    n_steps: int = 100,
    lr: float = 0.0001,
    gamma: float = 8.0,
    n_agents: int = 4,
    verbose: bool = True,
    A_params: list = None,
    B_params: list = None,
) -> dict:
    """Learn per-role C vectors via inverse EFE (Shin et al. 2022).

    Maximizes the likelihood of observed option selections under the
    EFE policy model by gradient descent on C.

    Parameters
    ----------
    trajectory : list of dicts
        Trajectory from BatchedAIFEngine.get_trajectory().
    n_steps : int
        Gradient descent steps.
    lr : float
        Learning rate (keep low — C is sensitive through softmax).
    gamma : float
        Policy precision (default 8.0, must match agent config).
    n_agents : int
        Number of agents in trajectory.
    verbose : bool
        Print progress.
    A_params, B_params : lists of arrays, optional
        Fixed A/B matrices. Default: hand-crafted.

    Returns
    -------
    dict with keys: C_miner, C_aligner, C_scout, loss_history
    """
    try:
        import optax
    except ImportError:
        print("optax not installed. Install with: pip install optax")
        sys.exit(1)

    # Fixed A and B (not learned here)
    A = [jnp.array(a, dtype=jnp.float32)
         for a in (A_params if A_params else build_default_A())]
    B = [jnp.array(b, dtype=jnp.float32)
         for b in (B_params if B_params else build_option_B())]

    # Per-role E vectors (fixed)
    n_policies = NUM_OPTIONS ** 2  # 25
    E_by_role = _build_E_vectors(n_policies)

    # Initialize C from hand-crafted defaults (raw values, not softmaxed)
    C_init = {
        "miner": build_C_miner(),
        "aligner": build_C_aligner(),
        "scout": build_C_scout(),
    }
    # Flatten to single list: [miner_0..miner_5, aligner_0..5, scout_0..5]
    n_modalities = len(A)
    C_flat = []
    for role in ("miner", "aligner", "scout"):
        for m in range(n_modalities):
            C_flat.append(jnp.array(C_init[role][m], dtype=jnp.float32))

    # Subsample trajectory
    stride = max(1, len(trajectory) // 50)
    traj_sub = trajectory[::stride]
    agent_indices = list(range(n_agents))

    # Count replan steps
    total_replans = sum(
        len(_detect_replan_steps(traj_sub, i)) for i in agent_indices)
    if verbose:
        print(f"[learn-c] {len(traj_sub)}/{len(trajectory)} steps "
              f"(stride={stride}), {total_replans} replan events, "
              f"gamma={gamma}")

    if total_replans == 0:
        print("[learn-c] WARNING: No replan events found in trajectory. "
              "Cannot learn C.")
        return {
            "C_miner": [np.asarray(c) for c in C_init["miner"]],
            "C_aligner": [np.asarray(c) for c in C_init["aligner"]],
            "C_scout": [np.asarray(c) for c in C_init["scout"]],
            "loss_history": [],
        }

    # Loss function
    def loss_fn(C_flat):
        return efe_policy_loss(
            C_flat, A, B, E_by_role, traj_sub,
            agent_indices, n_agents, gamma)

    # Initial loss
    initial_loss = float(loss_fn(C_flat))
    if verbose:
        print(f"[learn-c] Initial policy loss: {initial_loss:.4f}")

    # Gradient descent with Adam
    grad_fn = jax.grad(loss_fn)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(C_flat)

    loss_history = [initial_loss]

    for step in range(n_steps):
        grads = grad_fn(C_flat)
        updates, opt_state = optimizer.update(grads, opt_state)
        C_flat = [c + u for c, u in zip(C_flat, updates)]

        if (step + 1) % 10 == 0 or step == 0:
            current_loss = float(loss_fn(C_flat))
            loss_history.append(current_loss)
            if verbose:
                pct = (initial_loss - current_loss) / abs(initial_loss) * 100
                print(f"[learn-c] Step {step+1}/{n_steps}: "
                      f"loss={current_loss:.4f} ({pct:+.1f}%)")

    final_loss = float(loss_fn(C_flat))
    loss_history.append(final_loss)

    # Unpack per-role C
    result = {}
    for ridx, role in enumerate(("miner", "aligner", "scout")):
        result[f"C_{role}"] = [
            np.asarray(C_flat[ridx * n_modalities + m])
            for m in range(n_modalities)
        ]

    result["loss_history"] = loss_history
    result["initial_loss"] = initial_loss
    result["final_loss"] = final_loss

    if verbose:
        pct = (initial_loss - final_loss) / abs(initial_loss) * 100
        print(f"[learn-c] Final loss: {final_loss:.4f} ({pct:+.1f}%)")
        # Print C delta per role
        for role in ("miner", "aligner", "scout"):
            C_orig = C_init[role]
            C_new = result[f"C_{role}"]
            deltas = [float(np.abs(np.asarray(n) - o).max())
                      for n, o in zip(C_new, C_orig)]
            print(f"  {role}: max|ΔC| per modality = "
                  f"{[f'{d:.3f}' for d in deltas]}")

    return result


def _build_E_vectors(n_policies: int) -> dict:
    """Build per-role E vectors matching the strategic agent config.

    E has 25 entries (5 options × 5 options for two-step policies).
    Policies are ordered: [0-4]=MINE first, [5-9]=CRAFT first,
    [10-14]=CAPTURE first, [15-19]=EXPLORE first, [20-24]=DEFEND first.
    """
    E_miner = np.ones(n_policies)
    E_miner[0:5] = 4.0      # MINE first
    E_miner[15:20] = 2.0    # EXPLORE first
    E_miner[20:25] = 1.5    # DEFEND
    E_miner[5:10] = 0.001   # CRAFT blocked
    E_miner[10:15] = 0.001  # CAPTURE blocked

    E_aligner = np.ones(n_policies)
    E_aligner[5:10] = 4.0     # CRAFT first
    E_aligner[10:15] = 4.0    # CAPTURE first
    E_aligner[20:25] = 2.0    # DEFEND
    E_aligner[15:20] = 1.5    # EXPLORE
    E_aligner[0:5] = 0.001    # MINE blocked

    E_scout = np.ones(n_policies)
    E_scout[15:20] = 4.0      # EXPLORE first
    E_scout[20:25] = 2.0      # DEFEND
    E_scout[0:5] = 0.001      # MINE blocked
    E_scout[5:10] = 0.001     # CRAFT blocked
    E_scout[10:15] = 0.001    # CAPTURE blocked

    E_miner /= E_miner.sum()
    E_aligner /= E_aligner.sum()
    E_scout /= E_scout.sum()

    return {
        "miner": jnp.array(E_miner, dtype=jnp.float32),
        "aligner": jnp.array(E_aligner, dtype=jnp.float32),
        "scout": jnp.array(E_scout, dtype=jnp.float32),
    }


# ---------------------------------------------------------------------------
# Joint A + B + C optimization (Phase B-IV)
# ---------------------------------------------------------------------------

def learn_full_parameters(
    trajectory: list[dict],
    n_steps: int = 200,
    lr: float = 0.001,
    c_lr_scale: float = 0.1,
    a_weight: float = 1.0,
    b_weight: float = 1.0,
    c_weight: float = 0.5,
    kl_weight: float = 0.0,
    gamma: float = 8.0,
    n_agents: int = 4,
    verbose: bool = True,
) -> dict:
    """Joint A + B + C optimization with two-timescale learning.

    Loss = a_weight * L_vfe(A)         -- perception (accuracy + complexity)
         + b_weight * L_transition(B)  -- dynamics
         + c_weight * L_policy(C)      -- preferences (inverse EFE)
         + kl_weight * L_kl            -- regularization

    A and B use the VFE objective (fast, lr).
    C uses the inverse EFE objective (slow, lr * c_lr_scale).

    Parameters
    ----------
    trajectory : list of dicts
    n_steps : int
        Total gradient steps.
    lr : float
        Base learning rate for A and B.
    c_lr_scale : float
        C learning rate = lr * c_lr_scale (default 0.1 = 10x slower).
    a_weight, b_weight, c_weight : float
        Loss term weights.
    kl_weight : float
        KL regularization toward default A/B.
    gamma : float
        Policy precision for C learning.
    n_agents : int
    verbose : bool

    Returns
    -------
    dict with A, B, C_miner, C_aligner, C_scout, loss_history, etc.
    """
    try:
        import optax
    except ImportError:
        print("optax not installed. Install with: pip install optax")
        sys.exit(1)

    # Initialize A, B from defaults
    A_default = build_default_A()
    B_default = build_option_B()
    A_logits = params_to_logits(A_default)
    B_logits = params_to_logits(B_default)
    A_ref_logits = params_to_logits(A_default)
    B_ref_logits = params_to_logits(B_default)

    # Initialize per-role C
    n_modalities = len(A_default)
    C_flat = []
    for role_C in (build_C_miner(), build_C_aligner(), build_C_scout()):
        for c in role_C:
            C_flat.append(jnp.array(c, dtype=jnp.float32))

    # Fixed E vectors
    n_policies = NUM_OPTIONS ** 2
    E_by_role = _build_E_vectors(n_policies)

    # Subsample trajectory
    stride = max(1, len(trajectory) // 50)
    traj_sub = trajectory[::stride]
    agent_indices = list(range(n_agents))

    # Count replan events for C
    total_replans = sum(
        len(_detect_replan_steps(traj_sub, i)) for i in agent_indices)

    if verbose:
        print(f"[learn-full] {len(traj_sub)}/{len(trajectory)} steps "
              f"(stride={stride}), {total_replans} replans")
        print(f"  weights: a={a_weight}, b={b_weight}, c={c_weight}, "
              f"kl={kl_weight}")
        print(f"  lr={lr}, c_lr_scale={c_lr_scale}, gamma={gamma}")

    # VFE loss (A + B)
    def vfe_loss_fn(A_logits, B_logits):
        vfe = trajectory_vfe_multi_agent(
            A_logits, B_logits, traj_sub, agent_indices)
        if kl_weight > 0:
            kl = (_kl_divergence_logits(A_logits, A_ref_logits)
                  + _kl_divergence_logits(B_logits, B_ref_logits))
            vfe = vfe + kl_weight * kl
        return vfe

    # C loss (inverse EFE)
    A_fixed = logits_to_params(A_logits)  # Will be updated each step
    B_fixed = logits_to_params(B_logits)

    def c_loss_fn(C_flat, A_for_c, B_for_c):
        return efe_policy_loss(
            C_flat, A_for_c, B_for_c, E_by_role,
            traj_sub, agent_indices, n_agents, gamma)

    # Separate optimizers
    opt_ab = optax.adam(lr)
    opt_c = optax.adam(lr * c_lr_scale)
    opt_state_A = opt_ab.init(A_logits)
    opt_state_B = opt_ab.init(B_logits)
    opt_state_C = opt_c.init(C_flat)

    # Gradient functions
    grad_ab = jax.grad(vfe_loss_fn, argnums=(0, 1))
    grad_c = jax.grad(c_loss_fn, argnums=0)

    # Initial losses
    initial_vfe = float(vfe_loss_fn(A_logits, B_logits))
    initial_c = float(c_loss_fn(C_flat, A_fixed, B_fixed)) if total_replans > 0 else 0.0
    initial_total = a_weight * initial_vfe + c_weight * initial_c

    if verbose:
        print(f"[learn-full] Initial: VFE={initial_vfe:.4f}, "
              f"C_loss={initial_c:.4f}, total={initial_total:.4f}")

    loss_history = [{"step": 0, "vfe": initial_vfe,
                     "c_loss": initial_c, "total": initial_total}]

    for step in range(n_steps):
        # A + B gradient (VFE)
        grads_A, grads_B = grad_ab(A_logits, B_logits)
        updates_A, opt_state_A = opt_ab.update(grads_A, opt_state_A)
        updates_B, opt_state_B = opt_ab.update(grads_B, opt_state_B)
        A_logits = [a + u for a, u in zip(A_logits, updates_A)]
        B_logits = [b + u for b, u in zip(B_logits, updates_B)]

        # C gradient (inverse EFE), using current A, B
        if total_replans > 0 and c_weight > 0:
            A_for_c = logits_to_params(A_logits)
            B_for_c = logits_to_params(B_logits)
            grads_C = grad_c(C_flat, A_for_c, B_for_c)
            updates_C, opt_state_C = opt_c.update(grads_C, opt_state_C)
            C_flat = [c + u for c, u in zip(C_flat, updates_C)]

        if (step + 1) % 20 == 0 or step == 0:
            cur_vfe = float(vfe_loss_fn(A_logits, B_logits))
            cur_c = (float(c_loss_fn(C_flat, logits_to_params(A_logits),
                                     logits_to_params(B_logits)))
                     if total_replans > 0 else 0.0)
            cur_total = a_weight * cur_vfe + c_weight * cur_c
            loss_history.append({"step": step + 1, "vfe": cur_vfe,
                                 "c_loss": cur_c, "total": cur_total})
            if verbose:
                vfe_pct = (initial_vfe - cur_vfe) / abs(initial_vfe) * 100
                c_pct = ((initial_c - cur_c) / abs(initial_c) * 100
                         if initial_c != 0 else 0.0)
                print(f"[learn-full] Step {step+1}/{n_steps}: "
                      f"VFE={cur_vfe:.4f}({vfe_pct:+.1f}%) "
                      f"C={cur_c:.4f}({c_pct:+.1f}%)")

    # Final
    A_learned = [np.asarray(a) for a in logits_to_params(A_logits)]
    B_learned = [np.asarray(b) for b in logits_to_params(B_logits)]

    result = {
        "A": A_learned,
        "B": B_learned,
        "loss_history": loss_history,
        "initial_vfe": initial_vfe,
        "final_vfe": float(vfe_loss_fn(A_logits, B_logits)),
    }

    # Unpack per-role C
    for ridx, role in enumerate(("miner", "aligner", "scout")):
        result[f"C_{role}"] = [
            np.asarray(C_flat[ridx * n_modalities + m])
            for m in range(n_modalities)
        ]

    if verbose:
        final_vfe = result["final_vfe"]
        pct = (initial_vfe - final_vfe) / abs(initial_vfe) * 100
        print(f"[learn-full] Final VFE: {final_vfe:.4f} ({pct:+.1f}%)")

    return result


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
    has_q_pi = "q_pi" in trajectory[0]
    for t, record in enumerate(trajectory):
        for m, obs_m in enumerate(record["obs"]):
            data[f"obs_{t}_{m}"] = obs_m
        for f, qs_f in enumerate(record["qs"]):
            data[f"qs_{t}_{f}"] = qs_f
        for f, prior_f in enumerate(record["prior"]):
            data[f"prior_{t}_{f}"] = prior_f
        data[f"actions_{t}"] = record["actions"]
        data[f"options_{t}"] = record["options"]
        if has_q_pi and "q_pi" in record:
            data[f"q_pi_{t}"] = record["q_pi"]
            data[f"neg_efe_{t}"] = record["neg_efe"]
    data["n_steps"] = np.array(len(trajectory))
    data["n_obs_modalities"] = np.array(len(trajectory[0]["obs"]))
    data["n_state_factors"] = np.array(len(trajectory[0]["qs"]))
    data["has_q_pi"] = np.array(has_q_pi)
    np.savez_compressed(path, **data)
    print(f"[save] Trajectory ({len(trajectory)} steps) saved to {path}")


def load_trajectory(path: str) -> list[dict]:
    """Load trajectory data from .npz file."""
    data = np.load(path, allow_pickle=True)
    n_steps = int(data["n_steps"])
    n_obs = int(data["n_obs_modalities"])
    n_factors = int(data["n_state_factors"])
    has_q_pi = bool(data["has_q_pi"]) if "has_q_pi" in data else False

    trajectory = []
    for t in range(n_steps):
        record = {
            "obs": [data[f"obs_{t}_{m}"] for m in range(n_obs)],
            "qs": [data[f"qs_{t}_{f}"] for f in range(n_factors)],
            "prior": [data[f"prior_{t}_{f}"] for f in range(n_factors)],
            "actions": data[f"actions_{t}"],
            "options": data[f"options_{t}"],
        }
        if has_q_pi and f"q_pi_{t}" in data:
            record["q_pi"] = data[f"q_pi_{t}"]
            record["neg_efe"] = data[f"neg_efe_{t}"]
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
        "B": build_option_B(),
    }
    params_learned = load_params(args.learned_params)

    result = compare_models(
        trajectory, params_default, params_learned,
        agent_idx=args.agent,
        label_1="default", label_2="learned",
    )
    print(f"\n[compare] Result: {json.dumps(result, indent=2)}")


def cmd_learn_full(args):
    """Joint A + B + C learning from a saved trajectory."""
    trajectory = load_trajectory(args.trajectory)
    print(f"[learn-full] Loaded trajectory: {len(trajectory)} steps")

    result = learn_full_parameters(
        trajectory,
        n_steps=args.steps,
        lr=args.lr,
        c_lr_scale=args.c_lr_scale,
        a_weight=args.a_weight,
        b_weight=args.b_weight,
        c_weight=args.c_weight,
        kl_weight=args.kl_weight,
        gamma=args.gamma,
        n_agents=args.n_agents,
        verbose=True,
    )

    # Save learned parameters
    C_save = result["C_miner"]  # Default role C for main file
    save_params(
        args.output,
        A=result["A"],
        B=result["B"],
        C=C_save,
        D=[np.asarray(d) for d in build_D()],
        metadata={
            "source": "joint_abc_learning",
            "initial_vfe": result["initial_vfe"],
            "final_vfe": result["final_vfe"],
            "n_steps": args.steps,
            "lr": args.lr,
            "c_lr_scale": args.c_lr_scale,
            "weights": f"a={args.a_weight},b={args.b_weight},c={args.c_weight}",
        },
    )

    # Save per-role C separately
    role_path = args.output.replace(".npz", "_roles.npz")
    role_data = {}
    for role in ("miner", "aligner", "scout"):
        for i, c in enumerate(result[f"C_{role}"]):
            role_data[f"C_{role}_{i}"] = np.asarray(c)
    np.savez_compressed(role_path, **role_data)
    print(f"[learn-full] Per-role C saved to {role_path}")


def cmd_learn_c(args):
    """Learn C vectors from a saved trajectory via inverse EFE."""
    trajectory = load_trajectory(args.trajectory)
    print(f"[learn-c] Loaded trajectory: {len(trajectory)} steps")

    # Optionally use learned A/B
    A_params = None
    B_params = None
    if args.ab_params:
        ab = load_params(args.ab_params)
        A_params = ab["A"]
        B_params = ab["B"]
        print(f"[learn-c] Using A/B from {args.ab_params}")

    result = learn_C_from_trajectory(
        trajectory,
        n_steps=args.steps,
        lr=args.lr,
        gamma=args.gamma,
        n_agents=args.n_agents,
        verbose=True,
        A_params=A_params,
        B_params=B_params,
    )

    # Save: merge with A/B (default or provided)
    A_save = A_params if A_params else build_default_A()
    B_save = B_params if B_params else build_option_B()
    # Combine per-role C into a single list (use miner as default C[0..5])
    C_save = result["C_miner"]  # Default role for save_params

    save_params(
        args.output,
        A=A_save,
        B=B_save,
        C=C_save,
        D=[np.asarray(d) for d in build_D()],
        metadata={
            "source": "inverse_efe_c_learning",
            "initial_loss": result["initial_loss"],
            "final_loss": result["final_loss"],
            "n_steps": args.steps,
            "lr": args.lr,
            "gamma": args.gamma,
        },
    )

    # Also save per-role C separately
    role_path = args.output.replace(".npz", "_roles.npz")
    role_data = {}
    for role in ("miner", "aligner", "scout"):
        for i, c in enumerate(result[f"C_{role}"]):
            role_data[f"C_{role}_{i}"] = np.asarray(c)
    np.savez_compressed(role_path, **role_data)
    print(f"[learn-c] Per-role C saved to {role_path}")


def cmd_save_default(args):
    """Save default hand-crafted parameters for comparison."""
    save_params(
        args.output,
        A=build_default_A(),
        B=build_option_B(),
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

    # learn-full subcommand (Phase B-IV: joint A+B+C)
    p_full = sub.add_parser("learn-full",
                            help="Joint A+B+C learning (two-timescale)")
    p_full.add_argument("--trajectory", required=True,
                        help="Path to trajectory .npz")
    p_full.add_argument("--output", default="/tmp/learned_full.npz")
    p_full.add_argument("--steps", type=int, default=200)
    p_full.add_argument("--lr", type=float, default=0.001)
    p_full.add_argument("--c-lr-scale", type=float, default=0.1,
                        help="C learning rate = lr * scale (default 0.1)")
    p_full.add_argument("--a-weight", type=float, default=1.0)
    p_full.add_argument("--b-weight", type=float, default=1.0)
    p_full.add_argument("--c-weight", type=float, default=0.5)
    p_full.add_argument("--kl-weight", type=float, default=0.0)
    p_full.add_argument("--gamma", type=float, default=8.0)
    p_full.add_argument("--n-agents", type=int, default=4)

    # learn-c subcommand (Phase B-III: inverse EFE)
    p_learn_c = sub.add_parser("learn-c",
                               help="Learn C vectors via inverse EFE")
    p_learn_c.add_argument("--trajectory", required=True,
                           help="Path to trajectory .npz")
    p_learn_c.add_argument("--output", default="/tmp/learned_C.npz")
    p_learn_c.add_argument("--steps", type=int, default=100)
    p_learn_c.add_argument("--lr", type=float, default=0.0001,
                           help="Learning rate (keep low, C is sensitive)")
    p_learn_c.add_argument("--gamma", type=float, default=8.0,
                           help="Policy precision (must match agent config)")
    p_learn_c.add_argument("--n-agents", type=int, default=4)
    p_learn_c.add_argument("--ab-params", default=None,
                           help="Path to learned A/B .npz (optional)")

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
    elif args.command == "learn-full":
        cmd_learn_full(args)
    elif args.command == "learn-c":
        cmd_learn_c(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "save-default":
        cmd_save_default(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
