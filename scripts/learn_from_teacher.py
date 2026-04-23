#!/usr/bin/env python3
"""Learn C vectors (observation preferences) from teacher trajectories.

Given teacher obs/actions/rewards recorded during mixed-team eval
(via RecordingTeacherPolicy), derives preference vectors via inverse EFE:

    C[o] = log(p(o | reward > threshold)) - log(p(o | baseline))

This tells us: what observations does a competent agent seek?

Usage::

    python scripts/learn_from_teacher.py \\
        --teacher-trajectory /tmp/teacher_trajectory.npz \\
        --output /tmp/learned_C.npz \\
        [--reward-threshold 0.0] \\
        [--smoothing 0.1]

References:
    Shin et al. (2022). Imitation Learning via Active Inference.
    Da Costa et al. (2020). Active Inference on Discrete State-Spaces.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path for imports
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from aif_meta_cogames.aif_agent.discretizer import (
    NUM_OBS,
    OBS_MODALITY_NAMES,
    ObservationDiscretizer,
)
from aif_meta_cogames.aif_agent.generative_model import build_C


LOC_GLOBAL = 254
LOC_CENTER = (6 << 4) | 6  # 102
LOC_EMPTY = 255


def compute_pseudo_rewards(
    obs_tokens: np.ndarray,
    agent_ids: np.ndarray | None = None,
) -> np.ndarray:
    """Compute pseudo-rewards from observation token deltas.

    Since cogames doesn't deliver per-step rewards through the policy API,
    we infer reward signals from inventory state changes between consecutive
    steps for each agent.

    Global/center tokens (loc=254 or loc=102) with value>0 represent
    inventory items.  We track changes in the set of (feat_id, value) pairs:

      * Inventory gain  (new item or value increase) → +1.0 per item
      * Inventory loss   (item gone or value decrease) → +2.0 per item
        (loss = deposit at hub/junction for a competent teacher)
      * Inventory held   (maintaining resources)       → +0.1

    Parameters
    ----------
    obs_tokens : (N, 200, 3) uint8
    agent_ids  : (N,) int or None

    Returns
    -------
    (N,) float32 pseudo-rewards
    """
    N = len(obs_tokens)
    rewards = np.zeros(N, dtype=np.float32)

    def _inventory_sig(tokens: np.ndarray) -> dict[int, int]:
        """Map feat_id → value for global/center tokens with value > 0."""
        sig: dict[int, int] = {}
        for i in range(tokens.shape[0]):
            loc = int(tokens[i, 0])
            if loc == LOC_EMPTY:
                continue
            if loc != LOC_GLOBAL and loc != LOC_CENTER:
                continue
            feat_id, value = int(tokens[i, 1]), int(tokens[i, 2])
            if value > 0:
                sig[feat_id] = value
        return sig

    if agent_ids is not None:
        unique_agents = np.unique(agent_ids)
    else:
        unique_agents = np.array([0])
        agent_ids = np.zeros(N, dtype=np.int32)

    for agent in unique_agents:
        indices = np.where(agent_ids == agent)[0]
        prev_sig: dict[int, int] = {}
        for idx in indices:
            curr_sig = _inventory_sig(obs_tokens[idx])
            if prev_sig:
                gained = sum(
                    1
                    for fid, val in curr_sig.items()
                    if fid not in prev_sig or val > prev_sig[fid]
                )
                lost = sum(
                    1
                    for fid, val in prev_sig.items()
                    if fid not in curr_sig or curr_sig.get(fid, 0) < val
                )
                held = len(curr_sig)
                if gained > 0:
                    rewards[idx] += 1.0 * gained
                if lost > 0:
                    rewards[idx] += 2.0 * lost  # deposit = progress
            prev_sig = curr_sig

    return rewards


def load_teacher_trajectory(path: str) -> dict:
    """Load teacher trajectory NPZ (from RecordingTeacherPolicy)."""
    data = np.load(path, allow_pickle=True)
    result = {
        "obs": data["obs"],           # (N, 200, 3) uint8
        "actions": data["actions"],    # (N,) int32
        "rewards": data["rewards"],    # (N,) float32
        "agent_ids": data.get("agent_ids", None),
        "step_indices": data.get("step_indices", None),
    }
    # Load discretizer metadata if available
    if "obs_feature_names" in data:
        result["obs_feature_names"] = list(data["obs_feature_names"])
    if "tag_categories" in data:
        result["tag_categories"] = {
            int(k): str(v) for k, v in data["tag_categories"]
        }
    return result


def discretize_teacher_obs_proper(
    obs_tokens: np.ndarray,
    obs_feature_names: list[str],
    tag_categories: dict[int, str] | None = None,
) -> list[np.ndarray]:
    """Discretize raw observation tokens using the real ObservationDiscretizer.

    Parameters
    ----------
    obs_tokens : (N, 200, 3) uint8
    obs_feature_names : feature name list (index = feat_id)
    tag_categories : tag value → category string, or None for defaults

    Returns
    -------
    list of 6 arrays, each (N,) with discretized values
    """
    disc = ObservationDiscretizer(obs_feature_names, tag_categories)
    N = len(obs_tokens)
    n_modalities = len(NUM_OBS)
    result = [np.zeros(N, dtype=np.int32) for _ in range(n_modalities)]

    for i in range(N):
        obs = obs_tokens[i]  # (200, 3)
        d_obs = disc.discretize_obs(obs)
        for m in range(n_modalities):
            result[m][i] = d_obs[m]

    return result


def discretize_teacher_obs_heuristic(obs_tokens: np.ndarray) -> list[np.ndarray]:
    """Discretize raw observation tokens via heuristic token analysis.

    Since the full ObservationDiscretizer requires PolicyEnvInterface metadata
    (feature names, tag categories) which are not available at script-time,
    we use a simplified approach based on token patterns.

    For full fidelity, run with --use-discretizer flag inside a cogames eval
    context where PolicyEnvInterface is available.

    Parameters
    ----------
    obs_tokens : np.ndarray -- (N, 200, 3) raw observation tokens

    Returns
    -------
    list of 6 arrays, each (N,) with discretized values
    """
    N = len(obs_tokens)
    n_modalities = len(NUM_OBS)
    disc = [np.zeros(N, dtype=np.int32) for _ in range(n_modalities)]

    for i in range(N):
        tokens = obs_tokens[i]  # (200, 3)

        # Identify valid (non-padding) tokens
        valid_mask = tokens[:, 0] != 255
        n_valid = int(valid_mask.sum())

        if n_valid == 0:
            continue

        valid = tokens[valid_mask]
        tag_ids = valid[:, 0].astype(np.int32)
        feat1 = valid[:, 1].astype(np.int32)
        feat2 = valid[:, 2].astype(np.int32)

        # Heuristic discretization:
        # Token format in cogames: [tag_id, inventory_bits, health_or_status]
        # The exact mapping depends on the mission config, but common patterns:

        # o_resource (0): presence of extractor-tagged tokens nearby
        resource_tokens = np.sum((tag_ids >= 10) & (tag_ids < 40))
        if resource_tokens > 3:
            disc[0][i] = 2  # AT
        elif resource_tokens > 0:
            disc[0][i] = 1  # NEAR
        # else 0 = NONE

        # o_station (1): hub/craft/junction presence
        station_tokens = np.sum((tag_ids >= 40) & (tag_ids < 80))
        if station_tokens > 3:
            disc[1][i] = 3  # JUNCTION
        elif station_tokens > 1:
            disc[1][i] = 2  # CRAFT
        elif station_tokens > 0:
            disc[1][i] = 1  # HUB
        # else 0 = NONE

        # o_inventory (2): from self-token (first token typically)
        # ObsInventory: EMPTY=0, HAS_RESOURCE=1, HAS_GEAR=2, HAS_BOTH=3
        self_inv = feat1[0] if n_valid > 0 else 0
        disc[2][i] = min(int(self_inv) % 4, 3)

        # o_contest (3): enemy proximity
        enemy_near = np.sum(feat2 > 128)
        if enemy_near > 2:
            disc[3][i] = 2  # LOST
        elif enemy_near > 0:
            disc[3][i] = 1  # CONTESTED
        # else 0 = FREE

        # o_social (4): ally/enemy count
        n_agents_visible = np.sum((tag_ids > 0) & (tag_ids < 10))
        n_enemy = np.sum(feat2 > 128)
        if n_agents_visible > 0 and n_enemy > 0:
            disc[4][i] = 3  # BOTH_NEAR
        elif n_enemy > 0:
            disc[4][i] = 2  # ENEMY_NEAR
        elif n_agents_visible > 0:
            disc[4][i] = 1  # ALLY_NEAR
        # else 0 = ALONE

        # o_role_signal (5): simplified
        disc[5][i] = 0  # Default SAME_ROLE

    return disc


def learn_C_from_teacher(
    disc_obs: list[np.ndarray],
    rewards: np.ndarray,
    reward_threshold: float = 0.0,
    smoothing: float = 0.1,
) -> list[np.ndarray]:
    """Learn C vectors via inverse EFE from discretized teacher obs + rewards.

    C[m][o] = log(p(o | reward > threshold)) - log(p(o | baseline))

    Parameters
    ----------
    disc_obs : list of (N,) int arrays -- discretized observations per modality
    rewards : (N,) float array
    reward_threshold : float -- reward threshold for "positive" class
    smoothing : float -- Laplace smoothing concentration

    Returns
    -------
    list of np.ndarray -- learned C vectors per modality
    """
    pos_mask = rewards > reward_threshold
    n_pos = int(pos_mask.sum())
    n_total = len(rewards)

    print(f"  Positive reward steps: {n_pos}/{n_total} "
          f"({100 * n_pos / max(n_total, 1):.1f}%)")

    if n_pos == 0:
        print("  WARNING: No positive-reward steps! Using uniform C vectors.")
        return [np.zeros(d, dtype=np.float32) for d in NUM_OBS]

    C_learned = []
    for m, (d_m, name) in enumerate(zip(NUM_OBS, OBS_MODALITY_NAMES)):
        obs_m = disc_obs[m]

        # Frequency distributions with Laplace smoothing
        p_pos = np.zeros(d_m, dtype=np.float64) + smoothing
        p_base = np.zeros(d_m, dtype=np.float64) + smoothing

        for o in range(d_m):
            p_pos[o] += np.sum(obs_m[pos_mask] == o)
            p_base[o] += np.sum(obs_m == o)

        p_pos /= p_pos.sum()
        p_base /= p_base.sum()

        # Inverse EFE: C = log(p_pos) - log(p_base)
        C_m = np.log(p_pos) - np.log(p_base)
        C_learned.append(C_m.astype(np.float32))

        print(f"  {name} (dim={d_m}):")
        print(f"    p(o|r>0)  = {np.round(p_pos, 3)}")
        print(f"    p(o|all)  = {np.round(p_base, 3)}")
        print(f"    C_learned = {np.round(C_m, 3)}")

    return C_learned


def compare_C_vectors(C_learned: list[np.ndarray], C_default: list[np.ndarray]):
    """Print side-by-side comparison of learned vs hand-tuned C vectors."""
    print("\n=== C Vector Comparison (learned vs hand-tuned) ===")
    for m, name in enumerate(OBS_MODALITY_NAMES):
        if m >= len(C_learned) or m >= len(C_default):
            continue

        c_l = np.asarray(C_learned[m], dtype=np.float64)
        c_d = np.asarray(C_default[m], dtype=np.float64)

        # Pad shorter to match longer
        max_len = max(len(c_l), len(c_d))
        if len(c_l) < max_len:
            c_l = np.pad(c_l, (0, max_len - len(c_l)))
        if len(c_d) < max_len:
            c_d = np.pad(c_d, (0, max_len - len(c_d)))

        # Normalize to zero-mean for comparison
        c_l_norm = c_l - c_l.mean()
        c_d_norm = c_d - c_d.mean()

        # Cosine similarity
        norm_l = np.linalg.norm(c_l_norm)
        norm_d = np.linalg.norm(c_d_norm)
        cos_sim = (
            float(np.dot(c_l_norm, c_d_norm) / (norm_l * norm_d))
            if norm_l > 0 and norm_d > 0
            else 0.0
        )

        print(f"\n  {name}:")
        print(f"    Learned:    {np.round(c_l, 3)}")
        print(f"    Hand-tuned: {np.round(c_d, 3)}")
        print(f"    Cos sim:    {cos_sim:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Learn C vectors from teacher trajectories via inverse EFE"
    )
    parser.add_argument(
        "--teacher-trajectory", required=True,
        help="Path to teacher_trajectory.npz (from RecordingTeacherPolicy)",
    )
    parser.add_argument(
        "--output", default="/tmp/learned_C.npz",
        help="Output path for learned C vectors",
    )
    parser.add_argument(
        "--reward-threshold", type=float, default=0.0,
        help="Reward threshold for positive class",
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.1,
        help="Laplace smoothing alpha",
    )
    args = parser.parse_args()

    print("=== Teacher-Conditioned C Vector Learning ===")
    print(f"  Input:  {args.teacher_trajectory}")
    print(f"  Output: {args.output}")
    print(f"  Reward threshold: {args.reward_threshold}")
    print(f"  Smoothing: {args.smoothing}")

    # Load teacher data
    print("\nLoading teacher trajectory...")
    traj = load_teacher_trajectory(args.teacher_trajectory)
    N = len(traj["rewards"])
    rewards = traj["rewards"]
    print(f"  {N} steps loaded")
    print(f"  Reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")
    print(f"  Mean reward: {rewards.mean():.4f}")

    # If real rewards are all zero, compute pseudo-rewards from token deltas
    if rewards.max() == 0.0:
        print("\n  Real rewards are all zero — computing pseudo-rewards "
              "from inventory token deltas...")
        rewards = compute_pseudo_rewards(
            traj["obs"],
            traj.get("agent_ids"),
        )
        print(f"  Pseudo-reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")
        print(f"  Pseudo-reward mean:  {rewards.mean():.4f}")
        print(f"  Nonzero steps: {int((rewards > 0).sum())}/{N} "
              f"({100 * (rewards > 0).sum() / max(N, 1):.1f}%)")

    # Discretize — use real discretizer if metadata available
    if "obs_feature_names" in traj:
        print(f"\nDiscretizing with real ObservationDiscretizer "
              f"({len(traj['obs_feature_names'])} features)...")
        disc_obs = discretize_teacher_obs_proper(
            traj["obs"],
            traj["obs_feature_names"],
            traj.get("tag_categories"),
        )
    else:
        print("\nDiscretizing teacher observations (heuristic fallback)...")
        disc_obs = discretize_teacher_obs_heuristic(traj["obs"])

    # Learn C vectors
    print("\nLearning C vectors via inverse EFE...")
    C_learned = learn_C_from_teacher(
        disc_obs, rewards,
        reward_threshold=args.reward_threshold,
        smoothing=args.smoothing,
    )

    # Compare with hand-tuned
    try:
        C_default = build_C()
        compare_C_vectors(C_learned, C_default)
    except Exception as e:
        print(f"\nCould not compare with hand-tuned C: {e}")

    # Save
    save_dict = {}
    for m, C_m in enumerate(C_learned):
        save_dict[f"C_{m}"] = C_m
    save_dict["modality_names"] = np.array(OBS_MODALITY_NAMES)
    save_dict["reward_threshold"] = np.float32(args.reward_threshold)
    save_dict["smoothing"] = np.float32(args.smoothing)
    np.savez_compressed(args.output, **save_dict)
    print(f"\nSaved learned C vectors to {args.output}")


if __name__ == "__main__":
    main()
