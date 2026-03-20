"""Fit POMDP A/B matrices from trajectory data.

Loads trajectories from ``.npz`` files, discretizes observations using
:class:`ObservationDiscretizer`, and estimates maximum-likelihood A and B
matrices via transition counting with Dirichlet smoothing.

For B matrix fitting, primitive actions from trajectories are mapped to
task-level policies via ``infer_task_policy()``, since the POMDP action
space is 13 task-level policies (not 5 movement actions).

Can be run as a script::

    python -m aif_meta_cogames.aif_agent.fit_matrices \\
        --data ./trajectory_data_v3 \\
        --output ./fitted_pomdp
"""

import json
from pathlib import Path

import numpy as np

from .discretizer import (
    NUM_OBS,
    NUM_STATES,
    TASK_POLICY_NAMES,
    ObservationDiscretizer,
    infer_task_policy,
    state_factors,
    state_label,
)
from .generative_model import NUM_ACTIONS, CogsGuardPOMDP


def fit_variant(
    variant_dir: Path,
    discretizer: ObservationDiscretizer,
    smoothing: float = 0.01,
) -> dict:
    """Fit A and B matrices for one environment variant.

    Parameters
    ----------
    variant_dir
        Directory containing ``episode_*.npz`` and ``metadata.json``.
    discretizer
        Configured :class:`ObservationDiscretizer`.
    smoothing
        Dirichlet smoothing constant (prevents zero probabilities).

    Returns
    -------
    dict with keys ``A``, ``B``, ``state_counts``, ``n_transitions``.
    """
    npz_files = sorted(variant_dir.glob("episode_*.npz"))

    # Accumulators
    A_counts = [np.zeros((n_obs, NUM_STATES)) for n_obs in NUM_OBS]
    B_counts = np.zeros((NUM_STATES, NUM_STATES, NUM_ACTIONS))
    state_counts = np.zeros(NUM_STATES)
    n_transitions = 0

    for npz_path in npz_files:
        data = np.load(npz_path)
        obs = data["obs"]          # (T, N, 200, 3)
        T, N = obs.shape[:2]

        for t in range(T):
            for a in range(N):
                s = discretizer.infer_state(obs[t, a], agent_id=a)
                o = discretizer.discretize_obs(obs[t, a])

                state_counts[s] += 1
                for m in range(len(NUM_OBS)):
                    A_counts[m][o[m], s] += 1

                if t < T - 1:
                    s_next = discretizer.infer_state(obs[t + 1, a], agent_id=a)

                    # Infer task-level policy from state transition
                    p_t, h_t, _, _ = state_factors(s)
                    p_next, h_next, _, _ = state_factors(s_next)
                    task = infer_task_policy(p_t, h_t, p_next, h_next)

                    if 0 <= task < NUM_ACTIONS:
                        B_counts[s_next, s, task] += 1
                        n_transitions += 1

    # Normalise with Dirichlet smoothing
    A = []
    for m in range(len(NUM_OBS)):
        a_m = A_counts[m] + smoothing
        col_sums = a_m.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)
        A.append(a_m / col_sums)

    B_mat = B_counts + smoothing
    for s in range(NUM_STATES):
        for act in range(NUM_ACTIONS):
            col_sum = B_mat[:, s, act].sum()
            if col_sum > 0:
                B_mat[:, s, act] /= col_sum

    return {
        "A": A,
        "B": [B_mat],
        "state_counts": state_counts,
        "transition_counts": B_counts,
        "n_transitions": n_transitions,
    }


def fit_all_variants(
    data_root: Path,
    variant_names: list[str] | None = None,
    smoothing: float = 0.01,
) -> dict[str, dict]:
    """Fit A/B matrices for all environment variants.

    Parameters
    ----------
    data_root
        Root directory containing variant subdirectories.
    variant_names
        Optional list of variant names. If ``None``, discovers all
        variants with a ``metadata.json``.
    smoothing
        Dirichlet smoothing constant.

    Returns
    -------
    dict mapping variant_name to fit result (see :func:`fit_variant`).
    """
    data_root = Path(data_root)

    if variant_names is None:
        variant_names = sorted([
            d.name for d in data_root.iterdir()
            if d.is_dir() and (d / "metadata.json").exists()
        ])

    # Load obs_features from first variant's metadata
    first_meta_path = data_root / variant_names[0] / "metadata.json"
    with open(first_meta_path) as f:
        meta = json.load(f)
    obs_feature_names = meta["obs_features"]

    discretizer = ObservationDiscretizer(obs_feature_names)

    results = {}
    for i, name in enumerate(variant_names):
        print(
            f"  [{i + 1}/{len(variant_names)}] {name}...",
            end=" ", flush=True,
        )
        variant_dir = data_root / name
        result = fit_variant(variant_dir, discretizer, smoothing)
        results[name] = result
        print(f"{result['n_transitions']:,} transitions")

    return results


def save_fitted_models(results: dict, output_dir: Path):
    """Save fitted A/B matrices and a summary for all variants.

    Creates one ``.npz`` per variant plus a ``summary.json`` with
    state statistics and most-frequent transitions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name, result in results.items():
        model = CogsGuardPOMDP(A=result["A"], B=result["B"])
        model.save(output_dir / f"{name}.npz")

        summary[name] = {
            "n_transitions": result["n_transitions"],
            "state_distribution": {
                state_label(s): int(result["state_counts"][s])
                for s in range(NUM_STATES)
                if result["state_counts"][s] > 0
            },
            "top_transitions": _top_transitions(
                result["transition_counts"], k=10
            ),
        }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def _top_transitions(B_counts: np.ndarray, k: int = 10) -> list[dict]:
    """Return the k most frequent transitions as dicts."""
    entries = []
    for s in range(NUM_STATES):
        for s_next in range(NUM_STATES):
            for a in range(NUM_ACTIONS):
                count = int(B_counts[s_next, s, a])
                if count > 0:
                    entries.append({
                        "from": state_label(s),
                        "to": state_label(s_next),
                        "task_policy": TASK_POLICY_NAMES[a],
                        "count": count,
                    })
    entries.sort(key=lambda e: e["count"], reverse=True)
    return entries[:k]


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fit POMDP A/B matrices from trajectory data",
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to trajectory data directory (e.g. ./trajectory_data_v3)",
    )
    parser.add_argument(
        "--output", type=str, default="./fitted_pomdp",
        help="Output directory for fitted matrices (default: ./fitted_pomdp)",
    )
    parser.add_argument(
        "--variants", type=str, default=None,
        help="Comma-separated variant names (default: all)",
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.01,
        help="Dirichlet smoothing constant (default: 0.01)",
    )
    args = parser.parse_args()

    variant_names = None
    if args.variants:
        variant_names = [v.strip() for v in args.variants.split(",")]

    print(f"Fitting POMDP from: {args.data}")
    print(f"State space: {NUM_STATES} states, {NUM_ACTIONS} task-level policies")
    results = fit_all_variants(
        Path(args.data),
        variant_names=variant_names,
        smoothing=args.smoothing,
    )

    print(f"\nSaving fitted models to: {args.output}")
    save_fitted_models(results, Path(args.output))

    total = sum(r["n_transitions"] for r in results.values())
    print(f"\nFitted {len(results)} variants, {total:,} total transitions")

    # Show example state distribution
    first = next(iter(results.values()))
    print(f"\nExample state distribution ({next(iter(results))}):")
    for s in range(NUM_STATES):
        count = int(first["state_counts"][s])
        if count > 0:
            pct = 100.0 * count / first["state_counts"].sum()
            print(f"  {state_label(s):45s} {count:>8,}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
