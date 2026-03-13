"""PyTorch dataset for loading trajectory .npz files."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Dataset of (obs_t, action_t, obs_{t+1}, reward_t, done_t) transitions.

    Loads from .npz files produced by collect_trajectories.py.
    Each .npz has: obs(T, N, 200, 3), actions(T, N), rewards(T, N), dones(T, N).
    """

    def __init__(self, variant_dir: Path, agent_id: Optional[int] = None):
        """Load all episodes from a variant directory.

        Args:
            variant_dir: Path containing episode_XXX.npz files and metadata.json
            agent_id: If set, only load transitions for this agent index.
                      If None, load all agents as separate transitions.
        """
        self.variant_dir = Path(variant_dir)
        self.agent_id = agent_id

        with open(self.variant_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        self.transitions = []
        npz_files = sorted(self.variant_dir.glob("episode_*.npz"))

        for npz_path in npz_files:
            data = np.load(npz_path)
            obs = data["obs"]          # (T, N, 200, 3)
            actions = data["actions"]  # (T, N)
            rewards = data["rewards"]  # (T, N)
            dones = data["dones"]      # (T, N)

            T, N = obs.shape[0], obs.shape[1]

            for t in range(T - 1):
                agents = [agent_id] if agent_id is not None else range(N)
                for a in agents:
                    self.transitions.append((
                        obs[t, a],        # obs_t: (200, 3)
                        actions[t, a],    # action_t: scalar
                        obs[t + 1, a],    # obs_{t+1}: (200, 3)
                        rewards[t, a],    # reward_t: scalar
                        dones[t, a],      # done_t: bool
                    ))

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int):
        obs, action, next_obs, reward, done = self.transitions[idx]
        return {
            "obs": torch.from_numpy(obs),
            "action": torch.tensor(action, dtype=torch.long),
            "next_obs": torch.from_numpy(next_obs),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "done": torch.tensor(done, dtype=torch.float32),
        }


class MultiVariantDataset:
    """Loads trajectory datasets from multiple environment variants.

    Used for meta-learning: each variant is a "task".
    """

    def __init__(self, data_root: Path, variant_names: Optional[list[str]] = None):
        self.data_root = Path(data_root)

        if variant_names is None:
            variant_names = [
                d.name for d in sorted(self.data_root.iterdir())
                if d.is_dir() and (d / "metadata.json").exists()
            ]

        self.variant_names = variant_names
        self.datasets = {
            name: TrajectoryDataset(self.data_root / name) for name in variant_names
        }

    def sample_task(self, rng: np.random.Generator) -> TrajectoryDataset:
        """Sample a random variant as a meta-learning task."""
        name = rng.choice(self.variant_names)
        return self.datasets[name]

    def get_task(self, name: str) -> TrajectoryDataset:
        """Get a specific variant dataset."""
        return self.datasets[name]
