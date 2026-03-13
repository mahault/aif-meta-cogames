# aif-meta-cogames

Meta-learning world models with active inference for the CoGames benchmark.

## Overview

This project combines **meta-learning** (MAML-style) with **active inference** to build agents that rapidly adapt to new multi-agent cooperative environments. We meta-learn a neural world model across diverse CoGames environment variants, then use active inference (EFE minimization) for exploration and planning.

**Core idea**: The outer loop meta-learns a world model initialization that captures shared structure across environments. The inner loop adapts that model to a new environment in a few gradient steps. Active inference drives exploration by seeking informative observations (epistemic value), not reward signals.

## Team

- **Mahault Albarracin** - Active inference agent, generative model, project lead
- **Luca** - Meta-learning algorithms, world model training
- **Alejandro** - Infrastructure, experiments, evaluation

## Architecture

```
Trajectory Data (36 variants)          CoGames Environment
         |                                    |
         v                                    v
   Observation Encoder                 Observation Encoder
   (200 tokens) -> z_64               (200 tokens) -> z_64
         |                                    |
         v                                    v
   World Model                         AIF Agent
   f(z_t, a_t) -> z_{t+1}    ---->    Uses world model as
         |                             generative model (A, B)
         v                             Computes EFE
   MAML Meta-Learning                  Selects actions
   across 30+ task variants                 |
                                            v
                                        Action(5)
```

## Setup

```bash
# Clone
git clone https://github.com/mahault/aif-meta-cogames.git
cd aif-meta-cogames

# Install (editable)
pip install -e ".[dev]"

# Install with cogames environment support
pip install -e ".[dev,cogames]"
```

## Quick Start

```bash
# Collect trajectory data (requires cogames)
python scripts/collect_trajectories.py --episodes 10 --output data/trajectories

# Train world model on offline data
python scripts/train_world_model.py --data data/trajectories --model mlp

# Meta-train with MAML
python scripts/meta_train.py --data data/trajectories --inner-steps 5

# Evaluate adaptation on held-out variants
python scripts/evaluate_adaptation.py --checkpoint models/meta_model.pt
```

## Project Structure

```
src/aif_meta_cogames/
├── env/              # Environment wrappers and data loading
├── world_model/      # Neural world models (MLP, RNN)
├── meta_learning/    # MAML and task distribution
├── aif_agent/        # Active inference agents (discrete + neural)
└── evaluation/       # Metrics and baselines
```

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full development plan and [docs/DESIGN.md](docs/DESIGN.md) for technical architecture.
