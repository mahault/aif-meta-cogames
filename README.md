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
# Run unit tests (no cogames dependency needed)
python -m pytest tests/ -v -k "not integration"

# Run with cogames (requires cogames + mettagrid on Linux/AWS)
python -m pytest tests/ -v

# Live evaluation of AIF agent in CogsGuard
cogames eval -m cogsguard_arena.basic \
  -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy \
  -c 4 -e 3

# Collect trajectory data (for world model training)
python scripts/collect_trajectories_v3.py --episodes 10 --output data/trajectories
```

## Project Structure

```
src/aif_meta_cogames/
├── env/              # Environment wrappers and data loading
├── world_model/      # Neural world models (MLP, RNN)
├── meta_learning/    # MAML and task distribution
├── aif_agent/        # Active inference agents (discrete + neural)
│   ├── generative_model.py   # 18-state POMDP (phase×hand)
│   ├── discretizer.py        # Token obs → discrete modalities
│   └── cogames_policy.py     # AIFPolicy(MultiAgentPolicy) for live play
└── evaluation/       # Metrics and baselines
```

## Current Status (March 2026)

- **Phase 3 COMPLETE**: Discrete AIF agent runs live in CogsGuard. 18-state POMDP with pymdp JAX. Hybrid architecture: pymdp beliefs + rule-based navigation. 26/26 tests pass.
- **Phase 0**: Trajectory data v3 collected (3,600 episodes, 36 variants, trained agents).
- **Phases 1-2**: World model training and meta-learning in progress (Luca leading).
- **Phase 4**: Neural AIF (merge world model + AIF agent) — next milestone.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full development plan and [docs/DESIGN.md](docs/DESIGN.md) for technical architecture.
