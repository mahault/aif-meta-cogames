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
    CoGames Environment (mettagrid)
               |
               v
    Observation (200 tokens, uint8)
               |
               v
    ObservationDiscretizer → 6 modalities
               |
               v
    Level 2: Strategic POMDP (288 states, 5 macro-options)
    pymdp.Agent — belief update + EFE policy selection
               |
               v
    Level 1: OptionExecutor (reactive state machines)
    maps macro-option + obs → task policy (13 policies)
               |
               v
    Level 0: Navigation POMDP (16 states, 5 relative actions)
    online B-learning for obstacle avoidance
               |
               v
           Action (5)
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

# Live evaluation of AIF agent (4 agents, 5 episodes)
cogames eval -m arena -v no_clips \
  -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy \
  -c 4 -e 5 --action-timeout-ms 1000

# Evaluate with learned A matrices (Phase B)
AIF_LEARNED_PARAMS=/tmp/learned_params_B3.npz \
  cogames eval -m arena -v no_clips \
  -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy \
  -c 4 -e 5 --action-timeout-ms 1000

# Enable online A-learning (Dirichlet updates during play)
AIF_LEARN_A=1 cogames eval -m arena -v no_clips \
  -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy \
  -c 4 -e 5 --action-timeout-ms 1000

# Offline parameter learning from trajectory data
python scripts/learn_parameters.py learn \
  --trajectory /tmp/aif_traj.npz --multi-agent \
  --steps 200 --lr 0.001 --output /tmp/learned_params.npz
```

## Project Structure

```
src/aif_meta_cogames/
├── env/              # Environment wrappers and data loading
├── world_model/      # Neural world models (MLP, RNN)
├── meta_learning/    # MAML and task distribution
├── aif_agent/        # Active inference agents (discrete + neural)
│   ├── generative_model.py   # 288-state factored POMDP (phase×hand×target×role)
│   ├── discretizer.py        # Token obs → 6 discrete modalities
│   └── cogames_policy.py     # AIFPolicy(MultiAgentPolicy) — two nested POMDPs
└── evaluation/       # Metrics and baselines

scripts/
├── learn_parameters.py       # Phase B: offline gradient-based A/B learning
├── collect_trajectories_v3.py  # Trajectory data collection with trained agents
└── sweep/                    # Cortex/PPO hyperparameter sweep infrastructure
```

## Current Status (March 2026)

- **Phase 3b COMPLETE**: Deep AIF agent (v9.9) — 288-state factored POMDP, two nested POMDPs (strategic + navigation), 5 macro-options, online B-learning. **1.75 junctions/agent** (4-agent), 198/198 tests pass.
- **Phase 3c IN PROGRESS**: Parameter learning + social coordination.
  - **Phase A** (baseline stabilization): COMPLETE — 1.50 j/agent (4-agent), 0.12 j/agent (8-agent).
  - **Phase B** (differentiable parameter learning): IN PROGRESS — JAX gradient descent on VFE, multi-agent learning fixes role bias, online Dirichlet A-learning.
  - **Phase C** (plan broadcasting): Pending — SharedSpatialMemory intent sharing for 8-agent deconfliction.
- **Phase 0**: Trajectory data v3 collected (3,600 episodes, 36 variants, trained agents).
- **Phases 1-2**: World model training and meta-learning (Luca leading).

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full development plan.
