# aif-meta-cogames

Meta-learning world models with active inference for the CoGames benchmark.

## Overview

This project combines **meta-learning** (MAML-style) with **active inference** to build agents that rapidly adapt to new multi-agent cooperative environments. We meta-learn a neural world model across diverse CoGames environment variants, then use active inference (EFE minimization) for exploration and planning.

**Core idea**: The outer loop meta-learns a world model initialization that captures shared structure across environments. The inner loop adapts that model to a new environment in a few gradient steps. Active inference drives exploration by seeking informative observations (epistemic value), not reward signals.

## Team

- **Mahault Albarracin** - Active inference agent, generative model, project lead
- **Luca** - Meta-learning algorithms, world model training
- **Alejandro** - Infrastructure, experiments, evaluation
- **Daniel Friedman** - Evaluation, theoretical guidance

## Architecture

```
    CoGames Environment (mettagrid)
               |
               v
    Observation (200 tokens, uint8)
               |
               v
    ObservationDiscretizer -> 6 modalities
               |
               v
    Level 2: Strategic POMDP (288 states, 5 macro-options)
    pymdp.Agent -- belief update + EFE policy selection
               |
               v
    Level 1: OptionExecutor (reactive state machines)
    maps macro-option + obs -> task policy (13 policies)
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

# Evaluate with learned parameters (Phase B)
AIF_LEARNED_PARAMS=/tmp/learned_full.npz \
  cogames eval -m arena -v no_clips \
  -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy \
  -c 4 -e 5 --action-timeout-ms 1000

# Enable online A-learning (Dirichlet updates during play)
AIF_LEARN_A=1 cogames eval -m arena -v no_clips \
  -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy \
  -c 4 -e 5 --action-timeout-ms 1000

# Offline parameter learning (joint A+B+C)
python scripts/learn_parameters.py learn-full \
  --trajectory /tmp/aif_traj.npz \
  --steps 300 --lr 0.001 --c-weight 0.5 \
  --output /tmp/learned_full.npz

# De novo learning (Friston 2025 -- Dirichlet + BMR)
python scripts/denovo_learn.py learn \
  --trajectory /tmp/aif_traj.npz --bmr \
  --output /tmp/denovo_params.npz

# Differentiable BMR (gradient-norm pruning)
python scripts/differentiable_bmr.py prune \
  --trajectory /tmp/aif_traj.npz \
  --output /tmp/pruned_params.npz

# Compare multiple parameter sets
python scripts/differentiable_bmr.py compare \
  --trajectory /tmp/aif_traj.npz \
  --params /tmp/learned_full.npz /tmp/denovo_params.npz /tmp/pruned_params.npz
```

## Project Structure

```
src/aif_meta_cogames/
|-- env/              # Environment wrappers and data loading
|-- world_model/      # Neural world models (MLP, RNN)
|-- meta_learning/    # MAML and task distribution
|-- aif_agent/        # Active inference agents (discrete + neural)
|   |-- generative_model.py   # 288-state factored POMDP (phase x hand x target x role)
|   |-- discretizer.py        # Token obs -> 6 discrete modalities
|   +-- cogames_policy.py     # AIFPolicy(MultiAgentPolicy) -- two nested POMDPs
+-- evaluation/       # Metrics and baselines

scripts/
|-- learn_parameters.py       # Offline gradient-based A+B+C learning (VFE + inverse EFE)
|-- denovo_learn.py            # De novo learning: Dirichlet accumulation + BMR (Friston 2025)
|-- differentiable_bmr.py      # Differentiable BMR + model comparison
|-- collect_trajectories_v3.py # Trajectory data collection with trained agents
+-- sweep/                     # Cortex/PPO hyperparameter sweep infrastructure

tests/
+-- test_cogames_policy.py     # 217 tests (discretizer, POMDP, nav, parameter learning)
```

## Parameter Learning Pipeline

The project implements a complete differentiable parameter learning pipeline for active inference agents, building on pymdp's fully JAX-differentiable EFE computation.

### Approaches

| Phase | What | Mechanism | Script |
|-------|------|-----------|--------|
| B-I | A matrices (perception) | VFE gradient descent | `learn_parameters.py learn` |
| B-II | B matrices (dynamics) | Transition prediction loss | `learn_parameters.py learn` |
| B-III | C vectors (preferences) | Inverse EFE (Shin et al. 2022) | `learn_parameters.py learn-c` |
| B-IV | Joint A+B+C | Two-timescale optimization | `learn_parameters.py learn-full` |
| B-V | De novo (Friston 2025) | Dirichlet accumulation + BMR | `denovo_learn.py learn` |
| B-VI | Differentiable BMR | Gradient-norm pruning + refinement | `differentiable_bmr.py prune` |

### Key Results (4-agent, arena no_clips)

| Parameter Set | Junctions/Agent | Method |
|--------------|----------------|--------|
| Hand-tuned (default) | 1.50 | Baseline |
| B5 (A-only, kl=1.0) | **2.20** | Multi-agent VFE gradient |
| B7 (online Dirichlet) | 1.80 | Bayesian conjugate updates |

## Current Status (April 2026)

- **Phase 3b COMPLETE**: Deep AIF agent (v9.9) -- 288-state factored POMDP, two nested POMDPs (strategic + navigation), 5 macro-options, online B-learning. **1.75 junctions/agent** (4-agent).
- **Phase 3c: Parameter Learning -- B-I through B-VI COMPLETE** (implementation + unit tests):
  - **B-I** (A matrices): Multi-agent VFE gradient descent. Best: B5 = **2.20 j/agent** (+47%).
  - **B-II** (B matrices): Transition prediction loss via factored B contraction.
  - **B-III** (C vectors): Inverse EFE behavioral cloning (Shin et al. 2022), per-role C.
  - **B-IV** (Joint A+B+C): Two-timescale optimization with separate Adam optimizers.
  - **B-V** (De novo): Dirichlet accumulation + Bayesian model reduction (Friston 2025).
  - **B-VI** (Differentiable BMR): Gradient-norm pruning, de novo init + gradient refinement, model comparison.
  - **217 tests pass** (all local, no mettagrid dependency).
- **Phase C** (plan broadcasting): Pending -- SharedSpatialMemory intent sharing for 8-agent deconfliction.
- **Phase 0**: Trajectory data v3 collected (3,600 episodes, 36 variants, trained agents).
- **Phases 1-2**: World model training and meta-learning (Luca leading).

See [docs/ROADMAP.md](docs/ROADMAP.md) for the full development plan and [docs/LITERATURE_REVIEW.md](docs/LITERATURE_REVIEW.md) for the academic survey (45 papers).

## References

- Friston et al. (2025). Gradient-Free De Novo Learning. *Entropy* 27(9):992.
- Shin, Kim & Hwang (2022). Prior Preference Learning from Experts. *ICML*.
- Da Costa et al. (2020). Active Inference on Discrete State-Spaces.
- Fountas et al. (2020). Deep Active Inference as Variational Policy Gradient.
- Tschantz et al. (2020). Learning Action-Oriented Models.
- Pitliya et al. (2025). Factorised Active Inference for Strategic Multi-Agent Interactions. *AAMAS*.
- Catal et al. (2024). Bayesian Multi-Agent Active Inference via Belief Sharing.
