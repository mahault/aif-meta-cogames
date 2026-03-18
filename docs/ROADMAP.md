# Roadmap: AIF Meta-Learning on CoGames

**Mahault Albarracin, Luca, Alejandro** | March 2026

---

## Vision

Build an active inference agent that meta-learns a world model across diverse cooperative multi-agent environments, enabling rapid adaptation to unseen environment variants. Test on the CoGames (CogsGuard) benchmark.

**Core thesis**: Meta-learning shared structure across environments (transition dynamics, economy chains) gives the agent a strong prior. Active inference then drives efficient exploration in new environments via epistemic value — no reward signal needed for adaptation.

---

## Phase 0: Infrastructure Setup (Week 1)

**Lead**: Alejandro | **Reviewer**: Luca

| Task | Description | Status |
|------|-------------|--------|
| Repo + CI | GitHub repo, pyproject.toml, ruff, pytest | Done |
| Environment wrapper | Wrap cogames env into clean `(obs, action, next_obs, reward, done)` interface | TODO |
| Observation encoder | Token obs `(200, 3)` uint8 → flat vector or learned embedding | TODO |
| Trajectory dataset | PyTorch Dataset loading `.npz` files from data collection | TODO |
| **Trajectory data v3** | **3,600 episodes with trained agents across 36 variants** | **Done** |
| Variant sampler | Sample env configs for meta-learning task distribution | TODO |
| Abstract interfaces | `WorldModel` base class so parallel work can proceed | TODO |

**Data (v3 — March 2026)**:
- **36 environment variants** × 100 episodes = **3,600 episodes** (2,100,000 steps)
- **Trained agent policies**: Heterogeneous team using kickstarted LSTM checkpoints (miner, aligner, scrambler) — NOT random/biased-move
- Agent counts: n=4 (1M+1A+2S) and n=8 (2M+3A+3S); n=2 skipped to ensure all 3 roles present
- Variants: 30 arena (2 sizes × 3 difficulties × 5 terrains) + 6 machina1 (2 sizes × 3 difficulties)
- EnergizedVariant: agents stay alive for full episodes (500-1000 steps)
- Format: compressed `.npz` with `obs(T, N, 200, 3) uint8`, `actions(T, N) int32`, `rewards(T, N) float32`, `dones(T, N) bool`
- Size: **440MB** compressed (`trajectory_data_v3.tar.gz`)
- Previous v2 dataset (54 variants × 50 episodes, biased-move policy, 1.58M steps) also available

**Done when**: `trajectory_dataset.py` loads data and produces batches, `wrapper.py` steps a live env, tests pass.

---

## Phase 1: World Model Training (Weeks 2-3)

**Lead**: Luca | **Support**: Mahault

| Task | Description | Status |
|------|-------------|--------|
| Observation encoder | obs → z (64-dim latent). Two options: flat MLP or token transformer | TODO |
| MLP world model | `(z_t, a_t) → (z_{t+1}, r_t, done_t)` — simplest architecture | TODO |
| RNN world model | GRU-based for handling partial observability | TODO |
| Decoder | z → obs reconstruction (training signal for encoder) | TODO |
| Per-variant training | Train one model per variant — establishes upper bound | TODO |
| Cross-variant training | Train one model on all variants — establishes baseline | TODO |

**Key question**: Does the cross-variant model degrade meaningfully vs per-variant? If yes → meta-learning is motivated. If no → environments aren't diverse enough.

**Done when**: Prediction error validated on held-out episodes; measurable gap between per-variant and cross-variant models.

---

## Phase 2: Meta-Learning the World Model (Weeks 3-5)

**Lead**: Luca

| Task | Description | Status |
|------|-------------|--------|
| MAML implementation | Meta-learn world model initialization | TODO |
| Task distribution | Sample env variants as "tasks" for outer loop | TODO |
| Inner loop | K=5-10 gradient steps to adapt to new variant | TODO |
| Outer loop | Meta-gradient through inner loop, update initialization | TODO |
| Train/test split | ~30 variants meta-train, 6 held-out for meta-test | TODO |
| Baselines | Random-init, pre-trained (no meta), per-variant (upper bound) | TODO |

**MAML algorithm**:
```
for each meta-step:
    sample K environment variants (tasks)
    for each task:
        inner loop: adapt world model on support episodes (few gradient steps)
        evaluate adapted model on query episodes
    outer loop: meta-gradient across all tasks, update initialization
```

**Meta-test held-out variants** (test generalization along each axis):
- machina1 + 8 agents + medium (hardest combo)
- arena + desert biome
- arena + 8 agents + easy

**Done when**: MAML-init adapts to held-out variants in <10 gradient steps, matching per-variant model quality.

---

## Phase 3: Discrete AIF Agent (Weeks 3-5, parallel with Phase 2)

**Lead**: Mahault

| Task | Description | Status |
|------|-------------|--------|
| Generative model | 216-state factored POMDP: phase(6) × hand(3) × target_mode(3) × role(4) | TODO |
| A/B/C/D matrices | Hand-crafted for CogsGuard economy chain | TODO |
| pymdp agent | Discrete AIF with EFE (risk + ambiguity + epistemic) | TODO |
| Cogames integration | Implement `MultiAgentPolicy` interface for live env | TODO |
| Observation discretizer | Map token obs → discrete observation modalities | TODO |
| Port variational engine | Adapt from social-layer repo | TODO |

**State space**: 216 states capture the economy chain (EXPLORE → MINE → DEPOSIT → CRAFT → GEAR → CAPTURE) × inventory state × territorial context × role.

**Key advantage over PPO**: Epistemic value creates a gradient toward unexplored states (gear stations) even when no reward signal exists there. This directly addresses the "gear wall" — 43 PPO experiments, 2.3B steps, gear acquisition always zero.

**Done when**: AIF agent runs in cogames live environment, selects actions from EFE minimization, explores more state space than random/biased-move baselines.

---

## Phase 4: Neural AIF + Meta-Learned World Model (Weeks 5-7)

**Lead**: All three

| Task | Description | Status |
|------|-------------|--------|
| Neural AIF agent | Uses learned world model as generative model | TODO |
| EFE in latent space | A=decoder, B=transition, C=hand-crafted preferences | TODO |
| Epistemic value | Ensemble disagreement or encoder variance | TODO |
| MAML integration | Initialize world model from meta-learned params | TODO |
| Adaptation loop | On new variant: few gradient steps → adapted world model → AIF acts | TODO |

**This is the merge**: The world model (Phase 1) meta-learned initialization (Phase 2) becomes the generative model for the AIF agent (Phase 3). The neural agent:
1. Encodes observations: `z_t = encoder(obs_t)`
2. Predicts next states: `z_{t+1} = transition(z_t, a_t)` — this IS the B matrix
3. Computes EFE in latent space with epistemic uncertainty
4. Starts from MAML initialization → fast adaptation

**Done when**: Neural AIF with meta-learned WM outperforms discrete AIF on held-out variants AND adapts faster than non-meta-learned neural AIF.

---

## Phase 5: Experiments & Paper (Weeks 7-9)

| Experiment | What | Tests |
|-----------|------|-------|
| E1: WM sanity | MLP on 3 easy variants (100 episodes each) | Architecture works |
| E2: WM comparison | MLP vs RNN on all 36 variants (3,600 episodes) | Best world model |
| E3: Meta-learning | MAML vs None on 30/6 train/test split | H1: MAML helps adaptation |
| E4: Discrete AIF | pymdp on 3 arena variants | H3: Epistemic value helps |
| E5: Neural AIF | MLP+MAML+Neural AIF on all | H2: Better WM → better agent |
| E6: Ablation | ±epistemic value on 6 test variants | H3: Epistemic value contribution |

### Hypotheses

- **H1**: MAML-init world model adapts in fewer steps than random-init or pre-trained
- **H2**: Better world model → better AIF exploration (discovers gear stations faster)
- **H3**: Epistemic value drives 50%+ more state coverage than pragmatic-only
- **H4**: Meta-learned model captures task-invariant structure (economy chain logic)

### Metrics

| Category | Metric |
|----------|--------|
| World model | Prediction MSE, multi-step rollout divergence |
| Adaptation | Steps-to-threshold, shots-to-quality |
| Agent | Tiles visited, entities encountered, gear acquisition rate, junction captures |
| Transfer | Representation similarity across variants, per-factor adaptation analysis |

---

## Ownership Summary

| Component | Primary | Reviewer |
|-----------|---------|----------|
| `env/`, scripts, CI, configs | Alejandro | Luca |
| `world_model/`, `meta_learning/` | Luca | Mahault |
| `aif_agent/`, generative model | Mahault | Luca |
| `evaluation/`, notebooks | All | All |

---

## Dependencies

- **cogames** (pip): CoGames environment, MettaGrid simulator
- **pymdp**: Discrete active inference (pymdp.Agent)
- **PyTorch**: Neural world models, MAML
- **social-layer** (Mahault's repo): Variational engine, generative model template, affect engine

---

## Key References

- Kirsch & Schmidhuber (2022): Meta-train Transformer on offline RL datasets
- Finn et al. (2017): MAML — Model-Agnostic Meta-Learning
- Hafner et al. (2023): DreamerV3 — world model learning across 150+ tasks
- Mazzaglia et al. (2025): R-AIF — active inference for sparse reward tasks
- Friston et al. (2017): Active Inference, Curiosity and Insight
- Albarracin et al. (2026): Empathy Modeling in Active Inference
