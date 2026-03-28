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

## Phase 3: Discrete AIF Agent — v1 Prototype (Weeks 3-5) — COMPLETE

**Lead**: Mahault | **Completed**: 2026-03-17

| Task | Description | Status |
|------|-------------|--------|
| Generative model | 18-state POMDP: phase(6) × hand(3) | **Done** |
| A/B/C/D matrices | Hand-crafted for CogsGuard economy chain | **Done** |
| pymdp agent | Discrete AIF with EFE via pymdp JAX v1.0 | **Done** |
| Cogames integration | `AIFPolicy(MultiAgentPolicy)` — live play in CogsGuard | **Done** |
| Observation discretizer | Token obs → 3 discrete modalities | **Done** |
| Hybrid navigator | Rule-based movement toward AIF-selected targets | **Done** |

### Results

- Hearts: 6.0/agent, Aligner gear: 0.75, Junctions: 0
- Better than random, worse than starter

### Design Oversight Identified

The v1 prototype mapped the POMDP action space to 5 primitive movement actions (noop, N, S, E, W). Since phase/hand transitions happen automatically when stepping onto tiles, the B matrix is **action-independent** — identical for all 5 actions. When B is action-independent, pymdp computes the same EFE for every action → uniform distribution → effectively random action selection. The "hybrid navigator" works around this by bypassing pymdp's planning entirely and using hardcoded heuristics for goal selection.

**This is fixed in Phase 3b.**

---

## Phase 3b: Discrete AIF Agent — Deep AIF v9.7 (288-State, Two Nested POMDPs) — IN PROGRESS

**Lead**: Mahault | **Started**: 2026-03-19

### Core Architecture: Two Nested POMDPs + Shared Beliefs

**Level 2** (Strategic POMDP): 288-state factored POMDP with 5 macro-options (25 two-step policies). Replans at option termination (~42ms).
**Level 1** (OptionExecutor): Reactive state machines with role filter (miners≠CRAFT/CAPTURE, aligners MINE→CRAFT, scouts≠MINE/CRAFT/CAPTURE).
**Level 0.5** (EFE Goal Generation): G(e)=D_KL(Q(res|mine_e)||C_uniform) — scarcest element = lowest EFE.
**Level 0** (Navigation POMDP): 16-state, 5 relative actions, 25 two-step policies, online B-learning (~3-5ms/step).

| Task | Description | Status |
|------|-------------|--------|
| Task-level action space | 13 policies → 5 macro-options (mine_cycle, craft_cycle, capture_cycle, explore, defend) | Done |
| 288-state space | phase(6) × hand(4) × target_mode(3) × role(4) = 288 | Done |
| 6 observation modalities | resource, station, inventory(4), contest, social, role_signal | Done |
| Action-dependent B matrices | Hand-crafted B with distinct transitions per macro-option | Done |
| Navigation POMDP | 16-state nav with B-learning, frontier exploration | Done |
| SharedSpatialMemory | Belief sharing (Catal et al. 2024) — stations + explored cells | Done |
| Element-typed world model | `extractor:carbon`, `extractor:silicon` etc. + EFE element selection | Done |
| HOLDING_BOTH (v9.6) | 4-state hand: EMPTY, HOLDING_RESOURCE, HOLDING_GEAR, HOLDING_BOTH | Done |
| Scout epistemic agent (v9.7) | Near-uniform C → epistemic EFE, E-vector precision gate, explored cell sharing | Done |
| Aligner MINE redirect (v9.7) | MINE→CRAFT_CYCLE (not EXPLORE) — fixes stuck loop | Done |
| EXPLORE fallback (v9.7) | SharedSpatialMemory fallback when local frontier exhausted | Done |
| Mock realism (v9.7) | No pre-seeded stations, no free gear, discovery on adjacency | Done |
| Tests | 186 tests pass, mock eval: 7 captures/500 steps (no pre-seeded knowledge) | Done |
| AWS eval v9.7 (no_clips) | stuck=22.62, timeouts=3, all 4 resources mined, 0 junctions | Done |
| Junction captures | Aligners craft gear but never reach junctions — next focus | TODO |

### State Space (288 states)

```
phase(6):        EXPLORE, MINE, DEPOSIT, CRAFT, GEAR, CAPTURE
hand(4):         EMPTY, HOLDING_RESOURCE, HOLDING_GEAR, HOLDING_BOTH
target_mode(3):  FREE, CONTESTED, LOST
role(4):         GATHERER, CRAFTER, CAPTURER, SUPPORT
```

With action-dependent B matrices, all four factors are meaningful:
- **phase × hand**: Economy-chain progress — HOLDING_BOTH enables gear+resource co-holding
- **target_mode**: Junction contest status — affects EFE for CAPTURE vs YIELD
- **role**: Agent specialisation — affects which task policies are preferred via C

### Action Space (5 macro-options, resolved to 13 task policies)

| Macro-Option | OptionExecutor Steps | Key Transitions |
|-------------|---------------------|-----------------|
| MINE_CYCLE | NAV_RESOURCE → MINE → NAV_DEPOT → DEPOSIT | hand: EMPTY → RESOURCE → BOTH(if gear) → EMPTY/GEAR |
| CRAFT_CYCLE | NAV_CRAFT → CRAFT → NAV_GEAR → ACQUIRE_GEAR | hand: EMPTY → GEAR, RESOURCE → BOTH |
| CAPTURE_CYCLE | NAV_JUNCTION → CAPTURE | hand: GEAR → EMPTY, BOTH → RESOURCE |
| EXPLORE | Frontier exploration via nav POMDP | Epistemic foraging |
| WAIT | Noop | Self-transition |

### Observation Modalities (6)

| Modality | Values | Source |
|----------|--------|--------|
| o_resource(3) | NONE, NEAR, AT | extractor tag proximity |
| o_station(4) | NONE, HUB, CRAFT, JUNCTION | station tag proximity |
| o_inventory(4) | EMPTY, HAS_RESOURCE, HAS_GEAR, HAS_BOTH | global inventory tokens |
| o_contest(3) | FREE, CONTESTED, LOST | junction + agent:group tokens |
| o_social(4) | ALONE, ALLY_NEAR, ENEMY_NEAR, BOTH | agent:group tokens |
| o_role_signal(2) | SAME_ROLE, DIFFERENT_ROLE | vibe tokens |

### Why This Fixes the Problem

With 13 task-level actions, B is action-dependent:
- `B[:, s, NAV_RESOURCE]` → high prob of transitioning to MINE
- `B[:, s, MINE]` → high prob of gaining HOLDING_RESOURCE
- `B[:, s, WAIT]` → high prob of staying in current state

pymdp evaluates EFE for each task policy → different expected free energies → non-uniform policy selection → meaningful active inference planning.

### Architecture

```
Observation → Discretizer → 6 modalities
                                ↓
                          pymdp.Agent
                     (belief update + EFE)
                                ↓
                     Task policy selection
                                ↓
                     Navigator (spatial movement)
```

### AWS Eval Results

| Metric | v9.5c | v9.6 | v9.7 (no_clips) |
|--------|-------|------|-----------------|
| max_stuck | 1160 | 985 | **22.62** |
| timeouts | ~100 | 182 | **3** |
| carbon.gained | 17 | 0.25 | **11.50** |
| silicon.gained | 22.62 | 16.38 | **23.88** |
| germanium.gained | 0 | 0 | **9.00** |
| oxygen.gained | 0 | 0 | **6.38** |
| aligner.gained | 0 | 0 | **1.75** |
| miner.gained | 0 | 0 | **1.88** |
| junction.aligned | 0 | 0 | 0 |
| death | - | 3.0 | **1.25** |
| shared_stations | 0 | 0 | **83-96** |
| move.failed | - | 4280 | **2511** |

**v9.7 analysis**: All subsystems working — mining, depositing, heart withdrawal, gear crafting, spatial exploration, belief sharing. Remaining bottleneck: aligners craft gear but get stuck navigating to junctions (CRAFT_CYCLE timeout up to 192 steps). Junction capture is the last missing piece of the economy chain.

### MAML Integration (for Luca)

The A/B matrices at 288 states are the meta-learning target:
- **Inner loop**: Fit A/B from 2-3 episodes of a new variant
- **Outer loop**: Learn initialization that adapts fastest across variants
- **Task distribution**: 36 CogsGuard variants (30 train / 6 test)
- B matrix: factored as phase(6×6×4×5), hand(4×6×4×5), target_mode(3×3×5), role(4×4×5) — sparse but structured
- More trajectory data can be collected as needed

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

**This is the merge**: The world model (Phase 1) meta-learned initialization (Phase 2) becomes the generative model for the AIF agent (Phase 3b). The neural agent:
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
| E4: Discrete AIF | pymdp on 3 arena variants (216-state, 13 task policies) | H3: Epistemic value helps |
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
