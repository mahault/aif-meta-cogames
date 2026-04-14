# Roadmap: AIF Meta-Learning on CoGames

**Mahault Albarracin, Luca, Alejandro, Daniel Friedman** | April 2026

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
| Junction captures (v9.8) | NAV_GEAR fix + hub-proximity junction targeting → 0.75 j/agent | Done |

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

| Metric | v9.5c | v9.6 | v9.7 (no_clips) | v9.8 (junction fix) | v9.9 (C-vector + auto-chain) |
|--------|-------|------|-----------------|---------------------|-------------------------------|
| max_stuck | 1160 | 985 | **22.62** | **52.25** | **31.50** |
| timeouts | ~100 | 182 | **3** | **2** | **1** |
| carbon.gained | 17 | 0.25 | **11.50** | **3.00** | **6.00** |
| silicon.gained | 22.62 | 16.38 | **23.88** | **3.00** | **3.75** |
| germanium.gained | 0 | 0 | **9.00** | **3.00** | **6.00** |
| oxygen.gained | 0 | 0 | **6.38** | **7.00** | **12.75** |
| aligner.gained | 0 | 0 | **1.75** | **1.00** | - |
| miner.gained | 0 | 0 | **1.88** | **0.25** | **0.75** |
| heart.gained | - | - | - | - | **7.50** |
| junction.aligned | 0 | 0 | 0 | **0.75** | **1.75** |
| death | - | 3.0 | **1.25** | **0** | **0** |
| move.failed | - | 4280 | **2511** | **504** | - |
| reward | 0 | 0 | 0 | **0.91** | **1.27** |

**v9.8 analysis**: First junction alignment! Two fixes: (1) CRAFT_CYCLE now uses NAV_GEAR (role-specific station) instead of NAV_CRAFT (any station) — aligners reliably get aligner gear. (2) NAV_JUNCTION now prefers junctions within hub alignment radius (25 tiles). Reward 0.96-1.78 in 2/3 episodes.

**v9.9 analysis**: Junction captures +133% (0.75→1.75). Two fixes: (1) Aligner C vector corrected: c_inv[HAS_BOTH]=5.0 > c_inv[HAS_GEAR]=2.0 (was inverted at 2.0 < 5.0 — penalized capture-ready state). (2) Auto-chain: CRAFT→CAPTURE→CRAFT loop for aligners bypasses POMDP replan between phases. All 5/5 episodes score reward (was 2/3). Machina_1: 0.75j, 0.98 reward.

### Ablation: What Drives Junction Capture?

| Condition | Mechanism | Junctions | Reward | Principled? |
|-----------|-----------|-----------|--------|-------------|
| v9.8 baseline | old C (GEAR>BOTH) | 0.75 | 0.91 | yes (wrong C) |
| **C-only** | **fixed C, static E** | **1.50** | **0.85** | **yes** |
| C+E | fixed C, context-dep E | 0.50 | 0.35 | yes (E too strong) |
| v9.9 | fixed C, auto-chain | 1.75 | 1.27 | no (bypasses EFE) |

**Conclusions**:
- **C-vector correction is the primary driver**: +100% (0.75→1.50), fully principled AIF.
- **Auto-chain adds marginal +17%** (1.50→1.75) but bypasses `infer_policies()`.
- **Context-dependent E hurts** (-67%): E bias of ln(6.0/1.5)≈1.4 nats overrides EFE, causing wrong option selection. The static E with CRAFT=CAPTURE=4.0 already provides balanced habit priors; asymmetric context-switching disrupts this.
- **Recommendation**: Use C-vector fix as the principled baseline. Auto-chain kept as optional pragmatic improvement (`auto_chain=True`). Context-dependent E needs gentler biases or learned (not hand-tuned) values — candidate for meta-learning.

### MAML Integration (for Luca)

The A/B matrices at 288 states are the meta-learning target:
- **Inner loop**: Fit A/B from 2-3 episodes of a new variant
- **Outer loop**: Learn initialization that adapts fastest across variants
- **Task distribution**: 36 CogsGuard variants (30 train / 6 test)
- B matrix: factored as phase(6×6×4×5), hand(4×6×4×5), target_mode(3×3×5), role(4×4×5) — sparse but structured
- More trajectory data can be collected as needed

---

## Phase 3c: Parameter Learning + Social Coordination

**Lead**: Mahault | **Started**: 2026-03-31 | **Parameter Learning (B-I through B-VI): COMPLETE** (2026-04-01)

### Motivation

v9.9 achieves 1.75 junctions/agent (4-agent) and 0.62 j/agent (8-agent). The remaining gap
comes from two sources:
1. **Hand-tuned parameters**: A, B, C, D matrices are hand-crafted — suboptimal for actual
   environment dynamics. The agent's internal model doesn't match reality.
2. **No social coordination**: 8 agents compete for the same targets (3 aligners go to
   the same junction). No plan broadcasting, no theory of mind, no coordination.

### Phased Approach

Parameter learning (B-I through B-VI) builds incrementally, then social coordination (C-F):

| Phase | Goal | Key Mechanism | Status |
|-------|------|--------------|--------|
| A | Stabilize baseline | Revert nav regression, confirm 1.50j (4-agent) | **COMPLETE** |
| B-I | Learn A matrices | JAX gradient descent on VFE over trajectories | **COMPLETE** |
| B-II | Learn B matrices | Transition prediction loss (VFE complexity term) | **COMPLETE** |
| B-III | Learn C vectors | Inverse EFE / behavioral cloning (Shin et al. 2022) | **COMPLETE** |
| B-IV | Joint A+B+C optimization | Two-timescale loss with separate Adam optimizers | **COMPLETE** |
| B-V | De novo learning (literal) | Friston (2025): Dirichlet accumulation + BMR | **COMPLETE** |
| B-VI | De novo-inspired gradient | Differentiable BMR + model comparison | **COMPLETE** |
| C | Plan broadcasting | Agents share intended targets to avoid overlap | Pending |
| D | Bayesian model comparison | Compare learned vs hand-tuned via VFE evidence | Pending |
| E | Factorised social AIF | Joint preferences, factored multi-agent beliefs | Pending |
| F | Full theory of mind | Recursive agent models via sophisticated inference | Pending |

**See also**: [docs/LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) for the full academic survey
(45 papers) informing these phases.

---

### Phase A: Stabilize Baseline

**Problem**: Experimental nav changes (stronger B-bias, lower pB_scale, 4-step oscillation
detection) caused regression from 2.00 to 0.50 junctions. Also discovered critical bug:
`n_agents` was always 8 regardless of `-c` flag, so role optimization for n=4 was never used.

**Fixes applied**:
- Reverted B-matrix TOWARD bias to +0.10 (was +0.25)
- Reverted pB_scale to 5.0 (was 2.0)
- Reverted oscillation detection to 2-step inline (was 4-step _detect_blocked)
- Fixed n_agents auto-detection: `n_agents = getattr(policy_env_info, "num_agents", 8)`

**Results (2026-03-31)**:
- 4-agent: **1.50 j/agent** (matches C-only ablation, stable baseline)
- 8-agent: **0.12 j/agent** (3 aligners compete for same targets — needs Phase C)
- Status: **COMPLETE**

---

### Phase B-I: A Matrix Learning (Differentiable VFE)

**Why**: Hand-tuned A/B/C/D matrices are the #1 bottleneck. pymdp 1.0 is fully
JAX-differentiable, enabling `jax.grad(VFE)` through the inference pipeline.

**Infrastructure built**:
- Trajectory logging in `BatchedAIFEngine` (obs, beliefs, priors, actions per step)
- Custom `_compute_vfe_factored()` for factored A matrices with `A_DEPENDENCIES`
- Softmax parameterization: unconstrained logits θ → valid distributions via `softmax(θ, axis=0)`
- `scripts/learn_parameters.py`: gradient descent, Bayesian model comparison, save/load
- 8 tests (3 trajectory logging + 5 parameter learning), all passing

**Experiment sweep (2026-03-31)**:

| ID | Method | Agent(s) | KL reg | VFE reduction | Junctions | Status |
|----|--------|----------|--------|---------------|-----------|--------|
| v1 | Single-agent gradient | agent 0 (miner) | 0.0 | -5.5% | **0** (regression) | Done |
| v2 | Single-agent gradient (aggressive) | agent 0 (miner) | 0.0 | -37.4% | not tested | Done |
| B3 | **Multi-agent gradient** | **all 4** | **0.0** | **-10.1%** | **1.40** (no regression) | Done |
| B4 | Multi-agent + mild KL | all 4 | 0.1 | -10.1% | pending eval | Done |
| B5 | **Multi-agent + strong KL** | **all 4** | **1.0** | **-10.0%** | **2.20** (+47%) | **Done** |
| B6 | Single aligner (agent 1) | agent 1 | 0.0 | -10.5% | **0** (regression) | Done |
| B7 | Online Dirichlet (`learn_A=True`) | all (live) | N/A | N/A | **1.80** (+20%) | Done |

**Critical finding: role-biased learning corrupts A matrices**

When learning from a single agent's trajectory (v1/v2), the A matrix columns for
observation values that agent rarely encounters drift toward noise. Analysis of
the 4-agent trajectory data:

```
Agent 0 (miner):   inv=[EMPTY=19%, HAS_RES=80%, HAS_GEAR=0%, HAS_BOTH=0%]
Agent 1 (aligner): inv=[EMPTY=0%, HAS_RES=45%, HAS_GEAR=0%, HAS_BOTH=54%]
Agent 2 (aligner): inv=[EMPTY=3%, HAS_RES=6%, HAS_GEAR=0%, HAS_BOTH=90%]
Agent 3 (scout):   inv=[EMPTY=3%, HAS_RES=96%, HAS_GEAR=0%, HAS_BOTH=0%]
```

The miner (agent 0) **never observes HAS_GEAR or HAS_BOTH**. Learning from its
trajectory degrades those A columns:

```
P(HAS_GEAR | state=HAS_GEAR):
  default: 0.941  →  v2: 0.668 (BROKEN — 15% prob of observing EMPTY!)
  default: 0.941  →  B3: 0.932 (preserved — multi-agent averaging)
```

Multi-agent learning (B3) averages VFE across ALL agents, so aligner data
keeps the HAS_GEAR/HAS_BOTH columns accurate. Result: **no regression**.

**B3 live eval results (4-agent, 5 episodes)**:

| Metric | B3 (learned) | Default |
|--------|-------------|---------|
| Mean reward | **1.15** | 1.00 |
| Junctions | 1.40 | 1.40 |
| Move failures | **910** | 1035 |
| Aligner gear | 1.25 | 2.50 |
| Zero-reward episodes | **0/5** | 1/5 |

B3 is more consistent (no zero-reward episodes) and has 12% fewer movement
failures, suggesting the learned A matrices improve navigation. Junction rate
is equal. More episodes needed for statistical significance.

**Full live eval results (4-agent, 5 episodes each)**:

| Metric | Default | B3 (multi, kl=0) | B5 (multi, kl=1.0) | B6 (aligner) | B7 (online) |
|--------|---------|-------------------|---------------------|--------------|-------------|
| Junctions | 1.40 | 1.40 | **2.20** | 0 | 1.80 |
| Hearts (net) | - | - | 4.00 | 0 | 3.75 |
| Aligner gear (net) | 2.50 | 1.25 | 2.25 | 0 | 1.75 |
| Move failures | 1035 | 910 | 940 | 927 | 955 |
| Zero episodes | 1/5 | 0/5 | - | - | - |

**Key findings from live evals**:
- **B5 is the clear winner**: +47% junctions over default (2.20 vs 1.50 baseline).
  Strong KL regularization (kl_weight=1.0) prevents A drift while allowing
  meaningful improvement. Multi-agent averaging prevents role bias.
- **B6 produces complete regression**: Aligner-only learning creates degenerate
  A matrices. The aligner trajectory (54% HAS_BOTH) warps columns for
  observation states other roles depend on. Heart/gear gained exactly equals
  lost (net zero) — the agent acquires and immediately loses everything.
- **B7 (online Dirichlet) works without pre-training**: 1.80 junctions (+20%)
  from pure online Bayesian learning during live play. No pre-collected data
  needed. Could be combined with B5 (start from learned params + continue
  online adaptation).
- **Navigation is not the bottleneck**: Move failures are similar across all
  variants (~910-955), confirming the improvement comes from better state
  estimation (A matrices), not navigation.

**B5/B6 A-matrix analysis**:

All three multi-agent runs (B3, B5, B6) converge to the same A[2] (inventory)
structure. KL regularization has negligible effect at this scale:

```
P(HAS_GEAR|GEAR):   default=0.941  B3=0.932  B5=0.934  B6=0.923
P(HAS_BOTH|BOTH):   default=0.941  B3=0.959  B5=0.959  B6=0.959
Max A[2] diff:       --             0.018     0.017     0.019
Overall max diff:    --             0.098     0.086     0.098
```

B6 (aligner-only, agent 1) preserves inventory structure despite being single-agent
because aligners observe all inventory states (HAS_RES=45%, HAS_BOTH=54%).

**Technical findings**:
1. **Static-leaf JIT issue disproven**: All 33 agent leaves are dynamic (verified
   via `eqx.partition`). The `eqx.tree_at` replacement works correctly through JIT.
2. **VFE minimization improves perception, not planning directly**: A matrices map
   P(observation | state). Better A → better state estimation → indirectly better EFE.
   C vectors (preferences) drive policy selection directly.
3. **KL regularization (B4, B5)**: Both kl_weight=0.1 and kl_weight=1.0 have
   negligible effect — nearly identical VFE and A matrices as B3 (kl=0.0).
   The learned A matrices naturally stay close to default with multi-agent learning.
4. **VFE doesn't signal junction captures**: VFE increases on inventory transitions
   (surprise signal). Junction capture (HAS_BOTH→HAS_RES) produces a negligible
   VFE drop (-0.08). Junction value is encoded in C (preferences), not A.
5. **C-vector optimization is out of scope for VFE-based learning**: C affects EFE
   (planning) not VFE (inference). The hand-tuned C vectors (miner/aligner/scout)
   are already well-optimized — the v9.8→v9.9 ablation showed C-vector correction
   alone gives +100% junctions. Further C optimization would require black-box
   search (CMA-ES) with junction captures as the objective, which is expensive.

**B7: Online Dirichlet A-learning** (implemented):
- `AIF_LEARN_A=1` enables Dirichlet updates on the strategic POMDP A matrices
  during live play. Uses pymdp's built-in `infer_parameters()` with `pA` priors.
- Most conservative approach: Bayesian conjugate updates, no gradient descent.
- Complements offline learning (B3-B6) by adapting to the actual environment
  dynamics rather than pre-collected trajectory statistics.
- Implementation: `learn_A=True` in `create_strategic_agent`, `pA_scale=5.0`
  controls prior strength. Updates via `_update_parameters()` every 50 steps.

**References**:
- Da Costa et al. (2020): Active Inference on Discrete State-Spaces — Dirichlet parameter learning
- Fountas et al. (2020): Deep Active Inference — end-to-end differentiable AIF
- Sajid et al. (2021): Active Inference: Demystified and Compared — parameter learning review

---

### Phase B-II: B Matrix Learning (Transition Prediction Loss) — COMPLETE

**Why**: Phase B-I only learned A matrices. B matrices received **zero gradient**
because `trajectory_vfe()` line 173 explicitly discards B_logits (`_ = B_logits`).
All saved B matrices from B3-B7 are identical to defaults.

**Theoretical basis**: B learning from VFE is well-established (Friston et al. 2016,
Da Costa et al. 2020). The proper objective is the transition prediction cross-entropy:

```
L_B = -sum_t E_q(s_t) E_q(s_{t+1}) [ln B(s'|s, a)]
```

This is the VFE complexity term — it penalizes the KL between inferred posterior
over next state and the B-predicted prior. Equivalent to the Bayesian Dirichlet
update `alpha' = alpha + q(s') outer q(s) * 1(a)` but in gradient form.

**Implementation plan**:

1. **Add `_compute_transition_loss_factored()`** to `learn_parameters.py`:
   - For each factor f: predict P(s'_f | s_deps, a) via B contraction with q(s_deps)
   - Loss = -sum_f sum_{s'} q(s'_f, t+1) * ln P_predicted(s'_f)
   - Uses consecutive (qs_t, qs_{t+1}, action_t) from existing trajectory data

2. **Replace `_ = B_logits`** with actual B usage in `trajectory_vfe()`:
   - B enters through the transition prediction loss
   - Use **option-level B** (5 macro-options from `build_option_B()`), not task-level (13)
   - Freeze B_role (identity — roles never change)

3. **Factor-aware learning**:
   - B_phase (6,6,4,5) and B_hand (4,6,4,5) have joint dependencies on [phase, hand]
   - B_target (3,3,5) depends only on target_mode
   - B_role (4,4,5) is identity — freeze (zero gradient mask)

**Key insight from literature**: Tschantz et al. (2020) show learned B matrices need
not match true dynamics — only task-relevant transitions matter. Our factored B
naturally captures this: B_role stays identity, B_phase/B_hand learn economy chain.

**Validation**:
- `test_transition_loss_gradient_exists` — verify dL/dB non-zero for phase, hand, target
- `test_B_converges_toward_correct_transitions` — synthetic known-B recovery
- Live eval with learned B vs default: compare junctions and move failures

---

### Phase B-III: C Vector Learning (Inverse EFE) — COMPLETE

**Why**: C vectors (prior preferences over observations) drive policy selection
through EFE but **do not appear in VFE**. Therefore C cannot be learned from VFE —
it requires an EFE-based objective.

**Theoretical basis**: Shin, Kim & Hwang (2022) "Prior Preference Learning from
Experts" treat EFE as a negative value function and recover C from demonstrations
via MaxEnt IRL. The formal equivalence: `ln P(o|C)` in AIF = `r(s,a)` in MaxEnt RL
(Levine 2018, Millidge et al. 2020).

**Key principle**: Find C such that the EFE-optimal policy reproduces observed
(successful) behavior:

```
Loss_C = -ln q_pi(a_observed)
where q_pi = softmax(gamma * neg_G + ln E)
and G depends on C through compute_expected_utility(qo, C)
```

This is cross-entropy between the observed policy and the EFE-derived policy
distribution. The full pymdp EFE computation is differentiable in JAX — no
`stop_gradient` anywhere. C enters via `compute_expected_utility(qo, C) = sum_m qo[m] * C[m]`.

**Implementation plan**:

1. **Add `_efe_policy_loss()`** to `learn_parameters.py`:
   - At each replan step (when macro-option changes): extract qs and observed_option
   - Reconstruct a pymdp Agent with current A, B, C, E
   - Call `infer_policies(qs)` → get q_pi over 25 two-step policies
   - Loss = -ln q_pi[observed_option_policy_index]
   - Per-role C: use agent role to select miner/aligner/scout C for each agent

2. **Trajectory augmentation** (prerequisite):
   - Modify `_select_option` in `cogames_policy.py` to store neg_efe and q_pi
   - Record when replanning occurs (option termination events)
   - Needed: which option was chosen, the beliefs at decision time

3. **C parameterization**:
   - 6 C vectors (one per obs modality) × 3 roles = 18 learnable vectors
   - Unconstrained parameterization (C can be any real number — log-preferences)
   - Lower learning rate: 0.1× base LR (C is sensitive through softmax)

4. **Circular dependency mitigation**:
   - Learning C from trajectories collected with hand-tuned C would recover
     the hand-tuned values (tautological)
   - **Solution**: First learn A+B (B-I + B-II), collect NEW trajectories with
     learned A+B (different behavior), then learn C from improved trajectories
   - Alternative: Learn C from RL agent trajectories (PPO-trained), which use
     a different policy mechanism (reward-based) — breaks the circularity

**EFE degeneracy warning** (Champion et al. 2023): Monitor for entropy collapse
in q_pi during training. If the agent "becomes an expert at predicting outcomes
for a single action," the epistemic term degenerates. Use entropy regularization
on q_pi as a safeguard.

**Validation**:
- `test_efe_policy_loss_gradient_exists` — verify dL/dC non-zero
- `test_C_gradient_increases_observed_option_preference` — direction check
- Live eval with learned C: compare junctions vs B5 baseline (2.20 j/agent)

---

### Phase B-IV: Joint A+B+C Optimization — COMPLETE

**Why**: Learning A, B, C separately may miss interactions. Joint optimization
lets the model find parameter configurations that work together.

**Theoretical basis**: The two-timescale approach recommended by the deep AIF
literature (Millidge 2020, Fountas et al. 2020): fast VFE updates for world model
(A, B), slow EFE updates for preferences (C).

**Combined loss**:

```
L_total = w_A * L_vfe(A)           # Perception: VFE accuracy term
        + w_B * L_transition(B)    # Dynamics: VFE complexity term
        + w_C * L_policy(A,B,C)    # Preferences: inverse EFE cross-entropy
        + w_kl * L_regularization  # Prevent drift from priors
```

**Implementation plan**:

1. **New `learn_full_parameters()`** in `learn_parameters.py`:
   - Separate Adam optimizers per parameter group: A, B, C
   - A and B: base learning rate (0.001)
   - C: 0.1× learning rate (scale sensitivity)
   - Multi-agent averaging across all roles (prevents role bias)
   - Per-role C: miner/aligner/scout learned independently

2. **Training schedule**:
   - Phase 1 (steps 0-100): A+B only (w_C=0) — establish world model
   - Phase 2 (steps 100-300): A+B+C jointly — add preference learning
   - Phase 3 (steps 300-500): All with reduced LR — fine-tuning

3. **New CLI**: `python learn_parameters.py learn-full --trajectory X --output Y`
   - Flags: `--a-weight`, `--b-weight`, `--c-weight`, `--kl-weight`
   - `--schedule staged` (default) or `--schedule joint` (all from start)
   - `--c-source {same,rl}` — use same trajectory or RL agent trajectory for C

**Validation**:
- `test_joint_loss_all_params_update` — A, B, C all receive gradients
- Compare staged vs joint schedule on junction captures
- Compare with B5 (A-only, 2.20j) — target: exceed B5

---

### Phase B-V: De Novo Learning — Literal (Friston 2025) — COMPLETE

**Why**: Gradient-based learning requires differentiable objectives and may get
stuck in local optima. Friston et al. (2025) "Gradient-Free De Novo Learning"
offers a complementary approach that learns **all parameters (A, B, C, D)** from
scratch without gradients, using structure discovery + Bayesian model reduction.

**Reference**: Friston, K. et al. (2025). "Gradient-Free De Novo Learning."
*Entropy*, 27(9), 992. [PMC12468873]

**The three-phase pipeline**:

#### Phase 1: Structure Discovery (Spectral Clustering)

Discover hidden state structure from raw observation trajectories:

1. Collect observation sequences from CogsGuard episodes
2. Compute similarity matrix over observation windows (e.g., cosine similarity
   of 10-step observation subsequences)
3. Spectral decomposition: eigenvalue gap determines number of hidden states
4. Assign observation windows to discovered clusters
5. Compare discovered structure with our hand-designed 4-factor decomposition

**Key question**: Does spectral clustering recover something like phase × hand ×
target_mode × role? Or a different factorization? This would validate (or challenge)
our hand-designed state space.

**Implementation**:
- `scripts/denovo_structure.py`: spectral clustering on trajectory observations
- Input: raw `.npz` trajectory data (3,600 episodes)
- Output: discovered state assignments, eigenvalue spectrum, cluster statistics
- Comparison: mutual information between discovered clusters and our hand-designed states

#### Phase 2: Parameter Learning (Dirichlet + BMR)

Given discovered (or hand-designed) structure, learn A, B, C, D via:

1. **A learning**: Count observation-state co-occurrences → Dirichlet concentration
   parameters. `alpha_A[i,j] += sum_t 1{o=i} * q(s=j)`
2. **B learning**: Count state transition co-occurrences → Dirichlet concentration.
   `alpha_B[j,k,l] += sum_t q(s'=j) * q(s=k) * 1{a=l}`
3. **D learning**: Count initial state beliefs → Dirichlet. `alpha_D[j] += q(s_1=j)`
4. **C learning via goal-state identification**:
   - Identify observation patterns associated with reward (junction captures)
   - Set C to prefer observations that precede successful capture events
   - This is NOT gradient-based — it's reward-conditioned counting

5. **Bayesian Model Reduction (BMR)**:
   - After accumulating parameters, prune A/B connections that don't contribute
   - Test: does removing A[m][i,j] increase VFE? If not, set to zero
   - Bidirectional: also test adding connections not in the current model
   - **Reachability analysis**: identify states that connect to goal states

**Implementation**:
- `scripts/denovo_learn.py`: Dirichlet accumulation + BMR pipeline
- Input: trajectory data + state assignments (from Phase 1 or hand-designed)
- Output: learned A, B, C, D + pruned model structure
- Uses pymdp's `dirichlet_expected_value()` for expected parameters

#### Phase 3: Refinement (Online + Inductive Inference)

Deploy the de novo-learned model and refine during live play:

1. Agent uses learned model for planning (EFE-based policy selection)
2. Online Dirichlet updates continue accumulating evidence
3. BMR periodically simplifies the model (every N episodes)
4. **Inductive inference**: successful actions generate positive evidence
   for the transitions/observations that led to them

**Implementation**:
- Extend `BatchedAIFEngine` with periodic BMR sweeps
- Monitor VFE trajectory: if VFE increases after simplification, revert
- Compare: de novo model vs gradient-learned model (B5) vs default

**Expected outcomes**:
- Structure discovery reveals whether 288 states is the right granularity
- Dirichlet-learned A/B should match gradient-learned A (B5) if both are correct
- C from goal-state identification provides an independent C estimate
  (compare with inverse-EFE C from Phase B-III)
- BMR may discover that some state factors/modalities are redundant

---

### Phase B-VI: De Novo-Inspired Gradient Approach — COMPLETE

**Why**: The de novo pipeline (B-V) uses Bayesian counting — slow convergence,
no gradient information. This phase adapts the de novo **ideas** into our JAX
differentiable framework, combining the best of both worlds.

**What we borrow from de novo**:

1. **Growth-reduction cycle** → Differentiable model selection:
   - Start with a simpler model (fewer states/factors)
   - Learn parameters via gradient descent (B-I through B-IV)
   - Evaluate via VFE: does adding a state factor improve VFE?
   - Differentiable BMR: use `jax.grad` through the VFE to identify
     which A/B connections have near-zero gradient (candidates for pruning)

2. **Spectral structure as initialization** → Better starting point:
   - Use spectral clustering (B-V Phase 1) to initialize A/B/D
   - Then refine with gradient descent (B-I through B-IV)
   - Hypothesis: spectral init + gradient refinement > random init + gradient

3. **Goal-conditioned C initialization** → Better C starting point:
   - Use reward-conditioned observation counting (B-V Phase 2) to initialize C
   - Then refine C with inverse EFE gradients (B-III)
   - Breaks the circular dependency: C starts from environment structure,
     not from hand-tuned values

4. **Automatic model complexity selection**:
   - Define a family of models: 72-state (3-factor), 288-state (4-factor),
     576-state (5-factor), etc.
   - Learn parameters for each via gradient descent
   - Compare via VFE evidence (Bayesian model comparison from Phase D)
   - Select the model with best VFE-performance trade-off

**Implementation plan**:

1. **Differentiable BMR** (`scripts/differentiable_bmr.py`):
   - Compute gradient norm ||dL/dA[m][i,j]|| for all A/B entries
   - Entries with ||grad|| < epsilon are candidates for pruning
   - Prune by setting to uniform + re-optimizing remaining parameters
   - Iterate until VFE stabilizes

2. **Spectral-initialized gradient learning** (`scripts/learn_parameters.py`):
   - New flag: `--init spectral` (vs `--init default` or `--init random`)
   - Load spectral structure from B-V Phase 1 output
   - Convert cluster assignments to initial A/B matrices
   - Run gradient optimization from spectral initialization

3. **Model family comparison** (`scripts/model_comparison.py`):
   - Define model specifications: {3-factor, 4-factor, 5-factor} × {gradient, denovo}
   - For each: learn parameters, evaluate VFE on held-out trajectory
   - Bayes factor comparison: exp(VFE_1 - VFE_2)
   - Paired with junction capture performance

**Expected outcomes**:
- Differentiable BMR identifies which A/B entries are truly informative
- Spectral init may accelerate convergence over default init
- Model comparison reveals optimal state space granularity
- Combined approach (de novo structure + gradient parameters) should
  outperform either alone

**This phase connects to Phase 4 (Neural AIF)**: If model comparison reveals
that discrete state spaces are too coarse, this motivates moving to neural
generative models where the state space is learned end-to-end.

---

### Deep Pipeline Results (2026-04-02)

**Setup**: 50,000 trajectory steps (50 episodes × 1000 steps), 2,000 gradient steps,
`policy_len=4` (625 policies), arena map, 4-agent.

| Method | VFE | Reduction | Notes |
|--------|-----|-----------|-------|
| **Joint A+B+C** | **59.5** | **74.1%** | Best ever. All parameters jointly optimized |
| A-only | 122.0 | 46.8% | Likelihood learning alone is strong |
| BMR | 144.1 | 37.2% | Bayesian model reduction baseline |
| Default | 229.5 | — | Hand-designed parameters |
| De novo | worse | — | Structure discovery alone insufficient |
| Refined | worse | — | Spectral init + gradient didn't help |

**Live arena evaluation** (joint, policy_len=4): mean reward **0.70 j/agent**
(episodes: 0.00, 0.88, 1.20, 1.43, 0.00).

**Analysis findings**:
- **A matrices changed significantly**: Learned A diverges from hand-designed defaults,
  especially for inventory and station modalities. Observation model was the main bottleneck.
- **B matrices only changed for MINE/EXPLORE**: Other options (CRAFT, CAPTURE, WAIT) were
  never exercised during 50-episode trajectory collection — B learning requires
  exposure to all option transitions.
- **C vectors unchanged**: Learning rate too low for preference learning at 2,000 steps.
  C learning requires either higher LR or inverse-EFE gradient approach.
- **Iterative pipeline needed**: Single-pass learning hits a ceiling. Need: learn → deploy →
  collect new trajectories → re-learn cycle (see Phase 3d Step 4).

---

### Phase 3d — Literature-Informed AIF Upgrades

**Why**: Deep pipeline results show 74% VFE reduction but live performance (0.70 j/agent)
still below Softmax (5.0 j/agent). Literature survey identifies 7 targeted upgrades
to close this gap, ordered by impact/effort ratio.

**Key literature sources**:
- Ruiz-Serra et al. (AAMAS 2025): "Factorised Active Inference for Strategic Multi-Agent Interactions"
- Heins et al. (2025): "AXIOM — Active eXpanding Inference with Object-centric Models"
- Champion et al. (Neural Computation 2024): "Multimodal and Multifactor Branching Time Active Inference"
- Fountas et al. (NeurIPS 2020): "Deep Active Inference Agents Using Monte-Carlo Methods"
- Hyland et al. (ICML 2024): "Free-Energy Equilibria in Multi-Agent Systems"

| Step | Feature | Source | Impact | Effort | Files | Status |
|------|---------|--------|--------|--------|-------|--------|
| 1 | Adaptive precision γ | Ruiz-Serra AAMAS 2025 | High | Small | `cogames_policy.py` | **COMPLETE** (2026-04-03) |
| 2 | Novelty term η + Exploration E | Ruiz-Serra AAMAS 2025 | Medium | Small | `generative_model.py`, `cogames_policy.py` | **COMPLETE** (2026-04-03) |
| 3 | Habit bypass | Fountas NeurIPS 2020 | Medium | Small | `cogames_policy.py` | **COMPLETE** (2026-04-09) |
| 4 | Online streaming A/B | AXIOM (Heins 2025) | High | Medium | `cogames_policy.py` | **COMPLETE** (2026-04-09) |
| 5 | BTAI tree search | Champion Neural Comp 2024 | High | Large | new `btai_planner.py` | TODO |
| 6 | Opponent belief factors | Ruiz-Serra AAMAS 2025 | High | Large | `generative_model.py` | TODO |
| 7 | BMR compression | AXIOM rMM-style | Medium | Medium | `differentiable_bmr.py` | TODO |

---

#### Step 1: Adaptive Precision γ (Exploration-Exploitation Balance)

**Source**: Ruiz-Serra Eq. 17 — `γ = β₁ / (β₀ - ⟨G⟩)`

**Current state**: Fixed `gamma = 8.0` in `_select_option()`. Agent uses same
exploration level regardless of EFE landscape.

**Upgrade**: Self-tuning precision that drops when EFE values are ambiguous
(explore more) and rises when one option clearly dominates (exploit).

```python
# In cogames_policy.py, _select_option():
mean_G = neg_efe.mean(axis=-1)          # average EFE across options
gamma_adaptive = beta_1 / (beta_0 - mean_G)  # Ruiz-Serra Eq. 17
gamma_adaptive = np.clip(gamma_adaptive, 1.0, 32.0)  # stability bounds
# Use gamma_adaptive as temperature in softmax policy selection
```

**Parameters**: β₁ ∈ [15, 30], β₀ = 1.0 (from Ruiz-Serra experiments).

**Verification**:
- Early episodes (uncertain): γ should be low (~2-4), agent tries all options
- Late episodes (learned model): γ should be high (~16-32), agent exploits best option
- Monitor: γ time series should correlate inversely with VFE

---

#### Step 2: Novelty Term η (Explore Under-Tried Options)

**Source**: Ruiz-Serra Eq. 20 — `η(û) = Σₙ D_KL[B̄_{û,n} ‖ B_{û,n}]`

**Current state**: EFE has risk + ambiguity but no explicit novelty drive. Options
with low B matrix confidence (few observations) are not preferentially explored.

**Upgrade**: Add expected parameter information gain — the KL divergence between
expected posterior B and current prior B after taking each action.

```python
# In generative_model.py, compute_efe():
for option_idx in range(n_options):
    # Dirichlet concentration for B[option] transition counts
    alpha_prior = B_concentrations[option_idx]  # current Dirichlet params
    alpha_expected = alpha_prior + expected_counts  # posterior after one more obs
    eta = dirichlet_kl(alpha_expected, alpha_prior)  # expected parameter gain
    novelty[option_idx] = eta
# G = risk + ambiguity + novelty_weight * novelty
```

**Key insight**: This drives agents to try CRAFT_CYCLE and CAPTURE_CYCLE (which had
zero B learning in deep pipeline because they were never exercised).

**Verification**: Options with low Dirichlet concentration should get novelty bonus →
more uniform early exploration → all B matrices get training data.

---

#### Step 3: Habit Bypass (Skip Planning When Confident)

**Source**: Fountas et al. NeurIPS 2020, §3.4 — bootstrap confidence threshold.

**Current state**: Full EFE evaluation (625 policies at T=4) runs every decision step,
even when the agent is mid-option with high confidence.

**Upgrade**: When the E vector (habit prior) strongly favors the current option AND
recent VFE is low, skip full EFE computation and continue executing.

```python
# In cogames_policy.py, _select_option():
if self._current_option is not None:
    habit_confidence = E[self._current_option_idx]
    if habit_confidence > 0.7 and self._recent_vfe < vfe_threshold:
        return self._current_option  # bypass planning
# Otherwise: full EFE evaluation
```

**Benefits**: (a) Faster decision cycle (skip expensive T=4 evaluation), (b) more
stable behavior (no mid-task option switching), (c) prerequisite for deeper planning
(BTAI) — freed compute budget goes to tree search when planning IS needed.

**Verification**: Agent should plan at decision points (empty hands, arrived at station)
but coast during navigation and option execution.

---

#### Step 4: Online Streaming A/B Learning

**Source**: AXIOM (Heins 2025) — per-step conjugate updates with sufficient statistics.

**Current state**: Batch learning: collect 50-episode trajectory → offline gradient
optimization → deploy. No learning during live play.

**Upgrade**: Accumulate Dirichlet sufficient statistics during live play. Every
observation updates A/B concentrations incrementally.

```python
# In cogames_policy.py, after each step:
def _online_update(self, obs, prev_state, action, curr_state):
    # A learning: observation-state co-occurrence
    for modality_idx, o in enumerate(obs):
        self.A_counts[modality_idx][o, curr_state] += learning_rate

    # B learning: state transition given action
    self.B_counts[action][curr_state, prev_state] += learning_rate

    # Periodically normalize: A = Dirichlet_expected(A_counts)
    if self.step_count % update_interval == 0:
        self._update_parameters_from_counts()
```

**Integration with existing pipeline**:
- Initialize A_counts/B_counts from deep pipeline learned parameters (`.npz`)
- Online updates accumulate on top of batch-learned values
- Periodic BMR sweep (every 500 steps, per AXIOM) prunes low-evidence connections

**Files to modify**:
- `cogames_policy.py`: Add `_online_update()` method, call after each step
- `learn_parameters.py`: Export/import Dirichlet concentrations (not just expected values)

**Verification**: VFE should decrease monotonically during live play as A/B converge.

---

#### Step 5: BTAI Tree Search (Extend Planning Horizon)

**Source**: Champion et al. (Neural Computation 2024) — BTAI_3MF (Multimodal Multifactor).

**Current state**: Exhaustive policy evaluation at T=2 (25 policies) or T=4 (625 policies).
Cannot scale beyond T=4 without 3,125+ policy explosion.

**Upgrade**: Replace exhaustive evaluation with MCTS-guided tree search on the factor
graph. Selectively explores promising branches to effective depth 8-15 while evaluating
only ~150-200 nodes total.

**Algorithm** (BTAI_3MF adapted for our factored POMDP):
```
for iteration in range(max_planning_steps):  # 150-200
    1. SELECT: Walk tree via UCT (average EFE + exploration bonus)
    2. EXPAND: At leaf, create 5 children (one per macro-option)
    3. EVALUATE: Compute factored EFE at each child:
       - Forward predict: q(s_{t+1}|f) = B_f · q(s_t|f) for each factor f
       - Predict obs: q(o_m) = A_m · ⊗_f q(s_f) for each modality m
       - Risk: D_KL[q(o_m) ‖ C_m] per modality
       - Ambiguity: E_q(s)[H[P(o|s)]] per modality
    4. PROPAGATE: Best child's EFE propagates up to root
Select: most-visited root child (robust to noise)
```

**Key advantage**: Factored inference over each state factor independently.
Instead of 288×288 B matrix, operate on 6×6, 4×4, 3×3, 4×4 separately.
Cost per node: O(Σ_f |S_f|²) instead of O(|S|²).

**UCT formula**: `UCT(node) = -cost/visits + c_explore · √(ln(parent.visits)/visits)`

**Implementation**: New `src/aif_meta_cogames/aif_agent/btai_planner.py`:
- `TemporalSlice`: Node holding factored posteriors + factor graph
- `MCTS`: Select/Expand/Evaluate/Propagate loop
- Drop-in replacement for `evaluate_policies_multistep()` in variational engine

**Estimated performance**: 150 iterations × 5 children = 750 nodes evaluated.
At 2.5s per plan (from BTAI_3MF dSprites benchmarks), effective depth 8-12.

**Reference implementations**:
- Python: [ChampiB/BTAI_3MF](https://github.com/ChampiB/BTAI_3MF)
- C++: [ChampiB/Homing-Pigeon](https://github.com/ChampiB/Homing-Pigeon)

---

#### Step 6: Opponent Belief Factors (Infer Teammate Options)

**Source**: Ruiz-Serra AAMAS 2025 — mean-field factorisation over agents.

**Current state**: Agents are blind to teammates' macro-options. SharedSpatialMemory
provides station sharing but no ToM about what teammates are doing or planning.

**Upgrade**: Add 3 state factors (one per teammate), each tracking the teammate's
estimated current macro-option:

```
Current factors: [phase(6), hand(4), target_mode(3), role(4)] → 288 states
New factors:     + [ally_0_option(5), ally_1_option(5), ally_2_option(5)]
Total: 288 × 125 = 36,000 states (but factored → linear cost)
```

**Mean-field factorisation** (Ruiz-Serra): `q(s) = q(s_self) · Π_j q(s_ally_j)`

**New observation modality**: `o_ally_option` — inferred from teammate's observed
behavior (position trajectory, station visits, inventory changes).

**A matrix for ally observations**: Maps observed teammate behavior → beliefs
about their current option. E.g., teammate near mine → P(MINE_CYCLE) high.

**Pragmatic value with opponent marginalization** (Ruiz-Serra Eq. 12-13):
```python
# EFE marginalizes over ally beliefs:
for option in self_options:
    G_self = risk(option)
    # Marginalize social cost over ally beliefs
    G_social = 0
    for ally_opts in product(range(5), repeat=3):  # 125 combinations
        p_allies = prod(q_ally[j][ally_opts[j]] for j in range(3))
        G_social += p_allies * social_cost(option, ally_opts)
    G[option] = (1-lambda) * G_self + lambda * G_social
```

**Cost**: 5³ = 125 opponent marginalization per self-option — tractable.

**Verification**:
- Agent should avoid CAPTURE when ally is already capturing same junction
- Agent should prefer MINE when allies are crafting (supply chain coordination)
- Free-Energy Equilibria: stable role allocation should emerge

---

#### Step 7: BMR Compression (Merge Equivalent States)

**Source**: AXIOM rMM-style BMR — periodic pairwise merge of near-duplicate components.

**Current state**: Fixed 288-state space. Some states may be observationally equivalent
(e.g., different target_mode values that produce identical observations).

**Upgrade**: Periodic Bayesian Model Reduction sweep that merges states with
indistinguishable A/B profiles.

**Algorithm** (adapted from AXIOM):
```
Every 500 steps:
  1. Sample up to 2000 state pairs
  2. For each pair (i, j):
     a. Compute A-profile similarity: D_KL[A(·|i) ‖ A(·|j)] across all modalities
     b. Compute B-profile similarity: D_KL[B(·|i,a) ‖ B(·|j,a)] across all actions
     c. Propose merge: combine sufficient statistics
     d. Compute VFE for merged vs unmerged model
     e. Accept merge if VFE_merged ≤ VFE_unmerged + ε
  3. Update A/B/C/D to reflect merged state space
```

**Files to modify**:
- `differentiable_bmr.py`: Add `periodic_bmr_sweep()` function
- `generative_model.py`: Support variable state space size

**Expected outcome**: 288 states may reduce to ~100-150 effective states.
Faster inference + better generalization (fewer parameters to learn).

**Connection to Phase B-V/B-VI**: This completes the de novo pipeline —
structure discovery (B-V) proposes states, gradient learning (B-VI) fits parameters,
BMR compression removes redundancies.

---

#### Phase 3d Implementation Results

##### Steps 1-2: Adaptive γ + Novelty η + Exploration E (2026-04-03)

**Implementation details**:
- `AIF_ADAPTIVE_GAMMA=1`: γ = β₁/(β₀ - ⟨G⟩), clipped to [1, 32], with β₁=15, β₀=1. Recomputes q_pi after `infer_policies`.
- `AIF_EXPLORE_E=1`: Weakly informative E vector (4:1 ratio vs default 4000:1). `_build_exploration_E()` in `generative_model.py`.
- `AIF_NOVELTY_WEIGHT=X`: Per-option novelty η = Σ_f 1/Σα_f (inverse Dirichlet concentration). `_compute_novelty()` in `cogames_policy.py`.
- Default gamma bumped from 8 → 16 (better scale for adaptive formula).
- C learning parameters: c_lr_scale 0.1 → 0.5, c_weight 0.5 → 1.0, gradient steps 2000 → 5000.

**Iterative deep pipeline** (2026-04-07): 5 rounds × ~500k steps = ~2.5M total on arena no_clips, R2 learned params, policy_len=4.
- Rounds 1-3: Exploration E enabled (learn diverse B matrices).
- Rounds 4-5: Deploy E (standard E vector, exploit learned params).
- Adaptive γ + novelty enabled throughout all rounds.

| Round | VFE | VFE Reduction | Eval Mean (j/agent) | Best Episode | Notes |
|-------|-----|---------------|---------------------|--------------|-------|
| Baseline | 229.5 | — | 0.70 | 1.43 | Deep pipeline before changes |
| R1 | 60.5 | 79.3% | 5.01 | — | Strong initial learning |
| R2 | 64.9 | 78.4% | **16.01** | **34.08** | Best round overall |
| R3 | 65.1 | 78.3% | 13.05 | — | Diminishing returns |
| R4 | 65.1 | 78.3% | 11.77 | — | Explore→deploy transition |
| R5 | 65.1 | 78.3% | 3.96 | — | Performance dropped |
| Final (R5, 10-ep) | — | — | **10.43** | 29+ | 14.9× baseline |

**Key findings**:
- R2 model is the best performer. Explore → deploy transition at round 4 may be premature.
- High variance persists: some episodes score 0 (stuck cycles), others 29+.
- All B matrices now receive training data (CRAFT_CYCLE and CAPTURE_CYCLE exercised via exploration E).

##### Steps 3-4: Habit Bypass + Online Streaming A/B (2026-04-09)

**Implementation details**:
- `AIF_HABIT_BYPASS=1`: Skip planning when E[option] > 0.5 AND VFE_ema < 5.0. VFE EMA (α=0.1) tracks -ln max_belief per agent. Bypass counter in logs.
- `AIF_LEARN_INTERVAL=N`: Tunable Dirichlet update interval (default 50). Online `infer_parameters()` call frequency.
- VFE-gated learning rate: lr_scale = min(1, 0.1 + 0.9 × VFE/threshold). High VFE → fast learning; low VFE → slow (preserve good params).

**Evaluation** (2026-04-10): Arena no_clips, 5 episodes each, R2 learned params, policy_len=4.

| Config | Mean j/agent | Episodes | Timeouts | Notes |
|--------|-------------|----------|----------|-------|
| Baseline (no features) | 0.15 | (0,0,0,0,0.76) | 1986 | Non-JIT'd, most actions timed out |
| Habit bypass + VFE-gated + learn=20 | **1.43** | (0.94,1.69,1.74,1.85,0.92) | 3149 | VFE-gated lr is the actual win |

**Key findings**:
- Habit bypass never triggered (bypasses=0): default E splits ~50/50 between mine/craft, so no option exceeds the 0.5 threshold. Need to lower threshold to ~0.3 or always enable explore_E for bypass to trigger.
- VFE-gated learning rate is the primary contributor: lr_scale ~0.45-0.57, with learn_interval=20 providing 2.5× more frequent parameter updates.
- Neither eval had adaptive_gamma/explore_E/novelty enabled, explaining underperformance vs the iterative pipeline (10.43 j/agent).

##### JIT Performance Optimization (2026-04-12)

**Problem**: mettagrid enforces a 250ms action timeout. Non-JIT'd inference exceeds this:

| Component | Non-JIT (ms) | JIT (ms) | Speedup |
|-----------|-------------|---------|---------|
| `infer_states` (belief update) | 141 | 0.2 | 705× |
| `update_empirical_prior` | 71 | 21.4 | 3.3× (non-JIT, return-type incompatibility) |
| Nav `infer_states` + `infer_policies` + `sample_action` | 424 | 0.7 | 606× |
| `_select_option` (at termination only) | 917 | 0.5 | 1834× |
| **Total per step** | **~636 + overhead → 1200** | **~22** | **28×** |

**Strategy**: JIT-compile `_belief_update`, `_select_option`, and `_nav_infer` via `eqx.filter_jit()`. Keep `update_empirical_prior` non-JIT due to return-type mismatch between local and pip pymdp builds (same version 1.0.0, different return signatures — local returns `(pred, qs)` tuple, pip returns `pred` list directly). Handled via `isinstance` check at call sites.

**Warmup**: JIT compilation on first call takes several seconds. All three JIT'd functions are warmed up with dummy inputs during `__init__` to avoid timeouts on the first evaluation step. Remaining ~102 warmup timeouts per evaluation are from initial compilation overhead.

##### Systematic Ablation Study (2026-04-12)

6 configs × 5 episodes each, arena no_clips, R2 learned params, policy_len=4, fully JIT'd.

| Config | Mean j/agent | Per-Episode Scores | Timeouts | Notes |
|--------|-------------|-------------------|----------|-------|
| Baseline (no features) | 0.00 | (0, 0, 0, 0, 0) | 102 | JIT warmup only |
| Adaptive gamma only | 0.35 | (0, 0, 0, 0.88, 0.88) | 102 | Marginal improvement |
| Explore E only | 0.50 | (0, 0, 0, 1.27, 1.25) | 102 | Diversifies B learning |
| **Novelty only** | **1.83** | **(0, 4.11, 1.88, 1.51, 1.63)** | **102** | **Best single feature** |
| VFE-gated + learn=20 | 1.04 | (0, 0.94, 1.69, 1.74, 1.85) | 252 | Extra timeouts from non-JIT infer_parameters |
| All combined | 0.00 | (0, 0, 0, 0, 0) | 252 | Feature interference destroys performance |

**Key findings**:
- **Novelty (η) is the strongest single feature** at 1.83 j/agent, driving exploration of under-tried options (CRAFT_CYCLE, CAPTURE_CYCLE).
- **Feature interaction is destructive**: All features combined scores 0.00 — worse than any individual feature. The features are not additive; they interfere (e.g., adaptive γ amplifies noise from novelty + exploration E simultaneously).
- **VFE-gated learning** contributes 1.04 j/agent but adds ~150 extra timeouts per evaluation from non-JIT'd `infer_parameters` calls every 20 steps.
- **Adaptive gamma** alone is marginal (0.35) — needs novelty/exploration to provide meaningful EFE variance for the adaptive formula to exploit.
- **The iterative pipeline's 10.43 j/agent used a different protocol**: 5 rounds of collect-learn-deploy with exploration E in rounds 1-3, rather than single-shot evaluation. The iterative refinement of B/C matrices is the main driver of performance, not the runtime features alone.

##### Tournament Submission (2026-04-12)

**Bundle**: `aif-r2-jit-v4:v1` — submitted to beta-cvc qualifying pool.

**Docker environment**: `ghcr.io/metta-ai/episode-runner:compat-v0.24`, CPU-only, Python 3.12. `setup_bundle.py` installs `jaxlib==0.9.2`, `jax==0.9.2`, `equinox==0.13.6`, `inferactively-pymdp==1.0.0`, `mctx`.

**pymdp compatibility note**: Local pymdp 1.0.0 (from source) and pip pymdp 1.0.0 have different `update_empirical_prior` return types. Local returns `(pred, qs)` tuple; pip returns `pred` (list[Array]) directly. Both call sites use `isinstance` check to handle both builds transparently.

**Tournament results** (preliminary, 2 matches completed): 0.00 j/agent. Agents mine and craft but never align junctions. Root cause: R2 params were trained on arena (4 agents, 50×50) but tournament runs on machina_1 (8+ agents, 88×88). Navigation to junction stations takes much longer on the larger map, and the 4-agent B matrices don't transfer to 8-agent dynamics.

**Next steps**: Phase C (plan broadcasting for 8-agent coordination) and map-specific parameter learning are needed before tournament viability.

---

### PI Meeting Notes (2026-04-01) — Implications for AIF

**Context**: Call with Subhojeet. Presented AIF option-selection results and deep AIF architecture.

**Key takeaways for AIF direction**:

1. **Map size is a major bottleneck**: "Because the map is so large, it takes them a lot of time to go to different parts of the map." **Action**: Try shrinking the map to validate AIF agents align junctions normally, then scale up. If smaller map works well → confirms bottleneck is navigation, not planning.

2. **100M steps is too short**: Softmax trains for **billions** of steps internally. Kickstarting schedule: KL=1.0 for 4B steps, anneal 4B-8B, pure PPO after 10B. Our 50M training budget (for trajectory collection) may produce suboptimal teacher trajectories. **Implication**: Parameter learning from longer-trained agent trajectories may yield better A/B/C.

3. **Option discovery vs option selection**: PI explicitly distinguished these — "there is an option discovery problem, and then there is the option selection problem. Right now, you're fixing on the set of options and focusing on the option selection problem." Our 5 macro-options (MINE_CYCLE, CRAFT_CYCLE, CAPTURE_CYCLE, EXPLORE, WAIT) are hand-designed. **Future**: Could spectral clustering (B-V Phase 1) discover better options from data?

4. **Options are scripted, not learned**: "The individual options are scripted policies, they are not really learned." PI confirmed this is a limitation — the low-level execution within each option is rule-based. **Implication**: Learning option-internal policies (e.g., via RL sub-policies or learned nav POMDP) could improve upon scripted execution.

5. **Test on CLIPs variant**: "Apply the same framework in the CLIPs variant as well. We have successful policies on the leaderboard. Find the candidate options and learn the active inference approach over that." **Action**: After validating on no_clips, test AIF option selection on the adversarial CLIPs variant where coordination under enemy pressure matters more.

6. **Positive on AIF for exploration**: "This approach definitely has a lot more chance because it directly attacks the exploration problem... if we have a learned policy over [options] then we are actually seeing new results." Validates our approach direction.

7. **Submission issue**: Missing dependency in uploaded policies — need to check `cogames ship` bundle includes all AIF code.

8. **Game may be changing**: "I think the game is broken, to be honest... people are working on fixing the game." Results may shift as game mechanics evolve.

**Actionable items**:
- [ ] Shrink map experiment: eval AIF on smaller arena variant to isolate navigation vs planning bottleneck
- [ ] Collect trajectories from longer-trained agents (or use Softmax's trained checkpoints) for better parameter learning
- [ ] Test AIF on CLIPs variant (adversarial)
- [ ] Fix `cogames ship` bundle to include all dependencies
- [ ] Consider option discovery from data (spectral clustering on action sequences)

#### Available Maps for Validation

| Map | Size | Agents | Full mechanics? | Notes |
|-----|------|--------|-----------------|-------|
| `easy_hearts_training` | 13×13 | 1–4 | Partial | Hearts + energy only, no clips |
| `tutorial` / `tutorial.aligner` | 35×35 | 1–4 | Yes (minus clips) | Best for fast AIF validation |
| `arena` | 50×50 | 1–20 | Yes | Current main map |
| `machina_1` | 88×88 | 10 | Yes | Tournament standard |

Custom sizes: `cogames make-mission --width W --height H`. Variant `-v small_50` shrinks machina_1 to 50×50.

---

### Phase C: Plan Broadcasting for 8-Agent Coordination

**Why**: 8-agent performance (0.62 j/agent) is much worse than 4-agent (1.75 j/agent).
Root cause: multiple aligners target the same junctions, no target deconfliction.

**Mechanism**: Extend `SharedSpatialMemory` to include agent intent:
- Each agent broadcasts: `(agent_id, current_option, target_position, target_type)`
- Before selecting a junction target, check if another aligner already claims it
- Social modality: count of allied agents targeting same junction → `o_contest`
- Cost: just a dict lookup, no inference overhead

**Implementation**:
- Add `intended_targets: dict[int, tuple[int, int, str]]` to SharedSpatialMemory
- In `_find_nearest_target()`, penalize or skip already-claimed targets
- Nav POMDP social obs: `CLAIMED_BY_ALLY` → lower EFE for alternative targets

**Expected impact**: 8-agent should match or exceed 4-agent per-agent rate since
more miners supply more resources.

---

### Phase D: Bayesian Model Comparison

**Why**: After learning parameters (Phase B), we need a principled way to compare
model variants. Bayesian model comparison uses VFE as an approximation to log
model evidence: lower F = better model.

**Mechanism**:
- Run N episodes with parameter set θ₁ → compute mean F₁
- Run N episodes with parameter set θ₂ → compute mean F₂
- Bayes factor: BF = exp(F₁ - F₂) — if BF > 3, strong evidence for θ₂
- Also track junction captures as external validation

**Bayesian Model Reduction (BMR)**:
- Analytically compare nested models without refitting
- If removing a parameter doesn't increase F, the simpler model is preferred
- pymdp has `dirichlet_kl_divergence` for computing KL between Dirichlet posteriors

**AXIOM BMR Algorithm** (Heins 2025 — adapted from rMM periodic pruning):
- Every 500 steps, sample up to 2000 component pairs
- For each pair: score mutual expected log-likelihoods via ancestral sampling
- Propose merge: combine sufficient statistics of both components
- Accept merge if EFE of merged model ≤ EFE of unmerged model (greedy)
- Also prune components unused for >10 steps
- Key insight: merges near-duplicate interaction clusters (e.g., "mine copper" and
  "mine iron" merge into "mine resource"), enabling generalization
- See Phase 3d Step 7 for concrete implementation plan

---

### Phase E: Factorised Social AIF

**Why**: Plan broadcasting (Phase C) is a simple deconfliction mechanism. Factorised
AIF goes further: each agent maintains beliefs about other agents' states and
preferences, enabling principled multi-agent coordination.

**Key paper**: Ruiz-Serra, Sweeney & Harre (AAMAS 2025) — "Factorised Active Inference
for Strategic Multi-Agent Interactions". Code: github.com/RuizSerra/factorised-MA-AIF

**Mechanism** (from Ruiz-Serra detailed analysis):
- **Mean-field factorisation**: `q(s) = q(s_self) · Π_j q(s_ally_j)` — each agent
  maintains independent marginals over ally hidden states
- **C vector from payoffs**: `p*(o_self, o_ally) = softmax(joint_payoff)` (Eq. 9)
- **Pragmatic value with opponent marginalization** (Eq. 12-13): agent evaluates each
  option by marginalizing over beliefs about what allies are doing
- **Adaptive precision γ** (Eq. 17): `γ = β₁/(β₀ - ⟨G⟩)` — self-tuning explore/exploit
- **Novelty term η** (Eq. 20): `η(û) = Σₙ D_KL[B̄_{û,n} ‖ B_{û,n}]` — drives
  exploration of under-tried options
- **Hebbian B learning** (Eq. 18): outer product of consecutive belief states,
  accumulated over learning window T_L ∈ [18, 30]
- **Free-Energy Equilibria** (Hyland et al. ICML 2024): AIF analogue of Nash Equilibrium —
  no agent can unilaterally reduce its EFE. Ensemble EFE `𝔊 = Σᵢ ⟨G⟩ⁱ` characterizes
  basins of attraction but is NOT necessarily minimized at equilibrium

**Our adaptation** (Phase 3d Steps 1, 2, 6):
- Add 3 ally option factors: [ally_0(5), ally_1(5), ally_2(5)] → 288 × 125 = 36,000
  factored states (linear cost via mean-field)
- New obs modality: `o_ally_option` inferred from teammate trajectories
- Opponent marginalization: 5³ = 125 combinations per self-option — tractable

**Challenge**: State space explosion. With 8 agents × 288 states each, full factorisation
is intractable. Solutions: (a) group agents by role, (b) only model nearest 2 allies,
(c) mean-field approximation (Ruiz-Serra validates this for N=2,3 agents).

---

### Phase F: Full Theory of Mind (Stretch Goal)

**Why**: Factorised AIF assumes agents share preferences. Full ToM models OTHER agents'
generative models — each agent has a model of what other agents believe and want.

**Key papers**:
- Vasil et al. (2020): Generalised Free Energy for multi-agent systems
- "Theory of Mind via Sophisticated Inference" (2025, pymdp implementation)
- "Emergent Joint Agency" (2025): synergistic information via nested Markov blankets

**Mechanism**:
- Level 1: q(s_self) — own beliefs (existing)
- Level 2: q(s_other; model_other) — beliefs about other's beliefs
- Level 3: q(model_other) — beliefs about what model the other agent uses
- Sophisticated inference: recursive depth-limited ToM

**This connects to social-layer project**: CommitmentInference, IntentParticleFilter,
SocialEFE, GatedToM patterns from social-layer provide the architecture template.

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

### Research Direction: Context-Dependent (Switching) Preferences

**Motivation**: The discrete POMDP uses static C vectors, but the aligner economy chain requires *phase-dependent* preferences — an agent with EMPTY hands should prefer observing RESOURCE (to get hearts), while an agent with BOTH should prefer JUNCTION (to capture). With a fixed C vector, one preference dominates and suppresses the others, leading to either capture-avoidance (c_inv[GEAR]>c_inv[BOTH]) or complacency (c_inv[BOTH] peak → stay in comfortable state).

**Principled formulation**: Hierarchical active inference with **context-conditioned preferences** (Friston et al., 2017; Pezzulo et al., 2018):

```
C_m(o | context) where context ∈ {CRAFT_CONTEXT, CAPTURE_CONTEXT, EXPLORE_CONTEXT}
```

The higher-level POMDP selects a goal context (macro-option), which modulates the C vector for the lower level. This is a *deep temporal model* where each level of the hierarchy sets preferences for the level below.

**Full principled formulation**: ALL planning parameters are context-dependent:

```
q(π | context) = σ(-G(π; C_context) + ln E(π | context))
```

| Parameter | CRAFT_CONTEXT | CAPTURE_CONTEXT | EXPLORE_CONTEXT |
|-----------|--------------|-----------------|-----------------|
| C_inv | prefer GEAR/BOTH | prefer JUNCTION obs | flat (epistemic) |
| C_sta | prefer HUB/CRAFT | prefer JUNCTION | flat |
| E(π) | bias toward CRAFT_CYCLE | bias toward CAPTURE_CYCLE | bias toward EXPLORE |
| π set | {CRAFT, NAV_DEPOT, NAV_GEAR} | {CAPTURE, NAV_JUNCTION} | {EXPLORE, all nav} |

The higher-level POMDP selects context, the lower level plans within it. Context transitions are triggered by belief state changes (inventory transition → context switch). This IS how deep temporal models work: each level sets priors (C, E, π) for the level below.

**Current implementation**: Pragmatic approximation via option auto-chaining (`auto_chain=True`) — the OptionExecutor acts as the hierarchical context switch. The macro-option implicitly determines which observations matter (CRAFT_CYCLE → prefer HUB/GEAR stations; CAPTURE_CYCLE → prefer JUNCTION stations). Ablation: `auto_chain=False` tests whether corrected C vector alone is sufficient.

**Neural AIF path**: In Phase 4, context-dependent C/E/π can be implemented as:
- `C = f_C(z_context)`, `E = f_E(z_context)` where z_context is the higher-level latent state
- The world model learns context→{preference, habit, policy set} mappings jointly with transition dynamics
- MAML initialization would capture which hierarchical structures transfer across variants
- Innovation for paper: meta-learned context-dependent priors that generalize across environment variants

**Key references**:
- Friston et al. (2017): Active Inference, Curiosity and Insight — deep temporal models
- Pezzulo et al. (2018): Hierarchical Active Inference — multi-level goal selection
- Hesp et al. (2021): Deeply Felt Affect — emotion as precision on hierarchical priors
- Da Costa et al. (2020): Active Inference on Discrete State-Spaces — factored C vectors
- Sajid et al. (2021): Active Inference: Demystified and Compared — E vector as habit prior

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
- Ruiz-Serra, Sweeney & Harre (AAMAS 2025): Factorised Active Inference for Strategic Multi-Agent Interactions — mean-field factorisation, adaptive γ, novelty η, Free-Energy Equilibria
- Heins et al. (2025): AXIOM — Active eXpanding Inference with Object-centric Models — conjugate mixture models, online BMR, MPPI planning, 10K-step mastery
- Champion et al. (Neural Computation 2024): Multimodal and Multifactor Branching Time Active Inference — MCTS on factor graphs, factored belief propagation
- Fountas et al. (NeurIPS 2020): Deep Active Inference Agents Using Monte-Carlo Methods — MCTS for AIF, habit bypass, bootstrap confidence
- Hyland et al. (ICML 2024): Free-Energy Equilibria — AIF analogue of Nash Equilibrium, bounded rationality
- Friston et al. (2025): Gradient-Free De Novo Learning — structure discovery, Bayesian model reduction, renormalising generative models
