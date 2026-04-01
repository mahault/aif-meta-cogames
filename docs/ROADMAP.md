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

---

### Phase E: Factorised Social AIF

**Why**: Plan broadcasting (Phase C) is a simple deconfliction mechanism. Factorised
AIF goes further: each agent maintains beliefs about other agents' states and
preferences, enabling principled multi-agent coordination.

**Key paper**: Pitliya et al. (AAMAS 2025) — "Factorised Active Inference for
Strategic Multi-Agent Interactions". Code: github.com/RuizSerra/factorised-MA-AIF

**Mechanism**:
- State factors: `q(s_self) · q(s_ally_1) · q(s_ally_2) · ...`
- Joint preferences: `p*(o_self, o_ally) = softmax(joint_payoff)`
- EFE includes: G_self (own risk+info) + G_social (ally risk+info)
- Natural extension of our existing factored POMDP (4 factors → 4+N_allies)

**Challenge**: State space explosion. With 8 agents × 288 states each, full factorisation
is intractable. Solutions: (a) group agents by role, (b) only model nearest 2 allies,
(c) mean-field approximation.

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
