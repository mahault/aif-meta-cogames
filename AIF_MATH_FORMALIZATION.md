# Active Inference for CogsGuard: Mathematical Formalization

**Mahault Albarracin** | March 2026
**For**: Subhojeet (Softmax PI)

> **Related docs**:
> - [AIF_COGSGUARD_DESIGN.md](AIF_COGSGUARD_DESIGN.md) — Full 216-state architecture & references
> - [ROADMAP.md](ROADMAP.md) — Active execution plan
> - [LITERATURE.md](LITERATURE.md) — Papers & methods

---

## 1. Problem Setting: POMDP

CogsGuard is a **partially observable Markov decision process** (POMDP):

```
M = (S, A, O, B, A_obs, C, D)
```

| Symbol | Definition | CogsGuard instantiation |
|--------|-----------|------------------------|
| **S** | Hidden state space | 18 states: phase(6) x hand(3) |
| **A** | Action space | 5 actions: {noop, N, S, W, E} |
| **O** | Observation space | 3 modalities: o_resource(3), o_station(4), o_inventory(3) |
| **B** | Transition model P(s'|s,a) | Economy chain dynamics (mine→deposit→craft→gear→capture) |
| **A_obs** | Observation model P(o|s) | How hidden state generates observations |
| **C** | Preference distribution P(o|C) | Preferred observations (junctions >> resources >> nothing) |
| **D** | Prior over initial states P(s_0) | 90% EXPLORE/EMPTY |

The agent never observes its hidden state directly. It must *infer* its state from observations, *plan* actions to reach preferred outcomes, and *explore* to reduce uncertainty.

### Inputs and Outputs

```
Input:  o_t = (o_resource, o_station, o_inventory)   -- discrete observation tuple
        Raw: 200 tokens x 3 bytes, discretized via ObservationDiscretizer

Output: a_t in {noop, north, south, west, east}      -- selected action
        Selected by minimizing expected free energy G(pi)
```

---

## 2. Variational Free Energy (VFE) — The World Model Loss

At each timestep, the agent holds a belief q(s_t) over hidden states and updates it given new observations. The **variational free energy** F quantifies how well the agent's beliefs match reality:

```
F[q, o_t] = E_q(s)[ ln q(s) - ln P(o_t, s) ]
           = D_KL[ q(s) || P(s|o_t) ] - ln P(o_t)
           = D_KL[ q(s) || P(s) ] - E_q(s)[ ln P(o_t | s) ]
                complexity              accuracy
```

**Three equivalent decompositions:**

1. **Bound on surprise**: F >= -ln P(o_t). Minimizing F minimizes an upper bound on observation surprise.
2. **Complexity minus accuracy**: The agent balances fitting observations (accuracy) against staying close to its prior (complexity).
3. **KL to posterior**: F = D_KL[q || P(s|o)] + const. Minimizing F makes q(s) approximate the true posterior P(s|o_t).

### Belief Update (Inference)

Given observation o_t, the agent updates beliefs via:

```
q(s_t) ∝ P(o_t | s_t) · q̃(s_t)
```

where q̃(s_t) = Σ_s_{t-1} B(s_t | s_{t-1}, a_{t-1}) · q(s_{t-1}) is the predicted belief.

This is exact Bayesian filtering in our 18-state discrete model. In the neural variant, this becomes an approximate inference step.

### Connection to RL Prediction Loss

In RL world models (e.g., Dreamer), the world model loss is:

```
L_world = -E_q(s)[ ln p(o_t | s_t) ] + β · D_KL[ q(s_t | o_t) || p(s_t | s_{t-1}, a_{t-1}) ]
```

This is **exactly VFE** with β=1. The "prediction loss" Subhojeet mentioned is the accuracy term. The KL term is the complexity cost. Active inference makes this explicit and uses it for both inference and action selection.

---

## 3. Expected Free Energy (EFE) — The Decision Objective

The agent selects actions (or policies π = sequence of actions) by minimizing the **expected free energy** G(π):

```
G(π) = E_q(o,s|π)[ ln q(s|π) - ln P(o, s | C) ]
```

where P(o, s | C) is the agent's *preferred* joint distribution over observations and states, encoded by the C vector.

### Decomposition into Risk + Ambiguity

G(π) decomposes into two terms with distinct functional roles:

```
G(π) = E_q(s|π)[ E_P(o|s)[ ln P(o|s) - ln P(o|C) ] ]   ... (risk)
      + E_q(s|π)[ H[ P(o|s) ] ]                           ... (ambiguity)
```

**Risk** (pragmatic value):
```
G_risk(π) = E_q(s|π)[ D_KL[ P(o|s) || P(o|C) ] ]
          ≈ E_q(s|π)[ -ln P(ō|C) ]    where ō = argmax P(o|s)
          = Expected surprisal of observations under preferences
```

This is the cross-entropy between what the agent expects to observe under policy π and what it *prefers* to observe. A miner prefers to be AT resources; an aligner prefers to see JUNCTIONS.

**Ambiguity** (epistemic value):
```
G_ambiguity(π) = E_q(s|π)[ H[ P(o|s) ] ]
               = Expected entropy of the observation model
```

This measures how informative future observations will be. States where P(o|s) is broad (high entropy) are *ambiguous* — the agent doesn't know what it will see. **Policies leading to high-ambiguity states are penalized**, driving the agent toward states where observations are informative.

### Why This IS Exploration (Not Just Prediction)

The epistemic term creates a **constitutive drive** toward informative states:

```
G_ambiguity(π_explore) < G_ambiguity(π_stay)

because: exploring a new area leads to states where P(o|s) has
         lower entropy (more informative observations) than
         staying in a known area where observations are redundant
```

Critically, this drive exists **regardless of reward**. The agent seeks informative states even when no reward signal exists there. This is precisely what breaks the gear wall: gear stations produce distinctive observations (gear in inventory), creating low-ambiguity states that the agent is drawn toward before ever receiving gear-related reward.

### The Full Objective

For policy selection:

```
π* = argmin_π G(π) = argmin_π [ G_risk(π) + G_ambiguity(π) ]
```

Action sampling with precision β:

```
P(π) = σ(-β · G(π))    where σ is softmax
```

Higher β → sharper policy (more committed). Lower β → more exploratory.

---

## 4. Epistemic Value vs Successor Features

Subhojeet's "general agent substrate" uses **successor features** for intrinsic motivation:

```
ψ^π(s) = E_π[ Σ_{t=0}^∞ γ^t φ(s_t) | s_0 = s ]
```

where φ(s) are learned features. The geometry of ψ reveals which actions lead to different parts of the state space.

### Comparison

| Property | Successor Features | AIF Epistemic Value |
|----------|-------------------|-------------------|
| **What it measures** | Discounted feature occupancy | Expected information gain about hidden states |
| **Requires** | Learned features φ(s) | Generative model P(o\|s) |
| **Operates on** | State space (requires state access) | Belief space (works with observations only) |
| **POMDP-native** | No — requires state features | Yes — defined over beliefs |
| **Adaptation** | Fixed features after training | Adapts with belief updates |
| **Mathematical grounding** | Bellman equations (reward transfer) | Variational inference (free energy principle) |
| **Decay** | Inherent via discount γ | None — recomputed from current beliefs |

**Key difference for CogsGuard**: Successor features require learned representations of the full state, which is hard in a 13x13 partial-observation window on an 88x88 map. AIF epistemic value works directly on the agent's *belief state* — it measures uncertainty about hidden states, not about features of observed states.

**Where successor features win**: When you have a good state representation and need transfer across reward functions. GPI (generalized policy improvement) allows zero-shot adaptation to new reward functions, which AIF doesn't directly support.

**Where AIF wins**: When the environment is partially observable, exploration is needed without any reward signal, and the agent must reason about what it *doesn't know* rather than what it *has seen*.

### Mathematical Connection

Both can be seen as special cases of optimal information acquisition:

```
Successor features:  max_a  r(s,a) + α · ||∇_ψ ψ^π(s)||    (novelty = feature gradient)
AIF epistemic:       min_π  G_risk(π) + G_ambiguity(π)        (novelty = observation entropy)
```

The AIF formulation is Bayes-optimal for POMDPs (Wei, 2024, arXiv:2408.06542). Successor features are Bellman-optimal for fully observable MDPs.

---

## 5. Multi-Agent Extension: G-Coupling

For N agents, each agent i minimizes its own EFE with a **social coupling** term:

```
G_i(π_i) = (1 - λ) · G_self_i(π_i) + λ · G_other_i(π_i)
```

**G_self** — self-directed EFE:
```
G_self_i(π_i) = G_risk_self(π_i) + G_ambiguity_self(π_i)
              = E[-ln P(o_i | C_i)]  +  E_q(s_i)[H[P(o_i|s_i)]]
```

Uses agent i's own preference vector C_i (role-specific: miner prefers resources, aligner prefers junctions).

**G_other** — socially-coupled EFE:
```
G_other_i(π_i) = G_risk_other(π_i) + G_ambiguity_other(π_i)
               = E[-ln P(o_team | C_team)]  +  E_q(s_j)[H[P(o_j|s_j)]]
```

Uses team preference vector C_team and beliefs about teammate states q(s_j).

### How G-Coupling Acts as Communication

When agent i observes teammate j mining:

```
q_i(s_j = MINE) increases
→ P(o_team | C_team, s_j=MINE) shifts  (team already has a miner)
→ G_risk_other(π_i = MINE) increases   (diminishing returns)
→ G_risk_other(π_i = ALIGN) decreases  (complementary action)
→ Agent i shifts toward aligning
```

No message was sent. The coordination emerged from EFE minimization over beliefs about teammates. This is **implicit communication** — the information channel is the shared environment + the generative model.

### Why This Is NOT Just Shared Reward

| Mechanism | What drives coordination |
|-----------|------------------------|
| Shared reward (PPO) | All agents get same scalar R — no differentiation |
| MAPPO centralized critic | Critic sees all states but policy can't use it |
| IC3Net | Learned message → attention → policy modulation |
| **G-coupling** | Beliefs about teammates → EFE shift → policy modulation |

G-coupling provides interpretable, decomposable coordination. You can inspect *why* agent i chose to align: because G_risk_other(MINE) was 2.3 while G_risk_other(ALIGN) was 0.8. This decomposition is not available in learned communication.

---

## 6. Theory of Mind (ToM)

Each agent i maintains a generative model of each teammate j:

```
Teammate model: P(o_j | s_j, role_j), P(s_j' | s_j, role_j)
Belief:         q_i(s_j, role_j)  -- agent i's belief about j's state and role
```

### Update When Teammate Observed

```
q_i(s_j, t) ∝ P(o_j^obs | s_j) · q_i(s_j, t-1)     -- Bayesian update
```

where o_j^obs is j's observed behavior (movement direction, proximity to stations, vibe).

### Forward Prediction When Teammate NOT Observed

```
q_i(s_j, t) = Σ_{s'_j} B_j(s_j | s'_j, role_j) · q_i(s'_j, t-Δ)    -- propagate forward
```

where Δ is the number of ticks since last observation. The agent predicts where j is likely to be based on its last known state and role-specific transition dynamics.

### Role Inference

```
q_i(role_j) ∝ P(trajectory_j | role_j) · P(role_j)
            = Π_t P(o_j,t | s_j, role_j) · P(role_j)
```

Observing j repeatedly heading toward extractors increases q_i(role_j = MINER).

---

## 7. Five AIF Approaches for CogsGuard

We propose five approaches of increasing complexity. Each builds on the previous.

### Approach A: Discrete POMDP Agent (F1 — Softmax Deliverable)

**Objective**: Hand-crafted generative model, no learning.

```
Input:   o_t = discretize(raw_obs)           -- 3 discrete modalities
Model:   A (hand-crafted), B (hand-crafted), C (role-specific), D (uniform)
Infer:   q(s_t) ∝ A[o_t, :] ⊙ B @ q(s_{t-1})
Plan:    G(π) = Σ_τ [ G_risk(τ) + G_ambiguity(τ) ]   for T=2 lookahead
Action:  a_t = argmin_π G(π)
```

**State space**: 18 = phase(6) x hand(3)
**Implementation**: pymdp library (Heins et al., JOSS 2022)
**Strengths**: Fully interpretable, no training, immediate deployment
**Weaknesses**: Hand-crafted A/B may not match true dynamics

### Approach B: Fitted POMDP (Data-Driven A/B)

**Objective**: Learn A and B matrices from trajectory data via maximum likelihood.

```
Fitting:
  A_m[o, s] = (count(o, s) + α) / Σ_o' (count(o', s) + α)     -- MLE with Dirichlet prior
  B[s', s, a] = (count(s', s, a) + α) / Σ_{s''} (count(s'', s, a) + α)

Pool across 3,600 episodes (36 variants x 100 episodes)
Per-variant A/B matrices capture environment-specific dynamics
```

**Implementation**: `aif_meta_cogames.aif_agent.fit_matrices` (complete)
**Strengths**: Data-driven, captures true dynamics per variant
**Weaknesses**: Requires trajectory data, no cross-variant generalization

### Approach C: MAML over A/B Matrices (H1 — Meta-Learning)

**Objective**: Learn an A/B initialization that adapts quickly to any environment variant.

This IS meta-learning, not hierarchical RL. The distinction:

| | Hierarchical RL | MAML Meta-Learning (ours) |
|---|---|---|
| **What is learned** | A high-level policy over subroutines | An initialization θ₀ for the world model |
| **Inner loop** | Subroutine execution | Gradient descent on per-variant A/B |
| **Outer loop** | Policy gradient on manager | Gradient through inner-loop adaptation |
| **At test time** | Fixed manager selects subroutines | Few-shot adaptation to new variant |

**Mathematical formulation**:

```
Outer loop:
  θ₀ = {A₀, B₀}                     -- meta-learned initialization
  L_meta(θ₀) = Σ_τ L_inner(θ₀, D_τ)  -- sum over task distribution

Inner loop (per variant τ):
  θ_τ = θ₀ - α · ∇_θ L_VFE(θ, D_τ^train)    -- few gradient steps

  L_VFE(θ, D) = -Σ_t ln P_θ(o_t | s_t)       -- negative log-likelihood
               = Σ_t [ -Σ_m ln A_m[o_t^m, s_t] ]

Outer loop update:
  θ₀ ← θ₀ - β · ∇_θ₀ Σ_τ L_VFE(θ_τ, D_τ^test)    -- MAML second-order gradient
```

**Why meta-learning matters here**: The 36 CogsGuard variants share common task structure (economy chain) but differ in spatial layout, resource density, and agent count. A meta-learned θ₀ captures the shared structure while enabling fast adaptation to variant-specific dynamics.

**Task family**: P(τ) = {arena_n4_easy, arena_n8_hard, machina_1_n8_standard, ...}

**Implementation**: Luca (collaborator) building MAML outer loop on top of fitted A/B matrices.

### Approach D: Deep AIF / R-AIF (Neural World Model)

**Objective**: Replace discrete A/B matrices with neural networks for scalability.

```
Encoder:    z_t = f_enc(o_t; φ)              -- raw obs → latent state
World model: z̃_{t+1} = f_wm(z_t, a_t; ψ)    -- predict next latent
Decoder:    ô_t = f_dec(z_t; ξ)              -- reconstruct observations

Loss:       L = -E_q[ln p(o_t | z_t)]        -- reconstruction (accuracy)
              + D_KL[q(z_t | o_t) || p(z_t | z_{t-1}, a_{t-1})]  -- complexity
            = VFE in latent space

Planning:   G(π) computed in latent space via rollouts through f_wm
```

**Key insight**: This is the same VFE/EFE framework, but with learned continuous representations instead of discrete states. The "prediction loss" and "world model" are exactly VFE's accuracy and complexity terms.

**Reference**: R-AIF (Mazzaglia et al., ICRA 2025) — 100% success on sparse Robosuite Door where all RL baselines score 0%.

### Approach E: Hybrid Discrete/Neural

**Objective**: Discrete high-level planning + neural low-level control.

```
Level 2 (discrete AIF, ~100 ticks):
  State: phase x role x territory_mode
  Policy: {MINE_CHAIN, ALIGN_CHAIN, EXPLORE, DEFEND}
  Planning: EFE with T=2 lookahead over discrete states

Level 1 (neural, ~20 ticks):
  State: spatial embedding + entity features
  Policy: navigation to target, interaction sequences
  Training: PPO with L2-selected subgoal as reward

Level 0 (scripted, 1 tick):
  Actions: move_N/S/E/W, noop
  Controller: A* pathfinding or learned policy
```

**Strengths**: Principled high-level decisions + scalable low-level control
**Connection to Subhojeet's vision**: Level 2 = options, Level 1 = world model, Level 0 = motor policy

---

## 8. Connection to General Agent Substrate

Subhojeet described a "general agent substrate" with:

| Substrate Component | AIF Equivalent | Mathematical Connection |
|---|---|---|
| **World model** | Generative model P(o,s) | Both predict future from present; AIF adds preference-directed planning |
| **Prediction loss** | VFE accuracy term | E_q[-ln P(o\|s)] — identical |
| **Planning** | EFE minimization | G(π) = risk + ambiguity over T-step horizon |
| **Intrinsic motivation** (successor features) | Epistemic value (ambiguity term) | Both drive exploration; AIF is Bayes-optimal for POMDPs |
| **Options** | Hierarchical generative model | Each option = sub-POMDP with own A/B/C |
| **Continual learning** | Hyperprior adaptation | Meta-learning over A/B matrices (MAML) |

The AIF framework subsumes these components under a single objective (VFE/EFE minimization) rather than requiring separate loss functions for each component. Whether this integration is beneficial or whether modular losses work better is an empirical question — and CogsGuard is the testbed.

---

## 9. Summary: Inputs, Outputs, Objectives

### Per-timestep cycle

```
1. OBSERVE:  o_t = discretize(raw_obs_t)
2. INFER:    q(s_t) = argmin_q F[q, o_t]           -- minimize VFE
3. PLAN:     π* = argmin_π G(π)                     -- minimize EFE
4. ACT:      a_t ~ σ(-β · G(π*))                   -- softmax action selection
5. UPDATE:   q̃(s_{t+1}) = B @ q(s_t | a_t)         -- predict next state
```

### Loss functions

| What | Loss | What it optimizes |
|------|------|------------------|
| Beliefs about current state | VFE: F = D_KL[q\|\|P(s\|o)] | Accurate inference |
| Policy selection | EFE: G = risk + ambiguity | Goal-directed + exploratory behavior |
| World model (if neural) | VFE: -E[ln P(o\|s)] + D_KL[q(s\|o) \|\| p(s)] | Accurate prediction |
| Meta-learning (MAML) | L_meta = Σ_τ L_VFE(θ_τ, D_τ^test) | Fast adaptation |

### Key references

- **VFE/EFE framework**: Parr, Pezzulo & Friston (2022). *Active Inference* (MIT Press)
- **EFE as Bayes-optimal exploration**: Wei (2024). arXiv:2408.06542
- **EFE decomposition**: Champion et al. (2024). arXiv:2402.14460
- **R-AIF (neural)**: Mazzaglia et al. (2024). arXiv:2409.14216
- **Multi-agent factorization**: Ruiz-Serra et al. (2025). arXiv:2411.07362
- **Empathy / G-coupling**: Albarracin et al. (2026). arXiv:2602.20936
- **ToM in AIF**: Bramblett et al. (2025). arXiv:2501.03907
- **Deep AIF, long-horizon**: Mazzaglia et al. (2025). arXiv:2505.19867
- **pymdp implementation**: Heins et al. (2022). JOSS 7(73)
