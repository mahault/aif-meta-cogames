# Literature Review: Parameter Learning in Active Inference

**Compiled**: March 2026 | **Project**: aif-meta-cogames

This document surveys the academic literature on learning generative model parameters
(A, B, C, D, E) in active inference, with focus on methods applicable to our 288-state
factored POMDP for CogsGuard.

---

## Table of Contents

1. [The VFE/EFE Division of Labour](#1-the-vfeefe-division-of-labour)
2. [Learning A Matrices (Observation Likelihood)](#2-learning-a-matrices)
3. [Learning B Matrices (Transition Dynamics)](#3-learning-b-matrices)
4. [Learning C Vectors (Prior Preferences)](#4-learning-c-vectors)
5. [Joint Parameter Optimization](#5-joint-parameter-optimization)
6. [De Novo Learning (Friston 2025)](#6-de-novo-learning)
7. [Differentiable Planning](#7-differentiable-planning)
8. [Multi-Agent Parameter Learning](#8-multi-agent-parameter-learning)
9. [Factored POMDPs and Structure Learning](#9-factored-pomdps-and-structure-learning)
10. [Novelty of Our Work](#10-novelty-of-our-work)
11. [Reference List](#11-reference-list)

---

## 1. The VFE/EFE Division of Labour

Active inference posits two complementary free energy functionals with distinct
temporal scopes and parameter dependencies.

### 1.1 Variational Free Energy (F) -- Perception and Learning

```
F = E_Q(s,pi) [ -ln P(o,s,pi) + ln Q(s,pi) ]
  = sum_tau [ D_KL[Q(s_tau) || P(s_tau|s_{tau-1},a)] - E_Q(s_tau)[ln P(o_tau|s_tau)] ]
```

**Parameter dependencies**: F depends on A (through the likelihood `ln P(o|s)`)
and B (through the prior `ln P(s'|s,a)`) and D (through `ln P(s_1)`).
**C does not appear in VFE.**

Minimizing F w.r.t. Q(s) yields perception (state estimation).
Minimizing F w.r.t. model parameters yields learning.

**Citation**: Parr & Friston (2019). "Generalised free energy and active inference."
*Biological Cybernetics*, 113, 495-513. [PMC6848054]

### 1.2 Expected Free Energy (G) -- Action and Planning

```
G(pi) = sum_{tau>t} E_Q(o,s|pi) [ ln Q(s|pi) - ln P(o,s) ]
       = Risk + Ambiguity
       = D_KL[Q(o|pi) || P(o|C)] + E_Q(s|pi)[H[P(o|s)]]
```

Equivalently:
```
G(pi) = -Information_gain - Pragmatic_value
       = -E_Q(o|pi)[D_KL[Q(s|o,pi) || Q(s|pi)]] - E_Q(o|pi)[ln P(o|C)]
```

**Parameter dependencies**: G depends on A, B (through predicted future states/obs),
and **C** (through the risk/pragmatic term). Policy selection: `Q(pi) = sigma(ln E - G(pi))`.

**Citation**: Da Costa et al. (2020). "Active inference on discrete state-spaces: A synthesis."
*Journal of Mathematical Psychology*, 99, 102447. [arXiv:2001.07203]

### 1.3 The Critical Structural Result

| Parameter | In VFE? | In EFE? | Learnable from VFE? | Learnable from EFE? |
|-----------|---------|---------|---------------------|---------------------|
| **A** (likelihood) | Yes | Yes | Yes (accuracy term) | Yes (ambiguity) |
| **B** (transition) | Yes | Yes | Yes (complexity term) | Yes (state prediction) |
| **C** (preferences) | **No** | Yes | **No** | Yes (risk/pragmatic) |
| **D** (initial state) | Yes | Indirect | Yes (prior term) | Indirectly |
| **E** (habits) | No | Yes (policy prior) | No | Yes |

**Implication**: A and B can be learned from VFE (standard approach). C requires
EFE or an alternative objective. This is the fundamental gap our work addresses.

### 1.4 The EFE Controversy

Millidge, Tschantz & Buckley (2021) demonstrate that EFE is NOT simply "VFE in
the future" -- the natural temporal extension of VFE **discourages exploration**.
EFE requires additional assumptions (preference priors) to yield the standard
risk+ambiguity decomposition.

Champion et al. (2024) identify four distinct EFE formulations (RSA, ROA, IGPV, 3E)
and show they are NOT equivalent. Only the ROA formulation naturally accommodates
C-learning, but it "lacks a principled justification from the free energy principle."

van der Himst & Lanillos (2025) partially resolve this by showing EFE-based planning
arises from VFE on an **augmented** generative model with preference/epistemic priors.
This provides the strongest current theoretical license for differentiating through G.

**Key citations**:
- Millidge et al. (2021). "Whence the Expected Free Energy?" *Neural Computation*, 33(2). [MIT Press]
- Champion et al. (2024). "Reframing the Expected Free Energy." [arXiv:2402.14460]
- van der Himst & Lanillos (2025). "EFE-based Planning as Variational Inference." [arXiv:2504.14898]

### 1.5 The Generalised Free Energy (Unification)

Parr & Friston (2019) show VFE and EFE unify into **generalised free energy**:

```
F_bar(pi,tau) = Complexity + Risk + Ambiguity
              = D_KL[Q(s_tau|pi) || E_Q(s_{tau-1})[P(s_tau|s_{tau-1},pi)]]
              + D_KL[Q(o_tau|pi) || P(o_tau|C)]
              + E_Q(s_tau|pi)[H[P(o_tau|s_tau)]]
```

For past timesteps (tau <= t), the risk term vanishes (observations are known) and
generalised FE reduces to standard VFE. For future timesteps (tau > t), it includes
the preference-dependent risk term. This formally justifies using VFE for A/B learning
(past) and EFE for C learning (future).

---

## 2. Learning A Matrices

### 2.1 Bayesian Approach: Dirichlet Conjugate Updates

The canonical Bayesian update for A (likelihood) parameters:

```
alpha'_A[i,j] = alpha_A[i,j] + sum_t 1{o_t = i} * q_t(s = j)
```

This accumulates observation-state co-occurrences under posterior beliefs.
Formally identical to Hebbian plasticity.

**Key paper**: Friston et al. (2016). "Active inference and learning."
*Neuroscience & Biobehavioral Reviews*, 68, 862-879. [PMC5167251]

The step-by-step tutorial (Smith et al. 2022) provides the practical implementation:
place Dirichlet priors over categorical distributions; since Dirichlet is conjugate to
categorical, the posterior remains Dirichlet after updating with data. An effective
learning rate `eta` scales the accumulated evidence.

**Citation**: Smith, Friston & Whyte (2022). "A step-by-step tutorial on active
inference." *J. Math. Psych.*, 107, 102632. [PMC8956124]

### 2.2 Gradient-Based A Learning

Catal et al. (2020) replace discrete A with a neural likelihood model `p_xi(o|s)`
trained via ELBO optimization (reconstruction + KL). Three networks: transition,
likelihood, posterior -- learned end-to-end.

Mazzaglia et al. (2021) go further with **Contrastive Active Inference**, eliminating
pixel-level reconstruction. The contrastive free energy uses dot-product similarity:
`f(o,s) = h(o)^T g(s)`. 13.8x fewer operations than likelihood-based approaches.

**Key citations**:
- Catal et al. (2020). "Learning generative state space models for active inference."
  *Frontiers in Computational Neuroscience*, 14, 574372.
- Mazzaglia, Verbelen & Dhoedt (2021). "Contrastive Active Inference." *NeurIPS 2021*.

### 2.3 Structure Learning of A

Smith et al. (2020) learn not just A values but A **structure** -- which states
connect to which observations. Uses Bayesian Model Reduction to prune unnecessary
connections, combined with model expansion to add new hidden states.

**Citation**: Smith et al. (2020). "An active inference approach to modeling structure
learning." *Frontiers in Computational Neuroscience*, 14, 41. [PMC7250191]

### 2.4 Our Phase B Results

Our Phase B experiments validated gradient-based A learning via JAX autodiff through
VFE. Key finding: **multi-agent averaging prevents role-biased A corruption** (B5:
+47% junctions). Online Dirichlet updates (B7) also work without pre-training (+20%).

---

## 3. Learning B Matrices

### 3.1 Bayesian Approach: Dirichlet Conjugate Updates

The canonical update for B (transition) parameters:

```
alpha'_B[j,k,l] = alpha_B[j,k,l] + sum_t q_t(s_{t+1} = j) * q_t(s_t = k) * 1{a_t = l}
```

This accumulates joint posterior beliefs over consecutive state transitions paired
with executed actions. Derives from the VFE complexity term.

**Citation**: Da Costa et al. (2020), Equations 32-33. Also "A Concise Mathematical
Description of Active Inference in Discrete Time" [arXiv:2406.07726].

### 3.2 Transition Prediction Loss (Deep AIF)

For neural/differentiable models, the transition loss from VFE decomposes as:

```
L_transition = D_KL[q(s_tau|pi) || E_q(s_{tau-1}|pi)[P(s_tau|s_{tau-1}, a)]]
```

This is the KL between inferred posterior and B-predicted prior. Equivalently,
the negative expected log-likelihood of the transition model:

```
L_B = -E_q(s_t) E_q(s_{t+1}) [ln B(s'|s, a)]
```

This is standard in deep AIF (Fountas et al. 2020, Catal et al. 2020).

### 3.3 Action-Oriented Models

Tschantz, Seth & Buckley (2020) demonstrate that AIF agents learn **action-oriented**
models -- internal models biased toward task-relevant aspects rather than veridical
representations. This means learned B matrices need not match true environment dynamics,
only capture transitions relevant for goal-directed behavior.

**Warning**: "Bad bootstrap" -- action-oriented models can prematurely converge to
sub-optimal solutions because the agent's behavior determines its training data.
Active inference mitigates this via the EFE epistemic term (information gain).

**Citation**: Tschantz, Seth & Buckley (2020). "Learning action-oriented models
through active inference." *PLoS Comp. Bio.*, 16(4), e1007805.

### 3.4 Retrospective Inference for B

Friston et al. (2016) note parameter learning uses **postdicted** (smoothed) beliefs
rather than filtered beliefs, exploiting the full information from each trial. The VFE
naturally supports this: F for past timesteps depends on observed outcomes, allowing
refined transition estimates.

### 3.5 Our Gap

Our Phase B `trajectory_vfe()` computes VFE but **discards B_logits** (line 173:
`_ = B_logits`). B matrices received zero gradient in all experiments (v1-B7).
The transition prediction loss must be explicitly implemented.

---

## 4. Learning C Vectors

This is the least developed area in the literature and the most novel aspect of our work.

### 4.1 The Fundamental Problem

C does not appear in VFE. Therefore:
- VFE gradients w.r.t. C are zero
- C must be learned through EFE or an alternative objective
- Standard active inference treats C as a **design parameter**, not a learned one

### 4.2 Inverse Active Inference (Shin et al. 2022)

**The most directly relevant paper.** Treats EFE as a negative value function and
proposes inverse RL to recover C from expert demonstrations.

```
Loss = -log q_pi(a_observed)
where q_pi = sigma(gamma * G + ln E)
and G depends on C through compute_expected_utility(qo, C)
```

The EFE-optimal policy under learned C should reproduce observed expert behavior.
Formally analogous to MaxEnt IRL (Ziebart et al. 2008) but in AIF language.

**Key insight**: ln P(o|C) in AIF plays the role of r(s,a) in MaxEnt RL.

**Limitation**: Tested only on small, flat state spaces -- not factored 288-state POMDPs.

**Citations**:
- Shin, Kim & Hwang (2022). "Prior Preference Learning from Experts." *Neurocomputing*, 492, 508-515. [arXiv:2101.08937]
- Ziebart et al. (2008). "Maximum Entropy Inverse Reinforcement Learning." *AAAI*.
- Levine (2018). "RL and Control as Probabilistic Inference." [arXiv:1805.00909]

### 4.3 Online Preference Learning (Sajid et al. 2022)

Double-loop mechanism: inner loop runs standard AIF with current C; outer loop
updates C across episodes via Dirichlet accumulation. In static environments,
agents develop confident preferences; in volatile environments, they maintain
uncertainty and continue exploring.

**Citation**: Sajid, Tigas & Friston (2022). "Active inference, preference learning
and adaptive behaviour." *IOP Conf. Series*, 1261, 012020.

### 4.4 Connection to Control-as-Inference

Millidge et al. (2020) formally compare AIF and control-as-inference (CAI):

| | AIF | CAI (MaxEnt RL) |
|---|---|---|
| Value source | C vector (log preferences) | Reward function r(s,a) |
| Policy | q(pi) = sigma(-G) | pi(a|s) = exp(Q - V) |
| Exploration | Epistemic value (EFE) | Entropy bonus |

When ambiguity is zero and information gain negligible, EFE reduces to negative
expected reward under MaxEnt RL. This formally justifies using cross-entropy
Loss = -log q_pi(a_observed) for C learning.

**Citation**: Millidge, Tschantz, Seth & Buckley (2020). "On the Relationship Between
Active Inference and Control as Inference." [arXiv:2006.12964]

### 4.5 IRL in POMDPs (Choi & Kim 2011)

Extends IRL to POMDPs -- directly relevant since our 288-state model is a POMDP.
Their key limitation was non-differentiability of the POMDP solution, requiring
finite-difference approximations. **Our JAX approach solves this** by making the
entire EFE computation differentiable, enabling exact gradients.

**Citation**: Choi & Kim (2011). "Inverse RL in Partially Observable Environments."
*JMLR*, 12, 691-730.

### 4.6 Reward Maximization Equivalence (Da Costa et al. 2023)

Proves that for horizon T=1, active inference with appropriate C recovers
Bellman-optimal behavior. The C vector maps directly to a reward function under
specific conditions. This justifies treating inverse C-learning as equivalent to IRL.

**Citation**: Da Costa et al. (2023). "Reward Maximization Through Discrete Active
Inference." *Neural Computation*, 35(5), 807-852. [arXiv:2009.08111]

### 4.7 EFE Degeneracy Warning

Champion et al. (2023) find that EFE-minimizing agents can degenerate: "the agent
repeatedly picks a single action and becomes an expert at predicting the future
when selecting this action." The epistemic term can lose rather than gain information.

**Implication for C learning**: Lower learning rate for C is essential. Monitor for
entropy collapse in q_pi during training.

**Citation**: Champion et al. (2023). "Deconstructing deep active inference."
*Neural Computation*, 36(11), 2403. [arXiv:2303.01618]

### 4.8 Circular Dependency Risk

Learning C from trajectories collected with hand-tuned C would recover the hand-tuned
values (tautological). **Mitigation**: Learn A+B first (fix perception and dynamics),
collect NEW trajectories with learned A+B (may produce different behavior), then learn
C from those improved trajectories.

---

## 5. Joint Parameter Optimization

### 5.1 Classical: Sequential Bayesian Updates

Friston et al. (2016) learns A, B, D via Dirichlet accumulation at trial end.
C is fixed. E (habits) accumulates successful policy counts. All updates derive
from VFE minimization under conjugate priors.

### 5.2 Deep: Separate Objectives per Parameter Type

Millidge (2020) trains four neural networks jointly:
- Perception model (encoder/decoder) → learns A via VFE reconstruction
- Transition model → learns B via VFE prediction
- Policy network → trained via EFE
- EFE value network → bootstrapped: `L = ||G_psi(s,a) - G_target||^2`

**Critical insight**: A and B trained via VFE on past data; policy trained via
EFE for future planning. C is fixed in this work.

**Citation**: Millidge (2020). "Deep Active Inference as Variational Policy Gradients."
*J. Math. Psych.*, 96, 102348. [arXiv:1907.03876]

### 5.3 Deep AIF with Monte Carlo Methods (Fountas et al. 2020)

Four innovations: (1) MCTS over EFE, (2) habitual network approximating q(pi),
(3) MC dropout for parameter belief updates, (4) transition precision optimization.
All generative model components learned as neural networks, creating disentangled
representations.

**Citation**: Fountas et al. (2020). "Deep active inference agents using Monte-Carlo
methods." *NeurIPS 2020*. [GitHub: zfountas/deep-active-inference-mc]

### 5.4 Two-Timescale Approach (Recommended)

Based on the literature, the consensus approach for joint optimization:

```
L_total = w_A * L_vfe(A)           # Fast: perception (VFE)
        + w_B * L_transition(B)    # Fast: dynamics (VFE)
        + w_C * L_policy(A,B,C)    # Slow: preferences (inverse EFE)
        + w_kl * L_regularization  # Prevent drift from priors
```

Separate optimizers with different learning rates. C gets 0.1x base LR
due to scale sensitivity through the softmax.

---

## 6. De Novo Learning (Friston et al. 2025)

### 6.1 Overview

The most comprehensive recent paper on learning ALL parameters from scratch without
gradient descent.

**Citation**: Friston, K., Parr, T., Heins, C., Da Costa, L., Salvatori, T.,
Tschantz, A., Koudahl, M., Van de Maele, T., Buckley, C., & Verbelen, T. (2025).
"Gradient-Free De Novo Learning." *Entropy*, 27(9), 992. [PMC12468873]

### 6.2 Three-Phase Pipeline

**Phase 1 — Structure Discovery**: Spectral clustering discovers hidden state
structure from observations. The algorithm:
1. Collect raw observation trajectories
2. Compute similarity matrix over observation sequences
3. Spectral decomposition reveals latent state structure
4. Assign observations to discovered hidden states
5. Determine number of states via eigenvalue gap

**Phase 2 — Parameter Learning**: Dirichlet counts accumulate selectively under
reward constraints. The model "grows and then reduces until it discovers a
pullback attractor." Bayesian Model Reduction (BMR) is applied bidirectionally:
- **Reachability tests**: Identify states connecting to goals
- **State merging**: Combine states preserving mutual information
- **Pruning**: Remove states/transitions with insufficient evidence

**Phase 3 — Refinement**: Continual learning via inductive inference. The agent
uses the learned model for planning while continuing to update parameters from
new experience. BMR continues to simplify the model as evidence accumulates.

### 6.3 Key Mechanisms

**Bayesian Model Reduction (BMR)**: Analytically compares nested models without
refitting. If removing a parameter (pruning an A/B connection) doesn't increase F,
the simpler model is preferred. This provides automatic structure learning.

**Inductive Inference**: The agent's own successful actions generate data for
model refinement. The EFE epistemic term drives exploration toward informative
states, accelerating parameter convergence.

**Growth-Reduction Cycle**: The model first grows (adds states/connections to
explain data) then reduces (prunes via BMR to find minimal sufficient model).
This cycle converges to a "pullback attractor" — the simplest model that
explains the data and supports goal-directed behavior.

### 6.4 What C Learning Looks Like in De Novo

In the de novo framework, C is not learned via gradients but emerges from the
structure discovery process:
1. Observations associated with reward/goal states are identified
2. C is set to prefer these observations (high log-probability)
3. BMR refines C by testing whether preference removal hurts performance
4. The final C reflects which observations are instrumentally useful for reaching goals

### 6.5 Limitations

- Demonstrated on a small arcade game (not multi-agent, not factored POMDP)
- Spectral clustering assumes observation similarity reflects state similarity
- No gradient information — slower convergence than gradient methods
- Structure discovery is a one-time preprocessing step, not online

### 6.6 Relevance to Our Work

The de novo approach offers three ideas directly applicable to our setting:

1. **BMR for model simplification**: After gradient-based A/B/C learning, apply BMR
   to prune unnecessary connections in our 288-state POMDP. May discover that some
   state factors or observation modalities are redundant.

2. **Growth-reduction cycle**: Start with a simpler model (fewer states), learn,
   evaluate via VFE, add states/factors only if VFE improves. This addresses the
   question of whether 288 states is the right granularity.

3. **Structure discovery from data**: Rather than hand-designing the factored state
   space (phase x hand x target_mode x role), discover the factorization from
   trajectory data. Spectral clustering on observation sequences could reveal
   whether our four factors are the natural decomposition.

---

## 7. Differentiable Planning

### 7.1 QMDP-Net (Karkus et al. 2017)

Embeds a POMDP solver inside a differentiable neural network: Bayesian filter
(belief update) + QMDP planner (value iteration). End-to-end training learns
transition, observation, and reward models (equivalent to B, A, C).

**Architecturally closest to our approach.** Key difference: QMDP ignores future
observations (no epistemic value); our EFE includes information gain.

**Citation**: Karkus, Hsu & Lee (2017). "QMDP-Net: Deep Learning for Planning
under Partial Observability." *NeurIPS 2017*. [arXiv:1703.06692]

### 7.2 Differentiable MPC (Amos et al. 2018)

Makes MPC differentiable via KKT conditions at the fixed point. Gradients flow
through the planning computation back to cost function and dynamics parameters.

**Continuous-domain analog of our approach.** They differentiate through MPC; we
differentiate through EFE. Their implicit differentiation via KKT conditions; ours
uses explicit forward-mode autodiff through JAX's discrete computation graph.

**Citation**: Amos et al. (2018). "Differentiable MPC for End-to-end Planning and
Control." *NeurIPS 2018*. [arXiv:1810.13400]

### 7.3 DreamerV3 (Hafner et al. 2023/2025)

Learns a latent world model (RSSM) and optimizes policy by imagining trajectories.
Critically, DreamerV3 does NOT backpropagate through the world model for policy
learning — uses imagined rollouts + actor-critic. The world model is trained via
reconstruction loss separately.

**Relevant for B learning**: DreamerV3's world model training = learning transition
dynamics from observations. Our B-matrix learning via transition prediction loss is
the discrete POMDP analog.

**Citation**: Hafner et al. (2025). "Mastering Diverse Domains through World Models."
*Nature*, 640, 647-653. [arXiv:2301.04104]

### 7.4 Plan2Explore (Sekar et al. 2020)

Learns world model through pure exploration (maximizing expected information gain),
then adapts to downstream tasks. Uses ensemble disagreement as exploration bonus.

The information gain component of our EFE serves the same role. Validates keeping
the epistemic term during C-learning.

**Citation**: Sekar et al. (2020). "Planning to Explore via Self-Supervised World
Models." *ICML 2020*. [arXiv:2005.05960]

---

## 8. Multi-Agent Parameter Learning

### 8.1 Factorised Active Inference for Multi-Agent (Ruiz-Serra et al. 2024)

Each agent maintains individual-level beliefs about other agents' internal states
and uses them for strategic planning. Self-model + per-opponent models.

**Directly relevant to 8-agent CogsGuard.** Their factorisation pattern (self + others)
is what our ToM modulator already implements. Our contribution: **learning** those
models from trajectory data rather than hand-specifying.

**Citation**: Ruiz-Serra, Sweeney & Harre (2024). "Factorised Active Inference for
Strategic Multi-Agent Interactions." *AAMAS 2025*. [arXiv:2411.07362]

### 8.2 Theory of Mind via Active Inference (Pitliya et al. 2025)

ToM agents maintain distinct representations of own and others' beliefs/goals.
Uses adapted sophisticated inference with tree-based planning for recursive
reasoning. No shared models or communication required.

**Finding**: ToM agents cooperate better by avoiding collisions and reducing redundant
efforts — validates our social-layer architecture.

**Citation**: Pitliya et al. (2025). "Theory of Mind Using Active Inference."
[arXiv:2508.00401]

### 8.3 Empathy Modeling (Albarracin et al. 2026)

Self-other model transformation for empathic perspective-taking. Empathy parameter
lambda scales the other-model weight. Learning-enabled variant infers opponent type.

**Our lambda-weighted EFE decomposition** `G(pi) = (1-lambda)*G_self + lambda*G_other`
is a direct instantiation. Our contribution: learning lambda and C from data.

**Citation**: Albarracin, Mikeda, Jimenez Rodriguez et al. (2026). "Empathy Modeling
in Active Inference Agents." [arXiv:2602.20936]

### 8.4 Multi-Agent Role Bias (Our Finding)

Our Phase B experiments revealed that single-agent learning corrupts A columns for
observation values that agent rarely encounters (e.g., miner never sees HAS_GEAR).
**Multi-agent averaging across all roles** prevents this bias (B5 vs B6).

This finding is novel — no existing work addresses role-biased parameter corruption
in multi-agent active inference.

---

## 9. Factored POMDPs and Structure Learning

### 9.1 Factored Bayes-Adaptive POMDPs (Katt et al. 2019)

Maintains Dirichlet posteriors over factored transition parameters. Two modes:
(a) learn parameters given known factorisation, (b) learn both structure and
parameters. Uses particle-based belief tracking and MCTS.

**Directly relevant to our B-matrix learning.** Our 4-factor state with dependencies
is a factored POMDP. Their FBA-POMDP is the Bayesian (non-gradient) approach; our
JAX gradient approach is the differentiable alternative.

**Citation**: Katt, Oliehoek & Amato (2019). "Bayesian RL in Factored POMDPs."
*AAMAS 2019*. [arXiv:1811.05612]

### 9.2 Structure Learning in Factored MDPs (Strehl & Diuk 2007)

SLF-Rmax learns the Dynamic Bayesian Network structure of factored MDPs online.
Achieves near-optimal behavior with polynomial sample complexity.

**Relevant if** we want to **discover** the factorisation structure rather than
assuming phase x hand x target_mode x role.

**Citation**: Strehl & Diuk (2007). "Efficient Structure Learning in Factored-State
MDPs." *AAAI 2007*.

### 9.3 Message Passing Approximations

Parr et al. (2019) compare mean-field, Bethe, and marginal approximations:
- **Mean-field**: Fully factored Q(s) = prod_f Q(s_f). Tractable but ignores
  cross-factor correlations.
- **Bethe**: Captures pairwise temporal dependencies. More accurate but requires
  tracking pairwise marginals.
- **Variational Message Passing**: Enables parallel factor-wise updates — advantageous
  for our 4-factor POMDP.

**Our pymdp implementation uses mean-field across factors** (A_DEPENDENCIES, B_DEPENDENCIES
define sparse interactions). This is computationally tractable and aligned with standard
discrete AIF.

**Citation**: Parr, Markovic, Kiebel & Friston (2019). "Neuronal message passing
using Mean-field, Bethe, and Marginal approximations." *Scientific Reports*, 9, 1889.

### 9.4 Constrained Bethe Free Energy (van de Laar et al. 2022)

Combines Bethe FE tractability with EFE epistemic qualities. Expressible as message
passing on Forney-style factor graphs — enables **automatic derivation** of inference
and learning updates for arbitrary factored structures.

**Citation**: van de Laar et al. (2022). "Active Inference and Epistemic Value in
Graphical Models." *Frontiers in Robotics and AI*, 9, 794464. [PMC9019474]

---

## 10. Novelty of Our Work

Based on this literature review, our work is novel in four ways:

### 10.1 Inverse C-Learning via Differentiable EFE in Factored POMDPs

Shin et al. (2022) proposed inverse EFE for C learning but worked with small flat
state spaces. We scale to a **288-state factored POMDP** (phase x hand x target_mode
x role) using JAX autodiff through the full pymdp EFE computation. This hasn't been
done before.

### 10.2 Joint Gradient-Based A+B+C Optimization

No existing work performs joint gradient-based optimization of all three parameter
types in a single differentiable pipeline. Friston (2016) learns A, B via Dirichlet;
Shin (2022) learns C via inverse RL; we unify both in a JAX computation graph with
separate objectives (VFE for A/B, inverse EFE for C).

### 10.3 Multi-Agent Role Bias Discovery and Fix

Our finding that single-role trajectory learning corrupts A matrices for unobserved
states, and that multi-agent averaging fixes this, is novel. No existing work
addresses role-specific parameter bias in multi-agent AIF.

### 10.4 De Novo-Inspired Structure Learning for Factored AIF

Applying Friston (2025) de novo learning concepts (BMR, growth-reduction cycles,
spectral structure discovery) to a factored multi-agent POMDP is novel. The original
de novo paper demonstrates on small single-agent games.

---

## 11. Reference List

### Foundational Active Inference

1. Friston, K. et al. (2016). "Active inference and learning." *Neuroscience & Biobehavioral Reviews*, 68, 862-879. [PMC5167251]
2. Friston, K. et al. (2017). "Active Inference: A Process Theory." *Neural Computation*, 29(1), 1-49.
3. Parr, T. & Friston, K. (2019). "Generalised free energy and active inference." *Biological Cybernetics*, 113, 495-513. [PMC6848054]
4. Da Costa, L. et al. (2020). "Active inference on discrete state-spaces: A synthesis." *J. Math. Psych.*, 99, 102447. [arXiv:2001.07203]
5. Smith, R. et al. (2022). "A step-by-step tutorial on active inference." *J. Math. Psych.*, 107, 102632. [PMC8956124]
6. Sajid, N. et al. (2021). "Active Inference: Demystified and Compared." *Neural Computation*, 33(3), 674-712.

### EFE Theory and Critique

7. Millidge, B. et al. (2021). "Whence the Expected Free Energy?" *Neural Computation*, 33(2), 447-482.
8. Champion, T. et al. (2024). "Reframing the Expected Free Energy." [arXiv:2402.14460]
9. Champion, T. et al. (2023). "Deconstructing deep active inference." *Neural Computation*, 36(11). [arXiv:2303.01618]
10. van der Himst, O. & Lanillos, P. (2025). "EFE-based Planning as Variational Inference." [arXiv:2504.14898]
11. Gottwald, S. & Braun, D.A. (2020). "The two kinds of free energy and the Bayesian revolution." *PLoS Comp. Bio.*, 16(12).

### Preference and C-Vector Learning

12. Shin, J.Y. et al. (2022). "Prior Preference Learning from Experts." *Neurocomputing*, 492, 508-515. [arXiv:2101.08937]
13. Sajid, N. et al. (2022). "Active inference, preference learning and adaptive behaviour." *IOP Conf. Series*, 1261.
14. Da Costa, L. et al. (2023). "Reward Maximization Through Discrete Active Inference." *Neural Computation*, 35(5). [arXiv:2009.08111]
15. Wei, R. (2024). "Value of Information and Reward Specification in Active Inference." [arXiv:2408.06542]

### Control-as-Inference and IRL

16. Levine, S. (2018). "RL and Control as Probabilistic Inference." [arXiv:1805.00909]
17. Millidge, B. et al. (2020). "On the Relationship Between Active Inference and Control as Inference." [arXiv:2006.12964]
18. Ziebart, B.D. et al. (2008). "Maximum Entropy Inverse RL." *AAAI*.
19. Choi, J. & Kim, K.-E. (2011). "Inverse RL in Partially Observable Environments." *JMLR*, 12, 691-730.

### Deep Active Inference

20. Millidge, B. (2020). "Deep Active Inference as Variational Policy Gradients." *J. Math. Psych.*, 96. [arXiv:1907.03876]
21. Fountas, Z. et al. (2020). "Deep active inference agents using Monte-Carlo methods." *NeurIPS 2020*.
22. Mazzaglia, P. et al. (2021). "Contrastive Active Inference." *NeurIPS 2021*.
23. Catal, O. et al. (2020). "Learning generative state space models for active inference." *Frontiers Comp. Neurosci.*, 14.
24. Tschantz, A. et al. (2020). "Scaling active inference." *IJCNN 2020*. [arXiv:1911.10601]
25. Tschantz, A. et al. (2020). "Learning action-oriented models through active inference." *PLoS Comp. Bio.*, 16(4).
26. Yeganeh, Y.T. et al. (2025). "Deep AIF for Delayed and Long-Horizon Environments." [arXiv:2505.19867]

### De Novo and Structure Learning

27. Friston, K. et al. (2025). "Gradient-Free De Novo Learning." *Entropy*, 27(9), 992. [PMC12468873]
28. Smith, R. et al. (2020). "Active inference approach to modeling structure learning." *Frontiers Comp. Neurosci.*, 14. [PMC7250191]
29. Katt, S. et al. (2019). "Bayesian RL in Factored POMDPs." *AAMAS 2019*. [arXiv:1811.05612]
30. Strehl, A.L. & Diuk, C. (2007). "Efficient Structure Learning in Factored-State MDPs." *AAAI 2007*.

### Differentiable Planning

31. Karkus, P. et al. (2017). "QMDP-Net." *NeurIPS 2017*. [arXiv:1703.06692]
32. Amos, B. et al. (2018). "Differentiable MPC." *NeurIPS 2018*. [arXiv:1810.13400]
33. Hafner, D. et al. (2025). "DreamerV3." *Nature*, 640, 647-653. [arXiv:2301.04104]
34. Sekar, R. et al. (2020). "Planning to Explore via Self-Supervised World Models." *ICML 2020*. [arXiv:2005.05960]

### Multi-Agent Active Inference

35. Ruiz-Serra, J. et al. (2024). "Factorised Active Inference for Multi-Agent Interactions." *AAMAS 2025*. [arXiv:2411.07362]
36. Pitliya, R.J. et al. (2025). "Theory of Mind Using Active Inference." [arXiv:2508.00401]
37. Albarracin, M. et al. (2026). "Empathy Modeling in Active Inference Agents." [arXiv:2602.20936]
38. Matsumura, T. et al. (2024). "Active Inference With Empathy Mechanism." *Artificial Life*, 30(2), 277-297.

### Factored Models and Message Passing

39. Parr, T. et al. (2019). "Neuronal message passing using Mean-field, Bethe, and Marginal approximations." *Scientific Reports*, 9, 1889.
40. van de Laar, T.W. et al. (2022). "Active Inference and Epistemic Value in Graphical Models." *Frontiers in Robotics and AI*, 9. [PMC9019474]

### Software

41. Heins, C. et al. (2022). "pymdp: A Python library for active inference." *JOSS*, 7(73), 4098.
42. Nehrer, S. et al. (2025). "ActiveInference.jl." *Entropy*, 27(1), 62.
43. Bagaev, D. et al. (2023). "RxInfer." *JOSS*.

### World Model Learning

44. Fujii, K. et al. (2024). "Real-World Robot Control Based on Contrastive Deep AIF with Demonstrations." *IEEE Access*.
45. Collis, P. et al. (2024). "Learning in Hybrid Active Inference Models." *IWAI 2024*. [arXiv:2409.01066]
