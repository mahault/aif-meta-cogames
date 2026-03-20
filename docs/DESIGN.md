# Technical Design: AIF Meta-Learning on CoGames

## Data Format

**Source**: CoGames (CogsGuard) environments via MettaGrid simulator.

**Observations**: `(num_agents, 200, 3)` uint8 — 200 tokens per agent, each `(location, feature_id, value)`.
- Location: packed 8-bit `(row << 4 | col)` in 13x13 egocentric view. `0xFF` = empty, `0xFE` = global.
- Feature ID: index into 73 features (agent state, inventory, spatial, social).
- Value: 0-255 normalized.

**Actions**: `Discrete(5)` — noop, move_north, move_south, move_west, move_east.

**Trajectory files**: `.npz` with keys `obs(T, N, 200, 3)`, `actions(T, N)`, `rewards(T, N)`, `dones(T, N)`, `next_obs_final(N, 200, 3)`.

**Environment variants** (36 total):
- 2 maps: arena (50x50), machina1 (88x88)
- 3 agent counts: 2, 4, 8
- 2 difficulties: easy, medium
- 5 biomes: default, desert, forest, caves, city (arena only)

---

## World Model

### Observation Encoder

Two options (implement both, benchmark):

**Option A — Flat MLP**:
```python
x = obs.view(batch, -1).float() / 255.0  # (B, 600)
z = MLP(600 -> 256 -> 64)                # z ∈ R^64
```

**Option B — Token Transformer**:
```python
# Embed each token: (feature_id, value) -> 32-dim
token_emb = embed_feature(feature_id) + embed_value(value)  # (B, 200, 32)
# Positional encoding from location
pos = encode_location(location)                              # (B, 200, 32)
# Transformer over token set
z = transformer_pool(token_emb + pos)                        # (B, 64)
```

### Transition Model

```python
class WorldModel:
    encoder:    obs -> z_t           # (N, 200, 3) -> (N, 64)
    transition: (z_t, a_t) -> z_{t+1} # (N, 64+5) -> (N, 64)
    decoder:    z -> obs             # (N, 64) -> (N, 600)
    reward_head: z -> r              # (N, 64) -> (N, 1)
    done_head:   z -> done           # (N, 64) -> (N, 1)
```

**Training loss**:
```
L = reconstruction_loss(decoder(z_{t+1}), obs_{t+1})
  + reward_prediction_loss(reward_head(z_{t+1}), r_t)
  + done_prediction_loss(done_head(z_{t+1}), done_t)
```

---

## Meta-Learning (MAML)

**Task**: One environment variant = one task.

**Support set**: 2-3 episodes from the variant (for inner loop adaptation).
**Query set**: Remaining episodes (for meta-gradient evaluation).

```python
# Outer loop
for meta_step in range(N):
    meta_loss = 0
    for task in sample_tasks(K):
        theta_adapted = inner_loop(theta, task.support, steps=5)
        meta_loss += world_model_loss(theta_adapted, task.query)
    theta -= lr_outer * grad(meta_loss, theta)

# Inner loop
def inner_loop(theta, support_data, steps):
    theta_i = theta.clone()
    for _ in range(steps):
        loss = world_model_loss(theta_i, support_data)
        theta_i -= lr_inner * grad(loss, theta_i)
    return theta_i
```

**Train/test split**:
- Meta-train: ~30 variants
- Meta-test (held out): machina1_n8_medium, arena_n8_easy_desert, arena_n2_medium_caves (tests generalization along map, agent count, biome axes)

---

## AIF Agent

### Phase 3b (Current): Full 216-State Discrete AIF Agent

#### State Space (216 states = phase × hand × target_mode × role)

```
phase(6):        EXPLORE, MINE, DEPOSIT, CRAFT, GEAR, CAPTURE
hand(3):         EMPTY, HOLDING_RESOURCE, HOLDING_GEAR
target_mode(3):  FREE, CONTESTED, LOST
role(4):         GATHERER, CRAFTER, CAPTURER, SUPPORT
```

All four factors are meaningful because the POMDP action space is task-level
policies (not primitive movements), making B matrices action-dependent:
- **phase × hand**: Economy-chain progress
- **target_mode**: Junction contest status — affects EFE for CAPTURE vs YIELD
- **role**: Agent specialisation — affects which task policies are preferred via C

#### Action Space (13 task-level policies)

**Critical design decision**: The POMDP action space is 13 task-level policies,
NOT the 5 primitive movement actions. Each task policy has distinct B matrix
transitions, so pymdp's EFE computation produces non-uniform policy selection.

A navigator converts the selected task policy to primitive movement:

| Policy | B matrix effect | Navigator action |
|--------|----------------|-----------------|
| NAV_RESOURCE | EXPLORE → MINE | Move toward extractor |
| MINE | hand → HOLDING_RESOURCE | Noop (interact) |
| NAV_DEPOT | → DEPOSIT phase | Move toward hub |
| DEPOSIT | hand → EMPTY | Noop (interact) |
| NAV_CRAFT | → CRAFT phase | Move toward craft station |
| CRAFT | hand → HOLDING_GEAR | Noop (interact) |
| NAV_GEAR | → GEAR phase | Move toward craft station |
| ACQUIRE_GEAR | hand → HOLDING_GEAR | Noop (interact) |
| NAV_JUNCTION | → CAPTURE phase | Move toward junction |
| CAPTURE | cycle → EXPLORE/EMPTY | Noop (interact) |
| EXPLORE | self-transition | Wander |
| YIELD | self-transition | Move away from agent |
| WAIT | self-transition | Noop |

**Why not primitive movements?** CogsGuard has 5 actions (noop, N, S, E, W).
Phase/hand transitions happen when stepping onto specific tiles (extractors, hubs,
craft stations, junctions). So all 5 primitive actions produce the same state
transitions → B is action-independent → EFE is uniform → random action selection.
Task-level policies abstract over spatial movement, producing action-dependent B.

#### Observation Modalities (6)

| Modality | Dim | Source | Depends on |
|----------|-----|--------|-----------|
| o_resource | 3 | tag tokens (extractor proximity) | phase |
| o_station | 4 | tag tokens (hub/craft/junction) | phase |
| o_inventory | 3 | global inventory tokens | hand |
| o_contest | 3 | junction + agent:group tokens | target_mode |
| o_social | 4 | agent:group spatial tokens | weakly informative |
| o_role_signal | 2 | vibe tokens | role |

#### Architecture

```
Token Observation (200, 3) uint8
        ↓
ObservationDiscretizer
        ↓
(o_res, o_sta, o_inv, o_contest, o_social, o_role) — 6 discrete modalities
        ↓
pymdp.Agent (JAX, equinox Module)
  ├── infer_states() → posterior beliefs over 216 states
  ├── infer_policies() → EFE over 13 task policies
  └── update_empirical_prior() → state prediction for next step
        ↓
Task policy selection (argmax EFE)
        ↓
Navigator: task policy → primitive movement action
```

#### EFE Decomposition

```
G(π) = E_q(o,s|π) [ ln q(s|π) - ln P(o|C) - ln P(o|s) + ln q(s|o,π) ]
     = risk + ambiguity - information_gain

risk       = E[ D_KL[ q(o|π) || P(o|C) ] ]     — prefer preferred outcomes
ambiguity  = E[ H[P(o|s)] ]                      — avoid uncertain observations
info_gain  = E[ D_KL[ q(s|o,π) || q(s|π) ] ]    — seek informative states
```

With action-dependent B, different task policies lead to different expected
observations → different risk and information gain → meaningful policy selection.

#### Matrices

**A** (observation likelihood): 6 matrices, each `(n_obs_m, 216)`.
Hand-crafted defaults encode factor-observation dependencies:
- A[inventory] near-deterministic from hand
- A[contest] near-deterministic from target_mode
- A[resource/station] depend on phase

**B** (transition): `(216, 216, 13)`. Hand-crafted defaults, MAML-refinable.
Each task policy column encodes expected state transitions.
Role factor: self-transition (doesn't change within episode).
Target_mode: changes based on capture/contest outcomes.

**C** (preferences): 6 vectors.
- Prefer: AT resource, JUNCTION station, HAS_GEAR inventory, FREE contest, ALLY_NEAR
- Avoid: LOST contest, ENEMY_NEAR

**D** (prior): `(216,)`. Peak at EXPLORE/EMPTY/FREE/GATHERER.

#### MAML Target

For meta-learning (Luca's Phase 2):
- A/B matrices are the meta-learning parameters
- Inner loop: fit A/B from 2-3 episodes of a new variant via MLE + Dirichlet smoothing
- Outer loop: learn initialization θ* that minimizes post-adaptation loss across variants
- B matrix has 606K entries but is block-sparse (most transitions are within-factor)

### Phase 4 (NEXT): Neural AIF + Meta-Learned World Model

Replace hand-crafted A/B with learned world model:
- A matrix = decoder network
- B matrix = transition network
- C vector = hand-crafted (CogsGuard preferences: hearts > 0, junctions captured)
- Epistemic value = ensemble disagreement or encoder variance

```python
class NeuralAIFAgent:
    def act(self, obs):
        z = world_model.encode(obs)
        G = []
        for action in range(13):  # 13 task-level policies
            z_next = world_model.predict(z, action)
            g_pragmatic = -preferences.score(world_model.decode(z_next))
            g_epistemic = -world_model.uncertainty(z, action)
            G.append(g_pragmatic + g_epistemic)
        return softmax_sample(-beta * tensor(G))
```
