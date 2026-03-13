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

### Phase 1: Discrete (pymdp)

**State space** (factored, 216 states):
- `phase(6)`: EXPLORE, MINE, DEPOSIT, CRAFT, GEAR, CAPTURE
- `hand(3)`: EMPTY, HOLDING_RESOURCE, HOLDING_GEAR
- `target_mode(3)`: FREE, CONTESTED, LOST
- `role(4)`: MINER, ALIGNER, SCOUT, SCRAMBLER

**Observation modalities** (discretized from tokens):
- `o_resource`: {NONE, NEAR, AT_RESOURCE}
- `o_station`: {NONE, NEAR_DEPOT, NEAR_CRAFT, NEAR_GEAR, NEAR_JUNCTION}
- `o_inventory`: {EMPTY, HAS_RESOURCE, HAS_GEAR}
- `o_social`: {ALONE, TEAMMATE_NEAR, OPPONENT_NEAR}
- `o_junction`: {NONE, NEUTRAL, TEAM, OPPONENT}

**EFE**:
```
G(pi) = E[-ln P(o|C)]                    # pragmatic: match preferences
      - E[D_KL[q(s|o,pi) || q(s|pi)]]    # epistemic: information gain
```

### Phase 3: Neural AIF

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
        for action in range(5):
            z_next = world_model.predict(z, action)
            g_pragmatic = -preferences.score(world_model.decode(z_next))
            g_epistemic = -world_model.uncertainty(z, action)
            G.append(g_pragmatic + g_epistemic)
        return softmax_sample(-beta * tensor(G))
```
