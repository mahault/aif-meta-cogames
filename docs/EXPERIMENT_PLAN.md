# Experiment Plan

## Hypotheses

- **H1**: MAML-initialized world model adapts to held-out variants in fewer gradient steps than random-init or pre-trained baselines
- **H2**: AIF agent with adapted world model explores more effectively (discovers gear stations) than one with a generic model
- **H3**: Epistemic value drives 50%+ more unique state coverage than pragmatic-only
- **H4**: Meta-learned representations capture task-invariant structure (economy chain) while adapting environment-specific structure (spatial layout)

## Experiment Matrix

| ID | World Model | Meta-Learning | Agent | Variants | Tests |
|----|-------------|---------------|-------|----------|-------|
| E1 | MLP | None | N/A | 3 easy | Architecture sanity |
| E2 | MLP vs RNN | None | N/A | All 36 | Best world model architecture |
| E3 | MLP | MAML vs None | N/A | 30 train / 6 test | H1: adaptation speed |
| E4 | Hand-crafted | N/A | pymdp | 3 arena | H3: epistemic value |
| E5 | MLP (MAML) | MAML | Neural AIF | All | H2: end-to-end |
| E6 | MLP (MAML) | MAML | +/- epistemic | 6 test | H3: ablation |

## Baselines

| Baseline | Description |
|----------|-------------|
| Random agent | Uniform random actions |
| Biased-move agent | 90% move, 10% noop |
| Fixed world model | Single model trained on all data, no adaptation |
| Per-variant model | Trained from scratch per variant (upper bound) |
| Pre-trained (no meta) | Trained on all, fine-tuned per variant |
| Discrete AIF | pymdp with hand-crafted A/B/C |

## Metrics

### World Model Quality
- **Prediction MSE**: `||z_pred - z_true||^2` on held-out episodes
- **Reward prediction accuracy**: binary accuracy for reward sign
- **Multi-step divergence**: cumulative error over K=10 predicted steps

### Adaptation Speed
- **K-step error curve**: prediction error after K=1,5,10,20,50 inner-loop steps
- **Shots-to-threshold**: episodes needed to reach target prediction quality

### Agent Performance
- **Tiles visited**: unique grid cells in first 100 steps
- **Entity encounters**: count of distinct entity types observed
- **Gear acquisition rate**: fraction of episodes where gear is picked up
- **Junction captures**: aligned junctions per episode
- **Episode reward**: total reward

### Transfer Analysis
- **Representation similarity**: CKA or CCA across variant-adapted models
- **Per-factor adaptation**: which model layers change most during inner loop
- **Variant clustering**: do structurally similar variants cluster in representation space

## Meta-Test Held-Out Variants

Selected to test generalization along each diversity axis:
1. `machina1_n8_medium_default` — hardest combo (large map + many agents + clips pressure)
2. `arena_n8_easy_desert` — different biome + high agent count
3. `arena_n2_medium_caves` — maze terrain + low agent count
4. `arena_n4_easy_city` — structured urban terrain
5. `machina1_n4_easy_default` — large map + moderate agents
6. `arena_n4_medium_forest` — dense terrain + clips

## Success Criteria

1. MAML-init reaches per-variant quality in <10 gradient steps on held-out variants
2. Neural AIF with adapted WM outperforms discrete AIF on held-out variants
3. Epistemic value produces 50%+ more tile coverage in first 100 steps
4. End-to-end pipeline runs: meta-train → adapt → AIF agent acts in new variant
