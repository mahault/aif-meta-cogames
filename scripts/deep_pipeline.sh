#!/bin/bash
set -e

# Deep Planning Pipeline: policy_len=4, 50 episodes, 2000 gradient steps
# This tests whether deeper planning + more data yields better world models

source ~/projects/cogames-env/bin/activate
cd ~/projects/aif-meta-cogames

TRAJ=/tmp/aif_traj_deep.npz
STEPS=2000
TIMEOUT=5000

echo "============================================"
echo "  Deep Pipeline: policy_len=4, 50 episodes"
echo "  Started: $(date)"
echo "============================================"

# Step 1: Collect 50-episode trajectory with policy_len=4
echo ""
echo "=== Step 1: Collecting 50-episode trajectory (policy_len=4) ==="
rm -f "$TRAJ"

for batch in $(seq 1 10); do
    echo "--- Batch $batch/10 (5 episodes each) ---"
    AIF_POLICY_LEN=4 AIF_LOG_TRAJECTORY=1 AIF_TRAJECTORY_PATH="$TRAJ" \
      AIF_LEARNED_PARAMS=~/projects/aif-meta-cogames/params/learned_B5.npz \
      cogames eval -m tutorial -v no_clips \
      -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy \
      -c 4 -e 5 --action-timeout-ms "$TIMEOUT" 2>&1 | tail -5
done

# Check trajectory size
python3 -c "
import numpy as np
d = np.load('$TRAJ')
n = int(d['n_steps'])
print('Trajectory collected:', n, 'steps')
"

# Step 2: Learn with deep trajectory
echo ""
echo "=== Step 2a: Joint A+B+C (2000 steps) ==="
python3 scripts/learn_parameters.py learn-full \
  --trajectory "$TRAJ" --steps "$STEPS" --lr 0.001 \
  --c-lr-scale 0.1 --a-weight 1.0 --b-weight 1.0 --c-weight 0.5 \
  --output /tmp/learned_joint_deep.npz

echo ""
echo "=== Step 2b: A-only (2000 steps) ==="
python3 scripts/learn_parameters.py learn \
  --trajectory "$TRAJ" --steps "$STEPS" --lr 0.001 \
  --output /tmp/learned_aonly_deep.npz

echo ""
echo "=== Step 2c: De novo + BMR ==="
python3 scripts/denovo_learn.py learn \
  --trajectory "$TRAJ" --prior-scale 0.1 \
  --bmr --bmr-threshold 3.0 \
  --output /tmp/learned_denovo_deep.npz

echo ""
echo "=== Step 2d: Differentiable BMR (prune) ==="
python3 scripts/differentiable_bmr.py prune \
  --trajectory "$TRAJ" --rounds 3 \
  --prune-percentile 10.0 --refine-steps 200 \
  --output /tmp/learned_bmr_deep.npz

echo ""
echo "=== Step 2e: De novo -> gradient refinement ==="
python3 scripts/differentiable_bmr.py refine \
  --trajectory "$TRAJ" \
  --init-params /tmp/learned_denovo_deep.npz --steps 500 \
  --output /tmp/learned_refined_deep.npz

# Step 3: VFE comparison
echo ""
echo "=== Step 3: Model comparison ==="
python3 scripts/differentiable_bmr.py compare \
  --trajectory "$TRAJ" \
  --params /tmp/learned_joint_deep.npz /tmp/learned_aonly_deep.npz \
           /tmp/learned_denovo_deep.npz /tmp/learned_bmr_deep.npz \
           /tmp/learned_refined_deep.npz

# Step 4: Live eval - policy_len=4 with each approach
echo ""
echo "=== Step 4: Live eval (policy_len=4) ==="
for name in joint_deep aonly_deep denovo_deep bmr_deep refined_deep; do
    echo "--- Eval: ${name} ---"
    AIF_POLICY_LEN=4 AIF_LEARNED_PARAMS=/tmp/learned_${name}.npz \
      cogames eval -m tutorial -v no_clips \
      -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy \
      -c 4 -e 5 --action-timeout-ms "$TIMEOUT" 2>&1 | tail -20
done

# Step 5: Baseline comparisons
echo ""
echo "=== Step 5a: B5 baseline (policy_len=2) ==="
AIF_POLICY_LEN=2 AIF_LEARNED_PARAMS=~/projects/aif-meta-cogames/params/learned_B5.npz \
  cogames eval -m tutorial -v no_clips \
  -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy \
  -c 4 -e 5 --action-timeout-ms 1000 2>&1 | tail -20

echo ""
echo "=== Step 5b: B5 baseline (policy_len=4) ==="
AIF_POLICY_LEN=4 AIF_LEARNED_PARAMS=~/projects/aif-meta-cogames/params/learned_B5.npz \
  cogames eval -m tutorial -v no_clips \
  -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy \
  -c 4 -e 5 --action-timeout-ms "$TIMEOUT" 2>&1 | tail -20

# Step 6: Best approach on arena (if tutorial works)
echo ""
echo "=== Step 6: Best approach on arena (joint_deep, policy_len=4) ==="
AIF_POLICY_LEN=4 AIF_LEARNED_PARAMS=/tmp/learned_joint_deep.npz \
  cogames eval -m arena -v no_clips \
  -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy \
  -c 4 -e 5 --action-timeout-ms "$TIMEOUT" 2>&1 | tail -20

echo ""
echo "============================================"
echo "  Deep Pipeline complete: $(date)"
echo "============================================"
