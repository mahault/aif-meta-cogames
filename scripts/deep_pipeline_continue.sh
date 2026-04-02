#!/bin/bash
set -e

# Deep Pipeline continuation: Steps 3-6
# (Steps 1-2 already done — trajectory + learning complete)

source ~/projects/cogames-env/bin/activate
cd ~/projects/aif-meta-cogames

TIMEOUT=5000

echo "============================================"
echo "  Deep Pipeline (continued): Steps 3-6"
echo "  Started: $(date)"
echo "============================================"

# Step 3: Model comparison (now with subsampling fix)
echo ""
echo "=== Step 3: Model comparison ==="
python3 scripts/differentiable_bmr.py compare \
  --trajectory /tmp/aif_traj_deep.npz \
  --params /tmp/learned_joint_deep.npz /tmp/learned_aonly_deep.npz \
           /tmp/learned_denovo_deep.npz /tmp/learned_bmr_deep.npz \
           /tmp/learned_refined_deep.npz

# Step 4: Live eval - policy_len=4 with each approach
echo ""
echo "=== Step 4: Live eval (policy_len=4) ==="
for name in joint_deep aonly_deep bmr_deep refined_deep; do
    echo "--- Eval: ${name} ---"
    AIF_POLICY_LEN=4 AIF_LEARNED_PARAMS=/tmp/learned_${name}.npz \
      cogames eval -m tutorial -v no_clips \
      -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy \
      -c 4 -e 5 --action-timeout-ms "$TIMEOUT" 2>&1 | tail -25
done

# Step 5: Baseline comparisons
echo ""
echo "=== Step 5a: B5 baseline (policy_len=2) ==="
AIF_POLICY_LEN=2 AIF_LEARNED_PARAMS=~/projects/aif-meta-cogames/params/learned_B5.npz \
  cogames eval -m tutorial -v no_clips \
  -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy \
  -c 4 -e 5 --action-timeout-ms 1000 2>&1 | tail -25

echo ""
echo "=== Step 5b: B5 baseline (policy_len=4) ==="
AIF_POLICY_LEN=4 AIF_LEARNED_PARAMS=~/projects/aif-meta-cogames/params/learned_B5.npz \
  cogames eval -m tutorial -v no_clips \
  -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy \
  -c 4 -e 5 --action-timeout-ms "$TIMEOUT" 2>&1 | tail -25

# Step 6: Best approach on arena
echo ""
echo "=== Step 6: Best approach on arena (joint_deep, policy_len=4) ==="
AIF_POLICY_LEN=4 AIF_LEARNED_PARAMS=/tmp/learned_joint_deep.npz \
  cogames eval -m arena -v no_clips \
  -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy \
  -c 4 -e 5 --action-timeout-ms "$TIMEOUT" 2>&1 | tail -25

echo ""
echo "============================================"
echo "  Deep Pipeline complete: $(date)"
echo "============================================"
