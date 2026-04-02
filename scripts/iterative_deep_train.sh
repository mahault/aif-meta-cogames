#!/bin/bash
set -e

# Iterative Deep Training Pipeline (no_clips)
# Based on analysis: need longer episodes, iterative improvement, higher C lr
#
# This pipeline:
#   1. Arena no_clips with 10,000 steps/episode (full economy chains)
#   2. Iterative collect->learn->collect with improving model
#   3. Higher C learning rate (c_lr_scale=0.5 instead of 0.1)
#   4. Starts from the best model (learned_joint_deep) not defaults
#   5. 5 rounds x ~500k steps = ~2.5M total steps
#
# The goal: train MINE_CYCLE properly (currently partially learned),
# start exercising CRAFT_CYCLE through longer episodes, and shift
# C vectors with stronger gradient signal.

source ~/projects/cogames-env/bin/activate
cd ~/projects/aif-meta-cogames

MAP=arena
AGENTS=4
STEPS_PER_EP=10000
TIMEOUT=1000
POLICY_LEN=2
GRAD_STEPS=3000
C_LR_SCALE=0.5

# Start from best existing model
CURRENT_PARAMS=/tmp/learned_joint_deep.npz
TRAJ_BASE=/tmp/aif_iter

echo "============================================================"
echo "  ITERATIVE DEEP TRAINING PIPELINE"
echo "  Map: ${MAP} no_clips"
echo "  Steps/ep: ${STEPS_PER_EP}, Agents: ${AGENTS}"
echo "  Grad steps: ${GRAD_STEPS}, C_lr_scale: ${C_LR_SCALE}"
echo "  Starting from: ${CURRENT_PARAMS}"
echo "  Started: $(date)"
echo "============================================================"

for ROUND in 1 2 3 4 5; do
    echo ""
    echo "############################################################"
    echo "  ROUND ${ROUND}/5"
    echo "  Using params: ${CURRENT_PARAMS}"
    echo "  Time: $(date)"
    echo "############################################################"

    TRAJ="${TRAJ_BASE}_r${ROUND}.npz"
    OUTPUT="/tmp/learned_iter_r${ROUND}.npz"
    rm -f "$TRAJ"

    # -----------------------------------------------------------
    # Step 1: Collect trajectory with current model
    # -----------------------------------------------------------
    echo ""
    echo "=== Round ${ROUND} Step 1: Collecting trajectory ==="

    # 10 batches x 5 episodes x 10000 steps = 500k steps per round
    for batch in $(seq 1 10); do
        echo "--- Batch ${batch}/10 ---"
        AIF_POLICY_LEN=${POLICY_LEN} \
          AIF_LOG_TRAJECTORY=1 \
          AIF_TRAJECTORY_PATH="$TRAJ" \
          AIF_LEARNED_PARAMS="${CURRENT_PARAMS}" \
          cogames eval -m ${MAP} -v no_clips \
          -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy \
          -c ${AGENTS} -e 5 -s ${STEPS_PER_EP} \
          --action-timeout-ms ${TIMEOUT} 2>&1 | tail -8
    done

    # Check trajectory size
    python3 -c "
import numpy as np
d = np.load('${TRAJ}')
n = int(d['n_steps'])
print('Round ${ROUND} trajectory:', n, 'steps')
"

    # -----------------------------------------------------------
    # Step 2: Learn from trajectory (joint A+B+C)
    # -----------------------------------------------------------
    echo ""
    echo "=== Round ${ROUND} Step 2: Joint A+B+C learning (${GRAD_STEPS} steps) ==="
    python3 scripts/learn_parameters.py learn-full \
      --trajectory "$TRAJ" \
      --steps ${GRAD_STEPS} \
      --lr 0.001 \
      --c-lr-scale ${C_LR_SCALE} \
      --a-weight 1.0 --b-weight 1.0 --c-weight 1.0 \
      --output "$OUTPUT"

    # -----------------------------------------------------------
    # Step 3: Quick eval comparison
    # -----------------------------------------------------------
    echo ""
    echo "=== Round ${ROUND} Step 3: Quick eval ==="

    echo "--- Current (round ${ROUND}) ---"
    AIF_POLICY_LEN=${POLICY_LEN} \
      AIF_LEARNED_PARAMS="${OUTPUT}" \
      cogames eval -m ${MAP} -v no_clips \
      -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy \
      -c ${AGENTS} -e 5 -s ${STEPS_PER_EP} \
      --action-timeout-ms ${TIMEOUT} 2>&1 | tail -15

    if [ ${ROUND} -gt 1 ]; then
        PREV="/tmp/learned_iter_r$((ROUND-1)).npz"
        echo "--- Previous (round $((ROUND-1))) ---"
        AIF_POLICY_LEN=${POLICY_LEN} \
          AIF_LEARNED_PARAMS="${PREV}" \
          cogames eval -m ${MAP} -v no_clips \
          -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy \
          -c ${AGENTS} -e 5 -s ${STEPS_PER_EP} \
          --action-timeout-ms ${TIMEOUT} 2>&1 | tail -15
    fi

    # -----------------------------------------------------------
    # Step 4: VFE comparison
    # -----------------------------------------------------------
    echo ""
    echo "=== Round ${ROUND} Step 4: VFE comparison ==="
    python3 scripts/differentiable_bmr.py compare \
      --trajectory "$TRAJ" \
      --params "$OUTPUT"

    # Update for next round
    CURRENT_PARAMS="$OUTPUT"

    echo ""
    echo "Round ${ROUND} complete at $(date)"
done

# -----------------------------------------------------------
# Final comparison: all rounds
# -----------------------------------------------------------
echo ""
echo "============================================================"
echo "  FINAL COMPARISON: All Rounds"
echo "============================================================"
PARAMS_LIST=""
for r in 1 2 3 4 5; do
    f="/tmp/learned_iter_r${r}.npz"
    if [ -f "$f" ]; then
        PARAMS_LIST="${PARAMS_LIST} $f"
    fi
done
PARAMS_LIST="/tmp/learned_joint_deep.npz ${PARAMS_LIST}"

python3 scripts/differentiable_bmr.py compare \
  --trajectory "${TRAJ_BASE}_r5.npz" \
  --params ${PARAMS_LIST}

echo ""
echo "============================================================"
echo "  FINAL EVAL: Best model on arena no_clips"
echo "============================================================"
AIF_POLICY_LEN=${POLICY_LEN} \
  AIF_LEARNED_PARAMS="/tmp/learned_iter_r5.npz" \
  cogames eval -m arena -v no_clips \
  -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy \
  -c ${AGENTS} -e 10 -s ${STEPS_PER_EP} \
  --action-timeout-ms ${TIMEOUT} 2>&1 | tail -30

echo ""
echo "============================================================"
echo "  Pipeline complete: $(date)"
echo "  Total: 5 rounds x ~500k steps = ~2.5M steps"
echo "============================================================"
