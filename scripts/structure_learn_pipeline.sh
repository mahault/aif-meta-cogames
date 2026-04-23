#!/bin/bash
# Structure Learning Pipeline: Teacher-Conditioned BMR + Spare Capacity
#
# Iterative pipeline per round:
#   1. Kickstart with teacher (record both agent types)
#   2. Learn C from teacher via inverse EFE
#   3. Expand model with spare capacity
#   4. Learn A/B/C parameters (gradient descent)
#   5. BMR prune (Neacsu 2022 analytical) — remove states with DF > 0
#   6. Eval and compare VFE
#
# 3 rounds, each building on previous.
#
# Usage: nohup bash scripts/structure_learn_pipeline.sh > /tmp/structure_learn.log 2>&1 &

set -euo pipefail

source ~/projects/cogames-env/bin/activate
cd ~/projects/aif-meta-cogames

export PYTHONPATH="$HOME/projects/aif-meta-cogames/scripts:${PYTHONPATH:-}"
pip install --quiet --no-deps -e ~/projects/aif-meta-cogames

LOG=/tmp/structure_learn.log
POLICY_CLASS="class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy"
RECORDING_TEACHER="class=aif_meta_cogames.aif_agent.cogames_policy.RecordingTeacherPolicy"

# Teacher policy: use leaderboard policy if available, else local checkpoint
# Set TEACHER env var before running, or use default
TEACHER="${TEACHER:-metta://policy/Paz-Bot-9000:v47}"

# Pipeline parameters
GRAD_STEPS=5000
LR=0.001
C_LR_SCALE=0.5
MAP=arena
AGENTS=4
STEPS_PER_EP=10000
TIMEOUT=1000
POLICY_LEN=4
N_ROUNDS=3
GPU_COLLECT=0
GPU_LEARN=0

echo "============================================================" | tee $LOG
echo "  STRUCTURE LEARNING PIPELINE" | tee -a $LOG
echo "  Teacher: ${TEACHER}" | tee -a $LOG
echo "  ${N_ROUNDS} rounds of kickstart → C-learn → expand → learn → BMR" | tee -a $LOG
echo "  Started: $(date)" | tee -a $LOG
echo "============================================================" | tee -a $LOG

# Start with best kickstart params if available, else default
CURRENT_PARAMS="${AIF_INIT_PARAMS:-/tmp/learned_kickstart_r3.npz}"
if [ ! -f "$CURRENT_PARAMS" ]; then
    echo "  No init params at $CURRENT_PARAMS — starting from default" | tee -a $LOG
    CURRENT_PARAMS=""
fi

for ROUND in $(seq 1 $N_ROUNDS); do
    echo "" | tee -a $LOG
    echo "############################################################" | tee -a $LOG
    echo "  STRUCTURE ROUND ${ROUND}/${N_ROUNDS}" | tee -a $LOG
    echo "  Time: $(date)" | tee -a $LOG
    echo "############################################################" | tee -a $LOG

    AIF_TRAJ="/tmp/sl_aif_r${ROUND}.npz"
    TEACHER_TRAJ="/tmp/sl_teacher_r${ROUND}.npz"
    LEARNED_C="/tmp/sl_learned_C_r${ROUND}.npz"
    LEARNED_PARAMS="/tmp/sl_learned_r${ROUND}.npz"
    BMR_PARAMS="/tmp/sl_bmr_r${ROUND}.npz"
    rm -f "$AIF_TRAJ" "$TEACHER_TRAJ" "$LEARNED_C" "$LEARNED_PARAMS" "$BMR_PARAMS"

    # -----------------------------------------------------------
    # Step 1: Mixed-team collection with teacher recording
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== SL-R${ROUND} Step 1: Mixed-team collection (recording both) ===" | tee -a $LOG

    (
      export CUDA_VISIBLE_DEVICES=${GPU_COLLECT}
      export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
      export AIF_POLICY_LEN=${POLICY_LEN}
      export AIF_LOG_TRAJECTORY=1
      export AIF_TRAJECTORY_PATH="$AIF_TRAJ"
      export AIF_TEACHER_TRAJECTORY_PATH="$TEACHER_TRAJ"
      export AIF_EXPLORE_E=1
      export AIF_NOVELTY_WEIGHT=1.0
      export AIF_LEARN_INTERVAL=20
      if [ -n "$CURRENT_PARAMS" ]; then
        export AIF_LEARNED_PARAMS="$CURRENT_PARAMS"
      fi
      cogames eval -m ${MAP} -v no_clips \
        -p "${RECORDING_TEACHER},kw.inner=${TEACHER},proportion=1" \
        -p "${POLICY_CLASS},proportion=1" \
        -c ${AGENTS} -e 15 -s ${STEPS_PER_EP} \
        --action-timeout-ms ${TIMEOUT}
    ) 2>&1 | tee -a $LOG

    # Verify files
    for F in "$AIF_TRAJ" "$TEACHER_TRAJ"; do
        if [ -f "$F" ]; then
            python3 -c "
import numpy as np
d = np.load('${F}', allow_pickle=True)
keys = list(d.keys())
print(f'  ${F}: {len(keys)} arrays, keys={keys[:5]}...')
" 2>&1 | tee -a $LOG
        else
            echo "  WARNING: $F not found" | tee -a $LOG
        fi
    done

    # -----------------------------------------------------------
    # Step 2: Learn C vectors from teacher (inverse EFE)
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== SL-R${ROUND} Step 2: Learn C from teacher ===" | tee -a $LOG

    if [ -f "$TEACHER_TRAJ" ]; then
        CUDA_VISIBLE_DEVICES=${GPU_LEARN} \
        python3 scripts/learn_from_teacher.py \
          --teacher-trajectory "$TEACHER_TRAJ" \
          --output "$LEARNED_C" \
          --reward-threshold 3.0 \
          --smoothing 0.1 2>&1 | tee -a $LOG
    else
        echo "  Skipping C learning — no teacher trajectory" | tee -a $LOG
    fi

    # -----------------------------------------------------------
    # Step 3: Learn A/B/C parameters (gradient descent)
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== SL-R${ROUND} Step 3: Joint A+B+C learning (${GRAD_STEPS} steps) ===" | tee -a $LOG

    CUDA_VISIBLE_DEVICES=${GPU_LEARN} \
    python3 scripts/learn_parameters.py learn-full \
      --trajectory "$AIF_TRAJ" \
      --steps ${GRAD_STEPS} \
      --lr ${LR} \
      --c-lr-scale ${C_LR_SCALE} \
      --a-weight 1.0 --b-weight 1.0 --c-weight 1.0 \
      --n-agents 2 \
      --output "$LEARNED_PARAMS" 2>&1 | tee -a $LOG

    # -----------------------------------------------------------
    # Step 4: BMR prune (Neacsu 2022 analytical)
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== SL-R${ROUND} Step 4: Analytical BMR ===" | tee -a $LOG

    CUDA_VISIBLE_DEVICES=${GPU_LEARN} \
    python3 scripts/differentiable_bmr.py bmr-analytical \
      --trajectory "$AIF_TRAJ" \
      --params "$LEARNED_PARAMS" \
      --output "$BMR_PARAMS" \
      --max-merges 10 \
      --n-agents 2 2>&1 | tee -a $LOG

    # -----------------------------------------------------------
    # Step 5: Eval and compare VFE
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== SL-R${ROUND} Step 5: VFE comparison ===" | tee -a $LOG

    # Compare learned params only (BMR has different dims, can't be compared
    # against same trajectory beliefs without re-inference)
    CUDA_VISIBLE_DEVICES=${GPU_LEARN} \
    python3 scripts/differentiable_bmr.py compare \
      --trajectory "$AIF_TRAJ" \
      --params "$LEARNED_PARAMS" \
      --n-agents 2 2>&1 | tee -a $LOG

    # -----------------------------------------------------------
    # Step 6: Quick eval (all-AIF with BMR params)
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== SL-R${ROUND} Step 6: All-AIF eval with BMR params ===" | tee -a $LOG

    # BMR changes A dimensions — can't load into live agent yet.
    # Use learned (non-BMR) params for eval. BMR results are diagnostic only.
    BEST_PARAMS="$LEARNED_PARAMS"

    CUDA_VISIBLE_DEVICES=${GPU_LEARN} \
      AIF_POLICY_LEN=${POLICY_LEN} \
      AIF_LEARNED_PARAMS="${BEST_PARAMS}" \
      AIF_NOVELTY_WEIGHT=1.0 \
      AIF_LEARN_INTERVAL=20 \
      cogames eval -m ${MAP} -v no_clips \
      -p ${POLICY_CLASS} \
      -c ${AGENTS} -e 5 -s ${STEPS_PER_EP} \
      --action-timeout-ms ${TIMEOUT} 2>&1 | tee -a $LOG | tail -20

    CURRENT_PARAMS="$BEST_PARAMS"

    echo "" | tee -a $LOG
    echo "SL-R${ROUND} complete at $(date)" | tee -a $LOG
done

echo "" | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo "  STRUCTURE LEARNING PIPELINE COMPLETE" | tee -a $LOG
echo "  Finished: $(date)" | tee -a $LOG
echo "  Final params: /tmp/sl_bmr_r${N_ROUNDS}.npz" | tee -a $LOG
echo "============================================================" | tee -a $LOG
