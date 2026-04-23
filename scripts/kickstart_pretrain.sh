#!/bin/bash
# Kickstart Pre-training: AIF agents learn alongside a competent teacher
#
# Runs mixed-team games (2 LSTM teacher + 2 AIF agents) so the AIF agents
# observe a functioning economy. Then learns A/B/C parameters from these
# richer trajectories. The resulting params are used as init for all-AIF training.
#
# Usage: nohup /tmp/kickstart_pretrain.sh > /tmp/kickstart.log 2>&1 &

source ~/projects/cogames-env/bin/activate
cd ~/projects/aif-meta-cogames

export PYTHONPATH="$HOME/projects/aif-meta-cogames/scripts:${PYTHONPATH:-}"
pip install --quiet --no-deps -e ~/projects/aif-meta-cogames

LOG=/tmp/kickstart_pretrain.log
POLICY_CLASS="class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy"
# Teacher: leaderboard policy (preferred) or local checkpoint (fallback)
# Override via: TEACHER="metta://policy/Paz-Bot-9000:v47" ./kickstart_pretrain.sh
TEACHER="${TEACHER:-$HOME/train_dir/177507582855}"

# Verify teacher exists (skip check for metta:// URIs — resolved at runtime)
if [[ "$TEACHER" != metta://* ]] && [ ! -d "$TEACHER" ]; then
    echo "ERROR: Teacher checkpoint not found at $TEACHER" | tee $LOG
    exit 1
fi

GRAD_STEPS=5000
LR=0.001
C_LR_SCALE=0.5
MAP=arena
AGENTS=4
STEPS_PER_EP=10000
TIMEOUT=1000
POLICY_LEN=4

echo "============================================================" | tee $LOG
echo "  KICKSTART PRE-TRAINING" | tee -a $LOG
echo "  Teacher: LSTMPolicy (${TEACHER})" | tee -a $LOG
echo "  Mixed teams: 2 teacher + 2 AIF per game" | tee -a $LOG
echo "  3 rounds of 15 episodes each (parallel GPUs 0+3)" | tee -a $LOG
echo "  Started: $(date)" | tee -a $LOG
echo "============================================================" | tee -a $LOG

# Start with default params (no prior learning)
CURRENT_PARAMS=""

for ROUND in 1 2 3; do
    echo "" | tee -a $LOG
    echo "############################################################" | tee -a $LOG
    echo "  KICKSTART ROUND ${ROUND}/3" | tee -a $LOG
    if [ -n "$CURRENT_PARAMS" ]; then
        echo "  Using params: ${CURRENT_PARAMS}" | tee -a $LOG
    else
        echo "  Using params: default (no prior)" | tee -a $LOG
    fi
    echo "  Time: $(date)" | tee -a $LOG
    echo "############################################################" | tee -a $LOG

    TRAJ="/tmp/aif_kickstart_r${ROUND}.npz"
    TRAJ_GPU1="/tmp/aif_kickstart_r${ROUND}_gpu1.npz"
    TRAJ_GPU2="/tmp/aif_kickstart_r${ROUND}_gpu2.npz"
    OUTPUT="/tmp/learned_kickstart_r${ROUND}.npz"
    rm -f "$TRAJ" "$TRAJ_GPU1" "$TRAJ_GPU2"

    # CURRENT_PARAMS is exported inside subshells if non-empty

    # -----------------------------------------------------------
    # Step 1: Mixed-team collection (parallel GPUs 1+2)
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== KS-R${ROUND} Step 1: Mixed-team collection (2 teacher + 2 AIF) ===" | tee -a $LOG
    echo "  GPU 0: 8 episodes → ${TRAJ_GPU1}" | tee -a $LOG
    echo "  GPU 3: 7 episodes → ${TRAJ_GPU2}" | tee -a $LOG

    # GPU 0: 8 episodes, mixed team
    # Limit JAX memory to 40% so PyTorch teacher can coexist
    (
      export CUDA_VISIBLE_DEVICES=0
      export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
      export AIF_POLICY_LEN=${POLICY_LEN}
      export AIF_LOG_TRAJECTORY=1
      export AIF_TRAJECTORY_PATH="$TRAJ_GPU1"
      export AIF_EXPLORE_E=1
      export AIF_NOVELTY_WEIGHT=1.0
      export AIF_LEARN_INTERVAL=20
      if [ -n "$CURRENT_PARAMS" ]; then
        export AIF_LEARNED_PARAMS="$CURRENT_PARAMS"
      fi
      cogames eval -m ${MAP} -v no_clips \
        -p "${TEACHER},proportion=1" \
        -p "${POLICY_CLASS},proportion=1" \
        -c ${AGENTS} -e 8 -s ${STEPS_PER_EP} \
        --action-timeout-ms ${TIMEOUT}
    ) > /tmp/ks_gpu1.log 2>&1 &
    PID1=$!

    # GPU 3: 7 episodes, mixed team
    (
      export CUDA_VISIBLE_DEVICES=3
      export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
      export AIF_POLICY_LEN=${POLICY_LEN}
      export AIF_LOG_TRAJECTORY=1
      export AIF_TRAJECTORY_PATH="$TRAJ_GPU2"
      export AIF_EXPLORE_E=1
      export AIF_NOVELTY_WEIGHT=1.0
      export AIF_LEARN_INTERVAL=20
      if [ -n "$CURRENT_PARAMS" ]; then
        export AIF_LEARNED_PARAMS="$CURRENT_PARAMS"
      fi
      cogames eval -m ${MAP} -v no_clips \
        -p "${TEACHER},proportion=1" \
        -p "${POLICY_CLASS},proportion=1" \
        -c ${AGENTS} -e 7 -s ${STEPS_PER_EP} \
        --action-timeout-ms ${TIMEOUT}
    ) > /tmp/ks_gpu2.log 2>&1 &
    PID2=$!

    echo "  PIDs: GPU1=${PID1} GPU2=${PID2}" | tee -a $LOG
    echo "  Waiting for parallel collection..." | tee -a $LOG

    FAIL=0
    wait $PID1 || FAIL=1
    wait $PID2 || FAIL=1

    echo "--- GPU 1 log ---" >> $LOG
    cat /tmp/ks_gpu1.log >> $LOG 2>/dev/null
    echo "--- GPU 2 log ---" >> $LOG
    cat /tmp/ks_gpu2.log >> $LOG 2>/dev/null

    if [ $FAIL -ne 0 ]; then
        echo "ERROR: Collection failure on one or both GPUs" | tee -a $LOG
        echo "--- GPU 0 log ---" | tee -a $LOG
        cat /tmp/ks_gpu1.log 2>/dev/null | tee -a $LOG
        echo "--- GPU 3 log ---" | tee -a $LOG
        cat /tmp/ks_gpu2.log 2>/dev/null | tee -a $LOG
        exit 1
    fi

    # Merge trajectories
    echo "  Merging trajectory files..." | tee -a $LOG
    python3 -c "
import numpy as np, shutil, os, sys
sys.path.insert(0, '$HOME/projects/aif-meta-cogames/scripts')
from learn_parameters import accumulate_trajectory, save_trajectory_batched

f1 = '${TRAJ_GPU1}'
f2 = '${TRAJ_GPU2}'
out = '${TRAJ}'

if os.path.exists(f1) and os.path.exists(f2):
    shutil.copy(f1, out)
    data2 = dict(np.load(f2, allow_pickle=True))
    merged = accumulate_trajectory(out, data2)
    save_trajectory_batched(out, merged)
elif os.path.exists(f1):
    shutil.copy(f1, out)
elif os.path.exists(f2):
    shutil.copy(f2, out)
else:
    print('[merge] ERROR: No trajectory files found!')
    sys.exit(1)
" 2>&1 | tee -a $LOG

    # Keep GPU logs for debugging (don't delete)
    # rm -f "$TRAJ_GPU1" "$TRAJ_GPU2" /tmp/ks_gpu1.log /tmp/ks_gpu2.log

    python3 -c "
import numpy as np
d = np.load('${TRAJ}')
n = int(d['n_steps'])
print(f'KS-R${ROUND} trajectory: {n} steps (merged)')
" 2>&1 | tee -a $LOG

    # -----------------------------------------------------------
    # Step 2: Learn A/B/C from mixed-team trajectories
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== KS-R${ROUND} Step 2: Joint A+B+C learning (${GRAD_STEPS} steps) ===" | tee -a $LOG

    # Note: --n-agents 2 because only 2 of the 4 agents are AIF
    # The trajectory only contains AIF agent data (teacher doesn't log)
    CUDA_VISIBLE_DEVICES=0 \
    python3 scripts/learn_parameters.py learn-full \
      --trajectory "$TRAJ" \
      --steps ${GRAD_STEPS} \
      --lr ${LR} \
      --c-lr-scale ${C_LR_SCALE} \
      --a-weight 1.0 --b-weight 1.0 --c-weight 1.0 \
      --n-agents 2 \
      --output "$OUTPUT" 2>&1 | tee -a $LOG

    # -----------------------------------------------------------
    # Step 3: Quick eval (all-AIF, to see if kickstarted params help)
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== KS-R${ROUND} Step 3: All-AIF eval with kickstarted params ===" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=0 \
      AIF_POLICY_LEN=${POLICY_LEN} \
      AIF_LEARNED_PARAMS="${OUTPUT}" \
      AIF_NOVELTY_WEIGHT=1.0 \
      AIF_LEARN_INTERVAL=20 \
      cogames eval -m ${MAP} -v no_clips \
      -p ${POLICY_CLASS} \
      -c ${AGENTS} -e 5 -s ${STEPS_PER_EP} \
      --action-timeout-ms ${TIMEOUT} 2>&1 | tee -a $LOG | tail -20

    # -----------------------------------------------------------
    # Step 4: VFE comparison vs default
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== KS-R${ROUND} Step 4: VFE comparison ===" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=0 \
    python3 scripts/differentiable_bmr.py compare \
      --trajectory "$TRAJ" \
      --params "$OUTPUT" \
      --n-agents 2 2>&1 | tee -a $LOG

    CURRENT_PARAMS="$OUTPUT"

    echo "" | tee -a $LOG
    echo "KS-R${ROUND} complete at $(date)" | tee -a $LOG
done

echo "" | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo "  KICKSTART PRE-TRAINING COMPLETE" | tee -a $LOG
echo "  Finished: $(date)" | tee -a $LOG
echo "  Final kickstarted params: /tmp/learned_kickstart_r3.npz" | tee -a $LOG
echo "  Use as init: AIF_LEARNED_PARAMS=/tmp/learned_kickstart_r3.npz" | tee -a $LOG
echo "============================================================" | tee -a $LOG
