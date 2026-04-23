#!/bin/bash
# Extended Training Pipeline v2: Parallel GPU Collection
#
# Changes from v1:
#   - 15 episodes (was 50) split across GPUs 1+2 in parallel
#   - Each GPU does one cogames eval call → single trajectory save
#   - ~2-3h per round instead of ~15h
#   - Parallel eval (current + previous on separate GPUs)
#
# Continues from R3 (R1-R3 completed by v1)
# Phase A: rounds 4-10 on arena (4 agents, 10k steps/ep)
# Phase B: 5 rounds on machina_1 (8 agents, 20k steps/ep)

source ~/projects/cogames-env/bin/activate
cd ~/projects/aif-meta-cogames

export PYTHONPATH="$HOME/projects/aif-meta-cogames/scripts:${PYTHONPATH:-}"
pip install --quiet --no-deps -e ~/projects/aif-meta-cogames

LOG=/tmp/extended_train_v2.log
POLICY_CLASS="class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy"

GRAD_STEPS=5000
LR=0.001
C_LR_SCALE=0.5

echo "============================================================" | tee $LOG
echo "  EXTENDED TRAINING PIPELINE v2 (Parallel GPU)" | tee -a $LOG
echo "  Phase A: Arena rounds 4-10 (4 agents, 15 eps/round)" | tee -a $LOG
echo "  Phase B: machina_1 5 rounds (8 agents, 15 eps/round)" | tee -a $LOG
echo "  GPUs: 1+2 parallel collection" | tee -a $LOG
echo "  Started: $(date)" | tee -a $LOG
echo "============================================================" | tee -a $LOG

# ==============================================================
#  PHASE A: Arena Extended (rounds 4-10)
# ==============================================================

CURRENT_PARAMS=/tmp/learned_ext_arena_r3.npz
MAP=arena
AGENTS=4
STEPS_PER_EP=10000
TIMEOUT=1000
POLICY_LEN=4
TRAJ_BASE=/tmp/aif_ext_arena

echo "" | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo "  PHASE A: Arena Extended Training (rounds 4-10)" | tee -a $LOG
echo "  Map: ${MAP} no_clips, Agents: ${AGENTS}" | tee -a $LOG
echo "  Steps/ep: ${STEPS_PER_EP}, Policy len: ${POLICY_LEN}" | tee -a $LOG
echo "  Init: ${CURRENT_PARAMS}" | tee -a $LOG
echo "============================================================" | tee -a $LOG

# Wait for R3 params if not yet available
while [ ! -f "$CURRENT_PARAMS" ]; do
    echo "Waiting for ${CURRENT_PARAMS} (R3 still in progress)..." | tee -a $LOG
    sleep 300
done

for ROUND in 4 5 6 7 8 9 10; do
    echo "" | tee -a $LOG
    echo "############################################################" | tee -a $LOG
    echo "  PHASE A — ROUND ${ROUND}/10" | tee -a $LOG
    echo "  Using params: ${CURRENT_PARAMS}" | tee -a $LOG
    echo "  Time: $(date)" | tee -a $LOG
    echo "############################################################" | tee -a $LOG

    TRAJ="${TRAJ_BASE}_r${ROUND}.npz"
    OUTPUT="/tmp/learned_ext_arena_r${ROUND}.npz"
    TRAJ_GPU1="${TRAJ_BASE}_r${ROUND}_gpu1.npz"
    TRAJ_GPU2="${TRAJ_BASE}_r${ROUND}_gpu2.npz"
    rm -f "$TRAJ" "$TRAJ_GPU1" "$TRAJ_GPU2"

    # -----------------------------------------------------------
    # Step 1: Parallel trajectory collection (GPUs 1+2)
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== A-R${ROUND} Step 1: Collecting trajectory (parallel GPUs) ===" | tee -a $LOG

    # Exploration schedule: rounds 1-5 explore, rounds 6-10 deploy
    if [ ${ROUND} -le 5 ]; then
        EXPLORE_E=1
        echo "  Exploration mode (flat E)" | tee -a $LOG
    else
        EXPLORE_E=0
        echo "  Deploy mode (strong E)" | tee -a $LOG
    fi

    echo "  GPU 1: 8 episodes (80k steps) → ${TRAJ_GPU1}" | tee -a $LOG
    echo "  GPU 2: 7 episodes (70k steps) → ${TRAJ_GPU2}" | tee -a $LOG

    # GPU 1: 8 episodes
    (CUDA_VISIBLE_DEVICES=1 \
      AIF_POLICY_LEN=${POLICY_LEN} \
      AIF_LOG_TRAJECTORY=1 \
      AIF_TRAJECTORY_PATH="$TRAJ_GPU1" \
      AIF_LEARNED_PARAMS="${CURRENT_PARAMS}" \
      AIF_EXPLORE_E=${EXPLORE_E} \
      AIF_NOVELTY_WEIGHT=1.0 \
      AIF_LEARN_INTERVAL=20 \
      cogames eval -m ${MAP} -v no_clips \
      -p ${POLICY_CLASS} \
      -c ${AGENTS} -e 8 -s ${STEPS_PER_EP} \
      --action-timeout-ms ${TIMEOUT} > /tmp/collect_gpu1.log 2>&1) &
    PID_GPU1=$!

    # GPU 2: 7 episodes
    (CUDA_VISIBLE_DEVICES=2 \
      AIF_POLICY_LEN=${POLICY_LEN} \
      AIF_LOG_TRAJECTORY=1 \
      AIF_TRAJECTORY_PATH="$TRAJ_GPU2" \
      AIF_LEARNED_PARAMS="${CURRENT_PARAMS}" \
      AIF_EXPLORE_E=${EXPLORE_E} \
      AIF_NOVELTY_WEIGHT=1.0 \
      AIF_LEARN_INTERVAL=20 \
      cogames eval -m ${MAP} -v no_clips \
      -p ${POLICY_CLASS} \
      -c ${AGENTS} -e 7 -s ${STEPS_PER_EP} \
      --action-timeout-ms ${TIMEOUT} > /tmp/collect_gpu2.log 2>&1) &
    PID_GPU2=$!

    echo "  PIDs: GPU1=${PID_GPU1} GPU2=${PID_GPU2}" | tee -a $LOG
    echo "  Waiting for parallel collection..." | tee -a $LOG

    # Wait for both — capture exit codes
    FAIL=0
    wait $PID_GPU1 || FAIL=1
    wait $PID_GPU2 || FAIL=1

    # Append collection logs
    echo "--- GPU 1 collection log ---" >> $LOG
    cat /tmp/collect_gpu1.log >> $LOG 2>/dev/null
    echo "--- GPU 2 collection log ---" >> $LOG
    cat /tmp/collect_gpu2.log >> $LOG 2>/dev/null

    if [ $FAIL -ne 0 ]; then
        echo "WARNING: One or both collection processes failed" | tee -a $LOG
    fi

    # Merge trajectories (batched v2 — fast array concat)
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
    print(f'[merge] Only GPU1 trajectory found')
elif os.path.exists(f2):
    shutil.copy(f2, out)
    print(f'[merge] Only GPU2 trajectory found')
else:
    print(f'[merge] ERROR: No trajectory files found!')
    sys.exit(1)
" 2>&1 | tee -a $LOG

    # Clean up per-GPU files
    rm -f "$TRAJ_GPU1" "$TRAJ_GPU2" /tmp/collect_gpu1.log /tmp/collect_gpu2.log

    # Check merged trajectory
    python3 -c "
import numpy as np
d = np.load('${TRAJ}')
n = int(d['n_steps'])
print(f'A-R${ROUND} trajectory: {n} steps (merged)')
" 2>&1 | tee -a $LOG

    # -----------------------------------------------------------
    # Step 2: Learn parameters (joint A+B+C) — GPU 1
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== A-R${ROUND} Step 2: Joint A+B+C learning (${GRAD_STEPS} steps) ===" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=1 \
    python3 scripts/learn_parameters.py learn-full \
      --trajectory "$TRAJ" \
      --steps ${GRAD_STEPS} \
      --lr ${LR} \
      --c-lr-scale ${C_LR_SCALE} \
      --a-weight 1.0 --b-weight 1.0 --c-weight 1.0 \
      --n-agents ${AGENTS} \
      --output "$OUTPUT" 2>&1 | tee -a $LOG

    # -----------------------------------------------------------
    # Step 3: Parallel eval (GPUs 1+2)
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== A-R${ROUND} Step 3: Quick eval (parallel) ===" | tee -a $LOG

    # Current round on GPU 1
    (CUDA_VISIBLE_DEVICES=1 \
      AIF_POLICY_LEN=${POLICY_LEN} \
      AIF_LEARNED_PARAMS="${OUTPUT}" \
      AIF_NOVELTY_WEIGHT=1.0 \
      AIF_LEARN_INTERVAL=20 \
      cogames eval -m ${MAP} -v no_clips \
      -p ${POLICY_CLASS} \
      -c ${AGENTS} -e 5 -s ${STEPS_PER_EP} \
      --action-timeout-ms ${TIMEOUT} > /tmp/eval_curr.log 2>&1) &
    PID_EVAL1=$!

    # Previous round on GPU 2 (if exists)
    PREV="/tmp/learned_ext_arena_r$((ROUND-1)).npz"
    if [ -f "$PREV" ]; then
        (CUDA_VISIBLE_DEVICES=2 \
          AIF_POLICY_LEN=${POLICY_LEN} \
          AIF_LEARNED_PARAMS="${PREV}" \
          AIF_NOVELTY_WEIGHT=1.0 \
          AIF_LEARN_INTERVAL=20 \
          cogames eval -m ${MAP} -v no_clips \
          -p ${POLICY_CLASS} \
          -c ${AGENTS} -e 5 -s ${STEPS_PER_EP} \
          --action-timeout-ms ${TIMEOUT} > /tmp/eval_prev.log 2>&1) &
        PID_EVAL2=$!
        wait $PID_EVAL1 $PID_EVAL2 || true
    else
        wait $PID_EVAL1 || true
    fi

    echo "--- Current (round ${ROUND}) ---" | tee -a $LOG
    cat /tmp/eval_curr.log >> $LOG 2>/dev/null
    tail -15 /tmp/eval_curr.log 2>/dev/null | tee -a /dev/null

    if [ -f "/tmp/eval_prev.log" ]; then
        echo "--- Previous (round $((ROUND-1))) ---" | tee -a $LOG
        cat /tmp/eval_prev.log >> $LOG 2>/dev/null
    fi
    rm -f /tmp/eval_curr.log /tmp/eval_prev.log

    # -----------------------------------------------------------
    # Step 4: VFE comparison — GPU 1
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== A-R${ROUND} Step 4: VFE comparison ===" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=1 \
    python3 scripts/differentiable_bmr.py compare \
      --trajectory "$TRAJ" \
      --params "$OUTPUT" \
      --n-agents ${AGENTS} 2>&1 | tee -a $LOG

    # Update for next round
    CURRENT_PARAMS="$OUTPUT"

    echo "" | tee -a $LOG
    echo "A-R${ROUND} complete at $(date)" | tee -a $LOG
done

# -----------------------------------------------------------
# Phase A Final: Compare all arena rounds
# -----------------------------------------------------------
echo "" | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo "  PHASE A FINAL: All-Round Comparison" | tee -a $LOG
echo "============================================================" | tee -a $LOG

ARENA_PARAMS="/tmp/learned_iter_r2.npz"
for r in 1 2 3 4 5 6 7 8 9 10; do
    f="/tmp/learned_ext_arena_r${r}.npz"
    if [ -f "$f" ]; then
        ARENA_PARAMS="${ARENA_PARAMS} $f"
    fi
done

CUDA_VISIBLE_DEVICES=1 \
python3 scripts/differentiable_bmr.py compare \
  --trajectory "${TRAJ_BASE}_r10.npz" \
  --params ${ARENA_PARAMS} \
  --n-agents ${AGENTS} 2>&1 | tee -a $LOG

# Phase A final eval: 10 episodes
echo "" | tee -a $LOG
echo "=== Phase A Final Eval: 10 episodes ===" | tee -a $LOG
CUDA_VISIBLE_DEVICES=1 \
  AIF_POLICY_LEN=${POLICY_LEN} \
  AIF_LEARNED_PARAMS="/tmp/learned_ext_arena_r10.npz" \
  AIF_NOVELTY_WEIGHT=1.0 \
  AIF_LEARN_INTERVAL=20 \
  cogames eval -m ${MAP} -v no_clips \
  -p ${POLICY_CLASS} \
  -c ${AGENTS} -e 10 -s ${STEPS_PER_EP} \
  --action-timeout-ms ${TIMEOUT} 2>&1 | tee -a $LOG | tail -30

echo "" | tee -a $LOG
echo "Phase A complete at $(date)" | tee -a $LOG

# ==============================================================
#  PHASE B: machina_1 Adaptation (5 rounds)
# ==============================================================

CURRENT_PARAMS=/tmp/learned_ext_arena_r10.npz
MAP=machina_1
AGENTS=8
STEPS_PER_EP=20000
TIMEOUT=1000
POLICY_LEN=4
TRAJ_BASE=/tmp/aif_ext_machina

echo "" | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo "  PHASE B: machina_1 Adaptation" | tee -a $LOG
echo "  Map: ${MAP} no_clips, Agents: ${AGENTS}" | tee -a $LOG
echo "  Steps/ep: ${STEPS_PER_EP}, Policy len: ${POLICY_LEN}" | tee -a $LOG
echo "  Init: ${CURRENT_PARAMS}" | tee -a $LOG
echo "  Collection: 15 episodes/round on GPUs 1+2" | tee -a $LOG
echo "============================================================" | tee -a $LOG

for ROUND in 1 2 3 4 5; do
    echo "" | tee -a $LOG
    echo "############################################################" | tee -a $LOG
    echo "  PHASE B — ROUND ${ROUND}/5" | tee -a $LOG
    echo "  Using params: ${CURRENT_PARAMS}" | tee -a $LOG
    echo "  Time: $(date)" | tee -a $LOG
    echo "############################################################" | tee -a $LOG

    TRAJ="${TRAJ_BASE}_r${ROUND}.npz"
    OUTPUT="/tmp/learned_ext_machina_r${ROUND}.npz"
    TRAJ_GPU1="${TRAJ_BASE}_r${ROUND}_gpu1.npz"
    TRAJ_GPU2="${TRAJ_BASE}_r${ROUND}_gpu2.npz"
    rm -f "$TRAJ" "$TRAJ_GPU1" "$TRAJ_GPU2"

    # -----------------------------------------------------------
    # Step 1: Parallel trajectory collection (GPUs 1+2)
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== B-R${ROUND} Step 1: Collecting trajectory (parallel GPUs) ===" | tee -a $LOG

    if [ ${ROUND} -le 3 ]; then
        EXPLORE_E=1
        echo "  Exploration mode (flat E)" | tee -a $LOG
    else
        EXPLORE_E=0
        echo "  Deploy mode (strong E)" | tee -a $LOG
    fi

    echo "  GPU 1: 8 episodes (160k steps) → ${TRAJ_GPU1}" | tee -a $LOG
    echo "  GPU 2: 7 episodes (140k steps) → ${TRAJ_GPU2}" | tee -a $LOG

    # GPU 1: 8 episodes
    (CUDA_VISIBLE_DEVICES=1 \
      AIF_POLICY_LEN=${POLICY_LEN} \
      AIF_LOG_TRAJECTORY=1 \
      AIF_TRAJECTORY_PATH="$TRAJ_GPU1" \
      AIF_LEARNED_PARAMS="${CURRENT_PARAMS}" \
      AIF_EXPLORE_E=${EXPLORE_E} \
      AIF_NOVELTY_WEIGHT=1.0 \
      AIF_LEARN_INTERVAL=20 \
      cogames eval -m ${MAP} -v no_clips \
      -p ${POLICY_CLASS} \
      -c ${AGENTS} -e 8 -s ${STEPS_PER_EP} \
      --action-timeout-ms ${TIMEOUT} > /tmp/collect_gpu1.log 2>&1) &
    PID_GPU1=$!

    # GPU 2: 7 episodes
    (CUDA_VISIBLE_DEVICES=2 \
      AIF_POLICY_LEN=${POLICY_LEN} \
      AIF_LOG_TRAJECTORY=1 \
      AIF_TRAJECTORY_PATH="$TRAJ_GPU2" \
      AIF_LEARNED_PARAMS="${CURRENT_PARAMS}" \
      AIF_EXPLORE_E=${EXPLORE_E} \
      AIF_NOVELTY_WEIGHT=1.0 \
      AIF_LEARN_INTERVAL=20 \
      cogames eval -m ${MAP} -v no_clips \
      -p ${POLICY_CLASS} \
      -c ${AGENTS} -e 7 -s ${STEPS_PER_EP} \
      --action-timeout-ms ${TIMEOUT} > /tmp/collect_gpu2.log 2>&1) &
    PID_GPU2=$!

    echo "  PIDs: GPU1=${PID_GPU1} GPU2=${PID_GPU2}" | tee -a $LOG
    echo "  Waiting for parallel collection..." | tee -a $LOG

    FAIL=0
    wait $PID_GPU1 || FAIL=1
    wait $PID_GPU2 || FAIL=1

    echo "--- GPU 1 collection log ---" >> $LOG
    cat /tmp/collect_gpu1.log >> $LOG 2>/dev/null
    echo "--- GPU 2 collection log ---" >> $LOG
    cat /tmp/collect_gpu2.log >> $LOG 2>/dev/null

    if [ $FAIL -ne 0 ]; then
        echo "WARNING: One or both collection processes failed" | tee -a $LOG
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

    rm -f "$TRAJ_GPU1" "$TRAJ_GPU2" /tmp/collect_gpu1.log /tmp/collect_gpu2.log

    python3 -c "
import numpy as np
d = np.load('${TRAJ}')
n = int(d['n_steps'])
print(f'B-R${ROUND} trajectory: {n} steps (merged)')
" 2>&1 | tee -a $LOG

    # -----------------------------------------------------------
    # Step 2: Learn parameters (joint A+B+C) — GPU 1
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== B-R${ROUND} Step 2: Joint A+B+C learning (${GRAD_STEPS} steps) ===" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=1 \
    python3 scripts/learn_parameters.py learn-full \
      --trajectory "$TRAJ" \
      --steps ${GRAD_STEPS} \
      --lr ${LR} \
      --c-lr-scale ${C_LR_SCALE} \
      --a-weight 1.0 --b-weight 1.0 --c-weight 1.0 \
      --n-agents ${AGENTS} \
      --output "$OUTPUT" 2>&1 | tee -a $LOG

    # -----------------------------------------------------------
    # Step 3: Parallel eval (GPUs 1+2)
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== B-R${ROUND} Step 3: Quick eval (parallel) ===" | tee -a $LOG

    (CUDA_VISIBLE_DEVICES=1 \
      AIF_POLICY_LEN=${POLICY_LEN} \
      AIF_LEARNED_PARAMS="${OUTPUT}" \
      AIF_NOVELTY_WEIGHT=1.0 \
      AIF_LEARN_INTERVAL=20 \
      cogames eval -m ${MAP} -v no_clips \
      -p ${POLICY_CLASS} \
      -c ${AGENTS} -e 5 -s ${STEPS_PER_EP} \
      --action-timeout-ms ${TIMEOUT} > /tmp/eval_curr.log 2>&1) &
    PID_EVAL1=$!

    PREV="/tmp/learned_ext_machina_r$((ROUND-1)).npz"
    if [ ${ROUND} -gt 1 ] && [ -f "$PREV" ]; then
        (CUDA_VISIBLE_DEVICES=2 \
          AIF_POLICY_LEN=${POLICY_LEN} \
          AIF_LEARNED_PARAMS="${PREV}" \
          AIF_NOVELTY_WEIGHT=1.0 \
          AIF_LEARN_INTERVAL=20 \
          cogames eval -m ${MAP} -v no_clips \
          -p ${POLICY_CLASS} \
          -c ${AGENTS} -e 5 -s ${STEPS_PER_EP} \
          --action-timeout-ms ${TIMEOUT} > /tmp/eval_prev.log 2>&1) &
        PID_EVAL2=$!
        wait $PID_EVAL1 $PID_EVAL2 || true
    else
        wait $PID_EVAL1 || true
    fi

    echo "--- Current (round ${ROUND}) ---" | tee -a $LOG
    cat /tmp/eval_curr.log >> $LOG 2>/dev/null
    if [ -f "/tmp/eval_prev.log" ]; then
        echo "--- Previous (round $((ROUND-1))) ---" | tee -a $LOG
        cat /tmp/eval_prev.log >> $LOG 2>/dev/null
    fi
    rm -f /tmp/eval_curr.log /tmp/eval_prev.log

    # -----------------------------------------------------------
    # Step 4: VFE comparison — GPU 1
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== B-R${ROUND} Step 4: VFE comparison ===" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=1 \
    python3 scripts/differentiable_bmr.py compare \
      --trajectory "$TRAJ" \
      --params "$OUTPUT" \
      --n-agents ${AGENTS} 2>&1 | tee -a $LOG

    CURRENT_PARAMS="$OUTPUT"

    echo "" | tee -a $LOG
    echo "B-R${ROUND} complete at $(date)" | tee -a $LOG
done

# -----------------------------------------------------------
# Phase B Final
# -----------------------------------------------------------
echo "" | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo "  PHASE B FINAL: All-Round Comparison" | tee -a $LOG
echo "============================================================" | tee -a $LOG

MACHINA_PARAMS="/tmp/learned_ext_arena_r10.npz"
for r in 1 2 3 4 5; do
    f="/tmp/learned_ext_machina_r${r}.npz"
    if [ -f "$f" ]; then
        MACHINA_PARAMS="${MACHINA_PARAMS} $f"
    fi
done

CUDA_VISIBLE_DEVICES=1 \
python3 scripts/differentiable_bmr.py compare \
  --trajectory "${TRAJ_BASE}_r5.npz" \
  --params ${MACHINA_PARAMS} \
  --n-agents ${AGENTS} 2>&1 | tee -a $LOG

echo "" | tee -a $LOG
echo "=== Phase B Final Eval: 10 episodes ===" | tee -a $LOG
CUDA_VISIBLE_DEVICES=1 \
  AIF_POLICY_LEN=${POLICY_LEN} \
  AIF_LEARNED_PARAMS="/tmp/learned_ext_machina_r5.npz" \
  AIF_NOVELTY_WEIGHT=1.0 \
  AIF_LEARN_INTERVAL=20 \
  cogames eval -m ${MAP} -v no_clips \
  -p ${POLICY_CLASS} \
  -c ${AGENTS} -e 10 -s ${STEPS_PER_EP} \
  --action-timeout-ms ${TIMEOUT} 2>&1 | tee -a $LOG | tail -30

# ==============================================================
#  FINAL: Cross-map eval
# ==============================================================
echo "" | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo "  FINAL CROSS-MAP EVAL" | tee -a $LOG
echo "============================================================" | tee -a $LOG

echo "--- Best arena params on arena (10 eps) ---" | tee -a $LOG
CUDA_VISIBLE_DEVICES=1 \
  AIF_POLICY_LEN=4 \
  AIF_LEARNED_PARAMS="/tmp/learned_ext_arena_r10.npz" \
  AIF_NOVELTY_WEIGHT=1.0 \
  AIF_LEARN_INTERVAL=20 \
  cogames eval -m arena -v no_clips \
  -p ${POLICY_CLASS} \
  -c 4 -e 10 -s 10000 \
  --action-timeout-ms 1000 2>&1 | tee -a $LOG | tail -15

echo "--- Best machina_1 params on machina_1 (10 eps) ---" | tee -a $LOG
CUDA_VISIBLE_DEVICES=1 \
  AIF_POLICY_LEN=4 \
  AIF_LEARNED_PARAMS="/tmp/learned_ext_machina_r5.npz" \
  AIF_NOVELTY_WEIGHT=1.0 \
  AIF_LEARN_INTERVAL=20 \
  cogames eval -m machina_1 -v no_clips \
  -p ${POLICY_CLASS} \
  -c 8 -e 10 -s 20000 \
  --action-timeout-ms 1000 2>&1 | tee -a $LOG | tail -15

echo "" | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo "  EXTENDED PIPELINE v2 COMPLETE" | tee -a $LOG
echo "  Finished: $(date)" | tee -a $LOG
echo "  Arena params:   /tmp/learned_ext_arena_r{1..10}.npz" | tee -a $LOG
echo "  machina_1 params: /tmp/learned_ext_machina_r{1..5}.npz" | tee -a $LOG
echo "  Full log: ${LOG}" | tee -a $LOG
echo "============================================================" | tee -a $LOG
