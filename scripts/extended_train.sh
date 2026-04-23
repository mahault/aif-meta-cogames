#!/bin/bash
set -e

# Extended Training Pipeline: Arena (10 rounds) + machina_1 (5 rounds)
#
# Ablation-informed feature selection (2026-04-12):
#   ON:  Novelty η=1.0 (best single feature: 1.83 j/agent)
#   ON:  VFE-gated lr + learn_interval=20 (online adaptation: 1.04 j/agent)
#   ON:  Explore E (rounds 1-N/2 only, then deploy-mode)
#   OFF: Adaptive gamma (marginal alone: 0.35, destructive combined)
#   OFF: Habit bypass (never triggered — E splits 50/50)
#
# Phase A: 10 rounds on arena (50x50, 4 agents), init from R2
# Phase B: 5 rounds on machina_1 (88x88, 8 agents), init from best arena
#
# Expected runtime: ~100 hours total (50h arena + 50h machina_1)

source ~/projects/cogames-env/bin/activate
cd ~/projects/aif-meta-cogames

# Use GPU 1 (GPU 0 is occupied by PPO training)
export CUDA_VISIBLE_DEVICES=1

# Ensure learn_parameters is importable from cogames_policy trajectory saving
export PYTHONPATH="$HOME/projects/aif-meta-cogames/scripts:${PYTHONPATH:-}"

# Reinstall package from synced source
pip install --quiet --no-deps -e ~/projects/aif-meta-cogames

LOG=/tmp/extended_train.log
POLICY_CLASS="class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy"

# Common learning hyperparams
GRAD_STEPS=5000
LR=0.001
C_LR_SCALE=0.5

echo "============================================================" | tee $LOG
echo "  EXTENDED TRAINING PIPELINE" | tee -a $LOG
echo "  Phase A: Arena (10 rounds, 4 agents)" | tee -a $LOG
echo "  Phase B: machina_1 (5 rounds, 8 agents)" | tee -a $LOG
echo "  Features: novelty + VFE-gated lr (NO adaptive gamma, NO habit bypass)" | tee -a $LOG
echo "  Started: $(date)" | tee -a $LOG
echo "============================================================" | tee -a $LOG

# ==============================================================
#  PHASE A: Arena Extended (10 rounds)
# ==============================================================

CURRENT_PARAMS=/tmp/learned_iter_r2.npz
MAP=arena
AGENTS=4
STEPS_PER_EP=10000
TIMEOUT=1000
POLICY_LEN=4
TRAJ_BASE=/tmp/aif_ext_arena
BEST_ARENA_SCORE=0
BEST_ARENA_ROUND=1

echo "" | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo "  PHASE A: Arena Extended Training" | tee -a $LOG
echo "  Map: ${MAP} no_clips, Agents: ${AGENTS}" | tee -a $LOG
echo "  Steps/ep: ${STEPS_PER_EP}, Policy len: ${POLICY_LEN}" | tee -a $LOG
echo "  Init: ${CURRENT_PARAMS}" | tee -a $LOG
echo "============================================================" | tee -a $LOG

for ROUND in 1 2 3 4 5 6 7 8 9 10; do
    echo "" | tee -a $LOG
    echo "############################################################" | tee -a $LOG
    echo "  PHASE A — ROUND ${ROUND}/10" | tee -a $LOG
    echo "  Using params: ${CURRENT_PARAMS}" | tee -a $LOG
    echo "  Time: $(date)" | tee -a $LOG
    echo "############################################################" | tee -a $LOG

    TRAJ="${TRAJ_BASE}_r${ROUND}.npz"
    OUTPUT="/tmp/learned_ext_arena_r${ROUND}.npz"
    rm -f "$TRAJ"

    # -----------------------------------------------------------
    # Step 1: Collect trajectory
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== A-R${ROUND} Step 1: Collecting trajectory ===" | tee -a $LOG

    # Exploration schedule: rounds 1-5 explore, rounds 6-10 deploy
    if [ ${ROUND} -le 5 ]; then
        EXPLORE_E=1
        echo "  Exploration mode (flat E)" | tee -a $LOG
    else
        EXPLORE_E=0
        echo "  Deploy mode (strong E)" | tee -a $LOG
    fi

    # 10 batches x 5 episodes x 10,000 steps = 500k steps per round
    for batch in $(seq 1 10); do
        echo "--- Batch ${batch}/10 ---" | tee -a $LOG
        AIF_POLICY_LEN=${POLICY_LEN} \
          AIF_LOG_TRAJECTORY=1 \
          AIF_TRAJECTORY_PATH="$TRAJ" \
          AIF_LEARNED_PARAMS="${CURRENT_PARAMS}" \
          AIF_EXPLORE_E=${EXPLORE_E} \
          AIF_NOVELTY_WEIGHT=1.0 \
          AIF_LEARN_INTERVAL=20 \
          cogames eval -m ${MAP} -v no_clips \
          -p ${POLICY_CLASS} \
          -c ${AGENTS} -e 5 -s ${STEPS_PER_EP} \
          --action-timeout-ms ${TIMEOUT} 2>&1 | tee -a $LOG | tail -8
    done

    # Check trajectory size
    python3 -c "
import numpy as np
d = np.load('${TRAJ}')
n = int(d['n_steps'])
print(f'A-R${ROUND} trajectory: {n} steps')
" | tee -a $LOG

    # -----------------------------------------------------------
    # Step 2: Learn parameters (joint A+B+C)
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== A-R${ROUND} Step 2: Joint A+B+C learning (${GRAD_STEPS} steps) ===" | tee -a $LOG
    python3 scripts/learn_parameters.py learn-full \
      --trajectory "$TRAJ" \
      --steps ${GRAD_STEPS} \
      --lr ${LR} \
      --c-lr-scale ${C_LR_SCALE} \
      --a-weight 1.0 --b-weight 1.0 --c-weight 1.0 \
      --n-agents ${AGENTS} \
      --output "$OUTPUT" 2>&1 | tee -a $LOG

    # -----------------------------------------------------------
    # Step 3: Quick eval (5 episodes)
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== A-R${ROUND} Step 3: Quick eval ===" | tee -a $LOG

    echo "--- Current (round ${ROUND}) ---" | tee -a $LOG
    AIF_POLICY_LEN=${POLICY_LEN} \
      AIF_LEARNED_PARAMS="${OUTPUT}" \
      AIF_NOVELTY_WEIGHT=1.0 \
      AIF_LEARN_INTERVAL=20 \
      cogames eval -m ${MAP} -v no_clips \
      -p ${POLICY_CLASS} \
      -c ${AGENTS} -e 5 -s ${STEPS_PER_EP} \
      --action-timeout-ms ${TIMEOUT} 2>&1 | tee -a $LOG | tail -15

    if [ ${ROUND} -gt 1 ]; then
        PREV="/tmp/learned_ext_arena_r$((ROUND-1)).npz"
        echo "--- Previous (round $((ROUND-1))) ---" | tee -a $LOG
        AIF_POLICY_LEN=${POLICY_LEN} \
          AIF_LEARNED_PARAMS="${PREV}" \
          AIF_NOVELTY_WEIGHT=1.0 \
          AIF_LEARN_INTERVAL=20 \
          cogames eval -m ${MAP} -v no_clips \
          -p ${POLICY_CLASS} \
          -c ${AGENTS} -e 5 -s ${STEPS_PER_EP} \
          --action-timeout-ms ${TIMEOUT} 2>&1 | tee -a $LOG | tail -15
    fi

    # -----------------------------------------------------------
    # Step 4: VFE comparison
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== A-R${ROUND} Step 4: VFE comparison ===" | tee -a $LOG
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

python3 scripts/differentiable_bmr.py compare \
  --trajectory "${TRAJ_BASE}_r10.npz" \
  --params ${ARENA_PARAMS} \
  --n-agents ${AGENTS} 2>&1 | tee -a $LOG

# Phase A 10-episode final eval
echo "" | tee -a $LOG
echo "=== Phase A Final Eval: 10 episodes ===" | tee -a $LOG
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

# Use the last arena params as init (r10 — fully trained)
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
    rm -f "$TRAJ"

    # -----------------------------------------------------------
    # Step 1: Collect trajectory
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== B-R${ROUND} Step 1: Collecting trajectory ===" | tee -a $LOG

    # Exploration schedule: rounds 1-3 explore, rounds 4-5 deploy
    if [ ${ROUND} -le 3 ]; then
        EXPLORE_E=1
        echo "  Exploration mode (flat E)" | tee -a $LOG
    else
        EXPLORE_E=0
        echo "  Deploy mode (strong E)" | tee -a $LOG
    fi

    # 10 batches x 5 episodes x 20,000 steps = 1M steps per round
    for batch in $(seq 1 10); do
        echo "--- Batch ${batch}/10 ---" | tee -a $LOG
        AIF_POLICY_LEN=${POLICY_LEN} \
          AIF_LOG_TRAJECTORY=1 \
          AIF_TRAJECTORY_PATH="$TRAJ" \
          AIF_LEARNED_PARAMS="${CURRENT_PARAMS}" \
          AIF_EXPLORE_E=${EXPLORE_E} \
          AIF_NOVELTY_WEIGHT=1.0 \
          AIF_LEARN_INTERVAL=20 \
          cogames eval -m ${MAP} -v no_clips \
          -p ${POLICY_CLASS} \
          -c ${AGENTS} -e 5 -s ${STEPS_PER_EP} \
          --action-timeout-ms ${TIMEOUT} 2>&1 | tee -a $LOG | tail -8
    done

    # Check trajectory size
    python3 -c "
import numpy as np
d = np.load('${TRAJ}')
n = int(d['n_steps'])
print(f'B-R${ROUND} trajectory: {n} steps')
" | tee -a $LOG

    # -----------------------------------------------------------
    # Step 2: Learn parameters (joint A+B+C)
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== B-R${ROUND} Step 2: Joint A+B+C learning (${GRAD_STEPS} steps) ===" | tee -a $LOG
    python3 scripts/learn_parameters.py learn-full \
      --trajectory "$TRAJ" \
      --steps ${GRAD_STEPS} \
      --lr ${LR} \
      --c-lr-scale ${C_LR_SCALE} \
      --a-weight 1.0 --b-weight 1.0 --c-weight 1.0 \
      --n-agents ${AGENTS} \
      --output "$OUTPUT" 2>&1 | tee -a $LOG

    # -----------------------------------------------------------
    # Step 3: Quick eval (5 episodes)
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== B-R${ROUND} Step 3: Quick eval ===" | tee -a $LOG

    echo "--- Current (round ${ROUND}) ---" | tee -a $LOG
    AIF_POLICY_LEN=${POLICY_LEN} \
      AIF_LEARNED_PARAMS="${OUTPUT}" \
      AIF_NOVELTY_WEIGHT=1.0 \
      AIF_LEARN_INTERVAL=20 \
      cogames eval -m ${MAP} -v no_clips \
      -p ${POLICY_CLASS} \
      -c ${AGENTS} -e 5 -s ${STEPS_PER_EP} \
      --action-timeout-ms ${TIMEOUT} 2>&1 | tee -a $LOG | tail -15

    if [ ${ROUND} -gt 1 ]; then
        PREV="/tmp/learned_ext_machina_r$((ROUND-1)).npz"
        echo "--- Previous (round $((ROUND-1))) ---" | tee -a $LOG
        AIF_POLICY_LEN=${POLICY_LEN} \
          AIF_LEARNED_PARAMS="${PREV}" \
          AIF_NOVELTY_WEIGHT=1.0 \
          AIF_LEARN_INTERVAL=20 \
          cogames eval -m ${MAP} -v no_clips \
          -p ${POLICY_CLASS} \
          -c ${AGENTS} -e 5 -s ${STEPS_PER_EP} \
          --action-timeout-ms ${TIMEOUT} 2>&1 | tee -a $LOG | tail -15
    fi

    # -----------------------------------------------------------
    # Step 4: VFE comparison
    # -----------------------------------------------------------
    echo "" | tee -a $LOG
    echo "=== B-R${ROUND} Step 4: VFE comparison ===" | tee -a $LOG
    python3 scripts/differentiable_bmr.py compare \
      --trajectory "$TRAJ" \
      --params "$OUTPUT" \
      --n-agents ${AGENTS} 2>&1 | tee -a $LOG

    # Update for next round
    CURRENT_PARAMS="$OUTPUT"

    echo "" | tee -a $LOG
    echo "B-R${ROUND} complete at $(date)" | tee -a $LOG
done

# -----------------------------------------------------------
# Phase B Final: Compare all machina_1 rounds
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

python3 scripts/differentiable_bmr.py compare \
  --trajectory "${TRAJ_BASE}_r5.npz" \
  --params ${MACHINA_PARAMS} \
  --n-agents ${AGENTS} 2>&1 | tee -a $LOG

# Phase B 10-episode final eval
echo "" | tee -a $LOG
echo "=== Phase B Final Eval: 10 episodes ===" | tee -a $LOG
AIF_POLICY_LEN=${POLICY_LEN} \
  AIF_LEARNED_PARAMS="/tmp/learned_ext_machina_r5.npz" \
  AIF_NOVELTY_WEIGHT=1.0 \
  AIF_LEARN_INTERVAL=20 \
  cogames eval -m ${MAP} -v no_clips \
  -p ${POLICY_CLASS} \
  -c ${AGENTS} -e 10 -s ${STEPS_PER_EP} \
  --action-timeout-ms ${TIMEOUT} 2>&1 | tee -a $LOG | tail -30

# ==============================================================
#  FINAL: Cross-map eval with best params
# ==============================================================
echo "" | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo "  FINAL CROSS-MAP EVAL" | tee -a $LOG
echo "============================================================" | tee -a $LOG

echo "--- Best arena params on arena (10 eps) ---" | tee -a $LOG
AIF_POLICY_LEN=4 \
  AIF_LEARNED_PARAMS="/tmp/learned_ext_arena_r10.npz" \
  AIF_NOVELTY_WEIGHT=1.0 \
  AIF_LEARN_INTERVAL=20 \
  cogames eval -m arena -v no_clips \
  -p ${POLICY_CLASS} \
  -c 4 -e 10 -s 10000 \
  --action-timeout-ms 1000 2>&1 | tee -a $LOG | tail -15

echo "--- Best machina_1 params on machina_1 (10 eps) ---" | tee -a $LOG
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
echo "  EXTENDED PIPELINE COMPLETE" | tee -a $LOG
echo "  Finished: $(date)" | tee -a $LOG
echo "  Arena params:   /tmp/learned_ext_arena_r{1..10}.npz" | tee -a $LOG
echo "  machina_1 params: /tmp/learned_ext_machina_r{1..5}.npz" | tee -a $LOG
echo "  Full log: ${LOG}" | tee -a $LOG
echo "============================================================" | tee -a $LOG
