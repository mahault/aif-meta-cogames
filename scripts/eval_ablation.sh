#!/bin/bash
# Systematic ablation: which features actually drive performance?
# 6 configs × 5 episodes each on arena no_clips, R2 learned params, policy_len=4
# Expected runtime: ~18 hours total

set -e

LOG=/tmp/eval_ablation.log
COMMON="AIF_LEARNED_PARAMS=/tmp/learned_iter_r2.npz AIF_POLICY_LEN=4 CUDA_VISIBLE_DEVICES=0"
EVAL_CMD="cogames eval -m arena -v no_clips -c 4 -e 5 --seed 42 -p class=aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy"

echo "=== SYSTEMATIC ABLATION: $(date) ===" | tee $LOG
echo "Common: $COMMON" | tee -a $LOG
echo "" | tee -a $LOG

# --- Config 1: Baseline (no features) ---
echo "=== [1/6] BASELINE (no features) ===" | tee -a $LOG
echo "Start: $(date)" | tee -a $LOG
env $COMMON $EVAL_CMD 2>&1 | tee -a $LOG
echo "End: $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# --- Config 2: Adaptive gamma only ---
echo "=== [2/6] ADAPTIVE GAMMA ONLY ===" | tee -a $LOG
echo "Start: $(date)" | tee -a $LOG
env $COMMON AIF_ADAPTIVE_GAMMA=1 $EVAL_CMD 2>&1 | tee -a $LOG
echo "End: $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# --- Config 3: Explore E only ---
echo "=== [3/6] EXPLORE E ONLY ===" | tee -a $LOG
echo "Start: $(date)" | tee -a $LOG
env $COMMON AIF_EXPLORE_E=1 $EVAL_CMD 2>&1 | tee -a $LOG
echo "End: $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# --- Config 4: Novelty only ---
echo "=== [4/6] NOVELTY ONLY ===" | tee -a $LOG
echo "Start: $(date)" | tee -a $LOG
env $COMMON AIF_NOVELTY_WEIGHT=1.0 $EVAL_CMD 2>&1 | tee -a $LOG
echo "End: $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# --- Config 5: VFE-gated learning only (habit bypass + learn_interval=20) ---
echo "=== [5/6] VFE-GATED LEARNING ONLY ===" | tee -a $LOG
echo "Start: $(date)" | tee -a $LOG
env $COMMON AIF_HABIT_BYPASS=1 AIF_LEARN_INTERVAL=20 $EVAL_CMD 2>&1 | tee -a $LOG
echo "End: $(date)" | tee -a $LOG
echo "" | tee -a $LOG

# --- Config 6: ALL features combined ---
echo "=== [6/6] ALL FEATURES COMBINED ===" | tee -a $LOG
echo "Start: $(date)" | tee -a $LOG
env $COMMON AIF_ADAPTIVE_GAMMA=1 AIF_EXPLORE_E=1 AIF_NOVELTY_WEIGHT=1.0 AIF_HABIT_BYPASS=1 AIF_LEARN_INTERVAL=20 $EVAL_CMD 2>&1 | tee -a $LOG
echo "End: $(date)" | tee -a $LOG
echo "" | tee -a $LOG

echo "=== ABLATION COMPLETE: $(date) ===" | tee -a $LOG

# --- Summary extraction ---
echo "" | tee -a $LOG
echo "=== SUMMARY ===" | tee -a $LOG
grep -E '^\=\=\= \[|Per-Agent Reward|arena.*AIFPolicy|Timeouts' $LOG | tee -a /tmp/eval_ablation_summary.log
echo "Full log: $LOG" | tee -a $LOG
echo "Summary: /tmp/eval_ablation_summary.log" | tee -a $LOG
