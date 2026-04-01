#!/bin/bash
# B Evaluation Pipeline: Validate B-II through B-VI on AWS
#
# Prerequisites:
#   - cogames 0.19+ installed with aif_meta_cogames package
#   - B5 params file at ~/projects/aif-meta-cogames/params/learned_B5.npz
#   - custom_B/custom_C support in generative_model.py + cogames_policy.py
#
# Usage:
#   bash scripts/eval_b_pipeline.sh
#
# Output: per-approach junction counts + VFE model comparison
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

B5_PARAMS="${PROJECT_DIR}/params/learned_B5.npz"
TRAJ_PATH="/tmp/aif_traj_b5.npz"
AIF_CLASS="aif_meta_cogames.aif_agent.cogames_policy.AIFPolicy"
EVAL_ARGS="-c 4 -e 5 --action-timeout-ms 1000"

echo "============================================"
echo "  B Evaluation Pipeline"
echo "  $(date)"
echo "============================================"

# --- Step 1: Collect trajectory with B5 params (current best A-only) ---
echo ""
echo "=== Step 1: Collecting trajectory with B5 baseline ==="
if [ ! -f "$B5_PARAMS" ]; then
    echo "WARNING: B5 params not found at $B5_PARAMS"
    echo "Skipping trajectory collection. Create B5 params first:"
    echo "  python scripts/learn_parameters.py learn-a --trajectory <traj> --output $B5_PARAMS"
    exit 1
fi

AIF_LOG_TRAJECTORY=1 AIF_TRAJECTORY_PATH="$TRAJ_PATH" \
  AIF_LEARNED_PARAMS="$B5_PARAMS" \
  cogames eval -m arena -v no_clips \
  -p "class=$AIF_CLASS" $EVAL_ARGS

if [ ! -f "$TRAJ_PATH" ]; then
    echo "ERROR: Trajectory file not created at $TRAJ_PATH"
    exit 1
fi
echo "Trajectory saved: $TRAJ_PATH"

# --- Step 2: Run each learning approach ---
echo ""
echo "=== Step 2a: B-IV Joint A+B+C ==="
python "$SCRIPT_DIR/learn_parameters.py" learn-full \
  --trajectory "$TRAJ_PATH" --steps 300 --lr 0.001 \
  --c-lr-scale 0.1 --a-weight 1.0 --b-weight 1.0 --c-weight 0.5 \
  --output /tmp/learned_joint.npz

echo ""
echo "=== Step 2b: B-V De novo ==="
python "$SCRIPT_DIR/denovo_learn.py" learn \
  --trajectory "$TRAJ_PATH" --prior-scale 0.1 \
  --bmr --bmr-threshold 3.0 \
  --output /tmp/learned_denovo.npz

echo ""
echo "=== Step 2c: B-VI Differentiable BMR (prune) ==="
python "$SCRIPT_DIR/differentiable_bmr.py" prune \
  --trajectory "$TRAJ_PATH" --rounds 3 \
  --prune-percentile 10.0 --refine-steps 100 \
  --output /tmp/learned_bmr.npz

echo ""
echo "=== Step 2d: B-VI De novo -> gradient refinement ==="
python "$SCRIPT_DIR/differentiable_bmr.py" refine \
  --trajectory "$TRAJ_PATH" \
  --init-params /tmp/learned_denovo.npz --steps 200 \
  --output /tmp/learned_refined.npz

# --- Step 3: Model comparison (VFE ranking) ---
echo ""
echo "=== Step 3: Model comparison (VFE ranking) ==="
python "$SCRIPT_DIR/differentiable_bmr.py" compare \
  --trajectory "$TRAJ_PATH" \
  --params /tmp/learned_joint.npz /tmp/learned_denovo.npz \
          /tmp/learned_bmr.npz /tmp/learned_refined.npz

# --- Step 4: Eval each approach live (5 episodes each) ---
echo ""
echo "=== Step 4: Live evaluation ==="
for name in joint denovo bmr refined; do
  echo ""
  echo "--- Eval: ${name} ---"
  AIF_LEARNED_PARAMS="/tmp/learned_${name}.npz" \
    cogames eval -m arena -v no_clips \
    -p "class=$AIF_CLASS" $EVAL_ARGS
done

# --- Step 5: Baseline (B5 A-only) for comparison ---
echo ""
echo "=== Step 5: Baseline B5 (A-only) ==="
AIF_LEARNED_PARAMS="$B5_PARAMS" \
  cogames eval -m arena -v no_clips \
  -p "class=$AIF_CLASS" $EVAL_ARGS

# --- Step 6: Small map test (PI suggestion) ---
echo ""
echo "=== Step 6: Small map test (tutorial.aligner) ==="
AIF_LEARNED_PARAMS="/tmp/learned_joint.npz" \
  cogames eval -m tutorial.aligner \
  -p "class=$AIF_CLASS" $EVAL_ARGS

echo ""
echo "============================================"
echo "  Pipeline complete: $(date)"
echo "============================================"
echo ""
echo "Results summary:"
echo "  Trajectory:     $TRAJ_PATH"
echo "  B-IV (joint):   /tmp/learned_joint.npz"
echo "  B-V (de novo):  /tmp/learned_denovo.npz"
echo "  B-VI (BMR):     /tmp/learned_bmr.npz"
echo "  B-VI (refined): /tmp/learned_refined.npz"
