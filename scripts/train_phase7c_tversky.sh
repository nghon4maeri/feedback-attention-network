#!/bin/bash
# ==============================================================================
# FANET PHASE 7C: Learned Residual Decoupling + Curriculum Tversky Loss
# ==============================================================================
# This script executes the training of the Phase 7C architecture designed to
# overcome the "Soft-OR Monotonicity Trap" discovered during Phase 7B.
#
# Key Features:
# 1. --gating-mode learned_residual : Replaces the monotonic Soft-OR gate with a
#    1x1 Conv gate that can actively prune (subtract) False Positives.
# 2. --detach-feedback : Structurally preserves gradient decoupling to prevent 
#    recurrent error loops (The Feedback Trap).
# 3. --loss tversky : Uses Curriculum Adaptive Tversky Loss to heavily penalize
#    False Positives after the network stabilizes.
# 4. --seed 2024 : Fixes the seed for reproducibility against the 52% FPR baseline.
# ==============================================================================

# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints_phase7c

echo "Starting Phase 7C Training: Learned Residual Decoupling..."

python scripts/train.py \
    --config configs/kvasir_sessile.yaml \
    --gating-mode learned_residual \
    --detach-feedback \
    --loss tversky \
    --seed 2024 \
    --epochs 200

# After training finishes, the default train.py saves to `checkpoints/checkpoint.pth`
# based on the yaml config. We need to move it to our phase7c folder.
if [ -f "checkpoints/checkpoint.pth" ]; then
    mv checkpoints/checkpoint.pth checkpoints_phase7c/M12_Phase7C_best.pth
    echo "Training Complete! Best model saved to checkpoints_phase7c/M12_Phase7C_best.pth"
else
    echo "Training failed or checkpoint not found!"
fi
