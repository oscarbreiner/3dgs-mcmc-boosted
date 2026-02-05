#!/bin/bash

# Configuration
SESSION_NAME="training"
LOG_FILE="experiment_log.txt"

# PHRASE 1: LAUNCHER (Runs on Login Node)
# Checks if running inside tmux. If not, creates the session.
if [ -z "$TMUX" ]; then
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Tmux session '$SESSION_NAME' already exists."
        read -p "Do you want to kill the existing session? (y/n) " choice
        case "$choice" in 
            y|Y ) echo "Killing existing session..."; tmux kill-session -t "$SESSION_NAME";;
            * ) echo "Exiting."; exit 1;;
        esac
    fi

    echo "Launching tmux session '$SESSION_NAME'..."
    # We recursively call this script with "coordinator" mode inside tmux
    tmux new-session -s "$SESSION_NAME" "bash $0 coordinator"
    exit 0
fi

# PHASE 2: COORDINATOR (Runs inside tmux on Login Node)
# This phase requests the GPU resource using salloc.
if [ "$1" == "coordinator" ]; then
    echo "Tmux session started."
    echo "Requesting GPU allocation with salloc..."
    
    # We call this script again with "worker" mode INSIDE the allocation.
    # Note: We use 'time' to measure how long the whole job takes.
    # We pass the current directory to ensure salloc knows where we are.
    # We use 'srun' to ensure the script runs ON the compute node, not the login node.
    salloc --gpus=1 srun bash "$0" worker
    
    echo "Allocation finished. Press [Enter] to exit terminal."
    read
    exit 0
fi

# PHASE 3: WORKER (Runs on Compute Node)
# This is the actual workload running on the GPU.
if [ "$1" == "worker" ]; then
    
    # --- Environment Setup ---
    # Try to source conda.sh
    if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    fi

    echo "Activating environment: 3dgs-mcmc-env"
    conda activate 3dgs-mcmc-env

    # Initialize Log
    echo "Starting 3DGS Experiment Worker on host $(hostname) - $(date)" > "$LOG_FILE"

    # --- Parameters ---
    NOISE_ABSOLUTE_THRESHOLD=(0.02 0.1 0.5 2 5 10)

    # --- Training Loop ---
    for WINDOW_SIZE in "${WINDOW_SIZES[@]}"; do
        echo "" | tee -a "$LOG_FILE"
        echo "################################################################" | tee -a "$LOG_FILE"
        echo "Running test with error threshold: $NOISE_ABSOLUTE_THRESHOLD" | tee -a "$LOG_FILE"
        echo "################################################################" | tee -a "$LOG_FILE"

        OUTPUT_PATH="output/clear/opacity_error_abs_threshold_search${NOISE_ABSOLUTE_THRESHOLD}"
        echo "Output Path: $OUTPUT_PATH" | tee -a "$LOG_FILE"

        python train.py \
            --config configs/bicycle.json \
            --eval \
            --noie_absolute_threshold "$NOISE_ABSOLUTE_THRESHOLD" \
            --model_path "$OUTPUT_PATH" \
            2>&1 | tee -a "$LOG_FILE"

        echo "Completed experiment for threshold $NOISE_ABSOLUTE_THRESHOLD" | tee -a "$LOG_FILE"
    done

    exit 0
fi
