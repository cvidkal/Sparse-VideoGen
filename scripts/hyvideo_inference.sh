#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model

# TFP Values: 
# Set the following values to control the percentage of timesteps using dense attention:
# 35% → 0.07, 30% → 0.055, 25% → 0.04, 20% → 0.033, 15% → 0.02, 10% → 0.015
first_times_fp=0.055
first_layers_fp=0.025
# timestamp=$(date +%Y-%m-%d)
timestamp=2025-05-12

mode=$3
loop_num=$4

# if the output path exists, open resume mode
if [ -d "./output/${timestamp}-${mode}" ]; then
    resume="--resume"
else
    resume=""
fi

set -o xtrace
mkdir -p ./output/${timestamp}-${mode}

whoami && echo $MASTER_ADDR && echo $MASTER_PORT && echo $RANK && echo $WORLD_SIZE

if [ "$MASTER_ADDR" == "" ]; then
    MASTER_ADDR_ARG=""
else
    MASTER_ADDR_ARG="--master_addr $MASTER_ADDR"
fi

if [ "$MASTER_PORT" == "" ]; then
    MASTER_PORT_ARG=""
else
    MASTER_PORT_ARG="--master_port $MASTER_PORT"
fi

if [[ "$mode" == "sparse" ]]; then
    torchrun --nnodes=$1 --nproc_per_node=$2 $MASTER_ADDR_ARG $MASTER_PORT_ARG --node_rank $RANK hyvideo_inference.py \
        --video-size 720 1280 \
        --video-length 49 \
        --infer-steps 50 \
        --seed 0 \
        --prompt-file assets/vbench2_prompts_list.txt \
        --embedded-cfg-scale 6.0 \
        --flow-shift 7.0 \
        --flow-reverse \
        --output_path ./output/${timestamp}-${mode} \
        --pattern "SVG" \
        --num_sampled_rows 64 \
        --sparsity 0.2 ${resume} \
        --tea-cache \
        --first_times_fp $first_times_fp \
        --first_layers_fp $first_layers_fp \
        --loop-num $loop_num 
else
    torchrun --nnodes=$1 --nproc_per_node=$2 $MASTER_ADDR_ARG $MASTER_PORT_ARG --node_rank $RANK hyvideo_inference.py \
        --video-size 720 1280 \
        --video-length 49 \
        --infer-steps 50 \
        --seed 0 \
        --prompt-file assets/vbench2_prompts_list.txt \
        --embedded-cfg-scale 6.0 \
        --flow-shift 7.0 \
        --flow-reverse \
        --output_path ./output/${timestamp}-${mode}  \
        --pattern "dense" \
        --num_sampled_rows 64 \
        --sparsity 0.2 ${resume} \
        --first_times_fp $first_times_fp \
        --first_layers_fp $first_layers_fp \
        --loop-num $loop_num
fi