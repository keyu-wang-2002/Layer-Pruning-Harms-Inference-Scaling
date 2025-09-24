#!/bin/bash

export OPENAI_KEY=""
export OPENAI_API_KEY="$OPENAI_KEY"

TOKENS=(512 1024 2048 4096 8192)

PRUNE_LAYERS=(1 2)

declare -A TASKS
TASKS=(
  ["aime24_nofigures"]="aime"
  ["gpqa_diamond_openai"]="gpqa"
  ["openai_math"]="math"
)

for PRUNE in "${PRUNE_LAYERS[@]}"; do
  MODEL_PATH=""

  for TASK in "${!TASKS[@]}"; do
    for TOK in "${TOKENS[@]}"; do
      SUBDIR="${TASKS[$TASK]}"
      OUTPUT_PATH_BASE=""
      mkdir -p "$OUTPUT_PATH_BASE"

      for SEED in 7 11 42; do
        OUTPUT_PATH="${OUTPUT_PATH_BASE}/seed${SEED}"
        mkdir -p "$OUTPUT_PATH"

        PROCESSOR=gpt-4o-mini lm_eval \
          --model vllm \
          --model_args pretrained="${MODEL_PATH}",dtype=bfloat16,tensor_parallel_size=1 \
          --tasks $TASK \
          --batch_size auto \
          --apply_chat_template \
          --output_path "$OUTPUT_PATH" \
          --log_samples \
          --gen_kwargs "temperature=1.0,seed=${SEED},max_gen_toks=32768,max_tokens_thinking=${TOK}"
      done
    done
  done
done



















