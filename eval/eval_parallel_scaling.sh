#!/bin/bash

export OPENAI_KEY=""
export OPENAI_API_KEY="$OPENAI_KEY"
export PROCESSOR="gpt-4o-mini"
# export ZHIPU_API_KEY=""
# export PROCESSOR="GLM-4.5-Flash"

PRUNE_LAYERS=(1 2)

declare -A TASKS
TASKS=(
  ["aime_nofigures_pass16"]="aime"
)

for PRUNE in "${PRUNE_LAYERS[@]}"; do
  MODEL_PATH=""

  for TASK in "${!TASKS[@]}"; do
      SUBDIR="${TASKS[$TASK]}"
      OUTPUT_PATH_BASE=""
      mkdir -p "$OUTPUT_PATH_BASE"

      for SEED in 7 11 42; do
        OUTPUT_PATH="${OUTPUT_PATH_BASE}/seed${SEED}"
        mkdir -p "$OUTPUT_PATH"

        PROCESSOR="$PROCESSOR" lm_eval \
          --model vllm \
          --model_args pretrained="${MODEL_PATH}",dtype=bfloat16,tensor_parallel_size=1 \
          --tasks $TASK \
          --batch_size auto \
          --apply_chat_template \
          --output_path "$OUTPUT_PATH" \
          --log_samples \
          --gen_kwargs "temperature=1.0,seed=${SEED},max_gen_toks=32768"
    done
  done
done
