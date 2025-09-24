#!/bin/bash
#SBATCH --job-name=s1_train
#SBATCH --output=train_s1_2.out
#SBATCH --error=train_s1_2.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --time=00-20:00:00

# activate conda env
source /mnt/fast/nobackup/users/ly0008/miniconda3/etc/profile.d/conda.sh
conda activate s1

output="/mnt/fast/nobackup/scratch4weeks/ly0008/wangkeyu/s1/output_models/s1.1-7B-LaCo-merge-1-layer-sft-lora"
base_model="MrLvTian/s1.1-7B-LaCo-merge-1-layer"
lr=1e-5
min_lr=0
epochs=5
weight_decay=1e-4 # -> the same training pipe as slurm_training
micro_batch_size=1 # -> batch_size will be 16 if 16 gpus
gradient_accumulation_steps=1 # requires more GPU memory
max_steps=-1
push_to_hub=false


python  sft_lora.py \
    --block_size=32768 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path="simplescaling/s1K_tokenized" \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="no" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="${output}" \
    --push_to_hub=${push_to_hub} \
    --save_only_model=True \
    --report_to="none" \
    --gradient_checkpointing
    # --gradient_checkpointing=True \ Enable gradient checkpointing for efficient memory usage with 8 H100 GPUs.
    # --accelerator_config='{"gradient_accumulation_kwargs": {"sync_each_batch": true}}'
    # --train_file_path="simplescaling/s1K_tokenized" \
    # 