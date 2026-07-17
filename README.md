# When Fewer Layers Break More Chains: Layer Pruning Harms Test-Time Scaling in LLMs

## Structure
```
├── lib/                     # Pruning strategies implementation
├── eval/                    # Sequential & parallel test-time scaling evaluation
├── train/                   # LoRA and Full fine-tuning implementation
├── analysis/                # Model output analysis scripts
└── main.py                  # Pruning and model saving entry point
```

## Install
```
conda create -n layerpruning python==3.10 -y
conda activate layerpruning
pip install -r requirements.txt
pip install -e eval/lm-evaluation-harness/
```

## Usage
### Pruning
#### ShortGPT Pruning
```
python main.py \
    --model Qwen/Qwen3-8B \
    --prune_method shortgpt \
    --seed 0 \
    --remove_n_layers 3 \
    --n_samples 1000 \
    --max_seq_len 1024 \
    --batch_size 8
```

#### Reverse-order prune
```
python main.py \
    --model Qwen/Qwen3-8B \
    --prune_method tail \
    --seed 0 \
    --remove_n_layers 3 \
```

#### LaCo
Adjust `laco_threshold` to merge certain number of layers.
```
python main.py \
    --model Qwen/Qwen3-8B \
    --prune_method laco \
    --calibration_data laco \
    --laco_merge_layers 2 \
    --seed 0 \
    --laco_threshold 0.9
```

#### Selection
You can also specify the layer index you want to prune by:
```
python main.py \
    --model Qwen/Qwen3-8B \
    --prune_method selection \
    --seed 0 \
    --layers_to_remove 25 27 26
```

### Evaluation
Input the model path and output path in ./eval/eval_sequential_scaling.sh and ./eval/eval_parallel_scaling.sh
```
sh ./eval/eval_sequential_scaling.sh
sh ./eval/eval_parallel_scaling.sh
```

### Analysis
We provide an example sampled from output of `s1.1-7B` model on `AIME24` dataset.
To run analyse on diversity, just use:
```
python ./analysis/diversity.py --jsonl_path ./analysis/example/s1.1-7B_aime24.jsonl
```
To run analyse on self-reflection, use:
```
python ./analysis/self_reflection.py --jsonl_path ./analysis/example/s1.1-7B_aime24.jsonl --max_samples 200
```

## Citation
```
@inproceedings{wang2025fewer,
  title={When Fewer Layers Break More Chains: Layer Pruning Harms Test-Time Scaling in LLMs},
  author={Wang, Keyu and Lyu, Tianqi and Su, Guancheng and others},
  booktitle={Proceedings of the Conference on Language Modeling (COLM)},
  year={2026},
  address={San Francisco, CA},
  month={October}
}
```
