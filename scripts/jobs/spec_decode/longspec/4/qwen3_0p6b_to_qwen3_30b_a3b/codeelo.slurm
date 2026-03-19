#!/bin/bash
#SBATCH --job-name=longspec_qwen3_0p6b_to_qwen3_30b_a3b_codeelo_b4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=rrg-pnair_gpu
#SBATCH --qos=rrg-pnair
#SBATCH --gres=gpu:h100:2
#SBATCH --mem-per-gpu=80G
#SBATCH --time=03:00:00
#SBATCH --output=/home/junsuk87/scratch/specdec/MagicDec/scripts/logs/spec_decode/longspec/4/qwen3_0p6b_to_qwen3_30b_a3b/longspec_qwen3_0p6b_to_qwen3_30b_a3b_codeelo_b4.out
#SBATCH --error=/home/junsuk87/scratch/specdec/MagicDec/scripts/logs/spec_decode/longspec/4/qwen3_0p6b_to_qwen3_30b_a3b/longspec_qwen3_0p6b_to_qwen3_30b_a3b_codeelo_b4.err

set -euo pipefail

module load python/3.12 cuda/12.9 arrow/21.0.0

REPO_DIR='/home/junsuk87/scratch/specdec/MagicDec'
VENV_DIR='/home/junsuk87/scratch/venvs/vllm'

cd "${REPO_DIR}"
source "${VENV_DIR}/bin/activate"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

# If your models require HF auth, export HF_TOKEN before sbatch.
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME="${HF_HOME:-/home/junsuk87/scratch/.cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/home/junsuk87/scratch/.cache/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/home/junsuk87/scratch/.cache/huggingface_hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/junsuk87/scratch/.cache/datasets}"

mkdir -p '/home/junsuk87/scratch/specdec/MagicDec/results/spec_decode/longspec/4/codeelo/Qwen__Qwen3_0.6B__TO__Qwen__Qwen3_30B_A3B'
echo "JOB: longspec_qwen3_0p6b_to_qwen3_30b_a3b_codeelo_b4"
echo "RESULTS_DIR: '/home/junsuk87/scratch/specdec/MagicDec/results/spec_decode/longspec/4/codeelo/Qwen__Qwen3_0.6B__TO__Qwen__Qwen3_30B_A3B'"

python -u 'scripts/run_spec_decode_metrics.py' \
  --draft-model 'Qwen/Qwen3-0.6B' \
  --target-models 'Qwen/Qwen3-30B-A3B' \
  --tp-map 'Qwen/Qwen3-30B-A3B=2' \
  --datasets 'codeelo' \
  --batch-sizes 4 \
  --num-spec-tokens 7 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --dtype auto \
  --seed 0 \
  --warmup-iters 2 \
  --warmup-max-tokens 32 \
  --results-root '/home/junsuk87/scratch/specdec/MagicDec/results/spec_decode/longspec/4/codeelo/Qwen__Qwen3_0.6B__TO__Qwen__Qwen3_30B_A3B' \
  --trust-remote-code \
  --enable-chunked-prefill \
  --method 'longspec' \
  --profile-time \
  --benchmark
