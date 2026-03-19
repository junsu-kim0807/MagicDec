#!/bin/bash
#SBATCH --job-name=longspec_deepseekcoder_1p3b_to_33b_codeelo_b16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=rrg-pnair_gpu
#SBATCH --qos=rrg-pnair
#SBATCH --gres=gpu:h100:2
#SBATCH --mem-per-gpu=80G
#SBATCH --time=03:00:00
#SBATCH --output=/home/junsuk87/scratch/specdec/MagicDec/scripts/logs/spec_decode/longspec/16/deepseekcoder_1p3b_to_33b/longspec_deepseekcoder_1p3b_to_33b_codeelo_b16.out
#SBATCH --error=/home/junsuk87/scratch/specdec/MagicDec/scripts/logs/spec_decode/longspec/16/deepseekcoder_1p3b_to_33b/longspec_deepseekcoder_1p3b_to_33b_codeelo_b16.err

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

mkdir -p '/home/junsuk87/scratch/specdec/MagicDec/results/spec_decode/longspec/16/codeelo/deepseek_ai__deepseek_coder_1.3b_instruct__TO__deepseek_ai__deepseek_coder_33b_instruct'
echo "JOB: longspec_deepseekcoder_1p3b_to_33b_codeelo_b16"
echo "RESULTS_DIR: '/home/junsuk87/scratch/specdec/MagicDec/results/spec_decode/longspec/16/codeelo/deepseek_ai__deepseek_coder_1.3b_instruct__TO__deepseek_ai__deepseek_coder_33b_instruct'"

python -u 'scripts/run_spec_decode_metrics.py' \
  --draft-model 'deepseek-ai/deepseek-coder-1.3b-instruct' \
  --target-models 'deepseek-ai/deepseek-coder-33b-instruct' \
  --tp-map 'deepseek-ai/deepseek-coder-33b-instruct=2' \
  --datasets 'codeelo' \
  --batch-sizes 16 \
  --num-spec-tokens 7 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --dtype auto \
  --seed 0 \
  --warmup-iters 2 \
  --warmup-max-tokens 32 \
  --results-root '/home/junsuk87/scratch/specdec/MagicDec/results/spec_decode/longspec/16/codeelo/deepseek_ai__deepseek_coder_1.3b_instruct__TO__deepseek_ai__deepseek_coder_33b_instruct' \
  --trust-remote-code \
  --enable-chunked-prefill \
  --method 'longspec' \
  --profile-time \
  --benchmark
