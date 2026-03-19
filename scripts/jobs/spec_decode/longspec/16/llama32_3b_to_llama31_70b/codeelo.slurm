#!/bin/bash
#SBATCH --job-name=longspec_llama32_3b_to_llama31_70b_codeelo_b16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=rrg-pnair_gpu
#SBATCH --qos=rrg-pnair
#SBATCH --gres=gpu:h100:4
#SBATCH --mem-per-gpu=80G
#SBATCH --time=03:00:00
#SBATCH --output=/home/junsuk87/scratch/specdec/MagicDec/scripts/logs/spec_decode/longspec/16/llama32_3b_to_llama31_70b/longspec_llama32_3b_to_llama31_70b_codeelo_b16.out
#SBATCH --error=/home/junsuk87/scratch/specdec/MagicDec/scripts/logs/spec_decode/longspec/16/llama32_3b_to_llama31_70b/longspec_llama32_3b_to_llama31_70b_codeelo_b16.err

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

mkdir -p '/home/junsuk87/scratch/specdec/MagicDec/results/spec_decode/longspec/16/codeelo/meta_llama__Llama_3.2_3B_Instruct__TO__meta_llama__Meta_Llama_3.1_70B_Instruct'
echo "JOB: longspec_llama32_3b_to_llama31_70b_codeelo_b16"
echo "RESULTS_DIR: '/home/junsuk87/scratch/specdec/MagicDec/results/spec_decode/longspec/16/codeelo/meta_llama__Llama_3.2_3B_Instruct__TO__meta_llama__Meta_Llama_3.1_70B_Instruct'"

python -u 'scripts/run_spec_decode_metrics.py' \
  --draft-model 'meta-llama/Llama-3.2-3B-Instruct' \
  --target-models 'meta-llama/Meta-Llama-3.1-70B-Instruct' \
  --tp-map 'meta-llama/Meta-Llama-3.1-70B-Instruct=4' \
  --datasets 'codeelo' \
  --batch-sizes 16 \
  --num-spec-tokens 7 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --dtype auto \
  --seed 0 \
  --warmup-iters 2 \
  --warmup-max-tokens 32 \
  --results-root '/home/junsuk87/scratch/specdec/MagicDec/results/spec_decode/longspec/16/codeelo/meta_llama__Llama_3.2_3B_Instruct__TO__meta_llama__Meta_Llama_3.1_70B_Instruct' \
  --trust-remote-code \
  --enable-chunked-prefill \
  --method 'longspec' \
  --profile-time \
  --benchmark
