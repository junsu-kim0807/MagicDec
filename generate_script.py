#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate Slurm jobs for MagicDec longspec experiments.

It writes jobs under:
  ./scripts/jobs/spec_decode/<method>/<batch_size>/<pair_id>/<dataset>.slurm

It sets run output root (response + profile) under:
  ./results/spec_decode/<method>/<batch_size>/<dataset>/<draft__TO__target>/

Example:
  python generate_script.py --batch --datasets gov_report qmsum --profile-time
"""

from __future__ import annotations

import argparse
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

# Per user request (from their cluster layout)
REPO_DIR = "/home/junsuk87/scratch/specdec/MagicDec"
VENV_DIR = "/home/junsuk87/scratch/venvs/vllm"

# Always generate for the cluster paths.
REPO_ROOT = Path(REPO_DIR)

JOBS_ROOT = REPO_ROOT / "scripts" / "jobs" / "spec_decode"
LOGS_ROOT = REPO_ROOT / "scripts" / "logs" / "spec_decode"
RESULTS_ROOT = REPO_ROOT / "results" / "spec_decode"

# Assumption: there exists an evaluation/metrics runner compatible with these flags.
# If your repo uses a different script path/flags, edit RUN_SCRIPT and build_python_command().
RUN_SCRIPT = "scripts/run_spec_decode_metrics.py"

# Only run longspec jobs (per request)
METHOD = "longspec"


@dataclass(frozen=True)
class PairConfig:
    pair_id: str
    draft_model: str
    target_model: str
    tp_size: int
    gpu_count: int
    note: str = ""


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    max_new_tokens: int


PAIRS: list[PairConfig] = [
    # PairConfig(
    #     pair_id="llama32_1b_to_llama32_3b",
    #     draft_model="meta-llama/Llama-3.2-1B-Instruct",
    #     target_model="meta-llama/Llama-3.2-3B-Instruct",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="Llama small->small",
    # ),
    # PairConfig(
    #     pair_id="llama32_1b_to_llama31_8b",
    #     draft_model="meta-llama/Llama-3.2-1B-Instruct",
    #     target_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="Llama 1B draft -> 8B target",
    # ),
    PairConfig(
        pair_id="llama32_1b_to_llama33_70b",
        draft_model="meta-llama/Llama-3.2-1B-Instruct",
        target_model="meta-llama/Meta-Llama-3.3-70B-Instruct",
        tp_size=2,
        gpu_count=2,
        note="Llama 1B draft -> 70B target",
    ),
    # PairConfig(
    #     pair_id="llama32_3b_to_llama31_8b",
    #     draft_model="meta-llama/Llama-3.2-3B-Instruct",
    #     target_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="Llama 3B draft -> 8B target",
    # ),
    PairConfig(
        pair_id="llama32_3b_to_llama33_70b",
        draft_model="meta-llama/Llama-3.2-3B-Instruct",
        target_model="meta-llama/Meta-Llama-3.3-70B-Instruct",
        tp_size=2,
        gpu_count=2,
        note="Llama 3B draft -> 70B target",
    ),
    # PairConfig(
    #     pair_id="llama31_8b_to_llama33_70b",
    #     draft_model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    #     target_model="meta-llama/Meta-Llama-3.3-70B-Instruct",
    #     tp_size=4,
    #     gpu_count=4,
    #     note="Llama 8B draft -> 70B target",
    # ),
    # PairConfig(
    #     pair_id="deepseekcoder_1p3b_to_6p7b",
    #     draft_model="deepseek-ai/deepseek-coder-1.3b-instruct",
    #     target_model="deepseek-ai/deepseek-coder-6.7b-instruct",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="DeepSeek Coder 1.3B draft -> 6.7B target",
    # ),
    PairConfig(
        pair_id="deepseekcoder_1p3b_to_33b",
        draft_model="deepseek-ai/deepseek-coder-1.3b-instruct",
        target_model="deepseek-ai/deepseek-coder-33b-instruct",
        tp_size=2,
        gpu_count=2,
        note="DeepSeek Coder 1.3B draft -> 33B target",
    ),
    PairConfig(
        pair_id="deepseekcoder_6p7b_to_33b",
        draft_model="deepseek-ai/deepseek-coder-6.7b-instruct",
        target_model="deepseek-ai/deepseek-coder-33b-instruct",
        tp_size=2,
        gpu_count=2,
        note="DeepSeek Coder 6.7B draft -> 33B target",
    ),
    # Qwen3 speculative decoding pairs
    # PairConfig(
    #     pair_id="qwen3_0p6b_to_qwen3_4b",
    #     draft_model="Qwen/Qwen3-0.6B",
    #     target_model="Qwen/Qwen3-4B",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="Qwen3 0.6B draft -> 4B target",
    # ),
    # PairConfig(
    #     pair_id="qwen3_0p6b_to_qwen3_8b",
    #     draft_model="Qwen/Qwen3-0.6B",
    #     target_model="Qwen/Qwen3-8B",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="Qwen3 0.6B draft -> 8B target",
    # ),
    PairConfig(
        pair_id="qwen3_0p6b_to_qwen3_30b_a3b",
        draft_model="Qwen/Qwen3-0.6B",
        target_model="Qwen/Qwen3-30B-A3B",
        tp_size=2,
        gpu_count=2,
        note="Qwen3 0.6B draft -> 30B-A3B target",
    ),
    # PairConfig(
    #     pair_id="qwen3_4b_to_qwen3_8b",
    #     draft_model="Qwen/Qwen3-4B",
    #     target_model="Qwen/Qwen3-8B",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="Qwen3 4B draft -> 8B target",
    # ),
    PairConfig(
        pair_id="qwen3_4b_to_qwen3_30b_a3b",
        draft_model="Qwen/Qwen3-4B",
        target_model="Qwen/Qwen3-30B-A3B",
        tp_size=2,
        gpu_count=2,
        note="Qwen3 4B draft -> 30B-A3B target",
    ),
    # PairConfig(
    #     pair_id="qwen3_8b_to_qwen3_30b_a3b",
    #     draft_model="Qwen/Qwen3-8B",
    #     target_model="Qwen/Qwen3-30B-A3B",
    #     tp_size=2,
    #     gpu_count=2,
    #     note="Qwen3 8B draft -> 30B-A3B target",
    # ),

    # PairConfig(
    #     pair_id="qwen25_0p5b_to_qwen3_4b_instruct_2507",
    #     draft_model="Qwen/Qwen2.5-0.5B-Instruct",
    #     target_model="Qwen/Qwen3-4B-Instruct-2507",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="Qwen2.5 0.5B draft -> Qwen3 4B Instruct 2507",
    # ),
    PairConfig(
        pair_id="qwen25_0p5b_to_qwen3_30b_a3b_instruct_2507",
        draft_model="Qwen/Qwen2.5-0.5B-Instruct",
        target_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        tp_size=2,
        gpu_count=2,
        note="Qwen2.5 0.5B draft -> Qwen3 30B-A3B Instruct 2507",
    ),
    # PairConfig(
    #     pair_id="qwen3_0p6b_to_qwen3_4b_instruct_2507",
    #     draft_model="Qwen/Qwen3-0.6B",
    #     target_model="Qwen/Qwen3-4B-Instruct-2507",
    #     tp_size=1,
    #     gpu_count=1,
    #     note="Qwen3 0.6B draft -> Qwen3 4B Instruct 2507",
    # ),
    PairConfig(
        pair_id="qwen3_0p6b_to_qwen3_30b_a3b_instruct_2507",
        draft_model="Qwen/Qwen3-0.6B",
        target_model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        tp_size=2,
        gpu_count=2,
        note="Qwen3 0.6B draft -> Qwen3 30B-A3B Instruct 2507",
    ),
]


def sanitize_for_path(s: str) -> str:
    s = s.strip().replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9._+@=,]+", "_", s)
    return s.strip("._")


def pair_slug(draft_model: str, target_model: str) -> str:
    return f"{sanitize_for_path(draft_model)}__TO__{sanitize_for_path(target_model)}"


def shquote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def ensure_dirs() -> None:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


def time_limit_for_dataset(dataset: DatasetConfig) -> str:
    # Conservative default; adjust as needed for your cluster.
    if dataset.name in ("gov_report", "qmsum"):
        return "04:00:00"
    return "03:00:00"


def job_header(job_name: str, gpu_count: int, time_limit: str, log_dir: Path) -> str:
    log_dir.mkdir(parents=True, exist_ok=True)
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=rrg-pnair_gpu
#SBATCH --qos=rrg-pnair
#SBATCH --gres=gpu:h100:{gpu_count}
#SBATCH --mem-per-gpu=80G
#SBATCH --time={time_limit}
#SBATCH --output={log_dir / (job_name + ".out")}
#SBATCH --error={log_dir / (job_name + ".err")}

set -euo pipefail

module load python/3.12 cuda/12.9 arrow/21.0.0

REPO_DIR={shquote(REPO_DIR)}
VENV_DIR={shquote(VENV_DIR)}

cd "${{REPO_DIR}}"
source "${{VENV_DIR}}/bin/activate"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

# If your models require HF auth, export HF_TOKEN before sbatch.
export HF_TOKEN="${{HF_TOKEN:-}}"
export HF_HOME="${{HF_HOME:-$HOME/scratch/.cache}}"
export TRANSFORMERS_CACHE="${{TRANSFORMERS_CACHE:-$HOME/scratch/.cache/transformers}}"
export HUGGINGFACE_HUB_CACHE="${{HUGGINGFACE_HUB_CACHE:-$HOME/scratch/.cache/huggingface_hub}}"
export HF_DATASETS_CACHE="${{HF_DATASETS_CACHE:-$HOME/scratch/.cache/datasets}}"
"""


def ckpt_candidates_from_hf_repo(repo_id: str) -> list[str]:
    """
    Return candidate weight *file* paths in HF cache snapshots.

    The returned paths intentionally keep '$HUGGINGFACE_HUB_CACHE' so they
    can be expanded inside the generated Slurm job.
    """
    if "/" not in repo_id:
        raise ValueError(f"Expected HF repo_id like org/name, got: {repo_id!r}")
    org, name = repo_id.split("/", 1)
    org_lower = org.lower()

    # HF snapshot uses model repo name as stored in the cache. Sometimes
    # conversion outputs are produced under normalized names, so we try a
    # couple variants.
    name_variants: list[str] = [name]
    if name.endswith("-Instruct"):
        name_variants.append(name[: -len("-Instruct")])
    if name.startswith("Meta-"):
        name_variants.append(name[len("Meta-") :])
    if name.startswith("Meta-") and name.endswith("-Instruct"):
        base = name[len("Meta-") : -len("-Instruct")]
        name_variants.append(base)

    org_variants = [org, org_lower] if org_lower != org else [org]

    candidates: list[str] = []
    weight_files = [
        "model.pth",
        "model.pt",
        "pytorch_model.bin",
        "pytorch_model.pt",
        "pytorch_model.pth",
    ]

    for org_v in org_variants:
        for n_v in name_variants:
            snap_base = f"$HUGGINGFACE_HUB_CACHE/models--{org_v}--{n_v}/snapshots/*"
            for wf in weight_files:
                p = f"{snap_base}/{wf}"
                if p not in candidates:
                    candidates.append(p)
    return candidates


def dest_name_variants(name: str) -> list[str]:
    # Where to place symlinked `model.pth` under MODELS_ROOT/<org>/<name>/.
    out: list[str] = [name]
    if name.endswith("-Instruct"):
        out.append(name[: -len("-Instruct")])
    if name.startswith("Meta-"):
        out.append(name[len("Meta-") :])
    if name.startswith("Meta-") and name.endswith("-Instruct"):
        out.append(name[len("Meta-") : -len("-Instruct")])
    # De-dup while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def dest_org_variants(org: str) -> list[str]:
    org_lower = org.lower()
    return [org] + ([org_lower] if org_lower != org else [])


def build_python_command(
    *,
    method: str,
    pair: PairConfig,
    dataset: DatasetConfig,
    batch_size: int,
    gpu_mem_util: float,
    max_model_len: int,
    dtype: str,
    seed: int,
    warmup_iters: int,
    warmup_max_tokens: int,
    num_spec_tokens: int,
    profile_time: bool,
    results_root: Path,
    verbose: bool,
) -> str:
    out_root = results_root
    args = [
        f"python -u {shquote(RUN_SCRIPT)}",
        f"--draft-model {shquote(pair.draft_model)}",
        f"--target-models {shquote(pair.target_model)}",
        f"--tp-map {shquote(f'{pair.target_model}={pair.tp_size}')}",
        f"--datasets {shquote(dataset.name)}",
        f"--batch-sizes {batch_size}",
        f"--num-spec-tokens {num_spec_tokens}",
        f"--gpu-memory-utilization {gpu_mem_util:.2f}",
        f"--max-model-len {max_model_len}",
        f"--dtype {dtype}",
        f"--seed {seed}",
        f"--warmup-iters {warmup_iters}",
        f"--warmup-max-tokens {warmup_max_tokens}",
        f"--results-root {shquote(str(out_root))}",
        "--trust-remote-code",
        "--enable-chunked-prefill",
        f"--method {shquote(method)}",
        "--benchmark",
    ]
    if profile_time:
        args.append("--profile-time")
    if verbose:
        args.append("--verbose")

    # Dataset-specific max_new_tokens switches (kept consistent with the example script)
    if dataset.name == "gov_report":
        args.append(f"--gov-report-max-new-tokens {dataset.max_new_tokens}")
    elif dataset.name == "qmsum":
        args.append(f"--qmsum-max-new-tokens {dataset.max_new_tokens}")

    return " \\\n  ".join(args)


def write_job_script(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def parse_pairs(pair_args: list[list[str]]) -> list[PairConfig]:
    pairs: list[PairConfig] = []
    for (pair_id, draft_model, target_model, tp_size, gpu_count) in pair_args:
        pairs.append(
            PairConfig(
                pair_id=pair_id,
                draft_model=draft_model,
                target_model=target_model,
                tp_size=int(tp_size),
                gpu_count=int(gpu_count),
            )
        )
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Slurm jobs for MagicDec.")
    parser.add_argument("--batch", action="store_true", help="Generate jobs for default batch sizes.")
    parser.add_argument(
        "--batch-sizes",
        default="1,4,16,64,256",
        help="Comma-separated batch sizes.",
    )
    parser.add_argument("--datasets", nargs="+", default=["gov_report", "qmsum"], help="Dataset names.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--warmup-max-tokens", type=int, default=32)
    parser.add_argument("--num-spec-tokens", type=int, default=7)
    parser.add_argument("--profile-time", action="store_true", help="Enable profiler output.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=[],
        help="Optional subset of pair_id values (if empty, uses all PAIRS).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    dataset_map = {
        "gov_report": DatasetConfig("gov_report", max_new_tokens=512),
        "qmsum": DatasetConfig("qmsum", max_new_tokens=512),
        # Optional defaults for other datasets if you extend later.
        "aime2025": DatasetConfig("aime2025", max_new_tokens=256),
        "codeelo": DatasetConfig("codeelo", max_new_tokens=1024),
        "longbench-v2": DatasetConfig("longbench-v2", max_new_tokens=1024),
    }

    datasets: list[DatasetConfig] = []
    for name in args.datasets:
        if name not in dataset_map:
            raise SystemExit(f"Unknown dataset: {name}. Supported: {sorted(dataset_map.keys())}")
        datasets.append(dataset_map[name])

    # Per request: when generating batch jobs, generate codeelo only.
    if args.batch:
        datasets = [dataset_map["codeelo"]]

    selected_pairs = []
    pair_filter = set(args.pairs)
    for p in PAIRS:
        if not pair_filter or p.pair_id in pair_filter:
            selected_pairs.append(p)
    if not selected_pairs:
        raise SystemExit(f"--pairs did not match any PAIRS: {sorted(pair_filter)}")

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]

    written: list[Path] = []
    method = METHOD

    for bs in batch_sizes:
        for dataset in datasets:
            for pair in selected_pairs:
                slug = pair_slug(pair.draft_model, pair.target_model)
                results_dir = RESULTS_ROOT / method / str(bs) / dataset.name / slug

                job_name = f"{method}_{pair.pair_id}_{dataset.name}_b{bs}"
                time_limit = time_limit_for_dataset(dataset)

                jobs_path = JOBS_ROOT / method / str(bs) / pair.pair_id / f"{dataset.name}.slurm"
                log_dir = LOGS_ROOT / method / str(bs) / pair.pair_id

                header = job_header(
                    job_name=job_name,
                    gpu_count=pair.gpu_count,
                    time_limit=time_limit,
                    log_dir=log_dir,
                )
                command = build_python_command(
                    method=method,
                    pair=pair,
                    dataset=dataset,
                    batch_size=bs,
                    gpu_mem_util=args.gpu_memory_utilization,
                    max_model_len=args.max_model_len,
                    dtype=args.dtype,
                    seed=args.seed,
                    warmup_iters=args.warmup_iters,
                    warmup_max_tokens=args.warmup_max_tokens,
                    num_spec_tokens=args.num_spec_tokens,
                    profile_time=args.profile_time,
                    results_root=results_dir,
                    verbose=args.verbose,
                )

                # Build HF-cache -> /scratch/models symlink plan for this pair.
                draft_cands = ckpt_candidates_from_hf_repo(pair.draft_model)
                target_cands = ckpt_candidates_from_hf_repo(pair.target_model)
                draft_cands_bash = " ".join(draft_cands)
                target_cands_bash = " ".join(target_cands)

                draft_org, draft_name = pair.draft_model.split("/", 1)
                target_org, target_name = pair.target_model.split("/", 1)

                draft_symlink_lines_list: list[str] = []
                for org_v in dest_org_variants(draft_org):
                    for n_v in dest_name_variants(draft_name):
                        draft_symlink_lines_list.append(f'mkdir -p "$MODELS_ROOT/{org_v}/{n_v}"')
                        draft_symlink_lines_list.append(
                            f'ln -sf "$RAW_DRAFT_CKPT_FILE" "$MODELS_ROOT/{org_v}/{n_v}/model.pth"'
                        )
                draft_symlink_lines = "\n".join(draft_symlink_lines_list)

                target_symlink_lines_list: list[str] = []
                for org_v in dest_org_variants(target_org):
                    for n_v in dest_name_variants(target_name):
                        target_symlink_lines_list.append(f'mkdir -p "$MODELS_ROOT/{org_v}/{n_v}"')
                        target_symlink_lines_list.append(
                            f'ln -sf "$RAW_TARGET_CKPT_FILE" "$MODELS_ROOT/{org_v}/{n_v}/model.pth"'
                        )
                target_symlink_lines = "\n".join(target_symlink_lines_list)

                body = f"""
mkdir -p {shquote(str(results_dir))}
echo "JOB: {job_name}"
echo "RESULTS_DIR: {shquote(str(results_dir))}"

# Resolve HF snapshot weights and symlink into /scratch/models.
# This supports environments where only HF cache exists (no pre-converted model.pth).
MODELS_ROOT="${{MODELS_ROOT:-/scratch/models}}"

RAW_DRAFT_CKPT_FILE=""
for c in {draft_cands_bash}; do
  if [ -f "$c" ]; then RAW_DRAFT_CKPT_FILE="$c"; break; fi
done
if [ -z "$RAW_DRAFT_CKPT_FILE" ]; then
  echo "Missing draft weight file for {pair.draft_model}" >&2
  exit 1
fi

{draft_symlink_lines}

RAW_TARGET_CKPT_FILE=""
for c in {target_cands_bash}; do
  if [ -f "$c" ]; then RAW_TARGET_CKPT_FILE="$c"; break; fi
done
if [ -z "$RAW_TARGET_CKPT_FILE" ]; then
  echo "Missing target weight file for {pair.target_model}" >&2
  exit 1
fi

{target_symlink_lines}

{command}
"""

                content = header + "\n" + body.strip() + "\n"
                write_job_script(jobs_path, content)
                written.append(jobs_path)
                print(f"Wrote {jobs_path}")

    # Convenience submit script
    submit_path = REPO_ROOT / "scripts" / "submit_spec_decode.sh"
    submit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(submit_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for sp in written:
            f.write(f"echo sbatch {sp}\n")
            f.write(f"sbatch {sp}\n")
    submit_path.chmod(submit_path.stat().st_mode | stat.S_IXUSR)
    print(f"Submit script: {submit_path}")


if __name__ == "__main__":
    main()

