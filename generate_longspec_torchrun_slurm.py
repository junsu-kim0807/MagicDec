#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate Slurm jobs that run longspec directly via torchrun.

It uses README's command style (ENABLE_INTRA_NODE_COMM=1 torchrun ... tests/*/longspec_benchmark.py).

Outputs (stdout/stderr) are stored under:
  ./results/spec_decode/<method>/<batch_size>/<dataset>/<draft__TO__target>/
"""

from __future__ import annotations

import argparse
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path


# User/cluster layout (overrideable)
REPO_DIR = os.environ.get("REPO_DIR", str(Path(__file__).resolve().parent))
VENV_DIR = os.environ.get("VENV_DIR", "/home/jhwoo36/scratch/venvs/vllm")

REPO_ROOT = Path(REPO_DIR)
JOBS_ROOT = REPO_ROOT / "scripts" / "jobs" / "spec_decode"
LOGS_ROOT = REPO_ROOT / "scripts" / "logs" / "spec_decode"
RESULTS_ROOT = REPO_ROOT / "results" / "spec_decode"

# Benchmark scripts expect local checkpoint paths as .pth files.
# README examples use /scratch/models/<org>/<repo>/model.pth.
# In some environments (e.g. only HuggingFace cache exists), we resolve
# candidate weight files from HF snapshots and then symlink them into a
# temporary directory so that:
#   - the checkpoint file path is stable for the engine
#   - the checkpoint parent directory name matches transformer_config keys
MODELS_ROOT = "/scratch/models"


@dataclass(frozen=True)
class PairConfig:
    pair_id: str
    draft_model: str
    target_model: str
    tp_size: int
    gpu_count: int
    note: str = ""


def sanitize_for_path(s: str) -> str:
    s = s.strip().replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9._+@=,]+", "_", s)
    return s.strip("._")


def pair_slug(draft_model: str, target_model: str) -> str:
    return f"{sanitize_for_path(draft_model)}__TO__{sanitize_for_path(target_model)}"


def ckpt_candidates_from_hf_repo(repo_id: str) -> list[str]:
    """
    Map HF-style repo_id (e.g. Qwen/Qwen3-0.6B) to candidate local checkpoint paths.

    We try both:
      1) Converted local checkpoints (e.g. /scratch/models/<org>/<repo>/model.pth)
      2) HuggingFace cache snapshots (e.g. $HF_HUB_CACHE/models--<org>--<repo>/snapshots/*/model.pth)

    We also add a couple common normalizations (remove -Instruct, remove Meta- prefix)
    because checkpoint conversion outputs may differ by naming.
    """
    if "/" not in repo_id:
        raise ValueError(f"Expected HF repo_id like org/name, got: {repo_id!r}")
    org, name = repo_id.split("/", 1)
    org_lower = org.lower()

    candidates: list[str] = []

    def add(n: str) -> None:
        # 1) Local converted checkpoint candidates.
        local_pth = f"{MODELS_ROOT}/{org}/{n}/model.pth"
        if local_pth not in candidates:
            candidates.append(local_pth)
        local_pth_lower = f"{MODELS_ROOT}/{org_lower}/{n}/model.pth"
        if local_pth_lower not in candidates:
            candidates.append(local_pth_lower)

        # 2) HF cache snapshot candidates.
        # NOTE: we keep '$HF_HUB_CACHE' as a bash variable so job scripts
        # can override it (default is set in job body).
        snap_bases = [
            f"$HF_HUB_CACHE/models--{org}--{n}/snapshots/*",
            f"$HF_HUB_CACHE/models--{org_lower}--{n}/snapshots/*",
        ]
        for fname in [
            "model.pth",
            "model.pt",
            "pytorch_model.bin",
            "pytorch_model.pt",
            "pytorch_model.pth",
        ]:
            for snap_base in snap_bases:
                p = f"{snap_base}/{fname}"
                if p not in candidates:
                    candidates.append(p)

    add(name)
    if name.endswith("-Instruct"):
        add(name[: -len("-Instruct")])
    if name.startswith("Meta-"):
        add(name[len("Meta-") :])
    if name.startswith("Meta-") and name.endswith("-Instruct"):
        base = name[len("Meta-") : -len("-Instruct")]
        add(base)
    return candidates


def ranks_str(n: int) -> str:
    return " ".join(str(i) for i in range(n))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def shquote(value: str) -> str:
    """Bash-safe single-quote wrapper."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


def job_header(job_name: str, gpu_count: int, time_limit: str, out_dir: Path, script_path: Path) -> str:
    # NOTE: --output/--error directory must exist at submission time, so generator creates out_dir.
    out_dir = out_dir.resolve()
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
#SBATCH --output={out_dir / (job_name + ".out")}
#SBATCH --error={out_dir / (job_name + ".err")}

set -euo pipefail

module load python/3.12 cuda/12.9 arrow/21.0.0

export REPO_DIR="{REPO_DIR}"
export VENV_DIR="{VENV_DIR}"

cd "${{REPO_DIR}}"
source "${{VENV_DIR}}/bin/activate"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

# If your cluster requires HF auth, export HF_TOKEN before sbatch.
export HF_TOKEN="${{HF_TOKEN:-}}"

echo "Job: {job_name}"
echo "Script: {script_path}"
echo "PWD: $(pwd)"
"""


def runner_for_method(method: str) -> str:
    if method == "longspec_snap":
        return str(REPO_ROOT / "tests" / "SnapKV" / "longspec_benchmark.py")
    if method == "longspec_stream":
        return str(REPO_ROOT / "tests" / "StreamingLLM" / "longspec_benchmark.py")
    raise ValueError(f"Unknown method: {method}")


PAIRS: list[PairConfig] = [
    PairConfig(
        pair_id="llama32_1b_to_llama31_70b",
        draft_model="meta-llama/Llama-3.2-1B-Instruct",
        target_model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        tp_size=2,
        gpu_count=2,
        note="Llama 1B draft -> 70B target",
    ),
    PairConfig(
        pair_id="llama32_3b_to_llama31_70b",
        draft_model="meta-llama/Llama-3.2-3B-Instruct",
        target_model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        tp_size=2,
        gpu_count=2,
        note="Llama 3B draft -> 70B target",
    ),
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
    PairConfig(
        pair_id="qwen3_0p6b_to_qwen3_30b_a3b",
        draft_model="Qwen/Qwen3-0.6B",
        target_model="Qwen/Qwen3-30B-A3B",
        tp_size=2,
        gpu_count=2,
        note="Qwen3 0.6B draft -> 30B-A3B target",
    ),
    PairConfig(
        pair_id="qwen3_4b_to_qwen3_30b_a3b",
        draft_model="Qwen/Qwen3-4B",
        target_model="Qwen/Qwen3-30B-A3B",
        tp_size=2,
        gpu_count=2,
        note="Qwen3 4B draft -> 30B-A3B target",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Slurm jobs for longspec torchrun.")
    parser.add_argument("--batch-sizes", default="64", help="Comma-separated batch sizes, e.g. '1,4,16'.")
    parser.add_argument("--datasets", nargs="*", default=["gov_report", "qmsum"], help="gov_report qmsum only.")

    parser.add_argument("--prefix-len", type=int, default=16032, help="Prefill length.")
    parser.add_argument("--draft-budget", type=int, default=257, help="Draft KV budget (SnapKV/StreamingLLM asserts alignment).")

    parser.add_argument("--gov-report-max-len", type=int, default=16128, help="max_len for gov_report.")
    parser.add_argument("--qmsum-max-len", type=int, default=16128, help="max_len for qmsum.")

    parser.add_argument("--gamma", type=int, default=3, help="Gamma (keep 3 by request).")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")

    parser.add_argument(
        "--profile-time",
        action="store_true",
        help="If set, also pass --benchmark to longspec_benchmark.py to measure timing.",
    )
    parser.add_argument("--compile", action="store_true", default=True, help="Always pass --compile.")
    return parser.parse_args()


def time_limit_for_dataset(dataset_short: str) -> str:
    # Basic heuristic; adjust to your environment.
    if dataset_short in ("gov_report", "qmsum"):
        return "04:00:00"
    return "03:00:00"


def main() -> None:
    args = parse_args()

    ensure_dir(JOBS_ROOT)
    ensure_dir(LOGS_ROOT)
    ensure_dir(RESULTS_ROOT)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    datasets = args.datasets

    # Dataset args expected by longspec_benchmark.py:
    #   --dataset longbench-v1:<task_name>
    dataset_to_task = {"gov_report": "gov_report", "govreport": "gov_report", "qmsum": "qmsum"}
    dataset_to_max_len = {
        "gov_report": args.gov_report_max_len,
        "govreport": args.gov_report_max_len,
        "qmsum": args.qmsum_max_len,
    }

    methods = ["longspec_snap", "longspec_stream"]

    written: list[Path] = []
    for method in methods:
        runner = runner_for_method(method)

        for bs in batch_sizes:
            for ds_short in datasets:
                if ds_short not in dataset_to_task:
                    raise SystemExit(f"Unsupported dataset: {ds_short}. Expected gov_report/qmsum.")

                task_name = dataset_to_task[ds_short]
                dataset_arg = f"longbench-v1:{task_name}"
                max_len = dataset_to_max_len[ds_short]
                time_limit = time_limit_for_dataset(ds_short)

                for pair in PAIRS:
                    slug = pair_slug(pair.draft_model, pair.target_model)
                    draft_display_name = pair.draft_model.split("/", 1)[1]
                    target_display_name = pair.target_model.split("/", 1)[1]

                    out_dir = RESULTS_ROOT / method / str(bs) / ds_short / slug
                    ensure_dir(out_dir)

                    job_name = f"{method}_{pair.pair_id}_{ds_short}_b{bs}"
                    slurm_path = JOBS_ROOT / method / str(bs) / pair.pair_id / f"{ds_short}.slurm"
                    ensure_dir(slurm_path.parent)

                    # TP wiring:
                    # - torchrun nproc_per_node = pair.gpu_count
                    # - --rank_group uses GPUs [0..gpu_count-1]
                    # - --draft_rank_group uses GPUs [0..tp_size-1]
                    nproc = pair.gpu_count
                    rank_group = ranks_str(nproc)
                    draft_rank_group = ranks_str(pair.tp_size)

                    draft_cands = ckpt_candidates_from_hf_repo(pair.draft_model)
                    target_cands = ckpt_candidates_from_hf_repo(pair.target_model)
                    # Candidate paths can include globs (*). Don't quote them.
                    draft_cands_bash = " ".join(draft_cands)
                    target_cands_bash = " ".join(target_cands)

                    cmd_parts: list[str] = [
                        "ENABLE_INTRA_NODE_COMM=1",
                        "torchrun --standalone",
                        f"--nproc_per_node={nproc}",
                        shquote(runner),
                        '--target "$TARGET_CKPT"',
                        '--model "$DRAFT_CKPT"',
                        f"--model_name {shquote(pair.target_model)}",
                        f"--rank_group {rank_group}",
                        f"--draft_rank_group {draft_rank_group}",
                        f"--gamma {args.gamma}",
                        f"--B {bs}",
                        f"--prefix_len {args.prefix_len}",
                        f"--max_len {max_len}",
                        f"--draft_budget {args.draft_budget}",
                        f"--dataset {shquote(dataset_arg)}",
                    ]

                    if args.profile_time:
                        cmd_parts.append("--benchmark")
                    if args.compile:
                        cmd_parts.append("--compile")

                    cmd = " \\\n  ".join(cmd_parts)

                    header = job_header(
                        job_name=job_name,
                        gpu_count=pair.gpu_count,
                        time_limit=time_limit,
                        out_dir=out_dir,
                        script_path=slurm_path,
                    )

                    body = f"""
set -euo pipefail
: "${{HF_HUB_CACHE:=$HOME/scratch/.cache/huggingface_hub}}"
TMP_CKPT_ROOT="${{TMP_CKPT_ROOT:-$HOME/.magicdec_ckpt_tmp}}"
mkdir -p "$TMP_CKPT_ROOT"

DRAFT_REPO_DISPLAY={shquote(draft_display_name)}
TARGET_REPO_DISPLAY={shquote(target_display_name)}

# Resolve checkpoints on the cluster (first existing candidate wins).
RAW_DRAFT_CKPT=""
for c in {draft_cands_bash}; do
  if [ -f "$c" ]; then RAW_DRAFT_CKPT="$c"; break; fi
done

RAW_TARGET_CKPT=""
for c in {target_cands_bash}; do
  if [ -f "$c" ]; then RAW_TARGET_CKPT="$c"; break; fi
done

if [ -z "$RAW_DRAFT_CKPT" ] || [ -z "$RAW_TARGET_CKPT" ]; then
  echo "Missing checkpoint(s)." >&2
  exit 1
fi

DRAFT_CKPT="$TMP_CKPT_ROOT/$DRAFT_REPO_DISPLAY/model.pth"
TARGET_CKPT="$TMP_CKPT_ROOT/$TARGET_REPO_DISPLAY/model.pth"

mkdir -p "$TMP_CKPT_ROOT/$DRAFT_REPO_DISPLAY" "$TMP_CKPT_ROOT/$TARGET_REPO_DISPLAY"
ln -sf "$RAW_DRAFT_CKPT" "$DRAFT_CKPT"
ln -sf "$RAW_TARGET_CKPT" "$TARGET_CKPT"

echo "RAW_DRAFT_CKPT=$RAW_DRAFT_CKPT"
echo "RAW_TARGET_CKPT=$RAW_TARGET_CKPT"
echo "DRAFT_CKPT=$DRAFT_CKPT"
echo "TARGET_CKPT=$TARGET_CKPT"

echo "CMD:"
echo {cmd!r}

{cmd}
"""

                    content = header + "\n" + body.strip() + "\n"
                    slurm_path.write_text(content, encoding="utf-8")
                    slurm_path.chmod(slurm_path.stat().st_mode | stat.S_IXUSR)

                    written.append(slurm_path)
                    print(f"Wrote {slurm_path}")

    # Submit helper
    submit_path = REPO_ROOT / "scripts" / "submit_longspec_torchrun_jobs.sh"
    ensure_dir(submit_path.parent)
    with open(submit_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for sp in written:
            f.write(f"echo sbatch {sp}\n")
            f.write(f"sbatch {sp}\n")
    submit_path.chmod(submit_path.stat().st_mode | stat.S_IXUSR)
    print(f"Submit script written: {submit_path}")


if __name__ == "__main__":
    main()

