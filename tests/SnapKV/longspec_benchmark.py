import time
import os
import glob
import torch
import sys
sys.path.append("..")
from pathlib import Path
import torch.distributed as dist
from MagicDec.Engine.utils import setup_seed, cuda_graph_for_sampling_argmax_batch, sampling_argmax_batch
from MagicDec.Data.data_converter import (
    convert_pg19_dataset,
    convert_aime2025_dataset,
    convert_codeelo_dataset,
    convert_longbench_v1_dataset,
    convert_longbench_v2_dataset,
    repeat_dataset_to_min_len,
)
from transformers import AutoTokenizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import argparse
from MagicDec.Engine.SnapKV.backend import LMBackend
from MagicDec.Engine.SnapKV.backend_draft import LMBackend_Draft


def resolve_checkpoint_arg(arg_value) -> Path:
    """
    Resolve --model/--target value to a local checkpoint file path.

    Supports:
    - direct file path (/path/to/model.pth)
    - HF repo id (org/name), resolved from HF cache snapshots and symlinked to
      ~/.magicdec_ckpt_tmp/<repo_name>/model.pth so engine name matching works.
    """
    p = Path(str(arg_value))
    if p.is_file():
        return p

    raw = str(arg_value)
    looks_like_repo_id = ("/" in raw) and (not raw.endswith((".pth", ".pt", ".bin", ".safetensors")))
    if looks_like_repo_id:
        org, name = raw.split("/", 1)
        hub_cache = (
            os.environ.get("HF_HUB_CACHE")
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
            or (os.path.join(os.environ.get("HF_HOME"), "huggingface_hub") if os.environ.get("HF_HOME") else None)
            or os.path.expanduser("~/scratch/.cache/huggingface_hub")
        )

        org_variants = [org]
        if org.lower() != org:
            org_variants.append(org.lower())
        name_variants = [name]
        if name.endswith("-Instruct"):
            name_variants.append(name[: -len("-Instruct")])
        if name.startswith("Meta-"):
            name_variants.append(name[len("Meta-") :])
        if name.startswith("Meta-") and name.endswith("-Instruct"):
            name_variants.append(name[len("Meta-") : -len("-Instruct")])

        patterns = []
        for ov in org_variants:
            for nv in name_variants:
                base = os.path.join(hub_cache, f"models--{ov}--{nv}", "snapshots", "*")
                patterns.extend([
                    os.path.join(base, "model.pth"),
                    os.path.join(base, "model.pt"),
                    os.path.join(base, "pytorch_model.bin"),
                    os.path.join(base, "pytorch_model.pt"),
                    os.path.join(base, "pytorch_model.pth"),
                ])

        for pat in patterns:
            matches = glob.glob(pat)
            if matches:
                raw_ckpt = Path(matches[0])
                tmp_root = Path(os.environ.get("TMP_CKPT_ROOT", str(Path.home() / ".magicdec_ckpt_tmp")))
                link_dir = tmp_root / name
                link_dir.mkdir(parents=True, exist_ok=True)
                link_path = link_dir / "model.pth"
                if link_path.exists() or link_path.is_symlink():
                    link_path.unlink()
                os.symlink(str(raw_ckpt), str(link_path))
                print(f"Resolved repo id {raw} -> {link_path} (raw: {raw_ckpt})")
                return link_path

        # Fallback: download and convert to model.pth automatically.
        models_root = Path(os.environ.get("MODELS_ROOT", "/scratch/models"))
        local_ckpt_dir = models_root / org / name
        local_ckpt_file = local_ckpt_dir / "model.pth"

        should_sync = dist.is_available() and dist.is_initialized()
        is_rank0 = (not should_sync) or dist.get_rank() == 0

        if is_rank0 and not local_ckpt_file.is_file():
            local_ckpt_dir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {raw} to {local_ckpt_dir} ...")
            from download import hf_download
            hf_download(out_dir=str(local_ckpt_dir), repo_id=raw, hf_token=os.environ.get("HF_TOKEN"))

            print(f"Converting checkpoint to {local_ckpt_file} ...")
            from convert_hf_checkpoint import convert_hf_checkpoint
            convert_hf_checkpoint(checkpoint_dir=local_ckpt_dir, model_name=name)

        if should_sync:
            dist.barrier()

        if local_ckpt_file.is_file():
            print(f"Resolved repo id {raw} -> {local_ckpt_file} (download+convert)")
            return local_ckpt_file

        raise FileNotFoundError(
            f"Could not resolve repo id '{raw}' from HF cache ({hub_cache}), "
            "and automatic download+convert did not produce model.pth."
        )

    raise FileNotFoundError(f"Checkpoint file not found: {p}")


parser = argparse.ArgumentParser(description='Process model configuration and partitions.')
parser.add_argument('--model', type=str, default="/scratch/models/meta-llama/Llama-3.2-1B/model.pth", help='model')
# parser.add_argument('--model', type=Path, default=Path("/scratch/models/meta-llama/Meta-Llama-3.1-8B/model.pth"), help='model')
parser.add_argument('--target', type=str, default="/scratch/models/meta-llama/Meta-Llama-3.1-8B/model.pth", help='model')
parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3.1-8B", help='model name')
parser.add_argument('--dataset', type=str, default="pg19", help='Dataset name.')
parser.add_argument('--draft_budget', type=int, default=-1, help='Dataset end index.')
parser.add_argument('--rank_group', nargs='+', type=int, help='Target group of ranks')
parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
parser.add_argument('--draft_rank_group', nargs='+', type=int, help='Target group of ranks')

parser.add_argument('--gamma', type=int, default=5, help='start')

parser.add_argument('--B', type=int, default=1, help='Batch size.')
parser.add_argument('--prefix_len', type=int, default=4000, help='Prefix length')
parser.add_argument('--max_len', type=int, default=64, help='Generate length')
parser.add_argument('--window_size', type=int, default=32, help='Generate length')

parser.add_argument('--seed', type=int, default=123, help='Random seed.')

parser.add_argument('--printoutput', action='store_true', help='Whether to compile the model.')
parser.add_argument('--benchmark', action='store_true', help='Whether to compile the model.')

# Assert max length <= max context length

args = parser.parse_args()
assert args.prefix_len < args.max_len
assert (args.max_len + 127) // 128 == args.prefix_len // 128 + 1
if args.draft_budget != -1:
    assert (args.prefix_len - args.window_size) % 128 == 0
    assert (args.draft_budget - 1) % 128 == 0

draft_tp = len(args.draft_rank_group) > 1

# Init model parallelism
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
global print
from MagicDec.Engine.tp import init_dist
use_tp = len(args.draft_rank_group) > 1
global_group = None
draft_group = None
rank = 0
if use_tp:
    if draft_tp:
        rank, global_group, draft_group = init_dist(args.draft_rank_group)
    else:
        rank, global_group = init_dist()
    if rank != args.rank_group[0]:
        print = lambda *args, **kwargs: None

# if rank == 0:
#     with open("result.txt", "a") as file:
#         file.write(f"Prefix:{args.prefix_len}; Bsz:{args.B}; Gamma:{args.gamma}\n")

setup_seed(args.seed)
print(f"Using device={DEVICE}")

draft_target_equal = (len(args.draft_rank_group) == len(args.rank_group))

MAX_LEN_TARGET = args.max_len
DTYPE = torch.bfloat16
BATCH_SIZE = args.B
benchmark = args.benchmark
checkpoint_path = resolve_checkpoint_arg(args.target)
draft_checkpoint_path = resolve_checkpoint_arg(args.model)

target_dec_len = args.gamma + 1

# Load target model
engine = LMBackend(dtype=DTYPE, device=DEVICE, dec_len=target_dec_len)
engine.load_model(checkpoint_path, use_tp=use_tp, rank_group = args.rank_group, group=global_group)
if args.compile:
    engine.compile()
engine.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=MAX_LEN_TARGET)

# Load draft model
if not use_tp:
    draft = LMBackend_Draft(dtype=DTYPE, device=DEVICE, draft_budget=args.draft_budget)
    draft.load_model(draft_checkpoint_path, use_tp=False)
    if args.compile:
        draft.compile()
    draft.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=MAX_LEN_TARGET, draft_budget=args.draft_budget)
else:
    if rank in args.draft_rank_group:
        draft = LMBackend_Draft(dtype=DTYPE, device=DEVICE, draft_budget=args.draft_budget)
        draft.load_model(draft_checkpoint_path, use_tp=draft_tp, rank_group=args.draft_rank_group, group=draft_group)
        if args.compile:
            draft.compile()
        draft.setup_caches(max_batch_size=BATCH_SIZE, max_seq_length=MAX_LEN_TARGET, draft_budget=args.draft_budget)
    dist.barrier()

# Load dataset
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
eot_1 = tokenizer.eos_token_id
if tokenizer.unk_token_id is not None:
    eot_2 = tokenizer.unk_token_id
else:
    eot_2 = tokenizer.encode("<|eot_id|>")[-1]
print(f"eot_1: {eot_1}, eot_2: {eot_2}")
if args.dataset == "pg19":
    dataset = convert_pg19_dataset(tokenizer=tokenizer, seq_len=args.prefix_len)
elif args.dataset == "aime2025":
    dataset = convert_aime2025_dataset(tokenizer=tokenizer, seq_len=args.prefix_len)
elif args.dataset == "codeelo":
    dataset = convert_codeelo_dataset(tokenizer=tokenizer, seq_len=args.prefix_len)
elif args.dataset == "longbench-v2":
    dataset = convert_longbench_v2_dataset(tokenizer=tokenizer, seq_len=args.prefix_len)
elif args.dataset.startswith("longbench-v1"):
    # Support: --dataset longbench-v1:narrativeqa
    task_name = "narrativeqa"
    if ":" in args.dataset:
        task_name = args.dataset.split(":", 1)[1]
    dataset = convert_longbench_v1_dataset(tokenizer=tokenizer, seq_len=args.prefix_len, task_name=task_name)
else:
    raise ValueError(f"Unknown dataset {args.dataset}")

dataset = repeat_dataset_to_min_len(dataset, BATCH_SIZE * 2)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
num_eval_steps = min(10, len(dataloader))

total_time = 0.0
num_gen_tokens = 0
target_steps = 0
sample_time = []
if benchmark:
    draft_time = 0.0
    target_time = 0.0
    verify_loop = 0.0
    draft_time_list = []
    verification_time_list = []
    kv_cache_compression_time_list = []

for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
    if step >= num_eval_steps:
        break
    input_ids = batch[0].to(DEVICE)
    terminal = False
    tokens_buffer= torch.zeros((BATCH_SIZE, args.gamma+1), device=DEVICE).long()
    output = torch.zeros(BATCH_SIZE, MAX_LEN_TARGET+1, device=DEVICE).long()
    output[:, :input_ids.shape[1]] = input_ids
    num_nodes = torch.zeros(BATCH_SIZE,device=DEVICE).long()
    num_nodes += input_ids.shape[1]

    tokens_buffer[:, :1] = engine.encode(input_ids=input_ids)[:,-1:]
    kv_cache_compression_time = 0.0
    if benchmark:
        torch.cuda.synchronize()
        kv_t0 = time.time()
    if not use_tp:
        draft.encode(input_ids=input_ids)
    else:
        if rank in args.draft_rank_group:
            draft.encode(input_ids=input_ids)
        dist.barrier()
    if benchmark:
        torch.cuda.synchronize()
        kv_t1 = time.time()
        kv_cache_compression_time = kv_t1 - kv_t0
        if step >= 5:
            kv_cache_compression_time_list.append(kv_cache_compression_time)

        
    next_double = False
    double_buffer = None
    cachelens_update = None

    torch.cuda.synchronize()
    start = time.perf_counter()
    while terminal == False:

        if benchmark:
            torch.cuda.synchronize()
            t1 = time.time()

        # Draft speculation
        if not use_tp or draft_target_equal:
            for i in range(args.gamma):
                if i == 0:
                    if next_double:
                        # The cachelens should increase 1 or 2
                        next_tokens = draft.inference(double_buffer, cachelen_update=cachelens_update)
                        tokens_buffer[:,i+1:i+2] = next_tokens.gather(1, cachelens_update.view(-1,1) - 1)
                        next_double = False
                    else:
                        tokens_buffer[:,i+1:i+2] = draft.inference(tokens_buffer[:, i].view(-1,1))
                    continue
                tokens_buffer[:,i+1:i+2] = draft.inference(tokens_buffer[:, i].view(-1,1))
        else:
            if rank in args.draft_rank_group:
                for i in range(args.gamma):
                    if i == 0:
                        if next_double:
                            # The cachelens should increase 1 or 2
                            next_tokens = draft.inference(double_buffer, cachelen_update=cachelens_update)
                            tokens_buffer[:,i+1:i+2] = next_tokens.gather(1, cachelens_update.view(-1,1) - 1)
                            next_double = False
                        else:
                            tokens_buffer[:,i+1:i+2] = draft.inference(tokens_buffer[:, i].view(-1,1))
                        continue
                    tokens_buffer[:,i+1:i+2] = draft.inference(tokens_buffer[:, i].view(-1,1))
            dist.broadcast(tokens_buffer, src=args.draft_rank_group[0], group=global_group)


        if benchmark:
            torch.cuda.synchronize()
            t2 = time.time()
            delta_draft = t2 - t1
            draft_time += delta_draft
            if step >= 5:
                draft_time_list.append(delta_draft)

        # Target Verification
        target_tokens = engine.inference(tokens_buffer)

        if benchmark:
            torch.cuda.synchronize()
            t3 = time.time()
            delta_verification = t3 - t2
            target_time += delta_verification
            if step >= 5:
                verification_time_list.append(delta_verification)

        target_steps += 1


    # Verify loop
    # Vectorized Verify Loop
        draft_tokens = tokens_buffer[:, 1:args.gamma+1]
        flag_accept_matrix = (target_tokens[:, :args.gamma] == draft_tokens)  # shape: (BATCH_SIZE, gamma)
        eot_condition = ((draft_tokens == eot_1) | (draft_tokens == eot_2))  # shape: (BATCH_SIZE, gamma)

        # Compute accept_flags by considering both the acceptance condition and EOT tokens
        accept_flags_int = (flag_accept_matrix & (~eot_condition)).int()
        accept_flags_cumprod = torch.cumprod(accept_flags_int, dim=1)
        accept_flags_matrix = accept_flags_cumprod.bool()

        # Compute the number of accepted tokens
        accept_nums = accept_flags_matrix.sum(dim=1, keepdim=True) + 1  # shape: (BATCH_SIZE, 1)

        # Check for termination conditions
        condition = (eot_condition & accept_flags_matrix).any(dim=1, keepdim=True)
        if condition.any():
            terminal = True
        
        # Rollback the memory length
        engine.cachelens = engine.cachelens - args.gamma - 1
        engine.paged_kv_last_page_len = engine.paged_kv_last_page_len - args.gamma - 1

        # Put the accepted tokens to output
        positions = torch.arange(output.shape[1], device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
        mask = (positions < (engine.cachelens.view(-1,1) + accept_nums)) & (positions >= engine.cachelens.view(-1, 1))
        positions_buffer = torch.arange(args.gamma+1, device=DEVICE).view(1, -1).repeat(BATCH_SIZE, 1)
        mask_buffer = positions_buffer<accept_nums.view(-1,1)
        output[mask] = tokens_buffer[mask_buffer]

        # Set the cache length to the accepted length
        engine.cachelens += accept_nums.flatten().to(torch.int32)
        engine.paged_kv_last_page_len += accept_nums.flatten().to(torch.int32)

        # print(accept_nums)

        max_limit = torch.full_like(accept_nums, args.gamma, device = DEVICE)
        limited_accept_nums = torch.min(accept_nums, max_limit)
        if not use_tp:
            draft.cachelens = draft.cachelens - args.gamma
            draft.paged_kv_last_page_len = draft.paged_kv_last_page_len - args.gamma
            draft.cachelens += limited_accept_nums.flatten().to(torch.int32)
            draft.paged_kv_last_page_len += limited_accept_nums.flatten().to(torch.int32)
        else:
            if rank in args.draft_rank_group:
                draft.cachelens = draft.cachelens - args.gamma
                draft.paged_kv_last_page_len = draft.paged_kv_last_page_len - args.gamma
                draft.cachelens += limited_accept_nums.flatten().to(torch.int32)
                draft.paged_kv_last_page_len += limited_accept_nums.flatten().to(torch.int32)


        
        # Get the bonus tokens
        indices = accept_nums - 1
        bonus_tokens = target_tokens.gather(1, indices)
        if (bonus_tokens == eot_1).any() or (bonus_tokens == eot_2).any():
            terminal = True
        num_nodes += accept_nums.flatten()

        # Check Number of Nodes + Bonus Token <= max_target_token
        if num_nodes.max() >= args.prefix_len + 80:
            terminal = True
        # Put Bonus tokens to the tokens buffer, and prepare the variables for next itr
        if not terminal:
            tokens_buffer[:, :1] = bonus_tokens
            if accept_nums.max() == args.gamma + 1:
                next_double = True
                double_buffer = torch.zeros((BATCH_SIZE, 2), device=DEVICE).long()
                mask = (accept_nums == (args.gamma + 1)).squeeze()
                double_buffer[:, 0] = torch.where(mask, tokens_buffer[:, -1], bonus_tokens[:, 0])
                double_buffer[:, 1] = torch.where(mask, bonus_tokens[:, 0], torch.full((BATCH_SIZE,), -100, device=bonus_tokens.device))
                non_zero_mask = double_buffer != -100
                double_buffer[:, 1] = torch.where(mask, bonus_tokens[:, 0], torch.zeros_like(bonus_tokens[:, 0]))
                cachelens_update = non_zero_mask.sum(dim=1).flatten()
        
        if not terminal:
            if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                delta_sample = t4 - t3
                verify_loop += delta_sample
                if step >= 5:
                    sample_time.append(delta_sample)
        else:
            for i in range(BATCH_SIZE):
                output[i, num_nodes[i]] = bonus_tokens[i]
            num_nodes += 1
            if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                delta_sample = t4 - t3
                verify_loop += delta_sample
                if step >= 5:
                    sample_time.append(delta_sample)

    torch.cuda.synchronize()
    end=time.perf_counter()
    total_time += end-start
    num_gen_tokens += (num_nodes.sum() - (input_ids.shape[1]+1)*BATCH_SIZE)
    if args.printoutput:
        for i in range(BATCH_SIZE):
            print("Sequence ", i)
            print(tokenizer.decode(output[i, args.prefix_len:num_nodes[i]]))
    print("total time :{:.5f}s, time per iter :{:.5f}s, decoding step: {}, large model step: {}, avg latency: {}".format(total_time, total_time / target_steps, num_gen_tokens, target_steps, total_time / num_gen_tokens * BATCH_SIZE))
    if benchmark:
        print("target time :{:.5f}s, draft time :{:.5f}s, verify loop : {}, avg generate len per sentence: {}".format(target_time/target_steps, draft_time / target_steps, verify_loop/target_steps, num_gen_tokens/target_steps/BATCH_SIZE))

    if step < 5:   # TODO: revert to 10?
        total_time = 0.0
        num_gen_tokens = 0
        target_steps = 0
        if benchmark:
            draft_time = 0.0
            target_time = 0.0
            verify_loop = 0.0
    if use_tp:
        dist.barrier()

print(f"Final tokens per second :{num_gen_tokens/total_time}")
if benchmark:
    if len(sample_time) > 0:
        print(f"sample_time (accept/reject) n={len(sample_time)}, avg={sum(sample_time)/len(sample_time):.6f}s")
    if len(verification_time_list) > 0:
        print(f"verification_time n={len(verification_time_list)}, avg={sum(verification_time_list)/len(verification_time_list):.6f}s")
    if len(draft_time_list) > 0:
        print(f"draft_time n={len(draft_time_list)}, avg={sum(draft_time_list)/len(draft_time_list):.6f}s")
    if len(kv_cache_compression_time_list) > 0:
        print(f"kv_cache_compression_time per outer step n={len(kv_cache_compression_time_list)}, avg={sum(kv_cache_compression_time_list)/len(kv_cache_compression_time_list):.6f}s")

# if rank == 0:
#     with open("result.txt", "a") as file:
#         file.write("total time :{:.5f}s, time per iter :{:.5f}s, decoding step: {}, large model step: {}, avg latency: {} \n".format(total_time, total_time / target_steps, num_gen_tokens, target_steps, total_time / num_gen_tokens * BATCH_SIZE))
#         file.write("target time :{:.5f}s, draft time :{:.5f}s, verify loop : {}, avg generate len per sentence: {} \n".format(target_time/target_steps, draft_time / target_steps, verify_loop/target_steps, num_gen_tokens/target_steps/BATCH_SIZE))