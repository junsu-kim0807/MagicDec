import torch
from datasets import load_dataset
import os
import importlib
import yaml
from torch.utils.data import TensorDataset
from tqdm import tqdm
# from nemo.collections.asr.parts.utils.manifest_utils import read_manifest

def convert_c4_dataset(tokenizer, file_path):
    dataset = load_dataset("json", data_files=file_path, split="train")
    def tokenize_function(examples):
            input_ids = torch.Tensor(examples['input_ids'])
            labels = input_ids.clone()
            if tokenizer.pad_token_id is not None:
                 labels[labels == tokenizer.pad_token_id] = -100
            ret = {
                "input_ids": input_ids,
                "labels": labels
            }
            return ret
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['input_tokens'])
    dataset.set_format(type='torch', columns=['input_ids', "labels"])
    return dataset

def convert_wiki_dataset(tokenizer, seq_len = 256):
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[0:2000]")
    def tokenize_function(examples):
            return tokenizer(examples["text"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset

def convert_cnn_dataset(tokenizer, seq_len = 256):
    dataset = load_dataset("cnn_dailymail", "1.0.0", split="test[0:2000]")
    def tokenize_function(examples):
            return tokenizer(examples["article"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['article'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset

def convert_pg19_dataset(tokenizer, seq_len = 4096, end = 20):
    datasetparent = "Data/pg19/"
    d_files = os.listdir(datasetparent)
    dataset = load_dataset("json", data_files = [datasetparent + name for name in d_files], split = "train")
    tokenized_prompts = []
    for i in tqdm(range(0,50)):
        prompt = dataset[i]['text']
        tokenized_prompt = tokenizer.encode(prompt, return_tensors="pt")[:,8000:]
        tokenized_prompt = tokenized_prompt.split(seq_len, dim=-1)[:-1]
        
        for i in range(len(tokenized_prompt)):
            tokenized_prompt[i][:, 0] = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
            tokenized_prompts.append(tokenized_prompt[i])
    data = torch.cat(tokenized_prompts, dim=0).repeat(end,1)
    return TensorDataset(data)


def _tokenize_to_fixed_chunks(tokenizer, texts, seq_len: int, end_repeat: int = 20):
    """
    Create a TensorDataset of shape [N, seq_len] by padding/truncating tokenized texts.
    This mirrors the benchmark scripts' expectation that input_ids length == prefix_len.
    """
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    token_tensors = []
    for text in tqdm(texts):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) == 0:
            token_ids = [pad_id]

        # Truncate or pad to fixed seq_len
        if len(token_ids) >= seq_len:
            chunk = token_ids[:seq_len]
        else:
            chunk = token_ids + [pad_id] * (seq_len - len(token_ids))

        # Ensure BOS at the first position (keep behavior consistent with pg19 converter)
        chunk[0] = bos_id
        token_tensors.append(torch.tensor(chunk, dtype=torch.long))

    if len(token_tensors) == 0:
        # Fallback: at least one example to avoid empty dataloader
        token_tensors = [torch.tensor([bos_id] + [pad_id] * (seq_len - 1), dtype=torch.long)]

    data = torch.stack(token_tensors, dim=0).repeat(end_repeat, 1)
    return TensorDataset(data)


def convert_aime2025_dataset(tokenizer, seq_len: int = 4096, end: int = 20):
    """
    AIME 2025 problems from HF: math-ai/aime25
    Produces "Problem: ...\\nAnswer:" prompts for throughput evaluation.
    """
    dataset = load_dataset("math-ai/aime25", split="test")
    texts = [f"Problem: {row['problem']}\\nAnswer:" for row in dataset]
    return _tokenize_to_fixed_chunks(tokenizer, texts=texts, seq_len=seq_len, end_repeat=end)


def convert_codeelo_dataset(tokenizer, seq_len: int = 4096, end: int = 20):
    """
    CodeElo prompts from HF: Qwen/CodeElo
    Uses description + input (+interaction/note if present) as the prompt.
    """
    dataset = load_dataset("Qwen/CodeElo", split="train")

    texts = []
    for row in dataset:
        parts = []
        if row.get("description"):
            parts.append(str(row["description"]))
        if row.get("input"):
            parts.append(str(row["input"]))
        if row.get("interaction"):
            if row.get("interaction"):
                parts.append(str(row["interaction"]))
        if row.get("note"):
            if row.get("note"):
                parts.append(str(row["note"]))
        parts.append("Answer:")
        texts.append("\\n\\n".join(parts))

    return _tokenize_to_fixed_chunks(tokenizer, texts=texts, seq_len=seq_len, end_repeat=end)


def convert_longbench_v2_dataset(tokenizer, seq_len: int = 4096, end: int = 20):
    """
    LongBench-v2 prompts from HF: THUDM/LongBench-v2 (only 'train' split exists)
    Uses context + question + multiple-choice options.
    """
    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    texts = []
    for row in dataset:
        context = row.get("context") or ""
        question = row.get("question") or row.get("input") or ""
        choices = []
        for letter in ["A", "B", "C", "D"]:
            key = f"choice_{letter}"
            val = row.get(key)
            if val:
                choices.append(f"{letter}) {val}")
        prompt = context + "\n\n" + f"Question: {question}\n" + ("\n".join(choices) + "\n" if choices else "") + "Answer:"
        texts.append(prompt)
    return _tokenize_to_fixed_chunks(tokenizer, texts=texts, seq_len=seq_len, end_repeat=end)


def convert_longbench_v1_dataset(tokenizer, seq_len: int = 4096, task_name: str = "narrativeqa", end: int = 20):
    """
    LongBench-v1 prompts from HF: THUDM/LongBench (requires task_name config, loaded from 'test' split).
    Docs format includes: input, context, answers (list), dataset, language, _id.
    """
    dataset = load_dataset("THUDM/LongBench", task_name, split="test")
    texts = []
    for row in dataset:
        inp = row.get("input") or ""
        ctx = row.get("context") or ""
        prompt = (inp + "\n\n" + ctx + "\n\n" if ctx else inp + "\n") + "Answer:"
        texts.append(prompt)
    return _tokenize_to_fixed_chunks(tokenizer, texts=texts, seq_len=seq_len, end_repeat=end)


def repeat_dataset_to_min_len(dataset, min_len: int):
    """
    Repeat (and truncate) a TensorDataset so that len(dataset) >= min_len.
    This is useful for benchmarks where drop_last=True and we want at least 2 batches.
    """
    if len(dataset) >= min_len:
        return dataset

    # TensorDataset provides `.tensors` (tuple of tensors).
    # All tensors share the same first dimension length.
    if not hasattr(dataset, "tensors"):
        raise TypeError("repeat_dataset_to_min_len expects a torch.utils.data.TensorDataset")

    target_len = int(min_len)
    base_len = len(dataset)
    reps = (target_len + base_len - 1) // base_len

    repeated_tensors = []
    for t in dataset.tensors:
        # Repeat on dim=0 only, keep other dims.
        reps_shape = (reps,) + (1,) * (t.dim() - 1)
        repeated_tensors.append(t.repeat(*reps_shape)[:target_len])

    return TensorDataset(*repeated_tensors)

# def convert_ruler_dataset(tokenizer, task, model_name, seq_len = 4096, subset = "validation"):
#     curr_folder = os.path.dirname(os.path.abspath(__file__))
#     try:
#         module = importlib.import_module(f"MagicDec.Data.Ruler.synthetic.constants")
#     except ImportError:
#         print(f"Module MagicDec.Data.Ruler.synthetic.constants not found.")

#     tasks_base = module.TASKS
#     with open(os.path.join(curr_folder, f"Ruler/synthetic.yaml"), "r") as f:
#         tasks_customized = yaml.safe_load(f)

#     if task not in tasks_customized:
#         raise ValueError(f'{task} is not found in config_tasks.yaml')
        
#     config = tasks_customized.get(task)
#     config.update(tasks_base[config['task']])
    
#     root_task = tasks_customized[task]['task']
#     suffix = tasks_base[root_task]['template'].split('{context}')[-1]

#     task_file = os.path.join(curr_folder, "Ruler/benchmark_root", model_name, "data", task, f"{subset}.jsonl")
    
#     data = read_manifest(task_file)

#     tokenized_prompts = []
#     tokenized_suffix = tokenizer.encode(suffix, return_tensors="pt")[:, 1:] # remove the bos token
#     suffix_len = tokenized_suffix.shape[-1]
#     print("Total number of prompts", len(data))
#     for i in range(len(data)):
#         prompt = data[i]['input'][:-len(suffix)]
#         input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=seq_len - suffix_len, padding="max_length")
#         assert input_ids.shape[-1] == seq_len - suffix_len
#         tokenized_prompts.append(torch.cat([input_ids[:, :seq_len - suffix_len], tokenized_suffix], dim=-1))
#     data = torch.cat(tokenized_prompts, dim=0)
#     return TensorDataset(data)

# if __name__ == "__main__":
#     from transformers import LlamaTokenizer, DataCollatorForLanguageModeling
#     from torch.utils.data import DataLoader, TensorDataset
#     from tqdm import tqdm
#     tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
#     tokenizer.pad_token = tokenizer.eos_token
#     dataset = convert_pg19_dataset(tokenizer=tokenizer, seq_len=4096)

#     dataloader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=True)
#     num_eval_steps = len(dataloader)
#     for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
#         input_ids = batch[0]
    