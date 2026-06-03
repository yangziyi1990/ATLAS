"""
SAGE v4: Shard-based ESM-2 Feature Extraction (FP16)
=====================================================
Processes a single shard of genomes to extract ESM-2 protein embeddings.

Key optimizations:
  1. FP16 model weights + output: 50% VRAM savings + faster inference
  2. Per-shard independent .pt files: eliminates HDF5 write-lock contention
  3. Sorted-by-length batching: minimizes padding waste
  4. Prefetch DataLoader: I/O 与 GPU 推理并行
  5. 去除每 batch empty_cache(): 避免频繁 CUDA 内存回收开销

Output per shard:
  - esm_shard_XXXX.pt  (dict: {query_id: tensor[esm_dim], ...}, dtype=float16)

Usage:
    python extract_esm_shard.py \
        --shard_file /path/to/shard_index/shard_0000.json \
        --output_dir /path/to/esm_sharded \
        --batch_size 64 \
        --model_name facebook/esm2_t12_35M_UR50D
"""

import os
import json
import argparse
import logging
import torch
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ProteinSequenceDataset(Dataset):
    """将蛋白质序列包装为 Dataset，支持 DataLoader prefetch."""

    def __init__(self, records, tokenizer, max_seq_length=1024):
        # 按长度排序以减少 padding 浪费
        self.records = sorted(records, key=lambda x: len(x[1]))
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        seq_id, seq_str = self.records[idx]
        return seq_id, seq_str


def collate_fn_factory(tokenizer, max_seq_length):
    """创建 batch collate 函数，在 DataLoader worker 中完成 tokenize."""
    def collate_fn(batch):
        seq_ids = [item[0] for item in batch]
        sequences = [item[1] for item in batch]
        inputs = tokenizer(
            sequences, return_tensors="pt",
            padding=True, truncation=True,
            max_length=max_seq_length
        )
        return seq_ids, inputs
    return collate_fn


def extract_esm_for_shard(shard_file, output_dir, model=None, tokenizer=None, device=None,
                          model_name="facebook/esm2_t12_35M_UR50D",
                          batch_size=64, max_seq_length=1024, num_workers=4):
    """Extract ESM-2 features for all proteins in a shard.
    
    If model/tokenizer/device are provided, reuses them (avoids reloading).
    """
    from Bio import SeqIO

    os.makedirs(output_dir, exist_ok=True)

    with open(shard_file, 'r') as f:
        shard_data = json.load(f)

    shard_id = shard_data['shard_id']
    genomes = shard_data['genomes']

    output_path = os.path.join(output_dir, f"esm_shard_{shard_id:04d}.pt")

    # Skip if already processed
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        logging.info(f"Shard {shard_id}: already exists, skipping → {output_path}")
        return

    logging.info(f"Shard {shard_id}: collecting protein sequences from {len(genomes)} genomes...")

    # Collect all protein sequences
    all_records = []  # list of (seq_id, sequence_str)
    for genome_info in genomes:
        faa_path = genome_info.get('faa')
        if not faa_path or not os.path.exists(faa_path):
            continue
        try:
            for record in SeqIO.parse(faa_path, "fasta"):
                seq_str = str(record.seq)
                if len(seq_str) > 0:
                    all_records.append((record.id, seq_str))
        except Exception as e:
            logging.warning(f"Error reading {faa_path}: {e}")

    if not all_records:
        logging.warning(f"Shard {shard_id}: no protein sequences found")
        # Save empty dict
        torch.save({}, output_path)
        return

    logging.info(f"Shard {shard_id}: {len(all_records)} protein sequences to process")

    # Load model (or reuse provided one)
    if model is None or tokenizer is None or device is None:
        from transformers import EsmModel, EsmTokenizer
        logging.info(f"Loading ESM-2 model: {model_name}")
        tokenizer = EsmTokenizer.from_pretrained(model_name)
        model = EsmModel.from_pretrained(model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.half().to(device)  # FP16 模型权重
        model.eval()

    # 构建 DataLoader, 利用 num_workers prefetch 实现 I/O 与 GPU 并行
    dataset = ProteinSequenceDataset(all_records, tokenizer, max_seq_length)
    collate_fn = collate_fn_factory(tokenizer, max_seq_length)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
        pin_memory=(device.type == 'cuda'), prefetch_factor=2
    )

    # Extract features
    features = {}
    total_processed = 0

    with torch.no_grad():
        for batch_idx, (seq_ids, inputs) in enumerate(dataloader):
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

            # FP16 inference (模型已是 FP16，无需 autocast)
            outputs = model(**inputs)

            # Mean pooling
            last_hidden = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).half()
            sum_emb = torch.sum(last_hidden * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_emb = sum_emb / sum_mask

            # Store as FP16
            emb_cpu = mean_emb.cpu()
            for j, seq_id in enumerate(seq_ids):
                features[seq_id] = emb_cpu[j]

            total_processed += len(seq_ids)

            if (batch_idx + 1) % 50 == 0:
                logging.info(f"  Shard {shard_id}: processed {total_processed}/{len(all_records)} sequences")

    # Save as .pt (FP16)
    torch.save(features, output_path)
    logging.info(f"Shard {shard_id}: saved {len(features)} embeddings → {output_path} "
                 f"(size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="SAGE v4: Shard-based ESM-2 feature extraction (FP16)")
    parser.add_argument("--shard_file", type=str, default=None,
                        help="Path to a single shard JSON file")
    parser.add_argument("--shard_list", type=str, default=None,
                        help="Path to a text file listing shard JSON paths (one per line). "
                             "Model is loaded once and reused for all shards.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for ESM inference (V100-32GB 建议 64-128)")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers for prefetch (I/O 与 GPU 并行)")
    args = parser.parse_args()

    if not args.shard_file and not args.shard_list:
        parser.error("Either --shard_file or --shard_list is required")

    # Collect shard files to process
    shard_files = []
    if args.shard_list:
        with open(args.shard_list, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    shard_files.append(line)
    elif args.shard_file:
        shard_files.append(args.shard_file)

    if not shard_files:
        logging.warning("No shard files to process.")
        return

    # Load model ONCE, directly in FP16
    from transformers import EsmModel, EsmTokenizer
    logging.info(f"Loading ESM-2 model: {args.model_name} (shared across {len(shard_files)} shards)")
    tokenizer = EsmTokenizer.from_pretrained(args.model_name)
    model = EsmModel.from_pretrained(args.model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.half().to(device)  # FP16 模型权重, 省一半显存 + 推理更快
    model.eval()

    logging.info(f"Model loaded in FP16 | Device: {device} | batch_size: {args.batch_size} | "
                 f"num_workers: {args.num_workers}")

    # Process all shards with shared model
    for i, sf in enumerate(shard_files):
        logging.info(f"[{i+1}/{len(shard_files)}] Processing: {sf}")
        extract_esm_for_shard(
            shard_file=sf,
            output_dir=args.output_dir,
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            num_workers=args.num_workers,
        )

    logging.info(f"All {len(shard_files)} shards completed.")


if __name__ == "__main__":
    main()
