"""
SAGE v4.1: Streaming Pre-training Script (DDP) — ESM Regression MLM
=====================================================================
Adapted from v4 train_streaming.py with key changes:
  1. MLM loss: Hash classification (CE 65542 classes) → ESM Feature Regression
     (Cosine Similarity + Smooth L1 on normalized features)
  2. Strain embedding: contrastive_head projected → attention_pooling 480d
  3. Warm start from v3 checkpoint (skip old mlm_head weights)
  4. Loss warmup scheduling for new regression head

Launch:
  # Single node, 8 GPUs
  torchrun --nproc_per_node=8 train_streaming.py --metadata_dir ... --esm_dir ...

  # Warm start from v3 checkpoint
  torchrun --nproc_per_node=8 train_streaming.py --warm_start /path/to/sage_v3_best.pt ...
"""

import argparse
import itertools
import logging
import os
import json
import time
import random
import threading
from datetime import timedelta
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_streaming import SAGEStreamingDataset, get_streaming_collator, TOTAL_VOCAB_SIZE
from model import GenomicLanguageModelV3
from model_configs import add_config_argument, get_model_config, apply_config_to_args

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ==============================================================================
# DDP Utilities
# ==============================================================================

def setup_distributed():
    """Initialize DDP if launched via torchrun, otherwise single-GPU mode."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        # NCCL 默认 timeout 是 30 min, 对 IterableDataset + 726 shard 的
        # 长 epoch + checkpoint 写 cephfs 来说太短 (见 2026-04-24 崩溃)
        # 改为 2 小时留充足 safety margin.
        nccl_timeout_minutes = int(os.environ.get("NCCL_TIMEOUT_MINUTES", "120"))
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=nccl_timeout_minutes),
        )
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, True
    else:
        # Single GPU fallback
        return 0, 0, 1, False


def cleanup_distributed(use_ddp):
    """Clean up DDP process group."""
    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Check if current process is the main (rank 0) process."""
    return rank == 0


def log_info(msg, rank=0):
    """Only log on rank 0."""
    if is_main_process(rank):
        logging.info(msg)


# ==============================================================================
# Async checkpoint saver (avoid blocking rank 0 → DDP timeout)
# ==============================================================================

_save_lock = threading.Lock()
_save_thread = None


def _atomic_save(state_dict_snapshot, path):
    """Write to .tmp then rename (atomic). Runs in background thread."""
    tmp = path + ".tmp"
    try:
        torch.save(state_dict_snapshot, tmp)
        os.replace(tmp, path)
    except Exception as e:
        logging.warning(f"Async save failed for {path}: {e}")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def async_save_checkpoint(state, path):
    """Save a checkpoint asynchronously.

    注意: state 应为已经 clone 到 CPU 的 dict, 否则后台线程读取 GPU tensor 时
    会与主训练线程的 CUDA 调用竞争. 这里直接把调用方传入的 state 搬到 CPU.
    """
    global _save_thread

    # 如果上一个 save 还没完成, 等它完成 (避免并发写同一文件)
    with _save_lock:
        if _save_thread is not None and _save_thread.is_alive():
            _save_thread.join(timeout=60)   # 最多等 60s，cephfs 慢时避免长时间阻塞
            if _save_thread.is_alive():
                logging.warning(
                    f"Previous async save still running after 60s timeout, skipping "
                    f"this save to avoid blocking training. Path: {path}"
                )
                return

    # Snapshot state to CPU (同步, 因为涉及 GPU → CPU 拷贝)
    cpu_state = {}
    for k, v in state.items():
        if isinstance(v, dict):
            cpu_state[k] = {kk: (vv.detach().cpu().clone() if torch.is_tensor(vv) else vv)
                            for kk, vv in v.items()}
        elif torch.is_tensor(v):
            cpu_state[k] = v.detach().cpu().clone()
        else:
            cpu_state[k] = v

    t = threading.Thread(target=_atomic_save, args=(cpu_state, path), daemon=True)
    t.start()
    with _save_lock:
        _save_thread = t


# ==============================================================================
# Infinite data iterator (replaces itertools.cycle to avoid OOM)
# ==============================================================================

def _endless_iter(loader, dataset=None, base_epoch=0):
    """Infinite iterator that re-creates iter(loader) on exhaustion.

    Unlike itertools.cycle, this does NOT cache elements in memory.
    Each cycle triggers fresh IterableDataset reads with new epoch shuffling
    (controlled by dataset.epoch = base_epoch*10000 + sub_epoch).

    P0-4 fix:   prevents OOM from caching entire epoch's batches in memory.
    P0-5 fix:   increments dataset.epoch each cycle so shuffle order changes.
    P0-6 fix:   uses base_epoch (outer big epoch) as high bits so different
                big epochs produce distinct shuffles even at sub_epoch=0.
    """
    sub_epoch = 0
    while True:
        if dataset is not None:
            # 用 base_epoch*10000 做高位，确保不同大 epoch 的 sub_epoch=0 不碰撞
            dataset.epoch = base_epoch * 10000 + sub_epoch
        for batch in loader:
            yield batch
        sub_epoch += 1


# ==============================================================================
# Contrastive Loss with Cross-GPU Gathering
# ==============================================================================

def gather_from_all_gpus(tensor, world_size, use_ddp):
    """Gather tensors from all GPUs for contrastive learning."""
    if not use_ddp or world_size <= 1:
        return tensor

    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    # Replace own entry with original (to keep gradients)
    gathered[dist.get_rank()] = tensor
    return torch.cat(gathered, dim=0)


def supervised_contrastive_loss(embeddings, labels, temperature=0.07,
                                world_size=1, use_ddp=False):
    """Supervised Contrastive Loss (SupCon) with optional cross-GPU gathering.

    所有 rank 必须同时调用 all_gather，即使 local B<=1 也如此。
    否则参与数不匹配会导致 NCCL 死锁（超时 120min）。
    degenerate 判断放在 gather 之后，用 all_labels 而非 local labels。

    P2-2 fix:
      1) embeddings 在计算相似度前做 L2 normalize，确保 dot product = cosine sim，
         使 temperature 语义明确且尺度不随 d_model 变化。
      2) 当 gather 后有效样本数 <= 1 或无正样本对时，返回 embeddings.sum()*0.0
         保持计算图连通（DDP static_graph 要求所有参数参与 backward）。
    """
    # 无条件先 gather（所有 rank 都要参与），再判定
    if use_ddp and world_size > 1:
        all_embeddings = gather_from_all_gpus(embeddings, world_size, use_ddp)
        all_labels = gather_from_all_gpus(labels, world_size, use_ddp)
    else:
        all_embeddings = embeddings
        all_labels = labels

    # 用 all_labels 而非 local 判断
    valid_mask = all_labels >= 0
    if valid_mask.sum() <= 1:
        return embeddings.sum() * 0.0

    all_embeddings = all_embeddings[valid_mask]
    all_labels = all_labels[valid_mask]
    B = all_embeddings.size(0)
    dev = all_embeddings.device

    # P2-2: L2 normalize — 使 dot product 等价于 cosine similarity
    all_embeddings = F.normalize(all_embeddings, dim=-1)

    sim_matrix = torch.matmul(all_embeddings, all_embeddings.T) / temperature
    pos_mask = (all_labels.unsqueeze(1) == all_labels.unsqueeze(0)).float()
    pos_mask = pos_mask * (1.0 - torch.eye(B, device=dev))

    has_positive = pos_mask.sum(dim=1) > 0
    if has_positive.sum() == 0:
        return embeddings.sum() * 0.0

    logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()
    exp_logits = torch.exp(logits)
    self_mask = 1.0 - torch.eye(B, device=dev)
    exp_logits = exp_logits * self_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
    loss = -mean_log_prob_pos[has_positive].mean()
    return loss


# ==============================================================================
# Argument Parser
# ==============================================================================

def get_args():
    parser = argparse.ArgumentParser(description="SAGE v4: Streaming Pre-training (DDP)")

    # Data paths
    parser.add_argument("--metadata_dir", type=str, required=True,
                        help="Directory containing shard_XXXX_metadata.jsonl files")
    parser.add_argument("--esm_dir", type=str, required=True,
                        help="Directory containing esm_shard_XXXX.pt files")

    # Model architecture
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=6000)
    parser.add_argument("--distance_window", type=int, default=128)

    # Architecture switches
    parser.add_argument("--use_swiglu", action="store_true", default=True)
    parser.add_argument("--no_swiglu", dest="use_swiglu", action="store_false")
    parser.add_argument("--use_hierarchical_attention", action="store_true", default=True)
    parser.add_argument("--no_hierarchical_attention", dest="use_hierarchical_attention", action="store_false")
    parser.add_argument("--use_gated_fusion", action="store_true", default=True)
    parser.add_argument("--no_gated_fusion", dest="use_gated_fusion", action="store_false")
    parser.add_argument("--use_distance_bias", action="store_true", default=True)
    parser.add_argument("--no_distance_bias", dest="use_distance_bias", action="store_false")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=8)

    # Loss weights
    parser.add_argument("--cog_loss_weight", type=float, default=0.5)
    parser.add_argument("--strand_loss_weight", type=float, default=0.5)
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.1)
    parser.add_argument("--contrastive_temperature", type=float, default=0.07)
    parser.add_argument("--contrastive_proj_dim", type=int, default=128,
                        help="Output dimension for contrastive projection head")

    # Masking
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--span_length", type=int, default=3)
    parser.add_argument("--operon_mask_prob", type=float, default=0.05)

    # Experiment
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_v3")
    parser.add_argument("--exp_name", type=str, default="sage_v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps_per_epoch", type=int, default=0,
                        help="Steps per epoch for IterableDataset (0=auto, process all data). "
                             "强制对齐各 rank 进度, 避免 DDP 数据量不均导致的 NCCL 超时死锁.")
    parser.add_argument("--save_every_n_steps", type=int, default=0,
                        help="Save a rolling 'latest' checkpoint every N optimizer_steps (0=disable). "
                             "用于 epoch 极长时做 step-level 容灾保存.")
    parser.add_argument("--log_interval", type=int, default=100)

    # Resume training
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    # Warm start from v3 checkpoint (ESM regression MLM)
    parser.add_argument("--warm_start", type=str, default=None,
                        help="Path to v3 checkpoint for warm start (skips old mlm_head weights)")
    parser.add_argument("--mlm_warmup_steps", type=int, default=500,
                        help="Number of optimizer steps to linearly ramp up MLM regression loss weight (0→1)")

    # 模型配置预设 (可选, 设置后覆盖架构/训练默认值)
    add_config_argument(parser)

    args = parser.parse_args()

    # 如果指定了 --model_config, 用预设值覆盖默认参数
    # 传入 parser 以确保命令行显式传入的参数不被覆盖
    if args.model_config:
        config = get_model_config(args.model_config)
        apply_config_to_args(args, config, parser=parser)

    return args


# ==============================================================================
# Main Training Function
# ==============================================================================

def main():
    args = get_args()

    # --- DDP Setup ---
    rank, local_rank, world_size, use_ddp = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    # Seed (offset by rank for data diversity)
    seed = args.seed + rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    log_info(f"{'=' * 60}", rank)
    log_info(f"SAGE v4 Streaming Pre-training (DDP)", rank)
    log_info(f"  World size: {world_size} | Rank: {rank} | Local rank: {local_rank}", rank)
    log_info(f"  Device: {device} ({torch.cuda.get_device_name(local_rank)})", rank)
    log_info(f"{'=' * 60}", rank)

    # 1. Create streaming dataset
    # dataset_streaming.py already handles DDP shard partitioning internally
    dataset = SAGEStreamingDataset(
        metadata_dir=args.metadata_dir,
        esm_dir=args.esm_dir,
        max_seq_len=args.max_seq_len,
        shuffle_shards=True,
        seed=args.seed,  # Use same base seed for consistent shard shuffling
    )

    esm_dim = dataset.esm_dim
    log_info(f"ESM feature dimension: {esm_dim}", rank)
    log_info(f"Total shards: {len(dataset.shard_files)} | Shards per rank: ~{len(dataset.shard_files) // max(world_size, 1)}", rank)

    collate_fn = get_streaming_collator(
        max_seq_len=args.max_seq_len,
        mask_prob=args.mask_prob,
        span_length=args.span_length,
        operon_mask_prob=args.operon_mask_prob,
    )

    # pin_memory=False: PyTorch 2.0.1 + IterableDataset + 8 worker 在太极节点
    # 曾触发 "CUDA error: invalid argument" (pin_memory thread). 关闭后
    # H2D 拷贝路径改用 non-pinned async, 对吞吐影响 <5%, 但稳定性提升明显.
    # persistent_workers=False: P1-4 fix — _endless_iter 反复循环创建新 iter(dataloader)
    # 时，persistent_workers=True 与 IterableDataset 的交互在 PyTorch 2.0.1 未充分测试，
    # 可能导致 worker 进程卡住。关闭后稳定性更好，multi-epoch 训练不会有进程复用问题。
    # timeout: P1-14 fix — 单个 worker cephfs IO stall 时，主进程无限等待会导致
    # 整个 DDP 卡死 NCCL watchdog 超时（之前崩溃根因之一）。
    # 通过环境变量 DATALOADER_TIMEOUT 控制（秒），默认 1800s (30 min)。
    # CephFS 多卡并发时首个 shard 加载可能需要 15-20 分钟，600s 不够。
    # prefetch_factor=4: 从默认 2 提升到 4，提供额外缓冲以容忍慢 worker。
    dl_timeout = int(os.environ.get("DATALOADER_TIMEOUT", "1800"))
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=False,
        timeout=dl_timeout if args.num_workers > 0 else 0,
    )

    # 2. Initialize model
    vocab_size = TOTAL_VOCAB_SIZE

    model = GenomicLanguageModelV3(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        feature_dim=esm_dim,
        esm_features=None,
        use_swiglu=args.use_swiglu,
        use_hierarchical_attention=args.use_hierarchical_attention,
        use_gated_fusion=args.use_gated_fusion,
        use_distance_bias=args.use_distance_bias,
        max_seq_len=args.max_seq_len,
        distance_window=args.distance_window,
        gradient_checkpointing=args.gradient_checkpointing,
        contrastive_proj_dim=args.contrastive_proj_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log_info(f"Parameters: {total_params:,} ({total_params / 1e6:.2f}M)", rank)

    # --- DDP Wrapping ---
    if use_ddp:
        # static_graph=True + gradient_checkpointing 兼容 (PyTorch >= 1.11).
        # 前提是计算图结构在所有 iteration 不变 —— 我们通过以下保证:
        #   1) forward 中 cog_head/strand_head/contrastive_head/esm_regression_head 恒被调用
        #   2) contrastive_loss_weight=0 时, 用 contrastive_projected.sum()*0.0
        #      保持 contrastive_head 在计算图中 (不做 all_gather/SupCon)
        #   3) supervised_contrastive_loss 在 degenerate case 返回
        #      `embeddings.sum()*0.0` (带计算图) 而非 torch.tensor(0.0)
        #   4) loss_mlm 的 empty-mask 分支也走 esm_regression_head 的 0-loss 保底
        # 这样所有参数每步都参与 backward, 无 unused param.
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            static_graph=True,
        )
        log_info(f"Model wrapped with DistributedDataParallel (static_graph=True)", rank)

    # Access the underlying model (for esm_regression_head, etc.)
    raw_model = model.module if use_ddp else model

    # 3. Optimizer + Scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # For IterableDataset, estimate total steps (per rank)
    if args.steps_per_epoch > 0:
        steps_per_epoch = args.steps_per_epoch // args.gradient_accumulation_steps
        # P1-8: warn if steps_per_epoch is not divisible by grad_accum
        if args.steps_per_epoch % args.gradient_accumulation_steps != 0:
            log_info(
                f"[HINT] steps_per_epoch={args.steps_per_epoch} is not divisible by "
                f"grad_accum={args.gradient_accumulation_steps}. "
                f"Last {args.steps_per_epoch % args.gradient_accumulation_steps} micro "
                f"batches per epoch will not flush gradients (by design, see P1-3 fix).",
                rank
            )
    else:
        n_shards = len(dataset.shard_files)
        estimated_genomes = n_shards * 1000
        # Each rank sees ~1/world_size of total data
        genomes_per_rank = estimated_genomes // max(world_size, 1)
        steps_per_epoch = max(1, genomes_per_rank // (args.batch_size * args.gradient_accumulation_steps))
        log_info(f"Estimated steps per epoch (per rank): {steps_per_epoch}", rank)

    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps),
                                         eta_min=args.lr * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_steps])

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    if is_main_process(rank):
        os.makedirs(args.ckpt_dir, exist_ok=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    training_history = []
    best_loss = float('inf')
    global_step = 0
    optimizer_step = 0
    start_epoch = 1

    # --- Warm start from v3 checkpoint (ESM Regression MLM) ---
    if args.warm_start and os.path.exists(args.warm_start):
        log_info(f"Warm start from v3 checkpoint: {args.warm_start}", rank)
        ckpt = torch.load(args.warm_start, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        
        # Remove old mlm_head weights (no longer compatible)
        keys_to_remove = [k for k in state_dict if k.startswith("mlm_head")]
        for k in keys_to_remove:
            del state_dict[k]
        if keys_to_remove:
            log_info(f"  Removed {len(keys_to_remove)} old mlm_head keys: {keys_to_remove}", rank)
        
        missing, unexpected = raw_model.load_state_dict(state_dict, strict=False)
        log_info(f"  Warm start loaded. Missing: {missing[:10]}{'...' if len(missing)>10 else ''}", rank)
        if unexpected:
            log_info(f"  Unexpected keys: {unexpected[:5]}", rank)
        log_info(f"  New esm_regression_head will be trained from random init.", rank)

    # --- Resume from checkpoint ---
    elif args.resume and os.path.exists(args.resume):
        log_info(f"Resuming from checkpoint: {args.resume}", rank)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt.get("best_loss", float('inf'))
        global_step = ckpt.get("global_step", 0)
        optimizer_step = ckpt.get("optimizer_step", 0)
        if "training_history" in ckpt:
            training_history = ckpt["training_history"]
        log_info(f"  Resumed from epoch {ckpt['epoch']}, best_loss={best_loss:.4f}", rank)
        # P1-5: 诊断 scheduler 状态，防止 T_max 覆盖导致 cosine 曲线未延伸到新 epochs
        log_info(
            f"  Scheduler: last_lr={scheduler.get_last_lr()}, "
            f"_step_count={scheduler._step_count}, "
            f"(注意: resume ckpt 的 T_max 会被继承，若 epochs 增大需手动重建 scheduler)",
            rank
        )

    # Effective batch size info
    effective_batch = args.batch_size * args.gradient_accumulation_steps * world_size
    log_info(f"Effective batch size: {args.batch_size} x {args.gradient_accumulation_steps} x {world_size} = {effective_batch}", rank)

    # P0-7: warn if steps_per_epoch is close to or exceeds estimated available batches,
    # which would trigger _endless_iter cycling and worker respawn on each cycle.
    # 注意: steps_per_epoch 是 micro-batch 数 (每次 dataloader yield 一次计 1 次), 所以
    # 估算阈值也必须是 micro-batch 粒度 (只除 batch_size, 不除 grad_accum).
    if args.steps_per_epoch > 0:
        n_shards_per_rank = max(1, len(dataset.shard_files) // max(world_size, 1))
        est_genomes_per_rank = n_shards_per_rank * 1000
        est_micro_batches_per_rank = est_genomes_per_rank // args.batch_size
        log_info(
            f"Estimated micro-batches available per rank per epoch: ~{est_micro_batches_per_rank} "
            f"(steps_per_epoch={args.steps_per_epoch}, micro-batch granularity). "
            f"{'OK — no cycling needed.' if args.steps_per_epoch < est_micro_batches_per_rank * 0.9 else 'WARN — may trigger cycling, which re-spawns workers each cycle (persistent_workers=False).'}",
            rank
        )

    # 4. Training loop
    log_info("Starting training...", rank)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        dataset.set_epoch(epoch)

        total_loss = 0
        total_mlm = 0
        total_cog = 0
        total_strand = 0
        total_cl = 0
        num_batches = 0
        epoch_start = time.time()

        optimizer.zero_grad()

        # P0-3/P0-4/P0-5 fix: IterableDataset 各 rank shard 数不同，StopIteration
        # 时机也不同。用 itertools.islice(_endless_iter(...), limit) 确保所有 rank
        # 都精确跑完指定步数。_endless_iter 每次回环用新 dataset.epoch 重新 shuffle，
        # 既避免 itertools.cycle 的 OOM 风险，也防止同批数据被重复训练。
        steps_limit = args.steps_per_epoch if args.steps_per_epoch > 0 else None
        if steps_limit is not None:
            data_iter = itertools.islice(
                _endless_iter(dataloader, dataset, base_epoch=epoch),
                steps_limit,
            )
        else:
            data_iter = iter(dataloader)

        for batch_idx, batch in enumerate(data_iter):
            # P0-8: step_start 必须在循环体最前面初始化，否则下面的 slow-step detection
            # 会引用未定义变量导致 NameError。记录 "取到 batch + 前向 + 反向 + 可能的
            # optimizer step" 整体耗时。
            step_start = time.time()

            input_ids = batch["input_ids"].to(device)
            esm_features = batch["esm_features"].to(device)
            strand_ids = batch["strand_ids"].to(device)
            replicon_ids = batch["replicon_ids"].to(device)
            cog_ids = batch["cog_ids"].to(device)
            contig_ids = batch["contig_ids"].to(device)
            mutation_ids = batch["mutation_ids"].to(device)
            distance_ids = batch["distance_ids"].to(device)
            position_ids = batch["position_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            labels_strand = batch["labels_strand"].to(device)
            labels_cog = batch["labels_cog"].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(
                    gene_seqs=input_ids,
                    gene_features=esm_features,
                    strand_ids=strand_ids,
                    replicon_ids=replicon_ids,
                    cog_ids=cog_ids,
                    contig_ids=contig_ids,
                    mutation_ids=mutation_ids,
                    distance_ids=distance_ids,
                    position_ids=position_ids,
                    mask=attention_mask,
                )

                hidden_states = outputs["hidden_states"]
                mask_idx = (labels != -100)

                # ---- ESM Regression MLM Loss ----
                if mask_idx.sum() > 0:
                    masked_hidden = hidden_states[mask_idx]  # [N_masked, d_model]
                    
                    # Get ESM regression targets
                    esm_targets = batch["esm_targets"].to(device)  # [B, L, esm_dim]
                    target_esm = esm_targets[mask_idx]  # [N_masked, esm_dim]
                    
                    # Filter out positions with zero ESM target (special tokens, missing ESM)
                    valid_target = (target_esm.abs().sum(dim=-1) > 1e-6)
                    
                    if valid_target.sum() > 0:
                        predicted_esm = raw_model.esm_regression_head(
                            masked_hidden[valid_target]
                        )  # [N_valid, esm_dim]
                        target_valid = target_esm[valid_target]  # [N_valid, esm_dim]
                        
                        # Compute loss in FP32 for numerical stability
                        with torch.cuda.amp.autocast(enabled=False):
                            pred_fp32 = predicted_esm.float()
                            tgt_fp32 = target_valid.float()
                            
                            # Loss 1: Cosine similarity (direction alignment)
                            cos_sim = F.cosine_similarity(pred_fp32, tgt_fp32, dim=-1)
                            loss_direction = (1.0 - cos_sim).mean()
                            
                            # Loss 2: Smooth L1 on normalized features (shape matching)
                            pred_norm = F.normalize(pred_fp32, dim=-1)
                            tgt_norm = F.normalize(tgt_fp32, dim=-1)
                            loss_magnitude = F.smooth_l1_loss(pred_norm, tgt_norm)
                            
                            # Combined MLM loss
                            loss_mlm = 0.7 * loss_direction + 0.3 * loss_magnitude
                    else:
                        # All masked positions have zero ESM targets → keep graph connected
                        loss_mlm = raw_model.esm_regression_head(
                            hidden_states.mean(dim=(0, 1), keepdim=False).unsqueeze(0)
                        ).sum() * 0.0
                else:
                    # No masked positions → keep esm_regression_head in compute graph (DDP)
                    loss_mlm = raw_model.esm_regression_head(
                        hidden_states.mean(dim=(0, 1), keepdim=False).unsqueeze(0)
                    ).sum() * 0.0

                # MLM loss warmup: linearly ramp from 0 to 1 over first N optimizer steps
                if args.mlm_warmup_steps > 0 and optimizer_step < args.mlm_warmup_steps:
                    mlm_weight = optimizer_step / args.mlm_warmup_steps
                else:
                    mlm_weight = 1.0

                loss = mlm_weight * loss_mlm
                batch_mlm = loss_mlm.item()
                batch_cog = 0.0
                batch_strand = 0.0
                batch_cl = 0.0

                # --- 多任务 loss 聚合 ---
                # static_graph=True 要求所有参数每步都参与 backward.
                # 因此 head 的 loss 必须始终进入最终 loss 的计算图 (即使权重=0 也用 *0 保留图).

                # COG head (恒参与)
                loss_cog = criterion(outputs["cog_logits"].view(-1, 29), labels_cog.view(-1))
                loss = loss + args.cog_loss_weight * loss_cog
                batch_cog = loss_cog.item()

                # Strand head (恒参与)
                loss_strand = criterion(outputs["strand_logits"].view(-1, 3), labels_strand.view(-1))
                loss = loss + args.strand_loss_weight * loss_strand
                batch_strand = loss_strand.item()

                # Contrastive head
                # Phase 1: weight=0 时跳过昂贵的 all_gather + SupCon 计算,
                # 但仍让 contrastive_head 参与计算图 (DDP static_graph 要求).
                if "contrastive_projected" in outputs:
                    if args.contrastive_loss_weight > 0:
                        genus_labels = batch["genus_ids"].to(device)
                        loss_cl = supervised_contrastive_loss(
                            outputs["contrastive_projected"], genus_labels,
                            temperature=args.contrastive_temperature,
                            world_size=world_size, use_ddp=use_ddp,
                        )
                        loss = loss + args.contrastive_loss_weight * loss_cl
                        batch_cl = loss_cl.item()
                    else:
                        # weight=0: 保持 contrastive_head 在计算图中, 不做 all_gather
                        loss = loss + outputs["contrastive_projected"].sum() * 0.0

                loss = loss / args.gradient_accumulation_steps

            # P1-2/P1-6: NaN/Inf 检测。混合精度下 GradScaler 会 skip NaN/Inf step，
            # 静默丢失更新。前 200 步和之后每 100 步检查一次，其余步跳过避免
            # 每步 3×.item() GPU 同步破坏 CUDA pipeline 异步性。
            if global_step < 200 or global_step % 100 == 0:
                if not (torch.isfinite(loss_mlm).item()
                        and torch.isfinite(loss_cog).item()
                        and torch.isfinite(loss_strand).item()):
                    log_info(
                        f"[WARN] Non-finite loss at global_step={global_step}: "
                        f"mlm={batch_mlm}, cog={batch_cog}, strand={batch_strand}",
                        rank
                    )

            scaler.scale(loss).backward()

            total_loss += loss.item() * args.gradient_accumulation_steps
            total_mlm += batch_mlm
            total_cog += batch_cog
            total_strand += batch_strand
            total_cl += batch_cl
            num_batches += 1

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1

                # --- Step-level rolling checkpoint (容灾保存, 异步) ---
                if (args.save_every_n_steps > 0
                        and optimizer_step % args.save_every_n_steps == 0
                        and is_main_process(rank)):
                    async_save_checkpoint({
                        "epoch": epoch,
                        "model_state_dict": raw_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "args": vars(args),
                        "best_loss": best_loss,
                        "global_step": global_step,
                        "optimizer_step": optimizer_step,
                        "training_history": training_history,
                    }, os.path.join(args.ckpt_dir, "sage_v3_step_latest.pt"))
                    log_info(
                        f"  [Step-level ckpt] optimizer_step={optimizer_step} "
                        f"queued for async save",
                        rank
                    )

            global_step += 1

            # Slow step detection: 正常 ~3s/step，超过 60s 打印告警
            # P1-11: 直接用 logging.warning 而非 log_info, 这样所有 rank 都打印
            # (log_info 只在 rank==0 打, 会遮蔽故障 rank 的问题). 加 rank 前缀
            # 以便日志里定位到具体是哪个 rank stall.
            step_time = time.time() - step_start
            if step_time > 60.0:
                logging.warning(
                    f"[SLOW STEP rank={rank}] global_step={global_step} "
                    f"took {step_time:.1f}s "
                    f"(seq_len={input_ids.size(1)}, bs={input_ids.size(0)})"
                )

            if (batch_idx + 1) % args.log_interval == 0:
                avg = total_loss / num_batches
                lr = scheduler.get_last_lr()[0]
                cl_str = f", CL: {total_cl / num_batches:.4f}" if args.contrastive_loss_weight > 0 else ""
                log_info(
                    f"Epoch {epoch} | Step {batch_idx + 1} | Loss: {avg:.4f} "
                    f"(MLM: {total_mlm / num_batches:.4f}, "
                    f"COG: {total_cog / num_batches:.4f}"
                    f"{cl_str}) | "
                    f"LR: {lr:.2e} | mlm_w: {mlm_weight:.3f} | "
                    f"Seq_len: {input_ids.size(1)}",
                    rank
                )

            # Auto epoch boundary: islice above handles steps_per_epoch truncation
        # P1-3: 不做 leftover flush。对百万步训练来说，丢弃末尾不完整的
        # gradient accumulation 对收敛无实质影响，且完全避免了各 rank
        # num_batches 不对称导致的 DDP all-reduce 死锁风险。

        # --- Aggregate loss across ranks ---
        # P2-1 fix: 之前的实现先算 avg_loss = total_loss/num_batches, 再做
        # SUM(avg_loss)/SUM(num_batches), 等于多除了一次 num_batches.
        # 正确做法: all_reduce total_loss 和 num_batches 的原始累计值, 最后统一除.
        if use_ddp:
            # [total_loss, num_batches, total_mlm, total_cog, total_strand, total_cl]
            stats_tensor = torch.tensor(
                [total_loss, float(num_batches),
                 total_mlm, total_cog, total_strand, total_cl],
                device=device
            )
            dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
            all_total_loss = stats_tensor[0].item()
            all_num_batches = stats_tensor[1].item()
            all_mlm = stats_tensor[2].item()
            all_cog = stats_tensor[3].item()
            all_strand = stats_tensor[4].item()
            all_cl = stats_tensor[5].item()
            avg_loss = all_total_loss / max(all_num_batches, 1)
            avg_mlm = all_mlm / max(all_num_batches, 1)
            avg_cog = all_cog / max(all_num_batches, 1)
            avg_strand = all_strand / max(all_num_batches, 1)
            avg_cl = all_cl / max(all_num_batches, 1)
        else:
            avg_loss = total_loss / max(num_batches, 1)
            avg_mlm = total_mlm / max(num_batches, 1)
            avg_cog = total_cog / max(num_batches, 1)
            avg_strand = total_strand / max(num_batches, 1)
            avg_cl = total_cl / max(num_batches, 1)

        epoch_time = time.time() - epoch_start

        epoch_stats = {
            "epoch": epoch,
            "total_loss": avg_loss,
            "mlm_loss": avg_mlm,
            "cog_loss": avg_cog,
            "strand_loss": avg_strand,
            "contrastive_loss": avg_cl,
            "lr": scheduler.get_last_lr()[0],
            "optimizer_steps": optimizer_step,
            "epoch_time_sec": round(epoch_time, 1),
            "num_batches": num_batches,
            "world_size": world_size,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps * world_size,
        }

        if torch.cuda.is_available():
            epoch_stats["peak_gpu_memory_gb"] = round(
                torch.cuda.max_memory_allocated() / (1024 ** 3), 2
            )

        training_history.append(epoch_stats)

        log_info(
            f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
            f"Time: {epoch_time:.1f}s | Steps: {num_batches} | "
            f"GPUs: {world_size}",
            rank
        )

        # --- Save (rank 0 only) ---
        if is_main_process(rank):
            with open(os.path.join(args.ckpt_dir, "training_history.json"), 'w') as f:
                json.dump(training_history, f, indent=4)

            if avg_loss < best_loss:
                best_loss = avg_loss
                async_save_checkpoint({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "args": vars(args),
                    "best_loss": best_loss,
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    "training_history": training_history,
                }, os.path.join(args.ckpt_dir, "sage_v3_best.pt"))
                log_info(f"  New best model queued (loss={best_loss:.4f}, async)", rank)

            # Save periodic checkpoint every 10 epochs for long training
            if epoch % 10 == 0:
                async_save_checkpoint({
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "args": vars(args),
                    "best_loss": best_loss,
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    "training_history": training_history,
                }, os.path.join(args.ckpt_dir, f"sage_v3_epoch{epoch:03d}.pt"))
                log_info(f"  Periodic checkpoint queued: epoch {epoch} (async)", rank)

        # Synchronize all ranks before next epoch
        if use_ddp:
            dist.barrier()

    # Save final (rank 0 only)
    if is_main_process(rank):
        torch.save({
            "epoch": args.epochs,
            "model_state_dict": raw_model.state_dict(),
            "args": vars(args),
            "best_loss": best_loss,
            "training_history": training_history,
        }, os.path.join(args.ckpt_dir, "sage_v3_final.pt"))
        log_info(f"Training complete. Best loss: {best_loss:.4f}", rank)

    # Wait for any pending async checkpoint to finish before exit
    global _save_thread
    if _save_thread is not None and _save_thread.is_alive():
        log_info("Waiting for final async checkpoint to flush...", rank)
        _save_thread.join(timeout=600)

    cleanup_distributed(use_ddp)


if __name__ == "__main__":
    main()
