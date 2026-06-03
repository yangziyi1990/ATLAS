"""
SAGE v3 Training Script
========================
核心升级:
1. 支持 v3 架构 (Progressive Gated Fusion + 三阶段分层注意力)
2. 动态 Padding Collator
3. 梯度累积 (支持更大 effective batch size)
4. 更丰富的日志 (含 fusion weights 可解释性输出)
"""

import argparse
import logging
import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from model import GenomicLanguageModelV3
from dataset import GenomicSentenceDatasetV3, get_dynamic_mlm_collator
from model_configs import add_config_argument, get_model_config, apply_config_to_args

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_genus_ids(dataset, genomes_dir):
    """
    将 genome_ids 映射为 genus integer labels (用于对比学习).
    
    通过遍历 genomes_dir 的目录结构 (genomes_dir/Genus/GCA_xxx/) 建立映射.
    """
    genome_ids = dataset.genome_ids
    if genome_ids is None:
        logging.warning("Dataset has no genome_ids, cannot build genus labels.")
        return None
    
    gca_to_genus = {}
    if os.path.isdir(genomes_dir):
        for genus_dir in os.listdir(genomes_dir):
            genus_path = os.path.join(genomes_dir, genus_dir)
            if not os.path.isdir(genus_path):
                continue
            for gca_dir in os.listdir(genus_path):
                if gca_dir.startswith("GCA_"):
                    gca_to_genus[gca_dir] = genus_dir
    else:
        logging.warning(f"Genomes directory not found: {genomes_dir}")
        return None
    
    genus_names = [gca_to_genus.get(gid, 'Unknown') for gid in genome_ids]
    unknown_count = sum(1 for g in genus_names if g == 'Unknown')
    
    unique_genera = sorted(set(genus_names))
    genus_to_id = {g: i for i, g in enumerate(unique_genera)}
    genus_ids = [genus_to_id[g] for g in genus_names]
    
    logging.info(f"Genus labels: {len(unique_genera)} genera, "
                 f"{len(genus_names) - unknown_count}/{len(genus_names)} mapped "
                 f"({unknown_count} unknown)")
    return genus_ids


def supervised_contrastive_loss(embeddings, labels, temperature=0.07):
    """
    Supervised Contrastive Loss (SupCon, Khosla et al. 2020).
    
    同一 genus 的样本互为正样本对，不同 genus 为负样本对。
    
    Args:
        embeddings: [B, D] L2-normalized strain embeddings
        labels: [B] genus integer labels
        temperature: scalar
    Returns:
        loss: scalar
    """
    B = embeddings.size(0)
    if B <= 1:
        return torch.tensor(0.0, device=embeddings.device)
    
    # 过滤掉无效标签 (genus_id == -1, 即 collate_fn 中没有 genus_id 的样本)
    valid_mask = labels >= 0
    if valid_mask.sum() <= 1:
        return torch.tensor(0.0, device=embeddings.device)
    
    embeddings = embeddings[valid_mask]
    labels = labels[valid_mask]
    B = embeddings.size(0)
    
    # 相似度矩阵 [B, B]
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    
    # 正样本 mask: 同一 genus 的样本对
    labels_col = labels.unsqueeze(0)  # [1, B]
    labels_row = labels.unsqueeze(1)  # [B, 1]
    pos_mask = (labels_row == labels_col).float()  # [B, B]
    pos_mask = pos_mask * (1.0 - torch.eye(B, device=embeddings.device))  # 排除自身
    
    # 如果没有正样本对，返回 0
    has_positive = pos_mask.sum(dim=1) > 0
    if has_positive.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    
    # Log-sum-exp trick for numerical stability
    logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()
    
    # 分母: 所有非自身样本 (用非 inplace 方式排除对角线，避免 autograd 报错)
    exp_logits = torch.exp(logits)
    self_mask = 1.0 - torch.eye(B, device=embeddings.device)
    exp_logits = exp_logits * self_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    
    # 分子: 正样本对的平均 log probability
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
    
    # 只对有正样本的样本计算 loss
    loss = -mean_log_prob_pos[has_positive].mean()
    return loss


def get_args():
    parser = argparse.ArgumentParser(description="SAGE v3: Strain-Aware Genomic Language Model Pre-training")
    parser.add_argument("--data_path", type=str,
                        default="/opt/ai4g_chriszyyang/buddy2/SAGE/dataset/features/transformer_v3/transformer_inputs_v3.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size per step (降低以避免 OOM)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)")
    
    # 模型架构参数
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6,
                        help="Number of transformer layers (建议为3的倍数以匹配三阶段)")
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--distance_window", type=int, default=128,
                        help="Distance bias window size (±N genes)")
    
    # 架构开关
    parser.add_argument("--use_swiglu", action="store_true", default=True)
    parser.add_argument("--no_swiglu", dest="use_swiglu", action="store_false")
    parser.add_argument("--use_hierarchical_attention", action="store_true", default=True)
    parser.add_argument("--no_hierarchical_attention", dest="use_hierarchical_attention", action="store_false")
    parser.add_argument("--use_gated_fusion", action="store_true", default=True)
    parser.add_argument("--no_gated_fusion", dest="use_gated_fusion", action="store_false")
    parser.add_argument("--use_distance_bias", action="store_true", default=True)
    parser.add_argument("--no_distance_bias", dest="use_distance_bias", action="store_false")
    parser.add_argument("--use_targeted_masking", action="store_true", default=True)
    parser.add_argument("--no_targeted_masking", dest="use_targeted_masking", action="store_false")
    parser.add_argument("--use_token_level_gating", action="store_true", default=False,
                        help="Use token-level gating in fusion (需要大数据集)")
    parser.add_argument("--use_segment_level_gating", action="store_true", default=False,
                        help="Use segment-level (contig) gating in fusion (中间粒度方案)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                        help="Enable gradient checkpointing to save ~60%% GPU memory")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    
    # 训练超参数
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_ratio", type=float, default=0.10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.15,
                        help="Dropout rate (小数据建议 0.15)")
    
    # Loss 权重
    parser.add_argument("--cog_loss_weight", type=float, default=0.5)
    parser.add_argument("--strand_loss_weight", type=float, default=0.5)
    
    # 对比学习参数
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.1,
                        help="Weight for genus-level SupCon loss (0=disabled, 0.1=recommended)")
    parser.add_argument("--contrastive_temperature", type=float, default=0.07,
                        help="Temperature for SupCon InfoNCE loss")
    parser.add_argument("--contrastive_proj_dim", type=int, default=128,
                        help="Output dimension for contrastive projection head")
    parser.add_argument("--genomes_dir", type=str,
                        default="/opt/ai4g_chriszyyang/buddy2/SAGE/dataset/genomes",
                        help="Path to genomes directory for genus label inference")
    
    # 掩码参数
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--span_length", type=int, default=3)
    parser.add_argument("--operon_mask_prob", type=float, default=0.05,
                        help="Probability of masking entire detected operons")
    
    # 实验管理
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="sage_v3")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # 模型配置预设 (可选, 设置后覆盖架构/训练默认值)
    add_config_argument(parser)
    
    args = parser.parse_args()
    
    # 如果指定了 --model_config, 用预设值覆盖默认参数
    if args.model_config:
        config = get_model_config(args.model_config)
        apply_config_to_args(args, config)
    
    return args


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 互斥校验: token-level 和 segment-level gating 不能同时开启
    if args.use_token_level_gating and args.use_segment_level_gating:
        raise ValueError("--use_token_level_gating and --use_segment_level_gating are mutually exclusive.")
    
    # Seed 固定 (确保消融实验可复现)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
    
    logging.info(f"{'='*60}")
    logging.info(f"SAGE v3 Pre-training | Device: {device} | Exp: {args.exp_name} | Seed: {args.seed}")
    logging.info(f"{'='*60}")
    logging.info(f"Architecture: d_model={args.d_model}, layers={args.num_layers}, heads={args.num_heads}, "
                 f"ffn={args.dim_feedforward}, max_seq_len={args.max_seq_len}")
    logging.info(f"Features: SwiGLU={args.use_swiglu}, Hierarchical={args.use_hierarchical_attention}, "
                 f"GatedFusion={args.use_gated_fusion}, DistBias={args.use_distance_bias}, "
                 f"TargetedMask={args.use_targeted_masking}, TokenGating={args.use_token_level_gating}, "
                 f"SegGating={args.use_segment_level_gating}")
    logging.info(f"Contrastive: weight={args.contrastive_loss_weight}, temp={args.contrastive_temperature}")
    logging.info(f"Training: bs={args.batch_size}, grad_accum={args.gradient_accumulation_steps}, "
                 f"effective_bs={args.batch_size * args.gradient_accumulation_steps}, "
                 f"lr={args.lr}, wd={args.weight_decay}, dropout={args.dropout}")
    
    # 1. 准备数据集
    logging.info("1. Loading v3 Genomic Sentence Dataset...")
    dataset = GenomicSentenceDatasetV3(data_path=args.data_path)
    vocab_size = len(dataset.vocab)
    
    # COG 退化检测
    exclude_cog = False
    if hasattr(dataset, 'cogs') and dataset.cogs is not None:
        all_cog_values = set()
        for sample_cogs in dataset.cogs:
            if isinstance(sample_cogs, torch.Tensor):
                all_cog_values.update(sample_cogs.tolist())
            else:
                all_cog_values.update(sample_cogs)
            if len(all_cog_values) > 2:
                break
        if len(all_cog_values) <= 2:
            logging.warning(f"COG data degenerate (unique: {all_cog_values}). Setting cog_loss_weight=0, exclude_cog=True")
            args.cog_loss_weight = 0.0
            exclude_cog = True
        else:
            logging.info(f"COG data valid ({len(all_cog_values)} unique categories).")
    
    # Genus 标签构建 (对比学习)
    if args.contrastive_loss_weight > 0:
        genus_ids = build_genus_ids(dataset, args.genomes_dir)
        if genus_ids is not None:
            dataset.genus_ids = genus_ids
        else:
            logging.warning("Failed to build genus IDs. Disabling contrastive loss.")
            args.contrastive_loss_weight = 0.0
    
    collate_fn = get_dynamic_mlm_collator(
        dataset.vocab,
        mask_prob=args.mask_prob,
        span_length=args.span_length,
        use_targeted_masking=args.use_targeted_masking,
        operon_mask_prob=args.operon_mask_prob
    )
    # BUG9 fix: 添加 worker_init_fn 确保多 worker DataLoader 可复现
    # 注: epoch 信息通过 DataLoader 的 persistent_workers=False (默认) 在每个 epoch 重新 fork 时
    # 由 torch 的内部种子管理提供不同的随机性
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    
    logging.info(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")
    
    # 2. 初始化模型
    logging.info("2. Initializing SAGE v3 Model...")
    
    feature_dim = None
    if dataset.esm_features is not None:
        feature_dim = dataset.esm_features.shape[1]
        logging.info(f"Detected continuous ESM features (dim: {feature_dim}).")
    
    model = GenomicLanguageModelV3(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        feature_dim=feature_dim,
        esm_features=dataset.esm_features,
        use_swiglu=args.use_swiglu,
        use_hierarchical_attention=args.use_hierarchical_attention,
        use_gated_fusion=args.use_gated_fusion,
        use_distance_bias=args.use_distance_bias,
        exclude_cog=exclude_cog,
        max_seq_len=args.max_seq_len,
        distance_window=args.distance_window,
        use_token_level_gating=args.use_token_level_gating,
        use_segment_level_gating=args.use_segment_level_gating,
        gradient_checkpointing=args.gradient_checkpointing,
        contrastive_proj_dim=args.contrastive_proj_dim,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Parameters: Total={total_params:,} | Trainable={trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # 3. 优化器 + 学习率调度器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # BUG2 fix: total_steps 需要包含残余 batch 的额外优化器步数
    steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    if len(dataloader) % args.gradient_accumulation_steps != 0:
        steps_per_epoch += 1
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=args.lr * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    
    logging.info(f"LR Schedule: Warmup {warmup_steps} → Cosine decay, total {total_steps} optimizer steps")
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # 检查点目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = args.ckpt_dir if args.ckpt_dir else os.path.join(base_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    training_history = []
    best_loss = float('inf')
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # 4. 开始预训练
    logging.info("3. Starting SAGE v3 Pre-training...")
    global_step = 0
    optimizer_step = 0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        total_mlm = 0
        total_cog = 0
        total_strand = 0
        total_cl = 0
        num_batches = 0
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']  # BUG1 fix: 初始化 current_lr 防止空 dataloader 时未定义
        
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            strand_ids = batch["strand_ids"].to(device)
            replicon_ids = batch["replicon_ids"].to(device)
            cog_ids = batch["cog_ids"].to(device)
            contig_ids = batch["contig_ids"].to(device)
            mutation_ids = batch["mutation_ids"].to(device)
            distance_ids = batch["distance_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids_batch = batch["position_ids"].to(device)
            
            labels = batch["labels"].to(device)
            labels_strand = batch["labels_strand"].to(device)
            labels_cog = batch["labels_cog"].to(device)
            
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(
                    gene_seqs=input_ids,
                    strand_ids=strand_ids,
                    replicon_ids=replicon_ids,
                    cog_ids=cog_ids,
                    contig_ids=contig_ids,
                    mutation_ids=mutation_ids,
                    distance_ids=distance_ids,
                    position_ids=position_ids_batch,
                    mask=attention_mask
                )
                
                # MLM Loss
                hidden_states = outputs["hidden_states"]
                mask_idx = (labels != -100)
                if mask_idx.sum() > 0:
                    masked_hidden = hidden_states[mask_idx]
                    masked_logits = model.mlm_head(masked_hidden)
                    masked_labels = labels[mask_idx]
                    loss_mlm = criterion(masked_logits, masked_labels)
                else:
                    loss_mlm = torch.tensor(0.0, device=device)
                
                loss = loss_mlm
                batch_mlm = loss_mlm.item()
                batch_cog = 0.0
                batch_strand = 0.0
                
                # COG Auxiliary Loss
                if "cog_logits" in outputs and args.cog_loss_weight > 0:
                    loss_cog = criterion(outputs["cog_logits"].view(-1, 29), labels_cog.view(-1))
                    loss = loss + args.cog_loss_weight * loss_cog
                    batch_cog = loss_cog.item()
                
                # Strand Auxiliary Loss
                if "strand_logits" in outputs and args.strand_loss_weight > 0:
                    loss_strand = criterion(outputs["strand_logits"].view(-1, 3), labels_strand.view(-1))
                    loss = loss + args.strand_loss_weight * loss_strand
                    batch_strand = loss_strand.item()
                
                # Contrastive Loss (genus-level SupCon)
                batch_cl = 0.0
                if args.contrastive_loss_weight > 0 and "strain_embedding" in outputs:
                    genus_labels = batch["genus_ids"].to(device)
                    loss_cl = supervised_contrastive_loss(
                        outputs["strain_embedding"], genus_labels,
                        temperature=args.contrastive_temperature
                    )
                    loss = loss + args.contrastive_loss_weight * loss_cl
                    batch_cl = loss_cl.item()
                
                # 梯度累积: 除以累积步数
                loss = loss / args.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            total_loss += loss.item() * args.gradient_accumulation_steps
            total_mlm += batch_mlm
            total_cog += batch_cog
            total_strand += batch_strand
            total_cl += batch_cl
            num_batches += 1
            
            # 梯度累积: 每 N 步更新一次
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1
            
            global_step += 1
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                "lr": f"{current_lr:.2e}",
                "seq_len": input_ids.size(1)
            })
        
        # 处理最后不足 gradient_accumulation_steps 的 batch
        if num_batches % args.gradient_accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_step += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        epoch_elapsed = time.time() - epoch_start_time
        
        epoch_stats = {
            "epoch": epoch,
            "total_loss": avg_loss,
            "mlm_loss": total_mlm / num_batches if num_batches > 0 else 0,
            "cog_loss": total_cog / num_batches if num_batches > 0 else 0,
            "strand_loss": total_strand / num_batches if num_batches > 0 else 0,
            "contrastive_loss": total_cl / num_batches if num_batches > 0 else 0,
            "lr": current_lr,
            "optimizer_steps": optimizer_step,
            "epoch_time_sec": round(epoch_elapsed, 1),
            "batches_per_sec": round(num_batches / max(epoch_elapsed, 0.01), 2),
        }
        
        # GPU 内存峰值记录
        if torch.cuda.is_available():
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            epoch_stats["peak_gpu_memory_gb"] = round(peak_mem_gb, 2)
        
        # 输出融合权重 (可解释性)
        if hasattr(model, 'feature_fusion') and hasattr(model.feature_fusion, 'get_fusion_weights'):
            weights = model.feature_fusion.get_fusion_weights()
            if weights is not None:
                feature_names = ["gene", "strand", "replicon", "cog", "contig", "mutation", "distance"]
                if model.exclude_cog:
                    feature_names = ["gene", "strand", "replicon", "contig", "mutation", "distance"]
                weight_dict = {n: round(w.item(), 4) for n, w in zip(feature_names, weights)}
                epoch_stats["fusion_weights"] = weight_dict
                if epoch % 10 == 0:
                    logging.info(f"  Fusion weights: {weight_dict}")
        
        training_history.append(epoch_stats)
        
        logging.info(
            f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} "
            f"(MLM: {epoch_stats['mlm_loss']:.4f}, COG: {epoch_stats['cog_loss']:.4f}, "
            f"Strand: {epoch_stats['strand_loss']:.4f}"
            + (f", CL: {epoch_stats['contrastive_loss']:.4f}" if args.contrastive_loss_weight > 0 else "")
            + f") | LR: {current_lr:.2e} | OptSteps: {optimizer_step} | "
            f"Time: {epoch_elapsed:.1f}s ({epoch_stats['batches_per_sec']:.1f} batch/s)"
            + (f" | GPU: {epoch_stats.get('peak_gpu_memory_gb', 'N/A')} GB" if torch.cuda.is_available() else "")
        )
        
        # Save history
        history_path = os.path.join(ckpt_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=4)
        
        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "args": vars(args),
                "vocab_size": vocab_size,
                "best_loss": best_loss
            }, os.path.join(ckpt_dir, "sage_v3_best.pt"))
            logging.info(f"  New best model saved (loss={best_loss:.4f})")
    
    logging.info("=" * 60)
    logging.info("Pre-training Complete!")
    
    # GPU 内存汇总
    if torch.cuda.is_available():
        final_peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logging.info(f"Peak GPU Memory: {final_peak_mem:.2f} GB")
    
    # Save final model
    final_path = os.path.join(ckpt_dir, "sage_v3_final.pt")
    torch.save(model.state_dict(), final_path)
    logging.info(f"Final model: {final_path}")
    logging.info(f"Best loss: {best_loss:.4f}")
    
    # Save experiment summary
    exp_summary = {
        "exp_name": args.exp_name,
        "seed": args.seed,
        "best_loss": best_loss,
        "total_epochs": args.epochs,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "args": vars(args),
    }
    if torch.cuda.is_available():
        exp_summary["peak_gpu_memory_gb"] = round(final_peak_mem, 2)
    with open(os.path.join(ckpt_dir, "experiment_summary.json"), 'w') as f:
        json.dump(exp_summary, f, indent=2)
    
    # 释放资源
    del model, optimizer, scheduler, scaler, dataloader, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("GPU resources released.")


if __name__ == "__main__":
    main()
