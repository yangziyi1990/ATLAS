"""
SAGE v3.1 Pre-training Evaluation (ESM Regression MLM)
========================================================
评估指标:
1. MLM Cosine Similarity (被 mask 位置的预测 vs 真实 ESM 特征)
2. MLM Smooth L1 Loss (normalized features)
3. Strand Accuracy
4. COG Accuracy (if available)
5. GPU Peak Memory
6. Training Speed (from training_history.json)
"""

import os
import sys
import json
import argparse
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GenomicLanguageModelV3
from dataset import GenomicSentenceDatasetV3, get_dynamic_mlm_collator
from model_configs import add_config_argument, get_model_config, apply_config_to_args


def detect_cog_degenerate(data):
    """检测 COG 数据是否退化"""
    cogs = data.get('cogs', None)
    if cogs is None:
        return True
    all_cog_values = set()
    for sample_cogs in cogs:
        if isinstance(sample_cogs, torch.Tensor):
            all_cog_values.update(sample_cogs.tolist())
        else:
            all_cog_values.update(sample_cogs)
        if len(all_cog_values) > 2:
            return False
    return len(all_cog_values) <= 2


def evaluate_pretrain(model, dataloader, device, vocab_size):
    """
    评估预训练模型的完整指标 (ESM Regression MLM):
    - MLM Cosine Similarity (avg)
    - MLM Smooth L1 Loss (on normalized features)
    - Strand Accuracy
    - COG Accuracy
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')

    total_cos_sim = 0.0
    total_smooth_l1 = 0.0
    total_mlm_tokens = 0

    total_strand_correct = 0
    total_strand_tokens = 0
    total_cog_correct = 0
    total_cog_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
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

            # Get ESM features and targets
            esm_features = batch.get("esm_features")
            esm_targets = batch.get("esm_targets")
            if esm_features is not None:
                esm_features = esm_features.to(device)
            if esm_targets is not None:
                esm_targets = esm_targets.to(device)

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
                    mask=attention_mask
                )

            hidden_states = outputs["hidden_states"]

            # --- MLM metrics (ESM Regression) ---
            mask_idx = (labels != -100)
            n_masked = mask_idx.sum().item()
            if n_masked > 0 and esm_targets is not None:
                masked_hidden = hidden_states[mask_idx].float()
                target_esm = esm_targets[mask_idx].float()
                
                # Filter valid targets (non-zero ESM)
                valid = (target_esm.abs().sum(dim=-1) > 1e-6)
                n_valid = valid.sum().item()
                
                if n_valid > 0:
                    predicted = model.esm_regression_head(masked_hidden[valid])
                    target_valid = target_esm[valid]
                    
                    # Cosine similarity
                    cos_sim = F.cosine_similarity(predicted.float(), target_valid, dim=-1)
                    total_cos_sim += cos_sim.sum().item()
                    
                    # Smooth L1 on normalized
                    pred_norm = F.normalize(predicted.float(), dim=-1)
                    tgt_norm = F.normalize(target_valid, dim=-1)
                    sl1 = F.smooth_l1_loss(pred_norm, tgt_norm, reduction='sum')
                    total_smooth_l1 += sl1.item()
                    
                    total_mlm_tokens += n_valid

            # --- Strand metrics ---
            strand_mask = (labels_strand != -100)
            n_strand = strand_mask.sum().item()
            if n_strand > 0 and "strand_logits" in outputs:
                strand_logits = outputs["strand_logits"]
                strand_pred = strand_logits[strand_mask].argmax(dim=-1)
                strand_true = labels_strand[strand_mask]
                total_strand_correct += (strand_pred == strand_true).sum().item()
                total_strand_tokens += n_strand

            # --- COG metrics ---
            cog_mask = (labels_cog != -100)
            n_cog = cog_mask.sum().item()
            if n_cog > 0 and "cog_logits" in outputs:
                cog_logits = outputs["cog_logits"]
                cog_pred = cog_logits[cog_mask].argmax(dim=-1)
                cog_true = labels_cog[cog_mask]
                total_cog_correct += (cog_pred == cog_true).sum().item()
                total_cog_tokens += n_cog

    # Compute final metrics
    avg_cos_sim = total_cos_sim / max(total_mlm_tokens, 1)
    avg_smooth_l1 = total_smooth_l1 / max(total_mlm_tokens, 1)
    # Combined loss (same formula as training)
    avg_mlm_loss = 0.7 * (1.0 - avg_cos_sim) + 0.3 * avg_smooth_l1

    results = {
        "mlm_cosine_similarity": round(avg_cos_sim, 4),
        "mlm_smooth_l1_loss": round(avg_smooth_l1, 4),
        "mlm_combined_loss": round(avg_mlm_loss, 4),
        "strand_acc": round(total_strand_correct / max(total_strand_tokens, 1), 4),
        "cog_acc": round(total_cog_correct / max(total_cog_tokens, 1), 4),
        "total_masked_tokens": total_mlm_tokens,
    }

    return results


def get_args():
    parser = argparse.ArgumentParser(description="SAGE v3 Pre-training Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (.pt file)")
    parser.add_argument("--data_path", type=str,
                        default="/opt/ai4g_chriszyyang/buddy2/SAGE/dataset/features/transformer_v3/transformer_inputs_v3.pt")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results (default: same as checkpoint dir)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_eval_batches", type=int, default=0,
                        help="Number of batches to evaluate (0 = all)")
    parser.add_argument("--seed", type=int, default=42)

    # Model architecture (should match training)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=2048)
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
    parser.add_argument("--use_targeted_masking", action="store_true", default=True)
    parser.add_argument("--no_targeted_masking", dest="use_targeted_masking", action="store_false")
    parser.add_argument("--use_token_level_gating", action="store_true", default=False)
    parser.add_argument("--use_segment_level_gating", action="store_true", default=False)

    # Masking params (should match training)
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--span_length", type=int, default=3)
    parser.add_argument("--operon_mask_prob", type=float, default=0.05)

    # 以下参数仅训练时使用, 评估时忽略, 但需要声明以兼容消融实验的 EXTRA_ARGS 透传
    parser.add_argument("--cog_loss_weight", type=float, default=0.5,
                        help="(ignored in eval) COG loss weight")
    parser.add_argument("--strand_loss_weight", type=float, default=0.5,
                        help="(ignored in eval) Strand loss weight")
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.0,
                        help="(ignored in eval) Contrastive loss weight")
    parser.add_argument("--contrastive_temperature", type=float, default=0.07,
                        help="(ignored in eval) Contrastive temperature")
    parser.add_argument("--contrastive_proj_dim", type=int, default=128,
                        help="(ignored in eval) Contrastive projection head dimension")
    parser.add_argument("--genomes_dir", type=str, default=None,
                        help="(ignored in eval) Genomes directory")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                        help="(ignored in eval)")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                        action="store_false", help="(ignored in eval)")

    # 模型配置预设
    add_config_argument(parser)

    args = parser.parse_args()

    if args.model_config:
        config = get_model_config(args.model_config)
        apply_config_to_args(args, config)

    return args


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    logging.info("=" * 60)
    logging.info("SAGE v3 Pre-training Evaluation")
    logging.info("=" * 60)
    logging.info(f"Checkpoint: {args.checkpoint}")
    logging.info(f"Device: {device}")

    # 1. Load dataset
    logging.info("Loading dataset...")
    dataset = GenomicSentenceDatasetV3(data_path=args.data_path)
    vocab_size = len(dataset.vocab)

    # COG degenerate detection
    data = torch.load(args.data_path, map_location='cpu', weights_only=False)
    exclude_cog = detect_cog_degenerate(data)
    del data

    collate_fn = get_dynamic_mlm_collator(
        dataset.vocab,
        mask_prob=args.mask_prob,
        span_length=args.span_length,
        use_targeted_masking=args.use_targeted_masking,
        operon_mask_prob=args.operon_mask_prob
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )

    # 2. Load model
    logging.info("Loading model...")
    feature_dim = None
    if dataset.esm_features is not None:
        feature_dim = dataset.esm_features.shape[1]

    model = GenomicLanguageModelV3(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
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
        contrastive_proj_dim=args.contrastive_proj_dim,
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning(f"Missing keys: {len(missing)} — {missing[:5]}")
    if unexpected:
        logging.warning(f"Unexpected keys: {len(unexpected)} — {unexpected[:5]}")
    logging.info("Model loaded successfully.")

    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # 3. Evaluate
    logging.info("Evaluating...")
    start_time = time.time()

    if args.num_eval_batches > 0:
        # Limit evaluation to N batches
        limited_batches = []
        for i, batch in enumerate(dataloader):
            if i >= args.num_eval_batches:
                break
            limited_batches.append(batch)

        # Create a simple wrapper
        class BatchList:
            def __init__(self, batches):
                self.batches = batches
            def __iter__(self):
                return iter(self.batches)
            def __len__(self):
                return len(self.batches)

        eval_loader = BatchList(limited_batches)
    else:
        eval_loader = dataloader

    results = evaluate_pretrain(model, eval_loader, device, vocab_size)

    eval_time = time.time() - start_time
    results["eval_time_sec"] = round(eval_time, 1)
    results["num_samples"] = len(dataset)

    # GPU memory
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        results["peak_gpu_memory_gb"] = round(peak_mem, 2)

    # Fusion weights
    if hasattr(model, 'feature_fusion') and hasattr(model.feature_fusion, 'get_fusion_weights'):
        weights = model.feature_fusion.get_fusion_weights()
        if weights is not None:
            feature_names = ["gene", "strand", "replicon", "cog", "contig", "mutation", "distance"]
            if model.exclude_cog:
                feature_names = ["gene", "strand", "replicon", "contig", "mutation", "distance"]
            results["fusion_weights"] = {n: round(w.item(), 4) for n, w in zip(feature_names, weights)}

    # Model config
    results["model_config"] = {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dim_feedforward": args.dim_feedforward,
        "use_swiglu": args.use_swiglu,
        "use_hierarchical_attention": args.use_hierarchical_attention,
        "use_gated_fusion": args.use_gated_fusion,
        "use_distance_bias": args.use_distance_bias,
        "use_targeted_masking": args.use_targeted_masking,
        "use_token_level_gating": args.use_token_level_gating,
        "use_segment_level_gating": args.use_segment_level_gating,
        "exclude_cog": exclude_cog,
    }

    # Training history (if available)
    ckpt_dir = os.path.dirname(args.checkpoint)
    history_path = os.path.join(ckpt_dir, "training_history.json")
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        if history:
            last_epoch = history[-1]
            results["training_epochs"] = last_epoch.get("epoch", len(history))
            results["final_train_loss"] = last_epoch.get("total_loss", None)
            # Extract fusion weight evolution
            fw_history = []
            for ep in history:
                if "fusion_weights" in ep:
                    fw_history.append({"epoch": ep["epoch"], **ep["fusion_weights"]})
            if fw_history:
                results["fusion_weight_evolution"] = fw_history

    # 4. Output results
    logging.info("\n" + "=" * 60)
    logging.info("EVALUATION RESULTS (ESM Regression MLM)")
    logging.info("=" * 60)
    logging.info(f"  MLM Cosine Sim:   {results['mlm_cosine_similarity']:.4f}")
    logging.info(f"  MLM Smooth L1:    {results['mlm_smooth_l1_loss']:.4f}")
    logging.info(f"  MLM Combined:     {results['mlm_combined_loss']:.4f}")
    logging.info(f"  Strand Acc:       {results['strand_acc']:.4f}")
    logging.info(f"  COG Acc:          {results['cog_acc']:.4f}")
    if "peak_gpu_memory_gb" in results:
        logging.info(f"  Peak GPU Mem:     {results['peak_gpu_memory_gb']:.2f} GB")
    if "fusion_weights" in results:
        logging.info(f"  Fusion Weights:   {results['fusion_weights']}")
    logging.info("=" * 60)

    # Save
    output_dir = args.output_dir or ckpt_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pretrain_eval_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
