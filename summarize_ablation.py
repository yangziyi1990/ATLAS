"""
SAGE v3 Ablation Study: Result Summarization
==============================================
汇总所有消融实验的预训练评估和下游聚类评估结果.
生成:
1. ablation_summary_v3.json — 结构化结果汇总
2. 终端表格 — 快速对比
3. ablation_comparison.csv — 便于后续分析
"""

import os
import sys
import json
import argparse
import glob
import logging
from collections import OrderedDict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 已确认不再纳入 v3 消融对比的实验
SKIP_EXPERIMENTS = {"wo_cog"}


def load_json_safe(path):
    """安全加载 JSON 文件"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load {path}: {e}")
        return None


def collect_experiment_results(results_dir, ablation_dir):
    """
    收集所有实验的结果.
    
    搜索路径:
    - results_dir/<exp_name>/pretrain_eval_results.json  (预训练指标)
    - results_dir/<exp_name>/clustering_results_v3.json  (下游聚类指标)
    - ablation_dir/<exp_name>/training_history.json      (训练历史)
    - ablation_dir/<exp_name>/experiment_summary.json     (实验配置)
    """
    all_results = OrderedDict()
    
    # 从 results_dir 收集
    if os.path.isdir(results_dir):
        for exp_dir in sorted(os.listdir(results_dir)):
            exp_path = os.path.join(results_dir, exp_dir)
            if not os.path.isdir(exp_path):
                continue
            
            exp_name = exp_dir
            if exp_name in SKIP_EXPERIMENTS:
                logging.info(f"Skip deprecated ablation result: {exp_name}")
                continue
            result = {"exp_name": exp_name}
            
            # 预训练评估
            pretrain_results = load_json_safe(os.path.join(exp_path, "pretrain_eval_results.json"))
            if pretrain_results:
                result["pretrain"] = {
                    "mlm_loss": pretrain_results.get("mlm_loss"),
                    "perplexity": pretrain_results.get("perplexity"),
                    "mlm_top1_acc": pretrain_results.get("mlm_top1_acc"),
                    "mlm_top5_acc": pretrain_results.get("mlm_top5_acc"),
                    "mlm_top10_acc": pretrain_results.get("mlm_top10_acc"),
                    "strand_acc": pretrain_results.get("strand_acc"),
                    "peak_gpu_memory_gb": pretrain_results.get("peak_gpu_memory_gb"),
                    "fusion_weights": pretrain_results.get("fusion_weights"),
                }
            
            # 下游聚类
            clustering_results = load_json_safe(os.path.join(exp_path, "clustering_results_v3.json"))
            if clustering_results:
                attn_pool = clustering_results.get("sage_v3_attn_pool", {})
                mean_pool = clustering_results.get("sage_v3_mean_pool", {})
                result["clustering"] = {
                    "attn_pool": {
                        "ARI": attn_pool.get("ARI"),
                        "NMI": attn_pool.get("NMI"),
                        "Silhouette": attn_pool.get("Silhouette"),
                    },
                    "mean_pool": {
                        "ARI": mean_pool.get("ARI"),
                        "NMI": mean_pool.get("NMI"),
                        "Silhouette": mean_pool.get("Silhouette"),
                    },
                }
            
            all_results[exp_name] = result
    
    # 从 ablation_dir 补充训练信息
    if os.path.isdir(ablation_dir):
        for exp_dir in sorted(os.listdir(ablation_dir)):
            exp_path = os.path.join(ablation_dir, exp_dir)
            if not os.path.isdir(exp_path):
                continue
            
            exp_name = exp_dir
            if exp_name in SKIP_EXPERIMENTS:
                logging.info(f"Skip deprecated ablation checkpoint: {exp_name}")
                continue
            if exp_name not in all_results:
                all_results[exp_name] = {"exp_name": exp_name}
            
            # 实验配置
            exp_summary = load_json_safe(os.path.join(exp_path, "experiment_summary.json"))
            if exp_summary:
                all_results[exp_name]["config"] = {
                    "seed": exp_summary.get("seed"),
                    "best_loss": exp_summary.get("best_loss"),
                    "total_params": exp_summary.get("total_params"),
                    "peak_gpu_memory_gb": exp_summary.get("peak_gpu_memory_gb"),
                }
                args_info = exp_summary.get("args", {})
                all_results[exp_name]["config"]["architecture"] = {
                    "use_swiglu": args_info.get("use_swiglu"),
                    "use_hierarchical_attention": args_info.get("use_hierarchical_attention"),
                    "use_gated_fusion": args_info.get("use_gated_fusion"),
                    "use_distance_bias": args_info.get("use_distance_bias"),
                    "use_targeted_masking": args_info.get("use_targeted_masking"),
                    "use_token_level_gating": args_info.get("use_token_level_gating"),
                    "use_segment_level_gating": args_info.get("use_segment_level_gating"),
                    "gradient_checkpointing": args_info.get("gradient_checkpointing"),
                    "span_length": args_info.get("span_length"),
                    "operon_mask_prob": args_info.get("operon_mask_prob"),
                }
            
            # 训练历史 — 提取关键统计
            history = load_json_safe(os.path.join(exp_path, "training_history.json"))
            if history and len(history) > 0:
                last = history[-1]
                
                # 找到 best epoch
                best_epoch = min(history, key=lambda x: x.get("total_loss", float('inf')))
                
                # 计算平均训练速度
                epoch_times = [ep.get("epoch_time_sec", 0) for ep in history if ep.get("epoch_time_sec")]
                avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0
                avg_batch_speed = sum(ep.get("batches_per_sec", 0) for ep in history if ep.get("batches_per_sec")) / max(len(epoch_times), 1)
                
                all_results[exp_name]["training"] = {
                    "total_epochs": last.get("epoch", len(history)),
                    "best_epoch": best_epoch.get("epoch"),
                    "best_train_loss": round(best_epoch.get("total_loss", 0), 4),
                    "final_train_loss": round(last.get("total_loss", 0), 4),
                    "avg_epoch_time_sec": round(avg_epoch_time, 1),
                    "avg_batches_per_sec": round(avg_batch_speed, 2),
                    "peak_gpu_memory_gb": last.get("peak_gpu_memory_gb"),
                }
                
                # Fusion weight 演化 (首尾对比)
                fw_first = None
                fw_last = None
                for ep in history:
                    if "fusion_weights" in ep:
                        if fw_first is None:
                            fw_first = ep["fusion_weights"]
                        fw_last = ep["fusion_weights"]
                if fw_first and fw_last:
                    all_results[exp_name]["training"]["fusion_weights_first"] = fw_first
                    all_results[exp_name]["training"]["fusion_weights_last"] = fw_last
    
    return all_results


def print_summary_table(all_results):
    """打印格式化的对比表格"""
    print("\n" + "=" * 120)
    print("SAGE v3 ABLATION STUDY — SUMMARY")
    print("=" * 120)
    
    # 预训练指标表
    header = f"{'Experiment':<25} {'PPL':>8} {'Top1':>8} {'Top5':>8} {'Top10':>8} {'StrandAcc':>10} {'GPU(GB)':>8} {'Speed':>10}"
    print("\n--- Pre-training Metrics ---")
    print(header)
    print("-" * 112)
    
    for exp_name, result in all_results.items():
        pt = result.get("pretrain", {})
        tr = result.get("training", {})
        
        ppl = pt.get("perplexity", tr.get("best_train_loss", "—"))
        top1 = pt.get("mlm_top1_acc", "—")
        top5 = pt.get("mlm_top5_acc", "—")
        top10 = pt.get("mlm_top10_acc", "—")
        strand = pt.get("strand_acc", "—")
        gpu = pt.get("peak_gpu_memory_gb", tr.get("peak_gpu_memory_gb", "—"))
        speed = tr.get("avg_batches_per_sec", "—")
        
        # Format
        ppl_str = f"{ppl:.2f}" if isinstance(ppl, (int, float)) else str(ppl)
        top1_str = f"{top1:.4f}" if isinstance(top1, (int, float)) else str(top1)
        top5_str = f"{top5:.4f}" if isinstance(top5, (int, float)) else str(top5)
        top10_str = f"{top10:.4f}" if isinstance(top10, (int, float)) else str(top10)
        strand_str = f"{strand:.4f}" if isinstance(strand, (int, float)) else str(strand)
        gpu_str = f"{gpu:.2f}" if isinstance(gpu, (int, float)) else str(gpu)
        speed_str = f"{speed:.2f} b/s" if isinstance(speed, (int, float)) else str(speed)
        
        print(f"{exp_name:<25} {ppl_str:>8} {top1_str:>8} {top5_str:>8} {top10_str:>8} {strand_str:>10} {gpu_str:>8} {speed_str:>10}")
    
    # 下游聚类指标表
    has_clustering = any("clustering" in r for r in all_results.values())
    if has_clustering:
        print("\n--- Downstream Clustering Metrics (Attention Pooling) ---")
        header2 = f"{'Experiment':<25} {'ARI':>8} {'NMI':>8} {'Silhouette':>12}"
        print(header2)
        print("-" * 60)
        
        for exp_name, result in all_results.items():
            cl = result.get("clustering", {}).get("attn_pool", {})
            if not cl:
                continue
            ari = cl.get("ARI", "—")
            nmi = cl.get("NMI", "—")
            sil = cl.get("Silhouette", "—")
            
            ari_str = f"{ari:.4f}" if isinstance(ari, (int, float)) else str(ari)
            nmi_str = f"{nmi:.4f}" if isinstance(nmi, (int, float)) else str(nmi)
            sil_str = f"{sil:.4f}" if isinstance(sil, (int, float)) else str(sil)
            
            print(f"{exp_name:<25} {ari_str:>8} {nmi_str:>8} {sil_str:>12}")
    
    print("\n" + "=" * 120)


def save_csv(all_results, output_path):
    """保存为 CSV 便于后续分析"""
    csv_path = output_path.replace(".json", ".csv")
    
    rows = []
    header = [
        "experiment", "ppl", "mlm_top1", "mlm_top5", "mlm_top10",
        "strand_acc", "train_loss", "gpu_gb", "speed_batch_s",
        "ari_attn", "nmi_attn", "sil_attn",
        "ari_mean", "nmi_mean", "sil_mean"
    ]
    rows.append(",".join(header))
    
    for exp_name, result in all_results.items():
        pt = result.get("pretrain", {})
        tr = result.get("training", {})
        cl_attn = result.get("clustering", {}).get("attn_pool", {})
        cl_mean = result.get("clustering", {}).get("mean_pool", {})
        
        row = [
            exp_name,
            str(pt.get("perplexity", "")),
            str(pt.get("mlm_top1_acc", "")),
            str(pt.get("mlm_top5_acc", "")),
            str(pt.get("mlm_top10_acc", "")),
            str(pt.get("strand_acc", "")),
            str(tr.get("best_train_loss", "")),
            str(pt.get("peak_gpu_memory_gb", tr.get("peak_gpu_memory_gb", ""))),
            str(tr.get("avg_batches_per_sec", "")),
            str(cl_attn.get("ARI", "")),
            str(cl_attn.get("NMI", "")),
            str(cl_attn.get("Silhouette", "")),
            str(cl_mean.get("ARI", "")),
            str(cl_mean.get("NMI", "")),
            str(cl_mean.get("Silhouette", "")),
        ]
        rows.append(",".join(row))
    
    with open(csv_path, 'w') as f:
        f.write("\n".join(rows))
    logging.info(f"CSV saved to {csv_path}")


def compute_deltas(all_results):
    """计算相对于 full_model 的增量变化"""
    baseline_name = None
    for name in all_results:
        if "full_model" in name:
            baseline_name = name
            break
    
    if not baseline_name:
        logging.warning("No full_model baseline found, skipping delta computation")
        return
    
    baseline = all_results[baseline_name]
    baseline_pt = baseline.get("pretrain", {})
    baseline_cl = baseline.get("clustering", {}).get("attn_pool", {})
    
    for exp_name, result in all_results.items():
        if exp_name == baseline_name:
            continue
        
        deltas = {}
        pt = result.get("pretrain", {})
        cl = result.get("clustering", {}).get("attn_pool", {})
        
        # PPL delta (负数 = 改善)
        if pt.get("perplexity") and baseline_pt.get("perplexity"):
            deltas["ppl_delta"] = round(pt["perplexity"] - baseline_pt["perplexity"], 2)
        
        # Top-1 delta (正数 = 改善)
        if pt.get("mlm_top1_acc") and baseline_pt.get("mlm_top1_acc"):
            deltas["top1_delta"] = round(pt["mlm_top1_acc"] - baseline_pt["mlm_top1_acc"], 4)
        
        # Strand Acc delta
        if pt.get("strand_acc") and baseline_pt.get("strand_acc"):
            deltas["strand_delta"] = round(pt["strand_acc"] - baseline_pt["strand_acc"], 4)
        
        # ARI delta
        if cl.get("ARI") and baseline_cl.get("ARI"):
            deltas["ari_delta"] = round(cl["ARI"] - baseline_cl["ARI"], 4)
        
        # NMI delta
        if cl.get("NMI") and baseline_cl.get("NMI"):
            deltas["nmi_delta"] = round(cl["NMI"] - baseline_cl["NMI"], 4)
        
        result["deltas_vs_baseline"] = deltas


def get_args():
    parser = argparse.ArgumentParser(description="SAGE v3 Ablation Summary")
    parser.add_argument("--results_dir", type=str,
                        default="/opt/ai4g_chriszyyang/buddy2/SAGE/results/ablation_v3")
    parser.add_argument("--ablation_dir", type=str,
                        default="/opt/ai4g_chriszyyang/buddy2/SAGE/code_transformer_v3/ablation_checkpoints")
    parser.add_argument("--output_path", type=str,
                        default="/opt/ai4g_chriszyyang/buddy2/SAGE/results/ablation_v3/ablation_summary_v3.json")
    return parser.parse_args()


def main():
    args = get_args()
    
    logging.info("Collecting experiment results...")
    all_results = collect_experiment_results(args.results_dir, args.ablation_dir)
    
    if not all_results:
        logging.warning("No experiment results found!")
        logging.info(f"  Searched: {args.results_dir}")
        logging.info(f"  Searched: {args.ablation_dir}")
        return
    
    logging.info(f"Found {len(all_results)} experiments")
    
    # 计算相对增量
    compute_deltas(all_results)
    
    # 打印表格
    print_summary_table(all_results)
    
    # 保存 JSON
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"Summary saved to {args.output_path}")
    
    # 保存 CSV
    save_csv(all_results, args.output_path)
    
    # 打印 delta 分析
    baseline_name = None
    for name in all_results:
        if "full_model" in name:
            baseline_name = name
            break
    
    if baseline_name:
        print(f"\n--- Delta Analysis (vs {baseline_name}) ---")
        print(f"{'Experiment':<25} {'ΔPPL':>8} {'ΔTop1':>8} {'ΔStrand':>8} {'ΔARI':>8} {'ΔNMI':>8}")
        print("-" * 70)
        for exp_name, result in all_results.items():
            if exp_name == baseline_name:
                continue
            d = result.get("deltas_vs_baseline", {})
            ppl_d = d.get("ppl_delta", "—")
            top1_d = d.get("top1_delta", "—")
            strand_d = d.get("strand_delta", "—")
            ari_d = d.get("ari_delta", "—")
            nmi_d = d.get("nmi_delta", "—")
            
            fmt = lambda v: f"{v:+.4f}" if isinstance(v, (int, float)) else str(v)
            fmt_ppl = lambda v: f"{v:+.2f}" if isinstance(v, (int, float)) else str(v)
            
            print(f"{exp_name:<25} {fmt_ppl(ppl_d):>8} {fmt(top1_d):>8} {fmt(strand_d):>8} {fmt(ari_d):>8} {fmt(nmi_d):>8}")
        print()


if __name__ == "__main__":
    main()
