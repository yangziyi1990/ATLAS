"""
SAGE v3: 从 CARD (RGI) 注释结果中提取基因突变状态
===================================================
读取: dataset/annotations/card/*.txt (RGI 输出)
输出: dataset/annotations/mutations/*.tsv

突变分类:
  0 = PAD (填充)
  1 = WildType (未被 CARD 识别 或 SNPs 为 n/a)
  2 = Single-Point Mutation (仅 1 个 SNP)
  3 = Multi-Point Mutation (>=2 个 SNPs)

CARD/RGI 输出格式的关键列:
  - col 0: ORF_ID (蛋白序列 ID, 对应 .faa 的 header)
  - col 12: SNPs_in_Best_Hit_ARO (如 "D476N", "D350N, S357N", "n/a")
  - col 13: Other_SNPs
"""

import os
import re
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_snp_field(snp_str):
    """
    解析 CARD 的 SNP 字段, 返回 SNP 数量.
    
    Examples:
        "n/a" → 0
        "D476N" → 1
        "D350N, S357N" → 2
        "" → 0
    """
    if not snp_str or snp_str.strip().lower() in ('n/a', '', 'na', '-'):
        return 0
    # 以逗号分隔的 SNP 列表
    snps = [s.strip() for s in snp_str.split(',') if s.strip() and s.strip().lower() != 'n/a']
    return len(snps)


def extract_mutations_from_card(card_dir, output_dir):
    """
    从 CARD/RGI 输出中提取每个基因的突变状态.
    
    Args:
        card_dir: CARD 注释结果目录 (含 *.txt)
        output_dir: 输出 mutations TSV 目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    card_files = glob.glob(os.path.join(card_dir, "*.txt"))
    card_files = [f for f in card_files if '.ipynb_checkpoints' not in f]
    
    if not card_files:
        logging.warning(f"No CARD result files found in {card_dir}")
        return
    
    logging.info(f"Found {len(card_files)} CARD result files")
    
    total_genes = 0
    total_mutated = 0
    total_multi = 0
    
    for card_file in card_files:
        genome_id = os.path.splitext(os.path.basename(card_file))[0]
        gene_mutations = {}
        
        try:
            with open(card_file, 'r') as f:
                header = f.readline().strip().split('\t')
                
                # 查找关键列的索引
                orf_idx = None
                snp_best_idx = None
                snp_other_idx = None
                
                for i, col in enumerate(header):
                    col_lower = col.strip().lower()
                    if col_lower == 'orf_id':
                        orf_idx = i
                    elif col_lower == 'snps_in_best_hit_aro':
                        snp_best_idx = i
                    elif col_lower == 'other_snps':
                        snp_other_idx = i
                
                if orf_idx is None:
                    logging.warning(f"No ORF_ID column in {card_file}, skipping")
                    continue
                
                for line in f:
                    parts = line.strip('\n').split('\t')
                    if len(parts) <= orf_idx:
                        continue
                    
                    orf_id_raw = parts[orf_idx].strip()
                    if not orf_id_raw:
                        continue
                    
                    # ORF_ID 可能包含描述, 取第一个空格前的 ID
                    orf_id = orf_id_raw.split()[0]
                    
                    # 计算 SNP 数量 (取 Best Hit 和 Other 的合并)
                    n_snps = 0
                    if snp_best_idx is not None and snp_best_idx < len(parts):
                        n_snps += parse_snp_field(parts[snp_best_idx])
                    if snp_other_idx is not None and snp_other_idx < len(parts):
                        n_snps += parse_snp_field(parts[snp_other_idx])
                    
                    # 分类: 0 SNPs → CARD 识别但无突变 (仍标为 wildtype=1)
                    #       1 SNP → single_mut=2
                    #       >=2 SNPs → multi_mut=3
                    if n_snps == 0:
                        mut_type = 1  # wildtype (被 CARD 识别但无 SNP)
                    elif n_snps == 1:
                        mut_type = 2  # single mutation
                    else:
                        mut_type = 3  # multi mutation
                    
                    gene_mutations[orf_id] = mut_type
                    total_genes += 1
                    if mut_type == 2:
                        total_mutated += 1
                    elif mut_type == 3:
                        total_multi += 1
                        
        except Exception as e:
            logging.error(f"Error parsing {card_file}: {e}")
            continue
        
        # 写出 TSV
        if gene_mutations:
            out_path = os.path.join(output_dir, f"{genome_id}.tsv")
            with open(out_path, 'w') as f:
                f.write("# gene_id\tmutation_type\n")
                for gene_id, mut_type in sorted(gene_mutations.items()):
                    f.write(f"{gene_id}\t{mut_type}\n")
    
    logging.info(f"Mutation extraction complete:")
    logging.info(f"  Total CARD-identified genes: {total_genes}")
    logging.info(f"  Single-point mutations: {total_mutated}")
    logging.info(f"  Multi-point mutations: {total_multi}")
    logging.info(f"  Wildtype (CARD hit, no SNP): {total_genes - total_mutated - total_multi}")
    logging.info(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract mutation annotations from CARD/RGI results")
    parser.add_argument("--card_dir", type=str,
                        default="/opt/ai4g_chriszyyang/buddy2/SAGE/dataset/annotations/card",
                        help="Directory containing CARD/RGI *.txt result files")
    parser.add_argument("--output_dir", type=str,
                        default="/opt/ai4g_chriszyyang/buddy2/SAGE/dataset/annotations/mutations",
                        help="Output directory for mutation TSV files")
    
    args = parser.parse_args()
    extract_mutations_from_card(args.card_dir, args.output_dir)
