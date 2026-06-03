"""
SAGE v3: 从 GFF product 字段推断基因耐药/突变相关性
=====================================================
无需 CARD/RGI 即可获取近似的突变标注.

策略:
  真正的 SNP 突变信息只能从 CARD/RGI 比对中获取, 但 GFF product 字段
  可以识别出 **耐药相关基因** (beta-lactamase, efflux pump 等).
  这些基因在 靶向掩码 (targeted masking) 中应该被加权关注.

突变分类 (与 CARD 提取的 mutation_vocab 兼容):
  0 = PAD
  1 = WildType (非耐药相关基因)
  2 = Resistance-Associated (耐药相关基因, 但无 SNP 信息)
  3 = High-Confidence Resistance (product 明确提到 resistance/resistant)

注意: 这是一个 **近似** 方案, 精度不如 CARD/RGI.
      当 CARD 结果可用时, 应优先使用 extract_mutations.py.

输出格式: 与 extract_mutations.py 输出兼容
  annotations/mutations/<genome_id>.tsv
  格式: gene_id \\t mutation_type
"""

import os
import re
import glob
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# 耐药基因关键词规则 (从 product 字段推断)
# 分为两级:
#   Level 3 (高置信): product 明确包含 "resistance" / "resistant"
#   Level 2 (中置信): product 匹配已知耐药机制关键词
#   Level 1 (默认):   无匹配 → wildtype
# ==============================================================================

# 高置信度: 明确提到耐药性
HIGH_CONF_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'resistance protein',
        r'resistant\b',
        r'antibiotic resistance',
        r'drug resistance',
        r'multidrug resistance',
        r'tetracycline resistance',
        r'chloramphenicol resistance',
        r'macrolide resistance',
        r'vancomycin resistance',
        r'beta-lactam resistance',
        r'aminoglycoside resistance',
        r'quinolone resistance',
        r'sulfonamide resistance',
        r'trimethoprim resistance',
        r'polymyxin resistance',
        r'colistin resistance',
        r'linezolid resistance',
        r'fosfomycin resistance',
        r'rifampin resistance',
        r'streptomycin resistance',
        r'erythromycin resistance',
        r'kanamycin resistance',
    ]
]

# 中置信度: 已知耐药相关酶/蛋白家族 (不一定都有突变, 但与耐药机制相关)
MEDIUM_CONF_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        # Beta-lactamases
        r'beta-lactamase',
        r'carbapenemase',
        r'metallo-beta-lactamase',
        r'class [A-D] beta-lactamase',
        r'extended-spectrum beta-lactamase',
        r'ESBL',
        r'OXA-\d+',
        r'KPC-\d+',
        r'NDM-\d+',
        r'VIM-\d+',
        r'IMP-\d+',
        r'CTX-M',
        r'TEM-\d+',
        r'SHV-\d+',

        # Efflux pumps
        r'efflux pump',
        r'efflux transporter',
        r'multidrug efflux',
        r'MFS.*efflux',
        r'RND.*efflux',
        r'MATE.*efflux',
        r'SMR.*efflux',
        r'AcrAB',
        r'MexAB',
        r'EmrAB',
        r'MacAB',

        # Aminoglycoside modifying enzymes
        r"aminoglycoside.*phosphotransferase",
        r"aminoglycoside.*acetyltransferase",
        r"aminoglycoside.*nucleotidyltransferase",
        r"aminoglycoside.*adenylyltransferase",
        r"AAC\(\d",
        r"APH\(\d",
        r"ANT\(\d",

        # Chloramphenicol resistance
        r'chloramphenicol acetyltransferase',
        r'chloramphenicol.*exporter',

        # Tetracycline resistance
        r'tetracycline.*efflux',
        r'tet\([A-Z]\)',
        r'ribosomal protection protein',

        # Target modification
        r'16S rRNA methyltransferase',
        r'23S rRNA methyltransferase',
        r'ribosome methyltransferase',

        # Penicillin-binding proteins (mutation targets)
        r'penicillin-binding protein',
        r'PBP\d',

        # Quinolone resistance
        r'Qnr\w+',
        r'quinolone.*efflux',
        r'DNA gyrase.*mutation',

        # Colistin/polymyxin resistance
        r'MCR-\d+',
        r'lipid A modification',
        r'phosphoethanolamine.*transferase',

        # Vancomycin resistance
        r'VanA|VanB|VanC|VanD|VanE|VanG',
        r'D-Ala-D-Lac ligase',

        # Dihydrofolate reductase (trimethoprim target)
        r'dihydrofolate reductase.*trimethoprim',
        r'dfr[A-Z]\d*',

        # Sulfonamide resistance
        r'dihydropteroate synthase',
        r'sul[12]',

        # Fosfomycin resistance
        r'fosfomycin.*thiol transferase',
        r'FosA|FosB|FosC|FosX',

        # General
        r'antibiotic inactivation',
        r'drug.*inactivation',
    ]
]


def classify_product(product_str):
    """
    将 GFF product 字段分类为突变/耐药状态.

    Returns:
        1 = WildType (无耐药关联)
        2 = Resistance-Associated (中置信度)
        3 = High-Confidence Resistance (高置信度)
    """
    if not product_str:
        return 1

    # 先检查高置信度
    for pattern in HIGH_CONF_PATTERNS:
        if pattern.search(product_str):
            return 3

    # 再检查中置信度
    for pattern in MEDIUM_CONF_PATTERNS:
        if pattern.search(product_str):
            return 2

    return 1  # wildtype


def extract_mutations_from_gff(genomes_dir, output_dir):
    """
    从 GFF product 字段推断耐药基因标注.

    输出格式: 与 extract_mutations.py (CARD-based) 完全兼容
    每个基因组一个 TSV:
      # gene_id  mutation_type
      WP_123456.1   2
    """
    os.makedirs(output_dir, exist_ok=True)

    gff_files = glob.glob(os.path.join(genomes_dir, "**", "*.gff"), recursive=True)
    gff_files = [f for f in gff_files if '.ipynb_checkpoints' not in f]

    if not gff_files:
        logging.warning(f"No GFF files found in {genomes_dir}")
        return

    logging.info(f"Found {len(gff_files)} GFF files")

    total_genes = 0
    mut_counter = Counter()
    genomes_with_resistance = 0

    for gff_file in gff_files:
        genome_id = os.path.basename(os.path.dirname(gff_file))
        gene_mutations = {}
        has_resistance = False

        try:
            with open(gff_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) < 9 or parts[2] != 'CDS':
                        continue

                    attrs = parts[8]

                    # 提取 gene ID (优先 protein_id, 其次 locus_tag)
                    protein_match = re.search(r'protein_id=([^;]+)', attrs)
                    locus_match = re.search(r'locus_tag=([^;]+)', attrs)

                    query_id = None
                    if protein_match:
                        query_id = protein_match.group(1)
                    elif locus_match:
                        query_id = locus_match.group(1)

                    if not query_id:
                        continue

                    # 提取 product
                    product_match = re.search(r'product=([^;]+)', attrs)
                    product = product_match.group(1) if product_match else ''
                    product = product.replace('%3B', ';').replace('%2C', ',').replace('%25', '%')

                    # 分类
                    mut_type = classify_product(product)
                    gene_mutations[query_id] = mut_type
                    total_genes += 1
                    mut_counter[mut_type] += 1

                    if mut_type >= 2:
                        has_resistance = True

        except Exception as e:
            logging.error(f"Error parsing {gff_file}: {e}")
            continue

        if has_resistance:
            genomes_with_resistance += 1

        # 只写出含有耐药基因的记录 (wildtype 不需要写, 加载时默认就是 1)
        resistance_genes = {k: v for k, v in gene_mutations.items() if v >= 2}
        if resistance_genes:
            out_path = os.path.join(output_dir, f"{genome_id}.tsv")
            with open(out_path, 'w') as f:
                f.write("# gene_id\tmutation_type\n")
                for gene_id, mut_type in sorted(resistance_genes.items()):
                    f.write(f"{gene_id}\t{mut_type}\n")

    logging.info(f"GFF-based mutation extraction complete:")
    logging.info(f"  Total CDS genes scanned: {total_genes}")
    logging.info(f"  WildType (no resistance signal): {mut_counter.get(1, 0)}")
    logging.info(f"  Resistance-Associated (medium conf): {mut_counter.get(2, 0)}")
    logging.info(f"  High-Confidence Resistance: {mut_counter.get(3, 0)}")
    logging.info(f"  Genomes with >=1 resistance gene: {genomes_with_resistance}/{len(gff_files)}")
    logging.info(f"  Output directory: {output_dir}")

    resistance_total = mut_counter.get(2, 0) + mut_counter.get(3, 0)
    if resistance_total == 0:
        logging.warning("  No resistance genes detected! All genes will be WildType.")
        logging.warning("  Consider running CARD/RGI (SKIP_CARD=false) for better annotation.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract approximate mutation/resistance annotations from GFF product fields")
    parser.add_argument("--genomes_dir", type=str,
                        default="/opt/ai4g_chriszyyang/buddy2/SAGE/dataset/genomes")
    parser.add_argument("--output_dir", type=str,
                        default="/opt/ai4g_chriszyyang/buddy2/SAGE/dataset/annotations/mutations")

    args = parser.parse_args()
    extract_mutations_from_gff(args.genomes_dir, args.output_dir)
