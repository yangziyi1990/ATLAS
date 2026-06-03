"""
SAGE v3: 从 GFF product 字段推断近似 COG 功能分类
===================================================
GFF 的 CDS 行中 product= 字段包含蛋白功能描述 (NCBI 注释),
可以通过关键词规则映射到 COG 功能大类, 无需运行 eggNOG-mapper.

速度: 秒级 (vs eggNOG-mapper 小时级)
精度: 粗粒度 (单字母 COG), 约 60-70% 准确率, 对预训练辅助任务足够

COG 功能大类 (26 类):
  J - Translation, ribosomal structure and biogenesis
  A - RNA processing and modification  
  K - Transcription
  L - Replication, recombination and repair
  B - Chromatin structure and dynamics
  D - Cell cycle control, cell division, chromosome partitioning
  Y - Nuclear structure
  V - Defense mechanisms
  T - Signal transduction mechanisms
  M - Cell wall/membrane/envelope biogenesis
  N - Cell motility
  Z - Cytoskeleton
  W - Extracellular structures
  U - Intracellular trafficking, secretion, and vesicular transport
  O - Posttranslational modification, protein turnover, chaperones
  C - Energy production and conversion
  G - Carbohydrate transport and metabolism
  E - Amino acid transport and metabolism
  F - Nucleotide transport and metabolism
  H - Coenzyme transport and metabolism
  I - Lipid transport and metabolism
  P - Inorganic ion transport and metabolism
  Q - Secondary metabolites biosynthesis, transport and catabolism
  R - General function prediction only
  S - Function unknown
  - - Not assigned
"""

import os
import re
import glob
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# COG 功能关键词映射规则
# 优先级: 先匹配更具体的模式, 再匹配宽泛的模式
# ==============================================================================
COG_KEYWORD_RULES = [
    # V - Defense mechanisms (耐药性研究最关键的类别!)
    (r'beta-lactamase|carbapenemase|metallo-beta-lactamase|MBL fold metallo-hydrolase', 'V'),
    (r'efflux|multidrug|drug resistance|antimicrobial', 'V'),
    (r'restriction endonuclease|restriction-modification|methylase.*restriction', 'V'),
    (r'CRISPR|Cas[0-9]|toxin-antitoxin|abortive infection', 'V'),
    (r'chloramphenicol acetyltransferase|aminoglycoside.*transferase', 'V'),
    (r'tetracycline resistance|macrolide.*resistance|vancomycin resistance', 'V'),
    
    # J - Translation
    (r'ribosom|tRNA.*ligase|tRNA.*synthetase|translation.*factor|elongation factor|initiation factor', 'J'),
    (r'30S ribosomal|50S ribosomal|16S rRNA|23S rRNA|rRNA methyltransferase', 'J'),
    
    # K - Transcription
    (r'transcriptional regulator|transcription factor|sigma factor|RNA polymerase', 'K'),
    (r'LysR family|TetR.*family|AraC family|MarR family|GntR family|IclR family', 'K'),
    (r'helix-turn-helix.*regulator|HTH.*regulator|DNA-binding.*regulator', 'K'),
    
    # L - Replication, recombination and repair
    (r'DNA polymerase|DNA helicase|DNA ligase|DNA gyrase|topoisomerase', 'L'),
    (r'recombinase|integrase|transposase|resolvase|reverse transcriptase', 'L'),
    (r'DNA repair|DNA mismatch|uvrA|uvrB|uvrC|mutS|mutL|recA|recB', 'L'),
    (r'IS[0-9]|insertion sequence|IS element', 'L'),
    
    # D - Cell cycle, cell division
    (r'cell division|FtsZ|FtsA|FtsI|FtsW|FtsQ|FtsK|FtsB|FtsL|FtsN|MinC|MinD|MinE', 'D'),
    (r'chromosome partitioning|chromosome segregation|ParA|ParB|SMC', 'D'),
    
    # T - Signal transduction
    (r'signal transduction|two-component|sensor histidine kinase|response regulator', 'T'),
    (r'diguanylate cyclase|phosphodiesterase|c-di-GMP|GGDEF|EAL domain', 'T'),
    (r'chemotaxis|CheA|CheB|CheR|CheW|CheY|CheZ', 'T'),
    
    # M - Cell wall/membrane
    (r'lipopolysaccharide|peptidoglycan|murein|MurA|MurB|MurC|MurD|MurE|MurF|MurG', 'M'),
    (r'outer membrane protein|porin|OmpA|OmpC|OmpF|OmpW', 'M'),
    (r'lipid A|LPS|O-antigen|capsule|capsular|polysaccharide biosynthesis', 'M'),
    (r'penicillin-binding protein|PBP|transpeptidase', 'M'),
    
    # N - Cell motility
    (r'flagell|pilin|pilus|fimbr|type IV pil|motility|chemotaxis protein', 'N'),
    
    # U - Secretion
    (r'type I secretion|type II secretion|type III secretion|type IV secretion|type VI secretion', 'U'),
    (r'Sec-dependent|Tat pathway|signal peptidase|preprotein translocase', 'U'),
    (r'Sec[ABDEFY]|SecG|TatA|TatB|TatC', 'U'),
    
    # O - Posttranslational modification, chaperones
    (r'chaperone|GroEL|GroES|DnaK|DnaJ|GrpE|ClpA|ClpB|ClpP|ClpX|HslU|HslV', 'O'),
    (r'protease|peptidase|proteasome|Lon protease|FtsH', 'O'),
    (r'thioredoxin|glutaredoxin|peroxidase|catalase|superoxide dismutase', 'O'),
    (r'ubiquitin|SUMO|protein kinase.*Ser|protein kinase.*Thr', 'O'),
    
    # C - Energy production
    (r'cytochrome|NADH.*dehydrogenase|NADH.*oxidoreductase|ATP synthase|electron transfer', 'C'),
    (r'succinate dehydrogenase|fumarate reductase|nitrate reductase|nitrite reductase', 'C'),
    (r'ferredoxin|flavodoxin|hydrogenase|oxidoreductase.*NAD|dehydrogenase.*NAD', 'C'),
    (r'citrate synthase|aconitase|isocitrate dehydrogenase|oxoglutarate dehydrogenase', 'C'),
    
    # G - Carbohydrate metabolism
    (r'glycosyl|glucosidase|galactosidase|mannosidase|xylosidase|amylase', 'G'),
    (r'phosphotransferase system|PTS|sugar.*transport|sugar.*permease', 'G'),
    (r'glycolysis|gluconeogenesis|pentose phosphate|fructose.*bisphosphate', 'G'),
    
    # E - Amino acid metabolism
    (r'amino acid.*transport|amino acid.*permease|amino acid.*dehydrogenase', 'E'),
    (r'aminotransferase|transaminase|deaminase|amino acid.*biosynthesis', 'E'),
    (r'glutamate|glutamine|aspartate|asparagine|lysine.*biosynthesis', 'E'),
    (r'tryptophan|tyrosine|phenylalanine|histidine.*biosynthesis', 'E'),
    
    # F - Nucleotide metabolism
    (r'nucleotide|purine|pyrimidine|nucleoside|thymidylate|dihydrofolate reductase', 'F'),
    (r'ribonucleotide reductase|nucleoside kinase|adenylate kinase', 'F'),
    
    # H - Coenzyme metabolism
    (r'coenzyme|cofactor|biotin|thiamin|riboflavin|folate|pantothenate|pyridoxal', 'H'),
    (r'NAD biosynthesis|FAD biosynthesis|CoA biosynthesis|heme biosynthesis', 'H'),
    (r'porphyrin|corrin|menaquinone|ubiquinone', 'H'),
    
    # I - Lipid metabolism
    (r'fatty acid|lipase|acyl-CoA|acyltransferase|phospholipase|lipid.*biosynthesis', 'I'),
    (r'beta-oxidation|enoyl.*reductase|3-ketoacyl|acetyl-CoA carboxylase', 'I'),
    
    # P - Inorganic ion transport
    (r'iron.*transport|ferric|ferrous|siderophore|TonB-dependent receptor', 'P'),
    (r'zinc.*transport|copper.*transport|manganese.*transport|magnesium.*transport', 'P'),
    (r'sulfate.*transport|phosphate.*transport|potassium.*transport|sodium.*transport', 'P'),
    (r'ABC transporter.*metal|cation.*transport|anion.*transport', 'P'),
    
    # Q - Secondary metabolites
    (r'polyketide|non-ribosomal peptide|NRPS|PKS|siderophore biosynthesis', 'Q'),
    (r'terpene|phenazine|phenol|aromatic.*degradation|catechol', 'Q'),
    
    # Broader transport patterns (after specific ones)
    (r'MFS transporter|ABC transporter|MATE.*transporter|RND.*transporter', 'G'),  # general transport → G
    (r'transporter|permease|symporter|antiporter', 'R'),  # generic → R
    
    # Broad enzyme patterns
    (r'methyltransferase|acetyltransferase|acyltransferase', 'R'),
    (r'hydrolase|alpha/beta.*hydrolase|esterase', 'R'),
    (r'oxidoreductase|dehydrogenase|reductase|oxygenase|monooxygenase|dioxygenase', 'C'),
    (r'kinase|phosphatase', 'T'),
    (r'synthase|synthetase|ligase', 'R'),
    
    # S - Function unknown
    (r'hypothetical protein|uncharacterized protein|DUF[0-9]+|unknown function', 'S'),
    (r'domain.*containing protein|domain of unknown function', 'S'),
]

# 编译正则
COG_RULES_COMPILED = [(re.compile(pattern, re.IGNORECASE), cog) for pattern, cog in COG_KEYWORD_RULES]


def product_to_cog(product_str):
    """将 GFF product 描述映射到 COG 单字母分类"""
    if not product_str:
        return '-'
    for regex, cog in COG_RULES_COMPILED:
        if regex.search(product_str):
            return cog
    return '-'  # 无法分类


def extract_cog_from_gff(genomes_dir, output_dir):
    """
    从 GFF 文件的 product 字段推断 COG 分类.
    
    输出格式与 eggNOG-mapper 的 .annotations 文件兼容:
    - 每行: gene_id \\t ... \\t COG_category \\t ...
    - 第 7 列 (index 6) 为 COG category
    
    这样 generate_input.py 的 load_eggnog_annotations() 可以直接读取.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    gff_files = glob.glob(os.path.join(genomes_dir, "**", "*.gff"), recursive=True)
    gff_files = [f for f in gff_files if '.ipynb_checkpoints' not in f]
    
    if not gff_files:
        logging.warning(f"No GFF files found in {genomes_dir}")
        return
    
    logging.info(f"Found {len(gff_files)} GFF files")
    
    total_genes = 0
    cog_counter = Counter()
    all_records = []
    
    for gff_file in gff_files:
        genome_id = os.path.basename(os.path.dirname(gff_file))
        
        try:
            with open(gff_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) < 9 or parts[2] != 'CDS':
                        continue
                    
                    attrs = parts[8]
                    
                    # 提取 gene ID
                    locus_match = re.search(r'locus_tag=([^;]+)', attrs)
                    protein_match = re.search(r'protein_id=([^;]+)', attrs)
                    
                    gene_id = None
                    if locus_match:
                        gene_id = locus_match.group(1)
                    
                    query_id = gene_id
                    if protein_match:
                        query_id = protein_match.group(1)
                    
                    if not query_id:
                        continue
                    
                    # 提取 product
                    product_match = re.search(r'product=([^;]+)', attrs)
                    product = product_match.group(1) if product_match else ''
                    # URL decode (GFF 中 %3B = ; 等)
                    product = product.replace('%3B', ';').replace('%2C', ',').replace('%25', '%')
                    
                    # 推断 COG
                    cog = product_to_cog(product)
                    
                    # 格式: 与 eggNOG annotations 兼容 (col 0 = query, col 6 = COG)
                    # 用 genome_id@@ 前缀, 与 generate_input.py 的解析逻辑兼容
                    all_records.append(f"{genome_id}@@{query_id}\t-\t-\t-\t-\t-\t{cog}\t-\t-")
                    
                    total_genes += 1
                    cog_counter[cog] += 1
                    
        except Exception as e:
            logging.error(f"Error parsing {gff_file}: {e}")
    
    # 写出为单个合并注释文件 (兼容 eggNOG 格式)
    output_path = os.path.join(output_dir, "all_merged.emapper.annotations")
    with open(output_path, 'w') as f:
        for rec in all_records:
            f.write(rec + '\n')
    
    logging.info(f"COG extraction complete:")
    logging.info(f"  Total CDS genes: {total_genes}")
    logging.info(f"  Output: {output_path}")
    if total_genes > 0:
        logging.info(f"  COG distribution:")
        for cog, count in sorted(cog_counter.items(), key=lambda x: -x[1])[:15]:
            pct = count / total_genes * 100
            logging.info(f"    {cog}: {count} ({pct:.1f}%)")
        
        assigned = sum(v for k, v in cog_counter.items() if k != '-')
        logging.info(f"  Assigned: {assigned}/{total_genes} ({assigned/total_genes*100:.1f}%)")
        logging.info(f"  Unassigned (-): {cog_counter.get('-', 0)}")
    else:
        logging.warning(f"  No CDS genes found in any GFF file!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract approximate COG from GFF product fields")
    parser.add_argument("--genomes_dir", type=str,
                        default="/opt/ai4g_chriszyyang/buddy2/SAGE/dataset/genomes")
    parser.add_argument("--output_dir", type=str,
                        default="/opt/ai4g_chriszyyang/buddy2/SAGE/dataset/annotations/eggnog")
    
    args = parser.parse_args()
    extract_cog_from_gff(args.genomes_dir, args.output_dir)
