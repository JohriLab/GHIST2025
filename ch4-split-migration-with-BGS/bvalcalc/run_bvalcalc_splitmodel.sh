#!/bin/bash
#SBATCH -J bvalcalc
#SBATCH -p dschridelab
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adaigle@email.unc.edu

## CALCULATE B MAP
#Bvalcalc --pop_params ./params_splitmodel.py \
#    --gamma_dfe --bedgff_path ./GHIST_2025_demography_with_background_selection_coding.bed  \
#    --genome --out Bsplitmodel_all_chrom.csv \
#    --out_binsize 1000 --rec_map ./all_chrom_comeronmaps_dm6_reformatted.txt

## FILTER VCF TO KEEP ONLY SITES WITH B >= 0.75
  
#Bvalcalc -b Bsplitmodel_all_chrom.csv --positions GHIST_2025_split_migration.testing.vcf --out_minimum 0.85 --out B.85_GHIST_2025_split.testing.txt --bcftools_format
#awk -F':' '{ print $1"\t"$2 }'   B.85_GHIST_2025_split.testing.txt   > splitsites85.txt
module load bcftools 
#bcftools view -T splitsites85.txt -Ov GHIST_2025_split_migration.testing.vcf \
#| awk 'BEGIN{OFS="\t"}
#  /^#/ {print; next}
#  ($1=="3R" && $2<=4124278) {next}      # drop 3R:1–4,124,278
#  ($1=="2R" && $2<=4062495) {next}      # drop 2R:1–4,062,495
#  ($1=="2L" && $2>23057268) {next}      # drop anything after 2L:23,057,268
#  {print}
#' > B.85_GHIST_2025_split.testing.vcf

set -euo pipefail

# --- INPUT PATH (original gzipped VCF) ---
SRC="/nas/longleaf/home/adaigle/johri/projects/ghist_2025/data/demography/GHIST_2025_split_migration.final.vcf.gz"

# --- 0) Move the VCF here ---
cp "$SRC" .
VCF_GZ=$(basename "$SRC")                 # GHIST_2025_split_migration.final.vcf.gz
BASE=${VCF_GZ%.vcf.gz}                    # GHIST_2025_split_migration.final

# Optional but recommended: index the gz VCF for fast -T lookup
bcftools index -t "$VCF_GZ"

# --- 1) Make an uncompressed copy for Bvalcalc ---
VCF_DECOMP="${BASE}.decomp.vcf"
gunzip -c "$VCF_GZ" > "$VCF_DECOMP"

# --- 2) Run Bvalcalc on the decompressed VCF ---
# (outputs renamed to use 'final' so old files aren't overwritten)
BVAL_OUT="B.85_GHIST_2025_split.final.txt"
Bvalcalc -b Bsplitmodel_all_chrom.csv \
  --positions "$VCF_DECOMP" \
  --out_minimum 0.85 \
  --out "$BVAL_OUT" \
  --bcftools_format

# --- 3) Build the split-sites list for bcftools -T ---
SITES_LIST="splitsites85.final.txt"
awk -F':' '{ print $1"\t"$2 }' "$BVAL_OUT" > "$SITES_LIST"

# --- 4) Extract and filter sites; write a new FINAL VCF (plain text) ---
OUT_VCF="B.85_GHIST_2025_split.filtered.final.vcf"
bcftools view -T "$SITES_LIST" -Ov "$VCF_GZ" \
| awk 'BEGIN{OFS="\t"}
  /^#/ {print; next}
  ($1=="3R" && $2<=4124278) {next}      # drop 3R:1–4,124,278
  ($1=="2R" && $2<=4062495) {next}      # drop 2R:1–4,062,495
  ($1=="2L" && $2>23057268) {next}      # drop anything after 2L:23,057,268
  {print}
' > "$OUT_VCF"

echo "Done:"
echo "  Decompressed VCF : $VCF_DECOMP"
echo "  Bvalcalc output  : $BVAL_OUT"
echo "  Sites list       : $SITES_LIST"
echo "  Final VCF        : $OUT_VCF"
# later remove nonsyn sites. Can count with rev of this command to get callable sites
# was 4750 removed from testing and 4438 from final vcf
#grep -v 'Variant_type=Nonsynonymous' B.85_GHIST_2025_split.filtered.final.vcf > B.85_GHIST_2025_split.final.noNonsyn.vcf