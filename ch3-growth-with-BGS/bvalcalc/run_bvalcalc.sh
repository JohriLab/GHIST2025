#!/bin/bash
#SBATCH -J bvalcalc
#SBATCH -p general
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 4-12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=adaigle@email.unc.edu

## CALCULATE B MAP
Bvalcalc --pop_params ./params.py \
    --gamma_dfe --bedgff_path ./GHIST_2025_demography_with_background_selection_coding.bed  \
    --genome --out B_all_chrom.csv \
    --out_binsize 1000 --rec_map ./all_chrom_comeronmaps_dm6_reformatted.txt

## FILTER VCF TO KEEP ONLY SITES WITH B >= 0.75

#Bvalcalc -b B_all_chrom.csv --positions GHIST_2025_growth.testing.vcf --out_minimum 0.95 --out B.95_GHIST_2025_growth.testing.txt --bcftools_format
#awk -F':' '{ print $1"\t"$2 }'   B.95_GHIST_2025_growth.testing.txt   > sites95.txt
#module load bcftools 
#bcftools view   -T sites95.txt   -Ov GHIST_2025_growth.testing.vcf   -o B.95_GHIST_2025_growth.testing.vcf

# --- INPUT (gzipped VCF) ---
SRC="/nas/longleaf/home/adaigle/johri/projects/ghist_2025/data/demography/GHIST_2025_growth.final.vcf.gz"

# Move into the working dir
cp "$SRC" .
VCF_GZ=$(basename "$SRC")                 # GHIST_2025_growth.final.vcf.gz
BASE=${VCF_GZ%.vcf.gz}                    # GHIST_2025_growth.final

module load bcftools

# Index for fast -T lookups (recommended)
bcftools index -t "$VCF_GZ"

# Make an uncompressed copy for Bvalcalc
VCF_DECOMP="${BASE}.decomp.vcf"
gunzip -c "$VCF_GZ" > "$VCF_DECOMP"

# Run Bvalcalc (0.95 threshold), write FINAL-named output
BVAL_OUT="B.95_GHIST_2025_growth.final.txt"
Bvalcalc -b B_all_chrom.csv \
  --positions "$VCF_DECOMP" \
  --out_minimum 0.95 \
  --out "$BVAL_OUT" \
  --bcftools_format

# Build the sites list (CHR\tPOS)
SITES_LIST="sites95.final.txt"
awk -F':' '{ print $1"\t"$2 }' "$BVAL_OUT" > "$SITES_LIST"

# Extract those sites from the original gz VCF and write a new FINAL VCF
OUT_VCF="B.95_GHIST_2025_growth.final.vcf"
bcftools view -T "$SITES_LIST" -Ov "$VCF_GZ" -o "$OUT_VCF"

echo "Done:"
echo "  Decompressed VCF : $VCF_DECOMP"
echo "  Bvalcalc output  : $BVAL_OUT"
echo "  Sites list       : $SITES_LIST"
echo "  Final VCF        : $OUT_VCF"

#later ran this to remove incorrect parts of the bmap
#bcftools view -t ^chr2R:1-4062495,^chr3R:1-4124278 B.95_GHIST_2025_growth.final.vcf -Oz -o B.95_GHIST_2025_growth.final.filtered.vcf
# If you prefer gzipped final output instead:
# bcftools view -T "$SITES_LIST" -Oz "$VCF_GZ" -o "${OUT_VCF}.gz"
# tabix -p vcf "${OUT_VCF}.gz"
