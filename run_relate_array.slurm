#!/bin/bash
#SBATCH --job-name=relate_array
#SBATCH --output=logs/relate_%A_%a.out
#SBATCH --error=logs/relate_%A_%a.err
#SBATCH --array=1-20000%500
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Directory settings
VCF_DIR="/proj/johrilab/projects/DFEpos/stats_v2/batch04/vcf"
POLY_STATS_DIR="/proj/johrilab/projects/DFEpos/stats_v2/batch04/poly_based_stats"
BASE_OUTPUT_DIR="/proj/johrilab/projects/DFEpos/stats_v2/batch04/trees"
RELATE_PATH="/nas/longleaf/home/solsloat/bin/relate/bin"  # Verify this path exists

# Create base output and log directories if they don't exist
mkdir -p ${BASE_OUTPUT_DIR}
mkdir -p logs

# Get the list of VCF files and select the one for this array task
mapfile -t VCF_FILES < <(find ${VCF_DIR} -type f -name "*.vcf*" ! -name "*.csi")
VCF_FILE="${VCF_FILES[$SLURM_ARRAY_TASK_ID-1]}"

# Extract basename for input/output files
BASENAME=$(basename ${VCF_FILE} .vcf.gz)
BASENAME=${BASENAME%.vcf}  # Remove .vcf extension if it exists
POLY_STATS_FILE="${POLY_STATS_DIR}/${BASENAME}.vcf_windows.txt"

# Create VCF-specific output directory
OUTPUT_DIR="${BASE_OUTPUT_DIR}/${BASENAME}"
mkdir -p ${OUTPUT_DIR}

# Filter out multiallelic sites, duplicated positions, and create temporary filtered VCF
TEMP_PREFIX="${OUTPUT_DIR}/temp_filtered"
FILTERED_VCF="${TEMP_PREFIX}.vcf.gz"

# Remove multiallelic sites and duplicated positions, keep only biallelic SNPs
bcftools view -m2 -M2 -v snps "${VCF_FILE}" | \
    bcftools norm -d both | \
    bcftools sort -Oz -o "${FILTERED_VCF}"

# Index the filtered VCF
bcftools index "${FILTERED_VCF}"

# Check if poly stats file exists
if [ ! -f "${POLY_STATS_FILE}" ]; then
    echo "Error: Poly stats file not found: ${POLY_STATS_FILE}"
    exit 1
fi

# Calculate mean pi from poly stats file (skip header line)
MEAN_PI=$(tail -n +2 "${POLY_STATS_FILE}" | awk -F'\t' '{ sum += $2; count++ } END { if (count > 0) print sum/count }')

if [ -z "${MEAN_PI}" ]; then
    echo "Error: Could not calculate mean pi from ${POLY_STATS_FILE}"
    exit 1
fi

# Extract mutation rate from VCF header
MUT_RATE=$(zcat "${VCF_FILE}" | grep "##PARAM_mut_rate=" | cut -d'=' -f2)

if [ -z "${MUT_RATE}" ]; then
    echo "Error: Could not extract mutation rate from VCF header"
    exit 1
fi

# Apply scaling factor of 200 to mutation rate and handle scientific notation
SCALED_MUT_RATE=$(awk -v mut="$MUT_RATE" 'BEGIN {printf "%.10e\n", mut * 200}')

if [ -z "${SCALED_MUT_RATE}" ]; then
    echo "Error: Could not calculate scaled mutation rate"
    exit 1
fi

# Calculate Ne using pi/(4*mu) with scaled mutation rate
NE=$(awk -v pi="$MEAN_PI" -v mu="$SCALED_MUT_RATE" 'BEGIN {printf "%.0f\n", pi/(4*mu)}')

if [ -z "${NE}" ]; then
    echo "Error: Could not calculate Ne"
    exit 1
fi

echo "Mean pi: ${MEAN_PI}"
echo "Original mutation rate: ${MUT_RATE}"
echo "Scaled mutation rate (x200): ${SCALED_MUT_RATE}"
echo "Calculated Ne: ${NE}"
echo "Output directory: ${OUTPUT_DIR}"

# Run process_vcfs_with_relate.py with filtered VCF
python ~/DFEpos/process_vcfs_with_relate.py \
    --vcf-file "${FILTERED_VCF}" \
    --output-dir "${OUTPUT_DIR}" \
    --relate-path "${RELATE_PATH}" \
    --threads 1 \
    --effective-N "${NE}" \
    --convert-tskit \
    --verbose

# Check exit status
RELATE_STATUS=$?

# Clean up all files except .trees
if [ ${RELATE_STATUS} -eq 0 ]; then
    echo "Successfully processed ${VCF_FILE}"
    
    # First remove any write-protected files and directories recursively
    chmod -R u+w "${OUTPUT_DIR}"
    
    # Keep only .trees file and remove everything else
    find "${OUTPUT_DIR}" -type f ! -name "*.trees" -delete
    
    # Remove all empty directories (including nested ones)
    find "${OUTPUT_DIR}" -type d -empty -delete
    
    # Try again with -rf if some directories remain (except the output dir itself)
    find "${OUTPUT_DIR}" -mindepth 1 -type d -exec rm -rf {} +
    
    echo "Cleaned up temporary files, kept only .trees file"
else
    echo "Error processing ${VCF_FILE}"
    exit 1
fi 