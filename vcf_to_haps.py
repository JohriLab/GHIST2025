#!/usr/bin/env python3
"""
Convert VCF to Relate .haps/.sample format.
Handles haploid VCF format correctly.
"""

import argparse
import gzip
import sys

def convert_vcf_to_haps(vcf_path, haps_path, sample_path, chrom_index=0):
    """
    Convert VCF to Relate .haps/.sample format.
    
    Args:
        vcf_path: Input VCF file (can be .gz)
        haps_path: Output .haps file path
        sample_path: Output .sample file path
        chrom_index: Chromosome index to use (default: 0)
    """
    # Open VCF
    opener = gzip.open if vcf_path.endswith('.gz') else open
    
    samples = []
    variants = []
    
    with opener(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#CHROM'):
                # Parse sample names
                fields = line.strip().split('\t')
                samples = fields[9:]  # Samples start at column 10
                break
    
    if not samples:
        raise ValueError("No samples found in VCF")
    
    # Read variants
    with opener(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            
            chrom = fields[0]
            pos = int(fields[1])
            var_id = fields[2] if fields[2] != '.' else '.'
            ref = fields[3]
            alt = fields[4]
            qual = fields[5]
            filt = fields[6]
            info = fields[7]
            fmt = fields[8]
            genotypes = fields[9:]
            
            # Skip if not biallelic
            if ',' in alt or alt == '.':
                continue
            
            # Parse ancestral allele from INFO if available
            ancestral_allele = ref  # Default: assume ref is ancestral
            if 'AA=' in info:
                aa_part = [x for x in info.split(';') if x.startswith('AA=')]
                if aa_part:
                    aa_value = aa_part[0].split('=')[1]
                    if aa_value != '.':
                        ancestral_allele = aa_value
            
            # Determine which allele is ancestral (0 = ref, 1 = alt)
            if ancestral_allele == ref:
                # Ref is ancestral: 0 = ancestral, 1 = alternative
                allele_map = {'0': '0', '1': '1'}
            elif ancestral_allele == alt:
                # Alt is ancestral: need to flip
                allele_map = {'0': '1', '1': '0'}
            else:
                # Ancestral doesn't match either - skip this variant
                continue
            
            # For haploid: genotypes are single values (0, 1, or .)
            # For .haps format, we need pairs (0->0 0, 1->1 1, .->. .)
            # And we need to encode: 0 = ancestral, 1 = alternative
            hap_pairs = []
            for gt in genotypes:
                if gt == '.' or gt == './.':
                    hap_pairs.extend(['.', '.'])
                elif '|' in gt or '/' in gt:
                    # Already diploid format - parse and map
                    parts = gt.replace('|', '/').split('/')
                    if len(parts) == 2:
                        mapped = [allele_map.get(p, '.') for p in parts]
                        hap_pairs.extend(mapped)
                    else:
                        mapped = allele_map.get(parts[0], '.')
                        hap_pairs.extend([mapped, mapped])
                else:
                    # Haploid: map and duplicate
                    mapped = allele_map.get(gt, '.')
                    hap_pairs.extend([mapped, mapped])
            
            # Determine ancestral and alternative alleles for output
            if ancestral_allele == ref:
                anc_allele = ref
                alt_allele = alt
            else:
                anc_allele = alt
                alt_allele = ref
            
            variants.append({
                'chrom': chrom,
                'pos': pos,
                'var_id': var_id,
                'ancestral': anc_allele,
                'alternative': alt_allele,
                'hap_pairs': hap_pairs
            })
    
    # Write .haps file
    with open(haps_path, 'w') as f:
        for var in variants:
            # Format: chrom_index SNP_ID POS ancestral_allele alt_allele hap1_1 hap1_2 hap2_1 hap2_2 ...
            # Note: haplotypes are encoded as 0 (ancestral) or 1 (alternative)
            line = f"{chrom_index} {var['var_id']} {var['pos']} {var['ancestral']} {var['alternative']} {' '.join(var['hap_pairs'])}\n"
            f.write(line)
    
    # Write .sample file
    with open(sample_path, 'w') as f:
        f.write("ID_1\tID_2\tmissing\n")
        f.write("0\t0\t0\n")
        for sample in samples:
            # For haploid organisms, ID_2 = NA
            f.write(f"{sample}\tNA\t0\n")
    
    print(f"Converted {len(variants)} variants for {len(samples)} samples", file=sys.stderr)
    print(f"Output written to {haps_path} and {sample_path}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description='Convert VCF to Relate .haps/.sample format')
    parser.add_argument('--vcf', required=True, help='Input VCF file')
    parser.add_argument('--haps', required=True, help='Output .haps file')
    parser.add_argument('--sample', required=True, help='Output .sample file')
    parser.add_argument('--chr', type=int, default=0, help='Chromosome index (default: 0)')
    
    args = parser.parse_args()
    
    convert_vcf_to_haps(args.vcf, args.haps, args.sample, args.chr)

if __name__ == '__main__':
    main()

