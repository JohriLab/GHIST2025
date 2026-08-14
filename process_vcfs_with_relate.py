#!/usr/bin/env python3

import argparse
import os
import subprocess
from pathlib import Path
import sys
import logging
import gzip
import pandas as pd

def setup_logging(verbose):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def extract_vcf_params(vcf_path):
    """Extract parameters from VCF header"""
    params = {}
    # Check if file is gzipped
    opener = gzip.open if vcf_path.endswith('.gz') else open
    
    with opener(vcf_path, 'rt') as f:  # 'rt' mode for text reading from gzip
        for line in f:
            if not line.startswith('##'):
                break
            if line.startswith('##PARAM_mut_rate='):
                params['mut_rate'] = float(line.strip().split('=')[1])
            elif line.startswith('##PARAM_rec_rate='):
                params['rec_rate'] = float(line.strip().split('=')[1])
            elif line.startswith('##REGION_START='):
                params['start'] = int(line.strip().split('=')[1])
            elif line.startswith('##REGION_END='):
                params['end'] = int(line.strip().split('=')[1])
    return params

def create_map_file(output_prefix, rec_rate, start_pos, end_pos, rec_map_file=None, window_size=10000):
    """Create a genetic map file for Relate
    Args:
        rec_rate: Recombination rate per base pair per generation (used if rec_map_file is None)
        start_pos: Start position in bp
        end_pos: End position in bp
        rec_map_file: Optional path to recombination rate map file with columns: pos, rate
        window_size: Window size for discretizing recombination rates (default: 10kb)
    """
    map_file = f"{output_prefix}.map"
    
    if rec_map_file and os.path.exists(rec_map_file):
        # Use variable recombination rates from file
        logging.info(f"Using variable recombination rates from {rec_map_file}")
        try:
            rec_df = pd.read_csv(rec_map_file, sep='\t')
            
            # Ensure we have the required columns
            if 'pos' not in rec_df.columns or 'rate' not in rec_df.columns:
                raise ValueError("Recombination map file must have 'pos' and 'rate' columns")
            
            # Filter to region of interest
            region_rec = rec_df[(rec_df['pos'] >= start_pos) & (rec_df['pos'] <= end_pos)].copy()
            
            if len(region_rec) == 0:
                logging.warning(f"No recombination data in region {start_pos}-{end_pos}, using constant rate")
                return create_map_file_constant(output_prefix, rec_rate, start_pos, end_pos)
            
            # Create genetic map with variable rates
            with open(map_file, 'w') as f:
                f.write("pos COMBINED_rate Genetic_Map\n")
                
                # Sort by position to ensure proper ordering
                region_rec = region_rec.sort_values('pos').reset_index(drop=True)
                
                # Calculate genetic positions according to Relate's formula
                # r[i] = (rdist[i+1] - rdist[i])/(p[i+1] - p[i]) * 1e6
                # Rearranging: rdist[i+1] = rdist[i] + r[i] * (p[i+1] - p[i]) / 1e6
                
                genetic_positions = [0.0]  # Start at 0 cM
                positions = [start_pos]
                rates_cm_mb = []
                
                # Calculate rates and genetic positions
                for i, row in region_rec.iterrows():
                    pos = int(row['pos'])
                    rate_per_bp = float(row['rate'])
                    rate_cm_mb = rate_per_bp * 100 * 1e6  # Convert to cM/Mb
                    
                    if i == 0:
                        # First position: use the rate, genetic position is 0
                        rates_cm_mb.append(rate_cm_mb)
                    else:
                        # Calculate genetic position using the rate from previous position
                        prev_pos = positions[-1]
                        prev_genetic = genetic_positions[-1]
                        physical_dist_mb = (pos - prev_pos) / 1e6
                        genetic_dist = physical_dist_mb * (rates_cm_mb[-1] / 100)  # Convert cM/Mb to cM
                        genetic_pos = prev_genetic + genetic_dist
                        
                        genetic_positions.append(genetic_pos)
                        positions.append(pos)
                        rates_cm_mb.append(rate_cm_mb)
                
                # Add end position if not already included
                if positions[-1] < end_pos:
                    # Use the last known rate for the final segment
                    prev_pos = positions[-1]
                    prev_genetic = genetic_positions[-1]
                    physical_dist_mb = (end_pos - prev_pos) / 1e6
                    genetic_dist = physical_dist_mb * (rates_cm_mb[-1] / 100)
                    genetic_pos = prev_genetic + genetic_dist
                    
                    positions.append(end_pos)
                    genetic_positions.append(genetic_pos)
                    rates_cm_mb.append(rates_cm_mb[-1])  # Use same rate as last position
                
                # Write the map file in space-delimited format
                for i in range(len(positions)):
                    f.write(f"{positions[i]} {rates_cm_mb[i]:.4f} {genetic_positions[i]:.4f}\n")
                    
        except Exception as e:
            logging.warning(f"Error reading recombination map file: {e}. Using constant rate.")
            return create_map_file_constant(output_prefix, rec_rate, start_pos, end_pos)
    else:
        # Use constant recombination rate
        return create_map_file_constant(output_prefix, rec_rate, start_pos, end_pos)
    
    return map_file

def create_map_file_constant(output_prefix, rec_rate, start_pos, end_pos):
    """Create a genetic map file with constant recombination rate (original implementation)"""
    map_file = f"{output_prefix}.map"
    
    # Convert rec_rate to cM/Mb
    rate_cm_mb = rec_rate * 100 * 1e6  # Convert from per bp to cM/Mb
    
    # Calculate genetic positions according to Relate's formula
    # For constant rate: genetic_distance = physical_distance * rate_cm_mb / 1e6
    physical_dist_mb = (end_pos - start_pos) / 1e6
    genetic_dist_cm = physical_dist_mb * (rate_cm_mb / 100)  # Convert cM/Mb to cM
    
    with open(map_file, 'w') as f:
        f.write("pos COMBINED_rate Genetic_Map\n")  # Match format from docs
        # Start position
        f.write(f"{start_pos} {rate_cm_mb:.4f} 0.0000\n")
        # End position
        f.write(f"{end_pos} {rate_cm_mb:.4f} {genetic_dist_cm:.4f}\n")
    
    return map_file

def create_poplabels_file(vcf_path, output_prefix, pop_prefixes=None):
    """Create a poplabels file for Relate
    Args:
        vcf_path: Path to VCF file
        output_prefix: Output prefix for poplabels file
        pop_prefixes: Optional list of population prefixes. If None, all samples are assigned to POP1
    """
    poplabels_file = f"{output_prefix}.poplabels"
    
    # Check if file is gzipped
    opener = gzip.open if vcf_path.endswith('.gz') else open
    
    # Get sample names from VCF
    with opener(vcf_path, 'rt') as f:  # 'rt' mode for text reading from gzip
        for line in f:
            if line.startswith('#CHROM'):
                samples = line.strip().split('\t')[9:]  # Get sample names after FORMAT
                break
    
    # Write poplabels file
    with open(poplabels_file, 'w') as f:
        # Write header
        f.write("sample population group sex\n")
        
        # If no pop_prefixes provided, treat all samples as one population (original behavior)
        if not pop_prefixes:
            for sample in samples:
                f.write(f"{sample} POP1 POP1 NA\n")
            return poplabels_file
            
        # Handle multiple populations based on prefixes
        for sample in samples:
            sample_matched = False
            for prefix in pop_prefixes:
                if sample.startswith(f"{prefix}_"):
                    f.write(f"{sample} {prefix} {prefix} NA\n")
                    sample_matched = True
                    break
            if not sample_matched:
                logging.warning(f"Sample {sample} doesn't match any population prefix, assigning to {pop_prefixes[0]}")
                f.write(f"{sample} {pop_prefixes[0]} {pop_prefixes[0]} NA\n")
    
    return poplabels_file

def convert_vcf_to_haps(vcf_path, output_prefix, relate_path):
    """Convert VCF to haps format using RelateFileFormats"""
    try:
        # RelateFileFormats expects the input without .vcf extension
        # For gzipped files, we need to remove both .vcf.gz
        vcf_input = str(Path(vcf_path))
        if vcf_input.endswith('.vcf.gz'):
            vcf_input = vcf_input[:-7]  # Remove .vcf.gz
        elif vcf_input.endswith('.vcf'):
            vcf_input = vcf_input[:-4]  # Remove .vcf
        
        # Run RelateFileFormats
        logging.info(f"Converting {vcf_path} to haps/sample format")
        logging.info(f"Using input prefix: {vcf_input}")
        cmd = [
            str(Path(relate_path) / 'RelateFileFormats'),
            '--mode', 'ConvertFromVcf',
            '--haps', f"{output_prefix}.haps",
            '--sample', f"{output_prefix}.sample",
            '-i', vcf_input
        ]
        
        logging.debug(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              text=True)
        
        if result.returncode != 0:
            logging.error(f"RelateFileFormats failed for {vcf_path}")
            logging.error(f"Error: {result.stderr}")
            return None
        
        return output_prefix
        
    except Exception as e:
        logging.error(f"Error converting {vcf_path} to haps format: {str(e)}")
        return None

def run_relate(input_prefix, output_prefix, map_file, mut_rate, threads, relate_path, effective_N, years_per_gen, memory_gb=4, estimate_popsize=False, convert_tskit=False, num_iter=5):
    """Run Relate with optional population size estimation and tskit conversion
    Args:
        memory_gb: Memory limit in gigabytes (default: 4)
        estimate_popsize: Whether to run population size estimation (default: False)
        convert_tskit: Whether to convert output to tskit format (default: False)
    """
    try:
        # Get current directory to restore later
        original_dir = os.getcwd()
        output_dir = os.path.dirname(output_prefix)
        
        # Change to output directory
        os.chdir(output_dir)
        
        # Get just the filename parts
        output_name = os.path.basename(output_prefix)
        input_name = os.path.basename(input_prefix)
        map_name = os.path.basename(map_file)
        
        # Clean up any existing output files and directories
        logging.info("Cleaning up any existing output...")
        output_files = [
            f"{output_name}.mut",
            f"{output_name}.anc",
            f"{output_name}_popsize"
        ]
        for f in output_files:
            if os.path.exists(f):
                os.remove(f)
                
        if os.path.exists(output_name):
            import shutil
            try:
                shutil.rmtree(output_name)
            except Exception as e:
                logging.warning(f"Could not fully remove directory {output_name}: {e}")
        
        # First run basic Relate
        cmd_relate = [
            str(Path(relate_path) / 'Relate'),
            '--mode', 'All',
            '--haps', f"{input_name}.haps",
            '--sample', f"{input_name}.sample",
            '--map', map_name,
            '--output', output_name,
            '--mutation_rate', str(mut_rate),
            '--effectiveN', str(effective_N),
            '--seed', '1',
            '--memory', str(memory_gb)
        ]
        
        logging.info("Running initial Relate analysis...")
        logging.info(f"Working directory: {output_dir}")
        logging.info(f"Command: {' '.join(cmd_relate)}")
        
        # Run Relate and capture its output
        process = subprocess.run(cmd_relate, 
                               check=False,
                               capture_output=True,
                               text=True)
        
        # Log the full output for debugging
        if process.stdout:
            logging.info("Relate stdout:")
            for line in process.stdout.split('\n'):
                if line.strip():
                    logging.info(line)
        
        if process.stderr:
            logging.info("Relate stderr:")
            for line in process.stderr.split('\n'):
                if line.strip():
                    logging.info(line)
        
        # Check if required output files exist
        required_files = [f"{output_name}.mut", f"{output_name}.anc"]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            # List all files in output directory for debugging
            logging.info("Files in output directory:")
            for f in os.listdir('.'):
                logging.info(f"  {f}")
            raise FileNotFoundError(f"Required Relate output files missing: {missing_files}")
        
        # Run population size estimation if requested
        if estimate_popsize:
            # Get path to EstimatePopulationSize script relative to relate_path
            relate_base = Path(relate_path).parent  # Go up one level from bin/
            estimate_script = relate_base / 'scripts/EstimatePopulationSize/EstimatePopulationSize.sh'
            
            if not estimate_script.exists():
                raise FileNotFoundError(f"EstimatePopulationSize.sh not found at {estimate_script}")
                
            # Run EstimatePopulationSize script
            cmd_estimate = [
                str(estimate_script),
                '-i', output_name,
                '-m', str(mut_rate),
                '--poplabels', f"{output_name}.poplabels",
                '-o', f"{output_name}_popsize",
                '--num_iter', str(num_iter),
                '--years_per_gen', str(years_per_gen)
            ]
            
            logging.info(f"Using estimation script at: {estimate_script}")
            logging.info("Estimating population size...")
            subprocess.run(cmd_estimate, check=True)
        else:
            logging.info("Skipping population size estimation")
        
        # Convert to tskit format if requested
        if convert_tskit:
            logging.info("Converting output to tskit format...")
            cmd_convert = [
                str(Path(relate_path) / 'RelateFileFormats'),
                '--mode', 'ConvertToTreeSequence',
                '-i', output_name,
                '-o', f"{output_name}_tskit"
            ]
            
            logging.info(f"Running command: {' '.join(cmd_convert)}")
            subprocess.run(cmd_convert, check=True)
            logging.info(f"Conversion complete. Output saved to {output_name}_tskit.trees")
        
        # Change back to original directory
        os.chdir(original_dir)
        
    except subprocess.CalledProcessError as e:
        # Make sure we change back even if there's an error
        if 'original_dir' in locals():
            os.chdir(original_dir)
        logging.error(f"Relate/EstimatePopulationSize failed: {e}")
        if e.stdout:
            logging.error(f"stdout: {e.stdout}")
        if e.stderr:
            logging.error(f"stderr: {e.stderr}")
        raise
    except Exception as e:
        if 'original_dir' in locals():
            os.chdir(original_dir)
        logging.error(f"Error during Relate processing: {e}")
        raise

def process_vcf(vcf_path, output_dir, relate_path, threads, mut_rate=None, rec_rate=None, 
                start=None, end=None, effective_N=None, years_per_gen=None, pop_prefixes=None, 
                memory_gb=4, estimate_popsize=False, convert_tskit=False, rec_map_file=None, num_iter=5):
    """Process a single VCF file
    Args:
        estimate_popsize: Whether to run population size estimation (default: False)
        convert_tskit: Whether to convert output to tskit format (default: False)
    """
    if effective_N is None:
        raise ValueError("effective_N must be specified")
    if estimate_popsize and years_per_gen is None:
        raise ValueError("years_per_gen must be specified when using --estimate-popsize")
        
    vcf_name = Path(vcf_path).stem
    if vcf_name.endswith('.vcf'):  # Handle .vcf.gz files
        vcf_name = vcf_name[:-4]
    output_prefix = Path(output_dir) / vcf_name
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract parameters from VCF if not provided via command line
    params = {}
    if any(x is None for x in [mut_rate, rec_rate, start, end]):
        vcf_params = extract_vcf_params(vcf_path)
        params['mut_rate'] = mut_rate if mut_rate is not None else vcf_params.get('mut_rate')
        params['rec_rate'] = rec_rate if rec_rate is not None else vcf_params.get('rec_rate')
        params['start'] = start if start is not None else vcf_params.get('start', 1)
        params['end'] = end if end is not None else vcf_params.get('end', 100000000)
    else:
        params = {
            'mut_rate': mut_rate,
            'rec_rate': rec_rate,
            'start': start,
            'end': end
        }
    
    # Validate required parameters
    # If using recombination map file, rec_rate is not required
    if rec_map_file and os.path.exists(rec_map_file):
        # Remove rec_rate from validation when using recombination map
        params_to_check = {k: v for k, v in params.items() if k != 'rec_rate'}
        missing_params = [k for k, v in params_to_check.items() if v is None]
    else:
        missing_params = [k for k, v in params.items() if v is None]
    
    if missing_params:
        raise ValueError(f"Missing required parameters: {', '.join(missing_params)}. "
                        "Provide via command line or ensure they are in VCF header.")
    
    # Create map file
    # Use a default rec_rate if not provided and using recombination map
    rec_rate_for_map = params.get('rec_rate', 1e-8)  # Default fallback rate
    map_file = create_map_file(
        output_prefix,
        rec_rate_for_map,
        params['start'],
        params['end'],
        rec_map_file=rec_map_file
    )
    
    # Create poplabels file
    poplabels_file = create_poplabels_file(
        vcf_path, 
        output_prefix,
        pop_prefixes=pop_prefixes
    )
    
    # Convert VCF to haps format using RelateFileFormats
    haps_prefix = convert_vcf_to_haps(vcf_path, output_prefix, relate_path)
    if haps_prefix is None:
        return
    
    # Run Relate and estimate population size
    run_relate(
        output_prefix,
        output_prefix,
        map_file,
        params['mut_rate'],
        threads,
        relate_path,
        effective_N,
        years_per_gen,
        memory_gb=memory_gb,
        estimate_popsize=estimate_popsize,
        convert_tskit=convert_tskit,
        num_iter=num_iter
    )

def main():
    parser = argparse.ArgumentParser(
        description='Process VCFs with Relate to generate ARGs'
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--vcf-dir',
        help='Directory containing VCF files'
    )
    input_group.add_argument(
        '--vcf-file',
        help='Single VCF file to process'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for ARG files'
    )
    parser.add_argument(
        '--relate-path',
        required=True,
        help='Path to directory containing Relate executables'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=1,
        help='Number of threads to use (currently not used)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--mut-rate',
        type=float,
        help='Mutation rate (overrides VCF header)'
    )
    parser.add_argument(
        '--rec-rate',
        type=float,
        help='Recombination rate in cM/Mb (overrides VCF header)'
    )
    parser.add_argument(
        '--start',
        type=int,
        help='Start position (overrides VCF header)'
    )
    parser.add_argument(
        '--end',
        type=int,
        help='End position (overrides VCF header)'
    )
    parser.add_argument(
        '--effective-N',
        type=int,
        required=True,
        help='Initial effective population size (REQUIRED)'
    )
    parser.add_argument(
        '--years-per-gen',
        type=float,
        help='Number of years per generation (required if --estimate-popsize is used)'
    )
    parser.add_argument(
        '--pop-prefixes',
        nargs='+',
        help='List of population prefixes for sample names (e.g., --pop-prefixes mainland island). If not provided, all samples are treated as one population.'
    )
    parser.add_argument(
        '--memory',
        type=int,
        default=4,
        help='Memory limit in gigabytes (default: 4)'
    )
    parser.add_argument(
        '--estimate-popsize',
        action='store_true',
        default=False,
        help='Enable population size estimation (default: estimation is disabled)'
    )
    parser.add_argument(
        '--convert-tskit',
        action='store_true',
        default=False,
        help='Convert output to tskit tree sequence format (default: disabled)'
    )
    parser.add_argument(
        '--rec-map-file',
        help='Path to recombination rate map file. If provided, overrides --rec-rate.'
    )
    parser.add_argument(
        '--num-iter',
        type=int,
        default=5,
        help='Number of iterations for population size estimation (default: 5)'
    )

    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Determine whether to run population size estimation and tskit conversion (both disabled by default)
    estimate_popsize = args.estimate_popsize
    convert_tskit = args.convert_tskit
    
    try:
        if args.vcf_dir:
            # Process all VCF files in the directory
            vcf_dir = Path(args.vcf_dir)
            vcf_files = list(vcf_dir.glob('*.vcf*'))  # Modified to catch both .vcf and .vcf.gz
            if not vcf_files:
                raise ValueError(f"No VCF files found in directory: {args.vcf_dir}")
            for vcf_file in vcf_files:
                logging.info(f"Processing {vcf_file}")
                process_vcf(
                    str(vcf_file), 
                    args.output_dir, 
                    args.relate_path, 
                    args.threads,
                    mut_rate=args.mut_rate,
                    rec_rate=args.rec_rate,
                    start=args.start,
                    end=args.end,
                    effective_N=args.effective_N,
                    years_per_gen=args.years_per_gen,
                    pop_prefixes=args.pop_prefixes,
                    memory_gb=args.memory,
                    estimate_popsize=estimate_popsize,
                    convert_tskit=convert_tskit,
                    rec_map_file=args.rec_map_file,
                    num_iter=args.num_iter
                )
        else:
            # Process single VCF file
            vcf_file = Path(args.vcf_file)
            if not vcf_file.exists():
                raise ValueError(f"VCF file not found: {args.vcf_file}")
            logging.info(f"Processing {vcf_file}")
            process_vcf(
                str(vcf_file), 
                args.output_dir, 
                args.relate_path, 
                args.threads,
                mut_rate=args.mut_rate,
                rec_rate=args.rec_rate,
                start=args.start,
                end=args.end,
                effective_N=args.effective_N,
                years_per_gen=args.years_per_gen,
                pop_prefixes=args.pop_prefixes,
                memory_gb=args.memory,
                estimate_popsize=estimate_popsize,
                convert_tskit=convert_tskit,
                rec_map_file=args.rec_map_file,
                num_iter=args.num_iter
            )
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 