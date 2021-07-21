"""
Entry point script.
"""

import argparse

from scripts.ilp.kmer_frequency import compute_kmer_frequency
from scripts.ilp.ilp_kmer import solve_kmer_ilp
from scripts.deepsignal.label_deepsignal_extract import label_extract

def main_kmer_freq(args):
    compute_kmer_frequency(
        args.genome_ref,
        args.methyl_ref,
        args.freq_filename,
        args.print_prog,
        args.strand
    )

def main_ilp_kmer(args):
    solve_kmer_ilp(
        args.pct,
        args.obj_weights_filename,
        args.log_filename,
        args.sol_filename,
        args.kmer_set_filename,
        args.solver,
        args.threads,
        args.limit
    )

def main_label_extract(args):
    label_extract(
        args.feat_filename,
        args.methyl_ref,
        args.write_filename,
        args.min_coverage,
        args.methyl_thresh,
        args.unmethyl_thresh,
        args.chunksize
    )
    
def main():
    parser = argparse.ArgumentParser(
        prog="amortized_hmm",
        description="preprocessing and analysis tools for Amortized-HMM paper.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(title="modules", help="amortized_hmm submodules, use -h/--help for help")
    sub_kmer_freq = subparsers.add_parser("kmer_freq", description="compute frequencies of methylated k-mers for use in ILP")
    sub_ilp_kmer = subparsers.add_parser("ilp_kmer", description="solve ILP for k-mer selection")
    sub_label_extract = subparsers.add_parser("label_extract", description="add high-confidence labels to deepsignal extract outputs")
    
    sub_kmer_freq.add_argument('--genome_ref', '-g', action='store', type=str, required=True, default=None, help="genome reference filename")
    sub_kmer_freq.add_argument('--methyl_ref', '-m', action='store', type=str, required=True, default=None, help="methylation reference filename (BED)")
    sub_kmer_freq.add_argument('--freq_filename', '-f', action='store', type=str, required=True, default=None, help=".pkl file output")
    sub_kmer_freq.add_argument('--print_prog', '-p', action='store', type=int, required=False, default=-1, help='steps between logging messages')
    sub_kmer_freq.add_argument('--strand', '-s', action='store', choices=['+', '-'], required=False, default='+', help='strand identifier')
    sub_kmer_freq.add_argument('--limit', '-l', action='store', type=int, required=False, default=-1, help='max # of lines of reference genome to process')
    sub_kmer_freq.set_defaults(func=main_kmer_freq)
    
    sub_ilp_kmer.add_argument('--pct', '-p', action='store', type=int, required=True, help="pct of k-mers to retain")
    sub_ilp_kmer.add_argument('--obj_weights_filename', '-w', action='store', type=str, required=True,
                                  default=None, help="objective function weights dict pickle file")
    sub_ilp_kmer.add_argument('--log_filename', '-f', action='store', type=str, required=True, help="output file for ILP solver logs")
    sub_ilp_kmer.add_argument('--sol_filename', '-o', action='store', type=str, required=True, help="filename template for ILP solutions")
    sub_ilp_kmer.add_argument('--kmer_set_filename', '-k', action='store', type=str, required=True, help="filename for pickled k-mer set")
    sub_ilp_kmer.add_argument('--solver', '-s', action='store', type=str, required=True, help="name of ILP solver")
    sub_ilp_kmer.add_argument('--threads', '-t', action='store', type=int, required=False, default=1, help="number of threads for ILP solver")
    sub_ilp_kmer.add_argument('--limit', '-l', action='store', type=int, required=False, default=None, help="time limit for solver (seconds)")
    sub_ilp_kmer.set_defaults(func=main_ilp_kmer)
    
    sub_label_extract.add_argument('--feat_filename', '-f', action-'store', type=str, required=True, help="deepsignal extract output filename")
    sub_label_extract.add_argument('--methyl_ref', '-r', action='store', type=str, required=True, default=None, help="methylation reference filename (BED)")
    sub_label_extract.add_argument('--write_filename', '-w', action='store', type=str, required=True, help="filename to write processed output")
    sub_label_extract.add_argument('--min_coverage', '-c', action='store', type=int, required=False, default=5, help="minimum coverage for labeled sites")
    sub_label_extract.add_argument('--methyl_thresh', '-m', action='store', type=int, required=False,
                                       default=100, help="minimum methylation % for positive label")
    sub_label_extract.add_argument('--unmethyl_thresh', '-u', action='store', type=int, required=False,
                                       default=0, help="maximum methylation % for negative label")
    sub_label_extract.add_argument('--chunksize', '-s', action='store', type=int, required=False, default=None, help="chunksize for dataframe reading")
    sub_label_extract.set_defaults(func=main_label_extract)
    
    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
    
if __name__ == '__main__':
    main()
