"""
Modify reference genome as preprocessing step for nanopolish.
"""

import multiprocessing as mp
import pickle
import sys

from itertools import product

import pandas as pd

sys.path.append('..')

from utils.bio import valid_cpg_mc_kmer

from utils.constants.final_analysis import METHYL_BED_COLS

COVERAGE_THRESH = 5
METHYL_THRESH = 100
KMER_LEN = 6

def methylate_reference(ref_filename, methyl_filename, mod_ref_filename, kmer_set_filename, kmer_pct, print_prog, total):
    """
    Methylate reference genome.
    ref_filename : str
        location of base reference genome
    methyl_filename : str
        methylation reference data
    mod_ref_filename : str
        where to write modified ref
    kmer_set_filename : str
        where to load k-mer subset
    kmer_pct : int
        currently processed percentage of k-mers kept
    print_prog : int
        print progress steps
    total : int
        total lines in reference genome
    """
    print("{}: Loading methylation reference data...".format(kmer_pct))
    sys.stdout.flush()
    
    methyl_df = pd.read_csv(methyl_filename, sep='\t', names=METHYL_BED_COLS)
    methyl_df = methyl_df[methyl_df.strand == '+']
    
    with open(kmer_set_filename, 'rb') as f:
        kmer_set = set(pickle.load(f))
    
    prog = 0
    methyl_cnt = 0
    
    print("{}: Modifying reference genome...".format(kmer_pct))
    sys.stdout.flush()
    
    with open(ref_filename, 'r') as ref:
        with open(mod_ref_filename, 'w') as mod:
            prev_line = None
            line = ref.readline()
            next_line = ref.readline()
            
            while line:
                while line and 'chr' not in line: # guaranteed to get chr in the first line
                    try:
                        for i in range(len(line) - 1):
                            if ((i == ((len(line) - 1) - 1) and len(next_line) > 0 and line[i] + next_line[0] == 'CG') or
                                 (line[i] + line[i + 1] == 'CG')):

                                try:
                                    methyl_data = contig_methyl.loc[pos + i]
                                except KeyError:
                                    continue

                                if methyl_data.coverage >= COVERAGE_THRESH and methyl_data.methyl_pct >= METHYL_THRESH:
                                    curr_kmers = []
                                    for j in range(KMER_LEN):
                                        curr_kmer = []
                                        for k in range(KMER_LEN):
                                            if i - j + k < 0:
                                                curr_kmer.append(prev_line[i - j + k])
                                            elif i - j + k >= len(line) - 1:
                                                if 'chr' in next_line:
                                                    break
                                                curr_kmer.append(next_line[i - j + k - len(line) + 1])
                                            else:
                                                curr_kmer.append(line[i - j + k])

                                        assert(curr_kmer[j] == 'C')
                                        curr_kmer[j] = 'M'
                                        curr_kmers.append(''.join(curr_kmer))

                                    if all([kmer in kmer_set for kmer in curr_kmers]): # valid methylation
                                        line = line[:i] + 'M' + line[(i + 1):]
                                        methyl_cnt += 1
                    except KeyError:
                        print("{}: Missing methylation record near position {}, contig {}.".format(kmer_pct, pos, curr_contig))
                        sys.stdout.flush()
                    except Exception: # it's okay to not modify a line, as long as everything gets written in the right order
                        print(kmer_pct, e)
                        sys.stdout.flush()
                    
                    pos += len(line) - 1 # new line character
                    
                    mod.write(line)
                    
                    prev_line = line
                    line = next_line
                    next_line = ref.readline()
                    
                    prog += 1
                    
                    if not prog % print_prog:
                        print("{}: Processed {}/{} lines of reference genome. Methylated {} positions." \
                                  .format(kmer_pct, prog, total, methyl_cnt))
                        sys.stdout.flush()
                        mod.flush()
                
                if not line:
                    break
                
                curr_contig = line.split()[0][1:]
                pos = 0 # methylation data is 0-indexed
                contig_methyl = methyl_df[methyl_df.contig == curr_contig]
                contig_methyl = contig_methyl.set_index('start_pos')
                
                mod.write(line)
                
                prev_line = line
                line = next_line
                next_line = ref.readline()
                
if __name__ == '__main__':
    ref_filename = sys.argv[1]
    methyl_filename = sys.argv[2]
    mod_ref_filename = sys.argv[3] # should leave an open format string for pct
    kmer_set_filename = sys.argv[4] # should leave an open format string for pct
    kmer_pcts = [int(pct) for pct in sys.argv[5].split(',')]
    print_prog = int(sys.argv[6])
    total = int(sys.argv[7])
    
    args = [
        (
            ref_filename,
            methyl_filename,
            mod_ref_filename.format(pct),
            kmer_set_filename.format(pct),
            pct,
            print_prog,
            total
        ) for pct in kmer_pcts
    ]
    
    pool = mp.Pool(mp.cpu_count())
    with pool:
        pool.starmap(methylate_reference, args)
