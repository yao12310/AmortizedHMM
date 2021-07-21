"""
Obtain methylated k-mer frequencies for ILP objective function weighting.
"""

from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import pickle

from utils.constants import KMER_LEN
from utils.constants import METHYL_BED_COLS

from utils.misc import valid_cpg_mc_kmer

METHYL_11MERS = [''.join(kmer) for kmer in product('ACGTM', repeat=(2 * KMER_LEN - 1))]
METHYL_11MERS = [kmer for kmer in METHYL_11MERS if valid_cpg_mc_kmer(kmer)]
METHYL_ONLY_11MERS = [kmer for kmer in METHYL_11MERS if 'M' in kmer]
METHYL_CENTER_11MERS = [kmer for kmer in METHYL_ONLY_11MERS if kmer[KMER_LEN - 1] + kmer[KMER_LEN] == 'MG']

def compute_kmer_frequency(genome_ref, methyl_ref, freq_filename, print_prog, strand):
    """
    Writes frequency of methylated k-mers.
    genome_ref : str
        filename for genome reference
    methyl_ref : str
        filename for methylation reference (BED format)
    freq_filename : str
        filename to dump pickled frequency dict to
    print_prog : int
        interval between progress updates (-1 for no updates)
    strand : str
        strand identifier (+ or -)
    limit : int
        maximum number of lines of reference genome to process (-1 for no maximum)
    return : None
    """
    
    prog = 0
    
    methyl_df = pd.read_csv(
        methyl_ref,
        sep='\t',
        names=METHYL_BED_COLS,
        index_col=False
    )
    
    methyl_idx_df = methyl_df.set_index(['contig', 'strand', 'start_pos'])

    k2mer_ctr = defaultdict(int)

    with open(genome_ref, 'r') as f:
        prev = ''
        line = f.readline()
        post = f.readline()

        curr_contig = ''
        curr_pos = 0 # at start of line, ignoring prev/post buffers

        while line:
            try:
                if 'chr' not in line:
                    if 'chr' in prev:
                        subseq = line[:-1] + post[:(KMER_LEN - 1)]
                    elif 'chr' in post:
                        subseq = prev[-KMER_LEN:-1] + line[:-1]
                    else:
                        subseq = prev[-KMER_LEN:-1] + line[:-1] + post[:KMER_LEN] # need extra one here for CG checking

                    for i in range(KMER_LEN - 1, len(subseq) - KMER_LEN):
                        try:
                            if subseq[i] + subseq[i + 1] == 'CG':
                                k2mer = subseq[(i - (KMER_LEN - 1)):(i + KMER_LEN)]
                                k2mer = k2mer[:(KMER_LEN - 1)] + 'M' + k2mer[KMER_LEN:] # always replace middle C with M

                                # probabilistically modify CGs
                                cg_pos = curr_pos + i - (KMER_LEN - 1) # offset buffer
                                for j in range(-(KMER_LEN - 1), KMER_LEN):
                                    if j == 0: # middle position already changed
                                        continue
                                    if subseq[i + j] + subseq[i + j + 1] == 'CG':
                                        methyl_prob = methyl_idx_df.loc[(curr_contig, strand, cg_pos + j)].methyl_pct / 100
                                        if np.random.rand() <= methyl_prob:
                                            k2mer = k2mer[:(j + (KMER_LEN - 1))] + 'M' + k2mer[(j + KMER_LEN):]

                                k2mer_ctr[k2mer] += 1
                        except Exception: # skip CpG site if error, usually due to some issue with methylation reference
                            continue

                    curr_pos += len(line) - 1 # ignore newline char
                else:
                    curr_contig = line.split()[0][1:]
                    curr_pos = 0
            except Exception:
                print("Skipped line {} due to {}".format(prog, e))

            prev = line
            line = post
            post = f.readline()

            prog += 1
            if print_prog != -1 and not prog % print_prog:
                print("Processed {} lines of reference genome.".format(prog))
            
            if limit != -1 and prog >= limit:
                break

    total = sum(k2mer_ctr.values())
    k2mer_freqs = {kmer: k2mer_ctr[kmer] / total for kmer in k2mer_ctr}

    for kmer in METHYL_CENTER_11MERS:
        if kmer not in k2mer_freqs:
            k2mer_freqs[kmer] = 0.0

    with open(freq_filename, 'wb') as f:
        pickle.dump(k2mer_freqs, f, protocol=pickle.HIGHEST_PROTOCOL)
