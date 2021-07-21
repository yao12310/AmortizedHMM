"""
Removes rows from deepsignal extract associated with given set of k-mers.
"""

import pickle
import sys

from itertools import product

import numpy as np
import pandas as pd

from utils.bio import valid_cpg_mc_kmer

from utils.constants import METHYL_BED_COLS
from utils.constants import DEEPSIGNAL_EXTRACT_COLS

METHYL_KMERS = [''.join(kmer) for kmer in product('ACGTM', repeat=6)]
METHYL_KMERS = [kmer for kmer in METHYL_KMERS if valid_cpg_mc_kmer(kmer)]
METHYL_ONLY_KMERS = [kmer for kmer in METHYL_KMERS if 'M' in kmer]

def filter_kmers_ds_extract(df, remove, methyl_df, kmer_len=6, balance=True):
    """
    Filter out rows of a deepsignal extract dataset which include any given k-mer in remove.
    df : pd.DataFrame
        deepsignal extract (post-processed) output
    remove : []str
        list of k-mers to remove
    methyl_df : pd.DataFrame
        methylation reference (indexed on contig, strand, start_pos)
    kmer_len : int
        length of k-mers
    balance : bool
        filter out unmethylated versions of k-mers
    return pd.DataFrame
        updated dataframe
    """
    remove = set(remove)
    
    def subsequence_filter(row):
        """
        Helper method for determining whether a subsequence is valid or not.
        row : pd.Series
            extract feature vector
        return : bool
            whether sequence contains invalid k-mers
        """
        seq = row.kmer
        contig = row.contig
        strand = row.strand
        pos = row.pos
        
        subseq = seq[(len(seq) // 2 - kmer_len + 1):(len(seq) // 2 + kmer_len)]

        cg_count = subseq.count('CG')
        if cg_count > 1: # need to modify other CG sites
            for i in range(-kmer_len + 1, kmer_len - 1):
                if not i: # center
                    continue
                if subseq[i + kmer_len - 1] + subseq[i + kmer_len] == 'CG':
                    try:
                        prob = methyl_df.loc[(contig, strand, pos + i)].methyl_pct / 100
                    # most likely due to methyl ref not covering this pos, so assume CG
                    except (pd.core.indexing.IndexingError, KeyError):
                        continue
                    if np.random.rand() <= prob:
                        subseq = subseq[:(i + kmer_len - 1)] + 'M' + subseq[(i + kmer_len):]

        assert(subseq[len(subseq) // 2] == 'C')
        
        if balance or row.label: # also filter out unmethylated versions
            subseq = subseq[:(len(subseq) // 2)] + 'M' + subseq[(len(subseq) // 2 + 1):]

        for i in range(kmer_len):
            if subseq[i:(i + kmer_len)] in remove:
                return True

        return False
    
    return df[~df.apply(subsequence_filter, axis=1)]

def filter_extract_dataframe(extract_filename, methyl_filename, write_filename, 
                             kmer_set_filename, chunksize, balance, print_prog):
    """
    extract_filename : str
        base dataframe output from deepsignal extract
    methyl_filename : pd.DataFrame
        methylation reference data
    write_filename : str
        file to write modified df to
    kmer_set_filename : str
        file containing k-mers to keep (pickle)
    log_name : str
        name for logging messages
    chunksize : int
        chunk size for reading df
    balance : bool
        filter out unmethylated versions of k-mers
    print_prog : int
        print progress steps
    return : None
    """
    print("{}: Loading methylation dataframe...".format(log_name))
    sys.stdout.flush()
    
    methyl_df = pd.read_csv(methyl_filename, sep='\t', names=METHYL_BED_COLS)
    methyl_idx_df = methyl_df.set_index(['contig', 'strand', 'start_pos'])
    
    print("{}: Loading k-mer subset...".format(log_name))
    sys.stdout.flush()
    
    with open(kmer_set_filename, 'rb') as f:
        kmer_set = set(pickle.load(f))
        
    remove = [kmer for kmer in METHYL_ONLY_KMERS if kmer not in kmer_set]
    
    print("{}: Filtering dataframe...".format(log_name))
    sys.stdout.flush()
    
    prog = 0
    for feat_df in pd.read_csv(extract_filename, sep='\t', names=DEEPSIGNAL_EXTRACT_COLS, chunksize=chunksize):
        filtered_df = filter_kmers_ds_extract(
            feat_df,
            remove,
            methyl_idx_df,
            balance=balance
        )

        filtered_df.to_csv(
            write_filename,
            sep='\t',
            header=False,
            index=False,
            mode='a'
        )
        
        prog += 1
        if not prog % print_prog:
            print("\t{}: Processed {} chunks...".format(log_name, prog))
            sys.stdout.flush()
