"""
Process nanopolish methylation calls for evaluation.
"""

import multiprocessing as mp
import sys

import pandas as pd

sys.path.append("..")

from utils.constants.final_analysis import METHYL_BED_COLS
from utils.constants.final_analysis import NANOPOLISH_BINARY_SUMMARY_COLS

BUFFER = 5
SINGLETON_LEN = 11
NUM_NANOPOLISH_SUMMARY_COLS = 7
COVERAGE_THRESH = 5
METHYL_THRESH = 100
UNMETHYL_THRESH = 0

def unroll_nanopolish_methylation(df):
    """
    Unrolls methylation calling output (for groups of >1 CG sites).
    Filters columns to relevant set for binary calling summary.
    df : pd.DataFrame
        output of nanopolish call-methylation
    return : pd.DataFrame
        unrolled dataframe
    """    
    new_df = []

    for index, row in df.iterrows():
        try:
            if len(row.sequence) == SINGLETON_LEN:
                new_df.append(
                    [
                        row.chromosome,
                        row.strand,
                        row.start,
                        row.sequence,
                        row.read_name,
                        row.log_lik_ratio / row.num_motifs,
                        True
                    ]
                )
            else:
                for i in range(BUFFER, len(row.sequence) - BUFFER):
                    if row.sequence[i] + row.sequence[i + 1] != 'CG':
                        continue
                    new_df.append(
                        [
                            row.chromosome,
                            row.strand,
                            row.start + i - BUFFER,
                            row.sequence[(i - BUFFER):(i + BUFFER + 1)],
                            row.read_name,
                            row.log_lik_ratio,
                            False
                        ]
                    )
        except TypeError: # handles random nan
            continue

    new_df = pd.DataFrame(new_df, columns=NANOPOLISH_BINARY_SUMMARY_COLS[:NUM_NANOPOLISH_SUMMARY_COLS])
    return new_df

def label_summary(call_df, ref_df):
    """
    Joins unrolled methylation calling output with methylation reference data, adds labels.
    call_df : pd.DataFrame
        unrolled methylation calling dataframe
    ref_df : pd.DataFrame
        methylation reference dataframe
    return : pd.DataFrame
        labeled dataframe
    """
    sub_methyl_df = ref_df[['contig', 'strand', 'start_pos', 'coverage', 'methyl_pct']]
    sub_methyl_df['pos'] = sub_methyl_df['start_pos'] - (sub_methyl_df.strand == '-')
    sub_methyl_df = sub_methyl_df.drop('start_pos', axis=1)

    joined_df = call_df.merge(
        sub_methyl_df,
        how='left',
        on=[
            'contig',
            'strand',
            'pos'
        ]
    )
    
    label_df = joined_df.dropna()[
        (joined_df.coverage >= COVERAGE_THRESH) &
        ((joined_df.methyl_pct >= METHYL_THRESH) | (joined_df.methyl_pct <= UNMETHYL_THRESH))
    ]
    
    label_df['label'] = (label_df.methyl_pct >= METHYL_THRESH).astype(int)

    return label_df

def evaluate_nanopolish(call_filename, methyl_filename, write_filename, pct):
    """
    Adds binary labels to Nanopolish methylation calling outputs.
    call_filename : str
        filename for Nanopolish calls
    methyl_filename : str
        methylation reference
    write_filename : str
        filename for binary labels
    pct : int
        curr k-mer pct
    return : None
    """
    print("{}: Loading methylation calling output...".format(pct))
    sys.stdout.flush()
    
    call_df = pd.read_csv(call_filename, sep='\t', error_bad_lines=False) # very, very rare malformed row
    
    print("{}: Loading methylation reference data...".format(pct))
    sys.stdout.flush()
    
    methyl_df = pd.read_csv(methyl_filename, sep='\t', names=METHYL_BED_COLS)
    
    print("{}: Unrolling methylation calls...".format(pct))
    sys.stdout.flush()
    
    unrolled_df = unroll_nanopolish_methylation(call_df)
    
    print("{}: Joining labels with calls...".format(pct))
    sys.stdout.flush()
    
    label_df = label_summary(unrolled_df, methyl_df)
    
    print("{}: Saving joined summary...".format(pct))
    sys.stdout.flush()
    
    label_df.to_csv(write_filename, sep='\t', index=False)
    
if __name__ == '__main__':
    call_filename = sys.argv[1] # should leave one format string for pct
    methyl_filename = sys.argv[2]
    write_filename = sys.argv[3] # should leave one format string for pct
    pcts = [int(pct) for pct in sys.argv[4].split(',')]
    
    args = [
        (
            call_filename.format(pct),
            methyl_filename,
            write_filename.format(pct),
            pct
        )
        for pct in pcts
    ]
    
    pool = mp.Pool(mp.cpu_count())
    with pool:
        pool.starmap(evaluate_nanopolish, args)
