"""
Post-processes deepsignal extract output: add high-confidence labels, removing sites w/o high confidence in methylation status.
"""

import sys

import pandas as pd

from utils.constants import METHYL_BED_COLS
from utils.constants import DEEPSIGNAL_EXTRACT_COLS

def label_extract(feat_filename, methyl_filename, write_filename, min_coverage, methyl_thresh, unmethyl_thresh, chunksize):
    """
    feat_filename : str
        filename for output from deepsignal extract
    methyl_filename : str
        filename for methylation reference (BED)
    write_filename : str
        filename to write processed deepsignal extract output
    min_coverage : int
        minimum required coverage in sequencing data
    methyl_thresh : float
        minimum required methylation % for positive label
    unmethyl_thresh : float
        maximum required methylation % for negative label
    chunksize : int
        chunksize for loading deepsignal extract dataframe
    """
    print("Loading methylation reference...")
    methyl_df = pd.read_csv(methyl_filename, sep='\t', names=METHYL_BED_COLS)
    
    chunk_idx = 0
    for feat_df in pd.read_csv(feat_filename, chunksize=chunksize, sep='\t', names=DEEPSIGNAL_EXTRACT_COLS):
        print("Chunk {}: Joining dataframes...".format(chunk_idx))

        sub_methyl_df = methyl_df[['contig', 'strand', 'start_pos', 'coverage', 'methyl_pct']]
        sub_methyl_df['pos'] = sub_methyl_df['start_pos']
        sub_methyl_df = sub_methyl_df.drop('start_pos', axis=1)

        joined_df = feat_df.merge(
            sub_methyl_df,
            how='left',
            on=[
                'contig',
                'strand',
                'pos'
            ]
        )

        print("Chunk {}: Labeling data...".format(chunk_idx))

        label_df = joined_df.dropna()[
            (joined_df.coverage >= min_coverage) &
            ((joined_df.methyl_pct >= methyl_thresh) | (joined_df.methyl_pct <= unmethyl_thresh))
        ]

        label_df['label'] = (label_df.methyl_pct >= methyl_thresh).astype(int)
        label_df = label_df.drop(['coverage', 'methyl_pct'], axis=1)

        print("Chunk {}: Saving labeled data...".format(chunk_idx))

        label_df.to_csv(
            write_filename,
            sep='\t',
            mode='a',
            index=False,
            header=False
        )
        
        chunk_idx += 1
