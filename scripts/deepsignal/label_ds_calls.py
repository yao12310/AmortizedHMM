"""
Add ground-truth labels to deepsignal call_mods output.
"""

import sys

import pandas as pd

sys.path.append("..")

from utils.constants.functional import METHYL_BED_COLS_NA
from utils.constants.functional import DS_CALL_COLS

if __name__ == '__main__':
    call_filename = sys.argv[1]
    methyl_filename = sys.argv[2]
    write_filename = sys.argv[3]
    
    print("Loading dataframes...")
    
    call_df = pd.read_csv(call_filename, sep='\t', names=DS_CALL_COLS)
    methyl_df = pd.read_csv(methyl_filename, sep='\t', names=METHYL_BED_COLS_NA)
    
    print("Joining dataframes...")
    
    sub_methyl_df = methyl_df[['contig', 'strand', 'start_pos', 'coverage', 'methyl_pct']]
    sub_methyl_df['pos'] = sub_methyl_df['start_pos']
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
    
    print("Saving labeled data...")
    
    joined_df.to_csv(write_filename, sep='\t', index=False, header=False)
