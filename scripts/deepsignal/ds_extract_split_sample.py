"""
Train/test split for deepsignal extract output; additionally sample train/validation datasets.
"""

import multiprocessing as mp
import sys

import pandas as pd

sys.path.append('..')

from utils.constants.final_analysis import NUM_FOLDS
from utils.constants.final_analysis import DS_EXTRACT_COLS

def split_sample_extract(kmer_pct, train_df_filename, filter_df_filename, val_df_filename, val_conf_df_filename,
                         val_fold, train_size, val_size, split_val, skip_concat, balance):
    """
    Split and sample training/validation dataset for ds training.
    kmer_pct : int
        current percentage of k-mers to keep
    train_df_filename : str
        filename of train dataframe
    filter_df_filename : str
        format string for filename of filter dataframe
    val_df_filename : str
        filename of val dataframe
    val_conf_df_filename : str
        filename of val fold conf dataframe
    val_fold : int
        fold to use as validation data
    train_size : int
        maximum training set size
    val_size : int
        validation set size
    split_val : bool
        whether to split validation set
    skip_concat : bool
        whether to skip concatenation (go straight to sample)
    balance : bool
        balance +/- classes for training data
    return : None
    """
    if not skip_concat:
        for fold_idx in range(1, NUM_FOLDS + 1):        
            print("{}: Processing fold {}...".format(kmer_pct, fold_idx))
            sys.stdout.flush()

            if fold_idx == val_fold:
                if not split_val: # val set shared across %s, so should only do once
                    continue
                curr_df = pd.read_csv(
                    val_conf_df_filename,
                    sep='\t',
                    names=DS_EXTRACT_COLS,
                    index_col=False
                )

                if curr_df.shape[0] > val_size:
                    curr_df = curr_df.sample(val_size)

                curr_df.to_csv(
                    val_df_filename,
                    sep='\t',
                    header=False,
                    index=False
                )
            else:
                curr_df_filename = filter_df_filename.format(fold_idx)

                curr_df = pd.read_csv(
                    curr_df_filename,
                    sep='\t',
                    names=DS_EXTRACT_COLS,
                    index_col=False
                )
                
                # not strictly necessary since we'll do sampling on the full df later
                # but helpful for avoiding memory issues
                if balance:
                    pos_df = curr_df[curr_df.label.astype(bool)]
                    neg_df = curr_df[~curr_df.label.astype(bool)]
                    if pos_df.shape[0] > neg_df.shape[0]:
                        pos_df = pos_df.sample(neg_df.shape[0])
                    else:
                        neg_df = neg_df.sample(pos_df.shape[0])

                    curr_df = pd.concat((pos_df, neg_df))
                    curr_df = curr_df.sample(frac=1)
                    
                if curr_df.shape[0] > train_size:
                    curr_df = curr_df.sample(train_size)

                curr_df.to_csv(
                    train_df_filename,
                    sep='\t',
                    mode='a',
                    header=False,
                    index=False
                )
    
    print("{}: Sampling full dataframe...".format(kmer_pct))
    sys.stdout.flush()
    
    full_df = pd.read_csv(
        train_df_filename,
        sep='\t',
        names=DS_EXTRACT_COLS,
        index_col=False
    )
    
    if balance:
        pos_df = full_df[full_df.label.astype(bool)]
        neg_df = full_df[~full_df.label.astype(bool)]
        if pos_df.shape[0] > neg_df.shape[0]:
            pos_df = pos_df.sample(neg_df.shape[0])
        else:
            neg_df = neg_df.sample(pos_df.shape[0])
        
        full_df = pd.concat((pos_df, neg_df))
        full_df = full_df.sample(frac=1)

    if full_df.shape[0] > train_size:
        full_df = full_df.sample(train_size)

    full_df.to_csv(
        train_df_filename,
        sep='\t',
        mode='w',
        header=False,
        index=False
    )

if __name__ == '__main__':
    val_fold = int(sys.argv[1])
    kmer_pcts = [int(pct) for pct in sys.argv[2].split(',')]
    train_df_filename = sys.argv[3]
    filter_df_filename = sys.argv[4]
    val_df_filename = sys.argv[5]
    val_conf_df_filename = sys.argv[6]
    train_size = int(sys.argv[7])
    val_size = int(sys.argv[8])
    split_val = bool(int(sys.argv[9]))
    
    if len(sys.argv) > 10:
        skip_concat = [bool(int(arg)) for arg in sys.argv[10].split(',')]
    else:
        skip_concat = [False for _ in range(len(kmer_pcts))]
    
    if len(sys.argv) > 11:
        balance = bool(int(sys.argv[11]))
    else:
        balance = False
    
    args = [
               (
                   pct,
                   train_df_filename.format(pct),
                   filter_df_filename.format("{}", pct),
                   val_df_filename,
                   val_conf_df_filename,
                   val_fold,
                   train_size,
                   val_size,
                   split_val and pct == kmer_pcts[0], # only do once per fold
                   skip,
                   balance
               )
        for pct, skip in zip(kmer_pcts, skip_concat)
    ]
    
    pool = mp.Pool(mp.cpu_count())
    with pool:
        pool.starmap(split_sample_extract, args)