"""
Given a (k-mer incomplete-trained) methylated 6-mer pore model, expand using NN to learn parameters for missing k-mers.
By default, only update parameters for methylated CpG k-mers which were not updated during initial training.
"""

import sys

from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("..")

from utils.bio import parse_model
from utils.bio import valid_cpg_mc_kmer

from utils.nn import train
from utils.nn import FCNet

from utils.stats import symm_kl_loss_gaussian_torch

STANDARD_KMERS = [''.join(kmer) for kmer in product('ACGT', repeat=6)]
METHYL_KMERS = [''.join(kmer) for kmer in product('ACGTM', repeat=6)]
METHYL_KMERS = [kmer for kmer in METHYL_KMERS if valid_cpg_mc_kmer(kmer)]
METHYL_ONLY_KMERS = [kmer for kmer in METHYL_KMERS if 'M' in kmer]

np.random.seed(37)

def relearn_pm(df, val_pct, n_epochs, batch_size, units, p, non_updated_kmers, methyl_val=True,
               ckpt_name=None, loss_fn=F.mse_loss, evals={}, print_prog=float('inf')):
    """
    Trains and validates a neural network for relearning the pore model.
    df : pd.DataFrame
        contains feature vectors for each k-mer and pore model statistics
    val_pct : float
        percentage of trained methylated k-mers to use as validation set
    n_epochs : int
        number of epochs for training
    batch_size : int
        batch size for training
    units : []int
        sizes for hidden layers in FCNet
    p : float
        dropout rate in network
    non_updated_kmers : set
        pre-removed k-mers
    methyl_val : bool
        whether to evaluate only on methylated k-mers
    ckpt_name : str
        checkpoint file path for training
    loss_fn : (torch.Tensor, torch.Tensor) -> torch.Tensor
        member of torch.nn.functional, or some custom loss function
    evals : dict[str, (torch.Tensor, torch.Tensor) -> torch.Tensor]
        additional evaluation methods
    print_prog : int
        print progress steps for network training
    return : FCNet, [][]float, []float, (np.array, np.array), torch.Tensor, []dict[str, []float]
        trained network, train losses, validation losses, validation set, best predictions, additional metrics
    """
    feat_mat = df.drop(['mean', 'std'], axis=1).to_numpy()
    lab_vec = df[['mean', 'std']].to_numpy()

    networks = []
    train_losses = []
    validation_losses = []
    validation_sets = []
    best_preds = []
    additional_metrics = []
    
    if methyl_val:
        present_kmers = set([kmer for kmer in df.index if valid_cpg_mc_kmer(kmer) and 'M' in kmer])
        kmer_list = set(METHYL_ONLY_KMERS)
    else:
        present_kmers = set([kmer for kmer in df.index if valid_cpg_mc_kmer(kmer)])
        kmer_list = set(METHYL_KMERS)
        
    validation_kmers = list(present_kmers.intersection(kmer_list))
    validation_kmers = set(np.random.choice(validation_kmers, size=int(val_pct * len(validation_kmers)), replace=False))
    train_kmers = np.random.permutation(
        [kmer for kmer in METHYL_KMERS if kmer not in validation_kmers and kmer not in non_updated_kmers]
    )
    validation_kmers = np.array(list(validation_kmers))
    
    assert(len(validation_kmers) + len(train_kmers) == len(METHYL_KMERS) - len(non_updated_kmers))
    
    train_df = df.loc[train_kmers]
    val_df = df.loc[validation_kmers]
    
    feat_mat_train = train_df.drop(['mean', 'std'], axis=1).to_numpy()
    lab_vec_train = train_df[['mean', 'std']].to_numpy()
    feat_mat_val = val_df.drop(['mean', 'std'], axis=1).to_numpy()
    lab_vec_val = val_df[['mean', 'std']].to_numpy()
    
    validation_set = (feat_mat_val, lab_vec_val)
    
    train_data = torch.utils.data.TensorDataset(torch.Tensor(feat_mat_train), torch.Tensor(lab_vec_train))
    val_data = torch.utils.data.TensorDataset(torch.Tensor(feat_mat_val), torch.Tensor(lab_vec_val))

    network = FCNet(
        feat_mat_train.shape[1],
        2,
        len(units),
        units,
        p=p
    )

    network, train_loss, val_loss, best_pred, metrics = train(
        network,
        train_data,
        val_data,
        n_epochs=n_epochs,
        batch_size=batch_size,
        loss_fn=loss_fn,
        ckpt_name=ckpt_name,
        print_prog=print_prog,
        evals=evals
    )
    
    return network, train_loss, val_loss, validation_set, best_preds, metrics

def symm_kl_loss_gaussian(input, target):
    """
    Computes symmetric KL divergence for univariate Gaussians.
    input : torch.Tensor
        predicted distribution parameters
    target : torch.Tensor
        true distribution parameters
    return : torch.Tensor
        symmetric KL divergence
    """
    kl1 = (torch.log(target[:, 1] / input[:, 1]) + (torch.pow(input[:, 1], 2) + torch.pow(input[:, 0] - 
               target[:, 0], 2)) / (2 * torch.pow(target[:, 1], 2)) - .5)
    kl2 = (torch.log(input[:, 1] / target[:, 1]) + (torch.pow(target[:, 1], 2) + torch.pow(target[:, 0] - 
               input[:, 0], 2)) / (2 * torch.pow(input[:, 1], 2)) - .5)
    out = torch.mean(kl1 + kl2)
    if np.isnan(out.item()):
        input = F.relu(input) + 1e-5
        target = F.relu(target) + 1e-5
        return symm_kl_loss_gaussian(input, target)
        
    return out

if __name__ == '__main__':
    # filename parameters
    base_model_filename = sys.argv[1]
    trained_model_filename = sys.argv[2]
    nn_model_filename = sys.argv[3]
    nn_fofn_filename = sys.argv[4]
    kmer_feat_filename = sys.argv[5]
    nn_ckpt_filename = sys.argv[6]
    
    # neural network training parameters
    val_pct = float(sys.argv[7])
    n_epochs = int(sys.argv[8])
    batch_size = int(sys.argv[9])
    units_sets = [[int(unit) for unit in units.split(',')] for units in sys.argv[10].split('_')]
    p = float(sys.argv[11])
    methyl_val = bool(int(sys.argv[12]))
    print_prog = int(sys.argv[13])
    use_kl = bool(int(sys.argv[14]))
    
    # pm updates
    partial = bool(int(sys.argv[15]))
    
    print("Loading k-mer pore model parameters...")
    sys.stdout.flush()
    
    base_params, _ = parse_model(base_model_filename)
    trained_params, _ = parse_model(trained_model_filename)
    
    print("Identifying k-mers not updated during training...")
    sys.stdout.flush()
    
    non_updated_kmers = []
    for kmer in METHYL_ONLY_KMERS: # only methylated k-mers should be excluded
        if base_params[kmer][0] == trained_params[kmer][0]: # only need to check location parameter
            non_updated_kmers.append(kmer)
    
    print("Preparing datasets for training...")
    sys.stdout.flush()
    
    df = pd.read_csv(kmer_feat_filename, sep='\t', index_col=0) # header and index (of k-mers) written
    
    # either learn parameters which were not initially trained, or relearn all parameters
    if partial:
        df_learn = df.loc[non_updated_kmers]
    else:
        df_learn = df.copy()
    
    df = df.loc[df.index.difference(non_updated_kmers)]
    
    # append pore model parameters to df
    df['mean'] = [trained_params[kmer][0] for kmer in df.index]
    df['std'] = [trained_params[kmer][1] for kmer in df.index]
    
    # scale dataframe
    mean_scale = df['mean'].max() - df['mean'].min()
    std_scale = df['std'].max() - df['std'].min()
    mean_shift = df['mean'].min()
    std_shift = df['std'].min()
    
    scale = torch.Tensor([mean_scale, std_scale])
    shift = torch.Tensor([mean_shift, std_shift])
    
    df['mean'] = (df['mean'] - mean_shift) / mean_scale
    df['std'] = (df['std'] - std_shift) / std_scale
    
    non_updated_kmers = set(non_updated_kmers)
    
    print("Training neural network...")
    sys.stdout.flush()
    
    if use_kl:
        loss_fn = lambda i, t: symm_kl_loss_gaussian_torch(i * scale + shift, t * scale + shift)
        evals = {
            "Scaled MSE": F.mse_loss,
            "Unscaled MSE": lambda i, t: F.mse_loss(i * scale + shift, t * scale + shift)
        }
    else:
        loss_fn = F.mse_loss
        evals = {
            "Symmetric KL Divergence": lambda i, t: symm_kl_loss_gaussian_torch(i * scale + shift, t * scale + shift),
            "Unscaled MSE": lambda i, t: F.mse_loss(i * scale + shift, t * scale + shift)
        }
    
    train_results = []
    for units in units_sets:
        ckpt_name = nn_ckpt_filename.format('_'.join(list(map(str, units))))
        network, _, val_losses, _, _, _ = relearn_pm(
            df, val_pct, n_epochs, batch_size, units, p, non_updated_kmers, methyl_val=methyl_val,
            ckpt_name=ckpt_name, loss_fn=loss_fn, evals=evals, print_prog=print_prog
        )
        train_results.append((min(val_losses), units, network))
        print("Hidden layer sizes: {}, best validation loss: {}".format(units, train_results[-1][0]))
        sys.stdout.flush()
        
    _, units, network = min(train_results, key=lambda result: result[0])
    print("Selected network with hidden layer sizes of {}".format(units))
    sys.stdout.flush()
    
    print("Predicting parameters for missing k-mers...")
    sys.stdout.flush()
    
    new_params = network(torch.Tensor(df_learn.to_numpy()))
    new_params_dict = {}
    for i in range(df_learn.shape[0]):
        new_params_dict[df_learn.index[i]] = (
            new_params[i][0].item() * mean_scale + mean_shift,
            new_params[i][1].item() * std_scale + std_shift
        )
    
    print("Writing model file...")
    sys.stdout.flush()
    
    # note that the third and fourth parameters are never used, so no updates
    with open(trained_model_filename, 'r') as f1:
        with open(nn_model_filename, 'w') as f2:
            line = f1.readline()
            while line:
                if line[0] == '#': # header line
                    f2.write(line)
                else:
                    line_split = line.split('\t')
                    if ((not partial and valid_cpg_mc_kmer(line_split[0]) and 'M' in line_split[0])
                           or line_split[0] in non_updated_kmers):
                        line_split[1] = str(np.round(new_params_dict[line_split[0]][0], 4))
                        line_split[2] = str(np.round(new_params_dict[line_split[0]][1], 4))
                    f2.write('\t'.join(line_split))
                line = f1.readline()
    
    with open(nn_fofn_filename, 'w') as f:
        f.write(nn_model_filename)
