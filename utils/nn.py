"""
Neural network implementations and utilities.
"""

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def train(network, train_data, val_data, n_epochs=1, batch_size=1024,
          optim=torch.optim.Adam, loss_fn=F.mse_loss, ckpt_name=None, print_prog=float('inf'), evals={}):
    """
    network : nn.Module
        instance of torch NN
    train_data : torch.utils.data.TensorDataset
        train feature vectors + labels
    val_data : torch.utils.data.TensorDataset
        val feature vectors + labels
    n_epochs : int
        number of epochs to train for
    batch_size : int
        number of instances in each batch for train and validation
    optim : torch.optim.Optimizer
        optimizer for NN training
    loss_fn : (torch.Tensor, torch.Tensor) -> torch.Tensor
        member of torch.nn.functional, or some custom loss function
    ckpt_name : str
        file to save checkpoints too (no save if None)
    print_prog : int
        steps between epoch updates
    evals : dict[str, (torch.Tensor, torch.Tensor) -> torch.Tensor]
        additional evaluation methods
    return : nn.Module, [][]float, []float, torch.Tensor, dict[str, []float]
        trained network, list of train losses, list of validation losses, best predictions on val set, additional metrics
    """
    if type(train_data) == np.array:
        train_data = torch.Tensor(train_data)
    
    train_data_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_data_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False
    )
    
    optimizer = optim(network.parameters())

    best_score = float('inf')
    best_preds = None
    train_losses = []
    val_losses = []
    metrics = {eval_name: [] for eval_name in evals}

    for epoch in range(n_epochs):
        if not epoch % print_prog:
            print('Epoch', epoch)
            sys.stdout.flush()
        
        network.train()
        train_losses.append([])
        for batch in train_data_loader:
            feat_batch, label_batch = batch

            optimizer.zero_grad()
            outputs = network(feat_batch)
            
            train_loss = loss_fn(outputs, label_batch)
            train_losses[-1].append(train_loss.item())
            
            train_loss.backward()
            optimizer.step()

        network.eval()
        predictions = []
        labels = []
        for batch in val_data_loader:
            feat_batch, label_batch = batch
            predictions.extend(network(feat_batch))
            labels.extend(label_batch)
        
        val_loss = loss_fn(
            torch.stack(predictions),
            torch.stack(labels)
        ).item()
        
        val_losses.append(val_loss)
        
        if not epoch % print_prog:
            print('\tTrain loss: {:.8f}'.format(np.mean(train_losses[-1])))
            print('\tValidation loss: {:.8f}'.format(val_loss))
            sys.stdout.flush()
            
            for eval_name, eval_fn in evals.items():
                curr_eval = eval_fn(
                    torch.stack(predictions),
                    torch.stack(labels)
                ).item()
                
                print('\t{}: {:.8f}'.format(eval_name, curr_eval))
                metrics[eval_name].append(curr_eval)
                sys.stdout.flush()

        if val_loss < best_score:
            best_score = val_loss
            best_preds = predictions
            if ckpt_name is not None:
                torch.save(network.state_dict(), ckpt_name)
    
    if ckpt_name is not None:
        network.load_state_dict(torch.load(ckpt_name))
    
    return network, train_losses, val_losses, best_preds, metrics

class FCNet(nn.Module):
    """
    Implements a fully connected net with adjustable # hidden layers and # hidden units.
    By default, uses ReLU activation and dropout layers between each linear layer.
    """
    def __init__(self, input_size, output_size, hidden_layers, units, p=0.5, spec_layers=[]):
        """
        input_size : int
            input feature vector size
        output_size : input
            output / label size
        hidden_layers : int
            number of hidden layers
        units : []int
            number of units in each hidden layer
        p : float
            dropout probability
        spec_layers : []nn.Module
            additional layers to be specified at end of network
        """
        assert(len(units) == hidden_layers)
        super().__init__()
        
        self.layers = []

        layer_sizes = [input_size] + units
        for i in range(hidden_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(p=p)
                )
            )
        
        self.layers.append(nn.Linear(layer_sizes[-1], output_size))
        self.layers.extend(spec_layers)
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
    
def symm_kl_loss_gaussian_torch(input, target):
    """
    Computes symmetric KL divergence for univariate Gaussians.
    input : torch.Tensor
        predicted distribution parameters
    target : torch.Tensor
        true distribution parameters
    return : torch.Tensor
        symmetric KL divergence
    """
    kl1 = torch.log(target[:, 1] / input[:, 1]) + (torch.pow(input[:, 1], 2) + torch.pow(input[:, 0] - target[:, 0], 2)) / (2 * torch.pow(target[:, 1], 2)) - .5
    kl2 = torch.log(input[:, 1] / target[:, 1]) + (torch.pow(target[:, 1], 2) + torch.pow(target[:, 0] - input[:, 0], 2)) / (2 * torch.pow(input[:, 1], 2)) - .5
    out = torch.mean(kl1 + kl2)
    if np.isnan(out.item()):
        input = F.relu(input) + 1e-5
        target = F.relu(target) + 1e-5
        return symm_kl_loss_gaussian_torch(input, target)
        
    return out
