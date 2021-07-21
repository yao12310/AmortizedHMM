"""
Find the subset of k1-mers which enables the most k2-mers.
"""

import os
import pickle
import sys

from collections import defaultdict
from itertools import product

import pulp

from utils.misc import valid_cpg_mc_kmer
from utils.misc import parse_ilp_sol

from utils.constants import KMER_LEN

def solve_kmer_ilp(pct, obj_weights_filename, log_filename, sol_filename, kmer_set_filename, solver, threads, limit):
    """
    pct : int
        percentage of 6-mers to retain
    obj_weights_filename : str
        filename for pickled dict[str, float] mapping from 11-mers to obj fn weights
    log_filename : str
        log output filename
    sol_filename : str
        ILP solution filename template
    kmer_set_filename : str
        pickle file to write k-mer set to
    solver : str
        ILP solver choice
    threads : int
        number of threads to use in ILP solver
    limit : int
        time limit (seconds)
    """
    
    with open(obj_weights_filename, 'rb') as f:
        obj_weights = pickle.load(f)
    
    print("Generating k-mer lists...")
    sys.stdout.flush()
    
    k1 = KMER_LEN
    k2 = 2 * KMER_LEN - 1
    
    METHYL_K1MERS = [''.join(kmer) for kmer in product('ACGTM', repeat=k1)]
    METHYL_K1MERS = [kmer for kmer in METHYL_K1MERS if valid_cpg_mc_kmer(kmer)]
    METHYL_ONLY_K1MERS = [kmer for kmer in METHYL_K1MERS if 'M' in kmer]
    
    METHYL_K2MERS = [''.join(kmer) for kmer in product('ACGTM', repeat=k2)]
    METHYL_K2MERS = [kmer for kmer in METHYL_K2MERS if valid_cpg_mc_kmer(kmer)]
    METHYL_ONLY_K2MERS = [kmer for kmer in METHYL_K2MERS if 'M' in kmer]
    METHYL_ONLY_K2MERS = [kmer for kmer in METHYL_ONLY_K2MERS if kmer[k1 - 1] + kmer[k1] == 'MG']
    METHYL_ONLY_K2MERS = [kmer for kmer in METHYL_ONLY_K2MERS if kmer in obj_weights] # some k2-mers don't appear
    
    sys.stdout.flush()
    
    n = int(len(METHYL_ONLY_K1MERS) * pct / 100)
    
    print("Initializing ILP variables and constraints...")
    sys.stdout.flush()
    
    prob = pulp.LpProblem("k1_{}_k2_{}_n_{}".format(k1, k2, n), pulp.LpMaximize)
    
    xs = {
        kmer: pulp.LpVariable(
            "x_{}".format(kmer),
            lowBound=0,
            upBound=1,
            cat='Binary'
        ) for kmer in METHYL_ONLY_K1MERS
    } # binary variables for k1-mer membership
    ys = {
        kmer: pulp.LpVariable(
            "y_{}".format(kmer),
            lowBound=0,
            upBound=1,
            cat='Binary'
        ) for kmer in METHYL_ONLY_K2MERS
    } # binary variables for k2-mer validity
    
    prob += sum(xs.values()) <= n # budget constraint
    for k2mer in METHYL_ONLY_K2MERS: # validity constraint
        k1mers = [k2mer[i:(i + k1)] for i in range(k2 - k1 + 1)]
        prob += sum([xs[k1mer] for k1mer in k1mers]) - (k2 - k1 + 1) * ys[k2mer] <= (k2 - k1) # n-way AND
        prob += -sum([xs[k1mer] for k1mer in k1mers]) + (k2 - k1 + 1) * ys[k2mer] <= 0
        
    obj = pulp.LpAffineExpression({ys[kmer]: obj_weights[kmer] for kmer in METHYL_ONLY_K2MERS})
        
    prob += obj

    print("Initializing solver...")
    sys.stdout.flush()
    
    solver = pulp.getSolver(
        solver,
        timeLimit=limit,
        msg=True,
        **{
            "threads": threads,
            "logPath": log_filename,
            "solFiles": sol_filename
        }
    )
    
    print("Solving ILP...")
    sys.stdout.flush()
    
    status = prob.solve(solver)
    
    print("ILP solving status: {}".format(pulp.LpStatus[status]))
    sys.stdout.flush()
    
    sol_idx = 0
    while os.path.exists(sol_filename + '_{}.sol'.format(sol_idx)):
        sol_idx += 1
        
    obj_value, kmer_set = parse_ilp_sol(sol_filename + '_{}.sol'.format(sol_idx - 1), set(METHYL_ONLY_K1MERS))
    
    with open(kmer_set_filename, 'wb'):
        pickle.dump(kmer_set, kmer_set_filename, protocol=pickle.HIGHEST_PROTOCOL)
    