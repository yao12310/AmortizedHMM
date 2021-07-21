"""
Utils specific to biological / nanopore data.
"""

def valid_cpg_mc_kmer(kmer):
    """
    Determine if a k-mer is either unmethylated, or is methylated at a CpG site.
    kmer : str
        string k-mer (assume alphabet = 'ACGTM')
    return : bool
        whether kmer is valid or not
    """
    if 'M' not in kmer:
        return True
    for i in range(len(kmer)):
        if kmer[i] == 'M':
            if i + 1 < len(kmer) and kmer[i + 1] != 'G':
                return False
    return True

def parse_model(filename):
    """
    Parse a pore model, i.e. output from nanopolish train.
    filename : str
        path to model file
    return : dict[str, tuple], dict[str, str]
        k-mer parameters, model metadata
    """
    kmer_params = {}
    model_metadata = {}
    
    with open(filename, 'r') as f:
        line = f.readline()
        while line[0] == '#':
            try:
                tag, data = line.split('\t')
            except ValueError:
                tag, data = line.split()
            tag = tag[1:]
            data = data[:-1]
            model_metadata[tag] = data
            line = f.readline()
        
        while line:
            kmer, stat1, stat2, stat3, stat4 = line[:-1].split('\t')
            kmer_params[kmer] = tuple([float(stat) for stat in [stat1, stat2, stat3, stat4]])
            line = f.readline()
            
    return kmer_params, model_metadata

def parse_ilp_sol(sol_filename, full_set):
    """
    Parse a PuLP-generated ILP solution for k-mer selection.
    sol_filename : str
        soution filename
    full_set : set
        complete set of k-mers
    return float, set
        objective value, set of k-mers to keep
    """
    kmer_set = full_set
    with open(sol_filename, 'r') as f:
        f.readline()
        obj_value = float(f.readline().split(' = ')[-1][:-1])
        line = f.readline()
        
        while line[0] == 'x':
            var_name, val = line.split()
            kmer = var_name.split('_')[1]
            val = int(val[0])
            
            if not val:
                kmer_set.remove(kmer)
            
            line = f.readline()
    
    return obj_value, kmer_set
