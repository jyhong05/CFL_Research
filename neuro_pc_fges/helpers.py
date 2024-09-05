import numpy as np
from causallearn.search.ConstraintBased.PC import pc
import sys
sys.path.append("../astro_pc_fges_test")
from estimate_parameters import *

# subsampling
def subsample_data(data, n_samples):
    idxs = np.arange(data.shape[0])
    np.random.shuffle(idxs)
    
    assert data.shape[0] % n_samples == 0, "length of data needs to be divisible by num data points"
    subsample_idxs = np.array_split(idxs, data.shape[0] // n_samples)
    return [data[idxs] for idxs in subsample_idxs]

# getting data
def get_fmri_data():
    all_data = {}
    combined_data = {}

    for subject_num in range(1,17):
        if subject_num < 10:
            subject_num = f"0{subject_num}"
        subject_data = []
        for i in range(1,3):
            filename = f"fmri_data/sub-{subject_num}_Part{i}_Average_ROI_n50.csv"
            data = np.loadtxt(filename, delimiter=",", dtype=float, skiprows=1)
            all_data[f"s{subject_num}p{i}"] = data
            subject_data.append(data)
        combined_data[f"s{subject_num}"] = np.concatenate(subject_data, axis=0)
    return all_data, combined_data

# PC get avg adjacency matrix
def get_avg_adj_matrix(data, n_samples, num_samplings, alpha=0.05):
    adj_mats = []
    for _ in range(num_samplings):
        subsamples = subsample_data(data, n_samples)
        for subsample in subsamples:
            cg = pc(subsample, alpha=alpha, indep_test='fisherz')
            adj_mats.append(np.abs(cg.G.graph))
    
    return np.mean(adj_mats, axis=0)

# FGES infer edges
def infer_edges(data, s=8):
    '''
    Arguments:
        data : an n_samples x n_nodes array
        s : sparsity parameter for FGES (default = 8 as was used in Dubois et al.)
    Returns:
        edges : a list of tuples, where each tuple (i,j) represents an edge 
                found between node i and node j
        fges_result : dict of results from fges.search() (needed for estimating
                      the correlation matrix later on)
    '''

    # FGES takes a score function that depends on the data and a user-determined
    # sparsity level (penalty discount)
    score = SEMBicScore(penalty_discount=s, dataset=data)

    # run FGES
    fges = FGES(range(data.shape[1]), score, filename=data)
    fges_result = fges.search()
    edges = fges_result['graph'].edges()
    return edges, fges_result

# FGES edges to matrix
def fges_edges_to_mat(edges, n_nodes):
    adj_mat = np.zeros((n_nodes, n_nodes))
    for i, j in edges:
        adj_mat[i, j] = 1
        adj_mat[j, i] = 1
    return adj_mat