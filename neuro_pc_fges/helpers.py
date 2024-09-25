import numpy as np
from causallearn.search.ConstraintBased.PC import pc
import sys
sys.path.append("../astro_pc_fges_test")
from estimate_parameters import *
sys.path.append("../astro_pc_fges_test/fges-py")
from SEMScore import *
from fges import *
from SemEstimator import SemEstimator

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
def pc_get_avg_adj_matrix(data, n_samples, num_samplings, alpha=0.05):
    adj_mats = []
    for _ in range(num_samplings):
        subsamples = subsample_data(data, n_samples)
        for subsample in subsamples:
            cg = pc(subsample, alpha=alpha, indep_test='fisherz')
            adj_mats.append(np.abs(cg.G.graph))
    
    return np.mean(adj_mats, axis=0)

# FGES infer edges
def fges_infer_edges(data, s=8):
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

# FGES get avg adjacency matrix, also returns fges results for later use
def fges_avg_adj_mat_and_results(data, n_samples, num_samplings, s=8):
    adj_mats = []
    fges_results = []
    for _ in range(num_samplings):
        subsamples = subsample_data(data, n_samples)
        for subsample in subsamples:
            edges, fges_result = fges_infer_edges(subsample, s=s)
            adj_mats.append(fges_edges_to_mat(edges, data.shape[1]))
            fges_results.append(fges_result)
    
    return np.mean(adj_mats, axis=0), fges_results

# estimated correlation matrix
def fges_estimate_corr(data, fges_result):
    '''
    Arguments:
        data : an n_samples x n_nodes numpy array
        fges_result : a dict of results returned by fges.search()
    Returns:
        est_corr : an n_nodes x n_nodes numpy array estimated correlation matrix
    '''
    sem_est = SemEstimator(data, sparsity=4)

    # provide to the estimator the DAG found above
    sem_est.pattern = fges_result['graph']

    # estimate the weights and residuals
    sem_est.estimate()

    # get covariance matrix from SemEstimator
    est_cov = sem_est.graph_cov

    # compute correlation matrix from covariance matrix
    stdistdj = np.sqrt(np.diag(est_cov))
    est_corr = est_cov / np.outer(stdistdj, stdistdj)

    np.fill_diagonal(est_corr, 0)
    return est_corr

# mean absolute deviation, way to measure differences between adjacency matrices
def mean_abs_deviations(adj_mats):
    avg_mat = np.mean(adj_mats, axis=0)
    mad_mats = [np.abs(mat - avg_mat) for mat in adj_mats]
    total_mad = np.sum(np.sum(mad_mats, axis=0))

    return mad_mats, total_mad

def estimate_corr(data, fges_result):
    '''
    Arguments:
        data : an n_samples x n_nodes numpy array
        fges_result : a dict of results returned by fges.search()
    Returns:
        est_corr : an n_nodes x n_nodes numpy array estimated correlation matrix
    '''
    sem_est = SemEstimator(data, sparsity=4)

    # provide to the estimator the DAG found above
    sem_est.pattern = fges_result['graph']

    # estimate the weights and residuals
    sem_est.estimate()

    # get covariance matrix from SemEstimator
    est_cov = sem_est.graph_cov

    # compute correlation matrix from covariance matrix
    stdistdj = np.sqrt(np.diag(est_cov))
    est_corr = est_cov / np.outer(stdistdj, stdistdj)
    return est_corr

def edge_names_from_adj_mat(adj_mat):
    edge_names = []
    for i in range(adj_mat.shape[0]):
        for j in range(i, adj_mat.shape[1]):
            if adj_mat[i, j] == 1:
                edge_names.append((i, j))
    return edge_names
