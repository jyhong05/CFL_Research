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