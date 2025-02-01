import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def get_alt_temp_grids(data):
    asc_longs = np.sort(data['long'].unique())
    desc_lats = np.sort(data['lat'].unique())[::-1]

    temp_grid = []
    alt_grid = []

    for lat in desc_lats:
        temp_row = []
        alt_row = []
        for long in asc_longs:
            point = data.loc[(data['lat'] == lat) & (data['long'] == long)]
            if len(point) == 0:
                temp_row.append(np.nan)
                alt_row.append(np.nan)
                continue
            temp_row.append(point['generated_temp'].values[0])
            alt_row.append(point['elevation'].values[0])
        temp_grid.append(temp_row)
        alt_grid.append(alt_row)
    
    return alt_grid, temp_grid

def get_group_avgs(data, xlbls):
    temps = list(data['generated_temp'])
    xlbl_dict = {xlbl:[] for xlbl in set(xlbls)}
    xlbl_avgs = {xlbl:0 for xlbl in set(xlbls)}

    for i, xlbl in enumerate(xlbls):
        xlbl_dict[xlbl].append(temps[i])
    for key, arr in zip(xlbl_dict.keys(), xlbl_dict.values()):
        xlbl_avgs[key] = sum(arr)/len(arr)

    return xlbl_avgs, xlbl_dict

def reconstruct_groups(data, xlbls, plot=True, title='Reconstructed Temps'):
    df_copy = data[['lat', 'long']].copy()

    df_copy['xlbl'] = xlbls
    group_avgs, _ = get_group_avgs(data, xlbls)

    asc_longs = np.sort(df_copy['long'].unique())
    desc_lats = np.sort(df_copy['lat'].unique())[::-1]
    
    grid = []
    points_count = 0

    for lat in desc_lats:
        row = []
        for long in asc_longs:
            point = df_copy[(df_copy['lat'] == lat) & (df_copy['long'] == long)]
            if point.empty:
                row.append(np.nan)
                continue
            
            group = point['xlbl'].values[0]
            row.append(group_avgs[group])
            points_count += 1
        
        grid.append(row)
    
    if plot:
        plt.imshow(grid, cmap='hot_r', interpolation='nearest', vmin=data['generated_temp'].min(), vmax=data['generated_temp'].max())
        plt.colorbar()
        plt.title(title)

    return grid

def by_cluster_abs_err(data, xlbls, truth):
    pred_group_avgs, _ = get_group_avgs(data, xlbls)
    _, true_groups = get_group_avgs(truth, xlbls)
    abs_err = 0

    for group in pred_group_avgs.keys():
        true_temps = np.array(true_groups[group])
        pred_temp = pred_group_avgs[group]
        abs_err += np.mean(np.abs(true_temps - pred_temp))
    
    return abs_err / len(pred_group_avgs.keys())


# ALL DEPRECATED, WRONG IMPLEMENTATION (PREDICTED ELEVATION NOT TEMP)
'''
def assert_grids(true_grid, pred_grid):
    assert len(true_grid) == len(pred_grid)
    assert len(true_grid[0]) == len(pred_grid[0])
    for i in range(len(true_grid)):
        for j in range(len(true_grid[0])):
            assert np.isnan(true_grid[i][j]) == np.isnan(pred_grid[i][j])

def mean_abs_err(true_grid, pred_grid):
    diff = np.array(true_grid) - np.array(pred_grid)
    count = np.count_nonzero(~np.isnan(diff))
    return np.nansum(np.abs(diff)) / count

def plot_abs_err(true_grid, pred_grid):
    abs_diff = np.abs(np.array(true_grid) - np.array(pred_grid))
    abs_diff[abs_diff == 0] = np.nan
    plt.imshow(abs_diff, cmap='hot_r', interpolation='nearest', vmin=0)
    plt.colorbar()
    plt.title('Absolute Error')
    plt.show()

def mean_squared_err(true_grid, pred_grid):
    diff = np.array(true_grid) - np.array(pred_grid)
    count = np.count_nonzero(~np.isnan(diff))
    return np.nansum(diff**2)/count

def plot_squared_err(true_grid, pred_grid):
    diff = np.array(true_grid) - np.array(pred_grid)
    diff[diff == 0] = np.nan
    plt.imshow(diff**2, cmap='hot_r', interpolation='nearest', vmin=0)
    plt.colorbar()
    plt.title('Squared Error')
    plt.show()

def err(true_grid, pred_grid, err_type='mean_abs'):
    assert_grids(true_grid, pred_grid)
    if err_type == 'mean_abs':
        return mean_abs_err(true_grid, pred_grid)
    elif err_type == 'mean_squared':
        return mean_squared_err(true_grid, pred_grid)
    else:
        raise ValueError('Invalid error type')

def plot_err(true_grid, pred_grid, err_type='abs'):
    assert_grids(true_grid, pred_grid)
    if err_type == 'abs':
        plot_abs_err(true_grid, pred_grid)
    elif err_type == 'squared':
        plot_squared_err(true_grid, pred_grid)
    else:
        raise ValueError('Invalid error type')
    
def r2(true_grid, pred_grid):
    true_grid, pred_grid = np.array(true_grid), np.array(pred_grid)
    mask = ~np.isnan(true_grid) & ~np.isnan(pred_grid)
    return r2_score(np.ravel(true_grid[mask]), np.ravel(pred_grid[mask]))

def adjusted_r2(true_grid, pred_grid, n_clusters):
    true_grid, pred_grid = np.array(true_grid), np.array(pred_grid)
    mask = ~np.isnan(true_grid) & ~np.isnan(pred_grid)
    n_points = np.count_nonzero(mask)
    r2 = r2_score(np.ravel(true_grid[mask]), np.ravel(pred_grid[mask]))

    return 1 - (1 - r2) * (n_points - 1) / (n_points - n_clusters - 1)
'''