import numpy as np

def eeg_function(eegdata: dict, **kwargs):
    X = eegdata['X'].copy()

    global_mean = np.mean(X)
    local_median_per_electrode = np.median(X, axis = (0, -1), keepdims = True)
    
    offset_value = (global_mean * 5) + local_median_per_electrode
    eegdata_transformed = offset_value - X

    return eegdata_transformed