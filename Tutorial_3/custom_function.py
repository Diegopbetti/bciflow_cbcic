import numpy as np

def eeg_function(eegdata: dict, **kwargs):
    X = eegdata['X'].copy()

    global_mean = np.mean(X)
    print(f"- Média total (global_mean): {global_mean:.4f}")
    print(f"- Média total × 5: {global_mean * 5:.4f}")
    
    local_median_per_electrode = np.median(X, axis = (0, -1), keepdims = True)
    print(f"- Mediana local por eletrodo: {local_median_per_electrode.flatten()}")
    
    offset_value = (global_mean * 5) + local_median_per_electrode
    print(f"- Soma (passo2 + passo3): {offset_value.flatten()}")
     
    eegdata['X'] = X - offset_value
    print(f"- Resultado final (passo4 - dados): shape {eegdata['X'].shape}, média {np.mean(eegdata['X']):.4f}")

    return eegdata