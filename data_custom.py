import numpy as np
from bciflow.modules.core.kfold import kfold
from bciflow.modules.tf.bandpass.chebyshevII import chebyshevII
from bciflow.modules.fe.logpower import logpower
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import scipy.io 

filepath_train = 'data/cbcic/parsed_P01T.mat'
filepath_eval = 'data/cbcic/parsed_P01E.mat'

# scipy.io.loadmat lê o arquivo e retorna um dicionário
mat_data_train = scipy.io.loadmat(filepath_train)
mat_data_eval = scipy.io.loadmat(filepath_eval)

# 'RawEEGData' é a chave dentro do dicionário mat_data para os sinais EEG
rawdata_train = mat_data_train['RawEEGData']
rawdata_eval = mat_data_eval['RawEEGData']

# Concatena os dados de treinamento e avaliação ao longo do eixo 'trials'
X_combined = np.concatenate((rawdata_train, rawdata_eval), axis=0)

# === EEG data shaped as (trials, bands, channels, time) ===
X = np.expand_dims(X_combined, axis=1)

# y_train (80 trials):
y_train_part = []
y_train_part.extend([0] * 10) 
y_train_part.extend([1] * 20) 
y_train_part.extend([0] * 10) 
y_train_part.extend([1] * 20) 
y_train_part.extend([0] * 20) 
y_train = np.array(y_train_part, dtype=int)

# y_eval (40 trials):
y_eval_part = []
y_eval_part.extend([1] * 20) 
y_eval_part.extend([0] * 20) 
y_eval = np.array(y_eval_part, dtype=int)

# === Label vector (one integer per trial) ===
y = np.concatenate((y_train, y_eval), axis=0)

# === Sampling frequency (in Hz) ===
sfreq = float(mat_data_train['sampRate'].flatten()[0]) 

# === Dictionary mapping labels to class names ===
y_dict = {'left-hand': 0, 'right-hand': 1} 

# === List of EEG channel names ===
ch_names = np.array(['F3', 'FC3', 'C3', 'CP3', 'P3', 'FCz', 'CPz', 'F4', 'FC4', 'C4', 'CP4', 'P4'])

# === Time offset relative to the event ===
tmin = 0.0 

# events: Cria dicionário de meta-informações para corresponder à saída desejada
# 'cueAt' é a chave dentro do dicionário mat_data_train
cue_start_time_in_trial = float(mat_data_train['cueAt'].flatten()[0]) 
events = {
    'get_start': [0, cue_start_time_in_trial], 
    'beep_sound': [cue_start_time_in_trial - 1], 
    'cue': [cue_start_time_in_trial, cue_start_time_in_trial + 5], 
    'task_exec': [cue_start_time_in_trial, cue_start_time_in_trial + 5] 
}

# Group all elements into a single dictionary
custom_dataset = {
    'X': X,
    'y': y,
    'sfreq': sfreq,
    'y_dict': y_dict,
    'events': events,
    'ch_names': ch_names,
    'tmin': tmin
}

print(f"EEG signals shape: {custom_dataset['X'].shape}")
print(f"Labels: {custom_dataset['y']}") 
print(f"Class dictionary: {custom_dataset['y_dict']}")
print(f"Events: {custom_dataset['events']}")
print(f"Channel names: {custom_dataset['ch_names']}")
print(f"Sampling frequency (Hz): {custom_dataset['sfreq']}")
print(f"Start time (s): {custom_dataset['tmin']}")