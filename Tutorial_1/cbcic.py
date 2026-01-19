from bciflow.datasets.cbcic import cbcic

dataset = cbcic(subject=1, path='data/cbcic/')

# dimensões do array principal, contendo os sinais de EEG(trials, bandas, quantidade de eletrodos e tempo)
print("EEG signals shape:", dataset["X"].shape)

# sequência de rótulos de classe (inteiros) para cada trial do dataset 
print("Labels:", dataset["y"])

# dicionário que traduz os rótulos numéricos para nomes de classes descritivos
print("Class dictionary:", dataset["y_dict"])

# marcadores de tempo de eventos importantes dentro das trials
print("Events:", dataset["events"])

# lista dos nomes dos eletrodos de EEG utilizados na coleta dos dados
print("Channel names:", dataset["ch_names"])

# taxa em Hertz na qual os sinais de EEG foram digitalizados (amostras por segundo)
print("Sampling frequency (Hz):", dataset["sfreq"])

# ponto de partida temporal do segmento de EEG em relação ao evento principal do trial
print("Start time (s):", dataset["tmin"])