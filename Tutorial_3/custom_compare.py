# custom_compare.py
import numpy as np
# Importe suas implementações dos arquivos correspondentes
from custom_class import MyTransformer
from custom_function import eeg_function

def create_mock_eeg_dataset(num_trials=80, num_bands=1, num_electrodes=12, num_time_points=4096):
    X_data = (np.random.rand(num_trials, num_bands, num_electrodes, num_time_points) * 240) - 120
    y_labels = np.random.randint(0, 2, num_trials)
    
    dataset = {
        'X': X_data,
        'y': y_labels,
        'sfreq': 512,
        'y_dict': {0: "left hand", 1: "right hand"},
        'events': {"cue": np.array([[0,0,3.0]])}, # Usa np.array para compatibilidade
        'ch_names': ['F3', 'FC3', 'C3', 'CP3', 'P3', 'FCz', 'CPz', 'P4', 'FC4', 'C4', 'CP4', 'P4'],
        'tmin': 0.0
    }
    return dataset

if __name__ == "__main__":
    mock_dataset = create_mock_eeg_dataset()
    
    # Configurações para o eletrodo específico
    electrode_to_monitor_idx = 5 # Índice do eletrodo
    electrode_name = mock_dataset['ch_names'][electrode_to_monitor_idx]
    trial_idx = 0 # Primeiro trial para monitorar
    band_idx = 0 # Primeira (e única) banda para monitorar
    time_slice_start = 0 # Início do trecho de tempo para visualização
    time_slice_end = 10   # Fim do trecho de tempo para visualização (10 pontos)

    print(f"\n Eletrodo '{electrode_name}' (índice {electrode_to_monitor_idx}) ")
    print(f"Trecho monitorado: Trial {trial_idx}, Banda {band_idx}, amostras {time_slice_start}:{time_slice_end}")

    # Dados originais para o eletrodo específico
    original_eeg_segment = mock_dataset['X'][trial_idx, band_idx, electrode_to_monitor_idx, time_slice_start:time_slice_end]
    print(f"\nOriginal ('{electrode_name}'): {original_eeg_segment}")
    print("-" * 50)
    
    # --- CLASSE ---
    print("\n Classe")
    class_test_dataset = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in mock_dataset.items()}
    
    transformer = MyTransformer()
    transformer.fit(class_test_dataset)
    resultado_classe_dict = transformer.transform(class_test_dataset) 
    
    # Dados transformados pela CLASSE para o eletrodo específico
    transformed_class_eeg_segment = resultado_classe_dict['X'][trial_idx, band_idx, electrode_to_monitor_idx, time_slice_start:time_slice_end]
    print(f"\nTransformado pela Classe ('{electrode_name}'): {transformed_class_eeg_segment}")
    
    # --- FUNÇÃO ---
    print("\n Função")
    function_test_dataset = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in mock_dataset.items()}
    resultado_funcao_dict = eeg_function(function_test_dataset)
    
    # Dados transformados pela FUNÇÃO para o eletrodo específico
    transformed_function_eeg_segment = resultado_funcao_dict['X'][trial_idx, band_idx, electrode_to_monitor_idx, time_slice_start:time_slice_end]
    print(f"\nTransformado pela Função ('{electrode_name}'): {transformed_function_eeg_segment}")
