import numpy as np
from scipy import signal 

def apply_custom_sinc_filterbank(eegdata: dict,
                                  sfreq: float,
                                  low_cut_bands = [4, 8, 12, 16, 20, 24, 28, 32, 36],
                                  high_cut_bands = [8, 12, 16, 20, 24, 28, 32, 36, 40]): 
    
    # Extrai os dados de EEG e os rótulos do dicionário de entrada para o trial
    X = eegdata['X'].copy() 
    labels = eegdata['y']

    # remove todas as dimensões de tamanho 1
    eeg_array_to_filter = np.squeeze(X)

    # Heurística para transpor se os canais estiverem na segunda dimensão (e.g., amostras x canais)
    if eeg_array_to_filter.shape[0] > eeg_array_to_filter.shape[1] and eeg_array_to_filter.shape[1] < 100:
        eeg_array_to_filter = eeg_array_to_filter.T 

    n_channels, n_samples = eeg_array_to_filter.shape
    list_filtered_bands = [] 

    print(f"\n--- Dados ANTES da Filtragem (Trial {labels}) ---")
    print(f"  Formato dos dados originais (canais, amostras): {eeg_array_to_filter.shape}")
    if n_channels > 0 and n_samples > 0:
        print(f" 5 amostras do Canal 0: {eeg_array_to_filter[0, :5]}")
    else:
        print("  Dados originais vazios para amostrar.")


    for i, (f_low_hz, f_high_hz) in enumerate(zip(low_cut_bands, high_cut_bands)):
        # --- CÁLCULO DINÂMICO DA LARGURA DE TRANSIÇÃO E ORDEM DO FILTRO ---
        band_transition_width_hz = (f_high_hz - f_low_hz) / 2

        if band_transition_width_hz <= 0:
            raise ValueError(f"A largura de banda de transição calculada para a banda {f_low_hz}-{f_high_hz}Hz é zero ou negativa ({band_transition_width_hz} Hz). "
                             "Isso pode ocorrer se f_high_hz <= f_low_hz ou se o intervalo for muito estreito para a regra definida.")

        normalized_transition_width = band_transition_width_hz / (sfreq / 2)

        N = int(np.ceil(4 / normalized_transition_width))
        if not N % 2: N += 1 
        # --- FIM CÁLCULO DINÂMICO ---

        # Normalizar as frequências de corte pela frequência de amostragem
        fL_norm = f_low_hz / sfreq
        fH_norm = f_high_hz / sfreq
        
        n_coeffs = np.arange(N)

        # Compute a low-pass filter with cutoff frequency fH.
        hlpf = np.sinc(2 * fH_norm * (n_coeffs - (N - 1) / 2))
        hlpf *= np.blackman(N) 
        hlpf = hlpf / np.sum(hlpf)

        # Compute a high-pass filter with cutoff frequency fL.
        hhpf = np.sinc(2 * fL_norm * (n_coeffs - (N - 1) / 2))
        hhpf *= np.blackman(N) 
        hhpf = hhpf / np.sum(hhpf) 
        hhpf = -hhpf 
        hhpf[(N - 1) // 2] += 1

        # Convolve both filters.
        h_bandpass = signal.convolve(hlpf, hhpf, mode='full')

        # Aplica o filtro passa-banda a cada canal dos dados de EEG.
        filtered_eeg_band = np.zeros_like(eeg_array_to_filter, dtype=eeg_array_to_filter.dtype)
        for ch in range(n_channels):
            filtered_eeg_band[ch, :] = signal.convolve(eeg_array_to_filter[ch, :], h_bandpass, mode='same')

        list_filtered_bands.append(filtered_eeg_band)

    # Empilha as bandas em um único array (num_bands, n_channels, n_samples)
    final_filtered_X_for_trial = np.array(list_filtered_bands)

    # --- Resumo DEPOIS da filtragem ---
    print(f"\n--- Dados DEPOIS da Filtragem (Trial {labels}) ---")
    print(f"  Formato dos dados filtrados (bandas, canais, amostras): {final_filtered_X_for_trial.shape}")
    if final_filtered_X_for_trial.shape[0] > 0 and n_channels > 0 and n_samples > 0:
        print(f" 5 amostras do Canal 0 (primeira banda): {final_filtered_X_for_trial[0, 0, :5]}")
    else:
        print("  Dados filtrados vazios para amostrar.")

    eegdata['X'] = final_filtered_X_for_trial
    
    return eegdata

# --- BLOCO DE TESTE ---
if __name__ == "__main__": 
    num_electrodes = 12
    num_time_points = 4096
    mock_X = (np.random.rand(1, 1, num_electrodes, num_time_points) * 240) - 120 # Dados EEG simulados
    mock_y = 0
    mock_sfreq = 512.0 

    mock_eegdata = {
        'X': mock_X.copy(),
        'y': mock_y
    }
    
    test_low_cut_bands = [4, 8, 12, 16, 20, 24, 28, 32, 36]
    test_high_cut_bands = [8, 12, 16, 20, 24, 28, 32, 36, 40]

    processed_eegdata = apply_custom_sinc_filterbank(
        mock_eegdata,
        sfreq=mock_sfreq,
        low_cut_bands=test_low_cut_bands, 
        high_cut_bands=test_high_cut_bands 
    )