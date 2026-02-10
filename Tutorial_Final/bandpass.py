import numpy as np
from scipy import signal 

def apply_custom_sinc_filterbank(
    eegdata: dict,
    low_cut_bands=[4, 8, 12, 16, 20, 24, 28, 32, 36],
    high_cut_bands=[8, 12, 16, 20, 24, 28, 32, 36, 40],
):
    sfreq = eegdata['sfreq']
    X = eegdata['X']        # (n_trials, 1, n_channels, n_samples)
    labels = eegdata['y']

    n_trials = X.shape[0]
    n_channels = X.shape[2]
    n_samples = X.shape[3]

    all_trials_filtered = []

    print(f"\nTotal de trials: {n_trials}")

    for t in range(n_trials):

        # =============================
        # DADOS ANTES DA FILTRAGEM
        # =============================
        eeg_trial = X[t, 0, :, :]  # (canais, amostras)

        print(f"\n--- Dados ANTES da Filtragem (Trial {t}) ---")
        print(f"  Formato dos dados originais (canais, amostras): {eeg_trial.shape}")

        if n_channels > 0 and n_samples > 0:
            print(f"  5 amostras do Canal 0: {eeg_trial[0, :5]}")
        else:
            print("  Dados originais vazios para amostrar.")

        list_filtered_bands = []

        # =============================
        # FILTRAGEM POR BANDA
        # =============================
        for i, (f_low_hz, f_high_hz) in enumerate(zip(low_cut_bands, high_cut_bands)):

            band_transition_width_hz = (f_high_hz - f_low_hz) / 2

            if band_transition_width_hz <= 0:
                raise ValueError(
                    f"Largura de transição inválida na banda "
                    f"{f_low_hz}-{f_high_hz} Hz"
                )

            normalized_transition_width = band_transition_width_hz / (sfreq / 2)

            N = int(np.ceil(4 / normalized_transition_width))
            if N % 2 == 0:
                N += 1

            fL_norm = f_low_hz / sfreq
            fH_norm = f_high_hz / sfreq

            n = np.arange(N)

            # Low-pass
            hlpf = np.sinc(2 * fH_norm * (n - (N - 1) / 2))
            hlpf *= np.blackman(N)
            hlpf /= np.sum(hlpf)

            # High-pass
            hhpf = np.sinc(2 * fL_norm * (n - (N - 1) / 2))
            hhpf *= np.blackman(N)
            hhpf /= np.sum(hhpf)
            hhpf = -hhpf
            hhpf[(N - 1) // 2] += 1

            # Band-pass
            h_bandpass = signal.convolve(hlpf, hhpf, mode='full')

            filtered_band = np.zeros_like(eeg_trial)

            for ch in range(n_channels):
                filtered_band[ch, :] = signal.convolve(
                    eeg_trial[ch, :],
                    h_bandpass,
                    mode='same'
                )

            list_filtered_bands.append(filtered_band)

        # (bandas, canais, amostras)
        trial_filtered = np.array(list_filtered_bands)

        # =============================
        # DADOS DEPOIS DA FILTRAGEM
        # =============================
        print(f"\n--- Dados DEPOIS da Filtragem (Trial {t}) ---")
        print(
            f"  Formato dos dados filtrados (bandas, canais, amostras): "
            f"{trial_filtered.shape}"
        )

        if trial_filtered.shape[0] > 0:
            print(
                f"  5 amostras do Canal 0 (primeira banda): "
                f"{trial_filtered[0, 0, :5]}"
            )
        else:
            print("  Dados filtrados vazios para amostrar.")

        all_trials_filtered.append(trial_filtered)

    # (n_trials, n_bands, n_channels, n_samples)
    eegdata['X'] = np.array(all_trials_filtered)

    print(f"\nFormato FINAL do array X: {eegdata['X'].shape}")
    return eegdata


# ==================================
# BLOCO DE TESTE
# ==================================
if __name__ == "__main__":
    num_trials = 6
    num_electrodes = 12
    num_time_points = 4096

    mock_X = (np.random.rand(
        num_trials, 1, num_electrodes, num_time_points
    ) * 240) - 120

    mock_y = np.arange(num_trials)
    mock_sfreq = 512.0

    mock_eegdata = {
        'X': mock_X.copy(),
        'y': mock_y
    }

    processed_eegdata = apply_custom_sinc_filterbank(
        mock_eegdata,
        sfreq=mock_sfreq
    )
