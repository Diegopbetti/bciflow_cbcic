import numpy as np

class MyTransformer:
    def __init__(self):
        self.total_mean = None
        self.local_median_per_electrode = None

    def fit(self, eegdata:dict, **kwargs):
        X_train = eegdata['X'] 
        self.global_mean_ = np.mean(X_train)
        self.local_median_per_electrode = np.median(X_train, axis=(0, -1), keepdims=True)

        return self

    def transform(self, eegdata:dict, **kwargs):
        X_eegdata = eegdata['X'].copy()

        # Garante que fit() foi chamado antes de transform()
        if self.global_mean_ is None or self.local_median_per_electrode is None:
            raise RuntimeError(
                "Fit method must be called before transform. Parameters have not been learned yet."
            )

        # Realiza a conta usando os parâmetros aprendidos
        print(f"- Média total (global_mean): {self.global_mean_:.4f}")
        print(f"- Média total × 5: {self.global_mean_ * 5:.4f}")
        print(f"- Mediana local por eletrodo: {self.local_median_per_electrode.flatten()[:5]}... (primeiros 5)")
        offset_value = (self.global_mean_ * 5) + self.local_median_per_electrode
        print(f"- Soma (passo2 + passo3): {offset_value.flatten()[:5]}... (primeiros 5)")
        eegdata['X'] = offset_value - X_eegdata
        print(f"- Resultado final (passo4 - dados): shape {eegdata['X'].shape}, média {np.mean(eegdata['X']):.4f}")
        return eegdata 

    def fit_transform(self, eegdata:dict, **kwargs):
        return self.fit(eegdata).transform(eegdata)