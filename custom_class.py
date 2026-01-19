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
        offset_value = (self.global_mean_ * 5) + self.local_median_per_electrode
        eegdata['X'] = offset_value - X_eegdata
        eegdata_transformed = eegdata['X']

        return eegdata_transformed

    def fit_transform(self, eegdata:dict, **kwargs):
        return self.fit(eegdata).transform(eegdata)