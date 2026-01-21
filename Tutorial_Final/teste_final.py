from bciflow.datasets.cbcic import cbcic
from bciflow.modules.core.kfold import kfold
from bciflow.modules.tf import filterbank
from bciflow.modules.fs import MIBIF
from bciflow.modules.sf.csp import csp
from bciflow.modules.fe.logpower import logpower
from bciflow.modules.analysis.metric_functions import accuracy
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

accDict = {
    'fbcsp_chebyshevII' :  [],
    'fbcsp_conv' :[],
    'teu' :[],}


dataset = cbcic
path = 'C:/Users/Hychiro/Documents/Ufjf/bci/testes no codigo do bciflow/Data/CBCIC'
maxSubjects = 11 #1 a 10
for key in accDict.keys():
    data = dataset(subject=1, path=path, labels=['left-hand', 'right-hand'])
    if key == 'fbcsp_chebyshevII':
        pre_folding = {'tf': (filterbank, {'kind_bp': 'chebyshevII'})}

        sf = csp()
        fe = logpower
        fs = MIBIF(8, clf=lda())
        clf = lda()

        pos_folding = {
            'sf': (sf, {}),
            'fe': (fe, {'flating': True}),
            'fs': (fs, {}),
            'clf': (clf, {})
        }
    if key == 'fbcsp_conv':
        pre_folding = {'tf': (filterbank, {'kind_bp': 'conv'})}

        sf = csp()
        fe = logpower
        fs = MIBIF(8, clf=lda())
        clf = lda()

        pos_folding = {
            'sf': (sf, {}),
            'fe': (fe, {'flating': True}),
            'fs': (fs, {}),
            'clf': (clf, {})
        }
    if key == 'teu':
        pass
    
    results = kfold(
                target=data,
                start_window=data['events']['cue'][0] + 0.5,
                window_size=2.0,
                pre_folding=pre_folding,
                pos_folding=pos_folding
            )
    
    df = pd.DataFrame(results)
    # Calcula a acurácia média do modelo a partir dos resultados de todos os folds
    acc = accuracy(df)

    # Imprime dataframe completo com os resultados detalhados de cada fold
    print(df)

    # Imprime a acurácia final calculada para o pipeline FBCSP, formatada com 4 casas decimais
    print(f"Accuracy para o metodo {key}: {acc:.4f}")