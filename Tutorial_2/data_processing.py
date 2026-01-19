from bciflow.datasets.cbcic import cbcic
# banco de filtros temporais aos sinais EEG
from bciflow.modules.tf.filterbank import filterbank

# realiza o algoritmo Common Spatial Patterns (filtros espaciais)
from bciflow.modules.sf.csp import csp

# extrai a potência logarítmica como característica dos dados
from bciflow.modules.fe.logpower import logpower

# realiza a seleção de características baseada em informação mútua
from bciflow.modules.fs.mibif import MIBIF

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

# executa a validação cruzada k-fold do pipeline
from bciflow.modules.core.kfold import kfold

import pandas as pd
#  calcula a acurácia de classificação dos resultados
from bciflow.modules.analysis.metric_functions import accuracy

dataset = cbcic(subject=1, path='data/cbcic/')

# Define a etapa de pré-processamento para aplicar um banco de filtros
pre_folding = {'tf': (filterbank, {'kind_bp': 'chebyshevII'})}

# Instancia os módulos de processamento espacial (CSP),
#  extração de características (logpower), seleção de características (MIBIF) e o classificador (LDA)
sf = csp()
fe = logpower
fs = MIBIF(8, clf=lda())
clf = lda()

# Organiza as etapas de processamento que serão aplicadas dentro de cada fold de validação cruzada
pos_folding = {
    'sf': (sf, {}),
    'fe': (fe, {}),
    'fs': (fs, {}),
    'clf': (clf, {})
}

# Executa o pipeline completo 
# (pré-processamento e processamento) usando validação cruzada k-fold no dataset
results = kfold(
    target=dataset,
    start_window=dataset['events']['cue'][0] + 0.5,
    pre_folding=pre_folding,
    pos_folding=pos_folding
)

# Converte os resultados detalhados de cada fold em um DataFrame do pandas para facilitar a análise
df = pd.DataFrame(results)
# Calcula a acurácia média do modelo a partir dos resultados de todos os folds
acc = accuracy(df)

# Imprime dataframe completo com os resultados detalhados de cada fold
print(df)

# Imprime a acurácia final calculada para o pipeline FBCSP, formatada com 4 casas decimais
print(f"Accuracy: {acc:.4f}")