from bciflow.datasets.cbcic import cbcic
from bciflow.modules.tf.filterbank import filterbank

dataset = cbcic(subject=1, path='data/cbcic/')

pre_folding = {'tf': (filterbank, {'kind_bp': 'chebyshevII'})}
