import numpy as np
from mordred import Calculator, descriptors

from baseline.utils import get_fingerprint


class MorganFeaturizer(object):
    def __init__(self, count_based=True, **kwargs):
        self.count_based = count_based
        self.kwargs = kwargs

    def __call__(self, mol):
        return get_fingerprint(
            mol, method="morgan", count_based=self.count_based, **self.kwargs
        )


class MordredFeaturizer(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, mol):
        calc = Calculator(descriptors)
        return np.array(calc(mol))


FEATURIZERS = {
    "morgan": MorganFeaturizer,
    "mordred": MordredFeaturizer,
}
