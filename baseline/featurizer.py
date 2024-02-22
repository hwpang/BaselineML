import numpy as np
from mordred import Calculator, descriptors

from baseline.utils import get_fingerprint


class MorganFingerprints(object):
    def __init__(self, count_based=True, **kwargs):
        self.count_based = count_based
        self.kwargs = kwargs

    def __call__(self, mol):
        return get_fingerprint(
            mol, method="morgan", count_based=self.count_based, **self.kwargs
        )


class MordredDescriptors(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, mol):
        calc = Calculator(descriptors)
        return np.array(calc(mol))


FEATURIZERS = {
    "morgan": MorganFingerprints,
    "mordred": MordredDescriptors,
}
