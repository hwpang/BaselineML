from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def make_rdkit_mol(smi):
    return Chem.MolFromSmiles(smi)


def get_fingerprint(mol, method="morgan", count_based=False, **kwargs):
    if method == "rdkit":
        fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(**kwargs)
    elif method == "morgan":
        fpgen = rdFingerprintGenerator.GetMorganGenerator(**kwargs)
    if count_based:
        fps = fpgen.GetCountFingerprintAsNumPy(mol)
    else:
        fps = fpgen.GetFingerprintAsNumPy(mol)
    return fps
