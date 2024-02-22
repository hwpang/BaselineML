import numpy as np
from sklearn.preprocessing import StandardScaler
import torch


class MoleculeDatapoint:
    """
    A datapoint for molecular data.
    """

    def __init__(self, mol, label, featurizer):
        self.mol = mol
        self.label = label
        self.featurizer = featurizer

        self.features = featurizer(mol)


class MoleculeDataset(torch.utils.data.Dataset):
    """
    A dataset for molecular data.
    """

    def __init__(self, datapoints):
        self.datapoints = datapoints
        self.mols = [d.mol for d in datapoints]

        labels = np.array([d.label for d in datapoints]).reshape(-1, 1)
        labels = torch.from_numpy(labels).float()
        features = np.vstack([d.features for d in datapoints]).astype(np.float32)
        features = torch.from_numpy(features).float()

        self.labels = labels
        self.features = features
        self.unscaled_features = self.features
        self.unscaled_labels = self.labels

        self.feature_scaler = None
        self.label_scaler = None

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y

    def normalize_features(self, scaler=None):
        """
        Normalize features.
        """
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(self.unscaled_features)
        self.features = scaler.transform(self.unscaled_features)
        self.features = torch.from_numpy(self.features).float()
        self.feature_scaler = scaler
        return scaler

    def normalize_labels(self, scaler=None):
        """
        Normalize labels.
        """
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(self.unscaled_labels)
        self.labels = scaler.transform(self.unscaled_labels)
        self.labels = torch.from_numpy(self.labels).float()
        self.label_scaler = scaler
        return scaler

    def to(self, device):
        """
        Send dataset to device.
        """
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)
