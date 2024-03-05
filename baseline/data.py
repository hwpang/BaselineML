import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
            scaler.fit(self.features)
        self.features = scaler.transform(self.features)
        self.features = torch.from_numpy(self.features).float()
        return scaler
    
    def reduce_features(self, pca=None, n_components=30):
        """
        Reduce features using PCA.
        """
        if pca is None:
            pca = PCA(n_components=n_components)
            pca.fit(self.features)
        self.features = pca.transform(self.features)
        self.features = torch.from_numpy(self.features).float()
        return pca

    def normalize_labels(self, scaler=None):
        """
        Normalize labels.
        """
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(self.unscaled_labels)
        self.labels = scaler.transform(self.unscaled_labels)
        self.labels = torch.from_numpy(self.labels).float()
        return scaler
    
    def reset_features(self):
        """
        Reset features to unscaled features.
        """
        self.features = self.unscaled_features

    def reset_labels(self):
        """
        Reset labels to unscaled labels.
        """
        self.labels = self.unscaled_labels

    def reset(self):
        """
        Reset features and labels to unscaled features and labels.
        """
        self.reset_features()
        self.reset_labels()

    def to(self, device):
        """
        Send dataset to device.
        """
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)

def filter_features(datasets):
    """
    Filter NaN features.
    """
    nan_indices = torch.tensor([False] * datasets[0].features.shape[1])
    for dataset in datasets:
        nan_indices = torch.logical_or(nan_indices, torch.any(torch.isnan(dataset.features), dim=0))
    for dataset in datasets:
        features = dataset.features[:, ~nan_indices]
        dataset.features = None
        dataset.features = features
    return datasets
