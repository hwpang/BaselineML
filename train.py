import argparse
import numpy as np
import torch
import pandas as pd

from baseline.data import MoleculeDataset, MoleculeDatapoint
from baseline.models import FeedForwardNetwork, train_one_epoch, evaluate, MODELS
from baseline.splits import split_data, SPLIT_METHODS
from baseline.utils import get_fingerprint, make_rdkit_mol
from baseline.featurizer import FEATURIZERS


def get_datapoints(featurizer):
    path = "/home/hwpang/Projects/BaselineML/data/biogen_solubility.csv"
    data_df = pd.read_csv(path)
    data_df["mol"] = data_df["SMILES"].apply(make_rdkit_mol)
    datapoints = [
        MoleculeDatapoint(
            mol,
            solubility,
            featurizer=featurizer,
        )
        for mol, solubility in zip(data_df["mol"], data_df["logS"])
    ]
    return datapoints


def add_args(parser):
    data_parser = parser.add_argument_group("Data args")
    data_parser.add_argument(
        "--featurizer",
        type=str,
        default="morgan",
        help="featurizer",
        choices=FEATURIZERS.keys(),
    )
    data_parser.add_argument(
        "--featurizer_kwargs",
        type=dict,
        default={},
        help="featurizer kwargs",
    )
    data_parser.add_argument(
        "--split_method",
        type=str,
        default="random",
        help="split method",
        choices=SPLIT_METHODS,
    )

    model_parser = parser.add_argument_group("Model args")
    model_parser.add_argument(
        "--model_architecture",
        type=str,
        default="ffn",
        help="model architecture",
        choices=MODELS.keys(),
    )
    model_parser.add_argument(
        "--model_kwargs",
        type=dict,
        default={},
        help="model kwargs",
    )

    train_parser = parser.add_argument_group("Train args")
    train_parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    train_parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    train_parser.add_argument("--hidden_size", type=int, default=128, help="hidden size")
    train_parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="learning rate"
    )
    train_parser.add_argument("--dropout", type=float, default=0.0, help="dropout")
    train_parser.add_argument(
        "--activation", type=str, default="relu", help="activation function"
    )
    train_parser.add_argument("--seed", type=int, default=0, help="random seed")


def make_model(model_architecture, model_kwargs):
    model = MODELS[model_architecture](**model_kwargs)
    return model


def train_save_pytorch_model(train_dset, val_dset, test_dset, args):
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dset, shuffle=False
    )

    # Create model
    model = make_model(args.model_architecture, args.model_kwargs)

    # Create loss function and optimizer
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train model
    for t in range(args.epochs):
        train_one_epoch(train_loader, model, loss_function, optimizer)
        train_loss = evaluate(train_loader, model, loss_function)
        print(f"Epoch {t+1}, Training loss: {train_loss:.4f}")
        val_loss = evaluate(val_loader, model, loss_function)
        print(f"Epoch {t+1}, Validation loss: {val_loss:.4f}")

    # Evaluate model
    test_loss = evaluate(test_loader, model, loss_function)
    print(f"Test loss: {test_loss:.4f}")

    # Save model
    model_dict = {
        "state_dict": model.state_dict(),
        "featurizer": args.featurizer,
        "feature_scaler": train_loader.dataset.feature_scaler,
        "label_scaler": train_loader.dataset.label_scaler,
    }
    torch.save(model_dict, "model.pt")


def main(args):

    # Set random seed
    torch.manual_seed(args.seed)

    # Load data
    datapoints = get_datapoints(featurizer=FEATURIZERS[args.featurizer](**args.featurizer_kwargs))
    train_mols, val_mols, test_mols = split_data(
        datapoints, method="random", split_sizes=(0.8, 0.1, 0.1), seed=args.seed
    )

    train_dset = MoleculeDataset(train_mols)
    val_dset = MoleculeDataset(val_mols)
    test_dset = MoleculeDataset(test_mols)

    feature_scaler = train_dset.normalize_features()
    val_dset.normalize_features(feature_scaler)
    test_dset.normalize_features(feature_scaler)

    label_scaler = train_dset.normalize_labels()
    val_dset.normalize_labels(label_scaler)
    test_dset.normalize_labels(label_scaler)

    if args.model_architecture == "ffn":
        train_save_pytorch_model(train_dset, val_dset, test_dset, args)
    else:
        raise ValueError(f"Model architecture {args.model_architecture} not recognized")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    main(args)
