import torch
import numpy as np

ACTIVATION = {
    "relu": torch.nn.ReLU,
    "leaky_relu": torch.nn.LeakyReLU,
    "prelu": torch.nn.PReLU,
    "elu": torch.nn.ELU,
    "tanh": torch.nn.Tanh,
}


class FeedForwardNetwork(torch.nn.Module):
    """
    A simple feedforward neural network for regression.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        n_layers=2,
        dropout=0.0,
        activation="relu",
        bias=True,
    ):
        super(FeedForwardNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.activation = ACTIVATION[activation.lower()]()
        self.input_scaler = None
        self.output_scaler = None

        sequences = []
        sequences.append(torch.nn.Linear(input_size, hidden_size, bias=bias))
        sequences.append(self.activation)
        sequences.append(torch.nn.Dropout(dropout))
        for i in range(n_layers - 2):
            sequences.append(torch.nn.Linear(hidden_size, hidden_size, bias=bias))
            sequences.append(self.activation)
            sequences.append(torch.nn.Dropout(dropout))
        sequences.append(torch.nn.Linear(hidden_size, output_size, bias=bias))

        self.sequences = torch.nn.Sequential(*sequences)

    def forward(self, x):
        x = self.sequences(x)
        return x


def train_one_epoch(dataloader, model, loss_function, optimizer, log_interval=100):
    size = len(dataloader.dataset)

    model.train()

    for batch, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()

        y_pred = model(x)
        loss = loss_function(y_pred, y)

        loss.backward()
        optimizer.step()

        # if batch % log_interval == 0:
        #     train_loss, current = loss.item(), batch * len(x)
        #     print(f"Training loss: {train_loss:>7f}  [{current:>5d}/{size:>5d}]")


def evaluate(dataloader, model, loss_function):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for x, y in dataloader:
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            test_loss += loss.item()

    test_loss /= num_batches
    return test_loss

MODELS = {
    "ffn": FeedForwardNetwork,
}