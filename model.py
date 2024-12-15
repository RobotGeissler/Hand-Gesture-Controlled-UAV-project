import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class SensorDataset(Dataset):
    def __init__(self, data_dir, time_window=1):
        self.data_dir = data_dir
        self.time_window = time_window
        self.sensor_order = ["MPU1", "MPU2"]

        # Map folder names to class labels
        # self.labels = {label: idx for idx, label in enumerate(os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, label)) else [])}
        self.files = [f for f in os.listdir(data_dir)]
        self.labels = {file: idx for idx, file in enumerate(self.files)}
        self.data = []

        # Process files and assign labels
        for label, idx in self.labels.items():
            folder_path = os.path.join(data_dir, label)
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]

            for file in files:
                data = pd.read_csv(file)

                # Create data points based on the time window
                n_points = len(data) - len(data) % 2  # Ensure divisible by 2 for MPU1 and MPU2

                if n_points >= self.time_window:
                    self.data.append((data.iloc[:n_points].values, idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_data = self.data[idx][0]
        label = self.data[idx][1]
        # Expand label to be a 1 hot vector
        label = np.eye(len(self.labels))[label]
        # Ensure sensor order is consistent
        order_consistent = np.array_equal(
            np.tile(np.array(self.sensor_order, dtype=object), input_data.shape[0] // 2),
            input_data[:, -1]
        )

        if not order_consistent:
            input_data = input_data[::-1]
        # Reshape and convert to tensor
        input_tensor = torch.from_numpy(
            input_data[:, :-1].reshape(-1, 2, input_data.shape[1] - 1)
            .reshape(-1, (input_data.shape[1] - 1) * 2)
            .astype(np.float32)
        )
        sequence_length = input_tensor.shape[0]
        return input_tensor, label, sequence_length

def pad_inputs(batch):
    inputs, labels, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)

    # Sort by sequence length (descending) for packing
    lengths, perm_idx = lengths.sort(0, descending=True)
    inputs = [inputs[i] for i in perm_idx]
    labels = torch.tensor([labels[i] for i in perm_idx])

    # Pad sequences
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    return inputs_padded, labels, lengths

# dataloader = DataLoader(SensorDataset('data/', time_window=7), batch_size=16, shuffle=True, collate_fn=pad_inputs)
# for x, y, lengths in dataloader:
#     print(x.shape, y.shape)
# exit()

from torch.utils.data import DataLoader, random_split

def split_dataset(dataset, val_percent=0.2, test_percent=0.1):
    val_size = int(len(dataset) * val_percent)
    test_size = int(len(dataset) * test_percent)
    train_size = len(dataset) - val_size - test_size
    return random_split(dataset, [train_size, val_size, test_size])

#Collate None 
def get_data_loaders(data_dir, batch_size=32, time_window=1, collate_fn=pad_inputs):
    dataset = SensorDataset(data_dir, time_window)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader, len(dataset.labels)

import torch.nn as nn
import pytorch_lightning as pl

class LSTMClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_sizes, num_classes, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.lstm = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.fc1 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc3 = nn.Linear(hidden_sizes[2], num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, lengths):
        # Pack the padded sequence
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
        packed_out, (hidden, _) = self.lstm(packed_x)
        
        # Use the last hidden state for classification
        x = hidden[-1]  # Hidden state of the last LSTM layer
        x = self.relu(self.fc1(x))  # Use the last time step
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y, lengths = batch
        y_pred = self(x, lengths)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, lengths = batch
        y_pred = self(x, lengths)
        loss = self.loss_fn(y_pred, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y, lengths = batch
        y_pred = self(x, lengths)
        loss = self.loss_fn(y_pred, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

import optuna
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

def objective(trial):
    # Hyperparameter suggestions
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    n_timesteps = trial.suggest_int("n_timesteps", 4, 10)
    hidden_sizes = [
        trial.suggest_int("hidden_size_1", 32, 256),
        trial.suggest_int("hidden_size_2", 32, 256),
        trial.suggest_int("hidden_size_3", 32, 256),
    ]

    # Dataloaders
    data_dir = "data/"
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(data_dir, time_window=n_timesteps)

    # Model
    model = LSTMClassifier(input_size=12, hidden_sizes=hidden_sizes, num_classes=num_classes, learning_rate=learning_rate)

    # Training
    logger = TensorBoardLogger("tb_logs", name="optuna_run5")
    trainer = Trainer(
        max_epochs=10,
        logger=logger,
        enable_checkpointing=False,
        val_check_interval=0.25  # Validate every 25% of an epoch
    )
    trainer.fit(model, train_loader, val_loader)

    # Return validation loss for optimization
    return trainer.callback_metrics["val_loss"].item()

train_loader = None
if (False):
    # Run optimization
    data_dir = "data/" 
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=1)

    print("Best hyperparameters:", study.best_params)
    print("Best validation loss:", study.best_value)
    best_params = study.best_params
    model = LSTMClassifier(
        input_size=12,
        hidden_sizes=[
            best_params["hidden_size_1"],
            best_params["hidden_size_2"],
            best_params["hidden_size_3"],
        ],
        num_classes=8,
        learning_rate=best_params["learning_rate"],
    )
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(data_dir, time_window=best_params["n_timesteps"])
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    # Save the model
    torch.save({
        "model_state_dict": model.state_dict(),
        "best_params": best_params
    }, "best_model_2.pth")

# get model params
checkpoint = torch.load("best_model_2.pth")
best_params = checkpoint["best_params"]

# load the model

model = LSTMClassifier(
    input_size=12,
    hidden_sizes=[
        best_params["hidden_size_1"],
        best_params["hidden_size_2"],
        best_params["hidden_size_3"],
    ],
    num_classes=8,
    learning_rate=best_params["learning_rate"],
)
model.load_state_dict(checkpoint["model_state_dict"])

# Test the model on the test set accuracy
from torchmetrics import Accuracy
if (train_loader is None):
    # for i in range(20):
    data_dir = "data/"
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(data_dir, time_window=best_params["n_timesteps"])
accuracy = Accuracy(task="multiclass", num_classes=num_classes)
model.eval()
for x, y, lengths in test_loader:
    y_pred = model(x, lengths)
    accuracy(y_pred, y)
# for x, y in test_loader:
#     y_pred = model(x)
#     accuracy(y_pred, y)
print("Test accuracy:", accuracy.compute())
