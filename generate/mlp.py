# Developed as part of a BSc thesis at the Faculty of Computer Science, Bialystok Univesity of Technology

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from thesis.utils.utils import set_seed

set_seed(142)

target_column = "Label"
redundant_columns = [
    "Gender",
    "Non-ah-mmh speech disturbances",
    "Word and phrase repetitions",
    "Silent pauses",
    "Loudness",
    "Tension",
    "Participants name",
]

df = pd.read_excel("thesis/data/traits.xlsx")
df = df.set_index(df.columns[0])

for split_id in range(1, 4):
    train_fold = pd.read_csv(f"thesis/data/train_fold{split_id}.csv", header=None)
    train_ids = train_fold[0].tolist()

    val_fold = pd.read_csv(f"thesis/data/val_fold{split_id}.csv", header=None)
    val_ids = val_fold[0].tolist()

    test_fold = pd.read_csv(f"thesis/data/test_fold{split_id}.csv", header=None)
    test_ids = test_fold[0].tolist()

    train_df = df.loc[train_ids]
    val_df = df.loc[val_ids]
    test_df = df.loc[test_ids]

    X_train = train_df.drop(columns=[target_column] + redundant_columns).to_numpy()
    X_val = val_df.drop(columns=[target_column] + redundant_columns).to_numpy()
    X_test = test_df.drop(columns=[target_column] + redundant_columns).to_numpy()

    y_train = (train_df[target_column] == "deception").astype(float).to_numpy()
    y_val = (val_df[target_column] == "deception").astype(float).to_numpy()
    y_test = (test_df[target_column] == "deception").astype(float).to_numpy()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    input_size = X_train.shape[1]
    hidden_size = 256
    dropout_rate = 0.3

    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
        nn.Linear(hidden_size, 1),
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 30
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_train_logits = model(X_train)
            y_train_pred = torch.sigmoid(y_train_logits).round()
            train_acc = accuracy_score(y_train.numpy(), y_train_pred.numpy())

            y_val_logits = model(X_val)
            y_val_pred = torch.sigmoid(y_val_logits).round()
            val_acc = accuracy_score(y_val.numpy(), y_val_pred.numpy())

            y_test_logits = model(X_test)
            y_test_pred = torch.sigmoid(y_test_logits).round()
            test_acc = accuracy_score(y_test.numpy(), y_test_pred.numpy())

            TP = np.sum((y_test.numpy() == 1) & (y_test_pred.numpy() == 1))
            TN = np.sum((y_test.numpy() == 0) & (y_test_pred.numpy() == 0))
            FP = np.sum((y_test.numpy() == 0) & (y_test_pred.numpy() == 1))
            FN = np.sum((y_test.numpy() == 1) & (y_test_pred.numpy() == 0))

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"TP: {TP} | TN: {TN} | FP: {FP} | FN: {FN}"
        )
