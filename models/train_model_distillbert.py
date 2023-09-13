import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm_notebook as tqdm
import time
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, DistilBertForSequenceClassification
import evaluate

parser = argparse.ArgumentParser(description="Train a sequence classification model DistilBERT")
parser.add_argument("--train-file", type=str, required=True, help="Path to the train tab-delimited .csv file")
args = parser.parse_args()

RANDOM_STATE = 1052023
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_colwidth', 280)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

df = pd.read_csv(args.train_file, delimiter='\t')

train_df, val_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

# Tokenize data
train_encodings = tokenizer(list(train_df["title"]), truncation=True, padding=True)
val_encodings = tokenizer(list(val_df["title"]), truncation=True, padding=True)

# Prepare labels
train_labels = train_df["is_fake"].values
val_labels = val_df["is_fake"].values

# Create PyTorch datasets
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings["input_ids"]),
    torch.tensor(train_encodings["attention_mask"]),
    torch.tensor(train_labels)
)

val_dataset = torch.utils.data.TensorDataset(
    torch.tensor(val_encodings["input_ids"]),
    torch.tensor(val_encodings["attention_mask"]),
    torch.tensor(val_labels)
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

# Define model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 35
model.train()
train_losses = []
val_losses = []
val_predictions = []
val_labels = []

for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0

    # Training phase
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)  # Calculate the loss using criterion
            val_loss += loss.item()

            # Calculate validation metrics
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            val_predictions.extend(predictions)
            val_labels.extend(labels.cpu().numpy())

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Compute F1 score and AUC-ROC
    f1 = f1_score(val_labels, val_predictions)
    auc_roc = roc_auc_score(val_labels, val_predictions)

    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print(f"F1 Score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}")

# Plot the loss curves
epochs = range(1, num_epochs + 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Create the "reports" directory if it doesn't exist
os.makedirs("reports", exist_ok=True)

# Save the plot
plt.savefig("../reports/loss_user_transformer_plot.png")
plt.close()

# Save the trained model
model.save_pretrained("models/distillbert_model_user")
