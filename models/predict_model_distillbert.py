import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

import warnings
warnings.filterwarnings('ignore')

os.makedirs("reports", exist_ok=True)

parser = argparse.ArgumentParser(description="Predict fake news using DistilBERT")
parser.add_argument("--test-file", type=str, required=True, help="Path to the tab-delimited test file")
args = parser.parse_args()

test_df = pd.read_csv(args.test_file, delimiter='\t')

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Encode test data
test_encodings = tokenizer.batch_encode_plus(
    test_df['title'].tolist(),
    truncation=True,
    padding=True,
    return_tensors='pt'
)

# TensorDataset for test data
test_dataset = TensorDataset(
    test_encodings['input_ids'],
    test_encodings['attention_mask']
)

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#trained model
model = DistilBertForSequenceClassification.from_pretrained("models/distillbert_weights")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()

predictions = []
probabilities = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities_batch = torch.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probabilities_batch, dim=1)

        predictions.extend(predicted_labels.cpu().tolist())
        probabilities.extend(probabilities_batch[:, 1].cpu().tolist())

predictions_df = pd.DataFrame({'title': test_df['title'], 'is_fake': predictions, 'prob_fake': probabilities})

predictions_df.to_csv('../reports/predictions_distillbert.csv', index=False)
