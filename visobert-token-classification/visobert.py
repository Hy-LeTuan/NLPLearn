import torcheval.metrics
import pandas as pd
from dataset import read_ner_file
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
from transformers import AdamW
from tqdm import tqdm
import torcheval
from transformers import AutoTokenizer, AutoModelForMaskedLM

# hyperparameters
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
epochs = 15
batch_size = 128

LABEL_2_ID = {'B-PATIENT_ID': 0,
              'I-PATIENT_ID': 1,
              'B-NAME': 2,
              'I-NAME': 3,
              'B-AGE': 4,
              'I-AGE': 5,
              'B-GENDER': 6,
              'I-GENDER': 7,
              'B-JOB': 8,
              'I-JOB': 9,
              'B-LOCATION': 10,
              'I-LOCATION': 11,
              'B-ORGANIZATION': 12,
              'I-ORGANIZATION': 13,
              'B-SYMPTOM_AND_DISEASE': 14,
              'I-SYMPTOM_AND_DISEASE': 15,
              'B-TRANSPORTATION': 16,
              'I-TRANSPORTATION': 17,
              'B-DATE': 18,
              'I-DATE': 19,
              'O': 20
              }

ID_2_LABEL = {0: 'B-PATIENT_ID',
              1: 'I-PATIENT_ID',
              2: 'B-NAME',
              3: 'I-NAME',
              4: 'B-AGE',
              5: 'I-AGE',
              6: 'B-GENDER',
              7: 'I-GENDER',
              8: 'B-JOB',
              9: 'I-JOB',
              10: 'B-LOCATION',
              11: 'I-LOCATION',
              12: 'B-ORGANIZATION',
              13: 'I-ORGANIZATION',
              14: 'B-SYMPTOM_AND_DISEASE',
              15: 'I-SYMPTOM_AND_DISEASE',
              16: 'B-TRANSPORTATION',
              17: 'I-TRANSPORTATION',
              18: 'B-DATE',
              19: 'I-DATE',
              20: 'O'
              }


# dataset
def converter(tokens):
    converted_tokens = []

    for token in tokens:
        converted_tokens.append(LABEL_2_ID[token])

    return converted_tokens


df_train = read_ner_file("./data/syllable/train_syllable.conll")
df_test = read_ner_file("./data/syllable/test_syllable.conll")

df_train = pd.DataFrame(data=df_train)
df_train = df_train.convert_dtypes()

df_test = pd.DataFrame(data=df_test)
df_test = df_test.convert_dtypes()

df_train["tokens"] = df_train["tokens"].apply(func=converter)
df_test["tokens"] = df_test["tokens"].apply(func=converter)

# tokenizers and alignment
tokenizer = AutoTokenizer.from_pretrained("uitnlp/visobert")


def align_tokens(tokens, df_type, label_all_tokens=True):
    labels = []
    for i, label in enumerate(df_type["tokens"]):
        word_ids = tokens.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            # set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokens["labels"] = labels
    return tokens


train_tokens = tokenizer(df_train["words"].to_list(
), truncation=True, padding=True, return_tensors="pt", is_split_into_words=True)
test_tokens = tokenizer(df_test["words"].to_list(
), truncation=True, padding=True, return_tensors="pt", is_split_into_words=True)

train_tokens = align_tokens(train_tokens, df_type=df_train)
test_tokens = align_tokens(test_tokens, df_type=df_test)

# models
model = AutoModelForMaskedLM.from_pretrained("uitnlp/visobert")
model.lm_head.decoder = nn.Linear(
    in_features=768, out_features=len(ID_2_LABEL), bias=True)
model = model.to(device)
for params in model.base_model.parameters():
    params.requires_grad = False

# dataset and dataloader


class VisoDataset(Dataset):
    def __init__(self, tokens: pd.Series, labels: list):
        self.labels = labels
        self.input_ids = tokens["input_ids"]
        self.attention_mask = tokens["attention_mask"]

        self.length = len(self.input_ids)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        input_id = self.input_ids[idx]
        attention_mask = self.attention_mask[idx]

        return {
            "labels": label,
            "input_ids": input_id,
            "attention_mask": attention_mask,
        }


train_dataset = VisoDataset(tokens=train_tokens, labels=train_tokens["labels"])
test_dataset = VisoDataset(tokens=test_tokens, labels=test_tokens["labels"])

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=16, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                         shuffle=True, num_workers=16, pin_memory=True)

# optimizers
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)

output_metrics = {
    "train": {
        "loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    },
    "eval": {
        "loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }
}


for i in tqdm(range(epochs), desc="Epochs", total=epochs):

    # training loop
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    epoch_precision = 0.0
    epoch_recall = 0.0
    epoch_f1 = 0.0
    total_train_steps = len(train_loader)
    total_test_steps = len(test_loader)
    model = model.train()

    for input_dict in tqdm(train_loader, desc=f"Training batches epoch {i}", total=total_train_steps):
        input_ids = input_dict["input_ids"].to(device)
        labels = input_dict["labels"].to(device)
        attention_mask = input_dict["attention_mask"].to(device)

        logits = model(input_ids, attention_mask)["logits"]

        labels = labels.view(-1)
        logits = logits.view(-1, 21)

        # compute loss
        loss = criterion(logits, labels)

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update running loss
        epoch_loss += loss.item()

        # compute metrics
        with torch.no_grad():
            valid_mask = labels != -100
            logits = logits[valid_mask]
            labels = labels[valid_mask]

            epoch_accuracy += torcheval.metrics.functional.multiclass_accuracy(
                input=logits, target=labels, average="micro").item()
            epoch_f1 += torcheval.metrics.functional.multiclass_f1_score(
                input=logits, target=labels, average="micro").item()
            epoch_precision += torcheval.metrics.functional.multiclass_precision(
                input=logits, target=labels, average="micro").item()
            epoch_recall += torcheval.metrics.functional.multiclass_recall(
                input=logits, target=labels, average="micro").item()

    output_metrics["train"]["loss"].append(epoch_loss / total_train_steps)
    output_metrics["train"]["accuracy"].append(
        epoch_accuracy / total_train_steps)
    output_metrics["train"]["precision"].append(
        epoch_precision / total_train_steps)
    output_metrics["train"]["recall"].append(epoch_recall / total_train_steps)
    output_metrics["train"]["f1"].append(epoch_f1 / total_train_steps)

    # evaluation loop
    model = model.eval()
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    epoch_precision = 0.0
    epoch_recall = 0.0
    epoch_f1 = 0.0
    total_train_steps = len(train_loader)

    with torch.no_grad():
        for input_dict in tqdm(test_loader, desc=f"Testing batches epoch {i}", total=total_test_steps):
            input_ids = input_dict["input_ids"].to(device)
            labels = input_dict["labels"].to(device)
            attention_mask = input_dict["attention_mask"].to(device)

            logits = model(input_ids, attention_mask)["logits"]

            labels = labels.view(-1)
            logits = logits.view(-1, 21)

            # compute loss
            loss = criterion(logits, labels)

            # update running loss
            epoch_loss += loss.item()

            # compute metrics
            valid_mask = labels != -100
            logits = logits[valid_mask]
            labels = labels[valid_mask]

            epoch_accuracy += torcheval.metrics.functional.multiclass_accuracy(
                input=logits, target=labels, average="micro").item()
            epoch_f1 += torcheval.metrics.functional.multiclass_f1_score(
                input=logits, target=labels, average="micro").item()
            epoch_precision += torcheval.metrics.functional.multiclass_precision(
                input=logits, target=labels, average="micro").item()
            epoch_recall += torcheval.metrics.functional.multiclass_recall(
                input=logits, target=labels, average="micro").item()

    output_metrics["eval"]["loss"].append(epoch_loss / total_test_steps)
    output_metrics["eval"]["accuracy"].append(
        epoch_accuracy / total_test_steps)
    output_metrics["eval"]["precision"].append(
        epoch_precision / total_test_steps)
    output_metrics["eval"]["recall"].append(epoch_recall / total_test_steps)
    output_metrics["eval"]["f1"].append(epoch_f1 / total_test_steps)

print(output_metrics)

torch.save(model.state_dict(), "./model/visobert_ner.pth")
