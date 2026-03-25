import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import confusion_matrix  # not using this rn but might need later
from tqdm import tqdm
import warnings
import random
import os
import json

#this is my dissertation project a malware detection framwork using BERT to do binary classifaction on malware and bengin api calls

warnings.filterwarnings('ignore')

RANDOM_SEED = 99999
# RANDOM_SEED = 42
# RANDOM_SEED = 123  # tried these didnt work as well

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using {device}")  # always forget to check this


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # these two lines are for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(RANDOM_SEED)


class MalwareDataset(Dataset):
    def __init__(self, sequences, labels, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # this took forever to debug
        tokens = [int(x) for x in str(self.sequences[idx]).split()]

        sequence = [1] + tokens  # 1 is CLS token this is redudent because im using mean pooling but leavbing it here in case it breaks something
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]

        seq_len = len(sequence)
        padding = self.max_length - seq_len
        sequence = sequence + [0] * padding  # 0 is padding
        attention_mask = [1] * seq_len + [0] * padding

        return {
            'input_ids': torch.tensor(sequence, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# the actual bert model
class DirectIntegerBERT(nn.Module):
    def __init__(self, vocab_size, num_labels=2):
        super().__init__()

        # using bert tiny cause bert base was way too slow
        # hidden_size=768, num_hidden_layers=12 for bert base but takes forever and there is no need for the small dataset
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512,
            max_position_embeddings=512,
            num_labels=num_labels,
            pad_token_id=0
        )


        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(128, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state

        # mean pooling average all tokens instead of just using CLS
        # took me like 3 days to get this working properly :(
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        pooled = torch.sum(token_embeddings * mask, dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def combine_sequences(row):
    # combine the raw apis with bigrams and trigrams
    parts = []
    for col in ['raw_api_sequence', 'bigram_sequence', 'trigram_sequence']:
        val = str(row[col]).strip()
        if val and val != 'nan':
            parts.append(val)
    return ' '.join(parts)


def make_loader(df, split, batch_size=8):
    data = df[df['split'] == split]
    dataset = MalwareDataset(data['combined'].tolist(), data['labels'].tolist())
    shuffle = True if split == 'train' else False
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


# training function
def train_model(model, train_loader, val_loader, epochs=50, lr=5e-5, seed=None): #tuned
    # dont apply weight decay to bias and layernorm
    # copied this from huggingface examples
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer = AdamW([
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01}, # tuned chnaged from 0.01
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ], lr=lr)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps) #tuned

    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    patience = 0


    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # check validation
        val_results = evaluate(model, val_loader)
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Val F1: {val_results['f1']:.4f}")



        # save best model
        if val_results['f1'] > best_f1:
            best_f1 = val_results['f1']
            patience = 0
            torch.save(model.state_dict(), 'best_model.pt')
            print("  new best, saved")
        else:
            patience += 1
            if patience >= 10:
                print("early stopping")
                break


def evaluate(model, loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].numpy())

    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)

    return {'acc': acc, 'precision': p, 'recall': r, 'f1': f1}


def test_model(model, loader):

    model.load_state_dict(torch.load('best_model.pt', map_location=device, weights_only=True)) # load the best tranined model
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Testing'):
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    # calc metrics
    acc = accuracy_score(all_labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)

    # auc needs both classes
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float('nan')  # happens when only one class in test set

    print()
    print("Test Results")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"AUC:       {auc:.4f}")

    return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1, 'roc_auc': auc}


def get_vocab_size(df):
    # find max token id in the dataset
    # theres probably a cleaner way to do this but whatever
    max_id = 0
    for seq in df['combined']:
        if pd.isna(seq):
            continue
        tokens = [int(x) for x in str(seq).split() if x]
        if tokens:
            max_id = max(max_id, max(tokens))
    return max_id + 1



#   MAIN
if __name__ == "__main__":
    print()
    print("loading.......... :)")
    print()
    print("1 - Train model")
    print("2 - Cross-dataset test")
    print()

    choice = input("Enter choice: ").strip()

    if choice == '1':
        # training mode
        print(f"\n diss traning model")
        print(f"\nSeed: {RANDOM_SEED}")


        df = pd.read_csv("MalBehavD-V1-processed-clean.csv")
        df['combined'] = df.apply(combine_sequences, axis=1)

        vocab_size = get_vocab_size(df)
        print(f"Samples: {len(df)}")
        print(f"Vocab size: {vocab_size}")

        # print class balance
        print(f"Classes: {df['labels'].value_counts().to_dict()}")

        # create model
        model = DirectIntegerBERT(vocab_size).to(device)

        # dataloaders
        train_loader = make_loader(df, 'train', batch_size=8) # tuned
        val_loader = make_loader(df, 'val', batch_size=8) #tuned
        test_loader = make_loader(df, 'test', batch_size=8) # tuned

        print(f"\nTrain size: {len(train_loader.dataset)}")
        print(f"Val size: {len(val_loader.dataset)}")
        print(f"Test size: {len(test_loader.dataset)}")
        print()

        # train
        train_model(model, train_loader, val_loader, seed=RANDOM_SEED)

        # test on held out set
        print("\nEvaluating on test set...")
        results = test_model(model, test_loader)

        # save
        out = pd.DataFrame([{
            'seed': RANDOM_SEED,
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'roc_auc': results['roc_auc']
        }])
        out.to_csv(f'{RANDOM_SEED}_processed.csv', index=False)
        print(f"\nSaved results to {RANDOM_SEED}_processed.csv")


    elif choice == '2':
        # cross dataset mode

        print(f"\n diss crossdata set test model")
        print("Testing model trained on MalBehavD-V1 against Mal-API-2019 with samples swapped between them\n")

        # load the cross dataset

        cross_df = pd.read_csv("MalAPI2019-test.csv")
        cross_df['combined'] = cross_df.apply(combine_sequences, axis=1)
        print(f"Samples: {len(cross_df)}")

        # need vocab from original training data
        train_df = pd.read_csv("MalBehavD-V1-mixed.csv")
        train_df['combined'] = train_df.apply(combine_sequences, axis=1)
        vocab_size = get_vocab_size(train_df)
        print(f"Vocab size: {vocab_size}")

        # create model and loader
        model = DirectIntegerBERT(vocab_size).to(device)

        loader = DataLoader(
            MalwareDataset(cross_df['combined'].tolist(), cross_df['labels'].tolist()),
            batch_size=8,
            shuffle=False
        )

        # test
        results = test_model(model, loader)

        # save
        out = pd.DataFrame([{
            'dataset': 'Mal-API-2019',
            'samples': len(cross_df),
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'roc_auc': results['roc_auc']
        }])
        out.to_csv('cross_dataset_results.csv', index=False)
        print(f"\nSaved to cross_dataset_results.csv")

    else:
        print("invalid choice >:(")