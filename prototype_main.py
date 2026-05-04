# ==============================
# FINAL STABLE VERSION
# ==============================

import pandas as pd
import numpy as np
import os, re, torch, torch.nn as nn
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

print("Device:", torch.device('cpu'))

# ==============================
# LOAD DATA
# ==============================

tw = pd.read_csv('fake_profile_twitter.csv')
ig = pd.read_csv('Instagram_fake_profile_dataset.csv')
act = pd.read_csv('raw_user_activities 1(in).csv')

print("Twitter:", tw.shape)
print("Instagram:", ig.shape)
print("Activity:", act.shape)

# ==============================
# FIX INSTAGRAM COLUMNS
# ==============================

ig = ig.rename(columns={
    '#posts': 'statuses_count',
    '#followers': 'followers_count',
    '#follows': 'friends_count',
    'fake': 'is_fake'
})

# ==============================
# TABULAR MODEL
# ==============================

TAB_COLS = ['statuses_count','followers_count','friends_count']

for c in TAB_COLS:
    tw[c] = pd.to_numeric(tw[c], errors='coerce').fillna(0)
    ig[c] = pd.to_numeric(ig[c], errors='coerce').fillna(0)

# Simple fake label for twitter
tw['is_fake'] = (tw['followers_count'] < 5).astype(int)

X = pd.concat([tw[TAB_COLS], ig[TAB_COLS]], ignore_index=True)
y = pd.concat([tw['is_fake'], ig['is_fake']], ignore_index=True)

scaler = StandardScaler()
X = scaler.fit_transform(X)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier().fit(Xtr, ytr)
pred = rf.predict(Xte)

print("\nTabular Model:")
print(classification_report(yte, pred))

# ==============================
# TEXT MODEL (SAFE)
# ==============================

# Reduce dataset
texts = act['content'].fillna('').tolist()[:50000]
labels = act['is_fake'].map({True:1, False:0, 'TRUE':1, 'FALSE':0}).fillna(0).astype(int).tolist()[:50000]

print("\nUsing", len(texts), "texts")

# Tokenizer
def tokenize(t):
    return re.findall(r'\w+', t.lower())

# Build vocab
vocab = {'<PAD>':0, '<UNK>':1}
all_tokens = [tok for t in texts for tok in tokenize(t)]

for w, _ in Counter(all_tokens).most_common(5000):
    vocab[w] = len(vocab)

# Encode
def encode(text):
    ids = [vocab.get(t,1) for t in tokenize(text)][:50]
    return ids + [0]*(50-len(ids))

print("Encoding...")
X_text = []
for i, t in enumerate(texts):
    if i % 500 == 0:
        print(f"Encoding {i}/{len(texts)}")
    X_text.append(encode(t))

X_text = torch.tensor(X_text)
y_text = torch.tensor(labels, dtype=torch.float32)

# ==============================
# GRU MODEL
# ==============================

class GRUModel(nn.Module):
    def __init__(self, vsz):
        super().__init__()
        self.emb = nn.Embedding(vsz, 32)
        self.gru = nn.GRU(32, 64, batch_first=True)
        self.fc  = nn.Linear(64,1)

    def forward(self,x):
        _, h = self.gru(self.emb(x))
        return torch.sigmoid(self.fc(h[-1])).squeeze()

model = GRUModel(len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

# ==============================
# TRAINING (NO FREEZE)
# ==============================

print("\nTraining GRU...")

for ep in range(5):
    total = 0

    for i in range(0, len(X_text), 64):
        xb = X_text[i:i+64]
        yb = y_text[i:i+64]

        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()

        total += loss.item()

    print(f"Epoch {ep+1} Loss: {total:.4f}")

print("\nDONE ✅")