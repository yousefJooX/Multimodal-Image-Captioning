# 🖼️ Image Captioning — ResNet50 Encoder Project

## 📌 Project Overview

This project trains a **ResNet50-based image encoder** on interior room images (Houzz dataset) and exports it so that a collaborator can build an **LSTM decoder from scratch** on top of it.

```
Image → ResNet50 Encoder (this notebook) → Feature Vector [256-dim] → Friend's LSTM Decoder → Caption
```

The work is split into two parts:
- **Your part (this folder):** Train and export the encoder
- **Friend's part:** Load the encoder and train an LSTM decoder to generate captions

---

## 🗂️ File Guide

### 🔵 `best_model.pth`
**What it is:** The complete training checkpoint — saved automatically during training at the epoch with the lowest validation loss.

**What's inside:**
- `encoder` — ResNet50 encoder weights
- `decoder` — GRU decoder weights (temporary decoder used during training)
- `optimizer` — optimizer state (for resuming training)
- `epoch` — which epoch this was saved at
- `best_val_loss` — the validation loss at that point
- `vocab` — full vocabulary (word2idx, idx2word, vocab_size, special token indices)
- `embed_dim` — embedding dimension (256)

**When to use it:**
- If you want to **resume training** from where it stopped
- If you want to **reload your own full model** for inference
- For your own use only — do NOT send this to your friend (it contains the GRU decoder which she won't use)

---

### 🟢 `encoder_for_lstm.pth`
**What it is:** The encoder exported separately — everything your friend needs and nothing she doesn't.

**What's inside:**
| Key | Value | Why it matters |
|-----|-------|----------------|
| `encoder_state_dict` | ResNet50 weights | The trained encoder itself |
| `embed_dim` | 256 | Output size of encoder = input size of her LSTM |
| `vocab_size` | depends on dataset | She needs this to build her Embedding and FC layers |
| `pad_idx` / `start_idx` / `end_idx` | special token IDs | Required for training and generation |
| `word2idx` | dict word→id | She uses this to tokenize captions |
| `idx2word` | dict id→word | She uses this to decode generated token IDs back to words |
| `img_mean` / `img_std` | ImageNet values | She must apply the same normalization on images |
| `max_seq_len` | 40 | Maximum caption length |
| `best_val_loss` | float | Reference metric |

**When to use it:**
- ✅ Send this file to your friend — this is the handoff file
- ✅ She loads this to start building her LSTM decoder

---

### 🟡 `eval_metrics.json`
**What it is:** A summary of all evaluation scores measured on the **test set** after training.

**What's inside:**
```json
{
  "model": "Best ResNet50 (layer4 fine-tuned) + GRU",
  "best_val_loss": ...,
  "test_samples": ...,
  "BLEU-1": ...,
  "BLEU-4": ...,
  "METEOR": ...,
  "ROUGE-L": ...,
  "CIDEr": ...,
  "Room_Acc_%": ...,
  "hyperparams": { ... }
}
```

**Metrics explained:**
| Metric | What it measures |
|--------|-----------------|
| BLEU-1 | How many individual words match the reference caption |
| BLEU-4 | How many 4-word sequences match (stricter, more meaningful) |
| METEOR | Word overlap + synonyms + stemming — more flexible than BLEU |
| ROUGE-L | Longest common subsequence between generated and real caption |
| CIDEr | Consensus-based — rewards captions that match what humans would write |
| Room Acc % | Whether the generated caption mentions the correct room type |

**When to use it:**
- Include in your project report as your baseline encoder results
- Your friend can compare her LSTM results against these numbers

---

### 🟠 `test_predictions.csv`
**What it is:** The generated captions for every image in the test set, side by side with the ground truth.

**Columns:**
| Column | Description |
|--------|-------------|
| `local_path` | Path to the image file |
| `room_type` | The true room category (kitchen, bathroom, etc.) |
| `true_caption` | The original human-written caption |
| `generated` | The caption your model generated |

**When to use it:**
- Qualitative analysis — look at where the model does well and where it fails
- Include example predictions in your report
- Your friend can use it as a reference baseline to compare against her LSTM outputs

---

### 📓 `model_best_encoder.ipynb`
**What it is:** The full training notebook.

**What it does, step by step:**

| Section | What happens |
|---------|-------------|
| 1. Install & Imports | Installs NLTK, rouge-score, pycocoevalcap for evaluation |
| 2. Config & Device | Sets all hyperparameters, detects GPUs, enables DataParallel |
| 3. Vocabulary & Data | Loads vocab.pkl and train/val/test CSVs |
| 4. Dataset & DataLoader | Applies strong augmentation on train, clean transform on val/test |
| 5. Model Architecture | Defines ResNet50 encoder + GRU decoder |
| 6. Loss / Optimizer | CrossEntropyLoss + label smoothing, AdamW, ReduceLROnPlateau |
| 7. Smoke Test | Verifies shapes before training starts |
| 8. Training Loop | 50 epochs max, early stopping, saves best checkpoint |
| 9. Plot | Loss and accuracy curves |
| 10. Generate Captions | Greedy decoding on sample images |
| 11. Evaluation | BLEU, METEOR, ROUGE-L, CIDEr, Room Accuracy |
| 12. Export Encoder | Saves `encoder_for_lstm.pth` |
| 13. Verify Export | Reloads encoder from scratch to confirm it works |

---

## 👩‍💻 What Your Friend Needs to Do (LSTM Part)

She loads your exported encoder and builds her decoder on top of it:

```python
import torch
import torch.nn as nn

# 1. Load your encoder checkpoint
ckpt = torch.load('encoder_for_lstm.pth')

# 2. Rebuild the encoder class (she needs the EncoderResNet50 class definition)
encoder = EncoderResNet50(embed_dim=ckpt['embed_dim'], fine_tune=False, dropout=0.0)
encoder.load_state_dict(ckpt['encoder_state_dict'])
encoder.eval()

# Freeze encoder — she only trains the LSTM
for p in encoder.parameters():
    p.requires_grad = False

# 3. She builds her LSTM decoder from scratch
class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=1):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_fc = nn.Linear(embed_dim, hidden_dim)  # encoder output → h0
        self.cell_fc   = nn.Linear(embed_dim, hidden_dim)  # encoder output → c0
        self.fc_out    = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        h0 = self.hidden_fc(features).unsqueeze(0)   # (1, B, hidden)
        c0 = self.cell_fc(features).unsqueeze(0)     # (1, B, hidden)
        emb = self.embed(captions[:, :-1])            # (B, T-1, embed)
        out, _ = self.lstm(emb, (h0, c0))             # (B, T-1, hidden)
        return self.fc_out(out)                       # (B, T-1, vocab)

# 4. She uses the same image normalization as your training
from torchvision import transforms
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=ckpt['img_mean'], std=ckpt['img_std']),
])

# 5. She uses your vocab directly — no need to rebuild it
word2idx = ckpt['word2idx']
idx2word = ckpt['idx2word']
```

> ⚠️ **Important:** She must use the **same `word2idx` and `img_mean/std`** from your checkpoint. If she builds a new vocab or uses different normalization, the encoder output will not align with her captions.

---

## ⚙️ Key Hyperparameters Used

| Parameter | Value | Notes |
|-----------|-------|-------|
| `EMBED_DIM` | 256 | Encoder output = LSTM input size |
| `HIDDEN_DIM` | 512 | GRU hidden size (training only) |
| `BATCH_SIZE` | 32 × num_GPUs | Scales automatically |
| `LR` | 5e-5 | Low LR for stable fine-tuning |
| `NUM_EPOCHS` | 50 | With early stopping (patience=7) |
| `DROPOUT` | 0.3 | Applied in projection head and decoder |
| `LABEL_SMOOTHING` | 0.1 | Prevents overconfidence |
| `GRAD_CLIP` | 1.0 | Prevents exploding gradients |
| `FINE_TUNE` | True | Only `layer4` of ResNet50 is unfrozen |

---

## 📁 Folder Structure

```
project/
│
├── model_best_encoder.ipynb   ← Full training notebook
│
├── best_model.pth             ← Full checkpoint (encoder + GRU + optimizer)
├── encoder_for_lstm.pth       ← Encoder only — send this to your friend ✅
│
├── eval_metrics.json          ← BLEU / METEOR / ROUGE-L / CIDEr scores
├── test_predictions.csv       ← Generated vs real captions on test set
│
├── training_history.json      ← Loss & accuracy per epoch (raw numbers)
└── training_history.png       ← Loss & accuracy curves plot
```
