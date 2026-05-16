# 🏠 Real Estate Image Captioning — LSTM Decoder v2

An attention-enhanced LSTM decoder for generating natural-language captions of real estate interior photographs. The model takes an image of a room (bathroom, bedroom, kitchen, living room, home office, or pool) and produces a descriptive caption.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
  - [Encoder](#encoder)
  - [Decoder (LSTMDecoderV2)](#decoder-lstmdecoderv2)
- [Key Techniques](#key-techniques)
- [Hyperparameters](#hyperparameters)
- [Dataset](#dataset)
- [File Structure](#file-structure)
- [How to Run](#how-to-run)
- [Evaluation Metrics](#evaluation-metrics)
- [Outputs](#outputs)

---

## Project Overview

This project implements an **encoder–decoder** image captioning pipeline for real estate photos:

| Component | Details |
|-----------|---------|
| **Task** | Generate descriptive captions for real estate room images |
| **Encoder** | ResNet-50 (ImageNet V2 weights, frozen) — provided by teammate |
| **Decoder** | 2-layer LSTM with feature gating, scheduled sampling, and weight tying |
| **Room Types** | Bathroom, Bedroom, Home Office, Kitchen, Living Room, Pool |
| **Platform** | Kaggle (T4 GPU) |
| **Framework** | PyTorch |

---

## Architecture

### Encoder

A frozen **ResNet-50** backbone (ImageNet V2 pre-trained) with a custom projection head:

```
ResNet-50 backbone (frozen)
    └── 2048-d feature vector
         └── Linear(2048 → 1024) + ReLU + Dropout
              └── Linear(1024 → 256) + LayerNorm
                   └── 256-d image embedding
```

The encoder is loaded from a pre-trained checkpoint (`encoder_for_lstm.pth`) and kept completely frozen during decoder training.

### Decoder (LSTMDecoderV2)

An improved LSTM decoder that addresses the underfitting issues from v1:

```
Image features (256-d)
    ├── Initialize h₀, c₀ via learned linear projections
    │
    └── At each timestep t:
         ├── Word Embedding (256-d)
         ├── Feature Gate: σ(W·[h_top; word_embed]) ⊙ image_features
         ├── LSTM input = [word_embed ∥ gated_features] (512-d)
         ├── 2-layer LSTM (hidden=512)
         ├── Dropout (0.4)
         ├── Output projection (512 → 256) → shared with embedding weights
         └── Logits (vocab_size)
```

**Key improvements over v1:**

| Feature | v1 | v2 |
|---------|----|----|
| LSTM layers | 1 | 2 |
| Learning rate | 5e-5 | 3e-3 (60× higher) |
| Feature gating | ✗ | ✓ (learned sigmoid gate) |
| Scheduled sampling | ✗ | ✓ (linear decay from 100% → 70% teacher forcing) |
| Weight tying | ✗ | ✓ (embedding ↔ output FC share weights) |
| LSTM initialization | Default | Orthogonal recurrent + Xavier input + forget bias = 1 |
| Gradient clipping | 1.0 | 5.0 |
| Label smoothing | ✗ | ✓ (0.1) |
| LR scheduler | ReduceLROnPlateau | OneCycleLR (cosine annealing) |

---

## Key Techniques

### Feature Gating
At each timestep, a learned sigmoid gate modulates which dimensions of the image feature vector are relevant, allowing the decoder to focus on different visual aspects as it generates different parts of the caption.

### Scheduled Sampling
During training, the model gradually transitions from using ground-truth tokens (teacher forcing) to its own predictions as input. This bridges the train–inference gap:
- **Epoch 0**: 100% teacher forcing (SS probability = 0.0)
- **Final epoch**: 70% teacher forcing (SS probability = 0.3)

### Weight Tying
The word embedding matrix and the output projection layer share the same weight matrix. Since `embed_dim ≠ hidden_dim` (256 vs 512), an intermediate linear projection (`out_proj`) maps from hidden space to embedding space before the tied output layer.

### OneCycleLR Scheduling
A warm-up phase (10% of training) ramps the LR from `3e-3 / 25` up to `3e-3`, followed by cosine annealing down to near zero. This is more effective than plateau-based schedulers for training from scratch.

### Decoding Strategies
Two inference methods are implemented:
- **Greedy decoding**: Pick the highest-probability token at each step (fast)
- **Beam search** (beam=5): Maintain top-k hypotheses with length normalization (α=0.7) for higher-quality captions

---

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `EMBED_DIM` | 256 | Word embedding & encoder output dimension |
| `HIDDEN_DIM` | 512 | LSTM hidden state size |
| `NUM_LAYERS` | 2 | Stacked LSTM layers |
| `DROPOUT` | 0.4 | Applied between LSTM layers and before output |
| `BATCH_SIZE` | 32 | |
| `NUM_EPOCHS` | 50 | With early stopping |
| `LR` | 3e-3 | Peak learning rate (OneCycleLR) |
| `GRAD_CLIP` | 5.0 | Max gradient norm |
| `MAX_SEQ_LEN` | 40 | Maximum caption length in tokens |
| `PATIENCE` | 10 | Early stopping patience (epochs) |
| `LABEL_SMOOTH` | 0.1 | Cross-entropy label smoothing |
| `WEIGHT_DECAY` | 1e-4 | AdamW weight decay |
| `SS_START` | 1.0 | Initial teacher forcing ratio |
| `SS_END` | 0.7 | Final teacher forcing ratio |

---

## Dataset

The dataset consists of real estate images with corresponding captions, split into train/val/test sets via CSV files:

| Split | File |
|-------|------|
| Train | `split_train.csv` |
| Val | `split_val.csv` |
| Test | `split_test.csv` |

Each CSV contains columns: `local_path`, `tokens`, `caption`, `room_type`.

**Data augmentation** (train only):
- Resize to 256×256 → RandomCrop to 224×224
- Random horizontal flip (p=0.5)
- Color jitter (brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)
- Random grayscale (p=0.05)
- Normalize with dataset-specific mean/std

**Validation/Test transforms**:
- Resize to 224×224
- Normalize with dataset-specific mean/std

---

## File Structure

```
lstm model/
│
├── README.md                  ← This file
│
├── lstm_decoder_v2.py         ← Decoder architecture (LSTMDecoderV2 class)
│                                 Sections 1–9: imports, hyperparams, data loading,
│                                 encoder, and the full decoder definition
│
├── lstm_training_v2.py        ← Training loop
│                                 Sections 10–13: loss/optimizer/scheduler setup,
│                                 training loop with scheduled sampling,
│                                 training curve plots, and best model loading
│
├── lstm_evaluation_v2.py      ← Evaluation & inference
│                                 Sections 14–20: greedy & beam search decoding,
│                                 CIDEr implementation, qualitative examples,
│                                 full metric evaluation (BLEU, METEOR, ROUGE-L,
│                                 CIDEr), per-room accuracy analysis, and export
│
└── lstm_complete_v2.py        ← All-in-one notebook script
                                  Complete pipeline (sections 1–20) in a single
                                  file, designed to run end-to-end on Kaggle
```

### Modular Breakdown

The project is split into **three focused modules** for clarity, plus one all-in-one script:

| Module | Sections | Responsibility |
|--------|----------|----------------|
| `lstm_decoder_v2.py` | 1–9 | Environment setup, data pipeline, encoder loading, decoder architecture |
| `lstm_training_v2.py` | 10–13 | Optimizer, scheduler, training loop, curve visualization |
| `lstm_evaluation_v2.py` | 14–20 | Inference functions, metrics computation, room-type analysis, export |
| `lstm_complete_v2.py` | 1–20 | All of the above in one file (for Kaggle notebook execution) |

---

## How to Run

### On Kaggle (recommended)

1. Upload `lstm_complete_v2.py` as a Kaggle notebook (or copy cell by cell — sections are marked with `# %%`)
2. Attach the required datasets:
   - **Encoder checkpoint**: `duhamostafa/encoder-path` → contains `encoder_for_lstm.pth`
   - **Image + caption data**: `sarahsayed15102005/image-caption` → contains CSVs and images
3. Enable **GPU (T4)** accelerator
4. Run all cells

### Locally (modular)

```bash
# Install dependencies
pip install torch torchvision pandas numpy matplotlib nltk rouge-score

# Run in order:
python lstm_decoder_v2.py     # Loads data + builds encoder/decoder
python lstm_training_v2.py    # Trains the decoder
python lstm_evaluation_v2.py  # Evaluates and exports
```

> **Note**: Local execution requires updating the file paths at the top of each script to point to your local data directories.

---

## Evaluation Metrics

The model is evaluated using standard image captioning metrics:

| Metric | Description |
|--------|-------------|
| **BLEU-1** | Unigram precision (measures word overlap) |
| **BLEU-4** | 4-gram precision (measures fluency) |
| **METEOR** | Harmonic mean of precision/recall with synonym matching |
| **ROUGE-L** | Longest common subsequence F-measure |
| **CIDEr** | TF-IDF weighted n-gram similarity (consensus-based) |
| **Room Accuracy** | % of captions containing a synonym for the correct room type |

### Room-Type Classification

A synonym-based matching system checks whether generated captions correctly identify the room type:

| Room Type | Synonyms Checked |
|-----------|-----------------|
| Bathroom | bathroom, bath, shower, tub, vanity, toilet, sink, faucet, tile |
| Bedroom | bedroom, bed, mattress, pillow, nightstand, headboard, sleeping, duvet |
| Home Office | office, desk, study, workspace, workstation, computer, bookshelf, shelving, chair |
| Kitchen | kitchen, cabinet, counter, stove, oven, refrigerator, cooking, sink, island, appliance, backsplash |
| Living Room | living, lounge, sofa, couch, fireplace, mantel, seating, coffee table, sectional |
| Pool | pool, swim, patio, outdoor, deck, backyard, spa, hot tub, water, lounge |

---

## Outputs

After a full run, the following files are saved to `/kaggle/working/`:

| File | Description |
|------|-------------|
| `best_lstm_decoder_v2.pth` | Best decoder checkpoint (by val loss) |
| `lstm_decoder_export.pth` | Final export with decoder weights + vocab + metrics |
| `training_curves_v2.png` | Loss, accuracy, and LR schedule plots |
| `training_history_v2.json` | Epoch-by-epoch training/val loss and accuracy |
| `sample_captions_v2.png` | 8 qualitative examples (GT vs Greedy vs Beam) |
| `room_type_accuracy_v3.png` | Per-room-type accuracy bar chart |
| `test_predictions_v2.csv` | All test set predictions with ground truth |
| `eval_metrics_v2.json` | All evaluation metrics in JSON format |

---

## Dependencies

```
torch >= 1.12
torchvision >= 0.13
pandas
numpy
matplotlib
nltk
rouge-score
Pillow
```

---

