# GRU Decoder v2

Sibling of `DecoderLSTM/`. Identical training recipe — only the recurrent
cell is swapped from LSTM to GRU. Built so the GRU-vs-LSTM result is a
pure architecture comparison, not a recipe comparison.

## What's in `gru_Decoder-v2.ipynb`

20 sections that follow `DecoderLSTM/lstm_Decoder-v2.ipynb` cell-for-cell:

1. Install eval libs
2. Imports
3. Device + seeds
4. Hyperparameters (same constants as LSTM v2)
5. Load encoder checkpoint (`encoder_for_decoder.pth`)
6. Data paths (`joox1113/interior-room-images-captions-houzz`)
7. Dataset + DataLoaders
8. Rebuild `EncoderEfficientNetB0` and freeze it
9. **`GRUDecoderV2`** — the new class
10. Loss / AdamW / OneCycleLR
11. Training loop with scheduled sampling
12. Plot loss / accuracy / LR
13. Load best checkpoint
14. Greedy + beam-search decoding
15. CIDEr (inline implementation)
16. Qualitative examples (8 test images)
17. Full eval (BLEU-1/4, METEOR, ROUGE-L, CIDEr)
18. Per-room accuracy + bar chart
19. Save test predictions CSV
20. Save metrics + export `gru_decoder_export.pth`

## GRU vs LSTM — what actually differs

| | LSTM v2 | **GRU v2 (this notebook)** |
|---|---|---|
| Hidden state | `(h, c)` | `h` only |
| Init layers | `init_h` + `init_c` | `init_h` only |
| Recurrent init trick | forget-bias = 1 | **update-gate bias = −1** (PyTorch GRU bias layout `[reset, update, new]`) |
| Param count | ~25% more | ~25% fewer per recurrent layer |
| Everything else | — | unchanged |

Everything else — feature gating (sigmoid gate over image features), scheduled
sampling (1.0 → 0.7 over training), weight tying (embedding ↔ output FC),
label smoothing, gradient clipping, OneCycleLR, beam search with length
normalization — is held constant.

### Why update-gate bias = −1?

The LSTM forget-bias = 1 trick (Jozefowicz et al. 2015) lets the cell start
out remembering everything by default. The GRU analog is: at init, push the
update gate toward 0 so `h_new ≈ h_prev`. Since the update gate is
`sigmoid(z)`, setting its pre-activation bias to −1 yields `~0.27` — the
recurrent path dominates early in training, giving the optimizer time to
learn what's worth updating.

The bias slice for the update gate in PyTorch's GRU is `[hidden : 2*hidden]`
of `bias_ih_l*` and `bias_hh_l*` (layout: `[reset | update | new]`).

## Setup checklist before running on Kaggle

The decoder needs **two** Kaggle datasets attached:

1. **Image + caption data** — `joox1113/interior-room-images-captions-houzz`  
   (the one you used for the encoder run; paths in section 6 already point at it)
2. **Your encoder checkpoint** — upload `encoder_for_decoder.pth` (produced by
   `EncoderEfficientNet/efficientnetb0-encoder.ipynb`) as a private Kaggle
   dataset, then update `ENCODER_PATH` in section 5:
   ```python
   ENCODER_PATH = '/kaggle/input/<your-dataset-slug>/encoder_for_decoder.pth'
   ```

Also:
- Settings → Internet → ON (for pip + nltk downloads in section 1)
- Accelerator → GPU (T4 is plenty; the decoder is small)

Section 5 asserts the checkpoint's `'backbone'` tag is `'efficientnet_b0'`
and will refuse to load a ResNet encoder — that's intentional, prevents
silent architecture mismatches.

## Outputs (in `/kaggle/working/`)

| File | Description |
|---|---|
| `best_gru_decoder_v2.pth` | Best decoder checkpoint by val loss (full training state) |
| `gru_decoder_export.pth` | **Decoder export** — weights + vocab + arch info + metrics, self-contained |
| `training_history_v2.json` | Per-epoch loss / accuracy / LR |
| `training_curves_v2.png` | Loss, accuracy, LR plots |
| `sample_captions_v2.png` | 8 qualitative examples (GT / Greedy / Beam) |
| `room_type_accuracy.png` | Per-room accuracy bar chart |
| `test_predictions_v2.csv` | All test set predictions vs ground truth |
| `eval_metrics_v2.json` | BLEU-1/4 (greedy + beam), METEOR, ROUGE-L, CIDEr, Room-Acc% |

## Reading the metrics

For a clean comparison against the LSTM, look at the same metric on both
notebooks' `eval_metrics_v2.json`:

| Metric | What to compare |
|---|---|
| BLEU-4 (greedy) | Token-level fluency — most sensitive to the recurrent cell |
| METEOR | Word-overlap with synonyms — captures lexical diversity |
| CIDEr | Consensus-weighted n-gram match — best holistic captioning metric |
| Room-Acc % | Whether the caption identifies the room — proxy for "did the encoder give the decoder enough signal" |

If GRU ≈ LSTM on most metrics but the LSTM wins on long captions (look at
beam-search numbers more than greedy), the conclusion is "the cell state
matters when sequence length grows." If they're indistinguishable across
the board, GRU is the more efficient choice and the cell state isn't
buying you anything for this dataset.

## Loading the exported decoder downstream

```python
ckpt = torch.load('gru_decoder_export.pth')
decoder = GRUDecoderV2(
    vocab_size=ckpt['vocab_size'],
    embed_dim=ckpt['embed_dim'],
    hidden_dim=ckpt['hidden_dim'],
    num_layers=ckpt['num_layers'],
    dropout=ckpt['dropout'],
    pad_idx=ckpt['pad_idx'],
)
decoder.load_state_dict(ckpt['decoder_state_dict'])
decoder.eval()
```

Vocab (`word2idx`, `idx2word`) and special token IDs travel with the
checkpoint, so the inference code is fully self-contained.
