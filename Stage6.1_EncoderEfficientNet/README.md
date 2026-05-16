# EfficientNet-B0 Encoder

Alternative CNN encoder for the image captioning pipeline. Drop-in
replacement for `EncoderResnet/` — same 256-d output, same vocab, same
image normalization, same export schema — so any downstream decoder
(GRU, LSTM, attention) that consumes a 256-d feature vector accepts the
exported checkpoint without code changes.

## What's different from the ResNet-50 encoder

| | ResNet-50 (`EncoderResnet/`) | EfficientNet-B0 (this folder) |
|---|---|---|
| Backbone params | ~25M | ~5M |
| Backbone feature dim | 2048 | 1280 |
| Fine-tuned layers | `layer4` | `features[-2:]` (last two MBConv blocks) |
| Projection head | `2048 → 1024 → 256 + LN` | `1280 → 1024 → 256 + LN` |
| Output embedding | 256-d | 256-d (identical) |
| ImageNet weights | `ResNet50_Weights.IMAGENET1K_V2` | `EfficientNet_B0_Weights.IMAGENET1K_V1` |

Every other knob — augmentation, optimizer, LR, label smoothing, grad clip,
batch size, epochs, evaluation, export keys — is held constant so the
ResNet-vs-EfficientNet comparison is about the **backbone**, not the
training recipe.

## Files

| File | Description |
|---|---|
| `efficientnetb0-encoder.ipynb` | Full training/eval/export notebook (13 sections) |
| `CNN.ipynb` | Empty placeholder from the original repo — safe to delete |
| `model1.ipynb` | Reference notebook from a teammate — kept for path examples |

## How to run

Same as the ResNet notebook:

1. Upload to Kaggle, attach the
   `joox1113/interior-room-images-captions-houzz` dataset (the paths in
   section 2 are already set for this dataset)
2. Enable GPU (T4 x2 works; the notebook auto-detects multi-GPU and uses
   `DataParallel`)
3. Settings → Internet → ON (needed for the pip / nltk lines in section 1)
4. Run all cells

After a successful run, `/kaggle/working/` contains:

- `best_model.pth` — full training checkpoint (encoder + GRU + optimizer)
- `encoder_for_decoder.pth` — **the handoff file** for whatever decoder you use
- `eval_metrics.json` — BLEU-1/4, METEOR, ROUGE-L, CIDEr, Room-Acc%
- `test_predictions.csv` — generated vs. ground-truth captions
- `training_history.{json,png}` — loss/accuracy per epoch
- `sample_captions.png` — qualitative examples

## Joint training: why a GRU is used inside the notebook

The encoder is trained end-to-end against a small GRU decoder (section 5).
That GRU is the **temporary scaffolding** that gives the encoder a
sequence-level supervision signal — token-level cross-entropy across the
caption, much denser than a single classification label per image.

You can re-use the same GRU as your production decoder, or feed
`encoder_for_decoder.pth` into a different decoder downstream. The export
schema is decoder-agnostic.

## Loading the exported encoder downstream

```python
ckpt = torch.load('encoder_for_decoder.pth')
assert ckpt['backbone'] == 'efficientnet_b0'

encoder = EncoderEfficientNetB0(embed_dim=ckpt['embed_dim'],
                                 fine_tune=False, dropout=0.0)
encoder.load_state_dict(ckpt['encoder_state_dict'])
encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False
```

The vocab (`word2idx`, `idx2word`), special token IDs, image normalization
constants, and `max_seq_len` are all bundled inside the checkpoint — your
downstream code reads them from `ckpt` rather than rebuilding them.

## Choosing a different EfficientNet variant

The notebook uses B0 for a fair head-to-head against ResNet-50. To swap
to a larger variant, change two lines in section 5:

```python
from torchvision.models import EfficientNet_B3_Weights  # or B4, B5, ...
backbone = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
```

Larger variants expect bigger inputs (B3 → 300×300, B4 → 380×380, etc.).
Adjust `transforms.Resize` and `RandomCrop` accordingly, and expect higher
GPU memory pressure.
