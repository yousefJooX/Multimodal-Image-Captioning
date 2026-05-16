# 🖼️ Multimodal Image Captioning: A Comparative Study

A comparative study of Sequential and Attention-based architectures for automatic image captioning of interior design images.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red)
![CV](https://img.shields.io/badge/Computer_Vision-Image_Captioning-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

## 🎯 Overview

This project implements and compares **multiple encoder-decoder architectures** for generating descriptive captions of interior design images (kitchens, bedrooms, bathrooms, living rooms, etc.). Data was scraped from Houzz.com and models were trained end-to-end.

## 🏗️ Architectures Compared

| Stage | Encoder | Decoder | Approach |
|-------|---------|---------|----------|
| 4 | ResNet-50 | LSTM | Sequential baseline |
| 5 | ResNet-50 | LSTM + Attention | Attention mechanism |
| 6 | EfficientNet-B0 | GRU | Lightweight alternative |
| 7 | ViT (Vision Transformer) | Transformer Decoder | Full transformer |

## 📁 Project Structure

```
├── Stage1_scraping_notebooks/    # Web scraping from Houzz.com
├── Stage2_clean_data_final/      # Data cleaning & train/val/test split
├── stage3_preprocessing/         # Tokenization, vocab building
├── Stage4.1_EncoderResnet/       # ResNet-50 encoder
├── Stage4.2_DecoderLSTM/         # LSTM decoder (baseline)
├── Stage5_DecoderLSTM_Attention/ # LSTM + Attention decoder
├── Stage6.1_EncoderEfficientNet/ # EfficientNet-B0 encoder
├── Stage6.2_DecoderGRU/          # GRU decoder
├── Stage7_VitTransfomer/         # ViT + Transformer decoder
└── data/                         # Raw scraped datasets
```

## 🔄 Pipeline

```
Web Scraping → Data Cleaning → Preprocessing → Encoder Training → Decoder Training → Evaluation
```

1. **Data Collection**: Scraped ~6 categories of interior images with captions from Houzz.com
2. **Preprocessing**: Cleaned text, built vocabulary, resized images, created train/val/test splits
3. **Training**: Trained encoder-decoder pairs with teacher forcing
4. **Evaluation**: Compared using BLEU, METEOR, and qualitative analysis

## 🚀 Getting Started

```bash
git clone https://github.com/yousefJooX/Multimodal-Image-Captioning.git
cd Multimodal-Image-Captioning
pip install torch torchvision pandas numpy matplotlib tqdm pillow
```

Run notebooks in order (Stage 1 → Stage 7).

## 🛠️ Tech Stack

- **PyTorch** - Model implementation
- **ResNet-50 / EfficientNet-B0 / ViT** - Image encoders
- **LSTM / GRU / Transformer** - Caption decoders
- **BeautifulSoup** - Web scraping
- **Pandas / NumPy** - Data processing

## 👨‍💻 Author

**Yousef Mohammed** - [GitHub](https://github.com/yousefJooX)

---
⭐ Star this repo if you find it useful!
