# Telugu Text Summarization using NLP

> Undergraduate Capstone Project — Comparative study of transformer-based extractive summarization for Telugu, a low-resource Dravidian language.

---

## Overview

This project implements and evaluates three transformer-based models for **extractive text summarization of Telugu-language documents**. All models use sentence embeddings to score sentences by their semantic centrality to the document, then select the top-ranked sentences to form a summary.

Telugu is spoken by ~80 million people but remains underserved in NLP research — this project explores how multilingual and English-pretrained transformers generalize to it.

---

## Models Compared

| Model | Architecture | Pretrained On |
|-------|-------------|---------------|
| `bert-base-multilingual-cased` | BERT | 104 languages incl. Telugu |
| `t5-base` | T5 | English (C4 corpus) |
| `xlnet-base-cased` | XLNet | English (Books + Wikipedia) |

---

## Approach: Extractive Summarization

All three models follow the same pipeline:

```
Input Text
    ↓
Sentence Tokenization (NLTK)
    ↓
Sentence Embeddings (Transformer encoder)
    ↓
Cosine Similarity Matrix
    ↓
Sentence Scoring (sum of pairwise similarities)
    ↓
Top-N Sentences Selected → Summary
```

This is a **graph-free** variant of TextRank — sentences are ranked by their aggregate similarity to all other sentences (centrality), without building an explicit graph.

---

## Dataset

**Telugu Wikipedia Articles** — [Kaggle: disisbig/telugu-wikipedia-articles](https://www.kaggle.com/datasets/disisbig/telugu-wikipedia-articles)

- ~566 articles split across `train/` and `valid/` directories
- Plain `.txt` files, one article per file
- Topics: geography, culture, people, current events

---

## Results

Evaluated on a held-out validation article (`valid/123.txt`) using the full article as the reference for ROUGE, and a short reference sentence for the BLEU comparison.

| Model | BLEU Score | ROUGE-1 Recall | ROUGE-1 Precision | ROUGE-1 F1 |
|-------|-----------|----------------|-------------------|------------|
| **mBERT** | **0.4558** | **0.5625** | **1.0** | **0.7200** |
| XLNet | 0.0673 | 0.1875 | 1.0 | 0.3158 |
| T5 | 0.0036 | 0.1895 | 1.0 | 0.3187 |

**mBERT outperformed both XLNet and T5** by a wide margin. The result is consistent with its multilingual pretraining — it has seen Telugu during pretraining and can produce meaningful Telugu embeddings, while XLNet and T5 were trained on English-only corpora and effectively treat Telugu tokens as unknown.

---

## Repository Structure

```
telugu-text-summarization/
├── Telugu_text_summarization_Bert.ipynb    # mBERT extractive summarizer
├── Telugu_text_summarization_T5.ipynb      # T5 extractive summarizer
├── Telugu_text_summarization_XLNet.ipynb   # XLNet extractive summarizer
└── README.md
```

---

## Setup & Running

All notebooks are designed to run on **Google Colab** (GPU recommended for T5/XLNet).

### 1. Dataset

```bash
# Requires a Kaggle API key at /content/kaggle.json
!kaggle datasets download -d disisbig/telugu-wikipedia-articles
!unzip /content/telugu-wikipedia-articles.zip -d dataset
```

### 2. Dependencies

```bash
pip install transformers sentence-transformers scikit-learn nltk rouge torch
```

### 3. Run

Open any notebook in Colab and run all cells in order. Each notebook:
1. Downloads and unzips the dataset
2. Loads the model and tokenizer
3. Generates summaries for the training set
4. Runs inference on a validation file
5. Reports BLEU and ROUGE-1 scores

---

## Key Findings

- **Multilingual pretraining matters** — mBERT's inclusion of Telugu in its pretraining corpus gives it a decisive advantage over English-only models for this task.
- **Extractive summarization is language-agnostic in structure** — the cosine similarity ranking pipeline works for any language, but quality depends entirely on embedding quality.
- **XLNet slightly outperforms T5** on ROUGE-1 F1 (0.3158 vs 0.3187 — near-identical), suggesting both struggle equally when tokenizing unseen scripts.
- **ROUGE-1 Precision = 1.0 across all models** — this is an artifact of using a very short reference sentence; all summary tokens appear in it by coincidence. The recall and F1 values are the meaningful metrics here.

---

## Limitations

- NLTK `sent_tokenize` is English-optimized — Telugu sentence boundaries may be incorrectly detected, affecting summary quality
- T5 and XLNet were not fine-tuned on Telugu; results reflect zero-shot cross-lingual transfer only
- No abstractive summarization was attempted — future work could explore fine-tuning `google/mt5-base` or `ai4bharat/indic-bart` on a Telugu summarization dataset

---

## Tech Stack

`Python` · `PyTorch` · `HuggingFace Transformers` · `NLTK` · `scikit-learn` · `ROUGE` · `Google Colab`

---

## Author

**Sai Sarant Tottempudi**
Undergraduate Capstone Project — Natural Language Processing
