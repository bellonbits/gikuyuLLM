# 🇰🇪 Gĩkũyũ LLM — From-Scratch Language Model for the Kikuyu Language

> A GPT-style decoder-only transformer trained entirely from scratch on the Gĩkũyũ language — built for cultural preservation, low-resource NLP research, and as a foundation for Kikuyu-language AI applications.

**Author:** Peter Gatitu Mwangi · [github.com/bellonbits](https://github.com/bellonbits) · [huggingface.co/analystgatitu](https://huggingface.co/analystgatitu)

---

## 📖 Project Overview

This project builds a Gĩkũyũ language model completely from scratch — no pre-trained base, no transfer learning. Starting from raw Gĩkũyũ text (the Biblica Gĩkũyũ Bible corpus), the pipeline:

1. **Trains a BPE tokenizer** purpose-built for Kikuyu morphology and Unicode diacritics
2. **Pre-trains a GPT-style transformer** (~23M parameters) on next-token prediction
3. **Prepares the model for SFT (Supervised Fine-Tuning)** into a conversational assistant

The project demonstrates that capable language models can be built for low-resource African languages with standard hardware (Google Colab T4 GPU).

---

## 🗂️ Repository Structure

```
kikuyu-llm/
├── kikuyu_tokenizer.ipynb      # Step 1: BPE tokenizer training
├── kikuyu_pretrain.ipynb       # Step 2: GPT pre-training
├── data/
│   ├── kikuyu_corpus.txt       # Raw Gĩkũyũ text (upload to Drive)
│   ├── kikuyu_corpus.jsonl     # Corpus in JSONL format for training
│   └── kikuyu_sft_combined.jsonl  # SFT dataset (for next step)
└── checkpoints/                # Saved model weights (auto-created)
    ├── epoch_01.pt
    ├── epoch_02.pt
    ├── ...
    └── best_model.pt           # Best checkpoint by validation loss
```

> **Note:** All data files and checkpoints are stored in Google Drive at `MyDrive/kikuyu_llm/`. They are not committed to the repository due to size.

---

## 🚀 Quickstart

### Prerequisites

- Google Colab account (free tier works; T4 GPU recommended for pre-training)
- Google Drive with at least 2GB free
- `kikuyu_corpus.txt` uploaded to `MyDrive/kikuyu_llm/`

### Step 1 — Train the Tokenizer

Open `kikuyu_tokenizer.ipynb` in Colab. No GPU needed — CPU runtime is fine.

```
Runtime > Change runtime type > CPU
Run all cells
```

**Output:** `MyDrive/kikuyu_llm/kikuyu_bpe_tokenizer.json`  
**Time:** ~2–5 minutes

### Step 2 — Pre-train the Model

Open `kikuyu_pretrain.ipynb` in Colab. **GPU required.**

```
Runtime > Change runtime type > GPU > T4
Run all cells
```

**Output:** `MyDrive/kikuyu_llm/checkpoints/best_model.pt`  
**Time:** Varies by corpus size and epochs (typically 30–90 min on T4)

---

## 🔤 Tokenizer — `kikuyu_tokenizer.ipynb`

### Design Decisions

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Algorithm | BPE (Byte Pair Encoding) | Handles morphologically rich Gĩkũyũ word forms |
| Vocab size | 32,000 | Same as LLaMA; balances coverage vs. model size |
| Min frequency | 2 | Subword must appear at least twice to enter vocab |
| Normalizer | **NFC** (not NFKC) | Preserves Gĩkũyũ diacritics: `ĩ`, `ũ`, `ã` |
| Pre-tokenizer | Metaspace | Splits on spaces; encodes word-start as `_` prefix |

> ⚠️ **Critical:** `NFKC` normalization would decompose and strip the diacritics `ĩ`, `ũ`, and `ã` that are phonemically meaningful in Gĩkũyũ. This tokenizer uses `NFC` to preserve them as single Unicode code points.

### Special Tokens

| ID | Token | Purpose |
|----|-------|---------|
| 0 | `<pad>` | Padding for batch alignment |
| 1 | `<unk>` | Unknown token fallback |
| 2 | `<bos>` | Beginning of sequence |
| 3 | `<eos>` | End of sequence |
| 4 | `<sep>` | Segment separator |
| 5 | `<\|user\|>` | Chat format — user turn |
| 6 | `<\|assistant\|>` | Chat format — assistant turn |
| 7 | `<\|system\|>` | Chat format — system prompt |

### Validation

The tokenizer notebook runs three automatic checks:

- **Roundtrip test** — encode → decode must return the original text exactly
- **Special token injection** — `<bos>` and `<eos>` are verified to wrap every sequence automatically via post-processor
- **Fertility score** — tokens per word across a 5,000-line sample

| Fertility | Assessment |
|-----------|------------|
| < 2.0 | Excellent — well-optimised for Gĩkũyũ |
| 2.0 – 2.5 | Good — acceptable for Gĩkũyũ morphology |
| 2.5 – 3.0 | Fair — consider larger vocab or more data |
| > 3.0 | High — tokenizer needs improvement |

---

## 🧠 Model Architecture — `kikuyu_pretrain.ipynb`

A decoder-only GPT-style transformer built in pure PyTorch.

### Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `d_model` | 384 | Token embedding dimension |
| `n_layers` | 6 | Transformer blocks |
| `n_heads` | 6 | Attention heads per block |
| `d_ff` | 1,536 | Feed-forward hidden size (4 × d_model) |
| `vocab_size` | 32,000 | From BPE tokenizer |
| `max_seq_len` | 256 | Context window (tokens) |
| **Total params** | **~23M** | |

### Architecture Components

```
KikuyuGPT
├── token_emb        (Embedding: vocab_size × d_model)
├── pos_emb          (Embedding: max_seq_len × d_model)
├── blocks × 6
│   ├── LayerNorm
│   ├── MultiHeadSelfAttention   ← causal mask (lower-triangular)
│   ├── LayerNorm
│   └── FeedForward              ← GELU activation
├── ln_f             (final LayerNorm)
└── head             (Linear: d_model → vocab_size, weight-tied)
```

**Weight tying:** The output projection (`head`) shares weights with `token_emb`, saving ~12M parameters and improving convergence — a standard technique from the original GPT papers.

**Causal masking:** Each token attends only to itself and tokens before it (lower-triangular mask), enforcing the autoregressive next-token prediction objective.

### Training Setup

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1) |
| Peak LR | 3e-4 |
| LR schedule | Linear warmup (500 steps) → cosine decay to ~0 |
| Batch size | 32 sequences |
| Gradient clipping | 1.0 (global norm) |
| Max epochs | 10 |
| Train/val split | 90% / 10% |
| Objective | Cross-entropy next-token prediction |

### Dataset Construction

The corpus is tokenized into a single flat tensor of token IDs, then sliced into 256-token chunks:

```
tokens:  [A B C D E F ...]
input:   [A B C D]          → model input
target:  [B C D E]          → model must predict each next token
```

### Checkpointing

- Every epoch saves `epoch_NN.pt` to Drive
- `best_model.pt` always holds the checkpoint with the lowest validation loss
- Checkpoints include: model weights, optimizer state, epoch, global step, config — enabling full training resumption

### Training Curves

After training, the notebook plots loss and perplexity curves for both train and val sets, with automatic interpretation:

- **Val-train gap < 0.5** → healthy generalisation
- **Val-train gap > 0.8** → overfitting; add data or increase dropout
- **Val loss > 6.0** → underfitting; consider more epochs or lower LR

---

## 💬 Text Generation

The pre-trained model generates Gĩkũyũ text continuations using Top-K + Nucleus (Top-P) sampling:

```python
generate_text(
    prompt="Ngai niombire iguru na thi",
    max_new_tokens=80,
    temperature=0.8,   # higher = more creative
    top_k=50,
    top_p=0.9
)
```

**Temperature guide:**

| Temperature | Behaviour |
|-------------|-----------|
| 0.3 | Focused / deterministic |
| 0.6 | Balanced |
| 0.9 | Creative |
| 1.2 | Wild / unpredictable |

> At this stage the model generates plausible Gĩkũyũ text continuations but does not respond to questions in chat format. That requires the SFT step (next notebook).

---

## 📋 Files Needed at Each Step

| File | Produced by | Consumed by |
|------|-------------|-------------|
| `kikuyu_corpus.txt` | You (upload) | `kikuyu_tokenizer.ipynb` |
| `kikuyu_corpus.jsonl` | You (upload) | `kikuyu_pretrain.ipynb` |
| `kikuyu_bpe_tokenizer.json` | `kikuyu_tokenizer.ipynb` | `kikuyu_pretrain.ipynb`, SFT |
| `checkpoints/best_model.pt` | `kikuyu_pretrain.ipynb` | SFT notebook |
| `kikuyu_sft_combined.jsonl` | You (prepare) | SFT notebook |

---

## 🔧 Resuming Training

If your Colab session disconnects mid-training, uncomment the checkpoint resume block in Section 5 of `kikuyu_pretrain.ipynb`:

```python
ckpt = torch.load(f"{CHECKPOINT_DIR}/best_model.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
start_epoch = ckpt["epoch"]
best_val    = ckpt["val_loss"]
global_step = ckpt["global_step"]
```

---

## 🗺️ Roadmap

- [x] BPE tokenizer with Gĩkũyũ diacritic preservation
- [x] GPT decoder pre-training (~23M params)
- [x] Checkpointing, LR scheduling, perplexity curves
- [ ] Supervised Fine-Tuning (SFT) for conversational format
- [ ] Expand corpus beyond Bible (news, literature, oral tradition)
- [ ] Scale model to 125M parameters
- [ ] HuggingFace model card and public release

---

## 🌍 Why This Project

Gĩkũyũ is spoken by over 8 million people in Kenya yet has virtually no dedicated NLP resources or language models. Most multilingual models (mBERT, BLOOM, etc.) include little to no Gĩkũyũ data. This project is a step toward:

- **Cultural preservation** — encoding the language in a computational form
- **Low-resource NLP research** — demonstrating the full pipeline for an underrepresented African language
- **Future applications** — educational tools, translation aids, and conversational AI in Gĩkũyũ

---

## 📦 Dependencies

```
torch
tokenizers
tqdm
matplotlib
```

Install via:
```bash
pip install torch tokenizers tqdm matplotlib
```

> All notebooks are designed to run on **Google Colab** with Drive mounting. No local GPU required for the tokenizer step. T4 GPU recommended for pre-training.

---

## 📄 Licence

This project is open source. The Biblica Gĩkũyũ Bible corpus used for training is sourced under its respective licence — for research and educational use.

---

*Built with 🤍 for the Gĩkũyũ language and the people who speak it.*
