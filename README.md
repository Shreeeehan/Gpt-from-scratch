# GPT From Scratch

A minimal implementation of a GPT-style language model trained on the Tiny Shakespeare dataset, built from scratch using PyTorch.

---

## What is this?

This project implements a decoder-only transformer (the architecture behind GPT) at the character level. Given a sequence of characters, the model learns to predict the next character — and after training, it can generate Shakespeare-like text autoregressively.

The goal is to understand every component of a transformer by building it from the ground up: tokenization, embeddings, causal self-attention, multi-head attention, feed-forward layers, residual connections, and layer normalization.

---

## Architecture

The model is a decoder-only transformer with the following structure:

| Component | Detail |
|---|---|
| Embedding dim (`n_embd`) | 64 |
| Attention heads (`n_head`) | 4 |
| Transformer layers (`n_layer`) | 4 |
| Context length (`block_size`) | 8 tokens |
| Vocabulary | 65 unique characters |

**Data flow:**

```
Token IDs
    → Token Embeddings + Positional Embeddings
    → 4x Transformer Block:
        └─ Pre-LayerNorm → Multi-Head Self-Attention → Residual Add
        └─ Pre-LayerNorm → Feed-Forward (64 → 256 → 64) → Residual Add
    → Final LayerNorm
    → Linear head → Logits (vocab_size)
```

**Key design choices:**
- **Causal masking** — each token only attends to itself and earlier tokens (no future leakage)
- **Pre-norm** — LayerNorm applied before each sub-layer, which stabilises training
- **Scaled dot-product attention** — scores divided by √(head_size) to prevent softmax saturation
- **Residual connections** — allow gradients to flow cleanly through deep stacks

---

## Dataset

[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) — a 1.1 MB concatenation of Shakespeare plays (~1M characters). The dataset is split 90% train / 10% validation.

The file is expected at `input.txt` in the project root.

---

## How to Run

**Requirements:**
```
python >= 3.10
torch
```

**Install dependencies:**
```bash
pip install torch
```

**Download the dataset:**
```bash
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

**Train the model:**
```bash
python gpt.py
```

Training runs for 5000 steps and prints loss every 500 steps. A 200-character sample is generated at the end.

---

## Results

| Stage | Train Loss | Val Loss |
|---|---|---|
| Untrained (random) | 4.26 | 4.26 |
| After 5000 steps | **2.14** | **2.20** |

**Sample output after training:**

```
I I him.

Nendom the the stigt.

Olbrownour anly, there chopersce; are, petthent,
His arqued scenppeace,
Hoth nowpd as with ramt, not
orfor tor, a jouderll,
And whine.
```

The model learns word boundaries, punctuation, capitalization, and dialogue structure from scratch — with no pre-training, no tokenizer, and no external libraries beyond PyTorch.

---

## Project Structure

```
gpt-from-scratch/
├── gpt.py       # full implementation: tokenizer, model, training loop
└── input.txt    # Tiny Shakespeare dataset
```

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Andrej Karpathy
