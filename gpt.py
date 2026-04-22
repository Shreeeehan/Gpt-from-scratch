import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Read the dataset ──────────────────────────────────────────────────────────
# Load the raw text from the Tiny Shakespeare file downloaded from Karpathy's
# char-rnn repo (1M+ characters of Shakespeare plays).
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ── Build the vocabulary ──────────────────────────────────────────────────────
# Every unique character in the corpus becomes one token.
# stoi maps character → integer index; itos maps index → character.
chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# ── Encoder and decoder functions ─────────────────────────────────────────────
# encode() turns a string into a list of integers using stoi.
# decode() turns a list of integers back into a string using itos.
def encode(s: str) -> list[int]:
    return [stoi[ch] for ch in s]

def decode(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids)

# ── Convert the full text to a PyTorch tensor ─────────────────────────────────
# Each character becomes a Long (int64) integer — this is our training data.
data = torch.tensor(encode(text), dtype=torch.long)

# ── Print summary information ─────────────────────────────────────────────────
# Shows vocab size, a human-readable preview, and the raw encoded values.
print(f"Vocab size : {vocab_size}")
print(f"\nFirst 500 characters of text:\n{'-'*40}")
print(text[:500])
print(f"\nFirst 100 tensor values:\n{'-'*40}")
print(data[:100])

# ── Train / validation split ──────────────────────────────────────────────────
# First 90% of tokens go to training; the remaining 10% are held out for
# validation so we can measure generalisation without touching test data.
n = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

# ── Hyperparameters ───────────────────────────────────────────────────────────
# block_size: maximum context length the model sees when making a prediction.
# batch_size: number of independent sequences processed in parallel per step.
block_size = 8
batch_size = 4

# ── Batch sampler ─────────────────────────────────────────────────────────────
# Picks `batch_size` random starting positions, then slices a block of
# `block_size` tokens for inputs (x) and the same block shifted by one for
# targets (y), giving us every (context → next-token) pair in one shot.
def get_batch(split: str):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i      : i + block_size    ] for i in ix])
    y = torch.stack([source[i + 1  : i + block_size + 1] for i in ix])
    return x, y

# ── Verify with one sample batch ─────────────────────────────────────────────
# Print shapes and the raw token values so we can confirm the tensors look right
# before wiring them into the model.
xb, yb = get_batch("train")
print(f"\nSample batch inputs  shape : {xb.shape}")
print(f"Sample batch targets shape : {yb.shape}")
print(f"\nInputs:\n{xb}")
print(f"\nTargets:\n{yb}")

# ── Model hyperparameters ─────────────────────────────────────────────────────
# n_embd  : embedding dimension (width of every vector in the residual stream)
# n_head  : number of attention heads; each gets n_embd // n_head dimensions
# n_layer : number of stacked transformer blocks
# dropout : fraction of activations zeroed during training (0 = disabled)
n_embd  = 64
n_head  = 4
n_layer = 4
dropout = 0.0

# ── Single Self-Attention Head ────────────────────────────────────────────────
# Scaled dot-product causal self-attention for one head.
# Operates on a slice of the embedding (head_size = n_embd // n_head dims).
# key   : what this token broadcasts about itself
# query : what this token is searching for
# value : what this token contributes when attended to
# The causal mask (tril) blocks any position from peeking at future tokens.
class Head(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.key     = nn.Linear(n_embd, head_size, bias=False)
        self.query   = nn.Linear(n_embd, head_size, bias=False)
        self.value   = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Scale by 1/√head_size so dot-products don't grow with head dimension,
        # which would push softmax into near-zero gradient regions.
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)   # (B, T, T)
        wei = self.dropout(wei)

        v = self.value(x)  # (B, T, head_size)
        return wei @ v     # (B, T, head_size)


# ── Multi-Head Attention ──────────────────────────────────────────────────────
# Runs n_head independent attention heads in parallel, then concatenates their
# outputs and projects back to n_embd. The parallel heads let the model attend
# to different aspects of context simultaneously (syntax, semantics, etc.).
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads   = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj    = nn.Linear(n_embd, n_embd)  # mix head outputs back into residual stream
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        return self.dropout(self.proj(out))


# ── Position-Wise Feed-Forward Network ───────────────────────────────────────
# A two-layer MLP applied independently to each token position.
# The 4x inner expansion follows the original "Attention is All You Need" ratio
# and gives the model capacity to transform representations non-linearly.
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ── Transformer Block ─────────────────────────────────────────────────────────
# One full transformer layer: pre-LayerNorm → attention → residual add,
# then pre-LayerNorm → feed-forward → residual add.
# Pre-norm (applied before each sub-layer) stabilises training better than
# the original post-norm formulation, especially without a warmup schedule.
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.sa  = MultiHeadAttention(n_head, head_size)
        self.ff  = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # attention sub-layer with residual
        x = x + self.ff(self.ln2(x))  # feed-forward sub-layer with residual
        return x


# ── GPT Language Model ────────────────────────────────────────────────────────
# Full decoder-only transformer: token + position embeddings → n_layer blocks
# → final LayerNorm → linear head that maps to vocab logits.
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(n_embd)  # final norm before the linear head
        self.lm_head = nn.Linear(n_embd, vocab_size)

    # ── Forward pass ─────────────────────────────────────────────────────────
    # idx     : (B, T) integer token indices
    # targets : (B, T) next-token targets (optional; required for loss)
    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                 # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T))  # (T,  n_embd)
        x       = tok_emb + pos_emb                               # (B, T, n_embd)
        x       = self.blocks(x)                                  # (B, T, n_embd)
        x       = self.ln_f(x)                                    # (B, T, n_embd)
        logits  = self.lm_head(x)                                 # (B, T, vocab_size)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    # ── Autoregressive generation ─────────────────────────────────────────────
    # Appends one sampled token per step. Context is cropped to block_size so
    # the position embedding table is never asked for an out-of-range index.
    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits    = logits[:, -1, :]                          # (B, vocab_size)
            probs     = F.softmax(logits, dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1)   # (B, 1)
            idx       = torch.cat([idx, idx_next], dim=1)         # (B, T+1)
        return idx


# ── Instantiate model and verify shapes ──────────────────────────────────────
# A fresh random model should give loss ≈ ln(65) ≈ 4.17 (uniform over vocab).
model        = GPTLanguageModel()
logits, loss = model(xb, yb)
print(f"\nLogits shape : {logits.shape}")   # expect (4, 8, 65)
print(f"Initial loss : {loss.item():.4f}")  # expect ≈ 4.17

# ── Training hyperparameters ──────────────────────────────────────────────────
# eval_iters batches are averaged per estimate_loss() call to reduce variance.
max_iters     = 5000
eval_interval = 500
learning_rate = 1e-3
eval_iters    = 200

# ── Optimizer ─────────────────────────────────────────────────────────────────
# AdamW decouples weight decay from the gradient update, which regularises
# weights without interfering with the adaptive learning-rate mechanism.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ── Loss estimator ────────────────────────────────────────────────────────────
# Averages loss over eval_iters random batches for both splits.
# torch.no_grad() disables gradient tracking, saving memory during evaluation.
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ── Training loop ─────────────────────────────────────────────────────────────
# Each step: sample batch → forward → loss → backward → optimizer step.
# Loss is reported every eval_interval steps without stopping gradient flow.
print("\nStarting training...")
for step in range(max_iters):

    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"  step {step:>5} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    xb, yb       = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ── Final evaluation ──────────────────────────────────────────────────────────
final = estimate_loss()
print(f"\nTraining complete.")
print(f"  Final train loss : {final['train']:.4f}")
print(f"  Final val loss   : {final['val']:.4f}")

# ── Post-training sample ──────────────────────────────────────────────────────
# Generate 200 characters from a single newline seed to gauge text quality.
context   = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(context, max_new_tokens=200)
print(f"\nTrained sample:\n{decode(generated[0].tolist())}")
