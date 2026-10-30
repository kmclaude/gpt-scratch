# =====================================================================
# STEP 7: STACK BLOCKS + RESIDUAL + LAYERNORM  =  a real (tiny) GPT
# Run:  python3 step7_gpt.py      (takes ~3 to 6 minutes — it trains 2 models)
#
# Step 6 did ONE round of [communicate -> think]. Real GPTs repeat it many times.
# But just stacking rounds makes training hard. Two small tricks fix it:
#   1. RESIDUAL:   x = x + change      (add a correction, don't replace everything)
#   2. LAYERNORM:  keep the numbers in a normal range before each step
# =====================================================================

import math                                   # for math.sqrt
import torch                                  # main library: tensors + autograd
import torch.nn as nn                         # ready-made building blocks
import torch.nn.functional as F               # plain functions

torch.set_printoptions(precision=2, sci_mode=False)   # print tensors with 2 decimals


# ---------------------------------------------------------------------
# Data setup (same as before)
# ---------------------------------------------------------------------
text = open("input.txt", encoding="utf-8").read()       # read the Shakespeare file
chars = sorted(set(text))                     # vocabulary (65 unique chars)
vocab_size = len(chars)                       # 65
stoi = {ch: i for i, ch in enumerate(chars)}  # char -> ID
itos = {i: ch for i, ch in enumerate(chars)}  # ID -> char
encode = lambda s: [stoi[c] for c in s]       # text -> IDs
decode = lambda ids: "".join(itos[i] for i in ids)      # IDs -> text
data = torch.tensor(encode(text), dtype=torch.long)     # whole text as IDs
n = int(0.9 * len(data))                      # 90% split point
train_data, val_data = data[:n], data[n:]     # train / validation

block_size = 8                                # context length (same as before, for a fair comparison)
batch_size = 32                               # chunks per step
n_embd = 32                                   # vector size per char
n_head = 4                                    # heads per block
head_size = n_embd // n_head                  # 8 numbers per head
n_layer = 4                                   # NEW: how many blocks we stack


def get_batch(split):                         # one batch of inputs + answers
    d = train_data if split == "train" else val_data           # train or val
    ix = torch.randint(len(d) - block_size, (batch_size,))     # random starts
    x = torch.stack([d[i : i + block_size] for i in ix])       # inputs (32, 8)
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])   # answers (shifted by 1)
    return x, y                               # return both


# =====================================================================
# Building blocks from Step 6 (unchanged)
# =====================================================================
class Head(nn.Module):                        # one attention head
    def __init__(self):                       # constructor
        super().__init__()                    # required by torch
        self.query = nn.Linear(n_embd, head_size, bias=False)  # Query lens
        self.key   = nn.Linear(n_embd, head_size, bias=False)  # Key lens
        self.value = nn.Linear(n_embd, head_size, bias=False)  # Value lens
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))  # no-future mask

    def forward(self, x):                     # x: (B, T, 32)
        B, T, C = x.shape                     # unpack shape
        q, k, v = self.query(x), self.key(x), self.value(x)   # Q, K, V
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_size)   # match scores
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))   # hide future
        weights = F.softmax(scores, dim=-1)   # -> percentages
        return weights @ v                    # mix of Values


class MultiHeadAttention(nn.Module):          # several heads side by side
    def __init__(self):                       # constructor
        super().__init__()                    # required by torch
        self.heads = nn.ModuleList([Head() for _ in range(n_head)])   # 4 heads
        self.proj = nn.Linear(n_embd, n_embd) # NEW (small): mix the 4 heads' answers together after joining

    def forward(self, x):                     # x: (B, T, 32)
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # run all heads, join -> (B, T, 32)
        return self.proj(out)                 # blend the heads' findings -> (B, T, 32)


class FeedForward(nn.Module):                 # each char "thinks" alone
    def __init__(self):                       # constructor
        super().__init__()                    # required by torch
        self.net = nn.Sequential(             # layers in order
            nn.Linear(n_embd, 4 * n_embd),    # 32 -> 128
            nn.ReLU(),                        # negative -> 0
            nn.Linear(4 * n_embd, n_embd),    # 128 -> 32
        )

    def forward(self, x):                     # x: (B, T, 32)
        return self.net(x)                    # (B, T, 32)


# =====================================================================
# NEW: a "Block" = one round of [communicate -> think]
# =====================================================================
class Block(nn.Module):                       # one transformer block
    def __init__(self, use_tricks):           # use_tricks = True/False so we can compare
        super().__init__()                    # required by torch
        self.use_tricks = use_tricks          # remember the setting
        self.attn = MultiHeadAttention()      # communicate
        self.ffwd = FeedForward()             # think
        self.ln1 = nn.LayerNorm(n_embd)       # LayerNorm before attention (only used with tricks)
        self.ln2 = nn.LayerNorm(n_embd)       # LayerNorm before feed-forward (only used with tricks)

    def forward(self, x):                     # x: (B, T, 32)
        if self.use_tricks:                   # the real GPT way:
            x = x + self.attn(self.ln1(x))    # RESIDUAL: keep x, ADD what attention found (after normalizing)
            x = x + self.ffwd(self.ln2(x))    # RESIDUAL: keep x, ADD what thinking produced (after normalizing)
        else:                                 # the naive way (for comparison):
            x = self.attn(x)                  # REPLACE x with attention's output
            x = self.ffwd(x)                  # REPLACE x with feed-forward's output
        return x                              # (B, T, 32)


# =====================================================================
# The full GPT
# =====================================================================
class GPT(nn.Module):                         # the whole model
    def __init__(self, use_tricks):           # use_tricks = True/False
        super().__init__()                    # required by torch
        self.tok_emb = nn.Embedding(vocab_size, n_embd)   # WHAT
        self.pos_emb = nn.Embedding(block_size, n_embd)   # WHERE
        self.blocks = nn.Sequential(*[Block(use_tricks) for _ in range(n_layer)])   # 4 blocks, one after another
        self.ln_f = nn.LayerNorm(n_embd) if use_tricks else nn.Identity()           # final LayerNorm (Identity = "do nothing")
        self.lm_head = nn.Linear(n_embd, vocab_size)      # 32 -> 65 scores

    def forward(self, idx, targets=None):     # idx: (B, T)
        B, T = idx.shape                      # unpack
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T))   # WHAT + WHERE
        x = self.blocks(x)                    # 4 rounds of [communicate -> think]
        x = self.ln_f(x)                      # final normalize
        logits = self.lm_head(x)              # -> (B, T, 65)
        loss = None                           # no loss when generating
        if targets is not None:               # when training
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))   # measure error
        return logits, loss                   # return


def train(model, name):                       # helper: train and report val loss
    torch.manual_seed(1337)                   # same batches for both models
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)   # smaller step size (deeper models need gentler updates)
    params = sum(p.numel() for p in model.parameters())    # count learnable numbers
    print(f"\n--- {name}  ({params} parameters) ---")     # header
    for step in range(5001):                  # training loop
        xb, yb = get_batch("train")           # 1. batch
        _, loss = model(xb, yb)               # 2. forward + loss
        opt.zero_grad()                       # 3. clear gradients
        loss.backward()                       # 4. backprop
        opt.step()                            # 5. update
        if step % 1000 == 0:                  # every 1000 steps
            with torch.no_grad():             # exam mode
                vl = sum(model(*get_batch("val"))[1].item() for _ in range(20)) / 20   # avg val loss
            print(f"step {step:5d} | train loss {loss.item():.3f} | val loss {vl:.3f}")   # progress
    return vl                                 # final val loss


# =====================================================================
# PART A: 4 stacked blocks — WITHOUT vs WITH the tricks
# =====================================================================
print("========== PART A: TRAINING ==========")              # heading
torch.manual_seed(1337)                       # same starting numbers
naive = GPT(use_tricks=False)                 # model 1: just stacked
loss_naive = train(naive, "4 blocks, NO residual / LayerNorm")   # train it
torch.manual_seed(1337)                       # same starting numbers
gpt = GPT(use_tricks=True)                    # model 2: the real GPT recipe
loss_gpt = train(gpt, "4 blocks, WITH residual + LayerNorm")     # train it

print("\nSUMMARY (val loss, lower = better):")               # summary
print("   Step 2  bigram                    ~2.46")          # history
print("   Step 5  1 head + position         ~2.41")          # history
print("   Step 6b 4 heads + ffwd (1 block)  ~2.21")          # history
print(f"   Step 7a 4 blocks, no tricks       {loss_naive:.2f}")   # new
print(f"   Step 7b 4 blocks, GPT recipe      {loss_gpt:.2f}")     # new


# =====================================================================
# PART B: text from our tiny GPT
# =====================================================================
print("\n========== PART B: GENERATED TEXT (tiny GPT) ==========")   # heading
torch.manual_seed(42)                         # fixed dice
idx = torch.tensor([encode("T")])             # start with "T"
with torch.no_grad():                         # no learning
    for _ in range(300):                      # 300 chars
        logits, _ = gpt(idx[:, -block_size:]) # last 8 chars
        probs = F.softmax(logits[0, -1], dim=-1)          # next-char probabilities
        next_id = torch.multinomial(probs, 1)             # dice roll
        idx = torch.cat([idx, next_id.view(1, 1)], dim=1) # append
print(decode(idx[0].tolist()))                # print text


# =====================================================================
# PART C: our tiny GPT vs a real one
# =====================================================================
print("\n========== PART C: TINY GPT vs REAL GPT ==========")   # heading
print(f"   {'':<22}{'ours':>10}{'GPT-2 small':>14}")              # table header
print(f"   {'tokens':<22}{'characters':>10}{'BPE (~50k)':>14}")    # tokenizer
print(f"   {'context (block_size)':<22}{block_size:>10}{1024:>14}")   # context window
print(f"   {'n_embd':<22}{n_embd:>10}{768:>14}")                   # vector size
print(f"   {'n_head':<22}{n_head:>10}{12:>14}")                    # heads per block
print(f"   {'n_layer (blocks)':<22}{n_layer:>10}{12:>14}")         # blocks
print(f"   {'parameters':<22}{sum(p.numel() for p in gpt.parameters()):>10}{'~124 million':>14}")   # size
print("   Same recipe. Only the numbers are bigger.")             # the point