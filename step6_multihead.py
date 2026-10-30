# =====================================================================
# STEP 6: MULTI-HEAD ATTENTION + FEED-FORWARD
# Run:  python3 step6_multihead.py      (takes ~1.5 to 3 minutes — it trains 2 models)
#
# Problem from Step 5: ONE head = each character can ask only ONE question.
# Fix 1 (multi-head):   run several small heads side by side, each asks its own question.
# Fix 2 (feed-forward): after gathering info, let each character "think" about it alone.
# =====================================================================

import math                                   # for math.sqrt
import torch                                  # main library: tensors + autograd
import torch.nn as nn                         # ready-made building blocks
import torch.nn.functional as F               # plain functions (softmax, cross_entropy, relu)

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

block_size = 8                                # context length
batch_size = 32                               # chunks per step
n_embd = 32                                   # vector size per char
n_head = 4                                    # NEW: number of heads running side by side
head_size = n_embd // n_head                  # NEW: each head is smaller: 32 / 4 = 8 numbers


def get_batch(split):                         # one batch of inputs + answers
    d = train_data if split == "train" else val_data           # train or val
    ix = torch.randint(len(d) - block_size, (batch_size,))     # random starts
    x = torch.stack([d[i : i + block_size] for i in ix])       # inputs (32, 8)
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])   # answers (shifted by 1)
    return x, y                               # return both


# =====================================================================
# Building block 1: ONE head (same as Step 5)
# =====================================================================
class Head(nn.Module):                        # one attention head = one "question"
    def __init__(self):                       # constructor
        super().__init__()                    # required by torch
        self.query = nn.Linear(n_embd, head_size, bias=False)  # 32 -> 8: Query
        self.key   = nn.Linear(n_embd, head_size, bias=False)  # 32 -> 8: Key
        self.value = nn.Linear(n_embd, head_size, bias=False)  # 32 -> 8: Value
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))  # no-future mask
        self.last_weights = None              # saved for printing

    def forward(self, x):                     # x: (B, T, 32)
        B, T, C = x.shape                     # unpack shape
        q, k, v = self.query(x), self.key(x), self.value(x)   # Q, K, V for every char: (B, T, 8) each
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_size)   # Query . Key -> (B, T, T)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))   # hide the future
        weights = F.softmax(scores, dim=-1)   # -> percentages
        self.last_weights = weights.detach()  # keep for printing
        return weights @ v                    # mix of Values -> (B, T, 8)


# =====================================================================
# Building block 2: MULTI-HEAD = several heads side by side   (NEW)
# =====================================================================
class MultiHeadAttention(nn.Module):          # 4 heads working in parallel
    def __init__(self):                       # constructor
        super().__init__()                    # required by torch
        self.heads = nn.ModuleList([Head() for _ in range(n_head)])   # a list of 4 separate heads (JS: Array.from({length: 4}, () => new Head()))

    def forward(self, x):                     # x: (B, T, 32)
        outputs = [h(x) for h in self.heads]  # run every head on the SAME input -> 4 results, each (B, T, 8)
        return torch.cat(outputs, dim=-1)     # glue them side by side: 8+8+8+8 = 32 -> (B, T, 32)


# =====================================================================
# Building block 3: FEED-FORWARD = each char "thinks" alone   (NEW)
# =====================================================================
class FeedForward(nn.Module):                 # a small 2-layer network, applied to each char separately
    def __init__(self):                       # constructor
        super().__init__()                    # required by torch
        self.net = nn.Sequential(             # Sequential = run these layers one after another
            nn.Linear(n_embd, 4 * n_embd),    # expand: 32 -> 128 numbers (more room to think)
            nn.ReLU(),                        # ReLU: negative numbers -> 0 (this makes it non-linear / "able to decide")
            nn.Linear(4 * n_embd, n_embd),    # shrink back: 128 -> 32 numbers
        )

    def forward(self, x):                     # x: (B, T, 32)
        return self.net(x)                    # same shape out: (B, T, 32)


# =====================================================================
# The model: Step 5 + multi-head (+ optional feed-forward)
# =====================================================================
class Model(nn.Module):                       # the full model
    def __init__(self, use_ffwd):             # use_ffwd = True/False, so we can compare
        super().__init__()                    # required by torch
        self.tok_emb = nn.Embedding(vocab_size, n_embd)   # WHAT
        self.pos_emb = nn.Embedding(block_size, n_embd)   # WHERE
        self.attn = MultiHeadAttention()      # 4 heads instead of 1   <-- NEW
        self.ffwd = FeedForward() if use_ffwd else None   # optional "thinking" step   <-- NEW
        self.lm_head = nn.Linear(n_embd, vocab_size)      # 32 -> 65 scores

    def forward(self, idx, targets=None):     # idx: (B, T)
        B, T = idx.shape                      # unpack
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T))   # WHAT + WHERE -> (B, T, 32)
        x = self.attn(x)                      # STEP 1: COMMUNICATE — gather info from previous chars
        if self.ffwd is not None:             # if feed-forward is switched on...
            x = self.ffwd(x)                  # STEP 2: THINK — each char processes what it gathered
        logits = self.lm_head(x)              # -> (B, T, 65)
        loss = None                           # no loss when generating
        if targets is not None:               # when training
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))   # measure error
        return logits, loss                   # return


def train(model, name):                       # helper: train a model and report its val loss
    torch.manual_seed(1337)                   # same batches for every model = fair comparison
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)   # optimizer
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
# PART A: train two models and compare
# =====================================================================
print("========== PART A: TRAINING ==========")              # heading
torch.manual_seed(1337)                       # same starting numbers
model_mh = Model(use_ffwd=False)              # model 1: 4 heads, no feed-forward
loss_mh = train(model_mh, "4 heads")          # train it
torch.manual_seed(1337)                       # same starting numbers
model_ff = Model(use_ffwd=True)               # model 2: 4 heads + feed-forward
loss_ff = train(model_ff, "4 heads + feed-forward")   # train it

print("\nSUMMARY (val loss, lower = better):")               # summary table
print("   Step 2  bigram            ~2.46")                  # history
print("   Step 3  average           ~2.86")                  # history
print("   Step 4  1 head            ~2.46")                  # history
print("   Step 5  1 head + position ~2.41")                  # history
print(f"   Step 6a 4 heads           {loss_mh:.2f}")         # new
print(f"   Step 6b 4 heads + ffwd    {loss_ff:.2f}")         # new


# =====================================================================
# PART B: LOOK INSIDE — does each head ask a DIFFERENT question?
# =====================================================================
print("\n========== PART B: WHAT DOES EACH HEAD LOOK AT? ==========")   # heading
ctx = "KING: Th"                              # a context of exactly 8 chars
with torch.no_grad():                         # no learning
    model_ff(torch.tensor([encode(ctx)]))     # run once so every head saves its weights

print(f"   context {ctx!r} — where does the LAST char {ctx[-1]!r} look, in each head?\n")   # label
header = "   " + "".join(f"{c!r:>7}" for c in ctx)                     # column labels = the chars
print(header)                                 # print header row
for h_i, head in enumerate(model_ff.attn.heads):          # loop over the 4 heads
    w = head.last_weights[0, -1]              # this head's weights for the last char
    row = "".join(f"{v.item()*100:6.0f}%" for v in w)      # format as percentages
    print(f"   head {h_i}{row}")              # one row per head
print("\n   Each row adds up to 100%. Different rows = different heads = different questions.")   # note


# =====================================================================
# PART C: text from the best model
# =====================================================================
print("\n========== PART C: GENERATED TEXT (4 heads + ffwd) ==========")   # heading
torch.manual_seed(42)                         # fixed dice
idx = torch.tensor([encode("T")])             # start with "T"
with torch.no_grad():                         # no learning
    for _ in range(250):                      # 250 chars
        logits, _ = model_ff(idx[:, -block_size:])        # last 8 chars
        probs = F.softmax(logits[0, -1], dim=-1)          # next-char probabilities
        next_id = torch.multinomial(probs, 1)             # dice roll
        idx = torch.cat([idx, next_id.view(1, 1)], dim=1) # append
print(decode(idx[0].tolist()))                # print text


# =====================================================================
# PART D: what's next
# =====================================================================
print("\n========== PART D: NEXT IDEA ==========")            # heading
print("   One round of [communicate -> think] helped.")       # observation
print("   Real GPTs repeat this round MANY times (12, 32, 96+ layers).")   # fact
print("   But if you just stack them, training gets unstable. Step 7 fixes that.")   # teaser