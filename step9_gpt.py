# =====================================================================
# STEP 9: SAMPLING  —  how the model CHOOSES the next character
# Run:  python3 step9_sampling.py      (takes a few seconds — NO training)
#
# The model only gives PROBABILITIES. Something else must pick one character.
# That "something" is sampling, and these are the knobs you see in every LLM API:
#   temperature  -> make the probabilities sharper (safe) or flatter (creative)
#   top_k        -> keep only the K most likely characters, drop the rest
#   top_p        -> keep the smallest group of characters whose total is >= p
# Needs: tiny_gpt_ctx32.pt  (saved by Step 8)
# =====================================================================

import math                                   # for math.sqrt
import torch                                  # main library
import torch.nn as nn                         # building blocks
import torch.nn.functional as F               # softmax etc.


# ---------------------------------------------------------------------
# Load the saved model from Step 8
# ---------------------------------------------------------------------
ckpt = torch.load("tiny_gpt_ctx32.pt")        # read the file Step 8 saved
chars = ckpt["chars"]                         # vocabulary saved with the model
block_size = ckpt["block_size"]               # 32
vocab_size = len(chars)                       # 65
stoi = {ch: i for i, ch in enumerate(chars)}  # char -> ID
itos = {i: ch for i, ch in enumerate(chars)}  # ID -> char
encode = lambda s: [stoi[c] for c in s]       # text -> IDs
decode = lambda ids: "".join(itos[i] for i in ids)      # IDs -> text

n_embd, n_head, n_layer = 32, 4, 4            # must match Step 8 exactly, or the numbers won't fit
head_size = n_embd // n_head                  # 8


# ---------------------------------------------------------------------
# Same model classes as Step 8 (copied, so this file runs on its own)
# ---------------------------------------------------------------------
class Head(nn.Module):                        # one attention head
    def __init__(self, block_size):           # constructor
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
        return F.softmax(scores, dim=-1) @ v  # weights -> mix of Values


class MultiHeadAttention(nn.Module):          # several heads side by side
    def __init__(self, block_size):           # constructor
        super().__init__()                    # required by torch
        self.heads = nn.ModuleList([Head(block_size) for _ in range(n_head)])   # 4 heads
        self.proj = nn.Linear(n_embd, n_embd) # blend heads

    def forward(self, x):                     # x: (B, T, 32)
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))   # run, join, blend


class FeedForward(nn.Module):                 # each char "thinks" alone
    def __init__(self):                       # constructor
        super().__init__()                    # required by torch
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))   # 32 -> 128 -> 32

    def forward(self, x):                     # x: (B, T, 32)
        return self.net(x)                    # (B, T, 32)


class Block(nn.Module):                       # one transformer block
    def __init__(self, block_size):           # constructor
        super().__init__()                    # required by torch
        self.attn = MultiHeadAttention(block_size)   # communicate
        self.ffwd = FeedForward()             # think
        self.ln1 = nn.LayerNorm(n_embd)       # normalize before attention
        self.ln2 = nn.LayerNorm(n_embd)       # normalize before feed-forward

    def forward(self, x):                     # x: (B, T, 32)
        x = x + self.attn(self.ln1(x))        # residual + attention
        return x + self.ffwd(self.ln2(x))     # residual + feed-forward


class GPT(nn.Module):                         # the whole model
    def __init__(self, block_size):           # constructor
        super().__init__()                    # required by torch
        self.block_size = block_size          # memory length
        self.tok_emb = nn.Embedding(vocab_size, n_embd)   # WHAT
        self.pos_emb = nn.Embedding(block_size, n_embd)   # WHERE
        self.blocks = nn.Sequential(*[Block(block_size) for _ in range(n_layer)])   # 4 blocks
        self.ln_f = nn.LayerNorm(n_embd)      # final normalize
        self.lm_head = nn.Linear(n_embd, vocab_size)      # 32 -> 65 scores

    def forward(self, idx):                   # idx: (B, T)   (no training today, so no targets)
        T = idx.shape[1]                      # sequence length
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T))   # WHAT + WHERE
        return self.lm_head(self.ln_f(self.blocks(x)))          # -> (B, T, 65) raw scores ("logits")


model = GPT(block_size)                       # empty model with the right shape
model.load_state_dict(ckpt["state"])          # pour the trained numbers into it
model.eval()                                  # "exam mode"
print("Loaded tiny_gpt_ctx32.pt  (no training today)")   # confirm


# =====================================================================
# THE SAMPLER  —  this is the whole topic of today, in one function
# =====================================================================
def next_char_probs(logits, temperature=1.0, top_k=None, top_p=None):   # logits: 65 raw scores
    if temperature == 0:                      # temperature 0 = "greedy": always the #1 choice
        probs = torch.zeros_like(logits)      # all zeros...
        probs[logits.argmax()] = 1.0          # ...except 100% on the best one
        return probs                          # done
    logits = logits / temperature             # KNOB 1: divide scores. <1 = gaps grow (sharp), >1 = gaps shrink (flat)
    if top_k is not None:                     # KNOB 2: top-k
        kth_best = torch.topk(logits, top_k).values[-1]          # score of the K-th best char
        logits = logits.masked_fill(logits < kth_best, float("-inf"))   # everything below it -> impossible
    probs = F.softmax(logits, dim=-1)         # scores -> percentages (total 100%)
    if top_p is not None:                     # KNOB 3: top-p (a.k.a. "nucleus")
        sorted_p, order = torch.sort(probs, descending=True)     # biggest first
        running = torch.cumsum(sorted_p, dim=0)                  # running total: 40%, 65%, 80%, 91%...
        drop = running - sorted_p >= top_p    # drop a char if the ones BEFORE it already reached p
        sorted_p[drop] = 0.0                  # remove them
        probs = torch.zeros_like(probs).scatter(0, order, sorted_p)   # put back in original order
        probs = probs / probs.sum()           # re-scale so the survivors add up to 100% again
    return probs                              # final percentages to roll the dice on


def show(probs, title, n=8):                  # helper: print the top-n chars as a bar chart
    print(f"\n   {title}")                    # heading line
    top = torch.topk(probs, n)                # n most likely chars
    for p, i in zip(top.values, top.indices): # each one
        ch = repr(itos[i.item()])             # repr so ' ' and '\n' are visible
        bar = "#" * int(p.item() * 50)        # 50 chars = 100%
        print(f"     {ch:>6} {p.item()*100:5.1f}% {bar}")   # row
    alive = (probs > 0).sum().item()          # how many chars can still be picked at all
    print(f"     -> {alive} of {vocab_size} chars can still be picked")   # survivors


prompt = "KING RICHARD III:\nNow is the winter of our d"   # our test prompt
with torch.no_grad():                         # no learning
    logits = model(torch.tensor([encode(prompt)])[:, -block_size:])[0, -1]   # 65 raw scores for the next char
print(f"Prompt: {prompt!r}")                  # show prompt


# =====================================================================
# PART A: temperature — same scores, different sharpness
# =====================================================================
print("\n========== PART A: TEMPERATURE ==========")   # heading
for t in [0.2, 1.0, 1.5]:                     # cold, normal, hot
    show(next_char_probs(logits, temperature=t), f"temperature = {t}")   # print each


# =====================================================================
# PART B: top-k and top-p — cut off the long tail of bad choices
# =====================================================================
print("\n========== PART B: TOP-K and TOP-P (temperature 1.0) ==========")   # heading
show(next_char_probs(logits, top_k=3), "top_k = 3   (only the 3 best survive, always exactly 3)")   # top-k
show(next_char_probs(logits, top_p=0.9), "top_p = 0.9 (keep best chars until they cover 90%)")       # top-p


# =====================================================================
# PART C: the same knobs on real generated text
# =====================================================================
def generate(n_chars=200, **knobs):           # helper: write text using the given knobs
    torch.manual_seed(42)                     # SAME dice for every setting, so only the knob differs
    idx = torch.tensor([encode("ROMEO:\n")])  # start every sample with the same prompt
    with torch.no_grad():                     # no learning
        for _ in range(n_chars):              # one char at a time
            logits = model(idx[:, -block_size:])[0, -1]           # scores for the next char
            probs = next_char_probs(logits, **knobs)             # apply temperature / top-k / top-p
            next_id = torch.multinomial(probs, 1)                # roll the dice
            idx = torch.cat([idx, next_id.view(1, 1)], dim=1)    # append
    return decode(idx[0].tolist())            # IDs -> text


print("\n========== PART C: GENERATED TEXT, ONE KNOB AT A TIME ==========")   # heading
settings = [                                  # (label, knobs)
    ("temperature 0  (greedy, always #1)", dict(temperature=0)),
    ("temperature 0.5", dict(temperature=0.5)),
    ("temperature 1.0 (raw model)", dict(temperature=1.0)),
    ("temperature 1.8", dict(temperature=1.8)),
    ("temperature 1.8 + top_p 0.9", dict(temperature=1.8, top_p=0.9)),
]
for label, knobs in settings:                 # run each setting
    print(f"\n----- {label} -----")           # label
    print(generate(**knobs))                  # text