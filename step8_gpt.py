# =====================================================================
# STEP 8: CONTEXT WINDOW  —  how much can the model "remember"?
# Run:  python3 step8_context.py      (takes ~4 to 8 minutes — it trains 2 models)
#
# In Step 7 the model saw only the LAST 8 characters (block_size = 8).
# Today: train the SAME tiny GPT twice, once with 8 chars of memory, once with 32.
# Then look at:  loss,  what each model "sees",  loss by position,  generated text.
# Both trained models are SAVED to files, so Step 9 (sampling) can load them
# without training again.
# =====================================================================

import math                                   # for math.sqrt
import time                                   # to measure training time
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

batch_size = 32                               # chunks per step
n_embd = 32                                   # vector size per char
n_head = 4                                    # heads per block
head_size = n_embd // n_head                  # 8 numbers per head
n_layer = 4                                   # blocks stacked
train_steps = 5000                            # same training length for both models


def get_batch(split, block_size):             # NEW: block_size is now an argument (it differs per model)
    d = train_data if split == "train" else val_data           # train or val
    ix = torch.randint(len(d) - block_size, (batch_size,))     # random starts
    x = torch.stack([d[i : i + block_size] for i in ix])       # inputs (32, block_size)
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])   # answers (shifted by 1)
    return x, y                               # return both


# =====================================================================
# Same tiny GPT as Step 7 (GPT recipe only). Only change: block_size is passed in.
# =====================================================================
class Head(nn.Module):                        # one attention head
    def __init__(self, block_size):           # block_size needed for the mask size
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
    def __init__(self, block_size):           # pass block_size down to each head
        super().__init__()                    # required by torch
        self.heads = nn.ModuleList([Head(block_size) for _ in range(n_head)])   # 4 heads
        self.proj = nn.Linear(n_embd, n_embd) # blend the heads' answers

    def forward(self, x):                     # x: (B, T, 32)
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # run all heads, join
        return self.proj(out)                 # blend


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


class Block(nn.Module):                       # one transformer block (GPT recipe)
    def __init__(self, block_size):           # pass block_size down
        super().__init__()                    # required by torch
        self.attn = MultiHeadAttention(block_size)   # communicate
        self.ffwd = FeedForward()             # think
        self.ln1 = nn.LayerNorm(n_embd)       # normalize before attention
        self.ln2 = nn.LayerNorm(n_embd)       # normalize before feed-forward

    def forward(self, x):                     # x: (B, T, 32)
        x = x + self.attn(self.ln1(x))        # residual + attention
        x = x + self.ffwd(self.ln2(x))        # residual + feed-forward
        return x                              # (B, T, 32)


class GPT(nn.Module):                         # the whole model
    def __init__(self, block_size):           # NEW: memory length is a setting
        super().__init__()                    # required by torch
        self.block_size = block_size          # remember it (used when generating)
        self.tok_emb = nn.Embedding(vocab_size, n_embd)   # WHAT
        self.pos_emb = nn.Embedding(block_size, n_embd)   # WHERE (one vector per position, so 8 or 32 rows)
        self.blocks = nn.Sequential(*[Block(block_size) for _ in range(n_layer)])   # 4 blocks
        self.ln_f = nn.LayerNorm(n_embd)      # final normalize
        self.lm_head = nn.Linear(n_embd, vocab_size)      # 32 -> 65 scores

    def forward(self, idx, targets=None, reduce=True):   # reduce=False -> return loss for EVERY position
        B, T = idx.shape                      # unpack
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T))   # WHAT + WHERE
        x = self.blocks(x)                    # 4 rounds of [communicate -> think]
        x = self.ln_f(x)                      # final normalize
        logits = self.lm_head(x)              # -> (B, T, 65)
        loss = None                           # no loss when generating
        if targets is not None:               # when training / evaluating
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T),
                                   reduction="mean" if reduce else "none")   # one number, or one per position
        return logits, loss                   # return


def train(block_size):                        # build + train one model, return it and its val loss
    torch.manual_seed(1337)                   # same starting numbers for both
    model = GPT(block_size)                   # new model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)   # optimizer
    params = sum(p.numel() for p in model.parameters())    # count learnable numbers
    print(f"\n--- context = {block_size} chars  ({params} parameters) ---")   # header
    t0 = time.time()                          # start timer
    for step in range(train_steps + 1):       # training loop
        xb, yb = get_batch("train", block_size)   # 1. batch
        _, loss = model(xb, yb)               # 2. forward + loss
        opt.zero_grad()                       # 3. clear gradients
        loss.backward()                       # 4. backprop
        opt.step()                            # 5. update
        if step % 1000 == 0:                  # every 1000 steps
            with torch.no_grad():             # exam mode
                vl = sum(model(*get_batch("val", block_size))[1].item() for _ in range(50)) / 50   # avg val loss
            print(f"step {step:5d} | train loss {loss.item():.3f} | val loss {vl:.3f}")   # progress
    print(f"   training time: {time.time() - t0:.0f} seconds")   # how long it took
    return model, vl                          # trained model + final val loss


def save(model, path):                        # save a trained model to a file (for Step 9)
    torch.save({"state": model.state_dict(),  # all the learned numbers
                "block_size": model.block_size,   # memory length
                "chars": chars}, path)        # vocabulary, so we can encode/decode later
    print(f"   saved -> {path}")              # confirm


# =====================================================================
# PART A: train both models
# =====================================================================
print("========== PART A: TRAINING (same model, different memory) ==========")   # heading
model8, loss8 = train(8)                      # short memory
save(model8, "tiny_gpt_ctx8.pt")              # save it
model32, loss32 = train(32)                   # long memory
save(model32, "tiny_gpt_ctx32.pt")            # save it

print("\nSUMMARY (val loss, lower = better):")                   # summary
print(f"   context  8 chars   {loss8:.2f}")                       # short memory result
print(f"   context 32 chars   {loss32:.2f}")                      # long memory result
print("   Note: parameters went up only a little (more rows in pos_emb).")   # the cost of memory here


# =====================================================================
# PART B: what does each model actually SEE?
# =====================================================================
print("\n========== PART B: WHAT EACH MODEL SEES ==========")    # heading
prompt = "KING RICHARD III:\nNow is the winter of our disc"     # a prompt longer than both windows
print(f"   full prompt ({len(prompt)} chars): {prompt!r}")       # !r shows \n visibly
print(f"   ctx  8 sees : {prompt[-8:]!r}")                       # only the last 8 chars
print(f"   ctx 32 sees : {prompt[-32:]!r}")                      # the last 32 chars
print("   Everything outside the window is INVISIBLE to the model. It is simply cut off.")   # the point
print("   (One example is too noisy to judge a model. PART C measures it properly, over thousands of guesses.)")   # honest note


# =====================================================================
# PART C: loss by position — does more context actually help?
# =====================================================================
print("\n========== PART C: LOSS BY POSITION (ctx 32 model) ==========")   # heading
print("   Position 1 = model has seen 1 char. Position 32 = model has seen 32 chars.")   # explanation
torch.manual_seed(7)                          # fixed val batches
with torch.no_grad():                         # no learning
    per_pos = torch.zeros(32)                 # running total of loss at each position
    for _ in range(50):                       # 50 val batches
        xb, yb = get_batch("val", 32)         # (32, 32)
        _, l = model32(xb, yb, reduce=False)  # loss for every (row, position)
        per_pos += l.view(batch_size, 32).mean(0)   # average over rows, add per position
    per_pos /= 50                             # average over batches
for p in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]:  # a few positions
    bar = "#" * int(per_pos[p - 1].item() * 12)   # simple text bar chart
    print(f"   seen {p:2d} chars | loss {per_pos[p - 1].item():.2f} | {bar}")   # print row
print("   Shorter bar = better guess. Watch how fast it drops, and where it flattens.")   # hint


# =====================================================================
# PART D: generated text from both (same dice)
# =====================================================================
def generate(m, n_chars=250):                 # helper: generate text from a model
    torch.manual_seed(42)                     # same dice for both models
    idx = torch.tensor([encode("\n")])        # start with a new line
    with torch.no_grad():                     # no learning
        for _ in range(n_chars):              # one char at a time
            logits, _ = m(idx[:, -m.block_size:])         # only the last block_size chars
            probs = F.softmax(logits[0, -1], dim=-1)      # next-char probabilities
            next_id = torch.multinomial(probs, 1)         # dice roll
            idx = torch.cat([idx, next_id.view(1, 1)], dim=1)   # append
    return decode(idx[0].tolist())            # IDs -> text


print("\n========== PART D: GENERATED TEXT ==========")       # heading
print("----- ctx 8 -----")                                    # label
print(generate(model8))                                       # short memory text
print("\n----- ctx 32 -----")                                 # label
print(generate(model32))                                      # long memory text