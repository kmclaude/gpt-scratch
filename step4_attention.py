# =====================================================================
# STEP 4: SELF-ATTENTION (Q, K, V)  —  smart weights instead of equal weights
# Run:  python3 step4_attention.py      (takes ~30 seconds to 1 minute)
#
# Problem from Step 3: every previous character got EQUAL weight (plain average).
# Today's idea: let each character DECIDE how much weight to give each previous
# character, using Query, Key and Value. This is the heart of the transformer.
# =====================================================================

import math                                   # for math.sqrt and math.log
import torch                                  # main library: tensors + autograd
import torch.nn as nn                         # ready-made building blocks (Embedding, Linear, ...)
import torch.nn.functional as F               # plain functions (softmax, cross_entropy, ...)

torch.manual_seed(1337)                       # fix random numbers so your output matches mine
torch.set_printoptions(precision=2, sci_mode=False)   # print tensors with 2 decimals


# ---------------------------------------------------------------------
# Data setup (same as Step 3)
# ---------------------------------------------------------------------
text = open("input.txt", encoding="utf-8").read()       # read the whole Shakespeare file
chars = sorted(set(text))                     # unique characters = vocabulary (65)
vocab_size = len(chars)                       # 65
stoi = {ch: i for i, ch in enumerate(chars)}  # character -> ID
itos = {i: ch for i, ch in enumerate(chars)}  # ID -> character
encode = lambda s: [stoi[c] for c in s]       # text -> list of IDs
decode = lambda ids: "".join(itos[i] for i in ids)      # list of IDs -> text
data = torch.tensor(encode(text), dtype=torch.long)     # whole text as a tensor of IDs
n = int(0.9 * len(data))                      # 90% split point
train_data, val_data = data[:n], data[n:]     # train (90%) and validation (10%)

block_size = 8                                # context length: 8 characters
batch_size = 32                               # 32 chunks per training step
n_embd = 32                                   # each character = vector of 32 numbers
head_size = 16                                # size of each Query / Key / Value vector


def get_batch(split):                         # returns one batch of inputs (x) and answers (y)
    d = train_data if split == "train" else val_data           # choose train or val
    ix = torch.randint(len(d) - block_size, (batch_size,))     # 32 random start positions
    x = torch.stack([d[i : i + block_size] for i in ix])       # inputs, shape (32, 8)
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])   # answers = shifted by 1
    return x, y                               # give back both


# =====================================================================
# THE NEW PART: one attention head
# =====================================================================
class Head(nn.Module):                        # one self-attention "head"
    def __init__(self):                       # constructor: create the learnable parts
        super().__init__()                    # required by torch
        self.query = nn.Linear(n_embd, head_size, bias=False)  # lens 1: char vector -> Query ("what am I looking for?")
        self.key   = nn.Linear(n_embd, head_size, bias=False)  # lens 2: char vector -> Key   ("what do I contain?")
        self.value = nn.Linear(n_embd, head_size, bias=False)  # lens 3: char vector -> Value ("what I will hand over")
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))  # 8x8 lower-triangle (no future)
        self.last_weights = None              # we save the weights here so we can print them later

    def forward(self, x):                     # x = character vectors, shape (B, T, 32)
        B, T, C = x.shape                     # B = batch, T = number of chars, C = 32
        q = self.query(x)                     # every char makes its Query  -> (B, T, 16)
        k = self.key(x)                       # every char makes its Key    -> (B, T, 16)
        v = self.value(x)                     # every char makes its Value  -> (B, T, 16)

        # STEP 1: score = Query . Key   (how well does my question match your label?)
        scores = q @ k.transpose(-2, -1)      # (B,T,16) @ (B,16,T) -> (B, T, T): every char vs every char
        scores = scores / math.sqrt(head_size)    # shrink the numbers so softmax doesn't get too extreme

        # STEP 1b: hide the future (same trick as Step 3)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))   # future -> -infinity

        # STEP 2: softmax -> weights (each row adds up to 1)
        weights = F.softmax(scores, dim=-1)   # (B, T, T)  <- in Step 3 these were all EQUAL; now they are LEARNED
        self.last_weights = weights.detach()  # save a copy for printing (detach = "not part of training")

        # STEP 3: weighted mix of the Values
        return weights @ v                    # (B,T,T) @ (B,T,16) -> (B, T, 16)


class AttentionModel(nn.Module):              # the full model
    def __init__(self):                       # constructor
        super().__init__()                    # required by torch
        self.tok_emb = nn.Embedding(vocab_size, n_embd)   # char ID -> 32-number vector
        self.head = Head()                    # ONE attention head (the only change vs Step 3!)
        self.lm_head = nn.Linear(head_size, vocab_size)   # 16-number vector -> 65 scores

    def forward(self, idx, targets=None):     # runs when you call model(x) or model(x, y)
        B, T = idx.shape                      # batch size, number of chars
        x = self.tok_emb(idx)                 # (B, T, 32)
        x = self.head(x)                      # attention: each char gathers info from previous chars -> (B, T, 16)
        logits = self.lm_head(x)              # (B, T, 65) scores for the next char
        loss = None                           # no loss when just generating
        if targets is not None:               # if answers given, measure the error
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))   # flatten and compare
        return logits, loss                   # give back scores and loss


# =====================================================================
# PART A: train it
# =====================================================================
print("========== PART A: TRAINING ==========")              # section heading
model = AttentionModel()                      # create the model with random numbers
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)   # the "mechanic"
print("Parameters:", sum(p.numel() for p in model.parameters()))   # how many learnable numbers

for step in range(5001):                      # 5001 training steps
    xb, yb = get_batch("train")               # 1. new batch
    _, loss = model(xb, yb)                   # 2. predict + measure loss
    optimizer.zero_grad()                     # 3. clear old gradients
    loss.backward()                           # 4. compute gradients
    optimizer.step()                          # 5. update numbers
    if step % 1000 == 0:                      # every 1000 steps
        with torch.no_grad():                 # exam mode
            vl = sum(model(*get_batch("val"))[1].item() for _ in range(20)) / 20   # avg val loss
        print(f"step {step:5d} | train loss {loss.item():.3f} | val loss {vl:.3f}")   # show progress

print("\nCompare:  Step 2 (bigram) ~2.46   |   Step 3 (average) ~2.86")   # previous results


# =====================================================================
# PART B: are Step 3's problems fixed?
# =====================================================================
print("\n========== PART B: STEP 3 PROBLEMS — FIXED? ==========")   # section heading


def show(ctx):                                # print top-3 guesses after a given text
    with torch.no_grad():                     # no learning
        logits, _ = model(torch.tensor([encode(ctx)]))    # run the model
        probs = F.softmax(logits[0, -1], dim=-1)          # probabilities for the next char
    p, i = torch.topk(probs, 3)               # top 3
    s = "   ".join(f"{itos[j.item()]!r}:{q.item()*100:4.1f}%" for q, j in zip(p, i))   # format nicely
    print(f"   context {ctx!r:>10}  ->  {s}") # print one line


print("Problem 2 from Step 3 (last char drowned out):")    # label
show("q")                                     # alone
show("aaaaaaaq")                              # q after many a's -> should still say 'u' now
print("\nProblem 1 from Step 3 (th vs ht):")  # label
show("th")                                    # t then h
show("ht")                                    # h then t


# =====================================================================
# PART C: LOOK INSIDE — what did the attention learn?
# =====================================================================
print("\n========== PART C: ATTENTION WEIGHTS ==========")    # section heading


def attention_of_last(ctx):                   # show how much the LAST char attends to each char
    with torch.no_grad():                     # no learning
        model(torch.tensor([encode(ctx)]))    # run once so the head saves its weights
    w = model.head.last_weights[0, -1]        # weights row of the last position, shape (T,)
    print(f"\n   context {ctx!r}: last char {ctx[-1]!r} gives attention to:")   # label
    for ch, wt in zip(ctx, w):                # walk through each char and its weight
        bar = "#" * int(wt.item() * 40)       # simple text bar chart
        print(f"      {ch!r:>5}  {wt.item()*100:5.1f}%  {bar}")   # one line per char


attention_of_last("aaaaaaaq")                 # does 'q' look mostly at itself?
attention_of_last("the cat")                  # what does 't' look at?


# =====================================================================
# PART D: text written by the model
# =====================================================================
print("\n========== PART D: GENERATED TEXT ==========")       # section heading
idx = torch.tensor([encode("T")])             # start with "T"
with torch.no_grad():                         # no learning
    for _ in range(200):                      # 200 new chars
        logits, _ = model(idx[:, -block_size:])          # last 8 chars only
        probs = F.softmax(logits[0, -1], dim=-1)          # next-char probabilities
        next_id = torch.multinomial(probs, 1)             # roll the dice
        idx = torch.cat([idx, next_id.view(1, 1)], dim=1) # append
print(decode(idx[0].tolist()))                # print the text


# =====================================================================
# PART E: THE PROBLEM THAT IS STILL LEFT
# =====================================================================
print("\n========== PART E: REMAINING PROBLEM ==========")    # section heading
attention_of_last("a b a b")                  # the two 'a's are at different distances...
print("\n   Both 'a's get EXACTLY the same weight, and so do both 'b's.")   # observation
print("   The model can't tell 'near' from 'far'. Why? Think...")           # the question for Step 5