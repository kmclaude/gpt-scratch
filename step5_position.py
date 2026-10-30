# =====================================================================
# STEP 5: POSITION EMBEDDINGS  —  teach the model WHERE each character is
# Run:  python3 step5_position.py      (takes ~30 seconds to 1.5 minutes)
#
# Problem from Step 4: in "a b a b" both 'b's got EXACTLY the same attention.
# Attention only sees WHAT a character is, not WHERE it is (near or far).
# Today's fix: give every position (1st, 2nd, 3rd...) its own learned vector,
# and ADD it to the character's vector. Only 2 lines change vs Step 4.
# =====================================================================

import math                                   # for math.sqrt
import torch                                  # main library: tensors + autograd
import torch.nn as nn                         # ready-made building blocks
import torch.nn.functional as F               # plain functions (softmax, cross_entropy)

torch.manual_seed(1337)                       # fix random numbers so your output matches mine
torch.set_printoptions(precision=2, sci_mode=False)   # print tensors with 2 decimals


# ---------------------------------------------------------------------
# Data setup (same as Step 4)
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
head_size = 16                                # Q / K / V size


def get_batch(split):                         # one batch of inputs + answers
    d = train_data if split == "train" else val_data           # train or val
    ix = torch.randint(len(d) - block_size, (batch_size,))     # random starts
    x = torch.stack([d[i : i + block_size] for i in ix])       # inputs (32, 8)
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])   # answers (shifted by 1)
    return x, y                               # return both


class Head(nn.Module):                        # one attention head (EXACTLY the same as Step 4)
    def __init__(self):                       # constructor
        super().__init__()                    # required by torch
        self.query = nn.Linear(n_embd, head_size, bias=False)  # vector -> Query
        self.key   = nn.Linear(n_embd, head_size, bias=False)  # vector -> Key
        self.value = nn.Linear(n_embd, head_size, bias=False)  # vector -> Value
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))  # no-future mask
        self.last_weights = None              # saved for printing

    def forward(self, x):                     # x: (B, T, 32)
        B, T, C = x.shape                     # unpack shape
        q, k, v = self.query(x), self.key(x), self.value(x)   # make Q, K, V for every char
        scores = q @ k.transpose(-2, -1) / math.sqrt(head_size)   # Query . Key, scaled
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))   # hide the future
        weights = F.softmax(scores, dim=-1)   # scores -> weights (rows add to 1)
        self.last_weights = weights.detach()  # keep a copy to print later
        return weights @ v                    # weighted mix of Values -> (B, T, 16)


class PositionModel(nn.Module):               # Step 4 model + position embeddings
    def __init__(self):                       # constructor
        super().__init__()                    # required by torch
        self.tok_emb = nn.Embedding(vocab_size, n_embd)   # WHAT:  char ID  -> 32 numbers
        self.pos_emb = nn.Embedding(block_size, n_embd)   # WHERE: position (0..7) -> 32 numbers   <-- NEW
        self.head = Head()                    # one attention head
        self.lm_head = nn.Linear(head_size, vocab_size)   # 16 numbers -> 65 scores

    def forward(self, idx, targets=None):     # idx: (B, T) char IDs
        B, T = idx.shape                      # batch size, number of chars
        tok = self.tok_emb(idx)               # (B, T, 32)  "what is this char"
        pos = self.pos_emb(torch.arange(T))   # (T, 32)     "where is it": arange(T) = [0, 1, ..., T-1]   <-- NEW
        x = tok + pos                         # (B, T, 32)  each char vector now carries WHAT + WHERE  <-- NEW
        x = self.head(x)                      # attention (now it can see positions too)
        logits = self.lm_head(x)              # (B, T, 65)
        loss = None                           # no loss when generating
        if targets is not None:               # when training
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))   # measure error
        return logits, loss                   # return scores + loss


# =====================================================================
# PART A: train it
# =====================================================================
print("========== PART A: TRAINING ==========")              # heading
model = PositionModel()                       # new model, random numbers
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)   # the "mechanic"
print("Parameters:", sum(p.numel() for p in model.parameters()))   # count learnable numbers

for step in range(5001):                      # training loop
    xb, yb = get_batch("train")               # 1. batch
    _, loss = model(xb, yb)                   # 2. forward + loss
    optimizer.zero_grad()                     # 3. clear gradients
    loss.backward()                           # 4. backprop
    optimizer.step()                          # 5. update
    if step % 1000 == 0:                      # every 1000 steps
        with torch.no_grad():                 # exam mode
            vl = sum(model(*get_batch("val"))[1].item() for _ in range(20)) / 20   # avg val loss
        print(f"step {step:5d} | train loss {loss.item():.3f} | val loss {vl:.3f}")   # progress

print("\nCompare:  Step 2 ~2.46  |  Step 3 ~2.86  |  Step 4 ~2.46")   # history


# =====================================================================
# PART B: is the Step 4 problem fixed?
# =====================================================================
print("\n========== PART B: NEAR vs FAR ==========")           # heading


def attention_of_last(ctx):                   # show where the LAST char looks
    with torch.no_grad():                     # no learning
        model(torch.tensor([encode(ctx)]))    # run once so weights get saved
    w = model.head.last_weights[0, -1]        # last row of the weights table
    print(f"\n   context {ctx!r}: last char {ctx[-1]!r} gives attention to:")   # label
    for pos, (ch, wt) in enumerate(zip(ctx, w)):          # each char with its position number
        bar = "#" * int(wt.item() * 40)       # text bar
        print(f"      pos {pos}  {ch!r:>5}  {wt.item()*100:5.1f}%  {bar}")   # one line


attention_of_last("a b a b")                  # Step 4 gave both 'b's 48.7% — what now?
attention_of_last("the cat")                  # Step 4 gave both 't's 37.2% — what now?


# =====================================================================
# PART C: text written by the model
# =====================================================================
print("\n========== PART C: GENERATED TEXT ==========")       # heading
idx = torch.tensor([encode("T")])             # start with "T"
with torch.no_grad():                         # no learning
    for _ in range(200):                      # 200 chars
        logits, _ = model(idx[:, -block_size:])          # last 8 chars
        probs = F.softmax(logits[0, -1], dim=-1)          # next-char probabilities
        next_id = torch.multinomial(probs, 1)             # dice roll
        idx = torch.cat([idx, next_id.view(1, 1)], dim=1) # append
print(decode(idx[0].tolist()))                # print text


# =====================================================================
# PART D: the problem that is still left
# =====================================================================
print("\n========== PART D: REMAINING PROBLEM ==========")    # heading
print("   Our model has ONE head = ONE question per character.")               # observation 1
print("   But to guess the next char, a char may need SEVERAL things at once:") # observation 2
print("     - 'what was the previous char?'   (spelling)")                     # example 1
print("     - 'where did this word start?'    (word length)")                  # example 2
print("     - 'is this a NAME: line?'         (Shakespeare format)")           # example 3
print("   One Query can only ask one of these. How do we ask many? Think...")  # question for Step 6