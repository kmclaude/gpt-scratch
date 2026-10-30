# =====================================================================
# STEP 10: MINI SFT (instruction fine-tuning)  —  base model -> "assistant"
# Run:  python3 step10_sft.py      (takes ~2 to 4 minutes — it fine-tunes twice)
#
# Our Step 8 model is a BASE model: it only continues Shakespeare.
# Give it "Q: ..." and it just writes more play.
# Today: train the SAME saved model a little more, on question -> answer examples.
# Same weights, same code, same loss. Only the DATA changes.
# Needs: tiny_gpt_ctx32.pt  (saved by Step 8) and input.txt
# =====================================================================

import math                                   # for math.sqrt
import random                                 # to shuffle examples
import re                                     # to pull words out of the Shakespeare text
import torch                                  # main library
import torch.nn as nn                         # building blocks
import torch.nn.functional as F               # softmax, cross_entropy


# ---------------------------------------------------------------------
# Load the saved base model from Step 8
# ---------------------------------------------------------------------
ckpt = torch.load("tiny_gpt_ctx32.pt")        # read the file Step 8 saved
chars = ckpt["chars"]                         # vocabulary saved with the model
block_size = ckpt["block_size"]               # 32
vocab_size = len(chars)                       # 65
stoi = {ch: i for i, ch in enumerate(chars)}  # char -> ID
itos = {i: ch for i, ch in enumerate(chars)}  # ID -> char
encode = lambda s: [stoi[c] for c in s]       # text -> IDs
decode = lambda ids: "".join(itos[i] for i in ids)      # IDs -> text

n_embd, n_head, n_layer = 32, 4, 4            # must match Step 8 exactly
head_size = n_embd // n_head                  # 8


# ---------------------------------------------------------------------
# Same model classes as Step 8/9 (copied, so this file runs on its own)
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

    def forward(self, idx):                   # idx: (B, T)
        T = idx.shape[1]                      # sequence length
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T))   # WHAT + WHERE
        return self.lm_head(self.ln_f(self.blocks(x)))          # -> (B, T, 65) scores


model = GPT(block_size)                       # empty model with the right shape
model.load_state_dict(ckpt["state"])          # pour in the Shakespeare-trained numbers
print("Loaded base model: tiny_gpt_ctx32.pt")  # confirm


# ---------------------------------------------------------------------
# Helpers: answer a question, and write free text
# ---------------------------------------------------------------------
def ask(question, max_chars=20):              # ask the model one question, greedy (temperature 0)
    model.eval()                              # exam mode
    idx = torch.tensor([encode(f"Q: {question}\nA:")])   # the exact format we will train on
    with torch.no_grad():                     # no learning
        for _ in range(max_chars):            # one char at a time
            next_id = model(idx[:, -block_size:])[0, -1].argmax()   # pick the #1 char (greedy)
            idx = torch.cat([idx, next_id.view(1, 1)], dim=1)       # append it
            if itos[next_id.item()] == "\n":  # stop at end of line (= end of answer)
                break                         # done
    full = decode(idx[0].tolist())            # IDs -> text
    return full.split("\nA:", 1)[1].strip()   # keep only what came after "A:"


def write(start, n_chars=120):                # free writing with normal sampling (temperature 1)
    model.eval()                              # exam mode
    torch.manual_seed(42)                     # same dice every time
    idx = torch.tensor([encode(start)])       # starting text
    with torch.no_grad():                     # no learning
        for _ in range(n_chars):              # one char at a time
            probs = F.softmax(model(idx[:, -block_size:])[0, -1], dim=-1)   # next-char probabilities
            idx = torch.cat([idx, torch.multinomial(probs, 1).view(1, 1)], dim=1)   # roll dice, append
    return decode(idx[0].tolist())            # IDs -> text


def load_base():                              # fresh copy of the Shakespeare base model
    model.load_state_dict(ckpt["state"])      # overwrite any fine-tuning with the original numbers


def finetune(examples, steps=1500):           # the SAME training loop as pretraining, on new data
    torch.manual_seed(1337)                   # repeatable
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)   # same optimizer as before
    for step in range(steps + 1):             # much shorter than pretraining (5000 steps)
        random.shuffle(examples)              # new order each step
        stream = torch.tensor(encode("".join(examples)))   # all examples glued into one long text
        ix = torch.randint(len(stream) - block_size, (32,))   # 32 random windows
        xb = torch.stack([stream[i : i + block_size] for i in ix])           # inputs
        yb = torch.stack([stream[i + 1 : i + block_size + 1] for i in ix])   # answers (shifted by 1)
        model.train()                         # learning mode
        logits = model(xb)                    # forward
        loss = F.cross_entropy(logits.view(-1, vocab_size), yb.view(-1))     # same loss as pretraining
        opt.zero_grad()                       # clear gradients
        loss.backward()                       # backprop
        opt.step()                            # nudge the weights
        if step % 500 == 0:                   # every 500 steps
            print(f"   step {step:5d} | loss {loss.item():.3f}")   # progress


def score(title, pairs):                      # helper: ask each question, count correct
    print(f"\n-- {title} --")                 # sub-heading
    right = 0                                 # counter
    for q, want in pairs:                     # each question
        got = ask(q)                          # model's answer
        ok = got == want                      # exact match?
        right += ok                           # count it
        print(f"   {'OK ' if ok else 'XX '} Q: {q:<20} A: {got!r:<10} (correct: {want!r})")   # row
    print(f"   -> {right}/{len(pairs)} correct")   # total


# =====================================================================
# The SFT data: two kinds of tasks
#   1. SKILL: "shout <word>"     -> the word in CAPITALS   (a rule: a->A, b->B ...)
#   2. FACT:  "opposite of <w>"  -> the opposite word      (pure memory, no rule)
# Some examples are kept HIDDEN from training, to test on later.
# =====================================================================
shout_test = ["word", "peace", "face", "horse", "water", "head", "time",
              "friend", "lord", "earth", "shame", "fire", "storm"]   # 13 HIDDEN words, never trained on
small_words = ("love king night death heart blood crown queen sword grace honour "
               "father mother brother sister enemy house heaven world war hand eye "
               "tongue day life soul name light dark field gold power duty fear hope "
               "joy grief truth faith land sea ship moon star rose").split()   # 45 training words
opposites = [("hot", "cold"), ("good", "bad"), ("rich", "poor"), ("love", "hate"),
             ("light", "dark"), ("king", "queen"), ("war", "peace"), ("first", "last"),
             ("high", "low"), ("fast", "slow"), ("true", "false"), ("friend", "foe"),
             ("strong", "weak"), ("big", "small"), ("wise", "foolish"), ("joy", "grief"),
             ("heaven", "hell"), ("early", "late"), ("full", "empty"), ("day", "night")]   # 20 training pairs
opp_test = [("old", "young"), ("open", "shut"), ("sweet", "sour"),
            ("up", "down"), ("life", "death")]   # 5 HIDDEN pairs, never trained on

shout = lambda w: f"Q: shout {w}\nA: {w.upper()}\n"          # one skill example as text
opp = lambda a, b: f"Q: opposite of {a}\nA: {b}\n"           # one fact example as text
small_data = [shout(w) for w in small_words] + [opp(a, b) for a, b in opposites]   # 65 examples

text = open("input.txt", encoding="utf-8").read().lower()   # Shakespeare, lowercased
all_words = sorted(set(re.findall(r"\b[a-z]{3,7}\b", text)))   # every 3-7 letter word in it
random.seed(0)                                # same pick every run
big_words = random.sample([w for w in all_words if w not in shout_test], 600)   # 600 words, hidden ones excluded
big_data = [shout(w) for w in big_words] + [opp(a, b) for a, b in opposites]    # 620 examples
print(f"Example SFT lines: {small_data[0]!r}  {small_data[-1]!r}")   # show the format


# =====================================================================
# PART A: BEFORE fine-tuning (the base model)
# =====================================================================
print("\n========== PART A: BEFORE SFT (base model) ==========")   # heading
for q in ["shout love", "opposite of hot"]:   # two questions it will train on
    print(f"   Q: {q:<20} A: {ask(q)!r}") # print its answer
print("   (A base model doesn't know what a 'question' is. It just continues text.)")   # note


# =====================================================================
# PART B: SFT with a SMALL dataset (65 examples)
# =====================================================================
print("\n========== PART B: SFT, SMALL DATA (45 shout + 20 opposite) ==========")   # heading
finetune(small_data)                          # train
score("B1. SKILL, words it SAW", [(f"shout {w}", w.upper()) for w in small_words[:4]])   # seen skill
score("B2. SKILL, words it NEVER saw", [(f"shout {w}", w.upper()) for w in shout_test])   # unseen skill
score("B3. FACT, pairs it SAW", [(f"opposite of {a}", b) for a, b in opposites[:4]])      # seen facts
score("B4. FACT, pairs it NEVER saw", [(f"opposite of {a}", b) for a, b in opp_test])    # unseen facts


# =====================================================================
# PART C: SFT again from the base, with MORE VARIED data (620 examples)
# =====================================================================
print("\n========== PART C: SFT, BIG VARIED DATA (600 shout + same 20 opposite) ==========")   # heading
load_base()                                   # start again from the Shakespeare model
finetune(big_data)                            # train on the bigger set
score("C1. SKILL, words it NEVER saw", [(f"shout {w}", w.upper()) for w in shout_test])   # unseen skill
score("C2. FACT, pairs it NEVER saw", [(f"opposite of {a}", b) for a, b in opp_test])    # unseen facts


# =====================================================================
# PART D: what happened to Shakespeare?
# =====================================================================
print("\n========== PART D: ASK FOR SHAKESPEARE AGAIN ==========")   # heading
print(write("ROMEO:\n"))                      # same prompt as Step 9