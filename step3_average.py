# =====================================================================
# STEP 3: PICHHLE SAARE CHARACTERS KA "AVERAGE" — attention ki pehli seedhi
# Chalao:  python step3_average.py      (~1 minute)
#
# Step 2 ki problem: model sirf AAKHRI character dekhta tha.
# Aaj ka idea: pichhle SAARE characters ki info ka AVERAGE lo, aur usse guess karo.
# =====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
torch.set_printoptions(precision=2, sci_mode=False)


# =====================================================================
# PART A: average kya hai — 4 characters, haath se
# =====================================================================
print("========== PART A: RUNNING AVERAGE (loop se) ==========")
# Maan lo 4 characters hain, aur har character ek vector hai (2 numbers).
# (Step 2 mein har character ek 65-number ki row tha; yahan samajhne ke liye sirf 2.)
x = torch.tensor([[1.0, 0.0],    # char 1
                  [0.0, 1.0],    # char 2
                  [1.0, 1.0],    # char 3
                  [3.0, 0.0]])   # char 4

# Har position pe: "main aur mujhse PEHLE wale sab" ka average.
# JS:  for (let t = 0; t < 4; t++) out[t] = avg(x.slice(0, t + 1))
out = torch.zeros_like(x)
for t in range(4):
    out[t] = x[: t + 1].mean(dim=0)       # x[0..t] ka average
    print(f"position {t+1}: chars 1..{t+1} ka average = {[round(v, 2) for v in out[t].tolist()]}")
print("\nDhyaan do: position 2 ko char 3 ya 4 ki info NAHI mili. Future dekhna mana hai.")


# =====================================================================
# PART B: wahi average, ek "weights table" se  (yahi attention ka dhaancha hai)
# =====================================================================
print("\n========== PART B: WAHI KAAM, WEIGHTS TABLE SE ==========")
# Har row batati hai: "is position ne kis character ko kitna weight diya"
T = 4
scores = torch.zeros(T, T)                               # abhi sabka score 0 (koi pasand nahi)
mask = torch.tril(torch.ones(T, T))                      # lower triangle = 1, upar = 0
scores = scores.masked_fill(mask == 0, float("-inf"))    # future = -infinity
print("Scores (future = -inf):")
print(scores)

weights = F.softmax(scores, dim=-1)                       # haan, wahi softmax!
print("\nSoftmax ke baad WEIGHTS (har row ka total = 1):")
print(weights)

out2 = weights @ x                                        # @ = matrix multiply = "weighted sum"
print("\nweights @ x  (har row = us position ka weighted average):")
print(out2)
print("\nPART A wale answer se same?", torch.allclose(out, out2))


# =====================================================================
# PART C: asli model — embeddings + average + prediction
# =====================================================================
print("\n========== PART C: AVERAGE MODEL TRAIN KARO ==========")
text = open("input.txt", encoding="utf-8").read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

block_size = 8
batch_size = 32
n_embd = 32       # har character ka vector ab 32 numbers ka (meaning wala embedding, topic 1.2)

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x, y


class AverageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)   # char ID -> 32 numbers ka vector
        self.lm_head = nn.Linear(n_embd, vocab_size)      # 32 numbers -> 65 scores
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)                              # (B, T, 32)

        # PART B wala jugaad: barabar weights se pichhle sab ka average
        scores = torch.zeros(T, T)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)                # (T, T)
        x = weights @ x                                    # (B, T, 32) — har position = average

        logits = self.lm_head(x)                           # (B, T, 65)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss


model = AverageModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
for step in range(5001):
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 1000 == 0:
        with torch.no_grad():
            vl = sum(model(*get_batch("val"))[1].item() for _ in range(20)) / 20
        print(f"step {step:5d} | train loss {loss.item():.3f} | val loss {vl:.3f}")

print("\nYaad karo: Step 2 (bigram) ka val loss tha ~2.46")


# =====================================================================
# PART D: model ka likha text
# =====================================================================
print("\n========== PART D: GENERATED TEXT ==========")
idx = torch.tensor([encode("T")])
with torch.no_grad():
    for _ in range(200):
        logits, _ = model(idx[:, -block_size:])
        probs = F.softmax(logits[0, -1], dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1).view(1, 1)], dim=1)
print(decode(idx[0].tolist()))


# =====================================================================
# PART E: DO NAYI PROBLEMS
# =====================================================================
print("\n========== PART E: PROBLEMS ==========")
def show(ctx):
    with torch.no_grad():
        logits, _ = model(torch.tensor([encode(ctx)]))
        probs = F.softmax(logits[0, -1], dim=-1)
    p, i = torch.topk(probs, 3)
    s = "   ".join(f"{itos[j.item()]!r}:{q.item()*100:4.1f}%" for q, j in zip(p, i))
    print(f"   context {ctx!r:>8}  ->  {s}")

print("Problem 1: ORDER kho gaya (average mein kram ka pata nahi chalta)")
show("th")
show("ht")
print("\nProblem 2: AAKHRI character dab gaya (sabko barabar weight)")
show("q")
show("aaaaaaaq")