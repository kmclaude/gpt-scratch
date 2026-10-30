# =====================================================================
# STEP 2: SABSE SIMPLE "MODEL" (BIGRAM) + PEHLA TRAINING LOOP
# Chalao:  python step2_bigram.py      (~30 second se 1 minute)
#
# Bigram ka idea: "sirf PICHHLA EK character dekh ke agla guess karo."
#   'q' ke baad?  -> 'u'
#   'h' ke baad?  -> shayad 'e'
# Model asal mein ek 65 x 65 ki TABLE hai:
#   row = abhi wala character, column = agla character, value = kitna likely
# =====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)   # random numbers har baar same aayein (taaki tumhara output mere jaisa ho)

# ---------- Step 1 wala tokenizer (yahan dobara, taaki file akeli chal sake) ----------
text = open("input.txt", encoding="utf-8").read()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]                 # JS: s => [...s].map(c => stoi[c])
decode = lambda ids: "".join(itos[i] for i in ids)
data = torch.tensor(encode(text), dtype=torch.long)


# ---------- PART A: train / validation split ----------
print("========== PART A: TRAIN / VAL SPLIT ==========")
n = int(0.9 * len(data))
train_data = data[:n]     # pehla 90%: isse model seekhega
val_data = data[n:]       # aakhri 10%: model ne kabhi nahi dekha, sirf "exam" ke liye
print("Train tokens:", len(train_data))
print("Val tokens  :", len(val_data))


# ---------- PART B: batch kaise banta hai ----------
print("\n========== PART B: EK BATCH ==========")
block_size = 8    # ek tukde mein kitne characters (context length)
batch_size = 4    # ek saath kitne tukde

def get_batch(split):
    d = train_data if split == "train" else val_data
    # random starting points   JS: Array.from({length: batch_size}, () => randInt(...))
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])          # input
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])  # answer = ek position aage
    return x, y

xb, yb = get_batch("train")
print("x shape:", tuple(xb.shape), " (batch_size, block_size)")
print("Pehla tukda x:", repr(decode(xb[0].tolist())))
print("Pehla tukda y:", repr(decode(yb[0].tolist())), " <- x ka hi text, ek aage khiska hua")
print("\nIs EK tukde ke andar", block_size, "training examples chhupe hain:")
for t in range(block_size):
    context = decode(xb[0, : t + 1].tolist())
    target = decode([yb[0, t].item()])
    print(f"   {context!r:>12}  ke baad aana chahiye  ->  {target!r}")


# ---------- PART C: bigram model ----------
print("\n========== PART C: MODEL (TRAINING SE PEHLE) ==========")

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 65 x 65 ki table, random numbers se shuru.
        # nn.Embedding = "ID do, us row ke numbers wapas lo"  (array lookup: table[id])
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.table(idx)   # (B, T, 65): har position ke liye 65 "scores" (logits)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            # cross_entropy: "sahi answer ko model ne kitni kam probability di" = penalty
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]              # sirf aakhri character ki row chahiye
            probs = F.softmax(logits, dim=-1)       # scores -> probabilities (total 100%)
            next_id = torch.multinomial(probs, 1)   # probabilities ke hisaab se ek "sample"
            idx = torch.cat([idx, next_id], dim=1)  # jodo aur repeat
        return idx

model = BigramModel(vocab_size)
print("Parameters (table ke numbers):", sum(p.numel() for p in model.parameters()), "= 65 x 65")

_, loss = model(xb, yb)
import math
print(f"Loss abhi: {loss.item():.3f}")
print(f"Random guess ka loss: ln(65) = {math.log(vocab_size):.3f}   <- abhi wala isse thoda upar hoga (random table 'confident galat' hai)")

start = torch.zeros((1, 1), dtype=torch.long)   # ID 0 = '\n' se shuru
print("\nTraining se PEHLE model ka likha text:")
print(decode(model.generate(start, 150)[0].tolist()))


# ---------- PART D: TRAINING LOOP ----------
print("\n========== PART D: TRAINING ==========")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
batch_size = 32

for step in range(5001):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)        # 1. forward: guess karo, galti naapo
    optimizer.zero_grad()               # 2. purane gradients saaf
    loss.backward()                     # 3. backprop: har number ko kitna blame hai
    optimizer.step()                    # 4. table ke numbers thoda sudharo

    if step % 500 == 0:
        with torch.no_grad():
            _, val_loss = model(*get_batch("val"))
        print(f"step {step:5d} | train loss {loss.item():.3f} | val loss {val_loss.item():.3f}")


# ---------- PART E: training ke BAAD ----------
print("\n========== PART E: TRAINING KE BAAD MODEL KA TEXT ==========")
print(decode(model.generate(start, 300)[0].tolist()))


# ---------- PART F: model ne kya seekha? table ke andar jhaanko ----------
print("\n========== PART F: TABLE KE ANDAR ==========")
def top_next(ch, k=5):
    with torch.no_grad():
        probs = F.softmax(model.table.weight[stoi[ch]], dim=-1)
    top_p, top_i = torch.topk(probs, k)
    shown = "   ".join(f"{itos[i.item()]!r}:{p.item()*100:4.1f}%" for p, i in zip(top_p, top_i))
    print(f"   {ch!r} ke baad  ->  {shown}")

for ch in ["q", "h", "t", " ", "."]:
    top_next(ch)


# ---------- PART G: is model ki SABSE BADI KAMI ----------
print("\n========== PART G: PROBLEM ==========")
for ctx in ["th", "sh", "Wh", "and th"]:
    with torch.no_grad():
        logits, _ = model(torch.tensor([encode(ctx)]))
        probs = F.softmax(logits[0, -1], dim=-1)
    p, i = torch.topk(probs, 3)
    shown = "   ".join(f"{itos[j.item()]!r}:{q.item()*100:4.1f}%" for q, j in zip(p, i))
    print(f"   context {ctx!r:>9}  ->  {shown}")
print("Charon ka prediction BILKUL SAME hai. Kyun? Socho...")