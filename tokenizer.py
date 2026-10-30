# ---------- PART A: data padho ----------
# JS: const text = fs.readFileSync("input.txt", "utf-8")
text = open("input.txt", encoding="utf-8").read()

print("========== PART A: DATA ==========")
print("Total characters:", len(text))      # JS: text.length
print("Pehle 200 characters:")
print(text[:200])                           # JS: text.slice(0, 200)


# ---------- PART B: vocabulary banao ----------
# set(text)    -> unique characters       JS: new Set(text)
# sorted(...)  -> alphabetical list       JS: [...new Set(text)].sort()
chars = sorted(set(text))
vocab_size = len(chars)

print("\n========== PART B: VOCABULARY ==========")
print("Vocab size:", vocab_size)
print("Saare characters:", repr("".join(chars)))   # repr() se \n jaise hidden chars dikhte hain


# ---------- PART C: lookup tables ----------
# enumerate(chars) deta hai (index, char) pairs   JS: chars.map((ch, i) => ...)
# Neeche "dict comprehension" hai               JS: Object.fromEntries(chars.map((ch, i) => [ch, i]))
stoi = {ch: i for i, ch in enumerate(chars)}   # string -> integer
itos = {i: ch for i, ch in enumerate(chars)}   # integer -> string

print("\n========== PART C: LOOKUP TABLES ==========")
print("'a' ki ID:", stoi["a"])
print("'A' ki ID:", stoi["A"])
print("' ' (space) ki ID:", stoi[" "])
print("ID 46 kaunsa character:", repr(itos[46]))



# ---------- PART D: encode / decode ----------
# JS: const encode = s => [...s].map(c => stoi[c])
def encode(s):
    return [stoi[c] for c in s]

# JS: const decode = ids => ids.map(i => itos[i]).join("")
def decode(ids):
    return "".join(itos[i] for i in ids)

print("\n========== PART D: ENCODE / DECODE ==========")
sample = "Hello welcome there"
ids = encode(sample)
print("encode('hello there') =", ids)
print("decode(us list ka)    =", repr(decode(ids)))
print("Wapas same mila?      =", decode(ids) == sample)


# ---------- PART E: EXPERIMENT 1 — vocab ke bahar ka text ----------
print("\n========== PART E: HINDI TEXT TRY ==========")
# try/except = JS ka try/catch
try:
    print(encode("Hello नमस्ते"))
except KeyError as e:
    print("ERROR aaya! Yeh character vocab mein nahi hai:", e)

# ---------- PART F: EXPERIMENT 2 — token count compare ----------
print("\n========== PART F: TOKEN COUNT COMPARE ==========")
our_tokens = len(encode(text))
print("Hamara tokenizer (character-level):", our_tokens, "tokens")
try:
    import tiktoken
    gpt_tokens = len(tiktoken.get_encoding("o200k_base").encode(text))
    print("GPT ka tokenizer (BPE)            :", gpt_tokens, "tokens")
    print("Farak                             :", round(our_tokens / gpt_tokens, 1), "guna")
except Exception:
    print("(tiktoken nahi chala. Chalao: pip install tiktoken  — aur internet on rakho, pehli baar file download hoti hai)")

# ---------- PART G: torch tensor (Step 2 ki taiyari) ----------
print("\n========== PART G: TORCH TENSOR ==========")
import torch
data = torch.tensor(encode(text), dtype=torch.long)
print("Shape:", data.shape)
print("Pehle 20 IDs:", data[:20].tolist())
print("Wahi 20 IDs decode karke:", repr(decode(data[:20].tolist())))