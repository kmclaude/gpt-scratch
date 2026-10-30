# gpt-scratch

A tiny GPT built from scratch in Python + PyTorch, **one idea at a time**.

Each step is a single, self-contained script that adds exactly one new piece of the transformer, trains it on Shakespeare, and prints what changed. Every line of code has a comment, so you can follow along even if you're new to Python or PyTorch.

By Step 7 the model has the same architecture as GPT-2, just with much smaller numbers. Steps 8–10 then use it to explore context windows, sampling, and fine-tuning.

## The journey

| Step | File | What it adds | Val loss* |
|---|---|---|---|
| 1 | `tokenizer.py` | Character-level tokenizer: text → IDs → text | – |
| 2 | `step2_bigram.py` | Bigram model + the first training loop. Guesses the next char from the current char only | 2.46 |
| 3 | `step3_average.py` | Average of all previous chars. Shows why "equal attention" fails | 2.86 |
| 4 | `step4_attention.py` | **Self-attention** (Query, Key, Value). Each char decides what to look at | 2.46 |
| 5 | `step5_position.py` | **Position embeddings**. The model now knows WHERE each char is, not only WHAT it is | 2.41 |
| 6 | `step6_multihead.py` | **Multi-head attention + feed-forward**. Several heads ask different "questions" in parallel | 2.21 |
| 7 | `step7_gpt.py` | **Stacked blocks + residual connections + LayerNorm** = a real (tiny) GPT | 2.07 |
| 8 | `step8_gpt.py` | **Context window**: same model with 8 vs 32 chars of memory. Saves both models | 1.94 |
| 9 | `step9_gpt.py` | **Sampling**: temperature, top-k, top-p, and why greedy decoding loops | – |
| 10 | `step10_sft.py` | **Instruction fine-tuning (SFT)**: base model → Q/A "assistant". Memorisation vs generalisation, hallucination, catastrophic forgetting | – |

\* Validation loss on Tiny Shakespeare (lower is better). Exact numbers vary slightly between runs and machines.

## Things you'll see with your own eyes

- **Attention heads specialise on their own.** In Step 6, one head learns "what char came just before me?", another learns "where did this word start?". Nobody codes these rules.
- **Depth without residuals hurts.** In Step 7, 4 stacked blocks *without* residual + LayerNorm do worse (2.37) than a single block (2.21). With them: 2.07.
- **A bigger window ≠ a model that uses it.** In Step 8, loss drops fast for the first ~8 chars of context, then flattens.
- **Greedy decoding is the worst decoding.** In Step 9, temperature 0 gets stuck repeating `the some the some ...`.
- **Fine-tuning teaches format first, and data variety decides whether it learns a rule or just memorises.** In Step 10, 45 examples give 0/13 on unseen words, while 600 varied examples give 7/13.
- **Hallucination in miniature.** Asked a fact it never saw, the fine-tuned model never says "I don't know". It answers confidently and wrong.

## Setup

Python 3.10+ recommended.

```bash
git clone git@github.com:<your-username>/gpt-scratch.git
cd gpt-scratch
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Run the steps in order. Each one prints its results in labelled parts (PART A, PART B, ...).

```bash
python3 tokenizer.py
python3 step2_bigram.py
# ...
python3 step8_gpt.py     # trains and saves tiny_gpt_ctx8.pt + tiny_gpt_ctx32.pt
python3 step9_gpt.py     # needs tiny_gpt_ctx32.pt from Step 8
python3 step10_sft.py    # needs tiny_gpt_ctx32.pt from Step 8
```

Everything runs on a laptop CPU. Most steps take 1–6 minutes. Trained `.pt` files are not committed; Step 8 recreates them.

## The model at the end vs GPT-2 small

|  | This repo | GPT-2 small |
|---|---|---|
| Tokens | characters (65) | BPE (~50k) |
| Context window | 8–32 | 1024 |
| Embedding size | 32 | 768 |
| Heads per block | 4 | 12 |
| Blocks | 4 | 12 |
| Parameters | ~55 thousand | ~124 million |

Same recipe. Only the numbers are bigger.

## Credits

- Dataset: Tiny Shakespeare, from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).
- The step-by-step approach is inspired by Karpathy's "Let's build GPT" lecture and [nanoGPT](https://github.com/karpathy/nanoGPT).
