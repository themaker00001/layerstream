#!/usr/bin/env python3
"""
Produce ground truth from the official Hugging Face implementation, for the Rust tests.

    python3 scripts/reference.py models/Qwen2.5-0.5B-Instruct tests/golden.json

Needs `torch` and `transformers` (not needed to build or run the Rust program). The model
is run in float32 from the bf16 weights, so the weights are bit-for-bit what the Rust
engine reads; only the order of floating-point additions differs.

Writes:
  tokenizer: a list of {text, ids} for varied strings
  prompts:   for several prompts, the token ids, a summary of the last-position logits
             (top 10 plus a fixed set of probe indices), and a greedy continuation
"""

import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir, out_path = sys.argv[1], sys.argv[2]

tok = AutoTokenizer.from_pretrained(model_dir)

TEXTS = [
    "",
    "Hello, world!",
    "I'm sure it's 2024 and they've said we'll go, I'd like that.",
    "The quick brown fox jumps over the lazy dog.",
    "  leading and   multiple    spaces  ",
    "tabs\tand\nnewlines\n\nand a blank line\n",
    "x \n y",
    "trailing spaces   ",
    "!!word !word ...dots?! (parens) [brackets] {braces}",
    "numbers 3.14159 and 1,000,000 and 007",
    "def f(x):\n    return x * 2  # double\n",
    "Café déjà vu naïve résumé",
    "日本語のテキストです。",
    "Привет, мир!",
    "emoji 😀 and 🚀 together",
    "<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n",
    "<|endoftext|>",
    "URL https://example.com/path?q=1&r=2#frag and email a.b@c.org",
    "CamelCaseWord snake_case_word kebab-case-word",
    "ALL CAPS and MiXeD cAsE",
    "She said, \"Don't!\" 'quoted' it’s",
    "a" * 50,
    " non-breaking space",
    "line one\r\nline two\r\n",
]

golden = {"tokenizer": [{"text": t, "ids": tok.encode(t, add_special_tokens=False)} for t in TEXTS]}

try:
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
model.eval()

PROBES = [0, 1, 2, 100, 1000, 5000, 50000, 100000, 151642, 151935]


def chat(text):
    return tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )


PROMPTS = [
    ("chat", chat("What is the capital of France?")),
    ("chat", chat("Write one sentence about why the sky is blue.")),
    ("raw", "The quick brown fox"),
]

NEW_TOKENS = 24
results = []
for kind, text in PROMPTS:
    ids = tok.encode(text, add_special_tokens=False)
    x = torch.tensor([ids])
    with torch.no_grad():
        logits = model(x).logits[0, -1]
    top = torch.topk(logits, 10)
    with torch.no_grad():
        gen = model.generate(
            x,
            max_new_tokens=NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,
            eos_token_id=[151645, 151643],
            pad_token_id=151643,
        )[0][len(ids):].tolist()
    results.append(
        {
            "kind": kind,
            "text": text,
            "ids": ids,
            "top10": [{"id": int(i), "logit": float(v)} for v, i in zip(top.values, top.indices)],
            "probes": [{"id": p, "logit": float(logits[p])} for p in PROBES],
            "greedy_ids": gen,
            "greedy_text": tok.decode(gen, skip_special_tokens=True),
        }
    )
    print(f"[{kind}] {len(ids)} prompt tokens -> {tok.decode(gen, skip_special_tokens=True)!r}")

golden["prompts"] = results
with open(out_path, "w") as f:
    json.dump(golden, f, ensure_ascii=False, indent=1)
print(f"wrote {out_path}: {len(golden['tokenizer'])} tokenizer cases, {len(results)} prompts")
