#!/usr/bin/env python3
"""Does speculative decoding help a streamed model? Every configuration runs in a fresh
process, one at a time, and every speculative run is checked to have produced exactly the
tokens that plain greedy decoding of the big model produces.

    scripts/spec_bench.py models/Qwen2.5-0.5B-Instruct

Needs the quantized files made by scripts/quant_eval.sh in models/ (the guessing models and
the int8 big model). Standard library only.
"""
import subprocess
import sys

BIN = "./target/release/layerstream"
MODEL = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: spec_bench.py MODEL_DIR")
PROMPTS = [
    "Explain what a CPU does in two sentences.",
    "What is the capital of France, and what is it known for?",
    "Give three tips for learning a new language.",
    "Why do we have seasons on Earth?",
]
DRAFT_INT8 = "models/int8-g64.lsq"
DRAFT_INT4 = "models/int4-g16.lsq"

# (label, extra args for the big model)
TARGETS = {"bf16": [], "int8": ["--quantized", "models/int8-g64.lsq"]}
# (label, throttle MB/s or None, tokens to generate, number of prompts)
REGIMES = {"file cached": (None, 48, 4), "capped 2000 MB/s": (2000, 48, 4), "capped 500 MB/s": (500, 24, 3)}

# which (target, regime, draft, k) to run; None draft = plain decoding
PLAN = []
for regime in REGIMES:
    PLAN.append(("bf16", regime, None, 0))
    for k in (2, 4, 8):
        PLAN.append(("bf16", regime, DRAFT_INT8, k))
    PLAN.append(("bf16", regime, DRAFT_INT4, 4))
PLAN.append(("int8", "capped 500 MB/s", None, 0))
PLAN.append(("int8", "capped 500 MB/s", DRAFT_INT4, 4))
PLAN.append(("int8", "file cached", None, 0))
PLAN.append(("int8", "file cached", DRAFT_INT4, 4))


def fields(line):
    return dict(kv.split("=", 1) for kv in line.split() if "=" in kv)


def run(args):
    out = subprocess.run(args, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if line.startswith(("RESULT ", "SPEC_RESULT ")):
            return fields(line)
    sys.exit(f"run failed: {' '.join(args)}\n{out.stderr[-600:]}")


def cache_line(path):
    out = subprocess.run([BIN, "cache", path], capture_output=True, text=True).stdout.strip()
    return out.replace("before: ", "")


files = [MODEL + "/model.safetensors", DRAFT_INT8, DRAFT_INT4, TARGETS["int8"][1]]
print("warming the OS cache (reading each file once) ...", flush=True)
for f in files:
    subprocess.run([BIN, "cache", "--read", f], capture_output=True)
for f in dict.fromkeys(files):
    print("  cache at start:", cache_line(f))

plain_ids = {}  # (target, regime, prompt index) -> tokens
rows = []
baseline_rate = {}
for target, regime, draft, k in PLAN:
    mbps, max_new, n_prompts = REGIMES[regime]
    extra = TARGETS[target] + (["--throttle-mbps", str(mbps)] if mbps else [])
    tot_tokens = tot_secs = passes = drafted = accepted = 0
    peak = 0
    identical = True
    for i, prompt in enumerate(PROMPTS[:n_prompts]):
        if draft is None:
            r = run([BIN, "generate", "--report", "--model", MODEL, "--mode", "stream", "--max-new", str(max_new), "--prompt", prompt] + extra)
            ids = [int(x) for x in r["ids"].split(",")]
            plain_ids[(target, regime, i)] = ids
            tot_tokens += int(r["decode_forwards"])
        else:
            r = run([BIN, "speculate", "--report", "--model", MODEL, "--draft", draft, "--k", str(k), "--max-new", str(max_new), "--prompt", prompt] + extra)
            ids = [int(x) for x in r["ids"].split(",")]
            identical &= ids == plain_ids[(target, regime, i)]
            tot_tokens += int(r["new_tokens"]) - 1
            passes += int(r["passes"])
            drafted += int(r["drafted"])
            accepted += int(r["accepted"])
        tot_secs += float(r["decode_s"])
        peak = max(peak, int(r["peak_rss"]))
    rate = tot_tokens / tot_secs
    key = (target, regime)
    if draft is None:
        baseline_rate[key] = rate
    rows.append(
        (target, regime, "plain" if draft is None else ("int8 copy" if draft == DRAFT_INT8 else "int4 copy"), k, rate, rate / baseline_rate[key],
         (tot_tokens / passes) if passes else 1.0, (100.0 * accepted / drafted) if drafted else 0.0, peak / 2**20, identical)
    )
    print(f"  done: {target} / {regime} / {rows[-1][2]} k={k}: {rate:.2f} tokens/s", flush=True)

print()
print(f"{'big model':<9} {'storage':<17} {'draft':<10} {'k':>2} {'tokens/s':>9} {'speedup':>8} {'tok/pass':>9} {'guesses ok':>11} {'peak MB':>8}  same words as plain")
print("-" * 112)
last = None
for target, regime, name, k, rate, speedup, tpp, acc, peak, same in rows:
    if last and last != (target, regime):
        print()
    last = (target, regime)
    print(f"{target:<9} {regime:<17} {name:<10} {k if k else '-':>2} {rate:>9.2f} {speedup:>7.2f}x {tpp:>9.2f} {(f'{acc:.0f}%' if name != 'plain' else '-'):>11} {peak:>8.0f}  {'yes' if same else 'NO'}")
print()
print("tokens/s counts tokens after the first. tok/pass: tokens produced per pass over the big model's weights.")
print("guesses ok: share of the draft's guessed tokens the big model agreed with. 'same words as plain' compares every run to")
print("plain greedy decoding of the same big model with the same prompt.")
print()
for f in dict.fromkeys(files):
    print("  cache at end:  ", cache_line(f))
