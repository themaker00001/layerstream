#!/usr/bin/env bash
# How much does quantization cost in quality? For each variant, score two texts token by token
# (perplexity, and how often the model's top choice matches the bf16 model's), then compare the
# greedy answers to a few prompts with the bf16 model's answers.
#
#   scripts/quant_eval.sh models/Qwen2.5-0.5B-Instruct
#
# Uses ./target/release/layerstream and writes the quantized files to models/ (git-ignored).
set -euo pipefail

MODEL=${1:?usage: quant_eval.sh MODEL_DIR}
BIN=./target/release/layerstream
TMP=$(mktemp -d)
# Frozen copies (the README and model.rs as of commit 3aacc7f), so the numbers cannot drift
# when those files are edited.
TEXTS=(eval/prose.txt eval/code.txt)
MAX_TOKENS=${MAX_TOKENS:-512}

# name:bits:group:extra-flags
VARIANTS=(
  "int8-g64:8:64:"
  "int8-g32:8:32:"
  "int4-g64:4:64:"
  "int4-g32:4:32:"
  "int4-g32-rtn:4:32:--rtn"
  "int4-g32-e8:4:32:--embed-bits 8 --embed-group 64"
  "int4-g16:4:16:"
)

for v in "${VARIANTS[@]}"; do
  IFS=: read -r name bits group extra <<<"$v"
  [ -f "models/$name.lsq" ] || $BIN quantize --model "$MODEL" --out "models/$name.lsq" --bits "$bits" --group "$group" $extra | head -1
done

echo
echo "== perplexity and agreement with bf16 ($MAX_TOKENS tokens of each text) =="
printf '%-14s %-12s %9s %9s %9s\n' variant text ppl top1_acc agree_bf16
for t in "${TEXTS[@]}"; do
  base=$($BIN eval --model "$MODEL" --text "$t" --max-tokens "$MAX_TOKENS" --save-top1 "$TMP/base-$(basename "$t").top1")
  base_text=$(sed -E 's/.*text=([0-9a-f]+).*/\1/' <<<"$base")
  printf '%-14s %-12s %9s %9s %9s\n' bf16 "$(basename "$t")" \
    "$(sed -E 's/.*ppl=([0-9.]+).*/\1/' <<<"$base")" "$(sed -E 's/.*top1_acc=([0-9.]+).*/\1/' <<<"$base")" "1.0000"
  for v in "${VARIANTS[@]}"; do
    IFS=: read -r name _ _ _ <<<"$v"
    out=$($BIN eval --model "$MODEL" --quantized "models/$name.lsq" --text "$t" --max-tokens "$MAX_TOKENS" \
      --agree-with "$TMP/base-$(basename "$t").top1")
    if [ "$(sed -E 's/.*text=([0-9a-f]+).*/\1/' <<<"$out")" != "$base_text" ]; then
      echo "error: $name scored a different text than the baseline (was $t edited while this ran?)" >&2
      exit 1
    fi
    printf '%-14s %-12s %9s %9s %9s\n' "$name" "$(basename "$t")" \
      "$(sed -E 's/.*ppl=([0-9.]+).*/\1/' <<<"$out")" "$(sed -E 's/.*top1_acc=([0-9.]+).*/\1/' <<<"$out")" \
      "$(sed -E 's/.*top1_agree=([0-9.]+).*/\1/' <<<"$out")"
  done
done

echo
echo "== greedy answers vs bf16: tokens identical before the first difference (of 48) =="
PROMPTS=(
  "Explain what a CPU does in two sentences."
  "What is the capital of France, and what is it known for?"
  "Write a haiku about the ocean."
  "Give three tips for learning a new language."
  "Why do we have seasons on Earth?"
  "What is 17 times 23? Show the steps."
)
ids() { "$@" | sed -nE 's/.* ids=([0-9,]*).*/\1/p'; }
same_prefix() { python3 - "$1" "$2" <<'EOF'
import sys
a = [x for x in sys.argv[1].split(",") if x]
b = [x for x in sys.argv[2].split(",") if x]
n = 0
for x, y in zip(a, b):
    if x != y:
        break
    n += 1
print(f"{n}/{max(len(a), 1)}")
EOF
}
BASE=()
for i in "${!PROMPTS[@]}"; do
  BASE[$i]=$(ids $BIN generate --report --model "$MODEL" --mode resident --prompt "${PROMPTS[$i]}" --max-new 48)
done
printf '%-14s' variant; for i in "${!PROMPTS[@]}"; do printf ' %7s' "p$((i + 1))"; done; echo
for v in "${VARIANTS[@]}"; do
  IFS=: read -r name _ _ _ <<<"$v"
  printf '%-14s' "$name"
  for i in "${!PROMPTS[@]}"; do
    q=$(ids $BIN generate --report --model "$MODEL" --quantized "models/$name.lsq" --mode resident --prompt "${PROMPTS[$i]}" --max-new 48)
    printf ' %7s' "$(same_prefix "${BASE[$i]}" "$q")"
  done
  echo
done
rm -rf "$TMP"
