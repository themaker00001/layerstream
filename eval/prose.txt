# layerstream

Run a language model that is bigger than your memory budget by keeping only a couple of
layers in memory at a time, and measure what that costs.

A small Rust program with no dependencies. The tokenizer, the weights reader, the transformer
and the (optional) encryption are written from scratch and checked against the reference
implementations. It runs **Qwen2.5-0.5B-Instruct**, a real pretrained chat model.

```text
$ layerstream generate --model models/Qwen2.5-0.5B-Instruct --prompt "Explain what a CPU does in two sentences."
A CPU (Central Processing Unit) is the brain of a computer, responsible for executing instructions and
performing calculations. It processes data, manages memory, and controls the flow of operations within
the computer system.

[layerstream] stream, 2 slots (plain file) | 10 threads
[layerstream] peak memory 76 MB (weights held: 57 MB; tokenizer alone: 13 MB)
```

## The result in one table

Apple M4, 10 cores, 24 GB RAM, macOS 26, release build. Same prompt, greedy decoding, 32
tokens, every configuration in a fresh process. **Every configuration produced exactly the same
tokens.** Two complete runs agreed within about 3%.

| Configuration | Peak memory | Tokens/s | Read per token |
|---|---:|---:|---:|
| Everything in memory | 962 MB | 33 | n/a |
| Stream, 1 buffer, no read-ahead | **47 MB** | 10.1 | 942 MB |
| Stream, 2 buffers (read-ahead) | 76 MB | 13.6 | 942 MB |
| Stream, storage capped at 2,000 MB/s | 75 MB | 2.0 | 942 MB |
| Stream, storage capped at 500 MB/s | 76 MB | 0.5 | 942 MB |
| Encrypted, everything in memory | 961 MB | 34 | n/a |
| Encrypted, stream, 2 buffers | 77 MB | 2.9 | 944 MB |
| Encrypted, stream, capped at 500 MB/s | 75 MB | 0.4 | 950 MB |

What it says:

- **Memory:** 962 MB down to 76 MB (12.7x less), or 47 MB with a single buffer (20x less). Peak
  memory is set by the size of one layer, not by the size of the model.
- **Speed:** streaming costs speed because every token has to read the whole model from
  storage: 942 MB per token. With the file in the operating system's cache that is 13.6 tokens/s
  instead of 33. On slower storage the speed is capped by it: tokens/s can never exceed
  `storage MB/s / 942`, and the capped runs sit within about 5% of that ceiling.
- **Encryption at rest:** if you load everything once, decrypting costs a one-time 0.3 to 0.5 s
  at startup. If you stream, every token decrypts about 1 GB, and speed falls to 2.9 tokens/s.

### How to read these numbers honestly

- **The model is small and its answers can be wrong.** Qwen2.5-0.5B is a good test model, not a
  reliable assistant. For the benchmark prompt ("why is the sky blue") it says the sky is "a
  reflection of the sunlight", which is wrong (the reason is Rayleigh scattering). What is
  checked here is that the engine produces exactly the same words as the reference
  implementation, not that the words are true.
- **The model fits in this Mac's memory.** A 0.5B model is not "large", and this machine has
  24 GB. The "small device" is emulated: memory is the process's measured peak resident size,
  and slow storage is emulated with `--throttle-mbps`, which really limits read speed. The
  throttle was checked to deliver its cap (a 500 MB/s cap sustains 481 MB/s over a whole run).
- **The uncapped streaming speed is a RAM-speed number.** The operating system keeps the weights
  file cached, so the "13.6 tokens/s" row reads at memory speed. I tried to measure true cold
  SSD reads and could not: macOS's no-cache flag did not keep the file out of the cache in my
  tests (a freshly copied file read at 6.4 GB/s on the first run and 9.8 GB/s on the second), and
  I cannot evict the cache without administrator rights. I removed that feature rather than keep
  a benchmark row that did not measure what it claimed. For context, one raw read of a freshly
  written file measured 3.25 GB/s on this SSD, which would cap streaming at about 3.4 tokens/s.
- **Only the memory and storage limits are emulated.** A small device also has a slower CPU, and
  that is not.
- **100 MB/s (an SD card) is arithmetic, not a measurement.** 942 MB per token at 100 MB/s is
  about 9 seconds per token.
- **Tested on one model.** Peak memory depends on the layer size, so the arithmetic carries to
  larger models of the same architecture, but I have not run one. Models with a separate output
  matrix (untied embeddings) are not supported yet.
- **Peak memory** is the whole process (it includes a 14 MB tokenizer and the key/value cache
  pages actually used), measured with `getrusage`.

## This is not a new idea

Streaming layers from disk is what llama.cpp's memory mapping, AirLLM, FlexGen and Hugging
Face's offloading do in different ways. This project is a small, readable, dependency-free
version that is checked against the reference implementation and measured carefully, including
the costs.

## How it works

```text
 model.safetensors / model.sealed                       compute threads
        │                                                     ▲
        ▼                                                     │
  Source ── read (+ decrypt) ──▶ [buffer A] ──▶ layer k ──────┤
  (reads one layer's bytes)      [buffer B] ◀── read layer k+1 (background thread)
```

- **`Source`** fetches the bytes of any layer or any range of embedding rows: from the plain file,
  or from an encrypted file (decrypting in parallel).
- **`Provider`** decides what stays in memory. `ResidentProvider` loads everything.
  `StreamProvider` keeps `slots` layer-sized buffers; a background thread fills the next one
  while the current one computes, so reading overlaps with computing.
- **`Model`** runs the Qwen2 transformer (grouped-query attention, rotary embeddings, SwiGLU)
  one layer at a time, with a key/value cache. The prompt is pushed through each layer as a
  batch, so a streamed layer is read once per batch, not once per prompt token.
- **The output layer is streamed too.** The model ties its input and output embeddings, so
  computing the logits means reading the 272 MB embedding table once per token (29% of the
  942 MB read per token).
- Weights stay in bf16 as stored on disk and are widened to f32 on the fly, so a layer takes
  2 bytes per parameter in memory. There is no quantization.

## Correctness

- Output logits match Hugging Face `transformers` (float32) to 0.0000625 at most, and greedy
  generation produces the same words, on three prompts.
- The tokenizer gives identical token ids to the reference on 24 designed strings (contractions,
  digits, newlines, Chinese, emoji, special tokens) and on 1,210 chunks of real source code and
  prose (279,261 characters).
- Streaming (1, 2 or 3 buffers), resident and encrypted modes give **bit-for-bit identical**
  logits.
- The encrypted file was checked with an independent implementation: Python's `cryptography`
  decrypts sampled chunks to exactly the original weights, and a single flipped bit is rejected.
- 51 tests (`cargo test --release`). 9 of them run a whole synthetic model, so they need no
  download; the tests that use the real model skip themselves loudly if it is absent.

## Encryption at rest (optional)

`layerstream seal` encrypts the model with ChaCha20-Poly1305 in 1 MiB chunks; layers are
decrypted in parallel as they are read. A wrong key, or any modified byte, is detected, and the
program exits without producing text.

- **What it gives:** a copy of the encrypted file is useless without the key.
- **What it does not give:** the key and each layer are in memory while the model runs, so anyone
  who can read this process's memory gets them. `--wipe` overwrites each buffer as soon as it is
  used, which shortens that window.
- **The cipher is hand-written** to keep the crate dependency-free. It passes the RFC 8439 test
  vectors and matches Python's `cryptography` package on 23 message lengths, but it has not been
  audited. For anything real, use a vetted library.

## Quick start

```bash
# 1. get the code, then the model (988 MB, Apache-2.0)
git clone https://github.com/themaker00001/layerstream && cd layerstream
mkdir -p models/Qwen2.5-0.5B-Instruct && cd models/Qwen2.5-0.5B-Instruct
B=https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main
for f in config.json generation_config.json tokenizer_config.json vocab.json merges.txt model.safetensors; do
  curl -L -O $B/$f; done
cd ../..

# 2. build and run
cargo build --release
M=models/Qwen2.5-0.5B-Instruct
./target/release/layerstream generate --model $M --prompt "What is the capital of France?"
./target/release/layerstream generate --model $M --mode resident     # everything in memory
./target/release/layerstream generate --model $M --throttle-mbps 500 --max-new 8   # emulate slow storage

# 3. optional: encrypt, then run from the encrypted file
./target/release/layerstream gen-key models/my.key
./target/release/layerstream seal --model $M --out models/qwen.sealed --key-file models/my.key
./target/release/layerstream generate --model $M --sealed models/qwen.sealed --key-file models/my.key

# 4. reproduce the table (each configuration in its own process)
./target/release/layerstream bench --model $M --sealed models/qwen.sealed --key-file models/my.key
```

The reference scripts need Python with `torch` and `transformers` (`scripts/reference.py`, which
produced `tests/golden.json`) or `cryptography` (`scripts/verify_sealed.py`). Neither is needed to
build or run the Rust program.

## Limits

- Qwen2-architecture models with tied embeddings and bf16 weights only.
- No quantization. It is the obvious next step: INT8 would halve the bytes read per token, which
  is what limits speed on slow storage.
- Tokenizer: Unicode letter and number classes use Rust's `is_alphabetic` and `is_numeric`, and
  text is not NFC-normalised first. It matched the reference on everything tested.
- Greedy decoding is the default and the one that was verified; temperature, top-k and top-p
  sampling exist and are unit-tested but were not compared against the reference.
- The key/value cache is f32 and reserved up front (24 MB at the default 1,024-token context);
  only the pages actually used count towards resident memory.
- Unix only (positioned reads, `getrusage`).

## Things I got wrong along the way

Kept here because they are the kind of mistake worth checking for:

- The tokenizer first cost **43 MB**, more than the two layers being streamed, because of the way
  the vocabulary was stored. It now costs about 8 MB (14 MB including the process).
- A "cold SSD" benchmark row was not cold (see above), so it was removed.
- The first version of the storage throttle delivered only 85% of its cap, because sleep
  overshoot was never made up. It now sustains 96%.
- A "mismatch" between my tokenizer and the reference on 51 of 1,210 real chunks turned out to be
  my test harness mangling backslashes in transit, not the tokenizer.

## Layout

| Path | What |
|---|---|
| `src/json.rs`, `src/safetensors.rs` | JSON parser and weights-file header reader |
| `src/tokenizer.rs` | Qwen2 byte-level BPE |
| `src/weights.rs` | config and the layout of a layer's 12 tensors |
| `src/source.rs`, `src/seal.rs` | where bytes come from: plain file, or encrypted file |
| `src/provider.rs` | resident vs streaming (with read-ahead) |
| `src/model.rs`, `src/kernels.rs`, `src/pool.rs` | transformer, numeric kernels, thread pool |
| `src/main.rs` | command line (`generate`, `bench`, `seal`, `gen-key`, `tokenize`) |
| `tests/` | reference comparison, synthetic-model pipeline tests, `golden.json` |
| `scripts/` | `reference.py` (ground truth from Hugging Face), `verify_sealed.py` |

## Licence

The code is MIT licensed (see `LICENSE`). The Qwen2.5-0.5B-Instruct weights are not part of this
repository; they are published by Alibaba Cloud under Apache-2.0.
