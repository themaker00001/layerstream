# layerstream

Run a language model that is bigger than your memory budget by keeping only a couple of
layers in memory at a time, and measure what that costs.

A small Rust program with no dependencies. The tokenizer, the weights reader, the transformer,
the (optional) 8-bit and 4-bit quantization and the (optional) encryption are written from
scratch and checked against the reference implementations. It runs **Qwen2.5-0.5B-Instruct**,
a real pretrained chat model.

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
- **Fewer bits:** storing the weights as 8-bit integers halves the bytes read per token and about
  doubles the speed, with no quality loss I could measure; 4-bit is faster still but costs
  quality on this small model. See [Quantization](#quantization-fewer-bytes-per-token).

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
  pages actually used), measured with `getrusage`. It does not include the operating system's
  file cache; see [Memory accounting](#memory-accounting-the-process-is-not-the-whole-machine).

## Memory accounting: the process is not the whole machine

The "peak memory" in the tables is the program's own peak resident memory. It does not include
the operating system's **file cache**: when a program reads a file, the system keeps the pages
in RAM so the next read is fast. That memory belongs to the system and can be dropped at any
time, but while it is held, the machine is using it. So a streamed run touches more RAM than
its peak says, whenever the system has room to cache the weights.

Every run now reports this. `generate` prints how much of the weights file the system held in
its cache before and after the run, plus the system's free-memory percentage and pressure level
(macOS); `bench` has an "OS cache MB" column; `layerstream cache FILE [--read]` measures any
file. The probe was checked against files whose state is known: a large file nobody had read
reported 0% cached, and the same file reported 100% right after it was read.

| Mode | Process peak | Weights file in the OS cache after the run | Process + cache |
|---|---:|---:|---:|
| bf16, everything in memory | 962 MB | 942 MB | about 1,900 MB (the model is held twice) |
| bf16, streamed | 76 MB | 942 MB | about 1,020 MB |
| int8, streamed | 49 MB | 486 MB | about 535 MB |
| int4, streamed | 36 MB | 295 MB | about 330 MB |

What this means:

- **The process numbers are right, but they are not "the memory a device needs".** Streaming
  keeps the program at 76 MB (49 MB, 36 MB). It does not stop the system from also holding the
  file in its cache when it has the room.
- **On a device with little RAM the cache cannot hold the file.** Reads then come from flash, so
  the "file cached" speed rows do not apply and the capped-storage rows are the relevant ones.
  Streaming still works there, with that small process footprint.
- **The cache is not dependable on a busy machine.** When I started the last benchmark none of
  the weights files were in the cache (0 MB), although I had run them hours earlier: this Mac
  was under memory pressure (about 45% free, heavy swap use) and had dropped them. The first
  run of each file read it from storage (about 3 GB/s here) and left it cached for the runs
  after it. Most streamed rows ran with a warm cache because a run that loads everything comes
  first; the "OS cache MB" column shows which rows did not.
- **What I could not do.** Clearing the cache needs administrator rights, and macOS has no way
  to cap the total memory of one program, so I could not run a truly cold or RAM-limited test.
  A Linux cgroup memory limit with the cache dropped would be the proper test.

## Quantization: fewer bytes per token

Streaming speed is set by how many bytes must be read for every token, so storing the weights
in fewer bits speeds up everything that is limited by storage, and shrinks the layer buffers
too. `layerstream quantize` turns the bf16 model into a file where every weight is an 8-bit or
4-bit integer, with one shared scale for each group of 16 to 64 weights of a row. It takes about
a second.

```text
$ layerstream quantize --model models/Qwen2.5-0.5B-Instruct --out models/int8.lsq --bits 8 --group 64
[ok] int8, groups of 64 (clip search): 942 MB -> 486 MB (1.94x smaller); weight error 0.64% of the weights' RMS; 0.6 s
```

**Speed and memory.** Same machine and method as the table above; greedy decoding, every
configuration in a fresh process (`layerstream bench --quantized ...`).

| Weights | File (= read per token) | Peak memory, streamed | Tokens/s, file cached | Tokens/s, storage capped at 2,000 MB/s | Tokens/s, capped at 500 MB/s | Tokens/s, encrypted and streamed |
|---|---:|---:|---:|---:|---:|---:|
| bf16 (as shipped) | 942 MB | 76 MB | 13 | 2.0 | 0.5 | 2.8 |
| int8, groups of 64 | 486 MB | 49 MB | 25 | 3.9 | 1.0 | 5.2 |
| int4, groups of 16 | 295 MB | 36 MB | 41 | 6.5 | 1.6 | 8.2 |

Everything held in memory instead of streamed: 962 MB at 33 tokens/s (bf16), 505 MB at 61 to 63
tokens/s (int8), 314 MB at 49 tokens/s (int4).

Run to run, the cached streaming speeds varied by about 10% (int8: 24 to 27 tokens/s over seven
runs; int4: 41 to 42 over four; bf16: 12 to 13.6). One early benchmark run, which started with
every weights file cold on a machine under memory pressure, came out two to three times slower
on several streamed rows; running the same benchmark again gave the numbers above.

**Quality.** Perplexity (lower is better) and agreement with the bf16 model, scoring 512 tokens
of two texts token by token: frozen copies of this README ("prose") and of `src/model.rs`
("code") as of commit 3aacc7f, kept in `eval/` so the numbers cannot drift when those files change. "Top choice" is how often the model's most likely next
token matches the bf16 model's. "Answers" is how many of six prompts gave a word-for-word
identical 48-token greedy answer to bf16's. Reproduce with `scripts/quant_eval.sh`.

| Weights | File | Perplexity, prose | Perplexity, code | Top choice = bf16's, prose / code | Answers identical |
|---|---:|---:|---:|---:|---:|
| bf16 | 942 MB | 12.65 | 4.83 | 100% / 100% | 6 of 6 |
| int8, groups of 64 | 486 MB | 12.61 | 4.83 | 98.4% / 99.2% | 3 of 6 |
| int8, groups of 32 | 501 MB | 12.62 | 4.84 | 98.8% / 99.6% | 6 of 6 |
| int4, groups of 64 | 250 MB | 16.03 | 6.23 | 75.0% / 81.3% | 0 of 6 |
| int4, groups of 32 | 265 MB | 15.11 | 5.82 | 77.9% / 85.5% | 0 of 6 |
| int4, groups of 32, embedding int8 | 326 MB | 14.61 | 5.60 | 82.2% / 87.9% | 0 of 6 |
| int4, groups of 16 | 295 MB | 13.94 | 5.49 | 80.5% / 87.1% | 0 of 6 |

What it says:

- **8-bit costs nothing I could measure.** Perplexity is within 0.3% of bf16 on both texts, the
  top choice matches bf16's at 98 to 99.6% of positions, and with groups of 32 all six answers
  came out word for word the same as bf16's. With groups of 64, three of the six drift after 34
  to 42 tokens.
- **4-bit does cost quality on this model.** Perplexity is 10 to 29% higher and the answers
  differ from bf16's within the first few tokens. Some are wrong: asked why we have seasons,
  bf16 and int8 answer with the tilt of the Earth's axis, while int4 with groups of 32 says
  "gravitational forces of the planets and the Earth's rotation". A 0.5B model has little
  precision to spare. Larger models usually lose less at 4 bits, but I have not measured that
  here. Use 8-bit; 4-bit is for when memory or storage leaves no choice.
- **Speed follows bytes while storage is the limit.** Capped at 500 MB/s, int8 (1.94x fewer
  bytes) runs 2x faster and int4 (3.2x fewer) 3.2x faster. With the file cached the gains are
  about 2x and 3.2x too.
- **4-bit is not faster when everything is in memory.** Unpacking nibbles costs compute, so
  int4 resident (49 tokens/s) is slower than int8 resident (63). It only wins when bytes are the
  limit, which is the streaming case.
- **Encrypted streaming is limited by decryption** (about 2.5 GB/s here, with a hand-written
  cipher), so a smaller file helps directly: 2.8, 5.2 and 8.2 tokens/s. Quantize first, then
  encrypt (`seal --quantized`).

Things that did not help, kept here because they were reasonable guesses:

- **Clip search** (each group tries a few clipping ratios and keeps the best) lowers the weight
  error (4-bit, groups of 32: 10.2% to 9.7% of the weights' RMS) but not the perplexity (15.116
  with plain rounding, `--rtn`, against 15.110). It is on by default and harmless; it is not a win.
- **Keeping the embedding table at 8 bits** under 4-bit layers helped (15.11 to 14.61) but costs
  61 MB. Halving the group size to 16 does better (13.94) for less (295 MB against 326 MB), so
  groups of 16 is the best 4-bit setting I found.

Caveats: perplexity is measured on 512 tokens of two small texts, not a standard benchmark, so
differences below about 0.1 are noise; one 0.5B model; and this is plain weight rounding with
no calibration data, which is why 4-bit loses what it does (methods such as GPTQ or AWQ keep
more at 4 bits).

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
- Quantized models: when a model's weights lie exactly on the quantization grid, the quantized
  engine and the bf16 engine produce the same logits (to 1e-3), for 8-bit, 4-bit, several group
  sizes and a different format for the embedding table. All streaming and encrypted modes of a
  quantized model give bit-for-bit identical logits. Deliberately breaking the 4-bit nibble order
  or the embedding format makes these tests fail.
- 67 tests (`cargo test --release`). 15 of them run a whole synthetic model, so they need no
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

# 3. optional: quantize (takes a second), then run from the smaller file
./target/release/layerstream quantize --model $M --out models/int8.lsq --bits 8 --group 64
./target/release/layerstream quantize --model $M --out models/int4.lsq --bits 4 --group 16
./target/release/layerstream generate --model $M --quantized models/int8.lsq

# 4. optional: encrypt (the plain or the quantized model), then run from the encrypted file
./target/release/layerstream gen-key models/my.key
./target/release/layerstream seal --model $M --out models/qwen.sealed --key-file models/my.key
./target/release/layerstream seal --quantized models/int8.lsq --out models/int8.sealed --key-file models/my.key
./target/release/layerstream generate --model $M --sealed models/int8.sealed --key-file models/my.key

# 5. reproduce the tables (each configuration in its own process)
./target/release/layerstream bench --model $M --sealed models/qwen.sealed --key-file models/my.key \
    --quantized models/int8.lsq --quantized-sealed models/int8.sealed
scripts/quant_eval.sh $M          # perplexity and answer agreement for every quantized variant
```

The reference scripts need Python with `torch` and `transformers` (`scripts/reference.py`, which
produced `tests/golden.json`) or `cryptography` (`scripts/verify_sealed.py`). Neither is needed to
build or run the Rust program.

## Limits

- Qwen2-architecture models with tied embeddings and bf16 weights only.
- Quantization is weight-only (activations and the key/value cache stay f32), symmetric, with
  one scale per group and plain rounding or a small clip search. There is no calibration data
  (nothing like GPTQ or AWQ), which is why 4-bit costs real quality on this small model.
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
| `src/weights.rs` | config, weight formats, and the layout of a layer's 12 tensors |
| `src/quant.rs` | the quantizer: bf16 model to 8-bit / 4-bit file |
| `src/source.rs`, `src/seal.rs` | where bytes come from: plain file, or encrypted file |
| `src/provider.rs` | resident vs streaming (with read-ahead) |
| `src/model.rs`, `src/kernels.rs`, `src/pool.rs` | transformer, numeric kernels, thread pool |
| `src/main.rs` | command line (`generate`, `bench`, `quantize`, `eval`, `cache`, `seal`, `gen-key`, `tokenize`) |
| `src/sys.rs` | peak memory, OS file-cache and memory-pressure probes, buffer wiping |
| `tests/` | reference comparison, synthetic-model pipeline and quantization tests, `golden.json` |
| `scripts/` | `reference.py` (ground truth from Hugging Face), `verify_sealed.py`, `quant_eval.sh` (quality of each quantized variant) |

## Licence

The code is MIT licensed (see `LICENSE`). The Qwen2.5-0.5B-Instruct weights are not part of this
repository; they are published by Alibaba Cloud under Apache-2.0.
