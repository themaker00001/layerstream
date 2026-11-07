#!/usr/bin/env python3
"""
Independently check a sealed model file: decrypt sampled chunks with Python's `cryptography`
package (not the Rust cipher) and compare them with the original weights.

    python3 scripts/verify_sealed.py models/Qwen2.5-0.5B-Instruct models/qwen.sealed models/test.key [N]

Checks every layer's first and last chunk, the first, last and N random embedding chunks,
the final norm, and that flipping a bit makes decryption fail.
"""

import json
import random
import struct
import sys

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

model_dir, sealed_path, key_path = sys.argv[1:4]
n_random = int(sys.argv[4]) if len(sys.argv) > 4 else 40

key = bytes.fromhex(open(key_path).read().strip())
aead = ChaCha20Poly1305(key)

sealed = open(sealed_path, "rb").read()
assert sealed[:4] == b"LSS1", "bad magic"
version, = struct.unpack_from("<I", sealed, 4)
assert version == 1
prefix = sealed[8:16]
hlen, = struct.unpack_from("<I", sealed, 16)
frame_end = 20 + hlen
header = json.loads(sealed[20:frame_end])
cfg = json.loads(header["config_json"])
L, V, H = cfg["num_hidden_layers"], cfg["vocab_size"], cfg["hidden_size"]
rb = H * 2
layer_bytes, layer_chunk = header["layer_bytes"], header["layer_chunk"]
ecr = header["embed_chunk_rows"]
layer_chunks = -(-layer_bytes // layer_chunk)
embed_chunks = -(-V // ecr)
embed_first = L * layer_chunks
final_idx = embed_first + embed_chunks
n_chunks = final_idx + 1
tags_off = frame_end
ct_base = tags_off + 16 * n_chunks
embed_bytes = V * rb
total_plain = L * layer_bytes + embed_bytes + header["final_norm_bytes"]
assert len(sealed) == ct_base + total_plain, "file length does not match the header"

# the original weights
st = open(f"{model_dir}/model.safetensors", "rb")
n, = struct.unpack("<Q", st.read(8))
meta = json.loads(st.read(n))
base = 8 + n


def tensor_span(names):
    a = min(meta[k]["data_offsets"][0] for k in names)
    b = max(meta[k]["data_offsets"][1] for k in names)
    return base + a, base + b


def plain_bytes(start, length):
    st.seek(start)
    return st.read(length)


layer_names = [[k for k in meta if k.startswith(f"model.layers.{l}.")] for l in range(L)]
layer_start = [tensor_span(names)[0] for names in layer_names]
embed_start = base + meta["model.embed_tokens.weight"]["data_offsets"][0]
norm_start = base + meta["model.norm.weight"]["data_offsets"][0]


def chunk_plain(idx):
    """(plain offset within the sealed plaintext stream, length, original bytes)."""
    if idx < embed_first:
        l, k = divmod(idx, layer_chunks)
        off = k * layer_chunk
        ln = min(layer_chunk, layer_bytes - off)
        return l * layer_bytes + off, ln, plain_bytes(layer_start[l] + off, ln)
    if idx < final_idx:
        k = idx - embed_first
        ln = min(ecr * rb, embed_bytes - k * ecr * rb)
        return L * layer_bytes + k * ecr * rb, ln, plain_bytes(embed_start + k * ecr * rb, ln)
    return L * layer_bytes + embed_bytes, header["final_norm_bytes"], plain_bytes(norm_start, header["final_norm_bytes"])


def decrypt(idx, ct_override=None):
    off, ln, original = chunk_plain(idx)
    ct = ct_override if ct_override is not None else sealed[ct_base + off: ct_base + off + ln]
    tag = sealed[tags_off + 16 * idx: tags_off + 16 * idx + 16]
    nonce = prefix + struct.pack("<I", idx)
    aad = sealed[:frame_end] + struct.pack("<I", idx)
    return aead.decrypt(nonce, ct + tag, aad), original


wanted = set()
for l in range(L):
    wanted.update([l * layer_chunks, l * layer_chunks + layer_chunks - 1])
wanted.update([embed_first, final_idx - 1, final_idx])
rng = random.Random(1)
wanted.update(rng.randrange(n_chunks) for _ in range(n_random))

for idx in sorted(wanted):
    got, want = decrypt(idx)
    assert got == want, f"chunk {idx} decrypts to something other than the original weights"
print(f"[ok] {len(wanted)} sampled chunks of {n_chunks} decrypt (with the Python library) to exactly the original weights")

# tamper checks
off, ln, _ = chunk_plain(embed_first)
ct = bytearray(sealed[ct_base + off: ct_base + off + ln])
ct[ln // 2] ^= 1
try:
    decrypt(embed_first, bytes(ct))
    raise SystemExit("FAIL: a flipped bit was not detected")
except InvalidTag:
    print("[ok] a single flipped ciphertext bit is rejected")
