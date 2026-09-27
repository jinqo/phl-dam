"""Is the slot memory worth adding to a Transformer language model?

A small-scale proxy for the 150M question, on real text. Character-level
(byte) language models on WikiText-2, three arms at matched parameters:

* ``window``      Transformer whose attention sees only the last W positions
                  (the stand-in for any finite-context model);
* ``window_sml``  the same plus one SlotMemory layer (``slot_memory.py``) and
                  its copy readout, FFN of that block shrunk to match params;
* ``full``        Transformer with full causal attention over the sequence
                  (reference: what unlimited attention buys, at a KV cache
                  that grows with length).

Training data: WikiText-2 train, random crops of T bytes; a fraction of
crops (every arm alike) carry a passkey - "The pass key is 48213." near the
start and "The pass key is 48213" at the end - so retrieval beyond the window
is learnable.

Evaluation (fixed held-out sets from WikiText-2 valid):
1. bits per byte on natural text (must not get worse);
2. passkey digit accuracy and whole-key accuracy (distance >> W);
3. bits per byte on *long-range repeats*: positions whose preceding 8 bytes
   plus the current byte occurred earlier in the crop beyond the windowed
   stack's reach (layers x (W-1) back) and not within it - natural text
   where only memory beyond the attention reach can help.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from slot_memory import SlotMemory

# Fetched from raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/
DATA = Path(__file__).resolve().parent.parent / "data_cache"
VOCAB = 256
PASSKEY_PREFIX = b"The pass key is "


def load_bytes(split: str) -> Tensor:
    raw = (DATA / f"wikitext-2_{split}.txt").read_bytes()
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()


def passkey_crop(text: Tensor, length: int, rng: random.Random) -> tuple[Tensor, list[int]]:
    """A natural crop with a key stated near the start and asked at the end."""
    digits = "".join(rng.choice("0123456789") for _ in range(5)).encode()
    statement = PASSKEY_PREFIX + digits + b". "
    question = PASSKEY_PREFIX + digits
    start = rng.randrange(0, len(text) - length)
    crop = text[start:start + length].clone()
    at = rng.randrange(2, 20)
    crop[at:at + len(statement)] = torch.tensor(list(statement))
    crop[length - len(question):] = torch.tensor(list(question))
    answer_positions = list(range(length - 5, length))          # the 5 digits
    return crop, answer_positions


class Block(nn.Module):
    def __init__(self, d: int, heads: int, ffn: int, window: int | None,
                 memory: SlotMemory | None = None) -> None:
        super().__init__()
        self.heads, self.window = heads, window
        self.norm1, self.norm2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        self.memory = memory
        if memory is not None:
            self.norm_m = nn.RMSNorm(d)

    def forward(self, x: Tensor, mask: Tensor, disable_memory: bool,
                reset: Tensor | None = None, surprise_fn=None):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, d // self.heads).unbind(2)
        att = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask)
        x = x + self.proj(att.transpose(1, 2).reshape(b, t, d))
        m = None
        if self.memory is not None:
            hm = self.norm_m(x)
            surprise = surprise_fn(hm) if surprise_fn is not None else None
            delta, m = self.memory(hm, reset=reset, disable=disable_memory, surprise=surprise)
            x = x + delta
        return x + self.ffn(self.norm2(x)), m


class LM(nn.Module):
    def __init__(self, arm: str, length: int, window: int, d: int = 128, layers: int = 4,
                 heads: int = 4, ffn: int = 512, memory_layer: int = 1,
                 slots: int = 16, d_mem: int = 64, compile_step: bool = False,
                 surprise_gate: bool = False) -> None:
        super().__init__()
        self.arm, self.length, self.window = arm, length, window
        self.surprise_gate = surprise_gate and arm.startswith("window_sml")
        self.aux_loss = None
        self.embed = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(length, d)
        self.memory = None
        blocks = []
        for i in range(layers):
            memory = None
            width = ffn
            if arm in ("window_sml", "window_sml_local") and i == memory_layer:
                memory = self.memory = SlotMemory(d, slots, d_mem, d_mem,
                                                  compile_step=compile_step)
                # shrink this block's FFN to pay for the memory's parameters
                extra = sum(p.numel() for p in memory.parameters())
                width = ffn - round(extra / (2 * d + 1))
            blocks.append(Block(d, heads, width, None if arm == "full" else window, memory))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.RMSNorm(d)
        if self.surprise_gate:
            # Predicts x_t from the memory layer's input at t-1 (tied to the
            # embedding); its loss on the actual byte is the write-gate surprise.
            self.surprise_head = nn.Linear(d, d, bias=False)
        # GPT-2-style initialisation. With the head tied to a unit-variance
        # embedding the initial logits have std ~sqrt(d) and training crawls.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        if self.memory is not None:            # keep the memory's gate initialisations
            with torch.no_grad():
                self.memory.gates.bias.copy_(torch.tensor([-3.0, 0.0, 0.0]))
                self.memory.read_gate.bias.fill_(1.0)
        span = length if arm == "full" else window
        i = torch.arange(length)
        allowed = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - span)
        self.register_buffer("mask", allowed, persistent=False)

    def forward(self, tokens: Tensor, disable_memory: bool = False) -> Tensor:
        t = tokens.shape[1]
        x = self.embed(tokens) + self.pos(torch.arange(t))
        reset = None
        if self.arm == "window_sml_local":
            # Control: memory wiped every `window` positions, so it can carry
            # nothing beyond the attention window - any gain it gives is local
            # capacity, not long-range memory.
            reset = (torch.arange(t) % self.window == 0)[None].expand(tokens.shape[0], t)
        memory_read = None
        surprise_fn = None
        if self.surprise_gate:
            def surprise_fn(hm: Tensor) -> Tensor:
                pred = self.surprise_head(hm[:, :-1]) @ self.embed.weight.T
                nll = F.cross_entropy(pred.reshape(-1, VOCAB), tokens[:, 1:].reshape(-1),
                                      reduction="none").view(tokens.shape[0], -1)
                self.aux_loss = nll.mean()
                first = torch.zeros_like(nll[:, :1])     # nothing precedes t = 0
                return torch.cat([first, nll.detach()], dim=1)
        for block in self.blocks:
            x, m = block(x, self.mask[:t, :t], disable_memory, reset, surprise_fn)
            if m is not None:
                memory_read = m
        logits = self.norm(x) @ self.embed.weight.T                # tied head
        if memory_read is not None:
            logits = logits + self.memory.copy_logits(memory_read, self.embed.weight)
        return logits


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
def long_range_mask(crop: list[int], window: int, context: int = 8) -> list[bool]:
    """Positions predictable only by copying from more than `window` back.

    Position t is flagged when the context+1 bytes ending at t occurred
    earlier ending more than `window` positions back, and did not occur ending
    within the last `window` positions (where attention could copy them).
    """
    s = bytes(crop)
    flags = [False] * len(s)
    for t in range(context, len(s)):
        pattern = s[t - context:t + 1]
        if pattern in s[max(0, t - window - context):t]:
            continue
        flags[t] = s.find(pattern, 0, max(0, t - window)) != -1
    return flags


_RARE_WORDS: set[bytes] | None = None


def rare_words(max_count: int = 5) -> set[bytes]:
    """Words occurring at most `max_count` times in WikiText-2 train."""
    global _RARE_WORDS
    if _RARE_WORDS is None:
        counts: dict[bytes, int] = {}
        for word in (DATA / "wikitext-2_train.txt").read_bytes().split():
            counts[word] = counts.get(word, 0) + 1
        _RARE_WORDS = {w for w, c in counts.items() if c <= max_count}
    return _RARE_WORDS


def rare_repeat_mask(crop: list[int], reach: int, min_length: int = 4) -> list[bool]:
    """Characters 2+ of a rare word whose previous occurrence is beyond `reach`.

    The word must be rare in the training data (so it cannot be completed from
    general knowledge), have occurred earlier in the crop starting more than
    `reach` positions back, and not have occurred within `reach`. Only memory
    of the earlier occurrence can predict its continuation.
    """
    s = bytes(crop)
    rare = rare_words()
    flags = [False] * len(s)
    starts = [i for i in range(len(s)) if (i == 0 or s[i - 1:i] == b" ") and s[i:i + 1] != b" "]
    for i in starts:
        j = s.find(b" ", i)
        j = len(s) if j == -1 else j
        word = s[i:j]
        if len(word) < min_length or word not in rare or j == len(s):
            continue
        earlier = [k for k in starts if k < i and s[k:k + len(word) + 1] == word + b" "]
        if not earlier or max(earlier) > i - reach:
            continue
        for t in range(i + 1, j):
            flags[t] = True
    return flags


def build_eval(valid: Tensor, length: int, window: int, seed: int = 1234, crops: int = 96,
               passkeys: int = 192, rare_crop_limit: int = 400):
    rng = random.Random(seed)
    natural = torch.stack([valid[s:s + length] for s in
                           (rng.randrange(0, len(valid) - length) for _ in range(crops))])
    # A stack of `layers` windowed attention layers reaches layers*(window-1)
    # back; only copies from beyond that are out of the baseline's reach.
    reach = 4 * (window - 1)
    repeat = torch.tensor([long_range_mask(c.tolist(), reach) for c in natural])
    # Rare-word repeats need far more text than a crop to occur often enough:
    # scan the whole validation set for crops that contain at least one.
    # Held-out text for this metric: WikiText-2 valid + test (neither is trained on).
    held_out = torch.cat([valid, load_bytes("test")])
    rare_crops, starts = [], list(range(0, len(held_out) - length, length // 2))
    rng.shuffle(starts)
    for s in starts:
        c = held_out[s:s + length]
        flags = rare_repeat_mask(c.tolist(), reach)
        if any(flags):
            rare_crops.append((c, torch.tensor(flags)))
        if len(rare_crops) >= rare_crop_limit:
            break
    rare_x = torch.stack([c for c, _ in rare_crops])
    rare_m = torch.stack([f for _, f in rare_crops])
    keys, answers = zip(*(passkey_crop(valid, length, rng) for _ in range(passkeys)))
    return natural, repeat, rare_x, rare_m, torch.stack(keys), answers[0]


@torch.no_grad()
def evaluate(model: LM, natural: Tensor, repeat: Tensor, rare_x: Tensor, rare_m: Tensor,
             keys: Tensor, answer_positions, disable_memory: bool = False) -> dict:
    model.eval()
    nat_loss, rep_loss, rep_n, tokens = 0.0, 0.0, 0, 0
    for i in range(0, len(natural), 16):
        x = natural[i:i + 16]
        logits = model(x[:, :-1], disable_memory)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1), reduction="none")
        loss = loss.view(x.shape[0], -1)
        nat_loss += loss.sum().item(); tokens += loss.numel()
        m = repeat[i:i + 16, 1:]
        rep_loss += loss[m].sum().item(); rep_n += int(m.sum())
    rare_loss, rare_n = 0.0, 0
    for i in range(0, len(rare_x), 16):
        x = rare_x[i:i + 16]
        logits = model(x[:, :-1], disable_memory)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1),
                               reduction="none").view(x.shape[0], -1)
        m = rare_m[i:i + 16, 1:]
        rare_loss += loss[m].sum().item(); rare_n += int(m.sum())
    digit_hits, key_hits = 0, 0
    pos = torch.tensor(answer_positions)
    for i in range(0, len(keys), 16):
        x = keys[i:i + 16]
        logits = model(x[:, :-1], disable_memory)
        pred = logits.argmax(-1)[:, pos - 1]
        hit = pred.eq(x[:, pos])
        digit_hits += int(hit.sum()); key_hits += int(hit.all(-1).sum())
    model.train()
    log2 = math.log(2)
    return {
        "bits_per_byte": nat_loss / tokens / log2,
        "long_range_repeat_bits_per_byte": rep_loss / max(rep_n, 1) / log2,
        "long_range_repeat_positions": rep_n,
        "rare_word_repeat_bits_per_byte": rare_loss / max(rare_n, 1) / log2,
        "rare_word_repeat_positions": rare_n,
        "passkey_digit_accuracy": digit_hits / (len(keys) * len(answer_positions)),
        "passkey_key_accuracy": key_hits / len(keys),
    }


def run(arm: str, seed: int, steps: int, batch: int, length: int, window: int,
        lr: float, passkey_fraction: float, eval_every: int, save: Path | None = None,
        memory_layer: int = 1, slots: int = 16, d_mem: int = 64,
        compile_step: bool = False, write_penalty: float = 0.0,
        surprise_gate: bool = False) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed + 10_000)
    train, valid = load_bytes("train"), load_bytes("valid")
    natural, repeat, rare_x, rare_m, keys, answers = build_eval(valid, length + 1, window)
    eval_sets = (natural, repeat, rare_x, rare_m, keys, answers)
    model = LM(arm, length, window, memory_layer=memory_layer, slots=slots, d_mem=d_mem,
               compile_step=compile_step, surprise_gate=surprise_gate)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    warmup = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup)
                                              * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    history, started, train_seconds = [], time.perf_counter(), 0.0
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        rows = []
        for _ in range(batch):
            if rng.random() < passkey_fraction:
                rows.append(passkey_crop(train, length + 1, rng)[0])
            else:
                s = rng.randrange(0, len(train) - length - 1)
                rows.append(train[s:s + length + 1])
        x = torch.stack(rows)
        logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1))
        if model.aux_loss is not None:
            loss = loss + 0.1 * model.aux_loss          # trains the surprise head
        if write_penalty > 0.0 and model.memory is not None:
            # Writing costs something: the memory must spend writes on content
            # worth keeping rather than overwrite itself at every position.
            loss = loss + write_penalty * model.memory.last_write_rate
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step(); sched.step()
        train_seconds += time.perf_counter() - t0
        if step % eval_every == 0 or step == steps:
            metrics = evaluate(model, *eval_sets)
            history.append({"step": step, "train_loss": loss.item(), "grad_norm": norm,
                            "train_seconds": train_seconds, **metrics})
            print(f"{arm} seed={seed} step={step} loss={loss.item():.3f} "
                  f"bpb={metrics['bits_per_byte']:.3f} rare={metrics['rare_word_repeat_bits_per_byte']:.3f} "
                  f"key={metrics['passkey_key_accuracy']:.2f} t={train_seconds:.0f}s", flush=True)
    final = history[-1]
    if save is not None:
        torch.save(model.state_dict(), save)
    ablated = (evaluate(model, *eval_sets, disable_memory=True)
               if arm.startswith("window_sml") else None)
    return {
        "experiment": "Slot memory in a Transformer LM - WikiText-2 bytes",
        "arm": arm, "seed": seed,
        "configuration": {"steps": steps, "batch": batch, "length": length, "window": window,
                          "lr": lr, "passkey_fraction": passkey_fraction,
                          "d_model": 128, "layers": 4, "heads": 4,
                          "memory_layer": memory_layer, "slots": slots, "d_mem": d_mem,
                          "write_penalty": write_penalty, "surprise_gate": surprise_gate},
        "parameters": parameter_count(model),
        "memory_state_floats": model.memory.state_floats() if model.memory else 0,
        "kv_cache_floats": 2 * 4 * 128 * (length if arm == "full" else window),
        "seconds_per_step": train_seconds / steps,
        "final": final, "memory_disabled": ablated, "history": history,
        "finite": all(torch.isfinite(p).all() for p in model.parameters()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("window", "window_sml", "window_sml_local", "full"),
                   required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--length", type=int, default=256)
    p.add_argument("--window", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--passkey-fraction", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--output", type=Path)
    p.add_argument("--save", type=Path, help="write the trained weights here")
    p.add_argument("--memory-layer", type=int, default=1)
    p.add_argument("--slots", type=int, default=16)
    p.add_argument("--d-mem", type=int, default=64)
    p.add_argument("--compile-step", action="store_true")
    p.add_argument("--write-penalty", type=float, default=0.0)
    p.add_argument("--surprise-gate", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    result = run(a.arm, a.seed, a.steps, a.batch, a.length, a.window, a.lr,
                 a.passkey_fraction, a.eval_every, a.save, a.memory_layer, a.slots, a.d_mem,
                 a.compile_step, a.write_penalty, a.surprise_gate)
    if a.output:
        a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("arm", "parameters", "memory_state_floats",
                                             "seconds_per_step", "final")}, indent=1))


if __name__ == "__main__":
    main()
