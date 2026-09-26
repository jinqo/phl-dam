"""SlotMemory: the PHL-DAM v4 memory layer as a drop-in module for any model.

This is the reference implementation of IMPLEMENTATION_PROMPT_150M.md section
2, written against hidden states rather than token embeddings so it can sit
inside a Transformer (or any residual network). Per sequence it keeps N slots
of (key, value, occupancy), processes positions left to right, and at each
step writes, then reads:

    k_t = unit(W_k h_{t-1})    v_t = W_v h_t    q_t = unit(W_k h_t)   (tied)
    address = g * dnc_allocation(occupancy) + (1-g) * softmax(beta * K k_t)
    write   = w_t * address
    K <- unit((1-write) K + write k_t);  U <- (1-write) U + write v_t
    o <- (1-write) o + write
    a_t = softmax(K q_t / 0.1 + 0.25 log(o + 1e-6))
    r_t = sum_n a_t[n] U[n] / max(o[n], 1e-3)
    m_t = sigmoid(read gate) * r_t

It returns ``W_o m_t`` for the residual stream and ``m_t`` for an optional
copy readout at the LM head: ``logits += copy_scale * m_t @ (W_v E)^T``.

``reset`` (bool, [batch, T]) zeroes the state *before* position t - use it at
document boundaries so no binding crosses documents.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def unit(x: Tensor, eps: float = 0.05) -> Tensor:
    """Bounded unit-normalisation; F.normalize on a zero vector has Jacobian 1e12."""
    return x / (x.pow(2).sum(-1, keepdim=True) + eps * eps).sqrt()


def memory_step(keys: Tensor, values: Tensor, occ: Tensor, k: Tensor, v: Tensor,
                write_gate: Tensor, alloc_gate: Tensor, sharpness: Tensor, q: Tensor,
                read_temperature: float, floor: float):
    """One write-then-read step of the slot recurrence (all fp32)."""
    sorted_occ, order = occ.sort(dim=-1)
    exclusive = torch.cat([torch.ones_like(sorted_occ[:, :1]),
                           sorted_occ[:, :-1]], dim=-1).cumprod(dim=-1)
    alloc = torch.zeros_like(occ).scatter(1, order, (1.0 - sorted_occ) * exclusive)
    content = torch.softmax(sharpness[:, None] * torch.einsum("bd,bnd->bn", k, keys), -1)
    g = alloc_gate[:, None]
    write = write_gate[:, None] * (g * alloc + (1.0 - g) * content)
    remain = 1.0 - write
    keys = unit(remain[..., None] * keys + write[..., None] * k[:, None])
    values = remain[..., None] * values + write[..., None] * v[:, None]
    occ = remain * occ + write
    score = torch.einsum("bd,bnd->bn", q, keys) / read_temperature
    attn = torch.softmax(score + 0.25 * torch.log(occ + 1e-6), dim=-1)
    retrieved = torch.einsum("bn,bnv->bv", attn, values / occ.clamp_min(floor)[..., None])
    entropy = -(attn * attn.clamp_min(1e-12).log()).sum(-1)
    return keys, values, occ, retrieved, entropy


_COMPILED_STEP = None


def compiled_memory_step():
    """torch.compile'd step (about 2x faster on CPU; same function)."""
    global _COMPILED_STEP
    if _COMPILED_STEP is None:
        _COMPILED_STEP = torch.compile(memory_step, dynamic=False)
    return _COMPILED_STEP


class SlotMemory(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_slots: int = 16,
        d_key: int = 64,
        d_value: int = 64,
        read_temperature: float = 0.1,
        write_gate_bias: float = -3.0,
        read_gate_bias: float = 1.0,
        copy_scale: float = 2.0,
        occupancy_floor: float = 1e-3,
        compile_step: bool = False,
    ) -> None:
        super().__init__()
        self.num_slots, self.d_key, self.d_value = num_slots, d_key, d_value
        self.read_temperature = read_temperature
        self.occupancy_floor = occupancy_floor
        self.compile_step = compile_step
        self.key = nn.Linear(d_model, d_key, bias=False)        # tied key/query
        self.value = nn.Linear(d_model, d_value, bias=False)
        self.gates = nn.Linear(d_model, 3)                       # write, alloc, sharpness
        self.read_gate = nn.Linear(d_model + d_value + 1, 1)
        self.out = nn.Linear(d_value, d_model, bias=False)
        self.copy_scale = nn.Parameter(torch.tensor(float(copy_scale)))
        with torch.no_grad():
            self.gates.bias.copy_(torch.tensor([write_gate_bias, 0.0, 0.0]))
            self.read_gate.bias.fill_(read_gate_bias)

    def state_floats(self) -> int:
        return self.num_slots * (self.d_key + self.d_value + 1)

    def forward(self, h: Tensor, reset: Tensor | None = None,
                disable: bool = False) -> tuple[Tensor, Tensor]:
        batch, length, _ = h.shape
        previous = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
        if reset is not None:                   # no key from across a boundary
            previous = previous.masked_fill(reset[..., None], 0.0)
        cand_keys = unit(self.key(previous)).float()
        queries = unit(self.key(h)).float()
        cand_values = self.value(h).float()
        gates = self.gates(h).float()
        write_gate = torch.sigmoid(gates[..., 0])
        alloc_gate = torch.sigmoid(gates[..., 1])
        sharpness = F.softplus(gates[..., 2]) + 1.0

        n = self.num_slots
        keys = h.new_zeros(batch, n, self.d_key, dtype=torch.float32)
        values = h.new_zeros(batch, n, self.d_value, dtype=torch.float32)
        occ = h.new_zeros(batch, n, dtype=torch.float32)
        retrieved, entropy = [], []
        step = compiled_memory_step() if self.compile_step else memory_step
        for t in range(length):
            if reset is not None:
                keep = (~reset[:, t]).float()
                keys, values = keys * keep[:, None, None], values * keep[:, None, None]
                occ = occ * keep[:, None]
            keys, values, occ, r_t, e_t = step(
                keys, values, occ, cand_keys[:, t], cand_values[:, t], write_gate[:, t],
                alloc_gate[:, t], sharpness[:, t], queries[:, t],
                self.read_temperature, self.occupancy_floor)
            retrieved.append(r_t)
            entropy.append(e_t)
        r = torch.stack(retrieved, 1).to(h.dtype)
        conf = (1.0 - torch.stack(entropy, 1) / math.log(n)).to(h.dtype)
        rho = torch.sigmoid(self.read_gate(torch.cat([h, r, conf[..., None]], -1)))
        m = rho * r
        if disable:
            m = torch.zeros_like(m)
        return self.out(m), m

    def copy_logits(self, m: Tensor, embedding: Tensor) -> Tensor:
        """Pointer-style readout: score m against every vocabulary item's value."""
        return self.copy_scale * m @ self.value(embedding).T
