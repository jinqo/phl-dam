"""SlotMemory reference implementation (the code embedded in IMPLEMENTATION_PROMPT_150M.md).

Same maths and parameter names as work/slot_memory.py (the module every LM run
used), without the dev-only options that did not help (surprise-gated write,
values from token embeddings), plus a streaming API for generation.
test_slot_memory.py checks it matches slot_memory.py exactly and that
step-by-step decoding matches the parallel forward.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def unit(x: Tensor, eps: float = 0.05) -> Tensor:
    """Bounded unit-normalisation. F.normalize on a zero vector has Jacobian ~1e12."""
    return x / (x.pow(2).sum(-1, keepdim=True) + eps * eps).sqrt()


def memory_step(keys: Tensor, values: Tensor, occ: Tensor, k: Tensor, v: Tensor,
                write_gate: Tensor, alloc_gate: Tensor, sharpness: Tensor, q: Tensor,
                read_temperature: float, floor: float):
    """One write-then-read step of the slot recurrence. All tensors fp32.

    keys [B,N,dk], values [B,N,dv], occ [B,N]; k, q [B,dk]; v [B,dv];
    write_gate, alloc_gate, sharpness [B].
    """
    # DNC allocation: least-occupied slots first (sort indices carry no gradient)
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
    # read after write; occupancy prior; values normalised by occupancy
    score = torch.einsum("bd,bnd->bn", q, keys) / read_temperature
    attn = torch.softmax(score + 0.25 * torch.log(occ + 1e-6), dim=-1)
    retrieved = torch.einsum("bn,bnv->bv", attn, values / occ.clamp_min(floor)[..., None])
    entropy = -(attn * attn.clamp_min(1e-12).log()).sum(-1)
    return keys, values, occ, retrieved, entropy


class SlotMemory(nn.Module):
    """External slot memory as a residual sublayer.

    forward(h) -> (delta, m): add `delta` to the residual stream; pass `m` to
    copy_logits() at the LM head. `h` is the pre-normed hidden state at the
    insertion point, [B, T, d_model].
    """

    def __init__(self, d_model: int, num_slots: int = 16, d_key: int = 64,
                 d_value: int = 64, read_temperature: float = 0.1,
                 write_gate_bias: float = -3.0, read_gate_bias: float = 1.0,
                 copy_scale: float = 2.0, occupancy_floor: float = 1e-3,
                 compile_step: bool = False) -> None:
        super().__init__()
        self.num_slots, self.d_key, self.d_value = num_slots, d_key, d_value
        self.read_temperature, self.occupancy_floor = read_temperature, occupancy_floor
        self.key = nn.Linear(d_model, d_key, bias=False)        # tied key/query
        self.value = nn.Linear(d_model, d_value, bias=False)
        self.gates = nn.Linear(d_model, 3)                       # write, alloc, sharpness
        self.read_gate = nn.Linear(d_model + d_value + 1, 1)
        self.out = nn.Linear(d_value, d_model, bias=False)
        self.copy_scale = nn.Parameter(torch.tensor(float(copy_scale)))
        self.write_gate_bias, self.read_gate_bias = write_gate_bias, read_gate_bias
        self.step_fn = torch.compile(memory_step, dynamic=False) if compile_step else memory_step
        self.reset_gate_biases()

    def reset_gate_biases(self) -> None:
        """Call AFTER any model-wide init loop - a generic init zeroes these biases."""
        with torch.no_grad():
            self.gates.bias.copy_(torch.tensor([self.write_gate_bias, 0.0, 0.0]))
            self.read_gate.bias.fill_(self.read_gate_bias)

    def state_floats(self) -> int:
        return self.num_slots * (self.d_key + self.d_value + 1)

    # -- projections, parallel over positions ---------------------------------
    def _project(self, h: Tensor, previous: Tensor):
        cand_keys = unit(self.key(previous)).float()             # key from h_{t-1}
        queries = unit(self.key(h)).float()                      # query from h_t (tied W_k)
        cand_values = self.value(h).float()                      # value from h_t
        gates = self.gates(h).float()
        return (cand_keys, queries, cand_values, torch.sigmoid(gates[..., 0]),
                torch.sigmoid(gates[..., 1]), F.softplus(gates[..., 2]) + 1.0)

    def _read_out(self, h: Tensor, r: Tensor, entropy: Tensor) -> tuple[Tensor, Tensor]:
        r = r.to(h.dtype)
        conf = (1.0 - entropy / math.log(self.num_slots)).to(h.dtype)
        rho = torch.sigmoid(self.read_gate(torch.cat([h, r, conf[..., None]], -1)))
        m = rho * r
        return self.out(m), m

    def init_state(self, batch: int, device=None) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """(keys, values, occupancy, previous hidden) - all zero at a document start."""
        z = lambda *s: torch.zeros(*s, device=device, dtype=torch.float32)
        return (z(batch, self.num_slots, self.d_key), z(batch, self.num_slots, self.d_value),
                z(batch, self.num_slots), None)

    # -- training / prefill ----------------------------------------------------
    def forward(self, h: Tensor, reset: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """reset: bool [B, T], True where a new document starts (state zeroed before t)."""
        batch, length, _ = h.shape
        previous = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
        if reset is not None:                    # no key taken from across a boundary
            previous = previous.masked_fill(reset[..., None], 0.0)
        k, q, v, w, g, beta = self._project(h, previous)
        keys, values, occ, _ = self.init_state(batch, h.device)
        retrieved, entropy = [], []
        for t in range(length):
            if reset is not None:
                keep = (~reset[:, t]).float()
                keys, values = keys * keep[:, None, None], values * keep[:, None, None]
                occ = occ * keep[:, None]
            keys, values, occ, r_t, e_t = self.step_fn(
                keys, values, occ, k[:, t], v[:, t], w[:, t], g[:, t], beta[:, t], q[:, t],
                self.read_temperature, self.occupancy_floor)
            retrieved.append(r_t)
            entropy.append(e_t)
        return self._read_out(h, torch.stack(retrieved, 1), torch.stack(entropy, 1))

    # -- generation: one position at a time, constant-size state --------------
    def step(self, h_t: Tensor, state) -> tuple[Tensor, Tensor, tuple]:
        """h_t [B, d_model] -> (delta [B, d_model], m [B, d_value], new state)."""
        keys, values, occ, previous = state
        if previous is None:
            previous = torch.zeros_like(h_t)
        k, q, v, w, g, beta = self._project(h_t, previous)
        keys, values, occ, r, e = memory_step(keys, values, occ, k, v, w, g, beta, q,
                                              self.read_temperature, self.occupancy_floor)
        delta, m = self._read_out(h_t, r, e)
        return delta, m, (keys, values, occ, h_t)

    def copy_logits(self, m: Tensor, embedding: Tensor) -> Tensor:
        """Pointer-style readout: m scored against every vocabulary item's value, W_v E."""
        return self.copy_scale * m @ self.value(embedding).T
