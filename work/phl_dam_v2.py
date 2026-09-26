"""PHL-DAM v2: the Stage B model with every candidate improvement behind a switch.

``PHLDAMv2`` subclasses the Stage B ``PHLDAM`` so that, with default switches,
it creates the same parameters in the same order and computes the same
function (asserted by test). Each change can then be ablated on its own:

* ``fast`` - hoist every per-token projection (write/read keys, values, write
  gate, PHL input, memory readout and output head) out of the sequential loop.
  Only the slot recurrence and the PHL transport stay sequential. Same
  function, fewer tiny ops per timestep.
* ``write_gate_bias`` - Stage B initialises the write gate at sigmoid(-3), about
  5% open, so early writes are faint and the recall gradient reaching the
  addressing path is weak. The DNC baseline starts at sigmoid(-1).
* ``tie_query_key`` - read the slot bank with the same projection used to
  write it. At a query the current token *is* the key that was written, so a
  tied projection addresses correctly from initialisation instead of having to
  learn to align two independent projections first.
* ``learned_temperature`` / ``read_temperature`` - make the read temperature
  learnable, and choose where it starts (Stage B: fixed at 0.10).
* ``use_phl`` - drop the PHL horizon lattice; the state falls from 456 to 392
  floats, the DNC's size.
* ``direct_readout`` - feed the gated retrieved value straight into the output
  layer, beside the normalised residual stream, instead of adding a projection
  of it into the residual before RMSNorm. Early in training writes are ~5% open
  so the retrieved vector is tiny; summed into the residual and normalised
  with the much larger context it all but vanishes, starving the write gate of
  gradient. The DNC baseline reads out the direct way.
* ``normalized_values`` - read each slot's value divided by its occupancy.
  Writes are convex blends starting from zero, so ``values`` and
  ``occupancy`` accumulate with identical weights and their ratio is exactly
  the write-weighted *average* of what the slot received, independent of how
  far the write gate is open. Measured failure it targets: in Stage B the
  write gate collapses from 4.8% to 1.3% open in the first 25 steps (faint
  writes only add noise), sigmoid saturation then shrinks its gradient ~30-50x
  below the DNC's, and it sits in that trap for ~250 steps. With normalised
  values a faint write still retrieves at full strength, so the loss rewards
  write *selectivity* rather than write *magnitude*. Keys are already
  normalised; this treats values the same way.
* ``occupied_threshold`` - a slot is eligible for merge-on-matching-key only
  once its occupancy exceeds this (Stage B: 0.05). Early writes are ~2-3%
  open, so under 0.05 nothing ever counts as occupied, merging never fires,
  and the ~150 identical FILL writes per episode are spread round-robin over
  all eight slots, diluting the three bindings. A scale-free threshold lets
  repeated keys collapse into one slot from the first step, which is what
  the DNC's cosine content addressing does implicitly.
* ``key_bias`` - give the key and query projections a bias, as the DNC's do.
  A shared bias makes every key start with a common component, so early reads
  are near-uniform and retrieve an average of stored values; writing more at a
  binding then moves that average toward the answer, a smooth early gradient
  toward write selectivity that bias-free, sharply read keys do not provide.
* ``copy_readout`` - add ``scale * r . V(E)^T`` to the logits: score the
  retrieved value r against the value projection of every vocabulary token
  (a pointer/copy readout). A clean retrieval of token v is exactly V(E[v]),
  so it decodes correctly from initialisation. Measured motivation: in v2 the
  breakout (gate selectivity, addressing and recall all snapping on at step
  60-80) waits on the output layer learning to decode retrieved values, which
  rises only gradually (18% -> 39% -> 57% over the first 60 steps). One extra
  parameter, the learned ``copy_scale``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from phl_dam_stage_b import PHLDAM, VOCAB_SIZE

READ_TEMPERATURE = 0.10
MERGE_TEMPERATURE = 0.10
MERGE_SHARPNESS = 12.0
MERGE_THRESHOLD = 0.72
OCCUPANCY_FLOOR = 1e-3


class PHLDAMv2(PHLDAM):
    def __init__(
        self,
        *,
        fast: bool = False,
        write_gate_bias: float = -3.0,
        tie_query_key: bool = False,
        learned_temperature: bool = False,
        use_phl: bool = True,
        direct_readout: bool = False,
        normalized_values: bool = False,
        read_temperature: float = READ_TEMPERATURE,
        occupied_threshold: float = 0.05,
        key_bias: bool = False,
        copy_readout: bool = False,
        copy_scale: float = 1.0,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        d_key: int = 24,
        d_value: int = 24,
        num_slots: int = 8,
    ) -> None:
        super().__init__(d_model=d_model, d_key=d_key, d_value=d_value,
                         num_slots=num_slots, horizon_width=d_model // 4,
                         use_phl=use_phl)
        self.fast = fast
        self.vocab_size = vocab_size
        if vocab_size != VOCAB_SIZE:
            # Other tasks (the 004 pressure ladder) use a larger vocabulary.
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.output = nn.Linear(d_model, vocab_size)
        self.tie_query_key = tie_query_key
        nn.init.constant_(self.write_gate.bias, write_gate_bias)
        self.key_bias = key_bias
        if key_bias:
            self.key_projection = nn.Linear(d_model, d_key)
            self.query_projection = nn.Linear(d_model, d_key)
        if tie_query_key:
            del self.query_projection
        self.direct_readout = direct_readout
        self.copy_readout = copy_readout
        if copy_readout:
            self.copy_scale = nn.Parameter(torch.tensor(float(copy_scale)))
        self.normalized_values = normalized_values
        self.occupied_threshold = occupied_threshold
        if direct_readout:
            # The residual no longer carries memory, so its projection goes;
            # the output layer gains d_value inputs instead.
            del self.memory_projection
            self.output = nn.Linear(d_model + d_value, vocab_size)
        self.log_temperature = (
            nn.Parameter(torch.tensor(math.log(read_temperature)))
            if learned_temperature else None
        )
        self.read_temperature = read_temperature

    # ------------------------------------------------------------------
    def _query(self, current_token: Tensor) -> Tensor:
        projection = self.key_projection if self.tie_query_key else self.query_projection
        return F.normalize(projection(current_token), dim=-1)

    def _temperature(self) -> Tensor | float:
        if self.log_temperature is None:
            return self.read_temperature
        return self.log_temperature.exp()

    def state_floats(self) -> int:
        slots = self.num_slots * (self.d_key + self.d_value + 1)
        return slots + (self.horizons * self.horizon_width if self.use_phl else 0)

    # ------------------------------------------------------------------
    def forward(self, tokens: Tensor, disable_retrieval: bool = False,
                return_diagnostics: bool = False):
        if not (self.fast or self.tie_query_key or self.direct_readout
                or self.normalized_values or self.log_temperature is not None
                or self.read_temperature != READ_TEMPERATURE
                or self.occupied_threshold != 0.05 or self.key_bias
                or self.copy_readout or self.vocab_size != VOCAB_SIZE):
            return super().forward(tokens, disable_retrieval, return_diagnostics)
        if return_diagnostics:
            raise NotImplementedError("v2 fast path does not collect diagnostics")
        return self._fast_forward(tokens, disable_retrieval), None

    def _fast_forward(self, tokens: Tensor, disable_retrieval: bool) -> Tensor:
        context, previous, current = self.encode_features(tokens)
        batch, length, _ = context.shape
        device = tokens.device

        # Everything that depends only on the token stream, computed once.
        candidate_keys = F.normalize(self.key_projection(previous), dim=-1)
        candidate_values = self.value_projection(current)
        write_strengths = torch.sigmoid(self.write_gate(context)).squeeze(-1)
        queries = self._query(current)
        if self.use_phl:
            injected = self.phl_input(context).view(
                batch, length, self.horizons, self.horizon_width
            )
        temperature = self._temperature()

        keys = torch.zeros(batch, self.num_slots, self.d_key, device=device)
        values = torch.zeros(batch, self.num_slots, self.d_value, device=device)
        occupancy = torch.zeros(batch, self.num_slots, device=device)
        phl = torch.zeros(batch, self.horizons, self.horizon_width, device=device)
        slot_bias = self.slot_bias[None, :]
        minus_one = -torch.ones(batch, device=device)

        retrieved_steps = []
        phl_steps = []
        for t in range(length):
            if self.use_phl:
                transported = torch.einsum("ij,bjd->bid", self.phl_transport, phl)
                phl = self.phl_norm(transported + injected[:, t])
                phl_steps.append(phl.flatten(1))

            candidate_key = candidate_keys[:, t]
            occupied = occupancy > self.occupied_threshold
            similarity = torch.einsum("bd,bnd->bn", candidate_key, keys)
            any_occupied = occupied.any(dim=-1, keepdim=True)
            safe = torch.where(
                any_occupied, similarity.masked_fill(~occupied, -torch.inf),
                torch.zeros_like(similarity),
            )
            merge = torch.where(
                any_occupied, torch.softmax(safe / MERGE_TEMPERATURE, dim=-1),
                torch.zeros_like(safe),
            )
            max_similarity = torch.where(
                any_occupied.squeeze(-1), safe.max(dim=-1).values, minus_one
            )
            merge_strength = torch.sigmoid(
                MERGE_SHARPNESS * (max_similarity - MERGE_THRESHOLD)
            )[:, None]
            allocation = torch.softmax(
                (5.0 * (1.0 - occupancy) + slot_bias) / 0.10, dim=-1
            )
            write = write_strengths[:, t, None] * (
                merge_strength * merge + (1.0 - merge_strength) * allocation
            )
            keep = (1.0 - write)[:, :, None]
            keys = F.normalize(keep * keys + write[:, :, None] * candidate_key[:, None], dim=-1)
            values = keep * values + write[:, :, None] * candidate_values[:, t, None]
            occupancy = occupancy + write * (1.0 - occupancy)

            score = torch.einsum("bd,bnd->bn", queries[:, t], keys) / temperature
            attention = torch.softmax(score + 0.25 * torch.log(occupancy + 1e-6), dim=-1)
            readable = (
                values / occupancy.clamp_min(OCCUPANCY_FLOOR)[:, :, None]
                if self.normalized_values else values
            )
            retrieved_steps.append(torch.einsum("bn,bnv->bv", attention, readable))
            if t == 0:
                entropies = []
            entropies.append(-(attention * attention.clamp_min(1e-12).log()).sum(-1))

        retrieved = torch.stack(retrieved_steps, dim=1)
        confidence = 1.0 - torch.stack(entropies, dim=1) / math.log(self.num_slots)
        read_strength = torch.sigmoid(
            self.read_gate(torch.cat([context, retrieved, confidence[..., None]], dim=-1))
        )
        hidden = context
        if self.use_phl:
            hidden = hidden + self.phl_readout(torch.stack(phl_steps, dim=1))
        memory = read_strength * retrieved
        if disable_retrieval:
            memory = torch.zeros_like(memory)
        if self.direct_readout:
            logits = self.output(torch.cat([self.output_norm(hidden), memory], dim=-1))
        else:
            logits = self.output(self.output_norm(hidden + self.memory_projection(memory)))
        if self.copy_readout:
            vocabulary = self.value_projection(self.token_embedding.weight)
            logits = logits + self.copy_scale * memory @ vocabulary.T
        return logits


def active_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["PHLDAMv2", "active_parameter_count", "VOCAB_SIZE"]
