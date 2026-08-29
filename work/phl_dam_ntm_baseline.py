"""Parameter-matched NTM/DNC-style memory baseline for PHL-DAM.

This is the comparison the PHL-DAM line most needs and has never run. PHL-DAM
beats a Transformer and two SSMs on associative recall, but those baselines have
no addressable memory at all - losing an addressing task is close to their
expected behaviour. The question that actually discriminates is:

    does PHL-DAM beat a conventional differentiable memory at matched size?

Because PHL-DAM's working core - a fixed bank of slots, content-addressed by
similarity, written through learned gates - is the Neural Turing Machine /
Differentiable Neural Computer design. If a straightforward DNC-style memory
matches it, the contribution is the task framing rather than the architecture.

Implemented here, following Graves et al. (2014 NTM, 2016 DNC):

* content-based addressing by cosine similarity with a learned key strength
* usage-based allocation: writes prefer the least-used slot, as in the DNC
* an interpolation gate blending content addressing with allocation
* erase/add vector writes rather than convex blending
* a free gate that releases memory after reads

Temporal linkage is omitted deliberately: it serves sequential-order recall,
which this benchmark does not test, and including an unused mechanism would
only spend parameters. This is meant to be a strong baseline, not a strawman.

Streams, objective, optimiser and budget are imported unchanged from Stage B so
the numbers sit directly beside the PHL-DAM, Transformer and SSM results.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from phl_dam_stage_b import (
    SEQUENCE_LENGTH,
    VOCAB_SIZE,
    _gather_positions,
    common_objective,
    make_batch,
    seed_everything,
)

torch.set_num_threads(1)


class DNCMemory(nn.Module):
    """A bank of slots with content addressing and usage-based allocation."""

    def __init__(self, slots: int, width: int, controller_width: int) -> None:
        super().__init__()
        self.slots = slots
        self.width = width
        # Write interface: key, strength, erase, add, allocation gate, write gate.
        self.write_key = nn.Linear(controller_width, width)
        self.write_strength = nn.Linear(controller_width, 1)
        self.erase_vector = nn.Linear(controller_width, width)
        self.add_vector = nn.Linear(controller_width, width)
        self.allocation_gate = nn.Linear(controller_width, 1)
        self.write_gate = nn.Linear(controller_width, 1)
        # Read interface: key, strength, free gate.
        self.read_key = nn.Linear(controller_width, width)
        self.read_strength = nn.Linear(controller_width, 1)
        self.free_gate = nn.Linear(controller_width, 1)
        nn.init.constant_(self.write_gate.bias, -1.0)

    @staticmethod
    def allocation_weighting(usage: Tensor) -> Tensor:
        """DNC allocation, differentiable in usage (Graves et al. 2016, eq. 1).

        a[phi_j] = (1 - u[phi_j]) * prod_{i<j} u[phi_i], where phi sorts slots
        by ascending usage. Taking a hard one-hot on the least-used slot
        instead - as a first cut of this file did - is non-differentiable and
        silently starves the usage path, so the free gate never trains.
        The sort *indices* are treated as constants; the usage *values* carry
        gradient, which is what makes the published formula trainable.
        """
        sorted_usage, indices = usage.sort(dim=-1)
        ones = torch.ones_like(sorted_usage[:, :1])
        exclusive = torch.cat([ones, sorted_usage[:, :-1]], dim=-1).cumprod(dim=-1)
        weighting = (1.0 - sorted_usage) * exclusive
        return torch.zeros_like(usage).scatter(1, indices, weighting)

    def content_weighting(self, memory: Tensor, key: Tensor, strength: Tensor) -> Tensor:
        similarity = F.cosine_similarity(
            memory, key.unsqueeze(1).expand_as(memory), dim=-1
        )
        return torch.softmax(similarity * strength, dim=-1)

    def forward(
        self, controller: Tensor, memory: Tensor, usage: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # --- write ---------------------------------------------------------
        key = self.write_key(controller)
        strength = F.softplus(self.write_strength(controller)) + 1.0
        content = self.content_weighting(memory, key, strength)

        allocation = self.allocation_weighting(usage)

        gate_allocation = torch.sigmoid(self.allocation_gate(controller))
        gate_write = torch.sigmoid(self.write_gate(controller))
        write_weighting = gate_write * (
            gate_allocation * allocation + (1.0 - gate_allocation) * content
        )

        erase = torch.sigmoid(self.erase_vector(controller))
        add = self.add_vector(controller)
        weighting = write_weighting.unsqueeze(-1)
        memory = memory * (1.0 - weighting * erase.unsqueeze(1))
        memory = memory + weighting * add.unsqueeze(1)

        # --- read ----------------------------------------------------------
        read_key = self.read_key(controller)
        read_strength = F.softplus(self.read_strength(controller)) + 1.0
        read_weighting = self.content_weighting(memory, read_key, read_strength)
        read_vector = torch.einsum("bn,bnw->bw", read_weighting, memory)

        # --- usage ---------------------------------------------------------
        free = torch.sigmoid(self.free_gate(controller))
        usage = (usage + write_weighting - usage * write_weighting) * (
            1.0 - free * read_weighting
        )
        return memory, usage, read_vector


class NTMBaseline(nn.Module):
    """Feedforward controller plus a DNC-style external memory."""

    def __init__(
        self,
        d_model: int = 48,
        slots: int = 8,
        width: int = 48,          # parameter-matched: 32,836 params
        controller_width: int = 72,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.slots = slots
        self.width = width
        self.controller_width = controller_width
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        # Same three-token causal window the PHL-DAM controller sees.
        self.controller = nn.Sequential(
            nn.Linear(3 * d_model, controller_width), nn.Tanh()
        )
        self.memory = DNCMemory(slots, width, controller_width)
        self.output_norm = nn.RMSNorm(controller_width)
        self.output = nn.Linear(controller_width + width, VOCAB_SIZE)

    def windows(self, tokens: Tensor) -> Tensor:
        embedded = self.token_embedding(tokens)
        padding = torch.zeros(
            embedded.shape[0], 2, embedded.shape[2], device=embedded.device
        )
        padded = torch.cat([padding, embedded], dim=1)
        return torch.cat([padded[:, 0:-2], padded[:, 1:-1], padded[:, 2:]], dim=-1)

    def forward(self, tokens: Tensor, disable_memory: bool = False) -> Tensor:
        controller = self.controller(self.windows(tokens))
        batch, length, _ = controller.shape
        memory = torch.zeros(batch, self.slots, self.width, device=tokens.device)
        usage = torch.zeros(batch, self.slots, device=tokens.device)
        outputs = []
        for step in range(length):
            hidden = controller[:, step]
            if disable_memory:
                read_vector = torch.zeros(batch, self.width, device=tokens.device)
            else:
                memory, usage, read_vector = self.memory(hidden, memory, usage)
            outputs.append(
                self.output(torch.cat([self.output_norm(hidden), read_vector], dim=-1))
            )
        return torch.stack(outputs, dim=1)


def active_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def recurrent_state_floats(model: NTMBaseline) -> int:
    """Memory bank plus usage vector - constant in sequence length."""
    return model.slots * model.width + model.slots


def train(seed, steps, batch_size, learning_rate, device):
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed + 10_000)
    model = NTMBaseline().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    started = time.perf_counter()
    model.train()
    for step in range(1, steps + 1):
        batch = make_batch(generator, batch_size, device)
        logits = model(batch.tokens)
        loss, all_ce, recall_ce = common_objective(logits, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == steps:
            item = {
                "step": step,
                "loss": loss.item(),
                "all_token_ce": all_ce.item(),
                "recall_ce": recall_ce.item(),
                "gradient_norm": float(norm),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(item)
            print(
                f"seed={seed} step={step:4d} loss={item['loss']:.4f} "
                f"recall_ce={item['recall_ce']:.4f} elapsed={item['elapsed_seconds']:.0f}s",
                flush=True,
            )
    return model, history


@torch.no_grad()
def evaluate(model, seed, episodes, batch_size, device):
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    correct = disabled = queries = 0
    all_ce = recall_ce = 0.0
    batches = 0
    bins = {"29-63": [0, 0], "64-95": [0, 0], "96-169": [0, 0]}
    seen = 0
    while seen < episodes:
        count = min(batch_size, episodes - seen)
        seen += count
        batch = make_batch(generator, count, device)
        logits = model(batch.tokens)
        ablated = model(batch.tokens, disable_memory=True)
        _, a, r = common_objective(logits, batch)
        all_ce += a.item()
        recall_ce += r.item()
        batches += 1
        predictions = _gather_positions(logits.argmax(dim=-1), batch.query_key_positions)
        ablated_predictions = _gather_positions(
            ablated.argmax(dim=-1), batch.query_key_positions
        )
        targets = _gather_positions(batch.tokens, batch.query_key_positions + 1)
        hit = predictions.eq(targets)
        correct += int(hit.sum())
        disabled += int(ablated_predictions.eq(targets).sum())
        queries += targets.numel()
        for name, low, high in (("29-63", 29, 63), ("64-95", 64, 95), ("96-169", 96, 169)):
            mask = (batch.delays >= low) & (batch.delays <= high)
            bins[name][0] += int((hit & mask).sum())
            bins[name][1] += int(mask.sum())
    return {
        "episodes": episodes,
        "queries": queries,
        "recall_accuracy": correct / queries,
        "memory_disabled_accuracy": disabled / queries,
        "all_token_ce": all_ce / batches,
        "recall_token_ce": recall_ce / batches,
        "distance_accuracy": {k: v[0] / v[1] for k, v in bins.items() if v[1]},
        "finite": all(p.isfinite().all().item() for p in model.parameters()),
    }


def run(seed, steps, batch_size, eval_episodes, learning_rate, device):
    model, history = train(seed, steps, batch_size, learning_rate, device)
    metrics = evaluate(model, seed + 20_000, eval_episodes, batch_size, device)
    return {
        "experiment": "PHL-DAM NTM/DNC baseline - conventional differentiable memory",
        "model": "ntm_dnc",
        "configuration": {
            "seed": seed,
            "slots": model.slots,
            "memory_width": model.width,
            "controller_width": model.controller_width,
            "d_model": model.d_model,
            "sequence_length": SEQUENCE_LENGTH,
            "training_steps": steps,
            "batch_size": batch_size,
            "eval_episodes": eval_episodes,
            "learning_rate": learning_rate,
            "optimizer": "AdamW(weight_decay=1e-4)",
            "objective": "all-token next-token CE + marked recall-token CE",
            "active_parameters": active_parameter_count(model),
            "recurrent_state_floats": recurrent_state_floats(model),
            "addressing": "content + usage-based allocation (DNC)",
            "temporal_linkage": False,
        },
        "metrics": metrics,
        "training_history": history,
        "finite": metrics["finite"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=2_000)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        args.seed, args.steps, args.batch_size, args.eval_episodes,
        args.learning_rate, torch.device("cpu"),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
