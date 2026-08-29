"""Parameter-matched diagonal SSM baseline for PHL-DAM.

The PHL-DAM line has compared against a causal Transformer but never against a
state-space model, which is the more natural comparison: an SSM is also a
constant-state recurrent sequence model, so it competes with PHL-DAM on the
axis PHL-DAM claims to win on (bounded inference state) rather than only on
recall.

This is an S4D/Mamba-style *diagonal* SSM: per-channel complex-free diagonal
recurrence with a learned log-spaced timescale, a gated position-wise mixer, and
a residual block. It is deliberately a strong, conventional implementation - the
point of a baseline is to be hard to beat, not to be a strawman.

Streams, objective, optimiser, budget and evaluation are imported unchanged from
Stage B, so the numbers are directly comparable to
``PHL_DAM_Transformer_Rematch_Report.md`` and the zero-write-budget PHL-DAM runs.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from phl_dam_stage_b import (
    SEQUENCE_LENGTH,
    VOCAB_SIZE,
    _gather_positions,
    common_objective,
    make_batch,
    seed_everything,
)

torch.set_num_threads(1)


class DiagonalSSM(nn.Module):
    """One diagonal state-space layer, applied as a causal scan.

    State update per channel c:  h_t = a_c * h_{t-1} + b_c * u_t
    Output:                      y_t = sum_c ...  (per-channel, then mixed)

    ``a_c = exp(-exp(log_rate_c))`` lies in [0, 1] for any parameter value, so
    the scan can never explode and needs no clipping. Both ends saturate in
    float32 and both are benign: below about log_rate -16 it rounds to 1.0, a
    pure integrator (marginally stable, and a useful accumulator channel);
    above about +19 it underflows to 0.0, a channel that simply forgets. The
    guarantee this buys is `0 <= decay <= 1` - never amplification - which is
    what makes the recurrence safe without gradient clipping, in contrast to
    PHL-DAM's slot recurrence. Rates are initialised log-spaced so the layer
    covers short and long horizons at once, the standard S4D initialisation.
    """

    def __init__(
        self, d_model: int, state_multiplier: int = 1, selective: bool = False
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.channels = d_model * state_multiplier
        self.selective = selective
        # Log-spaced decay rates: fast channels track local structure, slow
        # channels carry information across the full sequence.
        minimum, maximum = 1e-3, 1.0
        rates = torch.exp(
            torch.linspace(math.log(minimum), math.log(maximum), self.channels)
        )
        self.log_rate = nn.Parameter(torch.log(rates))
        self.input_projection = nn.Linear(d_model, self.channels, bias=False)
        self.skip = nn.Parameter(torch.ones(self.channels) * 0.5)
        self.output_projection = nn.Linear(self.channels, d_model, bias=False)
        if selective:
            # S6/Mamba-style selection: an input-dependent timescale lets the
            # channel decide per token whether to hold or overwrite its state.
            # This is the mechanism that makes SSMs competitive on associative
            # recall, which a fixed-decay S4D cannot do.
            self.delta_projection = nn.Linear(d_model, self.channels)
            nn.init.constant_(self.delta_projection.bias, -1.0)

    def forward(self, sequence: Tensor) -> Tensor:
        batch, length, _ = sequence.shape
        rate = torch.exp(self.log_rate)
        drive = self.input_projection(sequence)
        if self.selective:
            delta = torch.nn.functional.softplus(self.delta_projection(sequence))
            decay_sequence = torch.exp(-delta * rate)  # in [0, 1] per token
        else:
            decay_sequence = None
            decay = torch.exp(-rate)  # in [0, 1]
        state = torch.zeros(batch, self.channels, device=sequence.device)
        outputs = []
        for step in range(length):
            step_decay = (
                decay_sequence[:, step] if decay_sequence is not None else decay
            )
            state = step_decay * state + drive[:, step]
            outputs.append(state)
        scanned = torch.stack(outputs, dim=1)
        return self.output_projection(scanned + self.skip * drive)


class SSMBlock(nn.Module):
    """Pre-norm SSM block with a gated position-wise mixer."""

    def __init__(
        self, d_model: int, mixer_width: int, state_multiplier: int,
        selective: bool = False,
    ) -> None:
        super().__init__()
        self.norm_ssm = nn.RMSNorm(d_model)
        self.ssm = DiagonalSSM(d_model, state_multiplier, selective)
        self.norm_mixer = nn.RMSNorm(d_model)
        self.gate = nn.Linear(d_model, mixer_width)
        self.up = nn.Linear(d_model, mixer_width)
        self.down = nn.Linear(mixer_width, d_model)

    def forward(self, sequence: Tensor) -> Tensor:
        sequence = sequence + self.ssm(self.norm_ssm(sequence))
        normed = self.norm_mixer(sequence)
        gated = torch.nn.functional.silu(self.gate(normed)) * self.up(normed)
        return sequence + self.down(gated)


class CausalSSM(nn.Module):
    """Parameter-matched decoder-only diagonal SSM."""

    def __init__(
        self,
        d_model: int = 48,
        mixer_width: int = 130,  # parameter-matched: 33,139 params
        state_multiplier: int = 2,
        blocks: int = 1,
        selective: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.mixer_width = mixer_width
        self.state_multiplier = state_multiplier
        self.blocks = blocks
        self.selective = selective
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.layers = nn.ModuleList(
            [
                SSMBlock(d_model, mixer_width, state_multiplier, selective)
                for _ in range(blocks)
            ]
        )
        self.final_norm = nn.RMSNorm(d_model)
        self.output = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, tokens: Tensor, disable_recurrence: bool = False) -> Tensor:
        hidden = self.token_embedding(tokens)
        for layer in self.layers:
            if disable_recurrence:
                # Ablation: keep the mixer, remove the temporal path entirely.
                normed = layer.norm_mixer(hidden)
                gated = torch.nn.functional.silu(layer.gate(normed)) * layer.up(normed)
                hidden = hidden + layer.down(gated)
            else:
                hidden = layer(hidden)
        return self.output(self.final_norm(hidden))


def active_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def recurrent_state_floats(model: CausalSSM) -> int:
    """Inference state: one vector per channel per block, independent of length."""
    return model.blocks * model.d_model * model.state_multiplier


def build_model(selective: bool) -> CausalSSM:
    """Parameter-matched in both variants: 33,139 fixed / 33,025 selective."""
    width = 97 if selective else 130
    return CausalSSM(mixer_width=width, selective=selective)


def train(
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    selective: bool = False,
) -> tuple[CausalSSM, list[dict[str, float]]]:
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed + 10_000)
    model = build_model(selective).to(device)
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
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == steps:
            item = {
                "step": step,
                "loss": loss.item(),
                "all_token_ce": all_ce.item(),
                "recall_ce": recall_ce.item(),
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(item)
            print(
                f"seed={seed} step={step:4d} loss={item['loss']:.4f} "
                f"all_ce={item['all_token_ce']:.4f} recall_ce={item['recall_ce']:.4f} "
                f"elapsed={item['elapsed_seconds']:.0f}s",
                flush=True,
            )
    return model, history


@torch.no_grad()
def evaluate(
    model: CausalSSM,
    seed: int,
    episodes: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    correct = disabled_correct = queries = 0
    all_ce_sum = recall_ce_sum = 0.0
    batches = 0
    bins = {"29-63": [0, 0], "64-95": [0, 0], "96-169": [0, 0]}
    seen = 0
    while seen < episodes:
        count = min(batch_size, episodes - seen)
        seen += count
        batch = make_batch(generator, count, device)
        logits = model(batch.tokens)
        ablated = model(batch.tokens, disable_recurrence=True)
        _, all_ce, recall_ce = common_objective(logits, batch)
        all_ce_sum += all_ce.item()
        recall_ce_sum += recall_ce.item()
        batches += 1
        predictions = _gather_positions(logits.argmax(dim=-1), batch.query_key_positions)
        ablated_predictions = _gather_positions(
            ablated.argmax(dim=-1), batch.query_key_positions
        )
        targets = _gather_positions(batch.tokens, batch.query_key_positions + 1)
        hit = predictions.eq(targets)
        correct += int(hit.sum())
        disabled_correct += int(ablated_predictions.eq(targets).sum())
        queries += targets.numel()
        for name, low, high in (("29-63", 29, 63), ("64-95", 64, 95), ("96-169", 96, 169)):
            mask = (batch.delays >= low) & (batch.delays <= high)
            bins[name][0] += int((hit & mask).sum())
            bins[name][1] += int(mask.sum())
    return {
        "episodes": episodes,
        "queries": queries,
        "recall_accuracy": correct / queries,
        "recurrence_disabled_accuracy": disabled_correct / queries,
        "all_token_ce": all_ce_sum / batches,
        "recall_token_ce": recall_ce_sum / batches,
        "distance_accuracy": {k: v[0] / v[1] for k, v in bins.items() if v[1]},
        "distance_counts": {k: v[1] for k, v in bins.items()},
        "finite": all(p.isfinite().all().item() for p in model.parameters()),
    }


def run(
    seed: int,
    steps: int,
    batch_size: int,
    eval_episodes: int,
    learning_rate: float,
    device: torch.device,
    selective: bool = False,
) -> dict[str, object]:
    model, history = train(seed, steps, batch_size, learning_rate, device, selective)
    metrics = evaluate(model, seed + 20_000, eval_episodes, batch_size, device)
    return {
        "experiment": "PHL-DAM SSM baseline - parameter-matched diagonal state-space model",
        "model": "selective_ssm" if model.selective else "diagonal_ssm",
        "configuration": {
            "seed": seed,
            "d_model": model.d_model,
            "blocks": model.blocks,
            "mixer_width": model.mixer_width,
            "state_multiplier": model.state_multiplier,
            "selective": model.selective,
            "sequence_length": SEQUENCE_LENGTH,
            "vocabulary": VOCAB_SIZE,
            "training_steps": steps,
            "batch_size": batch_size,
            "eval_episodes": eval_episodes,
            "learning_rate": learning_rate,
            "optimizer": "AdamW(weight_decay=1e-4)",
            "objective": "all-token next-token CE + marked recall-token CE",
            "active_parameters": active_parameter_count(model),
            "recurrent_state_floats": recurrent_state_floats(model),
            "lease_state_present": False,
            "promotion": False,
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
    parser.add_argument("--selective", action="store_true",
                        help="S6/Mamba-style input-dependent timescale")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes,
        learning_rate=args.learning_rate,
        device=torch.device("cpu"),
        selective=args.selective,
    )
    rendered = json.dumps(summary, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
