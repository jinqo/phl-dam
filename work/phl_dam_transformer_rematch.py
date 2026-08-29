"""Parameter-matched causal Transformer rematch for PHL-DAM.

Uses the exact synthetic streams and predictive objective from Stage B/C.
Temporal leases and promotion are absent. The Transformer has no controller,
so the DAM-specific position-agnostic write-budget regularizer is inapplicable.
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


def sinusoidal_positions(length: int, width: int) -> Tensor:
    positions = torch.arange(length, dtype=torch.float32)[:, None]
    frequencies = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    encoding = torch.zeros(length, width)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    return encoding


class CausalTransformer(nn.Module):
    """One-block parameter-matched decoder-only Transformer."""

    def __init__(
        self,
        d_model: int = 48,
        num_heads: int = 4,
        ffn_width: int = 195,
        max_length: int = SEQUENCE_LENGTH,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_width = ffn_width
        self.max_length = max_length
        self.token_embedding = nn.Embedding(VOCAB_SIZE, d_model)
        self.norm_attention = nn.RMSNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=0.0, batch_first=True
        )
        self.norm_ffn = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_width),
            nn.GELU(),
            nn.Linear(ffn_width, d_model),
        )
        self.final_norm = nn.RMSNorm(d_model)
        self.output = nn.Linear(d_model, VOCAB_SIZE)
        self.register_buffer(
            "position_encoding",
            sinusoidal_positions(max_length, d_model),
            persistent=True,
        )
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_length, max_length, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def forward(self, tokens: Tensor, disable_attention: bool = False) -> Tensor:
        length = tokens.shape[1]
        if length > self.max_length:
            raise ValueError(f"sequence length {length} exceeds {self.max_length}")
        hidden = self.token_embedding(tokens) + self.position_encoding[None, :length]
        if not disable_attention:
            normalized = self.norm_attention(hidden)
            attended, _ = self.attention(
                normalized,
                normalized,
                normalized,
                attn_mask=self.causal_mask[:length, :length],
                need_weights=False,
            )
            hidden = hidden + attended
        hidden = hidden + self.ffn(self.norm_ffn(hidden))
        return self.output(self.final_norm(hidden))


def active_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def kv_cache_floats(model: CausalTransformer, sequence_length: int) -> int:
    return 2 * sequence_length * model.d_model


def train(
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[CausalTransformer, list[dict[str, float]]]:
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed + 10_000)
    model = CausalTransformer().to(device)
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
                f"all_ce={item['all_token_ce']:.4f} "
                f"recall_ce={item['recall_ce']:.4f} "
                f"elapsed={item['elapsed_seconds']:.1f}s",
                flush=True,
            )
    return model, history


@torch.no_grad()
def evaluate(
    model: CausalTransformer,
    seed: int,
    episodes: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    totals = {
        "queries": 0,
        "correct": 0,
        "attention_disabled_correct": 0,
        "all_ce": 0.0,
        "recall_ce": 0.0,
        "batches": 0,
    }
    bin_counts = {"29-63": 0, "64-95": 0, "96-169": 0}
    bin_correct = {name: 0 for name in bin_counts}
    episodes_seen = 0
    model.eval()
    while episodes_seen < episodes:
        current_batch = min(batch_size, episodes - episodes_seen)
        episodes_seen += current_batch
        batch = make_batch(generator, current_batch, device)
        logits = model(batch.tokens)
        disabled_logits = model(batch.tokens, disable_attention=True)
        _, all_ce, recall_ce = common_objective(logits, batch)
        totals["all_ce"] += all_ce.item()
        totals["recall_ce"] += recall_ce.item()
        totals["batches"] += 1

        prediction_positions = batch.query_key_positions
        targets = _gather_positions(batch.tokens, prediction_positions + 1)
        predictions = _gather_positions(logits.argmax(dim=-1), prediction_positions)
        disabled_predictions = _gather_positions(
            disabled_logits.argmax(dim=-1), prediction_positions
        )
        is_correct = predictions.eq(targets)
        totals["queries"] += targets.numel()
        totals["correct"] += is_correct.sum().item()
        totals["attention_disabled_correct"] += disabled_predictions.eq(targets).sum().item()

        for name, low, high in (
            ("29-63", 29, 63),
            ("64-95", 64, 95),
            ("96-169", 96, 169),
        ):
            mask = (batch.delays >= low) & (batch.delays <= high)
            bin_counts[name] += mask.sum().item()
            bin_correct[name] += (is_correct & mask).sum().item()

    queries = int(totals["queries"])
    batches = int(totals["batches"])
    return {
        "episodes": episodes,
        "queries": queries,
        "recall_accuracy": totals["correct"] / queries,
        "attention_disabled_accuracy": totals["attention_disabled_correct"] / queries,
        "all_token_ce": totals["all_ce"] / batches,
        "recall_token_ce": totals["recall_ce"] / batches,
        "distance_accuracy": {
            name: bin_correct[name] / bin_counts[name] for name in bin_counts
        },
        "distance_counts": bin_counts,
        "finite": all(parameter.isfinite().all().item() for parameter in model.parameters()),
    }


def run(
    seed: int,
    steps: int,
    batch_size: int,
    eval_episodes: int,
    learning_rate: float,
    device: torch.device,
) -> dict[str, object]:
    model, history = train(seed, steps, batch_size, learning_rate, device)
    metrics = evaluate(
        model,
        seed=seed + 20_000,
        episodes=eval_episodes,
        batch_size=batch_size,
        device=device,
    )
    return {
        "experiment": "Clean PHL-DAM vs Transformer rematch",
        "model": "causal_transformer",
        "configuration": {
            "seed": seed,
            "d_model": model.d_model,
            "layers": 1,
            "heads": model.num_heads,
            "ffn_width": model.ffn_width,
            "sequence_length": SEQUENCE_LENGTH,
            "fixed_sinusoidal_positions": True,
            "dropout": 0.0,
            "causal": True,
            "active_parameters": active_parameter_count(model),
            "kv_cache_floats_at_sequence_length": kv_cache_floats(
                model, SEQUENCE_LENGTH
            ),
            "primary_objective": "all-token next-token CE + marked recall-token CE",
            "architecture_specific_regularizer": None,
            "training_steps": steps,
            "batch_size": batch_size,
            "eval_episodes": eval_episodes,
            "learning_rate": learning_rate,
            "lease_state_present": False,
            "promotion": False,
        },
        "metrics": metrics,
        "training_history": history,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--output", type=Path, required=True)
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
    )
    rendered = json.dumps(summary, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
