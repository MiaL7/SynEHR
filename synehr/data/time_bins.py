from __future__ import annotations
import math
import torch

FINE_BINS = 16
FINE_BIN_LABELS = [
    '0', '1', '2-3', '4-5', '6-7', '8-14', '15-21', '22-30', '31-45', '46-60',
    '61-90', '91-120', '121-180', '181-270', '271-365', '>365'
]
BIN_MIDPOINTS = [
    0, 1, 2.5, 4.5, 6.5, 11, 18, 26, 38, 53, 75.5, 105.5, 150.5, 225.5, 318,
    500
]
SHORT_BINS = list(range(0, 5))
MEDIUM_BINS = list(range(5, 11))
LONG_BINS = list(range(11, 16))
MAX_PREFIX_LEN = 20


def days_to_fine_bin(y_days: float) -> int:
    d = float(y_days)

    if d < 1:
        return 0

    if d < 2:
        return 1

    if d < 4:
        return 2

    if d < 6:
        return 3

    if d < 8:
        return 4

    if d < 15:
        return 5

    if d < 22:
        return 6

    if d < 31:
        return 7

    if d < 46:
        return 8

    if d < 61:
        return 9

    if d < 91:
        return 10

    if d < 121:
        return 11

    if d < 181:
        return 12

    if d < 271:
        return 13

    if d < 366:
        return 14

    return 15


def fine_bin_to_regime3(fine_bin: int) -> int:
    if fine_bin in SHORT_BINS:
        return 0

    if fine_bin in MEDIUM_BINS:
        return 1

    return 2


def fine_bin_to_label(fine_bin: int) -> str:
    return FINE_BIN_LABELS[int(fine_bin)]


def days_to_fine_label(y_days: float) -> str:
    return fine_bin_to_label(days_to_fine_bin(y_days))


def compute_bin_representatives(all_y_days: list[float],
                                all_fine_bins: list[int]) -> torch.Tensor:
    import statistics
    bin_to_days: dict[int, list[float]] = {b: [] for b in range(FINE_BINS)}

    for d, b in zip(all_y_days, all_fine_bins):
        bin_to_days[b].append(d)

    reps = []

    for b in range(FINE_BINS):
        vals = bin_to_days[b]
        reps.append(
            statistics.median(vals) if vals else float(BIN_MIDPOINTS[b]))

    return torch.tensor(reps, dtype=torch.float32)


def aggregate_fine_to_q3(q_fine: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        q_fine[..., SHORT_BINS].sum(-1), q_fine[..., MEDIUM_BINS].sum(-1),
        q_fine[..., LONG_BINS].sum(-1)
    ],
        dim=-1)


def compute_conf_from_q3(q3: torch.Tensor) -> torch.Tensor:
    eps = 1e-08
    entropy = -(q3 * torch.log(q3 + eps)).sum(dim=-1)
    conf = 1.0 - entropy / math.log(3)

    return conf.clamp(0.0, 1.0)
