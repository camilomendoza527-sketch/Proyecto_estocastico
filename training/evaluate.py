"""Evaluación offline del clasificador HMM 2D (training/evaluate.py).

Cumple los dos requisitos de evaluación aprobados:
1. Comparar K = 8 vs K = 16 (valores en config.features.K_compare).
2. Medir accuracy (Top-1 y Top-3) sobre prefijos del trazo en tiempo real
   (30 %, 50 %, 70 % y 100 % definidos en config.evaluation.prefix_fractions)
   usando FeatureExtractor2D.extract_prefix(..., realtime=True).

Ejecución:
    python -m training.evaluate
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.config import get_config
from core.FeatureExtractor import FeatureExtractor2D, FeatureExtractor3D
from core.HMMEngine import HMMClassifier
from training.augment import (
    generate_seed_dataset_2d,
    generate_seed_dataset_3d,
    load_raw_samples,
)
from training.train import train_letter_2d, train_letter_3d


@dataclass
class EvaluationReport:
    """Resultados de evaluar un valor de K en todas las fracciones de prefijo."""

    K: int
    top1_by_prefix: dict[float, float]
    top3_by_prefix: dict[float, float]
    per_letter_acc_full: dict[str, float]
    confusion_full: np.ndarray


def evaluate_k_2d(
    K: int,
    raw_dir: Path | None = None,
    train_ratio: float = 0.75,
    augment_factor: int = 1,
    max_iter: int = 25,
) -> EvaluationReport:
    """Divide las muestras por letra en train/test, entrena con K y evalúa prefijos."""
    cfg = get_config()
    generate_seed_dataset_2d(samples_per_letter=20, base_dir=raw_dir, only_if_empty=True)

    extractor = FeatureExtractor2D(K=K)
    classifier = HMMClassifier()

    test_set: dict[str, list[list[np.ndarray]]] = {}
    for idx, letter in enumerate(cfg.letters):
        samples = load_raw_samples(letter, mode="2d", base_dir=raw_dir)
        n_train = max(1, int(round(len(samples) * train_ratio)))
        train_samples = samples[:n_train]
        test_samples = samples[n_train:] if n_train < len(samples) else samples[-2:]
        test_set[letter] = test_samples

        hmm, _ = train_letter_2d(
            letter=letter,
            samples=train_samples,
            extractor=extractor,
            n_states=cfg.states[letter],
            augment_factor=augment_factor,
            max_iter=max_iter,
            seed=500 + idx,
        )
        classifier.add_model(hmm)

    fractions = cfg.prefix_fractions
    top1_hits = {f: 0 for f in fractions}
    top3_hits = {f: 0 for f in fractions}
    totals = {f: 0 for f in fractions}

    n_classes = len(cfg.letters)
    letter_to_idx = {ch: i for i, ch in enumerate(cfg.letters)}
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    per_letter_hits = {ch: 0 for ch in cfg.letters}
    per_letter_tot = {ch: 0 for ch in cfg.letters}

    for letter, samples in test_set.items():
        for strokes in samples:
            for frac in fractions:
                res = extractor.extract_prefix(strokes, frac)
                if len(res.symbols) == 0:
                    continue
                ranking = classifier.classify(res.symbols)
                pred_top1 = ranking[0][0]
                pred_top3 = [r[0] for r in ranking[:3]]

                totals[frac] += 1
                if pred_top1 == letter:
                    top1_hits[frac] += 1
                if letter in pred_top3:
                    top3_hits[frac] += 1

                if np.isclose(frac, 1.0):
                    confusion[letter_to_idx[letter], letter_to_idx[pred_top1]] += 1
                    per_letter_tot[letter] += 1
                    if pred_top1 == letter:
                        per_letter_hits[letter] += 1

    top1_by_prefix = {f: top1_hits[f] / max(1, totals[f]) for f in fractions}
    top3_by_prefix = {f: top3_hits[f] / max(1, totals[f]) for f in fractions}
    per_letter_acc = {
        ch: per_letter_hits[ch] / max(1, per_letter_tot[ch]) for ch in cfg.letters
    }
    return EvaluationReport(
        K=K,
        top1_by_prefix=top1_by_prefix,
        top3_by_prefix=top3_by_prefix,
        per_letter_acc_full=per_letter_acc,
        confusion_full=confusion,
    )


def evaluate_k_3d(
    K: int,
    raw_dir: Path | None = None,
    train_ratio: float = 0.75,
    augment_factor: int = 1,
    max_iter: int = 25,
) -> EvaluationReport:
    """Divide las muestras 3D por letra en train/test, entrena con K y evalúa prefijos geodésicos."""
    cfg = get_config()
    generate_seed_dataset_3d(samples_per_letter=20, base_dir=raw_dir, only_if_empty=True)

    extractor = FeatureExtractor3D(K=K)
    classifier = HMMClassifier()

    test_set: dict[str, list[list[np.ndarray]]] = {}
    for idx, letter in enumerate(cfg.letters):
        samples = load_raw_samples(letter, mode="3d", base_dir=raw_dir)
        n_train = max(1, int(round(len(samples) * train_ratio)))
        train_samples = samples[:n_train]
        test_samples = samples[n_train:] if n_train < len(samples) else samples[-2:]
        test_set[letter] = test_samples

        hmm, _ = train_letter_3d(
            letter=letter,
            samples=train_samples,
            extractor=extractor,
            n_states=cfg.states[letter],
            augment_factor=augment_factor,
            max_iter=max_iter,
            seed=700 + idx,
        )
        classifier.add_model(hmm)

    fractions = cfg.prefix_fractions
    top1_hits = {f: 0 for f in fractions}
    top3_hits = {f: 0 for f in fractions}
    totals = {f: 0 for f in fractions}

    n_classes = len(cfg.letters)
    letter_to_idx = {ch: i for i, ch in enumerate(cfg.letters)}
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    per_letter_hits = {ch: 0 for ch in cfg.letters}
    per_letter_tot = {ch: 0 for ch in cfg.letters}

    for letter, samples in test_set.items():
        for strokes in samples:
            for frac in fractions:
                res = extractor.extract_prefix(strokes, frac)
                if len(res.symbols) == 0:
                    continue
                ranking = classifier.classify(res.symbols)
                pred_top1 = ranking[0][0]
                pred_top3 = [r[0] for r in ranking[:3]]

                totals[frac] += 1
                if pred_top1 == letter:
                    top1_hits[frac] += 1
                if letter in pred_top3:
                    top3_hits[frac] += 1

                if np.isclose(frac, 1.0):
                    confusion[letter_to_idx[letter], letter_to_idx[pred_top1]] += 1
                    per_letter_tot[letter] += 1
                    if pred_top1 == letter:
                        per_letter_hits[letter] += 1

    top1_by_prefix = {f: top1_hits[f] / max(1, totals[f]) for f in fractions}
    top3_by_prefix = {f: top3_hits[f] / max(1, totals[f]) for f in fractions}
    per_letter_acc = {
        ch: per_letter_hits[ch] / max(1, per_letter_tot[ch]) for ch in cfg.letters
    }
    return EvaluationReport(
        K=K,
        top1_by_prefix=top1_by_prefix,
        top3_by_prefix=top3_by_prefix,
        per_letter_acc_full=per_letter_acc,
        confusion_full=confusion,
    )


def compare_k_configs(
    raw_dir: Path | None = None, mode: str = "2d", verbose: bool = True
) -> dict[int, EvaluationReport]:
    """Compara todos los valores de K en config.features.K_compare (8 vs 16)."""
    cfg = get_config()
    reports: dict[int, EvaluationReport] = {}
    eval_fn = evaluate_k_3d if mode == "3d" else evaluate_k_2d
    for K in cfg.features.K_compare:
        rep = eval_fn(K=K, raw_dir=raw_dir)
        reports[K] = rep

    if verbose:
        print(
            f"\n=== Comparación de Cuantización Angular ({mode.upper()}: K=8 vs K=16) en Tiempo Real ==="
        )
        header = "Prefijo  | " + " | ".join(
            f"K={K:2d} Top-1 / Top-3" for K in cfg.features.K_compare
        )
        print(header)
        print("-" * len(header))
        for frac in cfg.prefix_fractions:
            pct = int(round(frac * 100))
            cols = [
                f"  {reports[K].top1_by_prefix[frac]*100:5.1f}% / {reports[K].top3_by_prefix[frac]*100:5.1f}%"
                for K in cfg.features.K_compare
            ]
            print(f" {pct:3d}%    | " + " | ".join(cols))
    return reports


if __name__ == "__main__":
    compare_k_configs(mode="2d", verbose=True)
    compare_k_configs(mode="3d", verbose=True)

