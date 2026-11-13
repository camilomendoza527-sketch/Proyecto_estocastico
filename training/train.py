"""Entrenamiento offline de los 27 modelos HMM (training/train.py).

Ejecución:
    python -m training.train --mode all
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np

from core.config import PROJECT_ROOT, get_config
from core.FeatureExtractor import FeatureExtractor2D, FeatureExtractor3D
from core.HMMEngine import HMM, init_segmental_bakis_hmm
from training.augment import augment_strokes, augment_strokes_3d, load_dataset


def _train_letter(
    letter: str,
    samples: list[list[np.ndarray]],
    extractor: FeatureExtractor2D | FeatureExtractor3D,
    augment_fn: Callable[[list[np.ndarray], np.random.Generator], list[np.ndarray]],
    n_states: int,
    augment_factor: int = 1,
    max_iter: int = 30,
    tol: float = 1e-3,
    seed: int = 123,
) -> tuple[HMM, list[float]]:
    """Núcleo unificado de entrenamiento HMM (2D o 3D) con Baum-Welch y descriptores de forma."""
    rng = np.random.default_rng(seed)
    sequences: list[list[int]] = []
    shapes: list[np.ndarray] = []

    for strokes in samples:
        res = extractor.extract(strokes, realtime=False)
        if len(res.symbols) >= 2:
            sequences.append(res.symbols.tolist())
            shapes.append(res.shape)
        for _ in range(augment_factor):
            aug = augment_fn(strokes, rng)
            res_aug = extractor.extract(aug, realtime=False)
            if len(res_aug.symbols) >= 2:
                sequences.append(res_aug.symbols.tolist())
                shapes.append(res_aug.shape)

    if not sequences:
        raise ValueError(f"No hay secuencias válidas para entrenar la letra {letter!r}")

    hmm = init_segmental_bakis_hmm(letter, n_states, extractor.n_symbols, sequences)
    history = hmm.fit(sequences, max_iter=max_iter, tol=tol, pseudo_count=1e-2, shapes=shapes)
    return hmm, history


def train_letter_2d(
    letter: str,
    samples: list[list[np.ndarray]],
    extractor: FeatureExtractor2D,
    n_states: int,
    augment_factor: int = 1,
    max_iter: int = 30,
    tol: float = 1e-3,
    seed: int = 123,
) -> tuple[HMM, list[float]]:
    return _train_letter(
        letter, samples, extractor, augment_strokes, n_states, augment_factor, max_iter, tol, seed
    )


def train_letter_3d(
    letter: str,
    samples: list[list[np.ndarray]],
    extractor: FeatureExtractor3D,
    n_states: int,
    augment_factor: int = 1,
    max_iter: int = 30,
    tol: float = 1e-3,
    seed: int = 321,
) -> tuple[HMM, list[float]]:
    return _train_letter(
        letter, samples, extractor, augment_strokes_3d, n_states, augment_factor, max_iter, tol, seed
    )


def _train_all(
    mode: str,
    K: int | None = None,
    models_dir: Path | None = None,
    max_samples: int | None = None,
    augment_factor: int = 1,
    max_iter: int = 30,
    verbose: bool = True,
) -> dict[str, HMM]:
    cfg = get_config()
    is_3d = mode == "3d"
    extractor = FeatureExtractor3D(K=K) if is_3d else FeatureExtractor2D(K=K)
    aug_fn = augment_strokes_3d if is_3d else augment_strokes
    seed_base = 300 if is_3d else 100
    out_dir = models_dir if models_dir is not None else PROJECT_ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    trained: dict[str, HMM] = {}
    for idx, letter in enumerate(cfg.letters):
        samples = load_dataset(letter, mode=mode, max_samples=max_samples)
        n_states = cfg.states[letter]
        aug = 0 if len(samples) >= 80 else augment_factor
        hmm, history = _train_letter(
            letter, samples, extractor, aug_fn, n_states, aug, max_iter, 1e-3, seed_base + idx
        )
        model_path = out_dir / f"{mode}_{cfg.file_name(letter)}.json"
        hmm.save_json(model_path)
        trained[letter] = hmm
        if verbose and history:
            print(
                f"[{mode.upper()} K={extractor.K}] '{letter}' (N={n_states:2d}) | "
                f"muestras={len(samples):3d} | iter={len(history):2d} | "
                f"logL: {history[0]:9.1f} -> {history[-1]:9.1f} | {model_path.name}"
            )
    return trained


def train_all_2d(
    K: int | None = None,
    models_dir: Path | None = None,
    max_samples: int | None = None,
    augment_factor: int = 1,
    max_iter: int = 30,
    verbose: bool = True,
) -> dict[str, HMM]:
    return _train_all("2d", K, models_dir, max_samples, augment_factor, max_iter, verbose)


def train_all_3d(
    K: int | None = None,
    models_dir: Path | None = None,
    max_samples: int | None = None,
    augment_factor: int = 1,
    max_iter: int = 30,
    verbose: bool = True,
) -> dict[str, HMM]:
    return _train_all("3d", K, models_dir, max_samples, augment_factor, max_iter, verbose)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena los 27 HMM (2D/3D) y exporta a models/*.json")
    parser.add_argument("--mode", choices=("2d", "3d", "all"), default="all")
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--augment", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=25)
    args = parser.parse_args()

    if args.mode in ("2d", "all"):
        train_all_2d(K=args.K, augment_factor=args.augment, max_iter=args.max_iter)
    if args.mode in ("3d", "all"):
        train_all_3d(K=args.K, augment_factor=args.augment, max_iter=args.max_iter)


if __name__ == "__main__":
    main()
