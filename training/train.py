"""Entrenamiento offline de los 27 modelos HMM (training/train.py).

- Lee el alfabeto de 27 letras y la tabla de estados N por letra desde
  config/config.toml (a través de core/config.py).
- Carga los trazos crudos de data/raw/<mode>/<alias>/*.json; si una carpeta
  está vacía y seed_if_empty=True, genera muestras base con training/augment.py.
- Inicializa cada modelo con alineación segmentaria uniforme Bakis y lo entrena
  mediante Baum-Welch multisequencia (HMM.fit).
- Exporta cada modelo a models/<mode>_<alias>.json (ej. models/2d_a.json,
  models/2d_enie.json).

Ejecución:
    python -m training.train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from core.config import PROJECT_ROOT, get_config
from core.FeatureExtractor import FeatureExtractor2D, FeatureExtractor3D
from core.HMMEngine import HMM
from core.topology import init_segmental_bakis_hmm
from training.augment import (
    augment_strokes,
    augment_strokes_3d,
    generate_seed_dataset_2d,
    generate_seed_dataset_3d,
    load_raw_samples,
)


def train_letter_2d(
    letter: str,
    samples: list[list[np.ndarray]],
    extractor: FeatureExtractor2D,
    n_states: int,
    augment_factor: int = 2,
    max_iter: int = 35,
    tol: float = 1e-3,
    seed: int = 123,
) -> tuple[HMM, list[float]]:
    """Entrena el modelo HMM 2D de una letra sobre sus trazos crudos + aumentos."""
    rng = np.random.default_rng(seed)
    sequences: list[list[int]] = []

    for strokes in samples:
        res = extractor.extract(strokes, realtime=False)
        if len(res.symbols) >= 2:
            sequences.append(res.symbols.tolist())
        for _ in range(augment_factor):
            aug = augment_strokes(strokes, rng)
            res_aug = extractor.extract(aug, realtime=False)
            if len(res_aug.symbols) >= 2:
                sequences.append(res_aug.symbols.tolist())

    if not sequences:
        raise ValueError(f"No hay secuencias válidas para entrenar la letra {letter!r}")

    hmm = init_segmental_bakis_hmm(
        letter=letter,
        n_states=n_states,
        n_symbols=extractor.n_symbols,
        sequences=sequences,
    )
    history = hmm.fit(sequences, max_iter=max_iter, tol=tol, pseudo_count=1e-2)
    return hmm, history


def train_letter_3d(
    letter: str,
    samples: list[list[np.ndarray]],
    extractor: FeatureExtractor3D,
    n_states: int,
    augment_factor: int = 2,
    max_iter: int = 35,
    tol: float = 1e-3,
    seed: int = 321,
) -> tuple[HMM, list[float]]:
    """Entrena el modelo HMM 3D de una letra sobre trazos esféricos + aumentos 3D."""
    rng = np.random.default_rng(seed)
    sequences: list[list[int]] = []

    for strokes in samples:
        res = extractor.extract(strokes, realtime=False)
        if len(res.symbols) >= 2:
            sequences.append(res.symbols.tolist())
        for _ in range(augment_factor):
            aug = augment_strokes_3d(strokes, rng)
            res_aug = extractor.extract(aug, realtime=False)
            if len(res_aug.symbols) >= 2:
                sequences.append(res_aug.symbols.tolist())

    if not sequences:
        raise ValueError(f"No hay secuencias 3D válidas para entrenar la letra {letter!r}")

    hmm = init_segmental_bakis_hmm(
        letter=letter,
        n_states=n_states,
        n_symbols=extractor.n_symbols,
        sequences=sequences,
    )
    history = hmm.fit(sequences, max_iter=max_iter, tol=tol, pseudo_count=1e-2)
    return hmm, history


def train_all_2d(
    K: int | None = None,
    raw_dir: Path | None = None,
    models_dir: Path | None = None,
    samples_if_empty: int = 24,
    augment_factor: int = 2,
    max_iter: int = 35,
    verbose: bool = True,
) -> dict[str, HMM]:
    """Entrena los 27 caracteres en 2D y guarda models/2d_<alias>.json."""
    cfg = get_config()
    extractor = FeatureExtractor2D(K=K)
    out_dir = models_dir if models_dir is not None else PROJECT_ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_seed_dataset_2d(
        samples_per_letter=samples_if_empty, base_dir=raw_dir, only_if_empty=True
    )

    trained: dict[str, HMM] = {}
    for idx, letter in enumerate(cfg.letters):
        samples = load_raw_samples(letter, mode="2d", base_dir=raw_dir)
        n_states = cfg.states[letter]
        hmm, history = train_letter_2d(
            letter=letter,
            samples=samples,
            extractor=extractor,
            n_states=n_states,
            augment_factor=augment_factor,
            max_iter=max_iter,
            seed=100 + idx,
        )
        alias = cfg.file_name(letter)
        model_path = out_dir / f"2d_{alias}.json"
        hmm.save_json(model_path)
        trained[letter] = hmm
        if verbose:
            ll_start = history[0] if history else float("nan")
            ll_end = history[-1] if history else float("nan")
            print(
                f"[2D K={extractor.K}] '{letter}' (N={n_states:2d}) | "
                f"muestras={len(samples):2d} | iter={len(history):2d} | "
                f"logL: {ll_start:8.1f} -> {ll_end:8.1f} | {model_path.name}"
            )
    return trained


def train_all_3d(
    K: int | None = None,
    raw_dir: Path | None = None,
    models_dir: Path | None = None,
    samples_if_empty: int = 24,
    augment_factor: int = 2,
    max_iter: int = 35,
    verbose: bool = True,
) -> dict[str, HMM]:
    """Entrena los 27 caracteres en 3D (esfera) y guarda models/3d_<alias>.json."""
    cfg = get_config()
    extractor = FeatureExtractor3D(K=K)
    out_dir = models_dir if models_dir is not None else PROJECT_ROOT / "models"
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_seed_dataset_3d(
        samples_per_letter=samples_if_empty, base_dir=raw_dir, only_if_empty=True
    )

    trained: dict[str, HMM] = {}
    for idx, letter in enumerate(cfg.letters):
        samples = load_raw_samples(letter, mode="3d", base_dir=raw_dir)
        n_states = cfg.states[letter]
        hmm, history = train_letter_3d(
            letter=letter,
            samples=samples,
            extractor=extractor,
            n_states=n_states,
            augment_factor=augment_factor,
            max_iter=max_iter,
            seed=300 + idx,
        )
        alias = cfg.file_name(letter)
        model_path = out_dir / f"3d_{alias}.json"
        hmm.save_json(model_path)
        trained[letter] = hmm
        if verbose:
            ll_start = history[0] if history else float("nan")
            ll_end = history[-1] if history else float("nan")
            print(
                f"[3D K={extractor.K}] '{letter}' (N={n_states:2d}) | "
                f"muestras={len(samples):2d} | iter={len(history):2d} | "
                f"logL: {ll_start:8.1f} -> {ll_end:8.1f} | {model_path.name}"
            )
    return trained


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entrena los 27 HMM (2D y/o 3D) y exporta a models/*.json"
    )
    parser.add_argument(
        "--mode",
        choices=("2d", "3d", "all"),
        default="all",
        help="Modo a entrenar: 2d, 3d o all (por defecto: all)",
    )
    parser.add_argument("--K", type=int, default=None, help="Sectores angulares (8 o 16)")
    parser.add_argument("--samples", type=int, default=24, help="Muestras base si no hay capturas")
    parser.add_argument("--augment", type=int, default=2, help="Aumentos por muestra")
    parser.add_argument("--max-iter", type=int, default=35, help="Iteraciones máx. de Baum-Welch")
    args = parser.parse_args()

    if args.mode in ("2d", "all"):
        train_all_2d(
            K=args.K,
            samples_if_empty=args.samples,
            augment_factor=args.augment,
            max_iter=args.max_iter,
            verbose=True,
        )
    if args.mode in ("3d", "all"):
        train_all_3d(
            K=args.K,
            samples_if_empty=args.samples,
            augment_factor=args.augment,
            max_iter=args.max_iter,
            verbose=True,
        )


if __name__ == "__main__":
    main()

