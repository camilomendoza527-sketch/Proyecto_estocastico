"""Cargador único de config/config.toml.

Todo el proyecto obtiene el alfabeto de clases, los parámetros de
discretización y el número de estados por letra desde aquí. Así no hay
valores duplicados ni hardcodeados en otros módulos.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.toml"

# Restricción de la SPEC §3.3: 27 clases.
EXPECTED_N_LETTERS = 27


@dataclass(frozen=True)
class FeatureConfig:
    """Parámetros de discretización del trazo (sección [features])."""

    K: int
    K_compare: tuple[int, ...]
    n_jump_symbols: int
    resample_step: float
    smoothing_window: int
    expected_letter_size: float
    y_axis_down: bool
    expected_angle_3d: float = 0.7
    sphere_radius: float = 1.0

    @property
    def n_symbols(self) -> int:
        """Tamaño del alfabeto de observaciones: |V| = K + n_jump_symbols."""
        return self.K + self.n_jump_symbols


@dataclass(frozen=True)
class ProjectConfig:
    """Configuración completa y validada del proyecto."""

    letters: tuple[str, ...]
    file_alias: dict[str, str]
    features: FeatureConfig
    states: dict[str, int]
    stroke_convention: tuple[str, ...]
    prefix_fractions: tuple[float, ...]

    def file_name(self, letter: str) -> str:
        """Nombre ASCII seguro para archivos de una letra (p. ej. 'ñ' -> 'enie')."""
        return self.file_alias.get(letter, letter)


def _validate_letters(letters: list[str]) -> None:
    if len(letters) != EXPECTED_N_LETTERS:
        raise ValueError(f"Se esperaban {EXPECTED_N_LETTERS} letras, hay {len(letters)}.")
    if len(set(letters)) != len(letters):
        raise ValueError("El alfabeto contiene letras repetidas.")
    for letter in letters:
        # Una sola letra minúscula: excluye "ch", "ll" y mayúsculas.
        if len(letter) != 1 or not letter.isalpha() or letter != letter.lower():
            raise ValueError(f"Clase inválida en el alfabeto: {letter!r}")
        if letter in "áéíóúü":
            raise ValueError(f"No se admiten vocales con tilde: {letter!r}")


def _validate_features(f: FeatureConfig) -> None:
    if f.K not in (8, 16):
        raise ValueError("features.K debe ser 8 o 16.")
    if any(k not in (8, 16) for k in f.K_compare):
        raise ValueError("features.K_compare solo admite 8 y 16.")
    if f.n_jump_symbols != 4:
        raise ValueError("features.n_jump_symbols debe ser 4 (→ ↑ ← ↓).")
    if f.resample_step <= 0:
        raise ValueError("features.resample_step debe ser > 0.")
    if f.smoothing_window < 1 or f.smoothing_window % 2 == 0:
        raise ValueError("features.smoothing_window debe ser un entero impar >= 1.")
    if f.expected_letter_size <= 0:
        raise ValueError("features.expected_letter_size debe ser > 0.")
    if f.expected_angle_3d <= 0:
        raise ValueError("features.expected_angle_3d debe ser > 0.")
    if f.sphere_radius <= 0:
        raise ValueError("features.sphere_radius debe ser > 0.")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    """Lee y valida un archivo TOML de configuración."""
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    letters = list(raw["alphabet"]["letters"])
    _validate_letters(letters)

    file_alias = dict(raw["alphabet"].get("file_alias", {}))

    f = raw["features"]
    features = FeatureConfig(
        K=int(f["K"]),
        K_compare=tuple(int(k) for k in f["K_compare"]),
        n_jump_symbols=int(f["n_jump_symbols"]),
        resample_step=float(f["resample_step"]),
        smoothing_window=int(f["smoothing_window"]),
        expected_letter_size=float(f["expected_letter_size"]),
        y_axis_down=bool(f["y_axis_down"]),
        expected_angle_3d=float(f.get("expected_angle_3d", 0.7)),
        sphere_radius=float(f.get("sphere_radius", 1.0)),
    )
    _validate_features(features)

    states = {k: int(v) for k, v in raw["hmm"]["states"].items()}
    if set(states) != set(letters):
        missing = set(letters) - set(states)
        extra = set(states) - set(letters)
        raise ValueError(f"hmm.states no coincide con el alfabeto. Faltan: {missing}, sobran: {extra}")
    if any(n < 1 for n in states.values()):
        raise ValueError("Cada letra debe tener N >= 1 estados.")

    stroke_convention = tuple(str(s) for s in raw["capture"]["stroke_convention"])

    prefix_fractions = tuple(float(p) for p in raw["evaluation"]["prefix_fractions"])
    if not prefix_fractions or any(not 0.0 < p <= 1.0 for p in prefix_fractions):
        raise ValueError("evaluation.prefix_fractions debe contener valores en (0, 1].")

    return ProjectConfig(
        letters=tuple(letters),
        file_alias=file_alias,
        features=features,
        states=states,
        stroke_convention=stroke_convention,
        prefix_fractions=prefix_fractions,
    )


@lru_cache(maxsize=1)
def get_config() -> ProjectConfig:
    """Configuración por defecto del proyecto (se lee una sola vez)."""
    return load_config(DEFAULT_CONFIG_PATH)
