"""Validated SHAP input data, objective transforms, and atomic result publication."""
from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

import numpy as np
import pandas as pd

from src.bo.config import FEATURE_COLUMNS

TARGET_COLUMNS = ("eta500_V", "T80_h")
MAX_LOOCV_SAMPLES = 250
MAX_JOBS = max(1, os.cpu_count() or 1)

@dataclass(frozen=True)
class Dataset:
    features: pd.DataFrame
    eta: np.ndarray
    t80: np.ndarray
    fingerprint: str

    @property
    def size(self) -> int:
        return len(self.features)


def load_dataset(path: Path) -> Dataset:
    """Load the required numeric columns and remove incomplete rows."""
    try:
        header = pd.read_csv(path, nrows=0)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input data file does not exist: {path}") from exc
    required = [*FEATURE_COLUMNS, *TARGET_COLUMNS]
    missing = [column for column in required if column not in header.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    raw = pd.read_csv(
        path,
        usecols=required,
        nrows=MAX_LOOCV_SAMPLES + 1,
    )
    if len(raw) > MAX_LOOCV_SAMPLES:
        raise ValueError(
            "The SHAP LOOCV workflow supports at most "
            f"{MAX_LOOCV_SAMPLES} input rows."
        )
    numeric = raw.loc[:, required].apply(pd.to_numeric, errors="coerce")
    clean = numeric.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(clean) < 3:
        raise ValueError("At least three complete finite samples are required for LOOCV.")
    if (clean["T80_h"] < 0).any():
        raise ValueError("T80_h must contain non-negative hours.")
    features = clean.loc[:, list(FEATURE_COLUMNS)].astype(float)
    eta = clean["eta500_V"].to_numpy(dtype=float, copy=True)
    t80 = clean["T80_h"].to_numpy(dtype=float, copy=True)
    return Dataset(features, eta, t80, dataset_fingerprint(features, eta, t80))


def dataset_fingerprint(
    features: pd.DataFrame, eta: np.ndarray, t80: np.ndarray
) -> str:
    """Identify the cleaned numeric rows and their order across both stages."""
    frame = pd.DataFrame(
        np.column_stack((features.to_numpy(dtype=np.float64), eta, t80)),
        columns=[*FEATURE_COLUMNS, *TARGET_COLUMNS],
    )
    digest = hashlib.sha256("\0".join(frame.columns).encode("utf-8"))
    digest.update(np.ascontiguousarray(frame.to_numpy(dtype="<f8")).tobytes())
    return digest.hexdigest()


def validate_jobs(jobs: int) -> int:
    if jobs != -1 and not 1 <= jobs <= MAX_JOBS:
        raise ValueError(f"--jobs must be -1 or an integer from 1 through {MAX_JOBS}.")
    return jobs


@contextmanager
def atomic_output_directory(out_dir: Path) -> Iterator[Path]:
    """Publish a result directory only after all requested outputs succeed."""
    final_dir = Path(out_dir)
    if final_dir.exists() or final_dir.is_symlink():
        raise FileExistsError(f"Output directory already exists: {final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=f".{final_dir.name}.tmp-", dir=final_dir.parent
    ) as temporary:
        staging_dir = Path(temporary)
        yield staging_dir
        if final_dir.exists() or final_dir.is_symlink():
            raise FileExistsError(f"Output directory already exists: {final_dir}")
        staging_dir.rename(final_dir)


def transform_target(values: np.ndarray, target_kind: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Targets must contain only finite values.")
    if target_kind == "eta":
        return -array
    if target_kind == "t80":
        if np.any(array < 0):
            raise ValueError("T80_h must contain non-negative hours.")
        return np.log1p(array)
    raise ValueError(f"Unknown target kind: {target_kind}")


def inverse_target(values: np.ndarray, target_kind: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if target_kind == "eta":
        return -array
    if target_kind == "t80":
        return np.expm1(array)
    raise ValueError(f"Unknown target kind: {target_kind}")

