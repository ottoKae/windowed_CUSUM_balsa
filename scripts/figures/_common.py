"""Small shared helpers for the manuscript figure builders."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data/local/paper_v1"
RESULTS = ROOT / "results"
GREEN = "#77964A"
YELLOW = "#FBC040"
BLUE = "#2C7FB8"
DARK_BLUE = "#253494"
INK = "#252525"
GREY = "#8A8A8A"
LIGHT_GREY = "#E7E7E7"


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.8,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def panel_label(axis: plt.Axes, letter: str) -> None:
    axis.text(
        0.012, 0.985, f"({letter})", transform=axis.transAxes,
        ha="left", va="top", fontsize=9, fontweight="bold", zorder=20,
    )


def save_figure(figure: plt.Figure, output: Path, *, dpi: int = 600) -> Path:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", pad_inches=0.025)
    if output.suffix.lower() == ".pdf":
        figure.savefig(output.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)
    print(f"figure={output}")
    return output


def require(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n" + "\n".join(missing))


@dataclass(frozen=True)
class CaseSeries:
    sample_id: str
    polygon_id: str
    track_key: str
    calendar: pd.DatetimeIndex
    pixels: np.ndarray
    references: pd.DatetimeIndex

    @property
    def median(self) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered")
            return np.nanmedian(self.pixels, axis=0)

    @property
    def mean(self) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            return np.nanmean(self.pixels, axis=0)


def load_case_series(data_root: Path, historical_id: int = 1499) -> CaseSeries:
    references = pd.read_parquet(data_root / "references/reference_ground_events.parquet")
    selected = references.loc[
        references.historical_id.astype(str).eq(str(historical_id))
    ].sort_values("event_month")
    if selected.empty:
        raise ValueError(f"historical_id {historical_id} is absent from the reference ledger")
    sample_id = str(selected.iloc[0].sample_id)
    polygon_id = str(selected.iloc[0].polygon_id)
    track_key = str(selected.iloc[0].primary_track_key)
    associations = pd.read_parquet(data_root / "polygon_pixels/pixel_associations.parquet")
    local = associations.loc[
        associations.sample_id.astype(str).eq(sample_id)
        & associations.track_key.astype(str).eq(track_key)
    ]
    pixels = pd.read_parquet(data_root / f"polygon_pixels/cache/{track_key}.pixels.parquet")
    mapping = pixels.set_index("pixel_track_id").cache_index
    indexes = mapping.loc[local.pixel_track_id].to_numpy(int)
    with np.load(data_root / f"polygon_pixels/cache/{track_key}.npz") as arrays:
        values = np.asarray(arrays["values"][indexes], dtype=np.float32)
        calendar = pd.DatetimeIndex(arrays["calendar"])
    return CaseSeries(
        sample_id=sample_id,
        polygon_id=polygon_id,
        track_key=track_key,
        calendar=calendar,
        pixels=values,
        references=pd.DatetimeIndex(pd.to_datetime(selected.event_month)),
    )


def histogram_mode(values: np.ndarray, bin_width: float = 0.5) -> np.ndarray:
    """Return a robust monthly histogram mode for ``pixel x month`` values."""
    output = np.full(values.shape[1], np.nan, dtype=float)
    for month in range(values.shape[1]):
        finite = values[:, month][np.isfinite(values[:, month])]
        if not len(finite):
            continue
        lower = np.floor(finite.min() / bin_width) * bin_width
        upper = np.ceil(finite.max() / bin_width) * bin_width + bin_width
        edges = np.arange(lower, upper + 0.5 * bin_width, bin_width)
        counts, edges = np.histogram(finite, bins=edges)
        index = int(np.argmax(counts))
        output[month] = 0.5 * (edges[index] + edges[index + 1])
    return output


def representative_pixel(values: np.ndarray) -> np.ndarray:
    """Select the pixel closest to the polygon median over finite months."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        target = np.nanmedian(values, axis=0)
        distance = np.nanmedian(np.abs(values - target[None, :, :]), axis=(1, 2))
    return values[int(np.nanargmin(distance))]


def add_src_to_path() -> None:
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
