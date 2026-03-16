from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # Must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from PIL import Image, ImageDraw

# =========================================================
# PATHS & CONSTANTS
# =========================================================

# Navigate two levels up from scripts/member5/ to reach the project root
PROJECT_ROOT   = Path(__file__).resolve().parents[2]
CLASSIFIED_DIR = PROJECT_ROOT / "data" / "classified"
METRICS_CSV    = PROJECT_ROOT / "data" / "urban_growth_metrics.csv"
CHARTS_DIR     = PROJECT_ROOT / "data" / "charts"

CITIES = ["riverside", "phoenix", "las_vegas", "austin"]
YEARS  = [1990, 2000, 2010, 2020]

# Value used for nodata pixels in the classification TIFFs (from Member 3)
NODATA_VALUE = 255

CITY_LABELS = {
    "riverside": "Riverside",
    "phoenix":   "Phoenix",
    "las_vegas": "Las Vegas",
    "austin":    "Austin",
}

CITY_COLORS = {
    "riverside": "#2F6F4E",
    "phoenix":   "#E76F51",
    "las_vegas": "#4C9BE8",
    "austin":    "#7A9E7E",
}

THEME = {
    "background": "#F7F6F2",
    "primary": "#2F6F4E",
    "accent": "#E76F51",
    "secondary": "#4C9BE8",
    "card": "#FFFFFF",
    "text": "#000000",
    "muted": "#000000",
    "border": "#E4E2D9",
}

CHART_STYLE = {
    "title_size": 14,
    "subtitle_size": 11,
    "label_size": 10,
    "tick_size": 9,
    "annot_size": 8,
    "title_weight": "semibold",
    "grid_alpha": 0.10,
    "spine_width": 0.8,
    "scatter_alpha": 0.55,
    "card_round_feel": True,
}

SUBCLASS_PALETTES = {
    "urban": "OrRd",
    "vegetation": "YlGn",
    "farmland": "GnBu",
    "bare_soil": "YlOrBr",
    "water": "PuBu",
}

MAP_CLASS_COLORS = {
    "urban":      [214, 96, 77, 255],
    "vegetation": [126, 168, 120, 255],
    "water":      [100, 149, 237, 255],
    "other":      [232, 229, 222, 255],
}

# =========================================================
# PAGE CONFIG  (must be the very first Streamlit call)
# =========================================================

st.set_page_config(
    page_title="Urban Expansion Mapping",
    page_icon="U",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# GLOBAL CSS — nature-inspired light theme
# =========================================================

st.markdown("""
<style>
/* ── Base backgrounds ──────────────────────────────────────────── */
.stApp                         { background-color: #F7F6F2; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F3F1EA 0%, #ECE8DD 100%);
    border-right: 1px solid #E4E2D9;
}

/* ── Typography ────────────────────────────────────────────────── */
h1, h2, h3 { color: #000000 !important; }
p, li, label { color: #000000; }

/* ── Metric cards ───────────────────────────────────────────────── */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #E4E2D9;
    border-radius: 12px;
    padding: 18px 14px;
    text-align: center;
    margin-bottom: 10px;
    box-shadow: 0 4px 14px rgba(47, 59, 53, 0.06);
}
.metric-value {
    font-size: 2em;
    font-weight: 700;
    color: #000000;
}
.metric-label {
    font-size: 0.82em;
    color: #000000;
    margin-top: 4px;
}

/* ── Highlighted info boxes ─────────────────────────────────────── */
.info-box {
    background: #FFFFFF;
    border-left: 4px solid #4C9BE8;
    border-radius: 4px;
    padding: 10px 14px;
    margin: 6px 0;
    color: #000000;
    font-size: 0.9em;
    border: 1px solid #E4E2D9;
}

/* ── Hero banner ────────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(120deg, #F6EFE2 0%, #F7F6F2 45%, #EAF3EE 100%);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 24px;
    border: 1px solid #E4E2D9;
}

/* ── Tabs ───────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab"]          { color: #000000; }
.stTabs [aria-selected="true"]         { color: #000000 !important; }

/* ── Streamlit's native metric widget ──────────────────────────── */
[data-testid="stMetric"] label        { color: #000000 !important; }
[data-testid="stMetricValue"]          { color: #000000 !important; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA LOADERS  (cached so they only run once per session)
# =========================================================

@st.cache_data
def load_metrics() -> pd.DataFrame:
    """
    Load the urban growth metrics CSV produced by Member 4.
    Returns an empty DataFrame when the file does not exist yet.
    """
    if METRICS_CSV.exists():
        return pd.read_csv(METRICS_CSV)
    return pd.DataFrame(
        columns=["city", "year", "urban_area_km2", "growth_km2", "growth_pct", "growth_pct_display"]
    )


@st.cache_data
def load_riverside_training_samples() -> pd.DataFrame:
    """Load and combine all Riverside training CSV rows across available years."""
    frames = []
    csv_files = sorted((PROJECT_ROOT / "data").glob("riverside_*_training_samples.csv"))

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        required = {"year", "subclass", "red", "green", "blue", "nir", "ndvi"}
        if not required.issubset(df.columns):
            continue

        keep = df[["year", "subclass", "red", "green", "blue", "nir", "ndvi"]].copy()
        keep["subclass"] = keep["subclass"].astype(str).str.strip().str.lower()
        for col in ["red", "green", "blue", "nir", "ndvi"]:
            keep[col] = pd.to_numeric(keep[col], errors="coerce")
        keep["year"] = pd.to_numeric(keep["year"], errors="coerce")
        keep = keep.dropna(subset=["year"])
        keep["year"] = keep["year"].astype(int)
        frames.append(keep)

    if not frames:
        return pd.DataFrame(columns=["year", "subclass", "red", "green", "blue", "nir", "ndvi"])

    return pd.concat(frames, ignore_index=True)


def crop_to_valid(data: np.ndarray, nodata: int = NODATA_VALUE, pad: int = 10):
    valid = data != nodata
    if not np.any(valid):
        return data

    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    rmin = max(0, rmin - pad)
    rmax = min(data.shape[0], rmax + pad + 1)
    cmin = max(0, cmin - pad)
    cmax = min(data.shape[1], cmax + pad + 1)

    return data[rmin:rmax, cmin:cmax]

@st.cache_data
def load_classification_data(city: str, year: int) -> "np.ndarray | None":
    """Read a classified GeoTIFF band for a city/year and return cropped raw class values."""
    tif_path = CLASSIFIED_DIR / f"{city}_{year}_classification.tif"
    if not tif_path.exists():
        return None

    with rasterio.open(tif_path) as src:
        data = src.read(1)

    data = crop_to_valid(data, nodata=NODATA_VALUE, pad=10)
    return data

@st.cache_data
def load_classification_rgba(city: str, year: int) -> "np.ndarray | None":
    data = load_classification_data(city, year)
    if data is None:
        return None

    h, w = data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # light neutral background for all valid pixels
    rgba[data != NODATA_VALUE] = [240, 237, 232, 255]

    # urban only highlighted
    rgba[data == 1] = [201, 92, 74, 255]

    # nodata transparent
    rgba[data == NODATA_VALUE] = [0, 0, 0, 0]

    return rgba

@st.cache_data
def classification_pil(city: str, year: int, width: int = 640) -> "Image.Image | None":
    """
    Return a PIL Image of the classification map scaled to *width* pixels wide.
    Uses nearest-neighbour resampling to preserve hard classification boundaries.
    """
    rgba = load_classification_rgba(city, year)
    if rgba is None:
        return None
    img = Image.fromarray(rgba, mode="RGBA")
    aspect = img.height / img.width
    return img.resize((width, int(width * aspect)), Image.NEAREST)

@st.cache_data
def generate_timelapse_gif(city: str) -> "bytes | None":
    frames = []

    for year in YEARS:
        img = classification_pil(city, year, width=420)

        if img is None:
            continue

        frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=64))

    if not frames:
        return None

    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        loop=0,
        duration=1300,
        optimize=False,
        disposal=2,
    )
    buf.seek(0)
    return buf.read()


# =========================================================
# CHART HELPERS
# =========================================================

def _style_axis_title(ax, title: str):
    ax.set_title(
        title,
        color=THEME["text"],
        fontsize=CHART_STYLE["title_size"],
        fontweight=CHART_STYLE["title_weight"],
        pad=12,
    )

def _style_axis_labels(ax, xlabel: str = "", ylabel: str = ""):
    if xlabel:
        ax.set_xlabel(xlabel, color=THEME["muted"], fontsize=CHART_STYLE["label_size"], labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=THEME["muted"], fontsize=CHART_STYLE["label_size"], labelpad=8)

def _pretty_fig(figsize=(9, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(THEME["background"])
    ax.set_facecolor(THEME["card"])

    ax.tick_params(
        colors=THEME["muted"],
        labelsize=CHART_STYLE["tick_size"],
        length=0
    )

    for spine in ax.spines.values():
        spine.set_edgecolor(THEME["border"])
        spine.set_linewidth(CHART_STYLE["spine_width"])

    ax.grid(True, alpha=CHART_STYLE["grid_alpha"], color=THEME["border"])
    return fig, ax


def chart_area_over_time(df: pd.DataFrame, active_cities: "list[str] | None" = None):
    """
    Line chart — urban area (km²) vs year.
    When *active_cities* is provided, other cities are drawn faded.
    """
    fig, ax = _pretty_fig((10, 5))

    for city in CITIES:
        city_data = df[df["city"] == city].sort_values("year")
        highlighted = (active_cities is None) or (city in active_cities)
        alpha  = 1.0 if highlighted else 0.2
        lwidth = 2.5 if highlighted else 1.2

        ax.plot(
            city_data["year"],
            city_data["urban_area_km2"],
            marker="o",
            linewidth=lwidth,
            alpha=alpha,
            label=CITY_LABELS[city],
            color=CITY_COLORS[city],
        )

        # Annotate final data point for highlighted cities
        if highlighted:
            last = city_data.iloc[-1]
            if pd.notna(last["urban_area_km2"]):
                ax.annotate(
                    f"{last['urban_area_km2']:.0f}",
                    xy=(last["year"], last["urban_area_km2"]),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=CITY_COLORS[city],
                    fontweight="bold",
                )

    ax.set_xticks(YEARS)
    ax.set_xlabel("Year",             color=THEME["muted"], fontsize=11)
    ax.set_ylabel("Urban Area (km²)", color=THEME["muted"], fontsize=11)
    ax.set_title("Urban Area Over Time (1990–2020)",
                 color=THEME["text"], fontsize=13, fontweight="bold", pad=12)
    ax.legend(facecolor=THEME["card"], edgecolor=THEME["border"],
              labelcolor=THEME["text"], fontsize=9)
    plt.tight_layout()
    return fig


def chart_city_comparison(df: pd.DataFrame):
    """Bar chart — total urban growth (km²) from 1990 to 2020 per city."""
    comparison = []
    for city in CITIES:
        city_data = df[df["city"] == city].sort_values("year")
        a1990 = city_data.loc[city_data["year"] == 1990, "urban_area_km2"].values
        a2020 = city_data.loc[city_data["year"] == 2020, "urban_area_km2"].values
        if len(a1990) and len(a2020) and pd.notna(a1990[0]) and pd.notna(a2020[0]):
            growth = round(a2020[0] - a1990[0], 2)
            pct    = round(((a2020[0] - a1990[0]) / a1990[0]) * 100, 1) if a1990[0] > 0 else 0
            comparison.append({
                "city":       CITY_LABELS[city],
                "growth_km2": growth,
                "growth_pct": pct,
                "color":      CITY_COLORS[city],
            })
    comp_df = pd.DataFrame(comparison)

    fig, ax = _pretty_fig((8, 5))

    if not comp_df.empty:
        bars = ax.bar(
            comp_df["city"],
            comp_df["growth_km2"],
            color=comp_df["color"].tolist(),
            edgecolor="#0f172a",
            linewidth=0.8,
        )
        for bar, (_, row) in zip(bars, comp_df.iterrows()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(comp_df["growth_km2"]) * 0.02,
                f"+{row['growth_pct']:.0f}%",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#e2e8f0",
            )

    ax.set_title("Total Urban Growth 1990–2020 by City",
                 color=THEME["text"], fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("City",             color=THEME["muted"], fontsize=11)
    ax.set_ylabel("Growth (km²)",     color=THEME["muted"], fontsize=11)
    plt.tight_layout()
    return fig

def chart_growth_slopegraph(df: pd.DataFrame, active_cities: list[str]):
    """Slopegraph comparing 1990 vs 2020 urban area."""
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    fig.patch.set_facecolor(THEME["background"])
    ax.set_facecolor(THEME["card"])

    years_to_show = [1990, 2020]
    x_pos = [0, 1]

    max_val = 0
    for city in active_cities:
        city_data = df[df["city"] == city].sort_values("year")
        start = city_data.loc[city_data["year"] == 1990, "urban_area_km2"].values
        end = city_data.loc[city_data["year"] == 2020, "urban_area_km2"].values
        if len(start) and len(end) and pd.notna(start[0]) and pd.notna(end[0]):
            max_val = max(max_val, float(start[0]), float(end[0]))

    for city in active_cities:
        city_data = df[df["city"] == city].sort_values("year")
        start = city_data.loc[city_data["year"] == 1990, "urban_area_km2"].values
        end = city_data.loc[city_data["year"] == 2020, "urban_area_km2"].values
        if not (len(start) and len(end) and pd.notna(start[0]) and pd.notna(end[0])):
            continue

        y = [float(start[0]), float(end[0])]
        color = CITY_COLORS.get(city, THEME["secondary"])

        ax.plot(x_pos, y, color=color, linewidth=3, alpha=0.95)
        ax.scatter(x_pos, y, s=90, color=color, edgecolors=THEME["card"], linewidths=1.5, zorder=3)

        ax.text(-0.03, y[0], f"{CITY_LABELS[city]}  {y[0]:.0f}", ha="right", va="center",
                fontsize=9, color=THEME["text"])
        ax.text(1.03, y[1], f"{y[1]:.0f}", ha="left", va="center",
                fontsize=9, color=color, fontweight="bold")

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(0, max_val * 1.12 if max_val else 1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(y) for y in years_to_show], fontsize=10, color=THEME["muted"])
    ax.set_ylabel("Urban Area (km²)", fontsize=10, color=THEME["muted"])
    ax.set_title("1990 → 2020 Growth Snapshot", fontsize=14, fontweight="semibold",
                 color=THEME["text"], pad=12)

    ax.grid(axis="y", alpha=0.10, color=THEME["border"])
    ax.grid(axis="x", visible=False)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(THEME["border"])
    ax.spines["bottom"].set_color(THEME["border"])
    ax.tick_params(colors=THEME["muted"])

    plt.tight_layout(pad=1.2)
    return fig


def chart_city_year_heatmap(df: pd.DataFrame, active_cities: list[str]):
    """Heatmap of urban area by city and year."""
    sub = df[df["city"].isin(active_cities)].copy()
    if sub.empty:
        return None

    heat = (
        sub.pivot(index="city", columns="year", values="urban_area_km2")
        .reindex(active_cities)
        .reindex(columns=YEARS)
    )

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    fig.patch.set_facecolor(THEME["background"])
    ax.set_facecolor(THEME["card"])

    cmap = plt.cm.get_cmap("YlOrBr").copy()
    masked = np.ma.masked_invalid(heat.values)
    im = ax.imshow(masked, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(YEARS)))
    ax.set_xticklabels([str(y) for y in YEARS], fontsize=10, color=THEME["muted"])
    ax.set_yticks(np.arange(len(active_cities)))
    ax.set_yticklabels([CITY_LABELS[c] for c in active_cities], fontsize=10, color=THEME["muted"])

    ax.set_title("Urban Area Heatmap", fontsize=14, fontweight="semibold",
                 color=THEME["text"], pad=12)

    for i in range(masked.shape[0]):
        for j in range(masked.shape[1]):
            val = heat.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8.5, color=THEME["text"])

    for spine in ax.spines.values():
        spine.set_edgecolor(THEME["border"])
        spine.set_linewidth(0.8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.outline.set_edgecolor(THEME["border"])
    cbar.ax.tick_params(labelsize=8, colors=THEME["muted"])

    plt.tight_layout(pad=1.2)
    return fig


def get_growth_summary(df: pd.DataFrame, active_cities: list[str]) -> pd.DataFrame:
    rows = []
    for city in active_cities:
        city_data = df[df["city"] == city].sort_values("year")
        a1990 = city_data.loc[city_data["year"] == 1990, "urban_area_km2"].values
        a2020 = city_data.loc[city_data["year"] == 2020, "urban_area_km2"].values
        if len(a1990) and len(a2020) and pd.notna(a1990[0]) and pd.notna(a2020[0]):
            growth = float(a2020[0] - a1990[0])
            pct = ((a2020[0] - a1990[0]) / a1990[0]) * 100 if a1990[0] > 0 else 0
            rows.append({
                "city": city,
                "area_1990": float(a1990[0]),
                "area_2020": float(a2020[0]),
                "growth_km2": growth,
                "growth_pct": float(pct),
            })
    return pd.DataFrame(rows)


def render_growth_snapshot_cards(summary_df: pd.DataFrame):
    cols = st.columns(len(summary_df) if len(summary_df) > 0 else 1, gap="small")
    for col, (_, row) in zip(cols, summary_df.iterrows()):
        color = CITY_COLORS.get(row["city"], THEME["primary"])
        # arrow = "▲" if row["growth_km2"] >= 0 else "▼"
        with col:
            st.markdown(f"""
            <div style="
                background:{THEME['card']};
                border:1px solid {THEME['border']};
                border-radius:18px;
                padding:16px 16px 14px 16px;
                box-shadow:0 6px 20px rgba(47, 59, 53, 0.06);
                min-height:148px;
            ">
                <div style="font-size:0.86em; color:#6F7E73; margin-bottom:6px;">
                    {CITY_LABELS[row["city"]]}
                </div>
                <div style="font-size:1.8em; font-weight:700; color:{color}; line-height:1.1;">
                    {row["area_2020"]:.0f}
                </div>
                <div style="font-size:0.82em; color:#6F7E73; margin-top:2px;">
                    km² in 2020
                </div>
                <div style="font-size:0.82em; color:#6F7E73;">
                    {row["growth_pct"]:+.1f}% since 1990
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_decade_pulse_cards(df: pd.DataFrame, active_cities: list[str]):
    periods = [(1990, 2000), (2000, 2010), (2010, 2020)]
    cols = st.columns(3, gap="small")

    for col, (start, end) in zip(cols, periods):
        best_city = None
        best_growth = None

        for city in active_cities:
            s = df[(df["city"] == city) & (df["year"] == start)]["urban_area_km2"].values
            e = df[(df["city"] == city) & (df["year"] == end)]["urban_area_km2"].values
            if len(s) and len(e) and pd.notna(s[0]) and pd.notna(e[0]):
                growth = float(e[0] - s[0])
                if best_growth is None or growth > best_growth:
                    best_growth = growth
                    best_city = city

        if best_city is None:
            continue

        color = CITY_COLORS.get(best_city, THEME["primary"])
        with col:
            st.markdown(f"""
            <div style="
                background:{THEME['card']};
                border:1px solid {THEME['border']};
                border-radius:18px;
                padding:16px;
                box-shadow:0 6px 20px rgba(47, 59, 53, 0.06);
            ">
                <div style="font-size:0.82em; color:#6F7E73;">{start}–{end}</div>
                <div style="font-size:1.15em; font-weight:700; color:{THEME['text']}; margin-top:4px;">
                    Strongest Growth
                </div>
                <div style="font-size:1.2em; font-weight:700; color:{color}; margin-top:10px;">
                    {CITY_LABELS[best_city]}
                </div>
                <div style="font-size:0.92em; color:#6F7E73; margin-top:4px;">
                    +{best_growth:.0f} km²
                </div>
            </div>
            """, unsafe_allow_html=True)


## THIS DOWN IS FOR THE CITY COMPARISON PAGE HELPERS
def get_city_summary_1990_2020(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for city in CITIES:
        city_data = df[df["city"] == city].sort_values("year")
        a1990 = city_data.loc[city_data["year"] == 1990, "urban_area_km2"].values
        a2020 = city_data.loc[city_data["year"] == 2020, "urban_area_km2"].values

        if len(a1990) and len(a2020) and pd.notna(a1990[0]) and pd.notna(a2020[0]):
            growth_km2 = float(a2020[0] - a1990[0])
            growth_pct = ((a2020[0] - a1990[0]) / a1990[0] * 100) if a1990[0] > 0 else 0.0
            rows.append({
                "city": city,
                "label": CITY_LABELS[city],
                "area_1990": float(a1990[0]),
                "area_2020": float(a2020[0]),
                "growth_km2": growth_km2,
                "growth_pct": float(growth_pct),
                "color": CITY_COLORS[city],
            })

    return pd.DataFrame(rows)


def chart_bubble_matrix(df: pd.DataFrame):
    """
    Bubble matrix:
    x = year, y = city, bubble size = urban area
    """
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    fig.patch.set_facecolor(THEME["background"])
    ax.set_facecolor(THEME["card"])

    plot_df = df[df["city"].isin(CITIES)].copy()
    plot_df["city_label"] = plot_df["city"].map(CITY_LABELS)

    city_order = list(reversed(CITIES))
    y_map = {city: i for i, city in enumerate(city_order)}

    max_area = plot_df["urban_area_km2"].max()
    if pd.isna(max_area) or max_area <= 0:
        max_area = 1

    for city in CITIES:
        sub = plot_df[plot_df["city"] == city].sort_values("year")
        if sub.empty:
            continue

        sizes = (sub["urban_area_km2"] / max_area) * 2800 + 120

        ax.scatter(
            sub["year"],
            [y_map[city]] * len(sub),
            s=sizes,
            color=CITY_COLORS[city],
            alpha=0.75,
            edgecolors="white",
            linewidths=1.8,
            zorder=3,
        )

        for _, row in sub.iterrows():
            ax.text(
                row["year"],
                y_map[city],
                f"{row['urban_area_km2']:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
                zorder=4,
            )

    ax.set_xticks(YEARS)
    ax.set_yticks(range(len(city_order)))
    ax.set_yticklabels([CITY_LABELS[c] for c in city_order], fontsize=10, color=THEME["text"])
    ax.set_xlim(min(YEARS) - 4, max(YEARS) + 4)
    ax.set_ylim(-0.7, len(city_order) - 0.3)

    ax.set_title(
        "Urban Footprint Bubble Matrix",
        fontsize=14,
        fontweight="bold",
        color=THEME["text"],
        pad=14,
    )
    ax.set_xlabel("Year", fontsize=10, color=THEME["muted"])
    ax.set_ylabel("")

    ax.grid(axis="x", color=THEME["border"], alpha=0.25, linewidth=1)
    ax.grid(axis="y", visible=False)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(THEME["border"])
    ax.tick_params(axis="x", colors=THEME["muted"])
    ax.tick_params(axis="y", length=0)

    plt.tight_layout()
    return fig

def chart_nested_area_circles(df: pd.DataFrame):
    """
    For each city:
    outer circle = 2020 urban area
    inner circle = 1990 urban area
    """
    summary = get_city_summary_1990_2020(df)
    if summary.empty:
        return None

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    fig.patch.set_facecolor(THEME["background"])
    ax.set_facecolor(THEME["card"])

    max_area = summary["area_2020"].max()
    if pd.isna(max_area) or max_area <= 0:
        max_area = 1

    x_positions = np.arange(len(summary)) * 2.2

    for i, (_, row) in enumerate(summary.iterrows()):
        x = x_positions[i]

        s_2020 = (row["area_2020"] / max_area) * 12000 + 600
        s_1990 = (row["area_1990"] / max_area) * 12000 + 220

        # outer = 2020
        ax.scatter(
            x, 0,
            s=s_2020,
            color=row["color"],
            alpha=0.22,
            edgecolors=row["color"],
            linewidths=2.2,
            zorder=2,
        )

        # inner = 1990
        ax.scatter(
            x, 0,
            s=s_1990,
            color=THEME["card"],
            edgecolors=row["color"],
            linewidths=2.0,
            zorder=3,
        )

        ax.text(
            x, 0.02,
            f"{row['label']}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=THEME["text"],
        )

        ax.text(
            x, -0.10,
            f"{row['growth_km2']:+.0f} km²",
            ha="center",
            va="top",
            fontsize=9,
            color=row["color"],
            fontweight="bold",
        )

        ax.text(
            x, -0.18,
            f"{row['growth_pct']:+.1f}%",
            ha="center",
            va="top",
            fontsize=8.5,
            color="#6F7E73",
        )

    # tiny legend note
    ax.text(
        x_positions[0] - 1.1, 0.82,
        "Outer ring = 2020 area\nInner circle = 1990 area",
        ha="left",
        va="top",
        fontsize=9,
        color="#6F7E73",
    )

    ax.set_title(
        "1990 vs 2020 Urban Area by City",
        fontsize=14,
        fontweight="bold",
        color=THEME["text"],
        pad=12,
    )

    ax.set_xlim(x_positions.min() - 1.4, x_positions.max() + 1.4)
    ax.set_ylim(-0.55, 0.95)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return fig

def chart_decade_pulse_bubbles(df: pd.DataFrame):
    periods = [(1990, 2000), (2000, 2010), (2010, 2020)]
    totals = []

    for start, end in periods:
        total = 0.0
        for city in CITIES:
            s = df[(df["city"] == city) & (df["year"] == start)]["urban_area_km2"].values
            e = df[(df["city"] == city) & (df["year"] == end)]["urban_area_km2"].values
            if len(s) and len(e) and pd.notna(s[0]) and pd.notna(e[0]):
                total += float(e[0] - s[0])
        totals.append(total)

    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    fig.patch.set_facecolor(THEME["background"])
    ax.set_facecolor(THEME["card"])

    x = np.arange(len(periods))
    max_total = max(abs(t) for t in totals) if totals else 1
    if max_total == 0:
        max_total = 1

    bubble_colors = ["#D8B24C", "#E76F51", "#4C9BE8"]
    sizes = [(abs(t) / max_total) * 7000 + 1100 for t in totals]

    ax.scatter(
        x,
        [0] * len(x),
        s=sizes,
        c=bubble_colors,
        alpha=0.78,
        edgecolors="white",
        linewidths=2,
        zorder=3,
    )

    for i, ((start, end), total) in enumerate(zip(periods, totals)):
        ax.text(
            x[i], 0.04,
            f"{start}\N{EN DASH}{end}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=THEME["text"],
        )
        ax.text(
            x[i], -0.03,
            f"{total:+.0f} km²",
            ha="center",
            va="top",
            fontsize=10,
            color="white",
            fontweight="bold",
        )

    ax.set_title(
        "Decade Pulse (All Cities Combined)",
        fontsize=14,
        fontweight="bold",
        color=THEME["text"],
        pad=10,
    )

    ax.set_xlim(-0.65, len(x) - 0.35)
    ax.set_ylim(-0.45, 0.45)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return fig

def render_city_highlight_cards(df: pd.DataFrame):
    summary = get_city_summary_1990_2020(df)
    if summary.empty:
        return

    cols = st.columns(len(summary), gap="small")

    for col, (_, row) in zip(cols, summary.iterrows()):
        with col:
            st.html(f"""
<div style="
    background: linear-gradient(180deg, #FFFFFF 0%, #FBFAF7 100%);
    border: 1px solid {THEME['border']};
    border-radius: 20px;
    padding: 18px 16px;
    box-shadow: 0 8px 24px rgba(47, 59, 53, 0.06);
    min-height: 180px;
    position: relative;
    overflow: hidden;
">
    <div style="
        position:absolute;
        top:0;
        left:0;
        width:100%;
        height:6px;
        background:{row['color']};
    "></div>

    <div style="font-size:0.9em; color:#6F7E73; margin-top:6px;">
        {row['label']}
    </div>

    <div style="
        font-size:2.0em;
        font-weight:800;
        color:{row['color']};
        line-height:1.05;
        margin-top:8px;
    ">
        {row['area_2020']:.0f}
    </div>

    <div style="font-size:0.82em; color:#6F7E73;">
        km² in 2020
    </div>

    <div style="
        margin-top:14px;
        display:flex;
        justify-content:space-between;
        gap:10px;
        font-size:0.85em;
    ">
        <div>
            <div style="color:#6F7E73;">Since 1990</div>
            <div style="font-weight:700; color:#111827;">{row['growth_km2']:+.0f} km²</div>
        </div>
        <div>
            <div style="color:#6F7E73;">Percent</div>
            <div style="font-weight:700; color:#111827;">{row['growth_pct']:+.1f}%</div>
        </div>
    </div>
</div>
""")

## THIS DOWN IS FOR THE RVIERSIDE ONLY PAGE HELPERS
def get_subclass_year_means(df: pd.DataFrame, subclass: str, year: int):
    """Return mean spectral values for one subclass in one year."""
    features = ["red", "green", "blue", "nir", "ndvi"]
    sub_df = df[(df["subclass"] == subclass) & (df["year"] == year)].copy()

    if sub_df.empty:
        return None

    vals = sub_df[features].mean()
    if vals.isna().all():
        return None

    return vals


def get_subclass_means_by_year(df: pd.DataFrame, subclass: str):
    """Return mean spectral values by year for one subclass."""
    features = ["red", "green", "blue", "nir", "ndvi"]
    sub_df = df[df["subclass"] == subclass].copy()

    if sub_df.empty:
        return None

    mean_df = sub_df.groupby("year")[features].mean().reindex(YEARS)
    if mean_df.isna().all().all():
        return None

    return mean_df


def chart_riverside_subclass_year_heatmap(df: pd.DataFrame, subclass: str, year: int):
    """Mini 1x5 heatmap for one subclass in one year."""
    vals = get_subclass_year_means(df, subclass, year)
    if vals is None:
        return None

    features = ["red", "green", "blue", "nir", "ndvi"]
    matrix = vals[features].values.reshape(1, -1)

    fig, ax = plt.subplots(figsize=(5.8, 1.6))
    fig.patch.set_facecolor(THEME["background"])
    ax.set_facecolor(THEME["card"])

    cmap_name = SUBCLASS_PALETTES.get(subclass, "YlGnBu")
    cmap = plt.cm.get_cmap(cmap_name).copy()
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels([f.upper() for f in features], color=THEME["muted"], fontsize=9)
    ax.set_yticks([])

    _style_axis_title(ax, f"{year}")

    for j, val in enumerate(matrix[0]):
        ax.text(
            j, 0, f"{val:.3f}",
            ha="center", va="center",
            fontsize=CHART_STYLE["annot_size"],
            color=THEME["text"]
        )

    for spine in ax.spines.values():
        spine.set_edgecolor(THEME["border"])
        spine.set_linewidth(0.8)

    plt.tight_layout(pad=1.2)
    return fig

def chart_riverside_subclass_fingerprint_heatmap(df: pd.DataFrame, subclass: str):
    """Large summary heatmap with rows=years and cols=spectral features."""
    mean_df = get_subclass_means_by_year(df, subclass)
    if mean_df is None:
        return None

    features = ["red", "green", "blue", "nir", "ndvi"]
    matrix = mean_df[features].values

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    fig.patch.set_facecolor(THEME["background"])
    ax.set_facecolor(THEME["card"])

    cmap_name = SUBCLASS_PALETTES.get(subclass, "YlGnBu")
    cmap = plt.cm.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#F2F2F2")

    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, aspect="auto", cmap=cmap)

    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels([f.upper() for f in features], color=THEME["muted"], fontsize=10)
    ax.set_yticks(np.arange(len(YEARS)))
    ax.set_yticklabels([str(y) for y in YEARS], color=THEME["muted"], fontsize=10)

    _style_axis_title(ax, f"{subclass.replace('_', ' ').title()} Spectral Fingerprint")

    for i in range(masked.shape[0]):
        for j in range(masked.shape[1]):
            val = mean_df.iloc[i, j]
            if pd.notna(val):
                ax.text(
                    j, i, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=8.5,
                    color=THEME["text"]
                )

    for spine in ax.spines.values():
        spine.set_edgecolor(THEME["border"])
        spine.set_linewidth(0.8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.outline.set_edgecolor(THEME["border"])
    cbar.ax.tick_params(labelsize=8, colors=THEME["muted"])

    plt.tight_layout(pad=1.4)
    return fig

def chart_pixel_cloud_nir_red(df: pd.DataFrame):
    """Scatter plot of NIR vs RED for all subclasses."""
    fig, ax = _pretty_fig((7.8, 5.6))

    palette = {
        "urban": "#D97A66",
        "vegetation": "#7FB77E",
        "farmland": "#9BBF9B",
        "bare_soil": "#D8A66A",
        "water": "#6FA8DC",
    }

    subclasses = sorted(df["subclass"].dropna().unique().tolist())

    for subclass in subclasses:
        sub = df[df["subclass"] == subclass]
        ax.scatter(
            sub["red"],
            sub["nir"],
            s=20,
            alpha=0.35,
            label=subclass.replace("_", " ").title(),
            color=palette.get(subclass, THEME["secondary"]),
            edgecolors="none",
        )

    _style_axis_title(ax, "Pixel Cloud: NIR vs RED")
    _style_axis_labels(ax, "RED Reflectance", "NIR Reflectance")

    legend = ax.legend(
        frameon=True,
        fontsize=8,
        loc="best",
        borderpad=0.6,
        handletextpad=0.5,
    )
    legend.get_frame().set_facecolor(THEME["card"])
    legend.get_frame().set_edgecolor(THEME["border"])

    plt.tight_layout(pad=1.4)
    return fig

def chart_ndvi_distribution(df: pd.DataFrame):
    """Violin plot of NDVI by subclass."""
    fig, ax = _pretty_fig((8.0, 5.0))

    subclasses = sorted(df["subclass"].dropna().unique().tolist())
    data = [df[df["subclass"] == s]["ndvi"].dropna().values for s in subclasses]

    palette = {
        "urban": "#D97A66",
        "vegetation": "#7FB77E",
        "farmland": "#9BBF9B",
        "bare_soil": "#D8A66A",
        "water": "#6FA8DC",
    }

    parts = ax.violinplot(
        data,
        showmeans=True,
        showextrema=False,
        widths=0.78
    )

    for body, subclass in zip(parts["bodies"], subclasses):
        body.set_facecolor(palette.get(subclass, THEME["secondary"]))
        body.set_edgecolor(THEME["border"])
        body.set_alpha(0.72)

    if "cmeans" in parts:
        parts["cmeans"].set_color(THEME["text"])
        parts["cmeans"].set_linewidth(1.2)

    ax.set_xticks(range(1, len(subclasses) + 1))
    ax.set_xticklabels(
        [s.replace("_", " ").title() for s in subclasses],
        rotation=12,
        ha="right",
        fontsize=9
    )

    _style_axis_title(ax, "NDVI Distribution by Subclass")
    _style_axis_labels(ax, "", "NDVI")

    ax.grid(True, axis="y", alpha=0.10, color=THEME["border"])

    plt.tight_layout(pad=1.3)
    return fig

    parts = ax.violinplot(
        data,
        showmeans=True,
        showextrema=False,
        widths=0.78
    )

    for body, subclass in zip(parts["bodies"], subclasses):
        body.set_facecolor(palette.get(subclass, THEME["secondary"]))
        body.set_edgecolor(THEME["border"])
        body.set_alpha(0.72)

    if "cmeans" in parts:
        parts["cmeans"].set_color(THEME["text"])
        parts["cmeans"].set_linewidth(1.2)

    ax.set_xticks(range(1, len(subclasses) + 1))
    ax.set_xticklabels(
        [s.replace("_", " ").title() for s in subclasses],
        rotation=12,
        ha="right",
        fontsize=9
    )

    _style_axis_title(ax, "NDVI Distribution by Subclass")
    _style_axis_labels(ax, "", "NDVI")

    ax.grid(True, axis="y", alpha=0.10, color=THEME["border"])

    plt.tight_layout(pad=1.3)
    return fig

def chart_correlation_matrix(df: pd.DataFrame, subclass: str):
    """Correlation matrix for spectral bands within one subclass."""
    features = ["red", "green", "blue", "nir", "ndvi"]
    sub_df = df[df["subclass"] == subclass].copy()

    if sub_df.empty:
        return None

    corr = sub_df[features].corr()
    if corr.isna().all().all():
        return None

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    fig.patch.set_facecolor(THEME["background"])
    ax.set_facecolor(THEME["card"])

    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels([f.upper() for f in features], fontsize=9, color=THEME["muted"])
    ax.set_yticklabels([f.upper() for f in features], fontsize=9, color=THEME["muted"])

    _style_axis_title(ax, f"{subclass.replace('_', ' ').title()} Correlation Matrix")

    for i in range(len(features)):
        for j in range(len(features)):
            ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center",
                fontsize=8,
                color=THEME["text"]
            )

    for spine in ax.spines.values():
        spine.set_edgecolor(THEME["border"])
        spine.set_linewidth(0.8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_edgecolor(THEME["border"])
    cbar.ax.tick_params(labelsize=8, colors=THEME["muted"])

    plt.tight_layout(pad=1.3)
    return fig

# =========================================================
# PAGE: HOME
# =========================================================

def page_home():
    st.markdown("""
    <style>
    .hero-banner {
        background:
            linear-gradient(135deg, rgba(246,239,226,0.96) 0%, rgba(247,246,242,0.94) 45%, rgba(234,243,238,0.96) 100%);
        border-radius: 22px;
        padding: 34px 36px;
        margin-bottom: 24px;
        border: 1px solid #E4E2D9;
        box-shadow: 0 10px 28px rgba(47, 59, 53, 0.06);
    }

    .hero-kicker {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.7);
        border: 1px solid #DDD8CB;
        font-size: 0.78em;
        font-weight: 600;
        color: #4E5B53;
        margin-bottom: 14px;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.05;
        color: #000000;
        margin: 0 0 12px 0;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #2F3B35;
        margin: 0 0 18px 0;
        max-width: 620px;
    }

    .hero-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 12px;
    }

    .hero-chip {
        display: inline-block;
        padding: 8px 12px;
        border-radius: 999px;
        background: #FFFFFF;
        border: 1px solid #E4E2D9;
        font-size: 0.82em;
        color: #39453F;
        box-shadow: 0 2px 10px rgba(47, 59, 53, 0.04);
    }

    .hero-image-frame {
        background: rgba(255,255,255,0.58);
        border: 1px solid #E4E2D9;
        border-radius: 18px;
        padding: 10px;
        box-shadow: 0 8px 24px rgba(47, 59, 53, 0.05);
    }
    </style>
    """, unsafe_allow_html=True)

    hero_left, hero_right = st.columns([1.35, 0.95], gap="large")

    with hero_left:
        st.markdown("""
        <div class="hero-banner">
            <div class="hero-kicker">1990 → 2020 Urban Change Explorer</div>
            <div class="hero-title">Urban Expansion</div>
            <p class="hero-subtitle">
                Compare how Riverside, Phoenix, Las Vegas, and Austin changed over time
                using Landsat imagery, classification maps, and urban growth metrics.
            </p>
            <div class="hero-chip-row">
                <span class="hero-chip">4 Cities</span>
                <span class="hero-chip">1990 · 2000 · 2010 · 2020</span>
                <span class="hero-chip">Landsat 5 / 7 / 8</span>
                <span class="hero-chip">Google Earth Engine + Spark ML</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with hero_right:
        preview = classification_pil("las_vegas", 2020, width=700)
        if preview is not None:
            st.markdown('<div class="hero-image-frame">', unsafe_allow_html=True)
            st.image(preview, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="hero-banner" style="min-height: 260px; display:flex; align-items:center; justify-content:center;">
                <div style="color:#6F7E73; font-size:0.95em;">Map preview unavailable</div>
            </div>
            """, unsafe_allow_html=True)

    tab_summary, tab_pipeline = st.tabs(["Quick-Look Summary", "System Pipeline"])

    with tab_pipeline:
        step_cols = st.columns(5)
        pipeline_steps = [
            ("1", "Landsat Data", "Satellite imagery via Google Earth Engine"),
            ("2", "Preprocessing", "Cloud masking · median composites · band selection"),
            ("3", "Feature Extraction", "Red · Green · Blue · NIR · NDVI"),
            ("4", "Spark ML", "Random Forest classification at scale"),
            ("5", "Analytics", "Urban growth metrics & interactive visualizations"),
        ]
        for col, (icon, title, desc) in zip(step_cols, pipeline_steps):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size:1.8em; color:#000000; font-weight:700;">{icon}</div>
                    <div style="color:#000000; font-weight:600; margin:6px 0; font-size:0.95em;">{title}</div>
                    <div style="color:#000000; font-size:0.78em;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab_summary:
        df = load_metrics()
        if not df.empty:
            summary_cols = st.columns(4)
            for col, city in zip(summary_cols, CITIES):
                city_data = df[df["city"] == city].sort_values("year")
                a1990 = city_data.loc[city_data["year"] == 1990, "urban_area_km2"].values
                a2010 = city_data.loc[city_data["year"] == 2010, "urban_area_km2"].values
                with col:
                    if len(a1990) and len(a2010) and pd.notna(a1990[0]) and pd.notna(a2010[0]):
                        pct = round(((a2010[0] - a1990[0]) / a1990[0]) * 100, 1)
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="color:#000000; font-size:1.1em; font-weight:700;">
                                {CITY_LABELS[city]}
                            </div>
                            <div class="metric-value" style="color:#000000;">{pct:+.0f}%</div>
                            <div class="metric-label">urban growth (1990–2010)</div>
                            <div style="color:#000000; font-size:0.78em; margin-top:6px;">
                                {a2010[0]:.0f} km² in 2010
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="color:#000000; font-weight:700;">{CITY_LABELS[city]}</div>
                            <div style="color:#000000; font-size:0.85em; margin-top:8px;">Data unavailable —<br>run Member 4 script first</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info(
                "Metrics data not found. Run `python scripts/member4/urban_growth_metrics.py` "
                "from the project root to generate `data/urban_growth_metrics.csv`."
            )


# =========================================================
# PAGE: CLASSIFICATION MAPS
# =========================================================

def page_classification_maps():
    st.title("Main Map")
    st.markdown("Large central map showing the classification raster with urban areas highlighted.")

    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        selected_city = st.selectbox(
            "City", CITIES, format_func=lambda c: CITY_LABELS[c], key="map_city"
        )
    with sel_col2:
        selected_year = st.selectbox("Year", YEARS, key="map_year")

    map_col, info_col = st.columns([3, 1])

    with map_col:
        img = classification_pil(selected_city, selected_year, width=900)
        if img is not None:
            st.image(
                img,
                caption=f"{CITY_LABELS[selected_city]} {selected_year}  —  Classification Map",
                use_container_width=True,
            )
        else:
            st.warning(
                f"Classification map not found for {CITY_LABELS[selected_city]} {selected_year}.\n\n"
                "Run `scripts/member3/randomForest_toComposites.py` to generate it."
            )

    with info_col:
        df = load_metrics()
        row = df[(df["city"] == selected_city) & (df["year"] == selected_year)]

        st.markdown("### Metrics")
        if not row.empty and pd.notna(row.iloc[0]["urban_area_km2"]):
            r    = row.iloc[0]
            area = r["urban_area_km2"]
            gpct = r["growth_pct"]

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Urban Area</div>
                <div class="metric-value">{area:.1f} km²</div>
            </div>
            """, unsafe_allow_html=True)

            if pd.notna(gpct):
                arrow = "▲" if gpct > 0 else "▼"
                clr   = THEME["primary"] if gpct > 0 else THEME["accent"]
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Growth from previous decade</div>
                    <div class="metric-value" style="color:{clr};">{arrow} {abs(gpct):.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="info-box">Baseline year — no prior decade to compare.</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="info-box">Run the metrics script to see data here.</div>',
                unsafe_allow_html=True,
            )

# =========================================================
# PAGE: SIDE-BY-SIDE COMPARISON
# =========================================================

def page_side_by_side():
    st.title("Side-by-Side Comparison")
    st.markdown("Directly compare classification maps across years or cities.")

    tab_years, tab_cities = st.tabs(["Compare Years — Same City", "Compare Cities — Same Year"])

    # ── Tab 1: same city, different years ────────────────────────────────────
    with tab_years:
        city_sbs = st.selectbox(
            "City", CITIES, format_func=lambda c: CITY_LABELS[c], key="sbs_city"
        )
        yr_col1, yr_col2 = st.columns(2)
        with yr_col1:
            year_a = st.selectbox("Year A", YEARS, index=0, key="sbs_yr_a")
        with yr_col2:
            year_b = st.selectbox("Year B", YEARS, index=3, key="sbs_yr_b")

        map_a_col, map_b_col = st.columns(2)
        img_a = classification_pil(city_sbs, year_a, width=520)
        img_b = classification_pil(city_sbs, year_b, width=520)

        with map_a_col:
            if img_a:
                st.image(img_a, caption=f"{CITY_LABELS[city_sbs]} {year_a}", use_container_width=True)
            else:
                st.warning("Map not available.")
        with map_b_col:
            if img_b:
                st.image(img_b, caption=f"{CITY_LABELS[city_sbs]} {year_b}", use_container_width=True)
            else:
                st.warning("Map not available.")

        # Delta metrics below the maps
        df = load_metrics()
        val_a = df[(df["city"] == city_sbs) & (df["year"] == year_a)]["urban_area_km2"].values
        val_b = df[(df["city"] == city_sbs) & (df["year"] == year_b)]["urban_area_km2"].values

        if len(val_a) and len(val_b) and pd.notna(val_a[0]) and pd.notna(val_b[0]):
            delta    = val_b[0] - val_a[0]
            delta_pct = (delta / val_a[0]) * 100 if val_a[0] > 0 else 0
            m1, m2, m3 = st.columns(3)
            m1.metric(f"Urban Area — {year_a}", f"{val_a[0]:.1f} km²")
            m2.metric(f"Urban Area — {year_b}", f"{val_b[0]:.1f} km²",
                      delta=f"{delta:+.1f} km²")
            m3.metric("% Change", f"{delta_pct:+.1f}%")

    # ── Tab 2: same year, different cities ────────────────────────────────────
    with tab_cities:
        year_sbs = st.selectbox("Year", YEARS, index=3, key="sbs_year_only")
        ct_col1, ct_col2 = st.columns(2)
        with ct_col1:
            city_a = st.selectbox(
                "City A", CITIES, index=0,
                format_func=lambda c: CITY_LABELS[c], key="sbs_city_a"
            )
        with ct_col2:
            city_b = st.selectbox(
                "City B", CITIES, index=1,
                format_func=lambda c: CITY_LABELS[c], key="sbs_city_b"
            )

        ca_col, cb_col = st.columns(2)
        img_ca = classification_pil(city_a, year_sbs, width=520)
        img_cb = classification_pil(city_b, year_sbs, width=520)

        with ca_col:
            if img_ca:
                st.image(img_ca, caption=f"{CITY_LABELS[city_a]} {year_sbs}", use_container_width=True)
            else:
                st.warning("Map not available.")
        with cb_col:
            if img_cb:
                st.image(img_cb, caption=f"{CITY_LABELS[city_b]} {year_sbs}", use_container_width=True)
            else:
                st.warning("Map not available.")

        # Side-by-side urban area metrics for the two cities
        df = load_metrics()
        va = df[(df["city"] == city_a) & (df["year"] == year_sbs)]["urban_area_km2"].values
        vb = df[(df["city"] == city_b) & (df["year"] == year_sbs)]["urban_area_km2"].values
        if (len(va) and len(vb) and pd.notna(va[0]) and pd.notna(vb[0])):
            diff  = va[0] - vb[0]
            m1, m2, m3 = st.columns(3)
            m1.metric(CITY_LABELS[city_a],      f"{va[0]:.1f} km²")
            m2.metric(CITY_LABELS[city_b],      f"{vb[0]:.1f} km²")
            m3.metric("Difference",             f"{abs(diff):.1f} km²",
                      delta=f"{diff:+.1f} km² ({CITY_LABELS[city_a]} vs {CITY_LABELS[city_b]})")


# =========================================================
# PAGE: ANIMATED TIME-LAPSE
# =========================================================

def page_timelapse():
    st.title("Animated Time-Lapse")
    st.markdown(
        "Watch urban areas expand from **1990** to **2020**. "
        "Use the animated GIF tab for a looping animation, or "
        "drag the slider to step through each decade manually."
    )

    # City toggle — supports the "toggle between cities" requirement
    tl_city = st.selectbox(
        "Select City", CITIES, format_func=lambda c: CITY_LABELS[c], key="tl_city"
    )

    tab_gif, tab_slider = st.tabs(["Animated GIF", "Step Through Manually"])

    with tab_gif:
        with st.spinner(f"Generating time-lapse for {CITY_LABELS[tl_city]}…"):
            gif_bytes = generate_timelapse_gif(tl_city)

        if gif_bytes:
            # Centre the GIF
            _, gif_col, _ = st.columns([1, 2, 1])
            with gif_col:
                st.image(
                    gif_bytes,
                    caption=f"{CITY_LABELS[tl_city]} — Urban Growth 1990–2020",
                    use_container_width=True,
                )
            st.download_button(
                label="Download GIF",
                data=gif_bytes,
                file_name=f"{tl_city}_urban_timelapse.gif",
                mime="image/gif",
            )
        else:
            st.warning("Could not generate time-lapse — classification files may be missing.")

    with tab_slider:
        year = st.select_slider("Decade", options=YEARS, value=1990)
        st.caption(f"**{CITY_LABELS[tl_city]}  ·  {year}**")

        _, slider_col, _ = st.columns([1, 2, 1])
        with slider_col:
            slide_img = classification_pil(tl_city, year, width=500)
            if slide_img:
                st.image(slide_img, use_container_width=True)
            else:
                st.warning("Map not found for this city / year.")

        st.markdown("---")
        tl_cols = st.columns(4)
        for tc, y in zip(tl_cols, YEARS):
            indicator = "Current" if y == year else "-"
            tc.markdown(
                f"<div style='text-align:center; color:#6F7E73;'>"
                f"{indicator}<br><small>{y}</small></div>",
            unsafe_allow_html=True,
        )

# =========================================================
# PAGE: URBAN AREA vs TIME  (with city toggles)
# =========================================================

def page_urban_growth_trends():
    st.title("Urban Growth Trends")

    df = load_metrics()
    if df.empty:
        st.warning(
            "No metrics data found. "
            "Run `python scripts/member4/urban_growth_metrics.py` first."
        )
        return

    st.markdown("### Select Cities")
    toggle_cols = st.columns(4, gap="small")
    city_toggles = {}
    for i, city in enumerate(CITIES):
        with toggle_cols[i]:
            city_toggles[city] = st.checkbox(
                CITY_LABELS[city],
                value=True,
                key=f"toggle_{city}"
            )

    active = [c for c, on in city_toggles.items() if on]
    if not active:
        st.warning("Select at least one city to display.")
        return

    summary_df = get_growth_summary(df, active)

    st.markdown("### Snapshot")
    if not summary_df.empty:
        render_growth_snapshot_cards(summary_df)

    st.markdown("### Visual Summary")
    left, right = st.columns([1.15, 1], gap="large")

    with left:
        fig = chart_growth_slopegraph(df, active)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        fig = chart_city_year_heatmap(df, active)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown("### Decade Pulse")
    render_decade_pulse_cards(df, active)

    with st.expander("Show detailed growth table"):
        display_cols = ["city", "year", "urban_area_km2", "growth_km2", "growth_pct_display"]
        available_cols = [c for c in display_cols if c in df.columns]
        table_df = df[df["city"].isin(active)][available_cols].copy()
        table_df["city"] = table_df["city"].map(CITY_LABELS)
        table_df.columns = [
            c.replace("_km2", " (km²)").replace("_pct_display", " %")
             .replace("_", " ").title()
            for c in table_df.columns
        ]
        st.dataframe(table_df, use_container_width=True, height=260)


# =========================================================
# PAGE: RIVERSIDE LANDCOVER TRENDS
# =========================================================

def page_riverside_landcover():
    st.title("Riverside Spectral Analysis")

    df = load_riverside_training_samples()
    if df.empty:
        st.warning("No Riverside training samples available.")
        return

    # ----------------------------------------------------
    # GLOBAL SECTION
    # ----------------------------------------------------
    st.markdown("## Overall Comparison")

    global_tab1, global_tab2 = st.tabs(["Pixel Cloud (NIR vs RED)", "NDVI Distribution"])

    with global_tab1:
        fig = chart_pixel_cloud_nir_red(df)
        if fig is not None:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No pixel cloud data available.")

    with global_tab2:
        fig = chart_ndvi_distribution(df)
        if fig is not None:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No NDVI distribution data available.")

    st.markdown("---")

    # ----------------------------------------------------
    # SUBCLASS TABS
    # ----------------------------------------------------
    st.markdown("## Subclass Detail")

    preferred_order = ["urban", "vegetation", "farmland", "bare_soil", "water"]
    available = sorted(df["subclass"].dropna().unique().tolist())
    subclass_order = [s for s in preferred_order if s in available]
    subclass_order += [s for s in available if s not in subclass_order]

    subclass_tabs = st.tabs([s.replace("_", " ").title() for s in subclass_order])

    years = [1990, 2000, 2010, 2020]

    for tab, subclass in zip(subclass_tabs, subclass_order):
        with tab:
            st.subheader(subclass.replace("_", " ").title())

            sub_tab1, sub_tab2, sub_tab3 = st.tabs(
                ["Mini Heatmaps", "Fingerprint Heatmap", "Correlation Matrix"]
            )

            with sub_tab1:
                col1, col2 = st.columns(2)
                for i, year in enumerate(years):
                    target_col = col1 if i % 2 == 0 else col2
                    with target_col:
                        fig = chart_riverside_subclass_year_heatmap(df, subclass, year)
                        if fig is not None:
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.info(f"No data for {subclass.replace('_', ' ')} in {year}.")

            with sub_tab2:
                fig = chart_riverside_subclass_fingerprint_heatmap(df, subclass)
                if fig is not None:
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info(f"No fingerprint heatmap available for {subclass.replace('_', ' ')}.")

            with sub_tab3:
                fig = chart_correlation_matrix(df, subclass)
                if fig is not None:
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info(f"No correlation matrix available for {subclass.replace('_', ' ')}.")


# =========================================================
# PAGE: CITY COMPARISON
# =========================================================
def page_city_comparison():
    st.title("City Comparison")
    st.markdown("Compare urban growth patterns across all four cities side by side.")

    df = load_metrics()
    if df.empty:
        st.warning(
            "No metrics data found. "
            "Run `python scripts/member4/urban_growth_metrics.py` first."
        )
        return

    # top row
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("##### Total Growth 1990–2020")
        fig = chart_city_comparison(df)   # keep the one you like
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown("##### Urban Footprint Bubble Matrix")
        fig = chart_bubble_matrix(df)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("### City Growth Profiles")
    render_city_highlight_cards(df)

    st.markdown("### Growth Dynamics")
    left, right = st.columns(2)

    with left:
        st.markdown("##### 1990 vs 2020 Footprint Shift")
        fig = chart_nested_area_circles(df)
        if fig is not None:
            st.pyplot(fig)
            plt.close(fig)

    with right:
        st.markdown("##### Decade Pulse")
        fig = chart_decade_pulse_bubbles(df)
        if fig is not None:
            st.pyplot(fig)
            plt.close(fig)

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

PAGES = {
    "Home":                    page_home,
    "Riverside Information":     page_riverside_landcover,
    "Main Map":                page_classification_maps,
    "Side-by-Side":            page_side_by_side,
    "Urban Growth Trends":     page_urban_growth_trends,
    "Time-Lapse":              page_timelapse,
    "City Comparison":         page_city_comparison,
}

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:14px 0 6px 0;">
        <div style="color:#000000; font-weight:700; font-size:1.05em;">Urban Expansion</div>
        <div style="color:#000000; font-size:0.78em; margin-top:2px;">CS 224 Final Project</div>
    </div>
    <hr style="border-color:#E4E2D9; margin:10px 0 14px 0;">
    """, unsafe_allow_html=True)

    page_name = st.radio(
        "Navigate", list(PAGES.keys()), label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#E4E2D9; margin:14px 0 10px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="color:#000000; font-size:0.75em; line-height:1.8;">
        <strong style="color:#000000;">Data</strong><br>
        Landsat 5 / 7 / 8<br>
        Google Earth Engine<br><br>
        <strong style="color:#000000;">Processing</strong><br>
        Apache Spark · Random Forest<br><br>
        <strong style="color:#000000;">Cities</strong><br>
        Riverside · Phoenix<br>
        Las Vegas · Austin<br><br>
        <strong style="color:#000000;">Time Range</strong><br>
        1990 · 2000 · 2010 · 2020
    </div>
    """, unsafe_allow_html=True)

# Render the selected page
PAGES[page_name]()
