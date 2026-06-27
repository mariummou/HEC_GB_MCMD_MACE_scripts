#!/usr/bin/env python3
import os
import re
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from ase.io import read

# ============================================================
# USER CONTROLS
# ============================================================

# ---------- binning ----------
EQUAL_COUNT = True
BIN_WIDTH = 1.0
N_BINS = None

# ---------- save options ----------
SAVE_PDF = True
SAVE_PNG = True

# ---------- selectivity settings ----------
# Choose one:
# "first_n_bins"    -> use first N bins closest to GB
# "distance_cutoff" -> use bins with center < GB_CUTOFF_A
GB_MODE = "first_n_bins"

FIRST_N_BINS = 2
GB_CUTOFF_A = 2.0

# Bulk region for C_bulk
# If None, use farthest BULK_FRACTION_START of profile
BULK_START_A = None
BULK_FRACTION_START = 0.75

# If True, use nominal bulk composition (20 at.% for equimolar 5-cation HEC)
USE_NOMINAL_BULK = False
NOMINAL_BULK_ATPCT = 20.0

# ---------- species ----------
CANONICAL = ["Cr", "Hf", "Mo", "Nb", "Ta", "Ti", "V", "W", "Zr"]
METAL_SET = set(CANONICAL)

FIXED_COLORS = {
    "Hf": "#000000",  # black
    "Mo": "#E67E22",  # orange
    "Nb": "#8C8C8C",  # gray
    "Ta": "#F0E71D",  # yellow
    "Ti": "#7B4AB3",  # purple
    "V":  "#5AA469",  # green
    "W":  "#A85A22",  # dark brown/orange
    "Zr": "#4AA3D8",  # blue
    "Cr": "#C95A5A",  # red
}

# ---------- plot style ----------
BASE_FONT = 9
AXIS_LABEL_FONT = 18
TICK_FONT = 13
TITLE_FONT = 20

# Make species part bigger here
LEGEND_FONT = 16
LEGEND_TITLE_FONT = 22

FIGSIZE = (5.3, 4.2)
DPI = 600
FONT_NAME = "Liberation Sans"

# ============================================================


def setup_plot_style():
    font_prop = font_manager.FontProperties(family=FONT_NAME)

    try:
        font_path = font_manager.findfont(font_prop, fallback_to_default=False)
    except Exception:
        sys.exit(
            f"\n[ERROR] {FONT_NAME} was not found on this system.\n"
            "Run: fc-match \"Liberation Sans\"\n"
        )

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_NAME],
        "font.size": BASE_FONT,
        "axes.labelsize": AXIS_LABEL_FONT,
        "axes.titlesize": TITLE_FONT,
        "xtick.labelsize": TICK_FONT,
        "ytick.labelsize": TICK_FONT,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.major.size": 5.0,
        "ytick.major.size": 5.0,
        "axes.unicode_minus": False,
        "legend.frameon": False,
    })

    print(f"[INFO] Using font: {FONT_NAME}")
    print(f"[INFO] Font file: {font_path}")


def dist_to_plane(x, p, a):
    d = np.abs(x - p)
    return np.minimum(d, a - d)


def compute_bins_equal_count(all_dists, xmax, approx_bin_width, n_bins_override=None):
    all_dists = np.asarray(all_dists, dtype=float)
    all_dists = all_dists[(all_dists >= 0.0) & (all_dists <= xmax)]

    if all_dists.size == 0:
        return np.array([0.0, xmax]), np.array([0.5 * xmax])

    if n_bins_override is not None:
        nb = int(max(1, n_bins_override))
    else:
        nb = max(1, int(np.ceil(xmax / max(approx_bin_width, 1e-12))))

    qs = np.linspace(0.0, 1.0, nb + 1)

    try:
        edges = np.quantile(all_dists, qs, method="linear")
    except TypeError:
        edges = np.quantile(all_dists, qs, interpolation="linear")

    edges[0] = 0.0
    edges[-1] = xmax

    unique_edges = [edges[0]]
    for v in edges[1:]:
        if v - unique_edges[-1] > 1e-10:
            unique_edges.append(v)

    edges = np.array(unique_edges, dtype=float)

    if edges.size < 2:
        edges = np.array([0.0, xmax], dtype=float)

    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def assign_bins(d, bins):
    nb = len(bins) - 1
    idx = np.digitize(d, bins, right=False) - 1
    idx[np.isclose(d, bins[-1])] = nb - 1
    valid = (idx >= 0) & (idx < nb)
    return idx, valid


def extract_temperature_tag(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r"^\s*(\d+(?:\.\d+)?)", stem)

    if not m:
        return ""

    temp = m.group(1)
    if temp.endswith(".0"):
        temp = temp[:-2]

    return f"{temp} K"


def extract_temperature_value(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r"^\s*(\d+(?:\.\d+)?)", stem)
    if not m:
        return np.nan
    return float(m.group(1))


def weighted_region_atpct(df, species, mask):
    raw_col = f"avg_raw_count_{species}"

    if raw_col in df.columns and "avg_total_metal_count" in df.columns:
        numerator = df.loc[mask, raw_col].sum()
        denominator = df.loc[mask, "avg_total_metal_count"].sum()
        if denominator > 0:
            return 100.0 * numerator / denominator
        return np.nan

    pct_col = f"at_percent_{species}"
    if pct_col in df.columns:
        weights = df.loc[mask, "avg_total_metal_count"].values
        vals = df.loc[mask, pct_col].values
        if np.sum(weights) > 0:
            return np.average(vals, weights=weights)
        return np.nan

    return np.nan


def get_gb_mask(df):
    if GB_MODE == "first_n_bins":
        mask = pd.Series(False, index=df.index)
        n = min(FIRST_N_BINS, len(df))
        mask.iloc[:n] = True
        description = f"first_{n}_bins"

    elif GB_MODE == "distance_cutoff":
        mask = df["dist_to_nearest_GB_A"] < GB_CUTOFF_A
        if mask.sum() == 0:
            mask = pd.Series(False, index=df.index)
            mask.iloc[0] = True
        description = f"distance_less_than_{GB_CUTOFF_A:g}A"

    else:
        raise ValueError("GB_MODE must be 'first_n_bins' or 'distance_cutoff'")

    return mask, description


def analyze_selectivity_from_df(df, csv_name, sp_order):
    gb_mask, gb_description = get_gb_mask(df)

    selected_bins = df.loc[gb_mask].copy()

    if "bin_left_A" in df.columns and "bin_right_A" in df.columns:
        gb_distance_min = selected_bins["bin_left_A"].min()
        gb_distance_max = selected_bins["bin_right_A"].max()
    else:
        gb_distance_min = selected_bins["dist_to_nearest_GB_A"].min()
        gb_distance_max = selected_bins["dist_to_nearest_GB_A"].max()

    dmax = df["dist_to_nearest_GB_A"].max()

    if BULK_START_A is None:
        bulk_start = BULK_FRACTION_START * dmax
    else:
        bulk_start = BULK_START_A

    bulk_mask = df["dist_to_nearest_GB_A"] >= bulk_start
    if bulk_mask.sum() == 0:
        bulk_mask = pd.Series(False, index=df.index)
        bulk_mask.iloc[-1] = True

    gb_comp = {}
    bulk_comp = {}

    for s in sp_order:
        gb_comp[s] = weighted_region_atpct(df, s, gb_mask)
        if USE_NOMINAL_BULK:
            bulk_comp[s] = NOMINAL_BULK_ATPCT
        else:
            bulk_comp[s] = weighted_region_atpct(df, s, bulk_mask)

    ranked = sorted(gb_comp.items(), key=lambda x: x[1], reverse=True)

    if len(ranked) < 2:
        return None, []

    top_species, cmax_gb = ranked[0]
    second_species, csecond_gb = ranked[1]
    cbulk_top = bulk_comp[top_species]

    s_vs_bulk = cmax_gb - cbulk_top
    s_selectivity = cmax_gb - csecond_gb

    if s_selectivity >= 15:
        regime = "strongly selective"
    elif s_selectivity >= 7:
        regime = "moderately selective"
    else:
        regime = "multi-element/co-segregation"

    summary = {
        "file": csv_name,
        "temperature_K": extract_temperature_value(csv_name),
        "gb_region_used": gb_description,
        "gb_distance_min_A": gb_distance_min,
        "gb_distance_max_A": gb_distance_max,
        "bulk_start_A": bulk_start,
        "top_GB_species": top_species,
        "second_GB_species": second_species,
        "Cmax_GB_atpct": cmax_gb,
        "Csecond_GB_atpct": csecond_gb,
        "Cbulk_top_species_atpct": cbulk_top,
        "S_vs_bulk_atpct": s_vs_bulk,
        "S_selectivity_atpct": s_selectivity,
        "regime_label": regime,
    }

    long_rows = []
    for rank, (s, cgb) in enumerate(ranked, start=1):
        long_rows.append({
            "file": csv_name,
            "temperature_K": extract_temperature_value(csv_name),
            "gb_region_used": gb_description,
            "rank_in_GB": rank,
            "species": s,
            "C_GB_atpct": cgb,
            "C_bulk_atpct": bulk_comp[s],
            "enrichment_GB_minus_bulk_atpct": cgb - bulk_comp[s],
        })

    return summary, long_rows


def process(path):
    frames = read(path, index=":")

    if len(frames) == 0:
        print(f"[WARN] No frames in {path}. Skipping.")
        return None, []

    a = float(frames[0].cell[0, 0])

    gb0 = 0.0
    gbmid = 0.5 * a
    xmax = 0.25 * a

    present = set()
    for fr in frames:
        present.update(fr.get_chemical_symbols())

    sp_order = [s for s in CANONICAL if s in present and s in METAL_SET]

    if not sp_order:
        print(f"[WARN] No target metal atoms found in {path}. Skipping.")
        return None, []

    # ------------------------------------------------------------
    # Build distance bins
    # ------------------------------------------------------------
    if EQUAL_COUNT:
        all_dists = []

        for fr in frames:
            x = fr.positions[:, 0]
            syms = np.array(fr.get_chemical_symbols())
            metal_mask = np.isin(syms, list(METAL_SET))

            if not np.any(metal_mask):
                continue

            d0 = dist_to_plane(x[metal_mask], gb0, a)
            dmid = dist_to_plane(x[metal_mask], gbmid, a)
            d = np.minimum(d0, dmid)
            all_dists.append(d)

        all_dists = np.concatenate(all_dists) if all_dists else np.array([])
        bins, centers = compute_bins_equal_count(all_dists, xmax, BIN_WIDTH, N_BINS)

    else:
        bins = np.arange(0.0, xmax + BIN_WIDTH + 1e-12, BIN_WIDTH)
        if bins[-1] < xmax:
            bins = np.append(bins, xmax)
        centers = 0.5 * (bins[:-1] + bins[1:])

    nb = len(bins) - 1

    # ------------------------------------------------------------
    # Count metal atoms per bin
    # ------------------------------------------------------------
    raw_counts = {s: np.zeros(nb, dtype=float) for s in sp_order}
    total_counts = np.zeros(nb, dtype=float)

    for fr in frames:
        x = fr.positions[:, 0]
        syms = np.array(fr.get_chemical_symbols())

        metal_mask = np.isin(syms, list(METAL_SET))
        x = x[metal_mask]
        syms = syms[metal_mask]

        if x.size == 0:
            continue

        d0 = dist_to_plane(x, gb0, a)
        dmid = dist_to_plane(x, gbmid, a)
        d = np.minimum(d0, dmid)

        idx, valid = assign_bins(d, bins)

        for el, k in zip(syms[valid], idx[valid]):
            if el in raw_counts:
                raw_counts[el][k] += 1.0
                total_counts[k] += 1.0

    nframes = max(len(frames), 1)
    avg_counts = {s: raw_counts[s] / nframes for s in sp_order}
    avg_total = total_counts / nframes

    percent_counts = {}
    for s in sp_order:
        percent_counts[s] = np.where(
            avg_total > 0,
            100.0 * avg_counts[s] / avg_total,
            0.0
        )

    # ------------------------------------------------------------
    # Save per-bin CSV
    # ------------------------------------------------------------
    df = pd.DataFrame({
        "dist_to_nearest_GB_A": centers,
        "bin_left_A": bins[:-1],
        "bin_right_A": bins[1:],
        "bin_width_A": np.diff(bins),
        "avg_total_metal_count": avg_total,
    })

    for s in sp_order:
        df[f"avg_raw_count_{s}"] = avg_counts[s]
        df[f"at_percent_{s}"] = percent_counts[s]

    stem = os.path.splitext(os.path.basename(path))[0]
    suffix = "_equalcount" if EQUAL_COUNT else f"_bin{BIN_WIDTH:g}A"

    csv_out = f"{stem}_gb_metals_normalized{suffix}.csv"
    png_out = f"{stem}_gb_metals_normalized{suffix}.png"
    pdf_out = f"{stem}_gb_metals_normalized{suffix}.pdf"

    df.to_csv(csv_out, index=False)

    # ------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    lefts = bins[:-1]
    widths = np.diff(bins)
    bottom = np.zeros(nb, dtype=float)

    for s in sp_order:
        vals = percent_counts[s]
        ax.bar(
            lefts,
            vals,
            width=widths * 0.98,
            align="edge",
            bottom=bottom,
            label=s,
            color=FIXED_COLORS.get(s, "#B0B0B0"),
            edgecolor="none",
            linewidth=0.0,
        )
        bottom += vals

    for e in bins:
        ax.axvline(e, ls=":", lw=0.35, alpha=0.25, color="k", zorder=0)

    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, 100.0)

    ax.set_xlabel("Distance to nearest GB (Å)", labelpad=8)
    ax.set_ylabel("Metal content (at.%)", labelpad=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", pad=8)

    caption = extract_temperature_tag(path)

    handles, labels = ax.get_legend_handles_labels()

    leg = fig.legend(
        handles,
        labels,
        title=(caption if caption else None),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(sp_order),
        frameon=False,
        columnspacing=0.8,
        handlelength=1.2,
        handleheight=1.2,
        handletextpad=0.35,
        borderaxespad=0.0,
        prop={"size": LEGEND_FONT},
    )

    if caption:
        leg.get_title().set_fontweight("bold")
        leg.get_title().set_fontsize(LEGEND_TITLE_FONT)

    for text in fig.findobj(match=plt.Text):
        text.set_fontfamily(FONT_NAME)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.83])

    if SAVE_PNG:
        plt.savefig(png_out, dpi=DPI, bbox_inches="tight")
    if SAVE_PDF:
        plt.savefig(pdf_out, bbox_inches="tight")

    plt.close(fig)

    # ------------------------------------------------------------
    # Selectivity analysis
    # ------------------------------------------------------------
    summary, long_rows = analyze_selectivity_from_df(df, csv_out, sp_order)

    print(f"[OK] {path}")
    print(f"     Species order: {' '.join(sp_order)}")
    print(f"     CSV: {csv_out}")
    if SAVE_PNG:
        print(f"     PNG: {png_out}")
    if SAVE_PDF:
        print(f"     PDF: {pdf_out}")

    if summary is not None:
        print(f"     Top GB species: {summary['top_GB_species']}")
        print(f"     Second GB species: {summary['second_GB_species']}")
        print(f"     S_selectivity: {summary['S_selectivity_atpct']:.2f}")
        print(f"     S_vs_bulk: {summary['S_vs_bulk_atpct']:.2f}")

    return summary, long_rows


def main():
    setup_plot_style()

    files = sorted([f for f in os.listdir(".") if f.endswith(".xyz")])

    if not files:
        print("[WARN] No .xyz files found in the current folder.")
        return

    all_summaries = []
    all_long_rows = []

    for f in files:
        summary, long_rows = process(f)
        if summary is not None:
            all_summaries.append(summary)
            all_long_rows.extend(long_rows)

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        long_df = pd.DataFrame(all_long_rows)

        summary_df.to_csv("gb_top_segregant_selectivity_summary.csv", index=False)
        long_df.to_csv("gb_all_species_ranked_enrichment.csv", index=False)

        print("\n[OK] Wrote: gb_top_segregant_selectivity_summary.csv")
        print("[OK] Wrote: gb_all_species_ranked_enrichment.csv")

        print("\nTop GB segregant summary:")
        print(summary_df[
            [
                "file",
                "temperature_K",
                "gb_region_used",
                "top_GB_species",
                "second_GB_species",
                "Cmax_GB_atpct",
                "Csecond_GB_atpct",
                "S_selectivity_atpct",
                "S_vs_bulk_atpct",
                "regime_label",
            ]
        ].to_string(index=False))


if __name__ == "__main__":
    main()
