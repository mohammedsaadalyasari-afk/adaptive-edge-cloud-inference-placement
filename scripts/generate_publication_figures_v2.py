#!/usr/bin/env python3

from pathlib import Path
import hashlib
import math
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


ANALYSIS = Path(sys.argv[1])
OUT = Path(sys.argv[2])
SOURCE = OUT / "source_data"

OUT.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)


# ============================================================
# PUBLICATION STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.0,
    "legend.fontsize": 8.3,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,

    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "lines.markersize": 4.5,

    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,

    "grid.linewidth": 0.55,
    "grid.alpha": 0.24,

    "figure.dpi": 150,
    "savefig.dpi": 600,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    "axes.spines.top": False,
    "axes.spines.right": False,
})


# Accessible palette + distinct markers/styles.
STYLE = {
    "S5": {
        "color": "#0072B2",
        "marker": "o",
        "linestyle": "-",
        "label": "S5 — Edge only",
    },

    "S6": {
        "color": "#D55E00",
        "marker": "s",
        "linestyle": "--",
        "label": "S6 — Cloud only",
    },

    "S7": {
        "color": "#009E73",
        "marker": "^",
        "linestyle": "-.",
        "label": "S7 — Adaptive",
    },
}


PHASES = [
    ("L0", "Baseline", 1, 40, 0),
    ("L1", "Stable", 41, 80, 4),
    ("L2", "Onset", 81, 120, 6),
    ("L3", "Saturation", 121, 170, 8),
    ("L4", "Recovery", 171, 210, 4),
    ("L5", "Re-overload", 211, 250, 8),
    ("L6", "Final recovery", 251, 289, 0),
]

PHASE_ORDER = [
    "L0_BASELINE",
    "L1_STABLE",
    "L2_ONSET",
    "L3_SATURATION",
    "L4_RECOVERY",
    "L5_REOVERLOAD",
    "L6_FINAL_RECOVERY",
]

PHASE_SHORT = [
    "L0",
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L6",
]

B = 50000
SEED = 20261010


# ============================================================
# INPUT DATA
# ============================================================

master = pd.read_csv(
    ANALYSIS / "01_master_decisions.csv"
)

run = pd.read_csv(
    ANALYSIS / "03_run_level_metrics.csv"
)

phase = pd.read_csv(
    ANALYSIS / "04_phase_run_level_metrics.csv"
)

paired = pd.read_csv(
    ANALYSIS / "09_paired_block_values.csv"
)

inference = pd.read_csv(
    ANALYSIS / "10_paired_runlevel_inference.csv"
)

s7_run = pd.read_csv(
    ANALYSIS / "12_S7_behavior_by_run.csv"
)

s7_summary = pd.read_csv(
    ANALYSIS / "13_S7_behavior_summary.csv"
)

s7_phase = pd.read_csv(
    ANALYSIS / "14_S7_phase_behavior_summary.csv"
)


# ============================================================
# BASIC VALIDATION
# ============================================================

assert len(master) == 8670
assert len(run) == 30

for s in ["S5", "S6", "S7"]:
    assert (master["strategy"] == s).sum() == 2890
    assert (run["strategy"] == s).sum() == 10


# ============================================================
# HELPERS
# ============================================================

def save_all(fig, stem):
    """Save publication-ready vector + raster versions."""

    for ext in ["pdf", "svg", "eps"]:
        fig.savefig(
            OUT / f"{stem}.{ext}",
            bbox_inches="tight",
        )

    fig.savefig(
        OUT / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)


def phase_boundaries(ax, labels=True):

    boundaries = [40.5, 80.5, 120.5, 170.5, 210.5, 250.5]

    for x in boundaries:
        ax.axvline(
            x,
            color="0.55",
            linestyle=":",
            linewidth=0.9,
            zorder=0,
        )

    if labels:
        ymax = ax.get_ylim()[1]

        for short, name, start, end, workers in PHASES:

            mid = (start + end) / 2

            ax.text(
                mid,
                ymax * 0.975,
                short,
                ha="center",
                va="top",
                fontsize=7.4,
                color="0.30",
            )


def bootstrap_decision_mean(dataframe, value_col):

    rows = []

    rng = np.random.default_rng(SEED)

    # Same resampling structure for all strategies.
    idx = rng.integers(
        0,
        10,
        size=(B, 10),
    )

    for strategy in ["S5", "S6", "S7"]:

        d = (
            dataframe[
                dataframe["strategy"] == strategy
            ]
            .pivot(
                index="repetition",
                columns="decision_number",
                values=value_col,
            )
            .sort_index()
            .sort_index(axis=1)
        )

        if d.shape != (10, 289):
            raise RuntimeError(
                f"{strategy} decision matrix {d.shape}, expected (10,289)"
            )

        arr = d.to_numpy(dtype=float)

        mean = arr.mean(axis=0)

        # 50,000 matched-run bootstrap means.
        boot = arr[idx, :].mean(axis=1)

        low = np.quantile(
            boot,
            0.025,
            axis=0,
        )

        high = np.quantile(
            boot,
            0.975,
            axis=0,
        )

        decisions = d.columns.to_numpy(dtype=int)

        rows.append(
            pd.DataFrame({
                "strategy": strategy,
                "decision_number": decisions,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
            })
        )

        del boot

    return pd.concat(
        rows,
        ignore_index=True,
    )


def add_panel_label(ax, text):

    ax.text(
        -0.10,
        1.03,
        text,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="bottom",
    )


# ============================================================
# FIGURE 1
# Decision-by-decision latency evolution + exact load schedule
# ============================================================

print("Generating Figure 1...")

decision_latency = bootstrap_decision_mean(
    master,
    "total_latency_ms",
)

# Attach frozen phase/worker identity.
schedule = (
    master[
        master["strategy"] == "S5"
    ][
        [
            "decision_number",
            "load_phase",
            "load_target_workers",
        ]
    ]
    .drop_duplicates("decision_number")
    .sort_values("decision_number")
)

decision_latency = decision_latency.merge(
    schedule,
    on="decision_number",
    how="left",
)

decision_latency.to_csv(
    SOURCE / "fig01_decision_latency_trajectory.csv",
    index=False,
)


fig, axes = plt.subplots(
    2,
    1,
    figsize=(7.25, 6.0),
    sharex=True,
    gridspec_kw={
        "height_ratios": [4.3, 1.25],
        "hspace": 0.08,
    },
)


ax = axes[0]

for strategy in ["S5", "S6", "S7"]:

    g = decision_latency[
        decision_latency["strategy"] == strategy
    ]

    st = STYLE[strategy]

    ax.plot(
        g["decision_number"],
        g["mean"],
        color=st["color"],
        linestyle=st["linestyle"],
        label=st["label"],
        linewidth=1.55,
    )

    ax.fill_between(
        g["decision_number"],
        g["ci_low"],
        g["ci_high"],
        color=st["color"],
        alpha=0.10,
        linewidth=0,
    )


ax.set_ylabel(
    "Failure-inclusive total latency (ms)"
)

ax.set_xlim(1, 289)

ax.grid(
    True,
    axis="y",
)

ax.legend(
    loc="upper left",
    ncol=3,
    frameon=False,
)

phase_boundaries(
    ax,
    labels=True,
)

add_panel_label(
    ax,
    "(a)",
)


ax2 = axes[1]

schedule_plot = schedule.sort_values(
    "decision_number"
)

ax2.step(
    schedule_plot["decision_number"],
    schedule_plot["load_target_workers"],
    where="post",
    color="0.15",
    linewidth=1.8,
)

ax2.set_xlabel(
    "Logical inference decision"
)

ax2.set_ylabel(
    "CPU-load\nworkers"
)

ax2.set_yticks(
    [0, 4, 6, 8]
)

ax2.set_xlim(1, 289)

ax2.grid(
    True,
    axis="y",
)

phase_boundaries(
    ax2,
    labels=False,
)

for short, name, start, end, workers in PHASES:

    mid = (start + end) / 2

    ax2.text(
        mid,
        8.75,
        short,
        ha="center",
        va="bottom",
        fontsize=7.3,
        color="0.30",
    )

ax2.set_ylim(-0.8, 10.0)

add_panel_label(
    ax2,
    "(b)",
)


fig.suptitle(
    "Decision-level latency response under the frozen L0–L6 CPU-load schedule",
    y=0.995,
    fontsize=10.5,
)

save_all(
    fig,
    "Fig01_decision_latency_and_load_schedule",
)


# ============================================================
# FIGURE 2
# Exact matched-block performance over ten repetitions
# ============================================================

print("Generating Figure 2...")

metrics = [
    (
        "total_mean_ms",
        "Mean latency (ms)",
        "(a)",
    ),
    (
        "total_median_ms",
        "Median latency (ms)",
        "(b)",
    ),
    (
        "total_p95_ms",
        "P95 latency (ms)",
        "(c)",
    ),
    (
        "total_p99_ms",
        "P99 latency (ms)",
        "(d)",
    ),
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(7.35, 6.2),
    sharex=True,
)

axes = axes.ravel()

for ax, (metric, ylabel, label) in zip(
    axes,
    metrics,
):

    for strategy in ["S5", "S6", "S7"]:

        g = (
            run[
                run["strategy"] == strategy
            ]
            .sort_values("block")
        )

        st = STYLE[strategy]

        ax.plot(
            g["block"],
            g[metric],
            color=st["color"],
            linestyle=st["linestyle"],
            marker=st["marker"],
            markerfacecolor="white",
            markeredgewidth=0.9,
            label=st["label"],
        )

    ax.set_ylabel(
        ylabel
    )

    ax.set_xticks(
        range(1, 11)
    )

    ax.grid(
        True,
        axis="y",
    )

    add_panel_label(
        ax,
        label,
    )


for ax in axes[2:]:
    ax.set_xlabel(
        "Matched experimental block"
    )


axes[0].legend(
    ncol=3,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(1.05, 1.20),
)

fig.suptitle(
    "Run-level latency across the ten matched experimental blocks",
    y=0.995,
    fontsize=10.5,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.95]
)

run.to_csv(
    SOURCE / "fig02_matched_block_latency.csv",
    index=False,
)

save_all(
    fig,
    "Fig02_matched_block_latency_profiles",
)


# ============================================================
# FIGURE 3
# Paired S7-vs-S6 reductions across the exact ten blocks
# ============================================================

print("Generating Figure 3...")

metric_labels = {
    "total_mean_ms":
        "Mean latency reduction (%)",

    "total_median_ms":
        "Median latency reduction (%)",

    "total_p95_ms":
        "P95 latency reduction (%)",

    "total_p99_ms":
        "P99 latency reduction (%)",
}

panel_labels = [
    "(a)",
    "(b)",
    "(c)",
    "(d)",
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(7.35, 6.2),
    sharex=True,
)

axes = axes.ravel()


for ax, metric, panel in zip(
    axes,
    metric_labels,
    panel_labels,
):

    g = (
        paired[
            (paired["comparison"] == "S7_vs_S6_PRIMARY")
            &
            (paired["metric"] == metric)
        ]
        .sort_values("block")
    )

    inf = inference[
        (inference["comparison"] == "S7_vs_S6_PRIMARY")
        &
        (inference["metric"] == metric)
    ].iloc[0]

    ax.plot(
        g["block"],
        g["paired_reduction_pct"],
        color=STYLE["S7"]["color"],
        marker="o",
        markerfacecolor="white",
        markeredgewidth=0.9,
        linewidth=1.8,
    )

    ax.axhline(
        0,
        color="0.35",
        linestyle=":",
        linewidth=1.0,
    )

    mean_reduction = float(
        inf["mean_paired_reduction_pct"]
    )

    ci_low = float(
        inf["bootstrap95_reduction_low_pct"]
    )

    ci_high = float(
        inf["bootstrap95_reduction_high_pct"]
    )

    ax.axhline(
        mean_reduction,
        color=STYLE["S7"]["color"],
        linestyle="--",
        linewidth=1.4,
    )

    ax.axhspan(
        ci_low,
        ci_high,
        color=STYLE["S7"]["color"],
        alpha=0.10,
    )

    ax.text(
        9.8,
        mean_reduction,
        f" mean {mean_reduction:.1f}%",
        ha="right",
        va="bottom",
        fontsize=7.5,
    )

    ax.set_ylabel(
        metric_labels[metric]
    )

    ax.set_xticks(
        range(1, 11)
    )

    ax.grid(
        True,
        axis="y",
    )

    add_panel_label(
        ax,
        panel,
    )


for ax in axes[2:]:
    ax.set_xlabel(
        "Matched experimental block"
    )


fig.suptitle(
    "Paired latency reduction of adaptive S7 relative to cloud-only S6",
    y=0.995,
    fontsize=10.5,
)

fig.tight_layout(
    rect=[0, 0, 1, 0.96]
)

paired[
    paired["comparison"] == "S7_vs_S6_PRIMARY"
].to_csv(
    SOURCE / "fig03_S7_vs_S6_paired_reductions.csv",
    index=False,
)

save_all(
    fig,
    "Fig03_paired_S7_vs_S6_latency_reduction",
)


# ============================================================
# FIGURE 4
# L0-L6 performance + adaptive placement share
# ============================================================

print("Generating Figure 4...")

fig, axes = plt.subplots(
    2,
    1,
    figsize=(7.25, 6.1),
    sharex=True,
    gridspec_kw={
        "height_ratios": [3.2, 2.1],
        "hspace": 0.12,
    },
)


ax = axes[0]

rng = np.random.default_rng(SEED)
boot_idx = rng.integers(
    0,
    10,
    size=(B, 10),
)


phase_plot_rows = []

for strategy in ["S5", "S6", "S7"]:

    means = []
    lows = []
    highs = []

    for ph in PHASE_ORDER:

        g = (
            phase[
                (phase["strategy"] == strategy)
                &
                (phase["phase"] == ph)
            ]
            .sort_values("repetition")
        )

        values = g[
            "total_median_ms"
        ].to_numpy(dtype=float)

        if len(values) != 10:
            raise RuntimeError(
                f"{strategy}/{ph}: expected 10 runs"
            )

        means.append(
            values.mean()
        )

        boot = values[
            boot_idx
        ].mean(axis=1)

        low, high = np.quantile(
            boot,
            [0.025, 0.975],
        )

        lows.append(low)
        highs.append(high)

        phase_plot_rows.append({
            "strategy": strategy,
            "phase": ph,
            "mean_run_phase_median_ms": values.mean(),
            "ci_low": low,
            "ci_high": high,
        })


    x = np.arange(7)

    st = STYLE[strategy]

    ax.plot(
        x,
        means,
        color=st["color"],
        linestyle=st["linestyle"],
        marker=st["marker"],
        markerfacecolor="white",
        markeredgewidth=0.9,
        label=st["label"],
    )

    ax.fill_between(
        x,
        lows,
        highs,
        color=st["color"],
        alpha=0.10,
    )


ax.set_ylabel(
    "Mean run-level phase\nmedian latency (ms)"
)

ax.grid(
    True,
    axis="y",
)

ax.legend(
    frameon=False,
    ncol=3,
)

add_panel_label(
    ax,
    "(a)",
)


ax2 = axes[1]

x = np.arange(7)

ax2.plot(
    x,
    s7_phase["selected_edge_pct_all_decisions"],
    color=STYLE["S5"]["color"],
    marker="o",
    markerfacecolor="white",
    label="S7 decisions placed at edge",
)

ax2.plot(
    x,
    s7_phase["selected_cloud_pct_all_decisions"],
    color=STYLE["S6"]["color"],
    marker="s",
    markerfacecolor="white",
    linestyle="--",
    label="S7 decisions placed in cloud",
)

ax2.set_ylim(
    -2,
    102,
)

ax2.set_ylabel(
    "S7 placement share (%)"
)

ax2.set_xlabel(
    "Externally imposed CPU-load phase"
)

ax2.set_xticks(
    x,
    PHASE_SHORT,
)

ax2.grid(
    True,
    axis="y",
)

ax2.legend(
    frameon=False,
    ncol=2,
)

add_panel_label(
    ax2,
    "(b)",
)


pd.DataFrame(
    phase_plot_rows
).to_csv(
    SOURCE / "fig04_phase_latency_ci.csv",
    index=False,
)

s7_phase.to_csv(
    SOURCE / "fig04_S7_phase_placement.csv",
    index=False,
)

fig.suptitle(
    "Load-phase performance and adaptive edge–cloud placement",
    y=0.995,
    fontsize=10.5,
)

save_all(
    fig,
    "Fig04_phase_response_and_adaptive_placement",
)


# ============================================================
# FIGURE 5
# S7 internal adaptive dynamics over all 289 decisions
# ============================================================

print("Generating Figure 5...")

s7 = master[
    master["strategy"] == "S7"
].copy()

s7["edge_CPU_EWMA_percent"] = pd.to_numeric(
    s7["edge_CPU_EWMA_percent"],
    errors="coerce",
)

cpu = (
    s7
    .groupby("decision_number")[
        "edge_CPU_EWMA_percent"
    ]
    .agg(["mean", "std"])
    .reset_index()
)

cpu["sem"] = (
    cpu["std"]
    /
    np.sqrt(10)
)

cpu["ci_low"] = (
    cpu["mean"]
    -
    1.96
    *
    cpu["sem"]
)

cpu["ci_high"] = (
    cpu["mean"]
    +
    1.96
    *
    cpu["sem"]
)


placement = (
    s7.assign(
        selected_cloud=(
            s7["selected_site"]
            .astype(str)
            .eq("cloud")
            .astype(int)
        )
    )
    .groupby("decision_number")[
        "selected_cloud"
    ]
    .agg(["sum", "mean"])
    .reset_index()
)

placement["cloud_selection_pct"] = (
    placement["mean"]
    *
    100
)


s7_dynamics = cpu.merge(
    placement[
        [
            "decision_number",
            "sum",
            "cloud_selection_pct",
        ]
    ],
    on="decision_number",
)

s7_dynamics.rename(
    columns={
        "sum":
            "cloud_selected_runs_of_10"
    },
    inplace=True,
)

s7_dynamics.to_csv(
    SOURCE / "fig05_S7_controller_dynamics.csv",
    index=False,
)


fig, axes = plt.subplots(
    2,
    1,
    figsize=(7.25, 6.0),
    sharex=True,
    gridspec_kw={
        "hspace": 0.10,
    },
)


ax = axes[0]

ax.plot(
    cpu["decision_number"],
    cpu["mean"],
    color="#6A3D9A",
    linewidth=1.55,
)

ax.fill_between(
    cpu["decision_number"],
    cpu["ci_low"],
    cpu["ci_high"],
    color="#6A3D9A",
    alpha=0.11,
)

ax.axhline(
    55,
    color="0.35",
    linestyle="--",
    linewidth=1.0,
)

ax.axhline(
    80,
    color="0.35",
    linestyle="-.",
    linewidth=1.0,
)

ax.text(
    286,
    55,
    " U_LOW=55%",
    ha="right",
    va="bottom",
    fontsize=7.2,
)

ax.text(
    286,
    80,
    " U_HIGH=80%",
    ha="right",
    va="bottom",
    fontsize=7.2,
)

ax.set_ylabel(
    "S7 CPU EWMA (%)"
)

ax.grid(
    True,
    axis="y",
)

phase_boundaries(
    ax,
    labels=True,
)

add_panel_label(
    ax,
    "(a)",
)


ax2 = axes[1]

ax2.plot(
    placement["decision_number"],
    placement["cloud_selection_pct"],
    color=STYLE["S6"]["color"],
    linewidth=1.55,
)

ax2.set_ylim(
    -3,
    103,
)

ax2.set_xlim(
    1,
    289,
)

ax2.set_ylabel(
    "Runs selecting cloud (%)"
)

ax2.set_xlabel(
    "Logical inference decision"
)

ax2.set_yticks(
    [0, 25, 50, 75, 100]
)

ax2.grid(
    True,
    axis="y",
)

phase_boundaries(
    ax2,
    labels=False,
)

add_panel_label(
    ax2,
    "(b)",
)


fig.suptitle(
    "Adaptive-controller response across the complete S7 decision sequence",
    y=0.995,
    fontsize=10.5,
)

save_all(
    fig,
    "Fig05_S7_controller_dynamics",
)


# ============================================================
# FIGURE 6
# Exact empirical distribution and tail behavior
# ============================================================

print("Generating Figure 6...")

fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.5, 3.65),
)


for strategy in ["S5", "S6", "S7"]:

    values = np.sort(
        master.loc[
            master["strategy"] == strategy,
            "total_latency_ms",
        ].to_numpy(dtype=float)
    )

    n = len(values)

    cdf = (
        np.arange(1, n + 1)
        /
        n
    )

    ccdf = (
        (n - np.arange(1, n + 1) + 1)
        /
        n
    )

    st = STYLE[strategy]

    axes[0].plot(
        values,
        cdf * 100,
        color=st["color"],
        linestyle=st["linestyle"],
        label=st["label"],
        linewidth=1.6,
    )

    axes[1].semilogy(
        values,
        ccdf,
        color=st["color"],
        linestyle=st["linestyle"],
        label=st["label"],
        linewidth=1.6,
    )


axes[0].set_xlabel(
    "Failure-inclusive total latency (ms)"
)

axes[0].set_ylabel(
    "Empirical cumulative probability (%)"
)

axes[0].set_ylim(
    0,
    100,
)

axes[0].grid(
    True,
    alpha=0.23,
)

add_panel_label(
    axes[0],
    "(a)",
)


axes[1].set_xlabel(
    "Failure-inclusive total latency (ms)"
)

axes[1].set_ylabel(
    "Exceedance probability"
)

axes[1].grid(
    True,
    which="both",
    alpha=0.23,
)

add_panel_label(
    axes[1],
    "(b)",
)


axes[0].legend(
    frameon=False,
    loc="lower right",
)


fig.suptitle(
    "Empirical latency distribution and tail behaviour (2,890 decisions per strategy)",
    y=1.02,
    fontsize=10.5,
)

fig.tight_layout()

save_all(
    fig,
    "Fig06_latency_ECDF_and_tail_CCDF",
)


# ============================================================
# FIGURE 7
# Reliability + fallback recovery sequence
# ============================================================

print("Generating Figure 7...")

m = master.copy()

m["final_success_bool"] = (
    m["final_success"]
    .astype(str)
    .str.lower()
    .isin(["true", "1", "yes"])
)

m["fallback_bool"] = (
    m["fallback_used"]
    .astype(str)
    .str.lower()
    .isin(["true", "1", "yes"])
)

m["primary_failed_bool"] = (
    m["primary_attempt_status"]
    .astype(str)
    .str.lower()
    .eq("failed")
)


rows = []

for decision in range(1, 290):

    s6d = m[
        (m["strategy"] == "S6")
        &
        (m["decision_number"] <= decision)
    ]

    s7d = m[
        (m["strategy"] == "S7")
        &
        (m["decision_number"] <= decision)
    ]

    rows.append({
        "decision_number": decision,

        "S6_cumulative_final_failures":
            int(
                (~s6d["final_success_bool"])
                .sum()
            ),

        "S7_cumulative_primary_failures":
            int(
                s7d["primary_failed_bool"]
                .sum()
            ),

        "S7_cumulative_successful_fallbacks":
            int(
                (
                    s7d["fallback_bool"]
                    &
                    s7d["final_success_bool"]
                )
                .sum()
            ),

        "S7_cumulative_final_failures":
            int(
                (~s7d["final_success_bool"])
                .sum()
            ),
    })


reliability = pd.DataFrame(rows)

reliability.to_csv(
    SOURCE / "fig07_cumulative_reliability_events.csv",
    index=False,
)


fig, ax = plt.subplots(
    figsize=(7.25, 4.35),
)


ax.step(
    reliability["decision_number"],
    reliability["S6_cumulative_final_failures"],
    where="post",
    color=STYLE["S6"]["color"],
    linestyle="--",
    linewidth=1.9,
    label="S6 final failures",
)

ax.step(
    reliability["decision_number"],
    reliability["S7_cumulative_primary_failures"],
    where="post",
    color="#CC79A7",
    linewidth=1.7,
    label="S7 primary failures",
)

ax.step(
    reliability["decision_number"],
    reliability["S7_cumulative_successful_fallbacks"],
    where="post",
    color=STYLE["S7"]["color"],
    linestyle="-.",
    linewidth=1.7,
    label="S7 successful fallback recoveries",
)

ax.step(
    reliability["decision_number"],
    reliability["S7_cumulative_final_failures"],
    where="post",
    color="0.15",
    linestyle=":",
    linewidth=1.6,
    label="S7 final failures",
)


ax.set_xlabel(
    "Logical inference decision"
)

ax.set_ylabel(
    "Cumulative events across 10 runs"
)

ax.set_xlim(
    1,
    289,
)

ax.grid(
    True,
    axis="y",
)

phase_boundaries(
    ax,
    labels=True,
)

ax.legend(
    frameon=False,
    ncol=2,
    loc="upper left",
)

fig.suptitle(
    "Reliability and adaptive fallback behaviour across the frozen workload",
    y=0.995,
    fontsize=10.5,
)

save_all(
    fig,
    "Fig07_reliability_and_fallback_trace",
)


# ============================================================
# CAPTIONS
# ============================================================

captions = """\
PUBLICATION FIGURE CAPTIONS — V2
================================

Figure 1.
Decision-level failure-inclusive total latency under the frozen L0–L6
externally imposed CPU-load schedule. Panel (a) shows the mean total
latency at each of the 289 logical inference decisions across the ten
matched repetitions for S5 (edge-only), S6 (cloud-only), and S7
(adaptive edge–cloud placement); shaded regions denote 95% paired-run
bootstrap confidence bands. Panel (b) shows the exact number of CPU-load
workers imposed by the frozen controller schedule. Vertical dotted lines
mark the L0–L6 phase transitions.

Figure 2.
Run-level latency across the ten matched experimental blocks. The four
panels report failure-inclusive mean, median, P95, and P99 total latency,
respectively. Each point corresponds to one complete 289-decision final
run, preserving the matched-block structure used in the inferential
analysis.

Figure 3.
Paired block-wise latency reduction achieved by S7 relative to S6.
Positive values indicate lower latency for the adaptive strategy.
Each panel shows the ten exact paired reductions for mean, median, P95,
or P99 latency. The dashed horizontal line is the mean paired reduction,
and the shaded horizontal region is its 95% paired-block bootstrap
confidence interval based on 50,000 replications.

Figure 4.
Load-phase performance and S7 placement adaptation. Panel (a) shows the
mean of the ten run-level phase median latencies for S5, S6, and S7,
with 95% bootstrap confidence bands. Panel (b) shows the proportion of
S7 logical decisions executed at the edge and cloud during each frozen
L0–L6 load phase. The same externally imposed CPU-load schedule was
applied to all three strategies.

Figure 5.
Adaptive-controller dynamics throughout the 289-decision S7 workload.
Panel (a) shows the mean CPU-utilization EWMA across ten S7 repetitions,
with uncertainty bounds and the frozen U_LOW=55% and U_HIGH=80%
controller thresholds. Panel (b) shows, for every logical decision, the
percentage of the ten repetitions in which S7 selected cloud execution.
Vertical dotted lines identify the fixed load-phase transitions.

Figure 6.
Empirical distribution of failure-inclusive total inference latency.
Panel (a) shows the empirical cumulative distribution function and
panel (b) the complementary distribution on a logarithmic probability
scale, exposing tail behavior. Each strategy contributes 2,890 logical
decisions. These pooled curves are descriptive only; statistical
inference in the study uses the ten matched experimental blocks.

Figure 7.
Reliability and adaptive fallback behavior across the deterministic
289-decision workload. Curves show cumulative S6 final failures, S7
primary request failures, S7 successful fallback recoveries, and S7
final failures aggregated across the ten final repetitions. Phase
boundaries correspond to the common frozen L0–L6 load schedule.
"""

(OUT / "FIGURE_CAPTIONS.txt").write_text(
    captions
)


# ============================================================
# MANIFEST
# ============================================================

manifest_rows = []

for path in sorted(
    OUT.glob("Fig*.*")
):

    h = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    manifest_rows.append({
        "file": path.name,
        "sha256": h,
        "bytes": path.stat().st_size,
    })


pd.DataFrame(
    manifest_rows
).to_csv(
    OUT / "FIGURE_MANIFEST.csv",
    index=False,
)


print()
print("==============================================")
print("PUBLICATION FIGURES V2 = COMPLETE")
print("==============================================")
print("Figures generated : 7")
print("Formats per figure: PDF, SVG, EPS, PNG 600 dpi")
print("Source-data folder:", SOURCE)
print("Output folder     :", OUT)
print("NO EXPERIMENT WAS EXECUTED")
