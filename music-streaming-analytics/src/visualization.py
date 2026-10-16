"""
src/visualization.py
────────────────────
Matplotlib / Seaborn static chart helpers for the Jupyter notebook.
Each function saves a PNG to outputs/figures/ and returns the axes/figure.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

PALETTE = {
    "blue"  : "#3b82d4",
    "purple": "#7c5cd8",
    "red"   : "#e05c5c",
    "bg"    : "#f7f8fa",
    "border": "#e5e7eb",
}


def _save(fig: plt.Figure, filename: str):
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_stream_distribution(df: pd.DataFrame) -> plt.Figure:
    """Visualization 1 — Raw vs. log-transformed stream count distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df["stream_count"], bins=80,
                 color=PALETTE["blue"], edgecolor="white", linewidth=0.3)
    axes[0].set_title("Raw Stream Count Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("stream_count")
    axes[0].set_ylabel("Frequency")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    axes[0].annotate(f"Skew = {df['stream_count'].skew():.2f}",
                     xy=(0.65, 0.88), xycoords="axes fraction",
                     fontsize=10, color="red")

    axes[1].hist(df["log_stream_count"], bins=80,
                 color=PALETTE["purple"], edgecolor="white", linewidth=0.3)
    axes[1].axvline(df["log_stream_count"].median(), color="orange",
                    linestyle="--", linewidth=1.5,
                    label=f"Median = {df['log_stream_count'].median():.2f}")
    axes[1].set_title("log(Stream Count) Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("log(1 + stream_count)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
    axes[1].annotate(f"Skew = {df['log_stream_count'].skew():.2f}",
                     xy=(0.65, 0.88), xycoords="axes fraction",
                     fontsize=10, color="green")

    fig.suptitle("Visualization 1 — Stream Count: Before and After Log Transformation",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, "viz1_stream_distribution.png")
    return fig


def plot_genre_streams(df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """Visualization 2 — log(stream_count) by genre (top N)."""
    top_genres = df["genre"].value_counts().head(top_n).index.tolist()
    gdf = df[df["genre"].isin(top_genres)].copy()
    order = (gdf.groupby("genre")["log_stream_count"]
               .median().sort_values(ascending=False).index)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=gdf, x="genre", y="log_stream_count", order=order,
                palette="tab10", width=0.5, linewidth=0.8, ax=ax,
                flierprops=dict(marker="o", markersize=2, alpha=0.3))
    ax.set_title(f"Visualization 2 — log(Stream Count) by Genre (Top {top_n})",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Genre")
    ax.set_ylabel("log(1 + stream_count)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    _save(fig, "viz2_genre_streams.png")
    return fig


def plot_popularity_vs_streams(df: pd.DataFrame, n_sample: int = 5000) -> plt.Figure:
    """Visualization 3 — Popularity vs. log(stream_count) scatter."""
    from scipy.stats import pearsonr
    r, _ = pearsonr(df["log_stream_count"], df["popularity"])
    sample = df.sample(min(n_sample, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(sample["log_stream_count"], sample["popularity"],
               alpha=0.15, s=8, color=PALETTE["blue"], label=f"Track (n={len(sample):,})")
    m, b = np.polyfit(df["log_stream_count"], df["popularity"], 1)
    xs = np.linspace(df["log_stream_count"].min(), df["log_stream_count"].max(), 200)
    ax.plot(xs, m * xs + b, color="red", linewidth=1.8, label=f"OLS fit (r={r:.3f})")
    ax.set_title("Visualization 3 — Popularity vs. log(Stream Count)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("log(1 + stream_count)")
    ax.set_ylabel("Popularity (0–100)")
    ax.legend()
    plt.tight_layout()
    _save(fig, "viz3_popularity_vs_streams.png")
    return fig


def plot_audio_correlations(df: pd.DataFrame) -> plt.Figure:
    """Visualization 4 — Audio feature correlations with log(stream_count)."""
    from scipy.stats import pearsonr
    audio_feats = ["danceability", "energy", "loudness", "instrumentalness",
                   "tempo", "duration_ms", "upbeat_score"]
    corrs = {c: pearsonr(df[c], df["log_stream_count"])[0] for c in audio_feats}
    names  = list(corrs.keys())
    values = list(corrs.values())
    colors = [PALETTE["blue"] if v >= 0 else PALETTE["red"] for v in values]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(0,    color="black", linewidth=0.8)
    ax.axhline( 0.05, color="green", linewidth=0.8, linestyle="--", alpha=0.6, label="r = ±0.05")
    ax.axhline(-0.05, color="green", linewidth=0.8, linestyle="--", alpha=0.6)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + (0.001 if v >= 0 else -0.003),
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Visualization 4 — Audio Feature Correlations with log(Stream Count)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Audio Feature")
    ax.set_ylabel("Pearson r")
    ax.set_ylim(-0.03, 0.03)
    ax.legend()
    plt.tight_layout()
    _save(fig, "viz4_audio_correlations.png")
    return fig


def plot_release_year_trend(df: pd.DataFrame) -> plt.Figure:
    """Visualization 5 — Release year track volume and median streaming."""
    yearly = df.groupby("release_year").agg(
        track_count   = ("track_id", "count"),
        median_log_sc = ("log_stream_count", "median"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    ax1.bar(yearly["release_year"], yearly["track_count"],
            color=PALETTE["blue"], alpha=0.7, width=0.4, label="Tracks Released")
    ax2.plot(yearly["release_year"], yearly["median_log_sc"],
             color=PALETTE["red"], marker="o", linewidth=2, label="Median log(Streams)")
    ax1.set_xlabel("Release Year")
    ax1.set_ylabel("Number of Tracks", color=PALETTE["blue"])
    ax2.set_ylabel("Median log(Stream Count)", color=PALETTE["red"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["blue"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["red"])
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Visualization 5 — Release Year: Track Volume & Median Streaming (2020–2025)",
                  fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "viz5_release_year_trend.png")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Correlation heatmap for numerical features."""
    num_cols = ["popularity", "stream_count", "log_stream_count",
                "danceability", "energy", "loudness", "instrumentalness",
                "tempo", "duration_ms", "artist_track_count", "upbeat_score"]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.4, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Matrix — Numerical Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, "correlation_matrix.png")
    return fig


def plot_model_residuals(y_test: np.ndarray, y_pred: np.ndarray,
                          model_name: str = "Model") -> plt.Figure:
    """Residual plot for regression model evaluation."""
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.1, s=6, color=PALETTE["blue"])
    axes[0].axhline(0, color="red", linewidth=1.2, linestyle="--")
    axes[0].set_xlabel("Predicted log(stream_count)")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residual Plot", fontweight="bold")

    axes[1].hist(residuals, bins=60, color=PALETTE["purple"],
                 edgecolor="white", linewidth=0.3)
    axes[1].axvline(0, color="red", linewidth=1.2, linestyle="--")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution", fontweight="bold")

    fig.suptitle(f"Model Residuals — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "model_residuals.png")
    return fig
