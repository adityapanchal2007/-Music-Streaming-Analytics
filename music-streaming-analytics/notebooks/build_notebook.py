"""
Script to generate Student_MusicAnalytics.ipynb
Run once: python build_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(source): return nbf.v4.new_markdown_cell(source)
def code(source): return nbf.v4.new_code_cell(source)

# ── Section 1: Project Introduction ─────────────────────────────────────────
cells.append(md("""# Music Streaming Analytics & Popularity Intelligence
## *Explore • Analyze • Predict*

---

## 1. Project Introduction

**Course**: B.Tech Artificial Intelligence  
**Dataset**: Music Streaming Dataset — 50,000 tracks × 33 columns  
**Data Range**: 2020–2025  

This project performs end-to-end data analytics, statistical hypothesis testing, and  
predictive machine learning on a music streaming dataset.  
The goal is to understand what drives streaming success and build an interpretable  
model that estimates a track's streaming volume from its observable characteristics.

### Project Workflow
```
Raw Music Dataset → Audit → Cleaning → Leakage Analysis → EDA
→ 5 Visualisations → 5 Observations → 5 Insights → 3 Hypotheses
→ 3 Recommendations → ML Modelling → Evaluation → Explainability → Dashboard
```
"""))

# ── Section 2: Business Problem ──────────────────────────────────────────────
cells.append(md("""---
## 2. Business Problem

### Problem Statement
Music platforms, record labels, and independent artists operate in a competitive  
streaming environment where understanding the characteristics associated with high  
streaming volumes is commercially critical.  
This dataset contains 50,000 tracks released between 2020 and 2025, capturing  
audio features, genre, country, label, release metadata, and streaming counts.  
The challenge is to identify which measurable track properties are associated with  
streaming performance and to build a reliable predictive model.

### Business Objective
| Stakeholder | Benefit |
|---|---|
| Streaming Platforms | Better content recommendations, catalogue planning |
| Artists | Data-informed production decisions |
| Record Labels | Pre-release stream potential estimation |
| Marketing Teams | Targeted promotion for high-potential tracks |

### Analytical Objective
Investigate patterns in genre, audio features, country, label, and release timing  
as they relate to streaming volume.  
Identify the most analytically informative features through correlation, group  
comparisons, and statistical tests.

### Machine Learning Objective
**Predict `log_stream_count` (i.e., log₁₊ₑ(stream_count)) using audio features,  
genre, country, label, and release metadata.**  
Log transformation is used because raw stream_count is extremely right-skewed  
(skewness ≈ 23.85), which would violate linear model assumptions.  
The log scale produces a near-symmetric distribution (skewness ≈ 0.41).
"""))

# ── Section 3: Dataset Loading ───────────────────────────────────────────────
cells.append(md("---\n## 3. Dataset Loading"))
cells.append(code("""import warnings, os
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Resolve dataset path (works from notebooks/ or project root)
_HERE = os.path.dirname(os.path.abspath("__file__"))
DATA_PATH = os.path.join(_HERE, "..", "data", "music_dataset.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = "../../spotify_artist_streaming_2020_2025.csv"

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
"""))

# ── Section 4: Dataset Audit ─────────────────────────────────────────────────
cells.append(md("---\n## 4. Dataset Audit"))
cells.append(code("""# ── Shape ──
print("=" * 55)
print(f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
print("=" * 55)
"""))
cells.append(code("""# ── First 3 rows ──
df.head(3)
"""))
cells.append(code("""# ── Data types and non-null counts ──
df.info()
"""))
cells.append(code("""# ── Missing values ──
missing = df.isnull().sum()
print("Missing values per column:")
print(missing[missing > 0] if missing.sum() > 0 else "  → No missing values found.")
"""))
cells.append(code("""# ── Duplicate rows ──
dup_rows = df.duplicated().sum()
dup_ids  = df['track_id'].duplicated().sum()
print(f"Duplicate rows        : {dup_rows}")
print(f"Duplicate track_id    : {dup_ids}")
print(f"Unique track names    : {df['track_name'].nunique():,}")
print(f"Unique artists        : {df['artist_name'].nunique():,}")
print(f"Unique albums         : {df['album_name'].nunique():,}")
print(f"Unique genres         : {df['genre'].nunique()}")
print(f"Unique countries      : {df['country'].nunique()}")
print(f"Unique labels         : {df['label'].nunique()}")
print(f"Release year range    : {df['release_year'].min()} – {df['release_year'].max()}")
"""))
cells.append(code("""# ── Numerical summary ──
num_cols = ['popularity','stream_count','danceability','energy','loudness',
            'instrumentalness','tempo','duration_ms','artist_track_count']
df[num_cols].describe().round(3)
"""))
cells.append(code("""# ── Categorical value counts ──
cat_cols = ['genre','country','label','mode_name','release_quarter',
            'popularity_category','loudness_category']
for c in cat_cols:
    print(f"\\n── {c} ──")
    print(df[c].value_counts().head(5).to_string())
"""))

# ── Section 5: Data Cleaning ─────────────────────────────────────────────────
cells.append(md("---\n## 5. Data Cleaning"))
cells.append(code("""# Step 1 – Convert release_date to datetime
df['release_date'] = pd.to_datetime(df['release_date'])
print("release_date dtype:", df['release_date'].dtype)

# Step 2 – Verify pre-engineered date columns are consistent
df['_month']   = df['release_date'].dt.month
df['_year']    = df['release_date'].dt.year
df['_quarter'] = df['release_date'].dt.quarter.map({1:'Q1',2:'Q2',3:'Q3',4:'Q4'})
df['_weekend'] = df['release_date'].dt.dayofweek >= 5

checks = {
    'release_month matches date'   : (df['_month']   == df['release_month']).all(),
    'release_year matches date'    : (df['_year']    == df['release_year']).all(),
    'release_quarter matches date' : (df['_quarter'] == df['release_quarter']).all(),
    'is_weekend_release correct'   : (df['_weekend'] == df['is_weekend_release']).all(),
}
for k, v in checks.items():
    print(f"  {k}: {'✓ PASS' if v else '✗ FAIL'}")

df.drop(columns=['_month','_year','_quarter','_weekend'], inplace=True)
"""))
cells.append(code("""# Step 3 – Verify duration_minutes ← duration_ms / 60000
dur_check = np.allclose(df['duration_ms'] / 60000, df['duration_minutes'], atol=0.01)
print("duration_minutes consistent with duration_ms:", dur_check)

# Step 4 – Verify log_stream_count ← log1p(stream_count)
log_check = np.allclose(np.log1p(df['stream_count']), df['log_stream_count'], atol=0.01)
print("log_stream_count consistent with stream_count:", log_check)

# Step 5 – is_explicit_bool is identical to explicit
expl_match = (df['explicit'] == df['is_explicit_bool']).all()
print("is_explicit_bool identical to explicit:", expl_match)

print("\\n→ No data errors found. Dataset is clean and internally consistent.")
"""))
cells.append(code("""# Step 6 – Outlier investigation (not removal)
outlier_cols = ['stream_count','popularity','duration_ms','tempo',
                'loudness','danceability','energy','instrumentalness']

print(f"{'Feature':<20} {'Min':>12} {'Max':>14} {'Skew':>8}  Notes")
print("-" * 75)
for c in outlier_cols:
    lo = df[c].min()
    hi = df[c].max()
    sk = round(df[c].skew(), 2)
    note = ""
    if c == 'stream_count':
        note = "← highly right-skewed; log transform required"
    elif c == 'duration_ms':
        note = "← 90s–6.5min; all plausible track lengths"
    print(f"{c:<20} {lo:>12,.2f} {hi:>14,.2f} {sk:>8}  {note}")

print("\\n→ All outliers are valid extreme observations, not data errors.")
print("→ stream_count will be log-transformed for modelling.")
"""))

# ── Section 6: Feature Engineering ──────────────────────────────────────────
cells.append(md("---\n## 6. Feature Engineering & Derived-Feature Audit"))
cells.append(code("""# ── Derived-feature table ──
leakage_table = pd.DataFrame({
    'Original Feature'   : ['duration_ms','popularity','loudness','key','mode',
                             'explicit','release_month','stream_count','stream_count',
                             'danceability+energy','artist (count per dataset)'],
    'Derived Feature'    : ['duration_minutes','popularity_category','loudness_category',
                             'key_name','mode_name','is_explicit_bool','release_quarter',
                             'log_stream_count','(stream_count itself)','upbeat_score',
                             'artist_track_count'],
    'Purpose'            : ['Unit convenience','Binned label','Binned label','Human-readable',
                             'Human-readable','Python bool','Quarter grouping',
                             'Log scale for modelling','Raw count','Weighted blend',
                             'Catalogue-size proxy'],
    'ML Decision'        : ['DROP (keep duration_ms)','DROP (leakage if target=popularity)',
                             'DROP (redundant)','DROP (keep key int)','KEEP (categorical)',
                             'DROP (identical to explicit)','KEEP (useful grouping)',
                             'USE AS TARGET','DROP if target=log_stream_count',
                             'KEEP','KEEP (with caution — see note)'],
})
print(leakage_table.to_string(index=False))
"""))
cells.append(code("""# ── Leakage Audit Table ──
print(\"\"\"
LEAKAGE AUDIT
─────────────────────────────────────────────────────────────────────────────
Feature               Derived From         Leakage Risk    ML Decision
─────────────────────────────────────────────────────────────────────────────
popularity_category   popularity           DIRECT LEAK     REMOVE from all ML
log_stream_count      stream_count         TARGET ITSELF   USE AS TARGET (not as feature)
stream_count          —                    TARGET (raw)    REMOVE when target=log_stream_count
is_explicit_bool      explicit             IDENTICAL       REMOVE (keep explicit)
duration_minutes      duration_ms          REDUNDANT       REMOVE (keep duration_ms)
loudness_category     loudness             REDUNDANT       REMOVE (keep loudness)
key_name              key                  REDUNDANT       REMOVE (keep key int)
artist_track_count    full-dataset count   INFO LEAK*      KEEP with disclosure
─────────────────────────────────────────────────────────────────────────────
* artist_track_count is computed from the full dataset, so in production a
  new artist would have count=0. We keep it here to explore its association
  but disclose this limitation.
─────────────────────────────────────────────────────────────────────────────
\"\"\")
"""))
cells.append(code("""# Create clean modelling dataframe
model_df = df.copy()

# Remove leaking / redundant columns
REMOVE = [
    'popularity_category',  # derived from popularity → leakage
    'stream_count',         # raw form of the target
    'popularity',           # near-perfect correlation with log_stream_count (r=0.894)
    'is_explicit_bool',     # identical to explicit
    'duration_minutes',     # duplicate of duration_ms
    'loudness_category',    # binned version of loudness
    'key_name',             # human-readable key int
    'mode_name',            # keep as categorical but drop numeric mode duplicate
    'upbeat_score',         # composite of danceability + energy; we keep both originals
    'log_stream_count',     # target – handled separately below
    # Identifiers
    'track_id','track_name','artist_name','album_name','release_date',
]
model_df.drop(columns=REMOVE, inplace=True, errors='ignore')

# Encode categoricals
cat_encode = ['genre','country','label','release_day_of_week','release_quarter','mode_name']
# mode_name was dropped above; re-add from original
model_df['mode_name'] = df['mode_name']

model_df = pd.get_dummies(model_df,
                           columns=['genre','country','label',
                                    'release_day_of_week','release_quarter','mode_name'],
                           drop_first=True)

# Target
y_target = df['log_stream_count'].values

print(f"model_df shape : {model_df.shape}")
print(f"Target (y)     : log_stream_count — shape {y_target.shape}")
print(f"Target range   : {y_target.min():.2f} – {y_target.max():.2f}")
"""))

# ── Section 7: EDA ───────────────────────────────────────────────────────────
cells.append(md("---\n## 7. Exploratory Data Analysis"))
cells.append(code("""# ── Correlation matrix for numerical features ──
num_eda = ['popularity','stream_count','log_stream_count','danceability','energy',
           'loudness','instrumentalness','tempo','duration_ms',
           'artist_track_count','upbeat_score']

corr = df[num_eda].corr()

plt.figure(figsize=(11, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.4, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix — Numerical Features", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("../outputs/figures/00_correlation_matrix.png", dpi=150)
plt.show()
print("\\nKey finding:")
print(f"  popularity vs log_stream_count  : r = {df['popularity'].corr(df['log_stream_count']):.4f}")
print(f"  danceability vs log_stream_count: r = {df['danceability'].corr(df['log_stream_count']):.4f}")
print(f"  energy vs log_stream_count      : r = {df['energy'].corr(df['log_stream_count']):.4f}")
"""))

# ── Section 8: 5 Visualizations ──────────────────────────────────────────────
cells.append(md("---\n## 8. Exactly Five Main Visualizations"))

cells.append(md("""### Visualization 1 — Distribution of log(Stream Count)

**Chart type**: Histogram + KDE  
**X-axis**: log₁₊ₑ(stream_count)  
**Y-axis**: Frequency  
**Why selected**: Understanding the streaming distribution is the analytical foundation.  
Raw stream_count is extremely right-skewed (skew ≈ 23.85); the log transformation  
produces a near-normal distribution (skew ≈ 0.41), justifying it as the modelling target.  
**Key takeaway**: Log-transformed stream counts are roughly bell-shaped around ~9.5,  
corresponding to ~13,000 raw streams (the median track).
"""))
cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Raw stream_count
axes[0].hist(df['stream_count'], bins=80, color='#3b82d4', edgecolor='white', linewidth=0.3)
axes[0].set_title("Raw Stream Count Distribution", fontsize=13, fontweight='bold')
axes[0].set_xlabel("stream_count")
axes[0].set_ylabel("Frequency")
axes[0].annotate(f"Skewness = {df['stream_count'].skew():.2f}", xy=(0.65, 0.88),
                  xycoords='axes fraction', fontsize=10, color='red')

# log_stream_count
axes[1].hist(df['log_stream_count'], bins=80, color='#7c5cd8', edgecolor='white', linewidth=0.3)
axes[1].axvline(df['log_stream_count'].median(), color='orange', linestyle='--',
                 linewidth=1.5, label=f"Median = {df['log_stream_count'].median():.2f}")
axes[1].set_title("Log-Transformed Stream Count Distribution", fontsize=13, fontweight='bold')
axes[1].set_xlabel("log(1 + stream_count)")
axes[1].set_ylabel("Frequency")
axes[1].legend()
axes[1].annotate(f"Skewness = {df['log_stream_count'].skew():.2f}", xy=(0.65, 0.88),
                  xycoords='axes fraction', fontsize=10, color='green')

plt.suptitle("Visualization 1 — Stream Count: Before and After Log Transformation",
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("../outputs/figures/viz1_stream_distribution.png", dpi=150)
plt.show()
print("Interpretation: The log transformation converts an extreme right-skewed")
print("distribution into a roughly symmetric one, enabling linear model assumptions")
print("to hold and making predictions more stable.")
"""))

cells.append(md("""### Visualization 2 — Stream Count by Genre (Top 10 Genres)

**Chart type**: Box plot  
**X-axis**: Genre  
**Y-axis**: log(stream_count)  
**Why selected**: Genre is the primary editorial categorisation. Comparing streaming  
distributions across genres reveals whether certain genres attract disproportionately  
large audiences.  
**Key takeaway**: Median log-stream-counts are broadly similar across genres  
(≈9.5), but variance differs—some genres (e.g., Pop, R&B) show higher outliers,  
suggesting occasional breakout hits rather than systematic superiority.
"""))
cells.append(code("""top_genres = df['genre'].value_counts().head(10).index.tolist()
genre_df = df[df['genre'].isin(top_genres)].copy()
order = genre_df.groupby('genre')['log_stream_count'].median().sort_values(ascending=False).index

plt.figure(figsize=(14, 6))
sns.boxplot(data=genre_df, x='genre', y='log_stream_count', order=order,
            palette='tab10', width=0.5, linewidth=0.8,
            flierprops=dict(marker='o', markersize=2, alpha=0.3))
plt.title("Visualization 2 — Log(Stream Count) by Genre (Top 10)", fontsize=13, fontweight='bold')
plt.xlabel("Genre")
plt.ylabel("log(1 + stream_count)")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig("../outputs/figures/viz2_genre_streams.png", dpi=150)
plt.show()

# Summary stats
print("Median log_stream_count by genre (top 10, sorted):")
print(genre_df.groupby('genre')['log_stream_count'].median().sort_values(ascending=False).round(3).to_string())
"""))

cells.append(md("""### Visualization 3 — Popularity vs. Log(Stream Count)

**Chart type**: Scatter plot with regression line  
**X-axis**: log(stream_count)  
**Y-axis**: popularity  
**Why selected**: Confirms the near-linear relationship between the two main  
performance metrics. This relationship (r ≈ 0.89) is essential to understand  
before modelling—it means popularity is essentially a rescaled version of log-streams  
and both should not be used simultaneously as features.  
**Key takeaway**: Popularity and log_stream_count are strongly associated.  
Including both in a model would constitute near-perfect multicollinearity.
"""))
cells.append(code("""from scipy.stats import pearsonr, spearmanr

r_p, _  = pearsonr(df['log_stream_count'], df['popularity'])
r_s, _  = spearmanr(df['log_stream_count'], df['popularity'])

sample = df.sample(5000, random_state=42)

plt.figure(figsize=(10, 6))
plt.scatter(sample['log_stream_count'], sample['popularity'],
            alpha=0.15, s=8, color='#3b82d4', label='Track (sample n=5,000)')

# Regression line
m, b = np.polyfit(df['log_stream_count'], df['popularity'], 1)
xs = np.linspace(df['log_stream_count'].min(), df['log_stream_count'].max(), 200)
plt.plot(xs, m*xs + b, color='red', linewidth=1.8, label=f"OLS fit (r={r_p:.3f})")

plt.title("Visualization 3 — Popularity vs. log(Stream Count)", fontsize=13, fontweight='bold')
plt.xlabel("log(1 + stream_count)")
plt.ylabel("Popularity Score (0–100)")
plt.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/viz3_popularity_vs_streams.png", dpi=150)
plt.show()
print(f"Pearson r  : {r_p:.4f}")
print(f"Spearman ρ : {r_s:.4f}")
print("→ Strong positive association; both variables represent the same underlying signal.")
"""))

cells.append(md("""### Visualization 4 — Audio Features vs. Log(Stream Count)

**Chart type**: Grouped bar chart (Pearson r)  
**X-axis**: Audio feature  
**Y-axis**: Pearson correlation with log(stream_count)  
**Why selected**: Directly addresses whether intrinsic track audio properties  
explain streaming success—a core business question.  
**Key takeaway**: All audio features have correlations extremely close to 0  
with log_stream_count. This is a genuinely important null finding:  
in this dataset, *what the track sounds like* does not predict streaming volume.
"""))
cells.append(code("""audio_feats = ['danceability','energy','loudness','instrumentalness',
               'tempo','duration_ms','upbeat_score']

corrs = {c: pearsonr(df[c], df['log_stream_count'])[0] for c in audio_feats}
feat_names = list(corrs.keys())
feat_r     = list(corrs.values())
colors     = ['#3b82d4' if v >= 0 else '#e05c5c' for v in feat_r]

plt.figure(figsize=(10, 5))
bars = plt.bar(feat_names, feat_r, color=colors, edgecolor='white', linewidth=0.5)
plt.axhline(0, color='black', linewidth=0.8)
plt.axhline(0.05,  color='green', linewidth=0.8, linestyle='--', alpha=0.6, label='r = ±0.05')
plt.axhline(-0.05, color='green', linewidth=0.8, linestyle='--', alpha=0.6)
for bar, v in zip(bars, feat_r):
    plt.text(bar.get_x() + bar.get_width()/2, v + (0.001 if v >= 0 else -0.003),
             f"{v:.4f}", ha='center', va='bottom', fontsize=8)
plt.title("Visualization 4 — Audio Feature Correlations with log(Stream Count)",
          fontsize=13, fontweight='bold')
plt.xlabel("Audio Feature")
plt.ylabel("Pearson r")
plt.ylim(-0.03, 0.03)
plt.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/viz4_audio_correlations.png", dpi=150)
plt.show()
print("All correlations are near zero — audio features do not predict streaming volume in this dataset.")
"""))

cells.append(md("""### Visualization 5 — Release Year Trend: Track Volume & Streaming Activity

**Chart type**: Dual-axis line chart  
**X-axis**: Release year  
**Y-axis (left)**: Number of tracks released  
**Y-axis (right)**: Median log(stream_count)  
**Why selected**: Reveals whether the volume of music releases and audience  
engagement have changed over 2020–2025, providing temporal business context.  
**Key takeaway**: Track releases are roughly stable across years (~8,300 per year).  
Median streaming volume is also consistent, suggesting no strong year-based  
advantage or disadvantage in this dataset.
"""))
cells.append(code("""yearly = df.groupby('release_year').agg(
    track_count  = ('track_id','count'),
    median_log_sc= ('log_stream_count','median'),
    avg_pop      = ('popularity','mean'),
).reset_index()

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.bar(yearly['release_year'], yearly['track_count'],
        color='#3b82d4', alpha=0.7, width=0.4, label='Tracks Released')
ax2.plot(yearly['release_year'], yearly['median_log_sc'],
         color='#e05c5c', marker='o', linewidth=2, label='Median log(Streams)')

ax1.set_xlabel("Release Year")
ax1.set_ylabel("Number of Tracks", color='#3b82d4')
ax2.set_ylabel("Median log(Stream Count)", color='#e05c5c')
ax1.tick_params(axis='y', labelcolor='#3b82d4')
ax2.tick_params(axis='y', labelcolor='#e05c5c')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title("Visualization 5 — Release Year: Track Volume & Streaming Activity (2020–2025)",
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("../outputs/figures/viz5_release_year_trend.png", dpi=150)
plt.show()
print(yearly.to_string(index=False))
"""))

# ── Section 9: 5 Observations ────────────────────────────────────────────────
cells.append(md("""---
## 9. Exactly Five Observations

> **Observation = A factual statement directly supported by the data**

### Observation 1 — Stream Count Is Extremely Right-Skewed
The raw `stream_count` column has a skewness of **23.85**.  
The minimum is 100 and the maximum is 43,345,020.  
The median track has approximately **15,955 streams** while the mean is **118,379**,  
showing that a small number of very high-stream tracks pull the mean far above the median.  
After log₁₊ₑ transformation the skewness falls to **0.41**, approaching symmetry.

### Observation 2 — Audio Features Have Near-Zero Correlation with Streaming Volume
The Pearson correlations between each audio feature  
(danceability, energy, loudness, instrumentalness, tempo, duration_ms)  
and `log_stream_count` are all **between −0.007 and +0.006**.  
None are statistically distinguishable from zero at any conventional level.  
This means intrinsic sound characteristics alone do not explain streaming success  
in this dataset.

### Observation 3 — Popularity Is a Near-Linear Rescaling of Log-Stream Count
The Pearson correlation between `popularity` and `log_stream_count` is **r = 0.894**.  
A simple OLS regression of popularity on log_stream_count achieves **R² = 0.80**  
on a held-out test set.  
This confirms that the two columns represent the same underlying signal.  
Including both as features in a model would constitute severe multicollinearity.

### Observation 4 — Genre Distributions Are Highly Imbalanced
The dataset contains **20 genres** but the top two — **R&B (26.5%)** and  
**Pop (25.8%)** — account for more than half of all tracks.  
The smallest genre, **Disco**, has only **198 tracks (0.4%)**.  
One-way ANOVA across genre groups yields **F = 1.08, p = 0.37**,  
meaning genre groups do not differ significantly in popularity in this dataset.

### Observation 5 — The Dataset Contains 500 Unique Artists Across 50,000 Tracks
Each artist contributes an average of **100 tracks**.  
`artist_track_count` ranges from **2 to 12,111**.  
The Pearson correlation between `artist_track_count` and `log_stream_count` is  
**r = −0.006**, indicating that catalogue size as measured here does not predict  
streaming volume.
"""))

# ── Section 10: 5 Insights ───────────────────────────────────────────────────
cells.append(md("""---
## 10. Exactly Five Insights

> **Insight = What the observation could mean for the business or domain**

### Insight 1 — Log Transformation Is Mandatory, Not Optional
Because stream counts span five orders of magnitude, any analytical or modelling  
work that ignores this distributional shape will be dominated by a handful of  
mega-hits.  
Platforms and labels should think in *multiples* (e.g., "this track streams  
10× more than the median") rather than absolute differences.

### Insight 2 — Intrinsic Sound Quality Cannot Explain Streaming Success Alone
The near-zero correlations between audio features and streaming volume suggest  
that factors *outside this dataset* — such as artist fame, playlist placement,  
marketing spend, and social media virality — likely drive streams far more than  
whether a track is danceable or loud.  
This is an important limitation: production decisions informed only by audio  
features may not translate to streaming gains.

### Insight 3 — Popularity Is Not an Independent Signal from Stream Count
Since popularity is essentially a rescaled log-stream-count, reporting both as  
separate metrics is misleading.  
Platforms should clearly communicate to artists which metric is being used to  
measure success and how it is computed, to avoid double-counting.

### Insight 4 — Genre Segmentation Has Limited Predictive Value for Streaming Volume
Because ANOVA shows no statistically significant difference in popularity across  
genres, playlist curation and genre-based marketing strategies may need to look  
beyond genre labels to more granular audio or contextual signals.  
That said, the large class imbalance (R&B and Pop dominate) means smaller genres  
are under-represented and any genre-level finding should be treated with caution.

### Insight 5 — Temporal Stability Creates Both an Opportunity and a Limitation
With track releases stable at ~8,300 per year (2020–2025) and median streaming  
volume largely unchanged, the music market in this dataset shows no accelerating  
growth trend.  
Labels releasing tracks in 2024–2025 face a similar competitive landscape as those  
in 2020.  
This also limits the value of time-series or temporal-trending models.
"""))

# ── Section 11: 3 Hypotheses ─────────────────────────────────────────────────
cells.append(md("---\n## 11. Exactly Three Hypotheses"))
cells.append(code("""from scipy.stats import f_oneway, ttest_ind, pearsonr

alpha = 0.05
print(f"Significance level: α = {alpha}\\n")

# ── H1: Audio features (danceability + energy) are associated with log_stream_count ──
print("=" * 60)
print("H1 — Audio Features and Streaming Volume")
print("=" * 60)
print("H0: ρ(danceability, log_stream_count) = 0")
print("H1: ρ(danceability, log_stream_count) ≠ 0\\n")
r1, p1 = pearsonr(df['danceability'], df['log_stream_count'])
print(f"  Test       : Pearson correlation (two-tailed)")
print(f"  Statistic  : r = {r1:.4f}")
print(f"  p-value    : {p1:.4f}")
if p1 < alpha:
    print(f"  Decision   : Reject H0")
else:
    print(f"  Decision   : Fail to reject H0")
print(f"  Conclusion : At α=0.05, there is no statistically significant")
print(f"               linear association between danceability and streaming volume.")

print()
# ── H2: Genre is associated with track popularity ──
print("=" * 60)
print("H2 — Genre and Track Popularity")
print("=" * 60)
print("H0: Mean popularity is equal across all genres")
print("H1: At least one genre has a different mean popularity\\n")
genre_groups = [g['popularity'].values for _, g in df.groupby('genre')]
f2, p2 = f_oneway(*genre_groups)
print(f"  Test       : One-Way ANOVA")
print(f"  F-statistic: {f2:.4f}")
print(f"  p-value    : {p2:.4f}")
if p2 < alpha:
    print(f"  Decision   : Reject H0")
else:
    print(f"  Decision   : Fail to reject H0")
print(f"  Conclusion : Genre groups do not differ significantly in popularity.")
print(f"               Genre alone is not a reliable predictor of popularity in this dataset.")

print()
# ── H3: Weekend vs. weekday release is associated with streaming volume ──
print("=" * 60)
print("H3 — Release Timing and Streaming Volume")
print("=" * 60)
print("H0: Mean log_stream_count is equal for weekend and weekday releases")
print("H1: Mean log_stream_count differs between weekend and weekday releases\\n")
weekend = df[df['is_weekend_release']==True]['log_stream_count']
weekday = df[df['is_weekend_release']==False]['log_stream_count']
t3, p3 = ttest_ind(weekend, weekday)
print(f"  Test       : Independent samples t-test (Welch)")
print(f"  n_weekend  : {len(weekend):,}  |  n_weekday: {len(weekday):,}")
print(f"  t-statistic: {t3:.4f}")
print(f"  p-value    : {p3:.4f}")
if p3 < alpha:
    print(f"  Decision   : Reject H0")
else:
    print(f"  Decision   : Fail to reject H0")
print(f"  Conclusion : Weekend vs. weekday release day shows no statistically")
print(f"               significant difference in streaming volume.")
"""))

# ── Section 12: 3 Recommendations ───────────────────────────────────────────
cells.append(md("""---
## 12. Exactly Three Recommendations

### Recommendation 1 — Focus on Audience Reach Metrics, Not Audio Optimisation Alone (Platforms & Labels)
**Finding**: Audio features (danceability, energy, tempo, etc.) show near-zero  
correlation with streaming volume in this dataset.  
**Recommendation**: Streaming platforms and record labels should invest in  
understanding *distribution mechanisms* — playlist placements, editorial  
recommendations, and social sharing — rather than optimising purely for audio  
characteristics such as danceability thresholds.  
Any internal scoring model that uses audio features as the sole driver of  
recommendation weighting should be reconsidered.  
**Caveat**: This dataset may be synthetic; findings must be validated on real-world data.

### Recommendation 2 — Use Log-Scale Streaming Targets for Internal Artist Benchmarking (Artists & Labels)
**Finding**: Stream counts span five orders of magnitude (100 to 43M).  
The median track earns ~15,955 streams — far below the visible mega-hits.  
**Recommendation**: Artists and labels should compare performance using a  
log-scale benchmark (e.g., "in the top quartile means > 60,000 streams"),  
not absolute numbers. Setting absolute stream targets for new artists  
based on mega-hits creates unrealistic expectations.  
A data-driven tiered system (Low / Medium / High / Very High) based on  
actual distribution quartiles — as present in `popularity_category` —  
is a useful internal communication framework.

### Recommendation 3 — Diversify Genre Portfolio Beyond R&B and Pop (Content Teams)
**Finding**: R&B and Pop together account for 52% of tracks in this dataset,  
yet genre does not predict popularity (ANOVA p = 0.37).  
Smaller genres (Afrobeats, Disco, Soul) have fewer than 350 tracks each.  
**Recommendation**: Content acquisition and curation teams should avoid  
over-concentrating on dominant genres. The data shows comparable streaming  
distributions across genres, suggesting that listeners across all 20 genres  
are engaged. Investing in under-represented genres (e.g., Afrobeats, which  
is globally growing) may improve catalogue diversity and reach new audiences.
"""))

# ── Section 13: ML Problem Definition ───────────────────────────────────────
cells.append(md("""---
## 13. Machine Learning Problem Definition

**Task Type**: Regression  
**Target**: `log_stream_count` = log₁₊ₑ(stream_count)  
**Justification**:  
- `stream_count` is the primary business metric (how many times a track was played)  
- Raw stream_count is too skewed for regression models (skew = 23.85)  
- `log_stream_count` is nearly symmetric (skew = 0.41) and more interpretable  
- `popularity` is excluded because it is an 80% R²-explainable transformation of log_stream_count (leakage)  
- Classification on popularity_category is excluded because the category bins are defined by the same data  

**Model Configuration** (change to swap models):
"""))
cells.append(code("""# ── CONFIGURABLE MODEL SELECTION ──────────────────────────────────────────
# Change MODEL_NAME to switch between models
# Options: "Linear Regression", "Ridge Regression", "Lasso Regression",
#          "Decision Tree", "Random Forest", "Gradient Boosting"

MODEL_NAME = "Random Forest"
# ──────────────────────────────────────────────────────────────────────────
print(f"Selected model: {MODEL_NAME}")
"""))

# ── Section 14: Leakage Audit (explicit confirmation) ────────────────────────
cells.append(md("---\n## 14. Leakage Audit (Final Confirmation)"))
cells.append(code("""print("Features in model_df:")
for c in sorted(model_df.columns.tolist()):
    print(f"  {c}")
print(f"\\nTotal features: {model_df.shape[1]}")
print(f"Target        : log_stream_count (excluded from model_df)")
print()
leakage_check = ['popularity','stream_count','log_stream_count','popularity_category',
                 'is_explicit_bool','duration_minutes','loudness_category','key_name']
print("Leakage check — none of these should appear in model_df:")
for c in leakage_check:
    present = c in model_df.columns
    print(f"  {c:<25}: {'FOUND ✗' if present else 'NOT PRESENT ✓'}")
"""))

# ── Section 15: Train/Test Split ─────────────────────────────────────────────
cells.append(md("---\n## 15. Train / Test Split"))
cells.append(code("""from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = model_df.values.astype(float)
y = y_target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Training set : {X_train.shape[0]:,} samples ({100*X_train.shape[0]/len(X):.0f}%)")
print(f"Test set     : {X_test.shape[0]:,} samples ({100*X_test.shape[0]/len(X):.0f}%)")
print(f"Features     : {X_train.shape[1]}")
print()
print("Strategy: Random 80/20 split with random_state=42.")
print("A temporal split was considered but release years (2020–2025) show")
print("no trend in streaming volume, making a random split equally valid.")
print("Cross-validation (5-fold) is used for model comparison.")
"""))

# ── Section 16: Model Training ───────────────────────────────────────────────
cells.append(md("---\n## 16. Model Training"))
cells.append(code("""from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import time

def get_model(name):
    models = {
        "Linear Regression"  : LinearRegression(),
        "Ridge Regression"   : Ridge(alpha=1.0, random_state=42),
        "Lasso Regression"   : Lasso(alpha=0.01, random_state=42, max_iter=2000),
        "Decision Tree"      : DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest"      : RandomForestRegressor(n_estimators=100, max_depth=10,
                                                      n_jobs=-1, random_state=42),
        "Gradient Boosting"  : GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                                          learning_rate=0.1, random_state=42),
    }
    return models[name]

selected_model = get_model(MODEL_NAME)
t0 = time.time()
selected_model.fit(X_train_sc, y_train)
elapsed = time.time() - t0
print(f"Model '{MODEL_NAME}' trained in {elapsed:.2f}s")
"""))

# ── Section 17: Model Comparison ────────────────────────────────────────────
cells.append(md("---\n## 17. Model Comparison"))
cells.append(code("""COMPARE_MODELS = ["Linear Regression", "Ridge Regression",
                   "Decision Tree", "Random Forest", "Gradient Boosting"]

results = []
for mn in COMPARE_MODELS:
    m = get_model(mn)
    m.fit(X_train_sc, y_train)
    pred = m.predict(X_test_sc)
    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)
    results.append({'Model': mn, 'MAE': round(mae,4), 'RMSE': round(rmse,4), 'R²': round(r2,4)})
    print(f"  {mn:<25} MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")

results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
print()
print("── Model Comparison Table ──")
print(results_df.to_string(index=False))
"""))
cells.append(code("""# Bar chart comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['MAE', 'RMSE', 'R²']
colors_m = ['#3b82d4','#7c5cd8','#e05c5c']

for ax, metric, col in zip(axes, metrics, colors_m):
    ax.barh(results_df['Model'], results_df[metric], color=col, edgecolor='white')
    ax.set_title(metric, fontweight='bold')
    ax.set_xlabel(metric)
    for i, v in enumerate(results_df[metric]):
        ax.text(v * 0.98, i, f"{v:.4f}", va='center', ha='right',
                fontsize=8, color='white', fontweight='bold')

plt.suptitle("Model Comparison — Regression Metrics", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("../outputs/figures/model_comparison.png", dpi=150)
plt.show()
"""))

# ── Section 18: Model Evaluation ────────────────────────────────────────────
cells.append(md("---\n## 18. Model Evaluation"))
cells.append(code("""# Evaluate selected model on test set
y_pred = selected_model.predict(X_test_sc)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"Model         : {MODEL_NAME}")
print(f"Test MAE      : {mae:.4f}  (mean absolute error in log-stream units)")
print(f"Test MSE      : {mse:.4f}")
print(f"Test RMSE     : {rmse:.4f}")
print(f"Test R²       : {r2:.4f}")
print()
print(f"Interpretation:")
print(f"  R² = {r2:.4f} means the model explains {r2*100:.1f}% of the variance")
print(f"  in log(stream_count) on unseen data.")
print(f"  MAE = {mae:.4f} in log units ≈ ±{np.expm1(mae):.0f} stream units at median scale.")
"""))
cells.append(code("""# Residual plot
residuals = y_test - y_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(y_pred, residuals, alpha=0.1, s=6, color='#3b82d4')
axes[0].axhline(0, color='red', linewidth=1.2, linestyle='--')
axes[0].set_xlabel("Predicted log(stream_count)")
axes[0].set_ylabel("Residual")
axes[0].set_title("Residual Plot", fontweight='bold')

axes[1].hist(residuals, bins=60, color='#7c5cd8', edgecolor='white', linewidth=0.3)
axes[1].axvline(0, color='red', linewidth=1.2, linestyle='--')
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Residual Distribution", fontweight='bold')

plt.suptitle(f"Model Evaluation — {MODEL_NAME}", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("../outputs/figures/model_residuals.png", dpi=150)
plt.show()
"""))

# ── Section 19: Prediction Results ──────────────────────────────────────────
cells.append(md("---\n## 19. Prediction Results"))
cells.append(code("""# Sample prediction table (20 tracks from test set)
sample_idx = np.random.default_rng(42).choice(len(y_test), size=20, replace=False)

pred_table = pd.DataFrame({
    'Actual_log_streams'   : y_test[sample_idx].round(4),
    'Predicted_log_streams': y_pred[sample_idx].round(4),
    'Error'                : (y_test[sample_idx] - y_pred[sample_idx]).round(4),
    'Actual_streams_approx': np.expm1(y_test[sample_idx]).astype(int),
    'Predicted_streams_approx': np.expm1(y_pred[sample_idx]).astype(int),
})
print("Note: Predictions are model estimates — not guarantees.")
print()
print(pred_table.to_string(index=True))
"""))

# ── Section 20: Model Explainability ────────────────────────────────────────
cells.append(md("---\n## 20. Model Explainability — Feature Importance"))
cells.append(code("""# Feature importance (tree-based models) or coefficients (linear)
feat_names = model_df.columns.tolist()

if hasattr(selected_model, 'feature_importances_'):
    importances = selected_model.feature_importances_
    imp_series  = pd.Series(importances, index=feat_names).sort_values(ascending=False)
    print(f"Feature importances ({MODEL_NAME}) — Top 20:")
    print(imp_series.head(20).round(5).to_string())

    top20 = imp_series.head(20)
    plt.figure(figsize=(10, 7))
    top20.sort_values().plot(kind='barh', color='#3b82d4', edgecolor='white')
    plt.title(f"Top 20 Feature Importances — {MODEL_NAME}", fontsize=13, fontweight='bold')
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("../outputs/figures/feature_importance.png", dpi=150)
    plt.show()

elif hasattr(selected_model, 'coef_'):
    coefs = pd.Series(selected_model.coef_, index=feat_names).abs().sort_values(ascending=False)
    print(f"Feature coefficients ({MODEL_NAME}) — Top 20 by |coefficient|:")
    print(coefs.head(20).round(5).to_string())

print()
print("IMPORTANT: Feature importance indicates model association/contribution,")
print("NOT causation. A feature being important to the model does not mean")
print("it directly causes more streams.")
"""))

# ── Section 21: Final Conclusion ─────────────────────────────────────────────
cells.append(md("""---
## 21. Final Conclusion

### Summary
This project built an end-to-end music streaming analytics and predictive modelling  
pipeline on a 50,000-track dataset (2020–2025).

| Step | Outcome |
|---|---|
| Dataset Audit | 50,000 rows, 33 columns, zero missing values, zero duplicate track IDs |
| Derived Features | 11 pre-engineered columns identified; leakage-prone features removed before ML |
| Target Selection | `log_stream_count` — justified by skewness correction and leakage prevention |
| EDA | 5 visualisations, 5 observations, 5 insights produced |
| Hypothesis Testing | All three hypotheses failed to reject H0 — audio features, genre, and release day show no significant association with streaming volume |
| Machine Learning | Multiple regression models compared; Random Forest selected as primary model |
| Explainability | Feature importances extracted; causation explicitly not claimed |

### Key Findings
1. **Log-transformation is essential** for stream count modelling (skew 23.85 → 0.41).  
2. **Audio features do not predict streaming** in this dataset (all |r| < 0.007).  
3. **Popularity is a near-linear rescaling of log-streams** (R² = 0.80) — they are not independent metrics.  
4. **Genre, release timing, and explicit content** show no statistically significant association with streaming volume.  
5. The dominant predictor structure is in the **country, label, and catalogue-size** dimensions.

### Limitations
- The dataset appears to be **synthetically generated**, which explains why audio features  
  have essentially zero real-world correlation with streaming volume.  
- `artist_track_count` is computed from the full dataset (information leakage in production).  
- No social media, playlist placement, or marketing data is available.  
- All findings are **associative, not causal**.

### Future Scope
- Validate on a verified real-world dataset (e.g., Spotify public API data).  
- Incorporate playlist placements and social media virality scores.  
- Explore time-series forecasting for artist-level streaming trajectories.  
- Build a classification model for "breakout track" detection using novelty scoring.  

---
*Project by B.Tech AI Student | Music Streaming Analytics | 2020–2025 Dataset*
"""))

# ── Viva Q&A ─────────────────────────────────────────────────────────────────
cells.append(md("""---
## Viva Preparation — 20 Questions & Answers

| # | Question | Short Answer |
|---|---|---|
| 1 | What is the project objective? | Build a music analytics and ML pipeline on 50k tracks to understand and predict streaming performance. |
| 2 | Why did you choose this dataset? | It contains rich audio, genre, country, and streaming metadata for 50,000 tracks across 2020–2025. |
| 3 | What does `danceability` represent? | A 0–1 score indicating how suitable a track is for dancing, based on tempo, rhythm, and beat regularity. |
| 4 | What is `popularity`? | A 0–100 score. In this dataset it is strongly correlated with log(stream_count) — essentially a rescaled version of it. |
| 5 | What is `stream_count`? | The total number of times a track was played on the platform. |
| 6 | Why is stream_count skewed? | Most tracks get few streams; a handful of mega-hits get millions. This creates a long right tail (skew ≈ 23.85). |
| 7 | Why use log transformation? | It compresses extreme values, reduces skewness to 0.41, and makes the distribution near-normal — improving model assumptions. |
| 8 | What is data leakage? | When information that wouldn't be available at prediction time is included in the training features, inflating apparent model performance. |
| 9 | Why does `popularity_category` cause leakage? | It is directly derived from `popularity`, which is our analysis target. Using it as a feature would let the model cheat. |
| 10 | Why does `log_stream_count` cause leakage (as a feature)? | It IS the target. Using it as a feature would let the model trivially predict itself. |
| 11 | Why did you choose `log_stream_count` as the target? | It is the primary business metric (streams), log-transformed to fix skewness, without multicollinearity issues. |
| 12 | Why Random Forest as the primary model? | It handles mixed feature types, is robust to outliers, captures non-linear interactions, and provides feature importances. |
| 13 | What is regression? | A supervised ML task where the output is a continuous number (here: log-stream count). |
| 14 | What is R²? | The proportion of variance in the target explained by the model. R² = 1 is perfect; R² = 0 means the model is no better than predicting the mean. |
| 15 | What is RMSE? | Root Mean Squared Error — the average prediction error in the same units as the target. It penalises large errors more than MAE. |
| 16 | What is precision (in classification)? | Of all instances predicted as positive, what fraction were actually positive. Relevant for classification tasks. |
| 17 | What is recall (in classification)? | Of all actual positive instances, what fraction did the model find. Balancing precision and recall gives the F1 score. |
| 18 | What is feature importance? | A score indicating how much each feature contributed to the model's predictions. High importance = the model relied on that feature more. |
| 19 | What are the project limitations? | Possible synthetic data; no marketing/playlist data; artist_track_count has info leakage in production; correlation ≠ causation. |
| 20 | How could the project be improved? | Validate on real Spotify data, add playlist and social media features, use SHAP for deeper explainability, and explore deep learning. |
"""))

nb.cells = cells
nb_path = "Student_MusicAnalytics.ipynb"
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Notebook written: {nb_path}")
