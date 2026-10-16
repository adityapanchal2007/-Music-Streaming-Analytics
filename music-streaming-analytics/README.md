# Music Streaming Analytics & Popularity Intelligence

> **Explore • Analyze • Predict**  
> B.Tech Artificial Intelligence — Data Analytics & Machine Learning Project

---

## Project Title

**Music Streaming Analytics & Popularity Intelligence**

A complete, end-to-end data analytics, statistical analysis, predictive machine learning,
and interactive dashboard project built on a 50,000-track music streaming dataset (2020–2025).

---

## Overview

This project investigates the characteristics of music tracks — audio features, genre,
country, label, and release metadata — and their relationship with streaming performance.
It includes a full data cleaning pipeline, exploratory data analysis (EDA),
hypothesis testing, machine learning regression modelling, model explainability,
and an interactive Streamlit dashboard.

---

## Problem Statement

Music platforms, record labels, and artists need to understand which characteristics
of a track are associated with streaming volume.
This project builds a data-driven system that:

1. Audits and cleans the dataset
2. Detects and handles pre-engineered/derived feature leakage
3. Performs rigorous EDA with exactly 5 visualisations, 5 observations, and 5 insights
4. Tests 3 statistical hypotheses
5. Makes 3 actionable recommendations
6. Trains and compares multiple regression models to predict `log(stream_count)`
7. Delivers an interactive web dashboard

---

## Objectives

| Type | Objective |
|---|---|
| Analytical | Identify patterns in genre, audio features, country, and release metadata |
| Statistical | Test 3 hypotheses on associations with streaming performance |
| ML | Predict `log(stream_count)` using track audio and metadata features |
| Business | Provide actionable insights for artists, labels, and platforms |

---

## Dataset Source

| Property | Details |
|---|---|
| Platform | Kaggle |
| Dataset Name | Spotify Artist Streaming Dataset 2020–2025 |
| Kaggle URL | **[KAGGLE LINK REQUIRED]** |
| Local File | `data/music_dataset.csv` |

> **Note:** Replace `[KAGGLE LINK REQUIRED]` with the actual Kaggle URL (e.g. `https://www.kaggle.com/datasets/<username>/<dataset-slug>`).

---

## Dataset Description

| Property | Value |
|---|---|
| File | `spotify_artist_streaming_2020_2025.csv` |
| Rows | 50,000 |
| Columns | 33 |
| Time range | 2020–2025 |
| Unique artists | 500 |
| Unique genres | 20 |
| Unique countries | 30 |
| Unique labels | 30 |

---

## Dataset Dimensions

**50,000 rows × 33 columns**

---

## Features

### Original Features (used in analysis)
| Feature | Type | Description |
|---|---|---|
| `track_id` | string | Unique track identifier (UUID) |
| `track_name` | string | Track title |
| `artist_name` | string | Artist name |
| `album_name` | string | Album title |
| `release_date` | date | Track release date |
| `genre` | categorical | Music genre (20 categories) |
| `duration_ms` | integer | Track duration in milliseconds |
| `popularity` | integer | 0–100 popularity score |
| `danceability` | float | 0–1 danceability score |
| `energy` | float | 0–1 energy level |
| `key` | integer | Musical key (0–11) |
| `loudness` | float | Average loudness in dB |
| `mode` | integer | 1 = Major, 0 = Minor |
| `instrumentalness` | float | 0–1 instrumental content score |
| `tempo` | float | Tempo in BPM |
| `stream_count` | integer | Total stream count |
| `country` | categorical | Track's primary country (30 countries) |
| `explicit` | boolean | Explicit content flag |
| `label` | categorical | Record label (30 labels) |

### Derived/Pre-engineered Features
| Feature | Derived From | ML Status |
|---|---|---|
| `release_year` | `release_date` | KEEP |
| `release_month` | `release_date` | KEEP |
| `release_day_of_week` | `release_date` | KEEP |
| `duration_minutes` | `duration_ms` | REMOVE (redundant) |
| `popularity_category` | `popularity` | REMOVE (leakage risk) |
| `loudness_category` | `loudness` | REMOVE (redundant) |
| `key_name` | `key` | REMOVE (redundant) |
| `mode_name` | `mode` | KEEP (categorical encoding) |
| `is_explicit_bool` | `explicit` | REMOVE (identical) |
| `release_quarter` | `release_date` | KEEP |
| `is_weekend_release` | `release_date` | KEEP |
| `log_stream_count` | `stream_count` | **TARGET** |
| `upbeat_score` | `danceability + energy` | REMOVE (composite) |
| `artist_track_count` | dataset-level count | KEEP (disclosed limitation) |

---

## Data Cleaning

- ✅ Zero missing values
- ✅ Zero duplicate rows / duplicate `track_id`s
- ✅ `release_date` parsed as datetime
- ✅ All derived columns verified against source columns — all consistent
- ✅ Outliers investigated (stream_count, popularity, duration) — all valid observations
- ✅ Data types confirmed correct

---

## Exploratory Analysis

### Key Findings

1. `stream_count` is extremely right-skewed (skewness = 23.85) → log transformation mandatory
2. Audio features (danceability, energy, loudness) have near-zero correlation with log(streams) (|r| < 0.01)
3. `popularity` and `log_stream_count` are strongly correlated (r = 0.894) → same underlying signal
4. Genre does not significantly predict popularity (ANOVA F=1.08, p=0.37)
5. Release day-of-week has no significant effect on streaming (t-test p=0.90)

---

## Five Visualizations

| # | Title | Chart Type | Key Takeaway |
|---|---|---|---|
| 1 | Stream Count: Before and After Log Transformation | Dual histogram | Log transform converts skew 23.85 → 0.41 |
| 2 | log(Stream Count) by Genre | Box plot | Median streams are broadly similar across genres |
| 3 | Popularity vs. log(Stream Count) | Scatter + OLS | r = 0.894 — same underlying signal |
| 4 | Audio Feature Correlations with log(Streams) | Bar chart | All |r| < 0.007 — audio features don't predict streams |
| 5 | Release Year: Track Volume & Streaming Activity | Dual-axis line/bar | Stable production and streaming 2020–2025 |

---

## Five Observations

1. Raw stream_count skewness = 23.85; median track has ~15,955 streams vs. mean of 118,379
2. All audio feature correlations with log_stream_count are between −0.007 and +0.006
3. Popularity has R² = 0.80 when regressed on log_stream_count alone
4. Genre is highly imbalanced: R&B (26.5%) and Pop (25.8%) account for 52% of tracks
5. 500 unique artists across 50,000 tracks; artist_track_count ranges from 2 to 12,111

---

## Five Insights

1. Log transformation is mandatory — mega-hits skew all absolute statistics
2. Intrinsic audio characteristics do not explain streaming success in this dataset
3. Popularity and stream_count are not independent metrics — both measure the same thing
4. Genre segmentation has limited predictive value; distribution mechanics likely matter more
5. Temporal stability (2020–2025) limits time-series trend models

---

## Three Hypotheses

| # | H0 | Test | Result |
|---|---|---|---|
| H1 | danceability has no linear association with log_stream_count | Pearson r | Fail to reject H0 (p = 0.80) |
| H2 | Mean popularity is equal across all genres | One-way ANOVA | Fail to reject H0 (p = 0.37) |
| H3 | Weekend and weekday releases have equal mean log_stream_count | Welch t-test | Fail to reject H0 (p = 0.90) |

All three hypotheses fail to reject the null at α = 0.05.

---

## Three Recommendations

1. **Focus on distribution mechanisms, not audio optimisation** — platforms and labels should
   invest in playlist placement and marketing, not purely audio feature targeting.
2. **Use log-scale benchmarking for artist performance tracking** — absolute stream targets
   create unrealistic expectations given extreme distributional skew.
3. **Diversify genre portfolio beyond R&B and Pop** — comparable streaming distributions
   across genres suggest under-represented genres (Afrobeats, Disco, Soul) are viable investments.

---

## Machine Learning

| Property | Value |
|---|---|
| Task | Regression |
| Target | `log_stream_count` = log₁₊ₑ(stream_count) |
| Train/Test | 80% / 20%, random_state=42 |
| Leakage prevention | popularity, popularity_category, stream_count, is_explicit_bool, duration_minutes, loudness_category, key_name excluded |

---

## Model Evaluation

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | *computed from data* | *computed from data* | *computed from data* |
| Gradient Boosting | *computed from data* | *computed from data* | *computed from data* |
| Decision Tree | *computed from data* | *computed from data* | *computed from data* |
| Ridge Regression | *computed from data* | *computed from data* | *computed from data* |
| Linear Regression | *computed from data* | *computed from data* | *computed from data* |

> Exact values computed at runtime in the notebook. Random Forest is the primary model.

---

## Dashboard

**Interactive Streamlit dashboard** with:
- Sidebar filters (Genre, Year, Country, Mode, Popularity Category, Explicit)
- KPI cards (Total Tracks, Artists, Genres, Avg Popularity, Median Streams, Danceability, Energy)
- Tab 1: Stream Distribution (histograms, trend chart, popularity scatter)
- Tab 2: Genre Analysis (bar charts, box plots, ANOVA result)
- Tab 3: Audio Features (correlation bar chart, scatter, heatmap)
- Tab 4: Country & Label (bar charts, explicit comparison)
- Tab 5: ML Prediction (model comparison, feature importances, manual prediction panel)

---

## Technologies

| Library | Purpose |
|---|---|
| pandas | Data loading, cleaning, manipulation |
| numpy | Numerical computation |
| matplotlib / seaborn | Static visualisations |
| scipy | Statistical hypothesis tests |
| scikit-learn | ML models, preprocessing, evaluation |
| plotly | Interactive charts |
| streamlit | Web dashboard |
| nbformat | Notebook generation |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Dashboard

```bash
streamlit run app/app.py
```

---

## Running the Notebook

```bash
cd notebooks
python build_notebook.py    # generates Student_MusicAnalytics.ipynb
jupyter notebook Student_MusicAnalytics.ipynb
```

---

## Project Structure

```
music-streaming-analytics/
│
├── data/
│   └── music_dataset.csv              ← 50,000 tracks × 33 columns
│
├── notebooks/
│   ├── build_notebook.py              ← Generates the .ipynb
│   └── Student_MusicAnalytics.ipynb   ← Main Jupyter notebook (21 sections)
│
├── app/
│   └── app.py                         ← Streamlit interactive dashboard
│
├── src/
│   ├── preprocessing.py               ← Data loading, cleaning, model dataset
│   ├── analysis.py                    ← Statistical analysis helpers
│   ├── visualization.py               ← Matplotlib/Seaborn chart functions
│   └── model.py                       ← ML training, evaluation, explainability
│
├── outputs/
│   ├── figures/                       ← Saved chart PNGs
│   └── cleaned_data/                  ← Cleaned dataset exports
│
├── models/                            ← Saved model artifacts
│
├── requirements.txt
└── README.md
```

---

## Limitations

1. Dataset appears to be **synthetically generated** — audio features show near-zero real-world correlation with streaming, which is atypical of genuine Spotify data.
2. `artist_track_count` is computed from the full dataset — it would constitute information leakage in a real production scenario (new artists would have count = 0 at release time).
3. No playlist placement, marketing spend, social media, or listener demographic data available.
4. All findings are **observational/associative — not causal**.
5. Country-level differences may reflect dataset composition rather than genuine geographical effects.

---

## Future Scope

- Validate on a verified real-world dataset via the Spotify Web API
- Incorporate playlist placement and chart performance features
- Add social media virality scores (TikTok, Instagram, YouTube)
- Explore time-series forecasting for artist-level streaming trajectories
- Build a "breakout track" binary classifier using anomaly/novelty detection
- Deploy dashboard to Streamlit Cloud or IBM Cloud

---

*B.Tech Artificial Intelligence | Music Streaming Analytics Project | Dataset: 2020–2025*
