"""
src/preprocessing.py
────────────────────
Data loading, cleaning, and model dataset preparation.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ── Constants ─────────────────────────────────────────────────────────────────
LEAKAGE_COLUMNS = [
    "popularity_category",   # derived from popularity → target leakage
    "stream_count",          # raw form of target
    "popularity",            # near-linear rescaling of log_stream_count
    "is_explicit_bool",      # identical to 'explicit'
    "duration_minutes",      # duplicate of duration_ms (unit change only)
    "loudness_category",     # binned version of loudness
    "key_name",              # human-readable duplicate of 'key'
    "upbeat_score",          # composite of danceability + energy
    "log_stream_count",      # the target itself — excluded from feature set
]

IDENTIFIER_COLUMNS = [
    "track_id", "track_name", "artist_name", "album_name", "release_date",
    "mode",                  # kept as mode_name (categorical string)
]

CATEGORICAL_COLUMNS = [
    "genre", "country", "label",
    "release_day_of_week", "release_quarter", "mode_name",
]


def find_dataset(base_dir: str = None) -> str:
    """Locate the dataset CSV relative to a base directory."""
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "..", "data", "music_dataset.csv"),
        os.path.join(base_dir, "..", "..", "data", "music_dataset.csv"),
        os.path.join(base_dir, "..", "..", "spotify_artist_streaming_2020_2025.csv"),
        "data/music_dataset.csv",
        "spotify_artist_streaming_2020_2025.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "Dataset not found. Place music_dataset.csv in the data/ folder."
    )


def load_data(path: str = None) -> pd.DataFrame:
    """
    Load the raw music streaming dataset.

    Parameters
    ----------
    path : str, optional
        Explicit path to the CSV. If None, auto-detected via find_dataset().

    Returns
    -------
    pd.DataFrame
        Raw dataframe with release_date parsed as datetime.
    """
    if path is None:
        path = find_dataset()
    df = pd.read_csv(path, parse_dates=["release_date"])
    return df


def audit(df: pd.DataFrame) -> dict:
    """
    Perform a dataset audit and return a summary dict.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
        Keys: shape, missing, duplicates, unique_track_ids, unique_artists,
              unique_genres, unique_countries, unique_labels, year_range,
              stream_skew, popularity_skew
    """
    return {
        "shape"           : df.shape,
        "missing"         : df.isnull().sum().to_dict(),
        "duplicate_rows"  : int(df.duplicated().sum()),
        "duplicate_ids"   : int(df["track_id"].duplicated().sum()),
        "unique_artists"  : int(df["artist_name"].nunique()),
        "unique_genres"   : int(df["genre"].nunique()),
        "unique_countries": int(df["country"].nunique()),
        "unique_labels"   : int(df["label"].nunique()),
        "year_range"      : (int(df["release_year"].min()), int(df["release_year"].max())),
        "stream_skew"     : round(float(df["stream_count"].skew()), 4),
        "popularity_skew" : round(float(df["popularity"].skew()), 4),
    }


def verify_derived_columns(df: pd.DataFrame) -> dict:
    """
    Verify that pre-engineered derived columns are consistent with their sources.

    Returns a dict of {check_name: bool}.
    """
    results = {}
    results["release_month_ok"]   = bool((df["release_date"].dt.month == df["release_month"]).all())
    results["release_year_ok"]    = bool((df["release_date"].dt.year  == df["release_year"]).all())
    quarter_map = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}
    computed_q  = df["release_date"].dt.quarter.map(quarter_map)
    results["release_quarter_ok"] = bool((computed_q == df["release_quarter"]).all())
    is_wk = df["release_date"].dt.dayofweek >= 5
    results["is_weekend_ok"]      = bool((is_wk == df["is_weekend_release"]).all())
    dur_min = df["duration_ms"] / 60000
    results["duration_minutes_ok"]= bool(np.allclose(dur_min, df["duration_minutes"], atol=0.01))
    log_sc  = np.log1p(df["stream_count"])
    results["log_stream_count_ok"]= bool(np.allclose(log_sc,  df["log_stream_count"], atol=0.01))
    results["is_explicit_bool_ok"]= bool((df["explicit"] == df["is_explicit_bool"]).all())
    return results


def build_model_dataset(df: pd.DataFrame):
    """
    Create a clean modelling dataset.

    Removes leaking, redundant, and identifier columns.
    One-hot encodes categorical features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw (cleaned) dataframe.

    Returns
    -------
    model_df : pd.DataFrame
        Feature matrix (X).
    y : np.ndarray
        Target array — log_stream_count.
    feature_names : list[str]
        Column names of model_df.
    """
    to_remove = LEAKAGE_COLUMNS + IDENTIFIER_COLUMNS
    mdf = df.drop(columns=to_remove, errors="ignore").copy()

    # mode_name was dropped with IDENTIFIER (mode numeric); re-attach from df
    if "mode_name" not in mdf.columns and "mode_name" in df.columns:
        mdf["mode_name"] = df["mode_name"].values

    # One-hot encode categoricals
    encode_cols = [c for c in CATEGORICAL_COLUMNS if c in mdf.columns]
    mdf = pd.get_dummies(mdf, columns=encode_cols, drop_first=True)

    # Ensure all values are numeric
    mdf = mdf.apply(pd.to_numeric, errors="coerce").fillna(0)

    y = df["log_stream_count"].values
    return mdf, y, mdf.columns.tolist()


def prepare_train_test(model_df: pd.DataFrame, y: np.ndarray,
                        test_size: float = 0.20, random_state: int = 42):
    """
    Split into train/test and apply standard scaling.

    Returns
    -------
    X_train_sc, X_test_sc, y_train, y_test, scaler
    """
    X = model_df.values.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return X_train_sc, X_test_sc, y_train, y_test, scaler
