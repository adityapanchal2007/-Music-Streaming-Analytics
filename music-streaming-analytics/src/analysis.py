"""
src/analysis.py
───────────────
Statistical analysis helpers: hypothesis tests, group comparisons,
correlation summaries.
"""
import pandas as pd
import numpy as np
from scipy import stats


def correlation_summary(df: pd.DataFrame, target: str,
                         features: list) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations for a list of features against a target.

    Returns
    -------
    pd.DataFrame with columns: Feature, Pearson_r, Pearson_p, Spearman_rho, Spearman_p
    """
    rows = []
    for feat in features:
        if feat not in df.columns:
            continue
        pear_r, pear_p   = stats.pearsonr(df[feat].dropna(), df[target].dropna())
        spear_r, spear_p = stats.spearmanr(df[feat], df[target], nan_policy="omit")
        rows.append({
            "Feature"   : feat,
            "Pearson_r" : round(pear_r, 4),
            "Pearson_p" : round(pear_p, 6),
            "Spearman_ρ": round(spear_r, 4),
            "Spearman_p": round(spear_p, 6),
        })
    return pd.DataFrame(rows).sort_values("Pearson_r", key=abs, ascending=False)


def anova_test(df: pd.DataFrame, group_col: str, value_col: str) -> dict:
    """
    One-way ANOVA to test if group means differ.

    Returns
    -------
    dict: groups, F_stat, p_value, significant (bool at alpha=0.05)
    """
    groups = [g[value_col].dropna().values
              for _, g in df.groupby(group_col)]
    f_stat, p_val = stats.f_oneway(*groups)
    return {
        "group_col" : group_col,
        "value_col" : value_col,
        "n_groups"  : len(groups),
        "F_stat"    : round(f_stat, 4),
        "p_value"   : round(p_val, 6),
        "significant": bool(p_val < 0.05),
    }


def ttest_two_groups(df: pd.DataFrame, group_col: str, value_col: str,
                      group_true, group_false) -> dict:
    """
    Independent-samples Welch t-test between two groups.

    Returns
    -------
    dict: group_col, value_col, n_a, n_b, t_stat, p_value, significant
    """
    a = df[df[group_col] == group_true][value_col].dropna().values
    b = df[df[group_col] == group_false][value_col].dropna().values
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    return {
        "group_col"  : group_col,
        "value_col"  : value_col,
        "group_a"    : str(group_true),
        "group_b"    : str(group_false),
        "n_a"        : len(a),
        "n_b"        : len(b),
        "t_stat"     : round(t_stat, 4),
        "p_value"    : round(p_val, 6),
        "significant": bool(p_val < 0.05),
    }


def group_summary(df: pd.DataFrame, group_col: str,
                   value_cols: list) -> pd.DataFrame:
    """
    Compute count, mean, median, std for value_cols grouped by group_col.
    """
    agg_dict = {}
    for vc in value_cols:
        agg_dict[f"{vc}_count"]  = (vc, "count")
        agg_dict[f"{vc}_mean"]   = (vc, "mean")
        agg_dict[f"{vc}_median"] = (vc, "median")
        agg_dict[f"{vc}_std"]    = (vc, "std")
    return df.groupby(group_col).agg(**agg_dict).round(4)


def stream_distribution_stats(df: pd.DataFrame) -> dict:
    """
    Compute key streaming distribution statistics.
    """
    sc = df["stream_count"]
    lsc = df["log_stream_count"]
    return {
        "raw_min"       : int(sc.min()),
        "raw_max"       : int(sc.max()),
        "raw_mean"      : round(float(sc.mean()), 2),
        "raw_median"    : round(float(sc.median()), 2),
        "raw_std"       : round(float(sc.std()), 2),
        "raw_skew"      : round(float(sc.skew()), 4),
        "raw_q25"       : round(float(sc.quantile(0.25)), 2),
        "raw_q75"       : round(float(sc.quantile(0.75)), 2),
        "log_skew"      : round(float(lsc.skew()), 4),
        "log_mean"      : round(float(lsc.mean()), 4),
        "log_median"    : round(float(lsc.median()), 4),
    }
