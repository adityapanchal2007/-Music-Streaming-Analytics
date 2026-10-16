"""
Music Streaming Analytics & Popularity Intelligence Dashboard
============================================================
Run:  streamlit run app/app.py
"""

import os, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Music Streaming Analytics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.kpi-card {
    background: #f7f8fa;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 18px 12px;
    text-align: center;
}
.kpi-value { font-size: 28px; font-weight: 700; color: #1f2328; margin: 0; }
.kpi-label { font-size: 13px; color: #57606a; margin: 0; }
.section-header {
    font-size: 20px; font-weight: 700; color: #1f2328;
    border-bottom: 2px solid #3b82d4; padding-bottom: 6px;
    margin-top: 28px; margin-bottom: 18px;
}
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    _here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(_here, "..", "data", "music_dataset.csv"),
        os.path.join(_here, "..", "..", "spotify_artist_streaming_2020_2025.csv"),
        "data/music_dataset.csv",
        "spotify_artist_streaming_2020_2025.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=["release_date"])
            return df
    st.error("Dataset not found. Place music_dataset.csv in the data/ folder.")
    st.stop()

df_full = load_data()

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.title("🎛️ Filters")
st.sidebar.markdown("---")

all_genres   = sorted(df_full["genre"].unique())
all_countries= sorted(df_full["country"].unique())
all_years    = sorted(df_full["release_year"].unique())
all_quarters = sorted(df_full["release_quarter"].unique())
all_modes    = sorted(df_full["mode_name"].unique())
all_pop_cats = ["Low", "Medium", "High", "Very High"]

sel_genres = st.sidebar.multiselect("Genre", all_genres,
                                     default=all_genres[:5] if len(all_genres) > 5 else all_genres)
sel_years  = st.sidebar.slider("Release Year", int(min(all_years)), int(max(all_years)),
                                (int(min(all_years)), int(max(all_years))))
sel_countries = st.sidebar.multiselect("Country (top 10 default)",
                                        all_countries,
                                        default=all_countries[:10])
sel_explicit = st.sidebar.radio("Explicit Content", ["All", "Non-Explicit Only", "Explicit Only"])
sel_mode     = st.sidebar.multiselect("Mode", all_modes, default=all_modes)
sel_pop_cat  = st.sidebar.multiselect("Popularity Category", all_pop_cats, default=all_pop_cats)

# ── Apply filters ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def filter_data(genres, years, countries, explicit, modes, pop_cats):
    d = df_full.copy()
    if genres:       d = d[d["genre"].isin(genres)]
    if countries:    d = d[d["country"].isin(countries)]
    if modes:        d = d[d["mode_name"].isin(modes)]
    if pop_cats:     d = d[d["popularity_category"].isin(pop_cats)]
    d = d[(d["release_year"] >= years[0]) & (d["release_year"] <= years[1])]
    if explicit == "Non-Explicit Only": d = d[d["explicit"] == False]
    elif explicit == "Explicit Only":   d = d[d["explicit"] == True]
    return d

df = filter_data(
    tuple(sel_genres), tuple(sel_years), tuple(sel_countries),
    sel_explicit, tuple(sel_mode), tuple(sel_pop_cat)
)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🎵 Music Streaming Analytics & Popularity Intelligence")
st.markdown("**Explore • Analyze • Predict** &nbsp;|&nbsp; Dataset: 50,000 tracks · 2020–2025")
st.markdown(f"*Showing **{len(df):,}** tracks after current filters*")
st.markdown("---")

if len(df) == 0:
    st.warning("No tracks match the current filter selection. Adjust filters in the sidebar.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Key Performance Indicators</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
kpis = [
    (k1, f"{len(df):,}", "Total Tracks"),
    (k2, f"{df['artist_name'].nunique():,}", "Unique Artists"),
    (k3, f"{df['genre'].nunique()}", "Genres"),
    (k4, f"{df['popularity'].mean():.1f}", "Avg Popularity"),
    (k5, f"{int(df['stream_count'].median()):,}", "Median Streams"),
    (k6, f"{df['danceability'].mean():.3f}", "Avg Danceability"),
    (k7, f"{df['energy'].mean():.3f}", "Avg Energy"),
]
for col, val, lbl in kpis:
    col.markdown(
        f'<div class="kpi-card"><p class="kpi-value">{val}</p>'
        f'<p class="kpi-label">{lbl}</p></div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
tab_eda, tab_genre, tab_audio, tab_geo, tab_ml = st.tabs([
    "📈 Stream Distribution",
    "🎼 Genre Analysis",
    "🎹 Audio Features",
    "🌍 Country / Label",
    "🤖 ML Prediction",
])

# ── TAB 1: Stream Distribution ────────────────────────────────────────────────
with tab_eda:
    st.markdown('<div class="section-header">Stream Count Distribution</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.histogram(
            df, x="log_stream_count", nbins=60,
            title="Distribution of log(Stream Count)",
            labels={"log_stream_count": "log(1 + stream_count)"},
            color_discrete_sequence=["#3b82d4"],
        )
        fig1.add_vline(x=df["log_stream_count"].median(), line_dash="dash",
                       line_color="orange", annotation_text="Median")
        fig1.update_layout(bargap=0.05, height=380)
        st.plotly_chart(fig1, use_container_width=True)
        st.caption(f"Skewness of log(stream_count): {df['log_stream_count'].skew():.3f}")

    with c2:
        pop_dist = df["popularity_category"].value_counts().reset_index()
        pop_dist.columns = ["Category", "Count"]
        cat_order = ["Low", "Medium", "High", "Very High"]
        pop_dist["Category"] = pd.Categorical(pop_dist["Category"], categories=cat_order, ordered=True)
        pop_dist = pop_dist.sort_values("Category")
        fig2 = px.bar(pop_dist, x="Category", y="Count",
                      title="Tracks by Popularity Category",
                      color="Category",
                      color_discrete_map={"Low":"#aab4c4","Medium":"#3b82d4",
                                          "High":"#7c5cd8","Very High":"#e05c5c"})
        fig2.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Popularity vs. Log(Stream Count)**")
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig3 = px.scatter(sample, x="log_stream_count", y="popularity",
                      color="genre", opacity=0.4, size_max=4,
                      title="Popularity vs. log(Stream Count) — coloured by Genre",
                      labels={"log_stream_count": "log(1+stream_count)", "popularity": "Popularity"},
                      height=420)
    fig3.update_traces(marker=dict(size=4))
    st.plotly_chart(fig3, use_container_width=True)
    r_val = df["popularity"].corr(df["log_stream_count"])
    st.info(f"Pearson r = **{r_val:.4f}** — Popularity and log(stream_count) are strongly "
             f"associated (they measure the same underlying signal).")

    st.markdown("**Release Year Trend**")
    yearly = df.groupby("release_year").agg(
        Tracks=("track_id","count"),
        Median_log_streams=("log_stream_count","median"),
        Avg_popularity=("popularity","mean"),
    ).reset_index()
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=yearly["release_year"], y=yearly["Tracks"],
                           name="Track Count", marker_color="#3b82d4", yaxis="y"))
    fig4.add_trace(go.Scatter(x=yearly["release_year"], y=yearly["Median_log_streams"],
                               name="Median log(Streams)", mode="lines+markers",
                               line=dict(color="#e05c5c", width=2), yaxis="y2"))
    fig4.update_layout(
        title="Release Year: Track Volume & Median Streaming Activity",
        yaxis=dict(title="Track Count"),
        yaxis2=dict(title="Median log(Streams)", overlaying="y", side="right"),
        height=380, legend=dict(orientation="h"),
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── TAB 2: Genre Analysis ─────────────────────────────────────────────────────
with tab_genre:
    st.markdown('<div class="section-header">Genre Analysis</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        gc = df["genre"].value_counts().reset_index()
        gc.columns = ["Genre", "Tracks"]
        fig_gc = px.bar(gc, x="Tracks", y="Genre", orientation="h",
                        title="Tracks per Genre",
                        color="Tracks", color_continuous_scale="Blues",
                        height=480)
        fig_gc.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(fig_gc, use_container_width=True)

    with col_b:
        gp = df.groupby("genre")["log_stream_count"].median().reset_index()
        gp.columns = ["Genre", "Median_log_streams"]
        gp = gp.sort_values("Median_log_streams", ascending=False)
        fig_gp = px.bar(gp, x="Genre", y="Median_log_streams",
                        title="Median log(Streams) by Genre",
                        color="Median_log_streams", color_continuous_scale="Purples",
                        height=480)
        fig_gp.update_layout(xaxis_tickangle=40, showlegend=False)
        st.plotly_chart(fig_gp, use_container_width=True)

    st.markdown("**Popularity Distribution by Genre (Box Plot)**")
    fig_box = px.box(df, x="genre", y="popularity", color="genre",
                     title="Popularity Distribution by Genre",
                     height=420)
    fig_box.update_layout(xaxis_tickangle=40, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    from scipy.stats import f_oneway
    groups = [g["popularity"].values for _, g in df.groupby("genre")]
    if len(groups) > 1:
        f_stat, p_val = f_oneway(*groups)
        if p_val < 0.05:
            st.success(f"ANOVA: F={f_stat:.3f}, p={p_val:.4f} — Genres differ significantly in popularity.")
        else:
            st.info(f"ANOVA: F={f_stat:.3f}, p={p_val:.4f} — No statistically significant "
                    f"difference in popularity across genres (α=0.05).")

# ── TAB 3: Audio Features ─────────────────────────────────────────────────────
with tab_audio:
    st.markdown('<div class="section-header">Audio Feature Analysis</div>',
                unsafe_allow_html=True)

    audio_cols = ["danceability", "energy", "loudness", "instrumentalness",
                  "tempo", "duration_ms", "upbeat_score"]
    corr_vals = {c: df[c].corr(df["log_stream_count"]) for c in audio_cols}
    corr_df   = pd.DataFrame(list(corr_vals.items()), columns=["Feature", "r_with_log_streams"])
    corr_df["color"] = corr_df["r_with_log_streams"].apply(
        lambda x: "#3b82d4" if x >= 0 else "#e05c5c")

    fig_corr = px.bar(
        corr_df.sort_values("r_with_log_streams"),
        x="r_with_log_streams", y="Feature", orientation="h",
        title="Pearson Correlation: Audio Features vs. log(Stream Count)",
        color="r_with_log_streams",
        color_continuous_scale=["#e05c5c","#ffffff","#3b82d4"],
        range_color=[-0.02, 0.02],
        height=400,
    )
    fig_corr.add_vline(x=0, line_color="black", line_width=1)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.warning("All audio features show near-zero correlation (|r| < 0.01) with streaming volume. "
               "Intrinsic sound characteristics do not explain streaming success in this dataset.")

    c3, c4 = st.columns(2)
    sel_feat = c3.selectbox("Select Audio Feature", audio_cols)
    sample2  = df.sample(min(4000, len(df)), random_state=99)
    with c3:
        fig_sc = px.scatter(sample2, x=sel_feat, y="log_stream_count", opacity=0.25,
                            trendline="ols",
                            title=f"{sel_feat} vs. log(Stream Count)",
                            color_discrete_sequence=["#3b82d4"], height=380)
        st.plotly_chart(fig_sc, use_container_width=True)
    with c4:
        fig_hist = px.histogram(df, x=sel_feat, nbins=50, color="mode_name",
                                barmode="overlay", opacity=0.7,
                                title=f"Distribution of {sel_feat} by Mode",
                                color_discrete_sequence=["#3b82d4","#7c5cd8"],
                                height=380)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("**Audio Feature Correlation Matrix**")
    corr_matrix = df[["danceability","energy","loudness","instrumentalness",
                       "tempo","popularity","log_stream_count"]].corr().round(3)
    fig_heat = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu_r",
                          title="Audio Feature Correlation Matrix", height=480)
    st.plotly_chart(fig_heat, use_container_width=True)

# ── TAB 4: Country / Label ────────────────────────────────────────────────────
with tab_geo:
    st.markdown('<div class="section-header">Country & Label Analysis</div>',
                unsafe_allow_html=True)

    country_stats = df.groupby("country").agg(
        Tracks=("track_id","count"),
        Avg_popularity=("popularity","mean"),
        Median_streams=("stream_count","median"),
    ).reset_index().sort_values("Tracks", ascending=False)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_ctry = px.bar(country_stats.head(15), x="country", y="Tracks",
                          title="Top 15 Countries by Track Count",
                          color="Avg_popularity", color_continuous_scale="Blues",
                          height=420)
        fig_ctry.update_layout(xaxis_tickangle=40)
        st.plotly_chart(fig_ctry, use_container_width=True)
    with col_g2:
        fig_cpop = px.bar(country_stats.head(15).sort_values("Avg_popularity", ascending=False),
                          x="country", y="Avg_popularity",
                          title="Top 15 Countries by Avg Popularity",
                          color="Avg_popularity", color_continuous_scale="Purples",
                          height=420)
        fig_cpop.update_layout(xaxis_tickangle=40)
        st.plotly_chart(fig_cpop, use_container_width=True)

    label_stats = df.groupby("label").agg(
        Tracks=("track_id","count"),
        Avg_popularity=("popularity","mean"),
        Median_streams=("stream_count","median"),
    ).reset_index().sort_values("Avg_popularity", ascending=False)

    st.markdown("**Label Analysis**")
    l1, l2 = st.columns(2)
    with l1:
        fig_lbl = px.bar(label_stats.sort_values("Tracks", ascending=False).head(15),
                          x="label", y="Tracks",
                          title="Top 15 Labels by Track Count",
                          color="Tracks", color_continuous_scale="Teal",
                          height=420)
        fig_lbl.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_lbl, use_container_width=True)
    with l2:
        fig_lpop = px.box(df, x="explicit", y="log_stream_count",
                           color="explicit", title="Explicit vs. Non-Explicit — log(Streams)",
                           labels={"explicit":"Explicit","log_stream_count":"log(stream_count)"},
                           color_discrete_map={True:"#e05c5c", False:"#3b82d4"},
                           height=420)
        st.plotly_chart(fig_lpop, use_container_width=True)

# ── TAB 5: ML Prediction ─────────────────────────────────────────────────────
with tab_ml:
    st.markdown('<div class="section-header">🤖 Machine Learning — Stream Count Prediction</div>',
                unsafe_allow_html=True)

    st.markdown("""
**Target**: `log(stream_count)` — log-transformed stream count  
**Why log?**: Raw stream_count is highly right-skewed (skew ≈ 23.85). Log transformation  
produces a near-symmetric distribution (skew ≈ 0.41), improving model stability.  
**Leakage prevention**: `popularity`, `popularity_category`, `stream_count`,  
`is_explicit_bool`, `duration_minutes`, `loudness_category`, and `key_name`  
are all excluded from features.
""")

    # Model selection
    ml_model_name = st.selectbox(
        "Select Model",
        ["Random Forest", "Gradient Boosting", "Decision Tree",
         "Ridge Regression", "Linear Regression"],
    )

    @st.cache_resource(show_spinner="Training model…")
    def train_model(model_name, data_hash):
        _df = df_full.copy()
        REMOVE = ['popularity_category','stream_count','popularity','is_explicit_bool',
                  'duration_minutes','loudness_category','key_name','upbeat_score',
                  'log_stream_count','track_id','track_name','artist_name',
                  'album_name','release_date','mode']
        mdf = _df.drop(columns=REMOVE, errors='ignore')
        mdf['mode_name'] = _df['mode_name']
        mdf = pd.get_dummies(mdf, columns=['genre','country','label',
                                            'release_day_of_week','release_quarter','mode_name'],
                             drop_first=True)
        X = mdf.values.astype(float)
        y = _df['log_stream_count'].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        sc = StandardScaler()
        Xtr_sc = sc.fit_transform(Xtr)
        Xte_sc = sc.transform(Xte)
        model_map = {
            "Random Forest"     : RandomForestRegressor(n_estimators=100, max_depth=10,
                                                         n_jobs=-1, random_state=42),
            "Gradient Boosting" : GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                                              learning_rate=0.1, random_state=42),
            "Decision Tree"     : DecisionTreeRegressor(max_depth=8, random_state=42),
            "Ridge Regression"  : Ridge(alpha=1.0, random_state=42),
            "Linear Regression" : LinearRegression(),
        }
        m = model_map[model_name]
        m.fit(Xtr_sc, ytr)
        ypred = m.predict(Xte_sc)
        metrics = {
            "MAE" : round(mean_absolute_error(yte, ypred), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(yte, ypred))), 4),
            "R²"  : round(r2_score(yte, ypred), 4),
        }
        # Feature importance / coefficients
        feat_names = mdf.columns.tolist()
        if hasattr(m, 'feature_importances_'):
            imp = pd.Series(m.feature_importances_, index=feat_names).sort_values(ascending=False)
        elif hasattr(m, 'coef_'):
            imp = pd.Series(np.abs(m.coef_), index=feat_names).sort_values(ascending=False)
        else:
            imp = None
        sample_preds = pd.DataFrame({
            'Actual_log_streams'   : yte[:20].round(3),
            'Predicted_log_streams': ypred[:20].round(3),
            'Error'                : (yte[:20] - ypred[:20]).round(3),
        })
        return m, sc, mdf.columns.tolist(), metrics, imp, sample_preds

    # Use a simple hash to avoid retraining unnecessarily
    with st.spinner("Training model on full dataset (80/20 split)…"):
        model, scaler_ml, feature_cols, metrics, feat_imp, sample_preds = train_model(
            ml_model_name, "v1"
        )

    # Metrics
    m1, m2, m3 = st.columns(3)
    for col, label, val in [(m1,"MAE",metrics["MAE"]),
                             (m2,"RMSE",metrics["RMSE"]),
                             (m3,"R²",metrics["R²"])]:
        col.markdown(
            f'<div class="kpi-card"><p class="kpi-value">{val}</p>'
            f'<p class="kpi-label">{label} (Test Set)</p></div>',
            unsafe_allow_html=True)
    st.markdown(f"*R² = {metrics['R²']} means the model explains "
                f"{metrics['R²']*100:.1f}% of variance in log(stream_count) on unseen tracks.*")
    st.caption("Note: Predictions are model estimates, not guarantees.")

    # Feature importance chart
    if feat_imp is not None:
        st.markdown("**Top 20 Feature Importances**")
        top20 = feat_imp.head(20).reset_index()
        top20.columns = ["Feature", "Importance"]
        fig_fi = px.bar(top20.sort_values("Importance"), x="Importance", y="Feature",
                        orientation="h", title=f"Feature Importances — {ml_model_name}",
                        color="Importance", color_continuous_scale="Blues", height=520)
        st.plotly_chart(fig_fi, use_container_width=True)
        st.caption("Feature importance indicates model contribution, not causation.")

    # Sample predictions table
    st.markdown("**Sample Predictions (first 20 test tracks)**")
    st.dataframe(sample_preds, use_container_width=True)

    # ── Manual Prediction Panel ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🎯 Manual Track Prediction</div>',
                unsafe_allow_html=True)
    st.markdown("Enter track characteristics below to estimate log(stream_count).")
    st.warning("This prediction is a model estimate based on training data patterns — not a guarantee.")

    with st.form("predict_form"):
        p1c, p2c, p3c = st.columns(3)
        dance  = p1c.slider("Danceability",   0.0, 1.0, 0.65, 0.01)
        energy = p1c.slider("Energy",         0.0, 1.0, 0.70, 0.01)
        loud   = p2c.slider("Loudness (dB)",  -20.0, 0.0, -7.0, 0.1)
        instr  = p2c.slider("Instrumentalness", 0.0, 1.0, 0.01, 0.01)
        tempo  = p3c.slider("Tempo (BPM)",    60.0, 220.0, 120.0, 1.0)
        dur_ms = p3c.number_input("Duration (ms)", 60000, 400000, 200000, 5000)

        g1c, g2c, g3c, g4c = st.columns(4)
        sel_g  = g1c.selectbox("Genre",   sorted(df_full["genre"].unique()))
        sel_c  = g2c.selectbox("Country", sorted(df_full["country"].unique()))
        sel_l  = g3c.selectbox("Label",   sorted(df_full["label"].unique()))
        sel_yr = g4c.selectbox("Release Year", sorted(df_full["release_year"].unique(), reverse=True))
        sel_q  = g1c.selectbox("Quarter", ["Q1","Q2","Q3","Q4"])
        sel_dow= g2c.selectbox("Day of Week",
                                ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        sel_mo = g3c.selectbox("Mode", ["Major","Minor"])
        sel_k  = g4c.slider("Key (0–11)", 0, 11, 5)
        expl_v = g1c.checkbox("Explicit", value=False)
        atc    = g2c.number_input("Artist Track Count", 1, 15000, 100)
        rel_mo = g3c.slider("Release Month", 1, 12, 6)
        is_wk  = g4c.checkbox("Weekend Release", value=False)

        submitted = st.form_submit_button("Predict Stream Count")

    if submitted:
        # Build a single-row df matching model_df schema
        input_row = {
            'duration_ms'    : dur_ms,
            'danceability'   : dance,
            'energy'         : energy,
            'key'            : sel_k,
            'loudness'       : loud,
            'instrumentalness': instr,
            'tempo'          : tempo,
            'release_year'   : sel_yr,
            'release_month'  : rel_mo,
            'explicit'       : expl_v,
            'artist_track_count': atc,
            'is_weekend_release': is_wk,
        }
        # Build dummies matching training
        input_df = pd.DataFrame([input_row])
        # Add dummy cols for genre, country, label, release_day_of_week, release_quarter, mode_name
        for combo_col, val_chosen, prefix in [
            ('genre', sel_g, 'genre'),
            ('country', sel_c, 'country'),
            ('label', sel_l, 'label'),
            ('release_day_of_week', sel_dow, 'release_day_of_week'),
            ('release_quarter', sel_q, 'release_quarter'),
            ('mode_name', sel_mo, 'mode_name'),
        ]:
            input_df[combo_col] = val_chosen

        input_enc = pd.get_dummies(input_df,
                                    columns=['genre','country','label',
                                             'release_day_of_week','release_quarter','mode_name'],
                                    drop_first=False)
        # Align to training feature set
        for col in feature_cols:
            if col not in input_enc.columns:
                input_enc[col] = 0
        input_enc = input_enc[feature_cols].astype(float)
        input_sc  = scaler_ml.transform(input_enc)
        log_pred  = model.predict(input_sc)[0]
        raw_pred  = int(np.expm1(log_pred))

        st.success(f"**Predicted log(stream_count): {log_pred:.4f}**  →  "
                   f"Approx. **{raw_pred:,} streams**")
        pop_approx = max(0, min(100, int(round(log_pred * 8.7 - 3.5))))
        st.info(f"Estimated popularity range: approx. **{max(0,pop_approx-5)}–{min(100,pop_approx+5)}**")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small style='color:#57606a'>Music Streaming Analytics · B.Tech AI Project · "
    "Dataset: 50,000 Tracks 2020–2025 · Built with Streamlit + Plotly + scikit-learn</small></center>",
    unsafe_allow_html=True
)
