# ============================================================
# STREAMLIT DASHBOARD — Renewable Energy Load Balancing
# Full 2015–2020 range with flexible filtering
# Run: streamlit run dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib, os, warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title = "⚡ Energy Load Balancing",
    page_icon  = "⚡",
    layout     = "wide",
)

st.markdown("""
<style>
  .main { background-color: #0f1117; }
  .metric-card {
    background: #1a1d26; border-radius: 12px;
    padding: 1rem 1.2rem; border: 1px solid #2e3145;
    text-align: center;
  }
  .metric-val { font-size: 26px; font-weight: 700; }
  .metric-lbl { font-size: 11px; color: #8b8fa8; margin-top: 4px; }
  .intervention-box {
    border-radius: 10px; padding: 0.8rem 1.1rem;
    margin-bottom: 0.5rem; font-weight: 500;
  }
</style>
""", unsafe_allow_html=True)

COLORS = {
    'solar': '#F5A623', 'wind': '#4FC3F7',
    'load':  '#EF5350', 'gap_d': '#FF7043', 'gap_s': '#4FC3F7',
}
LEVEL_COLORS = {
    'BALANCED':        ('#66BB6A', '#0a1f0e'),
    'MODERATE':        ('#FFA726', '#1f180a'),
    'HIGH':            ('#FF7043', '#1f120a'),
    'CRITICAL':        ('#EF5350', '#1f0a0a'),
    'SURPLUS':         ('#29B6F6', '#0a1520'),
    'HIGH SURPLUS':    ('#7E57C2', '#120a1f'),
    'CRITICAL SURPLUS':('#546E7A', '#0a0f12'),
}
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_full():
    """Load the full 2015-2020 gap dataset."""
    paths = [
        'outputs/full_gap_2015_2020.csv',
        r'C:\Users\abhil\OneDrive\Desktop\project\outputs\full_gap_2015_2020.csv',
    ]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df.sort_index()
    return None

@st.cache_data
def load_interventions():
    """Fallback: load the smaller gap_interventions.csv."""
    paths = [
        'outputs/gap_interventions.csv',
        r'C:\Users\abhil\OneDrive\Desktop\project\outputs\gap_interventions.csv',
    ]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df.sort_index()
    return None

@st.cache_data
def load_demand_preds():
    paths = [
        'demand_predictions.csv',
        r'C:\Users\abhil\OneDrive\Desktop\project\demand_predictions.csv',
    ]
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df.sort_index()
    return None

# Try full dataset first, fall back to interventions CSV
df_full = load_full()
df_int  = load_interventions()
dmd     = load_demand_preds()

# Use whichever has more data
if df_full is not None and (df_int is None or len(df_full) >= len(df_int)):
    df = df_full
    data_source = "full_gap_2015_2020.csv"
elif df_int is not None:
    df = df_int
    data_source = "gap_interventions.csv"
else:
    df = None
    data_source = "none"


# ============================================================
# HEADER
# ============================================================
st.markdown("## ⚡ Renewable Energy Load Balancing Dashboard")
st.markdown("**Germany · OPSD Dataset · AI-Powered Gap Analysis**")
st.markdown("---")

if df is None:
    st.warning("⚠️ No data found. Run scripts in order:")
    st.code("python eda.py\npython model_demand_forecast.py\n"
            "python model_supply_gap.py\npython generate_full_gap.py\n"
            "streamlit run dashboard.py")
    st.stop()


# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.header("🔧 Date Filter")

ALL = "All"
years_avail = sorted(df.index.year.unique().tolist())

sel_year = st.sidebar.selectbox(
    "Year",
    [ALL] + years_avail,
    index=0,                   # default = All
)

if sel_year == ALL:
    sel_month = ALL
else:
    months_avail = sorted(df[df.index.year == int(sel_year)]
                          .index.month.unique().tolist())
    sel_month = st.sidebar.selectbox(
        "Month",
        [ALL] + months_avail,
        format_func=lambda x: x if x == ALL else MONTH_NAMES[int(x)-1],
        index=0,
    )

# Apply filter
if sel_year == ALL:
    filtered = df.copy()
    filter_label = "All Years (2015 → 2019)"
    resample_freq = 'W'        # weekly avg for full range
elif sel_month == ALL:
    filtered = df[df.index.year == int(sel_year)].copy()
    filter_label = f"Full Year {sel_year}"
    resample_freq = 'D'        # daily avg for single year
else:
    filtered = df[
        (df.index.year  == int(sel_year)) &
        (df.index.month == int(sel_month))
    ].copy()
    filter_label = f"{MONTH_NAMES[int(sel_month)-1]} {sel_year}"
    resample_freq = None       # hourly for single month

st.sidebar.success(f"📅 {filter_label}  ({len(filtered):,} rows)")
st.sidebar.markdown("---")

# ── Quantile-based thresholds (computed from full dataset) ────
@st.cache_data
def compute_thresholds(gap_series: pd.Series) -> dict:
    """
    Compute intervention thresholds from the actual gap distribution.
    Deficit thresholds = percentiles of positive gaps.
    Surplus thresholds = percentiles of negative gaps.
    Falls back to reasonable defaults if data is too sparse.
    """
    pos = gap_series[gap_series >  0]   # deficit hours
    neg = gap_series[gap_series <  0]   # surplus hours

    thr = {}

    # Deficit side — split positive gaps into 4 bands
    if len(pos) >= 10:
        thr['low_def']  = int(pos.quantile(0.25))   # mild deficit
        thr['high_def'] = int(pos.quantile(0.65))   # notable deficit
        thr['crit_def'] = int(pos.quantile(0.90))   # severe deficit
    else:
        thr['low_def']  =  500
        thr['high_def'] = 3000
        thr['crit_def'] = 8000

    # Surplus side — split negative gaps into 4 bands
    if len(neg) >= 10:
        thr['low_sur']  = int(neg.quantile(0.75))   # mild surplus
        thr['high_sur'] = int(neg.quantile(0.35))   # notable surplus
        thr['crit_sur'] = int(neg.quantile(0.10))   # severe surplus
    else:
        thr['low_sur']  =  -500
        thr['high_sur'] = -3000
        thr['crit_sur'] = -8000

    return thr

thr = compute_thresholds(df['pred_gap'])

st.sidebar.markdown("**Intervention Thresholds (MW)**")
st.sidebar.caption("Auto-computed from data quantiles — override if needed.")

# Show computed values and allow manual override via sliders
gap_min = int(df['pred_gap'].min())
gap_max = int(df['pred_gap'].max())

thr_crit = st.sidebar.slider(
    "Critical deficit (p90 of deficits)",
    min_value = max(0,         gap_max // 10),
    max_value = max(gap_max,   1),
    value     = thr['crit_def'],
    step      = max(100, (gap_max - gap_max // 10) // 50),
    help      = f"Auto value: {thr['crit_def']:,} MW  (90th percentile of deficit hours)"
)

thr_high = st.sidebar.slider(
    "High deficit (p65 of deficits)",
    min_value = max(0,         gap_max // 20),
    max_value = thr_crit,
    value     = min(thr['high_def'], thr_crit),
    step      = max(100, thr_crit // 50),
    help      = f"Auto value: {thr['high_def']:,} MW  (65th percentile of deficit hours)"
)

# Show all threshold values as an info table
st.sidebar.markdown(
    f"""
    | Level | Threshold |
    |---|---|
    | 🔴 Critical | > `{thr_crit:,}` MW |
    | 🟠 High | > `{thr_high:,}` MW |
    | 🟡 Moderate | > `{thr['low_def']:,}` MW |
    | 🟢 Balanced | ± `{thr['low_def']:,}` MW |
    | 🔵 Surplus | < `{thr['low_sur']:,}` MW |
    | 🟣 High Surplus | < `{thr['high_sur']:,}` MW |
    | ⚫ Crit. Surplus | < `{thr['crit_sur']:,}` MW |
    """
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Data source: `{data_source}`")
st.sidebar.caption(f"Full range: {df.index.min().date()} → {df.index.max().date()}")
st.sidebar.caption(
    f"Gap distribution: "
    f"min {gap_min:,} MW · "
    f"median {int(df['pred_gap'].median()):,} MW · "
    f"max {gap_max:,} MW"
)


# ============================================================
# LIVE THRESHOLD RECOMPUTATION
# The CSV stores pre-computed labels — but slider changes must
# recompute levels in real time so the UI actually responds.
# This runs on EVERY Streamlit rerun (i.e. every slider move).
# ============================================================

def recommend_live(gap_mw: float,
                   crit_def: float, high_def: float, low_def: float,
                   low_sur:  float, high_sur: float, crit_sur: float) -> tuple:
    """Classify a gap value using the CURRENT slider thresholds."""
    if   gap_mw >  crit_def:
        return ('CRITICAL',        '🔴 Emergency import + backup generators')
    elif gap_mw >  high_def:
        return ('HIGH',            '🟠 Demand response + gas peakers')
    elif gap_mw >  low_def:
        return ('MODERATE',        '🟡 Minor demand response')
    elif gap_mw >= low_sur:
        return ('BALANCED',        '🟢 Grid balanced — no action needed')
    elif gap_mw >= high_sur:
        return ('SURPLUS',         '🔵 Charge storage / export')
    elif gap_mw >= crit_sur:
        return ('HIGH SURPLUS',    '🟣 Curtail wind + charge all storage')
    else:
        return ('CRITICAL SURPLUS','⚫ Emergency curtailment')

# Recompute every time sliders move
# thr_crit / thr_high come from sliders; others from quantile dict
_args = dict(
    crit_def = thr_crit,
    high_def = thr_high,
    low_def  = thr['low_def'],
    low_sur  = thr['low_sur'],
    high_sur = thr['high_sur'],
    crit_sur = thr['crit_sur'],
)

live_results = [recommend_live(g, **_args) for g in filtered['pred_gap']]
filtered = filtered.copy()                             # avoid SettingWithCopyWarning
filtered['live_level']  = [r[0] for r in live_results]
filtered['live_action'] = [r[1] for r in live_results]

# Same for full df (used in Tab 4 gap patterns)
full_results = [recommend_live(g, **_args) for g in df['pred_gap']]
df = df.copy()
df['live_level'] = [r[0] for r in full_results]


# ============================================================
# KPI CARDS
# ============================================================
c1, c2, c3, c4, c5, c6 = st.columns(6)

avg_gap     = filtered['pred_gap'].mean()
pct_deficit = (filtered['pred_gap'] > 0).mean() * 100
pct_surplus = (filtered['pred_gap'] <= 0).mean() * 100
avg_solar   = filtered['pred_solar'].mean()
avg_wind    = filtered['pred_wind'].mean()
avg_demand  = filtered['pred_demand'].mean()

def kpi(col, val, label, color='#F5A623'):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-val" style="color:{color}">{val}</div>
      <div class="metric-lbl">{label}</div>
    </div>""", unsafe_allow_html=True)

kpi(c1, f"{avg_gap:+,.0f}",    "Avg Gap (MW)",     '#FF7043' if avg_gap > 0 else '#4FC3F7')
kpi(c2, f"{avg_demand:,.0f}",  "Avg Demand (MW)",  COLORS['load'])
kpi(c3, f"{avg_solar:,.0f}",   "Avg Solar (MW)",   COLORS['solar'])
kpi(c4, f"{avg_wind:,.0f}",    "Avg Wind (MW)",    COLORS['wind'])
kpi(c5, f"{pct_deficit:.1f}%", "Hours Deficit",    '#FF7043')
kpi(c6, f"{pct_surplus:.1f}%", "Hours Surplus",    '#4FC3F7')

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Time Series",
    "🔴 Interventions",
    "📊 Model Performance",
    "🔍 Gap Patterns",
    "📅 Year Overview",
])


# ── TAB 1: Time Series ────────────────────────────────────────
with tab1:
    st.subheader(f"Supply vs Demand — {filter_label}")

    # Resample for readability on larger ranges
    if resample_freq:
        plot_data = filtered.resample(resample_freq).mean(numeric_only=True)
        freq_label = "weekly avg" if resample_freq == 'W' else "daily avg"
    else:
        plot_data = filtered
        freq_label = "hourly"

    st.caption(f"Showing {freq_label} data  ·  {len(plot_data):,} data points")

    fig, axes = plt.subplots(2, 1, figsize=(15, 8),
                             facecolor='#0f1117', sharex=True)

    axes[0].fill_between(plot_data.index, plot_data['pred_solar'],
                         alpha=0.65, color=COLORS['solar'], label='Solar')
    axes[0].fill_between(plot_data.index,
                         plot_data['pred_solar'] + plot_data['pred_wind'],
                         plot_data['pred_solar'],
                         alpha=0.55, color=COLORS['wind'], label='Wind')
    axes[0].plot(plot_data.index, plot_data['pred_demand'],
                 color=COLORS['load'], lw=1.5, label='Demand (predicted)')
    if 'actual_demand' in plot_data.columns:
        axes[0].plot(plot_data.index, plot_data['actual_demand'],
                     color='white', lw=0.8, linestyle='--',
                     alpha=0.5, label='Demand (actual)')

    axes[1].fill_between(plot_data.index, plot_data['pred_gap'],
                         where=(plot_data['pred_gap'] > 0).values,
                         color=COLORS['gap_d'], alpha=0.75, label='Deficit')
    axes[1].fill_between(plot_data.index, plot_data['pred_gap'],
                         where=(plot_data['pred_gap'] <= 0).values,
                         color=COLORS['gap_s'], alpha=0.55, label='Surplus')
    axes[1].axhline(0,        color='white',   lw=0.8, alpha=0.4)
    axes[1].axhline( thr_high, color='#FF7043', lw=0.8, linestyle=':', alpha=0.6,
                    label=f'High deficit ({thr_high:,} MW)')
    axes[1].axhline( thr_crit, color='#EF5350', lw=0.8, linestyle=':', alpha=0.6,
                    label=f'Critical deficit ({thr_crit:,} MW)')
    axes[1].axhline( thr['low_sur'],  color='#29B6F6', lw=0.8, linestyle=':', alpha=0.6,
                    label=f'Surplus ({thr["low_sur"]:,} MW)')
    axes[1].axhline( thr['high_sur'], color='#7E57C2', lw=0.8, linestyle=':', alpha=0.5,
                    label=f'High surplus ({thr["high_sur"]:,} MW)')

    for ax in axes:
        ax.set_facecolor('#1a1d26')
        ax.tick_params(colors='#8b8fa8')
        ax.grid(True, color='#2e3145', linestyle='--', alpha=0.5)
        ax.spines[:].set_color('#2e3145')

    axes[0].set_ylabel('MW', color='#c8cad4')
    axes[0].set_title(f'Supply Stack vs Demand  ({freq_label})',
                      color='white', pad=6)
    axes[0].legend(fontsize=8, facecolor='#1a1d26', labelcolor='#c8cad4')

    axes[1].set_ylabel('Gap (MW)', color='#c8cad4')
    axes[1].set_title('Supply–Demand Gap  (positive = deficit  |  negative = surplus)',
                      color='white', pad=6)
    axes[1].legend(fontsize=8, facecolor='#1a1d26', labelcolor='#c8cad4')

    fig.patch.set_facecolor('#0f1117')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ── TAB 2: Interventions ─────────────────────────────────────
with tab2:
    st.subheader(f"🔴 Intervention Log — {filter_label}")

    left, right = st.columns([3, 1])

    with right:
        st.markdown("**Level Summary**")
        lv_counts = filtered['live_level'].value_counts()
        for lvl, cnt in lv_counts.items():
            fg, bg = LEVEL_COLORS.get(lvl, ('#fff', '#222'))
            pct = cnt / len(filtered) * 100
            st.markdown(f"""
            <div class="intervention-box"
                 style="background:{bg}; border-left:4px solid {fg};">
              <span style="color:{fg}">{lvl}</span>
              <span style="float:right; color:#c8cad4">
                {cnt:,} h &nbsp;({pct:.1f}%)
              </span>
            </div>""", unsafe_allow_html=True)

        # Pie chart
        fig_p, ax_p = plt.subplots(figsize=(4, 4), facecolor='#1a1d26')
        ax_p.set_facecolor('#1a1d26')
        pie_colors = [LEVEL_COLORS.get(l, ('#fff','#222'))[0]
                      for l in lv_counts.index]
        ax_p.pie(lv_counts.values, labels=lv_counts.index,
                 colors=pie_colors, autopct='%1.1f%%',
                 textprops={'fontsize': 8, 'color': 'white'},
                 startangle=90,
                 wedgeprops={'edgecolor': '#1a1d26', 'linewidth': 1.5})
        fig_p.patch.set_facecolor('#1a1d26')
        st.pyplot(fig_p)
        plt.close()

    with left:
        n_rows = st.slider("Show last N rows", 24, 500, 72, 24)
        disp = filtered.tail(n_rows)[['pred_gap','live_level','live_action']].copy()
        disp['pred_gap'] = disp['pred_gap'].map(lambda x: f"{x:+,.0f} MW")
        disp.index = disp.index.strftime('%Y-%m-%d %H:%M')
        disp.columns = ['Predicted Gap', 'Level', 'Recommended Action']

        def color_level(val):
            fg, bg = LEVEL_COLORS.get(val, ('#fff', '#222'))
            return f'background-color:{bg}; color:{fg}'

        styled = disp.style.map(color_level, subset=['Level'])
        st.dataframe(styled, use_container_width=True, height=400)


# ── TAB 3: Model Performance ─────────────────────────────────
with tab3:
    st.subheader("📊 Model Performance vs ENTSOE Benchmark")

    col_l, col_r = st.columns(2)

    with col_l:
        if dmd is not None:
            from sklearn.metrics import mean_absolute_error
            common = dmd.dropna(subset=['load_actual'])
            metrics_data = {}
            if 'pred_xgb' in common.columns:
                metrics_data['XGBoost'] = mean_absolute_error(
                    common['load_actual'], common['pred_xgb'])
            if 'pred_rf' in common.columns:
                metrics_data['Random Forest'] = mean_absolute_error(
                    common['load_actual'], common['pred_rf'])
            metrics_data['ENTSOE (baseline)'] = 9582

            fig, ax = plt.subplots(figsize=(7, 4), facecolor='#0f1117')
            ax.set_facecolor('#1a1d26')
            names  = list(metrics_data.keys())
            values = list(metrics_data.values())
            bar_colors = ['#F5A623', '#4FC3F7', '#66BB6A'][:len(names)]
            bars = ax.bar(names, values, color=bar_colors, alpha=0.85)
            ax.axhline(9582, color='#66BB6A', lw=1.5,
                       linestyle='--', alpha=0.7)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 80,
                        f'{val:,.0f}', ha='center',
                        fontsize=10, color='white')
            ax.set_ylabel('MAE (MW) — lower is better', color='#c8cad4')
            ax.set_title('Demand Prediction MAE', color='white')
            ax.tick_params(colors='#8b8fa8')
            ax.spines[:].set_color('#2e3145')
            ax.grid(True, color='#2e3145', linestyle='--',
                    axis='y', alpha=0.5)
            fig.patch.set_facecolor('#0f1117')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            best_mae = min(v for k,v in metrics_data.items()
                           if k != 'ENTSOE (baseline)')
            imp = (9582 - best_mae) / 9582 * 100
            st.success(f"🏆 Best model is **{imp:.1f}% better** than ENTSOE "
                       f"(MAE {best_mae:,.0f} MW vs 9,582 MW)")
        else:
            st.info("Run `model_demand_forecast.py` first.")

    with col_r:
        st.markdown("#### Supply Model Results")
        supply_metrics = {
            'Solar XGBoost': {'MAE': '255 MW',  'R²': '0.9955'},
            'Wind XGBoost':  {'MAE': '913 MW',  'R²': '0.9836'},
        }
        for model, m in supply_metrics.items():
            st.markdown(f"""
            <div class="metric-card" style="margin-bottom:0.8rem;">
              <div style="font-size:14px; font-weight:600;
                          color:#c8cad4; margin-bottom:6px;">{model}</div>
              <div style="display:flex; justify-content:space-around;">
                <div>
                  <div class="metric-val" style="color:#F5A623">{m['MAE']}</div>
                  <div class="metric-lbl">MAE</div>
                </div>
                <div>
                  <div class="metric-val" style="color:#66BB6A">{m['R²']}</div>
                  <div class="metric-lbl">R²</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### Germany Grid Reality")
        st.info(
            "Germany's renewables covered ~26% of demand (2015–2019). "
            "The remaining ~74% = conventional generation (coal, gas, nuclear). "
            "Your system correctly identifies this ~40,000 MW gap every hour."
        )


# ── TAB 4: Gap Patterns ──────────────────────────────────────
with tab4:
    st.subheader("🔍 When Does the Grid Need Help? (Full Dataset)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0f1117')

    hourly_gap  = df.groupby(df.index.hour )['pred_gap'].mean()
    monthly_gap = df.groupby(df.index.month)['pred_gap'].mean()

    for ax in axes:
        ax.set_facecolor('#1a1d26')
        ax.tick_params(colors='#8b8fa8')
        ax.grid(True, color='#2e3145', linestyle='--', alpha=0.5)
        ax.spines[:].set_color('#2e3145')

    bc_h = [COLORS['gap_d'] if v > 0 else COLORS['gap_s']
            for v in hourly_gap]
    axes[0].bar(hourly_gap.index, hourly_gap.values, color=bc_h, alpha=0.85)
    axes[0].axhline(0, color='white', lw=0.8, alpha=0.4)
    axes[0].set_xlabel('Hour of day', color='#c8cad4')
    axes[0].set_ylabel('Avg Gap (MW)', color='#c8cad4')
    axes[0].set_title('Average Gap by Hour of Day', color='white')

    bc_m = [COLORS['gap_d'] if v > 0 else COLORS['gap_s']
            for v in monthly_gap]
    axes[1].bar(range(len(monthly_gap)), monthly_gap.values,
                color=bc_m, alpha=0.85)
    axes[1].set_xticks(range(len(monthly_gap)))
    axes[1].set_xticklabels(
        [MONTH_NAMES[i-1] for i in monthly_gap.index],
        fontsize=8, color='#8b8fa8')
    axes[1].axhline(0, color='white', lw=0.8, alpha=0.4)
    axes[1].set_ylabel('Avg Gap (MW)', color='#c8cad4')
    axes[1].set_title('Average Gap by Month (all years)', color='white')

    fig.patch.set_facecolor('#0f1117')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.download_button(
        label     = "📥 Download Full Intervention Report (CSV)",
        data      = df.to_csv().encode('utf-8'),
        file_name = 'gap_interventions_full.csv',
        mime      = 'text/csv',
    )


# ── TAB 5: Year Overview ──────────────────────────────────────
with tab5:
    st.subheader("📅 Year-by-Year Summary")

    df['_year']  = df.index.year
    df['_month'] = df.index.month

    yearly = df.groupby('_year').agg(
        avg_gap     = ('pred_gap',    'mean'),
        avg_demand  = ('pred_demand', 'mean'),
        avg_solar   = ('pred_solar',  'mean'),
        avg_wind    = ('pred_wind',   'mean'),
        pct_deficit = ('pred_gap',    lambda x: (x > 0).mean() * 100),
    ).round(1)

    # Summary table
    st.dataframe(
        yearly.rename(columns={
            'avg_gap':    'Avg Gap (MW)',
            'avg_demand': 'Avg Demand (MW)',
            'avg_solar':  'Avg Solar (MW)',
            'avg_wind':   'Avg Wind (MW)',
            'pct_deficit':'% Hours Deficit',
        }).style.background_gradient(
            subset=['Avg Gap (MW)'], cmap='RdYlGn_r'
        ).format("{:,.1f}"),
        use_container_width=True,
    )

    # Monthly heatmap
    st.markdown("#### Monthly Gap Heatmap (MW)")
    pivot = df.pivot_table(
        values='pred_gap', index='_year',
        columns='_month', aggfunc='mean',
    )
    pivot.columns = [MONTH_NAMES[m-1] for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_facecolor('#1a1d26')
    fig.patch.set_facecolor('#0f1117')

    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=9, color='#c8cad4')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9, color='#c8cad4')
    ax.set_title('Monthly Average Gap (MW) · Red = Larger Deficit',
                 color='white', pad=8)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:,.0f}',
                        ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Gap (MW)', color='#c8cad4')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8b8fa8')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    df.drop(columns=['_year', '_month'], inplace=True)


# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#8b8fa8; font-size:12px;'>"
    "Renewable Energy Load Balancing System &nbsp;·&nbsp; "
    "XGBoost &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp; "
    "OPSD Germany Dataset &nbsp;·&nbsp; 2015 → 2020"
    "</div>",
    unsafe_allow_html=True,
)