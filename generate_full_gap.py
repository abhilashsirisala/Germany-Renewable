# ============================================================
# FULL RANGE GAP ANALYSIS — 2015 to 2020
# Applies trained models to ALL 5+ years of data
# Run AFTER model_supply_gap.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings, os, joblib
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d26',
    'axes.edgecolor':   '#2e3145', 'axes.labelcolor': '#c8cad4',
    'xtick.color':      '#8b8fa8', 'ytick.color':     '#8b8fa8',
    'text.color':       '#c8cad4', 'grid.color':      '#2e3145',
    'grid.linestyle':   '--',      'grid.alpha':       0.5,
    'font.family': 'DejaVu Sans',
})
COLORS = {
    'solar': '#F5A623', 'wind': '#4FC3F7',
    'load':  '#EF5350', 'gap':  '#FF7043',
    'surplus': '#4FC3F7', 'deficit': '#FF7043',
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
os.makedirs('outputs', exist_ok=True)

# ── Quantile-based thresholds — computed from data after gap is known ──
# Thresholds are set in STEP 5B below once full['pred_gap'] exists.
# Using quantiles means thresholds adapt to any dataset automatically:
#   CRITICAL  = top 10% worst deficit hours   (p90 of positive gaps)
#   HIGH      = top 35% deficit hours          (p65 of positive gaps)
#   MODERATE  = bottom 25% of deficit hours    (p25 of positive gaps)
#   SURPLUS   = mildest surplus hours          (p75 of negative gaps)
#   HIGH SUR  = moderate surplus               (p35 of negative gaps)
#   CRIT SUR  = top 10% worst surplus          (p10 of negative gaps)


def compute_thresholds(gap_series: pd.Series) -> dict:
    """Derive all 6 intervention thresholds from the gap distribution."""
    pos = gap_series[gap_series >  0]   # deficit hours only
    neg = gap_series[gap_series <  0]   # surplus hours only

    thr = {}

    if len(pos) >= 10:
        thr['low_def']  = pos.quantile(0.25)
        thr['high_def'] = pos.quantile(0.65)
        thr['crit_def'] = pos.quantile(0.90)
    else:                                # fall back if almost no deficit
        thr['low_def']  =   500.0
        thr['high_def'] =  3000.0
        thr['crit_def'] =  8000.0

    if len(neg) >= 10:
        thr['low_sur']  = neg.quantile(0.75)   # least negative = mildest
        thr['high_sur'] = neg.quantile(0.35)
        thr['crit_sur'] = neg.quantile(0.10)   # most negative = worst
    else:                                # fall back if almost no surplus
        thr['low_sur']  =   -500.0
        thr['high_sur'] =  -3000.0
        thr['crit_sur'] =  -8000.0

    return thr


def make_recommend(thr: dict):
    """Return a recommend() function closed over the given thresholds."""
    def recommend(gap_mw):
        if   gap_mw >  thr['crit_def']:
            return ('CRITICAL',        '🔴 Emergency import + backup generators')
        elif gap_mw >  thr['high_def']:
            return ('HIGH',            '🟠 Demand response + gas peakers')
        elif gap_mw >  thr['low_def']:
            return ('MODERATE',        '🟡 Minor demand response')
        elif gap_mw >= thr['low_sur']:
            return ('BALANCED',        '🟢 Grid balanced — no action needed')
        elif gap_mw >= thr['high_sur']:
            return ('SURPLUS',         '🔵 Charge storage / export')
        elif gap_mw >= thr['crit_sur']:
            return ('HIGH SURPLUS',    '🟣 Curtail wind + charge all storage')
        else:
            return ('CRITICAL SURPLUS','⚫ Emergency curtailment')
    return recommend


# ============================================================
# 1. LOAD ALL CLEANED DATA
# ============================================================
print("=" * 60)
print("  STEP 1 — LOAD FULL DATASET (2015 → 2020)")
print("=" * 60)

df = pd.read_csv('germany_energy_clean.csv',
                 index_col='utc_timestamp', parse_dates=True)
df = df.sort_index()

print(f"\n  Full data range : {df.index.min().date()}  →  {df.index.max().date()}")
print(f"  Total rows      : {len(df):,}")


# ============================================================
# 2. REBUILD SUPPLY FEATURES ON FULL DATASET
# ============================================================
print("\n" + "=" * 60)
print("  STEP 2 — REBUILD SUPPLY FEATURES ON ALL 5+ YEARS")
print("=" * 60)

# Load solar profile, wind profile, AND both capacity columns
# (solar_cap and wind_cap were used during model training)
COLS = [
    'utc_timestamp',
    'DE_solar_profile',
    'DE_solar_capacity',
    'DE_wind_profile',
    'DE_wind_capacity',
    'DE_load_actual_entsoe_transparency',
]

CSV_NAME = 'data.csv'   # ← change if your file has a different name

raw = pd.read_csv(CSV_NAME, usecols=COLS, low_memory=False)
raw['utc_timestamp'] = pd.to_datetime(raw['utc_timestamp'], utc=True)
raw = raw.set_index('utc_timestamp').sort_index()

# Rename to friendly column names
# NOTE: pandas returns usecols in the order they appear in the CSV,
#       so we rename by the original column names directly.
raw = raw.rename(columns={
    'DE_solar_profile':               'solar_profile',
    'DE_solar_capacity':              'solar_cap',
    'DE_wind_profile':                'wind_profile',
    'DE_wind_capacity':               'wind_cap',
    'DE_load_actual_entsoe_transparency': 'load_actual',
})

# Forward-fill capacity columns (they change slowly — monthly updates)
raw['solar_cap'] = raw['solar_cap'].ffill().bfill()
raw['wind_cap']  = raw['wind_cap'].ffill().bfill()

print(f"  solar_cap range : {raw['solar_cap'].min():,.0f} → {raw['solar_cap'].max():,.0f} MW")
print(f"  wind_cap  range : {raw['wind_cap'].min():,.0f}  → {raw['wind_cap'].max():,.0f} MW")

# ── Fix solar/wind generation (same logic as model_supply_gap.py) ──
SOLAR_PEAK_MW = 30_000
WIND_PEAK_MW  = 45_000

sp_max = raw['solar_profile'].max()
wp_max = raw['wind_profile'].max()

if sp_max > 1:
    raw['solar_gen'] = (raw['solar_profile'] / sp_max * SOLAR_PEAK_MW).clip(0)
    print(f"\n  ✅ Solar profile normalised (max was {sp_max:,.0f})")
else:
    raw['solar_gen'] = (raw['solar_profile'] * SOLAR_PEAK_MW).clip(0)
    print(f"\n  ✅ Solar profile already 0-1")

if wp_max > 1:
    raw['wind_gen'] = (raw['wind_profile'] / wp_max * WIND_PEAK_MW).clip(0)
    print(f"  ✅ Wind  profile normalised (max was {wp_max:,.0f})")
else:
    raw['wind_gen'] = (raw['wind_profile'] * WIND_PEAK_MW).clip(0)
    print(f"  ✅ Wind  profile already 0-1")

raw = raw.dropna(subset=['load_actual', 'solar_gen', 'wind_gen'])

# ── Time features ──────────────────────────────────────────
raw['hour']        = raw.index.hour
raw['month']       = raw.index.month
raw['day_of_week'] = raw.index.dayofweek
raw['is_weekend']  = (raw['day_of_week'] >= 5).astype(int)
raw['year']        = raw.index.year
raw['day_of_year'] = raw.index.dayofyear

raw['hour_sin']  = np.sin(2 * np.pi * raw['hour']        / 24)
raw['hour_cos']  = np.cos(2 * np.pi * raw['hour']        / 24)
raw['month_sin'] = np.sin(2 * np.pi * raw['month']       / 12)
raw['month_cos'] = np.cos(2 * np.pi * raw['month']       / 12)
raw['doy_sin']   = np.sin(2 * np.pi * raw['day_of_year'] / 365)
raw['doy_cos']   = np.cos(2 * np.pi * raw['day_of_year'] / 365)

# ── Lag features — solar ────────────────────────────────────
raw['solar_lag_1h']   = raw['solar_gen'].shift(1)
raw['solar_lag_24h']  = raw['solar_gen'].shift(24)
raw['solar_lag_168h'] = raw['solar_gen'].shift(168)
raw['solar_roll_24h'] = raw['solar_gen'].shift(1).rolling(24).mean()

# ── Lag features — wind ─────────────────────────────────────
raw['wind_lag_1h']   = raw['wind_gen'].shift(1)
raw['wind_lag_24h']  = raw['wind_gen'].shift(24)
raw['wind_lag_168h'] = raw['wind_gen'].shift(168)
raw['wind_roll_24h'] = raw['wind_gen'].shift(1).rolling(24).mean()
raw['wind_roll_72h'] = raw['wind_gen'].shift(1).rolling(72).mean()

raw = raw.dropna()

print(f"\n  Rows with full features : {len(raw):,}")
print(f"  Date range              : {raw.index.min().date()} → {raw.index.max().date()}")

# ── Feature lists — MUST match exactly what models were trained on ──
TIME_FEATURES = [
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'doy_sin',  'doy_cos',  'is_weekend', 'year',
]
SOLAR_FEATURES = TIME_FEATURES + [
    'solar_lag_1h', 'solar_lag_24h', 'solar_lag_168h',
    'solar_roll_24h', 'solar_cap',
]
WIND_FEATURES = TIME_FEATURES + [
    'wind_lag_1h', 'wind_lag_24h', 'wind_lag_168h',
    'wind_roll_24h', 'wind_roll_72h', 'wind_cap',
]

print(f"\n  Solar features ({len(SOLAR_FEATURES)}): {SOLAR_FEATURES}")
print(f"  Wind  features ({len(WIND_FEATURES)}): {WIND_FEATURES}")


# ============================================================
# 3. PREDICT DEMAND FOR ALL YEARS
# ============================================================
print("\n" + "=" * 60)
print("  STEP 3 — PREDICT DEMAND FOR ALL 5+ YEARS")
print("=" * 60)

xgb_demand = joblib.load('models/xgb_demand_model.pkl')

dmd = df[['load_actual']].copy()
dmd['hour']        = dmd.index.hour
dmd['month']       = dmd.index.month
dmd['day_of_week'] = dmd.index.dayofweek
dmd['is_weekend']  = (dmd['day_of_week'] >= 5).astype(int)
dmd['year']        = dmd.index.year

dmd['hour_sin']  = np.sin(2 * np.pi * dmd['hour']        / 24)
dmd['hour_cos']  = np.cos(2 * np.pi * dmd['hour']        / 24)
dmd['month_sin'] = np.sin(2 * np.pi * dmd['month']       / 12)
dmd['month_cos'] = np.cos(2 * np.pi * dmd['month']       / 12)
dmd['dow_sin']   = np.sin(2 * np.pi * dmd['day_of_week'] / 7)
dmd['dow_cos']   = np.cos(2 * np.pi * dmd['day_of_week'] / 7)

dmd['lag_1h']    = dmd['load_actual'].shift(1)
dmd['lag_2h']    = dmd['load_actual'].shift(2)
dmd['lag_3h']    = dmd['load_actual'].shift(3)
dmd['lag_24h']   = dmd['load_actual'].shift(24)
dmd['lag_48h']   = dmd['load_actual'].shift(48)
dmd['lag_168h']  = dmd['load_actual'].shift(168)
dmd['roll_6h']   = dmd['load_actual'].shift(1).rolling(6  ).mean()
dmd['roll_24h']  = dmd['load_actual'].shift(1).rolling(24 ).mean()
dmd['roll_168h'] = dmd['load_actual'].shift(1).rolling(168).mean()
dmd['roll_std_24h']  = dmd['load_actual'].shift(1).rolling(24 ).std()
dmd['roll_std_168h'] = dmd['load_actual'].shift(1).rolling(168).std()

dmd = dmd.dropna()

DMD_FEATURES = [
    'lag_1h','lag_2h','lag_3h','lag_24h','lag_48h','lag_168h',
    'roll_6h','roll_24h','roll_168h','roll_std_24h','roll_std_168h',
    'hour_sin','hour_cos','month_sin','month_cos','dow_sin','dow_cos',
    'is_weekend','year',
]

dmd['pred_demand'] = xgb_demand.predict(dmd[DMD_FEATURES])
print(f"  ✅ Demand predictions : {len(dmd):,} rows")
print(f"  Range: {dmd.index.min().date()} → {dmd.index.max().date()}")


# ============================================================
# 4. PREDICT SOLAR & WIND FOR ALL YEARS
# ============================================================
print("\n" + "=" * 60)
print("  STEP 4 — PREDICT SOLAR & WIND FOR ALL 5+ YEARS")
print("=" * 60)

solar_model = joblib.load('models/solar_model.pkl')
wind_model  = joblib.load('models/wind_model.pkl')

# Verify features match before predicting
solar_trained = solar_model.get_booster().feature_names
wind_trained  = wind_model.get_booster().feature_names

print(f"\n  Solar model expects : {solar_trained}")
print(f"  We are providing    : {SOLAR_FEATURES}")
assert list(solar_trained) == SOLAR_FEATURES, \
    f"Solar feature mismatch!\nExpected: {solar_trained}\nGot: {SOLAR_FEATURES}"

print(f"\n  Wind model expects  : {wind_trained}")
print(f"  We are providing    : {WIND_FEATURES}")
assert list(wind_trained) == WIND_FEATURES, \
    f"Wind feature mismatch!\nExpected: {wind_trained}\nGot: {WIND_FEATURES}"

raw['pred_solar'] = solar_model.predict(raw[SOLAR_FEATURES]).clip(min=0)
raw['pred_wind']  = wind_model.predict(raw[WIND_FEATURES] ).clip(min=0)

print(f"\n  ✅ Supply predictions : {len(raw):,} rows")
print(f"  Solar pred mean : {raw['pred_solar'].mean():,.0f} MW")
print(f"  Wind  pred mean : {raw['pred_wind'].mean():,.0f} MW")


# ============================================================
# 5. MERGE & COMPUTE FULL GAP
# ============================================================
print("\n" + "=" * 60)
print("  STEP 5 — COMPUTE FULL GAP 2015 → 2020")
print("=" * 60)

common = dmd.index.intersection(raw.index)

full = pd.DataFrame({
    'actual_demand': dmd.loc[common, 'load_actual'],
    'pred_demand':   dmd.loc[common, 'pred_demand'],
    'actual_solar':  raw.loc[common, 'solar_gen'],
    'actual_wind':   raw.loc[common, 'wind_gen'],
    'pred_solar':    raw.loc[common, 'pred_solar'],
    'pred_wind':     raw.loc[common, 'pred_wind'],
}, index=common).dropna()

full['pred_supply']   = full['pred_solar']   + full['pred_wind']
full['actual_supply'] = full['actual_solar'] + full['actual_wind']
full['pred_gap']      = full['pred_demand']  - full['pred_supply']
full['actual_gap']    = full['actual_demand']- full['actual_supply']

# ── STEP 5B: Compute quantile thresholds from actual gap data ──
print("\n  Computing quantile-based intervention thresholds...")
thr = compute_thresholds(full['pred_gap'])
recommend = make_recommend(thr)

print(f"\n  Threshold summary (all data-driven, no hardcoding):")
print(f"  {'Level':<20} {'Threshold (MW)':>16}")
print("  " + "-" * 38)
print(f"  {'CRITICAL (deficit)':<20} {thr['crit_def']:>+16,.0f}  (p90 of deficit hours)")
print(f"  {'HIGH (deficit)':<20} {thr['high_def']:>+16,.0f}  (p65 of deficit hours)")
print(f"  {'MODERATE (deficit)':<20} {thr['low_def']:>+16,.0f}  (p25 of deficit hours)")
print(f"  {'BALANCED':<20} {'± above/below':>16}")
print(f"  {'SURPLUS':<20} {thr['low_sur']:>+16,.0f}  (p75 of surplus hours)")
print(f"  {'HIGH SURPLUS':<20} {thr['high_sur']:>+16,.0f}  (p35 of surplus hours)")
print(f"  {'CRITICAL SURPLUS':<20} {thr['crit_sur']:>+16,.0f}  (p10 of surplus hours)")

levels, actions = zip(*[recommend(g) for g in full['pred_gap']])
full['level']  = levels
full['action'] = actions

print(f"\n  Full dataset rows    : {len(full):,}")
print(f"  Date range           : {full.index.min().date()} → {full.index.max().date()}")
print(f"  Avg predicted demand : {full['pred_demand'].mean():,.0f} MW")
print(f"  Avg predicted supply : {full['pred_supply'].mean():,.0f} MW")
print(f"  Avg predicted gap    : {full['pred_gap'].mean():,.0f} MW")
print(f"  % hours deficit      : {(full['pred_gap'] > 0).mean()*100:.1f}%")
print(f"  % hours surplus      : {(full['pred_gap'] <= 0).mean()*100:.1f}%")

level_counts = pd.Series(levels).value_counts()
print(f"\n  Intervention breakdown (full dataset):")
print(f"  {'Level':<20} {'Hours':>8}  {'%':>8}")
print("  " + "-" * 42)
for lvl, cnt in level_counts.items():
    print(f"  {lvl:<20} {cnt:>8,}   {cnt/len(full)*100:>7.1f}%")

full.to_csv('outputs/full_gap_2015_2020.csv')
print(f"\n  Saved → outputs/full_gap_2015_2020.csv")


# ============================================================
# 6. YEAR-BY-YEAR SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  STEP 6 — YEAR-BY-YEAR SUMMARY")
print("=" * 60)

full['year_col']  = full.index.year
full['month_col'] = full.index.month

yearly = full.groupby('year_col').agg(
    avg_gap     = ('pred_gap',    'mean'),
    avg_demand  = ('pred_demand', 'mean'),
    avg_solar   = ('pred_solar',  'mean'),
    avg_wind    = ('pred_wind',   'mean'),
    pct_deficit = ('pred_gap',    lambda x: (x > 0).mean() * 100),
).round(0)

print(f"\n  {'Year':<6} {'Avg Gap':>10} {'Avg Load':>10} "
      f"{'Avg Solar':>10} {'Avg Wind':>10} {'% Deficit':>10}")
print("  " + "-" * 60)
for yr, row in yearly.iterrows():
    print(f"  {yr:<6} {row['avg_gap']:>+10,.0f} {row['avg_demand']:>10,.0f} "
          f"{row['avg_solar']:>10,.0f} {row['avg_wind']:>10,.0f} "
          f"{row['pct_deficit']:>9.1f}%")


# ============================================================
# 7. VISUALISATIONS
# ============================================================
print("\n" + "=" * 60)
print("  STEP 7 — GENERATING FULL-RANGE PLOTS")
print("=" * 60)

# ── PLOT 14: Full 2015-2020 time series ───────────────────────
fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
fig.suptitle('Germany Renewable Energy — Full Dataset  2015 → 2020',
             fontsize=15, color='white', fontweight='bold', y=1.01)

daily = full.resample('D').mean(numeric_only=True)

axes[0].fill_between(daily.index, daily['pred_solar'],
                     alpha=0.70, color=COLORS['solar'], label='Solar (predicted)')
axes[0].fill_between(daily.index,
                     daily['pred_solar'] + daily['pred_wind'],
                     daily['pred_solar'],
                     alpha=0.55, color=COLORS['wind'],  label='Wind (predicted)')
axes[0].plot(daily.index, daily['pred_demand'],
             color=COLORS['load'], lw=1.2, label='Demand (predicted)')

axes[1].fill_between(daily.index, daily['pred_gap'],
                     where=daily['pred_gap'] > 0,
                     color=COLORS['deficit'], alpha=0.75,
                     label='Deficit (conventional generation needed)')
axes[1].fill_between(daily.index, daily['pred_gap'],
                     where=daily['pred_gap'] <= 0,
                     color=COLORS['surplus'], alpha=0.55,
                     label='Surplus (excess renewable)')
axes[1].axhline(0, color='white', lw=0.8, alpha=0.5)
axes[1].axhline(thr['crit_def'], color='#EF5350', lw=1.0,
                linestyle=':', alpha=0.8,
                label=f"Critical deficit ({thr['crit_def']:,.0f} MW · p90)")
axes[1].axhline(thr['high_def'], color='#FF7043', lw=0.8,
                linestyle=':', alpha=0.6,
                label=f"High deficit ({thr['high_def']:,.0f} MW · p65)")
if thr['low_sur'] < 0:
    axes[1].axhline(thr['low_sur'], color='#29B6F6', lw=0.8,
                    linestyle=':', alpha=0.6,
                    label=f"Surplus ({thr['low_sur']:,.0f} MW · p75)")

bar_colors = [COLORS['deficit'] if v > 0 else COLORS['surplus']
              for v in yearly['avg_gap']]
axes[2].bar(yearly.index, yearly['avg_gap'],
            color=bar_colors, alpha=0.85, width=0.6)
axes[2].axhline(0, color='white', lw=0.8, alpha=0.5)
for yr, val in yearly['avg_gap'].items():
    axes[2].text(yr, val + 300, f'{val:,.0f}',
                 ha='center', va='bottom', fontsize=9, color='white')
axes[2].set_xticks(yearly.index)

titles = [
    'Solar & Wind Supply vs Demand (daily avg)',
    'Supply–Demand Gap — orange=deficit, blue=surplus (daily avg)',
    'Average Annual Gap (MW)',
]
ylabels = ['MW', 'Gap (MW)', 'Avg Gap (MW)']

for ax, title, ylabel in zip(axes, titles, ylabels):
    ax.set_facecolor('#1a1d26')
    ax.grid(True, color='#2e3145', linestyle='--', alpha=0.5)
    ax.spines[:].set_color('#2e3145')
    ax.set_title(title, color='white', pad=6)
    ax.set_ylabel(ylabel, color='#c8cad4')

axes[0].legend(fontsize=9, facecolor='#1a1d26', labelcolor='#c8cad4')
axes[1].legend(fontsize=9, facecolor='#1a1d26', labelcolor='#c8cad4')

plt.tight_layout()
plt.savefig('plot14_full_range_overview.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  Saved → plot14_full_range_overview.png")


# ── PLOT 15: Year-by-year breakdown ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Year-by-Year Energy Breakdown  2015 → 2020',
             fontsize=13, color='white', fontweight='bold')

yrs  = list(yearly.index)
x    = np.arange(len(yrs))
w    = 0.25

axes[0].bar(x - w, yearly['avg_solar'].values, w,
            color=COLORS['solar'], alpha=0.85, label='Solar')
axes[0].bar(x,     yearly['avg_wind'].values,  w,
            color=COLORS['wind'],  alpha=0.85, label='Wind')
axes[0].bar(x + w, yearly['avg_demand'].values, w,
            color=COLORS['load'],  alpha=0.85, label='Load')
axes[0].set_xticks(x)
axes[0].set_xticklabels(yrs)
axes[0].set_ylabel('Average MW')
axes[0].set_title('Avg Solar, Wind & Load by Year')
axes[0].legend(fontsize=9, facecolor='#1a1d26', labelcolor='#c8cad4')

axes[1].bar(range(len(yrs)), yearly['pct_deficit'].values,
            color=COLORS['deficit'], alpha=0.85)
axes[1].set_xticks(range(len(yrs)))
axes[1].set_xticklabels(yrs)
axes[1].set_ylabel('% of hours in deficit')
axes[1].set_title('% Hours Renewables Alone Cannot Cover Demand')
for i, val in enumerate(yearly['pct_deficit'].values):
    axes[1].text(i, val + 0.3, f'{val:.1f}%',
                 ha='center', va='bottom', fontsize=9, color='white')

for ax in axes:
    ax.set_facecolor('#1a1d26')
    ax.spines[:].set_color('#2e3145')
    ax.tick_params(colors='#8b8fa8')
    ax.grid(True, color='#2e3145', linestyle='--', alpha=0.5, axis='y')

plt.tight_layout()
plt.savefig('plot15_yearly_breakdown.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  Saved → plot15_yearly_breakdown.png")


# ── PLOT 16: Month × Year heatmap ─────────────────────────────
pivot = full.pivot_table(
    values='pred_gap',
    index='year_col',
    columns='month_col',
    aggfunc='mean',
)
month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']
pivot.columns = [month_labels[m-1] for m in pivot.columns]

fig, ax = plt.subplots(figsize=(14, 5))
ax.set_facecolor('#1a1d26')
fig.patch.set_facecolor('#0f1117')

im = ax.imshow(pivot.values, cmap=plt.cm.RdYlGn_r, aspect='auto')
ax.set_xticks(range(pivot.shape[1]))
ax.set_xticklabels(pivot.columns, fontsize=9, color='#c8cad4')
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=9, color='#c8cad4')
ax.set_title(
    'Monthly Average Supply–Demand Gap (MW) · 2015 → 2020\n'
    'Red = larger deficit  ·  Green = smaller deficit / surplus',
    color='white', pad=10, fontsize=12,
)

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
plt.savefig('plot16_monthly_heatmap.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  Saved → plot16_monthly_heatmap.png")


# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"""
{'=' * 60}
  ✅ COMPLETE — FULL RANGE GAP ANALYSIS
{'=' * 60}

  Date range   : {full.index.min().date()} → {full.index.max().date()}
  Total hours  : {len(full):,}

  Avg gap      : {full['pred_gap'].mean():,.0f} MW
  % deficit    : {(full['pred_gap'] > 0).mean()*100:.1f}%
  % surplus    : {(full['pred_gap'] <= 0).mean()*100:.1f}%

  3 plots saved:
    plot14_full_range_overview.png
    plot15_yearly_breakdown.png
    plot16_monthly_heatmap.png

  Full CSV saved:
    outputs/full_gap_2015_2020.csv

  NEXT STEP → fixplots14_15
  Run: python fixplots14_15.py
""")