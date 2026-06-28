# ============================================================
# STEP 1: EDA — Renewable Energy Load Balancing System
# Dataset: Open Power System Data (OPSD) — Germany
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ── Plot style ───────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d26',
    'axes.edgecolor':   '#2e3145',
    'axes.labelcolor':  '#c8cad4',
    'xtick.color':      '#8b8fa8',
    'ytick.color':      '#8b8fa8',
    'text.color':       '#c8cad4',
    'grid.color':       '#2e3145',
    'grid.linestyle':   '--',
    'grid.alpha':       0.6,
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
})

COLORS = {
    'solar':    '#F5A623',
    'wind':     '#4FC3F7',
    'onshore':  '#26C6DA',
    'offshore': '#0288D1',
    'load':     '#EF5350',
    'forecast': '#AB47BC',
    'price':    '#66BB6A',
    'gap':      '#FF7043',
}

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 55)
print("  STEP 1 — LOADING DATA")
print("=" * 55)

COLS = [
    'utc_timestamp',
    'DE_solar_generation_actual',
    'DE_wind_generation_actual',
    'DE_wind_onshore_generation_actual',
    'DE_wind_offshore_generation_actual',
    'DE_load_actual_entsoe_transparency',
    'DE_load_forecast_entsoe_transparency',
    'DE_solar_capacity',
    'DE_LU_price_day_ahead'
]

# ── Change the filename below to match your downloaded file ──
df = pd.read_csv('data.csv',
                 usecols=COLS,
                 low_memory=False)

df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'], utc=True)
df = df.set_index('utc_timestamp').sort_index()

# Shorter column names for convenience
df = df.rename(columns={
    'DE_solar_generation_actual': 'solar_gen',
    'DE_wind_generation_actual': 'wind_gen',
    'DE_wind_onshore_generation_actual': 'wind_onshore',
    'DE_wind_offshore_generation_actual': 'wind_offshore',
    'DE_load_actual_entsoe_transparency': 'load_actual',
    'DE_load_forecast_entsoe_transparency': 'load_forecast',
    'DE_solar_capacity': 'solar_capacity',
    'DE_LU_price_day_ahead': 'price',
})

print(f"\n  Rows   : {df.shape[0]:,}")
print(f"  Columns: {df.shape[1]}")
print(f"  From   : {df.index.min()}")
print(f"  To     : {df.index.max()}")


# ============================================================
# 2. BASIC INSPECTION
# ============================================================
print("\n" + "=" * 55)
print("  STEP 2 — BASIC INSPECTION")
print("=" * 55)

print("\n── First 5 rows ──")
print(df.head())

print("\n── Data types ──")
print(df.dtypes)

print("\n── Statistical summary (MW) ──")
print(df.describe().round(1))


# ============================================================
# 3. MISSING VALUE ANALYSIS
# ============================================================
print("\n" + "=" * 55)
print("  STEP 3 — MISSING VALUE ANALYSIS")
print("=" * 55)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing,
                            'Missing %':    missing_pct})
print(missing_df)

# ── Fill missing values ──
# For generation/load: forward-fill short gaps (≤3 hours)
gen_cols = ['solar_gen', 'wind_gen', 'wind_onshore',
            'wind_offshore', 'load_actual', 'load_forecast']
df[gen_cols] = df[gen_cols].ffill( limit=3)

# For capacity: forward-fill (changes slowly)
df['solar_capacity'] = df['solar_capacity'].ffill()

# For price: interpolate (continuous market signal)
df['price'] = df['price'].interpolate(method='linear', limit=6)

print(f"\n  Missing values after fill: {df.isnull().sum().sum()}")

# ── Keep only rows where load_actual is not null ──
df = df.dropna(subset=['load_actual'])
print(f"  Rows after dropping null load rows: {len(df):,}")


# ============================================================
# 4. FEATURE ENGINEERING — time + forecast error
# ============================================================
print("\n" + "=" * 55)
print("  STEP 4 — FEATURE ENGINEERING")
print("=" * 55)

# Time features
df['hour']        = df.index.hour  #type:ignore
df['day_of_week'] = df.index.dayofweek        # 0=Mon, 6=Sun  #type:ignore
df['month']       = df.index.month   #type:ignore
df['year']        = df.index.year   #type:ignore
df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
df['season']      = df['month'].map({
    12:'Winter', 1:'Winter', 2:'Winter',
    3:'Spring',  4:'Spring', 5:'Spring',
    6:'Summer',  7:'Summer', 8:'Summer',
    9:'Autumn', 10:'Autumn', 11:'Autumn'
})

# Total renewable supply
df['renewable_total'] = df['solar_gen'] + df['wind_gen']

# Supply–demand gap (positive = deficit, negative = surplus)
df['supply_gap'] = df['load_actual'] - df['renewable_total']

# Forecast error (your model will try to beat this)
df['forecast_error'] = df['load_actual'] - df['load_forecast']

# Solar utilisation rate (% of capacity actually used)
df['solar_utilisation'] = (
    df['solar_gen'] / df['solar_capacity'].replace(0, np.nan) * 100
).clip(0, 100)

print("  New columns added:")
new_cols = ['hour', 'day_of_week', 'month', 'year', 'is_weekend',
            'season', 'renewable_total', 'supply_gap',
            'forecast_error', 'solar_utilisation']
for c in new_cols:
    print(f"    ✓ {c}")


# ============================================================
# 5. VISUALISATIONS
# ============================================================
print("\n" + "=" * 55)
print("  STEP 5 — GENERATING PLOTS")
print("=" * 55)

# Use a single year slice for cleaner plots
YEAR = df.index.year.value_counts().idxmax()      # most data   #type:ignore
yr   = df[df['year'] == YEAR].copy()

# ── PLOT 1: Full year time series ────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig.suptitle(f'Germany Renewable Energy — {YEAR}',
             fontsize=15, y=1.01, color='white', fontweight='bold')

axes[0].plot(yr.index, yr['solar_gen'],   color=COLORS['solar'],
             lw=0.6, label='Solar generation')
axes[0].plot(yr.index, yr['wind_gen'],    color=COLORS['wind'],
             lw=0.6, label='Wind generation', alpha=0.9)
axes[0].fill_between(yr.index, yr['solar_gen'], alpha=0.15,
                     color=COLORS['solar'])
axes[0].fill_between(yr.index, yr['wind_gen'],  alpha=0.10,
                     color=COLORS['wind'])
axes[0].set_ylabel('MW')
axes[0].legend(loc='upper right', fontsize=9)
axes[0].set_title('Solar & Wind Generation', pad=6)
axes[0].grid(True)

axes[1].plot(yr.index, yr['load_actual'],   color=COLORS['load'],
             lw=0.6, label='Actual demand')
axes[1].plot(yr.index, yr['load_forecast'], color=COLORS['forecast'],
             lw=0.5, label='Forecasted demand', alpha=0.7, linestyle='--')
axes[1].fill_between(yr.index, yr['load_actual'], alpha=0.12,
                     color=COLORS['load'])
axes[1].set_ylabel('MW')
axes[1].legend(loc='upper right', fontsize=9)
axes[1].set_title('Actual vs Forecasted Load Demand', pad=6)
axes[1].grid(True)

axes[2].fill_between(yr.index, yr['supply_gap'],
                     where=yr['supply_gap'] > 0,
                     color=COLORS['gap'], alpha=0.7, label='Deficit (need more power)')
axes[2].fill_between(yr.index, yr['supply_gap'],
                     where=yr['supply_gap'] <= 0,
                     color=COLORS['wind'], alpha=0.5, label='Surplus (too much power)')
axes[2].axhline(0, color='white', lw=0.8, linestyle='-', alpha=0.5)
axes[2].set_ylabel('MW')
axes[2].legend(loc='upper right', fontsize=9)
axes[2].set_title('Supply–Demand Gap  (positive = deficit)', pad=6)
axes[2].grid(True)

plt.tight_layout()
plt.savefig('plot1_timeseries.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("  Saved → plot1_timeseries.png")


# ── PLOT 2: Hourly patterns ──────────────────────────────────
hourly = df.groupby('hour')[
    ['solar_gen', 'wind_gen', 'load_actual']
].mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Average Hourly Patterns (all years)',
             fontsize=14, color='white', fontweight='bold')

axes[0].bar(hourly.index, hourly['solar_gen'],
            color=COLORS['solar'], alpha=0.8, label='Solar')
axes[0].bar(hourly.index, hourly['wind_gen'],
            bottom=hourly['solar_gen'],
            color=COLORS['wind'], alpha=0.8, label='Wind')
axes[0].plot(hourly.index, hourly['load_actual'],
             color=COLORS['load'], lw=2.5,
             marker='o', markersize=4, label='Load demand')
axes[0].set_xlabel('Hour of day')
axes[0].set_ylabel('Average MW')
axes[0].set_title('Generation vs Demand by Hour')
axes[0].legend(fontsize=9)
axes[0].grid(True, axis='y')

# Weekday vs Weekend
wd = df[df['is_weekend'] == 0].groupby('hour')['load_actual'].mean()
we = df[df['is_weekend'] == 1].groupby('hour')['load_actual'].mean()
axes[1].plot(wd.index, wd, color=COLORS['load'],  lw=2,
             label='Weekday', marker='o', markersize=3)
axes[1].plot(we.index, we, color=COLORS['price'], lw=2,
             label='Weekend', marker='s', markersize=3,
             linestyle='--')
axes[1].set_xlabel('Hour of day')
axes[1].set_ylabel('Average Load (MW)')
axes[1].set_title('Load: Weekday vs Weekend')
axes[1].legend(fontsize=9)
axes[1].grid(True)

plt.tight_layout()
plt.savefig('plot2_hourly_patterns.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("  Saved → plot2_hourly_patterns.png")


# ── PLOT 3: Seasonal patterns ────────────────────────────────
season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
season_avg = df.groupby('season')[
    ['solar_gen', 'wind_gen', 'load_actual', 'supply_gap']
].mean().reindex(season_order)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Seasonal Averages', fontsize=14,
             color='white', fontweight='bold')

x = np.arange(len(season_order))
w = 0.3
axes[0].bar(x - w, season_avg['solar_gen'],  w, color=COLORS['solar'],
            alpha=0.85, label='Solar')
axes[0].bar(x,     season_avg['wind_gen'],   w, color=COLORS['wind'],
            alpha=0.85, label='Wind')
axes[0].bar(x + w, season_avg['load_actual'], w, color=COLORS['load'],
            alpha=0.85, label='Load')
axes[0].set_xticks(x)
axes[0].set_xticklabels(season_order)
axes[0].set_ylabel('Average MW')
axes[0].set_title('Generation & Load by Season')
axes[0].legend(fontsize=9)
axes[0].grid(True, axis='y')

colors_gap = [COLORS['wind'] if v < 0 else COLORS['gap']
              for v in season_avg['supply_gap']]
axes[1].bar(season_order, season_avg['supply_gap'],
            color=colors_gap, alpha=0.85)
axes[1].axhline(0, color='white', lw=0.8, alpha=0.5)
axes[1].set_ylabel('Average Gap (MW)')
axes[1].set_title('Supply–Demand Gap by Season\n'
                  '(positive = deficit, negative = surplus)')
axes[1].grid(True, axis='y')

plt.tight_layout()
plt.savefig('plot3_seasonal.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("  Saved → plot3_seasonal.png")


# ── PLOT 4: Correlation heatmap ──────────────────────────────
corr_cols = ['solar_gen', 'wind_gen', 'wind_onshore',
             'wind_offshore', 'load_actual', 'load_forecast',
             'price', 'supply_gap', 'forecast_error',
             'hour', 'month', 'is_weekend']

corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, vmin=-1, vmax=1,
            linewidths=0.4, linecolor='#0f1117',
            annot_kws={'size': 8},
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', fontsize=14,
             color='white', fontweight='bold', pad=14)
ax.tick_params(labelsize=9)
plt.tight_layout()
plt.savefig('plot4_correlation.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("  Saved → plot4_correlation.png")


# ── PLOT 5: Forecast error distribution ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('ENTSOE Forecast Error Analysis\n'
             '(This is the benchmark your model must beat)',
             fontsize=13, color='white', fontweight='bold')

fe = df['forecast_error'].dropna()
axes[0].hist(fe, bins=80, color=COLORS['forecast'],
             alpha=0.75, edgecolor='none')
axes[0].axvline(0, color='white', lw=1.5, linestyle='--')
axes[0].axvline(fe.mean(), color=COLORS['gap'], lw=1.5,
                linestyle='-', label=f'Mean: {fe.mean():.0f} MW')
axes[0].set_xlabel('Forecast Error (MW)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Forecast Errors')
axes[0].legend(fontsize=9)
axes[0].grid(True, axis='y')

axes[1].plot(df.groupby('hour')['forecast_error'].mean(),
             color=COLORS['forecast'], lw=2, marker='o', markersize=4)
axes[1].axhline(0, color='white', lw=0.8, linestyle='--', alpha=0.5)
axes[1].set_xlabel('Hour of day')
axes[1].set_ylabel('Average Error (MW)')
axes[1].set_title('Forecast Error by Hour\n(when is ENTSOE most wrong?)')
axes[1].grid(True)

plt.tight_layout()
plt.savefig('plot5_forecast_error.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("  Saved → plot5_forecast_error.png")


# ============================================================
# 6. KEY INSIGHTS SUMMARY
# ============================================================
print("\n" + "=" * 55)
print("  STEP 6 — KEY INSIGHTS")
print("=" * 55)

mae_entsoe = fe.abs().mean()
print(f"\n  ENTSOE forecast MAE        : {mae_entsoe:,.0f} MW")
print(f"  (your model must beat this!)\n")

peak_load_hour = df.groupby('hour')['load_actual'].mean().idxmax()
peak_solar_hour = df.groupby('hour')['solar_gen'].mean().idxmax()
print(f"  Peak demand hour           : {peak_load_hour}:00")
print(f"  Peak solar hour            : {peak_solar_hour}:00")

deficit_pct = (df['supply_gap'] > 0).mean() * 100
surplus_pct = (df['supply_gap'] <= 0).mean() * 100
print(f"\n  % of hours in deficit      : {deficit_pct:.1f}%")
print(f"  % of hours in surplus      : {surplus_pct:.1f}%")

best_season = season_avg['solar_gen'].idxmax()
worst_season = season_avg['solar_gen'].idxmin()
print(f"\n  Best solar season          : {best_season}")
print(f"  Worst solar season         : {worst_season}")

print(f"\n  Renewable vs load corr     : "
      f"{df['renewable_total'].corr(df['load_actual']):.3f}")
print(f"  Solar vs load corr         : "
      f"{df['solar_gen'].corr(df['load_actual']):.3f}")
print(f"  Wind vs load corr          : "
      f"{df['wind_gen'].corr(df['load_actual']):.3f}")

# ============================================================
# 7. SAVE CLEAN DATASET
# ============================================================
df.to_csv('germany_energy_clean.csv')
print("\n" + "=" * 55)
print("  Clean dataset saved → germany_energy_clean.csv")
print("  This is the file you will use for model training.")
print("=" * 55)

print("""
  ✅ EDA COMPLETE — 5 plots saved:
     plot1_timeseries.png
     plot2_hourly_patterns.png
     plot3_seasonal.png
     plot4_correlation.png
     plot5_forecast_error.png

  NEXT STEP → Demand Prediction Model
  Run: python model_demand_forecast.py
""")