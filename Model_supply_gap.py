# ============================================================
# STEP 3: SUPPLY PREDICTION + GAP ANALYSIS + INTERVENTIONS
# Renewable Energy Load Balancing System
# ============================================================
# What this script does:
#   A) Fix the solar/wind data issue found in EDA
#   B) Build models to predict solar & wind supply
#   C) Calculate supply-demand gap
#   D) Recommend interventions to balance the grid
#   E) Save everything for the dashboard
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings, os, joblib
warnings.filterwarnings('ignore')

from sklearn.ensemble      import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics       import mean_absolute_error, r2_score
import xgboost as xgb

# ── Plot style ───────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117', 'axes.facecolor': '#1a1d26',
    'axes.edgecolor':   '#2e3145', 'axes.labelcolor': '#c8cad4',
    'xtick.color':      '#8b8fa8', 'ytick.color':     '#8b8fa8',
    'text.color':       '#c8cad4', 'grid.color':      '#2e3145',
    'grid.linestyle':   '--',      'grid.alpha':       0.5,
    'font.family': 'DejaVu Sans',  'axes.titlesize':  12,
    'axes.labelsize': 10,
})

COLORS = {
    'solar':    '#F5A623', 'wind':     '#4FC3F7',
    'load':     '#EF5350', 'gap':      '#FF7043',
    'surplus':  '#4FC3F7', 'deficit':  '#FF7043',
    'pred':     '#AB47BC', 'safe':     '#66BB6A',
    'warning':  '#FFA726', 'critical': '#EF5350',
}

os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)


# ============================================================
# PART A — RELOAD & FIX DATA
# ============================================================
print("=" * 60)
print("  PART A — RELOAD & FIX DATA")
print("=" * 60)

# ── Load original CSV with profile columns ───────────────────
COLS_NEEDED = [
    'utc_timestamp',
    'DE_solar_generation_actual',
    'DE_solar_capacity',
    'DE_solar_profile',
    'DE_wind_generation_actual',
    'DE_wind_capacity',
    'DE_wind_profile',
    'DE_wind_onshore_generation_actual',
    'DE_wind_offshore_generation_actual',
    'DE_load_actual_entsoe_transparency',
]

print("\n  Loading original OPSD CSV...")
raw = pd.read_csv(
    'data.csv',
    usecols   = COLS_NEEDED,
    low_memory = False,
)
raw['utc_timestamp'] = pd.to_datetime(raw['utc_timestamp'], utc=True)
raw = raw.set_index('utc_timestamp').sort_index()

raw = raw.rename(columns={
    'DE_solar_generation_actual': 'solar_raw',
    'DE_solar_capacity': 'solar_cap',
    'DE_solar_profile': 'solar_profile',
    'DE_wind_generation_actual': 'wind_raw',
    'DE_wind_capacity': 'wind_cap',
    'DE_wind_profile': 'wind_profile',
    'DE_wind_onshore_generation_actual': 'wind_onshore_raw',
    'DE_wind_offshore_generation_actual': 'wind_offshore_raw',
    'DE_load_actual_entsoe_transparency': 'load_actual',
})

# ── Fix solar & wind generation ──────────────────────────────
# The raw columns tracked capacity trends (confirmed in EDA).
# OPSD provides _profile columns = generation / capacity (0-1).
# True generation = profile × capacity.

print("\n  Checking profile columns...")
profile_available = (
    raw['solar_profile'].notna().sum() > 1000 and
    raw['wind_profile'].notna().sum()  > 1000
)

if profile_available:
    print("  ✅ Profile columns found — reconstructing true generation")
    raw['solar_cap']  = raw['solar_cap'].ffill()
    raw['wind_cap']   = raw['wind_cap'].ffill()
    raw['solar_gen']  = (raw['solar_profile'] * raw['solar_cap']
                         ).clip(lower=0)
    raw['wind_gen']   = (raw['wind_profile']  * raw['wind_cap']
                         ).clip(lower=0)
else:
    print("  ⚠️  Profile columns empty — using raw columns directly")
    print("     (values may reflect capacity; gap analysis will note this)")
    raw['solar_gen'] = raw['solar_raw'].clip(lower=0)
    raw['wind_gen']  = raw['wind_raw'].clip(lower=0)

raw = raw.dropna(subset=['load_actual', 'solar_gen', 'wind_gen'])
raw = raw.sort_index()

print(f"\n  Rows available : {len(raw):,}")
print(f"  Solar gen      : mean={raw['solar_gen'].mean():,.0f} MW  "
      f"max={raw['solar_gen'].max():,.0f} MW")
print(f"  Wind gen       : mean={raw['wind_gen'].mean():,.0f} MW  "
      f"max={raw['wind_gen'].max():,.0f} MW")
print(f"  Load actual    : mean={raw['load_actual'].mean():,.0f} MW  "
      f"max={raw['load_actual'].max():,.0f} MW")


# ============================================================
# PART B — FEATURE ENGINEERING FOR SUPPLY MODELS
# ============================================================
print("\n" + "=" * 60)
print("  PART B — FEATURE ENGINEERING")
print("=" * 60)

df = raw.copy()

# Time features
df['hour']        = df.index.hour
df['month']       = df.index.month
df['day_of_week'] = df.index.dayofweek
df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
df['year']        = df.index.year
df['day_of_year'] = df.index.dayofyear

# Cyclical encoding
df['hour_sin']  = np.sin(2 * np.pi * df['hour']        / 24)
df['hour_cos']  = np.cos(2 * np.pi * df['hour']        / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month']       / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month']       / 12)
df['doy_sin']   = np.sin(2 * np.pi * df['day_of_year'] / 365)
df['doy_cos']   = np.cos(2 * np.pi * df['day_of_year'] / 365)

# Lag features for solar
df['solar_lag_1h']   = df['solar_gen'].shift(1)
df['solar_lag_24h']  = df['solar_gen'].shift(24)
df['solar_lag_168h'] = df['solar_gen'].shift(168)
df['solar_roll_24h'] = df['solar_gen'].shift(1).rolling(24).mean()

# Lag features for wind
df['wind_lag_1h']    = df['wind_gen'].shift(1)
df['wind_lag_24h']   = df['wind_gen'].shift(24)
df['wind_lag_168h']  = df['wind_gen'].shift(168)
df['wind_roll_24h']  = df['wind_gen'].shift(1).rolling(24).mean()
df['wind_roll_72h']  = df['wind_gen'].shift(1).rolling(72).mean()

# Installed capacity (slowly growing — useful trend feature)
df['solar_cap']  = df['solar_cap'].ffill()
df['wind_cap']   = df['wind_cap'].ffill()

df = df.dropna()
print(f"  Rows after feature engineering : {len(df):,}")

# ── Shared time features ─────────────────────────────────────
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

print(f"  Solar features : {len(SOLAR_FEATURES)}")
print(f"  Wind features  : {len(WIND_FEATURES)}")


# ============================================================
# PART C — TRAIN/TEST SPLIT
# ============================================================
split_idx  = int(len(df) * 0.80)
train      = df.iloc[:split_idx]
test       = df.iloc[split_idx:]

print(f"\n  Train : {len(train):,} rows  "
      f"({train.index.min().date()} → {train.index.max().date()})")
print(f"  Test  : {len(test):,}  rows  "
      f"({test.index.min().date()} → {test.index.max().date()})")


# ============================================================
# PART D — SOLAR GENERATION MODEL
# ============================================================
print("\n" + "=" * 60)
print("  PART D — SOLAR GENERATION MODEL (XGBoost)")
print("=" * 60)

solar_model = xgb.XGBRegressor(
    n_estimators=400, learning_rate=0.05,
    max_depth=6, subsample=0.8,
    colsample_bytree=0.8, random_state=42,
    verbosity=0, n_jobs=-1,
)

print("\n  Training solar model...")
solar_model.fit(train[SOLAR_FEATURES], train['solar_gen'],
                eval_set=[(test[SOLAR_FEATURES], test['solar_gen'])],
                verbose=False)

solar_preds = solar_model.predict(test[SOLAR_FEATURES]).clip(min=0)
solar_mae   = mean_absolute_error(test['solar_gen'], solar_preds)
solar_r2    = r2_score(test['solar_gen'], solar_preds)

print(f"  Solar MAE : {solar_mae:,.0f} MW")
print(f"  Solar R²  : {solar_r2:.4f}")
joblib.dump(solar_model, 'models/solar_model.pkl')
print("  Saved → models/solar_model.pkl")


# ============================================================
# PART E — WIND GENERATION MODEL
# ============================================================
print("\n" + "=" * 60)
print("  PART E — WIND GENERATION MODEL (XGBoost)")
print("=" * 60)

wind_model = xgb.XGBRegressor(
    n_estimators=400, learning_rate=0.05,
    max_depth=7, subsample=0.8,
    colsample_bytree=0.8, random_state=42,
    verbosity=0, n_jobs=-1,
)

print("\n  Training wind model...")
wind_model.fit(train[WIND_FEATURES], train['wind_gen'],
               eval_set=[(test[WIND_FEATURES], test['wind_gen'])],
               verbose=False)

wind_preds = wind_model.predict(test[WIND_FEATURES]).clip(min=0)
wind_mae   = mean_absolute_error(test['wind_gen'], wind_preds)
wind_r2    = r2_score(test['wind_gen'], wind_preds)

print(f"  Wind MAE : {wind_mae:,.0f} MW")
print(f"  Wind R²  : {wind_r2:.4f}")
joblib.dump(wind_model, 'models/wind_model.pkl')
print("  Saved → models/wind_model.pkl")


# ============================================================
# PART F — GAP ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("  PART F — SUPPLY-DEMAND GAP ANALYSIS")
print("=" * 60)

# ── Load demand predictions from Step 2 ──────────────────────
try:
    demand_preds = pd.read_csv(
        'demand_predictions.csv',
        index_col=0, parse_dates=True
    )
    # Use best model (XGBoost)
    demand_col = 'pred_xgb' if 'pred_xgb' in demand_preds.columns \
                 else 'load_actual'
    # Align index with test set
    common_idx     = test.index.intersection(demand_preds.index)
    pred_demand    = demand_preds.loc[common_idx, demand_col].values
    actual_demand  = demand_preds.loc[common_idx, 'load_actual'].values
    pred_solar     = solar_model.predict(
                        test.loc[common_idx, SOLAR_FEATURES]).clip(min=0)
    pred_wind      = wind_model.predict(
                        test.loc[common_idx, WIND_FEATURES]).clip(min=0)
    print("  ✅ Demand predictions loaded from Step 2")
except FileNotFoundError:
    print("  ⚠️  demand_predictions.csv not found — using actual load")
    common_idx    = test.index
    pred_demand   = test['load_actual'].values
    actual_demand = test['load_actual'].values
    pred_solar    = solar_preds
    pred_wind     = wind_preds

# ── Calculate gaps ────────────────────────────────────────────
pred_supply     = pred_solar + pred_wind
actual_supply   = test.loc[common_idx, 'solar_gen'].values \
                + test.loc[common_idx, 'wind_gen'].values

# Positive gap = deficit (need more power)
# Negative gap = surplus (too much renewable)
pred_gap   = pred_demand  - pred_supply
actual_gap = actual_demand - actual_supply

print(f"\n  Predicted supply  : mean={pred_supply.mean():,.0f} MW")
print(f"  Predicted demand  : mean={pred_demand.mean():,.0f} MW")
print(f"  Mean predicted gap: {pred_gap.mean():,.0f} MW")
print(f"\n  % hours deficit   : {(pred_gap > 0).mean()*100:.1f}%")
print(f"  % hours surplus   : {(pred_gap <= 0).mean()*100:.1f}%")
print(f"  Max deficit       : {pred_gap.max():,.0f} MW")
print(f"  Max surplus       : {pred_gap.min():,.0f} MW")


# ============================================================
# PART G — INTERVENTION RECOMMENDATION ENGINE
# ============================================================
print("\n" + "=" * 60)
print("  PART G — INTERVENTION RECOMMENDATION ENGINE")
print("=" * 60)

# ── Thresholds (MW) ──────────────────────────────────────────
THR_CRIT_DEF  =  8000   # Critical deficit
THR_HIGH_DEF  =  3000   # High deficit
THR_LOW_DEF   =   500   # Low deficit
THR_LOW_SUR   =  -500   # Small surplus
THR_HIGH_SUR  = -3000   # Large surplus
THR_CRIT_SUR  = -8000   # Critical surplus


def recommend(gap_mw):
    """Return (level, action, detail) for a given gap in MW."""
    if gap_mw > THR_CRIT_DEF:
        return ('CRITICAL',
                '🔴 Emergency import + backup generators',
                f'Deficit of {gap_mw:,.0f} MW — activate emergency response, '
                f'import from neighbouring grids, spin up gas peakers immediately.')
    elif gap_mw > THR_HIGH_DEF:
        return ('HIGH',
                '🟠 Demand response + gas peakers',
                f'Deficit of {gap_mw:,.0f} MW — trigger industrial demand response, '
                f'activate dispatchable gas generation.')
    elif gap_mw > THR_LOW_DEF:
        return ('MODERATE',
                '🟡 Minor demand response',
                f'Deficit of {gap_mw:,.0f} MW — request voluntary load reduction '
                f'from industrial consumers.')
    elif gap_mw >= THR_LOW_SUR:
        return ('BALANCED',
                '🟢 Grid balanced — no action needed',
                f'Gap of {gap_mw:,.0f} MW — within normal operating range.')
    elif gap_mw >= THR_HIGH_SUR:
        return ('SURPLUS',
                '🔵 Charge storage / export to neighbours',
                f'Surplus of {abs(gap_mw):,.0f} MW — direct excess to battery '
                f'storage or export via interconnectors.')
    elif gap_mw >= THR_CRIT_SUR:
        return ('HIGH SURPLUS',
                '🟣 Curtail wind + charge all storage',
                f'Surplus of {abs(gap_mw):,.0f} MW — curtail onshore wind, '
                f'maximize storage charging, ramp up pumped hydro.')
    else:
        return ('CRITICAL SURPLUS',
                '⚫ Emergency curtailment',
                f'Surplus of {abs(gap_mw):,.0f} MW — emergency grid frequency '
                f'protection, curtail all curtailable generation immediately.')


# ── Apply recommendations to test set ────────────────────────
levels, actions, details = zip(*[recommend(g) for g in pred_gap])

results_df = pd.DataFrame({
    'actual_demand'  : actual_demand,
    'pred_demand'    : pred_demand,
    'pred_solar'     : pred_solar,
    'pred_wind'      : pred_wind,
    'pred_supply'    : pred_supply,
    'pred_gap'       : pred_gap,
    'actual_gap'     : actual_gap,
    'level'          : levels,
    'action'         : actions,
    'detail'         : details,
}, index=common_idx)

# ── Intervention statistics ───────────────────────────────────
level_counts = pd.Series(levels).value_counts()
print("\n  Intervention breakdown:")
print(f"  {'Level':<20} {'Hours':>8}  {'% of time':>10}")
print("  " + "-" * 42)
for lvl, cnt in level_counts.items():
    print(f"  {lvl:<20} {cnt:>8,}   {cnt/len(levels)*100:>8.1f}%")

results_df.to_csv('outputs/gap_interventions.csv',index=True)
print("\n  Saved → outputs/gap_interventions.csv")


# ============================================================
# PART H — VISUALISATIONS
# ============================================================
print("\n" + "=" * 60)
print("  PART H — GENERATING PLOTS")
print("=" * 60)

# ── Colour map for intervention levels ───────────────────────
LEVEL_COLORS = {
    'BALANCED':        COLORS['safe'],
    'MODERATE':        COLORS['warning'],
    'HIGH':            '#FF7043',
    'CRITICAL':        COLORS['critical'],
    'SURPLUS':         '#29B6F6',
    'HIGH SURPLUS':    '#7E57C2',
    'CRITICAL SURPLUS':'#37474F',
}

# ── PLOT 10: 2-week overview ──────────────────────────────────
WINDOW  = 24 * 14   # 14 days
idx_w   = results_df.index[:WINDOW]
r_w     = results_df.iloc[:WINDOW]

fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
fig.suptitle('Supply–Demand Gap  —  2-Week Test Sample',
             fontsize=14, color='white', fontweight='bold', y=1.01)

# Panel 1: Supply breakdown
axes[0].fill_between(idx_w, r_w['pred_solar'],
                     alpha=0.65, color=COLORS['solar'], label='Predicted solar')
axes[0].fill_between(idx_w, r_w['pred_solar'] + r_w['pred_wind'],
                     r_w['pred_solar'],
                     alpha=0.55, color=COLORS['wind'],  label='Predicted wind')
axes[0].plot(idx_w, r_w['pred_demand'],
             color=COLORS['load'], lw=2, label='Predicted demand')
axes[0].set_ylabel('MW')
axes[0].set_title('Predicted Supply (solar + wind) vs Demand')
axes[0].legend(fontsize=9, loc='upper right')
axes[0].grid(True)

# Panel 2: Gap
axes[1].fill_between(idx_w, r_w['pred_gap'],
                     where=r_w['pred_gap'] > 0,
                     color=COLORS['deficit'], alpha=0.7,
                     label='Deficit (need more)')
axes[1].fill_between(idx_w, r_w['pred_gap'],
                     where=r_w['pred_gap'] <= 0,
                     color=COLORS['surplus'], alpha=0.5,
                     label='Surplus (too much)')
axes[1].axhline(0, color='white', lw=0.8, alpha=0.5)
axes[1].axhline( THR_HIGH_DEF, color='#FF7043', lw=0.8,
                linestyle=':', alpha=0.6, label='High deficit threshold')
axes[1].axhline(-THR_HIGH_DEF, color='#29B6F6', lw=0.8,
                linestyle=':', alpha=0.6, label='High surplus threshold')
axes[1].set_ylabel('Gap (MW)')
axes[1].set_title('Supply–Demand Gap  (positive = deficit)')
axes[1].legend(fontsize=8, loc='upper right')
axes[1].grid(True)

# Panel 3: Intervention levels as coloured bands
level_to_num = {
    'CRITICAL': 4, 'HIGH': 3, 'MODERATE': 2,
    'BALANCED': 1, 'SURPLUS': 0,
    'HIGH SURPLUS': -1, 'CRITICAL SURPLUS': -2,
}
nums = [level_to_num.get(l, 1) for l in r_w['level']]
axes[2].bar(idx_w, [1]*len(idx_w), width=0.04,
            color=[LEVEL_COLORS.get(l, 'gray') for l in r_w['level']],
            align='edge', alpha=0.85)
axes[2].set_yticks([])
axes[2].set_title('Recommended Intervention Level')

# Legend for panel 3
patches = [mpatches.Patch(color=c, label=l, alpha=0.85)
           for l, c in LEVEL_COLORS.items()
           if l in set(r_w['level'])]
axes[2].legend(handles=patches, fontsize=8,
               loc='upper right', ncol=3)

plt.tight_layout()
plt.savefig('plot10_gap_overview.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  Saved → plot10_gap_overview.png")


# ── PLOT 11: Supply model accuracy ───────────────────────────
SAMPLE = min(24 * 7, len(test))   # 1 week

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
fig.suptitle('Solar & Wind Prediction Accuracy  —  1-Week Sample',
             fontsize=13, color='white', fontweight='bold')

axes[0].plot(test.index[:SAMPLE],
             test['solar_gen'].values[:SAMPLE],
             color=COLORS['solar'],   lw=1.6, label='Actual solar')
axes[0].plot(test.index[:SAMPLE],
             solar_preds[:SAMPLE],
             color='white', lw=1.0, linestyle='--',
             alpha=0.7, label=f'Predicted solar  (MAE={solar_mae:,.0f} MW)')
axes[0].fill_between(test.index[:SAMPLE],
                     test['solar_gen'].values[:SAMPLE],
                     solar_preds[:SAMPLE],
                     alpha=0.15, color=COLORS['solar'])
axes[0].set_ylabel('MW')
axes[0].set_title('Solar Generation')
axes[0].legend(fontsize=9)
axes[0].grid(True)

axes[1].plot(test.index[:SAMPLE],
             test['wind_gen'].values[:SAMPLE],
             color=COLORS['wind'],    lw=1.6, label='Actual wind')
axes[1].plot(test.index[:SAMPLE],
             wind_preds[:SAMPLE],
             color='white', lw=1.0, linestyle='--',
             alpha=0.7, label=f'Predicted wind  (MAE={wind_mae:,.0f} MW)')
axes[1].fill_between(test.index[:SAMPLE],
                     test['wind_gen'].values[:SAMPLE],
                     wind_preds[:SAMPLE],
                     alpha=0.12, color=COLORS['wind'])
axes[1].set_ylabel('MW')
axes[1].set_title('Wind Generation')
axes[1].legend(fontsize=9)
axes[1].grid(True)

plt.tight_layout()
plt.savefig('plot11_supply_accuracy.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  Saved → plot11_supply_accuracy.png")


# ── PLOT 12: Intervention distribution ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Intervention Analysis',
             fontsize=13, color='white', fontweight='bold')

# Pie chart of intervention levels
lv_counts  = results_df['level'].value_counts()
lv_colors  = [LEVEL_COLORS.get(l, 'gray') for l in lv_counts.index]
axes[0].pie(lv_counts.values, labels=lv_counts.index,
            colors=lv_colors, autopct='%1.1f%%',
            textprops={'fontsize': 9, 'color': 'white'},
            startangle=90,
            wedgeprops={'edgecolor': '#0f1117', 'linewidth': 1.5})
axes[0].set_title('% of Hours per Intervention Level')

# Gap distribution histogram
axes[1].hist(results_df['pred_gap'], bins=80,
             color=COLORS['gap'], alpha=0.75, edgecolor='none')
for thr, label, col in [
    ( THR_CRIT_DEF, 'Critical deficit',  '#EF5350'),
    ( THR_HIGH_DEF, 'High deficit',      '#FF7043'),
    (-THR_HIGH_DEF, 'High surplus',      '#29B6F6'),
    (-THR_CRIT_DEF, 'Critical surplus',  '#7E57C2'),
]:
    axes[1].axvline(thr, color=col, lw=1.2,
                    linestyle='--', alpha=0.8, label=label)
axes[1].axvline(0, color='white', lw=1.0, alpha=0.5)
axes[1].set_xlabel('Predicted Gap (MW)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Gap Distribution with Intervention Thresholds')
axes[1].legend(fontsize=8)
axes[1].grid(True, axis='y')

plt.tight_layout()
plt.savefig('plot12_intervention_analysis.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  Saved → plot12_intervention_analysis.png")


# ── PLOT 13: Hourly gap patterns ─────────────────────────────
hourly_gap = results_df.groupby(results_df.index.hour)['pred_gap'].mean()
monthly_gap = results_df.groupby(results_df.index.month)['pred_gap'].mean()
months = ['Jan','Feb','Mar','Apr','May','Jun',
          'Jul','Aug','Sep','Oct','Nov','Dec']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Gap Patterns — When Does the Grid Need Help?',
             fontsize=13, color='white', fontweight='bold')

bar_colors_h = [COLORS['deficit'] if v > 0 else COLORS['surplus']
                for v in hourly_gap]
axes[0].bar(hourly_gap.index, hourly_gap.values,
            color=bar_colors_h, alpha=0.85)
axes[0].axhline(0, color='white', lw=0.8, alpha=0.5)
axes[0].set_xlabel('Hour of day')
axes[0].set_ylabel('Average Gap (MW)')
axes[0].set_title('Average Gap by Hour')
axes[0].grid(True, axis='y')

bar_colors_m = [COLORS['deficit'] if v > 0 else COLORS['surplus']
                for v in monthly_gap]
axes[1].bar(range(1, len(monthly_gap)+1), monthly_gap.values,
            color=bar_colors_m, alpha=0.85)
axes[1].set_xticks(range(1, len(monthly_gap)+1))
axes[1].set_xticklabels(
    [months[i-1] for i in monthly_gap.index], fontsize=8)
axes[1].axhline(0, color='white', lw=0.8, alpha=0.5)
axes[1].set_ylabel('Average Gap (MW)')
axes[1].set_title('Average Gap by Month\n'
                  '(orange = deficit, blue = surplus)')
axes[1].grid(True, axis='y')

plt.tight_layout()
plt.savefig('plot13_gap_patterns.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  Saved → plot13_gap_patterns.png")


# ============================================================
# PART I — SAMPLE INTERVENTION REPORT
# ============================================================
print("\n" + "=" * 60)
print("  PART I — SAMPLE INTERVENTION REPORT (next 24 rows)")
print("=" * 60)

sample = results_df.head(24)
print(f"\n  {'Timestamp':<28} {'Gap (MW)':>10}  {'Level':<16}  Action")
print("  " + "-" * 90)
for ts, row in sample.iterrows():
    print(f"  {str(ts)[:25]:<28} "
          f"{row['pred_gap']:>+10,.0f}  "
          f"{row['level']:<16}  {row['action']}")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  FINAL SUMMARY — STEP 3")
print("=" * 60)

print(f"""
  Supply Models:
    Solar  MAE : {solar_mae:,.0f} MW   R² = {solar_r2:.4f}
    Wind   MAE : {wind_mae:,.0f} MW   R² = {wind_r2:.4f}

  Grid Balance:
    Hours in deficit : {(pred_gap > 0).mean()*100:.1f}%
    Hours in surplus : {(pred_gap <= 0).mean()*100:.1f}%
    Max deficit      : {pred_gap.max():,.0f} MW
    Max surplus      : {abs(pred_gap.min()):,.0f} MW

  ✅ STEP 3 COMPLETE — 4 plots saved:
     plot10_gap_overview.png
     plot11_supply_accuracy.png
     plot12_intervention_analysis.png
     plot13_gap_patterns.png

  Output saved:
     outputs/gap_interventions.csv

  NEXT STEP → generate_full_gap
  Run: python generate_full_gap.py
""")