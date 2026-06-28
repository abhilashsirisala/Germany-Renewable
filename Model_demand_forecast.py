# ============================================================
# STEP 2: DEMAND PREDICTION MODEL
# Renewable Energy Load Balancing System
# Models: XGBoost | Random Forest | LSTM
# Benchmark to beat: ENTSOE MAE = 9,582 MW
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ── Install check ────────────────────────────────────────────
# pip install pandas numpy matplotlib scikit-learn xgboost tensorflow joblib

from sklearn.ensemble         import RandomForestRegressor
from sklearn.metrics          import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing    import MinMaxScaler
import xgboost as xgb
import joblib
import os

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
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
    'axes.titlesize':   12,
    'axes.labelsize':   10,
})

COLORS = {
    'actual':  '#EF5350',
    'xgb':     '#F5A623',
    'rf':      '#4FC3F7',
    'lstm':    '#AB47BC',
    'entsoe':  '#66BB6A',
    'error':   '#FF7043',
}

ENTSOE_MAE = 1686        # MW — benchmark from EDA
os.makedirs('models', exist_ok=True)


# ============================================================
# 1. LOAD CLEAN DATA
# ============================================================
print("=" * 58)
print("  STEP 1 — LOAD CLEAN DATA")
print("=" * 58)

df = pd.read_csv('germany_energy_clean.csv',
                 index_col='utc_timestamp',
                 parse_dates=True)

# Keep only the columns we need
KEEP = ['load_actual', 'load_forecast', 'hour',
        'day_of_week', 'month', 'year', 'is_weekend', 'season']
df = df[[c for c in KEEP if c in df.columns]].copy()

# Drop any rows with null load
df = df.dropna(subset=['load_actual'])
df = df.sort_index()

print(f"\n  Rows    : {len(df):,}")
print(f"  From    : {df.index.min()}")
print(f"  To      : {df.index.max()}")
print(f"  Target  : load_actual  (mean={df['load_actual'].mean():,.0f} MW)")


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 58)
print("  STEP 2 — FEATURE ENGINEERING")
print("=" * 58)

# ── Lag features (what happened N hours ago) ─────────────────
df['lag_1h']    = df['load_actual'].shift(1)    # 1 hr ago
df['lag_2h']    = df['load_actual'].shift(2)    # 2 hrs ago
df['lag_3h']    = df['load_actual'].shift(3)    # 3 hrs ago
df['lag_24h']   = df['load_actual'].shift(24)   # same hour yesterday
df['lag_48h']   = df['load_actual'].shift(48)   # same hour 2 days ago
df['lag_168h']  = df['load_actual'].shift(168)  # same hour last week

# ── Rolling averages (recent trend) ──────────────────────────
df['roll_6h']   = df['load_actual'].shift(1).rolling(6  ).mean()
df['roll_24h']  = df['load_actual'].shift(1).rolling(24 ).mean()
df['roll_168h'] = df['load_actual'].shift(1).rolling(168).mean()

# ── Rolling std (volatility) ─────────────────────────────────
df['roll_std_24h']  = df['load_actual'].shift(1).rolling(24 ).std()
df['roll_std_168h'] = df['load_actual'].shift(1).rolling(168).std()

# ── Cyclical encoding for hour & month ───────────────────────
# (converts 23 and 0 to be "close", not far apart)
df['hour_sin']  = np.sin(2 * np.pi * df['hour']  / 24)
df['hour_cos']  = np.cos(2 * np.pi * df['hour']  / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['dow_sin']   = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos']   = np.cos(2 * np.pi * df['day_of_week'] / 7)

# ── Drop rows with NaN from lags/rolling ─────────────────────
df = df.dropna()
print(f"  Rows after lag/rolling drop : {len(df):,}")

# ── Final feature list ────────────────────────────────────────
FEATURES = [
    'lag_1h', 'lag_2h', 'lag_3h',
    'lag_24h', 'lag_48h', 'lag_168h',
    'roll_6h', 'roll_24h', 'roll_168h',
    'roll_std_24h', 'roll_std_168h',
    'hour_sin', 'hour_cos',
    'month_sin', 'month_cos',
    'dow_sin', 'dow_cos',
    'is_weekend', 'year',
]

TARGET = 'load_actual'

print(f"  Features used : {len(FEATURES)}")
for f in FEATURES:
    print(f"    ✓ {f}")


# ============================================================
# 3. TRAIN / TEST SPLIT  (time-based — never random!)
# ============================================================
print("\n" + "=" * 58)
print("  STEP 3 — TIME-BASED TRAIN / TEST SPLIT")
print("=" * 58)

# Use last 20% of data as test set
split_idx = int(len(df) * 0.80)
train = df.iloc[:split_idx]
test  = df.iloc[split_idx:]

X_train = train[FEATURES]
y_train = train[TARGET]
X_test  = test[FEATURES]
y_test  = test[TARGET]

print(f"\n  Train : {len(train):,} rows  "
      f"({train.index.min().date()} → {train.index.max().date()})")
print(f"  Test  : {len(test):,}  rows  "
      f"({test.index.min().date()} → {test.index.max().date()})")
print(f"\n  ⚠️  Using time-based split (NOT random)")
print(f"     Random split would leak future data into training!")


# ============================================================
# 4. MODEL A — XGBoost
# ============================================================
print("\n" + "=" * 58)
print("  MODEL A — XGBoost")
print("=" * 58)

xgb_model = xgb.XGBRegressor(
    n_estimators      = 500,
    learning_rate     = 0.05,
    max_depth         = 6,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    random_state      = 42,
    verbosity         = 0,
    n_jobs            = -1,
)

print("\n  Training XGBoost...")
xgb_model.fit(
    X_train, y_train,
    eval_set          = [(X_test, y_test)],
    verbose           = False,
)

xgb_preds = xgb_model.predict(X_test)
xgb_mae   = mean_absolute_error(y_test, xgb_preds)
xgb_rmse  = np.sqrt(mean_squared_error(y_test, xgb_preds))
xgb_r2    = r2_score(y_test, xgb_preds)

print(f"  MAE  : {xgb_mae:,.0f} MW  "
      f"({'✅ BEATS' if xgb_mae < ENTSOE_MAE else '❌ DOES NOT beat'} "
      f"ENTSOE {ENTSOE_MAE:,} MW)")
print(f"  RMSE : {xgb_rmse:,.0f} MW")
print(f"  R²   : {xgb_r2:.4f}")

joblib.dump(xgb_model, 'models/xgb_demand_model.pkl')
print("  Saved → models/xgb_demand_model.pkl")


# ============================================================
# 5. MODEL B — Random Forest
# ============================================================
print("\n" + "=" * 58)
print("  MODEL B — Random Forest")
print("=" * 58)

rf_model = RandomForestRegressor(
    n_estimators = 200,
    max_depth    = 20,
    min_samples_split = 5,
    random_state = 42,
    n_jobs       = -1,
)

print("\n  Training Random Forest...")
rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)
rf_mae   = mean_absolute_error(y_test, rf_preds)
rf_rmse  = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_r2    = r2_score(y_test, rf_preds)

print(f"  MAE  : {rf_mae:,.0f} MW  "
      f"({'✅ BEATS' if rf_mae < ENTSOE_MAE else '❌ DOES NOT beat'} "
      f"ENTSOE {ENTSOE_MAE:,} MW)")
print(f"  RMSE : {rf_rmse:,.0f} MW")
print(f"  R²   : {rf_r2:.4f}")

joblib.dump(rf_model, 'models/rf_demand_model.pkl')
print("  Saved → models/rf_demand_model.pkl")


# ============================================================
# 6. MODEL C — LSTM (Deep Learning)
# ============================================================
print("\n" + "=" * 58)
print("  MODEL C — LSTM (Deep Learning)")
print("=" * 58)

try:
    import tensorflow as tf
    from tensorflow.keras.models   import Sequential
    from tensorflow.keras.layers   import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    LSTM_AVAILABLE = True

    # ── Scale features for LSTM ──────────────────────────────
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_sc = scaler_X.fit_transform(X_train)
    X_test_sc  = scaler_X.transform(X_test)
    y_train_sc = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_sc  = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    # ── Reshape to 3D for LSTM: (samples, timesteps, features) ─
    TIMESTEPS = 24   # look at last 24 hours

    def make_sequences(X, y, timesteps):
        Xs, ys = [], []
        for i in range(timesteps, len(X)):
            Xs.append(X[i - timesteps:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    print("\n  Building sequences (this takes a moment)...")
    X_tr_seq, y_tr_seq = make_sequences(X_train_sc, y_train_sc, TIMESTEPS)
    X_te_seq, y_te_seq = make_sequences(X_test_sc,  y_test_sc,  TIMESTEPS)
    y_te_actual         = y_test.values[TIMESTEPS:]

    # ── Build LSTM model ──────────────────────────────────────
    lstm_model = Sequential([
        LSTM(64, return_sequences=True,
             input_shape=(TIMESTEPS, len(FEATURES))),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1),
    ])

    lstm_model.compile(optimizer='adam', loss='mae')
    lstm_model.summary()

    early_stop = EarlyStopping(
        monitor='val_loss', patience=5,
        restore_best_weights=True, verbose=1
    )

    print("\n  Training LSTM (max 30 epochs, early stopping)...")
    history = lstm_model.fit(
        X_tr_seq, y_tr_seq,
        epochs          = 30,
        batch_size      = 128,
        validation_split= 0.1,
        callbacks       = [early_stop],
        verbose         = 1,
    )

    # ── Predict & inverse-scale ───────────────────────────────
    lstm_preds_sc = lstm_model.predict(X_te_seq).flatten()
    lstm_preds    = scaler_y.inverse_transform(
                        lstm_preds_sc.reshape(-1, 1)).flatten()

    lstm_mae  = mean_absolute_error(y_te_actual, lstm_preds)
    lstm_rmse = np.sqrt(mean_squared_error(y_te_actual, lstm_preds))
    lstm_r2   = r2_score(y_te_actual, lstm_preds)

    print(f"\n  MAE  : {lstm_mae:,.0f} MW  "
          f"({'✅ BEATS' if lstm_mae < ENTSOE_MAE else '❌ DOES NOT beat'} "
          f"ENTSOE {ENTSOE_MAE:,} MW)")
    print(f"  RMSE : {lstm_rmse:,.0f} MW")
    print(f"  R²   : {lstm_r2:.4f}")

    lstm_model.save('models/lstm_demand_model.keras')
    joblib.dump(scaler_X, 'models/scaler_X.pkl')
    joblib.dump(scaler_y, 'models/scaler_y.pkl')
    print("  Saved → models/lstm_demand_model.keras")

except ImportError:
    LSTM_AVAILABLE = False
    lstm_mae  = None
    lstm_rmse = None
    lstm_r2   = None
    lstm_preds = None
    print("  ⚠️  TensorFlow not installed.")
    print("  pip install tensorflow")
    print("  Skipping LSTM — XGBoost & RF results still saved.")


# ============================================================
# 7. MODEL COMPARISON
# ============================================================
print("\n" + "=" * 58)
print("  STEP 7 — MODEL COMPARISON")
print("=" * 58)

results = {
    'ENTSOE Forecast': {'mae': ENTSOE_MAE,  'rmse': None, 'r2': None},
    'Random Forest':   {'mae': rf_mae,  'rmse': rf_rmse,  'r2': rf_r2},
    'XGBoost':         {'mae': xgb_mae, 'rmse': xgb_rmse, 'r2': xgb_r2},
}
if LSTM_AVAILABLE and lstm_mae:
    results['LSTM'] = {'mae': lstm_mae, 'rmse': lstm_rmse, 'r2': lstm_r2}

print(f"\n  {'Model':<20} {'MAE (MW)':>12} {'RMSE (MW)':>12} {'R²':>8}  {'vs ENTSOE':>12}")
print("  " + "-" * 70)
for name, m in results.items():
    mae_str  = f"{m['mae']:>10,.0f}"
    rmse_str = f"{m['rmse']:>10,.0f}" if m['rmse'] else '         N/A'
    r2_str   = f"{m['r2']:>7.4f}"     if m['r2']   else '     N/A'
    if name == 'ENTSOE Forecast':
        vs = '  (baseline)'
    else:
        imp = ((ENTSOE_MAE - m['mae']) / ENTSOE_MAE) * 100
        vs  = f"  {'+' if imp > 0 else ''}{imp:.1f}% {'better' if imp > 0 else 'worse'}"
    print(f"  {name:<20} {mae_str} {rmse_str} {r2_str} {vs}")

# ── Pick best model ───────────────────────────────────────────
best_name = min(
    {k: v for k, v in results.items() if k != 'ENTSOE Forecast'},
    key=lambda k: results[k]['mae']
)
print(f"\n  🏆  Best model : {best_name}  "
      f"(MAE = {results[best_name]['mae']:,.0f} MW)")


# ============================================================
# 8. VISUALISATIONS
# ============================================================
print("\n" + "=" * 58)
print("  STEP 8 — GENERATING PLOTS")
print("=" * 58)

# ── PLOT 6: Model comparison bar chart ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Demand Prediction — Model Comparison',
             fontsize=14, color='white', fontweight='bold')

models      = list(results.keys())
maes        = [results[m]['mae'] for m in models]
bar_colors  = [COLORS['entsoe'], COLORS['rf'], COLORS['xgb']]
if LSTM_AVAILABLE and lstm_mae:
    bar_colors.append(COLORS['lstm'])

bars = axes[0].bar(models, maes, color=bar_colors, alpha=0.85,
                   edgecolor='none')
axes[0].axhline(ENTSOE_MAE, color=COLORS['entsoe'],
                lw=1.5, linestyle='--', alpha=0.6)
for bar, val in zip(bars, maes):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 80,
                 f'{val:,.0f}', ha='center', va='bottom',
                 fontsize=9, color='white')
axes[0].set_ylabel('MAE (MW) — lower is better')
axes[0].set_title('Mean Absolute Error per Model')
axes[0].grid(True, axis='y')
axes[0].tick_params(axis='x', labelsize=9)

# XGBoost feature importance
fi    = pd.Series(xgb_model.feature_importances_, index=FEATURES)
fi    = fi.sort_values(ascending=True).tail(12)
axes[1].barh(fi.index, fi.values, color=COLORS['xgb'], alpha=0.8)
axes[1].set_xlabel('Importance score')
axes[1].set_title('XGBoost — Top 12 Feature Importances')
axes[1].grid(True, axis='x')

plt.tight_layout()
plt.savefig('plot6_model_comparison.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  Saved → plot6_model_comparison.png")


# ── PLOT 7: Actual vs Predicted (1 week sample) ──────────────
SAMPLE = 24 * 7     # 1 week of hours
sample_actual  = y_test.values[:SAMPLE]
sample_xgb     = xgb_preds[:SAMPLE]
sample_rf      = rf_preds[:SAMPLE]
sample_entsoe  = test['load_forecast'].values[:SAMPLE] \
                 if 'load_forecast' in test.columns else None

fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(sample_actual, color=COLORS['actual'],
        lw=1.8, label='Actual load', zorder=5)
ax.plot(sample_xgb,    color=COLORS['xgb'],
        lw=1.2, label='XGBoost prediction', alpha=0.85)
ax.plot(sample_rf,     color=COLORS['rf'],
        lw=1.2, label='Random Forest prediction',
        alpha=0.85, linestyle='--')
if sample_entsoe is not None:
    ax.plot(sample_entsoe, color=COLORS['entsoe'],
            lw=1.0, label='ENTSOE forecast',
            alpha=0.7, linestyle=':')
if LSTM_AVAILABLE and lstm_preds is not None:
    ax.plot(lstm_preds[:SAMPLE], color=COLORS['lstm'],
            lw=1.2, label='LSTM prediction',
            alpha=0.85, linestyle='-.')

ax.set_xlabel('Hours (1 week sample from test set)')
ax.set_ylabel('Load (MW)')
ax.set_title('Actual vs Predicted Demand — 1 Week Sample',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, loc='upper right')
ax.grid(True)
plt.tight_layout()
plt.savefig('plot7_actual_vs_predicted.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  Saved → plot7_actual_vs_predicted.png")


# ── PLOT 8: Error distribution per model ─────────────────────
xgb_err = y_test.values - xgb_preds
rf_err  = y_test.values - rf_preds

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Prediction Error Distribution',
             fontsize=13, color='white', fontweight='bold')

axes[0].hist(xgb_err, bins=80, color=COLORS['xgb'],
             alpha=0.7, label='XGBoost', edgecolor='none')
axes[0].hist(rf_err,  bins=80, color=COLORS['rf'],
             alpha=0.5, label='Random Forest', edgecolor='none')
axes[0].axvline(0, color='white', lw=1.5, linestyle='--')
axes[0].set_xlabel('Prediction Error (MW)  [actual − predicted]')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Error Distribution  (centred at 0 = perfect)')
axes[0].legend(fontsize=9)
axes[0].grid(True, axis='y')

hour_xgb = pd.Series(np.abs(xgb_err), index=test.index
                     ).groupby(test.index.hour).mean()
hour_rf  = pd.Series(np.abs(rf_err),  index=test.index
                     ).groupby(test.index.hour).mean()
axes[1].plot(hour_xgb, color=COLORS['xgb'], lw=2,
             marker='o', markersize=4, label='XGBoost')
axes[1].plot(hour_rf,  color=COLORS['rf'],  lw=2,
             marker='s', markersize=4, linestyle='--',
             label='Random Forest')
axes[1].axhline(ENTSOE_MAE, color=COLORS['entsoe'],
                lw=1.5, linestyle=':', label=f'ENTSOE ({ENTSOE_MAE:,} MW)')
axes[1].set_xlabel('Hour of day')
axes[1].set_ylabel('Mean Absolute Error (MW)')
axes[1].set_title('Error by Hour — where do models struggle?')
axes[1].legend(fontsize=9)
axes[1].grid(True)

plt.tight_layout()
plt.savefig('plot8_error_analysis.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  Saved → plot8_error_analysis.png")


# ── PLOT 9: Scatter — predicted vs actual ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle('Predicted vs Actual Scatter  (closer to diagonal = better)',
             fontsize=13, color='white', fontweight='bold')

for ax, preds, name, col in zip(
    axes,
    [xgb_preds, rf_preds],
    ['XGBoost', 'Random Forest'],
    [COLORS['xgb'], COLORS['rf']],
):
    ax.scatter(y_test.values, preds, alpha=0.08,
               s=3, color=col, rasterized=True)
    lim = [y_test.min() * 0.95, y_test.max() * 1.05]
    ax.plot(lim, lim, color='white', lw=1.2,
            linestyle='--', alpha=0.6, label='Perfect fit')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Actual load (MW)')
    ax.set_ylabel('Predicted load (MW)')
    mae_val = mean_absolute_error(
        y_test.values, preds)
    ax.set_title(f'{name}   MAE={mae_val:,.0f} MW  '
                 f'R²={r2_score(y_test.values, preds):.4f}')
    ax.legend(fontsize=9)
    ax.grid(True)

plt.tight_layout()
plt.savefig('plot9_scatter.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  Saved → plot9_scatter.png")


# ============================================================
# 9. SAVE PREDICTIONS FOR NEXT STEP
# ============================================================
test_out = test[['load_actual']].copy() if 'load_actual' in test else pd.DataFrame(index=test.index)
test_out['load_actual']       = y_test.values
test_out['pred_xgb']          = xgb_preds
test_out['pred_rf']           = rf_preds
if LSTM_AVAILABLE and lstm_preds is not None:
    aligned = np.full(len(test_out), np.nan)
    aligned[TIMESTEPS:TIMESTEPS + len(lstm_preds)] = lstm_preds
    test_out['pred_lstm'] = aligned
if 'load_forecast' in test.columns:
    test_out['entsoe_forecast'] = test['load_forecast'].values

test_out.to_csv('demand_predictions.csv',index=True)
print("\n  Saved → demand_predictions.csv  (used in Step 3: balancing)")


# ============================================================
# 10. FINAL SUMMARY
# ============================================================
print("\n" + "=" * 58)
print("  FINAL RESULTS SUMMARY")
print("=" * 58)

print(f"\n  Benchmark (ENTSOE)   : {ENTSOE_MAE:>8,} MW MAE")
print(f"  ─────────────────────────────────────────")
print(f"  Random Forest        : {rf_mae:>8,.0f} MW MAE  "
      f"{'✅' if rf_mae < ENTSOE_MAE else '❌'}")
print(f"  XGBoost              : {xgb_mae:>8,.0f} MW MAE  "
      f"{'✅' if xgb_mae < ENTSOE_MAE else '❌'}")
if LSTM_AVAILABLE and lstm_mae:
    print(f"  LSTM                 : {lstm_mae:>8,.0f} MW MAE  "
          f"{'✅' if lstm_mae < ENTSOE_MAE else '❌'}")
print(f"\n  🏆  Best model : {best_name}  ({results[best_name]['mae']:,.0f} MW MAE)")
improvement = ((ENTSOE_MAE - results[best_name]['mae']) / ENTSOE_MAE) * 100
print(f"  📈  Improvement over ENTSOE : {improvement:.1f}%")

print(f"""
  ✅ STEP 2 COMPLETE — 4 plots saved:
     plot6_model_comparison.png
     plot7_actual_vs_predicted.png
     plot8_error_analysis.png
     plot9_scatter.png

  Models saved in  models/ folder:
     xgb_demand_model.pkl
     rf_demand_model.pkl
     {'lstm_demand_model.keras' if LSTM_AVAILABLE else '(LSTM skipped)'}

  NEXT STEP → Supply Prediction + Gap Analysis
  Run: python model_supply_gap.py
""")