# ============================================================
# FIX: Re-generate plot14 and plot15 with corrected axes
# Run from your project folder AFTER generate_full_gap.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
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
    'load':  '#EF5350', 'deficit': '#FF7043', 'surplus': '#4FC3F7',
}

# ── Load the saved full gap CSV ───────────────────────────────
print("Loading outputs/full_gap_2015_2020.csv ...")
full = pd.read_csv('outputs/full_gap_2015_2020.csv',
                   index_col=0, parse_dates=True)

# FIX 1: Strip timezone → matplotlib handles naive datetimes cleanly
if full.index.tz is not None:
    full.index = full.index.tz_localize(None)

print(f"  Rows  : {len(full):,}")
print(f"  Range : {full.index.min().date()} → {full.index.max().date()}")

# ── Year-by-year aggregates ───────────────────────────────────
full['year_col'] = full.index.year
yearly = full.groupby('year_col').agg(
    avg_gap     = ('pred_gap',    'mean'),
    avg_demand  = ('pred_demand', 'mean'),
    avg_solar   = ('pred_solar',  'mean'),
    avg_wind    = ('pred_wind',   'mean'),
    pct_deficit = ('pred_gap',    lambda x: (x > 0).mean() * 100),
).round(0)

# ── Daily resample for time series (on tz-naive index) ────────
daily = full.resample('D').mean(numeric_only=True)


# ============================================================
# FIXED PLOT 14: Full 2015-2020 time series
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=False)
fig.suptitle('Germany Renewable Energy — Full Dataset  2015 → 2020',
             fontsize=15, color='white', fontweight='bold', y=1.01)

# Panel 1: Supply stack vs demand
axes[0].fill_between(daily.index, daily['pred_solar'],
                     alpha=0.70, color=COLORS['solar'], label='Solar (predicted)')
axes[0].fill_between(daily.index,
                     daily['pred_solar'] + daily['pred_wind'],
                     daily['pred_solar'],
                     alpha=0.55, color=COLORS['wind'],  label='Wind (predicted)')
axes[0].plot(daily.index, daily['pred_demand'],
             color=COLORS['load'], lw=1.2, label='Demand (predicted)')

# Panel 2: Gap
axes[1].fill_between(daily.index, daily['pred_gap'],
                     where=daily['pred_gap'] > 0,
                     color=COLORS['deficit'], alpha=0.75,
                     label='Deficit (conventional generation needed)')
axes[1].fill_between(daily.index, daily['pred_gap'],
                     where=daily['pred_gap'] <= 0,
                     color=COLORS['surplus'], alpha=0.55,
                     label='Surplus (excess renewable)')
axes[1].axhline(0, color='white', lw=0.8, alpha=0.5)

# Panel 3: Annual bar chart — FIX 2: use range(n) not year integers
yrs       = list(yearly.index)
n         = len(yrs)
positions = range(n)                       # 0,1,2,3,4 — not 2015,2016…

bar_colors = [COLORS['deficit'] if v > 0 else COLORS['surplus']
              for v in yearly['avg_gap']]
bars = axes[2].bar(positions, yearly['avg_gap'],
                   color=bar_colors, alpha=0.85, width=0.6)
axes[2].set_xticks(positions)
axes[2].set_xticklabels(yrs, color='#c8cad4')  # label with actual years
axes[2].axhline(0, color='white', lw=0.8, alpha=0.5)
for pos, val in zip(positions, yearly['avg_gap']):
    axes[2].text(pos, val + 300, f'{val:,.0f}',
                 ha='center', va='bottom', fontsize=9, color='white')

titles = [
    'Solar & Wind Supply vs Demand  (daily avg)',
    'Supply–Demand Gap — orange = deficit, blue = surplus  (daily avg)',
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
print("✅ Saved → plot14_full_range_overview.png")


# ============================================================
# FIXED PLOT 15: Year-by-year breakdown
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Year-by-Year Energy Breakdown  2015 → 2020',
             fontsize=13, color='white', fontweight='bold')

x = np.arange(n)
w = 0.25

# Grouped bars: solar / wind / load
axes[0].bar(x - w, yearly['avg_solar'].values,  w,
            color=COLORS['solar'],   alpha=0.85, label='Solar')
axes[0].bar(x,     yearly['avg_wind'].values,   w,
            color=COLORS['wind'],    alpha=0.85, label='Wind')
axes[0].bar(x + w, yearly['avg_demand'].values, w,
            color=COLORS['load'],    alpha=0.85, label='Load')
axes[0].set_xticks(x)
axes[0].set_xticklabels(yrs)
axes[0].set_ylabel('Average MW')
axes[0].set_title('Avg Solar, Wind & Load by Year')
axes[0].legend(fontsize=9, facecolor='#1a1d26', labelcolor='#c8cad4')

# % deficit bars
axes[1].bar(x, yearly['pct_deficit'].values,
            color=COLORS['deficit'], alpha=0.85, width=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(yrs)
axes[1].set_ylim(0, 105)
axes[1].set_ylabel('% of hours in deficit')
axes[1].set_title('% Hours Renewables Alone Cannot Cover Demand')
for i, val in enumerate(yearly['pct_deficit'].values):
    axes[1].text(i, val + 0.5, f'{val:.1f}%',
                 ha='center', va='bottom', fontsize=10, color='white')

for ax in axes:
    ax.set_facecolor('#1a1d26')
    ax.spines[:].set_color('#2e3145')
    ax.tick_params(colors='#8b8fa8')
    ax.grid(True, color='#2e3145', linestyle='--', alpha=0.5, axis='y')

plt.tight_layout()
plt.savefig('plot15_yearly_breakdown.png', dpi=150,
            bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("✅ Saved → plot15_yearly_breakdown.png")

print("\nDone! 2 plots regenerated correctly.")



print(f"""
{'=' * 60}
  ✅ 14,15 plots are fixed
{'=' * 60}

  NEXT STEP → Dashboard
  Run: streamlit run dashboard.py
""")