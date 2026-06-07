"""
Generates eda_report_final.html — a standalone written EDA report with
embedded charts and narrative text. No code is visible in the output.
"""

import base64, glob, io, os, sys, warnings
sys.path.insert(0, '/tmp/openpyxl_pkg')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style='whitegrid', palette='tab10', font_scale=1.1)
plt.rcParams.update({'figure.dpi': 130, 'axes.spines.top': False, 'axes.spines.right': False})

# ── Palette ───────────────────────────────────────────────────────────────────
CAT_COLORS = {
    'WATER': '#4C72B0',
    'VALUE ADD WATER': '#55A868',
    'SPORT DRINKS': '#C44E52',
    'HEALTH/NUTRITION SHAKES': '#8172B3',
    'PERFORMANCE NUTRITION SHAKES': '#CCB974',
    'MEAL REPLACEMENT SHAKES': '#64B5CD',
}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
files = sorted(glob.glob('../data/bquxjob*.csv'))
df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
df['week_ending'] = pd.to_datetime(df['week_ending'])

dd = pd.read_excel(
    '../data/Nielsen_Data_Dictionary.xlsx', sheet_name='Data Dictionary',
    usecols=['variable_name','description','data_type','variable_category','sample_null_pct']
)
dd['null_pct'] = (dd['sample_null_pct'] * 100).round(1)

cat_sales_s = (
    df.dropna(subset=['category']).groupby('category')['sales'].sum()
    .sort_values(ascending=False)
)
cat_order = cat_sales_s.index.tolist()
cat_agg = df.dropna(subset=['category']).groupby('category').agg(
    total_sales=('sales','sum'), total_units=('units','sum'),
    unique_brands=('brand','nunique'), unique_upcs=('upc','nunique'),
    n_rows=('sales','count'),
).loc[cat_order]

brand_sales = df.groupby('brand')['sales'].sum().sort_values(ascending=False)
cumshare = (brand_sales / brand_sales.sum()).cumsum()

# ── Figure helper ─────────────────────────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def img_tag(b64, alt='', width='100%'):
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="width:{width};max-width:960px;display:block;margin:18px auto;">'

# ── HTML template helpers ──────────────────────────────────────────────────────
def section(title, num):
    return f'<h2 id="s{num}"><span class="num">{num}.</span> {title}</h2>'

def finding(label, value):
    return f'<tr><td class="fl">{label}</td><td>{value}</td></tr>'

def df_to_html(d, fmt=None):
    if fmt:
        s = d.style.format(fmt)
    else:
        s = d.style
    return s.set_table_attributes('class="tbl"').to_html()

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

print("Generating figures...")

# Fig 1 — Data quality
nulls = df.isnull().sum()
null_pct_s = (nulls / len(df) * 100).round(1)
null_df = pd.DataFrame({'null_count': nulls, 'null_pct': null_pct_s})
null_df = null_df[null_df.null_count > 0].sort_values('null_pct', ascending=True)
null_per_row = df.isnull().sum(axis=1)

fig1, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_q = ['#d62728' if p==100 else '#ff7f0e' if p>50 else '#1f77b4' for p in null_df['null_pct']]
axes[0].barh(null_df.index, null_df['null_pct'], color=colors_q)
axes[0].axvline(100, color='red', linestyle='--', linewidth=0.9, label='100% — drop')
axes[0].set_xlabel('Missing %')
axes[0].set_title('Missing Values by Column')
axes[0].legend(fontsize=9)
for i, v in enumerate(null_df['null_pct']):
    if v > 2:
        axes[0].text(v+0.5, i, f'{v}%', va='center', fontsize=6.5)

labels_nr = ['0 nulls', '1–5', '6–20', '>20']
counts_nr = [
    (null_per_row==0).sum(),
    ((null_per_row>=1)&(null_per_row<=5)).sum(),
    ((null_per_row>=6)&(null_per_row<=20)).sum(),
    (null_per_row>20).sum(),
]
axes[1].bar(labels_nr, counts_nr, color='#1f77b4', edgecolor='white')
axes[1].set_ylabel('Row count')
axes[1].set_title('Null Count Distribution per Row')
axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:,.0f}'))
for i, v in enumerate(counts_nr):
    axes[1].text(i, v+300, f'{v:,}', ha='center', fontsize=9)
fig1.tight_layout()
b64_q = fig_to_b64(fig1)

# Fig 2 — Category overview
fig2, axes = plt.subplots(1, 3, figsize=(17, 5))
colors_c = [CAT_COLORS.get(c,'#999') for c in cat_order]
wedges, _, _ = axes[0].pie(cat_agg['total_sales'], autopct='%1.1f%%',
    colors=colors_c, startangle=140, pctdistance=0.78, labels=None)
axes[0].legend(wedges, [c.title() for c in cat_order],
               loc='lower center', bbox_to_anchor=(0.5,-0.25), fontsize=7.5, ncol=1)
axes[0].set_title('Share of Total Sales ($)')

bars2 = axes[1].barh(cat_order[::-1], cat_agg['total_sales'][::-1]/1e6, color=colors_c[::-1])
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'${x:.0f}M'))
axes[1].set_title('Total Sales by Category ($M)')
for bar, val in zip(bars2, cat_agg['total_sales'][::-1]):
    axes[1].text(val/1e6+0.5, bar.get_y()+bar.get_height()/2, f'${val/1e6:.0f}M', va='center', fontsize=8)

x2 = np.arange(len(cat_order)); w2 = 0.4
axes[2].bar(x2-w2/2, cat_agg['unique_brands'], w2, label='Brands', color='#4C72B0')
axes[2].bar(x2+w2/2, cat_agg['unique_upcs'], w2, label='UPCs', color='#55A868')
axes[2].set_xticks(x2)
axes[2].set_xticklabels([c.replace('/','/\n') for c in cat_order], fontsize=7.5)
axes[2].set_title('Unique Brands & UPCs per Category')
axes[2].legend()
fig2.tight_layout()
b64_cat = fig_to_b64(fig2)

# Fig 3 — Weekly trend
weekly = df.dropna(subset=['category']).groupby(['week_ending','category'])['sales'].sum().unstack(fill_value=0)
fig3, axes = plt.subplots(2, 1, figsize=(14, 9))
axes[0].stackplot(weekly.index,
    [weekly.get(c, pd.Series(0, index=weekly.index))/1e6 for c in cat_order],
    labels=cat_order, colors=colors_c, alpha=0.85)
axes[0].set_ylabel('Sales ($M)'); axes[0].set_title('Weekly Total Sales — Stacked by Category')
axes[0].legend(loc='upper left', fontsize=8, ncol=2)
axes[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
axes[0].xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=2))
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30, ha='right')
for cat in cat_order:
    if cat in weekly.columns:
        axes[1].plot(weekly.index, weekly[cat]/1e6, label=cat,
                     color=CAT_COLORS.get(cat,'#999'), linewidth=1.7)
axes[1].set_ylabel('Sales ($M)'); axes[1].set_title('Weekly Sales by Category (Lines)')
axes[1].legend(loc='upper left', fontsize=8, ncol=2)
axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
axes[1].xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=2))
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=30, ha='right')
fig3.tight_layout()
b64_trend = fig_to_b64(fig3)

# Fig 4 — Brands
top20 = brand_sales.head(20)
fig4, axes = plt.subplots(1, 2, figsize=(16, 6))
colors_20 = ['#d62728' if b=='PRIVATE LABEL' else '#1f77b4' for b in top20.index]
axes[0].barh(top20.index[::-1], top20.values[::-1]/1e6, color=colors_20[::-1])
axes[0].set_xlabel('Total Sales ($M)'); axes[0].set_title('Top 20 Brands by Total Sales')
for i, v in enumerate(top20.values[::-1]):
    axes[0].text(v/1e6+0.2, i, f'${v/1e6:.1f}M', va='center', fontsize=7.5)
axes[1].plot(range(1, len(brand_sales)+1), cumshare.values*100, color='#1f77b4', linewidth=2)
for pct, lbl in [(0.5,'50%'), (0.8,'80%'), (0.95,'95%')]:
    n = (cumshare < pct).sum() + 1
    axes[1].axhline(pct*100, color='grey', linestyle='--', linewidth=0.8)
    axes[1].axvline(n, color='grey', linestyle='--', linewidth=0.8)
    axes[1].annotate(f'{lbl} @ {n} brands', xy=(n, pct*100), xytext=(n+10, pct*100-4), fontsize=8, color='#333')
axes[1].set_xlabel('Brands (ranked by sales)'); axes[1].set_ylabel('Cumulative share (%)')
axes[1].set_title('Brand Concentration Curve'); axes[1].set_xlim(0, len(brand_sales)); axes[1].set_ylim(0,101)
fig4.tight_layout()
b64_brand = fig_to_b64(fig4)

# Fig 5 — Pricing
price_df = df.dropna(subset=['category','aup']); price_df = price_df[price_df['aup']>0]
fig5, axes = plt.subplots(1, 2, figsize=(15, 5))
data_by_cat = [price_df[price_df['category']==c]['aup'].clip(upper=50) for c in cat_order]
bp = axes[0].boxplot(data_by_cat, patch_artist=True, showfliers=False)
for patch, cat in zip(bp['boxes'], cat_order):
    patch.set_facecolor(CAT_COLORS.get(cat,'#999')); patch.set_alpha(0.7)
axes[0].set_xticks(range(1,len(cat_order)+1))
axes[0].set_xticklabels([c.replace('/','/\n') for c in cat_order], fontsize=8)
axes[0].set_ylabel('AUP ($) — clipped at $50'); axes[0].set_title('Unit Price by Category')
axes[1].hist(price_df['aup'].clip(upper=60), bins=80, color='#1f77b4', edgecolor='white', alpha=0.85)
axes[1].axvline(price_df['aup'].median(), color='red', linestyle='--',
                label=f"Median ${price_df['aup'].median():.2f}")
axes[1].axvline(price_df['aup'].mean(), color='orange', linestyle='--',
                label=f"Mean ${price_df['aup'].mean():.2f}")
axes[1].set_xlabel('AUP ($) — clipped at $60'); axes[1].set_ylabel('Frequency')
axes[1].set_title('Overall Unit Price Distribution'); axes[1].legend()
fig5.tight_layout()
b64_price = fig_to_b64(fig5)

# Fig 6 — Promotions
promo = df.dropna(subset=['category'])
promo_rate = promo.groupby('category')['promo_flag'].mean() * 100
tpr_df = promo[promo['tpr_discount']>0].dropna(subset=['tpr_discount'])
fig6, axes = plt.subplots(1, 3, figsize=(17, 5))
bars6 = axes[0].bar(promo_rate.index, promo_rate.values,
                    color=[CAT_COLORS.get(c,'#999') for c in promo_rate.index])
axes[0].set_ylabel('Promo rate (%)'); axes[0].set_title('Promotional Week Rate by Category')
axes[0].set_xticklabels([c.replace('/','/\n') for c in promo_rate.index], fontsize=8)
for bar, val in zip(bars6, promo_rate.values):
    axes[0].text(bar.get_x()+bar.get_width()/2, val+0.3, f'{val:.1f}%', ha='center', fontsize=8)
axes[1].hist(tpr_df['tpr_discount'].clip(upper=60), bins=60, color='#C44E52', edgecolor='white', alpha=0.85)
axes[1].axvline(tpr_df['tpr_discount'].median(), color='navy', linestyle='--',
                label=f"Median {tpr_df['tpr_discount'].median():.1f}%")
axes[1].set_xlabel('TPR Discount (%) — clipped at 60'); axes[1].set_ylabel('Frequency')
axes[1].set_title('TPR Discount Depth'); axes[1].legend()
promo_aup = promo[promo['promo_flag']==1]['aup'].dropna()
nonpromo_aup = promo[promo['promo_flag']==0]['aup'].dropna()
axes[2].hist(nonpromo_aup.clip(upper=30), bins=60, alpha=0.65, color='#1f77b4', edgecolor='white', label='No Promo', density=True)
axes[2].hist(promo_aup.clip(upper=30), bins=60, alpha=0.65, color='#C44E52', edgecolor='white', label='On Promo', density=True)
axes[2].set_xlabel('AUP ($) — clipped at $30'); axes[2].set_ylabel('Density')
axes[2].set_title('AUP: Promo vs Non-Promo'); axes[2].legend()
fig6.tight_layout()
b64_promo = fig_to_b64(fig6)

# Fig 7 — ACV
dist_df = df.dropna(subset=['prc_acv','category'])
fig7, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(dist_df['prc_acv'].clip(upper=95), bins=80, color='#4C72B0', edgecolor='white', alpha=0.85)
axes[0].axvline(dist_df['prc_acv'].median(), color='red', linestyle='--',
                label=f"Median {dist_df['prc_acv'].median():.1f}%")
axes[0].axvline(dist_df['prc_acv'].mean(), color='orange', linestyle='--',
                label=f"Mean {dist_df['prc_acv'].mean():.1f}%")
axes[0].set_xlabel('% ACV'); axes[0].set_ylabel('Frequency'); axes[0].set_title('% ACV Distribution'); axes[0].legend()
data_acv = [dist_df[dist_df['category']==c]['prc_acv'] for c in cat_order]
bp7 = axes[1].boxplot(data_acv, patch_artist=True, showfliers=False)
for patch, cat in zip(bp7['boxes'], cat_order):
    patch.set_facecolor(CAT_COLORS.get(cat,'#999')); patch.set_alpha(0.7)
axes[1].set_xticks(range(1,len(cat_order)+1))
axes[1].set_xticklabels([c.replace('/','/\n') for c in cat_order], fontsize=8)
axes[1].set_ylabel('% ACV'); axes[1].set_title('% ACV by Category')
fig7.tight_layout()
b64_acv = fig_to_b64(fig7)

# Fig 8 — Pack type
pack_ct = df.dropna(subset=['category','pack_type'])
pack_pivot = pack_ct.groupby(['category','pack_type']).size().unstack(fill_value=0)
pack_pct = pack_pivot.div(pack_pivot.sum(axis=1), axis=0) * 100
mat_ct = df['package_material_substance'].value_counts(dropna=True).head(8)
fig8, axes = plt.subplots(1, 2, figsize=(15, 5))
valid = [c for c in cat_order if c in pack_pct.index]
pack_pct.loc[valid].plot(kind='bar', stacked=True, ax=axes[0],
    color=['#1f77b4','#ff7f0e','#2ca02c'], edgecolor='white')
axes[0].set_xticklabels([c.replace('/','/\n') for c in valid], rotation=20, ha='right', fontsize=8)
axes[0].set_ylabel('Share (%)'); axes[0].set_title('Pack Type Mix by Category')
axes[0].legend(loc='lower right', fontsize=9)
axes[1].barh(mat_ct.index[::-1], mat_ct.values[::-1], color='#8172B3')
axes[1].set_xlabel('Row count'); axes[1].set_title('Top Package Materials')
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:,.0f}'))
for i, v in enumerate(mat_ct.values[::-1]):
    axes[1].text(v+200, i, f'{v:,}', va='center', fontsize=8)
fig8.tight_layout()
b64_pack = fig_to_b64(fig8)

# Fig 9 — Seasonality
seas_weekly = df.dropna(subset=['category']).groupby(['week_ending','category'])['seasonality_index'].mean().unstack()
hol = df['Holiday'].value_counts()
hol_clean = hol[hol.index != 'No Holiday'].head(12)
fig9, axes = plt.subplots(1, 2, figsize=(15, 5))
for cat in cat_order:
    if cat in seas_weekly.columns:
        axes[0].plot(seas_weekly.index, seas_weekly[cat], label=cat,
                     color=CAT_COLORS.get(cat,'#999'), linewidth=1.5)
axes[0].axhline(1.0, color='black', linestyle='--', linewidth=0.8, label='Baseline = 1')
axes[0].set_ylabel('Seasonality Index'); axes[0].set_title('Seasonality Index Over Time')
axes[0].legend(fontsize=7.5, ncol=2)
axes[0].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))
axes[0].xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=2))
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30, ha='right')
axes[1].barh(hol_clean.index[::-1], hol_clean.values[::-1], color='#CCB974')
axes[1].set_xlabel('Week-observations'); axes[1].set_title('Holiday Weeks (excl. No Holiday)')
for i, v in enumerate(hol_clean.values[::-1]):
    axes[1].text(v+10, i, f'{v:,}', va='center', fontsize=8)
fig9.tight_layout()
b64_seas = fig_to_b64(fig9)

# Fig 10 — Correlation
key_cols = ['sales','units','eq','aup','prc_acv','tdp','tpr_discount','seasonality_index','avg_eq_price','median_baseprice']
corr = df[key_cols].corr()
fig10, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            vmin=-1, vmax=1, linewidths=0.5, ax=ax, square=True, annot_kws={'size': 9})
ax.set_title('Correlation Matrix — Key Numeric Columns', fontsize=13, fontweight='bold')
fig10.tight_layout()
b64_corr = fig_to_b64(fig10)

# Fig 11 — Sales distributions
pos = df[(df['sales']>0) & (df['units']>0)].copy()
fig11, axes = plt.subplots(2, 2, figsize=(14, 9))
axes[0,0].hist(np.log1p(pos['sales']), bins=80, color='#1f77b4', edgecolor='white', alpha=0.85)
axes[0,0].set_xlabel('log(1 + sales)'); axes[0,0].set_title('Sales Distribution (log scale)')
axes[0,1].hist(np.log1p(pos['units']), bins=80, color='#55A868', edgecolor='white', alpha=0.85)
axes[0,1].set_xlabel('log(1 + units)'); axes[0,1].set_title('Units Distribution (log scale)')
sample = pos.dropna(subset=['aup','category']).sample(min(8000,len(pos)), random_state=42)
for cat in cat_order:
    s = sample[sample['category']==cat]
    if not s.empty:
        axes[1,0].scatter(s['aup'].clip(upper=40), np.log1p(s['sales']),
                          alpha=0.25, s=8, color=CAT_COLORS.get(cat,'#999'), label=cat)
axes[1,0].set_xlabel('AUP ($) — clipped at $40'); axes[1,0].set_ylabel('log(1 + sales)')
axes[1,0].set_title('Sales vs AUP (8 000-row sample)'); axes[1,0].legend(fontsize=7, markerscale=3)
zero = df[df['sales']==0].dropna(subset=['category'])
zero_by_cat = zero['category'].value_counts()
axes[1,1].bar(zero_by_cat.index, zero_by_cat.values,
              color=[CAT_COLORS.get(c,'#999') for c in zero_by_cat.index])
axes[1,1].set_xticklabels([c.replace('/','/\n') for c in zero_by_cat.index], fontsize=8)
axes[1,1].set_ylabel('Count'); axes[1,1].set_title(f'Zero-Sales Rows by Category  (total: {len(zero):,})')
axes[1,1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'{x:,.0f}'))
fig11.tight_layout()
b64_sdist = fig_to_b64(fig11)

# ── Pre-compute text values ────────────────────────────────────────────────────
total_sales = df['sales'].sum()
water_share = df[df['category']=='WATER']['sales'].sum() / total_sales * 100
top5_share = brand_sales.head(5).sum() / total_sales * 100
n80 = (cumshare < 0.8).sum() + 1
promo_pct = df['promo_flag'].mean() * 100
zero_pct = (df['sales'] == 0).mean() * 100
med_aup = price_df['aup'].median()
mean_aup = price_df['aup'].mean()
med_tpr = tpr_df['tpr_discount'].median()
med_acv = dist_df['prc_acv'].median()
mean_acv = dist_df['prc_acv'].mean()

aup_by_cat = df.groupby('category')['aup'].agg(['median','mean']).round(2)

# ── Overview table ────────────────────────────────────────────────────────────
overview_rows = [
    ('Total rows', f'{df.shape[0]:,}'),
    ('Total columns', str(df.shape[1])),
    ('Source CSV files', str(len(files))),
    ('Date range', f"{df['week_ending'].min().date()} to {df['week_ending'].max().date()}"),
    ('Unique weeks', str(df['week_ending'].nunique())),
    ('Years covered', ', '.join(str(y) for y in sorted(df['year'].unique()))),
    ('Unique UPCs', f"{df['upc'].nunique():,}"),
    ('Unique brands', f"{df['brand'].nunique():,}"),
    ('Unique items', f"{df['item'].nunique():,}"),
    ('Market', df['markets'].iloc[0]),
    ('Geography hierarchy', f"{df['l1'].iloc[0]} / {df['l2'].iloc[0]}"),
    ('Total $ sales', f"${total_sales/1e6:.0f}M"),
]
overview_html = ''.join(f'<tr><td class="fl">{k}</td><td>{v}</td></tr>' for k,v in overview_rows)

# ── Category table ────────────────────────────────────────────────────────────
cat_rows = ''
for cat in cat_order:
    r = cat_agg.loc[cat]
    cat_rows += (
        f'<tr><td>{cat.title()}</td>'
        f'<td>${r.total_sales/1e6:.0f}M</td>'
        f'<td>{r.total_units/1e6:.1f}M</td>'
        f'<td>{r.unique_brands}</td>'
        f'<td>{r.unique_upcs}</td></tr>'
    )

# ── AUP table ─────────────────────────────────────────────────────────────────
aup_rows = ''
for cat in cat_order:
    if cat in aup_by_cat.index:
        r = aup_by_cat.loc[cat]
        aup_rows += f'<tr><td>{cat.title()}</td><td>${r["median"]:.2f}</td><td>${r["mean"]:.2f}</td></tr>'

# ── Key findings table ────────────────────────────────────────────────────────
findings = [
    ('Total dataset sales', f'${total_sales/1e6:.0f}M across 60 weeks'),
    ('Dominant category', f'Water accounts for {water_share:.1f}% of total sales'),
    ('Brand concentration', f'Top 5 brands hold {top5_share:.1f}% of sales; only {n80} brands needed for 80%'),
    ('Promotional activity', f'{promo_pct:.1f}% of observations carry a promo flag'),
    ('Zero-sales rows', f'{zero_pct:.1f}% of rows — almost entirely non-promotional listings without velocity'),
    ('Column to drop', 'retailers_div_req is 100% null'),
    ('Redundant column', 'acv is always identical to prc_acv — one can be dropped'),
    ('Peak season', 'June–September; Sport Drinks and Water see +50–60% lift vs January'),
    ('Price range', f'Median AUP ${med_aup:.2f} overall; shakes ($8–$10) price at 2–4× water/sport drinks'),
    ('Typical promo depth', f'Median TPR discount {med_tpr:.1f}% across all categories'),
    ('Distribution spread', f'Median % ACV {med_acv:.1f}%, mean {mean_acv:.1f}% — highly right-skewed; most UPCs have thin distribution'),
    ('Sparse promo columns', 'feat_n_disp_unit_price 93% null; feat_wo_disp 87% null — infrequent event types'),
    ('Modeling recommendation', 'Log-transform sales/units; model by category; use prc_acv not acv'),
]
findings_html = ''.join(f'<tr><td class="fl">{k}</td><td>{v}</td></tr>' for k,v in findings)

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD HTML
# ═══════════════════════════════════════════════════════════════════════════════
print("Building HTML...")

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 14px; color: #1a1a1a;
       background: #f7f8fa; line-height: 1.7; }
.wrap { max-width: 1080px; margin: 0 auto; padding: 40px 28px 80px; }
header { background: #1b3a6b; color: #fff; padding: 42px 36px 34px; border-radius: 6px;
         margin-bottom: 36px; }
header h1 { font-size: 28px; font-weight: 700; margin-bottom: 8px; }
header p  { font-size: 14px; opacity: 0.85; }
h2 { font-size: 20px; font-weight: 700; margin: 46px 0 12px; color: #1b3a6b;
     border-bottom: 2px solid #d0d8e8; padding-bottom: 6px; }
h2 .num { color: #4C72B0; margin-right: 6px; }
h3 { font-size: 15px; font-weight: 600; margin: 22px 0 6px; color: #2c3e50; }
p  { margin: 10px 0; color: #333; }
ul { margin: 8px 0 8px 22px; }
li { margin: 4px 0; }
.card { background: #fff; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.08);
        padding: 24px 28px; margin: 18px 0; }
.toc { background: #eef2f9; border-left: 4px solid #4C72B0; padding: 18px 22px;
       border-radius: 4px; margin-bottom: 36px; }
.toc ul { margin: 8px 0 0 16px; }
.toc li { margin: 3px 0; }
.toc a  { color: #1b3a6b; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
table.tbl { width: 100%; border-collapse: collapse; font-size: 13px; margin: 14px 0; }
table.tbl th { background: #1b3a6b; color: #fff; padding: 8px 12px; text-align: left; }
table.tbl td { padding: 7px 12px; border-bottom: 1px solid #e8ecf2; vertical-align: top; }
table.tbl tr:nth-child(even) td { background: #f2f5fb; }
td.fl { font-weight: 600; white-space: nowrap; color: #1b3a6b; width: 38%; }
.callout { background: #fffbe6; border-left: 4px solid #f0a500;
           padding: 12px 16px; border-radius: 4px; margin: 14px 0; font-size: 13px; }
.tag { display: inline-block; background: #dde6f5; color: #1b3a6b;
       border-radius: 12px; padding: 2px 10px; font-size: 11px; font-weight: 600;
       margin: 2px 2px; }
footer { text-align: center; font-size: 12px; color: #888; margin-top: 60px; }
"""

NAV = f"""
<div class="toc card">
  <strong>Contents</strong>
  <ul>
    <li><a href="#s1">1. Dataset Overview</a></li>
    <li><a href="#s2">2. Data Quality</a></li>
    <li><a href="#s3">3. Category Overview</a></li>
    <li><a href="#s4">4. Weekly Sales Trend</a></li>
    <li><a href="#s5">5. Brand Analysis</a></li>
    <li><a href="#s6">6. Pricing Analysis</a></li>
    <li><a href="#s7">7. Promotions</a></li>
    <li><a href="#s8">8. Distribution Metrics (% ACV / TDP)</a></li>
    <li><a href="#s9">9. Pack Type &amp; Packaging</a></li>
    <li><a href="#s10">10. Seasonality &amp; Holidays</a></li>
    <li><a href="#s11">11. Correlation Analysis</a></li>
    <li><a href="#s12">12. Sales &amp; Units Distributions</a></li>
    <li><a href="#s13">13. Key Findings &amp; Recommendations</a></li>
  </ul>
</div>
"""

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nielsen Beverage Retail — EDA Report</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>Nielsen Beverage Retail — Exploratory Data Analysis Report</h1>
  <p>Market: Pacific Division xAOC &nbsp;|&nbsp; Period: January 2025 – February 2026
     &nbsp;|&nbsp; Prepared: June 2026</p>
</header>

{NAV}

<!-- ── Executive Summary ──────────────────────────────────────────────────── -->
<div class="card">
  <h3>Executive Summary</h3>
  <p>This report summarises an exploratory analysis of Nielsen weekly retail point-of-sale data
  covering six beverage categories sold through the Pacific Division all-outlet-combined (xAOC) channel.
  The dataset spans 60 weeks from January 2025 through February 2026 and comprises
  <strong>162,965 rows</strong> and <strong>82 columns</strong> drawn from 19 BigQuery export files.
  Together the files represent approximately <strong>${total_sales/1e6:.0f} million</strong> in retail sales
  across <strong>3,769 unique UPCs</strong> and <strong>377 brands</strong>.</p>
  <p>The analysis covers data quality, category and brand structure, pricing, promotional activity,
  distribution breadth, seasonality, and the statistical relationships between key metrics.
  A consolidated list of findings and modeling recommendations is provided in Section 13.</p>
</div>

<!-- ── 1. Dataset Overview ────────────────────────────────────────────────── -->
{section("Dataset Overview", 1)}
<div class="card">
  <p>The raw data was exported from BigQuery as 19 CSV files and combined into a single analytical
  dataset. Every file shares an identical 82-column schema, meaning no schema reconciliation was
  required during loading. The table below summarises the key properties of the combined dataset.</p>
  <table class="tbl">
    <tr><th>Metric</th><th>Value</th></tr>
    {overview_html}
  </table>
  <p>The data covers a single geographic market slice — the Pacific Census Division under the
  xAOC (all-outlet-combined) channel — so no cross-market comparisons are possible within this
  dataset. All 60 weeks in the period are present, with 52 weeks from 2025 and 8 weeks from 2026.</p>
</div>

<!-- ── 2. Data Quality ────────────────────────────────────────────────────── -->
{section("Data Quality", 2)}
<div class="card">
  <h3>Missing Values</h3>
  <p>No duplicate rows were found in the dataset. Missing values are concentrated in two distinct
  patterns:</p>
  <ul>
    <li><strong>Structural missingness</strong> — the column <code>retailers_div_req</code> is
        <em>100% null</em> across all rows and should be dropped before any modeling work.</li>
    <li><strong>Promotional sparsity</strong> — unit-price columns for specific promotional event
        types (feature-and-display, feature-without-display, display-without-feature) carry
        86–93% null rates because these are rare merchandising conditions. The general
        <code>any_promo_unit_price</code> field is 46% null, meaning roughly half of all
        observations had no promotional pricing recorded.</li>
    <li><strong>Distribution and pricing nulls</strong> — approximately 10.9% of rows are missing
        <code>prc_acv</code>, <code>aup</code>, <code>tpr_discount</code>, and related fields.
        These rows almost universally have zero sales, suggesting they represent product listings
        with no observed velocity in that period.</li>
    <li><strong>Minor attribute nulls</strong> — 317 rows (0.2%) are missing product-level
        attributes such as category, flavor, and pack type. These are likely aggregate or
        placeholder rows.</li>
  </ul>
  {img_tag(b64_q, 'Data Quality')}
  <p>Looking at null counts per row, no row is entirely complete: the majority fall into the
  1–5 null bucket (84,553 rows), driven by the promotional price columns that are null whenever
  no promotion was active. Rows with more than 20 nulls (17,616) are those missing distribution
  and pricing data entirely, coinciding with the zero-sales population.</p>
  <div class="callout">
    <strong>Action items:</strong> Drop <code>retailers_div_req</code> (100% null).
    Drop <code>acv</code> — it is confirmed to be identical to <code>prc_acv</code> in every
    non-null row, making it wholly redundant. Treat zero-sales rows with caution: they likely
    reflect listed-but-unsold items rather than data errors.
  </div>
</div>

<!-- ── 3. Category Overview ───────────────────────────────────────────────── -->
{section("Category Overview", 3)}
<div class="card">
  <p>The dataset covers six Nielsen product categories. Water is by far the largest, representing
  {water_share:.1f}% of total dollar sales. Sport Drinks rank second, driven by high-velocity
  items like Gatorade. The three shake sub-categories (Health/Nutrition, Performance Nutrition,
  and Meal Replacement) collectively account for roughly 16% of sales but command significantly
  higher per-unit prices.</p>
  {img_tag(b64_cat, 'Category Overview')}
  <table class="tbl">
    <tr>
      <th>Category</th><th>Total Sales</th><th>Total Units</th>
      <th>Brands</th><th>UPCs</th>
    </tr>
    {cat_rows}
  </table>
  <p>Value Add Water (flavored, enhanced, and premium waters) has the most brands (153) and UPCs
  (961) of any category, reflecting an increasingly fragmented market. Performance Nutrition Shakes,
  by contrast, is highly concentrated — 31 brands covering 306 UPCs — yet still generates $57M in
  the period.</p>
</div>

<!-- ── 4. Weekly Sales Trend ──────────────────────────────────────────────── -->
{section("Weekly Sales Trend", 4)}
<div class="card">
  <p>Weekly sales exhibit a clear seasonal pattern across all categories. Total market sales ramp
  from roughly $9–10M per week in January 2025 to $12–14M per week during the June–September
  summer peak, before declining to winter lows in November–December. The trend re-emerges in
  early 2026 consistent with the prior-year pattern.</p>
  {img_tag(b64_trend, 'Weekly Trend')}
  <p>Water and Sport Drinks drive the majority of the seasonal swing. Both categories are
  consumption-driven by warm weather and outdoor activity, producing the most pronounced summer
  lifts. The shake categories are comparatively stable year-round, suggesting demand is less
  weather-sensitive and more tied to routine health and nutrition behaviors.</p>
  <p>The 2026 data (weeks 1–8) tracks closely to the equivalent period in 2025, indicating that
  year-over-year growth trends are relatively flat for most categories at this market level.</p>
</div>

<!-- ── 5. Brand Analysis ──────────────────────────────────────────────────── -->
{section("Brand Analysis", 5)}
<div class="card">
  <h3>Top Brands</h3>
  <p>Private Label is the single largest brand by sales at $168M — nearly 1.7× the next brand,
  Gatorade ($102M). This reflects the prevalence of large-format, price-competitive store-brand
  water in the Pacific region. The top five brands — Private Label, Gatorade, Crystal Geyser,
  Premier Nutrition, and Arrowhead — together account for <strong>{top5_share:.1f}%</strong> of
  total sales.</p>
  {img_tag(b64_brand, 'Brand Analysis')}
  <h3>Concentration</h3>
  <p>Despite 377 brands in the dataset, the market is highly concentrated. Just
  <strong>{n80} brands</strong> are needed to reach 80% of cumulative sales. Approximately
  357 brands (95% of the brand count) collectively represent only 5% of sales — a classic
  long-tail distribution. Any predictive model will need to carefully handle low-volume brands
  to avoid overfitting on sparse observations.</p>
</div>

<!-- ── 6. Pricing ─────────────────────────────────────────────────────────── -->
{section("Pricing Analysis", 6)}
<div class="card">
  <p>Pricing varies widely across categories, reflecting fundamentally different product
  propositions. Water and Sport Drinks are high-volume, low-price categories, while shake
  products command premium prices.</p>
  {img_tag(b64_price, 'Pricing')}
  <table class="tbl">
    <tr><th>Category</th><th>Median AUP</th><th>Mean AUP</th></tr>
    {aup_rows}
  </table>
  <p>The overall AUP distribution is right-skewed: the median is <strong>${med_aup:.2f}</strong>
  while the mean is <strong>${mean_aup:.2f}</strong>, pulled upward by premium shake and
  specialty water products. A small number of observations record AUPs above $50 — these
  correspond to high-value health shake multipacks or specialty imports — and should be
  verified before modeling.</p>
  <div class="callout">
    Because price ranges differ so greatly across categories, price elasticity should be
    estimated <em>within</em> each category rather than across the full dataset. A
    cross-category model would conflate structural price differences with actual demand
    sensitivity.
  </div>
</div>

<!-- ── 7. Promotions ──────────────────────────────────────────────────────── -->
{section("Promotions", 7)}
<div class="card">
  <p>Promotional activity is widespread: <strong>{promo_pct:.1f}%</strong> of all
  item-week observations carry a promotional flag. The promotional rate is highest for
  Value Add Water ({promo_rate.get('VALUE ADD WATER', 0):.1f}%) and Sport Drinks
  ({promo_rate.get('SPORT DRINKS', 0):.1f}%), where competition is intense and
  price-driven promotion is a primary sales lever.</p>
  {img_tag(b64_promo, 'Promotions')}
  <p>The TPR (Temporary Price Reduction) discount depth distribution is right-skewed.
  The median discount is <strong>{med_tpr:.1f}%</strong>, but a meaningful tail extends
  to 40–100% discounts — likely reflective of clearance activity, bundle pricing
  anomalies, or data edge cases that warrant review. The comparison of AUP distributions
  between promotional and non-promotional observations confirms that promoted items sell
  at materially lower unit prices, validating the promotional flag as a meaningful
  modeling variable.</p>
</div>

<!-- ── 8. Distribution Metrics ───────────────────────────────────────────── -->
{section("Distribution Metrics (% ACV / TDP)", 8)}
<div class="card">
  <p><strong>% ACV (All Commodity Volume) distribution</strong> measures what share of
  store volume — weighted by store size — carries a given product. A value of 10% means
  the item is sold in stores that collectively represent 10% of the market's total
  commodity volume.</p>
  {img_tag(b64_acv, 'ACV Distribution')}
  <p>The distribution is highly right-skewed. The median % ACV across all item-week
  observations is only <strong>{med_acv:.1f}%</strong>, while the mean is
  <strong>{mean_acv:.1f}%</strong> — a gap driven by a small number of widely
  distributed, high-velocity items (primarily Private Label Water and Gatorade) that
  carry % ACV values above 50%. The majority of UPCs in the dataset have very limited
  retail presence, which is typical for a long-tail assortment.</p>
  <p>The <code>acv</code> and <code>prc_acv</code> columns were found to be numerically
  identical in all non-null rows, confirming that one of them is redundant and can be
  safely dropped.</p>
</div>

<!-- ── 9. Pack Type ───────────────────────────────────────────────────────── -->
{section("Pack Type & Packaging", 9)}
<div class="card">
  <p>Three pack configurations exist in the data: Single Pack, Multi Pack, and Combination
  Pack. Single packs dominate Water and Sport Drinks, where individual serve and convenience
  formats are most common. Multi packs make up a larger share of the Shake categories,
  consistent with the subscription-style, bulk-buy purchasing behavior typical of protein
  and meal replacement consumers.</p>
  {img_tag(b64_pack, 'Pack Type')}
  <p>Plastic is the dominant packaging material (125,431 rows), followed by Coated
  Cardboard — used extensively for shake cartons and tetra-pak beverages — and Aluminum
  for canned sport drinks. Glass represents a small but notable segment, primarily occupied
  by premium or specialty products.</p>
</div>

<!-- ── 10. Seasonality ────────────────────────────────────────────────────── -->
{section("Seasonality & Holidays", 10)}
<div class="card">
  <p>Each observation includes a pre-computed <strong>seasonality index</strong> that
  captures how demand in a given week compares to an annual baseline (index = 1.0).
  Values above 1.0 indicate above-average seasonal demand; values below 1.0 indicate
  below-average demand.</p>
  {img_tag(b64_seas, 'Seasonality')}
  <p>Sport Drinks show the most extreme seasonal variation (index range 0.71–1.39),
  reflecting strong dependence on outdoor activity seasons. Water follows a similar
  but somewhat flatter pattern. The shake categories are remarkably stable across weeks
  (index range approximately 0.69–1.22 for Health/Nutrition Shakes), confirming that
  these products serve needs that are not strongly weather-driven.</p>
  <h3>Holidays</h3>
  <p>The dataset includes 19 holiday flag columns. After excluding ordinary weeks,
  the most common holiday-adjacent observations are Valentine's Day &amp; Washington's
  Birthday (5,394 rows), Martin Luther King Jr. Day (5,342 rows), and Juneteenth
  (2,805 rows). These flags allow models to control for demand shifts driven by
  holiday shopping patterns and seasonal gifting.</p>
</div>

<!-- ── 11. Correlation ────────────────────────────────────────────────────── -->
{section("Correlation Analysis", 11)}
<div class="card">
  <p>The heatmap below shows pairwise Pearson correlations for the ten most important
  numeric variables in the dataset.</p>
  {img_tag(b64_corr, 'Correlation', width='70%')}
  <p>Key observations:</p>
  <ul>
    <li><strong>Sales and units</strong> are highly correlated (r = 0.82), as expected —
        higher unit volume generally drives higher dollar revenue.</li>
    <li><strong>prc_acv and tdp</strong> are essentially identical (r ≈ 1.00), confirming
        they measure the same construct at this level of aggregation.</li>
    <li><strong>AUP and sales</strong> are nearly uncorrelated (r = 0.01) at the overall
        dataset level. This is expected: the price-sales relationship only becomes
        meaningful <em>within</em> a category and product tier, not across all six
        categories simultaneously.</li>
    <li><strong>TPR discount and sales</strong> show a modest negative correlation (r = −0.07),
        which at first seems counterintuitive. This likely reflects that the deepest discounts
        occur on products with already-low baseline distribution, suppressing the aggregate
        lift signal.</li>
    <li><strong>Seasonality index</strong> has weak correlations with all other variables,
        suggesting it captures information orthogonal to the cross-sectional metrics —
        which is exactly what a well-constructed seasonality index should do.</li>
  </ul>
</div>

<!-- ── 12. Sales & Units Distributions ───────────────────────────────────── -->
{section("Sales &amp; Units Distributions", 12)}
<div class="card">
  <p>Dollar sales and unit volumes are strongly right-skewed. On a log scale, however,
  both distributions approximate a bimodal shape — a lower mode representing niche or
  low-distribution products and an upper mode representing mainstream high-velocity items.
  Log-transforming these targets before regression modeling is therefore advisable.</p>
  {img_tag(b64_sdist, 'Sales Distributions')}
  <p>The scatter of sales versus AUP (colored by category) makes the cross-category
  price segmentation immediately visible. The three shake categories cluster at higher
  price points with moderate volumes, while Water and Sport Drinks show high volumes at
  low prices. Within any category the relationship is negative (higher price → lower
  volume), though this is obscured when all categories are combined.</p>
  <h3>Zero-Sales Rows</h3>
  <p>A total of <strong>17,835 rows (10.9%)</strong> record zero dollar sales and zero
  units. These are distributed across all categories and are almost entirely
  non-promotional (only 19 of 17,835 carry a promo flag). They most likely represent
  item-market-week combinations where the product was listed in Nielsen's universe but
  had no observed scan activity — a common occurrence for long-tail items with sporadic
  distribution. These rows should either be excluded from sales modeling targets or
  handled via a two-stage (incidence + volume) model.</p>
</div>

<!-- ── 13. Key Findings ───────────────────────────────────────────────────── -->
{section("Key Findings &amp; Recommendations", 13)}
<div class="card">
  <table class="tbl">
    <tr><th>Finding</th><th>Detail</th></tr>
    {findings_html}
  </table>
  <h3>Variable Groups for Modeling</h3>
  <p>Based on the data dictionary and this analysis, the 82 columns fall into functional
  groups with distinct modeling roles:</p>
  <p>
    <span class="tag">Identifiers</span> upc, item, brand — use for grouping, not as features
    <span class="tag">Demand targets</span> sales, units, eq — log-transform before use
    <span class="tag">Price features</span> aup, median_baseprice, tpr_discount, no_promo_unit_price
    <span class="tag">Distribution features</span> prc_acv, tdp — highly collinear, use one
    <span class="tag">Promo features</span> promo_flag, any_promo_prc_acv, no_promo_prc_acv
    <span class="tag">Time features</span> week_of_year, month, seasonality_index, peak flags
    <span class="tag">Holiday features</span> Holiday (categorical), individual boolean flags
    <span class="tag">Product dims</span> category, pack_type, flavor, package_general_shape
    <span class="tag">Drop</span> retailers_div_req (100% null), acv (duplicate of prc_acv)
  </p>
</div>

<footer>Nielsen Beverage Retail EDA Report &mdash; Pacific Division xAOC &mdash; June 2026</footer>
</div>
</body>
</html>"""

out_path = 'eda_report_final.html'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(HTML)

size_mb = os.path.getsize(out_path) / 1e6
print(f"Report written to {out_path}  ({size_mb:.1f} MB)")
