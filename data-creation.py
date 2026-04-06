import pandas as pd

# import data as csv

auto = pd.read_csv('Enterprise_Car_Deals_history.csv')

ccl_info = pd.read_csv('CCL - Deal Info.csv')

ccl_cost = pd.read_csv('CCL - Cost and Margin Sales & FI.csv')

ccm_info = pd.read_csv('CCM - Deal Info.csv')

ccm_cost = pd.read_csv('CCM - Cost and Margin Sales & FI.csv')

print(auto.head(10))

# step 1 - load and clean data

import pandas as pd

# Row 0 is a summary "Overall Total" row — drop it
auto = auto[auto['Unnamed: 0'].isna()].drop(columns=['Unnamed: 0'])

# step 2 join tables

import pandas as pd

# --- CCL ---
ccl_info = pd.read_csv("CCL - Deal Info.csv")
ccl_margin = pd.read_csv("CCL - Cost and Margin Sales & FI.csv")

# Rename so the join key matches
ccl_margin = ccl_margin.rename(columns={'Deal Number': 'Deal #'})

# Join on Deal # — left join keeps all deals even if margin data is missing
ccl = pd.merge(ccl_info, ccl_margin, on='Deal #', how='left')

# --- CCM --- (exact same pattern)
ccm_info = pd.read_csv("CCM - Deal Info.csv")
ccm_margin = pd.read_csv("CCM - Cost and Margin Sales & FI.csv")
ccm_margin = ccm_margin.rename(columns={'Deal Number': 'Deal #'})
ccm = pd.merge(ccm_info, ccm_margin, on='Deal #', how='left')

# --- Stack both Powersports datasets into one ---
# Add a source column so you know which store each deal came from
ccl['moto_store'] = 'CCL'
ccm['moto_store'] = 'CCM'

moto = pd.concat([ccl, ccm], ignore_index=True)
print(moto.shape)  # should be ~18,500 rows

# step 3 cleaning

# A helper function to clean a name string
def clean_name(name):
    if pd.isna(name):
        return None
    # Uppercase, strip whitespace, remove punctuation
    return name.upper().strip().replace('.', '').replace(',', ' ').replace('  ', ' ')

def clean_zip(z):
    if pd.isna(z):
        return None
    # Keep only first 5 digits (some zips are "96825.0" as float)
    return str(z).split('.')[0].strip().zfill(5)[:5]

# Apply to Auto
auto['name_key'] = auto['Buyer'].apply(clean_name)
auto['zip_key']  = auto['Zip'].apply(clean_zip)

# Apply to Moto
moto['name_key'] = moto['Buyer Name'].apply(clean_name)
moto['zip_key']  = moto['Zip Code'].apply(clean_zip)

# The match key is name + zip combined
auto['match_key'] = auto['name_key'] + '|' + auto['zip_key']
moto['match_key'] = moto['name_key'] + '|' + moto['zip_key']

# step 4 parse dates and money

# Parse dates
auto['Date'] = pd.to_datetime(auto['Date'], format='%m/%d/%Y', errors='coerce')
moto['Date'] = pd.to_datetime(moto['Fin Date'], format='%m/%d/%Y', errors='coerce')

# Parse Sale Price — strip $ and commas, then convert to float
auto['sale_amt'] = (
    auto['Sale Price']
    .str.replace('[\$,]', '', regex=True)
    .str.strip()
    .pipe(pd.to_numeric, errors='coerce')
)

# CCL doesn't have a sale price column — you'd add it when you get that data
# For now we can count deals and use a placeholder
moto['sale_amt'] = 0  # update this when you have the column

# step 5 build per customer summaries

# --- Auto summary per customer ---
auto_summary = auto.groupby('match_key').agg(
    auto_deals        = ('Date', 'count'),
    auto_total_spend  = ('sale_amt', 'sum'),
    auto_first_date   = ('Date', 'min'),
    auto_last_date    = ('Date', 'max'),
    auto_stores       = ('Store', lambda x: ', '.join(sorted(x.dropna().unique()))),
    auto_makes        = ('Make', lambda x: ', '.join(sorted(x.dropna().unique()))),
    auto_name         = ('Buyer', 'first'),
    auto_city         = ('City', 'first'),
    auto_state        = ('State', 'first'),
    auto_zip          = ('Zip', 'first'),
).reset_index()

# --- Moto summary per customer ---
moto_summary = moto.groupby('match_key').agg(
    moto_deals        = ('Date', 'count'),
    moto_total_spend  = ('sale_amt', 'sum'),
    moto_first_date   = ('Date', 'min'),
    moto_last_date    = ('Date', 'max'),
    moto_makes        = ('Make', lambda x: ', '.join(sorted(x.dropna().unique()))),
    moto_name         = ('Buyer Name', 'first'),
).reset_index()

# step 6: merge

# Outer join = keep ALL customers from both files
master = pd.merge(auto_summary, moto_summary, on='match_key', how='outer')

# Fill NaN deal counts with 0 (customer exists in one file but not the other)
master['auto_deals'] = master['auto_deals'].fillna(0).astype(int)
master['moto_deals'] = master['moto_deals'].fillna(0).astype(int)

# Use whichever name is available
master['customer_name'] = master['auto_name'].combine_first(master['moto_name'])

# step 6: calculate LTV metrics

# Total across both divisions
master['total_deals'] = master['auto_deals'] + master['moto_deals']
master['total_spend']  = master['auto_total_spend'].fillna(0) + master['moto_total_spend'].fillna(0)

# Earliest and latest purchase across both divisions
master['first_purchase'] = master[['auto_first_date', 'moto_first_date']].min(axis=1)
master['last_purchase']  = master[['auto_last_date',  'moto_last_date']].max(axis=1)

# Tenure in years
master['tenure_years'] = (
    (master['last_purchase'] - master['first_purchase']).dt.days / 365.25
).round(1)

# Purchase frequency (deals per year — avoid divide by zero)
master['deals_per_year'] = (
    master['total_deals'] / master['tenure_years'].replace(0, 1)
).round(2)

# Days since last purchase (recency)
today = pd.Timestamp.today()
master['days_since_last'] = (today - master['last_purchase']).dt.days

# Recency segment
def recency_segment(days):
    if pd.isna(days):   return 'Unknown'
    if days < 365:      return 'Active'
    if days < 1095:     return 'Lapsing'
    return 'Inactive'

master['recency_segment'] = master['days_since_last'].apply(recency_segment)

# step 7: cross-division flags

master['in_auto']         = master['auto_deals'] > 0
master['in_moto']         = master['moto_deals'] > 0
master['is_multi_division'] = master['in_auto'] & master['in_moto']

# Count distinct auto stores (multi-store within Auto)
master['auto_store_count'] = (
    master['auto_stores']
    .fillna('')
    .apply(lambda x: len([s for s in x.split(',') if s.strip()]))
)
master['is_multi_store'] = master['auto_store_count'] > 1

# step 8: RFM scoring

# Only score customers with enough data
scored = master[master['total_deals'] > 0].copy()

scored['R'] = pd.qcut(scored['days_since_last'].rank(method='first'),
                       q=5, labels=[5,4,3,2,1])  # lower days = better = higher score

scored['F'] = pd.qcut(scored['total_deals'].rank(method='first'),
                       q=5, labels=[1,2,3,4,5])  # more deals = higher score

scored['M'] = pd.qcut(scored['total_spend'].rank(method='first'),
                       q=5, labels=[1,2,3,4,5])  # more spend = higher score

scored['RFM_score'] = scored['R'].astype(int) + scored['F'].astype(int) + scored['M'].astype(int)

def rfm_tier(score):
    if score >= 13: return 'Platinum'
    if score >= 10: return 'Gold'
    if score >= 7:  return 'Silver'
    return 'Bronze'

scored['customer_tier'] = scored['RFM_score'].apply(rfm_tier)

# step 10: export to excel

from openpyxl import load_workbook

with pd.ExcelWriter("customer_ltv_report.xlsx", engine="openpyxl") as writer:

    # Tab 1: Full master list
    scored.to_excel(writer, sheet_name="All Customers", index=False)

    # Tab 2: Cross-division customers only
    scored[scored['is_multi_division']].to_excel(
        writer, sheet_name="Multi-Division", index=False)

    # Tab 3: Active customers only
    scored[scored['recency_segment'] == 'Active'].to_excel(
        writer, sheet_name="Active", index=False)

    # Tab 4: Platinum and Gold tiers
    scored[scored['customer_tier'].isin(['Platinum', 'Gold'])].to_excel(
        writer, sheet_name="Top Tier", index=False)

print("Done! Open customer_ltv_report.xlsx")