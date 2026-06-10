"""
Generate realistic synthetic UAC program data matching the exact column schema.
Covers Oct 2014 – Sep 2023 (FY2015–FY2023), ~3287 rows.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

dates = pd.date_range("2014-10-01", "2023-09-30", freq="D")
n = len(dates)

# --- Annual baseline apprehensions (FY, roughly) ---
# FY2015~34k, FY2016~59k, FY2017~41k, FY2018~50k, FY2019~76k
# FY2020~30k (COVID), FY2021~147k, FY2022~152k, FY2023~118k
fy_daily_base = {
    2015: 93, 2016: 162, 2017: 112, 2018: 137,
    2019: 208, 2020: 82, 2021: 403, 2022: 416, 2023: 323,
}

def fy(d):
    return d.year if d.month < 10 else d.year + 1

apprehended = []
for d in dates:
    base = fy_daily_base.get(fy(d), 130)
    # seasonal peak: Jan-May
    month_factor = 1 + 0.35 * np.sin((d.month - 1) / 12 * 2 * np.pi + 0.5)
    # weekend dip in reporting
    wd_factor = 0.85 if d.weekday() >= 5 else 1.0
    val = int(base * month_factor * wd_factor * rng.lognormal(0, 0.15))
    apprehended.append(max(val, 0))

apprehended = np.array(apprehended)

# --- CBP custody (stock): cumulative inflow minus transfers, mean ~3-7 days hold ---
cbp_stock = []
stock = 1800
for i, d in enumerate(dates):
    inflow = apprehended[i]
    transfer_rate = 0.55 + 0.10 * rng.random()  # transfer 55-65% of stock daily
    transfer_rate *= (0.7 if d.weekday() >= 5 else 1.0)
    transferred_out = int(stock * transfer_rate * rng.lognormal(0, 0.08))
    transferred_out = min(transferred_out, stock + inflow)
    stock = max(0, stock + inflow - transferred_out)
    # mild reversion to a target
    target = apprehended[i] * 4
    stock = int(stock * 0.95 + target * 0.05)
    cbp_stock.append(max(stock, 0))

cbp_stock = np.array(cbp_stock)

# --- Transfers out of CBP (derived from stock changes + apprehensions) ---
transfers = []
for i in range(n):
    prev = cbp_stock[i-1] if i > 0 else 1800
    t = prev + apprehended[i] - cbp_stock[i]
    transfers.append(max(t, 0))
transfers = np.array(transfers)

# --- HHS care stock ---
hhs_stock = []
stock = 5000
for i, d in enumerate(dates):
    inflow = transfers[i]
    discharge_rate = 0.025 + 0.008 * rng.random()
    discharge_rate *= (0.6 if d.weekday() >= 5 else 1.0)
    # FY2021 surge: slower discharges
    if fy(d) == 2021:
        discharge_rate *= 0.75
    discharged = int(stock * discharge_rate * rng.lognormal(0, 0.10))
    discharged = min(discharged, stock + inflow)
    stock = max(0, stock + inflow - discharged)
    hhs_stock.append(max(stock, 0))

hhs_stock = np.array(hhs_stock)

# --- Discharges from HHS ---
discharges = []
for i in range(n):
    prev = hhs_stock[i-1] if i > 0 else 5000
    d_val = prev + transfers[i] - hhs_stock[i]
    discharges.append(max(d_val, 0))
discharges = np.array(discharges)

df = pd.DataFrame({
    "Date": dates,
    "Children apprehended and placed in CBP custody": apprehended,
    "Children in CBP custody": cbp_stock,
    "Children transferred out of CBP custody": transfers,
    "Children in HHS Care": hhs_stock,
    "Children discharged from HHS Care": discharges,
})

df.to_csv("/home/claude/uac_dashboard/uac_data.csv", index=False)
print(f"Generated {len(df)} rows")
print(df.head(3).to_string())
print("\nColumn sums:")
print(df.drop(columns="Date").sum())
