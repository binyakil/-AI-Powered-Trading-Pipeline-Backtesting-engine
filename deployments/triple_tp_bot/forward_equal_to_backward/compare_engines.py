#!/usr/bin/env python3
"""
Diagnostic script to compare two backtesting engines:
1. extract_trades_to_csv.py - Parses live bot.log
2. SWEEP PROJECT - Flask backtester
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Load both CSVs
export_csv = Path("trades_export_detailed.csv")
run_165_csv = Path("/Users/benni/Documents/personal/work/money/trading/SWEEP PROJECT/outputs/sweep outputs/run_2026-02-20_23-49-31/trade_lists/trades_run_165.csv")

print("=" * 80)
print("COMPARING TWO BACKTESTING ENGINES")
print("=" * 80)

# Read files
export_df = pd.read_csv(export_csv)
run_165_df = pd.read_csv(run_165_csv)

print(f"\nFile 1: extract_trades_to_csv.py (from bot.log)")
print(f"  - Trades: {len(export_df)}")
print(f"  - Start: {export_df['entry_time'].min()}")
print(f"  - End: {export_df['entry_time'].max()}")
print(f"  - Total R: {export_df['R_result'].sum():.2f}R")
print(f"  - Win rate: {(export_df['R_result'] > 0).sum() / len(export_df) * 100:.1f}%")

print(f"\nFile 2: SWEEP PROJECT run_165 (backtest)")
print(f"  - Trades: {len(run_165_df)}")

# Convert unix timestamps to datetime for run_165
run_165_df['entry_time_dt'] = pd.to_datetime(run_165_df['entry_time'], unit='s')
print(f"  - Start: {run_165_df['entry_time_dt'].min()}")
print(f"  - End: {run_165_df['entry_time_dt'].max()}")
print(f"  - Total R: {run_165_df['R_result'].sum():.2f}R")
print(f"  - Win rate: {(run_165_df['R_result'] > 0).sum() / len(run_165_df) * 100:.1f}%")

print("\n" + "=" * 80)
print("SIDE-BY-SIDE COMPARISON OF FIRST 10 TRADES")
print("=" * 80)

print("\n--- EXTRACT_TRADES (bot.log parser) ---")
for idx, row in export_df.head(10).iterrows():
    print(f"T{idx+1:3d} {row['side']:5s} Entry:{row['entry_px']:.5f} R_result:{row['R_result']:7.2f}")

print("\n--- RUN_165 (SWEEP backtest) ---")
for idx, row in run_165_df.head(10).iterrows():
    side_str = "LONG " if row['side'] == 1 else "SHORT"
    print(f"T{idx+1:3d} {side_str} Entry:{row['entry_px']:.5f} R_result:{row['R_result']:7.2f}")

print("\n" + "=" * 80)
print("KEY OBSERVATIONS")
print("=" * 80)

# Check if entry prices match
print("\n1. ENTRY PRICE MATCHING:")
print(f"   Export file shows entry_px values like: {export_df['entry_px'].head(5).values}")
print(f"   Run 165 shows entry_px values like: {run_165_df['entry_px'].head(5).values}")

# Check side distribution
print("\n2. SIDE DISTRIBUTION:")
export_long = (export_df['side'] == 'LONG').sum()
export_short = (export_df['side'] == 'SHORT').sum()
run_165_long = (run_165_df['side'] == 1).sum()
run_165_short = (run_165_df['side'] == -1).sum()
print(f"   Extract:  LONG={export_long}, SHORT={export_short}")
print(f"   Run 165:  LONG={run_165_long}, SHORT={run_165_short}")

# Check loss distribution
print("\n3. LOSS DISTRIBUTION (-1.0 R trades):")
export_losses = (export_df['R_result'] == -1.0).sum()
run_165_losses = (run_165_df['R_result'] == -1.0).sum()
print(f"   Extract:  {export_losses} trades with R_result = -1.0")
print(f"   Run 165:  {run_165_losses} trades with R_result = -1.0")

# Profitability per side
print("\n4. PROFITABILITY BY SIDE:")
export_long_r = export_df[export_df['side'] == 'LONG']['R_result'].sum()
export_short_r = export_df[export_df['side'] == 'SHORT']['R_result'].sum()
run_165_long_r = run_165_df[run_165_df['side'] == 1]['R_result'].sum()
run_165_short_r = run_165_df[run_165_df['side'] == -1]['R_result'].sum()
print(f"   Extract:  LONG={export_long_r:.2f}R, SHORT={export_short_r:.2f}R")
print(f"   Run 165:  LONG={run_165_long_r:.2f}R, SHORT={run_165_short_r:.2f}R")

print("\n" + "=" * 80)
print("HYPOTHESIS")
print("=" * 80)
print("""
The dramatic difference between the two engines suggests:

A. DIFFERENT DATA SOURCES:
   - Extract parses live bot.log (forward test, real execution)
   - Run 165 backtests on uploaded OHLCV data
   - Even if "same" data, there could be timing/precision differences

B. DIFFERENT ENTRY/EXIT TIMING:
   - Extract shows trades ~5 hours LATER (confirmed by timestamps)
   - This timing shift could lead to different price levels
   - Different price levels = different P&L outcomes

C. DIFFERENT BUSINESS LOGIC:
   - Extract might have additional filters or debounce logic NOT in SWEEP backtest
   - SWEEP might have simplified entry/exit rules
   - Position sizing, TP splits, or stop management could differ

D. DATA QUALITY ISSUES:
   - Extract parses unstructured bot.log (prone to parsing errors)
   - Run 165 uses structured OHLCV CSV (more reliable)
   - This suggests Run 165 might be MORE ACCURATE if bot.log parsing is buggy

NEXT STEPS:
1. Check if extract_trades_to_csv.py correctly parses entry_time from bot.log
2. Verify SWEEP uses the SAME OHLCV dataset
3. Compare the indicator signals at entry points
4. Check if debounce/other logic affects entry timing in extract vs Run 165
""")
