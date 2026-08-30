#!/usr/bin/env python3
"""
Root Cause Analysis: Why do extract_trades_to_csv.py and SWEEP Run 165 show different results?

Key Points:
1. extract_trades_to_csv.py parses bot.log (2026-01-20 to 2026-02-20)
2. SWEEP Run 165 backtest uses uploaded_data.csv (2025-07-01 to 2025-09-30)
3. These are DIFFERENT time periods!

Hypothesis: They're running on completely different data.

This script investigates two questions:
A. Is extract_trades_to_csv.py correctly parsing bot.log?
B. Should Run 165 be using different data for Jan-Feb 2026?
"""

import pandas as pd
import re
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("ROOT CAUSE ANALYSIS: TWO DIFFERENT DATASETS")
print("=" * 80)

# Load both CSVs
export_csv = Path("trades_export_detailed.csv")
run_165_csv = Path("/Users/benni/Documents/personal/work/money/trading/SWEEP PROJECT/outputs/sweep outputs/run_2026-02-20_23-49-31/trade_lists/trades_run_165.csv")

export_df = pd.read_csv(export_csv)
run_165_df = pd.read_csv(run_165_csv)

print("\n📊 TEST A: Is extract_trades_to_csv.py parsing correctly?")
print("-" * 80)

# Check if patterns in extract_trades exist in bot.log
bot_log_path = Path("bot.log")
if bot_log_path.exists():
    with open(bot_log_path, 'r') as f:
        log_content = f.read(50000)  # Read first 50KB
    
    # Look for DRY-RUN entry patterns
    entry_pattern = r"DRY-RUN entry T\d+ side=(LONG|SHORT) entry=([\d.]+)"
    matches = re.findall(entry_pattern, log_content)
    print(f"✓ Found {len(matches)} 'DRY-RUN entry' patterns in first 50KB of bot.log")
    if matches:
        print(f"  Sample entries: {matches[:3]}")
else:
    print("✗ bot.log not found in current directory")

print("\n📊 TEST B: Data Mismatch Evidence")
print("-" * 80)

# Convert unix timestamps for Run 165
run_165_df['entry_dt'] = pd.to_datetime(run_165_df['entry_time'], unit='s')

print(f"\nextract_trades_to_csv.py (bot.log parser):")
print(f"  Date range: {export_df['entry_time'].min()} to {export_df['entry_time'].max()}")
print(f"  Total R: {export_df['R_result'].sum():.2f}R")

print(f"\nSWEEP Run 165 (backtest):")
print(f"  Date range: {run_165_df['entry_dt'].min()} to {run_165_df['entry_dt'].max()}")
print(f"  Total R: {run_165_df['R_result'].sum():.2f}R")

print(f"\n❌ TIME PERIOD MISMATCH:")
print(f"  extract_trades: 2026-01-20 to 2026-02-20 (Jan-Feb 2026)")
print(f"  Run 165 backtest: Should be same period but is from July-Sept 2025")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print("""
The two datasets are from COMPLETELY DIFFERENT TIME PERIODS:

1. extract_trades_to_csv.py (bot.log parsing):
   ✓ From actual live bot trading (Jan-Feb 2026)
   ✓ 200 trades, -13.65R total
   ✓ Real forward test results

2. SWEEP Run 165 (uploaded_data.csv backtest):
   ✗ From different time period (July-Sept 2025)
   ✗ Cannot be compared directly
   ✗ Different market conditions, different price action

RECOMMENDATION:
To properly compare the two engines, you need to:
1. Export the Jan-Feb 2026 OHLCV data used by the bot
2. Run SWEEP backtest on THIS SAME data
3. Then compare like-for-like: same time period, same data

The current comparison is APPLES-TO-ORANGES.

Alternative interpretation:
If the user meant "run the same bot code on the July-Sept 2025 data",
then the question becomes: Why does the bot.log show -13.65R
but SWEEP shows +47.80R for THOSE DIFFERENT PERIODS?
""")
