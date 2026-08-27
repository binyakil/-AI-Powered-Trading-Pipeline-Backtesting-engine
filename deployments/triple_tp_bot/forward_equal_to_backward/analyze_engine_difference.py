#!/usr/bin/env python3
"""
CORRECTED ANALYSIS: Both engines use the SAME OHLCV data (Jan-Feb 2026)
So why do they produce completely different results?
"""

import pandas as pd
from datetime import datetime

print("=" * 80)
print("ENGINE COMPARISON: SAME DATA, DIFFERENT RESULTS")
print("=" * 80)

# Load both trade CSVs
export_csv = pd.read_csv('trades_export_detailed.csv')
run_165_csv = pd.read_csv('/Users/benni/Documents/personal/work/money/trading/SWEEP PROJECT/outputs/sweep outputs/run_2026-02-20_23-49-31/trade_lists/trades_run_165.csv')

# Convert timestamps for Run 165
run_165_csv['entry_dt'] = pd.to_datetime(run_165_csv['entry_time'], unit='s')

print("\n📊 DATA VERIFICATION")
print("-" * 80)
print(f"extract_trades_to_csv.py (from bot.log):")
print(f"  Trades: {len(export_csv)}")
print(f"  Period: {export_csv['entry_time'].min()} to {export_csv['entry_time'].max()}")
print(f"  Total R: {export_csv['R_result'].sum():.2f}R")

print(f"\nSWEEP Run 165 (backtest):")
print(f"  Trades: {len(run_165_csv)}")
print(f"  Period: {run_165_csv['entry_dt'].min()} to {run_165_csv['entry_dt'].max()}")
print(f"  Total R: {run_165_csv['R_result'].sum():.2f}R")

print(f"\n✅ CONFIRMED: Both use Jan-Feb 2026 OHLCV data")

print("\n" + "=" * 80)
print("DIAGNOSTIC: WHAT'S DIFFERENT?")
print("=" * 80)

# Check trade sequences
print("\n1. TRADE ENTRY TIMING:")
export_long_shorts = export_csv[['side', 'entry_time', 'entry_px']].head(10)
run_165_sides = run_165_csv[['side', 'entry_time', 'entry_px']].head(10)

print(f"\nExtract (first 10 trades):")
print(export_long_shorts.to_string(index=False))

print(f"\nRun 165 (first 10 trades):")
run_165_sides['entry_dt'] = pd.to_datetime(run_165_sides['entry_time'], unit='s')
print(run_165_sides[['side', 'entry_time', 'entry_px']].to_string(index=False))

# Check profitability by side
print("\n2. PROFITABILITY BY SIDE:")
export_long = export_csv[export_csv['side'] == 'LONG']['R_result'].sum()
export_short = export_csv[export_csv['side'] == 'SHORT']['R_result'].sum()
run_165_long = run_165_csv[run_165_csv['side'] == 1]['R_result'].sum()
run_165_short = run_165_csv[run_165_csv['side'] == -1]['R_result'].sum()

print(f"\nExtract:  LONG={export_long:+.2f}R  SHORT={export_short:+.2f}R")
print(f"Run 165:  LONG={run_165_long:+.2f}R  SHORT={run_165_short:+.2f}R")

# Check loss distribution
print("\n3. LOSS DISTRIBUTION:")
export_losses = (export_csv['R_result'] == -1.0).sum()
run_165_losses = (run_165_csv['R_result'] == -1.0).sum()
print(f"\nFull losses (-1.0R):")
print(f"  Extract:  {export_losses} trades")
print(f"  Run 165:  {run_165_losses} trades")
print(f"  Difference: {export_losses - run_165_losses} extra losses in Extract")

# Check TP structure
print("\n4. TAKE PROFIT EXITS:")
print(f"\nExtract exit_reason distribution:")
print(export_csv['exit_reason'].value_counts() if 'exit_reason' in export_csv.columns else "N/A")

print(f"\nRun 165 exit_reason distribution:")
print(run_165_csv['exit_reason'].value_counts())

print("\n" + "=" * 80)
print("HYPOTHESIS: WHY THE DIFFERENCE?")
print("=" * 80)
print("""
Possible explanations:

A. ENTRY SIGNAL DIFFERENCES
   - Different indicator implementations (QQE, Hull MA, RSI)
   - extract_trades parses DRY-RUN entries from bot.log
   - SWEEP might use different threshold or calculation

B. EXIT LOGIC DIFFERENCES  
   - SWEEP has cleaner TP1, TP2, TP3 distribution
   - extract_trades shows mostly TP1_CLOSED_SL (complex interaction)
   - Different stop management or trail logic

C. PARSING ERRORS
   - extract_trades parses unstructured log file (error-prone)
   - SWEEP uses structured OHLCV data (more reliable)
   - Regex patterns might miss or misinterpret entries

D. TIMING/CANDLE AGGREGATION
   - bot.log might aggregate candles differently than SWEEP
   - Entry execution timing could differ
   - Signal evaluation on different bar closes

E. PARAMETERS
   - extract_trades reads from bot.py/indicator.py
   - SWEEP reads from config.json
   - Are RSI lengths, QQE params, position sizes identical?
""")
