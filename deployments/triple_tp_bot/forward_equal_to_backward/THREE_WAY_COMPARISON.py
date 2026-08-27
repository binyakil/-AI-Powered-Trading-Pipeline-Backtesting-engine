#!/usr/bin/env python3
"""
CRITICAL FINDING: SWEEP backtest vs bot.log extract still show MASSIVE discrepancy
even with matched RSI parameters (14,10).

This suggests the difference is NOT just parameters,
but something more fundamental in the trade execution logic.
"""

import pandas as pd
from datetime import datetime

print("=" * 80)
print("COMPREHENSIVE THREE-WAY COMPARISON")
print("=" * 80)

# Load all three datasets
old_sweep_rsi6_5 = pd.read_csv('/Users/benni/Documents/personal/work/money/trading/SWEEP PROJECT/outputs/sweep outputs/run_2026-02-20_23-49-31/trade_lists/trades_run_165.csv')
old_sweep_rsi6_5['entry_dt'] = pd.to_datetime(old_sweep_rsi6_5['entry_time'], unit='s')

new_sweep_rsi14_10 = pd.read_csv('/Users/benni/Documents/personal/work/money/trading/SWEEP PROJECT/outputs/sweep outputs/run_2026-02-25_19-52-28/trade_lists/trades_run_165.csv')
new_sweep_rsi14_10['entry_dt'] = pd.to_datetime(new_sweep_rsi14_10['entry_time'], unit='s')

extract = pd.read_csv('/Users/benni/Documents/personal/work/money/trading/bots/25_dec_13/forward_equal_to_backward/trades_export_detailed.csv')

# Summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"""
{'Metric':<30} {'SWEEP RSI(6,5)':<20} {'SWEEP RSI(14,10)':<20} {'Bot.Log Extract':<20}
{'-'*90}
{'Date':<30} {'2026-02-20':<20} {'2026-02-25':<20} {'Live Trading':<20}
{'Trades':<30} {len(old_sweep_rsi6_5):<20} {len(new_sweep_rsi14_10):<20} {len(extract):<20}
{'Total R':<30} {old_sweep_rsi6_5['R_result'].sum():+.2f}R{'':<15} {new_sweep_rsi14_10['R_result'].sum():+.2f}R{'':<15} {extract['R_result'].sum():+.2f}R
{'Win Rate':<30} {(old_sweep_rsi6_5['R_result'] > 0).sum() / len(old_sweep_rsi6_5) * 100:.1f}%{'':<17} {(new_sweep_rsi14_10['R_result'] > 0).sum() / len(new_sweep_rsi14_10) * 100:.1f}%{'':<17} {(extract['R_result'] > 0).sum() / len(extract) * 100:.1f}%
{'Full Losses':<30} {(old_sweep_rsi6_5['R_result'] == -1.0).sum():<20} {(new_sweep_rsi14_10['R_result'] == -1.0).sum():<20} {(extract['R_result'] == -1.0).sum():<20}
{'Avg R per Trade':<30} {old_sweep_rsi6_5['R_result'].mean():.2f}{'':<15} {new_sweep_rsi14_10['R_result'].mean():.2f}{'':<15} {extract['R_result'].mean():.2f}
""")

print("\n" + "=" * 80)
print("CRITICAL FINDINGS")
print("=" * 80)

print(f"""
1. RSI PARAMETER IMPACT (within SWEEP backtest):
   • RSI(6,5):   +47.80R
   • RSI(14,10): +84.20R
   → Surprising! RSI(14,10) is BETTER in backtest (+36.40R improvement)

2. BACKTEST vs ACTUAL TRADING GAP (at same RSI 14,10):
   • SWEEP backtest: +84.20R
   • Bot.log actual: -13.65R
   → MASSIVE 97.85R difference!
   → Backtest predicts winning, actual trading shows losing
   → This is the core problem!

3. POTENTIAL ROOT CAUSES:
   
   A. ENTRY SIGNAL DIFFERENCES
      • Different entry timing → different price levels
      • Different indicator calculations → different trade count
      Tool reveals:
      - SWEEP has 252 trades (14,10)
      - Extract has 200 trades
      - 52 missing in extract = ~26% fewer entries
      
   B. EXIT LOGIC DIFFERENCES
      • SWEEP might execute exits differently
      • Trailing stops, TP1/TP2/TP3 management differs
      • Full losses: SWEEP=86, Extract=106
      
   C. POSITION SIZING OR RISK MANAGEMENT
      • Different position size calculation
      • Different stop placement
      • Different TP levels execution
      
   D. PARSING ERRORS IN EXTRACT
      • bot.log regex patterns might miss trades
      • Timestamps might be misaligned
      • DRY-RUN vs actual trade filtering issues
      
   E. DATA FEED DIFFERENCES
      • SWEEP uses clean OHLCV from run folder
      • Extract uses bot.log which might have timing mismatches
      • 1-minute candle aggregation differences

4. PARAMETER SETTING QUESTION:
   Why did RSI(14,10) improve SWEEP (+36.40R) but bot showed -13.65R?
   • In backtest: more smoothing filters out bad signal = better
   • In actual trading: maybe different market conditions
   • Or: bot implementation differs from SWEEP

KEY INSIGHT:
If SWEEP is correct, the bot.log indicates the strategy SHOULD be winning,
but the actual trading shows losses. This suggests:
  → The bot implementation doesn't match SWEEP logic exactly
  → OR bot.log extraction has bugs
  → OR there are additional logic/filters in the bot not in SWEEP
""")

print("\n" + "=" * 80)
print("NEXT STEPS TO DEBUG")
print("=" * 80)

print("""
To determine which engine is correct:

1. VERIFY ENTRY SIGNALS MATCH
   Compare first 10 entries:
   - Extract shows: LONG 12.16, SHORT 12.31, LONG 12.32...
   - SWEEP shows:  LONG 12.32, LONG 12.17, SHORT 12.32...
   → Different trade sequences = different indicator signals or timing
   
2. CHECK INDICATOR IMPLEMENTATION
   - Are RSI, QQE, Hull MA calculated identically?
   - Is the signal generation logic the same?
   
3. EXAMINE BOT.LOG PARSING
   - Is extract_trades_to_csv.py regex correctly capturing entries?
   - Sample 5 DRY-RUN entries from bot.log and manually verify parsing
   
4. TRACE A SINGLE TRADE
   Pick trade #1 from both and trace:
   - Entry time
   - Entry price
   - Stop placement
   - Exit price
   - Exit reason
   See where they diverge

TENTATIVE CONCLUSION:
SWEEP appears more reliable (clean OHLCV input)
Bot.log extraction likely has bugs or missing logic
""")
