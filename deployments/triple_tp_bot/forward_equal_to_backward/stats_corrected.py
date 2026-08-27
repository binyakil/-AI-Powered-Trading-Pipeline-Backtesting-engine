#!/usr/bin/env python3
"""Calculate corrected backtest statistics from normalized R-multiples"""

import csv
from pathlib import Path

csv_path = Path("trades_export_detailed.csv")

with open(csv_path, 'r') as f:
    rows = list(csv.DictReader(f))

r_values = [float(row['R_result']) for row in rows if row['R_result']]
wins = [r for r in r_values if r > 0]
losses = [r for r in r_values if r < 0]

print('\n' + '='*80)
print('CORRECTED BACKTEST RESULTS: R-NORMALIZED (Proper R-Multiple System)')
print('='*80 + '\n')

print(f'Total Trades:           {len(r_values)}')
print(f'Winning Trades:         {len(wins)} ({len(wins)/len(r_values)*100:.1f}%)')
print(f'Losing Trades:          {len(losses)} ({len(losses)/len(r_values)*100:.1f}%)')
print()
print(f'Total R Result:         {sum(r_values):.2f}R')
print(f'Average per Trade:      {sum(r_values)/len(r_values):.3f}R')
print()
print(f'Sum of Winning Rs:      {sum(wins):.2f}R')
print(f'Sum of Losing Rs:       {sum(losses):.2f}R')
print()
print(f'Avg Win:                {sum(wins)/len(wins):.3f}R')
print(f'Avg Loss:               {sum(losses)/len(losses):.3f}R')
print()
print(f'Profit Factor:          {abs(sum(wins)/sum(losses)):.2f}x')
print()

# Expectancy
win_pct = len(wins) / len(r_values)
loss_pct = len(losses) / len(r_values)
avg_win = sum(wins) / len(wins)
avg_loss = abs(sum(losses) / len(losses))
expectancy = win_pct * avg_win - loss_pct * avg_loss

print(f'Expectancy:             {expectancy:.3f}R per trade')
print(f'Calculation:            {win_pct:.3f} × {avg_win:.3f} - {loss_pct:.3f} × {avg_loss:.3f}')
print()
print('='*80)
print('NOW COMPARABLE TO FORWARD TEST')
print('='*80)
print()
print('COMPARISON: Backtest vs Forward')
print('-'*80)
print()
print('Backtest (corrected R-normalized):')
print(f'  Trades:     {len(r_values)}')
print(f'  Win Rate:   {len(wins)/len(r_values)*100:.1f}%')
print(f'  Total R:    {sum(r_values):.2f}R')
print(f'  Expectancy: {expectancy:.3f}R/trade')
print()
print('Forward (from your comparison):')
print(f'  Trades:     202')
print(f'  Win Rate:   64.4%')
print(f'  Total R:    +47.8R')
print(f'  Expectancy: +0.237R/trade')
print()
print('='*80)
