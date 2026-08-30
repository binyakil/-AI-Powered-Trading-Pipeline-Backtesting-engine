# Forward Testing Analysis

Analysis tools for bot.log from forward testing runs.

## Files

- **analyze_log.py** - Main analysis script
- **bot.log** - Raw bot execution log (symlink or copy)
- **trades_analysis.json** - Parsed trade data (generated)
- **trades_export.csv** - CSV export for spreadsheet analysis (generated)

## Usage

```bash
python analyze_log.py
```

## Output

The script generates:

1. **Console Summary**
   - Trade statistics (win rate, P&L, profit factor)
   - Daily performance breakdown
   - Per-trade metrics

2. **trades_analysis.json**
   - Structured trade data
   - Statistics and daily performance
   - Complete trade metadata

3. **trades_export.csv**
   - Spreadsheet-ready format
   - One row per closed trade
   - Entry/exit times, prices, P&L

## Key Statistics

- **Win Rate** - % of profitable trades
- **Profit Factor** - Winning P&L / |Losing P&L| (>1.0 is profitable)
- **Avg Win/Loss** - Average profit and loss per trade
- **Duration** - Average time from entry to exit

## Log Structure

The bot.log contains events in this format:

```
2026-01-21 01:12:29,138 [INFO] TripleTP_Trail05R_40-30-30_MEXC: DRY-RUN entry T1 side=LONG entry=12.16000 stop=12.09386 size=302.390000
2026-01-21 01:39:11,342 [INFO] TripleTP_Trail05R_40-30-30_MEXC: TP1_FILLED group=T1 price=12.22614
2026-01-21 01:39:11,342 [INFO] TripleTP_Trail05R_40-30-30_MEXC: TRAIL_ARMED reason=TP1_FILLED group=T1
2026-01-21 03:06:09,442 [INFO] TripleTP_Trail05R_40-30-30_MEXC: Trade T1 closed PnL=38.0005
```

Events parsed:
- `DRY-RUN entry` - Trade entry with side, price, stop, size
- `TP*_FILLED` - Take profit level filled
- `TRAIL_ARMED` - Trailing stop activated (reason = why)
- `Trade * closed` - Final close with P&L
