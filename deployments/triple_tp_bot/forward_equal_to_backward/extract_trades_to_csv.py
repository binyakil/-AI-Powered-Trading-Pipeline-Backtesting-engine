#!/usr/bin/env python3
"""
Extract trade data from bot.log and export to CSV with detailed columns.
Columns: trade_id, run_id, side, entry_time, entry_px, stop_px_initial, stop_px_final, 
          tp1_px, tp2_px, tp3_px, exit_time, exit_px, exit_reason, R_result
"""

import re
import csv
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

LOG_PATH = Path("bot.log")
OUTPUT_PATH = Path("trades_export_detailed.csv")

# Regex patterns
ENTRY_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,\.\d]+ \[INFO\].*DRY-RUN entry (T\d+) side=(LONG|SHORT) entry=([\d.]+) stop=([\d.]+) size=([\d.]+)"
)

TP_FILLED_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,\.\d]+ \[INFO\].*?(TP\d)_FILLED group=(T\d+) price=([\d.]+)"
)

TRAIL_ARMED_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,\.\d]+ \[INFO\].*?TRAIL_ARMED reason=(\S+) group=(T\d+)"
)

CLOSE_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,\.\d]+ \[INFO\].*?Trade (T\d+) closed PnL=([-\d.]+)"
)

@dataclass
class Trade:
    trade_id: str
    side: str
    entry_time: str
    entry_px: float
    stop_px_initial: float
    stop_px_final: Optional[float] = None
    size: float = 0.0
    tp_fills: Dict[str, Tuple[str, float]] = field(default_factory=dict)  # {tp_level: (time, price)}
    exit_time: Optional[str] = None
    exit_px: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    tp_prices: Dict[str, float] = field(default_factory=dict)  # {tp1, tp2, tp3: price}

def extract_tp_prices_from_orders(log_path: Path, trade_id: str) -> Dict[str, float]:
    """Extract TP prices from log (would be in the entry/order creation logs)"""
    tp_prices = {}
    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            if trade_id not in line:
                continue
            # Look for TP price info (may appear in order creation)
            if 'TP' in line and 'price' in line.lower():
                # Try to extract TP levels and prices
                tp1_match = re.search(r'TP1.*price[=:]?\s*([\d.]+)', line, re.IGNORECASE)
                tp2_match = re.search(r'TP2.*price[=:]?\s*([\d.]+)', line, re.IGNORECASE)
                tp3_match = re.search(r'TP3.*price[=:]?\s*([\d.]+)', line, re.IGNORECASE)
                
                if tp1_match:
                    tp_prices['tp1'] = float(tp1_match.group(1))
                if tp2_match:
                    tp_prices['tp2'] = float(tp2_match.group(1))
                if tp3_match:
                    tp_prices['tp3'] = float(tp3_match.group(1))
                
                if tp_prices:
                    break
    return tp_prices

def calculate_tp_prices(entry_px: float, side: str, risk_pts: float, tp_levels: Tuple[float, ...] = (1.0, 2.0, 3.0)) -> Dict[str, float]:
    """Calculate TP prices from entry and risk"""
    tp_prices = {}
    for i, level in enumerate(tp_levels, 1):
        if side == "LONG":
            tp_px = entry_px + risk_pts * level
        else:  # SHORT
            tp_px = entry_px - risk_pts * level
        tp_prices[f'tp{i}'] = tp_px
    return tp_prices

def calculate_r_result(pnl: float, position_size: float, entry_px: float, stop_px: float) -> float:
    """Calculate R-result: How many Rs (risk units) did we win/lose?
    
    R = |entry_px - stop_px| (risk in price points per unit)
    Risk_USD = position_size × R (total risk in dollars)
    R_result = PnL_USD / Risk_USD (normalized to risk units)
    """
    if entry_px == stop_px or position_size == 0:
        return 0.0
    
    risk_pts = abs(entry_px - stop_px)  # Price point distance
    risk_usd = position_size * risk_pts  # Total risk in dollars
    
    if risk_usd == 0:
        return 0.0
    
    return pnl / risk_usd

def parse_log(log_path: Path) -> Dict[str, Trade]:
    """Parse bot.log and extract all trades"""
    trades = {}
    
    print(f"📖 Parsing bot.log ({log_path.stat().st_size / 1e6:.1f} MB)...")
    
    lines_processed = 0
    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            lines_processed += 1
            if lines_processed % 500000 == 0:
                print(f"   Processed {lines_processed:,} lines...")
            
            # Parse entries
            if 'DRY-RUN entry' in line:
                match = ENTRY_PATTERN.search(line)
                if match:
                    entry_time, trade_id, side, entry_px, stop_px, size = match.groups()
                    trades[trade_id] = Trade(
                        trade_id=trade_id,
                        side=side,
                        entry_time=entry_time,
                        entry_px=float(entry_px),
                        stop_px_initial=float(stop_px),
                        stop_px_final=float(stop_px),
                        size=float(size)
                    )
            
            # Parse TP fills
            elif '_FILLED' in line:
                match = TP_FILLED_PATTERN.search(line)
                if match:
                    fill_time, tp_level, trade_id, price = match.groups()
                    if trade_id in trades:
                        trades[trade_id].tp_fills[tp_level] = (fill_time, float(price))
            
            # Parse trail armed (exit reason + potential trail stop update)
            elif 'TRAIL_ARMED' in line:
                match = TRAIL_ARMED_PATTERN.search(line)
                if match:
                    time, reason, trade_id = match.groups()
                    if trade_id in trades:
                        trades[trade_id].exit_reason = reason
            
            # Parse closes
            elif 'Trade' in line and 'closed' in line and 'PnL=' in line:
                match = CLOSE_PATTERN.search(line)
                if match:
                    close_time, trade_id, pnl = match.groups()
                    if trade_id in trades:
                        trades[trade_id].exit_time = close_time
                        trades[trade_id].pnl = float(pnl)
    
    print(f"✓ Processed {lines_processed:,} lines")
    print(f"✓ Found {len(trades)} trades\n")
    return trades

def infer_exit_px(trade: Trade) -> Optional[float]:
    """Infer exit price from TP fills or PnL calculation"""
    # If we have TP fills, the last one could indicate exit
    if trade.tp_fills:
        # Return the highest priced TP fill (likely the last exit point)
        last_tp = sorted(trade.tp_fills.items())[-1]
        return last_tp[1][1]
    return None

def write_csv(trades: Dict[str, Trade], output_path: Path) -> None:
    """Write trades to CSV file"""
    fieldnames = [
        'trade_id', 'run_id', 'side', 'entry_time', 'entry_px',
        'stop_px_initial', 'stop_px_final', 'tp1_px', 'tp2_px', 'tp3_px',
        'exit_time', 'exit_px', 'exit_reason', 'R_result'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for trade_id in sorted(trades.keys(), key=lambda x: int(x[1:])):
            trade = trades[trade_id]
            
            # Only export closed trades
            if trade.pnl is None:
                continue
            
            # Calculate TP prices from entry and risk
            risk_pts = abs(trade.entry_px - trade.stop_px_initial)
            tp_prices = calculate_tp_prices(trade.entry_px, trade.side, risk_pts)
            
            # Calculate R result (properly normalized by position size and risk in dollars)
            r_result = calculate_r_result(trade.pnl, trade.size, trade.entry_px, trade.stop_px_final or trade.stop_px_initial)
            
            # Infer exit price (from last TP fill or calculated)
            exit_px = infer_exit_px(trade)
            
            row = {
                'trade_id': trade.trade_id,
                'run_id': 'run_2026-01-20_21-32-34',  # From the log directory structure
                'side': trade.side,
                'entry_time': trade.entry_time,
                'entry_px': f"{trade.entry_px:.5f}",
                'stop_px_initial': f"{trade.stop_px_initial:.5f}",
                'stop_px_final': f"{trade.stop_px_final:.5f}" if trade.stop_px_final else '',
                'tp1_px': f"{tp_prices.get('tp1', 0):.5f}",
                'tp2_px': f"{tp_prices.get('tp2', 0):.5f}",
                'tp3_px': f"{tp_prices.get('tp3', 0):.5f}",
                'exit_time': trade.exit_time or '',
                'exit_px': f"{exit_px:.5f}" if exit_px else '',
                'exit_reason': trade.exit_reason or '',
                'R_result': f"{r_result:.2f}"
            }
            writer.writerow(row)
    
    print(f"✅ Exported {len([t for t in trades.values() if t.pnl is not None])} trades to {output_path}")

def main():
    print("=" * 80)
    print("TRADE EXTRACTION: bot.log → CSV")
    print("=" * 80 + "\n")
    
    # Parse log
    trades = parse_log(LOG_PATH)
    
    # Write CSV
    write_csv(trades, OUTPUT_PATH)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    closed_trades = [t for t in trades.values() if t.pnl is not None]
    print(f"\nTotal Trades Closed: {len(closed_trades)}")
    print(f"Output File: {OUTPUT_PATH}")
    print(f"File Size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    main()
