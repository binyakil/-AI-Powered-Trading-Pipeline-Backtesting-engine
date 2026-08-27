#!/usr/bin/env python3
"""
Forward Testing Log Analyzer
Parses bot.log and generates profitability analysis and visualizations
"""

import re
import json
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class TradeEvent:
    """Single trade event (entry, fill, close, etc)"""
    timestamp: str
    event_type: str  # 'ENTRY', 'TP_FILLED', 'TRAIL_ARMED', 'CLOSED'
    trade_id: str
    details: Dict


@dataclass
class Trade:
    """Complete trade lifecycle"""
    trade_id: str
    entry_time: str
    entry_price: float
    side: str  # LONG or SHORT
    stop_price: float
    size: float
    exit_price: Optional[float] = None
    close_time: Optional[str] = None
    pnl: Optional[float] = None
    tp_fills: List[Dict] = field(default_factory=list)
    trail_reason: Optional[str] = None
    duration_minutes: Optional[int] = None

    def to_dict(self):
        return asdict(self)


class BotLogAnalyzer:
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.trades: Dict[str, Trade] = {}
        self.events: List[TradeEvent] = []
        self.daily_pnl: defaultdict = defaultdict(lambda: {"pnl": 0.0, "trades": []})
        
        # Regex patterns
        self.entry_regex = re.compile(
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*?: DRY-RUN entry (T\d+) side=(LONG|SHORT) entry=([\d.]+) stop=([\d.]+) size=([\d.]+)'
        )
        self.tp_filled_regex = re.compile(
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*?: (TP\d)_FILLED group=(T\d+) price=([\d.]+)'
        )
        self.trail_armed_regex = re.compile(
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*?: TRAIL_ARMED reason=(\w+) group=(T\d+)'
        )
        self.closed_regex = re.compile(
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+.*?: Trade (T\d+) closed\s+PnL=([-\d.]+)'
        )

    def parse_log(self, verbose: bool = False) -> int:
        """Parse log file and extract trade events. Returns number of lines processed."""
        if not self.log_file.exists():
            raise FileNotFoundError(f"Log file not found: {self.log_file}")
        
        lines_processed = 0
        print(f"📖 Parsing log file: {self.log_file}")
        print(f"   File size: {self.log_file.stat().st_size / (1024*1024):.1f} MB\n")
        
        try:
            with open(self.log_file, 'r', errors='ignore') as f:
                for line in f:
                    lines_processed += 1
                    if lines_processed % 500000 == 0:
                        print(f"   Processed {lines_processed:,} lines...")
                    
                    # Quick keyword filter before regex (3.7M lines, so this matters)
                    if 'DRY-RUN entry' in line:
                        # Parse entries
                        match = self.entry_regex.search(line)
                        if match:
                            timestamp, trade_id, side, entry, stop, size = match.groups()
                            trade = Trade(
                                trade_id=trade_id,
                                entry_time=timestamp,
                                entry_price=float(entry),
                                side=side,
                                stop_price=float(stop),
                                size=float(size)
                            )
                            self.trades[trade_id] = trade
                            if verbose:
                                print(f"✓ ENTRY {trade_id}: {side} @ {entry}")
                    
                    elif '_FILLED' in line:
                        # Parse TP fills
                        match = self.tp_filled_regex.search(line)
                        if match:
                            timestamp, tp_level, trade_id, price = match.groups()
                            if trade_id in self.trades:
                                self.trades[trade_id].tp_fills.append({
                                    "level": tp_level,
                                    "price": float(price),
                                    "time": timestamp
                                })
                                if verbose:
                                    print(f"✓ {tp_level}_FILLED {trade_id} @ {price}")
                    
                    elif 'TRAIL_ARMED' in line:
                        # Parse trail armed
                        match = self.trail_armed_regex.search(line)
                        if match:
                            timestamp, reason, trade_id = match.groups()
                            if trade_id in self.trades:
                                self.trades[trade_id].trail_reason = reason
                                if verbose:
                                    print(f"✓ TRAIL_ARMED {trade_id}: {reason}")
                    
                    elif 'closed' in line and 'PnL=' in line:
                        # Parse closes
                        match = self.closed_regex.search(line)
                        if match:
                            timestamp, trade_id, pnl = match.groups()
                            if trade_id in self.trades:
                                self.trades[trade_id].close_time = timestamp
                                self.trades[trade_id].pnl = float(pnl)
                                
                                # Calculate duration
                                entry_dt = datetime.strptime(self.trades[trade_id].entry_time, "%Y-%m-%d %H:%M:%S")
                                close_dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                                duration = (close_dt - entry_dt).total_seconds() // 60
                                self.trades[trade_id].duration_minutes = int(duration)
                                
                                # Track daily P&L
                                date = timestamp.split()[0]
                                self.daily_pnl[date]["pnl"] += float(pnl)
                                self.daily_pnl[date]["trades"].append(trade_id)
                                
                                if verbose:
                                    print(f"✓ CLOSED {trade_id}: PnL={pnl}")
        except Exception as e:
            print(f"❌ Error parsing log: {e}")
            raise
        
        print(f"✓ Processed {lines_processed:,} lines\n")
        return lines_processed

    def calculate_statistics(self) -> Dict:
        """Calculate comprehensive trade statistics"""
        closed_trades = [t for t in self.trades.values() if t.pnl is not None]
        
        if not closed_trades:
            return {"error": "No closed trades found"}
        
        pnls = [t.pnl for t in closed_trades]
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]
        breakeven_trades = [t for t in closed_trades if t.pnl == 0]
        
        total_pnl = sum(pnls)
        winning_pnl = sum(t.pnl for t in winning_trades)
        losing_pnl = sum(t.pnl for t in losing_trades)
        
        stats = {
            "total_trades": len(closed_trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "breakeven_trades": len(breakeven_trades),
            "win_rate_pct": (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0,
            "total_pnl": round(total_pnl, 4),
            "winning_pnl": round(winning_pnl, 4),
            "losing_pnl": round(losing_pnl, 4),
            "avg_win": round(winning_pnl / len(winning_trades), 4) if winning_trades else 0,
            "avg_loss": round(losing_pnl / len(losing_trades), 4) if losing_trades else 0,
            "avg_trade": round(total_pnl / len(closed_trades), 4),
            "largest_win": round(max(pnls), 4),
            "largest_loss": round(min(pnls), 4),
            "profit_factor": round(winning_pnl / abs(losing_pnl), 2) if losing_pnl != 0 else 0,
            "avg_duration_minutes": round(sum(t.duration_minutes for t in closed_trades if t.duration_minutes) / len(closed_trades), 1) if closed_trades else 0,
        }
        
        return stats

    def daily_analysis(self) -> Dict:
        """Analyze daily performance"""
        daily_stats = {}
        
        for date in sorted(self.daily_pnl.keys()):
            trades_on_day = self.daily_pnl[date]["trades"]
            trade_objs = [self.trades[tid] for tid in trades_on_day]
            pnl_on_day = self.daily_pnl[date]["pnl"]
            
            daily_stats[date] = {
                "pnl": round(pnl_on_day, 4),
                "num_trades": len(trades_on_day),
                "num_wins": len([t for t in trade_objs if t.pnl > 0]),
                "num_losses": len([t for t in trade_objs if t.pnl < 0]),
                "trade_ids": trades_on_day,
            }
        
        return daily_stats

    def print_summary(self):
        """Print formatted summary to console"""
        stats = self.calculate_statistics()
        
        if "error" in stats:
            print(f"⚠️  {stats['error']}")
            return
        
        print("=" * 80)
        print("📊 TRADE STATISTICS SUMMARY")
        print("=" * 80)
        
        print(f"\n🎯 Performance Overview:")
        print(f"   Total Trades:        {stats['total_trades']}")
        print(f"   Winning:             {stats['winning_trades']} ({stats['win_rate_pct']:.1f}%)")
        print(f"   Losing:              {stats['losing_trades']}")
        print(f"   Breakeven:           {stats['breakeven_trades']}")
        
        print(f"\n💰 Profitability:")
        print(f"   Total P&L:           ${stats['total_pnl']:.4f}")
        print(f"   Winning P&L:         ${stats['winning_pnl']:.4f}")
        print(f"   Losing P&L:          ${stats['losing_pnl']:.4f}")
        print(f"   Profit Factor:       {stats['profit_factor']:.2f}x")
        
        print(f"\n📈 Per-Trade Metrics:")
        print(f"   Avg Win:             ${stats['avg_win']:.4f}")
        print(f"   Avg Loss:            ${stats['avg_loss']:.4f}")
        print(f"   Avg Trade:           ${stats['avg_trade']:.4f}")
        print(f"   Largest Win:         ${stats['largest_win']:.4f}")
        print(f"   Largest Loss:        ${stats['largest_loss']:.4f}")
        
        print(f"\n⏱️  Duration:")
        print(f"   Avg Duration:        {stats['avg_duration_minutes']:.1f} minutes")
        
        print("\n" + "=" * 80)
        print("📅 DAILY PERFORMANCE")
        print("=" * 80)
        
        daily_stats = self.daily_analysis()
        cumulative_pnl = 0
        
        for date in sorted(daily_stats.keys()):
            day_data = daily_stats[date]
            cumulative_pnl += day_data["pnl"]
            
            pnl_str = f"${day_data['pnl']:+.4f}".ljust(12)
            trades_str = f"{day_data['num_trades']}T ({day_data['num_wins']}W/{day_data['num_losses']}L)"
            cum_str = f"Cumul: ${cumulative_pnl:+.4f}"
            
            print(f"{date}  {pnl_str}  {trades_str:20s}  {cum_str}")
        
        print("\n" + "=" * 80)

    def export_trades_json(self, output_file: Optional[str] = None) -> str:
        """Export all trades as JSON"""
        if output_file is None:
            output_file = self.log_file.parent / "trades_analysis.json"
        
        trades_data = {
            "metadata": {
                "total_trades": len(self.trades),
                "closed_trades": len([t for t in self.trades.values() if t.pnl is not None]),
                "log_file": str(self.log_file),
                "generated": datetime.now().isoformat(),
            },
            "statistics": self.calculate_statistics(),
            "daily_performance": self.daily_analysis(),
            "trades": {tid: t.to_dict() for tid, t in self.trades.items()},
        }
        
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(trades_data, f, indent=2)
        
        print(f"\n✅ Exported trade data to: {output_path}")
        return str(output_path)

    def export_csv(self, output_file: Optional[str] = None) -> str:
        """Export closed trades as CSV for spreadsheet analysis"""
        if output_file is None:
            output_file = self.log_file.parent / "trades_export.csv"
        
        import csv
        
        closed_trades = sorted(
            [t for t in self.trades.values() if t.pnl is not None],
            key=lambda x: x.entry_time
        )
        
        output_path = Path(output_file)
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'trade_id', 'entry_time', 'close_time', 'side', 
                'entry_price', 'stop_price', 'size', 
                'pnl', 'duration_minutes', 'trail_reason'
            ])
            writer.writeheader()
            for trade in closed_trades:
                writer.writerow({
                    'trade_id': trade.trade_id,
                    'entry_time': trade.entry_time,
                    'close_time': trade.close_time,
                    'side': trade.side,
                    'entry_price': f"{trade.entry_price:.5f}",
                    'stop_price': f"{trade.stop_price:.5f}",
                    'size': f"{trade.size:.2f}",
                    'pnl': f"{trade.pnl:.4f}",
                    'duration_minutes': trade.duration_minutes,
                    'trail_reason': trade.trail_reason or 'N/A',
                })
        
        print(f"✅ Exported CSV to: {output_path}")
        return str(output_path)


def main():
    log_file = Path(__file__).parent.parent / "forward equal to backward" / "bot.log"
    
    print("\n🤖 Bot Log Analysis Tool")
    print("=" * 80)
    
    analyzer = BotLogAnalyzer(str(log_file))
    analyzer.parse_log(verbose=False)
    analyzer.print_summary()
    analyzer.export_trades_json()
    analyzer.export_csv()
    
    print("\n✨ Analysis complete!\n")


if __name__ == "__main__":
    main()
