"""
Portfolio Manager for Julaba Trading Bot
=========================================
Tracks overall performance, asset allocation, risk metrics, and equity curves.

Features:
- Daily/Weekly/Monthly P&L tracking
- Drawdown analysis (max, current)
- Win rate by pair, direction, time
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Equity curve with statistics
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import math

logger = logging.getLogger("Julaba.Portfolio")

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0  # minutes
    
@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    current_drawdown: float = 0.0
    current_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    recovery_factor: float = 0.0

@dataclass
class PeriodStats:
    """Stats for a specific time period"""
    period: str
    start_date: str
    end_date: str
    trades: int
    pnl: float
    win_rate: float
    best_trade: float
    worst_trade: float

class PortfolioManager:
    """
    Comprehensive portfolio tracking and analysis
    """
    
    def __init__(self, config_path: str = "julaba_config.json", 
                 history_path: str = "trade_history.json",
                 portfolio_path: str = "portfolio_data.json"):
        self.config_path = Path(config_path)
        self.history_path = Path(history_path)
        self.portfolio_path = Path(portfolio_path)
        
        # Load data
        self.config = self._load_json(self.config_path, {})
        self.trade_history = self._load_json(self.history_path, [])
        self.portfolio_data = self._load_json(self.portfolio_path, self._default_portfolio())
        
        # Cache for equity curve
        self._equity_cache = None
        self._equity_cache_time = None
        
        logger.info(f"📊 Portfolio Manager initialized with {len(self.trade_history)} trades")
    
    def _default_portfolio(self) -> dict:
        """Default portfolio structure"""
        return {
            "created_at": datetime.now().isoformat(),
            "initial_balance": 0,
            "equity_curve": [],
            "daily_pnl": {},
            "monthly_pnl": {},
            "pair_stats": {},
            "direction_stats": {"LONG": {}, "SHORT": {}},
            "hourly_stats": {},
            "snapshots": []
        }
    
    def _load_json(self, path: Path, default) -> any:
        """Load JSON file with fallback"""
        try:
            if path.exists():
                with open(path) as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
        return default
    
    def _save_portfolio(self):
        """Save portfolio data"""
        try:
            with open(self.portfolio_path, 'w') as f:
                json.dump(self.portfolio_data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save portfolio: {e}")
    
    def reload_trades(self):
        """Reload trade history from file"""
        self.trade_history = self._load_json(self.history_path, [])
        self._equity_cache = None  # Invalidate cache
    
    # ==================== PERFORMANCE METRICS ====================
    
    def get_performance_metrics(self, trades: List[dict] = None) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if trades is None:
            trades = self.trade_history
        
        if not trades:
            return PerformanceMetrics()
        
        metrics = PerformanceMetrics()
        metrics.total_trades = len(trades)
        
        wins = [t for t in trades if t.get('pnl', 0) > 0]
        losses = [t for t in trades if t.get('pnl', 0) < 0]
        
        metrics.winning_trades = len(wins)
        metrics.losing_trades = len(losses)
        metrics.win_rate = (len(wins) / len(trades) * 100) if trades else 0
        
        pnls = [t.get('pnl', 0) for t in trades]
        metrics.total_pnl = sum(pnls)
        
        win_pnls = [t.get('pnl', 0) for t in wins]
        loss_pnls = [abs(t.get('pnl', 0)) for t in losses]
        
        metrics.avg_win = sum(win_pnls) / len(wins) if wins else 0
        metrics.avg_loss = sum(loss_pnls) / len(losses) if losses else 0
        
        total_wins = sum(win_pnls)
        total_losses = sum(loss_pnls)
        metrics.profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')
        
        # Expectancy = (Win% × Avg Win) - (Loss% × Avg Loss)
        win_pct = len(wins) / len(trades) if trades else 0
        loss_pct = len(losses) / len(trades) if trades else 0
        metrics.expectancy = (win_pct * metrics.avg_win) - (loss_pct * metrics.avg_loss)
        
        metrics.largest_win = max(pnls) if pnls else 0
        metrics.largest_loss = min(pnls) if pnls else 0
        
        # Average trade duration
        durations = []
        for t in trades:
            if t.get('entry_time') and t.get('exit_time'):
                try:
                    entry = datetime.fromisoformat(t['entry_time'].replace('Z', '+00:00'))
                    exit = datetime.fromisoformat(t['exit_time'].replace('Z', '+00:00'))
                    durations.append((exit - entry).total_seconds() / 60)
                except Exception:
                    pass
        metrics.avg_trade_duration = sum(durations) / len(durations) if durations else 0
        
        return metrics
    
    # ==================== RISK METRICS ====================
    
    def get_risk_metrics(self, initial_balance: float = None) -> RiskMetrics:
        """Calculate risk-adjusted performance metrics"""
        if initial_balance is None:
            initial_balance = self.config.get('initial_balance', 
                                             self.portfolio_data.get('initial_balance', 350))
        
        if not self.trade_history:
            return RiskMetrics()
        
        metrics = RiskMetrics()
        
        # Build equity curve
        equity_curve = self._build_equity_curve(initial_balance)
        if len(equity_curve) < 2:
            return metrics
        
        # Drawdown calculations
        peak = initial_balance
        max_dd = 0
        max_dd_pct = 0
        
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            dd_pct = (dd / peak * 100) if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
        
        metrics.max_drawdown = max_dd
        metrics.max_drawdown_pct = max_dd_pct
        
        current_equity = equity_curve[-1] if equity_curve else initial_balance
        current_peak = max(equity_curve) if equity_curve else initial_balance
        metrics.current_drawdown = current_peak - current_equity
        metrics.current_drawdown_pct = (metrics.current_drawdown / current_peak * 100) if current_peak > 0 else 0
        
        # Daily returns for Sharpe/Sortino
        daily_returns = self._get_daily_returns(initial_balance)
        
        if daily_returns:
            avg_return = sum(daily_returns) / len(daily_returns)
            
            # Standard deviation
            variance = sum((r - avg_return) ** 2 for r in daily_returns) / len(daily_returns)
            std_dev = math.sqrt(variance) if variance > 0 else 0.001
            
            # Downside deviation (for Sortino)
            negative_returns = [r for r in daily_returns if r < 0]
            downside_variance = sum(r ** 2 for r in negative_returns) / len(daily_returns) if negative_returns else 0.001
            downside_dev = math.sqrt(downside_variance)
            
            # Annualized (assuming 365 trading days for crypto)
            risk_free_rate = 0.05 / 365  # ~5% annual
            
            # Sharpe Ratio = (Return - Risk-Free) / Std Dev
            metrics.sharpe_ratio = ((avg_return - risk_free_rate) / std_dev * math.sqrt(365)) if std_dev > 0 else 0
            
            # Sortino Ratio = (Return - Risk-Free) / Downside Dev
            metrics.sortino_ratio = ((avg_return - risk_free_rate) / downside_dev * math.sqrt(365)) if downside_dev > 0 else 0
            
            # Value at Risk (95%)
            sorted_returns = sorted(daily_returns)
            var_index = int(len(sorted_returns) * 0.05)
            metrics.var_95 = sorted_returns[var_index] if var_index < len(sorted_returns) else 0
        
        # Calmar Ratio = Annual Return / Max Drawdown
        total_pnl = sum(t.get('pnl', 0) for t in self.trade_history)
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = (total_pnl / initial_balance * 100) / metrics.max_drawdown_pct
        
        # Recovery Factor = Net Profit / Max Drawdown
        if metrics.max_drawdown > 0:
            metrics.recovery_factor = total_pnl / metrics.max_drawdown
        
        return metrics
    
    def _build_equity_curve(self, initial_balance: float) -> List[float]:
        """Build equity curve from trades"""
        if self._equity_cache is not None:
            return self._equity_cache
        
        equity = [initial_balance]
        current = initial_balance
        
        sorted_trades = sorted(self.trade_history, 
                              key=lambda t: t.get('exit_time', t.get('timestamp', '')))
        
        for trade in sorted_trades:
            pnl = trade.get('pnl', 0)
            current += pnl
            equity.append(current)
        
        self._equity_cache = equity
        return equity
    
    def _get_daily_returns(self, initial_balance: float) -> List[float]:
        """Get daily return percentages"""
        daily_pnl = defaultdict(float)
        
        for trade in self.trade_history:
            exit_time = trade.get('exit_time', trade.get('timestamp', ''))
            if exit_time:
                try:
                    date = exit_time[:10]  # YYYY-MM-DD
                    daily_pnl[date] += trade.get('pnl', 0)
                except Exception:
                    pass
        
        # Convert to returns
        balance = initial_balance
        returns = []
        for date in sorted(daily_pnl.keys()):
            pnl = daily_pnl[date]
            ret = pnl / balance if balance > 0 else 0
            returns.append(ret)
            balance += pnl
        
        return returns
    
    # ==================== BREAKDOWN ANALYSIS ====================
    
    def get_stats_by_pair(self) -> Dict[str, PerformanceMetrics]:
        """Get performance metrics broken down by trading pair"""
        pair_trades = defaultdict(list)
        
        for trade in self.trade_history:
            pair = trade.get('symbol', trade.get('pair', 'UNKNOWN'))
            # Normalize pair name
            pair = pair.replace('/USDT:USDT', '').replace(':USDT', '').replace('/USDT', '')
            pair_trades[pair].append(trade)
        
        return {pair: self.get_performance_metrics(trades) 
                for pair, trades in pair_trades.items()}
    
    def get_stats_by_direction(self) -> Dict[str, PerformanceMetrics]:
        """Get performance metrics broken down by trade direction"""
        direction_trades = {'LONG': [], 'SHORT': []}
        
        for trade in self.trade_history:
            direction = trade.get('direction', trade.get('side', 'UNKNOWN')).upper()
            if direction in direction_trades:
                direction_trades[direction].append(trade)
        
        return {direction: self.get_performance_metrics(trades) 
                for direction, trades in direction_trades.items()}
    
    def get_stats_by_hour(self) -> Dict[int, PerformanceMetrics]:
        """Get performance metrics broken down by hour of day (UTC)"""
        hourly_trades = defaultdict(list)
        
        for trade in self.trade_history:
            entry_time = trade.get('entry_time', trade.get('timestamp', ''))
            if entry_time:
                try:
                    hour = int(entry_time[11:13])  # Extract hour from ISO format
                    hourly_trades[hour].append(trade)
                except Exception:
                    pass
        
        return {hour: self.get_performance_metrics(trades) 
                for hour, trades in sorted(hourly_trades.items())}
    
    def get_stats_by_day_of_week(self) -> Dict[str, PerformanceMetrics]:
        """Get performance metrics broken down by day of week"""
        day_trades = defaultdict(list)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for trade in self.trade_history:
            entry_time = trade.get('entry_time', trade.get('timestamp', ''))
            if entry_time:
                try:
                    dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                    day = days[dt.weekday()]
                    day_trades[day].append(trade)
                except Exception:
                    pass
        
        return {day: self.get_performance_metrics(day_trades[day]) 
                for day in days if day in day_trades}
    
    # ==================== PERIOD ANALYSIS ====================
    
    def get_daily_pnl(self, days: int = 30) -> List[PeriodStats]:
        """Get P&L for each of the last N days"""
        daily_trades = defaultdict(list)
        
        cutoff = datetime.now() - timedelta(days=days)
        
        for trade in self.trade_history:
            exit_time = trade.get('exit_time', trade.get('timestamp', ''))
            if exit_time:
                try:
                    dt = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
                    if dt.replace(tzinfo=None) >= cutoff:
                        date = exit_time[:10]
                        daily_trades[date].append(trade)
                except Exception:
                    pass
        
        results = []
        for date in sorted(daily_trades.keys()):
            trades = daily_trades[date]
            pnls = [t.get('pnl', 0) for t in trades]
            wins = len([p for p in pnls if p > 0])
            
            results.append(PeriodStats(
                period="daily",
                start_date=date,
                end_date=date,
                trades=len(trades),
                pnl=sum(pnls),
                win_rate=(wins / len(trades) * 100) if trades else 0,
                best_trade=max(pnls) if pnls else 0,
                worst_trade=min(pnls) if pnls else 0
            ))
        
        return results
    
    def get_weekly_pnl(self, weeks: int = 12) -> List[PeriodStats]:
        """Get P&L for each of the last N weeks"""
        weekly_trades = defaultdict(list)
        
        cutoff = datetime.now() - timedelta(weeks=weeks)
        
        for trade in self.trade_history:
            exit_time = trade.get('exit_time', trade.get('timestamp', ''))
            if exit_time:
                try:
                    dt = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
                    if dt.replace(tzinfo=None) >= cutoff:
                        # ISO week format: YYYY-Www
                        week = dt.strftime('%Y-W%W')
                        weekly_trades[week].append(trade)
                except Exception:
                    pass
        
        results = []
        for week in sorted(weekly_trades.keys()):
            trades = weekly_trades[week]
            pnls = [t.get('pnl', 0) for t in trades]
            wins = len([p for p in pnls if p > 0])
            
            results.append(PeriodStats(
                period="weekly",
                start_date=week,
                end_date=week,
                trades=len(trades),
                pnl=sum(pnls),
                win_rate=(wins / len(trades) * 100) if trades else 0,
                best_trade=max(pnls) if pnls else 0,
                worst_trade=min(pnls) if pnls else 0
            ))
        
        return results
    
    def get_monthly_pnl(self, months: int = 12) -> List[PeriodStats]:
        """Get P&L for each of the last N months"""
        monthly_trades = defaultdict(list)
        
        cutoff = datetime.now() - timedelta(days=months * 30)
        
        for trade in self.trade_history:
            exit_time = trade.get('exit_time', trade.get('timestamp', ''))
            if exit_time:
                try:
                    dt = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
                    if dt.replace(tzinfo=None) >= cutoff:
                        month = exit_time[:7]  # YYYY-MM
                        monthly_trades[month].append(trade)
                except Exception:
                    pass
        
        results = []
        for month in sorted(monthly_trades.keys()):
            trades = monthly_trades[month]
            pnls = [t.get('pnl', 0) for t in trades]
            wins = len([p for p in pnls if p > 0])
            
            results.append(PeriodStats(
                period="monthly",
                start_date=month,
                end_date=month,
                trades=len(trades),
                pnl=sum(pnls),
                win_rate=(wins / len(trades) * 100) if trades else 0,
                best_trade=max(pnls) if pnls else 0,
                worst_trade=min(pnls) if pnls else 0
            ))
        
        return results
    
    # ==================== STREAKS & PATTERNS ====================
    
    def get_streaks(self) -> Dict[str, any]:
        """Analyze winning/losing streaks"""
        if not self.trade_history:
            return {}
        
        sorted_trades = sorted(self.trade_history,
                              key=lambda t: t.get('exit_time', t.get('timestamp', '')))
        
        current_streak = 0
        current_streak_type = None
        max_win_streak = 0
        max_loss_streak = 0
        streaks = []
        
        for trade in sorted_trades:
            pnl = trade.get('pnl', 0)
            is_win = pnl > 0
            
            if current_streak_type is None:
                current_streak_type = is_win
                current_streak = 1
            elif current_streak_type == is_win:
                current_streak += 1
            else:
                # Streak ended
                streaks.append(('WIN' if current_streak_type else 'LOSS', current_streak))
                if current_streak_type and current_streak > max_win_streak:
                    max_win_streak = current_streak
                elif not current_streak_type and current_streak > max_loss_streak:
                    max_loss_streak = current_streak
                current_streak_type = is_win
                current_streak = 1
        
        # Final streak
        if current_streak_type is not None:
            if current_streak_type and current_streak > max_win_streak:
                max_win_streak = current_streak
            elif not current_streak_type and current_streak > max_loss_streak:
                max_loss_streak = current_streak
        
        return {
            'current_streak': current_streak,
            'current_streak_type': 'WIN' if current_streak_type else 'LOSS',
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'all_streaks': streaks[-10:]  # Last 10 streaks
        }
    
    # ==================== SUMMARY REPORT ====================
    
    def get_full_report(self) -> Dict[str, any]:
        """Generate comprehensive portfolio report"""
        initial_balance = self.config.get('initial_balance', 350)
        
        perf = self.get_performance_metrics()
        risk = self.get_risk_metrics(initial_balance)
        streaks = self.get_streaks()
        
        direction_stats = self.get_stats_by_direction()
        pair_stats = self.get_stats_by_pair()
        
        # Top/worst pairs
        pair_pnl = {pair: stats.total_pnl for pair, stats in pair_stats.items()}
        sorted_pairs = sorted(pair_pnl.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'summary': {
                'total_trades': perf.total_trades,
                'win_rate': round(perf.win_rate, 1),
                'total_pnl': round(perf.total_pnl, 2),
                'profit_factor': round(perf.profit_factor, 2),
                'expectancy': round(perf.expectancy, 2),
                'avg_trade_duration_mins': round(perf.avg_trade_duration, 1),
            },
            'risk': {
                'max_drawdown': round(risk.max_drawdown, 2),
                'max_drawdown_pct': round(risk.max_drawdown_pct, 1),
                'current_drawdown': round(risk.current_drawdown, 2),
                'sharpe_ratio': round(risk.sharpe_ratio, 2),
                'sortino_ratio': round(risk.sortino_ratio, 2),
                'calmar_ratio': round(risk.calmar_ratio, 2),
                'var_95': round(risk.var_95 * 100, 2),
            },
            'direction': {
                'long': {
                    'trades': direction_stats.get('LONG', PerformanceMetrics()).total_trades,
                    'win_rate': round(direction_stats.get('LONG', PerformanceMetrics()).win_rate, 1),
                    'pnl': round(direction_stats.get('LONG', PerformanceMetrics()).total_pnl, 2),
                },
                'short': {
                    'trades': direction_stats.get('SHORT', PerformanceMetrics()).total_trades,
                    'win_rate': round(direction_stats.get('SHORT', PerformanceMetrics()).win_rate, 1),
                    'pnl': round(direction_stats.get('SHORT', PerformanceMetrics()).total_pnl, 2),
                }
            },
            'streaks': streaks,
            'top_pairs': sorted_pairs[:5],
            'worst_pairs': sorted_pairs[-5:][::-1] if len(sorted_pairs) > 5 else [],
            'equity_curve': self._build_equity_curve(initial_balance),
        }
    
    def print_report(self):
        """Print formatted portfolio report"""
        report = self.get_full_report()
        
        print("\n" + "="*60)
        print("📊 JULABA PORTFOLIO REPORT")
        print("="*60)
        
        s = report['summary']
        print(f"\n📈 PERFORMANCE SUMMARY")
        print(f"   Total Trades: {s['total_trades']}")
        print(f"   Win Rate: {s['win_rate']}%")
        print(f"   Total P&L: ${s['total_pnl']}")
        print(f"   Profit Factor: {s['profit_factor']}")
        print(f"   Expectancy: ${s['expectancy']}/trade")
        print(f"   Avg Duration: {s['avg_trade_duration_mins']} mins")
        
        r = report['risk']
        print(f"\n🛡️ RISK METRICS")
        print(f"   Max Drawdown: ${r['max_drawdown']} ({r['max_drawdown_pct']}%)")
        print(f"   Current Drawdown: ${r['current_drawdown']}")
        print(f"   Sharpe Ratio: {r['sharpe_ratio']}")
        print(f"   Sortino Ratio: {r['sortino_ratio']}")
        print(f"   Calmar Ratio: {r['calmar_ratio']}")
        print(f"   VaR 95%: {r['var_95']}%")
        
        d = report['direction']
        print(f"\n↕️ BY DIRECTION")
        print(f"   LONG:  {d['long']['trades']} trades | {d['long']['win_rate']}% WR | ${d['long']['pnl']}")
        print(f"   SHORT: {d['short']['trades']} trades | {d['short']['win_rate']}% WR | ${d['short']['pnl']}")
        
        st = report['streaks']
        print(f"\n🔥 STREAKS")
        print(f"   Current: {st.get('current_streak', 0)} {st.get('current_streak_type', 'N/A')}")
        print(f"   Max Win Streak: {st.get('max_win_streak', 0)}")
        print(f"   Max Loss Streak: {st.get('max_loss_streak', 0)}")
        
        print(f"\n🏆 TOP PAIRS")
        for pair, pnl in report['top_pairs'][:5]:
            print(f"   {pair}: ${pnl:.2f}")
        
        print("\n" + "="*60)


# CLI for testing
if __name__ == "__main__":
    pm = PortfolioManager()
    pm.print_report()
