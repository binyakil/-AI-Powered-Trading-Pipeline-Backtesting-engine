"""
Backtesting Engine for Julaba
Historical simulation with slippage modeling and performance analysis.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger("Julaba.Backtest")


@dataclass
class BacktestTrade:
    """Single trade in backtest."""
    entry_time: datetime
    exit_time: datetime
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # 'tp1', 'tp2', 'tp3', 'sl', 'signal_reversal'
    fees: float = 0.0
    slippage: float = 0.0


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    total_pnl_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    win_rate: float
    avg_trade_pnl: float
    avg_winner: float
    avg_loser: float
    largest_winner: float
    largest_loser: float
    avg_trade_duration: timedelta
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)


class BacktestEngine:
    """
    Backtesting Engine for Julaba Trading Strategy.
    
    Features:
    - Historical OHLCV data simulation
    - Realistic slippage modeling
    - Trading fee calculation
    - Multiple exit strategies (TP1/TP2/TP3/SL)
    - Performance metrics calculation
    - Monte Carlo analysis
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        fee_pct: float = 0.001,  # 0.1% per trade
        slippage_pct: float = 0.0005,  # 0.05% average slippage
        risk_pct: float = 0.02,
        atr_mult: float = 2.0,
        tp1_r: float = 1.0,
        tp2_r: float = 2.0,
        tp3_r: float = 3.0,
        tp1_pct: float = 0.4,
        tp2_pct: float = 0.3,
        tp3_pct: float = 0.3
    ):
        self.initial_balance = initial_balance
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.risk_pct = risk_pct
        self.atr_mult = atr_mult
        self.tp1_r = tp1_r
        self.tp2_r = tp2_r
        self.tp3_r = tp3_r
        self.tp1_pct = tp1_pct
        self.tp2_pct = tp2_pct
        self.tp3_pct = tp3_pct
        
        # State during backtest
        self.balance = initial_balance
        self.peak_balance = initial_balance
        self.position = None
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []
        self.drawdown_curve: List[float] = []
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR for the dataframe."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        return atr
    
    def _generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals using SMA crossover."""
        sma15 = df['close'].rolling(15).mean()
        sma40 = df['close'].rolling(40).mean()
        
        signals = pd.Series(0, index=df.index)
        
        # Long: SMA15 crosses above SMA40
        long_cross = (sma15 > sma40) & (sma15.shift(1) <= sma40.shift(1))
        signals[long_cross] = 1
        
        # Short: SMA15 crosses below SMA40
        short_cross = (sma15 < sma40) & (sma15.shift(1) >= sma40.shift(1))
        signals[short_cross] = -1
        
        return signals
    
    def _apply_slippage(self, price: float, side: str, is_entry: bool) -> float:
        """Apply realistic slippage to a price."""
        # Random slippage between 0 and 2x average
        slippage = np.random.uniform(0, self.slippage_pct * 2)
        
        if is_entry:
            # Entry: pay more for longs, less for shorts (worse price)
            if side == 'long':
                return price * (1 + slippage)
            else:
                return price * (1 - slippage)
        else:
            # Exit: get less for longs, more for shorts (worse price)
            if side == 'long':
                return price * (1 - slippage)
            else:
                return price * (1 + slippage)
    
    def _calculate_fees(self, notional: float) -> float:
        """Calculate trading fees."""
        return notional * self.fee_pct
    
    def run(
        self,
        df: pd.DataFrame,
        use_ai_filter: bool = False,
        ai_approval_rate: float = 0.7,
        verbose: bool = False
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            df: DataFrame with OHLCV data (must have columns: open, high, low, close, volume, timestamp)
            use_ai_filter: Simulate AI filter approval
            ai_approval_rate: Probability AI approves a signal (for simulation)
            verbose: Print trade details
        
        Returns:
            BacktestResult with all metrics
        """
        # Reset state
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.position = None
        self.trades = []
        self.equity_curve = [self.initial_balance]
        self.drawdown_curve = [0.0]
        
        # Ensure required columns
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Calculate indicators
        atr = self._calculate_atr(df)
        signals = self._generate_signals(df)
        
        # Skip warmup period
        start_idx = 50
        
        for i in range(start_idx, len(df)):
            current_bar = df.iloc[i]
            current_price = current_bar['close']
            current_high = current_bar['high']
            current_low = current_bar['low']
            current_atr = atr.iloc[i]
            current_signal = signals.iloc[i]
            current_time = df.index[i] if isinstance(df.index[i], datetime) else datetime.now()
            
            # Check existing position
            if self.position is not None:
                side = self.position['side']
                entry_price = self.position['entry_price']
                sl = self.position['sl']
                tp1 = self.position['tp1']
                tp2 = self.position['tp2']
                tp3 = self.position['tp3']
                size = self.position['size']
                remaining_size = self.position['remaining_size']
                
                exit_price = None
                exit_reason = None
                pnl = 0.0
                
                # Check stop loss
                if side == 'long' and current_low <= sl:
                    exit_price = self._apply_slippage(sl, side, is_entry=False)
                    exit_reason = 'sl'
                elif side == 'short' and current_high >= sl:
                    exit_price = self._apply_slippage(sl, side, is_entry=False)
                    exit_reason = 'sl'
                
                # Check take profits
                if exit_price is None:
                    if side == 'long':
                        if not self.position.get('tp1_hit') and current_high >= tp1:
                            # TP1 hit - close 40%
                            partial_size = size * self.tp1_pct
                            exit_p = self._apply_slippage(tp1, side, is_entry=False)
                            partial_pnl = (exit_p - entry_price) * partial_size
                            fees = self._calculate_fees(partial_size * exit_p)
                            partial_pnl -= fees
                            self.balance += partial_pnl
                            self.position['tp1_hit'] = True
                            self.position['remaining_size'] -= partial_size
                            self.position['realized_pnl'] = self.position.get('realized_pnl', 0) + partial_pnl
                        
                        if not self.position.get('tp2_hit') and current_high >= tp2:
                            # TP2 hit - close 30%
                            partial_size = size * self.tp2_pct
                            exit_p = self._apply_slippage(tp2, side, is_entry=False)
                            partial_pnl = (exit_p - entry_price) * partial_size
                            fees = self._calculate_fees(partial_size * exit_p)
                            partial_pnl -= fees
                            self.balance += partial_pnl
                            self.position['tp2_hit'] = True
                            self.position['remaining_size'] -= partial_size
                            self.position['realized_pnl'] = self.position.get('realized_pnl', 0) + partial_pnl
                        
                        if current_high >= tp3:
                            # TP3 hit - close remaining
                            exit_price = self._apply_slippage(tp3, side, is_entry=False)
                            exit_reason = 'tp3'
                    
                    else:  # short
                        if not self.position.get('tp1_hit') and current_low <= tp1:
                            partial_size = size * self.tp1_pct
                            exit_p = self._apply_slippage(tp1, side, is_entry=False)
                            partial_pnl = (entry_price - exit_p) * partial_size
                            fees = self._calculate_fees(partial_size * exit_p)
                            partial_pnl -= fees
                            self.balance += partial_pnl
                            self.position['tp1_hit'] = True
                            self.position['remaining_size'] -= partial_size
                            self.position['realized_pnl'] = self.position.get('realized_pnl', 0) + partial_pnl
                        
                        if not self.position.get('tp2_hit') and current_low <= tp2:
                            partial_size = size * self.tp2_pct
                            exit_p = self._apply_slippage(tp2, side, is_entry=False)
                            partial_pnl = (entry_price - exit_p) * partial_size
                            fees = self._calculate_fees(partial_size * exit_p)
                            partial_pnl -= fees
                            self.balance += partial_pnl
                            self.position['tp2_hit'] = True
                            self.position['remaining_size'] -= partial_size
                            self.position['realized_pnl'] = self.position.get('realized_pnl', 0) + partial_pnl
                        
                        if current_low <= tp3:
                            exit_price = self._apply_slippage(tp3, side, is_entry=False)
                            exit_reason = 'tp3'
                
                # Close position if exit triggered
                if exit_price is not None:
                    remaining = self.position['remaining_size']
                    if side == 'long':
                        final_pnl = (exit_price - entry_price) * remaining
                    else:
                        final_pnl = (entry_price - exit_price) * remaining
                    
                    fees = self._calculate_fees(remaining * exit_price)
                    final_pnl -= fees
                    
                    total_pnl = self.position.get('realized_pnl', 0) + final_pnl
                    self.balance += final_pnl
                    
                    trade = BacktestTrade(
                        entry_time=self.position['entry_time'],
                        exit_time=current_time,
                        side=side,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        size=size,
                        pnl=total_pnl,
                        pnl_pct=total_pnl / (entry_price * size) if entry_price * size > 0 else 0,
                        exit_reason=exit_reason,
                        fees=fees
                    )
                    self.trades.append(trade)
                    
                    if verbose:
                        print(f"{current_time}: CLOSE {side.upper()} @ {exit_price:.4f} | "
                              f"PnL: ${total_pnl:.2f} | Reason: {exit_reason}")
                    
                    self.position = None
            
            # Check for new signals
            if self.position is None and current_signal != 0:
                # Simulate AI filter
                if use_ai_filter and np.random.random() > ai_approval_rate:
                    continue  # AI rejected
                
                side = 'long' if current_signal == 1 else 'short'
                entry_price = self._apply_slippage(current_price, side, is_entry=True)
                
                # Calculate position size
                if side == 'long':
                    sl = entry_price - (current_atr * self.atr_mult)
                else:
                    sl = entry_price + (current_atr * self.atr_mult)
                
                risk_per_unit = abs(entry_price - sl)
                if risk_per_unit == 0:
                    continue
                
                risk_amount = self.balance * self.risk_pct
                size = risk_amount / risk_per_unit
                
                # Calculate take profits
                r_value = risk_per_unit
                if side == 'long':
                    tp1 = entry_price + (r_value * self.tp1_r)
                    tp2 = entry_price + (r_value * self.tp2_r)
                    tp3 = entry_price + (r_value * self.tp3_r)
                else:
                    tp1 = entry_price - (r_value * self.tp1_r)
                    tp2 = entry_price - (r_value * self.tp2_r)
                    tp3 = entry_price - (r_value * self.tp3_r)
                
                # Entry fees
                entry_fees = self._calculate_fees(size * entry_price)
                self.balance -= entry_fees
                
                self.position = {
                    'side': side,
                    'entry_price': entry_price,
                    'entry_time': current_time,
                    'size': size,
                    'remaining_size': size,
                    'sl': sl,
                    'tp1': tp1,
                    'tp2': tp2,
                    'tp3': tp3,
                    'tp1_hit': False,
                    'tp2_hit': False,
                    'realized_pnl': -entry_fees
                }
                
                if verbose:
                    print(f"{current_time}: OPEN {side.upper()} @ {entry_price:.4f} | "
                          f"Size: {size:.4f} | SL: {sl:.4f} | TP: {tp1:.4f}/{tp2:.4f}/{tp3:.4f}")
            
            # Update equity curve
            self.equity_curve.append(self.balance)
            self.peak_balance = max(self.peak_balance, self.balance)
            drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
            self.drawdown_curve.append(drawdown * 100)
        
        # Calculate results
        return self._calculate_results(df)
    
    def _calculate_results(self, df: pd.DataFrame) -> BacktestResult:
        """Calculate backtest performance metrics."""
        total_trades = len(self.trades)
        
        if total_trades == 0:
            return BacktestResult(
                start_date=df.index[0] if isinstance(df.index[0], datetime) else datetime.now(),
                end_date=df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now(),
                initial_balance=self.initial_balance,
                final_balance=self.balance,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_pnl=0,
                total_pnl_pct=0,
                max_drawdown=0,
                max_drawdown_pct=0,
                sharpe_ratio=0,
                profit_factor=0,
                win_rate=0,
                avg_trade_pnl=0,
                avg_winner=0,
                avg_loser=0,
                largest_winner=0,
                largest_loser=0,
                avg_trade_duration=timedelta(0),
                trades=self.trades,
                equity_curve=self.equity_curve,
                drawdown_curve=self.drawdown_curve
            )
        
        winners = [t for t in self.trades if t.pnl > 0]
        losers = [t for t in self.trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_pnl_pct = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        win_rate = len(winners) / total_trades if total_trades > 0 else 0
        
        avg_winner = np.mean([t.pnl for t in winners]) if winners else 0
        avg_loser = np.mean([t.pnl for t in losers]) if losers else 0
        
        gross_profit = sum(t.pnl for t in winners)
        gross_loss = abs(sum(t.pnl for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        max_dd = max(self.drawdown_curve) if self.drawdown_curve else 0
        
        # Sharpe ratio (simplified)
        returns = pd.Series([t.pnl_pct for t in self.trades])
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        else:
            sharpe = 0
        
        # Average trade duration
        durations = [(t.exit_time - t.entry_time) for t in self.trades if isinstance(t.entry_time, datetime)]
        avg_duration = np.mean(durations) if durations else timedelta(0)
        
        return BacktestResult(
            start_date=df.index[0] if isinstance(df.index[0], datetime) else datetime.now(),
            end_date=df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now(),
            initial_balance=self.initial_balance,
            final_balance=self.balance,
            total_trades=total_trades,
            winning_trades=len(winners),
            losing_trades=len(losers),
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            max_drawdown=max_dd * self.initial_balance / 100,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            profit_factor=profit_factor,
            win_rate=win_rate * 100,
            avg_trade_pnl=total_pnl / total_trades,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            largest_winner=max(t.pnl for t in self.trades) if self.trades else 0,
            largest_loser=min(t.pnl for t in self.trades) if self.trades else 0,
            avg_trade_duration=avg_duration,
            trades=self.trades,
            equity_curve=self.equity_curve,
            drawdown_curve=self.drawdown_curve
        )
    
    def monte_carlo(
        self,
        result: BacktestResult,
        simulations: int = 1000
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation on backtest results.
        
        Randomizes trade order to stress-test strategy robustness.
        """
        if len(result.trades) < 10:
            return {'error': 'Need at least 10 trades for Monte Carlo'}
        
        trade_pnls = [t.pnl for t in result.trades]
        
        final_balances = []
        max_drawdowns = []
        
        for _ in range(simulations):
            # Shuffle trade order
            shuffled = np.random.permutation(trade_pnls)
            
            # Simulate equity curve
            equity = [self.initial_balance]
            peak = self.initial_balance
            max_dd = 0
            
            for pnl in shuffled:
                new_balance = equity[-1] + pnl
                equity.append(new_balance)
                peak = max(peak, new_balance)
                dd = (peak - new_balance) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            
            final_balances.append(equity[-1])
            max_drawdowns.append(max_dd * 100)
        
        return {
            'simulations': simulations,
            'median_final_balance': np.median(final_balances),
            'mean_final_balance': np.mean(final_balances),
            'std_final_balance': np.std(final_balances),
            'percentile_5': np.percentile(final_balances, 5),
            'percentile_95': np.percentile(final_balances, 95),
            'median_max_drawdown': np.median(max_drawdowns),
            'worst_case_drawdown': np.percentile(max_drawdowns, 95),
            'probability_profit': sum(1 for b in final_balances if b > self.initial_balance) / simulations * 100,
            'probability_10pct_gain': sum(1 for b in final_balances if b > self.initial_balance * 1.1) / simulations * 100,
            'probability_20pct_loss': sum(1 for b in final_balances if b < self.initial_balance * 0.8) / simulations * 100
        }
    
    def format_results(self, result: BacktestResult) -> str:
        """Format results as a readable string."""
        return f"""
📊 **Backtest Results**
{'='*40}
📅 Period: {result.start_date.strftime('%Y-%m-%d') if isinstance(result.start_date, datetime) else 'N/A'} to {result.end_date.strftime('%Y-%m-%d') if isinstance(result.end_date, datetime) else 'N/A'}
💰 Initial: ${result.initial_balance:,.2f}
💵 Final: ${result.final_balance:,.2f}
📈 Total P&L: ${result.total_pnl:,.2f} ({result.total_pnl_pct:+.2f}%)

📊 **Trade Statistics**
{'='*40}
Total Trades: {result.total_trades}
Winners: {result.winning_trades} | Losers: {result.losing_trades}
Win Rate: {result.win_rate:.1f}%
Profit Factor: {result.profit_factor:.2f}
Sharpe Ratio: {result.sharpe_ratio:.2f}

💵 **P&L Analysis**
{'='*40}
Avg Trade: ${result.avg_trade_pnl:.2f}
Avg Winner: ${result.avg_winner:.2f}
Avg Loser: ${result.avg_loser:.2f}
Largest Win: ${result.largest_winner:.2f}
Largest Loss: ${result.largest_loser:.2f}

📉 **Risk Metrics**
{'='*40}
Max Drawdown: ${result.max_drawdown:.2f} ({result.max_drawdown_pct:.2f}%)
Avg Trade Duration: {result.avg_trade_duration}
"""


async def fetch_historical_data(
    exchange,
    symbol: str,
    timeframe: str = '1m',
    days: int = 30,
    limit: int = 1000
) -> pd.DataFrame:
    """Fetch historical OHLCV data from exchange."""
    all_data = []
    
    # Calculate start time
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    since = int(start_time.timestamp() * 1000)
    
    logger.info(f"Fetching {days} days of {timeframe} data for {symbol}...")
    
    while True:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        
        if not ohlcv:
            break
        
        all_data.extend(ohlcv)
        
        # Move to next batch
        since = ohlcv[-1][0] + 1
        
        if len(ohlcv) < limit:
            break
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    df = df.drop_duplicates()
    df = df.sort_index()
    
    logger.info(f"Fetched {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    return df


def save_results(result: BacktestResult, filepath: str = 'backtest_results.json'):
    """Save backtest results to JSON file."""
    data = {
        'start_date': result.start_date.isoformat() if isinstance(result.start_date, datetime) else str(result.start_date),
        'end_date': result.end_date.isoformat() if isinstance(result.end_date, datetime) else str(result.end_date),
        'initial_balance': result.initial_balance,
        'final_balance': result.final_balance,
        'total_trades': result.total_trades,
        'winning_trades': result.winning_trades,
        'losing_trades': result.losing_trades,
        'total_pnl': result.total_pnl,
        'total_pnl_pct': result.total_pnl_pct,
        'max_drawdown': result.max_drawdown,
        'max_drawdown_pct': result.max_drawdown_pct,
        'sharpe_ratio': result.sharpe_ratio,
        'profit_factor': result.profit_factor,
        'win_rate': result.win_rate,
        'avg_trade_pnl': result.avg_trade_pnl,
        'equity_curve': result.equity_curve,
        'drawdown_curve': result.drawdown_curve
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Results saved to {filepath}")
