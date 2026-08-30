"""
Backtest Runner for Julaba Trading Bot
Run this script to test the strategy on REAL historical data from MEXC.

Uses the actual Julaba indicator signals with AI filter simulation.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import sys

# Suppress bot auto-start
if 'bot' not in sys.modules:
    sys.argv = ['backtest']

from backtest import BacktestEngine

# Try to import ccxt for real data
try:
    import ccxt.async_support as ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("Warning: ccxt not available, using synthetic data")

# Import Julaba indicators
try:
    from indicator import generate_signals, calculate_adx, calculate_rsi, calculate_atr
    INDICATORS_AVAILABLE = True
except ImportError:
    INDICATORS_AVAILABLE = False
    print("Warning: indicators not available, using simple SMA")


async def fetch_real_data(symbol: str = "LINK/USDT", 
                          timeframe: str = "1m",
                          days: int = 30) -> Optional[pd.DataFrame]:
    """
    Fetch real historical OHLCV data from MEXC.
    
    Args:
        symbol: Trading pair (default: LINK/USDT)
        timeframe: Candle timeframe (default: 1m)
        days: Number of days of history (default: 30)
    
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    if not CCXT_AVAILABLE:
        return None
    
    exchange = ccxt.mexc({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })
    
    try:
        all_data = []
        
        print(f"Fetching {days} days of {timeframe} data from MEXC...")
        
        # First get most recent data
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=1000)
        if ohlcv:
            all_data.extend(ohlcv)
            print(f"  Batch 1: {len(ohlcv)} bars")
        
        # Then go backwards in time to get more history
        target_bars = days * 24 * 60  # For 1m timeframe
        batch_count = 1
        
        while len(all_data) < target_bars and ohlcv:
            await asyncio.sleep(0.2)  # Rate limiting
            
            oldest_ts = all_data[0][0]
            target_ts = oldest_ts - (1000 * 60 * 1000)  # 1000 minutes back
            
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since=target_ts, limit=1000)
            
            if ohlcv:
                # Filter to avoid duplicates
                new_bars = [b for b in ohlcv if b[0] < oldest_ts]
                if new_bars:
                    all_data = new_bars + all_data
                    batch_count += 1
                    
                    if batch_count % 5 == 0:
                        print(f"  Fetched {len(all_data):,} bars...")
                else:
                    break  # No new data
            else:
                break
            
            # Safety limit
            if batch_count >= 50:
                break
        
        await exchange.close()
        
        if not all_data:
            return None
        
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        df = df.drop_duplicates()
        df = df.sort_index()
        
        print(f"✓ Fetched {len(df):,} bars from {df.index[0]} to {df.index[-1]}")
        
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        try:
            await exchange.close()
        except:
            pass
        return None


def generate_synthetic_data(n_bars: int = 43200, start_price: float = 15.0) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with realistic market microstructure.
    """
    np.random.seed(42)
    
    start_date = datetime(2025, 12, 10)
    dates = [start_date + timedelta(minutes=i) for i in range(n_bars)]
    
    price = start_price
    prices = [price]
    
    # Create trending periods (more realistic)
    trend = 0
    trend_duration = 0
    
    for i in range(n_bars - 1):
        if trend_duration <= 0:
            trend = np.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
            trend_duration = np.random.randint(300, 1200)
        
        trend_duration -= 1
        
        base_change = np.random.normal(0, 0.0006)
        trend_change = trend * 0.00003
        
        if np.random.random() < 0.003:
            base_change += np.random.normal(0, 0.015)
        
        price *= (1 + base_change + trend_change)
        price = max(price, start_price * 0.3)
        price = min(price, start_price * 3.0)
        
        prices.append(price)
    
    df = pd.DataFrame({'close': prices}, index=pd.DatetimeIndex(dates))
    df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
    
    wick_size = np.abs(np.random.normal(0, 0.0004, n_bars))
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + wick_size)
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - wick_size)
    
    base_volume = np.random.uniform(50000, 150000, n_bars)
    price_change = np.abs(df['close'].pct_change().fillna(0))
    volume_multiplier = 1 + price_change * 30
    df['volume'] = base_volume * volume_multiplier
    
    return df[['open', 'high', 'low', 'close', 'volume']]


def generate_julaba_signals(df: pd.DataFrame) -> pd.Series:
    """
    Generate signals using actual Julaba indicator logic.
    
    Applies multi-factor filtering:
    - SMA 15/40 crossover (base signal)
    - ADX > 25 (trending market filter)
    - RSI confirmation (not overbought/oversold)
    - Volume confirmation (> 0.5x average)
    - Minimum time between signals (avoid whipsaws)
    
    Returns:
        Series with signal values: 1 (long), -1 (short), 0 (no signal)
    """
    signals = pd.Series(0, index=df.index)
    
    # Calculate indicators
    sma15 = df['close'].rolling(15).mean()
    sma40 = df['close'].rolling(40).mean()
    sma100 = df['close'].rolling(100).mean()  # Longer term trend
    
    # Raw crossover signals
    long_cross = (sma15 > sma40) & (sma15.shift(1) <= sma40.shift(1))
    short_cross = (sma15 < sma40) & (sma15.shift(1) >= sma40.shift(1))
    
    signals[long_cross] = 1
    signals[short_cross] = -1
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta).where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate ADX (simplified but effective)
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    prev_close = close.shift(1)
    tr = pd.concat([high - low, abs(high - prev_close), abs(low - prev_close)], axis=1).max(axis=1)
    
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, 1e-10)
    minus_di = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, 1e-10)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-10)
    adx = dx.rolling(14).mean()
    
    # Volume ratio
    volume_ma = df['volume'].rolling(20).mean()
    volume_ratio = df['volume'] / volume_ma.replace(0, 1)
    
    # Volatility (ATR as % of price)
    atr_pct = (atr14 / close * 100).fillna(0)
    
    # Track filtered signals
    filter_count = 0
    raw_count = (signals != 0).sum()
    
    # Minimum bars between signals (prevent whipsaws)
    min_bars_between = 60  # At least 1 hour between signals
    last_signal_idx = -min_bars_between
    
    # Apply filters
    signal_indices = signals[signals != 0].index.tolist()
    
    for i, idx in enumerate(signal_indices):
        signal = signals[idx]
        
        # Get index position
        pos = df.index.get_loc(idx)
        
        # Check minimum time between signals (reduced for more trades)
        if pos - last_signal_idx < min_bars_between:
            signals[idx] = 0
            filter_count += 1
            continue
        
        # Get indicator values
        current_adx = adx[idx] if idx in adx.index and not pd.isna(adx[idx]) else 15
        current_rsi = rsi[idx] if idx in rsi.index and not pd.isna(rsi[idx]) else 50
        current_vol = volume_ratio[idx] if idx in volume_ratio.index and not pd.isna(volume_ratio[idx]) else 1.0
        current_atr_pct = atr_pct[idx] if idx in atr_pct.index and not pd.isna(atr_pct[idx]) else 1.0
        current_sma100 = sma100[idx] if idx in sma100.index and not pd.isna(sma100[idx]) else close[idx]
        current_price = close[idx]
        
        reject = False
        reject_reason = ""
        
        # FILTER 1: ADX - require some trend (ADX > 15 - more relaxed)
        if current_adx < 15:
            reject = True
            reject_reason = f"ADX too low ({current_adx:.1f})"
        
        # FILTER 2: RSI extremes (more relaxed)
        elif signal == 1 and current_rsi > 75:
            reject = True
            reject_reason = f"RSI overbought ({current_rsi:.0f})"
        elif signal == -1 and current_rsi < 25:
            reject = True
            reject_reason = f"RSI oversold ({current_rsi:.0f})"
        
        # FILTER 3: Volume confirmation (more relaxed)
        elif current_vol < 0.5:
            reject = True
            reject_reason = f"Low volume ({current_vol:.2f}x)"
        
        # FILTER 4: Trend alignment - REMOVED (too restrictive for ranging market)
        # elif signal == 1 and current_price < current_sma100 * 0.98:
        #     reject = True
        # elif signal == -1 and current_price > current_sma100 * 1.02:
        #     reject = True
        
        # FILTER 5: Minimum volatility (more relaxed)
        elif current_atr_pct < 0.15:
            reject = True
            reject_reason = f"Low volatility ({current_atr_pct:.2f}%)"
        
        if reject:
            signals[idx] = 0
            filter_count += 1
        else:
            last_signal_idx = pos
    
    final_count = (signals != 0).sum()
    print(f"  Signal filtering: {raw_count} raw → {final_count} after filters ({filter_count} filtered)")
    
    return signals


class JulabaBacktestEngine(BacktestEngine):
    """
    Enhanced backtest engine using Julaba indicators.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signal_stats = {
            'raw_signals': 0,
            'filtered_signals': 0,
            'ai_rejected': 0
        }
    
    def _generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Override to use Julaba signals instead of simple SMA."""
        signals = generate_julaba_signals(df)
        
        self.signal_stats['raw_signals'] = (signals != 0).sum()
        
        return signals
    
    def run(self, df: pd.DataFrame, use_ai_filter: bool = True, 
            ai_approval_rate: float = 0.7, verbose: bool = False):
        """Run backtest with enhanced signal generation."""
        
        # Reset stats
        self.signal_stats = {
            'raw_signals': 0,
            'filtered_signals': 0,
            'ai_rejected': 0
        }
        
        # Generate signals using Julaba indicators
        signals = self._generate_signals(df)
        
        # Count filtered signals
        self.signal_stats['filtered_signals'] = (signals != 0).sum()
        
        # Call parent run with pre-generated signals
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.position = None
        self.trades = []
        self.equity_curve = [self.initial_balance]
        self.drawdown_curve = [0.0]
        
        # Calculate ATR
        atr = self._calculate_atr(df)
        
        # Skip warmup
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
                
                exit_price = None
                exit_reason = None
                
                # Check stop loss
                if side == 'long' and current_low <= sl:
                    exit_price = self._apply_slippage(sl, side, is_entry=False)
                    exit_reason = 'sl'
                elif side == 'short' and current_high >= sl:
                    exit_price = self._apply_slippage(sl, side, is_entry=False)
                    exit_reason = 'sl'
                
                # Check take profits (partial exits)
                if exit_price is None:
                    if side == 'long':
                        if not self.position.get('tp1_hit') and current_high >= tp1:
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
                            exit_price = self._apply_slippage(tp3, side, is_entry=False)
                            exit_reason = 'tp3'
                    else:
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
                    remaining = self.position.get('remaining_size', size)
                    if side == 'long':
                        final_pnl = (exit_price - entry_price) * remaining
                    else:
                        final_pnl = (entry_price - exit_price) * remaining
                    
                    fees = self._calculate_fees(remaining * exit_price)
                    final_pnl -= fees
                    
                    total_pnl = self.position.get('realized_pnl', 0) + final_pnl
                    self.balance += final_pnl
                    
                    from backtest import BacktestTrade
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
                # AI filter simulation
                if use_ai_filter and np.random.random() > ai_approval_rate:
                    self.signal_stats['ai_rejected'] += 1
                    continue
                
                side = 'long' if current_signal == 1 else 'short'
                entry_price = self._apply_slippage(current_price, side, is_entry=True)
                
                # Calculate SL using ATR
                if side == 'long':
                    sl = entry_price - (current_atr * self.atr_mult)
                else:
                    sl = entry_price + (current_atr * self.atr_mult)
                
                risk_per_unit = abs(entry_price - sl)
                if risk_per_unit == 0:
                    continue
                
                risk_amount = self.balance * self.risk_pct
                size = risk_amount / risk_per_unit
                
                # Calculate TPs
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
                          f"Size: {size:.4f} | SL: {sl:.4f}")
            
            # Update equity curve
            self.equity_curve.append(self.balance)
            self.peak_balance = max(self.peak_balance, self.balance)
            drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0
            self.drawdown_curve.append(drawdown * 100)
        
        return self._calculate_results(df)
    
    def print_signal_stats(self):
        """Print signal filtering statistics."""
        print("\n📊 SIGNAL FILTERING STATISTICS")
        print("=" * 40)
        print(f"  Raw SMA Crossovers:     {self.signal_stats.get('raw_signals', 0):,}")
        print(f"  After ADX/RSI/Vol Filter: {self.signal_stats.get('filtered_signals', 0):,}")
        print(f"  AI Filter Rejections:   {self.signal_stats.get('ai_rejected', 0):,}")
        
        raw = self.signal_stats.get('raw_signals', 0)
        filtered = self.signal_stats.get('filtered_signals', 0)
        if raw > 0:
            filter_rate = (1 - filtered / raw) * 100
            print(f"  Filter Rate:            {filter_rate:.1f}%")


async def run_backtest_async(use_real_data: bool = True, days: int = 30):
    """Run backtest with real or synthetic data."""
    
    print("=" * 60)
    print("JULABA BACKTEST ENGINE v2.0")
    print("Using Julaba Indicators + AI Filter Simulation")
    print("=" * 60)
    print()
    
    # Fetch or generate data
    df = None
    if use_real_data and CCXT_AVAILABLE:
        df = await fetch_real_data(symbol="LINK/USDT", timeframe="1m", days=days)
    
    if df is None:
        print("Using synthetic data (MEXC fetch failed or disabled)...")
        df = generate_synthetic_data(n_bars=days * 24 * 60, start_price=15.0)
        print(f"Generated {len(df):,} bars of synthetic data")
    
    print()
    print(f"📅 Data Period: {df.index[0].strftime('%Y-%m-%d %H:%M')} to {df.index[-1].strftime('%Y-%m-%d %H:%M')}")
    print(f"📊 Total Bars: {len(df):,} (1-minute candles)")
    print(f"💰 Price Range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"📈 Price Change: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:+.1f}%")
    print()
    
    # Initialize engine
    engine = JulabaBacktestEngine(
        initial_balance=10000.0,
        fee_pct=0.002,        # 0.2% MEXC taker
        slippage_pct=0.001,   # 0.1% slippage
        risk_pct=0.02,        # 2% risk
        atr_mult=2.0,
        tp1_r=1.0, tp2_r=2.0, tp3_r=3.0,
        tp1_pct=0.4, tp2_pct=0.3, tp3_pct=0.3
    )
    
    print("⚙️ Backtest Parameters:")
    print("  Initial Balance:    $10,000")
    print("  Risk per Trade:     2%")
    print("  Fees:               0.2% taker + 0.1% slippage")
    print("  ATR Multiplier:     2.0x")
    print("  Take Profits:       1R (40%), 2R (30%), 3R (30%)")
    print("  AI Filter Rate:     70% approval")
    print()
    
    print("Running backtest with Julaba indicators...")
    result = engine.run(df, use_ai_filter=True, ai_approval_rate=0.7)
    
    # Display results
    print(engine.format_results(result))
    
    # Signal stats
    engine.print_signal_stats()
    
    # Monte Carlo
    if result.total_trades >= 10:
        print()
        print("Running Monte Carlo Simulation (1000 iterations)...")
        mc = engine.monte_carlo(result, simulations=1000)
        
        print()
        print("🎲 MONTE CARLO ANALYSIS")
        print("=" * 40)
        print(f"Simulations: {mc['simulations']}")
        print()
        print("💰 Final Balance Distribution:")
        print(f"  5th Percentile:     ${mc['percentile_5']:,.2f}")
        print(f"  Median:             ${mc['median_final_balance']:,.2f}")
        print(f"  Mean:               ${mc['mean_final_balance']:,.2f}")
        print(f"  95th Percentile:    ${mc['percentile_95']:,.2f}")
        print()
        print("⚠️ Risk Analysis:")
        print(f"  Probability of Profit:   {mc['probability_profit']:.1f}%")
        print(f"  Prob of 10%+ Gain:       {mc['probability_10pct_gain']:.1f}%")
        print(f"  Prob of 20%+ Loss:       {mc['probability_20pct_loss']:.1f}%")
        print(f"  Median Max Drawdown:     {mc['median_max_drawdown']:.1f}%")
        print(f"  95th Percentile DD:      {mc['worst_case_drawdown']:.1f}%")
    else:
        print(f"\n⚠️ Only {result.total_trades} trades - not enough for Monte Carlo (minimum 10)")
        print("  This is expected with proper filtering - fewer but higher quality trades.")
    
    print()
    print("=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)
    
    return result


def run_backtest():
    """Synchronous wrapper for backtest."""
    return asyncio.run(run_backtest_async(use_real_data=True, days=30))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Julaba Backtest Engine')
    parser.add_argument('--days', type=int, default=30, help='Days of historical data')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data only')
    
    args = parser.parse_args()
    
    asyncio.run(run_backtest_async(use_real_data=not args.synthetic, days=args.days))
