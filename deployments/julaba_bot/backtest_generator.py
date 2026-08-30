"""
Backtest Signal Generator for Julaba ML Pipeline
Generates labeled training samples by running Julaba's signal logic on historical data.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger("Julaba.BacktestGenerator")


class IndicatorCalculator:
    """Calculate technical indicators for ML features."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def calculate_all(self) -> pd.DataFrame:
        """Calculate all required indicators."""
        df = self.df
        
        # Ensure numeric types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Moving Averages
        df['sma_15'] = df['close'].rolling(window=15, min_periods=1).mean()
        df['sma_40'] = df['close'].rolling(window=40, min_periods=1).mean()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # ADX
        df['adx'] = self._calculate_adx(df, 14)
        
        # ATR
        df['atr'] = self._calculate_atr(df, 14)
        df['atr_percent'] = (df['atr'] / df['close']) * 100
        
        # Volume
        df['volume_sma'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, 1)
        
        # SMA Distance
        df['sma_distance_percent'] = ((df['close'] - df['sma_40']) / df['sma_40'].replace(0, 1)) * 100
        
        # Hurst Exponent (simplified)
        df['hurst'] = self._calculate_hurst_rolling(df['close'], window=100)
        
        # Market Regime
        df['regime'] = df.apply(self._classify_regime, axis=1)
        
        # Time features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # === NEW ML FEATURES ===
        
        # RSI momentum (slope over 5 periods)
        df['rsi_slope'] = df['rsi'].diff(5) / 5
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Price momentum (5-bar percentage change)
        df['price_momentum'] = df['close'].pct_change(5) * 100
        
        # ATR expansion (current ATR vs 20-bar average)
        atr_avg = df['atr'].rolling(window=20, min_periods=1).mean()
        df['atr_expansion'] = df['atr'] / atr_avg.replace(0, 1)
        
        # Bollinger Band position (0 = lower band, 1 = upper band)
        bb_sma = df['close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['close'].rolling(window=20, min_periods=1).std()
        bb_upper = bb_sma + 2 * bb_std
        bb_lower = bb_sma - 2 * bb_std
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower).replace(0, 1)
        df['bb_position'] = df['bb_position'].clip(0, 1)
        
        # Volume trend (5-bar volume momentum)
        df['volume_trend'] = df['volume'].pct_change(5) * 100
        
        # Candle strength (-1 to 1, bearish to bullish)
        candle_body = df['close'] - df['open']
        candle_range = df['high'] - df['low']
        df['candle_strength'] = candle_body / candle_range.replace(0, 1)
        df['candle_strength'] = df['candle_strength'].clip(-1, 1)
        
        # Session detection (based on hour)
        if 'hour' in df.columns:
            df['is_london_session'] = ((df['hour'] >= 7) & (df['hour'] <= 16)).astype(int)
            df['is_nyc_session'] = ((df['hour'] >= 13) & (df['hour'] <= 22)).astype(int)
            df['is_asia_session'] = ((df['hour'] >= 0) & (df['hour'] <= 9)).astype(int)
        else:
            df['is_london_session'] = 0
            df['is_nyc_session'] = 0
            df['is_asia_session'] = 0
        
        return df
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(window=period, min_periods=1).mean()
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        atr = self._calculate_atr(df, period)
        
        plus_di = 100 * plus_dm.rolling(window=period, min_periods=1).mean() / atr.replace(0, 1e-10)
        minus_di = 100 * minus_dm.rolling(window=period, min_periods=1).mean() / atr.replace(0, 1e-10)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-10)
        adx = dx.rolling(window=period, min_periods=1).mean()
        
        return adx
    
    def _calculate_hurst_rolling(self, series: pd.Series, window: int = 100) -> pd.Series:
        """Calculate rolling Hurst exponent (simplified)."""
        
        def hurst(ts):
            if len(ts) < 20:
                return 0.5
            
            ts = np.array(ts)
            lags = range(2, min(20, len(ts) // 2))
            tau = []
            
            for lag in lags:
                std = np.std(np.subtract(ts[lag:], ts[:-lag]))
                tau.append(std if std > 0 else 1e-10)
            
            if len(tau) < 2:
                return 0.5
            
            try:
                poly = np.polyfit(np.log(list(lags)[:len(tau)]), np.log(tau), 1)
                return np.clip(poly[0], 0.1, 0.9)
            except:
                return 0.5
        
        return series.rolling(window=window, min_periods=50).apply(hurst, raw=False).fillna(0.5)
    
    def _classify_regime(self, row) -> str:
        """Classify market regime."""
        hurst = row.get('hurst', 0.5)
        adx = row.get('adx', 0)
        
        if pd.isna(hurst) or pd.isna(adx):
            return 'UNKNOWN'
        
        if hurst > 0.55 and adx > 25:
            return 'TRENDING'
        elif hurst < 0.45 or adx < 20:
            return 'CHOPPY'
        elif adx > 30:
            return 'VOLATILE'
        else:
            return 'WEAK_TRENDING'


class BacktestSignalGenerator:
    """Generate signals with outcomes for ML training."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
        # Parameters (MUST match live Julaba)
        self.adx_min = 25
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.volume_min_ratio = 0.75
        self.min_bars_between_signals = 8  # 2 hours at 15m
        
        # Position management
        self.atr_multiplier = 2.0
        self.tp1_r = 1.5
        self.tp2_r = 2.5
        self.tp3_r = 4.0
        self.tp1_exit_pct = 0.50
        self.tp2_exit_pct = 0.30
        self.tp3_exit_pct = 0.20
    
    def detect_crossovers(self) -> pd.DataFrame:
        """Detect SMA crossover events."""
        df = self.df
        
        df['sma_15_prev'] = df['sma_15'].shift(1)
        df['sma_40_prev'] = df['sma_40'].shift(1)
        
        df['cross_up'] = (df['sma_15_prev'] <= df['sma_40_prev']) & (df['sma_15'] > df['sma_40'])
        df['cross_down'] = (df['sma_15_prev'] >= df['sma_40_prev']) & (df['sma_15'] < df['sma_40'])
        
        return df
    
    def apply_filters(self, row, direction: str) -> bool:
        """Apply all filters to a potential signal."""
        
        # ADX filter
        if pd.isna(row['adx']) or row['adx'] < self.adx_min:
            return False
        
        # RSI filter
        if direction == 'LONG' and row['rsi'] > self.rsi_overbought:
            return False
        if direction == 'SHORT' and row['rsi'] < self.rsi_oversold:
            return False
        
        # Volume filter
        if pd.isna(row['volume_ratio']) or row['volume_ratio'] < self.volume_min_ratio:
            return False
        
        # Regime filter
        if row['regime'] == 'CHOPPY':
            return False
        
        return True
    
    def calculate_levels(self, entry_price: float, atr: float, direction: str) -> dict:
        """Calculate SL and TP levels."""
        stop_distance = atr * self.atr_multiplier
        
        if direction == 'LONG':
            sl = entry_price - stop_distance
            tp1 = entry_price + (stop_distance * self.tp1_r)
            tp2 = entry_price + (stop_distance * self.tp2_r)
            tp3 = entry_price + (stop_distance * self.tp3_r)
        else:
            sl = entry_price + stop_distance
            tp1 = entry_price - (stop_distance * self.tp1_r)
            tp2 = entry_price - (stop_distance * self.tp2_r)
            tp3 = entry_price - (stop_distance * self.tp3_r)
        
        return {
            'stop_loss': sl,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'stop_distance': stop_distance
        }
    
    def simulate_trade_outcome(self, entry_idx: int, direction: str, levels: dict) -> dict:
        """Simulate trade using future price data."""
        df = self.df
        entry_row = df.iloc[entry_idx]
        entry_price = entry_row['close']
        
        max_bars = min(100, len(df) - entry_idx - 1)
        
        tp1_hit = False
        tp2_hit = False
        tp3_hit = False
        sl_hit = False
        exit_bar = None
        exit_price = None
        exit_reason = None
        
        remaining_position = 1.0
        realized_r = 0.0
        max_drawdown = 0.0
        max_profit = 0.0
        
        for i in range(1, max_bars + 1):
            bar = df.iloc[entry_idx + i]
            high = bar['high']
            low = bar['low']
            close = bar['close']
            
            # Track drawdown and profit
            if direction == 'LONG':
                current_r = (close - entry_price) / levels['stop_distance']
                if low < entry_price:
                    dd_r = (entry_price - low) / levels['stop_distance']
                    max_drawdown = max(max_drawdown, dd_r)
                if high > entry_price:
                    profit_r = (high - entry_price) / levels['stop_distance']
                    max_profit = max(max_profit, profit_r)
            else:
                current_r = (entry_price - close) / levels['stop_distance']
                if high > entry_price:
                    dd_r = (high - entry_price) / levels['stop_distance']
                    max_drawdown = max(max_drawdown, dd_r)
                if low < entry_price:
                    profit_r = (entry_price - low) / levels['stop_distance']
                    max_profit = max(max_profit, profit_r)
            
            if direction == 'LONG':
                # Check stop loss first
                if low <= levels['stop_loss']:
                    sl_hit = True
                    exit_bar = i
                    exit_price = levels['stop_loss']
                    exit_reason = 'STOP_LOSS'
                    realized_r -= remaining_position * 1.0
                    break
                
                # Check TP levels
                if not tp1_hit and high >= levels['tp1']:
                    tp1_hit = True
                    realized_r += self.tp1_exit_pct * self.tp1_r
                    remaining_position -= self.tp1_exit_pct
                
                if not tp2_hit and high >= levels['tp2']:
                    tp2_hit = True
                    realized_r += self.tp2_exit_pct * self.tp2_r
                    remaining_position -= self.tp2_exit_pct
                
                if not tp3_hit and high >= levels['tp3']:
                    tp3_hit = True
                    realized_r += self.tp3_exit_pct * self.tp3_r
                    remaining_position = 0
                    exit_bar = i
                    exit_price = levels['tp3']
                    exit_reason = 'TP3_FULL'
                    break
            
            else:  # SHORT
                if high >= levels['stop_loss']:
                    sl_hit = True
                    exit_bar = i
                    exit_price = levels['stop_loss']
                    exit_reason = 'STOP_LOSS'
                    realized_r -= remaining_position * 1.0
                    break
                
                if not tp1_hit and low <= levels['tp1']:
                    tp1_hit = True
                    realized_r += self.tp1_exit_pct * self.tp1_r
                    remaining_position -= self.tp1_exit_pct
                
                if not tp2_hit and low <= levels['tp2']:
                    tp2_hit = True
                    realized_r += self.tp2_exit_pct * self.tp2_r
                    remaining_position -= self.tp2_exit_pct
                
                if not tp3_hit and low <= levels['tp3']:
                    tp3_hit = True
                    realized_r += self.tp3_exit_pct * self.tp3_r
                    remaining_position = 0
                    exit_bar = i
                    exit_price = levels['tp3']
                    exit_reason = 'TP3_FULL'
                    break
        
        # If trade still open after max_bars, close at current price
        if exit_reason is None:
            exit_bar = max_bars
            exit_price = df.iloc[entry_idx + max_bars]['close']
            
            if direction == 'LONG':
                remaining_r = (exit_price - entry_price) / levels['stop_distance']
            else:
                remaining_r = (entry_price - exit_price) / levels['stop_distance']
            
            realized_r += remaining_position * remaining_r
            exit_reason = 'TIMEOUT'
        
        outcome = 1 if realized_r > 0 else 0
        
        return {
            'exit_bar': exit_bar,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'tp1_hit': tp1_hit,
            'tp2_hit': tp2_hit,
            'tp3_hit': tp3_hit,
            'sl_hit': sl_hit,
            'realized_r': realized_r,
            'outcome': outcome,
            'duration_minutes': exit_bar * 15,
            'max_drawdown_r': max_drawdown,
            'max_profit_r': max_profit
        }
    
    def generate_signals(self) -> list:
        """Generate all signals with outcomes."""
        df = self.detect_crossovers()
        last_signal_idx = -self.min_bars_between_signals
        
        signals = []
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'UNKNOWN'
        
        for idx in range(50, len(df) - 100):
            row = df.iloc[idx]
            
            if idx - last_signal_idx < self.min_bars_between_signals:
                continue
            
            direction = None
            if row['cross_up']:
                direction = 'LONG'
            elif row['cross_down']:
                direction = 'SHORT'
            
            if direction is None:
                continue
            
            if not self.apply_filters(row, direction):
                continue
            
            if pd.isna(row['atr']) or row['atr'] <= 0:
                continue
            
            levels = self.calculate_levels(row['close'], row['atr'], direction)
            outcome = self.simulate_trade_outcome(idx, direction, levels)
            
            signal = {
                'signal_id': f"{symbol}_{idx}",
                'timestamp': row['timestamp'] if 'timestamp' in row else idx,
                'symbol': symbol,
                'direction': direction,
                'entry_price': row['close'],
                'entry_atr': row['atr'],
                'entry_atr_percent': row['atr_percent'],
                'entry_rsi': row['rsi'],
                'entry_adx': row['adx'],
                'entry_volume_ratio': row['volume_ratio'],
                'entry_hurst': row['hurst'],
                'entry_sma_distance_percent': row['sma_distance_percent'],
                'entry_regime': row['regime'],
                'entry_hour': row.get('hour', 12),
                'entry_day_of_week': row.get('day_of_week', 0),
                # NEW: Enhanced ML features
                'entry_rsi_slope': row.get('rsi_slope', 0.0),
                'entry_macd_hist': row.get('macd_hist', 0.0),
                'entry_price_momentum': row.get('price_momentum', 0.0),
                'entry_atr_expansion': row.get('atr_expansion', 1.0),
                'entry_bb_position': row.get('bb_position', 0.5),
                'entry_volume_trend': row.get('volume_trend', 0.0),
                'entry_candle_strength': row.get('candle_strength', 0.0),
                'is_london_session': row.get('is_london_session', 0),
                'is_nyc_session': row.get('is_nyc_session', 0),
                'is_asia_session': row.get('is_asia_session', 0),
                # Trade levels and outcome
                'stop_loss': levels['stop_loss'],
                'tp1': levels['tp1'],
                'tp2': levels['tp2'],
                'tp3': levels['tp3'],
                'exit_price': outcome['exit_price'],
                'exit_reason': outcome['exit_reason'],
                'tp1_hit': outcome['tp1_hit'],
                'tp2_hit': outcome['tp2_hit'],
                'tp3_hit': outcome['tp3_hit'],
                'sl_hit': outcome['sl_hit'],
                'realized_r': outcome['realized_r'],
                'duration_minutes': outcome['duration_minutes'],
                'max_drawdown_r': outcome['max_drawdown_r'],
                'max_profit_r': outcome['max_profit_r'],
                'outcome': outcome['outcome'],
                'source': 'backtest',
                'sample_weight': 0.3
            }
            
            signals.append(signal)
            last_signal_idx = idx
        
        return signals


def run_backtest_pipeline(data_dir: str = "./historical_data", output_file: str = None):
    """Run the complete backtest signal generation pipeline."""
    
    data_path = Path(data_dir)
    pairs = ["LINKUSDT", "ETHUSDT", "SOLUSDT"]
    
    if output_file is None:
        output_file = data_path / "backtest_training_data.csv"
    
    all_signals = []
    
    print("="*60)
    print("🎯 BACKTEST SIGNAL GENERATOR")
    print("="*60)
    
    for pair in pairs:
        csv_file = data_path / f"{pair}_15m_90d.csv"
        
        if not csv_file.exists():
            print(f"⚠️  {csv_file} not found, skipping {pair}")
            continue
        
        print(f"\n📊 Processing {pair}...")
        
        # Load data
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['symbol'] = pair
        
        print(f"  📈 Loaded {len(df)} candles")
        
        # Calculate indicators
        print(f"  🧮 Calculating indicators...")
        calculator = IndicatorCalculator(df)
        df = calculator.calculate_all()
        
        # Generate signals
        print(f"  🎯 Generating signals...")
        generator = BacktestSignalGenerator(df)
        signals = generator.generate_signals()
        
        wins = sum(1 for s in signals if s['outcome'] == 1)
        win_rate = wins / len(signals) * 100 if signals else 0
        
        print(f"  ✅ Generated {len(signals)} signals (Win rate: {win_rate:.1f}%)")
        
        all_signals.extend(signals)
    
    # Save combined signals
    if all_signals:
        signals_df = pd.DataFrame(all_signals)
        signals_df.to_csv(output_file, index=False)
        
        print("\n" + "="*60)
        print("📊 BACKTEST SUMMARY")
        print("="*60)
        print(f"Total signals: {len(all_signals)}")
        
        wins = sum(1 for s in all_signals if s['outcome'] == 1)
        print(f"Overall win rate: {wins/len(all_signals)*100:.1f}%")
        
        avg_r = np.mean([s['realized_r'] for s in all_signals])
        print(f"Average R: {avg_r:.2f}")
        
        print(f"\n💾 Saved to {output_file}")
        print("="*60)
    else:
        print("\n⚠️  No signals generated!")
    
    return all_signals


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_backtest_pipeline()
