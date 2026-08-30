# indicator.py
# Mathematical Trading Signal Generator for Julaba
# Uses rigorous statistical analysis and mathematical proofs

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Tuple, Dict, Optional, Any, List
import logging
from scipy import stats as scipy_stats
from arch import arch_model
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

logger = logging.getLogger("Julaba.Indicator")


# ============== MATHEMATICAL FUNCTIONS ==============

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.
    
    Mathematical Definition:
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
    
    Range: [0, 100]
    - RSI > 70: Overbought (statistically likely to reverse down)
    - RSI < 30: Oversold (statistically likely to reverse up)
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range.
    
    Mathematical Definition:
        TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        ATR = SMA(TR, period)
    
    Measures volatility in absolute dollar terms.
    """
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']
    
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr


def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate Average Directional Index.
    
    Mathematical Definition:
        +DM = High - PrevHigh (if > 0 and > -DM, else 0)
        -DM = PrevLow - Low (if > 0 and > +DM, else 0)
        TR = max(H-L, |H-PC|, |L-PC|)
        +DI = 100 * SMA(+DM) / SMA(TR)
        -DI = 100 * SMA(-DM) / SMA(TR)
        DX = 100 * |+DI - -DI| / (+DI + -DI)
        ADX = SMA(DX, period)
    
    Interpretation:
        ADX < 20: Weak trend (ranging market)
        ADX 20-25: Trend developing
        ADX 25-50: Strong trend
        ADX > 50: Very strong trend
    
    Returns the last ADX value.
    """
    high = df['high'] if 'high' in df.columns else df['High']
    low = df['low'] if 'low' in df.columns else df['Low']
    close = df['close'] if 'close' in df.columns else df['Close']
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period, min_periods=1).mean()
    plus_di = 100 * plus_dm.rolling(window=period, min_periods=1).mean() / atr.replace(0, 1e-10)
    minus_di = 100 * minus_dm.rolling(window=period, min_periods=1).mean() / atr.replace(0, 1e-10)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1e-10)
    adx = dx.rolling(window=period, min_periods=1).mean()
    
    return float(adx.iloc[-1]) if len(adx) > 0 else 0


def calculate_volatility_regime(prices: pd.Series, short_window: int = 20, long_window: int = 100) -> Dict[str, Any]:
    """
    Classify volatility regime.
    
    Compares recent volatility to historical volatility.
    """
    if len(prices) < long_window:
        return {'regime': 'unknown', 'volatility_ratio': 1.0}
    
    returns = prices.pct_change().dropna()
    
    short_vol = returns.tail(short_window).std() * np.sqrt(365 * 24 * 4)  # 15-min candles: 365 days * 24 hours * 4 per hour
    long_vol = returns.tail(long_window).std() * np.sqrt(365 * 24 * 4)
    
    ratio = short_vol / long_vol if long_vol > 0 else 1.0
    
    if ratio > 1.5:
        regime = 'high'
    elif ratio < 0.5:
        regime = 'low'
    else:
        regime = 'normal'
    
    return {
        'regime': regime,
        'short_vol': float(short_vol),
        'long_vol': float(long_vol),
        'volatility_ratio': float(ratio)
    }


def calculate_hurst_exponent(prices: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate Hurst Exponent using R/S analysis with proper sub-segment method.
    
    H > 0.5: Trending (persistent)
    H = 0.5: Random walk
    H < 0.5: Mean-reverting
    """
    n = len(prices)
    if n < max_lag * 4:
        return 0.5
    
    try:
        returns = prices.pct_change().dropna().values
        if len(returns) < 20:
            return 0.5
        
        # Use sub-segment sizes from 10 to n/4
        min_seg = 10
        max_seg = len(returns) // 4
        if max_seg <= min_seg:
            return 0.5
        
        # Generate segment sizes (log-spaced for better regression)
        seg_sizes = np.unique(np.logspace(np.log10(min_seg), np.log10(max_seg), num=15).astype(int))
        seg_sizes = seg_sizes[seg_sizes >= min_seg]
        
        if len(seg_sizes) < 3:
            return 0.5
        
        rs_values = []
        for seg_size in seg_sizes:
            n_segments = len(returns) // seg_size
            if n_segments < 1:
                continue
            
            rs_seg = []
            for i in range(n_segments):
                segment = returns[i * seg_size:(i + 1) * seg_size]
                mean_ret = segment.mean()
                std_ret = segment.std()
                if std_ret == 0:
                    continue
                cumdev = np.cumsum(segment - mean_ret)
                r = cumdev.max() - cumdev.min()
                rs_seg.append(r / std_ret)
            
            if rs_seg:
                rs_values.append((seg_size, np.mean(rs_seg)))
        
        if len(rs_values) < 3:
            return 0.5
        
        log_sizes = np.log([x[0] for x in rs_values])
        log_rs = np.log([x[1] for x in rs_values])
        
        slope, _ = np.polyfit(log_sizes, log_rs, 1)
        return float(np.clip(slope, 0, 1))
        
    except Exception:
        return 0.5


def calculate_kelly_fraction(returns: pd.Series) -> float:
    """
    Calculate Kelly Criterion fraction.
    
    f* = (p * W - q * L) / (W * L)
    where p = win rate, W = avg win, q = loss rate, L = avg loss
    """
    if len(returns) < 20:
        return 0.01
    
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    if len(wins) == 0 or len(losses) == 0:
        return 0.01
    
    win_rate = len(wins) / len(returns)
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    
    if avg_loss == 0 or avg_win == 0:
        return 0.01
    
    # Standard Kelly: f* = (p * b - q) / b where b = avg_win/avg_loss
    win_loss_ratio = avg_win / avg_loss
    kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
    return float(np.clip(kelly, 0.005, 0.25))


def calculate_expectancy(returns: pd.Series) -> float:
    """Calculate expected value per trade."""
    if len(returns) < 10:
        return 0
    
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    if len(wins) == 0 or len(losses) == 0:
        return returns.mean()
    
    win_rate = len(wins) / len(returns)
    avg_win = wins.mean()
    avg_loss = abs(losses.mean())
    
    return win_rate * avg_win - (1 - win_rate) * avg_loss


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05, periods_per_day: int = 96) -> float:
    """Calculate annualized Sharpe ratio.
    
    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate (default 5%)
        periods_per_day: Number of periods per day (96 for 15-min, 24 for hourly, 1 for daily)
    """
    if len(returns) < 20:
        return 0
    
    periods_per_year = 365 * periods_per_day  # Crypto trades 365 days/year × periods per day
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if excess_returns.std() == 0:
        return 0
    
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(periods_per_year)
    return float(sharpe)


# ============== MARKET REGIME CLASSIFICATION ==============

class MarketRegime:
    """
    Classify market regime using multiple indicators.
    """
    
    @staticmethod
    def classify(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive market regime classification.
        """
        if len(df) < 50:
            return {
                'regime': 'UNKNOWN',
                'tradeable': False,
                'confidence': 0,
                'reason': 'Insufficient data'
            }
        
        close = df['close'] if 'close' in df.columns else df['Close']
        
        adx = calculate_adx(df)
        hurst = calculate_hurst_exponent(close)
        vol_regime = calculate_volatility_regime(close)
        
        # Regime classification
        if adx >= 30 and hurst >= 0.55:
            regime = 'STRONG_TRENDING'
            confidence = 0.9
            tradeable = True
        elif adx >= 25:
            regime = 'TRENDING'
            confidence = 0.7
            tradeable = True
        elif adx >= 20:
            regime = 'WEAK_TRENDING'
            confidence = 0.5
            tradeable = True
        elif hurst < 0.35:  # Lowered from 0.4 to allow more trades
            regime = 'CHOPPY'
            confidence = 0.3
            tradeable = False
        else:
            regime = 'RANGING'
            confidence = 0.4
            tradeable = False
        
        # Adjust for extreme volatility
        if vol_regime['regime'] == 'high':
            confidence *= 0.8
        
        reason = ''
        if not tradeable:
            if adx < 25:
                reason = f'ADX too low ({adx:.1f} < 25)'
            elif hurst < 0.35:
                reason = f'Choppy market (Hurst={hurst:.2f})'
            else:
                reason = 'Ranging market'
        
        return {
            'regime': regime,
            'tradeable': tradeable,
            'confidence': confidence,
            'adx': adx,
            'hurst': hurst,
            'volatility': vol_regime['regime'],
            'volatility_ratio': vol_regime['volatility_ratio'],
            'reason': reason
        }


# ============== SIGNAL GENERATION ==============

def generate_signals(df: pd.DataFrame, track_filters: bool = True) -> pd.DataFrame:
    """
    Enhanced signal generation with multi-factor confluence.
    
    Strategy: SMA(15) / SMA(40) crossover + RSI + Volume + BTC alignment
    Filters: ADX regime, volume confirmation, momentum confluence
    
    Also tracks filtered signals in df._filter_info for prefilter stats.
    """
    df = df.copy()
    
    # Initialize filter tracking
    filter_info = {
        'raw_signals': 0,
        'filtered': 0,
        'filter_reasons': [],
        'original_signal': 0
    }
    
    # Normalize column names
    col_map = {}
    for col in df.columns:
        if col.lower() == 'close':
            col_map[col] = 'close'
        elif col.lower() == 'high':
            col_map[col] = 'high'
        elif col.lower() == 'low':
            col_map[col] = 'low'
        elif col.lower() == 'open':
            col_map[col] = 'open'
        elif col.lower() == 'volume':
            col_map[col] = 'volume'
    df = df.rename(columns=col_map)
    
    # SMA calculation (OPTIMIZED: 15/40 based on research)
    df['SMA15'] = df['close'].rolling(window=15, min_periods=1).mean()
    df['SMA40'] = df['close'].rolling(window=40, min_periods=1).mean()
    
    # Generate raw signals
    df['Side'] = 0
    
    # Long: SMA15 crosses above SMA40
    long_condition = (df['SMA15'] > df['SMA40']) & (df['SMA15'].shift(1) <= df['SMA40'].shift(1))
    df.loc[long_condition, 'Side'] = 1
    
    # Short: SMA15 crosses below SMA40
    short_condition = (df['SMA15'] < df['SMA40']) & (df['SMA15'].shift(1) >= df['SMA40'].shift(1))
    df.loc[short_condition, 'Side'] = -1
    
    # ENHANCED FILTERING (only if we have enough data)
    if len(df) >= 50:
        adx = calculate_adx(df)
        rsi = calculate_rsi(df['close'])
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # Get volume features
        vol_features = calculate_volume_features(df)
        volume_ratio = vol_features.get('volume_ratio', 1.0)
        
        # Get BTC alignment
        btc_features = calculate_btc_correlation(df)
        btc_correlation = btc_features.get('btc_correlation', 0)
        
        signal_rows = df['Side'] != 0
        
        if signal_rows.any():
            # Track raw signal before filtering
            last_signal = df.loc[signal_rows, 'Side'].iloc[-1] if signal_rows.any() else 0
            filter_info['raw_signals'] = 1
            filter_info['original_signal'] = int(last_signal)
            
            filter_reasons_list = []
            should_filter = False
            
            # FILTER 1: ADX regime (adaptive threshold)
            # Lower threshold for SHORTS since they work in weaker trends too
            # Also lower if volume is strong
            if last_signal == -1:  # SHORT signals get relaxed ADX
                adx_threshold = 15
            elif volume_ratio > 1.5:
                adx_threshold = 20
            else:
                adx_threshold = 25
            if adx < adx_threshold:
                filter_reasons_list.append(f"ADX={adx:.1f} < {adx_threshold}")
                should_filter = True
            
            # FILTER 2: Volume confirmation
            # Require at least 0.3x average volume (relaxed from 0.5x)
            min_vol = 0.2 if last_signal == -1 else 0.5  # Shorts need less volume confirmation
            if volume_ratio < min_vol:
                filter_reasons_list.append(f"Low volume ({volume_ratio:.2f}x avg)")
                should_filter = True
            
            # FILTER 3: RSI extremes - STRICT overbought/oversold rejection
            # Don't go LONG if already overbought (price likely to reverse DOWN)
            # Don't go SHORT if already oversold (price likely to reverse UP)
            if last_signal == 1 and current_rsi > 70:
                filter_reasons_list.append(f"RSI OVERBOUGHT ({current_rsi:.0f})")
                should_filter = True
                logger.warning(f"[SIGNAL BLOCKED] LONG rejected - RSI overbought at {current_rsi:.1f} (>70)")
            elif last_signal == -1 and current_rsi < 30:
                filter_reasons_list.append(f"RSI OVERSOLD ({current_rsi:.0f})")
                should_filter = True
                logger.warning(f"[SIGNAL BLOCKED] SHORT rejected - RSI oversold at {current_rsi:.1f} (<30)")
            
            # FILTER 4: BTC alignment (if highly correlated)
            # If LINK correlates >0.7 with BTC, check BTC trend
            if abs(btc_correlation) > 0.7:
                try:
                    btc_df = _fetch_btc_data_sync()
                    if btc_df is not None and len(btc_df) >= 20:
                        btc_sma_fast = btc_df['close'].rolling(10).mean().iloc[-1]
                        btc_sma_slow = btc_df['close'].rolling(30).mean().iloc[-1]
                        btc_bullish = btc_sma_fast > btc_sma_slow
                        
                        # Don't short LINK when BTC is bullish (high correlation)
                        if last_signal == -1 and btc_bullish and btc_correlation > 0.7:
                            filter_reasons_list.append("BTC bullish (high correlation)")
                            should_filter = True
                        # Don't long LINK when BTC is bearish (high correlation)
                        elif last_signal == 1 and not btc_bullish and btc_correlation > 0.7:
                            filter_reasons_list.append("BTC bearish (high correlation)")
                            should_filter = True
                except Exception as e:
                    logger.debug(f"BTC filter error: {e}")
            
            filter_info['filter_reasons'] = filter_reasons_list
            
            if should_filter:
                filter_info['filtered'] = 1
                logger.debug(f"Signals filtered: {', '.join(filter_reasons_list)}")
                # Only zero the LATEST signal, not ALL historical signals
                # This preserves historical data integrity for backtests/training
                last_signal_idx = df.index[signal_rows][-1]  # Get index of last signal
                df.loc[last_signal_idx, 'Side'] = 0
            else:
                # Log confluence factors for signal
                confluence = []
                if adx >= 25:
                    confluence.append(f"ADX={adx:.0f}")
                if volume_ratio >= 1.0:
                    confluence.append(f"Vol={volume_ratio:.1f}x")
                if (last_signal == 1 and current_rsi < 50) or (last_signal == -1 and current_rsi > 50):
                    confluence.append(f"RSI={current_rsi:.0f}")
                logger.info(f"Signal confirmed with confluence: {', '.join(confluence)}")
    
    # Attach filter info to dataframe for bot to access
    df.attrs['filter_info'] = filter_info
    
    return df


def generate_mean_reversion_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean-Reversion Strategy for RANGING markets.
    
    Uses Bollinger Bands + RSI for mean-reversion entries.
    Only activates when ADX < 25 (non-trending market).
    """
    df = df.copy()
    
    # Normalize column names
    col_map = {}
    for col in df.columns:
        if col.lower() == 'close':
            col_map[col] = 'close'
    df = df.rename(columns=col_map)
    
    if len(df) < 50:
        df['MR_Side'] = 0
        return df
    
    closes = df['close']
    
    # Calculate Bollinger Bands
    sma20 = closes.rolling(20).mean()
    std20 = closes.rolling(20).std()
    bb_upper = sma20 + (std20 * 2)
    bb_lower = sma20 - (std20 * 2)
    
    # Calculate RSI
    rsi = calculate_rsi(closes, period=14)
    
    # Check if we're in a ranging market (ADX < 25)
    adx = calculate_adx(df)
    is_ranging = adx < 25
    
    df['MR_Side'] = 0
    
    if not is_ranging:
        return df  # Only use mean-reversion in ranging markets
    
    # Mean-reversion signals:
    # Long: Price touches lower BB + RSI < 30 (oversold)
    # Short: Price touches upper BB + RSI > 70 (overbought)
    
    for i in range(20, len(df)):
        price = closes.iloc[i]
        current_rsi = rsi.iloc[i]
        lower_band = bb_lower.iloc[i]
        upper_band = bb_upper.iloc[i]
        
        # Long signal: oversold bounce
        if price <= lower_band and current_rsi < 30:
            df.iloc[i, df.columns.get_loc('MR_Side')] = 1
        # Short signal: overbought reversal  
        elif price >= upper_band and current_rsi > 70:
            df.iloc[i, df.columns.get_loc('MR_Side')] = -1
    
    return df


def generate_regime_aware_signals(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate signals based on market regime.
    
    - TRENDING (ADX >= 25): Use SMA crossover (trend-following)
    - RANGING (ADX < 25): Use mean-reversion (Bollinger + RSI)
    - CHOPPY (Hurst < 0.35): No trades (unless strong momentum override)
    
    Returns (df_with_signals, regime_info)
    """
    if len(df) < 50:
        df['Side'] = 0
        return df, {'regime': 'UNKNOWN', 'strategy': 'none'}
    
    # Get regime classification
    regime = MarketRegime.classify(df)
    regime_type = regime['regime']
    adx = regime['adx']
    hurst = regime['hurst']
    
    # Calculate momentum for potential override
    close = df['close']
    roc_5 = ((close.iloc[-1] / close.iloc[-6]) - 1) * 100 if len(close) >= 6 else 0
    strong_momentum = abs(roc_5) > 1.0  # ROC5 > 1% = strong momentum
    
    # CHOPPY: Don't trade (unless strong momentum overrides)
    if (regime_type == 'CHOPPY' or hurst < 0.35) and not strong_momentum:
        # Check if there would have been a signal (before we zero it out)
        temp_df = generate_signals(df.copy())
        filter_info = temp_df.attrs.get('filter_info', {})
        
        # If there was a raw signal, mark it as filtered by regime
        if filter_info.get('raw_signals', 0) > 0:
            filter_info['filtered'] = 1
            reasons = filter_info.get('filter_reasons', [])
            reasons.append(f"CHOPPY regime (Hurst={hurst:.2f})")
            filter_info['filter_reasons'] = reasons
        
        df['Side'] = 0
        return df, {
            'regime': 'CHOPPY',
            'strategy': 'NO_TRADE',
            'reason': f'Choppy market (Hurst={hurst:.2f}, ROC5={roc_5:.2f}% - need >1%)',
            'adx': adx,
            'hurst': hurst,
            'filter_info': filter_info
        }
    
    # RANGING: Use mean-reversion
    if regime_type == 'RANGING' or adx < 25:
        # First check what trend signal would have been
        temp_df = generate_signals(df.copy())
        trend_filter_info = temp_df.attrs.get('filter_info', {})
        
        # Generate mean reversion signals
        df = generate_mean_reversion_signals(df)
        df['Side'] = df['MR_Side']
        
        # For MR signals, track them too
        mr_signals = (df['MR_Side'] != 0).sum()
        filter_info = {
            'raw_signals': max(trend_filter_info.get('raw_signals', 0), mr_signals),
            'filtered': trend_filter_info.get('filtered', 0),
            'filter_reasons': trend_filter_info.get('filter_reasons', []),
            'original_signal': trend_filter_info.get('original_signal', 0)
        }
        
        return df, {
            'regime': 'RANGING',
            'strategy': 'MEAN_REVERSION',
            'reason': f'Ranging market (ADX={adx:.1f})',
            'adx': adx,
            'hurst': hurst,
            'filter_info': filter_info
        }
    
    # TRENDING: Use SMA crossover
    df = generate_signals(df)
    
    # Get filter info from generate_signals
    filter_info = df.attrs.get('filter_info', {})
    
    return df, {
        'regime': regime_type,
        'strategy': 'TREND_FOLLOWING',
        'reason': f'Trending market (ADX={adx:.1f})',
        'adx': adx,
        'hurst': hurst,
        'filter_info': filter_info  # Pass filter tracking info to bot
    }


def smart_btc_filter(df: pd.DataFrame, signal: int, btc_df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Smarter BTC correlation filter that considers:
    1. Rolling correlation (not just static)
    2. Relative strength (LINK vs BTC momentum)
    3. Divergence opportunities
    
    Returns dict with filter decision and reasoning.
    """
    result = {
        'should_filter': False,
        'reason': None,
        'btc_trend': 'neutral',
        'correlation': 0,
        'relative_strength': 0,
        'divergence': False
    }
    
    try:
        btc_features = calculate_btc_correlation(df, btc_df)
        correlation = btc_features['btc_correlation']
        relative_strength = btc_features['btc_relative_strength']
        beta = btc_features['btc_beta']
        
        result['correlation'] = correlation
        result['relative_strength'] = relative_strength
        
        # Get BTC trend
        if btc_df is None:
            btc_df = _fetch_btc_data_sync()
        
        if btc_df is None or len(btc_df) < 20:
            return result
        
        btc_closes = btc_df['close']
        btc_sma_fast = btc_closes.rolling(10).mean().iloc[-1]
        btc_sma_slow = btc_closes.rolling(30).mean().iloc[-1]
        btc_bullish = btc_sma_fast > btc_sma_slow
        btc_rsi = calculate_rsi(btc_closes).iloc[-1]
        
        result['btc_trend'] = 'bullish' if btc_bullish else 'bearish'
        result['btc_rsi'] = btc_rsi
        
        # === SMART FILTER LOGIC ===
        
        # Case 1: High correlation (> 0.7) - follow BTC
        if correlation > 0.7:
            if signal == 1 and not btc_bullish:
                result['should_filter'] = True
                result['reason'] = f"High BTC correlation ({correlation:.2f}), BTC bearish"
            elif signal == -1 and btc_bullish:
                result['should_filter'] = True
                result['reason'] = f"High BTC correlation ({correlation:.2f}), BTC bullish"
        
        # Case 2: Negative correlation (< -0.3) - inverse relationship
        elif correlation < -0.3:
            # Inverse logic: long LINK when BTC is weak
            if signal == 1 and btc_bullish and relative_strength < -5:
                result['should_filter'] = True
                result['reason'] = f"Negative correlation, BTC outperforming"
        
        # Case 3: Divergence opportunity
        # LINK showing strength while BTC weak, or vice versa
        if abs(correlation) < 0.5 and abs(relative_strength) > 3:
            result['divergence'] = True
            if signal == 1 and relative_strength > 3:
                result['should_filter'] = False  # LINK leading, good for longs
                result['reason'] = f"LINK outperforming BTC (RS={relative_strength:.1f}%)"
            elif signal == -1 and relative_strength < -3:
                result['should_filter'] = False  # LINK lagging, good for shorts
                result['reason'] = f"LINK underperforming BTC (RS={relative_strength:.1f}%)"
        
        # Case 4: BTC extreme RSI - be cautious
        if btc_rsi > 80 and signal == 1:
            result['should_filter'] = True
            result['reason'] = f"BTC overbought (RSI={btc_rsi:.0f}), risky for longs"
        elif btc_rsi < 20 and signal == -1:
            result['should_filter'] = True
            result['reason'] = f"BTC oversold (RSI={btc_rsi:.0f}), risky for shorts"
            
    except Exception as e:
        logger.debug(f"Smart BTC filter error: {e}")
    
    return result


def generate_signals_with_details(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generate signals and return detailed analysis.
    Returns (df_with_signals, analysis_dict)
    """
    df_signals = generate_signals(df)
    
    analysis = {
        'signal': 0,
        'confluence_score': 0,
        'filters_passed': [],
        'filters_failed': [],
        'recommendation': 'WAIT'
    }
    
    if len(df) < 50:
        analysis['recommendation'] = 'INSUFFICIENT_DATA'
        return df_signals, analysis
    
    # Get current signal
    last_signal = df_signals['Side'].iloc[-1]
    analysis['signal'] = int(last_signal)
    
    # Calculate confluence
    adx = calculate_adx(df)
    rsi = calculate_rsi(df['close']).iloc[-1]
    vol_features = calculate_volume_features(df)
    btc_features = calculate_btc_correlation(df)
    momentum = calculate_momentum_divergence(df)
    
    confluence = 0
    
    # ADX strength
    if adx >= 30:
        confluence += 2
        analysis['filters_passed'].append(f'Strong trend (ADX={adx:.0f})')
    elif adx >= 25:
        confluence += 1
        analysis['filters_passed'].append(f'Trending (ADX={adx:.0f})')
    else:
        analysis['filters_failed'].append(f'Weak trend (ADX={adx:.0f})')
    
    # Volume
    if vol_features['volume_ratio'] >= 1.2:
        confluence += 2
        analysis['filters_passed'].append(f"High volume ({vol_features['volume_ratio']:.1f}x)")
    elif vol_features['volume_ratio'] >= 0.8:
        confluence += 1
        analysis['filters_passed'].append(f"Normal volume ({vol_features['volume_ratio']:.1f}x)")
    else:
        analysis['filters_failed'].append(f"Low volume ({vol_features['volume_ratio']:.1f}x)")
    
    # RSI alignment
    if last_signal == 1:
        if rsi < 40:
            confluence += 2
            analysis['filters_passed'].append(f'RSI favorable ({rsi:.0f})')
        elif rsi < 60:
            confluence += 1
    elif last_signal == -1:
        if rsi > 60:
            confluence += 2
            analysis['filters_passed'].append(f'RSI favorable ({rsi:.0f})')
        elif rsi > 40:
            confluence += 1
    
    # Momentum
    if abs(momentum['momentum_strength']) > 1:
        if (last_signal == 1 and momentum['momentum_strength'] > 0) or \
           (last_signal == -1 and momentum['momentum_strength'] < 0):
            confluence += 1
            analysis['filters_passed'].append(f"Momentum aligned")
    
    analysis['confluence_score'] = confluence
    
    # Recommendation
    if last_signal != 0:
        if confluence >= 4:
            analysis['recommendation'] = 'STRONG_ENTRY'
        elif confluence >= 2:
            analysis['recommendation'] = 'ENTRY'
        else:
            analysis['recommendation'] = 'WEAK_ENTRY'
    else:
        analysis['recommendation'] = 'NO_SIGNAL'
    
    return df_signals, analysis


def get_regime_analysis(df_or_closes, atr: float = 0) -> Dict[str, Any]:
    """
    Get comprehensive regime analysis.
    """
    if isinstance(df_or_closes, pd.DataFrame):
        df = df_or_closes
        close = df['close'] if 'close' in df.columns else df['Close']
    else:
        close = pd.Series(df_or_closes)
        df = pd.DataFrame({'close': close, 'high': close, 'low': close})
    
    if len(close) < 50:
        return {
            'tradeable': False,
            'reason': 'Insufficient data',
            'regime': 'UNKNOWN',
            'adx': 0,
            'hurst': 0.5,
            'confidence': 0,
            'volatility': 'unknown'
        }
    
    regime = MarketRegime.classify(df)
    
    returns = close.pct_change().dropna()
    kelly = calculate_kelly_fraction(returns.tail(100))
    expectancy = calculate_expectancy(returns.tail(50))
    
    regime['kelly_fraction'] = kelly
    regime['expectancy'] = expectancy
    
    return regime


def calculate_optimal_size(df: pd.DataFrame, balance: float, base_risk: float = 0.015) -> Tuple[float, Dict]:
    """
    Calculate mathematically optimal position size.
    
    Uses Kelly Criterion adjusted for regime confidence.
    """
    close = df['close'] if 'close' in df.columns else df['Close']
    returns = close.pct_change().dropna()
    
    regime = MarketRegime.classify(df)
    kelly = calculate_kelly_fraction(returns.tail(100))
    
    confidence_multiplier = regime['confidence']
    
    adjusted_risk = base_risk * confidence_multiplier * (kelly / base_risk)
    adjusted_risk = float(np.clip(adjusted_risk, 0.005, 0.025))
    
    analysis = {
        'base_risk': base_risk,
        'kelly_fraction': kelly,
        'regime_confidence': regime['confidence'],
        'adjusted_risk': adjusted_risk,
        'regime': regime['regime'],
        'tradeable': regime['tradeable']
    }
    
    logger.debug(f"Position sizing: Kelly={kelly:.3f}, Conf={confidence_multiplier:.2f}, "
                 f"Risk={adjusted_risk:.2%}")
    
    return adjusted_risk, analysis


# ============== INTELLIGENT FEATURES ==============

def detect_candlestick_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect key candlestick patterns for entry timing.
    """
    if len(df) < 3:
        return {'pattern': None, 'bullish': None, 'strength': 0}
    
    c = df.tail(3).copy().reset_index(drop=True)
    
    open_col = 'open' if 'open' in c.columns else 'Open'
    high_col = 'high' if 'high' in c.columns else 'High'
    low_col = 'low' if 'low' in c.columns else 'Low'
    close_col = 'close' if 'close' in c.columns else 'Close'
    
    curr_open = c[open_col].iloc[-1]
    curr_high = c[high_col].iloc[-1]
    curr_low = c[low_col].iloc[-1]
    curr_close = c[close_col].iloc[-1]
    
    prev_open = c[open_col].iloc[-2]
    prev_close = c[close_col].iloc[-2]
    
    body = abs(curr_close - curr_open)
    upper_wick = curr_high - max(curr_open, curr_close)
    lower_wick = min(curr_open, curr_close) - curr_low
    total_range = curr_high - curr_low
    
    if total_range == 0:
        return {'pattern': None, 'bullish': None, 'strength': 0}
    
    body_ratio = body / total_range
    patterns = []
    
    # Doji
    if body_ratio < 0.1:
        patterns.append({'pattern': 'doji', 'bullish': None, 'strength': 0.5, 'description': 'Indecision'})
    
    # Bullish Engulfing
    if (prev_close < prev_open and curr_close > curr_open and 
        curr_close > prev_open and curr_open < prev_close):
        patterns.append({'pattern': 'bullish_engulfing', 'bullish': True, 'strength': 0.8, 'description': 'Strong bullish reversal'})
    
    # Bearish Engulfing
    if (prev_close > prev_open and curr_close < curr_open and 
        curr_close < prev_open and curr_open > prev_close):
        patterns.append({'pattern': 'bearish_engulfing', 'bullish': False, 'strength': 0.8, 'description': 'Strong bearish reversal'})
    
    # Hammer
    if body_ratio < 0.3 and lower_wick > body * 2 and upper_wick < body:
        patterns.append({'pattern': 'hammer', 'bullish': True, 'strength': 0.7, 'description': 'Rejection of lower prices'})
    
    # Shooting Star
    if body_ratio < 0.3 and upper_wick > body * 2 and lower_wick < body:
        patterns.append({'pattern': 'shooting_star', 'bullish': False, 'strength': 0.7, 'description': 'Rejection of higher prices'})
    
    # Pin Bars
    if lower_wick > total_range * 0.6:
        patterns.append({'pattern': 'pin_bar_bullish', 'bullish': True, 'strength': 0.75, 'description': 'Strong rejection wick (bullish)'})
    elif upper_wick > total_range * 0.6:
        patterns.append({'pattern': 'pin_bar_bearish', 'bullish': False, 'strength': 0.75, 'description': 'Strong rejection wick (bearish)'})
    
    if patterns:
        return max(patterns, key=lambda x: x['strength'])
    
    return {'pattern': None, 'bullish': None, 'strength': 0}


def calculate_drawdown_adjusted_risk(
    base_risk: float,
    current_balance: float,
    peak_balance: float,
    consecutive_losses: int,
    consecutive_wins: int
) -> Dict[str, Any]:
    """
    Smart Drawdown Control - automatically adjust risk based on performance.
    """
    if peak_balance <= 0:
        peak_balance = current_balance
    
    drawdown = (peak_balance - current_balance) / peak_balance if peak_balance > 0 else 0
    drawdown_pct = drawdown * 100
    
    if drawdown_pct >= 20:
        dd_multiplier = 0.25
        mode = 'EMERGENCY'
    elif drawdown_pct >= 10:
        dd_multiplier = 0.5
        mode = 'CAUTIOUS'
    elif drawdown_pct >= 5:
        dd_multiplier = 0.75
        mode = 'REDUCED'
    else:
        dd_multiplier = 1.0
        mode = 'NORMAL'
    
    if consecutive_losses >= 4:
        streak_multiplier = 0.25
        mode = 'EMERGENCY'
    elif consecutive_losses >= 3:
        streak_multiplier = 0.5
    elif consecutive_losses >= 2:
        streak_multiplier = 0.75
    elif consecutive_wins >= 5:
        streak_multiplier = 1.25
    elif consecutive_wins >= 3:
        streak_multiplier = 1.1
    else:
        streak_multiplier = 1.0
    
    if dd_multiplier < 1 or streak_multiplier < 1:
        combined_multiplier = min(dd_multiplier, streak_multiplier)
    else:
        combined_multiplier = max(dd_multiplier, streak_multiplier)
    
    adjusted_risk = base_risk * combined_multiplier
    adjusted_risk = float(np.clip(adjusted_risk, 0.002, 0.03))
    
    return {
        'base_risk': base_risk,
        'adjusted_risk': adjusted_risk,
        'multiplier': combined_multiplier,
        'drawdown_pct': round(drawdown_pct, 2),
        'mode': mode,
        'dd_multiplier': dd_multiplier,
        'streak_multiplier': streak_multiplier,
        'message': f'{mode}: {adjusted_risk:.1%} risk (DD:{drawdown_pct:.1f}%, L:{consecutive_losses}/W:{consecutive_wins})'
    }


def multi_timeframe_confirmation(df_5m: pd.DataFrame, df_15m: pd.DataFrame = None, df_1h: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Multi-Timeframe Analysis - confirm signals across timeframes.
    """
    result = {
        '5m_trend': 'neutral',
        '15m_trend': 'neutral',
        '1h_trend': 'neutral',
        'alignment': 0,
        'confirmation': False,
        'message': ''
    }
    
    if len(df_5m) >= 40:
        sma15 = df_5m['close'].rolling(15).mean().iloc[-1]
        sma40 = df_5m['close'].rolling(40).mean().iloc[-1]
        result['5m_trend'] = 'bullish' if sma15 > sma40 else 'bearish'
    
    if df_15m is not None and len(df_15m) >= 20:
        sma10 = df_15m['close'].rolling(10).mean().iloc[-1]
        sma20 = df_15m['close'].rolling(20).mean().iloc[-1]
        result['15m_trend'] = 'bullish' if sma10 > sma20 else 'bearish'
    
    if df_1h is not None and len(df_1h) >= 20:
        sma10 = df_1h['close'].rolling(10).mean().iloc[-1]
        sma20 = df_1h['close'].rolling(20).mean().iloc[-1]
        result['1h_trend'] = 'bullish' if sma10 > sma20 else 'bearish'
    
    trends = []
    if result['5m_trend'] != 'neutral':
        trends.append(1 if result['5m_trend'] == 'bullish' else -1)
    if result['15m_trend'] != 'neutral':
        trends.append(1 if result['15m_trend'] == 'bullish' else -1)
    if result['1h_trend'] != 'neutral':
        trends.append(1 if result['1h_trend'] == 'bullish' else -1)
    
    if trends:
        result['alignment'] = sum(trends) / len(trends)
    
    bullish_count = sum(1 for t in trends if t == 1)
    bearish_count = sum(1 for t in trends if t == -1)
    
    if bullish_count >= 2:
        result['confirmation'] = True
        result['confirmed_direction'] = 'bullish'
        result['message'] = f"✅ BULLISH: {bullish_count}/{len(trends)} TFs align"
    elif bearish_count >= 2:
        result['confirmation'] = True
        result['confirmed_direction'] = 'bearish'
        result['message'] = f"✅ BEARISH: {bearish_count}/{len(trends)} TFs align"
    else:
        result['confirmation'] = False
        result['confirmed_direction'] = 'mixed'
        result['message'] = f"⚠️ MIXED: 5m={result['5m_trend']}, 15m={result['15m_trend']}, 1h={result['1h_trend']}"
    
    return result


def get_intelligence_summary(
    df: pd.DataFrame,
    signal: int,
    btc_analysis: Dict[str, Any] = None,
    mtf_analysis: Dict[str, Any] = None,
    pattern: Dict[str, Any] = None,
    drawdown_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Combine all intelligence into a summary.
    """
    summary = {
        'signal': 'LONG' if signal == 1 else ('SHORT' if signal == -1 else 'NONE'),
        'conflicts': [],
        'confirmations': [],
        'risk_mode': 'NORMAL',
        'trade_allowed': True,
        'confidence_adjustment': 1.0
    }
    
    if pattern and pattern.get('pattern'):
        if pattern['bullish'] == (signal == 1):
            summary['confirmations'].append(f"Pattern: {pattern['pattern']}")
            summary['confidence_adjustment'] *= (1 + pattern['strength'] * 0.1)
        elif pattern['bullish'] is not None and pattern['bullish'] != (signal == 1):
            summary['conflicts'].append(f"Pattern conflict: {pattern['pattern']}")
            summary['confidence_adjustment'] *= 0.9
    
    if drawdown_info:
        summary['risk_mode'] = drawdown_info['mode']
        if drawdown_info['mode'] == 'EMERGENCY':
            summary['confidence_adjustment'] *= 0.5
        elif drawdown_info['mode'] == 'CAUTIOUS':
            summary['confidence_adjustment'] *= 0.75
    
    summary['confidence_adjustment'] = float(np.clip(summary['confidence_adjustment'], 0.3, 1.5))
    
    return summary


# ============== MACHINE LEARNING REGIME CLASSIFIER ==============

# Global cache for BTC data (shared across calls to reduce API hits)
_btc_cache = {'data': None, 'timestamp': 0}


def _fetch_btc_data_sync() -> Optional[pd.DataFrame]:
    """Fetch BTC/USDT data for correlation analysis (synchronous for ML).
    
    Uses Bybit for consistency with main trading exchange.
    """
    import time
    global _btc_cache
    
    # Cache for 5 minutes
    if _btc_cache['data'] is not None and (time.time() - _btc_cache['timestamp']) < 300:
        return _btc_cache['data']
    
    try:
        import ccxt
        # Use Bybit for consistency with main trading exchange (not MEXC)
        exchange = ccxt.bybit({'enableRateLimit': True})
        ohlcv = exchange.fetch_ohlcv('BTC/USDT:USDT', '5m', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        _btc_cache['data'] = df
        _btc_cache['timestamp'] = time.time()
        return df
    except Exception as e:
        logger.debug(f"Could not fetch BTC data from Bybit: {e}")
        return None


def calculate_volume_features(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate volume-based features."""
    try:
        volume = df['volume'] if 'volume' in df.columns else df.get('Volume', pd.Series([1]))
        if len(volume) < 20:
            return {'volume_ratio': 1.0, 'volume_trend': 0.0, 'volume_spike': 0.0}
        
        # Volume ratio: current vs 20-period average
        vol_ma = volume.rolling(20).mean()
        volume_ratio = volume.iloc[-1] / vol_ma.iloc[-1] if vol_ma.iloc[-1] > 0 else 1.0
        
        # Volume trend: slope of volume MA
        vol_ma_recent = vol_ma.tail(10)
        if len(vol_ma_recent) >= 2:
            volume_trend = (vol_ma_recent.iloc[-1] - vol_ma_recent.iloc[0]) / vol_ma_recent.iloc[0] * 100
        else:
            volume_trend = 0.0
        
        # Volume spike: max volume in last 5 bars vs average
        vol_max_5 = volume.tail(5).max()
        volume_spike = vol_max_5 / vol_ma.iloc[-1] if vol_ma.iloc[-1] > 0 else 1.0
        
        return {
            'volume_ratio': float(np.clip(volume_ratio, 0, 10)),
            'volume_trend': float(np.clip(volume_trend, -100, 100)),
            'volume_spike': float(np.clip(volume_spike, 0, 10))
        }
    except Exception as e:
        logger.debug(f"Volume feature error: {e}")
        return {'volume_ratio': 1.0, 'volume_trend': 0.0, 'volume_spike': 0.0}


def calculate_btc_correlation(df: pd.DataFrame, btc_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    """Calculate correlation with BTC price movements."""
    try:
        if btc_df is None:
            btc_df = _fetch_btc_data_sync()
        
        if btc_df is None or len(btc_df) < 20:
            return {'btc_correlation': 0.0, 'btc_beta': 1.0, 'btc_relative_strength': 0.0}
        
        closes = df['close'].values if 'close' in df.columns else df['Close'].values
        btc_closes = btc_df['close'].values
        
        # Align lengths
        min_len = min(len(closes), len(btc_closes), 50)
        asset_returns = pd.Series(closes[-min_len:]).pct_change().dropna()
        btc_returns = pd.Series(btc_closes[-min_len:]).pct_change().dropna()
        
        # Align again after pct_change
        min_len = min(len(asset_returns), len(btc_returns))
        asset_returns = asset_returns.tail(min_len)
        btc_returns = btc_returns.tail(min_len)
        
        if len(asset_returns) < 10:
            return {'btc_correlation': 0.0, 'btc_beta': 1.0, 'btc_relative_strength': 0.0}
        
        # Correlation
        correlation = asset_returns.corr(btc_returns)
        
        # Beta (sensitivity to BTC moves)
        if btc_returns.std() > 0:
            beta = asset_returns.cov(btc_returns) / btc_returns.var()
        else:
            beta = 1.0
        
        # Relative strength: asset performance vs BTC over period
        asset_perf = (closes[-1] - closes[-min_len]) / closes[-min_len] * 100
        btc_perf = (btc_closes[-1] - btc_closes[-min_len]) / btc_closes[-min_len] * 100
        relative_strength = asset_perf - btc_perf
        
        return {
            'btc_correlation': float(np.clip(correlation, -1, 1)) if not np.isnan(correlation) else 0.0,
            'btc_beta': float(np.clip(beta, 0, 5)) if not np.isnan(beta) else 1.0,
            'btc_relative_strength': float(np.clip(relative_strength, -50, 50)) if not np.isnan(relative_strength) else 0.0
        }
    except Exception as e:
        logger.debug(f"BTC correlation error: {e}")
        return {'btc_correlation': 0.0, 'btc_beta': 1.0, 'btc_relative_strength': 0.0}


def calculate_momentum_divergence(df: pd.DataFrame) -> Dict[str, float]:
    """Detect divergence between price and RSI momentum."""
    try:
        closes = df['close'] if 'close' in df.columns else df['Close']
        if len(closes) < 30:
            return {'rsi_divergence': 0.0, 'momentum_strength': 0.0}
        
        rsi = calculate_rsi(closes)
        
        # Look at last 20 bars for divergence
        price_recent = closes.tail(20)
        rsi_recent = rsi.tail(20)
        
        # Simple divergence: price making new high but RSI not (bearish) or vice versa
        price_slope = (price_recent.iloc[-1] - price_recent.iloc[0]) / price_recent.iloc[0] * 100
        rsi_slope = rsi_recent.iloc[-1] - rsi_recent.iloc[0]
        
        # Divergence score: opposite directions = strong divergence
        if price_slope > 0 and rsi_slope < 0:
            divergence = -abs(price_slope)  # Bearish divergence
        elif price_slope < 0 and rsi_slope > 0:
            divergence = abs(price_slope)   # Bullish divergence
        else:
            divergence = 0.0
        
        # Momentum strength (rate of change)
        momentum = closes.pct_change(10).iloc[-1] * 100 if len(closes) > 10 else 0.0
        
        return {
            'rsi_divergence': float(np.clip(divergence, -20, 20)),
            'momentum_strength': float(np.clip(momentum, -20, 20)) if not np.isnan(momentum) else 0.0
        }
    except Exception as e:
        logger.debug(f"Momentum divergence error: {e}")
        return {'rsi_divergence': 0.0, 'momentum_strength': 0.0}


def calculate_microstructure_features(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate market microstructure features."""
    try:
        closes = df['close'] if 'close' in df.columns else df['Close']
        highs = df['high'] if 'high' in df.columns else df['High']
        lows = df['low'] if 'low' in df.columns else df['Low']
        
        if len(closes) < 30:
            return {'volatility_clustering': 0.0, 'range_expansion': 0.0, 'trend_consistency': 0.0}
        
        # Volatility clustering: autocorrelation of squared returns
        returns = closes.pct_change().dropna()
        squared_returns = returns ** 2
        vol_clustering = squared_returns.autocorr(lag=1) if len(squared_returns) > 5 else 0.0
        
        # Range expansion: current range vs average
        ranges = highs - lows
        avg_range = ranges.rolling(20).mean()
        range_expansion = ranges.iloc[-1] / avg_range.iloc[-1] if avg_range.iloc[-1] > 0 else 1.0
        
        # Trend consistency: % of bars in trend direction
        price_changes = closes.diff().tail(20)
        up_bars = (price_changes > 0).sum()
        down_bars = (price_changes < 0).sum()
        trend_consistency = abs(up_bars - down_bars) / len(price_changes) if len(price_changes) > 0 else 0.0
        
        return {
            'volatility_clustering': float(np.clip(vol_clustering, -1, 1)) if not np.isnan(vol_clustering) else 0.0,
            'range_expansion': float(np.clip(range_expansion, 0, 5)),
            'trend_consistency': float(np.clip(trend_consistency, 0, 1))
        }
    except Exception as e:
        logger.debug(f"Microstructure error: {e}")
        return {'volatility_clustering': 0.0, 'range_expansion': 0.0, 'trend_consistency': 0.0}


class MLRegimeClassifier:
    """
    Enhanced ML-based regime classifier using Gradient Boosting.
    Now with 22 features including volume, BTC correlation, and microstructure.
    Learns from trade outcomes to predict favorable conditions.
    """
    
    # Version for tracking feature changes
    MODEL_VERSION = 2
    
    def __init__(self, model_path: str = "ml_regime_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.training_data = []
        # REDUCED: Per ML Acceleration Plan - collect data faster, train sooner
        # Old: 300 (10 months at 1 trade/day)
        # New: 50 (can start predicting sooner, gradual ramp-up of influence)
        self.min_samples_to_train = 50  # Reduced from 300 per ML Acceleration Plan
        self.min_cv_score = 0.55  # Minimum cross-validation score to use model
        self._pending_milestone = None  # For async milestone notifications
        self._cv_scores = []  # Track cross-validation scores
        
        # Enhanced feature set (22 features total)
        self.feature_names = [
            # Original 10 features
            'adx', 'hurst', 'volatility_ratio', 'rsi',
            'price_vs_sma15', 'price_vs_sma40', 'sma_spread',
            'recent_return_mean', 'recent_return_std', 'recent_max_dd',
            # Volume features (3)
            'volume_ratio', 'volume_trend', 'volume_spike',
            # BTC correlation features (3)
            'btc_correlation', 'btc_beta', 'btc_relative_strength',
            # Momentum divergence (2)
            'rsi_divergence', 'momentum_strength',
            # Microstructure features (4)
            'volatility_clustering', 'range_expansion', 'trend_consistency',
            'hour_of_day'
        ]
        self._load_model()
    
    def _load_model(self):
        import os
        if os.path.exists(self.model_path):
            try:
                import pickle
                with open(self.model_path, 'rb') as f:
                    saved = pickle.load(f)
                    
                    # Check model version for migration
                    saved_version = saved.get('version', 1)
                    
                    if saved_version < self.MODEL_VERSION:
                        # Old model with fewer features - need to reset
                        logger.warning(f"ML model upgrade: v{saved_version} -> v{self.MODEL_VERSION}. Keeping samples but retraining required.")
                        old_data = saved.get('training_data', [])
                        # We can't use old samples with different feature counts
                        # But we log how many were lost for transparency
                        logger.info(f"Discarding {len(old_data)} old samples (incompatible features). Starting fresh with enhanced model.")
                        self.model = None
                        self.scaler = None
                        self.training_data = []
                        self.is_trained = False
                    else:
                        self.model = saved.get('model')
                        self.scaler = saved.get('scaler')
                        self.training_data = saved.get('training_data', [])
                        self.is_trained = self.model is not None
                        logger.info(f"ML model loaded: {len(self.training_data)} samples, trained={self.is_trained}")
            except Exception as e:
                logger.warning(f"Could not load ML model: {e}")
    
    def _save_model(self):
        try:
            import pickle
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'training_data': self.training_data,
                    'version': self.MODEL_VERSION
                }, f)
        except Exception as e:
            logger.error(f"Could not save ML model: {e}")
    
    def extract_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract 22 features for ML prediction."""
        if len(df) < 50:
            return None
        
        try:
            from datetime import datetime
            
            closes = df['close'].values if 'close' in df.columns else df['Close'].values
            
            # === Original 10 features ===
            adx = calculate_adx(df)
            hurst = calculate_hurst_exponent(pd.Series(closes))
            vol_regime = calculate_volatility_regime(pd.Series(closes))
            volatility_ratio = vol_regime.get('volatility_ratio', 1.0)
            rsi_series = calculate_rsi(pd.Series(closes))
            rsi = rsi_series.iloc[-1] if len(rsi_series) > 0 else 50
            
            sma15 = pd.Series(closes).rolling(15).mean().iloc[-1]
            sma40 = pd.Series(closes).rolling(40).mean().iloc[-1]
            current_price = closes[-1]
            
            price_vs_sma15 = (current_price - sma15) / sma15 * 100
            price_vs_sma40 = (current_price - sma40) / sma40 * 100
            sma_spread = (sma15 - sma40) / sma40 * 100
            
            returns = pd.Series(closes).pct_change().dropna().tail(20)
            if len(returns) > 5:
                recent_return_mean = returns.mean() * 100
                recent_return_std = returns.std() * 100
                cumulative = (1 + returns).cumprod()
                recent_max_dd = (cumulative / cumulative.cummax() - 1).min() * 100
            else:
                recent_return_mean = 0
                recent_return_std = 1
                recent_max_dd = 0
            
            # === NEW: Volume features (3) ===
            vol_features = calculate_volume_features(df)
            
            # === NEW: BTC correlation features (3) ===
            btc_features = calculate_btc_correlation(df)
            
            # === NEW: Momentum divergence (2) ===
            momentum_features = calculate_momentum_divergence(df)
            
            # === NEW: Microstructure features (3) ===
            micro_features = calculate_microstructure_features(df)
            
            # === NEW: Time feature (1) ===
            # Hour of day (0-23) normalized to -1 to 1
            hour_of_day = datetime.now(timezone.utc).hour / 12.0 - 1.0  # Normalize to [-1, 1]
            
            # Build feature array (22 features total)
            features = np.array([
                # Original 10
                adx, hurst, volatility_ratio, rsi,
                price_vs_sma15, price_vs_sma40, sma_spread,
                recent_return_mean, recent_return_std, recent_max_dd,
                # Volume (3)
                vol_features['volume_ratio'],
                vol_features['volume_trend'],
                vol_features['volume_spike'],
                # BTC correlation (3)
                btc_features['btc_correlation'],
                btc_features['btc_beta'],
                btc_features['btc_relative_strength'],
                # Momentum (2)
                momentum_features['rsi_divergence'],
                momentum_features['momentum_strength'],
                # Microstructure (4)
                micro_features['volatility_clustering'],
                micro_features['range_expansion'],
                micro_features['trend_consistency'],
                hour_of_day
            ])
            
            # Replace any NaN/Inf with 0
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None
    
    def record_sample(self, df: pd.DataFrame, outcome: bool, notifier=None):
        """Record a trade sample for ML learning.
        
        Args:
            df: DataFrame with market data at entry
            outcome: True for win, False for loss
            notifier: Optional TelegramNotifier for milestone notifications
        """
        features = self.extract_features(df)
        if features is not None:
            # === PERSISTENCE: Save sample to human-readable JSON ===
            try:
                import json
                import os
                from datetime import datetime
                
                backup_file = "models/ml_samples.json"
                if not os.path.exists("models"):
                    os.makedirs("models")
                
                # Create detailed record
                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "outcome": "WIN" if outcome else "LOSS",
                    "label": 1 if outcome else 0,
                    "features": dict(zip(self.feature_names, features.tolist()))
                }
                
                # Append as JSON line (safer than rewriting whole file)
                with open(backup_file, "a") as f:
                    f.write(json.dumps(record) + "\n")
                    
            except Exception as e:
                logger.warning(f"Failed to save ML sample backup: {e}")

            self.training_data.append((features, 1 if outcome else 0))
            samples = len(self.training_data)
            logger.debug(f"ML sample recorded: outcome={'WIN' if outcome else 'LOSS'}, total={samples}")
            
            # === MILESTONE NOTIFICATIONS ===
            milestones = [10, 25, 50, 75, 100, 150, 200]
            if samples in milestones:
                milestone_msg = f"🧠 ML Milestone: {samples} samples collected!"
                if samples >= self.min_samples_to_train:
                    milestone_msg += " Model is now trained."
                else:
                    remaining = self.min_samples_to_train - samples
                    milestone_msg += f" {remaining} more needed to train."
                logger.info(milestone_msg)
                # Store milestone for later notification (async context needed)
                self._pending_milestone = milestone_msg
            
            # Auto-train when reaching minimum samples, then every 20 after
            if samples >= self.min_samples_to_train:
                if samples == self.min_samples_to_train or samples % 20 == 0:
                    logger.info(f"🧠 ML Auto-training triggered at {samples} samples...")
                    self.train()
            
            self._save_model()
    
    def train(self):
        if len(self.training_data) < self.min_samples_to_train:
            logger.info(f"ML: Need {self.min_samples_to_train - len(self.training_data)} more samples to train")
            return False
        
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            
            X = np.array([x[0] for x in self.training_data])
            y = np.array([x[1] for x in self.training_data])
            
            win_rate = y.mean()
            logger.info(f"ML Training: {len(y)} samples, {win_rate:.1%} win rate, {len(self.feature_names)} features")
            
            # Check for class imbalance
            n_wins = y.sum()
            n_losses = len(y) - n_wins
            if min(n_wins, n_losses) < 10:
                logger.warning(f"ML: Class imbalance detected (wins: {n_wins}, losses: {n_losses}). Need more diverse samples.")
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Enhanced model with regularization
            self.model = GradientBoostingClassifier(
                n_estimators=100,       # Increased from 50
                max_depth=4,            # Increased from 3
                learning_rate=0.05,     # Reduced for better generalization
                min_samples_split=10,   # Increased from 5
                min_samples_leaf=5,     # Increased from 3
                subsample=0.8,          # Stochastic gradient boosting
                max_features='sqrt',    # Feature subsampling per tree
                random_state=42,
                validation_fraction=0.1,  # NEW: Hold-out validation for early stopping
                n_iter_no_change=10      # NEW: Stop if no improvement for 10 iterations
            )
            
            # Stratified Cross-Validation (maintains class balance)
            if len(y) >= 50 and min(n_wins, n_losses) >= 10:
                try:
                    cv_folds = min(5, min(n_wins, n_losses))  # Ensure each fold has both classes
                    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(self.model, X_scaled, y, cv=skf, scoring='accuracy')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    
                    logger.info(f"ML Stratified CV ({cv_folds}-fold): {cv_mean:.1%} (+/- {cv_std*2:.1%})")
                    
                    # Warning if overfitting suspected
                    if cv_std > 0.15:
                        logger.warning(f"ML: High variance in CV scores ({cv_std:.1%}). Model may be overfitting.")
                    
                    # Warning if poor performance
                    if cv_mean < 0.55:
                        logger.warning(f"ML: Low CV accuracy ({cv_mean:.1%}). Model has weak predictive power.")
                    
                except ValueError as e:
                    logger.debug(f"ML CV skipped: {e}")
            
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # Training set accuracy
            train_acc = self.model.score(X_scaled, y)
            logger.info(f"ML Training accuracy: {train_acc:.1%}")
            
            # Log top 5 features
            importances = dict(zip(self.feature_names, self.model.feature_importances_))
            top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"ML Top features: {', '.join([f'{k}={v:.2f}' for k,v in top_features])}")
            
            self._save_model()
            return True
            
        except ImportError:
            logger.warning("scikit-learn not installed. Run: pip install scikit-learn")
            return False
        except Exception as e:
            logger.error(f"ML training error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def predict(self, df: pd.DataFrame) -> Dict[str, Any]:
        result = {
            'ml_score': 0.5,
            'ml_signal': 'NEUTRAL',
            'is_trained': self.is_trained,
            'confidence': 0,
            'samples': len(self.training_data)
        }
        
        if not self.is_trained or self.model is None:
            result['message'] = f"Need {self.min_samples_to_train - len(self.training_data)} more trades to train"
            return result
        
        features = self.extract_features(df)
        if features is None:
            result['message'] = "Insufficient data for prediction"
            return result
        
        try:
            X = self.scaler.transform(features.reshape(1, -1))
            proba = self.model.predict_proba(X)[0]
            ml_score = proba[1] if len(proba) > 1 else proba[0]
            confidence = abs(ml_score - 0.5) * 2
            
            if ml_score >= 0.65:
                ml_signal = 'FAVORABLE'
            elif ml_score <= 0.35:
                ml_signal = 'UNFAVORABLE'
            else:
                ml_signal = 'NEUTRAL'
            
            result.update({
                'ml_score': round(ml_score, 3),
                'ml_signal': ml_signal,
                'confidence': round(confidence, 2),
                'message': f"ML: {ml_signal} ({ml_score:.0%})"
            })
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            result['message'] = f"Prediction error: {e}"
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {
            'total_samples': len(self.training_data),
            'is_trained': self.is_trained,
            'min_samples_needed': self.min_samples_to_train,
            'samples_until_training': max(0, self.min_samples_to_train - len(self.training_data)),
            'model_version': self.MODEL_VERSION,
            'num_features': len(self.feature_names),
            'feature_categories': {
                'original': 10,
                'volume': 3,
                'btc_correlation': 3,
                'momentum': 2,
                'microstructure': 4
            }
        }
        
        if self.training_data:
            outcomes = [x[1] for x in self.training_data]
            stats['historical_win_rate'] = sum(outcomes) / len(outcomes)
            stats['wins'] = sum(outcomes)
            stats['losses'] = len(outcomes) - sum(outcomes)
        
        if self.is_trained and self.model is not None:
            try:
                importances = dict(zip(self.feature_names, self.model.feature_importances_))
                stats['top_features'] = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
                stats['all_features'] = self.feature_names
            except Exception:
                pass  # Non-critical: feature importance extraction
        
        return stats


# Global ML classifier instance
_ml_classifier = None

def get_ml_classifier() -> MLRegimeClassifier:
    global _ml_classifier
    if _ml_classifier is None:
        _ml_classifier = MLRegimeClassifier()
    return _ml_classifier


def ml_predict_regime(df: pd.DataFrame) -> Dict[str, Any]:
    classifier = get_ml_classifier()
    return classifier.predict(df)


def ml_record_trade(df: pd.DataFrame, won: bool):
    classifier = get_ml_classifier()
    classifier.record_sample(df, won)

# ═══════════════════════════════════════════════════════════════════════════
# ⭐ PhD-LEVEL STATISTICAL ENHANCEMENTS ⭐
# ═══════════════════════════════════════════════════════════════════════════

def calculate_hypothesis_test(returns: pd.Series, null_mean: float = 0) -> Dict[str, Any]:
    """
    Perform t-test to verify if strategy has statistically significant edge.
    
    H₀ (Null): Mean return = 0 (no edge)
    H₁ (Alt): Mean return ≠ 0 (has edge)
    
    Returns p-value: if < 0.05, we reject H₀ with 95% confidence
    """
    if len(returns) < 10:
        return {'t_statistic': 0, 'p_value': 1.0, 'significant': False, 'confidence': 0.0}
    
    returns_clean = returns.dropna()
    t_stat, p_value = scipy_stats.ttest_1samp(returns_clean, null_mean)
    
    mean_return = returns_clean.mean()
    std_error = returns_clean.sem()
    ci_lower = mean_return - 1.96 * std_error
    ci_upper = mean_return + 1.96 * std_error
    
    significant = p_value < 0.05
    confidence = 1 - p_value
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': significant,
        'confidence': float(confidence),
        'mean_return': float(mean_return),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'interpretation': f"Mean return {mean_return:.4f}% with 95% CI [{ci_lower:.4f}%, {ci_upper:.4f}%]"
    }


def calculate_garch_volatility(returns: pd.Series, p: int = 1, q: int = 1) -> Dict[str, Any]:
    """
    Calculate GARCH(p,q) conditional volatility.
    
    Better than simple volatility ratio because it models:
    - Volatility clustering (high vol → high vol)
    - Mean reversion in volatility
    - Temporal dynamics
    
    GARCH(1,1) is most commonly used in finance
    """
    if len(returns) < 50:
        return {'status': 'insufficient_data', 'garch_vol': float(returns.std())}
    
    try:
        returns_pct = returns.dropna()
        model = arch_model(returns_pct * 100, vol='Garch', p=p, q=q, rescale=False)
        res = model.fit(disp='off')
        
        # Get conditional volatility (annualized)
        # Crypto: 365 days × 24 hours × 4 bars/hour for 15-min candles
        conditional_vol = res.conditional_volatility.iloc[-1] / 100 * np.sqrt(365 * 24 * 4)
        
        # Get volatility forecast using warnings filter
        try:
            forecast = res.get_forecast(horizon=1)
            forecast_vol = forecast.variance.values[-1, 0] ** 0.5 / 100 * np.sqrt(365 * 24 * 4)
        except Exception:
            # Fallback: use next period conditional vol estimate
            forecast_vol = conditional_vol
        
        return {
            'status': 'success',
            'current_vol': float(conditional_vol),
            'forecast_vol': float(forecast_vol),
            'vol_trend': 'increasing' if forecast_vol > conditional_vol else 'decreasing',
            'vol_change_pct': float((forecast_vol / conditional_vol - 1) * 100) if conditional_vol > 0 else 0,
            'model_aic': float(res.aic),
            'half_life': float(calculate_garch_halflife(res.params))
        }
    except Exception as e:
        logger.warning(f"GARCH model failed: {e}")
        return {'status': 'error', 'garch_vol': float(returns.std())}


def calculate_garch_halflife(params) -> float:
    """Calculate half-life of volatility mean reversion from GARCH parameters."""
    try:
        alpha = params.get('alpha[1]', 0.05)
        beta = params.get('beta[1]', 0.90)
        
        if alpha + beta >= 1:
            return float('inf')  # Non-stationary
        
        decay = alpha + beta
        half_life = np.log(0.5) / np.log(decay)
        return float(max(0, half_life))
    except Exception:
        return 0.0


def calculate_regime_hmm(returns: pd.Series, n_regimes: int = 3) -> Dict[str, Any]:
    """
    Hidden Markov Model for regime detection.
    
    Automatically detects regimes (trending, mean-reverting, choppy)
    without hardcoded thresholds.
    
    Better than Hurst + ADX because it:
    - Captures regime persistence
    - Probabilistic transitions
    - Adaptive to market changes
    """
    if len(returns) < 100:
        return {'status': 'insufficient_data', 'regime': 'unknown'}
    
    try:
        returns_clean = returns.dropna().values.reshape(-1, 1)
        
        model = MarkovRegression(returns_clean, k_regimes=n_regimes, trend='c')
        res = model.fit(disp=False)
        
        # Get current regime from smoothed state probabilities
        current_regime = res.smoothed_marginal_probabilities[-1].argmax()
        regime_names = ['CALM', 'NORMAL', 'VOLATILE'][:n_regimes]
        
        regime_probs = res.smoothed_marginal_probabilities[-1]
        confidence = float(regime_probs[current_regime])
        
        # Get transition matrix - handle both naming conventions
        if hasattr(res, 'markov_chain'):
            transition_matrix = res.markov_chain.transition_matrix
        elif hasattr(res, '_markov_chain'):
            transition_matrix = res._markov_chain.transition_matrix
        else:
            # Fallback: return basic result without transition matrix
            return {
                'status': 'success',
                'regime': regime_names[current_regime],
                'regime_number': int(current_regime),
                'confidence': float(confidence),
                'persistence': 0.5,
                'regime_probs': {name: float(prob) for name, prob in zip(regime_names, regime_probs)},
                'loglik': float(res.llf)
            }
        
        persistence = float(transition_matrix[current_regime, current_regime])
        
        return {
            'status': 'success',
            'regime': regime_names[current_regime],
            'regime_number': int(current_regime),
            'confidence': float(confidence),
            'persistence': float(persistence),  # P(stay in same regime)
            'regime_probs': {name: float(prob) for name, prob in zip(regime_names, regime_probs)},
            'transition_matrix': transition_matrix.tolist(),
            'loglik': float(res.llf)
        }
    except Exception as e:
        logger.warning(f"HMM model failed: {e}")
        return {'status': 'error', 'regime': 'unknown'}


def calculate_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> Dict[str, float]:
    """
    Calculate confidence interval for any metric.
    
    Returns upper/lower bounds where we're (confidence)% sure the true value lies
    """
    if len(values) < 5:
        return {'lower': float(values.min()), 'upper': float(values.max()), 'mean': float(values.mean())}
    
    mean = values.mean()
    stderr = scipy_stats.sem(values)
    alpha = 1 - confidence
    z_score = scipy_stats.t.ppf(1 - alpha/2, len(values) - 1)
    
    margin = z_score * stderr
    
    return {
        'mean': float(mean),
        'lower': float(mean - margin),
        'upper': float(mean + margin),
        'margin': float(margin),
        'confidence_level': confidence
    }


def test_signal_significance(signal_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, Any]:
    """
    Test if signal returns are significantly different from benchmark (market).
    
    Uses Mann-Whitney U test (non-parametric, robust to outliers)
    """
    if len(signal_returns) < 10 or len(benchmark_returns) < 10:
        return {'significant': False, 'p_value': 1.0}
    
    signal_clean = signal_returns.dropna()
    benchmark_clean = benchmark_returns.dropna()
    
    # Mann-Whitney U test
    statistic, p_value = scipy_stats.mannwhitneyu(signal_clean, benchmark_clean, alternative='two-sided')
    
    # Effect size (Cohen's d)
    cohens_d = (signal_clean.mean() - benchmark_clean.mean()) / np.sqrt((signal_clean.std()**2 + benchmark_clean.std()**2) / 2)
    
    return {
        'significant': p_value < 0.05,
        'p_value': float(p_value),
        'statistic': float(statistic),
        'cohens_d': float(cohens_d),
        'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small',
        'signal_mean': float(signal_clean.mean()),
        'benchmark_mean': float(benchmark_clean.mean()),
        'interpretation': f"Signal mean: {signal_clean.mean():.4f}%, Benchmark: {benchmark_clean.mean():.4f}% (p={p_value:.4f})"
    }


def calculate_walk_forward_performance(df: pd.DataFrame, window: int = 100, step: int = 20) -> Dict[str, Any]:
    """
    Walk-forward out-of-sample testing.
    
    Prevents overfitting by:
    1. Training on past data
    2. Testing on future data
    3. Rolling window forward
    
    More realistic than in-sample backtesting
    """
    if len(df) < window + step:
        return {'status': 'insufficient_data'}
    
    try:
        out_of_sample_returns = []
        sharpe_ratios = []
        win_rates = []
        
        for i in range(0, len(df) - window - step, step):
            train_period = df.iloc[i:i+window]
            test_period = df.iloc[i+window:i+window+step]
            
            # Train on historical data (could fit any model here)
            train_returns = train_period.get('returns', pd.Series([0]))
            if len(train_returns) > 0:
                mean_return = train_returns.mean()
                std_return = train_returns.std()
            else:
                mean_return, std_return = 0, 0
            
            # Test on forward data
            if 'returns' in test_period.columns:
                test_returns = test_period['returns']
                out_of_sample_returns.extend(test_returns)
                
                if len(test_returns) > 0:
                    sharpe = test_returns.mean() / test_returns.std() * np.sqrt(252 * 24) if test_returns.std() > 0 else 0
                    sharpe_ratios.append(sharpe)
                    win_rates.append((test_returns > 0).sum() / len(test_returns))
        
        if not out_of_sample_returns:
            return {'status': 'no_data'}
        
        oos_returns = np.array(out_of_sample_returns)
        
        return {
            'status': 'success',
            'mean_oos_return': float(oos_returns.mean()),
            'std_oos_return': float(oos_returns.std()),
            'mean_sharpe': float(np.mean(sharpe_ratios)) if sharpe_ratios else 0,
            'mean_win_rate': float(np.mean(win_rates)) if win_rates else 0,
            'profit_factor': abs(oos_returns[oos_returns > 0].sum() / oos_returns[oos_returns < 0].sum()) if len(oos_returns[oos_returns < 0]) > 0 else 0,
            'rolling_windows': len(sharpe_ratios),
            'windows_profitable': sum(1 for s in sharpe_ratios if s > 0)
        }
    except Exception as e:
        logger.warning(f"Walk-forward analysis failed: {e}")
        return {'status': 'error'}


# ============== ADVANCED ENHANCEMENTS (PhD-Level) ==============

def calculate_volume_profile(df: pd.DataFrame, num_bins: int = 24) -> Dict[str, Any]:
    """
    Calculate Volume Profile Analysis with TPO (Time-Price-Opportunity) style metrics.
    
    Volume Profile shows volume distribution across price levels, revealing:
    - Point of Control (POC): Price level with highest volume - strongest support/resistance
    - Value Area (VA): Range containing 70% of volume - fair value zone
    - High Volume Nodes (HVN): Price levels with concentrated activity
    - Low Volume Nodes (LVN): Price gaps where price may move quickly
    
    Mathematical Foundation:
        POC = argmax_p(V(p)) where V(p) = sum of volume at price level p
        VA_High/Low = bounds containing 70% of total volume centered on POC
    
    Returns detailed volume profile metrics for trade decision support.
    """
    try:
        closes = df['close'] if 'close' in df.columns else df['Close']
        highs = df['high'] if 'high' in df.columns else df['High']
        lows = df['low'] if 'low' in df.columns else df['Low']
        volumes = df['volume'] if 'volume' in df.columns else None
        
        if volumes is None or len(closes) < 20:
            return {'status': 'insufficient_data'}
        
        # Define price range
        price_min = lows.min()
        price_max = highs.max()
        price_range = price_max - price_min
        
        if price_range <= 0:
            return {'status': 'no_price_range'}
        
        # Create price bins
        bin_size = price_range / num_bins
        bins = np.linspace(price_min, price_max, num_bins + 1)
        
        # Build volume profile: distribute volume across price levels touched by each bar
        volume_at_level = np.zeros(num_bins)
        tpo_count = np.zeros(num_bins)  # Time-Price-Opportunity count
        
        for idx in range(len(df)):
            bar_high = highs.iloc[idx]
            bar_low = lows.iloc[idx]
            bar_vol = volumes.iloc[idx]
            bar_close = closes.iloc[idx]
            
            # Distribute volume proportionally across price levels
            for i in range(num_bins):
                level_low = bins[i]
                level_high = bins[i + 1]
                
                # Check if bar touched this price level
                if bar_low <= level_high and bar_high >= level_low:
                    # Calculate overlap proportion
                    overlap_low = max(bar_low, level_low)
                    overlap_high = min(bar_high, level_high)
                    bar_range = bar_high - bar_low if bar_high > bar_low else bin_size
                    overlap_pct = (overlap_high - overlap_low) / bar_range
                    
                    volume_at_level[i] += bar_vol * overlap_pct
                    tpo_count[i] += 1  # Count TPO
        
        total_volume = volume_at_level.sum()
        if total_volume == 0:
            return {'status': 'no_volume'}
        
        # Find Point of Control (POC) - price level with maximum volume
        poc_idx = np.argmax(volume_at_level)
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        poc_volume_pct = volume_at_level[poc_idx] / total_volume * 100
        
        # Calculate Value Area (70% of volume centered on POC)
        va_target = total_volume * 0.70
        va_volume = volume_at_level[poc_idx]
        va_low_idx = poc_idx
        va_high_idx = poc_idx
        
        while va_volume < va_target and (va_low_idx > 0 or va_high_idx < num_bins - 1):
            # Expand in direction of more volume
            vol_below = volume_at_level[va_low_idx - 1] if va_low_idx > 0 else 0
            vol_above = volume_at_level[va_high_idx + 1] if va_high_idx < num_bins - 1 else 0
            
            if vol_below >= vol_above and va_low_idx > 0:
                va_low_idx -= 1
                va_volume += volume_at_level[va_low_idx]
            elif va_high_idx < num_bins - 1:
                va_high_idx += 1
                va_volume += volume_at_level[va_high_idx]
            elif va_low_idx > 0:
                va_low_idx -= 1
                va_volume += volume_at_level[va_low_idx]
            else:
                break
        
        va_low = bins[va_low_idx]
        va_high = bins[va_high_idx + 1]
        
        # Find High Volume Nodes (above average) and Low Volume Nodes
        avg_volume = volume_at_level.mean()
        std_volume = volume_at_level.std()
        hvn_threshold = avg_volume + 0.5 * std_volume
        lvn_threshold = avg_volume - 0.5 * std_volume
        
        hvn_levels = []
        lvn_levels = []
        
        for i in range(num_bins):
            level_price = (bins[i] + bins[i + 1]) / 2
            if volume_at_level[i] > hvn_threshold:
                hvn_levels.append({
                    'price': float(level_price),
                    'volume_pct': float(volume_at_level[i] / total_volume * 100)
                })
            elif volume_at_level[i] < lvn_threshold and volume_at_level[i] > 0:
                lvn_levels.append({
                    'price': float(level_price),
                    'volume_pct': float(volume_at_level[i] / total_volume * 100)
                })
        
        # Current price position relative to POC and Value Area
        current_price = closes.iloc[-1]
        price_vs_poc = (current_price - poc_price) / poc_price * 100
        in_value_area = va_low <= current_price <= va_high
        
        # Distance to nearest support/resistance (HVN levels)
        hvn_prices = [h['price'] for h in hvn_levels]
        nearest_support = max([p for p in hvn_prices if p < current_price], default=va_low)
        nearest_resistance = min([p for p in hvn_prices if p > current_price], default=va_high)
        
        return {
            'status': 'success',
            'poc_price': float(poc_price),
            'poc_volume_pct': float(poc_volume_pct),
            'value_area_high': float(va_high),
            'value_area_low': float(va_low),
            'value_area_pct': float((va_high - va_low) / current_price * 100),
            'current_vs_poc_pct': float(price_vs_poc),
            'in_value_area': bool(in_value_area),
            'hvn_levels': hvn_levels[:5],  # Top 5 HVN
            'lvn_levels': lvn_levels[:5],  # Top 5 LVN
            'nearest_support': float(nearest_support),
            'nearest_resistance': float(nearest_resistance),
            'support_distance_pct': float((current_price - nearest_support) / current_price * 100),
            'resistance_distance_pct': float((nearest_resistance - current_price) / current_price * 100),
            'tpo_poc_count': int(tpo_count[poc_idx]),
            'volume_profile': volume_at_level.tolist()  # Full profile for visualization
        }
        
    except Exception as e:
        logger.warning(f"Volume Profile calculation failed: {e}")
        return {'status': 'error', 'error': str(e)}


def calculate_kalman_momentum(prices: pd.Series, process_noise: float = 0.01, 
                              measurement_noise: float = 0.1) -> Dict[str, Any]:
    """
    Kalman Filter smoothed momentum estimation.
    
    The Kalman Filter provides optimal state estimation for noisy time series,
    separating true momentum signal from market noise.
    
    Mathematical Model:
        State: x_t = [price, velocity, acceleration]
        
        Prediction:
            x_t|t-1 = F * x_t-1|t-1
            P_t|t-1 = F * P_t-1|t-1 * F' + Q
        
        Update:
            K_t = P_t|t-1 * H' * (H * P_t|t-1 * H' + R)^(-1)
            x_t|t = x_t|t-1 + K_t * (z_t - H * x_t|t-1)
            P_t|t = (I - K_t * H) * P_t|t-1
    
    Returns:
        - Smoothed momentum (velocity)
        - Momentum acceleration
        - Confidence bounds based on Kalman covariance
        - Signal strength indicator
    """
    try:
        if len(prices) < 20:
            return {'status': 'insufficient_data'}
        
        prices_arr = prices.values.astype(float)
        n = len(prices_arr)
        
        # State dimension: [price, velocity (momentum), acceleration]
        dim_state = 3
        dim_obs = 1
        
        # State transition matrix F (constant acceleration model)
        dt = 1.0  # Time step
        F = np.array([
            [1, dt, 0.5 * dt**2],
            [0, 1, dt],
            [0, 0, 1]
        ])
        
        # Observation matrix H (we observe price only)
        H = np.array([[1, 0, 0]])
        
        # Process noise covariance Q
        q = process_noise
        Q = np.array([
            [dt**4/4, dt**3/2, dt**2/2],
            [dt**3/2, dt**2, dt],
            [dt**2/2, dt, 1]
        ]) * q
        
        # Measurement noise covariance R
        R = np.array([[measurement_noise]])
        
        # Initial state estimate
        x = np.array([prices_arr[0], 0.0, 0.0])
        
        # Initial covariance
        P = np.eye(dim_state) * 1.0
        
        # Storage for filtered estimates
        filtered_prices = np.zeros(n)
        filtered_velocity = np.zeros(n)
        filtered_acceleration = np.zeros(n)
        velocity_variance = np.zeros(n)
        
        for i in range(n):
            # Prediction step
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            
            # Update step
            z = np.array([prices_arr[i]])
            y = z - H @ x_pred  # Innovation
            S = H @ P_pred @ H.T + R  # Innovation covariance
            K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
            
            x = x_pred + K.flatten() * y[0]
            P = (np.eye(dim_state) - K @ H) @ P_pred
            
            # Store estimates
            filtered_prices[i] = x[0]
            filtered_velocity[i] = x[1]
            filtered_acceleration[i] = x[2]
            velocity_variance[i] = P[1, 1]
        
        # Analyze momentum characteristics
        current_momentum = filtered_velocity[-1]
        momentum_std = np.sqrt(velocity_variance[-1])
        current_acceleration = filtered_acceleration[-1]
        
        # Momentum strength (z-score of current momentum vs historical)
        momentum_history = filtered_velocity[max(0, n-50):]
        if len(momentum_history) > 5 and momentum_history.std() > 0:
            momentum_zscore = (current_momentum - momentum_history.mean()) / momentum_history.std()
        else:
            momentum_zscore = 0
        
        # Momentum signal: direction * confidence
        confidence = 1 - min(1, momentum_std / (abs(current_momentum) + 1e-8))
        signal_strength = abs(momentum_zscore) * confidence
        
        # Momentum trend (is it strengthening or weakening?)
        momentum_slope = filtered_velocity[-1] - filtered_velocity[-5] if n >= 5 else 0
        
        return {
            'status': 'success',
            'kalman_momentum': float(current_momentum),
            'kalman_acceleration': float(current_acceleration),
            'momentum_std': float(momentum_std),
            'momentum_zscore': float(momentum_zscore),
            'momentum_confidence': float(confidence),
            'signal_strength': float(signal_strength),
            'momentum_trend': 'strengthening' if momentum_slope > 0 else 'weakening',
            'momentum_slope': float(momentum_slope),
            'filtered_price': float(filtered_prices[-1]),
            'price_deviation': float(prices_arr[-1] - filtered_prices[-1]),
            # For charting
            'filtered_prices': filtered_prices.tolist(),
            'filtered_velocity': filtered_velocity.tolist()
        }
        
    except Exception as e:
        logger.warning(f"Kalman momentum calculation failed: {e}")
        return {'status': 'error', 'error': str(e)}


def calculate_rsi_divergence_mtf(df: pd.DataFrame, 
                                  df_higher: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Multi-timeframe RSI divergence detection.
    
    Divergence Types:
    - Regular Bullish: Price makes lower low, RSI makes higher low → Potential reversal up
    - Regular Bearish: Price makes higher high, RSI makes lower high → Potential reversal down
    - Hidden Bullish: Price makes higher low, RSI makes lower low → Continuation up
    - Hidden Bearish: Price makes lower high, RSI makes higher high → Continuation down
    
    Multi-timeframe confirmation increases signal reliability:
    - Divergence on both timeframes = stronger signal
    - Only lower TF = weaker signal, may need more confirmation
    
    Mathematical Definition:
        For n-bar lookback window:
        Regular Bullish: ΔP < 0 AND ΔRSI > 0 (price down, RSI up)
        Regular Bearish: ΔP > 0 AND ΔRSI < 0 (price up, RSI down)
    """
    try:
        closes = df['close'] if 'close' in df.columns else df['Close']
        
        if len(closes) < 30:
            return {'status': 'insufficient_data'}
        
        # Calculate RSI
        rsi = calculate_rsi(closes)
        
        def find_divergences(price_series: pd.Series, rsi_series: pd.Series, lookback: int = 20) -> Dict:
            """Find divergences in given series."""
            divergences = {
                'regular_bullish': False,
                'regular_bearish': False,
                'hidden_bullish': False,
                'hidden_bearish': False,
                'strength': 0.0,
                'bars_since': 0
            }
            
            if len(price_series) < lookback + 5:
                return divergences
            
            # Find recent swing points
            recent_prices = price_series.tail(lookback)
            recent_rsi = rsi_series.tail(lookback)
            
            # Compare first half vs second half for swing point detection
            mid = lookback // 2
            
            first_half_price = recent_prices.iloc[:mid]
            second_half_price = recent_prices.iloc[mid:]
            first_half_rsi = recent_rsi.iloc[:mid]
            second_half_rsi = recent_rsi.iloc[mid:]
            
            # Price and RSI changes
            price_low1, price_low2 = first_half_price.min(), second_half_price.min()
            price_high1, price_high2 = first_half_price.max(), second_half_price.max()
            rsi_at_price_low1 = first_half_rsi.iloc[first_half_price.idxmin() - recent_prices.index[0]]
            rsi_at_price_low2 = second_half_rsi.iloc[second_half_price.idxmin() - recent_prices.index[mid]]
            rsi_at_price_high1 = first_half_rsi.iloc[first_half_price.idxmax() - recent_prices.index[0]]
            rsi_at_price_high2 = second_half_rsi.iloc[second_half_price.idxmax() - recent_prices.index[mid]]
            
            # Regular Bullish Divergence: Price lower low, RSI higher low
            if price_low2 < price_low1 and rsi_at_price_low2 > rsi_at_price_low1:
                divergences['regular_bullish'] = True
                divergences['strength'] = abs(rsi_at_price_low2 - rsi_at_price_low1)
            
            # Regular Bearish Divergence: Price higher high, RSI lower high
            if price_high2 > price_high1 and rsi_at_price_high2 < rsi_at_price_high1:
                divergences['regular_bearish'] = True
                divergences['strength'] = abs(rsi_at_price_high2 - rsi_at_price_high1)
            
            # Hidden Bullish Divergence: Price higher low, RSI lower low (continuation)
            if price_low2 > price_low1 and rsi_at_price_low2 < rsi_at_price_low1:
                divergences['hidden_bullish'] = True
                divergences['strength'] = abs(rsi_at_price_low2 - rsi_at_price_low1)
            
            # Hidden Bearish Divergence: Price lower high, RSI higher high (continuation)
            if price_high2 < price_high1 and rsi_at_price_high2 > rsi_at_price_high1:
                divergences['hidden_bearish'] = True
                divergences['strength'] = abs(rsi_at_price_high2 - rsi_at_price_high1)
            
            return divergences
        
        # Find divergences on primary timeframe
        primary_div = find_divergences(closes, rsi)
        
        # Multi-timeframe analysis
        mtf_confirmation = False
        higher_tf_divergence = None
        
        if df_higher is not None and len(df_higher) >= 30:
            closes_htf = df_higher['close'] if 'close' in df_higher.columns else df_higher['Close']
            rsi_htf = calculate_rsi(closes_htf)
            higher_tf_divergence = find_divergences(closes_htf, rsi_htf)
            
            # Check for MTF confirmation
            if primary_div['regular_bullish'] and higher_tf_divergence['regular_bullish']:
                mtf_confirmation = True
            elif primary_div['regular_bearish'] and higher_tf_divergence['regular_bearish']:
                mtf_confirmation = True
            elif primary_div['hidden_bullish'] and higher_tf_divergence['hidden_bullish']:
                mtf_confirmation = True
            elif primary_div['hidden_bearish'] and higher_tf_divergence['hidden_bearish']:
                mtf_confirmation = True
        
        # Calculate overall signal
        bullish_signal = primary_div['regular_bullish'] or primary_div['hidden_bullish']
        bearish_signal = primary_div['regular_bearish'] or primary_div['hidden_bearish']
        
        signal_confidence = primary_div['strength']
        if mtf_confirmation:
            signal_confidence *= 1.5  # Boost for MTF confirmation
        
        return {
            'status': 'success',
            'regular_bullish_divergence': bool(primary_div['regular_bullish']),
            'regular_bearish_divergence': bool(primary_div['regular_bearish']),
            'hidden_bullish_divergence': bool(primary_div['hidden_bullish']),
            'hidden_bearish_divergence': bool(primary_div['hidden_bearish']),
            'divergence_strength': float(primary_div['strength']),
            'mtf_confirmation': bool(mtf_confirmation),
            'higher_tf_divergence': higher_tf_divergence,
            'bullish_signal': bool(bullish_signal),
            'bearish_signal': bool(bearish_signal),
            'signal_confidence': float(min(100, signal_confidence * 10)),
            'current_rsi': float(rsi.iloc[-1])
        }
        
    except Exception as e:
        logger.warning(f"RSI divergence MTF calculation failed: {e}")
        return {'status': 'error', 'error': str(e)}


def calculate_cointegration_test(prices1: pd.Series, prices2: pd.Series, 
                                  test_type: str = 'engle_granger') -> Dict[str, Any]:
    """
    Cointegration testing for pairs trading.
    
    Cointegration means two series share a common stochastic drift, and their
    linear combination is stationary - essential for pairs/stat-arb trading.
    
    Tests Implemented:
    1. Engle-Granger Two-Step:
       - Regress Y on X: Y = α + βX + ε
       - Test residuals ε for stationarity (ADF test)
       - Cointegrated if residuals are stationary
    
    2. Johansen Test:
       - Tests for multiple cointegrating relationships
       - More powerful but requires more data
       - Returns trace and eigenvalue statistics
    
    Trading Application:
    - If cointegrated, trade the spread: spread = Y - β*X
    - When spread > μ + k*σ: Short spread (short Y, long X)
    - When spread < μ - k*σ: Long spread (long Y, short X)
    
    Returns:
        - Cointegration status
        - Hedge ratio (β)
        - Spread statistics
        - Trading levels
    """
    try:
        from statsmodels.tsa.stattools import adfuller, coint
        
        if len(prices1) < 60 or len(prices2) < 60:
            return {'status': 'insufficient_data'}
        
        # Align series
        min_len = min(len(prices1), len(prices2))
        p1 = prices1.tail(min_len).values
        p2 = prices2.tail(min_len).values
        
        result = {
            'status': 'success',
            'test_type': test_type
        }
        
        if test_type == 'engle_granger':
            # Step 1: OLS regression Y = α + β*X
            X = np.column_stack([np.ones(len(p1)), p1])
            betas, residuals, rank, s = np.linalg.lstsq(X, p2, rcond=None)
            
            alpha = betas[0]
            hedge_ratio = betas[1]
            
            # Calculate spread
            spread = p2 - hedge_ratio * p1 - alpha
            
            # Step 2: ADF test on residuals
            adf_result = adfuller(spread, maxlag=int(np.sqrt(len(spread))))
            adf_stat = adf_result[0]
            adf_pvalue = adf_result[1]
            adf_critical = adf_result[4]
            
            # Cointegration test
            cointegrated = adf_pvalue < 0.05
            
            # Spread statistics
            spread_mean = spread.mean()
            spread_std = spread.std()
            spread_current = spread[-1]
            spread_zscore = (spread_current - spread_mean) / spread_std if spread_std > 0 else 0
            
            # Half-life of mean reversion (Ornstein-Uhlenbeck)
            spread_lag = spread[:-1]
            spread_diff = np.diff(spread)
            
            if len(spread_lag) > 10:
                X_hl = np.column_stack([np.ones(len(spread_lag)), spread_lag])
                theta, _, _, _ = np.linalg.lstsq(X_hl, spread_diff, rcond=None)
                lambda_ou = -theta[1]
                half_life = np.log(2) / lambda_ou if lambda_ou > 0 else np.inf
            else:
                half_life = np.inf
            
            # Trading signals
            entry_threshold = 2.0  # Enter at 2 sigma
            exit_threshold = 0.5   # Exit at 0.5 sigma
            
            if spread_zscore > entry_threshold:
                trade_signal = 'short_spread'  # Short Y, Long X
                trade_reason = f"Spread overbought (z={spread_zscore:.2f})"
            elif spread_zscore < -entry_threshold:
                trade_signal = 'long_spread'   # Long Y, Short X
                trade_reason = f"Spread oversold (z={spread_zscore:.2f})"
            elif abs(spread_zscore) < exit_threshold:
                trade_signal = 'close_spread'
                trade_reason = f"Spread mean-reverted (z={spread_zscore:.2f})"
            else:
                trade_signal = 'hold'
                trade_reason = f"No action (z={spread_zscore:.2f})"
            
            result.update({
                'cointegrated': bool(cointegrated),
                'adf_statistic': float(adf_stat),
                'adf_pvalue': float(adf_pvalue),
                'adf_critical_1pct': float(adf_critical['1%']),
                'adf_critical_5pct': float(adf_critical['5%']),
                'adf_critical_10pct': float(adf_critical['10%']),
                'hedge_ratio': float(hedge_ratio),
                'alpha': float(alpha),
                'spread_mean': float(spread_mean),
                'spread_std': float(spread_std),
                'spread_current': float(spread_current),
                'spread_zscore': float(spread_zscore),
                'half_life': float(min(half_life, 1000)),  # Cap at 1000
                'trade_signal': trade_signal,
                'trade_reason': trade_reason,
                'entry_long': float(spread_mean - entry_threshold * spread_std),
                'entry_short': float(spread_mean + entry_threshold * spread_std),
                'exit_level': float(spread_mean),
                'spread_series': spread.tolist()
            })
            
        elif test_type == 'johansen':
            # Johansen cointegration test
            try:
                from statsmodels.tsa.vector_ar.vecm import coint_johansen
                
                data = np.column_stack([p1, p2])
                
                # Perform Johansen test (det_order=-1: no deterministic trend)
                jres = coint_johansen(data, det_order=0, k_ar_diff=1)
                
                # Trace statistic
                trace_stat = jres.lr1
                trace_crit_95 = jres.cvt[:, 1]  # 95% critical values
                
                # Max eigenvalue statistic
                max_eig_stat = jres.lr2
                max_eig_crit_95 = jres.cvm[:, 1]
                
                # Number of cointegrating relationships
                n_coint = sum(trace_stat > trace_crit_95)
                
                # Cointegrating vector (first eigenvector if cointegrated)
                if n_coint > 0:
                    coint_vector = jres.evec[:, 0]
                    hedge_ratio = -coint_vector[1] / coint_vector[0]
                    
                    # Calculate spread
                    spread = p2 - hedge_ratio * p1
                    spread_mean = spread.mean()
                    spread_std = spread.std()
                    spread_current = spread[-1]
                    spread_zscore = (spread_current - spread_mean) / spread_std
                else:
                    hedge_ratio = 1.0
                    spread_zscore = 0.0
                    spread_mean = 0.0
                    spread_std = 1.0
                
                result.update({
                    'cointegrated': bool(n_coint > 0),
                    'n_cointegrating_relations': int(n_coint),
                    'trace_stat_r0': float(trace_stat[0]),
                    'trace_stat_r1': float(trace_stat[1]) if len(trace_stat) > 1 else 0,
                    'trace_crit_95_r0': float(trace_crit_95[0]),
                    'max_eig_stat_r0': float(max_eig_stat[0]),
                    'max_eig_crit_95_r0': float(max_eig_crit_95[0]),
                    'hedge_ratio': float(hedge_ratio),
                    'spread_zscore': float(spread_zscore),
                    'spread_mean': float(spread_mean),
                    'spread_std': float(spread_std)
                })
                
            except ImportError:
                result['error'] = 'statsmodels.tsa.vector_ar.vecm not available'
                result['cointegrated'] = False
        
        return result
        
    except Exception as e:
        logger.warning(f"Cointegration test failed: {e}")
        return {'status': 'error', 'error': str(e)}