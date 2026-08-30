import numpy as np
import pandas as pd


# =========================
# Helper functions
# =========================

def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=length, adjust=False).mean()


def wma(series: pd.Series, length: int) -> pd.Series:
    """Weighted moving average (like Pine ta.wma)."""
    length = int(length)
    if length <= 0:
        return series * np.nan
    weights = np.arange(1, length + 1, dtype=float)
    return series.rolling(length).apply(
        lambda x: np.dot(x, weights) / weights.sum(),
        raw=True
    )


def hma(series: pd.Series, length: int) -> pd.Series:
    """Hull Moving Average."""
    length = int(length)
    if length <= 0:
        return series * np.nan
    half_len = max(int(length / 2), 1)
    sqrt_len = max(int(np.sqrt(length)), 1)
    wma1 = wma(series, half_len)
    wma2 = wma(series, length)
    hull_raw = 2 * wma1 - wma2
    return wma(hull_raw, sqrt_len)


def ehma(series: pd.Series, length: int) -> pd.Series:
    """Exponential Hull MA variant."""
    length = int(length)
    if length <= 0:
        return series * np.nan
    half_len = max(int(length / 2), 1)
    sqrt_len = max(int(np.sqrt(length)), 1)
    ema1 = ema(series, half_len)
    ema2 = ema(series, length)
    hull_raw = 2 * ema1 - ema2
    return ema(hull_raw, sqrt_len)


def thma(series: pd.Series, length: int) -> pd.Series:
    """T3-like Hull MA variant (approx of THMA in Pine)."""
    length = int(length)
    if length <= 0:
        return series * np.nan
    l3 = max(int(length / 3), 1)
    l2 = max(int(length / 2), 1)
    wma1 = wma(series, l3)
    wma2 = wma(series, l2)
    wma3 = wma(series, length)
    combo = wma1 * 3 - wma2 - wma3
    return wma(combo, length)


def rsi(series: pd.Series, length: int) -> pd.Series:
    """Classic Wilder RSI."""
    length = int(length)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


def calc_qqe(src: pd.Series, rsi_len: int, rsi_smooth: int, qqe_factor: float):
    """
    Approximation of the QQE MOD logic from Pine.
    Returns:
        trend_line: pd.Series
        rsis: smoothed RSI series
    """
    rsi_raw = rsi(src, rsi_len)
    rsis = ema(rsi_raw, rsi_smooth)
    wl = rsi_len * 2 - 1

    atr = (rsis.shift(1) - rsis).abs()
    atrs = ema(atr, wl)
    d_atr = atrs * qqe_factor

    n = len(src)
    long_band = np.full(n, np.nan, dtype=float)
    short_band = np.full(n, np.nan, dtype=float)
    direction = np.zeros(n, dtype=int)
    trend_line = np.full(n, np.nan, dtype=float)

    rsis_vals = rsis.values
    d_vals = d_atr.values

    for i in range(1, n):
        if np.isnan(rsis_vals[i]) or np.isnan(d_vals[i]):
            # before enough data accumulates
            direction[i] = direction[i - 1]
            long_band[i] = long_band[i - 1]
            short_band[i] = short_band[i - 1]
            trend_line[i] = trend_line[i - 1]
            continue

        new_s_band = rsis_vals[i] + d_vals[i]
        new_l_band = rsis_vals[i] - d_vals[i]

        # previous values with nz-like behaviour
        prev_long = long_band[i - 1] if not np.isnan(long_band[i - 1]) else new_l_band
        prev_short = short_band[i - 1] if not np.isnan(short_band[i - 1]) else new_s_band
        prev_rsi = rsis_vals[i - 1]

        # update long band
        if prev_rsi > prev_long and rsis_vals[i] > prev_long:
            long_band[i] = max(prev_long, new_l_band)
        else:
            long_band[i] = new_l_band

        # update short band
        if prev_rsi < prev_short and rsis_vals[i] < prev_short:
            short_band[i] = min(prev_short, new_s_band)
        else:
            short_band[i] = new_s_band

        # cross logic (approx of ta.cross in Pine)
        cross_up = (prev_rsi < prev_short) and (rsis_vals[i] > prev_short)
        cross_down = (prev_rsi > prev_long) and (rsis_vals[i] < prev_long)

        if cross_up:
            direction[i] = 1
        elif cross_down:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

        trend_line[i] = long_band[i] if direction[i] == 1 else short_band[i]

    trend_line_series = pd.Series(trend_line, index=src.index)
    rsis_series = rsis
    return trend_line_series, rsis_series


def rolling_zscore(series: pd.Series, length: int) -> pd.Series:
    """Rolling Z-score."""
    mean = series.rolling(length).mean()
    std = series.rolling(length).std()
    return (series - mean) / std


def vwap_running(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Simple rolling VWAP anchored from start of series."""
    cum_vol = volume.cumsum()
    cum_pv = (close * volume).cumsum()
    return cum_pv / cum_vol.replace(0, np.nan)


# =========================
# Main strategy logic
# =========================

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reimplementation of the Pine Script indicator:
    'v2 TSA QQE MOD + Hull + Vol-Osc'

    Expects df with columns:
        'Open', 'High', 'Low', 'Close', 'Volume'
    (plus 'Epoch' or datetime index, but that's not required here)

    Outputs at least:
        df['Side'] = 1  (LONG entry signal)
        df['Side'] = -1 (SHORT entry signal)
        df['Side'] = 0  (no signal)
    """

    # Make sure columns exist
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # ---------------------------------
    # Hull Suite
    # ---------------------------------
    hull_source = close
    hull_length = 55
    length_mult = 1.0
    # modeSwitch could be "Hma", "Ehma", "Thma". We use "Hma" as in your Pine defaults.
    mode_switch = "Hma"

    hull_len_effective = int(hull_length * length_mult)

    if mode_switch == "Hma":
        hull = hma(hull_source, hull_len_effective)
    elif mode_switch == "Ehma":
        hull = ehma(hull_source, hull_len_effective)
    else:  # "Thma"
        hull = thma(hull_source, hull_len_effective)

    # In Pine, SHULL = HULL[2]; we just need slope over 2 bars:
    hull_shift2 = hull.shift(2)

    hull_up = hull > hull_shift2
    hull_dn = hull < hull_shift2

    # ---------------------------------
    # QQE MOD (Primary + Secondary)
    # ---------------------------------
    # Parameters as in your Pine script
    rsi_len_p = 6
    rsi_smooth_p = 5
    qqe_fact_p = 3.0
    thr_p = 3.0  # primary threshold (used only indirectly with BB)

    rsi_len_s = 6
    rsi_smooth_s = 5
    qqe_fact_s = 1.61
    thr_s = 3.0  # secondary threshold

    # Primary QQE
    pQQE, pRSI = calc_qqe(close, rsi_len_p, rsi_smooth_p, qqe_fact_p)
    # Secondary QQE
    sQQE, sRSI = calc_qqe(close, rsi_len_s, rsi_smooth_s, qqe_fact_s)

    # Bollinger on (pQQE - 50)
    bb_len = 50
    bb_mult = 0.35

    p_centered = pQQE - 50.0
    bb_basis = p_centered.rolling(bb_len).mean()
    bb_dev = bb_mult * p_centered.rolling(bb_len).std()
    bb_up = bb_basis + bb_dev
    bb_dn = bb_basis - bb_dev

    # QQE Up / Down signals (boolean)
    qqe_up = (sRSI - 50.0 > thr_s) & (pRSI - 50.0 > bb_up)
    qqe_dn = (sRSI - 50.0 < -thr_s) & (pRSI - 50.0 < bb_dn)

    # First bar of condition (like Pine's qqeUp1st / qqeDn1st)
    qqe_up_1st = qqe_up & (~qqe_up.shift(1).fillna(False))
    qqe_dn_1st = qqe_dn & (~qqe_dn.shift(1).fillna(False))

    # ---------------------------------
    # Volume Oscillator
    # ---------------------------------
    v_short_len = 5
    v_long_len = 10

    v_short = ema(volume, v_short_len)
    v_long = ema(volume, v_long_len)
    osc = 100.0 * (v_short - v_long) / v_long.replace(0, np.nan)

    vol_ok = osc > 0  # only trade if short-term volume > long-term

    # ---------------------------------
    # Composite raw signals (before filters)
    # ---------------------------------
    long_signal_raw = qqe_up_1st & hull_up & vol_ok
    short_signal_raw = qqe_dn_1st & hull_dn & vol_ok

    # ---------------------------------
    # Volatility Regime Filters (ATR + Volume Z-Score)
    # ---------------------------------
    atr_len = 14

    # True range components
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR using Wilder smoothing (approx)
    atr = tr.rolling(atr_len).mean()

    atr_med = atr.rolling(20).mean()
    high_vol = atr > atr_med * 1.05  # "milder" high-vol regime

    vol_z_len = 50
    vol_z = rolling_zscore(volume, vol_z_len)
    vol_spike = vol_z > 0.5

    vol_gate = high_vol | vol_spike   # at least one condition true

    # ---------------------------------
    # VWAP Bias (optional, default OFF like in Pine)
    # ---------------------------------
    use_bias = False         # change to True if you want to enable VWAP bias
    bias_tol = 0.002         # 0.2 %

    vwap_sess = vwap_running(close, volume)

    if use_bias:
        bias_long = close > vwap_sess * (1.0 - bias_tol)
        bias_short = close < vwap_sess * (1.0 + bias_tol)
    else:
        bias_long = pd.Series(True, index=df.index)
        bias_short = pd.Series(True, index=df.index)

    # ---------------------------------
    # Final qualified signals
    # ---------------------------------
    valid_long = long_signal_raw & vol_gate & bias_long
    valid_short = short_signal_raw & vol_gate & bias_short

    # ---------------------------------
    # Map into Side = 1 / -1 / 0
    # ---------------------------------
    side = pd.Series(0, index=df.index, dtype=int)
    side[valid_long] = 1
    side[valid_short] = -1

    # attach useful columns (optional, but nice for debugging)
    df['Hull'] = hull
    df['HullUp'] = hull_up.astype(int)
    df['HullDn'] = hull_dn.astype(int)
    df['pQQE'] = pQQE
    df['pRSI'] = pRSI
    df['sQQE'] = sQQE
    df['sRSI'] = sRSI
    df['VolOsc'] = osc
    df['VolGate'] = vol_gate.astype(int)
    df['BiasLong'] = bias_long.astype(int)
    df['BiasShort'] = bias_short.astype(int)
    df['ValidLong'] = valid_long.astype(int)
    df['ValidShort'] = valid_short.astype(int)

    df['Side'] = side

    return df
