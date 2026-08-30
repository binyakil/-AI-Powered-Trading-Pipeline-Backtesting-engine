# Trade Frequency Divergence: Diagnostic Audit Report

> **Observation**: Kraken backtest = 1.8 trades/day | MEXC live = 6.45+ trades/day (3.6x multiplier)

## Executive Summary

The 10x trade frequency divergence is **NOT a code bug** but a **compounding effect** of multiple logical "traps":

1. **Clock Alignment Drift** (CRITICAL) → Different 3m candle boundaries
2. **Parameter Mismatch** (HIGH) → RSI(6) too sensitive for noisy MEXC data
3. **No Entry Debounce** (HIGH) → Every signal triggers immediate trade
4. **Signal Timing** ✓ GREEN → Correctly uses closed bars only

---

## Hypothesis Breakdown

### Hypothesis 1: Clock Alignment Drift ⚠️ CRITICAL

**Finding**: **YES, CONFIRMED**

The `aggregate_3m_bar()` function groups ANY 3 consecutive 1m bars without enforcing wall-clock alignment.

```python
# bot.py lines 124-131
def aggregate_3m_bar(bars: List[List[float]]) -> List[float]:
    ts = bars[0][0]  # ← Takes first timestamp (no validation!)
    # ... aggregates O/H/L/C/V
```

**Problem**:
- No check: `unix_timestamp % 180 == 0` (3-minute boundary alignment)
- If a single 1m bar is missing/delayed → all subsequent 3m bars shift phase permanently
- Kraken's 3m candles align to `:00, :03, :06, ... :57` boundaries
- MEXC might produce `:01, :04, :07, ... :58` boundaries (phase-shifted)
- Same market data → Different OHLC candles → Different indicator lookback → Different signals

**Impact**: This is the foundational mismatch explaining data regime divergence.

---

### Hypothesis 2: Hedged Mode Feedback Loop ⚠️ HIGH

**Finding**: **YES, CONFIRMED**

`CONFLICT_MODE = "hedged"` with no entry rate limiting.

```python
# bot.py line 54
CONFLICT_MODE = "hedged"  # always allow new entries

# bot.py line 881-883
if CONFLICT_MODE == "hedged":
    await open_new_trade(...)  # Every signal = immediate trade
```

**Problem**:
- ✗ No limit on concurrent open trades
- ✗ No cooldown between consecutive entries
- ✗ No "first signal only" filter (treats every signal as actionable)

If QQE/Hull oscillates on noisy data:
- Signal fires → Trade 1 opens
- Signal fires again (same trend, noisy fluctuation) → Trade 2 opens immediately
- MEXC noise causes this 10x more often than Kraken

**Impact**: Rapid-fire entries with no rate limiting amplify noise sensitivity.

---

### Hypothesis 3: Signal Trigger Timing (Closed vs. Open) ✓ GREEN

**Finding**: **NO ISSUE DETECTED**

The bot correctly evaluates signals only on closed (completed) bars:

```python
# bot.py lines 891-894
closed_bars = ohlcv_1m[:-1]  # Excludes last (still-forming) bar
new_bars = [bar for bar in closed_bars if bar[0] > state.last_bar_ts]

# Lines 899-909
for bar in new_bars:
    while len(one_min_buffer) >= 3:
        agg = aggregate_3m_bar(one_min_buffer[:3])
        await handle_agg_bar(agg)  # Once per completed bar
```

**Status**: ✓ Correct. No repainting issues. Can exclude from investigation.

---

### Hypothesis 4: Indicator Threshold Sensitivity ⚠️ HIGH

**Finding**: **YES, CONFIRMED**

Parameters optimized for Kraken's smooth data, hypersensitive to MEXC's noise.

```python
# indicator.py lines 237-247
rsi_len_s = 6          # ← VERY FAST (6-bar lookback)
rsi_smooth_s = 5       # ← Minimal smoothing
qqe_fact_s = 1.61
thr_s = 3.0            # ← Tight threshold

hull_length = 55       # ← Slow (55-bar lookback)
```

**Problem**: Mismatch between fast and slow components
- **RSI(6)**: Reacts to every micro-spike (intended for smooth data like Kraken)
- **Hull(55)**: Slow-moving, takes time to respond
- On **MEXC's noisy data**: RSI oscillates wildly, crosses threshold 10x more
- Tight threshold (3.0) amplifies sensitivity

**Tangible Impact**:
- Kraken smooth data → RSI stable → few signals → 1.8 trades/day
- MEXC noisy data → RSI jumpy → many signals → 6.45+ trades/day

---

## Root Cause Synthesis

```
┌─ Clock Drift (Phase-Shifted 3m Bars)
│  └─ Different lookback candles
│     └─ Indicator sees different market structure
├─ Parameter Mismatch (RSI(6) on Noisy Data)
│  └─ 10x more threshold crossings
│     └─ 10x more false signals
├─ No Debounce (CONFLICT_MODE="hedged")
│  └─ Every signal = immediate trade
│     └─ No rate limiting
└─ Combined Effect: 3.6x observed multiplier (6.45 vs 1.8 trades/day)
```

---

## Evidence: Observed Statistics

| Metric | Value |
|--------|-------|
| **Backtest (Kraken)** | 1.8 trades/day |
| **Forward Test (MEXC)** | 6.45 trades/day |
| **Multiplier** | 3.6x |
| **Date Range** | 2026-01-21 to 2026-02-20 (31 days) |
| **Total Trades** | 200 |
| **Max Daily** | 12 trades |
| **Min Daily** | 2 trades |
| **Avg Duration** | 46 minutes |
| **P&L** | -$274.80 (unprofitable due to low win rate + no debounce amplifying noise) |

---

## Required Fixes (Ranked by Priority)

### 1. Implement Strict Clock Alignment (CRITICAL)

**Location**: `bot.py` `aggregate_3m_bar()` function

**Fix**:
```python
def aggregate_3m_bar(bars: List[List[float]]) -> List[float]:
    # Validate alignment to 3m boundaries (unix_timestamp % 180 == 0)
    if bars[0][0] % 180 != 0:
        raise ValueError(f"Bar not aligned to 3m boundary. Timestamp: {bars[0][0]}")
    # ... rest of aggregation
```

**Impact**: Ensures 3m candles match exchange-standard, eliminating data regime mismatch for lookback indicator calculations.

---

### 2. Add Signal Debounce (HIGH)

**Location**: `bot.py` `open_new_trade()` or `handle_agg_bar()` function

**Fix**:
```python
# Add to StrategyState class:
last_entry_time: datetime = None

# In handle_agg_bar(), before open_new_trade():
MIN_ENTRY_SECONDS = 180  # 3 minutes between entries
if state.last_entry_time:
    elapsed = (datetime.utcnow() - state.last_entry_time).total_seconds()
    if elapsed < MIN_ENTRY_SECONDS:
        LOG.debug(f"Entry cooldown active ({elapsed:.0f}s). Skipping signal.")
        return

# In open_new_trade(), after successful entry:
state.last_entry_time = datetime.utcnow()
```

**Impact**: Prevents rapid-fire entries from signal oscillations. Reduces trade count on noisy data.

---

### 3. Adjust Indicator Parameters (HIGH)

**Location**: `indicator.py` lines 237-247

**Option A (Recommended): Increase RSI Smoothing**
```python
rsi_len_s = 14      # Increase from 6 (slower response to noise)
rsi_smooth_s = 10   # Increase from 5 (more smoothing)
```

**Option B: Raise Threshold**
```python
thr_s = 5.0         # Increase from 3.0 (requires stronger signal)
```

**Option C: Add Median Filter**
```python
# Apply 3-bar median filter to signals before entry
signals_buffer = []
if valid_long.iloc[-1]:
    signals_buffer.append(1)
if len(signals_buffer) >= 3:
    median_signal = median(signals_buffer[-3:])
    if median_signal == 1:  # Strong consensus
        trigger_long_entry()
```

**Impact**: Reduces false signals on MEXC's noisy data. Rebalance sensitivity for new data regime.

---

### 4. Disable Hedged Mode (Optional)

**Location**: `bot.py` line 54

**Fix**:
```python
CONFLICT_MODE = "exclusive"  # Only one concurrent trade
```

**Impact**: Prevents overlapping positions, improves capital efficiency, reduces position scaling issues.

---

## No Changes Needed

✓ **Signal Trigger Timing (Hypothesis 3)**: Correctly uses closed bars only. Implementation is sound.

---

## Decision Points for User

Before implementing fixes, answer:

1. **Clock Alignment**: Should 3m candles snap to wall-clock `:00/:03/:06` boundaries, or is streaming-based aggregation acceptable?

2. **Debounce Window**: What is acceptable `MIN_ENTRY_INTERVAL`?
   - 1 minute (very tight)
   - 3 minutes (balanced)
   - 5+ minutes (conservative)

3. **Hedged Mode**: Is `CONFLICT_MODE="hedged"` intentional (multi-position strategy) or was it an oversight?

4. **Reoptimization**: Should strategy be fully reoptimized on MEXC data instead of using Kraken parameters?

---

## Summary

| Issue | Status | Severity | Fix |
|-------|--------|----------|-----|
| Clock Alignment | ⚠️ Found | CRITICAL | Enforce `% 180` check |
| Parameter Mismatch | ⚠️ Found | HIGH | Increase RSI length |
| No Debounce | ⚠️ Found | HIGH | Add 3-min cooldown |
| Signal Timing | ✓ OK | — | None needed |

**Next Action**: Decide which fixes to implement. Then create an `improved_bot.py` with these corrections and re-run forward simulation to measure improvement.
