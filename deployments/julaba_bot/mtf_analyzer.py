"""
Multi-Timeframe Analyzer for Julaba
Provides confluence confirmation across multiple timeframes.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger("Julaba.MTF")


class MultiTimeframeAnalyzer:
    """
    Multi-Timeframe Analysis Engine.
    
    Confirms signals by checking alignment across:
    - Primary timeframe (3m aggregated)
    - Secondary timeframe (15m)
    - Higher timeframe (1H)
    
    Features:
    - Trend alignment scoring
    - Momentum confluence
    - Volume confirmation
    - Support/Resistance proximity
    """
    
    def __init__(self):
        self.cached_15m: Optional[pd.DataFrame] = None
        self.cached_1h: Optional[pd.DataFrame] = None
        self.last_update_15m: Optional[float] = None
        self.last_update_1h: Optional[float] = None
        self.cache_duration_15m: int = 900  # 15 minutes
        self.cache_duration_1h: int = 3600  # 1 hour
    
    def update_data(self, df_15m: pd.DataFrame = None, df_1h: pd.DataFrame = None):
        """Update cached higher timeframe data."""
        import time
        now = time.time()
        
        if df_15m is not None:
            self.cached_15m = df_15m.copy()
            self.last_update_15m = now
            logger.debug(f"Updated 15m data: {len(df_15m)} bars")
        
        if df_1h is not None:
            self.cached_1h = df_1h.copy()
            self.last_update_1h = now
            logger.debug(f"Updated 1H data: {len(df_1h)} bars")
    
    def _calculate_trend(self, df: pd.DataFrame, fast_period: int = 10, slow_period: int = 20) -> Dict[str, Any]:
        """Calculate trend direction for a single timeframe."""
        if df is None or len(df) < slow_period:
            return {'direction': 'neutral', 'strength': 0, 'score': 0}
        
        close = df['close'] if 'close' in df.columns else df['Close']
        
        # SMA-based trend
        sma_fast = close.rolling(fast_period).mean()
        sma_slow = close.rolling(slow_period).mean()
        
        current_fast = sma_fast.iloc[-1]
        current_slow = sma_slow.iloc[-1]
        current_price = close.iloc[-1]
        
        # Trend direction
        if current_fast > current_slow and current_price > current_fast:
            direction = 'bullish'
            score = 1
        elif current_fast < current_slow and current_price < current_fast:
            direction = 'bearish'
            score = -1
        elif current_fast > current_slow:
            direction = 'weak_bullish'
            score = 0.5
        elif current_fast < current_slow:
            direction = 'weak_bearish'
            score = -0.5
        else:
            direction = 'neutral'
            score = 0
        
        # Trend strength (distance between SMAs as % of price)
        if current_price > 0:
            strength = abs(current_fast - current_slow) / current_price * 100
        else:
            strength = 0
        
        return {
            'direction': direction,
            'strength': round(strength, 3),
            'score': score,
            'price': current_price,
            'sma_fast': current_fast,
            'sma_slow': current_slow
        }
    
    def _calculate_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate momentum indicators."""
        if df is None or len(df) < 14:
            return {'rsi': 50, 'momentum': 0, 'score': 0}
        
        close = df['close'] if 'close' in df.columns else df['Close']
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # Rate of Change
        if len(close) >= 10:
            roc = (close.iloc[-1] / close.iloc[-10] - 1) * 100
        else:
            roc = 0
        
        # Momentum score
        if current_rsi > 60 and roc > 0:
            score = 1
        elif current_rsi < 40 and roc < 0:
            score = -1
        elif current_rsi > 50:
            score = 0.5
        elif current_rsi < 50:
            score = -0.5
        else:
            score = 0
        
        return {
            'rsi': round(current_rsi, 1),
            'roc': round(roc, 2),
            'score': score
        }
    
    def _calculate_volume_profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate volume characteristics."""
        if df is None or len(df) < 20:
            return {'volume_ratio': 1.0, 'trend': 'normal', 'score': 0}
        
        # Check for volume column (case-insensitive)
        vol_col = None
        for col in df.columns:
            if col.lower() == 'volume':
                vol_col = col
                break
        
        if vol_col is None:
            return {'volume_ratio': 1.0, 'trend': 'normal', 'score': 0}
        
        volume = df[vol_col]
        
        avg_volume = volume.tail(20).mean()
        recent_volume = volume.tail(3).mean()
        
        if avg_volume > 0:
            ratio = recent_volume / avg_volume
        else:
            ratio = 1.0
        
        if ratio > 1.5:
            trend = 'high'
            score = 0.5
        elif ratio < 0.5:
            trend = 'low'
            score = -0.5
        else:
            trend = 'normal'
            score = 0
        
        return {
            'volume_ratio': round(ratio, 2),
            'trend': trend,
            'score': score,
            'avg_volume': avg_volume,
            'recent_volume': recent_volume
        }
    
    def analyze(
        self,
        df_primary: pd.DataFrame,
        proposed_signal: int = 0
    ) -> Dict[str, Any]:
        """
        Perform multi-timeframe analysis.
        
        Args:
            df_primary: Primary timeframe data (3m aggregated)
            proposed_signal: The signal being evaluated (1=long, -1=short, 0=none)
        
        Returns:
            Dict with alignment score, confirmation status, and detailed analysis
        """
        result = {
            'confirmed': False,
            'alignment_score': 0.0,
            'confluence_pct': 0,
            'primary': {},
            'secondary': {},
            'higher': {},
            'volume': {},
            'recommendation': 'WAIT',
            'conflicts': [],
            'confirmations': [],
            'message': ''
        }
        
        # Always calculate trends for display, even without a signal
        # === Analyze Primary Timeframe (3m) ===
        primary_trend = self._calculate_trend(df_primary, fast_period=15, slow_period=40)
        primary_momentum = self._calculate_momentum(df_primary)
        primary_volume = self._calculate_volume_profile(df_primary)
        
        result['primary'] = {
            'trend': primary_trend,
            'momentum': primary_momentum,
            'volume': primary_volume
        }
        
        # === Analyze Secondary Timeframe (15m) ===
        if self.cached_15m is not None and len(self.cached_15m) >= 20:
            secondary_trend = self._calculate_trend(self.cached_15m, fast_period=10, slow_period=20)
            secondary_momentum = self._calculate_momentum(self.cached_15m)
            result['secondary'] = {
                'trend': secondary_trend,
                'momentum': secondary_momentum
            }
        else:
            secondary_trend = {'direction': 'neutral', 'score': 0}
            result['secondary'] = {'trend': secondary_trend, 'available': False}
        
        # === Analyze Higher Timeframe (1H) ===
        if self.cached_1h is not None and len(self.cached_1h) >= 20:
            higher_trend = self._calculate_trend(self.cached_1h, fast_period=10, slow_period=20)
            result['higher'] = {'trend': higher_trend}
        else:
            higher_trend = {'direction': 'neutral', 'score': 0}
            result['higher'] = {'trend': higher_trend, 'available': False}
        
        result['volume'] = primary_volume
        
        if proposed_signal == 0:
            result['message'] = 'No active signal - showing current trends'
            result['recommendation'] = primary_trend.get('direction', 'neutral').upper()
            return result
        
        signal_direction = 'bullish' if proposed_signal == 1 else 'bearish'
        
        # Get trends from earlier analysis for alignment calculation
        secondary_trend = result['secondary'].get('trend', {'direction': 'neutral', 'score': 0})
        secondary_momentum = result['secondary'].get('momentum', {'score': 0})
        higher_trend = result['higher'].get('trend', {'direction': 'neutral', 'score': 0})
        
        # === Calculate Alignment Score ===
        # Each component contributes to the score
        # Primary trend: 30%, Secondary trend: 30%, Higher TF: 25%, Momentum: 15%
        
        scores = []
        weights = []
        
        # Primary trend (weighted by signal alignment)
        if signal_direction == 'bullish':
            primary_aligned = primary_trend['score'] if primary_trend['score'] > 0 else -abs(primary_trend['score'])
        else:
            primary_aligned = -primary_trend['score'] if primary_trend['score'] < 0 else -abs(primary_trend['score'])
        scores.append(primary_aligned)
        weights.append(0.30)
        
        # Secondary trend
        if self.cached_15m is not None:
            if signal_direction == 'bullish':
                secondary_aligned = secondary_trend['score'] if secondary_trend['score'] > 0 else -abs(secondary_trend['score'])
            else:
                secondary_aligned = -secondary_trend['score'] if secondary_trend['score'] < 0 else -abs(secondary_trend['score'])
            scores.append(secondary_aligned)
            weights.append(0.30)
        
        # Higher timeframe
        if self.cached_1h is not None:
            if signal_direction == 'bullish':
                higher_aligned = higher_trend['score'] if higher_trend['score'] > 0 else -abs(higher_trend['score'])
            else:
                higher_aligned = -higher_trend['score'] if higher_trend['score'] < 0 else -abs(higher_trend['score'])
            scores.append(higher_aligned)
            weights.append(0.25)
        
        # Momentum
        momentum_aligned = primary_momentum['score'] if (
            (signal_direction == 'bullish' and primary_momentum['score'] > 0) or
            (signal_direction == 'bearish' and primary_momentum['score'] < 0)
        ) else -abs(primary_momentum['score'])
        scores.append(momentum_aligned)
        weights.append(0.15)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calculate weighted alignment
        alignment_score = sum(s * w for s, w in zip(scores, weights))
        result['alignment_score'] = round(alignment_score, 2)
        
        # Convert to percentage (0-100)
        confluence_pct = int((alignment_score + 1) / 2 * 100)
        result['confluence_pct'] = confluence_pct
        
        # === Determine Confirmation ===
        # Confirmation criteria:
        # - Alignment score > 0.3 for weak confirmation
        # - Alignment score > 0.6 for strong confirmation
        # - No major conflicts
        
        conflicts = []
        confirmations = []
        
        # Check for conflicts
        if signal_direction == 'bullish':
            if primary_trend['direction'] in ['bearish', 'weak_bearish']:
                conflicts.append(f"3m trend is {primary_trend['direction']}")
            else:
                confirmations.append("3m trend aligned")
            
            if self.cached_15m is not None and secondary_trend['direction'] in ['bearish', 'weak_bearish']:
                conflicts.append(f"15m trend is {secondary_trend['direction']}")
            elif self.cached_15m is not None:
                confirmations.append("15m trend aligned")
            
            if self.cached_1h is not None and higher_trend['direction'] in ['bearish', 'weak_bearish']:
                conflicts.append(f"1H trend is {higher_trend['direction']}")
            elif self.cached_1h is not None:
                confirmations.append("1H trend aligned")
        else:  # bearish
            if primary_trend['direction'] in ['bullish', 'weak_bullish']:
                conflicts.append(f"3m trend is {primary_trend['direction']}")
            else:
                confirmations.append("3m trend aligned")
            
            if self.cached_15m is not None and secondary_trend['direction'] in ['bullish', 'weak_bullish']:
                conflicts.append(f"15m trend is {secondary_trend['direction']}")
            elif self.cached_15m is not None:
                confirmations.append("15m trend aligned")
            
            if self.cached_1h is not None and higher_trend['direction'] in ['bullish', 'weak_bullish']:
                conflicts.append(f"1H trend is {higher_trend['direction']}")
            elif self.cached_1h is not None:
                confirmations.append("1H trend aligned")
        
        # Volume check
        if primary_volume['volume_ratio'] < 0.5:
            conflicts.append("Low volume")
        elif primary_volume['volume_ratio'] > 1.2:
            confirmations.append("Volume confirmation")
        
        result['conflicts'] = conflicts
        result['confirmations'] = confirmations
        
        # Final confirmation decision
        if alignment_score >= 0.6 and len(conflicts) == 0:
            result['confirmed'] = True
            result['recommendation'] = 'STRONG_CONFIRM'
            result['message'] = f"✅ STRONG: {confluence_pct}% confluence | {len(confirmations)} confirmations"
        elif alignment_score >= 0.3 and len(conflicts) <= 1:
            result['confirmed'] = True
            result['recommendation'] = 'CONFIRM'
            result['message'] = f"✅ CONFIRMED: {confluence_pct}% confluence | {len(confirmations)} conf, {len(conflicts)} conflict"
        elif alignment_score >= 0.1:
            result['confirmed'] = False
            result['recommendation'] = 'WEAK'
            result['message'] = f"⚠️ WEAK: {confluence_pct}% confluence | Consider waiting"
        else:
            result['confirmed'] = False
            result['recommendation'] = 'REJECT'
            result['message'] = f"❌ REJECT: {confluence_pct}% confluence | {len(conflicts)} conflicts"
        
        logger.info(f"MTF Analysis: {result['message']}")
        
        return result
    
    def get_summary(self) -> str:
        """Get a text summary of current MTF status."""
        lines = ["📊 *Multi-Timeframe Status*\n"]
        
        if self.cached_15m is not None:
            trend_15m = self._calculate_trend(self.cached_15m)
            lines.append(f"15m: {trend_15m['direction'].upper()} (strength: {trend_15m['strength']:.2f}%)")
        else:
            lines.append("15m: No data")
        
        if self.cached_1h is not None:
            trend_1h = self._calculate_trend(self.cached_1h)
            lines.append(f"1H: {trend_1h['direction'].upper()} (strength: {trend_1h['strength']:.2f}%)")
        else:
            lines.append("1H: No data")
        
        return "\n".join(lines)


# Singleton instance
_mtf_analyzer: Optional[MultiTimeframeAnalyzer] = None


def get_mtf_analyzer() -> MultiTimeframeAnalyzer:
    """Get the global MTF analyzer instance."""
    global _mtf_analyzer
    if _mtf_analyzer is None:
        _mtf_analyzer = MultiTimeframeAnalyzer()
    return _mtf_analyzer
