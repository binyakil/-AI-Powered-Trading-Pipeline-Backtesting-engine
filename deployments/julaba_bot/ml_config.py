"""
ML Configuration Module for Julaba
Centralized ML settings following the ML Acceleration Plan.

KEY DECISIONS:
1. ML does NOT block trades - it collects data passively
2. Gemini AI is the primary decision maker
3. ML learns from YOUR actual trade outcomes
4. Gradually enable ML influence as samples grow
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import logging

logger = logging.getLogger("Julaba.MLConfig")


@dataclass
class MLConfig:
    """
    ML Model Configuration.
    
    The ML model provides supplementary trade filtering.
    Gemini AI remains the primary decision maker.
    ML influence ramps up automatically as samples grow.
    """
    # Master switch
    enabled: bool = True  # Keep collecting data
    
    # CRITICAL: ML influence on trade decisions
    # 0.0 = ML never blocks trades
    # 0.15 = ML can provide 15% weight (RECOMMENDED starting point)
    # 0.25 = ML can reject 25% confidence signals
    # 0.50 = ML has equal weight with Gemini
    # 1.0 = ML is primary (NOT RECOMMENDED)
    influence_weight: float = 0.15  # Re-enabled with conservative starting weight
    
    # Sample thresholds
    min_samples_for_predictions: int = 50  # Reduced from 300
    min_samples_for_full_weight: int = 200  # Gradual ramp-up
    
    # Auto-ramp ML influence as samples grow
    auto_ramp_influence: bool = True
    
    # Retrain frequency
    retrain_every_n_trades: int = 10
    
    def get_current_influence(self, current_samples: int) -> float:
        """
        Calculate ML influence based on sample count.
        
        0-49 samples: 0% influence (learning only)
        50-99 samples: 0% influence (can predict, but don't block)
        100-149 samples: 10% influence
        150-199 samples: 20% influence
        200+ samples: 30% influence (never exceed 30% - Gemini is primary)
        """
        if not self.auto_ramp_influence:
            return self.influence_weight
        
        if current_samples < self.min_samples_for_predictions:
            return 0.0
        elif current_samples < 100:
            return 0.0  # Can predict, but don't influence
        elif current_samples < 150:
            return 0.10
        elif current_samples < self.min_samples_for_full_weight:
            return 0.20
        else:
            return 0.30  # Max 30% - Gemini remains primary
    
    def should_block_trade(self, ml_score: float, current_samples: int) -> bool:
        """
        Determine if ML should block a trade.
        
        With current settings (influence_weight=0.0), this always returns False.
        ML collects data but doesn't block.
        """
        influence = self.get_current_influence(current_samples)
        
        if influence == 0.0:
            return False  # Never block
        
        # Only block if ML is very confident trade will lose
        # AND we have enough influence weight
        block_threshold = 1.0 - influence  # Higher influence = lower threshold to block
        
        return ml_score < (1.0 - block_threshold)


@dataclass
class TradingPairConfig:
    """Configuration for a single trading pair."""
    pair: str
    enabled: bool = True
    risk_multiplier: float = 1.0  # Reduces position size for less-tested pairs
    
    # Pair-specific overrides (optional)
    min_atr: float = 0.0
    max_spread_pct: float = 0.005  # 0.5% max spread


@dataclass
class MultiPairConfig:
    """
    Multi-pair trading configuration.
    
    Start with 3 pairs as agreed:
    - LINK/USDT (primary, 1.0x risk)
    - ETH/USDT (0.8x risk - more stable)
    - SOL/USDT (0.7x risk - more volatile)
    """
    # Trading pairs with risk multipliers
    pairs: List[TradingPairConfig] = field(default_factory=lambda: [
        TradingPairConfig(pair="LINK/USDT", enabled=True, risk_multiplier=1.0),
        TradingPairConfig(pair="ETH/USDT", enabled=True, risk_multiplier=0.8),
        TradingPairConfig(pair="SOL/USDT", enabled=True, risk_multiplier=0.7),
    ])
    
    # Portfolio-level limits
    max_total_positions: int = 2  # Max 2 positions at once (conservative)
    max_correlated_positions: int = 2  # All crypto is correlated
    
    # Portfolio-level risk
    portfolio_daily_loss_limit: float = 0.05  # 5% of total portfolio
    portfolio_weekly_loss_limit: float = 0.10  # 10% of total portfolio
    
    # Position limits per pair
    max_position_per_pair: int = 1
    
    def get_enabled_pairs(self) -> List[TradingPairConfig]:
        """Get list of enabled trading pairs."""
        return [p for p in self.pairs if p.enabled]
    
    def get_pair_config(self, pair: str) -> TradingPairConfig:
        """Get config for a specific pair."""
        for p in self.pairs:
            if p.pair == pair:
                return p
        # Return default config if not found
        return TradingPairConfig(pair=pair, risk_multiplier=0.5)


@dataclass
class TradeLogSchema:
    """
    Complete trade logging schema for ML training.
    
    Every trade (win or lose) must log these fields for ML to learn.
    """
    # Identifiers
    trade_id: str = ""
    timestamp: str = ""
    pair: str = ""
    source: str = "live"  # "live", "backtest", "augmented"
    
    # Entry conditions
    entry_price: float = 0.0
    entry_regime: str = ""  # TRENDING, CHOPPY, VOLATILE
    entry_atr: float = 0.0
    entry_atr_pct: float = 0.0
    entry_rsi: float = 0.0
    entry_adx: float = 0.0
    entry_volume_ratio: float = 0.0
    entry_btc_correlation: float = 0.0
    entry_hurst: float = 0.0
    entry_sma_distance_pct: float = 0.0
    entry_timeframe_alignment: bool = False
    entry_hour: int = 0
    entry_day_of_week: int = 0
    
    # AI assessments at entry
    gemini_confidence: float = 0.0
    gemini_approved: bool = False
    gemini_reasoning: str = ""
    ml_score: float = 0.0
    ml_regime: str = ""
    
    # Outcome (filled after trade closes)
    exit_price: float = 0.0
    exit_reason: str = ""  # TP1, TP2, TP3, STOP, MANUAL, CIRCUIT_BREAKER
    pnl_dollars: float = 0.0
    pnl_r_multiple: float = 0.0
    duration_minutes: int = 0
    max_drawdown_during_trade: float = 0.0
    max_profit_during_trade: float = 0.0
    
    # Label for ML
    outcome: int = 0  # 1 = win (any TP), 0 = loss (stop hit)
    sample_weight: float = 1.0  # 1.0 for live, 0.3 for backtest, 0.2 for augmented
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp,
            "pair": self.pair,
            "source": self.source,
            "entry_price": self.entry_price,
            "entry_regime": self.entry_regime,
            "entry_atr": self.entry_atr,
            "entry_atr_pct": self.entry_atr_pct,
            "entry_rsi": self.entry_rsi,
            "entry_adx": self.entry_adx,
            "entry_volume_ratio": self.entry_volume_ratio,
            "entry_btc_correlation": self.entry_btc_correlation,
            "entry_hurst": self.entry_hurst,
            "entry_sma_distance_pct": self.entry_sma_distance_pct,
            "entry_timeframe_alignment": self.entry_timeframe_alignment,
            "entry_hour": self.entry_hour,
            "entry_day_of_week": self.entry_day_of_week,
            "gemini_confidence": self.gemini_confidence,
            "gemini_approved": self.gemini_approved,
            "gemini_reasoning": self.gemini_reasoning,
            "ml_score": self.ml_score,
            "ml_regime": self.ml_regime,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "pnl_dollars": self.pnl_dollars,
            "pnl_r_multiple": self.pnl_r_multiple,
            "duration_minutes": self.duration_minutes,
            "max_drawdown_during_trade": self.max_drawdown_during_trade,
            "max_profit_during_trade": self.max_profit_during_trade,
            "outcome": self.outcome,
            "sample_weight": self.sample_weight,
        }


# Global configuration instances
ML_CONFIG = MLConfig()
MULTI_PAIR_CONFIG = MultiPairConfig()


def get_ml_config() -> MLConfig:
    """Get the global ML configuration."""
    return ML_CONFIG


def get_multi_pair_config() -> MultiPairConfig:
    """Get the global multi-pair configuration."""
    return MULTI_PAIR_CONFIG


# Log configuration on import
logger.info(f"ML Config: influence_weight={ML_CONFIG.influence_weight}, "
           f"min_samples={ML_CONFIG.min_samples_for_predictions}")
logger.info(f"Multi-Pair: {len(MULTI_PAIR_CONFIG.get_enabled_pairs())} pairs enabled, "
           f"max_positions={MULTI_PAIR_CONFIG.max_total_positions}")
