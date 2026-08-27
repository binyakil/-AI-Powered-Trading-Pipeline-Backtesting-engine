"""
Risk Manager Module for Julaba
Centralized risk management with dynamic position sizing, drawdown limits, and time-based cooldowns.
With state persistence (Notebook #2 §12) — survives restarts.

RiskShield: Hedge-fund grade circuit breaker layer that sits ABOVE all trading logic.
No trade opens without passing the Shield. Period.
"""

import logging
import threading
import json
import os
from datetime import date, datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

logger = logging.getLogger("Julaba.RiskManager")

RISK_STATE_FILE = Path(__file__).parent / "risk_state.json"
SHIELD_STATE_FILE = Path(__file__).parent / "shield_state.json"


@dataclass
class TradeOutcome:
    """Record of a completed trade."""
    timestamp: datetime
    symbol: str
    side: str
    pnl: float
    pnl_pct: float
    duration_minutes: int
    ai_approved: bool = True
    ai_confidence: float = 0.0


class RiskManager:
    """
    Centralized Risk Manager for Julaba.
    
    Features:
    - Dynamic position sizing (Kelly Criterion + volatility)
    - Daily/Weekly loss limits with circuit breakers
    - Time-based cooldown after losses
    - Streak-based risk adjustment
    - Correlation-aware position limits
    - Portfolio-level multi-pair risk management (ML Acceleration Plan)
    """
    
    def __init__(
        self,
        base_risk_pct: float = 0.02,
        max_risk_pct: float = 0.04,
        min_risk_pct: float = 0.005,
        daily_loss_limit: float = 0.05,
        weekly_loss_limit: float = 0.10,
        cooldown_minutes: int = 30,
        max_consecutive_losses: int = 3,
        # === PORTFOLIO-LEVEL LIMITS (ML Acceleration Plan) ===
        max_total_positions: int = 2,  # Max simultaneous positions across all pairs
        max_correlated_positions: int = 2,  # All crypto is correlated
        portfolio_max_risk_pct: float = 0.04  # Max total portfolio risk at any time
    ):
        # Base parameters
        self.base_risk_pct = base_risk_pct
        self.max_risk_pct = max_risk_pct
        self.min_risk_pct = min_risk_pct
        
        # Loss limits
        self.daily_loss_limit = daily_loss_limit
        self.weekly_loss_limit = weekly_loss_limit
        
        # Cooldown settings
        self.cooldown_minutes = cooldown_minutes
        self.max_consecutive_losses = max_consecutive_losses
        
        # === PORTFOLIO-LEVEL LIMITS (ML Acceleration Plan) ===
        self.max_total_positions = max_total_positions
        self.max_correlated_positions = max_correlated_positions
        self.portfolio_max_risk_pct = portfolio_max_risk_pct
        self.current_positions: Dict[str, float] = {}  # symbol -> risk_amount
        
        # State tracking
        self.trade_outcomes: List[TradeOutcome] = []
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.last_trade_time: Optional[datetime] = None
        self.last_loss_time: Optional[datetime] = None
        self.consecutive_losses: int = 0
        self.consecutive_wins: int = 0
        
        # Circuit breaker state
        self.daily_limit_hit: bool = False
        self.weekly_limit_hit: bool = False
        self.cooldown_active: bool = False
        self.cooldown_until: Optional[datetime] = None
        
        # Date tracking for resets
        self.current_date: date = datetime.now(timezone.utc).date()
        self.week_start: date = self._get_week_start()
        
        # Performance metrics for Kelly
        self.win_rate: float = 0.5
        self.avg_win: float = 0.01
        self.avg_loss: float = 0.01
        self._last_kelly_warn: Optional[datetime] = None
        
        logger.info(f"RiskManager initialized | Base risk: {base_risk_pct:.1%} | "
                   f"Daily limit: {daily_loss_limit:.1%} | Weekly limit: {weekly_loss_limit:.1%} | "
                   f"Max positions: {max_total_positions}")
        
        # Load persisted state (Notebook #2 §12)
        self._load_state()
    
    def _save_state(self):
        """Persist risk manager state to disk (Notebook #2 §12: survives restarts)."""
        try:
            state = {
                'daily_pnl': self.daily_pnl,
                'weekly_pnl': self.weekly_pnl,
                'consecutive_losses': self.consecutive_losses,
                'consecutive_wins': self.consecutive_wins,
                'cooldown_active': self.cooldown_active,
                'cooldown_until': self.cooldown_until.isoformat() if self.cooldown_until else None,
                'daily_limit_hit': self.daily_limit_hit,
                'weekly_limit_hit': self.weekly_limit_hit,
                'current_date': str(self.current_date),
                'week_start': str(self.week_start),
                'win_rate': self.win_rate,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'last_saved': datetime.now(timezone.utc).isoformat(),
                'trade_count': len(self.trade_outcomes),
            }
            with open(RISK_STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save risk state: {e}")
    
    def _load_state(self):
        """Load persisted state from disk."""
        try:
            if not RISK_STATE_FILE.exists():
                return
            
            with open(RISK_STATE_FILE, 'r') as f:
                state = json.load(f)
            
            # Only restore if same day (daily resets automatically)
            saved_date = state.get('current_date', '')
            today = str(datetime.now(timezone.utc).date())
            
            if saved_date == today:
                self.daily_pnl = state.get('daily_pnl', 0.0)
                self.daily_limit_hit = state.get('daily_limit_hit', False)
                logger.info(f"📊 Risk state restored: Daily P&L=${self.daily_pnl:+.2f}")
            else:
                logger.info(f"📊 New day detected — daily risk state reset (was {saved_date})")
            
            # Weekly state
            saved_week = state.get('week_start', '')
            current_week = str(self._get_week_start())
            if saved_week == current_week:
                self.weekly_pnl = state.get('weekly_pnl', 0.0)
                self.weekly_limit_hit = state.get('weekly_limit_hit', False)
            
            # Streaks and cooldown
            self.consecutive_losses = state.get('consecutive_losses', 0)
            self.consecutive_wins = state.get('consecutive_wins', 0)
            self.win_rate = state.get('win_rate', 0.5)
            self.avg_win = state.get('avg_win', 0.01)
            self.avg_loss = state.get('avg_loss', 0.01)
            
            # Restore cooldown if still active
            cooldown_str = state.get('cooldown_until')
            if cooldown_str and state.get('cooldown_active'):
                cooldown_time = datetime.fromisoformat(cooldown_str)
                if cooldown_time > datetime.now(timezone.utc):
                    self.cooldown_active = True
                    self.cooldown_until = cooldown_time
                    remaining = (cooldown_time - datetime.now(timezone.utc)).total_seconds() / 60
                    logger.info(f"🛑 Cooldown restored: {remaining:.0f}min remaining")
            
            logger.info(f"✅ Risk state loaded: W{self.consecutive_wins}/L{self.consecutive_losses}, "
                       f"Daily=${self.daily_pnl:+.2f}, Weekly=${self.weekly_pnl:+.2f}")
        except Exception as e:
            logger.warning(f"Could not load risk state: {e} — starting fresh")
    
    def _get_week_start(self) -> datetime:
        """Get the start of the current week (Monday)."""
        today = datetime.now(timezone.utc)
        days_since_monday = today.weekday()
        return (today - timedelta(days=days_since_monday)).date()
    
    def _check_date_resets(self):
        """Reset daily/weekly counters if needed."""
        now = datetime.now(timezone.utc)
        today = now.date()
        week_start = self._get_week_start()
        
        # Daily reset
        if today != self.current_date:
            logger.info(f"New day detected - resetting daily P&L from ${self.daily_pnl:.2f}")
            self.daily_pnl = 0.0
            self.daily_limit_hit = False
            self.current_date = today
        
        # Weekly reset
        if week_start != self.week_start:
            logger.info(f"New week detected - resetting weekly P&L from ${self.weekly_pnl:.2f}")
            self.weekly_pnl = 0.0
            self.weekly_limit_hit = False
            self.week_start = week_start
    
    def record_trade(
        self,
        symbol: str,
        side: str,
        pnl: float,
        pnl_pct: float,
        duration_minutes: int = 0,
        ai_approved: bool = True,
        ai_confidence: float = 0.0
    ):
        """Record a completed trade outcome."""
        now = datetime.now(timezone.utc)
        
        outcome = TradeOutcome(
            timestamp=now,
            symbol=symbol,
            side=side,
            pnl=pnl,
            pnl_pct=pnl_pct,
            duration_minutes=duration_minutes,
            ai_approved=ai_approved,
            ai_confidence=ai_confidence
        )
        self.trade_outcomes.append(outcome)
        
        # Keep only last 100 trades
        if len(self.trade_outcomes) > 100:
            self.trade_outcomes = self.trade_outcomes[-100:]
        
        # Update P&L tracking
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        
        # Update streaks
        is_win = pnl > 0
        if is_win:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.last_loss_time = now
            
            # Activate cooldown after consecutive losses
            if self.consecutive_losses >= 2:
                self._activate_cooldown(now)
        
        # Update performance metrics
        self._update_metrics()
        
        self.last_trade_time = now
        
        # Persist state (Notebook #2 §12)
        self._save_state()
        
        logger.info(f"Trade recorded: {side} {symbol} P&L=${pnl:+.2f} ({pnl_pct:+.2%}) | "
                   f"Streak: {self.consecutive_wins}W/{self.consecutive_losses}L | "
                   f"Daily: ${self.daily_pnl:+.2f} Weekly: ${self.weekly_pnl:+.2f}")
    
    def _activate_cooldown(self, now: datetime):
        """Activate trading cooldown."""
        self.cooldown_active = True
        self.cooldown_until = now + timedelta(minutes=self.cooldown_minutes)
        self._save_state()  # Persist cooldown (Notebook #2 §12)
        logger.warning(f"🛑 Cooldown activated for {self.cooldown_minutes}min after {self.consecutive_losses} losses")
    
    def _update_metrics(self):
        """Update win rate and average win/loss for Kelly calculation."""
        if len(self.trade_outcomes) < 5:
            return
        
        recent = self.trade_outcomes[-50:]  # Use last 50 trades
        wins = [t for t in recent if t.pnl > 0]
        losses = [t for t in recent if t.pnl <= 0]
        
        if len(recent) > 0:
            self.win_rate = len(wins) / len(recent)
        
        if wins:
            self.avg_win = np.mean([t.pnl_pct for t in wins])
        if losses:
            self.avg_loss = abs(np.mean([t.pnl_pct for t in losses]))
    
    def calculate_kelly_fraction(self) -> float:
        """Calculate optimal bet size using Kelly Criterion."""
        if self.avg_loss == 0:
            return self.base_risk_pct
        
        # Kelly: f* = (p * W - q * L) / (W * L)
        # Simplified: f* = W/L * p - (1-p) / (W/L)
        p = self.win_rate
        q = 1 - p
        w = self.avg_win
        l = self.avg_loss
        
        if l == 0:
            l = 0.01
        
        win_loss_ratio = w / l
        
        kelly = (p * win_loss_ratio - q) / win_loss_ratio
        
        # Negative Kelly = negative edge = STOP TRADING
        if kelly <= 0:
            now = datetime.now(timezone.utc)
            if self._last_kelly_warn is None or (now - self._last_kelly_warn).total_seconds() >= 300:
                logger.warning(f"⚠️ Kelly criterion negative ({kelly:.4f}) - edge is losing, reducing risk to 0")
                self._last_kelly_warn = now
            return 0.0
        
        # Use half-Kelly for safety
        half_kelly = kelly / 2
        
        # Clip to reasonable bounds
        return float(np.clip(half_kelly, self.min_risk_pct, self.max_risk_pct))
    
    def can_trade(self, balance: float, initial_balance: float) -> Dict[str, Any]:
        """
        Check if trading is allowed based on risk limits.
        
        Returns:
            Dict with 'allowed' bool, 'reason' str, and additional context
        """
        self._check_date_resets()
        now = datetime.now(timezone.utc)
        
        result = {
            'allowed': True,
            'reason': 'OK',
            'daily_pnl': self.daily_pnl,
            'weekly_pnl': self.weekly_pnl,
            'daily_pnl_pct': self.daily_pnl / initial_balance if initial_balance > 0 else 0,
            'weekly_pnl_pct': self.weekly_pnl / initial_balance if initial_balance > 0 else 0,
            'cooldown_active': False,
            'cooldown_remaining': 0
        }
        
        # Check cooldown
        if self.cooldown_active and self.cooldown_until:
            if now < self.cooldown_until:
                remaining = (self.cooldown_until - now).total_seconds() / 60
                result['allowed'] = False
                result['reason'] = f"Cooldown active ({remaining:.0f}min remaining after {self.consecutive_losses} losses)"
                result['cooldown_active'] = True
                result['cooldown_remaining'] = int(remaining)
                return result
            else:
                self.cooldown_active = False
                self.cooldown_until = None
                logger.info("Cooldown ended - trading resumed")
        
        # Check daily loss limit
        daily_loss_pct = abs(self.daily_pnl / initial_balance) if initial_balance > 0 else 0
        if self.daily_pnl < 0 and daily_loss_pct >= self.daily_loss_limit:
            self.daily_limit_hit = True
            result['allowed'] = False
            result['reason'] = f"Daily loss limit hit ({daily_loss_pct:.1%} >= {self.daily_loss_limit:.1%})"
            return result
        
        # Check weekly loss limit
        weekly_loss_pct = abs(self.weekly_pnl / initial_balance) if initial_balance > 0 else 0
        if self.weekly_pnl < 0 and weekly_loss_pct >= self.weekly_loss_limit:
            self.weekly_limit_hit = True
            result['allowed'] = False
            result['reason'] = f"Weekly loss limit hit ({weekly_loss_pct:.1%} >= {self.weekly_loss_limit:.1%})"
            return result
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            if not self.cooldown_active:
                self._activate_cooldown(now)
            result['allowed'] = False
            result['reason'] = f"Max consecutive losses reached ({self.consecutive_losses})"
            return result
        
        return result
    
    def get_adjusted_risk(
        self,
        balance: float,
        peak_balance: float,
        volatility_pct: float = 1.0,
        ai_confidence: float = 0.0
    ) -> Dict[str, Any]:
        """
        Calculate dynamically adjusted risk percentage.
        
        Factors:
        - Kelly Criterion based on historical performance
        - Current drawdown level
        - Win/loss streak
        - Market volatility
        - AI confidence level
        """
        self._check_date_resets()
        
        # Start with Kelly-optimal or base risk
        if len(self.trade_outcomes) >= 10:
            kelly_risk = self.calculate_kelly_fraction()
        else:
            kelly_risk = self.base_risk_pct
        
        # === Factor 1: Drawdown Adjustment ===
        drawdown = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
        drawdown_pct = drawdown * 100
        
        if drawdown_pct >= 20:
            dd_multiplier = 0.25
            dd_mode = 'EMERGENCY'
        elif drawdown_pct >= 15:
            dd_multiplier = 0.4
            dd_mode = 'SEVERE'
        elif drawdown_pct >= 10:
            dd_multiplier = 0.6
            dd_mode = 'CAUTIOUS'
        elif drawdown_pct >= 5:
            dd_multiplier = 0.8
            dd_mode = 'REDUCED'
        else:
            dd_multiplier = 1.0
            dd_mode = 'NORMAL'
        
        # === Factor 2: Streak Adjustment ===
        if self.consecutive_losses >= 3:
            streak_multiplier = 0.5
        elif self.consecutive_losses >= 2:
            streak_multiplier = 0.75
        elif self.consecutive_losses >= 1:
            streak_multiplier = 0.9
        elif self.consecutive_wins >= 5:
            streak_multiplier = 1.15  # Slight boost on hot streak
        elif self.consecutive_wins >= 3:
            streak_multiplier = 1.1
        else:
            streak_multiplier = 1.0
        
        # === Factor 3: Volatility Adjustment ===
        # High volatility = lower risk
        if volatility_pct > 2.0:
            vol_multiplier = 0.7
        elif volatility_pct > 1.5:
            vol_multiplier = 0.85
        elif volatility_pct < 0.5:
            vol_multiplier = 0.8  # Too quiet = suspicious
        else:
            vol_multiplier = 1.0
        
        # === Factor 4: AI Confidence Adjustment ===
        # Higher AI confidence = slightly higher risk allowed
        if ai_confidence >= 0.9:
            ai_multiplier = 1.1
        elif ai_confidence >= 0.8:
            ai_multiplier = 1.05
        elif ai_confidence < 0.6:
            ai_multiplier = 0.8
        else:
            ai_multiplier = 1.0
        
        # === Combine Factors ===
        # RISK SCALING LOGIC:
        # - Multiple reducing factors: Use the MOST restrictive (minimum multiplier)
        #   This prevents compounding effects from making position sizes too small
        # - Multiple increasing factors: Use the MOST conservative increase (minimum multiplier)
        # - Example: If DD=0.6 and streak=0.75, use 0.6 (most restrictive)
        # - This is intentional to prevent over-reduction but ensures at least one
        #   major risk factor is fully accounted for
        
        reducing_factors = [f for f in [dd_multiplier, streak_multiplier, vol_multiplier] if f < 1.0]
        increasing_factors = [f for f in [streak_multiplier, ai_multiplier] if f > 1.0]
        
        if reducing_factors:
            combined = min(reducing_factors)  # Most restrictive wins
        elif increasing_factors:
            combined = min(increasing_factors)  # Conservative increase
        else:
            combined = 1.0
        
        adjusted_risk = kelly_risk * combined
        adjusted_risk = float(np.clip(adjusted_risk, self.min_risk_pct, self.max_risk_pct))
        
        return {
            'base_risk': self.base_risk_pct,
            'kelly_risk': kelly_risk,
            'adjusted_risk': adjusted_risk,
            'dd_mode': dd_mode,
            'drawdown_pct': round(drawdown_pct, 2),
            'dd_multiplier': dd_multiplier,
            'streak_multiplier': streak_multiplier,
            'vol_multiplier': vol_multiplier,
            'ai_multiplier': ai_multiplier,
            'combined_multiplier': combined,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'win_rate': self.win_rate,
            'message': f"{dd_mode}: {adjusted_risk:.2%} risk (DD:{drawdown_pct:.1f}%, W{self.consecutive_wins}/L{self.consecutive_losses})"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get risk manager statistics."""
        recent_trades = self.trade_outcomes[-20:] if self.trade_outcomes else []
        
        return {
            'total_trades': len(self.trade_outcomes),
            'win_rate': round(self.win_rate * 100, 1),
            'avg_win_pct': round(self.avg_win * 100, 2),
            'avg_loss_pct': round(self.avg_loss * 100, 2),
            'kelly_fraction': round(self.calculate_kelly_fraction() * 100, 2) if len(self.trade_outcomes) >= 10 else 0.0,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'daily_pnl': round(self.daily_pnl, 2),
            'weekly_pnl': round(self.weekly_pnl, 2),
            'daily_limit_hit': self.daily_limit_hit,
            'weekly_limit_hit': self.weekly_limit_hit,
            'cooldown_active': self.cooldown_active,
            'recent_trades': len(recent_trades)
        }
    
    def reset_daily(self):
        """Manually reset daily limits."""
        self.daily_pnl = 0.0
        self.daily_limit_hit = False
        logger.info("Daily risk limits manually reset")
    
    def reset_weekly(self):
        """Manually reset weekly limits."""
        self.weekly_pnl = 0.0
        self.weekly_limit_hit = False
        logger.info("Weekly risk limits manually reset")
    
    def reset_cooldown(self):
        """Manually reset cooldown."""
        self.cooldown_active = False
        self.cooldown_until = None
        self.consecutive_losses = 0
        logger.info("Cooldown manually reset")
    
    # === PORTFOLIO-LEVEL POSITION MANAGEMENT (ML Acceleration Plan) ===
    
    def register_position(self, symbol: str, risk_pct: float):
        """Register an open position for portfolio tracking."""
        self.current_positions[symbol] = risk_pct
        logger.debug(f"Position registered: {symbol} ({risk_pct:.2%} risk)")
    
    def unregister_position(self, symbol: str):
        """Remove a closed position from portfolio tracking."""
        if symbol in self.current_positions:
            del self.current_positions[symbol]
            logger.debug(f"Position unregistered: {symbol}")
    
    def can_open_position(self, symbol: str, proposed_risk_pct: float) -> Dict[str, Any]:
        """
        Check if a new position can be opened based on portfolio limits.
        
        Checks:
        1. Max total positions not exceeded
        2. Total portfolio risk not exceeded
        3. Symbol not already in position
        
        Returns:
            Dict with 'allowed' bool and 'reason' str
        """
        result = {
            'allowed': True,
            'reason': 'OK',
            'current_positions': len(self.current_positions),
            'max_positions': self.max_total_positions,
            'current_risk_pct': sum(self.current_positions.values()),
            'proposed_risk_pct': proposed_risk_pct
        }
        
        # Check if already in position for this symbol
        if symbol in self.current_positions:
            result['allowed'] = False
            result['reason'] = f"Already in position for {symbol}"
            return result
        
        # Check max total positions
        if len(self.current_positions) >= self.max_total_positions:
            result['allowed'] = False
            result['reason'] = f"Max positions reached ({len(self.current_positions)}/{self.max_total_positions})"
            return result
        
        # Check total portfolio risk
        total_risk = sum(self.current_positions.values()) + proposed_risk_pct
        if total_risk > self.portfolio_max_risk_pct:
            result['allowed'] = False
            result['reason'] = f"Portfolio risk exceeded ({total_risk:.2%} > {self.portfolio_max_risk_pct:.2%})"
            return result
        
        return result
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get current portfolio risk status."""
        return {
            'open_positions': len(self.current_positions),
            'max_positions': self.max_total_positions,
            'positions': dict(self.current_positions),
            'total_risk_pct': sum(self.current_positions.values()),
            'max_risk_pct': self.portfolio_max_risk_pct,
            'available_slots': self.max_total_positions - len(self.current_positions),
            'can_add_position': len(self.current_positions) < self.max_total_positions
        }

    # ═══════════════════════════════════════════════════════════════════
    # REC #2: REAL-TIME CORRELATION MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════
    
    def check_correlation_gate(
        self,
        candidate_df: 'pd.DataFrame',
        existing_positions_dfs: Dict[str, 'pd.DataFrame'],
        candidate_symbol: str,
        candidate_side: str,
        max_correlation: float = 0.75,
        lookback: int = 30
    ) -> Dict[str, Any]:
        """
        Check if opening a new position would create excessive portfolio correlation.
        
        All crypto correlates with BTC to some degree. This gate prevents
        opening two positions that move in lockstep (e.g., two alt longs when
        BTC is leading the move). True diversification = low correlation OR
        opposite directions on correlated pairs.
        
        Args:
            candidate_df: OHLCV DataFrame for the candidate pair
            existing_positions_dfs: {symbol: DataFrame} for each open position
            candidate_symbol: Symbol being considered
            candidate_side: 'long' or 'short'
            max_correlation: Maximum acceptable correlation (default 0.75)
            lookback: Number of bars for rolling correlation
            
        Returns:
            {
                'allowed': bool,
                'reason': str,
                'correlations': {symbol: corr_value},
                'highest_corr': float,
                'highest_corr_symbol': str,
                'direction_diversified': bool,
                'adjusted_size_mult': float (1.0 = full, 0.5 = half for high corr)
            }
        """
        result = {
            'allowed': True,
            'reason': 'OK',
            'correlations': {},
            'highest_corr': 0.0,
            'highest_corr_symbol': '',
            'direction_diversified': False,
            'adjusted_size_mult': 1.0
        }
        
        if not existing_positions_dfs:
            return result  # No existing positions → no correlation concern
        
        try:
            # Get candidate returns
            if candidate_df is None or len(candidate_df) < lookback:
                result['reason'] = 'Insufficient data for correlation check - allowing'
                return result
            
            cand_close = candidate_df['close'].tail(lookback).pct_change().dropna()
            if len(cand_close) < 15:
                return result  # Not enough data
            
            for existing_symbol, existing_df in existing_positions_dfs.items():
                if existing_df is None or len(existing_df) < lookback:
                    continue
                
                exist_close = existing_df['close'].tail(lookback).pct_change().dropna()
                if len(exist_close) < 15:
                    continue
                
                # Align lengths
                min_len = min(len(cand_close), len(exist_close))
                corr = float(np.corrcoef(
                    cand_close.values[-min_len:],
                    exist_close.values[-min_len:]
                )[0, 1])
                
                if np.isnan(corr):
                    continue
                
                result['correlations'][existing_symbol] = round(corr, 3)
                
                if abs(corr) > result['highest_corr']:
                    result['highest_corr'] = abs(corr)
                    result['highest_corr_symbol'] = existing_symbol
            
            # === DECISION LOGIC ===
            if result['highest_corr'] >= max_correlation:
                # High correlation — check if directions provide natural hedge
                # If existing is LONG and candidate is SHORT on correlated pair = hedge = OK
                # If both same direction on correlated pair = concentrated risk = BAD
                
                # For simplicity, check if candidate side differs from existing
                # (We don't have existing side info here, so we block by default
                # and let the caller check direction diversification)
                
                result['allowed'] = False
                result['reason'] = (
                    f"High correlation with {result['highest_corr_symbol']} "
                    f"(ρ={result['highest_corr']:.2f} > {max_correlation}). "
                    f"Would create concentrated risk."
                )
                
                # EXCEPTION: If direction is different, allow with reduced size
                # (natural hedge scenario)
                result['adjusted_size_mult'] = 0.5  # If overridden, use half size
                
            elif result['highest_corr'] >= 0.5:
                # Moderate correlation — allow but reduce size
                result['adjusted_size_mult'] = 0.75
                result['reason'] = f"Moderate correlation (ρ={result['highest_corr']:.2f}) — size reduced 25%"
                logger.info(f"📊 CORRELATION: {candidate_symbol} moderate corr with "
                           f"{result['highest_corr_symbol']} (ρ={result['highest_corr']:.2f}) → 75% size")
            else:
                result['reason'] = f"Low correlation (ρ={result['highest_corr']:.2f}) — full diversification"
                logger.info(f"📊 CORRELATION: {candidate_symbol} low corr (ρ={result['highest_corr']:.2f}) → full size ✅")
            
        except Exception as e:
            logger.warning(f"⚠️ Correlation check error: {e} — allowing trade")
            result['reason'] = f'Correlation check error: {e}'
        
        return result


# Singleton instance
_risk_manager: Optional[RiskManager] = None
_risk_manager_lock = threading.Lock()


def get_risk_manager() -> RiskManager:
    """Get the global risk manager instance (thread-safe)."""
    global _risk_manager
    if _risk_manager is None:
        with _risk_manager_lock:
            if _risk_manager is None:  # Double-checked locking
                _risk_manager = RiskManager()
    return _risk_manager


def reset_risk_manager():
    """Reset the global risk manager."""
    global _risk_manager
    _risk_manager = None


# ═══════════════════════════════════════════════════════════════════════════════
# RISK SHIELD — Hedge-fund grade circuit breaker wall
# ═══════════════════════════════════════════════════════════════════════════════
#
# Architecture:
#   Signal → RiskShield.check() → if BLOCKED → return immediately, no analysis
#                                → if CLEAR  → proceed to TA / ML / AI pipeline
#
# This is NOT a "check." It is a WALL. Nothing passes without clearance.
#
# Breach Levels:
#   SOFT  — trading paused, existing positions kept, auto-recovers
#   HARD  — trading halted, emergency_liquidate() available, manual reset required
#
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ShieldBreach:
    """Record of a circuit breaker trip."""
    timestamp: datetime
    breach_type: str          # 'daily_loss' | 'weekly_loss' | 'max_drawdown' | 'tail_risk' | 'btc_crash' | 'cooldown'
    severity: str             # 'SOFT' | 'HARD'
    reason: str
    value: float              # The metric that triggered the breach
    threshold: float          # The threshold that was exceeded
    auto_recovers: bool       # True = clears on its own, False = needs manual reset


class RiskShield:
    """
    Hedge-fund grade risk shield.
    
    Sits at Phase 0 of _process_signal — BEFORE any TA, ML, or AI runs.
    Every trade must pass through check_shield_status() or it does not happen.
    
    Circuit Breakers:
      1. Daily Loss:   SOFT breach at -3% of initial balance
      2. Weekly Loss:  SOFT breach at -7% of initial balance  
      3. Max Drawdown: HARD breach at -10% from peak equity
      4. Tail Risk:    SOFT breach when tail_risk score < 30
      5. BTC Crash:    SOFT breach when BTC drops >5% in 1h (blocks LONGs only)
      6. Cooldown:     SOFT breach during loss cooldown periods
    
    HARD breaches unlock emergency_liquidate() — close all positions to USDT.
    HARD breaches require manual reset via /shield reset.
    """
    
    def __init__(
        self,
        risk_manager: RiskManager,
        daily_loss_limit: float = 0.03,      # -3% daily hard stop
        weekly_loss_limit: float = 0.07,      # -7% weekly hard stop
        max_drawdown_limit: float = 0.10,     # -10% from peak = HARD stop
        tail_risk_floor: float = 30.0,        # tail_risk score < 30 = stop
        btc_crash_threshold: float = -0.05,   # BTC -5% in 1h = block LONGs
        enable_auto_liquidate: bool = False,   # Must be explicitly enabled
    ):
        self.rm = risk_manager
        
        # === Thresholds ===
        self.daily_loss_limit = daily_loss_limit
        self.weekly_loss_limit = weekly_loss_limit
        self.max_drawdown_limit = max_drawdown_limit
        self.tail_risk_floor = tail_risk_floor
        self.btc_crash_threshold = btc_crash_threshold
        self.enable_auto_liquidate = enable_auto_liquidate
        
        # === State ===
        self.active_breaches: List[ShieldBreach] = []
        self.breach_history: List[Dict] = []
        self.hard_breach_active: bool = False
        self.hard_breach_reason: str = ""
        self.shield_engaged: bool = False           # True = all trading blocked
        self.longs_blocked: bool = False             # True = only LONGs blocked (BTC crash)
        self.last_tail_risk_score: float = 50.0
        self.last_btc_1h_change: float = 0.0
        self.last_check_time: Optional[datetime] = None
        self.total_blocks: int = 0
        self.total_hard_breaches: int = 0
        
        # Liquidation tracking
        self._liquidation_in_progress: bool = False
        self._last_liquidation_time: Optional[datetime] = None
        self._liquidation_callback: Optional[Callable] = None  # Set by bot.py
        
        # Load persisted state
        self._load_state()
        
        logger.info(
            f"🛡️ RiskShield initialized | Daily: -{daily_loss_limit:.0%} | "
            f"Weekly: -{weekly_loss_limit:.0%} | MaxDD: -{max_drawdown_limit:.0%} | "
            f"TailFloor: {tail_risk_floor} | BTC: {btc_crash_threshold:+.0%}/1h | "
            f"AutoLiquidate: {'ON' if enable_auto_liquidate else 'OFF'}"
        )
    
    def check_shield_status(
        self,
        balance: float,
        initial_balance: float,
        peak_balance: float,
        signal_side: str = "",               # "LONG" or "SHORT" — needed for BTC crash
        tail_risk_score: float = 50.0,       # From ai_filter component scores
        btc_1h_change: float = 0.0,          # BTC % change over last 1h
    ) -> Dict[str, Any]:
        """
        THE WALL. Every signal must pass through here before ANY analysis.
        
        Returns:
            {
                'allowed': bool,             # False = trade blocked
                'reason': str,               # Human-readable reason
                'severity': str,             # 'CLEAR' | 'SOFT' | 'HARD'
                'breaches': List[str],       # All active breach reasons
                'shield_engaged': bool,      # Overall shield status
                'longs_blocked': bool,       # BTC crash: only LONGs blocked
                'daily_pnl_pct': float,
                'weekly_pnl_pct': float,
                'drawdown_pct': float,
                'tail_risk_score': float,
                'btc_1h_change': float,
                'hard_breach_active': bool,
            }
        """
        now = datetime.now(timezone.utc)
        self.last_check_time = now
        self.last_tail_risk_score = tail_risk_score
        self.last_btc_1h_change = btc_1h_change
        
        # Reset date-based limits via the underlying RiskManager
        self.rm._check_date_resets()
        
        # Collect all breaches this check
        breaches: List[ShieldBreach] = []
        
        # ── 1. DAILY LOSS CHECK ────────────────────────────────────────────
        daily_pnl_pct = abs(self.rm.daily_pnl / initial_balance) if initial_balance > 0 else 0
        if self.rm.daily_pnl < 0 and daily_pnl_pct >= self.daily_loss_limit:
            breaches.append(ShieldBreach(
                timestamp=now,
                breach_type='daily_loss',
                severity='SOFT',
                reason=f"Daily loss -{daily_pnl_pct:.1%} breached -{self.daily_loss_limit:.1%} limit",
                value=daily_pnl_pct,
                threshold=self.daily_loss_limit,
                auto_recovers=True  # Resets at midnight UTC
            ))
        
        # ── 2. WEEKLY LOSS CHECK (THE BUG FIX) ────────────────────────────
        weekly_pnl_pct = abs(self.rm.weekly_pnl / initial_balance) if initial_balance > 0 else 0
        if self.rm.weekly_pnl < 0 and weekly_pnl_pct >= self.weekly_loss_limit:
            breaches.append(ShieldBreach(
                timestamp=now,
                breach_type='weekly_loss',
                severity='SOFT',
                reason=f"Weekly loss -{weekly_pnl_pct:.1%} breached -{self.weekly_loss_limit:.1%} limit",
                value=weekly_pnl_pct,
                threshold=self.weekly_loss_limit,
                auto_recovers=True  # Resets Monday UTC
            ))
        
        # ── 3. MAX DRAWDOWN CHECK (HARD BREACH) ───────────────────────────
        drawdown_pct = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
        if drawdown_pct >= self.max_drawdown_limit:
            breaches.append(ShieldBreach(
                timestamp=now,
                breach_type='max_drawdown',
                severity='HARD',
                reason=f"Max drawdown -{drawdown_pct:.1%} breached -{self.max_drawdown_limit:.1%} limit — HARD STOP",
                value=drawdown_pct,
                threshold=self.max_drawdown_limit,
                auto_recovers=False  # Requires manual /shield reset
            ))
        
        # ── 4. TAIL RISK CHECK ─────────────────────────────────────────────
        if tail_risk_score < self.tail_risk_floor:
            breaches.append(ShieldBreach(
                timestamp=now,
                breach_type='tail_risk',
                severity='SOFT',
                reason=f"Tail risk score {tail_risk_score:.0f} < {self.tail_risk_floor:.0f} floor — fat-tail danger",
                value=tail_risk_score,
                threshold=self.tail_risk_floor,
                auto_recovers=True  # Next candle may clear it
            ))
        
        # ── 5. BTC CRASH CHECK (LONGs only) ────────────────────────────────
        if btc_1h_change <= self.btc_crash_threshold:
            breaches.append(ShieldBreach(
                timestamp=now,
                breach_type='btc_crash',
                severity='SOFT',
                reason=f"BTC crashed {btc_1h_change:+.1%} in 1h (threshold: {self.btc_crash_threshold:+.1%}) — LONGs blocked",
                value=btc_1h_change,
                threshold=self.btc_crash_threshold,
                auto_recovers=True  # Clears when BTC stabilizes
            ))
        
        # ── 6. COOLDOWN CHECK ──────────────────────────────────────────────
        if self.rm.cooldown_active and self.rm.cooldown_until and now < self.rm.cooldown_until:
            remaining = (self.rm.cooldown_until - now).total_seconds() / 60
            breaches.append(ShieldBreach(
                timestamp=now,
                breach_type='cooldown',
                severity='SOFT',
                reason=f"Loss cooldown active ({remaining:.0f}min remaining, {self.rm.consecutive_losses} consecutive losses)",
                value=self.rm.consecutive_losses,
                threshold=self.rm.max_consecutive_losses,
                auto_recovers=True
            ))
        
        # ── EVALUATE BREACHES ──────────────────────────────────────────────
        self.active_breaches = breaches
        hard_breaches = [b for b in breaches if b.severity == 'HARD']
        soft_breaches = [b for b in breaches if b.severity == 'SOFT']
        btc_breaches = [b for b in breaches if b.breach_type == 'btc_crash']
        non_btc_breaches = [b for b in breaches if b.breach_type != 'btc_crash']
        
        # Determine overall shield state
        if hard_breaches:
            self.shield_engaged = True
            self.hard_breach_active = True
            self.hard_breach_reason = hard_breaches[0].reason
            severity = 'HARD'
            if not any(b.breach_type == 'max_drawdown' for b in self.active_breaches 
                      if b.timestamp < now - timedelta(seconds=5)):
                # First time hitting this hard breach
                self.total_hard_breaches += 1
                self._record_breach(hard_breaches[0])
                self._save_state()
        elif non_btc_breaches:
            self.shield_engaged = True
            self.longs_blocked = bool(btc_breaches)
            severity = 'SOFT'
            for b in non_btc_breaches:
                self._record_breach(b)
        elif btc_breaches:
            # Only BTC crash — block LONGs, allow SHORTs
            self.shield_engaged = False
            self.longs_blocked = True
            severity = 'SOFT'
            for b in btc_breaches:
                self._record_breach(b)
        else:
            # All clear — but respect previous hard breach lock
            if not self.hard_breach_active:
                self.shield_engaged = False
                self.longs_blocked = False
            severity = 'HARD' if self.hard_breach_active else 'CLEAR'
        
        # ── DETERMINE IF THIS SPECIFIC SIGNAL IS ALLOWED ───────────────────
        allowed = True
        block_reason = "OK"
        
        if self.hard_breach_active:
            allowed = False
            block_reason = f"HARD BREACH: {self.hard_breach_reason} — requires /shield reset"
        elif self.shield_engaged:
            allowed = False
            block_reason = "; ".join(b.reason for b in non_btc_breaches)
        elif self.longs_blocked and signal_side.upper() == "LONG":
            allowed = False
            block_reason = btc_breaches[0].reason if btc_breaches else "BTC crash — LONGs blocked"
        
        if not allowed:
            self.total_blocks += 1
        
        result = {
            'allowed': allowed,
            'reason': block_reason,
            'severity': severity,
            'breaches': [b.reason for b in breaches],
            'breach_types': [b.breach_type for b in breaches],
            'shield_engaged': self.shield_engaged,
            'longs_blocked': self.longs_blocked,
            'hard_breach_active': self.hard_breach_active,
            'daily_pnl_pct': -daily_pnl_pct if self.rm.daily_pnl < 0 else daily_pnl_pct,
            'weekly_pnl_pct': -weekly_pnl_pct if self.rm.weekly_pnl < 0 else weekly_pnl_pct,
            'drawdown_pct': drawdown_pct,
            'tail_risk_score': tail_risk_score,
            'btc_1h_change': btc_1h_change,
            'total_blocks': self.total_blocks,
        }
        
        return result
    
    async def emergency_liquidate(
        self,
        positions: Dict[str, Any],
        close_func: Callable,
        reason: str = "HARD BREACH — emergency liquidation"
    ) -> Dict[str, Any]:
        """
        PANIC BUTTON: Close ALL open positions immediately.
        
        Only callable when:
          1. enable_auto_liquidate is True, OR
          2. Called manually from /shield liquidate
        
        Args:
            positions: Dict of {symbol: Position} from bot.positions
            close_func: The bot's _close_position_by_symbol coroutine
            reason: Why we're liquidating
            
        Returns:
            Dict with liquidation results
        """
        if self._liquidation_in_progress:
            return {'success': False, 'reason': 'Liquidation already in progress'}
        
        self._liquidation_in_progress = True
        self._last_liquidation_time = datetime.now(timezone.utc)
        
        results = {
            'success': True,
            'reason': reason,
            'positions_closed': 0,
            'positions_failed': 0,
            'details': [],
            'timestamp': self._last_liquidation_time.isoformat()
        }
        
        logger.critical(f"🚨🚨🚨 EMERGENCY LIQUIDATION TRIGGERED: {reason}")
        
        try:
            open_positions = {sym: pos for sym, pos in positions.items() if pos is not None}
            
            if not open_positions:
                results['reason'] = "No open positions to liquidate"
                logger.info("🛡️ Emergency liquidation: no open positions")
                return results
            
            for symbol, pos in open_positions.items():
                try:
                    logger.critical(f"🚨 LIQUIDATING {symbol} ({pos.side}) — {reason}")
                    # Use a market price (0 lets the close function fetch current price)
                    await close_func(symbol, f"🚨 EMERGENCY: {reason}", 0)
                    results['positions_closed'] += 1
                    results['details'].append({
                        'symbol': symbol,
                        'side': pos.side,
                        'entry': pos.entry_price,
                        'status': 'CLOSED'
                    })
                except Exception as e:
                    logger.error(f"❌ Failed to liquidate {symbol}: {e}")
                    results['positions_failed'] += 1
                    results['details'].append({
                        'symbol': symbol,
                        'status': 'FAILED',
                        'error': str(e)
                    })
            
            if results['positions_failed'] > 0:
                results['success'] = False
                results['reason'] += f" — {results['positions_failed']} positions failed to close"
        
        except Exception as e:
            logger.critical(f"🚨 Emergency liquidation error: {e}")
            results['success'] = False
            results['reason'] = f"Liquidation error: {e}"
        finally:
            self._liquidation_in_progress = False
            self._record_breach(ShieldBreach(
                timestamp=datetime.now(timezone.utc),
                breach_type='emergency_liquidation',
                severity='HARD',
                reason=reason,
                value=results['positions_closed'],
                threshold=0,
                auto_recovers=False
            ))
            self._save_state()
        
        return results
    
    def reset_hard_breach(self) -> str:
        """
        Manual reset of a HARD breach — only way to resume trading after max drawdown.
        Called via /shield reset.
        """
        if not self.hard_breach_active:
            return "No hard breach active — shield is clear."
        
        old_reason = self.hard_breach_reason
        self.hard_breach_active = False
        self.hard_breach_reason = ""
        self.shield_engaged = False
        self.longs_blocked = False
        
        self._record_breach(ShieldBreach(
            timestamp=datetime.now(timezone.utc),
            breach_type='manual_reset',
            severity='HARD',
            reason=f"Manual reset of: {old_reason}",
            value=0,
            threshold=0,
            auto_recovers=False
        ))
        self._save_state()
        
        logger.warning(f"🛡️ HARD BREACH MANUALLY RESET: was '{old_reason}'")
        return f"Hard breach reset. Was: {old_reason}"
    
    def get_shield_status(self, balance: float = 0, initial_balance: float = 0, 
                          peak_balance: float = 0) -> Dict[str, Any]:
        """Get current shield state for display (Telegram /shield command)."""
        drawdown_pct = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
        daily_pnl_pct = abs(self.rm.daily_pnl / initial_balance) if initial_balance > 0 else 0
        weekly_pnl_pct = abs(self.rm.weekly_pnl / initial_balance) if initial_balance > 0 else 0
        
        return {
            'shield_engaged': self.shield_engaged,
            'hard_breach_active': self.hard_breach_active,
            'hard_breach_reason': self.hard_breach_reason,
            'longs_blocked': self.longs_blocked,
            'total_blocks': self.total_blocks,
            'total_hard_breaches': self.total_hard_breaches,
            'active_breaches': [b.reason for b in self.active_breaches],
            'active_breach_types': [b.breach_type for b in self.active_breaches],
            'recent_breaches': self.breach_history[-10:],
            'auto_liquidate_enabled': self.enable_auto_liquidate,
            'last_check': self.last_check_time.isoformat() if self.last_check_time else None,
            'last_tail_risk': self.last_tail_risk_score,
            'last_btc_1h_change': self.last_btc_1h_change,
            # Thresholds for display
            'thresholds': {
                'daily_loss': f"-{self.daily_loss_limit:.0%}",
                'weekly_loss': f"-{self.weekly_loss_limit:.0%}",
                'max_drawdown': f"-{self.max_drawdown_limit:.0%}",
                'tail_risk_floor': self.tail_risk_floor,
                'btc_crash': f"{self.btc_crash_threshold:+.0%}/1h",
            },
            # Current levels for display
            'levels': {
                'daily_pnl': f"{self.rm.daily_pnl:+.2f}",
                'daily_pnl_pct': f"-{daily_pnl_pct:.1%}" if self.rm.daily_pnl < 0 else f"+{daily_pnl_pct:.1%}",
                'weekly_pnl': f"{self.rm.weekly_pnl:+.2f}",
                'weekly_pnl_pct': f"-{weekly_pnl_pct:.1%}" if self.rm.weekly_pnl < 0 else f"+{weekly_pnl_pct:.1%}",
                'drawdown_pct': f"-{drawdown_pct:.1%}",
                'tail_risk': f"{self.last_tail_risk_score:.0f}",
                'btc_1h': f"{self.last_btc_1h_change:+.1%}",
            }
        }
    
    def _record_breach(self, breach: ShieldBreach):
        """Record a breach to history (deduped within 60s)."""
        now = breach.timestamp
        # Deduplicate: don't record the same breach type within 60 seconds
        for existing in self.breach_history[-5:]:
            if (existing.get('breach_type') == breach.breach_type and
                existing.get('severity') == breach.severity):
                try:
                    prev_time = datetime.fromisoformat(existing['timestamp'])
                    if (now - prev_time).total_seconds() < 60:
                        return  # Skip duplicate
                except (ValueError, TypeError):
                    pass
        
        self.breach_history.append({
            'timestamp': now.isoformat(),
            'breach_type': breach.breach_type,
            'severity': breach.severity,
            'reason': breach.reason,
            'value': round(breach.value, 4),
            'threshold': round(breach.threshold, 4),
            'auto_recovers': breach.auto_recovers,
        })
        # Keep last 100 breaches
        if len(self.breach_history) > 100:
            self.breach_history = self.breach_history[-100:]
    
    def _save_state(self):
        """Persist shield state to disk (atomic write)."""
        try:
            data = {
                'hard_breach_active': self.hard_breach_active,
                'hard_breach_reason': self.hard_breach_reason,
                'total_blocks': self.total_blocks,
                'total_hard_breaches': self.total_hard_breaches,
                'breach_history': self.breach_history[-100:],
                'enable_auto_liquidate': self.enable_auto_liquidate,
                'last_saved': datetime.now(timezone.utc).isoformat(),
                'thresholds': {
                    'daily_loss_limit': self.daily_loss_limit,
                    'weekly_loss_limit': self.weekly_loss_limit,
                    'max_drawdown_limit': self.max_drawdown_limit,
                    'tail_risk_floor': self.tail_risk_floor,
                    'btc_crash_threshold': self.btc_crash_threshold,
                }
            }
            tmp_path = SHIELD_STATE_FILE.with_suffix('.json.tmp')
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.rename(str(tmp_path), str(SHIELD_STATE_FILE))
        except Exception as e:
            logger.warning(f"Could not save shield state: {e}")
    
    def _load_state(self):
        """Load persisted shield state from disk."""
        try:
            if not SHIELD_STATE_FILE.exists():
                return
            
            with open(SHIELD_STATE_FILE, 'r') as f:
                data = json.load(f)
            
            self.hard_breach_active = data.get('hard_breach_active', False)
            self.hard_breach_reason = data.get('hard_breach_reason', '')
            self.total_blocks = data.get('total_blocks', 0)
            self.total_hard_breaches = data.get('total_hard_breaches', 0)
            self.breach_history = data.get('breach_history', [])[-100:]
            self.enable_auto_liquidate = data.get('enable_auto_liquidate', self.enable_auto_liquidate)
            
            if self.hard_breach_active:
                self.shield_engaged = True
                logger.warning(f"🛡️ HARD BREACH RESTORED from disk: {self.hard_breach_reason}")
                logger.warning(f"🛡️ Trading is BLOCKED until /shield reset")
            
            logger.info(f"🛡️ Shield state loaded: {self.total_blocks} blocks, "
                       f"{self.total_hard_breaches} hard breaches, "
                       f"hard_active={self.hard_breach_active}")
        except Exception as e:
            logger.warning(f"Could not load shield state: {e} — starting fresh")


# Shield singleton
_risk_shield: Optional[RiskShield] = None
_risk_shield_lock = threading.Lock()


def get_risk_shield(risk_manager: Optional[RiskManager] = None) -> RiskShield:
    """Get the global risk shield instance (thread-safe)."""
    global _risk_shield
    if _risk_shield is None:
        with _risk_shield_lock:
            if _risk_shield is None:
                rm = risk_manager or get_risk_manager()
                _risk_shield = RiskShield(risk_manager=rm)
    return _risk_shield


def reset_risk_shield():
    """Reset the global risk shield."""
    global _risk_shield
    _risk_shield = None
