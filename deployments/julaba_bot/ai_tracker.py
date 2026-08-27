"""
AI Decision Tracker Module for Julaba
Logs all AI decisions and their outcomes to measure AI-specific accuracy.

This module:
1. Records every AI decision (approve/reject) with context
2. Links decisions to trade outcomes when trades close
3. Calculates AI-specific accuracy metrics
4. Provides insights for improving AI prompts
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger("Julaba.AITracker")

AI_DECISIONS_FILE = Path(__file__).parent / "ai_decisions.json"


@dataclass
class AIDecision:
    """A single AI decision record."""
    decision_id: str
    timestamp: str
    symbol: str
    signal_direction: str  # LONG or SHORT
    price: float
    
    # AI decision details
    approved: bool
    confidence: float
    reasoning: str
    threshold_used: float
    
    # Context at decision time
    regime: str
    tech_score: float
    system_score: float
    ml_probability: float
    ml_confidence: str
    
    # Outcome (filled when trade closes, if approved)
    trade_opened: bool = False
    trade_outcome: Optional[str] = None  # "WIN", "LOSS", None
    trade_pnl: Optional[float] = None
    trade_exit_reason: Optional[str] = None
    trade_duration_minutes: Optional[int] = None
    
    # Validation
    was_correct: Optional[bool] = None  # True if decision aligned with outcome
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AIDecisionTracker:
    """
    Tracks AI decisions and outcomes for accuracy measurement.
    
    Accuracy Metrics:
    1. Approval Accuracy: % of approved trades that were winners
    2. Rejection Value: % of rejected trades that would have been losers
    3. Overall AI Value: Net benefit from AI filtering
    """
    
    def __init__(self, max_decisions: int = 1000):
        self.decisions: List[AIDecision] = []
        self.max_decisions = max_decisions
        self._load_decisions()
    
    def _load_decisions(self):
        """Load existing decisions from file."""
        try:
            if AI_DECISIONS_FILE.exists():
                with open(AI_DECISIONS_FILE, 'r') as f:
                    data = json.load(f)
                    self.decisions = [
                        AIDecision(**d) for d in data.get('decisions', [])
                    ]
                    logger.info(f"Loaded {len(self.decisions)} AI decisions from history")
        except Exception as e:
            logger.warning(f"Could not load AI decisions: {e}")
            self.decisions = []
    
    def _save_decisions(self):
        """Save decisions to file."""
        try:
            # Keep only recent decisions
            if len(self.decisions) > self.max_decisions:
                self.decisions = self.decisions[-self.max_decisions:]
            
            data = {
                'decisions': [d.to_dict() for d in self.decisions],
                'last_updated': datetime.utcnow().isoformat(),
                'summary': self.get_accuracy_summary()
            }
            with open(AI_DECISIONS_FILE, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save AI decisions: {e}")
    
    def record_decision(
        self,
        symbol: str,
        signal_direction: str,
        price: float,
        approved: bool,
        confidence: float,
        reasoning: str,
        threshold_used: float,
        regime: str,
        tech_score: float,
        system_score: float,
        ml_probability: float = 0.5,
        ml_confidence: str = "N/A"
    ) -> str:
        """
        Record a new AI decision.
        
        Returns:
            decision_id: Unique ID for linking to trade outcome later
        """
        decision_id = f"{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        decision = AIDecision(
            decision_id=decision_id,
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
            signal_direction=signal_direction,
            price=price,
            approved=approved,
            confidence=confidence,
            reasoning=reasoning[:500],  # Limit length
            threshold_used=threshold_used,
            regime=regime,
            tech_score=tech_score,
            system_score=system_score,
            ml_probability=ml_probability,
            ml_confidence=ml_confidence
        )
        
        self.decisions.append(decision)
        self._save_decisions()
        
        logger.info(f"📊 AI Decision recorded: {decision_id} | {'APPROVE' if approved else 'REJECT'} | {confidence:.0%}")
        
        return decision_id
    
    def record_trade_opened(self, decision_id: str):
        """Mark that a trade was opened for this decision."""
        for d in reversed(self.decisions):
            if d.decision_id == decision_id:
                d.trade_opened = True
                self._save_decisions()
                return True
        return False
    
    def record_trade_outcome(
        self,
        decision_id: str,
        outcome: str,  # "WIN" or "LOSS"
        pnl: float,
        exit_reason: str,
        duration_minutes: int
    ):
        """
        Record the outcome of a trade.
        
        Args:
            decision_id: The decision ID from record_decision
            outcome: "WIN" or "LOSS"
            pnl: Profit/loss in dollars
            exit_reason: TP1, TP2, TP3, STOP, MANUAL, etc.
            duration_minutes: How long the trade was open
        """
        for d in reversed(self.decisions):
            if d.decision_id == decision_id:
                d.trade_outcome = outcome
                d.trade_pnl = pnl
                d.trade_exit_reason = exit_reason
                d.trade_duration_minutes = duration_minutes
                
                # Calculate if AI was correct
                # For APPROVED trades: correct if outcome is WIN
                # For REJECTED trades: we can't know directly
                if d.approved:
                    d.was_correct = (outcome == "WIN")
                
                self._save_decisions()
                logger.info(f"📊 Trade outcome recorded: {decision_id} | {outcome} | ${pnl:+.2f}")
                return True
        
        logger.warning(f"Decision {decision_id} not found for outcome recording")
        return False
    
    def record_hypothetical_outcome(
        self,
        decision_id: str,
        would_have_won: bool,
        simulated_pnl: float
    ):
        """
        Record hypothetical outcome for a REJECTED signal.
        
        This is used when we track what would have happened if we took the trade.
        Helps measure AI rejection accuracy.
        """
        for d in reversed(self.decisions):
            if d.decision_id == decision_id and not d.approved:
                d.trade_outcome = "WOULD_WIN" if would_have_won else "WOULD_LOSE"
                d.trade_pnl = simulated_pnl
                # For rejections: correct if trade would have lost
                d.was_correct = not would_have_won
                
                self._save_decisions()
                logger.info(f"📊 Hypothetical outcome: {decision_id} | Would have {'WON' if would_have_won else 'LOST'}")
                return True
        return False
    
    def get_accuracy_summary(self) -> Dict[str, Any]:
        """
        Calculate AI accuracy metrics.
        
        Returns comprehensive accuracy breakdown.
        """
        total = len(self.decisions)
        approved = [d for d in self.decisions if d.approved]
        rejected = [d for d in self.decisions if not d.approved]
        
        # Approved trades with outcomes
        approved_with_outcome = [d for d in approved if d.trade_outcome is not None]
        approved_wins = [d for d in approved_with_outcome if d.trade_outcome == "WIN"]
        approved_losses = [d for d in approved_with_outcome if d.trade_outcome == "LOSS"]
        
        # Rejected with hypothetical outcomes
        rejected_with_outcome = [d for d in rejected if d.trade_outcome is not None]
        would_have_won = [d for d in rejected_with_outcome if d.trade_outcome == "WOULD_WIN"]
        would_have_lost = [d for d in rejected_with_outcome if d.trade_outcome == "WOULD_LOSE"]
        
        # Calculate metrics
        approval_accuracy = len(approved_wins) / max(1, len(approved_with_outcome))
        rejection_accuracy = len(would_have_lost) / max(1, len(rejected_with_outcome))
        
        # P&L impact
        total_pnl_approved = sum(d.trade_pnl or 0 for d in approved_with_outcome)
        avoided_loss = sum(abs(d.trade_pnl or 0) for d in would_have_lost)
        missed_wins = sum(d.trade_pnl or 0 for d in would_have_won)
        net_ai_value = total_pnl_approved + avoided_loss - missed_wins
        
        return {
            'total_decisions': total,
            'approved_count': len(approved),
            'rejected_count': len(rejected),
            'approval_rate': len(approved) / max(1, total),
            
            # Approval accuracy
            'approved_with_outcome': len(approved_with_outcome),
            'approved_wins': len(approved_wins),
            'approved_losses': len(approved_losses),
            'approval_accuracy': approval_accuracy,
            
            # Rejection accuracy
            'rejected_with_outcome': len(rejected_with_outcome),
            'would_have_won': len(would_have_won),
            'would_have_lost': len(would_have_lost),
            'rejection_accuracy': rejection_accuracy,
            
            # P&L impact
            'total_pnl_from_approved': total_pnl_approved,
            'avoided_losses': avoided_loss,
            'missed_opportunities': missed_wins,
            'net_ai_value': net_ai_value,
            
            # Overall
            'overall_accuracy': (approval_accuracy + rejection_accuracy) / 2 if (len(approved_with_outcome) + len(rejected_with_outcome)) > 0 else None
        }
    
    def get_recent_decisions(self, count: int = 10) -> List[Dict]:
        """Get the most recent decisions."""
        return [d.to_dict() for d in self.decisions[-count:]]
    
    def get_decision_by_id(self, decision_id: str) -> Optional[AIDecision]:
        """Find a decision by ID."""
        for d in reversed(self.decisions):
            if d.decision_id == decision_id:
                return d
        return None


# Singleton instance
_tracker: Optional[AIDecisionTracker] = None


def get_ai_tracker() -> AIDecisionTracker:
    """Get the global AI decision tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = AIDecisionTracker()
    return _tracker


if __name__ == "__main__":
    # Test the tracker
    logging.basicConfig(level=logging.INFO)
    
    tracker = get_ai_tracker()
    
    # Simulate some decisions
    d1 = tracker.record_decision(
        symbol="LINK/USDT",
        signal_direction="LONG",
        price=15.50,
        approved=True,
        confidence=0.85,
        reasoning="Strong trend, good volume",
        threshold_used=0.80,
        regime="TRENDING",
        tech_score=75,
        system_score=70,
        ml_probability=0.55,
        ml_confidence="MEDIUM"
    )
    
    # Simulate outcome
    tracker.record_trade_opened(d1)
    tracker.record_trade_outcome(d1, "WIN", pnl=25.50, exit_reason="TP1", duration_minutes=45)
    
    # Print summary
    summary = tracker.get_accuracy_summary()
    print("\n📊 AI Accuracy Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
