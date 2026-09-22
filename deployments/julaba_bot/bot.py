#!/usr/bin/env python3
"""
Julaba - AI-Enhanced Crypto Trading Bot
Combines the original trading strategy with AI signal filtering and Telegram notifications.
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
import fcntl
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Berlin timezone for accurate time display
BERLIN_TZ = ZoneInfo("Europe/Berlin")

def get_berlin_time() -> datetime:
    """Get current time in Berlin timezone."""
    return datetime.now(BERLIN_TZ)

def get_utc_time() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Load environment variables
load_dotenv()

# Configure logging with file output
_logging_initialized = False

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging to both console and file. Only runs once."""
    global _logging_initialized
    if _logging_initialized:
        return  # Prevent duplicate handlers
    _logging_initialized = True  # Set immediately to prevent re-entry
    
    log_dir = Path(__file__).parent
    log_file = log_dir / "julaba.log"
    
    # Custom formatter that uses Berlin timezone instead of UTC
    class BerlinFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            dt = datetime.fromtimestamp(record.created, tz=BERLIN_TZ)
            if datefmt:
                return dt.strftime(datefmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    formatter = BerlinFormatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Clear any existing handlers first to prevent duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # File handler with rotation (always DEBUG for full history)
    file_handler = RotatingFileHandler(
        log_file, 
        mode='a', 
        maxBytes=10*1024*1024,  # 10MB per file
        backupCount=5,  # Keep 5 backup files
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Configure root logger
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logging.info("="*60)
    logging.info("Log file: %s", log_file)

# Initialize logging immediately when module loads
setup_logging()

logger = logging.getLogger("Julaba")

# Import ccxt
try:
    import ccxt.pro as ccxt
except ImportError:
    import ccxt

# Import our modules
from ai_filter import AISignalFilter
from telegram_bot import get_telegram_notifier, TelegramNotifier
from indicator import (
    generate_signals,
    generate_regime_aware_signals,
    smart_btc_filter,
    detect_candlestick_patterns,
    calculate_drawdown_adjusted_risk,
    get_regime_analysis,
    ml_predict_regime,
    ml_record_trade,
    get_ml_classifier,
    calculate_rsi,
    calculate_atr,
    calculate_adx
)

# Import new enhancement modules
from risk_manager import get_risk_manager, RiskManager, get_risk_shield, RiskShield
from mtf_analyzer import get_mtf_analyzer, MultiTimeframeAnalyzer
from dashboard import get_dashboard
from chart_generator import get_chart_generator
from ml_config import get_ml_config, get_multi_pair_config, MLConfig, MultiPairConfig, TradeLogSchema
from ml_predictor import get_ml_predictor, MLPredictor
from ai_tracker import get_ai_tracker, AIDecisionTracker
from ml_historical import get_historical_model, HistoricalMLModel
from ml_live import get_live_model, LiveMLModel

# ============== Data Classes ==============

def validate_ohlcv_data(ohlcv: List, symbol: str = "UNKNOWN") -> List:
    """Validate and filter OHLCV data to prevent garbage data from causing issues.
    
    Checks for:
    - Missing/None values
    - Negative or zero prices
    - Invalid OHLC relationships (high < low)
    - Extreme price spikes (>50% in one candle)
    - Negative volume
    - Duplicate timestamps
    
    Returns: Filtered list with only valid candles
    """
    import math
    
    if not ohlcv:
        logger.warning(f"⚠️ OHLCV VALIDATION {symbol}: Empty data received")
        return []
    
    valid_candles = []
    prev_close = None
    seen_timestamps = set()
    rejected_count = 0
    
    for candle in ohlcv:
        try:
            # Check basic structure - needs at least 6 elements [timestamp, open, high, low, close, volume]
            if len(candle) < 6:
                rejected_count += 1
                continue
            
            timestamp, open_p, high, low, close, volume = candle[0], candle[1], candle[2], candle[3], candle[4], candle[5]
            
            # Skip None values
            if any(v is None for v in [timestamp, open_p, high, low, close, volume]):
                rejected_count += 1
                continue
            
            # Convert to float for validation
            open_p, high, low, close, volume = float(open_p), float(high), float(low), float(close), float(volume)
            
            # Skip NaN/Infinity
            if any(math.isnan(v) or math.isinf(v) for v in [open_p, high, low, close, volume]):
                rejected_count += 1
                continue
            
            # Skip non-positive prices
            if any(v <= 0 for v in [open_p, high, low, close]):
                rejected_count += 1
                continue
            
            # Skip negative volume
            if volume < 0:
                rejected_count += 1
                continue
            
            # Check OHLC relationship: high >= max(open, close), low <= min(open, close)
            if high < max(open_p, close) or low > min(open_p, close):
                rejected_count += 1
                continue
            
            # Check high >= low
            if high < low:
                rejected_count += 1
                continue
            
            # Check for extreme price spike (>50% from previous close)
            if prev_close is not None:
                max_change = max(abs(open_p - prev_close), abs(close - prev_close)) / prev_close * 100
                if max_change > 50:
                    # Log but don't reject - could be legitimate gap
                    logger.warning(f"⚠️ OHLCV {symbol}: Large price move {max_change:.1f}% at {timestamp}")
            
            # Skip duplicate timestamps
            if timestamp in seen_timestamps:
                rejected_count += 1
                continue
            seen_timestamps.add(timestamp)
            
            valid_candles.append(candle)
            prev_close = close
            
        except (TypeError, ValueError, IndexError) as e:
            rejected_count += 1
            continue
    
    if rejected_count > 0:
        logger.warning(f"⚠️ OHLCV VALIDATION {symbol}: Rejected {rejected_count}/{len(ohlcv)} invalid candles")
    
    return valid_candles


def validate_price(price: float, entry_price: float = None, symbol: str = "UNKNOWN", max_deviation_pct: float = 500.0) -> bool:
    """Validate that a price is reasonable and not garbage data.
    
    Returns True if price is valid, False otherwise.
    Protects against:
    - Negative prices
    - Zero prices
    - Extreme prices (> max_deviation_pct% from entry) - default 500%
    - NaN/Infinity values
    
    NOTE: 500% default allows meme coins to 5x without blocking exits.
    The real phantom trade protection is in Position.__post_init__ and
    _record_trade_safe which block $0 entries/exits.
    """
    import math
    
    # Basic checks
    if price is None:
        logger.error(f"❌ PRICE VALIDATION FAILED {symbol}: price is None")
        return False
    
    if not isinstance(price, (int, float)):
        logger.error(f"❌ PRICE VALIDATION FAILED {symbol}: price is not numeric: {type(price)}")
        return False
    
    if math.isnan(price) or math.isinf(price):
        logger.error(f"❌ PRICE VALIDATION FAILED {symbol}: price is NaN or Infinity: {price}")
        return False
    
    if price <= 0:
        logger.error(f"❌ PRICE VALIDATION FAILED {symbol}: price is non-positive: {price}")
        return False
    
    # If we have an entry price, check for extreme deviation
    if entry_price is not None:
        if entry_price <= 0:
            logger.error(f"❌ PRICE VALIDATION FAILED {symbol}: entry_price is {entry_price} (zero/negative) - PHANTOM TRADE DETECTED")
            return False
        pnl_pct = abs((price - entry_price) / entry_price) * 100
        # Reject if price deviates more than max_deviation_pct (default 50%)
        if pnl_pct > max_deviation_pct:
            logger.error(f"❌ PRICE VALIDATION FAILED {symbol}: price {price} is {pnl_pct:.1f}% from entry {entry_price} (max allowed: {max_deviation_pct}%) - GARBAGE DATA REJECTED")
            return False
    
    return True


def normalize_symbol(sym: str) -> str:
    """Normalize any symbol format to a consistent format (e.g., 'ETHUSDT').
    
    Handles various formats:
    - 'ETH/USDT:USDT' (ccxt futures) -> 'ETHUSDT'
    - 'ETH/USDT' (ccxt spot) -> 'ETHUSDT'
    - 'ETHUSDT' (raw) -> 'ETHUSDT'
    - 'ETHUSDT:USDT' -> 'ETHUSDT'
    
    This ensures consistent storage and comparison of symbols.
    """
    if not sym:
        return sym
    return sym.replace('/USDT:USDT', 'USDT').replace(':USDT', '').replace('/', '')


def to_futures_symbol(sym: str) -> str:
    """Convert any symbol format to ccxt futures format (e.g., 'ETH/USDT:USDT').
    
    This is the standard format used by ccxt for Bybit USDT perpetuals.
    """
    if not sym:
        return sym
    # First normalize to base format
    base = normalize_symbol(sym)
    # Extract the coin name (remove USDT suffix)
    coin = base.replace('USDT', '')
    return f"{coin}/USDT:USDT"


@dataclass
class Position:
    """Represents an open trading position."""
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    size: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    trailing_stop: Optional[float] = None
    original_r_value: float = 0.0  # Store original risk value for trailing (don't recalculate after BE)
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    entry_df_snapshot: Optional[pd.DataFrame] = None  # For ML learning
    ml_entry_features: Optional[Dict] = None  # For dual ML learning at close time
    entry_component_scores: Optional[Dict] = None  # For Bayesian weight learning at close time
    # Peak profit tracking for intelligent reversal detection (like Bybit display)
    peak_profit_pct: float = 0.0  # Peak profit in percentage (primary metric)
    peak_profit_usd: float = 0.0  # Peak profit in USD (secondary, for logging)
    # Loss depth tracking for real-time recovery monitoring (negative side of peak tracking)
    deepest_loss_pct: float = 0.0  # Deepest loss in percentage (tracks worst drawdown)
    deepest_loss_usd: float = 0.0  # Deepest loss in USD
    recovery_attempts: int = 0  # Count how many times we tried to recover
    last_recovery_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # Prevent spam
    in_recovery_mode: bool = False  # Flag when actively monitoring recovery
    # Real-time price tracking for accurate exit price on manual closes
    last_known_price: float = 0.0  # Updated every tick - used as exit price if externally closed
    last_price_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # When price was last updated
    # Dynamic TP info — stores the math-calculated TP percentages for this trade
    dynamic_tp_pct: Dict = field(default_factory=dict)  # {'tp1_pct': 0.25, 'tp2_pct': 0.60, ...}
    
    def __post_init__(self):
        """Validate position data on creation - PREVENTS PHANTOM TRADES."""
        if self.entry_price <= 0:
            raise ValueError(f"🚨 PHANTOM TRADE BLOCKED: Cannot create position with entry_price={self.entry_price} for {self.symbol}")
        if self.size <= 0:
            raise ValueError(f"🚨 PHANTOM TRADE BLOCKED: Cannot create position with size={self.size} for {self.symbol}")
    
    @property
    def remaining_size(self) -> float:
        """Calculate remaining position size.
        
        Uses actual TP_PCT values: TP1=50%, TP2=30%, TP3=20%
        """
        closed = 0.0
        if self.tp1_hit:
            closed += 0.5  # TP1_PCT = 50%
        if self.tp2_hit:
            closed += 0.3  # TP2_PCT = 30%
        if self.tp3_hit:
            closed += 0.2  # TP3_PCT = 20%
        return self.size * (1 - closed)
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.side == "long":
            return (current_price - self.entry_price) * self.remaining_size
        else:
            return (self.entry_price - current_price) * self.remaining_size


@dataclass 
class TradeStats:
    """Track trading statistics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    today_pnl: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc).replace(hour=0, minute=0, second=0))
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    def record_trade(self, pnl: float, is_win: bool):
        """Record a completed trade in statistics."""
        self.total_trades += 1
        self.total_pnl += pnl
        self.today_pnl += pnl
        
        if is_win:
            self.winning_trades += 1
            if pnl > self.max_win:
                self.max_win = pnl
        else:
            self.losing_trades += 1
            if pnl < self.max_loss:
                self.max_loss = pnl


# ============== Dry-Run Shadow Tracker ==============

class DryRunTracker:
    """Shadow-tracks positions that WOULD have been opened in dry-run mode.
    
    Unlike paper mode (which simulates full execution with balance updates),
    dry-run mode is a pure observation layer:
    - Creates shadow positions when signals fire
    - Monitors SL/TP hits against live prices
    - Calculates what P&L would have been
    - Persists a journal for post-hoc analysis
    - Does NOT touch real balance, stats, or position state
    
    Use case: Enable dry-run ON TOP of paper mode to A/B test new logic
    without risking even your paper balance.
    """
    
    JOURNAL_FILE = Path(__file__).parent / "dry_run_journal.json"
    MAX_HISTORY = 200  # Keep last N closed shadow trades
    FEE_RATE = 0.00055  # Taker fee for realistic P&L
    
    def __init__(self):
        self.shadow_positions: Dict[str, Dict] = {}  # symbol → shadow position
        self.closed_trades: List[Dict] = []
        self.pending_notifications: List[str] = []  # Telegram messages to send
        self.stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'max_win': 0.0,
            'max_loss': 0.0,
            'started_at': datetime.now(timezone.utc).isoformat(),
        }
        self._load()
        logger.info(f"📝 DryRunTracker initialized | {len(self.shadow_positions)} open shadows | "
                    f"{self.stats['total_trades']} historical trades | PnL: ${self.stats['total_pnl']:+.2f}")
    
    def open_shadow(self, symbol: str, side: str, entry_price: float, size: float,
                    stop_loss: float, tp1: float, tp2: float, tp3: float,
                    source: str = "", ai_confidence: float = 0.0):
        """Record a shadow position (what WOULD have been opened)."""
        self.shadow_positions[symbol] = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'size': size,
            'stop_loss': stop_loss,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'tp1_hit': False,
            'tp2_hit': False,
            'opened_at': datetime.now(timezone.utc).isoformat(),
            'source': source,
            'ai_confidence': ai_confidence,
            'peak_pnl_pct': 0.0,
            'deepest_pnl_pct': 0.0,
        }
        logger.info(f"👻 SHADOW OPEN: {side.upper()} {symbol} @ ${entry_price:.4f} | "
                    f"SL: ${stop_loss:.4f} | TP1: ${tp1:.4f} | TP2: ${tp2:.4f} | TP3: ${tp3:.4f}")
        self._save()
    
    def update_shadows(self, prices: Dict[str, float]):
        """Check shadow positions against current prices. Close on SL/TP hits.
        
        Args:
            prices: {symbol: current_price} for all tracked symbols
        """
        to_close = []
        
        for symbol, shadow in self.shadow_positions.items():
            price = prices.get(symbol)
            if not price or price <= 0:
                continue
            
            side = shadow['side']
            entry = shadow['entry_price']
            
            # Calculate current P&L %
            if side == 'long':
                pnl_pct = ((price / entry) - 1) * 100
            else:
                pnl_pct = ((entry / price) - 1) * 100
            
            # Track peak / trough
            shadow['peak_pnl_pct'] = max(shadow.get('peak_pnl_pct', 0), pnl_pct)
            shadow['deepest_pnl_pct'] = min(shadow.get('deepest_pnl_pct', 0), pnl_pct)
            
            # Check TP1 (50% of position)
            if not shadow['tp1_hit']:
                if (side == 'long' and price >= shadow['tp1']) or \
                   (side == 'short' and price <= shadow['tp1']):
                    shadow['tp1_hit'] = True
                    logger.info(f"👻 SHADOW TP1 HIT: {side.upper()} {symbol} @ ${price:.4f}")
            
            # Check TP2 (30% of position)  
            if shadow['tp1_hit'] and not shadow['tp2_hit']:
                if (side == 'long' and price >= shadow['tp2']) or \
                   (side == 'short' and price <= shadow['tp2']):
                    shadow['tp2_hit'] = True
                    logger.info(f"👻 SHADOW TP2 HIT: {side.upper()} {symbol} @ ${price:.4f}")
            
            # Check TP3 (final 20% → full close)
            if shadow['tp2_hit']:
                if (side == 'long' and price >= shadow['tp3']) or \
                   (side == 'short' and price <= shadow['tp3']):
                    to_close.append((symbol, price, 'TP3 Hit'))
                    continue
            
            # Check SL → full close
            if (side == 'long' and price <= shadow['stop_loss']) or \
               (side == 'short' and price >= shadow['stop_loss']):
                to_close.append((symbol, price, 'Stop Loss'))
                continue
            
            # Check max duration (24h)
            opened_str = shadow.get('opened_at', '')
            if opened_str:
                try:
                    opened_at = datetime.fromisoformat(opened_str)
                    if opened_at.tzinfo is None:
                        opened_at = opened_at.replace(tzinfo=timezone.utc)
                    hours_held = (datetime.now(timezone.utc) - opened_at).total_seconds() / 3600
                    if hours_held >= 24:
                        to_close.append((symbol, price, f'Max Duration ({hours_held:.0f}h)'))
                        continue
                except (ValueError, TypeError):
                    pass
        
        # Close all triggered shadows
        for symbol, exit_price, reason in to_close:
            self._close_shadow(symbol, exit_price, reason)
        
        if to_close:
            self._save()
    
    def _close_shadow(self, symbol: str, exit_price: float, reason: str):
        """Close a shadow position and record the result."""
        shadow = self.shadow_positions.get(symbol)
        if not shadow:
            return
        
        entry = shadow['entry_price']
        side = shadow['side']
        size = shadow['size']
        
        # Calculate P&L with partial TPs (matching real bot logic: 50/30/20 split)
        total_pnl = 0.0
        
        # TP1 portion (50%)
        if shadow['tp1_hit']:
            tp1_size = size * 0.5
            if side == 'long':
                total_pnl += (shadow['tp1'] - entry) * tp1_size
            else:
                total_pnl += (entry - shadow['tp1']) * tp1_size
        
        # TP2 portion (30%)
        if shadow['tp2_hit']:
            tp2_size = size * 0.3
            if side == 'long':
                total_pnl += (shadow['tp2'] - entry) * tp2_size
            else:
                total_pnl += (entry - shadow['tp2']) * tp2_size
        
        # Remaining portion at exit price
        remaining_pct = 1.0
        if shadow['tp1_hit']:
            remaining_pct -= 0.5
        if shadow['tp2_hit']:
            remaining_pct -= 0.3
        remaining_size = size * remaining_pct
        
        if side == 'long':
            total_pnl += (exit_price - entry) * remaining_size
        else:
            total_pnl += (entry - exit_price) * remaining_size
        
        # Deduct fees (entry + exit)
        entry_fee = size * entry * self.FEE_RATE
        exit_fee = size * exit_price * self.FEE_RATE  # Simplified: full size fee
        total_pnl -= (entry_fee + exit_fee)
        
        pnl_pct = (total_pnl / (size * entry)) * 100
        is_win = total_pnl >= 0
        
        # Calculate hold duration
        hours_held = 0.0
        opened_str = shadow.get('opened_at', '')
        if opened_str:
            try:
                opened_at = datetime.fromisoformat(opened_str)
                if opened_at.tzinfo is None:
                    opened_at = opened_at.replace(tzinfo=timezone.utc)
                hours_held = (datetime.now(timezone.utc) - opened_at).total_seconds() / 3600
            except (ValueError, TypeError):
                pass
        
        # Record closed trade
        trade_record = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry,
            'exit_price': exit_price,
            'size': size,
            'pnl': round(total_pnl, 4),
            'pnl_pct': round(pnl_pct, 4),
            'reason': reason,
            'tp1_hit': shadow['tp1_hit'],
            'tp2_hit': shadow['tp2_hit'],
            'peak_pnl_pct': round(shadow.get('peak_pnl_pct', 0), 4),
            'deepest_pnl_pct': round(shadow.get('deepest_pnl_pct', 0), 4),
            'hours_held': round(hours_held, 2),
            'source': shadow.get('source', ''),
            'ai_confidence': shadow.get('ai_confidence', 0),
            'opened_at': shadow.get('opened_at', ''),
            'closed_at': datetime.now(timezone.utc).isoformat(),
        }
        self.closed_trades.append(trade_record)
        
        # Trim history
        if len(self.closed_trades) > self.MAX_HISTORY:
            self.closed_trades = self.closed_trades[-self.MAX_HISTORY:]
        
        # Update stats
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] = round(self.stats['total_pnl'] + total_pnl, 4)
        if is_win:
            self.stats['wins'] += 1
            self.stats['max_win'] = max(self.stats['max_win'], total_pnl)
        else:
            self.stats['losses'] += 1
            self.stats['max_loss'] = min(self.stats['max_loss'], total_pnl)
        
        # Remove from active shadows
        del self.shadow_positions[symbol]
        
        emoji = "✅" if is_win else "❌"
        logger.info(f"👻 SHADOW CLOSED {emoji}: {side.upper()} {symbol} | "
                    f"Reason: {reason} | PnL: ${total_pnl:+.2f} ({pnl_pct:+.2f}%) | "
                    f"Held: {hours_held:.1f}h | Peak: {shadow.get('peak_pnl_pct', 0):+.2f}%")
        
        # Queue Telegram notification
        win_rate = (self.stats['wins'] / self.stats['total_trades'] * 100) if self.stats['total_trades'] > 0 else 0
        self.pending_notifications.append(
            f"👻 *Shadow Close* {emoji}\n\n"
            f"Side: `{side.upper()}` {symbol}\n"
            f"Entry: `${entry:.4f}` → Exit: `${exit_price:.4f}`\n"
            f"PnL: `${total_pnl:+.2f}` ({pnl_pct:+.2f}%)\n"
            f"Reason: `{reason}`\n"
            f"Held: `{hours_held:.1f}h` | Peak: `{shadow.get('peak_pnl_pct', 0):+.2f}%`\n\n"
            f"📊 Shadow totals: {self.stats['total_trades']} trades | "
            f"{win_rate:.0f}% WR | ${self.stats['total_pnl']:+.2f}"
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get dry-run summary for Telegram display."""
        win_rate = (self.stats['wins'] / self.stats['total_trades'] * 100) if self.stats['total_trades'] > 0 else 0
        
        # Current shadow positions info
        open_shadows = []
        for sym, s in self.shadow_positions.items():
            open_shadows.append({
                'symbol': sym,
                'side': s['side'],
                'entry': s['entry_price'],
                'pnl_pct': s.get('peak_pnl_pct', 0),  # Approximate — real P&L needs live price
            })
        
        return {
            'open_count': len(self.shadow_positions),
            'open_shadows': open_shadows,
            'total_trades': self.stats['total_trades'],
            'wins': self.stats['wins'],
            'losses': self.stats['losses'],
            'win_rate': round(win_rate, 1),
            'total_pnl': self.stats['total_pnl'],
            'max_win': self.stats['max_win'],
            'max_loss': self.stats['max_loss'],
            'started_at': self.stats.get('started_at', 'unknown'),
            'recent_trades': self.closed_trades[-5:],  # Last 5 for display
        }
    
    def reset(self):
        """Reset all dry-run data."""
        self.shadow_positions.clear()
        self.closed_trades.clear()
        self.stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0.0,
            'max_win': 0.0,
            'max_loss': 0.0,
            'started_at': datetime.now(timezone.utc).isoformat(),
        }
        self._save()
        logger.info("📝 DryRunTracker reset — all shadow data cleared")
    
    def _save(self):
        """Atomic save to disk."""
        data = {
            'shadow_positions': self.shadow_positions,
            'closed_trades': self.closed_trades,
            'stats': self.stats,
            'saved_at': datetime.now(timezone.utc).isoformat(),
        }
        tmp_path = self.JOURNAL_FILE.with_suffix('.tmp')
        try:
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=2, cls=NumpyEncoder)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.rename(self.JOURNAL_FILE)
        except Exception as e:
            logger.error(f"❌ DryRunTracker save failed: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
    
    def _load(self):
        """Load from disk if exists."""
        if not self.JOURNAL_FILE.exists():
            return
        try:
            with open(self.JOURNAL_FILE, 'r') as f:
                data = json.load(f)
            self.shadow_positions = data.get('shadow_positions', {})
            self.closed_trades = data.get('closed_trades', [])
            saved_stats = data.get('stats', {})
            # Merge with defaults (in case new fields were added)
            for k, v in saved_stats.items():
                self.stats[k] = v
        except Exception as e:
            logger.warning(f"⚠️ DryRunTracker load failed: {e} — starting fresh")


# ============== WebSocket Position Stream Manager ==============

class WebSocketPositionStream:
    """
    Real-time WebSocket position streaming for instant position updates.
    
    PhD-Level Architecture:
    - Connects to Bybit private WebSocket for position updates
    - Instantly detects position changes (fills, closes, liquidations)
    - Eliminates 5-second polling delay
    - Critical for accurate PnL and fast reaction to market changes
    - Includes heartbeat/keepalive to prevent timeout disconnects
    """
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.ws_exchange = None
        self.running = False
        self._ws_task = None
        self._heartbeat_task = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._last_message_time = None
        self._heartbeat_interval = 20  # Send heartbeat every 20 seconds
        self._heartbeat_timeout = 60   # Consider dead if no response in 60 seconds
        self.last_positions = {}  # {symbol: position_data}
        self.position_callbacks = []  # Functions to call on position update
        
    async def start(self):
        """Start WebSocket position streaming."""
        if self.running:
            return
            
        try:
            import ccxt.pro as ccxtpro
            
            self.ws_exchange = ccxtpro.bybit({
                'apiKey': os.getenv('BYBIT_API_KEY'),
                'secret': os.getenv('BYBIT_API_SECRET'),
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',
                    'defaultSubType': 'linear'
                }
            })
            
            await self.ws_exchange.load_markets()
            self.running = True
            self._reconnect_attempts = 0
            
            # Start streaming in background
            self._ws_task = asyncio.create_task(self._stream_positions())
            logger.info("🔌 WebSocket POSITION stream STARTED - real-time sync enabled")
            
        except ImportError:
            logger.warning("⚠️ ccxt.pro not available - position WebSocket disabled")
            self.running = False
        except Exception as e:
            logger.error(f"🚨 Position WebSocket start failed: {e}")
            self.running = False
    
    async def stop(self):
        """Stop WebSocket streaming."""
        self.running = False
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        if self.ws_exchange:
            try:
                await self.ws_exchange.close()
            except Exception:
                pass
        logger.info("🔌 WebSocket position stream STOPPED")
    
    def register_callback(self, callback):
        """Register a function to call on position update."""
        self.position_callbacks.append(callback)
    
    async def _stream_positions(self):
        """Main WebSocket position streaming loop with heartbeat monitoring."""
        _consecutive_timeouts = 0
        while self.running:
            try:
                # Watch positions with timeout to detect stale connections
                try:
                    positions = await asyncio.wait_for(
                        self.ws_exchange.watch_positions(),
                        timeout=self._heartbeat_timeout
                    )
                except asyncio.TimeoutError:
                    _consecutive_timeouts += 1
                    # When no positions are open, Bybit sends NO data.
                    # Timeout is NORMAL — not an error. Only warn every 10th
                    # timeout (10 min) and never count as reconnect failure.
                    if _consecutive_timeouts % 10 == 1:
                        logger.debug(f"🔌 Position WebSocket idle (no open positions) — {_consecutive_timeouts} cycles")
                    # Only force reconnect if we've been timing out for 30+ min
                    # (30 consecutive timeouts × 60s each) — likely a real disconnect
                    if _consecutive_timeouts >= 30:
                        logger.warning("⚠️ Position WebSocket stale (30+ min idle) — reconnecting")
                        _consecutive_timeouts = 0
                        raise Exception("Heartbeat timeout - no data received")
                    continue
                
                # Got real data — reset timeout counter
                _consecutive_timeouts = 0
                
                # Update last message time for health monitoring
                self._last_message_time = datetime.now(timezone.utc)
                
                # Process position updates
                for pos in positions:
                    symbol = pos.get('symbol', '')
                    contracts = float(pos.get('contracts', 0))
                    
                    # Detect position changes
                    old_contracts = self.last_positions.get(symbol, {}).get('contracts', 0)
                    if contracts != old_contracts:
                        change_type = "OPENED" if contracts > 0 and old_contracts == 0 else \
                                     "CLOSED" if contracts == 0 and old_contracts != 0 else \
                                     "MODIFIED"
                        logger.info(f"🔌 POSITION UPDATE: {symbol} {change_type} | {old_contracts} → {contracts} contracts")
                        
                        # Store new state
                        self.last_positions[symbol] = {
                            'contracts': contracts,
                            'side': pos.get('side'),
                            'entryPrice': pos.get('entryPrice'),
                            'markPrice': pos.get('markPrice'),
                            'unrealizedPnl': pos.get('unrealizedPnl'),
                            'timestamp': datetime.now(timezone.utc)
                        }
                        
                        # Call registered callbacks
                        for callback in self.position_callbacks:
                            try:
                                await callback(symbol, pos, change_type)
                            except Exception as cb_err:
                                logger.error(f"Position callback error: {cb_err}")
                
                self._reconnect_attempts = 0
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._reconnect_attempts += 1
                logger.warning(f"⚠️ Position WebSocket error (attempt {self._reconnect_attempts}): {e}")
                
                if self._reconnect_attempts >= self._max_reconnect_attempts:
                    logger.error("🚨 Position WebSocket max reconnect attempts - disabling")
                    self.running = False
                    break
                
                wait_time = min(2 ** self._reconnect_attempts, 30)
                await asyncio.sleep(wait_time)
                
                # Reconnect — suppress leaked futures from old connection
                try:
                    if self.ws_exchange:
                        try:
                            await self.ws_exchange.close()
                        except Exception:
                            pass  # Old connection cleanup — suppress leaked futures
                    import ccxt.pro as ccxtpro
                    self.ws_exchange = ccxtpro.bybit({
                        'apiKey': os.getenv('BYBIT_API_KEY'),
                        'secret': os.getenv('BYBIT_API_SECRET'),
                        'enableRateLimit': True,
                        'options': {'defaultType': 'swap', 'defaultSubType': 'linear'}
                    })
                    await self.ws_exchange.load_markets()
                except Exception as reconn_err:
                    logger.error(f"🚨 Position WebSocket reconnect failed: {reconn_err}")


# ============== WebSocket Price Stream Manager ==============

class WebSocketPriceStream:
    """
    Real-time WebSocket price streaming for instant TP/SL execution.
    
    PhD-Level Architecture:
    - Connects to Bybit WebSocket for real-time ticker updates
    - Processes every price tick (not just polling every 5 seconds)
    - Instantly detects TP/SL levels being hit
    - Calls back to bot for execution
    - Includes heartbeat/keepalive to prevent timeout disconnects
    
    This ensures we NEVER miss a price spike that would have hit our TP.
    """
    
    def __init__(self, bot_instance, symbols: List[str] = None):
        """
        Initialize WebSocket manager.
        
        Args:
            bot_instance: Reference to Julaba bot for callbacks
            symbols: List of symbols to stream (e.g., ['SEI/USDT:USDT', 'SOL/USDT:USDT'])
        """
        self.bot = bot_instance
        self.symbols = symbols or []
        self.ws_exchange = None
        self.running = False
        self.last_prices = {}  # {symbol: price}
        self.price_callbacks = []  # Functions to call on price update
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._ws_task = None
        self._last_message_time = None
        self._heartbeat_timeout = 60  # Consider dead if no data in 60 seconds
        
    async def start(self):
        """Start WebSocket price streaming."""
        if self.running:
            logger.debug("WebSocket already running")
            return
            
        try:
            # Import ccxt.pro for WebSocket support
            import ccxt.pro as ccxtpro
            
            self.ws_exchange = ccxtpro.bybit({
                'apiKey': os.getenv('BYBIT_API_KEY'),
                'secret': os.getenv('BYBIT_API_SECRET'),
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',
                    'defaultSubType': 'linear'
                }
            })
            
            # CRITICAL: Load markets first for proper symbol resolution
            await self.ws_exchange.load_markets()
            
            self.running = True
            self._reconnect_attempts = 0
            
            # Start streaming in background task
            self._ws_task = asyncio.create_task(self._stream_prices())
            
            logger.info(f"🔌 WebSocket price stream STARTED for {len(self.symbols)} symbols")
            
        except ImportError:
            logger.warning("⚠️ ccxt.pro not available - WebSocket disabled, using polling")
            self.running = False
        except Exception as e:
            logger.error(f"🚨 WebSocket start failed: {e}")
            self.running = False
    
    async def stop(self):
        """Stop WebSocket streaming."""
        self.running = False
        
        if self._ws_task:
            self._ws_task.cancel()
            try:
                await self._ws_task
            except asyncio.CancelledError:
                pass
        
        if self.ws_exchange:
            try:
                await self.ws_exchange.close()
            except Exception:
                pass
        
        logger.info("🔌 WebSocket price stream STOPPED")
    
    def add_symbol(self, symbol: str):
        """Add a symbol to stream."""
        # Normalize to ccxt format
        if ':USDT' not in symbol:
            symbol_clean = symbol.replace('/', '').replace('USDT', '')
            symbol = f"{symbol_clean}/USDT:USDT"
        
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            logger.info(f"🔌 Added {symbol} to WebSocket stream")
    
    def remove_symbol(self, symbol: str):
        """Remove a symbol from stream."""
        if ':USDT' not in symbol:
            symbol_clean = symbol.replace('/', '').replace('USDT', '')
            symbol = f"{symbol_clean}/USDT:USDT"
        
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            logger.info(f"🔌 Removed {symbol} from WebSocket stream")
    
    def register_callback(self, callback):
        """Register a function to call on every price update.
        
        Callback signature: async def callback(symbol: str, price: float, timestamp: datetime)
        """
        self.price_callbacks.append(callback)
    
    async def _stream_prices(self):
        """Main WebSocket streaming loop with heartbeat monitoring."""
        while self.running:
            try:
                if not self.symbols:
                    await asyncio.sleep(1)
                    continue
                
                # Watch tickers with timeout to detect stale connections
                try:
                    tickers = await asyncio.wait_for(
                        self.ws_exchange.watch_tickers(self.symbols),
                        timeout=self._heartbeat_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning("⚠️ Price WebSocket timeout - forcing reconnect")
                    raise Exception("Heartbeat timeout - no price data received")
                
                # Update last message time for health monitoring
                self._last_message_time = datetime.now(timezone.utc)
                
                # Process each ticker update
                for symbol, ticker in tickers.items():
                    price = ticker.get('last') or ticker.get('close')
                    if not price:
                        continue
                    
                    price = float(price)
                    old_price = self.last_prices.get(symbol)
                    self.last_prices[symbol] = price
                    
                    # Call registered callbacks
                    timestamp = datetime.now(timezone.utc)
                    for callback in self.price_callbacks:
                        try:
                            await callback(symbol, price, timestamp)
                        except Exception as cb_err:
                            logger.error(f"WebSocket callback error: {cb_err}")
                    
                    # Log significant price changes (>0.1%)
                    if old_price and abs(price - old_price) / old_price > 0.001:
                        logger.debug(f"🔌 {symbol}: ${old_price:.4f} → ${price:.4f}")
                
                # Reset reconnect counter on successful iteration
                self._reconnect_attempts = 0
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._reconnect_attempts += 1
                logger.warning(f"⚠️ WebSocket error (attempt {self._reconnect_attempts}): {e}")
                
                if self._reconnect_attempts >= self._max_reconnect_attempts:
                    logger.error("🚨 WebSocket max reconnect attempts reached - disabling")
                    self.running = False
                    break
                
                # Exponential backoff
                wait_time = min(2 ** self._reconnect_attempts, 60)
                await asyncio.sleep(wait_time)
                
                # Try to reconnect — suppress leaked futures from old connection
                try:
                    if self.ws_exchange:
                        try:
                            await self.ws_exchange.close()
                        except Exception:
                            pass  # Old connection cleanup — suppress leaked futures
                    import ccxt.pro as ccxtpro
                    self.ws_exchange = ccxtpro.bybit({
                        'apiKey': os.getenv('BYBIT_API_KEY'),
                        'secret': os.getenv('BYBIT_API_SECRET'),
                        'enableRateLimit': True,
                        'options': {'defaultType': 'swap', 'defaultSubType': 'linear'}
                    })
                    # CRITICAL: Load markets for proper symbol resolution
                    await self.ws_exchange.load_markets()
                except Exception as reconn_err:
                    logger.error(f"🚨 WebSocket reconnect failed: {reconn_err}")
    
    def get_price(self, symbol: str) -> Optional[float]:
        """Get the latest cached price for a symbol."""
        if ':USDT' not in symbol:
            symbol_clean = symbol.replace('/', '').replace('USDT', '')
            symbol = f"{symbol_clean}/USDT:USDT"
        return self.last_prices.get(symbol)


# ============== Main Bot Class ==============

class Julaba:
    """
    AI-Enhanced Trading Bot with Telegram Integration.
    """
    
    # Strategy parameters (from original)
    BASE_TF = "1m"
    AGG_TF_MINUTES = 3
    ATR_PERIOD = 14
    ATR_MULT = 1.5  # Balanced SL multiplier
    MIN_STOP_PCT = 0.003  # 0.30% minimum stop distance
    # MAX_STOP_PCT is now DYNAMIC — calculated as ATR × ATR_MULT × 1.5 safety margin
    # This adapts to each coin's volatility instead of a fixed cap
    # Floor: 0.5%, Ceiling: 3.0% (prevents extreme values)
    DYNAMIC_STOP_FLOOR = 0.005   # 0.50% — never cap tighter than this
    DYNAMIC_STOP_CEILING = 0.030 # 3.00% — never wider than this (risk control)
    RISK_PCT = 0.05  # 5% risk per trade (aggressive for small account)
    # === TAKE PROFIT STRATEGY ===
    # DYNAMIC TP SYSTEM — calculates optimal TP levels per-trade using:
    # ATR (volatility), momentum (ROC), RSI, ADX, volume, S/R proximity
    # These are FALLBACK defaults only — calculate_dynamic_tp() overrides per-trade
    
    # PERCENTAGE-BASED TPs (fallback when no market data available)
    # Mini-TP (0.15%) is handled in ai_filter.py real-time exit logic
    TP1_PROFIT_PCT = 0.0025  # Fallback TP1 at +0.25%
    TP2_PROFIT_PCT = 0.0060  # Fallback TP2 at +0.60%
    TP3_PROFIT_PCT = 0.0120  # Fallback TP3 at +1.20%
    
    # Position sizing at each TP level
    TP1_PCT = 0.5   # Take 50% at TP1 (+0.40%)
    TP2_PCT = 0.3   # Take 30% at TP2 (+0.80%)
    TP3_PCT = 0.2   # Take 20% at TP3 (+1.50%)
    
    # Legacy R-based values (kept for compatibility, but using PCT now)
    TP1_R = 1.0  # Not used for main TPs anymore
    TP2_R = 1.5
    TP3_R = 2.5
    
    TRAIL_TRIGGER_R = 0.75  # Activate trailing after 0.75R profit
    TRAIL_OFFSET_R = 0.5
    MAX_POSITION_HOURS = 24  # Force close after 24 hours
    
    # === PRE-FILTER HARD RULES (AI cannot override) ===
    MIN_ATR_PCT = 0.005  # Reject if ATR < 0.5% of price
    MIN_VOLUME_MULT = 1.2  # Reject if volume < 1.2x average
    
    # === DIRECTION BIAS (from backtest analysis) ===
    # Shorts: 62.5% win rate vs Longs: 52% win rate
    SHORT_SCORE_BONUS = 5  # Add 5 points to short signal scores
    WARMUP_BARS = 50  # Matches 150 1m bars → 50 3m bars
    
    # === AI-TUNABLE PARAMETERS ===
    # These parameters can be auto-adjusted by AI based on performance
    # Format: {param: {current, min, max, step, description}}
    ADAPTIVE_PARAMS_FILE = Path(__file__).parent / "adaptive_params.json"
    ADAPTIVE_PARAMS_DEFAULTS = {
        'atr_mult': {'current': 2.0, 'min': 1.5, 'max': 4.0, 'step': 0.25, 'desc': 'Stop loss ATR multiplier'},
        'trail_trigger_r': {'current': 0.5, 'min': 0.3, 'max': 1.5, 'step': 0.1, 'desc': 'Trailing stop activation R'},
        'short_score_bonus': {'current': 5, 'min': 0, 'max': 15, 'step': 1, 'desc': 'Short signal bonus points'},
        'min_score_threshold': {'current': 30, 'min': 20, 'max': 60, 'step': 5, 'desc': 'Minimum score to pass pre-filter'},
        'divergence_threshold': {'current': 3.0, 'min': 1.0, 'max': 6.0, 'step': 0.5, 'desc': 'Divergence filter sensitivity'},
        'tp1_r': {'current': 1.0, 'min': 0.8, 'max': 1.5, 'step': 0.1, 'desc': 'First take profit R multiple (PhD: 1.0R = 65% hit rate)'},
        'tp2_r': {'current': 1.5, 'min': 1.2, 'max': 2.5, 'step': 0.1, 'desc': 'Second take profit R multiple'},
        'confidence_threshold': {'current': 0.75, 'min': 0.6, 'max': 0.90, 'step': 0.05, 'desc': 'AI confidence threshold'},
        # === AI POWER PARAMETERS (SAFETY-CAPPED) ===
        'risk_pct': {'current': 0.02, 'min': 0.01, 'max': 0.04, 'step': 0.01, 'desc': 'Risk % per trade (max 4% — AI cannot exceed)'},
        'max_leverage': {'current': 10, 'min': 3, 'max': 10, 'step': 1, 'desc': 'Maximum leverage (hard cap 10x — no exceptions)'},
        'ai_size_mult': {'current': 1.0, 'min': 0.5, 'max': 1.5, 'step': 0.1, 'desc': 'AI position size multiplier (max 1.5x)'},
        'aggressive_mode': {'current': 0, 'min': 0, 'max': 0, 'step': 1, 'desc': 'Aggressive mode DISABLED for safety'},
        'max_concurrent_positions': {'current': 1, 'min': 1, 'max': 2, 'step': 1, 'desc': 'Max simultaneous positions'},
    }
    # === SIDE FILTER: Control which directions are allowed ===
    # Values: "both", "long", "short"
    # NOTE: Config file overrides this at startup (julaba_config.json -> allowed_sides)
    ALLOWED_SIDES = "both"  # DEFAULT: Both directions (config overrides at startup)
    
    AI_AUTO_TUNE_ENABLED = True  # Master switch for AI parameter tuning
    AI_FULL_AUTONOMY = True      # NEW: Allow AI to make more aggressive decisions
    TUNE_AFTER_N_TRADES = 5  # Review parameters every N trades
    
    # Execution costs (realistic modeling)
    SLIPPAGE_PCT = 0.0005  # 0.05% slippage on market orders (Bybit has better liquidity)
    FEE_TAKER = 0.00055   # 0.055% taker fee (Bybit USDT Perpetual)
    FEE_MAKER = 0.0002    # 0.02% maker fee (Bybit USDT Perpetual)
    ROUND_TRIP_COST = 0.0012  # 0.12% total (2x taker)
    MIN_WIN_R = 1.5  # Minimum win target considering costs
    
    # === REALISTIC EXECUTION PARAMETERS ===
    # Note: Actual fees use FEE_TAKER/FEE_MAKER above (Bybit rates)
    USE_LIMIT_ORDERS = False  # Use limit orders when possible
    LIMIT_ORDER_TIMEOUT = 30  # Seconds to wait for limit fill before market
    
    # === BTC CORRELATION CRASH PROTECTION ===
    BTC_CRASH_THRESHOLD = -0.03  # -3% BTC move triggers protection
    BTC_CRASH_COOLDOWN = 3600    # 1 hour pause after BTC crash
    BTC_CHECK_INTERVAL = 60      # Check BTC every 60 seconds
    
    # Config file path for persisting settings
    CONFIG_FILE = Path(__file__).parent / "julaba_config.json"
    TRADE_HISTORY_FILE = Path(__file__).parent / "trade_history.json"
    
    @classmethod
    def _load_persisted_symbol(cls) -> Optional[str]:
        """Load persisted symbol from config file."""
        try:
            if cls.CONFIG_FILE.exists():
                with open(cls.CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    symbol = config.get('symbol')
                    if symbol:
                        logger.info(f"📍 Loaded persisted symbol: {symbol}")
                        return symbol
        except Exception as e:
            logger.debug(f"Could not load persisted symbol: {e}")
        return None
    
    def _save_persisted_symbol(self):
        """Save current symbol to config file."""
        try:
            config = {}
            if self.CONFIG_FILE.exists():
                with open(self.CONFIG_FILE, 'r') as f:
                    config = json.load(f)
            config['symbol'] = self.SYMBOL
            config['last_updated'] = datetime.now(timezone.utc).isoformat()
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"💾 Saved symbol to config: {self.SYMBOL}")
        except Exception as e:
            logger.error(f"Could not save persisted symbol: {e}")
    
    def _load_full_trade_history(self) -> List[Dict]:
        """Load full trade history from dedicated file, enriched with ai_decisions data."""
        history = []
        try:
            # 1. Load ai_decisions for enrichment data (has symbol, side, entry price)
            ai_enrichment = {}
            ai_file = Path('ai_decisions.json')
            if ai_file.exists():
                try:
                    with open(ai_file, 'r') as f:
                        ai_data = json.load(f)
                    for d in ai_data.get('decisions', []):
                        if d.get('trade_outcome'):
                            ts = d.get('timestamp', '')
                            # FIX: Only use entry price if it's actually valid (> 0)
                            # ai_decisions 'price' field is often 0 for older entries
                            _enrich_price = d.get('price', 0)
                            ai_enrichment[ts] = {
                                "symbol": d.get('symbol', 'UNKNOWN'),
                                "side": d.get('signal_direction', 'UNKNOWN').upper(),
                                "entry": _enrich_price if _enrich_price > 0 else None,
                                "confidence": d.get('confidence', 0),
                                "exit_reason": d.get('trade_exit_reason', '')
                            }
                    logger.debug(f"📜 Loaded {len(ai_enrichment)} trade enrichments from ai_decisions")
                except Exception as e:
                    logger.warning(f"Could not load ai_decisions for enrichment: {e}")
            
            # 2. Try loading from dedicated history file
            if self.TRADE_HISTORY_FILE.exists():
                logger.info(f"📜 DEBUG: Loading from {self.TRADE_HISTORY_FILE.absolute()}")
                with open(self.TRADE_HISTORY_FILE, 'r') as f:
                    raw_content = f.read()
                    logger.info(f"📜 DEBUG: Raw content length: {len(raw_content)} bytes")
                    data = json.loads(raw_content)
                
                logger.info(f"📜 DEBUG: Data type: {type(data).__name__}, len: {len(data) if hasattr(data, '__len__') else 'N/A'}")
                
                # Handle both formats: list of trades OR dict with 'recent_trades' key
                if isinstance(data, list):
                    history = data
                    logger.info(f"📜 DEBUG: Loaded {len(history)} trades as list")
                elif isinstance(data, dict):
                    # Extract trades from dict format (synced from ai_decisions)
                    history = data.get('recent_trades', [])
                    logger.info(f"📜 DEBUG: Loaded {len(history)} trades from dict")
                    logger.info(f"📜 Converted trade history from dict format")
                
                logger.info(f"📜 Loaded {len(history)} trades from history file")
            
            # 3. If empty and config has history, migrate it
            if not history and self.CONFIG_FILE.exists():
                with open(self.CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    if 'trade_history' in config:
                        history = config['trade_history']
                        logger.info(f"📜 Migrated {len(history)} trades from config file")
            
            # 4. Normalize and ENRICH trades with ai_decisions data
            normalized = []
            for trade in history:
                # Get closed_at for matching with ai_decisions
                closed_at = trade.get("closed_at", trade.get("time", ""))
                
                # Try to enrich from ai_decisions if data is missing
                enrichment = ai_enrichment.get(closed_at, {})
                
                symbol = trade.get("symbol", "UNKNOWN")
                side = trade.get("side", "UNKNOWN").upper()
                entry = trade.get("entry", 0)
                
                # If data is missing/unknown, use enrichment
                if symbol == "UNKNOWN" and enrichment.get("symbol"):
                    symbol = enrichment["symbol"]
                if side == "UNKNOWN" and enrichment.get("side"):
                    side = enrichment["side"]
                if entry == 0 and enrichment.get("entry"):
                    entry = enrichment["entry"]
                
                # Derive date/time fields from closed_at if missing
                trade_date = trade.get("date")
                entry_time = trade.get("entry_time")
                exit_time = trade.get("exit_time")
                
                if closed_at and (not trade_date or not exit_time):
                    try:
                        closed_dt = datetime.fromisoformat(closed_at.replace('Z', '+00:00'))
                        if not trade_date:
                            trade_date = closed_dt.strftime('%Y-%m-%d')
                        if not exit_time:
                            exit_time = closed_dt.strftime('%H:%M:%S')
                    except Exception:
                        pass  # Non-critical: timestamp parsing
                
                # Clean up entry_time if it's an ISO timestamp (extract just HH:MM:SS)
                if entry_time and 'T' in str(entry_time):
                    try:
                        dt = datetime.fromisoformat(str(entry_time).replace('Z', '+00:00'))
                        entry_time = dt.strftime('%H:%M:%S')
                    except Exception:
                        pass  # Non-critical: timestamp parsing
                
                # Entry time fallback to 'time' field (clean up ISO format if needed)
                if not entry_time and trade.get("time"):
                    time_val = trade.get("time")
                    # If it's an ISO timestamp, extract just the time part
                    if 'T' in str(time_val):
                        try:
                            dt = datetime.fromisoformat(time_val.replace('Z', '+00:00'))
                            entry_time = dt.strftime('%H:%M:%S')
                        except Exception:
                            entry_time = time_val  # Fallback to raw value
                    else:
                        entry_time = time_val
                
                normalized.append({
                    "symbol": symbol,
                    "side": side,
                    "entry": entry,
                    "exit": trade.get("exit", entry),  # Default to entry if no exit
                    "pnl": trade.get("pnl", 0),
                    "time": trade.get("time", ""),
                    "closed_at": closed_at,
                    "result": trade.get("result", "WIN" if trade.get("pnl", 0) >= 0 else "LOSS").upper(),
                    "exit_reason": trade.get("exit_reason", enrichment.get("exit_reason", "")),
                    "confidence": trade.get("confidence", enrichment.get("confidence", 0)),
                    # Dashboard display fields - derived from closed_at if missing
                    "date": trade_date,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "position_num": trade.get("position_num", 1),
                    "tp1_hit": trade.get("tp1_hit", False),
                    "tp2_hit": trade.get("tp2_hit", False)
                })
            
            # 5. Save enriched data back to file
            if normalized and history:
                self._save_full_trade_history(normalized)
                logger.info(f"📜 Saved enriched trade history ({len(normalized)} trades)")
            
            return normalized
        except Exception as e:
            logger.error(f"Could not load trade history: {e}")
            return []

    def _save_full_trade_history(self, history: List[Dict]):
        """Save full trade history to dedicated file (ATOMIC write)."""
        try:
            # SAFETY: Never overwrite with empty list if file has data
            if not history and self.TRADE_HISTORY_FILE.exists():
                existing_size = self.TRADE_HISTORY_FILE.stat().st_size
                if existing_size > 10:  # More than just "[]"
                    logger.warning(f"⚠️ PREVENTED overwriting {existing_size} bytes of trade history with empty list!")
                    return
            
            # ATOMIC WRITE: Write to temp file first, then rename
            # os.replace() is atomic on Linux — no partial writes on crash
            tmp_file = self.TRADE_HISTORY_FILE.with_suffix('.tmp')
            with open(tmp_file, 'w') as f:
                json.dump(history, f, indent=2)
            os.replace(tmp_file, self.TRADE_HISTORY_FILE)
            logger.debug(f"💾 Saved {len(history)} trades to history file (atomic)")
        except Exception as e:
            logger.error(f"Could not save trade history: {e}")

    async def _record_externally_closed_trade(self, pos):
        """Record a trade that was closed externally (manually on Bybit or by SL/TP).
        
        PRIORITY ORDER:
        1. Bybit closed PnL API - has actual execution price and realized PnL
        2. Bybit trades API - for manual closes not in closed PnL yet  
        3. Real-time tracked last_known_price - last resort fallback
        
        DUPLICATE PREVENTION:
        Checks _recently_closed_symbols to avoid double-recording when both
        the normal close path and position sync/WebSocket detect the same close.
        """
        # === DUPLICATE PREVENTION: Skip if this symbol was recently closed by normal path ===
        if not hasattr(self, '_recently_closed_symbols'):
            self._recently_closed_symbols = {}  # symbol_key -> timestamp
        
        symbol_raw = getattr(pos, 'symbol', 'UNKNOWN')
        symbol_key = normalize_symbol(symbol_raw)
        
        now = datetime.now(timezone.utc)
        if symbol_key in self._recently_closed_symbols:
            last_closed = self._recently_closed_symbols[symbol_key]
            age = (now - last_closed).total_seconds()
            if age < 120:  # Within 2 minutes = same close event
                logger.warning(f"⚠️ SKIPPING _record_externally_closed_trade for {symbol_raw}: already closed {age:.0f}s ago by normal path")
                return False
        
        # Mark this symbol as being closed now
        self._recently_closed_symbols[symbol_key] = now
        
        # Clean old entries (older than 5 minutes)
        for k in list(self._recently_closed_symbols.keys()):
            if (now - self._recently_closed_symbols[k]).total_seconds() > 300:
                del self._recently_closed_symbols[k]
        try:
            symbol = getattr(pos, 'symbol', 'UNKNOWN')
            entry_price = getattr(pos, 'entry_price', 0)
            size = getattr(pos, 'size', 0)
            side = getattr(pos, 'side', 'unknown')
            opened_at = getattr(pos, 'opened_at', datetime.now(timezone.utc))
            
            # === PHANTOM TRADE PREVENTION: Reject positions with $0 entry ===
            if not entry_price or entry_price <= 0:
                logger.error(f"🚨 PHANTOM TRADE BLOCKED: {symbol} has entry_price=${entry_price} - NOT recording externally closed trade")
                logger.error(f"🚨 This position was never properly opened. Cleaning up without PnL impact.")
                return False
            if not size or size <= 0:
                logger.error(f"🚨 PHANTOM TRADE BLOCKED: {symbol} has size={size} - NOT recording externally closed trade")
                return False
            
            close_price = None
            realized_pnl = 0.0
            verified_closed = False
            price_source = "unknown"
            
            # Store last_known as fallback
            last_known = getattr(pos, 'last_known_price', 0)
            last_update = getattr(pos, 'last_price_update', None)
            
            # === PRIORITY 1: Try Bybit's closed PnL endpoint FIRST ===
            # This has the ACTUAL execution price and realized PnL from the exchange
            try:
                norm_sym = normalize_symbol(symbol)  # e.g. HBARUSDT
                logger.info(f"📊 Fetching closed PnL from Bybit for {norm_sym}...")
                
                # Fetch closed PnL records from Bybit (MUST await - async exchange)
                closed_pnl = await self.exchange.privateGetV5PositionClosedPnl({
                    'category': 'linear',
                    'symbol': norm_sym,
                    'limit': 10
                })
                
                if closed_pnl and closed_pnl.get('result', {}).get('list'):
                    # Get the position open time to match the right closure
                    pos_open_time = opened_at if isinstance(opened_at, datetime) else datetime.now(timezone.utc)
                    
                    for record in closed_pnl['result']['list']:
                        # Check if this close record is AFTER our position opened
                        close_time_ms = int(record.get('updatedTime', 0))
                        close_time = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)
                        
                        if close_time > pos_open_time:
                            record_pnl = float(record.get('closedPnl', 0))
                            avg_exit_price = float(record.get('avgExitPrice', 0))
                            
                            if avg_exit_price > 0:
                                close_price = avg_exit_price
                                realized_pnl = record_pnl
                                verified_closed = True
                                price_source = "Bybit closed PnL API"
                                logger.info(f"📊 VERIFIED closed PnL from Bybit: exit=${close_price:.6f}, PnL=${realized_pnl:.2f}")
                                break
                else:
                    logger.warning(f"⚠️ Bybit closed PnL returned no records for {norm_sym}")
                            
            except Exception as pnl_err:
                logger.warning(f"⚠️ Could not fetch closed PnL from Bybit: {pnl_err}")
                
            # === PRIORITY 2: Fallback - try to get from recent trades ===
            if not verified_closed:
                try:
                    if hasattr(self.exchange, 'fetch_my_trades'):
                        ccxt_symbol = symbol if '/' in symbol else f"{symbol.replace('USDT', '')}/USDT:USDT"
                        trades = await self.exchange.fetch_my_trades(ccxt_symbol, limit=10)
                        pos_open_time = opened_at if isinstance(opened_at, datetime) else datetime.now(timezone.utc)
                        
                        for t in reversed(trades):
                            if normalize_symbol(t.get('symbol', '')) == normalize_symbol(symbol):
                                trade_time = datetime.fromisoformat(t.get('datetime', '').replace('Z', '+00:00'))
                                # Only use trades that happened AFTER position opened
                                if trade_time > pos_open_time:
                                    close_price = float(t.get('price', 0))
                                    if close_price > 0:
                                        # Calculate PnL based on direction
                                        if side.lower() == 'long':
                                            realized_pnl = (close_price - entry_price) * size
                                        else:
                                            realized_pnl = (entry_price - close_price) * size
                                        verified_closed = True
                                        price_source = "Bybit trades API"
                                        logger.info(f"📊 VERIFIED close from trades: exit=${close_price:.6f}, PnL=${realized_pnl:.2f}")
                                        break
                except Exception as trade_err:
                    logger.warning(f"⚠️ Could not fetch trades from Bybit: {trade_err}")
            
            # === PRIORITY 3: Use last_known_price as last resort ===
            # Better than nothing - at least it's a real price from when we monitored
            if not verified_closed or close_price is None:
                if last_known > 0:
                    close_price = last_known
                    if side.lower() == 'long':
                        realized_pnl = (close_price - entry_price) * size
                    else:
                        realized_pnl = (entry_price - close_price) * size
                    verified_closed = True
                    age_str = f"{(datetime.now(timezone.utc) - last_update).total_seconds():.0f}s old" if last_update else "age unknown"
                    price_source = f"last tracked price ({age_str})"
                    logger.warning(f"⚠️ Using stale tracked price: ${close_price:.6f} ({price_source})")
            
            # === PRIORITY 4: Fetch CURRENT market price as last resort ===
            # If all other sources failed, at least get the current price
            # It won't be exact but it's better than losing the trade record entirely
            if not verified_closed or close_price is None:
                try:
                    ccxt_symbol = symbol if '/' in symbol else f"{symbol.replace('USDT', '').replace(':USDT', '')}/USDT:USDT"
                    logger.warning(f"⚠️ All price sources failed for {symbol} - fetching current market price as fallback...")
                    ticker = await self.exchange.fetch_ticker(ccxt_symbol)
                    if ticker and ticker.get('last', 0) > 0:
                        close_price = ticker['last']
                        if side.lower() == 'long':
                            realized_pnl = (close_price - entry_price) * size
                        else:
                            realized_pnl = (entry_price - close_price) * size
                        verified_closed = True
                        price_source = "current market price (approx)"
                        logger.warning(f"⚠️ Using current market price as exit: ${close_price:.6f}, est. PnL=${realized_pnl:.2f} (APPROXIMATE)")
                except Exception as ticker_err:
                    logger.error(f"🚨 Could not fetch current price for {symbol}: {ticker_err}")
            
            # Final check - if STILL no price after all 4 sources, reject
            if not verified_closed or close_price is None:
                logger.error(f"🚨 Cannot record {symbol} close - ALL 4 price sources failed. Position was never monitored?")
                return False
            
            # Validate price makes sense
            price_deviation = abs(close_price - entry_price) / entry_price * 100 if entry_price > 0 else 0
            if price_deviation > 30:
                logger.error(f"🚨 REJECTING external close with {price_deviation:.1f}% price deviation for {symbol}")
                return False
            
            # Determine win/loss
            result = 'WIN' if realized_pnl > 0 else 'LOSS' if realized_pnl < 0 else 'BREAKEVEN'
            
            # Build trade record
            trade_record = {
                'symbol': symbol.replace('/USDT:USDT', 'USDT').replace('/', ''),
                'side': side.upper(),
                'entry': entry_price,
                'exit': close_price,
                'size': size,
                'pnl': realized_pnl,
                'pnl_pct': (realized_pnl / (entry_price * size) * 100) if entry_price * size > 0 else 0,
                'result': result,
                'duration_minutes': int((datetime.now(timezone.utc) - opened_at).total_seconds() / 60) if opened_at else 0,
                'entry_time': opened_at.strftime('%H:%M:%S') if isinstance(opened_at, datetime) else 'N/A',
                'exit_time': datetime.now(timezone.utc).strftime('%H:%M:%S'),
                'date': opened_at.strftime('%Y-%m-%d') if isinstance(opened_at, datetime) else datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                'opened_at': opened_at.isoformat() if isinstance(opened_at, datetime) else str(opened_at),
                'closed_at': datetime.now(timezone.utc).isoformat(),
                'close_reason': 'MANUAL_CLOSE',
                'ai_approved': False,
                'price_source': price_source,
                'notes': f'Externally closed (price from: {price_source})'
            }
            
            logger.info(f"📊 Recording external close: {symbol} {side.upper()} Entry=${entry_price:.6f} Exit=${close_price:.6f} PnL=${realized_pnl:.2f} (source: {price_source})")
            
            # Record the trade
            if self._record_trade_safe(trade_record):
                self._save_full_trade_history(self.trade_history)
                
                # Update stats
                self.stats.total_trades += 1
                if result == 'WIN':
                    self.stats.winning_trades += 1
                    self.stats.max_win = max(self.stats.max_win, realized_pnl)
                    self.consecutive_wins += 1
                    self.consecutive_losses = 0
                elif result == 'LOSS':
                    self.stats.losing_trades += 1
                    self.stats.max_loss = min(self.stats.max_loss, realized_pnl)
                    self.consecutive_losses += 1
                    self.consecutive_wins = 0
                
                self.stats.total_pnl += realized_pnl
                self.stats.today_pnl += realized_pnl
                
                # === CRITICAL FIX: Don't manually adjust balance in LIVE mode ===
                # _sync_balance_with_exchange() already set the correct balance from Bybit.
                # Doing self.balance += realized_pnl would DOUBLE-COUNT.
                if self.live_mode:
                    try:
                        await self._sync_balance_with_exchange()
                        logger.info(f"💰 Balance synced from exchange after external close: ${self.balance:.2f}")
                    except Exception as sync_err:
                        logger.warning(f"⚠️ Could not sync balance: {sync_err} - applying PnL manually")
                        self.balance += realized_pnl
                else:
                    self.balance += realized_pnl
                
                logger.info(f"📝 EXTERNALLY CLOSED TRADE RECORDED: {symbol} {side.upper()} | PnL: ${realized_pnl:+.2f} ({result})")
                
                # Send Telegram notification
                if hasattr(self, 'telegram') and self.telegram.enabled:
                    msg = (
                        f"📋 **Manually Closed Position Recorded**\n\n"
                        f"Symbol: {symbol}\n"
                        f"Side: {side.upper()}\n"
                        f"Entry: ${entry_price:.4f}\n"
                        f"Exit: ${close_price:.4f}\n"
                        f"PnL: ${realized_pnl:+.2f} ({result})\n"
                        f"Balance: ${self.balance:.2f}"
                    )
                    await self.telegram.send_message(msg)
                    
        except Exception as e:
            logger.error(f"Error recording externally closed trade: {e}")
            import traceback
            traceback.print_exc()

    def _record_trade_safe(self, trade: Dict) -> bool:
        """Record a trade to history with duplicate prevention.
        
        Prevents duplicate trades from being recorded during bot restarts or errors.
        A duplicate is defined as a trade with the same symbol, entry price, and similar PnL
        within 2 minutes of an existing trade.
        
        Returns:
            True if trade was recorded, False if it was a duplicate
        """
        try:
            symbol = trade.get('symbol', '')
            entry = trade.get('entry', trade.get('entry_price', 0))
            exit_price = trade.get('exit', trade.get('exit_price', 0))
            pnl = trade.get('pnl', 0)
            closed_at = trade.get('closed_at', '')
            
            # === PHANTOM TRADE PREVENTION: Reject trades with $0 entry or exit ===
            if not entry or entry <= 0:
                logger.error(f"🚨 PHANTOM TRADE BLOCKED in _record_trade_safe: {symbol} has entry_price=${entry} - NOT recording")
                return False
            if not exit_price or exit_price <= 0:
                logger.error(f"🚨 PHANTOM TRADE BLOCKED in _record_trade_safe: {symbol} has exit_price=${exit_price} - NOT recording")
                return False
            
            # === PNL SANITY CHECK: Prevent obviously invalid PnL values ===
            # Max expected PnL scales with balance: 20% of balance or $2K minimum
            # With 10x leverage, max single-trade PnL shouldn't exceed ~15% of balance
            MAX_VALID_PNL = max(2000, self.balance * 0.20)
            if abs(pnl) > MAX_VALID_PNL:
                logger.warning(f"⚠️ INVALID PNL BLOCKED: {symbol} ${pnl:+.2f} exceeds max ${MAX_VALID_PNL:.0f} (20% of balance)")
                return False
            
            # Parse the new trade's timestamp
            try:
                if closed_at:
                    new_time = datetime.fromisoformat(closed_at.replace('Z', '+00:00'))
                else:
                    new_time = datetime.now(timezone.utc)
            except Exception:
                new_time = datetime.now(timezone.utc)  # Fallback to now
            
            # Check for duplicates in last 10 trades
            # Use normalize_symbol for robust matching across formats
            # (SOLUSDT:USDT vs SOLUSDT vs SOL/USDT:USDT all match)
            norm_symbol = normalize_symbol(symbol)
            
            for existing in self.trade_history[-10:]:
                existing_symbol = existing.get('symbol', '')
                existing_entry = existing.get('entry', existing.get('entry_price', 0))
                existing_pnl = existing.get('pnl', 0)
                existing_closed = existing.get('closed_at', '')
                existing_side = existing.get('side', '').upper()
                
                # Must match NORMALIZED symbol
                if normalize_symbol(existing_symbol) != norm_symbol:
                    continue
                
                # Must match side
                trade_side = trade.get('side', '').upper()
                if existing_side and trade_side and existing_side != trade_side:
                    continue
                
                # Must have similar entry price (within 0.5%)
                if existing_entry > 0 and entry > 0 and abs(entry - existing_entry) / existing_entry > 0.005:
                    continue
                
                # Must be within 2 minutes — if so, it's a duplicate regardless of PnL
                # (Different exit sources give different PnL values for the same close)
                try:
                    if existing_closed:
                        existing_time = datetime.fromisoformat(existing_closed.replace('Z', '+00:00'))
                        time_diff = abs((new_time - existing_time).total_seconds())
                        if time_diff < 120:  # Within 2 minutes
                            logger.warning(f"⚠️ DUPLICATE TRADE PREVENTED: {symbol} ${pnl:+.2f} (matches {existing_symbol} ${existing_pnl:+.2f} from {time_diff:.0f}s ago)")
                            return False
                except Exception:
                    pass  # Non-critical: duplicate check time comparison
            
            # Not a duplicate - record it
            self.trade_history.append(trade)
            logger.info(f"📝 Trade recorded: {symbol} ${pnl:+.2f}")
            
            # === STAMP _recently_closed_symbols so position sync/WebSocket won't double-record ===
            # This is the DEFINITIVE guard: once a trade is recorded here, any subsequent
            # _record_externally_closed_trade call for the same symbol will be blocked.
            if not hasattr(self, '_recently_closed_symbols'):
                self._recently_closed_symbols = {}
            self._recently_closed_symbols[norm_symbol] = datetime.now(timezone.utc)
            logger.debug(f"🔒 _record_trade_safe: Stamped {norm_symbol} in _recently_closed_symbols (duplicate prevention)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in _record_trade_safe: {e}")
            # On error, still record the trade to be safe
            self.trade_history.append(trade)
            return True

    def _load_trading_state(self):
        """Load persisted trading state (balance, stats, streaks)."""
        try:
            if self.CONFIG_FILE.exists():
                with open(self.CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                
                # Load balance
                if 'balance' in config:
                    self.balance = config['balance']
                    logger.info(f"💰 Loaded persisted balance: ${self.balance:,.2f}")
                
                if 'initial_balance' in config:
                    self.initial_balance = config['initial_balance']
                
                if 'peak_balance' in config:
                    self.peak_balance = config['peak_balance']
                
                # Load balance protection threshold
                if 'balance_protection' in config:
                    self.balance_protection_threshold = config['balance_protection']
                    logger.info(f"🛡️ Balance protection threshold: ${self.balance_protection_threshold:,.2f}")
                else:
                    self.balance_protection_threshold = 320  # Default
                
                # Load stats
                if 'stats' in config:
                    s = config['stats']
                    self.stats.total_trades = s.get('total_trades', 0)
                    self.stats.winning_trades = s.get('winning_trades', 0)
                    self.stats.losing_trades = s.get('losing_trades', 0)
                    self.stats.total_pnl = s.get('total_pnl', 0.0)
                    self.stats.max_win = s.get('max_win', 0.0)
                    self.stats.max_loss = s.get('max_loss', 0.0)
                    
                    # Check if today_pnl should be reset (new day)
                    last_updated_str = config.get('last_updated', '')
                    if last_updated_str:
                        try:
                            from dateutil import parser
                            last_updated = parser.parse(last_updated_str)
                            if last_updated.date() < datetime.now(timezone.utc).date():
                                # Config is from a previous day - reset today_pnl
                                self.stats.today_pnl = 0.0
                                logger.info(f"📅 New day detected - reset today_pnl (was ${s.get('today_pnl', 0.0):+,.2f})")
                            else:
                                self.stats.today_pnl = s.get('today_pnl', 0.0)
                        except Exception as e:
                            logger.warning(f"Could not parse last_updated date: {e}")
                            self.stats.today_pnl = s.get('today_pnl', 0.0)
                    else:
                        self.stats.today_pnl = s.get('today_pnl', 0.0)
                    
                    logger.info(f"📊 Loaded persisted stats: {self.stats.total_trades} trades, ${self.stats.total_pnl:+,.2f} P&L, today: ${self.stats.today_pnl:+,.2f}")
                
                # Load streaks
                if 'consecutive_wins' in config:
                    self.consecutive_wins = config['consecutive_wins']
                if 'consecutive_losses' in config:
                    self.consecutive_losses = config['consecutive_losses']
                
                # Load weekly PnL (persisted for circuit breaker)
                if 'weekly_pnl' in config:
                    self.weekly_pnl = config['weekly_pnl']
                    logger.info(f"📊 Loaded weekly PnL: ${self.weekly_pnl:+,.2f}")
                if 'weekly_loss_reset_week' in config:
                    self.weekly_loss_reset_week = config['weekly_loss_reset_week']
                
                # Load trade history from dedicated file - this is the SOURCE OF TRUTH for stats
                self.trade_history = self._load_full_trade_history()
                
                # ALWAYS recalculate stats from trade_history (source of truth)
                today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
                wins = sum(1 for t in self.trade_history if t.get('result') == 'WIN')
                losses = sum(1 for t in self.trade_history if t.get('result') == 'LOSS')
                total_pnl = sum(t.get('pnl', 0.0) for t in self.trade_history)
                today_pnl = sum(t.get('pnl', 0.0) for t in self.trade_history if t.get('closed_at', '').startswith(today_str))
                
                # Override stats with trade_history values (source of truth)
                if len(self.trade_history) != self.stats.total_trades:
                    logger.info(f"📊 Correcting stats from trade_history: {self.stats.total_trades} → {len(self.trade_history)} trades")
                self.stats.total_trades = len(self.trade_history)
                self.stats.winning_trades = wins
                self.stats.losing_trades = losses
                self.stats.total_pnl = total_pnl
                self.stats.today_pnl = today_pnl
                
                logger.info(f"📊 Stats from trade_history: {wins}W/{losses}L, total: ${total_pnl:+,.2f}, today: ${today_pnl:+,.2f}")
                
                # === BALANCE VERIFICATION ===
                # For LIVE trading: Exchange balance is source of truth (has real fees, liquidations, etc.)
                # For PAPER trading: trade_history is source of truth
                correct_balance = self.initial_balance + total_pnl
                if abs(self.balance - correct_balance) > 0.01:  # Allow 1 cent tolerance
                    if not self.paper_mode:
                        # LIVE MODE: Trust the persisted balance (synced from exchange)
                        logger.info(f"💰 Live mode: Using persisted balance ${self.balance:,.2f} (trade calc: ${correct_balance:,.2f})")
                        # Don't override - exchange balance is truth
                    else:
                        # PAPER MODE: trade_history is source of truth
                        logger.warning(f"💰 Balance mismatch detected: ${self.balance:,.2f} → ${correct_balance:,.2f} (diff: ${self.balance - correct_balance:+.2f})")
                        logger.info(f"💰 Correcting balance based on trade_history (source of truth)")
                        self.balance = correct_balance
                
                # Load prefilter stats (ensure all required keys exist)
                if 'prefilter_stats' in config:
                    loaded_stats = config['prefilter_stats']
                    # Merge with defaults to ensure all keys exist
                    default_stats = {
                        'total_signals': 0, 'passed': 0, 'blocked_score': 0,
                        'blocked_adx_low': 0, 'blocked_adx_danger': 0, 'blocked_volume': 0,
                        'blocked_confluence': 0, 'blocked_divergence': 0, 'blocked_btc_filter': 0,
                        'raw_signals': 0, 'by_regime': {}
                    }
                    for key, default_val in default_stats.items():
                        if key not in loaded_stats:
                            loaded_stats[key] = default_val
                    self.prefilter_stats = loaded_stats
                    logger.info(f"🔍 Loaded prefilter stats: {self.prefilter_stats.get('total_signals', 0)} signals")
                
                # Load equity curve
                if 'equity_curve' in config and hasattr(self, 'equity_curve'):
                    self.equity_curve = config['equity_curve'][-100:]  # Keep last 100 points
                
                # Load persisted position (critical for restarts!)
                if 'open_position' in config and config['open_position']:
                    pos = config['open_position']
                    try:
                        # PHANTOM TRADE PREVENTION: Skip positions with $0 entry
                        if not pos.get('entry_price') or pos['entry_price'] <= 0:
                            logger.error(f"🚨 PHANTOM POSITION BLOCKED: Refusing to restore position with entry_price={pos.get('entry_price')} for {pos.get('symbol')}")
                            self.position = None
                        else:
                            self.position = Position(
                                symbol=pos['symbol'],
                                side=pos['side'],
                                entry_price=pos['entry_price'],
                                size=pos['size'],
                                stop_loss=pos['stop_loss'],
                                tp1=pos['tp1'],
                                tp2=pos['tp2'],
                                tp3=pos['tp3'],
                                tp1_hit=pos.get('tp1_hit', False),
                                tp2_hit=pos.get('tp2_hit', False),
                                tp3_hit=pos.get('tp3_hit', False),
                                trailing_stop=pos.get('trailing_stop'),
                                original_r_value=pos.get('original_r_value', abs(pos['entry_price'] - pos['stop_loss'])),
                                opened_at=datetime.fromisoformat(pos['opened_at']) if pos.get('opened_at') else datetime.now(timezone.utc)
                            )
                            logger.info(f"📈 RESTORED OPEN POSITION: {self.position.side.upper()} {self.position.symbol} @ ${self.position.entry_price:.4f}")
                            logger.info(f"   SL: ${self.position.stop_loss:.4f} | TP1: ${self.position.tp1:.4f} | Size: {self.position.size}")
                    except Exception as e:
                        logger.error(f"Failed to restore position: {e}")
                        self.position = None
                
                # === RESTORE MULTI-POSITION DICT (critical for multi-pair continuity) ===
                if 'multi_positions' in config and config['multi_positions']:
                    multi_pos_data = config['multi_positions']
                    for symbol, pos_data in multi_pos_data.items():
                        try:
                            # PHANTOM TRADE PREVENTION: Skip positions with $0 entry
                            if not pos_data.get('entry_price') or pos_data['entry_price'] <= 0:
                                logger.error(f"🚨 PHANTOM MULTI-POSITION BLOCKED: Refusing to restore {symbol} with entry_price={pos_data.get('entry_price')}")
                                self.positions[symbol] = None
                                continue
                            restored_pos = Position(
                                symbol=pos_data['symbol'],
                                side=pos_data['side'],
                                entry_price=pos_data['entry_price'],
                                size=pos_data['size'],
                                stop_loss=pos_data['stop_loss'],
                                tp1=pos_data['tp1'],
                                tp2=pos_data['tp2'],
                                tp3=pos_data['tp3'],
                                tp1_hit=pos_data.get('tp1_hit', False),
                                tp2_hit=pos_data.get('tp2_hit', False),
                                tp3_hit=pos_data.get('tp3_hit', False),
                                trailing_stop=pos_data.get('trailing_stop'),
                                original_r_value=pos_data.get('original_r_value', abs(pos_data['entry_price'] - pos_data['stop_loss'])),
                                opened_at=datetime.fromisoformat(pos_data['opened_at']) if pos_data.get('opened_at') else datetime.now(timezone.utc)
                            )
                            # Restore peak profit tracking for intelligent reversal detection
                            restored_pos.peak_profit_usd = pos_data.get('peak_profit_usd', 0)
                            restored_pos.peak_profit_pct = pos_data.get('peak_profit_pct', 0)
                            # Restore loss depth tracking for real-time recovery
                            restored_pos.deepest_loss_pct = pos_data.get('deepest_loss_pct', 0)
                            restored_pos.deepest_loss_usd = pos_data.get('deepest_loss_usd', 0)
                            restored_pos.recovery_attempts = pos_data.get('recovery_attempts', 0)
                            restored_pos.in_recovery_mode = pos_data.get('in_recovery_mode', False)
                            self.positions[symbol] = restored_pos
                            logger.info(f"📈 RESTORED MULTI-POSITION: {restored_pos.side.upper()} {symbol} @ ${restored_pos.entry_price:.4f} (Peak: ${restored_pos.peak_profit_usd:.2f}, Deepest: {restored_pos.deepest_loss_pct:.2f}%)")
                        except Exception as e:
                            logger.error(f"Failed to restore multi-position {symbol}: {e}")
                            self.positions[symbol] = None
                    
                    logger.info(f"📊 Multi-position dict restored: {len([p for p in self.positions.values() if p])} open positions")
                
                # === RESTORE POSITION SLOTS (for fixed position numbering) ===
                if 'position_slots' in config and config['position_slots']:
                    self.position_slots = config['position_slots']
                    logger.info(f"📍 Restored position slots: {self.position_slots}")
                else:
                    # Rebuild slots from open positions if not saved
                    if hasattr(self, 'position_slots'):
                        for symbol in self.positions.keys():
                            if self.positions.get(symbol):
                                self._assign_position_slot(symbol)
                
                # === RESTORE COOLDOWN STATE (prevent abuse across restarts) ===
                if 'last_switch_time' in config:
                    type(self)._last_switch_time = config['last_switch_time']
                    logger.info(f"🔄 Restored switch cooldown state")
                
                # === RESTORE ALLOWED_SIDES SETTING ===
                if 'allowed_sides' in config:
                    self.allowed_sides = config['allowed_sides']
                    logger.info(f"📊 Restored allowed_sides: {self.allowed_sides.upper()}")
                else:
                    self.allowed_sides = self.ALLOWED_SIDES  # Default from class
                
                # === RESTORE AI MODE SETTING ===
                if 'ai_mode' in config:
                    self.ai_mode = config['ai_mode']
                    logger.info(f"🤖 Restored AI mode: {self.ai_mode.upper()}")

        except Exception as e:
            logger.warning(f"Could not load trading state: {e}")
    
    def _save_trading_state(self):
        """Save trading state to config file with atomic write."""
        try:
            config = {}
            # Try to load existing config, but don't fail if corrupted
            if self.CONFIG_FILE.exists():
                try:
                    with open(self.CONFIG_FILE, 'r') as f:
                        config = json.load(f)
                except json.JSONDecodeError:
                    # Config corrupted - try backup
                    backup_file = Path("julaba_config_live_backup.json")
                    if backup_file.exists():
                        try:
                            with open(backup_file, 'r') as f:
                                config = json.load(f)
                            logger.warning("⚠️ Config corrupted, loaded from backup")
                        except Exception:
                            config = {}
                            logger.warning("⚠️ Config and backup corrupted, starting fresh")
            
            config['balance'] = self.balance
            config['initial_balance'] = self.initial_balance
            config['peak_balance'] = self.peak_balance
            config['consecutive_wins'] = self.consecutive_wins
            config['consecutive_losses'] = self.consecutive_losses
            
            # ALWAYS recalculate stats from trade_history (source of truth)
            self._recalculate_stats_from_history()
            
            # Save stats (convert to native Python types to avoid numpy int64 serialization errors)
            config['stats'] = {
                'total_trades': int(self.stats.total_trades),
                'winning_trades': int(self.stats.winning_trades),
                'losing_trades': int(self.stats.losing_trades),
                'total_pnl': float(self.stats.total_pnl),
                'today_pnl': float(self.stats.today_pnl),
                'max_win': float(self.stats.max_win),
                'max_loss': float(self.stats.max_loss)
            }
            
            # Remove legacy trade_history from config if present (now in dedicated file)
            if 'trade_history' in config:
                del config['trade_history']
            
            # Save full history to dedicated file - BUT NEVER OVERWRITE WITH EMPTY
            if hasattr(self, 'trade_history') and self.trade_history:
                 self._save_full_trade_history(self.trade_history)
            elif hasattr(self, 'trade_history') and not self.trade_history:
                 # Don't overwrite existing file with empty list!
                 logger.debug(f"💾 Skipping trade history save - memory is empty (preserving file)")

            # Save prefilter stats
            config['prefilter_stats'] = self.prefilter_stats if hasattr(self, 'prefilter_stats') else {}
            # Save equity curve (last 100 points)
            config['equity_curve'] = self.equity_curve[-100:] if hasattr(self, 'equity_curve') else []
            # Save adaptive params if they exist
            if hasattr(self, 'adaptive_params'):
                config['adaptive_params'] = self.adaptive_params
            
            # Save allowed_sides setting
            config['allowed_sides'] = getattr(self, 'allowed_sides', 'both')
            
            # Save AI mode (persists across restarts)
            config['ai_mode'] = getattr(self, 'ai_mode', 'autonomous')
            
            # Save balance protection threshold
            config['balance_protection'] = getattr(self, 'balance_protection_threshold', 280)
            
            # Persist weekly PnL for circuit breaker (survives restarts)
            config['weekly_pnl'] = getattr(self, 'weekly_pnl', 0.0)
            config['weekly_loss_reset_week'] = getattr(self, 'weekly_loss_reset_week', 0)

            # Save open position (critical for restarts!)
            if hasattr(self, 'position') and self.position:
                config['open_position'] = {
                    'symbol': self.position.symbol,
                    'side': self.position.side,
                    'entry_price': self.position.entry_price,
                    'size': self.position.size,
                    'stop_loss': self.position.stop_loss,
                    'tp1': self.position.tp1,
                    'tp2': self.position.tp2,
                    'tp3': self.position.tp3,
                    'tp1_hit': self.position.tp1_hit,
                    'tp2_hit': self.position.tp2_hit,
                    'tp3_hit': self.position.tp3_hit,
                    'trailing_stop': self.position.trailing_stop,
                    'original_r_value': getattr(self.position, 'original_r_value', 0),
                    'opened_at': self.position.opened_at.isoformat() if self.position.opened_at else None
                }
                logger.debug(f"💾 Saved open position: {self.position.side} {self.position.symbol}")
            else:
                config['open_position'] = None
            
            # === SAVE MULTI-POSITION DICT (for position replacement) ===
            if hasattr(self, 'positions') and self.positions:
                multi_positions = {}
                for symbol, pos in self.positions.items():
                    if pos is not None:
                        multi_positions[symbol] = {
                            'symbol': pos.symbol,
                            'side': pos.side,
                            'entry_price': pos.entry_price,
                            'size': pos.size,
                            'stop_loss': pos.stop_loss,
                            'tp1': pos.tp1,
                            'tp2': pos.tp2,
                            'tp3': pos.tp3,
                            'tp1_hit': pos.tp1_hit,
                            'tp2_hit': pos.tp2_hit,
                            'tp3_hit': pos.tp3_hit,
                            'trailing_stop': pos.trailing_stop,
                            'original_r_value': getattr(pos, 'original_r_value', 0),
                            'opened_at': pos.opened_at.isoformat() if pos.opened_at else None,
                            # Peak profit tracking for intelligent reversal detection
                            'peak_profit_usd': getattr(pos, 'peak_profit_usd', 0),
                            'peak_profit_pct': getattr(pos, 'peak_profit_pct', 0),
                            'deepest_loss_pct': getattr(pos, 'deepest_loss_pct', 0),
                            'deepest_loss_usd': getattr(pos, 'deepest_loss_usd', 0),
                            'recovery_attempts': getattr(pos, 'recovery_attempts', 0),
                            'in_recovery_mode': getattr(pos, 'in_recovery_mode', False)
                        }
                config['multi_positions'] = multi_positions
                logger.debug(f"💾 Saved {len(multi_positions)} open positions to multi-position dict")
            else:
                config['multi_positions'] = {}
            
            # === SAVE POSITION SLOTS (for fixed position numbering) ===
            if hasattr(self, 'position_slots') and self.position_slots:
                config['position_slots'] = self.position_slots
            else:
                config['position_slots'] = {}
            
            # === SAVE COOLDOWN STATE (prevent abuse across restarts) ===
            import time
            config['last_switch_time'] = type(self)._last_switch_time
            
            config['last_updated'] = datetime.now(timezone.utc).isoformat()
            
            # === ATOMIC WRITE: Write to temp file, then rename ===
            # This prevents corruption from interrupted writes
            temp_file = Path(str(self.CONFIG_FILE) + '.tmp')
            with open(temp_file, 'w') as f:
                json.dump(config, f, indent=2, cls=NumpyEncoder)
            temp_file.rename(self.CONFIG_FILE)  # Atomic on POSIX systems
            logger.debug(f"💾 Saved trading state: ${self.balance:,.2f}")
        except Exception as e:
            logger.error(f"Could not save trading state: {e}")
    
    def _recalculate_stats_from_history(self):
        """
        Recalculate all stats from trade_history (the source of truth).
        This ensures stats are always accurate and not double-counted.
        """
        if not hasattr(self, 'trade_history') or not self.trade_history:
            return
        
        today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        
        wins = 0
        losses = 0
        total_pnl = 0.0
        today_pnl = 0.0
        max_win = 0.0
        max_loss = 0.0
        
        for trade in self.trade_history:
            pnl = trade.get('pnl', 0.0)
            result = trade.get('result', '')
            closed_at = str(trade.get('closed_at', ''))
            
            if result == 'WIN':
                wins += 1
            elif result == 'LOSS':
                losses += 1
            
            total_pnl += pnl
            
            if closed_at.startswith(today_str):
                today_pnl += pnl
            
            if pnl > max_win:
                max_win = pnl
            if pnl < max_loss:
                max_loss = pnl
        
        # Update stats object
        self.stats.total_trades = len(self.trade_history)
        self.stats.winning_trades = wins
        self.stats.losing_trades = losses
        self.stats.total_pnl = total_pnl
        self.stats.today_pnl = today_pnl
        self.stats.max_win = max_win
        self.stats.max_loss = max_loss

    def _sync_from_ai_decisions(self):
        """
        PERSISTENCE GUARD: Log any discrepancy between trade_history and ai_decisions.
        trade_history is the SOURCE OF TRUTH (loaded earlier in _load_trading_state).
        This function only validates and logs, it does NOT override trade_history.
        """
        try:
            ai_file = Path('ai_decisions.json')
            if not ai_file.exists():
                return
            
            with open(ai_file, 'r') as f:
                ai_data = json.load(f)
            
            # Get completed trades from ai_decisions
            decisions = ai_data.get('decisions', [])
            completed = [d for d in decisions if d.get('trade_outcome') in ['WIN', 'LOSS']]
            
            if not completed:
                return
            
            # Count from ai_decisions (for logging only)
            ai_wins = sum(1 for d in completed if d.get('trade_outcome') == 'WIN')
            ai_losses = sum(1 for d in completed if d.get('trade_outcome') == 'LOSS')
            ai_total = len(completed)
            ai_pnl = sum(d.get('trade_pnl', 0) for d in completed)
            
            # Log comparison (trade_history is source of truth, already loaded)
            if ai_total != self.stats.total_trades:
                logger.info(f"📋 Note: ai_decisions has {ai_total} trades, trade_history has {self.stats.total_trades} (trade_history is source of truth)")
            
            logger.info(f"✅ PERSISTENCE GUARD: Reconciled {self.stats.total_trades} trades ({self.stats.winning_trades}W/{self.stats.losing_trades}L, ${self.stats.total_pnl:+,.2f})")
                
        except Exception as e:
            logger.error(f"Persistence guard error: {e}")

    # === ADAPTIVE PARAMETERS SYSTEM ===
    # AI can auto-tune these parameters based on performance
    
    def _load_adaptive_params(self):
        """Load adaptive parameters from file or initialize defaults."""
        try:
            if self.ADAPTIVE_PARAMS_FILE.exists():
                with open(self.ADAPTIVE_PARAMS_FILE, 'r') as f:
                    saved = json.load(f)
                # Merge with defaults (in case new params were added)
                self.adaptive_params = self.ADAPTIVE_PARAMS_DEFAULTS.copy()
                for param, values in saved.get('params', {}).items():
                    if param in self.adaptive_params:
                        self.adaptive_params[param]['current'] = values.get('current', self.adaptive_params[param]['current'])
                # Load adjustment history
                self.param_adjustment_history = saved.get('history', [])
                self.trades_since_last_tune = saved.get('trades_since_tune', 0)
                logger.info(f"🎛️ Loaded adaptive params: {len(self.adaptive_params)} tunable")
            else:
                self.adaptive_params = self.ADAPTIVE_PARAMS_DEFAULTS.copy()
                self.param_adjustment_history = []
                self.trades_since_last_tune = 0
                self._save_adaptive_params()
                logger.info("🎛️ Initialized default adaptive params")
        except Exception as e:
            logger.error(f"Failed to load adaptive params: {e}")
            self.adaptive_params = self.ADAPTIVE_PARAMS_DEFAULTS.copy()
            self.param_adjustment_history = []
            self.trades_since_last_tune = 0
    
    def _save_adaptive_params(self):
        """Save adaptive parameters to file."""
        try:
            data = {
                'params': self.adaptive_params,
                'history': self.param_adjustment_history[-50:],  # Keep last 50 adjustments
                'trades_since_tune': self.trades_since_last_tune,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }
            with open(self.ADAPTIVE_PARAMS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug("💾 Saved adaptive params")
        except Exception as e:
            logger.error(f"Failed to save adaptive params: {e}")
    
    def _apply_adaptive_params(self):
        """Apply current adaptive params to class attributes with HARD SAFETY CLAMPS."""
        try:
            p = self.adaptive_params
            
            # === HARD SAFETY CLAMPS (override any AI/JSON values) ===
            # These prevent the AI or corrupted JSON from creating dangerous conditions
            SAFE_LIMITS = {
                'risk_pct': {'max': 0.04},      # 4% max risk per trade
                'max_leverage': {'max': 10},     # 10x max leverage
                'ai_size_mult': {'max': 1.5},    # 1.5x max size multiplier
                'aggressive_mode': {'max': 0},   # Aggressive mode DISABLED
            }
            for param, limits in SAFE_LIMITS.items():
                if param in p:
                    if p[param]['current'] > limits['max']:
                        old_val = p[param]['current']
                        p[param]['current'] = limits['max']
                        p[param]['max'] = limits['max']
                        logger.warning(f"🛡️ SAFETY CLAMP: {param} {old_val} → {limits['max']} (hard limit)")
            
            self.ATR_MULT = p['atr_mult']['current']
            self.TRAIL_TRIGGER_R = p['trail_trigger_r']['current']
            self.SHORT_SCORE_BONUS = p['short_score_bonus']['current']
            self.TP1_R = p['tp1_r']['current']
            self.TP2_R = p['tp2_r']['current']
            
            # === CRITICAL: Apply risk_pct from adaptive params ===
            self.RISK_PCT = p['risk_pct']['current']
            
            # Update AI filter confidence if available
            if hasattr(self, 'ai_filter'):
                self.ai_filter.confidence_threshold = p['confidence_threshold']['current']
            
            logger.info(f"🎛️ Applied params: ATR={self.ATR_MULT}, Trail={self.TRAIL_TRIGGER_R}R, Short+{self.SHORT_SCORE_BONUS}, Risk={self.RISK_PCT*100:.0f}%")
        except Exception as e:
            logger.error(f"Failed to apply adaptive params: {e}")
    
    def get_adaptive_params_summary(self) -> str:
        """Get human-readable summary of current adaptive parameters."""
        if not hasattr(self, 'adaptive_params'):
            return "Adaptive params not initialized"
        
        lines = ["🎛️ *Adaptive Parameters*", ""]
        for name, info in self.adaptive_params.items():
            current = info['current']
            min_v, max_v = info['min'], info['max']
            lines.append(f"• {info['desc']}: `{current}` (range: {min_v}-{max_v})")
        
        lines.append("")
        lines.append(f"Trades since last tune: {self.trades_since_last_tune}/{self.TUNE_AFTER_N_TRADES}")
        lines.append(f"AI Auto-Tune: {'✅ Enabled' if self.AI_AUTO_TUNE_ENABLED else '❌ Disabled'}")
        
        return "\n".join(lines)
    
    async def _ai_suggest_param_adjustments(self) -> Dict[str, Any]:
        """Ask AI to suggest parameter adjustments based on recent performance.
        Caches results for 5 minutes to provide stable suggestions."""
        if not self.AI_AUTO_TUNE_ENABLED:
            return {'suggested': False, 'reason': 'Auto-tune disabled'}
        
        if not hasattr(self, 'ai_filter') or not self.ai_filter.use_ai:
            return {'suggested': False, 'reason': 'AI not available'}
        
        # Check cache - return cached suggestions if < 5 min old
        cache_valid_seconds = 300  # 5 minutes
        if hasattr(self, '_autotune_cache') and self._autotune_cache:
            cache_time = self._autotune_cache.get('timestamp')
            if cache_time and (datetime.now(timezone.utc) - cache_time).total_seconds() < cache_valid_seconds:
                logger.info("Using cached auto-tune suggestions (valid for 5 min)")
                return self._autotune_cache.get('result', {'suggested': False})
        
        try:
            # Gather performance data
            recent_trades = self.trade_history[-10:] if hasattr(self, 'trade_history') else []
            if len(recent_trades) < 3:
                return {'suggested': False, 'reason': 'Need at least 3 trades for analysis'}
            
            # Calculate metrics
            wins = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
            losses = len(recent_trades) - wins
            win_rate = wins / len(recent_trades) * 100
            
            sl_hits = sum(1 for t in recent_trades if t.get('exit_type', '') == 'stop_loss')
            sl_rate = sl_hits / len(recent_trades) * 100 if recent_trades else 0
            
            avg_win = sum(t.get('pnl', 0) for t in recent_trades if t.get('pnl', 0) > 0) / max(wins, 1)
            avg_loss = sum(t.get('pnl', 0) for t in recent_trades if t.get('pnl', 0) < 0) / max(losses, 1)
            
            # Shorts vs Longs performance
            longs = [t for t in recent_trades if t.get('direction', '').upper() == 'LONG']
            shorts = [t for t in recent_trades if t.get('direction', '').upper() == 'SHORT']
            long_wr = sum(1 for t in longs if t.get('pnl', 0) > 0) / max(len(longs), 1) * 100
            short_wr = sum(1 for t in shorts if t.get('pnl', 0) > 0) / max(len(shorts), 1) * 100
            
            # Current params summary
            params_str = "\n".join([
                f"- {name}: {info['current']} (range: {info['min']}-{info['max']}, step: {info['step']})"
                for name, info in self.adaptive_params.items()
            ])
            
            prompt = f"""You are an AI trading strategy optimizer. Analyze recent performance and suggest parameter adjustments.

=== CURRENT PARAMETERS ===
{params_str}

=== RECENT PERFORMANCE (last {len(recent_trades)} trades) ===
Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)
Stop Loss Hit Rate: {sl_rate:.1f}%
Avg Win: ${avg_win:.2f}
Avg Loss: ${avg_loss:.2f}
Long Win Rate: {long_wr:.1f}% ({len(longs)} trades)
Short Win Rate: {short_wr:.1f}% ({len(shorts)} trades)

=== YOUR TASK ===
Based on this data, suggest parameter changes that could improve performance.

RULES:
1. Only suggest changes within the allowed ranges
2. Use the exact step sizes specified
3. Be conservative - small incremental changes
4. If performance is good (>60% WR), suggest no changes
5. Focus on the most impactful parameter first

Respond ONLY with this JSON format:
{{"should_adjust": true/false, "reasoning": "why", "adjustments": [{{"param": "param_name", "old_value": X, "new_value": Y, "reason": "why"}}]}}"""

            # Call AI
            result_text = self.ai_filter._generate_content(prompt)
            if not result_text:
                return {'suggested': False, 'reason': 'AI returned empty response'}
            
            # Parse response
            result_text = result_text.strip()
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            result = json.loads(result_text.strip())
            ai_result = {
                'suggested': result.get('should_adjust', False),
                'reasoning': result.get('reasoning', ''),
                'adjustments': result.get('adjustments', [])
            }
            
            # Cache the result for 5 minutes
            self._autotune_cache = {
                'timestamp': datetime.now(timezone.utc),
                'result': ai_result
            }
            
            return ai_result
            
        except Exception as e:
            logger.error(f"AI param suggestion failed: {e}")
            return {'suggested': False, 'reason': f'Error: {e}'}
    
    def _apply_param_adjustment(self, param: str, new_value: float) -> bool:
        """Apply a single parameter adjustment with validation."""
        if param not in self.adaptive_params:
            logger.warning(f"Unknown param: {param}")
            return False
        
        info = self.adaptive_params[param]
        old_value = info['current']
        
        # Validate bounds
        if new_value < info['min'] or new_value > info['max']:
            logger.warning(f"Param {param} value {new_value} out of bounds [{info['min']}, {info['max']}]")
            return False
        
        # Apply change
        info['current'] = new_value
        
        # Record in history
        self.param_adjustment_history.append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'param': param,
            'old_value': old_value,
            'new_value': new_value,
            'trades_at_change': self.stats.total_trades if hasattr(self, 'stats') else 0
        })
        
        # Apply to class attributes
        self._apply_adaptive_params()
        
        # Save to disk
        self._save_adaptive_params()
        
        logger.info(f"🎛️ PARAM ADJUSTED: {param}: {old_value} → {new_value}")
        return True
    
    async def maybe_auto_tune_params(self) -> Dict[str, Any]:
        """Suggest AI parameter changes, do NOT apply."""
        if not self.AI_AUTO_TUNE_ENABLED:
            return {'adjusted': False, 'reason': 'Auto-tune disabled'}
        # Only suggest, do not apply
        result = await self._ai_suggest_param_adjustments()
        changes_list = []
        for adj in result.get('adjustments', []):
            param = adj.get('param')
            new_value = adj.get('new_value')
            old_value = adj.get('old_value')
            if param and new_value is not None:
                changes_list.append({'param': param, 'old': old_value, 'new': new_value})
        return {
            'adjusted': len(changes_list) > 0,
            'changes': changes_list,
            'reasoning': result.get('reasoning', 'Performance optimization'),
            'summary': [f"{c['param']}: {c['old']} → {c['new']}" for c in changes_list]
        }
    
    def _set_adaptive_param(self, param: str, value: float) -> str:
        """Manually set an adaptive parameter (for Telegram /tune command)."""
        if param not in self.adaptive_params:
            return f"❌ Unknown parameter: {param}\nAvailable: {', '.join(self.adaptive_params.keys())}"
        
        info = self.adaptive_params[param]
        if value < info['min'] or value > info['max']:
            return f"❌ Value {value} out of range [{info['min']}, {info['max']}]"
        
        old_value = info['current']
        if self._apply_param_adjustment(param, value):
            return f"✅ {param}: {old_value} → {value}"
        return f"❌ Failed to apply change"
    
    async def _trigger_manual_tune(self) -> str:
        """Manually trigger AI parameter review (for Telegram /autotune command)."""
        self.trades_since_last_tune = self.TUNE_AFTER_N_TRADES  # Force trigger
        await self.maybe_auto_tune_params()
        return "🎛️ AI parameter review triggered"
    
    def _toggle_auto_tune(self) -> str:
        """Toggle AI auto-tune on/off (for Telegram command)."""
        self.AI_AUTO_TUNE_ENABLED = not self.AI_AUTO_TUNE_ENABLED
        status = "✅ Enabled" if self.AI_AUTO_TUNE_ENABLED else "❌ Disabled"
        return f"🎛️ AI Auto-Tune: {status}"
    
    def _get_adaptive_params_dict(self) -> Dict[str, Any]:
        """Get adaptive params as dictionary for dashboard."""
        if not hasattr(self, 'adaptive_params'):
            return {'params': {}, 'enabled': False}
        
        return {
            'params': {
                name: {
                    'current': info['current'],
                    'min': info['min'],
                    'max': info['max'],
                    'step': info.get('step', 0.1),
                    'desc': info['desc']
                }
                for name, info in self.adaptive_params.items()
            },
            'enabled': self.AI_AUTO_TUNE_ENABLED,
            'trades_until_tune': max(0, self.TUNE_AFTER_N_TRADES - self.trades_since_last_tune),
            'history': self.param_adjustment_history[-5:]  # Last 5 adjustments
        }
    
    # === CONTROL PANEL METHODS ===
    
    def _dashboard_pause(self) -> None:
        """Pause trading from dashboard."""
        self.paused = True
        logger.info("⏸ Trading PAUSED from dashboard")
    
    def _dashboard_resume(self) -> None:
        """Resume trading from dashboard."""
        self.paused = False
        logger.info("▶ Trading RESUMED from dashboard")
    
    def _dashboard_toggle_hold(self, symbol: str, hold_state: bool) -> Dict[str, Any]:
        """Toggle HOLD state for a position - prevents bot from closing when active.
        
        💎 DIAMOND HANDS MODE:
        When HOLD is ENABLED:
          - Client-side SL check is SKIPPED (position rides through drawdowns)
          - Server-side SL on Bybit is REMOVED (prevents exchange from closing it)
          - HARD_LOSS_CAP and SOFT_LOSS_CAP in AI filter are bypassed
          - RT quick exits are blocked
          - Max duration timer is paused
          - Position can ride for up to 24h for potential recovery
        
        When HOLD is DISABLED:
          - Server-side SL is RESTORED on Bybit
          - All normal exit logic resumes
        """
        try:
            if not symbol:
                return {'success': False, 'error': 'No symbol provided'}
            
            # Normalize symbol
            symbol_key = symbol.replace("/", "").replace(":USDT", "")
            
            # Verify position exists and find it (with non-None value)
            position_exists = False
            found_pos = None
            found_sym = None
            for sym, pos in self.positions.items():
                if pos is None:
                    continue  # Skip cleared positions
                sym_normalized = sym.replace("/", "").replace(":USDT", "")
                if sym_normalized == symbol_key:
                    position_exists = True
                    found_pos = pos
                    found_sym = sym
                    break
            
            # Also check primary position
            if not position_exists and self.position:
                primary_key = getattr(self.position, 'symbol', '').replace('/USDT:USDT', 'USDT').replace('/', '')
                if primary_key == symbol_key:
                    position_exists = True
                    found_pos = self.position
                    found_sym = getattr(self.position, 'symbol', '')
            
            if not position_exists:
                return {'success': False, 'error': f'No position found for {symbol}'}
            
            # Set hold state
            self.held_positions[symbol_key] = hold_state
            
            action = "ENABLED" if hold_state else "DISABLED"
            logger.info(f"💎 DIAMOND HANDS {action} for {symbol_key} - Bot exit {'blocked' if hold_state else 'allowed'}")
            
            # === CRITICAL: Manage server-side SL on Bybit ===
            # When HOLD is active, we MUST remove the server-side SL or Bybit will close the position
            # When HOLD is released, we restore the SL for protection
            sl_status = ""
            if self.live_mode and found_pos:
                import asyncio
                try:
                    if hold_state:
                        # REMOVE server-side SL — let the position ride
                        raw_sym = symbol_key if 'USDT' in symbol_key else f"{symbol_key}USDT"
                        async def _remove_sl():
                            try:
                                response = await self.exchange.privatePostV5PositionTradingStop({
                                    'category': 'linear',
                                    'symbol': raw_sym,
                                    'stopLoss': '0',
                                    'positionIdx': 0,
                                    'slOrderType': 'Market',
                                })
                                if str(response.get('retCode', '')) == '0':
                                    logger.info(f"💎 Removed server-side SL for {symbol_key} — diamond hands mode")
                                else:
                                    logger.warning(f"⚠️ Could not remove server SL: {response.get('retMsg', 'Unknown')}")
                            except Exception as e:
                                logger.error(f"Failed to remove server SL: {e}")
                        asyncio.create_task(_remove_sl())
                        if not hasattr(self, '_held_original_sl'):
                            self._held_original_sl = {}
                        self._held_original_sl[symbol_key] = getattr(found_pos, 'stop_loss', 0)
                        sl_status = "\n🚫 Server-side SL removal scheduled"
                    else:
                        # RESTORE server-side SL — protection back on
                        original_sl = getattr(self, '_held_original_sl', {}).get(symbol_key, 0)
                        if original_sl and original_sl > 0:
                            pos_side = getattr(found_pos, 'side', 'long')
                            pos_size = getattr(found_pos, 'size', 0)
                            async def _restore_sl():
                                await self._update_server_side_sl(found_sym or symbol_key, original_sl, pos_size, pos_side)
                            asyncio.create_task(_restore_sl())
                            sl_status = f"\n✅ Server-side SL restore scheduled @ ${original_sl:.4f}"
                            # Clean up
                            if hasattr(self, '_held_original_sl') and symbol_key in self._held_original_sl:
                                del self._held_original_sl[symbol_key]
                except Exception as e:
                    logger.error(f"Server-side SL management error during hold toggle: {e}")
                    sl_status = f"\n⚠️ SL management error: {e}"
            
            # Build position info
            pos_info = ""
            if found_pos:
                entry = getattr(found_pos, 'entry_price', 0)
                sl = getattr(found_pos, 'stop_loss', 0)
                side = getattr(found_pos, 'side', 'unknown').upper()
                pos_info = f"\n📊 {side} @ ${entry:.4f} | SL: ${sl:.4f}"
            
            # Notify via Telegram
            if self.telegram and self.telegram.enabled:
                emoji = "💎" if hold_state else "🔓"
                msg = f"{emoji} DIAMOND HANDS {action}: {symbol_key}\n"
                msg += f"Bot exit: {'BLOCKED — riding through drawdowns' if hold_state else 'ALLOWED — normal SL active'}"
                msg += pos_info
                msg += sl_status
                if hold_state:
                    msg += "\n\n⚠️ Position will ride without SL protection!"
                    msg += "\nUse /unhold to restore SL and normal exits."
                try:
                    asyncio.create_task(self.telegram.send_message(msg))
                except Exception:
                    pass
            
            return {
                'success': True,
                'symbol': symbol_key,
                'hold': hold_state,
                'message': f'Diamond hands {action.lower()} for {symbol_key}{sl_status}'
            }
            
        except Exception as e:
            logger.error(f"Toggle hold error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _dashboard_close_position(self, symbol: str = None) -> Dict[str, Any]:
        """Close position from dashboard - INSTANT execution with manual override flag."""
        import ccxt
        try:
            target_symbol = symbol or self.SYMBOL
            # Normalize to simple format for comparison
            symbol_key = target_symbol.replace("/", "").replace(":USDT", "")
            
            # Find position for this symbol (check multiple formats)
            position = None
            pos_key = None
            for sym, pos in self.positions.items():
                if pos is None:
                    continue  # Skip cleared/None positions
                sym_normalized = sym.replace("/", "").replace(":USDT", "")
                if sym_normalized == symbol_key:
                    position = pos
                    pos_key = sym
                    break
            
            if not position:
                logger.warning(f"Position search: target={target_symbol}, normalized={symbol_key}")
                logger.warning(f"Available positions: {list(self.positions.keys())}")
                return {'success': False, 'error': 'No position to close'}
            
            # Get remaining size
            remaining_size = getattr(position, 'remaining_size', None)
            if remaining_size is None:
                remaining_size = getattr(position, 'size', 0)
            
            if remaining_size <= 0:
                # Clean up stale position using standardized method
                self._clear_position(pos_key or target_symbol)
                return {'success': False, 'error': 'Position already closed'}
            
            # === INSTANT EXECUTION - Use synchronous exchange ===
            logger.info(f"⚡ MANUAL OVERRIDE: Instantly closing {pos_key}")
            
            # Create sync exchange for immediate execution (Bybit futures)
            sync_exchange = ccxt.bybit({
                'apiKey': self.exchange.apiKey,
                'secret': self.exchange.secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'swap', 'defaultSubType': 'linear', 'recvWindow': 20000}
            })
            sync_exchange.load_markets()
            
            # Convert to ccxt futures format for Bybit
            ccxt_symbol = f"{symbol_key.replace('USDT', '')}/USDT:USDT"
            
            # Get current price
            ticker = sync_exchange.fetch_ticker(ccxt_symbol)
            current_price = ticker['last']
            
            # Calculate P&L
            entry = getattr(position, 'entry_price', current_price)
            side = getattr(position, 'side', 'LONG')
            
            if side.upper() == 'LONG':
                pnl_pct = (current_price - entry) / entry * 100
                pnl_usd = (current_price - entry) * remaining_size
            else:
                pnl_pct = (entry - current_price) / entry * 100
                pnl_usd = (entry - current_price) * remaining_size
            
            # Execute close order IMMEDIATELY
            order_side = 'sell' if side.upper() == 'LONG' else 'buy'
            order = sync_exchange.create_market_order(
                ccxt_symbol, 
                order_side, 
                remaining_size,
                params={'reduceOnly': True}
            )
            
            # Get actual fill price if available
            fill_price = current_price
            if order.get('average'):
                fill_price = float(order['average'])
            elif order.get('price'):
                fill_price = float(order['price'])
            
            # Recalculate with fill price
            if side.upper() == 'LONG':
                pnl_pct = (fill_price - entry) / entry * 100
                pnl_usd = (fill_price - entry) * remaining_size
            else:
                pnl_pct = (entry - fill_price) / entry * 100
                pnl_usd = (entry - fill_price) * remaining_size
            
            # Record trade with MANUAL OVERRIDE flag
            is_win = pnl_usd >= 0
            opened_at = getattr(position, 'opened_at', None)
            hold_hours = 0
            if opened_at:
                hold_hours = (datetime.now(timezone.utc) - opened_at).total_seconds() / 3600
            
            self._record_trade_safe({
                'symbol': symbol_key,
                'side': side.upper(),
                'entry': entry,
                'exit': fill_price,
                'pnl': pnl_usd,
                'time': datetime.now(timezone.utc).strftime("%H:%M:%S"),
                'entry_time': opened_at.strftime("%H:%M:%S") if opened_at else "N/A",
                'exit_time': datetime.now(timezone.utc).strftime("%H:%M:%S"),
                'date': opened_at.strftime("%Y-%m-%d") if opened_at else datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                'opened_at': opened_at.isoformat() if opened_at else None,
                'closed_at': datetime.now(timezone.utc).isoformat(),
                'result': 'WIN' if is_win else 'LOSS',
                'exit_reason': 'MANUAL_OVERRIDE',  # Clear flag for AI/bot
                'position_num': 2 if pos_key != self.SYMBOL.replace("/", "") else 1,
                'tp1_hit': getattr(position, 'tp1_hit', False),
                'tp2_hit': getattr(position, 'tp2_hit', False),
                'manual_close': True  # Explicit flag
            })
            
            # Update balance - in LIVE mode, Bybit already applied the real PnL.
            # We still add pnl_usd here because this is a sync function and can't
            # easily await _sync_balance_with_exchange. The next periodic sync will
            # correct any minor discrepancy from the fill price.
            self.balance += pnl_usd
            self.weekly_pnl += pnl_usd  # Track weekly P&L for circuit breaker
            
            # Let _recalculate_stats_from_history() handle stats to avoid double-counting
            self._recalculate_stats_from_history()
            
            # Clear position using standardized method
            self._clear_position(pos_key or symbol_key)
            
            # Log success
            emoji = "✅" if is_win else "❌"
            logger.info(f"{emoji} ⚡ MANUAL CLOSE {side.upper()} {symbol_key} | "
                       f"PnL: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%) | Held: {hold_hours:.1f}h | "
                       f"Entry: ${entry:.4f} → Exit: ${fill_price:.4f}")
            
            # Notify via Telegram (non-blocking)
            if self.telegram and self.telegram.enabled:
                msg = f"⚡ MANUAL CLOSE {symbol_key}\n"
                msg += f"{side.upper()} closed by user\n"
                msg += f"Entry: ${entry:.4f} → Exit: ${fill_price:.4f}\n"
                msg += f"P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f})"
                try:
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self.telegram.send_message(msg))
                    else:
                        asyncio.create_task(self.telegram.send_message(msg))
                except Exception:
                    pass  # Non-critical: Don't block on notification failure
            
            return {
                'success': True, 
                'message': f'Position closed instantly',
                'pnl': pnl_usd,
                'pnl_pct': pnl_pct,
                'fill_price': fill_price
            }
            
        except Exception as e:
            logger.error(f"Manual close error: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _dashboard_open_trade(self, side: str, symbol: str = None) -> Dict[str, Any]:
        """Open a trade manually from dashboard - uses sync ccxt for reliability."""
        import ccxt
        try:
            side = side.lower()
            if side not in ['long', 'short']:
                return {'success': False, 'error': 'Side must be "long" or "short"'}
            
            # === SIDE FILTER CHECK ===
            allowed = getattr(self, 'allowed_sides', 'both').lower()
            if allowed != 'both':
                if allowed == 'long' and side == 'short':
                    return {'success': False, 'error': '🚫 SHORT blocked - /longonly mode active. Use /bothsides to enable.'}
                elif allowed == 'short' and side == 'long':
                    return {'success': False, 'error': '🚫 LONG blocked - /shortonly mode active. Use /bothsides to enable.'}
            
            # Determine target symbol
            target_symbol = symbol or self.SYMBOL
            target_symbol = target_symbol.upper().replace('/', '').replace('-', '')
            if not target_symbol.endswith('USDT'):
                target_symbol = target_symbol + 'USDT'
            
            logger.info(f"📊 Dashboard: Open {side.upper()} {target_symbol} requested")
            
            # Check max positions
            open_positions = [s for s, p in self.positions.items() if p is not None]
            max_positions = self.multi_pair_config.max_total_positions
            
            if len(open_positions) >= max_positions:
                return {
                    'success': False,
                    'error': f"Max positions ({max_positions}) reached. Open: {', '.join(open_positions)}. Close one first."
                }
            
            # Check if already have position on this symbol
            if self.positions.get(target_symbol):
                return {
                    'success': False,
                    'error': f"Already have position on {target_symbol}."
                }
            
            # Check if paused
            if self.paused:
                return {
                    'success': False,
                    'error': "Bot is paused. Use /resume to enable trading."
                }
            
            # === SYNCHRONOUS EXECUTION - More reliable for dashboard ===
            sync_exchange = ccxt.bybit({
                'apiKey': os.getenv("BYBIT_API_KEY"),
                'secret': os.getenv("BYBIT_API_SECRET"),
                'enableRateLimit': True,
                'options': {'defaultType': 'swap', 'defaultSubType': 'linear', 'recvWindow': 20000}
            })
            
            # Format symbol for ccxt (e.g., LINKUSDT -> LINK/USDT:USDT)
            ccxt_symbol = f"{target_symbol[:-4]}/USDT:USDT"
            
            # Fetch OHLCV data
            ohlcv = sync_exchange.fetch_ohlcv(ccxt_symbol, '15m', limit=100)
            if len(ohlcv) < 20:
                return {'success': False, 'error': f"Insufficient data for {target_symbol}"}
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            price = float(df['close'].iloc[-1])
            
            # Calculate ATR
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = float(true_range.rolling(14).mean().iloc[-1])
            if atr <= 0:
                atr = price * 0.01
            
            # Calculate position size with AI-adjustable parameters
            # Use ATR-based stop with DYNAMIC max based on each coin's volatility
            # === PER-VOLATILITY SL ADJUSTMENT (Notebook #2 §8) ===
            _vol_pct = (atr / price) * 100 if price > 0 else 1.0
            _vol_mult = 0.85 if _vol_pct < 0.5 else (1.15 if _vol_pct > 2.0 else 1.0)
            atr_stop = atr * self.ATR_MULT * _vol_mult
            min_stop = price * self.MIN_STOP_PCT
            # Dynamic max: ATR × multiplier × 1.5 safety, clamped to floor/ceiling
            dynamic_max_pct = max(self.DYNAMIC_STOP_FLOOR, min((atr / price) * self.ATR_MULT * _vol_mult * 1.5, self.DYNAMIC_STOP_CEILING))
            max_stop = price * dynamic_max_pct
            
            stop_distance = max(min_stop, min(atr_stop, max_stop))  # Clamp between min and dynamic max
            logger.info(f"📊 SL Calc: ATR={atr:.4f} ({atr/price*100:.2f}%), ATR_stop={atr_stop:.4f}, Min={min_stop:.4f}, DynMax={max_stop:.4f} ({dynamic_max_pct:.2%}), Final={stop_distance:.4f} ({stop_distance/price*100:.2f}%)")
            
            if side == 'long':
                stop_loss = price - stop_distance
            else:
                stop_loss = price + stop_distance
            
            risk_per_unit = abs(price - stop_loss)
            
            # AI-POWERED: Use adaptive risk_pct from AI tuning
            ai_risk_pct = self.adaptive_params.get('risk_pct', {}).get('current', self.RISK_PCT)
            ai_size_mult = self.adaptive_params.get('ai_size_mult', {}).get('current', 1.0)
            aggressive_mode = self.adaptive_params.get('aggressive_mode', {}).get('current', 0)
            
            # Aggressive mode DISABLED — too dangerous for small accounts
            # Previously allowed 1.5x risk boost up to 10% — caused outsized losses
            if aggressive_mode:
                logger.warning(f"🛡️ Aggressive mode attempted but DISABLED for safety")
                aggressive_mode = 0
            
            risk_amount = self.balance * ai_risk_pct
            size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
            
            # Apply AI size multiplier based on confidence
            size = size * ai_size_mult
            
            # Leverage check - HARD CAP at 10x regardless of params
            MAX_LEVERAGE = min(int(self.adaptive_params.get('max_leverage', {}).get('current', 10)), 10)
            position_value = size * price
            current_leverage = position_value / self.balance
            if current_leverage > MAX_LEVERAGE:
                size = (self.balance * MAX_LEVERAGE) / price
                current_leverage = MAX_LEVERAGE
            
            logger.info(f"🤖 AI SIZING: Risk={ai_risk_pct*100:.1f}%, SizeMult={ai_size_mult:.1f}x, Leverage={current_leverage:.1f}x")
            
            # 🧮 DYNAMIC TP — calculated per-trade using math
            dynamic_tp = self.calculate_dynamic_tp(
                entry_price=price, side=side, df=df, atr=atr, stop_loss=stop_loss
            )
            tp1 = dynamic_tp['tp1']
            tp2 = dynamic_tp['tp2']
            tp3 = dynamic_tp['tp3']
            
            logger.info(f"🎯 Dynamic TPs: TP1={dynamic_tp['tp1_display']:.2f}% @ ${tp1:.6f} | TP2={dynamic_tp['tp2_display']:.2f}% @ ${tp2:.6f} | TP3={dynamic_tp['tp3_display']:.2f}% @ ${tp3:.6f}")
            logger.info(f"🧮 TP factors: {dynamic_tp['factors']}")
            
            logger.info(f"📊 {target_symbol}: Price=${price:.4f}, Size={size:.6f}, SL=${stop_loss:.4f}")
            
            # Execute order
            order_side = 'buy' if side == 'long' else 'sell'
            order = sync_exchange.create_market_order(ccxt_symbol, order_side, size)
            
            if not order or not order.get('id'):
                return {'success': False, 'error': 'Order execution failed - no order ID returned'}
            
            # Get fill price
            fill_price = float(order.get('average') or order.get('price') or price)
            
            logger.info(f"✅ ORDER EXECUTED: {order_side.upper()} {size:.6f} {ccxt_symbol} @ ${fill_price:.4f} | ID: {order['id']}")
            
            # Create position object - use ccxt futures format for consistency
            position = Position(
                symbol=ccxt_symbol,
                side=side,
                entry_price=fill_price,
                size=size,
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                opened_at=datetime.now(timezone.utc),
                entry_df_snapshot=df.copy()
            )
            
            # Store position with ccxt format key
            self.positions[ccxt_symbol] = position
            if ccxt_symbol == self.SYMBOL or target_symbol == self.SYMBOL.replace('/', '').replace(':USDT', ''):
                self.position = position
            
            # Save state
            self._save_trading_state()
            
            logger.info(f"OPENED {side.upper()} [📊 Dashboard] | {target_symbol} | Entry: {fill_price:.4f} | Size: {size:.4f} | SL: {stop_loss:.4f}")
            
            return {
                'success': True,
                'message': f"✅ Opened {side.upper()} {target_symbol} at ${fill_price:.4f}",
                'price': fill_price,
                'size': size,
                'symbol': target_symbol
            }
            
        except Exception as e:
            logger.error(f"Dashboard open trade error: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def _dashboard_set_param(self, param: str, value: float) -> Dict[str, Any]:
        """Set adaptive parameter from dashboard."""
        try:
            if param not in self.adaptive_params:
                return {'success': False, 'error': f'Unknown parameter: {param}'}
            
            config = self.adaptive_params[param]
            
            # Validate bounds
            if value < config['min'] or value > config['max']:
                return {'success': False, 'error': f'Value out of range [{config["min"]}, {config["max"]}]'}
            
            old_value = config['current']
            config['current'] = value
            self._save_adaptive_params()
            self._apply_adaptive_params()
            
            logger.info(f"⚙️ Dashboard: {param} changed {old_value} → {value}")
            return {'success': True, 'message': f'{param} set to {value}'}
            
        except Exception as e:
            logger.error(f"Dashboard set param error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _dashboard_chat_with_ai(self, message: str) -> str:
        """Chat with AI from dashboard."""
        import asyncio
        import re
        try:
            if not self.ai_filter:
                return "AI not available"
            
            # Get current position details - DETAILED for AI context
            position_info = "No open position"
            has_position = False
            position_details = ""
            
            # Check self.position first (primary position object)
            if self.position:
                has_position = True
                pos = self.position
                symbol = getattr(pos, 'symbol', 'UNKNOWN')
                side = getattr(pos, 'side', 'UNKNOWN').upper()
                entry = getattr(pos, 'entry_price', 0)
                sl = getattr(pos, 'stop_loss', 0)
                tp1 = getattr(pos, 'tp1', 0)
                tp2 = getattr(pos, 'tp2', 0)
                tp3 = getattr(pos, 'tp3', 0)
                tp1_hit = getattr(pos, 'tp1_hit', False)
                tp2_hit = getattr(pos, 'tp2_hit', False)
                tp3_hit = getattr(pos, 'tp3_hit', False)
                
                # FIX: Get correct price for THIS position's symbol, not self.last_price which is main symbol
                symbol_key = symbol.upper().replace('/', '').replace(':USDT', '')
                current_price = entry  # Default fallback
                if hasattr(self, '_symbol_prices') and symbol_key in self._symbol_prices:
                    current_price = self._symbol_prices[symbol_key]
                elif hasattr(self, '_symbol_prices'):
                    # Try alternate key format
                    alt_key = symbol_key.replace('USDT', '') + 'USDT'
                    if alt_key in self._symbol_prices:
                        current_price = self._symbol_prices[alt_key]
                # Use self.last_price ONLY if this is the main symbol
                if current_price == entry and hasattr(self, 'last_price') and self.last_price:
                    main_symbol_key = self.SYMBOL.replace('/', '').replace(':USDT', '')
                    if symbol_key == main_symbol_key:
                        current_price = self.last_price
                
                # Calculate PnL
                if side == 'LONG':
                    pnl_pct = ((current_price - entry) / entry) * 100 if entry > 0 else 0
                else:
                    pnl_pct = ((entry - current_price) / entry) * 100 if entry > 0 else 0
                
                position_info = f"{side} {symbol} @ ${entry:.4f}"
                position_details = f"""
📊 OPEN POSITION DETAILS:
- Side: {side}
- Symbol: {symbol}
- Entry Price: ${entry:.4f}
- Current Price: ${current_price:.4f}
- Unrealized P&L: {pnl_pct:+.2f}%
- Stop Loss: ${sl:.4f} {'(distance: ' + f'{abs((sl-current_price)/current_price)*100:.2f}%' + ')' if sl > 0 else ''}
- TP1: ${tp1:.4f} {'✓ HIT' if tp1_hit else '(distance: ' + f'{abs((tp1-current_price)/current_price)*100:.2f}%' + ')' if tp1 > 0 else ''}
- TP2: ${tp2:.4f} {'✓ HIT' if tp2_hit else '(distance: ' + f'{abs((tp2-current_price)/current_price)*100:.2f}%' + ')' if tp2 > 0 else ''}
- TP3: ${tp3:.4f} {'✓ HIT' if tp3_hit else '(distance: ' + f'{abs((tp3-current_price)/current_price)*100:.2f}%' + ')' if tp3 > 0 else ''}

WHY TRADE MIGHT NOT BE CLOSED YET:
- Price has NOT hit Stop Loss (${sl:.4f})
- Price has NOT hit any remaining TP level
- No manual close requested
- AI position monitor may be holding (if hold score > exit score)
"""
            
            # Also check positions dict
            elif self.positions:
                has_position = True
                position_parts = []
                for sym, pos in self.positions.items():
                    if pos is None:
                        continue
                    side = getattr(pos, 'side', 'UNKNOWN').upper()
                    entry = getattr(pos, 'entry_price', 0)
                    
                    # FIX: Get correct price for each position's symbol
                    symbol_key = sym.upper().replace('/', '').replace(':USDT', '')
                    pos_current_price = entry  # Default
                    if hasattr(self, '_symbol_prices') and symbol_key in self._symbol_prices:
                        pos_current_price = self._symbol_prices[symbol_key]
                    elif hasattr(self, '_symbol_prices'):
                        alt_key = symbol_key.replace('USDT', '') + 'USDT'
                        if alt_key in self._symbol_prices:
                            pos_current_price = self._symbol_prices[alt_key]
                    
                    # Calculate PnL for this position
                    if side == 'LONG':
                        pos_pnl_pct = ((pos_current_price - entry) / entry) * 100 if entry > 0 else 0
                    else:
                        pos_pnl_pct = ((entry - pos_current_price) / entry) * 100 if entry > 0 else 0
                    
                    position_parts.append(f"{side} {sym} @ ${entry:.4f} (PnL: {pos_pnl_pct:+.2f}%)")
                
                position_info = " | ".join(position_parts) if position_parts else "No valid positions"
            
            # Build context with EXPLICIT symbol
            current_symbol = self.symbol if hasattr(self, 'symbol') else 'UNKNOWN'
            context = f"""
🚨 IMPORTANT: CURRENT TRADING SYMBOL IS: {current_symbol}
DO NOT say we are trading a different symbol!

Current Status:
- Symbol: {current_symbol}
- Balance: ${self.balance:.2f}
- Position: {position_info}
- Has Open Position: {has_position}
- Current Price: ${self.last_price if hasattr(self, 'last_price') else 0:.4f}
- Win Rate: {self.stats.winning_trades}/{self.stats.total_trades} trades ({self.stats.win_rate*100:.1f}%)
- Current PnL: ${self.balance - self.initial_balance:.2f}
- Regime: {getattr(self, 'current_regime', 'Unknown')}
{position_details}
Recent Performance:
- Consecutive Wins: {self.consecutive_wins}
- Consecutive Losses: {self.consecutive_losses}

Parameters:
- ATR Mult: {self.ATR_MULT}
- TP1/TP2/TP3: {self.TP1_R}R/{self.TP2_R}R/{self.TP3_R}R
- AI Threshold: {self.ai_filter.confidence_threshold if self.ai_filter else 'N/A'}
- AI Mode: {getattr(self, 'ai_mode', 'unknown')}

User Question: {message}
"""
            
            # Use AI filter's chat method if available (it's async)
            if hasattr(self.ai_filter, 'chat'):
                # Create a new event loop since we're not in an async context
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an event loop, use run_coroutine_threadsafe
                    import concurrent.futures
                    future = asyncio.run_coroutine_threadsafe(
                        self.ai_filter.chat(message, context),
                        loop
                    )
                    response = future.result(timeout=30)
                except RuntimeError:
                    # No running loop, create one
                    response = asyncio.run(self.ai_filter.chat(message, context))
            else:
                response = "AI chat method not available"
            
            # === ANTI-HALLUCINATION CHECK FOR DASHBOARD CHAT ===
            if response:
                response_lower = response.lower()
                corrections = []
                
                # Check for wrong symbol claims
                actual_base = current_symbol.replace('USDT', '').lower()
                wrong_symbols = ['btc', 'eth', 'sol', 'link', 'tia', 'inj', 'arb', 'apt', 'avax', 'matic', 'dot', 'ada', 'xrp', 'doge', 'op', 'sui', 'pepe', 'wif', 'bonk', 'sei', 'near']
                for sym in wrong_symbols:
                    if sym != actual_base:
                        if f"trading {sym}" in response_lower or f"on {sym}" in response_lower or f"with {sym}" in response_lower:
                            corrections.append(f"Symbol: We are trading {current_symbol}, not {sym.upper()}")
                            logger.warning(f"🚨 Dashboard AI hallucination: said {sym.upper()} but trading {current_symbol}")
                            break
                
                # Check for position claims when none
                position_claim_phrases = ["we have a position", "our position", "current position is", "holding a", "i opened", "we opened"]
                claims_position = any(phrase in response_lower for phrase in position_claim_phrases)
                if claims_position and not has_position:
                    corrections.append("Position: NO open position exists")
                    logger.warning("🚨 Dashboard AI hallucination: claimed position but none exists")
                
                # Add corrections if any
                if corrections:
                    response += "\n\n⚠️ *Reality Check:*\n• " + "\n• ".join(corrections)
            
            return response or "I couldn't generate a response. Please try again."
            
        except Exception as e:
            logger.error(f"Dashboard AI chat error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def _dashboard_trigger_autotune(self) -> Dict[str, Any]:
        """Suggest AI auto-tune changes from dashboard (do NOT apply)."""
        import asyncio
        try:
            # Only suggest, do not apply
            result = None
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(
                    self.maybe_auto_tune_params(),
                    loop
                )
                result = future.result(timeout=30)
            except RuntimeError:
                result = asyncio.run(self.maybe_auto_tune_params())
            if result:
                return {
                    'success': True,
                    'adjusted': result.get('adjusted', False),
                    'message': 'Suggested changes',
                    'changes': result.get('changes', []),
                    'reasoning': result.get('reasoning', ''),
                    'summary': result.get('summary', [])
                }
            else:
                return {'success': True, 'adjusted': False, 'message': 'No changes suggested'}
        except Exception as e:
            logger.error(f"Dashboard auto-tune error: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def _dashboard_apply_autotune(self, changes: list) -> Dict[str, Any]:
        """Apply suggested changes from dashboard."""
        applied = []
        for change in changes:
            param = change.get('param')
            new_value = change.get('new')
            old_value = change.get('old')
            if param and new_value is not None:
                if self._apply_param_adjustment(param, new_value):
                    applied.append(f"{param}: {old_value} → {new_value}")
        
        # Invalidate the cache so next autotune gets fresh suggestions
        self._autotune_cache = None
        
        # Only notify Telegram here (fire-and-forget async)
        if applied and self.telegram:
            msg = f"🎛️ *AI Auto-Tuned Parameters Applied*\n\n"
            msg += "Changes:\n" + "\n".join(f"• {a}" for a in applied)
            try:
                # Schedule async notification without waiting (fire-and-forget)
                asyncio.create_task(self.telegram.send_message(msg))
            except Exception as e:
                logger.warning(f"Failed to schedule Telegram notification: {e}")
        self._save_adaptive_params()
        return {'success': True, 'applied': applied}

    async def _handle_force_close(self):
        """Handle force close request from dashboard."""
        symbol = self.force_close_symbol
        self.force_close_symbol = None  # Reset immediately
        
        if not symbol:
            return
            
        logger.info(f"🔴 DASHBOARD: Force closing position for {symbol}")
        
        try:
            # Normalize symbol
            symbol_key = symbol.replace("/", "")
            
            # Find position in multi-positions dict first
            position = None
            pos_key = None
            
            # Check positions dict
            if hasattr(self, 'positions') and self.positions:
                for key, pos in self.positions.items():
                    if key.replace("/", "") == symbol_key:
                        position = pos
                        pos_key = key
                        break
            
            # Fall back to legacy position
            if not position and self.position:
                if getattr(self.position, 'symbol', '').replace("/", "") == symbol_key:
                    position = self.position
            
            if not position:
                logger.warning(f"No position to close for {symbol}")
                return
            
            # Get remaining size (accounts for partial closes from TPs)
            remaining_size = getattr(position, 'remaining_size', 0) if hasattr(position, 'remaining_size') else getattr(position, 'size', 0)
            
            # Check if position is actually valid (has remaining size)
            if remaining_size <= 0:
                logger.warning(f"Position {symbol} already closed (remaining_size={remaining_size}) - cleaning up")
                # Clean up the stale position
                if pos_key and hasattr(self, 'positions') and pos_key in self.positions:
                    del self.positions[pos_key]
                    logger.info(f"✓ Removed stale position {pos_key} from positions dict")
                if self.position and getattr(self.position, 'symbol', '').replace("/", "") == symbol_key:
                    self.position = None
                self._save_trading_state()
                return
            
            # Get current price
            ticker = await self.exchange.fetch_ticker(symbol_key)
            current_price = ticker['last']
            
            # Calculate P&L - Position is a dataclass, use getattr
            entry = getattr(position, 'entry_price', current_price)
            side = getattr(position, 'side', 'LONG')
            size = remaining_size  # Use remaining size, not original size
            
            if side.upper() == 'LONG':
                pnl_pct = (current_price - entry) / entry * 100
                pnl_usd = (current_price - entry) * size
            else:
                pnl_pct = (entry - current_price) / entry * 100
                pnl_usd = (entry - current_price) * size
            
            # Close via exchange
            order_side = 'sell' if side.upper() == 'LONG' else 'buy'
            order = await self.exchange.create_market_order(symbol_key, order_side, size, params={'reduceOnly': True})
            
            # Record trade with duplicate prevention
            is_win = pnl_usd >= 0
            opened_at_dash = getattr(position, 'opened_at', None)
            self._record_trade_safe({
                'symbol': symbol_key,
                'side': side.upper(),
                'entry': entry,
                'exit': current_price,
                'pnl': pnl_usd,
                'time': datetime.now(timezone.utc).strftime("%H:%M:%S"),
                'entry_time': opened_at_dash.strftime("%H:%M:%S") if opened_at_dash else "N/A",
                'exit_time': datetime.now(timezone.utc).strftime("%H:%M:%S"),
                'date': opened_at_dash.strftime("%Y-%m-%d") if opened_at_dash else datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                'opened_at': opened_at_dash.isoformat() if opened_at_dash else None,
                'closed_at': datetime.now(timezone.utc).isoformat(),
                'result': 'WIN' if is_win else 'LOSS',
                'exit_reason': 'dashboard_force_close',
                'tp1_hit': getattr(position, 'tp1_hit', False),
                'tp2_hit': getattr(position, 'tp2_hit', False)
            })
            
            # Update balance - sync from exchange in LIVE mode
            if self.live_mode:
                try:
                    await self._sync_balance_with_exchange()
                    logger.info(f"💰 Balance synced from exchange after force close: ${self.balance:.2f}")
                except Exception as sync_err:
                    logger.warning(f"⚠️ Could not sync balance: {sync_err} - applying PnL manually")
                    self.balance += pnl_usd
            else:
                self.balance += pnl_usd
            self.stats.today_pnl += pnl_usd
            self.stats.total_pnl += pnl_usd
            if pnl_usd >= 0:
                self.stats.winning_trades += 1
            else:
                self.stats.losing_trades += 1
            self.stats.total_trades += 1
            
            # Clear position using standardized method (stamps _recently_closed_symbols)
            self._clear_position(pos_key or symbol_key)
            
            # Log and notify
            emoji = "🟢" if pnl_usd > 0 else "🔴"
            msg = f"{emoji} FORCE CLOSED {side.upper()} {symbol_key}\n"
            msg += f"Entry: ${entry:.4f} → Exit: ${current_price:.4f}\n"
            msg += f"P&L: {pnl_pct:+.2f}% (${pnl_usd:+.2f})"
            
            logger.info(msg)
            if self.telegram:
                await self.telegram.send_message(msg)
                
        except Exception as e:
            logger.error(f"Force close error: {e}")
            import traceback
            traceback.print_exc()

    def __init__(
        self,
        paper_balance: Optional[float] = None,
        ai_confidence: float = 0.7,
        log_level: str = "INFO",
        ai_mode: str = "autonomous",  # "filter", "advisory", "autonomous", "hybrid"
        symbol: str = "LINK/USDT",
        scan_interval: int = 30,  # Scan every 30 seconds for faster responsiveness
        summary_interval: int = 14400  # 4 hours default
    ):
        # Set log level
        logging.getLogger().setLevel(getattr(logging, log_level.upper()))
        
        # Symbol (now configurable!) - Load persisted symbol if available
        self.SYMBOL = self._load_persisted_symbol() or symbol
        
        # AI Trading Mode
        # "filter" = AI only validates technical signals (default)
        # "advisory" = AI can suggest trades, requires Telegram confirmation
        # "autonomous" = AI can open trades directly with high confidence
        # "hybrid" = AI suggests via Telegram, you confirm
        self.ai_mode = ai_mode
        self.pending_ai_trade = None  # For advisory/hybrid mode confirmation
        self.last_ai_scan_time = None  # Rate limit AI scans
        self.ai_scan_interval = scan_interval  # Configurable scan interval
        self.ai_scan_notify_opportunities_only = True  # Only notify when AI finds opportunity
        self.ai_scan_quiet_interval = 1800  # Notify "no opportunity" every 30 min (if not opportunities_only)
        self.ai_scan_telegram_enabled = False  # Disable ALL telegram notifications for AI scans
        self.ai_position_monitor_telegram_enabled = False  # Disable position HOLD reminders (only alert on close)
        
        # === AI AUTONOMOUS DECISION TRACKING ===
        self.ai_decision_interval = 300  # Notify every 5 minutes about AI decisions
        self.last_ai_decision_notification = None
        self.ai_decisions_log = []  # Track all AI decisions for transparency
        self.ai_self_adjust_enabled = True  # Allow AI to adjust settings based on performance
        self.last_ai_self_adjust = None  # Rate limit self-adjustments
        self.ai_self_adjust_interval = 1800  # Check every 30 minutes
        
        # === WEEKEND/WEEKDAY THRESHOLD AUTO-UPDATE ===
        self._current_threshold_mode = None  # 'weekend' or 'weekday' — set on first check
        
        # === PRE-FILTER STATISTICS ===
        self.prefilter_stats = {
            'total_signals': 0,
            'passed': 0,
            'blocked_score': 0,
            'blocked_adx_low': 0,
            'blocked_adx_danger': 0,
            'blocked_volume': 0,
            'blocked_confluence': 0,
            'blocked_divergence': 0,  # NEW: Track divergence blocks
            'blocked_btc_filter': 0,
            'raw_signals': 0,  # Track raw signals before any filtering
            'by_regime': {}  # Track per-regime stats
        }
        
        # Trading state
        self.paper_mode = paper_balance is not None
        self.live_mode = False  # Set via --live flag - REAL ORDER EXECUTION
        self.balance = paper_balance or 10000.0
        self.initial_balance = self.balance
        self.peak_balance = self.balance  # Track peak for drawdown calculation
        self.position: Optional[Position] = None
        self.stats = TradeStats()
        self.start_time = None  # Set when bot actually starts running
        self.paused = False  # Trading pause state
        self.force_close_symbol = None  # Dashboard force close request
        
        # === LIVE TRADING ORDER TRACKING ===
        self.pending_orders: Dict[str, Any] = {}  # order_id -> order info
        self.last_order_id: Optional[str] = None
        
        # === ENGINE STATUS TRACKING ===
        self.engine_running = False  # True when main loop is active
        self.cycle_count = 0  # Total main loop iterations
        self.last_cycle_time: Optional[datetime] = None  # When last cycle completed
        self.last_cycle_duration_ms: float = 0  # How long last cycle took
        self.last_signal_time: Optional[datetime] = None  # When last signal was generated
        self.last_trade_time: Optional[datetime] = None  # When last trade opened/closed
        
        # === ERROR TRACKING FOR DASHBOARD ===
        self.error_history: List[Dict[str, Any]] = []  # List of recent errors
        self.max_error_history = 50  # Keep last 50 errors
        self.last_error: Optional[str] = None
        self.last_error_time: Optional[datetime] = None
        self.total_error_count = 0
        
        # Streak tracking for intelligent risk management
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        
        # Autonomous summary notifications
        self.last_summary_time = None
        self.summary_interval = summary_interval  # Configurable via CLI
        self.last_daily_summary_date = None  # Track daily summary
        self.summary_notifications_enabled = True  # Can be toggled via /summary command
        
        # === DAILY LOSS LIMIT (Circuit Breaker) ===
        self.daily_loss_limit = 0.05  # 5% max daily loss (configurable)
        self.daily_loss_triggered = False  # Circuit breaker state
        self.daily_loss_reset_date = None  # Track when to reset
        self.daily_loss_override_until = None  # Manual override expires at this date
        
        # === WEEKLY LOSS LIMIT (Circuit Breaker) ===
        self.weekly_loss_limit = 0.10  # 10% max weekly loss
        self.weekly_loss_triggered = False  # Weekly circuit breaker state
        self.weekly_loss_reset_week = None  # Track ISO week number for reset
        self.weekly_pnl = 0.0  # Track weekly P&L
        
        # === MAX DRAWDOWN CIRCUIT BREAKER ===
        self.max_drawdown_limit = 0.20  # 20% max drawdown from peak
        self.max_drawdown_triggered = False  # Ultimate circuit breaker
        
        # === BTC CRASH PROTECTION ===
        self.btc_crash_cooldown = False
        self.btc_crash_cooldown_until = None
        self.btc_crash_threshold = -0.05  # -5% BTC drop
        self.btc_crash_cooldown_minutes = 30  # 30 min cooldown (was 60 - faster sample collection)
        
        # === DRY-RUN MODE ===
        self.dry_run_mode = False  # Log trades without executing
        self.dry_run_tracker = DryRunTracker()  # Shadow position tracker
        
        # === ML TRAINING SETTINGS ===
        self.ml_learn_all_trades = True  # Learn from ALL trades (manual+auto) by default
        
        # === AI POSITION MONITOR SETTINGS ===
        self.ai_position_monitor_interval = 10  # Check every 10 seconds for fast reaction
        self._last_position_monitor = None
        
        # === POSITION WATCHDOG (Safety Net for Non-Autonomous Modes) ===
        self.watchdog_enabled = True  # Enable position watchdog
        self.watchdog_timeout = 600  # 10 minutes to respond
        self.watchdog_alert_sent = None  # When alert was sent
        self.watchdog_user_confirmed = False  # User confirmed they're watching
        self.watchdog_original_mode = None  # Store original mode before auto-switch
        
        # === REALISTIC EXECUTION TRACKING ===
        self.total_fees_paid = 0.0  # Track all fees for P&L
        self.total_slippage_cost = 0.0  # Track slippage costs
        
        # === BTC CRASH PROTECTION STATE ===
        self.btc_crash_detected = False
        self.btc_crash_until: Optional[datetime] = None
        self.last_btc_price: Optional[float] = None
        self.last_btc_check: Optional[datetime] = None
        self.btc_1h_ago_price: Optional[float] = None  # For 1h change tracking
        
        # === MULTI-SYMBOL SUPPORT (ML Acceleration Plan) ===
        self.ml_config = get_ml_config()
        self.multi_pair_config = get_multi_pair_config()
        
        # Primary symbol (backward compatible)
        self.symbols: List[str] = [symbol]
        
        # Multi-pair state - initialize from config
        enabled_pairs = self.multi_pair_config.get_enabled_pairs()
        self.additional_symbols: List[str] = [p.pair for p in enabled_pairs if p.pair != symbol]
        
        # Track positions per symbol (for multi-pair support)
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.position_slots: Dict[str, int] = {}  # symbol -> slot number (1, 2, etc.)
        self.open_position_count: int = 0
        self.held_positions: Dict[str, bool] = {}  # symbol -> True if HOLD active (prevents bot from closing)
        
        # === RACE CONDITION PROTECTION ===
        # asyncio.Lock for thread-safe access to shared state
        self._position_lock = asyncio.Lock()  # Protects positions dict
        self._balance_lock = asyncio.Lock()   # Protects balance operations
        self._state_lock = asyncio.Lock()     # General state protection
        self._close_lock = asyncio.Lock()     # Prevents duplicate close operations
        self._closing_symbols: set = set()    # Track symbols currently being closed
        
        # === PER-PAIR LOSS COOLDOWN (prevent re-opening bad trades immediately) ===
        self.pair_loss_cooldown: Dict[str, datetime] = {}  # symbol -> cooldown_until
        self.PAIR_LOSS_COOLDOWN_MINUTES = 30  # 30 min per-symbol cooldown after loss (Notebook #2 §14: 5 min was too short, revenge trades)
        
        # === RE-ENTRY COOLDOWN (prevent re-entering same symbol+direction too fast) ===
        # FIX: Bot re-entered PIPPIN LONG 12 seconds after closing previous PIPPIN LONG
        self.reentry_cooldown: Dict[str, datetime] = {}  # "SYMBOL_SIDE" -> cooldown_until
        self.REENTRY_COOLDOWN_SECONDS = 180  # 180s cooldown before re-entering same symbol+direction (was 60s — caused 5-trade revenge loops on CLO)
        
        # ML data collection (passive - doesn't block trades)
        self.ml_trade_log: List[Dict] = []  # Complete trade logs for ML training
        
        logger.info(f"Multi-pair enabled: {[p.pair for p in enabled_pairs]}")
        logger.info(f"ML Config: influence={self.ml_config.influence_weight}, min_samples={self.ml_config.min_samples_for_predictions}")
        
        # History for Telegram commands
        self.trade_history: List[Dict] = []
        self.signal_history: List[Dict] = []
        self.cached_last_price: float = 0.0
        self._realtime_price: float = 0.0  # WebSocket real-time price
        self._symbol_prices: Dict[str, float] = {}  # Multi-symbol price cache
        self.cached_last_ticker: Dict = {}
        self.cached_last_atr: float = 0.0
        
        # Data
        self.bars_1m: List[Dict] = []
        self.bars_agg: pd.DataFrame = pd.DataFrame()
        self._pending_data_refetch = False  # Flag to trigger refetch after symbol change
        self._pending_switch_notification = None  # Queue for symbol switch notifications
        self._pending_second_pair = None  # Queue for second pair to scan (multi-position)
        self._needs_full_pair_scan = False  # Flag to scan all pairs when no positions
        
        # Exchange
        self.exchange: Optional[ccxt.Exchange] = None
        
        # AI Filter
        self.ai_filter = AISignalFilter(confidence_threshold=ai_confidence)
        

        
        # === NEW ENHANCEMENT MODULES ===
        # Risk Manager (centralized risk control)
        self.risk_manager = get_risk_manager()
        
        # Risk Shield (hedge-fund grade circuit breaker wall — Phase 0)
        self.risk_shield = get_risk_shield(self.risk_manager)
        
        # Multi-Timeframe Analyzer
        self.mtf_analyzer = get_mtf_analyzer()
        
        # Chart Generator
        self.chart_generator = get_chart_generator()
        
        # ML Predictor (XGBoost trained on backtest data) - LEGACY
        self.ml_predictor = get_ml_predictor()
        if self.ml_predictor.is_loaded:
            logger.info(f"🧠 ML Predictor loaded: {self.ml_predictor.metrics.get('accuracy', 0):.1%} accuracy")
        else:
            logger.info("🧠 ML Predictor: Model not loaded (will use when available)")
        
        # === DUAL ML MODELS (NEW) ===
        # Historical ML: Trained on backtested historical data (static baseline)
        self.ml_historical = get_historical_model()
        if self.ml_historical.is_loaded:
            logger.info(f"🧠 Historical ML loaded: {self.ml_historical.metrics.get('accuracy', 0):.1%} accuracy, "
                        f"{self.ml_historical.metrics.get('total_samples', 0)} samples")
        else:
            logger.info("🧠 Historical ML: Not trained yet (run ml_historical.py)")
        
        # Live ML: Auto-trains on YOUR actual live trades (adaptive)
        self.ml_live = get_live_model()
        live_status = self.ml_live.get_status()
        if self.ml_live.is_trained:
            logger.info(f"🧠 Live ML loaded: {live_status.get('accuracy', 0):.1%} accuracy, "
                        f"{live_status['total_samples']} samples")
        else:
            logger.info(f"🧠 Live ML: {live_status['total_samples']} samples collected, "
                        f"{live_status['samples_until_training']} more needed to train")
        
        # AI Decision Tracker (for measuring AI accuracy)
        self.ai_tracker = get_ai_tracker()
        logger.info(f"📊 AI Tracker: {len(self.ai_tracker.decisions)} historical decisions loaded")
        

        
        # Current decision ID for linking trades to AI decisions
        self.current_decision_id: Optional[str] = None
        
        # Performance Dashboard (ALWAYS ENABLED by default)
        self.dashboard = get_dashboard(port=5000)
        self.dashboard_enabled = True  # Always enabled (can be disabled via --no-dashboard)
        
        # === WEBSOCKET REAL-TIME STREAMS ===
        # PhD-optimal: React to every price tick, not just 5-second polling
        # This ensures we NEVER miss a TP/SL level being hit
        self.ws_price_stream = WebSocketPriceStream(self, symbols=[])
        # Register recovery tracker callback for real-time loss monitoring
        # Note: ws_position_stream is created separately, register callback later
        self.ws_position_stream = WebSocketPositionStream(self)  # Real-time position sync
        self.ws_enabled = True  # Can be disabled if causing issues
        
        # Higher timeframe data caching
        self.bars_15m: pd.DataFrame = pd.DataFrame()
        self.bars_1h: pd.DataFrame = pd.DataFrame()
        self.last_htf_update: Optional[datetime] = None
        
        # Equity curve tracking
        self.equity_curve: List[float] = [self.balance]
        
        # === LOAD PERSISTED TRADING STATE ===
        self._load_trading_state()  # Restore balance, stats, streaks from disk
        self._sync_from_ai_decisions()  # PERSISTENCE GUARD: Validate & reconcile with source of truth
        
        # === LOAD ADAPTIVE PARAMETERS ===
        self._load_adaptive_params()  # Load AI-tunable parameters
        self._apply_adaptive_params()  # Apply to class attributes
        
        # Update equity curve to reflect loaded balance
        if self.equity_curve and self.equity_curve[0] != self.balance:
            self.equity_curve = [self.balance]
        
        # Telegram
        self.telegram = get_telegram_notifier()
        self._setup_telegram_callbacks()
        
        # Control
        self.running = False
        
        logger.info(f"Julaba initialized | Paper: {self.paper_mode} | Balance: ${self.balance:,.2f}")
    
    def _setup_telegram_callbacks(self):
        """Setup callbacks for Telegram commands."""
        self.telegram.get_status = self._get_status
        self.telegram.get_positions = self._get_positions
        self.telegram.get_pnl = self._get_pnl
        self.telegram.get_ai_stats = self._get_ai_stats_for_dashboard  # Unified with dashboard
        self.telegram.get_balance = self._get_balance
        self.telegram.get_trades = self._get_trades
        self.telegram.get_market = self._get_market
        self.telegram.get_signals = self._get_signals
        self.telegram.do_stop = self._do_stop
        self.telegram.do_pause = self._do_pause
        self.telegram.do_resume = self._do_resume
        self.telegram.chat_with_ai = self._chat_with_ai
        # AI mode callbacks
        self.telegram.get_ai_mode = lambda: self.ai_mode
        self.telegram.set_ai_mode = self._set_ai_mode
        self.telegram.confirm_ai_trade = self._confirm_ai_trade
        self.telegram.reject_ai_trade = self._reject_ai_trade
        self.telegram.execute_ai_trade = self._execute_ai_trade
        self.telegram.close_ai_trade = self._close_ai_trade
        # Pending trade execution callback (for below-minimum size trades)
        self.telegram.execute_pending_trade = self._execute_pending_trade
        # Watchdog confirmation callback
        self.telegram.confirm_watchdog = self.confirm_watchdog
        # Intelligence callbacks
        self.telegram.get_intelligence = self._get_intelligence
        self.telegram.get_ml_stats = self._get_ml_stats
        self.telegram.get_regime = self._get_regime
        # Summary notification toggle
        self.telegram.toggle_summary = self._toggle_summary_notifications
        self.telegram.get_summary_status = lambda: self.summary_notifications_enabled
        # System control for AI
        self.telegram.get_system_params = self._get_system_params
        self.telegram.set_system_param = self._set_system_param
        self.telegram.get_full_system_state = self._get_full_system_state
        # NEW: Position monitor analysis for Telegram AI consistency
        self.telegram.get_position_monitor_analysis = self._get_position_monitor_analysis
        # NEW: Enhanced module callbacks
        self.telegram.get_risk_stats = self._get_risk_stats
        self.telegram.get_shield_status = self._get_shield_status
        self.telegram.reset_shield = self._reset_shield
        self.telegram.emergency_liquidate = self._emergency_liquidate_manual
        # Dry-run callbacks
        self.telegram.get_dry_run_summary = lambda: self.dry_run_tracker.get_summary()
        self.telegram.toggle_dry_run = self._toggle_dry_run
        self.telegram.reset_dry_run = lambda: self.dry_run_tracker.reset()
        self.telegram.get_mtf_analysis = self._get_mtf_analysis
        self.telegram.run_backtest = self._run_backtest
        self.telegram.get_chart = self._get_chart
        self.telegram.get_equity_curve = lambda: self.equity_curve
        # Symbol switch callback (unified with dashboard)
        self.telegram.switch_symbol = self._switch_trading_symbol
        # AI market analysis (SAME function as dashboard for consistency)
        self.telegram.ai_analyze_markets = self._ai_analyze_all_markets
        # Symbol trade context (S/R, F&G, volume for advisory mode)
        self.telegram.get_symbol_trade_context = self._get_symbol_trade_context
        # NEW: Adaptive parameters callbacks
        self.telegram.get_adaptive_params = self.get_adaptive_params_summary
        self.telegram.set_adaptive_param = self._set_adaptive_param
        self.telegram.trigger_auto_tune = self._trigger_manual_tune
        self.telegram.toggle_auto_tune = self._toggle_auto_tune

        # Trading mode switch callbacks
        self.telegram.get_trading_mode = self._get_trading_mode
        self.telegram.switch_trading_mode = self._switch_trading_mode
        # Direction filter callbacks
        self.telegram.get_allowed_sides = lambda: getattr(self, 'allowed_sides', 'both')
        self.telegram.set_allowed_sides = self._set_allowed_sides
        
        # Dashboard callbacks
        self.dashboard.get_status = self._get_status
        self.dashboard.get_balance = self._get_balance
        self.dashboard.get_pnl = self._get_pnl
        self.dashboard.get_position = self._get_current_position_dict
        self.dashboard.get_additional_positions = self._get_additional_positions
        self.dashboard.get_trades = self._get_trades
        self.dashboard.get_regime = self._get_regime
        self.dashboard.get_ai_stats = self._get_ai_stats_for_dashboard
        self.dashboard.get_equity_curve = lambda: self.equity_curve
        # Enhanced dashboard callbacks
        self.dashboard.get_indicators = self._get_indicators_for_dashboard
        self.dashboard.get_current_signal = self._get_current_signal
        self.dashboard.get_risk_stats = self._get_risk_stats
        self.dashboard.get_mtf_analysis = self._get_mtf_analysis
        self.dashboard.get_params = self._get_system_params
        self.dashboard.get_signals = self._get_signals
        self.dashboard.get_ohlc_data = self._get_ohlc_for_chart
        # NEW: ML status and system logs
        self.dashboard.get_ml_status = self._get_ml_status
        self.dashboard.get_system_logs = self._get_system_logs
        # NEW: AI tracker and pre-filter stats
        self.dashboard.get_ai_tracker_stats = self._get_ai_tracker_stats
        self.dashboard.get_prefilter_stats = self._get_prefilter_stats
        # AI explanation for dashboard info buttons
        self.dashboard.get_ai_explanation = self._get_ai_explanation_for_dashboard
        # Market scanner callbacks
        self.dashboard.get_market_scan = self._get_market_scan_data
        self.dashboard.switch_symbol = self._switch_trading_symbol
        self.dashboard.ai_analyze_markets = self._ai_analyze_all_markets
        # Unified system state (single source of truth)
        self.dashboard.get_full_state = self._get_full_system_state
        # NEW: Pipeline monitoring - real-time component status
        self.dashboard.get_pipeline_status = self._get_pipeline_status
        # NEW: Adaptive parameters for dashboard
        self.dashboard.get_adaptive_params = self._get_adaptive_params_dict
        
        # === CONTROL PANEL CALLBACKS ===
        self.dashboard.do_pause = self._dashboard_pause
        self.dashboard.do_resume = self._dashboard_resume
        self.dashboard.do_close_position = self._dashboard_close_position
        self.dashboard.toggle_position_hold = self._dashboard_toggle_hold  # NEW: Hold toggle
        self.dashboard.do_open_trade = self._dashboard_open_trade  # NEW: Manual trade
        self.dashboard.set_adaptive_param = self._dashboard_set_param
        self.dashboard.set_system_param = self._set_system_param  # For force_resume, etc.
        self.dashboard.chat_with_ai = self._dashboard_chat_with_ai
        self.dashboard.trigger_auto_tune = self._dashboard_trigger_autotune
        self.dashboard.set_ai_mode = self._set_ai_mode  # NEW: Set AI mode from dashboard
        self.dashboard.get_error_summary = self.get_error_summary  # NEW: Error tracking for dashboard
        self.dashboard.clear_errors = self.clear_errors  # NEW: Clear errors from dashboard
        
        # === SECURITY MONITOR INTEGRATION ===
        try:
            from security_monitor import get_security_monitor
            security = get_security_monitor()
            
            # Set Telegram alert callback for security events
            async def security_telegram_alert(message: str):
                """Send security alert via Telegram."""
                try:
                    await self.telegram.send_message(f"🛡️ SECURITY ALERT\n\n{message}")
                except Exception as e:
                    logger.error(f"Failed to send security alert: {e}")
            
            security.telegram_callback = security_telegram_alert
            logger.info("🛡️ Security Monitor integrated with Telegram alerts")
        except Exception as e:
            logger.warning(f"Security monitor integration failed: {e}")
    
    def _get_ai_explanation_for_dashboard(self, topic: str, display_name: str) -> str:
        """Get AI explanation for a dashboard topic."""
        # Build context with current system state
        context_parts = []
        
        try:
            # Get current values based on topic
            if topic == "current_signal":
                signal = self._get_current_signal()
                context_parts.append(f"Current signal data: {signal}")
            elif topic == "technical_indicators":
                indicators = self._get_indicators_for_dashboard()
                context_parts.append(f"Current indicators: {indicators}")
            elif topic == "market_regime":
                regime = self._get_regime()
                context_parts.append(f"Current regime: {regime}")
            elif topic == "risk_manager":
                risk = self._get_risk_stats()
                context_parts.append(f"Risk stats: {risk}")
            elif topic == "multi_timeframe":
                mtf = self._get_mtf_analysis()
                context_parts.append(f"MTF analysis: {mtf}")
            elif topic == "ai_filter":
                ai_stats = self.ai_filter.get_stats()
                context_parts.append(f"AI stats: {ai_stats}")
                context_parts.append(f"AI mode: {self.ai_mode}")
            elif topic == "current_position":
                pos = self._get_current_position_dict()
                context_parts.append(f"Position: {pos}")
                # Use the ACTUAL position symbol, not the bot's main symbol
                if pos and pos.get('symbol'):
                    pos_symbol = pos.get('symbol', '').replace('/USDT:USDT', '').replace(':USDT', '').replace('/', '')
                    context_parts.insert(0, f"🚨 POSITION SYMBOL: {pos_symbol} - This is the symbol the user has a position in!")
            elif topic == "ml_model":
                ml = self._get_ml_status()
                context_parts.append(f"ML status: {ml}")
            elif topic == "trading_parameters":
                params = self._get_system_params()
                context_parts.append(f"Parameters: {params}")
            elif topic == "live_price_chart":
                context_parts.append(f"Symbol: {self.SYMBOL}")
                context_parts.append(f"Current price: ${self.cached_last_price:.4f}" if self.cached_last_price else "Price: Loading...")
                context_parts.append(f"ATR: {self.cached_last_atr:.4f}" if self.cached_last_atr else "ATR: Loading...")
                regime = self._get_regime()
                context_parts.append(f"Market regime: {regime.get('regime', 'unknown') if regime else 'unknown'}")
            elif topic == "equity_curve":
                context_parts.append(f"Starting balance: ${self.initial_balance:,.2f}")
                context_parts.append(f"Current balance: ${self.balance:,.2f}")
                pnl = self.balance - self.initial_balance
                pnl_pct = (pnl / self.initial_balance) * 100
                context_parts.append(f"Total P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
                context_parts.append(f"Equity curve points: {len(self.equity_curve)}")
                context_parts.append(f"Peak balance: ${self.peak_balance:,.2f}")
                drawdown = ((self.peak_balance - self.balance) / self.peak_balance) * 100 if self.peak_balance > 0 else 0
                context_parts.append(f"Current drawdown: {drawdown:.2f}%")
            
            # For position-related topics, use the actual position symbol
            # For other topics, use the bot's main symbol
            if topic == "current_position":
                # Symbol already added above from position data
                pass
            else:
                # ALWAYS include current symbol to prevent hallucination
                current_symbol = self.symbol if hasattr(self, 'symbol') else self.SYMBOL
                context_parts.insert(0, f"🚨 CURRENT TRADING SYMBOL: {current_symbol} - DO NOT mention any other symbol!")
            
            context = "\\n".join(context_parts)
        except Exception as e:
            context = f"Error getting context: {e}"
        
        # Create prompt for AI
        prompt = f"""You are Julaba's AI assistant explaining a dashboard section to the user.

The user clicked the info button on: **{display_name}**

Current system data:
{context}

Please explain:
1. What this section shows and why it's important
2. What the current values mean
3. Any actionable insights based on current data
4. Tips for using this information in trading decisions

Keep your response concise (under 200 words), friendly, and educational. Use bullet points where helpful.
Format with markdown for readability."""

        try:
            explanation = self.ai_filter._generate_content(prompt)
            return explanation if explanation else "AI explanation temporarily unavailable."
        except Exception as e:
            logger.error(f"AI explanation error: {e}")
            return f"Could not generate explanation: {str(e)}"
    
    # === DYNAMIC MARKET SCAN - ALL TRADEABLE BYBIT PAIRS ===
    # Fallback pairs if dynamic fetch fails
    FALLBACK_SCAN_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "INJUSDT", "SUIUSDT", 
                           "ADAUSDT", "SEIUSDT", "XRPUSDT", "NEARUSDT", "APTUSDT",
                           "WLDUSDT", "DOGEUSDT", "AVAXUSDT", "ARBUSDT", "OPUSDT"]
    
    # Dynamic pair discovery settings
    MIN_VOLUME_USD = 10_000_000  # Minimum $10M 24h volume
    MAX_SCAN_PAIRS = 50  # Scan top 50 liquid pairs max (balance between coverage and speed)
    _tradeable_pairs_cache = {"pairs": None, "timestamp": 0}
    TRADEABLE_PAIRS_CACHE_DURATION = 6 * 3600  # Refresh pair list every 6 hours
    
    # Blacklist: Ultra-volatile meme coins, illiquid pairs, or pairs with known issues
    # These are excluded even if they have high volume (often pump-and-dump)
    PAIR_BLACKLIST = {
        "RIVERUSDT", "FARTCOINUSDT", "WHITEWHALEUSDT", "PUMPFUNUSDT",  # Pure meme/pump coins
        "1000PEPEUSDT", "1000BONKUSDT", "1000FLOKIUSDT", "1000SHIBUSDT",  # Meme derivatives
        "XAUTUSDT", "PAXGUSDT",  # Gold-backed (different dynamics)
        "SHIB1000USDT", "1000RATSUSDT", "1000CATUSDT",  # More memes
        "WLFIUSDT", "TRUMPUSDT", "MELANIAUSDT",  # Political meme coins - extremely volatile
    }
    
    # Autonomous pair switch settings
    AUTO_SWITCH_ENABLED = True
    AUTO_SWITCH_INTERVAL = 90  # Check every 90 seconds for better responsiveness
    AUTO_SWITCH_MIN_SCORE_DIFF = 8  # Lowered from 10 - be more responsive to better opportunities
    _last_auto_switch_check = 0
    _last_switch_time = 0  # Track when last switch happened (for cooldown)
    SWITCH_COOLDOWN_MINUTES = 1  # Minimum time between pair switches (prevent excessive switching)
    _full_scan_cache = {"data": None, "timestamp": 0}
    FULL_SCAN_CACHE_DURATION = 180  # Cache for 3 minutes (PhD math is expensive)
    
    def _get_tradeable_pairs(self) -> List[str]:
        """Dynamically fetch all tradeable USDT perpetual pairs from Bybit.
        
        Filters by:
        - Active USDT perpetual contracts
        - Minimum 24h volume threshold
        - Returns top pairs by volume
        """
        import time
        current_time = time.time()
        
        # Return cached pairs if still valid
        if (type(self)._tradeable_pairs_cache["pairs"] and 
            current_time - type(self)._tradeable_pairs_cache["timestamp"] < type(self).TRADEABLE_PAIRS_CACHE_DURATION):
            return type(self)._tradeable_pairs_cache["pairs"]
        
        try:
            import ccxt as ccxt_sync
            sync_exchange = ccxt_sync.bybit({
                "enableRateLimit": True,
                "options": {"defaultType": "swap", "defaultSubType": "linear", "recvWindow": 20000}
            })
            
            # Load markets and tickers
            markets = sync_exchange.load_markets()
            tickers = sync_exchange.fetch_tickers()
            
            # Filter and sort by volume
            tradeable = []
            blacklisted_count = 0
            for symbol, ticker in tickers.items():
                # Only USDT perpetuals
                if not symbol.endswith(':USDT'):
                    continue
                    
                market = markets.get(symbol, {})
                if not market.get('active', True):
                    continue
                
                vol_24h = ticker.get('quoteVolume', 0) or 0
                if vol_24h < type(self).MIN_VOLUME_USD:
                    continue
                
                # Convert symbol format: "BTC/USDT:USDT" -> "BTCUSDT"
                clean_symbol = symbol.replace('/USDT:USDT', 'USDT')
                
                # Skip blacklisted pairs (meme coins, etc.)
                if clean_symbol in type(self).PAIR_BLACKLIST:
                    blacklisted_count += 1
                    continue
                    
                tradeable.append((clean_symbol, vol_24h))
            
            # Sort by volume (highest first) and take top N
            tradeable.sort(key=lambda x: x[1], reverse=True)
            pairs = [p[0] for p in tradeable[:type(self).MAX_SCAN_PAIRS]]
            
            # Cache the result
            type(self)._tradeable_pairs_cache = {
                "pairs": pairs,
                "timestamp": current_time
            }
            
            logger.info(f"🔍 Dynamic pair discovery: Found {len(tradeable)} pairs with >${type(self).MIN_VOLUME_USD/1e6:.0f}M volume (excluded {blacklisted_count} blacklisted), using top {len(pairs)}")
            
            return pairs
            
        except Exception as e:
            logger.warning(f"Dynamic pair fetch failed: {e}, using fallback list")
            return type(self).FALLBACK_SCAN_PAIRS

    def _get_market_scan_data_full(self) -> Dict[str, Any]:
        """Get multi-pair market data WITH full indicator calculations."""
        import time
        current_time = time.time()
        
        # Return cached data if still valid
        if (type(self)._full_scan_cache["data"] and 
            current_time - type(self)._full_scan_cache["timestamp"] < type(self).FULL_SCAN_CACHE_DURATION):
            cached = type(self)._full_scan_cache["data"].copy()
            cached["cached"] = True
            cached["cache_age"] = int(current_time - type(self)._full_scan_cache["timestamp"])
            # ALWAYS refresh position data even on cached results (positions change frequently)
            symbols_with_positions = []
            if hasattr(self, 'positions') and self.positions:
                symbols_with_positions = [sym for sym, pos in self.positions.items() if pos is not None]
            if hasattr(self, 'position') and self.position:
                pos_symbol = getattr(self.position, 'symbol', '').replace('/', '')
                if pos_symbol and pos_symbol not in symbols_with_positions:
                    symbols_with_positions.append(pos_symbol)
            cached["symbols_with_positions"] = symbols_with_positions
            cached["position_count"] = len(symbols_with_positions)
            return cached
        
        pairs_data = []
        
        try:
            import ccxt as ccxt_sync
            sync_exchange = ccxt_sync.bybit({
                "enableRateLimit": True,
                "options": {"defaultType": "swap", "defaultSubType": "linear", "recvWindow": 20000}
            })
            
            # Get dynamic tradeable pairs
            scan_pairs = self._get_tradeable_pairs()
            
            for symbol in scan_pairs:
                try:
                    # Fetch OHLCV data (100 bars of 15m = 25 hours of data)
                    ohlcv = sync_exchange.fetch_ohlcv(symbol, '15m', limit=100)
                    
                    if len(ohlcv) < 50:
                        continue
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # Calculate indicators
                    rsi = calculate_rsi(df['close'], period=14)
                    current_rsi = float(rsi.iloc[-1]) if len(rsi) > 0 else 50
                    
                    adx = calculate_adx(df, period=14)
                    
                    # MACD
                    ema12 = df['close'].ewm(span=12).mean()
                    ema26 = df['close'].ewm(span=26).mean()
                    macd = ema12 - ema26
                    macd_signal = macd.ewm(span=9).mean()
                    macd_hist = float(macd.iloc[-1] - macd_signal.iloc[-1])
                    macd_bullish = macd_hist > 0
                    
                    # Volatility (ATR-based)
                    atr = calculate_atr(df, period=14)
                    current_atr = float(atr.iloc[-1]) if len(atr) > 0 else 0
                    current_price = float(df['close'].iloc[-1])
                    atr_pct = (current_atr / current_price * 100) if current_price > 0 else 0
                    
                    # Price momentum (% change over last 6 bars = 1.5 hours)
                    momentum = ((df['close'].iloc[-1] / df['close'].iloc[-6]) - 1) * 100 if len(df) >= 6 else 0
                    
                    # Volume trend
                    vol_sma = df['volume'].rolling(20).mean()
                    volume_ratio = float(df['volume'].iloc[-1] / vol_sma.iloc[-1]) if vol_sma.iloc[-1] > 0 else 1
                    
                    # =================================================================
                    # ALPHA LAYER INTEGRATION (Single Source of Truth)
                    # =================================================================
                    # Run indicator.py strategy logic on this pair's dataframe
                    df_signaled = generate_signals(df.copy())
                    
                    signal = 0
                    signal_text = "none"
                    
                    # Read the 'Side' output generated by indicator.py
                    if 'Side' in df_signaled.columns:
                        for i in range(-3, 0):
                            val = df_signaled['Side'].iloc[i]
                            if val == 1:
                                signal = 1
                                signal_text = "LONG"
                                break
                            elif val == -1:
                                signal = -1
                                signal_text = "SHORT"
                                break
                                
                    # Display metrics for UI
                    sma40_val = df['close'].rolling(40).mean().iloc[-1]
                    trend = "bullish" if df['close'].iloc[-1] > sma40_val else "bearish"
                    trend_strength = 0
                    # =================================================================
                    
                    # === STABILITY SCORE: Count EMA crossings (fewer = more stable) ===
                    ema20 = df['close'].ewm(span=20).mean()
                    price_above_ema = df['close'] > ema20
                    ema_crossings = (price_above_ema != price_above_ema.shift(1)).sum()
                    # High crossings (>15 in 100 bars) = unstable, Low (<8) = stable
                    stability_score = max(0, 1 - (ema_crossings - 8) / 20)  # 0-1 scale
                    
                    # Calculate basic tradability score (0-100)
                    basic_score = self._calculate_pair_score(
                        rsi=current_rsi, adx=adx, atr_pct=atr_pct,
                        volume_ratio=volume_ratio, macd_bullish=macd_bullish,
                        trend_strength=trend_strength, signal=signal,
                        stability=stability_score
                    )
                    
                    # === PhD MATH SCORING FOR BOTH DIRECTIONS ===
                    math_long = 0
                    math_short = 0
                    best_direction = "NONE"
                    ai_score = 0
                    combined_score = basic_score
                    
                    if hasattr(self, 'ai_filter') and self.ai_filter:
                        try:
                            # Build context for PhD math analysis
                            context = self.ai_filter._build_market_context(df, current_price, current_atr)
                            context['symbol'] = symbol  # For component debug logging
                            
                            # Get PhD-level math scores for BOTH directions
                            long_check = self.ai_filter._comprehensive_math_check(1, df, current_price, current_atr, context)
                            short_check = self.ai_filter._comprehensive_math_check(-1, df, current_price, current_atr, context)
                            
                            math_long = long_check.get('score', 0)
                            math_short = short_check.get('score', 0)
                            
                            # Determine best direction based on PhD math
                            if math_long > math_short and math_long >= 45:
                                best_direction = "LONG"
                                best_math_score = math_long
                            elif math_short > math_long and math_short >= 45:
                                best_direction = "SHORT"
                                best_math_score = math_short
                            else:
                                best_direction = "NONE"
                                best_math_score = max(math_long, math_short)
                            
                            # Combined score: Math (60%) + Basic indicators (40%)
                            combined_score = (best_math_score * 0.6) + (basic_score * 0.4)
                            
                        except Exception as math_err:
                            logger.debug(f"Math scoring error for {symbol}: {math_err}")
                            combined_score = basic_score
                    
                    # 24h change from ticker
                    ticker = sync_exchange.fetch_ticker(symbol)
                    change_24h = ticker.get('percentage', 0) or 0
                    volume_24h = ticker.get('quoteVolume', 0) or 0
                    
                    # === SMART ZONE DETECTION (Bounce-based) ===
                    high_24h = float(df['high'].max())
                    low_24h = float(df['low'].min())
                    
                    # Find swing highs (resistance bounces)
                    cluster_tolerance = current_atr * 0.5
                    resistance_bounces = []
                    support_bounces = []
                    
                    for i in range(2, len(df) - 2):
                        high_val = float(df['high'].iloc[i])
                        low_val = float(df['low'].iloc[i])
                        
                        # Swing high detection (resistance)
                        is_swing_high = (
                            high_val > float(df['high'].iloc[i-1]) and
                            high_val > float(df['high'].iloc[i-2]) and
                            high_val > float(df['high'].iloc[i+1]) and
                            high_val > float(df['high'].iloc[i+2])
                        )
                        if is_swing_high:
                            next_close = float(df['close'].iloc[i+1])
                            if next_close < high_val * 0.995:  # Bounced down
                                resistance_bounces.append(high_val)
                        
                        # Swing low detection (support)
                        is_swing_low = (
                            low_val < float(df['low'].iloc[i-1]) and
                            low_val < float(df['low'].iloc[i-2]) and
                            low_val < float(df['low'].iloc[i+1]) and
                            low_val < float(df['low'].iloc[i+2])
                        )
                        if is_swing_low:
                            next_close = float(df['close'].iloc[i+1])
                            if next_close > low_val * 1.005:  # Bounced up
                                support_bounces.append(low_val)
                    
                    # Cluster bounces to find strongest levels
                    def find_strongest_level(bounces, tolerance, fallback, is_resistance=True):
                        if not bounces:
                            return fallback, 0
                        bounces = sorted(bounces, reverse=is_resistance)
                        clusters = []
                        current_cluster = [bounces[0]]
                        for b in bounces[1:]:
                            if abs(b - current_cluster[-1]) <= tolerance:
                                current_cluster.append(b)
                            else:
                                clusters.append((sum(current_cluster)/len(current_cluster), len(current_cluster)))
                                current_cluster = [b]
                        if current_cluster:
                            clusters.append((sum(current_cluster)/len(current_cluster), len(current_cluster)))
                        # Find cluster with most touches
                        clusters.sort(key=lambda x: -x[1])
                        if clusters and clusters[0][1] >= 2:
                            return clusters[0][0], clusters[0][1]
                        return fallback, 0
                    
                    resistance_level, resistance_touches = find_strongest_level(
                        [b for b in resistance_bounces if b > current_price],
                        cluster_tolerance, high_24h, True
                    )
                    support_level, support_touches = find_strongest_level(
                        [b for b in support_bounces if b < current_price],
                        cluster_tolerance, low_24h, False
                    )
                    
                    # Calculate zone boundaries - REDUCED zone width
                    zone_width = current_atr * 0.5  # CHANGED from 1.5x to 0.5x ATR
                    min_zone = current_price * 0.003  # CHANGED from 1% to 0.3%
                    max_zone = current_price * 0.015  # CHANGED from 5% to 1.5%
                    zone_width = max(min_zone, min(max_zone, zone_width))
                    shadow_width = zone_width * 1.5
                    
                    # Danger zones (centered on bounce level)
                    resistance_upper = resistance_level
                    resistance_lower = resistance_level - zone_width
                    support_upper = support_level + zone_width
                    support_lower = support_level
                    
                    # Caution/shadow zones
                    resistance_caution_lower = resistance_lower - shadow_width
                    support_caution_upper = support_upper + shadow_width
                    
                    # Check current price position in zones
                    in_resistance_zone = resistance_lower <= current_price <= resistance_upper
                    in_support_zone = support_lower <= current_price <= support_upper
                    in_resistance_caution = resistance_caution_lower <= current_price < resistance_lower
                    in_support_caution = support_upper < current_price <= support_caution_upper
                    
                    pairs_data.append({
                        "symbol": symbol,
                        "price": current_price,
                        "change": change_24h,
                        "volatility": atr_pct,
                        "volume": volume_24h,
                        "high": high_24h,
                        "low": low_24h,
                        # === ZONE DATA (bounce-based detection) ===
                        "zone_width": round(zone_width, 6),
                        "resistance_upper": round(resistance_upper, 6),
                        "resistance_lower": round(resistance_lower, 6),
                        "resistance_caution_lower": round(resistance_caution_lower, 6),
                        "resistance_level": round(resistance_level, 6),
                        "resistance_touches": resistance_touches,
                        "support_upper": round(support_upper, 6),
                        "support_lower": round(support_lower, 6),
                        "support_caution_upper": round(support_caution_upper, 6),
                        "support_level": round(support_level, 6),
                        "support_touches": support_touches,
                        "in_resistance_zone": in_resistance_zone,
                        "in_support_zone": in_support_zone,
                        "in_resistance_caution": in_resistance_caution,
                        "in_support_caution": in_support_caution,
                        # Indicator fields
                        "rsi": round(current_rsi, 1),
                        "adx": round(adx, 1),
                        "macd_bullish": macd_bullish,
                        "trend": trend,
                        "trend_strength": round(trend_strength, 2),
                        "volume_ratio": round(volume_ratio, 2),
                        "momentum": round(momentum, 2),
                        "signal": signal,
                        "signal_text": signal_text,
                        # === PhD MATH + AI SCORES ===
                        "basic_score": round(basic_score, 1),
                        "math_long": round(math_long, 1),
                        "math_short": round(math_short, 1),
                        "best_direction": best_direction,
                        "score": round(combined_score, 1)  # Combined score for ranking
                    })
                    
                except Exception as e:
                    logger.debug(f"Error scanning {symbol}: {e}")
                    continue
            
            # Sort by score (highest first)
            pairs_data.sort(key=lambda x: x.get('score', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Full market scan error: {e}")
        
        # Get list of symbols with ACTUAL open positions (only non-None positions)
        symbols_with_positions = []
        if hasattr(self, 'positions') and self.positions:
            symbols_with_positions = [sym for sym, pos in self.positions.items() if pos is not None]
        # Also add primary position if exists
        if hasattr(self, 'position') and self.position:
            pos_symbol = getattr(self.position, 'symbol', '').replace('/', '')
            if pos_symbol and pos_symbol not in symbols_with_positions:
                symbols_with_positions.append(pos_symbol)
        
        result = {
            "pairs": pairs_data,
            "current_symbol": self.SYMBOL,
            "best_pair": pairs_data[0] if pairs_data else None,
            "cached": False,
            "timestamp": current_time,
            # Multi-pair mode info
            "multi_pair_enabled": len(self.additional_symbols) > 0,
            "multi_pair_count": 1 + len(self.additional_symbols),
            "active_pairs": [self.SYMBOL] + self.additional_symbols,
            # ACTUAL open positions (for highlighting in scanner)
            "symbols_with_positions": symbols_with_positions,
            "position_count": len(symbols_with_positions),
            # Dynamic scan info
            "scan_pair_count": len(scan_pairs),
            "scan_source": "dynamic" if len(scan_pairs) > len(type(self).FALLBACK_SCAN_PAIRS) else "fallback"
        }
        
        # Cache the result
        type(self)._full_scan_cache = {"data": result, "timestamp": current_time}
        
        return result

    def _calculate_pair_score(self, rsi: float, adx: float, atr_pct: float,
                              volume_ratio: float, macd_bullish: bool,
                              trend_strength: float, signal: int,
                              stability: float = 1.0) -> float:
        """
        Calculate a tradability score (0-100) for a pair.
        
        Scoring breakdown:
        - ADX (trend strength): 0-25 points
        - Volatility (ATR%): 0-15 points  
        - Volume: 0-15 points
        - RSI positioning: 0-15 points
        - Signal presence: 0-15 points
        - MACD alignment: 0-5 points
        - Stability bonus: 0-10 points (new)
        """
        score = 0
        
        # ADX: Higher is better for trending (sweet spot 25-50)
        if adx >= 25:
            adx_score = min(25, (adx - 10) * 0.8)
        else:
            adx_score = adx * 0.5
        score += adx_score
        
        # Volatility: Moderate ATR% is best (2-5%), too high = unstable
        if 2.0 <= atr_pct <= 5.0:
            vol_score = 15  # Sweet spot
        elif atr_pct >= 1.5:
            vol_score = min(12, atr_pct * 2)
        else:
            vol_score = atr_pct * 5
        # Penalize very high volatility (>8%)
        if atr_pct > 8:
            vol_score = max(0, vol_score - 5)
        score += vol_score
        
        # Volume: Above average is good
        if volume_ratio >= 1.0:
            vol_mult_score = min(15, volume_ratio * 7)
        else:
            vol_mult_score = volume_ratio * 10
        score += vol_mult_score
        
        # RSI: Best when not at extremes (40-60 is ideal for entries)
        if 35 <= rsi <= 65:
            rsi_score = 15
        elif 25 <= rsi <= 75:
            rsi_score = 10
        else:
            # Extremes can be good for reversal plays
            rsi_score = 8 if (rsi < 25 or rsi > 75) else 5
        score += rsi_score
        
        # Signal presence: Recent signal is highly valuable
        if signal != 0:
            score += 15
        
        # MACD alignment with trend (reduced from 10 to 5)
        # MACD must AGREE with trend direction (bullish MACD + uptrend, or bearish MACD + downtrend)
        if (macd_bullish and trend_strength > 0.5 and signal >= 0) or (not macd_bullish and trend_strength > 0.5 and signal <= 0):
            score += 5
        elif (macd_bullish and signal >= 0) or (not macd_bullish and signal <= 0):
            score += 2
        # No points if MACD contradicts trend direction
        
        # Stability bonus: More stable = higher score (0-10 points)
        stability_bonus = stability * 10
        score += stability_bonus
        
        return min(100, max(0, score))

    async def _unified_math_ai_scan(self) -> Dict[str, Any]:
        """
        UNIFIED MATH + AI SCANNER
        
        This is the MASTER scanner that combines:
        1. Multi-pair data fetch (all tradeable pairs)
        2. Math scoring for each pair in both directions
        3. AI strategic analysis on top candidates
        4. Returns the BEST opportunity with full context
        
        IMPORTANT: Only considers pairs from the dashboard's top opportunities
        to ensure consistency between what user sees and what bot trades.
        
        Used when AI_FULL_AUTONOMY = True for maximum decision power.
        """
        try:
            logger.info("🧠 UNIFIED MATH+AI SCAN starting...")
            
            # === BALANCE PROTECTION: Stop trading below threshold ===
            MIN_BALANCE_TO_TRADE = getattr(self, 'balance_protection_threshold', 320)
            current_balance = getattr(self, 'balance', 0)
            if current_balance < MIN_BALANCE_TO_TRADE:
                logger.warning(f"🛑 BALANCE PROTECTION: ${current_balance:.2f} < ${MIN_BALANCE_TO_TRADE} - NO NEW TRADES")
                return {
                    'has_opportunity': False,
                    'reason': f'Balance protection: ${current_balance:.2f} < ${MIN_BALANCE_TO_TRADE} minimum'
                }
            
            # Check if we have room for more positions
            open_positions = [p for p in self.positions.values() if p is not None]
            max_pos_param = getattr(self, 'adaptive_params', {}).get('max_concurrent_positions', {})
            # adaptive_params stores nested dicts with 'current' key
            max_positions = max_pos_param.get('current', 2) if isinstance(max_pos_param, dict) else 2
            
            if len(open_positions) >= max_positions:
                return {
                    'has_opportunity': False,
                    'reason': f'Max positions reached ({len(open_positions)}/{max_positions})'
                }
            
            # Get full market scan data (same as dashboard uses)
            # This ensures we trade from the SAME ranked list the user sees
            market_scan = self._get_market_scan_data_full()
            scan_pairs_with_scores = market_scan.get('pairs', [])
            
            if not scan_pairs_with_scores:
                # Fallback to volume-based selection
                scan_pairs = self._get_tradeable_pairs()[:10]
                pairs_info = {}  # No pre-calculated scores
            else:
                # Use top 10 by SCORE (same ranking as dashboard shows)
                # Sort by score descending to get best opportunities first
                scan_pairs_with_scores.sort(key=lambda x: x.get('score', 0), reverse=True)
                scan_pairs = [p['symbol'] for p in scan_pairs_with_scores[:10]]
                
                # Store pre-calculated math scores from dashboard scan
                pairs_info = {p['symbol']: p for p in scan_pairs_with_scores[:10]}
                
                # Log which pairs we're considering with their PhD math scores
                top_10_info = []
                for p in scan_pairs_with_scores[:10]:
                    info = f"{p['symbol']}(L:{p.get('math_long', 0):.0f}/S:{p.get('math_short', 0):.0f}={p.get('score', 0):.0f})"
                    top_10_info.append(info)
                logger.info(f"📊 Dashboard TOP 10 (PhD Math): {', '.join(top_10_info)}")
            
            # Prepare data for all pairs
            pairs_data = []
            
            # Use sync exchange for scanning
            import ccxt as ccxt_sync
            sync_exchange = ccxt_sync.bybit({
                "enableRateLimit": True,
                "options": {"defaultType": "swap", "defaultSubType": "linear", "recvWindow": 20000}
            })
            
            for symbol in scan_pairs:
                try:
                    # Fetch OHLCV data
                    ohlcv = sync_exchange.fetch_ohlcv(symbol, '15m', limit=100)
                    
                    if len(ohlcv) < 50:
                        continue
                    
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # Get current price and ATR
                    current_price = float(df['close'].iloc[-1])
                    
                    # Calculate ATR
                    from indicator import calculate_atr
                    atr_series = calculate_atr(df, period=14)
                    current_atr = float(atr_series.iloc[-1]) if len(atr_series) > 0 else current_price * 0.02
                    
                    # Volume ratio
                    vol_sma = df['volume'].rolling(20).mean()
                    volume_ratio = float(df['volume'].iloc[-1] / vol_sma.iloc[-1]) if vol_sma.iloc[-1] > 0 else 1
                    
                    # Get pre-calculated PhD math scores from dashboard scan
                    pair_info = pairs_info.get(symbol, {}) if pairs_info else {}
                    
                    pairs_data.append({
                        'symbol': symbol,
                        'df': df,
                        'price': current_price,
                        'atr': current_atr,
                        'volume_ratio': volume_ratio,
                        # Pre-calculated scores from dashboard (same PhD math)
                        'dashboard_score': pair_info.get('score', 0),
                        'dashboard_math_long': pair_info.get('math_long', 0),
                        'dashboard_math_short': pair_info.get('math_short', 0),
                        'dashboard_best_direction': pair_info.get('best_direction', 'NONE'),
                        'dashboard_rsi': pair_info.get('rsi', 50),
                        'dashboard_trend': pair_info.get('trend', 'neutral')
                    })
                    
                except Exception as e:
                    logger.debug(f"Error fetching {symbol}: {e}")
                    continue
            
            if not pairs_data:
                return {'has_opportunity': False, 'reason': 'No pairs data available'}
            
            # Call the unified scanner in ai_filter
            result = self.ai_filter.unified_math_ai_scan(
                pairs_data=pairs_data,
                current_positions=open_positions,
                balance=self.balance,
                max_positions=max_positions,
                allowed_sides=getattr(self, 'allowed_sides', 'both'),
                atomic_candle=None
            )
            
            if result.get('has_opportunity'):
                best = result.get('best_pair', {})
                logger.info(f"🎯 UNIFIED SCAN found: {best.get('symbol')} {best.get('direction')}")
                logger.info(f"   Math: {best.get('math_score', 0):.0f} | AI: {best.get('ai_score', 0):.0f} | Combined: {best.get('combined_score', 0):.0f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Unified Math+AI scan error: {e}")
            return {'has_opportunity': False, 'reason': str(e)}

    async def _autonomous_pair_check(self):
        """Check if we should auto-switch to a better pair (autonomous mode).
        
        AI validates the switch decision before executing to prevent
        unnecessary switching or switching to unsuitable pairs.
        """
        import time
        current_time = time.time()
        
        # === EARLY BALANCE PROTECTION: Don't even try to scan if balance too low ===
        BALANCE_PROTECTION_THRESHOLD = getattr(self, 'balance_protection_threshold', 320)
        current_balance = getattr(self, 'balance', 0)
        if current_balance <= BALANCE_PROTECTION_THRESHOLD:
            # Log only occasionally to avoid spam
            if not hasattr(type(self), '_last_balance_warning') or (current_time - type(self)._last_balance_warning) > 300:
                type(self)._last_balance_warning = current_time
                logger.warning(f"🛑 BALANCE PROTECTION ACTIVE: ${current_balance:.2f} <= ${BALANCE_PROTECTION_THRESHOLD} - ALL TRADING SUSPENDED")
            return
        
        # Only check periodically
        time_since_last = current_time - type(self)._last_auto_switch_check
        if time_since_last < type(self).AUTO_SWITCH_INTERVAL:
            return
        
        type(self)._last_auto_switch_check = current_time
        logger.info(f"🔍 Autonomous pair check running (interval: {type(self).AUTO_SWITCH_INTERVAL}s)")
        
        # === SWITCH COOLDOWN CHECK (prevent excessive switching) ===
        if type(self)._last_switch_time > 0:
            mins_since_switch = (current_time - type(self)._last_switch_time) / 60
            if mins_since_switch < type(self).SWITCH_COOLDOWN_MINUTES:
                logger.info(f"📊 Switch cooldown: {mins_since_switch:.1f}/{type(self).SWITCH_COOLDOWN_MINUTES}min - skipping pair check")
                return
        
        # Only in autonomous mode
        if self.ai_mode != "autonomous":
            logger.debug(f"Skipping pair check: not in autonomous mode (mode={self.ai_mode})")
            return
        
        if not type(self).AUTO_SWITCH_ENABLED:
            return
        
        try:
            logger.info(f"📊 Pair check: Fetching market scan data...")
            # Get full market scan
            scan_data = self._get_market_scan_data_full()
            pairs = scan_data.get("pairs", [])
            
            logger.info(f"📊 Pair check: got {len(pairs)} pairs from scan (cached={scan_data.get('cached', False)})")
            
            if not pairs:
                logger.info(f"📊 Pair check: No pairs returned from scan - aborting")
                return
            
            # Get best pair for later use
            best_pair = pairs[0] if pairs else None
            
            # Check current open positions
            open_positions = [p for p in self.positions.values() if p is not None]
            max_positions = self.multi_pair_config.max_total_positions
            
            logger.info(f"📊 Pair check: {len(open_positions)}/{max_positions} positions open, best_pair={best_pair.get('symbol') if best_pair else None}")
            
            # LOGIC FLOW:
            # 1. If NO positions: Can switch to any pair (old behavior)
            # 2. If BELOW max positions: Can open new pair (multi-pair enabled)
            # 3. If AT max positions: Can REPLACE worst position with best pair (new Option 3)
            
            # === CASE 1: No positions (Normal pair switching) ===
            if len(open_positions) == 0:
                logger.info(f"📊 Pair check: No positions open - can switch to any pair")
                
                # === FULL PAIR SCAN: When all positions closed, switch to BEST pair immediately ===
                if self._needs_full_pair_scan or True:  # Always compare when no positions
                    best_pair = pairs[0]
                    best_symbol = best_pair.get('symbol', '')
                    best_score = best_pair.get('score', 0)
                    
                    current_symbol_normalized = normalize_symbol(self.SYMBOL)
                    best_symbol_normalized = normalize_symbol(best_symbol)
                    
                    if best_symbol_normalized != current_symbol_normalized:
                        logger.info(f"🔄 SWITCHING TO BEST PAIR: {best_symbol} (score: {best_score:.1f}) - was on {self.SYMBOL}")
                        
                        # Convert to ccxt format and switch
                        ccxt_symbol = best_symbol if '/' in best_symbol else f"{best_symbol.replace('USDT', '')}/USDT:USDT"
                        self.SYMBOL = ccxt_symbol
                        self._pending_data_refetch = True
                        self._needs_full_pair_scan = False  # Reset the flag
                        
                        # FIX: Reset stale price caches so _last_price doesn't return old symbol's price
                        self._realtime_price = 0.0
                        self.cached_last_price = 0.0
                        
                        # FIX: Add new symbol to WebSocket for real-time price streaming
                        if hasattr(self, 'ws_price_stream') and self.ws_price_stream:
                            self.ws_price_stream.add_symbol(ccxt_symbol)
                        
                        # Save the new symbol
                        self._save_trading_state()
                        logger.info(f"✅ Now scanning {self.SYMBOL} for opportunities")
                        return
                    else:
                        logger.info(f"📊 Already on best pair: {self.SYMBOL} (score: {best_score:.1f})")
                        self._needs_full_pair_scan = False
                
                # Find current pair's score
                current_score = 0
                current_pair_data = None
                for p in pairs:
                    if normalize_symbol(p["symbol"]) == normalize_symbol(self.SYMBOL):
                        current_score = p.get("score", 0)
                        current_pair_data = p
                        break
                
                best_pair = pairs[0]
                best_score = best_pair.get("score", 0)
                score_diff = best_score - current_score
                
                # Log current vs best for visibility
                if normalize_symbol(best_pair["symbol"]) != normalize_symbol(self.SYMBOL):
                    logger.info(
                        f"📊 Pair comparison: {self.SYMBOL}={current_score:.1f} vs {best_pair['symbol']}={best_score:.1f} "
                        f"(diff={score_diff:.1f})"
                    )
                
                # === AI-DRIVEN PAIR SWITCHING ===
                MIN_SCORE_FOR_AI_EVAL = 3
                
                if best_pair["symbol"] != self.SYMBOL and score_diff >= MIN_SCORE_FOR_AI_EVAL:
                    # Perform standard pair switch logic
                    math_decision = await self._validate_pair_switch_math(
                        current_score=current_score,
                        target_score=best_score,
                        current_symbol=self.SYMBOL,
                        target_symbol=best_pair["symbol"],
                        current_data=current_pair_data,
                        target_data=best_pair
                    )
                    
                    for reasoning in math_decision['reasoning']:
                        logger.info(f"📊 Math: {reasoning}")
                    
                    if not math_decision.get('approved', False):
                        logger.info(f"⚠️ Math: Cannot switch {self.SYMBOL} → {best_pair['symbol']}: {math_decision['reason']}")
                        return
                    
                    logger.info(f"✅ Switch approved ({math_decision['confidence']:.0%} confidence): {math_decision['reason']}")
                    
                    ai_decision = await self._ai_validate_pair_switch(
                        current_symbol=self.SYMBOL,
                        current_data=current_pair_data,
                        target_symbol=best_pair["symbol"],
                        target_data=best_pair,
                        score_diff=score_diff,
                        all_pairs=pairs[:5]
                    )
                    
                    if not ai_decision.get("approved", False):
                        logger.info(f"🤖 AI rejected auto-switch {self.SYMBOL} → {best_pair['symbol']}: {ai_decision.get('reason', 'No reason given')}")
                        return
                    
                    ai_reason = ai_decision.get("reason", "Score improvement justified the switch")
                    
                    self._auto_switch_in_progress = True
                    old_symbol = self.SYMBOL
                    switch_result = self._switch_trading_symbol(best_pair["symbol"])
                    self._auto_switch_in_progress = False
                    
                    if switch_result.get("success"):
                        msg = (
                            f"🤖 **AI-Approved Pair Switch**\n\n"
                            f"Switched: {old_symbol} → {best_pair['symbol']}\n\n"
                            f"**AI Reasoning:**\n{ai_reason}\n\n"
                            f"**Scores:**\n"
                            f"• {old_symbol}: {current_score:.1f}\n"
                            f"• {best_pair['symbol']}: {best_score:.1f} (+{score_diff:.1f})\n\n"
                            f"**{best_pair['symbol']} Stats:**\n"
                            f"• RSI: {best_pair.get('rsi', '--')}\n"
                            f"• ADX: {best_pair.get('adx', '--')}\n"
                            f"• Trend: {best_pair.get('trend', '--')}\n"
                            f"• Signal: {best_pair.get('signal_text', 'none')}\n"
                            f"• Volume: {best_pair.get('volume_ratio', 1):.1f}x avg"
                        )
                        
                        logger.info(f"🤖 AI approved switch {old_symbol} → {best_pair['symbol']}: {ai_reason[:50]}...")
                        
                        if self.telegram.enabled:
                            await self.telegram.send_message(msg)
                        
                        self.ai_filter.record_system_message(f"AI approved switch from {old_symbol} to {best_pair['symbol']}: {ai_reason}")
                        
                        self.bars_1m = []
                        self.bars_agg = []
                        await self.fetch_initial_data()
            
            # === CASE 2: Below max positions (Can open new pair) ===
            elif len(open_positions) < max_positions:
                # === BALANCE PROTECTION: Stop trading below threshold ===
                MIN_BALANCE_TO_TRADE = getattr(self, 'balance_protection_threshold', 320)
                current_balance = getattr(self, 'balance', 0)
                if current_balance < MIN_BALANCE_TO_TRADE:
                    logger.warning(f"🛑 BALANCE PROTECTION: ${current_balance:.2f} < ${MIN_BALANCE_TO_TRADE} - NO NEW POSITIONS")
                    return
                
                # === DOUBLE-CHECK position count (prevent race conditions) ===
                fresh_open_positions = [p for p in self.positions.values() if p is not None]
                if len(fresh_open_positions) >= max_positions:
                    logger.info(f"📊 Race condition detected: now at {len(fresh_open_positions)}/{max_positions} - switching to REPLACEMENT mode")
                    await self._check_position_replacement(best_pair, pairs)
                    return
                
                logger.info(f"📊 Pair check: {len(fresh_open_positions)}/{max_positions} positions - scanning for NEW pair to open")
                
                # Cooldown: Don't open second position too quickly after first
                last_open_time = None
                for pos in fresh_open_positions:
                    if hasattr(pos, 'opened_at') and pos.opened_at:
                        if last_open_time is None or pos.opened_at > last_open_time:
                            last_open_time = pos.opened_at
                
                if last_open_time:
                    mins_since_last = (datetime.now(timezone.utc) - last_open_time).total_seconds() / 60
                    MIN_COOLDOWN_MINUTES = 1  # Wait at least 1 min between position opens (was 5)
                    if mins_since_last < MIN_COOLDOWN_MINUTES:
                        logger.info(f"📊 Position cooldown: {mins_since_last:.1f}min since last open (need {MIN_COOLDOWN_MINUTES}min)")
                        return
                
                # Find best pair that we don't already have a position on
                open_symbols = set()
                for pos in fresh_open_positions:
                    open_symbols.add(getattr(pos, 'symbol', '').replace('/', ''))
                
                # Find best available pair (not already in a position)
                best_available = None
                for p in pairs:
                    pair_symbol = p.get('symbol', '').replace('/', '')
                    if pair_symbol not in open_symbols:
                        best_available = p
                        break
                
                if best_available:
                    logger.info(f"📊 Best available pair for second position: {best_available['symbol']} (score: {best_available.get('score', 0):.1f})")
                    
                    # Check if this pair has a strong signal/opportunity
                    signal = best_available.get('signal', 0)
                    score = best_available.get('score', 0)
                    
                    # === PhD-LEVEL ADAPTIVE THRESHOLD FOR SECOND POSITION ===
                    # Get current regime for adaptive threshold
                    regime_info = self._get_regime() if hasattr(self, '_get_regime') else {}
                    current_regime = regime_info.get('regime', 'UNKNOWN')
                    hurst = regime_info.get('hurst', 0.5)
                    volatility = regime_info.get('volatility', 'normal')
                    
                    # Base threshold by regime - LOWERED for more multi-position trading
                    REGIME_THRESHOLDS = {
                        'TRENDING': 40,       # Trends are reliable, aggressive
                        'STRONG_TREND': 35,   # Very strong trends, very aggressive
                        'WEAK_TREND': 45,     # Standard
                        'RANGING': 55,        # Mean-reversion, need higher conviction
                        'CHOPPY': 60,         # Noisy, selective
                        'VOLATILE': 60,       # High risk, only best setups
                        'UNKNOWN': 50
                    }
                    MIN_SCORE_FOR_SECOND_POSITION = REGIME_THRESHOLDS.get(current_regime, 50)
                    
                    # Hurst adjustment: trending (>0.55) = lower threshold, mean-reverting (<0.45) = higher
                    if hurst > 0.55:
                        MIN_SCORE_FOR_SECOND_POSITION -= 5  # More confident in trends
                    elif hurst < 0.45:
                        MIN_SCORE_FOR_SECOND_POSITION += 3  # Slightly more cautious in mean-reversion
                    
                    # Volatility adjustment (reduced penalties - PhD calibration)
                    if volatility == 'high':
                        MIN_SCORE_FOR_SECOND_POSITION += 3  # More selective in high vol
                    elif volatility == 'low':
                        MIN_SCORE_FOR_SECOND_POSITION -= 3   # Slightly more aggressive in calm
                    
                    logger.info(f"📊 PhD Adaptive Threshold: {MIN_SCORE_FOR_SECOND_POSITION} (regime={current_regime}, hurst={hurst:.2f}, vol={volatility})")
                    
                    # Check for same-direction risk (avoid doubling down on same trend)
                    existing_sides = set()
                    for pos in fresh_open_positions:
                        existing_sides.add(getattr(pos, 'side', '').lower())
                    
                    proposed_side = "short" if signal == -1 else "long" if signal == 1 else ("short" if best_available.get('trend') == 'bearish' else "long")
                    
                    # If all existing positions are same side, require higher score for same-side entry
                    # REDUCED penalty from 10 to 5
                    if len(existing_sides) == 1 and proposed_side in existing_sides:
                        MIN_SCORE_FOR_SECOND_POSITION += 5  # Small penalty for concentration risk
                        logger.info(f"📊 Same-direction ({proposed_side}) position - raising threshold to {MIN_SCORE_FOR_SECOND_POSITION}")
                    
                    # === SECOND POSITION: USE UNIFIED PIPELINE ===
                    # The unified scan (_unified_math_ai_scan) already handles multi-pair
                    # through the EXACT same Stage 1→2→3 pipeline as the first position.
                    # We just need to not block it here — the scan runs separately.
                    # This pair-check code only handles symbol SWITCHING for the main symbol.
                    logger.info(f"📊 Second position will be opened by unified scan pipeline (same Stage 1→2→3 logic)")
                    # Don't return — let the main loop continue to the unified scan
                else:
                    logger.debug("📊 No available pairs for second position (all have positions)")
            
            # === CASE 3: At max positions (Try POSITION REPLACEMENT) ===
            elif len(open_positions) >= max_positions:
                logger.info(f"📊 Pair check: {len(open_positions)}/{max_positions} positions at max - checking for REPLACEMENT opportunity")
                await self._check_position_replacement(best_pair, pairs)
                    
        except Exception as e:
            logger.error(f"Autonomous pair check error: {e}")

    async def _check_position_replacement(self, best_pair: Dict, all_pairs: List[Dict]):
        """
        OPTION 3: Position Replacement Logic (ENHANCED)
        When at max positions, check if replacing worst position with best pair is justified.
        
        SAFEGUARDS:
        1. Don't replace positions in heavy drawdown (< -15%)
        2. Don't replace positions opened less than 5 bars ago (too early)
        3. Prefer replacing losing/flat positions over winners
        4. Only replace if score difference is significant (+15 points)
        
        Strategy:
        - Find worst open position (lowest score + not recent + not in deep drawdown)
        - Compare vs best_pair
        - If best_pair is SIGNIFICANTLY better (+15 score diff), replace it
        - Keep the best open position running
        """
        try:
            current_time = datetime.now(timezone.utc)
            MIN_BARS_SINCE_ENTRY = 5  # Don't replace positions opened < 5 bars ago
            MAX_ALLOWED_DRAWDOWN = -15.0  # Don't replace positions in > -15% drawdown
            
            logger.info(f"📊 _check_position_replacement called with best_pair={best_pair.get('symbol') if best_pair else None}, {len(all_pairs)} pairs")
            
            # Find worst and best open positions (with safeguards)
            worst_symbol = None
            worst_pos = None
            worst_score = 100
            worst_pnl_pct = 0
            best_open_symbol = None
            best_open_score = 0
            
            open_positions_list = []
            for symbol, pos in self.positions.items():
                if pos is not None:
                    # Find this symbol in the pairs list to get score
                    pair_score = 0
                    # Normalize both for comparison
                    symbol_normalized = normalize_symbol(symbol)
                    for pair_data in all_pairs:
                        pair_symbol_normalized = normalize_symbol(pair_data.get("symbol", ""))
                        if pair_symbol_normalized == symbol_normalized:
                            pair_score = pair_data.get("score", 0)
                            logger.debug(f"📊 Found score for {symbol}: {pair_score}")
                            break
                    
                    if pair_score == 0:
                        logger.info(f"📊 Position {symbol} not found in scan pairs (may be outside top {len(all_pairs)})")
                    
                    # Calculate position age (in bars, roughly)
                    time_diff = (current_time - pos.opened_at).total_seconds()
                    bars_since_entry = int(time_diff / 900)  # 15min bars = 900 seconds
                    
                    # FIX: Get correct price for THIS position's symbol
                    symbol_key = symbol.upper().replace('/', '').replace(':USDT', '')
                    pos_current_price = pos.entry_price  # Default
                    if hasattr(self, '_symbol_prices') and symbol_key in self._symbol_prices:
                        pos_current_price = self._symbol_prices[symbol_key]
                    elif hasattr(self, '_symbol_prices'):
                        alt_key = symbol_key.replace('USDT', '') + 'USDT'
                        if alt_key in self._symbol_prices:
                            pos_current_price = self._symbol_prices[alt_key]
                    # Only use _last_price if this IS the main symbol
                    if pos_current_price == pos.entry_price:
                        main_symbol_key = self.SYMBOL.replace('/', '').replace(':USDT', '')
                        if symbol_key == main_symbol_key and self._last_price:
                            pos_current_price = self._last_price
                    
                    # Calculate unrealized PnL %
                    unrealized_pnl = pos.unrealized_pnl(pos_current_price)
                    pnl_pct = (unrealized_pnl / (pos.entry_price * pos.size)) * 100 if pos.size > 0 else 0
                    
                    open_positions_list.append((symbol, pos, pair_score, bars_since_entry, pnl_pct))
                    
                    # Track best (always, no restrictions)
                    if pair_score > best_open_score:
                        best_open_score = pair_score
                        best_open_symbol = symbol
                    
                    # Track worst (WITH SAFEGUARDS)
                    # Exclude: positions < 5 bars old OR in deep drawdown (< -15%)
                    if bars_since_entry >= MIN_BARS_SINCE_ENTRY and pnl_pct >= MAX_ALLOWED_DRAWDOWN:
                        if pair_score < worst_score:
                            worst_score = pair_score
                            worst_symbol = symbol
                            worst_pos = pos
                            worst_pnl_pct = pnl_pct
            
            if not worst_symbol or not best_pair:
                if not worst_symbol:
                    logger.debug("⚠️ No replaceable position found (all too new or in heavy drawdown)")
                return
            
            best_pair_symbol = best_pair.get("symbol")
            best_pair_score = best_pair.get("score", 0)
            
            # Find bars_since_entry and pnl for worst position
            worst_bars = 0
            worst_pnl_for_check = 0
            for symbol, pos, score, bars, pnl in open_positions_list:
                if symbol == worst_symbol:
                    worst_bars = bars
                    worst_pnl_for_check = pnl
                    break
            
            # SAFEGUARD: Don't replace negative positions UNLESS they are stale
            # Stale = position has been open for a long time with small loss (dead money)
            STALE_POSITION_BARS = 8  # ~2 hours at 15min bars
            STALE_LOSS_THRESHOLD = -3.0  # Allow replacing positions with up to -3% loss if stale
            SMALL_LOSS_THRESHOLD = -0.5  # Very small losses can be replaced anytime
            
            is_stale = worst_bars >= STALE_POSITION_BARS
            is_small_loss = worst_pnl_for_check >= SMALL_LOSS_THRESHOLD
            is_acceptable_stale_loss = is_stale and worst_pnl_for_check >= STALE_LOSS_THRESHOLD
            
            if worst_pnl_for_check < 0:
                if is_small_loss:
                    logger.info(f"📊 {worst_symbol} has tiny loss ({worst_pnl_for_check:+.2f}%) - allowing replacement")
                elif is_acceptable_stale_loss:
                    logger.info(f"📊 {worst_symbol} is STALE ({worst_bars} bars, {worst_pnl_for_check:+.2f}%) - allowing replacement (dead money)")
                else:
                    logger.info(f"⚠️ Replacement blocked: {worst_symbol} has negative PnL ({worst_pnl_for_check:+.2f}%) and not stale ({worst_bars} bars < {STALE_POSITION_BARS}) - won't lock in losses")
                    return
            
            logger.info(f"📊 Position Replacement Analysis (SAFE):")
            logger.info(f"  Worst replaceable: {worst_symbol} (score: {worst_score:.1f}, PnL: {worst_pnl_pct:+.2f}%, age: {worst_bars} bars)")
            logger.info(f"  Best open: {best_open_symbol} (score: {best_open_score:.1f})")
            logger.info(f"  Best available: {best_pair_symbol} (score: {best_pair_score:.1f})")
            
            # Replacement Threshold: Only replace if best_pair is SIGNIFICANTLY better
            REPLACEMENT_THRESHOLD = 15  # Need +15 point advantage
            score_diff = best_pair_score - worst_score
            
            if score_diff < REPLACEMENT_THRESHOLD:
                logger.info(f"⚠️ Replacement skipped: Score diff ({score_diff:.1f}) below threshold ({REPLACEMENT_THRESHOLD})")
                return
            
            logger.info(f"✅ Replacement justified: {worst_symbol} ({worst_score:.1f}) → {best_pair_symbol} ({best_pair_score:.1f}) [+{score_diff:.1f}]")
            
            # === MATH VALIDATION ===
            # Build minimal current_data for closed position (we only know score)
            replacement_current_data = {
                'symbol': worst_symbol,
                'score': worst_score,
                'score_variance': 150,  # Conservative estimate for unknown position
                'regime': 'UNKNOWN',
                'regime_age_bars': 20
            }
            
            math_decision = await self._validate_pair_switch_math(
                current_score=worst_score,
                target_score=best_pair_score,
                current_symbol=worst_symbol,
                target_symbol=best_pair_symbol,
                current_data=replacement_current_data,  # Minimal data for closed position
                target_data=best_pair
            )
            
            for reasoning in math_decision['reasoning']:
                logger.info(f"📊 Math: {reasoning}")
            
            if not math_decision.get('approved', False):
                logger.info(f"⚠️ Replacement rejected by math: {math_decision['reason']}")
                return
            
            # === AI APPROVAL ===
            ai_decision = await self._ai_validate_pair_switch(
                current_symbol=worst_symbol,
                current_data=replacement_current_data,  # Use same minimal data as math validation
                target_symbol=best_pair_symbol,
                target_data=best_pair,
                score_diff=score_diff,
                all_pairs=all_pairs[:5]
            )
            
            if not ai_decision.get("approved", False):
                logger.info(f"🤖 AI rejected replacement: {ai_decision.get('reason', 'No reason given')}")
                return
            
            logger.info(f"🤖 AI APPROVED position replacement!")
            
            # === EXECUTE REPLACEMENT ===
            # Get current price for the position being closed
            try:
                close_price = await self._get_current_price_for_symbol(worst_symbol)
                if not close_price:
                    close_price = worst_pos.entry_price  # Fallback
                    logger.warning(f"Could not get current price for {worst_symbol}, using entry price as fallback")
            except Exception as e:
                close_price = worst_pos.entry_price
                logger.warning(f"Error getting price for {worst_symbol}: {e}")
            
            # Close worst position PROPERLY (with order execution, balance update, trade recording)
            logger.info(f"🔌 Closing {worst_symbol} (PnL: {worst_pnl_pct:+.2f}%) to make room for {best_pair_symbol}...")
            
            close_reason = f"Position Replacement → {best_pair_symbol} (score +{score_diff:.0f})"
            await self._close_position_by_symbol(worst_symbol, close_reason, close_price)
            
            # Send replacement notification (separate from close notification)
            if self.telegram.enabled:
                pnl_emoji = "🟢" if worst_pnl_pct >= 0 else "🔴"
                msg = (
                    f"🔄 **Position Replacement Complete**\n\n"
                    f"Closed: {worst_symbol} (score: {worst_score:.1f})\n"
                    f"{pnl_emoji} PnL: {worst_pnl_pct:+.2f}% | Age: {worst_bars} bars\n\n"
                    f"Now watching: {best_pair_symbol} (score: {best_pair_score:.1f})\n"
                    f"Improvement: +{score_diff:.1f} points\n\n"
                    f"⏳ Waiting for entry signal on {best_pair_symbol}..."
                )
                await self.telegram.send_message(msg)
            
            logger.info(f"✅ Position {worst_symbol} closed, now watching {best_pair_symbol} for entry")
            
            # Signal will be generated in next bar for the new pair
            
        except Exception as e:
            logger.error(f"Position replacement error: {e}")

    async def _ai_validate_pair_switch(self, current_symbol: str, current_data: Optional[Dict],
                                        target_symbol: str, target_data: Dict, 
                                        score_diff: float, all_pairs: list = None) -> Dict[str, Any]:
        """AI makes the pair switch decision using mathematical analysis.
        
        This is the PRIMARY decision maker, not just a validator.
        Uses comprehensive analysis including:
        - Trend strength and direction
        - Volatility and stability metrics
        - RSI positioning and momentum
        - Volume and liquidity
        - Market regime and Hurst exponent
        - Multi-timeframe alignment
        - Risk-adjusted opportunity assessment
        
        Returns:
            Dict with 'approved' (bool) and 'reason' (str)
        """
        try:
            # === GATHER COMPREHENSIVE MATHEMATICAL ANALYSIS ===
            # Get current regime for context
            regime_info = self._get_regime() or {}
            current_regime = regime_info.get('regime', 'unknown')
            market_hurst = regime_info.get('hurst', 0.5)
            
            # Get MTF analysis if available
            mtf_context = ""
            if self.mtf_analyzer and self.mtf_analyzer.cached_1h is not None:
                try:
                    mtf = self.mtf_analyzer.analyze(pd.DataFrame(self.bars_agg), 1)
                    mtf_context = f"""
**Multi-Timeframe Analysis (Current Pair):**
- 15m Trend: {mtf.get('trends', {}).get('15m', {}).get('direction', 'n/a')}
- 1H Trend: {mtf.get('trends', {}).get('1h', {}).get('direction', 'n/a')}
- Alignment Score: {mtf.get('alignment_score', 0):.0f}%
- MTF Recommendation: {mtf.get('recommendation', 'n/a')}"""
                except Exception:
                    pass
            
            # Build comprehensive current pair analysis
            current_info = ""
            if current_data:
                # Calculate stability indicator from the data
                current_vol = current_data.get('volatility', 0)
                current_adx = current_data.get('adx', 0)
                current_rsi = current_data.get('rsi', 50)
                
                # Stability assessment
                stability_concerns = []
                if current_vol > 5:
                    stability_concerns.append(f"high volatility ({current_vol:.1f}%)")
                if current_adx < 20:
                    stability_concerns.append(f"weak trend (ADX {current_adx:.0f})")
                if current_rsi > 75 or current_rsi < 25:
                    stability_concerns.append(f"extreme RSI ({current_rsi:.0f})")
                
                stability_note = f" ⚠️ Issues: {', '.join(stability_concerns)}" if stability_concerns else " ✅ Stable"
                
                current_info = f"""
**Current Pair: {current_symbol}** {stability_note}
- Score: {current_data.get('score', 0):.1f}/100
- RSI(14): {current_rsi:.1f}
- ADX(14): {current_adx:.1f}
- Trend: {current_data.get('trend', 'unknown')} (strength: {current_data.get('trend_strength', 0):.2f}%)
- Signal: {current_data.get('signal_text', 'none')}
- Volume: {current_data.get('volume_ratio', 1):.1f}x average
- Volatility: {current_vol:.2f}%
- 24h Change: {current_data.get('change', 0):.2f}%
- Momentum: {current_data.get('momentum', 0):.2f}%"""
            else:
                current_info = f"**Current Pair: {current_symbol}** (no data - may indicate data issues)"
            
            # Target pair comprehensive analysis
            target_vol = target_data.get('volatility', 0)
            target_adx = target_data.get('adx', 0)
            target_rsi = target_data.get('rsi', 50)
            
            target_concerns = []
            if target_vol > 5:
                target_concerns.append(f"high volatility ({target_vol:.1f}%)")
            if target_adx < 20:
                target_concerns.append(f"weak trend (ADX {target_adx:.0f})")
            if target_rsi > 75 or target_rsi < 25:
                target_concerns.append(f"extreme RSI ({target_rsi:.0f})")
            if target_data.get('volume_ratio', 1) < 0.7:
                target_concerns.append(f"low volume ({target_data.get('volume_ratio', 1):.1f}x)")
                
            target_positives = []
            if target_adx > 25:
                target_positives.append(f"strong trend (ADX {target_adx:.0f})")
            if target_data.get('signal_text', 'none') != 'none':
                target_positives.append(f"active signal ({target_data.get('signal_text')})")
            if target_data.get('volume_ratio', 1) > 1.2:
                target_positives.append(f"above-avg volume ({target_data.get('volume_ratio', 1):.1f}x)")
            if 2 <= target_vol <= 4:
                target_positives.append(f"good volatility ({target_vol:.1f}%)")
            
            target_assessment = ""
            if target_positives:
                target_assessment += f" ✅ Positives: {', '.join(target_positives)}"
            if target_concerns:
                target_assessment += f" ⚠️ Concerns: {', '.join(target_concerns)}"
            
            target_info = f"""
**Proposed Pair: {target_symbol}** {target_assessment}
- Score: {target_data.get('score', 0):.1f}/100
- RSI(14): {target_rsi:.1f}
- ADX(14): {target_adx:.1f}
- Trend: {target_data.get('trend', 'unknown')} (strength: {target_data.get('trend_strength', 0):.2f}%)
- Signal: {target_data.get('signal_text', 'none')}
- Volume: {target_data.get('volume_ratio', 1):.1f}x average
- Volatility: {target_vol:.2f}%
- 24h Change: {target_data.get('change', 0):.2f}%
- Momentum: {target_data.get('momentum', 0):.2f}%"""

            # Include top alternatives for context
            alternatives_info = ""
            if all_pairs and len(all_pairs) > 2:
                alt_lines = []
                for i, p in enumerate(all_pairs[:5]):
                    if p['symbol'] not in [current_symbol, target_symbol]:
                        alt_lines.append(f"  • {p['symbol']}: Score {p.get('score', 0):.0f}, ADX {p.get('adx', 0):.0f}, {p.get('trend', '')}")
                if alt_lines:
                    alternatives_info = f"\n**Other Top Pairs:**\n" + "\n".join(alt_lines[:3])

            prompt = f"""You are a quantitative analyst evaluating a trading pair switch decision.

{current_info}

{target_info}
{alternatives_info}

**Market Context:**
- Current Regime: {current_regime}
- Hurst Exponent: {market_hurst:.3f} (>0.5 = trending, <0.5 = mean-reverting/choppy)
- Score Difference: +{score_diff:.1f} points ({target_symbol} vs {current_symbol})
{mtf_context}

**YOUR ANALYSIS FRAMEWORK (Mathematical):**

1. **Trend Quality Assessment:**
   - Is the target in a cleaner trend than current? (ADX, trend strength)
   - Is momentum aligned with trend direction?

2. **Risk-Adjusted Opportunity:**
   - Volatility: Is it tradeable (2-5% ideal) or dangerous (>6%)?
   - RSI: Entering at good levels (30-70) or extremes?

3. **Stability Analysis:**
   - High ADX + consistent trend = stable, tradeable
   - Low ADX + high volatility = choppy, avoid

4. **Volume Confirmation:**
   - Is there sufficient liquidity for clean entries/exits?

5. **Comparative Value:**
   - Does target genuinely offer better risk-adjusted opportunity?
   - Or is current pair actually fine and switching would be churn?

**DECISION CRITERIA:**
- APPROVE: Target has meaningfully better setup (cleaner trend, better RSI positioning, good volume)
- REJECT: Target isn't significantly better, or current pair is actually fine, or target has red flags

Be decisive. If current pair has issues (choppy, weak trend, extreme RSI), switching to a cleaner setup is smart.
If current pair is fine and target is only marginally better, avoid unnecessary switching.

Respond with EXACTLY this format:
DECISION: APPROVE or REJECT
REASON: [2-3 sentences with specific numbers explaining your analysis]"""

            # Use AI to evaluate
            response = self.ai_filter._generate_content(prompt)
            
            if not response:
                # AI unavailable - use comprehensive math criteria
                logger.warning("⚠️ AI unavailable - using math criteria for pair switch decision")
                return self._math_pair_switch_decision(current_data, target_data, score_diff, current_symbol, target_symbol)
            
            # Parse response
            response_upper = response.upper()
            approved = "APPROVE" in response_upper and "REJECT" not in response_upper.split("APPROVE")[0]
            
            # Extract reason
            reason = "Score-based switch"
            if "REASON:" in response.upper():
                reason_start = response.upper().find("REASON:") + 7
                reason = response[reason_start:].strip()
                # Limit length
                if len(reason) > 300:
                    reason = reason[:300] + "..."
            elif approved:
                reason = "AI approved based on comprehensive analysis"
            else:
                reason = "AI determined switch not beneficial"
            
            return {"approved": approved, "reason": reason}
            
        except Exception as e:
            logger.error(f"AI pair switch decision error: {e}")
            # On error, use math fallback
            logger.info("⚠️ AI error - falling back to math decision")
            try:
                return self._math_pair_switch_decision(current_data, target_data, score_diff, current_symbol, target_symbol)
            except Exception as math_err:
                return {"approved": False, "reason": f"Both AI and math failed: {str(math_err)[:50]}"}

    def _math_pair_switch_decision(self, current_data: Optional[Dict], target_data: Dict, 
                                    score_diff: float, current_symbol: str, target_symbol: str) -> Dict[str, Any]:
        """Math-based pair switch decision when AI is unavailable.
        
        Uses multiple factors:
        1. Trend strength (ADX) - strong trend is tradeable
        2. RSI positioning - avoid extremes, favor neutral zones
        3. Volume - need liquidity for clean trades
        4. Volatility - moderate is good (2-5%), extreme is risky
        5. Momentum - should align with trend direction
        6. Stability - fewer whipsaws = better
        """
        # Extract target metrics
        t_adx = target_data.get('adx', 0)
        t_rsi = target_data.get('rsi', 50)
        t_vol = target_data.get('volume_ratio', 1)
        t_volatility = target_data.get('volatility', 0)
        t_momentum = target_data.get('momentum', 0)
        t_trend = target_data.get('trend', 'unknown')
        t_signal = target_data.get('signal_text', 'none')
        
        # Extract current metrics (with defaults)
        c_adx = current_data.get('adx', 0) if current_data else 0
        c_rsi = current_data.get('rsi', 50) if current_data else 50
        c_vol = current_data.get('volume_ratio', 1) if current_data else 1
        c_volatility = current_data.get('volatility', 0) if current_data else 0
        c_momentum = current_data.get('momentum', 0) if current_data else 0
        
        # === HARD REJECTION CRITERIA ===
        reject_reasons = []
        
        # 1. RSI extremes - don't enter overbought/oversold
        if t_rsi > 80:
            reject_reasons.append(f"RSI overbought ({t_rsi:.0f} > 80)")
        elif t_rsi < 20:
            reject_reasons.append(f"RSI oversold ({t_rsi:.0f} < 20)")
        
        # 2. No trend - ADX too weak
        if t_adx < 15:
            reject_reasons.append(f"No trend (ADX {t_adx:.0f} < 15)")
        
        # 3. Low liquidity - can't execute cleanly
        if t_vol < 0.5:
            reject_reasons.append(f"Low volume ({t_vol:.1f}x < 0.5x)")
        
        # 4. Extreme volatility - too risky
        if t_volatility > 8:
            reject_reasons.append(f"Extreme volatility ({t_volatility:.1f}% > 8%)")
        
        # If any hard rejection, reject immediately
        if reject_reasons:
            return {"approved": False, "reason": f"Math rejected: {reject_reasons[0]}"}
        
        # === COMPARATIVE HEALTH SCORING ===
        # Score both pairs on 0-100 scale
        
        def calculate_health(adx, rsi, vol, volatility, momentum, has_signal=False):
            health = 0
            
            # ADX score (0-30): Higher = stronger trend
            if adx >= 25:
                health += min(30, 15 + (adx - 25) * 0.5)  # 25+ is excellent
            elif adx >= 20:
                health += 15 + (adx - 20)  # 20-25 is good
            else:
                health += adx * 0.5  # Below 20 is weak
            
            # RSI score (0-25): Neutral zone is best
            if 40 <= rsi <= 60:
                health += 25  # Perfect entry zone
            elif 30 <= rsi <= 70:
                health += 18  # Acceptable
            elif 25 <= rsi <= 75:
                health += 12  # Caution zone
            else:
                health += 5  # Extreme zone
            
            # Volume score (0-20): Above average is good
            if vol >= 1.5:
                health += 20  # Excellent liquidity
            elif vol >= 1.0:
                health += 15  # Good liquidity
            elif vol >= 0.7:
                health += 10  # Acceptable
            else:
                health += vol * 10  # Low liquidity penalty
            
            # Volatility score (0-15): Sweet spot is 2-5%
            if 2 <= volatility <= 5:
                health += 15  # Perfect volatility
            elif 1.5 <= volatility <= 6:
                health += 12  # Good volatility
            elif volatility < 1.5:
                health += 5  # Too quiet
            else:
                health += max(0, 15 - (volatility - 5) * 2)  # Penalize high volatility
            
            # Momentum alignment (0-5)
            if abs(momentum) > 0.3:
                health += 5  # Has direction
            
            # Signal presence (0-5)
            if has_signal:
                health += 5  # Active signal is valuable
            
            return health
        
        current_health = calculate_health(c_adx, c_rsi, c_vol, c_volatility, c_momentum, False)
        target_health = calculate_health(t_adx, t_rsi, t_vol, t_volatility, t_momentum, t_signal != 'none')
        
        health_diff = target_health - current_health
        
        # === DECISION LOGIC ===
        
        # Strong improvement - approve
        if health_diff >= 15 and target_health >= 50:
            return {
                "approved": True, 
                "reason": f"Math approved: {target_symbol} significantly healthier (+{health_diff:.0f}). ADX={t_adx:.0f}, RSI={t_rsi:.0f}, Vol={t_vol:.1f}x"
            }
        
        # Moderate improvement with good absolute health - approve
        if health_diff >= 10 and target_health >= 60:
            return {
                "approved": True,
                "reason": f"Math approved: {target_symbol} better setup (+{health_diff:.0f}, health={target_health:.0f}). ADX={t_adx:.0f}, Vol={t_vol:.1f}x"
            }
        
        # Target has active signal and is reasonably healthy - approve
        if t_signal != 'none' and target_health >= 55 and health_diff >= 5:
            return {
                "approved": True,
                "reason": f"Math approved: {target_symbol} has {t_signal} signal with good health ({target_health:.0f})"
            }
        
        # Current pair is problematic and target is better
        if current_health < 40 and target_health >= 50 and health_diff >= 5:
            return {
                "approved": True,
                "reason": f"Math approved: Current {current_symbol} struggling (health={current_health:.0f}), {target_symbol} better ({target_health:.0f})"
            }
        
        # Not enough improvement - reject
        if health_diff < 10:
            return {
                "approved": False,
                "reason": f"Math: Difference too small (+{health_diff:.0f}). {current_symbol}={current_health:.0f} vs {target_symbol}={target_health:.0f}"
            }
        
        # Target not healthy enough
        if target_health < 50:
            return {
                "approved": False,
                "reason": f"Math: {target_symbol} not healthy enough ({target_health:.0f} < 50). ADX={t_adx:.0f}, Vol={t_vol:.1f}x"
            }
        
        # Default: conservative - don't switch
        return {
            "approved": False,
            "reason": f"Math: No compelling reason to switch. Current={current_health:.0f}, Target={target_health:.0f}"
        }

    def _get_market_scan_data(self) -> Dict[str, Any]:
        """Get multi-pair market data for the scanner (delegates to full scan)."""
        # Use the full scan which has indicators
        return self._get_market_scan_data_full()
    
    def _get_market_scan_data_simple(self) -> Dict[str, Any]:
        """Get basic multi-pair market data (fast, no indicators)."""
        # Use dynamic tradeable pairs
        scan_pairs = self._get_tradeable_pairs()
        
        pairs_data = []
        
        try:
            # Use synchronous ccxt for market scanning (the main exchange is async)
            import ccxt as ccxt_sync
            sync_exchange = ccxt_sync.bybit({
                "enableRateLimit": True,
                "options": {"defaultType": "swap", "defaultSubType": "linear", "recvWindow": 20000}
            })
            
            for symbol in scan_pairs:
                try:
                    # Fetch ticker data using sync exchange
                    ticker = sync_exchange.fetch_ticker(symbol)
                    
                    # Calculate volatility from high/low
                    high = ticker.get('high', 0) or 0
                    low = ticker.get('low', 0) or 0
                    price = ticker.get('last', 0) or ticker.get('close', 0) or 0
                    
                    if price > 0 and high > 0 and low > 0:
                        volatility = (high - low) / price * 100
                    else:
                        volatility = 0
                    
                    change = ticker.get('percentage', 0) or 0
                    volume = ticker.get('quoteVolume', 0) or 0
                    
                    pairs_data.append({
                        "symbol": symbol,
                        "price": price,
                        "change": change,
                        "volatility": volatility,
                        "volume": volume,
                        "high": high,
                        "low": low
                    })
                except Exception as e:
                    logger.debug(f"Error fetching {symbol}: {e}")
                    continue
            
            # Sort by volatility (highest first)
            pairs_data.sort(key=lambda x: x.get('volatility', 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Market scan error: {e}")
        
        return {
            "pairs": pairs_data,
            "current_symbol": self.SYMBOL
        }
    
    def _switch_trading_symbol(self, symbol: str) -> Dict[str, Any]:
        """Switch to a different trading symbol.
        
        UNIFIED SYMBOL SWITCH - Called by both Dashboard and Telegram.
        Ensures all components are updated consistently:
        - Bot state (self.SYMBOL)
        - AI Filter default symbol
        - Data caches (candles, price, ATR)
        - Market scan cache
        - Persisted config file
        - Triggers data refetch
        """
        try:
            # Normalize symbol format to BASEUSDT (e.g., "SOLUSDT", "BTCUSDT")
            symbol = symbol.upper().strip()
            if "/" in symbol:
                symbol = symbol.replace("/", "")  # "SOL/USDT" -> "SOLUSDT"
            if not symbol.endswith("USDT"):
                symbol = symbol + "USDT"
            
            base = symbol.replace("USDT", "")
            
            # Dynamic validation: check if symbol exists on exchange instead of hardcoded list
            try:
                ccxt_sym = f"{base}/USDT:USDT"
                if hasattr(self.exchange, 'markets') and self.exchange.markets:
                    if ccxt_sym not in self.exchange.markets:
                        return {
                            "success": False,
                            "error": f"Symbol {symbol} not found on exchange"
                        }
                elif hasattr(self, '_get_tradeable_pairs'):
                    tradeable = self._get_tradeable_pairs()
                    if symbol not in tradeable:
                        return {
                            "success": False,
                            "error": f"Symbol {symbol} not tradeable on exchange"
                        }
            except Exception as e:
                logger.warning(f"Could not validate symbol {symbol} against exchange: {e}")
            
            # Check position slots - allow switch if there's room for another position
            max_positions = self.multi_pair_config.max_total_positions if hasattr(self, 'multi_pair_config') else 2
            open_positions = len([p for p in self.positions.values() if p]) if hasattr(self, 'positions') else (1 if self.position else 0)
            
            # Check if already have a position on this symbol
            # Normalize symbol to check multiple formats
            sym_clean = symbol.replace('/', '').replace(':USDT', '').replace('USDT', '')
            ccxt_format = f"{sym_clean}/USDT:USDT"
            simple_format = f"{sym_clean}USDT"
            already_in_symbol = (symbol in self.positions or 
                                 ccxt_format in self.positions or 
                                 simple_format in self.positions)
            if already_in_symbol:
                return {
                    "success": False,
                    "error": f"Already have a position on {symbol}"
                }
            
            # If at max positions, can't add another
            if open_positions >= max_positions:
                return {
                    "success": False,
                    "error": f"Max positions ({max_positions}) reached. Close one first or wait for position replacement."
                }
            
            # Same symbol - no change needed
            if symbol == self.SYMBOL:
                return {
                    "success": True,
                    "message": f"Already trading {symbol}"
                }
            
            old_symbol = self.SYMBOL
            
            # === UPDATE ALL COMPONENTS ===
            
            # 1. Update bot state
            self.SYMBOL = symbol
            
            # 2. Update AI Filter
            if hasattr(self.ai_filter, 'default_symbol'):
                self.ai_filter.default_symbol = symbol
            
            # 3. Clear all data caches
            self.bars_1m = []
            self.bars_agg = pd.DataFrame()
            self.cached_last_price = 0.0
            self._realtime_price = 0.0  # WebSocket real-time price
            self.cached_last_ticker = {}
            self.cached_last_atr = 0.0
            if hasattr(self, 'latest_candles'):
                self.latest_candles = {}
            if hasattr(self, 'price_cache') and self.price_cache:
                self.price_cache.clear()
            
            # 4. Invalidate AI market cache
            type(self)._ai_market_cache = {
                "recommendation": None,
                "best_pair": None,
                "timestamp": None,
                "top_pair_at_analysis": None
            }
            
            # 5. Invalidate full scan cache
            type(self)._full_scan_cache = {"data": None, "timestamp": 0}
            
            # 6. Signal main loop to refetch data
            self._pending_data_refetch = True
            
            # 7. Update switch cooldown timestamp (prevent excessive switching)
            import time
            type(self)._last_switch_time = time.time()
            
            # 8. Add new symbol to WebSocket for real-time price streaming
            if hasattr(self, 'ws_price_stream') and self.ws_price_stream:
                self.ws_price_stream.add_symbol(symbol)
            
            # 9. Save to config for persistence across restarts
            self._save_persisted_symbol()
            
            logger.info(f"🔄 SYMBOL SWITCH: {old_symbol} → {symbol} (all components updated)")
            
            # 9. Queue Telegram notification for manual/API switches
            # Note: Auto-switch sends its own detailed notification, so we skip if caller is auto-switch
            if not getattr(self, '_auto_switch_in_progress', False):
                self._pending_switch_notification = f"📍 Symbol switched: {old_symbol} → {symbol}"
            
            return {
                "success": True,
                "message": f"Switched from {old_symbol} to {symbol}. Fetching new data..."
            }
            
        except Exception as e:
            logger.error(f"Symbol switch error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # Cache for AI market analysis to prevent flip-flopping recommendations
    _ai_market_cache = {
        "recommendation": None,
        "best_pair": None,
        "timestamp": None,
        "top_pair_at_analysis": None
    }
    AI_MARKET_CACHE_DURATION = 300  # 5 minutes
    
    def _get_market_scanner_context_for_ai(self) -> Dict[str, Any]:
        """Get market scanner context for AI Signal Filter alignment.
        
        This ensures the Signal Filter AI knows what the Market Scanner AI recommended,
        so they can make aligned decisions.
        """
        try:
            cache = type(self)._ai_market_cache
            
            # Get best pair from cache
            best_pair = cache.get('best_pair', '')
            current_pair = self.SYMBOL.replace('/', '')
            
            # Find current pair's rank in recent scan
            current_rank = 0
            try:
                scan_data = self._get_market_scan_data()
                pairs = scan_data.get('pairs', [])
                for i, p in enumerate(pairs):
                    if p['symbol'] == current_pair or p['symbol'] == self.SYMBOL:
                        current_rank = i + 1
                        break
            except Exception:
                current_rank = 0  # Default rank if scan fails
            
            is_recommended = (best_pair == current_pair) or (best_pair == self.SYMBOL) or not best_pair
            
            return {
                'best_pair': best_pair or current_pair,
                'current_pair_rank': current_rank,
                'is_recommended_pair': is_recommended,
                'cache_age': int(time.time() - cache.get('timestamp', 0)) if cache.get('timestamp') else 999
            }
        except Exception as e:
            logger.debug(f"Market scanner context error: {e}")
            return {}
    
    def _ai_analyze_all_markets(self) -> Dict[str, Any]:
        """Use AI to analyze all market pairs and recommend the best one.
        
        Caches the recommendation for 5 minutes to prevent flip-flopping.
        Only refreshes early if the top volatile pair changes significantly.
        """
        try:
            # Get market scan data
            scan_data = self._get_market_scan_data()
            pairs = scan_data.get('pairs', [])
            
            if not pairs:
                return {"recommendation": "No market data available.", "best_pair": ""}
            
            # Check if we have a valid cached recommendation
            now = time.time()
            cache = self._ai_market_cache
            
            if cache["recommendation"] and cache["timestamp"]:
                cache_age = now - cache["timestamp"]
                
                # Check if cache is still valid (under 5 minutes)
                if cache_age < self.AI_MARKET_CACHE_DURATION:
                    # Check if market conditions changed significantly
                    current_top = pairs[0]["symbol"] if pairs else None
                    cached_top = cache.get("top_pair_at_analysis")
                    
                    # Only invalidate if top pair changed
                    if current_top == cached_top:
                        remaining = int(self.AI_MARKET_CACHE_DURATION - cache_age)
                        logger.debug(f"Using cached AI recommendation (expires in {remaining}s)")
                        return {
                            "recommendation": cache["recommendation"],
                            "best_pair": cache["best_pair"],
                            "cached": True,
                            "cache_expires_in": remaining
                        }
                    else:
                        logger.info(f"Market leader changed: {cached_top} → {current_top}, refreshing AI analysis")
            
            # Build analysis prompt
            pairs = scan_data.get('pairs', [])
            
            if not pairs:
                return {"recommendation": "No market data available.", "best_pair": ""}
            
            # Build analysis prompt with rich indicator data
            pairs_summary = []
            for p in pairs[:10]:  # Top 10 pairs
                signal_txt = ""
                if p.get('signal') == 1:
                    signal_txt = " [LONG SIGNAL!]"
                elif p.get('signal') == -1:
                    signal_txt = " [SHORT SIGNAL!]"
                
                pairs_summary.append(
                    f"- {p['symbol']}: ${p['price']:,.2f} | {p['change']:+.2f}% | "
                    f"Score: {p.get('score', 0):.0f}/100 | RSI: {p.get('rsi', 50):.0f} | "
                    f"ADX: {p.get('adx', 0):.0f} | Trend: {p.get('trend', 'n/a')} | "
                    f"Vol: {p.get('volume_ratio', 1):.1f}x{signal_txt}"
                )
            
            current_balance = self.balance
            risk_pct = self.RISK_PCT * 100
            current_symbol = self.SYMBOL
            
            # Find current symbol's score
            current_score = 0
            for p in pairs:
                if p['symbol'] == current_symbol:
                    current_score = p.get('score', 0)
                    break
            
            # Get multi-position info for replacement logic
            position_count = len([p for p in self.positions.values() if p]) if hasattr(self, 'positions') else (1 if self.position else 0)
            max_positions = self.multi_pair_config.max_total_positions if hasattr(self, 'multi_pair_config') else 2
            
            # Build position status with PhD math health analysis
            position_details = []
            position_health_analysis = []
            
            # Helper function to get position health
            def get_position_health(pos, sym, clean_sym, current_price, pnl_pct):
                """Get PhD math verdict for a position using available data."""
                health_verdict = "UNKNOWN"
                hold_score = 0
                exit_score = 0
                
                try:
                    if self.ai_filter and hasattr(self, 'bars_agg') and self.bars_agg is not None and len(self.bars_agg) >= 20:
                        # Use bars_agg for main symbol, estimate ATR
                        atr_val = getattr(self, 'current_atr', current_price * 0.02)  # Default 2%
                        if atr_val <= 0:
                            atr_val = current_price * 0.02
                        
                        health_result = self.ai_filter.monitor_position(
                            df=self.bars_agg,
                            position_side=pos.side.upper(),
                            entry_price=pos.entry_price,
                            current_price=current_price,
                            atr=atr_val,
                            symbol=clean_sym,
                            unrealized_pnl_pct=pnl_pct,
                            tp1_hit=getattr(pos, 'tp1_hit', False),
                            tp2_hit=getattr(pos, 'tp2_hit', False),
                            position_size=getattr(pos, 'size', 0)
                        )
                        if health_result:
                            # Extract verdict from action
                            action = health_result.get('action', 'hold')
                            if action == 'hold':
                                health_verdict = "HOLD"
                            elif action == 'close':
                                health_verdict = "EXIT"
                            elif action == 'partial_close':
                                health_verdict = "TAKE_PROFIT"
                            
                            # Extract scores from math_analysis
                            math_analysis = health_result.get('math_analysis', {})
                            hold_score = math_analysis.get('hold_score', 50)
                            exit_score = math_analysis.get('exit_score', 50)
                except Exception as e:
                    logger.debug(f"Could not get position health for {sym}: {e}")
                
                return health_verdict, hold_score, exit_score
            
            if hasattr(self, 'positions') and self.positions:
                for sym, pos in self.positions.items():
                    if pos:
                        # Get current price for PnL calculation
                        clean_sym = sym.replace('/USDT:USDT', 'USDT').replace('/', '')
                        # Use _symbol_prices cache (populated by position monitor)
                        current_price = pos.entry_price  # Default fallback
                        if hasattr(self, '_symbol_prices') and clean_sym in self._symbol_prices:
                            current_price = self._symbol_prices[clean_sym]
                        
                        # Calculate PnL
                        if pos.side == 'long':
                            pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
                        else:
                            pnl_pct = ((pos.entry_price - current_price) / pos.entry_price) * 100
                        pnl_usd = pnl_pct / 100 * pos.size * pos.entry_price
                        
                        # Get PhD math analysis for this position
                        health_verdict, hold_score, exit_score = get_position_health(pos, sym, clean_sym, current_price, pnl_pct)
                        
                        position_details.append(f"{clean_sym} ({pos.side.upper()}) PnL: {pnl_pct:+.1f}% (${pnl_usd:+.2f})")
                        
                        # Add detailed health for prompt
                        position_health_analysis.append(
                            f"  • {clean_sym}: {pos.side.upper()} | Entry: ${pos.entry_price:.4f} → ${current_price:.4f} | "
                            f"PnL: {pnl_pct:+.1f}% (${pnl_usd:+.2f}) | "
                            f"Math: {health_verdict} (Hold:{hold_score:.0f} vs Exit:{exit_score:.0f})"
                        )
            elif self.position:
                pos = self.position
                clean_sym = self.SYMBOL.replace('/USDT:USDT', 'USDT').replace('/', '')
                # Use _symbol_prices cache or current_price from main loop
                current_price = pos.entry_price  # Default fallback
                if hasattr(self, '_symbol_prices') and clean_sym in self._symbol_prices:
                    current_price = self._symbol_prices[clean_sym]
                elif hasattr(self, 'current_price') and self.current_price:
                    current_price = self.current_price
                
                if pos.side == 'long':
                    pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
                else:
                    pnl_pct = ((pos.entry_price - current_price) / pos.entry_price) * 100
                pnl_usd = pnl_pct / 100 * pos.size * pos.entry_price
                
                # Use helper function for PhD math analysis
                health_verdict, hold_score, exit_score = get_position_health(pos, self.SYMBOL, clean_sym, current_price, pnl_pct)
                
                position_details.append(f"{clean_sym} ({pos.side.upper()}) PnL: {pnl_pct:+.1f}% (${pnl_usd:+.2f})")
                position_health_analysis.append(
                    f"  • {clean_sym}: {pos.side.upper()} | Entry: ${pos.entry_price:.4f} → ${current_price:.4f} | "
                    f"PnL: {pnl_pct:+.1f}% (${pnl_usd:+.2f}) | "
                    f"Math: {health_verdict} (Hold:{hold_score:.0f} vs Exit:{exit_score:.0f})"
                )
            
            positions_text = ", ".join(position_details) if position_details else "No positions"
            health_text = "\n".join(position_health_analysis) if position_health_analysis else "No open positions"
            
            # Calculate actual bot thresholds for second position
            existing_sides = set()
            if hasattr(self, 'positions') and self.positions:
                for sym, pos in self.positions.items():
                    if pos:
                        existing_sides.add(pos.side.lower())
            elif self.position:
                existing_sides.add(self.position.side.lower())
            
            # Base threshold + same-direction penalty
            base_threshold = 53  # Default base
            same_direction_penalty = 10
            has_long = 'long' in existing_sides
            has_short = 'short' in existing_sides
            
            # Calculate thresholds
            long_threshold = base_threshold + (same_direction_penalty if has_long else 0)
            short_threshold = base_threshold + (same_direction_penalty if has_short else 0)
            
            threshold_text = ""
            if position_count > 0 and position_count < max_positions:
                if has_long and not has_short:
                    threshold_text = f"""
**BOT ENTRY THRESHOLDS (for 2nd position):**
- LONG entry requires: Score ≥{long_threshold} (same-direction penalty +{same_direction_penalty})
- SHORT entry requires: Score ≥{base_threshold} (opposite direction - no penalty)
NOTE: Bot will NOT open a position below these thresholds regardless of AI recommendation."""
                elif has_short and not has_long:
                    threshold_text = f"""
**BOT ENTRY THRESHOLDS (for 2nd position):**
- LONG entry requires: Score ≥{base_threshold} (opposite direction - no penalty)
- SHORT entry requires: Score ≥{short_threshold} (same-direction penalty +{same_direction_penalty})
NOTE: Bot will NOT open a position below these thresholds regardless of AI recommendation."""
                else:
                    threshold_text = f"""
**BOT ENTRY THRESHOLDS (for 2nd position):**
- Any entry requires: Score ≥{base_threshold}
NOTE: Bot will NOT open a position below this threshold regardless of AI recommendation."""
            elif position_count == 0:
                threshold_text = f"""
**BOT ENTRY THRESHOLDS:**
- Entry requires: Score ≥{base_threshold} OR valid signal
NOTE: Bot will NOT open a position below threshold without a signal."""

            prompt = f"""You are Julaba's AI trading advisor. Analyze these market pairs and recommend action.

**Current Trading Setup:**
- Primary symbol: {current_symbol} (Score: {current_score:.0f}/100)
- Account Balance: ${current_balance:,.2f}
- Risk per trade: {risk_pct:.1f}%
- Positions: {position_count}/{max_positions} | {positions_text}
- Auto-switch enabled: {'Yes' if type(self).AUTO_SWITCH_ENABLED else 'No'}

**OPEN POSITION HEALTH (PhD Math Analysis):**
{health_text}
NOTE: PhD Math verdict is what the bot WILL follow. If PhD says HOLD, position will be held.
{threshold_text}

**Market Pairs (sorted by tradability score):**
{chr(10).join(pairs_summary)}

**UNIFIED SCORING CRITERIA:**
Score thresholds:
- Score ≥75 = STRONG setup, high probability trade
- Score ≥60 = GOOD setup, worth taking
- Score 50-59 = MARGINAL, needs other confirmation
- Score <50 = WEAK, avoid trading

Indicator interpretation:
- ADX ≥25 = trending, ADX ≥35 = strong trend
- ADX 35-40 = DANGER ZONE (historically 22% win rate - AVOID!)
- RSI 40-60 = ideal entry zone
- Volume ≥1.5x = strong confirmation

**Your Task:**
1. If slot available: Check if best pair meets BOT ENTRY THRESHOLD
   - If YES: recommend opening position
   - If NO: explain why bot won't open (score vs threshold gap)
2. For open positions: ALIGN with PhD Math verdict
3. Be HONEST about what bot will actually do vs what you'd ideally recommend
4. Rate confidence: HIGH/MEDIUM/LOW

Keep response under 150 words. Be direct and actionable."""

            recommendation = self.ai_filter._generate_content(prompt)
            
            if recommendation:
                # Try to extract the recommended symbol
                best_pair = ""
                for p in pairs:
                    if p['symbol'].replace('USDT', '') in recommendation.upper():
                        best_pair = p['symbol']
                        break
                
                # Cache the result
                type(self)._ai_market_cache = {
                    "recommendation": recommendation,
                    "best_pair": best_pair,
                    "timestamp": time.time(),
                    "top_pair_at_analysis": pairs[0]['symbol'] if pairs else None
                }
                
                return {
                    "recommendation": recommendation,
                    "best_pair": best_pair,
                    "cached": False,
                    "cache_expires_in": type(self).AI_MARKET_CACHE_DURATION
                }
            else:
                return {
                    "recommendation": "AI analysis temporarily unavailable.",
                    "best_pair": "",
                    "cached": False
                }
                
        except Exception as e:
            logger.error(f"AI market analysis error: {e}")
            return {
                "recommendation": f"Error analyzing markets: {str(e)}",
                "best_pair": ""
            }
    
    def _get_system_params(self) -> Dict[str, Any]:
        """Get all configurable system parameters."""
        return {
            "risk_pct": self.RISK_PCT,
            "atr_mult": self.ATR_MULT,
            "tp1_r": self.TP1_R,
            "tp2_r": self.TP2_R,
            "tp3_r": self.TP3_R,
            "tp1_pct": self.TP1_PCT,
            "tp2_pct": self.TP2_PCT,
            "tp3_pct": self.TP3_PCT,
            "trail_trigger_r": self.TRAIL_TRIGGER_R,
            "trail_offset_r": self.TRAIL_OFFSET_R,
            "ai_mode": self.ai_mode,
            "ai_confidence": self.ai_filter.confidence_threshold,
            "ai_scan_interval": self.ai_scan_interval,
            "ai_scan_notify_opportunities_only": self.ai_scan_notify_opportunities_only,
            "ai_scan_quiet_interval": self.ai_scan_quiet_interval,
            "summary_interval": self.summary_interval,
            "symbol": self.SYMBOL,
            "paused": self.paused,
            "daily_loss_limit": self.daily_loss_limit,
            "daily_loss_triggered": self.daily_loss_triggered,
            "dry_run_mode": self.dry_run_mode,
            "auto_switch": type(self).AUTO_SWITCH_ENABLED,
            "auto_switch_interval": type(self).AUTO_SWITCH_INTERVAL,
            "auto_switch_min_diff": type(self).AUTO_SWITCH_MIN_SCORE_DIFF,
            # Multi-pair mode
            "multi_pair_enabled": len(self.additional_symbols) > 0,
            "multi_pair_count": 1 + len(self.additional_symbols),
            "active_pairs": [self.SYMBOL] + self.additional_symbols,
            # ML settings
            "ml_learn_all_trades": self.ml_learn_all_trades,
            # Proactive scan settings
            "proactive_threshold": self.ai_filter.proactive_threshold if self.ai_filter else 65,
        }
    
    def _set_system_param(self, param: str, value: Any) -> Dict[str, Any]:
        """Set a system parameter. Returns success status and message."""
        param = param.lower().replace(" ", "_")
        
        try:
            if param == "risk_pct":
                val = float(value)
                if 0.001 <= val <= 0.1:  # 0.1% to 10%
                    self.RISK_PCT = val
                    return {"success": True, "message": f"Risk set to {val*100:.1f}%"}
                return {"success": False, "message": "Risk must be between 0.1% and 10%"}
            
            elif param == "atr_mult":
                val = float(value)
                if 0.5 <= val <= 5.0:
                    self.ATR_MULT = val
                    return {"success": True, "message": f"ATR multiplier set to {val}"}
                return {"success": False, "message": "ATR mult must be between 0.5 and 5.0"}
            
            elif param == "ai_confidence":
                val = float(value)
                if 0.1 <= val <= 1.0:
                    self.ai_filter.confidence_threshold = val
                    return {"success": True, "message": f"AI confidence threshold set to {val*100:.0f}%"}
                return {"success": False, "message": "Confidence must be between 10% and 100%"}
            
            elif param == "ai_mode":
                if value in ["filter", "advisory", "autonomous", "hybrid"]:
                    self.ai_mode = value
                    return {"success": True, "message": f"AI mode set to {value}"}
                return {"success": False, "message": "Mode must be: filter, advisory, autonomous, or hybrid"}
            
            elif param in ["tp1_r", "tp2_r", "tp3_r"]:
                val = float(value)
                if 0.5 <= val <= 10.0:
                    setattr(self, param.upper(), val)
                    return {"success": True, "message": f"{param.upper()} set to {val}R"}
                return {"success": False, "message": "TP must be between 0.5R and 10R"}
            
            elif param == "ai_scan_interval":
                val = int(value)
                if 30 <= val <= 3600:
                    self.ai_scan_interval = val
                    return {"success": True, "message": f"AI scan interval set to {val}s"}
                return {"success": False, "message": "Interval must be 30-3600 seconds"}
            
            elif param == "ai_scan_notify_opportunities_only":
                self.ai_scan_notify_opportunities_only = str(value).lower() in ["true", "1", "yes"]
                return {"success": True, "message": f"Scan notifications: {'opportunities only' if self.ai_scan_notify_opportunities_only else 'all scans'}"}
            
            elif param == "ai_scan_quiet_interval":
                val = int(value)
                if 300 <= val <= 7200:
                    self.ai_scan_quiet_interval = val
                    return {"success": True, "message": f"Quiet notification interval set to {val}s ({val//60} min)"}
                return {"success": False, "message": "Quiet interval must be 300-7200 seconds (5-120 min)"}
            
            elif param == "ai_scan_telegram":
                self.ai_scan_telegram_enabled = str(value).lower() in ["true", "1", "yes", "on"]
                return {"success": True, "message": f"AI scan telegram notifications: {'enabled' if self.ai_scan_telegram_enabled else 'disabled'}"}
            
            elif param == "ai_position_telegram":
                self.ai_position_monitor_telegram_enabled = str(value).lower() in ["true", "1", "yes", "on"]
                return {"success": True, "message": f"Position monitor telegram notifications: {'enabled' if self.ai_position_monitor_telegram_enabled else 'disabled'}"}
            
            elif param == "paused":
                self.paused = str(value).lower() in ["true", "1", "yes"]
                return {"success": True, "message": f"Bot {'paused' if self.paused else 'resumed'}"}
            
            elif param == "daily_loss_limit":
                val = float(value)
                if 0.01 <= val <= 0.20:  # 1% to 20%
                    self.daily_loss_limit = val
                    return {"success": True, "message": f"Daily loss limit set to {val*100:.1f}%"}
                return {"success": False, "message": "Daily loss limit must be between 1% and 20%"}
            
            elif param == "dry_run" or param == "dry_run_mode":
                self.dry_run_mode = str(value).lower() in ["true", "1", "yes"]
                return {"success": True, "message": f"Dry-run mode {'enabled' if self.dry_run_mode else 'disabled'}"}
            
            elif param == "ml_learn_all" or param == "ml_learn_all_trades":
                self.ml_learn_all_trades = str(value).lower() in ["true", "1", "yes", "on", "all"]
                if self.ml_learn_all_trades:
                    return {"success": True, "message": "🧠 ML learning: ALL trades (manual + autonomous)"}
                else:
                    return {"success": True, "message": "🧠 ML learning: AUTONOMOUS trades only (manual excluded)"}
            
            elif param == "reset_daily_loss":
                self.daily_loss_triggered = False
                # Set override until end of today (midnight UTC)
                today = datetime.now(timezone.utc).date()
                self.daily_loss_override_until = today
                logger.info(f"🔓 Daily loss override active until midnight UTC ({today})")
                return {"success": True, "message": "Daily loss circuit breaker reset (override until midnight)"}
            
            elif param == "reset_drawdown" or param == "reset_max_drawdown":
                # Reset max drawdown circuit breaker and peak balance
                old_peak = self.peak_balance
                self.max_drawdown_triggered = False
                self.peak_balance = self.balance  # Reset peak to current balance
                # Save to config
                self._save_trading_state()
                logger.info(f"🔓 MAX DRAWDOWN RESET: peak_balance reset from ${old_peak:.2f} to ${self.balance:.2f}")
                return {"success": True, "message": f"🔓 Max drawdown reset! Peak balance: ${old_peak:.2f} → ${self.balance:.2f}"}
            
            elif param == "force_resume" or param == "override":
                # Force clear ALL halt conditions (including drawdown)
                self.paused = False
                self.daily_loss_triggered = False
                self.max_drawdown_triggered = False  # Also clear drawdown
                # Reset peak to current if drawdown was the issue
                if self.peak_balance > self.balance * 1.2:  # Peak was >20% higher
                    old_peak = self.peak_balance
                    self.peak_balance = self.balance
                    self._save_trading_state()
                    logger.info(f"🔓 FORCE RESUME: Peak balance reset ${old_peak:.2f} → ${self.balance:.2f}")
                # Reset risk manager cooldowns if available
                if hasattr(self, 'risk_manager') and self.risk_manager:
                    self.risk_manager.cooldown_until = None
                    self.risk_manager.consecutive_losses = 0
                logger.info("🔓 FORCE RESUME: All trading halts cleared")
                return {"success": True, "message": "🔓 All trading halts cleared! Bot resumed."}
            
            elif param == "symbol":
                # Use unified symbol switch method
                return self._switch_trading_symbol(str(value))
            
            elif param == "auto_switch":
                type(self).AUTO_SWITCH_ENABLED = str(value).lower() in ["true", "1", "yes", "on"]
                status = "enabled" if type(self).AUTO_SWITCH_ENABLED else "disabled"
                return {"success": True, "message": f"🔄 Autonomous pair switching {status}"}
            
            elif param == "auto_switch_interval":
                val = int(value)
                if 60 <= val <= 3600:
                    type(self).AUTO_SWITCH_INTERVAL = val
                    return {"success": True, "message": f"Auto-switch check interval set to {val}s ({val//60} min)"}
                return {"success": False, "message": "Interval must be 60-3600 seconds"}
            
            elif param == "auto_switch_min_diff":
                val = int(value)
                if 5 <= val <= 50:
                    type(self).AUTO_SWITCH_MIN_SCORE_DIFF = val
                    return {"success": True, "message": f"Auto-switch minimum score difference set to {val} points"}
                return {"success": False, "message": "Min difference must be 5-50 points"}
            
            elif param == "proactive_threshold":
                val = int(value)
                if 50 <= val <= 85:
                    if self.ai_filter:
                        self.ai_filter.proactive_threshold = val
                    return {"success": True, "message": f"🎯 Proactive scan threshold set to {val} (lower = more aggressive)"}
                return {"success": False, "message": "Threshold must be 50-85"}
            
            elif param == "multi_pair_enabled":
                enabled = str(value).lower() in ["true", "1", "yes", "on"]
                if not enabled:
                    # Disable multi-pair mode by clearing additional symbols
                    self.additional_symbols = []
                    return {"success": True, "message": "🔄 Multi-pair mode disabled. Now trading only primary pair."}
                else:
                    return {"success": True, "message": "ℹ️ Multi-pair is enabled when you have additional pairs. Use 'add_pair' to add pairs."}
            
            elif param == "add_pair":
                # Add a pair to multi-pair list
                new_pair = str(value).upper().replace("/", "")
                if not new_pair.endswith("USDT"):
                    new_pair = new_pair + "USDT"
                
                base = new_pair.replace("USDT", "")
                
                # Dynamic validation against exchange
                try:
                    ccxt_sym = f"{base}/USDT:USDT"
                    if hasattr(self.exchange, 'markets') and self.exchange.markets and ccxt_sym not in self.exchange.markets:
                        return {"success": False, "message": f"Symbol {new_pair} not found on exchange"}
                except Exception:
                    pass  # Allow if can't validate
                
                if new_pair == self.SYMBOL:
                    return {"success": False, "message": f"{new_pair} is already the primary pair."}
                
                if new_pair in self.additional_symbols:
                    return {"success": False, "message": f"{new_pair} is already in the multi-pair list."}
                
                self.additional_symbols.append(new_pair)
                all_pairs = [self.SYMBOL] + self.additional_symbols
                return {"success": True, "message": f"✅ Added {new_pair} to multi-pair mode. Active pairs: {', '.join(all_pairs)}"}
            
            elif param == "remove_pair":
                # Remove a pair from multi-pair list
                rm_pair = str(value).upper().replace("/", "")
                if not rm_pair.endswith("USDT"):
                    rm_pair = rm_pair + "USDT"
                
                if rm_pair == self.SYMBOL:
                    return {"success": False, "message": f"Cannot remove the primary pair. Use 'symbol' to change the primary pair instead."}
                
                if rm_pair not in self.additional_symbols:
                    return {"success": False, "message": f"{rm_pair} is not in the multi-pair list."}
                
                self.additional_symbols.remove(rm_pair)
                all_pairs = [self.SYMBOL] + self.additional_symbols
                if self.additional_symbols:
                    return {"success": True, "message": f"❌ Removed {rm_pair}. Active pairs: {', '.join(all_pairs)}"}
                else:
                    return {"success": True, "message": f"❌ Removed {rm_pair}. Multi-pair mode disabled (only primary pair active)."}
            
            elif param == "toggle_hold":
                # Toggle hold state for a position
                if isinstance(value, dict):
                    symbol = value.get('symbol', '')
                    hold_state = value.get('hold', True)
                    return self._dashboard_toggle_hold(symbol, hold_state)
                return {"success": False, "message": "Invalid toggle_hold value - expected {symbol, hold}"}
            
            else:
                return {"success": False, "message": f"Unknown parameter: {param}"}
                
        except (ValueError, TypeError) as e:
            return {"success": False, "message": f"Invalid value: {e}"}
    
    def _get_pipeline_status(self) -> Dict[str, Any]:
        """Get real-time pipeline status for monitoring visualization.
        
        Returns component health, data flow status, errors, and timing.
        Like a ship's engine room monitoring system.
        """
        import time
        now = datetime.now(timezone.utc)
        
        # Track component status
        components = {}
        data_flow = []
        errors = []
        warnings = []
        
        # === ENGINE STATUS ===
        try:
            engine_status = 'ok' if self.engine_running else 'stopped'
            cycle_age = None
            if self.last_cycle_time:
                cycle_age = (now - self.last_cycle_time).total_seconds()
                if cycle_age > 30:  # More than 30s since last cycle - warning
                    engine_status = 'warning'
                if cycle_age > 120:  # More than 2 mins - error
                    engine_status = 'error'
            
            components['engine'] = {
                'name': 'Trading Engine',
                'icon': '⚙️',
                'status': engine_status,
                'running': self.engine_running,
                'cycle_count': self.cycle_count,
                'last_cycle_seconds_ago': cycle_age,
                'last_cycle_duration_ms': self.last_cycle_duration_ms,
                'paused': self.paused,
                'uptime_seconds': (now - self.start_time).total_seconds() if self.start_time else 0
            }
            
            if not self.engine_running:
                errors.append("Engine not running")
            elif self.paused:
                warnings.append("Trading paused")
            elif cycle_age and cycle_age > 30:
                warnings.append(f"Engine stalled ({cycle_age:.0f}s since last cycle)")
        except Exception as e:
            components['engine'] = {'status': 'error', 'error': str(e)}
            errors.append(f"Engine status error: {e}")
        
        # === MARKET DATA FEED ===
        try:
            has_data = self.bars_agg is not None and len(self.bars_agg) > 0
            last_bar_time = None
            data_age_seconds = None
            
            if has_data:
                last_bar = self.bars_agg.iloc[-1]
                if 'timestamp' in self.bars_agg.columns:
                    last_bar_time = pd.to_datetime(last_bar['timestamp'])
                    data_age_seconds = (now - last_bar_time.replace(tzinfo=timezone.utc)).total_seconds()
            
            # With 15m candles, data can be up to 900s old between cycles
            # Warning at 5 mins, error at 20 mins (missed cycle)
            data_status = 'ok'
            if data_age_seconds is not None:
                if data_age_seconds > 1200:  # 20 mins - missed cycle
                    data_status = 'error'
                elif data_age_seconds > 300:  # 5 mins
                    data_status = 'warning'
            elif not has_data:
                data_status = 'error'
            
            components['market_feed'] = {
                'name': 'Market Data Feed',
                'icon': '📊',
                'status': data_status,
                'bars_loaded': len(self.bars_agg) if has_data else 0,
                'last_price': self._last_price,
                'data_age_seconds': data_age_seconds,
                'symbol': self.SYMBOL
            }
            
            if data_age_seconds and data_age_seconds > 300:
                warnings.append(f"Market data is {data_age_seconds:.0f}s old")
        except Exception as e:
            components['market_feed'] = {'status': 'error', 'error': str(e)}
            errors.append(f"Market feed error: {e}")
        
        # === TECHNICAL INDICATORS ===
        try:
            indicators = self._get_indicators_for_dashboard()
            has_indicators = indicators and indicators.get('rsi') is not None
            
            components['indicators'] = {
                'name': 'Technical Indicators',
                'icon': '📈',
                'status': 'ok' if has_indicators else 'warning',
                'rsi': indicators.get('rsi') if indicators else None,
                'adx': indicators.get('adx') if indicators else None,
                'atr': indicators.get('atr') if indicators else None,
                'last_calc': now.isoformat()
            }
        except Exception as e:
            components['indicators'] = {'status': 'error', 'error': str(e)}
            errors.append(f"Indicators error: {e}")
        
        # === MARKET REGIME ===
        try:
            regime = self._get_regime()
            components['regime'] = {
                'name': 'Market Regime',
                'icon': '🌊',
                'status': 'ok' if regime else 'warning',
                'current': regime.get('regime', 'unknown') if regime else 'unknown',
                'tradeable': regime.get('tradeable', False) if regime else False,
                'hurst': regime.get('hurst') if regime else None
            }
        except Exception as e:
            components['regime'] = {'status': 'error', 'error': str(e)}
            errors.append(f"Regime error: {e}")
        
        # === AI FILTER ===
        try:
            ai_available = False
            ai_cooldown = False
            ai_stats = {}
            
            if self.ai_filter:
                if hasattr(self.ai_filter, '_is_ai_available'):
                    ai_available = self.ai_filter._is_ai_available()
                if hasattr(self.ai_filter, 'get_ai_status'):
                    ai_stats = self.ai_filter.get_ai_status()
                    ai_cooldown = ai_stats.get('in_cooldown', False)
            
            components['ai_filter'] = {
                'name': 'AI Filter',
                'icon': '🤖',
                'status': 'ok' if ai_available else 'cooldown' if ai_cooldown else 'warning',
                'available': ai_available,
                'in_cooldown': ai_cooldown,
                'cooldown_remaining': ai_stats.get('cooldown_remaining_seconds', 0),
                'call_count': ai_stats.get('call_count', 0),
                'mode': self.ai_mode
            }
            
            if ai_cooldown:
                warnings.append(f"AI in cooldown ({ai_stats.get('cooldown_remaining_seconds', 0):.0f}s remaining)")
        except Exception as e:
            components['ai_filter'] = {'status': 'error', 'error': str(e)}
            errors.append(f"AI filter error: {e}")
        
        # === RISK MANAGER ===
        try:
            risk_stats = self._get_risk_stats()
            can_trade = risk_stats.get('can_trade', False) if risk_stats else False
            
            components['risk_manager'] = {
                'name': 'Risk Manager',
                'icon': '🛡️',
                'status': 'ok' if can_trade else 'blocked',
                'can_trade': can_trade,
                'mode': risk_stats.get('mode', 'unknown') if risk_stats else 'unknown',
                'daily_pnl': risk_stats.get('daily_pnl') if risk_stats else None,
                'daily_limit_hit': risk_stats.get('daily_limit_hit', False) if risk_stats else False
            }
            
            if risk_stats and risk_stats.get('daily_limit_hit'):
                warnings.append("Daily loss limit reached")
        except Exception as e:
            components['risk_manager'] = {'status': 'error', 'error': str(e)}
            errors.append(f"Risk manager error: {e}")
        
        # === POSITION MONITOR ===
        try:
            has_position = self.position is not None
            monitor_timing = None
            
            if has_position and self._last_position_monitor:
                monitor_age = (now - self._last_position_monitor).total_seconds()
                monitor_timing = monitor_age
            
            components['position_monitor'] = {
                'name': 'Position Monitor',
                'icon': '👁️',
                'status': 'active' if has_position else 'idle',
                'has_position': has_position,
                'position_side': self.position.side if has_position else None,
                'last_check_seconds_ago': monitor_timing,
                'sl_protection_active': has_position
            }
        except Exception as e:
            components['position_monitor'] = {'status': 'error', 'error': str(e)}
            errors.append(f"Position monitor error: {e}")
        
        # === TELEGRAM BOT ===
        try:
            if self.telegram:
                tg_status = self.telegram.get_telegram_status()
                status = tg_status.get('status', 'unknown')
                
                components['telegram'] = {
                    'name': 'Telegram Bot',
                    'icon': '📱',
                    'status': status,
                    'enabled': tg_status.get('enabled', False),
                    'is_started': tg_status.get('is_started', False),
                    'bot_initialized': tg_status.get('bot_initialized', False),
                    'connected': status in ['ok', 'warning'],
                    'last_error': tg_status.get('last_error'),
                    'error_count': tg_status.get('error_count', 0),
                    'seconds_since_error': tg_status.get('seconds_since_last_error'),
                    'seconds_since_success': tg_status.get('seconds_since_last_success')
                }
                
                # Add warnings/errors for recent failures
                if status == 'warning' and tg_status.get('last_error'):
                    warnings.append(f"Telegram: {tg_status['last_error'][:50]}")
                elif status == 'error':
                    errors.append("Telegram: Bot not initialized")
            else:
                components['telegram'] = {
                    'name': 'Telegram Bot',
                    'icon': '📱',
                    'status': 'disabled',
                    'connected': False,
                    'enabled': False
                }
        except Exception as e:
            components['telegram'] = {'status': 'error', 'error': str(e)}
            errors.append(f"Telegram error: {e}")
        
        # === EXCHANGE CONNECTION ===
        try:
            exchange_ok = self.exchange is not None
            
            components['exchange'] = {
                'name': 'Exchange (Bybit Futures)',
                'icon': '💱',
                'status': 'ok' if exchange_ok else 'error',
                'connected': exchange_ok,
                'paper_trading': self.paper_mode,
                'dry_run': self.dry_run_mode
            }
        except Exception as e:
            components['exchange'] = {'status': 'error', 'error': str(e)}
            errors.append(f"Exchange error: {e}")
        
        # === DATA FLOW TRACKING ===
        # Track recent data flow events (you could enhance this with actual event tracking)
        data_flow = [
            {'from': 'market_feed', 'to': 'indicators', 'active': components.get('market_feed', {}).get('status') == 'ok'},
            {'from': 'indicators', 'to': 'regime', 'active': components.get('indicators', {}).get('status') == 'ok'},
            {'from': 'indicators', 'to': 'ai_filter', 'active': components.get('indicators', {}).get('status') == 'ok'},
            {'from': 'regime', 'to': 'ai_filter', 'active': components.get('regime', {}).get('status') == 'ok'},
            {'from': 'ai_filter', 'to': 'risk_manager', 'active': components.get('ai_filter', {}).get('status') in ['ok', 'cooldown']},
            {'from': 'risk_manager', 'to': 'position_monitor', 'active': components.get('risk_manager', {}).get('status') in ['ok', 'blocked']},
            {'from': 'position_monitor', 'to': 'exchange', 'active': components.get('position_monitor', {}).get('status') in ['active', 'idle']},
            {'from': 'exchange', 'to': 'telegram', 'active': components.get('exchange', {}).get('status') == 'ok'}
        ]
        
        return {
            'timestamp': now.isoformat(),
            'components': components,
            'data_flow': data_flow,
            'errors': errors,
            'warnings': warnings,
            'overall_status': 'error' if errors else 'warning' if warnings else 'ok'
        }
    
    def _get_position_monitor_analysis(self) -> Dict[str, Any]:
        """Get current position monitor analysis - same data used by autonomous monitor.
        
        This ensures Telegram AI uses SAME scores as the autonomous position monitor,
        providing consistency between what AI recommends and what autonomous mode does.
        """
        try:
            if not self.position:
                return None
            
            current_price = self._last_price or self.position.entry_price
            atr = self._last_atr or 0.01
            
            if not self.bars_agg or len(self.bars_agg) < 20:
                return {
                    'action': 'hold',
                    'confidence': 0.5,
                    'reasoning': 'Insufficient data for analysis',
                    'hold_score': 50,
                    'exit_score': 50,
                    'math_score': 50
                }
            
            # Use SAME function as autonomous position monitor
            position_side = self.position.side.upper()
            entry_price = self.position.entry_price
            
            # Calculate unrealized PnL
            if position_side == "LONG":
                unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                unrealized_pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            # Get monitor decision from ai_filter (same as autonomous monitor)
            decision = self.ai_filter.monitor_position(
                df=self.bars_agg,
                position_side=position_side,
                entry_price=entry_price,
                current_price=current_price,
                atr=atr,
                symbol=self.SYMBOL,
                unrealized_pnl_pct=unrealized_pnl_pct,
                tp1_hit=self.position.tp1_hit,
                tp2_hit=self.position.tp2_hit,
                position_size=self.position.size
            )
            
            return {
                'action': decision.get('action', 'hold'),
                'confidence': decision.get('confidence', 0.5),
                'reasoning': decision.get('reasoning', ''),
                'hold_score': decision.get('hold_score', 50),
                'exit_score': decision.get('exit_score', 50),
                'math_score': decision.get('math_score', 50),
                'pnl_pct': unrealized_pnl_pct,
                'position_side': position_side
            }
            
        except Exception as e:
            logger.warning(f"Position monitor analysis error: {e}")
            return None
    
    def _get_full_system_state(self) -> Dict[str, Any]:
        """Get complete system state for AI context - SINGLE SOURCE OF TRUTH.
        
        All data flows through this method to ensure consistency between
        what the AI sees and what Telegram commands display.
        """
        ml_stats = self._get_ml_stats()  # Already normalized
        regime_info = self._get_regime()
        
        # Get market scan data for AI context (top 5 pairs by score)
        market_scan = {}
        try:
            scan_data = self._get_market_scan_data()
            if scan_data and scan_data.get('pairs'):
                pairs = scan_data['pairs'][:5]  # Top 5 by score
                market_scan = {
                    "current_symbol": scan_data.get('current_symbol', ''),
                    "auto_switch_enabled": type(self).AUTO_SWITCH_ENABLED,
                    "multi_pair_enabled": len(self.additional_symbols) > 0,
                    "multi_pair_count": 1 + len(self.additional_symbols),
                    "active_pairs": [self.SYMBOL] + self.additional_symbols,
                    "best_pair": scan_data.get('best_pair', {}).get('symbol') if scan_data.get('best_pair') else None,
                    "top_pairs": [
                        {
                            "symbol": p['symbol'],
                            "price": p['price'],
                            "change": p['change'],
                            "volatility": p.get('volatility', 0),
                            "score": p.get('score', 0),
                            "rsi": p.get('rsi', 50),
                            "adx": p.get('adx', 0),
                            "trend": p.get('trend', 'n/a'),
                            "signal": p.get('signal_text', 'none')
                        }
                        for p in pairs
                    ]
                }
        except Exception as e:
            logger.debug(f"Market scan for context: {e}")
        
        return {
            "parameters": self._get_system_params(),
            "status": self._get_status(),
            "open_position": self._get_current_position_dict(),  # Primary position for backward compatibility
            "additional_positions": self._get_additional_positions(),  # Secondary positions for multi-pair
            "position": self._get_positions(),
            "held_positions": self.held_positions.copy(),  # Which positions are held (bot exit blocked)
            "current_signal": self._get_current_signal(),  # Current signal for display
            "pnl": self._get_pnl(),
            "market": self._get_market(),
            "market_scan": market_scan,
            "ml": ml_stats,  # Use normalized stats directly
            "ai": self._get_ai_stats_for_dashboard(),  # Unified AI stats with mode
            "ai_tracker": self._get_ai_tracker_stats(),  # NEW: AI decision tracking
            "prefilter": self._get_prefilter_stats(),  # Pre-filter statistics
            "regime": regime_info.get("regime", "unknown") if regime_info else "unknown",
            "regime_details": regime_info if regime_info else {},
            "signals": self._get_signals()[-10:] if self._get_signals() else [],
            "trades": self._get_trades()[-10:] if self._get_trades() else [],
            "intelligence": self._get_intelligence(),
        }
    
    def _toggle_summary_notifications(self) -> bool:
        """Toggle summary notifications on/off. Returns new state."""
        self.summary_notifications_enabled = not self.summary_notifications_enabled
        return self.summary_notifications_enabled
    
    async def _chat_with_ai(self, message: str, context: str) -> str:
        """Chat with AI through Telegram."""
        return await self.ai_filter.chat(message, context)
    
    def _get_symbol_trade_context(self, symbol: str) -> Dict[str, Any]:
        """Get critical pre-trade context for a specific symbol.
        
        Returns F&G index, S/R levels, volume, ADX, RSI — everything the AI
        needs to know BEFORE recommending a trade, so it won't get blocked later.
        """
        result = {
            'symbol': symbol,
            'fear_greed_index': 50,
            'fear_greed_label': 'Neutral',
            'support_resistance': {},
            'price': 0,
            'adx': 0,
            'rsi': 50,
            'volume_ratio': 1.0,
            'atr': 0,
            'regime': 'unknown',
            'warnings': []
        }
        
        try:
            # 1. Fear & Greed Index
            news_ctx = self.ai_filter._get_news_context_sync()
            fg = news_ctx.get('fear_greed_index', 50)
            fg_label = news_ctx.get('fear_greed_label', 'Neutral')
            result['fear_greed_index'] = fg
            result['fear_greed_label'] = fg_label
            if fg <= 10:
                result['warnings'].append(f"🔴 EXTREME FEAR ({fg}) — LONG trades will be hard-blocked by bot if momentum is negative")
            elif fg <= 20:
                result['warnings'].append(f"🟠 HIGH FEAR ({fg}) — LONG trades face extra scrutiny")
            elif fg >= 80:
                result['warnings'].append(f"🟢 EXTREME GREED ({fg}) — SHORT trades face extra scrutiny")
            
            # 2. Get symbol data from scan cache
            sym_clean = symbol.upper().replace('/USDT:USDT', '').replace('USDT', '').replace('/', '')
            ccxt_sym = f"{sym_clean}/USDT:USDT"
            
            # Try to get OHLCV data for S/R analysis
            import ccxt as ccxt_sync
            sync_exchange = ccxt_sync.bybit({
                "enableRateLimit": True,
                "options": {"defaultType": "swap", "defaultSubType": "linear", "recvWindow": 20000}
            })
            
            ohlcv = sync_exchange.fetch_ohlcv(ccxt_sym, timeframe='15m', limit=100)
            if ohlcv and len(ohlcv) >= 50:
                import pandas as pd
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                price = df['close'].iloc[-1]
                result['price'] = price
                
                # Calculate ATR
                tr_values = []
                for i in range(1, len(df)):
                    h = df['high'].iloc[i]
                    l = df['low'].iloc[i]
                    pc = df['close'].iloc[i-1]
                    tr_values.append(max(h - l, abs(h - pc), abs(l - pc)))
                if len(tr_values) >= 14:
                    atr = sum(tr_values[-14:]) / 14
                    result['atr'] = atr
                    df['atr'] = atr  # simplified
                
                # 3. Support/Resistance levels
                sr = self.ai_filter._detect_resistance_support_levels(df, price)
                result['support_resistance'] = {
                    'position_in_range_pct': sr.get('position_in_range_pct', 50),
                    'in_resistance_zone': sr.get('in_resistance_zone', False),
                    'in_support_zone': sr.get('in_support_zone', False),
                    'resistance_upper': sr.get('resistance_upper', 0),
                    'support_lower': sr.get('support_lower', 0),
                    'resistance_touches': sr.get('resistance_touches', 0),
                    'support_touches': sr.get('support_touches', 0),
                }
                pos_range = sr.get('position_in_range_pct', 50)
                
                # Add S/R warnings that match final block thresholds
                if pos_range >= 88:
                    result['warnings'].append(f"🚫 TOO CLOSE TO RESISTANCE ({pos_range:.0f}% of range) — LONG will be BLOCKED by bot")
                elif pos_range >= 75:
                    result['warnings'].append(f"⚠️ Near resistance ({pos_range:.0f}% of range) — risky for LONG")
                if pos_range <= 12:
                    result['warnings'].append(f"🚫 TOO CLOSE TO SUPPORT ({pos_range:.0f}% of range) — SHORT will be BLOCKED by bot")
                elif pos_range <= 25:
                    result['warnings'].append(f"⚠️ Near support ({pos_range:.0f}% of range) — risky for SHORT")
                
                # 4. RSI
                closes = df['close'].values
                if len(closes) >= 15:
                    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
                    gains = [d if d > 0 else 0 for d in deltas[-14:]]
                    losses = [-d if d < 0 else 0 for d in deltas[-14:]]
                    avg_gain = sum(gains) / 14
                    avg_loss = sum(losses) / 14
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        result['rsi'] = 100 - (100 / (1 + rs))
                    else:
                        result['rsi'] = 100
                
                # 5. Volume ratio
                if 'volume' in df.columns and len(df) >= 20:
                    recent_vol = df['volume'].iloc[-1]
                    avg_vol = df['volume'].tail(20).mean()
                    if avg_vol > 0:
                        result['volume_ratio'] = round(recent_vol / avg_vol, 2)
                        if result['volume_ratio'] < 0.3:
                            result['warnings'].append(f"📉 Very low volume ({result['volume_ratio']:.1f}x avg) — penalty likely")
                
                # 6. ADX (simplified)
                if len(df) >= 20:
                    # Use scanner cache if available
                    scan_data = self._get_market_scan_data_full()
                    if scan_data and scan_data.get('pairs'):
                        for p in scan_data['pairs']:
                            p_sym = p.get('symbol', '').replace('/USDT:USDT', '').replace('USDT', '')
                            if p_sym == sym_clean:
                                result['adx'] = p.get('adx', 0)
                                result['rsi'] = p.get('rsi', result['rsi'])
                                break
            
            # 7. Session info
            market_hours = self.ai_filter._get_market_hours_context()
            session = market_hours.get('session', 'Unknown')
            activity = market_hours.get('activity_level', 'moderate')
            if activity == 'low':
                result['warnings'].append(f"⏰ Low activity session ({session}) — -10 penalty")
            
        except Exception as e:
            logger.warning(f"⚠️ Symbol trade context error for {symbol}: {e}")
            result['warnings'].append(f"⚠️ Could not fetch full data: {e}")
        
        return result
    
    async def _execute_ai_trade(self, side: str, symbol: str = None) -> Dict[str, Any]:
        """Execute a trade requested by AI chat.
        
        Args:
            side: 'long' or 'short'
            symbol: Optional symbol to trade. If None, uses current symbol.
                    Supports formats: LINKUSDT, LINK, LINK/USDT
            
        Returns:
            Dict with success status and message
        """
        try:
            # Determine target symbol - use ccxt futures format
            target_symbol = self.SYMBOL
            if symbol:
                # Normalize to ccxt futures format (e.g., LINK/USDT:USDT)
                sym_clean = symbol.upper().replace('/', '').replace('-', '').replace(':USDT', '').replace('USDT', '')
                target_symbol = f"{sym_clean}/USDT:USDT"
            
            logger.info(f"🤖 AI Chat: Execute {side.upper()} {target_symbol} requested")
            
            # Check max positions
            open_positions = [s for s, p in self.positions.items() if p is not None]
            max_positions = self.multi_pair_config.max_total_positions
            
            if len(open_positions) >= max_positions:
                return {
                    "success": False,
                    "message": f"Max positions ({max_positions}) reached. Open: {', '.join(open_positions)}. Close one first."
                }
            
            # Check if already have position on this symbol
            existing_pos = self.positions.get(target_symbol)
            if existing_pos:
                return {
                    "success": False,
                    "message": f"Already have {existing_pos.side.upper()} position on {target_symbol}."
                }
            
            # Check if paused
            if self.paused:
                return {
                    "success": False,
                    "message": "Bot is paused. Use /resume to enable trading."
                }
            
            # Get price and data for TARGET symbol (not just primary symbol)
            signal = 1 if side.lower() == "long" else -1
            
            if target_symbol == self.SYMBOL:
                # Use existing price/ATR for primary symbol
                price = self._last_price
                atr = self._calculate_atr()
                df = pd.DataFrame(self.bars_agg) if hasattr(self, 'bars_agg') and len(self.bars_agg) > 0 else None
                
                # FIX: Validate cached price against fresh ticker to catch stale _realtime_price
                # This prevents using old symbol's price after a pair switch
                try:
                    ticker = await self.exchange.fetch_ticker(target_symbol)
                    fresh_price = ticker.get('last', 0)
                    if fresh_price and fresh_price > 0:
                        if price and price > 0:
                            deviation = abs(price - fresh_price) / fresh_price * 100
                            if deviation > 10:  # More than 10% off = stale price
                                logger.warning(f"⚠️ STALE PRICE DETECTED: cached=${price:.6f} vs ticker=${fresh_price:.6f} ({deviation:.1f}% off)")
                                logger.info(f"📊 Using fresh ticker price for {target_symbol}")
                                price = fresh_price
                        else:
                            price = fresh_price
                except Exception as ticker_err:
                    logger.warning(f"⚠️ Could not validate price via ticker: {ticker_err}")
                
                if not price or price <= 0 or atr <= 0:
                    return {
                        "success": False,
                        "message": "Cannot execute - no price data available yet."
                    }
                
                # Execute on primary symbol using _open_position
                await self._open_position(signal, price, atr, source="ai_chat")
                
                # Verify position was opened
                opened_pos = self.position or self.positions.get(self.SYMBOL)
                if not opened_pos:
                    logger.error(f"❌ Position opening FAILED - not persisted")
                    return {
                        "success": False,
                        "message": "Position opening FAILED - order may not have executed. Check Bybit."
                    }
                
                # FIX: Update _symbol_prices so PnL displays correctly immediately
                symbol_key = self.SYMBOL.upper().replace('/', '').replace(':USDT', '')
                if not hasattr(self, '_symbol_prices'):
                    self._symbol_prices = {}
                self._symbol_prices[symbol_key] = price
                logger.debug(f"📊 Updated _symbol_prices[{symbol_key}] = ${price:.4f} for immediate PnL display")
                
            else:
                # Different symbol - need to fetch its data and use _open_position_for_symbol
                logger.info(f"📊 Fetching data for {target_symbol}...")
                
                # Fetch OHLCV for target symbol
                try:
                    import ccxt as ccxt_sync
                    sync_exchange = ccxt_sync.bybit({
                        'apiKey': os.getenv("BYBIT_API_KEY"),
                        'secret': os.getenv("BYBIT_API_SECRET"),
                        'enableRateLimit': True,
                        'options': {'defaultType': 'swap', 'defaultSubType': 'linear', 'recvWindow': 20000}
                    })
                    
                    # target_symbol is already in ccxt format (e.g., ELSA/USDT:USDT)
                    # Use it directly for fetch_ohlcv
                    ohlcv = sync_exchange.fetch_ohlcv(target_symbol, '15m', limit=100)
                    
                    if len(ohlcv) < 20:
                        return {
                            "success": False,
                            "message": f"Insufficient data for {target_symbol} (only {len(ohlcv)} candles)"
                        }
                    
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    price = float(df['close'].iloc[-1])
                    
                    # Calculate ATR
                    tr1 = df['high'] - df['low']
                    tr2 = abs(df['high'] - df['close'].shift(1))
                    tr3 = abs(df['low'] - df['close'].shift(1))
                    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    atr = float(true_range.rolling(14).mean().iloc[-1])
                    
                    if atr <= 0:
                        atr = price * 0.01  # Fallback: 1% of price
                    
                    logger.info(f"📊 {target_symbol}: Price=${price:.4f}, ATR=${atr:.4f}")
                    
                except Exception as fetch_err:
                    logger.error(f"Failed to fetch data for {target_symbol}: {fetch_err}")
                    return {
                        "success": False,
                        "message": f"Failed to fetch data for {target_symbol}: {str(fetch_err)}"
                    }
                
                # Execute on secondary symbol using _open_position_for_symbol
                await self._open_position_for_symbol(
                    symbol=target_symbol,
                    signal=signal,
                    price=price,
                    atr=atr,
                    df=df,
                    risk_pct=self.RISK_PCT
                )
                
                # Verify position was opened
                opened_pos = self.positions.get(target_symbol)
                if not opened_pos:
                    logger.error(f"❌ Position opening FAILED for {target_symbol} - not persisted")
                    return {
                        "success": False,
                        "message": f"Position opening FAILED for {target_symbol} - order may not have executed. Check Bybit."
                    }
                
                # FIX: Update _symbol_prices so PnL displays correctly immediately
                symbol_key = target_symbol.upper().replace('/', '').replace(':USDT', '')
                if not hasattr(self, '_symbol_prices'):
                    self._symbol_prices = {}
                self._symbol_prices[symbol_key] = price
                logger.debug(f"📊 Updated _symbol_prices[{symbol_key}] = ${price:.4f} for immediate PnL display")
            
            return {
                "success": True,
                "message": f"✅ Opened {side.upper()} {target_symbol} at ${price:.4f}",
                "price": price,
                "side": side.upper(),
                "symbol": target_symbol
            }
            
        except Exception as e:
            logger.error(f"AI trade execution error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Trade failed: {str(e)}"
            }
    
    async def _close_ai_trade(self, symbol: str = None) -> Dict[str, Any]:
        """Close position requested by AI chat.
        
        Args:
            symbol: Optional symbol to close. If None, closes primary position.
                    Supports formats: LINKUSDT, LINK, LINK/USDT
        
        Returns:
            Dict with success status and message
        """
        try:
            target_symbol = symbol
            pos = None
            
            if target_symbol:
                # Normalize symbol format
                target_symbol = target_symbol.upper().replace('/', '').replace('-', '')
                if not target_symbol.endswith('USDT'):
                    target_symbol = target_symbol + 'USDT'
                
                # Find position for specified symbol
                for sym, p in self.positions.items():
                    sym_normalized = sym.upper().replace('/', '').replace('-', '')
                    if sym_normalized == target_symbol and p is not None:
                        pos = p
                        break
                
                if not pos:
                    # List open positions for user
                    open_symbols = [s for s, p in self.positions.items() if p is not None]
                    if open_symbols:
                        return {
                            "success": False,
                            "message": f"No position found for {target_symbol}. Open positions: {', '.join(open_symbols)}"
                        }
                    return {
                        "success": False,
                        "message": f"No position found for {target_symbol}. No open positions."
                    }
            else:
                # No symbol specified - close primary position
                pos = self.position
                if not pos and self.SYMBOL in self.positions:
                    pos = self.positions[self.SYMBOL]
                target_symbol = self.SYMBOL
            
            if not pos:
                # List open positions for user
                open_symbols = [s for s, p in self.positions.items() if p is not None]
                if open_symbols:
                    return {
                        "success": False,
                        "message": f"No primary position. Open positions: {', '.join(open_symbols)}. Specify symbol: e.g., 'close ETHUSDT'"
                    }
                return {
                    "success": False,
                    "message": "No open position to close."
                }
            
            # Get the correct price for the target symbol (multi-pair support)
            actual_symbol = pos.symbol if hasattr(pos, 'symbol') else target_symbol
            if actual_symbol and actual_symbol != self.SYMBOL:
                # Fetch price for the specific symbol being closed
                try:
                    ticker = await self.exchange.fetch_ticker(actual_symbol)
                    price = float(ticker.get('last') or ticker.get('close', 0))
                    logger.info(f"📊 Fetched price for {actual_symbol}: ${price:.4f}")
                except Exception as e:
                    logger.error(f"Failed to fetch price for {actual_symbol}: {e}")
                    # Fallback: try sync ccxt
                    try:
                        import ccxt
                        sync_exchange = ccxt.bybit({
                            'apiKey': os.getenv("BYBIT_API_KEY"),
                            'secret': os.getenv("BYBIT_API_SECRET"),
                            'options': {'defaultType': 'swap', 'defaultSubType': 'linear', 'recvWindow': 20000}
                        })
                        ticker = sync_exchange.fetch_ticker(actual_symbol)
                        price = float(ticker.get('last') or ticker.get('close', 0))
                        logger.info(f"📊 Fetched price (sync fallback) for {actual_symbol}: ${price:.4f}")
                    except Exception as e2:
                        logger.error(f"Sync fallback also failed: {e2}")
                        price = None
            else:
                price = self._last_price
                
            if not price:
                return {
                    "success": False,
                    "message": "Cannot close - no price data available."
                }
            
            side = pos.side.upper()
            
            # Use the correct close method based on whether it's multi-position
            close_success = False
            if actual_symbol and actual_symbol != self.SYMBOL and actual_symbol in self.positions:
                # Close secondary position using multi-position close
                close_success = await self._close_position_by_symbol(actual_symbol, "AI Chat Request", price)
            else:
                # Close primary position
                close_success = await self._close_position("AI Chat Request", price, is_manual=True)
            
            if close_success:
                return {
                    "success": True,
                    "message": f"Closed {side} position at ${price:.4f}"
                }
            else:
                return {
                    "success": False,
                    "message": f"Close rejected - price validation failed (${price:.4f} for {actual_symbol})"
                }
            
        except Exception as e:
            logger.error(f"AI close error: {e}")
            return {
                "success": False,
                "message": f"Close failed: {str(e)}"
            }
    
    # Minimum margin required to open a position (in USD)
    MIN_MARGIN_THRESHOLD = 5.0  # Skip trades if available margin is below $5
    
    def _get_status(self) -> Dict[str, Any]:
        """Get bot status for Telegram."""
        if self.start_time:
            uptime = datetime.now(timezone.utc) - self.start_time
            hours, remainder = divmod(int(uptime.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            uptime_str = f"{hours}h {minutes}m {seconds}s"
        else:
            uptime_str = "Starting..."
        
        # Get current price and ATR
        current_price = self.cached_last_price or 0
        # Fall back to bars data if no cached price yet
        if current_price == 0 and hasattr(self, 'bars_agg') and len(self.bars_agg) > 0:
            current_price = float(self.bars_agg.iloc[-1]['close'])
        current_atr = self.cached_last_atr or 0
        
        # Position info - count ALL positions, not just the current symbol
        pos = None
        num_positions = 0
        
        # Check positions dict for ANY open positions
        if hasattr(self, 'positions') and self.positions:
            for sym, p in self.positions.items():
                if p is not None:
                    num_positions += 1
                    if pos is None:
                        pos = p  # Use first position for display
        
        # Also check legacy self.position
        if not pos and self.position:
            pos = self.position
            num_positions = max(num_positions, 1)
        
        has_position = num_positions > 0
        position_side = pos.side.upper() if pos else "None"
        position_pnl = pos.unrealized_pnl(current_price) if pos and current_price else 0
        
        # Get entry price for status
        entry_price = pos.entry_price if has_position else None
        stop_loss = pos.stop_loss if has_position else None
        tp1 = pos.tp1 if has_position else None
        
        # Stats - use correct attribute names from TradeStats
        total_trades = self.stats.total_trades
        wins = self.stats.winning_trades
        losses = self.stats.losing_trades
        win_rate = self.stats.win_rate * 100  # Convert to percentage
        
        # Get UTA wallet info for live mode
        uta_info = {}
        if self.live_mode and hasattr(self, '_cached_uta_info'):
            uta_info = self._cached_uta_info
        
        return {
            "connected": self.exchange is not None,
            "symbol": self.SYMBOL,
            "uptime": uptime_str,
            "mode": ("📝 DRY-RUN" if self.dry_run_mode else "Paper") if self.paper_mode else ("🔴 LIVE" if self.live_mode else "Sim"),
            "paused": self.paused,
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "current_price": current_price,
            "atr": current_atr,
            "has_position": has_position,
            "num_positions": num_positions,
            "position_side": position_side,
            "position_pnl": position_pnl,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "tp1": tp1,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": self.stats.total_pnl,
            "signals_checked": len(self.signal_history),
            # UTA (Unified Trading Account) margin info
            "uta_wallet_balance": uta_info.get('wallet_balance', 0),
            "uta_equity": uta_info.get('equity', 0),
            "uta_borrowed": uta_info.get('borrowed', 0),
            "uta_available_margin": uta_info.get('available_margin', 0),
            "uta_margin_used": uta_info.get('margin_used', 0),
            "uta_ltv": uta_info.get('ltv', 0),
        }
    
    def _get_ml_status(self) -> Dict[str, Any]:
        """Get ML model status for dashboard."""
        from indicator import get_ml_classifier
        
        # Get live classifier stats
        live_classifier = get_ml_classifier()
        live_stats = live_classifier.get_stats() if live_classifier else {}
        
        if not self.ml_predictor or not self.ml_predictor.is_loaded:
            return {
                "loaded": False,
                "status": "Not Available",
                "accuracy": 0,
                "samples": 0,
                "influence": 0,
                "last_prediction": None,
                "live_samples": live_stats.get('total_samples', 0),
                "live_trained": live_stats.get('is_trained', False),
                "live_wins": live_stats.get('wins', 0),
                "live_losses": live_stats.get('losses', 0),
                "live_needed": live_stats.get('samples_until_training', 50)
            }
        
        # Use metrics attribute instead of model_metadata
        metrics = getattr(self.ml_predictor, 'metrics', {})
        last_pred = None
        if hasattr(self.ml_predictor, 'prediction_log') and self.ml_predictor.prediction_log:
            last_pred = self.ml_predictor.prediction_log[-1] if self.ml_predictor.prediction_log else None
        
        return {
            "loaded": True,
            "status": "Active (Advisory)" if metrics.get('accuracy', 0) > 0 else "Loaded",
            "accuracy": metrics.get('accuracy', 0),
            "samples": metrics.get('total_samples', 0),
            "influence": 0.0,  # Currently advisory only
            "model_path": str(self.ml_predictor.model_path),
            "features": len(self.ml_predictor.feature_columns),
            "last_prediction": last_pred,
            # Live classifier stats (learns from live trades)
            "live_samples": live_stats.get('total_samples', 0),
            "live_trained": live_stats.get('is_trained', False),
            "live_wins": live_stats.get('wins', 0),
            "live_losses": live_stats.get('losses', 0),
            "live_needed": live_stats.get('samples_until_training', 50)
        }

    def _get_ai_stats_for_dashboard(self) -> Dict[str, Any]:
        """Get AI stats for dashboard with correct mode from bot."""
        stats = self.ai_filter.get_stats()
        # Add the actual bot ai_mode (not from ai_filter)
        stats['mode'] = self.ai_mode
        stats['threshold'] = self.ai_filter.confidence_threshold
        stats['last_decision'] = getattr(self.ai_filter, 'last_decision', None)
        return stats

    def _get_ai_tracker_stats(self) -> Dict[str, Any]:
        """Get AI decision tracking statistics."""
        summary = self.ai_tracker.get_accuracy_summary()
        recent = self.ai_tracker.get_recent_decisions(count=5)
        
        return {
            "summary": summary,
            "recent_decisions": recent,
            "total_tracked": summary.get('total_decisions', 0),
            "approval_rate": f"{summary.get('approval_rate', 0) * 100:.1f}%",
            "approval_accuracy": f"{summary.get('approval_accuracy', 0) * 100:.1f}%" if summary.get('approved_with_outcome', 0) > 0 else "N/A",
            "net_ai_value": f"${summary.get('net_ai_value', 0):+.2f}"
        }

    def _get_prefilter_stats(self) -> Dict[str, Any]:
        """Get pre-filter statistics for dashboard."""
        stats = self.prefilter_stats.copy()
        total = stats.get('total_signals', 0)
        passed = stats.get('passed', 0)
        raw = stats.get('raw_signals', 0)
        
        # Calculate pass rate
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        # Calculate block reasons breakdown
        blocked_total = total - passed
        
        return {
            "raw_signals": raw,
            "total_signals": total,
            "passed": passed,
            "blocked": blocked_total,
            "pass_rate": f"{pass_rate:.1f}%",
            "blocked_by_score": stats.get('blocked_score', 0),
            "blocked_by_adx_low": stats.get('blocked_adx_low', 0),
            "blocked_by_adx_danger": stats.get('blocked_adx_danger', 0),
            "blocked_by_volume": stats.get('blocked_volume', 0),
            "blocked_by_confluence": stats.get('blocked_confluence', 0),
            "blocked_by_btc_filter": stats.get('blocked_btc_filter', 0),
            "by_regime": stats.get('by_regime', {}),
            "adx_danger_zone": self.ADX_DANGER_ZONE,
            "min_volume_ratio": self.MIN_VOLUME_RATIO
        }

    def _get_system_logs(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent system logs for dashboard."""
        logs = []
        log_file = Path(__file__).parent / "julaba.log"
        
        try:
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    # Get last N lines
                    recent_lines = lines[-count:] if len(lines) > count else lines
                    
                    for line in recent_lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Parse log line: "2026-01-09 20:42:57 [INFO] Julaba: message"
                        try:
                            parts = line.split(' ', 3)
                            if len(parts) >= 4:
                                timestamp = f"{parts[0]} {parts[1]}"
                                level = parts[2].strip('[]')
                                message = parts[3] if len(parts) > 3 else ""
                                
                                logs.append({
                                    "time": timestamp,
                                    "level": level,
                                    "message": message[:200]  # Truncate long messages
                                })
                            else:
                                logs.append({
                                    "time": "",
                                    "level": "INFO",
                                    "message": line[:200]
                                })
                        except Exception:
                            logs.append({
                                "time": "",
                                "level": "INFO", 
                                "message": line[:200]
                            })
        except Exception as e:
            logs.append({
                "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "level": "ERROR",
                "message": f"Failed to read logs: {e}"
            })
        
        return logs

    def _get_next_available_slot(self) -> int:
        """Get the next available position slot number (1 or 2)."""
        used_slots = set(self.position_slots.values())
        max_slots = self.multi_pair_config.max_total_positions if hasattr(self, 'multi_pair_config') else 2
        for slot in range(1, max_slots + 1):
            if slot not in used_slots:
                return slot
        return len(used_slots) + 1  # Fallback if all slots used

    def _assign_position_slot(self, symbol: str) -> int:
        """Assign a slot number to a new position."""
        symbol_key = symbol.upper().replace('/', '')
        if symbol_key in self.position_slots:
            return self.position_slots[symbol_key]  # Already has a slot
        slot = self._get_next_available_slot()
        self.position_slots[symbol_key] = slot
        logger.info(f"📍 Assigned slot {slot} to {symbol_key}")
        return slot

    def _release_position_slot(self, symbol: str):
        """Release a slot when position is closed and promote remaining positions."""
        symbol_key = symbol.upper().replace('/', '').replace(':USDT', '')
        if symbol_key in self.position_slots:
            released_slot = self.position_slots.pop(symbol_key)
            logger.info(f"📍 Released slot {released_slot} from {symbol_key}")
            
            # Promote remaining positions to fill the gap (slot 2 -> slot 1 if slot 1 is freed)
            if released_slot == 1 and self.position_slots:
                # Find the position in slot 2 and promote it to slot 1
                for sym, slot in list(self.position_slots.items()):
                    if slot == 2:
                        self.position_slots[sym] = 1
                        logger.info(f"📍 Promoted {sym} from slot 2 to slot 1")
                        
                        # Also update self.position to point to the promoted position
                        # Try multiple key formats to find the position
                        promoted_pos = None
                        if sym in self.positions and self.positions[sym]:
                            promoted_pos = self.positions[sym]
                        else:
                            # Try alternate key formats
                            for pos_key, pos in self.positions.items():
                                if pos and normalize_symbol(pos_key) == normalize_symbol(sym):
                                    promoted_pos = pos
                                    break
                        
                        if promoted_pos:
                            self.position = promoted_pos
                            old_symbol = self.SYMBOL
                            self.SYMBOL = promoted_pos.symbol  # Use the position's actual symbol
                            logger.info(f"📍 {promoted_pos.symbol} is now the primary position (was {old_symbol})")
                            
                            # Save state to persist the promotion
                            self._save_trading_state()
                            
                            # Restart WebSocket with new symbol if needed
                            if hasattr(self, 'ws_stream') and self.ws_stream:
                                import asyncio
                                try:
                                    asyncio.create_task(self._restart_websocket_for_promotion())
                                except Exception as e:
                                    logger.warning(f"Could not restart WebSocket: {e}")
                        break
            
            # === ALL POSITIONS CLOSED: Reset to scan all pairs ===
            # Check if we now have zero positions after this release
            open_positions = [p for p in self.positions.values() if p is not None]
            if len(open_positions) == 0 and not self.position:
                logger.info(f"📊 ALL POSITIONS CLOSED - will scan all pairs for best opportunity")
                # Set a flag to trigger full pair scan on next iteration
                self._needs_full_pair_scan = True

    def _clear_position(self, symbol: str):
        """Standardized method to clear a position from all tracking structures.
        
        This ensures consistent cleanup by:
        1. Setting positions[symbol] = None (not del) to preserve dict structure
        2. Releasing the position slot
        3. Clearing legacy self.position if it matches
        4. Clearing any hold state
        5. Saving state
        6. Marking as recently closed (prevents duplicate recording)
        
        Use this instead of manually doing `del self.positions[x]` or `= None`.
        """
        # Normalize symbol to find it in various formats
        symbol_key = normalize_symbol(symbol)
        
        # === Mark as recently closed so _record_externally_closed_trade skips it ===
        if not hasattr(self, '_recently_closed_symbols'):
            self._recently_closed_symbols = {}
        self._recently_closed_symbols[symbol_key] = datetime.now(timezone.utc)
        logger.debug(f"🧹 Marked {symbol_key} as recently closed (duplicate prevention)")
        
        # Clear from positions dict - set to None (consistent approach)
        # Check multiple key formats
        cleared = False
        for key in list(self.positions.keys()):
            if normalize_symbol(key) == symbol_key:
                self.positions[key] = None
                cleared = True
                logger.debug(f"🧹 Cleared position for {key}")
                break
        
        # Also try direct symbol
        if not cleared and symbol in self.positions:
            self.positions[symbol] = None
            cleared = True
        
        # Release the slot
        self._release_position_slot(symbol)
        
        # Clear any hold state for this symbol
        if symbol_key in self.held_positions:
            del self.held_positions[symbol_key]
            logger.debug(f"🧹 Cleared hold state for {symbol_key}")
        
        # Clear stored original SL (from diamond hands mode)
        if hasattr(self, '_held_original_sl') and symbol_key in self._held_original_sl:
            del self._held_original_sl[symbol_key]
            logger.debug(f"🧹 Cleared stored original SL for {symbol_key}")
        
        # Clear diamond hands log tracking
        if hasattr(self, '_dh_last_log') and symbol_key in self._dh_last_log:
            del self._dh_last_log[symbol_key]
        
        # Clear atomic SL tighten flag so next trade starts fresh
        if hasattr(self, '_atomic_sl_tightened') and symbol_key in self._atomic_sl_tightened:
            del self._atomic_sl_tightened[symbol_key]
            logger.debug(f"🧹 Cleared atomic SL tighten flag for {symbol_key}")
        
        # Clear peak profit cache so next trade starts fresh
        if hasattr(self, 'ai_filter') and self.ai_filter:
            # Try all symbol format variations
            for sym_variant in [symbol, symbol_key, symbol.replace('/', '').replace(':USDT', '')]:
                self.ai_filter.clear_peak_profit(sym_variant)
            logger.info(f"🧹 Cleared peak profit cache for {symbol_key}")
        
        # Also clear momentum cache so stale data doesn't affect new trades
        if hasattr(self, 'ai_filter') and self.ai_filter and hasattr(self.ai_filter, '_momentum_cache'):
            for sym_variant in [symbol, symbol_key, symbol.replace('/', '').replace(':USDT', '')]:
                if sym_variant in self.ai_filter._momentum_cache:
                    del self.ai_filter._momentum_cache[sym_variant]
            logger.info(f"🧹 Cleared momentum cache for {symbol_key}")
        
        # Clear legacy position if it matches
        if self.position:
            pos_symbol_key = normalize_symbol(getattr(self.position, 'symbol', ''))
            if pos_symbol_key == symbol_key:
                self.position = None
                logger.debug(f"🧹 Cleared legacy self.position")
        
        # Save state
        self._save_trading_state()

    async def _restart_websocket_for_promotion(self):
        """Restart WebSocket subscription after position promotion."""
        try:
            if self.ws_stream:
                await self.ws_stream.stop()
                await asyncio.sleep(1)
                await self._start_websocket()
                logger.info(f"📍 WebSocket restarted for new primary symbol: {self.SYMBOL}")
        except Exception as e:
            logger.warning(f"WebSocket restart failed: {e}")

    def _get_positions(self) -> List[Dict[str, Any]]:
        """Get positions for Telegram - includes legacy single position and multi-pair positions."""
        positions_list = []
        added_symbols = set()
        
        # Check legacy single position first (for backward compatibility)
        if self.position:
            symbol_key = self.position.symbol.upper().replace('/', '').replace(':USDT', '')
            # Ensure it has a slot
            if symbol_key not in self.position_slots:
                self._assign_position_slot(symbol_key)
            slot = self.position_slots.get(symbol_key, 1)
            
            # FIX: Use correct price for THIS position's symbol, not _last_price which is main symbol only
            current_price = self.position.entry_price  # Default fallback
            if hasattr(self, '_symbol_prices') and symbol_key in self._symbol_prices:
                current_price = self._symbol_prices[symbol_key]
            elif hasattr(self, '_symbol_prices'):
                # Try alternate key format
                alt_key = symbol_key.replace('USDT', '') + 'USDT'
                if alt_key in self._symbol_prices:
                    current_price = self._symbol_prices[alt_key]
            # Use _last_price ONLY if this is the main symbol
            if not current_price or current_price == self.position.entry_price:
                main_symbol_key = self.SYMBOL.replace('/', '').replace(':USDT', '')
                if symbol_key == main_symbol_key and self._last_price:
                    current_price = self._last_price
            
            positions_list.append({
                "slot": slot,
                "symbol": self.position.symbol,
                "side": self.position.side.upper(),
                "entry": self.position.entry_price,
                "size": self.position.remaining_size,
                "pnl": self.position.unrealized_pnl(current_price),
                "current_price": current_price
            })
            added_symbols.add(symbol_key)
        
        # Check multi-pair positions dict
        for symbol, position in self.positions.items():
            symbol_key = symbol.upper().replace('/', '').replace(':USDT', '')
            # Skip None positions (closed) and already added
            if position and symbol_key not in added_symbols:
                # Ensure it has a slot
                if symbol_key not in self.position_slots:
                    self._assign_position_slot(symbol_key)
                slot = self.position_slots.get(symbol_key, 2)
                
                # FIX: Use correct price for THIS position's symbol from _symbol_prices cache
                current_price = position.entry_price  # Default fallback
                if hasattr(self, '_symbol_prices') and symbol_key in self._symbol_prices:
                    current_price = self._symbol_prices[symbol_key]
                elif hasattr(self, '_symbol_prices'):
                    # Try alternate key format (LINKUSDT vs LINK)
                    alt_key = symbol_key.replace('USDT', '') + 'USDT'
                    if alt_key in self._symbol_prices:
                        current_price = self._symbol_prices[alt_key]
                # Use _last_price ONLY if this is the main symbol
                if not current_price or current_price == position.entry_price:
                    main_symbol_key = self.SYMBOL.replace('/', '').replace(':USDT', '')
                    if symbol_key == main_symbol_key and self._last_price:
                        current_price = self._last_price
                
                positions_list.append({
                    "slot": slot,
                    "symbol": position.symbol,
                    "side": position.side.upper(),
                    "entry": position.entry_price,
                    "size": position.remaining_size,
                    "pnl": position.unrealized_pnl(current_price),
                    "current_price": current_price
                })
                added_symbols.add(symbol_key)
        
        # Sort by slot number so positions are always shown in consistent order
        positions_list.sort(key=lambda x: x.get('slot', 99))
        
        return positions_list
    
    def _get_current_position_dict(self) -> Optional[Dict[str, Any]]:
        """Get current position as dict for dashboard (Position 1 / Slot 1).
        
        FIXED: Returns the position in SLOT 1 (first open position), NOT based on self.SYMBOL.
        This ensures Position 1 chart always matches Position 1 data, regardless of
        what symbol the bot is currently scanning.
        
        PnL includes both realized (from TPs) and unrealized profit for accurate display.
        """
        pos = None
        pos_symbol_key = None
        
        # CLEANUP: Remove stale slots that don't have corresponding open positions
        current_open_symbols = set()
        if hasattr(self, 'positions') and self.positions:
            for sym_key, position in self.positions.items():
                if position:
                    sym_clean = sym_key.upper().replace('/USDT:USDT', 'USDT').replace('/', '').replace(':USDT', '')
                    current_open_symbols.add(sym_clean)
        if self.position:
            pos_sym = (getattr(self.position, 'symbol', '') or '').upper()
            pos_sym_clean = pos_sym.replace('/USDT:USDT', 'USDT').replace('/', '').replace(':USDT', '')
            current_open_symbols.add(pos_sym_clean)
        
        # Remove slots for symbols that no longer have open positions
        stale_slots = [sym for sym in self.position_slots if sym not in current_open_symbols]
        for sym in stale_slots:
            del self.position_slots[sym]
        
        # Reassign slots if needed (ensure slot 1 is used first)
        if current_open_symbols and 1 not in self.position_slots.values():
            # No slot 1 - reassign the first open position to slot 1
            for sym in current_open_symbols:
                if sym not in self.position_slots:
                    self.position_slots[sym] = 1
                    break
                elif self.position_slots[sym] != 1:
                    # Move this to slot 1
                    self.position_slots[sym] = 1
                    break
        
        # Find SLOT 1 position - this is Position 1 for the dashboard
        slot1_symbol = None
        for sym, slot in self.position_slots.items():
            if slot == 1:
                slot1_symbol = sym
                break
        
        # First, check if we have a position in the assigned slot 1
        if slot1_symbol:
            # Check positions dict
            if hasattr(self, 'positions') and self.positions:
                for sym_key, position in self.positions.items():
                    if position:
                        sym_clean = sym_key.upper().replace('/USDT:USDT', 'USDT').replace('/', '').replace(':USDT', '')
                        if sym_clean == slot1_symbol:
                            pos = position
                            pos_symbol_key = sym_clean
                            break
            # Check legacy self.position
            if not pos and self.position:
                pos_sym = (getattr(self.position, 'symbol', '') or '').upper()
                pos_sym_clean = pos_sym.replace('/USDT:USDT', 'USDT').replace('/', '').replace(':USDT', '')
                if pos_sym_clean == slot1_symbol:
                    pos = self.position
                    pos_symbol_key = pos_sym_clean
        
        # Fallback: If no slot 1 assigned, just get the FIRST open position
        if not pos:
            # Check legacy position first
            if self.position:
                pos = self.position
                pos_symbol_key = (getattr(self.position, 'symbol', '') or '').upper().replace('/USDT:USDT', 'USDT').replace('/', '').replace(':USDT', '')
            # Check positions dict
            elif hasattr(self, 'positions') and self.positions:
                for sym_key, position in self.positions.items():
                    if position:
                        pos = position
                        pos_symbol_key = sym_key.upper().replace('/USDT:USDT', 'USDT').replace('/', '').replace(':USDT', '')
                        # Assign slot 1 if not already assigned
                        if pos_symbol_key not in self.position_slots:
                            self._assign_position_slot(pos_symbol_key)
                        break
        
        if not pos:
            return None
        
        # Get correct price for THIS position's symbol (not self._last_price which is main scan symbol)
        current_price = pos.entry_price  # Default fallback
        if pos_symbol_key and hasattr(self, '_symbol_prices'):
            if pos_symbol_key in self._symbol_prices:
                current_price = self._symbol_prices[pos_symbol_key]
            elif pos_symbol_key.replace('USDT', '') + 'USDT' in self._symbol_prices:
                current_price = self._symbol_prices[pos_symbol_key.replace('USDT', '') + 'USDT']
        # Only use _last_price if this position's symbol matches current scan symbol
        if current_price == pos.entry_price:
            main_scan_key = self.SYMBOL.replace('/', '').replace(':USDT', '')
            if pos_symbol_key == main_scan_key and self._last_price:
                current_price = self._last_price
        
        # Calculate TOTAL PnL (realized from TPs + unrealized on remaining)
        # This gives accurate display while position is open
        unrealized_pnl = pos.unrealized_pnl(current_price)
        realized_pnl = 0.0
        
        # Add realized profit from TP1 if hit
        if pos.tp1_hit:
            tp1_size = pos.size * self.TP1_PCT  # 50%
            if pos.side == "long":
                realized_pnl += (pos.tp1 - pos.entry_price) * tp1_size
            else:
                realized_pnl += (pos.entry_price - pos.tp1) * tp1_size
        
        # Add realized profit from TP2 if hit
        if pos.tp2_hit:
            tp2_size = pos.size * self.TP2_PCT  # 30%
            if pos.side == "long":
                realized_pnl += (pos.tp2 - pos.entry_price) * tp2_size
            else:
                realized_pnl += (pos.entry_price - pos.tp2) * tp2_size
        
        total_pnl = realized_pnl + unrealized_pnl
        
        # Calculate PnL percentage based on ORIGINAL position value
        original_value = pos.entry_price * pos.size
        pnl_percent = (total_pnl / original_value * 100) if original_value > 0 else 0
        
        return {
            "symbol": pos.symbol,
            "side": pos.side.upper(),
            "entry": pos.entry_price,
            "size": pos.remaining_size,
            "original_size": pos.size,
            "pnl": total_pnl,  # Total PnL (realized + unrealized)
            "unrealized_pnl": unrealized_pnl,  # Just unrealized on remaining
            "realized_pnl": realized_pnl,  # Realized from TPs
            "pnl_percent": pnl_percent,  # Percentage based on original position
            "stop_loss": pos.stop_loss,
            "tp1": pos.tp1,
            "tp2": pos.tp2,
            "tp3": pos.tp3,
            "current_price": current_price,
            "entry_time": int(pos.opened_at.timestamp() * 1000) if hasattr(pos, 'opened_at') and pos.opened_at else None,
            "leverage": getattr(pos, 'leverage', 1),
            "tp1_hit": getattr(pos, 'tp1_hit', False),
            "tp2_hit": getattr(pos, 'tp2_hit', False),
            "tp3_hit": getattr(pos, 'tp3_hit', False)
        }
    
    def _get_additional_positions(self) -> List[Dict[str, Any]]:
        """Get all additional open positions (multi-pair) for dashboard.
        
        PnL includes both realized (from TPs) and unrealized profit for accurate display.
        """
        positions = []
        
        # Find which symbol is in SLOT 1 (shown as Position 1 in dashboard)
        slot1_symbol = None
        for sym, slot in self.position_slots.items():
            if slot == 1:
                slot1_symbol = sym
                break
        
        # Also track legacy position symbol to avoid duplicates
        legacy_symbol_key = None
        if self.position:
            legacy_symbol_key = self.position.symbol.upper().replace('/USDT:USDT', 'USDT').replace('/', '').replace(':USDT', '')
        
        # FIXED: Get all positions NOT in slot 1 (these go to Position 2 chart)
        # No longer checks against self.SYMBOL which changes with scanning
        for symbol, pos in self.positions.items():
            if pos:
                symbol_key = symbol.upper().replace('/USDT:USDT', 'USDT').replace('/', '').replace(':USDT', '')
                
                # Skip if this is slot 1 position (already shown as Position 1)
                if slot1_symbol and symbol_key == slot1_symbol:
                    continue
                
                # Skip if this is the same as legacy position AND legacy is slot 1
                if legacy_symbol_key and symbol_key == legacy_symbol_key:
                    if not slot1_symbol or legacy_symbol_key == slot1_symbol:
                        continue  # Skip - already shown as Position 1
                
                # Use cached price for this specific symbol
                # The position monitor updates _symbol_prices cache
                current_price = pos.entry_price  # Default fallback
                
                if hasattr(self, '_symbol_prices') and symbol_key in self._symbol_prices:
                    current_price = self._symbol_prices[symbol_key]
                else:
                    # Try alternate key formats
                    alt_key = symbol.replace('/', '').replace(':USDT', '')
                    if hasattr(self, '_symbol_prices') and alt_key in self._symbol_prices:
                        current_price = self._symbol_prices[alt_key]
                
                # Calculate TOTAL PnL (realized from TPs + unrealized on remaining)
                unrealized_pnl = pos.unrealized_pnl(current_price)
                realized_pnl = 0.0
                
                # Add realized profit from TP1 if hit
                if pos.tp1_hit:
                    tp1_size = pos.size * self.TP1_PCT
                    if pos.side == "long":
                        realized_pnl += (pos.tp1 - pos.entry_price) * tp1_size
                    else:
                        realized_pnl += (pos.entry_price - pos.tp1) * tp1_size
                
                # Add realized profit from TP2 if hit
                if pos.tp2_hit:
                    tp2_size = pos.size * self.TP2_PCT
                    if pos.side == "long":
                        realized_pnl += (pos.tp2 - pos.entry_price) * tp2_size
                    else:
                        realized_pnl += (pos.entry_price - pos.tp2) * tp2_size
                
                total_pnl = realized_pnl + unrealized_pnl
                
                # Calculate PnL percentage based on ORIGINAL position value
                original_value = pos.entry_price * pos.size
                pnl_percent = (total_pnl / original_value * 100) if original_value > 0 else 0
                
                positions.append({
                    "symbol": pos.symbol,
                    "side": pos.side.upper(),
                    "entry": pos.entry_price,
                    "entry_price": pos.entry_price,  # Alias for dashboard compatibility
                    "size": pos.remaining_size,
                    "original_size": pos.size,
                    "pnl": total_pnl,  # Total PnL (realized + unrealized)
                    "unrealized_pnl": unrealized_pnl,
                    "realized_pnl": realized_pnl,
                    "pnl_percent": pnl_percent,
                    "stop_loss": pos.stop_loss,
                    "tp1": pos.tp1,
                    "tp2": pos.tp2,
                    "tp3": pos.tp3,
                    "current_price": current_price,
                    "entry_time": int(pos.opened_at.timestamp() * 1000) if hasattr(pos, 'opened_at') and pos.opened_at else None,
                    "leverage": getattr(pos, 'leverage', 1),
                    "tp1_hit": getattr(pos, 'tp1_hit', False),
                    "tp2_hit": getattr(pos, 'tp2_hit', False),
                    "tp3_hit": getattr(pos, 'tp3_hit', False)
                })
        
        return positions
    
    def _get_pnl(self) -> Dict[str, Any]:
        """Get P&L for Telegram."""
        # Reset daily P&L if new day
        now = datetime.now(timezone.utc)
        if now.date() > self.stats.last_reset.date():
            self.stats.today_pnl = 0.0
            self.stats.last_reset = now
        
        return {
            "today": self.stats.today_pnl,
            "total": self.stats.total_pnl,
            "win_rate": self.stats.win_rate,
            "trades": self.stats.total_trades,
            "winning": self.stats.winning_trades,
            "max_win": self.stats.max_win,
            "max_loss": self.stats.max_loss,
            "avg_trade": self.stats.total_pnl / max(1, self.stats.total_trades)
        }
    
    def _get_balance(self) -> Dict[str, Any]:
        """Get balance info for Telegram."""
        change = self.balance - self.initial_balance
        change_pct = (change / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        return {
            "current": self.balance,
            "initial": self.initial_balance,
            "change": change,
            "change_pct": change_pct
        }
    
    def _get_trades(self) -> List[Dict]:
        """Get trade history for Telegram."""
        return self.trade_history
    
    def _get_market(self) -> Dict[str, Any]:
        """Get market info for Telegram."""
        return {
            "symbol": self.SYMBOL,
            "price": self.cached_last_price,
            "change_24h": self.cached_last_ticker.get('percentage', 0),
            "volume_24h": self.cached_last_ticker.get('quoteVolume', 0),
            "high_24h": self.cached_last_ticker.get('high', 0),
            "low_24h": self.cached_last_ticker.get('low', 0),
            "atr": self.cached_last_atr
        }
    
    def _get_signals(self) -> List[Dict]:
        """Get signal history for Telegram."""
        return self.signal_history
    
    async def _do_stop(self):
        """Stop the bot from Telegram command."""
        self.running = False
    
    def _do_pause(self):
        """Pause trading from Telegram command."""
        self.paused = True
        logger.info("Trading paused via Telegram")
    
    def _do_resume(self):
        """Resume trading from Telegram command."""
        self.paused = False
        logger.info("Trading resumed via Telegram")

    def _get_trading_mode(self) -> Dict[str, Any]:
        """Get current trading mode info for Telegram /mode command."""
        return {
            'mode': 'live' if self.live_mode else 'paper',
            'live_mode': self.live_mode,
            'paper_mode': self.paper_mode,
            'dry_run_mode': self.dry_run_mode,
            'balance': self.balance,
            'initial_balance': self.initial_balance,
            'total_trades': self.stats.total_trades,
            'total_pnl': self.stats.total_pnl,
            'today_pnl': self.stats.today_pnl
        }
    
    def _set_allowed_sides(self, sides: str) -> bool:
        """Set allowed trading sides (long/short/both) via Telegram command.
        
        Args:
            sides: 'long', 'short', or 'both'
            
        Returns:
            True if successful
        """
        sides = sides.lower()
        if sides not in ['long', 'short', 'both']:
            return False
        
        self.allowed_sides = sides
        self._save_trading_state()
        logger.info(f"📊 Allowed sides changed to: {sides.upper()}")
        return True
    
    
    async def _switch_trading_mode(self, target_mode: str) -> Dict[str, Any]:
        """Switch between live and paper trading modes.
        
        Args:
            target_mode: 'live' or 'paper'
            
        Returns:
            Dict with 'success' bool and optional 'error' message
        """
        try:
            import subprocess
            import sys
            
            if target_mode == 'paper':
                # === SWITCH TO PAPER MODE ===
                logger.info("🔄 SWITCHING TO PAPER TRADING MODE")
                
                # 1. Backup live config BEFORE resetting (only if currently in live mode)
                if self.live_mode or (self.CONFIG_FILE.exists()):
                    live_config_backup = Path(__file__).parent / "julaba_config_live_backup.json"
                    if self.CONFIG_FILE.exists():
                        with open(self.CONFIG_FILE, 'r') as f:
                            current_config = json.load(f)
                        # Only backup if it looks like live data (not paper $10k)
                        if current_config.get('trading_mode') != 'paper' or current_config.get('balance', 10000) != 10000:
                            with open(live_config_backup, 'w') as f:
                                json.dump(current_config, f, indent=2)
                            logger.info(f"💾 Backed up live config - Balance: ${current_config.get('balance', 0):,.2f}")
                
                # 2. Backup live trade history if it exists and has data
                live_backup_file = Path(__file__).parent / "trade_history_live_backup.json"
                if self.TRADE_HISTORY_FILE.exists():
                    with open(self.TRADE_HISTORY_FILE, 'r') as f:
                        existing_trades = json.load(f)
                    if existing_trades and len(existing_trades) > 0:
                        with open(live_backup_file, 'w') as f:
                            json.dump(existing_trades, f, indent=2)
                        logger.info(f"💾 Backed up {len(existing_trades)} live trades to {live_backup_file}")
                
                # 3. Restore paper trade history from backup (if exists)
                paper_backup_file = Path(__file__).parent / "trade_history_paper_backup.json"
                if paper_backup_file.exists():
                    with open(paper_backup_file, 'r') as f:
                        paper_trades = json.load(f)
                    with open(self.TRADE_HISTORY_FILE, 'w') as f:
                        json.dump(paper_trades, f, indent=2)
                    logger.info(f"📥 Restored {len(paper_trades)} paper trades from backup")
                else:
                    # No paper backup - start fresh
                    with open(self.TRADE_HISTORY_FILE, 'w') as f:
                        json.dump([], f)
                    logger.info("🗑️ No paper backup found - starting fresh")
                
                # 4. Clear AI decisions
                ai_file = Path(__file__).parent / "ai_decisions.json"
                with open(ai_file, 'w') as f:
                    json.dump([], f)
                logger.info("🗑️ Cleared AI decisions")
                
                # 5. Restore paper config from backup OR reset to defaults
                paper_config_backup = Path(__file__).parent / "julaba_config_paper_backup.json"
                if paper_config_backup.exists():
                    with open(paper_config_backup, 'r') as f:
                        paper_config = json.load(f)
                    paper_config['trading_mode'] = 'paper'
                    paper_config['last_updated'] = datetime.now(timezone.utc).isoformat()
                    logger.info(f"📥 Restored paper config - Balance: ${paper_config.get('balance', 10000):,.2f}")
                else:
                    # No paper config backup - use defaults
                    paper_config = {
                        'symbol': self.SYMBOL,
                        'balance': 10000.0,
                        'initial_balance': 10000.0,
                        'peak_balance': 10000.0,
                        'balance_protection': 1000,  # Paper mode protection
                        'consecutive_wins': 0,
                        'consecutive_losses': 0,
                        'stats': {
                            'total_trades': 0,
                            'winning_trades': 0,
                            'losing_trades': 0,
                            'total_pnl': 0.0,
                            'today_pnl': 0.0,
                            'max_win': 0.0,
                            'max_loss': 0.0
                        },
                        'open_position': None,
                        'multi_positions': {},
                        'position_slots': {},
                        'prefilter_stats': {},
                        'equity_curve': [],
                        'trading_mode': 'paper',
                        'last_updated': datetime.now(timezone.utc).isoformat()
                    }
                    logger.info("💾 No paper config backup - using defaults ($10,000)")
                
                with open(self.CONFIG_FILE, 'w') as f:
                    json.dump(paper_config, f, indent=2)
                
                # 5. Save state and restart without --live flag
                self._save_trading_state()
                
                # 6. Restart the bot without --live flag
                logger.info("🔄 Restarting bot in PAPER mode...")
                
                # Build restart command (without --live)
                restart_cmd = [
                    sys.executable, 'bot.py',
                    '--dashboard', f'--dashboard-port={self.dashboard.port if self.dashboard else 5000}'
                ]
                
                # Schedule restart
                asyncio.create_task(self._delayed_restart(restart_cmd, 'paper'))
                
                return {'success': True, 'message': 'Switching to paper mode...'}
                
            elif target_mode == 'live':
                # === SWITCH TO LIVE MODE ===
                logger.info("🔄 SWITCHING TO LIVE TRADING MODE")
                
                # 1. Backup paper config BEFORE switching
                paper_config_backup = Path(__file__).parent / "julaba_config_paper_backup.json"
                if self.CONFIG_FILE.exists():
                    with open(self.CONFIG_FILE, 'r') as f:
                        current_config = json.load(f)
                    # Only backup if it looks like paper data
                    if current_config.get('trading_mode') == 'paper' or current_config.get('balance', 0) >= 1000:
                        with open(paper_config_backup, 'w') as f:
                            json.dump(current_config, f, indent=2)
                        logger.info(f"💾 Backed up paper config - Balance: ${current_config.get('balance', 0):,.2f}")
                
                # 2. Backup paper trade history
                paper_backup_file = Path(__file__).parent / "trade_history_paper_backup.json"
                if self.TRADE_HISTORY_FILE.exists():
                    with open(self.TRADE_HISTORY_FILE, 'r') as f:
                        paper_trades = json.load(f)
                    if paper_trades and len(paper_trades) > 0:
                        with open(paper_backup_file, 'w') as f:
                            json.dump(paper_trades, f, indent=2)
                        logger.info(f"💾 Backed up {len(paper_trades)} paper trades")
                
                # 3. Restore live trade history from backup
                live_backup_file = Path(__file__).parent / "trade_history_live_backup.json"
                if live_backup_file.exists():
                    with open(live_backup_file, 'r') as f:
                        live_trades = json.load(f)
                    with open(self.TRADE_HISTORY_FILE, 'w') as f:
                        json.dump(live_trades, f, indent=2)
                    logger.info(f"📥 Restored {len(live_trades)} live trades from backup")
                else:
                    # No backup - start fresh
                    with open(self.TRADE_HISTORY_FILE, 'w') as f:
                        json.dump([], f)
                    logger.info("📝 No live backup found - starting fresh")
                
                # 4. Restore live config from backup
                live_config_backup = Path(__file__).parent / "julaba_config_live_backup.json"
                if live_config_backup.exists():
                    with open(live_config_backup, 'r') as f:
                        config = json.load(f)
                    logger.info(f"📥 Restored live config - Balance: ${config.get('balance', 0):,.2f}, Protection: ${config.get('balance_protection', 50):,.2f}")
                else:
                    # No config backup - use current config but mark as live
                    config = {}
                    if self.CONFIG_FILE.exists():
                        with open(self.CONFIG_FILE, 'r') as f:
                            config = json.load(f)
                    # Set reasonable defaults for live mode
                    if 'balance_protection' not in config:
                        config['balance_protection'] = 50  # Default $50 protection
                    logger.info("📝 No live config backup - using current values")
                
                # 4. Update config to mark as live mode
                config['trading_mode'] = 'live'
                config['last_updated'] = datetime.now(timezone.utc).isoformat()
                
                with open(self.CONFIG_FILE, 'w') as f:
                    json.dump(config, f, indent=2)
                
                # 5. Restart with --live flag
                logger.info("🔄 Restarting bot in LIVE mode...")
                
                restart_cmd = [
                    sys.executable, 'bot.py',
                    '--live',
                    '--dashboard', f'--dashboard-port={self.dashboard.port if self.dashboard else 5000}'
                ]
                
                # Schedule restart
                asyncio.create_task(self._delayed_restart(restart_cmd, 'live'))
                
                return {'success': True, 'message': 'Switching to live mode...'}
            
            else:
                return {'success': False, 'error': f'Invalid mode: {target_mode}'}
                
        except Exception as e:
            logger.error(f"Failed to switch trading mode: {e}")
            return {'success': False, 'error': str(e)}
    
    async def _delayed_restart(self, restart_cmd: list, mode: str):
        """Delay restart to allow Telegram message to be sent.
        
        Uses systemd restart if available, otherwise manual restart.
        """
        try:
            import subprocess
            import os
            
            await asyncio.sleep(2)  # Wait for Telegram message to send
            
            # Get the directory where bot.py is located
            bot_dir = Path(__file__).parent.resolve()
            
            logger.info(f"🔄 Initiating restart to {mode} mode...")
            
            # Stop current bot
            self.running = False
            
            # Give time for graceful shutdown
            await asyncio.sleep(1)
            
            # Remove lockfile
            lockfile_path = bot_dir / "julaba.lock"
            if lockfile_path.exists():
                lockfile_path.unlink()
                logger.info("🔓 Removed lockfile")
            
            # Check if running under systemd
            is_systemd = os.environ.get('INVOCATION_ID') or os.path.exists('/run/systemd/system')
            
            if is_systemd:
                # Use systemd restart - config mode will be read on startup
                logger.info(f"🔄 Systemd detected - exiting to trigger auto-restart in {mode} mode")
                logger.info(f"   Config trading_mode is set to '{mode}' - will be respected on restart")
                await asyncio.sleep(0.5)
                os._exit(0)  # Exit cleanly, systemd will restart
            else:
                # Manual restart (not under systemd)
                bot_script = bot_dir / "bot.py"
                fixed_cmd = [restart_cmd[0], str(bot_script)] + restart_cmd[2:]
                
                logger.info(f"🔄 Manual restart to {mode} mode: {' '.join(fixed_cmd)}")
                
                log_file = bot_dir / "julaba.log"
                
                process = subprocess.Popen(
                    fixed_cmd,
                    cwd=str(bot_dir),
                    stdout=open(str(log_file), 'w'),
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={**os.environ, 'PYTHONUNBUFFERED': '1'}
                )
                
                logger.info(f"✅ New bot process started in {mode} mode (PID: {process.pid})")
                await asyncio.sleep(0.5)
                os._exit(0)
            
        except Exception as e:
            logger.error(f"Failed to restart bot: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _get_intelligence(self) -> Dict[str, Any]:
        """Get intelligence summary for Telegram /intel command."""
        result = {
            'drawdown_mode': 'NORMAL',
            'drawdown_pct': 0.0,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'pattern': None,
            'regime': 'UNKNOWN',
            'tradeable': False,
            'ml_status': 'Not trained'
        }
        
        # Drawdown calculation
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - self.balance) / self.peak_balance * 100
            result['drawdown_pct'] = round(drawdown, 2)
            if drawdown >= 20:
                result['drawdown_mode'] = 'EMERGENCY'
            elif drawdown >= 10:
                result['drawdown_mode'] = 'CAUTIOUS'
            elif drawdown >= 5:
                result['drawdown_mode'] = 'REDUCED'
        
        # Market regime
        if len(self.bars_agg) >= 50:
            regime = get_regime_analysis(self.bars_agg)
            result['regime'] = regime.get('regime', 'UNKNOWN')
            result['tradeable'] = regime.get('tradeable', False)
            result['adx'] = regime.get('adx', 0)
            result['hurst'] = regime.get('hurst', 0.5)
            
            # Pattern detection
            pattern = detect_candlestick_patterns(self.bars_agg)
            if pattern.get('pattern'):
                result['pattern'] = pattern
        
        # ML status
        ml_stats = get_ml_classifier().get_stats()
        result['ml_samples'] = ml_stats.get('total_samples', 0)
        result['ml_trained'] = ml_stats.get('is_trained', False)
        if result['ml_trained']:
            result['ml_status'] = f"Trained ({ml_stats.get('total_samples', 0)} samples)"
        else:
            needed = ml_stats.get('samples_until_training', 50)
            result['ml_status'] = f"Learning ({needed} more needed)"
        
        return result
    
    def _get_ml_stats(self) -> Dict[str, Any]:
        """Get ML classifier stats - SINGLE SOURCE OF TRUTH for both Telegram and AI context."""
        raw = get_ml_classifier().get_stats()
        # Normalize to consistent key names used everywhere
        return {
            'total_samples': raw.get('total_samples', 0),
            'samples': raw.get('total_samples', 0),  # Alias for convenience
            'is_trained': raw.get('is_trained', False),
            'trained': raw.get('is_trained', False),  # Alias for convenience
            'samples_until_training': raw.get('samples_until_training', 50),
            'historical_win_rate': raw.get('historical_win_rate', 0),
            'wins': raw.get('wins', 0),
            'losses': raw.get('losses', 0),
            'top_features': raw.get('top_features', []),
            'model_version': raw.get('model_version', 'v2'),
            'num_features': raw.get('num_features', 22),
        }
    
    def _get_regime(self) -> Dict[str, Any]:
        """Get current market regime for Telegram /regime command."""
        result = {
            'regime': 'UNKNOWN',
            'adx': 0,
            'hurst': 0.5,
            'volatility': 'unknown',
            'volatility_ratio': 1.0,
            'trend_strength': 0.0,
            'confidence': 0.0,
            'tradeable': False,
            'ml_prediction': None,
            'ml_confidence': 0,
            'description': 'Insufficient data'
        }
        
        if len(self.bars_agg) < 50:
            result['description'] = f'Need more data ({len(self.bars_agg)}/50 bars)'
            return result
        
        # Get regime analysis - returns all calculated fields
        regime = get_regime_analysis(self.bars_agg)
        result['regime'] = regime.get('regime', 'UNKNOWN')
        result['adx'] = round(regime.get('adx', 0), 1)
        result['hurst'] = round(regime.get('hurst', 0.5), 3)
        result['tradeable'] = regime.get('tradeable', False)
        
        # Volatility - use from regime or calculate proxy if unknown
        volatility = regime.get('volatility', 'unknown')
        if volatility == 'unknown':
            # Use ADX as proxy for volatility when not enough data
            adx = regime.get('adx', 0)
            if adx > 40:
                volatility = 'high'
            elif adx > 25:
                volatility = 'normal'
            else:
                volatility = 'low'
        result['volatility'] = volatility
        result['volatility_ratio'] = round(regime.get('volatility_ratio', 1.0), 2)
        result['confidence'] = round(regime.get('confidence', 0) * 100, 1)
        
        # Calculate trend strength from ADX (0-100 normalized)
        adx_value = regime.get('adx', 0)
        # ADX ranges from 0-100, normalize to 0-1 and then to percentage
        result['trend_strength'] = round((adx_value / 100.0) * 100, 2)
        
        # ML prediction if trained
        ml = get_ml_classifier()
        if ml.is_trained:
            try:
                pred = ml.predict(self.bars_agg)  # Pass dataframe directly
                if pred:
                    result['ml_prediction'] = pred.get('regime')
                    result['ml_confidence'] = round(pred.get('confidence', 0) * 100, 1)
            except Exception as e:
                logger.debug(f"ML prediction error: {e}")
        
        # Description
        regime_desc = {
            'STRONG_TRENDING': 'Strong directional move - trend following works well',
            'TRENDING': 'Clear trend - good for momentum strategies',
            'WEAK_TRENDING': 'Weak trend - caution advised',
            'RANGING': 'Sideways market - mean reversion may work',
            'CHOPPY': 'Choppy/noisy - avoid trading'
        }
        result['description'] = regime_desc.get(result['regime'], 'Unknown market condition')
        
        return result

    def _get_indicators_for_dashboard(self) -> Dict[str, Any]:
        """Get current technical indicator values for dashboard."""
        result = {
            'rsi': None,
            'macd_signal': '--',
            'adx': None,
            'atr': None,
            'bb_position': '--',
            'volume_ratio': None
        }
        
        if len(self.bars_agg) < 20:
            return result
        
        try:
            from indicator import calculate_rsi, calculate_atr, calculate_adx
            
            close = self.bars_agg['close']
            
            # RSI
            rsi = calculate_rsi(close, 14)
            if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]):
                result['rsi'] = float(rsi.iloc[-1])
            
            # ATR
            atr = calculate_atr(self.bars_agg, 14)
            if len(atr) > 0 and not pd.isna(atr.iloc[-1]):
                result['atr'] = float(atr.iloc[-1])
            
            # ADX
            result['adx'] = calculate_adx(self.bars_agg, 14)
            
            # MACD Signal
            if len(self.bars_agg) >= 26:
                ema12 = close.ewm(span=12).mean()
                ema26 = close.ewm(span=26).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9).mean()
                if len(macd) > 0:
                    if macd.iloc[-1] > signal.iloc[-1]:
                        result['macd_signal'] = 'BULLISH' if macd.iloc[-1] > 0 else 'WEAK BULL'
                    else:
                        result['macd_signal'] = 'BEARISH' if macd.iloc[-1] < 0 else 'WEAK BEAR'
            
            # Bollinger Bands position
            if len(close) >= 20:
                sma20 = close.rolling(20).mean()
                std20 = close.rolling(20).std()
                upper = sma20 + 2 * std20
                lower = sma20 - 2 * std20
                current = close.iloc[-1]
                if current >= upper.iloc[-1]:
                    result['bb_position'] = 'ABOVE (OB)'
                elif current <= lower.iloc[-1]:
                    result['bb_position'] = 'BELOW (OS)'
                else:
                    pct = (current - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) * 100
                    result['bb_position'] = f'{pct:.0f}%'
            
            # Volume ratio
            if 'volume' in self.bars_agg.columns and len(self.bars_agg) >= 20:
                vol = self.bars_agg['volume']
                avg_vol = vol.rolling(20).mean().iloc[-1]
                if avg_vol > 0:
                    result['volume_ratio'] = float(vol.iloc[-1] / avg_vol)
        except Exception as e:
            logger.debug(f"Error getting indicators for dashboard: {e}")
        
        return result

    def _get_current_signal(self) -> Dict[str, Any]:
        """Get current signal state for dashboard."""
        result = {
            'direction': None,
            'confidence': 0,
            'entry': None,
            'stop_loss': None,
            'take_profit': None
        }
        
        # If we have recent signal history, return the latest
        if self.signal_history:
            latest = self.signal_history[-1]
            result['direction'] = latest.get('direction')
            result['confidence'] = latest.get('confidence', 0)
            result['entry'] = latest.get('entry')
            result['stop_loss'] = latest.get('sl')
            result['take_profit'] = latest.get('tp')
        
        return result

    def _get_ohlc_for_chart(self, timeframe: str = '1m', date_range: str = '1d', date_from: str = None, date_to: str = None, symbol: str = None) -> List[Dict]:
        """Get OHLC data for live price chart on dashboard with date range support.
        
        Args:
            symbol: Optional symbol override (for Position 2 chart). If None, uses primary symbol.
        """
        result = []
        
        try:
            import datetime
            
            # Use provided symbol or default to primary
            chart_symbol = symbol if symbol else self.SYMBOL
            logger.info(f"OHLC fetch START: symbol={chart_symbol}, tf={timeframe}, range={date_range}")
            
            # Handle direct timestamps (from/to in milliseconds)
            start_time = None
            end_time = None
            use_custom = False
            
            if date_from and date_to:
                try:
                    start_time = int(float(date_from))
                    end_time = int(float(date_to))
                    use_custom = True
                except (ValueError, TypeError):
                    pass  # Non-critical: invalid date params, use defaults
            
            # Calculate how many candles we need based on range
            range_map = {
                '1h': 1/24, '6h': 0.25, '24h': 1, '1d': 1, '3d': 3, '7d': 7, '30d': 30
            }
            days = range_map.get(date_range, 1)
            
            # Calculate candle counts based on timeframe
            tf_minutes = {'1m': 1, '3m': 3, '5m': 5, '15m': 15, '1h': 60}
            minutes = tf_minutes.get(timeframe, 1)
            
            if use_custom and start_time and end_time:
                # Calculate candles for custom range
                delta_ms = end_time - start_time
                # Limit based on standard API limits (max 1000 per request, we'll paginate up to 50000)
                candles_needed = min(50000, delta_ms // (minutes * 60 * 1000))
            else:
                candles_needed = min(50000, int((days * 24 * 60) // minutes))
            
            candles_needed = max(50, candles_needed)  # Minimum 50 candles
            
            # Fetch historical data from Bybit API
            # Using Bybit for chart data since we're trading on Bybit
            try:
                # Map timeframe to Bybit interval
                tf_map = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '1h': '60'}
                interval = tf_map.get(timeframe, '1')
                
                # Calculate time range if not custom
                if not use_custom:
                    import time
                    end_time = int(time.time() * 1000)
                    start_time = end_time - int(days * 24 * 60 * 60 * 1000)
                
                # Use Bybit API for historical klines
                # Convert symbol format to simple Bybit format: SUI/USDT:USDT -> SUIUSDT
                bybit_symbol = chart_symbol.replace('/USDT:USDT', 'USDT').replace(':USDT', '').replace('_', '').replace('/', '')
                logger.info(f"Fetching Bybit klines: {bybit_symbol}, interval={interval}, start={start_time}, end={end_time}")
                
                # Paginate if we need more than 1000 candles
                all_klines = []
                current_start = start_time
                max_retries = 3
                
                import requests as http_req
                
                # Loop until we cover the time range or reach the end
                # We stop when current_start exceeds end_time, NOT when we hit 'candels_needed' count
                # This ensures we cover the FULL time range (dates) requested by user
                while current_start < end_time:
                    # Calculate reasonable limit per request (Bybit max is 1000)
                    # For standard ranges, limit to what we actually need
                    remaining_needed = candles_needed - len(all_klines)
                    if remaining_needed <= 0:
                        break
                    limit_per_req = min(1000, max(100, remaining_needed + 5))  # Small buffer
                    
                    try:
                        # Use Bybit API for linear perpetual klines
                        response = http_req.get('https://api.bybit.com/v5/market/kline', params={
                            'category': 'linear',
                            'symbol': bybit_symbol,
                            'interval': interval,
                            'start': current_start,
                            'end': end_time,
                            'limit': limit_per_req
                        }, timeout=10)
                        
                        if response.status_code != 200:
                            logger.error(f"Bybit API error: {response.text}")
                            break
                        
                        data = response.json()
                        if data.get('retCode') != 0:
                            logger.error(f"Bybit API error: {data.get('retMsg')}")
                            break
                        
                        # Bybit returns klines in reverse order (newest first)
                        klines_raw = data.get('result', {}).get('list', [])
                        # Convert to standard format [time, open, high, low, close, volume]
                        klines = [[int(k[0]), k[1], k[2], k[3], k[4], k[5]] for k in klines_raw]
                        klines.reverse()  # Oldest first
                        if not klines:
                            break
                        
                        all_klines.extend(klines)
                        
                        # Move start to after last candle
                        last_time = int(klines[-1][0])
                        next_start = last_time + 1 # +1ms to avoid overlap
                        
                        # If we didn't advance, break to avoid infinite loop
                        if next_start <= current_start:
                            break
                            
                        current_start = next_start
                        
                        # Safety break if we have too much data
                        if len(all_klines) > 50000:
                            break
                            
                    except Exception as e:
                        logger.error(f"Error fetching klines batch: {e}")
                        break
                    
                    # Safety: limit to 50000 candles max to avoid too many API calls
                    if len(all_klines) >= 50000:
                        break
                
                logger.info(f"OHLC: Fetched {len(all_klines)} klines from Bybit")
                for k in all_klines:
                    candle = {
                        't': int(k[0]),      # Open time
                        'o': float(k[1]),    # Open
                        'h': float(k[2]),    # High
                        'l': float(k[3]),    # Low
                        'c': float(k[4]),    # Close
                        'v': float(k[5])     # Volume
                    }
                    result.append(candle)
                if result:
                    return result
            except Exception as e:
                logger.error(f"Failed to fetch historical klines from Bybit: {e}")
                import traceback
                logger.error(traceback.format_exc())
                # If a specific symbol was requested, DON'T fall through to cached data
                # because cached data is for self.SYMBOL which rotates during scanning
                if symbol:
                    return result  # Return empty rather than wrong symbol's data
                # Fall through to cached data only if no symbol specified
            
            # For 1-day range, use cached data (original logic)
            # Handle 1m bars (stored as list of dicts)
            if timeframe == '1m':
                if hasattr(self, 'bars_1m') and len(self.bars_1m) > 0:
                    # bars_1m is a list of dicts
                    data = list(self.bars_1m)[-candles_needed:]
                    for bar in data:
                        ts = bar.get('timestamp', 0)
                        # Ensure timestamp is in milliseconds
                        if isinstance(ts, (int, float)) and ts < 10000000000:
                            ts = int(ts * 1000)
                        candle = {
                            't': int(ts),
                            'o': float(bar.get('open', 0)),
                            'h': float(bar.get('high', 0)),
                            'l': float(bar.get('low', 0)),
                            'c': float(bar.get('close', 0)),
                            'v': float(bar.get('volume', 0))
                        }
                        result.append(candle)
                    return result
            
            # Handle DataFrames (3m, 15m, 1h)
            if timeframe == '3m':
                df = self.bars_agg
            elif timeframe == '15m':
                df = self.bars_15m if hasattr(self, 'bars_15m') and isinstance(self.bars_15m, pd.DataFrame) and len(self.bars_15m) > 0 else None
            elif timeframe == '5m':
                # Resample 1m to 5m
                if hasattr(self, 'bars_1m') and len(self.bars_1m) >= 5:
                    df_1m = pd.DataFrame(list(self.bars_1m))
                    df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], unit='ms')
                    df_1m.set_index('timestamp', inplace=True)
                    df = df_1m.resample('5min', label='left').agg({
                        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                    }).dropna().reset_index()
                else:
                    df = None
            else:
                df = self.bars_agg
            
            if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
                return result
            
            # Get last N candles
            df_slice = df.tail(candles_needed)
            
            for _, row in df_slice.iterrows():
                # Get timestamp from 'timestamp' column
                ts = row.get('timestamp', row.get('Timestamp', None))
                if ts is None:
                    continue
                
                # Convert to milliseconds
                if hasattr(ts, 'timestamp'):
                    ts = int(ts.timestamp() * 1000)
                elif isinstance(ts, (int, float)):
                    if ts < 10000000000:  # seconds, not ms
                        ts = int(ts * 1000)
                    else:
                        ts = int(ts)
                else:
                    ts = int(pd.Timestamp(ts).timestamp() * 1000)
                
                candle = {
                    't': ts,
                    'o': float(row.get('open', row.get('Open', 0))),
                    'h': float(row.get('high', row.get('High', 0))),
                    'l': float(row.get('low', row.get('Low', 0))),
                    'c': float(row.get('close', row.get('Close', 0))),
                    'v': float(row.get('volume', row.get('Volume', 0)))
                }
                result.append(candle)
        except Exception as e:
            logger.error(f"Error getting OHLC for chart: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        return result

    def _set_ai_mode(self, mode: str) -> bool:
        """Set AI trading mode."""
        valid_modes = ["filter", "advisory", "autonomous", "hybrid"]
        if mode.lower() in valid_modes:
            self.ai_mode = mode.lower()
            logger.info(f"AI mode changed to: {self.ai_mode}")
            # Save to config for persistence across restarts
            self._save_trading_state()
            return True
        return False
    
    async def _confirm_ai_trade(self):
        """Confirm pending AI trade (for advisory/hybrid mode)."""
        if self.pending_ai_trade and self.position is None:
            trade = self.pending_ai_trade
            self.pending_ai_trade = None
            
            # Use the SCANNED symbol/price/atr stored when the advisory was created
            # Without this, _open_position uses self.SYMBOL which may be a different
            # pair, causing completely wrong position sizing (e.g. BTC sizing on ELSA = 1 contract)
            trade_symbol = trade.get('_scan_symbol', self.SYMBOL)
            trade_price = trade.get('_scan_price', self.cached_last_price)
            trade_atr = trade.get('_scan_atr', self.cached_last_atr)
            
            logger.info(f"AI trade CONFIRMED by user: {trade['action']} on {normalize_symbol(trade_symbol)}")
            
            # Temporarily switch to the scanned symbol (like autonomous mode does)
            original_symbol = self.SYMBOL
            if trade_symbol != self.SYMBOL:
                self.SYMBOL = trade_symbol
                logger.info(f"🔄 Advisory: switching symbol to {trade_symbol} for this trade")
            
            try:
                await self._open_position(
                    trade['signal'],
                    trade_price,
                    trade_atr,
                    source="ai_confirmed"
                )
            finally:
                # Always restore original symbol
                self.SYMBOL = original_symbol
            
            return True
        return False
    
    async def _reject_ai_trade(self):
        """Reject pending AI trade."""
        if self.pending_ai_trade:
            logger.info(f"AI trade REJECTED by user: {self.pending_ai_trade['action']}")
            self.pending_ai_trade = None
            return True
        return False
    
    async def _execute_pending_trade(self, trade: Dict[str, Any]) -> bool:
        """
        Execute a pending trade that was awaiting user confirmation.
        Called when user confirms a below-minimum size trade via Telegram.
        
        Args:
            trade: Dict with trade details (symbol, side, size, price, stop_loss, tp1, tp2, tp3)
        
        Returns:
            True if trade executed successfully, False otherwise
        """
        if not trade:
            logger.warning("No trade details provided for execution")
            return False
        
        try:
            symbol = trade.get('symbol', self.SYMBOL)
            side = trade.get('side', 'long')
            size = trade.get('size', 0)
            price = trade.get('price', self.cached_last_price)
            stop_loss = trade.get('stop_loss', 0)
            tp1 = trade.get('tp1', 0)
            tp2 = trade.get('tp2', 0)
            tp3 = trade.get('tp3', 0)
            
            logger.info(f"📱 Executing user-confirmed trade: {side.upper()} {size:.6f} {symbol} @ ${price:.4f}")
            
            if not self.live_mode:
                logger.info("[PAPER] Would execute confirmed trade")
                return True
            
            # Execute market order
            futures_symbol = symbol if '/' in symbol else f"{symbol.replace('USDT', '')}/USDT:USDT"
            order_side = 'buy' if side.lower() == 'long' else 'sell'
            
            order = await self._execute_market_order(futures_symbol, order_side, size)
            
            if not order:
                logger.error(f"🚨 Confirmed trade order failed for {symbol}")
                return False
            
            # Get fill price
            fill_price = float(order.get('average') or order.get('price') or price)
            
            # Create position
            signal = 1 if side.lower() == 'long' else -1
            pos = Position(
                symbol=futures_symbol,
                side=side.lower(),
                entry_price=fill_price,
                size=size,
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                tp1_pct=self.TP1_PCT,
                tp2_pct=self.TP2_PCT,
                tp3_pct=self.TP3_PCT
            )
            
            # Store position
            symbol_key = normalize_symbol(futures_symbol)
            self.positions[symbol_key] = pos
            self.position = pos  # Legacy single position
            
            # Place server-side SL/TP
            if self.live_mode:
                await self._place_server_side_sl_tp(
                    symbol=futures_symbol,
                    side=pos.side,
                    size=pos.size,
                    entry_price=pos.entry_price,
                    stop_loss=pos.stop_loss,
                    tp1=pos.tp1,
                    tp2=pos.tp2
                )
            
            self._save_trading_state()
            
            logger.info(f"✅ User-confirmed trade executed: {side.upper()} {size:.6f} {symbol} @ ${fill_price:.4f}")
            
            # Clear pending trade
            self._pending_trade = None
            
            return True
            
        except Exception as e:
            logger.error(f"🚨 Failed to execute pending trade: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    # ============== NEW ENHANCEMENT METHODS ==============
    
    def _get_risk_stats(self) -> Dict[str, Any]:
        """Get risk manager statistics for Telegram."""
        stats = self.risk_manager.get_stats()
        can_trade = self.risk_manager.can_trade(self.balance, self.initial_balance)
        risk_info = self.risk_manager.get_adjusted_risk(
            self.balance, 
            self.peak_balance,
            self.cached_last_atr / self.cached_last_price * 100 if self.cached_last_price > 0 else 1.0
        )
        return {
            **stats,
            'can_trade': can_trade['allowed'],
            'can_trade_reason': can_trade['reason'],
            'adjusted_risk': risk_info['adjusted_risk'],
            'dd_mode': risk_info['dd_mode'],
            'kelly_risk': risk_info['kelly_risk']
        }
    
    def _get_shield_status(self) -> Dict[str, Any]:
        """Get Risk Shield status for Telegram /shield command."""
        return self.risk_shield.get_shield_status(
            balance=self.balance,
            initial_balance=self.initial_balance,
            peak_balance=self.peak_balance
        )
    
    def _reset_shield(self) -> str:
        """Reset Risk Shield hard breach — called from /shield reset."""
        return self.risk_shield.reset_hard_breach()
    
    def _toggle_dry_run(self, enable: bool) -> str:
        """Toggle dry-run mode on/off."""
        self.dry_run_mode = enable
        if enable:
            logger.info("📝 Dry-run mode ENABLED — trades will be shadow-tracked only")
            return "📝 Dry-run mode *ENABLED*\n\nAll signals will be shadow-tracked but NOT executed.\nUse `/dryrun status` to check shadow performance."
        else:
            logger.info("✅ Dry-run mode DISABLED — trades will execute normally")
            summary = self.dry_run_tracker.get_summary()
            return (f"✅ Dry-run mode *DISABLED*\n\nTrades will execute normally again.\n\n"
                    f"Shadow session results:\n"
                    f"• Trades: {summary['total_trades']} ({summary['wins']}W/{summary['losses']}L)\n"
                    f"• Win rate: {summary['win_rate']}%\n"
                    f"• Shadow PnL: ${summary['total_pnl']:+.2f}")
    
    def _get_mtf_analysis(self) -> Dict[str, Any]:
        """Get multi-timeframe analysis for Telegram."""
        if len(self.bars_agg) < 20:
            return {'error': 'Insufficient data for MTF analysis'}
        
        # Get last signal
        try:
            signals_df = generate_signals(self.bars_agg)
            last_signal = int(signals_df.iloc[-1].get("Side", 0))
        except (KeyError, IndexError, ValueError) as e:
            logger.debug(f"Could not get MTF signal: {e}")
            last_signal = 0
        
        # Run MTF analysis
        result = self.mtf_analyzer.analyze(self.bars_agg, proposed_signal=last_signal)
        return result
    
    async def _run_backtest(self, days: int = 7) -> Dict[str, Any]:
        """Run backtest on recent historical data."""
        from backtest import BacktestEngine, fetch_historical_data
        
        try:
            if not self.exchange:
                await self.connect()
            
            # Fetch historical data
            df = await fetch_historical_data(
                self.exchange,
                self.SYMBOL,
                timeframe='3m',
                days=days
            )
            
            if len(df) < 100:
                return {'error': f'Insufficient data: only {len(df)} bars'}
            
            # Run backtest
            engine = BacktestEngine(
                initial_balance=10000,
                risk_pct=self.RISK_PCT,
                atr_mult=self.ATR_MULT,
                tp1_r=self.TP1_R,
                tp2_r=self.TP2_R,
                tp3_r=self.TP3_R
            )
            
            result = engine.run(df)
            
            # Run Monte Carlo
            mc_results = engine.monte_carlo(result, simulations=500)
            
            return {
                'success': True,
                'days': days,
                'bars': len(df),
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'total_pnl': result.total_pnl,
                'total_pnl_pct': result.total_pnl_pct,
                'max_drawdown_pct': result.max_drawdown_pct,
                'sharpe_ratio': result.sharpe_ratio,
                'profit_factor': result.profit_factor,
                'monte_carlo': mc_results,
                'formatted': engine.format_results(result)
            }
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return {'error': str(e)}
    
    def _get_chart(self) -> Optional[bytes]:
        """Generate a chart of recent price action."""
        if len(self.bars_agg) < 20:
            return None
        
        entry_price = None
        entry_side = None
        stop_loss = None
        take_profits = None
        
        if self.position:
            entry_price = self.position.entry_price
            entry_side = self.position.side
            stop_loss = self.position.stop_loss
            take_profits = [self.position.tp1, self.position.tp2, self.position.tp3]
        
        return self.chart_generator.generate_candlestick_chart(
            self.bars_agg,
            symbol=self.SYMBOL,
            entry_price=entry_price,
            entry_side=entry_side,
            stop_loss=stop_loss,
            take_profits=take_profits
        )
    
    async def _fetch_higher_timeframes(self):
        """Fetch 15m and 1H data for MTF analysis."""
        if not self.exchange:
            return
        
        now = datetime.now(timezone.utc)
        
        # Only update every 15 minutes
        if self.last_htf_update and (now - self.last_htf_update).total_seconds() < 900:
            return
        
        try:
            # Fetch 15m bars
            ohlcv_15m = await self.exchange.fetch_ohlcv(self.SYMBOL, '15m', limit=100)
            if ohlcv_15m:
                self.bars_15m = pd.DataFrame(ohlcv_15m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                self.mtf_analyzer.update_data(df_15m=self.bars_15m)
            
            # Fetch 1H bars
            ohlcv_1h = await self.exchange.fetch_ohlcv(self.SYMBOL, '1h', limit=50)
            if ohlcv_1h:
                self.bars_1h = pd.DataFrame(ohlcv_1h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                self.mtf_analyzer.update_data(df_1h=self.bars_1h)
            
            self.last_htf_update = now
            logger.debug(f"Updated HTF data: 15m={len(self.bars_15m)} bars, 1H={len(self.bars_1h)} bars")
            
        except Exception as e:
            logger.warning(f"Failed to fetch HTF data: {e}")
    
    def _update_equity_curve(self):
        """Update equity curve after balance changes."""
        self.equity_curve.append(self.balance)
        # Keep last 1000 points
        if len(self.equity_curve) > 1000:
            self.equity_curve = self.equity_curve[-1000:]

    async def connect(self):
        """Connect to the exchange (Bybit USDT Perpetual Futures)."""
        api_key = os.getenv("BYBIT_API_KEY", "") or os.getenv("API_KEY", "")
        api_secret = os.getenv("BYBIT_API_SECRET", "") or os.getenv("API_SECRET", "")
        
        config = {
            "enableRateLimit": True,
            "timeout": 30000,  # 30 second timeout for API calls
            "options": {
                "defaultType": "swap",  # Bybit USDT Perpetual (long AND short)
                "defaultSubType": "linear",  # USDT-margined
                "recvWindow": 20000,  # Increase receive window to 20 seconds for timestamp tolerance
            }
        }
        
        # Live mode requires API keys
        if self.live_mode:
            if not api_key or not api_secret:
                raise ValueError("🚨 LIVE MODE requires BYBIT_API_KEY and BYBIT_API_SECRET in .env!")
            config["apiKey"] = api_key
            config["secret"] = api_secret
            logger.warning("⚠️ LIVE TRADING MODE - REAL MONEY ORDERS WILL BE EXECUTED!")
        elif api_key and api_secret and not self.paper_mode:
            config["apiKey"] = api_key
            config["secret"] = api_secret
            logger.info("Using authenticated connection (paper mode)")
        else:
            logger.info("Using public connection (paper trading)")
        
        self.exchange = ccxt.bybit(config)
        
        # Load markets with timeout handling
        try:
            await asyncio.wait_for(self.exchange.load_markets(), timeout=60)
        except asyncio.TimeoutError:
            logger.warning("⚠️ Market loading timed out - retrying...")
            await self.exchange.load_markets()
        
        logger.info(f"Connected to Bybit | Symbol: {self.SYMBOL}")
        
        # Fetch actual balance if in live mode
        if self.live_mode:
            await self._sync_balance_with_exchange()
            
            # === CRITICAL: Recalibrate peak_balance after exchange sync ===
            # If peak_balance is from a previous era (much higher than actual balance),
            # reset it to current balance to prevent false max_drawdown triggers.
            # This handles: account recharges, fresh starts, withdrawals.
            if self.peak_balance > self.balance * 1.5 and self.balance > 0:
                old_peak = self.peak_balance
                self.peak_balance = self.balance
                self.max_drawdown_triggered = False
                logger.info(f"📈 PEAK RECALIBRATED: ${old_peak:.2f} → ${self.peak_balance:.2f} (stale peak detected)")
            
            # Also recalibrate initial_balance to actual exchange balance on live start
            if abs(self.initial_balance - self.balance) > self.balance * 0.5 and self.balance > 0:
                old_init = self.initial_balance
                self.initial_balance = self.balance
                logger.info(f"💰 INITIAL BALANCE RECALIBRATED: ${old_init:.2f} → ${self.initial_balance:.2f}")
            
            # Save recalibrated values
            self._save_trading_state()
    
    async def _sync_balance_with_exchange(self):
        """Fetch actual total portfolio value from exchange (USDT + open positions)."""
        try:
            balance = await self.exchange.fetch_balance()
            usdt_free = float(balance.get('USDT', {}).get('free', 0))
            usdt_total = float(balance.get('USDT', {}).get('total', 0))
            
            # Also sync UTA wallet info for borrowed funds tracking
            try:
                wallet_info = await self.exchange.private_get_v5_account_wallet_balance({'accountType': 'UNIFIED'})
                wallet_data = wallet_info.get('result', {}).get('list', [{}])[0]
                
                borrowed_amount = 0
                for coin in wallet_data.get('coin', []):
                    if coin.get('coin') == 'USDT':
                        borrowed_amount = float(coin.get('borrowAmount', 0) or 0)
                        break
                
                self._cached_uta_info = {
                    'wallet_balance': float(wallet_data.get('totalWalletBalance', 0)),
                    'equity': float(wallet_data.get('totalEquity', 0)),
                    'borrowed': borrowed_amount,
                    'available_margin': float(wallet_data.get('totalAvailableBalance', 0)),
                    'margin_used': float(wallet_data.get('totalInitialMargin', 0)),
                    'ltv': float(wallet_data.get('accountLTV', 0)) * 100,
                }
                
                if borrowed_amount > 0:
                    logger.info(f"💳 UTA: Borrowed ${borrowed_amount:.2f}, Available margin: ${self._cached_uta_info['available_margin']:.2f}, LTV: {self._cached_uta_info['ltv']:.1f}%")
            except Exception as uta_err:
                logger.debug(f"UTA wallet sync: {uta_err}")
            
            # Calculate total portfolio value (USDT + value of held coins)
            total_portfolio_value = usdt_total
            held_assets = []
            
            # Check for any non-USDT assets with value
            for asset, amounts in balance.items():
                if asset in ['USDT', 'info', 'timestamp', 'datetime', 'free', 'used', 'total']:
                    continue
                if isinstance(amounts, dict):
                    asset_total = float(amounts.get('total', 0))
                    if asset_total > 0.0001:  # Skip dust
                        # Try to get current price for this asset
                        try:
                            ticker = await self.exchange.fetch_ticker(f"{asset}/USDT")
                            current_price = ticker.get('last', 0)
                            if current_price > 0:
                                asset_value = asset_total * current_price
                                total_portfolio_value += asset_value
                                held_assets.append(f"{asset}:{asset_total:.4f}=${asset_value:.2f}")
                        except Exception:
                            pass  # Can't price this asset, skip it
            
            if total_portfolio_value > 0:
                old_balance = self.balance
                self.balance = total_portfolio_value
                if self.initial_balance == 10000.0:  # Default value, update it
                    self.initial_balance = total_portfolio_value
                
                # === DEPOSIT DETECTION ===
                # If balance increased significantly (>5% or >$10), likely a deposit
                # FIX: Cross-check against recent trade PnL — big wins are NOT deposits
                balance_increase = total_portfolio_value - old_balance
                _recent_trade_pnl = 0.0
                if hasattr(self, 'trade_history') and self.trade_history:
                    _last_trade = self.trade_history[-1]
                    _last_trade_time = _last_trade.get('closed_at', '')
                    try:
                        _lt = datetime.fromisoformat(_last_trade_time.replace('Z', '+00:00'))
                        if (datetime.now(timezone.utc) - _lt).total_seconds() < 300:  # Last 5 min
                            _recent_trade_pnl = abs(_last_trade.get('pnl', 0.0))
                    except Exception:
                        pass
                # Only flag as deposit if increase exceeds recent trade PnL by significant margin
                _net_increase = balance_increase - _recent_trade_pnl
                if _net_increase > 10 and _net_increase > old_balance * 0.05:
                    logger.info(f"💵 DEPOSIT DETECTED: +${_net_increase:.2f} (${old_balance:.2f} → ${total_portfolio_value:.2f}, trade PnL: ${_recent_trade_pnl:.2f})")
                    # Update peak AND initial to new balance (fresh capital base)
                    old_peak = self.peak_balance
                    old_init = self.initial_balance
                    self.peak_balance = total_portfolio_value
                    self.initial_balance = total_portfolio_value
                    # Clear drawdown trigger since we have fresh capital
                    if self.max_drawdown_triggered:
                        self.max_drawdown_triggered = False
                        logger.info(f"🔓 Drawdown trigger cleared - new deposit resets peak")
                    # Save new values
                    self._save_trading_state()
                    logger.info(f"📈 Peak balance updated: ${old_peak:.2f} → ${self.peak_balance:.2f}")
                    logger.info(f"💰 Initial balance updated: ${old_init:.2f} → ${self.initial_balance:.2f}")
                    # Notify via Telegram
                    if self.telegram and self.telegram.enabled:
                        asyncio.create_task(self.telegram.send_message(
                            f"💵 *DEPOSIT DETECTED*\n\n"
                            f"Balance: `${old_balance:.2f}` → `${total_portfolio_value:.2f}`\n"
                            f"Deposit: `+${balance_increase:.2f}`\n"
                            f"Peak Updated: `${self.peak_balance:.2f}`\n\n"
                            f"_Ready to trade with new capital! 🚀_"
                        ))
                else:
                    # Normal peak tracking (profits)
                    self.peak_balance = max(self.peak_balance, total_portfolio_value)
                
                if held_assets:
                    logger.info(f"💰 Portfolio synced: ${total_portfolio_value:,.2f} (USDT: ${usdt_total:,.2f} + {', '.join(held_assets)})")
                else:
                    logger.info(f"💰 Exchange balance synced: ${usdt_total:,.2f} (free: ${usdt_free:,.2f})")
            else:
                logger.warning(f"⚠️ No balance found on exchange")
        except Exception as e:
            logger.error(f"Failed to sync balance: {e}")
    
    async def _sync_positions_with_bybit(self):
        """Sync internal position state with actual Bybit positions.
        
        This prevents issues where:
        1. Bot thinks it has a position but Bybit doesn't (ghost position - clean up)
        2. Bybit has positions the bot doesn't know about (import them to prevent over-opening)
        """
        if not self.live_mode:
            return
        
        # Use global normalize_symbol function
        
        try:
            # Fetch actual positions from Bybit
            bybit_positions = await self.exchange.fetch_positions()
            
            # Build dict of actual positions {normalized_symbol: {side, size, ...}}
            actual_positions = {}
            for bp in bybit_positions:
                contracts = float(bp.get('contracts', 0))
                if contracts != 0:
                    sym_normalized = normalize_symbol(bp['symbol'])
                    actual_positions[sym_normalized] = {
                        'side': bp['side'],
                        'size': abs(contracts),
                        'entry': float(bp['entryPrice']) if bp['entryPrice'] else 0,
                        'original_symbol': bp['symbol'],
                        'leverage': int(bp.get('leverage', 10)),
                        'mark_price': float(bp.get('markPrice', 0) or 0)  # Get mark price for PnL
                    }
                    
                    # === CRITICAL: Update _symbol_prices for real-time dashboard PnL ===
                    # This is the fallback when WebSocket fails
                    if not hasattr(self, '_symbol_prices'):
                        self._symbol_prices = {}
                    mark_price = float(bp.get('markPrice', 0) or 0)
                    pos_entry = float(bp.get('entryPrice', 0) or 0)
                    if mark_price > 0:
                        # Validate before updating dashboard cache
                        if pos_entry > 0:
                            dev_pct = abs(mark_price - pos_entry) / pos_entry * 100
                            if dev_pct <= 500:
                                self._symbol_prices[sym_normalized] = mark_price
                            else:
                                logger.warning(f"🚨 Position sync: markPrice ${mark_price:.6f} rejected ({dev_pct:.1f}% from entry ${pos_entry:.6f})")
                        else:
                            self._symbol_prices[sym_normalized] = mark_price
            
            # Only log when positions change (reduce log spam)
            if not hasattr(self, '_last_synced_positions'):
                self._last_synced_positions = set()
            current_positions = set(actual_positions.keys()) if actual_positions else set()
            if current_positions != self._last_synced_positions:
                logger.info(f"🔄 Position sync - Bybit positions: {list(actual_positions.keys()) if actual_positions else 'NONE'}")
                self._last_synced_positions = current_positions
            
            # Build set of normalized symbols the bot knows about
            bot_positions_normalized = set()
            for symbol, pos in self.positions.items():
                if pos is not None:
                    bot_positions_normalized.add(normalize_symbol(symbol))
            
            # === CHECK 1: Remove ghost positions (in bot but not on Bybit) - RECORD AS TRADES ===
            async with self._position_lock:
              for symbol, pos in list(self.positions.items()):
                if pos is not None:
                    sym_key = normalize_symbol(symbol)
                    logger.info(f"🔄 Position sync - checking bot position {symbol} (normalized: {sym_key})")
                    if sym_key not in actual_positions:
                        logger.warning(f"⚠️ POSITION SYNC: {symbol} exists in bot but NOT on Bybit - recording as externally closed")
                        
                        # Record the trade before removing
                        await self._record_externally_closed_trade(pos)
                        
                        self.positions[symbol] = None
                        self._release_position_slot(symbol)
                        if self.position and normalize_symbol(getattr(self.position, 'symbol', '')) == sym_key:
                            self.position = None
                        self._save_trading_state()
            
            # Also check primary position
            if self.position:
                sym_key = normalize_symbol(getattr(self.position, 'symbol', ''))
                if sym_key and sym_key not in actual_positions:
                    logger.warning(f"⚠️ POSITION SYNC: Primary position {sym_key} NOT on Bybit - recording as externally closed")
                    
                    # Record the trade before removing
                    await self._record_externally_closed_trade(self.position)
                    
                    self.position = None
                    self._save_trading_state()
            
            # === CHECK 2: Import unknown positions from Bybit (prevent over-opening) ===
            max_positions = self.multi_pair_config.max_total_positions if hasattr(self, 'multi_pair_config') else 2
            for sym_normalized, bybit_data in actual_positions.items():
                if sym_normalized not in bot_positions_normalized:
                    # === DUST POSITION CHECK: Skip tiny residual positions ===
                    # When close orders round down (e.g., 8.03 → 8.02), 0.01 remains
                    # These are not real positions - close them instead of importing
                    position_value = bybit_data.get('size', 0) * bybit_data.get('entry', 0)
                    if position_value < 1.0:  # Less than $1 = dust
                        logger.warning(f"🧹 DUST POSITION: {sym_normalized} size={bybit_data.get('size', 0):.6f} worth ${position_value:.2f} - closing instead of importing")
                        try:
                            order_side = 'buy' if bybit_data['side'] == 'short' else 'sell'
                            ccxt_sym = f"{sym_normalized.replace('USDT', '')}/USDT:USDT"
                            await self.exchange.create_market_order(ccxt_sym, order_side, bybit_data['size'], params={'reduceOnly': True})
                            logger.info(f"✅ DUST POSITION CLOSED: {sym_normalized}")
                        except Exception as dust_err:
                            logger.warning(f"⚠️ Could not close dust position {sym_normalized}: {dust_err}")
                        continue
                    
                    # Bybit has a position the bot doesn't know about!
                    logger.warning(f"⚠️ POSITION SYNC: Found unknown Bybit position {sym_normalized} ({bybit_data['side']}) - importing")
                    
                    # Create a Position object to track it
                    # Use normalized symbol format (ETHUSDT) for consistency
                    storage_symbol = sym_normalized  # Already normalized
                    try:
                        # FIX #9: Use Bybit's existing SL if set, otherwise use ATR-based dynamic SL
                        _import_entry = bybit_data['entry']
                        _import_side = bybit_data['side']
                        _bybit_sl = float(bybit_data.get('stopLoss', 0) or 0)
                        
                        if _bybit_sl > 0:
                            # Use Bybit's existing SL (user manually set it)
                            _import_sl = _bybit_sl
                            logger.info(f"📊 Using Bybit SL for import: ${_import_sl:.4f}")
                        else:
                            # Dynamic SL: Use MIN_STOP_PCT as floor (safer than hardcoded 3%)
                            _import_sl_pct = getattr(self, 'MIN_STOP_PCT', 0.015)  # ~1.5% default
                            if _import_side == 'short':
                                _import_sl = _import_entry * (1 + _import_sl_pct)
                            else:
                                _import_sl = _import_entry * (1 - _import_sl_pct)
                            logger.info(f"📊 Using dynamic SL for import: ${_import_sl:.4f} ({_import_sl_pct*100:.1f}%)")
                        
                        # Dynamic TP: Use reasonable defaults based on SL distance
                        _sl_dist_pct = abs(_import_entry - _import_sl) / _import_entry
                        _tp1_pct = _sl_dist_pct * 1.0   # 1R
                        _tp2_pct = _sl_dist_pct * 2.0   # 2R
                        _tp3_pct = _sl_dist_pct * 4.0   # 4R
                        
                        if _import_side == 'short':
                            _tp1 = _import_entry * (1 - _tp1_pct)
                            _tp2 = _import_entry * (1 - _tp2_pct)
                            _tp3 = _import_entry * (1 - _tp3_pct)
                        else:
                            _tp1 = _import_entry * (1 + _tp1_pct)
                            _tp2 = _import_entry * (1 + _tp2_pct)
                            _tp3 = _import_entry * (1 + _tp3_pct)
                        
                        imported_pos = Position(
                            symbol=storage_symbol,
                            side=_import_side,
                            entry_price=_import_entry,
                            size=bybit_data['size'],
                            stop_loss=_import_sl,
                            tp1=_tp1,
                            tp2=_tp2,
                            tp3=_tp3,
                            opened_at=datetime.now(timezone.utc)
                        )
                        async with self._position_lock:
                            self.positions[storage_symbol] = imported_pos
                            self._assign_position_slot(storage_symbol)
                        self._save_trading_state()
                        logger.info(f"✅ POSITION SYNC: Imported {bybit_data['side'].upper()} {storage_symbol} @ ${bybit_data['entry']:.4f}")
                        
                        # Set server-side SL/TP protection for imported position
                        await self._place_server_side_sl_tp(
                            symbol=storage_symbol,
                            side=bybit_data['side'],
                            size=bybit_data['size'],
                            entry_price=bybit_data['entry'],
                            stop_loss=imported_pos.stop_loss,
                            tp1=imported_pos.tp1,
                            tp2=imported_pos.tp2
                        )
                    except Exception as import_err:
                        logger.error(f"Failed to import position {sym_normalized}: {import_err}")
                    
        except Exception as e:
            logger.debug(f"Position sync check failed (non-critical): {e}")

    # =====================================================================
    # SERVER-SIDE SL/TP ORDER MANAGEMENT (Bybit Position Trading-Stop)
    # =====================================================================
    # CURRENT IMPLEMENTATION: Uses Bybit's position-level trading-stop API
    # - SL is server-side (executes even if bot offline)
    # - TP1 is server-side (single TP for full position)
    # - TP2/TP3 are client-side (managed by bot after TP1 hit)
    # 
    # LIMITATION: Bybit's tpslMode='Full' only supports ONE take profit price.
    # Multi-TP (partial closes) requires separate conditional orders which
    # have different behavior (order-based vs position-based).
    #
    # PhD-level optimal: TP1 at 1.0R (65% hit rate), TP2 at 1.5R, SL protected
    # =====================================================================
    # =====================================================================
    
    async def _place_server_side_sl_tp(self, symbol: str, side: str, size: float,
                                        entry_price: float, stop_loss: float, 
                                        tp1: float, tp2: float = None) -> Dict[str, str]:
        """Place server-side Stop Loss and Take Profit orders on Bybit.
        
        This ensures SL/TP execute even if the bot disconnects.
        Uses conditional orders (stop market for SL, limit for TP).
        
        Args:
            symbol: Trading pair (e.g., 'ETHUSDT' or 'ETH/USDT:USDT')
            side: Position side ('long' or 'short')
            size: Position size in base currency
            entry_price: Entry price for reference
            stop_loss: Stop loss trigger price
            tp1: First take profit price (50% of position)
            tp2: Second take profit price (30% of position), optional
            
        Returns:
            Dict with order IDs: {'sl_order_id': ..., 'tp1_order_id': ..., 'tp2_order_id': ...}
        """
        if not self.live_mode:
            logger.debug(f"[PAPER] Would place server-side SL/TP for {symbol}")
            return {'sl_order_id': 'paper', 'tp1_order_id': 'paper'}
        
        order_ids = {}
        
        try:
            # Normalize symbol for Bybit futures
            if ':USDT' not in symbol:
                symbol_clean = symbol.replace('/', '').replace('USDT', '')
                futures_symbol = f"{symbol_clean}/USDT:USDT"
            else:
                futures_symbol = symbol
            
            # For Bybit API, we need the raw symbol (e.g., "ETHUSDT" not "ETH/USDT:USDT")
            raw_symbol = futures_symbol.replace('/USDT:USDT', 'USDT').replace('/', '')
            
            logger.info(f"📋 Setting position SL/TP for {futures_symbol} ({side.upper()})")
            logger.info(f"   Entry: ${entry_price:.6f} | SL: ${stop_loss:.6f} | TP1: ${tp1:.6f}")
            
            # === CRITICAL: Validate SL/TP prices against current market ===
            # Bybit rejects SL if it's not clearly on the correct side of current price
            # Get current market price and add safety buffer
            try:
                ticker = await self.exchange.fetch_ticker(futures_symbol)
                current_price = ticker['last']
                
                # Safety buffer: SL must be clearly below/above current price
                safety_buffer = current_price * 0.002  # 0.2% buffer
                
                if side == "long":
                    # For LONG: SL must be BELOW current price
                    if stop_loss >= current_price - safety_buffer:
                        old_sl = stop_loss
                        stop_loss = current_price - safety_buffer
                        logger.warning(f"   ⚠️ SL adjusted for safety: ${old_sl:.6f} → ${stop_loss:.6f} (current: ${current_price:.6f})")
                else:
                    # For SHORT: SL must be ABOVE current price
                    if stop_loss <= current_price + safety_buffer:
                        old_sl = stop_loss
                        stop_loss = current_price + safety_buffer
                        logger.warning(f"   ⚠️ SL adjusted for safety: ${old_sl:.6f} → ${stop_loss:.6f} (current: ${current_price:.6f})")
            except Exception as e:
                logger.warning(f"   ⚠️ Could not validate SL against market: {e}")
            
            # === CHECK AI MODE - Skip server-side TP in advisory/filter mode ===
            # In advisory/filter mode, user wants to control the position manually
            # So we only set SL (for safety), not TP (user decides when to take profit)
            skip_server_tp = self.ai_mode in ['advisory', 'filter']
            if skip_server_tp:
                logger.info(f"   📝 {self.ai_mode.upper()} mode: Skipping server-side TP (you control the exit)")
            
            # === USE BYBIT'S POSITION TRADING-STOP API ===
            # This sets SL/TP on the POSITION level, which is the proper way for Bybit
            try:
                # Bybit trading-stop API requires:
                # - symbol: raw symbol like "ETHUSDT"
                # - stopLoss: stop loss price (string)
                # - takeProfit: take profit price (string) - only if not in advisory mode
                # - positionIdx: 0 for one-way mode
                
                # Build params - only include TP if not in advisory/filter mode
                trading_stop_params = {
                    'category': 'linear',
                    'symbol': raw_symbol,
                    'stopLoss': str(stop_loss),
                    'positionIdx': 0,  # One-way mode
                    'slOrderType': 'Market',
                }
                
                # Only add TP if not in advisory/filter mode
                if not skip_server_tp:
                    trading_stop_params['takeProfit'] = str(tp1)
                    trading_stop_params['tpslMode'] = 'Full'  # SL/TP for full position
                    trading_stop_params['tpOrderType'] = 'Market'  # Must be Market when tpslMode is Full
                
                # Use ccxt's privatePostV5PositionTradingStop
                response = await self.exchange.privatePostV5PositionTradingStop(trading_stop_params)
                
                ret_code = response.get('retCode')
                ret_msg = response.get('retMsg', 'Unknown')
                logger.debug(f"   Trading-stop response: retCode={ret_code} ({type(ret_code).__name__}), retMsg={ret_msg}")
                
                # Handle retCode as int or string - 0 means success
                if str(ret_code) == '0':
                    order_ids['sl_order_id'] = 'position-sl'
                    if skip_server_tp:
                        order_ids['tp1_order_id'] = 'client-side'
                        logger.info(f"   ✅ Position SL set: SL @ ${stop_loss:.6f} (TP managed by you)")
                    else:
                        order_ids['tp1_order_id'] = 'position-tp'
                        logger.info(f"   ✅ Position SL/TP set: SL @ ${stop_loss:.6f} | TP @ ${tp1:.6f}")
                else:
                    logger.warning(f"   ⚠️ Position SL/TP failed (code={ret_code}): {ret_msg}")
                    order_ids['sl_order_id'] = None
                    order_ids['tp1_order_id'] = None
                    
            except Exception as e:
                logger.warning(f"   ⚠️ Position SL/TP failed (will use client-side): {e}")
                order_ids['sl_order_id'] = None
                order_ids['tp1_order_id'] = None
            
            # Note: Using position-level SL/TP means we can only have one TP target
            # TP2 tracking will be done client-side after TP1 hits
            if tp2:
                logger.info(f"   📝 TP2 @ ${tp2:.6f} will be managed client-side after TP1 hit")
                order_ids['tp2_order_id'] = 'client-side'
            
            # Store order IDs for position tracking
            if hasattr(self, 'server_side_orders'):
                self.server_side_orders[futures_symbol] = order_ids
            else:
                self.server_side_orders = {futures_symbol: order_ids}
            
            return order_ids
            
        except Exception as e:
            logger.error(f"🚨 Failed to set position SL/TP: {e}")
            return order_ids
    
    async def _cancel_server_side_orders(self, symbol: str) -> bool:
        """Cancel all server-side SL/TP orders for a symbol.
        
        Call this when closing a position manually or when position is closed externally.
        """
        if not self.live_mode:
            return True
        
        try:
            # Normalize symbol
            if ':USDT' not in symbol:
                symbol_clean = symbol.replace('/', '').replace('USDT', '')
                futures_symbol = f"{symbol_clean}/USDT:USDT"
            else:
                futures_symbol = symbol
            
            # Get stored order IDs
            order_ids = getattr(self, 'server_side_orders', {}).get(futures_symbol, {})
            
            cancelled = 0
            for order_type, order_id in order_ids.items():
                if order_id and order_id not in ['paper', 'unknown', None]:
                    try:
                        await self.exchange.cancel_order(order_id, futures_symbol)
                        logger.info(f"   ✅ Cancelled {order_type}: {order_id}")
                        cancelled += 1
                    except Exception as e:
                        # Order might already be filled or cancelled
                        logger.debug(f"   Could not cancel {order_type} {order_id}: {e}")
            
            # Also cancel all open orders for this symbol (safety)
            try:
                open_orders = await self.exchange.fetch_open_orders(futures_symbol)
                for order in open_orders:
                    if order.get('reduceOnly') or 'stop' in str(order.get('type', '')).lower():
                        await self.exchange.cancel_order(order['id'], futures_symbol)
                        logger.info(f"   ✅ Cancelled orphan order: {order['id']}")
                        cancelled += 1
            except Exception as e:
                logger.debug(f"   Could not fetch/cancel open orders: {e}")
            
            # Clear stored order IDs
            if hasattr(self, 'server_side_orders') and futures_symbol in self.server_side_orders:
                del self.server_side_orders[futures_symbol]
            
            logger.info(f"🧹 Cancelled {cancelled} server-side orders for {futures_symbol}")
            return True
            
        except Exception as e:
            logger.error(f"🚨 Failed to cancel server-side orders: {e}")
            return False
    
    async def _update_server_side_sl(self, symbol: str, new_sl: float, size: float, side: str) -> bool:
        """Update server-side stop loss to a new price (e.g., move to breakeven).
        
        This cancels the old SL order and places a new one.
        """
        if not self.live_mode:
            return True
        
        try:
            # Normalize symbol
            if ':USDT' not in symbol:
                symbol_clean = symbol.replace('/', '').replace('USDT', '')
                futures_symbol = f"{symbol_clean}/USDT:USDT"
            else:
                futures_symbol = symbol
            
            # For Bybit API, we need the raw symbol (e.g., "ETHUSDT")
            raw_symbol = futures_symbol.replace('/USDT:USDT', 'USDT').replace('/', '')
            
            # Update SL using position trading-stop API
            try:
                response = await self.exchange.privatePostV5PositionTradingStop({
                    'category': 'linear',
                    'symbol': raw_symbol,
                    'stopLoss': str(new_sl),
                    'positionIdx': 0,  # One-way mode
                    'tpslMode': 'Full',
                    'slOrderType': 'Market',
                })
                
                if response.get('retCode') == 0:
                    logger.info(f"   ✅ Position SL updated to ${new_sl:.6f}")
                    return True
                else:
                    error_msg = response.get('retMsg', 'Unknown error')
                    logger.warning(f"   ⚠️ Failed to update position SL: {error_msg}")
                    return False
                    
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to update position SL: {e}")
                return False
            
        except Exception as e:
            logger.error(f"🚨 Failed to update server-side SL: {e}")
            return False
    
    async def _execute_market_order(self, symbol: str, side: str, size: float, 
                                     reduce_only: bool = False,
                                     max_retries: int = 3) -> Optional[Dict]:
        """Execute a market order on Bybit Futures with retry logic.
        
        Retries on transient network/rate-limit errors with exponential backoff.
        Does NOT retry on validation errors (insufficient balance, invalid size).
        
        Args:
            symbol: Trading pair (e.g., 'ETHUSDT')
            side: 'buy' or 'sell'
            size: Order size in base currency
            reduce_only: If True, only reduces position (for closing)
            max_retries: Max retry attempts for transient errors (default 3)
            
        Returns:
            Order dict with fill info, or None on failure
        """
        if not self.live_mode:
            logger.debug(f"[PAPER] Would execute {side} {size} {symbol}")
            return {'status': 'paper', 'average': None}
        
        try:
            # Normalize symbol for Bybit futures
            # Convert various formats to ETH/USDT:USDT (ccxt linear perpetual format)
            # Handle: ETHUSDT, ETH/USDT, ETH/USDT:USDT
            if ':USDT' in symbol:
                # Already in futures format like ETH/USDT:USDT
                futures_symbol = symbol if '/' in symbol else f"{symbol.replace('USDT:USDT', '')}/USDT:USDT"
            else:
                # Convert ETHUSDT or ETH/USDT to ETH/USDT:USDT
                symbol_clean = symbol.replace('/', '').replace('USDT', '')
                futures_symbol = f"{symbol_clean}/USDT:USDT"
            
            # === FETCH MARKET INFO FOR MINIMUM ORDER SIZE ===
            # Get minimum order amount and precision from exchange
            try:
                if not hasattr(self, '_market_cache'):
                    self._market_cache = {}
                
                if futures_symbol not in self._market_cache:
                    markets = await self.exchange.load_markets()
                    if futures_symbol in markets:
                        self._market_cache[futures_symbol] = markets[futures_symbol]
                
                market_info = self._market_cache.get(futures_symbol, {})
                limits = market_info.get('limits', {}).get('amount', {})
                min_amount = limits.get('min', 0.001)  # Default fallback
                precision = market_info.get('precision', {}).get('amount', 2)  # Default to 2 decimals
                
                # Round size to exchange precision
                # precision can be:
                # - integer (e.g., 2 = 2 decimal places)
                # - float representing step size (e.g., 0.01 = 2 decimal places)
                import math
                if isinstance(precision, int) and precision > 0:
                    # Integer precision = number of decimal places
                    # CRITICAL FIX: round() first to fix floating-point errors
                    # e.g., 8.03 * 100 = 802.9999... → round → 803 → floor → 803
                    # Without round: floor(802.999) = 802 → 8.02 (WRONG! leaves 0.01 dust)
                    if reduce_only:
                        # For close orders: round to nearest (don't lose dust)
                        size = round(size, precision)
                    else:
                        # For open orders: floor is fine (conservative)
                        size = math.floor(round(size * (10 ** precision), 2)) / (10 ** precision)
                elif isinstance(precision, (int, float)) and precision < 1 and precision > 0:
                    # Decimal step size (e.g., 0.01 = 2 decimals)
                    decimal_places = max(0, int(-math.log10(precision)))
                    if reduce_only:
                        size = round(size, decimal_places)
                    else:
                        size = math.floor(round(size * (10 ** decimal_places), 2)) / (10 ** decimal_places)
                else:
                    # Fallback: round to 2 decimal places (safe for most coins)
                    if reduce_only:
                        size = round(size, 2)
                    else:
                        size = math.floor(size * 100) / 100
                
                # Check minimum order amount AND ensure size is positive
                if size <= 0:
                    logger.error(f"🚨 INVALID ORDER SIZE: {size:.8f} (must be positive) for {futures_symbol}")
                    return None
                if size < min_amount:
                    logger.error(f"🚨 ORDER SIZE TOO SMALL: {size:.8f} < min {min_amount} for {futures_symbol}")
                    logger.warning(f"💡 Tip: Need larger balance or lower-priced asset to trade {futures_symbol}")
                    if self.telegram.enabled:
                        await self.telegram.send_message(
                            f"⚠️ *Order Size Too Small*\\n\\n"
                            f"Symbol: `{futures_symbol}`\\n"
                            f"Size: `{size:.8f}`\\n"
                            f"Minimum: `{min_amount}`\\n"
                            f"\\n_Consider a lower-priced asset or increase balance_"
                        )
                    return None
                
                logger.debug(f"Market info: min_amount={min_amount}, precision={precision}, adjusted_size={size:.8f}")
            except Exception as market_err:
                logger.warning(f"⚠️ Could not fetch market info: {market_err} - proceeding with original size")
            
            logger.info(f"🔴 LIVE ORDER: {side.upper()} {size:.6f} {futures_symbol}")
            
            # Bybit futures order parameters
            params = {}
            if reduce_only:
                params['reduceOnly'] = True
            
            # === RETRY WITH EXPONENTIAL BACKOFF ===
            # Transient errors (network, rate limit, timeout) get retried
            # Validation errors (insufficient balance, invalid params) fail immediately
            import asyncio as _asyncio
            last_error = None
            for attempt in range(1, max_retries + 1):
                try:
                    order = await self.exchange.create_market_order(futures_symbol, side, size, params=params)
                    last_error = None
                    break  # Success
                except Exception as order_err:
                    last_error = order_err
                    err_str = str(order_err).lower()
                    # Non-retryable errors: fail immediately
                    non_retryable = ['insufficient', 'invalid', 'not enough', 'exceed', 
                                     'below minimum', 'too small', 'position side', 'reduce only']
                    if any(nr in err_str for nr in non_retryable):
                        logger.error(f"🚨 ORDER FAILED (non-retryable): {order_err}")
                        break
                    # Retryable: network, timeout, rate limit
                    if attempt < max_retries:
                        wait = 0.5 * (2 ** (attempt - 1))  # 0.5s, 1s, 2s
                        logger.warning(f"⚠️ ORDER ATTEMPT {attempt}/{max_retries} failed: {order_err} — retrying in {wait:.1f}s")
                        await _asyncio.sleep(wait)
                    else:
                        logger.error(f"🚨 ORDER FAILED after {max_retries} attempts: {order_err}")
            
            if last_error is not None:
                if self.telegram.enabled:
                    await self.telegram.send_message(
                        f"🚨 *ORDER FAILED*\\n\\n"
                        f"Symbol: `{futures_symbol}`\\n"
                        f"Side: `{side}`\\n"
                        f"Size: `{size}`\\n"
                        f"Attempts: `{max_retries}`\\n"
                        f"Error: `{str(last_error)[:100]}`"
                    )
                return None
            
            # Extract fill price
            fill_price = None
            if order.get('average') is not None:
                fill_price = float(order['average'])
            elif order.get('price') is not None:
                fill_price = float(order['price'])
            
            order_id = order.get('id', 'unknown')
            status = order.get('status') or 'filled'
            filled = float(order.get('filled', 0)) if order.get('filled') is not None else size
            
            # === PARTIAL FILL DETECTION ===
            if filled > 0 and filled < size * 0.99:  # Allow 1% tolerance for rounding
                fill_ratio = filled / size * 100
                logger.warning(f"⚠️ PARTIAL FILL: {filled:.6f}/{size:.6f} ({fill_ratio:.1f}%) — "
                             f"unfilled {size - filled:.6f} may need attention")
                if self.telegram.enabled:
                    await self.telegram.send_message(
                        f"⚠️ *PARTIAL FILL*\\n\\n"
                        f"Symbol: `{futures_symbol}`\\n"
                        f"Filled: `{filled:.6f}` / `{size:.6f}` ({fill_ratio:.1f}%)\\n"
                        f"_Unfilled portion abandoned (market order)_"
                    )
                # Update size to actual fill for downstream tracking
                order['_original_size'] = size
                order['filled'] = filled
            
            if fill_price:
                logger.info(f"✅ ORDER FILLED: {side.upper()} {filled:.6f} {futures_symbol} @ ${fill_price:.4f} | ID: {order_id}")
            else:
                logger.info(f"✅ ORDER SUBMITTED: {side.upper()} {size:.6f} {futures_symbol} | ID: {order_id}")
            
            # Store order info
            self.last_order_id = order_id
            self.pending_orders[order_id] = {
                'symbol': futures_symbol,
                'side': side,
                'size': filled,
                'price': fill_price,
                'status': status,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Sync balance after order
            await self._sync_balance_with_exchange()
            
            return order
            
        except Exception as e:
            logger.error(f"🚨 ORDER SETUP FAILED: {side} {size} {symbol} - {e}")
            if self.telegram.enabled:
                await self.telegram.send_message(
                    f"🚨 *ORDER FAILED*\\n\\n"
                    f"Symbol: `{symbol}`\\n"
                    f"Side: `{side}`\\n"
                    f"Size: `{size}`\\n"
                    f"Error: `{str(e)[:100]}`"
                )
            return None
    
    async def fetch_initial_data(self):
        """Fetch initial historical data for warmup."""
        logger.info(f"Fetching initial data ({self.WARMUP_BARS} bars) for {self.SYMBOL}...")
        
        # CRITICAL: Clear existing bars when fetching for a (potentially new) symbol
        # This prevents mixing data from different symbols after a symbol switch
        self.bars_1m = []
        self.bars_agg = pd.DataFrame()
        
        # FIX: Reset price caches to prevent stale prices from previous symbol
        self._realtime_price = 0.0
        self.cached_last_price = 0.0
        
        # Need WARMUP_BARS * 3 (for 3:1 aggregation) + extra buffer for aggregation
        bars_to_fetch = (self.WARMUP_BARS * 3) + 10
        ohlcv = await self.exchange.fetch_ohlcv(
            self.SYMBOL,
            self.BASE_TF,
            limit=bars_to_fetch
        )
        
        # Validate OHLCV data before processing
        ohlcv = validate_ohlcv_data(ohlcv, self.SYMBOL)
        
        if len(ohlcv) < self.WARMUP_BARS:
            logger.error(f"❌ Insufficient valid OHLCV data: {len(ohlcv)} bars (need {self.WARMUP_BARS})")
            return
        
        for candle in ohlcv:
            self.bars_1m.append({
                "timestamp": candle[0],
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5])
            })
        
        self._aggregate_bars()
        logger.info(f"Loaded {len(self.bars_agg)} aggregated bars for {self.SYMBOL}")
    
    async def _sync_position_with_exchange(self):
        """Verify restored position still exists and sync state. CRITICAL for restarts."""
        if not self.position:
            return
        
        try:
            logger.info(f"🔄 CRITICAL: Syncing restored position with current market...")
            
            # Get current price for position
            ticker = await self.exchange.fetch_ticker(self.SYMBOL)
            current_price = ticker.get('last', 0)
            
            if current_price <= 0:
                logger.error("❌ Could not get current price - position sync failed!")
                return
            
            pos = self.position
            
            # Calculate unrealized P&L
            upnl = pos.unrealized_pnl(current_price)
            pnl_pct = ((current_price / pos.entry_price) - 1) * 100
            if pos.side == "short":
                pnl_pct = -pnl_pct
            
            # Check stop loss status
            sl_breached = False
            if (pos.side == "long" and current_price <= pos.stop_loss) or \
               (pos.side == "short" and current_price >= pos.stop_loss):
                sl_breached = True
            
            # Check TP levels
            tp1_should_be_hit = False
            tp2_should_be_hit = False
            if pos.side == "long":
                if current_price >= pos.tp1:
                    tp1_should_be_hit = True
                if current_price >= pos.tp2:
                    tp2_should_be_hit = True
            else:  # short
                if current_price <= pos.tp1:
                    tp1_should_be_hit = True
                if current_price <= pos.tp2:
                    tp2_should_be_hit = True
            
            # Log detailed position info
            logger.info(f"📈 POSITION RESTORED: {pos.side.upper()} {pos.symbol}")
            logger.info(f"   Entry: ${pos.entry_price:.4f} | Current: ${current_price:.4f}")
            logger.info(f"   uPNL: ${upnl:+.2f} ({pnl_pct:+.2f}%)")
            logger.info(f"   SL: ${pos.stop_loss:.4f} {'⚠️ BREACHED!' if sl_breached else '✓'}")
            logger.info(f"   TP1: ${pos.tp1:.4f} (hit: {pos.tp1_hit})")
            logger.info(f"   TP2: ${pos.tp2:.4f} (hit: {pos.tp2_hit})")
            logger.info(f"   TP3: ${pos.tp3:.4f}")
            
            # Warnings for offline events
            warnings = []
            if sl_breached:
                warnings.append(f"⚠️ STOP LOSS BREACHED while offline! SL: ${pos.stop_loss:.4f}, Current: ${current_price:.4f}")
                logger.error(warnings[-1])
            if tp1_should_be_hit and not pos.tp1_hit:
                warnings.append(f"⚠️ TP1 likely hit while offline (price passed ${pos.tp1:.4f})")
                logger.warning(warnings[-1])
            if tp2_should_be_hit and not pos.tp2_hit:
                warnings.append(f"⚠️ TP2 likely hit while offline (price passed ${pos.tp2:.4f})")
                logger.warning(warnings[-1])
            
            # Send Telegram notification
            if self.telegram and self.telegram.enabled:
                msg = f"🔄 *Position Restored After Restart*\n\n"
                msg += f"Symbol: `{pos.symbol}`\n"
                msg += f"Side: `{pos.side.upper()}`\n"
                msg += f"Entry: `${pos.entry_price:.4f}`\n"
                msg += f"Current: `${current_price:.4f}`\n"
                msg += f"uPNL: `${upnl:+.2f}` ({pnl_pct:+.2f}%)\n"
                msg += f"SL: `${pos.stop_loss:.4f}`\n"
                msg += f"TP1: `${pos.tp1:.4f}` {'✅' if pos.tp1_hit else '⏳'}\n"
                msg += f"TP2: `${pos.tp2:.4f}` {'✅' if pos.tp2_hit else '⏳'}\n"
                msg += f"TP3: `${pos.tp3:.4f}` {'✅' if pos.tp3_hit else '⏳'}\n"
                
                if warnings:
                    msg += f"\n*⚠️ WARNINGS:*\n"
                    for w in warnings:
                        msg += f"• {w}\n"
                
                await self.telegram.send_message(msg)
            
            # Re-save state to ensure it's current
            self._save_trading_state()
            logger.info(f"💾 Position state verified and saved")
            
            # Ensure server-side SL/TP protection
            await self._ensure_position_protection()
            
        except Exception as e:
            logger.error(f"❌ Failed to sync position with exchange: {e}")
            import traceback
            traceback.print_exc()
            # DON'T clear the position on error - it's still valid locally
            logger.info(f"⚠️ Position kept despite sync error - will retry on next loop")

    async def _ensure_position_protection(self):
        """Ensure all open positions have server-side SL/TP protection.
        
        Call this on startup and after restoring positions to set/verify protection.
        """
        if not self.live_mode:
            return
        
        try:
            # Get all open positions
            all_positions = []
            if self.position:
                all_positions.append(self.position)
            for sym, pos in self.positions.items():
                if pos and pos != self.position:
                    all_positions.append(pos)
            
            if not all_positions:
                return
            
            logger.info(f"🛡️ Ensuring SL/TP protection for {len(all_positions)} position(s)...")
            
            for pos in all_positions:
                try:
                    # Check if position already has server-side SL/TP
                    raw_symbol = pos.symbol.replace('/USDT:USDT', 'USDT').replace('/', '').replace(':USDT', '')
                    
                    # Query Bybit for existing position SL/TP
                    try:
                        bybit_positions = await self.exchange.fetch_positions([pos.symbol])
                        bybit_pos = None
                        for bp in bybit_positions:
                            if raw_symbol in bp['symbol'].replace('/', '').replace(':USDT', ''):
                                bybit_pos = bp
                                break
                        
                        if bybit_pos:
                            has_sl = float(bybit_pos.get('stopLossPrice', 0) or 0) > 0
                            has_tp = float(bybit_pos.get('takeProfitPrice', 0) or 0) > 0
                            
                            if has_sl and has_tp:
                                logger.info(f"   ✅ {pos.symbol}: Already protected (SL: ${float(bybit_pos.get('stopLossPrice', 0)):.4f}, TP: ${float(bybit_pos.get('takeProfitPrice', 0)):.4f})")
                                continue
                            else:
                                logger.info(f"   ⚠️ {pos.symbol}: Missing protection - setting SL/TP...")
                        else:
                            logger.warning(f"   ⚠️ {pos.symbol}: Could not find on Bybit - skipping protection")
                            continue
                            
                    except Exception as check_err:
                        logger.debug(f"   Could not check existing SL/TP: {check_err}")
                    
                    # Set SL/TP on the position
                    await self._place_server_side_sl_tp(
                        symbol=pos.symbol,
                        side=pos.side,
                        size=pos.remaining_size if hasattr(pos, 'remaining_size') else pos.size,
                        entry_price=pos.entry_price,
                        stop_loss=pos.stop_loss,
                        tp1=pos.tp1,
                        tp2=pos.tp2
                    )
                    
                except Exception as pos_err:
                    logger.warning(f"   ⚠️ Failed to protect {pos.symbol}: {pos_err}")
            
            logger.info(f"🛡️ Position protection check complete")
            
        except Exception as e:
            logger.error(f"Failed to ensure position protection: {e}")

    async def pretrain_ml_model(self):
        """Pre-train ML model on historical data using simulated trades."""
        logger.info("🧠 Starting ML pre-training on historical data...")
        
        # Connect if not already connected
        if not self.exchange:
            await self.connect()
        
        # Fetch extended historical data (500 bars = ~25 hours of 3m data)
        logger.info("Fetching extended historical data for ML pre-training...")
        ohlcv = await self.exchange.fetch_ohlcv(
            self.SYMBOL,
            "3m",  # Direct 3m bars for more history
            limit=500
        )
        
        if len(ohlcv) < 100:
            logger.warning("Insufficient historical data for pre-training")
            return
        
        # Build DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Simulate trades on historical data
        ml_classifier = get_ml_classifier()
        samples_before = len(ml_classifier.training_data)
        
        # Use a sliding window to generate training samples
        window_size = 100  # Need at least 100 bars for feature extraction
        
        for i in range(window_size, len(df) - 20):  # Leave 20 bars for outcome
            window_df = df.iloc[i-window_size:i].copy()
            
            # Generate signal for this window
            signals_df = generate_signals(window_df)
            signal = int(signals_df.iloc[-1].get("Side", 0))
            
            if signal == 0:
                continue
            
            # Simulate trade outcome: check if price moved in favorable direction
            entry_price = df.iloc[i]['close']
            future_prices = df.iloc[i:i+20]['close'].values
            
            if signal == 1:  # Long
                max_gain = (max(future_prices) - entry_price) / entry_price * 100
                max_loss = (entry_price - min(future_prices)) / entry_price * 100
                # Win if max gain > 0.5% and > max loss
                won = max_gain > 0.5 and max_gain > max_loss
            else:  # Short
                max_gain = (entry_price - min(future_prices)) / entry_price * 100
                max_loss = (max(future_prices) - entry_price) / entry_price * 100
                won = max_gain > 0.5 and max_gain > max_loss
            
            # Record sample
            ml_classifier.record_sample(window_df, won)
        
        samples_after = len(ml_classifier.training_data)
        new_samples = samples_after - samples_before
        
        logger.info(f"🧠 ML Pre-training complete: {new_samples} new samples added")
        logger.info(f"🧠 Total samples: {samples_after}, Trained: {ml_classifier.is_trained}")
        
        if self.telegram and self.telegram.enabled:
            await self.telegram.send_message(
                f"🧠 *ML Pre-Training Complete*\\n\\n"
                f"New samples: `{new_samples}`\\n"
                f"Total samples: `{samples_after}`\\n"
                f"Model trained: `{'Yes' if ml_classifier.is_trained else 'No'}`"
            )
    
    def _aggregate_bars(self):
        """Aggregate 1m bars to 3m bars."""
        if len(self.bars_1m) < self.AGG_TF_MINUTES:
            return
        
        df = pd.DataFrame(self.bars_1m)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        # Resample to 3-minute bars
        agg = df.resample(f"{self.AGG_TF_MINUTES}min", label="left").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()
        
        self.bars_agg = agg.reset_index()
    
    def _track_error(self, error: Exception, context: str = "unknown") -> None:
        """Track an error for dashboard display.
        
        Args:
            error: The exception that occurred
            context: Where the error happened (main_loop, trade, signal, etc.)
        """
        import traceback
        
        error_msg = str(error)
        
        # === AUTO-EXPIRE OLD ERRORS (older than 30 seconds) ===
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=30)
        self.error_history = [
            e for e in self.error_history 
            if datetime.fromisoformat(e["timestamp"].replace('Z', '+00:00')) > cutoff
        ]
        
        # === DEDUPLICATE: Skip if same error message was just logged ===
        if self.error_history:
            last_error = self.error_history[-1]
            # If same message within last 30 seconds, just update timestamp
            last_time = datetime.fromisoformat(last_error["timestamp"].replace('Z', '+00:00'))
            if (last_error["message"] == error_msg and 
                last_error["context"] == context and
                (datetime.now(timezone.utc) - last_time).total_seconds() < 30):
                # Update count if exists, or add it
                last_error["count"] = last_error.get("count", 1) + 1
                last_error["timestamp"] = datetime.now(timezone.utc).isoformat()
                return  # Don't add duplicate
        
        error_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": type(error).__name__,
            "message": error_msg,
            "context": context,
            "count": 1,
            "traceback": traceback.format_exc() if traceback.format_exc() != "NoneType: None\n" else None
        }
        
        self.error_history.append(error_entry)
        self.total_error_count += 1
        self.last_error = f"{context}: {str(error)}"
        self.last_error_time = datetime.now(timezone.utc)
        
        # Keep only the last N errors
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
        
        logger.error(f"🔴 Error tracked [{context}]: {error}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary for dashboard."""
        # Auto-expire old errors when getting summary too (30 seconds)
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=30)
        self.error_history = [
            e for e in self.error_history 
            if datetime.fromisoformat(e["timestamp"].replace('Z', '+00:00')) > cutoff
        ]
        
        # Count is based on active (non-expired) errors only
        active_count = len(self.error_history)
        
        return {
            "total_errors": active_count,  # Show active errors, not lifetime total
            "recent_errors": active_count,
            "last_error": self.last_error if active_count > 0 else None,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time and active_count > 0 else None,
            "error_history": self.error_history[-10:],  # Last 10 for display
            "engine_running": self.engine_running,
            "cycle_count": self.cycle_count
        }
    
    def clear_errors(self) -> Dict[str, Any]:
        """Clear all error history."""
        self.error_history = []
        self.total_error_count = 0
        self.last_error = None
        self.last_error_time = None
        logger.info("🧹 Error history cleared")
        return {"success": True, "message": "Errors cleared"}
    
    def _calculate_atr(self) -> float:
        """Calculate ATR from aggregated bars."""
        if len(self.bars_agg) < self.ATR_PERIOD + 1:
            return 0.0
        
        df = self.bars_agg.tail(self.ATR_PERIOD + 1).copy()
        df["prev_close"] = df["close"].shift(1)
        df["tr"] = df.apply(
            lambda r: max(
                r["high"] - r["low"],
                abs(r["high"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0,
                abs(r["low"] - r["prev_close"]) if pd.notna(r["prev_close"]) else 0
            ),
            axis=1
        )
        return df["tr"].tail(self.ATR_PERIOD).mean()
    
    @property
    def _last_price(self) -> Optional[float]:
        """Get last known price - prefers real-time WebSocket price if available."""
        # First check real-time price from WebSocket/initialization
        if hasattr(self, '_realtime_price') and self._realtime_price and self._realtime_price > 0:
            return self._realtime_price
        # Fall back to cached_last_price
        if hasattr(self, 'cached_last_price') and self.cached_last_price and self.cached_last_price > 0:
            return self.cached_last_price
        # Fall back to last bar close
        if len(self.bars_1m) > 0:
            return self.bars_1m[-1]["close"]
        return None
    
    def _set_realtime_price(self, price: float):
        """Set real-time price from WebSocket or initialization."""
        if price and price > 0:
            self._realtime_price = price
    
    async def _get_current_price_for_symbol(self, symbol: str) -> Optional[float]:
        """Get current price for any symbol from exchange.
        
        Used for closing positions on symbols other than the current active symbol.
        """
        try:
            # Use the async exchange to fetch ticker
            ticker = await self.exchange.fetch_ticker(symbol)
            if ticker:
                price = ticker.get('last') or ticker.get('close') or ticker.get('bid')
                if price and price > 0:
                    return float(price)
            return None
        except Exception as e:
            logger.warning(f"Error fetching price for {symbol}: {e}")
            return None
    
    async def _initialize_prices_for_dashboard(self):
        """Fetch and cache current prices for all positions at startup.
        
        CRITICAL: This ensures dashboard shows correct PnL immediately after restart,
        before WebSocket starts streaming or main loop fetches data.
        """
        try:
            symbols_to_fetch = set()
            
            # Primary symbol
            symbols_to_fetch.add(self.SYMBOL)
            
            # Primary position symbol (if different)
            if self.position:
                symbols_to_fetch.add(self.position.symbol)
            
            # All multi-position symbols
            for sym in self.positions.keys():
                if sym:
                    symbols_to_fetch.add(sym)
            
            # Additional configured symbols
            for sym in self.additional_symbols:
                symbols_to_fetch.add(sym)
            
            logger.info(f"📊 Initializing prices for {len(symbols_to_fetch)} symbols...")
            
            for symbol in symbols_to_fetch:
                try:
                    ticker = await self.exchange.fetch_ticker(symbol)
                    price = ticker.get('last', 0)
                    
                    if price and price > 0:
                        # Normalize symbol key
                        sym_key = symbol.replace('/USDT:USDT', 'USDT').replace('/', '')
                        
                        # Cache the price
                        self._symbol_prices[sym_key] = price
                        
                        # Set primary _last_price if this is the main symbol
                        main_sym_key = self.SYMBOL.replace('/USDT:USDT', 'USDT').replace('/', '')
                        if sym_key == main_sym_key:
                            self._set_realtime_price(price)
                        
                        # Also set cached_last_price for positions
                        if self.position and self.position.symbol == symbol:
                            self.cached_last_price = price
                        
                        for pos_sym, pos in self.positions.items():
                            if pos and pos_sym == symbol:
                                pos.cached_price = price  # Cache on position object too
                        
                        logger.debug(f"   💰 {sym_key}: ${price:.4f}")
                        
                except Exception as sym_err:
                    logger.warning(f"   ⚠️ Could not fetch price for {symbol}: {sym_err}")
            
            logger.info(f"✅ Prices initialized - dashboard ready")
            
        except Exception as e:
            logger.error(f"❌ Price initialization failed: {e}")
    
    async def run(self):
        """Main bot loop."""
        self.running = True
        
        # === Suppress noisy "Future exception was never retrieved" from ccxt WebSocket ===
        # When WebSocket reconnects, old connection's internal futures complete with
        # RequestTimeout but nobody catches them. This is expected and harmless.
        _original_handler = asyncio.get_event_loop().get_exception_handler()
        def _ws_exception_filter(loop, context):
            msg = context.get('message', '')
            exc = context.get('exception')
            # Suppress ccxt WebSocket timeout futures (expected during reconnect)
            if 'Future exception was never retrieved' in msg:
                if exc and ('RequestTimeout' in type(exc).__name__ or 
                           'ping-pong' in str(exc) or 'timed out' in str(exc)):
                    return  # Silently swallow — this is normal during WS reconnect
            # Pass everything else to default handler
            if _original_handler:
                _original_handler(loop, context)
            else:
                loop.default_exception_handler(context)
        asyncio.get_event_loop().set_exception_handler(_ws_exception_filter)
        
        # Start Telegram bot FIRST (so it's available even if exchange connection fails)
        if self.telegram.enabled:
            telegram_started = False
            for attempt in range(3):
                try:
                    await self.telegram.start()
                    telegram_started = True
                    logger.info("✅ Telegram bot started successfully")
                    break
                except Exception as tg_err:
                    logger.warning(f"⚠️ Telegram start failed (attempt {attempt+1}/3): {tg_err}")
                    if attempt < 2:
                        await asyncio.sleep(3)
            
            if not telegram_started:
                logger.error("❌ Telegram bot failed to start after 3 attempts - continuing without telegram")
        
        # Connect to exchange
        await self.connect()
        
        # === CRITICAL: Initialize prices FIRST for dashboard ===
        # This ensures PnL is accurate immediately after restart
        await self._initialize_prices_for_dashboard()
        
        await self.fetch_initial_data()
        
        # === CRITICAL: Sync ALL positions with Bybit on startup ===
        # This catches positions that were closed externally while bot was offline
        logger.info("🔄 STARTUP: Syncing positions with Bybit exchange...")
        await self._sync_positions_with_bybit()
        
        # Sync restored position with exchange (verify it still exists)
        if self.position:
            await self._sync_position_with_exchange()
        
        # === START WEBSOCKET REAL-TIME STREAMS ===
        if self.ws_enabled:
            try:
                # Add symbols for all configured pairs
                symbols_to_watch = [self.SYMBOL]
                for pos_symbol in self.positions.keys():
                    if pos_symbol and pos_symbol not in symbols_to_watch:
                        symbols_to_watch.append(pos_symbol)
                # Add additional configured pairs
                for extra_sym in self.additional_symbols:
                    if extra_sym not in symbols_to_watch:
                        symbols_to_watch.append(extra_sym)
                
                # === ALWAYS STREAM BTC FOR LEAD INDICATOR ===
                # BTC moves 5-15 seconds before alts. Uses BTC's
                # velocity/acceleration as lead indicator for alts.
                btc_symbol = "BTC/USDT:USDT"
                if btc_symbol not in symbols_to_watch:
                    symbols_to_watch.append(btc_symbol)
                    logger.info("🧭 BTC added to WebSocket for lead indicator")
                
                # Initialize WebSocket with all symbols
                for sym in symbols_to_watch:
                    self.ws_price_stream.add_symbol(sym)
                
                # Register real-time position management callback
                self.ws_price_stream.register_callback(self._on_realtime_price_update)
                
                # Start WebSocket stream
                await self.ws_price_stream.start()
                logger.info(f"🔌 WebSocket PRICE stream started for {len(symbols_to_watch)} symbols")
                
                # === POSITION WEBSOCKET FOR REAL-TIME RECOVERY TRACKING ===
                # ENABLED for instant loss monitoring and fast AI recovery triggers
                # Monitors position PnL in real-time (like peak tracking but for losses)
                try:
                    # Check if ws_position_stream exists (created in __init__)
                    if hasattr(self, 'ws_position_stream'):
                        # Register recovery monitoring callback
                        self.ws_position_stream.register_callback(self._on_position_update_recovery)
                        await self.ws_position_stream.start()
                        logger.info(f"🔌 WebSocket POSITION stream started - REAL-TIME RECOVERY TRACKING enabled")
                except Exception as pos_ws_err:
                    logger.warning(f"⚠️ Position WebSocket failed (recovery tracking uses polling): {pos_ws_err}")
                
            except Exception as ws_err:
                logger.warning(f"⚠️ WebSocket failed to start (using polling fallback): {ws_err}")
                self.ws_enabled = False
        
        # === START NEWS MONITOR BACKGROUND TASK ===
        try:
            from news_monitor import NewsMonitor
            self.news_monitor = NewsMonitor()
            # NOTE: NOT setting Telegram callback - user prefers /news command only
            # No auto-alerts, just background data collection for /news command
            # Start background monitoring (120 second interval)
            self._news_monitor_task = asyncio.create_task(
                self.news_monitor.run_continuous_monitor(interval_seconds=120)
            )
            logger.info("📰 News monitor background task started (alerts disabled, use /news)")
        except Exception as news_err:
            logger.warning(f"⚠️ News monitor failed to start: {news_err}")
            self.news_monitor = None
        
        # Set start time now (after warmup complete)
        self.start_time = datetime.now(timezone.utc)
        self.engine_running = True
        
        logger.info("Bot started - entering main loop")
        
        last_bar_ts = self.bars_1m[-1]["timestamp"] if self.bars_1m else 0
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        try:
            while self.running:
                cycle_start = datetime.now(timezone.utc)
                try:
                    # Track cycle
                    self.cycle_count += 1
                    
                    # Check for pending switch notification (from sync switch method)
                    if hasattr(self, '_pending_switch_notification') and self._pending_switch_notification:
                        msg = self._pending_switch_notification
                        self._pending_switch_notification = None
                        if self.telegram.enabled:
                            await self.telegram.send_message(msg)
                    
                    # Check for pending data refetch (after symbol change)
                    if self._pending_data_refetch:
                        logger.info(f"🔄 Refetching data for new symbol: {self.SYMBOL}")
                        await self.fetch_initial_data()
                        last_bar_ts = self.bars_1m[-1]["timestamp"] if self.bars_1m else 0
                        self._pending_data_refetch = False
                        continue  # Skip this iteration to process new data
                    
                    # Fetch latest candles
                    ohlcv = await self.exchange.fetch_ohlcv(
                        self.SYMBOL,
                        self.BASE_TF,
                        limit=5
                    )
                    
                    # Reset error counter on success
                    consecutive_errors = 0
                    
                    # Process new bars
                    for candle in ohlcv:
                        if candle[0] > last_bar_ts:
                            bar = {
                                "timestamp": candle[0],
                                "open": float(candle[1]),
                                "high": float(candle[2]),
                                "low": float(candle[3]),
                                "close": float(candle[4]),
                                "volume": float(candle[5])
                            }
                            self.bars_1m.append(bar)
                            last_bar_ts = candle[0]
                            
                            # Keep only recent bars
                            if len(self.bars_1m) > 1000:
                                self.bars_1m = self.bars_1m[-800:]
                    
                    # Aggregate and check for closed 3m bar
                    old_len = len(self.bars_agg)
                    self._aggregate_bars()
                    
                    if len(self.bars_agg) > old_len:
                        # New 3m bar closed
                        await self._on_bar_close()
                    
                    # Fetch ticker for /market command (less frequently)
                    try:
                        self.cached_last_ticker = await self.exchange.fetch_ticker(self.SYMBOL)
                    except Exception:
                        pass  # Non-critical, ignore errors
                    
                    # Check position management
                    if self.position:
                        await self._manage_position()
                    # Also manage multi-positions (secondary positions with correct prices)
                    await self._manage_all_multi_positions()
                    
                    # === DRY-RUN SHADOW POSITION MANAGEMENT ===
                    if self.dry_run_mode and self.dry_run_tracker.shadow_positions:
                        shadow_prices = {}
                        # Get live prices for all shadow positions
                        for sym in list(self.dry_run_tracker.shadow_positions.keys()):
                            if sym == self.SYMBOL:
                                shadow_prices[sym] = self._last_price or (self.cached_last_price if hasattr(self, 'cached_last_price') else 0)
                            else:
                                try:
                                    ticker = await self.exchange.fetch_ticker(sym)
                                    shadow_prices[sym] = ticker.get('last', 0)
                                except Exception:
                                    pass
                        if shadow_prices:
                            self.dry_run_tracker.update_shadows(shadow_prices)
                    
                    # === DRY-RUN TELEGRAM NOTIFICATIONS ===
                    if self.dry_run_mode and self.dry_run_tracker.pending_notifications:
                        if self.telegram.enabled:
                            for notif_msg in self.dry_run_tracker.pending_notifications:
                                try:
                                    await self.telegram.send_message(notif_msg)
                                except Exception:
                                    pass
                        self.dry_run_tracker.pending_notifications.clear()
                    
                    # === AI-POWERED POSITION MONITORING (runs every cycle, rate-limited internally) ===
                    # This provides smart profit capture and reversal detection
                    open_positions = [p for p in self.positions.values() if p is not None]
                    if open_positions:
                        # Get price/ATR from cache or from bars if available
                        price = self.cached_last_price
                        atr = self.cached_last_atr
                        if (not price or not atr) and len(self.bars_agg) > 0:
                            price = float(self.bars_agg.iloc[-1]["close"])
                            atr = self._calculate_atr()
                        if price and atr:
                            await self._ai_position_monitor_all(price, atr)
                    
                    # Periodic position save (every 60 sec) to ensure we don't lose it
                    if self.position or any(p for p in self.positions.values() if p):
                        if not hasattr(self, '_last_position_save') or \
                           (datetime.now(timezone.utc) - self._last_position_save).total_seconds() > 60:
                            self._save_trading_state()
                            self._last_position_save = datetime.now(timezone.utc)
                            # Log safely - check for self.position before accessing
                            if self.position:
                                logger.debug(f"💾 Position state persisted: {self.position.side} {self.position.symbol}")
                            else:
                                multi_count = len([p for p in self.positions.values() if p])
                                logger.debug(f"💾 Multi-position state persisted: {multi_count} positions")
                    else:
                        # Also periodically save stats even when no position (every 120 sec)
                        if not hasattr(self, '_last_stats_save') or \
                           (datetime.now(timezone.utc) - self._last_stats_save).total_seconds() > 120:
                            self._save_trading_state()
                            self._last_stats_save = datetime.now(timezone.utc)
                            logger.debug(f"💾 Stats persisted: {self.prefilter_stats.get('total_signals', 0)} signals")
                    
                    # === DASHBOARD FORCE CLOSE ===
                    if self.force_close_symbol:
                        await self._handle_force_close()
                    
                    # AI Proactive Scan in main loop (when no position)
                    # Fixed: Also check multi-positions dict, not just self.position
                    open_positions = [p for p in self.positions.values() if p is not None]
                    if not open_positions and self.ai_mode in ["advisory", "autonomous", "hybrid"]:
                        # Get current price/ATR from bars if cache not set
                        price = self.cached_last_price
                        atr = self.cached_last_atr
                        if (not price or not atr) and len(self.bars_agg) > 0:
                            price = float(self.bars_agg.iloc[-1]["close"])
                            atr = self._calculate_atr()
                        if price and atr:
                            await self._ai_proactive_scan(price, atr)
                    
                    # Sync positions with Bybit (detect externally closed positions)
                    await self._sync_positions_with_bybit()
                    
                    # Autonomous pair switching (check for better opportunities)
                    await self._autonomous_pair_check()
                    
                    # Autonomous periodic summaries
                    await self._check_send_autonomous_summary()
                    
                    # AI self-adjustment based on performance
                    await self._ai_self_adjust()
                    
                    # Weekend/Weekday threshold auto-update
                    await self._check_threshold_mode()
                    
                    # Track cycle completion
                    self.last_cycle_time = datetime.now(timezone.utc)
                    self.last_cycle_duration_ms = (self.last_cycle_time - cycle_start).total_seconds() * 1000
                    
                except Exception as loop_error:
                    consecutive_errors += 1
                    logger.warning(f"⚠️ Loop error ({consecutive_errors}/{max_consecutive_errors}): {loop_error}")
                    
                    # Track error for dashboard
                    self._track_error(loop_error, "main_loop")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"❌ Too many consecutive errors, shutting down")
                        self._track_error(Exception("Too many consecutive errors - shutting down engine"), "fatal")
                        break
                    
                    # Wait longer on errors before retrying
                    await asyncio.sleep(10)
                    continue
                
                # Sleep before next iteration
                # Faster loop when positions open (2s) vs idle (5s)
                open_positions = [p for p in self.positions.values() if p is not None]
                sleep_time = 2 if (self.position or open_positions) else 5
                await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested")
        finally:
            self.engine_running = False
            await self.shutdown()
    
    async def _on_bar_close(self):
        """Handle a new aggregated bar close."""
        # Update equity curve on every bar
        self._update_equity_curve()
        
        if len(self.bars_agg) < self.WARMUP_BARS:
            remaining = self.WARMUP_BARS - len(self.bars_agg)
            if remaining % 10 == 0 or remaining <= 5:
                logger.info(f"⏳ Warming up... {remaining} bars remaining")
            return
        
        current_price = self.bars_agg.iloc[-1]["close"]
        atr = self._calculate_atr()
        
        # Cache values for Telegram commands
        self.cached_last_price = current_price
        self.cached_last_atr = atr
        
        if atr == 0:
            logger.warning("ATR is 0, skipping signal check")
            return
        
        # Check if trading is paused
        if self.paused:
            return
        
        # === BALANCE PROTECTION: Don't process signals if balance too low ===
        BALANCE_PROTECTION_THRESHOLD = getattr(self, 'balance_protection_threshold', 320)
        current_balance = getattr(self, 'balance', 0)
        if current_balance <= BALANCE_PROTECTION_THRESHOLD:
            return  # Silent return, main warning is in _autonomous_pair_check
        
        # === DAILY LOSS LIMIT CHECK (Circuit Breaker) ===
        await self._check_daily_loss_limit()
        if self.daily_loss_triggered:
            return  # Stop trading for the day
        
        # === WEEKLY LOSS LIMIT CHECK (Circuit Breaker) ===
        await self._check_weekly_loss_limit()
        if self.weekly_loss_triggered:
            return  # Stop trading for the week
        
        # === MAX DRAWDOWN CHECK (Ultimate Circuit Breaker) ===
        await self._check_max_drawdown()
        if self.max_drawdown_triggered:
            return  # Trading halted until manual reset
        
        # === REGIME-AWARE SIGNAL GENERATION ===
        # Use regime-aware signals that switch between trend-following and mean-reversion
        df_signals, regime_info = generate_regime_aware_signals(self.bars_agg.copy())
        signal = int(df_signals.iloc[-1].get("Side", 0))
        
        # === TRACK PRE-FILTER STATS FROM INDICATOR ===
        # The indicator may have filtered signals before returning - track those too
        filter_info = regime_info.get('filter_info', {}) if regime_info else {}
        if filter_info and filter_info.get('raw_signals', 0) > 0:
            # A raw signal was generated but may have been filtered
            self.prefilter_stats['raw_signals'] += filter_info['raw_signals']
            self.prefilter_stats['total_signals'] += filter_info['raw_signals']
            
            original_signal = filter_info.get('original_signal', 0)
            filter_reasons = filter_info.get('filter_reasons', [])
            
            if filter_info.get('filtered', 0) > 0:
                # Signal was filtered in indicator - categorize the reason
                for reason in filter_reasons:
                    reason_lower = reason.lower()
                    if 'choppy' in reason_lower or 'hurst' in reason_lower or 'regime' in reason_lower:
                        # Blocked by regime detection (choppy market)
                        if 'blocked_regime' not in self.prefilter_stats:
                            self.prefilter_stats['blocked_regime'] = 0
                        self.prefilter_stats['blocked_regime'] += 1
                    elif 'adx' in reason_lower:
                        self.prefilter_stats['blocked_adx_low'] += 1
                    elif 'volume' in reason_lower:
                        self.prefilter_stats['blocked_volume'] += 1
                    elif 'rsi' in reason_lower or 'overbought' in reason_lower or 'oversold' in reason_lower:
                        self.prefilter_stats['blocked_confluence'] += 1
                    elif 'btc' in reason_lower:
                        self.prefilter_stats['blocked_btc_filter'] += 1
                
                logger.info(f"📊 Indicator filtered: {'LONG' if original_signal == 1 else 'SHORT'} | Reasons: {', '.join(filter_reasons)}")
                self._save_trading_state()
            else:
                # Signal passed indicator filter
                regime = regime_info.get('regime', 'UNKNOWN')
                if regime not in self.prefilter_stats['by_regime']:
                    self.prefilter_stats['by_regime'][regime] = {'total': 0, 'passed': 0}
                self.prefilter_stats['by_regime'][regime]['total'] += 1
                logger.info(f"📊 Raw signal PASSED indicator: {'LONG' if signal == 1 else 'SHORT'} | Regime: {regime}")
                self._save_trading_state()
        
        # Log every bar close with regime info
        logger.info(f"📊 Bar close: Price=${current_price:.4f} ATR=${atr:.4f} Regime={regime_info.get('regime')} Signal={signal}")
        logger.info(f"📊 Position={self.position is not None} | AI_mode={self.ai_mode}")
        
        # Log regime if it changed or on first signal
        current_regime = regime_info.get('regime') if regime_info else 'UNKNOWN'
        if hasattr(self, '_last_regime') and self._last_regime != current_regime:
            strategy = regime_info.get('strategy') if regime_info else 'UNKNOWN'
            logger.info(f"📊 Regime changed: {self._last_regime} → {current_regime} ({strategy})")
        self._last_regime = current_regime
        
        # === TRACK PRE-FILTER STATS FOR SIGNALS THAT PASSED INDICATOR ===
        # These are signals that passed indicator filter but may be blocked by other checks
        if signal != 0:
            regime = regime_info.get('regime', 'UNKNOWN') if regime_info else 'UNKNOWN'
            if regime not in self.prefilter_stats['by_regime']:
                self.prefilter_stats['by_regime'][regime] = {'total': 0, 'passed': 0}
            # Only log if not already tracked above
            if not filter_info or filter_info.get('raw_signals', 0) == 0:
                self.prefilter_stats['raw_signals'] += 1
                self.prefilter_stats['total_signals'] += 1
                self.prefilter_stats['by_regime'][regime]['total'] += 1
                strategy = regime_info.get('strategy') if regime_info else 'UNKNOWN'
                logger.info(f"📊 Raw signal: {'LONG' if signal == 1 else 'SHORT'} | Regime: {regime} | Strategy: {strategy}")
                self._save_trading_state()
        
        
        # === SMART BTC FILTER ===
        if signal != 0:
            btc_result = smart_btc_filter(self.bars_agg, signal)
            if btc_result['should_filter']:
                self.prefilter_stats['blocked_btc_filter'] += 1
                logger.info(f"🔒 Signal filtered by smart BTC filter: {btc_result['reason']}")
                signal = 0
        
        # === MULTI-PAIR CONCURRENT POSITIONS ===
        # Allow up to 2 positions simultaneously (from multi_pair_config)
        max_positions = self.multi_pair_config.max_total_positions
        current_open_positions = len([p for p in self.positions.values() if p is not None])
        
        # NOTE: Signal-based reversal removed - redundant with existing systems:
        # 1. unified_position_decision() - Math+AI exit detection with momentum reversal
        # 2. _ai_position_monitor_all() - AI monitors position health continuously
        # 3. profit_reversal_detected flag in ai_filter - catches momentum against profit
        # These systems already handle closing positions when momentum reverses.
        
        # Process signal if we have room for more positions
        if current_open_positions < max_positions and signal != 0:
            await self._process_signal(signal, current_price, atr, regime_info)
        elif current_open_positions >= max_positions and signal != 0:
            logger.info(f"📊 Signal ignored: {current_open_positions}/{max_positions} positions already open")
            self.prefilter_stats['blocked_max_positions'] += 1
            self._save_trading_state()
        
        # === POSITION MONITORING (Always runs when we have positions) ===
        # Monitor ALL open positions first
        open_positions = [p for p in self.positions.values() if p is not None]
        if open_positions:
            await self._ai_position_monitor_all(current_price, atr)
        
        # === AI PROACTIVE SCAN / PAIR SCANNING ===
        # This runs in parallel with position monitoring:
        # - If below max positions: scan for NEW pair to open
        # - If at max positions: scan for REPLACEMENT opportunity (pair scanner handles this)
        if signal == 0 and self.ai_mode in ["advisory", "autonomous", "hybrid"]:
            if current_open_positions < max_positions:
                logger.info(f"🔍 Triggering AI proactive scan (mode={self.ai_mode}, {current_open_positions}/{max_positions} open)...")
                await self._ai_proactive_scan(current_price, atr)
            # Note: When at max positions, the autonomous pair check handles replacement scanning
    
    async def _ai_position_monitor_all(self, current_price: float, atr: float):
        """
        Monitor ALL open positions (supports multi-position).
        Each position is evaluated independently for hold/close decision.
        """
        try:
            if not self.ai_filter:
                return
            
            open_positions = [(sym, pos) for sym, pos in self.positions.items() if pos is not None]
            
            if not open_positions:
                # Reset watchdog when no positions
                self.watchdog_alert_sent = None
                self.watchdog_user_confirmed = False
                if self.watchdog_original_mode:
                    logger.info(f"🔄 All positions closed - restoring mode to {self.watchdog_original_mode}")
                    self.ai_mode = self.watchdog_original_mode
                    self.watchdog_original_mode = None
                return
            
            # Rate limit based on configurable interval
            # NEW: Monitor new positions (< 60s old) faster (3s) vs established (10s)
            now = datetime.now(timezone.utc)
            _fastest_position_age = 999
            for _sym, _pos in open_positions:
                _opened = getattr(_pos, 'opened_at', None)
                if _opened:
                    try:
                        if isinstance(_opened, datetime):
                            _age = (now - _opened).total_seconds() if _opened.tzinfo else (now - _opened.replace(tzinfo=timezone.utc)).total_seconds()
                        elif isinstance(_opened, str):
                            _dt = datetime.fromisoformat(_opened.replace('Z', '+00:00'))
                            _age = (now - _dt).total_seconds()
                        else:
                            _age = 999
                        _fastest_position_age = min(_fastest_position_age, _age)
                    except Exception:
                        pass
            
            # New positions (< 60s): check every 3s for fast reaction
            # Established positions: check every 10s (normal)
            effective_interval = 3 if _fastest_position_age < 60 else self.ai_position_monitor_interval
            
            if self._last_position_monitor:
                elapsed = (now - self._last_position_monitor).total_seconds()
                if elapsed < effective_interval:
                    await self._check_watchdog_timeout(now)
                    return
            
            self._last_position_monitor = now
            
            logger.info(f"🔍 Position monitor running for {len(open_positions)} positions")
            
            # Initialize price cache if not exists
            if not hasattr(self, '_symbol_prices'):
                self._symbol_prices = {}
            
            # Monitor each position
            for symbol, pos in open_positions:
                try:
                    # For multi-symbol, fetch current price for each symbol
                    if symbol != self.SYMBOL:
                        try:
                            ticker = await self.exchange.fetch_ticker(symbol)
                            sym_price = ticker.get('last', current_price)
                            # Cache the price for dashboard use - use simple key like LINKUSDT
                            sym_key = symbol.replace('/USDT:USDT', 'USDT').replace('/', '')
                            self._symbol_prices[sym_key] = sym_price
                            # Estimate ATR from recent range
                            sym_atr = (ticker.get('high', sym_price) - ticker.get('low', sym_price)) / 14
                            if sym_atr <= 0:
                                sym_atr = sym_price * 0.01
                        except Exception as ticker_err:
                            logger.warning(f"Failed to get ticker for {symbol}, using entry price: {ticker_err}")
                            sym_price = pos.entry_price
                            sym_atr = atr
                    else:
                        sym_price = current_price
                        sym_atr = atr
                        # Cache primary symbol price too - use simple key like LINKUSDT
                        sym_key = symbol.replace('/USDT:USDT', 'USDT').replace('/', '')
                        self._symbol_prices[sym_key] = sym_price
                    
                    # === REAL-TIME PRICE TRACKING: Update last_known_price on position ===
                    # This ensures we have the exact exit price if position is closed externally
                    if hasattr(pos, 'last_known_price'):
                        pos.last_known_price = sym_price
                        pos.last_price_update = datetime.now(timezone.utc)
                    
                    await self._monitor_single_position(pos, sym_price, sym_atr, symbol)
                except Exception as pos_err:
                    logger.error(f"Error monitoring position {symbol}: {pos_err}")
        
        except Exception as e:
            logger.error(f"Position monitor all error: {e}")
    
    async def _monitor_single_position(self, pos, current_price: float, atr: float, symbol: str):
        """Monitor a single position and make hold/close decision.
        
        When AI_FULL_AUTONOMY is True, uses the UNIFIED position decision
        for maximum Math+AI integration.
        
        CRITICAL FIX: Fetch the CORRECT bars for secondary positions!
        """
        try:
            position_side = pos.side.upper()
            entry_price = pos.entry_price
            
            # Get price for this specific symbol if available
            pos_current_price = current_price  # Default to main symbol price
            
            # ===== VALIDATE PRICE BEFORE USING IT =====
            if not validate_price(pos_current_price, entry_price, symbol):
                logger.error(f"🚨 Invalid price {pos_current_price} in _monitor_single_position for {symbol} - skipping")
                return
            
            # Calculate unrealized PnL percentage
            if position_side == "LONG":
                unrealized_pnl_pct = ((pos_current_price - entry_price) / entry_price) * 100
            else:
                unrealized_pnl_pct = ((entry_price - pos_current_price) / entry_price) * 100
            
            # Check TP status
            tp1_hit = getattr(pos, 'tp1_hit', False)
            tp2_hit = getattr(pos, 'tp2_hit', False)
            
            # === USE UNIFIED POSITION DECISION IF FULL AUTONOMY ===
            if getattr(self, 'AI_FULL_AUTONOMY', False):
                # Build position dict for unified decision
                # Include peak_profit tracking for intelligent reversal detection
                position_info = {
                    'symbol': symbol,
                    'side': position_side,
                    'entry_price': entry_price,
                    'size': getattr(pos, 'size', 0),
                    'stop_loss': getattr(pos, 'stop_loss', 0),
                    'tp1_hit': tp1_hit,
                    'tp2_hit': tp2_hit,
                    # Peak profit tracking for intelligent reversal detection
                    'peak_profit_usd': getattr(pos, 'peak_profit_usd', 0),
                    'peak_profit_pct': getattr(pos, 'peak_profit_pct', 0),
                    'deepest_loss_pct': getattr(pos, 'deepest_loss_pct', 0),
                    'deepest_loss_usd': getattr(pos, 'deepest_loss_usd', 0),
                    'recovery_attempts': getattr(pos, 'recovery_attempts', 0),
                    'in_recovery_mode': getattr(pos, 'in_recovery_mode', False),
                    # CRITICAL: Include open_time for new position detection
                    'open_time': getattr(pos, 'opened_at', None),
                    'entry_time': getattr(pos, 'opened_at', None)
                }
                
                # ═══════════════════════════════════════════════════════════════
                # CRITICAL FIX: ALWAYS fetch 15m bars for position monitoring!
                # 
                # ROOT CAUSE OF ENTRY/EXIT S/R MISMATCH:
                # - Entry scanner uses 15m bars (fetch_ohlcv '15m', limit=100) = 24h window
                # - Position monitor was using self.bars_agg = 3m aggregated websocket bars
                # - _detect_resistance_support_levels uses lookback_bars=96
                #   → 96 × 15m = 24 hours of S/R data (entry)
                #   → 96 × 3m  = 4.8 hours of S/R data (exit) ← WRONG!
                # 
                # This caused entry to see Range=14% (wide 24h range) while exit 
                # saw Range=92% (narrow 4.8h range) for the EXACT SAME position,
                # triggering immediate "IN RESISTANCE ZONE" exits.
                #
                # FIX: Always use 15m bars for consistency with entry pipeline.
                # ═══════════════════════════════════════════════════════════════
                pos_symbol_ccxt = symbol  # Already in format like SKR/USDT:USDT
                pos_df = self.bars_agg  # Fallback to main symbol bars
                pos_atr = atr  # Default ATR
                
                # ALWAYS fetch 15m bars — even for main symbol — to match entry scanner
                try:
                    ohlcv = await self.exchange.fetch_ohlcv(pos_symbol_ccxt, '15m', limit=100)
                    if ohlcv and len(ohlcv) >= 50:
                        pos_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        pos_df['timestamp'] = pd.to_datetime(pos_df['timestamp'], unit='ms')
                        # Calculate ATR for this symbol
                        high_low = pos_df['high'] - pos_df['low']
                        high_close = abs(pos_df['high'] - pos_df['close'].shift())
                        low_close = abs(pos_df['low'] - pos_df['close'].shift())
                        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                        pos_atr = tr.rolling(window=14).mean().iloc[-1]
                        if pd.isna(pos_atr) or pos_atr <= 0:
                            pos_atr = pos_current_price * 0.01  # Fallback: 1% of price
                        logger.debug(f"📊 Fetched {len(pos_df)} 15m bars for {symbol} position monitoring (consistent with entry)")
                    else:
                        logger.warning(f"⚠️ Insufficient 15m bars for {symbol} ({len(ohlcv) if ohlcv else 0}) — falling back to agg bars")
                except Exception as ohlcv_err:
                    logger.warning(f"⚠️ Failed to fetch 15m bars for {symbol}: {ohlcv_err} — falling back to agg bars")
                
                # Get unified decision (combines Math + AI)
                unified = self.ai_filter.unified_position_decision(
                    position=position_info,
                    df=pos_df,
                    current_price=pos_current_price,
                    atr=pos_atr,
                    atomic_candle=None
                )
                
                action = unified.get('action', 'hold')
                reasoning = unified.get('reasoning', '')
                confidence = unified.get('confidence', 0.5)
                ai_validated = unified.get('ai_validated', False)
                
                # === UPDATE PEAK PROFIT TRACKING ===
                # Store peak values in position object for next cycle
                peak_tracking = unified.get('peak_tracking', {})
                if peak_tracking:
                    new_peak_usd = peak_tracking.get('peak_profit_usd', 0)
                    new_peak_pct = peak_tracking.get('peak_profit_pct', 0)
                    if new_peak_usd > getattr(pos, 'peak_profit_usd', 0):
                        pos.peak_profit_usd = new_peak_usd
                        logger.debug(f"📈 {symbol}: New peak profit USD: ${new_peak_usd:.2f}")
                    if new_peak_pct > getattr(pos, 'peak_profit_pct', 0):
                        pos.peak_profit_pct = new_peak_pct
                        logger.debug(f"📈 {symbol}: New peak profit %: {new_peak_pct:.2f}%")
                    
                    # Log reversal score if significant
                    reversal_score = peak_tracking.get('reversal_score', 0)
                    drawdown_pct = peak_tracking.get('drawdown_pct_of_peak', 0)
                    if reversal_score >= 20:
                        logger.info(f"📊 {symbol}: Reversal Score={reversal_score:.0f}/100 | Drawdown={drawdown_pct:.0f}% from peak")
                
                logger.info(f"🧠 UNIFIED: {symbol} {position_side} | Action: {action.upper()} | Conf: {confidence:.0%} | PnL: {unrealized_pnl_pct:+.2f}%")
                
                
                # Handle recovery mode
                if action == 'hold_recovery':
                    recovery_score = unified.get('recovery_score', 0)
                    logger.info(f"🔄 RECOVERY MODE: {symbol} holding for recovery | Score: {recovery_score}/100 | Reasoning: {reasoning}")
                    # Position will be held - return without closing
                    return
                
                # Handle SL tightening
                if action == 'tighten_sl' and unified.get('sl_adjustment') and self.ai_mode == 'autonomous':
                    new_sl = unified['sl_adjustment']
                    old_sl = getattr(pos, 'stop_loss', 0)
                    logger.info(f"🛡️ UNIFIED: Tightening SL from {old_sl:.4f} to {new_sl:.4f}")
                    # Update position SL
                    if hasattr(pos, 'stop_loss'):
                        pos.stop_loss = new_sl
                    # Also update on exchange
                    pos_size = getattr(pos, 'size', 0)
                    await self._update_server_side_sl(symbol, new_sl, pos_size, position_side)
                
                # Handle close recommendation - AI OVERRIDE EXECUTION
                if action == 'close':
                    # === CHECK HOLD STATE FIRST (Diamond Hands overrides everything) ===
                    sym_key = symbol.replace('/USDT:USDT', 'USDT').replace('/', '')
                    if self.held_positions.get(sym_key, False):
                        logger.warning(f"💎 DIAMOND HANDS: {symbol} close blocked by hold — AI wanted to close: {reasoning}")
                        return  # Exit without closing - user has diamond hands active
                    
                    # === LOSS PROTECTION OVERRIDE ===
                    # HARD LOSS CAP and critical exits (confidence >= 0.95) ALWAYS execute
                    # regardless of ai_mode. Advisory mode should NEVER block loss protection.
                    is_loss_protection = confidence >= 0.95 or unified.get('realtime', False)
                    
                    if self.ai_mode not in ['autonomous', 'hybrid'] and not is_loss_protection:
                        logger.warning(f"⚠️ UNIFIED wants to close {symbol} but ai_mode={self.ai_mode} (only auto-close in autonomous/hybrid)")
                    else:
                        mode_override = ' [LOSS PROTECTION OVERRIDE]' if is_loss_protection and self.ai_mode not in ['autonomous', 'hybrid'] else ''
                        
                        # === SYSTEM #1: EXIT REASON LEARNER CHECK (Unified) ===
                        _unified_suppress = False
                        # CRITICAL FIX: NEVER suppress exits when position is in LOSS
                        # The learner should only suppress exits on WINNERS (to avoid cutting winners short)
                        # Suppressing a loss exit costs real money (COWUSDT: -$6.69 → -$14.77 from suppression)
                        _is_in_loss = unrealized_pnl_pct < -0.02  # Position is losing money
                        if not is_loss_protection and not _is_in_loss and hasattr(self, 'ai_filter') and hasattr(self.ai_filter, 'exit_learner'):
                            try:
                                _exit_cat = self.ai_filter.exit_learner._classify_exit(reasoning)
                                _exit_mod = self.ai_filter.exit_learner.get_exit_modifier(_exit_cat)
                                if _exit_mod.get('suppress', False):
                                    logger.warning(f"🧠 EXIT LEARNER SUPPRESS: {_exit_cat} blocked for {symbol} (Unified) | {_exit_mod.get('reason', '')}")
                                    _unified_suppress = True
                                elif _exit_mod.get('delay_seconds', 0) > 0:
                                    logger.info(f"🧠 EXIT LEARNER DELAY: {_exit_cat} suggests {_exit_mod['delay_seconds']}s delay (logged only)")
                            except Exception as _el_err:
                                logger.debug(f"Exit learner check failed (non-fatal): {_el_err}")
                        elif _is_in_loss and not is_loss_protection:
                            logger.info(f"🧠 EXIT LEARNER BYPASSED: {symbol} is in LOSS ({unrealized_pnl_pct:+.2f}%) — never suppress loss exits")
                        
                        if not _unified_suppress:
                            logger.warning(f"🚨 AI EXECUTING CLOSE{mode_override} on {symbol}: {reasoning}")
                            logger.warning(f"   Confidence: {confidence:.0%} | AI Validated: {ai_validated}")
                            try:
                                # Execute the close
                                pos_symbol = getattr(self.position, 'symbol', '').replace('/', '')
                                if symbol == pos_symbol or symbol.replace('/', '') == pos_symbol:
                                    logger.info(f"📍 Closing primary position {symbol}")
                                    await self._close_position(f"Unified Exit: {reasoning}", pos_current_price)
                                else:
                                    logger.info(f"📍 Closing multi-position {symbol}")
                                    await self._close_position_by_symbol(symbol, f"Unified Exit: {reasoning}", pos_current_price)
                                logger.info(f"✅ CLOSE EXECUTED for {symbol}")
                            except Exception as close_err:
                                logger.error(f"❌ CLOSE FAILED for {symbol}: {close_err}")
                return
            
            # === FALLBACK: Standard monitor_position ===
            # Get AI/Math decision
            decision = self.ai_filter.monitor_position(
                df=self.bars_agg,
                position_side=position_side,
                entry_price=entry_price,
                current_price=pos_current_price,
                atr=atr,
                symbol=symbol,
                tp1_hit=tp1_hit,
                tp2_hit=tp2_hit,
                unrealized_pnl_pct=unrealized_pnl_pct,
                position_size=getattr(pos, 'size', 0)
            )
            
            if not decision:
                return
            
            action = decision.get('action', 'hold')
            reasoning = decision.get('reasoning', '')
            
            # Log the decision
            logger.debug(f"📊 Position {symbol}: {action.upper()} | PnL: {unrealized_pnl_pct:+.2f}% | {reasoning[:50]}...")
            
            # Handle close recommendation
            if action == 'close' and self.ai_mode == 'autonomous':
                # === SYSTEM #1: EXIT REASON LEARNER CHECK (Fallback) ===
                _fb_suppress = False
                # CRITICAL FIX: NEVER suppress exits when position is in LOSS
                _is_in_loss_fb = unrealized_pnl_pct < -0.02
                if not _is_in_loss_fb and hasattr(self, 'ai_filter') and hasattr(self.ai_filter, 'exit_learner'):
                    try:
                        _exit_cat = self.ai_filter.exit_learner._classify_exit(reasoning)
                        _exit_mod = self.ai_filter.exit_learner.get_exit_modifier(_exit_cat)
                        if _exit_mod.get('suppress', False):
                            logger.warning(f"🧠 EXIT LEARNER SUPPRESS: {_exit_cat} blocked for {symbol} (Fallback) | {_exit_mod.get('reason', '')}")
                            _fb_suppress = True
                        elif _exit_mod.get('delay_seconds', 0) > 0:
                            logger.info(f"🧠 EXIT LEARNER DELAY: {_exit_cat} suggests {_exit_mod['delay_seconds']}s delay (logged only)")
                    except Exception as _el_err:
                        logger.debug(f"Exit learner check failed (non-fatal): {_el_err}")
                elif _is_in_loss_fb:
                    logger.info(f"🧠 EXIT LEARNER BYPASSED: {symbol} is in LOSS ({unrealized_pnl_pct:+.2f}%) — never suppress loss exits (Fallback)")
                
                if not _fb_suppress:
                    logger.info(f"🤖 AI recommends closing {symbol} position: {reasoning}")
                    # Close this specific position
                    if symbol == getattr(self.position, 'symbol', '').replace('/', ''):
                        await self._close_position(f"AI Exit: {reasoning}", pos_current_price)
                    else:
                        # Close from multi-position dict
                        await self._close_position_by_symbol(symbol, f"AI Exit: {reasoning}", pos_current_price)
        
        except Exception as e:
            logger.error(f"Error in _monitor_single_position for {symbol}: {e}")
    
    async def _ai_position_monitor(self, current_price: float, atr: float):
        """
        AI monitors open positions.
        - Autonomous mode: Can close positions directly
        - Advisory/Hybrid mode: Watchdog alerts user, auto-switches to autonomous if no response
        """
        try:
            if not self.ai_filter or not self.position:
                # Reset watchdog when no position
                self.watchdog_alert_sent = None
                self.watchdog_user_confirmed = False
                if self.watchdog_original_mode:
                    # Restore original mode if we auto-switched
                    logger.info(f"🔄 Position closed - restoring mode to {self.watchdog_original_mode}")
                    self.ai_mode = self.watchdog_original_mode
                    self.watchdog_original_mode = None
                return
            
            # Rate limit based on configurable interval
            now = datetime.now(timezone.utc)
            if self._last_position_monitor:
                elapsed = (now - self._last_position_monitor).total_seconds()
                if elapsed < self.ai_position_monitor_interval:
                    # Still check watchdog timeout even when rate-limited
                    await self._check_watchdog_timeout(now)
                    return
            
            self._last_position_monitor = now
            
            # Calculate position metrics (Position is a dataclass, use attributes)
            position_side = self.position.side.upper()  # "long" -> "LONG"
            entry_price = self.position.entry_price
            size = self.position.size
            
            # Calculate unrealized PnL percentage
            if position_side == "LONG":
                unrealized_pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                unrealized_pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            # Check TP status
            tp1_hit = self.position.tp1_hit
            tp2_hit = self.position.tp2_hit
            
            # Get AI/Math decision
            decision = self.ai_filter.monitor_position(
                df=self.bars_agg,
                position_side=position_side,
                entry_price=entry_price,
                current_price=current_price,
                atr=atr,
                symbol=self.SYMBOL,
                unrealized_pnl_pct=unrealized_pnl_pct,
                tp1_hit=tp1_hit,
                tp2_hit=tp2_hit,
                position_size=size
            )
            
            action = decision.get('action', 'hold')
            confidence = decision.get('confidence', 0.5)
            reasoning = decision.get('reasoning', '')
            math_score = decision.get('math_score', 50)
            
            # === MATH-AWARE DYNAMIC SL ADJUSTMENT ===
            await self._math_aware_sl_adjustment(
                math_score=math_score,
                unrealized_pnl_pct=unrealized_pnl_pct,
                current_price=current_price,
                position_side=position_side
            )
            
            # Only act if confidence is high enough
            MIN_CLOSE_CONFIDENCE = 0.65
            
            # === MANUAL MODE: Only SL adjustment (already done above), no close actions ===
            if self.ai_mode == "manual":
                # Math-aware SL adjustment already applied above
                # In manual mode, user controls closing - we just protect with SL
                logger.debug(f"📊 Manual mode: Position monitored (Hold:{math_score}), SL protection active")
                return
            
            # === AUTONOMOUS MODE: Direct action ===
            if self.ai_mode == "autonomous":
                if action == "close" and confidence >= MIN_CLOSE_CONFIDENCE:
                    logger.info(f"🤖 AI AUTONOMOUS: Closing {position_side} position | Confidence: {confidence:.0%}")
                    logger.info(f"🤖 Reason: {reasoning}")
                    
                    await self._close_position(
                        reason=f"AI Monitor: {reasoning[:50]}",
                        price=current_price,
                        is_manual=False
                    )
                elif action == "close" and confidence < MIN_CLOSE_CONFIDENCE:
                    logger.debug(f"🤖 AI considering close but confidence too low ({confidence:.0%} < {MIN_CLOSE_CONFIDENCE:.0%})")
            
            # === ADVISORY/HYBRID MODE: Watchdog with user confirmation ===
            else:
                await self._handle_watchdog_alert(
                    position_side=position_side,
                    entry_price=entry_price,
                    current_price=current_price,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    action=action,
                    confidence=confidence,
                    reasoning=reasoning,
                    math_score=math_score,
                    now=now
                )
                
        except Exception as e:
            logger.error(f"🚨 AI Position Monitor error: {type(e).__name__}: {e}")
    
    async def _handle_watchdog_alert(self, position_side: str, entry_price: float, 
                                      current_price: float, unrealized_pnl_pct: float,
                                      action: str, confidence: float, reasoning: str,
                                      math_score: int, now: datetime):
        """Handle watchdog alerts for non-autonomous modes."""
        if not self.watchdog_enabled:
            return
        
        # If user already confirmed, skip alerting
        if self.watchdog_user_confirmed:
            return
        
        # Determine if position needs attention
        needs_attention = (
            action == "close" or  # AI recommends closing
            math_score < 40 or  # Low hold score
            unrealized_pnl_pct < -1.0  # Losing more than 1%
        )
        
        if not needs_attention:
            return
        
        # Send alert if not already sent
        if self.watchdog_alert_sent is None:
            pnl_emoji = "🟢" if unrealized_pnl_pct >= 0 else "🔴"
            alert_msg = (
                f"⚠️ *POSITION WATCHDOG ALERT*\n\n"
                f"Your {position_side} {self.SYMBOL} position needs attention!\n\n"
                f"📊 Entry: ${entry_price:.4f}\n"
                f"📊 Current: ${current_price:.4f}\n"
                f"{pnl_emoji} PnL: {unrealized_pnl_pct:+.2f}%\n"
                f"🧠 AI Score: {math_score}/100\n"
                f"🎯 AI Recommendation: {action.upper()}\n"
                f"📝 Reason: {reasoning[:100]}\n\n"
                f"⏰ *Respond within 10 minutes* or I'll switch to autonomous mode to protect the position.\n\n"
                f"Reply `/watching` to confirm you're monitoring.\n"
                f"Reply `/close` to close the position now."
            )
            
            try:
                await self.telegram.send_message(alert_msg)
                self.watchdog_alert_sent = now
                logger.info(f"🐕 Watchdog alert sent - waiting for user confirmation")
            except Exception as e:
                logger.error(f"Failed to send watchdog alert: {e}")
    
    async def _check_watchdog_timeout(self, now: datetime):
        """Check if watchdog timeout has expired and auto-switch to autonomous."""
        if not self.watchdog_enabled or not self.watchdog_alert_sent:
            return
        
        if self.watchdog_user_confirmed:
            return
        
        elapsed = (now - self.watchdog_alert_sent).total_seconds()
        
        if elapsed >= self.watchdog_timeout:
            # User didn't respond - switch to autonomous mode
            logger.warning(f"⚠️ Watchdog timeout ({self.watchdog_timeout}s) - switching to autonomous mode")
            
            # Store original mode to restore later
            self.watchdog_original_mode = self.ai_mode
            self.ai_mode = "autonomous"
            
            # Notify user
            try:
                await self.telegram.send_message(
                    f"🤖 *AUTO-SWITCH TO AUTONOMOUS MODE*\n\n"
                    f"No response received within {self.watchdog_timeout // 60} minutes.\n"
                    f"I'm now managing your {self.position.side.upper()} {self.SYMBOL} position.\n\n"
                    f"Previous mode ({self.watchdog_original_mode}) will be restored when position closes.\n\n"
                    f"Reply `/mode advisory` or `/mode hybrid` to switch back manually."
                )
            except Exception as e:
                logger.error(f"Failed to send auto-switch notification: {e}")
            
            # Reset alert so we don't keep triggering
            self.watchdog_alert_sent = None
    
    def confirm_watchdog(self) -> str:
        """User confirms they're watching the position."""
        if not self.position:
            return "No open position to watch."
        
        self.watchdog_user_confirmed = True
        self.watchdog_alert_sent = None
        
        return (
            f"✅ *Confirmed!* You're watching the {self.position.side.upper()} {self.SYMBOL} position.\n"
            f"I won't auto-switch to autonomous mode for this position.\n"
            f"Stay vigilant! 👀"
        )
    
    async def _check_send_autonomous_summary(self):
        """Automatically send periodic trading summaries via Telegram."""
        if not self.summary_notifications_enabled:
            return
        
        now = datetime.now(timezone.utc)
        
        # Check for daily summary (at midnight UTC)
        today = now.date()
        if self.last_daily_summary_date != today and now.hour == 0:
            # Send daily summary
            await self._send_daily_summary()
            self.last_daily_summary_date = today
            return
        
        # Check for periodic summary (every 4 hours)
        if self.last_summary_time is None:
            self.last_summary_time = now
            return
        
        elapsed = (now - self.last_summary_time).total_seconds()
        if elapsed >= self.summary_interval:
            await self._send_periodic_summary()
            self.last_summary_time = now
    
    async def _ai_self_adjust(self):
        """AI autonomously adjusts settings based on trading performance and market conditions."""
        if not self.ai_self_adjust_enabled or self.ai_mode != "autonomous":
            return
        
        now = datetime.now(timezone.utc)
        if self.last_ai_self_adjust:
            elapsed = (now - self.last_ai_self_adjust).total_seconds()
            if elapsed < self.ai_self_adjust_interval:
                return
        
        self.last_ai_self_adjust = now
        
        # Analyze recent performance
        adjustments_made = []
        
        # === NEW: AI STRATEGIC ANALYSIS (Full Power Mode) ===
        if self.AI_FULL_AUTONOMY and hasattr(self.ai_filter, 'ai_strategic_analysis'):
            try:
                # Gather market conditions
                market_conditions = {
                    'btc_trend': 'UNKNOWN',
                    'volatility': 'NORMAL',
                    'sentiment': 'NEUTRAL'
                }
                
                # Get current positions info
                current_positions = []
                if self.position:
                    current_positions.append({
                        'symbol': self.position.symbol,
                        'side': self.position.side,
                        'entry': self.position.entry_price,
                        'pnl_pct': self.position.unrealized_pnl_pct(self.cached_last_price) if self.cached_last_price else 0
                    })
                
                # Call AI strategic analysis
                strategy_result = self.ai_filter.ai_strategic_analysis(
                    balance=self.balance,
                    recent_trades=self.trade_history[-10:] if hasattr(self, 'trade_history') else [],
                    current_positions=current_positions,
                    market_conditions=market_conditions,
                    adaptive_params=self.adaptive_params
                )
                
                if strategy_result.get('available') and strategy_result.get('strategy'):
                    strategy = strategy_result['strategy']
                    
                    # Apply trading mode
                    trading_mode = strategy.get('trading_mode', 'normal')
                    if trading_mode == 'aggressive':
                        # SAFETY: Aggressive mode DISABLED — too dangerous for small accounts
                        logger.warning(f"🛡️ AI requested AGGRESSIVE mode but it is DISABLED for safety")
                        adjustments_made.append(f"🛡️ AGGRESSIVE mode blocked (safety)")
                    elif trading_mode in ['defensive', 'conservative']:
                        self.adaptive_params['aggressive_mode']['current'] = 0
                        # Also reduce risk
                        current_risk = self.adaptive_params.get('risk_pct', {}).get('current', 0.05)
                        new_risk = max(0.02, current_risk * 0.7)
                        self.adaptive_params['risk_pct']['current'] = new_risk
                        adjustments_made.append(f"🛡️ Mode: DEFENSIVE (risk→{new_risk*100:.1f}%)")
                    
                    # Apply risk adjustment
                    risk_adj = strategy.get('risk_adjustment', 0)
                    if risk_adj != 0:
                        current_risk = self.adaptive_params.get('risk_pct', {}).get('current', 0.02)
                        new_risk = max(0.01, min(0.04, current_risk + risk_adj))  # SAFETY: Max 4%
                        if new_risk != current_risk:
                            self.adaptive_params['risk_pct']['current'] = new_risk
                            adjustments_made.append(f"📊 Risk: {current_risk*100:.1f}% → {new_risk*100:.1f}%")
                    
                    # Apply param changes
                    for change in strategy.get('param_changes', []):
                        param = change.get('param')
                        new_val = change.get('new_value')
                        if param in self.adaptive_params and new_val is not None:
                            info = self.adaptive_params[param]
                            if info['min'] <= new_val <= info['max']:
                                old_val = info['current']
                                info['current'] = new_val
                                adjustments_made.append(f"🎛️ {param}: {old_val} → {new_val}")
                    
                    # Handle urgent actions
                    urgent = strategy.get('urgent_action')
                    
                    # === SAFETY RE-CLAMP after AI changes ===
                    # Ensure AI cannot push params beyond safe limits through any path
                    self._apply_adaptive_params()  # Re-applies with safety clamps
                    
                    if urgent:
                        action = urgent.get('action')
                        reason = urgent.get('reason', 'AI decision')
                        if action == 'close_all' and self.position:
                            logger.warning(f"⚠️ AI URGENT: Closing all positions - {reason}")
                            # Will be handled by position management
                            adjustments_made.append(f"⚠️ URGENT: {action} - {reason}")
                    
                    # Save updated params
                    self._save_adaptive_params()
                    
            except Exception as strat_err:
                logger.warning(f"AI strategic analysis error: {strat_err}")
        
        # === EXISTING: Basic Performance-Based Adjustments ===
        # Check win rate and consecutive losses
        if self.stats.total_trades >= 5:
            win_rate = self.stats.win_rate
            
            # If losing streak >= 3, reduce risk
            if self.consecutive_losses >= 3:
                old_risk = self.RISK_PCT
                new_risk = max(0.01, old_risk * 0.5)  # Halve risk, min 1%
                if new_risk != old_risk:
                    self.RISK_PCT = new_risk
                    adjustments_made.append(f"📉 Risk reduced: {old_risk*100:.1f}% → {new_risk*100:.1f}% (losing streak: {self.consecutive_losses})")
                    logger.info(f"🤖 AI SELF-ADJUST: Reduced risk to {new_risk*100:.1f}% due to {self.consecutive_losses} losses")
            
            # If winning streak >= 3 and good win rate, slightly increase risk
            elif self.consecutive_wins >= 3 and win_rate > 0.55:
                old_risk = self.RISK_PCT
                new_risk = min(0.04, old_risk * 1.25)  # Increase by 25%, max 4%
                if new_risk != old_risk:
                    self.RISK_PCT = new_risk
                    adjustments_made.append(f"📈 Risk increased: {old_risk*100:.1f}% → {new_risk*100:.1f}% (winning streak: {self.consecutive_wins})")
                    logger.info(f"🤖 AI SELF-ADJUST: Increased risk to {new_risk*100:.1f}% due to {self.consecutive_wins} wins")
            
            # If win rate drops below 40%, tighten AI confidence threshold
            if win_rate < 0.40 and self.ai_filter.confidence_threshold < 0.85:
                old_conf = self.ai_filter.confidence_threshold
                new_conf = min(0.85, old_conf + 0.05)
                self.ai_filter.confidence_threshold = new_conf
                adjustments_made.append(f"🎯 AI confidence raised: {old_conf*100:.0f}% → {new_conf*100:.0f}% (win rate: {win_rate*100:.0f}%)")
                logger.info(f"🤖 AI SELF-ADJUST: Raised confidence threshold to {new_conf*100:.0f}%")
            
            # If win rate above 60%, can relax confidence slightly
            elif win_rate > 0.60 and self.ai_filter.confidence_threshold > 0.65:
                old_conf = self.ai_filter.confidence_threshold
                new_conf = max(0.65, old_conf - 0.05)
                self.ai_filter.confidence_threshold = new_conf
                adjustments_made.append(f"🎯 AI confidence relaxed: {old_conf*100:.0f}% → {new_conf*100:.0f}% (win rate: {win_rate*100:.0f}%)")
                logger.info(f"🤖 AI SELF-ADJUST: Relaxed confidence threshold to {new_conf*100:.0f}%")
        
        # Notify about adjustments
        if adjustments_made and self.telegram.enabled:
            msg = "🤖 *AI AUTONOMOUS ADJUSTMENT*\n\n"
            msg += "\n".join(adjustments_made)
            msg += f"\n\n_Current: Risk={self.RISK_PCT*100:.1f}%, Confidence={self.ai_filter.confidence_threshold*100:.0f}%_"
            await self.telegram.send_message(msg)
            
            # Log the decision
            self.ai_decisions_log.append({
                "time": now.isoformat(),
                "type": "self_adjust",
                "adjustments": adjustments_made
            })
    
    async def _notify_ai_decision(self, decision_type: str, details: Dict[str, Any]):
        """Notify user about AI decision with full transparency."""
        now = datetime.now(timezone.utc)
        
        # Log the decision
        self.ai_decisions_log.append({
            "time": now.isoformat(),
            "type": decision_type,
            "details": details
        })
        
        # Keep only last 100 decisions
        if len(self.ai_decisions_log) > 100:
            self.ai_decisions_log = self.ai_decisions_log[-100:]
        
        # Always notify for important decisions
        if self.telegram.enabled:
            if decision_type == "scan_result":
                # Skip ALL scan notifications if ai_scan_telegram_enabled is False
                if not self.ai_scan_telegram_enabled:
                    logger.debug(f"AI scan result: notification disabled (ai_scan_telegram_enabled=False)")
                    return
                
                if details.get("opportunity"):
                    opp = details["opportunity"]
                    
                    # Skip notification if position already open on this symbol
                    symbol = opp.get('symbol', self.SYMBOL)
                    symbol_key = symbol.replace('/', '')  # Normalize LINK/USDT -> LINKUSDT
                    has_position = False
                    
                    # Check primary position
                    if self.position and getattr(self.position, 'symbol', '').replace('/', '') == symbol_key:
                        has_position = True
                        logger.debug(f"AI scan: Skipping notification - position already open on {symbol}")
                    
                    # Check multi-positions
                    if not has_position and hasattr(self, 'positions') and self.positions:
                        for pos_symbol in self.positions.keys():
                            if pos_symbol.replace('/', '') == symbol_key:
                                has_position = True
                                logger.debug(f"AI scan: Skipping notification - position already open on {symbol}")
                                break
                    
                    if has_position:
                        return  # Don't notify - position already open
                    
                    msg = f"""🔍 *AI SCAN RESULT*

📊 *{self.SYMBOL}* @ ${details.get('price', 0):,.4f}
🎯 *Decision:* {opp.get('action', 'WAIT')}
📈 *Confidence:* `{opp.get('confidence', 0)*100:.0f}%`
💡 *Reasoning:* {opp.get('reasoning', 'N/A')[:200]}
⚠️ *Risk Level:* {opp.get('risk_assessment', 'medium')}

_AI Mode: {self.ai_mode}_"""
                    await self.telegram.send_message(msg)
                else:
                    # Skip "no opportunity" notifications if configured
                    if self.ai_scan_notify_opportunities_only:
                        logger.debug("AI scan: No opportunity (notification suppressed)")
                        return
                    
                    # Rate limit "no opportunity" notifications
                    if self.last_ai_decision_notification:
                        elapsed = (now - self.last_ai_decision_notification).total_seconds()
                        if elapsed < self.ai_scan_quiet_interval:
                            return
                    self.last_ai_decision_notification = now
                    
                    msg = f"""🔍 *AI SCAN*

📊 *{self.SYMBOL}* @ ${details.get('price', 0):,.4f}
🎯 *Decision:* No trade opportunity
💭 *Status:* Watching market...

_Next scan in ~{self.ai_scan_interval//60} min_"""
                    await self.telegram.send_message(msg)
            
            elif decision_type == "trade_opened":
                msg = f"""🚀 *AI TRADE OPENED*

📊 *{details.get('symbol', self.SYMBOL)}*
💰 *Side:* `{details.get('side', 'N/A')}`
📈 *Entry:* `${details.get('price', 0):,.4f}`
🎯 *Confidence:* `{details.get('confidence', 0)*100:.0f}%`
💵 *Risk:* `{details.get('risk_pct', 0)*100:.1f}%` (${details.get('risk_amount', 0):,.2f})
💡 *Reason:* {details.get('reasoning', 'AI Autonomous')}

_Executed autonomously by AI_"""
                await self.telegram.send_message(msg)
            
            elif decision_type == "trade_closed":
                emoji = "✅" if details.get('pnl', 0) >= 0 else "❌"
                msg = f"""{emoji} *AI TRADE CLOSED*

📊 *{details.get('symbol', self.SYMBOL)}*
💰 *Side:* `{details.get('side', 'N/A')}`
📈 *Exit:* `${details.get('price', 0):,.4f}`
💵 *P&L:* `${details.get('pnl', 0):+,.2f}` ({details.get('pnl_pct', 0):+.2f}%)
📝 *Reason:* {details.get('reason', 'N/A')}

_Balance: ${self.balance:,.2f}_"""
                await self.telegram.send_message(msg)
            
            elif decision_type == "signal_filtered":
                msg = f"""🛡️ *AI SIGNAL FILTERED*

📊 *{self.SYMBOL}* @ ${details.get('price', 0):,.4f}
⚠️ *Original Signal:* `{details.get('signal', 'N/A')}`
❌ *AI Decision:* REJECT
📈 *AI Confidence:* `{details.get('confidence', 0)*100:.0f}%`
💭 *Reason:* {details.get('reasoning', 'N/A')[:200]}

_Technical signal filtered by AI_"""
                await self.telegram.send_message(msg)

    async def _check_daily_loss_limit(self):
        """Check and enforce daily loss limit (circuit breaker)."""
        today = datetime.now(timezone.utc).date()
        
        # Reset circuit breaker at midnight UTC
        if self.daily_loss_reset_date != today:
            self.daily_loss_reset_date = today
            if self.daily_loss_triggered:
                logger.info("🔄 Daily loss circuit breaker reset (new day)")
                self.daily_loss_triggered = False
                if self.telegram.enabled:
                    await self.telegram.send_message("🔄 *Daily Loss Limit Reset*\nTrading resumed for new day.")
        
        # Check if daily loss exceeds limit
        # Use CURRENT balance as the base, not initial_balance (which is the original starting amount)
        if self.stats.today_pnl < 0:
            # Calculate loss as percentage of current balance (more accurate for grown accounts)
            base_for_pct = max(self.balance, self.initial_balance)  # Use larger of current or initial
            daily_loss_pct = abs(self.stats.today_pnl) / base_for_pct
            
            # Check for manual override (user reset the limit for today)
            override_active = (self.daily_loss_override_until == today)
            
            if daily_loss_pct >= self.daily_loss_limit and not self.daily_loss_triggered and not override_active:
                self.daily_loss_triggered = True
                logger.warning(f"🛑 DAILY LOSS LIMIT HIT: {daily_loss_pct*100:.1f}% (limit: {self.daily_loss_limit*100:.1f}%)")
                logger.warning("Trading halted until tomorrow or manual reset")
                
                if self.telegram.enabled:
                    await self.telegram.send_message(
                        f"🛑 *DAILY LOSS LIMIT TRIGGERED*\n\n"
                        f"Daily Loss: `${self.stats.today_pnl:.2f}` ({daily_loss_pct*100:.1f}%)\n"
                        f"Limit: `{self.daily_loss_limit*100:.1f}%`\n\n"
                        f"_Trading halted until tomorrow._\n"
                        f"Use `/reset daily_loss` to override."
                    )
    
    async def _check_weekly_loss_limit(self):
        """Check and enforce weekly loss limit (circuit breaker)."""
        from datetime import timezone
        now = datetime.now(timezone.utc)
        current_week = now.isocalendar()[1]  # ISO week number
        
        # Reset circuit breaker on new week (Monday)
        if self.weekly_loss_reset_week != current_week:
            self.weekly_loss_reset_week = current_week
            self.weekly_pnl = 0.0  # Reset weekly P&L tracking
            if self.weekly_loss_triggered:
                logger.info("🔄 Weekly loss circuit breaker reset (new week)")
                self.weekly_loss_triggered = False
                if self.telegram.enabled:
                    await self.telegram.send_message("🔄 *Weekly Loss Limit Reset*\nTrading resumed for new week.")
        
        # Track weekly P&L (add today's to cumulative)
        # Note: This is simplified - for accuracy, should sum all trades in the week
        weekly_loss_pct = abs(self.weekly_pnl) / max(self.balance, self.initial_balance) if self.weekly_pnl < 0 else 0
        
        if weekly_loss_pct >= self.weekly_loss_limit and not self.weekly_loss_triggered:
            self.weekly_loss_triggered = True
            logger.warning(f"🛑 WEEKLY LOSS LIMIT HIT: {weekly_loss_pct*100:.1f}% (limit: {self.weekly_loss_limit*100:.1f}%)")
            logger.warning("Trading halted until next week!")
            
            if self.telegram.enabled:
                await self.telegram.send_message(
                    f"🛑 *WEEKLY LOSS LIMIT TRIGGERED*\n\n"
                    f"Weekly Loss: `${self.weekly_pnl:.2f}` ({weekly_loss_pct*100:.1f}%)\n"
                    f"Limit: `{self.weekly_loss_limit*100:.1f}%`\n\n"
                    f"_Trading halted until next week._"
                )
    
    async def _check_max_drawdown(self):
        """Check and enforce max drawdown circuit breaker (ultimate safety)."""
        if self.peak_balance <= 0:
            return
        
        # Sanity check: if peak is wildly stale (>2x current balance),
        # auto-recalibrate instead of triggering false drawdown
        if self.peak_balance > self.balance * 2 and self.balance > 0:
            old_peak = self.peak_balance
            self.peak_balance = self.balance
            self.max_drawdown_triggered = False
            self._save_trading_state()
            logger.info(f"📈 AUTO-RECALIBRATED peak in drawdown check: ${old_peak:.2f} → ${self.balance:.2f}")
            return
        
        drawdown_pct = (self.peak_balance - self.balance) / self.peak_balance
        
        if drawdown_pct >= self.max_drawdown_limit and not self.max_drawdown_triggered:
            self.max_drawdown_triggered = True
            logger.critical(f"🚨 MAX DRAWDOWN CIRCUIT BREAKER: {drawdown_pct*100:.1f}% drawdown from peak ${self.peak_balance:.2f}")
            logger.critical("TRADING HALTED - Manual intervention required!")
            
            if self.telegram.enabled:
                await self.telegram.send_message(
                    f"🚨 *CRITICAL: MAX DRAWDOWN TRIGGERED*\n\n"
                    f"Current Drawdown: `{drawdown_pct*100:.1f}%`\n"
                    f"Peak Balance: `${self.peak_balance:.2f}`\n"
                    f"Current Balance: `${self.balance:.2f}`\n"
                    f"Limit: `{self.max_drawdown_limit*100:.1f}%`\n\n"
                    f"_⚠️ TRADING HALTED - Manual review required!_\n"
                    f"Use `/reset drawdown` to resume after review."
                )
    
    async def _send_periodic_summary(self):
        """Send a periodic status summary."""
        if not self.telegram.enabled:
            return
        
        # Calculate stats
        uptime = datetime.now(timezone.utc) - self.start_time if self.start_time else timedelta(0)
        hours = int(uptime.total_seconds() // 3600)
        minutes = int((uptime.total_seconds() % 3600) // 60)
        
        pnl_pct = ((self.balance - self.initial_balance) / self.initial_balance * 100) if self.initial_balance > 0 else 0
        drawdown = ((self.peak_balance - self.balance) / self.peak_balance * 100) if self.peak_balance > 0 else 0
        
        position_status = "None"
        if self.position:
            # FIX: Get correct price for position's symbol
            pos_symbol = getattr(self.position, 'symbol', '').upper().replace('/', '').replace(':USDT', '')
            pos_price = self.position.entry_price  # Default
            if hasattr(self, '_symbol_prices') and pos_symbol in self._symbol_prices:
                pos_price = self._symbol_prices[pos_symbol]
            elif hasattr(self, '_symbol_prices'):
                alt_key = pos_symbol.replace('USDT', '') + 'USDT'
                if alt_key in self._symbol_prices:
                    pos_price = self._symbol_prices[alt_key]
            # Use cached_last_price only if this IS the main symbol
            if pos_price == self.position.entry_price:
                main_sym = self.SYMBOL.replace('/', '').replace(':USDT', '')
                if pos_symbol == main_sym and self.cached_last_price:
                    pos_price = self.cached_last_price
            unrealized = self.position.unrealized_pnl(pos_price)
            position_status = f"{self.position.side.upper()} (${unrealized:+.2f})"
        
        # Get ML status
        ml_status = "N/A"
        try:
            ml_classifier = get_ml_classifier()
            ml_stats = ml_classifier.get_stats()
            ml_status = f"Trained ({ml_stats.get('total_samples', 0)} samples)" if ml_stats.get('is_trained') else f"Learning ({ml_stats.get('samples_until_training', 50)} needed)"
        except (AttributeError, RuntimeError) as e:
            logger.debug(f"Could not get ML stats: {e}")
            ml_status = "N/A"
        except Exception:
            ml_status = "N/A"
        
        emoji = "📈" if self.stats.total_pnl >= 0 else "📉"
        
        # Add circuit breaker status
        circuit_status = "🛑 HALTED" if self.daily_loss_triggered else "✅ Active"
        dry_run_status = "📝 DRY-RUN" if self.dry_run_mode else ""
        
        msg = f"""
🤖 *Julaba Status Update* {dry_run_status}

⏱ *Uptime:* `{hours}h {minutes}m`
💰 *Balance:* `${self.balance:,.2f}` ({pnl_pct:+.2f}%)
{emoji} *Total P&L:* `${self.stats.total_pnl:+,.2f}`
📊 *Drawdown:* `{drawdown:.1f}%`
🚦 *Circuit:* `{circuit_status}`

*Trading Stats*
🎯 Trades: `{self.stats.total_trades}` ({self.stats.winning_trades}W / {self.stats.losing_trades}L)
📈 Win Rate: `{self.stats.win_rate * 100:.1f}%`
🔥 Streak: `{self.consecutive_wins}W / {self.consecutive_losses}L`

*Current State*
📍 Position: `{position_status}`
💵 Price: `${self.cached_last_price:,.4f}`
🧠 ML: `{ml_status}`
🤖 AI Mode: `{self.ai_mode}`

_Auto-update every 4 hours_
"""
        await self.telegram.send_message(msg)
        logger.info("Periodic summary sent to Telegram")
        
        # === PERIODIC STATE SAVE ===
        self._save_trading_state()
    
    async def _send_daily_summary(self):
        """Send daily trading summary at midnight UTC."""
        if not self.telegram.enabled:
            return
        
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Daily P&L from today_pnl stat
        today_pnl = self.stats.today_pnl
        
        await self.telegram.notify_daily_summary(
            date=yesterday,
            trades=self.stats.total_trades,
            wins=self.stats.winning_trades,
            losses=self.stats.losing_trades,
            pnl=today_pnl,
            balance=self.balance,
            win_rate=self.stats.win_rate * 100
        )
        
        # Reset daily stats
        self.stats.today_pnl = 0.0
        logger.info(f"Daily summary sent for {yesterday}")

    async def _notify_safe(self, message: str, critical: bool = False) -> bool:
        """
        SAFE Telegram notification wrapper.
        
        Catches ALL errors so notifications never crash the bot.
        Returns True if sent successfully, False if failed.
        
        Args:
            message: Message to send
            critical: If True, logs as error; if False, as warning
        
        Usage:
            success = await self._notify_safe("📊 Trade opened: LONG BTC")
            if not success and it_was_critical:
                logger.error("Failed to notify about critical trade!")
        """
        try:
            if not self.telegram.enabled:
                return False
            
            await self.telegram.send_message(message)
            return True
        except Exception as e:
            level = "ERROR" if critical else "WARNING"
            logger.log(
                logging.ERROR if critical else logging.WARNING,
                f"[{level}] Failed to send Telegram notification: {e}"
            )
            return False

    async def _check_threshold_mode(self):
        """Check if we transitioned between weekend and weekday.
        Auto-updates all thresholds and sends Telegram notification on change.
        Called every main loop cycle — only acts on actual transitions.
        """
        try:
            now_berlin = get_berlin_time()
            is_weekend = now_berlin.weekday() >= 5  # Saturday=5, Sunday=6
            new_mode = 'weekend' if is_weekend else 'weekday'
            
            # First run — set mode silently, NO notification
            if self._current_threshold_mode is None:
                self._current_threshold_mode = new_mode
                logger.info(f"📊 Threshold mode initialized: {new_mode.upper()} (no notification on startup)")
                return
            
            # No change — nothing to do
            if new_mode == self._current_threshold_mode:
                return
            
            # === TRANSITION DETECTED ===
            old_mode = self._current_threshold_mode
            self._current_threshold_mode = new_mode
            logger.info(f"📊 Threshold mode changed: {old_mode.upper()} → {new_mode.upper()}")
            
            await self._send_threshold_notification(new_mode, is_initial=False)
            
        except Exception as e:
            logger.warning(f"Threshold mode check error: {e}")
    
    async def _send_threshold_notification(self, mode: str, is_initial: bool = False):
        """Send Telegram notification about threshold settings."""
        try:
            now_berlin = get_berlin_time()
            time_str = now_berlin.strftime("%H:%M %Z")
            day_str = now_berlin.strftime("%A %d %b")
            
            if mode == 'weekend':
                msg = (
                    f"🌙 {'WEEKEND MODE ACTIVE' if is_initial else 'SWITCHED TO WEEKEND MODE'}\n"
                    f"📅 {day_str} • {time_str}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📊 Thresholds Updated:\n"
                    f"  • Stage 2 Min Score: 65 → 38\n"
                    f"  • Stage 3 Min Math: 65 → 38\n"
                    f"  • Stage 3 Combined: 70 → 45\n"
                    f"  • Proactive Threshold: 65 → 43\n"
                    f"  • AI Confidence Min: 70% → 55%\n"
                    f"  • Penalty Cap: 60% → 45%\n"
                    f"  • Session Penalty: -15 → -5\n"
                    f"  • MTF 1H Penalty: -15 → -10\n"
                    f"  • MTF 4H Penalty: -20 → -12\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"💡 Relaxed thresholds for weekend\n"
                    f"   crypto market conditions"
                )
            else:
                msg = (
                    f"☀️ {'WEEKDAY MODE ACTIVE' if is_initial else 'SWITCHED TO WEEKDAY MODE'}\n"
                    f"📅 {day_str} • {time_str}\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📊 Thresholds Updated:\n"
                    f"  • Stage 2 Min Score: 38 → 65\n"
                    f"  • Stage 3 Min Math: 38 → 65\n"
                    f"  • Stage 3 Combined: 45 → 70\n"
                    f"  • Proactive Threshold: 43 → 65\n"
                    f"  • AI Confidence Min: 55% → 70%\n"
                    f"  • Penalty Cap: 45% → 60%\n"
                    f"  • Session Penalty: -5 → -15\n"
                    f"  • MTF 1H Penalty: -10 → -15\n"
                    f"  • MTF 4H Penalty: -12 → -20\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"💡 STRICT thresholds for weekday\n"
                    f"   Math must be 100% accurate"
                )
            
            if hasattr(self, 'telegram') and self.telegram:
                await self.telegram.send_message(msg)
                logger.info(f"📨 Threshold notification sent: {mode.upper()}")
            else:
                logger.info(f"📊 Threshold {mode}: (Telegram not available)")
                
        except Exception as e:
            logger.warning(f"Threshold notification error: {e}")
    
    async def _ai_proactive_scan(self, price: float, atr: float):
        """Let AI proactively scan for opportunities.
        
        Wrapped in comprehensive error handling to prevent crashes in autonomous mode.
        MULTI-PAIR AWARE: When there's an open slot, AI compares all available pairs
        and chooses the BEST one instead of just using the current symbol.
        
        When AI_FULL_AUTONOMY is True, uses the UNIFIED MATH+AI SCANNER
        for maximum Math+AI integration.
        """
        try:
            # === EARLY BALANCE PROTECTION: Don't scan if balance too low ===
            BALANCE_PROTECTION_THRESHOLD = getattr(self, 'balance_protection_threshold', 320)
            current_balance = getattr(self, 'balance', 0)
            if current_balance <= BALANCE_PROTECTION_THRESHOLD:
                logger.debug(f"AI scan skipped: balance ${current_balance:.2f} <= ${BALANCE_PROTECTION_THRESHOLD}")
                return
            
            # Check how many positions we have
            open_positions = [p for p in self.positions.values() if p is not None]
            max_positions = self.multi_pair_config.max_total_positions
            current_count = len(open_positions)
            
            # Skip scan if position already open on this symbol (we're managing it, not looking for new)
            symbol_key = normalize_symbol(self.SYMBOL)
            has_position_on_current = False
            
            # Check multi-positions
            if hasattr(self, 'positions') and self.positions:
                pos = self.positions.get(symbol_key)
                if pos is not None:
                    has_position_on_current = True
            
            if has_position_on_current:
                # We have a position on the current symbol - let position management handle it
                logger.debug(f"AI scan skipped for {self.SYMBOL}: position exists ({current_count}/{max_positions})")
                return
            
            # Rate limit: only scan based on ai_scan_interval
            now = datetime.now(timezone.utc)
            if self.last_ai_scan_time:
                elapsed = (now - self.last_ai_scan_time).total_seconds()
                if elapsed < self.ai_scan_interval:
                    logger.debug(f"AI scan rate-limited: {elapsed:.0f}s / {self.ai_scan_interval}s")
                    return
            
            self.last_ai_scan_time = now
            logger.info(f"🔍 AI Proactive Scan starting... (interval: {self.ai_scan_interval}s)")
            
            # === USE UNIFIED MATH+AI SCANNER IF FULL AUTONOMY ===
            if getattr(self, 'AI_FULL_AUTONOMY', False):
                logger.info("🧠 UNIFIED MATH+AI SCAN starting...")
                
                unified_result = await self._unified_math_ai_scan()
                
                if unified_result.get('has_opportunity'):
                    best = unified_result.get('best_pair', {})
                    
                    # Convert unified result to opportunity format
                    opportunity = {
                        'found': True,
                        'action': best.get('direction', 'LONG').upper(),
                        'confidence': best.get('ai_confidence', 0.7),
                        'suggested_risk_pct': self.RISK_PCT,
                        'risk_assessment': 'medium',
                        'math_score': best.get('math_score', 0),
                        'ai_score': best.get('ai_score', 0),
                        'combined_score': best.get('combined_score', 0),
                        'reasons_for': best.get('reasons_for', []),
                        'reasons_against': best.get('reasons_against', []),
                        # === REC #3: BAYESIAN FEEDBACK ===
                        # Carry component scores from math_check through to Position object
                        'entry_component_scores': best.get('math_check', {}).get('scores', {}),
                    }
                    
                    # Get the correct symbol data
                    trade_symbol = best.get('symbol', self.SYMBOL)
                    trade_price = best.get('price', price)
                    trade_atr = best.get('atr', atr)
                    scan_bars = best.get('df', self.bars_agg)
                    
                    # Proceed directly to trade execution with unified result
                    if self.ai_mode == "autonomous":
                        logger.info(f"🤖 UNIFIED AUTONOMOUS: Opening {opportunity['action']} on {normalize_symbol(trade_symbol)}")
                        logger.info(f"   Math: {opportunity['math_score']:.0f} | AI: {opportunity['ai_score']:.0f} | Combined: {opportunity['combined_score']:.0f}")
                        
                        # Pass to trade execution
                        await self._execute_unified_trade(
                            trade_symbol=trade_symbol,
                            trade_price=trade_price,
                            trade_atr=trade_atr,
                            scan_bars=scan_bars,
                            opportunity=opportunity
                        )
                        return
                    else:
                        # Suggest mode - notify user
                        await self._notify_ai_decision("opportunity", {
                            "price": trade_price,
                            "opportunity": opportunity,
                            "symbol": trade_symbol
                        })
                        return
                else:
                    logger.info(f"🔍 Unified scan: No opportunity ({unified_result.get('reason', 'unknown')})")
                    return
            
            # === FALLBACK: Original multi-pair scan logic ===
            # Validate inputs
            if price <= 0 or atr <= 0:
                logger.warning(f"AI scan skipped: invalid price={price} or atr={atr}")
                return
            
            if self.bars_agg is None or len(self.bars_agg) < 20:
                logger.warning(f"AI scan skipped: insufficient data ({len(self.bars_agg) if self.bars_agg is not None else 0} bars)")
                return
            
            # === MULTI-PAIR AWARE: Find the BEST pair to scan ===
            # Don't just use self.SYMBOL - compare all available pairs first
            scan_symbol = self.SYMBOL
            scan_price = price
            scan_atr = atr
            scan_bars = self.bars_agg
            
            try:
                # Get market scan data to find best pair (NOT async - don't await)
                scan_data = self._get_market_scan_data()
                # Extract the pairs list from the dict (scan_data has 'pairs' key)
                pairs = scan_data.get('pairs', []) if isinstance(scan_data, dict) else scan_data
                
                if pairs and len(pairs) > 0:
                    # Get symbols we already have positions on
                    open_symbols = set()
                    for pos in open_positions:
                        open_symbols.add(normalize_symbol(getattr(pos, 'symbol', '')))
                    
                    # Find best available pair (not already in a position)
                    best_pair = None
                    for p in pairs:
                        pair_symbol = normalize_symbol(p.get('symbol', ''))
                        if pair_symbol not in open_symbols:
                            best_pair = p
                            break
                    
                    if best_pair:
                        best_symbol = normalize_symbol(best_pair.get('symbol', ''))
                        best_score = best_pair.get('score', 0)
                        current_score = 0
                        
                        # Find score of current symbol in the pairs list
                        for p in pairs:
                            if normalize_symbol(p.get('symbol', '')) == symbol_key:
                                current_score = p.get('score', 0)
                                break
                        
                        # ALWAYS use best pair when:
                        # 1. We have 0 positions (fresh start - should always use best)
                        # 2. Best pair is different AND has higher score
                        # 3. _needs_full_pair_scan flag is set (positions just closed)
                        should_switch = False
                        switch_reason = ""
                        
                        if current_count == 0:
                            # No positions open - always use the best available pair
                            should_switch = True
                            switch_reason = "0 positions open"
                        elif getattr(self, '_needs_full_pair_scan', False):
                            # Just closed all positions - do full scan
                            should_switch = True
                            switch_reason = "full pair scan required"
                            self._needs_full_pair_scan = False  # Reset flag
                        elif best_symbol != symbol_key and best_score > current_score:
                            # Best pair is better than current
                            should_switch = True
                            switch_reason = f"better score ({best_score:.1f} vs {current_score:.1f})"
                        
                        if should_switch and best_symbol != symbol_key:
                            logger.info(f"🔄 AI switching to BEST pair: {best_symbol} (score: {best_score:.1f}) - reason: {switch_reason}")
                            
                            # Fetch data for the better pair
                            try:
                                ccxt_symbol = f"{best_symbol.replace('USDT', '')}/USDT:USDT"
                                ticker = await self.exchange.fetch_ticker(ccxt_symbol)
                                scan_price = ticker['last']
                                
                                # Fetch OHLCV for the new symbol (using 15m timeframe)
                                ohlcv = await self.exchange.fetch_ohlcv(ccxt_symbol, '15m', limit=100)
                                if ohlcv and len(ohlcv) >= 20:
                                    import pandas as pd
                                    scan_bars = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                                    scan_bars['timestamp'] = pd.to_datetime(scan_bars['timestamp'], unit='ms')
                                    scan_bars.set_index('timestamp', inplace=True)
                                    
                                    # Calculate ATR for new symbol
                                    from indicator import calculate_atr
                                    scan_atr = calculate_atr(scan_bars, period=14)
                                    if scan_atr <= 0:
                                        scan_atr = scan_bars['close'].rolling(14).std().iloc[-1]
                                    
                                    scan_symbol = ccxt_symbol
                                    logger.info(f"📊 AI will scan {best_symbol}: price=${scan_price:.4f}, ATR={scan_atr:.4f}")
                                else:
                                    logger.warning(f"Insufficient data for {best_symbol}, falling back to {symbol_key}")
                            except Exception as fetch_err:
                                logger.warning(f"Failed to fetch data for {best_symbol}: {fetch_err}, using {symbol_key}")
                        else:
                            logger.debug(f"Current symbol {symbol_key} is best or equal (score: {current_score:.1f} vs {best_score:.1f})")
            except Exception as pair_err:
                logger.warning(f"Pair comparison failed: {pair_err}, using current symbol {symbol_key}")
            
            # AI scans the market with error protection
            try:
                opportunity = self.ai_filter.proactive_scan(
                    df=scan_bars,
                    current_price=scan_price,
                    atr=scan_atr,
                    symbol=scan_symbol
                )
            except Exception as scan_err:
                logger.error(f"🚨 AI proactive_scan error: {scan_err}")
                return
            
            logger.info(f"🔍 AI Scan result: {'Found opportunity!' if opportunity else 'No opportunity'}")
            
            # Notify about scan result (opportunity or not)
            try:
                await self._notify_ai_decision("scan_result", {
                    "price": price,
                    "opportunity": opportunity
                })
            except Exception as notify_err:
                logger.warning(f"AI scan notification failed: {notify_err}")
            
            if not opportunity:
                return
            
            # Handle based on AI mode
            if self.ai_mode == "autonomous":
                try:
                    # AI opens trade directly with full autonomy (70%+ confidence)
                    ai_risk_pct = opportunity.get('suggested_risk_pct', self.RISK_PCT)
                    risk_level = opportunity.get('risk_assessment', 'medium')
                    
                    # AI can adjust risk based on its assessment
                    if risk_level == "low":
                        ai_risk_pct = min(ai_risk_pct * 1.5, 0.05)  # Up to 5% on low risk
                    elif risk_level == "high":
                        ai_risk_pct = ai_risk_pct * 0.5  # Reduce on high risk
                    
                    # Use the BEST pair we found (scan_symbol), not necessarily self.SYMBOL
                    trade_symbol = scan_symbol if scan_symbol else self.SYMBOL
                    trade_price = scan_price if scan_price else price
                    trade_atr = scan_atr if scan_atr else atr
                    
                    logger.info(f"🤖 AI AUTONOMOUS: Opening {opportunity['action']} on {normalize_symbol(trade_symbol)} | Risk: {ai_risk_pct:.1%} ({risk_level})")
                    
                    # === RECORD AI DECISION FOR TRACKING ===
                    try:
                        self.current_decision_id = self.ai_tracker.record_decision(
                            symbol=trade_symbol,
                            signal_direction=opportunity['action'],
                            price=trade_price,
                            approved=True,  # AI approved its own decision
                            confidence=opportunity['confidence'],
                            reasoning=opportunity['reasoning'],
                            threshold_used=0.70,  # AI autonomous threshold
                            regime=opportunity.get('regime', 'UNKNOWN'),
                            tech_score=opportunity.get('tech_score', 0),
                            system_score=opportunity.get('system_score', 0),
                            ml_probability=opportunity.get('ml_probability', 0.5),
                            ml_confidence=opportunity.get('ml_confidence', 'N/A')
                        )
                        logger.info(f"📊 AI Decision recorded: {self.current_decision_id}")
                    except Exception as tracker_err:
                        logger.warning(f"AI tracker error (non-fatal): {tracker_err}")
                        self.current_decision_id = None
                    
                    # Store AI's risk preference temporarily
                    original_risk = self.RISK_PCT
                    self.RISK_PCT = ai_risk_pct
                    
                    # Temporarily switch to the best symbol if different
                    original_symbol = self.SYMBOL
                    if trade_symbol != self.SYMBOL:
                        self.SYMBOL = trade_symbol
                        logger.info(f"🔄 AI temporarily switching symbol to {trade_symbol} for this trade")
                    
                    try:
                        await self._open_position(
                            opportunity['signal'],
                            trade_price,
                            trade_atr,
                            source="ai_autonomous",
                            ai_reasoning=opportunity.get('reasoning', ''),
                            ai_confidence=opportunity.get('confidence', 0)
                        )
                    except Exception as open_err:
                        logger.error(f"🚨 AI autonomous trade open failed: {open_err}")
                    finally:
                        # Always restore original settings
                        self.RISK_PCT = original_risk
                        self.SYMBOL = original_symbol
                
                except Exception as auto_err:
                    logger.error(f"🚨 AI autonomous mode error: {auto_err}")
            
            elif self.ai_mode in ["advisory", "hybrid"]:
                try:
                    # Store pending trade with the CORRECT scanned symbol/price/atr
                    # Without this, _confirm_ai_trade uses self.SYMBOL (wrong symbol)
                    # which causes wrong position sizing (e.g. BTC sizing on ELSA)
                    trade_symbol = scan_symbol if scan_symbol else self.SYMBOL
                    trade_price = scan_price if scan_price else price
                    trade_atr = scan_atr if scan_atr else atr
                    
                    opportunity['_scan_symbol'] = trade_symbol
                    opportunity['_scan_price'] = trade_price
                    opportunity['_scan_atr'] = trade_atr
                    
                    self.pending_ai_trade = opportunity
                    logger.info(f"🤖 AI {self.ai_mode.upper()}: Suggesting {opportunity['action']} on {normalize_symbol(trade_symbol)} - awaiting confirmation")
                    
                    if self.telegram.enabled:
                        await self.telegram.notify_ai_trade(
                            symbol=trade_symbol,
                            action=opportunity['action'],
                            price=trade_price,
                            confidence=opportunity['confidence'],
                            reasoning=opportunity['reasoning'],
                            mode=self.ai_mode
                        )
                except Exception as advisory_err:
                    logger.error(f"🚨 AI advisory mode error: {advisory_err}")
        
        except Exception as e:
            logger.error(f"🚨 AI Proactive Scan FATAL error: {e}")
            # Don't crash - just log and continue

    async def _check_btc_crash_protection(self) -> Dict[str, Any]:
        """Check if BTC has crashed and activate cooldown if needed."""
        result = {"cooldown_active": False, "btc_change": 0, "reason": None}
        
        try:
            # Get current BTC price
            ticker = await self.exchange.fetch_ticker("BTC/USDT")
            current_btc = ticker['last']
            
            # Initialize if first check
            if self.last_btc_price is None:
                self.last_btc_price = current_btc
                return result
            
            # Calculate BTC change
            btc_change = (current_btc - self.last_btc_price) / self.last_btc_price
            result['btc_change'] = btc_change
            
            # Check for crash
            if btc_change <= self.btc_crash_threshold:
                self.btc_crash_cooldown = True
                self.btc_crash_cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=self.btc_crash_cooldown_minutes)
                result['cooldown_active'] = True
                result['reason'] = f"BTC crashed {btc_change:.1%} - activating {self.btc_crash_cooldown_minutes}min cooldown"
                logger.warning(f"🚨 {result['reason']}")
                
                if self.telegram.enabled:
                    await self.telegram.send_message(
                        f"🚨 *BTC CRASH PROTECTION ACTIVATED*\n\n"
                        f"BTC dropped: `{btc_change:.1%}`\n"
                        f"Cooldown: `{self.btc_crash_cooldown_minutes} minutes`\n"
                        f"Until: `{self.btc_crash_cooldown_until.strftime('%H:%M:%S')}`\n\n"
                        f"_All new trades paused during crash correlation spike_"
                    )
            
            # Update last price periodically (every 5 minutes)
            if btc_change > 0.02 or btc_change < -0.02:  # >2% move
                self.last_btc_price = current_btc
            
            # Check if cooldown expired
            if self.btc_crash_cooldown and datetime.now(timezone.utc) > self.btc_crash_cooldown_until:
                self.btc_crash_cooldown = False
                logger.info("✅ BTC crash cooldown expired - trading resumed")
                if self.telegram.enabled:
                    await self.telegram.send_message("✅ *BTC crash cooldown expired* - Trading resumed")
            
            if self.btc_crash_cooldown:
                result['cooldown_active'] = True
                result['reason'] = f"BTC crash cooldown active until {self.btc_crash_cooldown_until.strftime('%H:%M:%S')}"
                
        except Exception as e:
            logger.debug(f"BTC crash check error: {e}")
        
        return result

    async def _check_risk_shield(self, signal_side: str) -> Dict[str, Any]:
        """
        Phase 0 Risk Shield check — MUST pass before any TA, ML, or AI runs.
        
        Gathers all required inputs (balance, drawdown, tail risk, BTC 1h change)
        and delegates to RiskShield.check_shield_status().
        
        Returns the shield result dict.
        """
        # Get tail risk score — use cached last math check if available
        tail_risk_score = 50.0  # Default neutral
        try:
            if hasattr(self.ai_filter, '_last_component_scores'):
                tail_risk_score = self.ai_filter._last_component_scores.get('tail_risk', 50.0)
        except Exception:
            pass
        
        # Get BTC 1h change
        btc_1h_change = 0.0
        try:
            ticker = await self.exchange.fetch_ticker("BTC/USDT")
            current_btc = ticker['last']
            
            # Update 1h-ago price tracking
            now = datetime.now(timezone.utc)
            if self.btc_1h_ago_price is None:
                self.btc_1h_ago_price = current_btc
                self._btc_1h_timestamp = now
            elif hasattr(self, '_btc_1h_timestamp') and (now - self._btc_1h_timestamp).total_seconds() >= 3600:
                # Slide the 1h window forward
                self.btc_1h_ago_price = current_btc
                self._btc_1h_timestamp = now
            
            if self.btc_1h_ago_price and self.btc_1h_ago_price > 0:
                btc_1h_change = (current_btc - self.btc_1h_ago_price) / self.btc_1h_ago_price
        except Exception as e:
            logger.debug(f"BTC 1h change check error: {e}")
        
        # Call the Shield
        shield_result = self.risk_shield.check_shield_status(
            balance=self.balance,
            initial_balance=self.initial_balance,
            peak_balance=self.peak_balance,
            signal_side=signal_side,
            tail_risk_score=tail_risk_score,
            btc_1h_change=btc_1h_change,
        )
        
        # If HARD breach + auto-liquidate enabled → trigger emergency liquidation
        if (shield_result.get('hard_breach_active') and 
            self.risk_shield.enable_auto_liquidate and 
            not self.risk_shield._liquidation_in_progress):
            logger.critical(f"🚨 HARD BREACH + AUTO-LIQUIDATE ON — triggering emergency liquidation")
            liquidation_result = await self.risk_shield.emergency_liquidate(
                positions=self.positions,
                close_func=self._close_position_by_symbol,
                reason=shield_result['reason']
            )
            if self.telegram.enabled:
                closed = liquidation_result['positions_closed']
                failed = liquidation_result['positions_failed']
                await self.telegram.send_message(
                    f"🚨🚨🚨 *EMERGENCY LIQUIDATION*\n\n"
                    f"Reason: `{shield_result['reason']}`\n"
                    f"Positions closed: `{closed}`\n"
                    f"Failed: `{failed}`\n\n"
                    f"_Trading HALTED. Use /shield reset to resume._"
                )
        
        return shield_result

    async def _emergency_liquidate_manual(self) -> Dict[str, Any]:
        """
        Manual emergency liquidation — called from /shield liquidate.
        Bypasses the enable_auto_liquidate flag.
        """
        logger.critical(f"🚨 MANUAL EMERGENCY LIQUIDATION triggered via Telegram")
        result = await self.risk_shield.emergency_liquidate(
            positions=self.positions,
            close_func=self._close_position_by_symbol,
            reason="Manual emergency liquidation via /shield liquidate"
        )
        # Force hard breach so no new trades open
        self.risk_shield.hard_breach_active = True
        self.risk_shield.shield_engaged = True
        self.risk_shield.hard_breach_reason = "Manual emergency liquidation"
        self.risk_shield._save_state()
        return result

    def _calculate_technical_score(self, regime_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate a technical signal quality score (0-100).
        
        Factors:
        - ADX strength (0-30 points): Higher ADX = stronger trend
        - Hurst exponent (0-20 points): >0.5 = trending, <0.5 = mean-reverting
        - Volume ratio (0-20 points): Higher volume = more conviction
        - RSI position (0-15 points): Not overbought/oversold
        - Regime clarity (0-15 points): Clear trend or clear ranging
        """
        score = 0
        factors = []
        adx_score = 0
        hurst_score = 0
        vol_score = 0
        rsi_score = 0
        regime_score = 0
        
        try:
            # Get indicators from bars
            df = self.bars_agg
            adx = regime_info.get('adx', 0) if regime_info else 0
            hurst = regime_info.get('hurst', 0.5) if regime_info else 0.5
            regime = regime_info.get('regime', 'UNKNOWN') if regime_info else 'UNKNOWN'
            
            rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
            volume_ratio = df['volume_ratio'].iloc[-1] if 'volume_ratio' in df.columns else 1.0
            
            # ADX Score (0-30): Trend strength
            if adx >= 40:
                adx_score = 30
                factors.append(f"Strong ADX ({adx:.0f})")
            elif adx >= 30:
                adx_score = 25
            elif adx >= 25:
                adx_score = 20
            elif adx >= 20:
                adx_score = 10
            else:
                adx_score = 5
                factors.append(f"Weak ADX ({adx:.0f})")
            score += adx_score
            
            # Hurst Score (0-20): Trendiness
            if hurst >= 0.6:
                hurst_score = 20
                factors.append(f"Trending (H={hurst:.2f})")
            elif hurst >= 0.5:
                hurst_score = 15
            elif hurst >= 0.4:
                hurst_score = 10
            else:
                hurst_score = 5
                factors.append(f"Choppy (H={hurst:.2f})")
            score += hurst_score
            
            # Volume Score (0-20): Conviction
            if volume_ratio >= 2.0:
                vol_score = 20
                factors.append(f"High volume ({volume_ratio:.1f}x)")
            elif volume_ratio >= 1.5:
                vol_score = 15
            elif volume_ratio >= 1.0:
                vol_score = 10
            elif volume_ratio >= 0.5:
                vol_score = 5
            else:
                vol_score = 0
                factors.append(f"Low volume ({volume_ratio:.1f}x)")
            score += vol_score
            
            # RSI Score (0-15): Not at extremes
            if 40 <= rsi <= 60:
                rsi_score = 15
            elif 30 <= rsi <= 70:
                rsi_score = 10
            elif 20 <= rsi <= 80:
                rsi_score = 5
            else:
                rsi_score = 0
                factors.append(f"RSI extreme ({rsi:.0f})")
            score += rsi_score
            
            # Regime Score (0-15): Clear market state
            if regime in ['TRENDING', 'STRONG_TREND']:
                regime_score = 15
            elif regime == 'RANGING':
                regime_score = 10
            elif regime == 'WEAK_TREND':
                regime_score = 5
            else:
                regime_score = 0
                factors.append(f"Unclear regime")
            score += regime_score
            
        except Exception as e:
            logger.debug(f"Technical score error: {e}")
            score = 50  # Default middle score
            factors.append("Error calculating")
        
        # Quality label
        if score >= 80:
            quality = "EXCELLENT"
        elif score >= 65:
            quality = "GOOD"
        elif score >= 50:
            quality = "MODERATE"
        elif score >= 35:
            quality = "WEAK"
        else:
            quality = "POOR"
        
        return {
            'score': score,
            'quality': quality,
            'factors': factors,
            'breakdown': {
                'adx': adx_score,
                'hurst': hurst_score,
                'volume': vol_score,
                'rsi': rsi_score,
                'regime': regime_score
            }
        }

    def _calculate_system_score(
        self,
        tech_score: Dict[str, Any],
        ml_result: Dict[str, Any],
        regime_info: Dict[str, Any],
        signal: int = 0  # NEW: Signal direction for short bonus
    ) -> Dict[str, Any]:
        """
        Calculate combined system score (0-100).
        
        Weights:
        - Technical score: 50%
        - ML prediction: 30%
        - Regime alignment: 20%
        
        This gives AI a single "system confidence" number.
        
        IMPROVEMENT: Adds SHORT_SCORE_BONUS when signal is short (-1)
        Based on backtest data: shorts 62.5% win vs longs 52% win
        """
        # Technical component (0-50)
        tech_component = tech_score['score'] * 0.50
        
        # ML component (0-30)
        if ml_result.get('ml_available'):
            # Convert probability (0.3-0.7 range typically) to 0-100 scale
            ml_prob = ml_result['ml_win_probability']
            # Scale: 0.40=0, 0.50=50, 0.60=100
            ml_scaled = min(100, max(0, (ml_prob - 0.40) * 500))
            ml_component = ml_scaled * 0.30
        else:
            ml_component = 15  # Neutral if ML not available
        
        # Regime alignment component (0-20)
        regime = regime_info.get('regime', 'UNKNOWN') if regime_info else 'UNKNOWN'
        strategy = regime_info.get('strategy', 'NONE') if regime_info else 'NONE'
        
        if regime in ['TRENDING', 'STRONG_TREND'] and strategy == 'TREND_FOLLOWING':
            regime_component = 20  # Perfect alignment
        elif regime == 'RANGING' and strategy == 'MEAN_REVERSION':
            regime_component = 18  # Good alignment
        elif regime == 'WEAK_TREND':
            regime_component = 10  # Acceptable
        elif regime == 'CHOPPY' or strategy == 'NO_TRADE':
            regime_component = 0  # Should not trade
        else:
            regime_component = 10  # Unknown = neutral
        
        combined = tech_component + ml_component + regime_component
        
        # === SHORT BIAS BONUS ===
        # Based on backtest analysis: shorts have 62.5% win rate vs 52% for longs
        # Only apply when shorts are actually allowed (skip if allowed_sides == 'long')
        short_bonus = 0
        if signal == -1 and getattr(self, 'allowed_sides', 'both').lower() != 'long':
            short_bonus = self.SHORT_SCORE_BONUS
            combined += short_bonus
        
        # Cap at 100
        combined = min(100, combined)
        
        # Recommendation based on combined score
        if combined >= 75:
            recommendation = "STRONG_BUY"
        elif combined >= 60:
            recommendation = "BUY"
        elif combined >= 45:
            recommendation = "NEUTRAL"
        elif combined >= 30:
            recommendation = "WEAK"
        else:
            recommendation = "AVOID"
        
        # Build breakdown string with short bonus if applied
        if short_bonus > 0:
            breakdown = f"Tech:{tech_component:.0f} + ML:{ml_component:.0f} + Regime:{regime_component:.0f} + ShortBonus:{short_bonus}"
        else:
            breakdown = f"Tech:{tech_component:.0f} + ML:{ml_component:.0f} + Regime:{regime_component:.0f}"
        
        return {
            'combined': combined,
            'recommendation': recommendation,
            'tech_component': tech_component,
            'ml_component': ml_component,
            'regime_component': regime_component,
            'short_bonus': short_bonus,
            'breakdown': breakdown
        }

    # === PRE-FILTER CONFIGURATION ===
    # Optimized based on backtest data analysis (2026-01-10)
    # Key findings:
    # - WEAK_TRENDING: 80% win rate (BEST!)
    # - ADX 25-35: 66-69% win rate (sweet spot)
    # - ADX 35-40: 22% win rate (DANGER ZONE - AVOID)
    # - Volume >= 1.5x: 100% win rate
    # - Volume >= 1.0x: 61% win rate
    
    # === PhD-LEVEL REGIME-ADAPTIVE THRESHOLDS ===
    # Calibrated based on statistical analysis:
    # - TRENDING markets: Lower threshold (strong edge, let trends run)
    # - RANGING/CHOPPY: Higher threshold (edge is weaker, be selective)
    # - VOLATILE: Highest threshold (noise dominates, only best setups)
    # 
    # Mathematical justification:
    # - Round-trip cost = 0.4% (fees + slippage)
    # - Minimum edge needed = 0.8% (2x costs for positive expectancy)
    # - Score maps to expected edge: score 50 = ~0.5% edge, 60 = ~1% edge
    # - Regime modifies edge: TRENDING +30%, VOLATILE -40%
    MIN_SCORE_THRESHOLDS = {
        'TRENDING': 45,       # Strong edge, trend continuation likely
        'STRONG_TREND': 40,   # Very strong edge, lower threshold OK
        'WEAK_TREND': 50,     # Moderate edge, standard threshold
        'WEAK_TRENDING': 50,  # Moderate edge
        'CHOPPY': 70,         # Weak edge, be very selective
        'RANGING': 60,        # Mean-reversion, need higher conviction
        'VOLATILE': 65,       # High noise, only best setups
        'UNKNOWN': 55         # Default conservative
    }
    
    # Minimum ADX for each regime (ensures trend strength)
    # REDUCED FOR FASTER SAMPLE COLLECTION
    MIN_ADX_THRESHOLDS = {
        'TRENDING': 20,       # Lowered from 25
        'STRONG_TREND': 20,   # Lowered from 25
        'WEAK_TREND': 18,     # Lowered from 25
        'WEAK_TRENDING': 18,  # Lowered from 25
        'CHOPPY': 30,         # Lowered from 40
        'RANGING': 15,        # Lowered from 20
        'VOLATILE': 20,       # Lowered from 25
        'UNKNOWN': 20         # Lowered from 25
    }
    
    # ADX danger zone - avoid ADX 35-40 (only 22% win rate)
    # DISABLED FOR SAMPLE COLLECTION - We'll let AI filter this
    ADX_DANGER_ZONE = None  # Was (35, 40), now disabled
    
    # Volume requirement - LOWERED for more trades
    MIN_VOLUME_RATIO = 0.7  # Lowered from 0.9

    def _apply_pre_filters(
        self,
        signal: int,
        price: float,
        atr: float,
        regime_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply fast pre-filters to reject low-quality signals early.
        
        This saves API calls to Gemini for obviously bad signals.
        
        Filters:
        1. Regime-aware minimum score threshold
        2. Minimum ADX requirement by regime
        3. Volume confirmation (must have above-average volume)
        4. Confluence check (multiple indicators agreeing)
        """
        result = {'passed': True, 'reason': None, 'filters_checked': []}
        regime = regime_info.get('regime', 'UNKNOWN') if regime_info else 'UNKNOWN'
        
        # NOTE: total_signals and by_regime tracking happens earlier in main loop to avoid double-counting
        # This only tracks the blocking reasons
        
        # === HARD RULES (AI CANNOT OVERRIDE) ===
        # These are non-negotiable filters based on data showing noise-level stops fail
        
        # HARD RULE 1: Minimum ATR % (volatility check)
        # If ATR is too small relative to price, stops will be noise-level
        atr_pct = atr / price if price > 0 else 0
        result['filters_checked'].append(f"atr_pct={atr_pct:.3%}>={self.MIN_ATR_PCT:.1%}")
        
        if atr_pct < self.MIN_ATR_PCT:
            result['passed'] = False
            result['reason'] = f"HARD REJECT: Volatility too low ({atr_pct:.3%} < {self.MIN_ATR_PCT:.1%} min). Stops would be noise-level."
            self.prefilter_stats['blocked_low_volatility'] = self.prefilter_stats.get('blocked_low_volatility', 0) + 1
            logger.warning(f"🚫 HARD FILTER: ATR% too low ({atr_pct:.3%}) - stops would be noise")
            return result
        
        try:
            df = self.bars_agg
            if len(df) < 10:
                self.prefilter_stats['passed'] += 1
                self.prefilter_stats['by_regime'][regime]['passed'] += 1
                return result  # Not enough data, let AI decide
            
            latest = df.iloc[-1]
            
            # === FILTER 1: Regime-specific minimum score ===
            # Quick pre-calculation of technical score components
            adx = latest.get('adx', 0) if 'adx' in df.columns else 0
            rsi = latest.get('rsi', 50) if 'rsi' in df.columns else 50
            volume_ratio = latest.get('volume_ratio', 1) if 'volume_ratio' in df.columns else 1
            
            quick_score = 0
            # ADX contribution
            quick_score += min(30, adx * 1.0) if adx >= 20 else adx * 0.5
            # RSI contribution (not extreme = good)
            if 35 <= rsi <= 65:
                quick_score += 15
            elif 25 <= rsi <= 75:
                quick_score += 10
            # Volume contribution
            quick_score += min(15, volume_ratio * 7) if volume_ratio >= 1 else volume_ratio * 5
            
            min_threshold = self.MIN_SCORE_THRESHOLDS.get(regime, 50)
            result['filters_checked'].append(f"score={quick_score:.0f}>={min_threshold}")
            
            if quick_score < min_threshold:
                result['passed'] = False
                result['reason'] = f"Score too low for {regime} regime ({quick_score:.0f} < {min_threshold})"
                self.prefilter_stats['blocked_score'] += 1
                return result
            
            # === FILTER 2: ADX threshold by regime ===
            min_adx = self.MIN_ADX_THRESHOLDS.get(regime, 25)
            result['filters_checked'].append(f"adx={adx:.0f}>={min_adx}")
            
            if adx < min_adx:
                result['passed'] = False
                result['reason'] = f"ADX too weak for {regime} regime ({adx:.0f} < {min_adx})"
                self.prefilter_stats['blocked_adx_low'] += 1
                return result
            
            # === FILTER 2b: ADX DANGER ZONE check - DISABLED FOR SAMPLE COLLECTION ===
            # We let AI handle this instead of hard-blocking
            if self.ADX_DANGER_ZONE and self.ADX_DANGER_ZONE[0] <= adx <= self.ADX_DANGER_ZONE[1]:
                result['filters_checked'].append(f"adx_danger_zone={adx:.0f}")
                result['passed'] = False
                result['reason'] = f"ADX in DANGER ZONE ({adx:.0f} is in {self.ADX_DANGER_ZONE[0]}-{self.ADX_DANGER_ZONE[1]}, only 22% win rate)"
                self.prefilter_stats['blocked_adx_danger'] += 1
                return result
            
            # === FILTER 3: Volume confirmation ===
            result['filters_checked'].append(f"vol={volume_ratio:.1f}>={self.MIN_VOLUME_RATIO}")
            
            if volume_ratio < self.MIN_VOLUME_RATIO:  # Below minimum threshold
                result['passed'] = False
                result['reason'] = f"Volume too low ({volume_ratio:.1f}x < {self.MIN_VOLUME_RATIO}x minimum)"
                self.prefilter_stats['blocked_volume'] += 1
                return result
            
            # === FILTER 4: Confluence check (at least 2/3 must agree) ===
            confluence_count = 0
            
            # Check 1: RSI agreement with signal direction
            if signal == 1 and rsi < 65:  # Long with RSI not overbought
                confluence_count += 1
            elif signal == -1 and rsi > 35:  # Short with RSI not oversold
                confluence_count += 1
            
            # Check 2: ADX shows trend strength
            if adx >= 25:
                confluence_count += 1
            
            # Check 3: Volume confirms interest
            if volume_ratio >= 1.0:
                confluence_count += 1
            
            result['filters_checked'].append(f"confluence={confluence_count}/3")
            
            if confluence_count < 2:
                result['passed'] = False
                result['reason'] = f"Insufficient confluence ({confluence_count}/3 indicators agreeing)"
                self.prefilter_stats['blocked_confluence'] += 1
                return result
            
            # === FILTER 5: Divergence filter (avoid trading against momentum) ===
            # Based on analysis: entries with price/RSI divergence had higher failure rate
            try:
                from indicator import calculate_momentum_divergence
                divergence_data = calculate_momentum_divergence(self.current_df)
                rsi_divergence = divergence_data.get('rsi_divergence', 0.0)
                
                # Block longs when bearish divergence (price up, RSI down)
                # Block shorts when bullish divergence (price down, RSI up)
                divergence_threshold = 3.0  # Moderate threshold
                
                if signal == 1 and rsi_divergence < -divergence_threshold:
                    result['filters_checked'].append(f"divergence={rsi_divergence:.1f}")
                    result['passed'] = False
                    result['reason'] = f"Bearish divergence detected ({rsi_divergence:.1f}), blocking long"
                    self.prefilter_stats['blocked_divergence'] += 1
                    return result
                elif signal == -1 and rsi_divergence > divergence_threshold:
                    result['filters_checked'].append(f"divergence={rsi_divergence:.1f}")
                    result['passed'] = False
                    result['reason'] = f"Bullish divergence detected ({rsi_divergence:.1f}), blocking short"
                    self.prefilter_stats['blocked_divergence'] += 1
                    return result
                else:
                    result['filters_checked'].append(f"divergence_ok={rsi_divergence:.1f}")
            except Exception as e:
                logger.debug(f"Divergence check skipped: {e}")
            
            # All filters passed
            self.prefilter_stats['passed'] += 1
            self.prefilter_stats['by_regime'][regime]['passed'] += 1
            logger.debug(f"✅ Pre-filters PASSED: {', '.join(result['filters_checked'])}")
            return result
            
        except Exception as e:
            logger.debug(f"Pre-filter error: {e}")
            # On error, let the signal through for AI to evaluate
            return result

    def _log_decision_alignment(
        self,
        ml_result: Dict[str, Any],
        ai_result: Dict[str, Any],
        system_score: Dict[str, Any]
    ):
        """Log how different system components align on the decision."""
        ml_available = ml_result.get('ml_available', False)
        ml_favorable = ml_result.get('ml_win_probability', 0.5) >= 0.55 if ml_available else None
        ai_approved = ai_result.get('approved', False)
        system_favorable = system_score['combined'] >= 60
        
        if not ml_available:
            if ai_approved:
                logger.info(f"🤖 AI APPROVED (ML not available) | System: {system_score['combined']:.0f}")
            else:
                logger.info(f"🤖 AI REJECTED (ML not available) | System: {system_score['combined']:.0f}")
            return
        
        # All three align
        if ml_favorable == ai_approved == system_favorable:
            if ai_approved:
                logger.info(f"✅ FULL ALIGNMENT: ML+AI+System all approve ({system_score['combined']:.0f}/100)")
            else:
                logger.info(f"⛔ FULL ALIGNMENT: ML+AI+System all reject ({system_score['combined']:.0f}/100)")
        # AI overrides
        elif ai_approved and not ml_favorable:
            logger.info(f"🤖 AI OVERRIDE: Approved despite weak ML ({ml_result['ml_win_probability']:.1%})")
        elif not ai_approved and ml_favorable:
            logger.info(f"🤖 AI VETO: Rejected despite favorable ML ({ml_result['ml_win_probability']:.1%})")
        # Mixed signals
        else:
            logger.info(f"⚠️ MIXED: AI={ai_approved}, ML={ml_favorable}, System={system_favorable}")

    # NOTE: _check_signal_reversal was removed - redundant with existing systems:
    # - unified_position_decision() handles momentum reversal detection
    # - _ai_position_monitor_all() monitors position health with AI
    # - profit_reversal_detected in ai_filter catches momentum against profit
    # These systems already close positions when momentum reverses against them.

    async def _process_signal(self, signal: int, price: float, atr: float, regime_info: Dict[str, Any] = None):
        """
        Process a trading signal through the complete decision pipeline.
        
        DECISION FLOW (Shield → Pre-filters → ML → AI):
        0. RISK SHIELD: Circuit breaker wall — blocks before ANY analysis runs
        1. PRE-FILTERS: Minimum score and regime checks
        2. MATH SIGNAL: Technical indicators generate raw signal (+1/-1)
        3. ML SCORING: XGBoost predicts win probability based on historical patterns
        4. AI DECISION: Gemini AI makes FINAL decision with all context
        
        The AI sees:
        - Technical signal strength and regime
        - ML prediction and confidence
        - Combined system score
        - Trading performance history
        """
        side = "LONG" if signal == 1 else "SHORT"
        strategy = regime_info.get('strategy', 'TREND_FOLLOWING') if regime_info else 'TREND_FOLLOWING'
        
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 0: RISK SHIELD — THE WALL
        # Nothing passes without clearance. This runs BEFORE any TA/ML/AI.
        # Checks: daily loss, weekly loss, max drawdown, tail risk, BTC crash,
        #         loss cooldown. Hard breaches halt ALL trading.
        # ═══════════════════════════════════════════════════════════════════
        shield_status = await self._check_risk_shield(signal_side=side)
        if not shield_status['allowed']:
            severity_emoji = "🔴" if shield_status.get('hard_breach_active') else "🛡️"
            logger.warning(f"{severity_emoji} Signal {side} BLOCKED by Risk Shield: {shield_status['reason']}")
            if shield_status.get('hard_breach_active') and self.telegram.enabled:
                await self.telegram.send_message(
                    f"🔴 *RISK SHIELD — HARD BREACH*\n\n"
                    f"Signal: `{side}` blocked\n"
                    f"Reason: `{shield_status['reason']}`\n"
                    f"Drawdown: `{shield_status['drawdown_pct']:.1%}`\n\n"
                    f"_All trading HALTED. /shield reset to resume._"
                )
            return
        
        # === PHASE 0.1: PER-SYMBOL LOSS COOLDOWN (Notebook #2 §14) ===
        symbol_key = self.SYMBOL.replace('/', '').replace(':USDT', '').upper()
        if symbol_key in self.pair_loss_cooldown:
            cooldown_until = self.pair_loss_cooldown[symbol_key]
            if datetime.now(timezone.utc) < cooldown_until:
                remaining = (cooldown_until - datetime.now(timezone.utc)).total_seconds() / 60
                logger.info(f"⏰ Signal {side} BLOCKED: {symbol_key} on {self.PAIR_LOSS_COOLDOWN_MINUTES}min cooldown ({remaining:.0f}min left)")
                return
            else:
                del self.pair_loss_cooldown[symbol_key]
                logger.info(f"✅ {symbol_key} cooldown expired — can trade again")
        
        # === PHASE 0.15: RE-ENTRY COOLDOWN (prevent re-entering same symbol+direction too fast) ===
        reentry_key = f"{symbol_key}_{side}"
        if reentry_key in self.reentry_cooldown:
            reentry_until = self.reentry_cooldown[reentry_key]
            if datetime.now(timezone.utc) < reentry_until:
                remaining_s = (reentry_until - datetime.now(timezone.utc)).total_seconds()
                logger.info(f"⏰ RE-ENTRY BLOCKED: {reentry_key} — wait {remaining_s:.0f}s before re-entering same direction")
                return
            else:
                del self.reentry_cooldown[reentry_key]
                logger.info(f"✅ {reentry_key} re-entry cooldown expired")
        
        # === PHASE 0.5: PRE-FILTERS (Fast rejection for low-quality signals) ===
        pre_filter_result = self._apply_pre_filters(signal, price, atr, regime_info)
        if not pre_filter_result['passed']:
            logger.info(f"🚫 Signal {side} PRE-FILTERED: {pre_filter_result['reason']}")
            return
        
        # === PHASE 0.6: FEAR & GREED GATE (matches scanner Stage 3 thresholds) ===
        try:
            _fg_ctx = self.ai_filter._get_news_context()
            _fg_val = _fg_ctx.get('fear_greed_index', 50)
            if _fg_val < 5:
                logger.warning(f"🚫 Signal {side} F&G BLOCKED: Fear & Greed = {_fg_val} (TRUE PANIC < 5) — all trades blocked")
                return
            elif _fg_val < 10 and side == "LONG":
                logger.warning(f"🚫 Signal {side} F&G BLOCKED: Fear & Greed = {_fg_val} (EXTREME FEAR) — LONGs blocked")
                return
            elif _fg_val > 80 and side == "LONG":
                logger.warning(f"🚫 Signal {side} F&G BLOCKED: Fear & Greed = {_fg_val} (EXTREME GREED) — LONGs at tops blocked")
                return
            elif _fg_val > 90 and side == "SHORT":
                logger.warning(f"🚫 Signal {side} F&G BLOCKED: Fear & Greed = {_fg_val} (PEAK GREED) — SHORTs during euphoria blocked")
                return
            else:
                logger.info(f"✅ F&G check passed: {_fg_val} — {side} allowed")
        except Exception as fg_err:
            logger.warning(f"⚠️ F&G check failed: {fg_err} — proceeding without gate")
        
        # === PHASE 1: CALCULATE TECHNICAL SCORE ===
        tech_score = self._calculate_technical_score(regime_info)
        logger.info(f"📈 Technical Score: {tech_score['score']:.0f}/100 ({tech_score['quality']})")
        
        # === PHASE 2: DUAL ML PREDICTION ===
        ml_result = {'ml_available': False, 'ml_win_probability': 0.5}
        ml_insight = {'ml_available': False}
        
        # Build comprehensive feature dict from current market state
        now_utc = datetime.now(timezone.utc)
        _rsi = self.bars_agg['rsi'].iloc[-1] if 'rsi' in self.bars_agg.columns else 50
        _adx = self.bars_agg['adx'].iloc[-1] if 'adx' in self.bars_agg.columns else 0
        _vol_ratio = self.bars_agg['volume_ratio'].iloc[-1] if 'volume_ratio' in self.bars_agg.columns else 1
        _hurst = self.bars_agg['hurst'].iloc[-1] if 'hurst' in self.bars_agg.columns else 0.5
        _regime = regime_info.get('regime', 'UNKNOWN') if regime_info else 'UNKNOWN'
        
        ml_features = {
            'atr_percent': (atr / price) * 100 if price > 0 else 0,
            'rsi': _rsi,
            'adx': _adx,
            'volume_ratio': _vol_ratio,
            'hurst': _hurst,
            'sma_distance_percent': 0,
            'hour': now_utc.hour,
            'day_of_week': now_utc.weekday(),
            'is_weekend': 1 if now_utc.weekday() >= 5 else 0,
            'regime': _regime,
            'regime_trending': 1 if _regime == 'TRENDING' else 0,
            'regime_choppy': 1 if _regime == 'CHOPPY' else 0,
            'regime_ranging': 1 if _regime not in ('TRENDING', 'CHOPPY') else 0,
            'is_long': 1 if signal == 1 else 0,
            'rsi_slope': float(self.bars_agg['rsi'].diff(5).iloc[-1] / 5) if 'rsi' in self.bars_agg.columns and len(self.bars_agg) > 5 else 0,
            'macd_hist': float(self.bars_agg['macd_hist'].iloc[-1]) if 'macd_hist' in self.bars_agg.columns else 0,
            'price_momentum': float(self.bars_agg['close'].pct_change(5).iloc[-1] * 100) if len(self.bars_agg) > 5 else 0,
            'atr_expansion': 1.0,
            'bb_position': 0.5,
            'volume_trend': float(self.bars_agg['volume'].pct_change(10).iloc[-1] * 100) if 'volume' in self.bars_agg.columns and len(self.bars_agg) > 10 else 0,
            'btc_correlation': 0,
            'candle_strength': 0,
            'higher_highs': 0,
            'distance_from_support': 0.5,
            'distance_from_resistance': 0.5,
            'ai_confidence': 0,  # Will be updated after AI decision
            'system_score': 0,   # Will be updated after system score
            'tech_score': tech_score['score'],
            'is_london_session': 1 if 7 <= now_utc.hour <= 16 else 0,
            'is_nyc_session': 1 if 13 <= now_utc.hour <= 22 else 0,
            'is_asia_session': 1 if 0 <= now_utc.hour <= 9 else 0,
        }
        
        # --- Historical ML prediction ---
        hist_pred = self.ml_historical.predict(ml_features)
        if hist_pred.get('available'):
            logger.info(f"🧠 Historical ML: {hist_pred['win_probability']:.1%} win prob ({hist_pred['confidence']})")
        
        # --- Live ML prediction ---
        live_pred = self.ml_live.predict(ml_features)
        if live_pred.get('available'):
            logger.info(f"🧠 Live ML: {live_pred['win_probability']:.1%} win prob ({live_pred['confidence']})")
        
        # --- Legacy ML predictor (backward compat) ---
        if self.ml_predictor.is_loaded:
            ml_result = self.ml_predictor.predict(ml_features)
        
        # Combine dual ML into a single insight for AI
        any_ml_available = hist_pred.get('available') or live_pred.get('available') or ml_result.get('ml_available')
        
        if any_ml_available:
            # Weighted average: Historical=0.4, Live=0.6 (live learns YOUR patterns)
            # If one is unavailable, use the other at 100%
            hist_prob = hist_pred.get('win_probability', 0.5)
            live_prob = live_pred.get('win_probability', 0.5)
            hist_avail = hist_pred.get('available', False)
            live_avail = live_pred.get('available', False)
            
            if hist_avail and live_avail:
                combined_prob = hist_prob * 0.4 + live_prob * 0.6
                ml_source = 'dual'
            elif hist_avail:
                combined_prob = hist_prob
                ml_source = 'historical'
            elif live_avail:
                combined_prob = live_prob
                ml_source = 'live'
            else:
                combined_prob = ml_result.get('ml_win_probability', 0.5)
                ml_source = 'legacy'
            
            logger.info(f"🧠 Combined ML: {combined_prob:.1%} win prob (source: {ml_source})")
            
            # === ML VETO GATE (matches scanner path threshold) ===
            from ai_filter import get_threshold_mode as _get_tm2
            _ml_veto_thr = 0.20 if _get_tm2() == 'testing' else 0.40
            if combined_prob < _ml_veto_thr:
                logger.warning(f"🚫 Signal {side} ML VETOED: Combined ML probability {combined_prob:.1%} < {_ml_veto_thr:.0%} threshold")
                return
            
            ml_insight = {
                'ml_available': True,
                'ml_win_probability': combined_prob,
                'ml_confidence': 'HIGH' if combined_prob >= 0.65 else ('LOW' if combined_prob < 0.45 else 'MEDIUM'),
                'ml_accuracy': self.ml_historical.metrics.get('accuracy', 0.5) if hist_avail else self.ml_live.metrics.get('accuracy', 0.5),
                'ml_samples': (self.ml_historical.metrics.get('total_samples', 0) + len(self.ml_live.samples)),
                'ml_influence': 0.0,
                # Dual ML details for AI prompt
                'historical_ml': hist_pred,
                'live_ml': live_pred,
                'ml_source': ml_source,
            }
        
        # === PHASE 3: CALCULATE COMBINED SYSTEM SCORE ===
        system_score = self._calculate_system_score(
            tech_score=tech_score,
            ml_result=ml_result,
            regime_info=regime_info,
            signal=signal  # Pass signal for short bonus calculation
        )
        logger.info(f"🎯 System Score: {system_score['combined']:.0f}/100 ({system_score['recommendation']})")
        
        # === PHASE 3.5: GET MARKET SCANNER CONTEXT FOR AI ALIGNMENT ===
        market_scanner_context = self._get_market_scanner_context_for_ai()
        
        institutional_data = {}
        
        
        # === PHASE 4: AI FINAL DECISION (with complete context) ===
        ai_result = self.ai_filter.analyze_signal(
            signal=signal,
            df=self.bars_agg,
            current_price=price,
            atr=atr,
            symbol=self.SYMBOL,
            ml_insight=ml_insight,
            system_score=system_score,
            market_scanner_context=market_scanner_context,
            tech_score=tech_score,
            institutional_data=institutional_data
        )
        
        # Log decision alignment
        self._log_decision_alignment(ml_result, ai_result, system_score)
        
        # === RECORD AI DECISION FOR TRACKING ===
        self.current_decision_id = self.ai_tracker.record_decision(
            symbol=self.SYMBOL,
            signal_direction=side,
            price=price,
            approved=ai_result["approved"],
            confidence=ai_result["confidence"],
            reasoning=ai_result.get("reasoning", ""),
            threshold_used=ai_result.get("threshold_used", self.ai_filter.confidence_threshold),
            regime=regime_info.get('regime', 'UNKNOWN') if regime_info else 'UNKNOWN',
            tech_score=tech_score['score'],
            system_score=system_score['combined'],
            ml_probability=ml_result.get('ml_win_probability', 0.5),
            ml_confidence=ml_result.get('ml_confidence', 'N/A')
        )
        
        # Record signal in history (complete data for analysis)
        self.signal_history.append({
            "direction": side.upper(),  # Dashboard expects 'direction'
            "side": side,
            "entry": price,  # Dashboard expects 'entry'
            "price": price,
            "approved": ai_result["approved"],
            "executed": ai_result["approved"],  # Dashboard expects 'executed'
            "rejected": not ai_result["approved"],  # Dashboard expects 'rejected'
            "confidence": ai_result["confidence"],
            "ai_decision": "APPROVED" if ai_result["approved"] else "REJECTED",
            "strategy": strategy,
            "regime": regime_info.get('regime', 'UNKNOWN') if regime_info else 'UNKNOWN',
            "tech_score": tech_score['score'],
            "ml_probability": ml_result.get('ml_win_probability', 0.5),
            "ml_confidence": ml_result.get('ml_confidence', 'N/A'),
            "system_score": system_score['combined'],
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S")
        })
        # Keep only last 50 signals
        if len(self.signal_history) > 50:
            self.signal_history = self.signal_history[-50:]
        
        # Notify via Telegram
        if self.telegram.enabled:
            await self.telegram.notify_signal(
                symbol=self.SYMBOL,
                side=side,
                price=price,
                ai_approved=ai_result["approved"],
                confidence=ai_result["confidence"],
                reasoning=ai_result["reasoning"]
            )
        
        # Execute if approved
        if ai_result["approved"]:
            await self._open_position(signal, price, atr, ai_result=ai_result)
        else:
            logger.info(f"Signal {side} REJECTED by AI filter: {ai_result['reasoning']}")
            # Notify about AI filtering decision
            await self._notify_ai_decision("signal_filtered", {
                "signal": side,
                "price": price,
                "confidence": ai_result["confidence"],
                "reasoning": ai_result["reasoning"]
            })
    
    # ============================================================================
    # MATH-LEVEL AUTONOMOUS TRADING ENHANCEMENTS
    # ============================================================================
    
    async def _validate_position_setup_math(self, signal: int, current_price: float, 
                                            df: pd.DataFrame) -> Dict[str, Any]:
        """
        Math-level validation before opening position.
        
        Checks:
        1. Is current market regime suitable?
        2. Will volatility spike soon?
        3. Is edge statistically significant?
        
        Returns dict with approved/rejected + reasoning
        """
        from indicator import (
            calculate_regime_hmm, 
            calculate_garch_volatility,
            calculate_hypothesis_test
        )
        
        validation = {
            'approved': True,
            'reasons_for': [],
            'reasons_against': [],
            'regime_ok': True,
            'vol_ok': True,
            'edge_ok': True,
            'size_adjustment': 1.0
        }
        
        try:
            # === CHECK 1: Market Regime ===
            returns = df['close'].pct_change().dropna()
            if len(returns) < 10:
                validation['reasons_for'].append("Insufficient data for regime check")
                return validation
            
            regime_result = calculate_regime_hmm(returns, n_regimes=3)
            
            if regime_result.get('status') == 'success':
                regime = regime_result.get('regime', 'NORMAL')
                confidence = regime_result.get('confidence', 0)
                
                if regime == 'VOLATILE':
                    validation['regime_ok'] = False
                    validation['reasons_against'].append(
                        f"Market in VOLATILE regime ({confidence:.0%} confidence) - reduce position 30%"
                    )
                    validation['size_adjustment'] *= 0.7
                elif regime == 'CALM':
                    validation['reasons_for'].append(
                        f"Market in CALM regime ({confidence:.0%} confidence) - good for breakouts"
                    )
                else:
                    validation['reasons_for'].append(f"Market in {regime} regime")
            
            # === CHECK 2: Volatility Forecast ===
            vol_result = calculate_garch_volatility(returns)
            
            if vol_result.get('status') == 'success':
                vol_trend = vol_result.get('vol_trend', 'stable')
                vol_change = vol_result.get('vol_change_pct', 0)
                
                if vol_trend == 'increasing' and vol_change > 30:
                    # Severe volatility spike - high penalty
                    validation['vol_ok'] = False
                    validation['reasons_against'].append(
                        f"Volatility SURGE forecasted ({vol_change:+.1f}%) - reduce position 40%"
                    )
                    validation['size_adjustment'] *= 0.6
                elif vol_trend == 'increasing' and vol_change > 15:
                    # Moderate volatility increase - mild penalty
                    validation['reasons_against'].append(
                        f"Volatility increase forecasted ({vol_change:+.1f}%) - reduce position 15%"
                    )
                    validation['size_adjustment'] *= 0.85
                elif vol_trend == 'decreasing':
                    validation['reasons_for'].append(
                        f"Volatility stabilizing ({vol_change:.1f}% decrease) - good entry"
                    )
            
            # === CHECK 3: Edge Significance ===
            if len(returns) >= 30:
                hypo_result = calculate_hypothesis_test(returns)
                p_value = hypo_result.get('p_value', 1.0)
                
                if p_value > 0.40:
                    # Very weak edge - significant penalty
                    validation['reasons_against'].append(
                        f"Edge very weak (p={p_value:.3f} > 0.40) - reduce position 40%"
                    )
                    validation['size_adjustment'] *= 0.6
                elif p_value > 0.25:
                    # Weak edge - mild penalty
                    validation['reasons_against'].append(
                        f"Edge weak (p={p_value:.3f} > 0.25) - reduce position 20%"
                    )
                    validation['size_adjustment'] *= 0.8
                elif p_value < 0.10:
                    # Strong edge - boost confidence
                    validation['reasons_for'].append(
                        f"Edge is strong (p={p_value:.4f}) - high confidence"
                    )
                    validation['size_adjustment'] *= 1.2  # Boost confident trades
                else:
                    # Moderate edge - normal position
                    validation['reasons_for'].append(
                        f"Edge is moderate (p={p_value:.3f}) - normal position"
                    )
            
            # Final decision
            if not validation['edge_ok']:
                validation['approved'] = False
            
        except Exception as e:
            logger.warning(f"Math validation error: {e} - proceeding with standard checks")
        
        return validation

    async def _get_regime_adaptive_tp_levels(self, entry_price: float, stop_loss: float,
                                            df: pd.DataFrame, side: str = 'long') -> Dict[str, float]:
        """
        INTELLIGENT TP SYSTEM - Multi-factor analysis for optimal profit targets.
        
        Factors considered:
        1. Market regime (trending vs mean-reverting)
        2. Support/Resistance levels (place TPs before key levels)
        3. Recent swing highs/lows (realistic targets)
        4. ATR-normalized volatility (adjust for market conditions)
        5. Volume profile (where is liquidity?)
        6. Momentum strength (let winners run in strong trends)
        """
        from indicator import calculate_regime_hmm, calculate_garch_volatility
        import numpy as np
        
        # Calculate R-value
        r_value = abs(entry_price - stop_loss)
        is_long = side.lower() == 'long'
        
        # Default levels from adaptive params
        tp1_mult = float(self.adaptive_params.get('tp1_r', {}).get('current', 1.0))
        tp2_mult = float(self.adaptive_params.get('tp2_r', {}).get('current', 1.5))
        tp3_mult = tp2_mult * 1.8  # TP3 ~80% further than TP2
        
        adjustments = []
        
        try:
            if len(df) < 20:
                logger.debug("Insufficient data for smart TPs, using defaults")
                return self._calculate_tp_output(entry_price, r_value, tp1_mult, tp2_mult, tp3_mult, is_long, adjustments)
            
            # === 1. SUPPORT/RESISTANCE ANALYSIS ===
            # Find key levels that might act as barriers
            highs = df['high'].rolling(10).max()
            lows = df['low'].rolling(10).min()
            
            recent_high = float(df['high'].iloc[-20:].max())
            recent_low = float(df['low'].iloc[-20:].min())
            swing_range = recent_high - recent_low
            
            # Find pivot levels (local highs/lows)
            pivot_highs = []
            pivot_lows = []
            for i in range(5, len(df) - 2):
                if df['high'].iloc[i] == df['high'].iloc[i-5:i+3].max():
                    pivot_highs.append(float(df['high'].iloc[i]))
                if df['low'].iloc[i] == df['low'].iloc[i-5:i+3].min():
                    pivot_lows.append(float(df['low'].iloc[i]))
            
            # Sort and filter relevant levels
            if is_long:
                # For longs, we care about resistance above
                relevant_levels = [p for p in pivot_highs if p > entry_price]
                relevant_levels = sorted(relevant_levels)[:3]  # Top 3 nearest resistance
            else:
                # For shorts, we care about support below
                relevant_levels = [p for p in pivot_lows if p < entry_price]
                relevant_levels = sorted(relevant_levels, reverse=True)[:3]  # Top 3 nearest support
            
            # Adjust TPs if they clash with S/R levels
            if relevant_levels:
                nearest_level = relevant_levels[0]
                level_distance = abs(nearest_level - entry_price)
                level_distance_r = level_distance / r_value if r_value > 0 else 1.0
                
                # If nearest resistance/support is very close, be conservative with TP1
                # BUT never reduce below 0.6R (was allowing 0.1R which is way too tight!)
                if level_distance_r < tp1_mult * 0.9:
                    # Place TP1 just before the level (90% of distance), but minimum 0.6R
                    adjusted_mult = max(level_distance_r * 0.9, 0.6)
                    if adjusted_mult < tp1_mult:
                        tp1_mult = adjusted_mult
                        adjustments.append(f"TP1 adjusted to {tp1_mult:.1f}R (S/R at {level_distance_r:.1f}R)")
                
                # Adjust TP2/TP3 based on next levels
                if len(relevant_levels) >= 2:
                    second_level_dist = abs(relevant_levels[1] - entry_price) / r_value if r_value > 0 else tp2_mult
                    if second_level_dist < tp2_mult * 0.9:
                        tp2_mult = min(tp2_mult, second_level_dist * 0.9)
                        adjustments.append(f"TP2 adjusted for S/R")
            
            # === 2. RECENT SWING ANALYSIS ===
            # What have been realistic price moves recently?
            recent_moves = []
            for i in range(max(0, len(df) - 30), len(df) - 1):
                move = abs(df['close'].iloc[i+1] - df['close'].iloc[i]) / df['close'].iloc[i]
                recent_moves.append(move)
            
            if recent_moves:
                avg_move = np.mean(recent_moves)
                max_move = max(recent_moves)
                
                # Convert to R-multiples
                avg_move_r = (avg_move * entry_price) / r_value if r_value > 0 else 1.0
                max_move_r = (max_move * entry_price) / r_value if r_value > 0 else 2.0
                
                # If typical moves are small, don't expect huge TPs
                if avg_move_r < 0.3 and tp1_mult > 0.8:
                    tp1_mult = min(tp1_mult, 0.8)
                    adjustments.append(f"TP1 reduced for low volatility (avg {avg_move_r:.2f}R)")
            
            # === 3. REGIME ANALYSIS ===
            returns = df['close'].pct_change().dropna()
            if len(returns) >= 10:
                regime_result = calculate_regime_hmm(returns, n_regimes=3)
                
                if regime_result.get('status') == 'success':
                    regime = regime_result.get('regime', 'NORMAL')
                    
                    if regime == 'CALM':
                        # Mean-reverting: take profits faster
                        tp1_mult *= 0.85
                        tp2_mult *= 0.85
                        tp3_mult *= 0.85
                        adjustments.append(f"Regime CALM: -15%")
                    elif regime == 'VOLATILE':
                        # Volatile: wider TPs (let it run)
                        tp1_mult *= 1.1
                        tp2_mult *= 1.1
                        tp3_mult *= 1.1
                        adjustments.append(f"Regime VOLATILE: +10%")
                    # NORMAL: no adjustment
            
            # === 4. VOLATILITY TREND ===
            if len(returns) >= 10:
                vol_result = calculate_garch_volatility(returns)
                if vol_result.get('status') == 'success':
                    vol_trend = vol_result.get('vol_trend', '')
                    vol_change = vol_result.get('vol_change_pct', 0)
                    
                    if vol_trend == 'increasing' and vol_change > 15:
                        # Vol expanding: widen TPs
                        tp1_mult *= 1.1
                        tp2_mult *= 1.15
                        tp3_mult *= 1.2
                        adjustments.append(f"Vol expanding +{vol_change:.0f}%: wider TPs")
                    elif vol_trend == 'decreasing' and vol_change < -15:
                        # Vol contracting: tighter TPs
                        tp1_mult *= 0.9
                        tp2_mult *= 0.85
                        tp3_mult *= 0.8
                        adjustments.append(f"Vol contracting {vol_change:.0f}%: tighter TPs")
            
            # === 5. MOMENTUM STRENGTH ===
            # If momentum is strong in our direction, let it run
            rsi = self._calculate_rsi(df['close'], 14)
            momentum_5 = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100 if len(df) >= 5 else 0
            
            strong_momentum = False
            if is_long:
                strong_momentum = rsi > 55 and momentum_5 > 1.0
            else:
                strong_momentum = rsi < 45 and momentum_5 < -1.0
            
            if strong_momentum:
                # Strong momentum: widen TP2 and TP3 to let winners run
                tp2_mult *= 1.15
                tp3_mult *= 1.25
                adjustments.append(f"Strong momentum: wider TP2/TP3")
            
            # === 6. ENSURE MINIMUM SPACING ===
            # TP1 should be at least 0.5R, TP2 at least 1.0R, TP3 at least 1.5R
            tp1_mult = max(tp1_mult, 0.5)
            tp2_mult = max(tp2_mult, tp1_mult + 0.3)  # At least 0.3R more than TP1
            tp3_mult = max(tp3_mult, tp2_mult + 0.5)  # At least 0.5R more than TP2
            
            # === 7. CAP MAXIMUM VALUES ===
            # Don't set unrealistic targets
            tp1_mult = min(tp1_mult, 1.5)
            tp2_mult = min(tp2_mult, 3.0)
            tp3_mult = min(tp3_mult, 5.0)
            
        except Exception as e:
            logger.warning(f"Smart TP error: {e} - using defaults")
            adjustments.append(f"Error: {str(e)[:30]}")
        
        if adjustments:
            logger.info(f"🎯 Smart TP adjustments: {'; '.join(adjustments)}")
        
        return self._calculate_tp_output(entry_price, r_value, tp1_mult, tp2_mult, tp3_mult, is_long, adjustments)
    
    def _calculate_tp_output(self, entry_price: float, r_value: float, 
                            tp1_mult: float, tp2_mult: float, tp3_mult: float,
                            is_long: bool, adjustments: list) -> Dict[str, float]:
        """
        Calculate TPs dynamically based on:
        1. ATR (Average True Range) - actual market volatility
        2. R-value (risk per trade) - ensures good risk:reward
        3. Recent price action - what moves are realistic
        
        NO HARDCODED VALUES - everything calculated from market data.
        """
        try:
            # === GET MARKET DATA FOR SMART CALCULATION ===
            df = self.bars_agg if hasattr(self, 'bars_agg') and self.bars_agg is not None else None
            
            if df is not None and len(df) >= 20:
                # Calculate ATR (14-period) - this tells us typical price movement
                high = df['high']
                low = df['low']
                close = df['close'].shift(1)
                
                tr1 = high - low
                tr2 = abs(high - close)
                tr3 = abs(low - close)
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr = float(tr.rolling(14).mean().iloc[-1])
                
                # ATR as percentage of price
                atr_pct = atr / entry_price
                
                # Recent volatility (standard deviation of returns)
                returns = df['close'].pct_change().dropna().tail(20)
                volatility = float(returns.std()) if len(returns) > 5 else atr_pct
                
                # Average candle range (how much price typically moves per candle)
                avg_candle_range = float((df['high'] - df['low']).tail(20).mean())
                avg_candle_pct = avg_candle_range / entry_price
                
                # === SMART TP CALCULATION ===
                # TP1: 1x ATR or 2-3 candles worth of movement (quick partial profit)
                # TP2: 1.5-2x ATR (solid profit target)
                # TP3: 2.5-3x ATR (runner target)
                
                # Base TPs on ATR (scales with actual market movement)
                tp1_pct = atr_pct * 1.0  # 1 ATR
                tp2_pct = atr_pct * 1.8  # 1.8 ATR
                tp3_pct = atr_pct * 2.8  # 2.8 ATR
                
                # Adjust based on regime multipliers from smart analysis
                # tp1_mult comes from S/R and regime analysis (typically 0.6-1.5)
                tp1_pct *= tp1_mult
                tp2_pct *= tp2_mult / 1.5  # Normalize since tp2_mult baseline is 1.5
                tp3_pct *= tp3_mult / 2.7  # Normalize since tp3_mult baseline is ~2.7
                
                # === SANITY CHECKS (not hardcoded, but reasonable bounds) ===
                # Minimum: At least 2 candles worth of movement for TP1
                min_tp1 = avg_candle_pct * 2
                # Maximum: Don't expect more than 5 ATR for TP3 (unrealistic)
                max_tp3 = atr_pct * 5
                
                # R-value based minimum (ensure at least 0.5R for TP1)
                r_pct = r_value / entry_price
                min_tp1_r = r_pct * 0.5
                
                # Apply bounds dynamically based on calculated values
                tp1_pct = max(tp1_pct, min_tp1, min_tp1_r)
                tp2_pct = max(tp2_pct, tp1_pct * 1.5)  # TP2 at least 1.5x TP1
                tp3_pct = min(max(tp3_pct, tp2_pct * 1.5), max_tp3)  # TP3 at least 1.5x TP2
                
                # Log the smart calculation
                adjustments.append(f"ATR-based: ATR={atr_pct*100:.2f}%")
                
            else:
                # Fallback: Use R-value based calculation
                r_pct = r_value / entry_price if entry_price > 0 else 0.005
                tp1_pct = r_pct * tp1_mult
                tp2_pct = r_pct * tp2_mult
                tp3_pct = r_pct * tp3_mult
                adjustments.append("R-based (no ATR data)")
            
            # Calculate final prices
            if is_long:
                tp1 = entry_price * (1 + tp1_pct)
                tp2 = entry_price * (1 + tp2_pct)
                tp3 = entry_price * (1 + tp3_pct)
            else:
                tp1 = entry_price * (1 - tp1_pct)
                tp2 = entry_price * (1 - tp2_pct)
                tp3 = entry_price * (1 - tp3_pct)
            
            # Log for debugging
            logger.info(f"🎯 Smart TPs: TP1 +{tp1_pct*100:.2f}% | TP2 +{tp2_pct*100:.2f}% | TP3 +{tp3_pct*100:.2f}%")
            
            return {
                'tp1': tp1,
                'tp2': tp2,
                'tp3': tp3,
                'tp1_mult': tp1_pct * 100,  # Store as percentage for display
                'tp2_mult': tp2_pct * 100,
                'tp3_mult': tp3_pct * 100,
                'adjustments': adjustments
            }
            
        except Exception as e:
            logger.warning(f"Smart TP calculation error: {e}, using R-based fallback")
            # Ultimate fallback: simple R-based
            r_pct = r_value / entry_price if entry_price > 0 else 0.005
            tp1_pct = max(r_pct * 0.8, 0.004)  # 0.8R or 0.4% min
            tp2_pct = max(r_pct * 1.5, 0.008)
            tp3_pct = max(r_pct * 2.5, 0.015)
            
            if is_long:
                tp1 = entry_price * (1 + tp1_pct)
                tp2 = entry_price * (1 + tp2_pct)
                tp3 = entry_price * (1 + tp3_pct)
            else:
                tp1 = entry_price * (1 - tp1_pct)
                tp2 = entry_price * (1 - tp2_pct)
                tp3 = entry_price * (1 - tp3_pct)
            
            return {
                'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
                'tp1_mult': tp1_pct * 100, 'tp2_mult': tp2_pct * 100, 'tp3_mult': tp3_pct * 100,
                'adjustments': ['Fallback R-based']
            }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI for TP analysis."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
        except Exception:
            return 50.0

    # =====================================================================
    # 🧮 DYNAMIC TP CALCULATOR — Math-driven per-trade TP levels
    # =====================================================================
    def calculate_dynamic_tp(self, entry_price: float, side: str, df: pd.DataFrame = None,
                              atr: float = None, stop_loss: float = None) -> Dict[str, Any]:
        """
        Calculate DYNAMIC take-profit levels for each trade using multi-factor math.
        
        Instead of fixed % TPs, this calculates optimal targets based on:
        1. ATR (Average True Range) — base volatility measure
        2. Momentum (ROC5, ROC10) — strong → wider TPs, weak → tighter
        3. RSI — overbought/oversold → tighter (reversion risk)
        4. ADX (trend strength) — trending → wider, ranging → tighter
        5. Volume ratio — high vol → wider (momentum continuation)
        6. S/R proximity — don't set TP beyond key levels
        7. Recent candle range — calibrate to actual price moves
        8. Fee floor — TP1 never below 0.12% (fee breakeven)
        
        Returns: {
            'tp1_pct': float, 'tp2_pct': float, 'tp3_pct': float,
            'tp1': float, 'tp2': float, 'tp3': float,
            'factors': str  # Human-readable explanation
        }
        """
        import numpy as np
        
        is_long = side.lower() == 'long'
        factors = []
        
        # ══════════════════════════════════════════════════════════════
        # STEP 1: BASE TP from ATR (scales with actual market volatility)
        # ══════════════════════════════════════════════════════════════
        if df is not None and len(df) >= 14:
            try:
                high = df['high']
                low = df['low']
                close_prev = df['close'].shift(1)
                
                tr1 = high - low
                tr2 = abs(high - close_prev)
                tr3 = abs(low - close_prev)
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                calc_atr = float(tr.rolling(14).mean().iloc[-1])
                
                if calc_atr > 0:
                    atr = calc_atr
            except Exception:
                pass
        
        if atr is None or atr <= 0:
            # Ultimate fallback: use entry price * 0.3% as proxy ATR
            atr = entry_price * 0.003
            factors.append("ATR=fallback")
        
        atr_pct = (atr / entry_price) * 100  # ATR as percentage
        
        # Base TPs: TP1=0.7×ATR, TP2=1.4×ATR, TP3=2.2×ATR
        # These scale naturally — volatile coins get wider TPs, stable ones get tighter
        tp1_pct = atr_pct * 0.70
        tp2_pct = atr_pct * 1.40
        tp3_pct = atr_pct * 2.20
        factors.append(f"ATR={atr_pct:.2f}%")
        
        # ═══════════════════════════════════════════════════════════════════
        # REGIME-ADAPTIVE TP MULTIPLIER (Notebook #2 §9)
        # TRENDING: let winners run (wider TPs)
        # CHOPPY: take profits quickly (tighter TPs)
        # HIGH_VOL: big moves possible (wider TPs)
        # ═══════════════════════════════════════════════════════════════════
        regime_tp_mult = 1.0
        if df is not None and len(df) >= 20:
            try:
                from indicator import calculate_adx
                _regime_adx = calculate_adx(df)
                if _regime_adx > 30:
                    regime_tp_mult = 1.20  # Strong trend → let winners run
                    factors.append(f"REGIME=TREND(ADX{_regime_adx:.0f})→TP×1.20")
                elif _regime_adx < 15:
                    regime_tp_mult = 0.60  # Choppy → take profits fast
                    factors.append(f"REGIME=CHOPPY(ADX{_regime_adx:.0f})→TP×0.60")
                elif _regime_adx < 20:
                    regime_tp_mult = 0.80  # Ranging → tighter TPs
                    factors.append(f"REGIME=RANGE(ADX{_regime_adx:.0f})→TP×0.80")
                # High volatility check (ATR > 2% of price)
                if atr_pct > 2.0 and regime_tp_mult >= 1.0:
                    regime_tp_mult = max(regime_tp_mult, 1.30)  # High vol → wider
                    factors.append(f"HIVOL(ATR{atr_pct:.1f}%)→TP×{regime_tp_mult:.2f}")
            except Exception:
                pass
        
        tp1_pct *= regime_tp_mult
        tp2_pct *= regime_tp_mult
        tp3_pct *= regime_tp_mult
        
        # ══════════════════════════════════════════════════════════════
        # STEP 2: MOMENTUM ADJUSTMENT (ROC5, ROC10)
        # Strong momentum in our direction → widen TPs (let winners run)
        # Weak/counter momentum → tighten TPs (take what we can get)
        # ══════════════════════════════════════════════════════════════
        momentum_mult = 1.0
        if df is not None and len(df) >= 11:
            try:
                close = df['close']
                roc_5 = float((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] * 100)
                roc_10 = float((close.iloc[-1] - close.iloc[-11]) / close.iloc[-11] * 100)
                
                # Momentum aligned with trade direction?
                if is_long:
                    mom_score = (roc_5 * 2 + roc_10) / 3  # Weight recent more
                else:
                    mom_score = -(roc_5 * 2 + roc_10) / 3  # Invert for shorts
                
                if mom_score > 0.5:
                    # Strong aligned momentum → widen TPs
                    momentum_mult = 1.0 + min(mom_score * 0.15, 0.40)  # Up to +40%
                    factors.append(f"MOM+{mom_score:.1f}→×{momentum_mult:.2f}")
                elif mom_score < -0.3:
                    # Counter momentum → tighten TPs
                    momentum_mult = max(1.0 + mom_score * 0.20, 0.60)  # Down to -40%
                    factors.append(f"MOM{mom_score:.1f}→×{momentum_mult:.2f}")
                else:
                    factors.append("MOM=neutral")
            except Exception:
                pass
        
        tp1_pct *= momentum_mult
        tp2_pct *= momentum_mult
        tp3_pct *= momentum_mult
        
        # ══════════════════════════════════════════════════════════════
        # STEP 3: RSI ADJUSTMENT
        # Overbought (LONG) or oversold (SHORT) → tighten TPs, reversion likely
        # Neutral RSI → no change
        # ══════════════════════════════════════════════════════════════
        if df is not None and len(df) >= 14:
            try:
                rsi = self._calculate_rsi(df['close'], 14)
                
                if is_long:
                    if rsi > 70:
                        rsi_mult = 0.75  # Overbought long → very tight TPs
                        factors.append(f"RSI={rsi:.0f}(OB)→×0.75")
                    elif rsi > 60:
                        rsi_mult = 0.90
                        factors.append(f"RSI={rsi:.0f}→×0.90")
                    elif rsi < 35:
                        rsi_mult = 1.15  # Oversold → room to run
                        factors.append(f"RSI={rsi:.0f}(OS)→×1.15")
                    else:
                        rsi_mult = 1.0
                else:  # SHORT
                    if rsi < 30:
                        rsi_mult = 0.75  # Oversold short → very tight TPs
                        factors.append(f"RSI={rsi:.0f}(OS)→×0.75")
                    elif rsi < 40:
                        rsi_mult = 0.90
                        factors.append(f"RSI={rsi:.0f}→×0.90")
                    elif rsi > 65:
                        rsi_mult = 1.15  # Overbought → room to fall
                        factors.append(f"RSI={rsi:.0f}(OB)→×1.15")
                    else:
                        rsi_mult = 1.0
                
                tp1_pct *= rsi_mult
                tp2_pct *= rsi_mult
                tp3_pct *= rsi_mult
            except Exception:
                pass
        
        # ══════════════════════════════════════════════════════════════
        # STEP 4: ADX (Trend Strength) ADJUSTMENT
        # ADX > 25 = trending → wider TPs (ride the trend)
        # ADX < 20 = ranging → tighter TPs (take quick profits)
        # ══════════════════════════════════════════════════════════════
        if df is not None and len(df) >= 20:
            try:
                from indicator import calculate_adx
                adx = calculate_adx(df)
                
                if adx > 35:
                    adx_mult = 1.25  # Strong trend → much wider
                    factors.append(f"ADX={adx:.0f}(trend)→×1.25")
                elif adx > 25:
                    adx_mult = 1.10  # Moderate trend → slightly wider
                    factors.append(f"ADX={adx:.0f}→×1.10")
                elif adx < 15:
                    adx_mult = 0.70  # Dead market → very tight
                    factors.append(f"ADX={adx:.0f}(flat)→×0.70")
                elif adx < 20:
                    adx_mult = 0.85  # Weak trend → tighter
                    factors.append(f"ADX={adx:.0f}(weak)→×0.85")
                else:
                    adx_mult = 1.0
                
                tp1_pct *= adx_mult
                # TP2/TP3 more sensitive to trend strength
                tp2_pct *= (adx_mult ** 1.2)
                tp3_pct *= (adx_mult ** 1.4)
            except Exception:
                pass
        
        # ══════════════════════════════════════════════════════════════
        # STEP 5: VOLUME RATIO ADJUSTMENT
        # High volume → momentum more likely to continue → wider TPs
        # Low volume → moves may fizzle → tighter TPs
        # ══════════════════════════════════════════════════════════════
        if df is not None and len(df) >= 10:
            try:
                avg_vol = float(df['volume'].iloc[-20:-1].mean()) if len(df) >= 20 else float(df['volume'].mean())
                cur_vol = float(df['volume'].iloc[-2]) if len(df) > 1 else avg_vol
                vol_ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
                
                if vol_ratio > 2.0:
                    vol_mult = 1.20  # Very high volume → wider
                    factors.append(f"VOL={vol_ratio:.1f}x→×1.20")
                elif vol_ratio > 1.5:
                    vol_mult = 1.10
                    factors.append(f"VOL={vol_ratio:.1f}x→×1.10")
                elif vol_ratio < 0.5:
                    vol_mult = 0.80  # Very low volume → tighter
                    factors.append(f"VOL={vol_ratio:.1f}x→×0.80")
                elif vol_ratio < 0.8:
                    vol_mult = 0.90
                    factors.append(f"VOL={vol_ratio:.1f}x→×0.90")
                else:
                    vol_mult = 1.0
                
                tp1_pct *= vol_mult
                tp2_pct *= vol_mult
                tp3_pct *= vol_mult
            except Exception:
                pass
        
        # ══════════════════════════════════════════════════════════════
        # STEP 6: S/R PROXIMITY — Cap TPs at key levels
        # Don't set TP beyond next resistance (LONG) or support (SHORT)
        # Place TP just before the level (90% of distance)
        # ══════════════════════════════════════════════════════════════
        if df is not None and len(df) >= 50 and hasattr(self, 'ai_filter'):
            try:
                sr_levels = self.ai_filter._detect_resistance_support_levels(df, entry_price)
                
                if is_long:
                    dist_to_r = sr_levels.get('distance_to_resistance_pct', 99)
                    if dist_to_r < 99:
                        # Cap TP at 90% of distance to resistance
                        sr_cap = dist_to_r * 0.90
                        if tp1_pct > sr_cap and sr_cap > 0.12:  # Don't cap below fee floor
                            old_tp1 = tp1_pct
                            tp1_pct = sr_cap
                            factors.append(f"SR-cap:TP1 {old_tp1:.2f}→{tp1_pct:.2f}%")
                        if tp2_pct > dist_to_r * 1.5:
                            tp2_pct = dist_to_r * 1.5
                else:
                    dist_to_s = sr_levels.get('distance_to_support_pct', 99)
                    if dist_to_s < 99:
                        sr_cap = dist_to_s * 0.90
                        if tp1_pct > sr_cap and sr_cap > 0.12:
                            old_tp1 = tp1_pct
                            tp1_pct = sr_cap
                            factors.append(f"SR-cap:TP1 {old_tp1:.2f}→{tp1_pct:.2f}%")
                        if tp2_pct > dist_to_s * 1.5:
                            tp2_pct = dist_to_s * 1.5
            except Exception:
                pass
        
        # ══════════════════════════════════════════════════════════════
        # STEP 7: RECENT CANDLE RANGE CALIBRATION
        # Use actual recent candle sizes to sanity-check TPs
        # If avg candle = 0.15%, TP1 at 2% is unrealistic
        # ══════════════════════════════════════════════════════════════
        if df is not None and len(df) >= 10:
            try:
                candle_ranges = ((df['high'] - df['low']) / df['close']).tail(20) * 100
                avg_candle_pct = float(candle_ranges.mean())
                
                # TP1 should be achievable in 2-4 candles
                max_reasonable_tp1 = avg_candle_pct * 4
                if tp1_pct > max_reasonable_tp1:
                    old_tp1 = tp1_pct
                    tp1_pct = max_reasonable_tp1
                    factors.append(f"Candle-cap:TP1 {old_tp1:.2f}→{tp1_pct:.2f}%")
            except Exception:
                pass
        
        # ══════════════════════════════════════════════════════════════
        # STEP 8: HARD FLOORS AND CAPS (fee protection + sanity)
        # ══════════════════════════════════════════════════════════════
        FEE_FLOOR_PCT = 0.15   # TP1 must be at least 0.15% (above 0.11% fee round-trip)
        MIN_TP1_PCT = 0.15     # Absolute minimum TP1
        MAX_TP1_PCT = 2.00     # Absolute maximum TP1 (don't go crazy)
        
        tp1_pct = max(tp1_pct, MIN_TP1_PCT)
        tp1_pct = min(tp1_pct, MAX_TP1_PCT)
        
        # TP2 must be at least 1.5× TP1, TP3 at least 2× TP1
        tp2_pct = max(tp2_pct, tp1_pct * 1.5)
        tp3_pct = max(tp3_pct, tp1_pct * 2.5)
        
        # Maximum caps
        tp2_pct = min(tp2_pct, 4.0)
        tp3_pct = min(tp3_pct, 8.0)
        
        # ══════════════════════════════════════════════════════════════
        # CALCULATE FINAL PRICES
        # ══════════════════════════════════════════════════════════════
        if is_long:
            tp1 = entry_price * (1 + tp1_pct / 100)
            tp2 = entry_price * (1 + tp2_pct / 100)
            tp3 = entry_price * (1 + tp3_pct / 100)
        else:
            tp1 = entry_price * (1 - tp1_pct / 100)
            tp2 = entry_price * (1 - tp2_pct / 100)
            tp3 = entry_price * (1 - tp3_pct / 100)
        
        # Risk:Reward ratio (if SL provided)
        rr_ratio = "N/A"
        if stop_loss and stop_loss > 0:
            sl_dist = abs(entry_price - stop_loss)
            tp1_dist = abs(tp1 - entry_price)
            if sl_dist > 0:
                rr_ratio = f"{tp1_dist / sl_dist:.1f}R"
        
        factors_str = " | ".join(factors)
        logger.info(f"🧮 DYNAMIC TP: TP1={tp1_pct:.2f}% TP2={tp2_pct:.2f}% TP3={tp3_pct:.2f}% | R:R={rr_ratio} | {factors_str}")
        
        return {
            'tp1_pct': tp1_pct / 100,   # As decimal for bot.py compatibility
            'tp2_pct': tp2_pct / 100,
            'tp3_pct': tp3_pct / 100,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'factors': factors_str,
            'tp1_display': tp1_pct,      # As percentage for display
            'tp2_display': tp2_pct,
            'tp3_display': tp3_pct,
        }

    async def _get_momentum_trail_offset(self, current_price: float, entry_price: float, 
                                         side: str, r_value: float) -> float:
        """
        MOMENTUM-AWARE TRAILING STOP - Adapts trail tightness based on momentum.
        
        Logic:
        - STRONG momentum in our favor → WIDER trail (let it run)
        - WEAK/FADING momentum → TIGHTER trail (protect profit)
        - COUNTER momentum → VERY TIGHT trail (exit soon)
        
        Returns: Trail offset as R-multiple (e.g., 0.5R, 0.3R, 0.8R)
        """
        import numpy as np
        
        # Default offset from params
        base_offset = self.TRAIL_OFFSET_R
        
        try:
            if not hasattr(self, 'bars_agg') or len(self.bars_agg) < 10:
                return base_offset
            
            df = pd.DataFrame(self.bars_agg)
            if len(df) < 10:
                return base_offset
            
            is_long = side.lower() == 'long'
            
            # Calculate momentum metrics
            prices = df['close'].values
            
            # 1. Short-term momentum (last 3 candles)
            momentum_3 = (prices[-1] - prices[-4]) / prices[-4] * 100 if len(prices) >= 4 else 0
            
            # 2. Medium-term momentum (last 5 candles)
            momentum_5 = (prices[-1] - prices[-6]) / prices[-6] * 100 if len(prices) >= 6 else 0
            
            # 3. RSI
            rsi = self._calculate_rsi(df['close'], 14)
            
            # 4. Price vs SMA
            sma_20 = df['close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else prices[-1]
            price_vs_sma = ((prices[-1] - sma_20) / sma_20) * 100
            
            # 5. Current profit %
            if is_long:
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            # === MOMENTUM SCORING ===
            # Positive = momentum in our favor, Negative = momentum against us
            if is_long:
                momentum_score = (
                    (momentum_3 * 2) +  # Weight short-term more
                    (momentum_5 * 1) +
                    ((rsi - 50) * 0.1) +  # RSI contribution
                    (price_vs_sma * 0.5)  # Above SMA = good for long
                )
            else:
                momentum_score = (
                    (-momentum_3 * 2) +  # Inverse for short
                    (-momentum_5 * 1) +
                    ((50 - rsi) * 0.1) +  # Low RSI good for short
                    (-price_vs_sma * 0.5)  # Below SMA = good for short
                )
            
            # === ADAPTIVE TRAIL OFFSET ===
            # Strong positive momentum → wider trail (let it run)
            # Negative momentum → tighter trail (protect profit)
            
            if momentum_score > 3.0:
                # STRONG momentum in our favor - give it room
                trail_mult = 1.3  # 30% wider
            elif momentum_score > 1.5:
                # Moderate momentum - normal trail
                trail_mult = 1.0
            elif momentum_score > 0:
                # Weak momentum - slightly tighter
                trail_mult = 0.85
            elif momentum_score > -1.5:
                # Slightly against us - tighter trail
                trail_mult = 0.7
            else:
                # Strong momentum against - very tight
                trail_mult = 0.5
            
            # === PROFIT-BASED ADJUSTMENT ===
            # The more profit we have, the tighter we should trail to protect it
            if pnl_pct > 2.0:
                trail_mult *= 0.85  # 15% tighter for large profits
            elif pnl_pct > 1.0:
                trail_mult *= 0.9   # 10% tighter for decent profits
            
            # Calculate final offset
            final_offset = base_offset * trail_mult
            
            # Enforce bounds (0.3R to 1.0R)
            final_offset = max(0.3, min(1.0, final_offset))
            
            # Log significant changes
            if abs(final_offset - base_offset) > 0.1:
                logger.debug(f"📐 Trail offset adjusted: {base_offset:.2f}R → {final_offset:.2f}R "
                           f"(momentum_score={momentum_score:.1f}, pnl={pnl_pct:.1f}%)")
            
            return final_offset
            
        except Exception as e:
            logger.debug(f"Momentum trail calculation error: {e}")
            return base_offset

    async def _validate_pair_switch_math(self, current_score: float, target_score: float,
                                        current_symbol: str, target_symbol: str,
                                        current_data: Dict, target_data: Dict) -> Dict[str, Any]:
        """
        Math-level pair switch validation using statistical hypothesis testing.
        
        NOT just "is 8-point difference?" 
        BUT "is the difference statistically significant and economically justified?"
        """
        from indicator import calculate_confidence_interval
        import numpy as np
        
        result = {
            'approved': False,
            'confidence': 0.0,
            'reason': '',
            'reasoning': []
        }
        
        try:
            score_diff = target_score - current_score
            
            # === TEST 1: Is the difference statistically significant? ===
            # Simulate score distributions (based on component variance)
            current_variance = current_data.get('score_variance', 100) if current_data else 100
            target_variance = target_data.get('score_variance', 100) if target_data else 100
            
            # Calculate 95% CI on score difference
            pooled_variance = (current_variance + target_variance) / 2
            se_diff = np.sqrt(pooled_variance / 10)  # 10 trades estimate sample size
            
            ci_result = {
                'mean': score_diff,
                'lower': score_diff - 1.96 * se_diff,
                'upper': score_diff + 1.96 * se_diff,
                'margin': 1.96 * se_diff
            }
            
            logger.debug(f"Score diff CI: {ci_result['mean']:.1f} ± {ci_result['margin']:.1f}")
            
            # If 0 is in CI: Not confident enough
            if ci_result['lower'] < 0 < ci_result['upper']:
                result['reason'] = f"Score difference ({score_diff:.1f}) NOT statistically significant (CI: [{ci_result['lower']:.1f}, {ci_result['upper']:.1f}])"
                result['reasoning'].append(result['reason'])
                return result
            
            result['confidence'] = min(0.95, 1 - (2 * ci_result['margin'] / abs(score_diff))) if score_diff != 0 else 0
            result['reasoning'].append(
                f"Score difference IS significant: {score_diff:+.1f} (95% CI: [{ci_result['lower']:.1f}, {ci_result['upper']:.1f}])"
            )
            
            # === TEST 2: Cost-Benefit Analysis ===
            SWITCH_COST = 0.002  # 0.2% estimated (fees + slippage)
            
            # Expected improvement in daily return from score difference
            # 1-point score = ~0.02% daily improvement (calibrate from backtest)
            expected_daily_improvement = score_diff * 0.0002
            
            # How many days needed to break even?
            days_to_breakeven = SWITCH_COST / expected_daily_improvement if expected_daily_improvement > 0 else float('inf')
            
            if days_to_breakeven > 30:
                result['reason'] = f"Switch cost ({SWITCH_COST:.1%}) > expected benefit over 30 days"
                result['reasoning'].append(result['reason'])
                return result
            
            result['reasoning'].append(
                f"Cost-benefit positive: breakeven in {days_to_breakeven:.1f} days (cost: {SWITCH_COST:.1%}, "
                f"expected daily improvement: +{expected_daily_improvement:.3%})"
            )
            
            # === TEST 3: Correlation Check (no point switching to correlated pair) ===
            if current_data and target_data:
                # If we have historical data, check correlation
                current_returns = current_data.get('returns', [])
                target_returns = target_data.get('returns', [])
                
                if current_returns and target_returns and len(current_returns) >= 20:
                    try:
                        correlation = np.corrcoef(current_returns[-20:], target_returns[-20:])[0, 1]
                        
                        if correlation > 0.75:
                            result['reason'] = f"Target pair too correlated ({correlation:.2f}) with current - no diversification benefit"
                            result['reasoning'].append(result['reason'])
                            return result
                        
                        result['reasoning'].append(
                            f"Good diversification: correlation = {correlation:.2f} (want < 0.7)"
                        )
                    except Exception:
                        pass  # Non-critical: Can't compute correlation, continue
            
            # === TEST 4: Regime Persistence (wait if regime just changed) ===
            target_regime = target_data.get('regime', 'NORMAL') if target_data else 'NORMAL'
            target_regime_age = target_data.get('regime_age_bars', 100) if target_data else 100  # How many bars in this regime
            
            if target_regime_age < 10:  # Just changed regime
                result['reason'] = f"Target pair regime just changed (only {target_regime_age} bars) - wait for stability"
                result['reasoning'].append(result['reason'])
                return result
            
            result['reasoning'].append(f"Target regime stable: {target_regime} ({target_regime_age} bars)")
            
            # === ALL TESTS PASSED ===
            result['approved'] = True
            result['reason'] = f"Switch approved: +{score_diff:.1f} points, {result['confidence']:.0%} confidence"
            
        except Exception as e:
            logger.error(f"Math pair switch validation error: {e}")
            result['reason'] = f"Validation error: {str(e)}"
        
        return result
    
    # ============================================================================
    
    async def _open_position_for_symbol(self, symbol: str, signal: int, price: float, 
                                        atr: float, df: pd.DataFrame = None, risk_pct: float = None,
                                        source: str = "multi_pair", ai_reasoning: str = "",
                                        ai_confidence: float = 0.7, ai_result: Dict = None):
        """
        Open a position for a SPECIFIC symbol (for multi-position support).
        This allows opening positions on different symbols than the main one.
        
        Now accepts AI power parameters for unified scanner integration.
        """
        try:
            # ═══════════════════════════════════════════════════════════════════
            # 🚨 CRITICAL: RISK MANAGER GATE - CHECK BEFORE ANY TRADE
            # This prevents trading when daily loss limits, weekly limits, or
            # max drawdown thresholds have been breached
            # ═══════════════════════════════════════════════════════════════════
            can_trade_result = self.risk_manager.can_trade(self.balance, self.initial_balance)
            if not can_trade_result['allowed']:
                logger.error(f"🚫 RISK MANAGER BLOCKED: {symbol} - {can_trade_result['reason']}")
                return
            
            # Bybit Futures: Both LONG and SHORT supported
            side = "long" if signal == 1 else "short"
            
            # === SIDE FILTER CHECK (must match _open_position) ===
            allowed = getattr(self, 'allowed_sides', 'both').lower()
            if allowed != 'both':
                if allowed == 'long' and side == 'short':
                    logger.warning(f"🚫 SHORT {symbol} blocked by /longonly mode")
                    return
                elif allowed == 'short' and side == 'long':
                    logger.warning(f"🚫 LONG {symbol} blocked by /shortonly mode")
                    return
            
            risk_pct = risk_pct or self.RISK_PCT
            
            # Apply AI power size multiplier if provided
            if ai_result and ai_result.get('ai_power'):
                ai_power = ai_result['ai_power']
                size_mult = ai_power.get('size_multiplier', 1.0)
                if size_mult != 1.0:
                    risk_pct = risk_pct * size_mult
                    logger.info(f"🤖 AI POWER: Adjusting risk by {size_mult:.1f}x → {risk_pct:.2%}")
            
            logger.info(f"🚀 _open_position_for_symbol CALLED: {side.upper()} {symbol} @ ${price:.4f} | live_mode={self.live_mode} | source={source}")
            
            # === FINAL S/R VALIDATION (LAST LINE OF DEFENSE) ===
            # This is the FINAL CHECK before opening - prevents entering at resistance/support
            # Even if scanner passed, market may have moved - check again at execution time
            try:
                if df is not None and len(df) >= 50:
                    sr_levels = self.ai_filter._detect_resistance_support_levels(df, price)
                    pos_range = sr_levels.get('position_in_range_pct', 50)
                    in_r_zone = sr_levels.get('in_resistance_zone', False)
                    in_s_zone = sr_levels.get('in_support_zone', False)
                    r_touches = sr_levels.get('resistance_touches', 0)
                    s_touches = sr_levels.get('support_touches', 0)
                    
                    # Adjust thresholds based on mode
                    from ai_filter import get_threshold_mode
                    _final_sr_mode = get_threshold_mode()
                    _final_r_limit = 99 if _final_sr_mode == 'testing' else 88
                    _final_s_limit = 1 if _final_sr_mode == 'testing' else 12
                    _final_touch_req = 99 if _final_sr_mode == 'testing' else 3  # testing: effectively disable zone block
                    
                    # BLOCK LONG at resistance
                    if side == "long":
                        if in_r_zone and r_touches >= _final_touch_req:
                            logger.error(f"🚫 FINAL BLOCK: LONG {symbol} - IN RESISTANCE ZONE with {r_touches} rejections @ ${price:.4f}")
                            return
                        if pos_range >= _final_r_limit:
                            logger.error(f"🚫 FINAL BLOCK: LONG {symbol} - TOO CLOSE TO RESISTANCE ({pos_range:.0f}% >= {_final_r_limit}%) @ ${price:.4f}")
                            return
                    
                    # BLOCK SHORT at support
                    if side == "short":
                        if in_s_zone and s_touches >= _final_touch_req:
                            logger.error(f"🚫 FINAL BLOCK: SHORT {symbol} - IN SUPPORT ZONE with {s_touches} bounces @ ${price:.4f}")
                            return
                        if pos_range <= _final_s_limit:
                            logger.error(f"🚫 FINAL BLOCK: SHORT {symbol} - TOO CLOSE TO SUPPORT ({pos_range:.0f}% <= {_final_s_limit}%) @ ${price:.4f}")
                            return
                    
                    logger.info(f"✅ FINAL S/R CHECK PASSED: {symbol} {side.upper()} at {pos_range:.0f}% of range")
            except Exception as sr_err:
                logger.warning(f"⚠️ Final S/R check failed for {symbol}: {sr_err} (proceeding with caution)")
            
            # === CRITICAL: Check max positions BEFORE opening ===
            open_positions = [p for p in self.positions.values() if p is not None]
            max_positions = self.multi_pair_config.max_total_positions if hasattr(self, 'multi_pair_config') else 2
            if len(open_positions) >= max_positions:
                logger.warning(f"⚠️ BLOCKED: Already at max positions ({len(open_positions)}/{max_positions}) - cannot open {symbol}")
                return
            
            # Check if we already have a position on this symbol
            if self.positions.get(symbol):
                logger.warning(f"Already have position on {symbol} - skipping")
                return
            
            # Use balance lock to prevent race condition on margin allocation
            # Lock scope: margin check + position sizing + order execution
            # Released after order so other coroutines can proceed
            async with self._balance_lock:
                # Check total portfolio MARGIN used (not notional value!)
                # For futures, margin = notional_value / leverage
                # Use configured leverage from adaptive_params
                CONFIGURED_LEVERAGE = int(self.adaptive_params.get('max_leverage', {}).get('current', 10))
                total_margin_used = 0
                for sym, pos in self.positions.items():
                    if pos:
                        notional_value = getattr(pos, 'size', 0) * getattr(pos, 'entry_price', 0)
                        margin_used = notional_value / CONFIGURED_LEVERAGE
                        total_margin_used += margin_used
                
                MAX_MARGIN_PCT = 0.8  # Use up to 80% of balance as margin (was 70%)
                current_margin_pct = total_margin_used / self.balance if self.balance > 0 else 0
                
                logger.info(f"📊 Margin check: {current_margin_pct:.1%} of balance (limit: {MAX_MARGIN_PCT:.0%}, margin: ${total_margin_used:.2f}, balance: ${self.balance:.2f})")
                
                if current_margin_pct >= MAX_MARGIN_PCT:
                    logger.warning(f"⚠️ Total margin {current_margin_pct:.1%} exceeds limit {MAX_MARGIN_PCT:.0%} - skipping position")
                    return
                
                # Reduce risk for second position if already have meaningful exposure
                if current_margin_pct > 0.25:
                    risk_pct = risk_pct * 0.75  # 25% smaller position when already exposed
                    logger.info(f"📊 Reducing risk to {risk_pct:.1%} due to existing margin {current_margin_pct:.1%}")
            
            # Calculate stop loss with MIN floor and DYNAMIC MAX based on ATR
            # === PER-VOLATILITY SL ADJUSTMENT (Notebook #2 §8) ===
            vol_atr_pct = (atr / price) * 100 if price > 0 else 1.0
            if vol_atr_pct < 0.5:
                vol_sl_mult = 0.85  # Tighter — low vol
            elif vol_atr_pct > 2.0:
                vol_sl_mult = 1.15  # Wider — high vol
            else:
                vol_sl_mult = 1.0
            
            atr_stop_distance = atr * self.ATR_MULT * vol_sl_mult
            min_stop_distance = price * self.MIN_STOP_PCT
            # Dynamic max: ATR × multiplier × 1.5 safety, clamped to floor/ceiling
            dynamic_max_pct = max(self.DYNAMIC_STOP_FLOOR, min((atr / price) * self.ATR_MULT * vol_sl_mult * 1.5, self.DYNAMIC_STOP_CEILING))
            max_stop_distance = price * dynamic_max_pct
            stop_distance = max(min_stop_distance, min(atr_stop_distance, max_stop_distance))  # Clamp between min and dynamic max
            logger.info(f"📊 SL Calc: ATR={atr/price*100:.2f}%, VolMult={vol_sl_mult}, DynMax={dynamic_max_pct:.2%}, Final={stop_distance/price*100:.2f}%")
            
            if side == "long":
                stop_loss = price - stop_distance
            else:
                stop_loss = price + stop_distance
            
            # Position sizing - account for round-trip fees on position notional
            risk_per_unit = abs(price - stop_loss)
            risk_amount = self.balance * risk_pct
            # Fees are charged on notional value, so subtract fee cost from risk budget
            # Round-trip fee cost per unit = price * 2 * taker_fee
            fee_cost_per_unit = price * self.FEE_TAKER * 2
            effective_risk_per_unit = risk_per_unit + fee_cost_per_unit  # Total cost per unit if stopped out
            size = risk_amount / effective_risk_per_unit if effective_risk_per_unit > 0 else 0
            
            # Leverage safety — use adaptive_params (matches _open_position)
            MAX_LEVERAGE = int(self.adaptive_params.get('max_leverage', {}).get('current', 10))
            MAX_LEVERAGE = min(MAX_LEVERAGE, 10)  # HARD CAP: Never exceed 10x (was 15x — now matches _open_position)
            position_value = size * price
            current_leverage = position_value / self.balance
            if current_leverage > MAX_LEVERAGE:
                size = (self.balance * MAX_LEVERAGE) / price
            
            # 🧮 DYNAMIC TP — calculated per-trade using ATR, momentum, RSI, ADX, S/R
            dynamic_tp = self.calculate_dynamic_tp(
                entry_price=price, side=side, df=df, atr=atr, stop_loss=stop_loss
            )
            tp1 = dynamic_tp['tp1']
            tp2 = dynamic_tp['tp2']
            tp3 = dynamic_tp['tp3']
            
            logger.info(f"🎯 Dynamic TP for {symbol}: TP1 +{dynamic_tp['tp1_display']:.2f}%=${tp1:.6f} | TP2 +{dynamic_tp['tp2_display']:.2f}%=${tp2:.6f} | TP3 +{dynamic_tp['tp3_display']:.2f}%=${tp3:.6f}")
            logger.info(f"🧮 TP factors: {dynamic_tp['factors']}")
            
            # === CRITICAL: Verify available margin before trading ===
            if self.live_mode:
                try:
                    # For Bybit UTA, use proper wallet balance API
                    try:
                        wallet_info = await self.exchange.private_get_v5_account_wallet_balance({'accountType': 'UNIFIED'})
                        wallet_data = wallet_info.get('result', {}).get('list', [{}])[0]
                        available_balance = float(wallet_data.get('totalAvailableBalance', 0))
                        borrowed_amount = 0
                        for coin in wallet_data.get('coin', []):
                            if coin.get('coin') == 'USDT':
                                borrowed_amount = float(coin.get('borrowAmount', 0) or 0)
                                break
                        if borrowed_amount > 0:
                            logger.info(f"💳 {symbol} Borrowed USDT: ${borrowed_amount:.2f}")
                    except Exception:
                        exchange_balance = await self.exchange.fetch_balance()
                        available_balance = float(exchange_balance['USDT']['free'])
                    
                    # Use configured leverage from adaptive_params
                    ACTUAL_LEVERAGE = int(self.adaptive_params.get('max_leverage', {}).get('current', 10))
                    notional_value = size * price
                    required_margin = notional_value / ACTUAL_LEVERAGE
                    
                    # Add 20% buffer
                    required_margin_with_buffer = required_margin * 1.2
                    
                    if required_margin_with_buffer > available_balance:
                        # Reduce size to fit available margin
                        max_notional = (available_balance / 1.2) * ACTUAL_LEVERAGE
                        old_size = size
                        size = max_notional / price
                        logger.warning(f"⚠️ {symbol} Margin check: Required ${required_margin_with_buffer:.2f} > Available ${available_balance:.2f}")
                        logger.warning(f"⚠️ Reducing position size: {old_size:.4f} → {size:.4f}")
                    else:
                        logger.info(f"💰 {symbol} Margin check OK: ${required_margin:.2f} required, ${available_balance:.2f} available")
                except Exception as margin_err:
                    logger.warning(f"⚠️ Could not verify margin for {symbol}: {margin_err}")
            
            # === SET LEVERAGE ON EXCHANGE ===
            # Use configured leverage from adaptive_params  
            TARGET_LEVERAGE = int(self.adaptive_params.get('max_leverage', {}).get('current', 10))
            logger.info(f"🔧 {symbol} Using {TARGET_LEVERAGE}x leverage (from adaptive_params)")
            if self.live_mode:
                try:
                    await self.exchange.set_leverage(TARGET_LEVERAGE, symbol)
                    logger.info(f"⚙️ {symbol} Leverage set to {TARGET_LEVERAGE}x")
                except Exception as lev_err:
                    # Some exchanges may not support leverage change or it's already set
                    logger.debug(f"Leverage setting note for {symbol}: {lev_err}")
            
            # === EXECUTE ORDER (Live or Paper) ===
            actual_entry_price = price
            if self.live_mode:
                # LIVE MODE: Execute real order on Bybit
                # Use balance lock to prevent concurrent order placement (race condition)
                async with self._balance_lock:
                    order_side = 'buy' if side == 'long' else 'sell'
                    order = await self._execute_market_order(symbol, order_side, size)
                
                if not order:
                    logger.error(f"🚨 LIVE ORDER FAILED for {symbol} - Position NOT opened")
                    return  # Order failed, don't create position
                
                # Use actual fill price if available
                if order.get('average'):
                    actual_entry_price = float(order['average'])
                    logger.info(f"📊 {symbol} fill price: ${actual_entry_price:.4f} (requested: ${price:.4f})")
                    
                    # Check for high slippage
                    slippage = abs(actual_entry_price - price) / price
                    if slippage > 0.005:  # >0.5% slippage warning
                        logger.warning(f"⚠️ {symbol} high slippage: {slippage:.2%}")
                    
                    # Recalculate SL/TP based on actual fill price to maintain correct risk
                    if slippage > 0.001:  # Only adjust if slippage is meaningful (>0.1%)
                        price_diff = actual_entry_price - price
                        stop_loss = stop_loss + price_diff
                        tp1 = tp1 + price_diff
                        tp2 = tp2 + price_diff
                        tp3 = tp3 + price_diff
                        logger.info(f"📊 {symbol} SL/TP adjusted for slippage: SL={stop_loss:.4f} TP1={tp1:.4f}")
            else:
                # PAPER MODE: Simulate order execution (still creates position!)
                logger.info(f"📝 PAPER TRADE: Opening {side.upper()} {symbol} @ ${price:.4f} (simulated)")
                # Continue to position creation below - paper mode tracks positions!
            
            # === PHANTOM TRADE PREVENTION: Validate entry price before creating position ===
            if actual_entry_price <= 0:
                logger.error(f"🚨 PHANTOM TRADE BLOCKED: entry_price=${actual_entry_price} for {symbol} - NOT creating position")
                logger.error(f"🚨 Order may have failed silently. Check Bybit manually.")
                return
            
            # FIX: Cross-validate entry price against fresh ticker to catch stale price bugs
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                ticker_price = ticker.get('last', 0)
                if ticker_price and ticker_price > 0:
                    entry_deviation = abs(actual_entry_price - ticker_price) / ticker_price * 100
                    if entry_deviation > 50:  # Entry >50% off from current ticker = clearly wrong
                        logger.error(f"🚨 ENTRY PRICE MISMATCH: stored=${actual_entry_price:.6f} vs ticker=${ticker_price:.6f} ({entry_deviation:.1f}% off)")
                        logger.info(f"📊 Overriding entry price with ticker price for {symbol}")
                        # Recalculate everything with correct price
                        old_price = actual_entry_price
                        actual_entry_price = ticker_price
                        # Shift SL/TP by the price difference
                        price_diff = actual_entry_price - old_price
                        stop_loss = stop_loss + price_diff
                        tp1 = tp1 + price_diff
                        tp2 = tp2 + price_diff
                        tp3 = tp3 + price_diff
                        logger.info(f"📊 {symbol} SL/TP recalculated: Entry=${actual_entry_price:.6f} SL=${stop_loss:.6f} TP1=${tp1:.6f}")
            except Exception as val_err:
                logger.warning(f"⚠️ Could not validate entry price vs ticker: {val_err}")
            
            # Create Position object with entry snapshot for ML learning
            position = Position(
                symbol=symbol,
                side=side,
                entry_price=actual_entry_price,  # Use actual fill price
                size=size,
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                opened_at=datetime.now(timezone.utc),
                entry_df_snapshot=df.copy() if df is not None and len(df) >= 20 else None,
                dynamic_tp_pct={
                    'tp1_pct': dynamic_tp.get('tp1_pct', self.TP1_PROFIT_PCT),
                    'tp2_pct': dynamic_tp.get('tp2_pct', self.TP2_PROFIT_PCT),
                    'tp3_pct': dynamic_tp.get('tp3_pct', self.TP3_PROFIT_PCT),
                    'tp1_display': dynamic_tp.get('tp1_display', 0.25),
                    'factors': dynamic_tp.get('factors', 'static'),
                },
                entry_component_scores=ai_result.get('entry_component_scores') if ai_result else None,  # For Bayesian weight learning
            )
            
            # Debug: Log Bayesian component scores attachment
            if position.entry_component_scores:
                logger.info(f"📊 BAYESIAN: Attached {len(position.entry_component_scores)} component scores to {symbol} position")
            else:
                logger.warning(f"⚠️ BAYESIAN: No entry_component_scores for {symbol} - Bayesian learner will NOT update on close")
            
            # Store in multi-position dict
            async with self._position_lock:
                self.positions[symbol] = position
            
            # CRITICAL: Add symbol to WebSocket stream for real-time price updates
            if hasattr(self, 'ws_price_stream') and self.ws_price_stream:
                self.ws_price_stream.add_symbol(symbol)
                logger.info(f"🔌 Added {symbol} to WebSocket for real-time PnL tracking")
            
            # DO NOT set as primary position - keep primary position reserved for main symbol only
            # Even if self.position is None, we don't set multi-positions as primary
            # The _get_current_position_dict() will only return the main symbol's position

            logger.info(f"OPENED {side.upper()} [🤖 AI Multi-Pair] | {symbol} | Entry: {price:.4f} | Size: {size:.4f} | SL: {stop_loss:.4f} | TP1: {tp1:.4f}")
            
            # === SERVER-SIDE SL/TP (execute even if bot disconnects) ===
            try:
                server_orders = await self._place_server_side_sl_tp(
                    symbol=symbol,
                    side=side,
                    size=size,
                    entry_price=actual_entry_price,
                    stop_loss=stop_loss,
                    tp1=tp1,
                    tp2=tp2
                )
                if server_orders.get('sl_order_id'):
                    logger.info(f"🛡️ Server-side SL/TP placed for {symbol} - position protected")
            except Exception as e:
                logger.warning(f"⚠️ Server-side SL/TP failed for {symbol} (using client-side fallback): {e}")
            
            # CRITICAL: Save position to disk immediately so it survives restarts
            self._save_trading_state()
            
            # Notify via Telegram
            try:
                if self.telegram and self.telegram.enabled:
                    await self.telegram.notify_trade_opened(
                        symbol=symbol,
                        side=side.upper(),
                        entry_price=price,
                        size=size,
                        stop_loss=stop_loss,
                        tp1=tp1,
                        tp2=tp2,
                        tp3=tp3
                    )
            except Exception as tg_err:
                logger.warning(f"Telegram notification failed: {tg_err}")
            
            # === IMMEDIATE POSITION MONITORING ===
            # Start monitoring right away - don't wait for the 30s cycle
            # This ensures peak profit tracking begins immediately
            try:
                logger.info(f"🔍 Starting IMMEDIATE monitoring for {symbol}")
                # Get current price and ATR
                ticker = await self.exchange.fetch_ticker(symbol)
                current_price = ticker.get('last', price)
                current_atr = atr if atr else (ticker.get('high', price) - ticker.get('low', price)) / 14
                if current_atr <= 0:
                    current_atr = current_price * 0.01
                
                # Run immediate position check to initialize peak tracking
                await self._monitor_single_position(position, current_price, current_atr, symbol)
                logger.info(f"✅ Initial monitoring complete for {symbol}")
            except Exception as mon_err:
                logger.warning(f"⚠️ Immediate monitoring failed for {symbol}: {mon_err} (will catch on next cycle)")
            
        except Exception as e:
            logger.error(f"Error opening position for {symbol}: {e}")
            import traceback
            traceback.print_exc()
    
    async def _execute_unified_trade(
        self,
        trade_symbol: str,
        trade_price: float,
        trade_atr: float,
        scan_bars: pd.DataFrame,
        opportunity: Dict[str, Any]
    ):
        """
        Execute a trade from the UNIFIED MATH+AI scanner.
        
        This is the execution pathway for trades identified by the unified scanner.
        It applies AI power adjustments and uses the best available data.
        """
        try:
            # Normalize symbol
            symbol_clean = normalize_symbol(trade_symbol)
            ccxt_symbol = f"{symbol_clean.replace('USDT', '')}/USDT:USDT"
            
            # === CHECK PER-PAIR LOSS COOLDOWN ===
            symbol_key = symbol_clean.upper()
            if symbol_key in self.pair_loss_cooldown:
                cooldown_until = self.pair_loss_cooldown[symbol_key]
                if datetime.now(timezone.utc) < cooldown_until:
                    remaining = (cooldown_until - datetime.now(timezone.utc)).total_seconds() / 60
                    logger.info(f"⏰ SKIPPING {symbol_key}: On cooldown for {remaining:.1f} more minutes (recent loss)")
                    return
                else:
                    # Cooldown expired, remove it
                    del self.pair_loss_cooldown[symbol_key]
                    logger.info(f"✅ {symbol_key} cooldown expired - can trade again")
            
            # === CHECK RE-ENTRY COOLDOWN (same symbol+direction) ===
            _side_str = opportunity['action'].upper()
            reentry_key = f"{symbol_key}_{_side_str}"
            if reentry_key in self.reentry_cooldown:
                reentry_until = self.reentry_cooldown[reentry_key]
                if datetime.now(timezone.utc) < reentry_until:
                    remaining_s = (reentry_until - datetime.now(timezone.utc)).total_seconds()
                    logger.info(f"⏰ RE-ENTRY BLOCKED: {reentry_key} — wait {remaining_s:.0f}s before re-entering same direction")
                    return
                else:
                    del self.reentry_cooldown[reentry_key]
                    logger.info(f"✅ {reentry_key} re-entry cooldown expired")
            
            # Convert action to signal
            signal = 1 if opportunity['action'].upper() == 'LONG' else -1
            
            # Calculate risk based on combined score
            combined_score = opportunity.get('combined_score', 60)
            base_risk = self.RISK_PCT
            
            # Score-based risk adjustment (60-100 score range)
            if combined_score >= 80:
                risk_mult = 1.3  # High conviction = 30% more risk
            elif combined_score >= 70:
                risk_mult = 1.15  # Medium-high = 15% more
            else:
                risk_mult = 1.0  # Normal
            
            adjusted_risk = base_risk * risk_mult
            
            # Cap at max allowed
            max_risk_param = self.adaptive_params.get('risk_pct', {})
            max_risk = max_risk_param.get('current', 0.05) if isinstance(max_risk_param, dict) else 0.05
            adjusted_risk = min(adjusted_risk, max_risk)
            
            logger.info(f"🎯 UNIFIED TRADE EXECUTION:")
            logger.info(f"   Symbol: {ccxt_symbol}")
            logger.info(f"   Direction: {'LONG' if signal == 1 else 'SHORT'}")
            logger.info(f"   Combined Score: {combined_score:.0f}")
            logger.info(f"   Risk: {adjusted_risk:.2%}")
            
            # === ML VETO GATE: Block trades where ML predicts low win probability ===
            # Build features for ML prediction
            now_utc = datetime.now(timezone.utc)
            ml_features = {}
            try:
                if scan_bars is not None and len(scan_bars) > 10:
                    _close = scan_bars['close']
                    _rsi = scan_bars['rsi'].iloc[-1] if 'rsi' in scan_bars.columns else 50
                    _adx = scan_bars['adx'].iloc[-1] if 'adx' in scan_bars.columns else 0
                    _vol_ratio = scan_bars['volume_ratio'].iloc[-1] if 'volume_ratio' in scan_bars.columns else 1
                    _hurst = scan_bars['hurst'].iloc[-1] if 'hurst' in scan_bars.columns else 0.5
                    ml_features = {
                        'rsi': float(_rsi), 'adx': float(_adx),
                        'atr_percent': (trade_atr / trade_price) * 100 if trade_price > 0 else 0,
                        'volume_ratio': float(_vol_ratio), 'hurst': float(_hurst),
                        'sma_distance_percent': 0,
                        'hour': now_utc.hour, 'day_of_week': now_utc.weekday(),
                        'is_weekend': 1 if now_utc.weekday() >= 5 else 0,
                        'regime_trending': 1, 'regime_choppy': 0, 'regime_ranging': 0,
                        'is_long': 1 if signal == 1 else 0,
                        'rsi_slope': float(scan_bars['rsi'].diff(5).iloc[-1] / 5) if 'rsi' in scan_bars.columns and len(scan_bars) > 5 else 0,
                        'macd_hist': float(scan_bars['macd_hist'].iloc[-1]) if 'macd_hist' in scan_bars.columns else 0,
                        'price_momentum': float(_close.pct_change(5).iloc[-1] * 100) if len(_close) > 5 else 0,
                        'atr_expansion': 1.0, 'bb_position': 0.5,
                        'volume_trend': float(scan_bars['volume'].pct_change(10).iloc[-1] * 100) if 'volume' in scan_bars.columns and len(scan_bars) > 10 else 0,
                        'btc_correlation': 0, 'candle_strength': 0,
                        'higher_highs': 0, 'distance_from_support': 0.5, 'distance_from_resistance': 0.5,
                        'ai_confidence': opportunity.get('confidence', 0),
                        'system_score': combined_score, 'tech_score': opportunity.get('math_score', 0),
                        'is_london_session': 1 if 7 <= now_utc.hour <= 16 else 0,
                        'is_nyc_session': 1 if 13 <= now_utc.hour <= 22 else 0,
                        'is_asia_session': 1 if 0 <= now_utc.hour <= 9 else 0,
                    }
            except Exception as ml_err:
                logger.debug(f"ML feature build error (non-fatal): {ml_err}")
            
            # Get ML predictions
            ml_veto = False
            if ml_features:
                hist_pred = self.ml_historical.predict(ml_features)
                live_pred = self.ml_live.predict(ml_features)
                
                if hist_pred.get('available'):
                    logger.info(f"🧠 Historical ML: {hist_pred['win_probability']:.1%} win prob ({hist_pred['confidence']})")
                if live_pred.get('available'):
                    logger.info(f"🧠 Live ML: {live_pred['win_probability']:.1%} win prob ({live_pred['confidence']})")
                
                # Combine predictions
                hist_avail = hist_pred.get('available', False)
                live_avail = live_pred.get('available', False)
                if hist_avail and live_avail:
                    combined_ml_prob = hist_pred['win_probability'] * 0.4 + live_pred['win_probability'] * 0.6
                elif hist_avail:
                    combined_ml_prob = hist_pred['win_probability']
                elif live_avail:
                    combined_ml_prob = live_pred['win_probability']
                else:
                    combined_ml_prob = 0.5  # No ML available, neutral
                
                logger.info(f"🧠 Combined ML: {combined_ml_prob:.1%} win prob")
                
                # ML VETO: Block if ML says <40% win probability
                # In testing mode, lower threshold to 20% to let more trades through
                from ai_filter import get_threshold_mode as _get_tm
                ML_VETO_THRESHOLD = 0.20 if _get_tm() == 'testing' else 0.40
                if combined_ml_prob < ML_VETO_THRESHOLD and (hist_avail or live_avail):
                    logger.warning(f"🧠❌ ML VETO: Blocking trade — ML predicts {combined_ml_prob:.1%} win prob (< {ML_VETO_THRESHOLD:.0%} threshold)")
                    ml_veto = True
            
            if ml_veto:
                logger.info(f"🚫 Trade blocked by ML veto for {ccxt_symbol}")
                return
            
            inst_adjustment = 0
            
            # Build AI result for the _open_position_for_symbol call
            # === CONVICTION-SCALED SIZING (Rec #1) ===
            # Pull the 6-tier conviction system from ai_filter and override
            # the simple 3-tier risk_mult above with granular sizing.
            conviction_info = opportunity.get('conviction_info', {})
            if conviction_info:
                conv_mult = conviction_info.get('size_multiplier', risk_mult)
                conv_label = conviction_info.get('conviction', 'NORMAL')
            else:
                # Fallback: map combined_score to conviction tiers
                if combined_score >= 85:
                    conv_mult, conv_label = 1.5, 'VERY_HIGH'
                elif combined_score >= 75:
                    conv_mult, conv_label = 1.3, 'HIGH'
                elif combined_score >= 65:
                    conv_mult, conv_label = 1.15, 'ABOVE_AVERAGE'
                elif combined_score >= 55:
                    conv_mult, conv_label = 1.0, 'NORMAL'
                elif combined_score >= 45:
                    conv_mult, conv_label = 0.75, 'LOW'
                else:
                    conv_mult, conv_label = 0.5, 'MINIMAL'
            
            # === KELLY-BLENDED SIZING (Rec #1) ===
            # Pull Kelly fraction from risk_manager and blend with conviction
            kelly_mult = 1.0
            try:
                vol_pct = (trade_atr / trade_price * 100) if trade_price > 0 else 1.0
                kelly_info = self.risk_manager.get_adjusted_risk(
                    self.balance, self.peak_balance, vol_pct,
                    ai_confidence=opportunity.get('confidence', 0.5)
                )
                kelly_risk = kelly_info.get('kelly_risk', self.RISK_PCT)
                kelly_mult = kelly_risk / self.RISK_PCT if self.RISK_PCT > 0 else 1.0
                kelly_mult = max(0.5, min(1.5, kelly_mult))  # Clamp
                # Blend: 60% conviction + 40% Kelly
                blended_mult = conv_mult * 0.6 + kelly_mult * 0.4
                blended_mult = max(0.5, min(1.5, blended_mult))  # Final clamp
                logger.info(f"🎯 CONVICTION+KELLY SIZING: conv={conv_label}({conv_mult:.2f}x) kelly={kelly_mult:.2f}x → blended={blended_mult:.2f}x")
            except Exception as ks_err:
                logger.warning(f"⚠️ Kelly sizing fallback: {ks_err}")
                blended_mult = conv_mult
            
            ai_result = {
                'approved': True,
                'confidence': opportunity.get('confidence', 0.7),
                'reasoning': f"Unified Math+AI scan: Combined score {combined_score:.0f}",
                'ai_power': {
                    'size_multiplier': blended_mult,
                    'conviction': conv_label,
                    'confidence_boost': (combined_score - 60) / 100,
                    'kelly_mult': kelly_mult,
                    'conviction_mult': conv_mult,
                },
                # === BAYESIAN FEEDBACK FIX (Rec #3) ===
                # Pass component scores so they're stored on the Position object
                # and used for Bayesian weight updates when the trade closes.
                'entry_component_scores': opportunity.get('entry_component_scores', {})
            }
            
            # Track the decision
            self.current_decision_id = self.ai_tracker.record_decision(
                symbol=ccxt_symbol,
                signal_direction=opportunity['action'],
                price=trade_price,
                approved=True,
                confidence=opportunity.get('confidence', 0.7),
                reasoning=f"UNIFIED: Math {opportunity.get('math_score', 0):.0f} + AI {opportunity.get('ai_score', 0):.0f} = {combined_score:.0f}",
                threshold_used=40.0,
                regime="UNIFIED",
                tech_score=opportunity.get('math_score', 0),
                system_score=combined_score,
                ml_probability=opportunity.get('confidence', 0.5),
                ml_confidence="UNIFIED"
            )
            
            # ═══════════════════════════════════════════════════════════════
            # REC #2: CORRELATION GATE — Check before opening 2nd position
            # ═══════════════════════════════════════════════════════════════
            open_positions = [p for p in self.positions.values() if p is not None]
            if open_positions:
                # Gather DataFrames for existing positions
                existing_dfs = {}
                for pos in open_positions:
                    pos_sym = getattr(pos, 'symbol', '')
                    pos_df = getattr(pos, 'entry_df_snapshot', None)
                    if pos_sym and pos_df is not None and len(pos_df) > 0:
                        existing_dfs[pos_sym] = pos_df
                
                if existing_dfs:
                    try:
                        corr_result = self.risk_manager.check_correlation_gate(
                            candidate_df=scan_bars,
                            existing_positions_dfs=existing_dfs,
                            candidate_symbol=ccxt_symbol,
                            candidate_side='long' if signal == 1 else 'short',
                            max_correlation=0.75,
                            lookback=30
                        )
                        
                        if not corr_result['allowed']:
                            # Check if opposite directions provide natural hedge
                            existing_sides = [getattr(p, 'side', '').lower() for p in open_positions]
                            candidate_side = 'long' if signal == 1 else 'short'
                            is_hedge = any(s != candidate_side for s in existing_sides if s)
                            
                            if is_hedge:
                                # Natural hedge — allow with reduced size
                                logger.info(f"📊 CORRELATION GATE: High corr (ρ={corr_result['highest_corr']:.2f}) "
                                           f"but OPPOSITE sides → natural hedge, allowing with 50% size")
                                ai_result['ai_power']['size_multiplier'] *= 0.5
                                ai_result['ai_power']['correlation_reduced'] = True
                            else:
                                logger.warning(f"🚫 CORRELATION GATE BLOCKED: {ccxt_symbol} too correlated with "
                                              f"{corr_result['highest_corr_symbol']} (ρ={corr_result['highest_corr']:.2f}) "
                                              f"— same direction, no diversification benefit")
                                return
                        elif corr_result['adjusted_size_mult'] < 1.0:
                            # Moderate correlation — reduce size
                            ai_result['ai_power']['size_multiplier'] *= corr_result['adjusted_size_mult']
                            ai_result['ai_power']['correlation_reduced'] = True
                            logger.info(f"📊 CORRELATION: Size reduced by {(1-corr_result['adjusted_size_mult'])*100:.0f}% "
                                       f"(ρ={corr_result['highest_corr']:.2f})")
                    except Exception as corr_err:
                        logger.warning(f"⚠️ Correlation gate error (non-fatal): {corr_err}")
            
            # Execute using existing position opening logic
            await self._open_position_for_symbol(
                symbol=ccxt_symbol,
                signal=signal,
                price=trade_price,
                atr=trade_atr,
                df=scan_bars,  # Pass the dataframe for ML learning
                source="unified_scan",
                ai_reasoning=f"Unified Math+AI: {combined_score:.0f} score",
                ai_confidence=opportunity.get('confidence', 0.7),
                ai_result=ai_result
            )
            
            logger.info(f"✅ UNIFIED TRADE EXECUTED: {ccxt_symbol} {'LONG' if signal == 1 else 'SHORT'}")
            
        except Exception as e:
            logger.error(f"Unified trade execution error: {e}")
            import traceback
            traceback.print_exc()
    
    async def _open_position(self, signal: int, price: float, atr: float, source: str = "technical", 
                             ai_reasoning: str = "", ai_confidence: float = 0, ai_result: Dict = None):
        """Open a new position with intelligent risk management and AI-powered sizing."""
        # ═══════════════════════════════════════════════════════════════════
        # 🚨 CRITICAL: RISK MANAGER GATE - CHECK BEFORE ANY TRADE
        # This prevents trading when daily loss limits, weekly limits, or
        # max drawdown thresholds have been breached
        # ═══════════════════════════════════════════════════════════════════
        can_trade_result = self.risk_manager.can_trade(self.balance, self.initial_balance)
        if not can_trade_result['allowed']:
            logger.error(f"🚫 RISK MANAGER BLOCKED: {self.ccxt_symbol} - {can_trade_result['reason']}")
            return
        
        # Bybit Futures: Both LONG and SHORT supported
        side = "long" if signal == 1 else "short"
        
        # === SIDE FILTER CHECK (Telegram /longonly /shortonly /bothsides) ===
        allowed = getattr(self, 'allowed_sides', 'both').lower()
        if allowed != 'both':
            if allowed == 'long' and side == 'short':
                logger.warning(f"🚫 SHORT blocked by /longonly mode - only LONG trades allowed")
                return
            elif allowed == 'short' and side == 'long':
                logger.warning(f"🚫 LONG blocked by /shortonly mode - only SHORT trades allowed")
                return
        
        # === AI POWER: Get recommendations from AI analysis ===
        ai_power = {}
        if ai_result and 'ai_power' in ai_result:
            ai_power = ai_result.get('ai_power', {})
            ai_confidence = ai_result.get('confidence', ai_confidence)
            ai_reasoning = ai_result.get('reasoning', ai_reasoning)
            logger.info(f"🤖 AI POWER ENABLED: Conviction={ai_power.get('conviction', 'N/A')}, SizeMult={ai_power.get('size_multiplier', 1.0):.1f}x")
        
        # Intelligent pattern detection
        pattern = detect_candlestick_patterns(self.bars_agg) if len(self.bars_agg) >= 3 else {}
        
        # ML regime prediction
        ml_prediction = ml_predict_regime(self.bars_agg) if len(self.bars_agg) >= 50 else {}
        
        # === MATH-LEVEL POSITION SETUP VALIDATION ===
        math_validation = await self._validate_position_setup_math(signal, price, pd.DataFrame(self.bars_agg))
        
        for reason in math_validation['reasons_against']:
            logger.warning(f"⚠️ {reason}")
        
        # Log reasons for approval
        for reason in math_validation['reasons_for']:
            logger.info(f"✅ {reason}")
        
        # Smart drawdown-adjusted risk
        drawdown_info = calculate_drawdown_adjusted_risk(
            base_risk=self.RISK_PCT,
            current_balance=self.balance,
            peak_balance=self.peak_balance,
            consecutive_losses=self.consecutive_losses,
            consecutive_wins=self.consecutive_wins
        )
        
        # Use adjusted risk for position sizing
        adjusted_risk = drawdown_info['adjusted_risk']
        
        # Log intelligence
        if pattern.get('pattern'):
            logger.info(f"📊 Pattern: {pattern['pattern']} ({'Bullish' if pattern.get('bullish') else 'Bearish'})")
        if ml_prediction.get('is_trained'):
            logger.info(f"🧠 ML: {ml_prediction.get('ml_signal', 'N/A')} ({ml_prediction.get('ml_score', 0):.0%})")
        logger.info(f"🎯 Risk Mode: {drawdown_info['mode']} ({adjusted_risk:.1%} risk)")
        
        # Calculate stop loss with MINIMUM FLOOR and DYNAMIC MAXIMUM
        # Prevents noise-level stops that get clipped instantly
        # Dynamic max adapts to each coin's ATR (volatile coins get wider stops)
        
        # === PER-VOLATILITY SL ADJUSTMENT (Notebook #2 §8) ===
        # Low volatility (ATR < 0.5%): tighter SL (less dead capital)
        # Normal volatility: standard ATR_MULT
        # High volatility (ATR > 2%): wider SL (avoid whipsaws)
        vol_atr_pct = (atr / price) * 100 if price > 0 else 1.0
        if vol_atr_pct < 0.5:
            vol_sl_mult = 0.85  # Tighter — low vol, less noise
            logger.info(f"📏 Low-vol SL adjust: ATR={vol_atr_pct:.2f}% → ATR_MULT × 0.85")
        elif vol_atr_pct > 2.0:
            vol_sl_mult = 1.15  # Wider — high vol, avoid whipsaws
            logger.info(f"📏 High-vol SL adjust: ATR={vol_atr_pct:.2f}% → ATR_MULT × 1.15")
        else:
            vol_sl_mult = 1.0  # Normal
        
        atr_stop_distance = atr * self.ATR_MULT * vol_sl_mult
        min_stop_distance = price * self.MIN_STOP_PCT  # 0.30% minimum
        # Dynamic max: ATR × multiplier × 1.5 safety, clamped to floor/ceiling
        dynamic_max_pct = max(self.DYNAMIC_STOP_FLOOR, min((atr / price) * self.ATR_MULT * vol_sl_mult * 1.5, self.DYNAMIC_STOP_CEILING))
        max_stop_distance = price * dynamic_max_pct
        stop_distance = max(min_stop_distance, min(atr_stop_distance, max_stop_distance))  # Clamp between min and dynamic max
        
        if stop_distance > atr_stop_distance:
            logger.info(f"📏 Stop floor applied: ATR={atr_stop_distance:.4f} → Min={stop_distance:.4f} ({self.MIN_STOP_PCT:.1%})")
        elif stop_distance < atr_stop_distance:
            logger.info(f"📏 Stop cap applied: ATR={atr_stop_distance:.4f} → DynMax={stop_distance:.4f} ({dynamic_max_pct:.2%})")
        
        if side == "long":
            stop_loss = price - stop_distance
        else:
            stop_loss = price + stop_distance
        
        risk_per_unit = abs(price - stop_loss)
        risk_amount = self.balance * adjusted_risk  # Use adjusted risk
        
        # === REALISTIC COST MODEL ===
        # Account for slippage + fees in position sizing
        # Entry: price + slippage + fee
        # Exit: SL/TP - slippage + fee
        # Total cost = 2 * (slippage + fee) = ~0.6% of position
        effective_entry_price = price * (1 + self.SLIPPAGE_PCT) if side == "long" else price * (1 - self.SLIPPAGE_PCT)
        fee_cost_pct = self.FEE_TAKER * 2  # Open + close
        
        # Adjust size to account for costs
        # Net risk = (entry - SL) - (entry * fee_cost)
        cost_adjusted_risk_per_unit = risk_per_unit * (1 - fee_cost_pct)
        size = risk_amount / cost_adjusted_risk_per_unit if cost_adjusted_risk_per_unit > 0 else 0
        
        # === AI POWER: CONFIDENCE-BASED POSITION SIZING ===
        # Apply AI's recommended size multiplier based on conviction level
        ai_size_multiplier = ai_power.get('size_multiplier', 1.0) if ai_power else 1.0
        ai_conviction = ai_power.get('conviction', 'NORMAL') if ai_power else 'NORMAL'
        
        if self.AI_FULL_AUTONOMY and ai_size_multiplier != 1.0:
            original_size = size
            size = size * ai_size_multiplier
            logger.info(f"🤖 AI SIZE ADJUSTMENT: {ai_conviction} conviction → {ai_size_multiplier:.1f}x ({original_size:.4f} → {size:.4f})")
        
        # === CAUTION ZONE SIZE REDUCTION ===
        # Reduce position size by 50% when in shadow/caution zones near danger areas
        if ai_power:
            in_resistance_caution = ai_power.get('in_resistance_caution', False)
            in_support_caution = ai_power.get('in_support_caution', False)
            
            if in_resistance_caution and side == 'long':
                original_size = size
                size = size * 0.5  # Reduce by 50%
                logger.warning(f"⚠️ CAUTION ZONE (near resistance) - reducing LONG size: {original_size:.4f} → {size:.4f} (50%)")
            elif in_support_caution and side == 'short':
                original_size = size
                size = size * 0.5  # Reduce by 50%
                logger.warning(f"⚠️ CAUTION ZONE (near support) - reducing SHORT size: {original_size:.4f} → {size:.4f} (50%)")
        
        # === ABSOLUTE MAX DOLLAR LOSS PER TRADE ===
        # Hard cap: no single trade can risk more than 3% of balance in dollars
        MAX_RISK_DOLLAR = self.balance * 0.03  # 3% of balance (~$8 on $264)
        actual_dollar_risk = size * risk_per_unit
        if actual_dollar_risk > MAX_RISK_DOLLAR:
            old_size = size
            size = MAX_RISK_DOLLAR / risk_per_unit if risk_per_unit > 0 else 0
            logger.warning(f"🛡️ DOLLAR RISK CAP: ${actual_dollar_risk:.2f} exceeds max ${MAX_RISK_DOLLAR:.2f} — reducing size {old_size:.2f} → {size:.2f}")
        
        # === MAX LEVERAGE SAFETY CHECK ===
        # Prevent excessive leverage on tight stops
        MAX_LEVERAGE = int(self.adaptive_params.get('max_leverage', {}).get('current', 10))
        MAX_LEVERAGE = min(MAX_LEVERAGE, 10)  # HARD CAP: Never exceed 10x regardless of params
        position_value = size * price
        current_leverage = position_value / self.balance
        if current_leverage > MAX_LEVERAGE:
            old_size = size
            size = (self.balance * MAX_LEVERAGE) / price
            logger.warning(f"⚠️ Leverage {current_leverage:.1f}x exceeds max {MAX_LEVERAGE}x - reducing size {old_size:.2f} → {size:.2f}")
        
        logger.debug(f"Position sizing: Risk ${risk_amount:.2f}, Cost-adj R: ${cost_adjusted_risk_per_unit:.4f}, Size: {size:.4f}")
        
        # === EARLY MINIMUM SIZE CHECK ===
        # Check if position size meets minimum requirements before proceeding
        # If below minimum, request user confirmation via Telegram instead of blocking
        min_amount = 0.001  # Default fallback
        needs_confirmation = False
        
        if self.live_mode:
            try:
                # Get market info for minimum order size
                futures_symbol = self.SYMBOL if '/' in self.SYMBOL else f"{self.SYMBOL.replace('USDT', '')}/USDT:USDT"
                if not hasattr(self, '_market_cache'):
                    self._market_cache = {}
                
                if futures_symbol not in self._market_cache:
                    markets = await self.exchange.load_markets()
                    if futures_symbol in markets:
                        self._market_cache[futures_symbol] = markets[futures_symbol]
                
                market_info = self._market_cache.get(futures_symbol, {})
                limits = market_info.get('limits', {}).get('amount', {})
                min_amount = limits.get('min', 0.001)
                
                # Apply precision rounding
                precision = market_info.get('precision', {}).get('amount', 8)
                import math
                if isinstance(precision, int):
                    size = math.floor(size * (10 ** precision)) / (10 ** precision)
                
                if size < min_amount:
                    logger.warning(f"⚠️ Position size {size:.8f} below minimum {min_amount} for {futures_symbol}")
                    needs_confirmation = True
                else:
                    logger.debug(f"Minimum size check passed: {size:.8f} >= {min_amount}")
            except Exception as min_check_err:
                logger.warning(f"⚠️ Minimum size check failed: {min_check_err} - proceeding anyway")
        
        # === DYNAMIC MATH-BASED TP LEVELS ===
        r_value = risk_per_unit
        dtp = self.calculate_dynamic_tp(
            entry_price=price, side=side,
            df=pd.DataFrame(self.bars_agg),
            atr=atr, stop_loss=stop_loss
        )
        tp1 = dtp['tp1']
        tp2 = dtp['tp2']
        tp3 = dtp['tp3']
        dtp_tp1_pct = dtp['tp1_pct']
        dtp_tp2_pct = dtp['tp2_pct']
        dtp_tp3_pct = dtp['tp3_pct']
        
        logger.info(f"🧮 DYNAMIC TP (_open_position): TP1 +{dtp_tp1_pct:.2f}% @ ${tp1:.6f} | "
                    f"TP2 +{dtp_tp2_pct:.2f}% @ ${tp2:.6f} | TP3 +{dtp_tp3_pct:.2f}% @ ${tp3:.6f}")
        logger.info(f"🧮 TP factors: {dtp.get('factors', 'N/A')}")
        
        # === REQUEST USER CONFIRMATION FOR BELOW-MINIMUM TRADES ===
        if needs_confirmation and self.live_mode and self.telegram.enabled:
            logger.info(f"📱 Requesting user confirmation for below-minimum size trade...")
            
            # Request confirmation via Telegram
            await self.telegram.request_trade_confirmation(
                symbol=self.SYMBOL,
                side=side,
                price=price,
                calculated_size=size,
                min_amount=min_amount,
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                risk_pct=adjusted_risk,
                balance=self.balance,
                reason=f"Size {size:.6f} below minimum {min_amount}"
            )
            
            # Store trade details for later execution
            self._pending_trade = {
                'signal': signal,
                'price': price,
                'atr': atr,
                'side': side,
                'size': size,
                'min_amount': min_amount,
                'stop_loss': stop_loss,
                'tp1': tp1,
                'tp2': tp2,
                'tp3': tp3,
                'symbol': self.SYMBOL,
                'adjusted_risk': adjusted_risk,
                'math_validation': math_validation
            }
            
            logger.info(f"⏳ Trade pending user confirmation - check Telegram")
            return  # Wait for user confirmation
        
        # Adjust position size by math confidence
        size = size * math_validation['size_adjustment']
        if math_validation['size_adjustment'] != 1.0:
            logger.info(f"📊 Position size: {math_validation['size_adjustment']:.0%} (regime/volatility/edge based)")
        
        # === CRITICAL: Verify available margin before trading ===
        # Use lock to prevent race condition between check and execution
        # Sync balance from exchange to ensure we have latest available funds
        if self.live_mode:
            async with self._balance_lock:  # RACE CONDITION FIX: Protect margin check + execution
                try:
                    # Initialize defaults
                    wallet_data = None
                    borrowed_amount = 0
                    available_balance = self.balance
                    
                    # For Bybit UTA (Unified Trading Account), use proper wallet balance API
                    # ccxt's fetch_balance doesn't account for borrowed funds correctly
                    try:
                        wallet_info = await self.exchange.private_get_v5_account_wallet_balance({'accountType': 'UNIFIED'})
                        wallet_data = wallet_info.get('result', {}).get('list', [{}])[0]
                        available_balance = float(wallet_data.get('totalAvailableBalance', 0))
                        for coin in wallet_data.get('coin', []):
                            if coin.get('coin') == 'USDT':
                                borrowed_amount = float(coin.get('borrowAmount', 0) or 0)
                                break
                        if borrowed_amount > 0:
                            logger.info(f"💳 Borrowed USDT: ${borrowed_amount:.2f} (affects available margin)")
                    except Exception as wallet_err:
                        logger.warning(f"⚠️ Could not fetch UTA wallet: {wallet_err} - falling back to standard balance")
                        exchange_balance = await self.exchange.fetch_balance()
                        available_balance = float(exchange_balance['USDT']['free'])
                    
                    # Calculate required margin using configured leverage
                    ACTUAL_LEVERAGE = int(self.adaptive_params.get('max_leverage', {}).get('current', 10))
                    notional_value = size * price
                    required_margin = notional_value / ACTUAL_LEVERAGE
                    
                    # Add 20% buffer for fees, slippage, and Bybit's margin requirements
                    required_margin_with_buffer = required_margin * 1.2
                    
                    # Cache UTA info for dashboard
                    uta_wallet_balance = float(wallet_data.get('totalWalletBalance', 0)) if wallet_data else self.balance
                    uta_equity = float(wallet_data.get('totalEquity', 0)) if wallet_data else self.balance
                    uta_margin_used = float(wallet_data.get('totalInitialMargin', 0)) if wallet_data else 0
                    uta_ltv = float(wallet_data.get('accountLTV', 0)) * 100 if wallet_data else 0
                    
                    self._cached_uta_info = {
                        'wallet_balance': uta_wallet_balance,
                        'equity': uta_equity,
                        'borrowed': borrowed_amount,
                        'available_margin': available_balance,
                        'margin_used': uta_margin_used,
                        'ltv': uta_ltv,
                    }
                    
                    # Check minimum margin threshold
                    if available_balance < type(self).MIN_MARGIN_THRESHOLD:
                        logger.warning(f"⚠️ Available margin ${available_balance:.2f} below minimum ${type(self).MIN_MARGIN_THRESHOLD:.2f} - skipping trade")
                        logger.warning(f"💡 Tip: Repay some of your ${borrowed_amount:.2f} loan or deposit more USDT")
                        return
                    
                    if required_margin_with_buffer > available_balance:
                        # Reduce size to fit available margin
                        max_notional = (available_balance / 1.2) * ACTUAL_LEVERAGE
                        old_size = size
                        size = max_notional / price
                        logger.warning(f"⚠️ Margin check: Required ${required_margin_with_buffer:.2f} > Available ${available_balance:.2f}")
                        logger.warning(f"⚠️ Reducing position size: {old_size:.4f} → {size:.4f}")
                        
                        # Check if reduced size is too small (minimum $10 notional)
                        min_notional = 10.0
                        if size * price < min_notional:
                            logger.warning(f"⚠️ Reduced position too small (${size * price:.2f} < ${min_notional:.2f}) - skipping trade")
                            return
                    else:
                        logger.info(f"💰 Margin check OK: ${required_margin:.2f} required, ${available_balance:.2f} available")
                except Exception as margin_err:
                    logger.warning(f"⚠️ Could not verify margin: {margin_err} - proceeding with calculated size")
        
        # Store entry snapshot for ML learning (reduced requirement from 50 to 20 bars)
        entry_snapshot = self.bars_agg.copy() if len(self.bars_agg) >= 20 else None
        if entry_snapshot is None:
            logger.warning(f"⚠️ ML snapshot not saved - only {len(self.bars_agg)} bars available (need 20)")
        
        # Store entry features for dual ML learning at close time
        _ml_entry_features = None
        
        # === DRY-RUN MODE: Shadow-track but don't execute ===
        if self.dry_run_mode:
            self.dry_run_tracker.open_shadow(
                symbol=self.SYMBOL,
                side=side,
                entry_price=price,
                size=size,
                stop_loss=stop_loss,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                source=source,
                ai_confidence=ai_confidence,
            )
            if self.telegram.enabled:
                open_count = len(self.dry_run_tracker.shadow_positions)
                total_shadow = self.dry_run_tracker.stats['total_trades']
                await self.telegram.send_message(
                    f"👻 *DRY-RUN SHADOW OPEN*\n\n"
                    f"Side: `{side.upper()}`\n"
                    f"Entry: `${price:.4f}`\n"
                    f"Size: `{size:.4f}`\n"
                    f"SL: `${stop_loss:.4f}`\n"
                    f"TP1: `${tp1:.4f}`\n"
                    f"TP2: `${tp2:.4f}`\n"
                    f"TP3: `${tp3:.4f}`\n"
                    f"Source: `{source}`\n\n"
                    f"📊 Shadows: {open_count} open | {total_shadow} completed\n"
                    f"_Trade NOT executed — shadow tracking only_"
                )
            return  # Don't actually open position
        
        # === LIVE MODE: Execute real order ===
        actual_entry_price = price
        if self.live_mode:
            # Use balance lock to prevent concurrent order placement (race condition)
            async with self._balance_lock:
                order_side = 'buy' if side == 'long' else 'sell'
                order = await self._execute_market_order(self.SYMBOL, order_side, size)
            
            if not order:
                logger.error(f"🚨 LIVE ORDER FAILED - Position NOT opened")
                return  # Order failed, don't create position
            
            # Use actual fill price if available
            if order.get('average'):
                actual_entry_price = float(order['average'])
                logger.info(f"📊 Fill price: ${actual_entry_price:.4f} (requested: ${price:.4f})")
                
                # Recalculate SL/TPs based on actual fill
                slippage = abs(actual_entry_price - price) / price
                if slippage > 0.005:  # >0.5% slippage warning
                    logger.warning(f"⚠️ High slippage: {slippage:.2%}")
        
        # === PHANTOM TRADE PREVENTION: Validate entry price before creating position ===
        if actual_entry_price <= 0:
            logger.error(f"🚨 PHANTOM TRADE BLOCKED: entry_price=${actual_entry_price} for {self.SYMBOL} - NOT creating position")
            logger.error(f"🚨 Order may have failed silently. Check Bybit manually.")
            return
        
        # FIX: Cross-validate entry price against fresh ticker to catch stale price bugs
        try:
            ticker = await self.exchange.fetch_ticker(self.SYMBOL)
            ticker_price = ticker.get('last', 0)
            if ticker_price and ticker_price > 0:
                entry_deviation = abs(actual_entry_price - ticker_price) / ticker_price * 100
                if entry_deviation > 50:  # Entry >50% off from current ticker = clearly wrong
                    logger.error(f"🚨 ENTRY PRICE MISMATCH: stored=${actual_entry_price:.6f} vs ticker=${ticker_price:.6f} ({entry_deviation:.1f}% off)")
                    logger.info(f"📊 Overriding entry price with ticker price for {self.SYMBOL}")
                    old_price = actual_entry_price
                    actual_entry_price = ticker_price
                    # Shift SL/TP by the price difference
                    price_diff = actual_entry_price - old_price
                    stop_loss = stop_loss + price_diff
                    tp1 = tp1 + price_diff
                    tp2 = tp2 + price_diff
                    tp3 = tp3 + price_diff
                    logger.info(f"📊 {self.SYMBOL} SL/TP recalculated: Entry=${actual_entry_price:.6f} SL=${stop_loss:.6f} TP1=${tp1:.6f}")
        except Exception as val_err:
            logger.warning(f"⚠️ Could not validate entry price vs ticker: {val_err}")
        
        self.position = Position(
            symbol=self.SYMBOL,
            side=side,
            entry_price=actual_entry_price,  # Use actual fill price
            size=size,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            original_r_value=abs(actual_entry_price - stop_loss),  # Store for trailing after BE
            entry_df_snapshot=entry_snapshot,
            ml_entry_features=_ml_entry_features,  # For dual ML recording at close
            entry_component_scores=ai_result.get('entry_component_scores') if ai_result else None,  # For Bayesian weight learning
            dynamic_tp_pct={
                'tp1_pct': dtp_tp1_pct,
                'tp2_pct': dtp_tp2_pct,
                'tp3_pct': dtp_tp3_pct,
                'tp1_display': round(dtp_tp1_pct, 4),
                'factors': dtp.get('factors', ''),
            }
        )
        
        # CRITICAL: Add to positions dict for multi-pair tracking
        self.positions[self.SYMBOL] = self.position
        
        # CRITICAL: Save position to disk immediately so it survives restarts
        self._save_trading_state()
        
        # === SERVER-SIDE SL/TP (execute even if bot disconnects) ===
        # PhD-optimal: Bybit conditional orders for guaranteed execution
        # Only in LIVE mode — paper mode doesn't need exchange orders
        if self.live_mode:
            try:
                server_orders = await self._place_server_side_sl_tp(
                    symbol=self.SYMBOL,
                    side=side,
                    size=size,
                    entry_price=actual_entry_price,
                    stop_loss=stop_loss,
                    tp1=tp1,
                    tp2=tp2
                )
                if server_orders.get('sl_order_id'):
                    logger.info(f"🛡️ Server-side SL/TP placed - position protected even if bot disconnects")
            except Exception as e:
                logger.warning(f"⚠️ Server-side SL/TP failed (using client-side fallback): {e}")
        
        mode_label = "📝 PAPER" if not self.live_mode else ""
        source_label = "🤖 AI" if "ai" in source else "📊 Technical"
        logger.info(
            f"OPENED {side.upper()} [{source_label}] {mode_label} | Entry: {price:.4f} | Size: {size:.4f} | "
            f"SL: {stop_loss:.4f} | TP1: {tp1:.4f} | TP2: {tp2:.4f} | TP3: {tp3:.4f}"
        )
        
        if self.telegram.enabled:
            # Use AI-specific notification if this is an AI trade
            if "ai" in source.lower() and ai_confidence > 0:
                await self.telegram.notify_ai_trade(
                    symbol=self.SYMBOL,
                    action=side.upper(),
                    price=price,
                    confidence=ai_confidence,
                    reasoning=ai_reasoning or "AI Autonomous",
                    mode="autonomous"
                )
            else:
                await self.telegram.notify_trade_opened(
                    symbol=self.SYMBOL,
                    side=side.upper(),
                    entry_price=price,
                    size=size,
                    stop_loss=stop_loss,
                    tp1=tp1,
                    tp2=tp2,
                    tp3=tp3
                )
        
        # === IMMEDIATE POSITION MONITORING ===
        # Start monitoring right away - don't wait for the 30s cycle
        # This ensures peak profit tracking begins immediately
        try:
            logger.info(f"🔍 Starting IMMEDIATE monitoring for {self.SYMBOL}")
            await self._monitor_single_position(self.position, actual_entry_price, atr, self.SYMBOL)
            logger.info(f"✅ Initial monitoring complete for {self.SYMBOL}")
        except Exception as mon_err:
            logger.warning(f"⚠️ Immediate monitoring failed: {mon_err} (will catch on next cycle)")
    
    async def _math_aware_sl_adjustment(self, math_score: int, unrealized_pnl_pct: float, 
                                         current_price: float, position_side: str):
        """Dynamically adjust SL based on math scoring + AI validation.
        
        Architecture: MATH-FIRST, AI-VALIDATES
        - Math makes the primary decision
        - AI validates borderline cases (scores 40-55)
        - If AI unavailable, math decision stands
        
        Logic:
        - Strong math (hold >= 70) → Keep SL as-is (let it run)
        - Weak math (hold < 50) + profit (>= 0.3%) → Move SL to breakeven early
        - Very weak math (hold < 35) + profit (> 0.5%) → Lock 50% of profit
        - Borderline (40-55) → Ask AI to validate adjustment
        """
        if not self.position:
            return
        
        pos = self.position
        
        # Skip if already at breakeven or better
        is_long = pos.side == "long"
        at_breakeven = (is_long and pos.stop_loss >= pos.entry_price) or \
                       (not is_long and pos.stop_loss <= pos.entry_price)
        
        # Strong math - let it run, no early tightening
        if math_score >= 70:
            return
        
        # Calculate profit distance
        profit_distance = abs(current_price - pos.entry_price)
        sl_distance = abs(pos.entry_price - pos.stop_loss)
        
        # Determine math decision
        math_action = None
        math_reason = ""
        
        if not at_breakeven and math_score < 50 and unrealized_pnl_pct >= 0.3:
            math_action = "breakeven"
            math_reason = f"Weak hold ({math_score}) + profit ({unrealized_pnl_pct:+.2f}%)"
        elif math_score < 35 and unrealized_pnl_pct > 0.5:
            math_action = "lock_50"
            math_reason = f"Very weak hold ({math_score}) + good profit ({unrealized_pnl_pct:+.2f}%)"
        
        if not math_action:
            return
        
        # === APPLY MATH DECISION IMMEDIATELY (SPEED FIRST) ===
        # For critical SL adjustments, apply math first, AI validates after
        import time
        start_time = time.perf_counter()
        
        # Very weak scores: Apply immediately without AI (< 40)
        if math_score < 40:
            self._apply_sl_adjustment(pos, math_action, profit_distance, is_long, "[MATH]", math_reason)
            elapsed = (time.perf_counter() - start_time) * 1000
            logger.debug(f"⚡ SL adjustment applied in {elapsed:.1f}ms (math-only)")
            return
        
        # Borderline (40-55): Try AI with SHORT timeout, default to math
        ai_validated = False
        ai_agrees = True
        
        if 40 <= math_score <= 55 and self.ai_filter and hasattr(self.ai_filter, '_is_ai_available'):
            if self.ai_filter._is_ai_available():
                try:
                    # Use asyncio.wait_for with 1.5 second timeout
                    import asyncio
                    ai_result = await asyncio.wait_for(
                        self._ai_validate_sl_adjustment(
                            math_action=math_action,
                            math_score=math_score,
                            unrealized_pnl_pct=unrealized_pnl_pct,
                            current_price=current_price,
                            position_side=position_side
                        ),
                        timeout=1.5  # 1.5 second max
                    )
                    ai_validated = True
                    ai_agrees = ai_result.get('agrees', True)
                    ai_note = ai_result.get('note', '')
                    
                    if not ai_agrees:
                        logger.info(f"🤖 AI overrides SL adjustment ({(time.perf_counter() - start_time)*1000:.0f}ms): {ai_note}")
                        return  # AI says don't adjust
                    else:
                        logger.debug(f"🤖 AI confirms SL ({(time.perf_counter() - start_time)*1000:.0f}ms)")
                except asyncio.TimeoutError:
                    logger.debug(f"⚡ AI timeout, using math decision")
                    ai_validated = False
                except Exception as e:
                    logger.debug(f"AI SL validation failed, using math: {e}")
                    ai_validated = False
        
        # === APPLY THE ADJUSTMENT ===
        validation_tag = "[AI✓]" if ai_validated and ai_agrees else "[MATH]"
        self._apply_sl_adjustment(pos, math_action, profit_distance, is_long, validation_tag, math_reason)
        
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug(f"⚡ SL adjustment total time: {elapsed:.1f}ms")
    
    def _apply_sl_adjustment(self, pos, math_action: str, profit_distance: float, 
                              is_long: bool, tag: str, reason: str):
        """Apply SL adjustment - fast, no async."""
        if math_action == "breakeven":
            old_sl = pos.stop_loss
            pos.stop_loss = pos.entry_price
            logger.info(f"🧮 {tag} SL ADJUST: {reason} → Early breakeven")
            logger.info(f"   SL moved: {old_sl:.4f} → {pos.stop_loss:.4f}")
            self._save_trading_state()
            
        elif math_action == "lock_50":
            if is_long:
                lock_price = pos.entry_price + (profit_distance * 0.5)
                if lock_price > pos.stop_loss:
                    old_sl = pos.stop_loss
                    pos.stop_loss = lock_price
                    logger.info(f"🧮 {tag} SL ADJUST: {reason} → Locking 50% profit")
                    logger.info(f"   SL moved: {old_sl:.4f} → {pos.stop_loss:.4f}")
                    self._save_trading_state()
            else:
                lock_price = pos.entry_price - (profit_distance * 0.5)
                if lock_price < pos.stop_loss:
                    old_sl = pos.stop_loss
                    pos.stop_loss = lock_price
                    logger.info(f"🧮 {tag} SL ADJUST: {reason} → Locking 50% profit")
                    logger.info(f"   SL moved: {old_sl:.4f} → {pos.stop_loss:.4f}")
                    self._save_trading_state()
    
    async def _ai_validate_sl_adjustment(self, math_action: str, math_score: int,
                                          unrealized_pnl_pct: float, current_price: float,
                                          position_side: str) -> dict:
        """Ask AI to validate a borderline SL adjustment decision.
        
        Returns: {'agrees': bool, 'note': str}
        """
        try:
            prompt = f"""You are validating a stop-loss adjustment decision.

POSITION:
- Side: {position_side}
- Entry: ${self.position.entry_price:.4f}
- Current: ${current_price:.4f}
- Current SL: ${self.position.stop_loss:.4f}
- Unrealized PnL: {unrealized_pnl_pct:+.2f}%

MATH DECISION:
- Hold score: {math_score}/100 (borderline: 40-55)
- Proposed action: {math_action}
  - breakeven = Move SL to entry price
  - lock_50 = Move SL to lock 50% of current profit

Should we proceed with this SL adjustment?
Consider: Is the profit secure enough? Is the math score truly weak? Market volatility?

Reply in JSON:
{{"agrees": true/false, "note": "brief reason (max 30 words)"}}"""

            response = await self.ai_filter._generate_content(prompt)
            if response:
                import json
                import re
                # Extract JSON
                json_match = re.search(r'\{[^}]+\}', response)
                if json_match:
                    return json.loads(json_match.group())
        except Exception as e:
            logger.debug(f"AI SL validation parse error: {e}")
        
        return {'agrees': True, 'note': 'AI unavailable, using math decision'}
    
    # =====================================================================
    # WEBSOCKET REAL-TIME PRICE CALLBACK
    # =====================================================================
    # PhD-optimal: Process EVERY price tick for instant TP/SL execution
    # This is called by WebSocketPriceStream on every price update
    # =====================================================================
    
    async def _on_realtime_price_update(self, symbol: str, price: float, timestamp: datetime):
        """Handle real-time price update from WebSocket.
        
        This is called on EVERY price tick - much faster than 5-second polling.
        Checks TP/SL levels instantly when price hits them.
        
        Args:
            symbol: The symbol that updated (e.g., 'SEI/USDT:USDT')
            price: Current price
            timestamp: When the price was received
        """
        # Normalize symbol for comparison
        symbol_key = symbol.replace('/USDT:USDT', 'USDT').replace('/', '')
        

        
        # === CRITICAL: Update _symbol_prices cache for REAL-TIME dashboard updates ===
        if not hasattr(self, '_symbol_prices'):
            self._symbol_prices = {}
        
        # === FIX: Validate price BEFORE updating _symbol_prices for positioned symbols ===
        # If we have a position in this symbol, check the price is reasonable
        # before writing it to the cache that the dashboard reads for PnL display.
        # Garbage WS prices (e.g., $0.17 when entry was $0.078) would show -120% PnL.
        has_position_in_symbol = False
        position_entry_price = None
        
        if self.position:
            pos_sym_key = self.position.symbol.replace('/USDT:USDT', 'USDT').replace('/', '')
            if pos_sym_key == symbol_key:
                has_position_in_symbol = True
                position_entry_price = self.position.entry_price
        
        if not has_position_in_symbol:
            for sym, pos in self.positions.items():
                if pos is not None:
                    p_key = sym.replace('/USDT:USDT', 'USDT').replace('/', '')
                    if p_key == symbol_key:
                        has_position_in_symbol = True
                        position_entry_price = pos.entry_price
                        break
        
        if has_position_in_symbol and position_entry_price and position_entry_price > 0:
            # Validate: reject if price deviates >500% from entry (clearly garbage)
            deviation_pct = abs(price - position_entry_price) / position_entry_price * 100
            if deviation_pct > 500:
                logger.debug(f"🚨 WS price rejected for {symbol_key}: ${price:.6f} is {deviation_pct:.1f}% from entry ${position_entry_price:.6f}")
                return  # Don't update anything with garbage data
            # Price is reasonable - update cache
            self._symbol_prices[symbol_key] = price
        else:
            # No position - always accept (for scanner/dashboard display)
            self._symbol_prices[symbol_key] = price
        
        # Check primary position
        if self.position:
            pos_symbol_key = self.position.symbol.replace('/USDT:USDT', 'USDT').replace('/', '')
            
            if pos_symbol_key == symbol_key:
                # Validate price BEFORE updating cached prices
                if not validate_price(price, self.position.entry_price, symbol):
                    return
                
                # Update cached price (only after validation)
                self._set_realtime_price(price)
                self.cached_last_price = price
                
                # === INSTANT TP/SL CHECK ===
                pos = self.position
                
                # ===== EXTRA: Check if price beyond SL is reasonable =====
                sl_distance_pct = abs(pos.stop_loss - pos.entry_price) / pos.entry_price * 100
                price_triggers_sl = (pos.side == "long" and price <= pos.stop_loss) or \
                                   (pos.side == "short" and price >= pos.stop_loss)
                if price_triggers_sl:
                    if pos.side == "long":
                        price_beyond_sl_pct = abs(pos.stop_loss - price) / pos.entry_price * 100
                    else:
                        price_beyond_sl_pct = abs(price - pos.stop_loss) / pos.entry_price * 100
                    
                    if price_beyond_sl_pct > sl_distance_pct * 3:
                        logger.error(f"🚨 WS GARBAGE DATA {symbol}: Price ${price:.4f} is {price_beyond_sl_pct:.1f}% beyond SL")
                        return  # Don't trigger SL with garbage data
                
                # Check Stop Loss (instant execution!)
                if (pos.side == "long" and price <= pos.stop_loss) or \
                   (pos.side == "short" and price >= pos.stop_loss):
                    ws_held_key = symbol.replace('/USDT:USDT', 'USDT').replace('/', '')
                    if self.held_positions.get(ws_held_key, False):
                        # 💎 DIAMOND HANDS: SL breached but riding it out
                        ws_pnl = ((price - pos.entry_price) / pos.entry_price * 100) if pos.side == 'long' else ((pos.entry_price - price) / pos.entry_price * 100)
                        # Only log every ~30 seconds to avoid spam (use modulo on price)
                        if not hasattr(self, '_dh_last_log') or (datetime.now(timezone.utc) - self._dh_last_log.get(ws_held_key, datetime.min.replace(tzinfo=timezone.utc))).total_seconds() > 30:
                            logger.warning(f"💎 DIAMOND HANDS: {symbol} SL breached @ ${price:.4f} (SL: ${pos.stop_loss:.4f}) | PnL: {ws_pnl:+.2f}% — HOLDING")
                            if not hasattr(self, '_dh_last_log'):
                                self._dh_last_log = {}
                            self._dh_last_log[ws_held_key] = datetime.now(timezone.utc)
                    else:
                        logger.info(f"🔴 WEBSOCKET SL HIT: {symbol} @ ${price:.4f} (SL: ${pos.stop_loss:.4f})")
                        await self._close_position("Stop Loss (WebSocket)", price)
                        return
                
                # Check Trailing Stop (instant execution!)
                if pos.trailing_stop:
                    if (pos.side == "long" and price <= pos.trailing_stop) or \
                       (pos.side == "short" and price >= pos.trailing_stop):
                        trail_held_key = symbol.replace('/USDT:USDT', 'USDT').replace('/', '')
                        if self.held_positions.get(trail_held_key, False):
                            logger.warning(f"💎 DIAMOND HANDS: Trail stop hit but HOLD active for {symbol} @ ${price:.4f}")
                        else:
                            logger.info(f"🔴 WEBSOCKET TRAIL HIT: {symbol} @ ${price:.4f} (Trail: ${pos.trailing_stop:.4f})")
                            await self._close_position("Trailing Stop (WebSocket)", price)
                            return
                
                # Check Take Profit 1
                if not pos.tp1_hit:
                    if (pos.side == "long" and price >= pos.tp1) or \
                       (pos.side == "short" and price <= pos.tp1):
                        logger.info(f"🟢 WEBSOCKET TP1 HIT: {symbol} @ ${price:.4f} (TP1: ${pos.tp1:.4f})")
                        await self._hit_tp(1, price)
                
                # Check Take Profit 2
                if not pos.tp2_hit and pos.tp1_hit:
                    if (pos.side == "long" and price >= pos.tp2) or \
                       (pos.side == "short" and price <= pos.tp2):
                        logger.info(f"🟢 WEBSOCKET TP2 HIT: {symbol} @ ${price:.4f} (TP2: ${pos.tp2:.4f})")
                        await self._hit_tp(2, price)
                
                # Check Take Profit 3
                if not pos.tp3_hit and pos.tp2_hit:
                    if (pos.side == "long" and price >= pos.tp3) or \
                       (pos.side == "short" and price <= pos.tp3):
                        logger.info(f"🟢 WEBSOCKET TP3 HIT: {symbol} @ ${price:.4f} (TP3: ${pos.tp3:.4f})")
                        await self._hit_tp(3, price)
                        await self._close_position("TP3 Hit (WebSocket)", price)
                
                # Update trailing stop if in profit and TP1 hit
                # Using MOMENTUM-AWARE trailing that adapts to market conditions
                if pos.tp1_hit and pos.trailing_stop:
                    # FIXED: Use stored r_value instead of recalculating (was 0 after breakeven)
                    r_value = pos.original_r_value if pos.original_r_value > 0 else abs(pos.entry_price - pos.stop_loss)
                    
                    # Get momentum-adjusted trail offset
                    trail_offset = await self._get_momentum_trail_offset(
                        price, pos.entry_price, pos.side, r_value
                    )
                    
                    trail_moved = False
                    if pos.side == "long":
                        new_trail = price - (r_value * trail_offset)
                        if new_trail > pos.trailing_stop:
                            pos.trailing_stop = new_trail
                            trail_moved = True
                    else:
                        new_trail = price + (r_value * trail_offset)
                        if new_trail < pos.trailing_stop:
                            pos.trailing_stop = new_trail
                            trail_moved = True
                    
                    # === SERVER-SIDE TRAILING STOP SYNC ===
                    # Sync trail to Bybit so position is protected even if bot disconnects
                    # Rate-limited: max 1 server update per 30 seconds
                    if trail_moved and self.live_mode:
                        now_ts = datetime.now(timezone.utc)
                        last_sync = getattr(pos, '_last_trail_sync', None)
                        if last_sync is None or (now_ts - last_sync).total_seconds() >= 30:
                            try:
                                await self._update_server_side_sl(pos.symbol, pos.trailing_stop, pos.remaining_size, pos.side)
                                pos._last_trail_sync = now_ts
                                logger.info(f"🔄 Server SL synced to trail @ ${pos.trailing_stop:.4f}")
                            except Exception as sync_err:
                                logger.warning(f"⚠️ Trail sync to server failed: {sync_err}")
                
                # === REAL-TIME PEAK TRACKING (Primary Position) ===
                # Track peak profit % in real-time for intelligent exit
                # ALWAYS update if current pnl is higher than recorded peak
                # This ensures we catch peaks even if they happen between checks
                entry_price = pos.entry_price
                if pos.side == "long":
                    pnl_pct = ((price - entry_price) / entry_price) * 100
                else:
                    pnl_pct = ((entry_price - price) / entry_price) * 100
                
                # Initialize peak to very negative if not set (so any pnl beats it)
                current_peak_pct = getattr(pos, 'peak_profit_pct', -999)
                if current_peak_pct == -999:
                    pos.peak_profit_pct = pnl_pct  # First tick - set peak to current
                    current_peak_pct = pnl_pct
                
                # Update peak if current profit is higher (works for negative too!)
                if pnl_pct > current_peak_pct:
                    pos.peak_profit_pct = pnl_pct
                    # Estimate USD based on position value
                    position_value = getattr(pos, 'size', 0) * entry_price
                    pos.peak_profit_usd = pnl_pct * position_value / 100
                    # Only log significant new peaks (>0.05% above previous, and positive)
                    if pnl_pct > 0 and pnl_pct > current_peak_pct + 0.05:
                        logger.info(f"📈 {symbol}: RT NEW PEAK! {pnl_pct:.2f}% (${pos.peak_profit_usd:.2f})")
                    # SYNC peak to ai_filter cache immediately (every tick)
                    # This ensures check_realtime_quick_exit always has the latest peak
                    if hasattr(self, 'ai_filter') and self.ai_filter and pnl_pct > 0:
                        self.ai_filter.update_peak_profit(symbol, pnl_pct)
                
                # === ⚡ REAL-TIME QUICK EXIT CHECK (FAST-AS-LIGHT) ===
                # Check cached momentum to see if we should exit immediately
                # This runs on every WebSocket tick, not every 30 seconds!
                if hasattr(self, 'ai_filter') and self.ai_filter:
                    # Calculate hold time
                    import time
                    opened_at = getattr(pos, 'opened_at', None)
                    if opened_at:
                        try:
                            if isinstance(opened_at, datetime):
                                hold_seconds = time.time() - opened_at.timestamp()
                            elif isinstance(opened_at, str):
                                opened_dt = datetime.fromisoformat(opened_at.replace('Z', '+00:00'))
                                hold_seconds = time.time() - opened_dt.timestamp()
                            else:
                                hold_seconds = 999
                        except Exception:
                            hold_seconds = 999
                    else:
                        hold_seconds = 999
                    
                    # Check real-time exit for ANY position (HARD LOSS CUT works at any time)
                    # Quick exit zone and fisherman have their own time checks inside the function
                    peak_profit_pct = getattr(pos, 'peak_profit_pct', 0)
                    dynamic_tp_pct = getattr(pos, 'dynamic_tp_pct', {})
                    rt_exit = self.ai_filter.check_realtime_quick_exit(
                        symbol=symbol,
                        side=pos.side.upper(),
                        pnl_pct=pnl_pct,
                        hold_seconds=hold_seconds,
                        peak_profit_pct=peak_profit_pct,
                        dynamic_tp_pct=dynamic_tp_pct,
                        atomic_candle=None
                    )
                    
                    if rt_exit and rt_exit.get('action') == 'close':
                            # === LOSS PROTECTION OVERRIDE ===
                            # HARD LOSS CAP (realtime=True, confidence>=0.95) ALWAYS executes
                            # regardless of ai_mode. Money protection > mode restrictions.
                            # EXCEPTION: Diamond Hands (HOLD) overrides even loss protection!
                            is_loss_protection = rt_exit.get('realtime', False) or rt_exit.get('confidence', 0) >= 0.95
                            
                            # === CHECK HOLD STATE FIRST (Diamond Hands overrides everything) ===
                            sym_held_key = symbol.replace('/USDT:USDT', 'USDT').replace('/', '')
                            if self.held_positions.get(sym_held_key, False):
                                pnl_info = f"PnL: {pnl_pct:+.2f}%"
                                reason_info = rt_exit.get('reasoning', 'RT exit')[:80]
                                logger.warning(f"💎 DIAMOND HANDS: RT exit blocked for {symbol} ({reason_info}) {pnl_info}")
                            # === CHECK AI MODE - Only skip non-critical exits in advisory ===
                            elif self.ai_mode not in ['autonomous', 'hybrid'] and not is_loss_protection:
                                logger.debug(f"📋 RT exit skipped - {self.ai_mode} mode (user controls exits)")
                            else:
                                reason = rt_exit.get('reasoning', 'RT Quick Exit')
                                
                                # === SYSTEM #1: EXIT REASON LEARNER CHECK ===
                                # If this exit type historically loses money, suppress it
                                # NEVER suppress loss protection exits (HARD/SOFT LOSS CAP)
                                # CRITICAL FIX: NEVER suppress when position is in LOSS
                                _is_in_loss_rt = pnl_pct < -0.02
                                if not is_loss_protection and not _is_in_loss_rt and hasattr(self, 'ai_filter') and hasattr(self.ai_filter, 'exit_learner'):
                                    try:
                                        _exit_cat = self.ai_filter.exit_learner._classify_exit(reason)
                                        _exit_mod = self.ai_filter.exit_learner.get_exit_modifier(_exit_cat)
                                        if _exit_mod.get('suppress', False):
                                            logger.warning(f"🧠 EXIT LEARNER SUPPRESS: {_exit_cat} blocked for {symbol} | {_exit_mod.get('reason', '')}")
                                            # Don't close — learner says this exit type is bad
                                            return
                                        elif _exit_mod.get('delay_seconds', 0) > 0:
                                            logger.info(f"🧠 EXIT LEARNER DELAY: {_exit_cat} suggests {_exit_mod['delay_seconds']}s delay (logged only)")
                                    except Exception as _el_err:
                                        logger.debug(f"Exit learner check failed (non-fatal): {_el_err}")
                                elif _is_in_loss_rt and not is_loss_protection:
                                    logger.info(f"🧠 EXIT LEARNER BYPASSED: {symbol} in LOSS ({pnl_pct:+.2f}%) — never suppress loss exits (RT)")
                                
                                mode_override = ' [LOSS PROTECTION OVERRIDE]' if is_loss_protection and self.ai_mode not in ['autonomous', 'hybrid'] else ''
                                logger.warning(f"⚡⚡ RT QUICK EXIT{mode_override}: {symbol} {pos.side.upper()} at {pnl_pct:.2f}% | {reason}")
                                await self._close_position(f"RT: {reason}", price)
                                return  # Position closed - exit handler
        
        # Check multi-positions (secondary positions)
        for sym, pos in list(self.positions.items()):
            if not pos:
                continue
            
            pos_sym_key = sym.replace('/USDT:USDT', 'USDT').replace('/', '')
            if pos_sym_key != symbol_key:
                continue
            
            # Skip if already handled by primary position
            if self.position and sym == self.position.symbol:
                continue
            
            # Validate price
            if not validate_price(price, pos.entry_price, sym):
                continue
            
            # ===== EXTRA: Check if price beyond SL is reasonable =====
            sl_distance_pct = abs(pos.stop_loss - pos.entry_price) / pos.entry_price * 100
            price_triggers_sl = (pos.side.lower() == "long" and price <= pos.stop_loss) or \
                               (pos.side.lower() == "short" and price >= pos.stop_loss)
            if price_triggers_sl:
                if pos.side.lower() == "long":
                    price_beyond_sl_pct = abs(pos.stop_loss - price) / pos.entry_price * 100
                else:
                    price_beyond_sl_pct = abs(price - pos.stop_loss) / pos.entry_price * 100
                
                if price_beyond_sl_pct > sl_distance_pct * 3:
                    logger.error(f"🚨 WS GARBAGE DATA (Multi) {sym}: Price ${price:.4f} is {price_beyond_sl_pct:.1f}% beyond SL")
                    continue  # Don't trigger SL with garbage data
            
            # Check Stop Loss
            if (pos.side.lower() == "long" and price <= pos.stop_loss) or \
               (pos.side.lower() == "short" and price >= pos.stop_loss):
                ws_held_key2 = sym.replace('/USDT:USDT', 'USDT').replace('/', '')
                if self.held_positions.get(ws_held_key2, False):
                    ws_pnl2 = ((price - pos.entry_price) / pos.entry_price * 100) if pos.side.lower() == 'long' else ((pos.entry_price - price) / pos.entry_price * 100)
                    if not hasattr(self, '_dh_last_log') or (datetime.now(timezone.utc) - self._dh_last_log.get(ws_held_key2, datetime.min.replace(tzinfo=timezone.utc))).total_seconds() > 30:
                        logger.warning(f"💎 DIAMOND HANDS: {sym} (Multi) SL breached @ ${price:.4f} | PnL: {ws_pnl2:+.2f}% — HOLDING")
                        if not hasattr(self, '_dh_last_log'):
                            self._dh_last_log = {}
                        self._dh_last_log[ws_held_key2] = datetime.now(timezone.utc)
                else:
                    logger.info(f"🔴 WEBSOCKET SL HIT (Multi): {sym} @ ${price:.4f}")
                    await self._close_position_by_symbol(sym, "Stop Loss (WebSocket)", price)
                    continue
            
            # Check Trailing Stop
            if hasattr(pos, 'trailing_stop') and pos.trailing_stop:
                if (pos.side.lower() == "long" and price <= pos.trailing_stop) or \
                   (pos.side.lower() == "short" and price >= pos.trailing_stop):
                    ws_trail_key2 = sym.replace('/USDT:USDT', 'USDT').replace('/', '')
                    if self.held_positions.get(ws_trail_key2, False):
                        logger.warning(f"💎 DIAMOND HANDS: {sym} (Multi) Trail stop hit but HOLD active")
                    else:
                        logger.info(f"🔴 WEBSOCKET TRAIL HIT (Multi): {sym} @ ${price:.4f}")
                        await self._close_position_by_symbol(sym, "Trailing Stop (WebSocket)", price)
                        continue
            
            # Check Take Profits for multi-positions (similar to main)
            if not getattr(pos, 'tp1_hit', False):
                if (pos.side.lower() == "long" and price >= pos.tp1) or \
                   (pos.side.lower() == "short" and price <= pos.tp1):
                    pos.tp1_hit = True
                    # Save old SL BEFORE moving to breakeven for r_value calculation
                    old_sl = pos.stop_loss
                    pos.stop_loss = pos.entry_price  # Move to breakeven
                    # Activate trailing stop using distance from entry to old SL
                    r_value = abs(pos.entry_price - old_sl) if old_sl != pos.entry_price else abs(pos.tp1 - pos.entry_price)
                    if pos.side.lower() == "long":
                        pos.trailing_stop = price - (r_value * 0.5)
                    else:
                        pos.trailing_stop = price + (r_value * 0.5)
                    logger.info(f"🟢 WEBSOCKET TP1 HIT (Multi): {sym} @ ${price:.4f}, SL→BE, trail={pos.trailing_stop:.4f}")
            
            # === REAL-TIME PEAK TRACKING (Multi-Position) ===
            # Track peak profit % in real-time - ALWAYS update if higher
            entry_price = pos.entry_price
            if pos.side.lower() == "long":
                pnl_pct = ((price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - price) / entry_price) * 100
            
            # Initialize peak to very negative if not set
            current_peak_pct = getattr(pos, 'peak_profit_pct', -999)
            if current_peak_pct == -999:
                pos.peak_profit_pct = pnl_pct  # First tick - set peak to current
                current_peak_pct = pnl_pct
            
            # Update peak if current profit is higher (works for negative too!)
            if pnl_pct > current_peak_pct:
                pos.peak_profit_pct = pnl_pct
                # Estimate USD based on position value
                position_value = getattr(pos, 'size', 0) * entry_price
                pos.peak_profit_usd = pnl_pct * position_value / 100
                # Only log significant new peaks
                if pnl_pct > 0 and pnl_pct > current_peak_pct + 0.05:
                    logger.info(f"📈 {sym}: RT NEW PEAK! {pnl_pct:.2f}% (${pos.peak_profit_usd:.2f})")
                # SYNC peak to ai_filter cache immediately (every tick)
                if hasattr(self, 'ai_filter') and self.ai_filter and pnl_pct > 0:
                    self.ai_filter.update_peak_profit(sym, pnl_pct)
            
            # === ⚡ REAL-TIME QUICK EXIT CHECK (FAST-AS-LIGHT) for Multi-Position ===
            if hasattr(self, 'ai_filter') and self.ai_filter:
                import time
                opened_at = getattr(pos, 'opened_at', None)
                if opened_at:
                    try:
                        if isinstance(opened_at, datetime):
                            hold_seconds = time.time() - opened_at.timestamp()
                        elif isinstance(opened_at, str):
                            opened_dt = datetime.fromisoformat(opened_at.replace('Z', '+00:00'))
                            hold_seconds = time.time() - opened_dt.timestamp()
                        else:
                            hold_seconds = 999
                    except Exception:
                        hold_seconds = 999
                else:
                    hold_seconds = 999
                
                # Check real-time exit for ANY position (HARD LOSS CUT works at any time)
                # Quick exit zone and fisherman have their own time checks inside the function
                peak_profit_pct = getattr(pos, 'peak_profit_pct', 0)
                dynamic_tp_pct = getattr(pos, 'dynamic_tp_pct', {})
                rt_exit = self.ai_filter.check_realtime_quick_exit(
                    symbol=sym,
                    side=pos.side.upper() if hasattr(pos, 'side') else 'LONG',
                    pnl_pct=pnl_pct,
                    hold_seconds=hold_seconds,
                    peak_profit_pct=peak_profit_pct,
                    dynamic_tp_pct=dynamic_tp_pct,
                    atomic_candle=None
                )
                
                if rt_exit and rt_exit.get('action') == 'close':
                    # === LOSS PROTECTION OVERRIDE ===
                    # HARD LOSS CAP (realtime=True, confidence>=0.95) ALWAYS executes
                    # EXCEPTION: Diamond Hands (HOLD) overrides even loss protection!
                    is_loss_protection = rt_exit.get('realtime', False) or rt_exit.get('confidence', 0) >= 0.95
                    
                    # === CHECK HOLD STATE FIRST (Diamond Hands overrides everything) ===
                    sym_held_key2 = sym.replace('/USDT:USDT', 'USDT').replace('/', '')
                    if self.held_positions.get(sym_held_key2, False):
                        reason_info = rt_exit.get('reasoning', 'RT exit')[:80]
                        logger.warning(f"💎 DIAMOND HANDS: RT exit blocked for {sym} (Multi) ({reason_info})")
                    # === CHECK AI MODE - Only skip non-critical exits in advisory ===
                    elif self.ai_mode not in ['autonomous', 'hybrid'] and not is_loss_protection:
                        logger.debug(f"📋 RT exit skipped - {self.ai_mode} mode (user controls exits)")
                    else:
                        reason = rt_exit.get('reasoning', 'RT Quick Exit')
                        
                        # === SYSTEM #1: EXIT REASON LEARNER CHECK (Multi) ===
                        # CRITICAL FIX: NEVER suppress when position is in LOSS
                        _is_in_loss_multi = pnl_pct < -0.02
                        if not is_loss_protection and not _is_in_loss_multi and hasattr(self, 'ai_filter') and hasattr(self.ai_filter, 'exit_learner'):
                            try:
                                _exit_cat = self.ai_filter.exit_learner._classify_exit(reason)
                                _exit_mod = self.ai_filter.exit_learner.get_exit_modifier(_exit_cat)
                                if _exit_mod.get('suppress', False):
                                    logger.warning(f"🧠 EXIT LEARNER SUPPRESS: {_exit_cat} blocked for {sym} (Multi) | {_exit_mod.get('reason', '')}")
                                    continue  # Skip this exit — learner says it's bad
                                elif _exit_mod.get('delay_seconds', 0) > 0:
                                    logger.info(f"🧠 EXIT LEARNER DELAY: {_exit_cat} suggests {_exit_mod['delay_seconds']}s delay (logged only)")
                            except Exception as _el_err:
                                logger.debug(f"Exit learner check failed (non-fatal): {_el_err}")
                        elif _is_in_loss_multi and not is_loss_protection:
                            logger.info(f"🧠 EXIT LEARNER BYPASSED: {sym} in LOSS ({pnl_pct:+.2f}%) — never suppress loss exits (Multi)")
                        
                        mode_override = ' [LOSS PROTECTION OVERRIDE]' if is_loss_protection and self.ai_mode not in ['autonomous', 'hybrid'] else ''
                        logger.warning(f"⚡⚡ RT QUICK EXIT (Multi){mode_override}: {sym} at {pnl_pct:.2f}% | {reason}")
                        await self._close_position_by_symbol(sym, f"RT: {reason}", price)
                        continue  # Position closed - continue to next

    # =====================================================================
    # WEBSOCKET REAL-TIME POSITION CALLBACK
    # =====================================================================
    # PhD-optimal: Process position updates instantly from Bybit WebSocket
    # This eliminates the 5-second polling delay for position detection
    # =====================================================================
    
    async def _on_realtime_position_update(self, symbol: str, position_data: dict, change_type: str):
        """Handle real-time position update from WebSocket.
        
        This is called INSTANTLY when Bybit detects a position change.
        Much faster than 5-second polling - critical for detecting:
        - External closes (manual or liquidation)
        - Fill confirmations
        - Server-side SL/TP triggers
        
        Args:
            symbol: The symbol that updated (e.g., 'ETH/USDT:USDT')
            position_data: Position info from Bybit
            change_type: 'OPENED', 'CLOSED', or 'MODIFIED'
        """
        try:
            # Normalize symbol for comparison
            symbol_key = normalize_symbol(symbol)
            contracts = float(position_data.get('contracts', 0))
            side = position_data.get('side', '')
            entry_price = float(position_data.get('entryPrice', 0) or 0)
            mark_price = float(position_data.get('markPrice', 0) or 0)
            unrealized_pnl = float(position_data.get('unrealizedPnl', 0) or 0)
            
            logger.info(f"🔌 RT POSITION: {symbol_key} | {change_type} | {side} | {contracts} contracts | Entry=${entry_price:.4f} | Mark=${mark_price:.4f} | uPnL=${unrealized_pnl:.4f}")
            
            # === Update price cache for dashboard (with validation) ===
            if mark_price > 0:
                if not hasattr(self, '_symbol_prices'):
                    self._symbol_prices = {}
                # Validate against entry price before updating dashboard cache
                if entry_price and entry_price > 0:
                    dev_pct = abs(mark_price - entry_price) / entry_price * 100
                    if dev_pct <= 500:
                        self._symbol_prices[symbol_key] = mark_price
                    else:
                        logger.warning(f"🚨 RT POSITION garbage markPrice rejected: ${mark_price:.6f} is {dev_pct:.1f}% from entry ${entry_price:.6f}")
                else:
                    self._symbol_prices[symbol_key] = mark_price
            
            # === Handle CLOSED positions (external close detection) ===
            if change_type == "CLOSED" or contracts == 0:
                # Check if we have this position tracked
                tracked = False
                
                # Check primary position
                if self.position:
                    pos_key = normalize_symbol(getattr(self.position, 'symbol', ''))
                    if pos_key == symbol_key:
                        logger.warning(f"⚠️ RT POSITION CLOSED: {symbol_key} - detected via WebSocket (external close)")
                        await self._record_externally_closed_trade(self.position)
                        self.position = None
                        tracked = True
                
                # Check multi-positions
                for sym, pos in list(self.positions.items()):
                    if pos is not None:
                        pos_key = normalize_symbol(sym)
                        if pos_key == symbol_key:
                            logger.warning(f"⚠️ RT POSITION CLOSED (Multi): {sym} - detected via WebSocket")
                            await self._record_externally_closed_trade(pos)
                            self.positions[sym] = None
                            self._release_position_slot(sym)
                            tracked = True
                            break
                
                if tracked:
                    self._save_trading_state()
                    logger.info(f"✅ Position cleanup complete for {symbol_key}")
            
            # === Handle OPENED/MODIFIED - update internal state ===
            elif change_type in ["OPENED", "MODIFIED"] and contracts > 0:
                # Update cached position data for matching positions
                if self.position:
                    pos_key = normalize_symbol(getattr(self.position, 'symbol', ''))
                    if pos_key == symbol_key:
                        # Update mark price for PnL calculation
                        if hasattr(self.position, 'current_price'):
                            self.position.current_price = mark_price
                
                for sym, pos in self.positions.items():
                    if pos is not None:
                        pos_key = normalize_symbol(sym)
                        if pos_key == symbol_key:
                            if hasattr(pos, 'current_price'):
                                pos.current_price = mark_price
            
        except Exception as e:
            logger.error(f"🚨 RT position update error: {e}")
    
    async def _manage_position(self):
        """Manage open position - check TP/SL levels."""
        if not self.position:
            return
        
        pos = self.position
        
        # CRITICAL: Get the correct price for the position's symbol
        # Keep the ccxt futures format (LINK/USDT:USDT) for API calls
        pos_symbol = getattr(pos, 'symbol', self.SYMBOL)
        main_symbol = self.SYMBOL
        
        if pos_symbol == main_symbol:
            # Primary symbol - use cached price
            price = self._last_price
        else:
            # Different symbol - fetch the correct price!
            try:
                ticker = await self.exchange.fetch_ticker(pos_symbol)
                price = ticker.get('last', 0)
                if not price or price <= 0:
                    logger.warning(f"Could not get price for {pos_symbol}")
                    return
            except Exception as e:
                logger.error(f"Error fetching price for {pos_symbol}: {e}")
                return
        
        if not price:
            return
        
        # ===== VALIDATE PRICE BEFORE USING IT =====
        if not validate_price(price, pos.entry_price, pos_symbol):
            logger.error(f"🚨 Invalid price {price} for {pos_symbol} - skipping position management cycle")
            return
        
        # Check stop loss
        if (pos.side == "long" and price <= pos.stop_loss) or \
           (pos.side == "short" and price >= pos.stop_loss):
            pos_key_sl = getattr(pos, 'symbol', '').replace('/USDT:USDT', 'USDT').replace('/', '')
            if self.held_positions.get(pos_key_sl, False):
                # 💎 DIAMOND HANDS: SL hit but HOLD is active — ride it out
                pnl_pct_sl = ((price - pos.entry_price) / pos.entry_price * 100) if pos.side == 'long' else ((pos.entry_price - price) / pos.entry_price * 100)
                logger.warning(f"💎 DIAMOND HANDS: {pos_key_sl} SL hit @ ${price:.4f} but HOLD active — PnL: {pnl_pct_sl:+.2f}% — riding it out")
            else:
                await self._close_position("Stop Loss", price)
                return
        
        # Check max position duration (prevent capital lockup)
        # Skip for HELD positions or in advisory/filter mode (user controls position)
        if pos.opened_at:
            hours_held = (datetime.now(timezone.utc) - pos.opened_at).total_seconds() / 3600
            pos_key = getattr(pos, 'symbol', '').replace('/USDT:USDT', 'USDT').replace('/', '')
            is_held = self.held_positions.get(pos_key, False)
            is_user_controlled = self.ai_mode in ['advisory', 'filter']
            
            if hours_held >= self.MAX_POSITION_HOURS:
                if is_held:
                    logger.info(f"⏰ Position held {hours_held:.1f}h but HOLD is active - skipping force close")
                elif is_user_controlled:
                    logger.info(f"⏰ Position held {hours_held:.1f}h but in {self.ai_mode} mode - user controls close")
                else:
                    logger.warning(f"⏰ Position held for {hours_held:.1f}h - forcing close (max {self.MAX_POSITION_HOURS}h)")
                    await self._close_position(f"Max Duration ({hours_held:.0f}h)", price)
                    return
        
        # Check trailing stop
        if pos.trailing_stop:
            if (pos.side == "long" and price <= pos.trailing_stop) or \
               (pos.side == "short" and price >= pos.trailing_stop):
                trail_key = getattr(pos, 'symbol', '').replace('/USDT:USDT', 'USDT').replace('/', '')
                if self.held_positions.get(trail_key, False):
                    logger.warning(f"💎 DIAMOND HANDS: {trail_key} trailing stop hit but HOLD active — riding it out")
                else:
                    await self._close_position("Trailing Stop", price)
                    return
        
        # Check take profits
        if not pos.tp1_hit:
            if (pos.side == "long" and price >= pos.tp1) or \
               (pos.side == "short" and price <= pos.tp1):
                await self._hit_tp(1, price)
        
        if not pos.tp2_hit and pos.tp1_hit:
            if (pos.side == "long" and price >= pos.tp2) or \
               (pos.side == "short" and price <= pos.tp2):
                await self._hit_tp(2, price)
        
        if not pos.tp3_hit and pos.tp2_hit:
            if (pos.side == "long" and price >= pos.tp3) or \
               (pos.side == "short" and price <= pos.tp3):
                await self._hit_tp(3, price)
                await self._close_position("TP3 Hit", price)
    
    async def _manage_all_multi_positions(self):
        """Manage ALL positions in self.positions dict with correct prices.
        
        This handles secondary positions that are on different symbols than the main one.
        It fetches the correct price for each symbol and checks SL/TP.
        """
        try:
            # Get all open positions (excluding the primary if it's already being managed)
            primary_symbol = self.position.symbol.replace('/', '') if self.position else None
            
            for symbol, pos in list(self.positions.items()):
                if not pos:
                    continue
                
                # Skip if this is the primary position (already managed by _manage_position)
                pos_symbol_key = symbol.replace('/', '')
                if primary_symbol and pos_symbol_key == primary_symbol:
                    continue
                
                try:
                    # Get the correct price for this symbol
                    if symbol != self.SYMBOL and symbol.replace('/', '') != self.SYMBOL.replace('/', ''):
                        # Fetch current price for the secondary symbol
                        ticker = await self.exchange.fetch_ticker(symbol)
                        price = ticker.get('last', 0)
                        if not price or price <= 0:
                            continue
                    else:
                        price = self._last_price
                        if not price:
                            continue
                    
                    # ===== VALIDATE PRICE BEFORE USING IT =====
                    if not validate_price(price, pos.entry_price, symbol):
                        logger.error(f"🚨 Invalid price {price} for {symbol} - skipping this position")
                        continue
                    
                    # ===== EXTRA VALIDATION: Price must be reasonable vs SL =====
                    # If price triggers SL, it should be CLOSE to the SL, not wildly beyond it
                    sl_distance_pct = abs(pos.stop_loss - pos.entry_price) / pos.entry_price * 100
                    price_beyond_sl_pct = 0
                    if pos.side.lower() == "long" and price <= pos.stop_loss:
                        price_beyond_sl_pct = abs(pos.stop_loss - price) / pos.entry_price * 100
                    elif pos.side.lower() == "short" and price >= pos.stop_loss:
                        price_beyond_sl_pct = abs(price - pos.stop_loss) / pos.entry_price * 100
                    
                    # If price is more than 3x SL distance beyond SL, it's garbage data
                    if price_beyond_sl_pct > sl_distance_pct * 3:
                        logger.error(f"🚨 GARBAGE DATA DETECTED {symbol}: Price ${price:.4f} is {price_beyond_sl_pct:.1f}% beyond SL (SL distance was only {sl_distance_pct:.1f}%)")
                        logger.error(f"🚨 REFUSING to trigger SL with suspicious price - waiting for valid data")
                        continue
                    
                    # Check stop loss
                    if (pos.side.lower() == "long" and price <= pos.stop_loss) or \
                       (pos.side.lower() == "short" and price >= pos.stop_loss):
                        pos_key_sl2 = symbol.replace('/USDT:USDT', 'USDT').replace('/', '')
                        if self.held_positions.get(pos_key_sl2, False):
                            # 💎 DIAMOND HANDS: SL hit but HOLD active — ride it out
                            pnl_pct_sl2 = ((price - pos.entry_price) / pos.entry_price * 100) if pos.side.lower() == 'long' else ((pos.entry_price - price) / pos.entry_price * 100)
                            logger.warning(f"💎 DIAMOND HANDS: {pos_key_sl2} Position 2 SL hit @ ${price:.4f} but HOLD active — PnL: {pnl_pct_sl2:+.2f}% — riding it out")
                        else:
                            logger.info(f"🛑 Position 2 SL hit: {symbol} @ ${price:.4f}")
                            await self._close_position_by_symbol(symbol, "Stop Loss", price)
                            continue
                    
                    # Check max position duration
                    # Skip for HELD positions or in advisory/filter mode
                    if pos.opened_at:
                        hours_held = (datetime.now(timezone.utc) - pos.opened_at).total_seconds() / 3600
                        pos_key = symbol.replace('/USDT:USDT', 'USDT').replace('/', '')
                        is_held = self.held_positions.get(pos_key, False)
                        is_user_controlled = self.ai_mode in ['advisory', 'filter']
                        
                        if hours_held >= self.MAX_POSITION_HOURS:
                            if is_held:
                                logger.info(f"⏰ Position 2 {symbol} held {hours_held:.1f}h but HOLD active - skipping")
                            elif is_user_controlled:
                                logger.info(f"⏰ Position 2 {symbol} held {hours_held:.1f}h but in {self.ai_mode} mode - user controls")
                            else:
                                logger.warning(f"⏰ Position 2 {symbol} held for {hours_held:.1f}h - forcing close")
                                await self._close_position_by_symbol(symbol, f"Max Duration ({hours_held:.0f}h)", price)
                                continue
                    
                    # Check trailing stop
                    if hasattr(pos, 'trailing_stop') and pos.trailing_stop:
                        if (pos.side.lower() == "long" and price <= pos.trailing_stop) or \
                           (pos.side.lower() == "short" and price >= pos.trailing_stop):
                            trail_key2 = symbol.replace('/USDT:USDT', 'USDT').replace('/', '')
                            if self.held_positions.get(trail_key2, False):
                                logger.warning(f"💎 DIAMOND HANDS: {trail_key2} Position 2 trailing stop hit but HOLD active")
                            else:
                                await self._close_position_by_symbol(symbol, "Trailing Stop", price)
                                continue
                    
                    # Check take profits for Position 2 with trailing stop updates
                    r_value = abs(pos.entry_price - pos.stop_loss) if not getattr(pos, 'tp1_hit', False) else abs(pos.tp1 - pos.entry_price)
                    
                    if not getattr(pos, 'tp1_hit', False):
                        if (pos.side.lower() == "long" and price >= pos.tp1) or \
                           (pos.side.lower() == "short" and price <= pos.tp1):
                            pos.tp1_hit = True
                            # Move SL to breakeven
                            pos.stop_loss = pos.entry_price
                            # Activate trailing stop
                            if pos.side.lower() == "long":
                                pos.trailing_stop = price - (r_value * 0.5)
                            else:
                                pos.trailing_stop = price + (r_value * 0.5)
                            logger.info(f"🎯 Position 2 {symbol} TP1 hit @ ${price:.4f}, SL→BE, trail activated @ ${pos.trailing_stop:.4f}")
                    
                    if not getattr(pos, 'tp2_hit', False) and getattr(pos, 'tp1_hit', False):
                        if (pos.side.lower() == "long" and price >= pos.tp2) or \
                           (pos.side.lower() == "short" and price <= pos.tp2):
                            pos.tp2_hit = True
                            # Move SL to TP1
                            pos.stop_loss = pos.tp1
                            logger.info(f"🎯 Position 2 {symbol} TP2 hit @ ${price:.4f}, SL→TP1")
                    
                    # Update trailing stop if in profit (only if TP1 hit)
                    if getattr(pos, 'tp1_hit', False) and hasattr(pos, 'trailing_stop') and pos.trailing_stop:
                        if pos.side.lower() == "long":
                            new_trail = price - (r_value * 0.5)
                            if new_trail > pos.trailing_stop:
                                pos.trailing_stop = new_trail
                        else:
                            new_trail = price + (r_value * 0.5)
                            if new_trail < pos.trailing_stop:
                                pos.trailing_stop = new_trail
                    
                    if not getattr(pos, 'tp3_hit', False) and getattr(pos, 'tp2_hit', False):
                        if (pos.side.lower() == "long" and price >= pos.tp3) or \
                           (pos.side.lower() == "short" and price <= pos.tp3):
                            logger.info(f"🎯 Position 2 {symbol} TP3 hit @ ${price:.4f} - closing")
                            await self._close_position_by_symbol(symbol, "TP3 Hit", price)
                            continue
                
                except Exception as pos_err:
                    logger.error(f"Error managing position {symbol}: {pos_err}")
        
        except Exception as e:
            logger.error(f"Error in _manage_all_multi_positions: {e}")

    async def _hit_tp(self, level: int, price: float):
        """Handle take profit hit."""
        pos = self.position
        
        # ===== VALIDATE PRICE BEFORE USING IT =====
        if not validate_price(price, pos.entry_price, pos.symbol):
            logger.error(f"🚨 Invalid price {price} for TP{level} - skipping")
            return
        
        if level == 1:
            pos.tp1_hit = True
            pct = self.TP1_PCT
            
            # Save original SL before modification (used for revert if partial close fails)
            pos._original_stop_loss = pos.stop_loss
            
            # Move stop loss to breakeven (entry price) - protect profits!
            old_sl = pos.stop_loss
            pos.stop_loss = pos.entry_price
            logger.info(f"🛡️ BREAKEVEN activated: SL moved from {old_sl:.4f} to {pos.stop_loss:.4f} (entry)")
            
            # === UPDATE SERVER-SIDE SL TO BREAKEVEN ===
            try:
                await self._update_server_side_sl(pos.symbol, pos.entry_price, pos.remaining_size, pos.side)
                logger.info(f"🛡️ Server-side SL updated to breakeven @ ${pos.entry_price:.4f}")
            except Exception as e:
                logger.warning(f"⚠️ Could not update server-side SL: {e}")
            
            # Activate trailing stop
            r_value = abs(pos.entry_price - old_sl) / self.ATR_MULT
            if pos.side == "long":
                pos.trailing_stop = price - (r_value * self.TRAIL_OFFSET_R)
            else:
                pos.trailing_stop = price + (r_value * self.TRAIL_OFFSET_R)
            logger.info(f"Trailing stop activated at {pos.trailing_stop:.4f}")
            
            # === RECALCULATE TP2/TP3 WITH FRESH ATR (Improvement #9) ===
            # After TP1 hit, volatility may have changed — adapt remaining targets
            # Only tighten (bring closer), never widen (don't chase)
            try:
                if hasattr(self, 'bars_agg') and len(self.bars_agg) >= 20:
                    fresh_atr = self.bars_agg['atr'].iloc[-1] if 'atr' in self.bars_agg.columns else None
                    if fresh_atr and fresh_atr > 0:
                        fresh_tp = self.calculate_dynamic_tp(
                            entry_price=pos.entry_price, side=pos.side,
                            df=pd.DataFrame(self.bars_agg), atr=fresh_atr, stop_loss=pos.stop_loss
                        )
                        old_tp2, old_tp3 = pos.tp2, pos.tp3
                        # Only tighten: bring TPs closer to entry, never push them further
                        if pos.side == 'long':
                            if fresh_tp['tp2'] < pos.tp2:
                                pos.tp2 = fresh_tp['tp2']
                            if fresh_tp['tp3'] < pos.tp3:
                                pos.tp3 = fresh_tp['tp3']
                        else:  # short
                            if fresh_tp['tp2'] > pos.tp2:
                                pos.tp2 = fresh_tp['tp2']
                            if fresh_tp['tp3'] > pos.tp3:
                                pos.tp3 = fresh_tp['tp3']
                        if pos.tp2 != old_tp2 or pos.tp3 != old_tp3:
                            logger.info(f"🎯 TP RECALC after TP1: TP2 ${old_tp2:.4f}→${pos.tp2:.4f} | TP3 ${old_tp3:.4f}→${pos.tp3:.4f} (fresh ATR={fresh_atr:.6f})")
                        else:
                            logger.info(f"🎯 TP RECALC: No change — original TPs are already tighter than fresh calculation")
            except Exception as tp_recalc_err:
                logger.warning(f"⚠️ TP recalculation failed: {tp_recalc_err} — keeping original TPs")
        elif level == 2:
            pos.tp2_hit = True
            pct = self.TP2_PCT
            
            # After TP2, move SL to TP1 level - lock in TP1 profit on remaining 20%
            old_sl = pos.stop_loss
            pos.stop_loss = pos.tp1
            logger.info(f"🛡️ SL tightened after TP2: moved from {old_sl:.4f} to {pos.stop_loss:.4f} (TP1 level)")
            
            # === UPDATE SERVER-SIDE SL TO TP1 LEVEL ===
            try:
                await self._update_server_side_sl(pos.symbol, pos.tp1, pos.remaining_size, pos.side)
                logger.info(f"🛡️ Server-side SL updated to TP1 @ ${pos.tp1:.4f}")
            except Exception as e:
                logger.warning(f"⚠️ Could not update server-side SL: {e}")
            
            # Tighten trailing stop as well
            r_value = abs(pos.entry_price - old_sl) / self.ATR_MULT if hasattr(pos, '_original_sl') else abs(pos.tp1 - pos.entry_price)
            if pos.side == "long":
                new_trail = price - (r_value * self.TRAIL_OFFSET_R * 0.5)  # Tighter trail after TP2
                pos.trailing_stop = max(pos.trailing_stop or 0, new_trail)
            else:
                new_trail = price + (r_value * self.TRAIL_OFFSET_R * 0.5)
                pos.trailing_stop = min(pos.trailing_stop or float('inf'), new_trail)
            logger.info(f"Trailing stop tightened to {pos.trailing_stop:.4f}")
        else:
            pos.tp3_hit = True
            pct = self.TP3_PCT
        
        # Calculate P&L for this portion
        portion_size = pos.size * pct
        if pos.side == "long":
            pnl = (price - pos.entry_price) * portion_size
        else:
            pnl = (pos.entry_price - price) * portion_size
        
        # === EXECUTE PARTIAL CLOSE ON EXCHANGE ===
        # Actually reduce the position on the exchange, not just track internally
        partial_close_succeeded = True  # Assume success for paper mode
        if self.live_mode and level <= 2:  # Partial close for TP1 and TP2 only
            partial_close_succeeded = False
            try:
                close_side = 'sell' if pos.side.lower() == 'long' else 'buy'
                partial_order = await self._execute_market_order(
                    pos.symbol, close_side, portion_size, reduce_only=True
                )
                if partial_order:
                    actual_fill = float(partial_order.get('average', price))
                    # Use actual filled quantity (may differ from requested due to precision)
                    actual_filled_qty = float(partial_order.get('filled', portion_size))
                    # Recalculate PnL with actual fill
                    if pos.side == "long":
                        pnl = (actual_fill - pos.entry_price) * actual_filled_qty
                    else:
                        pnl = (pos.entry_price - actual_fill) * actual_filled_qty
                    portion_size = actual_filled_qty  # Use actual fill for fee calc
                    partial_close_succeeded = True
                    logger.info(f"✅ TP{level} partial close executed: {actual_filled_qty:.6f} @ ${actual_fill:.4f}")
                else:
                    logger.error(f"🚨 TP{level} partial close FAILED — reverting TP{level} flag to prevent desync")
            except Exception as e:
                logger.error(f"🚨 TP{level} partial close error: {e} — reverting TP{level} flag to prevent desync")
        
        # === CRITICAL: If live partial close failed, revert TP flag ===
        # This prevents internal tracking from going out of sync with exchange position
        if not partial_close_succeeded:
            if level == 1:
                pos.tp1_hit = False
                pos.stop_loss = getattr(pos, '_original_stop_loss', pos.stop_loss)
                pos.trailing_stop = None
            elif level == 2:
                pos.tp2_hit = False
            elif level == 3:
                pos.tp3_hit = False
            logger.warning(f"⚠️ TP{level} flags reverted — will retry on next tick")
            return  # Don't update balance or notify
        
        # Deduct fees from PnL for accurate tracking
        fee_cost = portion_size * price * self.FEE_TAKER  # Exit fee
        pnl -= fee_cost
        
        # Add partial P&L to balance (but NOT to stats - stats recalculated from trade_history)
        self.balance += pnl
        
        remaining = 1.0 - (self.TP1_PCT if pos.tp1_hit else 0) - \
                         (self.TP2_PCT if pos.tp2_hit else 0) - \
                         (self.TP3_PCT if pos.tp3_hit else 0)
        
        logger.info(f"TP{level} HIT | Price: {price:.4f} | P&L: ${pnl:+.2f} | Remaining: {remaining:.0%}")
        
        # Save position state after TP hit (tp1_hit, tp2_hit, trailing_stop changed)
        self._save_trading_state()
        
        if self.telegram.enabled:
            await self.telegram.notify_tp_hit(
                symbol=self.SYMBOL,
                tp_level=level,
                price=price,
                pnl=pnl,
                remaining_pct=remaining
            )
    
    async def _on_position_update_recovery(self, symbol: str, position_data: dict, change_type: str):
        """
        Real-time recovery monitoring via WebSocket - INSTANT PnL tracking.
        Called on EVERY position update (like peak tracking but for losses).
        
        Fast AI triggers:
        - Loss deepens by 0.1% → Check recovery signals
        - Loss hits -0.3% → AI validation
        - Loss hits -0.5% → URGENT AI decision
        """
        try:
            # Only monitor existing positions
            if change_type == "CLOSED":
                return
            
            # Find position object
            pos = None
            if self.position and self.position.symbol.replace('/', '') == symbol.replace('/', ''):
                pos = self.position
            elif symbol in self.positions:
                pos = self.positions[symbol]
            elif symbol.replace('/', '') in self.positions:
                pos = self.positions[symbol.replace('/', '')]
            
            if not pos:
                return
            
            # Calculate real-time PnL
            current_price = float(position_data.get('markPrice', 0))
            if current_price == 0:
                return
            
            entry_price = pos.entry_price
            side = pos.side
            
            # === VALIDATE: Reject garbage mark price before calculating PnL ===
            if entry_price and entry_price > 0:
                deviation_pct = abs(current_price - entry_price) / entry_price * 100
                if deviation_pct > 500:
                    logger.debug(f"🚨 Position stream garbage markPrice rejected: ${current_price:.6f} is {deviation_pct:.1f}% from entry ${entry_price:.6f}")
                    return
            
            # Calculate PnL %
            if side == 'long':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            pnl_usd = (current_price - entry_price) * pos.remaining_size if side == 'long' else \
                      (entry_price - current_price) * pos.remaining_size
            
            # === TRACK LOSS DEPTH (like peak tracking but negative) ===
            if pnl_pct < 0:  # Only track losses
                # Update deepest loss if this is worse
                if pnl_pct < pos.deepest_loss_pct:
                    old_deepest = pos.deepest_loss_pct
                    pos.deepest_loss_pct = pnl_pct
                    pos.deepest_loss_usd = pnl_usd
                    
                    loss_deepened_by = abs(pnl_pct - old_deepest)
                    
                    logger.info(f"📉 {symbol}: Loss DEEPENED {loss_deepened_by:.2f}% | Now: {pnl_pct:.2f}% (${pnl_usd:.2f})")
                    
                    # === FAST AI TRIGGERS ===
                    # Prevent spam: Only check recovery every 10 seconds
                    now = datetime.now(timezone.utc)
                    time_since_last_check = (now - pos.last_recovery_check).total_seconds()
                    
                    if time_since_last_check < 10:
                        return  # Too soon, skip
                    
                    pos.last_recovery_check = now
                    
                    # Trigger levels (like circuit breakers)
                    trigger_ai = False
                    urgency = "normal"
                    
                    if loss_deepened_by >= 0.1 and pnl_pct <= -0.2:
                        # Loss deepening fast + in danger zone
                        trigger_ai = True
                        urgency = "high"
                        logger.warning(f"⚠️ {symbol}: LOSS DEEPENING FAST (-{loss_deepened_by:.2f}% drop)")
                    elif pnl_pct <= -0.5:
                        # Deep loss - URGENT
                        trigger_ai = True
                        urgency = "urgent"
                        logger.error(f"🚨 {symbol}: DEEP LOSS {pnl_pct:.2f}% - URGENT AI CHECK")
                    elif pnl_pct <= -0.3 and not pos.in_recovery_mode:
                        # Entering recovery zone
                        trigger_ai = True
                        urgency = "normal"
                        logger.info(f"🔄 {symbol}: Entering recovery zone ({pnl_pct:.2f}%)")
                    
                    # === EXECUTE FAST AI CHECK ===
                    if trigger_ai and self.ai_mode == 'autonomous':
                        pos.recovery_attempts += 1
                        logger.info(f"🤖 {symbol}: FAST AI RECOVERY CHECK #{pos.recovery_attempts} | Urgency: {urgency.upper()}")
                        
                        # Get fresh market data
                        try:
                            symbol_ccxt = to_futures_symbol(symbol)
                            # Fetch OHLCV directly (same as position monitor)
                            ohlcv = await self.exchange.fetch_ohlcv(symbol_ccxt, '15m', limit=100)
                            if not ohlcv or len(ohlcv) < 50:
                                logger.warning(f"⚠️ {symbol}: Cannot fetch data for recovery check")
                                return
                            
                            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                            
                            # Calculate ATR for context
                            high_low = df['high'] - df['low']
                            atr = high_low.rolling(14).mean().iloc[-1]
                            
                            # Build position dict for unified decision
                            position_info = {
                                'symbol': symbol,
                                'side': side,
                                'entry_price': entry_price,
                                'size': pos.remaining_size,
                                'stop_loss': getattr(pos, 'stop_loss', 0),
                                'tp1_hit': getattr(pos, 'tp1_hit', False),
                                'tp2_hit': getattr(pos, 'tp2_hit', False),
                                'peak_profit_pct': pos.peak_profit_pct,
                                'peak_profit_usd': pos.peak_profit_usd,
                                'deepest_loss_pct': pos.deepest_loss_pct,
                                'deepest_loss_usd': pos.deepest_loss_usd,
                                'recovery_attempts': pos.recovery_attempts,
                                'in_recovery_mode': pos.in_recovery_mode,
                                'open_time': getattr(pos, 'opened_at', None),
                                'entry_time': getattr(pos, 'opened_at', None)
                            }
                            
                            # Call unified position decision (same as regular monitor)
                            unified_decision = self.ai_filter.unified_position_decision(
                                position=position_info,
                                df=df,
                                current_price=current_price,
                                atr=atr,
                                atomic_candle=None
                            )
                            
                            action = unified_decision.get('action', 'hold')
                            reasoning = unified_decision.get('reasoning', '')
                            confidence = unified_decision.get('confidence', 0.5)
                            
                            logger.info(f"🤖 {symbol}: FAST AI DECISION: {action.upper()} ({confidence:.0%})")
                            logger.info(f"   Reasoning: {reasoning[:120]}")
                            
                            # Handle decision
                            if action == 'hold_recovery':
                                pos.in_recovery_mode = True
                                recovery_score = unified_decision.get('recovery_score', 0)
                                logger.info(f"🔄 {symbol}: HOLDING FOR RECOVERY | Score: {recovery_score}/100")
                            elif action == 'close':
                                # AI says exit NOW
                                logger.error(f"🚨 {symbol}: AI FAST EXIT | {reasoning}")
                                if self.position and symbol == self.position.symbol.replace('/', ''):
                                    await self._close_position(f"Fast AI Exit: {reasoning}", current_price)
                                else:
                                    await self._close_position_by_symbol(symbol, f"Fast AI Exit: {reasoning}", current_price)
                        
                        except Exception as ai_err:
                            logger.error(f"❌ {symbol}: Fast AI check failed: {ai_err}")
            
            else:
                # Position is profitable - check if recovering from loss
                if pos.in_recovery_mode and pos.deepest_loss_pct < 0:
                    # RECOVERY SUCCESS!
                    recovery_gain = pnl_pct - pos.deepest_loss_pct
                    logger.info(f"✅ {symbol}: RECOVERY SUCCESS! | From {pos.deepest_loss_pct:.2f}% → {pnl_pct:.2f}% (+{recovery_gain:.2f}%)")
                    pos.in_recovery_mode = False
                    # Reset loss tracking
                    pos.deepest_loss_pct = 0.0
                    pos.deepest_loss_usd = 0.0
        
        except Exception as e:
            logger.error(f"❌ Recovery monitoring error for {symbol}: {e}")
    
    async def _close_position_by_symbol(self, symbol: str, reason: str, price: float):
        """Close a specific position from the multi-position dict by symbol.
        
        This is used when closing a secondary position in multi-pair mode.
        """
        # Serialize close operations to prevent race conditions
        async with self._close_lock:
            norm_sym = symbol.replace('/', '').replace(':USDT', '').upper()
            if norm_sym in self._closing_symbols:
                logger.warning(f"⚠️ {symbol} close already in progress - skipping duplicate")
                return False
            self._closing_symbols.add(norm_sym)
        try:
            return await self._close_position_by_symbol_inner(symbol, reason, price)
        finally:
            self._closing_symbols.discard(norm_sym)
    
    async def _close_position_by_symbol_inner(self, symbol: str, reason: str, price: float):
        """Inner implementation of close_position_by_symbol (protected by race guard)."""
        try:
            # Find the position
            pos = self.positions.get(symbol) or self.positions.get(symbol.replace('/', ''))
            
            if not pos:
                logger.warning(f"Position {symbol} not found in positions dict")
                return False
            
            # ===== CRITICAL: VALIDATE PRICE TO PREVENT GARBAGE DATA =====
            if not validate_price(price, pos.entry_price, symbol):
                logger.error(f"🚨 REFUSING TO CLOSE {symbol} with invalid price: {price} (entry was {pos.entry_price})")
                logger.error(f"🚨 This would have corrupted balance! Skipping close until valid price.")
                return False
            
            # === CANCEL SERVER-SIDE SL/TP ORDERS ===
            try:
                await self._cancel_server_side_orders(symbol)
            except Exception as e:
                logger.debug(f"Could not cancel server-side orders for {symbol}: {e}")
            
            # === LIVE MODE: Execute real close order ===
            # === PAPER MODE: Simulate close without sending order ===
            if not self.live_mode:
                logger.info(f"📝 PAPER MODE: Simulating close for {symbol} (no real order)")
                # In paper mode, we still update internal state - just don't send order to Bybit
                # This allows testing the bot's logic without real money
                
                # Calculate P&L for paper trade
                if pos.side.lower() == "long":
                    pnl = (price - pos.entry_price) * pos.remaining_size
                    pnl_pct = ((price / pos.entry_price) - 1) * 100
                else:
                    pnl = (pos.entry_price - price) * pos.remaining_size
                    pnl_pct = ((pos.entry_price / price) - 1) * 100
                
                # Calculate hold duration
                hold_hours = 0
                if hasattr(pos, 'opened_at') and pos.opened_at:
                    hold_hours = (datetime.now(timezone.utc) - pos.opened_at).total_seconds() / 3600
                
                # Log the paper close
                emoji = "✅" if pnl >= 0 else "❌"
                logger.info(f"{emoji} PAPER CLOSED {pos.side.upper()} {symbol} | Reason: {reason} | "
                           f"PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%) | Held: {hold_hours:.1f}h | "
                           f"Entry: ${pos.entry_price:.4f} → Exit: ${price:.4f}")
                
                # Update balance (paper)
                self.balance += pnl
                
                # Update stats
                is_win = pnl >= 0
                self.stats.record_trade(pnl, is_win)
                
                # === RE-ENTRY COOLDOWN (applies to ALL closes, win or loss) ===
                # SMART COOLDOWN: RT Quick Exit for reversal signals gets LONGER cooldown
                # because the exit reason itself detected the trade thesis was wrong.
                # Re-entering the same idea after reversal detection = revenge trading.
                _reentry_sym = symbol.replace('/', '').replace(':USDT', '').upper()
                _reentry_key = f"{_reentry_sym}_{pos.side.upper()}"
                _rt_exit_reason = reason.lower() if reason else ''
                _is_reversal_exit = any(kw in _rt_exit_reason for kw in ['rt:', 'quick exit', 'reversal', 'momentum against', 'efficiency drop'])
                _cooldown_secs = 300 if _is_reversal_exit else self.REENTRY_COOLDOWN_SECONDS  # 5min for reversal exits, 3min for normal
                self.reentry_cooldown[_reentry_key] = datetime.now(timezone.utc) + timedelta(seconds=_cooldown_secs)
                logger.info(f"⏰ RE-ENTRY COOLDOWN: {_reentry_key} blocked for {_cooldown_secs}s{' (REVERSAL EXIT → extended)' if _is_reversal_exit else ''}")
                
                # Update streaks
                if is_win:
                    self.consecutive_wins += 1
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1
                    self.consecutive_wins = 0
                    # === PER-PAIR LOSS COOLDOWN (paper path) ===
                    symbol_key = symbol.replace('/', '').replace(':USDT', '').upper()
                    cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=self.PAIR_LOSS_COOLDOWN_MINUTES)
                    self.pair_loss_cooldown[symbol_key] = cooldown_until
                    logger.info(f"⏰ PAIR COOLDOWN: {symbol_key} on cooldown until {cooldown_until.strftime('%H:%M:%S')} UTC (after loss)")
                
                # Record trade in history
                trade_record = {
                    'symbol': symbol,
                    'side': pos.side.upper(),
                    'entry_price': pos.entry_price,
                    'exit_price': price,
                    'size': pos.remaining_size,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'result': 'WIN' if is_win else 'LOSS',
                    'reason': reason,
                    'entry_time': pos.opened_at.strftime('%H:%M:%S') if hasattr(pos, 'opened_at') and pos.opened_at else 'N/A',
                    'exit_time': datetime.now(timezone.utc).strftime('%H:%M:%S'),
                    'date': pos.opened_at.strftime('%Y-%m-%d') if hasattr(pos, 'opened_at') and pos.opened_at else datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                    'opened_at': pos.opened_at.isoformat() if hasattr(pos, 'opened_at') and pos.opened_at else None,
                    'closed_at': datetime.now(timezone.utc).isoformat(),
                    'hold_hours': hold_hours,
                    'mode': 'paper'
                }
                self._record_trade_safe(trade_record)
                
                # === HISTORICAL ML: Record trade for model training (paper close path) ===
                # FIX: This was missing — Historical ML never learned from paper trades!
                try:
                    if pos.entry_df_snapshot is not None:
                        ml_record_trade(pos.entry_df_snapshot, is_win)
                        logger.info(f"🧠 ML sample recorded (paper): {'WIN' if is_win else 'LOSS'}")
                    elif len(self.bars_agg) >= 20:
                        ml_record_trade(self.bars_agg, is_win)
                        logger.info(f"🧠 ML sample recorded (paper fallback): {'WIN' if is_win else 'LOSS'}")
                except Exception as ml_err:
                    logger.warning(f"⚠️ Historical ML recording failed (paper): {ml_err}")
                
                # === DUAL ML: Record trade for Live ML model (paper close path) ===
                try:
                    entry_feats = getattr(pos, 'ml_entry_features', None) or {}
                    self.ml_live.record_trade(
                        entry_features=entry_feats,
                        outcome=is_win,
                        pnl=pnl,
                        symbol=symbol,
                        side=pos.side.upper(),
                        reason=reason
                    )
                except Exception as ml_err:
                    logger.warning(f"⚠️ Dual ML recording failed (paper): {ml_err}")
                
                # === BAYESIAN WEIGHT LEARNING (paper close path) ===
                # FIX: This was missing — Bayesian learner never updated in paper mode!
                _hold_seconds = hold_hours * 3600 if hold_hours else 0
                self.ai_filter.record_trade_result(
                    is_win, pnl, symbol=symbol, side=pos.side.upper(),
                    entry_component_scores=getattr(pos, 'entry_component_scores', None),
                    exit_reason=reason,
                    pnl_pct=pnl_pct,
                    hold_time_seconds=_hold_seconds,
                    entry_time=getattr(pos, 'opened_at', None)
                )
                
                # Clean up position state using standardized method (stamps _recently_closed_symbols)
                self._clear_position(symbol)
                
                # Send notification
                try:
                    if self.telegram:
                        await self.telegram.notify_trade_closed(
                            symbol=symbol,
                            side=pos.side,
                            entry_price=pos.entry_price,
                            exit_price=price,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            reason=f"[PAPER] {reason}",
                            balance=self.balance,
                            total_pnl=self.stats.total_pnl,
                            win_rate=self.stats.win_rate * 100
                        )
                except Exception as e:
                    logger.warning(f"Could not send paper close notification: {e}")
                
                logger.info(f"✅ PAPER CLOSE EXECUTED for {symbol}")
                return True
            
            # ===== VERIFY POSITION EXISTS ON BYBIT BEFORE CLOSING =====
            try:
                import ccxt as ccxt_sync
                sync_ex = ccxt_sync.bybit({
                    'apiKey': os.getenv('BYBIT_API_KEY'),
                    'secret': os.getenv('BYBIT_API_SECRET'),
                    'enableRateLimit': True,
                    'options': {'defaultType': 'swap', 'defaultSubType': 'linear', 'recvWindow': 20000}
                })
                sync_ex.load_markets()
                
                # Check if position actually exists on Bybit
                # Normalize symbol to ccxt futures format (LINK/USDT:USDT)
                if ':USDT' in symbol and '/' in symbol:
                    ccxt_symbol = symbol  # Already in correct format
                elif ':USDT' in symbol:
                    # Format like LINKUSDT:USDT -> LINK/USDT:USDT
                    sym_clean = symbol.replace('USDT:USDT', '').replace(':USDT', '')
                    ccxt_symbol = f"{sym_clean}/USDT:USDT"
                else:
                    # Simple format like LINKUSDT -> LINK/USDT:USDT
                    sym_clean = symbol.replace('/', '').replace('USDT', '')
                    ccxt_symbol = f"{sym_clean}/USDT:USDT"
                
                bybit_positions = sync_ex.fetch_positions([ccxt_symbol])
                actual_pos = None
                for bp in bybit_positions:
                    if float(bp.get('contracts', 0)) != 0:
                        actual_pos = bp
                        break
                
                if not actual_pos:
                    logger.warning(f"⚠️ Position {symbol} not found on Bybit (already closed by SL/TP?)")
                    logger.info(f"📊 Recording trade and cleaning up internal state for {symbol}...")
                    
                    # === PHANTOM TRADE PREVENTION: Don't record if entry_price is invalid ===
                    if not pos.entry_price or pos.entry_price <= 0:
                        logger.error(f"🚨 PHANTOM TRADE BLOCKED: {symbol} has entry_price=${pos.entry_price} - cleaning up without PnL impact")
                        if symbol in self.positions:
                            self.positions[symbol] = None
                            self._release_position_slot(symbol)
                        if self.position and getattr(self.position, 'symbol', '') == symbol:
                            self.position = None
                        self._save_trading_state()
                        return True
                    
                    # === GET ACTUAL EXIT PRICE FROM BYBIT (not stale tracked price) ===
                    # Priority: Bybit closed PnL API > SL price > fallback price
                    exit_price = None
                    pnl = None
                    pnl_source = "unknown"
                    
                    # Try Bybit closed PnL API FIRST (has actual fill price)
                    try:
                        bybit_symbol = normalize_symbol(symbol)
                        closed_pnl_data = sync_ex.privateGetV5PositionClosedPnl({
                            'category': 'linear',
                            'symbol': bybit_symbol,
                            'limit': 10
                        })
                        if closed_pnl_data and closed_pnl_data.get('result', {}).get('list'):
                            pos_open_time = pos.opened_at if hasattr(pos, 'opened_at') and pos.opened_at else datetime.now(timezone.utc) - timedelta(hours=24)
                            for record in closed_pnl_data['result']['list']:
                                close_time_ms = int(record.get('updatedTime', 0))
                                close_time = datetime.fromtimestamp(close_time_ms / 1000, tz=timezone.utc)
                                if close_time > pos_open_time:
                                    avg_exit = float(record.get('avgExitPrice', 0))
                                    record_pnl = float(record.get('closedPnl', 0))
                                    if avg_exit > 0:
                                        exit_price = avg_exit
                                        pnl = record_pnl
                                        pnl_source = "Bybit closed PnL API"
                                        logger.info(f"📊 VERIFIED from Bybit: exit=${exit_price:.6f}, PnL=${pnl:.4f}")
                                        break
                    except Exception as cpnl_err:
                        logger.debug(f"Could not fetch Bybit closed PnL: {cpnl_err}")
                    
                    # Fallback: use SL price (reasonable estimate)
                    if exit_price is None:
                        exit_price = pos.stop_loss if pos.stop_loss else price
                        if pos.side.lower() == "long":
                            pnl = (exit_price - pos.entry_price) * pos.remaining_size
                        else:
                            pnl = (pos.entry_price - exit_price) * pos.remaining_size
                        pnl_source = "SL price fallback" if pos.stop_loss else "passed-in price"
                        logger.warning(f"⚠️ Using {pnl_source} for {symbol}: exit=${exit_price:.6f}")
                    
                    if pos.entry_price > 0:
                        if pos.side.lower() == "long":
                            pnl_pct = ((exit_price / pos.entry_price) - 1) * 100
                        else:
                            pnl_pct = ((pos.entry_price / exit_price) - 1) * 100
                    else:
                        pnl_pct = 0
                    
                    is_win = pnl >= 0
                    
                    # Calculate hold duration
                    hold_hours = 0
                    if hasattr(pos, 'opened_at') and pos.opened_at:
                        hold_hours = (datetime.now(timezone.utc) - pos.opened_at).total_seconds() / 3600
                    
                    # Log the close
                    emoji = "✅" if pnl >= 0 else "❌"
                    logger.info(f"{emoji} CLOSED BY EXCHANGE {pos.side.upper()} {symbol} | "
                               f"PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%) | Held: {hold_hours:.1f}h | "
                               f"Entry: ${pos.entry_price:.4f} → Exit: ${exit_price:.4f} | Source: {pnl_source}")
                    
                    # === CRITICAL FIX: DON'T apply PnL to balance manually ===
                    # _sync_balance_with_exchange() already set self.balance from Bybit.
                    # Doing self.balance += pnl would DOUBLE-COUNT the loss/gain.
                    # Instead, sync balance from exchange to get the TRUE value.
                    try:
                        await self._sync_balance_with_exchange()
                        logger.info(f"💰 Balance synced from exchange after external close: ${self.balance:.2f}")
                    except Exception as sync_err:
                        logger.warning(f"⚠️ Could not sync balance after external close: {sync_err}")
                        # Only apply PnL manually if exchange sync fails
                        self.balance += pnl
                        logger.warning(f"⚠️ Applied PnL manually as fallback: ${pnl:+.2f}")
                    
                    # Update stats
                    self.stats.record_trade(pnl, is_win)
                    
                    # Update streaks
                    if is_win:
                        self.consecutive_wins += 1
                        self.consecutive_losses = 0
                    else:
                        self.consecutive_losses += 1
                        self.consecutive_wins = 0
                    
                    # Record trade in history
                    trade_record = {
                        'symbol': symbol,
                        'side': pos.side.upper(),
                        'entry': pos.entry_price,
                        'exit': exit_price,
                        'size': pos.remaining_size if pos.remaining_size > 0 else pos.size,
                        'pnl': pnl,
                        'result': 'WIN' if is_win else 'LOSS',
                        'exit_reason': 'Server-side SL/TP (Exchange)',
                        'entry_time': pos.opened_at.strftime('%H:%M:%S') if hasattr(pos, 'opened_at') and pos.opened_at else 'N/A',
                        'exit_time': datetime.now(timezone.utc).strftime('%H:%M:%S'),
                        'date': pos.opened_at.strftime('%Y-%m-%d') if hasattr(pos, 'opened_at') and pos.opened_at else datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                        'opened_at': pos.opened_at.isoformat() if hasattr(pos, 'opened_at') and pos.opened_at else None,
                        'closed_at': datetime.now(timezone.utc).isoformat(),
                        'tp1_hit': getattr(pos, 'tp1_hit', False),
                        'tp2_hit': getattr(pos, 'tp2_hit', False)
                    }
                    self._record_trade_safe(trade_record)
                    
                    # Record for AI filter (with Bayesian weight learning)
                    _hold_seconds_ext = hold_hours * 3600 if hold_hours else 0
                    self.ai_filter.record_trade_result(
                        is_win, pnl, symbol=symbol, side=pos.side.upper(),
                        entry_component_scores=getattr(pos, 'entry_component_scores', None),
                        exit_reason='Server-side SL/TP (Exchange)',
                        pnl_pct=pnl_pct,
                        hold_time_seconds=_hold_seconds_ext,
                        entry_time=getattr(pos, 'opened_at', None)
                    )
                    
                    # === DUAL ML: Record trade for Live ML model (external close path) ===
                    try:
                        entry_feats = getattr(pos, 'ml_entry_features', None) or {}
                        self.ml_live.record_trade(
                            entry_features=entry_feats,
                            outcome=is_win,
                            pnl=pnl,
                            symbol=symbol,
                            side=pos.side.upper(),
                            reason='Server-side SL/TP'
                        )
                    except Exception as ml_err:
                        logger.warning(f"⚠️ Dual ML recording failed (external): {ml_err}")
                    
                    # Clean up internal state using standardized method (stamps _recently_closed_symbols)
                    self._clear_position(symbol)
                    
                    # Send notification
                    try:
                        if self.telegram:
                            await self.telegram.notify_trade_closed(
                                symbol=symbol,
                                side=pos.side,
                                entry_price=pos.entry_price,
                                exit_price=exit_price,
                                pnl=pnl,
                                pnl_pct=pnl_pct,
                                reason="Server-side SL/TP (Exchange)",
                                balance=self.balance,
                                total_pnl=self.stats.total_pnl,
                                win_rate=self.stats.win_rate * 100
                            )
                    except Exception as e:
                        logger.warning(f"Could not send exchange close notification: {e}")
                    
                    logger.info(f"✅ Trade recorded and internal state cleaned up for {symbol}")
                    return True  # Return true since position is effectively closed
                
                # Position exists - use actual size from Bybit to close
                actual_size = abs(float(actual_pos['contracts']))
                if abs(actual_size - pos.remaining_size) > 0.0001:
                    logger.warning(f"⚠️ Size mismatch: Bot={pos.remaining_size:.6f}, Bybit={actual_size:.6f} - using Bybit size")
                
            except Exception as verify_err:
                logger.warning(f"⚠️ Could not verify position on Bybit: {verify_err} - proceeding with close attempt")
                actual_size = pos.remaining_size
            
            order_side = 'sell' if pos.side.lower() == 'long' else 'buy'
            order = await self._execute_market_order(symbol, order_side, actual_size, reduce_only=True)
            
            if not order:
                logger.error(f"🚨 LIVE CLOSE ORDER FAILED for {symbol}")
                return False
            
            if order.get('average'):
                price = float(order['average'])
                logger.info(f"📊 Close fill price: ${price:.4f}")
            
            # ===== CRITICAL: Validate close price before calculating PnL =====
            # Use 500% to allow meme coin extreme moves - real protection is $0 entry/exit check
            if not validate_price(price, pos.entry_price, symbol, max_deviation_pct=500.0):
                logger.error(f"🚨 REFUSING TO RECORD TRADE with garbage exit price: {price} for {symbol}")
                logger.error(f"🚨 Entry was {pos.entry_price}, exit {price} is {abs(price-pos.entry_price)/pos.entry_price*100:.1f}% off")
                # Clean up internal state but DON'T record fake PnL
                if symbol in self.positions:
                    self.positions[symbol] = None
                    self._release_position_slot(symbol)
                symbol_key = symbol.replace('/', '')
                if symbol_key in self.positions:
                    self.positions[symbol_key] = None
                    self._release_position_slot(symbol_key)
                self._save_trading_state()
                logger.info(f"✅ Position state cleaned up, but NO fake loss recorded")
                return True
            
            # Calculate P&L
            if pos.side.lower() == "long":
                pnl = (price - pos.entry_price) * pos.remaining_size
                pnl_pct = ((price / pos.entry_price) - 1) * 100
            else:
                pnl = (pos.entry_price - price) * pos.remaining_size
                pnl_pct = ((pos.entry_price / price) - 1) * 100
            
            # Calculate hold duration
            hold_hours = 0
            if hasattr(pos, 'opened_at') and pos.opened_at:
                hold_hours = (datetime.now(timezone.utc) - pos.opened_at).total_seconds() / 3600
            
            # ===== FINAL SANITY CHECK: Log large losses but ALWAYS allow close =====
            # NOTE: We used to reject losses >25% here, but that PREVENTED closing
            # real losing trades (e.g. meme coin shorts that 2x), making losses grow forever.
            # Real phantom trade protection is at position CREATION ($0 entry block).
            if pnl_pct < -50.0:
                logger.warning(f"⚠️ LARGE LOSS: {pnl_pct:.1f}% on {symbol} | Entry: {pos.entry_price}, Exit: {price}")
                logger.warning(f"⚠️ Recording this trade - loss is real. Close it to stop the bleeding.")
            
            # Log detailed close info
            emoji = "✅" if pnl >= 0 else "❌"
            logger.info(f"{emoji} CLOSED {pos.side.upper()} {symbol} | Reason: {reason} | "
                       f"PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%) | Held: {hold_hours:.1f}h | "
                       f"Entry: ${pos.entry_price:.4f} → Exit: ${price:.4f}")
            
            # === CRITICAL FIX: Sync balance from exchange instead of manual PnL math ===
            # Bybit already applied the real PnL. Using self.balance += pnl can cause
            # double-counting if _sync_balance_with_exchange already ran (which it does
            # frequently). Exchange balance is the source of truth in LIVE mode.
            if self.live_mode:
                try:
                    await self._sync_balance_with_exchange()
                    logger.info(f"💰 Balance synced from exchange after close: ${self.balance:.2f}")
                except Exception as sync_err:
                    logger.warning(f"⚠️ Could not sync balance after close: {sync_err}")
                    # Only apply PnL manually if exchange sync fails
                    self.balance += pnl
                    logger.warning(f"⚠️ Applied PnL manually as fallback: ${pnl:+.2f}")
            else:
                # Paper mode: no exchange to sync from
                self.balance += pnl
            
            # Update stats
            is_win = pnl >= 0
            self.stats.total_trades += 1
            self.stats.total_pnl += pnl
            self.stats.today_pnl += pnl
            self.weekly_pnl += pnl  # Track weekly P&L for circuit breaker
            
            # === RE-ENTRY COOLDOWN (applies to ALL closes, win or loss) ===
            # SMART COOLDOWN: RT Quick Exit / reversal exits get LONGER cooldown (300s vs 180s)
            _reentry_sym = symbol.replace('/', '').replace(':USDT', '').upper()
            _reentry_key = f"{_reentry_sym}_{pos.side.upper()}"
            _rt_exit_reason2 = reason.lower() if reason else ''
            _is_reversal_exit2 = any(kw in _rt_exit_reason2 for kw in ['rt:', 'quick exit', 'reversal', 'momentum against', 'efficiency drop'])
            _cooldown_secs2 = 300 if _is_reversal_exit2 else self.REENTRY_COOLDOWN_SECONDS
            self.reentry_cooldown[_reentry_key] = datetime.now(timezone.utc) + timedelta(seconds=_cooldown_secs2)
            logger.info(f"⏰ RE-ENTRY COOLDOWN: {_reentry_key} blocked for {_cooldown_secs2}s{' (REVERSAL EXIT → extended)' if _is_reversal_exit2 else ''}")
            
            if is_win:
                self.stats.winning_trades += 1
                self.consecutive_wins += 1
                self.consecutive_losses = 0
            else:
                self.stats.losing_trades += 1
                self.consecutive_losses += 1
                # === ADD PER-PAIR LOSS COOLDOWN ===
                symbol_key = symbol.replace('/', '').replace(':USDT', '').upper()
                cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=self.PAIR_LOSS_COOLDOWN_MINUTES)
                self.pair_loss_cooldown[symbol_key] = cooldown_until
                logger.info(f"⏰ PAIR COOLDOWN: {symbol_key} on cooldown until {cooldown_until.strftime('%H:%M:%S')} UTC (after {pnl_pct:.2f}% loss)")
                self.consecutive_wins = 0
            
            # Note: win_rate is a computed property, no need to set it
            
            # Record trade to history (with duplicate prevention)
            opened_at_str = pos.opened_at.strftime("%H:%M:%S") if hasattr(pos, 'opened_at') and pos.opened_at else "N/A"
            trade_date = pos.opened_at.strftime("%Y-%m-%d") if hasattr(pos, 'opened_at') and pos.opened_at else datetime.now(timezone.utc).strftime("%Y-%m-%d")
            
            self._record_trade_safe({
                "symbol": symbol.replace('/', ''),
                "side": pos.side.upper(),
                "entry": pos.entry_price,
                "exit": price,
                "pnl": pnl,
                "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "entry_time": opened_at_str,
                "exit_time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "date": trade_date,
                "opened_at": pos.opened_at.isoformat() if hasattr(pos, 'opened_at') and pos.opened_at else None,
                "closed_at": datetime.now(timezone.utc).isoformat(),
                "result": "WIN" if is_win else "LOSS",
                "exit_reason": reason,
                "position_num": 2,
                "tp1_hit": getattr(pos, 'tp1_hit', False),
                "tp2_hit": getattr(pos, 'tp2_hit', False)
            })
            
            # Log and notify
            emoji = "✅" if pnl >= 0 else "❌"
            logger.info(f"{emoji} CLOSED {pos.side.upper()} {symbol} | Reason: {reason} | PnL: ${pnl:+.2f}")
            
            # Clean up using standardized method (stamps _recently_closed_symbols)
            self._clear_position(symbol)
            
            # === AI OUTCOME TRACKING (Notebook #2 §13: populate was_correct) ===
            # Match this trade back to its AI decision and record the outcome
            try:
                duration_minutes = 0
                if hasattr(pos, 'opened_at') and pos.opened_at:
                    duration_minutes = int((datetime.now(timezone.utc) - pos.opened_at).total_seconds() / 60)
                # Try to find matching decision by symbol + timestamp proximity
                for dec in reversed(self.ai_tracker.decisions):
                    if dec.trade_opened and dec.trade_outcome is None:
                        # Match by symbol (normalize both)
                        dec_sym = dec.symbol.replace('/', '').replace(':USDT', '').upper()
                        pos_sym = symbol.replace('/', '').replace(':USDT', '').upper()
                        if dec_sym == pos_sym:
                            dec.trade_outcome = "WIN" if is_win else "LOSS"
                            dec.trade_pnl = pnl
                            dec.trade_exit_reason = reason
                            dec.trade_duration_minutes = duration_minutes
                            dec.was_correct = is_win if dec.approved else (not is_win)
                            self.ai_tracker._save_decisions()
                            logger.info(f"📊 AI decision {dec.decision_id} outcome: {'WIN' if is_win else 'LOSS'} (was_correct={dec.was_correct})")
                            break
            except Exception as ai_track_err:
                logger.warning(f"⚠️ AI outcome tracking failed: {ai_track_err}")
            
            # Notify
            if self.telegram.enabled:
                await self.telegram.notify_trade_closed(
                    symbol=symbol,
                    pnl=pnl,
                    pnl_pct=(pnl / (pos.entry_price * pos.size) * 100) if pos.size > 0 else 0,
                    reason=reason
                )
            
            # Save state
            self._save_trading_state()
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return False
    
    async def _close_position(self, reason: str, price: float, is_manual: bool = False):
        """Close the position completely with ML learning.
        
        Args:
            reason: Why the position is being closed
            price: Exit price
            is_manual: If True, this is a manual close (dashboard, telegram, AI chat)
        """
        pos = self.position
        if not pos:
            logger.warning("No position to close")
            return False
        
        # Serialize close operations to prevent race conditions
        async with self._close_lock:
            norm_sym = pos.symbol.replace('/', '').replace(':USDT', '').upper() if pos.symbol else 'PRIMARY'
            if norm_sym in self._closing_symbols:
                logger.warning(f"⚠️ {pos.symbol} close already in progress - skipping duplicate")
                return False
            self._closing_symbols.add(norm_sym)
        try:
            return await self._close_position_inner(reason, price, is_manual)
        finally:
            self._closing_symbols.discard(norm_sym)
    
    async def _close_position_inner(self, reason: str, price: float, is_manual: bool = False):
        """Inner implementation of close_position (protected by race guard)."""
        pos = self.position
        
        # ===== CRITICAL: VALIDATE PRICE TO PREVENT GARBAGE DATA =====
        if not validate_price(price, pos.entry_price, pos.symbol):
            logger.error(f"🚨 REFUSING TO CLOSE {pos.symbol} with invalid price: {price} (entry was {pos.entry_price})")
            logger.error(f"🚨 This would have corrupted balance! Skipping close until valid price.")
            return False  # Don't close with garbage data
        
        # === CANCEL SERVER-SIDE SL/TP ORDERS ===
        # These are no longer needed since we're closing the position
        try:
            await self._cancel_server_side_orders(pos.symbol)
        except Exception as e:
            logger.debug(f"Could not cancel server-side orders: {e}")
        
        # === LIVE MODE: Execute real close order ===
        # === PAPER MODE: Simulate close without sending order ===
        if not self.live_mode:
            logger.info(f"📝 PAPER MODE: Simulating close for {pos.symbol} (no real order)")
            
            # Calculate P&L on remaining size (with fees for realistic paper trading)
            if pos.side == "long":
                remaining_pnl = (price - pos.entry_price) * pos.remaining_size
                pnl_pct = ((price / pos.entry_price) - 1) * 100
            else:
                remaining_pnl = (pos.entry_price - price) * pos.remaining_size
                pnl_pct = ((pos.entry_price / price) - 1) * 100
            
            # Deduct entry + exit fees for realistic paper trading
            entry_fee = pos.size * pos.entry_price * self.FEE_TAKER
            exit_fee = pos.remaining_size * price * self.FEE_TAKER
            remaining_pnl -= (entry_fee + exit_fee)
            
            # Calculate TOTAL P&L for stats (remaining + TPs)
            # NOTE: TP partial profits were ALREADY added to self.balance in _hit_tp()
            # So we only add remaining_pnl to balance here, but use total for stats/recording
            total_trade_pnl = remaining_pnl
            if pos.tp1_hit:
                tp1_size = pos.size * self.TP1_PCT
                if pos.side == "long":
                    total_trade_pnl += (pos.tp1 - pos.entry_price) * tp1_size
                else:
                    total_trade_pnl += (pos.entry_price - pos.tp1) * tp1_size
            if pos.tp2_hit:
                tp2_size = pos.size * self.TP2_PCT
                if pos.side == "long":
                    total_trade_pnl += (pos.tp2 - pos.entry_price) * tp2_size
                else:
                    total_trade_pnl += (pos.entry_price - pos.tp2) * tp2_size
            
            # Update balance (paper) - ONLY add remaining_pnl (TP portions already added in _hit_tp)
            self.balance += remaining_pnl
            
            # Update peak balance
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance
            
            # Determine win/loss
            is_win = total_trade_pnl >= 0
            
            # === RE-ENTRY COOLDOWN (applies to ALL closes, win or loss) ===
            # SMART COOLDOWN: RT Quick Exit / reversal exits get LONGER cooldown
            _reentry_sym = pos.symbol.replace('/', '').replace(':USDT', '').upper()
            _reentry_key = f"{_reentry_sym}_{pos.side.upper()}"
            _rt_exit_reason3 = reason.lower() if isinstance(reason, str) else ''
            _is_reversal_exit3 = any(kw in _rt_exit_reason3 for kw in ['rt:', 'quick exit', 'reversal', 'momentum against', 'efficiency drop'])
            _cooldown_secs3 = 300 if _is_reversal_exit3 else self.REENTRY_COOLDOWN_SECONDS
            self.reentry_cooldown[_reentry_key] = datetime.now(timezone.utc) + timedelta(seconds=_cooldown_secs3)
            logger.info(f"⏰ RE-ENTRY COOLDOWN: {_reentry_key} blocked for {_cooldown_secs3}s{' (REVERSAL EXIT → extended)' if _is_reversal_exit3 else ''}")
            
            if is_win:
                self.consecutive_wins += 1
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                # === PER-PAIR LOSS COOLDOWN (paper unified close path) ===
                symbol_key = pos.symbol.replace('/', '').replace(':USDT', '').upper()
                cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=self.PAIR_LOSS_COOLDOWN_MINUTES)
                self.pair_loss_cooldown[symbol_key] = cooldown_until
                logger.info(f"⏰ PAIR COOLDOWN: {symbol_key} on cooldown until {cooldown_until.strftime('%H:%M:%S')} UTC (after loss)")
            
            # Update stats
            self.stats.record_trade(total_trade_pnl, is_win)
            
            # Calculate hold duration
            hold_hours = 0
            if pos.opened_at:
                hold_hours = (datetime.now(timezone.utc) - pos.opened_at).total_seconds() / 3600
            
            # Log the paper close
            emoji = "✅" if total_trade_pnl >= 0 else "❌"
            logger.info(f"{emoji} PAPER CLOSED {pos.side.upper()} {pos.symbol} | Reason: {reason} | "
                       f"PnL: ${total_trade_pnl:+.2f} ({pnl_pct:+.2f}%) | Held: {hold_hours:.1f}h | "
                       f"Entry: ${pos.entry_price:.4f} → Exit: ${price:.4f}")
            
            # Record trade
            trade_record = {
                'symbol': pos.symbol,
                'side': pos.side.upper(),
                'entry_price': pos.entry_price,
                'exit_price': price,
                'size': pos.size,
                'pnl': total_trade_pnl,
                'pnl_pct': pnl_pct,
                'result': 'WIN' if is_win else 'LOSS',
                'reason': reason,
                'entry_time': pos.opened_at.strftime('%H:%M:%S') if pos.opened_at else 'N/A',
                'exit_time': datetime.now(timezone.utc).strftime('%H:%M:%S'),
                'date': pos.opened_at.strftime('%Y-%m-%d') if pos.opened_at else datetime.now(timezone.utc).strftime('%Y-%m-%d'),
                'opened_at': pos.opened_at.isoformat() if pos.opened_at else None,
                'closed_at': datetime.now(timezone.utc).isoformat(),
                'hold_hours': hold_hours,
                'tp1_hit': pos.tp1_hit,
                'tp2_hit': pos.tp2_hit,
                'mode': 'paper'
            }
            self._record_trade_safe(trade_record)
            
            # === HISTORICAL ML: Record trade for model training (paper unified close path) ===
            # FIX: This was missing — Historical ML never learned from paper trades!
            try:
                if pos.entry_df_snapshot is not None:
                    ml_record_trade(pos.entry_df_snapshot, is_win)
                    logger.info(f"🧠 ML sample recorded (paper): {'WIN' if is_win else 'LOSS'}")
                elif len(self.bars_agg) >= 20:
                    ml_record_trade(self.bars_agg, is_win)
                    logger.info(f"🧠 ML sample recorded (paper fallback): {'WIN' if is_win else 'LOSS'}")
            except Exception as ml_err:
                logger.warning(f"⚠️ Historical ML recording failed (paper): {ml_err}")
            
            # === DUAL ML: Record trade for Live ML model (paper unified close path) ===
            try:
                entry_feats = getattr(pos, 'ml_entry_features', None) or {}
                self.ml_live.record_trade(
                    entry_features=entry_feats,
                    outcome=is_win,
                    pnl=total_trade_pnl,
                    symbol=pos.symbol,
                    side=pos.side.upper(),
                    reason=reason
                )
                logger.info(f"🧠 Dual ML: Live sample recorded — {self.ml_live.get_status()['total_samples']} total")
            except Exception as ml_err:
                logger.warning(f"⚠️ Dual ML recording failed (paper): {ml_err}")
                
            # === BAYESIAN WEIGHT LEARNING (paper unified close path) ===
            # FIX: This was missing — Bayesian learner never updated in paper mode!
            _hold_seconds_uni = hold_hours * 3600 if hold_hours else 0
            self.ai_filter.record_trade_result(
                is_win, total_trade_pnl, symbol=pos.symbol, side=pos.side.upper(),
                entry_component_scores=getattr(pos, 'entry_component_scores', None),
                exit_reason=reason,
                pnl_pct=pnl_pct,
                hold_time_seconds=_hold_seconds_uni,
                entry_time=getattr(pos, 'opened_at', None)
            )
            
            # Clear position using standardized method (stamps _recently_closed_symbols)
            self._clear_position(pos.symbol)
            self.position = None  # Also clear legacy pointer
            
            # Send notification
            try:
                if self.telegram and self.telegram.enabled:
                    await self.telegram.notify_trade_closed(
                        symbol=pos.symbol,
                        side=pos.side,
                        entry_price=pos.entry_price,
                        exit_price=price,
                        pnl=total_trade_pnl,
                        pnl_pct=pnl_pct,
                        reason=f"[PAPER] {reason}",
                        balance=self.balance,
                        total_pnl=self.stats.total_pnl,
                        win_rate=self.stats.win_rate * 100
                    )
            except Exception as e:
                logger.warning(f"Could not send paper close notification: {e}")
            
            logger.info(f"✅ PAPER CLOSE EXECUTED for {pos.symbol}")
            return True
        
        order_side = 'sell' if pos.side.lower() == 'long' else 'buy'
        order = await self._execute_market_order(pos.symbol, order_side, pos.remaining_size, reduce_only=True)
        
        if not order:
            logger.error(f"🚨 LIVE CLOSE ORDER FAILED for {pos.symbol}")
            return False
        
        if order.get('average'):
            price = float(order['average'])
            logger.info(f"📊 Close fill price: ${price:.4f}")
        
        # Calculate P&L on remaining size
        if pos.side == "long":
            remaining_pnl = (price - pos.entry_price) * pos.remaining_size
        else:
            remaining_pnl = (pos.entry_price - price) * pos.remaining_size
        
        # Deduct exit fee from remaining PnL
        exit_fee = pos.remaining_size * price * self.FEE_TAKER
        remaining_pnl -= exit_fee
        
        # Calculate TOTAL P&L for this trade (including already-taken TPs)
        # This is for accurate record-keeping
        total_trade_pnl = remaining_pnl
        if pos.tp1_hit:
            tp1_size = pos.size * self.TP1_PCT
            if pos.side == "long":
                total_trade_pnl += (pos.tp1 - pos.entry_price) * tp1_size
            else:
                total_trade_pnl += (pos.entry_price - pos.tp1) * tp1_size
        if pos.tp2_hit:
            tp2_size = pos.size * self.TP2_PCT
            if pos.side == "long":
                total_trade_pnl += (pos.tp2 - pos.entry_price) * tp2_size
            else:
                total_trade_pnl += (pos.entry_price - pos.tp2) * tp2_size
        
        # Add remaining P&L to balance (TPs already added to balance in _hit_tp)
        # In LIVE mode, sync from exchange (Bybit already applied real PnL).
        # In PAPER mode, apply manually since there's no exchange.
        if self.live_mode:
            try:
                await self._sync_balance_with_exchange()
                logger.info(f"💰 Balance synced from exchange after primary close: ${self.balance:.2f}")
            except Exception as sync_err:
                logger.warning(f"⚠️ Could not sync balance: {sync_err} - applying PnL manually")
                self.balance += remaining_pnl
        else:
            self.balance += remaining_pnl
        
        # Update peak balance for drawdown tracking
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        # Determine win/loss based on TOTAL trade P&L
        is_win = total_trade_pnl >= 0
        
        # === RE-ENTRY COOLDOWN (applies to ALL closes, win or loss) ===
        # SMART COOLDOWN: RT Quick Exit / reversal exits get LONGER cooldown
        _reentry_sym = pos.symbol.replace('/', '').replace(':USDT', '').upper()
        _reentry_key = f"{_reentry_sym}_{pos.side.upper()}"
        _rt_exit_reason4 = reason.lower() if isinstance(reason, str) else ''
        _is_reversal_exit4 = any(kw in _rt_exit_reason4 for kw in ['rt:', 'quick exit', 'reversal', 'momentum against', 'efficiency drop'])
        _cooldown_secs4 = 300 if _is_reversal_exit4 else self.REENTRY_COOLDOWN_SECONDS
        self.reentry_cooldown[_reentry_key] = datetime.now(timezone.utc) + timedelta(seconds=_cooldown_secs4)
        logger.info(f"⏰ RE-ENTRY COOLDOWN: {_reentry_key} blocked for {_cooldown_secs4}s{' (REVERSAL EXIT → extended)' if _is_reversal_exit4 else ''}")
        
        if is_win:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            # === ADD PER-PAIR LOSS COOLDOWN ===
            symbol_key = pos.symbol.replace('/', '').replace(':USDT', '').upper()
            cooldown_until = datetime.now(timezone.utc) + timedelta(minutes=self.PAIR_LOSS_COOLDOWN_MINUTES)
            self.pair_loss_cooldown[symbol_key] = cooldown_until
            logger.info(f"⏰ PAIR COOLDOWN: {symbol_key} on cooldown until {cooldown_until.strftime('%H:%M:%S')} UTC (after loss)")
        
        # ML Learning: record trade outcome for model training
        should_learn = self.ml_learn_all_trades or not is_manual
        
        if not should_learn:
            logger.info(f"🧠 ML sample SKIPPED: Manual trade (ml_learn_all_trades=False)")
        elif pos.entry_df_snapshot is not None:
            ml_record_trade(pos.entry_df_snapshot, is_win)
            logger.info(f"🧠 ML sample recorded: {'WIN' if is_win else 'LOSS'} (total: {len(self.bars_agg)} bars)")
        else:
            # Fallback: use current bars if entry snapshot was missing
            if len(self.bars_agg) >= 20:
                ml_record_trade(self.bars_agg, is_win)
                logger.info(f"🧠 ML sample recorded (fallback): {'WIN' if is_win else 'LOSS'}")
            else:
                logger.warning(f"⚠️ ML sample NOT recorded - no snapshot and only {len(self.bars_agg)} bars available")
        
        # === DUAL ML: Record trade for Live ML model ===
        if should_learn:
            try:
                entry_feats = getattr(pos, 'ml_entry_features', None) or {}
                self.ml_live.record_trade(
                    entry_features=entry_feats,
                    outcome=is_win,
                    pnl=total_trade_pnl,
                    symbol=pos.symbol,
                    side=pos.side.upper(),
                    reason=reason
                )
                logger.info(f"🧠 Dual ML: Live sample recorded — {self.ml_live.get_status()['total_samples']} total")
            except Exception as ml_err:
                logger.warning(f"⚠️ Dual ML recording failed: {ml_err}")
        
        # Record in trade history with TOTAL P&L (including TPs) - with duplicate prevention
        opened_at_str = pos.opened_at.strftime("%H:%M:%S") if pos.opened_at else "N/A"
        trade_date = pos.opened_at.strftime("%Y-%m-%d") if pos.opened_at else datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        self._record_trade_safe({
            "symbol": pos.symbol,
            "side": pos.side.upper(),
            "entry": pos.entry_price,
            "exit": price,
            "size": pos.remaining_size if pos.remaining_size > 0 else pos.size,
            "pnl": total_trade_pnl,  # Use total P&L including TPs
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "entry_time": opened_at_str,
            "exit_time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "date": trade_date,
            "opened_at": pos.opened_at.isoformat() if pos.opened_at else None,
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "result": "WIN" if is_win else "LOSS",
            "exit_reason": reason,
            "position_num": 1,
            "tp1_hit": pos.tp1_hit,
            "tp2_hit": pos.tp2_hit
        })
        
        # Log the close with full P&L
        pnl_pct = (total_trade_pnl / (pos.entry_price * pos.size)) * 100 if pos.size > 0 else 0
        logger.info(f"CLOSED {pos.side.upper()} | {reason} | P&L: ${total_trade_pnl:+.2f} ({pnl_pct:+.2f}%) | TPs: {pos.tp1_hit}/{pos.tp2_hit}")
        
        # Save state immediately to persist trade history
        self._save_trading_state()
        
        # Record result for AI filter learning (with Bayesian weight learning)
        _hold_seconds_live = 0
        if pos.opened_at:
            _hold_seconds_live = (datetime.now(timezone.utc) - pos.opened_at).total_seconds()
        self.ai_filter.record_trade_result(
            is_win, total_trade_pnl, symbol=pos.symbol, side=pos.side.upper(),
            entry_component_scores=getattr(pos, 'entry_component_scores', None),
            exit_reason=reason,
            pnl_pct=pnl_pct,
            hold_time_seconds=_hold_seconds_live,
            entry_time=getattr(pos, 'opened_at', None)
        )
        
        # === RECORD OUTCOME FOR AI DECISION TRACKING ===
        if self.current_decision_id:
            # Calculate trade duration using opened_at
            duration_minutes = 0
            if pos.opened_at:
                duration_minutes = int((datetime.now(timezone.utc) - pos.opened_at).total_seconds() / 60)
            
            self.ai_tracker.record_trade_outcome(
                decision_id=self.current_decision_id,
                outcome="WIN" if is_win else "LOSS",
                pnl=total_trade_pnl,
                exit_reason=reason,
                duration_minutes=duration_minutes
            )
            self.current_decision_id = None  # Clear for next trade
        
        if self.telegram.enabled:
            if "Stop" in reason:
                await self.telegram.notify_stop_loss(
                    symbol=self.SYMBOL,
                    price=price,
                    pnl=total_trade_pnl
                )
            else:
                await self.telegram.notify_trade_closed(
                    symbol=self.SYMBOL,
                    pnl=total_trade_pnl,
                    pnl_pct=pnl_pct,
                    reason=reason
                )
            # Note: Removed duplicate _notify_ai_decision("trade_closed") - already handled above
        
        # Release the position slot BEFORE clearing position
        if pos and hasattr(pos, 'symbol'):
            self._release_position_slot(pos.symbol)
        
        # Use standardized _clear_position to stamp _recently_closed_symbols
        self._clear_position(pos.symbol if pos else self.SYMBOL)
        self.position = None  # Ensure legacy pointer is also cleared
        
        # === PERSIST TRADING STATE AFTER TRADE CLOSES ===
        self._save_trading_state()
        
        # === AI AUTO-TUNE PARAMETERS ===
        # Check if it's time to let AI review and adjust parameters
        await self.maybe_auto_tune_params()
        
        return True  # Position closed successfully
    
    async def shutdown(self):
        """Clean shutdown."""
        logger.info("Shutting down...")
        self.running = False
        
        # === STOP WEBSOCKET STREAMS ===
        if self.ws_price_stream and self.ws_enabled:
            await self.ws_price_stream.stop()
            logger.info("📡 WebSocket price stream stopped")
        
        if hasattr(self, 'ws_position_stream') and self.ws_position_stream:
            await self.ws_position_stream.stop()
            logger.info("📡 WebSocket position stream stopped")
        
        # === SAVE TRADING STATE ON SHUTDOWN ===
        self._save_trading_state()
        logger.info("💾 Trading state saved on shutdown")
        
        if self.telegram.enabled:
            await self.telegram.send_message("🛑 *Julaba Bot Stopped*")
            await self.telegram.stop()
        
        if self.exchange:
            await self.exchange.close()
        
        logger.info(f"Final Balance: ${self.balance:,.2f} | Total P&L: ${self.stats.total_pnl:+,.2f}")


# ============== CLI Entry Point ==============

def main():
    parser = argparse.ArgumentParser(
        description="Julaba - AI-Enhanced Crypto Trading Bot"
    )
    parser.add_argument(
        "--paper-balance",
        type=float,
        default=None,
        help="Paper trading balance (enables paper mode)"
    )
    parser.add_argument(
        "--ai-confidence",
        type=float,
        default=0.7,
        help="AI confidence threshold (0.0-1.0)"
    )
    parser.add_argument(
        "--ai-mode",
        choices=["filter", "advisory", "autonomous", "hybrid"],
        default="autonomous",
        help="AI mode: filter (validate only), advisory (AI suggests), autonomous (AI trades), hybrid (AI scans + suggests)"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="LINK/USDT",
        help="Trading symbol (e.g., LINK/USDT, BTC/USDT)"
    )
    parser.add_argument(
        "--scan-interval",
        type=int,
        default=60,
        help="AI proactive scan interval in seconds (default: 60 = 1 min)"
    )
    parser.add_argument(
        "--summary-interval",
        type=int,
        default=14400,
        help="Autonomous summary interval in seconds (default: 14400 = 4 hours)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Dry-run mode: log trades without executing"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="LIVE TRADING: Execute real orders on exchange (USE WITH CAUTION!)"
    )
    parser.add_argument(
        "--daily-loss-limit",
        type=float,
        default=0.05,
        help="Daily loss limit as decimal (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--pretrain-ml",
        action="store_true",
        default=False,
        help="Pre-train ML model on historical data before starting"
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        default=False,
        help="Disable web dashboard (default: dashboard is ENABLED)"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        default=True,
        help="Enable web dashboard (default: enabled - kept for backward compatibility)"
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=5000,
        help="Dashboard port (default: 5000)"
    )
    
    args = parser.parse_args()
    
    # Setup file logging before anything else
    setup_logging(args.log_level)
    
    # ── HARDWARE IDENTITY LOCK ──────────────────────────────────────
    # Prevents unauthorized use on other machines
    from hwid_lock import enforce_hardware_lock, enforce_file_integrity
    enforce_hardware_lock()
    # ── FILE INTEGRITY CHECK ─────────────────────────────────────────
    # Prevents modification of compiled binaries and scripts
    enforce_file_integrity()
    # ────────────────────────────────────────────────────────────────
    
    # Create lockfile — kill old instance if still running
    lockfile_path = Path(__file__).parent / "julaba.lock"
    lockfile = None
    
    def _kill_old_bot(lock_path: Path) -> bool:
        """Read PID from lockfile and kill the old process if alive."""
        try:
            if not lock_path.exists():
                return True
            old_pid_str = lock_path.read_text().strip()
            if not old_pid_str or not old_pid_str.isdigit():
                logger.warning(f"🔓 Stale lockfile (no valid PID) — removing")
                lock_path.unlink(missing_ok=True)
                return True
            old_pid = int(old_pid_str)
            if old_pid == os.getpid():
                return True
            # Check if old process is alive
            try:
                os.kill(old_pid, 0)  # Signal 0 = check existence
            except ProcessLookupError:
                logger.info(f"🔓 Old bot (PID {old_pid}) already dead — removing stale lockfile")
                lock_path.unlink(missing_ok=True)
                return True
            except PermissionError:
                logger.warning(f"⚠️ Old bot (PID {old_pid}) exists but no permission to signal it")
                return False
            # Old bot is alive — kill it gracefully, then force
            logger.warning(f"⚠️ Killing old bot instance (PID {old_pid})...")
            try:
                os.kill(old_pid, 15)  # SIGTERM — graceful
                import time
                for i in range(10):  # Wait up to 5 seconds
                    time.sleep(0.5)
                    try:
                        os.kill(old_pid, 0)
                    except ProcessLookupError:
                        logger.info(f"✅ Old bot (PID {old_pid}) terminated gracefully")
                        lock_path.unlink(missing_ok=True)
                        return True
                # Still alive — force kill
                logger.warning(f"⚠️ Old bot (PID {old_pid}) didn't stop — sending SIGKILL")
                os.kill(old_pid, 9)  # SIGKILL
                time.sleep(1)
                lock_path.unlink(missing_ok=True)
                logger.info(f"✅ Old bot (PID {old_pid}) force-killed")
                return True
            except ProcessLookupError:
                logger.info(f"✅ Old bot (PID {old_pid}) already exited")
                lock_path.unlink(missing_ok=True)
                return True
            except Exception as kill_err:
                logger.error(f"❌ Failed to kill old bot (PID {old_pid}): {kill_err}")
                return False
        except Exception as e:
            logger.warning(f"⚠️ Lockfile cleanup error: {e}")
            try:
                lock_path.unlink(missing_ok=True)
            except Exception:
                pass
            return True
    
    # Kill old bot before acquiring lock
    if not _kill_old_bot(lockfile_path):
        logger.error("❌ Cannot kill old instance — aborting startup")
        sys.exit(1)
    
    # Small delay to let port free up after old process dies
    import time
    time.sleep(2)
    
    try:
        lockfile = open(lockfile_path, 'w')
        fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lockfile.write(str(os.getpid()))
        lockfile.flush()
        logger.info(f"🔒 Lockfile acquired (PID {os.getpid()}): {lockfile_path}")
    except BlockingIOError:
        logger.error("❌ Another instance of Julaba is already running (lock held)!")
        logger.error(f"   Lockfile: {lockfile_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to create lockfile: {e}")
        sys.exit(1)
    
    try:
        bot = Julaba(
            paper_balance=args.paper_balance,
            ai_confidence=args.ai_confidence,
            ai_mode=args.ai_mode,
            log_level=args.log_level,
            symbol=args.symbol,
            scan_interval=args.scan_interval,
            summary_interval=args.summary_interval
        )
        
        # Apply CLI overrides
        bot.dry_run_mode = args.dry_run
        bot.daily_loss_limit = args.daily_loss_limit
        
        # === TRADING MODE FROM CONFIG (overrides CLI) ===
        # This allows /mode command to persist across restarts
        config_mode = None
        config_file = Path(__file__).parent / "julaba_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                config_mode = config_data.get('trading_mode', None)
                logger.info(f"📋 Config trading_mode: {config_mode}")
            except Exception as e:
                logger.warning(f"Could not read config mode: {e}")
        
        # === LIVE TRADING MODE ===
        # Config mode takes priority over CLI (for /mode command persistence)
        if config_mode == 'paper':
            # Force paper mode even if --live was passed (mode switch command)
            logger.info("=" * 60)
            logger.info("📝 PAPER TRADING MODE (from config)")
            logger.info("   Simulated trading - no real orders")
            logger.info("=" * 60)
            bot.live_mode = False
            bot.paper_mode = True
            
            # Reload paper mode values from config (balance was loaded before paper_mode was set)
            if config_data:
                bot.balance = config_data.get('balance', 10000.0)
                bot.initial_balance = config_data.get('initial_balance', bot.balance)
                bot.peak_balance = config_data.get('peak_balance', bot.balance)
                stats = config_data.get('stats', {})
                bot.stats.total_trades = stats.get('total_trades', 0)
                bot.stats.winning_trades = stats.get('winning_trades', 0)
                bot.stats.losing_trades = stats.get('losing_trades', 0)
                bot.stats.total_pnl = stats.get('total_pnl', 0.0)
                bot.stats.today_pnl = stats.get('today_pnl', 0.0)
                bot.consecutive_wins = config_data.get('consecutive_wins', 0)
                bot.consecutive_losses = config_data.get('consecutive_losses', 0)
                logger.info(f"📝 Paper mode values: Balance=${bot.balance:,.2f}, Trades={bot.stats.total_trades}")
                
        elif args.live or config_mode == 'live':
            logger.warning("=" * 60)
            logger.warning("⚠️  LIVE TRADING MODE ENABLED  ⚠️")
            logger.warning("   REAL MONEY ORDERS WILL BE EXECUTED!")
            logger.warning("=" * 60)
            bot.live_mode = True
            bot.paper_mode = False
        
        # Setup signal handlers to save state on SIGTERM/SIGINT
        import signal as sig
        def graceful_shutdown(signum, frame):
            logger.info(f"⚠️ Received signal {signum}, saving state and shutting down...")
            bot._save_trading_state()
            logger.info("💾 Trading state saved on signal")
            bot.running = False
        
        sig.signal(sig.SIGTERM, graceful_shutdown)
        sig.signal(sig.SIGINT, graceful_shutdown)
        
        # Start dashboard (ENABLED by default, disabled with --no-dashboard)
        if not args.no_dashboard:
            bot.dashboard_enabled = True
            bot.dashboard.port = args.dashboard_port
            bot.dashboard.start()
            logger.info(f"🖥️ Dashboard enabled at http://localhost:{args.dashboard_port}")
        else:
            logger.info("⚠️ Dashboard disabled (--no-dashboard flag set)")
        
        # Pre-train ML model if requested
        if args.pretrain_ml:
            logger.info("🧠 Pre-training ML model on historical data...")
            asyncio.run(bot.pretrain_ml_model())
        
        # === RESILIENT BOT RUNNER ===
        # Keep dashboard running even if bot engine crashes
        max_restarts = 5
        restart_count = 0
        restart_cooldown = 30  # seconds between restarts
        should_run = True  # Local flag independent of bot.running
        
        while restart_count < max_restarts and should_run:
            try:
                logger.info(f"🚀 Starting trading engine (attempt {restart_count + 1}/{max_restarts})")
                asyncio.run(bot.run())
                
                # Normal shutdown (not a crash)
                break
                
            except KeyboardInterrupt:
                logger.info("🛑 Keyboard interrupt - shutting down")
                should_run = False
                break
                
            except Exception as fatal_error:
                restart_count += 1
                error_msg = f"Fatal engine error: {str(fatal_error)}"
                logger.error(f"💥 {error_msg}")
                
                # Track the fatal error
                bot._track_error(fatal_error, "fatal_crash")
                
                # === CRASH LOGGING TO FILE ===
                try:
                    import traceback
                    crash_log_path = Path(__file__).parent / "crash_log.txt"
                    crash_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    crash_tb = traceback.format_exc()
                    
                    with open(crash_log_path, "a") as crash_file:
                        crash_file.write("\n" + "="*80 + "\n")
                        crash_file.write(f"🔴 CRASH at {crash_timestamp}\n")
                        crash_file.write(f"Restart Count: {restart_count}/{max_restarts}\n")
                        crash_file.write(f"Error Type: {type(fatal_error).__name__}\n")
                        crash_file.write(f"Error Message: {str(fatal_error)}\n")
                        crash_file.write(f"\nFull Traceback:\n{crash_tb}\n")
                        crash_file.write("="*80 + "\n")
                    
                    logger.info(f"📝 Crash logged to {crash_log_path}")
                except Exception as log_error:
                    logger.error(f"Failed to write crash log: {log_error}")
                
                if restart_count < max_restarts:
                    logger.warning(f"⏳ Dashboard still running. Restarting engine in {restart_cooldown}s...")
                    
                    # Keep dashboard alive while we wait
                    import time
                    time.sleep(restart_cooldown)
                    
                    # Reset engine state for restart
                    bot.engine_running = False
                else:
                    logger.error(f"❌ Max restarts ({max_restarts}) reached. Engine stopped.")
                    logger.info("🖥️ Dashboard will continue running - check /api/errors for details")
                    
                    # Keep dashboard alive for error viewing
                    try:
                        while True:
                            import time
                            time.sleep(60)
                    except KeyboardInterrupt:
                        logger.info("🛑 Final shutdown requested")
                        break
    finally:
        # Release lockfile on exit
        if lockfile:
            fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)
            lockfile.close()
            try:
                lockfile_path.unlink()
                logger.info("🔓 Lockfile released")
            except (OSError, FileNotFoundError):
                pass  # Non-critical: lockfile already removed


if __name__ == "__main__":
    main()
