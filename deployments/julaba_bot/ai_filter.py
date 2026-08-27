"""
AI Signal Filter Module
Validates trading signals using AI analysis before execution.
Powered by Google Gemini.
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

# ── Hardware identity check (prevents code theft) ──────────────────
from hwid_lock import verify_hardware_lock as _hwid_check
if not _hwid_check():
    raise RuntimeError("🚫 Hardware identity mismatch — unauthorized machine")
# ───────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# Berlin timezone for accurate time detection
BERLIN_TZ = ZoneInfo("Europe/Berlin")

def get_berlin_time() -> datetime:
    """Get current time in Berlin timezone."""
    return datetime.now(BERLIN_TZ)

def get_utc_time() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)

# Gemini AI imports - try new package first, fall back to old
GENAI_AVAILABLE = False
GENAI_NEW = False

try:
    # Try new google-genai package first
    from google import genai as genai_new
    from google.genai import types
    GENAI_AVAILABLE = True
    GENAI_NEW = True
    logger.info("Using new google-genai package")
except ImportError:
    try:
        # Fall back to old google-generativeai
        import google.generativeai as genai_old
        GENAI_AVAILABLE = True
        GENAI_NEW = False
        logger.info("Using legacy google-generativeai package (deprecated)")
    except ImportError:
        logger.debug("No Gemini AI package installed")

# Persistent history file paths
HISTORY_DIR = Path(__file__).parent
AI_HISTORY_FILE = HISTORY_DIR / "ai_history.json"
TRADE_HISTORY_FILE = HISTORY_DIR / "trade_history.json"
CHAT_HISTORY_FILE = HISTORY_DIR / "chat_history.json"
EXIT_REASON_STATS_FILE = HISTORY_DIR / "exit_reason_stats.json"
SESSION_PERFORMANCE_FILE = HISTORY_DIR / "session_performance.json"
SYMBOL_MEMORY_FILE = HISTORY_DIR / "symbol_memory.json"
AUTO_TUNE_FILE = HISTORY_DIR / "auto_tune_history.json"
ADAPTIVE_PARAMS_FILE = HISTORY_DIR / "adaptive_params.json"


# ═══════════════════════════════════════════════════════════════════════════
# BAYESIAN WEIGHT LEARNER — Dirichlet-Multinomial Conjugate Model
# ═══════════════════════════════════════════════════════════════════════════
# Learns optimal component weights from trade outcomes.
#
# Theory:
#   Prior:     Dir(α₁, α₂, ..., αₖ)  — initial belief about weight importance
#   Posterior: Dir(α₁ + Δ₁, α₂ + Δ₂, ..., αₖ + Δₖ)
#   where Δᵢ reflects how well component i predicted the trade outcome.
#
# A component that scored HIGH on WINNING trades gets its alpha boosted.
# A component that scored HIGH on LOSING trades gets its alpha reduced.
# The posterior mean α_i / Σα gives the new weight for component i.
#
# This naturally:
#   - Converges to optimal weights over many trades
#   - Maintains regularization (won't collapse to 0 or 1)
#   - Respects initial priors when data is scarce
#   - Adapts slowly = stable, adapts fast = responsive (tunable)
# ═══════════════════════════════════════════════════════════════════════════
class ExitReasonLearner:
    """
    Learns from exit outcomes to identify which exit types are profitable
    and which consistently cut winners short or let losers run.
    
    After MIN_TRADES_FOR_LEARNING trades, provides:
    - Per-exit-type win rate, avg PnL, and recommendation
    - A penalty/bonus modifier for exits that are demonstrably bad/good
    """
    
    # Canonical exit reason categories (raw reasons get classified into these)
    EXIT_CATEGORIES = [
        'TRAILING_STOP', 'BREAK_EVEN_STOP', 'FISHERMAN', 'AI_REVERSAL',
        'CRITICAL_URGENCY', 'PROFIT_TO_LOSS', 'RT_QUICK_EXIT', 'TP1_HIT',
        'TP2_HIT', 'STOP_LOSS', 'AI_EXIT', 'MANUAL', 'OTHER'
    ]
    
    MIN_TRADES_FOR_LEARNING = 10  # Need this many trades before we adjust
    MIN_EXITS_PER_TYPE = 3        # Need this many of a specific exit type
    
    def __init__(self, filepath: Path = EXIT_REASON_STATS_FILE):
        self.filepath = filepath
        self.exit_records = []  # List of {category, pnl, pnl_pct, symbol, side, timestamp}
        self._load()
    
    def _classify_exit(self, reason: str) -> str:
        """Classify a raw exit reason string into a canonical category."""
        r = reason.upper()
        if 'TRAIL' in r:
            return 'TRAILING_STOP'
        elif 'BREAK' in r and ('EVEN' in r or 'BE' in r):
            return 'BREAK_EVEN_STOP'
        elif 'FISHER' in r:
            return 'FISHERMAN'
        elif 'AI' in r and 'REVERSAL' in r:
            return 'AI_REVERSAL'
        elif 'CRITICAL' in r or 'URGENCY' in r:
            return 'CRITICAL_URGENCY'
        elif 'PROFIT' in r and 'LOSS' in r:
            return 'PROFIT_TO_LOSS'
        elif 'RT' in r and ('QUICK' in r or 'EXIT' in r):
            return 'RT_QUICK_EXIT'
        elif 'TP2' in r:
            return 'TP2_HIT'
        elif 'TP1' in r:
            return 'TP1_HIT'
        elif 'STOP' in r and 'LOSS' in r:
            return 'STOP_LOSS'
        elif 'AI' in r and 'EXIT' in r:
            return 'AI_EXIT'
        elif 'MANUAL' in r or 'FORCE' in r or 'DASHBOARD' in r:
            return 'MANUAL'
        else:
            return 'OTHER'
    
    def record(self, exit_reason: str, pnl: float, pnl_pct: float = 0.0,
               symbol: str = "", side: str = "", hold_time_seconds: float = 0):
        """Record an exit outcome."""
        category = self._classify_exit(exit_reason)
        self.exit_records.append({
            'category': category,
            'raw_reason': exit_reason[:100],
            'pnl': round(pnl, 4),
            'pnl_pct': round(pnl_pct, 4),
            'symbol': symbol,
            'side': side,
            'hold_time': round(hold_time_seconds, 1),
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        # Keep last 200
        if len(self.exit_records) > 200:
            self.exit_records = self.exit_records[-200:]
        self._save()
        logger.info(f"📊 EXIT LEARNER: Recorded {category} → ${pnl:+.2f} ({pnl_pct:+.2f}%)")
    
    def get_stats(self) -> Dict[str, Dict]:
        """Get per-category statistics."""
        stats = {}
        for cat in self.EXIT_CATEGORIES:
            records = [r for r in self.exit_records if r['category'] == cat]
            if not records:
                continue
            wins = sum(1 for r in records if r['pnl'] > 0.01)
            losses = sum(1 for r in records if r['pnl'] < -0.01)
            total = len(records)
            avg_pnl = sum(r['pnl'] for r in records) / total
            avg_pnl_pct = sum(r['pnl_pct'] for r in records) / total
            avg_hold = sum(r.get('hold_time', 0) for r in records) / total
            win_rate = wins / total * 100 if total > 0 else 0
            
            # Quality rating
            if total >= self.MIN_EXITS_PER_TYPE:
                if avg_pnl > 0 and win_rate > 55:
                    quality = 'GOOD'
                elif avg_pnl < -0.5 or win_rate < 30:
                    quality = 'BAD'
                elif avg_pnl < 0:
                    quality = 'WEAK'
                else:
                    quality = 'NEUTRAL'
            else:
                quality = 'INSUFFICIENT_DATA'
            
            stats[cat] = {
                'count': total,
                'wins': wins,
                'losses': losses,
                'breakeven': total - wins - losses,
                'win_rate': round(win_rate, 1),
                'avg_pnl': round(avg_pnl, 4),
                'avg_pnl_pct': round(avg_pnl_pct, 4),
                'avg_hold_seconds': round(avg_hold, 1),
                'quality': quality
            }
        return stats
    
    def get_exit_modifier(self, exit_category: str) -> Dict[str, Any]:
        """
        Get a modifier for a specific exit type based on learned performance.
        Returns {'suppress': bool, 'delay_seconds': int, 'reason': str}
        
        Used by position monitor to suppress or delay exits that are demonstrably bad.
        """
        if len(self.exit_records) < self.MIN_TRADES_FOR_LEARNING:
            return {'suppress': False, 'delay_seconds': 0, 'reason': 'insufficient_data'}
        
        records = [r for r in self.exit_records if r['category'] == exit_category]
        if len(records) < self.MIN_EXITS_PER_TYPE:
            return {'suppress': False, 'delay_seconds': 0, 'reason': 'insufficient_type_data'}
        
        avg_pnl = sum(r['pnl'] for r in records) / len(records)
        win_rate = sum(1 for r in records if r['pnl'] > 0.01) / len(records) * 100
        
        # If this exit type consistently loses money, suggest suppression
        if avg_pnl < -0.3 and win_rate < 35:
            return {
                'suppress': True,
                'delay_seconds': 30,
                'reason': f'{exit_category} avg_pnl=${avg_pnl:.2f}, WR={win_rate:.0f}% → suppress'
            }
        elif avg_pnl < -0.1 and win_rate < 45:
            return {
                'suppress': False,
                'delay_seconds': 15,
                'reason': f'{exit_category} weak (avg=${avg_pnl:.2f}, WR={win_rate:.0f}%) → delay 15s'
            }
        
        return {'suppress': False, 'delay_seconds': 0, 'reason': 'exit_type_ok'}
    
    def get_diagnostics(self) -> str:
        """Human-readable diagnostics for Telegram/dashboard."""
        stats = self.get_stats()
        if not stats:
            return "📊 Exit Learner: No data yet"
        lines = [f"📊 *Exit Reason Learning* ({len(self.exit_records)} exits recorded)"]
        for cat, s in sorted(stats.items(), key=lambda x: x[1]['count'], reverse=True):
            emoji = '✅' if s['quality'] == 'GOOD' else '❌' if s['quality'] == 'BAD' else '⚠️' if s['quality'] == 'WEAK' else '📊'
            lines.append(f"  {emoji} {cat}: {s['count']}x | WR={s['win_rate']:.0f}% | avg=${s['avg_pnl']:+.2f} | {s['quality']}")
        return "\n".join(lines)
    
    def _load(self):
        try:
            if self.filepath.exists():
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                self.exit_records = data.get('records', [])[-200:]
                logger.info(f"📊 Exit learner loaded: {len(self.exit_records)} records")
        except Exception as e:
            logger.warning(f"Could not load exit stats: {e}")
            self.exit_records = []
    
    def _save(self):
        try:
            tmp = self.filepath.with_suffix('.json.tmp')
            with open(tmp, 'w') as f:
                json.dump({'records': self.exit_records[-200:],
                          'updated_at': datetime.now(timezone.utc).isoformat()}, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.rename(str(tmp), str(self.filepath))
        except Exception as e:
            logger.warning(f"Could not save exit stats: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# INTELLIGENT SYSTEM #2: SESSION / TIME-OF-DAY LEARNER
# Tracks win rate and PnL per UTC hour to avoid bad trading hours.
# ═══════════════════════════════════════════════════════════════════════════
class SessionPerformanceLearner:
    """
    Learns which UTC hours and day-of-week produce the best trading results.
    After enough data, provides a score modifier per hour.
    """
    
    MIN_TRADES_PER_HOUR = 3  # Need this many trades in an hour slot to judge it
    MIN_TOTAL_TRADES = 15    # Need this many total trades before applying modifiers
    
    def __init__(self, filepath: Path = SESSION_PERFORMANCE_FILE):
        self.filepath = filepath
        # {hour_0..23: [{pnl, result, symbol, side, timestamp}]}
        self.hourly_records = {str(h): [] for h in range(24)}
        # {day_0..6: [{pnl, result, timestamp}]} (0=Monday)
        self.daily_records = {str(d): [] for d in range(7)}
        self._load()
    
    def record(self, pnl: float, is_win: bool, symbol: str = "", side: str = "",
               entry_time: datetime = None):
        """Record a trade result with its entry hour."""
        if entry_time is None:
            entry_time = datetime.now(timezone.utc)
        
        hour = str(entry_time.hour)
        day = str(entry_time.weekday())
        
        record = {
            'pnl': round(pnl, 4),
            'result': 'WIN' if is_win else 'LOSS',
            'symbol': symbol,
            'side': side,
            'timestamp': entry_time.isoformat()
        }
        
        self.hourly_records[hour].append(record)
        self.daily_records[day].append(record)
        
        # Keep last 30 per slot
        if len(self.hourly_records[hour]) > 30:
            self.hourly_records[hour] = self.hourly_records[hour][-30:]
        if len(self.daily_records[day]) > 50:
            self.daily_records[day] = self.daily_records[day][-50:]
        
        self._save()
        logger.info(f"📊 SESSION LEARNER: Hour {hour}:00 UTC, Day {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][int(day)]} → {'WIN' if is_win else 'LOSS'} ${pnl:+.2f}")
    
    def get_hour_modifier(self, hour: int) -> Dict[str, Any]:
        """
        Get a score modifier for the given UTC hour.
        Returns {'modifier': int, 'reason': str, 'confidence': str}
        
        Modifier is added to the entry score:
        - Positive = good hour (boost score)
        - Negative = bad hour (penalize score)
        - Zero = insufficient data or neutral
        """
        total_trades = sum(len(v) for v in self.hourly_records.values())
        if total_trades < self.MIN_TOTAL_TRADES:
            return {'modifier': 0, 'reason': f'need {self.MIN_TOTAL_TRADES} trades (have {total_trades})', 'confidence': 'none'}
        
        records = self.hourly_records.get(str(hour), [])
        if len(records) < self.MIN_TRADES_PER_HOUR:
            return {'modifier': 0, 'reason': f'need {self.MIN_TRADES_PER_HOUR} trades at {hour}:00 (have {len(records)})', 'confidence': 'low'}
        
        wins = sum(1 for r in records if r['result'] == 'WIN')
        win_rate = wins / len(records) * 100
        avg_pnl = sum(r['pnl'] for r in records) / len(records)
        
        # Strong data (5+ trades)
        if len(records) >= 5:
            if win_rate >= 70 and avg_pnl > 0:
                return {'modifier': 5, 'reason': f'{hour}:00 UTC WR={win_rate:.0f}% avg=${avg_pnl:+.2f} (GOOD)', 'confidence': 'high'}
            elif win_rate <= 30 or avg_pnl < -0.5:
                return {'modifier': -10, 'reason': f'{hour}:00 UTC WR={win_rate:.0f}% avg=${avg_pnl:+.2f} (BAD)', 'confidence': 'high'}
            elif avg_pnl < -0.2:
                return {'modifier': -5, 'reason': f'{hour}:00 UTC WR={win_rate:.0f}% avg=${avg_pnl:+.2f} (WEAK)', 'confidence': 'medium'}
        else:
            # Moderate data (3-4 trades) — smaller modifiers
            if win_rate <= 25:
                return {'modifier': -5, 'reason': f'{hour}:00 UTC WR={win_rate:.0f}% (preliminary BAD)', 'confidence': 'low'}
            elif win_rate >= 75:
                return {'modifier': 3, 'reason': f'{hour}:00 UTC WR={win_rate:.0f}% (preliminary GOOD)', 'confidence': 'low'}
        
        return {'modifier': 0, 'reason': f'{hour}:00 UTC WR={win_rate:.0f}% (neutral)', 'confidence': 'medium'}
    
    def get_day_modifier(self, day: int) -> Dict[str, Any]:
        """Get modifier for day of week (0=Monday, 6=Sunday)."""
        total_trades = sum(len(v) for v in self.daily_records.values())
        if total_trades < self.MIN_TOTAL_TRADES:
            return {'modifier': 0, 'reason': 'insufficient_data'}
        
        records = self.daily_records.get(str(day), [])
        if len(records) < 3:
            return {'modifier': 0, 'reason': f'need 3 trades on this day (have {len(records)})'}
        
        wins = sum(1 for r in records if r['result'] == 'WIN')
        win_rate = wins / len(records) * 100
        avg_pnl = sum(r['pnl'] for r in records) / len(records)
        day_name = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day]
        
        if win_rate <= 30 or avg_pnl < -0.5:
            return {'modifier': -8, 'reason': f'{day_name} WR={win_rate:.0f}% avg=${avg_pnl:+.2f} (BAD)'}
        elif win_rate >= 70 and avg_pnl > 0:
            return {'modifier': 5, 'reason': f'{day_name} WR={win_rate:.0f}% avg=${avg_pnl:+.2f} (GOOD)'}
        
        return {'modifier': 0, 'reason': f'{day_name} WR={win_rate:.0f}% (neutral)'}
    
    def get_diagnostics(self) -> str:
        """Human-readable summary."""
        total = sum(len(v) for v in self.hourly_records.values())
        if total == 0:
            return "📊 Session Learner: No data yet"
        
        lines = [f"📊 *Session Performance* ({total} trades)"]
        lines.append("*By Hour (UTC):*")
        for h in range(24):
            records = self.hourly_records.get(str(h), [])
            if not records:
                continue
            wins = sum(1 for r in records if r['result'] == 'WIN')
            wr = wins / len(records) * 100
            avg = sum(r['pnl'] for r in records) / len(records)
            mod = self.get_hour_modifier(h)
            emoji = '🟢' if mod['modifier'] > 0 else '🔴' if mod['modifier'] < 0 else '⚪'
            lines.append(f"  {emoji} {h:02d}:00: {len(records)}x WR={wr:.0f}% avg=${avg:+.2f} mod={mod['modifier']:+d}")
        
        lines.append("*By Day:*")
        for d in range(7):
            records = self.daily_records.get(str(d), [])
            if not records:
                continue
            wins = sum(1 for r in records if r['result'] == 'WIN')
            wr = wins / len(records) * 100
            avg = sum(r['pnl'] for r in records) / len(records)
            day_name = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][d]
            lines.append(f"  {day_name}: {len(records)}x WR={wr:.0f}% avg=${avg:+.2f}")
        
        return "\n".join(lines)
    
    def _load(self):
        try:
            if self.filepath.exists():
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                for h in range(24):
                    self.hourly_records[str(h)] = data.get('hourly', {}).get(str(h), [])[-30:]
                for d in range(7):
                    self.daily_records[str(d)] = data.get('daily', {}).get(str(d), [])[-50:]
                total = sum(len(v) for v in self.hourly_records.values())
                logger.info(f"📊 Session learner loaded: {total} records")
        except Exception as e:
            logger.warning(f"Could not load session stats: {e}")
    
    def _save(self):
        try:
            tmp = self.filepath.with_suffix('.json.tmp')
            with open(tmp, 'w') as f:
                json.dump({'hourly': self.hourly_records, 'daily': self.daily_records,
                          'updated_at': datetime.now(timezone.utc).isoformat()}, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.rename(str(tmp), str(self.filepath))
        except Exception as e:
            logger.warning(f"Could not save session stats: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# INTELLIGENT SYSTEM #3: PER-SYMBOL PERSISTENT MEMORY
# Remembers win rate, PnL, and best direction per symbol across restarts.
# ═══════════════════════════════════════════════════════════════════════════
class SymbolMemory:
    """
    Persistent per-symbol performance tracker that survives restarts and resets.
    Provides score bonus/penalty per symbol at Stage 2 based on historical results.
    """
    
    MIN_TRADES_FOR_BIAS = 3  # Need 3+ trades on a symbol to apply bias
    MAX_SYMBOLS = 100        # Track up to 100 symbols
    
    def __init__(self, filepath: Path = SYMBOL_MEMORY_FILE):
        self.filepath = filepath
        # {SYMBOL_KEY: {long_wins, long_losses, long_pnl, short_wins, short_losses, short_pnl, last_trade_time}}
        self.symbols: Dict[str, Dict] = {}
        self._load()
    
    def record(self, symbol: str, side: str, pnl: float, is_win: bool):
        """Record a trade result for a symbol."""
        key = symbol.replace('/', '').replace(':USDT', '').upper()
        
        if key not in self.symbols:
            self.symbols[key] = {
                'long_wins': 0, 'long_losses': 0, 'long_pnl': 0.0,
                'short_wins': 0, 'short_losses': 0, 'short_pnl': 0.0,
                'total_trades': 0, 'last_trade_time': None
            }
        
        s = self.symbols[key]
        direction = side.upper()
        
        if direction == 'LONG':
            if is_win:
                s['long_wins'] += 1
            else:
                s['long_losses'] += 1
            s['long_pnl'] = round(s['long_pnl'] + pnl, 4)
        else:
            if is_win:
                s['short_wins'] += 1
            else:
                s['short_losses'] += 1
            s['short_pnl'] = round(s['short_pnl'] + pnl, 4)
        
        s['total_trades'] += 1
        s['last_trade_time'] = datetime.now(timezone.utc).isoformat()
        
        # Prune oldest symbols if over limit
        if len(self.symbols) > self.MAX_SYMBOLS:
            sorted_syms = sorted(self.symbols.items(), 
                               key=lambda x: x[1].get('last_trade_time', ''), reverse=True)
            self.symbols = dict(sorted_syms[:self.MAX_SYMBOLS])
        
        self._save()
        total = s['long_wins'] + s['long_losses'] + s['short_wins'] + s['short_losses']
        logger.info(f"📊 SYMBOL MEMORY: {key} {direction} {'WIN' if is_win else 'LOSS'} ${pnl:+.2f} (total: {total} trades)")
    
    def get_symbol_modifier(self, symbol: str, direction: str = None) -> Dict[str, Any]:
        """
        Get a score modifier for a symbol based on historical performance.
        
        Returns {'modifier': int, 'preferred_direction': str|None, 'reason': str}
        """
        key = symbol.replace('/', '').replace(':USDT', '').upper()
        s = self.symbols.get(key)
        
        if not s:
            return {'modifier': 0, 'preferred_direction': None, 'reason': 'no_history'}
        
        total = s['long_wins'] + s['long_losses'] + s['short_wins'] + s['short_losses']
        if total < self.MIN_TRADES_FOR_BIAS:
            return {'modifier': 0, 'preferred_direction': None, 
                    'reason': f'need {self.MIN_TRADES_FOR_BIAS} trades (have {total})'}
        
        # Overall stats
        total_wins = s['long_wins'] + s['short_wins']
        total_pnl = s['long_pnl'] + s['short_pnl']
        overall_wr = total_wins / total * 100 if total > 0 else 50
        
        # Direction-specific
        long_total = s['long_wins'] + s['long_losses']
        short_total = s['short_wins'] + s['short_losses']
        long_wr = (s['long_wins'] / long_total * 100) if long_total >= 2 else 50
        short_wr = (s['short_wins'] / short_total * 100) if short_total >= 2 else 50
        
        # Determine preferred direction
        preferred = None
        if long_total >= 2 and short_total >= 2:
            if long_wr > short_wr + 20:
                preferred = 'LONG'
            elif short_wr > long_wr + 20:
                preferred = 'SHORT'
        elif long_total >= 2 and long_wr > 60:
            preferred = 'LONG'
        elif short_total >= 2 and short_wr > 60:
            preferred = 'SHORT'
        
        # Score modifier based on overall performance
        modifier = 0
        if overall_wr >= 70 and total_pnl > 0:
            modifier = 5  # Strong performer
        elif overall_wr >= 60 and total_pnl > 0:
            modifier = 3  # Good performer
        elif overall_wr <= 30 or (total_pnl < -1.0 and total >= 5):
            modifier = -10  # Consistent loser — strong penalty
        elif overall_wr <= 40 and total_pnl < 0:
            modifier = -5  # Weak performer
        
        # Direction-specific penalty: if asking about a direction with bad track record
        if direction and direction.upper() == 'LONG' and long_total >= 3 and long_wr < 30:
            modifier -= 5
        elif direction and direction.upper() == 'SHORT' and short_total >= 3 and short_wr < 30:
            modifier -= 5
        
        return {
            'modifier': modifier,
            'preferred_direction': preferred,
            'overall_wr': round(overall_wr, 1),
            'total_trades': total,
            'total_pnl': round(total_pnl, 2),
            'long_wr': round(long_wr, 1),
            'short_wr': round(short_wr, 1),
            'reason': f'{key}: {total}t WR={overall_wr:.0f}% ${total_pnl:+.2f} pref={preferred}'
        }
    
    def get_diagnostics(self) -> str:
        """Human-readable summary of all tracked symbols."""
        if not self.symbols:
            return "📊 Symbol Memory: No data yet"
        
        lines = [f"📊 *Symbol Memory* ({len(self.symbols)} symbols tracked)"]
        
        # Sort by total trades descending
        for key, s in sorted(self.symbols.items(), 
                           key=lambda x: x[1].get('total_trades', 0), reverse=True)[:15]:
            total = s['total_trades']
            wins = s['long_wins'] + s['short_wins']
            wr = wins / total * 100 if total > 0 else 0
            pnl = s['long_pnl'] + s['short_pnl']
            mod = self.get_symbol_modifier(key)
            emoji = '🟢' if mod['modifier'] > 0 else '🔴' if mod['modifier'] < 0 else '⚪'
            pref = f" pref={mod['preferred_direction']}" if mod['preferred_direction'] else ""
            lines.append(f"  {emoji} {key}: {total}t WR={wr:.0f}% ${pnl:+.2f}{pref}")
        
        return "\n".join(lines)
    
    def _load(self):
        try:
            if self.filepath.exists():
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                self.symbols = data.get('symbols', {})
                logger.info(f"📊 Symbol memory loaded: {len(self.symbols)} symbols")
        except Exception as e:
            logger.warning(f"Could not load symbol memory: {e}")
            self.symbols = {}
    
    def _save(self):
        try:
            tmp = self.filepath.with_suffix('.json.tmp')
            with open(tmp, 'w') as f:
                json.dump({'symbols': self.symbols,
                          'updated_at': datetime.now(timezone.utc).isoformat()}, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.rename(str(tmp), str(self.filepath))
        except Exception as e:
            logger.warning(f"Could not save symbol memory: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# INTELLIGENT SYSTEM #4: AUTO-TUNER (Parameter Optimization Loop)
# Periodically evaluates if adjusting parameters would improve results.
# ═══════════════════════════════════════════════════════════════════════════
class AutoTuner:
    """
    After every TUNE_INTERVAL trades, evaluates recent trade results and
    makes small adjustments to trailing stop, break-even, and TP parameters.
    
    Uses a simple hill-climbing approach: try small changes, keep what helps.
    """
    
    TUNE_INTERVAL = 15   # Evaluate after every N trades
    MIN_TRADES = 10      # Don't tune until we have this many trades
    MAX_STEP_PCT = 10    # Max change per tune cycle: 10% of current value
    
    # Parameters we can tune and their sensible ranges
    TUNABLE = {
        'atr_mult': {'min': 1.5, 'max': 4.0, 'step': 0.25},
        'tp1_r': {'min': 0.8, 'max': 2.0, 'step': 0.1},
        'tp2_r': {'min': 1.2, 'max': 3.0, 'step': 0.2},
        'trail_trigger_r': {'min': 0.3, 'max': 1.5, 'step': 0.1},
    }
    
    def __init__(self, filepath: Path = AUTO_TUNE_FILE):
        self.filepath = filepath
        self.trade_count = 0
        self.last_tune_trade = 0
        self.tune_history = []  # [{trade_num, changes, reason, recent_wr, recent_pnl}]
        self._load()
    
    def should_tune(self) -> bool:
        """Check if it's time to evaluate parameters."""
        return (self.trade_count >= self.MIN_TRADES and 
                self.trade_count - self.last_tune_trade >= self.TUNE_INTERVAL)
    
    def evaluate_and_suggest(self, recent_trades: List[Dict], 
                            current_params: Dict) -> Dict[str, Any]:
        """
        Analyze recent trades and suggest parameter adjustments.
        
        Args:
            recent_trades: Last N trades with pnl, exit_reason, hold_time, etc.
            current_params: Current adaptive_params dict
            
        Returns:
            {'should_adjust': bool, 'adjustments': {param: new_value}, 'reason': str}
        """
        if len(recent_trades) < self.MIN_TRADES:
            return {'should_adjust': False, 'adjustments': {}, 'reason': 'insufficient_trades'}
        
        self.trade_count = len(recent_trades)
        
        if not self.should_tune():
            return {'should_adjust': False, 'adjustments': {},
                    'reason': f'next tune at trade #{self.last_tune_trade + self.TUNE_INTERVAL}'}
        
        # Analyze recent performance (last TUNE_INTERVAL trades)
        recent = recent_trades[-self.TUNE_INTERVAL:]
        wins = sum(1 for t in recent if t.get('pnl', 0) > 0.01)
        losses = len(recent) - wins
        wr = wins / len(recent) * 100 if recent else 0
        avg_pnl = sum(t.get('pnl', 0) for t in recent) / len(recent) if recent else 0
        
        # Analyze exit patterns
        sl_exits = sum(1 for t in recent if 'STOP_LOSS' in str(t.get('exit_reason', '')).upper() or 'SL' in str(t.get('exit_reason', '')).upper())
        early_exits = sum(1 for t in recent if any(x in str(t.get('exit_reason', '')).upper() 
                         for x in ['BREAK_EVEN', 'FISHERMAN', 'RT_QUICK', 'PROFIT_TO_LOSS']))
        tp_exits = sum(1 for t in recent if 'TP' in str(t.get('exit_reason', '')).upper())
        
        adjustments = {}
        reasons = []
        
        # Rule 1: Too many stop-loss hits → widen stops
        if sl_exits / len(recent) > 0.4 and 'atr_mult' in current_params:
            current = current_params['atr_mult'].get('current', 2.5)
            limit = self.TUNABLE['atr_mult']
            new_val = min(limit['max'], current + limit['step'])
            if new_val != current:
                adjustments['atr_mult'] = new_val
                reasons.append(f"SL rate {sl_exits}/{len(recent)} too high → ATR {current}→{new_val}")
        
        # Rule 2: Too many early exits cutting winners → relax trailing/break-even
        if early_exits / len(recent) > 0.5 and avg_pnl < 0.1:
            if 'trail_trigger_r' in current_params:
                current = current_params['trail_trigger_r'].get('current', 0.75)
                limit = self.TUNABLE['trail_trigger_r']
                new_val = min(limit['max'], current + limit['step'])
                if new_val != current:
                    adjustments['trail_trigger_r'] = new_val
                    reasons.append(f"Early exit rate {early_exits}/{len(recent)} → trail trigger {current}→{new_val}")
        
        # Rule 3: TP rarely hit → lower TP targets
        if tp_exits / len(recent) < 0.1 and wr < 50:
            if 'tp1_r' in current_params:
                current = current_params['tp1_r'].get('current', 1.2)
                limit = self.TUNABLE['tp1_r']
                new_val = max(limit['min'], current - limit['step'])
                if new_val != current:
                    adjustments['tp1_r'] = new_val
                    reasons.append(f"TP rate {tp_exits}/{len(recent)} low → TP1 {current}→{new_val}")
        
        # Rule 4: High win rate + positive PnL → slightly tighten (capture more profit)
        if wr >= 65 and avg_pnl > 0.2:
            if 'tp1_r' in current_params:
                current = current_params['tp1_r'].get('current', 1.2)
                limit = self.TUNABLE['tp1_r']
                new_val = max(limit['min'], current - limit['step'])
                # Only tighten if not already at minimum
                if new_val != current and new_val >= limit['min']:
                    adjustments['tp1_r'] = new_val
                    reasons.append(f"Good WR={wr:.0f}% → tighten TP1 {current}→{new_val}")
        
        should_adjust = len(adjustments) > 0
        
        if should_adjust:
            self.last_tune_trade = self.trade_count
            self.tune_history.append({
                'trade_num': self.trade_count,
                'time': datetime.now(timezone.utc).isoformat(),
                'adjustments': adjustments,
                'reason': '; '.join(reasons),
                'recent_wr': round(wr, 1),
                'recent_avg_pnl': round(avg_pnl, 4)
            })
            if len(self.tune_history) > 50:
                self.tune_history = self.tune_history[-50:]
            self._save()
            logger.info(f"🎛️ AUTO-TUNE: {'; '.join(reasons)}")
        
        return {
            'should_adjust': should_adjust,
            'adjustments': adjustments,
            'reason': '; '.join(reasons) if reasons else 'no adjustments needed',
            'recent_wr': round(wr, 1),
            'recent_avg_pnl': round(avg_pnl, 4)
        }
    
    def get_diagnostics(self) -> str:
        lines = [f"🎛️ *Auto-Tuner* (trades: {self.trade_count}, tunes: {len(self.tune_history)})"]
        if self.tune_history:
            for t in self.tune_history[-3:]:
                lines.append(f"  #{t['trade_num']}: {t['reason']} (WR={t['recent_wr']:.0f}%)")
        else:
            lines.append("  No adjustments made yet")
        return "\n".join(lines)
    
    def _load(self):
        try:
            if self.filepath.exists():
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                self.trade_count = data.get('trade_count', 0)
                self.last_tune_trade = data.get('last_tune_trade', 0)
                self.tune_history = data.get('history', [])[-50:]
                logger.info(f"📊 Auto-tuner loaded: {self.trade_count} trades, {len(self.tune_history)} tunes")
        except Exception as e:
            logger.warning(f"Could not load auto-tune history: {e}")
    
    def _save(self):
        try:
            tmp = self.filepath.with_suffix('.json.tmp')
            with open(tmp, 'w') as f:
                json.dump({'trade_count': self.trade_count,
                          'last_tune_trade': self.last_tune_trade,
                          'history': self.tune_history[-50:],
                          'updated_at': datetime.now(timezone.utc).isoformat()}, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.rename(str(tmp), str(self.filepath))
        except Exception as e:
            logger.warning(f"Could not save auto-tune history: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# INTELLIGENT SYSTEM #5: CROSS-POSITION CORRELATION GUARD
# Prevents opening correlated positions that double risk instead of diversifying.
# ═══════════════════════════════════════════════════════════════════════════
class CorrelationGuard:
    """
    Before opening position #2, checks if the candidate symbol is highly
    correlated with existing positions. High correlation = concentrated risk.
    
    Uses rolling 100-bar return correlation on 15m candles.
    """
    
    HIGH_CORRELATION_THRESHOLD = 0.70   # Block same-direction if corr > this
    MEDIUM_CORRELATION_THRESHOLD = 0.50  # Require higher score if corr > this
    SCORE_PENALTY_HIGH_CORR = 15         # Score penalty for high correlation
    SCORE_PENALTY_MEDIUM_CORR = 8        # Score penalty for medium correlation
    
    @staticmethod
    def calculate_correlation(df1, df2, lookback: int = 100) -> float:
        """
        Calculate Pearson correlation of returns between two price series.
        
        Args:
            df1, df2: DataFrames with 'close' column
            lookback: Number of bars to use
            
        Returns:
            Correlation coefficient (-1 to +1)
        """
        try:
            closes1 = df1['close'].tail(lookback).values
            closes2 = df2['close'].tail(lookback).values
            
            min_len = min(len(closes1), len(closes2))
            if min_len < 20:
                return 0.0  # Not enough data
            
            closes1 = closes1[-min_len:]
            closes2 = closes2[-min_len:]
            
            # Calculate returns
            ret1 = np.diff(closes1) / closes1[:-1]
            ret2 = np.diff(closes2) / closes2[:-1]
            
            # Pearson correlation
            if len(ret1) < 20:
                return 0.0
            
            corr = np.corrcoef(ret1, ret2)[0, 1]
            
            if np.isnan(corr):
                return 0.0
            
            return round(float(corr), 4)
        except Exception as e:
            logger.debug(f"Correlation calculation error: {e}")
            return 0.0
    
    @classmethod
    def check_correlation(cls, candidate_df, candidate_symbol: str,
                         candidate_direction: str,
                         existing_positions: list,
                         existing_dfs: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check if candidate is too correlated with existing positions.
        
        Args:
            candidate_df: DataFrame for the candidate symbol
            candidate_symbol: Symbol being considered
            candidate_direction: 'LONG' or 'SHORT'
            existing_positions: List of position objects/dicts
            existing_dfs: {symbol: DataFrame} for existing positions
            
        Returns:
            {'allowed': bool, 'penalty': int, 'correlations': dict, 'reason': str}
        """
        if not existing_positions or existing_dfs is None:
            return {'allowed': True, 'penalty': 0, 'correlations': {}, 'reason': 'no_existing_positions'}
        
        max_corr = 0.0
        max_corr_symbol = ""
        same_direction_high_corr = False
        correlations = {}
        
        for pos in existing_positions:
            if pos is None:
                continue
            
            pos_symbol = pos.symbol if hasattr(pos, 'symbol') else pos.get('symbol', '')
            pos_side = (pos.side if hasattr(pos, 'side') else pos.get('side', '')).upper()
            pos_key = pos_symbol.replace('/', '').replace(':USDT', '').upper()
            cand_key = candidate_symbol.replace('/', '').replace(':USDT', '').upper()
            
            if pos_key == cand_key:
                continue  # Same symbol — already blocked elsewhere
            
            pos_df = existing_dfs.get(pos_symbol) or existing_dfs.get(pos_key)
            if pos_df is None or len(pos_df) < 20:
                continue
            
            corr = cls.calculate_correlation(candidate_df, pos_df)
            correlations[pos_key] = corr
            
            if abs(corr) > abs(max_corr):
                max_corr = corr
                max_corr_symbol = pos_key
            
            # Same direction + high positive correlation = concentrated risk
            if corr > cls.HIGH_CORRELATION_THRESHOLD and candidate_direction == pos_side:
                same_direction_high_corr = True
        
        # Decision
        penalty = 0
        allowed = True
        reason_parts = []
        
        if same_direction_high_corr:
            penalty = cls.SCORE_PENALTY_HIGH_CORR
            reason_parts.append(f"HIGH corr with {max_corr_symbol} ({max_corr:.2f}) same direction → -{penalty}")
        elif abs(max_corr) > cls.HIGH_CORRELATION_THRESHOLD:
            penalty = cls.SCORE_PENALTY_HIGH_CORR
            reason_parts.append(f"HIGH corr with {max_corr_symbol} ({max_corr:.2f}) → -{penalty}")
        elif abs(max_corr) > cls.MEDIUM_CORRELATION_THRESHOLD:
            penalty = cls.SCORE_PENALTY_MEDIUM_CORR
            reason_parts.append(f"MEDIUM corr with {max_corr_symbol} ({max_corr:.2f}) → -{penalty}")
        
        reason = '; '.join(reason_parts) if reason_parts else f'low correlation ({max_corr:.2f})'
        
        if penalty > 0:
            logger.info(f"📊 CORRELATION GUARD: {candidate_symbol} vs {max_corr_symbol}: r={max_corr:.2f} → penalty={penalty}")
        
        return {
            'allowed': allowed,
            'penalty': penalty,
            'max_correlation': max_corr,
            'max_corr_symbol': max_corr_symbol,
            'correlations': correlations,
            'reason': reason
        }


# ═══════════════════════════════════════════════════════════════
# THRESHOLD PRESETS — switchable via /threshold command
# ═══════════════════════════════════════════════════════════════
THRESHOLD_PRESETS = {
    'default': {
        'label': '🔒 DEFAULT (proven)',
        'stage2': 50,
        'weekday': {
            'combined': 70,
            'math': 65,
            'ai_high_conf': 58,
            'long_preferred': 55,
            'reversal': 70,
        },
        'weekend': {
            'combined': 45,
            'math': 50,
            'ai_high_conf': 50,
            'long_preferred': 50,
            'reversal': 55,
        },
    },
    'loose': {
        'label': '🔓 LOOSE (more trades)',
        'stage2': 40,
        'weekday': {
            'combined': 50,
            'math': 48,
            'ai_high_conf': 45,
            'long_preferred': 45,
            'reversal': 55,
        },
        'weekend': {
            'combined': 38,
            'math': 42,
            'ai_high_conf': 40,
            'long_preferred': 40,
            'reversal': 48,
        },
    },
    'testing': {
        'label': '🧪 TESTING (aggressive entries for position mgmt testing)',
        'stage2': 20,
        'weekday': {
            'combined': 30,
            'math': 25,
            'ai_high_conf': 25,
            'long_preferred': 25,
            'reversal': 35,
        },
        'weekend': {
            'combined': 25,
            'math': 20,
            'ai_high_conf': 20,
            'long_preferred': 20,
            'reversal': 30,
        },
    },
}

THRESHOLD_MODE_FILE = os.path.join(os.path.dirname(__file__), 'threshold_mode.json')

def get_threshold_mode() -> str:
    """Load persisted threshold mode. Returns 'default' or 'loose'."""
    try:
        if os.path.exists(THRESHOLD_MODE_FILE):
            with open(THRESHOLD_MODE_FILE, 'r') as f:
                data = json.load(f)
            mode = data.get('mode', 'default')
            if mode in THRESHOLD_PRESETS:
                return mode
    except Exception:
        pass
    return 'default'

def set_threshold_mode(mode: str) -> bool:
    """Persist threshold mode. Returns True if valid."""
    if mode not in THRESHOLD_PRESETS:
        return False
    try:
        with open(THRESHOLD_MODE_FILE, 'w') as f:
            json.dump({'mode': mode, 'changed_at': datetime.now(timezone.utc).isoformat()}, f)
        return True
    except Exception as e:
        logger.error(f"Failed to save threshold mode: {e}")
        return False

def get_active_thresholds(is_weekend: bool = False) -> dict:
    """Get the currently active threshold values."""
    mode = get_threshold_mode()
    preset = THRESHOLD_PRESETS[mode]
    period = 'weekend' if is_weekend else 'weekday'
    return {
        'mode': mode,
        'label': preset['label'],
        'stage2': preset['stage2'],
        **preset[period],
    }


class AISignalFilter:
    """
    AI-powered signal filter that analyzes market conditions
    and validates trading signals before execution.
    Powered by Google Gemini.
    
    STRICT MODE: Higher threshold, skeptic prompt, loss cooldown.
    """
    
    def __init__(self, confidence_threshold: float = 0.75, notifier=None):
        """
        Initialize the AI Signal Filter.
        
        Args:
            confidence_threshold: Minimum confidence (0-1) required to approve a trade
                                 (RAISED to 0.75 - AI was rubber-stamping at 0.65)
            notifier: Optional TelegramNotifier instance for notifications
        """
        self.confidence_threshold = confidence_threshold
        self.notifier = notifier
        self.loss_cooldown_threshold = 0.85  # Raised from 0.80 - be more skeptical after losses
        
        # Rate limiting and cooldown tracking
        self.last_ai_call_time = None
        self.ai_call_count = 0
        self.ai_cooldown_until = None  # Set when API is rate limited
        self.ai_failures_in_row = 0
        self.max_failures_before_cooldown = 3
        
        # Gemini AI configuration
        self.ai_provider = "gemini"
        self.api_keys = []
        self.current_key_index = 0
        self.api_key = ""
        self.use_ai = False
        self.trade_history = []
        self.model = None
        self.client = None
        
        # Gemini configuration
        self.model_name = 'gemini-2.0-flash'  # Fast model for quick decisions
        
        # Primary key
        primary_key = os.getenv("GEMINI_API_KEY", "")
        if primary_key and "your_" not in primary_key.lower():
            self.api_keys.append(primary_key)
        
        # Secondary/backup key
        backup_key = os.getenv("GEMINI_API_KEY_2", "")
        if backup_key and "your_" not in backup_key.lower():
            self.api_keys.append(backup_key)
        
        self.api_key = self.api_keys[0] if self.api_keys else ""
        self.use_ai = bool(self.api_key and GENAI_AVAILABLE)
        
        if len(self.api_keys) > 1:
            logger.info(f"🔑 Dual Gemini API keys configured ({len(self.api_keys)} keys)")
        
        # Initialize Gemini model if available
        if self.use_ai:
            try:
                if GENAI_NEW:
                    # New google-genai package
                    self.client = genai_new.Client(api_key=self.api_key)
                    logger.info(f"Gemini AI initialized (new SDK) with model: {self.model_name}")
                else:
                    # Legacy google-generativeai package
                    genai_old.configure(api_key=self.api_key)
                    self.model = genai_old.GenerativeModel(self.model_name)
                    logger.info(f"Gemini AI initialized (legacy SDK) with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini model: {e}")
                self.use_ai = False
                self.model = None
                self.client = None
        else:
            if not GENAI_AVAILABLE:
                logger.warning("Gemini AI disabled - google-generativeai not installed")
            elif not self.api_key:
                logger.info("Gemini AI disabled - GEMINI_API_KEY not set")
        
        # Trading performance tracking - load from persistent storage
        self.recent_trades = []  # List of {"result": "win"/"loss", "pnl": float, "time": str, "symbol": str, "side": str}
        self.total_wins = 0
        self.total_losses = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        
        # === SIDE PERFORMANCE TRACKER ===
        # Tracks win rate per side (LONG/SHORT) to detect and block the losing side
        # If one side has 4+ trades with <25% win rate, BLOCK that side until it recovers
        self._side_stats = {
            'LONG': {'wins': 0, 'losses': 0, 'recent_pnl': []},
            'SHORT': {'wins': 0, 'losses': 0, 'recent_pnl': []}
        }
        self._side_blocked = {'LONG': False, 'SHORT': False}
        self.SIDE_BLOCK_MIN_TRADES = 4     # Need at least N trades to judge a side
        self.SIDE_BLOCK_MAX_LOSS_RATE = 0.75  # Block if >= 75% of recent trades are losses
        self.SIDE_BLOCK_LOOKBACK = 8       # Only look at last N trades per side
        
        # Load historical trades for AI context
        self._load_trade_history()
        

        
        # === INTELLIGENT SYSTEM #1: EXIT REASON LEARNER ===
        self.exit_learner = ExitReasonLearner()
        
        # === INTELLIGENT SYSTEM #2: SESSION PERFORMANCE LEARNER ===
        self.session_learner = SessionPerformanceLearner()
        
        # === INTELLIGENT SYSTEM #3: PER-SYMBOL PERSISTENT MEMORY ===
        self.symbol_memory = SymbolMemory()
        
        # === INTELLIGENT SYSTEM #4: AUTO-TUNER ===
        self.auto_tuner = AutoTuner()
        
        # === INTELLIGENT SYSTEM #5: CORRELATION GUARD ===
        self.correlation_guard = CorrelationGuard()
        
        logger.info("🧠 All 5 intelligent learning systems initialized")
        
        # Per-symbol loss tracking for cooldown periods
        # Dict of {symbol: {"last_loss_time": datetime, "consecutive_losses": int}}
        self.symbol_loss_tracker = {}
        _cooldown_mode = get_threshold_mode()
        self.SYMBOL_COOLDOWN_MINUTES = 5 if _cooldown_mode == 'testing' else (45 if _cooldown_mode == 'loose' else 120)  # Testing: 5min, Loose: 45min, Normal: 2hrs
        self.MAX_SYMBOL_LOSSES = 2  # After N consecutive losses, longer cooldown
        self.EXTENDED_COOLDOWN_MINUTES = 15 if _cooldown_mode == 'testing' else (120 if _cooldown_mode == 'loose' else 360)  # Testing: 15min, Loose: 2hrs, Normal: 6hrs
        
        # === GLOBAL LOSS COOLDOWN (any loss = all symbols pause) ===
        self._global_cooldown_until = None  # datetime when cooldown expires
        _cooldown_mode2 = get_threshold_mode()
        self.GLOBAL_COOLDOWN_MINUTES = 3 if _cooldown_mode2 == 'testing' else (8 if _cooldown_mode2 == 'loose' else 15)  # Testing: 3min, Loose: 8min, Normal: 15min
        self.GLOBAL_COOLDOWN_BIG_LOSS_MINUTES = 5 if _cooldown_mode2 == 'testing' else (15 if _cooldown_mode2 == 'loose' else 30)  # Testing: 5min, Loose: 15min, Normal: 30min
        self.BIG_LOSS_THRESHOLD = 1.0  # Dollar amount for "big loss"
        
        # === AI REJECTION CONSISTENCY CHECK ===
        # Track recent AI rejections per symbol to prevent flip-flopping
        # {symbol: {'rejections': int, 'last_rejection': datetime, 'last_direction': str}}
        self._ai_rejection_tracker: Dict[str, Dict] = {}
        self.AI_REJECTION_MEMORY_MINUTES = 3 if _cooldown_mode2 == 'testing' else (5 if _cooldown_mode2 == 'loose' else 10)  # Testing: 3min, Loose: 5min, Normal: 10min
        self.AI_REJECTION_BLOCK_THRESHOLD = 8 if _cooldown_mode2 == 'testing' else (6 if _cooldown_mode2 == 'loose' else 4)   # Testing: 8, Loose: 6, Normal: 4
        
        # === DAILY TRADE LIMIT ===
        self.MAX_TRADES_PER_DAY = 30 if get_threshold_mode() == 'testing' else (8 if get_threshold_mode() == 'loose' else 4)  # Testing: 30, Loose: 8, Normal: 4
        self._daily_trade_count = 0
        self._daily_trade_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        
        # Proactive scan threshold - lower = more aggressive
        # Weekday: 65 (strict), Weekend: reduced by 12 → 53 (capped at 43)
        self.proactive_threshold = 65
        
        # Chat history for conversational context
        self.chat_history = []
        self.max_chat_history = 20
        self._load_chat_history()
        
        # === REAL-TIME MOMENTUM CACHE ===
        # Cache indicators per symbol for fast-as-light exit decisions
        # Updated every 30s during position monitoring, checked every WebSocket tick
        self._momentum_cache = {}  # {symbol: {'roc_5': x, 'roc_10': y, 'rsi': z, 'ema_trend': bool, 'updated_at': datetime}}
        self._MOMENTUM_CACHE_MAX_AGE_SECONDS = 60  # Cache valid for 60 seconds
        # Also track peak profit per symbol for cross-check
        self._peak_profit_cache = {}  # {symbol: peak_pct}
        # Last context per symbol — fallback for soft loss cap when momentum_cache is empty
        self._last_context = {}  # {symbol: {'roc_5': x, 'roc_10': y, 'rsi': z}}
    
    def update_momentum_cache(self, symbol: str, roc_5: float, roc_10: float, rsi: float, 
                               ema_trend: bool, momentum_hope: int, accelerating_against: bool,
                               prev_rsi: float = None, volume_ratio: float = 1.0,
                               atomic_candle=None):
        """Update the cached momentum indicators for a symbol.
        
        Called during unified_position_decision to cache indicators for real-time use.
        prev_rsi: RSI from ~5 bars ago (≈1 min on 1m candles) for delta computation
        volume_ratio: Current volume vs average — for reversal confirmation
        """
        cache_entry = {
            'roc_5': roc_5,
            'roc_10': roc_10,
            'rsi': rsi,
            'prev_rsi': prev_rsi if prev_rsi is not None else rsi,
            'volume_ratio': volume_ratio,
            'ema_trend': ema_trend,
            'momentum_hope': momentum_hope,
            'accelerating_against': accelerating_against,
            'updated_at': datetime.now(timezone.utc)
        }
        
        # 🔬 ADVANCED ANALYSIS SNAPSHOT — embed tick-level metrics into the cache
        # This way ALL consumers of the momentum cache get atomic data automatically
        if atomic_candle is not None:
            try:
                ac_snap = atomic_candle.get_analysis(symbol)
                if ac_snap and ac_snap.tick_count >= 10:
                    cache_entry['atomic_velocity'] = ac_snap.velocity
                    cache_entry['atomic_acceleration'] = ac_snap.acceleration
                    cache_entry['atomic_path_efficiency'] = ac_snap.path_efficiency
                    cache_entry['atomic_direction_consistency'] = ac_snap.direction_consistency
                    cache_entry['atomic_momentum_decay'] = ac_snap.momentum_decay_rate
                    cache_entry['atomic_reversal_prob'] = ac_snap.reversal_probability
                    cache_entry['atomic_entry_quality'] = ac_snap.entry_quality
                    cache_entry['atomic_hurst'] = ac_snap.hurst_micro
                    cache_entry['atomic_swing_count'] = ac_snap.swing_count
            except Exception:
                pass  # Never let atomic break cache population
        
        self._momentum_cache[symbol] = cache_entry
        _prev_rsi_log = f"{prev_rsi:.0f}" if prev_rsi is not None else f"{rsi:.0f}"
        _ac_vel = cache_entry.get('atomic_velocity')
        _ac_log = f", 🔬v={_ac_vel:+.4f}" if _ac_vel is not None else ""
        logger.debug(f"📊 Cached momentum for {symbol}: ROC5={roc_5:+.2f}%, ROC10={roc_10:+.2f}%, RSI={rsi:.0f}, prevRSI={_prev_rsi_log}, vol={volume_ratio:.1f}x{_ac_log}")
    
    def update_peak_profit(self, symbol: str, peak_pct: float):
        """Update peak profit tracking for a symbol."""
        current = self._peak_profit_cache.get(symbol, 0)
        if peak_pct > current:
            self._peak_profit_cache[symbol] = peak_pct
            logger.debug(f"📈 Peak cached for {symbol}: {peak_pct:.2f}%")
    
    def get_peak_profit(self, symbol: str) -> float:
        """Get cached peak profit for a symbol."""
        return self._peak_profit_cache.get(symbol, 0)
    
    def clear_peak_profit(self, symbol: str):
        """Clear peak profit when position closes."""
        if symbol in self._peak_profit_cache:
            del self._peak_profit_cache[symbol]
    
    def get_intelligent_systems_diagnostics(self) -> str:
        """Get combined diagnostics from all 5 intelligent learning systems."""
        lines = ["🧠 *Intelligent Learning Systems*\n"]
        

        
        try:
            # Exit Reason Learner
            if hasattr(self, 'exit_learner'):
                lines.append(self.exit_learner.get_diagnostics())
                lines.append("")
        except Exception as e:
            lines.append(f"Exit Learner: Error - {e}\n")
        
        try:
            # Session Performance
            if hasattr(self, 'session_learner'):
                lines.append(self.session_learner.get_diagnostics())
                lines.append("")
        except Exception as e:
            lines.append(f"Session Learner: Error - {e}\n")
        
        try:
            # Symbol Memory
            if hasattr(self, 'symbol_memory'):
                lines.append(self.symbol_memory.get_diagnostics())
                lines.append("")
        except Exception as e:
            lines.append(f"Symbol Memory: Error - {e}\n")
        
        try:
            # Auto-Tuner
            if hasattr(self, 'auto_tuner'):
                lines.append(self.auto_tuner.get_diagnostics())
        except Exception as e:
            lines.append(f"Auto-Tuner: Error - {e}")
        
        return "\n".join(lines)
    
    def check_realtime_quick_exit(self, symbol: str, side: str, pnl_pct: float, 
                                   hold_seconds: float, peak_profit_pct: float = 0,
                                   dynamic_tp_pct: Dict = None,
                                   atomic_candle=None) -> Optional[Dict]:
        """
        FAST-AS-LIGHT real-time exit check using cached indicators.
        
        Called on every WebSocket price update to detect when we should exit immediately.
        Returns None if no action needed, or exit dict if should close.
        
        dynamic_tp_pct: Dict with per-trade TP percentages from calculate_dynamic_tp()
            {'tp1_pct': 0.0025, 'tp2_pct': 0.006, 'tp1_display': 0.25, 'factors': '...'}
        """
        # ═══════════════════════════════════════════════════════════════════
        # PEAK PROFIT SYNC - Use cached peak if passed peak is 0
        # This fixes the issue where WebSocket handler hasn't seen positive PnL
        # but the 30-second check has tracked a peak
        # ═══════════════════════════════════════════════════════════════════
        cached_peak = self.get_peak_profit(symbol)
        if cached_peak > peak_profit_pct:
            peak_profit_pct = cached_peak
            logger.debug(f"📈 {symbol}: Using cached peak {peak_profit_pct:.2f}% (passed peak was lower)")
        
        # Also update cache if we got a higher peak from position object
        if peak_profit_pct > 0:
            self.update_peak_profit(symbol, peak_profit_pct)
        
        # ═══════════════════════════════════════════════════════════════════
        # 🧮 DYNAMIC TP THRESHOLD SCALING
        # If the trade was opened with calculate_dynamic_tp(), scale all exit 
        # thresholds proportionally. A trade with TP1=0.50% should have higher
        # MINI-TP than one with TP1=0.20%.
        # ═══════════════════════════════════════════════════════════════════
        if dynamic_tp_pct is None:
            dynamic_tp_pct = {}
        
        # TP1 as display percentage (e.g., 0.25 means 0.25%)
        dyn_tp1_display = dynamic_tp_pct.get('tp1_display', 0.25)  # Default fallback
        # Scale factor: how much bigger/smaller is this trade's TP1 vs the 0.25% baseline
        tp_scale = max(dyn_tp1_display / 0.25, 0.5) if dyn_tp1_display > 0 else 1.0
        tp_scale = min(tp_scale, 3.0)  # Cap at 3x to prevent crazy values
        
        # === DYNAMIC GRACE PERIOD MULTIPLIER (Notebook #2 §11) ===
        # High volatility (wide TP) → extend grace periods by 50%
        # Low volatility (tight TP) → reduce grace periods by 25%
        # This prevents premature exits during volatile surges
        if tp_scale > 1.5:
            grace_mult = 1.50  # High vol → 50% longer grace
        elif tp_scale < 0.8:
            grace_mult = 0.75  # Low vol → 25% shorter grace
        else:
            grace_mult = 1.0   # Normal
        
        # ═══════════════════════════════════════════════════════════════════
        # 🛡️ GLOBAL FEE GUARD - NEVER exit in no-man's land
        # Round-trip fees = ~0.11%, so exits below fee floor are net losses
        # This guard ONLY applies to PROFIT exits (not loss caps!)
        # ═══════════════════════════════════════════════════════════════════
        FEE_GUARD_THRESHOLD = 0.12  # Minimum gross profit for ANY profit-motivated exit
        
        # ═══════════════════════════════════════════════════════════════════
        # 🚨🚨🚨 SMART LOSS CAP - Cut losses EARLY, ZERO LOSS STRATEGY
        # Better to exit at -0.10% than risk -0.50%+
        # ═══════════════════════════════════════════════════════════════════
        _rt_mode = get_threshold_mode()
        HARD_LOSS_CAP = -0.50 if _rt_mode == 'testing' else (-0.22 if _rt_mode == 'loose' else -0.20)  # Loose was -0.35 → losses blew past avg win of +0.17%
        SOFT_LOSS_CAP = -0.25 if _rt_mode == 'testing' else (-0.12 if _rt_mode == 'loose' else -0.10)  # Loose was -0.18 → too wide, let -0.22% through
        RT_GRACE_PERIOD = 30  # FIX: Always 30s — 45s was too long, let losses deepen from -0.25% to -0.45%
        
        # Check momentum for potential recovery (STRICT thresholds for zero-loss strategy)
        # FIX: Also check context/indicator data as fallback when cache is empty
        # OLD BUG: Cache was only populated inside reversal detection (only for profitable positions)
        # → Losing positions never had cache → soft loss cap always killed them instantly
        cache = self._momentum_cache.get(symbol)
        momentum_with_us = False
        
        if cache:
            roc_5 = cache.get('roc_5', 0)
            roc_10 = cache.get('roc_10', 0)
            is_long = side.upper() == "LONG"
            
            # Require STRONG momentum signal to allow recovery (raised thresholds)
            if is_long:
                momentum_with_us = roc_5 > 0.25 and roc_10 > 0.15  # Both must be positive & strong
            else:
                momentum_with_us = roc_5 < -0.25 and roc_10 < -0.15  # Both must be negative & strong
        else:
            # FALLBACK: No cache yet — check context from indicator if available
            # This prevents blind exits when the position just opened and cache isn't populated
            if hasattr(self, '_last_context') and symbol in self._last_context:
                ctx = self._last_context[symbol]
                _roc5 = ctx.get('roc_5', 0)
                _roc10 = ctx.get('roc_10', 0)
                is_long = side.upper() == "LONG"
                if is_long:
                    momentum_with_us = _roc5 > 0.25 and _roc10 > 0.15
                else:
                    momentum_with_us = _roc5 < -0.25 and _roc10 < -0.15
                if momentum_with_us:
                    logger.info(f"🔄 {symbol}: No momentum cache but context shows momentum WITH US (ROC5={_roc5:+.2f}%)")
        
        # HARD CAP: -0.50% - No recovery possible, just cut
        if pnl_pct <= HARD_LOSS_CAP:
            logger.warning(f"🚨🚨 {symbol}: HARD LOSS CAP! PnL {pnl_pct:.2f}% hit max limit!")
            return {
                'action': 'close',
                'confidence': 0.99,
                'reasoning': f"🚨 HARD LOSS CAP: {pnl_pct:.2f}% exceeds max loss limit ({HARD_LOSS_CAP}%). Cutting loss NOW!",
                'realtime': True
            }
        
        # SOFT CAP: -0.25% - Cut unless momentum is with us for recovery
        if pnl_pct <= SOFT_LOSS_CAP:
            # GRACE PERIOD: Don't cut brand new positions — spread/slippage can look like a loss
            if hold_seconds < RT_GRACE_PERIOD:
                logger.info(f"⏳ {symbol}: Soft loss {pnl_pct:.2f}% but only {hold_seconds:.0f}s old — RT GRACE PERIOD (need {RT_GRACE_PERIOD}s). HOLD.")
            elif momentum_with_us:
                # Give it a chance to recover - momentum is favorable
                # BUT: Check atomic candle — if momentum shows acceleration AGAINST us,
                # momentum indicator is lagging and recovery is unlikely
                atomic_overrides_momentum = False
                if atomic_candle is not None:
                    try:
                        ac_analysis = atomic_candle.get_analysis(symbol)
                        if ac_analysis and ac_analysis.tick_count >= 30:
                            is_long = side.upper() == "LONG"
                            vel = ac_analysis.velocity
                            # Atomic shows price accelerating AGAINST us despite "momentum with us"
                            vel_against = (is_long and vel < -0.005) or (not is_long and vel > 0.005)
                            if vel_against and ac_analysis.momentum_decay_rate < -30 and ac_analysis.path_efficiency > 0.4:
                                atomic_overrides_momentum = True
                                logger.warning(f"🔬 {symbol}: Atomic OVERRIDES momentum! v={vel:+.4f}%/s AGAINST, decay={ac_analysis.momentum_decay_rate:+.0f}%, eff={ac_analysis.path_efficiency:.2f}")
                    except Exception:
                        pass
                
                if atomic_overrides_momentum:
                    logger.warning(f"🚨🔬 {symbol}: SOFT LOSS + MOMENTUM OVERRIDE! PnL {pnl_pct:.2f}%, momentum lagging - momentum confirms loss deepening")
                    return {
                        'action': 'close',
                        'confidence': 0.96,
                        'reasoning': f"🔬 ADVANCED LOSS CUT: {pnl_pct:.2f}% — momentum indicator lagging, tick-level shows accelerating against us",
                        'realtime': True
                    }
                else:
                    logger.info(f"🔄 {symbol}: At {pnl_pct:.2f}% but momentum WITH US - allowing recovery attempt")
            else:
                # No favorable momentum - check if atomic says there's hope
                atomic_gives_hope = False
                if atomic_candle is not None:
                    try:
                        ac_analysis = atomic_candle.get_analysis(symbol)
                        if ac_analysis and ac_analysis.tick_count >= 30:
                            is_long = side.upper() == "LONG"
                            vel = ac_analysis.velocity
                            vel_with_us = (is_long and vel > 0.003) or (not is_long and vel < -0.003)
                            if vel_with_us and ac_analysis.path_efficiency > 0.5 and ac_analysis.direction_consistency > 55:
                                atomic_gives_hope = True
                                logger.info(f"🔬 {symbol}: Atomic shows recovery (v={vel:+.4f}%/s WITH us, eff={ac_analysis.path_efficiency:.2f}) — holding")
                    except Exception:
                        pass
                
                if atomic_gives_hope:
                    logger.info(f"🔄🔬 {symbol}: At {pnl_pct:.2f}% — no candle momentum but ATOMIC shows recovery — HOLD")
                else:
                    # No favorable momentum - cut the loss
                    logger.warning(f"🚨 {symbol}: SOFT LOSS CAP! PnL {pnl_pct:.2f}%, no momentum support - exiting")
                    return {
                        'action': 'close',
                        'confidence': 0.95,
                        'reasoning': f"🚨 LOSS CUT: {pnl_pct:.2f}% with no momentum support. Cutting before it gets worse!",
                        'realtime': True
                    }
        
        if atomic_candle is not None and hold_seconds > RT_GRACE_PERIOD:
            try:
                ac_analysis = atomic_candle.get_analysis(symbol)
                if ac_analysis and ac_analysis.tick_count >= 30:
                    is_long = side.upper() == "LONG"
                    vel = ac_analysis.velocity
                    accel = ac_analysis.acceleration
                    jerk = ac_analysis.jerk
                    eff = ac_analysis.path_efficiency
                    decay = ac_analysis.momentum_decay_rate
                    
                    # ─────────────────────────────────────────────────────
                    # CHECK 1: Advanced exit signal
                    # Advanced exit check
                    
                    
                    
                    # ─────────────────────────────────────────────────────
                    if pnl_pct > 0.30:
                        vel_with_us = (is_long and vel > 0.001) or (not is_long and vel < -0.001)
                        accel_against = (is_long and accel < -0.001) or (not is_long and accel > 0.001)
                        jerk_crash = (is_long and jerk < -0.5) or (not is_long and jerk > 0.5)
                        
                        if vel_with_us and accel_against and jerk_crash:
                            logger.warning(
                                f"🔬⚡ {symbol}: ATOMIC JERK REVERSAL! "
                                f"v={vel:+.5f} (WITH us) but a={accel:+.5f} (AGAINST), "
                                f"jerk={jerk:+.3f} (CRASH). Engine died — locking +{pnl_pct:.2f}%"
                            )
                            return {
                                'action': 'close',
                                'confidence': 0.94,
                                'reasoning': (
                                    f"🔬⚡ JERK REVERSAL: Price still moving WITH us (v={vel:+.5f}) "
                                    f"but acceleration reversed (a={accel:+.5f}) and jerk crashed "
                                    f"(j={jerk:+.3f}). Buyers exhausted — locking +{pnl_pct:.2f}% "
                                    f"BEFORE the reversal."
                                ),
                                'realtime': True
                            }
                    
                    # ─────────────────────────────────────────────────────
                    # CHECK 2: Advanced exit signal
                    # Path efficiency < 0.3 means price is going nowhere
                    # despite lots of movement. Pure noise = death by chop.
                    # Lock any profit > 0.20% before it evaporates.
                    # ─────────────────────────────────────────────────────
                    if pnl_pct > 0.20 and eff < 0.30:
                        # Confirm with swing count — high swings = confirmed chop
                        if ac_analysis.swing_count >= 6:
                            logger.warning(
                                f"🔬🌊 {symbol}: EFFICIENCY COLLAPSE! "
                                f"eff={eff:.2f} (<0.30), swings={ac_analysis.swing_count}, "
                                f"locking +{pnl_pct:.2f}% before chop kills it"
                            )
                            return {
                                'action': 'close',
                                'confidence': 0.90,
                                'reasoning': (
                                    f"🔬🌊 EFFICIENCY COLLAPSE: Path efficiency {eff:.2f} (price going "
                                    f"nowhere), {ac_analysis.swing_count} direction changes. Market is "
                                    f"pure noise — locking +{pnl_pct:.2f}%."
                                ),
                                'realtime': True
                            }
                    
                    # ─────────────────────────────────────────────────────
                    # CHECK 3: Advanced exit signal
                    # Velocity still with us but decelerating + momentum
                    # decay > 40%. The move is dying — get out at profit.
                    # ─────────────────────────────────────────────────────
                    if pnl_pct > 0.25 and peak_profit_pct > 0.30:
                        vel_with_us = (is_long and vel > 0.0005) or (not is_long and vel < -0.0005)
                        accel_against = (is_long and accel < 0) or (not is_long and accel > 0)
                        
                        if vel_with_us and accel_against and decay < -40:
                            pullback = peak_profit_pct - pnl_pct
                            if pullback > 0.05:  # Already losing some ground
                                logger.warning(
                                    f"🔬💨 {symbol}: MOMENTUM EXHAUSTION! "
                                    f"v={vel:+.5f} (fading), a={accel:+.5f}, "
                                    f"decay={decay:+.0f}%. Peak +{peak_profit_pct:.2f}% → +{pnl_pct:.2f}%"
                                )
                                return {
                                    'action': 'close',
                                    'confidence': 0.88,
                                    'reasoning': (
                                        f"🔬💨 MOMENTUM EXHAUSTION: Velocity fading (a={accel:+.5f}), "
                                        f"momentum lost {abs(decay):.0f}%. Peak +{peak_profit_pct:.2f}% "
                                        f"eroding to +{pnl_pct:.2f}%. Locking before full reversal."
                                    ),
                                    'realtime': True
                                }
                    
                    # ─────────────────────────────────────────────────────
                    # CHECK 4: VELOCITY REVERSAL — Price actively moving AGAINST
                    # us with high confidence (replaces old profit protect)
                    # ─────────────────────────────────────────────────────
                    if pnl_pct > 0.15 and peak_profit_pct > 0.20:
                        vel_against = (is_long and vel < -0.004) or (not is_long and vel > 0.004)
                        if vel_against and ac_analysis.reversal_probability > 55 and ac_analysis.entry_quality < 35:
                            pullback = peak_profit_pct - pnl_pct
                            if pullback > 0.08:
                                logger.warning(
                                    f"🔬🔄 {symbol}: ATOMIC VELOCITY REVERSAL! "
                                    f"v={vel:+.4f} AGAINST, rev_prob={ac_analysis.reversal_probability:.0f}%, "
                                    f"quality={ac_analysis.entry_quality:.0f}. Peak +{peak_profit_pct:.2f}% → +{pnl_pct:.2f}%"
                                )
                                return {
                                    'action': 'close',
                                    'confidence': 0.88,
                                    'reasoning': (
                                        f"🔬🔄 VELOCITY REVERSAL: Price moving against us "
                                        f"(v={vel:+.4f}%/s), reversal prob {ac_analysis.reversal_probability:.0f}%. "
                                        f"Locking +{pnl_pct:.2f}%."
                                    ),
                                    'realtime': True
                                }
                    
                    # ─────────────────────────────────────────────────────
                    # CHECK 5: CHOPPY MARKET — No direction, lock any profit
                    # ─────────────────────────────────────────────────────
                    if pnl_pct > 0.12 and ac_analysis.swing_count > 10 and eff < 0.15:
                        logger.warning(
                            f"🔬🌪️ {symbol}: CHOPPY MARKET EXIT — "
                            f"{ac_analysis.swing_count} swings, eff={eff:.2f}, "
                            f"locking +{pnl_pct:.2f}%"
                        )
                        return {
                            'action': 'close',
                            'confidence': 0.82,
                            'reasoning': (
                                f"🔬🌪️ CHOPPY MARKET: {ac_analysis.swing_count} micro-swings, "
                                f"efficiency {eff:.2f}. Locking +{pnl_pct:.2f}%."
                            ),
                            'realtime': True
                        }
            except Exception:
                pass  # Never let atomic analysis break the fast exit path
        
        # ═══════════════════════════════════════════════════════════════════
        # 🎯 SERVER-SIDE TP MISSED DETECTION
        # If we exceeded TP1 level (0.40%) but position is still open and profit dropping,
        # the server-side TP failed to trigger! Capture remaining profit NOW before it evaporates
        # ═══════════════════════════════════════════════════════════════════
        TP1_LEVEL = dyn_tp1_display  # Dynamic! Server-side TP1 from calculate_dynamic_tp()
        TP_MISSED_CAPTURE_MIN = 0.10  # Minimum profit to capture if TP missed
        
        if peak_profit_pct >= TP1_LEVEL and pnl_pct < peak_profit_pct * 0.5:
            # Peak exceeded TP1 level but profit has dropped significantly
            # This means server-side TP didn't trigger - capture what's left!
            if pnl_pct >= TP_MISSED_CAPTURE_MIN:
                logger.warning(f"🎯 {symbol}: TP MISSED! Peak was +{peak_profit_pct:.2f}% (≥TP1 +{TP1_LEVEL}%) but now +{pnl_pct:.2f}%. Server-side TP failed - locking profit NOW!")
                return {
                    'action': 'close',
                    'confidence': 0.98,
                    'reasoning': f"🎯 TP MISSED: Peak +{peak_profit_pct:.2f}% exceeded TP1 +{TP1_LEVEL}% but server-side TP didn't trigger. Capturing +{pnl_pct:.2f}% before it's gone!",
                    'realtime': True
                }
            else:
                logger.warning(f"🎯 {symbol}: TP MISSED and profit evaporated! Peak was +{peak_profit_pct:.2f}%, now +{pnl_pct:.2f}%")
                # Continue to normal profit protection - don't return here
        
        # ═══════════════════════════════════════════════════════════════════
        # 🛡️🛡️ SENSIBLE REVERSAL SYSTEM (v2) — 3-Tier Profit Management
        # Replaces the nervous MINI-TP / MICRO-TP logic that panic-sold
        # POWER at +0.38% on normal noise.
        #
        # PHILOSOPHY: Never turn a >0.5% winner into a loser, but also
        # don't exit prematurely on market noise.
        #
        # Tier 0: BREAKEVEN SHIELD — peak ≥ 0.5% → hard floor at +0.10%
        # Tier 1: SENSIBLE REVERSAL — 0.2%-0.8% only exit on CONFIRMED reversal
        # Tier 2: HIGH PROFIT FLOORS — ≥1.0% keep existing hard floors
        # Tier 3: STALENESS — ignore < 3min, exit > 15min if stuck
        # ═══════════════════════════════════════════════════════════════════
        
        is_long = side.upper() == "LONG"
        cache = self._momentum_cache.get(symbol)
        
        # --- Helper: compute reversal strength from cached indicators ---
        def _reversal_is_confirmed(c, is_long_pos: bool) -> tuple:
            """
            Returns (is_confirmed: bool, reason: str)
            A reversal is CONFIRMED only if:
              1) ROC_5 strongly against (< -0.50% for LONG, > +0.50% for SHORT)
              OR
              2) ROC_5 mildly against AND volume > 1.5× average (high-volume selling)
              OR
              3) RSI dropped > 5 pts in ~1 min AND ROC_5 against us
              OR
              4) 🔮 Advanced exit signal >= 45 (PREPARE_EXIT or EXIT_NOW)
            
            🔮 ADVANCED ANALYSIS VETO: Even if candle indicators say "reversal confirmed",
            if Analysis sees strong velocity WITH us + clean trajectory,
            the candle is lagging — VETO the reversal. This is the "future sight".
            Low-volume price wiggles are NOT confirmed → HOLD.
            """
            if not c:
                # 🔮 No candle cache — Additional analysis not available
                if atomic_candle is not None:
                    try:
                        _pred = atomic_candle._check_exit_signal(symbol, 'long' if is_long_pos else 'short')
                        if _pred['urgency'] >= 55:
                            return (True, f"🔮 Atomic-only: urgency={_pred['urgency']}, {' | '.join(_pred['signals'][:2])}")
                    except Exception:
                        pass
                return (False, "no cache")
            
            roc5 = c.get('roc_5', 0)
            roc10 = c.get('roc_10', 0)
            vol_r = c.get('volume_ratio', 1.0)
            rsi_now = c.get('rsi', 50)
            rsi_prev = c.get('prev_rsi', rsi_now)
            rsi_delta = rsi_now - rsi_prev  # negative = RSI dropped
            accel = c.get('accelerating_against', False)
            
            reasons = []
            
            if is_long_pos:
                mom_against = roc5 < 0
                strong_dump = roc5 < -0.50
                vol_confirm = mom_against and vol_r > 1.5
                rsi_confirm = rsi_delta < -5 and roc5 < -0.10
                accel_confirm = accel and roc5 < -0.30 and roc10 < -0.20
            else:  # SHORT
                mom_against = roc5 > 0
                strong_dump = roc5 > 0.50
                vol_confirm = mom_against and vol_r > 1.5
                rsi_confirm = rsi_delta > 5 and roc5 > 0.10  # RSI surging = bad for short
                accel_confirm = accel and roc5 > 0.30 and roc10 > 0.20
            
            if strong_dump:
                reasons.append(f"STRONG momentum reversal ROC5={roc5:+.2f}%")
            if vol_confirm:
                reasons.append(f"High-vol selling (ROC5={roc5:+.2f}%, vol={vol_r:.1f}x)")
            if rsi_confirm:
                reasons.append(f"RSI drop {rsi_delta:+.1f}pts ({rsi_prev:.0f}→{rsi_now:.0f})")
            if accel_confirm:
                reasons.append(f"Accelerating against (ROC5={roc5:+.2f}%, ROC10={roc10:+.2f}%)")
            
            candle_confirmed = strong_dump or vol_confirm or rsi_confirm or accel_confirm
            
            # 🔮 ADVANCED ANALYSIS PREDICTIVE LAYER
            atomic_confirmed = False
            atomic_vetoes = False
            if atomic_candle is not None:
                try:
                    _pred = atomic_candle._check_exit_signal(symbol, 'long' if is_long_pos else 'short')
                    
                    # ATOMIC CONFIRMS: Candle says nothing but ticks say EXIT
                    if not candle_confirmed and _pred['urgency'] >= 55:
                        atomic_confirmed = True
                        reasons.append(f"🔮 Atomic sees reversal: urgency={_pred['urgency']}, {_pred['verdict']}")
                    
                    # ATOMIC EARLY WARNING: Even mild candle signal + atomic urgency = confirmed
                    elif not candle_confirmed and mom_against and _pred['urgency'] >= 35:
                        atomic_confirmed = True
                        reasons.append(f"🔮 Atomic + mild candle: urgency={_pred['urgency']}")
                    
                    # ATOMIC VETO: Candle says reversal but ticks show strong momentum WITH us
                    elif candle_confirmed and _pred['verdict'] == 'HOLD_STRONG' and _pred['urgency'] < 15:
                        atomic_vetoes = True
                        reasons.clear()
                        reasons.append(f"🔮 Atomic VETOES candle reversal: {_pred['verdict']}, urgency={_pred['urgency']}")
                    
                    # ATOMIC BOOST: Both agree — boost confidence
                    elif candle_confirmed and _pred['urgency'] >= 45:
                        reasons.append(f"🔮 Atomic confirms: {_pred['verdict']}")
                except Exception:
                    pass
            
            confirmed = (candle_confirmed or atomic_confirmed) and not atomic_vetoes
            reason_str = " + ".join(reasons) if reasons else "noise only"
            return (confirmed, reason_str)
        
        # ─────────────────────────────────────────────────────────────
        # TIER 2: HIGH PROFIT FLOORS (≥1.0% peak) — UNCHANGED, working fine
        # These are HARD LIMITS for big winners
        # ─────────────────────────────────────────────────────────────
        HIGH_PROFIT_FLOORS = [
            # (peak_threshold, min_profit_floor, label)
            (2.00, 1.20, "MEGA"),    # Hit +2.0% → NEVER below +1.20%
            (1.50, 0.80, "HUGE"),    # Hit +1.5% → NEVER below +0.80%
            (1.00, 0.50, "BIG"),     # Hit +1.0% → NEVER below +0.50%
        ]
        
        for floor_peak, floor_min, floor_label in HIGH_PROFIT_FLOORS:
            if peak_profit_pct >= floor_peak and pnl_pct < floor_min:
                # Before triggering floor, check momentum — if STRONGLY with us, give room
                strong_mom_with_us = False
                if cache:
                    roc_5 = cache.get('roc_5', 0)
                    roc_10 = cache.get('roc_10', 0)
                    if is_long:
                        strong_mom_with_us = roc_5 > 0.25 and roc_10 > 0.15
                    else:
                        strong_mom_with_us = roc_5 < -0.25 and roc_10 < -0.15
                
                # 🔮 ADVANCED ANALYSIS: If candle says "strong momentum" but ticks 
                # show the move is exhausted, DON'T delay — save the profit NOW
                if strong_mom_with_us and atomic_candle is not None:
                    try:
                        _floor_pred = atomic_candle._check_exit_signal(symbol, side.lower())
                        if _floor_pred['urgency'] >= 50 or _floor_pred['exhaustion_level'] >= 60:
                            strong_mom_with_us = False  # Override delay
                            logger.warning(f"🔮🏦 {symbol}: Atomic OVERRIDES floor delay! urgency={_floor_pred['urgency']}, exhaust={_floor_pred['exhaustion_level']}% — candle momentum is stale")
                    except Exception:
                        pass
                
                # If momentum is STRONGLY with us AND profit is still above fee breakeven, delay
                if strong_mom_with_us and pnl_pct >= 0.15:
                    logger.info(f"🏦⏳ {symbol}: {floor_label} FLOOR delayed - profit +{pnl_pct:.2f}% above fees, STRONG momentum with us")
                else:
                    logger.warning(f"🏦 {symbol}: {floor_label} PROFIT FLOOR! Peak +{peak_profit_pct:.2f}% (≥{floor_peak}%) but now +{pnl_pct:.2f}% (below floor +{floor_min}%). LOCKING PROFIT!")
                    return {
                        'action': 'close',
                        'confidence': 0.99,
                        'reasoning': f"🏦 {floor_label} PROFIT FLOOR: Peak +{peak_profit_pct:.2f}% hit {floor_peak}% milestone. Profit dropped to +{pnl_pct:.2f}% below floor +{floor_min}%. ZERO LOSS = lock it!",
                        'realtime': True
                    }
        
        # ─────────────────────────────────────────────────────────────
        # TIER 0: BREAKEVEN SHIELD — The No-Loss Guarantee
        # Once peak ≥ 0.5%, ARM a hard floor at +0.10% (covers fees).
        # We do NOT panic-sell at +0.3% because momentum slowed.
        # We let it breathe. If it drops to +0.10%, we stop out GREEN.
        # ─────────────────────────────────────────────────────────────
        SHIELD_ARM_THRESHOLD = 0.50   # Peak must reach this to arm the shield
        SHIELD_PROFIT_FLOOR = 0.10    # Hard stop — NEVER go below this once armed
        
        if peak_profit_pct >= SHIELD_ARM_THRESHOLD and pnl_pct <= SHIELD_PROFIT_FLOOR:
            # 🔮 ATOMIC: Check for V-recovery — ticks surging WITH us at this exact moment?
            # Only save if we're at the floor (0.10%), not already negative
            _shield_saved = False
            if atomic_candle is not None and pnl_pct > 0:
                try:
                    _sh_pred = atomic_candle._check_exit_signal(symbol, side.lower())
                    # Strong V-recovery: urgency very low, move alive, not exhausted
                    if _sh_pred['verdict'] == 'HOLD_STRONG' and _sh_pred['urgency'] < 8 and _sh_pred['exhaustion_level'] < 15:
                        _shield_saved = True
                        logger.info(f"🔮🛡️ {symbol}: Breakeven shield SAVED by atomic V-recovery! urgency={_sh_pred['urgency']}, exhaust={_sh_pred['exhaustion_level']:.0f} — give it 1 more cycle")
                except Exception:
                    pass
            
            if not _shield_saved:
                # Shield armed and floor breached — EXIT to guarantee green
                logger.warning(f"🛡️ {symbol}: BREAKEVEN SHIELD! Peak +{peak_profit_pct:.2f}% armed shield, profit dropped to +{pnl_pct:.2f}% (≤ floor +{SHIELD_PROFIT_FLOOR}%). Exiting GREEN!")
                return {
                    'action': 'close',
                    'confidence': 0.99,
                    'reasoning': f"🛡️ BREAKEVEN SHIELD: Peak +{peak_profit_pct:.2f}% armed shield at +{SHIELD_PROFIT_FLOOR}%. Profit dropped to +{pnl_pct:.2f}%. Guaranteed green exit!",
                    'realtime': True
                }
        
        # ─────────────────────────────────────────────────────────────
        # TIER 1: SENSIBLE REVERSAL VALIDATION — The Noise Filter
        # In the 'Small Win Zone' (0.20% – 0.80%), we DO NOT panic-sell
        # on simple price wiggles. We only exit if the reversal is
        # CONFIRMED by momentum + volume + RSI evidence.
        #
        # Below 0.20%: more sensitive (capital protection)
        # 0.20% – 0.80%: lenient — only exit on confirmed reversal
        # Above 0.80%: handled by high floors + trailing above
        # ─────────────────────────────────────────────────────────────
        SENSIBLE_GRACE_SECONDS = int(20 * grace_mult)
        
        if 0.20 <= pnl_pct <= 0.80 and hold_seconds >= SENSIBLE_GRACE_SECONDS:
            confirmed, rev_reason = _reversal_is_confirmed(cache, is_long)
            
            if confirmed:
                # Reversal is REAL — confirmed by momentum + volume/RSI evidence
                logger.warning(f"🔄 {symbol}: SENSIBLE REVERSAL EXIT at +{pnl_pct:.2f}% (peak +{peak_profit_pct:.2f}%) | {rev_reason}")
                return {
                    'action': 'close',
                    'confidence': 0.93,
                    'reasoning': f"🔄 SENSIBLE REVERSAL: Confirmed reversal at +{pnl_pct:.2f}% | {rev_reason}. Locking profit.",
                    'realtime': True
                }
            else:
                # NOT confirmed — this is just noise. HOLD and let the trade breathe.
                # The Breakeven Shield protects us if peak was ≥ 0.5%.
                logger.debug(f"🔇 {symbol}: Small Win Zone +{pnl_pct:.2f}% — reversal NOT confirmed ({rev_reason}). HOLD.")
        
        # ─────────────────────────────────────────────────────────────
        # TIER 3: TIME-BASED STALENESS — Opportunity Cost Filter
        # < 3 min: IGNORE staleness — let the move develop
        # > 15 min AND PnL < 0.3%: EXIT — money is stuck
        # ─────────────────────────────────────────────────────────────
        STALE_IGNORE_SECONDS = 180     # 3 minutes — too early to call it stale
        STALE_EXIT_SECONDS = 900       # 15 minutes — too long for a micro-profit
        STALE_EXIT_MAX_PNL = 0.30      # Only exit if PnL below this (above = worth holding)
        STALE_EXIT_MIN_PNL = FEE_GUARD_THRESHOLD  # Must be above fees to be worth exiting
        
        if hold_seconds >= STALE_EXIT_SECONDS and STALE_EXIT_MIN_PNL <= pnl_pct < STALE_EXIT_MAX_PNL:
            # 🔮 ADVANCED ANALYSIS: Don't kill a trade that's about to break out!
            # If ticks show sudden velocity surge + clean trajectory, HOLD
            _stale_vetoed = False
            if atomic_candle is not None:
                try:
                    _stale_pred = atomic_candle._check_exit_signal(symbol, side.lower())
                    if _stale_pred['verdict'] == 'HOLD_STRONG' and _stale_pred['urgency'] < 10:
                        _stale_vetoed = True
                        logger.info(f"🔮⏰ {symbol}: Stale exit VETOED by atomic — breakout forming! urgency={_stale_pred['urgency']}, {' | '.join(_stale_pred['signals'][:2])}")
                except Exception:
                    pass
            
            if not _stale_vetoed:
                logger.warning(f"⏰ {symbol}: STALE PROFIT! Held {hold_seconds/60:.1f}min with only +{pnl_pct:.2f}% (< {STALE_EXIT_MAX_PNL}%). Opportunity cost — exiting.")
                return {
                    'action': 'close',
                    'confidence': 0.88,
                    'reasoning': f"⏰ STALE PROFIT: {hold_seconds/60:.1f}min holding, profit stuck at +{pnl_pct:.2f}% (< {STALE_EXIT_MAX_PNL}%). Capital is better deployed elsewhere.",
                    'realtime': True
                }
        
        # ═══════════════════════════════════════════════════════════════════
        # 🔇 LEGACY MINI-TP / MICRO-TP — SUPPRESSED in 0.20%–0.80% zone
        # The Sensible Reversal system above handles this zone now.
        # These only fire OUTSIDE the sensible zone (below 0.20% or above 0.80%)
        # to provide backward-compatible fallback protection.
        # ═══════════════════════════════════════════════════════════════════
        _in_sensible_zone = 0.20 <= pnl_pct <= 0.80
        
        # MINI-TP: Only fires OUTSIDE the sensible zone
        MINI_TP_THRESHOLD = max(0.35 * tp_scale, 0.12)
        MINI_TP_MIN_PEAK = max(0.40 * tp_scale, 0.15)
        MINI_TP_GRACE_SECONDS = int(20 * grace_mult)
        
        if not _in_sensible_zone and pnl_pct >= MINI_TP_THRESHOLD and peak_profit_pct >= MINI_TP_MIN_PEAK and hold_seconds >= MINI_TP_GRACE_SECONDS:
            cache_mt = self._momentum_cache.get(symbol)
            momentum_fading = False
            momentum_with_us = False
            
            if cache_mt:
                roc_5 = cache_mt.get('roc_5', 0)
                is_long_mt = side.upper() == "LONG"
                if is_long_mt:
                    if roc_5 < -0.08:
                        momentum_fading = True
                    elif roc_5 > 0.05:
                        momentum_with_us = True
                else:
                    if roc_5 > 0.08:
                        momentum_fading = True
                    elif roc_5 < -0.05:
                        momentum_with_us = True
            
            drawdown_from_mini_peak = peak_profit_pct - pnl_pct
            significant_pullback = drawdown_from_mini_peak > 0.08
            
            if momentum_fading:
                # 🔮 ATOMIC: Check if ticks disagree — candle ROC fading but ticks surging?
                _mini_tp_vetoed = False
                if atomic_candle is not None:
                    try:
                        _mt_pred = atomic_candle._check_exit_signal(symbol, side.lower())
                        if _mt_pred['verdict'] == 'HOLD_STRONG' and _mt_pred['urgency'] < 12:
                            _mini_tp_vetoed = True
                            logger.info(f"🔮💰 {symbol}: MINI-TP VETOED by atomic — ticks show momentum alive! urgency={_mt_pred['urgency']}")
                    except Exception:
                        pass
                
                if not _mini_tp_vetoed:
                    logger.warning(f"💰 {symbol}: MINI-TP (outside sensible zone)! Locking +{pnl_pct:.2f}% (peak +{peak_profit_pct:.2f}%, ROC5={cache_mt.get('roc_5', 0) if cache_mt else 0:+.2f}%)")
                    return {
                        'action': 'close',
                        'confidence': 0.92,
                        'reasoning': f"💰 MINI-TP: Locking +{pnl_pct:.2f}% profit (momentum fading). Peak +{peak_profit_pct:.2f}%. Net profit after fees.",
                        'realtime': True
                    }
            elif significant_pullback and not momentum_with_us:
                logger.warning(f"💰 {symbol}: MINI-TP (outside sensible zone)! Locking +{pnl_pct:.2f}% (peak +{peak_profit_pct:.2f}%, pullback {drawdown_from_mini_peak:.2f}%)")
                return {
                    'action': 'close',
                    'confidence': 0.90,
                    'reasoning': f"💰 MINI-TP: Locking +{pnl_pct:.2f}% profit (lost {drawdown_from_mini_peak:.2f}% from peak, no momentum). Net profit after fees.",
                    'realtime': True
                }
        
        # MICRO-TP stale detector: SUPPRESSED in first 3 minutes, only outside sensible zone
        MICRO_TP_MIN_PNL = 0.25
        MICRO_TP_STALE_SECONDS = min(90 * tp_scale, 180)
        MICRO_TP_DRIFT = 0.06
        MICRO_TP_GRACE_SECONDS = max(int(30 * grace_mult), STALE_IGNORE_SECONDS)  # At least 3 min!
        
        if not hasattr(self, '_stale_profit_tracker'):
            self._stale_profit_tracker = {}
        
        stale_key = f"{symbol}_{side}"
        
        if not _in_sensible_zone and pnl_pct >= MICRO_TP_MIN_PNL and hold_seconds >= MICRO_TP_GRACE_SECONDS:
            now = datetime.now(timezone.utc)
            
            if stale_key not in self._stale_profit_tracker:
                self._stale_profit_tracker[stale_key] = {
                    'first_seen': now,
                    'pnl_at_start': pnl_pct,
                    'peak_at_start': peak_profit_pct
                }
            else:
                tracker = self._stale_profit_tracker[stale_key]
                elapsed = (now - tracker['first_seen']).total_seconds()
                pnl_drift = abs(pnl_pct - tracker['pnl_at_start'])
                
                if pnl_drift > MICRO_TP_DRIFT:
                    self._stale_profit_tracker[stale_key] = {
                        'first_seen': now,
                        'pnl_at_start': pnl_pct,
                        'peak_at_start': peak_profit_pct
                    }
                elif elapsed >= MICRO_TP_STALE_SECONDS:
                    # 🔮 ATOMIC: Stale profit but check if breakout just started in ticks
                    _micro_tp_vetoed = False
                    if atomic_candle is not None:
                        try:
                            _utp_pred = atomic_candle._check_exit_signal(symbol, side.lower())
                            if _utp_pred['verdict'] == 'HOLD_STRONG' and _utp_pred['urgency'] < 10:
                                _micro_tp_vetoed = True
                                logger.info(f"🔮🔬 {symbol}: MICRO-TP VETOED — breakout forming in ticks! urgency={_utp_pred['urgency']}")
                                # Reset tracker so it doesn't fire immediately next cycle
                                self._stale_profit_tracker[stale_key] = {
                                    'first_seen': now,
                                    'pnl_at_start': pnl_pct,
                                    'peak_at_start': peak_profit_pct
                                }
                        except Exception:
                            pass
                    
                    if not _micro_tp_vetoed:
                        logger.warning(f"🔬 {symbol}: MICRO-TP! Stale profit +{pnl_pct:.2f}% for {elapsed:.0f}s (drift {pnl_drift:.3f}%). Taking before reversal!")
                        del self._stale_profit_tracker[stale_key]
                        return {
                            'action': 'close',
                            'confidence': 0.88,
                            'reasoning': f"🔬 MICRO-TP: Profit +{pnl_pct:.2f}% stale for {elapsed:.0f}s. Not growing → will reverse. Lock it!",
                            'realtime': True
                        }
        else:
            if stale_key in self._stale_profit_tracker:
                del self._stale_profit_tracker[stale_key]
        
        # ═══════════════════════════════════════════════════════════════════
        # 🛡️ PROFIT PROTECTION - Protect profit but CONSIDER MOMENTUM!
        # If momentum is strongly WITH us, give more room for recovery
        # BUT if we keep touching breakeven, lose hope and exit!
        # WIDENED: Was 0.03% — too tight! Tiny peaks triggered protection
        # on ARB (-$0.38 had +0.03%), BIRB (-$0.07 had +0.04%), ELSA (-$0.05 had +0.26%)
        # Now only protects after meaningful profit is established
        # ═══════════════════════════════════════════════════════════════════
        MIN_PROFIT_TO_PROTECT = max(0.15 * tp_scale, 0.30)  # Protect profit from +0.30% peak (was 0.18% — barely above fees, triggered too often)
        
        # Track breakeven touches - if we keep bouncing off 0, eventually we'll break through
        if not hasattr(self, '_breakeven_touches'):
            self._breakeven_touches = {}
        
        position_key = f"{symbol}_{side}"
        
        # Trigger profit protection when profit drops to near-breakeven (0.02%) or below
        # WIDENED: Was 0.02% — only trigger when actually going negative
        # This lets small profits breathe instead of being killed at +0.02%
        BREAKEVEN_TRIGGER = 0.00  # Only protect when profit hits zero or below
        
        if peak_profit_pct >= MIN_PROFIT_TO_PROTECT and pnl_pct <= BREAKEVEN_TRIGGER:
            # === BREAKEVEN TOUCH TRACKING ===
            # Count how many times we've touched breakeven from profit
            BREAKEVEN_ZONE = 0.02  # Consider -0.02% to +0.02% as "breakeven zone"
            MAX_BREAKEVEN_TOUCHES = 2  # After 2 touches, lose hope
            
            if position_key not in self._breakeven_touches:
                self._breakeven_touches[position_key] = {
                    'count': 0,
                    'last_touch_time': None,
                    'was_in_profit': False
                }
            
            touch_data = self._breakeven_touches[position_key]
            now = datetime.now(timezone.utc)
            
            # If we were in profit (>0.05%) and now near breakeven, count it as a touch
            if not touch_data['was_in_profit'] and pnl_pct > 0.05:
                touch_data['was_in_profit'] = True
            elif touch_data['was_in_profit'] and abs(pnl_pct) <= BREAKEVEN_ZONE:
                # Just touched breakeven - count it if enough time passed since last touch
                min_touch_gap = 10  # seconds between touches
                if touch_data['last_touch_time'] is None or (now - touch_data['last_touch_time']).total_seconds() > min_touch_gap:
                    touch_data['count'] += 1
                    touch_data['last_touch_time'] = now
                    touch_data['was_in_profit'] = False  # Reset for next cycle
                    logger.warning(f"🔄 {symbol}: BREAKEVEN TOUCH #{touch_data['count']} - price keeps returning to 0%")
            
            # If too many breakeven touches, exit immediately - we've lost hope!
            if touch_data['count'] >= MAX_BREAKEVEN_TOUCHES:
                logger.warning(f"🚫 {symbol}: {touch_data['count']} BREAKEVEN TOUCHES - losing hope, exiting at {pnl_pct:+.2f}%!")
                # Clear the tracking
                del self._breakeven_touches[position_key]
                return {
                    'action': 'close',
                    'confidence': 0.95,
                    'reasoning': f"🚫 HOPE LOST: {touch_data['count']}x breakeven touches - price won't break through. Exit at {pnl_pct:+.2f}%",
                    'realtime': True
                }
            
            # Check momentum before exiting! Maybe price is just pulling back before continuing
            cache = self._momentum_cache.get(symbol)
            momentum_with_us = False
            momentum_reason = ""
            
            if cache:
                roc_5 = cache.get('roc_5', 0)
                roc_10 = cache.get('roc_10', 0)
                is_long = side.upper() == "LONG"
                
                if is_long:
                    # LONG: momentum with us if ROC positive
                    if roc_5 > 0.15 or roc_10 > 0.10:
                        momentum_with_us = True
                        momentum_reason = f"ROC5={roc_5:+.2f}%, ROC10={roc_10:+.2f}% - bullish momentum"
                else:
                    # SHORT: momentum with us if ROC negative
                    if roc_5 < -0.15 or roc_10 < -0.10:
                        momentum_with_us = True
                        momentum_reason = f"ROC5={roc_5:+.2f}%, ROC10={roc_10:+.2f}% - bearish momentum"
            
            # If momentum is WITH us and loss is small, give it a chance
            # BUT: We should exit at breakeven, NOT in negative territory!
            MOMENTUM_GRACE_LOSS = -0.02  # Much tighter - only 0.02% grace, then exit at ~breakeven
            
            if momentum_with_us and pnl_pct > MOMENTUM_GRACE_LOSS:
                logger.info(f"🛡️⏳ {symbol}: PROFIT PROTECTION DELAYED - Peak +{peak_profit_pct:.2f}% → {pnl_pct:+.2f}% but {momentum_reason}")
                # Don't exit yet - momentum is with us, might recover
            else:
                # No favorable momentum OR loss exceeds grace period - EXIT NOW AT BREAKEVEN!
                # The goal is to exit at ~breakeven, not deep in the red
                exit_note = "No momentum support" if not momentum_with_us else f"Loss {pnl_pct:.2f}% exceeded grace"
                logger.warning(f"🛡️ {symbol}: PROFIT PROTECTION! Peak +{peak_profit_pct:.2f}% → {pnl_pct:+.2f}% | {exit_note}")
                # Clear the tracking on exit
                if position_key in self._breakeven_touches:
                    del self._breakeven_touches[position_key]
                return {
                    'action': 'close',
                    'confidence': 0.97,
                    'reasoning': f"🛡️ PROFIT PROTECTION: Had +{peak_profit_pct:.2f}% profit, now {pnl_pct:+.2f}%. {exit_note}",
                    'realtime': True
                }
        else:
            # Position in profit or never had profit to protect - clear any stale tracking
            if position_key in self._breakeven_touches and pnl_pct > 0.10:
                # Reset touch count if we recovered strongly
                self._breakeven_touches[position_key]['was_in_profit'] = True
        
        # Check if we have cached data for this symbol
        cache = self._momentum_cache.get(symbol)
        if not cache:
            # No cache - let the 30-second check handle it
            return None
        
        # Check cache age
        age = (datetime.now(timezone.utc) - cache['updated_at']).total_seconds()
        if age > self._MOMENTUM_CACHE_MAX_AGE_SECONDS:
            # Cache too old - skip real-time check
            return None
        
        is_long = side.upper() == "LONG"
        
        # === 🚨 ABSOLUTE CEILING: DRAWDOWN FROM PEAK = MANDATORY EXIT ===
        # This is the highest priority check - NO EXCUSES!
        # But must ensure exit is ABOVE fee breakeven (0.15%)
        # The PROFIT FLOORS above handle absolute levels, this handles % drawdown
        if peak_profit_pct >= 0.40:  # Activate at 0.40%+ peak
            drawdown_pct_of_peak = ((peak_profit_pct - pnl_pct) / peak_profit_pct) * 100 if peak_profit_pct > 0 else 0
            # Tiered drawdown limits - bigger peaks get tighter protection
            if peak_profit_pct >= 2.0:
                max_drawdown = 30  # Big profit - allow 30% drawdown
            elif peak_profit_pct >= 1.0:
                max_drawdown = 35  # Medium profit - 35% drawdown
            else:
                max_drawdown = 40  # Small profit - 40% drawdown
            
            if drawdown_pct_of_peak >= max_drawdown:
                remaining_pct = 100 - drawdown_pct_of_peak
                # FEE GUARD: Don't exit if remaining profit is below fee breakeven
                if pnl_pct < FEE_GUARD_THRESHOLD and pnl_pct > 0:
                    logger.info(f"🛡️ {symbol}: CEILING blocked by FEE GUARD - would exit at +{pnl_pct:.2f}% (below {FEE_GUARD_THRESHOLD}%). Let loss cap handle if it goes negative.")
                else:
                    # 🔬 ADVANCED ANALYSIS LAST CHANCE — Is the drawdown truly dead or recovering?
                    # Sometimes candle lag makes it look like a 40% drawdown but ticks show V-recovery
                    _atomic_ceiling_save = False
                    if atomic_candle is not None and drawdown_pct_of_peak < max_drawdown + 15:
                        try:
                            ac_ceil = atomic_candle.get_analysis(symbol)
                            if ac_ceil and ac_ceil.tick_count >= 30:
                                c_vel = ac_ceil.velocity
                                c_vel_with = (is_long and c_vel > 0.006) or (not is_long and c_vel < -0.006)
                                c_acc_with = (is_long and ac_ceil.acceleration > 0.001) or (not is_long and ac_ceil.acceleration < -0.001)
                                # STRONG tick-level recovery: velocity + acceleration + clean direction
                                if c_vel_with and c_acc_with and ac_ceil.direction_consistency > 65 and ac_ceil.path_efficiency > 0.5:
                                    _atomic_ceiling_save = True
                                    logger.info(f"🔬🚨 {symbol}: Atomic SAVES from ceiling exit! v={c_vel:+.4f} WITH us, acc={ac_ceil.acceleration:+.5f}, dir={ac_ceil.direction_consistency:.0f}% — V-recovery in progress")
                        except Exception:
                            pass
                    
                    if not _atomic_ceiling_save:
                        logger.warning(f"🚨🚨 {symbol}: RT ABSOLUTE CEILING! Lost {drawdown_pct_of_peak:.0f}% of {peak_profit_pct:.2f}% peak (max {max_drawdown}%)! Saving {remaining_pct:.0f}%")
                        return {
                            'action': 'close',
                            'confidence': 0.95,
                            'reasoning': f"🚨🚨 ABSOLUTE CEILING: Lost {drawdown_pct_of_peak:.0f}% of {peak_profit_pct:.2f}% peak (limit {max_drawdown}%)! Saving remaining {pnl_pct:.2f}%",
                            'realtime': True
                        }
        
        # === 🔒 TRAILING STOP AFTER +0.80% PROFIT ===
        # UPDATED: Raised from 0.35% to 0.80% — Breakeven Shield handles the 0.20-0.80% zone
        # Only activates for positions that reached meaningful profit (>0.80%)
        TRAILING_LOCK_PEAK = 0.80    # Activate trailing at 0.80% peak (was 0.35% — too nervous)
        TRAILING_MIN_PROFIT = 0.40   # Lock at least 0.40% (was 0.18%)
        TRAILING_GRACE_SECONDS = int(20 * grace_mult)  # Dynamic grace (Notebook #2 §11)
        
        if peak_profit_pct >= TRAILING_LOCK_PEAK and pnl_pct < TRAILING_MIN_PROFIT and hold_seconds >= TRAILING_GRACE_SECONDS:
            # Check momentum but with STRICT criteria - only strong momentum delays exit
            cache = self._momentum_cache.get(symbol)
            strong_momentum_with_us = False
            
            if cache:
                roc_5 = cache.get('roc_5', 0)
                roc_10 = cache.get('roc_10', 0)
                is_long = side.upper() == "LONG"
                # STRICT: Both ROC5 AND ROC10 must be with us (was OR)
                strong_momentum_with_us = (roc_5 > 0.20 and roc_10 > 0.10) if is_long else (roc_5 < -0.20 and roc_10 < -0.10)
            
            # Only delay if STRONG momentum AND still have profit (not at zero)
            if strong_momentum_with_us and pnl_pct >= 0.10:
                # 🔬 ADVANCED ANALYSIS CHECK: Even with strong candle momentum, 
                # if tick-level shows velocity reversing, candle is lagging
                _atomic_overrides_trailing_delay = False
                if atomic_candle is not None:
                    try:
                        ac_trail = atomic_candle.get_analysis(symbol)
                        if ac_trail and ac_trail.tick_count >= 30:
                            t_vel = ac_trail.velocity
                            t_vel_against = (is_long and t_vel < -0.005) or (not is_long and t_vel > 0.005)
                            if t_vel_against and ac_trail.momentum_decay_rate < -40 and ac_trail.reversal_probability > 55:
                                _atomic_overrides_trailing_delay = True
                                logger.warning(f"🔬🔒 {symbol}: Atomic OVERRIDES trailing delay! v={t_vel:+.4f} AGAINST, decay={ac_trail.momentum_decay_rate:+.0f}%, revProb={ac_trail.reversal_probability:.0f}%")
                    except Exception:
                        pass
                
                if _atomic_overrides_trailing_delay:
                    logger.warning(f"🔒🔬 {symbol}: TRAILING STOP (atomic override)! Peak +{peak_profit_pct:.2f}% → Now {pnl_pct:+.2f}% | Candle momentum lagging, ticks show reversal")
                    return {
                        'action': 'close',
                        'confidence': 0.93,
                        'reasoning': f"🔒🔬 TRAILING STOP: Peak +{peak_profit_pct:.2f}% fell to {pnl_pct:+.2f}% — atomic override (candle momentum lagging)",
                        'realtime': True
                    }
                else:
                    logger.info(f"🔒⏳ {symbol}: RT TRAILING delayed - Peak +{peak_profit_pct:.2f}% → {pnl_pct:+.2f}% but STRONG momentum with us")
            elif pnl_pct < FEE_GUARD_THRESHOLD and pnl_pct > 0:
                # FEE GUARD: Don't exit at a fee-losing level
                logger.info(f"🛡️ {symbol}: TRAILING blocked by FEE GUARD - would exit at +{pnl_pct:.2f}% (below {FEE_GUARD_THRESHOLD}%). Waiting for recovery or loss cap.")
            else:
                mom_note = "no strong momentum" if not strong_momentum_with_us else f"profit {pnl_pct:.2f}% too low"
                logger.warning(f"🔒 {symbol}: TRAILING STOP! Peak +{peak_profit_pct:.2f}% → Now {pnl_pct:+.2f}% | {mom_note}")
                return {
                    'action': 'close',
                    'confidence': 0.93,
                    'reasoning': f"🔒 TRAILING STOP: Peak +{peak_profit_pct:.2f}% fell to {pnl_pct:+.2f}% ({mom_note})",
                    'realtime': True
                }
        
        # NOTE: BREAK-EVEN STOP removed - PROFIT PROTECTION above now handles
        # all cases where peak >= 0.03% and pnl falls to 0 or below
        
        # === ⏰ TIME-BASED PROFIT LOCK ===
        # After 5+ minutes in profit, tighten protection significantly
        # Older profitable positions should be protected more aggressively
        TIME_PROFIT_LOCK_SECONDS = 300  # 5 minutes
        TIME_PROFIT_LOCK_MIN_PNL = 0.15  # Minimum profit to activate
        TIME_PROFIT_LOCK_THRESHOLD = 0.05  # Exit if profit falls below this after 5 mins
        
        if hold_seconds >= TIME_PROFIT_LOCK_SECONDS and peak_profit_pct >= TIME_PROFIT_LOCK_MIN_PNL:
            if pnl_pct < TIME_PROFIT_LOCK_THRESHOLD:
                # 🔮 ATOMIC: Don't exit if ticks show sudden recovery forming
                _time_lock_vetoed = False
                if atomic_candle is not None:
                    try:
                        _tl_pred = atomic_candle._check_exit_signal(symbol, side.lower())
                        if _tl_pred['verdict'] == 'HOLD_STRONG' and _tl_pred['urgency'] < 15:
                            _time_lock_vetoed = True
                            logger.info(f"🔮⏰ {symbol}: Time profit lock VETOED by atomic — recovery forming! urgency={_tl_pred['urgency']}")
                    except Exception:
                        pass
                
                if not _time_lock_vetoed:
                    logger.warning(f"⏰ {symbol}: TIME PROFIT LOCK! After {hold_seconds/60:.1f}min, peak +{peak_profit_pct:.2f}% → {pnl_pct:+.2f}%")
                    return {
                        'action': 'close',
                        'confidence': 0.91,
                        'reasoning': f"⏰ TIME PROFIT LOCK: {hold_seconds/60:.1f}min holding, peak +{peak_profit_pct:.2f}% falling to {pnl_pct:+.2f}%",
                        'realtime': True
                    }
        
        # === 📉 MOMENTUM FADE EXIT - FAST DETECTION ===
        # UPDATED: Sensible Reversal handles 0.20-0.80% zone now.
        # This only fires OUTSIDE that zone (< 0.20% or > 0.80%) as a fallback.
        # Thresholds RAISED: Old 0.08/0.12 were too nervous and killed POWER.
        MOMENTUM_FADE_GRACE_SECONDS = int(15 * grace_mult)
        cache = self._momentum_cache.get(symbol)
        _in_sensible_zone_mf = 0.20 <= pnl_pct <= 0.80
        if cache and not _in_sensible_zone_mf and pnl_pct > 0.05 and peak_profit_pct > 0.10 and hold_seconds >= MOMENTUM_FADE_GRACE_SECONDS:
            roc_5 = cache.get('roc_5', 0)
            roc_10 = cache.get('roc_10', 0)
            is_long = side.upper() == "LONG"
            
            # RAISED thresholds — only trigger on STRONG momentum shifts
            if peak_profit_pct >= 1.0:
                fade_threshold_5 = 0.20   # Tighter at high profit
                fade_threshold_10 = 0.15
            elif peak_profit_pct >= 0.8:
                fade_threshold_5 = 0.30
                fade_threshold_10 = 0.20
            else:
                fade_threshold_5 = 0.40   # Standard — only strong reversals
                fade_threshold_10 = 0.25
            
            # Check for momentum against position
            momentum_fade_detected = False
            if is_long:
                if roc_5 < -fade_threshold_5 and roc_10 < -fade_threshold_10:
                    momentum_fade_detected = True
                    fade_reason = f"ROC5={roc_5:+.2f}%, ROC10={roc_10:+.2f}% falling (thresholds: {fade_threshold_5}/{fade_threshold_10})"
                elif roc_5 < -(fade_threshold_5 * 2):
                    momentum_fade_detected = True
                    fade_reason = f"ROC5={roc_5:+.2f}% SHARP reversal"
            else:
                if roc_5 > fade_threshold_5 and roc_10 > fade_threshold_10:
                    momentum_fade_detected = True
                    fade_reason = f"ROC5={roc_5:+.2f}%, ROC10={roc_10:+.2f}% rising (thresholds: {fade_threshold_5}/{fade_threshold_10})"
                elif roc_5 > (fade_threshold_5 * 2):
                    momentum_fade_detected = True
                    fade_reason = f"ROC5={roc_5:+.2f}% SHARP reversal"
            
            if momentum_fade_detected:
                # 🔬 ADVANCED ANALYSIS CONFIRMATION — Don't exit on candle lag alone
                # Candle ROC can be 15-60s behind reality. If analysis shows
                # velocity is actually WITH us (just a temporary pullback), hold.
                _atomic_saves_fade = False
                if atomic_candle is not None:
                    try:
                        ac_fade = atomic_candle.get_analysis(symbol)
                        if ac_fade and ac_fade.tick_count >= 30:
                            f_vel = ac_fade.velocity
                            f_vel_with = (is_long and f_vel > 0.004) or (not is_long and f_vel < -0.004)
                            f_acc_with = (is_long and ac_fade.acceleration > 0) or (not is_long and ac_fade.acceleration < 0)
                            # Ticks show strong recovery direction — candle ROC is lagging
                            if f_vel_with and f_acc_with and ac_fade.path_efficiency > 0.45 and ac_fade.direction_consistency > 55:
                                _atomic_saves_fade = True
                                logger.info(f"🔬 {symbol}: Atomic SAVES from momentum fade exit! v={f_vel:+.4f} WITH us, acc={ac_fade.acceleration:+.5f}, eff={ac_fade.path_efficiency:.2f} — candle ROC lagging")
                    except Exception:
                        pass
                
                if not _atomic_saves_fade:
                    logger.warning(f"📉 {symbol}: MOMENTUM FADE! Profit +{pnl_pct:.2f}% (peak +{peak_profit_pct:.2f}%) at risk - {fade_reason}")
                    return {
                        'action': 'close',
                        'confidence': 0.92,
                        'reasoning': f"📉 MOMENTUM FADE: Profit +{pnl_pct:.2f}% at risk, {fade_reason}. ZERO LOSS strategy.",
                        'realtime': True
                    }
        
        # === 🔥 SMART LOSS CUT: Consider Recovery Before Cutting ===
        # Track deepest loss and check if we're recovering before cutting
        if not hasattr(self, '_loss_tracking'):
            self._loss_tracking = {}
        
        loss_key = f"{symbol}_{side}"
        
        # Track deepest loss for this position
        if pnl_pct < 0:
            if loss_key not in self._loss_tracking:
                self._loss_tracking[loss_key] = {
                    'deepest_loss': pnl_pct,
                    'last_pnl': pnl_pct,
                    'recovering': False,
                    'recovery_count': 0
                }
            else:
                prev = self._loss_tracking[loss_key]
                # Update deepest loss
                if pnl_pct < prev['deepest_loss']:
                    prev['deepest_loss'] = pnl_pct
                    prev['recovering'] = False
                    prev['recovery_count'] = 0
                # Check if recovering (pnl improving)
                elif pnl_pct > prev['last_pnl'] + 0.02:  # Improved by 0.02%+
                    prev['recovering'] = True
                    prev['recovery_count'] = prev.get('recovery_count', 0) + 1
                
                # 🔬 ADVANCED ANALYSIS EARLY RECOVERY DETECTION
                # PnL hasn't improved by 0.02% yet, but Analysis sees 
                # velocity has turned in our favor — recovery is starting at 
                # tick level before candle PnL catches up
                if not prev['recovering'] and atomic_candle is not None:
                    try:
                        ac_rec = atomic_candle.get_analysis(symbol)
                        if ac_rec and ac_rec.tick_count >= 20:
                            r_vel = ac_rec.velocity
                            r_vel_with = (is_long and r_vel > 0.003) or (not is_long and r_vel < -0.003)
                            r_acc_with = (is_long and ac_rec.acceleration > 0) or (not is_long and ac_rec.acceleration < 0)
                            if r_vel_with and r_acc_with and ac_rec.direction_consistency > 55:
                                prev['recovering'] = True
                                prev['recovery_count'] = max(prev.get('recovery_count', 0), 1)
                                logger.info(f"🔬🔄 {symbol}: Atomic EARLY recovery detected! v={r_vel:+.4f} WITH us, acc={ac_rec.acceleration:+.5f}, dir={ac_rec.direction_consistency:.0f}% — PnL hasn't caught up yet")
                    except Exception:
                        pass
                
                prev['last_pnl'] = pnl_pct
        else:
            # Position back in profit - clear tracking
            if loss_key in self._loss_tracking:
                del self._loss_tracking[loss_key]
        
        # Get loss tracking data
        loss_data = self._loss_tracking.get(loss_key, {})
        is_recovering = loss_data.get('recovering', False)
        recovery_count = loss_data.get('recovery_count', 0)
        deepest_loss = loss_data.get('deepest_loss', pnl_pct)
        
        # Check momentum — HYBRID: candle ROC + tick velocity
        roc_5 = cache.get('roc_5', 0)
        roc_10 = cache.get('roc_10', 0)
        
        # Is momentum WITH us (might recover) or AGAINST us?
        # Require STRONG signals for recovery allowance (zero-loss strategy)
        # FIX: Also check cache freshness — stale 15-min bar ROC misleads when real-time price drops
        momentum_with_us = False
        _cache_age = (datetime.now(timezone.utc) - cache.get('updated_at', datetime.now(timezone.utc))).total_seconds() if cache.get('updated_at') else 999
        if _cache_age <= 45:  # Only trust momentum if cache is fresh (< 45s old)
            if is_long:
                momentum_with_us = roc_5 > 0.20 and roc_10 > 0.10  # Both must agree
            else:
                momentum_with_us = roc_5 < -0.20 and roc_10 < -0.10  # Both must agree
        else:
            logger.info(f"⚠️ {symbol}: Momentum cache is {_cache_age:.0f}s old — NOT trusting ROC for loss cut decision")
        
        # 🔬 ADVANCED ANALYSIS RECOVERY INTELLIGENCE — see what candles can't
        # When candle ROC says "momentum with us" but Analysis sees the 
        # real tick-level story, override the lagging candle indicator.
        # When cache is stale, Analysis becomes the PRIMARY momentum source.
        _atomic_recovery_override = False
        _atomic_recovery_hope = False
        if atomic_candle is not None:
            try:
                ac_analysis = atomic_candle.get_analysis(symbol)
                if ac_analysis and ac_analysis.tick_count >= 30:
                    vel = ac_analysis.velocity
                    acc = ac_analysis.acceleration
                    eff = ac_analysis.path_efficiency
                    decay = ac_analysis.momentum_decay_rate
                    dir_cons = ac_analysis.direction_consistency
                    
                    vel_with_us = (is_long and vel > 0.003) or (not is_long and vel < -0.003)
                    vel_against = (is_long and vel < -0.003) or (not is_long and vel > 0.003)
                    acc_with_us = (is_long and acc > 0) or (not is_long and acc < 0)
                    
                    # OVERRIDE: Candle says "momentum with us" but ticks show otherwise
                    if momentum_with_us and vel_against and decay < -30 and eff > 0.4:
                        _atomic_recovery_override = True
                        momentum_with_us = False
                        logger.warning(f"🔬 {symbol}: Atomic OVERRIDES recovery momentum! v={vel:+.4f} AGAINST, decay={decay:+.0f}%, eff={eff:.2f} — candle ROC is lagging")
                    
                    # HOPE: No candle momentum (or stale cache) but ticks show real recovery
                    if not momentum_with_us and vel_with_us and acc_with_us and dir_cons > 55 and eff > 0.4:
                        _atomic_recovery_hope = True
                        momentum_with_us = True
                        logger.info(f"🔬 {symbol}: Atomic shows REAL recovery! v={vel:+.4f} WITH us, acc={acc:+.5f}, dir={dir_cons:.0f}%, eff={eff:.2f}")
                    
                    # STALE CACHE RESCUE: When candle cache is too old, atomic is the only truth
                    if _cache_age > 45 and vel_with_us and dir_cons > 60:
                        momentum_with_us = True
                        logger.info(f"🔬 {symbol}: Stale cache ({_cache_age:.0f}s) but atomic shows momentum WITH us (v={vel:+.4f}, dir={dir_cons:.0f}%)")
            except Exception:
                pass  # Never let atomic analysis break loss management
        
        # === DECISION LOGIC ===
        _loss_mode = get_threshold_mode()
        SOFT_LOSS_THRESHOLD = -0.25 if _loss_mode == 'testing' else (-0.12 if _loss_mode == 'loose' else -0.12)  # Start considering exit (loose was -0.18 → too wide, losses > wins)
        HARD_LOSS_THRESHOLD = -0.40 if _loss_mode == 'testing' else (-0.20 if _loss_mode == 'loose' else -0.20)  # Must exit unless strong recovery (loose was -0.30 → losses 1.7x bigger than wins)
        ABSOLUTE_MAX_LOSS = -0.50 if _loss_mode == 'testing' else (-0.35 if _loss_mode == 'loose' else -0.30)    # EXIT NO MATTER WHAT (loose was -0.45 → too generous)
        _LOSS_GRACE_PERIOD = 30  # FIX: Always 30s — 45s let losses deepen too much before acting
        
        if pnl_pct <= SOFT_LOSS_THRESHOLD:
            # GRACE PERIOD: Don't cut brand new positions — spread noise looks like a loss
            if hold_seconds < _LOSS_GRACE_PERIOD and pnl_pct > ABSOLUTE_MAX_LOSS:
                logger.info(f"⏳ {symbol}: Smart loss {pnl_pct:.2f}% but only {hold_seconds:.0f}s old — GRACE (need {_LOSS_GRACE_PERIOD}s). HOLD.")
                return None
            # At -0.35%: Only exit if NOT recovering AND momentum against
            if pnl_pct > HARD_LOSS_THRESHOLD:
                if not is_recovering and not momentum_with_us:
                    logger.warning(f"🔥 {symbol}: LOSS CUT at {pnl_pct:.2f}%! No recovery (ROC5={roc_5:+.2f}%)")
                    if loss_key in self._loss_tracking:
                        del self._loss_tracking[loss_key]
                    return {
                        'action': 'close',
                        'confidence': 0.90,
                        'reasoning': f"🔥 LOSS CUT: {pnl_pct:.2f}% | No recovery, momentum against (ROC5={roc_5:+.2f}%)",
                        'realtime': True
                    }
                elif is_recovering:
                    logger.info(f"🔄 {symbol}: At {pnl_pct:.2f}% but RECOVERING (deepest={deepest_loss:.2f}%, count={recovery_count})")
            
            # At -0.50%: Exit unless actively recovering with strong momentum
            elif pnl_pct > ABSOLUTE_MAX_LOSS:
                strong_recovery = is_recovering and recovery_count >= 2 and momentum_with_us
                if not strong_recovery:
                    logger.warning(f"🔥🔥 {symbol}: HARD LOSS CUT at {pnl_pct:.2f}%! Recovery weak (rec={is_recovering}, mom={momentum_with_us})")
                    if loss_key in self._loss_tracking:
                        del self._loss_tracking[loss_key]
                    return {
                        'action': 'close',
                        'confidence': 0.93,
                        'reasoning': f"🔥🔥 HARD LOSS: {pnl_pct:.2f}% | Recovery too weak to continue",
                        'realtime': True
                    }
                else:
                    logger.info(f"🔄 {symbol}: At {pnl_pct:.2f}% but STRONG RECOVERY (count={recovery_count}, ROC5={roc_5:+.2f}%)")
            
            # At -0.70%: ABSOLUTE MAX - EXIT NO EXCEPTIONS
            else:
                logger.warning(f"🔥🔥🔥 {symbol}: ABSOLUTE MAX LOSS at {pnl_pct:.2f}%! No more patience!")
                if loss_key in self._loss_tracking:
                    del self._loss_tracking[loss_key]
                return {
                    'action': 'close',
                    'confidence': 0.98,
                    'reasoning': f"🔥🔥🔥 ABSOLUTE MAX: {pnl_pct:.2f}% exceeds -0.70% limit!",
                    'realtime': True
                }
        
        # === QUICK EXIT ZONE: 0.00% to 0.05% profit in first 2 minutes ===
        # Exit at breakeven/tiny profit BEFORE it goes negative!
        QUICK_EXIT_WINDOW = 120  # 2 minutes
        QUICK_EXIT_GRACE = 30  # Don't quick-exit in first 30 seconds
        if hold_seconds > QUICK_EXIT_GRACE and hold_seconds <= QUICK_EXIT_WINDOW and pnl_pct >= 0 and pnl_pct <= 0.05:
            against_count = 0
            reasons = []
            
            roc_5 = cache['roc_5']
            roc_10 = cache['roc_10']
            current_rsi = cache['rsi']
            ema_trend = cache['ema_trend']
            momentum_hope = cache['momentum_hope']
            accelerating_against = cache['accelerating_against']
            
            if is_long:
                if roc_5 < -0.2:  # Momentum against
                    against_count += 1
                    reasons.append(f"ROC5={roc_5:+.2f}%")
                if roc_10 < -0.1:  # Trend against
                    against_count += 1
                    reasons.append(f"ROC10={roc_10:+.2f}%")
                if accelerating_against:
                    against_count += 1
                    reasons.append("Accelerating↓")
                if current_rsi < 45:  # RSI weak
                    against_count += 1
                    reasons.append(f"RSI={current_rsi:.0f}")
                if not ema_trend:
                    against_count += 1
                    reasons.append("EMA↓")
            else:  # SHORT
                if roc_5 > 0.2:  # Momentum against
                    against_count += 1
                    reasons.append(f"ROC5={roc_5:+.2f}%")
                if roc_10 > 0.1:  # Trend against
                    against_count += 1
                    reasons.append(f"ROC10={roc_10:+.2f}%")
                if accelerating_against:
                    against_count += 1
                    reasons.append("Accelerating↑")
                if current_rsi > 55:  # RSI strong
                    against_count += 1
                    reasons.append(f"RSI={current_rsi:.0f}")
                if ema_trend:
                    against_count += 1
                    reasons.append("EMA↑")
            
            if momentum_hope == 0:
                against_count += 1
                reasons.append("NoHope")
            
            # 🔬 ADVANCED ANALYSIS QUICK EXIT INTELLIGENCE — tick-level signals
            # Analysis detects: micro-velocity against us,
            # low path efficiency (noise), high reversal probability
            if atomic_candle is not None:
                try:
                    ac_qe = atomic_candle.get_analysis(symbol)
                    if ac_qe and ac_qe.tick_count >= 20:
                        qe_vel = ac_qe.velocity
                        qe_vel_against = (is_long and qe_vel < -0.002) or (not is_long and qe_vel > 0.002)
                        if qe_vel_against:
                            against_count += 1
                            reasons.append(f"🔬v={qe_vel:+.4f}")
                        if ac_qe.path_efficiency < 0.25:
                            against_count += 1
                            reasons.append(f"🔬eff={ac_qe.path_efficiency:.2f}")
                        if ac_qe.reversal_probability > 55:
                            against_count += 1
                            reasons.append(f"🔬revP={ac_qe.reversal_probability:.0f}%")
                except Exception:
                    pass
            
            # Exit if 3+ indicators against (now includes up to 9 signals with atomic)
            if against_count >= 3:
                reason_str = ', '.join(reasons[:4])
                logger.warning(f"⚡ {symbol}: RT QUICK EXIT at +{pnl_pct:.2f}%! {against_count}/6 against | {reason_str}")
                return {
                    'action': 'close',
                    'confidence': 0.88,
                    'reasoning': f"⚡ RT QUICK EXIT: {against_count}/6 against ({reason_str})",
                    'realtime': True
                }
        
        # === 🎣 SMART FISHERMAN v3: Lightning fast + 3-strike breakeven rule ===
        # CRASH PROTECTION with INTELLIGENCE — only 2 fast checks:
        #   1. ROC_5 momentum: WITH us (+) or AGAINST us (-)
        #   2. Volume: High vol bounce = real, low vol = dead cat
        # If both WITH us → let it run. Otherwise → catch at breakeven.
        # 3-STRIKE RULE: If breakeven bounce happens 3 times → EXIT, no questions asked.
        FISHERMAN_WINDOW = 240
        _FISHERMAN_BASE_LOSS = -0.15
        # FIX: Scale fisherman bail by volatility — volatile pairs (PIPPIN, memecoins) need wider threshold
        # tp_scale > 1 means more volatile, so allow deeper fishing
        FISHERMAN_MAX_LOSS = _FISHERMAN_BASE_LOSS * max(1.0, tp_scale)
        FISHERMAN_GRACE_PERIOD = 30
        
        if not hasattr(self, '_fisherman_tracking'):
            self._fisherman_tracking = {}
        if not hasattr(self, '_fisherman_strikes'):
            self._fisherman_strikes = {}  # {fish_key: bounce_count}
        
        if hold_seconds >= FISHERMAN_GRACE_PERIOD and hold_seconds <= FISHERMAN_WINDOW:
            fish_key = f"{symbol}_{side}"
            
            if pnl_pct < 0 and pnl_pct >= FISHERMAN_MAX_LOSS:
                if fish_key not in self._fisherman_tracking:
                    self._fisherman_tracking[fish_key] = {
                        'lowest_pnl': pnl_pct,
                        'started_fishing': True,
                        'started_at': hold_seconds
                    }
                    logger.info(f"🎣 {symbol}: FISHING at {pnl_pct:.2f}%")
                else:
                    if pnl_pct < self._fisherman_tracking[fish_key]['lowest_pnl']:
                        self._fisherman_tracking[fish_key]['lowest_pnl'] = pnl_pct
            
            if fish_key in self._fisherman_tracking and self._fisherman_tracking[fish_key]['started_fishing']:
                lowest = self._fisherman_tracking[fish_key]['lowest_pnl']
                fish_time = hold_seconds - self._fisherman_tracking[fish_key].get('started_at', 0)
                
                # Breakeven zone: 0.00% to 0.02%
                if pnl_pct >= 0 and pnl_pct <= 0.02:
                    strikes = self._fisherman_strikes.get(fish_key, 0)
                    
                    # 3-STRIKE RULE: 3rd bounce at breakeven → EXIT unconditionally
                    if strikes >= 2:
                        logger.warning(f"🎣⚾ {symbol}: STRIKE 3! Bounced to breakeven {strikes + 1} times — OUT! (from {lowest:.2f}%)")
                        del self._fisherman_tracking[fish_key]
                        self._fisherman_strikes[fish_key] = 0
                        return {
                            'action': 'close',
                            'confidence': 0.95,
                            'reasoning': f"🎣⚾ FISHERMAN 3-STRIKE: Bounced to breakeven {strikes + 1}x — this trade is going nowhere!",
                            'realtime': True
                        }
                    
                    # LIGHTNING CHECK: ROC_5 + volume + analysis tick-level truth
                    _let_run = False
                    _fish_candle_roc = cache.get('roc_5', 0) if cache else 0
                    _fish_vol = cache.get('volume_ratio', 1.0) if cache else 1.0
                    if cache:
                        # Momentum WITH us AND decent volume → let it run
                        if is_long:
                            _let_run = _fish_candle_roc > 0.10 and _fish_vol > 0.8
                        else:
                            _let_run = _fish_candle_roc < -0.10 and _fish_vol > 0.8
                    
                    # 🔬 ADVANCED ANALYSIS FISHERMAN INTELLIGENCE
                    # Candle ROC might say "let run" but tick-level velocity reveals
                    # dead-cat bounces (high reversal prob, low path efficiency).
                    # Also: even if candle cache is empty/stale, atomic can save us.
                    _fish_atomic_info = ""
                    if atomic_candle is not None:
                        try:
                            ac_fish = atomic_candle.get_analysis(symbol)
                            if ac_fish and ac_fish.tick_count >= 20:
                                f_vel = ac_fish.velocity
                                f_vel_with = (is_long and f_vel > 0.002) or (not is_long and f_vel < -0.002)
                                f_vel_against = (is_long and f_vel < -0.002) or (not is_long and f_vel > 0.002)
                                
                                # OVERRIDE LET-RUN: Candle says go but ticks show dead cat
                                if _let_run and f_vel_against and ac_fish.reversal_probability > 55:
                                    _let_run = False
                                    _fish_atomic_info = f" | 🔬ATOMIC BLOCKED: v={f_vel:+.4f} AGAINST, revProb={ac_fish.reversal_probability:.0f}%"
                                elif _let_run and ac_fish.path_efficiency < 0.25 and ac_fish.swing_count >= 5:
                                    _let_run = False
                                    _fish_atomic_info = f" | 🔬ATOMIC BLOCKED: choppy eff={ac_fish.path_efficiency:.2f}, swings={ac_fish.swing_count}"
                                
                                # RESCUE: No candle data but atomic shows real momentum
                                if not _let_run and not cache and f_vel_with and ac_fish.direction_consistency > 60 and ac_fish.path_efficiency > 0.5:
                                    _let_run = True
                                    _fish_atomic_info = f" | 🔬ATOMIC RESCUED: v={f_vel:+.4f} WITH, dir={ac_fish.direction_consistency:.0f}%, eff={ac_fish.path_efficiency:.2f}"
                        except Exception:
                            pass
                    
                    if _let_run:
                        # Count as a strike but let it try
                        self._fisherman_strikes[fish_key] = strikes + 1
                        # Reset fishing tracker so next dip starts fresh
                        del self._fisherman_tracking[fish_key]
                        logger.info(f"🎣🚀 {symbol}: LET RUN (strike {strikes + 1}/3) ROC5={_fish_candle_roc:+.2f}% vol={_fish_vol:.1f}x{_fish_atomic_info}")
                    else:
                        logger.warning(f"🎣 {symbol}: CAUGHT! {lowest:.2f}% → {pnl_pct:.2f}% (ROC5={_fish_candle_roc:+.2f}%{_fish_atomic_info})")
                        del self._fisherman_tracking[fish_key]
                        self._fisherman_strikes.pop(fish_key, None)
                        return {
                            'action': 'close',
                            'confidence': 0.92,
                            'reasoning': f"🎣 FISHERMAN: Caught bounce {lowest:.2f}% → {pnl_pct:.2f}%",
                            'realtime': True
                        }
                
                # Partial recovery after 60s
                if fish_time > 60 and pnl_pct > lowest + 0.10 and pnl_pct >= -0.05:
                    logger.warning(f"🎣 {symbol}: PARTIAL CATCH {lowest:.2f}% → {pnl_pct:.2f}%")
                    del self._fisherman_tracking[fish_key]
                    self._fisherman_strikes.pop(fish_key, None)
                    return {
                        'action': 'close',
                        'confidence': 0.88,
                        'reasoning': f"🎣 FISHERMAN PARTIAL: {lowest:.2f}% → {pnl_pct:.2f}%",
                        'realtime': True
                    }
            
            # If loss goes too deep, stop fishing
            if fish_key in self._fisherman_tracking and pnl_pct < FISHERMAN_MAX_LOSS:
                logger.info(f"🎣 {symbol}: Fish got away! Loss {pnl_pct:.2f}% too deep (< {FISHERMAN_MAX_LOSS}%)")
                del self._fisherman_tracking[fish_key]
        
        # === PROFIT-TO-LOSS QUICK EXIT ===
        # Extended window and more aggressive exit when profit turns to loss
        # FIX: Require minimum 30s hold time — price noise on new trades shouldn't trigger
        PROFIT_TO_LOSS_WINDOW = 90  # 90 seconds
        PROFIT_TO_LOSS_MIN_AGE = 30  # Don't fire in first 30 seconds
        if pnl_pct < 0 and peak_profit_pct > 0 and hold_seconds >= PROFIT_TO_LOSS_MIN_AGE:
            # 🔬 HYBRID MOMENTUM CHECK: candle ROC + analysis velocity
            _ptl_momentum_against = False
            _ptl_source = "candle"
            if cache:
                roc_5 = cache['roc_5']
                _ptl_momentum_against = (is_long and roc_5 < -0.15) or (not is_long and roc_5 > 0.15)
                _ptl_source = f"ROC5={roc_5:+.2f}%"
            
            # analysis: override or provide data when cache is missing/stale
            if atomic_candle is not None:
                try:
                    ac_ptl = atomic_candle.get_analysis(symbol)
                    if ac_ptl and ac_ptl.tick_count >= 20:
                        ptl_vel = ac_ptl.velocity
                        ptl_vel_against = (is_long and ptl_vel < -0.003) or (not is_long and ptl_vel > 0.003)
                        
                        # RESCUE: No cache but atomic sees momentum against us
                        if not cache and ptl_vel_against and ac_ptl.reversal_probability > 50:
                            _ptl_momentum_against = True
                            _ptl_source = f"🔬v={ptl_vel:+.4f}, revP={ac_ptl.reversal_probability:.0f}%"
                        
                        # CONFIRM: Candle says against AND atomic confirms → stronger signal
                        elif _ptl_momentum_against and ptl_vel_against:
                            _ptl_source += f" + 🔬v={ptl_vel:+.4f}"
                        
                        # SAVE: Candle says against but atomic shows recovery → hold
                        elif _ptl_momentum_against and not ptl_vel_against:
                            ptl_vel_with = (is_long and ptl_vel > 0.004) or (not is_long and ptl_vel < -0.004)
                            if ptl_vel_with and ac_ptl.direction_consistency > 60 and ac_ptl.path_efficiency > 0.45:
                                _ptl_momentum_against = False
                                logger.info(f"🔬 {symbol}: Atomic SAVES profit-to-loss exit! v={ptl_vel:+.4f} WITH us, dir={ac_ptl.direction_consistency:.0f}% — candle ROC lagging")
                except Exception:
                    pass
            
            # If we HAD profit and now in loss with adverse momentum
            if _ptl_momentum_against:
                # Small peak that turned to loss - exit IMMEDIATELY (zero-loss: don't wait for -0.10%)
                if peak_profit_pct < 0.4 and pnl_pct < -0.03:
                    logger.warning(f"⚡ {symbol}: RT PROFIT→LOSS EXIT! Peak +{peak_profit_pct:.2f}% → Now {pnl_pct:.2f}%")
                    return {
                        'action': 'close',
                        'confidence': 0.90,
                        'reasoning': f"⚡ RT PROFIT→LOSS: Peak +{peak_profit_pct:.2f}% → {pnl_pct:.2f}%",
                        'realtime': True
                    }
                
                # Any peak that lost more than 50% and now in loss
                if peak_profit_pct >= 0.4 and pnl_pct < 0:
                    pct_lost = ((peak_profit_pct - pnl_pct) / peak_profit_pct) * 100 if peak_profit_pct > 0 else 0
                    if pct_lost >= 100:  # Lost all profit and now in red
                        logger.warning(f"⚡ {symbol}: PROFIT WIPED! Peak +{peak_profit_pct:.2f}% → Now {pnl_pct:.2f}% (lost {pct_lost:.0f}%)")
                        return {
                            'action': 'close',
                            'confidence': 0.90,
                            'reasoning': f"⚡ PROFIT WIPED: +{peak_profit_pct:.2f}% → {pnl_pct:.2f}% (lost {pct_lost:.0f}%)",
                            'realtime': True
                        }
        
        return None
    
    def _load_trade_history(self):
        """Load trade history from persistent file for AI context."""
        try:
            trade_file = Path("trade_history.json")
            if trade_file.exists():
                with open(trade_file, 'r') as f:
                    data = json.load(f)
                
                # Handle both formats: list or dict with trades key
                trades = data if isinstance(data, list) else data.get('trades', data.get('recent_trades', []))
                
                # Convert to our format and keep last 25 for direction analysis
                for t in trades[-25:]:
                    pnl = t.get('pnl', 0)
                    is_win = pnl > 0
                    # Handle both symbol formats (BTCUSDT and BTCUSDT:USDT)
                    symbol = t.get('symbol', '').replace(':USDT', '')
                    # Field is 'side' in trade_history.json, not 'direction'
                    side = t.get('side', t.get('direction', '')).upper()
                    
                    self.recent_trades.append({
                        "result": "win" if is_win else "loss",
                        "pnl": pnl,
                        "time": t.get('close_time', t.get('closed_at', t.get('time', ''))),
                        "symbol": symbol,
                        "side": side
                    })
                    
                    if is_win:
                        self.total_wins += 1
                    else:
                        self.total_losses += 1
                
                # === POPULATE SIDE PERFORMANCE TRACKER ===
                self._update_side_stats_from_history()
                
                # Log direction breakdown
                longs = [t for t in self.recent_trades if t.get('side') == 'LONG']
                shorts = [t for t in self.recent_trades if t.get('side') == 'SHORT']
                long_wins = sum(1 for t in longs if t.get('result') == 'win')
                short_wins = sum(1 for t in shorts if t.get('result') == 'win')
                
                logger.info(f"📊 Loaded {len(self.recent_trades)} trades for AI context: LONG {long_wins}W/{len(longs)-long_wins}L, SHORT {short_wins}W/{len(shorts)-short_wins}L")
        except Exception as e:
            logger.warning(f"Could not load trade history for AI: {e}")
    
    def _update_side_stats_from_history(self):
        """Rebuild side performance stats from recent_trades list."""
        self._side_stats = {
            'LONG': {'wins': 0, 'losses': 0, 'recent_pnl': []},
            'SHORT': {'wins': 0, 'losses': 0, 'recent_pnl': []}
        }
        for t in self.recent_trades:
            side = t.get('side', '').upper()
            if side not in ('LONG', 'SHORT'):
                continue
            pnl = t.get('pnl', 0)
            if t.get('result') == 'win':
                self._side_stats[side]['wins'] += 1
            else:
                self._side_stats[side]['losses'] += 1
            self._side_stats[side]['recent_pnl'].append(pnl)
            # Keep only last N per side
            if len(self._side_stats[side]['recent_pnl']) > self.SIDE_BLOCK_LOOKBACK:
                self._side_stats[side]['recent_pnl'] = self._side_stats[side]['recent_pnl'][-self.SIDE_BLOCK_LOOKBACK:]
        
        # Evaluate blocking for each side
        for side in ('LONG', 'SHORT'):
            stats = self._side_stats[side]
            total = stats['wins'] + stats['losses']
            # Only block after enough data AND lookback window shows heavy losses
            recent_total = len(stats['recent_pnl'])
            if recent_total >= self.SIDE_BLOCK_MIN_TRADES:
                recent_losses = sum(1 for p in stats['recent_pnl'] if p <= 0)
                loss_rate = recent_losses / recent_total
                if loss_rate >= self.SIDE_BLOCK_MAX_LOSS_RATE:
                    if not self._side_blocked[side]:
                        logger.warning(f"🚫 SIDE BLOCKER: {side} BLOCKED — {recent_losses}/{recent_total} recent trades are losses ({loss_rate:.0%} loss rate)")
                    self._side_blocked[side] = True
                else:
                    if self._side_blocked[side]:
                        logger.info(f"✅ SIDE BLOCKER: {side} UNBLOCKED — loss rate improved to {loss_rate:.0%}")
                    self._side_blocked[side] = False
            else:
                self._side_blocked[side] = False
        
        # Log summary
        for side in ('LONG', 'SHORT'):
            s = self._side_stats[side]
            blocked_str = " 🚫BLOCKED" if self._side_blocked[side] else ""
            logger.info(f"📊 SIDE TRACKER: {side} = {s['wins']}W/{s['losses']}L (total PnL ${sum(s['recent_pnl']):+.2f}){blocked_str}")

    def _record_side_result(self, side: str, pnl: float, is_win: bool):
        """Record a trade result for the side performance tracker."""
        side = side.upper()
        if side not in ('LONG', 'SHORT'):
            return
        if is_win:
            self._side_stats[side]['wins'] += 1
        else:
            self._side_stats[side]['losses'] += 1
        self._side_stats[side]['recent_pnl'].append(pnl)
        if len(self._side_stats[side]['recent_pnl']) > self.SIDE_BLOCK_LOOKBACK:
            self._side_stats[side]['recent_pnl'] = self._side_stats[side]['recent_pnl'][-self.SIDE_BLOCK_LOOKBACK:]
        
        # Re-evaluate blocking
        self._update_side_stats_from_history()

    def is_side_blocked(self, side: str) -> bool:
        """Check if a trading side is currently blocked due to poor performance."""
        return self._side_blocked.get(side.upper(), False)
    
    def _switch_api_key(self) -> bool:
        """Switch to next available API key. Returns True if switched successfully."""
        if len(self.api_keys) <= 1:
            return False
        
        # Move to next key
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.api_key = self.api_keys[self.current_key_index]
        
        # Reinitialize client with new key
        try:
            if GENAI_NEW:
                self.client = genai_new.Client(api_key=self.api_key)
                logger.info(f"🔄 Switched to API key #{self.current_key_index + 1}")
                return True
            else:
                genai_old.configure(api_key=self.api_key)
                self.model = genai_old.GenerativeModel(self.model_name)
                logger.info(f"🔄 Switched to API key #{self.current_key_index + 1}")
                return True
        except Exception as e:
            logger.error(f"Failed to switch API key: {e}")
            return False
    
    def _is_ai_available(self) -> bool:
        """Check if AI is available (not in cooldown, not too many failures)."""
        if not self.use_ai:
            return False
        
        # Check cooldown
        if self.ai_cooldown_until:
            now = datetime.now(timezone.utc)
            if now < self.ai_cooldown_until:
                remaining = (self.ai_cooldown_until - now).total_seconds()
                if remaining > 60:  # Only log if > 1 minute
                    logger.debug(f"AI in cooldown for {remaining:.0f}s more")
                return False
            else:
                # Cooldown expired
                self.ai_cooldown_until = None
                self.ai_failures_in_row = 0
                logger.info("🔄 AI cooldown expired, resuming AI calls")
        
        return True
    
    def get_ai_status(self) -> Dict[str, Any]:
        """Get current AI status for monitoring/dashboard."""
        now = datetime.now(timezone.utc)
        cooldown_remaining = 0
        if self.ai_cooldown_until and now < self.ai_cooldown_until:
            cooldown_remaining = (self.ai_cooldown_until - now).total_seconds()
        
        return {
            "available": self._is_ai_available(),
            "enabled": self.use_ai,
            "provider": self.ai_provider,
            "model": self.model_name if self.use_ai else None,
            "api_keys_count": len(self.api_keys),
            "current_key_index": self.current_key_index + 1,
            "call_count": self.ai_call_count,
            "failures_in_row": self.ai_failures_in_row,
            "in_cooldown": cooldown_remaining > 0,
            "cooldown_remaining_seconds": int(cooldown_remaining),
            "last_call": self.last_ai_call_time.isoformat() if self.last_ai_call_time else None
        }
    
    def _generate_content(self, prompt: str, retry_count: int = 0, timeout_ms: int = 30000, temperature: float = 0.2) -> Optional[str]:
        """Generate content using Gemini API.
        Automatically switches API key on quota errors.
        Includes rate limiting and cooldown management.
        
        Args:
            prompt: The prompt to send
            retry_count: Current retry attempt (for key rotation)
            timeout_ms: HTTP timeout in milliseconds (default 30000, chat uses 60000)
            temperature: Gemini temperature (0.2 for scanning, 0.5 for chat)
        """
        
        # Check if we're in cooldown
        if not self._is_ai_available():
            return None
        
        try:
            self.last_ai_call_time = datetime.now(timezone.utc)
            self.ai_call_count += 1
            
            # Gemini (Google) API - new SDK
            if GENAI_NEW and self.client:
                logger.info(f"🤖 Calling Gemini API (new SDK) with model: {self.model_name}")
                from google.genai import types
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        http_options={'timeout': timeout_ms},
                    )
                )
                # Success - reset failure counter
                self.ai_failures_in_row = 0
                logger.info(f"🤖 Gemini response received ({len(response.text) if response.text else 0} chars)")
                return response.text
            
            # Gemini (Google) API - legacy SDK
            elif self.model:
                response = self.model.generate_content(prompt)
                # Success - reset failure counter
                self.ai_failures_in_row = 0
                return response.text
            else:
                logger.error("No AI model available")
                return None
        except Exception as e:
            error_str = str(e)
            self.ai_failures_in_row += 1
            
            # Check for quota exhaustion (429 error)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower() or "rate_limit" in error_str.lower():
                logger.warning(f"API quota exhausted on key #{self.current_key_index + 1}")
                
                # Try switching to backup key (only once per call)
                if retry_count < len(self.api_keys) - 1 and self._switch_api_key():
                    logger.info("Retrying with backup API key...")
                    return self._generate_content(prompt, retry_count + 1)
                else:
                    # All keys exhausted - activate cooldown to prevent hammering API
                    self.cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=60)
                    logger.error(f"AI generation error: All API keys exhausted. Cooldown for 60s.")
                    return None
            else:
                logger.error(f"AI generation error: {e}")
                
                # Activate cooldown after too many consecutive failures
                if self.ai_failures_in_row >= self.max_failures_before_cooldown:
                    self.cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=30)
                    logger.warning(f"🟡 {self.ai_failures_in_row} consecutive AI failures. Cooldown for 30s.")
                
                return None
    
    def record_trade_result(self, is_win: bool, pnl: float, symbol: str = "", side: str = "",
                            entry_component_scores: Dict[str, float] = None,
                            exit_reason: str = "", pnl_pct: float = 0.0,
                            hold_time_seconds: float = 0, entry_time: datetime = None):
        """Record a trade result to inform future AI decisions and update Bayesian weights."""
        result = "win" if is_win else "loss"
        self.recent_trades.append({
            "result": result,
            "pnl": pnl,
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "symbol": symbol,
            "side": side.upper() if side else ""
        })
        # Keep only last 10 trades
        if len(self.recent_trades) > 10:
            self.recent_trades = self.recent_trades[-10:]
        
        # FIX: Breakeven exits ($0.00) are NOT real wins — they indicate the trade setup
        # was problematic. Treat as neutral: don't count as loss, but don't clear cooldowns
        # and add a short re-entry cooldown to prevent open→breakeven→reopen loops.
        is_breakeven = is_win and abs(pnl) < 0.01  # $0.00 PnL = breakeven
        
        if is_win and not is_breakeven:
            self.total_wins += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            # Real win resets the cooldown for this symbol
            if symbol:
                symbol_key = symbol.replace('/', '').replace(':USDT', '').upper()
                if symbol_key in self.symbol_loss_tracker:
                    del self.symbol_loss_tracker[symbol_key]
                    logger.info(f"✅ {symbol}: Cooldown cleared after WIN")
        elif is_breakeven:
            # Breakeven: Don't count as win OR loss, but add short cooldown
            # to prevent immediate re-entry on the same symbol
            logger.warning(f"⚖️ {symbol}: BREAKEVEN exit ($0.00) — adding 5min re-entry cooldown")
            if symbol:
                symbol_key = symbol.replace('/', '').replace(':USDT', '').upper()
                now_be = datetime.now(timezone.utc)
                self.symbol_loss_tracker[symbol_key] = {
                    "last_loss_time": now_be,
                    "consecutive_losses": 0  # Not a real loss, but blocks re-entry
                }
                # Short global cooldown too — 3 minutes to let market settle
                from datetime import timedelta
                self._global_cooldown_until = now_be + timedelta(minutes=3)
                logger.warning(f"⏸️ BREAKEVEN COOLDOWN: 3min global + 5min symbol pause")
        else:
            self.total_losses += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            # Track per-symbol losses for cooldown
            if symbol:
                symbol_key = symbol.replace('/', '').replace(':USDT', '').upper()
                now = datetime.now(timezone.utc)
                if symbol_key not in self.symbol_loss_tracker:
                    self.symbol_loss_tracker[symbol_key] = {
                        "last_loss_time": now,
                        "consecutive_losses": 1
                    }
                else:
                    self.symbol_loss_tracker[symbol_key]["last_loss_time"] = now
                    self.symbol_loss_tracker[symbol_key]["consecutive_losses"] += 1
                
                consec = self.symbol_loss_tracker[symbol_key]["consecutive_losses"]
                cooldown = self.EXTENDED_COOLDOWN_MINUTES if consec >= self.MAX_SYMBOL_LOSSES else self.SYMBOL_COOLDOWN_MINUTES
                logger.warning(f"🚫 {symbol}: LOSS #{consec} - Cooldown {cooldown} mins")
            
            # === GLOBAL COOLDOWN: Any loss pauses ALL trading ===
            now_gc = datetime.now(timezone.utc)
            if abs(pnl) >= self.BIG_LOSS_THRESHOLD:
                gc_mins = self.GLOBAL_COOLDOWN_BIG_LOSS_MINUTES
            else:
                gc_mins = self.GLOBAL_COOLDOWN_MINUTES
            from datetime import timedelta
            self._global_cooldown_until = now_gc + timedelta(minutes=gc_mins)
            logger.warning(f"⏸️ GLOBAL COOLDOWN: {gc_mins}min pause after ${pnl:.2f} loss (until {self._global_cooldown_until.strftime('%H:%M:%S')} UTC)")
        
        # === DAILY TRADE COUNTER ===
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if today != self._daily_trade_date:
            self._daily_trade_date = today
            self._daily_trade_count = 0
        self._daily_trade_count += 1
        logger.info(f"📊 Daily trades: {self._daily_trade_count}/{self.MAX_TRADES_PER_DAY}")
        
        logger.info(f"Trade recorded: {result} ${pnl:+.2f} | Streak: {self.consecutive_wins}W / {self.consecutive_losses}L")
        
        # === SIDE PERFORMANCE TRACKER: Record result per side ===
        if side:
            self._record_side_result(side, pnl, is_win and not is_breakeven)
        
        # ═══════════════════════════════════════════════════════════════
        # INTELLIGENT SYSTEM HOOKS — Feed all 5 learners with trade data
        # ═══════════════════════════════════════════════════════════════
        try:
            # System #1: Exit Reason Learning
            if exit_reason and hasattr(self, 'exit_learner'):
                self.exit_learner.record(
                    exit_reason=exit_reason,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    symbol=symbol,
                    side=side,
                    hold_time_seconds=hold_time_seconds
                )
            
            # System #2: Session Performance Learning
            if hasattr(self, 'session_learner'):
                self.session_learner.record(
                    pnl=pnl,
                    is_win=is_win,
                    symbol=symbol,
                    side=side,
                    entry_time=entry_time
                )
            
            # System #3: Symbol Memory
            if symbol and hasattr(self, 'symbol_memory'):
                self.symbol_memory.record(
                    symbol=symbol,
                    side=side,
                    pnl=pnl,
                    is_win=is_win
                )
            
            # System #4: Auto-Tuner — evaluate and apply parameter adjustments
            if hasattr(self, 'auto_tuner'):
                self.auto_tuner.trade_count += 1
                self.auto_tuner._save()
                
                # Actually evaluate and apply tuning suggestions
                if self.auto_tuner.should_tune():
                    try:
                        # Load trade history and adaptive params directly from files
                        trade_history = []
                        if TRADE_HISTORY_FILE.exists():
                            with open(TRADE_HISTORY_FILE, 'r') as _thf:
                                trade_history = json.load(_thf)
                        
                        current_params = {}
                        if ADAPTIVE_PARAMS_FILE.exists():
                            with open(ADAPTIVE_PARAMS_FILE, 'r') as _apf:
                                ap_data = json.load(_apf)
                                current_params = ap_data.get('params', {})
                        
                        if trade_history and current_params:
                            suggestion = self.auto_tuner.evaluate_and_suggest(trade_history, current_params)
                            
                            if suggestion.get('should_adjust') and suggestion.get('adjustments'):
                                # Apply adjustments to adaptive_params.json
                                for param_name, new_value in suggestion['adjustments'].items():
                                    if param_name in current_params:
                                        old_val = current_params[param_name].get('current', 'N/A')
                                        current_params[param_name]['current'] = new_value
                                        logger.warning(f"🎛️ AUTO-TUNE APPLIED: {param_name} {old_val} → {new_value}")
                                
                                # Save updated params back to file
                                ap_data['params'] = current_params
                                ap_data['last_updated'] = datetime.now(timezone.utc).isoformat()
                                with open(ADAPTIVE_PARAMS_FILE, 'w') as _apf:
                                    json.dump(ap_data, _apf, indent=2)
                                
                                logger.warning(f"🎛️ AUTO-TUNE: {suggestion.get('reason', 'adjustments applied')} "
                                             f"| WR={suggestion.get('recent_wr', 0):.0f}% "
                                             f"| avgPnL=${suggestion.get('recent_avg_pnl', 0):.2f}")
                            else:
                                logger.info(f"🎛️ Auto-tune evaluated: {suggestion.get('reason', 'no changes needed')}")
                    except Exception as tune_err:
                        logger.warning(f"⚠️ Auto-tuner evaluation failed (non-fatal): {tune_err}")
            
        except Exception as e:
            logger.warning(f"⚠️ Intelligent system hook error (non-fatal): {e}")
        
        # Reset peak profit tracking for closed position
        if hasattr(self, '_peak_profits') and symbol:
            # Clear all peak profits for this symbol (both LONG and SHORT)
            keys_to_remove = [k for k in self._peak_profits.keys() if symbol.upper() in k.upper()]
            for key in keys_to_remove:
                del self._peak_profits[key]
                logger.debug(f"Reset peak profit tracking for {key}")
        
        # Also clear the new peak profit cache
        if symbol:
            self.clear_peak_profit(symbol)

    def _is_symbol_on_cooldown(self, symbol: str) -> tuple:
        """
        Check if a symbol is on cooldown after recent losses.
        Returns: (is_on_cooldown: bool, reason: str, minutes_remaining: int)
        """
        symbol_key = symbol.replace('/', '').replace(':USDT', '').upper()
        
        if symbol_key not in self.symbol_loss_tracker:
            return False, "", 0
        
        tracker = self.symbol_loss_tracker[symbol_key]
        last_loss = tracker.get("last_loss_time")
        consec_losses = tracker.get("consecutive_losses", 1)
        
        if not last_loss:
            return False, "", 0
        
        # Determine cooldown duration based on consecutive losses
        if consec_losses == 0:
            # Breakeven exit — short cooldown only
            cooldown_mins = 5
            reason = "breakeven exit"
        elif consec_losses >= self.MAX_SYMBOL_LOSSES:
            cooldown_mins = self.EXTENDED_COOLDOWN_MINUTES
            reason = f"{consec_losses} consecutive losses"
        else:
            cooldown_mins = self.SYMBOL_COOLDOWN_MINUTES
            reason = "recent loss"
        
        now = datetime.now(timezone.utc)
        elapsed = (now - last_loss).total_seconds() / 60
        
        if elapsed < cooldown_mins:
            remaining = int(cooldown_mins - elapsed)
            return True, reason, remaining
        
        # Cooldown expired, clean up
        del self.symbol_loss_tracker[symbol_key]
        return False, "", 0

    def _get_symbol_performance(self, symbol: str) -> Dict[str, Any]:
        """
        Get performance statistics for a specific symbol WITH per-direction breakdown.
        Returns recent win/loss history and PnL for this symbol, split by LONG/SHORT.
        This is critical for per-pair direction bias detection.
        """
        symbol_key = symbol.replace('/', '').replace(':USDT', '').upper()
        
        # Filter recent trades for this symbol
        symbol_trades = [
            t for t in self.recent_trades 
            if t.get('symbol', '').replace('/', '').replace(':USDT', '').upper() == symbol_key
        ]
        
        if not symbol_trades:
            return {
                "has_history": False,
                "wins": 0,
                "losses": 0,
                "total_pnl": 0,
                "win_rate": "N/A",
                "last_result": "N/A",
                # Per-direction stats (empty)
                "long_wins": 0,
                "long_losses": 0,
                "long_pnl": 0,
                "long_wr": 0,
                "short_wins": 0,
                "short_losses": 0,
                "short_pnl": 0,
                "short_wr": 0,
                "preferred_direction": None,
                "direction_advice": ""
            }
        
        # Overall stats
        wins = sum(1 for t in symbol_trades if t.get('result', '').lower() == 'win')
        losses = len(symbol_trades) - wins
        total_pnl = sum(t.get('pnl', 0) for t in symbol_trades)
        win_rate = wins / len(symbol_trades) * 100 if symbol_trades else 0
        last_result = symbol_trades[-1].get('result', 'unknown').upper() if symbol_trades else "N/A"
        
        # Per-direction breakdown for THIS symbol
        long_trades = [t for t in symbol_trades if t.get('side', '').upper() == 'LONG']
        short_trades = [t for t in symbol_trades if t.get('side', '').upper() == 'SHORT']
        
        long_wins = sum(1 for t in long_trades if t.get('result', '').lower() == 'win')
        long_losses = len(long_trades) - long_wins
        long_pnl = sum(t.get('pnl', 0) for t in long_trades)
        long_wr = (long_wins / len(long_trades) * 100) if long_trades else 0
        
        short_wins = sum(1 for t in short_trades if t.get('result', '').lower() == 'win')
        short_losses = len(short_trades) - short_wins
        short_pnl = sum(t.get('pnl', 0) for t in short_trades)
        short_wr = (short_wins / len(short_trades) * 100) if short_trades else 0
        
        # Determine preferred direction for THIS symbol
        preferred_direction = None
        direction_advice = ""
        
        # Need at least 2 trades in each direction to make judgment
        if len(long_trades) >= 2 and len(short_trades) >= 2:
            if long_wr >= 60 and short_wr < 45:
                preferred_direction = "LONG"
                direction_advice = f"🎯 {symbol_key}: LONG strongly preferred ({long_wins}W/{long_losses}L={long_wr:.0f}% vs SHORT {short_wins}W/{short_losses}L={short_wr:.0f}%)"
            elif short_wr >= 60 and long_wr < 45:
                preferred_direction = "SHORT"
                direction_advice = f"🎯 {symbol_key}: SHORT strongly preferred ({short_wins}W/{short_losses}L={short_wr:.0f}% vs LONG {long_wins}W/{long_losses}L={long_wr:.0f}%)"
            elif long_pnl > 0 and short_pnl < -2:
                preferred_direction = "LONG"
                direction_advice = f"💰 {symbol_key}: LONG profitable (${long_pnl:+.2f}) but SHORT losing (${short_pnl:+.2f})"
            elif short_pnl > 0 and long_pnl < -2:
                preferred_direction = "SHORT"
                direction_advice = f"💰 {symbol_key}: SHORT profitable (${short_pnl:+.2f}) but LONG losing (${long_pnl:+.2f})"
        elif len(long_trades) >= 3 and len(short_trades) == 0:
            if long_wr >= 60:
                preferred_direction = "LONG"
                direction_advice = f"📊 {symbol_key}: Only LONG trades ({long_wins}W/{long_losses}L={long_wr:.0f}%) - no SHORT history"
        elif len(short_trades) >= 3 and len(long_trades) == 0:
            if short_wr >= 60:
                preferred_direction = "SHORT"
                direction_advice = f"📊 {symbol_key}: Only SHORT trades ({short_wins}W/{short_losses}L={short_wr:.0f}%) - no LONG history"
        
        return {
            "has_history": True,
            "wins": wins,
            "losses": losses,
            "total_pnl": total_pnl,
            "win_rate": f"{win_rate:.0f}%",
            "last_result": last_result,
            # Per-direction stats for this symbol
            "long_wins": long_wins,
            "long_losses": long_losses,
            "long_trades": len(long_trades),
            "long_pnl": long_pnl,
            "long_wr": long_wr,
            "short_wins": short_wins,
            "short_losses": short_losses,
            "short_trades": len(short_trades),
            "short_pnl": short_pnl,
            "short_wr": short_wr,
            "preferred_direction": preferred_direction,
            "direction_advice": direction_advice
        }

    def _build_symbol_history_section(self, symbol_perf: Dict, symbol: str) -> str:
        """Build a formatted string showing per-pair direction history for AI prompt.
        NOTE: This is SECONDARY info - math analysis is primary!"""
        if not symbol_perf.get("has_history"):
            return f"No trade history for {symbol} yet - decide based on math analysis (this is fine!)."
        
        lines = []
        lines.append(f"📊 Past trades (for context only, math is primary):")
        lines.append(f"   Total: {symbol_perf['wins']}W/{symbol_perf['losses']}L (${symbol_perf['total_pnl']:+.2f})")
        
        # Per-direction breakdown
        if symbol_perf.get("long_trades", 0) > 0:
            lines.append(f"   LONG: {symbol_perf['long_wins']}W/{symbol_perf['long_losses']}L ({symbol_perf['long_wr']:.0f}%)")
        if symbol_perf.get("short_trades", 0) > 0:
            lines.append(f"   SHORT: {symbol_perf['short_wins']}W/{symbol_perf['short_losses']}L ({symbol_perf['short_wr']:.0f}%)")
        
        # Note that this is secondary
        lines.append(f"⚠️ Note: History is from OLD system - trust math analysis more!")
        
        return "\n".join(lines)

    def _get_direction_performance(self) -> Dict[str, Any]:
        """
        Get performance statistics by trade direction (LONG vs SHORT).
        Critical for understanding which direction is actually profitable.
        """
        long_trades = [t for t in self.recent_trades if t.get('side', '').upper() == 'LONG']
        short_trades = [t for t in self.recent_trades if t.get('side', '').upper() == 'SHORT']
        
        long_wins = sum(1 for t in long_trades if t.get('result', '').upper() == 'WIN')
        long_losses = len(long_trades) - long_wins
        long_pnl = sum(t.get('pnl', 0) for t in long_trades)
        long_wr = (long_wins / len(long_trades) * 100) if long_trades else 0
        
        short_wins = sum(1 for t in short_trades if t.get('result', '').upper() == 'WIN')
        short_losses = len(short_trades) - short_wins
        short_pnl = sum(t.get('pnl', 0) for t in short_trades)
        short_wr = (short_wins / len(short_trades) * 100) if short_trades else 0
        
        # Determine which direction is performing better
        better_direction = None
        direction_warning = ""
        
        if len(long_trades) >= 5 and len(short_trades) >= 5:
            if long_wr >= 55 and short_wr < 45:
                better_direction = "LONG"
                direction_warning = f"🚨 CRITICAL: LONG trades have {long_wr:.0f}% WR (${long_pnl:+.2f}), but SHORT trades only {short_wr:.0f}% WR (${short_pnl:+.2f}). STRONGLY PREFER LONG!"
            elif short_wr >= 55 and long_wr < 45:
                better_direction = "SHORT"
                direction_warning = f"🚨 CRITICAL: SHORT trades have {short_wr:.0f}% WR (${short_pnl:+.2f}), but LONG trades only {long_wr:.0f}% WR (${long_pnl:+.2f}). STRONGLY PREFER SHORT!"
            elif long_pnl > 0 and short_pnl < -5:
                better_direction = "LONG"
                direction_warning = f"⚠️ WARNING: LONG is profitable (${long_pnl:+.2f}), but SHORT is losing (${short_pnl:+.2f}). Favor LONG trades!"
            elif short_pnl > 0 and long_pnl < -5:
                better_direction = "SHORT"
                direction_warning = f"⚠️ WARNING: SHORT is profitable (${short_pnl:+.2f}), but LONG is losing (${long_pnl:+.2f}). Favor SHORT trades!"
        
        return {
            "long_trades": len(long_trades),
            "long_wins": long_wins,
            "long_losses": long_losses,
            "long_pnl": long_pnl,
            "long_wr": long_wr,
            "short_trades": len(short_trades),
            "short_wins": short_wins,
            "short_losses": short_losses,
            "short_pnl": short_pnl,
            "short_wr": short_wr,
            "better_direction": better_direction,
            "direction_warning": direction_warning
        }

    def _get_news_context(self) -> Dict[str, Any]:
        """
        Get current market news and sentiment context for AI decision making.
        Uses cached data to avoid API rate limits during trading.
        CACHED: 5-minute TTL to prevent 100+ redundant RSS+LLM calls per scan cycle.
        """
        # ═══════════════════════════════════════════════════════════════
        # CACHE CHECK: News doesn't change every second — cache 5 minutes
        # This prevents 100+ redundant RSS fetches + Gemini LLM calls
        # during dashboard scoring (50 pairs × 2 directions = 100 calls!)
        # ═══════════════════════════════════════════════════════════════
        import time as _time
        _now = _time.time()
        _NEWS_CACHE_TTL = 300  # 5 minutes
        
        if hasattr(self, '_news_context_cache') and self._news_context_cache:
            _cache_age = _now - self._news_context_cache.get('_cached_at', 0)
            if _cache_age < _NEWS_CACHE_TTL:
                logger.debug(f"📰 NEWS CONTEXT: Using cache ({_cache_age:.0f}s old)")
                return self._news_context_cache
        
        try:
            import asyncio
            from news_monitor import NewsMonitor
            
            monitor = NewsMonitor()
            
            # Run async function - handle both sync and async contexts
            try:
                # Check if there's already a running event loop
                loop = asyncio.get_running_loop()
                # Already in async context - can't run sync event loop
                # Use synchronous HTTP to fetch Fear & Greed directly
                return self._get_news_context_sync()
            except RuntimeError:
                # No running loop - safe to create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    summary = loop.run_until_complete(monitor.get_market_summary())
                finally:
                    loop.close()
            
            # Extract key info for AI
            sentiment = summary.get('sentiment', {})
            news_summary = summary.get('news_summary', {})
            critical_news = summary.get('critical_news', [])
            
            # Build news context
            news_context = {
                'fear_greed_index': sentiment.get('fear_greed_index', 50),
                'fear_greed_label': sentiment.get('fear_greed_label', 'Neutral'),
                'market_cap_change_24h': sentiment.get('market_cap_change_24h', 0),
                'btc_dominance': sentiment.get('btc_dominance', 0),
                'news_sentiment': news_summary.get('average_sentiment', 0),
                'bullish_news_count': news_summary.get('bullish_count', 0),
                'bearish_news_count': news_summary.get('bearish_count', 0),
                'critical_news_count': news_summary.get('critical_count', 0),
                'critical_headlines': [n.get('title', '')[:80] for n in critical_news[:3]],
                'recommendation': summary.get('recommendation', '').split('\n')[0] if summary.get('recommendation') else ''
            }
            
            # Generate warning based on sentiment
            warning = ""
            fg = news_context['fear_greed_index']
            cap_change = news_context['market_cap_change_24h']
            
            if fg <= 20:
                warning = f"🔴 EXTREME FEAR ({fg}) - Market panic, potential oversold bounce opportunity"
            elif fg <= 35:
                warning = f"🟠 FEAR ({fg}) - Cautious sentiment, watch for reversals"
            elif fg >= 80:
                warning = f"🟢 EXTREME GREED ({fg}) - Euphoria, potential top forming"
            elif fg >= 65:
                warning = f"🟡 GREED ({fg}) - Bullish sentiment, momentum likely continues"
            
            if abs(cap_change) >= 5:
                direction = "📈 SURGING" if cap_change > 0 else "📉 CRASHING"
                warning += f"\n{direction}: Market cap {cap_change:+.1f}% in 24h - MAJOR MOVE"
            
            if news_context['critical_news_count'] > 0:
                warning += f"\n🚨 {news_context['critical_news_count']} CRITICAL NEWS items - check before trading!"
            
            news_context['warning'] = warning
            
            # Log news context for visibility
            logger.info(f"📰 NEWS CONTEXT: Fear&Greed={fg} ({news_context['fear_greed_label']}), Market 24h={cap_change:+.1f}%, Sentiment={news_context['news_sentiment']:+.2f}")
            
            # Cache the result
            news_context['_cached_at'] = _now
            self._news_context_cache = news_context
            
            return news_context
            
        except Exception as e:
            logger.debug(f"News context fetch error: {e}")
            return self._get_default_news_context()
    
    def _get_news_context_sync(self) -> Dict[str, Any]:
        """Fetch Fear & Greed synchronously when inside async event loop.
        Uses urllib (stdlib) to avoid async issues. Cached for 5 minutes."""
        import time as _time
        _now = _time.time()
        
        # Check cache first (same cache as async version)
        if hasattr(self, '_news_context_cache') and self._news_context_cache:
            _cache_age = _now - self._news_context_cache.get('_cached_at', 0)
            if _cache_age < 300:
                return self._news_context_cache
        
        try:
            import urllib.request
            import json as _json
            
            # Fetch Fear & Greed Index (simple GET, no auth needed)
            req = urllib.request.Request(
                'https://api.alternative.me/fng/?limit=1',
                headers={'User-Agent': 'JulabaBot/1.0'}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = _json.loads(resp.read().decode())
                fg_data = data.get('data', [{}])[0]
                fg_index = int(fg_data.get('value', 50))
                fg_label = fg_data.get('value_classification', 'Neutral')
            
            # Build context with real F&G data
            news_context = {
                'fear_greed_index': fg_index,
                'fear_greed_label': fg_label,
                'market_cap_change_24h': 0,
                'news_sentiment': 0,
                'critical_headlines': [],
                'warning': '',
                '_cached_at': _now
            }
            
            # Generate warning
            if fg_index <= 20:
                news_context['warning'] = f"🔴 EXTREME FEAR ({fg_index}) - Market panic"
            elif fg_index <= 35:
                news_context['warning'] = f"🟠 FEAR ({fg_index}) - Cautious sentiment"
            elif fg_index >= 80:
                news_context['warning'] = f"🟢 EXTREME GREED ({fg_index}) - Euphoria"
            elif fg_index >= 65:
                news_context['warning'] = f"🟡 GREED ({fg_index}) - Bullish sentiment"
            
            # Cache it (shared with async version)
            self._news_context_cache = news_context
            logger.info(f"📰 NEWS CONTEXT (sync): Fear&Greed={fg_index} ({fg_label})")
            
            return news_context
            
        except Exception as e:
            logger.warning(f"📰 Sync F&G fetch failed: {type(e).__name__}: {e}")
            return self._get_default_news_context()
    
    def _get_default_news_context(self) -> Dict[str, Any]:
        """Return default news context when ALL fetch methods fail."""
        return {
            'fear_greed_index': 50,
            'fear_greed_label': 'Neutral',
            'market_cap_change_24h': 0,
            'news_sentiment': 0,
            'critical_headlines': [],
            'warning': ''
        }

    def record_system_message(self, message: str):
        """Record a system-level event to the chat history."""
        self.chat_history.append({
            "role": "model",
            "content": f"[SYSTEM EVENT]: {message}"
        })
        self._save_chat_history()

    def _ai_analysis(
        self,
        signal: int,
        context: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """Use Google Gemini API for signal analysis with SKEPTIC MODE. Retries once and notifies via Telegram on fallback."""
        signal_type = "LONG" if signal == 1 else "SHORT"
        perf = self._get_performance_context()
        market = self._get_market_hours_context()
        extra_caution = ""
        if perf["last_trade_was_loss"]:
            extra_caution = f"\n⚠️ CAUTION: Last {perf['consecutive_losses']} trade(s) were losses. Be extra skeptical!"
        if perf["consecutive_losses"] >= 2:
            extra_caution += "\n🛑 LOSING STREAK: Require very high confidence to approve."
        if market["is_weekend"]:
            extra_caution += "\n📅 WEEKEND: Lower liquidity, higher risk of false moves."
        if market["activity_level"] == "low":
            extra_caution += "\n🌙 LOW ACTIVITY HOURS: Increased slippage risk."
        # Build ML insight section for prompt
        ml_insight = context.get('ml_insight', {})
        ml_section = self._build_ml_section(ml_insight)
        
        # Build system score section for prompt
        system_score = context.get('system_score', {})
        system_section = self._build_system_score_section(system_score)
        
        # Build advanced math analysis section for prompt
        math_check = context.get('math_check', {})
        math_section = self._build_math_analysis_section(math_check)
        
        # Build institutional microstructure section for prompt
        inst_data = context.get('institutional_data', {})
        inst_section = self._build_institutional_section(inst_data)
        
        # Log that advanced math is being included in AI prompt
        if math_check.get('detailed_analysis'):
            detailed = math_check['detailed_analysis']
            logger.info(f"🧮 AI MATH INTEGRATION: Score={math_check.get('score', 0):.0f}, "
                       f"Kalman={detailed.get('kalman_momentum', 'N/A')}, "
                       f"POC=${detailed.get('poc_price', 0):,.0f}, "
                       f"RSI_Div={bool(detailed.get('rsi_divergence_mtf', {}).get('regular_bullish_divergence') or detailed.get('rsi_divergence_mtf', {}).get('regular_bearish_divergence'))}")
        
        prompt = (
            f"You are the FINAL DECISION MAKER for an autonomous crypto trading bot.\n"
            f"Act as a PhD Quantitative Analyst and Risk Manager.\n"
            f"Your job is to PROTECT capital by applying rigorous mathematical verification.\n"
            f"The system has already analyzed this signal through multiple layers:\n"
            f"  1. Technical indicators generated this signal\n"
            f"  2. ML model evaluated historical pattern similarity\n"
            f"  3. NOW YOU verify the mathematical probability of success\n\n"
            f"=== SIGNAL ===\n"
            f"Proposed Trade: {signal_type} on {symbol}\n"
            f"=== MARKET DATA ===\n"
            f"Current Price: ${context['current_price']}\n"
            f"1-Hour Price Change: {context['price_change_1h']}%\n"
            f"Volume Ratio (vs avg): {context['volume_ratio']}x\n"
            f"Trend (SMA10 vs SMA20): {context['trend']}\n"
            f"Volatility (ATR%): {context['volatility_pct']}%\n"
            f"{ml_section}"
            f"{system_section}"
            f"{math_section}"
            f"{inst_section}"
            f"=== TRADING PERFORMANCE ===\n"
            f"Total Trades: {perf['total_trades']}\n"
            f"Win Rate: {perf['win_rate']}%\n"
            f"Current Streak: {perf['consecutive_wins']}W / {perf['consecutive_losses']}L\n"
            f"Recent P&L (last 5): ${perf['recent_pnl']}\n"
            f"=== MARKET SESSION ===\n"
            f"Session: {market['session']} ({market['hour_utc']}:00 UTC)\n"
            f"Activity Level: {market['activity_level']}\n"
            f"Weekend: {market['is_weekend']}\n"
            f"{extra_caution}\n"
            f"=== YOUR DECISION ===\n"
            f"Use the ADVANCED MATH ANALYSIS above to make your decision:\n"
            f"1. Is the Math Score ({math_check.get('score', 50):.0f}/100) sufficient? (need >= 55 to approve)\n"
            f"2. Review the REASONS FOR and AGAINST from the math analysis\n"
            f"3. Check Kalman momentum direction - does it support this trade?\n"
            f"4. Check Volume Profile (POC) - is price at a good entry level?\n"
            f"5. Check for RSI divergences - any warning signals?\n"
            f"6. Check INSTITUTIONAL DATA: Does funding rate, order book depth, and trade tape CVD confirm or contradict?\n"
            f"IMPORTANT: Trust the PhD Math Score over technical momentum.\n"
            f"Only approve if Math Score >= 55 AND reasons FOR outweigh reasons AGAINST.\n"
            f"Respond ONLY with this JSON format, no other text:\n"
            f'{{"reasons_against": ["math_reason1", "math_reason2", "math_reason3"], "approved": false, "confidence": 0.65, "reasoning": "quantitative justification referencing the math analysis", "risk_assessment": "low/medium/high", "ml_agreement": "agree/disagree/neutral", "math_score_assessment": "appropriate/too_high/too_low"}}'
        )
        for attempt in range(2):
            try:
                result_text = self._generate_content(prompt)
                if not result_text:
                    continue
                result_text = result_text.strip()
                # Parse JSON from response
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0]
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0]
                result_text = result_text.strip()
                result = json.loads(result_text)
                result.setdefault("approved", False)
                result.setdefault("confidence", 0.5)
                result.setdefault("reasoning", "AI analysis")
                result.setdefault("risk_assessment", "medium")
                result.setdefault("reasons_against", [])
                result.setdefault("ml_agreement", "neutral")  # AI's take on ML prediction
                # Platt scaling: calibrate raw Gemini confidence
                raw_conf = result["confidence"]
                result["confidence"] = self._calibrate_ai_confidence(raw_conf)
                result["raw_confidence"] = raw_conf  # Keep original for tracking
                # Apply confidence threshold with LOSS COOLDOWN
                perf = self._get_performance_context()
                if perf["consecutive_losses"] >= 2:
                    required_threshold = self.loss_cooldown_threshold
                    logger.info(f"Loss cooldown active: requiring {required_threshold:.0%} confidence")
                elif perf["last_trade_was_loss"]:
                    required_threshold = 0.85
                else:
                    required_threshold = self.confidence_threshold
                result["approved"] = result["approved"] and result["confidence"] >= required_threshold
                result["threshold_used"] = required_threshold
                if result.get("reasons_against"):
                    logger.info(f"AI reasons against trade: {result['reasons_against']}")
                # Log ML agreement if present
                if result.get("ml_agreement") != "neutral":
                    logger.info(f"AI on ML prediction: {result['ml_agreement']}")
                return result
            except Exception as e:
                logger.warning(f"Gemini analysis attempt {attempt+1} failed: {e}")
        # If both attempts fail, notify via Telegram and fallback
        logger.error(f"Gemini analysis failed twice, falling back to rules")
        if self.notifier and hasattr(self.notifier, 'send_message'):
            try:
                import asyncio
                msg = f"⚠️ Gemini AI analysis failed twice for {symbol} {signal_type}. Falling back to rule-based analysis."
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.notifier.send_message(msg))
                else:
                    loop.run_until_complete(self.notifier.send_message(msg))
            except Exception as notify_err:
                logger.warning(f"Failed to send Telegram notification: {notify_err}")
            return self._rule_based_analysis(signal, context, symbol)
    
    def _calibrate_ai_confidence(self, raw_confidence: float) -> float:
        """Apply Platt scaling calibration to raw Gemini confidence.
        
        LLMs tend to output overconfident values (e.g. 0.85 when true probability
        is ~0.60). This applies a sigmoid calibration to map raw confidence to
        more realistic probability estimates.
        
        Calibration params (a, b) should be tuned on historical data via:
            calibrated = 1 / (1 + exp(-(a * raw + b)))
        Defaults are conservative: slight downward shift.
        """
        import math
        # Calibration parameters — tune from historical AI decisions
        # a controls steepness, b controls horizontal shift
        # Negative b shifts curve right (lowers output for same input)
        a = 1.8   # Steeper than identity — amplifies differences
        b = -0.85  # Shifts right — 0.85 raw → ~0.72 calibrated
        
        # Clamp input
        raw_confidence = max(0.0, min(1.0, raw_confidence))
        
        # Convert to logit space, apply Platt scaling
        # Map [0,1] → [-6,6] first to avoid log(0)
        eps = 1e-6
        raw_clamped = max(eps, min(1 - eps, raw_confidence))
        logit = math.log(raw_clamped / (1 - raw_clamped))
        
        calibrated = 1.0 / (1.0 + math.exp(-(a * logit + b)))
        
        if abs(calibrated - raw_confidence) > 0.01:
            logger.debug(f"AI confidence calibrated: {raw_confidence:.3f} → {calibrated:.3f}")
        
        return round(calibrated, 4)

    def _get_performance_context(self) -> Dict[str, Any]:
        """Get trading performance context for AI prompts."""
        total_trades = self.total_wins + self.total_losses
        win_rate_pct = (self.total_wins / max(1, total_trades)) * 100
        recent_pnl = sum(t.get('pnl', 0) for t in self.recent_trades[-5:]) if self.recent_trades else 0
        return {
            "total_trades": total_trades,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "last_trade_was_loss": self.consecutive_losses > 0,
            "recent_trades": self.recent_trades[-5:] if self.recent_trades else [],
            "win_rate": round(win_rate_pct, 1),
            "recent_pnl": round(recent_pnl, 2)
        }
    
    def _get_market_hours_context(self) -> Dict[str, Any]:
        """Get market timing context using proper UTC time."""
        now_utc = get_utc_time()
        now_berlin = get_berlin_time()
        hour_utc = now_utc.hour
        
        # Crypto market sessions (based on UTC - global standard)
        if 0 <= hour_utc < 8:
            session = "Asia"
            activity = "moderate"
        elif 8 <= hour_utc < 14:
            session = "Europe"
            activity = "high"
        elif 14 <= hour_utc < 21:
            session = "US"
            activity = "high"
        else:
            session = "Late US/Early Asia"
            activity = "low"
        
        # Weekend check (lower liquidity)
        is_weekend = now_utc.weekday() >= 5
        
        return {
            "session": session,
            "activity_level": activity,
            "is_weekend": is_weekend,
            "hour_utc": hour_utc,
            "hour_berlin": now_berlin.hour,
            "time_berlin": now_berlin.strftime("%H:%M:%S"),
            "time_utc": now_utc.strftime("%H:%M:%S")
        }
    
    def _build_ml_section(self, ml_insight: Dict[str, Any]) -> str:
        """Build ML insight section for AI prompt — DUAL ML models."""
        if not ml_insight or not ml_insight.get('ml_available', False):
            return "=== ML MODELS ===\nStatus: Not available (training in progress)\n\n"
        
        win_prob = ml_insight.get('ml_win_probability', 0.5)
        confidence = ml_insight.get('ml_confidence', 'UNKNOWN')
        ml_source = ml_insight.get('ml_source', 'unknown')
        
        # Determine ML recommendation
        if win_prob >= 0.60:
            ml_rec = "FAVORABLE - ML suggests this trade type has historically performed well"
        elif win_prob <= 0.40:
            ml_rec = "UNFAVORABLE - ML suggests caution, similar setups had lower win rates"
        else:
            ml_rec = "NEUTRAL - ML has no strong signal for this setup"
        
        # Build dual ML breakdown
        hist = ml_insight.get('historical_ml', {})
        live = ml_insight.get('live_ml', {})
        
        hist_section = ""
        if hist.get('available'):
            hist_section = (
                f"  Historical Model: {hist['win_probability']:.1%} win prob ({hist['confidence']})\n"
                f"    Accuracy: {hist.get('model_accuracy', 0):.1%} | "
                f"Samples: {hist.get('training_samples', 0)} | "
                f"Rec: {hist.get('recommendation', 'N/A')}\n"
            )
        else:
            hist_section = "  Historical Model: Not trained yet\n"
        
        live_section = ""
        if live.get('available'):
            live_section = (
                f"  Live Model: {live['win_probability']:.1%} win prob ({live['confidence']})\n"
                f"    Accuracy: {live.get('model_accuracy', 0):.1%} | "
                f"Samples: {live.get('training_samples', 0)} | "
                f"Rec: {live.get('recommendation', 'N/A')}\n"
            )
        else:
            remaining = live.get('total_samples', 0)
            live_section = f"  Live Model: Learning ({remaining} samples, needs 30 to train)\n"
        
        # Agreement check
        if hist.get('available') and live.get('available'):
            hist_take = hist.get('recommendation') == 'TAKE'
            live_take = live.get('recommendation') == 'TAKE'
            if hist_take and live_take:
                agreement = "✅ BOTH MODELS AGREE: TAKE"
            elif not hist_take and not live_take:
                agreement = "🚫 BOTH MODELS AGREE: SKIP"
            else:
                agreement = "⚠️ MODELS DISAGREE — proceed with caution"
        else:
            agreement = "Only one model available"
        
        return (
            f"=== DUAL ML MODEL INSIGHT ===\n"
            f"Combined Win Probability: {win_prob:.1%} ({confidence})\n"
            f"Source: {ml_source.upper()}\n"
            f"ML Recommendation: {ml_rec}\n"
            f"\n"
            f"--- Model Breakdown ---\n"
            f"{hist_section}"
            f"{live_section}"
            f"Agreement: {agreement}\n\n"
        )

    def _build_system_score_section(self, system_score: Dict[str, Any]) -> str:
        """Build system score section for AI prompt."""
        if not system_score or 'combined' not in system_score:
            return "=== SYSTEM SCORE ===\nNot available\n\n"
        
        combined = system_score.get('combined', 50)
        recommendation = system_score.get('recommendation', 'NEUTRAL')
        breakdown = system_score.get('breakdown', 'N/A')
        
        # Interpret the score
        if combined >= 75:
            interpretation = "STRONG - All system components align favorably"
        elif combined >= 60:
            interpretation = "GOOD - Most components favorable, minor concerns"
        elif combined >= 45:
            interpretation = "NEUTRAL - Mixed signals, proceed with caution"
        elif combined >= 30:
            interpretation = "WEAK - Several concerning factors"
        else:
            interpretation = "POOR - System recommends avoiding this trade"
        
        return (
            f"=== SYSTEM SCORE (0-100) ===\n"
            f"Combined Score: {combined:.0f}/100\n"
            f"Recommendation: {recommendation}\n"
            f"Breakdown: {breakdown}\n"
            f"Interpretation: {interpretation}\n\n"
        )

    def _build_institutional_section(self, institutional_data: Dict[str, Any]) -> str:
        """
        Build institutional microstructure data section for AI prompt.
        Includes funding rate, open interest, order book depth, trade tape CVD.
        """
        if not institutional_data or institutional_data.get('composite_bias') is None:
            return ""
        
        lines = ["=== INSTITUTIONAL MICROSTRUCTURE DATA ==="]
        
        # Funding rate
        funding = institutional_data.get('funding', {})
        fr = funding.get('current_rate', 0)
        if fr != 0:
            fr_ann = funding.get('current_rate_annualized', 0)
            fr_trend = funding.get('trend', 'stable')
            extreme_str = " ⚠️ EXTREME" if funding.get('extreme') else ""
            lines.append(f"📊 FUNDING RATE: {fr*100:.4f}% ({fr_ann:.0f}% annualized, {fr_trend}){extreme_str}")
            if fr > 0:
                lines.append(f"   → Longs paying shorts — crowded long positioning")
            elif fr < 0:
                lines.append(f"   → Shorts paying longs — crowded short positioning")
        
        # Open interest
        oi = institutional_data.get('open_interest', {})
        oi_change = oi.get('oi_change_pct', 0)
        if oi.get('oi_contracts', 0) > 0:
            lines.append(f"📊 OPEN INTEREST: {oi['oi_contracts']:,.0f} contracts (${oi.get('oi_value_usd', 0):,.0f})")
            if abs(oi_change) > 0.5:
                lines.append(f"   → Change: {oi_change:+.1f}%")
        
        # Order book
        ob = institutional_data.get('orderbook', {})
        if ob.get('bid_depth_usd', 0) > 0:
            imb = ob.get('imbalance', 0)
            bias_str = "BUY pressure" if imb > 0.2 else "SELL pressure" if imb < -0.2 else "balanced"
            lines.append(f"📊 ORDER BOOK: Spread={ob.get('spread_pct', 0):.4f}%, Imbalance={imb:+.2f} ({bias_str})")
            lines.append(f"   Bids: ${ob['bid_depth_usd']:,.0f} | Asks: ${ob.get('ask_depth_usd', 0):,.0f}")
            if ob.get('thin_side') != 'balanced':
                lines.append(f"   ⚠️ Thin {ob['thin_side']} side — vulnerable to fast move")
        
        # Trade tape / CVD
        tape = institutional_data.get('tape', {})
        if tape.get('buy_volume', 0) > 0 or tape.get('sell_volume', 0) > 0:
            lines.append(f"📊 TAPE (CVD): Buy={tape.get('buy_pct', 50):.0f}% | CVD={tape.get('cvd', 0):+.4f}")
            if tape.get('large_trade_count', 0) > 0:
                lines.append(f"   Whale trades: {tape['large_trade_count']} ({tape.get('large_trade_bias', 'balanced')} bias)")
        
        # Composite
        score = institutional_data.get('composite_score', 0)
        bias = institutional_data.get('composite_bias', 'neutral')
        lines.append(f"📊 COMPOSITE: Score={score:+.0f}, Bias={bias.upper()}")
        
        components = institutional_data.get('components', [])
        if components:
            lines.append(f"   Components: {' | '.join(components[:6])}")
        
        lines.append("")
        return "\n".join(lines) + "\n"
    
    def _build_math_analysis_section(self, math_check: Dict[str, Any]) -> str:
        """
        Build advanced math analysis section for AI prompt.
        Includes Kalman momentum, Volume Profile (POC), RSI divergence MTF, etc.
        """
        if not math_check:
            return "=== ADVANCED MATH ANALYSIS ===\nNot available\n\n"
        
        score = math_check.get('score', 50)
        approved = math_check.get('approved', False)
        confidence_level = math_check.get('confidence_level', 'unknown')
        reasons_for = math_check.get('reasons_for', [])
        reasons_against = math_check.get('reasons_against', [])
        detailed = math_check.get('detailed_analysis', {})
        
        # Build the section
        lines = [
            "=== ADVANCED MATH ANALYSIS (PhD-Level) ===",
            f"Math Score: {score:.0f}/100 ({'APPROVED' if approved else 'NOT APPROVED'})",
            f"Confidence Level: {confidence_level.upper()}",
            ""
        ]
        
        # Kalman Filter Momentum
        kalman_mom = detailed.get('kalman_momentum')
        if kalman_mom is not None:
            kalman_zscore = detailed.get('kalman_momentum_zscore', 0)
            kalman_trend = detailed.get('kalman_momentum_trend', 'unknown')
            lines.append(f"📊 KALMAN MOMENTUM: {kalman_mom:.2f} (z-score: {kalman_zscore:.2f}, trend: {kalman_trend})")
        
        # Volume Profile / POC
        poc_price = detailed.get('poc_price')
        if poc_price is not None:
            price_vs_poc = detailed.get('price_vs_poc', 0)
            in_va = detailed.get('in_value_area', False)
            va_high = detailed.get('value_area_high', 0)
            va_low = detailed.get('value_area_low', 0)
            lines.append(f"📊 VOLUME PROFILE: POC=${poc_price:,.2f}, Price vs POC: {price_vs_poc:+.2f}%")
            lines.append(f"   Value Area: ${va_low:,.2f}-${va_high:,.2f} ({'IN' if in_va else 'OUT'})")
        
        # RSI Divergence
        rsi_div = detailed.get('rsi_divergence_mtf')
        if rsi_div:
            rsi = rsi_div.get('current_rsi', 50)
            reg_bull = rsi_div.get('regular_bullish_divergence', False)
            reg_bear = rsi_div.get('regular_bearish_divergence', False)
            hid_bull = rsi_div.get('hidden_bullish_divergence', False)
            hid_bear = rsi_div.get('hidden_bearish_divergence', False)
            mtf_conf = rsi_div.get('mtf_confirmation', False)
            
            div_signals = []
            if reg_bull: div_signals.append("REGULAR BULLISH")
            if reg_bear: div_signals.append("REGULAR BEARISH")
            if hid_bull: div_signals.append("HIDDEN BULLISH")
            if hid_bear: div_signals.append("HIDDEN BEARISH")
            
            if div_signals:
                mtf_str = " (MTF CONFIRMED!)" if mtf_conf else ""
                lines.append(f"📊 RSI DIVERGENCE: {', '.join(div_signals)}{mtf_str} (RSI={rsi:.1f})")
            else:
                lines.append(f"📊 RSI: {rsi:.1f} (No divergence detected)")
        
        # Other key metrics
        hurst = detailed.get('hurst_exponent')
        if hurst is not None:
            regime = "TRENDING" if hurst > 0.55 else "MEAN-REVERTING" if hurst < 0.45 else "RANDOM"
            lines.append(f"📊 HURST EXPONENT: {hurst:.3f} ({regime})")
        
        zscore = detailed.get('z_score')
        if zscore is not None:
            lines.append(f"📊 PRICE Z-SCORE: {zscore:.2f}σ")
        
        garch_vol = detailed.get('garch_vol')
        if garch_vol is not None:
            vol_trend = detailed.get('vol_trend', 'stable')
            lines.append(f"📊 GARCH VOLATILITY: {garch_vol:.2%} ({vol_trend})")
        
        sharpe = detailed.get('sharpe_ratio')
        sortino = detailed.get('sortino_ratio')
        if sharpe is not None:
            sortino_str = f"{sortino:.2f}" if sortino else "N/A"
            lines.append(f"📊 RISK-ADJUSTED: Sharpe={sharpe:.2f}, Sortino={sortino_str}")
        
        lines.append("")
        
        # Escape curly braces in reasons to prevent f-string format issues when used in prompts
        def safe_reason(r):
            return str(r).replace('{', '(').replace('}', ')')
        
        # Reasons FOR
        if reasons_for:
            lines.append("✅ REASONS FOR TRADE:")
            for i, reason in enumerate(reasons_for[:5], 1):
                lines.append(f"   {i}. {safe_reason(reason)}")
        else:
            lines.append("✅ REASONS FOR TRADE: None identified")
        
        lines.append("")
        
        # Reasons AGAINST
        if reasons_against:
            lines.append("❌ REASONS AGAINST TRADE:")
            for i, reason in enumerate(reasons_against[:5], 1):
                lines.append(f"   {i}. {safe_reason(reason)}")
        else:
            lines.append("❌ REASONS AGAINST TRADE: None identified")
        
        lines.append("")
        
        return "\n".join(lines) + "\n"

    def analyze_signal(
        self,
        signal: int,  # 1 = long, -1 = short, 0 = none
        df: pd.DataFrame,
        current_price: float,
        atr: float,
        symbol: str,
        ml_insight: Dict[str, Any] = None,
        system_score: Dict[str, Any] = None,  # Combined system scoring
        market_scanner_context: Dict[str, Any] = None,  # Market scanner recommendation
        tech_score: Dict[str, Any] = None,  # Technical score breakdown
        institutional_data: Dict[str, Any] = None  # Real exchange microstructure data
    ) -> Dict[str, Any]:
        """
        Analyze a trading signal and return AI validation result.
        AI is the FINAL DECISION MAKER in the autonomous pipeline.
        
        Decision Pipeline: Signal → ML → AI (final)
        
        Args:
            ml_insight: Dict with ML prediction data
            system_score: Dict with combined system scoring:
                - combined: 0-100 overall score
                - recommendation: STRONG_BUY/BUY/NEUTRAL/WEAK/AVOID
                - breakdown: Component scores
            market_scanner_context: Dict with market scanner data:
                - best_pair: Recommended pair from scanner
                - current_pair_rank: Rank of current pair in scanner
            tech_score: Dict with technical score breakdown:
                - score: 0-100 technical quality
                - quality: EXCELLENT/GOOD/MODERATE/WEAK/POOR
                - factors: List of notable factors
                - breakdown: Component scores
        
        Returns:
            Dict with keys: approved, confidence, reasoning, risk_assessment
        """
        if signal == 0:
            return {
                "approved": False,
                "confidence": 0.0,
                "reasoning": "No signal to analyze",
                "risk_assessment": "N/A"
            }
        
        # Gather market context
        context = self._build_market_context(df, current_price, atr)
        
        # Add ML insight and system score to context
        context['ml_insight'] = ml_insight or {'ml_available': False}
        context['system_score'] = system_score or {'combined': 50, 'recommendation': 'NEUTRAL'}
        context['market_scanner'] = market_scanner_context or {}
        context['tech_score'] = tech_score or {'score': 50, 'quality': 'UNKNOWN', 'factors': []}
        context['institutional_data'] = institutional_data or {}
        
        # === PHASE 1: MATHEMATICAL PRE-CHECK ===
        # Calculate objective math score before AI decision
        math_check = self._comprehensive_math_check(signal, df, current_price, atr, context)
        context['math_check'] = math_check
        
        if self.use_ai:
            result = self._ai_analysis(signal, context, symbol)
            
            # === PHASE 2: MATH VALIDATION OF AI DECISION ===
            # If AI wants to block but math says it's a good trade, override
            result = self._validate_ai_decision_with_math(result, math_check, signal, symbol)
            
            # === PHASE 3: AI POWER RECOMMENDATIONS ===
            # Add AI-powered sizing and aggressiveness recommendations
            result = self._add_ai_power_recommendations(result, math_check, context, signal)
        else:
            result = self._rule_based_analysis(signal, context, symbol)
        
        # === ATTACH COMPONENT SCORES FOR BAYESIAN LEARNING ===
        # These get stored on the Position object and used at trade close
        # to update weights based on which components predicted correctly.
        result['entry_component_scores'] = math_check.get('scores', {})
        
        # Cache on instance for RiskShield to read tail_risk between signals
        self._last_component_scores = result['entry_component_scores']
        
        # Log the analysis
        self._log_analysis(signal, symbol, result)
        
        return result
    
    def _add_ai_power_recommendations(
        self,
        result: Dict[str, Any],
        math_check: Dict[str, Any],
        context: Dict[str, Any],
        signal: int
    ) -> Dict[str, Any]:
        """
        Add AI power recommendations for position sizing and trade aggressiveness.
        This gives the AI more control over the trade execution.
        """
        confidence = result.get('confidence', 0.5)
        math_score = math_check.get('score', 50)
        approved = result.get('approved', False)
        
        if not approved:
            result['ai_power'] = {
                'size_multiplier': 0.0,
                'aggressive_mode': False,
                'conviction': 'BLOCKED'
            }
            return result
        
        # === CONFIDENCE-BASED POSITION SIZING ===
        # High confidence = larger position, low confidence = smaller position
        if confidence >= 0.90 and math_score >= 80:
            size_mult = 1.5  # 50% larger position for high conviction trades
            conviction = 'VERY_HIGH'
            aggressive = True
        elif confidence >= 0.85 and math_score >= 70:
            size_mult = 1.3  # 30% larger
            conviction = 'HIGH'
            aggressive = True
        elif confidence >= 0.80 and math_score >= 65:
            size_mult = 1.15  # 15% larger
            conviction = 'GOOD'
            aggressive = False
        elif confidence >= 0.75:
            size_mult = 1.0  # Standard size
            conviction = 'NORMAL'
            aggressive = False
        elif confidence >= 0.70:
            size_mult = 0.75  # 25% smaller for borderline trades
            conviction = 'LOW'
            aggressive = False
        else:
            size_mult = 0.5  # 50% smaller for low confidence
            conviction = 'MINIMAL'
            aggressive = False
        
        # === ADJUST FOR RECENT PERFORMANCE ===
        # Check consecutive results
        recent_wins = sum(1 for t in self.recent_trades[-5:] if t.get('result', '').lower() == 'win')
        recent_losses = 5 - recent_wins if len(self.recent_trades) >= 5 else 0
        
        # Hot streak boost
        if recent_wins >= 4:
            size_mult = min(size_mult * 1.2, 2.0)  # Cap at 2x
            logger.info(f"🔥 AI HOT STREAK: {recent_wins} wins - size boost to {size_mult:.1f}x")
        
        # Cold streak protection
        elif recent_losses >= 3:
            size_mult = max(size_mult * 0.7, 0.5)  # Floor at 0.5x
            aggressive = False
            logger.info(f"🧊 AI COLD STREAK: {recent_losses} losses - size reduced to {size_mult:.1f}x")
        
        result['ai_power'] = {
            'size_multiplier': round(size_mult, 2),
            'aggressive_mode': aggressive,
            'conviction': conviction,
            'math_score': math_score,
            'confidence': confidence,
            'recent_streak': f"{recent_wins}W/{recent_losses}L"
        }
        
        signal_type = "LONG" if signal == 1 else "SHORT"
        logger.info(f"🤖 AI POWER: {signal_type} | Conviction={conviction} | Size={size_mult:.1f}x | Aggressive={aggressive}")
        
        return result
    
    def ai_strategic_analysis(
        self,
        balance: float,
        recent_trades: List[Dict],
        current_positions: List[Dict],
        market_conditions: Dict[str, Any],
        adaptive_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        AI Strategic Market Analysis - Makes high-level trading decisions.
        
        This gives the AI MORE POWER to:
        1. Recommend parameter adjustments based on market regime
        2. Suggest position sizing changes
        3. Decide aggressive vs conservative mode
        4. Recommend symbol switches
        5. Set overall trading bias (long/short/neutral)
        
        Returns strategic recommendations that the bot can act on.
        """
        if not self.use_ai:
            return {'available': False, 'reason': 'AI not enabled'}
        
        try:
            # Build comprehensive context
            recent_pnl = sum(t.get('pnl', 0) for t in recent_trades[-10:]) if recent_trades else 0
            win_count = sum(1 for t in recent_trades[-10:] if t.get('pnl', 0) > 0) if recent_trades else 0
            loss_count = len(recent_trades[-10:]) - win_count if recent_trades else 0
            win_rate = win_count / max(len(recent_trades[-10:]), 1) * 100
            
            # Format current params
            params_str = "\n".join([
                f"- {name}: {info.get('current', 'N/A')} (range: {info.get('min', 'N/A')}-{info.get('max', 'N/A')})"
                for name, info in adaptive_params.items()
            ])
            
            # Format positions
            positions_str = "No open positions"
            if current_positions:
                positions_str = "\n".join([
                    f"- {p.get('symbol', 'N/A')}: {p.get('side', 'N/A')} @ ${p.get('entry', 0):.4f}, PnL: {p.get('pnl_pct', 0):.2f}%"
                    for p in current_positions
                ])
            
            prompt = f"""You are Julaba's Strategic AI Brain with FULL DECISION-MAKING POWER.

═══════════════════════════════════════════════════════════════════════
ACCOUNT STATUS
═══════════════════════════════════════════════════════════════════════
Balance: ${balance:.2f}
Recent PnL (10 trades): ${recent_pnl:.2f}
Win Rate: {win_rate:.1f}% ({win_count}W / {loss_count}L)

═══════════════════════════════════════════════════════════════════════
CURRENT POSITIONS
═══════════════════════════════════════════════════════════════════════
{positions_str}

═══════════════════════════════════════════════════════════════════════
MARKET CONDITIONS
═══════════════════════════════════════════════════════════════════════
BTC Trend: {market_conditions.get('btc_trend', 'UNKNOWN')}
Market Volatility: {market_conditions.get('volatility', 'UNKNOWN')}
Overall Sentiment: {market_conditions.get('sentiment', 'NEUTRAL')}

═══════════════════════════════════════════════════════════════════════
CURRENT PARAMETERS (YOU CAN ADJUST)
═══════════════════════════════════════════════════════════════════════
{params_str}

═══════════════════════════════════════════════════════════════════════
YOUR STRATEGIC DECISIONS
═══════════════════════════════════════════════════════════════════════
Make strategic decisions to optimize trading performance.

Respond ONLY with JSON:
{{
    "trading_mode": "aggressive" or "normal" or "conservative" or "defensive",
    "bias": "long" or "short" or "neutral",
    "risk_adjustment": -0.02 to +0.02 (change to risk_pct),
    "max_positions": 1 to 4,
    "reasoning": "brief strategic rationale (max 100 chars)",
    "urgent_action": null or {{"action": "close_all" or "reduce_size", "reason": "why"}},
    "param_changes": [
        {{"param": "param_name", "new_value": X, "reason": "why"}}
    ]
}}"""

            result_text = self._generate_content(prompt)
            if not result_text:
                return {'available': False, 'reason': 'AI returned empty response'}
            
            result_text = result_text.strip()
            
            # Parse JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            strategy = json.loads(result_text.strip())
            
            logger.info(f"🧠 AI STRATEGIC ANALYSIS:")
            logger.info(f"   Mode: {strategy.get('trading_mode', 'N/A')}")
            logger.info(f"   Bias: {strategy.get('bias', 'N/A')}")
            logger.info(f"   Reasoning: {strategy.get('reasoning', 'N/A')}")
            
            if strategy.get('urgent_action'):
                logger.warning(f"⚠️ AI URGENT ACTION: {strategy['urgent_action']}")
            
            return {
                'available': True,
                'strategy': strategy,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"AI strategic analysis failed: {e}")
            return {'available': False, 'reason': str(e)}
    
    def unified_math_ai_scan(
        self,
        pairs_data: List[Dict[str, Any]],
        current_positions: List[Dict[str, Any]],
        balance: float,
        max_positions: int = 2,
        allowed_sides: str = 'both',
        atomic_candle=None
    ) -> Dict[str, Any]:
        """
        UNIFIED MATH + AI SCANNER - 3-STAGE PIPELINE
        
        ═══════════════════════════════════════════════════════════════
        STAGE 1: PROACTIVE PRE-FILTER (Quick elimination)
        ═══════════════════════════════════════════════════════════════
        - Fast math check to eliminate obviously bad pairs
        - Check momentum, trend exhaustion, basic thresholds
        - Reduces candidates from 50 → ~5-10
        
        ═══════════════════════════════════════════════════════════════
        STAGE 2: UNIFIED MATH ANALYSIS (Deep analysis)
        ═══════════════════════════════════════════════════════════════
        - Comprehensive PhD math scoring on remaining pairs
        - Both LONG and SHORT directions analyzed
        - Statistical edge, GARCH, Hurst, trend analysis
        - Ranks pairs by math score
        
        ═══════════════════════════════════════════════════════════════
        STAGE 3: AI FINAL DECISION (Strategic validation)
        ═══════════════════════════════════════════════════════════════
        - AI reviews top 3 math candidates
        - Can APPROVE, REVERSE direction, or REJECT
        - Final combined score (Math + AI weighted)
        - Returns BEST opportunity or None
        
        Args:
            pairs_data: List of {symbol, df, price, atr, volume_ratio} for each pair
            current_positions: List of open positions
            balance: Current account balance
            max_positions: Maximum allowed concurrent positions
            
        Returns:
            {
                'has_opportunity': bool,
                'best_pair': {symbol, signal, math_score, ai_score, combined_score, ...},
                'ranked_pairs': [...],  # All pairs ranked by combined score
                'ai_bias': 'long' | 'short' | 'neutral',
                'market_regime': str,
                'recommended_action': str,
                'pipeline_stats': {stage1_input, stage1_output, stage2_output, stage3_output}
            }
        """
        try:
            pipeline_stats = {
                'stage1_input': len(pairs_data),
                'stage1_output': 0,
                'stage2_output': 0,
                'stage3_output': 0
            }
            
            logger.info(f"═══════════════════════════════════════════════════════════════")
            logger.info(f"🔬 UNIFIED SCAN PIPELINE: {len(pairs_data)} pairs starting...")
            logger.info(f"═══════════════════════════════════════════════════════════════")
            
            # Check if we can open more positions
            open_count = len([p for p in current_positions if p])
            if open_count >= max_positions:
                return {
                    'has_opportunity': False,
                    'reason': f'Max positions reached ({open_count}/{max_positions})',
                    'ranked_pairs': [],
                    'pipeline_stats': pipeline_stats
                }
            
            # Symbols we already have positions on
            open_symbols = set()
            for pos in current_positions:
                if pos:
                    # Handle both dict and Position object
                    if hasattr(pos, 'symbol'):
                        sym = pos.symbol  # Position object
                    elif isinstance(pos, dict):
                        sym = pos.get('symbol', '')
                    else:
                        sym = str(pos)
                    open_symbols.add(sym.replace('/', '').replace(':USDT', '').upper())
            
            # ═══════════════════════════════════════════════════════════════
            # PRE-STAGE: GLOBAL GUARDS (cooldown, daily limit)
            # ═══════════════════════════════════════════════════════════════
            
            # CHECK 1: Global loss cooldown — any recent loss pauses all trading
            if self._global_cooldown_until:
                now_gc = datetime.now(timezone.utc)
                if now_gc < self._global_cooldown_until:
                    remaining_gc = (self._global_cooldown_until - now_gc).total_seconds() / 60
                    logger.info(f"⏸️ GLOBAL COOLDOWN active: {remaining_gc:.0f}min remaining")
                    return {
                        'has_opportunity': False,
                        'reason': f'GLOBAL COOLDOWN: {remaining_gc:.0f}min remaining after recent loss',
                        'ranked_pairs': [],
                        'pipeline_stats': pipeline_stats
                    }
                else:
                    self._global_cooldown_until = None  # Expired
                    logger.info("✅ Global cooldown expired — scanning resumed")
            
            # CHECK 2: Daily trade limit
            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            if today != self._daily_trade_date:
                self._daily_trade_date = today
                self._daily_trade_count = 0
            if self._daily_trade_count >= self.MAX_TRADES_PER_DAY:
                logger.info(f"🚫 DAILY LIMIT: {self._daily_trade_count}/{self.MAX_TRADES_PER_DAY} trades today — no more trading")
                return {
                    'has_opportunity': False,
                    'reason': f'DAILY LIMIT: {self._daily_trade_count}/{self.MAX_TRADES_PER_DAY} trades today',
                    'ranked_pairs': [],
                    'pipeline_stats': pipeline_stats
                }
            
            # ═══════════════════════════════════════════════════════════════
            # STAGE 1: PROACTIVE PRE-FILTER (Quick elimination)
            # ═══════════════════════════════════════════════════════════════
            logger.info(f"📋 STAGE 1: Proactive Pre-filter ({len(pairs_data)} pairs)")
            
            stage1_passed = []
            stage1_rejected = []
            
            for pair in pairs_data:
                symbol = pair.get('symbol', '')
                symbol_key = symbol.replace('/', '').replace(':USDT', '').upper()
                
                # Skip pairs we already have positions on
                if symbol_key in open_symbols:
                    stage1_rejected.append((symbol, "Already in position"))
                    continue
                
                # ═══════════════════════════════════════════════════════════════
                # PER-SYMBOL COOLDOWN: Skip symbols that recently had losses
                # Prevents repeatedly trading the same losing symbol
                # ═══════════════════════════════════════════════════════════════
                on_cooldown, cooldown_reason, mins_remaining = self._is_symbol_on_cooldown(symbol)
                if on_cooldown:
                    stage1_rejected.append((symbol, f"COOLDOWN: {cooldown_reason} ({mins_remaining}min left)"))
                    logger.info(f"   🚫 {symbol}: On cooldown ({cooldown_reason}, {mins_remaining}min remaining)")
                    continue
                
                df = pair.get('df')
                price = pair.get('price', 0)
                atr = pair.get('atr', 0)
                
                if df is None or len(df) < 20:
                    stage1_rejected.append((symbol, "Insufficient data"))
                    continue
                if price <= 0 or atr <= 0:
                    stage1_rejected.append((symbol, "Invalid price/ATR"))
                    continue
                
                # === ALLOWED_SIDES FILTER: Skip blocked directions early ===
                # Zero out scores for blocked sides to avoid wasting compute
                _allowed = allowed_sides.lower() if isinstance(allowed_sides, str) else 'both'
                
                # Quick math scores (use dashboard pre-calculated or calculate fresh)
                dashboard_math_long = pair.get('dashboard_math_long', 0)
                dashboard_math_short = pair.get('dashboard_math_short', 0)
                
                # Zero out blocked directions
                if _allowed == 'long':
                    dashboard_math_short = 0
                elif _allowed == 'short':
                    dashboard_math_long = 0
                
                # FIRST: Build context to get momentum info
                context = self._build_market_context(df, price, atr)
                momentum_5 = context.get('roc_5', 0)
                momentum_10 = context.get('roc_10', 0)
                rsi = context.get('rsi', 50)
                adx = context.get('adx', 25)
                
                # ═══════════════════════════════════════════════════════════════
                # MOMENTUM-FIRST SCORING: Simple and effective
                # Forget PhD math when momentum is clear - just follow the trend!
                # ═══════════════════════════════════════════════════════════════
                
                # Calculate simple momentum-based scores
                # FIXED: Start at 40, require momentum to EARN points
                momentum_long_score = 40  # Lower base - must earn it
                momentum_short_score = 40
                
                # FIXED: Higher thresholds (0.3 → 0.5) and stronger scoring
                # LONG benefits from bullish momentum
                if momentum_5 > 0.5:
                    momentum_long_score += min(35, momentum_5 * 25)  # Up to +35 for strong up
                elif momentum_5 > 0.3:
                    momentum_long_score += min(20, momentum_5 * 15)  # Moderate boost
                if momentum_10 > 0.3:
                    momentum_long_score += min(20, momentum_10 * 12)
                if momentum_5 < -0.5:
                    momentum_long_score -= min(40, abs(momentum_5) * 30)  # STRONGER penalty for counter-trend
                elif momentum_5 < -0.3:
                    momentum_long_score -= min(25, abs(momentum_5) * 20)
                    
                # SHORT benefits from bearish momentum
                if momentum_5 < -0.5:
                    momentum_short_score += min(35, abs(momentum_5) * 25)
                elif momentum_5 < -0.3:
                    momentum_short_score += min(20, abs(momentum_5) * 15)
                if momentum_10 < -0.3:
                    momentum_short_score += min(20, abs(momentum_10) * 12)
                if momentum_5 > 0.5:
                    momentum_short_score -= min(40, momentum_5 * 30)  # STRONGER penalty
                elif momentum_5 > 0.3:
                    momentum_short_score -= min(25, momentum_5 * 20)
                    
                # RSI extremes boost reversal trades AND penalize wrong direction
                if rsi < 30:  # Oversold - LONG gets boost, SHORT gets penalty
                    momentum_long_score += 20
                    momentum_short_score -= 25  # FIXED: Penalize SHORT in oversold
                elif rsi < 40:
                    momentum_long_score += 10
                    momentum_short_score -= 10
                if rsi > 70:  # Overbought - SHORT gets boost, LONG gets penalty
                    momentum_short_score += 20
                    momentum_long_score -= 25  # FIXED: Penalize LONG in overbought
                elif rsi > 60:
                    momentum_short_score += 10
                    momentum_long_score -= 10
                    
                # ADX confirms trend strength - STRONGER differentiation
                if adx > 30:  # Strong trend
                    if momentum_5 > 0.3:
                        momentum_long_score += 15
                        momentum_short_score -= 10  # Penalize SHORT in strong uptrend
                    elif momentum_5 < -0.3:
                        momentum_short_score += 15
                        momentum_long_score -= 10  # Penalize LONG in strong downtrend
                elif adx > 25:  # Moderate trend
                    if momentum_5 > 0:
                        momentum_long_score += 8
                    elif momentum_5 < 0:
                        momentum_short_score += 8
                
                # === ENFORCE ALLOWED_SIDES ON MOMENTUM SCORES ===
                if _allowed == 'long':
                    momentum_short_score = 0
                elif _allowed == 'short':
                    momentum_long_score = 0
                
                # Use momentum scores as primary, PhD math as secondary
                if dashboard_math_long > 0 or dashboard_math_short > 0:
                    # Blend: 60% momentum, 40% PhD math
                    long_score = (momentum_long_score * 0.6) + (dashboard_math_long * 0.4)
                    short_score = (momentum_short_score * 0.6) + (dashboard_math_short * 0.4)
                else:
                    # If no dashboard scores, use momentum-only
                    long_score = momentum_long_score
                    short_score = momentum_short_score
                
                logger.debug(f"   {symbol}: Momentum scores L={momentum_long_score:.0f}/S={momentum_short_score:.0f}, Final L={long_score:.0f}/S={short_score:.0f}")
                
                # PRE-FILTER 1: At least one direction must be viable (score >= 55)
                # RAISED from 40→55: Be ultra-selective, only trade strong signals
                STAGE1_MIN_SCORE = 55
                best_score = max(long_score, short_score)
                if best_score < STAGE1_MIN_SCORE:
                    stage1_rejected.append((symbol, f"Math too low ({best_score:.0f}<{STAGE1_MIN_SCORE})"))
                    continue
                
                # PRE-FILTER 2: Quick momentum check - STRICT VERSION
                # Don't enter against immediate momentum!
                context = self._build_market_context(df, price, atr)
                momentum_5 = context.get('roc_5', 0)  # Use roc_5 (rate of change over 5 bars)
                momentum_10 = context.get('roc_10', 0)  # Use roc_10 (rate of change over 10 bars)
                rsi = context.get('rsi', 50)
                
                # Block LONG into ANY bearish momentum
                # STRICT: Block if ROC5 < -0.3% OR ROC10 < -0.2%
                if long_score > short_score and (momentum_5 < -0.3 or momentum_10 < -0.2):
                    stage1_rejected.append((symbol, f"Bearish momentum (ROC5={momentum_5:+.1f}%, ROC10={momentum_10:+.1f}%) blocks LONG"))
                    continue
                # Block SHORT into ANY bullish momentum  
                # STRICT: Block if ROC5 > 0.3% OR ROC10 > 0.2%
                if short_score > long_score and (momentum_5 > 0.3 or momentum_10 > 0.2):
                    stage1_rejected.append((symbol, f"Bullish momentum (ROC5={momentum_5:+.1f}%, ROC10={momentum_10:+.1f}%) blocks SHORT"))
                    continue
                
                # PRE-FILTER 3: RSI trend exhaustion check
                if short_score > long_score and rsi <= 30:
                    stage1_rejected.append((symbol, f"RSI oversold ({rsi:.0f}) blocks SHORT"))
                    continue
                if long_score > short_score and rsi >= 70:
                    stage1_rejected.append((symbol, f"RSI overbought ({rsi:.0f}) blocks LONG"))
                    continue
                
                # PRE-FILTER 4: ADX minimum — only trade when there IS a trend
                # ADX < 20 = choppy/ranging market = signals are unreliable
                ADX_MINIMUM = 20
                if adx < ADX_MINIMUM:
                    stage1_rejected.append((symbol, f"ADX too low ({adx:.0f}<{ADX_MINIMUM}) — no trend"))
                    continue
                
                # PRE-FILTER 5: CONFIRMATION CANDLE — soft penalty instead of hard block
                # FIXED: Was hard-blocking ~50% of valid signals. Now applies score penalty
                # so strong setups can still pass even without perfect candle alignment
                try:
                    prev_candle = df.iloc[-2]  # Most recent COMPLETED candle
                    candle_bullish = prev_candle['close'] > prev_candle['open']
                    candle_bearish = prev_candle['close'] < prev_candle['open']
                    
                    CANDLE_PENALTY = 8  # Deduct points instead of hard block
                    if long_score > short_score and not candle_bullish:
                        long_score -= CANDLE_PENALTY
                        logger.debug(f"   {symbol}: No LONG confirmation candle → score penalty -{CANDLE_PENALTY}")
                    if short_score > long_score and not candle_bearish:
                        short_score -= CANDLE_PENALTY
                        logger.debug(f"   {symbol}: No SHORT confirmation candle → score penalty -{CANDLE_PENALTY}")
                    
                    # Re-check: if BOTH directions now below minimum, reject
                    if max(long_score, short_score) < STAGE1_MIN_SCORE:
                        stage1_rejected.append((symbol, f"Score too low after candle penalty (L={long_score:.0f}/S={short_score:.0f}<{STAGE1_MIN_SCORE})"))
                        continue
                except (IndexError, KeyError):
                    pass  # Skip check if data insufficient
                
                # Passed Stage 1 - keep for Stage 2
                stage1_passed.append({
                    'pair': pair,
                    'long_score': long_score,
                    'short_score': short_score,
                    'context': context
                })
            
            pipeline_stats['stage1_output'] = len(stage1_passed)
            logger.info(f"📋 STAGE 1 RESULT: {len(stage1_passed)}/{len(pairs_data)} passed pre-filter")
            if stage1_rejected:
                reject_summary = {}
                for sym, reason in stage1_rejected[:10]:
                    if reason not in reject_summary:
                        reject_summary[reason] = 0
                    reject_summary[reason] += 1
                for reason, count in reject_summary.items():
                    logger.debug(f"   Rejected ({count}): {reason}")
            
            if not stage1_passed:
                return {
                    'has_opportunity': False,
                    'reason': 'STAGE 1: No pairs passed proactive pre-filter',
                    'ranked_pairs': [],
                    'pipeline_stats': pipeline_stats
                }
            
            # ═══════════════════════════════════════════════════════════════
            # STAGE 2: UNIFIED MATH ANALYSIS (Deep analysis)
            # ═══════════════════════════════════════════════════════════════
            logger.info(f"📊 STAGE 2: Deep Math Analysis ({len(stage1_passed)} pairs)")
            
            math_ranked = []
            
            for item in stage1_passed:
                pair = item['pair']
                symbol = pair.get('symbol', '')
                df = pair.get('df')
                price = pair.get('price', 0)
                atr = pair.get('atr', 0)
                context = item['context']
                long_score = item['long_score']
                short_score = item['short_score']
                
                # Stage 2: Determine best direction based on MOMENTUM ALIGNMENT
                # Threshold controlled by /threshold command (default vs loose)
                market_timing = self._get_market_hours_context()
                _active_thr = get_active_thresholds(market_timing.get('is_weekend', False))
                STAGE2_MIN_SCORE = _active_thr['stage2']
                
                # ═══════════════════════════════════════════════════════════
                # REGIME-AWARE THRESHOLD (Rec #4 integration)
                # In strong trending regimes, trend trades have a real edge
                # but score lower because mean-reversion components (zscore,
                # hurst, autocorr) fight the trend. Lower threshold by 5 pts
                # to let more candidates through to Stage 3 for AI review.
                # ═══════════════════════════════════════════════════════════
                _s2_adx = context.get('adx', 25)
                _s2_roc5 = context.get('roc_5', 0)
                _s2_roc10 = context.get('roc_10', 0)
                _regime_is_trending = _s2_adx >= 28 and (abs(_s2_roc5) > 0.4 or abs(_s2_roc10) > 0.3)
                if _regime_is_trending:
                    STAGE2_MIN_SCORE = max(35, STAGE2_MIN_SCORE - 5)
                    logger.debug(f"   📊 REGIME: Trending (ADX={_s2_adx:.0f}) → Stage2 threshold lowered to {STAGE2_MIN_SCORE}")
                
                # Get momentum from context
                momentum_5 = context.get('roc_5', 0)
                momentum_10 = context.get('roc_10', 0)
                rsi = context.get('rsi', 50)
                
                # ═══════════════════════════════════════════════════════════════
                # SIMPLE RULE: GO WITH MOMENTUM!
                # If momentum is bullish → LONG, if bearish → SHORT
                # Only reject if the aligned direction score is too low
                # FIXED: Increased thresholds to avoid trading on noise
                # ═══════════════════════════════════════════════════════════════
                
                # FIXED: 0.2% → 0.5% for ROC5, 0.1% → 0.3% for ROC10
                # Tiny moves are noise, need REAL momentum to pick direction
                bullish_momentum = momentum_5 > 0.5 or (momentum_5 > 0.2 and momentum_10 > 0.3)
                bearish_momentum = momentum_5 < -0.5 or (momentum_5 < -0.2 and momentum_10 < -0.3)
                
                # ═══════════════════════════════════════════════════════════════
                # MICRO-STRUCTURE CHECK: Override momentum if recent bars already reversed
                # ROC5 is LAGGING — uses 5 bars of history. If the last 2-3 bars have
                # already reversed (e.g., pump→dump), ROC5 still shows positive but
                # entering LONG is suicide. Check the MOST RECENT bars directly.
                # This prevents "buying the top" / "shorting the bottom" scenarios.
                # ═══════════════════════════════════════════════════════════════
                micro_reversed = False
                micro_reason = ""
                try:
                    if len(df) >= 4:
                        bar_0 = df.iloc[-1]  # Current forming bar
                        bar_1 = df.iloc[-2]  # Last completed bar
                        bar_2 = df.iloc[-3]  # 2 bars ago
                        
                        # Calculate recent bar returns
                        ret_0 = (bar_0['close'] - bar_0['open']) / bar_0['open'] * 100 if bar_0['open'] > 0 else 0
                        ret_1 = (bar_1['close'] - bar_1['open']) / bar_1['open'] * 100 if bar_1['open'] > 0 else 0
                        
                        # Upper wick ratio: upper_wick / body — measures rejection
                        body_0 = abs(bar_0['close'] - bar_0['open'])
                        upper_wick_0 = bar_0['high'] - max(bar_0['close'], bar_0['open'])
                        lower_wick_0 = min(bar_0['close'], bar_0['open']) - bar_0['low']
                        bar_range_0 = bar_0['high'] - bar_0['low'] if bar_0['high'] > bar_0['low'] else 0.0001
                        
                        if bullish_momentum:
                            # Want to go LONG, but check if price is already dumping
                            # Condition: last 2 bars are BOTH red (close < open) = reversal in progress
                            both_red = ret_0 < -0.02 and ret_1 < -0.02
                            # OR: current bar has strong upper wick rejection (bought up then sold off)
                            upper_wick_rejection = upper_wick_0 > 2 * body_0 and ret_0 < 0 and upper_wick_0 / bar_range_0 > 0.5
                            # OR: price falling from recent high (last 2 bars declining closes)
                            declining_closes = bar_0['close'] < bar_1['close'] < bar_2['close']
                            
                            if both_red and declining_closes:
                                micro_reversed = True
                                micro_reason = f"2 red bars + declining closes (ret0={ret_0:+.2f}%, ret1={ret_1:+.2f}%)"
                            elif both_red and upper_wick_rejection:
                                micro_reversed = True
                                micro_reason = f"2 red bars + upper wick rejection (wick={upper_wick_0/bar_range_0:.0%})"
                            elif ret_0 < -0.15 and upper_wick_rejection:
                                micro_reversed = True
                                micro_reason = f"Strong red bar + upper wick rejection (ret0={ret_0:+.2f}%)"
                                
                        elif bearish_momentum:
                            # Want to go SHORT, but check if price is already pumping
                            both_green = ret_0 > 0.02 and ret_1 > 0.02
                            lower_wick_rejection = lower_wick_0 > 2 * body_0 and ret_0 > 0 and lower_wick_0 / bar_range_0 > 0.5
                            rising_closes = bar_0['close'] > bar_1['close'] > bar_2['close']
                            
                            if both_green and rising_closes:
                                micro_reversed = True
                                micro_reason = f"2 green bars + rising closes (ret0={ret_0:+.2f}%, ret1={ret_1:+.2f}%)"
                            elif both_green and lower_wick_rejection:
                                micro_reversed = True
                                micro_reason = f"2 green bars + lower wick rejection (wick={lower_wick_0/bar_range_0:.0%})"
                            elif ret_0 > 0.15 and lower_wick_rejection:
                                micro_reversed = True
                                micro_reason = f"Strong green bar + lower wick rejection (ret0={ret_0:+.2f}%)"
                except (IndexError, KeyError):
                    pass
                
                if micro_reversed:
                    logger.warning(f"   🔄 {symbol}: MICRO-STRUCTURE REVERSAL — {micro_reason}")
                    logger.warning(f"      ROC5={momentum_5:+.2f}% says {'LONG' if bullish_momentum else 'SHORT'} but recent bars say NO → SKIP")
                    continue
                
                # ═══════════════════════════════════════════════════════════════
                # ATOMIC DIRECTION INFLUENCE: Use tick-level velocity to
                # OVERRIDE momentum direction when sub-candle data strongly
                # contradicts ROC5. ROC5 is a lagging 5-bar indicator;
                # atomic velocity sees what's happening RIGHT NOW at tick level.
                # This catches direction flips that candles haven't printed yet.
                # ═══════════════════════════════════════════════════════════════
                atomic_direction_flipped = False
                if atomic_candle is not None:
                    try:
                        _atom = atomic_candle.get_analysis(symbol)
                        if _atom is not None:
                            _vel = _atom.velocity            # %/sec — positive=price rising, negative=falling
                            _dir_cons = _atom.direction_consistency  # 0-100 — how consistently ticks move one way
                            _path_eff = _atom.path_efficiency        # 0-1 — straight line vs choppy
                            _micro_trend = _atom.micro_trend          # "strong_up", "up", "neutral", "down", "strong_down"
                            _entry_q = _atom.entry_quality            # 0-100
                            
                            # CASE 1: ROC5 says LONG but ticks are aggressively falling
                            # Velocity strongly negative + high consistency = real selling pressure
                            if bullish_momentum and not bearish_momentum:
                                _vel_contradicts = _vel < -0.003  # Strong negative velocity
                                _consistent_down = _dir_cons > 60 and _micro_trend in ("down", "strong_down")
                                _efficient_down = _path_eff > 0.4 and _vel < 0  # Clean downward path
                                
                                if _vel_contradicts and _consistent_down:
                                    # Strong contradiction — flip to SHORT
                                    logger.warning(f"   🔬 {symbol}: ATOMIC DIRECTION FLIP: ROC5 says LONG but ticks aggressively FALLING")
                                    logger.warning(f"      vel={_vel:.5f}%/s, dir_consistency={_dir_cons:.0f}%, micro_trend={_micro_trend}, path_eff={_path_eff:.2f}")
                                    bullish_momentum = False
                                    bearish_momentum = True
                                    atomic_direction_flipped = True
                                elif _efficient_down and _entry_q < 30:
                                    # Moderate contradiction + poor entry quality — neutralize (skip)
                                    logger.warning(f"   🔬 {symbol}: ATOMIC DIRECTION NEUTRALIZE: Ticks falling + poor entry quality ({_entry_q:.0f})")
                                    logger.warning(f"      vel={_vel:.5f}%/s, path_eff={_path_eff:.2f}, entry_quality={_entry_q:.0f}")
                                    bullish_momentum = False
                                    # Don't set bearish — let it fall to "no clear momentum" skip
                            
                            # CASE 2: ROC5 says SHORT but ticks are aggressively rising
                            elif bearish_momentum and not bullish_momentum:
                                _vel_contradicts = _vel > 0.003  # Strong positive velocity
                                _consistent_up = _dir_cons > 60 and _micro_trend in ("up", "strong_up")
                                _efficient_up = _path_eff > 0.4 and _vel > 0
                                
                                if _vel_contradicts and _consistent_up:
                                    logger.warning(f"   🔬 {symbol}: ATOMIC DIRECTION FLIP: ROC5 says SHORT but ticks aggressively RISING")
                                    logger.warning(f"      vel={_vel:.5f}%/s, dir_consistency={_dir_cons:.0f}%, micro_trend={_micro_trend}, path_eff={_path_eff:.2f}")
                                    bearish_momentum = False
                                    bullish_momentum = True
                                    atomic_direction_flipped = True
                                elif _efficient_up and _entry_q < 30:
                                    logger.warning(f"   🔬 {symbol}: ATOMIC DIRECTION NEUTRALIZE: Ticks rising + poor entry quality ({_entry_q:.0f})")
                                    logger.warning(f"      vel={_vel:.5f}%/s, path_eff={_path_eff:.2f}, entry_quality={_entry_q:.0f}")
                                    bearish_momentum = False
                            
                            # CASE 3: No clear momentum from ROC5 — atomic can SET direction
                            elif not bullish_momentum and not bearish_momentum:
                                if _vel > 0.002 and _dir_cons > 65 and _path_eff > 0.5 and _entry_q > 50:
                                    logger.info(f"   🔬 {symbol}: ATOMIC DIRECTION SET: No ROC5 momentum but ticks strongly RISING")
                                    logger.info(f"      vel={_vel:.5f}%/s, dir_consistency={_dir_cons:.0f}%, entry_quality={_entry_q:.0f}")
                                    bullish_momentum = True
                                    atomic_direction_flipped = True
                                elif _vel < -0.002 and _dir_cons > 65 and _path_eff > 0.5 and _entry_q > 50:
                                    logger.info(f"   🔬 {symbol}: ATOMIC DIRECTION SET: No ROC5 momentum but ticks strongly FALLING")
                                    logger.info(f"      vel={_vel:.5f}%/s, dir_consistency={_dir_cons:.0f}%, entry_quality={_entry_q:.0f}")
                                    bearish_momentum = True
                                    atomic_direction_flipped = True
                    except Exception as _atom_dir_err:
                        logger.debug(f"   🔬 analysis direction check error for {symbol}: {_atom_dir_err}")
                
                # ═══════════════════════════════════════════════════════════════
                # ADVANCED ANALYSIS CHECK: Sub-candle micro-granularity analysis
                # Uses WebSocket tick data (100-500ms resolution) to see the
                # TRUE price trajectory that even 1-minute candles hide.
                # PhD metrics: velocity, acceleration, path efficiency,
                # direction consistency, momentum decay, micro-Hurst exponent
                # ═══════════════════════════════════════════════════════════════
                atomic_blocked = False
                atomic_boost = 0.0
                atomic_log = ""
                predictive_intel = None
                if atomic_candle is not None:
                    try:
                        intended_dir = "long" if bullish_momentum else "short" if bearish_momentum else "long"
                        blocked, block_reason = atomic_candle.should_block_entry(symbol, intended_dir)
                        
                        if blocked:
                            atomic_blocked = True
                            atomic_log = block_reason
                            logger.warning(f"   🔬 {symbol}: ADVANCED ANALYSIS BLOCK ({intended_dir.upper()}) — {block_reason}")
                            continue
                        
                        # Get score boost/penalty from atomic analysis
                        atomic_boost = atomic_candle.get_entry_boost(symbol, intended_dir)
                        
                        # === PREDICTIVE INTELLIGENCE — Multi-source prediction ===
                        try:
                            predictive_intel = atomic_candle.get_predictive_intelligence(symbol, intended_dir)
                            pi_verdict = predictive_intel.get('verdict', 'UNKNOWN')
                            pi_confidence = predictive_intel.get('prediction_confidence', 0)
                            pi_entry_adj = predictive_intel.get('composite_entry_adj', 0)
                            
                            # Log predictive signals
                            pi_signals = predictive_intel.get('all_signals', [])
                            if pi_signals:
                                logger.info(f"   🧠 PREDICTIVE INTEL [{symbol}]: {pi_verdict} (conf={pi_confidence:.0f}%)")
                                for sig in pi_signals[:5]:  # Cap at 5 signals to avoid log spam
                                    logger.info(f"      {sig}")
                            
                            # BLOCK if predictive intelligence says BLOCK
                            if pi_verdict == 'BLOCK':
                                atomic_blocked = True
                                atomic_log = f"Predictive intelligence: BLOCK (conf={pi_confidence:.0f}%)"
                                logger.warning(f"   🧠🚫 {symbol}: PREDICTIVE BLOCK — {atomic_log}")
                                continue
                            
                            # AVOID/CAUTION → score penalty (not a block, but makes threshold harder to pass)
                            # BTRUSDT autopsy: AVOID verdict at 60% conf was only logged, trade opened anyway → instant loss
                            if pi_verdict == 'AVOID':
                                _avoid_penalty = -5 if pi_confidence >= 60 else -3
                                atomic_boost += _avoid_penalty
                                logger.warning(f"   🧠⚠️ {symbol}: PREDICTIVE AVOID (conf={pi_confidence:.0f}%) → penalty {_avoid_penalty:+.0f}")
                            elif pi_verdict == 'CAUTION':
                                _caution_penalty = -3 if pi_confidence >= 60 else -2
                                atomic_boost += _caution_penalty
                                logger.info(f"   🧠⚠️ {symbol}: PREDICTIVE CAUTION (conf={pi_confidence:.0f}%) → penalty {_caution_penalty:+.0f}")
                            
                            # Apply composite entry adjustment from all prediction sources
                            # pi_entry_adj is a float (-10 to +10) combining BTC lead, liquidation, session, volume signals
                            if pi_entry_adj != 0:
                                atomic_boost += pi_entry_adj
                                logger.info(f"   🧠 {symbol}: Predictive entry adj {pi_entry_adj:+.1f} → total atomic_boost={atomic_boost:+.1f}")
                        except Exception as pi_err:
                            logger.debug(f"   🧠 Predictive intelligence error: {pi_err}")
                        
                        # Log atomic analysis for transparency
                        analysis_str = atomic_candle.format_analysis_log(symbol)
                        logger.info(f"   {analysis_str} | boost={atomic_boost:+.1f}")
                        
                    except Exception as atomic_err:
                        logger.debug(f"   🔬 analysis error for {symbol}: {atomic_err}")
                
                # === ALLOWED_SIDES: Block momentum directions that aren't allowed ===
                if _allowed == 'long':
                    bearish_momentum = False  # Never pick SHORT
                elif _allowed == 'short':
                    bullish_momentum = False  # Never pick LONG
                
                # ═══════════════════════════════════════════════════════════════
                # SIDE PERFORMANCE BLOCKER: If a side has 75%+ loss rate
                # in recent trades, BLOCK that side and try the other one.
                # This prevents the bot from repeatedly losing on one side
                # when the market regime clearly favors the opposite direction.
                # ═══════════════════════════════════════════════════════════════
                _long_blocked_by_perf = self.is_side_blocked('LONG')
                _short_blocked_by_perf = self.is_side_blocked('SHORT')
                
                if _long_blocked_by_perf and bullish_momentum:
                    # Bullish momentum says LONG, but LONGs keep losing → try SHORT if bearish exists, else skip
                    logger.warning(f"   🚫 {symbol}: SIDE BLOCKER — LONG blocked (losing side). Bullish ROC5={momentum_5:+.2f}% ignored.")
                    bullish_momentum = False
                    # If the other side has decent momentum, switch. Otherwise skip.
                    if momentum_5 < 0.1 and momentum_10 < 0.1:
                        # Neutral or slightly bearish — allow SHORT attempt
                        bearish_momentum = True
                        logger.info(f"   🔄 {symbol}: Flipping to SHORT (momentum neutral/bearish enough)")
                    else:
                        # Momentum is truly bullish but LONGs keep losing = dead cat bounce territory, skip
                        stage1_rejected.append((symbol, f"SIDE BLOCKER: LONG blocked + bullish momentum = dead cat bounce, skip"))
                        continue
                
                if _short_blocked_by_perf and bearish_momentum:
                    logger.warning(f"   🚫 {symbol}: SIDE BLOCKER — SHORT blocked (losing side). Bearish ROC5={momentum_5:+.2f}% ignored.")
                    bearish_momentum = False
                    if momentum_5 > -0.1 and momentum_10 > -0.1:
                        bullish_momentum = True
                        logger.info(f"   🔄 {symbol}: Flipping to LONG (momentum neutral/bullish enough)")
                    else:
                        stage1_rejected.append((symbol, f"SIDE BLOCKER: SHORT blocked + bearish momentum = falling knife, skip"))
                        continue
                
                # Pick direction based on momentum, not complex math
                if bullish_momentum:
                    best_signal = 1  # LONG
                    best_math_score = long_score
                    if long_score < STAGE2_MIN_SCORE:
                        logger.info(f"   ⚠️ {symbol}: Bullish momentum but LONG score too low ({long_score:.0f}<{STAGE2_MIN_SCORE})")
                        continue
                elif bearish_momentum:
                    best_signal = -1  # SHORT
                    best_math_score = short_score
                    if short_score < STAGE2_MIN_SCORE:
                        logger.info(f"   ⚠️ {symbol}: Bearish momentum but SHORT score too low ({short_score:.0f}<{STAGE2_MIN_SCORE})")
                        continue
                else:
                    # Neutral momentum - use higher score if meets threshold
                    # Respect allowed_sides AND side blocker in neutral momentum too
                    can_long = (_allowed in ('both', 'long') and long_score >= STAGE2_MIN_SCORE
                                and not self.is_side_blocked('LONG'))
                    can_short = (_allowed in ('both', 'short') and short_score >= STAGE2_MIN_SCORE
                                 and not self.is_side_blocked('SHORT'))
                    
                    # In Extreme Fear (F&G < 15), neutral momentum should prefer SHORT
                    _neutral_fg = self._get_news_context().get('fear_greed_index', 50)
                    _fear_bias = _neutral_fg < 15  # Strong fear = prefer SHORT in neutral
                    
                    if _fear_bias and can_short:
                        # In fear market, prefer SHORT even if long score is higher
                        best_signal = -1
                        best_math_score = short_score
                        logger.info(f"   📉 {symbol}: Neutral momentum + FEAR (F&G={_neutral_fg}) → prefer SHORT")
                    elif can_long and (not can_short or long_score >= short_score):
                        best_signal = 1
                        best_math_score = long_score
                    elif can_short and (not can_long or short_score > long_score):
                        best_signal = -1
                        best_math_score = short_score
                    else:
                        logger.debug(f"   {symbol}: Neutral momentum and no allowed direction >= {STAGE2_MIN_SCORE}")
                        continue
                
                # Inject atomic analysis into context for math scoring
                if atomic_candle is not None:
                    try:
                        context['atomic_analysis'] = atomic_candle.get_analysis(symbol)
                    except Exception:
                        context['atomic_analysis'] = None
                
                # Do comprehensive check for the selected direction
                best_check = self._comprehensive_math_check(best_signal, df, price, atr, context)
                
                # Define direction string early (needed for logging)
                dir_str = "LONG" if best_signal == 1 else "SHORT"
                mom_str = f"ROC5={momentum_5:+.2f}%"
                
                # ═══════════════════════════════════════════════════════════════
                # CRITICAL FIX: Use the COMPREHENSIVE SCORE after penalties applied!
                # The previous best_math_score was the raw proactive score BEFORE
                # penalties like LOCAL_TROUGH, SPIKE_HARD_BLOCK etc were applied.
                # ═══════════════════════════════════════════════════════════════
                comprehensive_score = best_check.get('score', 0)
                
                # If comprehensive check failed (score = 0 or below threshold), skip this pair
                if comprehensive_score < STAGE2_MIN_SCORE:
                    logger.warning(f"   ❌ {symbol}: {dir_str} BLOCKED by comprehensive check (Score={comprehensive_score:.0f} < {STAGE2_MIN_SCORE})")
                    continue
                
                # Update best_math_score to the comprehensive score (with all penalties applied)
                best_math_score = comprehensive_score
                
                if atomic_boost != 0:
                    best_math_score += atomic_boost
                

                
                # Log momentum-aligned decision WITH CORRECT SCORE
                logger.info(f"   ✅ {symbol}: {dir_str} follows momentum | Score={best_math_score:.0f} | {mom_str}")
                
                # ═══════════════════════════════════════════════════════════════
                # RESISTANCE/SUPPORT LEVEL CHECK
                # Don't go LONG at 24h resistance, don't go SHORT at 24h support
                # ═══════════════════════════════════════════════════════════════
                levels = self._detect_resistance_support_levels(df, price)
                
                # Debug log for resistance/support check
                r_touch = levels.get('resistance_touches', 0)
                s_touch = levels.get('support_touches', 0)
                r_dist = levels.get('distance_to_resistance_pct', 99)
                s_dist = levels.get('distance_to_support_pct', 99)
                pos_range = levels.get('position_in_range_pct', 50)
                zone_pct = levels.get('zone_width_pct', 2.0)
                in_r_zone = levels.get('in_resistance_zone', False)
                in_s_zone = levels.get('in_support_zone', False)
                
                # Log zone info with 4 price levels
                r_upper = levels.get('resistance_upper', 0)
                r_lower = levels.get('resistance_lower', 0)
                s_upper = levels.get('support_upper', 0)
                s_lower = levels.get('support_lower', 0)
                
                zone_status = ""
                if in_r_zone:
                    zone_status = " 🔴IN_R_ZONE"
                elif in_s_zone:
                    zone_status = " 🔵IN_S_ZONE"
                elif pos_range >= 88:
                    zone_status = " ⚠️NEAR_RESISTANCE"
                elif pos_range <= 12:
                    zone_status = " ⚠️NEAR_SUPPORT"
                
                logger.info(f"   📍 {symbol}: Range={pos_range:.0f}% | Zone={zone_pct:.1f}% | R[${r_lower:.4f}-${r_upper:.4f}] S[${s_lower:.4f}-${s_upper:.4f}]{zone_status}")
                
                # ═══════════════════════════════════════════════════════════════
                # ZONE BLOCKING: Only block if BOTH in zone AND has 3+ bounces
                # This prevents blocking valid breakout trades
                # ═══════════════════════════════════════════════════════════════
                
                # Block LONG only if in resistance zone AND 3+ bounces confirmed rejection
                if best_signal == 1 and levels.get('in_resistance_zone') and r_touch >= 3:
                    logger.warning(f"   🚫🚫 {symbol}: LONG BLOCKED - IN RESISTANCE ZONE with {r_touch} bounces [${r_lower:.4f} - ${r_upper:.4f}]")
                    continue
                
                # Block SHORT only if in support zone AND 3+ bounces confirmed support
                if best_signal == -1 and levels.get('in_support_zone') and s_touch >= 3:
                    logger.warning(f"   🚫🚫 {symbol}: SHORT BLOCKED - IN SUPPORT ZONE with {s_touch} bounces [${s_lower:.4f} - ${s_upper:.4f}]")
                    continue
                
                # Block LONG if price is too close to resistance
                # Range limits: testing mode uses 88/12, loose uses 90/10, default uses 80/20
                # FIX: Testing was 97/3 = basically no protection! Entries at 93-94% Range
                # kept hitting resistance ceiling and losing. 88/12 is still lenient but real.
                _threshold_mode = get_threshold_mode()
                if _threshold_mode == 'testing':
                    _range_long_limit = 88
                    _range_short_limit = 12
                elif _threshold_mode == 'loose':
                    _range_long_limit = 92 if market_timing.get('is_weekend', False) else 90
                    _range_short_limit = 8 if market_timing.get('is_weekend', False) else 10
                else:
                    _range_long_limit = 85 if market_timing.get('is_weekend', False) else 80
                    _range_short_limit = 15 if market_timing.get('is_weekend', False) else 20
                if best_signal == 1 and pos_range >= _range_long_limit:
                    logger.warning(f"   🚫🚫 {symbol}: LONG BLOCKED - TOO CLOSE TO RESISTANCE (Range={pos_range:.0f}% >= {_range_long_limit}%)")
                    continue
                
                # Block SHORT if price is too close to support
                if best_signal == -1 and pos_range <= _range_short_limit:
                    logger.warning(f"   🚫🚫 {symbol}: SHORT BLOCKED - TOO CLOSE TO SUPPORT (Range={pos_range:.0f}% <= {_range_short_limit}%)")
                    continue
                
                # ═══════════════════════════════════════════════════════════════
                # REMOVED: Legacy blocking based on just being near a level
                # The zone + bounce requirement above is sufficient
                # ═══════════════════════════════════════════════════════════════
                
                # REMOVED: Old "at_resistance" / "at_support" blocks - too strict
                # The new zone + 3 bounces requirement above is the ONLY zone check now
                
                # ═══════════════════════════════════════════════════════════════
                # EXTREME SENTIMENT FILTER: When market is in extreme fear/greed,
                # require stronger confirmation before entry
                # ═══════════════════════════════════════════════════════════════
                news_ctx = self._get_news_context()
                fear_greed = news_ctx.get('fear_greed_index', 50)
                
                # In EXTREME FEAR (<8), be cautious about LONGs - market might keep falling
                _fear_is_weekend = market_timing.get('is_weekend', False)
                _fear_testing = get_threshold_mode() == 'testing'
                if _fear_testing:
                    # Testing mode: still require LONGs to be near support in fear markets
                    # FIX: Was 97/97/3/5 = basically disabled! LONGs at 93% range lost money.
                    _fear_long_limit = 75
                    _fear_mid_limit = 85
                    _fear_threshold_low = 8
                    _fear_threshold_mid = 15
                else:
                    _fear_long_limit = 65 if _fear_is_weekend else 40
                    _fear_mid_limit = 80 if _fear_is_weekend else 60
                    _fear_threshold_low = 8
                    _fear_threshold_mid = 15
                if fear_greed < _fear_threshold_low and best_signal == 1:
                    if pos_range >= _fear_long_limit:
                        logger.warning(f"   🚫 {symbol}: LONG BLOCKED - EXTREME FEAR ({fear_greed}) requires price in lower range (range<{_fear_long_limit}%), currently at {pos_range:.0f}%")
                        continue
                elif fear_greed < _fear_threshold_mid and best_signal == 1 and pos_range >= _fear_mid_limit:
                    logger.warning(f"   🚫 {symbol}: LONG BLOCKED - EXTREME FEAR ({fear_greed}) + Price in upper range ({pos_range:.0f}% >= {_fear_mid_limit}%). Wait for price near support.")
                    continue
                
                # In EXTREME GREED (>80), be cautious about SHORTs - market might keep rising
                # Require price to be in upper part of range (near resistance) for better entries  
                if fear_greed > 80 and best_signal == -1 and pos_range <= 50:
                    logger.warning(f"   🚫 {symbol}: SHORT BLOCKED - EXTREME GREED ({fear_greed}) + Price in lower range ({pos_range:.0f}%). Wait for price near resistance.")
                    continue
                
                # Add level warning to context for AI to consider (info only, not blocking)
                level_warning = levels.get('warning')
                
                # Get reasons from math check (for AI context, NOT filtering)
                reasons_for = best_check.get('reasons_for', [])
                reasons_against = best_check.get('reasons_against', [])
                
                # ═══════════════════════════════════════════════════════════════
                # MOMENTUM DIRECTION MUST MATCH TRADE DIRECTION!
                # This is the MOST IMPORTANT filter - we don't trade against momentum
                # ═══════════════════════════════════════════════════════════════
                
                # Only one remaining filter: Is momentum STRONG enough?
                _mom_testing = get_threshold_mode() == 'testing'
                MIN_MOMENTUM_STRENGTH = 0.15 if _mom_testing else 0.35  # Testing: lower bar
                momentum_strength = abs(momentum_5)
                if momentum_strength < MIN_MOMENTUM_STRENGTH:
                    logger.info(f"   ❌ {symbol}: WEAK momentum ({momentum_strength:.2f}% < {MIN_MOMENTUM_STRENGTH}%)")
                    continue
                
                # 🚨 CRITICAL: Momentum direction MUST match trade direction!
                # LONG requires positive momentum (price rising)
                # SHORT requires negative momentum (price falling)
                if best_signal == 1 and momentum_5 < 0:
                    logger.warning(f"   🚫 {symbol}: LONG BLOCKED - Momentum is NEGATIVE ({momentum_5:+.2f}%). Price is falling, not rising!")
                    continue
                    
                if best_signal == -1 and momentum_5 > 0:
                    logger.warning(f"   🚫 {symbol}: SHORT BLOCKED - Momentum is POSITIVE ({momentum_5:+.2f}%). Price is rising, not falling!")
                    continue
                
                logger.info(f"📊 STAGE 2 PASS: {symbol} {'LONG' if best_signal == 1 else 'SHORT'} | Score={best_math_score:.0f} | ROC5={momentum_5:+.2f}%")
                
                # Get PhD math score from comprehensive check for better sorting
                phd_math_score = best_check.get('score', best_math_score)
                
                math_ranked.append({
                    'symbol': symbol,
                    'signal': best_signal,
                    'direction': 'LONG' if best_signal == 1 else 'SHORT',
                    'math_score': best_math_score,  # Stage 2 score (momentum weighted)
                    'phd_score': phd_math_score,    # PhD comprehensive score (for sorting)
                    'math_check': best_check,
                    'price': price,
                    'atr': atr,
                    'df': df,
                    'reasons_for': reasons_for,
                    'reasons_against': reasons_against,
                    'level_info': levels,  # Include resistance/support info for AI
                    'level_warning': level_warning,  # Include any warning
                    'atomic_boost': atomic_boost,  # analysis score modifier
                    'atomic_analysis': atomic_candle.get_analysis(symbol) if atomic_candle else None,
                    'atomic_direction_flipped': atomic_direction_flipped  # True if atomic changed direction
                })
            
            pipeline_stats['stage2_output'] = len(math_ranked)
            logger.info(f"📊 STAGE 2 RESULT: {len(math_ranked)}/{len(stage1_passed)} passed deep analysis")
            
            if not math_ranked:
                _s2_thr = STAGE2_MIN_SCORE  # Uses weekend-aware value from above
                return {
                    'has_opportunity': False,
                    'reason': f'STAGE 2: No pairs passed deep math analysis (need score >= {_s2_thr})',
                    'ranked_pairs': [],
                    'pipeline_stats': pipeline_stats
                }
            
            # Sort by PhD math score (comprehensive analysis) for Stage 3
            # This prioritizes trades with better mathematical backing over pure momentum
            math_ranked.sort(key=lambda x: x.get('phd_score', x['math_score']), reverse=True)
            pipeline_stats['stage2_passed'] = len(math_ranked)
            
            # ═══════════════════════════════════════════════════════════════
            # INTELLIGENT SYSTEM #5: CORRELATION GUARD
            # Before Stage 3, check if candidates are too correlated with
            # existing positions. Apply penalty or block as needed.
            # ═══════════════════════════════════════════════════════════════
            if current_positions and len(current_positions) > 0 and hasattr(self, 'correlation_guard'):
                try:
                    # Build existing position DFs
                    existing_dfs = {}
                    for pos in current_positions:
                        if pos is None:
                            continue
                        pos_sym = pos.symbol if hasattr(pos, 'symbol') else pos.get('symbol', '')
                        # Find DF for this position from our pairs_data
                        for pair_data in pairs_data:
                            pair_sym = pair_data.get('symbol', '') if isinstance(pair_data, dict) else ''
                            if pos_sym.replace('/', '').upper() in pair_sym.replace('/', '').upper():
                                pair_df = pair_data.get('df') if isinstance(pair_data, dict) else None
                                if pair_df is not None and len(pair_df) >= 20:
                                    existing_dfs[pos_sym] = pair_df
                                break
                    
                    if existing_dfs:
                        for candidate in math_ranked:
                            corr_result = self.correlation_guard.check_correlation(
                                candidate_df=candidate['df'],
                                candidate_symbol=candidate['symbol'],
                                candidate_direction=candidate['direction'],
                                existing_positions=current_positions,
                                existing_dfs=existing_dfs
                            )
                            if corr_result['penalty'] > 0:
                                candidate['math_score'] = max(0, candidate['math_score'] - corr_result['penalty'])
                                candidate['phd_score'] = max(0, candidate.get('phd_score', candidate['math_score']) - corr_result['penalty'])
                                candidate['correlation_info'] = corr_result
                                candidate.setdefault('reasons_against', []).append(
                                    f"📊 Correlation guard: {corr_result['reason']}"
                                )
                                logger.info(f"📊 CORR GUARD: {candidate['symbol']} penalty={corr_result['penalty']} "
                                          f"(corr={corr_result['max_correlation']:.2f} with {corr_result['max_corr_symbol']})")
                        
                        # Re-sort after correlation penalties
                        math_ranked.sort(key=lambda x: x.get('phd_score', x['math_score']), reverse=True)
                except Exception as e:
                    logger.warning(f"⚠️ Correlation guard error (non-fatal): {e}")
            
            # ═══════════════════════════════════════════════════════════════
            # STAGE 3: AI FINAL DECISION (Strategic validation)
            # ═══════════════════════════════════════════════════════════════
            logger.info(f"🤖 STAGE 3: AI Final Decision (top {min(3, len(math_ranked))} candidates)")
            logger.info(f"   Candidates: {[(c['symbol'], c['direction'], c['math_score']) for c in math_ranked[:3]]}")
            
            # Only send top 3 to AI to save API calls
            top_candidates = math_ranked[:3]
            ai_approved_candidates = []
            best_rejected = None
            
            if self.use_ai:
                for candidate in top_candidates:
                    try:
                        # === AI FINAL ENTRY DECISION ===
                        # AI receives BOTH direction math scores and makes the FINAL call
                        # AI can: APPROVE the math direction, REVERSE it, or REJECT completely
                        
                        context = self._build_market_context(candidate['df'], candidate['price'], candidate['atr'])
                        
                        # Get BOTH direction scores for AI to see
                        long_check = self._comprehensive_math_check(1, candidate['df'], candidate['price'], candidate['atr'], context)
                        short_check = self._comprehensive_math_check(-1, candidate['df'], candidate['price'], candidate['atr'], context)
                        
                        long_score = long_check.get('score', 0)
                        short_score = short_check.get('score', 0)
                        math_direction = candidate['direction']
                        math_score = candidate['math_score']
                        
                        # Get level info for AI to consider
                        level_info = candidate.get('level_info', {})
                        
                        # Build advanced math section for the winning direction
                        # IMPORTANT: Use Stage 2 momentum-blended score for consistency
                        # PhD metrics are shown as details, but the overall score must match
                        # what was used to rank candidates (prevents AI seeing conflicting scores)
                        winning_check = long_check if math_direction == 'LONG' else short_check
                        winning_check_for_display = winning_check.copy()
                        winning_check_for_display['score'] = math_score  # Use Stage 2 score, not PhD score
                        winning_check_for_display['approved'] = math_score >= 45  # Match Stage 2 threshold
                        advanced_math_section = self._build_math_analysis_section(winning_check_for_display)
                        
                        # Log advanced math integration for Stage 3
                        detailed = winning_check.get('detailed_analysis', {})
                        if detailed:
                            kalman_val = detailed.get('kalman_momentum')
                            poc_val = detailed.get('poc_price')
                            hurst_val = detailed.get('hurst_exponent')
                            kalman_str = f"{kalman_val:.2f}" if isinstance(kalman_val, (int, float)) else "N/A"
                            poc_str = f"${poc_val:,.0f}" if isinstance(poc_val, (int, float)) else "N/A"
                            hurst_str = f"{hurst_val:.2f}" if isinstance(hurst_val, (int, float)) else "N/A"
                            logger.info(f"🧮 STAGE3 MATH: {candidate['symbol']} | Kalman={kalman_str} | POC={poc_str} | Hurst={hurst_str}")
                        
                        # Build atomic candle section for AI prompt
                        _atomic_ai_section = ""
                        _atomic_data = candidate.get('atomic_analysis')
                        if _atomic_data is not None:
                            try:
                                _a_vel = _atomic_data.velocity
                                _a_acc = _atomic_data.acceleration
                                _a_jerk = _atomic_data.jerk
                                _a_peff = _atomic_data.path_efficiency
                                _a_dcons = _atomic_data.direction_consistency
                                _a_mdecay = _atomic_data.momentum_decay_rate
                                _a_hurst = _atomic_data.hurst_micro
                                _a_eq = _atomic_data.entry_quality
                                _a_rp = _atomic_data.reversal_probability
                                _a_mt = _atomic_data.micro_trend
                                _a_sc = _atomic_data.swing_count
                                _a_ticks = _atomic_data.tick_count
                                _a_window = _atomic_data.window_seconds
                                _a_flipped = candidate.get('atomic_direction_flipped', False)
                                
                                _atomic_ai_section = (
                                    f"=== 🔬 ADVANCED ANALYSIS ANALYSIS (Sub-Candle PhD Metrics) ===\n"
                                    f"These metrics are computed from raw WebSocket ticks (100-500ms resolution),\n"
                                    f"revealing the TRUE price trajectory that even 1-minute candles hide.\n"
                                    f"Data: {_a_ticks} ticks over {_a_window:.0f}s window\n"
                                    f"\n"
                                    f"Tick Velocity (dP/dt): {_a_vel:+.5f} %/sec {'⬆️ RISING' if _a_vel > 0 else '⬇️ FALLING' if _a_vel < 0 else '➡️ FLAT'}\n"
                                    f"Acceleration (d²P/dt²): {_a_acc:+.6f} {'🚀 Accelerating' if (_a_vel > 0 and _a_acc > 0) or (_a_vel < 0 and _a_acc < 0) else '🛑 Decelerating' if abs(_a_acc) > 0.0001 else '➡️ Steady'}\n"
                                    f"Jerk (d³P/dt³): {_a_jerk:+.6f} {'⚠️ Unstable' if abs(_a_jerk) > 0.001 else '✅ Smooth'}\n"
                                    f"Micro-Trend: {_a_mt.upper()}\n"
                                    f"Direction Consistency: {_a_dcons:.0f}% {'(Strong one-way flow)' if _a_dcons > 70 else '(Mixed)' if _a_dcons > 40 else '(Choppy/directionless)'}\n"
                                    f"Path Efficiency: {_a_peff:.2f} {'(Clean straight-line move)' if _a_peff > 0.6 else '(Moderate noise)' if _a_peff > 0.3 else '(Very choppy, going nowhere)'}\n"
                                    f"Momentum Decay: {_a_mdecay:+.4f} {'(Momentum FADING ⚠️)' if _a_mdecay < -0.01 else '(Momentum BUILDING 🚀)' if _a_mdecay > 0.01 else '(Steady)'}\n"
                                    f"Swing Count: {_a_sc} {'(Too many reversals ⚠️)' if _a_sc > 6 else '(Clean)'}\n"
                                    f"Micro-Hurst: {_a_hurst:.2f} {'(Trending ✅)' if _a_hurst > 0.55 else '(Mean-reverting ⚠️)' if _a_hurst < 0.45 else '(Random walk)'}\n"
                                    f"Entry Quality: {_a_eq:.0f}/100 {'🟢 GOOD' if _a_eq > 60 else '🟡 FAIR' if _a_eq > 40 else '🔴 POOR'}\n"
                                    f"Reversal Probability: {_a_rp:.0f}% {'🚨 HIGH REVERSAL RISK' if _a_rp > 60 else '⚠️ Moderate reversal risk' if _a_rp > 40 else '✅ Low reversal risk'}\n"
                                    f"{'🔬 NOTE: Atomic ticks FLIPPED the direction from ROC5 — trust tick data over candle momentum!' if _a_flipped else ''}\n"
                                    f"\n"
                                    f"KEY INSIGHTS FOR YOUR DECISION:\n"
                                    f"- If velocity contradicts trade direction → HIGH RISK of immediate adverse move\n"
                                    f"- If path efficiency < 0.3 → Price is choppy, entry timing is BAD\n"
                                    f"- If reversal probability > 50% → Consider REJECTING or flipping direction\n"
                                    f"- If entry quality < 30 → Poor entry conditions, tighten or skip\n"
                                    f"- If momentum is decaying → Move may be exhausted, be cautious\n"
                                )
                                logger.info(f"🔬 STAGE3 ATOMIC: {candidate['symbol']} | vel={_a_vel:+.5f} | eq={_a_eq:.0f} | rev_prob={_a_rp:.0f}% | trend={_a_mt}")
                            except Exception as _ae:
                                logger.debug(f"🔬 Atomic section build error: {_ae}")
                        
                        # Call enhanced AI decision that can choose direction
                        ai_result = self._ai_final_entry_decision(
                            symbol=candidate['symbol'],
                            math_direction=math_direction,
                            math_score=math_score,
                            long_score=long_score,
                            short_score=short_score,
                            long_reasons_for=long_check.get('reasons_for', []),
                            long_reasons_against=long_check.get('reasons_against', []),
                            short_reasons_for=short_check.get('reasons_for', []),
                            short_reasons_against=short_check.get('reasons_against', []),
                            context=context,
                            level_info=level_info,  # Pass resistance/support info to AI
                            advanced_math=advanced_math_section,  # Advanced math to AI
                        )
                        
                        # AI has FULL POWER to decide
                        ai_decision = ai_result.get('decision', 'REJECT')  # LONG, SHORT, or REJECT
                        ai_confidence = ai_result.get('confidence', 0.5)
                        ai_analysis_score = ai_result.get('ai_analysis', ai_confidence * 100)  # Independent AI assessment
                        ai_reasoning = ai_result.get('reasoning', '')
                        
                        # ═══════════════════════════════════════════════════════════════
                        # MATH SUPREMACY: If PhD-level math score >= 80, override AI rejection
                        # The math incorporates: Kalman, GARCH, Hurst, orderflow, volume profile,
                        # support/resistance, momentum, mean reversion - it should be self-sufficient
                        # ═══════════════════════════════════════════════════════════════
                        MATH_SUPREMACY_THRESHOLD = 90
                        
                        # === FIX: AI REJECTION CONSISTENCY CHECK ===
                        # Only block if AI was rejecting in ONE direction and suddenly flips to APPROVE
                        # a DIFFERENT direction. If Gemini consistently wants the SAME direction, trust it.
                        _consistency_blocked = False
                        if ai_decision != 'REJECT':
                            _sym = candidate['symbol']
                            _tracker = self._ai_rejection_tracker.get(_sym)
                            if _tracker and _tracker['rejections'] >= self.AI_REJECTION_BLOCK_THRESHOLD:
                                _since = (datetime.now(timezone.utc) - _tracker['last_rejection']).total_seconds() / 60
                                if _since >= self.AI_REJECTION_MEMORY_MINUTES:
                                    # Memory expired, clear tracker
                                    del self._ai_rejection_tracker[_sym]
                                elif _tracker.get('last_direction', '') and _tracker['last_direction'] != ai_decision:
                                    # Direction FLIPPED — AI was rejecting for one direction, now approving opposite
                                    logger.warning(f"🚫 AI CONSISTENCY BLOCK: {_sym} - AI rejected {_tracker['rejections']}x ({_tracker['last_direction']}) in last {_since:.0f}min, now flipping to {ai_decision}. Blocked.")
                                    ai_decision = 'REJECT'
                                    ai_confidence = 0.45
                                    ai_reasoning = f"Consistency block: direction flip {_tracker['last_direction']}→{ai_decision} after {_tracker['rejections']}x reject. Original: {ai_reasoning[:60]}"
                                    _consistency_blocked = True
                                else:
                                    # Same direction — Gemini consistently wants this direction, approve it!
                                    logger.info(f"✅ AI CONSISTENCY OK: {_sym} - AI approves {ai_decision} (same direction as recent rejects, consistent signal)")
                                    # Clear the tracker since AI is now approving
                                    del self._ai_rejection_tracker[_sym]
                        
                        # Track AI rejections for consistency checking
                        # IMPORTANT: Don't count consistency-blocked decisions (avoids runaway counter)
                        if ai_decision == 'REJECT' and not _consistency_blocked:
                            _sym = candidate['symbol']
                            if _sym not in self._ai_rejection_tracker:
                                self._ai_rejection_tracker[_sym] = {'rejections': 0, 'last_rejection': None, 'last_direction': ''}
                            self._ai_rejection_tracker[_sym]['rejections'] += 1
                            self._ai_rejection_tracker[_sym]['last_rejection'] = datetime.now(timezone.utc)
                            self._ai_rejection_tracker[_sym]['last_direction'] = math_direction
                        elif ai_decision != 'REJECT':
                            # AI approved — clear rejection tracker for this symbol
                            _sym = candidate['symbol']
                            if _sym in self._ai_rejection_tracker:
                                del self._ai_rejection_tracker[_sym]
                        
                        if ai_decision == 'REJECT' and math_score >= MATH_SUPREMACY_THRESHOLD:
                            # MATH OVERRIDES AI - PhD analysis is strong enough
                            logger.warning(f"🧮 MATH SUPREMACY: {candidate['symbol']} - Math score {math_score} >= {MATH_SUPREMACY_THRESHOLD} OVERRIDES AI rejection")
                            logger.info(f"   AI wanted to reject: {ai_reasoning[:80]}")
                            candidate['ai_approved'] = True
                            candidate['ai_confidence'] = 0.70  # Assign reasonable confidence
                            candidate['ai_score'] = 70  # Base AI score for override
                            candidate['ai_reasoning'] = f"MATH OVERRIDE ({math_score} >= {MATH_SUPREMACY_THRESHOLD}): {ai_reasoning[:60]}"
                            candidate['math_override'] = True
                        elif ai_decision == 'REJECT':
                            # AI says NO TRADE and math not strong enough to override
                            # ═══════════════════════════════════════════════════════════
                            # WEEKEND MOMENTUM OVERRIDE: If weekend + Stage 2 passed + 
                            # momentum strongly aligns → auto-approve. ML veto is the 
                            # real safety gate, not AI conservatism on low scores.
                            # ═══════════════════════════════════════════════════════════
                            _is_wknd = self._get_market_hours_context().get('is_weekend', False)
                            _roc5 = context.get('roc_5', 0)
                            _roc10 = context.get('roc_10', 0)
                            # Require STRONG momentum alignment (ROC5 > 0.4 AND ROC10 agrees)
                            _momentum_aligns = False
                            if math_direction == 'LONG' and _roc5 > 0.4 and _roc10 > 0.15:
                                _momentum_aligns = True
                            elif math_direction == 'SHORT' and _roc5 < -0.4 and _roc10 < -0.15:
                                _momentum_aligns = True
                            
                            # DIRECTION-AWARE F&G check for weekend override
                            # FIXED: Also block SHORT in EXTREME fear (< 15) — bounce zone!
                            # Extreme Fear (15-30) + SHORT = OK (market falling)
                            # Extreme Fear (< 15) + SHORT = BLOCKED (capitulation → bounce imminent!)
                            # Extreme Fear + LONG = BLOCKED (going against panic)
                            # Extreme Greed + LONG = OK (market rising, long aligns)
                            # Extreme Greed + SHORT = BLOCKED (going against euphoria)
                            _override_news = self._get_news_context()
                            _override_fg = _override_news.get('fear_greed_index', 50)
                            _fg_allows_override = True
                            if _override_fg < 8 and math_direction == 'LONG':
                                _fg_allows_override = False  # Don't LONG in extreme fear (< 8)
                            elif _override_fg > 80 and math_direction == 'SHORT':
                                _fg_allows_override = False  # Don't SHORT in extreme greed
                            elif _override_fg < 5 and math_direction == 'SHORT':
                                _fg_allows_override = False  # Don't SHORT in extreme fear bottom (< 5) — bounce imminent!
                            
                            # RSI GUARD: Don't override AI when RSI is in danger zone
                            # RSI < 35 for SHORT = oversold, bounce likely
                            # RSI > 65 for LONG = overbought, pullback likely
                            _override_rsi = context.get('rsi', 50)
                            _rsi_allows_override = True
                            _rsi_testing = get_threshold_mode() == 'testing'
                            _rsi_short_limit = 25 if _rsi_testing else 35  # Testing: only block at RSI < 25
                            _rsi_long_limit = 75 if _rsi_testing else 65   # Testing: only block at RSI > 75
                            if math_direction == 'SHORT' and _override_rsi < _rsi_short_limit:
                                _rsi_allows_override = False
                                if _is_wknd:
                                    logger.warning(f"📅 WEEKEND OVERRIDE RSI BLOCK: {candidate['symbol']} SHORT - RSI={_override_rsi:.0f} < {_rsi_short_limit} (oversold, bounce likely!)")
                            elif math_direction == 'LONG' and _override_rsi > _rsi_long_limit:
                                _rsi_allows_override = False
                                if _is_wknd:
                                    logger.warning(f"📅 WEEKEND OVERRIDE RSI BLOCK: {candidate['symbol']} LONG - RSI={_override_rsi:.0f} > {_rsi_long_limit} (overbought, pullback likely!)")
                            
                            _wknd_testing = get_threshold_mode() == 'testing'
                            _wknd_fg_min = 3 if _wknd_testing else 15  # Testing: allow F&G >= 3
                            
                            # === ADVANCED ANALYSIS QUALITY GATE FOR WEEKEND OVERRIDE ===
                            # FIX: H/USDT loss — weekend override approved despite:
                            #   entry_quality=0, reversal_probability=80%, path_efficiency=0.30
                            # analysis PhD metrics must agree before overriding AI
                            _atomic_allows_override = True
                            _analysis_block_reason = ""
                            _candidate_atomic = candidate.get('atomic_analysis')
                            if _candidate_atomic is None and atomic_candle is not None:
                                try:
                                    _candidate_atomic = atomic_candle.get_analysis(candidate['symbol'])
                                except Exception:
                                    pass
                            if _candidate_atomic is not None and _candidate_atomic.tick_count >= 30:
                                if _candidate_atomic.entry_quality < 20:
                                    _atomic_allows_override = False
                                    _analysis_block_reason = f"entry_quality={_candidate_atomic.entry_quality:.0f} < 20"
                                elif _candidate_atomic.reversal_probability > 65:
                                    _atomic_allows_override = False
                                    _analysis_block_reason = f"reversal_prob={_candidate_atomic.reversal_probability:.0f}% > 65%"
                                elif _candidate_atomic.path_efficiency < 0.20:
                                    _atomic_allows_override = False
                                    _analysis_block_reason = f"path_efficiency={_candidate_atomic.path_efficiency:.2f} < 0.20"
                            
                            if _is_wknd and math_score >= 50 and _momentum_aligns and _fg_allows_override and _rsi_allows_override and _override_fg >= _wknd_fg_min and _atomic_allows_override:
                                # Weekend override: relaxed AI — let ML veto be the real gate
                                # Requires solid math (50+) AND strong dual-momentum AND no extreme sentiment AND F&G >= 15
                                # AND analysis quality gate passed (entry quality, reversal prob, efficiency)
                                # Range block (80%) already filtered out near-resistance entries in Stage 2
                                logger.warning(f"📅 WEEKEND OVERRIDE: {candidate['symbol']} {math_direction} - Math={math_score:.0f}, ROC5={_roc5:+.2f}%, ROC10={_roc10:+.2f}%, F&G={_override_fg} → Auto-approve (ML veto is safety net)")
                                candidate['ai_approved'] = True
                                candidate['ai_confidence'] = 0.62
                                candidate['ai_score'] = 62
                                candidate['ai_reasoning'] = f"WEEKEND OVERRIDE: Math={math_score:.0f}, momentum (ROC5={_roc5:+.2f}%, ROC10={_roc10:+.2f}%), F&G={_override_fg}. ML veto is safety net. AI said: {ai_reasoning[:60]}"
                                candidate['math_override'] = True
                            elif _is_wknd and math_score >= 50 and _momentum_aligns and not _atomic_allows_override:
                                # analysis PhD metrics blocked the override — momentum is bad
                                logger.warning(f"📅 WEEKEND OVERRIDE BLOCKED (ANALYSIS): {candidate['symbol']} {math_direction} - {_analysis_block_reason}! Math={math_score:.0f} but momentum too risky")
                                candidate['ai_approved'] = False
                                candidate['ai_confidence'] = ai_confidence
                                candidate['ai_score'] = 20 + (1 - ai_confidence) * 30
                                candidate['ai_reasoning'] = f"Weekend override blocked by analysis: {_analysis_block_reason}. AI: {ai_reasoning[:60]}"
                                candidate['math_override'] = False
                            elif _is_wknd and math_score >= 50 and _momentum_aligns and (not _fg_allows_override or not _rsi_allows_override or _override_fg < _wknd_fg_min):
                                # Extreme sentiment OR dangerous RSI blocks override — AI rejection stands
                                _block_reason = f"F&G={_override_fg}" if not _fg_allows_override else f"RSI={_override_rsi:.0f}"
                                logger.warning(f"📅 WEEKEND OVERRIDE BLOCKED: {candidate['symbol']} {math_direction} - {_block_reason} is dangerous! Math={math_score:.0f} but conditions too risky")
                                candidate['ai_approved'] = False
                                candidate['ai_confidence'] = ai_confidence
                                candidate['ai_score'] = 20 + (1 - ai_confidence) * 30
                                candidate['ai_reasoning'] = f"Weekend override blocked: {_block_reason} dangerous. AI: {ai_reasoning[:60]}"
                                candidate['math_override'] = False
                            else:
                                candidate['ai_approved'] = False
                                candidate['ai_confidence'] = ai_confidence
                                candidate['ai_score'] = 20 + (1 - ai_confidence) * 30
                                candidate['ai_reasoning'] = ai_reasoning
                                candidate['math_override'] = False
                                logger.info(f"📊 {candidate['symbol']}: AI ❌ REJECT (conf={ai_confidence:.0%}) - {ai_reasoning[:60]}")
                        elif ai_confidence < (_wk_conf := ((0.45 if get_threshold_mode() == 'testing' else (0.50 if get_threshold_mode() == 'loose' else 0.55)) if self._get_market_hours_context().get('is_weekend', False) else (0.50 if get_threshold_mode() == 'testing' else (0.55 if get_threshold_mode() == 'loose' else 0.60)))):
                            # AI says yes but not confident enough - TREAT AS REJECT
                            # Testing: Weekend=45%, Weekday=50%. Normal: Weekend=55%, Weekday=60%
                            candidate['ai_approved'] = False
                            candidate['ai_confidence'] = ai_confidence
                            candidate['ai_score'] = ai_confidence * 50
                            candidate['ai_reasoning'] = f"Low confidence ({ai_confidence:.0%}) - need {_wk_conf:.0%}+: {ai_reasoning}"
                            logger.warning(f"📊 {candidate['symbol']}: AI {ai_decision} but LOW CONF ({ai_confidence:.0%} < {_wk_conf:.0%}) - treating as REJECT")
                        else:
                            # AI approved with sufficient confidence
                            candidate['ai_approved'] = True
                            candidate['ai_confidence'] = ai_confidence
                            candidate['ai_analysis_score'] = ai_analysis_score  # Gemini's independent assessment
                            candidate['ai_score'] = ai_analysis_score  # Use analysis score, not just confidence
                            candidate['ai_reasoning'] = ai_reasoning
                            
                            # Check if AI changed direction from math
                            if ai_decision != math_direction:
                                # AI REVERSED the direction!
                                logger.warning(f"🔄 {candidate['symbol']}: AI REVERSED direction! Math said {math_direction}, AI says {ai_decision}")
                                candidate['direction'] = ai_decision
                                candidate['signal'] = 1 if ai_decision == 'LONG' else -1
                                candidate['ai_reversed_direction'] = True  # Flag for penalty
                                # Update math score for new direction
                                if ai_decision == 'LONG':
                                    candidate['math_score'] = long_score
                                    candidate['math_check'] = long_check
                                else:
                                    candidate['math_score'] = short_score
                                    candidate['math_check'] = short_check
                            else:
                                candidate['ai_reversed_direction'] = False
                            
                            logger.info(f"📊 {candidate['symbol']}: AI ✅ {ai_decision} (conf={ai_confidence:.0%}) - {ai_reasoning[:60]}")
                        
                        # Combined score: 55% Math + 25% AI Analysis + 20% Confidence
                        # AI analysis = Gemini's independent assessment (momentum, patterns, risk)
                        # This gives AI real intelligence weight — it can help OR hurt the score
                        _ai_anal = candidate.get('ai_analysis_score', candidate.get('ai_score', 50))
                        _ai_conf_score = candidate.get('ai_confidence', 0.5) * 100
                        candidate['combined_score'] = (candidate['math_score'] * 0.55) + (_ai_anal * 0.25) + (_ai_conf_score * 0.20)
                        logger.info(f"📊 {candidate['symbol']}: Combined = Math({candidate['math_score']:.0f})*0.55 + AI_Analysis({_ai_anal:.0f})*0.25 + Conf({_ai_conf_score:.0f})*0.20 = {candidate['combined_score']:.1f}")
                        
                    except Exception as ai_err:
                        logger.warning(f"AI final entry decision failed for {candidate['symbol']}: {ai_err}")
                        # AI REQUIRED - if AI fails, mark as not approved
                        candidate['ai_approved'] = False
                        candidate['ai_confidence'] = 0.0
                        candidate['ai_score'] = 0
                        candidate['ai_reasoning'] = f"AI error: {ai_err}"
                        candidate['combined_score'] = 0  # Cannot open without AI
            else:
                # No AI available - cannot open positions without AI
                logger.warning("⚠️ AI not available - positions require AI approval")
                for candidate in top_candidates:
                    candidate['ai_approved'] = False
                    candidate['ai_confidence'] = 0.0
                    candidate['ai_score'] = 50
                    candidate['combined_score'] = candidate['math_score']
            
            # Re-sort by combined score
            top_candidates.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # === STAGE 3 FINAL DECISION ===
            # AI APPROVAL IS MANDATORY - no position opens without AI saying YES
            
            # Filter to only AI-approved candidates
            ai_approved_candidates = [c for c in top_candidates if c.get('ai_approved', False)]
            pipeline_stats['stage3_approved'] = len(ai_approved_candidates)
            
            if not ai_approved_candidates:
                # NO AI APPROVAL - NO TRADE
                best_rejected = top_candidates[0] if top_candidates else None
                if best_rejected:
                    logger.warning(f"🚫 STAGE 3 FAILED: AI did not approve any candidates")
                    logger.warning(f"   Best rejected: {best_rejected['symbol']} {best_rejected['direction']}")
                    logger.warning(f"   Math: {best_rejected['math_score']:.0f} | AI rejected: {best_rejected.get('ai_reasoning', 'unknown')[:50]}")
                logger.info(f"📊 PIPELINE: {pipeline_stats}")
                return {
                    'has_opportunity': False,
                    'reason': f"STAGE 3: AI rejected all candidates. Best was {best_rejected['symbol']} {best_rejected['direction']}" if best_rejected else 'No candidates',
                    'ranked_pairs': top_candidates,
                    'best_score': best_rejected['combined_score'] if best_rejected else 0,
                    'pipeline_stats': pipeline_stats
                }
            
            # Select best AI-approved candidate
            best = ai_approved_candidates[0]
            
            # === ABSOLUTE MATH FLOOR (no exceptions) ===
            _floor_mode = get_threshold_mode()
            ABSOLUTE_MIN_MATH = 30 if _floor_mode == 'testing' else (40 if _floor_mode == 'loose' else 45)
            if best['math_score'] < ABSOLUTE_MIN_MATH:
                logger.warning(f"🚫 {best['symbol']}: HARD MATH FLOOR — Math score {best['math_score']:.1f} < {ABSOLUTE_MIN_MATH}. AI cannot override this.")
                logger.info(f"📊 PIPELINE: {pipeline_stats}")
                return {
                    'has_opportunity': False,
                    'reason': f'STAGE 3: Math score {best["math_score"]:.0f} below absolute minimum {ABSOLUTE_MIN_MATH} — AI cannot override',
                    'ranked_pairs': top_candidates,
                    'best_score': best['combined_score'],
                    'pipeline_stats': pipeline_stats
                }
            
            # === F&G EXTREME FEAR BLOCK ===
            # F&G < 5:  TRUE PANIC — hard block on ALL trades (capitulation/flash crash territory)
            # F&G 5-15: SHORTs allowed if math score >= 55 (shorting with the panic = valid play)
            #           LONGs still blocked (catching falling knives in a crash)
            # F&G >= 15: Normal pipeline continues
            # NOTE: Testing mode used to disable this (F&G<1 = never triggers).
            #       FIX: Testing mode should STILL protect against Extreme Fear LONGs!
            #       The whole point of testing is to validate the bot works correctly.
            _fg_news = self._get_news_context()
            _fg_value = _fg_news.get('fear_greed_index', 50)
            _fg_testing_mode = get_threshold_mode() == 'testing'
            _fg_hard_block = 3 if _fg_testing_mode else 3    # Block ALL trades at F&G < 3
            _fg_extreme_block = 12 if _fg_testing_mode else 15  # Block LONGs at F&G < 12 (testing) / 15 (normal)
            if _fg_value < _fg_hard_block:  # TRUE PANIC
                logger.warning(f"🚫 {best['symbol']}: F&G HARD BLOCK — Fear & Greed = {_fg_value} (TRUE PANIC < {_fg_hard_block}). No trading.")
                logger.info(f"📊 PIPELINE: {pipeline_stats}")
                return {
                    'has_opportunity': False,
                    'reason': f'STAGE 3: Fear & Greed = {_fg_value} (TRUE PANIC < {_fg_hard_block}) — hard block on ALL trades',
                    'ranked_pairs': top_candidates,
                    'best_score': best['combined_score'],
                    'pipeline_stats': pipeline_stats
                }
            elif _fg_value < _fg_extreme_block:  # EXTREME FEAR
                if best['direction'] == 'LONG':
                    # LONGs in extreme fear are risky (falling knives) BUT
                    # very high conviction signals should still be allowed through.
                    # Exception: Math >= 70 AND AI >= 70 = strong enough to override fear
                    _fg_math = best.get('math_score', 0)
                    _fg_ai = best.get('ai_confidence', best.get('combined_score', 0) - _fg_math)
                    _fg_combined = best.get('combined_score', 0)
                    _fg_override_math = 70 if _fg_testing_mode else 70
                    _fg_override_combined = 70 if _fg_testing_mode else 70
                    _fg_long_override = _fg_math >= _fg_override_math and _fg_combined >= _fg_override_combined
                    
                    if _fg_long_override:
                        logger.warning(f"⚠️ {best['symbol']}: F&G={_fg_value} EXTREME FEAR but LONG ALLOWED — exceptional scores (Math={_fg_math:.0f}, Combined={_fg_combined:.0f})")
                    else:
                        logger.warning(f"🚫 {best['symbol']}: F&G LONG BLOCK — Fear & Greed = {_fg_value} (EXTREME FEAR < {_fg_extreme_block}). Math={_fg_math:.0f} < {_fg_override_math} needed to override.")
                        logger.info(f"📊 PIPELINE: {pipeline_stats}")
                        return {
                            'has_opportunity': False,
                            'reason': f'STAGE 3: Fear & Greed = {_fg_value} (EXTREME FEAR) — LONGs blocked (Math {_fg_math:.0f} < {_fg_override_math} override threshold)',
                            'ranked_pairs': top_candidates,
                            'best_score': best['combined_score'],
                            'pipeline_stats': pipeline_stats
                        }
                elif best['direction'] != 'LONG':
                    _fg_short_min_math = 50 if _fg_testing_mode else 50
                    if best['math_score'] < _fg_short_min_math:
                        logger.warning(f"🚫 {best['symbol']}: F&G SHORT needs stronger math — Math {best['math_score']:.0f} < {_fg_short_min_math} at F&G={_fg_value}")
                        logger.info(f"📊 PIPELINE: {pipeline_stats}")
                        return {
                            'has_opportunity': False,
                            'reason': f'STAGE 3: Fear & Greed = {_fg_value} — SHORT needs math >= {_fg_short_min_math} (got {best["math_score"]:.0f})',
                            'ranked_pairs': top_candidates,
                            'best_score': best['combined_score'],
                            'pipeline_stats': pipeline_stats
                        }
                    else:
                        logger.info(f"✅ {best['symbol']}: F&G={_fg_value} — SHORT allowed (math={best['math_score']:.0f} >= {_fg_short_min_math})")
            
            # === THRESHOLDS FOR AI-APPROVED TRADES ===
            # Controlled by /threshold command (default vs loose)
            # ML veto (<40% win prob), RiskShield, and Sensible Reversal are the real safety nets
            is_weekend = self._get_market_hours_context().get('is_weekend', False)
            _thr = get_active_thresholds(is_weekend)
            MIN_COMBINED_SCORE = _thr['combined']
            MIN_MATH_SCORE = _thr['math']
            MIN_MATH_SCORE_AI_HIGH_CONF = _thr['ai_high_conf']
            MIN_MATH_SCORE_LONG_PREFERRED = _thr['long_preferred']
            MIN_MATH_SCORE_REVERSAL = _thr['reversal']
            
            ai_confidence = best.get('ai_confidence', 0.5)
            ai_direction = best.get('direction', '')
            ai_reversed = best.get('ai_reversed_direction', False)
            
            # NOTE: Removed market regime filter (Fear & Greed based blocking)
            # Reason: API unreliable, data stale (daily), and the math analysis
            # already accounts for momentum. The real fix is the reversal penalty below.
            
            # Get historical direction performance  
            dir_perf = self._get_direction_performance()
            long_wr = dir_perf.get('long_wr', 50)
            
            # === DIRECTION REVERSAL PENALTY ===
            # When AI reverses math's direction, require HIGHER math score
            # This prevents AI from fighting strong trends
            if ai_reversed:
                effective_min_math = MIN_MATH_SCORE_REVERSAL
                logger.warning(f"   ⚠️ AI REVERSED direction - requiring higher math: {effective_min_math}")
            # SPECIAL RULE: If AI says LONG and our LONG WR is excellent, trust the AI!
            # The math scores low for LONG in bearish momentum, but LONG historically wins
            elif ai_direction == 'LONG' and long_wr >= 65:
                effective_min_math = MIN_MATH_SCORE_LONG_PREFERRED
                logger.info(f"   📈 LONG preferred (WR={long_wr:.0f}%) - using lower math threshold: {effective_min_math}")
            elif ai_confidence >= 0.75:
                effective_min_math = MIN_MATH_SCORE_AI_HIGH_CONF
            else:
                effective_min_math = MIN_MATH_SCORE
            
            # Check math score - but give AI some flexibility with high confidence
            if best['math_score'] < effective_min_math:
                logger.warning(f"🚫 {best['symbol']}: Math score {best['math_score']:.0f} too low (min {effective_min_math}) - AI conf={ai_confidence:.0%}")
                logger.info(f"📊 PIPELINE: {pipeline_stats}")
                return {
                    'has_opportunity': False,
                    'reason': f'STAGE 3: Math score too low ({best["math_score"]:.0f} < {effective_min_math}) - AI cannot override',
                    'ranked_pairs': top_candidates,
                    'best_score': best['combined_score'],
                    'pipeline_stats': pipeline_stats
                }
            
            # When LONG is preferred, lower combined score requirement
            # (since math score will be low due to counter-trend design)
            MIN_COMBINED_SCORE_LONG_PREFERRED = 40 if is_weekend else 55
            effective_min_combined = MIN_COMBINED_SCORE_LONG_PREFERRED if (ai_direction == 'LONG' and long_wr >= 65) else MIN_COMBINED_SCORE
            
            # Require minimum combined score even with AI approval
            if best['combined_score'] >= effective_min_combined:
                # We have a winner! AI approved AND score is good
                logger.info(f"✅ ALL 3 STAGES PASSED! Approving {best['symbol']} {best['direction']}")
                logger.info(f"   Math: {best['math_score']:.0f} | AI_Analysis: {best.get('ai_analysis_score', best.get('ai_score', 0)):.0f} | Conf: {best.get('ai_confidence', 0):.0%} | Combined: {best['combined_score']:.0f}")
                logger.info(f"   AI Reasoning: {best.get('ai_reasoning', 'N/A')[:80]}")
                logger.info(f"📊 PIPELINE: {pipeline_stats}")
                
                return {
                    'has_opportunity': True,
                    'best_pair': best,
                    'ranked_pairs': top_candidates,
                    'ai_bias': best['direction'].lower(),
                    'market_regime': best['math_check'].get('detailed_analysis', {}).get('regime', 'UNKNOWN'),
                    'recommended_action': f"Open {best['direction']} on {best['symbol']}",
                    'pipeline_stats': pipeline_stats
                }
            else:
                logger.info(f"📊 PIPELINE: {pipeline_stats}")
                return {
                    'has_opportunity': False,
                    'reason': f'STAGE 3: AI approved {best["symbol"]} but combined score too low ({best["combined_score"]:.0f} < {effective_min_combined})',
                    'ranked_pairs': top_candidates,
                    'best_score': best['combined_score'],
                    'pipeline_stats': pipeline_stats
                }
                
        except Exception as e:
            logger.error(f"Unified Math+AI scan error: {e}")
            return {'has_opportunity': False, 'reason': str(e), 'ranked_pairs': []}
    
    def unified_position_decision(
        self,
        position: Dict[str, Any],
        df: pd.DataFrame,
        current_price: float,
        atr: float,
        atomic_candle=None
    ) -> Dict[str, Any]:
        """
        UNIFIED MATH + AI POSITION DECISION
        
        Makes hold/close/adjust decisions using BOTH Math and AI together.
        
        SMART EXIT RULES:
        1. IN PROFIT + High reversal risk → Exit to protect gains
        2. IN DEEP LOSS (>-1.5% or >-$15) → Force exit to prevent deeper loss
           (Server-side SL should have triggered - if we're here, exit manually)
        3. SMALL LOSS → Can exit if strong reversal confirmed
        4. AI + Math must BOTH agree for exit (except deep loss protection which is automatic)
        
        Args:
            position: {symbol, side, entry_price, size, tp1_hit, tp2_hit, ...}
            df: OHLCV DataFrame
            current_price: Current market price
            atr: Current ATR
            
        Returns:
            {
                'action': 'hold' | 'close' | 'tighten_sl',
                'confidence': 0.0-1.0,
                'math_score': float,
                'ai_validated': bool,
                'reasoning': str,
                'sl_adjustment': float or None
            }
        """
        try:
            symbol = position.get('symbol', 'UNKNOWN')
            side = position.get('side', 'LONG').upper()
            entry_price = position.get('entry_price', current_price)
            size = position.get('size', 0)
            tp1_hit = position.get('tp1_hit', False)
            tp2_hit = position.get('tp2_hit', False)
            
            # === GET PEAK PROFIT EARLY - Needed for multiple checks ===
            # Read from position dict (tracked by position monitor)
            peak_profit_pct = position.get('peak_profit_pct', 0)
            peak_profit_usd = position.get('peak_profit_usd', 0)
            
            # Calculate PnL for logging
            if side == 'LONG':
                quick_pnl = ((current_price - entry_price) / entry_price) * 100
            else:
                quick_pnl = ((entry_price - current_price) / entry_price) * 100
            
            logger.info(f"📊 POSITION CHECK: {symbol} {side} | Entry=${entry_price:.4f} | Now=${current_price:.4f} | PnL={quick_pnl:+.2f}%")
            
            # === S/R DISTANCE CHECK ===
            # Show how close we are to support and resistance levels
            try:
                sr_levels = self._detect_resistance_support_levels(df, current_price)
                dist_to_resistance = sr_levels.get('distance_to_resistance_pct', 99)
                dist_to_support = sr_levels.get('distance_to_support_pct', 99)
                range_position = sr_levels.get('position_in_range_pct', 50)
                in_r_zone = sr_levels.get('in_resistance_zone', False)
                in_s_zone = sr_levels.get('in_support_zone', False)
                
                # Build S/R status string
                sr_status = f"📍 Range: {range_position:.0f}% | "
                if in_r_zone:
                    sr_status += "🚫 IN RESISTANCE ZONE"
                elif in_s_zone:
                    sr_status += "🚫 IN SUPPORT ZONE"
                elif dist_to_resistance < 1.0:
                    sr_status += f"⚠️ Near R ({dist_to_resistance:.1f}% away)"
                elif dist_to_support < 1.0:
                    sr_status += f"⚠️ Near S ({dist_to_support:.1f}% away)"
                else:
                    sr_status += f"R: {dist_to_resistance:.1f}% | S: {dist_to_support:.1f}%"
                
                logger.info(f"📊 {symbol}: {sr_status}")
            except Exception as sr_err:
                logger.debug(f"S/R check error: {sr_err}")
            
            # Calculate PnL
            if side == 'LONG':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            
            # Estimate position value
            position_value = size * entry_price if size > 0 else 500
            pnl_usd = pnl_pct * position_value / 100
            
            # === MINIMUM HOLD TIME CHECK ===
            # Don't run reversal detection on freshly opened positions
            # Give trades at least 2 minutes to develop before considering exit
            import time
            entry_time = position.get('entry_time', 0) or position.get('open_time', 0)
            if entry_time:
                try:
                    if isinstance(entry_time, str):
                        # String datetime - parse with fromisoformat
                        entry_dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                        entry_timestamp = entry_dt.timestamp()
                    elif isinstance(entry_time, datetime):
                        # Datetime object - get timestamp directly
                        entry_timestamp = entry_time.timestamp()
                    else:
                        # Assume it's a numeric timestamp
                        entry_timestamp = float(entry_time)
                    hold_seconds = time.time() - entry_timestamp
                    logger.debug(f"📊 Hold time calculated: {hold_seconds:.0f}s (entry_type={type(entry_time).__name__})")
                except Exception as e:
                    logger.warning(f"⚠️ Could not calculate hold time: {e} (entry_time={entry_time}, type={type(entry_time)})")
                    hold_seconds = 999  # Assume old position
            else:
                hold_seconds = 999  # Assume old position
            
            MIN_HOLD_SECONDS = 30  # 30 seconds minimum before reversal detection
            is_new_position = hold_seconds < MIN_HOLD_SECONDS
            
            # === PHASE 0: SMART PROFIT MANAGEMENT (ZERO-LOSS STRATEGY) ===
            # REDESIGNED: Check peak profit FIRST - if we had good profit, protect it!
            # Only hold for more if momentum is STRONGLY in our favor AND we haven't peaked
            # The PROFIT FLOORS in realtime exit handle hard limits, this adds 30-sec awareness
            
            # Get peak profit for this position
            cached_peak_30s = self.get_peak_profit(symbol)
            if cached_peak_30s < peak_profit_pct:
                cached_peak_30s = peak_profit_pct
            
            if 0 < pnl_pct < 0.6:  # Small win (0% to 0.6%)
                # === GRACE PERIOD: Don't exit new positions in SMALL WIN zone ===
                if is_new_position:
                    logger.info(f"⏳ {symbol}: SMALL WIN +{pnl_pct:.2f}% but only {hold_seconds:.0f}s old — GRACE PERIOD (need {MIN_HOLD_SECONDS}s). HOLD.")
                    return {'action': 'hold', 'confidence': 0.5, 'reasoning': f'Grace period: {hold_seconds:.0f}s/{MIN_HOLD_SECONDS}s', 'sl_adjustment': None}
                
                logger.info(f"💰 {symbol}: SMALL WIN zone ({pnl_pct:.2f}%, peak: {cached_peak_30s:.2f}%) - checking momentum + pullback")
                try:
                    context = self._build_market_context(df, current_price, atr)
                    roc_5 = context.get('roc_5', 0)
                    roc_10 = context.get('roc_10', 0)
                    rsi = context.get('rsi', 50)
                    
                    sr_levels = self._detect_resistance_support_levels(df, current_price)
                    in_r_zone = sr_levels.get('in_resistance_zone', False)
                    in_s_zone = sr_levels.get('in_support_zone', False)
                    dist_to_resistance = sr_levels.get('distance_to_resistance_pct', 99)
                    dist_to_support = sr_levels.get('distance_to_support_pct', 99)
                    
                    # ZERO-LOSS CHECK: If we peaked higher and are pulling back, EXIT
                    # Require meaningful peak AND meaningful pullback before cutting winners
                    # OLD: peak>=0.20 pullback>=0.08 pnl>=0.12 → exited at +0.19% from +0.29% peak (too tight!)
                    pullback_from_peak = cached_peak_30s - pnl_pct
                    if cached_peak_30s >= 0.28 and pullback_from_peak >= 0.12 and pnl_pct >= 0.14:
                        # Exit if peak pulled back significantly
                        logger.warning(f"🛡️ {symbol}: PULLBACK EXIT! Peak +{cached_peak_30s:.2f}% → Now +{pnl_pct:.2f}% (lost {pullback_from_peak:.2f}%)")
                        return {
                            'action': 'close',
                            'confidence': 0.90,
                            'math_score': 0,
                            'exit_score': 90,
                            'ai_validated': True,
                            'reasoning': f"🛡️ PULLBACK: Peak +{cached_peak_30s:.2f}% → +{pnl_pct:.2f}%. Lost {pullback_from_peak:.2f}% from peak. Locking profit.",
                            'sl_adjustment': None
                        }
                    
                    # Check if momentum STILL FAVORS our direction
                    # For tiny profits (<0.15%), require STRONGER momentum to hold
                    # Otherwise we hold forever at +0.10% until it reverses to -0.10%
                    momentum_still_good = False
                    reversal_confirmed = False
                    
                    # Scale momentum requirement with profit level
                    # Tiny profit → need strong momentum to justify holding
                    # Good profit → can hold with moderate momentum
                    if pnl_pct < 0.15:
                        mom_threshold = 0.20   # Need strong momentum to hold tiny profit
                        mom_threshold_10 = 0.10
                    else:
                        mom_threshold = 0.10   # Moderate momentum OK for decent profit
                        mom_threshold_10 = 0.0
                    
                    if side == 'LONG':
                        # LONG: Good if momentum still up, bad if turning down
                        if roc_5 > mom_threshold and roc_10 > mom_threshold_10:
                            momentum_still_good = True
                            logger.info(f"   ✅ Momentum still UP (roc_5={roc_5:+.2f}%, roc_10={roc_10:+.2f}%) - HOLD")
                        elif roc_5 < -0.10 and (in_r_zone or dist_to_resistance < 0.5):
                            reversal_confirmed = True
                            logger.info(f"   🔴 REVERSAL: Momentum turning DOWN at resistance (roc_5={roc_5:+.2f}%) - EXIT")
                        elif roc_5 < -0.15:  # Need strong reversal
                            reversal_confirmed = True
                            logger.info(f"   🔴 REVERSAL: Downward momentum (roc_5={roc_5:+.2f}%) - EXIT")
                    else:
                        # SHORT: Good if momentum still down, bad if turning up
                        if roc_5 < -mom_threshold and roc_10 < -mom_threshold_10:
                            momentum_still_good = True
                            logger.info(f"   ✅ Momentum still DOWN (roc_5={roc_5:+.2f}%, roc_10={roc_10:+.2f}%) - HOLD")
                        elif roc_5 > 0.10 and (in_s_zone or dist_to_support < 0.5):  # Reversal at support
                            reversal_confirmed = True
                            logger.info(f"   🔴 REVERSAL: Momentum turning UP at support (roc_5={roc_5:+.2f}%) - EXIT")
                        elif roc_5 > 0.15:  # Need strong reversal
                            reversal_confirmed = True
                            logger.info(f"   🔴 REVERSAL: Upward momentum (roc_5={roc_5:+.2f}%) - EXIT")
                    
                    # === AGGRESSIVE CANDLE REVERSAL CHECK (even during grace period!) ===
                    # A sharp 1-bar crash should override HOLD even with good momentum
                    # This catches sudden dumps/pumps that ROC5 hasn't reflected yet
                    aggressive_candle_reversal = False
                    if len(df) >= 3:
                        c_last = df.iloc[-1]
                        c_prev = df.iloc[-2]
                        c_body_pct = abs(c_last['close'] - c_last['open']) / c_last['open'] * 100 if c_last['open'] > 0 else 0
                        c_wick_against = 0
                        
                        if side == 'LONG':
                            # Big red candle OR close below previous low = aggressive reversal
                            if c_last['close'] < c_last['open'] and c_body_pct > 0.15:  # Strong red candle
                                aggressive_candle_reversal = True
                                logger.warning(f"   🔴⚡ AGGRESSIVE CANDLE: Big red bar ({c_body_pct:.2f}%)!")
                            elif c_last['close'] < c_prev['low'] and c_body_pct > 0.10:  # Close below prev low with body
                                aggressive_candle_reversal = True
                                logger.warning(f"   🔴⚡ AGGRESSIVE CANDLE: Close {c_last['close']:.4f} below prev low {c_prev['low']:.4f}!")
                        else:
                            # SHORT: Big green candle OR close above previous high
                            if c_last['close'] > c_last['open'] and c_body_pct > 0.15:  # Strong green candle
                                aggressive_candle_reversal = True
                                logger.warning(f"   🟢⚡ AGGRESSIVE CANDLE: Big green bar ({c_body_pct:.2f}%)!")
                            elif c_last['close'] > c_prev['high'] and c_body_pct > 0.10:  # Close above prev high with body
                                aggressive_candle_reversal = True
                                logger.warning(f"   🟢⚡ AGGRESSIVE CANDLE: Close {c_last['close']:.4f} above prev high {c_prev['high']:.4f}!")
                    
                    # Override HOLD if aggressive candle detected (even with good momentum)
                    if aggressive_candle_reversal and not reversal_confirmed:
                        reversal_confirmed = True
                        momentum_still_good = False
                        logger.warning(f"   ⚡ CANDLE OVERRIDE: Aggressive reversal overrides good momentum!")
                    
                    # Also check: S/R zone danger (approaching resistance for LONG, support for SHORT)
                    sr_danger = False
                    if side == 'LONG' and (in_r_zone or dist_to_resistance < 0.3):
                        sr_danger = True
                        logger.info(f"   ⚠️ LONG at resistance zone (dist={dist_to_resistance:.2f}%) - risky to hold")
                    elif side == 'SHORT' and (in_s_zone or dist_to_support < 0.3):
                        sr_danger = True
                        logger.info(f"   ⚠️ SHORT at support zone (dist={dist_to_support:.2f}%) - risky to hold")
                    
                    # DECISION LOGIC — require meaningful profit before reversal exit
                    # Raised from 0.12% → 0.15%: 0.12% barely covers fees and gave +0.15% avg win vs -0.28% avg loss
                    if reversal_confirmed and pnl_pct >= 0.15:
                        # Clear reversal with meaningful profit - EXIT
                        logger.info(f"✅ {symbol}: REVERSAL CONFIRMED at +{pnl_pct:.2f}% - locking profit!")
                        return {
                            'action': 'close',
                            'confidence': 0.88,
                            'math_score': 0,
                            'exit_score': 88,
                            'ai_validated': True,
                            'reasoning': f"🔴 REVERSAL at +{pnl_pct:.2f}%: Momentum turning against us. Locking profit.",
                            'sl_adjustment': None
                        }
                    elif sr_danger and pnl_pct >= 0.15:
                        # At S/R zone with profit covering fees - take it before bounce
                        logger.info(f"🎯 {symbol}: S/R DANGER + profit +{pnl_pct:.2f}% - taking profit")
                        return {
                            'action': 'close',
                            'confidence': 0.85,
                            'math_score': 0,
                            'exit_score': 85,
                            'ai_validated': True,
                            'reasoning': f"🎯 S/R DANGER: At key level with +{pnl_pct:.2f}% profit. High bounce/rejection risk.",
                            'sl_adjustment': None
                        }
                    elif momentum_still_good and not sr_danger:
                        # Momentum still in our favor AND not at S/R danger - HOLD
                        # BUT: Populate momentum cache FIRST so real-time WS exit has data!
                        ema_20 = context.get('ema_20', 0)
                        ema_trend_val = current_price > ema_20 if ema_20 > 0 else True
                        is_long_dir = side == 'LONG'
                        accel_against = (is_long_dir and roc_5 < roc_10 < 0) or (not is_long_dir and roc_5 > roc_10 > 0)
                        # Compute prev_rsi (~5 bars ago ≈ 1 min on 1m candles) for RSI delta
                        _prev_rsi = df['rsi'].iloc[-6] if 'rsi' in df.columns and len(df) >= 6 else rsi
                        _vol_ratio = context.get('volume_ratio', 1.0)
                        self.update_momentum_cache(
                            symbol=symbol, roc_5=roc_5, roc_10=roc_10,
                            rsi=rsi, ema_trend=ema_trend_val,
                            momentum_hope=20, accelerating_against=accel_against,
                            prev_rsi=_prev_rsi, volume_ratio=_vol_ratio,
                            atomic_candle=atomic_candle
                        )
                        # Also cache context for RT exit fallback
                        self._last_context[symbol] = {'roc_5': roc_5, 'roc_10': roc_10, 'rsi': rsi}
                        logger.info(f"🔒 {symbol}: MOMENTUM STILL GOOD → HOLD (cache populated for RT exit)")
                        return {
                            'action': 'hold',
                            'confidence': 0.75,
                            'math_score': 0,
                            'exit_score': 0,
                            'ai_validated': False,
                            'reasoning': f"💰 WIN ({pnl_pct:.2f}%) momentum favors us, no S/R danger - holding.",
                            'sl_adjustment': None
                        }
                    else:
                        # Neutral momentum - if we have meaningful profit above fees, lean toward exit
                        # Raised from 0.15% → 0.18%: need real profit to justify neutral exit
                        if pnl_pct >= 0.18:
                            logger.info(f"⚖️ {symbol}: NEUTRAL momentum at +{pnl_pct:.2f}% - locking profit (zero loss)")
                            return {
                                'action': 'close',
                                'confidence': 0.80,
                                'math_score': 0,
                                'exit_score': 80,
                                'ai_validated': True,
                                'reasoning': f"⚖️ NEUTRAL EXIT: +{pnl_pct:.2f}% with neutral momentum. ZERO LOSS = take the win.",
                                'sl_adjustment': None
                            }
                        elif pnl_pct >= 0.15 and sr_danger:
                            # Small profit at S/R danger zone — take it before bounce
                            logger.info(f"⚖️ {symbol}: NEUTRAL momentum at +{pnl_pct:.2f}% at S/R danger - exiting")
                            return {
                                'action': 'close',
                                'confidence': 0.78,
                                'math_score': 0,
                                'exit_score': 78,
                                'ai_validated': True,
                                'reasoning': f"⚖️ S/R NEUTRAL EXIT: +{pnl_pct:.2f}% at key S/R level, neutral momentum. Take small win.",
                                'sl_adjustment': None
                            }
                        elif pnl_pct >= 0.12 and hold_seconds > 120:
                            # Held for 2+ minutes at tiny profit, no momentum — stale trade
                            logger.info(f"⚖️ {symbol}: NEUTRAL momentum at +{pnl_pct:.2f}% held {hold_seconds:.0f}s - stale, exiting")
                            return {
                                'action': 'close',
                                'confidence': 0.75,
                                'math_score': 0,
                                'exit_score': 75,
                                'ai_validated': True,
                                'reasoning': f"⚖️ STALE EXIT: +{pnl_pct:.2f}% for {hold_seconds:.0f}s with neutral momentum. Won't grow → take it.",
                                'sl_adjustment': None
                            }
                        else:
                            logger.info(f"⚖️ {symbol}: NEUTRAL momentum at +{pnl_pct:.2f}% - let AI decide")
                            # Continue to normal exit logic
                
                except Exception as tiny_err:
                    logger.warning(f"Small win check failed: {tiny_err}")
            
            # === PHASE 0B: BIGGER WIN MANAGEMENT (0.6%+) - ZERO LOSS STRATEGY ===
            # Positions with 0.6%+ profit MUST be protected aggressively
            # Only hold if momentum is STRONGLY with us AND no pullback from peak
            elif pnl_pct >= 0.6:
                logger.info(f"🏆 {symbol}: BIG WIN zone ({pnl_pct:.2f}%, peak: {cached_peak_30s:.2f}%) - tight protection!")
                try:
                    context = self._build_market_context(df, current_price, atr)
                    roc_5 = context.get('roc_5', 0)
                    roc_10 = context.get('roc_10', 0)
                    
                    # Check pullback from peak
                    pullback_from_peak = cached_peak_30s - pnl_pct
                    
                    # At 0.6%+, any pullback > 0.15% from peak = EXIT
                    if cached_peak_30s >= 0.6 and pullback_from_peak >= 0.15:
                        logger.warning(f"🏆🛡️ {symbol}: BIG WIN PULLBACK! Peak +{cached_peak_30s:.2f}% → +{pnl_pct:.2f}% (lost {pullback_from_peak:.2f}%)")
                        return {
                            'action': 'close',
                            'confidence': 0.95,
                            'math_score': 0,
                            'exit_score': 95,
                            'ai_validated': True,
                            'reasoning': f"🏆 BIG WIN PROTECT: Peak +{cached_peak_30s:.2f}% → +{pnl_pct:.2f}%. Lost {pullback_from_peak:.2f}% from peak. LOCKING!",
                            'sl_adjustment': None
                        }
                    
                    # Check if momentum is even slightly against us
                    is_long = side == 'LONG'
                    momentum_against = False
                    if is_long and roc_5 < -0.05:
                        momentum_against = True
                    elif not is_long and roc_5 > 0.05:
                        momentum_against = True
                    
                    if momentum_against:
                        logger.warning(f"🏆📉 {symbol}: BIG WIN + MOMENTUM FADING! +{pnl_pct:.2f}% at risk (ROC5={roc_5:+.2f}%)")
                        return {
                            'action': 'close',
                            'confidence': 0.93,
                            'math_score': 0,
                            'exit_score': 93,
                            'ai_validated': True,
                            'reasoning': f"🏆 BIG WIN FADE: +{pnl_pct:.2f}% profit with momentum turning (ROC5={roc_5:+.2f}%). LOCK IT!",
                            'sl_adjustment': None
                        }
                    
                    # Momentum with us - hold but log it
                    logger.info(f"🏆✅ {symbol}: BIG WIN +{pnl_pct:.2f}% with momentum (ROC5={roc_5:+.2f}%) - holding carefully")
                    # Don't return HOLD - let it flow to Phase 1C reversal detection for extra safety
                    
                except Exception as big_err:
                    logger.warning(f"Big win check failed: {big_err}")
            
            # NOTE: We NO LONGER early-return HOLD for new positions!
            # The old code skipped everything for positions < 30s, which meant:
            #   1. Momentum cache never populated → RT exit had no data
            #   2. Peak tracking in ai_filter cache not synced
            #   3. Reversal detection completely blind for first 30s
            # Now: We ALWAYS flow through to populate momentum cache + basic checks.
            # The grace period is enforced INSIDE _check_fast_reversal (won't trigger
            # aggressive exits) and the loss thresholds are wider for new positions.
            if is_new_position and pnl_pct > -0.5:
                logger.info(f"📊 {symbol}: New position ({hold_seconds:.0f}s old) - monitoring with grace period (wider thresholds)")
                # Don't return early! Flow through to populate momentum cache
                # The is_new_position flag is passed to _check_fast_reversal
                # which uses wider thresholds during the grace period
            
            # === PHASE 1: LOSS MANAGEMENT - WITH GRACE PERIOD + RECOVERY ===
            # During grace period: Only exit on CATASTROPHIC loss (>5%) 
            # Give trades at least 30s to recover from initial spread/slippage
            # After grace period: Use normal tighter thresholds
            
            # LOSS THRESHOLDS - scale USD limits based on position value
            # The % threshold is the PRIMARY control, USD is a safety net
            if is_new_position:
                # GRACE PERIOD: Only exit on truly catastrophic moves
                # Normal volatility (0.1-0.5%) should be tolerated for recovery
                max_loss_threshold_pct = -5.0   # Only exit on catastrophic moves during grace
                max_loss_threshold_usd = -(position_value * 5.0 / 100)  # Aligned with 5%
                
                # Check momentum to see if recovery is likely
                try:
                    grace_context = self._build_market_context(df, current_price, atr)
                    grace_roc5 = grace_context.get('roc_5', 0)
                    is_long = side == 'LONG'
                    momentum_recovering = (is_long and grace_roc5 > -0.3) or (not is_long and grace_roc5 < 0.3)
                    
                    if pnl_pct < -0.3 and not momentum_recovering:
                        # In loss AND momentum is strongly against us — use tighter threshold
                        max_loss_threshold_pct = -2.5
                        max_loss_threshold_usd = -(position_value * 2.5 / 100)
                        logger.info(f"⏳ {symbol}: Grace period but momentum against (ROC5={grace_roc5:+.2f}%) - tighter threshold: {max_loss_threshold_pct}%")
                    else:
                        logger.debug(f"⏳ {symbol}: Grace period - wide loss threshold: {max_loss_threshold_pct}% (recovery possible, ROC5={grace_roc5:+.2f}%)")
                except Exception:
                    max_loss_threshold_pct = -2.5  # Fallback if momentum check fails
                    max_loss_threshold_usd = -(position_value * 2.5 / 100)
                    logger.debug(f"⏳ {symbol}: Grace period - fallback threshold: {max_loss_threshold_pct}%")
            else:
                # AFTER GRACE PERIOD: Use tighter but still reasonable thresholds
                max_loss_threshold_pct = -1.5   # 1.5% loss after grace period
                max_loss_threshold_usd = -(position_value * 1.5 / 100)  # Aligned with 1.5%
            
            in_deep_loss = (pnl_pct < max_loss_threshold_pct) or (pnl_usd < max_loss_threshold_usd)
            
            if in_deep_loss:
                # BEYOND ACCEPTABLE LOSS - EXIT NOW to prevent deeper loss
                # The server-side SL should have triggered - if we're here, exit manually
                grace_status = "DURING GRACE" if is_new_position else "AFTER GRACE"
                logger.warning(f"🚨 {symbol}: LOSS EXCEEDED THRESHOLD {grace_status} (PnL: {pnl_pct:.2f}% / ${pnl_usd:.2f}) - EXITING!")
                return {
                    'action': 'close',
                    'confidence': 0.95,
                    'math_score': 0,
                    'exit_score': 100,
                    'ai_validated': True,
                    'reasoning': f"🚨 LOSS PROTECTION: {pnl_pct:.2f}% loss exceeds {max_loss_threshold_pct}% threshold ({grace_status})",
                    'sl_adjustment': None,
                    'pnl_pct': pnl_pct,
                    'pnl_usd': pnl_usd
                }
            
            # === PHASE 0.4: PROFIT-TO-LOSS EARLY EXIT ===
            # CRITICAL: If we HAD profit and now in loss with momentum against, exit IMMEDIATELY
            # This catches scenarios like XPL: Peak +0.10%, now -0.37%, momentum against
            # Don't wait for recovery logic - this is a reversal situation!
            # FIXED: Raised threshold from 0.05% to 0.20% - tiny peaks are noise!
            # FIX: During grace period (<45s), suppress this for small peaks (<0.4%)
            # Small peaks on new trades are normal price noise, not reversals
            is_profit_to_loss = pnl_pct < -0.1 and peak_profit_pct > 0.20  # Was profitable, now in loss
            if is_profit_to_loss:
                # FIX: New positions with small peaks — let them develop
                if is_new_position and peak_profit_pct < 0.40:
                    logger.info(f"🚨⏳ {symbol}: PROFIT→LOSS suppressed (grace) - Peak +{peak_profit_pct:.2f}% on {position_age_seconds:.0f}s trade is noise")
                else:
                    # Check if momentum is against our position
                    context_p2l = self._build_market_context(df, current_price, atr)
                    roc_5_p2l = context_p2l.get('roc_5', 0)
                    roc_10_p2l = context_p2l.get('roc_10', 0)
                    
                    # For LONG: negative ROC means price dropping (against)
                    # For SHORT: positive ROC means price rising (against)
                    is_long = side == 'LONG'
                    strong_against = (is_long and roc_5_p2l < -0.2) or (not is_long and roc_5_p2l > 0.2)
                    sustained_against = (is_long and roc_5_p2l < 0 and roc_10_p2l < 0) or (not is_long and roc_5_p2l > 0 and roc_10_p2l > 0)
                    
                    if strong_against or sustained_against:
                        # 🔮 ATOMIC: Candle ROC says against, but check if ticks show recovery starting
                        _p2l_vetoed = False
                        if atomic_candle is not None:
                            try:
                                _p2l_pred = atomic_candle._check_exit_signal(symbol, side.lower())
                                if _p2l_pred['verdict'] == 'HOLD_STRONG' and _p2l_pred['urgency'] < 15 and _p2l_pred['exhaustion_level'] < 20:
                                    _p2l_vetoed = True
                                    logger.info(f"🔮🚨 {symbol}: PROFIT→LOSS VETOED by atomic — ticks show recovery! urgency={_p2l_pred['urgency']}, verdict={_p2l_pred['verdict']}")
                            except Exception:
                                pass
                        
                        if not _p2l_vetoed:
                            logger.warning(f"🚨 {symbol}: PROFIT→LOSS EXIT! Peak +{peak_profit_pct:.2f}% → Now {pnl_pct:.2f}% | Momentum against (ROC5={roc_5_p2l:+.2f}%)")
                            return {
                                'action': 'close',
                                'confidence': 0.88,
                                'math_score': 0,
                                'exit_score': 88,
                                'ai_validated': True,
                                'reasoning': f"🚨 PROFIT→LOSS: Peak +{peak_profit_pct:.2f}% → {pnl_pct:.2f}%, momentum against (ROC5={roc_5_p2l:+.2f}%)",
                                'sl_adjustment': None,
                                'pnl_pct': pnl_pct,
                                'pnl_usd': pnl_usd
                            }
            
            # === PHASE 0.5: ADVANCED POSITION RECOVERY CHECK ===
            # EXPANDED to cover -1.5% to -0.15% (increased from -1.2%)
            # This prevents positions from sitting in a "dead zone" with no logic
            # Uses MATH (momentum + S/R) + AI validation for smart recovery decisions
            in_recovery_zone = -1.5 < pnl_pct < -0.15 and pnl_usd > -10
            
            # === MODERATE LOSS ZONE: -0.5% to -1.5% ===
            # IMPORTANT: If in MODERATE LOSS with MOMENTUM AGAINST, exit early!
            # Don't let small losses become big losses
            in_moderate_loss = -1.5 <= pnl_pct < -0.5
            
            if in_moderate_loss:
                # Quick momentum check - if strongly against, don't wait for recovery
                context = self._build_market_context(df, current_price, atr)
                roc_5 = context.get('roc_5', 0)
                roc_10 = context.get('roc_10', 0)
                
                # Check if momentum is strongly against our position
                # MORE AGGRESSIVE: Exit if EITHER condition is met:
                # 1. ROC5 is STRONGLY against (> 0.5%) - immediate danger
                # 2. Both ROC5 and ROC10 are moderately against - sustained pressure
                momentum_strongly_against = False
                if side == 'LONG':
                    # For LONG: Exit if ROC5 is very negative OR both ROC5 and ROC10 are negative
                    if roc_5 < -0.5:  # Strong short-term bearish = EXIT
                        momentum_strongly_against = True
                    elif roc_5 < -0.2 and roc_10 < -0.1:  # Both bearish = EXIT
                        momentum_strongly_against = True
                else:  # SHORT
                    # For SHORT: Exit if ROC5 is very positive OR both ROC5 and ROC10 are positive
                    if roc_5 > 0.5:  # Strong short-term bullish = EXIT
                        momentum_strongly_against = True
                    elif roc_5 > 0.2 and roc_10 > 0.1:  # Both bullish = EXIT
                        momentum_strongly_against = True
                
                if momentum_strongly_against:
                    logger.warning(f"🚨 {symbol}: MODERATE LOSS EXIT! PnL={pnl_pct:.2f}% with momentum against (ROC5={roc_5:+.2f}%, ROC10={roc_10:+.2f}%)")
                    return {
                        'action': 'close',
                        'confidence': 0.85,
                        'math_score': 0,
                        'exit_score': 85,
                        'ai_validated': True,
                        'reasoning': f"🚨 MODERATE LOSS + MOMENTUM AGAINST: {pnl_pct:.2f}% loss with ROC5={roc_5:+.2f}%, ROC10={roc_10:+.2f}%",
                        'sl_adjustment': None,
                        'pnl_pct': pnl_pct,
                        'pnl_usd': pnl_usd
                    }
                else:
                    # Momentum is with us or neutral - check recovery potential
                    logger.info(f"🔄 {symbol}: MODERATE LOSS ({pnl_pct:.2f}%) but momentum OK (ROC5={roc_5:+.2f}%) - checking recovery")
            
            if in_recovery_zone:
                # Build context for recovery decision
                context = self._build_market_context(df, current_price, atr)
                roc_5 = context.get('roc_5', 0)
                roc_10 = context.get('roc_10', 0)
                roc_20 = context.get('roc_20', 0)  # Longer-term momentum
                rsi = context.get('rsi', 50)
                volume_ratio = context.get('volume_ratio', 1.0)
                
                # Get S/R info
                try:
                    sr_levels = self._detect_resistance_support_levels(df, current_price)
                    dist_to_support = sr_levels.get('distance_to_support_pct', 99)
                    dist_to_resistance = sr_levels.get('distance_to_resistance_pct', 99)
                    in_support_zone = sr_levels.get('in_support_zone', False)
                    in_resistance_zone = sr_levels.get('in_resistance_zone', False)
                except Exception:
                    dist_to_support = 99
                    dist_to_resistance = 99
                    in_support_zone = False
                    in_resistance_zone = False
                
                # === ADVANCED MOMENTUM ANALYSIS ===
                # Check if momentum is REVERSING (turning in our favor)
                momentum_reversing = False
                momentum_strength = 0
                
                if side == 'LONG':
                    # For LONG: Need momentum turning UP (positive acceleration)
                    if roc_5 > 0 and roc_5 > roc_10:
                        momentum_reversing = True
                        momentum_strength = min(abs(roc_5), 30)  # Cap at 30 points
                    elif roc_10 > 0 and roc_10 > roc_20:
                        momentum_reversing = True
                        momentum_strength = min(abs(roc_10) * 0.5, 15)  # Less weight for slower momentum
                else:
                    # For SHORT: Need momentum turning DOWN (negative acceleration)
                    if roc_5 < 0 and roc_5 < roc_10:
                        momentum_reversing = True
                        momentum_strength = min(abs(roc_5), 30)
                    elif roc_10 < 0 and roc_10 < roc_20:
                        momentum_reversing = True
                        momentum_strength = min(abs(roc_10) * 0.5, 15)
                
                # Calculate recovery score (0-100)
                recovery_score = 0
                recovery_reasons = []
                
                # === MOMENTUM FACTORS (0-45 points) ===
                if momentum_reversing:
                    recovery_score += momentum_strength
                    recovery_reasons.append(f"🔄 Momentum REVERSING in our favor (strength: {momentum_strength:.0f})")
                elif side == 'LONG' and roc_5 > 0:
                    recovery_score += 15
                    recovery_reasons.append("📈 Short-term momentum turning up")
                elif side == 'SHORT' and roc_5 < 0:
                    recovery_score += 15
                    recovery_reasons.append("📉 Short-term momentum turning down")
                
                # === RSI FACTORS (0-20 points) ===
                if side == 'LONG':
                    if rsi < 30:
                        recovery_score += 20
                        recovery_reasons.append(f"✅ RSI EXTREME oversold ({rsi:.0f}) - strong bounce expected")
                    elif rsi < 40:
                        recovery_score += 12
                        recovery_reasons.append(f"✅ RSI oversold ({rsi:.0f}) - bounce likely")
                else:
                    if rsi > 70:
                        recovery_score += 20
                        recovery_reasons.append(f"✅ RSI EXTREME overbought ({rsi:.0f}) - strong pullback expected")
                    elif rsi > 60:
                        recovery_score += 12
                        recovery_reasons.append(f"✅ RSI overbought ({rsi:.0f}) - pullback likely")
                
                # === S/R FACTORS (0-30 points) ===
                if side == 'LONG':
                    if in_support_zone:
                        recovery_score += 30
                        recovery_reasons.append(f"💪 IN SUPPORT ZONE - very strong bounce zone")
                    elif dist_to_support < 0.5:
                        recovery_score += 25
                        recovery_reasons.append(f"💪 Near support ({dist_to_support:.1f}% away) - strong bounce zone")
                    elif dist_to_support < 1.5:
                        recovery_score += 15
                        recovery_reasons.append(f"📍 Approaching support ({dist_to_support:.1f}% away)")
                    # Penalty
                    if in_resistance_zone:
                        recovery_score -= 25
                        recovery_reasons.append("⚠️ At resistance - recovery very difficult")
                    elif dist_to_resistance < 1.0:
                        recovery_score -= 15
                        recovery_reasons.append("⚠️ Near resistance - limited upside")
                else:
                    if in_resistance_zone:
                        recovery_score += 30
                        recovery_reasons.append(f"💪 IN RESISTANCE ZONE - very strong rejection zone")
                    elif dist_to_resistance < 0.5:
                        recovery_score += 25
                        recovery_reasons.append(f"💪 Near resistance ({dist_to_resistance:.1f}% away) - strong rejection zone")
                    elif dist_to_resistance < 1.5:
                        recovery_score += 15
                        recovery_reasons.append(f"📍 Approaching resistance ({dist_to_resistance:.1f}% away)")
                    # Penalty
                    if in_support_zone:
                        recovery_score -= 25
                        recovery_reasons.append("⚠️ At support - recovery very difficult")
                    elif dist_to_support < 1.0:
                        recovery_score -= 15
                        recovery_reasons.append("⚠️ Near support - limited downside")
                
                # === VOLUME CONFIRMATION (0-10 points) ===
                if volume_ratio > 1.5:
                    recovery_score += 10
                    recovery_reasons.append(f"📊 High volume ({volume_ratio:.1f}x) - strong conviction")
                elif volume_ratio < 0.7:
                    recovery_score -= 5
                    recovery_reasons.append(f"⚠️ Low volume ({volume_ratio:.1f}x) - weak move")
                
                # === DECISION LOGIC ===
                # Math score >= 50: Strong recovery signals, ask AI for validation
                # Math score 40-49: Borderline, ask AI to decide
                # Math score < 40: Too weak, exit normally
                
                logger.info(f"🔄 {symbol}: RECOVERY CHECK (PnL: {pnl_pct:.2f}%) | Math Score: {recovery_score}/100")
                for reason in recovery_reasons[:4]:  # Show top 4 reasons
                    logger.info(f"   {reason}")
                
                if recovery_score >= 40:
                    # === AI VALIDATION FOR RECOVERY ===
                    # Ask AI to validate the recovery decision
                    ai_recovery_decision = None
                    
                    if self.use_ai and recovery_score >= 40:
                        logger.info(f"🤖 {symbol}: Consulting AI for recovery validation (math score: {recovery_score})")
                        
                        # Extract real-time tracking data from position
                        deepest_loss_pct = position.get('deepest_loss_pct', pnl_pct)
                        deepest_loss_usd = position.get('deepest_loss_usd', pnl_usd)
                        recovery_attempts = position.get('recovery_attempts', 0)
                        in_recovery_mode = position.get('in_recovery_mode', False)
                        urgency = position.get('urgency', 'normal')
                        
                        # Calculate if recovering
                        is_recovering = deepest_loss_pct < pnl_pct < 0  # Loss getting smaller
                        recovery_progress = abs(pnl_pct - deepest_loss_pct) if is_recovering else 0
                        
                        # Build recovery-specific AI prompt with real-time context
                        recovery_context = f"""You are analyzing a position that's in a SMALL LOSS and deciding if it should be HELD for RECOVERY.

Position: {side} {symbol}
Entry: ${entry_price:.4f} → Current: ${current_price:.4f}
Loss: {pnl_pct:.2f}% (${pnl_usd:.2f})

REAL-TIME TRACKING:
• Deepest Loss: {deepest_loss_pct:.2f}% (worst point)
• Recovery Status: {'✅ RECOVERING (+' + f'{recovery_progress:.2f}%)' if is_recovering else '❌ STILL DEEPENING'}
• Recovery Attempts: #{recovery_attempts} (times AI has evaluated this)
• Already in Recovery Mode: {'YES' if in_recovery_mode else 'NO'}
• Urgency Level: {urgency.upper()} {'🚨' if urgency == 'urgent' else '⚠️' if urgency == 'high' else ''}

MATH RECOVERY SCORE: {recovery_score}/100 (40+ = consider recovery)

Recovery Factors Detected:
{chr(10).join(recovery_reasons[:5])}

Market Context:
• Momentum: roc_5={roc_5:+.2f}%, roc_10={roc_10:+.2f}%, roc_20={roc_20:+.2f}%
• RSI: {rsi:.0f} ({('OVERSOLD' if rsi < 40 else 'OVERBOUGHT' if rsi > 60 else 'NEUTRAL')})
• S/R: {dist_to_resistance:.1f}% to R | {dist_to_support:.1f}% to S
• Volume: {volume_ratio:.1f}x average

QUESTION: Should we HOLD this position for recovery, or EXIT to prevent further loss?

Consider:
1. Is momentum TRULY reversing in our favor?
2. Are we at a KEY support/resistance level likely to bounce/reject?
3. Is the loss small enough that a reversal could quickly turn profitable?
4. Are there any hidden risks (trend, volume, market conditions)?

Your decision (respond EXACTLY in this format):
<DECISION>
Action: HOLD_RECOVERY or EXIT
Confidence: [0-100]
Reasoning: [Your 1-sentence analysis]
</DECISION>"""
                        
                        try:
                            ai_response = self._call_ai_api(recovery_context)
                            
                            if ai_response and '<DECISION>' in ai_response:
                                decision_text = ai_response.split('<DECISION>')[1].split('</DECISION>')[0].strip()
                                
                                # Parse AI decision
                                ai_action = None
                                ai_confidence = 0
                                ai_reasoning = ""
                                
                                for line in decision_text.split('\n'):
                                    if 'Action:' in line:
                                        ai_action = 'hold_recovery' if 'HOLD' in line.upper() else 'close'
                                    elif 'Confidence:' in line:
                                        try:
                                            ai_confidence = int(''.join(filter(str.isdigit, line)))
                                        except Exception:
                                            ai_confidence = 50
                                    elif 'Reasoning:' in line:
                                        ai_reasoning = line.split('Reasoning:')[1].strip()
                                
                                logger.info(f"🤖 AI Recovery Decision: {ai_action.upper()} ({ai_confidence}%) - {ai_reasoning[:80]}")
                                
                                # AI validates recovery
                                if ai_action == 'hold_recovery' and ai_confidence >= 55:
                                    return {
                                        'action': 'hold_recovery',
                                        'confidence': min(0.85, ai_confidence / 100),
                                        'math_score': recovery_score,
                                        'exit_score': 0,
                                        'ai_validated': True,
                                        'reasoning': f"🔄 AI+MATH RECOVERY: {ai_reasoning[:100]}. Math: {recovery_score}/100, AI: {ai_confidence}%",
                                        'sl_adjustment': None,
                                        'pnl_pct': pnl_pct,
                                        'pnl_usd': pnl_usd,
                                        'recovery_score': recovery_score,
                                        'ai_confidence': ai_confidence
                                    }
                                else:
                                    logger.info(f"❌ AI rejected recovery: {ai_reasoning}")
                        
                        except Exception as ai_err:
                            logger.warning(f"⚠️ AI recovery validation failed: {ai_err}")
                    
                    # Fallback: If AI unavailable or didn't validate, use math score
                    if recovery_score >= 60:
                        # Very high math score - hold even without AI
                        return {
                            'action': 'hold_recovery',
                            'confidence': 0.70,
                            'math_score': recovery_score,
                            'exit_score': 0,
                            'ai_validated': False,
                            'reasoning': f"🔄 MATH RECOVERY: Strong signals (score: {recovery_score}/100). " + "; ".join(recovery_reasons[:2]),
                            'sl_adjustment': None,
                            'pnl_pct': pnl_pct,
                            'pnl_usd': pnl_usd,
                            'recovery_score': recovery_score
                        }
                else:
                    logger.info(f"❌ {symbol}: NO RECOVERY - Math score {recovery_score}/100 too low (need 40+)")
            
            # === PHASE 1B: SMART PROFIT PROTECTION ===
            # CRITICAL: This is NOT about hitting a profit number!
            # It's about detecting WHEN profit is REVERSING and capturing it!
            # 
            # LOGIC: 
            # - Track momentum using roc_5 and roc_10 (rate of change)
            # - If we're in good profit AND momentum is reversing against us = EXIT
            # - Don't exit just because profit hit $8 - exit when $8 profit starts DROPPING
            #
            # The AI should react FAST when it sees:
            # 1. Good profit ($5+ or 1%+)
            # 2. Momentum turning against the position
            # 3. Price dropping from recent highs (for LONG) or rising from lows (for SHORT)
            
            # === PHASE 1C: INTELLIGENT PROFIT REVERSAL DETECTION ===
            # DYNAMIC SYSTEM: Track peak profit and detect drawdown from peak
            # Using PERCENTAGE as primary metric (like Bybit display: $0.05 (+0.72%))
            # This is better because:
            # 1. Independent of position size
            # 2. Matches exchange display
            # 3. 0.72% peak → 0.10% current = 86% of gain lost (clear signal!)
            
            # Use existing momentum indicators (roc_5, roc_10) for faster reaction
            context = self._build_market_context(df, current_price, atr)
            roc_5 = context.get('roc_5', 0)   # Rate of change over 5 bars
            roc_10 = context.get('roc_10', 0)  # Rate of change over 10 bars
            
            # CACHE context for soft loss cap fallback (in case momentum_cache isn't populated yet)
            self._last_context[symbol] = {'roc_5': roc_5, 'roc_10': roc_10, 'rsi': context.get('rsi', 50)}
            
            # === TRACK PEAK PROFIT using PERCENTAGE (primary metric) ===
            # peak_profit_pct already initialized at function start from position dict
            
            # Also check cached peak in case WebSocket tracked a higher peak
            cached_peak = self.get_peak_profit(symbol)
            if cached_peak > peak_profit_pct:
                peak_profit_pct = cached_peak
            
            # Update peak if current profit is higher (use PCT as primary)
            if pnl_pct > peak_profit_pct and pnl_pct > 0:  # Only track positive peaks
                peak_profit_pct = pnl_pct
                peak_profit_usd = pnl_usd  # Update USD too for logging
                logger.info(f"📈 {symbol}: NEW PEAK! {pnl_pct:.2f}% (${pnl_usd:.2f})")
            
            # Always update cache with the highest known peak
            if peak_profit_pct > 0:
                self.update_peak_profit(symbol, peak_profit_pct)
            
            # Calculate DRAWDOWN from peak using PERCENTAGE
            # Example: Peak was 0.72%, now 0.10% → lost 86% of the gain!
            # FIXED: Raised threshold from 0.05% to 0.20% - tiny peaks are noise!
            if peak_profit_pct > 0.20:  # Only track if we had meaningful profit (>0.20%)
                drawdown_pct_of_peak = ((peak_profit_pct - pnl_pct) / peak_profit_pct * 100) if peak_profit_pct > 0 else 0
                drawdown_usd = peak_profit_usd - pnl_usd  # For logging
                # Log peak tracking status if we have a meaningful peak
                if peak_profit_pct > 0.25:
                    if pnl_pct < 0:
                        # Gone from profit to loss - special messaging
                        logger.info(f"📊 {symbol}: Peak: {peak_profit_pct:.2f}% → Now: {pnl_pct:.2f}% | ⚠️ IN LOSS (was profitable!)")
                    else:
                        logger.info(f"📊 {symbol}: Peak: {peak_profit_pct:.2f}% → Now: {pnl_pct:.2f}% | Lost {drawdown_pct_of_peak:.0f}% of gain")
            else:
                drawdown_pct_of_peak = 0
                drawdown_usd = 0
            
            # Track reversal signals for AI to consider
            profit_reversal_detected = False
            reversal_urgency = 'none'  # none, low, medium, high, critical
            reversal_reason = ''
            
            # Initialize fast reversal score (used for AI+Momentum combo)
            fast_reversal_score = 0
            fast_reversal_signals = []
            very_strong_reversal = False
            
            # Initialize variables that may be set in conditional blocks
            # (required for Cython compilation - no dir() tricks)
            hardcoded_triggered = False
            momentum_hope = 0
            momentum_hope_factors = []
            
            is_long = side == 'LONG'
            
            # === MINIMUM HOLD TIME CHECK ===
            # Don't run aggressive reversal detection on brand new positions
            # They need time to develop - normal market noise can trigger false reversals
            position_open_time = position.get('open_time')
            min_hold_seconds = 30  # 30 seconds minimum before fast reversal triggers
            position_age_seconds = 0
            
            if position_open_time:
                if isinstance(position_open_time, datetime):
                    position_age_seconds = (datetime.now(timezone.utc) - position_open_time).total_seconds()
                else:
                    try:
                        open_dt = datetime.fromisoformat(str(position_open_time).replace('Z', '+00:00'))
                        position_age_seconds = (datetime.now(timezone.utc) - open_dt).total_seconds()
                    except Exception:
                        position_age_seconds = 999  # Assume old position if can't parse
            
            is_new_position = position_age_seconds < min_hold_seconds
            
            # === REVERSAL DETECTION - RUN IF WE HAVE/HAD PROFIT ===
            # CRITICAL FIX: Also run if we HAD meaningful profit (peak_profit_pct > 0.25)
            # This catches the case where we go from profit into loss!
            # Example: Peak was +0.26%, now -0.68% - we MUST detect this reversal!
            # FIXED: Raised threshold from 0.1% to 0.25% - tiny peaks are noise!
            has_any_profit = pnl_usd > 0 or pnl_pct > 0
            had_meaningful_profit = peak_profit_pct > 0.25  # We were profitable before
            
            # For NEW positions (< 30 sec): DON'T skip reversal detection entirely!
            # We MUST still populate the momentum cache so the real-time WebSocket exit
            # has data for soft loss cap decisions.
            # POLICY: Moderate reversals → suppress (let trade develop)
            #         AGGRESSIVE reversals (fast_score >= 50) → STILL trigger exit!
            grace_period_active = False
            if is_new_position:
                logger.debug(f"⏳ {symbol}: New position ({position_age_seconds:.0f}s < 30s) - grace period active (aggressive reversals still detected)")
                grace_period_active = True
                # Reversal detection runs fully for cache population
                # has_any_profit and had_meaningful_profit stay as-is
            
            # Initialize reversal_score here so it's always defined
            reversal_score = 0
            
            # CRITICAL: Run reversal detection for ALL positions:
            # 1. Currently in profit (need to protect it)
            # 2. Had meaningful profit before (profit reversal)
            # 3. In a LOSS for > 60s (detect bounces against us, populate momentum cache)
            # 4. NEW POSITIONS - always run to populate momentum cache!
            # OLD BUG: Only ran for profitable positions → couldn't detect bounces on losing
            # positions → momentum_cache never populated → soft loss cap couldn't check momentum
            position_needs_reversal_check = has_any_profit or had_meaningful_profit
            # NEW: Also check losing positions (to detect adverse momentum + populate cache)
            if not is_new_position and pnl_pct < 0 and position_age_seconds > 60:
                position_needs_reversal_check = True
            # CRITICAL: ALWAYS run for new positions to populate momentum cache!
            # Without this, real-time exit has no momentum data for soft loss cap decisions
            if is_new_position:
                position_needs_reversal_check = True
            
            if len(df) >= 5 and position_needs_reversal_check:
                # Check recent price action
                recent_closes = df['close'].tail(10).values if len(df) >= 10 else df['close'].tail(5).values
                recent_high = max(recent_closes)
                recent_low = min(recent_closes)
                current_close = recent_closes[-1]
                
                # === FAST REVERSAL DETECTION (1-3 candles) ===
                # Get last 3 candles for immediate pattern detection
                last_3_candles = df.tail(3)
                if len(last_3_candles) >= 3:
                    c1_open, c1_high, c1_low, c1_close = last_3_candles.iloc[-3][['open', 'high', 'low', 'close']]
                    c2_open, c2_high, c2_low, c2_close = last_3_candles.iloc[-2][['open', 'high', 'low', 'close']]
                    c3_open, c3_high, c3_low, c3_close = last_3_candles.iloc[-1][['open', 'high', 'low', 'close']]  # Current
                    
                    # Calculate ATR for context
                    recent_atr = context.get('atr', 0) or df['close'].tail(14).std() * 1.5
                    atr_pct = (recent_atr / current_close * 100) if current_close > 0 else 1.0
                    
                    # FAST detection signals (immediate, 1-2 candle patterns)
                    fast_reversal_signals = []
                    fast_reversal_score = 0
                    
                    if is_long:
                        # === LONG POSITION FAST REVERSAL SIGNALS ===
                        # Signal 1: Big red candle (bearish engulfing or strong sell)
                        c3_body_pct = abs(c3_close - c3_open) / c3_open * 100 if c3_open > 0 else 0
                        if c3_close < c3_open and c3_body_pct > atr_pct * 0.5:
                            fast_reversal_score += 20
                            fast_reversal_signals.append(f"🔴 Big red candle ({c3_body_pct:.2f}%)")
                        
                        # Signal 2: Current close below previous low (breakdown)
                        if c3_close < c2_low:
                            fast_reversal_score += 25
                            fast_reversal_signals.append("⬇️ Breakdown below prev low")
                        
                        # Signal 3: Sharp 1-candle drop from high (rejected at top)
                        one_candle_drop = (c3_high - c3_close) / c3_high * 100 if c3_high > 0 else 0
                        if one_candle_drop > atr_pct * 0.7:
                            fast_reversal_score += 20
                            fast_reversal_signals.append(f"📉 Sharp 1-bar drop ({one_candle_drop:.2f}%)")
                        
                        # Signal 4: Two consecutive red candles
                        if c3_close < c3_open and c2_close < c2_open:
                            fast_reversal_score += 15
                            fast_reversal_signals.append("🔻 2 consecutive red bars")
                        
                        # Signal 5: Price dropped from 2-candle high (quick reversal)
                        two_candle_high = max(c2_high, c3_high)
                        drop_from_2c_high = (two_candle_high - c3_close) / two_candle_high * 100 if two_candle_high > 0 else 0
                        if drop_from_2c_high > atr_pct * 1.0:
                            fast_reversal_score += 25
                            fast_reversal_signals.append(f"🚨 Quick {drop_from_2c_high:.2f}% drop from peak")
                        
                        # Signal 6: Upper wick rejection (buying exhaustion)
                        upper_wick = c3_high - max(c3_open, c3_close)
                        body = abs(c3_close - c3_open)
                        if body > 0 and upper_wick > body * 1.5:
                            fast_reversal_score += 15
                            fast_reversal_signals.append("📍 Upper wick rejection")
                            
                    else:  # SHORT
                        # === SHORT POSITION FAST REVERSAL SIGNALS ===
                        # Signal 1: Big green candle (bullish)
                        c3_body_pct = abs(c3_close - c3_open) / c3_open * 100 if c3_open > 0 else 0
                        if c3_close > c3_open and c3_body_pct > atr_pct * 0.5:
                            fast_reversal_score += 20
                            fast_reversal_signals.append(f"🟢 Big green candle ({c3_body_pct:.2f}%)")
                        
                        # Signal 2: Current close above previous high (breakout)
                        if c3_close > c2_high:
                            fast_reversal_score += 25
                            fast_reversal_signals.append("⬆️ Breakout above prev high")
                        
                        # Signal 3: Sharp 1-candle pump from low
                        one_candle_pump = (c3_close - c3_low) / c3_low * 100 if c3_low > 0 else 0
                        if one_candle_pump > atr_pct * 0.7:
                            fast_reversal_score += 20
                            fast_reversal_signals.append(f"📈 Sharp 1-bar pump ({one_candle_pump:.2f}%)")
                        
                        # Signal 4: Two consecutive green candles
                        if c3_close > c3_open and c2_close > c2_open:
                            fast_reversal_score += 15
                            fast_reversal_signals.append("🔺 2 consecutive green bars")
                        
                        # Signal 5: Price jumped from 2-candle low
                        two_candle_low = min(c2_low, c3_low)
                        jump_from_2c_low = (c3_close - two_candle_low) / two_candle_low * 100 if two_candle_low > 0 else 0
                        if jump_from_2c_low > atr_pct * 1.0:
                            fast_reversal_score += 25
                            fast_reversal_signals.append(f"🚨 Quick {jump_from_2c_low:.2f}% pump from low")
                        
                        # Signal 6: Lower wick rejection (selling exhaustion)
                        lower_wick = min(c3_open, c3_close) - c3_low
                        body = abs(c3_close - c3_open)
                        if body > 0 and lower_wick > body * 1.5:
                            fast_reversal_score += 15
                            fast_reversal_signals.append("📍 Lower wick rejection")
                    
                    # Log fast reversal if detected
                    if fast_reversal_score >= 25:
                        logger.info(f"⚡ FAST REVERSAL {symbol}: Score {fast_reversal_score} | {' | '.join(fast_reversal_signals[:3])}")
                
                else:
                    fast_reversal_score = 0
                    fast_reversal_signals = []
                
                # ═══════════════════════════════════════════════════════════════
                # ADVANCED ANALYSIS REVERSAL DETECTION (sub-candle, tick-level)
                # PhD metrics from WebSocket ticks: velocity, acceleration,
                # momentum decay, path efficiency, reversal probability.
                # This catches reversals FASTER than even 1-candle patterns
                # because it sees the intra-candle price trajectory.
                # ═══════════════════════════════════════════════════════════════
                atomic_reversal_boost = 0
                if atomic_candle is not None:
                    try:
                        atomic_analysis = atomic_candle.get_analysis(symbol)
                        if atomic_analysis and atomic_analysis.tick_count >= 30:
                            # Log atomic state for this position
                            logger.info(f"   {atomic_candle.format_analysis_log(symbol)}")
                            
                            # === ATOMIC SIGNAL 1: Velocity against our position ===
                            vel = atomic_analysis.velocity
                            if is_long and vel < -0.003:
                                atomic_reversal_boost += min(15, abs(vel) * 2000)
                                fast_reversal_signals.append(f"🔬 Atomic: price falling (v={vel:+.4f}%/s)")
                            elif not is_long and vel > 0.003:
                                atomic_reversal_boost += min(15, abs(vel) * 2000)
                                fast_reversal_signals.append(f"🔬 Atomic: price rising (v={vel:+.4f}%/s)")
                            
                            # === ATOMIC SIGNAL 2: High reversal probability ===
                            if atomic_analysis.reversal_probability > 60:
                                atomic_reversal_boost += min(15, (atomic_analysis.reversal_probability - 40) * 0.5)
                                fast_reversal_signals.append(f"🔬 Atomic: reversal prob {atomic_analysis.reversal_probability:.0f}%")
                            
                            # === ATOMIC SIGNAL 3: Momentum dying (decay) ===
                            if atomic_analysis.momentum_decay_rate < -40:
                                vel_aligned = (is_long and vel > 0) or (not is_long and vel < 0)
                                if vel_aligned:
                                    # Momentum in our direction but DYING fast
                                    atomic_reversal_boost += 10
                                    fast_reversal_signals.append(f"🔬 Atomic: momentum dying ({atomic_analysis.momentum_decay_rate:+.0f}% decay)")
                            
                            # === ATOMIC SIGNAL 4: Mean-reverting regime while we hold a trend trade ===
                            if atomic_analysis.hurst_micro < 0.35 and atomic_analysis.path_efficiency > 0.5:
                                atomic_reversal_boost += 8
                                fast_reversal_signals.append(f"🔬 Atomic: mean-reverting (H={atomic_analysis.hurst_micro:.2f})")
                            
                            # === ATOMIC SIGNAL 5: Acceleration against us ===
                            acc = atomic_analysis.acceleration
                            if is_long and acc < -0.005:
                                atomic_reversal_boost += min(10, abs(acc) * 1000)
                                fast_reversal_signals.append(f"🔬 Atomic: accelerating down (a={acc:+.4f})")
                            elif not is_long and acc > 0.005:
                                atomic_reversal_boost += min(10, abs(acc) * 1000)
                                fast_reversal_signals.append(f"🔬 Atomic: accelerating up (a={acc:+.4f})")
                            
                            # Add atomic boost to fast reversal score
                            if atomic_reversal_boost > 0:
                                fast_reversal_score += int(atomic_reversal_boost)
                                logger.info(f"🔬 {symbol}: Atomic reversal boost +{atomic_reversal_boost:.0f} → fast_score={fast_reversal_score}")
                            
                            # === ATOMIC HOLD SIGNAL: Strong momentum WITH us ===
                            # If atomic shows clean trend in our direction, reduce reversal urgency
                            vel_with_us = (is_long and vel > 0.005) or (not is_long and vel < -0.005)
                            if vel_with_us and atomic_analysis.path_efficiency > 0.6 and atomic_analysis.direction_consistency > 65:
                                # Price is moving cleanly in our direction at tick level
                                hold_bonus = -10  # Reduce reversal score
                                fast_reversal_score = max(0, fast_reversal_score + hold_bonus)
                                logger.info(f"🔬 {symbol}: Atomic HOLD signal — clean trend in our direction (eff={atomic_analysis.path_efficiency:.2f}, cons={atomic_analysis.direction_consistency:.0f}%)")
                    except Exception as atomic_err:
                        logger.debug(f"🔬 Atomic position check error: {atomic_err}")
                
                # === MOMENTUM ANALYSIS (slower, 5-10 bars) - MORE SENSITIVE ===
                if is_long:
                    momentum_against = roc_5 < -0.1  # Slightly negative = turning
                    strong_momentum_against = roc_5 < -0.3 or roc_10 < -0.2  # Was -0.5/-0.3
                    accelerating_against = roc_5 < roc_10
                    price_drop_pct = ((recent_high - current_close) / recent_high) * 100 if recent_high > 0 else 0
                    very_strong_reversal = roc_5 < -0.7  # Was -1.0
                else:  # SHORT
                    momentum_against = roc_5 > 0.1  # Slightly positive = turning
                    strong_momentum_against = roc_5 > 0.3 or roc_10 > 0.2  # Was 0.5/0.3
                    accelerating_against = roc_5 > roc_10
                    price_drop_pct = ((current_close - recent_low) / recent_low) * 100 if recent_low > 0 else 0
                    very_strong_reversal = roc_5 > 0.7  # Was 1.0
                
                # === INTELLIGENT REVERSAL DETECTION ===
                # Combines FAST signals + DRAWDOWN + MOMENTUM
                
                # Calculate reversal severity score (0-100)
                reversal_score = 0
                reversal_factors = []
                
                # Factor 0: FAST REVERSAL SIGNALS (immediate, 1-3 candles) - 0-40 points
                if fast_reversal_score >= 40:
                    reversal_score += min(40, fast_reversal_score * 0.8)  # Cap at 40
                    reversal_factors.extend(fast_reversal_signals[:2])
                elif fast_reversal_score >= 25:
                    reversal_score += min(25, fast_reversal_score * 0.7)
                    reversal_factors.extend(fast_reversal_signals[:1])
                
                # Factor 1: Drawdown from peak (0-70 points) - MOST IMPORTANT!
                # Using PERCENTAGE-based thresholds (like Bybit display)
                # Lost 20% = 15pts, 30% = 25pts, 40% = 35pts, 50%+ = 50pts
                # FIXED: Raised from 0.1% to 0.25% - tiny peaks don't need protection
                if peak_profit_pct > 0.25 and drawdown_pct_of_peak > 0:  # Any meaningful profit (>0.25%)
                    # Aggressive scaling - protect profits early!
                    if drawdown_pct_of_peak >= 50:
                        drawdown_score = 50  # Lost half = BIG penalty
                    elif drawdown_pct_of_peak >= 40:
                        drawdown_score = 35
                    elif drawdown_pct_of_peak >= 30:
                        drawdown_score = 25
                    elif drawdown_pct_of_peak >= 20:
                        drawdown_score = 15
                    else:
                        drawdown_score = drawdown_pct_of_peak * 0.5
                    
                    reversal_score += drawdown_score
                    logger.debug(f"📊 REVERSAL: drawdown_score={drawdown_score:.1f}, total reversal_score={reversal_score:.1f}")
                    
                    # CRITICAL: If we went from PROFIT into LOSS, this is a MAJOR reversal!
                    # Peak was positive, now we're negative = 100%+ of peak lost
                    if pnl_pct < 0 and peak_profit_pct > 0.1:
                        reversal_score += 35  # SEVERE penalty - we lost ALL profit and went negative
                        reversal_factors.append(f"🚨🚨 PROFIT→LOSS: Was +{peak_profit_pct:.2f}% now {pnl_pct:.2f}%!")
                        logger.warning(f"🚨 {symbol}: PROFIT→LOSS REVERSAL! Peak +{peak_profit_pct:.2f}% → Now {pnl_pct:.2f}%")
                    # ALERT: If we lost 30%+ of peak profit, this is concerning
                    elif drawdown_pct_of_peak >= 30 and peak_profit_pct >= 0.3:
                        reversal_score += 15  # Bonus penalty for significant profit loss
                        reversal_factors.append(f"🚨 PROFIT DECAY: Lost {drawdown_pct_of_peak:.0f}% of {peak_profit_pct:.2f}% peak!")
                    elif drawdown_pct_of_peak >= 20:
                        reversal_factors.append(f"📉 Lost {drawdown_pct_of_peak:.0f}% of peak ({peak_profit_pct:.2f}%→{pnl_pct:.2f}%)")
                
                # Factor 2: Momentum against position (0-25 points)
                if momentum_against:
                    if very_strong_reversal:
                        reversal_score += 25
                        reversal_factors.append(f"🔥 STRONG momentum reversal (ROC5={roc_5:+.2f}%)")
                    elif strong_momentum_against:
                        reversal_score += 18
                        reversal_factors.append(f"⚡ Strong momentum against (ROC5={roc_5:+.2f}%)")
                    else:
                        reversal_score += 10
                        reversal_factors.append(f"📊 Momentum turning (ROC5={roc_5:+.2f}%)")
                
                # Factor 3: Accelerating against (0-15 points)
                if accelerating_against and momentum_against:
                    reversal_score += 15
                    reversal_factors.append("⬇️ Momentum accelerating against")
                
                # Factor 4: Price action reversal (0-10 points)
                if price_drop_pct > 0.5:
                    reversal_score += min(10, price_drop_pct * 2)
                    reversal_factors.append(f"📉 Price dropped {price_drop_pct:.1f}% from peak")
                
                # Factor 5: Position already profitable - protect it! (0-10 points)
                if pnl_pct > 0.5:
                    profit_bonus = min(10, pnl_pct * 3)
                    reversal_score += profit_bonus
                    reversal_factors.append(f"💰 Profit at risk: ${pnl_usd:.2f}/{pnl_pct:.1f}%")
                
                # Factor 6: RSI OVERBOUGHT/OVERSOLD - EXIT EXTREME CONDITIONS (0-25 points)
                # LONG in overbought = likely to reverse down
                # SHORT in oversold = likely to reverse up
                rsi = context.get('rsi', 50)
                if is_long and rsi >= 70:
                    # LONG position with RSI overbought - danger zone!
                    if rsi >= 80:
                        reversal_score += 25
                        reversal_factors.append(f"🔥 EXTREME OVERBOUGHT RSI={rsi:.0f}")
                        logger.warning(f"⚠️ {symbol}: LONG in EXTREME overbought RSI={rsi:.0f}")
                    elif rsi >= 75:
                        reversal_score += 18
                        reversal_factors.append(f"🔴 OVERBOUGHT RSI={rsi:.0f}")
                    else:
                        reversal_score += 12
                        reversal_factors.append(f"⚠️ RSI overbought ({rsi:.0f})")
                elif not is_long and rsi <= 30:
                    # SHORT position with RSI oversold - danger zone!
                    if rsi <= 20:
                        reversal_score += 25
                        reversal_factors.append(f"🔥 EXTREME OVERSOLD RSI={rsi:.0f}")
                        logger.warning(f"⚠️ {symbol}: SHORT in EXTREME oversold RSI={rsi:.0f}")
                    elif rsi <= 25:
                        reversal_score += 18
                        reversal_factors.append(f"🟢 OVERSOLD RSI={rsi:.0f}")
                    else:
                        reversal_score += 12
                        reversal_factors.append(f"⚠️ RSI oversold ({rsi:.0f})")
                
                # === MOMENTUM HOPE - CAN REDUCE REVERSAL SCORE ===
                # If momentum is STILL WITH US despite drawdown, give it a chance
                # This prevents exiting during temporary dips when trend is intact
                momentum_hope = 0
                momentum_hope_factors = []
                
                if is_long:
                    # LONG: Positive momentum = hope
                    if roc_5 > 0.2:  # Strong positive momentum
                        momentum_hope += 15
                        momentum_hope_factors.append(f"📈 ROC5 still positive ({roc_5:+.2f}%)")
                    elif roc_5 > 0:  # Slight positive
                        momentum_hope += 8
                        momentum_hope_factors.append(f"📊 ROC5 flat/positive ({roc_5:+.2f}%)")
                    
                    if roc_10 > 0.3:  # Longer term trend intact
                        momentum_hope += 10
                        momentum_hope_factors.append(f"📈 10-bar trend up ({roc_10:+.2f}%)")
                    
                    # RSI not overbought yet = room to run
                    if 40 <= rsi <= 65:
                        momentum_hope += 8
                        momentum_hope_factors.append(f"✅ RSI healthy ({rsi:.0f})")
                    
                    # Price still above key MAs
                    if current_close > context.get('ema_20', 0) and context.get('ema_20', 0) > 0:
                        momentum_hope += 10
                        momentum_hope_factors.append("📊 Above EMA20")
                        
                else:  # SHORT
                    # SHORT: Negative momentum = hope
                    if roc_5 < -0.2:  # Strong downward momentum
                        momentum_hope += 15
                        momentum_hope_factors.append(f"📉 ROC5 still negative ({roc_5:+.2f}%)")
                    elif roc_5 < 0:  # Slight negative
                        momentum_hope += 8
                        momentum_hope_factors.append(f"📊 ROC5 flat/negative ({roc_5:+.2f}%)")
                    
                    if roc_10 < -0.3:  # Longer term trend intact
                        momentum_hope += 10
                        momentum_hope_factors.append(f"📉 10-bar trend down ({roc_10:+.2f}%)")
                    
                    # RSI not oversold yet = room to fall
                    if 35 <= rsi <= 60:
                        momentum_hope += 8
                        momentum_hope_factors.append(f"✅ RSI healthy ({rsi:.0f})")
                    
                    # Price still below key MAs
                    if current_close < context.get('ema_20', float('inf')):
                        momentum_hope += 10
                        momentum_hope_factors.append("📊 Below EMA20")
                
                # Apply momentum hope reduction to reversal score
                # BUT only if drawdown is not critical (under 40%)
                # FIX: If position is in LOSS (pnl < 0), cap hope much lower — stale 15-min ROC
                # was overriding real-time price action and preventing loss cuts
                #
                # 🔮 ADVANCED ANALYSIS: The most critical override in the system.
                # Candle ROC is 15-min stale. Atomic ticks are real-time.
                # If atomic says move is DYING (exhaustion, deceleration, regime collapse),
                # ZERO OUT hope — don't let stale bars prevent a necessary exit.
                # If atomic says HOLD_STRONG, BOOST hope — ticks confirm the move is alive.
                _atomic_hope_override = None
                if atomic_candle is not None and momentum_hope > 0:
                    try:
                        _hope_pred = atomic_candle._check_exit_signal(symbol, side.lower())
                        if _hope_pred['verdict'] == 'EXIT_NOW' or _hope_pred['urgency'] >= 60:
                            # Atomic says EXIT — zero out all candle-based hope
                            _atomic_hope_override = 0
                            logger.warning(f"🔮💀 {symbol}: MOMENTUM HOPE KILLED by atomic! urgency={_hope_pred['urgency']}, exhaustion={_hope_pred['exhaustion_level']:.0f} — candle ROC is STALE, ticks say move is DEAD")
                            reversal_factors.append(f"🔮 Atomic killed hope: urgency={_hope_pred['urgency']}")
                        elif _hope_pred['verdict'] == 'PREPARE_EXIT' or _hope_pred['urgency'] >= 40:
                            # Atomic says caution — halve the hope
                            _atomic_hope_override = max(0, momentum_hope // 2)
                            logger.info(f"🔮⚠️ {symbol}: Momentum hope HALVED by atomic ({momentum_hope}→{_atomic_hope_override}) — urgency={_hope_pred['urgency']}")
                        elif _hope_pred['verdict'] == 'HOLD_STRONG' and _hope_pred['urgency'] < 10 and pnl_pct > 0:
                            # Atomic confirms move is alive — boost hope by 5 (only in profit)
                            _atomic_hope_override = min(momentum_hope + 5, 30)
                            logger.info(f"🔮💪 {symbol}: Momentum hope BOOSTED by atomic ({momentum_hope}→{_atomic_hope_override}) — ticks confirm move alive")
                    except Exception:
                        pass
                
                if _atomic_hope_override is not None:
                    momentum_hope = _atomic_hope_override
                
                if momentum_hope > 0 and drawdown_pct_of_peak < 40:
                    if pnl_pct < -0.10:
                        # IN LOSS: Momentum hope is DANGEROUS — likely stale bars
                        # Cap at 5 points max, don't let stale ROC prevent loss cut
                        hope_reduction = min(momentum_hope, 5)
                        logger.info(f"⚠️ {symbol}: LOSS POSITION {pnl_pct:.2f}% — momentum hope CAPPED to {hope_reduction}pts (was {momentum_hope})")
                    elif pnl_pct < 0:
                        # Small loss: Reduce hope cap to 10 (was 25)
                        hope_reduction = min(momentum_hope, 10)
                    else:
                        hope_reduction = min(momentum_hope, 25)  # In profit: normal cap
                    original_score = reversal_score
                    reversal_score = max(0, reversal_score - hope_reduction)
                    if hope_reduction > 5:
                        logger.info(f"💪 {symbol}: MOMENTUM HOPE reduces reversal score {original_score:.0f}→{reversal_score:.0f} | {momentum_hope_factors[:2]}")
                        reversal_factors.append(f"💪 Hope -{hope_reduction}pts: {momentum_hope_factors[0]}")
                
                # Log momentum hope if significant
                if momentum_hope >= 20:
                    logger.info(f"💪 {symbol}: Strong momentum hope ({momentum_hope}pts): {' | '.join(momentum_hope_factors[:3])}")
                
                # === CACHE MOMENTUM FOR REAL-TIME QUICK EXIT ===
                # This enables the "fast-as-light" detection on every WebSocket tick
                # Calculate ema_trend here for caching (price above EMA20 = bullish trend)
                ema_20 = context.get('ema_20', 0)
                ema_trend = current_close > ema_20 if ema_20 > 0 else True
                # Compute prev_rsi (~5 bars ago ≈ 1 min on 1m candles) for RSI delta
                _prev_rsi = df['rsi'].iloc[-6] if 'rsi' in df.columns and len(df) >= 6 else rsi
                _vol_ratio = context.get('volume_ratio', 1.0)
                self.update_momentum_cache(
                    symbol=symbol,
                    roc_5=roc_5,
                    roc_10=roc_10,
                    rsi=rsi,
                    ema_trend=ema_trend,
                    momentum_hope=momentum_hope,
                    accelerating_against=accelerating_against,
                    prev_rsi=_prev_rsi,
                    volume_ratio=_vol_ratio,
                    atomic_candle=atomic_candle
                )
                
                # === HARDCODED FALLBACK SAFETY NET ===
                # These trigger REGARDLESS of intelligent score - absolute protection
                # In case intelligent detection fails, these ensure we don't lose big profits
                # 
                # KEY INSIGHT: Exit when we've lost 30-40% of peak, NOT 50-70%!
                # Example: Peak 0.72%, lost 30% = exit at 0.50%, NOT at 0.03%
                # 
                # ABSOLUTE CEILING: NO MATTER WHAT, exit at 50% drawdown!
                # Strong hope only gives +5% extra room, not +10%
                hardcoded_triggered = False
                strong_hope = momentum_hope >= 35  # Only VERY strong momentum (was 30)
                
                # === ABSOLUTE CEILING: 50% DRAWDOWN = MANDATORY EXIT ===
                # No excuses, no momentum hope override - if you lost half, you're out!
                if peak_profit_pct >= 0.40 and drawdown_pct_of_peak >= 50:
                    reversal_score = 90  # VERY high score
                    hardcoded_triggered = True
                    remaining_pct = 100 - drawdown_pct_of_peak
                    reversal_factors.append(f"🚨🚨 ABSOLUTE CEILING: Lost 60%+ of peak! Saving {remaining_pct:.0f}% of {peak_profit_pct:.2f}%!")
                    logger.warning(f"🚨🚨 {symbol}: ABSOLUTE CEILING - Lost {drawdown_pct_of_peak:.0f}% of {peak_profit_pct:.2f}% peak! FORCED EXIT at {pnl_pct:.2f}%!")
                
                # Adjust thresholds based on momentum hope
                # Strong hope = only +5% extra room (was +10%)
                drawdown_threshold_1 = 30 if strong_hope else 25  # Good profit peak (0.5%+)
                drawdown_threshold_2 = 40 if strong_hope else 35  # Medium profit peak (0.3%+)
                drawdown_threshold_3 = 50 if strong_hope else 45  # Small profit peak (0.35%+)
                
                # === 🤖 AI CONSULTATION FOR AMBIGUOUS ZONE (20-35% drawdown) ===
                # This is where AI adds the most value - the zone between "fine" and "exit now"
                # AI can consider context that rules can't: news sentiment, market conditions, etc.
                ai_early_exit_triggered = False
                if peak_profit_pct >= 0.40 and drawdown_pct_of_peak >= 20 and drawdown_pct_of_peak < 35:
                    # AMBIGUOUS ZONE: Profit fading but not critical yet
                    # Let's ask AI if we should exit early or hold
                    try:
                        ai_exit_prompt = f"""URGENT: Should we exit this profitable trade NOW or hold?

SITUATION:
• Symbol: {symbol} {side}
• Peak profit: +{peak_profit_pct:.2f}%
• Current profit: +{pnl_pct:.2f}%  
• Already lost: {drawdown_pct_of_peak:.0f}% of peak profit
• Momentum ROC5: {roc_5:+.2f}%, ROC10: {roc_10:+.2f}%
• RSI: {rsi:.0f}
• Momentum hope score: {momentum_hope}/50

If we don't exit now:
• Next protection level at {drawdown_threshold_1}% drawdown (currently {drawdown_pct_of_peak:.0f}%)
• Could save +{pnl_pct:.2f}% profit now vs potentially less later

QUESTION: Based on momentum, should we EXIT NOW to lock profit, or HOLD expecting recovery?
Answer with ONLY: EXIT or HOLD and one sentence why."""

                        ai_response = self._generate_content(ai_exit_prompt)
                        if ai_response:
                            ai_response_upper = ai_response.upper().strip()
                            if ai_response_upper.startswith('EXIT'):
                                logger.warning(f"🤖 {symbol}: AI EARLY EXIT in ambiguous zone! {ai_response[:80]}")
                                reversal_score = max(reversal_score, 70)
                                hardcoded_triggered = True
                                ai_early_exit_triggered = True
                                reversal_factors.append(f"🤖 AI advised early exit at {drawdown_pct_of_peak:.0f}% drawdown")
                            else:
                                logger.info(f"🤖 {symbol}: AI says HOLD in ambiguous zone - {ai_response[:80]}")
                    except Exception as ai_err:
                        logger.debug(f"AI ambiguous zone check failed: {ai_err}")
                
                # === PERCENTAGE-BASED PROFIT PROTECTION ===
                # Exit at 25-30% drawdown instead of waiting for 50%+
                
                # TIER 1: Good profit (0.5%+) - protect at 25-30% drawdown
                if not hardcoded_triggered and peak_profit_pct >= 0.5 and drawdown_pct_of_peak >= drawdown_threshold_1:
                    reversal_score = max(reversal_score, 75)
                    hardcoded_triggered = True
                    remaining_pct = 100 - drawdown_pct_of_peak
                    hope_note = " (strong momentum +5%)" if strong_hope else ""
                    reversal_factors.append(f"🚨 PROTECT: Keep {remaining_pct:.0f}% of {peak_profit_pct:.2f}% peak!{hope_note}")
                    logger.warning(f"🚨 {symbol}: PROFIT PROTECTION - Lost {drawdown_pct_of_peak:.0f}% of {peak_profit_pct:.2f}% peak, exiting at {pnl_pct:.2f}%!{hope_note}")
                
                # TIER 2: Medium profit (0.3%+) - protect at 35-45% drawdown
                elif peak_profit_pct >= 0.3 and drawdown_pct_of_peak >= drawdown_threshold_2:
                    reversal_score = max(reversal_score, 75)
                    hardcoded_triggered = True
                    hope_note = " (momentum gave extra room)" if strong_hope else ""
                    reversal_factors.append(f"🚨 CRITICAL: Lost {drawdown_pct_of_peak:.0f}% of {peak_profit_pct:.2f}% peak!{hope_note}")
                    logger.warning(f"🚨 {symbol}: CRITICAL - Lost {drawdown_pct_of_peak:.0f}% of {peak_profit_pct:.2f}% peak!{hope_note}")
                
                # TIER 3: Small profit (0.35%+) - protect at 45-55% drawdown
                # FIXED: Raised from 0.2% to 0.35% - don't protect tiny peaks!
                elif peak_profit_pct >= 0.35 and drawdown_pct_of_peak >= drawdown_threshold_3:
                    reversal_score = max(reversal_score, 65)
                    hardcoded_triggered = True
                    reversal_factors.append(f"⚠️ Lost {drawdown_pct_of_peak:.0f}% of {peak_profit_pct:.2f}% peak")
                    logger.warning(f"⚠️ {symbol}: Lost {drawdown_pct_of_peak:.0f}% of {peak_profit_pct:.2f}% peak!")
                
                # ABSOLUTE FLOOR: Never let meaningful profit disappear completely!
                # If we had 0.5%+ peak and now under 0.15% = EXIT NOW
                # FIXED: Raised from 0.4%/0.1% to 0.5%/0.15%
                elif peak_profit_pct >= 0.5 and pnl_pct <= 0.15 and pnl_pct > 0:
                    reversal_score = max(reversal_score, 80)
                    hardcoded_triggered = True
                    reversal_factors.append(f"🚨 FLOOR: {peak_profit_pct:.2f}% peak → {pnl_pct:.2f}% remaining!")
                    logger.warning(f"🚨🚨 {symbol}: PROFIT FLOOR - Had {peak_profit_pct:.2f}%, now only {pnl_pct:.2f}%! EXIT NOW!")
                
                # === 🔒 NEW: TRAILING STOP after +0.3% peak ===
                # If we reached +0.3%+ peak, lock in at least +0.1% profit
                # BUT consider momentum first!
                elif peak_profit_pct >= 0.30 and pnl_pct < 0.10 and pnl_pct >= 0:
                    # Check if momentum is with us - if so, delay the trailing stop
                    is_long = side.upper() == "LONG"
                    momentum_with = (roc_5 > 0.15 or roc_10 > 0.10) if is_long else (roc_5 < -0.15 or roc_10 < -0.10)
                    
                    # FIX: During grace period, small peaks (<0.5%) are noise
                    if grace_period_active and peak_profit_pct < 0.50:
                        logger.info(f"🔒⏳ {symbol}: TRAILING suppressed (grace period) - Peak +{peak_profit_pct:.2f}% too small on {position_age_seconds:.0f}s trade")
                    elif momentum_with and pnl_pct >= 0.02:  # Still have 0.02%+ and momentum with us
                        logger.info(f"🔒⏳ {symbol}: TRAILING delayed - Peak +{peak_profit_pct:.2f}% → {pnl_pct:+.2f}% but ROC5={roc_5:+.2f}% with us")
                    else:
                        reversal_score = max(reversal_score, 85)
                        hardcoded_triggered = True
                        reversal_factors.append(f"🔒 TRAILING: Peak +{peak_profit_pct:.2f}% → {pnl_pct:+.2f}% (below +0.10% lock)")
                        logger.warning(f"🔒 {symbol}: TRAILING STOP - Peak +{peak_profit_pct:.2f}% fell to {pnl_pct:+.2f}%! No momentum support")
                
                # === 🛡️ BREAK-EVEN STOP after +0.2% peak ===
                # If we reached +0.2%+ peak and now negative, check momentum before exiting
                # FIX: During grace period, small peaks are NOISE — don't exit on them
                elif peak_profit_pct >= 0.20 and pnl_pct < 0:
                    is_long = side.upper() == "LONG"
                    momentum_with = (roc_5 > 0.15 or roc_10 > 0.10) if is_long else (roc_5 < -0.15 or roc_10 < -0.10)
                    
                    # Give grace if momentum is with us and loss is tiny
                    MOMENTUM_GRACE = -0.15  # Allow up to -0.15% if momentum is with us (was -0.08%)
                    
                    # FIX: During grace period (<45s), small peaks (<0.4%) are price noise
                    # Don't trigger break-even exit — let the trade develop
                    if grace_period_active and peak_profit_pct < 0.40:
                        logger.info(f"🛡️⏳ {symbol}: BREAK-EVEN suppressed (grace period) - Peak +{peak_profit_pct:.2f}% too small on {position_age_seconds:.0f}s trade")
                    elif momentum_with and pnl_pct > MOMENTUM_GRACE:
                        logger.info(f"🛡️⏳ {symbol}: BREAK-EVEN delayed - Peak +{peak_profit_pct:.2f}% → {pnl_pct:+.2f}% but ROC5={roc_5:+.2f}% with us")
                    else:
                        reversal_score = max(reversal_score, 90)
                        hardcoded_triggered = True
                        exit_note = "momentum against" if not momentum_with else f"loss {pnl_pct:.2f}% too deep"
                        reversal_factors.append(f"🛡️ BREAK-EVEN: Peak +{peak_profit_pct:.2f}% → {pnl_pct:+.2f}% ({exit_note})")
                        logger.warning(f"🛡️ {symbol}: BREAK-EVEN STOP - Peak +{peak_profit_pct:.2f}% → {pnl_pct:+.2f}%! {exit_note}")
                
                # LARGE USD PROFIT: $10+ with strong reversal = CRITICAL
                elif pnl_usd >= 10.0 and momentum_against and very_strong_reversal:
                    if reversal_urgency not in ['critical']:
                        reversal_score = max(reversal_score, 75)
                        hardcoded_triggered = True
                        reversal_factors.append("🛡️ HARDCODED: $10+ with strong reversal")
                        logger.warning(f"🛡️ {symbol}: HARDCODED FALLBACK triggered - $10+ profit at risk!")
                
                # MEDIUM USD PROFIT: $5+ with strong momentum against = HIGH minimum
                elif pnl_usd >= 5.0 and momentum_against and strong_momentum_against:
                    if reversal_urgency not in ['critical', 'high']:
                        reversal_score = max(reversal_score, 55)
                        hardcoded_triggered = True
                        reversal_factors.append("🛡️ HARDCODED: $5+ with momentum against")
                        logger.warning(f"🛡️ {symbol}: HARDCODED FALLBACK triggered - $5+ profit at risk!")
                
                elif pnl_pct >= 2.0 and momentum_against:
                    # 2%+ profit with any momentum against = HIGH minimum
                    if reversal_urgency not in ['critical', 'high']:
                        reversal_score = max(reversal_score, 55)
                        hardcoded_triggered = True
                        reversal_factors.append("🛡️ HARDCODED: 2%+ profit reversal")
                        logger.warning(f"🛡️ {symbol}: HARDCODED FALLBACK triggered - {pnl_pct:.1f}% profit at risk!")
                
                elif pnl_usd >= 3.0 and very_strong_reversal:
                    # $3+ profit with very strong reversal = MEDIUM minimum
                    if reversal_urgency not in ['critical', 'high', 'medium']:
                        reversal_score = max(reversal_score, 40)
                        hardcoded_triggered = True
                        reversal_factors.append("🛡️ HARDCODED: $3+ very strong reversal")
                
                # === FAST REVERSAL HARDCODED TRIGGERS ===
                # These use the FAST 1-3 candle detection for immediate response
                if fast_reversal_score >= 50 and pnl_usd >= 1.5:
                    # Strong fast reversal with meaningful profit = ACT NOW
                    reversal_score = max(reversal_score, 65)
                    hardcoded_triggered = True
                    reversal_factors.append(f"⚡ FAST: Score {fast_reversal_score} + ${pnl_usd:.2f} profit")
                    logger.warning(f"⚡ {symbol}: FAST REVERSAL TRIGGERED - {fast_reversal_signals[:2]}")
                
                elif fast_reversal_score >= 40 and pnl_usd >= 3.0:
                    # Medium fast reversal with good profit = HIGH urgency
                    reversal_score = max(reversal_score, 55)
                    hardcoded_triggered = True
                    reversal_factors.append(f"⚡ FAST: ${pnl_usd:.2f} at risk")
                    logger.warning(f"⚡ {symbol}: FAST REVERSAL - ${pnl_usd:.2f} profit at risk!")
                
                elif fast_reversal_score >= 35 and pnl_pct >= 1.5:
                    # Fast reversal with % profit = HIGH urgency
                    reversal_score = max(reversal_score, 50)
                    hardcoded_triggered = True
                    reversal_factors.append(f"⚡ FAST: {pnl_pct:.1f}% at risk")
                
                # === SMALL PROFIT REVERSAL - BALANCED APPROACH ===
                # Don't cut tiny profits too early, but protect meaningful ones
                # Score 60+ with $0.15+ profit = worth protecting
                elif fast_reversal_score >= 60 and pnl_usd >= 0.15:
                    # Strong reversal on small but meaningful profit = consider exit
                    reversal_score = max(reversal_score, 50)
                    hardcoded_triggered = True
                    reversal_factors.append(f"⚡ Strong reversal on ${pnl_usd:.2f}")
                    logger.warning(f"⚡ {symbol}: STRONG REVERSAL on profit ${pnl_usd:.2f}")
                
                elif fast_reversal_score >= 45 and very_strong_reversal and pnl_usd >= 0.10:
                    # Fast reversal + strong ROC momentum on any profit = protect it
                    reversal_score = max(reversal_score, 45)
                    hardcoded_triggered = True
                    reversal_factors.append(f"⚡ Momentum+Fast on ${pnl_usd:.2f}")
                    logger.warning(f"⚡ {symbol}: MOMENTUM+FAST reversal on ${pnl_usd:.2f}")
                
                # === RSI EXTREME + PROFIT = EXIT NOW ===
                # If we're in profit and RSI is extreme against our position, exit!
                if pnl_usd >= 1.0:
                    if is_long and rsi >= 78:
                        # LONG with RSI very overbought + profit = EXIT
                        reversal_score = max(reversal_score, 60)
                        hardcoded_triggered = True
                        reversal_factors.append(f"🔴 RSI {rsi:.0f} + ${pnl_usd:.2f} profit")
                        logger.warning(f"🔴 {symbol}: LONG overbought RSI={rsi:.0f} + profit ${pnl_usd:.2f} - EXIT!")
                    elif not is_long and rsi <= 22:
                        # SHORT with RSI very oversold + profit = EXIT
                        reversal_score = max(reversal_score, 60)
                        hardcoded_triggered = True
                        reversal_factors.append(f"🟢 RSI {rsi:.0f} + ${pnl_usd:.2f} profit")
                        logger.warning(f"🟢 {symbol}: SHORT oversold RSI={rsi:.0f} + profit ${pnl_usd:.2f} - EXIT!")
                
                # === CLASSIFY URGENCY BASED ON REVERSAL SCORE ===
                # MORE AGGRESSIVE thresholds to exit earlier and protect profits!
                # GRACE PERIOD: Suppress moderate reversals for new positions,
                # but ALWAYS let critical/aggressive ones through!
                # FIX: During grace period, raise critical threshold from 55→80
                # This prevents small-peak noise (score 60-75) from killing new trades
                # True emergencies (break-even=90, trailing=85) still punch through
                critical_threshold = 80 if grace_period_active else 55  # Was always 55
                if reversal_score >= critical_threshold:
                    profit_reversal_detected = True
                    reversal_urgency = 'critical'
                    reversal_reason = f"CRITICAL: Score {reversal_score:.0f}/100 | Peak ${peak_profit_usd:.2f}→${pnl_usd:.2f} | " + " | ".join(reversal_factors[:2])
                    if grace_period_active:
                        logger.warning(f"🚨🚨 {symbol}: {reversal_reason} [GRACE OVERRIDE - too aggressive to ignore!]")
                    else:
                        logger.warning(f"🚨🚨 {symbol}: {reversal_reason}")
                    
                elif reversal_score >= 40:  # Was 50
                    if grace_period_active:
                        # HIGH but not critical — suppress during grace period
                        logger.info(f"🔴⏳ {symbol}: HIGH reversal (score {reversal_score:.0f}) suppressed during grace period")
                        # Still set for logging but don't trigger urgency
                        reversal_urgency = 'low'  # Downgrade during grace
                        reversal_reason = f"HIGH (suppressed): Score {reversal_score:.0f}/100 | grace period"
                    else:
                        profit_reversal_detected = True
                        reversal_urgency = 'high'
                        reversal_reason = f"HIGH: Score {reversal_score:.0f}/100 | Peak ${peak_profit_usd:.2f}→${pnl_usd:.2f} | " + " | ".join(reversal_factors[:2])
                        logger.warning(f"🔴 {symbol}: {reversal_reason}")
                    
                elif reversal_score >= 28:  # Was 35
                    if grace_period_active:
                        # Medium — fully suppress during grace period
                        logger.debug(f"🟠⏳ {symbol}: MEDIUM reversal (score {reversal_score:.0f}) suppressed during grace period")
                        reversal_urgency = 'none'
                        reversal_reason = ''
                    else:
                        profit_reversal_detected = True
                        reversal_urgency = 'medium'
                        reversal_reason = f"MEDIUM: Score {reversal_score:.0f}/100 | Peak ${peak_profit_usd:.2f}→${pnl_usd:.2f}"
                        logger.warning(f"🟠 {symbol}: {reversal_reason}")
                    
                elif reversal_score >= 15 and pnl_pct > 0:  # Was 20
                    profit_reversal_detected = True
                    reversal_urgency = 'low'
                    reversal_reason = f"LOW: Score {reversal_score:.0f}/100 | Watching..."
                    logger.info(f"🟡 {symbol}: {reversal_reason}")
                
                # Log peak tracking for debugging (like Bybit: $0.05 (+0.72%))
                if peak_profit_pct > pnl_pct and peak_profit_pct > 0.3:
                    logger.info(f"📊 {symbol}: Peak ${peak_profit_usd:.2f} ({peak_profit_pct:.2f}%) → Now ${pnl_usd:.2f} ({pnl_pct:.2f}%) | Lost {drawdown_pct_of_peak:.0f}% of gain | RevScore: {reversal_score:.0f}")
                
                # Always log peak status for positions in meaningful profit 
                if pnl_pct > 0.3 or pnl_usd > 0.5:
                    hc_tag = " [HC]" if hardcoded_triggered else ""
                    logger.info(f"📈 {symbol}: PnL ${pnl_usd:.2f} ({pnl_pct:+.2f}%) | Peak {peak_profit_pct:.2f}% | RevScore: {reversal_score:.0f}{hc_tag}")
            
            # Store updated peak values for next cycle (return to caller to save)
            peak_tracking = {
                'peak_profit_usd': peak_profit_usd,
                'peak_profit_pct': peak_profit_pct,
                'drawdown_usd': drawdown_usd,
                'drawdown_pct_of_peak': drawdown_pct_of_peak,
                'reversal_score': reversal_score,
                'hardcoded_triggered': hardcoded_triggered
            }
            
            # === PHASE 2: MATH ANALYSIS ===
            # context already built above for momentum checks
            
            # Get scores for holding vs exiting
            hold_signal = 1 if side == 'LONG' else -1
            exit_signal = -hold_signal
            
            hold_check = self._comprehensive_math_check(hold_signal, df, current_price, atr, context)
            exit_check = self._comprehensive_math_check(exit_signal, df, current_price, atr, context)
            
            hold_score = hold_check.get('score', 50)
            exit_score = exit_check.get('score', 50)
            
            # === PHASE 3: SMART PROFIT PROTECTION ===
            # Be MORE aggressive about protecting profits - our balance is under $500!
            # Key insight: Small profits ($5-10) matter when total balance is <$500
            profit_urgency = 0
            profit_context = []
            
            if pnl_pct > 0:  # In profit
                if pnl_usd >= 25 or pnl_pct >= 1.5:
                    profit_urgency = 35  # VERY strong - protect this profit!
                    profit_context.append(f"💰 GOOD PROFIT (${pnl_usd:.0f}/{pnl_pct:.1f}%) - PROTECT IT!")
                elif pnl_usd >= 10 or pnl_pct >= 0.75:
                    profit_urgency = 25  # Strong protection
                    profit_context.append(f"💰 Solid profit (${pnl_usd:.0f}/{pnl_pct:.1f}%) - protect it!")
                elif pnl_usd >= 5 or pnl_pct >= 0.4:
                    profit_urgency = 18  # Moderate protection - lowered threshold!
                    profit_context.append(f"📈 Profit (${pnl_usd:.0f}/{pnl_pct:.1f}%) - watching closely")
                elif pnl_usd >= 2 or pnl_pct >= 0.2:
                    profit_urgency = 10  # Light protection - capture small gains
                    profit_context.append(f"📊 Small profit (${pnl_usd:.0f}) - considering lock-in")
            elif pnl_pct > -1.0:  # Small loss (can exit if strong signal)
                profit_context.append(f"📉 Small loss ({pnl_pct:.2f}%) - can exit on strong signal")
            
            # Boost exit score based on profit urgency
            adjusted_exit_score = exit_score + profit_urgency
            
            # === PHASE 4: REVERSAL DETECTION ===
            # Look for signs the market is about to reverse against us
            reversal_signals = []
            reversal_strength = 0
            
            # Check momentum reversal
            if 'close' in df.columns and len(df) >= 10:
                recent_closes = df['close'].tail(10)
                momentum_5 = (recent_closes.iloc[-1] / recent_closes.iloc[-5] - 1) * 100
                momentum_10 = (recent_closes.iloc[-1] / recent_closes.iloc[-10] - 1) * 100
                
                if side == 'LONG':
                    if momentum_5 < -0.5:
                        reversal_signals.append(f"Negative 5-bar momentum ({momentum_5:.2f}%)")
                        reversal_strength += 15
                    if momentum_10 < -0.3 and momentum_5 < momentum_10:
                        reversal_signals.append("Accelerating downward momentum")
                        reversal_strength += 10
                else:  # SHORT
                    if momentum_5 > 0.5:
                        reversal_signals.append(f"Positive 5-bar momentum ({momentum_5:.2f}%)")
                        reversal_strength += 15
                    if momentum_10 > 0.3 and momentum_5 > momentum_10:
                        reversal_signals.append("Accelerating upward momentum")
                        reversal_strength += 10
            
            # Check RSI extremes
            if 'rsi' in df.columns and len(df) > 0:
                rsi = df['rsi'].iloc[-1]
                if side == 'LONG' and rsi > 75:
                    reversal_signals.append(f"Overbought RSI ({rsi:.0f})")
                    reversal_strength += 12
                elif side == 'SHORT' and rsi < 25:
                    reversal_signals.append(f"Oversold RSI ({rsi:.0f})")
                    reversal_strength += 12
            
            # Check volume spike (potential reversal)
            volume_ratio = context.get('volume_ratio', 1.0)
            if volume_ratio > 2.0:
                reversal_signals.append(f"Volume spike ({volume_ratio:.1f}x)")
                reversal_strength += 8
            
            # Add reversal strength to exit score if in profit
            if pnl_pct > 0 and reversal_strength > 0:
                adjusted_exit_score += reversal_strength
                profit_context.append(f"⚠️ Reversal risk: {', '.join(reversal_signals)}")
            
            # === PHASE 5: MATH DECISION ===
            math_action = 'hold'
            math_reasoning = []
            
            # === INITIALIZE VARIABLES THAT MAY BE USED LATER ===
            # These need defaults in case early returns or skipped code paths
            momentum_hope = 0
            
            # Calculate accelerating_against BEFORE fast exit uses it
            # For LONG: accelerating if ROC5 is more negative than ROC10 (dropping faster)
            # For SHORT: accelerating if ROC5 is more positive than ROC10 (rising faster)
            if is_long:
                accelerating_against = roc_5 < roc_10 < 0  # Both negative AND accelerating down
            else:
                accelerating_against = roc_5 > roc_10 > 0  # Both positive AND accelerating up
            
            # === 🎣 FISHERMAN LOGIC: Catch the bounce back to breakeven! ===
            # If trade went negative but bounces back to 0.00% or 0.01%, EXIT at breakeven!
            # Like fishing - wait for the fish to come back up, then catch it!
            FISHERMAN_WINDOW = 180  # 3 minutes - patience for the bounce
            _FISH_BASE_LOSS = -0.15  # Base threshold for normal volatility pairs
            # FIX: Scale by ATR — volatile pairs need wider bail threshold
            _atr_pct = (atr / current_price * 100) if current_price > 0 and atr > 0 else 0.5
            _fish_scale = max(1.0, _atr_pct / 0.5)  # 0.5% ATR = baseline, higher = wider
            FISHERMAN_MAX_LOSS = _FISH_BASE_LOSS * min(_fish_scale, 3.0)  # Cap at 3x (-0.45%)
            FISHERMAN_GRACE_PERIOD = 30  # Don't fish in first 30s — spread noise is NOT a real dip
            
            # Track if we were in negative territory
            if not hasattr(self, '_fisherman_tracking'):
                self._fisherman_tracking = {}
            
            if hold_seconds >= FISHERMAN_GRACE_PERIOD and hold_seconds <= FISHERMAN_WINDOW:
                fish_key = f"{symbol}_{side}"
                
                # If currently in small loss, mark as "fishing"
                if pnl_pct < 0 and pnl_pct >= FISHERMAN_MAX_LOSS:
                    if fish_key not in self._fisherman_tracking:
                        self._fisherman_tracking[fish_key] = {
                            'lowest_pnl': pnl_pct,
                            'started_fishing': True
                        }
                        logger.info(f"🎣 {symbol}: FISHING started at {pnl_pct:.2f}% - waiting for bounce to breakeven")
                    else:
                        # Update lowest point
                        if pnl_pct < self._fisherman_tracking[fish_key]['lowest_pnl']:
                            self._fisherman_tracking[fish_key]['lowest_pnl'] = pnl_pct
                
                # If we WERE fishing and now bounced back to breakeven (0.00% to 0.02%)
                if fish_key in self._fisherman_tracking and self._fisherman_tracking[fish_key]['started_fishing']:
                    if pnl_pct >= 0 and pnl_pct <= 0.02:
                        lowest = self._fisherman_tracking[fish_key]['lowest_pnl']
                        logger.warning(f"🎣 {symbol}: FISH CAUGHT! Bounced from {lowest:.2f}% → {pnl_pct:.2f}% - EXIT AT BREAKEVEN!")
                        # Clear tracking
                        del self._fisherman_tracking[fish_key]
                        math_reasoning.append(f"🎣 FISHERMAN: Caught bounce from {lowest:.2f}% → breakeven")
                        return {
                            'action': 'close',
                            'confidence': 0.92,
                            'math_score': 88,
                            'ai_validated': True,
                            'reasoning': '; '.join(math_reasoning),
                            'sl_adjustment': None,
                            'peak_profit_pct': peak_profit_pct,
                            'peak_profit_usd': peak_profit_usd
                        }
                
                # If loss goes too deep, stop fishing
                if fish_key in self._fisherman_tracking and pnl_pct < FISHERMAN_MAX_LOSS:
                    logger.info(f"🎣 {symbol}: Fish got away! Loss {pnl_pct:.2f}% too deep (< {FISHERMAN_MAX_LOSS}%)")
                    del self._fisherman_tracking[fish_key]
            
            # === FAST EXIT: Exit bad entries quickly IF no recovery chance ===
            # EXTENDED: Now covers first 2 minutes and slightly deeper losses
            # This catches bad entries where direction was wrong AND there's NO recovery chance
            # Must be VERY strict - only exit if ALL indicators confirm no hope
            FAST_EXIT_WINDOW = 120  # First 2 minutes (was 45 seconds - too short)
            FAST_EXIT_MAX_LOSS = -0.10  # Between 0.0% and -0.10%
            FAST_EXIT_MIN_LOSS = 0.0  # Don't exit if we're in profit!
            
            # Calculate "no recovery chance" score - must be VERY high to trigger fast exit
            no_recovery_score = 0
            no_recovery_reasons = []
            
            # Factor 1: Strong momentum against (ROC5 and ROC10 both against us)
            if is_long:
                if roc_5 < -0.4:  # Strong bearish
                    no_recovery_score += 30
                    no_recovery_reasons.append(f"ROC5={roc_5:+.2f}%")
                if roc_10 < -0.3:  # Sustained bearish
                    no_recovery_score += 20
                    no_recovery_reasons.append(f"ROC10={roc_10:+.2f}%")
                if accelerating_against:  # Getting worse
                    no_recovery_score += 15
                    no_recovery_reasons.append("Accelerating down")
            else:  # SHORT
                if roc_5 > 0.4:  # Strong bullish
                    no_recovery_score += 30
                    no_recovery_reasons.append(f"ROC5={roc_5:+.2f}%")
                if roc_10 > 0.3:  # Sustained bullish
                    no_recovery_score += 20
                    no_recovery_reasons.append(f"ROC10={roc_10:+.2f}%")
                if accelerating_against:  # Getting worse
                    no_recovery_score += 15
                    no_recovery_reasons.append("Accelerating up")
            
            # Factor 2: No momentum hope at all
            momentum_hope_score = momentum_hope
            if momentum_hope_score == 0:
                no_recovery_score += 20
                no_recovery_reasons.append("No momentum hope")
            elif momentum_hope_score < 10:
                no_recovery_score += 10
                no_recovery_reasons.append(f"Low hope ({momentum_hope_score})")
            
            # Factor 3: Fast reversal already detected
            if fast_reversal_score >= 40:
                no_recovery_score += 15
                no_recovery_reasons.append(f"Fast rev={fast_reversal_score}")
            
            # ONLY trigger fast exit if:
            # 1. Within first 45 seconds
            # 2. At breakeven to -0.1% loss (not in profit, not too deep loss)
            # 3. No recovery score >= 70 (very high confidence no recovery)
            no_recovery_chance = no_recovery_score >= 70
            
            in_fast_exit_zone = pnl_pct <= FAST_EXIT_MIN_LOSS and pnl_pct >= FAST_EXIT_MAX_LOSS
            
            if hold_seconds <= FAST_EXIT_WINDOW and in_fast_exit_zone and no_recovery_chance:
                # Price went against us immediately AND multiple factors confirm no recovery
                logger.warning(f"🚨 {symbol}: FAST EXIT! PnL={pnl_pct:.2f}% in {hold_seconds:.0f}s | No Recovery Score={no_recovery_score} | {', '.join(no_recovery_reasons[:3])}")
                math_action = 'close'
                math_reasoning.append(f"🚨 FAST EXIT: No recovery ({no_recovery_score}pts) | {', '.join(no_recovery_reasons[:2])}")
                return {
                    'action': 'close',
                    'confidence': 0.95,
                    'math_score': 90,
                    'ai_validated': True,
                    'reasoning': '; '.join(math_reasoning),
                    'sl_adjustment': None,
                    'peak_profit_pct': peak_profit_pct,
                    'peak_profit_usd': peak_profit_usd
                }
            elif hold_seconds <= FAST_EXIT_WINDOW and in_fast_exit_zone:
                # Log why we're NOT fast exiting
                logger.info(f"⏳ {symbol}: In fast exit zone ({pnl_pct:.2f}%) but recovery possible (score={no_recovery_score}/70)")
            
            # === MOMENTUM AGAINST - EXIT AT BREAKEVEN/TINY PROFIT ===
            # Exit BEFORE going negative! At +0.05%, +0.02%, +0.01%, or 0.00%
            QUICK_EXIT_WINDOW = 120  # Within first 2 minutes
            QUICK_EXIT_GRACE_2 = 30  # Don't quick-exit in first 30 seconds
            if hold_seconds > QUICK_EXIT_GRACE_2 and hold_seconds <= QUICK_EXIT_WINDOW and pnl_pct >= 0 and pnl_pct <= 0.05:
                # We're at breakeven to tiny profit - check if indicators are against us
                against_count = 0
                momentum_against_reasons = []
                _cache = self._momentum_cache.get(symbol)
                current_rsi = _cache.get('rsi', 50) if _cache else 50
                ema_trend = _cache.get('ema_trend', None) if _cache else None
                
                if is_long:
                    if roc_5 < -0.2:  # Momentum against
                        against_count += 1
                        momentum_against_reasons.append(f"ROC5={roc_5:+.2f}%")
                    if roc_10 < -0.1:  # Trend against
                        against_count += 1
                        momentum_against_reasons.append(f"ROC10={roc_10:+.2f}%")
                    if accelerating_against:
                        against_count += 1
                        momentum_against_reasons.append("Accelerating↓")
                    if current_rsi < 45:  # RSI weak
                        against_count += 1
                        momentum_against_reasons.append(f"RSI={current_rsi:.0f}")
                    if ema_trend is not None and not ema_trend:
                        against_count += 1
                        momentum_against_reasons.append("EMA↓")
                else:  # SHORT
                    if roc_5 > 0.2:  # Momentum against
                        against_count += 1
                        momentum_against_reasons.append(f"ROC5={roc_5:+.2f}%")
                    if roc_10 > 0.1:  # Trend against
                        against_count += 1
                        momentum_against_reasons.append(f"ROC10={roc_10:+.2f}%")
                    if accelerating_against:
                        against_count += 1
                        momentum_against_reasons.append("Accelerating↑")
                    if current_rsi > 55:  # RSI strong
                        against_count += 1
                        momentum_against_reasons.append(f"RSI={current_rsi:.0f}")
                    if ema_trend is not None and ema_trend:
                        against_count += 1
                        momentum_against_reasons.append("EMA↑")
                
                # Also check momentum hope
                if momentum_hope_score == 0:
                    against_count += 1
                    momentum_against_reasons.append("NoHope")
                
                # Exit if 3+ indicators against
                everything_against = against_count >= 3
                
                if everything_against:
                    logger.warning(f"🚨 {symbol}: QUICK EXIT at +{pnl_pct:.2f}%! {against_count}/6 indicators against | {', '.join(momentum_against_reasons[:4])}")
                    math_action = 'close'
                    math_reasoning.append(f"🚨 QUICK EXIT: {against_count}/6 against ({', '.join(momentum_against_reasons[:3])})")
                    return {
                        'action': 'close',
                        'confidence': 0.88,
                        'math_score': 82,
                        'ai_validated': True,
                        'reasoning': '; '.join(math_reasoning),
                        'sl_adjustment': None,
                        'peak_profit_pct': peak_profit_pct,
                        'peak_profit_usd': peak_profit_usd
                    }
            
            # === PROFIT-TO-LOSS QUICK EXIT ===
            # If trade was briefly profitable but quickly reversed to loss, exit immediately
            # This catches the SKR scenario: peak +0.12%, then dropped to -0.05%, then -1.40%
            PROFIT_TO_LOSS_WINDOW = 90  # Within first 90 seconds
            if hold_seconds <= PROFIT_TO_LOSS_WINDOW and pnl_pct < 0 and peak_profit_pct > 0:
                # Trade was profitable but now in loss - check if we should cut
                # Trigger if: small peak (< 0.5%) AND momentum is against us
                small_peak = peak_profit_pct < 0.5 or peak_profit_usd < 0.50
                momentum_against = (is_long and roc_5 < -0.2) or (not is_long and roc_5 > 0.2)
                
                if small_peak and momentum_against and pnl_pct < -0.2:  # Reduced from -0.3 to exit earlier
                    # 🔮 ATOMIC: Check if ticks show recovery forming
                    _math_p2l_vetoed = False
                    if atomic_candle is not None:
                        try:
                            _mp2l = atomic_candle._check_exit_signal(symbol, side.lower())
                            if _mp2l['verdict'] == 'HOLD_STRONG' and _mp2l['urgency'] < 12:
                                _math_p2l_vetoed = True
                                logger.info(f"🔮🚨 {symbol}: Math PROFIT→LOSS VETOED by atomic — recovery in ticks! urgency={_mp2l['urgency']}")
                        except Exception:
                            pass
                    
                    if not _math_p2l_vetoed:
                        logger.warning(f"🚨 {symbol}: PROFIT→LOSS EXIT! Peak was +{peak_profit_pct:.2f}%, now {pnl_pct:.2f}% | Momentum against")
                        math_action = 'close'
                        math_reasoning.append(f"🚨 PROFIT→LOSS: Peak +{peak_profit_pct:.2f}% → Now {pnl_pct:.2f}%, momentum against")
                        return {
                            'action': 'close',
                            'confidence': 0.90,
                            'math_score': 85,
                            'ai_validated': True,
                            'reasoning': '; '.join(math_reasoning),
                            'sl_adjustment': None,
                            'peak_profit_pct': peak_profit_pct,
                            'peak_profit_usd': peak_profit_usd
                        }
            
            # === PROFIT-TO-LOSS EXIT FOR OLDER POSITIONS ===
            # For positions held longer, still catch PROFIT→LOSS reversal with strong momentum against
            # XPL case: Peak +0.10%, now -0.37%, 28 candles old, momentum against → should exit!
            if pnl_pct < -0.15 and peak_profit_pct > 0.05:
                # Position HAD profit but now in loss
                strong_momentum_against_pos = (is_long and roc_5 < -0.3) or (not is_long and roc_5 > 0.3)
                sustained_momentum_against = (is_long and roc_5 < -0.15 and roc_10 < -0.1) or (not is_long and roc_5 > 0.15 and roc_10 > 0.1)
                
                if strong_momentum_against_pos or sustained_momentum_against:
                    # 🔮 ATOMIC: Check ticks before cutting older position
                    _math_p2l_old_vetoed = False
                    if atomic_candle is not None:
                        try:
                            _mp2lo = atomic_candle._check_exit_signal(symbol, side.lower())
                            if _mp2lo['verdict'] == 'HOLD_STRONG' and _mp2lo['urgency'] < 12:
                                _math_p2l_old_vetoed = True
                                logger.info(f"🔮🚨 {symbol}: Math PROFIT→LOSS (older) VETOED by atomic — recovery in ticks! urgency={_mp2lo['urgency']}")
                        except Exception:
                            pass
                    
                    if not _math_p2l_old_vetoed:
                        logger.warning(f"🚨 {symbol}: PROFIT→LOSS EXIT (older)! Peak +{peak_profit_pct:.2f}% → Now {pnl_pct:.2f}% | ROC5={roc_5:+.2f}%")
                        return {
                            'action': 'close',
                            'confidence': 0.85,
                            'math_score': 80,
                            'ai_validated': True,
                            'reasoning': f"🚨 PROFIT→LOSS (older position): Peak +{peak_profit_pct:.2f}% → {pnl_pct:.2f}%, momentum against (ROC5={roc_5:+.2f}%)",
                            'sl_adjustment': None,
                            'peak_profit_pct': peak_profit_pct,
                            'peak_profit_usd': peak_profit_usd
                        }
            
            # === NO GRACE PERIOD - Let math and AI decide ===
            # Grace period was causing losses by holding bad positions too long
            # Now we rely on math exit score vs hold score + fast exit for bad entries
            
            # Decision thresholds - based on P&L state
            if pnl_pct > 0:
                exit_threshold = 12  # Easier to exit when in profit (after grace)
            else:
                exit_threshold = 25  # Harder to cut losses (was 15, increased to give more time)
            
            if adjusted_exit_score > hold_score + exit_threshold:
                math_action = 'close'
                math_reasoning.append(f"Exit score ({adjusted_exit_score:.0f}) > Hold ({hold_score:.0f}) by {exit_threshold}+")
            elif exit_score > hold_score and pnl_pct > 0.5:
                math_action = 'tighten_sl'
                math_reasoning.append(f"Exit strengthening - tighten SL to lock profit")
            else:
                math_reasoning.append(f"Hold score ({hold_score:.0f}) favored over exit ({adjusted_exit_score:.0f})")
            
            # Add context to reasoning
            math_reasoning.extend(profit_context)
            if reversal_signals and pnl_pct > 0:
                math_reasoning.append(f"Reversal indicators: {len(reversal_signals)}")
            
            # === PHASE 5.5: INTELLIGENT PROFIT REVERSAL MATH OVERRIDE ===
            # Uses the dynamic reversal_score calculated above (0-100)
            # No hardcoded dollar thresholds - decision based on:
            # 1. Reversal score (drawdown + momentum + acceleration)
            # 2. Math analysis (exit_score vs hold_score)
            # 3. Drawdown from peak profit
            math_override_exit = False
            
            # Get reversal score from peak_tracking (calculated above)
            reversal_score = peak_tracking.get('reversal_score', 0)
            drawdown_pct = peak_tracking.get('drawdown_pct_of_peak', 0)
            
            # IMPORTANT: Only trigger reversal protection if peak profit was meaningful
            # Don't trigger on tiny fluctuations
            # FIX: Raised from $1.00/0.50% — $1.01 peaks on 10s trades are noise
            min_peak_for_reversal = 1.50  # At least $1.50 peak profit to trigger reversal protection
            min_peak_pct_for_reversal = 0.35  # At least 0.35% profit to trigger reversal protection
            
            peak_was_meaningful = peak_profit_usd >= min_peak_for_reversal or peak_profit_pct >= min_peak_pct_for_reversal
            
            if profit_reversal_detected and peak_was_meaningful:
                # CRITICAL: Reversal score >= 70 OR lost 50%+ of peak profit → EXIT NOW
                if reversal_urgency == 'critical' or (drawdown_pct >= 50 and peak_profit_usd > 1.0):
                    math_override_exit = True
                    math_action = 'close'
                    math_reasoning.append(f"🚨 MATH+AI OVERRIDE: CRITICAL (Score={reversal_score:.0f}) Peak ${peak_profit_usd:.2f}→${pnl_usd:.2f} ({drawdown_pct:.0f}% lost)")
                    logger.warning(f"🚨🚨 INTELLIGENT EXIT {symbol}: Score {reversal_score:.0f}/100, lost {drawdown_pct:.0f}% of peak profit")
                    
                # HIGH: Score >= 50 AND (math confirms OR lost 30%+ of peak)
                elif reversal_urgency == 'high' and (exit_score >= hold_score or drawdown_pct >= 30):
                    math_override_exit = True
                    math_action = 'close'
                    math_reasoning.append(f"⚡ MATH+AI OVERRIDE: HIGH (Score={reversal_score:.0f}) + Math confirms exit")
                    logger.warning(f"⚡⚡ INTELLIGENT EXIT {symbol}: Score {reversal_score:.0f}/100, math exit={exit_score:.0f} vs hold={hold_score:.0f}")
                    
                # MEDIUM: Score >= 35 AND math strongly favors exit AND losing profit
                elif reversal_urgency == 'medium' and adjusted_exit_score > hold_score + 10 and drawdown_pct >= 20:
                    math_override_exit = True
                    math_action = 'close'
                    math_reasoning.append(f"📊 MATH+AI OVERRIDE: MEDIUM (Score={reversal_score:.0f}) + Strong exit signal")
                    logger.warning(f"📊 INTELLIGENT EXIT {symbol}: Score {reversal_score:.0f}/100, drawdown {drawdown_pct:.0f}%")
                    
                # LOW: Just flag for AI consideration, don't override
                elif reversal_urgency == 'low':
                    math_reasoning.append(f"⚠️ Reversal detected (Score={reversal_score:.0f}) - AI will decide")
            elif profit_reversal_detected and not peak_was_meaningful:
                # Peak profit was too small to trigger reversal protection
                logger.info(f"💡 {symbol}: Reversal detected but peak was small (${peak_profit_usd:.2f}/{peak_profit_pct:.2f}%) - letting position develop")
                math_reasoning.append(f"💡 Small peak (${peak_profit_usd:.2f}) - holding for bigger move")
            
            # === PHASE 6: AI VALIDATION FOR ALL DECISIONS ===
            # AI now has POWERFUL control - it can override or refine decisions
            # EXCEPT when math_override_exit is True (profit protection)
            ai_validated = False
            ai_agrees_to_exit = False
            ai_confidence = 0.5
            ai_suggested_action = 'hold'
            
            # Log whether AI will be consulted
            if math_override_exit:
                logger.info(f"🛡️ {symbol}: Math Override active - skipping AI consultation for profit protection")
            
            # Always consult AI for position decisions (not just when math says close)
            # BUT skip AI consultation if math override is active (profit protection takes priority)
            # FIX: Also skip AI consultation for positions < 30s old — entry pipeline JUST approved this,
            # don't let AI panic-exit on the same S/R data that entry already evaluated
            if is_new_position and self.use_ai and not math_override_exit:
                logger.info(f"🛡️ {symbol}: Skipping AI exit consultation — position only {hold_seconds:.0f}s old (grace period)")
            
            if self.use_ai and not math_override_exit and not is_new_position:
                logger.info(f"🤖 {symbol}: Consulting AI for exit decision (reversal_score={reversal_score:.0f}, hope={momentum_hope})")
                # Get AI opinion with deep math analysis
                # Include momentum hope so AI knows if there's reason to hold
                ai_result = self._ai_smart_exit_decision(
                    symbol=symbol,
                    side=side,
                    entry_price=entry_price,
                    current_price=current_price,
                    pnl_pct=pnl_pct,
                    pnl_usd=pnl_usd,
                    hold_score=hold_score,
                    exit_score=adjusted_exit_score,
                    reversal_signals=reversal_signals,
                    context=context,
                    df=df,
                    profit_reversal_detected=profit_reversal_detected,
                    reversal_urgency=reversal_urgency,
                    reversal_reason=reversal_reason,
                    momentum_hope=momentum_hope,
                    momentum_hope_factors=momentum_hope_factors,
                    reversal_score=reversal_score
                )
                
                if ai_result:
                    ai_validated = True
                    ai_agrees_to_exit = ai_result.get('should_exit', False)
                    ai_confidence = ai_result.get('confidence', 0.5)
                    ai_reason = ai_result.get('reasoning', '')
                    ai_suggested_action = ai_result.get('suggested_action', 'hold')
                    
                    # === SMART GRACE PERIOD - ALLOWS INTELLIGENT REVERSALS ===
                    # During grace period, AI can still force exit IF:
                    # 1. Loss exceeds -1.0% (meaningful loss)
                    # 2. AI has 85%+ confidence (strong signal)
                    # 3. Reversal score >= 50 (strong reversal detected)
                    # 4. Fast reversal score >= 40 (quick momentum shift)
                    # 5. We're in profit AND momentum is strongly against us
                    # 6. PROFIT→LOSS: We had meaningful profit but now in loss
                    # NOTE: Grace period REMOVED - was causing losses by blocking exits
                    
                    # AI has POWERFUL control - respect its suggestion
                    # AI needs 75% confidence on both profits and losses to exit
                    # FIX: Old code needed 85% on losses → blocked AI EXIT at -0.02% (DOGE lost -$2.09)
                    ai_override_threshold = 0.75
                    
                    # CRITICAL: Detect PROFIT→LOSS reversal - should ALWAYS be allowed to exit!
                    is_profit_to_loss_reversal = pnl_pct < 0 and peak_profit_pct > 0.05
                    
                    # CRITICAL: Detect AI seeing reversal signals (bounce, support, etc.)
                    # If AI mentions these danger words, it detected a fundamental problem
                    _ai_reason_lower = ai_reason.lower() if ai_reason else ''
                    _ai_sees_reversal = any(w in _ai_reason_lower for w in [
                        'support', 'bounce', 'reversal', 'positive momentum',
                        'near support', 'risk of bounce', 'oversold',
                        'resistance', 'overbought', 'rejection', 'bearish',
                        'downtrend', 'selling', 'divergence', 'weakening'
                    ])
                    
                    if ai_suggested_action == 'exit' and ai_confidence >= ai_override_threshold:
                        # FIX: NEVER override AI when it's protecting us.
                        # AI says EXIT with ≥75% confidence = RESPECT IT ALWAYS.
                        # Previous versions blocked AI EXIT at small losses (-0.02% to -0.20%)
                        # which led to ETH/USDT -$54 loss (AI was right 6/6 times).
                        # Only exception: breakeven noise filter (0.00-0.08% in testing mode)
                        _ai_exit_testing = get_threshold_mode() == 'testing'
                        _ai_exit_at_breakeven = _ai_exit_testing and 0 <= pnl_pct < 0.08
                        
                        if _ai_exit_at_breakeven and not is_profit_to_loss_reversal:
                            math_reasoning.append(f"🤖 AI wanted exit at breakeven {pnl_pct:.2f}% — TESTING MODE: let it develop")
                            logger.info(f"⏳ {symbol}: AI EXIT blocked at breakeven {pnl_pct:.2f}% — testing mode lets trades develop")
                        else:
                            # AI is PROTECTING us — respect it unconditionally
                            math_action = 'close'
                            ai_agrees_to_exit = True  # CRITICAL: Mark AI agrees to exit!
                            exit_reason = f"🤖 AI PROTECT EXIT ({ai_confidence:.0%})"
                            if is_profit_to_loss_reversal:
                                exit_reason = f"🤖 AI PROFIT→LOSS EXIT: Peak +{peak_profit_pct:.2f}% → Now {pnl_pct:.2f}%"
                            elif _ai_sees_reversal and pnl_pct < 0:
                                exit_reason = f"🤖 AI REVERSAL EXIT: {pnl_pct:.2f}% + structural danger detected"
                            math_reasoning.append(f"{exit_reason}: {ai_reason}")
                            logger.warning(f"🚨 {exit_reason} for {symbol}: {ai_reason}")
                    # AI + FAST REVERSAL COMBO: Even small profit, if AI says exit + fast signals strong
                    elif ai_suggested_action == 'exit' and ai_confidence >= 0.60 and fast_reversal_score >= 35 and pnl_usd > 0:
                        math_action = 'close'
                        ai_agrees_to_exit = True
                        math_reasoning.append(f"🤖⚡ AI+FAST COMBO: Exit ${pnl_usd:.2f} (AI={ai_confidence:.0%}, Fast={fast_reversal_score})")
                        logger.warning(f"🤖⚡ {symbol}: AI+FAST REVERSAL CLOSE - AI {ai_confidence:.0%} + Fast score {fast_reversal_score}")
                    elif ai_suggested_action == 'wait_for_bounce' and pnl_pct < 0 and ai_confidence >= 0.65:
                        # DISABLED: Don't wait for bounce - cut losses fast!
                        # Old behavior held losers hoping for mean reversion
                        # Now we exit if math says exit, regardless of AI bounce hope
                        math_reasoning.append(f"🤖 AI wanted bounce but we CUT LOSSES FAST now")
                        # Don't override math_action - let it proceed
                    elif ai_suggested_action == 'tighten_sl' and pnl_pct > 0:
                        math_action = 'tighten_sl'
                        ai_agrees_to_exit = False
                        math_reasoning.append(f"🤖 AI: TIGHTEN SL ({ai_confidence:.0%}): {ai_reason}")
                    elif ai_suggested_action == 'hold':
                        # FIX: AI should NOT override math exit decisions
                        # Math exits are data-driven (reversal score, momentum, S/R analysis)
                        # AI holding against math led to deeper losses (ETH/USDT -$54)
                        # Rule: AI cannot veto math. If math says close, we close.
                        ai_agrees_to_exit = False
                        if math_action == 'close':
                            # Math says close — AI HOLD does NOT override it
                            math_reasoning.append(f"🤖 AI said HOLD but MATH EXIT stands — AI cannot override math ({ai_confidence:.0%}): {ai_reason}")
                            logger.warning(f"⚡ {symbol}: AI wanted to hold but MATH EXIT stands — AI does not override math!")
                            ai_agrees_to_exit = True  # Math exit proceeds
                        elif math_override_exit:
                            math_reasoning.append(f"🤖 AI said HOLD but MATH OVERRIDE active - exiting anyway!")
                            logger.warning(f"⚡ {symbol}: AI wanted to hold but MATH OVERRIDE takes priority for profit protection!")
                            ai_agrees_to_exit = True  # Force this for confidence calculation
                        else:
                            math_reasoning.append(f"🤖 AI confirms HOLD ({ai_confidence:.0%}): {ai_reason}")
            
            # For tighten_sl, AI validation is optional
            if math_action == 'tighten_sl' and not ai_validated:
                ai_validated = True  # Auto-approve SL tightening
            
            # === PHASE 7: FINAL DECISION ===
            # Calculate confidence based on agreement
            score_diff = abs(hold_score - adjusted_exit_score)
            base_confidence = min(0.5 + (score_diff / 100), 0.95)
            
            if math_action == 'close':
                # MATH OVERRIDE gets high confidence automatically
                if math_override_exit:
                    final_confidence = 0.90  # High confidence for profit protection
                    ai_validated = True  # Mark as validated
                    ai_agrees_to_exit = True  # Mark agreement
                    logger.warning(f"💰 {symbol}: MATH OVERRIDE EXIT - Protecting ${pnl_usd:.2f} profit with 90% confidence")
                # For normal exits, require both math and AI agreement
                elif ai_validated and ai_agrees_to_exit:
                    final_confidence = min(base_confidence, ai_confidence)
                else:
                    final_confidence = base_confidence * 0.7  # Reduce confidence without AI
            else:
                final_confidence = base_confidence
            
            # Calculate SL adjustment if needed
            sl_adjustment = None
            if math_action == 'tighten_sl':
                current_sl = position.get('stop_loss', entry_price)
                if side == 'LONG':
                    # Move SL up to lock profit (at least breakeven)
                    breakeven_sl = entry_price * 1.001  # Tiny buffer above entry
                    profit_lock_sl = current_price - atr * 1.2  # Tighter than usual
                    new_sl = max(current_sl, breakeven_sl, profit_lock_sl)
                    if new_sl > current_sl:
                        sl_adjustment = new_sl
                        math_reasoning.append(f"SL tightened: ${current_sl:.4f} → ${new_sl:.4f}")
                else:
                    breakeven_sl = entry_price * 0.999
                    profit_lock_sl = current_price + atr * 1.2
                    new_sl = min(current_sl, breakeven_sl, profit_lock_sl)
                    if new_sl < current_sl:
                        sl_adjustment = new_sl
                        math_reasoning.append(f"SL tightened: ${current_sl:.4f} → ${new_sl:.4f}")
            
            # === FINAL DECISION READY ===
            # Grace period safety removed - was causing losses by blocking exits
            
            result = {
                'action': math_action,
                'confidence': final_confidence,
                'math_score': hold_score,
                'exit_score': adjusted_exit_score,
                'ai_validated': ai_validated,
                'math_override': math_override_exit,  # Track if this was a math override
                'reversal_urgency': reversal_urgency if profit_reversal_detected else None,
                'reversal_score': peak_tracking.get('reversal_score', 0),  # NEW: Dynamic score
                'reasoning': '; '.join(math_reasoning),
                'sl_adjustment': sl_adjustment,
                'pnl_pct': pnl_pct,
                'pnl_usd': pnl_usd,
                'reversal_strength': reversal_strength,
                # NEW: Peak profit tracking for intelligent reversal detection
                'peak_tracking': peak_tracking
            }
            
            override_tag = " ⚡MATH_OVERRIDE" if math_override_exit else ""
            action_emoji = {'hold': '⏸️', 'close': '🚪', 'tighten_sl': '🔒'}.get(math_action, '❓')
            logger.info(f"📊 UNIFIED POSITION: {symbol} {side} | {action_emoji} {math_action.upper()}{override_tag} | Conf: {final_confidence:.0%}")
            logger.info(f"   Hold: {hold_score:.0f} | Exit: {adjusted_exit_score:.0f} | PnL: {pnl_pct:+.2f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Unified position decision error: {e}")
            return {
                'action': 'hold',
                'confidence': 0.5,
                'math_score': 50,
                'ai_validated': False,
                'reasoning': f'Error: {e}',
                'sl_adjustment': None
            }
    
    def _detect_resistance_support_levels(
        self,
        df: pd.DataFrame,
        current_price: float,
        lookback_bars: int = 96  # ~24 hours on 15m timeframe
    ) -> Dict[str, Any]:
        """
        Detect RESISTANCE and SUPPORT ZONES using 4 key price levels:
        
        RESISTANCE ZONE (Top):
        - resistance_upper = 24h HIGH
        - resistance_lower = 24h HIGH - (ATR * zone_multiplier)
        - If price is BETWEEN these 2 points → DON'T GO LONG
        
        SUPPORT ZONE (Bottom):
        - support_upper = 24h LOW + (ATR * zone_multiplier)
        - support_lower = 24h LOW
        - If price is BETWEEN these 2 points → DON'T GO SHORT
        
        Zone size is calculated dynamically based on market volatility (ATR).
        More volatile markets = wider zones for safety.
        
        Returns:
            {
                'in_resistance_zone': bool,    # Price in top danger zone (no LONG)
                'in_support_zone': bool,       # Price in bottom danger zone (no SHORT)
                'resistance_upper': float,     # Top of resistance zone (24h high)
                'resistance_lower': float,     # Bottom of resistance zone
                'support_upper': float,        # Top of support zone
                'support_lower': float,        # Bottom of support zone (24h low)
                'zone_width_pct': float,       # Zone width as % of price
                ...
            }
        """
        try:
            if df is None or len(df) < 20:
                return {
                    'in_resistance_zone': False,
                    'in_support_zone': False,
                    'at_resistance': False,
                    'at_support': False,
                    'resistance_level': None,
                    'support_level': None,
                    'warning': None
                }
            
            # Get recent highs and lows (last 24 hours ~ 96 bars on 15m)
            recent_df = df.tail(min(lookback_bars, len(df)))
            highs = recent_df['high'].values
            lows = recent_df['low'].values
            closes = recent_df['close'].values
            
            # Find the 24h high and low
            high_24h = highs.max()
            low_24h = lows.min()
            
            # ═══════════════════════════════════════════════════════════════
            # SMART SUPPORT/RESISTANCE DETECTION
            # Find price levels where price has bounced multiple times
            # This is more accurate than simple 24h high/low
            # ═══════════════════════════════════════════════════════════════
            
            # Calculate ATR for clustering tolerance
            if 'atr' in df.columns and len(df) > 0:
                atr = df['atr'].iloc[-1]
            else:
                tr_values = []
                for i in range(1, min(14, len(recent_df))):
                    high = recent_df['high'].iloc[i]
                    low = recent_df['low'].iloc[i]
                    prev_close = recent_df['close'].iloc[i-1]
                    tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                    tr_values.append(tr)
                atr = sum(tr_values) / len(tr_values) if tr_values else current_price * 0.02
            
            # Clustering tolerance: prices within this distance are considered "same level"
            cluster_tolerance = atr * 0.5
            
            # ═══════════════════════════════════════════════════════════════
            # FIND RESISTANCE LEVELS (Price highs that got rejected/bounced down)
            # A valid resistance is where price touched HIGH and then DROPPED
            # ═══════════════════════════════════════════════════════════════
            resistance_clusters = []
            for i in range(2, len(recent_df) - 2):
                high = recent_df['high'].iloc[i]
                # Check if this is a local high (swing high)
                is_swing_high = (
                    high > recent_df['high'].iloc[i-1] and
                    high > recent_df['high'].iloc[i-2] and
                    high > recent_df['high'].iloc[i+1] and
                    high > recent_df['high'].iloc[i+2]
                )
                if is_swing_high:
                    # Check if price bounced down after (rejection)
                    next_close = recent_df['close'].iloc[i+1]
                    if next_close < high * 0.995:  # Price dropped at least 0.5%
                        resistance_clusters.append(high)
            
            # ═══════════════════════════════════════════════════════════════
            # FIND SUPPORT LEVELS (Price lows that got bounced up)
            # A valid support is where price touched LOW and then ROSE
            # ═══════════════════════════════════════════════════════════════
            support_clusters = []
            for i in range(2, len(recent_df) - 2):
                low = recent_df['low'].iloc[i]
                # Check if this is a local low (swing low)
                is_swing_low = (
                    low < recent_df['low'].iloc[i-1] and
                    low < recent_df['low'].iloc[i-2] and
                    low < recent_df['low'].iloc[i+1] and
                    low < recent_df['low'].iloc[i+2]
                )
                if is_swing_low:
                    # Check if price bounced up after
                    next_close = recent_df['close'].iloc[i+1]
                    if next_close > low * 1.005:  # Price rose at least 0.5%
                        support_clusters.append(low)
            
            # ═══════════════════════════════════════════════════════════════
            # CLUSTER THE LEVELS (Group nearby bounces into zones)
            # ═══════════════════════════════════════════════════════════════
            def cluster_levels(levels, tolerance):
                """Group nearby price levels and count touches"""
                if not levels:
                    return []
                levels = sorted(levels)
                clusters = []
                current_cluster = [levels[0]]
                
                for level in levels[1:]:
                    if level - current_cluster[-1] <= tolerance:
                        current_cluster.append(level)
                    else:
                        # Save cluster: (avg_price, touch_count)
                        avg = sum(current_cluster) / len(current_cluster)
                        clusters.append({'level': avg, 'touches': len(current_cluster)})
                        current_cluster = [level]
                
                # Don't forget last cluster
                if current_cluster:
                    avg = sum(current_cluster) / len(current_cluster)
                    clusters.append({'level': avg, 'touches': len(current_cluster)})
                
                return clusters
            
            resistance_zones = cluster_levels(resistance_clusters, cluster_tolerance)
            support_zones = cluster_levels(support_clusters, cluster_tolerance)
            
            # ═══════════════════════════════════════════════════════════════
            # SELECT STRONGEST ZONES (Most touches near current price)
            # ═══════════════════════════════════════════════════════════════
            # Filter to zones above current price for resistance
            valid_resistances = [z for z in resistance_zones if z['level'] > current_price]
            # Sort by touches (most touched first), then by distance (closest first)
            valid_resistances.sort(key=lambda z: (-z['touches'], z['level'] - current_price))
            
            # Filter to zones below current price for support
            valid_supports = [z for z in support_zones if z['level'] < current_price]
            # Sort by touches (most touched first), then by distance (closest first)
            valid_supports.sort(key=lambda z: (-z['touches'], current_price - z['level']))
            
            # ═══════════════════════════════════════════════════════════════
            # DETERMINE FINAL ZONE LEVELS
            # Use bounce-based levels if we have enough data, else fall back to 24h high/low
            # ═══════════════════════════════════════════════════════════════
            zone_width = atr * 0.5  # REDUCED: Zone width based on ATR (was 1.5x, now 0.5x)
            zone_width_pct = (zone_width / current_price) * 100 if current_price > 0 else 1.0
            
            # Clamp zone width - REDUCED for tighter zones
            min_zone_pct, max_zone_pct = 0.3, 1.5  # Was 1.0-5.0%, now 0.3-1.5%
            if zone_width_pct < min_zone_pct:
                zone_width = current_price * (min_zone_pct / 100)
                zone_width_pct = min_zone_pct
            elif zone_width_pct > max_zone_pct:
                zone_width = current_price * (max_zone_pct / 100)
                zone_width_pct = max_zone_pct
            
            # RESISTANCE: Use strongest bounce level if we have 3+ touches, else 24h high
            resistance_touches = 0
            if valid_resistances and valid_resistances[0]['touches'] >= 3:  # CHANGED: Require 3+ bounces (was 2)
                resistance_level = valid_resistances[0]['level']
                resistance_touches = valid_resistances[0]['touches']
                logger.debug(f"Using bounce-based resistance: ${resistance_level:.4f} ({resistance_touches} touches)")
            else:
                resistance_level = high_24h
                # Count how many times price touched 24h high
                for h in highs:
                    if abs(h - high_24h) <= cluster_tolerance:
                        resistance_touches += 1
            
            # SUPPORT: Use strongest bounce level if we have 3+ touches, else 24h low
            support_touches = 0
            if valid_supports and valid_supports[0]['touches'] >= 3:  # CHANGED: Require 3+ bounces (was 2)
                support_level = valid_supports[0]['level']
                support_touches = valid_supports[0]['touches']
                logger.debug(f"Using bounce-based support: ${support_level:.4f} ({support_touches} touches)")
            else:
                support_level = low_24h
                # Count how many times price touched 24h low
                for l in lows:
                    if abs(l - low_24h) <= cluster_tolerance:
                        support_touches += 1
            
            # Define the 4 zone boundaries
            resistance_upper = resistance_level
            resistance_lower = resistance_level - zone_width
            support_upper = support_level + zone_width
            support_lower = support_level
            
            # Shadow/Caution zones (1.5x width of danger zone)
            shadow_width = zone_width * 1.5
            resistance_caution_lower = resistance_lower - shadow_width
            support_caution_upper = support_upper + shadow_width
            
            # Check if current price is in the danger zones
            in_resistance_zone = resistance_lower <= current_price <= resistance_upper
            in_support_zone = support_lower <= current_price <= support_upper
            
            # Check if in caution/shadow zones (allowed but reduce size)
            in_resistance_caution = resistance_caution_lower <= current_price < resistance_lower
            in_support_caution = support_upper < current_price <= support_caution_upper
            
            # ═══════════════════════════════════════════════════════════════
            # NOTE: Touch counts already calculated above using smart detection
            # Legacy counting removed to prevent overwriting smart counts
            # resistance_touches and support_touches are preserved from lines 3602-3624
            # ═══════════════════════════════════════════════════════════════
            
            # Calculate distances
            distance_to_resistance = high_24h - current_price
            distance_to_support = current_price - low_24h
            distance_to_resistance_pct = (distance_to_resistance / current_price) * 100 if current_price > 0 else 0
            distance_to_support_pct = (distance_to_support / current_price) * 100 if current_price > 0 else 0
            
            # Position in 24h range
            range_24h = high_24h - low_24h
            if range_24h > 0:
                position_in_range_pct = ((current_price - low_24h) / range_24h) * 100
            else:
                position_in_range_pct = 50
            
            # Legacy flags (for backward compatibility)
            at_24h_high = distance_to_resistance_pct < 0.15
            at_24h_low = distance_to_support_pct < 0.15
            near_top_of_range = position_in_range_pct >= 95
            near_bottom_of_range = position_in_range_pct <= 5
            at_resistance = distance_to_resistance_pct < 0.5 and resistance_touches >= 2
            at_support = distance_to_support_pct < 0.5 and support_touches >= 2
            
            # ═══════════════════════════════════════════════════════════════
            # GENERATE WARNING MESSAGE
            # ═══════════════════════════════════════════════════════════════
            warning = None
            if in_resistance_zone:
                warning = f"🚫 IN RESISTANCE ZONE [${resistance_lower:.4f} - ${resistance_upper:.4f}] | Zone={zone_width_pct:.1f}% | NO LONG - will likely drop!"
            elif in_support_zone:
                warning = f"🚫 IN SUPPORT ZONE [${support_lower:.4f} - ${support_upper:.4f}] | Zone={zone_width_pct:.1f}% | NO SHORT - will likely bounce!"
            elif in_resistance_caution:
                warning = f"⚠️ CAUTION ZONE (near resistance) - reduce position size"
            elif in_support_caution:
                warning = f"⚠️ CAUTION ZONE (near support) - reduce position size"
            elif distance_to_resistance_pct < zone_width_pct * 1.5:
                warning = f"⚠️ APPROACHING RESISTANCE ({distance_to_resistance_pct:.1f}% away from zone)"
            elif distance_to_support_pct < zone_width_pct * 1.5:
                warning = f"⚠️ APPROACHING SUPPORT ({distance_to_support_pct:.1f}% away from zone)"
            
            return {
                # NEW: Zone-based detection
                'in_resistance_zone': in_resistance_zone,
                'in_support_zone': in_support_zone,
                'in_resistance_caution': in_resistance_caution,  # NEW: Caution zone
                'in_support_caution': in_support_caution,        # NEW: Caution zone
                'resistance_upper': resistance_upper,
                'resistance_lower': resistance_lower,
                'support_upper': support_upper,
                'support_lower': support_lower,
                'zone_width': zone_width,
                'zone_width_pct': zone_width_pct,
                # Legacy fields
                'at_resistance': at_resistance or in_resistance_zone,
                'at_support': at_support or in_support_zone,
                'at_24h_high': at_24h_high,
                'at_24h_low': at_24h_low,
                'near_top_of_range': near_top_of_range or in_resistance_zone,
                'near_bottom_of_range': near_bottom_of_range or in_support_zone,
                'position_in_range_pct': position_in_range_pct,
                'resistance_level': high_24h,
                'support_level': low_24h,
                'resistance_touches': resistance_touches,
                'support_touches': support_touches,
                'distance_to_resistance_pct': distance_to_resistance_pct,
                'distance_to_support_pct': distance_to_support_pct,
                'warning': warning
            }
            
        except Exception as e:
            logger.debug(f"Resistance/support detection error: {e}")
            return {
                'in_resistance_zone': False,
                'in_support_zone': False,
                'at_resistance': False,
                'at_support': False,
                'resistance_level': None,
                'support_level': None,
                'warning': None
            }
    
    def _comprehensive_math_check(
        self,
        signal: int,
        df: pd.DataFrame,
        current_price: float,
        atr: float,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        PhD-Level Comprehensive Mathematical Analysis of Trade Setup.
        
        Implements advanced quantitative methods:
        - Statistical hypothesis testing
        - Bayesian probability updates  
        - Time series analysis (Hurst, autocorrelation)
        - Risk metrics (VaR, Sharpe, Sortino, Calmar)
        - Information theory (entropy-based uncertainty)
        - Regression analysis (price momentum slopes)
        - Order flow imbalance estimation
        - Fractal market analysis
        
        Returns objective scores that can override AI decisions.
        """
        scores = {}
        reasons_for = []
        reasons_against = []
        detailed_analysis = {}
        
        # Get price and return series
        closes = df['close'] if 'close' in df.columns else pd.Series([current_price])
        returns = closes.pct_change().dropna()
        
        # ═══════════════════════════════════════════════════════════════════
        # 0. SOFT PENALTY: OVERBOUGHT LONGS / OVERSOLD SHORTS
        # ═══════════════════════════════════════════════════════════════════
        # These conditions apply heavy penalties but don't hard-block.
        # Math evaluates, AI sees the score, combined decision is made.
        # This allows AI recommendation + math scoring to work together.
        if 'rsi' in df.columns:
            current_rsi = df['rsi'].iloc[-1] if len(df) > 0 else 50
        else:
            # Calculate RSI if not present
            from indicator import calculate_rsi
            rsi_series = calculate_rsi(closes)
            current_rsi = rsi_series.iloc[-1] if len(rsi_series) > 0 else 50
        
        rsi_penalty = 0
        rsi_warning = None
        
        # SOFT PENALTY for overbought LONG (was hard block at 75)
        if signal == 1 and current_rsi >= 70:
            if current_rsi >= 80:
                rsi_penalty = -50  # Extreme overbought: massive penalty
                rsi_warning = f"⚠️ EXTREME OVERBOUGHT: RSI={current_rsi:.0f} (≥80) - Very high reversal risk"
            elif current_rsi >= 75:
                rsi_penalty = -35  # Very overbought: heavy penalty
                rsi_warning = f"⚠️ OVERBOUGHT: RSI={current_rsi:.0f} (≥75) - High reversal risk"
            else:  # 70-74
                rsi_penalty = -20  # Moderately overbought: moderate penalty
                rsi_warning = f"⚠️ RSI elevated: {current_rsi:.0f} (≥70) - Caution for LONG"
            logger.debug(f"📊 [SOFT PENALTY] LONG gets RSI penalty: {rsi_penalty} (RSI={current_rsi:.1f})")
            reasons_against.append(rsi_warning)
            detailed_analysis['rsi_penalty'] = rsi_penalty
        
        # SOFT PENALTY for oversold SHORT (was hard block at 25)
        # Moderate penalties - between harsh original and weak test
        elif signal == -1 and current_rsi <= 30:
            if current_rsi <= 20:
                rsi_penalty = -35  # Moderate: Was -50 original, -20 test
                rsi_warning = f"⚠️ EXTREME OVERSOLD: RSI={current_rsi:.0f} (≤20) - Very high bounce risk"
            elif current_rsi <= 25:
                rsi_penalty = -25  # Moderate: Was -35 original, -15 test
                rsi_warning = f"⚠️ OVERSOLD: RSI={current_rsi:.0f} (≤25) - High bounce risk"
            else:  # 26-30
                rsi_penalty = -15  # Moderate: Was -20 original, -10 test
                rsi_warning = f"⚠️ RSI depressed: {current_rsi:.0f} (≤30) - Caution for SHORT"
            logger.debug(f"📊 [SOFT PENALTY] SHORT gets RSI penalty: {rsi_penalty} (RSI={current_rsi:.1f})")
            reasons_against.append(rsi_warning)
            detailed_analysis['rsi_penalty'] = rsi_penalty
        
        detailed_analysis['rsi'] = current_rsi
        
        # ═══════════════════════════════════════════════════════════════════
        # 0b. SOFT PENALTY: COUNTER-TREND TRADES IN STRONG TRENDS
        # ═══════════════════════════════════════════════════════════════════
        # Apply penalties but allow math to evaluate the full picture.
        # AI + Math combined can make better decisions than hard blocks.
        sma_15 = closes.rolling(15).mean()
        sma_40 = closes.rolling(40).mean()
        
        trend_penalty = 0
        trend_warning = None
        
        if len(sma_15) >= 40 and len(sma_40) >= 40:
            trend_bullish = sma_15.iloc[-1] > sma_40.iloc[-1]
            trend_bearish = sma_15.iloc[-1] < sma_40.iloc[-1]
            
            # Calculate trend strength (how far apart are the SMAs)
            trend_strength = abs((sma_15.iloc[-1] - sma_40.iloc[-1]) / sma_40.iloc[-1] * 100)
            detailed_analysis['trend_strength'] = trend_strength
            
            # SOFT PENALTY: Don't SHORT in a BULLISH trend (reduced penalties)
            if signal == -1 and trend_bullish and trend_strength > 0.5:
                if trend_strength > 2.0:
                    trend_penalty = -25  # Very strong bullish trend
                    trend_warning = f"⚠️ STRONG UPTREND: SMA15 > SMA40 by {trend_strength:.2f}% - SHORT risky"
                elif trend_strength > 1.0:
                    trend_penalty = -15  # Moderate bullish trend
                    trend_warning = f"⚠️ UPTREND: SMA15 > SMA40 by {trend_strength:.2f}% - SHORT caution"
                else:  # 0.5-1.0
                    trend_penalty = -8  # Weak bullish trend
                    trend_warning = f"⚠️ Mild uptrend: SMA15 > SMA40 by {trend_strength:.2f}%"
                logger.debug(f"📊 [SOFT PENALTY] SHORT gets trend penalty: {trend_penalty} (trend={trend_strength:.2f}%)")
                reasons_against.append(trend_warning)
                detailed_analysis['trend_penalty'] = trend_penalty
                detailed_analysis['trend'] = 'bullish'
            
            # SOFT PENALTY: Don't LONG in a BEARISH trend (reduced penalties)
            elif signal == 1 and trend_bearish and trend_strength > 0.5:
                if trend_strength > 2.0:
                    trend_penalty = -25  # Very strong bearish trend
                    trend_warning = f"⚠️ STRONG DOWNTREND: SMA15 < SMA40 by {trend_strength:.2f}% - LONG risky"
                elif trend_strength > 1.0:
                    trend_penalty = -15  # Moderate bearish trend
                    trend_warning = f"⚠️ DOWNTREND: SMA15 < SMA40 by {trend_strength:.2f}% - LONG caution"
                else:  # 0.5-1.0
                    trend_penalty = -8  # Weak bearish trend
                    trend_warning = f"⚠️ Mild downtrend: SMA15 < SMA40 by {trend_strength:.2f}%"
                logger.debug(f"📊 [SOFT PENALTY] LONG gets trend penalty: {trend_penalty} (trend={trend_strength:.2f}%)")
                reasons_against.append(trend_warning)
                detailed_analysis['trend_penalty'] = trend_penalty
                detailed_analysis['trend'] = 'bearish'
        
        # Store penalties to apply at the end
        detailed_analysis['total_soft_penalty'] = rsi_penalty + trend_penalty
        
        # ═══════════════════════════════════════════════════════════════════
        # 1. RISK-REWARD OPTIMIZATION & HYPOTHESIS TESTING (PhD Enhancement)
        # ═══════════════════════════════════════════════════════════════════
        stop_distance = atr * 1.5
        tp1_distance = atr * 1.5
        tp2_distance = atr * 2.5
        tp3_distance = atr * 4.0
        
        # Expected R:R using probability-weighted payoff
        # P(TP1) = 0.50, P(TP2|TP1) = 0.60, P(TP3|TP2) = 0.40
        # This is more realistic than simple averaging
        p_tp1 = 0.50
        p_tp2_given_tp1 = 0.60
        p_tp3_given_tp2 = 0.40
        
        expected_profit = (
            p_tp1 * tp1_distance +
            p_tp1 * p_tp2_given_tp1 * (tp2_distance - tp1_distance) +
            p_tp1 * p_tp2_given_tp1 * p_tp3_given_tp2 * (tp3_distance - tp2_distance)
        )
        expected_loss = (1 - p_tp1) * stop_distance
        expected_rr = expected_profit / stop_distance if stop_distance > 0 else 1
        
        # Kelly optimal fraction: f* = (p*b - q) / b where b = win/loss ratio
        win_prob = p_tp1
        win_loss_ratio = expected_profit / expected_loss if expected_loss > 0 else 1
        kelly_f = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        kelly_f = max(0, min(0.25, kelly_f))  # Cap at 25%
        
        scores['risk_reward'] = min(100, expected_rr * 40 + kelly_f * 200)
        detailed_analysis['kelly_fraction'] = kelly_f
        detailed_analysis['expected_rr'] = expected_rr
        
        # PhD Enhancement: Test if returns have statistically significant edge
        try:
            from indicator import calculate_hypothesis_test
            
            hypo_result = calculate_hypothesis_test(returns, null_mean=0)
            detailed_analysis['hypothesis_test'] = hypo_result
            
            p_value = hypo_result.get('p_value', 1.0)
            significant = hypo_result.get('significant', False)
            mean_return = hypo_result.get('mean_return', 0)
            
            if significant and mean_return > 0:
                reasons_for.append(f"Statistically significant edge (p={p_value:.4f}, mean={mean_return:.4f}%)")
                scores['risk_reward'] = min(100, scores['risk_reward'] * 1.2)
            elif significant and mean_return < 0:
                reasons_against.append(f"Significant negative returns (p={p_value:.4f})")
                scores['risk_reward'] = max(0, scores['risk_reward'] * 0.7)
            else:
                # No statistical edge - this goes AGAINST the trade, not for it
                reasons_against.append(f"No significant edge yet (p={p_value:.3f})")
                scores['risk_reward'] = max(0, scores['risk_reward'] * 0.85)  # Penalize score
                detailed_analysis['no_statistical_edge'] = True
        except Exception as e:
            logger.debug(f"Hypothesis test failed: {e}")
        
        if expected_rr >= 1.8:
            reasons_for.append(f"Excellent R:R ({expected_rr:.2f}:1), Kelly f*={kelly_f:.1%}")
        elif expected_rr >= 1.2:
            reasons_for.append(f"Good R:R ({expected_rr:.2f}:1)")
        else:
            reasons_against.append(f"Poor R:R ({expected_rr:.2f}:1)")
        
        # ═══════════════════════════════════════════════════════════════════
        # 2. TREND STRENGTH VIA LINEAR REGRESSION SLOPE & R²
        # ═══════════════════════════════════════════════════════════════════
        if len(closes) >= 20:
            # Fit OLS regression to log prices (better for exponential trends)
            log_prices = np.log(closes.tail(20).values)
            x = np.arange(len(log_prices))
            
            # Calculate regression coefficients
            x_mean = x.mean()
            y_mean = log_prices.mean()
            
            numerator = np.sum((x - x_mean) * (log_prices - y_mean))
            denominator = np.sum((x - x_mean) ** 2)
            
            if denominator > 0:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean
                
                # Calculate R² (coefficient of determination)
                y_pred = slope * x + intercept
                ss_res = np.sum((log_prices - y_pred) ** 2)
                ss_tot = np.sum((log_prices - y_mean) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Annualized slope (assuming 15-min bars)
                bars_per_year = 365 * 24 * 4  # 4 bars per hour
                # Clip to prevent overflow (max reasonable annual return is ~10000%)
                exp_arg = np.clip(slope * bars_per_year, -10, 10)
                annualized_return = (np.exp(exp_arg) - 1) * 100
                
                detailed_analysis['regression_slope'] = slope
                detailed_analysis['r_squared'] = r_squared
                detailed_analysis['annualized_trend'] = annualized_return
                
                # Score based on slope direction matching signal AND R²
                trend_matches = (signal == 1 and slope > 0) or (signal == -1 and slope < 0)
                slope_strength = abs(slope) * 10000  # Scale for scoring
                
                if trend_matches:
                    scores['trend'] = min(100, 50 + slope_strength * 50 + r_squared * 30)
                    if r_squared > 0.7:
                        reasons_for.append(f"Strong trend (R²={r_squared:.2f}, ann. {annualized_return:+.0f}%)")
                    elif r_squared > 0.4:
                        reasons_for.append(f"Moderate trend (R²={r_squared:.2f})")
                else:
                    scores['trend'] = max(0, 50 - slope_strength * 50)
                    reasons_against.append(f"Counter-trend trade (R²={r_squared:.2f})")
            else:
                scores['trend'] = 50
        else:
            scores['trend'] = 50
        
        # ═══════════════════════════════════════════════════════════════════
        # 3. MARKET REGIME DETECTION - HIDDEN MARKOV MODEL (PhD Enhancement)
        # ═══════════════════════════════════════════════════════════════════
        try:
            from indicator import calculate_regime_hmm
            
            hmm_result = calculate_regime_hmm(returns)
            if hmm_result.get('status') == 'success':
                regime = hmm_result.get('regime', 'NORMAL')
                regime_confidence = hmm_result.get('confidence', 0.5)
                persistence = hmm_result.get('persistence', 0.5)
                
                detailed_analysis['market_regime'] = regime
                detailed_analysis['regime_confidence'] = regime_confidence
                detailed_analysis['regime_persistence'] = persistence
                
                # Use HMM regime instead of Hurst for scoring
                if regime == 'CALM' and signal == 1:
                    scores['hurst'] = 85
                    reasons_for.append(f"HMM: CALM regime (conf={regime_confidence:.0%}) - good for breakout LONG")
                elif regime == 'CALM' and signal == -1:
                    scores['hurst'] = 50
                    reasons_against.append(f"HMM: CALM regime - SHORT less favorable")
                elif regime == 'VOLATILE' and signal == 1:
                    scores['hurst'] = 40
                    reasons_against.append(f"HMM: VOLATILE regime - risky LONG")
                elif regime == 'VOLATILE' and signal == -1:
                    scores['hurst'] = 80
                    reasons_for.append(f"HMM: VOLATILE regime (conf={regime_confidence:.0%}) - good for SHORT")
                else:
                    scores['hurst'] = 70
                    reasons_for.append(f"HMM: {regime} regime (persistence={persistence:.0%})")
            else:
                # Fallback to Hurst calculation
                hurst = self._calculate_hurst(closes)
                detailed_analysis['hurst_exponent'] = hurst
                
                if hurst > 0.6:
                    scores['hurst'] = 90
                    reasons_for.append(f"Trending market (H={hurst:.2f}>0.5)")
                elif hurst < 0.4:
                    scores['hurst'] = 70
                    reasons_for.append(f"Mean-reverting (H={hurst:.2f})")
                else:
                    scores['hurst'] = 60
        except Exception as e:
            logger.warning(f"HMM regime detection failed: {e}")
            # Fallback to Hurst
            hurst = self._calculate_hurst(closes)
            detailed_analysis['hurst_exponent'] = hurst
            scores['hurst'] = 80 if hurst > 0.55 else 60 if hurst > 0.45 else 70
        
        # ═══════════════════════════════════════════════════════════════════
        # 4. VOLATILITY CLUSTERING & GARCH-STYLE ANALYSIS (PhD Enhancement)
        # ═══════════════════════════════════════════════════════════════════
        try:
            # Import GARCH function from indicator.py
            from indicator import calculate_garch_volatility
            
            garch_result = calculate_garch_volatility(returns)
            if garch_result.get('status') == 'success':
                current_vol = garch_result.get('current_vol', 0)
                forecast_vol = garch_result.get('forecast_vol', 0)
                vol_trend = garch_result.get('vol_trend', 'unknown')
                vol_change = garch_result.get('vol_change_pct', 0)
                
                detailed_analysis['garch_vol'] = current_vol
                detailed_analysis['garch_forecast'] = forecast_vol
                detailed_analysis['vol_trend'] = vol_trend
                detailed_analysis['vol_change_pct'] = vol_change
                
                # Score based on volatility regime and forecast
                if vol_trend == 'decreasing' and 0.7 <= current_vol / forecast_vol <= 1.3:
                    scores['volatility'] = 90
                    reasons_for.append(f"GARCH: Stable vol (curr={current_vol:.2%}, forecast {vol_trend})")
                elif vol_trend == 'increasing':
                    scores['volatility'] = 40 + vol_change / 10
                    reasons_against.append(f"GARCH: Vol expanding (+{vol_change:.1f}%) - rising risk")
                else:
                    scores['volatility'] = 75
                    reasons_for.append(f"GARCH: Vol contracting ({vol_change:.1f}%)")
            else:
                # Fallback to simple volatility if GARCH fails
                short_vol = returns.tail(10).std() * np.sqrt(365 * 24 * 4)
                long_vol = returns.tail(30).std() * np.sqrt(365 * 24 * 4)
                vol_ratio = short_vol / long_vol if long_vol > 0 else 1
                
                detailed_analysis['short_vol'] = short_vol
                detailed_analysis['long_vol'] = long_vol
                detailed_analysis['vol_ratio'] = vol_ratio
                
                if 0.7 <= vol_ratio <= 1.3:
                    scores['volatility'] = 85
                else:
                    scores['volatility'] = 70
        except Exception as e:
            logger.warning(f"GARCH analysis failed: {e}")
            # Simple fallback
            short_vol = returns.tail(10).std() * np.sqrt(365 * 24 * 4)
            long_vol = returns.tail(30).std() * np.sqrt(365 * 24 * 4)
            vol_ratio = short_vol / long_vol if long_vol > 0 else 1
            scores['volatility'] = 75 if 0.7 <= vol_ratio <= 1.3 else 60
        
        # ═══════════════════════════════════════════════════════════════════
        # 5. Z-SCORE MEAN REVERSION ANALYSIS
        # ═══════════════════════════════════════════════════════════════════
        if len(closes) >= 50:
            rolling_mean = closes.rolling(20).mean()
            rolling_std = closes.rolling(20).std()
            
            z_score = (current_price - rolling_mean.iloc[-1]) / rolling_std.iloc[-1] if rolling_std.iloc[-1] > 0 else 0
            detailed_analysis['z_score'] = z_score
            
            # Extreme z-scores suggest mean reversion
            if signal == 1:  # LONG
                if z_score < -2:
                    scores['zscore'] = 95
                    reasons_for.append(f"Extreme oversold (z={z_score:.2f}σ) - high reversion probability")
                elif z_score < -1:
                    scores['zscore'] = 80
                    reasons_for.append(f"Oversold (z={z_score:.2f}σ)")
                elif z_score > 2:
                    scores['zscore'] = 30
                    reasons_against.append(f"Overbought (z={z_score:.2f}σ) - risky LONG")
                else:
                    scores['zscore'] = 60
            else:  # SHORT
                if z_score > 2:
                    scores['zscore'] = 95
                    reasons_for.append(f"Extreme overbought (z={z_score:.2f}σ) - high reversion probability")
                elif z_score > 1:
                    scores['zscore'] = 80
                    reasons_for.append(f"Overbought (z={z_score:.2f}σ)")
                elif z_score < -2:
                    scores['zscore'] = 30
                    reasons_against.append(f"Oversold (z={z_score:.2f}σ) - risky SHORT")
                else:
                    scores['zscore'] = 60
        else:
            scores['zscore'] = 50
        
        # ═══════════════════════════════════════════════════════════════════
        # 6. ENHANCED RSI DIVERGENCE DETECTION (Multi-Timeframe)
        # ═══════════════════════════════════════════════════════════════════
        if 'rsi' in df.columns and len(df) >= 20:
            rsi = df['rsi'].iloc[-1]
            rsi_prev = df['rsi'].iloc[-10] if len(df) >= 10 else rsi
            price_change = (current_price / closes.iloc[-10] - 1) * 100 if len(closes) >= 10 else 0
            rsi_change = rsi - rsi_prev
            
            # Basic divergence detection (fallback)
            bullish_divergence = price_change < 0 and rsi_change > 5
            bearish_divergence = price_change > 0 and rsi_change < -5
            
            detailed_analysis['rsi'] = rsi
            detailed_analysis['bullish_divergence'] = bullish_divergence
            detailed_analysis['bearish_divergence'] = bearish_divergence
            
            # === ENHANCED: Multi-Timeframe RSI Divergence ===
            try:
                from indicator import calculate_rsi_divergence_mtf
                
                # Get higher timeframe data from context if available
                df_higher = context.get('df_higher_tf', None)
                div_result = calculate_rsi_divergence_mtf(df, df_higher)
                
                if div_result.get('status') == 'success':
                    reg_bullish = div_result['regular_bullish_divergence']
                    reg_bearish = div_result['regular_bearish_divergence']
                    hid_bullish = div_result['hidden_bullish_divergence']
                    hid_bearish = div_result['hidden_bearish_divergence']
                    mtf_confirm = div_result['mtf_confirmation']
                    div_strength = div_result['signal_confidence']
                    
                    detailed_analysis['rsi_divergence_mtf'] = div_result
                    
                    # Override basic divergence with enhanced detection
                    bullish_divergence = reg_bullish or hid_bullish
                    bearish_divergence = reg_bearish or hid_bearish
                    
                    if signal == 1:  # LONG
                        if reg_bullish:
                            mtf_bonus = 10 if mtf_confirm else 0
                            scores['momentum'] = min(100, 90 + mtf_bonus)
                            mtf_str = " (MTF confirmed!)" if mtf_confirm else ""
                            reasons_for.append(f"🔄 Regular bullish RSI divergence{mtf_str} (RSI={rsi:.0f}, strength={div_strength:.0f})")
                        elif hid_bullish:
                            scores['momentum'] = 85
                            reasons_for.append(f"🔄 Hidden bullish divergence - continuation signal (RSI={rsi:.0f})")
                        elif 30 <= rsi <= 50:
                            scores['momentum'] = 80
                            reasons_for.append(f"RSI {rsi:.0f} - optimal LONG zone")
                        elif rsi > 70:
                            scores['momentum'] = 35
                            reasons_against.append(f"RSI {rsi:.0f} - overbought")
                        elif bearish_divergence:
                            scores['momentum'] = 40
                            reasons_against.append(f"⚠️ Bearish RSI divergence conflicts with LONG (RSI={rsi:.0f})")
                        else:
                            scores['momentum'] = 65
                    else:  # SHORT
                        if reg_bearish:
                            mtf_bonus = 10 if mtf_confirm else 0
                            scores['momentum'] = min(100, 90 + mtf_bonus)
                            mtf_str = " (MTF confirmed!)" if mtf_confirm else ""
                            reasons_for.append(f"🔄 Regular bearish RSI divergence{mtf_str} (RSI={rsi:.0f}, strength={div_strength:.0f})")
                        elif hid_bearish:
                            scores['momentum'] = 85
                            reasons_for.append(f"🔄 Hidden bearish divergence - continuation signal (RSI={rsi:.0f})")
                        elif 50 <= rsi <= 70:
                            scores['momentum'] = 80
                            reasons_for.append(f"RSI {rsi:.0f} - optimal SHORT zone")
                        elif rsi < 30:
                            scores['momentum'] = 35
                            reasons_against.append(f"RSI {rsi:.0f} - oversold")
                        elif bullish_divergence:
                            scores['momentum'] = 40
                            reasons_against.append(f"⚠️ Bullish RSI divergence conflicts with SHORT (RSI={rsi:.0f})")
                        else:
                            scores['momentum'] = 65
                else:
                    # Fallback to basic divergence
                    if signal == 1:
                        if bullish_divergence:
                            scores['momentum'] = 95
                            reasons_for.append(f"Bullish RSI divergence detected (RSI={rsi:.0f})")
                        elif 30 <= rsi <= 50:
                            scores['momentum'] = 85
                            reasons_for.append(f"RSI {rsi:.0f} - optimal LONG zone")
                        elif rsi > 70:
                            scores['momentum'] = 35
                            reasons_against.append(f"RSI {rsi:.0f} - overbought")
                        else:
                            scores['momentum'] = 65
                    else:
                        if bearish_divergence:
                            scores['momentum'] = 95
                            reasons_for.append(f"Bearish RSI divergence detected (RSI={rsi:.0f})")
                        elif 50 <= rsi <= 70:
                            scores['momentum'] = 85
                            reasons_for.append(f"RSI {rsi:.0f} - optimal SHORT zone")
                        elif rsi < 30:
                            scores['momentum'] = 35
                            reasons_against.append(f"RSI {rsi:.0f} - oversold")
                        else:
                            scores['momentum'] = 65
                            
            except Exception as div_error:
                logger.debug(f"MTF RSI divergence failed: {div_error}")
                # Fallback to basic logic
                if signal == 1:
                    if bullish_divergence:
                        scores['momentum'] = 95
                        reasons_for.append(f"Bullish RSI divergence detected (RSI={rsi:.0f})")
                    elif 30 <= rsi <= 50:
                        scores['momentum'] = 85
                        reasons_for.append(f"RSI {rsi:.0f} - optimal LONG zone")
                    elif rsi > 70:
                        scores['momentum'] = 35
                        reasons_against.append(f"RSI {rsi:.0f} - overbought")
                    else:
                        scores['momentum'] = 65
                else:
                    if bearish_divergence:
                        scores['momentum'] = 95
                        reasons_for.append(f"Bearish RSI divergence detected (RSI={rsi:.0f})")
                    elif 50 <= rsi <= 70:
                        scores['momentum'] = 85
                        reasons_for.append(f"RSI {rsi:.0f} - optimal SHORT zone")
                    elif rsi < 30:
                        scores['momentum'] = 35
                        reasons_against.append(f"RSI {rsi:.0f} - oversold")
                    else:
                        scores['momentum'] = 65
        else:
            scores['momentum'] = 50
        
        # ═══════════════════════════════════════════════════════════════════
        # 7. ADVANCED VOLUME PROFILE ANALYSIS (TPO-style with POC)
        # ═══════════════════════════════════════════════════════════════════
        volume_ratio = context.get('volume_ratio', 1.0)
        
        if 'volume' in df.columns and len(df) >= 20:
            volumes = df['volume'].tail(20)
            vol_mean = volumes.mean()
            vol_std = volumes.std()
            current_vol = volumes.iloc[-1]
            
            # Volume z-score
            vol_zscore = (current_vol - vol_mean) / vol_std if vol_std > 0 else 0
            detailed_analysis['volume_zscore'] = vol_zscore
            
            # Calculate approximate VWAP
            if 'close' in df.columns:
                typical_price = (df['high'].tail(20) + df['low'].tail(20) + df['close'].tail(20)) / 3
                vwap = (typical_price * volumes).sum() / volumes.sum()
                vwap_distance = (current_price - vwap) / vwap * 100
                detailed_analysis['vwap_distance'] = vwap_distance
                
                # === ENHANCED: Full Volume Profile with POC ===
                try:
                    from indicator import calculate_volume_profile
                    vp_result = calculate_volume_profile(df)
                    
                    if vp_result.get('status') == 'success':
                        poc_price = vp_result['poc_price']
                        va_high = vp_result['value_area_high']
                        va_low = vp_result['value_area_low']
                        in_value_area = vp_result['in_value_area']
                        price_vs_poc = vp_result['current_vs_poc_pct']
                        support_distance = vp_result['support_distance_pct']
                        resistance_distance = vp_result['resistance_distance_pct']
                        
                        detailed_analysis['poc_price'] = poc_price
                        detailed_analysis['value_area_high'] = va_high
                        detailed_analysis['value_area_low'] = va_low
                        detailed_analysis['in_value_area'] = in_value_area
                        detailed_analysis['price_vs_poc'] = price_vs_poc
                        detailed_analysis['nearest_support'] = vp_result['nearest_support']
                        detailed_analysis['nearest_resistance'] = vp_result['nearest_resistance']
                        
                        # Volume Profile Scoring:
                        # - LONG: Better if near support (POC below price) with room to resistance
                        # - SHORT: Better if near resistance (POC above price) with room to support
                        if signal == 1:  # LONG
                            if price_vs_poc < -0.5 and support_distance < 1.0:
                                # Price below POC, near support - excellent for LONG
                                scores['volume'] = min(100, 85 + vol_zscore * 5)
                                reasons_for.append(f"VP: Price below POC ({price_vs_poc:.1f}%), near support, VA:{va_low:.0f}-{va_high:.0f}")
                            elif in_value_area and current_price > vwap:
                                # In value area above VWAP - good for LONG
                                scores['volume'] = min(100, 75 + vol_zscore * 5)
                                reasons_for.append(f"VP: In value area, above VWAP ({vwap_distance:+.2f}%)")
                            elif price_vs_poc > 1.0:
                                # Far above POC - risky LONG, may revert
                                scores['volume'] = max(30, 50 - abs(price_vs_poc) * 5)
                                reasons_against.append(f"VP: Extended above POC ({price_vs_poc:.1f}%) - mean reversion risk")
                            else:
                                scores['volume'] = 60 + vol_zscore * 5
                        else:  # SHORT
                            if price_vs_poc > 0.5 and resistance_distance < 1.0:
                                # Price above POC, near resistance - excellent for SHORT
                                scores['volume'] = min(100, 85 + vol_zscore * 5)
                                reasons_for.append(f"VP: Price above POC ({price_vs_poc:.1f}%), near resistance, VA:{va_low:.0f}-{va_high:.0f}")
                            elif in_value_area and current_price < vwap:
                                # In value area below VWAP - good for SHORT
                                scores['volume'] = min(100, 75 + vol_zscore * 5)
                                reasons_for.append(f"VP: In value area, below VWAP ({vwap_distance:.2f}%)")
                            elif price_vs_poc < -1.0:
                                # Far below POC - risky SHORT, may bounce
                                scores['volume'] = max(30, 50 - abs(price_vs_poc) * 5)
                                reasons_against.append(f"VP: Extended below POC ({price_vs_poc:.1f}%) - bounce risk")
                            else:
                                scores['volume'] = 60 + vol_zscore * 5
                    else:
                        # Fallback to basic VWAP analysis
                        if signal == 1 and current_price > vwap:
                            scores['volume'] = min(100, 70 + vol_zscore * 10)
                            reasons_for.append(f"Price above VWAP (+{vwap_distance:.2f}%), volume z={vol_zscore:.1f}")
                        elif signal == -1 and current_price < vwap:
                            scores['volume'] = min(100, 70 + vol_zscore * 10)
                            reasons_for.append(f"Price below VWAP ({vwap_distance:.2f}%), volume z={vol_zscore:.1f}")
                        else:
                            scores['volume'] = 50
                except Exception as vp_error:
                    logger.debug(f"Volume Profile calculation failed: {vp_error}")
                    # Fallback to basic VWAP
                    if signal == 1 and current_price > vwap:
                        scores['volume'] = min(100, 70 + vol_zscore * 10)
                        reasons_for.append(f"Price above VWAP (+{vwap_distance:.2f}%), volume z={vol_zscore:.1f}")
                    elif signal == -1 and current_price < vwap:
                        scores['volume'] = min(100, 70 + vol_zscore * 10)
                        reasons_for.append(f"Price below VWAP ({vwap_distance:.2f}%), volume z={vol_zscore:.1f}")
                    else:
                        scores['volume'] = 50
            else:
                scores['volume'] = 50 + volume_ratio * 25
        else:
            if volume_ratio >= 1.5:
                scores['volume'] = 90
                reasons_for.append(f"Strong volume confirmation ({volume_ratio:.1f}x)")
            elif volume_ratio >= 1.0:
                scores['volume'] = 70
            else:
                scores['volume'] = 40
                reasons_against.append(f"Weak volume ({volume_ratio:.1f}x)")
        
        # ═══════════════════════════════════════════════════════════════════
        # 8. KALMAN FILTER SMOOTHED MOMENTUM ANALYSIS
        # ═══════════════════════════════════════════════════════════════════
        try:
            from indicator import calculate_kalman_momentum
            
            kalman_result = calculate_kalman_momentum(closes)
            
            if kalman_result.get('status') == 'success':
                kalman_momentum = kalman_result['kalman_momentum']
                kalman_accel = kalman_result['kalman_acceleration']
                momentum_zscore = kalman_result['momentum_zscore']
                momentum_conf = kalman_result['momentum_confidence']
                momentum_trend = kalman_result['momentum_trend']
                signal_strength = kalman_result['signal_strength']
                
                detailed_analysis['kalman_momentum'] = kalman_momentum
                detailed_analysis['kalman_acceleration'] = kalman_accel
                detailed_analysis['kalman_momentum_zscore'] = momentum_zscore
                detailed_analysis['kalman_momentum_trend'] = momentum_trend
                detailed_analysis['kalman_signal_strength'] = signal_strength
                
                # Kalman Momentum Scoring:
                # - Positive momentum + strengthening = good for LONG
                # - Negative momentum + strengthening = good for SHORT
                # - High signal strength = high confidence
                
                if signal == 1:  # LONG
                    if kalman_momentum > 0 and momentum_trend == 'strengthening':
                        kalman_score = min(100, 75 + signal_strength * 15)
                        reasons_for.append(f"⚡ Kalman: Bullish momentum strengthening (z={momentum_zscore:.1f}, conf={momentum_conf:.0%})")
                    elif kalman_momentum > 0:
                        kalman_score = 70 + signal_strength * 10
                        reasons_for.append(f"⚡ Kalman: Positive momentum (z={momentum_zscore:.1f})")
                    elif kalman_momentum < 0 and momentum_trend == 'weakening':
                        kalman_score = 60  # Bearish momentum weakening - potential reversal
                        reasons_for.append(f"⚡ Kalman: Bearish momentum weakening - potential reversal")
                    elif kalman_momentum < 0 and momentum_trend == 'strengthening':
                        kalman_score = 30  # Bearish momentum strengthening - bad for LONG
                        reasons_against.append(f"⚡ Kalman: Bearish momentum strengthening (z={momentum_zscore:.1f})")
                    else:
                        kalman_score = 50
                else:  # SHORT
                    if kalman_momentum < 0 and momentum_trend == 'strengthening':
                        kalman_score = min(100, 75 + signal_strength * 15)
                        reasons_for.append(f"⚡ Kalman: Bearish momentum strengthening (z={momentum_zscore:.1f}, conf={momentum_conf:.0%})")
                    elif kalman_momentum < 0:
                        kalman_score = 70 + signal_strength * 10
                        reasons_for.append(f"⚡ Kalman: Negative momentum (z={momentum_zscore:.1f})")
                    elif kalman_momentum > 0 and momentum_trend == 'weakening':
                        kalman_score = 60  # Bullish momentum weakening - potential reversal
                        reasons_for.append(f"⚡ Kalman: Bullish momentum weakening - potential reversal")
                    elif kalman_momentum > 0 and momentum_trend == 'strengthening':
                        kalman_score = 30  # Bullish momentum strengthening - bad for SHORT
                        reasons_against.append(f"⚡ Kalman: Bullish momentum strengthening (z={momentum_zscore:.1f})")
                    else:
                        kalman_score = 50
                
                scores['kalman'] = kalman_score
            else:
                scores['kalman'] = 50
        except Exception as kalman_error:
            logger.debug(f"Kalman momentum failed: {kalman_error}")
            scores['kalman'] = 50
        
        # ═══════════════════════════════════════════════════════════════════
        # 9. AUTOCORRELATION ANALYSIS (Predictability)
        # ═══════════════════════════════════════════════════════════════════
        if len(returns) >= 30:
            # Lag-1 autocorrelation
            autocorr_1 = returns.autocorr(lag=1) if hasattr(returns, 'autocorr') else 0
            autocorr_1 = autocorr_1 if not np.isnan(autocorr_1) else 0
            detailed_analysis['autocorrelation'] = autocorr_1
            
            # Positive autocorr = momentum, Negative = mean reversion
            if abs(autocorr_1) > 0.1:
                if autocorr_1 > 0:
                    scores['autocorr'] = 80
                    reasons_for.append(f"Momentum persistence (ρ={autocorr_1:.2f})")
                else:
                    scores['autocorr'] = 75
                    reasons_for.append(f"Mean-reversion pattern (ρ={autocorr_1:.2f})")
            else:
                scores['autocorr'] = 50
                reasons_against.append(f"Low predictability (ρ={autocorr_1:.2f})")
        else:
            scores['autocorr'] = 50
        
        # ═══════════════════════════════════════════════════════════════════
        # 10. SKEWNESS & KURTOSIS (Tail Risk)
        # FIX: Use continuous scoring instead of hard thresholds.
        # Previously: skew>0.3 → LONG=85/SHORT=40, a 45-point gap.
        # Now: Score continuously relative to neutral (60), capped ±20.
        # ═══════════════════════════════════════════════════════════════════
        if len(returns) >= 30:
            from scipy import stats as scipy_stats
            skewness = scipy_stats.skew(returns.dropna())
            kurtosis = scipy_stats.kurtosis(returns.dropna())
            
            detailed_analysis['skewness'] = skewness
            detailed_analysis['kurtosis'] = kurtosis
            
            # Continuous symmetric scoring: skew * impact factor
            # Positive skew helps LONG, hurts SHORT. Negative skew does opposite.
            # Cap at ±20 from neutral 60
            skew_impact = min(20, max(-20, skewness * 15))
            
            if signal == 1:  # LONG
                scores['tail_risk'] = max(30, min(80, 60 + skew_impact))
                if skewness > 0.5:
                    reasons_for.append(f"Positive skew ({skewness:.2f}) - upside potential")
                elif skewness < -0.5:
                    reasons_against.append(f"Negative skew ({skewness:.2f}) - downside risk")
            else:  # SHORT
                scores['tail_risk'] = max(30, min(80, 60 - skew_impact))
                if skewness < -0.5:
                    reasons_for.append(f"Negative skew ({skewness:.2f}) - downside potential")
                elif skewness > 0.5:
                    reasons_against.append(f"Positive skew ({skewness:.2f}) - upside risk")
            
            # Fat tails = higher risk of extreme moves
            if kurtosis > 3:
                reasons_against.append(f"Fat tails (kurt={kurtosis:.1f}) - extreme move risk")
                scores['tail_risk'] = scores.get('tail_risk', 50) * 0.9
        else:
            scores['tail_risk'] = 50
        
        # ═══════════════════════════════════════════════════════════════════
        # 11. SHARPE & SORTINO RATIO PROJECTION
        # FIX: Score SYMMETRICALLY for both directions. A positive Sharpe
        # should boost LONG AND penalize SHORT equally, and vice versa.
        # Previously: LONG=100/SHORT=50 gap was the #1 source of LONG bias.
        # Now: Both directions scored relative to neutral (50).
        # ═══════════════════════════════════════════════════════════════════
        if len(returns) >= 30:
            # Historical Sharpe (annualized for display)
            excess_returns = returns - 0.05 / (365 * 24 * 4)  # 5% annual risk-free rate
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(365 * 24 * 4) if excess_returns.std() > 0 else 0
            
            # Sortino (annualized for display)
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 5 else returns.std()
            sortino = excess_returns.mean() / downside_std * np.sqrt(365 * 24 * 4) if downside_std > 0 else 0
            
            detailed_analysis['sharpe_ratio'] = sharpe
            detailed_analysis['sortino_ratio'] = sortino
            
            # For SCORING: use non-annualized ratio to avoid 187x amplification
            # Raw Sharpe (mean/std) is typically -0.1 to +0.1 for 15m bars
            raw_sharpe = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
            
            # SYMMETRIC scoring: positive returns → LONG gets boost, SHORT gets penalty
            # Scale: raw_sharpe of ±0.05 = moderate signal, ±0.1 = strong signal
            # Cap impact at ±15 from neutral 50 (more conservative than before)
            sharpe_impact = min(15, max(-15, raw_sharpe * 200))  # ±15 max
            
            if signal == 1:  # LONG
                scores['risk_metrics'] = max(35, min(65, 50 + sharpe_impact))
                if sharpe > 1.0:
                    reasons_for.append(f"Favorable risk metrics (Sharpe={sharpe:.2f}, Sortino={sortino:.2f})")
                elif sharpe < -1.0:
                    reasons_against.append(f"Negative Sharpe ({sharpe:.2f}) - unfavorable for LONG")
            else:  # SHORT
                scores['risk_metrics'] = max(35, min(65, 50 - sharpe_impact))
                if sharpe < -1.0:
                    reasons_for.append(f"Favorable risk metrics for SHORT (Sharpe={sharpe:.2f})")
                elif sharpe > 1.0:
                    reasons_against.append(f"Positive Sharpe ({sharpe:.2f}) - unfavorable for SHORT")
        else:
            scores['risk_metrics'] = 50
        
        # ═══════════════════════════════════════════════════════════════════
        # 11. SYSTEM SCORE & ML ALIGNMENT
        # ═══════════════════════════════════════════════════════════════════
        system_combined = context.get('system_score', {}).get('combined', 50)
        scores['system'] = system_combined
        if system_combined >= 75:
            reasons_for.append(f"System score {system_combined}/100 (excellent)")
        elif system_combined >= 60:
            reasons_for.append(f"System score {system_combined}/100 (good)")
        elif system_combined < 40:
            reasons_against.append(f"System score {system_combined}/100 (weak)")
        
        ml_prob = context.get('ml_insight', {}).get('ml_win_probability', 0.5)
        ml_available = context.get('ml_insight', {}).get('ml_available', False)
        scores['ml'] = ml_prob * 100
        if ml_available:
            if ml_prob >= 0.65:
                reasons_for.append(f"ML: {ml_prob*100:.0f}% win probability")
            elif ml_prob < 0.4:
                reasons_against.append(f"ML: only {ml_prob*100:.0f}% win probability")
        
        # ═══════════════════════════════════════════════════════════════════
        # 12. ADVANCED ANALYSIS: Tick-Level PhD Metrics Integration
        # Uses real-time WebSocket tick data to score the ACTUAL price
        # trajectory at sub-candle granularity. This sees what even 1-min
        # candles miss: velocity changes, momentum decay, path efficiency.
        # The Bayesian system will LEARN how much to trust this vs candle data.
        # ═══════════════════════════════════════════════════════════════════
        _atomic = context.get('atomic_analysis') if context else None
        if _atomic is not None:
            try:
                _a_vel = _atomic.velocity              # %/sec
                _a_acc = _atomic.acceleration           # d²P/dt²
                _a_peff = _atomic.path_efficiency       # 0-1
                _a_dcons = _atomic.direction_consistency # 0-100
                _a_mdecay = _atomic.momentum_decay_rate
                _a_hurst = _atomic.hurst_micro          # tick-level Hurst
                _a_eq = _atomic.entry_quality           # 0-100
                _a_rp = _atomic.reversal_probability    # 0-100
                _a_mt = _atomic.micro_trend
                _a_sc = _atomic.swing_count
                
                atomic_score = 50  # Neutral baseline
                atomic_reasons_for = []
                atomic_reasons_against = []
                
                # A) VELOCITY ALIGNMENT: Does tick velocity match trade direction?
                # This is the most direct sub-candle signal.
                vel_aligns = (signal == 1 and _a_vel > 0) or (signal == -1 and _a_vel < 0)
                vel_strength = abs(_a_vel) * 10000  # Normalize for scoring
                
                if vel_aligns:
                    vel_bonus = min(15, vel_strength * 5)  # Cap +15
                    atomic_score += vel_bonus
                    if vel_strength > 1.0:
                        atomic_reasons_for.append(f"🔬 Tick velocity strongly aligns ({_a_vel:+.5f}%/s)")
                else:
                    vel_penalty = min(15, vel_strength * 5)  # Cap -15
                    atomic_score -= vel_penalty
                    if vel_strength > 1.0:
                        atomic_reasons_against.append(f"🔬 Tick velocity CONTRADICTS ({_a_vel:+.5f}%/s)")
                
                # B) PATH EFFICIENCY: Clean moves are more reliable
                if _a_peff > 0.6:
                    atomic_score += 8
                    atomic_reasons_for.append(f"🔬 Clean price path (eff={_a_peff:.2f})")
                elif _a_peff < 0.2:
                    atomic_score -= 8
                    atomic_reasons_against.append(f"🔬 Choppy price path (eff={_a_peff:.2f})")
                
                # C) ENTRY QUALITY: Composite atomic entry score
                if _a_eq > 70:
                    atomic_score += 10
                    atomic_reasons_for.append(f"🔬 High entry quality ({_a_eq:.0f}/100)")
                elif _a_eq > 50:
                    atomic_score += 5
                elif _a_eq < 25:
                    atomic_score -= 10
                    atomic_reasons_against.append(f"🔬 Poor entry quality ({_a_eq:.0f}/100)")
                
                # D) REVERSAL PROBABILITY: High reversal = danger
                if _a_rp > 60:
                    atomic_score -= 12
                    atomic_reasons_against.append(f"🔬 High reversal probability ({_a_rp:.0f}%)")
                elif _a_rp > 40:
                    atomic_score -= 5
                elif _a_rp < 20:
                    atomic_score += 5
                    atomic_reasons_for.append(f"🔬 Low reversal risk ({_a_rp:.0f}%)")
                
                # E) MOMENTUM DECAY: Fading momentum = exhaustion
                if _a_mdecay < -0.02:
                    atomic_score -= 8
                    atomic_reasons_against.append(f"🔬 Momentum fading (decay={_a_mdecay:.4f})")
                elif _a_mdecay > 0.02:
                    atomic_score += 5
                    atomic_reasons_for.append(f"🔬 Momentum building (decay={_a_mdecay:+.4f})")
                
                # F) MICRO-HURST: Trending vs mean-reverting at tick level
                if _a_hurst > 0.6:
                    # Trending at tick level — good for momentum trades
                    if vel_aligns:
                        atomic_score += 5
                        atomic_reasons_for.append(f"🔬 Tick-level trending (H={_a_hurst:.2f})")
                elif _a_hurst < 0.4:
                    # Mean-reverting at tick level — momentum may snap back
                    atomic_score -= 5
                    atomic_reasons_against.append(f"🔬 Tick-level mean-reverting (H={_a_hurst:.2f})")
                
                # G) CHOPPINESS: Too many swings = unreliable signal
                if _a_sc > 8:
                    atomic_score -= 8
                    atomic_reasons_against.append(f"🔬 Choppy ({_a_sc} swings)")
                elif _a_sc <= 3:
                    atomic_score += 5
                    atomic_reasons_for.append(f"🔬 Clean trend ({_a_sc} swings)")
                
                # Clamp to valid range
                scores['atomic'] = max(0, min(100, atomic_score))
                
                # Store details and add reasons
                detailed_analysis['atomic_score'] = scores['atomic']
                detailed_analysis['atomic_velocity'] = _a_vel
                detailed_analysis['atomic_entry_quality'] = _a_eq
                detailed_analysis['atomic_reversal_prob'] = _a_rp
                detailed_analysis['atomic_path_efficiency'] = _a_peff
                
                reasons_for.extend(atomic_reasons_for)
                reasons_against.extend(atomic_reasons_against)
                
                logger.info(f"🔬 ADVANCED MATH: score={scores['atomic']:.0f} | vel={_a_vel:+.5f} | eq={_a_eq:.0f} | rev={_a_rp:.0f}% | peff={_a_peff:.2f} | hurst={_a_hurst:.2f}")
            except Exception as _atom_math_err:
                logger.debug(f"🔬 Atomic math scoring error: {_atom_math_err}")
                scores['atomic'] = 40  # Slight penalty on error (not neutral)
        else:
            scores['atomic'] = 30  # PENALTY: No atomic data = no real-time tick insight = elevated risk
            reasons_against.append("🔬 No atomic tick data (blind entry)")
        
        # ═══════════════════════════════════════════════════════════════════
        # FINAL BAYESIAN-WEIGHTED SCORE CALCULATION WITH CONFIDENCE INTERVALS
        # ═══════════════════════════════════════════════════════════════════
        # Weights are now LEARNED from trade outcomes via Dirichlet-Multinomial model.
        # After each trade close, we update based on which components predicted correctly.
        # get_weights() returns the posterior mean: αᵢ / Σαⱼ (always sums to 1.0)
        # === COMPONENT SCORE DEBUG LOG (helps diagnose direction bias) ===
        _dir_label = "LONG" if signal == 1 else "SHORT"
        _symbol = context.get('symbol', '???') if context else '???'
        _score_parts = {k: f"{scores.get(k, 50):.0f}" for k in weights.keys()}
        logger.info(f"📊 COMPONENTS [{_symbol} {_dir_label}]: {_score_parts}")
        
        # Calculate weighted score
        final_score = sum(scores.get(k, 50) * w for k, w in weights.items())
        
        # ═══════════════════════════════════════════════════════════════════
        # BAYESIAN CONVICTION BOOST (Rec #1/#3 integration)
        # When the components that carry the highest Bayesian weight all
        # agree (score > 70), it means the "most trusted" signals are
        # aligned. Give a confidence bonus BEFORE penalties hit, so that
        # a broad consensus of strong signals can survive the penalty gauntlet.
        # This makes Bayesian weights actively influence pass/fail decisions.
        # ═══════════════════════════════════════════════════════════════════
        try:
            HIGH_SCORE_THRESHOLD = 70
            MIN_HIGH_WEIGHT = 0.06  # Components with >= 6% weight are "important"
            high_conviction_count = 0
            high_conviction_total = 0.0
            for comp, w in weights.items():
                if w >= MIN_HIGH_WEIGHT and scores.get(comp, 50) >= HIGH_SCORE_THRESHOLD:
                    high_conviction_count += 1
                    high_conviction_total += w
            
            conviction_boost = 0
            if high_conviction_count >= 5:
                conviction_boost = 12  # 5+ high-weight components agree = strong consensus
            elif high_conviction_count >= 4:
                conviction_boost = 8
            elif high_conviction_count >= 3:
                conviction_boost = 5
            
            if conviction_boost > 0:
                final_score = min(100, final_score + conviction_boost)
                detailed_analysis['bayesian_conviction_boost'] = conviction_boost
                detailed_analysis['high_conviction_components'] = high_conviction_count
                logger.info(f"📊 BAYESIAN CONVICTION: {high_conviction_count} high-weight components agree → +{conviction_boost} boost (score: {final_score - conviction_boost:.0f}→{final_score:.0f})")
        except Exception as bc_err:
            logger.debug(f"Bayesian conviction boost error: {bc_err}")
        
        # ═══════════════════════════════════════════════════════════════════
        # APPLY SOFT PENALTIES FROM RSI AND TREND CHECKS
        # These replace the old hard blocks - penalties reduce score but don't 
        # block. AI + Math can now work together on final decision.
        # ═══════════════════════════════════════════════════════════════════
        soft_penalty = detailed_analysis.get('total_soft_penalty', 0)
        if soft_penalty != 0:
            pre_penalty_score = final_score
            final_score = max(0, final_score + soft_penalty)  # soft_penalty is negative
            logger.debug(f"📊 Score after soft penalties: {pre_penalty_score:.0f} → {final_score:.0f} (penalty: {soft_penalty})")
        
        # Track pre-penalty score to cap cumulative penalty impact
        pre_individual_penalty_score = final_score
        _mt = self._get_market_hours_context()
        MAX_CUMULATIVE_PENALTY_PCT = 0.45 if _mt.get('is_weekend', False) else 0.60
        
        # ═══════════════════════════════════════════════════════════════════
        # REGIME-AWARE PENALTY CAP (Rec #4 integration)
        # In strong trending regimes, trend-following trades naturally accumulate
        # counter-trend penalties (MTF, mean-reversion indicators, etc.) but the
        # trend itself IS the edge. Reduce the penalty cap so trend signals survive.
        # ═══════════════════════════════════════════════════════════════════
        _regime_adx = context.get('adx', 25) if context else 25
        _regime_roc5 = context.get('roc_5', 0) if context else 0
        _regime_trend_str = detailed_analysis.get('trend_strength', 0)
        # Detect trending regime: ADX > 30 AND momentum aligns with trade direction
        _momentum_aligns_trade = (
            (signal == 1 and _regime_roc5 > 0.3) or
            (signal == -1 and _regime_roc5 < -0.3)
        )
        if _regime_adx >= 30 and _momentum_aligns_trade and _regime_trend_str > 0.5:
            # Strong trend with aligned momentum → softer penalty cap
            MAX_CUMULATIVE_PENALTY_PCT = min(MAX_CUMULATIVE_PENALTY_PCT, 0.40)
            detailed_analysis['regime_penalty_reduction'] = True
            logger.info(f"📊 REGIME BOOST: Trending regime (ADX={_regime_adx:.0f}, ROC5={_regime_roc5:+.2f}%) → penalty cap reduced to {MAX_CUMULATIVE_PENALTY_PCT:.0%}")
        elif _regime_adx >= 25 and _momentum_aligns_trade:
            # Moderate trend → slightly softer cap
            MAX_CUMULATIVE_PENALTY_PCT = min(MAX_CUMULATIVE_PENALTY_PCT, 0.50)
            detailed_analysis['regime_penalty_reduction'] = True
        
        # ═══════════════════════════════════════════════════════════════════
        # LOCAL PEAK/TROUGH CHECK - DON'T ENTER AT TOPS/BOTTOMS!
        # This is the #1 cause of immediate reversals after entry
        # FIX: Make trend-aware. In a strong downtrend, price at the bottom
        # of the range is NORMAL — don't penalize SHORT. Same for LONG in uptrend.
        # ═══════════════════════════════════════════════════════════════════
        local_peak_penalty = 0
        _peak_testing = get_threshold_mode() == 'testing'
        if len(closes) >= 10:
            current = closes.iloc[-1]
            recent_high = closes.tail(10).max()
            recent_low = closes.tail(10).min()
            price_range = recent_high - recent_low if recent_high > recent_low else 0.001
            
            # Testing mode: relax peak/trough thresholds and reduce penalties
            _peak_severe = 0.97 if _peak_testing else 0.90
            _peak_moderate = 0.90 if _peak_testing else 0.80
            _peak_severe_pen = -20 if _peak_testing else -35
            _peak_moderate_pen = -10 if _peak_testing else -20
            
            # Detect trend direction from regression slope (calculated earlier)
            _trend_slope = detailed_analysis.get('regression_slope', 0)
            _trend_r2 = detailed_analysis.get('r_squared', 0)
            _strong_uptrend = _trend_slope > 0 and _trend_r2 > 0.3
            _strong_downtrend = _trend_slope < 0 and _trend_r2 > 0.3
            
            # For LONG: Check if we're buying near local high
            if signal == 1:
                pct_in_range = (current - recent_low) / price_range
                # In a strong uptrend, being at the top is normal — reduce penalty
                if _strong_uptrend:
                    _peak_severe_pen = max(_peak_severe_pen // 2, -10)
                    _peak_moderate_pen = max(_peak_moderate_pen // 2, -5)
                    
                if pct_in_range >= _peak_severe:
                    local_peak_penalty = _peak_severe_pen
                    reasons_against.append(f"🚨 BUYING AT LOCAL TOP: Price at {pct_in_range:.0%} of 10-bar range!")
                    logger.warning(f"📊 LOCAL PEAK PENALTY: LONG at {pct_in_range:.0%} of range → {_peak_severe_pen}")
                elif pct_in_range >= _peak_moderate:
                    local_peak_penalty = _peak_moderate_pen
                    reasons_against.append(f"⚠️ Near local high: Price at {pct_in_range:.0%} of 10-bar range")
                    logger.debug(f"📊 LOCAL PEAK PENALTY: LONG at {pct_in_range:.0%} of range → {_peak_moderate_pen}")
                elif pct_in_range >= 0.70 and not _strong_uptrend:
                    local_peak_penalty = -10
                    reasons_against.append(f"⚠️ Upper range: Price at {pct_in_range:.0%} of 10-bar range")
            
            # For SHORT: Check if we're selling near local low
            elif signal == -1:
                pct_in_range = (current - recent_low) / price_range
                _trough_severe = 0.03 if _peak_testing else 0.10
                _trough_moderate = 0.10 if _peak_testing else 0.20
                
                # In a strong downtrend, being at the bottom is NORMAL — reduce/skip penalty
                if _strong_downtrend:
                    _peak_severe_pen = max(_peak_severe_pen // 2, -10)
                    _peak_moderate_pen = max(_peak_moderate_pen // 2, -5)
                    logger.debug(f"📊 TROUGH PENALTY REDUCED: Strong downtrend (slope={_trend_slope:.6f}, R²={_trend_r2:.2f})")
                    
                if pct_in_range <= _trough_severe:
                    local_peak_penalty = _peak_severe_pen
                    reasons_against.append(f"🚨 SELLING AT LOCAL BOTTOM: Price at {pct_in_range:.0%} of 10-bar range!")
                    logger.warning(f"📊 LOCAL TROUGH PENALTY: SHORT at {pct_in_range:.0%} of range → {_peak_severe_pen}")
                elif pct_in_range <= _trough_moderate:
                    local_peak_penalty = _peak_moderate_pen
                    reasons_against.append(f"⚠️ Near local low: Price at {pct_in_range:.0%} of 10-bar range")
                    logger.debug(f"📊 LOCAL TROUGH PENALTY: SHORT at {pct_in_range:.0%} of range → {_peak_moderate_pen}")
                elif pct_in_range <= 0.30 and not _strong_downtrend:
                    local_peak_penalty = -10
                    reasons_against.append(f"⚠️ Lower range: Price at {pct_in_range:.0%} of 10-bar range")
            
            if local_peak_penalty != 0:
                final_score = max(0, final_score + local_peak_penalty)
                detailed_analysis['local_peak_penalty'] = local_peak_penalty
        
        # ═══════════════════════════════════════════════════════════════════
        # RECENT SPIKE CHECK - DON'T CHASE AFTER RAPID MOVES!
        # If price just spiked, it will likely pull back - wait for pullback
        # HARD BLOCK for extreme spikes (>3%), penalty for smaller ones
        # Relaxed thresholds for volatile crypto markets
        # ═══════════════════════════════════════════════════════════════════
        spike_penalty = 0
        spike_hard_block = False
        _spike_testing = get_threshold_mode() == 'testing'
        if len(closes) >= 5:
            last_3_bars_change = (closes.iloc[-1] - closes.iloc[-4]) / closes.iloc[-4] * 100
            last_5_bars_change = (closes.iloc[-1] - closes.iloc[-6]) / closes.iloc[-6] * 100 if len(closes) >= 6 else last_3_bars_change
            
            # Testing mode: relax spike thresholds (5→8% hard block, 3→5% penalty)
            _spike_hard = 8.0 if _spike_testing else 5.0
            _spike_penalty_thr = 5.0 if _spike_testing else 3.0
            _spike_minor_thr = 3.0 if _spike_testing else 2.0
            
            # For LONG: Penalize/Block if price just spiked UP (chasing)
            if signal == 1:
                if last_3_bars_change >= _spike_hard:  # HARD BLOCK
                    spike_hard_block = True
                    reasons_against.append(f"🚫 SPIKE HARD BLOCK: +{last_3_bars_change:.2f}% in 3 bars - too risky!")
                    logger.warning(f"🚫 SPIKE HARD BLOCK: LONG after +{last_3_bars_change:.2f}% spike - BLOCKED")
                elif last_3_bars_change >= _spike_penalty_thr:  # spike penalty
                    spike_penalty = -20
                    reasons_against.append(f"🚨 CHASING SPIKE: +{last_3_bars_change:.2f}% in last 3 bars - wait for pullback!")
                    logger.warning(f"📊 SPIKE PENALTY: LONG after +{last_3_bars_change:.2f}% spike → -20")
                elif last_3_bars_change >= _spike_minor_thr:  # minor rise
                    spike_penalty = -10
                    reasons_against.append(f"⚠️ Recent rise: +{last_3_bars_change:.2f}% in 3 bars")
                elif last_5_bars_change >= 4.0:  # >4% rise in 5 bars
                    spike_penalty = -5
                    reasons_against.append(f"⚠️ Extended rise: +{last_5_bars_change:.2f}% in 5 bars")
            
            # For SHORT: Penalize/Block if price just spiked DOWN (chasing) or UP (reversal risk)
            elif signal == -1:
                # HARD BLOCK if price just spiked UP - shorting into strength is dangerous!
                if last_3_bars_change >= _spike_hard:  # Price rose significantly
                    spike_hard_block = True
                    reasons_against.append(f"🚫 SPIKE HARD BLOCK: Price rose +{last_3_bars_change:.2f}% - can't SHORT into strength!")
                    logger.warning(f"🚫 SPIKE HARD BLOCK: SHORT blocked - price just rose +{last_3_bars_change:.2f}%")
                elif last_3_bars_change <= -_spike_hard:  # Big drop = HARD BLOCK
                    spike_hard_block = True
                    reasons_against.append(f"🚫 SPIKE HARD BLOCK: {last_3_bars_change:.2f}% dump - too late to chase!")
                    logger.warning(f"🚫 SPIKE HARD BLOCK: SHORT after {last_3_bars_change:.2f}% dump - BLOCKED")
                elif last_3_bars_change <= -2.0:  # >2% drop in 3 bars = spike penalty
                    spike_penalty = -20
                    reasons_against.append(f"🚨 CHASING DUMP: {last_3_bars_change:.2f}% in last 3 bars - wait for bounce!")
                    logger.warning(f"📊 SPIKE PENALTY: SHORT after {last_3_bars_change:.2f}% dump → -20")
                elif last_3_bars_change <= -1.5:  # >1.5% drop
                    spike_penalty = -10
                    reasons_against.append(f"⚠️ Recent drop: {last_3_bars_change:.2f}% in 3 bars")
                elif last_5_bars_change <= -2.5:  # >2.5% drop in 5 bars
                    spike_penalty = -5
                    reasons_against.append(f"⚠️ Extended drop: {last_5_bars_change:.2f}% in 5 bars")
            
            # Apply spike hard block
            if spike_hard_block:
                final_score = 0
                logger.info(f"📊 SPIKE HARD BLOCK: Score set to 0 (3-bar change={last_3_bars_change:+.2f}%)")
            elif spike_penalty != 0:
                final_score = max(0, final_score + spike_penalty)
                detailed_analysis['spike_penalty'] = spike_penalty
        
        # ═══════════════════════════════════════════════════════════════════
        # VOLUME CONFIRMATION CHECK - DON'T ENTER ON WEAK VOLUME!
        # Good entries should have above-average volume confirming the move
        # ═══════════════════════════════════════════════════════════════════
        volume_penalty = 0
        volume_ratio_check = context.get('volume_ratio', 1.0) if context else 1.0
        
        # Require minimum volume for entry
        MIN_VOLUME_RATIO = 0.8   # At least 80% of average volume
        GOOD_VOLUME_RATIO = 1.5  # 1.5x average = strong confirmation
        
        if volume_ratio_check < 0.5:
            # Very weak volume - likely false signal
            volume_penalty = -25
            reasons_against.append(f"🚨 WEAK VOLUME: Only {volume_ratio_check:.1f}x average - likely false signal!")
            logger.warning(f"📊 VOLUME PENALTY: Very low volume {volume_ratio_check:.1f}x → -25")
        elif volume_ratio_check < MIN_VOLUME_RATIO:
            # Below average volume - risky entry
            volume_penalty = -15
            reasons_against.append(f"⚠️ Below avg volume: {volume_ratio_check:.1f}x average")
            logger.debug(f"📊 VOLUME PENALTY: Low volume {volume_ratio_check:.1f}x → -15")
        elif volume_ratio_check >= GOOD_VOLUME_RATIO:
            # Strong volume confirmation - reduce other penalties slightly
            volume_bonus = 10
            reasons_for.append(f"📊 Strong volume: {volume_ratio_check:.1f}x average")
            final_score = min(100, final_score + volume_bonus)
            logger.debug(f"📊 VOLUME BONUS: High volume {volume_ratio_check:.1f}x → +10")
        
        if volume_penalty != 0:
            final_score = max(0, final_score + volume_penalty)
            detailed_analysis['volume_penalty'] = volume_penalty
        
        # ═══════════════════════════════════════════════════════════════════
        # ADX RANGING MARKET FILTER - DON'T TRADE IN CHOPPY MARKETS!
        # ADX < 20 = weak trend (ranging) = avoid trading
        # ADX 20-25 = trend developing = caution
        # ADX > 25 = strong trend = good for directional trades
        # ═══════════════════════════════════════════════════════════════════
        adx_value = context.get('adx', 25) if context else 25
        adx_penalty = 0
        
        if adx_value < 15:
            # Very weak trend - ranging market, avoid
            adx_penalty = -35
            reasons_against.append(f"🚫 RANGING MARKET: ADX={adx_value:.0f} (<15) - No clear trend!")
            logger.warning(f"📊 ADX PENALTY: Ranging market ADX={adx_value:.0f} → -35")
        elif adx_value < 20:
            # Weak trend - likely choppy
            adx_penalty = -20
            reasons_against.append(f"⚠️ Weak trend: ADX={adx_value:.0f} (<20) - Choppy conditions")
            logger.debug(f"📊 ADX PENALTY: Weak trend ADX={adx_value:.0f} → -20")
        elif adx_value < 25:
            # Developing trend - caution
            adx_penalty = -10
            reasons_against.append(f"⚠️ Developing trend: ADX={adx_value:.0f} - May not continue")
        elif adx_value >= 35:
            # Strong trend - bonus
            adx_bonus = 10
            reasons_for.append(f"📈 Strong trend: ADX={adx_value:.0f} - Clear direction")
            final_score = min(100, final_score + adx_bonus)
        
        if adx_penalty != 0:
            final_score = max(0, final_score + adx_penalty)
            detailed_analysis['adx_penalty'] = adx_penalty
        
        # ═══════════════════════════════════════════════════════════════════
        # INSTITUTIONAL MICROSTRUCTURE — Real exchange data integration
        # Replaces candle-based order flow estimation with REAL data:
        # - Order book depth & imbalance (actual bid/ask walls)
        # - Trade tape CVD (real buy/sell aggressor from exchange)
        # - Funding rate (crowded positioning signal)
        # Falls back to candle-based estimation if institutional data unavailable
        # ═══════════════════════════════════════════════════════════════════
        order_flow_penalty = 0
        inst_data = context.get('institutional_data', {})
        
        if inst_data and inst_data.get('composite_bias') != 'neutral':
            # === USE REAL INSTITUTIONAL DATA ===
            try:
                # 1. Order book imbalance (real depth)
                ob = inst_data.get('orderbook', {})
                ob_imbalance = ob.get('imbalance', 0)
                ob_strength = ob.get('signal_strength', 0)
                
                if abs(ob_imbalance) > 0.15:
                    detailed_analysis['order_flow_imbalance'] = ob_imbalance
                    detailed_analysis['order_flow_source'] = 'exchange_orderbook'
                    detailed_analysis['bid_depth_usd'] = ob.get('bid_depth_usd', 0)
                    detailed_analysis['ask_depth_usd'] = ob.get('ask_depth_usd', 0)
                    detailed_analysis['spread_pct'] = ob.get('spread_pct', 0)
                
                # 2. Trade tape CVD (real buy/sell from exchange)
                tape = inst_data.get('tape', {})
                buy_pct = tape.get('buy_pct', 50)
                cvd = tape.get('cvd', 0)
                
                detailed_analysis['buy_pressure'] = buy_pct / 100
                detailed_analysis['sell_pressure'] = 1 - (buy_pct / 100)
                detailed_analysis['cvd'] = cvd
                detailed_analysis['large_trade_count'] = tape.get('large_trade_count', 0)
                detailed_analysis['large_trade_bias'] = tape.get('large_trade_bias', 'balanced')
                
                # 3. Funding rate (crowded positioning)
                funding = inst_data.get('funding', {})
                detailed_analysis['funding_rate'] = funding.get('current_rate', 0)
                detailed_analysis['funding_annualized'] = funding.get('current_rate_annualized', 0)
                detailed_analysis['funding_extreme'] = funding.get('extreme', False)
                detailed_analysis['funding_trend'] = funding.get('trend', 'stable')
                
                # 4. Open interest
                oi = inst_data.get('open_interest', {})
                detailed_analysis['oi_change_pct'] = oi.get('oi_change_pct', 0)
                
                # Combined institutional penalty/boost
                inst_penalty = inst_data.get('penalty', 0)
                inst_bias = inst_data.get('composite_bias', 'neutral')
                
                if inst_bias == 'contradicts':
                    order_flow_penalty = inst_penalty  # Negative penalty
                    reasons_against.append(
                        f"📊 Institutional data CONTRADICTS signal: "
                        f"OB imbalance={ob_imbalance:+.2f}, CVD buy%={buy_pct:.0f}%, "
                        f"FR={funding.get('current_rate', 0)*100:.4f}%"
                    )
                    logger.debug(f"📊 INSTITUTIONAL: Contradicts → {inst_penalty}")
                elif inst_bias == 'confirms':
                    order_flow_penalty = inst_penalty  # Positive boost
                    reasons_for.append(
                        f"📊 Institutional data CONFIRMS signal: "
                        f"OB imbalance={ob_imbalance:+.2f}, CVD buy%={buy_pct:.0f}%, "
                        f"FR={funding.get('current_rate', 0)*100:.4f}%"
                    )
                    logger.debug(f"📊 INSTITUTIONAL: Confirms → +{inst_penalty}")
                
                # Extreme funding rate warning (always flag regardless of direction)
                if funding.get('extreme'):
                    reasons_against.append(
                        f"⚠️ EXTREME funding rate: {funding.get('current_rate', 0)*100:.4f}% "
                        f"(annualized {funding.get('current_rate_annualized', 0):.0f}%) — crowded positioning"
                    )
                    
            except Exception as e:
                logger.debug(f"Institutional data integration error: {e}")
        else:
            # === FALLBACK: Candle-based order flow estimation ===
            if len(df) >= 10:
                try:
                    recent_df = df.tail(10)
                    up_volume = 0
                    down_volume = 0
                    
                    for i in range(1, len(recent_df)):
                        price_change = recent_df['close'].iloc[i] - recent_df['close'].iloc[i-1]
                        vol = recent_df['volume'].iloc[i] if 'volume' in recent_df.columns else 1
                        if price_change > 0:
                            up_volume += vol
                        else:
                            down_volume += vol
                    
                    total_volume = up_volume + down_volume
                    if total_volume > 0:
                        buy_pressure = up_volume / total_volume
                        sell_pressure = down_volume / total_volume
                        imbalance = buy_pressure - sell_pressure
                        
                        detailed_analysis['order_flow_imbalance'] = imbalance
                        detailed_analysis['buy_pressure'] = buy_pressure
                        detailed_analysis['sell_pressure'] = sell_pressure
                        detailed_analysis['order_flow_source'] = 'candle_estimate'
                        
                        if signal == 1 and imbalance < -0.3:
                            order_flow_penalty = -20
                            reasons_against.append(f"🔻 Order flow against LONG: {sell_pressure:.0%} selling pressure (candle est.)")
                        elif signal == -1 and imbalance > 0.3:
                            order_flow_penalty = -20
                            reasons_against.append(f"🔺 Order flow against SHORT: {buy_pressure:.0%} buying pressure (candle est.)")
                        elif signal == 1 and imbalance > 0.3:
                            reasons_for.append(f"🔺 Order flow supports LONG: {buy_pressure:.0%} buying pressure (candle est.)")
                        elif signal == -1 and imbalance < -0.3:
                            reasons_for.append(f"🔻 Order flow supports SHORT: {sell_pressure:.0%} selling pressure (candle est.)")
                except Exception as e:
                    logger.debug(f"Order flow fallback calculation error: {e}")
        
        if order_flow_penalty != 0:
            final_score = max(0, final_score + order_flow_penalty)
            detailed_analysis['order_flow_penalty'] = order_flow_penalty
        
        # ═══════════════════════════════════════════════════════════════════
        # VWAP BANDS - Better entry zones around volume-weighted average
        # Entry above upper band = overextended, entry below lower = oversold
        # ═══════════════════════════════════════════════════════════════════
        vwap_penalty = 0
        if len(df) >= 20 and 'volume' in df.columns:
            try:
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                volumes = df['volume']
                vwap = (typical_price * volumes).sum() / volumes.sum()
                
                # Calculate VWAP standard deviation for bands
                vwap_std = ((typical_price - vwap) ** 2 * volumes).sum() / volumes.sum()
                vwap_std = vwap_std ** 0.5 if vwap_std > 0 else current_price * 0.01
                
                upper_band = vwap + 2 * vwap_std
                lower_band = vwap - 2 * vwap_std
                
                detailed_analysis['vwap'] = vwap
                detailed_analysis['vwap_upper'] = upper_band
                detailed_analysis['vwap_lower'] = lower_band
                
                # Check position relative to VWAP bands
                if signal == 1:  # LONG
                    if current_price > upper_band:
                        vwap_penalty = -25
                        pct_above = (current_price - upper_band) / vwap * 100
                        reasons_against.append(f"🚨 Above VWAP upper band: +{pct_above:.1f}% overextended")
                        logger.warning(f"📊 VWAP: LONG above upper band → -25")
                    elif current_price < lower_band:
                        # Good for LONG - below lower band
                        reasons_for.append(f"✅ Below VWAP lower band: Good entry for LONG")
                    elif current_price < vwap:
                        # Below VWAP - decent entry
                        reasons_for.append(f"✅ Below VWAP: Reasonable entry zone")
                elif signal == -1:  # SHORT
                    if current_price < lower_band:
                        vwap_penalty = -25
                        pct_below = (lower_band - current_price) / vwap * 100
                        reasons_against.append(f"🚨 Below VWAP lower band: -{pct_below:.1f}% overextended")
                        logger.warning(f"📊 VWAP: SHORT below lower band → -25")
                    elif current_price > upper_band:
                        # Good for SHORT - above upper band
                        reasons_for.append(f"✅ Above VWAP upper band: Good entry for SHORT")
                    elif current_price > vwap:
                        # Above VWAP - decent entry
                        reasons_for.append(f"✅ Above VWAP: Reasonable entry zone")
            except Exception as e:
                logger.debug(f"VWAP bands calculation error: {e}")
        
        if vwap_penalty != 0:
            final_score = max(0, final_score + vwap_penalty)
            detailed_analysis['vwap_penalty'] = vwap_penalty
        
        # ═══════════════════════════════════════════════════════════════════
        # SESSION-BASED FILTERING - Different behavior for different sessions
        # ═══════════════════════════════════════════════════════════════════
        session_penalty = 0
        market_timing = self._get_market_hours_context()
        current_session = market_timing.get('session', 'Unknown')
        is_weekend = market_timing.get('is_weekend', False)
        activity_level = market_timing.get('activity_level', 'moderate')
        
        # Weekend: NO session penalty — crypto trades 24/7
        # The penalty was stacking with extreme fear + other filters, killing all trades
        if is_weekend:
            session_penalty = 0  # Crypto is 24/7, weekend is not a penalty
            logger.debug(f"📊 SESSION: Weekend → no penalty (crypto 24/7)")
        
        # Low activity session penalty
        elif activity_level == 'low':
            session_penalty = -10
            reasons_against.append(f"⚠️ Low activity session: {current_session}")
        
        # High activity bonus
        elif activity_level == 'high' and current_session in ['Europe', 'US']:
            reasons_for.append(f"✅ High activity session: {current_session}")
        
        if session_penalty != 0:
            final_score = max(0, final_score + session_penalty)
            detailed_analysis['session_penalty'] = session_penalty
        
        # ═══════════════════════════════════════════════════════════════════
        # LIQUIDITY ZONE DETECTION - Avoid entries near major liquidity pools
        # Price tends to wick through these zones before reversing
        # ═══════════════════════════════════════════════════════════════════
        liquidity_penalty = 0
        if len(df) >= 50:
            try:
                # Find high volume price levels (liquidity zones)
                high_24 = df['high'].tail(96).max()  # 24h high (assuming 15m bars)
                low_24 = df['low'].tail(96).min()   # 24h low
                
                # Round numbers are often liquidity zones
                round_levels = []
                price_magnitude = 10 ** (len(str(int(current_price))) - 2)  # Get order of magnitude
                base = int(current_price / price_magnitude) * price_magnitude
                for mult in range(-2, 3):
                    round_levels.append(base + mult * price_magnitude)
                
                # Check if price is very close to 24h extremes (liquidity zones)
                dist_to_high = abs(current_price - high_24) / current_price * 100
                dist_to_low = abs(current_price - low_24) / current_price * 100
                
                # Check if near round number
                min_round_dist = min(abs(current_price - lvl) / current_price * 100 for lvl in round_levels)
                
                if signal == 1:  # LONG
                    # LONG near 24h high is risky (resistance)
                    if dist_to_high < 0.3:
                        liquidity_penalty = -20
                        reasons_against.append(f"🚫 LONG near 24h HIGH: Only {dist_to_high:.2f}% away")
                        logger.debug(f"📊 LIQUIDITY: Near 24h high → -20")
                elif signal == -1:  # SHORT
                    # SHORT near 24h low is risky (support)
                    if dist_to_low < 0.3:
                        liquidity_penalty = -20
                        reasons_against.append(f"🚫 SHORT near 24h LOW: Only {dist_to_low:.2f}% away")
                        logger.debug(f"📊 LIQUIDITY: Near 24h low → -20")
                
                # Near round number - added uncertainty
                if min_round_dist < 0.2:
                    closest_round = min(round_levels, key=lambda x: abs(x - current_price))
                    liquidity_penalty = min(liquidity_penalty - 10, -10)
                    reasons_against.append(f"⚠️ Near round number ${closest_round:.0f}: Liquidity zone")
                
                detailed_analysis['dist_to_24h_high'] = dist_to_high
                detailed_analysis['dist_to_24h_low'] = dist_to_low
            except Exception as e:
                logger.debug(f"Liquidity zone calculation error: {e}")
        
        if liquidity_penalty != 0:
            final_score = max(0, final_score + liquidity_penalty)
            detailed_analysis['liquidity_penalty'] = liquidity_penalty
        
        # ═══════════════════════════════════════════════════════════════════
        # MULTI-TIMEFRAME CONFIRMATION - Check higher timeframe trend alignment
        # Simulate 1H and 4H trends from 15m data for trend alignment
        # Counter-trend trades on higher timeframes are riskier
        # ═══════════════════════════════════════════════════════════════════
        mtf_penalty = 0
        if len(df) >= 100:  # Need at least 100 bars for 4H simulation
            try:
                # Simulate 1H trend (4 x 15m bars)
                closes_1h = df['close'].iloc[::4].tail(24)  # Last 24 hours of 1H data
                if len(closes_1h) >= 10:
                    sma_10_1h = closes_1h.rolling(10).mean().iloc[-1]
                    sma_20_1h = closes_1h.rolling(20).mean().iloc[-1] if len(closes_1h) >= 20 else sma_10_1h
                    trend_1h = 'bullish' if sma_10_1h > sma_20_1h else 'bearish'
                    
                    # Check 1H trend alignment
                    mtf_1h_penalty = -10 if is_weekend else -15
                    if signal == 1 and trend_1h == 'bearish':
                        mtf_penalty += mtf_1h_penalty
                        reasons_against.append(f"⚠️ LONG against 1H trend (bearish)")
                    elif signal == -1 and trend_1h == 'bullish':
                        mtf_penalty += mtf_1h_penalty
                        reasons_against.append(f"⚠️ SHORT against 1H trend (bullish)")
                    elif signal == 1 and trend_1h == 'bullish':
                        reasons_for.append(f"✅ 1H trend confirms LONG")
                    elif signal == -1 and trend_1h == 'bearish':
                        reasons_for.append(f"✅ 1H trend confirms SHORT")
                    
                    detailed_analysis['trend_1h'] = trend_1h
                
                # Simulate 4H trend (16 x 15m bars)
                closes_4h = df['close'].iloc[::16].tail(24)  # ~4 days of 4H data
                if len(closes_4h) >= 6:
                    sma_5_4h = closes_4h.rolling(5).mean().iloc[-1]
                    sma_10_4h = closes_4h.rolling(10).mean().iloc[-1] if len(closes_4h) >= 10 else sma_5_4h
                    trend_4h = 'bullish' if sma_5_4h > sma_10_4h else 'bearish'
                    
                    # 4H counter-trend is riskier but don't over-penalize
                    mtf_4h_penalty = -12 if is_weekend else -20
                    if signal == 1 and trend_4h == 'bearish':
                        mtf_penalty += mtf_4h_penalty
                        reasons_against.append(f"🚫 LONG against 4H trend (bearish) - HIGH RISK")
                        logger.debug(f"📊 MTF: LONG against 4H bearish → {mtf_4h_penalty}")
                    elif signal == -1 and trend_4h == 'bullish':
                        mtf_penalty += mtf_4h_penalty
                        reasons_against.append(f"🚫 SHORT against 4H trend (bullish) - HIGH RISK")
                        logger.debug(f"📊 MTF: SHORT against 4H bullish → {mtf_4h_penalty}")
                    elif signal == 1 and trend_4h == 'bullish':
                        reasons_for.append(f"✅ 4H trend confirms LONG - Strong setup")
                    elif signal == -1 and trend_4h == 'bearish':
                        reasons_for.append(f"✅ 4H trend confirms SHORT - Strong setup")
                    
                    detailed_analysis['trend_4h'] = trend_4h
                    
            except Exception as e:
                logger.debug(f"Multi-timeframe analysis error: {e}")
        
        if mtf_penalty != 0:
            final_score = max(0, final_score + mtf_penalty)
            detailed_analysis['mtf_penalty'] = mtf_penalty
        
        # ═══════════════════════════════════════════════════════════════════
        # MOMENTUM DIRECTION CHECK - HARD BLOCK!
        # This is the #1 cause of losses - entering against momentum
        # ═══════════════════════════════════════════════════════════════════
        roc_5 = context.get('roc_5', 0) if context else 0
        roc_10 = context.get('roc_10', 0) if context else 0
        kalman_momentum = context.get('kalman_momentum', 0) if context else 0
        
        # Get Fear & Greed for stricter LONG filtering in bearish markets
        news_ctx = self._get_news_context()
        fear_greed = news_ctx.get('fear_greed_index', 50)
        
        momentum_direction_penalty = 0
        momentum_hard_block = False
        _mom_testing_mode = get_threshold_mode() == 'testing'
        
        if signal == 1:  # LONG - need POSITIVE momentum (or at least not against us)
            if fear_greed < 10:
                _is_wk = self._get_market_hours_context().get('is_weekend', False)
                if roc_5 < -1.0 and roc_10 < -0.5:  # BOTH must be strongly negative
                    if _mom_testing_mode:  # Testing: penalty not hard block
                        momentum_direction_penalty = -15
                        reasons_against.append(f"⚠️ LONG in EXTREME FEAR ({fear_greed}) with falling price (ROC5={roc_5:+.2f}%)")
                        logger.warning(f"⚠️ EXTREME FEAR penalty -15 (testing mode) (ROC5={roc_5:+.2f}%)")
                    elif not _is_wk:  # Weekday: HARD BLOCK
                        momentum_hard_block = True
                        reasons_against.append(f"🚫 HARD BLOCK: LONG in EXTREME FEAR ({fear_greed}) with FALLING price (ROC5={roc_5:+.2f}%)")
                        logger.warning(f"🚫 EXTREME FEAR HARD BLOCK: LONG blocked ({fear_greed}) ROC5={roc_5:+.2f}%")
                    else:  # Weekend: penalty
                        momentum_direction_penalty = -15
                        reasons_against.append(f"⚠️ LONG in EXTREME FEAR ({fear_greed}) with falling price (ROC5={roc_5:+.2f}%)")
                        logger.warning(f"⚠️ EXTREME FEAR ({fear_greed}) LONG penalty -15 (weekend) (ROC5={roc_5:+.2f}%)")
                elif roc_5 < -0.3:  # Slight negative = lighter penalty
                    momentum_direction_penalty = -10
                    reasons_against.append(f"⚠️ LONG against momentum in EXTREME FEAR (ROC5={roc_5:+.2f}%)")
                    logger.warning(f"⚠️ MOMENTUM PENALTY: EXTREME FEAR ({fear_greed}) - LONG against momentum -10 (ROC5={roc_5:+.2f}%)")
            elif roc_5 < -0.5 and roc_10 < -0.3:  # BOTH must be significantly negative
                if _mom_testing_mode:  # Testing: penalty instead of hard block
                    momentum_direction_penalty = -20
                    reasons_against.append(f"⚠️ LONG with falling price (ROC5={roc_5:+.2f}%, ROC10={roc_10:+.2f}%)")
                    logger.warning(f"⚠️ MOMENTUM PENALTY (testing): LONG ROC5={roc_5:+.2f}%, ROC10={roc_10:+.2f}%")
                else:
                    momentum_hard_block = True
                    reasons_against.append(f"🚫 HARD BLOCK: LONG with FALLING price (ROC5={roc_5:+.2f}%, ROC10={roc_10:+.2f}%)")
                    logger.warning(f"🚫 MOMENTUM HARD BLOCK: Can't LONG when ROC5={roc_5:+.2f}%, ROC10={roc_10:+.2f}%)")
            elif roc_5 < -0.3:  # Slight negative momentum = penalty (not hard block)
                momentum_direction_penalty = -20
                reasons_against.append(f"⚠️ LONG against momentum (ROC5={roc_5:+.2f}%)")
                
        elif signal == -1:  # SHORT - need NEGATIVE momentum (or at least not against us)
            if roc_5 > 1.5 and roc_10 > 1.0:  # BOTH must be significantly positive
                if _mom_testing_mode:  # Testing: penalty instead of hard block
                    momentum_direction_penalty = -20
                    reasons_against.append(f"⚠️ SHORT with rising price (ROC5={roc_5:+.2f}%, ROC10={roc_10:+.2f}%)")
                    logger.warning(f"⚠️ MOMENTUM PENALTY (testing): SHORT ROC5={roc_5:+.2f}%, ROC10={roc_10:+.2f}%")
                else:
                    momentum_hard_block = True
                    reasons_against.append(f"🚫 HARD BLOCK: SHORT with RISING price (ROC5={roc_5:+.2f}%, ROC10={roc_10:+.2f}%)")
                    logger.warning(f"🚫 MOMENTUM HARD BLOCK: Can't SHORT when ROC5={roc_5:+.2f}%, ROC10={roc_10:+.2f}%")
            elif roc_5 > 0.8:  # Moderate positive momentum = penalty (not hard block)
                momentum_direction_penalty = -15
                reasons_against.append(f"⚠️ SHORT against momentum (ROC5={roc_5:+.2f}%)")
        
        # If hard block triggered, set score to 0 to prevent entry
        if momentum_hard_block:
            final_score = 0
            logger.info(f"📊 MOMENTUM HARD BLOCK: Score set to 0 (ROC5={roc_5:+.2f}%, ROC10={roc_10:+.2f}%)")
        elif momentum_direction_penalty != 0:
            final_score = max(0, final_score + momentum_direction_penalty)
            logger.info(f"📊 Momentum direction penalty: {momentum_direction_penalty} (ROC5={roc_5:+.2f}%)")
        
        # ═══════════════════════════════════════════════════════════════════
        # HIGH VOLATILITY PENALTY (Added after SUI/XRP loss analysis)
        # When absolute volatility is high, direction is harder to predict
        # ═══════════════════════════════════════════════════════════════════
        garch_vol = detailed_analysis.get('garch_vol', 0)
        if garch_vol > 0.20:  # >20% annualized vol = very high
            vol_penalty = 15
            reasons_against.append(f"Very high volatility ({garch_vol:.1%}) - direction uncertain")
            final_score = max(0, final_score - vol_penalty)
            logger.debug(f"Applied -15 penalty for very high vol ({garch_vol:.1%})")
        elif garch_vol > 0.15:  # >15% = high
            vol_penalty = 8
            reasons_against.append(f"High volatility ({garch_vol:.1%}) - increased uncertainty")
            final_score = max(0, final_score - vol_penalty)
            logger.debug(f"Applied -8 penalty for high vol ({garch_vol:.1%})")
        
        # Also require STRONGER trend confirmation when volatility is high
        r_squared = detailed_analysis.get('r_squared', 0)
        if garch_vol > 0.15 and r_squared < 0.5:
            # High vol + weak trend = very uncertain direction
            weak_trend_penalty = 10
            reasons_against.append(f"Weak trend (R²={r_squared:.2f}) in high vol environment")
            final_score = max(0, final_score - weak_trend_penalty)
            logger.debug(f"Applied -10 penalty for weak trend in high vol")
        
        # PhD Enhancement: Calculate confidence intervals on final score
        try:
            from indicator import calculate_confidence_interval
            
            score_values = np.array(list(scores.values()))
            ci_result = calculate_confidence_interval(score_values, confidence=0.95)
            
            detailed_analysis['score_ci'] = ci_result
            detailed_analysis['score_ci_lower'] = ci_result.get('lower', final_score)
            detailed_analysis['score_ci_upper'] = ci_result.get('upper', final_score)
            
            # Confidence level based on CI width
            ci_width = ci_result.get('upper', final_score) - ci_result.get('lower', final_score)
            if ci_width < 20:
                confidence_level = 'very_high'
            elif ci_width < 40:
                confidence_level = 'high'
            elif ci_width < 60:
                confidence_level = 'medium'
            else:
                confidence_level = 'low'
        except Exception as e:
            logger.debug(f"Confidence interval calculation failed: {e}")
            score_values = np.array(list(scores.values()))
            score_variance = np.var(score_values) if score_values else 0
            confidence_penalty = min(10, score_variance / 50)
            
            final_score = max(0, final_score - confidence_penalty)
            confidence_level = 'high' if score_variance < 200 else 'medium' if score_variance < 400 else 'low'
            detailed_analysis['score_variance'] = score_variance
        
        detailed_analysis['component_scores'] = scores
        detailed_analysis['weights'] = weights
        detailed_analysis['confidence_level'] = confidence_level
        
        # ═══════════════════════════════════════════════════════════════════
        # INTELLIGENT SYSTEM MODIFIERS (Session, Symbol Memory)
        # Applied after all technical penalties but before the penalty cap
        # ═══════════════════════════════════════════════════════════════════
        intelligent_modifier = 0
        
        # System #2: Session/Hour modifier
        try:
            if hasattr(self, 'session_learner'):
                current_hour = datetime.now(timezone.utc).hour
                hour_mod = self.session_learner.get_hour_modifier(current_hour)
                if hour_mod['modifier'] != 0:
                    intelligent_modifier += hour_mod['modifier']
                    if hour_mod['modifier'] > 0:
                        reasons_for.append(f"🕐 Session learner: {hour_mod['reason']}")
                    else:
                        reasons_against.append(f"🕐 Session learner: {hour_mod['reason']}")
                    detailed_analysis['session_hour_modifier'] = hour_mod
                    logger.info(f"📊 SESSION MOD: {hour_mod['reason']} → {hour_mod['modifier']:+d}")
                
                current_day = datetime.now(timezone.utc).weekday()
                day_mod = self.session_learner.get_day_modifier(current_day)
                if day_mod['modifier'] != 0:
                    intelligent_modifier += day_mod['modifier']
                    if day_mod['modifier'] > 0:
                        reasons_for.append(f"📅 Day learner: {day_mod['reason']}")
                    else:
                        reasons_against.append(f"📅 Day learner: {day_mod['reason']}")
                    detailed_analysis['session_day_modifier'] = day_mod
                    logger.info(f"📊 DAY MOD: {day_mod['reason']} → {day_mod['modifier']:+d}")
        except Exception as e:
            logger.debug(f"Session modifier error: {e}")
        
        # System #3: Symbol memory modifier
        try:
            if hasattr(self, 'symbol_memory') and symbol:
                direction_str = 'LONG' if signal == 1 else 'SHORT'
                sym_mod = self.symbol_memory.get_symbol_modifier(symbol, direction_str)
                if sym_mod['modifier'] != 0:
                    intelligent_modifier += sym_mod['modifier']
                    if sym_mod['modifier'] > 0:
                        reasons_for.append(f"🧠 Symbol memory: {sym_mod['reason']}")
                    else:
                        reasons_against.append(f"🧠 Symbol memory: {sym_mod['reason']}")
                    detailed_analysis['symbol_memory_modifier'] = sym_mod
                    logger.info(f"📊 SYMBOL MEM: {sym_mod['reason']} → {sym_mod['modifier']:+d}")
        except Exception as e:
            logger.debug(f"Symbol memory modifier error: {e}")
        
        if intelligent_modifier != 0:
            pre_intelligent = final_score
            final_score = max(0, min(100, final_score + intelligent_modifier))
            detailed_analysis['intelligent_modifier'] = intelligent_modifier
            logger.info(f"📊 INTELLIGENT SYSTEMS: total modifier {intelligent_modifier:+d} "
                       f"(score: {pre_intelligent:.0f} → {final_score:.0f})")
        
        # === ENFORCE CUMULATIVE PENALTY CAP ===
        # Prevent penalty stacking from destroying otherwise decent signals
        # Score cannot drop below (1 - MAX_CUMULATIVE_PENALTY_PCT) of pre-penalty score
        min_allowed_score = pre_individual_penalty_score * (1 - MAX_CUMULATIVE_PENALTY_PCT)
        if final_score < min_allowed_score and pre_individual_penalty_score >= 40:
            logger.info(f"📊 Penalty cap: {final_score:.0f} → {min_allowed_score:.0f} (capped at {MAX_CUMULATIVE_PENALTY_PCT:.0%} reduction from {pre_individual_penalty_score:.0f})")
            final_score = min_allowed_score
        
        # Determine approval thresholds
        math_approved = final_score >= 50
        math_strong = final_score >= 65
        can_override = final_score >= 60
        
        # PhD Enhancement: Add walk-forward validation diagnostics
        try:
            from indicator import calculate_walk_forward_performance
            
            # Create a minimal dataframe for walk-forward analysis
            if 'returns' not in df.columns:
                df_analysis = df.copy()
                df_analysis['returns'] = df_analysis['close'].pct_change()
            else:
                df_analysis = df.copy()
            
            wf_result = calculate_walk_forward_performance(df_analysis, window=50, step=10)
            if wf_result.get('status') == 'success':
                detailed_analysis['walk_forward'] = wf_result
                mean_oos = wf_result.get('mean_oos_return', 0)
                mean_win_rate = wf_result.get('mean_win_rate', 0)
                
                if mean_oos > 0 and mean_win_rate > 0.5:
                    reasons_for.append(f"Walk-forward validated: {mean_win_rate:.0%} win rate, OOS return {mean_oos:.4f}%")
                    # Boost score if walk-forward is positive
                    final_score = min(100, final_score * 1.05)
        except Exception as e:
            logger.debug(f"Walk-forward analysis skipped: {e}")
        
        # ═══════════════════════════════════════════════════════════════════
        # LOG SUMMARY OF ALL APPLIED FILTERS (for debugging)
        # ═══════════════════════════════════════════════════════════════════
        applied_penalties = []
        if detailed_analysis.get('volume_penalty'):
            applied_penalties.append(f"Vol:{detailed_analysis['volume_penalty']}")
        if detailed_analysis.get('adx_penalty'):
            applied_penalties.append(f"ADX:{detailed_analysis['adx_penalty']}")
        if detailed_analysis.get('order_flow_penalty'):
            applied_penalties.append(f"OrderFlow:{detailed_analysis['order_flow_penalty']}")
        if detailed_analysis.get('vwap_penalty'):
            applied_penalties.append(f"VWAP:{detailed_analysis['vwap_penalty']}")
        if detailed_analysis.get('session_penalty'):
            applied_penalties.append(f"Session:{detailed_analysis['session_penalty']}")
        if detailed_analysis.get('liquidity_penalty'):
            applied_penalties.append(f"Liquidity:{detailed_analysis['liquidity_penalty']}")
        if detailed_analysis.get('mtf_penalty'):
            applied_penalties.append(f"MTF:{detailed_analysis['mtf_penalty']}")
        if detailed_analysis.get('local_peak_penalty'):
            applied_penalties.append(f"Peak:{detailed_analysis['local_peak_penalty']}")
        if detailed_analysis.get('spike_penalty'):
            applied_penalties.append(f"Spike:{detailed_analysis['spike_penalty']}")
        
        if applied_penalties:
            direction = "LONG" if signal == 1 else "SHORT"
            logger.info(f"📊 FILTERS APPLIED ({direction}): {', '.join(applied_penalties)} → Score={final_score}")
        
        return {
            'score': final_score,
            'approved': math_approved,
            'strong': math_strong,
            'scores': scores,
            'reasons_for': reasons_for,
            'reasons_against': reasons_against,
            'can_override_ai': can_override,
            'detailed_analysis': detailed_analysis,
            'confidence_level': confidence_level if 'confidence_level' in locals() else 'high'
        }
    
    def _calculate_hurst(self, prices: pd.Series, max_lag: int = 20) -> float:
        """Calculate Hurst exponent using R/S analysis with proper sub-segment method."""
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
    
    def calculate_cointegration_analysis(
        self,
        prices1: pd.Series,
        prices2: pd.Series,
        symbol1: str = "Asset1",
        symbol2: str = "Asset2"
    ) -> Dict[str, Any]:
        """
        Perform cointegration analysis for pairs trading.
        
        Uses Engle-Granger or Johansen tests to determine if two assets
        share a common stochastic trend (are cointegrated).
        
        Cointegrated pairs allow for statistical arbitrage:
        - When spread deviates from mean, expect mean-reversion
        - Long spread when oversold, short when overbought
        
        Args:
            prices1: Price series for first asset
            prices2: Price series for second asset
            symbol1: Name of first asset (for logging)
            symbol2: Name of second asset (for logging)
            
        Returns:
            Dict containing:
            - cointegrated: bool - Whether pair is cointegrated
            - hedge_ratio: float - Optimal hedge ratio
            - spread_zscore: float - Current spread z-score
            - trade_signal: str - 'long_spread', 'short_spread', 'close_spread', 'hold'
            - half_life: float - Mean reversion half-life in periods
            - entry/exit levels
        """
        try:
            from indicator import calculate_cointegration_test
            
            # Run Engle-Granger cointegration test
            result = calculate_cointegration_test(prices1, prices2, test_type='engle_granger')
            
            if result.get('status') != 'success':
                logger.warning(f"Cointegration test failed for {symbol1}/{symbol2}: {result.get('error', 'unknown')}")
                return {
                    'cointegrated': False,
                    'reason': result.get('error', 'test_failed'),
                    'symbol1': symbol1,
                    'symbol2': symbol2
                }
            
            cointegrated = result.get('cointegrated', False)
            hedge_ratio = result.get('hedge_ratio', 1.0)
            spread_zscore = result.get('spread_zscore', 0.0)
            half_life = result.get('half_life', np.inf)
            trade_signal = result.get('trade_signal', 'hold')
            
            logger.info(f"🔗 Cointegration {symbol1}/{symbol2}: "
                       f"{'✓ COINTEGRATED' if cointegrated else '✗ NOT cointegrated'} | "
                       f"Hedge={hedge_ratio:.4f} | Z={spread_zscore:.2f} | "
                       f"Half-life={half_life:.1f} | Signal={trade_signal}")
            
            return {
                'cointegrated': cointegrated,
                'symbol1': symbol1,
                'symbol2': symbol2,
                'hedge_ratio': hedge_ratio,
                'spread_zscore': spread_zscore,
                'half_life': half_life,
                'trade_signal': trade_signal,
                'trade_reason': result.get('trade_reason', ''),
                'adf_pvalue': result.get('adf_pvalue', 1.0),
                'entry_long': result.get('entry_long', 0),
                'entry_short': result.get('entry_short', 0),
                'exit_level': result.get('exit_level', 0),
                'spread_mean': result.get('spread_mean', 0),
                'spread_std': result.get('spread_std', 0),
                # Additional Johansen test if available
                'johansen_available': False  # Placeholder for future enhancement
            }
            
        except ImportError as ie:
            logger.warning(f"Cointegration analysis unavailable: {ie}")
            return {
                'cointegrated': False,
                'reason': 'missing_dependencies',
                'symbol1': symbol1,
                'symbol2': symbol2
            }
        except Exception as e:
            logger.error(f"Cointegration analysis error for {symbol1}/{symbol2}: {e}")
            return {
                'cointegrated': False,
                'reason': str(e),
                'symbol1': symbol1,
                'symbol2': symbol2
            }
    
    def analyze_pairs_cointegration(
        self,
        pairs_data: List[Dict[str, Any]],
        reference_symbol: str = 'BTCUSDT'
    ) -> Dict[str, Any]:
        """
        Analyze cointegration of multiple pairs with a reference asset.
        
        Useful for identifying pairs that move together and can be used
        for hedging or statistical arbitrage strategies.
        
        Args:
            pairs_data: List of {symbol, df} dictionaries
            reference_symbol: Reference asset to test against (default: BTC)
            
        Returns:
            Dict with cointegrated pairs and trading signals
        """
        cointegrated_pairs = []
        signals = []
        
        # Find reference asset
        ref_data = None
        for pair in pairs_data:
            if reference_symbol in pair.get('symbol', ''):
                ref_data = pair
                break
        
        if ref_data is None:
            return {
                'status': 'no_reference',
                'message': f'Reference {reference_symbol} not found in pairs_data',
                'cointegrated_pairs': [],
                'signals': []
            }
        
        ref_prices = ref_data['df']['close'] if 'df' in ref_data else None
        if ref_prices is None:
            return {
                'status': 'no_reference_prices',
                'cointegrated_pairs': [],
                'signals': []
            }
        
        for pair in pairs_data:
            symbol = pair.get('symbol', '')
            if symbol == reference_symbol:
                continue
            
            df = pair.get('df')
            if df is None or 'close' not in df.columns:
                continue
            
            prices = df['close']
            
            # Perform cointegration test
            coint_result = self.calculate_cointegration_analysis(
                ref_prices, prices, reference_symbol, symbol
            )
            
            if coint_result.get('cointegrated', False):
                cointegrated_pairs.append({
                    'symbol': symbol,
                    'hedge_ratio': coint_result['hedge_ratio'],
                    'half_life': coint_result['half_life'],
                    'adf_pvalue': coint_result['adf_pvalue']
                })
                
                if coint_result['trade_signal'] in ['long_spread', 'short_spread']:
                    signals.append({
                        'symbol': symbol,
                        'signal': coint_result['trade_signal'],
                        'spread_zscore': coint_result['spread_zscore'],
                        'reason': coint_result['trade_reason']
                    })
        
        return {
            'status': 'success',
            'reference': reference_symbol,
            'total_tested': len(pairs_data) - 1,
            'cointegrated_count': len(cointegrated_pairs),
            'cointegrated_pairs': cointegrated_pairs,
            'active_signals': signals
        }
    
    def _validate_ai_decision_with_math(
        self,
        ai_result: Dict[str, Any],
        math_check: Dict[str, Any],
        signal: int,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Validate AI decision against mathematical analysis.
        Override AI if math strongly disagrees (SYMMETRICALLY in both directions).
        
        - If AI blocks but math score >= 65: Math can OVERRIDE to APPROVE
        - If AI approves but math score < 30: Math can OVERRIDE to BLOCK
        """
        ai_approved = ai_result.get('approved', False)
        ai_confidence = ai_result.get('confidence', 0.5)
        math_score = math_check.get('score', 50)
        math_approved = math_check.get('approved', False)
        math_can_override = math_check.get('can_override_ai', False)
        
        # Math can BLOCK if AI approves but math score is very low
        math_can_block = math_score < 30  # Symmetric threshold: strong rejection
        
        signal_type = "LONG" if signal == 1 else "SHORT"
        
        # === CASE 1: AI approves, math confirms ===
        if ai_approved and math_approved:
            logger.info(f"✅ AI + Math AGREE: Approve {signal_type} (AI: {ai_confidence:.0%}, Math: {math_score:.0f}/100)")
            return ai_result
        
        # === CASE 2: AI approves, but math STRONGLY disagrees (score < 30) ===
        if ai_approved and math_can_block:
            # SYMMETRIC OVERRIDE: Math blocks the trade
            logger.warning(f"🔄 MATH OVERRIDE (BLOCK): AI approved but Math score VERY LOW ({math_score:.0f}/100) - BLOCKING trade")
            return {
                'approved': False,
                'confidence': (100 - math_score) / 100,  # High confidence in blocking
                'reasoning': f"AI approved but blocked by very weak math analysis (score: {math_score:.0f}/100). " +
                            f"Math concerns: {', '.join(math_check.get('reasons_against', [])[:3])}",
                'risk_assessment': 'high',
                'override_reason': 'math_override_block',
                'original_ai_decision': ai_result,
                'math_score': math_score
            }
        
        # === CASE 2B: AI approves, math says NO but not strongly ===
        if ai_approved and not math_approved:
            # Let AI decision stand but log warning
            logger.warning(f"⚠️ AI approves but Math score low ({math_score:.0f}/100) - proceeding with caution")
            ai_result['math_warning'] = f"Math score {math_score:.0f}/100 below threshold"
            return ai_result
        
        # === CASE 3: AI blocks, math says it's a good trade ===
        # FIX: AI REJECT = AI is protecting us. Don't override it here.
        # Only MATH SUPREMACY (score ≥ 90 in Stage 3 pipeline) can override AI REJECT.
        # A score of 60-89 is NOT strong enough to dismiss AI's protection.
        if not ai_approved and math_can_override:
            logger.info(f"📊 {symbol}: AI REJECT stands despite math score {math_score:.0f}/100 — AI protection respected (need ≥90 MATH SUPREMACY to override)")
            ai_result['math_score'] = math_score
            ai_result['math_wanted_override'] = True  # Flag for diagnostics
            return ai_result
        
        # === CASE 4: AI blocks, math agrees ===
        if not ai_approved and not math_approved:
            logger.info(f"❌ AI + Math AGREE: Block {signal_type} (AI: {ai_confidence:.0%}, Math: {math_score:.0f}/100)")
            ai_result['math_confirms'] = True
            ai_result['math_score'] = math_score
            return ai_result
        
        # Default: return AI result
        return ai_result
    
    def _build_market_context(
        self,
        df: pd.DataFrame,
        current_price: float,
        atr: float
    ) -> Dict[str, Any]:
        """Build market context from DataFrame with defensive handling."""
        
        recent = df.tail(20) if len(df) >= 20 else df
        
        # Ensure we have data to work with
        if len(recent) < 1:
            return {
                "current_price": current_price,
                "price_change_1h": 0,
                "price_change_5m": 0,
                "volume_ratio": 1,
                "trend": "unknown",
                "volatility_pct": 0,
                "atr": atr,
                "sma_10": current_price,
                "sma_20": current_price
            }
        
        # Calculate key metrics with defensive checks
        try:
            price_change_1h = (current_price - recent.iloc[-12]["close"]) / recent.iloc[-12]["close"] * 100 if len(recent) >= 12 else 0
        except (IndexError, KeyError):
            price_change_1h = 0
            
        try:
            price_change_5m = (current_price - recent.iloc[-1]["close"]) / recent.iloc[-1]["close"] * 100
        except (IndexError, KeyError, ZeroDivisionError):
            price_change_5m = 0
        
        # Volume analysis
        # Use previous completed candle (not current incomplete candle) for fair comparison
        try:
            # Average of completed candles (excluding current)
            avg_volume = recent["volume"].iloc[:-1].mean() if len(recent) > 1 else recent["volume"].mean()
            # Use second-to-last candle (most recent completed) for comparison
            current_volume = recent.iloc[-2]["volume"] if len(recent) > 1 else recent.iloc[-1]["volume"]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        except (KeyError, IndexError):
            volume_ratio = 1
        
        # Trend analysis
        try:
            sma_10 = recent["close"].tail(10).mean()
            sma_20 = recent["close"].tail(20).mean()
            trend = "bullish" if sma_10 > sma_20 else "bearish"
        except (KeyError, ValueError):
            sma_10 = sma_20 = current_price
            trend = "unknown"
        
        # Volatility
        volatility = atr / current_price * 100 if current_price > 0 else 0
        
        # === ENHANCED TECHNICAL MATH DATA ===
        # These are the actual numbers the signal generation uses
        
        # SMA15/SMA40 crossover (the core signal)
        try:
            sma_15 = df['close'].rolling(15).mean().iloc[-1] if len(df) >= 15 else current_price
            sma_40 = df['close'].rolling(40).mean().iloc[-1] if len(df) >= 40 else current_price
            sma_15_prev = df['close'].rolling(15).mean().iloc[-2] if len(df) >= 16 else sma_15
            sma_40_prev = df['close'].rolling(40).mean().iloc[-2] if len(df) >= 41 else sma_40
            sma_crossover = "BULLISH" if sma_15 > sma_40 and sma_15_prev <= sma_40_prev else \
                           "BEARISH" if sma_15 < sma_40 and sma_15_prev >= sma_40_prev else \
                           "NONE"
            sma_spread_pct = ((sma_15 - sma_40) / sma_40 * 100) if sma_40 > 0 else 0
        except Exception:
            sma_15 = sma_40 = current_price
            sma_crossover = "UNKNOWN"
            sma_spread_pct = 0
        
        # RSI calculation
        try:
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
            else:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss.replace(0, 0.001)
                rsi = (100 - (100 / (1 + rs))).iloc[-1]
        except Exception:
            rsi = 50  # Default neutral RSI
        
        # ADX calculation (trend strength)
        try:
            if 'adx' in df.columns:
                adx = df['adx'].iloc[-1]
            else:
                from indicator import calculate_adx
                adx = calculate_adx(df)
        except Exception:
            adx = 25  # Default moderate trend
        
        # Momentum (rate of change)
        try:
            roc_5 = ((df['close'].iloc[-1] - df['close'].iloc[-6]) / df['close'].iloc[-6] * 100) if len(df) > 6 else 0
            roc_10 = ((df['close'].iloc[-1] - df['close'].iloc[-11]) / df['close'].iloc[-11] * 100) if len(df) > 11 else 0
        except Exception:
            roc_5 = roc_10 = 0  # Default no momentum
        
        return {
            "current_price": current_price,
            "price_change_1h": round(price_change_1h, 2),
            "price_change_5m": round(price_change_5m, 3),
            "volume_ratio": round(volume_ratio, 2),
            "trend": trend,
            "volatility_pct": round(volatility, 2),
            "atr": round(atr, 4),
            "sma_10": round(sma_10, 4),
            "sma_20": round(sma_20, 4),
            # Enhanced math data
            "sma_15": round(sma_15, 4),
            "sma_40": round(sma_40, 4),
            "sma_crossover": sma_crossover,
            "sma_spread_pct": round(sma_spread_pct, 3),
            "rsi": round(rsi, 1),
            "adx": round(adx, 1),
            "roc_5": round(roc_5, 2),
            "roc_10": round(roc_10, 2)
        }
    
    def _ai_final_entry_decision(
        self,
        symbol: str,
        math_direction: str,
        math_score: float,
        long_score: float,
        short_score: float,
        long_reasons_for: List[str],
        long_reasons_against: List[str],
        short_reasons_for: List[str],
        short_reasons_against: List[str],
        context: Dict[str, Any],
        level_info: Dict[str, Any] = None,  # Resistance/support info
        advanced_math: str = "",  # Advanced math analysis section (Kalman, POC, RSI divergence, etc.)
    ) -> Dict[str, Any]:
        """
        AI FINAL ENTRY DECISION - AI has full power to choose direction.
        
        AI receives:
        - Both LONG and SHORT math scores
        - All reasons for/against each direction
        - Market context
        
        AI outputs:
        - decision: LONG, SHORT, or REJECT
        - confidence: 0.0-1.0
        - reasoning: explanation
        """
        try:
            # Get performance and market context
            perf = self._get_performance_context()
            market = self._get_market_hours_context()
            
            # Get direction-based performance (CRITICAL for decision making)
            dir_perf = self._get_direction_performance()
            
            # Build extra caution warnings
            extra_caution = ""
            if perf["last_trade_was_loss"]:
                extra_caution = f"\n⚠️ CAUTION: Last {perf['consecutive_losses']} trade(s) were losses. Be extra skeptical!"
            if perf["consecutive_losses"] >= 2:
                extra_caution += "\n🛑 LOSING STREAK: Require very high confidence to approve any trade."
            if market["is_weekend"]:
                extra_caution += "\n📅 WEEKEND: Lower liquidity, higher risk."
            if market["activity_level"] == "low":
                extra_caution += "\n🌙 LOW ACTIVITY: Increased slippage risk."
            
            # Historical direction info (secondary - for context only)
            # NOTE: Don't add strong warnings - math is primary!
            
            # Get symbol-specific performance WITH per-direction breakdown
            symbol_perf = self._get_symbol_performance(symbol)
            symbol_warning = ""
            if symbol_perf["has_history"]:
                # Just provide context, not strong warnings
                # Math analysis is PRIMARY, history is SECONDARY
                if symbol_perf.get("long_trades", 0) > 0 or symbol_perf.get("short_trades", 0) > 0:
                    long_str = f"LONG: {symbol_perf['long_wins']}W/{symbol_perf['long_losses']}L" if symbol_perf.get("long_trades", 0) > 0 else "LONG: new"
                    short_str = f"SHORT: {symbol_perf['short_wins']}W/{symbol_perf['short_losses']}L" if symbol_perf.get("short_trades", 0) > 0 else "SHORT: new"
                    symbol_warning = f"\n📊 {symbol} past trades (context only): {long_str} | {short_str}"
                
                # Only warn on extreme cases (5+ more losses than wins)
                if symbol_perf["losses"] > symbol_perf["wins"] + 4:
                    symbol_warning += f"\n⚠️ Note: {symbol} has struggled - but math may have found new pattern!"
            
            # Check for momentum contradiction
            momentum_5 = context.get('roc_5', 0)
            momentum_10 = context.get('roc_10', 0)
            momentum_warning = ""
            
            if math_direction == "SHORT" and (momentum_5 > 0.5 or momentum_10 > 0.3):
                momentum_warning = f"\n📈 MOMENTUM CONFLICT: Math says SHORT but price is RISING (5bar={momentum_5:+.2f}%, 10bar={momentum_10:+.2f}%). Consider: 1) LONG if LONG score >= 45, or 2) REJECT if neither direction is good."
            elif math_direction == "LONG" and (momentum_5 < -0.5 or momentum_10 < -0.3):
                momentum_warning = f"\n📉 MOMENTUM CONFLICT: Math says LONG but price is FALLING (5bar={momentum_5:+.2f}%, 10bar={momentum_10:+.2f}%). Consider: 1) SHORT if SHORT score >= 45, or 2) REJECT if neither direction is good."
            
            # Build resistance/support level warning
            level_warning = ""
            if level_info:
                if level_info.get('warning'):
                    level_warning = f"\n{level_info['warning']}"
                    
                # Add specific context
                dist_to_res = level_info.get('distance_to_resistance_pct', 999)
                dist_to_sup = level_info.get('distance_to_support_pct', 999)
                res_touches = level_info.get('resistance_touches', 0)
                sup_touches = level_info.get('support_touches', 0)
                
                if math_direction == "LONG" and dist_to_res < 2.0 and res_touches >= 2:
                    level_warning += f"\n🚨 DANGER FOR LONG: Price is only {dist_to_res:.1f}% below 24h resistance (tested {res_touches}x and rejected). High probability of reversal!"
                elif math_direction == "SHORT" and dist_to_sup < 2.0 and sup_touches >= 2:
                    level_warning += f"\n🚨 DANGER FOR SHORT: Price is only {dist_to_sup:.1f}% above 24h support (tested {sup_touches}x and held). High probability of bounce!"
            
            # Get market news and sentiment context (RECOMMENDATION ONLY!)
            news_ctx = self._get_news_context()
            news_section = ""
            if news_ctx.get('fear_greed_index', 50) != 50 or news_ctx.get('warning'):
                news_section = f"""
=== 📰 MARKET SENTIMENT (RECOMMENDATION ONLY - NOT A HARD RULE) ===
Fear & Greed Index: {news_ctx.get('fear_greed_index', 50)} ({news_ctx.get('fear_greed_label', 'Unknown')})
Market Cap 24h Change: {news_ctx.get('market_cap_change_24h', 0):+.2f}%
News Sentiment: {news_ctx.get('news_sentiment', 0):+.2f} (Bullish: {news_ctx.get('bullish_news_count', 0)}, Bearish: {news_ctx.get('bearish_news_count', 0)})
{f"Critical News: {news_ctx.get('critical_news_count', 0)} alerts" if news_ctx.get('critical_news_count', 0) > 0 else ''}
{chr(10).join('• ' + h for h in news_ctx.get('critical_headlines', [])[:2]) if news_ctx.get('critical_headlines') else ''}
⚠️ NOTE: Use this as CONTEXT only. Extreme fear can be a BUY opportunity (contrarian). 
   Math analysis (momentum, Kalman) is MORE RELIABLE than daily sentiment.
"""
            
            # Add advanced math section if provided
            advanced_math_section = ""
            if advanced_math:
                advanced_math_section = "\n" + advanced_math
            
            
            # Load active thresholds so prompt adapts to /threshold changes via Telegram
            _prompt_thr = get_active_thresholds(market.get('is_weekend', False))
            # Minimum AI confidence: weekday=60%, weekend=55% (aligned with post-AI gate)
            _prompt_min_conf = 0.55 if market.get('is_weekend', False) else 0.60
            
            prompt = f"""You are the FINAL DECISION MAKER for a crypto trading bot. Your role is to PROTECT CAPITAL first, then find profitable opportunities.

=== TRADE CANDIDATE ===
Symbol: {symbol}
Math's Preferred Direction: {math_direction} (score: {math_score:.0f}/100)
{momentum_warning}
{level_warning}
{news_section}
{advanced_math_section}

=== {symbol} SPECIFIC HISTORY (PER-PAIR DIRECTION STATS) ===
{self._build_symbol_history_section(symbol_perf, symbol)}
{symbol_warning}

=== BOTH DIRECTIONS COMPARED ===

📈 LONG ANALYSIS (Score: {long_score:.0f}/100):
Strengths: {', '.join(long_reasons_for[:4]) if long_reasons_for else 'None'}
Weaknesses: {', '.join(long_reasons_against[:4]) if long_reasons_against else 'None'}

📉 SHORT ANALYSIS (Score: {short_score:.0f}/100):
Strengths: {', '.join(short_reasons_for[:4]) if short_reasons_for else 'None'}
Weaknesses: {', '.join(short_reasons_against[:4]) if short_reasons_against else 'None'}

=== CURRENT MARKET STATE ===
Price: ${context.get('current_price', 0)}
1h Change: {context.get('price_change_1h', 0)}%
RSI: {context.get('rsi', 50):.1f} (<30=oversold bounce likely, >70=overbought drop likely)
ADX: {context.get('adx', 25):.1f} (>25=strong trend, >50=extreme)
Momentum 5-bar: {context.get('roc_5', 0):+.2f}% {'⬆️ RISING' if context.get('roc_5', 0) > 0 else '⬇️ FALLING'}
Momentum 10-bar: {context.get('roc_10', 0):+.2f}% {'⬆️ RISING' if context.get('roc_10', 0) > 0 else '⬇️ FALLING'}
Trend: {context.get('trend', 'UNKNOWN')}
Volume: {context.get('volume_ratio', 1):.1f}x average

=== HISTORICAL DIRECTION PERFORMANCE (SECONDARY - for context only) ===
📈 LONG trades: {dir_perf['long_trades']} total, {dir_perf['long_wins']}W/{dir_perf['long_losses']}L ({dir_perf['long_wr']:.0f}% WR), P&L: ${dir_perf['long_pnl']:+.2f}
📉 SHORT trades: {dir_perf['short_trades']} total, {dir_perf['short_wins']}W/{dir_perf['short_losses']}L ({dir_perf['short_wr']:.0f}% WR), P&L: ${dir_perf['short_pnl']:+.2f}
Note: Historical data is from OLD system - use as minor context only, NOT as decision maker!

=== PERFORMANCE CONTEXT ===
Overall Win Rate: {perf['win_rate']}%
Current Streak: {perf['consecutive_wins']}W / {perf['consecutive_losses']}L
{extra_caution}

=== INTELLIGENT DECISION FRAMEWORK ===

**SYSTEM MODE: {_prompt_thr['label']} | Period: {'WEEKEND' if market['is_weekend'] else 'WEEKDAY'}**

**🧮 MATH ANALYSIS IS PRIMARY (70% weight):**
- The math score incorporates: Kalman momentum, POC distance, Hurst exponent, RSI divergence, GARCH volatility
{"- 📅 WEEKEND MODE: Lower scores expected due to reduced liquidity — ML veto is the real safety net" if market["is_weekend"] else "- ☀️ WEEKDAY MODE: Standard scoring active"}
- Math score {_prompt_thr['math']}+ = STRONG signal, trust it!
- Math score {_prompt_thr['ai_high_conf']}-{_prompt_thr['math'] - 1} = Good signal, approve if momentum aligns
- Math score {_prompt_thr['stage2']}-{_prompt_thr['ai_high_conf'] - 1} = Marginal signal, needs strong momentum

**📊 HISTORICAL DATA IS SECONDARY (30% weight):**
- Historical data is advisory only - it should NOT override good math
- Use it as a TIE-BREAKER when math scores are similar for both directions
- A good math signal should be approved even if historical data is limited

**🚨 CRITICAL CONFIDENCE REQUIREMENTS:**
- You MUST provide at least {_prompt_min_conf:.0%} confidence to approve a trade!
- Below {_prompt_min_conf:.0%} confidence = automatic REJECT by the system!
- When approving, AIM for {_prompt_min_conf + 0.05:.0%}-{_prompt_min_conf + 0.20:.0%} confidence!

**APPROVAL GUIDELINES (FOLLOW THESE!):**
✅ Math score {_prompt_thr['math'] + 5}+ AND momentum aligns → APPROVE with {_prompt_min_conf + 0.15:.0%}-{_prompt_min_conf + 0.25:.0%} confidence
✅ Math score {_prompt_thr['math']}-{_prompt_thr['math'] + 4} AND momentum aligns → APPROVE with {_prompt_min_conf + 0.10:.0%}-{_prompt_min_conf + 0.15:.0%} confidence
✅ Math score {_prompt_thr['ai_high_conf']}-{_prompt_thr['math'] - 1} AND momentum aligns → APPROVE with {_prompt_min_conf + 0.05:.0%}-{_prompt_min_conf + 0.10:.0%} confidence
⚠️ Math score {_prompt_thr['stage2']}-{_prompt_thr['ai_high_conf'] - 1} → APPROVE with {_prompt_min_conf:.0%}-{_prompt_min_conf + 0.05:.0%} ONLY with VERY STRONG momentum

**🚨 CRITICAL: CHECK MOMENTUM DIRECTION!**
- For LONG: ROC5 MUST be positive (price rising) - otherwise REJECT!
- For SHORT: ROC5 MUST be negative (price falling) - otherwise REJECT!
- Math score alone is NOT enough - momentum direction MUST align!

**REJECTION IS REQUIRED FOR THESE CASES:**
❌ Math score < {_prompt_thr['stage2']} (weak signal - insufficient edge)
❌ MOMENTUM CONFLICT: Math says LONG but ROC5 is negative (price falling!)
❌ MOMENTUM CONFLICT: Math says SHORT but ROC5 is positive (price rising!)
❌ RSI extreme AND against direction (RSI<20 for SHORT, RSI>80 for LONG)

**DO NOT REJECT FOR:**
✓ "Low volume" - already factored into math score
✓ "Limited history" - fresh start is intentional
✓ "Uncertain direction" - math score tells you direction
✓ "Multiple factors" - vague reasons are not valid
✓ "Near support/resistance" - already factored into math score and level analysis

**PICK THE BEST DIRECTION:**
- If LONG score > SHORT score by 5+ AND ROC5 positive → approve LONG
- If SHORT score > LONG score by 5+ AND ROC5 negative → approve SHORT
- If momentum conflicts with math direction → REJECT (don't trade against momentum!)

**CONFIDENCE GUIDELINES (CRITICAL — READ CAREFULLY!):**
- Math {_prompt_thr['stage2']}-{_prompt_thr['ai_high_conf'] - 1} + strong momentum → confidence {_prompt_min_conf:.2f}-{_prompt_min_conf + 0.05:.2f}
- Math {_prompt_thr['ai_high_conf']}-{_prompt_thr['math'] - 1} + momentum aligns → confidence {_prompt_min_conf + 0.05:.2f}-{_prompt_min_conf + 0.10:.2f}
- Math {_prompt_thr['math']}+ + momentum aligns → confidence {_prompt_min_conf + 0.10:.2f}-{_prompt_min_conf + 0.20:.2f}
- REMEMBER: Below {_prompt_min_conf:.0%} = auto-REJECT! If you want to approve, give {_prompt_min_conf:.0%}+!

**YOUR ROLE — INDEPENDENT AI ANALYSIS (25% of final score!):**
You provide your OWN independent analysis score (0-100) based on what YOU see:
- Momentum alignment (ROC5 direction vs trade direction)
- RSI positioning (oversold/overbought context)
- Volume patterns (conviction behind moves)
- Price structure (trend clarity, consolidation, reversal signs)
- Cross-signal confluence (do multiple indicators agree?)
- Risk/reward context (news, sentiment, unusual patterns)

This is YOUR score — not a copy of the math score! Rate the trade setup from your perspective.
- 70-100 = Strong setup: multiple signals align, clear momentum, good risk/reward
- 50-69 = Decent setup: some signals align but mixed picture
- 30-49 = Weak setup: conflicting signals, poor momentum, or high risk
- 0-29 = Bad setup: momentum conflicts, extreme RSI against direction, danger signs

If momentum conflicts with math, you MUST reject AND give low ai_analysis (<30).
If you see something math missed (bullish divergence, news catalyst, etc.), give HIGH ai_analysis!

**DECISION WEIGHT: Math 55% + Your AI Analysis 25% + Confidence 20%**

=== YOUR RESPONSE ===
Respond ONLY with this JSON (nothing else):
{{"decision": "LONG", "confidence": {_prompt_min_conf + 0.10:.2f}, "ai_analysis": 65, "reasoning": "Brief explanation"}}

Valid decisions: "LONG", "SHORT", "REJECT"
Confidence: 0.50-1.00 ({_prompt_min_conf:.2f}+ to approve)
ai_analysis: 0-100 (YOUR independent assessment of the trade setup)
"""
            
            # Make API call
            for attempt in range(2):
                result_text = self._generate_content(prompt)
                if not result_text:
                    continue
                    
                result_text = result_text.strip()
                
                # Parse JSON
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0]
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0]
                
                result_text = result_text.strip()
                result = json.loads(result_text)
                
                # Validate and normalize
                decision = result.get('decision', 'REJECT').upper()
                if decision not in ['LONG', 'SHORT', 'REJECT']:
                    decision = 'REJECT'
                
                confidence = float(result.get('confidence', 0.5))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1
                
                reasoning = result.get('reasoning', 'AI decision')
                
                # Parse AI's independent analysis score (0-100)
                # This is Gemini's OWN assessment of the trade setup quality
                ai_analysis = float(result.get('ai_analysis', confidence * 100))
                ai_analysis = max(0.0, min(100.0, ai_analysis))  # Clamp to 0-100
                
                # Apply loss cooldown - require slightly higher confidence after losses
                # NOTE: Post-AI gate (0.60 weekday / 0.55 weekend) is the REAL threshold.
                # This inner gate only adds extra caution after losing streaks.
                _inner_testing = get_threshold_mode() == 'testing'
                if _inner_testing:
                    # Testing mode: minimal inner gate to avoid death spiral
                    # (can't trade → can't win → can't reset streak → can't trade)
                    required_conf = 0.50  # Testing: let post-AI gate be the real filter
                elif perf["consecutive_losses"] >= 3:
                    required_conf = 0.70  # After 3+ losses, need solid confidence
                elif perf["consecutive_losses"] >= 2:
                    required_conf = 0.65  # After 2 losses, reasonable confidence
                elif perf["last_trade_was_loss"]:
                    required_conf = 0.60  # After 1 loss, slightly cautious
                else:
                    required_conf = 0.55  # Normal: let post-AI gate be the real filter
                
                # If confidence too low, force REJECT
                if decision != 'REJECT' and confidence < required_conf:
                    logger.info(f"AI chose {decision} but confidence {confidence:.0%} < required {required_conf:.0%}, forcing REJECT")
                    decision = 'REJECT'
                    reasoning = f"Low confidence ({confidence:.0%}) below threshold ({required_conf:.0%})"
                
                logger.info(f"🤖 AI FINAL DECISION for {symbol}: {decision} (conf={confidence:.0%}, analysis={ai_analysis:.0f})")
                
                return {
                    'decision': decision,
                    'confidence': confidence,
                    'ai_analysis': ai_analysis,
                    'reasoning': reasoning
                }
            
            # API failed - default to REJECT (safety)
            logger.warning(f"AI API failed for {symbol} entry decision - defaulting to REJECT")
            return {
                'decision': 'REJECT',
                'confidence': 0.0,
                'ai_analysis': 0,
                'reasoning': 'AI API failed - safety reject'
            }
            
        except Exception as e:
            import traceback
            logger.error(f"AI final entry decision error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'decision': 'REJECT',
                'confidence': 0.0,
                'ai_analysis': 0,
                'reasoning': f'AI error: {e}'
            }
    
    def _ai_analysis(
        self,
        signal: int,
        context: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """Use Google Gemini API for signal analysis with SKEPTIC MODE."""
        try:
            signal_type = "LONG" if signal == 1 else "SHORT"
            
            # Get additional context
            perf = self._get_performance_context()
            market = self._get_market_hours_context()
            
            # Extract system score and ML insight from context
            system_score = context.get('system_score', {})
            ml_insight = context.get('ml_insight', {})
            tech_score_data = context.get('tech_score', {})
            # NEW: Extract PhD Math Check
            math_check = context.get('math_check', {})
            
            # Build PhD Math section
            math_section = ""
            if math_check:
                math_score = math_check.get('final_score', 50)
                kelly = math_check.get('detailed', {}).get('kelly_fraction', 0)
                hurst = math_check.get('detailed', {}).get('hurst_exponent', 0.5)
                r_squared = math_check.get('detailed', {}).get('r_squared', 0)
                reasons_for = math_check.get('reasons_for', [])
                reasons_against = math_check.get('reasons_against', [])
                
                math_section = f"""
=== PHD MATHEMATICAL VERIFICATION ===
Overall Math Score: {math_score}/100
• Kelly Criterion (Optimal Size): {kelly:.1%} (Risk Management)
• Hurst Exponent (Trend State): {hurst:.2f} (0.5=Random, >0.6=Trend, <0.4=Revert)
• Trend Fit (R²): {r_squared:.2f}
• Quant Strengths: {', '.join(reasons_for[:3]) if reasons_for else 'None'}
• Quant Weaknesses: {', '.join(reasons_against[:3]) if reasons_against else 'None'}"""

            # Build technical score section
            tech_section = ""
            if tech_score_data:
                t_score = tech_score_data.get('score', 50)
                t_quality = tech_score_data.get('quality', 'UNKNOWN')
                t_factors = tech_score_data.get('factors', [])
                t_breakdown = tech_score_data.get('breakdown', {})
                factors_str = ", ".join(t_factors) if t_factors else "None notable"
                
                # Build breakdown string if available
                breakdown_parts = []
                if t_breakdown:
                    for k, v in t_breakdown.items():
                        if v > 0:
                            breakdown_parts.append(f"{k}={v}")
                breakdown_str = " + ".join(breakdown_parts) if breakdown_parts else "N/A"
                
                tech_section = f"""
=== TECHNICAL SCORE (Math-based quality) ===
Score: {t_score}/100 ({t_quality})
Breakdown: {breakdown_str}
Notable Factors: {factors_str}
NOTE: Score based on ADX, Hurst exponent, Volume, RSI, Regime clarity"""
            
            # Build system assessment section
            system_section = ""
            if system_score:
                score = system_score.get('combined', 50)
                rec = system_score.get('recommendation', 'NEUTRAL')
                breakdown = system_score.get('breakdown', 'N/A')
                short_bonus = system_score.get('short_bonus', 0)
                system_section = f"""
=== SYSTEM SCORE (Pre-computed) ===
Combined Score: {score:.0f}/100
Recommendation: {rec}
Breakdown: {breakdown}
{"Short Bias Applied: +5 points" if short_bonus > 0 else ""}
NOTE: Score ≥60 = favorable setup, ≥75 = strong setup"""
            
            # Build ML section
            ml_section = ""
            # Build ML section — DUAL ML MODELS
            ml_section = ""
            if ml_insight.get('ml_available'):
                win_prob = ml_insight.get('ml_win_probability', 0.5)
                ml_conf = ml_insight.get('ml_confidence', 'low')
                ml_source = ml_insight.get('ml_source', 'unknown')
                
                hist = ml_insight.get('historical_ml', {})
                live = ml_insight.get('live_ml', {})
                
                hist_line = ""
                if hist.get('available'):
                    hist_line = f"Historical Model: {hist['win_probability']:.1%} ({hist['confidence']}) — {hist.get('training_samples', 0)} samples"
                else:
                    hist_line = "Historical Model: Not trained"
                
                live_line = ""
                if live.get('available'):
                    live_line = f"Live Model: {live['win_probability']:.1%} ({live['confidence']}) — {live.get('training_samples', 0)} samples"
                else:
                    live_line = f"Live Model: Learning ({live.get('total_samples', 0)} samples, needs 30)"
                
                ml_section = f"""
=== DUAL ML PREDICTION ===
Combined Win Probability: {win_prob:.1%} ({ml_conf}) — Source: {ml_source}
{hist_line}
{live_line}
NOTE: Historical model provides baseline, Live model adapts to YOUR trading patterns"""
            else:
                ml_section = """
=== ML MODELS ===
Status: Not available (still learning)"""
            
            # Build market scanner context section
            scanner = context.get('market_scanner', {})
            scanner_section = ""
            if scanner.get('best_pair'):
                best = scanner.get('best_pair', '')
                current_rank = scanner.get('current_pair_rank', 0)
                is_recommended = scanner.get('is_recommended_pair', False)
                scanner_section = f"""
=== MARKET SCANNER CONTEXT ===
Scanner's Best Pair: {best}
Current Pair Rank: #{current_rank} in market scan
Is This The Recommended Pair: {'YES ✓' if is_recommended else 'NO - scanner prefers ' + best}
NOTE: If scanner recommends a different pair, consider if this trade is worth taking"""
            
            # Determine if we need extra caution
            extra_caution = ""
            if perf["last_trade_was_loss"]:
                extra_caution = f"\n⚠️ CAUTION: Last {perf['consecutive_losses']} trade(s) were losses. Be extra skeptical!"
            if perf["consecutive_losses"] >= 2:
                extra_caution += "\n🛑 LOSING STREAK: Require very high confidence to approve."
            if market["is_weekend"]:
                extra_caution += "\n📅 WEEKEND: Lower liquidity, higher risk of false moves."
            if market["activity_level"] == "low":
                extra_caution += "\n🌙 LOW ACTIVITY HOURS: Increased slippage risk."
            
            prompt = f"""You are a SKEPTICAL crypto trading supervisor. Your job is to PROTECT capital by rejecting bad trades.

=== SIGNAL ===
Proposed Trade: {signal_type} on {symbol}

=== PHD MATHEMATICAL VERIFICATION ===
{math_section}

=== TECHNICAL MATH (Signal Generation) ===
SMA Crossover Signal: {context.get('sma_crossover', 'UNKNOWN')}
SMA15: ${context.get('sma_15', 0):.4f}
SMA40: ${context.get('sma_40', 0):.4f}
SMA Spread: {context.get('sma_spread_pct', 0):+.3f}% (positive = bullish)
RSI: {context.get('rsi', 50):.1f} (30-70 normal, <30 oversold, >70 overbought)
ADX: {context.get('adx', 0):.1f} (>25 trending, >35 strong trend, 35-40 DANGER ZONE)
Momentum 5-bar: {context.get('roc_5', 0):+.2f}%
Momentum 10-bar: {context.get('roc_10', 0):+.2f}%

=== MARKET DATA ===
Current Price: ${context['current_price']}
1-Hour Price Change: {context['price_change_1h']}%
Volume Ratio (vs avg): {context['volume_ratio']}x
Trend (SMA10 vs SMA20): {context['trend']}
Volatility (ATR%): {context['volatility_pct']}%
{tech_section}
{system_section}
{ml_section}
{scanner_section}

=== TRADING PERFORMANCE ===
Total Trades: {perf['total_trades']}
Win Rate: {perf['win_rate']}%
Current Streak: {perf['consecutive_wins']}W / {perf['consecutive_losses']}L
Recent P&L (last 5): ${perf['recent_pnl']}

=== MARKET SESSION ===
Session: {market['session']} ({market['hour_utc']}:00 UTC)
Activity Level: {market['activity_level']}
Weekend: {market['is_weekend']}
{extra_caution}

=== MATH-BASED DECISION RULES ===
APPROVE signals when:
- PhD Math Score is high (>70)
- Kelly Criterion suggests positive sizing (>0%)
- SMA crossover matches signal direction
- Hurst Exponent confirms regime

REJECT signals when:
- PhD Math detects "Quant Weaknesses" that are critical
- Kelly ~0% (Negative expectancy)
- ADX 35-40 (DANGER ZONE)
- Momentum diverges from signal

=== YOUR TASK ===
1. Analyze the "PHD MATHEMATICAL VERIFICATION" section first.
2. List 3 quantified reasons why this trade could FAIL (e.g., low R², poor Hurst).
3. Decide if the mathematical edge is sufficient to risk capital.

Only approve if the MATH is compelling.

Respond ONLY with this JSON format, no other text:
{{"reasons_against": ["reason1", "reason2", "reason3"], "approved": false, "confidence": 0.65, "reasoning": "why approved or rejected", "risk_assessment": "low/medium/high"}}"""

            # Retry loop for robustness
            for attempt in range(2):
                result_text = self._generate_content(prompt)
                if not result_text:
                    continue
                result_text = result_text.strip()
            
                # Parse JSON from response
                if "```json" in result_text:
                    result_text = result_text.split("```json")[1].split("```")[0]
                elif "```" in result_text:
                    result_text = result_text.split("```")[1].split("```")[0]
            
                # Clean up common issues
                result_text = result_text.strip()
            
                result = json.loads(result_text)
            
                # Ensure required fields exist
                result.setdefault("approved", False)
                result.setdefault("confidence", 0.5)
                result.setdefault("reasoning", "AI analysis")
                result.setdefault("risk_assessment", "medium")
                result.setdefault("reasons_against", [])
            
                # Platt scaling: calibrate raw Gemini confidence
                raw_conf = result["confidence"]
                result["confidence"] = self._calibrate_ai_confidence(raw_conf)
                result["raw_confidence"] = raw_conf  # Keep original for tracking

                # Apply confidence threshold with LOSS COOLDOWN
                perf = self._get_performance_context()
            
                if perf["consecutive_losses"] >= 2:
                    # After 2+ consecutive losses, require 90% confidence
                    required_threshold = self.loss_cooldown_threshold
                    logger.info(f"Loss cooldown active: requiring {required_threshold:.0%} confidence")
                elif perf["last_trade_was_loss"]:
                    # After 1 loss, require 85% confidence
                    required_threshold = 0.85
                else:
                    required_threshold = self.confidence_threshold
            
                result["approved"] = result["approved"] and result["confidence"] >= required_threshold
                result["threshold_used"] = required_threshold
            
                # Log the skeptic analysis
                if result.get("reasons_against"):
                    logger.info(f"AI reasons against trade: {result['reasons_against']}")
            
                return result
            
            # If we get here, both attempts failed
            return self._rule_based_analysis(signal, context, symbol)
            
        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}, falling back to rules")
            return self._rule_based_analysis(signal, context, symbol)
    
    def _rule_based_analysis(
        self,
        signal: int,
        context: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """
        Rule-based signal validation when AI is unavailable.
        Uses comprehensive math check as primary decision maker.
        """
        signal_type = "LONG" if signal == 1 else "SHORT"
        
        # Use math check if available
        math_check = context.get('math_check')
        if math_check:
            math_score = math_check.get('score', 50)
            math_approved = math_check.get('approved', False)
            reasons_for = math_check.get('reasons_for', [])
            reasons_against = math_check.get('reasons_against', [])
            
            logger.info(f"📊 Rule-based using Math Check: {math_score:.0f}/100, Approved: {math_approved}")
            
            if math_approved:
                risk = "low" if math_score >= 70 else "medium"
            else:
                risk = "high" if math_score < 40 else "medium"
            
            return {
                "approved": math_approved,
                "confidence": math_score / 100,
                "reasoning": f"Math-based decision (AI unavailable). Score: {math_score:.0f}/100. " +
                           f"Pros: {', '.join(reasons_for[:3]) or 'None'}. " +
                           f"Cons: {', '.join(reasons_against[:3]) or 'None'}.",
                "risk_assessment": risk,
                "source": "math_rules",
                "math_score": math_score
            }
        
        # Fallback to basic rules if no math check
        confidence = 0.5  # Start neutral
        reasons = []
        risk = "medium"
        
        # Rule 1: Trend alignment
        if (signal == 1 and context["trend"] == "bullish") or \
           (signal == -1 and context["trend"] == "bearish"):
            confidence += 0.15
            reasons.append(f"Signal aligns with {context['trend']} trend")
        else:
            confidence -= 0.1
            reasons.append(f"Counter-trend trade ({context['trend']} market)")
        
        # Rule 2: Volume confirmation
        if context["volume_ratio"] > 1.2:
            confidence += 0.1
            reasons.append("Strong volume confirmation")
        elif context["volume_ratio"] < 0.5:
            confidence -= 0.1
            reasons.append("Low volume - weak confirmation")
        
        # Rule 3: Volatility check
        if context["volatility_pct"] > 3:
            confidence -= 0.1
            risk = "high"
            reasons.append("High volatility environment")
        elif context["volatility_pct"] < 1:
            confidence += 0.05
            risk = "low"
            reasons.append("Low volatility - stable conditions")
        
        # Rule 4: Recent momentum
        if (signal == 1 and context["price_change_1h"] > 0) or \
           (signal == -1 and context["price_change_1h"] < 0):
            confidence += 0.1
            reasons.append("Momentum supports direction")
        
        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "approved": confidence >= self.confidence_threshold,
            "confidence": round(confidence, 2),
            "reasoning": "; ".join(reasons),
            "risk_assessment": risk
        }
    
    def _log_analysis(
        self,
        signal: int,
        symbol: str,
        result: Dict[str, Any]
    ):
        """Log analysis for tracking."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "signal": "LONG" if signal == 1 else "SHORT",
            **result
        }
        self.trade_history.append(entry)
        
        status = "✅ APPROVED" if result["approved"] else "❌ REJECTED"
        logger.info(
            f"AI Filter {status}: {symbol} {entry['signal']} | "
            f"Confidence: {result['confidence']:.0%} | "
            f"Risk: {result['risk_assessment']} | "
            f"Reason: {result['reasoning']}"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filter statistics from memory and/or file."""
        # First try in-memory history
        history = self.trade_history.copy() if self.trade_history else []
        
        # If empty, load from ai_decisions.json
        if not history:
            try:
                decisions_file = Path(__file__).parent / "ai_decisions.json"
                if decisions_file.exists():
                    import json
                    with open(decisions_file, 'r') as f:
                        data = json.load(f)
                        if isinstance(data, dict) and 'decisions' in data:
                            history = data['decisions']
                        elif isinstance(data, list):
                            history = data
            except Exception as e:
                logger.debug(f"Could not load ai_decisions.json: {e}")
        
        if not history:
            return {"total_signals": 0, "approved": 0, "rejected": 0}
        
        approved = sum(1 for t in history if t.get("approved", False))
        return {
            "total_signals": len(history),
            "approved": approved,
            "rejected": len(history) - approved,
            "approval_rate": f"{approved / len(history):.1%}"
        }
    
    def _math_proactive_scan(
        self,
        df: pd.DataFrame,
        current_price: float,
        atr: float,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Math-based proactive scan when AI is unavailable.
        Uses comprehensive math check to identify high-quality opportunities.
        Requires higher thresholds than signal-based trades.
        """
        try:
            context = self._build_market_context(df, current_price, atr)
            
            # Check both directions and pick the better one
            long_check = self._comprehensive_math_check(1, df, current_price, atr, context)
            short_check = self._comprehensive_math_check(-1, df, current_price, atr, context)
            
            # Log BOTH scores for debugging direction choices
            long_score = long_check.get('score', 0)
            short_score = short_check.get('score', 0)
            logger.info(f"📊 Math direction comparison {symbol}: LONG={long_score:.0f}, SHORT={short_score:.0f} (diff: {abs(long_score-short_score):.0f})")
            
            # Proactive trades threshold - configurable, default 65
            # Lower threshold = more aggressive, higher = more conservative
            # Crypto trades 24/7 — slightly lower threshold on weekends
            base_threshold = getattr(self, 'proactive_threshold', 65)
            market_timing = self._get_market_hours_context()
            if market_timing.get('is_weekend', False):
                PROACTIVE_THRESHOLD = max(38, base_threshold - 22)  # Reduce by 22 on weekends (65→43)
                logger.debug(f"📊 Weekend proactive threshold: {PROACTIVE_THRESHOLD} (base: {base_threshold})")
            else:
                PROACTIVE_THRESHOLD = base_threshold
            
            best_direction = None
            best_score = 0
            best_check = None
            
            # === PRE-FILTERS: Block contradictory signals before flagging opportunity ===
            volume_ratio = context.get('volume_ratio', 1.0)
            
            # Filter 1: Volume too low (illiquid market)
            if volume_ratio < 0.1:
                logger.info(f"📊 Math scan: BLOCKED - Volume too low ({volume_ratio:.2f}x) - illiquid market")
                return None
            
            # Choose best direction
            if long_check['score'] >= PROACTIVE_THRESHOLD and long_check['score'] > short_check['score']:
                best_direction = "LONG"
                best_score = long_check['score']
                best_check = long_check
            elif short_check['score'] >= PROACTIVE_THRESHOLD and short_check['score'] > long_check['score']:
                best_direction = "SHORT"
                best_score = short_check['score']
                best_check = short_check
            
            if not best_direction:
                # Log at INFO level so scores are visible in logs
                logger.info(f"📊 Math scan: No opportunity (LONG: {long_check['score']:.0f}, SHORT: {short_check['score']:.0f}, need {PROACTIVE_THRESHOLD}+)")
                return None
            
            # Filter 2: Direction margin check - if LONG and SHORT too close, direction is unclear
            # FIXED: Increased from 8 to 15 - need CLEAR direction advantage
            score_margin = abs(long_check['score'] - short_check['score'])
            MIN_DIRECTION_MARGIN = 15  # Require at least 15-point difference to be confident in direction
            if score_margin < MIN_DIRECTION_MARGIN:
                logger.info(f"📊 Math scan: BLOCKED - Direction unclear for {symbol} (LONG={long_check['score']:.0f}, SHORT={short_check['score']:.0f}, margin={score_margin:.0f} < {MIN_DIRECTION_MARGIN})")
                return None
            
            # Filter 3: No statistical edge detected (DISABLED - too strict for live trading)
            # The score already gets penalized by 15% when p-value isn't significant
            # Blocking entirely prevents too many valid trades
            has_no_edge = best_check.get('detailed_analysis', {}).get('no_statistical_edge', False)
            if has_no_edge:
                logger.debug(f"📊 Math scan: Note - p-value not significant for {best_direction} {symbol}, but proceeding (score {best_score:.0f})")
                # Don't block - the score penalty is sufficient
            
            # Filter 4: Too many reasons against vs for
            reasons_for_count = len(best_check.get('reasons_for', []))
            reasons_against_count = len(best_check.get('reasons_against', []))
            if reasons_against_count > reasons_for_count:
                logger.info(f"📊 Math scan: BLOCKED - More negatives ({reasons_against_count}) than positives ({reasons_for_count}) for {best_direction} {symbol}")
                return None
            
            # ═══════════════════════════════════════════════════════════════
            # Filter 5: RESISTANCE/SUPPORT ZONE CHECK (NEW!)
            # Don't go LONG in resistance zone, don't go SHORT in support zone
            # Zones are calculated dynamically based on ATR (market volatility)
            # ═══════════════════════════════════════════════════════════════
            levels = self._detect_resistance_support_levels(df, current_price)
            
            in_r_zone = levels.get('in_resistance_zone', False)
            in_s_zone = levels.get('in_support_zone', False)
            zone_pct = levels.get('zone_width_pct', 2.0)
            r_upper = levels.get('resistance_upper', 0)
            r_lower = levels.get('resistance_lower', 0)
            s_upper = levels.get('support_upper', 0)
            s_lower = levels.get('support_lower', 0)
            pos_range = levels.get('position_in_range_pct', 50)
            
            # Log zone info
            zone_status = ""
            if in_r_zone:
                zone_status = " 🔴IN_RESISTANCE_ZONE"
            elif in_s_zone:
                zone_status = " 🔵IN_SUPPORT_ZONE"
            elif pos_range >= 88:
                zone_status = " ⚠️NEAR_RESISTANCE"
            elif pos_range <= 12:
                zone_status = " ⚠️NEAR_SUPPORT"
            logger.info(f"📊 {symbol}: Range={pos_range:.0f}% | Zone={zone_pct:.1f}% | R[${r_lower:.4f}-${r_upper:.4f}] S[${s_lower:.4f}-${s_upper:.4f}]{zone_status}")
            
            # Block LONG if in resistance zone
            if best_direction == "LONG" and in_r_zone:
                logger.warning(f"🚫 Math scan: BLOCKED LONG {symbol} - IN RESISTANCE ZONE [${r_lower:.4f}-${r_upper:.4f}] | Price at {pos_range:.0f}% of 24h range")
                return None
            
            # Block SHORT if in support zone
            if best_direction == "SHORT" and in_s_zone:
                logger.warning(f"🚫 Math scan: BLOCKED SHORT {symbol} - IN SUPPORT ZONE [${s_lower:.4f}-${s_upper:.4f}] | Price at {pos_range:.0f}% of 24h range")
                return None
            
            # Block LONG if price is in top 20% of range (near resistance even if not in zone)
            # HBAR at 84% went straight to loss — 80% cutoff prevents this
            if best_direction == "LONG" and pos_range >= 80:
                logger.warning(f"🚫 Math scan: BLOCKED LONG {symbol} - TOO CLOSE TO RESISTANCE (Range={pos_range:.0f}% >= 80%) | Risk of rejection")
                return None
            
            # Block SHORT if price is in bottom 20% of range (near support even if not in zone)
            if best_direction == "SHORT" and pos_range <= 20:
                logger.warning(f"🚫 Math scan: BLOCKED SHORT {symbol} - TOO CLOSE TO SUPPORT (Range={pos_range:.0f}% <= 20%) | Risk of bounce")
                return None
            
            # Determine risk based on score
            if best_score >= 80:
                risk_level = "low"
                risk_pct = 0.025  # 2.5%
            elif best_score >= 75:
                risk_level = "medium"
                risk_pct = 0.02  # 2%
            else:
                risk_level = "medium"
                risk_pct = 0.015  # 1.5%
            
            reasons = best_check.get('reasons_for', [])[:3]
            reasoning = f"Math scan score {best_score:.0f}/100. " + ", ".join(reasons)
            
            logger.info(
                f"📊 MATH OPPORTUNITY: {best_direction} {symbol} | "
                f"Score: {best_score:.0f}/100 | {reasoning}"
            )
            
            return {
                "action": best_direction,
                "signal": 1 if best_direction == "LONG" else -1,
                "confidence": best_score / 100,
                "reasoning": reasoning,
                "risk_assessment": risk_level,
                "suggested_risk_pct": risk_pct,
                "source": "math_proactive",
                "math_score": best_score
            }
            
        except Exception as e:
            logger.error(f"Math proactive scan error: {type(e).__name__}: {e}")
            return None
    
    def _ai_validate_position_decision(
        self,
        position_side: str,
        entry_price: float,
        current_price: float,
        unrealized_pnl_pct: float,
        math_action: str,
        math_score: float,
        hold_score: float,
        exit_score: float,
        math_reasoning: str,
        symbol: str
    ) -> Dict[str, Any]:
        """
        AI validates the math-based position monitoring decision.
        Returns adjusted recommendation or None if AI unavailable.
        Math decision is ALWAYS the fallback.
        """
        if not self.use_ai:
            return None
        
        try:
            pnl_emoji = "🟢" if unrealized_pnl_pct >= 0 else "🔴"
            prompt = f"""You are a position management AI validating a math-based decision.

═══════════════════════════════════════════════════════════════════════
CURRENT POSITION STATUS
═══════════════════════════════════════════════════════════════════════
Symbol: {symbol}
Side: {position_side}
Entry: ${entry_price:.4f}
Current: ${current_price:.4f}
{pnl_emoji} PnL: {unrealized_pnl_pct:+.2f}%

═══════════════════════════════════════════════════════════════════════
MATH ANALYSIS RESULT
═══════════════════════════════════════════════════════════════════════
Recommended Action: {math_action.upper()}
Hold Score: {hold_score:.0f}/100
Exit Score: {exit_score:.0f}/100
Adjusted Score: {math_score:.0f}/100
Reasoning: {math_reasoning}

═══════════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════════
Validate the math decision. You can:
1. CONFIRM: Agree with math recommendation
2. REFINE: Adjust confidence slightly based on market insight
3. OVERRIDE: Only if you see a CRITICAL issue (use sparingly)

Respond ONLY with JSON:
{{"validate": "confirm" or "refine" or "override", "action": "hold" or "close", "confidence_adjustment": -0.1 to +0.1, "note": "brief reason (max 50 chars)"}}"""

            result_text = self._generate_content(prompt)
            if not result_text:
                return None
            
            result_text = result_text.strip()
            
            # Parse JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            ai_result = json.loads(result_text.strip())
            
            validation = ai_result.get("validate", "confirm")
            ai_action = ai_result.get("action", math_action)
            confidence_adj = ai_result.get("confidence_adjustment", 0)
            note = ai_result.get("note", "")
            
            logger.info(f"🤖 AI Position Validation: {validation} | Action: {ai_action} | Note: {note}")
            
            return {
                "validation": validation,
                "action": ai_action,
                "confidence_adjustment": confidence_adj,
                "note": note
            }
            
        except Exception as e:
            logger.warning(f"AI position validation failed: {e} - using math decision")
            return None

    def _ai_smart_exit_decision(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        current_price: float,
        pnl_pct: float,
        pnl_usd: float,
        hold_score: float,
        exit_score: float,
        reversal_signals: List[str],
        context: Dict[str, Any],
        df: pd.DataFrame = None,
        profit_reversal_detected: bool = False,
        reversal_urgency: str = 'none',
        reversal_reason: str = '',
        momentum_hope: int = 0,
        momentum_hope_factors: List[str] = None,
        reversal_score: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        AI makes a POWERFUL exit decision with deep math understanding.
        Uses market microstructure, momentum analysis, and risk assessment.
        
        KEY PRINCIPLE: Balance between cutting losses quickly and giving
        positions room to breathe through normal market noise.
        
        Returns:
            {
                'should_exit': bool,
                'confidence': 0.0-1.0,
                'reasoning': str,
                'suggested_action': 'hold' | 'exit' | 'tighten_sl' | 'wait_for_bounce'
            }
        """
        if not self.use_ai:
            return None
        
        try:
            # === DEEP MATH ANALYSIS ===
            math_analysis = self._calculate_deep_position_metrics(
                df, side, entry_price, current_price, context
            )
            
            pnl_emoji = "🟢" if pnl_pct >= 0 else "🔴"
            reversal_text = '\n'.join([f"  • {s}" for s in reversal_signals]) if reversal_signals else "  • None detected"
            
            # Build comprehensive context for AI
            trend_status = context.get('trend', 'unknown')
            rsi = context.get('rsi', 50)
            volume_ratio = context.get('volume_ratio', 1.0)
            volatility = context.get('volatility_pct', 0)
            
            # === S/R LEVEL ANALYSIS ===
            sr_levels = self._detect_resistance_support_levels(df, current_price)
            dist_to_resistance = sr_levels.get('distance_to_resistance_pct', 99)
            dist_to_support = sr_levels.get('distance_to_support_pct', 99)
            range_position = sr_levels.get('position_in_range_pct', 50)
            in_r_zone = sr_levels.get('in_resistance_zone', False)
            in_s_zone = sr_levels.get('in_support_zone', False)
            
            # Build S/R context for AI with side-specific warnings
            sr_context = f"Position in 24h range: {range_position:.0f}%\n"
            sr_context += f"  • Distance to Resistance: {dist_to_resistance:.2f}%"
            if in_r_zone:
                sr_context += " 🚫 IN DANGER ZONE"
            elif dist_to_resistance < 1.0:
                sr_context += " ⚠️ VERY CLOSE"
            sr_context += f"\n  • Distance to Support: {dist_to_support:.2f}%"
            if in_s_zone:
                sr_context += " 🚫 IN DANGER ZONE"
            elif dist_to_support < 1.0:
                sr_context += " ⚠️ VERY CLOSE"
            
            # Add side-specific S/R warnings
            if side == 'LONG':
                if in_r_zone:
                    sr_context += "\n\n🚨 LONG IN RESISTANCE ZONE - HIGH EXIT URGENCY!"
                    sr_context += f"\n   Price likely to reject here. Only {dist_to_resistance:.2f}% upside before resistance."
                elif dist_to_resistance < 0.5:
                    sr_context += f"\n\n⚠️ LONG very close to resistance ({dist_to_resistance:.2f}% away) - limited upside!"
                if dist_to_support < 1.0:
                    sr_context += f"\n   ✅ Support nearby ({dist_to_support:.2f}% below) - good protection"
            else:  # SHORT
                if in_s_zone:
                    sr_context += "\n\n🚨 SHORT IN SUPPORT ZONE - HIGH EXIT URGENCY!"
                    sr_context += f"\n   Price likely to bounce here. Only {dist_to_support:.2f}% downside before support."
                elif dist_to_support < 0.5:
                    sr_context += f"\n\n⚠️ SHORT very close to support ({dist_to_support:.2f}% away) - limited downside!"
                if dist_to_resistance < 1.0:
                    sr_context += f"\n   ✅ Resistance nearby ({dist_to_resistance:.2f}% above) - good protection"
            
            # Determine market regime
            if rsi < 30:
                rsi_status = "OVERSOLD (bounce likely)"
            elif rsi > 70:
                rsi_status = "OVERBOUGHT (pullback likely)"
            else:
                rsi_status = f"Neutral ({rsi:.0f})"
            
            prompt = f"""You are an EXPERT AI position manager with deep mathematical understanding. Analyze this position and decide the OPTIMAL action.

═══════════════════════════════════════════════════════════════════════════════
📊 POSITION STATUS
═══════════════════════════════════════════════════════════════════════════════
Symbol: {symbol}
Direction: {side}
Entry Price: ${entry_price:.4f}
Current Price: ${current_price:.4f}
{pnl_emoji} Unrealized PnL: {pnl_pct:+.2f}% (${pnl_usd:+.2f})
Distance to Entry: {abs(pnl_pct):.2f}%

═══════════════════════════════════════════════════════════════════════════════
📈 MATHEMATICAL ANALYSIS (PhD-Level)
═══════════════════════════════════════════════════════════════════════════════
Math Hold Score: {hold_score:.0f}/100 (higher = position looks good)
Math Exit Score: {exit_score:.0f}/100 (higher = should consider exit)
Score Difference: {hold_score - exit_score:+.0f} (positive = favor HOLD)

Momentum Analysis:
  • 3-bar momentum: {math_analysis.get('momentum_3', 0):+.3f}%
  • 5-bar momentum: {math_analysis.get('momentum_5', 0):+.3f}%
  • 10-bar momentum: {math_analysis.get('momentum_10', 0):+.3f}%
  • Momentum trend: {math_analysis.get('momentum_trend', 'unknown')}

Mean Reversion Analysis:
  • Price vs 20-SMA: {math_analysis.get('price_vs_sma20', 0):+.2f}%
  • Bollinger position: {math_analysis.get('bb_position', 'middle')}
  • Mean reversion probability: {math_analysis.get('mean_reversion_prob', 50):.0f}%

Volatility & Risk:
  • Current volatility: {volatility:.2f}%
  • ATR distance to SL: {math_analysis.get('atr_to_sl', 0):.1f}x ATR
  • Risk/reward at current price: {math_analysis.get('current_rr', 0):.2f}

═══════════════════════════════════════════════════════════════════════════════
📍 SUPPORT/RESISTANCE LEVELS
═══════════════════════════════════════════════════════════════════════════════
{sr_context}

CRITICAL FOR EXITS:
  • LONG near resistance → Higher chance of rejection, consider exit
  • LONG near support → Support may hold, can hold
  • SHORT near support → Higher chance of bounce, consider exit
  • SHORT near resistance → Resistance may reject, can hold

═══════════════════════════════════════════════════════════════════════════════
🔮 MARKET CONDITIONS
═══════════════════════════════════════════════════════════════════════════════
Trend: {trend_status.upper()}
RSI: {rsi_status}
Volume: {volume_ratio:.1f}x average {'(HIGH ACTIVITY)' if volume_ratio > 1.5 else '(normal)'}

Reversal Warning Signals:
{reversal_text}

{'🚨🚨🚨 PROFIT REVERSAL ALERT 🚨🚨🚨' if profit_reversal_detected else ''}
{f'URGENCY: {reversal_urgency.upper()}' if profit_reversal_detected else ''}
{f'REASON: {reversal_reason}' if profit_reversal_detected else ''}
{f'REVERSAL SCORE: {reversal_score}/100 (55+=CRITICAL, 40+=HIGH)' if reversal_score > 0 else ''}
{f'The code detected momentum reversing against our profitable position!' if profit_reversal_detected else ''}
{f'YOU MUST DECIDE: Capture this profit NOW or let it potentially evaporate?' if profit_reversal_detected else ''}

═══════════════════════════════════════════════════════════════════════════════
💪 MOMENTUM HOPE ANALYSIS (Reasons to HOLD)
═══════════════════════════════════════════════════════════════════════════════
Hope Score: {momentum_hope}/50 (higher = more reason to hold through dip)
{chr(10).join(['  • ' + f for f in (momentum_hope_factors or [])]) if momentum_hope_factors else '  • No favorable momentum factors detected'}

{'⚡ STRONG HOPE: Momentum is still WITH us despite the dip!' if momentum_hope >= 30 else ''}
{'📊 MODERATE HOPE: Some favorable factors present' if 15 <= momentum_hope < 30 else ''}
{'⚠️ LOW HOPE: Few reasons to expect recovery' if 0 < momentum_hope < 15 else ''}

IMPORTANT: If REVERSAL SCORE is high but HOPE SCORE is also high, this is a critical decision point!
- High reversal + Low hope = EXIT to protect profit
- High reversal + High hope = Consider holding if trend is intact
- Low reversal + High hope = HOLD, let it run

═══════════════════════════════════════════════════════════════════════════════
🧠 DECISION FRAMEWORK
═══════════════════════════════════════════════════════════════════════════════
IMPORTANT CONSIDERATIONS:

FOR POSITIONS IN LOSS:
  • Small loss (< -0.5%): Normal noise - HOLD if momentum supports our direction
  • Medium loss (-0.5% to -1.0%): Check momentum and mean reversion potential
  • Large loss (> -1.0%): Analyze carefully - exit if no recovery signals

FOR POSITIONS IN PROFIT:
  • Small profit (< +0.5%): LET IT RUN - only exit on VERY strong reversal (score>70)
  • Medium profit (+0.5% to +1.0%): Can tighten SL but don't exit prematurely
  • Good profit (> +1.0%): Protect actively, exit on clear reversal signals
  
{'⚡ PROFIT REVERSAL DETECTED - Weight this heavily in your decision!' if profit_reversal_detected else ''}

KEY QUESTION: Based on momentum, mean reversion, and volatility analysis:
Is the current price move likely to CONTINUE against us, or is this a temporary fluctuation?

═══════════════════════════════════════════════════════════════════════════════
📋 YOUR TASK
═══════════════════════════════════════════════════════════════════════════════
Analyze all the math and decide:

1. should_exit: true/false
2. confidence: 0.0 to 1.0 (be honest - only high confidence for clear signals)
3. reasoning: Brief explanation (max 80 chars)
4. suggested_action: One of:
   - "hold": Keep position, math favors our direction
   - "exit": Close now, clear risk of further loss
   - "tighten_sl": Keep but move SL to protect (for profits)
   - "wait_for_bounce": In loss but mean reversion likely, give it a few bars

Respond ONLY with JSON:
{{"should_exit": bool, "confidence": 0.0-1.0, "reasoning": "...", "suggested_action": "hold|exit|tighten_sl|wait_for_bounce"}}"""

            result_text = self._generate_content(prompt)
            if not result_text:
                return None
            
            result_text = result_text.strip()
            
            # Parse JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            ai_result = json.loads(result_text.strip())
            
            should_exit = ai_result.get("should_exit", False)
            confidence = ai_result.get("confidence", 0.5)
            reasoning = ai_result.get("reasoning", "")
            suggested_action = ai_result.get("suggested_action", "hold")
            
            logger.info(f"🤖 AI Exit Decision: {suggested_action.upper()} ({confidence:.0%}) - {reasoning}")
            
            return {
                "should_exit": should_exit,
                "confidence": confidence,
                "reasoning": reasoning,
                "suggested_action": suggested_action
            }
            
        except Exception as e:
            logger.warning(f"AI smart exit decision failed: {e}")
            return None
    
    def _calculate_deep_position_metrics(
        self, 
        df: pd.DataFrame, 
        side: str, 
        entry_price: float, 
        current_price: float,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate deep mathematical metrics for position analysis."""
        metrics = {
            'momentum_3': 0,
            'momentum_5': 0,
            'momentum_10': 0,
            'momentum_trend': 'unknown',
            'price_vs_sma20': 0,
            'bb_position': 'middle',
            'mean_reversion_prob': 50,
            'atr_to_sl': 0,
            'current_rr': 0
        }
        
        try:
            if df is None or len(df) < 20:
                return metrics
            
            closes = df['close'].values if 'close' in df.columns else None
            if closes is None or len(closes) < 20:
                return metrics
            
            current = closes[-1]
            
            # Momentum calculations
            if len(closes) >= 3:
                metrics['momentum_3'] = (current / closes[-3] - 1) * 100
            if len(closes) >= 5:
                metrics['momentum_5'] = (current / closes[-5] - 1) * 100
            if len(closes) >= 10:
                metrics['momentum_10'] = (current / closes[-10] - 1) * 100
            
            # Determine momentum trend
            m3, m5, m10 = metrics['momentum_3'], metrics['momentum_5'], metrics['momentum_10']
            if side == 'LONG':
                if m3 > 0 and m5 > 0:
                    metrics['momentum_trend'] = 'favorable (bullish)'
                elif m3 < 0 and m5 < 0:
                    metrics['momentum_trend'] = 'against us (bearish)'
                else:
                    metrics['momentum_trend'] = 'mixed'
            else:  # SHORT
                if m3 < 0 and m5 < 0:
                    metrics['momentum_trend'] = 'favorable (bearish)'
                elif m3 > 0 and m5 > 0:
                    metrics['momentum_trend'] = 'against us (bullish)'
                else:
                    metrics['momentum_trend'] = 'mixed'
            
            # SMA and mean reversion
            sma20 = np.mean(closes[-20:])
            metrics['price_vs_sma20'] = (current / sma20 - 1) * 100
            
            # Bollinger Band position
            std20 = np.std(closes[-20:])
            upper_bb = sma20 + 2 * std20
            lower_bb = sma20 - 2 * std20
            
            if current > upper_bb:
                metrics['bb_position'] = 'above upper band (overbought)'
                metrics['mean_reversion_prob'] = 70 if side == 'SHORT' else 30
            elif current < lower_bb:
                metrics['bb_position'] = 'below lower band (oversold)'
                metrics['mean_reversion_prob'] = 70 if side == 'LONG' else 30
            elif current > sma20:
                metrics['bb_position'] = 'above SMA (bullish zone)'
                metrics['mean_reversion_prob'] = 55 if side == 'SHORT' else 45
            else:
                metrics['bb_position'] = 'below SMA (bearish zone)'
                metrics['mean_reversion_prob'] = 55 if side == 'LONG' else 45
            
            # ATR-based metrics
            atr = context.get('atr', current * 0.01)
            if atr > 0:
                # How many ATRs from entry
                distance_from_entry = abs(current - entry_price)
                metrics['atr_to_sl'] = distance_from_entry / atr
            
            # Current risk/reward estimate
            pnl_pct = ((current - entry_price) / entry_price * 100) if side == 'LONG' else ((entry_price - current) / entry_price * 100)
            if pnl_pct < 0:
                # In loss - R/R is potential upside vs current drawdown
                potential_upside = 1.5  # Assume TP at 1.5%
                metrics['current_rr'] = potential_upside / abs(pnl_pct) if abs(pnl_pct) > 0 else 10
            else:
                # In profit - R/R is current profit vs SL distance
                sl_distance = 1.5  # Assume 1.5% SL
                metrics['current_rr'] = pnl_pct / sl_distance if sl_distance > 0 else 1
            
        except Exception as e:
            logger.debug(f"Error calculating deep metrics: {e}")
        
        return metrics

    def monitor_position(
        self,
        df: pd.DataFrame,
        position_side: str,  # "LONG" or "SHORT"
        entry_price: float,
        current_price: float,
        atr: float,
        symbol: str,
        unrealized_pnl_pct: float,
        tp1_hit: bool = False,
        tp2_hit: bool = False,
        position_size: float = 0.0  # Actual position size in base currency
    ) -> Dict[str, Any]:
        """
        Monitor an open position and decide whether to hold or close.
        Uses PhD-level math analysis to make the decision.
        SPEED: Target < 50ms for math, < 1.5s with AI validation.
        
        Returns:
            {
                "action": "hold" | "close" | "partial_close",
                "confidence": 0.0-1.0,
                "reasoning": str,
                "math_score": float
            }
        """
        import time
        start_time = time.perf_counter()
        
        try:
            if df is None or len(df) < 20:
                return {"action": "hold", "confidence": 0.5, "reasoning": "Insufficient data", "math_score": 50}
            
            context = self._build_market_context(df, current_price, atr)
            context_time = (time.perf_counter() - start_time) * 1000
            
            # Get math scores for both directions
            position_signal = 1 if position_side == "LONG" else -1
            opposite_signal = -position_signal
            
            hold_check = self._comprehensive_math_check(position_signal, df, current_price, atr, context)
            exit_check = self._comprehensive_math_check(opposite_signal, df, current_price, atr, context)
            
            hold_score = hold_check.get('score', 50)
            exit_score = exit_check.get('score', 50)
            
            # === GRADUATED PROFIT PROTECTION ===
            # Balance between letting profits run to TP1 vs protecting meaningful gains
            # 
            # Philosophy:
            # - Small profit (<$50): Be patient, let it develop toward TP1
            # - Medium profit ($50-100): Light protection, still favor holding
            # - Good profit ($100-200): Real protection, AI/math can decide
            # - Large profit ($200+): Strong protection, don't lose it!
            
            profit_protection_boost = 0
            patience_penalty = 0
            profit_protection_reason = ""
            
            # Calculate unrealized PnL in dollars using actual position value
            position_notional = position_size * entry_price if position_size > 0 else 500  # Default to $500 if unknown
            estimated_pnl_usd = abs(unrealized_pnl_pct * position_notional / 100)
            
            # Track peak profit for this position
            position_key = f"{symbol}_{position_side}"
            if not hasattr(self, '_peak_profits'):
                self._peak_profits = {}
            
            current_peak = self._peak_profits.get(position_key, 0)
            if estimated_pnl_usd > current_peak:
                self._peak_profits[position_key] = estimated_pnl_usd
                current_peak = estimated_pnl_usd
            
            # === GRADUATED APPROACH ===
            if tp1_hit:
                # TP1 already hit - position has proven itself, protect remaining profit
                if unrealized_pnl_pct >= 1.5:
                    profit_protection_boost = 25
                    profit_protection_reason = f"TP1 HIT + HIGH PROFIT (+{unrealized_pnl_pct:.2f}%)"
                elif unrealized_pnl_pct >= 1.0:
                    profit_protection_boost = 20
                    profit_protection_reason = f"TP1 HIT + PROFIT (+{unrealized_pnl_pct:.2f}%)"
                elif unrealized_pnl_pct >= 0.5:
                    profit_protection_boost = 15
                    profit_protection_reason = f"TP1 hit - protecting (+{unrealized_pnl_pct:.2f}%)"
            else:
                # TP1 NOT hit yet - use graduated protection based on dollar profit
                if estimated_pnl_usd >= 200:
                    # $200+ profit - STRONG protection, don't let it evaporate!
                    profit_protection_boost = 25
                    patience_penalty = 0  # No patience penalty for large profits
                    profit_protection_reason = f"LARGE PROFIT (${estimated_pnl_usd:.0f}) - protect it!"
                elif estimated_pnl_usd >= 100:
                    # $100-200 profit - MODERATE protection, let AI/math decide
                    profit_protection_boost = 15
                    patience_penalty = 5  # Small patience penalty
                    profit_protection_reason = f"GOOD PROFIT (${estimated_pnl_usd:.0f}) - consider protecting"
                elif estimated_pnl_usd >= 50:
                    # $50-100 profit - LIGHT protection, still favor holding for TP1
                    profit_protection_boost = 5
                    patience_penalty = 10
                    profit_protection_reason = f"Building profit (${estimated_pnl_usd:.0f})"
                elif unrealized_pnl_pct > 0:
                    # <$50 profit - BE PATIENT, wait for TP1
                    patience_penalty = 15
                    logger.debug(f"⏳ Small profit ${estimated_pnl_usd:.0f} - patience penalty -{patience_penalty}")
                else:
                    # In the red - standard patience
                    patience_penalty = 10
            
            # Profit erosion detection: protect if we've lost significant gains
            if current_peak > 100 and estimated_pnl_usd < current_peak * 0.5:
                # Lost 50%+ of peak profit that was >$100
                erosion_boost = 20
                if erosion_boost > profit_protection_boost:
                    profit_protection_boost = erosion_boost
                    profit_protection_reason = f"PROFIT EROSION (was ${current_peak:.0f}, now ${estimated_pnl_usd:.0f})"
            
            # === S/R ZONE PENALTY: Exit when near danger zones ===
            sr_penalty_to_hold = 0
            sr_boost_to_exit = 0
            sr_reason = ""
            
            sr_levels = self._detect_resistance_support_levels(df, current_price)
            dist_to_resistance = sr_levels.get('distance_to_resistance_pct', 99)
            dist_to_support = sr_levels.get('distance_to_support_pct', 99)
            in_r_zone = sr_levels.get('in_resistance_zone', False)
            in_s_zone = sr_levels.get('in_support_zone', False)
            
            if position_side == 'LONG':
                if in_r_zone or dist_to_resistance < 0.5:
                    # LONG hitting resistance - high exit urgency
                    sr_penalty_to_hold = -25
                    sr_boost_to_exit = 30
                    sr_reason = f"🚫 LONG in/near resistance ({dist_to_resistance:.1f}% away)"
                elif dist_to_resistance < 1.0:
                    # LONG very close to resistance
                    sr_penalty_to_hold = -15
                    sr_boost_to_exit = 20
                    sr_reason = f"⚠️ LONG close to resistance ({dist_to_resistance:.1f}% away)"
                elif dist_to_resistance < 2.0:
                    # LONG approaching resistance
                    sr_penalty_to_hold = -8
                    sr_boost_to_exit = 10
                    sr_reason = f"📍 LONG approaching resistance ({dist_to_resistance:.1f}% away)"
            else:  # SHORT
                if in_s_zone or dist_to_support < 0.5:
                    # SHORT hitting support - high exit urgency
                    sr_penalty_to_hold = -25
                    sr_boost_to_exit = 30
                    sr_reason = f"🚫 SHORT in/near support ({dist_to_support:.1f}% away)"
                elif dist_to_support < 1.0:
                    # SHORT very close to support
                    sr_penalty_to_hold = -15
                    sr_boost_to_exit = 20
                    sr_reason = f"⚠️ SHORT close to support ({dist_to_support:.1f}% away)"
                elif dist_to_support < 2.0:
                    # SHORT approaching support
                    sr_penalty_to_hold = -8
                    sr_boost_to_exit = 10
                    sr_reason = f"📍 SHORT approaching support ({dist_to_support:.1f}% away)"
            
            # Apply S/R adjustments
            if sr_penalty_to_hold != 0:
                hold_score += sr_penalty_to_hold
                exit_score += sr_boost_to_exit
                logger.info(f"{sr_reason} | Hold penalty: {sr_penalty_to_hold}, Exit boost: +{sr_boost_to_exit}")
            
            # Apply profit protection adjustments
            exit_score += profit_protection_boost - patience_penalty
            
            if profit_protection_boost > 0:
                logger.info(f"💰 {profit_protection_reason} | Exit boost: +{profit_protection_boost} → Exit:{exit_score:.0f}")
            
            # Calculate position health metrics
            pnl_factor = 0
            if unrealized_pnl_pct > 2.0:
                pnl_factor = 15  # Good profit, slight bias to hold
            elif unrealized_pnl_pct > 1.0:
                pnl_factor = 10
            elif unrealized_pnl_pct > 0.5:
                pnl_factor = 5  # Getting close to TP1, lean toward hold
            elif unrealized_pnl_pct < -1.5:
                pnl_factor = -20  # Losing, bias toward exit
            elif unrealized_pnl_pct < -0.5:
                pnl_factor = -10
            
            # Bonus for hitting TPs
            tp_bonus = 0
            if tp2_hit:
                tp_bonus = 10  # Already took profit, can be more aggressive
            elif tp1_hit:
                tp_bonus = 5
            
            # TP proximity bonus - removed in graduated approach (handled by profit tiers)
            tp_proximity_bonus = 0
            
            # Final adjusted hold score includes: base hold + pnl factor + tp bonus
            adjusted_hold_score = hold_score + pnl_factor + tp_bonus + tp_proximity_bonus
            
            # Decision thresholds - RAISED to prevent premature exits
            STRONG_EXIT_THRESHOLD = 80  # Was 70 - need stronger signal to exit
            WEAK_HOLD_THRESHOLD = 35    # Was 40 - more tolerant of weak holds
            
            reasons_for_hold = hold_check.get('reasons_for', [])[:2]
            reasons_for_exit = exit_check.get('reasons_for', [])[:2]
            
            action = "hold"
            reasoning = ""
            confidence = 0.5
            
            # Decision logic - MODIFIED to be more patient
            if exit_score >= STRONG_EXIT_THRESHOLD and exit_score > adjusted_hold_score + 20:  # Was +15
                # Strong reversal signal - but require higher margin
                action = "close"
                confidence = exit_score / 100
                reasoning = f"Strong reversal signal (Exit:{exit_score:.0f} vs Hold:{adjusted_hold_score:.0f}). {', '.join(reasons_for_exit)}"
            
            elif adjusted_hold_score < WEAK_HOLD_THRESHOLD and unrealized_pnl_pct < -1.0:  # Was < 0
                # Weak hold + SIGNIFICANT loss = exit (not just any loss)
                action = "close"
                confidence = (100 - adjusted_hold_score) / 100
                reasoning = f"Weak hold score ({adjusted_hold_score:.0f}) + losing position ({unrealized_pnl_pct:+.2f}%)"
            
            elif adjusted_hold_score < 45 and unrealized_pnl_pct > 2.0 and tp1_hit:  # Was > 1.0, no TP1 check
                # Weak hold but good profit AND TP1 already hit = take remaining profit
                action = "close"
                confidence = 0.7
                reasoning = f"Weakening after TP1 (score:{adjusted_hold_score:.0f}) - securing {unrealized_pnl_pct:+.2f}% remaining"
            
            elif tp2_hit and adjusted_hold_score < 55:  # Was < 60
                # TP2 hit and momentum fading = close remaining
                action = "close"
                confidence = 0.65
                reasoning = f"TP2 hit + fading momentum (score:{adjusted_hold_score:.0f})"
            
            else:
                # Hold position
                action = "hold"
                confidence = adjusted_hold_score / 100
                reasoning = f"Position healthy (Hold:{adjusted_hold_score:.0f}, Exit:{exit_score:.0f}). {', '.join(reasons_for_hold)}"
            
            # === AI VALIDATION (optional, math is primary) ===
            ai_validation = None
            final_action = action
            final_confidence = confidence
            final_reasoning = reasoning
            
            # Only call AI for borderline cases to save API calls
            is_borderline = (
                (action == "hold" and adjusted_hold_score < 55) or  # Weak hold
                (action == "close" and confidence < 0.7) or  # Uncertain close
                (abs(hold_score - exit_score) < 15)  # Close scores
            )
            
            if self.use_ai and is_borderline:
                ai_validation = self._ai_validate_position_decision(
                    position_side=position_side,
                    entry_price=entry_price,
                    current_price=current_price,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    math_action=action,
                    math_score=adjusted_hold_score,
                    hold_score=hold_score,
                    exit_score=exit_score,
                    math_reasoning=reasoning,
                    symbol=symbol
                )
                
                if ai_validation:
                    validation_type = ai_validation.get("validation", "confirm")
                    
                    if validation_type == "override":
                        # AI wants to override - use graduated logic based on profit
                        ai_action = ai_validation.get("action", action)
                        
                        # Calculate estimated profit in dollars
                        est_profit_usd = abs(unrealized_pnl_pct * 23000 / 100)
                        
                        # GRADUATED AI OVERRIDE RULES:
                        # - $200+ profit: ALWAYS allow AI to protect (close or hold)
                        # - $100-200: Allow AI override if it wants to CLOSE (protect profit)
                        # - $50-100: Allow only if math is truly borderline (45-55)
                        # - <$50: Block AI override to close, let position develop
                        
                        should_accept = False
                        block_reason = ""
                        
                        if est_profit_usd >= 200:
                            # Large profit - trust AI judgment fully
                            should_accept = True
                            logger.info(f"💰 Large profit ${est_profit_usd:.0f} - AI override allowed")
                        elif est_profit_usd >= 100 and ai_action == "close":
                            # Good profit and AI wants to protect it
                            should_accept = True
                            logger.info(f"💰 Good profit ${est_profit_usd:.0f} - AI protection allowed")
                        elif est_profit_usd >= 50 and 45 <= adjusted_hold_score <= 55:
                            # Medium profit, truly borderline math
                            should_accept = True
                        elif est_profit_usd < 50 and ai_action == "close":
                            # Small profit - don't let AI close prematurely
                            block_reason = f"Profit too small (${est_profit_usd:.0f}). Wait for development."
                        elif 40 <= adjusted_hold_score <= 60:
                            # Standard borderline case
                            should_accept = True
                        else:
                            block_reason = f"Math score {adjusted_hold_score:.0f} outside borderline range"
                        
                        if should_accept:
                            final_action = ai_action
                            final_confidence = min(1.0, max(0.4, confidence + ai_validation.get("confidence_adjustment", 0)))
                            final_reasoning = f"{reasoning} [AI override: {ai_validation.get('note', '')}]"
                            logger.info(f"⚠️ AI override accepted: {action} → {final_action}")
                        else:
                            logger.info(f"🛡️ AI override BLOCKED - {block_reason}")
                    
                    elif validation_type == "refine":
                        # AI refines confidence
                        final_confidence = min(1.0, max(0.4, confidence + ai_validation.get("confidence_adjustment", 0)))
                        final_reasoning = f"{reasoning} [AI: {ai_validation.get('note', '')}]"
                    
                    # "confirm" - no changes needed
            
            # Calculate total time
            total_time = (time.perf_counter() - start_time) * 1000
            
            logger.info(
                f"🔍 Position Monitor [{position_side}]: {final_action.upper()} | "
                f"Hold:{adjusted_hold_score:.0f} Exit:{exit_score:.0f} | "
                f"PnL:{unrealized_pnl_pct:+.2f}% | AI:{ai_validation is not None} | ⚡{total_time:.0f}ms | {final_reasoning[:50]}..."
            )
            
            return {
                "action": final_action,
                "confidence": final_confidence,
                "reasoning": final_reasoning,
                "math_score": adjusted_hold_score,
                "hold_score": hold_score,
                "exit_score": exit_score,
                "pnl_factor": pnl_factor,
                "ai_validated": ai_validation is not None,
                "ai_validation": ai_validation.get("validation") if ai_validation else None
            }
            
        except Exception as e:
            logger.error(f"Position monitor error: {type(e).__name__}: {e}")
            return {"action": "hold", "confidence": 0.5, "reasoning": f"Error: {e}", "math_score": 50}
    
    def proactive_scan(
        self,
        df: pd.DataFrame,
        current_price: float,
        atr: float,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Proactively scans market for opportunities using PhD-level mathematical analysis.
        Math is the PRIMARY decision maker. AI only confirms/refines the math decision.
        Returns a trade suggestion or None.
        """
        # Require minimum data
        if df is None or len(df) < 20:
            logger.debug(f"Proactive scan skipped: insufficient data ({len(df) if df is not None else 0}/20 bars)")
            return None
        
        # === PHASE 1: PhD-LEVEL MATHEMATICAL ANALYSIS ===
        # Always run math scan first - it's the foundation
        math_result = self._math_proactive_scan(df, current_price, atr, symbol)
        
        # If math says no opportunity with high confidence, trust it
        if not math_result:
            # Already logged at INFO level in _math_proactive_scan
            return None
        
        # Math found an opportunity - if AI is not available, use math result directly
        if not self.use_ai:
            return math_result
        
        # === PHASE 2: AI VALIDATION OF MATH DECISION ===
        # AI can only CONFIRM or REFINE the math decision, not override it
        try:
            context = self._build_market_context(df, current_price, atr)
            perf = self._get_performance_context()
            market = self._get_market_hours_context()
            
            # Get math details for AI context
            math_score = math_result.get('math_score', 0)
            math_direction = math_result.get('action', 'UNKNOWN')
            math_reasoning = math_result.get('reasoning', '')
            math_confidence = math_result.get('confidence', 0)
            
            prompt = f"""You are validating a MATHEMATICAL TRADING DECISION. The math has already identified an opportunity.

═══════════════════════════════════════════════════════════════════════
MATHEMATICAL ANALYSIS RESULT (PhD-Level - This is the PRIMARY decision)
═══════════════════════════════════════════════════════════════════════
Direction: {math_direction}
Math Score: {math_score:.0f}/100
Confidence: {math_confidence:.0%}
Analysis: {math_reasoning}

═══════════════════════════════════════════════════════════════════════
MARKET DATA for {symbol}
═══════════════════════════════════════════════════════════════════════
Current Price: ${context['current_price']}
1-Hour Change: {context['price_change_1h']}%
5-Min Change: {context['price_change_5m']}%
Volume Ratio: {context['volume_ratio']}x
Trend: {context['trend']}
Volatility (ATR%): {context['volatility_pct']}%
SMA10: ${context['sma_10']} | SMA20: ${context['sma_20']}

═══════════════════════════════════════════════════════════════════════
PERFORMANCE CONTEXT
═══════════════════════════════════════════════════════════════════════
Total Trades: {perf.get('total_trades', 0)}
Win Rate: {perf.get('win_rate', 0)}%
Current Streak: {perf.get('consecutive_wins', 0)}W / {perf.get('consecutive_losses', 0)}L
Session: {market['session']} | Activity: {market['activity_level']}

═══════════════════════════════════════════════════════════════════════
YOUR TASK (VALIDATION ONLY)
═══════════════════════════════════════════════════════════════════════
The MATH has identified a {math_direction} opportunity with {math_score:.0f}/100 score.

You can:
1. CONFIRM: Agree with math (respond with same direction)
2. REFINE: Adjust risk assessment if you see specific concerns
3. VETO: Only if you see a CRITICAL flaw the math missed (rare)

IMPORTANT: Math is the primary decision maker. Only veto if there's a clear mathematical error or critical market condition the algorithm couldn't detect.

Respond ONLY with JSON:
{{"validate": "confirm" or "refine" or "veto", "direction": "{math_direction}", "confidence_adjustment": 0.0 to 0.1 (add or subtract), "risk_assessment": "low/medium/high", "note": "brief reason"}}"""

            result_text = self._generate_content(prompt)
            if not result_text:
                # AI failed, use math result
                return math_result
            
            result_text = result_text.strip()
            
            # Parse JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            ai_validation = json.loads(result_text.strip())
            
            validation = ai_validation.get("validate", "confirm")
            
            if validation == "veto":
                # AI vetoed - log it but still require strong justification
                veto_reason = ai_validation.get("note", "No reason given")
                logger.warning(f"⚠️ AI vetoed math decision: {veto_reason}")
                # Only accept veto if math score wasn't very high
                if math_score >= 80:
                    logger.info(f"🛡️ Math score {math_score:.0f} is high - overriding AI veto")
                    return math_result
                return None
            
            # Confirm or refine - use math result with possible adjustments
            confidence_adj = ai_validation.get("confidence_adjustment", 0)
            final_confidence = min(1.0, max(0.5, math_confidence + confidence_adj))
            
            risk_assessment = ai_validation.get("risk_assessment", math_result.get("risk_assessment", "medium"))
            
            logger.info(
                f"🤖📊 MATH+AI OPPORTUNITY: {math_direction} {symbol} | "
                f"Math: {math_score:.0f}/100 | AI: {validation} | "
                f"Final Conf: {final_confidence:.0%}"
            )
            
            return {
                "action": math_direction,
                "signal": 1 if math_direction == "LONG" else -1,
                "confidence": final_confidence,
                "reasoning": f"Math({math_score:.0f}/100): {math_reasoning}. AI: {validation}",
                "risk_assessment": risk_assessment,
                "suggested_risk_pct": math_result.get("suggested_risk_pct", 0.02),
                "source": "math_ai_combined",
                "math_score": math_score
            }
            
        except Exception as e:
            logger.error(f"AI validation error: {e} - using math result")
            return math_result

    async def chat(self, user_message: str, trading_context: str = "") -> str:
        """
        Chat with the AI about trading, market analysis, or general questions.
        Maintains conversation history for context across messages.
        
        Args:
            user_message: The user's message
            trading_context: Current trading context (positions, balance, etc.)
            
        Returns:
            AI response string
        """
        logger.info(f"💬 CHAT: Received message: '{user_message[:50]}...'")
        logger.info(f"💬 CHAT: AI Provider={self.ai_provider}, use_ai={self.use_ai}, client={self.client is not None}, model={self.model is not None}")
        
        # Check for AI availability - Gemini SDK (client or legacy model)
        if not self.use_ai or (not self.client and not self.model):
            logger.warning("AI chat unavailable - using simple response")
            return self._simple_chat_response(user_message)
        
        try:
            # Build conversation history context
            history_text = ""
            if self.chat_history:
                history_text = "\n\nRecent Conversation History:\n"
                for msg in self.chat_history[-10:]:  # Last 10 exchanges
                    role = "Trader" if msg["role"] == "user" else "Julaba"
                    history_text += f"{role}: {msg['content']}\n"
            
            # Build trade performance summary
            trade_summary = ""
            if self.total_wins + self.total_losses > 0:
                win_rate = self.total_wins / (self.total_wins + self.total_losses) * 100
                trade_summary = f"\n\nMy Trade Performance: {self.total_wins}W/{self.total_losses}L ({win_rate:.1f}% win rate)"
                if self.consecutive_wins > 0:
                    trade_summary += f" | Current streak: {self.consecutive_wins} wins 🔥"
                elif self.consecutive_losses > 0:
                    trade_summary += f" | Current streak: {self.consecutive_losses} losses 📉"
            
            # Extract current AI mode from trading context
            current_ai_mode = 'unknown'
            if trading_context:
                import re
                mode_match = re.search(r'AI Mode:\s*(\w+)', trading_context, re.IGNORECASE)
                if mode_match:
                    current_ai_mode = mode_match.group(1).lower()
            
            prompt = f"""You are Julaba, a smart AI trading assistant with PhD-level quant skills. Be helpful, show real numbers, use emojis.

MODE: {current_ai_mode.upper()} — {'User must confirm trades via execute button' if current_ai_mode == 'advisory' else 'Bot trades autonomously when math criteria met' if current_ai_mode == 'autonomous' else 'Only validate signals' if current_ai_mode == 'filter' else 'Suggest but never auto-execute' if current_ai_mode == 'hybrid' else 'Unknown'}

RULES:
- Always show actual numbers/prices/percentages from CURRENT STATUS below
- For status: show Position, Entry, PnL, Balance, Mode, Market regime, Score, ADX, RSI
- For analysis: include Market Regime, Key Indicators (ADX/RSI/Volume), Score, Verdict (TRADE/NO TRADE), 2-3 reasons
- For actions: ALWAYS include the ```command``` block with valid JSON inside it
- In ADVISORY mode: You MUST ALWAYS include the ```command``` block even if you recommend against the trade. The user needs the execute button to decide. Never skip it. Never say "I cannot execute" — just provide the command block and let the user decide.
- Match response depth to question: simple=short, analysis=detailed
- Never lazy one-liners. Never vague. Never trail off

COMMANDS (use ```command``` blocks):
Trade: {{"action":"open_trade","side":"long/short"}} or {{"action":"open_trade","side":"long","symbol":"ETHUSDT"}}
Close: {{"action":"close_trade"}} or {{"action":"close_trade","symbol":"ETHUSDT"}}
Settings: {{"action":"set_param","param":"PARAM","value":VAL}} — params: risk_pct, ai_confidence, atr_mult, tp1_r, tp2_r, tp3_r, ai_mode, paused, proactive_threshold
Switch: {{"action":"switch_symbol","symbol":"SOL"}}
Slash commands: /status /positions /market /analyze /chart /ml_stats /risk /help

SYSTEM: Bybit futures via CCXT | ATR-based SL + 3-tier TP | Up to 2 positions on different symbols | Scanner: 50+ pairs, score=60% math+40% AI, threshold 40 | ML classifier learns from trades | Pair switch only if: score diff>15, new score>=60, ADX>=25, no open trade on old pair

CURRENT STATUS:
{trading_context if trading_context else "No trading context available"}
{trade_summary}
{history_text if history_text else ""}

Trader: "{user_message}"
"""

            # Run blocking API call in thread pool to avoid blocking event loop
            # Chat uses longer timeout (60s) and higher temperature (0.5) for engaging responses
            loop = asyncio.get_event_loop()
            ai_response = await loop.run_in_executor(None, self._generate_content, prompt, 0, 60000, 0.5)
            
            # Retry once on timeout/failure with even longer timeout
            if not ai_response:
                logger.warning("💬 CHAT: First attempt failed, retrying with 90s timeout...")
                await asyncio.sleep(2)  # Brief backoff
                ai_response = await loop.run_in_executor(None, self._generate_content, prompt, 0, 90000, 0.5)
            
            if not ai_response:
                return self._simple_chat_response(user_message)
            ai_response = ai_response.strip()
            
            # === STRICT MATHEMATICAL COMMAND VALIDATION ===
            # Commands require explicit action verbs - no guessing, no interpretation
            import re
            
            user_msg_lower = user_message.lower().strip()
            
            # Action verbs that indicate user wants to change something
            action_verbs = [
                'change', 'set', 'switch', 'reduce', 'increase', 'adjust', 
                'make', 'put', 'raise', 'lower', 'modify', 'update', 'enable',
                'disable', 'turn on', 'turn off', 'activate', 'deactivate',
                'open', 'close', 'exit', 'buy', 'sell', 'go long', 'go short', 'pause', 'resume',
                'execute', 'confirm', 'proceed', 'do it', 'yes'
            ]
            
            has_action_verb = any(verb in user_msg_lower for verb in action_verbs)
            
            # Acknowledgements - NEVER trigger commands
            acknowledgement_words = [
                'okay', 'ok', 'yes', 'got it', 'thanks', 'thank you', 'alright', 
                'cool', 'noted', 'understood', 'sure', 'fine', 'great', 'good',
                'nice', 'perfect', 'awesome', 'kk', 'k', 'yep', 'yup', 'right',
                'i see', 'makes sense', 'roger', 'copy', 'affirmative', 'hmm',
                'ah', 'oh', 'hm', 'interesting', 'wow', 'lol', 'haha'
            ]
            
            is_acknowledgement = (
                user_msg_lower in acknowledgement_words or 
                (len(user_msg_lower) < 12 and any(ack == user_msg_lower for ack in acknowledgement_words))
            )
            
            # Questions without action verbs don't trigger commands
            is_pure_question = (
                any(q in user_msg_lower for q in ['what ', 'how ', 'why ', 'when ', 'where ', '?']) 
                and not has_action_verb
            )
            
            # User explicitly asking for/about commands - don't strip!
            asking_for_command = any(phrase in user_msg_lower for phrase in [
                'command', 'show me', 'give me', 'what is the', 'how do i',
                'reset', 'daily loss', 'daily_loss', 'override', 'halt'
            ])
            
            # MATHEMATICAL RULE: Strip commands unless explicit action is requested
            # CRITICAL: If telegram_bot injected MANUAL TRADE REQUEST or ADVISORY MODE markers,
            # NEVER strip — telegram_bot.py needs command blocks for /execute buttons
            is_manual_trade_request = (
                'MANUAL TRADE REQUEST' in trading_context or
                'ADVISORY MODE' in trading_context or
                'USER MANUAL CLOSE REQUEST' in trading_context or
                'AUTONOMOUS MODE ACTIVE' in trading_context
            )
            
            if '```command' in ai_response:
                should_strip = False
                reason = ""
                
                if is_manual_trade_request:
                    should_strip = False  # telegram_bot needs commands for execute buttons!
                    logger.info(f"🛡️ MATH GUARD: Keeping command block — manual trade request detected")
                elif is_acknowledgement:
                    should_strip = True
                    reason = f"acknowledgement '{user_message}'"
                elif asking_for_command:
                    should_strip = False  # User wants to see the command!
                elif not has_action_verb:
                    should_strip = True
                    reason = f"no action verb in '{user_message[:30]}'"
                elif is_pure_question:
                    should_strip = True
                    reason = f"question without action"
                elif len(user_msg_lower) < 5:
                    should_strip = True
                    reason = f"message too short to be a command"
                
                if should_strip:
                    ai_response = re.sub(r'```command\s*\n?\{[^}]+\}\s*\n?```', '', ai_response)
                    ai_response = ai_response.strip()
                    logger.info(f"🛡️ MATH GUARD: Stripped unauthorized command - {reason}")
            
            # === COMPREHENSIVE ANTI-HALLUCINATION CHECK ===
            # Detect if AI is making claims that contradict the trading_context
            response_lower = ai_response.lower()
            hallucinations_detected = []
            
            # 1. POSITION HALLUCINATION
            position_claim_phrases = [
                "i have opened", "i opened", "i've opened", "position is open",
                "we have a", "we are in a", "holding a long", "holding a short",
                "current position", "your position", "our position"
            ]
            claims_position = any(phrase in response_lower for phrase in position_claim_phrases)
            no_actual_position = "POSITION: **NONE**" in trading_context or "POSITION: None" in trading_context
            if claims_position and no_actual_position:
                hallucinations_detected.append("position")
                logger.warning("🚨 AI hallucinated: claimed position but none exists!")
            
            # 2. WRONG SYMBOL CLAIMS
            # Extract actual symbol from context
            import re
            symbol_match = re.search(r'Symbol:\s*(\w+)', trading_context)
            actual_symbol = symbol_match.group(1) if symbol_match else None
            if actual_symbol:
                actual_base = actual_symbol.replace('USDT', '').lower()
                wrong_symbol_phrases = [
                    ("trading btc", "btc"), ("trading eth", "eth"), ("trading sol", "sol"),
                    ("on btc", "btc"), ("on eth", "eth"), ("on sol", "sol"),
                    ("trading link", "link"), ("trading tia", "tia"), ("trading inj", "inj")
                ]
                for phrase, sym in wrong_symbol_phrases:
                    if phrase in response_lower and sym != actual_base:
                        hallucinations_detected.append(f"symbol (said {sym.upper()}, actual {actual_symbol})")
                        logger.warning(f"🚨 AI hallucinated: said trading {sym.upper()} but actual is {actual_symbol}")
                        break
            
            # 3. BALANCE HALLUCINATION (check for wildly wrong numbers)
            balance_match = re.search(r'Balance:\s*\$([0-9,]+(?:\.[0-9]+)?)', trading_context)
            if balance_match:
                actual_balance = float(balance_match.group(1).replace(',', ''))
                # Find balance claims in response
                claimed_balances = re.findall(r'\$([0-9,]+(?:\.[0-9]{2})?)', ai_response)
                for claim in claimed_balances:
                    try:
                        claimed = float(claim.replace(',', ''))
                        if claimed > 1000 and abs(claimed - actual_balance) / actual_balance > 0.25:
                            hallucinations_detected.append(f"balance (claimed ${claimed:,.0f}, actual ${actual_balance:,.0f})")
                            logger.warning(f"🚨 AI hallucinated: claimed ${claimed} but actual is ${actual_balance:.2f}")
                            break
                    except (ValueError, TypeError):
                        pass  # Non-critical: couldn't parse balance claim
            
            # 4. WIN RATE HALLUCINATION
            winrate_match = re.search(r'Win Rate:\s*([0-9.]+)%', trading_context)
            if winrate_match:
                actual_wr = float(winrate_match.group(1))
                claimed_wr_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%\s*win', response_lower)
                for claimed in claimed_wr_matches:
                    try:
                        claimed_wr = float(claimed)
                        if abs(claimed_wr - actual_wr) > 20:  # More than 20% off
                            hallucinations_detected.append(f"win rate (claimed {claimed_wr}%, actual {actual_wr:.1f}%)")
                            logger.warning(f"🚨 AI hallucinated: claimed {claimed_wr}% win rate but actual is {actual_wr:.1f}%")
                            break
                    except (ValueError, TypeError):
                        pass  # Non-critical: couldn't parse win rate claim
            
            # 5. MODE/STATUS HALLUCINATION - Check all modes properly
            actual_mode = None
            import re as _re
            mode_match = _re.search(r'ai mode:\s*(\w+)', trading_context.lower())
            if mode_match:
                actual_mode = mode_match.group(1).lower()
            # Also check the prominent header
            if not actual_mode:
                mode_match2 = _re.search(r'current mode:\s*(\w+)', trading_context.lower())
                if mode_match2:
                    actual_mode = mode_match2.group(1).lower()
            
            if actual_mode:
                claimed_modes = []
                # Check for any mention of being in a specific mode
                for check_mode in ['autonomous', 'advisory', 'filter', 'hybrid']:
                    if check_mode == actual_mode:
                        continue  # Skip the actual mode
                    # Broad pattern matching
                    patterns = [
                        f'{check_mode} mode',
                        f'in {check_mode}',
                        f'i\'m in **{check_mode}',
                        f'currently {check_mode}',
                        f'mode is {check_mode}',
                        f'operating in {check_mode}',
                        f'running in {check_mode}',
                    ]
                    for pat in patterns:
                        if pat in response_lower:
                            claimed_modes.append(check_mode)
                            break
                
                for claimed in claimed_modes:
                    hallucinations_detected.append(f"mode (said {claimed}, actual {actual_mode})")
                    logger.warning(f"🚨 AI MODE HALLUCINATION: Claimed {claimed} but actual mode is {actual_mode}")
            
            if "Paused: False" in trading_context and ("bot is paused" in response_lower or "i am paused" in response_lower):
                hallucinations_detected.append("status (said paused, but bot is running)")
            elif "Paused: True" in trading_context and ("bot is running" in response_lower or "bot is active" in response_lower):
                hallucinations_detected.append("status (said running, but bot is paused)")
            
            # If hallucinations detected, don't save to history and add warning
            if hallucinations_detected:
                logger.warning(f"🚨 AI hallucinations detected: {hallucinations_detected} - NOT saving to history")
                warning = "\n\n⚠️ *Reality Check - Please verify on dashboard:*\n"
                for h in hallucinations_detected[:3]:  # Max 3 corrections
                    warning += f"• Incorrect claim about {h}\n"
                return ai_response + warning
            
            # Save to conversation history - only if not hallucinating
            self.chat_history.append({"role": "user", "content": user_message})
            self.chat_history.append({"role": "assistant", "content": ai_response})
            
            # Keep only recent history (limit to prevent long context)
            if len(self.chat_history) > self.max_chat_history:
                self.chat_history = self.chat_history[-self.max_chat_history:]
            
            # Persist to disk
            self._save_chat_history()
            
            return ai_response
            
        except Exception as e:
            logger.error(f"AI chat error: {e}")
            return self._simple_chat_response(user_message)
    
    def clear_chat_history(self):
        """Clear chat history to prevent false memories."""
        self.chat_history = []
        self._save_chat_history()
        logger.info("Chat history cleared")
    
    def _save_chat_history(self):
        """Save chat history to disk for persistence."""
        try:
            with open(CHAT_HISTORY_FILE, 'w') as f:
                json.dump(self.chat_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")
    
    def _load_chat_history(self):
        """Load chat history from disk."""
        try:
            if CHAT_HISTORY_FILE.exists():
                with open(CHAT_HISTORY_FILE, 'r') as f:
                    data = json.load(f)
                # Handle both old format {"messages": [...]} and new format [...]
                if isinstance(data, dict) and "messages" in data:
                    self.chat_history = data["messages"]
                elif isinstance(data, list):
                    self.chat_history = data
                else:
                    self.chat_history = []
                logger.debug(f"Loaded {len(self.chat_history)} chat history entries")
        except Exception as e:
            logger.error(f"Failed to load chat history: {e}")
            self.chat_history = []
    
    def _simple_chat_response(self, message: str) -> str:
        """Simple rule-based chat responses when AI is not available."""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["hello", "hi", "hey", "sup"]):
            return "👋 Hey there! I'm Julaba, your trading assistant. How can I help you today?"
        
        if any(word in message_lower for word in ["how are you", "how's it going"]):
            return "🤖 I'm running smoothly and watching the markets! How can I help you?"
        
        if any(word in message_lower for word in ["help", "what can you do"]):
            return "🤖 I can help you with:\n\n• Check /status for bot status\n• Use /market for price info\n• See /positions for open trades\n• Try /pnl for profit/loss\n\nOr just chat with me about trading! 📊"
        
        if any(word in message_lower for word in ["thank", "thanks"]):
            return "You're welcome! 😊 Let me know if you need anything else."
        
        if any(word in message_lower for word in ["price", "market", "link"]):
            return "📊 Use /market to see current price, volume, and market data!"
        
        if any(word in message_lower for word in ["trade", "position", "buy", "sell"]):
            return "📈 Check /positions for open trades or /signals for recent trading signals!"
        
        if any(word in message_lower for word in ["profit", "loss", "pnl", "money"]):
            return "💰 Use /pnl to see your profit/loss summary or /balance for your current balance!"
        
        return "🤖 I'm here to help! Try asking about trading, or use /help to see all commands."
