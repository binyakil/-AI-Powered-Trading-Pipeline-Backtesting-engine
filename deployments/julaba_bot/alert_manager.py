"""
Alert System for Julaba Trading Bot
====================================
Price alerts, market news, volatility detection, and custom notifications.

Features:
- Price alerts (crosses above/below, percentage change)
- Volume alerts (unusual volume spikes)
- Volatility alerts (ATR expansion, sudden moves)
- Market news integration (optional)
- Custom Telegram notifications
"""

import json
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import time

logger = logging.getLogger("Julaba.Alerts")

class AlertType(Enum):
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    PRICE_CHANGE_PCT = "price_change_pct"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    DRAWDOWN = "drawdown"
    WIN_STREAK = "win_streak"
    LOSS_STREAK = "loss_streak"
    DAILY_PNL = "daily_pnl"
    NEWS = "news"
    CUSTOM = "custom"

class AlertPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Alert:
    """Alert configuration"""
    id: str
    type: AlertType
    symbol: str = ""
    condition_value: float = 0.0
    priority: AlertPriority = AlertPriority.MEDIUM
    message: str = ""
    enabled: bool = True
    one_time: bool = False  # Delete after trigger
    cooldown_minutes: int = 60  # Don't re-trigger within this time
    last_triggered: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['type'] = self.type.value
        d['priority'] = self.priority.value
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'Alert':
        d['type'] = AlertType(d['type'])
        d['priority'] = AlertPriority(d['priority'])
        return cls(**d)

@dataclass  
class TriggeredAlert:
    """Record of a triggered alert"""
    alert_id: str
    type: str
    symbol: str
    message: str
    priority: str
    triggered_at: str
    current_value: float = 0.0
    condition_value: float = 0.0

class AlertManager:
    """
    Manages price alerts, notifications, and market monitoring
    """
    
    def __init__(self, alerts_path: str = "alerts_config.json",
                 history_path: str = "alerts_history.json"):
        self.alerts_path = Path(alerts_path)
        self.history_path = Path(history_path)
        
        # Load alerts
        self.alerts: Dict[str, Alert] = {}
        self.history: List[TriggeredAlert] = []
        self._load_alerts()
        self._load_history()
        
        # Callbacks
        self._notification_callback: Optional[Callable] = None
        
        # Price cache (symbol -> last price)
        self._price_cache: Dict[str, float] = {}
        self._price_history: Dict[str, List[tuple]] = {}  # symbol -> [(timestamp, price), ...]
        
        # Volume cache
        self._volume_cache: Dict[str, float] = {}
        self._avg_volume: Dict[str, float] = {}
        
        # News cache
        self._news_cache: List[dict] = []
        self._last_news_fetch = 0
        
        logger.info(f"🔔 Alert Manager initialized with {len(self.alerts)} alerts")
    
    def _load_alerts(self):
        """Load alerts from file"""
        try:
            if self.alerts_path.exists():
                with open(self.alerts_path) as f:
                    data = json.load(f)
                    for alert_data in data.get('alerts', []):
                        alert = Alert.from_dict(alert_data)
                        self.alerts[alert.id] = alert
        except Exception as e:
            logger.warning(f"Failed to load alerts: {e}")
    
    def _save_alerts(self):
        """Save alerts to file"""
        try:
            data = {
                'alerts': [a.to_dict() for a in self.alerts.values()],
                'updated_at': datetime.now().isoformat()
            }
            with open(self.alerts_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save alerts: {e}")
    
    def _load_history(self):
        """Load alert history"""
        try:
            if self.history_path.exists():
                with open(self.history_path) as f:
                    data = json.load(f)
                    self.history = [TriggeredAlert(**h) for h in data.get('history', [])]
        except Exception as e:
            logger.warning(f"Failed to load alert history: {e}")
    
    def _save_history(self):
        """Save alert history"""
        try:
            data = {
                'history': [asdict(h) for h in self.history[-1000:]],  # Keep last 1000
                'updated_at': datetime.now().isoformat()
            }
            with open(self.history_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save alert history: {e}")
    
    # ==================== ALERT MANAGEMENT ====================
    
    def set_notification_callback(self, callback: Callable):
        """Set callback for sending notifications (e.g., Telegram)"""
        self._notification_callback = callback
    
    def add_alert(self, alert_type: AlertType, symbol: str = "",
                  condition_value: float = 0.0, message: str = "",
                  priority: AlertPriority = AlertPriority.MEDIUM,
                  one_time: bool = False, cooldown_minutes: int = 60) -> str:
        """Add a new alert"""
        alert_id = f"{alert_type.value}_{symbol}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            type=alert_type,
            symbol=symbol.upper(),
            condition_value=condition_value,
            message=message or f"{alert_type.value} alert for {symbol}",
            priority=priority,
            one_time=one_time,
            cooldown_minutes=cooldown_minutes
        )
        
        self.alerts[alert_id] = alert
        self._save_alerts()
        
        logger.info(f"🔔 Alert added: {alert_id}")
        return alert_id
    
    def remove_alert(self, alert_id: str) -> bool:
        """Remove an alert"""
        if alert_id in self.alerts:
            del self.alerts[alert_id]
            self._save_alerts()
            logger.info(f"🔕 Alert removed: {alert_id}")
            return True
        return False
    
    def enable_alert(self, alert_id: str, enabled: bool = True):
        """Enable or disable an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].enabled = enabled
            self._save_alerts()
    
    def get_alerts(self, symbol: str = None, alert_type: AlertType = None) -> List[Alert]:
        """Get alerts with optional filtering"""
        alerts = list(self.alerts.values())
        
        if symbol:
            alerts = [a for a in alerts if a.symbol == symbol.upper()]
        if alert_type:
            alerts = [a for a in alerts if a.type == alert_type]
        
        return alerts
    
    # ==================== QUICK ALERT CREATION ====================
    
    def alert_price_above(self, symbol: str, price: float, message: str = None) -> str:
        """Create alert when price goes above target"""
        msg = message or f"🚀 {symbol} crossed above ${price}"
        return self.add_alert(AlertType.PRICE_ABOVE, symbol, price, msg,
                            AlertPriority.MEDIUM, one_time=True)
    
    def alert_price_below(self, symbol: str, price: float, message: str = None) -> str:
        """Create alert when price goes below target"""
        msg = message or f"📉 {symbol} dropped below ${price}"
        return self.add_alert(AlertType.PRICE_BELOW, symbol, price, msg,
                            AlertPriority.MEDIUM, one_time=True)
    
    def alert_price_change(self, symbol: str, pct_change: float, message: str = None) -> str:
        """Create alert when price changes by X%"""
        direction = "up" if pct_change > 0 else "down"
        msg = message or f"⚡ {symbol} moved {abs(pct_change)}% {direction}"
        return self.add_alert(AlertType.PRICE_CHANGE_PCT, symbol, abs(pct_change), msg,
                            AlertPriority.HIGH, cooldown_minutes=30)
    
    def alert_volume_spike(self, symbol: str, multiplier: float = 3.0, message: str = None) -> str:
        """Create alert when volume spikes above X times average"""
        msg = message or f"📊 {symbol} volume spike ({multiplier}x average)"
        return self.add_alert(AlertType.VOLUME_SPIKE, symbol, multiplier, msg,
                            AlertPriority.MEDIUM, cooldown_minutes=60)
    
    def alert_drawdown(self, pct: float, message: str = None) -> str:
        """Create alert when drawdown exceeds threshold"""
        msg = message or f"⚠️ Portfolio drawdown exceeded {pct}%"
        return self.add_alert(AlertType.DRAWDOWN, "", pct, msg,
                            AlertPriority.HIGH, cooldown_minutes=120)
    
    def alert_loss_streak(self, count: int, message: str = None) -> str:
        """Create alert when losing streak reaches count"""
        msg = message or f"🔴 {count} consecutive losses"
        return self.add_alert(AlertType.LOSS_STREAK, "", count, msg,
                            AlertPriority.HIGH, cooldown_minutes=180)
    
    # ==================== PRICE UPDATES ====================
    
    def update_price(self, symbol: str, price: float, volume: float = None):
        """Update price and check alerts"""
        symbol = symbol.upper().replace('/USDT:USDT', '').replace(':USDT', '')
        
        # Store price history (last 100 prices per symbol)
        now = time.time()
        if symbol not in self._price_history:
            self._price_history[symbol] = []
        self._price_history[symbol].append((now, price))
        self._price_history[symbol] = self._price_history[symbol][-100:]
        
        old_price = self._price_cache.get(symbol)
        self._price_cache[symbol] = price
        
        if volume is not None:
            self._volume_cache[symbol] = volume
        
        # Check price alerts
        self._check_price_alerts(symbol, price, old_price)
    
    def _check_price_alerts(self, symbol: str, current_price: float, old_price: float = None):
        """Check if any price alerts should trigger"""
        now = datetime.now()
        
        for alert in self.alerts.values():
            if not alert.enabled:
                continue
            if alert.symbol and alert.symbol != symbol:
                continue
            
            # Check cooldown
            if alert.last_triggered:
                last = datetime.fromisoformat(alert.last_triggered)
                if (now - last).total_seconds() < alert.cooldown_minutes * 60:
                    continue
            
            triggered = False
            current_value = current_price
            
            if alert.type == AlertType.PRICE_ABOVE:
                if old_price and old_price < alert.condition_value <= current_price:
                    triggered = True
            
            elif alert.type == AlertType.PRICE_BELOW:
                if old_price and old_price > alert.condition_value >= current_price:
                    triggered = True
            
            elif alert.type == AlertType.PRICE_CHANGE_PCT:
                if symbol in self._price_history and len(self._price_history[symbol]) > 10:
                    # Check price change over last few minutes
                    old_prices = self._price_history[symbol]
                    five_min_ago = [p for t, p in old_prices if time.time() - t > 300]
                    if five_min_ago:
                        pct_change = abs((current_price - five_min_ago[-1]) / five_min_ago[-1] * 100)
                        if pct_change >= alert.condition_value:
                            triggered = True
                            current_value = pct_change
            
            elif alert.type == AlertType.VOLUME_SPIKE:
                if symbol in self._volume_cache and symbol in self._avg_volume:
                    vol = self._volume_cache[symbol]
                    avg = self._avg_volume[symbol]
                    if avg > 0 and vol > avg * alert.condition_value:
                        triggered = True
                        current_value = vol / avg
            
            if triggered:
                self._trigger_alert(alert, current_value)
    
    def _trigger_alert(self, alert: Alert, current_value: float = 0.0):
        """Trigger an alert and send notification"""
        now = datetime.now()
        
        # Create triggered alert record
        triggered = TriggeredAlert(
            alert_id=alert.id,
            type=alert.type.value,
            symbol=alert.symbol,
            message=alert.message,
            priority=alert.priority.value,
            triggered_at=now.isoformat(),
            current_value=current_value,
            condition_value=alert.condition_value
        )
        
        self.history.append(triggered)
        self._save_history()
        
        # Update last triggered time
        alert.last_triggered = now.isoformat()
        
        # Remove one-time alerts
        if alert.one_time:
            del self.alerts[alert.id]
        
        self._save_alerts()
        
        # Send notification
        logger.info(f"🔔 ALERT TRIGGERED: {alert.message}")
        
        if self._notification_callback:
            try:
                # Build notification message
                emoji = self._get_priority_emoji(alert.priority)
                msg = f"{emoji} **ALERT**\n{alert.message}"
                if current_value:
                    msg += f"\nCurrent: {current_value:.4f}"
                
                asyncio.create_task(self._send_notification(msg))
            except Exception as e:
                logger.error(f"Failed to send alert notification: {e}")
    
    async def _send_notification(self, message: str):
        """Send notification via callback"""
        if self._notification_callback:
            if asyncio.iscoroutinefunction(self._notification_callback):
                await self._notification_callback(message)
            else:
                self._notification_callback(message)
    
    def _get_priority_emoji(self, priority: AlertPriority) -> str:
        """Get emoji for priority level"""
        emojis = {
            AlertPriority.LOW: "ℹ️",
            AlertPriority.MEDIUM: "🔔",
            AlertPriority.HIGH: "⚠️",
            AlertPriority.CRITICAL: "🚨"
        }
        return emojis.get(priority, "🔔")
    
    # ==================== PORTFOLIO ALERTS ====================
    
    def check_portfolio_alerts(self, current_balance: float, peak_balance: float,
                               consecutive_losses: int, daily_pnl: float):
        """Check portfolio-level alerts"""
        now = datetime.now()
        
        for alert in self.alerts.values():
            if not alert.enabled:
                continue
            
            # Check cooldown
            if alert.last_triggered:
                last = datetime.fromisoformat(alert.last_triggered)
                if (now - last).total_seconds() < alert.cooldown_minutes * 60:
                    continue
            
            triggered = False
            current_value = 0
            
            if alert.type == AlertType.DRAWDOWN:
                if peak_balance > 0:
                    dd_pct = (peak_balance - current_balance) / peak_balance * 100
                    if dd_pct >= alert.condition_value:
                        triggered = True
                        current_value = dd_pct
            
            elif alert.type == AlertType.LOSS_STREAK:
                if consecutive_losses >= alert.condition_value:
                    triggered = True
                    current_value = consecutive_losses
            
            elif alert.type == AlertType.WIN_STREAK:
                # Would need consecutive_wins parameter
                pass
            
            elif alert.type == AlertType.DAILY_PNL:
                if daily_pnl <= -abs(alert.condition_value):
                    triggered = True
                    current_value = daily_pnl
            
            if triggered:
                self._trigger_alert(alert, current_value)
    
    # ==================== NEWS INTEGRATION ====================
    
    async def fetch_crypto_news(self, keywords: List[str] = None) -> List[dict]:
        """Fetch crypto news from CoinGecko (free tier)"""
        if time.time() - self._last_news_fetch < 300:  # Cache for 5 minutes
            return self._news_cache
        
        try:
            # Using CryptoCompare news API (free)
            url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        news = data.get('Data', [])[:20]  # Top 20 news
                        
                        # Filter by keywords if provided
                        if keywords:
                            keywords_lower = [k.lower() for k in keywords]
                            news = [n for n in news 
                                   if any(k in n.get('title', '').lower() or 
                                         k in n.get('body', '').lower() 
                                         for k in keywords_lower)]
                        
                        self._news_cache = news
                        self._last_news_fetch = time.time()
                        return news
        except Exception as e:
            logger.warning(f"Failed to fetch news: {e}")
        
        return self._news_cache
    
    async def check_news_alerts(self, watched_symbols: List[str] = None):
        """Check for relevant news about watched symbols"""
        if watched_symbols is None:
            watched_symbols = ['BTC', 'ETH', 'SOL']
        
        news = await self.fetch_crypto_news(watched_symbols)
        
        # Check for high-impact news
        for article in news[:5]:
            title = article.get('title', '').lower()
            
            # Look for high-impact keywords
            high_impact = ['crash', 'surge', 'hack', 'sec', 'regulation', 
                          'etf', 'blackrock', 'fed', 'rate', 'bankruptcy']
            
            if any(word in title for word in high_impact):
                # Trigger news alert
                for alert in self.alerts.values():
                    if alert.type == AlertType.NEWS and alert.enabled:
                        alert.message = f"📰 {article.get('title', 'News Alert')}"
                        self._trigger_alert(alert, 0)
                        break
    
    # ==================== SUMMARY ====================
    
    def get_summary(self) -> dict:
        """Get alert system summary"""
        active_alerts = [a for a in self.alerts.values() if a.enabled]
        recent_triggers = self.history[-10:]
        
        return {
            'total_alerts': len(self.alerts),
            'active_alerts': len(active_alerts),
            'alerts_by_type': {
                t.value: len([a for a in active_alerts if a.type == t])
                for t in AlertType
            },
            'recent_triggers': len([h for h in self.history 
                                   if datetime.fromisoformat(h.triggered_at) > 
                                   datetime.now() - timedelta(hours=24)]),
            'last_10_triggers': [asdict(t) for t in recent_triggers]
        }


# CLI for testing
if __name__ == "__main__":
    am = AlertManager()
    
    # Add some test alerts
    am.alert_price_above("BTCUSDT", 100000, "BTC hit 100k! 🚀")
    am.alert_price_below("ETHUSDT", 3000, "ETH dropped below 3k")
    am.alert_drawdown(10, "Portfolio drawdown exceeded 10%")
    am.alert_loss_streak(5, "5 consecutive losses!")
    
    print(json.dumps(am.get_summary(), indent=2))
