#!/usr/bin/env python3
"""
Julaba News Monitor - Real-time crypto news and market sentiment tracking.

Features:
- Multiple news sources (CryptoCompare, CoinGecko, Reddit, Twitter sentiment)
- Major event detection (liquidations, whale movements, regulatory news)
- Market sentiment analysis
- Breaking news alerts via Telegram
- Fear & Greed Index tracking
"""

import asyncio
import aiohttp
import json
import logging
import time
import feedparser
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Callable, Any
from pathlib import Path
from enum import Enum
from email.utils import parsedate_to_datetime

logger = logging.getLogger(__name__)

class NewsPriority(Enum):
    """News priority levels."""
    CRITICAL = "critical"   # Major market-moving events (500B pullback, exchange hacks)
    HIGH = "high"           # Significant news (ETF decisions, major regulations)
    MEDIUM = "medium"       # Regular crypto news
    LOW = "low"             # Minor updates


class NewsCategory(Enum):
    """News categories."""
    MARKET = "market"           # General market movements
    REGULATORY = "regulatory"   # Government/regulatory actions
    WHALE = "whale"             # Large transactions
    LIQUIDATION = "liquidation" # Mass liquidations
    EXCHANGE = "exchange"       # Exchange news (hacks, listings)
    DEFI = "defi"               # DeFi protocols
    NFT = "nft"                 # NFT market
    MACRO = "macro"             # Macro economic (Fed, inflation)
    TECHNICAL = "technical"     # Network upgrades, forks


@dataclass
class NewsItem:
    """Single news item."""
    id: str
    title: str
    body: str
    source: str
    url: str
    published: datetime
    categories: List[str] = field(default_factory=list)
    priority: str = "medium"
    sentiment: float = 0.0  # -1 (bearish) to +1 (bullish)
    coins: List[str] = field(default_factory=list)  # Mentioned coins
    impact_score: float = 0.0  # 0-100 market impact estimate


@dataclass
class MarketSentiment:
    """Overall market sentiment data."""
    fear_greed_index: int = 50  # 0-100
    fear_greed_label: str = "Neutral"
    btc_dominance: float = 0.0
    total_market_cap: float = 0.0
    market_cap_change_24h: float = 0.0
    total_volume_24h: float = 0.0
    liquidations_24h: float = 0.0
    long_short_ratio: float = 1.0
    updated: str = ""


class NewsMonitor:
    """
    Real-time crypto news monitoring and alerting system.
    """
    
    # Keywords that indicate major market events
    CRITICAL_KEYWORDS = [
        'billion pullback', 'billion outflow', 'mass liquidation',
        'exchange hack', 'rug pull', 'sec lawsuit', 'ban crypto',
        'emergency shutdown', 'flash crash', 'etf rejected',
        'tether depeg', 'usdc depeg', 'stablecoin depeg',
        'mt gox', 'ftx', 'binance halt', 'coinbase down'
    ]
    
    HIGH_KEYWORDS = [
        'etf approved', 'etf decision', 'whale alert', 'whale moves',
        'fed rate', 'interest rate', 'cpi data', 'inflation',
        'regulation', 'regulatory', 'lawsuit', 'investigation',
        'delisting', 'listing', 'partnership', 'acquisition',
        'billion', 'trillion', 'all-time high', 'all-time low'
    ]
    
    BEARISH_KEYWORDS = [
        'crash', 'dump', 'sell', 'bearish', 'decline', 'fall',
        'pullback', 'correction', 'outflow', 'liquidation', 
        'ban', 'hack', 'exploit', 'rug', 'scam', 'fraud',
        'lawsuit', 'investigation', 'sec', 'warning'
    ]
    
    BULLISH_KEYWORDS = [
        'surge', 'pump', 'rally', 'bullish', 'rise', 'gain',
        'inflow', 'accumulation', 'buy', 'adoption', 'approved',
        'partnership', 'integration', 'upgrade', 'milestone',
        'ath', 'breakout', 'institutional'
    ]
    
    def __init__(self, config_path: str = "news_config.json"):
        self.config_path = Path(config_path)
        self.cache_path = Path("news_cache.json")
        
        # Load config
        self.config = self._load_config()
        
        # RSS feed sources (primary news source - free, no API key needed)
        self.RSS_FEEDS = [
            {'url': 'https://cointelegraph.com/rss', 'name': 'CoinTelegraph', 'icon': '📰'},
            {'url': 'https://decrypt.co/feed', 'name': 'Decrypt', 'icon': '🔓'},
            {'url': 'https://www.newsbtc.com/feed/', 'name': 'NewsBTC', 'icon': '📊'},
            {'url': 'https://bitcoinist.com/feed/', 'name': 'Bitcoinist', 'icon': '₿'},
        ]
        
        # News cache
        self._news_cache: List[NewsItem] = []
        self._sentiment_cache: Optional[MarketSentiment] = None
        self._last_fetch: Dict[str, float] = {}
        
        # Network failure tracking - prevent log spam during outages
        self._consecutive_failures: Dict[str, int] = {}  # source -> count
        self._failure_cooldown: float = 60  # seconds to wait after failure before retry
        
        # Alert callback (for Telegram notifications)
        self._alert_callback: Optional[Callable] = None
        
        # Seen news IDs to avoid duplicates
        self._seen_ids: set = set()
        
        # Load cached data
        self._load_cache()
    
    def _load_config(self) -> dict:
        """Load configuration."""
        default_config = {
            "enabled": True,
            "fetch_interval_seconds": 300,  # 5 minutes
            "sources": {
                "cryptocompare": True,
                "coingecko": True,
                "fear_greed": True,
                "liquidations": True
            },
            "alert_on_priority": ["critical", "high"],
            "watched_coins": ["BTC", "ETH", "SOL", "LINK"],
            "min_impact_score": 50,
            "telegram_alerts": True
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    return {**default_config, **json.load(f)}
            except Exception:
                pass
        
        # Save default config
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _load_cache(self):
        """Load cached news data."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    data = json.load(f)
                    self._seen_ids = set(data.get('seen_ids', []))
            except Exception:
                pass
    
    def _save_cache(self):
        """Save news cache."""
        try:
            # Only keep last 1000 seen IDs
            seen_list = list(self._seen_ids)[-1000:]
            with open(self.cache_path, 'w') as f:
                json.dump({
                    'seen_ids': seen_list,
                    'last_update': datetime.now().isoformat()
                }, f)
        except Exception as e:
            logger.error(f"Failed to save news cache: {e}")
    
    def set_alert_callback(self, callback: Callable):
        """Set callback for news alerts (e.g., Telegram notification)."""
        self._alert_callback = callback
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        Simple sentiment analysis based on keywords.
        Returns -1 (very bearish) to +1 (very bullish).
        """
        text_lower = text.lower()
        
        bullish_count = sum(1 for kw in self.BULLISH_KEYWORDS if kw in text_lower)
        bearish_count = sum(1 for kw in self.BEARISH_KEYWORDS if kw in text_lower)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        return (bullish_count - bearish_count) / total
    
    def _calculate_priority(self, text: str) -> str:
        """Determine news priority based on content."""
        text_lower = text.lower()
        
        # Check for critical keywords
        for kw in self.CRITICAL_KEYWORDS:
            if kw in text_lower:
                return NewsPriority.CRITICAL.value
        
        # Check for high priority keywords
        for kw in self.HIGH_KEYWORDS:
            if kw in text_lower:
                return NewsPriority.HIGH.value
        
        return NewsPriority.MEDIUM.value
    
    def _calculate_impact(self, news: NewsItem) -> float:
        """Calculate estimated market impact score (0-100)."""
        score = 30  # Base score
        
        # Priority bonus
        if news.priority == "critical":
            score += 50
        elif news.priority == "high":
            score += 30
        
        # Sentiment magnitude
        score += abs(news.sentiment) * 20
        
        # Source reliability
        reliable_sources = ['bloomberg', 'reuters', 'coindesk', 'cointelegraph']
        if any(s in news.source.lower() for s in reliable_sources):
            score += 10
        
        return min(100, score)
    
    def _extract_coins(self, text: str) -> List[str]:
        """Extract mentioned cryptocurrency symbols from text."""
        # Common coin patterns
        coin_patterns = [
            'BTC', 'ETH', 'SOL', 'LINK', 'XRP', 'ADA', 'DOT', 'AVAX',
            'MATIC', 'DOGE', 'SHIB', 'LTC', 'BCH', 'UNI', 'ATOM',
            'Bitcoin', 'Ethereum', 'Solana', 'Chainlink', 'Ripple'
        ]
        
        found = []
        text_upper = text.upper()
        
        for coin in coin_patterns:
            if coin.upper() in text_upper:
                # Normalize to symbol
                symbol = coin.upper()
                if symbol == 'BITCOIN':
                    symbol = 'BTC'
                elif symbol == 'ETHEREUM':
                    symbol = 'ETH'
                elif symbol == 'SOLANA':
                    symbol = 'SOL'
                elif symbol == 'CHAINLINK':
                    symbol = 'LINK'
                elif symbol == 'RIPPLE':
                    symbol = 'XRP'
                
                if symbol not in found:
                    found.append(symbol)
        
        return found
    
    async def fetch_cryptocompare_news(self) -> List[NewsItem]:
        """Fetch news from CryptoCompare API (free)."""
        cache_key = 'cryptocompare'
        if time.time() - self._last_fetch.get(cache_key, 0) < 120:
            return []
        
        news_items = []
        
        try:
            url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&sortOrder=latest"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check for API error (rate limit, etc.)
                        if data.get('Response') == 'Error':
                            logger.debug(f"CryptoCompare API error: {data.get('Message', 'Unknown error')}")
                            return []
                        
                        # Ensure Data is a list, not a dict
                        articles_data = data.get('Data', [])
                        if not isinstance(articles_data, list):
                            logger.debug(f"CryptoCompare returned non-list Data: {type(articles_data)}")
                            return []
                        
                        articles = articles_data[:30]
                        
                        for article in articles:
                            try:
                                news_id = f"cc_{article.get('id', '')}"
                                
                                if news_id in self._seen_ids:
                                    continue
                                
                                title = str(article.get('title', '') or '')
                                body_raw = article.get('body', '') or ''
                                body = str(body_raw)[:500] if body_raw else ''
                                full_text = f"{title} {body}"
                            
                                news_item = NewsItem(
                                    id=news_id,
                                    title=title,
                                    body=body,
                                    source=str(article.get('source', 'CryptoCompare') or 'CryptoCompare'),
                                    url=str(article.get('url', '') or ''),
                                    published=datetime.fromtimestamp(article.get('published_on', time.time())),
                                    categories=str(article.get('categories', '') or '').split('|'),
                                    priority=self._calculate_priority(full_text),
                                    sentiment=self._analyze_sentiment(full_text),
                                    coins=self._extract_coins(full_text)
                                )
                                news_item.impact_score = self._calculate_impact(news_item)
                                
                                news_items.append(news_item)
                                self._seen_ids.add(news_id)
                            except Exception as article_err:
                                logger.debug(f"Skipping malformed article: {article_err}")
            
            self._last_fetch[cache_key] = time.time()
            self._consecutive_failures[cache_key] = 0  # Reset on success
            
        except Exception as e:
            fails = self._consecutive_failures.get(cache_key, 0) + 1
            self._consecutive_failures[cache_key] = fails
            # Cooldown: wait before retrying (prevents log spam during outage)
            self._last_fetch[cache_key] = time.time() - 120 + self._failure_cooldown
            if fails <= 1:
                logger.warning(f"CryptoCompare news fetch error: {type(e).__name__}: {e}")
            else:
                logger.debug(f"CryptoCompare fetch retry #{fails} failed: {type(e).__name__}")
        
        return news_items
    
    async def fetch_rss_news(self) -> List[NewsItem]:
        """Fetch news from multiple RSS feeds (primary source - free, no API key)."""
        cache_key = 'rss_feeds'
        if time.time() - self._last_fetch.get(cache_key, 0) < 120:
            return []
        
        news_items = []
        
        for feed_info in self.RSS_FEEDS:
            try:
                feed_url = feed_info['url']
                source_name = feed_info['name']
                
                # Fetch RSS feed content via aiohttp
                async with aiohttp.ClientSession() as session:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (compatible; JulabaBot/1.0)',
                        'Accept': 'application/rss+xml, application/xml, text/xml'
                    }
                    async with session.get(feed_url, timeout=aiohttp.ClientTimeout(total=10), headers=headers) as response:
                        if response.status != 200:
                            logger.debug(f"RSS feed {source_name} returned status {response.status}")
                            continue
                        feed_content = await response.text()
                
                # Parse with feedparser (sync - it's fast)
                feed = feedparser.parse(feed_content)
                
                if not feed.entries:
                    logger.debug(f"RSS feed {source_name} returned no entries")
                    continue
                
                for entry in feed.entries[:15]:  # Max 15 per source
                    try:
                        # Generate unique ID from source + title hash
                        title = entry.get('title', '').strip()
                        if not title:
                            continue
                        
                        news_id = f"rss_{source_name.lower()}_{hash(title) & 0xFFFFFFFF}"
                        
                        if news_id in self._seen_ids:
                            continue
                        
                        # Parse body/summary - strip HTML tags
                        body_raw = entry.get('summary', '') or entry.get('description', '') or ''
                        # Simple HTML tag stripping
                        import re
                        body = re.sub(r'<[^>]+>', '', body_raw)[:500]
                        
                        # Parse publish date
                        published = datetime.now()
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            try:
                                published = datetime(*entry.published_parsed[:6])
                            except Exception:
                                pass
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            try:
                                published = datetime(*entry.updated_parsed[:6])
                            except Exception:
                                pass
                        
                        # Get link
                        url = entry.get('link', '') or ''
                        
                        # Extract categories from feed tags
                        categories = []
                        if hasattr(entry, 'tags'):
                            categories = [t.get('term', '') for t in entry.tags if t.get('term')][:5]
                        
                        full_text = f"{title} {body}"
                        
                        news_item = NewsItem(
                            id=news_id,
                            title=title,
                            body=body,
                            source=source_name,
                            url=url,
                            published=published,
                            categories=categories,
                            priority=self._calculate_priority(full_text),
                            sentiment=self._analyze_sentiment(full_text),
                            coins=self._extract_coins(full_text)
                        )
                        news_item.impact_score = self._calculate_impact(news_item)
                        
                        news_items.append(news_item)
                        self._seen_ids.add(news_id)
                    except Exception as entry_err:
                        logger.debug(f"Skipping malformed RSS entry from {source_name}: {entry_err}")
                        
            except asyncio.TimeoutError:
                logger.debug(f"RSS feed {feed_info['name']} timed out")
            except Exception as e:
                logger.debug(f"RSS feed {feed_info['name']} error: {type(e).__name__}: {e}")
        
        # Sort by published date (newest first)
        news_items.sort(key=lambda x: x.published, reverse=True)
        
        self._last_fetch[cache_key] = time.time()
        self._consecutive_failures[cache_key] = 0
        
        if news_items:
            logger.info(f"📰 Fetched {len(news_items)} news articles from RSS feeds")
        
        return news_items
    
    async def fetch_fear_greed_index(self) -> Optional[MarketSentiment]:
        """Fetch Fear & Greed Index and market sentiment."""
        cache_key = 'fear_greed'
        if time.time() - self._last_fetch.get(cache_key, 0) < 300:
            return self._sentiment_cache
        
        try:
            # Fear & Greed Index API (free)
            fg_url = "https://api.alternative.me/fng/?limit=1"
            
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # Fetch Fear & Greed
                async with session.get(fg_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        fg_data = data.get('data', [{}])[0]
                        
                        self._sentiment_cache = MarketSentiment(
                            fear_greed_index=int(fg_data.get('value', 50)),
                            fear_greed_label=fg_data.get('value_classification', 'Neutral'),
                            updated=datetime.now().isoformat()
                        )
                
                # Try to get global market data from CoinGecko
                try:
                    global_url = "https://api.coingecko.com/api/v3/global"
                    async with session.get(global_url, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            global_data = data.get('data', {})
                            
                            if self._sentiment_cache:
                                self._sentiment_cache.total_market_cap = global_data.get('total_market_cap', {}).get('usd', 0)
                                self._sentiment_cache.market_cap_change_24h = global_data.get('market_cap_change_percentage_24h_usd', 0)
                                self._sentiment_cache.total_volume_24h = global_data.get('total_volume', {}).get('usd', 0)
                                self._sentiment_cache.btc_dominance = global_data.get('market_cap_percentage', {}).get('btc', 0)
                except Exception:
                    pass
            
            self._last_fetch[cache_key] = time.time()
            self._consecutive_failures[cache_key] = 0  # Reset on success
            
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            fails = self._consecutive_failures.get(cache_key, 0) + 1
            self._consecutive_failures[cache_key] = fails
            # Cooldown: wait before retrying (prevents 1000s of errors during outage)
            self._last_fetch[cache_key] = time.time() - 300 + self._failure_cooldown
            if fails <= 1:
                logger.warning(f"Fear & Greed network error: {type(e).__name__}: {e}")
            elif fails % 10 == 0:
                logger.warning(f"Fear & Greed still unreachable ({fails} failures)")
            else:
                logger.debug(f"Fear & Greed retry #{fails} failed: {type(e).__name__}")
        except Exception as e:
            self._last_fetch[cache_key] = time.time() - 300 + self._failure_cooldown
            logger.warning(f"Fear & Greed fetch error: {type(e).__name__}: {e}")
        
        return self._sentiment_cache
    
    async def fetch_liquidation_data(self) -> Dict[str, Any]:
        """Fetch recent liquidation data from multiple sources."""
        cache_key = 'liquidations'
        if time.time() - self._last_fetch.get(cache_key, 0) < 30:  # 30 second cache
            return getattr(self, '_liquidation_cache', {})
        
        result = {
            'total_24h_usd': 0,
            'long_liquidations': 0,
            'short_liquidations': 0,
            'largest_single': 0,
            'btc_liquidations': 0,
            'eth_liquidations': 0,
            'alert_level': 'normal',
            'exchanges_count': 0,
            'updated': datetime.now().isoformat()
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Try Coinglass public API (limited but free)
                # Note: Full API requires key, but we can scrape public data
                try:
                    coinglass_url = "https://open-api.coinglass.com/public/v2/liquidation_history"
                    async with session.get(coinglass_url, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('success') and data.get('data'):
                                liq_data = data['data']
                                # Sum last 24h liquidations
                                for item in liq_data[-24:]:  # Last 24 hours
                                    result['total_24h_usd'] += float(item.get('longLiquidationUsd', 0) or 0)
                                    result['total_24h_usd'] += float(item.get('shortLiquidationUsd', 0) or 0)
                                    result['long_liquidations'] += float(item.get('longLiquidationUsd', 0) or 0)
                                    result['short_liquidations'] += float(item.get('shortLiquidationUsd', 0) or 0)
                except Exception as cg_err:
                    logger.debug(f"Coinglass API not available: {cg_err}")
                
                # Fallback: CoinGecko derivatives for open interest changes
                try:
                    cg_url = "https://api.coingecko.com/api/v3/derivatives"
                    async with session.get(cg_url, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            result['exchanges_count'] = len(data)
                            
                            # Estimate liquidations from OI changes and volume
                            total_volume = sum(float(d.get('trade_volume_24h_btc', 0) or 0) for d in data[:20])
                            # Rough estimate: high volume + negative OI change = liquidations
                            if total_volume > 500000:  # > 500K BTC volume
                                result['alert_level'] = 'elevated'
                except Exception as deriv_err:
                    logger.debug(f"CoinGecko derivatives error: {deriv_err}")
                
                # Determine alert level based on liquidation volume
                total_liq = result['total_24h_usd']
                if total_liq >= 500_000_000:  # $500M+
                    result['alert_level'] = 'critical'
                    logger.warning(f"🚨 CRITICAL LIQUIDATIONS: ${total_liq/1e6:.0f}M in 24h!")
                elif total_liq >= 200_000_000:  # $200M+
                    result['alert_level'] = 'high'
                    logger.warning(f"⚠️ HIGH LIQUIDATIONS: ${total_liq/1e6:.0f}M in 24h")
                elif total_liq >= 100_000_000:  # $100M+
                    result['alert_level'] = 'elevated'
            
            self._last_fetch[cache_key] = time.time()
            self._liquidation_cache = result
            self._consecutive_failures[cache_key] = 0
            
        except Exception as e:
            fails = self._consecutive_failures.get(cache_key, 0) + 1
            self._consecutive_failures[cache_key] = fails
            self._last_fetch[cache_key] = time.time() - 30 + self._failure_cooldown
            if fails <= 1:
                logger.warning(f"Liquidation data fetch error: {type(e).__name__}: {e}")
            else:
                logger.debug(f"Liquidation fetch retry #{fails} failed: {type(e).__name__}")
        
        return result
    
    async def fetch_whale_alerts(self) -> List[Dict[str, Any]]:
        """
        Fetch whale transaction alerts from multiple sources.
        Detects large BTC/ETH movements that often precede price action.
        """
        cache_key = 'whale_alerts'
        if time.time() - self._last_fetch.get(cache_key, 0) < 60:
            return getattr(self, '_whale_cache', [])
        
        whale_alerts = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Method 1: Blockchain.com API for large BTC transactions
                try:
                    # Get recent BTC blocks and check for large txs
                    btc_url = "https://blockchain.info/unconfirmed-transactions?format=json"
                    async with session.get(btc_url, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            txs = data.get('txs', [])
                            
                            for tx in txs[:50]:  # Check last 50 unconfirmed
                                # Calculate total output value
                                total_btc = sum(out.get('value', 0) for out in tx.get('out', [])) / 1e8
                                
                                if total_btc >= 100:  # 100+ BTC movement
                                    alert = {
                                        'coin': 'BTC',
                                        'amount': total_btc,
                                        'usd_value': total_btc * 95000,  # Approximate BTC price
                                        'type': 'transfer',
                                        'hash': tx.get('hash', '')[:16],
                                        'time': datetime.now().isoformat(),
                                        'significance': 'high' if total_btc >= 500 else 'medium'
                                    }
                                    whale_alerts.append(alert)
                                    
                                    if total_btc >= 1000:
                                        logger.warning(f"🐋 WHALE ALERT: {total_btc:.0f} BTC (${total_btc * 95000 / 1e6:.1f}M) moving!")
                except Exception as btc_err:
                    logger.debug(f"BTC whale check error: {btc_err}")
                
                # Method 2: Etherscan API for large ETH transactions (needs API key for full access)
                # Using public endpoint with limitations
                try:
                    # Check ETH gas tracker for unusual activity
                    eth_gas_url = "https://api.etherscan.io/api?module=gastracker&action=gasoracle"
                    async with session.get(eth_gas_url, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('status') == '1':
                                gas_data = data.get('result', {})
                                fast_gas = int(gas_data.get('FastGasPrice', 0))
                                
                                # Very high gas = potential whale activity or major event
                                if fast_gas >= 100:  # 100+ gwei
                                    whale_alerts.append({
                                        'coin': 'ETH',
                                        'type': 'high_gas_activity',
                                        'gas_price': fast_gas,
                                        'significance': 'high' if fast_gas >= 200 else 'medium',
                                        'time': datetime.now().isoformat()
                                    })
                                    logger.info(f"⛽ High ETH gas: {fast_gas} gwei - potential whale activity")
                except Exception as eth_err:
                    logger.debug(f"ETH activity check error: {eth_err}")
                
                # Method 3: Check exchange inflows/outflows via CryptoQuant-style metrics
                # This is approximated from public data
                try:
                    # Use Glassnode public metrics if available
                    glassnode_url = "https://api.glassnode.com/v1/metrics/transactions/transfers_volume_sum"
                    # Note: Requires API key for actual data
                except Exception:
                    pass
            
            self._last_fetch[cache_key] = time.time()
            self._whale_cache = whale_alerts
            self._consecutive_failures[cache_key] = 0
            
        except Exception as e:
            fails = self._consecutive_failures.get(cache_key, 0) + 1
            self._consecutive_failures[cache_key] = fails
            self._last_fetch[cache_key] = time.time() - 300 + self._failure_cooldown
            if fails <= 1:
                logger.warning(f"Whale alerts fetch error: {type(e).__name__}: {e}")
            else:
                logger.debug(f"Whale alerts fetch retry #{fails} failed: {type(e).__name__}")
        
        return whale_alerts
    
    async def fetch_social_sentiment(self) -> Dict[str, Any]:
        """
        Fetch social media sentiment from Twitter/X and Reddit.
        Uses public APIs and sentiment analysis.
        """
        cache_key = 'social_sentiment'
        if time.time() - self._last_fetch.get(cache_key, 0) < 300:  # 5 min cache
            return getattr(self, '_social_cache', {})
        
        result = {
            'twitter_sentiment': 0.0,
            'reddit_sentiment': 0.0,
            'combined_sentiment': 0.0,
            'trending_coins': [],
            'hot_topics': [],
            'social_volume': 'normal',
            'updated': datetime.now().isoformat()
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                # Method 1: LunarCrush API (if key available) or public metrics
                try:
                    # LunarCrush free tier
                    lunar_url = "https://lunarcrush.com/api3/coins/list?sort=galaxy_score"
                    async with session.get(lunar_url, timeout=10, headers={'Accept': 'application/json'}) as response:
                        if response.status == 200:
                            data = await response.json()
                            coins = data.get('data', [])[:10]
                            
                            # Extract trending coins and sentiment
                            for coin in coins:
                                if coin.get('symbol') in ['BTC', 'ETH', 'SOL', 'LINK']:
                                    sentiment = coin.get('sentiment', 50) / 100 - 0.5  # Normalize to -0.5 to 0.5
                                    result['trending_coins'].append({
                                        'symbol': coin.get('symbol'),
                                        'sentiment': sentiment,
                                        'social_volume': coin.get('social_volume', 0)
                                    })
                except Exception as lunar_err:
                    logger.debug(f"LunarCrush API not available: {lunar_err}")
                
                # Method 2: Reddit API (r/cryptocurrency, r/bitcoin sentiment)
                try:
                    # Reddit public JSON endpoint
                    reddit_url = "https://www.reddit.com/r/CryptoCurrency/hot.json?limit=25"
                    headers = {'User-Agent': 'Julaba Trading Bot 1.0'}
                    async with session.get(reddit_url, timeout=10, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            posts = data.get('data', {}).get('children', [])
                            
                            sentiments = []
                            topics = []
                            for post in posts:
                                post_data = post.get('data', {})
                                title = post_data.get('title', '')
                                
                                # Analyze sentiment
                                sentiment = self._analyze_sentiment(title)
                                sentiments.append(sentiment)
                                
                                # Extract hot topics
                                if post_data.get('score', 0) > 500:  # High upvotes
                                    topics.append(title[:60])
                            
                            if sentiments:
                                result['reddit_sentiment'] = sum(sentiments) / len(sentiments)
                                result['hot_topics'] = topics[:5]
                except Exception as reddit_err:
                    logger.debug(f"Reddit API error: {reddit_err}")
                
                # Calculate combined sentiment
                sentiments = [result['twitter_sentiment'], result['reddit_sentiment']]
                valid_sentiments = [s for s in sentiments if s != 0]
                if valid_sentiments:
                    result['combined_sentiment'] = sum(valid_sentiments) / len(valid_sentiments)
                
                # Determine social volume level
                if result.get('trending_coins'):
                    avg_volume = sum(c.get('social_volume', 0) for c in result['trending_coins']) / len(result['trending_coins'])
                    if avg_volume > 1000000:
                        result['social_volume'] = 'very_high'
                    elif avg_volume > 500000:
                        result['social_volume'] = 'high'
            
            self._last_fetch[cache_key] = time.time()
            self._social_cache = result
            self._consecutive_failures[cache_key] = 0
            
        except Exception as e:
            fails = self._consecutive_failures.get(cache_key, 0) + 1
            self._consecutive_failures[cache_key] = fails
            self._last_fetch[cache_key] = time.time() - 300 + self._failure_cooldown
            if fails <= 1:
                logger.warning(f"Social sentiment fetch error: {type(e).__name__}: {e}")
            else:
                logger.debug(f"Social sentiment fetch retry #{fails} failed: {type(e).__name__}")
        
        return result
    
    async def analyze_with_llm(self, news_items: List[NewsItem]) -> Dict[str, Any]:
        """
        Use Gemini/Claude to analyze news sentiment and market impact.
        Provides more nuanced analysis than keyword matching.
        """
        if not news_items:
            return {'llm_sentiment': 0, 'analysis': 'No news to analyze'}
        
        try:
            # Try to use Gemini for analysis
            try:
                from google import genai
                client = genai.Client()
                
                try:
                    # Prepare news summary for LLM
                    news_text = "\n".join([
                        f"- [{n.priority.upper()}] {n.title}: {n.body[:150]}"
                        for n in news_items[:10]
                    ])
                    
                    prompt = f"""Analyze these crypto news headlines and provide:
1. Overall market sentiment (-1 bearish to +1 bullish)
2. Key themes/concerns
3. Trading recommendation (bullish/bearish/neutral)
4. Risk level (low/medium/high)

News:
{news_text}

Respond in JSON format:
{{"sentiment": 0.0, "themes": [], "recommendation": "", "risk_level": "", "summary": ""}}"""

                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=prompt
                    )
                    
                    # Parse response
                    response_text = response.text.strip()
                    if response_text.startswith('```'):
                        response_text = response_text.split('```')[1]
                        if response_text.startswith('json'):
                            response_text = response_text[4:]
                    
                    result = json.loads(response_text)
                    result['llm_used'] = 'gemini'
                    logger.info(f"📊 LLM Analysis: Sentiment={result.get('sentiment', 0):.2f}, Risk={result.get('risk_level', 'unknown')}")
                    return result
                finally:
                    # Properly close the Gemini client to prevent resource leaks
                    try:
                        if hasattr(client, 'aclose'):
                            import asyncio
                            asyncio.create_task(client.aclose())
                        elif hasattr(client, 'close'):
                            client.close()
                    except Exception:
                        pass  # Ignore cleanup errors
                
            except ImportError:
                logger.debug("Gemini not available for LLM analysis")
            except Exception as gemini_err:
                logger.debug(f"Gemini analysis error: {gemini_err}")
            
            # Fallback to keyword-based analysis
            avg_sentiment = sum(n.sentiment for n in news_items) / len(news_items)
            return {
                'llm_sentiment': avg_sentiment,
                'themes': ['keyword-based analysis'],
                'recommendation': 'bullish' if avg_sentiment > 0.2 else 'bearish' if avg_sentiment < -0.2 else 'neutral',
                'risk_level': 'medium',
                'llm_used': 'fallback'
            }
            
        except Exception as e:
            logger.error(f"LLM analysis error: {e}")
            return {'llm_sentiment': 0, 'analysis': str(e)}

    async def check_for_major_events(self) -> List[NewsItem]:
        """
        Check for major market events that need immediate attention.
        This is the main function to detect things like "$500B pullback".
        """
        major_events = []
        
        # Fetch all news
        news = await self.fetch_cryptocompare_news()
        
        # Filter for critical/high priority
        for item in news:
            if item.priority in ['critical', 'high']:
                major_events.append(item)
                
                # Send alert if callback is set
                if self._alert_callback and self.config.get('telegram_alerts'):
                    await self._send_news_alert(item)
        
        # Check market sentiment
        sentiment = await self.fetch_fear_greed_index()
        
        if sentiment:
            # Alert on extreme fear (potential buying opportunity) or extreme greed (potential top)
            if sentiment.fear_greed_index <= 20:
                logger.warning(f"🔴 EXTREME FEAR: Fear & Greed Index at {sentiment.fear_greed_index}")
            elif sentiment.fear_greed_index >= 80:
                logger.warning(f"🟢 EXTREME GREED: Fear & Greed Index at {sentiment.fear_greed_index}")
            
            # Alert on large market cap changes
            if abs(sentiment.market_cap_change_24h) >= 5:
                direction = "📈" if sentiment.market_cap_change_24h > 0 else "📉"
                logger.warning(f"{direction} MAJOR MARKET MOVE: {sentiment.market_cap_change_24h:.1f}% in 24h")
        
        # Save cache
        self._save_cache()
        
        return major_events
    
    async def _send_news_alert(self, news: NewsItem):
        """Send news alert via callback (Telegram)."""
        if not self._alert_callback:
            return
        
        # Format alert message
        priority_emoji = {
            'critical': '🚨🚨🚨',
            'high': '⚠️',
            'medium': 'ℹ️',
            'low': '📰'
        }
        
        sentiment_emoji = '🐂' if news.sentiment > 0.3 else '🐻' if news.sentiment < -0.3 else '➖'
        
        message = f"""
{priority_emoji.get(news.priority, '📰')} **{news.priority.upper()} PRIORITY NEWS**

**{news.title}**

{news.body[:300]}...

📊 Sentiment: {sentiment_emoji} ({news.sentiment:+.2f})
💥 Impact Score: {news.impact_score:.0f}/100
🪙 Coins: {', '.join(news.coins) if news.coins else 'General'}
📰 Source: {news.source}
🔗 {news.url}
"""
        
        try:
            await self._alert_callback(message)
        except Exception as e:
            logger.error(f"Failed to send news alert: {e}")
    
    async def get_market_summary(self) -> Dict[str, Any]:
        """Get comprehensive market summary including news, sentiment, liquidations, whales, and social."""
        # Fetch all data sources in parallel (RSS primary, CryptoCompare fallback)
        rss_task = self.fetch_rss_news()
        cc_task = self.fetch_cryptocompare_news()
        sentiment_task = self.fetch_fear_greed_index()
        liquidations_task = self.fetch_liquidation_data()
        whale_task = self.fetch_whale_alerts()
        social_task = self.fetch_social_sentiment()
        
        # Gather results
        rss_news, cc_news, sentiment, liquidations, whale_alerts, social = await asyncio.gather(
            rss_task, cc_task, sentiment_task, liquidations_task, whale_task, social_task,
            return_exceptions=True
        )
        
        # Handle any exceptions
        if isinstance(rss_news, Exception):
            rss_news = []
            logger.debug(f"RSS news fetch failed: {rss_news}")
        if isinstance(cc_news, Exception):
            cc_news = []
            logger.debug(f"CryptoCompare news fetch failed: {cc_news}")
        if isinstance(sentiment, Exception):
            sentiment = None
        if isinstance(liquidations, Exception):
            liquidations = {}
        if isinstance(whale_alerts, Exception):
            whale_alerts = []
        if isinstance(social, Exception):
            social = {}
        
        # Merge news: RSS primary, CryptoCompare supplement (deduplicate by title similarity)
        news = list(rss_news)
        rss_titles = {n.title.lower()[:50] for n in rss_news}
        for cc_item in cc_news:
            # Only add CryptoCompare articles that aren't duplicates of RSS articles
            if cc_item.title.lower()[:50] not in rss_titles:
                news.append(cc_item)
        
        # Sort all news by published date (newest first)
        news.sort(key=lambda x: x.published, reverse=True)
        if isinstance(sentiment, Exception):
            sentiment = None
        if isinstance(liquidations, Exception):
            liquidations = {}
        if isinstance(whale_alerts, Exception):
            whale_alerts = []
        if isinstance(social, Exception):
            social = {}
        
        # Categorize recent news
        bullish_news = [n for n in news if n.sentiment > 0.2]
        bearish_news = [n for n in news if n.sentiment < -0.2]
        critical_news = [n for n in news if n.priority == 'critical']
        
        # LLM analysis if we have critical news
        llm_analysis = {}
        if critical_news or len(news) > 0:
            try:
                llm_analysis = await self.analyze_with_llm(news[:10])
            except Exception:
                pass
        
        return {
            'sentiment': asdict(sentiment) if sentiment else None,
            'news_summary': {
                'total_articles': len(news),
                'bullish_count': len(bullish_news),
                'bearish_count': len(bearish_news),
                'critical_count': len(critical_news),
                'average_sentiment': sum(n.sentiment for n in news) / len(news) if news else 0
            },
            'critical_news': [asdict(n) for n in critical_news[:5]],
            'recent_news': [asdict(n) for n in news[:10]],
            'liquidations': liquidations,
            'whale_alerts': whale_alerts[:10],
            'social_sentiment': social,
            'llm_analysis': llm_analysis,
            'recommendation': self._get_market_recommendation(sentiment, news, liquidations, whale_alerts)
        }
    
    def _get_market_recommendation(self, sentiment: Optional[MarketSentiment], news: List[NewsItem], 
                                     liquidations: Dict = None, whale_alerts: List = None) -> str:
        """Generate operational briefing aligned with bot's actual entry logic.
        
        This mirrors the real gates in ai_filter.py so the dashboard tells you
        exactly what the bot will and won't do right now.
        """
        if not sentiment:
            return "⏳ WAITING — No sentiment data yet"
        
        fg = sentiment.fear_greed_index
        fg_label = sentiment.fear_greed_label
        cap_change = sentiment.market_cap_change_24h or 0
        btc_dom = sentiment.btc_dominance or 0
        
        # ── Build lines that explain what's happening ──
        lines = []
        
        # ── 1. F&G STATUS — the biggest gate ──
        if fg < 5:
            header = f"🚫 HARD BLOCK — F&G {fg} (TRUE PANIC)"
            lines.append(f"ALL trades blocked. Spreads wide, liquidations cascading.")
            lines.append(f"Bot will not open any position until F&G ≥ 5.")
        elif fg < 10:
            header = f"🔴 EXTREME FEAR — F&G {fg}"
            lines.append(f"LONGs blocked (catching falling knives).")
            lines.append(f"SHORTs allowed only if math score ≥ 55.")
        elif fg < 20:
            header = f"🟠 HIGH FEAR — F&G {fg}"
            lines.append(f"LONGs restricted: need price near support + positive momentum.")
            lines.append(f"SHORTs open: standard thresholds apply.")
        elif fg < 40:
            header = f"🟡 FEAR — F&G {fg} ({fg_label})"
            lines.append(f"Both directions open, LONGs need stronger confirmation.")
        elif fg <= 60:
            header = f"🟢 NEUTRAL — F&G {fg} ({fg_label})"
            lines.append(f"Best conditions. Both directions open, normal thresholds.")
        elif fg <= 80:
            header = f"🟡 GREED — F&G {fg} ({fg_label})"
            lines.append(f"Both directions open, SHORTs near resistance preferred.")
        else:
            header = f"🔴 EXTREME GREED — F&G {fg}"
            lines.append(f"SHORTs restricted: need price near resistance (upper range).")
            lines.append(f"LONGs risky: euphoria often precedes correction.")
        
        # ── 2. MARKET MOVEMENT ──
        if abs(cap_change) >= 5:
            direction = "📈 SURGING" if cap_change > 0 else "📉 CRASHING"
            lines.append(f"{direction} {cap_change:+.1f}% in 24h — major move in progress.")
        elif abs(cap_change) >= 2:
            direction = "↗️ Up" if cap_change > 0 else "↘️ Down"
            lines.append(f"{direction} {cap_change:+.1f}% in 24h.")
        else:
            lines.append(f"Market flat ({cap_change:+.1f}% 24h). BTC dom {btc_dom:.1f}%.")
        
        # ── 3. NEWS SENTIMENT ──
        if news:
            avg_sent = sum(n.sentiment for n in news) / len(news)
            bullish_ct = sum(1 for n in news if n.sentiment > 0.2)
            bearish_ct = sum(1 for n in news if n.sentiment < -0.2)
            critical = [n for n in news if n.priority == 'critical']
            
            if critical:
                for c in critical[:2]:
                    icon = "🚨" if c.sentiment < 0 else "⚡"
                    lines.append(f"{icon} CRITICAL: {c.title[:60]}")
            
            if abs(avg_sent) < 0.1:
                lines.append(f"📰 News mixed (↑{bullish_ct} / ↓{bearish_ct}) — no strong bias.")
            elif avg_sent > 0:
                lines.append(f"📰 News leaning bullish ({avg_sent:+.2f}, ↑{bullish_ct} vs ↓{bearish_ct}).")
            else:
                lines.append(f"📰 News leaning bearish ({avg_sent:+.2f}, ↑{bullish_ct} vs ↓{bearish_ct}).")
        
        # ── 4. LIQUIDATIONS ──
        if liquidations:
            alert_level = liquidations.get('alert_level', 'normal')
            total_liq = liquidations.get('total_24h_usd', 0)
            long_liq = liquidations.get('long_liquidations', 0)
            short_liq = liquidations.get('short_liquidations', 0)
            
            if alert_level == 'critical':
                if long_liq > short_liq * 1.5:
                    lines.append(f"🚨 Mass LONG liquidations (${total_liq/1e6:.0f}M) — longs getting wiped.")
                elif short_liq > long_liq * 1.5:
                    lines.append(f"🚨 Mass SHORT liquidations (${total_liq/1e6:.0f}M) — short squeeze.")
                else:
                    lines.append(f"🚨 Heavy liquidations (${total_liq/1e6:.0f}M) — extreme volatility.")
            elif alert_level == 'high':
                lines.append(f"⚠️ Elevated liquidations (${total_liq/1e6:.0f}M).")
        
        # ── 5. WHALE ACTIVITY ──
        if whale_alerts:
            high_sig = [w for w in whale_alerts if w.get('significance') == 'high']
            if high_sig:
                total_btc = sum(w.get('amount', 0) for w in high_sig if w.get('coin') == 'BTC')
                if total_btc > 500:
                    lines.append(f"🐋 Whale moves: {total_btc:.0f} BTC in transit.")
        
        # ── Assemble ──
        body = "\n".join(f"• {l}" for l in lines)
        return f"{header}\n\n{body}"
    
    async def run_continuous_monitor(self, interval_seconds: int = 120):
        """
        Run continuous news monitoring loop with proactive alerts.
        This is designed to be run as a background task in the main bot.
        """
        logger.info(f"📰 News monitor started (interval: {interval_seconds}s)")
        
        # Track last alert times to avoid spam
        last_liquidation_alert = 0
        last_whale_alert = 0
        
        while True:
            try:
                # Check for major events (news)
                events = await self.check_for_major_events()
                
                if events:
                    logger.info(f"📰 Found {len(events)} significant news items")
                    for event in events:
                        logger.info(f"  [{event.priority.upper()}] {event.title[:60]}...")
                
                # Check liquidations (more frequently)
                liquidations = await self.fetch_liquidation_data()
                if liquidations.get('alert_level') in ['critical', 'high']:
                    if time.time() - last_liquidation_alert > 300:  # 5 min cooldown
                        alert_msg = f"🚨 LIQUIDATION ALERT: ${liquidations.get('total_24h_usd', 0)/1e6:.0f}M in 24h ({liquidations.get('alert_level').upper()})"
                        logger.warning(alert_msg)
                        if self._alert_callback:
                            try:
                                await self._alert_callback(alert_msg)
                            except Exception:
                                pass
                        last_liquidation_alert = time.time()
                
                # Check whale movements
                whales = await self.fetch_whale_alerts()
                high_sig_whales = [w for w in whales if w.get('significance') == 'high']
                if high_sig_whales and time.time() - last_whale_alert > 600:  # 10 min cooldown
                    for whale in high_sig_whales[:3]:
                        if whale.get('coin') == 'BTC' and whale.get('amount', 0) >= 500:
                            alert_msg = f"🐋 WHALE ALERT: {whale.get('amount'):.0f} BTC (${whale.get('usd_value', 0)/1e6:.1f}M) moving!"
                            logger.warning(alert_msg)
                            if self._alert_callback:
                                try:
                                    await self._alert_callback(alert_msg)
                                except Exception:
                                    pass
                    last_whale_alert = time.time()
                
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                logger.info("News monitor stopped")
                break
            except Exception as e:
                logger.error(f"News monitor error: {e}")
                await asyncio.sleep(60)
    
    def get_trading_bias(self) -> Dict[str, Any]:
        """
        Get current trading bias from news/sentiment data.
        This is a synchronous method that returns cached data for quick access.
        
        Returns:
            dict with 'bias' (bullish/bearish/neutral), 'confidence', 'reasons'
        """
        try:
            sentiment = self._sentiment_cache
            liquidations = getattr(self, '_liquidation_cache', {})
            whales = getattr(self, '_whale_cache', [])
            
            bias_score = 0
            reasons = []
            
            # Fear & Greed
            if sentiment:
                fg = sentiment.fear_greed_index
                if fg <= 25:
                    bias_score += 2
                    reasons.append(f"Extreme fear ({fg})")
                elif fg <= 40:
                    bias_score += 1
                    reasons.append(f"Fear ({fg})")
                elif fg >= 75:
                    bias_score -= 2
                    reasons.append(f"Extreme greed ({fg})")
                elif fg >= 60:
                    bias_score -= 1
                    reasons.append(f"Greed ({fg})")
            
            # Liquidations
            if liquidations.get('alert_level') == 'critical':
                long_liq = liquidations.get('long_liquidations', 0)
                short_liq = liquidations.get('short_liquidations', 0)
                if long_liq > short_liq * 1.5:
                    bias_score += 1
                    reasons.append("Mass long liquidations (contrarian buy)")
                elif short_liq > long_liq * 1.5:
                    bias_score -= 1
                    reasons.append("Mass short liquidations (contrarian sell)")
            
            # Whale activity
            if whales:
                btc_whales = [w for w in whales if w.get('coin') == 'BTC' and w.get('significance') == 'high']
                if len(btc_whales) >= 3:
                    bias_score -= 1
                    reasons.append("High whale activity")
            
            # Determine bias
            if bias_score >= 2:
                bias = 'bullish'
                confidence = min(0.9, 0.5 + bias_score * 0.1)
            elif bias_score <= -2:
                bias = 'bearish'
                confidence = min(0.9, 0.5 + abs(bias_score) * 0.1)
            else:
                bias = 'neutral'
                confidence = 0.5
            
            return {
                'bias': bias,
                'confidence': confidence,
                'score': bias_score,
                'reasons': reasons,
                'fear_greed': sentiment.fear_greed_index if sentiment else 50,
                'liquidation_alert': liquidations.get('alert_level', 'normal')
            }
            
        except Exception as e:
            logger.error(f"Get trading bias error: {e}")
            return {'bias': 'neutral', 'confidence': 0.5, 'score': 0, 'reasons': []}


# Standalone test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    async def test():
        monitor = NewsMonitor()
        
        print("\n" + "="*60)
        print("    JULABA NEWS MONITOR - MARKET SUMMARY")
        print("="*60 + "\n")
        
        summary = await monitor.get_market_summary()
        
        # Sentiment
        if summary['sentiment']:
            s = summary['sentiment']
            print(f"📊 MARKET SENTIMENT")
            print(f"   Fear & Greed Index: {s['fear_greed_index']} ({s['fear_greed_label']})")
            if s['total_market_cap']:
                print(f"   Total Market Cap: ${s['total_market_cap']/1e12:.2f}T")
                print(f"   24h Change: {s['market_cap_change_24h']:.2f}%")
            if s['btc_dominance']:
                print(f"   BTC Dominance: {s['btc_dominance']:.1f}%")
        
        print(f"\n📰 NEWS SUMMARY")
        ns = summary['news_summary']
        print(f"   Total Articles: {ns['total_articles']}")
        print(f"   Bullish: {ns['bullish_count']} | Bearish: {ns['bearish_count']}")
        print(f"   Critical Alerts: {ns['critical_count']}")
        print(f"   Avg Sentiment: {ns['average_sentiment']:.2f}")
        
        if summary['critical_news']:
            print(f"\n🚨 CRITICAL NEWS:")
            for n in summary['critical_news'][:3]:
                print(f"   • {n['title'][:70]}...")
        
        print(f"\n📈 RECENT HEADLINES:")
        for n in summary['recent_news'][:5]:
            emoji = '🟢' if n['sentiment'] > 0.2 else '🔴' if n['sentiment'] < -0.2 else '⚪'
            print(f"   {emoji} {n['title'][:65]}...")
        
        # Liquidations
        liq = summary.get('liquidations', {})
        if liq:
            print(f"\n💥 LIQUIDATIONS")
            print(f"   24h Total: ${liq.get('total_24h_usd', 0)/1e6:.1f}M")
            print(f"   Long: ${liq.get('long_liquidations', 0)/1e6:.1f}M | Short: ${liq.get('short_liquidations', 0)/1e6:.1f}M")
            print(f"   Alert Level: {liq.get('alert_level', 'normal').upper()}")
        
        # Whale Alerts
        whales = summary.get('whale_alerts', [])
        if whales:
            print(f"\n🐋 WHALE ALERTS")
            for w in whales[:5]:
                if w.get('coin') == 'BTC':
                    print(f"   • {w.get('amount', 0):.0f} BTC (${w.get('usd_value', 0)/1e6:.1f}M) - {w.get('significance', 'unknown')}")
                elif w.get('type') == 'high_gas_activity':
                    print(f"   • ETH Gas: {w.get('gas_price')} gwei")
        
        # Social Sentiment
        social = summary.get('social_sentiment', {})
        if social and social.get('combined_sentiment', 0) != 0:
            print(f"\n🐦 SOCIAL SENTIMENT")
            print(f"   Combined: {social.get('combined_sentiment', 0):+.2f}")
            print(f"   Reddit: {social.get('reddit_sentiment', 0):+.2f}")
            print(f"   Volume: {social.get('social_volume', 'normal')}")
            if social.get('hot_topics'):
                print(f"   Hot Topics:")
                for topic in social['hot_topics'][:3]:
                    print(f"      • {topic[:50]}...")
        
        # LLM Analysis
        llm = summary.get('llm_analysis', {})
        if llm and llm.get('llm_used'):
            print(f"\n🤖 LLM ANALYSIS ({llm.get('llm_used', 'N/A')})")
            print(f"   Sentiment: {llm.get('sentiment', 0):+.2f}")
            print(f"   Recommendation: {llm.get('recommendation', 'N/A')}")
            print(f"   Risk Level: {llm.get('risk_level', 'N/A')}")
            if llm.get('summary'):
                print(f"   Summary: {llm.get('summary', '')[:80]}...")
        
        print(f"\n💡 RECOMMENDATION:")
        print(f"   {summary['recommendation']}")
        
        # Trading Bias (cached sync method)
        print(f"\n⚡ TRADING BIAS (cached):")
        bias = monitor.get_trading_bias()
        print(f"   Bias: {bias['bias'].upper()} (conf: {bias['confidence']:.0%})")
        print(f"   Score: {bias['score']}")
        if bias['reasons']:
            for r in bias['reasons']:
                print(f"      • {r}")
        
        print("\n" + "="*60)
    
    asyncio.run(test())

