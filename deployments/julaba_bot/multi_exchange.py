"""
Multi-Exchange Support for Julaba Trading Bot
==============================================
Unified interface for multiple exchanges (Bybit, Binance, OKX).

Features:
- Unified API for all exchanges
- Exchange-specific configurations
- Automatic failover
- Best execution routing (optional)
"""

import ccxt.async_support as ccxt
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger("Julaba.MultiExchange")

class Exchange(Enum):
    BYBIT = "bybit"
    BINANCE = "binance"
    OKX = "okx"

@dataclass
class ExchangeConfig:
    """Configuration for an exchange"""
    exchange: Exchange
    api_key: str
    api_secret: str
    passphrase: str = ""  # OKX requires this
    testnet: bool = False
    enabled: bool = True
    default_leverage: int = 10
    
@dataclass
class Position:
    """Unified position structure"""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: int
    liquidation_price: float
    exchange: str
    raw: dict = field(default_factory=dict)

@dataclass
class Order:
    """Unified order structure"""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    type: str  # 'market', 'limit'
    price: float
    amount: float
    filled: float
    status: str
    exchange: str
    timestamp: int
    raw: dict = field(default_factory=dict)

@dataclass
class Balance:
    """Unified balance structure"""
    exchange: str
    total: float
    free: float
    used: float
    currency: str = "USDT"

class ExchangeClient:
    """
    Wrapper for a single exchange connection
    """
    
    def __init__(self, config: ExchangeConfig):
        self.config = config
        self.exchange = None
        self.connected = False
        self._last_error = None
    
    async def connect(self) -> bool:
        """Initialize connection to exchange"""
        try:
            exchange_class = getattr(ccxt, self.config.exchange.value)
            
            options = {
                'apiKey': self.config.api_key,
                'secret': self.config.api_secret,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'swap',  # Futures/Perpetuals
                    'adjustForTimeDifference': True,
                }
            }
            
            # Exchange-specific configs
            if self.config.exchange == Exchange.BYBIT:
                options['options']['recvWindow'] = 60000
                if self.config.testnet:
                    options['sandbox'] = True
            
            elif self.config.exchange == Exchange.BINANCE:
                options['options']['defaultType'] = 'future'
                if self.config.testnet:
                    options['sandbox'] = True
            
            elif self.config.exchange == Exchange.OKX:
                options['password'] = self.config.passphrase
                options['options']['defaultType'] = 'swap'
                if self.config.testnet:
                    options['sandbox'] = True
            
            self.exchange = exchange_class(options)
            
            # Load markets
            await self.exchange.load_markets()
            
            self.connected = True
            logger.info(f"✅ Connected to {self.config.exchange.value}")
            return True
            
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"❌ Failed to connect to {self.config.exchange.value}: {e}")
            return False
    
    async def close(self):
        """Close connection"""
        if self.exchange:
            await self.exchange.close()
            self.connected = False
    
    # ==================== BALANCE ====================
    
    async def get_balance(self) -> Optional[Balance]:
        """Get USDT balance"""
        try:
            balance = await self.exchange.fetch_balance()
            
            # Different exchanges have different structures
            if self.config.exchange == Exchange.BYBIT:
                usdt = balance.get('USDT', {})
                return Balance(
                    exchange=self.config.exchange.value,
                    total=usdt.get('total', 0),
                    free=usdt.get('free', 0),
                    used=usdt.get('used', 0)
                )
            
            elif self.config.exchange == Exchange.BINANCE:
                usdt = balance.get('USDT', {})
                return Balance(
                    exchange=self.config.exchange.value,
                    total=usdt.get('total', 0),
                    free=usdt.get('free', 0),
                    used=usdt.get('used', 0)
                )
            
            elif self.config.exchange == Exchange.OKX:
                usdt = balance.get('USDT', {})
                return Balance(
                    exchange=self.config.exchange.value,
                    total=usdt.get('total', 0),
                    free=usdt.get('free', 0),
                    used=usdt.get('used', 0)
                )
            
        except Exception as e:
            logger.error(f"Failed to get balance from {self.config.exchange.value}: {e}")
            return None
    
    # ==================== POSITIONS ====================
    
    async def get_positions(self, symbol: str = None) -> List[Position]:
        """Get open positions"""
        try:
            if symbol:
                positions = await self.exchange.fetch_positions([symbol])
            else:
                positions = await self.exchange.fetch_positions()
            
            result = []
            for pos in positions:
                if pos.get('contracts', 0) > 0 or abs(pos.get('notional', 0)) > 0:
                    result.append(Position(
                        symbol=pos.get('symbol', ''),
                        side=pos.get('side', ''),
                        size=abs(pos.get('contracts', 0)),
                        entry_price=pos.get('entryPrice', 0),
                        mark_price=pos.get('markPrice', 0),
                        unrealized_pnl=pos.get('unrealizedPnl', 0),
                        leverage=pos.get('leverage', 1),
                        liquidation_price=pos.get('liquidationPrice', 0),
                        exchange=self.config.exchange.value,
                        raw=pos
                    ))
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get positions from {self.config.exchange.value}: {e}")
            return []
    
    # ==================== ORDERS ====================
    
    async def create_market_order(self, symbol: str, side: str, amount: float,
                                   reduce_only: bool = False) -> Optional[Order]:
        """Create market order"""
        try:
            params = {}
            
            if self.config.exchange == Exchange.BYBIT:
                params['reduceOnly'] = reduce_only
            elif self.config.exchange == Exchange.BINANCE:
                params['reduceOnly'] = reduce_only
            elif self.config.exchange == Exchange.OKX:
                params['reduceOnly'] = reduce_only
            
            order = await self.exchange.create_market_order(
                symbol, side, amount, params=params
            )
            
            return Order(
                id=order.get('id', ''),
                symbol=order.get('symbol', ''),
                side=order.get('side', ''),
                type='market',
                price=order.get('average', order.get('price', 0)),
                amount=order.get('amount', 0),
                filled=order.get('filled', 0),
                status=order.get('status', ''),
                exchange=self.config.exchange.value,
                timestamp=order.get('timestamp', 0),
                raw=order
            )
            
        except Exception as e:
            logger.error(f"Failed to create order on {self.config.exchange.value}: {e}")
            return None
    
    async def create_limit_order(self, symbol: str, side: str, amount: float,
                                  price: float, reduce_only: bool = False) -> Optional[Order]:
        """Create limit order"""
        try:
            params = {}
            
            if self.config.exchange == Exchange.BYBIT:
                params['reduceOnly'] = reduce_only
            elif self.config.exchange == Exchange.BINANCE:
                params['reduceOnly'] = reduce_only
                params['timeInForce'] = 'GTC'
            elif self.config.exchange == Exchange.OKX:
                params['reduceOnly'] = reduce_only
            
            order = await self.exchange.create_limit_order(
                symbol, side, amount, price, params=params
            )
            
            return Order(
                id=order.get('id', ''),
                symbol=order.get('symbol', ''),
                side=order.get('side', ''),
                type='limit',
                price=order.get('price', 0),
                amount=order.get('amount', 0),
                filled=order.get('filled', 0),
                status=order.get('status', ''),
                exchange=self.config.exchange.value,
                timestamp=order.get('timestamp', 0),
                raw=order
            )
            
        except Exception as e:
            logger.error(f"Failed to create limit order on {self.config.exchange.value}: {e}")
            return None
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel order"""
        try:
            await self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order on {self.config.exchange.value}: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: str = None) -> bool:
        """Cancel all open orders"""
        try:
            if symbol:
                await self.exchange.cancel_all_orders(symbol)
            else:
                # Get all open orders and cancel
                orders = await self.exchange.fetch_open_orders()
                for order in orders:
                    await self.exchange.cancel_order(order['id'], order['symbol'])
            return True
        except Exception as e:
            logger.error(f"Failed to cancel orders on {self.config.exchange.value}: {e}")
            return False
    
    # ==================== LEVERAGE ====================
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol"""
        try:
            if self.config.exchange == Exchange.BYBIT:
                await self.exchange.set_leverage(leverage, symbol)
            elif self.config.exchange == Exchange.BINANCE:
                await self.exchange.fapiPrivate_post_leverage({
                    'symbol': symbol.replace('/', '').replace(':USDT', ''),
                    'leverage': leverage
                })
            elif self.config.exchange == Exchange.OKX:
                await self.exchange.set_leverage(leverage, symbol, params={'mgnMode': 'cross'})
            
            return True
        except Exception as e:
            logger.warning(f"Failed to set leverage on {self.config.exchange.value}: {e}")
            return False
    
    # ==================== MARKET DATA ====================
    
    async def get_ticker(self, symbol: str) -> Optional[dict]:
        """Get current ticker"""
        try:
            return await self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Failed to get ticker: {e}")
            return None
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '15m', 
                        limit: int = 100) -> List[list]:
        """Get OHLCV data"""
        try:
            return await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception as e:
            logger.error(f"Failed to get OHLCV: {e}")
            return []
    
    # ==================== SYMBOL CONVERSION ====================
    
    def normalize_symbol(self, symbol: str) -> str:
        """Convert symbol to exchange format"""
        # Remove common suffixes
        symbol = symbol.replace(':USDT', '').replace('/USDT', '')
        
        if self.config.exchange == Exchange.BYBIT:
            return f"{symbol}/USDT:USDT"
        elif self.config.exchange == Exchange.BINANCE:
            return f"{symbol}/USDT:USDT"
        elif self.config.exchange == Exchange.OKX:
            return f"{symbol}/USDT:USDT"
        
        return symbol


class MultiExchangeManager:
    """
    Manages multiple exchange connections with unified interface
    """
    
    def __init__(self, config_path: str = "exchanges_config.json"):
        self.config_path = Path(config_path)
        self.exchanges: Dict[str, ExchangeClient] = {}
        self.primary_exchange: Optional[str] = None
        
        # Load config
        self._load_config()
    
    def _load_config(self):
        """Load exchange configurations"""
        if not self.config_path.exists():
            # Create default config
            default_config = {
                "primary": "bybit",
                "exchanges": {
                    "bybit": {
                        "enabled": True,
                        "api_key": "",
                        "api_secret": "",
                        "testnet": False,
                        "leverage": 10
                    },
                    "binance": {
                        "enabled": False,
                        "api_key": "",
                        "api_secret": "",
                        "testnet": False,
                        "leverage": 10
                    },
                    "okx": {
                        "enabled": False,
                        "api_key": "",
                        "api_secret": "",
                        "passphrase": "",
                        "testnet": False,
                        "leverage": 10
                    }
                }
            }
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default exchange config at {self.config_path}")
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all enabled exchanges"""
        try:
            with open(self.config_path) as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load exchange config: {e}")
            return {}
        
        self.primary_exchange = config.get('primary', 'bybit')
        results = {}
        
        for name, exc_config in config.get('exchanges', {}).items():
            if not exc_config.get('enabled', False):
                continue
            if not exc_config.get('api_key') or not exc_config.get('api_secret'):
                logger.warning(f"Skipping {name}: Missing API credentials")
                continue
            
            try:
                exchange_enum = Exchange(name)
                client_config = ExchangeConfig(
                    exchange=exchange_enum,
                    api_key=exc_config['api_key'],
                    api_secret=exc_config['api_secret'],
                    passphrase=exc_config.get('passphrase', ''),
                    testnet=exc_config.get('testnet', False),
                    enabled=True,
                    default_leverage=exc_config.get('leverage', 10)
                )
                
                client = ExchangeClient(client_config)
                success = await client.connect()
                
                if success:
                    self.exchanges[name] = client
                
                results[name] = success
                
            except Exception as e:
                logger.error(f"Failed to setup {name}: {e}")
                results[name] = False
        
        logger.info(f"📊 Multi-exchange: {len(self.exchanges)} connected, primary: {self.primary_exchange}")
        return results
    
    async def close_all(self):
        """Close all connections"""
        for client in self.exchanges.values():
            await client.close()
        self.exchanges.clear()
    
    def get_primary(self) -> Optional[ExchangeClient]:
        """Get primary exchange client"""
        return self.exchanges.get(self.primary_exchange)
    
    def get_exchange(self, name: str) -> Optional[ExchangeClient]:
        """Get specific exchange client"""
        return self.exchanges.get(name)
    
    # ==================== AGGREGATED OPERATIONS ====================
    
    async def get_total_balance(self) -> Tuple[float, Dict[str, Balance]]:
        """Get total balance across all exchanges"""
        balances = {}
        total = 0
        
        for name, client in self.exchanges.items():
            balance = await client.get_balance()
            if balance:
                balances[name] = balance
                total += balance.total
        
        return total, balances
    
    async def get_all_positions(self) -> Dict[str, List[Position]]:
        """Get all positions across all exchanges"""
        all_positions = {}
        
        for name, client in self.exchanges.items():
            positions = await client.get_positions()
            if positions:
                all_positions[name] = positions
        
        return all_positions
    
    async def get_best_price(self, symbol: str) -> Tuple[str, float, float]:
        """Get best bid/ask across exchanges"""
        best_bid = 0
        best_ask = float('inf')
        best_exchange = self.primary_exchange
        
        for name, client in self.exchanges.items():
            ticker = await client.get_ticker(client.normalize_symbol(symbol))
            if ticker:
                bid = ticker.get('bid', 0)
                ask = ticker.get('ask', float('inf'))
                
                if bid > best_bid:
                    best_bid = bid
                    best_exchange = name
                if ask < best_ask:
                    best_ask = ask
        
        return best_exchange, best_bid, best_ask
    
    # ==================== SMART ORDER ROUTING ====================
    
    async def execute_best_price(self, symbol: str, side: str, amount: float) -> Optional[Order]:
        """Execute on exchange with best price"""
        best_exchange, best_bid, best_ask = await self.get_best_price(symbol)
        
        client = self.exchanges.get(best_exchange)
        if not client:
            client = self.get_primary()
        
        if client:
            normalized_symbol = client.normalize_symbol(symbol)
            return await client.create_market_order(normalized_symbol, side, amount)
        
        return None
    
    async def execute_on_primary(self, symbol: str, side: str, amount: float) -> Optional[Order]:
        """Execute on primary exchange only"""
        client = self.get_primary()
        if client:
            normalized_symbol = client.normalize_symbol(symbol)
            return await client.create_market_order(normalized_symbol, side, amount)
        return None
    
    # ==================== STATUS ====================
    
    def get_status(self) -> dict:
        """Get status of all exchanges"""
        return {
            'primary': self.primary_exchange,
            'connected': list(self.exchanges.keys()),
            'total_exchanges': len(self.exchanges),
            'exchanges': {
                name: {
                    'connected': client.connected,
                    'last_error': client._last_error
                }
                for name, client in self.exchanges.items()
            }
        }


# CLI for testing
if __name__ == "__main__":
    async def test():
        manager = MultiExchangeManager()
        results = await manager.connect_all()
        print(f"Connection results: {results}")
        
        total, balances = await manager.get_total_balance()
        print(f"Total balance: ${total:.2f}")
        
        for name, balance in balances.items():
            print(f"  {name}: ${balance.total:.2f}")
        
        await manager.close_all()
    
    asyncio.run(test())
