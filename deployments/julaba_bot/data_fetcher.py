"""
MEXC Historical Data Fetcher for Julaba ML Pipeline
Fetches 90 days of 15m candles for ML training data generation.
"""

import requests
import pandas as pd
import time
import os
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger("Julaba.DataFetcher")


class MEXCDataFetcher:
    """Fetch historical OHLCV data from MEXC API."""
    
    BASE_URL = "https://api.mexc.com/api/v3/klines"
    
    def __init__(self, output_dir: str = "./historical_data"):
        self.pairs = ["LINKUSDT", "ETHUSDT", "SOLUSDT"]
        self.interval = "15m"
        self.days_back = 90
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def fetch_klines(self, symbol: str, start_time: int, end_time: int) -> list:
        """Fetch klines for a single time window."""
        params = {
            "symbol": symbol,
            "interval": self.interval,
            "startTime": start_time,
            "endTime": end_time,
            "limit": 1000
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return []
    
    def fetch_full_history(self, symbol: str) -> pd.DataFrame:
        """Fetch complete 90-day history with pagination."""
        
        end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time = int((datetime.now(timezone.utc) - timedelta(days=self.days_back)).timestamp() * 1000)
        
        all_klines = []
        current_start = start_time
        request_count = 0
        
        print(f"  Fetching {symbol} from {datetime.fromtimestamp(start_time/1000, tz=timezone.utc)} to now...")
        
        while current_start < end_time:
            klines = self.fetch_klines(symbol, current_start, end_time)
            
            if not klines:
                break
            
            all_klines.extend(klines)
            request_count += 1
            
            # Move start to last candle time + 1
            current_start = klines[-1][0] + 1
            
            # Progress indicator
            if request_count % 5 == 0:
                print(f"    {len(all_klines)} candles fetched...")
            
            # Rate limiting - be nice to the API
            time.sleep(0.2)
        
        if not all_klines:
            logger.warning(f"No data fetched for {symbol}")
            return pd.DataFrame()
        
        # MEXC API returns 8 columns for spot klines
        # [timestamp, open, high, low, close, volume, close_time, quote_volume]
        num_cols = len(all_klines[0]) if all_klines else 0
        
        if num_cols == 8:
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                      'close_time', 'quote_volume']
        elif num_cols == 12:
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                      'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                      'taker_buy_quote', 'ignore']
        else:
            # Dynamic column names
            columns = [f'col_{i}' for i in range(num_cols)]
            columns[0] = 'timestamp'
            if num_cols >= 5:
                columns[1:5] = ['open', 'high', 'low', 'close']
            if num_cols >= 6:
                columns[5] = 'volume'
        
        df = pd.DataFrame(all_klines, columns=columns)
        
        # Type conversions
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df['symbol'] = symbol
        df = df.drop_duplicates(subset=['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def fetch_all_pairs(self) -> dict:
        """Fetch history for all configured pairs."""
        data = {}
        
        for pair in self.pairs:
            print(f"\n📊 Fetching {pair}...")
            data[pair] = self.fetch_full_history(pair)
            print(f"  ✅ Got {len(data[pair])} candles")
            time.sleep(1)  # Rate limiting between pairs
        
        return data
    
    def save_to_csv(self, data: dict):
        """Save each pair to CSV."""
        for pair, df in data.items():
            if df.empty:
                continue
            filepath = self.output_dir / f"{pair}_15m_90d.csv"
            df.to_csv(filepath, index=False)
            print(f"💾 Saved {filepath}")
    
    def run(self) -> dict:
        """Run the full data fetch pipeline."""
        print("="*60)
        print("🚀 MEXC Historical Data Fetcher")
        print(f"📅 Fetching {self.days_back} days of {self.interval} data")
        print(f"📈 Pairs: {', '.join(self.pairs)}")
        print("="*60)
        
        data = self.fetch_all_pairs()
        self.save_to_csv(data)
        
        # Summary
        print("\n" + "="*60)
        print("📊 FETCH SUMMARY")
        print("="*60)
        total = 0
        for pair, df in data.items():
            print(f"  {pair}: {len(df)} candles")
            total += len(df)
        print(f"\n  TOTAL: {total} candles")
        print("="*60)
        
        return data


def fetch_btc_data(days_back: int = 90, output_dir: str = "./historical_data") -> pd.DataFrame:
    """Fetch BTC data separately for correlation analysis."""
    fetcher = MEXCDataFetcher(output_dir)
    fetcher.pairs = ["BTCUSDT"]
    fetcher.days_back = days_back
    
    data = fetcher.fetch_all_pairs()
    fetcher.save_to_csv(data)
    
    return data.get("BTCUSDT", pd.DataFrame())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    fetcher = MEXCDataFetcher()
    data = fetcher.run()
    
    # Also fetch BTC for correlation
    print("\n📊 Fetching BTC for correlation analysis...")
    btc_data = fetch_btc_data()
    print(f"  ✅ Got {len(btc_data)} BTC candles")
