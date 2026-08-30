"""
Julaba Historical ML Model
===========================
Trained on backtested historical data (90d of 15m candles).
Provides baseline win probability predictions.

This model:
1. Generates thousands of simulated trade signals from historical data
2. Labels them as WIN/LOSS based on forward price movement
3. Trains XGBoost classifier on the resulting dataset
4. Provides win probability for new signals

This model is PRE-TRAINED and does NOT change during live trading.
It provides a stable baseline that the Live ML model complements.
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

logger = logging.getLogger("Julaba.MLHistorical")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed")

try:
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Feature columns for historical model
FEATURE_COLUMNS = [
    # Core indicators
    'rsi', 'adx', 'atr_percent', 'volume_ratio', 'hurst',
    'sma_distance_percent',
    # Time
    'hour', 'day_of_week',
    # Regime one-hot
    'regime_trending', 'regime_choppy', 'regime_ranging',
    # Momentum
    'rsi_slope', 'macd_hist', 'price_momentum',
    # Volatility
    'atr_expansion', 'bb_position',
    # Volume
    'volume_trend',
    # Pattern
    'candle_strength',
    # Session
    'is_london_session', 'is_nyc_session', 'is_asia_session',
    # Price structure
    'higher_highs', 'distance_from_support', 'distance_from_resistance',
]


class HistoricalMLModel:
    """
    XGBoost model trained on historical backtested data.
    
    This is a STATIC model — trained once on historical data,
    provides stable baseline predictions during live trading.
    """
    
    MODEL_PATH = "./models/ml_historical_v2.json"
    META_PATH = "./models/ml_historical_v2_meta.json"
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_loaded = False
        self.metrics = {}
        self.feature_columns = FEATURE_COLUMNS.copy()
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model from disk."""
        if not XGBOOST_AVAILABLE:
            return
        
        model_path = Path(self.MODEL_PATH)
        meta_path = Path(self.META_PATH)
        
        if not model_path.exists():
            logger.info("Historical ML model not found — needs training")
            return
        
        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(model_path))
            
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    self.feature_columns = meta.get('feature_columns', FEATURE_COLUMNS)
                    self.metrics = meta.get('metrics', {})
                    # Load scaler params
                    scaler_data = meta.get('scaler')
                    if scaler_data and SKLEARN_AVAILABLE:
                        self.scaler = StandardScaler()
                        self.scaler.mean_ = np.array(scaler_data['mean'])
                        self.scaler.scale_ = np.array(scaler_data['scale'])
                        self.scaler.var_ = np.array(scaler_data['var'])
                        self.scaler.n_features_in_ = len(scaler_data['mean'])
                        self.scaler.n_samples_seen_ = scaler_data.get('n_samples', 1000)
            
            self.is_loaded = True
            acc = self.metrics.get('accuracy', 0)
            samples = self.metrics.get('total_samples', 0)
            logger.info(f"✅ Historical ML loaded: {acc:.1%} accuracy, {samples} training samples")
        except Exception as e:
            logger.error(f"Failed to load historical ML model: {e}")
            self.model = None
            self.is_loaded = False
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators on a DataFrame."""
        df = df.copy()
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ADX
        df['adx'] = self._calculate_adx(df, 14)
        
        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_percent'] = (df['atr'] / close) * 100
        
        # Volume
        vol_sma = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / vol_sma.replace(0, 1)
        
        # SMAs
        sma15 = close.rolling(15).mean()
        sma40 = close.rolling(40).mean()
        df['sma_distance_percent'] = ((close - sma40) / sma40.replace(0, 1)) * 100
        
        # Hurst (simplified rolling)
        df['hurst'] = self._rolling_hurst(close, 100)
        
        # Time
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
            df['hour'] = ts.dt.hour
            df['day_of_week'] = ts.dt.dayofweek
        else:
            df['hour'] = 12
            df['day_of_week'] = 2
        
        # Regime
        df['regime_trending'] = ((df['adx'] > 25) & (df['hurst'] > 0.55)).astype(int)
        df['regime_choppy'] = ((df['adx'] < 20) & (df['hurst'] < 0.45)).astype(int)
        df['regime_ranging'] = (~df['regime_trending'].astype(bool) & ~df['regime_choppy'].astype(bool)).astype(int)
        
        # RSI slope
        df['rsi_slope'] = df['rsi'].diff(5) / 5
        
        # MACD histogram
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['macd_hist'] = macd - signal
        
        # Price momentum
        df['price_momentum'] = close.pct_change(5) * 100
        
        # ATR expansion
        atr_avg = df['atr'].rolling(20).mean()
        df['atr_expansion'] = df['atr'] / atr_avg.replace(0, 1)
        
        # Bollinger Band position
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_width = (bb_upper - bb_lower).replace(0, 1e-10)
        df['bb_position'] = (close - bb_lower) / bb_width
        
        # Volume trend
        df['volume_trend'] = df['volume'].pct_change(10) * 100
        
        # Candle strength
        body = (close - df['open']).abs()
        total_range = (high - low).replace(0, 1e-10)
        df['candle_strength'] = body / total_range
        # Make negative for bearish candles
        df.loc[close < df['open'], 'candle_strength'] *= -1
        
        # Sessions
        df['is_london_session'] = df['hour'].apply(lambda h: 1 if 7 <= h <= 16 else 0)
        df['is_nyc_session'] = df['hour'].apply(lambda h: 1 if 13 <= h <= 22 else 0)
        df['is_asia_session'] = df['hour'].apply(lambda h: 1 if 0 <= h <= 9 else 0)
        
        # Higher highs / lower lows (trend structure)
        df['higher_highs'] = (high.rolling(5).max() > high.shift(5).rolling(5).max()).astype(int)
        
        # Support/resistance distance
        recent_low = low.rolling(20).min()
        recent_high = high.rolling(20).max()
        price_range = (recent_high - recent_low).replace(0, 1e-10)
        df['distance_from_support'] = (close - recent_low) / price_range
        df['distance_from_resistance'] = (recent_high - close) / price_range
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX indicator."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, 1e-10))
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, 1e-10))
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-10)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _rolling_hurst(self, series: pd.Series, window: int = 100) -> pd.Series:
        """Simplified rolling Hurst exponent."""
        result = pd.Series(0.5, index=series.index)
        
        for i in range(window, len(series)):
            segment = series.iloc[i-window:i].values
            if len(segment) < 20:
                continue
            try:
                returns = np.diff(np.log(segment + 1e-10))
                if len(returns) < 10:
                    continue
                    
                # R/S method (simplified)
                mean_r = np.mean(returns)
                deviations = np.cumsum(returns - mean_r)
                R = np.max(deviations) - np.min(deviations)
                S = np.std(returns)
                
                if S > 0 and R > 0:
                    # Simple H estimation
                    n = len(returns)
                    H = np.log(R / S) / np.log(n)
                    result.iloc[i] = np.clip(H, 0.0, 1.0)
            except Exception:
                pass
        
        return result
    
    def generate_training_data(self, csv_paths: List[str],
                                tp_percent: float = 0.5,
                                sl_percent: float = 0.8,
                                forward_bars: int = 40) -> pd.DataFrame:
        """
        Generate labeled training data from historical CSV files.
        
        For each bar, check if future price hits TP or SL first.
        Label: 1 = TP hit first (WIN), 0 = SL hit first (LOSS)
        
        Args:
            csv_paths: List of paths to historical 15m CSV files
            tp_percent: Take profit target as percentage (e.g., 0.5 = 0.5%)
            sl_percent: Stop loss as percentage  
            forward_bars: How many bars forward to check (40 bars * 15m = 10 hours)
        """
        all_samples = []
        
        for path in csv_paths:
            try:
                df = pd.read_csv(path)
                if len(df) < 200:
                    logger.warning(f"Skipping {path}: too few rows ({len(df)})")
                    continue
                
                symbol = Path(path).stem.split('_')[0]
                logger.info(f"Processing {symbol}: {len(df)} candles")
                
                # Calculate indicators
                df = self._calculate_indicators(df)
                
                # Drop warmup period
                df = df.iloc[100:].reset_index(drop=True)
                
                # Generate signals and labels
                signals = self._generate_signals_from_data(df, tp_percent, sl_percent, forward_bars, symbol)
                all_samples.extend(signals)
                
                logger.info(f"  {symbol}: {len(signals)} labeled samples")
                
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
        
        if not all_samples:
            logger.error("No training samples generated!")
            return pd.DataFrame()
        
        result_df = pd.DataFrame(all_samples)
        
        wins = (result_df['outcome'] == 1).sum()
        losses = (result_df['outcome'] == 0).sum()
        logger.info(f"Total: {len(result_df)} samples — {wins} wins ({wins/len(result_df):.1%}), {losses} losses")
        
        return result_df
    
    def _generate_signals_from_data(self, df: pd.DataFrame,
                                      tp_pct: float, sl_pct: float,
                                      forward_bars: int,
                                      symbol: str) -> List[Dict]:
        """Generate labeled signals from indicator-enriched DataFrame."""
        samples = []
        
        for i in range(0, len(df) - forward_bars, 3):  # Step by 3 to avoid overlap
            row = df.iloc[i]
            
            # Skip if indicators are NaN
            if pd.isna(row.get('rsi')) or pd.isna(row.get('adx')):
                continue
            
            entry_price = row['close']
            
            # Determine signal direction based on indicators
            rsi = row.get('rsi', 50)
            macd_hist = row.get('macd_hist', 0)
            sma_dist = row.get('sma_distance_percent', 0)
            
            # LONG signal conditions
            is_long = (rsi < 65 and macd_hist > 0 and sma_dist > -2)
            # SHORT signal conditions
            is_short = (rsi > 35 and macd_hist < 0 and sma_dist < 2)
            
            if not is_long and not is_short:
                continue
            
            direction = 'LONG' if is_long else 'SHORT'
            
            # Check forward bars for outcome
            future = df.iloc[i+1:i+1+forward_bars]
            
            if direction == 'LONG':
                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 - sl_pct / 100)
                
                # Check which hits first
                tp_hit = False
                sl_hit = False
                for _, bar in future.iterrows():
                    if bar['high'] >= tp_price:
                        tp_hit = True
                        break
                    if bar['low'] <= sl_price:
                        sl_hit = True
                        break
                
                if not tp_hit and not sl_hit:
                    # Timeout — check if in profit
                    final_price = future.iloc[-1]['close'] if len(future) > 0 else entry_price
                    outcome = 1 if final_price > entry_price else 0
                else:
                    outcome = 1 if tp_hit else 0
            else:
                tp_price = entry_price * (1 - tp_pct / 100)
                sl_price = entry_price * (1 + sl_pct / 100)
                
                tp_hit = False
                sl_hit = False
                for _, bar in future.iterrows():
                    if bar['low'] <= tp_price:
                        tp_hit = True
                        break
                    if bar['high'] >= sl_price:
                        sl_hit = True
                        break
                
                if not tp_hit and not sl_hit:
                    final_price = future.iloc[-1]['close'] if len(future) > 0 else entry_price
                    outcome = 1 if final_price < entry_price else 0
                else:
                    outcome = 1 if tp_hit else 0
            
            # Build feature dict
            sample = {
                'symbol': symbol,
                'direction': direction,
                'outcome': outcome,
            }
            
            # Add all feature columns
            for col in FEATURE_COLUMNS:
                if col in df.columns:
                    val = row.get(col, 0)
                    sample[col] = float(val) if not pd.isna(val) else 0.0
                else:
                    sample[col] = 0.0
            
            samples.append(sample)
        
        return samples
    
    def train(self, training_data: pd.DataFrame = None, 
              csv_paths: List[str] = None) -> Dict[str, Any]:
        """
        Train the historical model.
        
        Can use either:
        - Pre-generated training DataFrame
        - Raw CSV paths (will generate training data)
        """
        if not XGBOOST_AVAILABLE or not SKLEARN_AVAILABLE:
            return {'error': 'XGBoost/scikit-learn not installed'}
        
        # Generate data if needed
        if training_data is None:
            if csv_paths is None:
                # Default: use all available historical CSVs
                hist_dir = Path("./historical_data")
                csv_paths = sorted(hist_dir.glob("*_15m_*.csv"))
                csv_paths = [str(p) for p in csv_paths if 'BTC' not in p.name]  # Exclude BTC (used for correlation)
            
            logger.info(f"Generating training data from {len(csv_paths)} files...")
            training_data = self.generate_training_data(csv_paths)
        
        if training_data.empty:
            return {'error': 'No training data available'}
        
        # Ensure feature columns exist
        for col in FEATURE_COLUMNS:
            if col not in training_data.columns:
                training_data[col] = 0.0
        
        X = training_data[FEATURE_COLUMNS].copy()
        y = training_data['outcome'].copy()
        
        # Replace NaN/Inf
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Training data: {len(X)} samples, {y.sum()} wins ({y.mean():.1%})")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=FEATURE_COLUMNS)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create model
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            objective='binary:logistic',
            eval_metric='auc',
            reg_alpha=0.3,
            reg_lambda=1.5,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        
        # Cross-validation
        cv_folds = min(5, min(y.sum(), (1-y).sum()) // 2)
        cv_folds = max(2, cv_folds)
        try:
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            logger.info(f"CV Accuracy ({cv_folds}-fold): {cv_mean:.3f} +/- {cv_std:.3f}")
        except Exception:
            cv_mean = 0
            cv_std = 0
        
        # Train
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            auc = 0.5
        
        self.metrics = {
            'accuracy': float(accuracy),
            'auc_roc': float(auc),
            'cv_accuracy': float(cv_mean),
            'cv_std': float(cv_std),
            'total_samples': int(len(training_data)),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'win_rate_in_data': float(y.mean()),
            'trained_at': datetime.now(timezone.utc).isoformat(),
            'model_type': 'historical',
        }
        
        # Feature importance
        importance = dict(zip(FEATURE_COLUMNS, [float(x) for x in self.model.feature_importances_]))
        self.metrics['top_features'] = dict(sorted(importance.items(), key=lambda x: -x[1])[:8])
        
        logger.info(f"Historical ML trained: accuracy={accuracy:.1%}, AUC={auc:.3f}, samples={len(training_data)}")
        logger.info(f"Top features: {list(self.metrics['top_features'].keys())[:5]}")
        
        # Save
        self._save_model()
        self.is_loaded = True
        
        return self.metrics
    
    def _save_model(self):
        """Save model and metadata."""
        if self.model is None:
            return
        
        os.makedirs("models", exist_ok=True)
        
        try:
            self.model.save_model(self.MODEL_PATH)
        except TypeError:
            self.model.get_booster().save_model(self.MODEL_PATH)
        
        meta = {
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'version': '2.0',
            'saved_at': datetime.now(timezone.utc).isoformat(),
        }
        
        if self.scaler is not None:
            meta['scaler'] = {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist(),
                'var': self.scaler.var_.tolist(),
                'n_samples': int(getattr(self.scaler, 'n_samples_seen_', 1000)) if isinstance(getattr(self.scaler, 'n_samples_seen_', 1000), (int, np.integer)) else 1000,
            }
        
        with open(self.META_PATH, 'w') as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"Historical ML model saved to {self.MODEL_PATH}")
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get win probability from historical model.
        
        Args:
            features: Dict with indicator values
            
        Returns:
            Dict with win probability and confidence
        """
        if not self.is_loaded or self.model is None:
            return {
                'available': False,
                'win_probability': 0.5,
                'confidence': 'N/A',
                'reason': 'Historical model not loaded'
            }
        
        try:
            # Build feature array
            X = []
            for col in self.feature_columns:
                val = features.get(col, 0.0)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = 0.0
                X.append(float(val))
            
            X = np.array([X])
            
            # Scale if scaler available
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            prob = float(self.model.predict_proba(X)[0][1])
            
            if prob >= 0.65:
                confidence = 'HIGH'
            elif prob >= 0.50:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
            
            return {
                'available': True,
                'win_probability': round(prob, 4),
                'confidence': confidence,
                'recommendation': 'TAKE' if prob >= 0.55 else ('SKIP' if prob < 0.45 else 'NEUTRAL'),
                'model_accuracy': self.metrics.get('accuracy', 0),
                'training_samples': self.metrics.get('total_samples', 0),
            }
        
        except Exception as e:
            logger.error(f"Historical ML prediction error: {e}")
            return {
                'available': False,
                'win_probability': 0.5,
                'confidence': 'ERROR',
                'reason': str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get model status."""
        return {
            'type': 'historical',
            'is_loaded': self.is_loaded,
            'accuracy': self.metrics.get('accuracy', 0),
            'auc_roc': self.metrics.get('auc_roc', 0),
            'training_samples': self.metrics.get('total_samples', 0),
            'trained_at': self.metrics.get('trained_at', 'never'),
            'top_features': list(self.metrics.get('top_features', {}).keys())[:5],
        }


# Singleton
_historical_model: Optional[HistoricalMLModel] = None

def get_historical_model() -> HistoricalMLModel:
    """Get the global historical ML model instance."""
    global _historical_model
    if _historical_model is None:
        _historical_model = HistoricalMLModel()
    return _historical_model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    print("=" * 60)
    print("🧠 TRAINING HISTORICAL ML MODEL")
    print("=" * 60)
    
    model = HistoricalMLModel()
    
    # Find all historical CSVs (prefer 90d, fallback to 30d)
    hist_dir = Path("./historical_data")
    csv_90d = sorted(hist_dir.glob("*_15m_90d.csv"))
    csv_30d = sorted(hist_dir.glob("*_15m_30d.csv"))
    
    # Use 90d if available, otherwise 30d
    csv_paths = [str(p) for p in csv_90d if 'BTC' not in p.name]
    if not csv_paths:
        csv_paths = [str(p) for p in csv_30d if 'BTC' not in p.name]
    
    print(f"\nUsing {len(csv_paths)} data files:")
    for p in csv_paths:
        print(f"  📊 {Path(p).name}")
    
    metrics = model.train(csv_paths=csv_paths)
    
    if 'error' in metrics:
        print(f"\n❌ Training failed: {metrics['error']}")
    else:
        print(f"\n✅ Historical ML Model Trained!")
        print(f"   Accuracy: {metrics['accuracy']:.1%}")
        print(f"   AUC-ROC: {metrics['auc_roc']:.3f}")
        print(f"   Samples: {metrics['total_samples']}")
        print(f"   Top features: {list(metrics.get('top_features', {}).keys())[:5]}")
