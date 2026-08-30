"""
Julaba Live ML Model
=====================
Auto-trains on YOUR actual live trade data.
Learns from real outcomes, not simulated backtests.

This model:
1. Records every trade's entry conditions + outcome
2. Auto-trains once enough samples accumulate (30+ trades)
3. Re-trains after every 10 new trades
4. Adapts to current market conditions (recent data weighted more)

This model IMPROVES OVER TIME as it sees more of YOUR trades.
Combined with the Historical model, it provides dual ML intelligence.
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from collections import deque

logger = logging.getLogger("Julaba.MLLive")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Feature columns for live model — matches what the bot has at entry time
LIVE_FEATURE_COLUMNS = [
    # Core indicators (always available from bot's bars_agg)
    'rsi', 'adx', 'atr_percent', 'volume_ratio', 'hurst',
    'sma_distance_percent',
    # Time
    'hour', 'day_of_week', 'is_weekend',
    # Regime
    'regime_trending', 'regime_choppy', 'regime_ranging',
    # Direction
    'is_long',
    # Momentum
    'rsi_slope', 'macd_hist', 'price_momentum',
    # Volatility
    'atr_expansion', 'bb_position',
    # Volume
    'volume_trend',
    # Market context
    'btc_correlation',
    # AI & system scores
    'ai_confidence', 'system_score', 'tech_score',
    # Session
    'is_london_session', 'is_nyc_session', 'is_asia_session',
]

MIN_SAMPLES_TO_TRAIN = 30    # Start training after 30 trades
RETRAIN_INTERVAL = 10        # Retrain every 10 new trades
SAMPLES_FILE = "./models/ml_live_samples.json"
MODEL_PATH = "./models/ml_live_v1.json"
META_PATH = "./models/ml_live_v1_meta.json"


class LiveMLModel:
    """
    ML model that learns from YOUR live trading outcomes.
    
    Auto-trains and improves as samples grow.
    Recent trades weighted more heavily than old ones.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.metrics = {}
        self.feature_columns = LIVE_FEATURE_COLUMNS.copy()
        self.samples: List[Dict] = []
        self.trades_since_last_train = 0
        self.prediction_log = deque(maxlen=500)
        
        # Load existing samples
        self._load_samples()
        
        # Load trained model if exists
        self._load_model()
    
    def _load_samples(self):
        """Load existing trade samples from disk."""
        try:
            if os.path.exists(SAMPLES_FILE):
                with open(SAMPLES_FILE, 'r') as f:
                    self.samples = json.load(f)
                logger.info(f"📊 Live ML: Loaded {len(self.samples)} trade samples")
            
            # Also import from old ml_samples.json (indicator.py format)
            old_samples_file = "./models/ml_samples.json"
            if os.path.exists(old_samples_file) and not self.samples:
                self._import_old_samples(old_samples_file)
        except Exception as e:
            logger.error(f"Failed to load live ML samples: {e}")
            self.samples = []
    
    def _import_old_samples(self, path: str):
        """Import samples from the old ml_samples.json format."""
        try:
            imported = 0
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        old = json.loads(line)
                        features = old.get('features', {})
                        
                        # Map old feature names to new
                        sample = {
                            'timestamp': old.get('timestamp', ''),
                            'outcome': old.get('label', 0),
                            'source': 'migrated',
                            'features': {
                                'rsi': features.get('rsi', 50),
                                'adx': features.get('adx', 0),
                                'atr_percent': 0,  # Not in old format
                                'volume_ratio': features.get('volume_ratio', 1),
                                'hurst': features.get('hurst', 0.5),
                                'sma_distance_percent': features.get('price_vs_sma40', 0),
                                'hour': max(0, min(23, int((features.get('hour_of_day', 0) + 1) * 12))),
                                'day_of_week': 3,
                                'is_weekend': 0,
                                'regime_trending': 1 if features.get('adx', 0) > 25 else 0,
                                'regime_choppy': 1 if features.get('adx', 0) < 20 else 0,
                                'regime_ranging': 0,
                                'is_long': 1,
                                'rsi_slope': features.get('rsi_divergence', 0),
                                'macd_hist': 0,
                                'price_momentum': features.get('momentum_strength', 0),
                                'atr_expansion': features.get('range_expansion', 1),
                                'bb_position': 0.5,
                                'volume_trend': features.get('volume_trend', 0),
                                'btc_correlation': features.get('btc_correlation', 0),
                                'ai_confidence': 0,
                                'system_score': 0,
                                'tech_score': 0,
                                'is_london_session': 0,
                                'is_nyc_session': 0,
                                'is_asia_session': 0,
                            }
                        }
                        self.samples.append(sample)
                        imported += 1
                    except json.JSONDecodeError:
                        continue
            
            if imported > 0:
                logger.info(f"📥 Imported {imported} samples from old ML format")
                self._save_samples()
        except Exception as e:
            logger.warning(f"Could not import old ML samples: {e}")
    
    def _load_model(self):
        """Load pre-trained live model."""
        if not XGBOOST_AVAILABLE:
            return
        
        if not os.path.exists(MODEL_PATH):
            return
        
        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(MODEL_PATH)
            
            if os.path.exists(META_PATH):
                with open(META_PATH, 'r') as f:
                    meta = json.load(f)
                    self.metrics = meta.get('metrics', {})
                    scaler_data = meta.get('scaler')
                    if scaler_data and SKLEARN_AVAILABLE:
                        self.scaler = StandardScaler()
                        self.scaler.mean_ = np.array(scaler_data['mean'])
                        self.scaler.scale_ = np.array(scaler_data['scale'])
                        self.scaler.var_ = np.array(scaler_data['var'])
                        self.scaler.n_features_in_ = len(scaler_data['mean'])
                        self.scaler.n_samples_seen_ = scaler_data.get('n_samples', 100)
            
            self.is_trained = True
            logger.info(f"✅ Live ML model loaded: {self.metrics.get('accuracy', 0):.1%} accuracy, "
                       f"{len(self.samples)} samples")
        except Exception as e:
            logger.error(f"Failed to load live ML model: {e}")
            self.model = None
            self.is_trained = False
    
    def _save_samples(self):
        """Save samples to disk."""
        try:
            os.makedirs("models", exist_ok=True)
            with open(SAMPLES_FILE, 'w') as f:
                json.dump(self.samples, f, indent=1)
        except Exception as e:
            logger.error(f"Failed to save live ML samples: {e}")
    
    def _save_model(self):
        """Save trained model and metadata."""
        if self.model is None:
            return
        
        os.makedirs("models", exist_ok=True)
        
        try:
            self.model.save_model(MODEL_PATH)
        except TypeError:
            self.model.get_booster().save_model(MODEL_PATH)
        
        meta = {
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'version': '1.0',
            'saved_at': datetime.now(timezone.utc).isoformat(),
        }
        
        if self.scaler is not None:
            meta['scaler'] = {
                'mean': self.scaler.mean_.tolist(),
                'scale': self.scaler.scale_.tolist(),
                'var': self.scaler.var_.tolist(),
                'n_samples': int(getattr(self.scaler, 'n_samples_seen_', 100)) if isinstance(getattr(self.scaler, 'n_samples_seen_', 100), (int, np.integer)) else 100,
            }
        
        with open(META_PATH, 'w') as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"Live ML model saved ({len(self.samples)} samples, {self.metrics.get('accuracy', 0):.1%} accuracy)")
    
    def record_trade(self, entry_features: Dict[str, Any], outcome: bool,
                     pnl: float = 0, symbol: str = '', side: str = '',
                     reason: str = ''):
        """
        Record a completed trade for ML learning.
        
        Args:
            entry_features: Dict with indicator values at entry time
            outcome: True = win, False = loss
            pnl: Dollar PnL
            symbol: Trading pair
            side: LONG or SHORT
            reason: Exit reason
        """
        # Ensure feature dict is properly formatted
        features = {}
        for col in LIVE_FEATURE_COLUMNS:
            val = entry_features.get(col, 0.0)
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                val = 0.0
            features[col] = float(val)
        
        sample = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol,
            'side': side,
            'outcome': 1 if outcome else 0,
            'pnl': pnl,
            'exit_reason': reason,
            'source': 'live',
            'features': features,
        }
        
        self.samples.append(sample)
        self.trades_since_last_train += 1
        self._save_samples()
        
        total = len(self.samples)
        wins = sum(1 for s in self.samples if s['outcome'] == 1)
        logger.info(f"📊 Live ML: Recorded {'WIN' if outcome else 'LOSS'} — "
                    f"Total: {total} samples ({wins} wins, {total-wins} losses)")
        
        # Auto-train check
        should_train = False
        if total >= MIN_SAMPLES_TO_TRAIN:
            if not self.is_trained:
                should_train = True
                logger.info(f"🧠 Live ML: First training triggered! ({total} samples)")
            elif self.trades_since_last_train >= RETRAIN_INTERVAL:
                should_train = True
                logger.info(f"🧠 Live ML: Retraining triggered ({self.trades_since_last_train} new trades)")
        else:
            remaining = MIN_SAMPLES_TO_TRAIN - total
            logger.info(f"🧠 Live ML: {remaining} more trades needed before first training")
        
        if should_train:
            self._auto_train()
    
    def _auto_train(self):
        """Auto-train on accumulated samples."""
        if not XGBOOST_AVAILABLE or not SKLEARN_AVAILABLE:
            logger.warning("Cannot train: XGBoost/sklearn not available")
            return
        
        if len(self.samples) < MIN_SAMPLES_TO_TRAIN:
            return
        
        try:
            # Build training DataFrame
            rows = []
            for s in self.samples:
                row = s.get('features', {}).copy()
                row['outcome'] = s['outcome']
                
                # Weight recent trades more heavily
                ts = s.get('timestamp', '')
                try:
                    sample_age_days = (datetime.now(timezone.utc) - 
                                       datetime.fromisoformat(ts.replace('Z', '+00:00'))).days
                    # Recent = higher weight: 1.0 for today, 0.5 for 30 days ago
                    row['sample_weight'] = max(0.3, 1.0 - (sample_age_days / 60.0))
                except Exception:
                    row['sample_weight'] = 0.5
                
                rows.append(row)
            
            df = pd.DataFrame(rows)
            
            # Ensure columns
            for col in LIVE_FEATURE_COLUMNS:
                if col not in df.columns:
                    df[col] = 0.0
            
            X = df[LIVE_FEATURE_COLUMNS].fillna(0).replace([np.inf, -np.inf], 0)
            y = df['outcome']
            weights = df.get('sample_weight', pd.Series([1.0] * len(df)))
            
            # Check class balance
            n_wins = y.sum()
            n_losses = len(y) - n_wins
            if min(n_wins, n_losses) < 3:
                logger.warning(f"Live ML: Too few of one class (wins={n_wins}, losses={n_losses}). Skipping training.")
                return
            
            # Scale
            self.scaler = StandardScaler()
            X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=LIVE_FEATURE_COLUMNS)
            
            # Model with conservative settings (small dataset)
            n_trees = min(100, max(20, len(X) // 2))
            max_d = min(4, max(2, len(X) // 15))
            
            self.model = xgb.XGBClassifier(
                n_estimators=n_trees,
                max_depth=max_d,
                learning_rate=0.05,
                objective='binary:logistic',
                eval_metric='auc',
                reg_alpha=0.5,       # More regularization (small dataset)
                reg_lambda=2.0,
                min_child_weight=max(3, len(X) // 20),
                subsample=0.8,
                colsample_bytree=0.7,
                random_state=42,
            )
            
            # Train with sample weights
            self.model.fit(X_scaled, y, sample_weight=weights)
            
            # Evaluate
            y_pred = self.model.predict(X_scaled)
            train_accuracy = accuracy_score(y, y_pred)
            
            # Cross-validation if enough samples
            cv_accuracy = 0
            auc_roc = 0.5
            if len(y) >= 30 and min(n_wins, n_losses) >= 5:
                try:
                    cv_folds = min(3, min(n_wins, n_losses))
                    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(self.model, X_scaled, y, cv=skf, scoring='accuracy')
                    cv_accuracy = cv_scores.mean()
                    # AUC-ROC (Notebook #2 §10: ML performance tracking)
                    try:
                        y_proba = self.model.predict_proba(X_scaled)[:, 1]
                        auc_roc = float(roc_auc_score(y, y_proba))
                    except Exception:
                        auc_roc = 0.5
                except Exception:
                    cv_accuracy = 0
            
            # === ML AUTO-DISABLE CHECK (Notebook #2 §10) ===
            # If AUC-ROC < 0.52, model is worse than coin flip — disable veto
            if auc_roc < 0.52 and len(y) >= 30:
                logger.warning(f"⚠️ Live ML AUC-ROC={auc_roc:.3f} < 0.52 — model worse than coin flip, disabling predictions")
                self.is_trained = False
                self._save_model()
                return
            
            # Feature importance
            importance = dict(zip(LIVE_FEATURE_COLUMNS, [float(x) for x in self.model.feature_importances_]))
            top_features = dict(sorted(importance.items(), key=lambda x: -x[1])[:8])
            
            self.metrics = {
                'accuracy': float(train_accuracy),
                'cv_accuracy': float(cv_accuracy),
                'auc_roc': float(auc_roc),
                'total_samples': len(self.samples),
                'wins': int(n_wins),
                'losses': int(n_losses),
                'win_rate': float(n_wins / len(y)),
                'trained_at': datetime.now(timezone.utc).isoformat(),
                'model_type': 'live',
                'n_trees': n_trees,
                'max_depth': max_d,
                'top_features': top_features,
            }
            
            self.is_trained = True
            self.trades_since_last_train = 0
            self._save_model()
            
            logger.info(f"✅ Live ML trained: accuracy={train_accuracy:.1%}, "
                       f"CV={cv_accuracy:.1%}, AUC-ROC={auc_roc:.3f}, samples={len(self.samples)}, "
                       f"top features: {list(top_features.keys())[:3]}")
            
        except Exception as e:
            logger.error(f"Live ML training error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get win probability from live model.
        
        Args:
            features: Dict with current indicator values
            
        Returns:
            Dict with prediction details
        """
        if not self.is_trained or self.model is None:
            remaining = max(0, MIN_SAMPLES_TO_TRAIN - len(self.samples))
            return {
                'available': False,
                'win_probability': 0.5,
                'confidence': 'N/A',
                'reason': f'Need {remaining} more trades to train' if remaining > 0 else 'Model not trained',
                'total_samples': len(self.samples),
            }
        
        try:
            X = []
            for col in self.feature_columns:
                val = features.get(col, 0.0)
                if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                    val = 0.0
                X.append(float(val))
            
            X = np.array([X])
            
            if self.scaler is not None:
                X = self.scaler.transform(X)
            
            prob = float(self.model.predict_proba(X)[0][1])
            
            if prob >= 0.65:
                confidence = 'HIGH'
            elif prob >= 0.50:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
            
            result = {
                'available': True,
                'win_probability': round(prob, 4),
                'confidence': confidence,
                'recommendation': 'TAKE' if prob >= 0.55 else ('SKIP' if prob < 0.45 else 'NEUTRAL'),
                'model_accuracy': self.metrics.get('accuracy', 0),
                'cv_accuracy': self.metrics.get('cv_accuracy', 0),
                'training_samples': len(self.samples),
            }
            
            # Log prediction
            self.prediction_log.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'probability': prob,
                'confidence': confidence,
            })
            
            return result
        
        except Exception as e:
            logger.error(f"Live ML prediction error: {e}")
            return {
                'available': False,
                'win_probability': 0.5,
                'confidence': 'ERROR',
                'reason': str(e),
                'total_samples': len(self.samples),
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get model status."""
        remaining = max(0, MIN_SAMPLES_TO_TRAIN - len(self.samples))
        wins = sum(1 for s in self.samples if s.get('outcome') == 1)
        losses = len(self.samples) - wins
        
        status = {
            'type': 'live',
            'is_trained': self.is_trained,
            'total_samples': len(self.samples),
            'wins': wins,
            'losses': losses,
            'win_rate': f"{wins/(wins+losses):.1%}" if (wins+losses) > 0 else 'N/A',
            'samples_until_training': remaining,
            'trades_since_last_train': self.trades_since_last_train,
            'retrain_interval': RETRAIN_INTERVAL,
        }
        
        if self.is_trained:
            status.update({
                'accuracy': self.metrics.get('accuracy', 0),
                'cv_accuracy': self.metrics.get('cv_accuracy', 0),
                'trained_at': self.metrics.get('trained_at', 'unknown'),
                'top_features': list(self.metrics.get('top_features', {}).keys())[:5],
            })
        
        return status
    
    def force_train(self) -> Dict[str, Any]:
        """Force retraining (called from Telegram command etc.)."""
        if len(self.samples) < 10:
            return {'error': f'Need at least 10 samples, have {len(self.samples)}'}
        
        self._auto_train()
        
        if self.is_trained:
            return {'success': True, 'metrics': self.metrics}
        else:
            return {'error': 'Training failed'}


# Singleton
_live_model: Optional[LiveMLModel] = None

def get_live_model() -> LiveMLModel:
    """Get the global live ML model instance."""
    global _live_model
    if _live_model is None:
        _live_model = LiveMLModel()
    return _live_model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    model = LiveMLModel()
    print(f"\nLive ML Status:")
    status = model.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
