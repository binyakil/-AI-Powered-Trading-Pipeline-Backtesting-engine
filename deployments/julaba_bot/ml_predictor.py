"""
ML Predictor Module for Julaba Live Bot Integration
Provides win probability predictions for incoming signals.
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("Julaba.MLPredictor")

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed. ML predictions disabled.")


class MLPredictor:
    """
    ML Predictor for live trade signal evaluation.
    
    This module:
    1. Loads the trained XGBoost model
    2. Converts signal data to features
    3. Returns win probability predictions
    
    IMPORTANT: This is INFORMATIONAL ONLY by default.
    Set influence_weight > 0 to let ML affect trade decisions.
    """
    
    # Enhanced feature columns matching ml_trainer.py
    FEATURE_COLUMNS = [
        # Core indicators
        'entry_atr_percent',
        'entry_rsi',
        'entry_adx',
        'entry_volume_ratio',
        'entry_hurst',
        'entry_sma_distance_percent',
        # Time features
        'entry_hour',
        'entry_day_of_week',
        # Regime one-hot
        'regime_trending',
        'regime_choppy',
        'regime_weak_trending',
        # Momentum features
        'entry_rsi_slope',
        'entry_macd_hist',
        'entry_price_momentum',
        # Volatility context
        'entry_atr_expansion',
        'entry_bb_position',
        # Volume context  
        'entry_volume_trend',
        # Pattern detection
        'entry_candle_strength',
        # Session context
        'is_london_session',
        'is_nyc_session',
        'is_asia_session',
    ]
    
    def __init__(self, model_path: str = "./models/julaba_ml_v1.json"):
        self.model_path = Path(model_path)
        self.model = None
        self.is_loaded = False
        self.feature_columns = self.FEATURE_COLUMNS.copy()
        self.metrics = {}
        
        # Prediction settings
        self.min_confidence = 0.50  # Below this = LOW confidence
        self.high_confidence = 0.60  # Above this = HIGH confidence
        
        # Track predictions for validation
        self.prediction_log = []
        self.max_log_size = 1000
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and metadata."""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available. ML predictions disabled.")
            return
        
        if not self.model_path.exists():
            logger.info(f"ML model not found at {self.model_path}. Will use when available.")
            return
        
        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model(str(self.model_path))
            
            # Load metadata
            meta_path = str(self.model_path).replace('.json', '_meta.json')
            if Path(meta_path).exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    self.feature_columns = meta.get('feature_columns', self.FEATURE_COLUMNS)
                    self.metrics = meta.get('metrics', {})
            
            self.is_loaded = True
            logger.info(f"✅ ML model loaded from {self.model_path}")
            
            if self.metrics:
                logger.info(f"   Model accuracy: {self.metrics.get('accuracy', 0):.1%}")
                logger.info(f"   Trained on: {self.metrics.get('total_samples', 0)} samples")
        
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            self.model = None
            self.is_loaded = False
    
    def reload_model(self):
        """Reload model (call after retraining)."""
        self.is_loaded = False
        self._load_model()
    
    def prepare_features(self, signal: Dict[str, Any]) -> pd.DataFrame:
        """Convert signal dict to feature DataFrame."""
        
        # Map signal fields to feature names - core indicators
        features = {
            'entry_atr_percent': signal.get('atr_percent', signal.get('entry_atr_percent', 0)),
            'entry_rsi': signal.get('rsi', signal.get('entry_rsi', 50)),
            'entry_adx': signal.get('adx', signal.get('entry_adx', 0)),
            'entry_volume_ratio': signal.get('volume_ratio', signal.get('entry_volume_ratio', 1)),
            'entry_hurst': signal.get('hurst', signal.get('entry_hurst', 0.5)),
            'entry_sma_distance_percent': signal.get('sma_distance_percent', signal.get('entry_sma_distance_percent', 0)),
            'entry_hour': signal.get('hour', signal.get('entry_hour', 12)),
            'entry_day_of_week': signal.get('day_of_week', signal.get('entry_day_of_week', 0)),
        }
        
        # Regime one-hot encoding
        regime = signal.get('regime', signal.get('entry_regime', 'UNKNOWN'))
        features['regime_trending'] = 1 if regime == 'TRENDING' else 0
        features['regime_choppy'] = 1 if regime == 'CHOPPY' else 0
        features['regime_weak_trending'] = 1 if regime == 'WEAK_TRENDING' else 0
        
        # NEW: Momentum features
        features['entry_rsi_slope'] = signal.get('rsi_slope', signal.get('entry_rsi_slope', 0.0))
        features['entry_macd_hist'] = signal.get('macd_hist', signal.get('entry_macd_hist', 0.0))
        features['entry_price_momentum'] = signal.get('price_momentum', signal.get('entry_price_momentum', 0.0))
        
        # NEW: Volatility context
        features['entry_atr_expansion'] = signal.get('atr_expansion', signal.get('entry_atr_expansion', 1.0))
        features['entry_bb_position'] = signal.get('bb_position', signal.get('entry_bb_position', 0.5))
        
        # NEW: Volume context
        features['entry_volume_trend'] = signal.get('volume_trend', signal.get('entry_volume_trend', 0.0))
        
        # NEW: Pattern detection
        features['entry_candle_strength'] = signal.get('candle_strength', signal.get('entry_candle_strength', 0.0))
        
        # NEW: Session context
        hour = features['entry_hour']
        features['is_london_session'] = 1 if 7 <= hour <= 16 else signal.get('is_london_session', 0)
        features['is_nyc_session'] = 1 if 13 <= hour <= 22 else signal.get('is_nyc_session', 0)
        features['is_asia_session'] = 1 if 0 <= hour <= 9 else signal.get('is_asia_session', 0)
        
        # Create DataFrame
        df = pd.DataFrame([features])
        
        # Ensure all columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        return df[self.feature_columns]
    
    def predict(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get ML prediction for a signal.
        
        Args:
            signal: Dict with indicator values (rsi, adx, atr_percent, etc.)
        
        Returns:
            Dict with:
                - ml_win_probability: Float 0-1
                - ml_confidence: 'HIGH', 'MEDIUM', or 'LOW'
                - ml_recommendation: 'TAKE' or 'SKIP'
                - ml_available: Whether prediction was made
        """
        
        # Default response if model not available
        if not self.is_loaded or self.model is None:
            return {
                'ml_win_probability': 0.5,
                'ml_confidence': 'N/A',
                'ml_recommendation': 'N/A',
                'ml_available': False,
                'ml_reason': 'Model not loaded'
            }
        
        try:
            X = self.prepare_features(signal)
            prob = float(self.model.predict_proba(X)[0][1])
            
            # Determine confidence level
            if prob >= self.high_confidence:
                confidence = 'HIGH'
            elif prob >= self.min_confidence:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
            
            result = {
                'ml_win_probability': round(prob, 4),
                'ml_confidence': confidence,
                'ml_recommendation': 'TAKE' if prob >= self.min_confidence else 'SKIP',
                'ml_available': True
            }
            
            # Log prediction for later validation
            self._log_prediction(signal, result)
            
            return result
        
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return {
                'ml_win_probability': 0.5,
                'ml_confidence': 'ERROR',
                'ml_recommendation': 'N/A',
                'ml_available': False,
                'ml_reason': str(e)
            }
    
    def _log_prediction(self, signal: Dict, result: Dict):
        """Log prediction for validation tracking."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'signal_id': signal.get('signal_id', 'unknown'),
            'symbol': signal.get('symbol', 'unknown'),
            'ml_probability': result['ml_win_probability'],
            'ml_recommendation': result['ml_recommendation'],
            'actual_outcome': None  # To be filled later
        }
        
        self.prediction_log.append(log_entry)
        
        # Trim log if too large
        if len(self.prediction_log) > self.max_log_size:
            self.prediction_log = self.prediction_log[-self.max_log_size:]
    
    def record_outcome(self, signal_id: str, outcome: int):
        """Record actual outcome for a previous prediction."""
        for entry in reversed(self.prediction_log):
            if entry['signal_id'] == signal_id:
                entry['actual_outcome'] = outcome
                break
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get prediction accuracy statistics."""
        completed = [e for e in self.prediction_log if e['actual_outcome'] is not None]
        
        if not completed:
            return {
                'total_predictions': len(self.prediction_log),
                'validated': 0,
                'accuracy': None,
                'message': 'No validated predictions yet'
            }
        
        # Calculate accuracy
        correct = 0
        for e in completed:
            predicted_win = e['ml_probability'] >= self.min_confidence
            actual_win = e['actual_outcome'] == 1
            if predicted_win == actual_win:
                correct += 1
        
        accuracy = correct / len(completed)
        
        return {
            'total_predictions': len(self.prediction_log),
            'validated': len(completed),
            'correct': correct,
            'accuracy': round(accuracy, 4),
            'message': f"ML accuracy on real trades: {accuracy:.1%}"
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get predictor status."""
        return {
            'is_loaded': self.is_loaded,
            'model_path': str(self.model_path),
            'model_accuracy': self.metrics.get('accuracy'),
            'trained_samples': self.metrics.get('total_samples'),
            'predictions_made': len(self.prediction_log),
            'validation_stats': self.get_validation_stats()
        }


# Singleton instance
_predictor: Optional[MLPredictor] = None


def get_ml_predictor(model_path: str = "./models/julaba_ml_v1.json") -> MLPredictor:
    """Get the global ML predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = MLPredictor(model_path)
    return _predictor


def predict_signal(signal: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for single prediction."""
    return get_ml_predictor().predict(signal)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test prediction
    predictor = MLPredictor()
    
    if predictor.is_loaded:
        test_signal = {
            'atr_percent': 2.5,
            'rsi': 55,
            'adx': 28,
            'volume_ratio': 1.2,
            'hurst': 0.58,
            'sma_distance_percent': 0.5,
            'hour': 14,
            'day_of_week': 2,
            'regime': 'TRENDING'
        }
        
        result = predictor.predict(test_signal)
        print("\n📊 Test Prediction:")
        print(f"  Win Probability: {result['ml_win_probability']:.1%}")
        print(f"  Confidence: {result['ml_confidence']}")
        print(f"  Recommendation: {result['ml_recommendation']}")
    else:
        print("⚠️  Model not loaded. Train first with ml_trainer.py")
