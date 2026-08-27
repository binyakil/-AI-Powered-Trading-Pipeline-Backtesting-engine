"""
ML Model Trainer for Julaba
Trains XGBoost classifier on backtest data to predict trade outcomes.
"""

import pandas as pd
import numpy as np
import logging
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger("Julaba.MLTrainer")

# Try to import ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed. Run: pip install xgboost")

try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed. Run: pip install scikit-learn")


class JulabaMLTrainer:
    """Train ML model for trade outcome prediction."""
    
    # Enhanced feature columns with momentum and context features
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
        # NEW: Momentum features
        'entry_rsi_slope',      # RSI momentum (rising/falling)
        'entry_macd_hist',       # MACD histogram value
        'entry_price_momentum',  # Price change % over 5 bars
        # NEW: Volatility context
        'entry_atr_expansion',   # Is ATR expanding? (vs 20-bar avg)
        'entry_bb_position',     # Position within Bollinger Bands (0-1)
        # NEW: Volume context  
        'entry_volume_trend',    # Volume trend (rising/falling)
        # NEW: Pattern detection
        'entry_candle_strength', # Bullish/bearish candle strength
        # NEW: Session context
        'is_london_session',     # London market hours
        'is_nyc_session',        # NYC market hours
        'is_asia_session',       # Asia market hours
    ]
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.model = None
        self.feature_columns = self.FEATURE_COLUMNS.copy()
        self.metrics = {}
        
        if XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                objective='binary:logistic',
                eval_metric='auc',
                use_label_encoder=False,
                random_state=42,
                # Regularization to prevent overfitting
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_weight=3
            )
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix with one-hot encoding and derived features."""
        df = df.copy()
        
        # One-hot encode regime
        df['regime_trending'] = (df['entry_regime'] == 'TRENDING').astype(int)
        df['regime_choppy'] = (df['entry_regime'] == 'CHOPPY').astype(int)
        df['regime_weak_trending'] = (df['entry_regime'] == 'WEAK_TRENDING').astype(int)
        
        # NEW: Fill missing momentum features with defaults
        momentum_defaults = {
            'entry_rsi_slope': 0.0,
            'entry_macd_hist': 0.0,
            'entry_price_momentum': 0.0,
            'entry_atr_expansion': 1.0,
            'entry_bb_position': 0.5,
            'entry_volume_trend': 0.0,
            'entry_candle_strength': 0.0,
            'is_london_session': 0,
            'is_nyc_session': 0,
            'is_asia_session': 0,
        }
        for col, default in momentum_defaults.items():
            if col not in df.columns:
                df[col] = default
        
        # Handle missing values
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
        
        return df
    
    def train(self, data_path: str) -> Dict[str, Any]:
        """Train the model and return metrics."""
        
        if not XGBOOST_AVAILABLE or not SKLEARN_AVAILABLE:
            return {'error': 'Required libraries not installed (xgboost, scikit-learn)'}
        
        print("="*60)
        print("🧠 JULABA ML TRAINER")
        print("="*60)
        
        # Load data
        print(f"\n📊 Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        print(f"  Loaded {len(df)} samples")
        
        # Prepare features
        print("🧮 Preparing features...")
        df = self.prepare_features(df)
        
        # Check for required columns
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            return {'error': f'Missing columns: {missing_cols}'}
        
        # Extract features and labels
        X = df[self.feature_columns].copy()
        y = df['outcome'].copy()
        weights = df['sample_weight'].copy() if 'sample_weight' in df.columns else pd.Series([1.0] * len(df))
        
        print(f"  Features shape: {X.shape}")
        print(f"  Target distribution: {y.value_counts().to_dict()}")
        
        # Split data
        print("\n📈 Splitting data (80/20)...")
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights, test_size=0.2, random_state=42, stratify=y
        )
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Cross-validation
        print("\n🔄 Running 5-fold cross-validation...")
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        
        # Train final model
        print("\n🚀 Training final model...")
        self.model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate
        print("\n📊 Evaluating model...")
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_prob)
        
        self.metrics = {
            'accuracy': float(accuracy),
            'auc_roc': float(auc_roc),
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'total_samples': int(len(df)),
            'train_samples': int(len(X_train)),
            'test_samples': int(len(X_test)),
            'win_rate_actual': float(y.mean()),
            'trained_at': datetime.now().isoformat(),
        }
        
        # Feature importance
        importance = dict(zip(self.feature_columns, 
                             [float(x) for x in self.model.feature_importances_]))
        self.metrics['feature_importance'] = importance
        
        # Print results
        print("\n" + "="*60)
        print("📊 TRAINING RESULTS")
        print("="*60)
        print(f"Total samples: {self.metrics['total_samples']}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"AUC-ROC: {auc_roc:.3f}")
        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        print(f"Actual win rate in data: {self.metrics['win_rate_actual']:.3f}")
        
        print("\n🎯 Feature Importance:")
        for feat, imp in sorted(importance.items(), key=lambda x: -x[1])[:5]:
            print(f"  {feat}: {imp:.3f}")
        
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Loss', 'Win']))
        
        # Quality check
        if accuracy < 0.52:
            print("\n⚠️  WARNING: Model accuracy is near random (50%). May not be useful.")
        elif accuracy < 0.55:
            print("\n⚠️  WARNING: Model accuracy is marginal. Use with caution.")
        else:
            print("\n✅ Model accuracy is acceptable for production use.")
        
        print("="*60)
        
        return self.metrics
    
    def save_model(self, filename: str = "julaba_ml_v1.json"):
        """Save trained model."""
        if self.model is None:
            return {'error': 'No model to save'}
        
        model_path = self.model_dir / filename
        
        # Save XGBoost model using booster (more reliable)
        try:
            # Try direct save first
            self.model.save_model(str(model_path))
        except TypeError:
            # Fallback: save the booster directly
            booster = self.model.get_booster()
            booster.save_model(str(model_path))
        
        # Save metadata
        meta_path = self.model_dir / filename.replace('.json', '_meta.json')
        meta = {
            'feature_columns': self.feature_columns,
            'metrics': self.metrics,
            'version': '1.0',
            'saved_at': datetime.now().isoformat()
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"\n💾 Model saved to {model_path}")
        print(f"💾 Metadata saved to {meta_path}")
        
        return {'model_path': str(model_path), 'meta_path': str(meta_path)}
    
    def load_model(self, filename: str = "julaba_ml_v1.json"):
        """Load trained model."""
        model_path = self.model_dir / filename
        meta_path = self.model_dir / filename.replace('.json', '_meta.json')
        
        if not model_path.exists():
            return {'error': f'Model not found: {model_path}'}
        
        if XGBOOST_AVAILABLE:
            # Load as booster first, then wrap in classifier
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(str(model_path))
            
            # Create classifier and set booster
            self.model = xgb.XGBClassifier()
            self.model._Booster = booster
        
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                self.feature_columns = meta.get('feature_columns', self.FEATURE_COLUMNS)
                self.metrics = meta.get('metrics', {})
        
        print(f"✅ Model loaded from {model_path}")
        return {'success': True}
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict win probability for a single signal."""
        if self.model is None:
            return {'error': 'Model not trained/loaded'}
        
        # Prepare features
        df = pd.DataFrame([features])
        df = self.prepare_features(df)
        
        X = df[self.feature_columns]
        
        prob = float(self.model.predict_proba(X)[0][1])
        
        return {
            'ml_win_probability': prob,
            'ml_confidence': 'HIGH' if prob > 0.6 else 'MEDIUM' if prob > 0.5 else 'LOW',
            'ml_recommendation': 'TAKE' if prob >= 0.50 else 'SKIP'
        }


def train_ml_model(data_path: str = "./historical_data/backtest_training_data.csv",
                   model_dir: str = "./models") -> Dict[str, Any]:
    """Convenience function to train and save ML model."""
    
    trainer = JulabaMLTrainer(model_dir)
    metrics = trainer.train(data_path)
    
    if 'error' not in metrics:
        trainer.save_model()
    
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Check dependencies
    if not XGBOOST_AVAILABLE:
        print("❌ XGBoost not installed. Run: pip install xgboost")
        exit(1)
    
    if not SKLEARN_AVAILABLE:
        print("❌ scikit-learn not installed. Run: pip install scikit-learn")
        exit(1)
    
    # Train model
    metrics = train_ml_model()
    
    if 'error' in metrics:
        print(f"❌ Training failed: {metrics['error']}")
    else:
        print(f"\n✅ Training complete! Accuracy: {metrics['accuracy']:.1%}")
