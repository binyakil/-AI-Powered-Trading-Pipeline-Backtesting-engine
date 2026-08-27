"""
Julaba ML Pipeline - Master Script
Runs the complete ML training pipeline:
1. Fetch 90 days of historical data from MEXC
2. Calculate indicators and generate signals
3. Train XGBoost classifier
4. Save model for live bot integration
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Julaba.Pipeline")


def run_full_pipeline():
    """Run the complete ML training pipeline."""
    
    print("\n" + "="*70)
    print("🚀 JULABA ML PIPELINE - FULL EXECUTION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Step 1: Fetch historical data
    print("\n" + "="*70)
    print("📡 STEP 1/4: FETCHING HISTORICAL DATA FROM MEXC")
    print("="*70)
    
    from data_fetcher import MEXCDataFetcher, fetch_btc_data
    
    fetcher = MEXCDataFetcher()
    data = fetcher.run()
    
    # Also fetch BTC for correlation analysis
    print("\n📊 Fetching BTC for correlation analysis...")
    btc_data = fetch_btc_data()
    print(f"  ✅ Got {len(btc_data)} BTC candles")
    
    total_candles = sum(len(df) for df in data.values()) + len(btc_data)
    print(f"\n✅ Step 1 Complete: {total_candles} total candles fetched")
    
    # Step 2: Generate training signals
    print("\n" + "="*70)
    print("🎯 STEP 2/4: GENERATING TRAINING SIGNALS")
    print("="*70)
    
    from backtest_generator import run_backtest_pipeline
    
    signals = run_backtest_pipeline()
    
    if not signals:
        print("❌ No signals generated! Check data quality.")
        return {'error': 'No signals generated'}
    
    print(f"\n✅ Step 2 Complete: {len(signals)} signals generated")
    
    # Step 3: Train ML model
    print("\n" + "="*70)
    print("🧠 STEP 3/4: TRAINING ML MODEL")
    print("="*70)
    
    from ml_trainer import train_ml_model
    
    metrics = train_ml_model()
    
    if 'error' in metrics:
        print(f"❌ Training failed: {metrics['error']}")
        return metrics
    
    print(f"\n✅ Step 3 Complete: Model trained with {metrics['accuracy']:.1%} accuracy")
    
    # Step 4: Verify model loads correctly
    print("\n" + "="*70)
    print("✅ STEP 4/4: VERIFYING MODEL INTEGRATION")
    print("="*70)
    
    from ml_predictor import get_ml_predictor
    
    predictor = get_ml_predictor()
    
    if not predictor.is_loaded:
        print("❌ Model failed to load!")
        return {'error': 'Model failed to load'}
    
    # Test prediction
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
    
    print(f"  Test prediction: {result['ml_win_probability']:.1%} win probability")
    print(f"  Recommendation: {result['ml_recommendation']}")
    
    print("\n✅ Step 4 Complete: Model verified and ready!")
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETE!")
    print("="*70)
    print(f"📊 Data: {total_candles} candles from 90 days")
    print(f"🎯 Signals: {len(signals)} training samples")
    print(f"🧠 Model: {metrics['accuracy']:.1%} accuracy, {metrics['auc_roc']:.3f} AUC-ROC")
    print(f"📁 Model saved to: ./models/julaba_ml_v1.json")
    print("\n🚀 Ready to integrate with live bot!")
    print("   ML influence is currently set to 0.0 (passive learning)")
    print("   Increase ml_config.influence_weight after validation")
    print("="*70)
    
    return {
        'success': True,
        'candles_fetched': total_candles,
        'signals_generated': len(signals),
        'model_accuracy': metrics['accuracy'],
        'model_auc': metrics['auc_roc']
    }


if __name__ == "__main__":
    try:
        result = run_full_pipeline()
        
        if result.get('success'):
            print("\n✅ Pipeline completed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ Pipeline failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
