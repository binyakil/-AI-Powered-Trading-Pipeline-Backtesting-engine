# Algorithmic Trading Pipeline

A modular framework for quantitative research, historical backtesting, strategy optimization, and automated live execution.

---

## 🎯 Architecture Plan & Roadmap

The objective of this repository is to unify previously fragmented trading scripts into a single, cohesive pipeline. The architecture enforces strict separation of concerns across three core pillars:
1. **Data & Simulation:** Historical data processing and strategy optimization via parameter sweeps.
2. **Strategy Derivation:** Translating backtest optimization results into concrete execution strategies.
3. **Live Deployment:** Production-ready execution bots incorporating machine learning and risk management.

* **Current Development Focus:** Transitioning core indicator logic into a dynamic, **pluggable architecture (`indicator_plugins`)** to allow runtime loading of decoupled indicators, filters, and regime classifiers.

---

## 📂 Repository Structure

```text
ai-trading-pipeline/
├── backtesting_engine/    # Historical simulation & parameter sweep core
├── deployments/
│   ├── triple_tp_bot/     # Strategy derived from backtest sweep results (MEXC/Binance)
│   └── julaba_bot/        # AI-enhanced live execution bot (Bybit, XGBoost, Telegram)
└── research/              # Quantitative analysis & mathematical modeling notebooks
