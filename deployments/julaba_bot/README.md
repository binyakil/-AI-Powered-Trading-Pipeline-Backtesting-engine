# Julaba Trading Bot

AI-powered cryptocurrency trading bot for Bybit USDT perpetual futures.

## Features

- **AI Filter** — Google Gemini analyzes every trade before execution
- **Dual ML Models** — XGBoost regime detection + live prediction
- **Bayesian Weight Learning** — Strategy weights evolve with every trade
- **Multi-Timeframe Analysis** — 5m, 15m, 1h, 4h confluence scoring
- **Smart Risk Management** — Dynamic position sizing, trailing stops, drawdown protection
- **Live Dashboard** — Real-time P&L, positions, and AI decision pipeline
- **Telegram Control** — Full bot control and notifications via Telegram
- **Multi-Pair Trading** — Scan and trade multiple pairs simultaneously

## Quick Start

```bash
git clone https://github.com/lillybaba1/jula.git
cd jula
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys (see SETUP_GUIDE.md)
python3 bot.py --dashboard --dashboard-port 5000
```

## Setup

See **[SETUP_GUIDE.md](SETUP_GUIDE.md)** for full instructions:

- Bybit API keys
- Google Gemini API key (free)
- Telegram bot setup
- Dashboard access
- VPS deployment

## Commands

See **[JULABA_COMMANDS.md](JULABA_COMMANDS.md)** for all bot and Telegram commands.

## Requirements

- Python 3.10+
- Bybit account with API access
- Google Gemini API key (free tier works)
