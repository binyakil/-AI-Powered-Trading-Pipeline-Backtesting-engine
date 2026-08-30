# TripleTP MEXC Bot

## Setup
1. Create a virtual environment.
2. Install dependencies:
   pip install -r requirements.txt
3. Copy the environment template:
   cp .env.example .env
4. Edit .env with your API keys (leave blank for paper trading).

## Run (paper)
python bot.py --paper-balance 1000

## Run (live)
python bot.py --live

## Telegram (optional)
Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env.
Commands:
- /positions (open positions and unrealized PnL)
- /pnl (total/daily PnL, win rate, trade count)
- /status (mode, balance, open positions)

Notifications are one-way (new trades, TP hits, SL, closes, daily PnL).

## Logs
bot.log is written in the project directory.
