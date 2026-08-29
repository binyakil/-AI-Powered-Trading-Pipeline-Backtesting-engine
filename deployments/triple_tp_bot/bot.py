from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib.util
import logging
import os
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from ta.volatility import AverageTrueRange

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes
except ImportError:
    Application = None
    CommandHandler = None
    ContextTypes = None
    Update = None


try:
    import ccxt.pro as ccxtpro
except ImportError:
    try:
        import ccxtpro as ccxtpro
    except ImportError as exc:  # pragma: no cover - explicit dependency error
        raise ImportError("ccxt.pro is required for websocket support (pip install ccxtpro).") from exc


# =========================
# Strategy parameters (Run 165)
# =========================

STRATEGY_NAME = "TripleTP_Trail05R_40-30-30_MEXC"
SYMBOL = "LINK/USDT"
TIMEFRAME = "1m"

TP_LEVELS = (1.0, 2.0, 3.0)
TP_SPLITS = (0.40, 0.30, 0.30)
TRAIL_R = 0.5
ATR_LENGTH = 14
ATR_MULT = 2.0
RISK_PCT = 0.02
CONFLICT_MODE = "hedged"  # matches Run 165; hedged = always allow new entries
MIN_ENTRY_SECONDS = 180  # 3-minute debounce between consecutive entries (prevents rapid-fire signals)

MIN_BARS = 200

LOG = logging.getLogger(STRATEGY_NAME)

RUNS_DIR = Path("runs")


# =========================
# Data + indicator loading
# =========================

def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_indicator_module() -> Tuple[object, Path, str]:
    env_path = os.getenv("SWEEP_INDICATOR_PATH")
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "indicator.py")
        candidates.append(parent / "input indicators" / "python_designed" / "25_dec_13.py")
    for candidate in candidates:
        if not candidate or not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location("sweep_indicator_live", candidate)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "generate_signals"):
                indicator_hash = sha256_file(candidate)
                LOG.info("Loaded indicator from %s", candidate)
                return module, candidate, indicator_hash
    raise FileNotFoundError(
        "Unable to locate indicator module. Set SWEEP_INDICATOR_PATH or place indicator.py in the project root."
    )


INDICATOR, INDICATOR_PATH, INDICATOR_HASH = load_indicator_module()


def ohlcv_to_df(ohlcv: List[List[float]]) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=["Epoch", "Open", "High", "Low", "Close", "Volume"])
    df["Epoch"] = (df["Epoch"] // 1000).astype(int)
    return df


def append_bar(df: pd.DataFrame, bar: List[float]) -> pd.DataFrame:
    row = {
        "Epoch": int(bar[0] // 1000),
        "Open": float(bar[1]),
        "High": float(bar[2]),
        "Low": float(bar[3]),
        "Close": float(bar[4]),
        "Volume": float(bar[5]),
    }
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def aggregate_3m_bar(bars: List[List[float]]) -> List[float]:
    if len(bars) != 3:
        raise ValueError("Need exactly 3 bars to aggregate.")
    ts = bars[0][0]
    
    # Validate clock alignment: 3m bars should align to unix_timestamp % 180 == 0
    # This ensures same 3m boundaries as backtest/exchange standard (00:00, 00:03, 00:06...)
    ts_sec = int(ts // 1000)  # Convert ms to seconds
    if ts_sec % 180 != 0:
        LOG.warning(f"Bar not aligned to 3m boundary. Timestamp {ts_sec} (% 180 = {ts_sec % 180}). Expected alignment at :00, :03, :06...")
    
    open_px = float(bars[0][1])
    high_px = max(float(bar[2]) for bar in bars)
    low_px = min(float(bar[3]) for bar in bars)
    close_px = float(bars[-1][4])
    volume = sum(float(bar[5]) for bar in bars)
    return [ts, open_px, high_px, low_px, close_px, volume]


def downsample_1m_to_3m(bars: List[List[float]]) -> Tuple[List[List[float]], List[List[float]]]:
    aggregated = []
    buffer = []
    for bar in bars:
        buffer.append(bar)
        if len(buffer) == 3:
            aggregated.append(aggregate_3m_bar(buffer))
            buffer = []
    return aggregated, buffer


def write_ohlcv_csv(path: Path, rows: List[List[float]], append: bool = False) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, newline="") as handle:
        writer = csv.writer(handle)
        if not append:
            writer.writerow(["EpochMs", "Open", "High", "Low", "Close", "Volume"])
        writer.writerows(rows)


def compute_signal(df: pd.DataFrame) -> int:
    indicator_df = INDICATOR.generate_signals(df.copy())
    try:
        return int(indicator_df.iloc[-1]["Side"])
    except Exception:
        return 0


def compute_atr(df: pd.DataFrame) -> float:
    atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=ATR_LENGTH).average_true_range()
    return float(atr.iloc[-1]) if len(atr) else 0.0


def format_bar_time(epoch_ms: int) -> str:
    ts = int(epoch_ms // 1000)
    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


# =========================
# Execution state
# =========================


@dataclass
class TradeLeg:
    label: str
    r_multiple: float
    pct: float
    tp_price: float
    amount: float
    order_id: Optional[str] = None
    filled: float = 0.0
    closed: bool = False
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None


@dataclass
class TradeGroup:
    id: str
    side: int
    entry_price: float
    entry_time: datetime
    stop_price: float
    risk_pts: float
    amount: float
    legs: List[TradeLeg]
    pending_trail: bool = False

    def remaining_amount(self) -> float:
        return sum(max(leg.amount - leg.filled, 0.0) for leg in self.legs if not leg.closed)

    def is_closed(self) -> bool:
        return all(leg.closed for leg in self.legs)


@dataclass
class StrategyState:
    groups: List[TradeGroup] = field(default_factory=list)
    last_bar_ts: int = 0
    trade_seq: int = 0
    paper_balance: float = 0.0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    daily_pnl_date: Optional[date] = None
    last_price: Optional[float] = None
    last_entry_time: Optional[datetime] = None  # Track last entry for debounce mechanism


def compute_unrealized_pnl(group: TradeGroup, last_price: Optional[float]) -> Optional[float]:
    if last_price is None:
        return None
    remaining = group.remaining_amount()
    if remaining <= 0:
        return 0.0
    if group.side == 1:
        return (last_price - group.entry_price) * remaining
    return (group.entry_price - last_price) * remaining


\
class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, state: StrategyState, exchange, live: bool) -> None:
        self.token = token
        self.chat_id = chat_id
        self.state = state
        self.exchange = exchange
        self.live = live
        self.application = None

    @classmethod
    def from_env(cls, state: StrategyState, exchange, live: bool) -> Optional["TelegramNotifier"]:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return None
        if Application is None:
            LOG.warning("python-telegram-bot not installed; telegram disabled.")
            return None
        return cls(token, chat_id.strip(), state, exchange, live)

    async def start(self) -> None:
        self.application = Application.builder().token(self.token).build()
        self.application.add_handler(CommandHandler("positions", self._positions))
        self.application.add_handler(CommandHandler("pnl", self._pnl))
        self.application.add_handler(CommandHandler("status", self._status))
        await self.application.initialize()
        await self.application.start()
        if self.application.updater:
            await self.application.updater.start_polling()

    async def stop(self) -> None:
        if not self.application:
            return
        if self.application.updater:
            await self.application.updater.stop()
        await self.application.stop()
        await self.application.shutdown()

    def _is_authorized(self, update: Update) -> bool:
        if not update or not update.effective_chat:
            return False
        if self.chat_id.lstrip("-").isdigit():
            return str(update.effective_chat.id) == self.chat_id
        if update.effective_chat.username:
            return f"@{update.effective_chat.username}" == self.chat_id
        return True

    async def send(self, message: str) -> None:
        if not self.application:
            return
        try:
            await self.application.bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as exc:
            LOG.warning("Telegram send failed: %s", exc)

    def _format_group_line(self, group: TradeGroup) -> str:
        side = "LONG" if group.side == 1 else "SHORT"
        entry_time = group.entry_time.strftime("%Y-%m-%d %H:%M:%S UTC")
        tp_prices = {leg.label: leg.tp_price for leg in group.legs}
        upnl = compute_unrealized_pnl(group, self.state.last_price)
        upnl_text = "n/a" if upnl is None else f"{upnl:.4f}"
        return (
            f"{group.id} {side} entry={group.entry_price:.5f} entry_time={entry_time} "
            f"stop={group.stop_price:.5f} TP1={tp_prices.get('TP1', 0):.5f} "
            f"TP2={tp_prices.get('TP2', 0):.5f} TP3={tp_prices.get('TP3', 0):.5f} uPNL={upnl_text}"
        )

    async def _positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        message = update.effective_message
        if not message:
            return
        if not self.state.groups:
            await message.reply_text("No open positions.")
            return
        lines = ["Open positions:"]
        for group in self.state.groups:
            lines.append(self._format_group_line(group))
        await message.reply_text("\n".join(lines))

    async def _pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        message = update.effective_message
        if not message:
            return
        trade_count = self.state.trade_count
        win_rate = (self.state.win_count / trade_count * 100.0) if trade_count else 0.0
        text = (
            f"Total PnL: {self.state.total_pnl:.4f} USDT\n"
            f"Daily PnL: {self.state.daily_pnl:.4f} USDT\n"
            f"Trades: {trade_count} (wins: {self.state.win_count}, losses: {self.state.loss_count}, "
            f"win rate: {win_rate:.1f}%)"
        )
        await message.reply_text(text)

    async def _status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._is_authorized(update):
            return
        message = update.effective_message
        if not message:
            return
        mode = "LIVE" if self.live else "PAPER"
        last_price = self.state.last_price
        last_text = f"{last_price:.5f}" if last_price is not None else "n/a"
        if self.live:
            balance_text = "n/a"
            try:
                balance = await get_free_balance(self.exchange, "USDT")
                balance_text = f"{balance:.4f} USDT"
            except Exception as exc:
                LOG.warning("Telegram status balance failed: %s", exc)
        else:
            balance_text = f"{self.state.paper_balance:.4f} USDT"
        text = (
            f"Status: running ({mode})\n"
            f"Symbol: {SYMBOL}\n"
            f"Last price: {last_text}\n"
            f"Open positions: {len(self.state.groups)}\n"
            f"Balance: {balance_text}"
        )
        await message.reply_text(text)

    async def notify_trade_open(self, group: TradeGroup) -> None:
        side = "LONG" if group.side == 1 else "SHORT"
        tp_prices = {leg.label: leg.tp_price for leg in group.legs}
        message = (
            f"New trade {group.id} {side} entry={group.entry_price:.5f} stop={group.stop_price:.5f} "
            f"TP1={tp_prices.get('TP1', 0):.5f} TP2={tp_prices.get('TP2', 0):.5f} "
            f"TP3={tp_prices.get('TP3', 0):.5f} size={group.amount:.6f}"
        )
        await self.send(message)

    async def notify_tp_hit(self, group: TradeGroup, leg: TradeLeg) -> None:
        price = leg.exit_price if leg.exit_price is not None else leg.tp_price
        if group.side == 1:
            pnl = (price - group.entry_price) * leg.amount
        else:
            pnl = (group.entry_price - price) * leg.amount
        message = f"{group.id} {leg.label} hit at {price:.5f} PnL={pnl:.4f}"
        await self.send(message)

    async def notify_sl(self, group: TradeGroup, price: float) -> None:
        message = f"{group.id} SL hit at {price:.5f}"
        await self.send(message)

    async def notify_trade_close(self, group: TradeGroup, pnl: float) -> None:
        message = f"Trade {group.id} closed PnL={pnl:.4f}"
        await self.send(message)

    async def notify_daily_pnl(self, day: date, pnl: float) -> None:
        message = f"Daily PnL {day.isoformat()}: {pnl:.4f}"
        await self.send(message)


# =========================
# Exchange helpers
# =========================


def build_exchange(live: bool):
    load_dotenv()
    params = {"enableRateLimit": True, "timeout": 30000}
    if live:
        api_key = os.getenv("API_KEY")
        api_secret = os.getenv("API_SECRET")
        if not api_key or not api_secret:
            raise ValueError("API_KEY/API_SECRET required for --live mode (load via .env).")
        params.update({"apiKey": api_key, "secret": api_secret})
    exchange = ccxtpro.mexc(params)
    exchange.options["defaultType"] = "spot"
    return exchange


def to_price_precision(exchange, symbol: str, value: float) -> float:
    try:
        return float(exchange.price_to_precision(symbol, value))
    except Exception:
        return float(value)


def to_amount_precision(exchange, symbol: str, value: float) -> float:
    try:
        return float(exchange.amount_to_precision(symbol, value))
    except Exception:
        return float(value)


def split_amounts(exchange, symbol: str, total: float) -> List[float]:
    amounts = []
    remaining = total
    for pct in TP_SPLITS[:-1]:
        amt = to_amount_precision(exchange, symbol, total * pct)
        amounts.append(amt)
        remaining -= amt
    last = to_amount_precision(exchange, symbol, max(remaining, 0.0))
    amounts.append(last)
    return amounts


async def get_entry_price(exchange, symbol: str, side: int) -> float:
    book = await exchange.fetch_order_book(symbol)
    best_bid = book["bids"][0][0] if book.get("bids") else None
    best_ask = book["asks"][0][0] if book.get("asks") else None
    if best_bid and best_ask:
        mid = (best_bid + best_ask) / 2.0
    else:
        mid = None

    if side == 1:
        if best_ask:
            return float(best_ask)
        if mid:
            return float(mid)
    else:
        if best_bid:
            return float(best_bid)
        if mid:
            return float(mid)

    ticker = await exchange.fetch_ticker(symbol)
    return float(ticker.get("last") or ticker.get("close"))


async def get_free_balance(exchange, currency: str) -> float:
    balance = await exchange.fetch_balance()
    entry = balance.get(currency, {})
    if isinstance(entry, dict):
        return float(entry.get("free") or 0.0)
    return float(balance.get("free", {}).get(currency) or 0.0)


# =========================
# Dry-run simulation
# =========================


async def simulate_bar_for_group(group: TradeGroup, high: float, low: float, notifier: Optional["TelegramNotifier"] = None) -> None:
    sl_triggered = False
    for leg in group.legs:
        if leg.closed:
            continue
        if group.side == 1:
            if low <= group.stop_price:
                exit_price = group.stop_price
                exit_reason = "SL"
            elif high >= leg.tp_price:
                exit_price = leg.tp_price
                exit_reason = "TP"
            else:
                continue
        else:
            if high >= group.stop_price:
                exit_price = group.stop_price
                exit_reason = "SL"
            elif low <= leg.tp_price:
                exit_price = leg.tp_price
                exit_reason = "TP"
            else:
                continue

        leg.closed = True
        leg.filled = leg.amount
        leg.exit_price = exit_price
        leg.exit_reason = exit_reason
        if leg.label == "TP1":
            if exit_reason == "TP":
                LOG.info("TP1_FILLED group=%s price=%.5f", group.id, exit_price)
                LOG.info("TRAIL_ARMED reason=TP1_FILLED group=%s", group.id)
            else:
                LOG.info("TRAIL_ARMED reason=TP1_CLOSED_%s group=%s", exit_reason, group.id)
            group.pending_trail = True

        if exit_reason == "TP" and notifier:
            await notifier.notify_tp_hit(group, leg)
        if exit_reason == "SL":
            sl_triggered = True

    if sl_triggered and notifier:
        await notifier.notify_sl(group, group.stop_price)


def apply_trailing(group: TradeGroup) -> bool:
    if not group.pending_trail:
        return False
    if any(not leg.closed for leg in group.legs):
        group.stop_price = group.entry_price + group.side * TRAIL_R * group.risk_pts
        group.pending_trail = False
        LOG.info("TRAIL_APPLIED group=%s stop=%.5f", group.id, group.stop_price)
        return True
    group.pending_trail = False
    return False


def compute_group_pnl(group: TradeGroup) -> float:
    pnl = 0.0
    for leg in group.legs:
        if leg.exit_price is None:
            continue
        if group.side == 1:
            pnl += (leg.exit_price - group.entry_price) * leg.amount
        else:
            pnl += (group.entry_price - leg.exit_price) * leg.amount
    return pnl


# =========================
# Live order management
# =========================


async def sync_group_orders(exchange, group: TradeGroup, notifier: Optional["TelegramNotifier"] = None) -> None:
    for leg in group.legs:
        if leg.closed or not leg.order_id:
            continue
        try:
            order = await exchange.fetch_order(leg.order_id, SYMBOL)
        except Exception as exc:
            LOG.warning("Failed to fetch order %s: %s", leg.order_id, exc)
            continue

        status = order.get("status")
        filled = float(order.get("filled") or 0.0)
        leg.filled = filled
        if status == "closed":
            leg.closed = True
            leg.exit_price = float(order.get("average") or order.get("price") or leg.tp_price)
            leg.exit_reason = "TP"
            if leg.label == "TP1":
                LOG.info("TP1_FILLED group=%s price=%.5f order_id=%s", group.id, leg.exit_price, leg.order_id)
                LOG.info("TRAIL_ARMED reason=TP1_FILLED group=%s", group.id)
                group.pending_trail = True
            if notifier:
                await notifier.notify_tp_hit(group, leg)
        elif status == "canceled":
            leg.order_id = None
            LOG.warning("TP order %s canceled; leg still open without TP.")


async def cancel_group_orders(exchange, group: TradeGroup) -> None:
    for leg in group.legs:
        if leg.closed or not leg.order_id:
            continue
        try:
            await exchange.cancel_order(leg.order_id, SYMBOL)
        except Exception as exc:
            LOG.warning("Failed to cancel order %s: %s", leg.order_id, exc)
        finally:
            leg.order_id = None


async def close_group_market(exchange, group: TradeGroup, reason: str, notifier: Optional["TelegramNotifier"] = None) -> None:
    remaining = group.remaining_amount()
    if remaining <= 0:
        return
    await cancel_group_orders(exchange, group)

    side = "sell" if group.side == 1 else "buy"
    try:
        order = await exchange.create_order(SYMBOL, "market", side, remaining)
        fill_price = float(order.get("average") or order.get("price") or group.stop_price)
    except Exception as exc:
        LOG.error("Failed to close position at market: %s", exc)
        fill_price = group.stop_price

    for leg in group.legs:
        if leg.closed:
            continue
        leg.closed = True
        leg.filled = leg.amount
        leg.exit_price = fill_price
        leg.exit_reason = reason

    if notifier and reason == "SL":
        await notifier.notify_sl(group, fill_price)


async def handle_stop_triggers(exchange, group: TradeGroup, high: float, low: float, notifier: Optional["TelegramNotifier"] = None) -> bool:
    if group.side == 1 and low <= group.stop_price:
        await close_group_market(exchange, group, "SL", notifier)
        return True
    if group.side == -1 and high >= group.stop_price:
        await close_group_market(exchange, group, "SL", notifier)
        return True
    return False


# =========================
# Entry logic
# =========================


async def open_new_trade(exchange, state: StrategyState, df: pd.DataFrame, side: int, live: bool, notifier: Optional["TelegramNotifier"] = None) -> None:
    atr_value = compute_atr(df)
    if atr_value <= 0:
        LOG.debug("ATR not ready; skipping signal.")
        return

    entry_price = await get_entry_price(exchange, SYMBOL, side)
    entry_time = datetime.utcnow()
    stop_price = entry_price - side * ATR_MULT * atr_value
    risk_pts = abs(entry_price - stop_price)
    if risk_pts <= 0:
        LOG.warning("Invalid risk distance; skipping trade.")
        return

    if live:
        quote_free = await get_free_balance(exchange, "USDT")
        risk_usdt = quote_free * RISK_PCT
    else:
        risk_usdt = state.paper_balance * RISK_PCT

    total_amount = risk_usdt / risk_pts
    total_amount = to_amount_precision(exchange, SYMBOL, total_amount)

    market = exchange.market(SYMBOL)
    min_amount = float(market.get("limits", {}).get("amount", {}).get("min") or 0.0)
    if total_amount <= 0 or (min_amount and total_amount < min_amount):
        LOG.info("Position size %.6f below minimum; skipping trade.", total_amount)
        return

    if live:
        if side == 1:
            max_amount = quote_free / entry_price if entry_price else 0.0
            if total_amount > max_amount:
                total_amount = to_amount_precision(exchange, SYMBOL, max_amount)
        else:
            base_free = await get_free_balance(exchange, "LINK")
            if total_amount > base_free:
                LOG.info("Short signal ignored on spot (insufficient LINK balance).")
                return

    split_amounts_list = split_amounts(exchange, SYMBOL, total_amount)
    if min_amount:
        if any(amt < min_amount for amt in split_amounts_list):
            LOG.info("One or more TP legs below min amount; skipping trade.")
            return

    tp_prices = [entry_price + side * risk_pts * r for r in TP_LEVELS]

    state.trade_seq += 1
    group_id = f"T{state.trade_seq}"
    legs = [
        TradeLeg(label="TP1", r_multiple=TP_LEVELS[0], pct=TP_SPLITS[0], tp_price=tp_prices[0], amount=split_amounts_list[0]),
        TradeLeg(label="TP2", r_multiple=TP_LEVELS[1], pct=TP_SPLITS[1], tp_price=tp_prices[1], amount=split_amounts_list[1]),
        TradeLeg(label="TP3", r_multiple=TP_LEVELS[2], pct=TP_SPLITS[2], tp_price=tp_prices[2], amount=split_amounts_list[2]),
    ]
    group = TradeGroup(
        id=group_id,
        side=side,
        entry_price=entry_price,
        entry_time=entry_time,
        stop_price=stop_price,
        risk_pts=risk_pts,
        amount=total_amount,
        legs=legs,
    )

    if not live:
        state.groups.append(group)
        LOG.info(
            "DRY-RUN entry %s side=%s entry=%.5f stop=%.5f size=%.6f",
            group.id,
            "LONG" if side == 1 else "SHORT",
            entry_price,
            stop_price,
            total_amount,
        )
        if notifier:
            await notifier.notify_trade_open(group)
        return

    # Live entry: market order, then place limit TPs.
    side_str = "buy" if side == 1 else "sell"
    try:
        if side_str == "buy" and exchange.options.get("createMarketBuyOrderRequiresPrice", False):
            entry_order = await exchange.create_order(SYMBOL, "market", side_str, total_amount, entry_price)
        else:
            entry_order = await exchange.create_order(SYMBOL, "market", side_str, total_amount)
        fill_price = float(entry_order.get("average") or entry_price)
        group.entry_price = fill_price
        group.stop_price = fill_price - side * ATR_MULT * atr_value
        group.risk_pts = abs(fill_price - group.stop_price)
        group_tp_prices = [fill_price + side * group.risk_pts * r for r in TP_LEVELS]
        for leg, price in zip(group.legs, group_tp_prices):
            leg.tp_price = price
        LOG.info(
            "LIVE entry %s side=%s entry=%.5f stop=%.5f size=%.6f",
            group.id,
            "LONG" if side == 1 else "SHORT",
            fill_price,
            group.stop_price,
            total_amount,
        )
    except Exception as exc:
        LOG.error("Entry order failed: %s", exc)
        return

    tp_side = "sell" if side == 1 else "buy"
    for leg in group.legs:
        price = to_price_precision(exchange, SYMBOL, leg.tp_price)
        try:
            order = await exchange.create_order(SYMBOL, "limit", tp_side, leg.amount, price)
            leg.order_id = order.get("id")
            if leg.label == "TP1":
                LOG.info("TP1_PLACED price=%.5f qty=%.6f order_id=%s", price, leg.amount, leg.order_id)
            LOG.info("Placed %s limit at %.5f for %.6f", leg.label, price, leg.amount)
        except Exception as exc:
            LOG.error("Failed to place %s order: %s", leg.label, exc)

    state.groups.append(group)
    if notifier:
        await notifier.notify_trade_open(group)


# =========================
# Main loop
# =========================


async def run_strategy(live: bool, paper_balance: float, log_level: str, ws_verbose: bool) -> None:
    handlers = [logging.StreamHandler(), logging.FileHandler("bot.log")]
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

    RUNS_DIR.mkdir(exist_ok=True)
    run_id = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = RUNS_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    LOG.info("RUN_DIR=%s", run_dir.resolve())
    LOG.info("INDICATOR_PATH_USED=%s", INDICATOR_PATH)
    LOG.info("INDICATOR_SHA256=%s", INDICATOR_HASH)
    try:
        indicator_copy = run_dir / "indicator.py"
        shutil.copy2(INDICATOR_PATH, indicator_copy)
        LOG.info("Indicator copied to %s", indicator_copy)
    except Exception as exc:
        LOG.warning("Indicator copy failed: %s", exc)

    exchange = build_exchange(live)
    exchange.verbose = ws_verbose
    await exchange.load_markets()

    notifier: Optional[TelegramNotifier] = None

    try:
        ohlcv_1m = await exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=MIN_BARS * 3 + 50)
        live_1m_path = run_dir / "live_1m.csv"
        seed_bars = ohlcv_1m[:-1] if len(ohlcv_1m) > 1 else []
        write_ohlcv_csv(live_1m_path, seed_bars, append=False)
        if seed_bars:
            LOG.info("LIVE_1M_PATH=%s (closed bars only, seed=%s)", live_1m_path.resolve(), len(seed_bars))
        ohlcv_3m, one_min_buffer = downsample_1m_to_3m(ohlcv_1m)
        df = ohlcv_to_df(ohlcv_3m)
        state = StrategyState(paper_balance=paper_balance)
        if len(ohlcv_1m) >= 2:
            state.last_bar_ts = ohlcv_1m[-2][0]
        if not df.empty:
            state.last_price = float(df.iloc[-1]["Close"])

        notifier = TelegramNotifier.from_env(state, exchange, live)
        if notifier:
            try:
                await notifier.start()
                await notifier.send(f"Starting {STRATEGY_NAME} ({'LIVE' if live else 'DRY-RUN'})")
            except Exception as exc:
                LOG.warning("Telegram disabled: %s", exc)
                notifier = None

        LOG.info("Starting %s (%s)", STRATEGY_NAME, "LIVE" if live else "DRY-RUN")

        async def handle_agg_bar(agg: List[float]) -> None:
            nonlocal df, state
            df = append_bar(df, agg)
            df = df.tail(MIN_BARS + 50).reset_index(drop=True)

            high = float(agg[2])
            low = float(agg[3])
            close = float(agg[4])
            state.last_price = close
            LOG.debug("Closed 3m bar O=%.5f H=%.5f L=%.5f C=%.5f", agg[1], high, low, close)
            bar_time = format_bar_time(agg[0])
            bar_date = datetime.utcfromtimestamp(agg[0] // 1000).date()

            if state.daily_pnl_date is None:
                state.daily_pnl_date = bar_date
            elif bar_date != state.daily_pnl_date:
                if notifier:
                    await notifier.notify_daily_pnl(state.daily_pnl_date, state.daily_pnl)
                state.daily_pnl = 0.0
                state.daily_pnl_date = bar_date

            # Update open groups based on the closed bar.
            if live:
                for group in state.groups:
                    await sync_group_orders(exchange, group, notifier)
                    if not group.is_closed():
                        await handle_stop_triggers(exchange, group, high, low, notifier)
            else:
                for group in state.groups:
                    if not group.is_closed():
                        await simulate_bar_for_group(group, high, low, notifier)

            # Apply trailing stops at end of bar (matches app.py logic).
            for group in state.groups:
                if apply_trailing(group):
                    LOG.info("Trail updated %s stop=%.5f", group.id, group.stop_price)

            # Clean up closed groups and update paper balance.
            still_open = []
            for group in state.groups:
                if group.is_closed():
                    pnl = compute_group_pnl(group)
                    state.total_pnl += pnl
                    state.daily_pnl += pnl
                    state.trade_count += 1
                    if pnl > 0:
                        state.win_count += 1
                    elif pnl < 0:
                        state.loss_count += 1
                    if not live:
                        state.paper_balance += pnl
                    LOG.info("Trade %s closed PnL=%.4f", group.id, pnl)
                    if notifier:
                        await notifier.notify_trade_close(group, pnl)
                else:
                    still_open.append(group)
            state.groups = still_open

            if len(df) < MIN_BARS:
                return

            side = compute_signal(df)
            LOG.debug("Signal side=%s", side)
            if side == 1:
                LOG.info("Signal LONG at %s close=%.5f", bar_time, close)
            elif side == -1:
                LOG.info("Signal SHORT at %s close=%.5f", bar_time, close)
            if side in (1, -1):
                if CONFLICT_MODE == "hedged":
                    # Check debounce: prevent rapid-fire entries within MIN_ENTRY_SECONDS window
                    now = datetime.utcnow()
                    if state.last_entry_time:
                        elapsed = (now - state.last_entry_time).total_seconds()
                        if elapsed < MIN_ENTRY_SECONDS:
                            LOG.debug(
                                "Entry debounce active. Last entry %.0f seconds ago (threshold: %d sec). Skipping signal.",
                                elapsed,
                                MIN_ENTRY_SECONDS
                            )
                            pass  # Skip this signal, still in cooldown
                        else:
                            await open_new_trade(exchange, state, df, side, live, notifier)
                            state.last_entry_time = now
                    else:
                        await open_new_trade(exchange, state, df, side, live, notifier)
                        state.last_entry_time = now

        while True:
            import time
            current_second = time.time() % 60
            # Schläft exakt bis zur 2. Sekunde der NEUEN Minute, um fertige Kerzen zu garantieren
            sleep_time = 62 - current_second if current_second >= 2 else 2 - current_second
            await asyncio.sleep(sleep_time)
            try:
                ohlcv_1m = await exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=MIN_BARS + 50)
            except Exception as exc:
                LOG.warning("Polling error (%s): %s", type(exc).__name__, exc)
                LOG.debug("Polling error details", exc_info=True)
                await asyncio.sleep(5)
                continue

            if len(ohlcv_1m) < 2:
                continue

            closed_bars = ohlcv_1m[:-1]
            new_bars = [bar for bar in closed_bars if bar[0] > state.last_bar_ts]
            if not new_bars:
                continue
            write_ohlcv_csv(live_1m_path, new_bars, append=True)

            for bar in new_bars:
                state.last_bar_ts = bar[0]
                one_min_buffer.append(bar)
                while len(one_min_buffer) >= 3:
                    agg = aggregate_3m_bar(one_min_buffer[:3])
                    one_min_buffer = one_min_buffer[3:]
                    await handle_agg_bar(agg)
    finally:
        if notifier:
            await notifier.stop()
        try:
            await exchange.close()
        except Exception as exc:
            LOG.debug("Exchange close error: %s", exc)


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Run the TripleTP_Trail05R_40-30-30 strategy on MEXC spot.")
    parser.add_argument("--live", action="store_true", help="Enable live trading (requires API keys in .env).")
    parser.add_argument("--paper-balance", type=float, default=1000.0, help="Paper balance for dry-run sizing (USDT).")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING).")
    parser.add_argument("--ws-verbose", action="store_true", help="Enable CCXT debug logging (noisy).")
    args = parser.parse_args()

    try:
        await run_strategy(args.live, args.paper_balance, args.log_level, args.ws_verbose)
    except KeyboardInterrupt:
        LOG.info("Shutdown requested.")


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass
