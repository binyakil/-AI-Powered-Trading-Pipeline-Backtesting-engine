from __future__ import annotations

import argparse
import importlib.util
import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv
from ta.volatility import AverageTrueRange

try:
    import ccxt.pro as ccxtpro
except ImportError as exc:  # pragma: no cover - explicit dependency error
    raise ImportError("ccxt.pro is required for websocket support (pip install ccxtpro).") from exc


# =========================
# Strategy parameters (Run 165)
# =========================

STRATEGY_NAME = "TripleTP_Trail05R_40-30-30_Binance"
SYMBOL = "LINK/USDT"
TIMEFRAME = "3m"

TP_LEVELS = (1.0, 2.0, 3.0)
TP_SPLITS = (0.40, 0.30, 0.30)
TRAIL_R = 0.5
ATR_LENGTH = 14
ATR_MULT = 2.0
RISK_PCT = 0.02
CONFLICT_MODE = "hedged"  # matches Run 165; hedged = always allow new entries

MIN_BARS = 200
WS_TIMEOUT = 30.0

LOG = logging.getLogger(STRATEGY_NAME)


# =========================
# Data + indicator loading
# =========================

def load_indicator_module():
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
                LOG.info("Loaded indicator from %s", candidate)
                return module
    raise FileNotFoundError(
        "Unable to locate indicator module. Set SWEEP_INDICATOR_PATH or place indicator.py in the project root."
    )


INDICATOR = load_indicator_module()


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


def compute_signal(df: pd.DataFrame) -> int:
    indicator_df = INDICATOR.generate_signals(df.copy())
    try:
        return int(indicator_df.iloc[-1]["Side"])
    except Exception:
        return 0


def compute_atr(df: pd.DataFrame) -> float:
    atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=ATR_LENGTH).average_true_range()
    return float(atr.iloc[-1]) if len(atr) else 0.0


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


# =========================
# Exchange helpers
# =========================


def build_exchange(live: bool):
    load_dotenv()
    params = {"enableRateLimit": True}
    if live:
        api_key = os.getenv("API_KEY")
        api_secret = os.getenv("API_SECRET")
        if not api_key or not api_secret:
            raise ValueError("API_KEY/API_SECRET required for --live mode (load via .env).")
        params.update({"apiKey": api_key, "secret": api_secret})
    exchange = ccxtpro.binance(params)
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


def simulate_bar_for_group(group: TradeGroup, high: float, low: float) -> None:
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
            group.pending_trail = True


def apply_trailing(group: TradeGroup) -> bool:
    if not group.pending_trail:
        return False
    if any(not leg.closed for leg in group.legs):
        group.stop_price = group.entry_price + group.side * TRAIL_R * group.risk_pts
        group.pending_trail = False
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


async def sync_group_orders(exchange, group: TradeGroup) -> None:
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
                group.pending_trail = True
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


async def close_group_market(exchange, group: TradeGroup, reason: str) -> None:
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


async def handle_stop_triggers(exchange, group: TradeGroup, high: float, low: float) -> bool:
    if group.side == 1 and low <= group.stop_price:
        await close_group_market(exchange, group, "SL")
        return True
    if group.side == -1 and high >= group.stop_price:
        await close_group_market(exchange, group, "SL")
        return True
    return False


# =========================
# Entry logic
# =========================


async def open_new_trade(exchange, state: StrategyState, df: pd.DataFrame, side: int, live: bool) -> None:
    atr_value = compute_atr(df)
    if atr_value <= 0:
        LOG.debug("ATR not ready; skipping signal.")
        return

    entry_price = await get_entry_price(exchange, SYMBOL, side)
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
            LOG.info("Placed %s limit at %.5f for %.6f", leg.label, price, leg.amount)
        except Exception as exc:
            LOG.error("Failed to place %s order: %s", leg.label, exc)

    state.groups.append(group)


# =========================
# Main loop
# =========================


async def run_strategy(live: bool, paper_balance: float, log_level: str, ws_verbose: bool) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    exchange = build_exchange(live)
    exchange.verbose = ws_verbose
    await exchange.load_markets()

    ohlcv = await exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=MIN_BARS + 50)
    df = ohlcv_to_df(ohlcv)
    state = StrategyState(paper_balance=paper_balance)

    LOG.info("Starting %s (%s)", STRATEGY_NAME, "LIVE" if live else "DRY-RUN")

    while True:
        try:
            candles = await asyncio.wait_for(
                exchange.watch_ohlcv(SYMBOL, TIMEFRAME), timeout=WS_TIMEOUT
            )
        except asyncio.TimeoutError:
            LOG.warning("No websocket data for %.0fs; still waiting...", WS_TIMEOUT)
            continue
        except Exception as exc:
            LOG.warning("Websocket error: %s", exc)
            await asyncio.sleep(1)
            continue

        if len(candles) < 2:
            continue
        closed = candles[-2]
        closed_ts = closed[0]
        if closed_ts <= state.last_bar_ts:
            continue
        state.last_bar_ts = closed_ts

        df = append_bar(df, closed)
        df = df.tail(MIN_BARS + 50).reset_index(drop=True)

        high = float(closed[2])
        low = float(closed[3])
        close = float(closed[4])
        LOG.debug("Closed bar O=%.5f H=%.5f L=%.5f C=%.5f", closed[1], high, low, close)

        # Update open groups based on the closed bar.
        if live:
            for group in state.groups:
                await sync_group_orders(exchange, group)
                if not group.is_closed():
                    await handle_stop_triggers(exchange, group, high, low)
        else:
            for group in state.groups:
                if not group.is_closed():
                    simulate_bar_for_group(group, high, low)

        # Apply trailing stops at end of bar (matches app.py logic).
        for group in state.groups:
            if apply_trailing(group):
                LOG.info("Trail updated %s stop=%.5f", group.id, group.stop_price)

        # Clean up closed groups and update paper balance.
        still_open = []
        for group in state.groups:
            if group.is_closed():
                pnl = compute_group_pnl(group)
                if not live:
                    state.paper_balance += pnl
                LOG.info("Trade %s closed PnL=%.4f", group.id, pnl)
            else:
                still_open.append(group)
        state.groups = still_open

        if len(df) < MIN_BARS:
            continue

        side = compute_signal(df)
        LOG.debug("Signal side=%s", side)
        if side in (1, -1):
            if CONFLICT_MODE == "hedged":
                await open_new_trade(exchange, state, df, side, live)


async def async_main() -> None:
    parser = argparse.ArgumentParser(description="Run the TripleTP_Trail05R_40-30-30 strategy on Binance spot.")
    parser.add_argument("--live", action="store_true", help="Enable live trading (requires API keys in .env).")
    parser.add_argument("--paper-balance", type=float, default=1000.0, help="Paper balance for dry-run sizing (USDT).")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING).")
    parser.add_argument("--ws-verbose", action="store_true", help="Enable CCXT websocket debug logging.")
    args = parser.parse_args()

    try:
        await run_strategy(args.live, args.paper_balance, args.log_level, args.ws_verbose)
    except KeyboardInterrupt:
        LOG.info("Shutdown requested.")


if __name__ == "__main__":
    asyncio.run(async_main())
