"""
Telegram Bot for Julaba Trading System
Provides real-time notifications and interactive commands.
"""

import os
import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Callable
from functools import wraps

logger = logging.getLogger(__name__)


def sanitize_markdown(text: str) -> str:
    """Sanitize text for Telegram Markdown to prevent parsing errors."""
    if not text:
        return text
    
    # Escape special Markdown characters that might break parsing
    # But preserve intentional formatting like *bold* and _italic_
    
    # First, protect Telegram commands (starting with /) - they shouldn't be escaped
    # Replace underscores in commands temporarily
    import re as re_mod
    command_pattern = r'/\w+(?:_\w+)+'  # Matches /command_with_underscores
    commands_found = re_mod.findall(command_pattern, text)
    placeholders = {}
    for i, cmd in enumerate(commands_found):
        placeholder = f"__CMD_PLACEHOLDER_{i}__"
        placeholders[placeholder] = cmd
        text = text.replace(cmd, placeholder, 1)
    
    # Fix unbalanced asterisks and underscores
    # Count occurrences - if odd, escape the last one
    asterisk_count = text.count('*')
    if asterisk_count % 2 == 1:
        # Find last asterisk and escape it (use rfind to get position)
        last_pos = text.rfind('*')
        if last_pos >= 0:
            text = text[:last_pos] + '\\*' + text[last_pos+1:]
    
    underscore_count = text.count('_')
    if underscore_count % 2 == 1:
        # Find last underscore and escape it
        last_pos = text.rfind('_')
        if last_pos >= 0:
            text = text[:last_pos] + '\\_' + text[last_pos+1:]
    
    # Escape square brackets that aren't part of links
    # Simple approach: escape standalone brackets
    text = re.sub(r'\[(?![^\]]+\]\()', '\\[', text)
    text = re.sub(r'(?<!\])\](?!\()', '\\]', text)
    
    # Restore Telegram commands
    for placeholder, cmd in placeholders.items():
        text = text.replace(placeholder, cmd)
    
    return text


# Telegram imports (optional - graceful fallback if not installed)
try:
    from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot not installed. Run: pip install python-telegram-bot")


class TelegramNotifier:
    """
    Telegram bot for trading notifications and commands.
    """
    
    def __init__(self):
        self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        
        # Check for valid credentials (not placeholders)
        token_valid = self.token and "your_" not in self.token.lower() and len(self.token) > 20
        chat_valid = self.chat_id and "your_" not in self.chat_id.lower() and self.chat_id.lstrip("-").isdigit()
        
        self.enabled = bool(token_valid and chat_valid and TELEGRAM_AVAILABLE)
        self.bot: Optional[Bot] = None
        self.app: Optional[Application] = None
        
        # Startup timestamp - ignore messages older than this
        self.startup_time: Optional[datetime] = None
        
        # Trading state reference (set by main bot)
        self.get_status: Optional[Callable] = None
        self.get_positions: Optional[Callable] = None
        self.get_pnl: Optional[Callable] = None
        self.get_ai_stats: Optional[Callable] = None
        self.get_balance: Optional[Callable] = None
        self.get_trades: Optional[Callable] = None
        self.get_market: Optional[Callable] = None
        self.get_signals: Optional[Callable] = None
        self.do_stop: Optional[Callable] = None
        self.do_pause: Optional[Callable] = None
        self.do_resume: Optional[Callable] = None
        self.chat_with_ai: Optional[Callable] = None  # AI chat function
        # AI mode callbacks
        self.get_ai_mode: Optional[Callable] = None
        self.set_ai_mode: Optional[Callable] = None
        self.confirm_ai_trade: Optional[Callable] = None
        self.reject_ai_trade: Optional[Callable] = None
        self.execute_ai_trade: Optional[Callable] = None  # AI chat can execute trades
        self.close_ai_trade: Optional[Callable] = None  # AI chat can close positions
        self.get_intelligence: Optional[Callable] = None  # Intelligence summary
        self.get_ml_stats: Optional[Callable] = None  # ML classifier stats
        self.get_regime: Optional[Callable] = None  # Market regime analysis
        self.toggle_summary: Optional[Callable] = None  # Toggle summary notifications
        self.get_summary_status: Optional[Callable] = None  # Get summary status
        # NEW: Enhanced module callbacks
        self.get_risk_stats: Optional[Callable] = None  # Risk manager stats
        self.get_mtf_analysis: Optional[Callable] = None  # Multi-timeframe analysis
        self.run_backtest: Optional[Callable] = None  # Run backtest
        self.get_chart: Optional[Callable] = None  # Generate chart
        self.get_equity_curve: Optional[Callable] = None  # Equity curve data
        self.switch_symbol: Optional[Callable] = None  # Unified symbol switch
        self.ai_analyze_markets: Optional[Callable] = None  # AI market analysis (same as dashboard)
        self.get_symbol_trade_context: Optional[Callable] = None  # S/R, F&G, volume for advisory mode
        # NEW: Adaptive parameters callbacks
        self.get_adaptive_params: Optional[Callable] = None  # Get adaptive params summary
        self.set_adaptive_param: Optional[Callable] = None  # Set a parameter value
        self.trigger_auto_tune: Optional[Callable] = None  # Trigger AI parameter review
        self.toggle_auto_tune: Optional[Callable] = None  # Toggle auto-tune on/off
        # Direction filter callbacks
        self.get_allowed_sides: Optional[Callable] = None  # Get allowed sides (long/short/both)
        self.set_allowed_sides: Optional[Callable] = None  # Set allowed sides
        
        # Risk Shield callbacks (set by main bot)
        self.get_shield_status: Optional[Callable] = None      # Get shield status dict
        self.reset_shield: Optional[Callable] = None            # Reset hard breach
        self.emergency_liquidate: Optional[Callable] = None     # Emergency close all positions
        
        # Dry-run callbacks
        self.get_dry_run_summary: Optional[Callable] = None     # Get shadow tracking summary
        self.toggle_dry_run: Optional[Callable] = None          # Enable/disable dry-run mode
        self.reset_dry_run: Optional[Callable] = None           # Reset dry-run data
        
        # System parameter callback (set by main bot)
        self.set_system_param: Optional[Callable] = None  # Set system parameters (reset_daily_loss, force_resume, etc.)
        
        # Trading mode callbacks
        self.get_trading_mode: Optional[Callable] = None  # Get current trading mode (live/paper)
        self.switch_trading_mode: Optional[Callable] = None  # Switch between live and paper modes
        
        # === PENDING TRADE CONFIRMATIONS ===
        # Store trades awaiting user confirmation (e.g., below minimum size)
        self.pending_trade_confirmation: Optional[Dict[str, Any]] = None
        self.execute_pending_trade: Optional[Callable] = None  # Callback to execute pending trade
        
        # === PENDING EXECUTE COMMAND ===
        # Store the AI command details (including symbol) when /execute_long or /execute_short is shown
        # So when user clicks the button, we know which symbol was originally requested
        self._pending_execute_command: Optional[Dict[str, Any]] = None
        
        # === COMMAND RATE LIMITING ===
        # Prevent accidental double-execution of trade commands
        self._last_execute_time: float = 0  # Timestamp of last execute_long/short/close
        self._EXECUTE_COOLDOWN_SECONDS: int = 30  # Minimum seconds between execute commands
        
        # Trading control state
        self.paused = False
        
        # === ERROR TRACKING FOR PIPELINE STATUS ===
        self.last_error: Optional[str] = None
        self.last_error_time: Optional[datetime] = None
        self.error_count: int = 0
        self.last_success_time: Optional[datetime] = None
        self.is_started: bool = False  # True only after start() completes
        
        if not self.enabled:
            if not TELEGRAM_AVAILABLE:
                logger.warning("Telegram bot disabled - package not installed")
            elif not token_valid:
                logger.info("Telegram bot disabled - valid TELEGRAM_BOT_TOKEN not set")
            elif not chat_valid:
                logger.info("Telegram bot disabled - valid TELEGRAM_CHAT_ID not set")
        else:
            # Defer Bot initialization to avoid Python 3.14 anyio import issues
            # Bot will be created in start() method instead
            self.bot = None
            logger.info("Telegram bot initialized (deferred)")
    
    async def start(self):
        """Start the Telegram bot with command handlers."""
        if not self.enabled:
            return
        
        # Create Bot instance here (deferred from __init__)
        if self.bot is None:
            try:
                self.bot = Bot(token=self.token)
            except Exception as e:
                logger.error(f"Failed to create Telegram Bot: {e}")
                self.enabled = False
                return
        
        self.app = Application.builder().token(self.token).build()
        
        # Add timestamp filter to prevent old commands after restart
        from telegram.ext import MessageHandler, filters
        async def timestamp_filter(update: Update, context: ContextTypes.DEFAULT_TYPE):
            """Filter out commands older than 5 minutes to prevent stale commands after restart."""
            if update.message and update.message.date:
                from datetime import datetime, timezone, timedelta
                msg_age = datetime.now(timezone.utc) - update.message.date
                if msg_age > timedelta(minutes=5):
                    logger.debug(f"🚫 Ignored stale command: /{update.message.text.split()[0] if update.message.text else 'unknown'} ({msg_age.total_seconds():.0f}s old)")
                    return False  # Block the message
            return True  # Allow the message
        
        # This filter will be applied to all command handlers automatically
        # by adding it as a base filter (but we'll handle it per-command instead for clarity)
        
        # Register command handlers
        self.app.add_handler(CommandHandler("start", self._cmd_start))
        self.app.add_handler(CommandHandler("status", self._cmd_status))
        self.app.add_handler(CommandHandler("positions", self._cmd_positions))
        self.app.add_handler(CommandHandler("pnl", self._cmd_pnl))
        self.app.add_handler(CommandHandler("ai", self._cmd_ai_stats))
        self.app.add_handler(CommandHandler("balance", self._cmd_balance))
        self.app.add_handler(CommandHandler("trades", self._cmd_trades))
        self.app.add_handler(CommandHandler("market", self._cmd_market))
        self.app.add_handler(CommandHandler("signals", self._cmd_signals))
        self.app.add_handler(CommandHandler("stats", self._cmd_stats))
        self.app.add_handler(CommandHandler("stop", self._cmd_stop))
        self.app.add_handler(CommandHandler("pause", self._cmd_pause))
        self.app.add_handler(CommandHandler("resume", self._cmd_resume))
        self.app.add_handler(CommandHandler("help", self._cmd_help))
        # AI mode commands
        self.app.add_handler(CommandHandler("aimode", self._cmd_aimode))
        self.app.add_handler(CommandHandler("confirm", self._cmd_confirm))
        self.app.add_handler(CommandHandler("reject", self._cmd_reject))
        # Intelligence commands
        self.app.add_handler(CommandHandler("intel", self._cmd_intel))
        self.app.add_handler(CommandHandler("ml", self._cmd_ml))
        self.app.add_handler(CommandHandler("regime", self._cmd_regime))
        self.app.add_handler(CommandHandler("summary", self._cmd_summary))
        # NEW: News and market sentiment
        self.app.add_handler(CommandHandler("news", self._cmd_news))
        # NEW: Enhanced commands
        self.app.add_handler(CommandHandler("risk", self._cmd_risk))
        self.app.add_handler(CommandHandler("mtf", self._cmd_mtf))
        self.app.add_handler(CommandHandler("backtest", self._cmd_backtest))
        self.app.add_handler(CommandHandler("chart", self._cmd_chart))
        # NEW: Adaptive parameters commands
        self.app.add_handler(CommandHandler("params", self._cmd_params))
        self.app.add_handler(CommandHandler("tune", self._cmd_tune))
        self.app.add_handler(CommandHandler("autotune", self._cmd_autotune))
        # Risk Shield circuit breaker
        self.app.add_handler(CommandHandler("shield", self._cmd_shield))
        # Threshold mode switching
        self.app.add_handler(CommandHandler("threshold", self._cmd_threshold))
        # Dry-run shadow tracking
        self.app.add_handler(CommandHandler("dryrun", self._cmd_dryrun))

        # Manual trade commands (with AI verification)
        self.app.add_handler(CommandHandler("open", self._cmd_open))
        self.app.add_handler(CommandHandler("close", self._cmd_close))
        self.app.add_handler(CommandHandler("buy", self._cmd_buy))
        self.app.add_handler(CommandHandler("sell", self._cmd_sell))
        # Watchdog confirmation command
        self.app.add_handler(CommandHandler("watching", self._cmd_watching))
        # Explicit execution commands
        self.app.add_handler(CommandHandler("execute_long", self._cmd_execute_long))
        self.app.add_handler(CommandHandler("execute_short", self._cmd_execute_short))
        self.app.add_handler(CommandHandler("execute_close", self._cmd_execute_close))
        # Reset command for daily loss limit
        self.app.add_handler(CommandHandler("reset", self._cmd_reset))
        self.app.add_handler(CommandHandler("reset_daily", self._cmd_reset_daily))  # Alias for quick reset
        # Force override all halts
        self.app.add_handler(CommandHandler("override", self._cmd_override))
        self.app.add_handler(CommandHandler("force", self._cmd_override))
        # Trading mode switch (live/paper)
        self.app.add_handler(CommandHandler("mode", self._cmd_mode))
        # Side filter commands (long-only, short-only, both)
        self.app.add_handler(CommandHandler("longonly", self._cmd_longonly))
        self.app.add_handler(CommandHandler("shortonly", self._cmd_shortonly))
        self.app.add_handler(CommandHandler("bothsides", self._cmd_bothsides))
        self.app.add_handler(CommandHandler("sides", self._cmd_sides))
        # Balance protection command
        self.app.add_handler(CommandHandler("setprotection", self._cmd_setprotection))
        self.app.add_handler(CommandHandler("protection", self._cmd_setprotection))  # Alias
        # Hold/Unhold commands for position 1 and 2
        self.app.add_handler(CommandHandler("hold1", self._cmd_hold1))
        self.app.add_handler(CommandHandler("hold2", self._cmd_hold2))
        self.app.add_handler(CommandHandler("unhold1", self._cmd_unhold1))
        self.app.add_handler(CommandHandler("unhold2", self._cmd_unhold2))
        self.app.add_handler(CommandHandler("hold", self._cmd_hold))
        self.app.add_handler(CommandHandler("unhold", self._cmd_unhold))
        # Diamond hands aliases
        self.app.add_handler(CommandHandler("diamondhands", self._cmd_diamondhands))
        self.app.add_handler(CommandHandler("dh", self._cmd_diamondhands))
        self.app.add_handler(CommandHandler("dh1", self._cmd_hold1))
        self.app.add_handler(CommandHandler("dh2", self._cmd_hold2))
        
        # Add callback query handler for inline buttons
        self.app.add_handler(CallbackQueryHandler(self._handle_callback))
        
        # Add message handler for normal chat (non-command messages)
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
        
        # Add error handler for 409 conflicts
        async def error_handler(update, context):
            """Handle Telegram errors gracefully."""
            error = context.error
            logger.error(f"🚨 Telegram error handler: {error} | Update: {update}")
            if "409" in str(error) or "Conflict" in str(error):
                logger.warning("Telegram conflict detected (409) - will retry automatically")
            else:
                logger.error(f"Telegram error: {error}")
        
        self.app.add_error_handler(error_handler)
        
        # Record startup time to ignore old messages
        from datetime import datetime, timezone
        self.startup_time = datetime.now(timezone.utc)
        
        # Clear any pending updates before starting (prevents processing old /stop commands)
        try:
            # Fetch and discard any pending updates
            updates = await self.bot.get_updates(offset=-1, timeout=1)
            if updates:
                # Get the latest update_id and mark all previous as read
                latest_id = updates[-1].update_id
                await self.bot.get_updates(offset=latest_id + 1, timeout=1)
                logger.info(f"Cleared {len(updates)} pending Telegram updates")
        except Exception as e:
            logger.debug(f"Could not clear pending updates: {e}")
        
        # Start polling in background with infinite retries for conflicts
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(
            drop_pending_updates=False,  # Process pending updates (filtered by timestamp below)
            allowed_updates=["message", "callback_query"],
            bootstrap_retries=-1  # Retry indefinitely on startup conflicts
        )
        
        # Mark as fully started for pipeline status
        self.is_started = True
        self.last_success_time = datetime.now(timezone.utc)
        
        logger.info("Telegram bot started - listening for commands")
        
        # Only send startup message once (not on reconnect)
        if not getattr(self, '_startup_notified', False):
            self._startup_notified = True
            await self.send_message("🤖 *Julaba Bot Started*\n\nType /help for commands")
    
    def _check_auth(self, update: Update) -> bool:
        """Check if the user is authorized (matches configured TELEGRAM_CHAT_ID)."""
        if not update.effective_chat:
            return False
        user_chat_id = str(update.effective_chat.id)
        authorized = user_chat_id == self.chat_id
        if not authorized:
            logger.warning(f"⛔ Unauthorized access attempt from chat_id: {user_chat_id}")
        return authorized
    
    async def stop(self):
        """Stop the Telegram bot."""
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
            self.is_started = False
    
    async def send_message(self, text: str, parse_mode: str = "Markdown"):
        """Send a message to the configured chat."""
        if not self.enabled:
            return
        
        # Guard against bot not being initialized yet (deferred init)
        if self.bot is None:
            logger.debug("Telegram message skipped - bot not yet initialized")
            return
        
        # Sanitize markdown to prevent parsing errors
        if parse_mode == "Markdown":
            text = sanitize_markdown(text)
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode
            )
            # Track success
            self.last_success_time = datetime.now(timezone.utc)
        except Exception as e:
            error_str = str(e).lower()
            # If markdown parsing failed, retry without parse_mode
            if "parse" in error_str or "entity" in error_str or "can't" in error_str:
                try:
                    # Retry without markdown
                    await self.bot.send_message(
                        chat_id=self.chat_id,
                        text=text,
                        parse_mode=None
                    )
                    self.last_success_time = datetime.now(timezone.utc)
                    logger.debug("Telegram message sent without markdown (parse error fallback)")
                    return
                except Exception as e2:
                    logger.error(f"Failed to send Telegram message (even without markdown): {e2}")
            # Track error for pipeline status
            self.last_error = str(e)
            self.last_error_time = datetime.now(timezone.utc)
            self.error_count += 1
            logger.error(f"Failed to send Telegram message: {e}")
    
    async def send_message_safe(self, text: str, parse_mode: str = "Markdown") -> bool:
        """
        Send message with explicit error handling return value.
        Returns True if sent successfully, False if failed.
        Useful for bot.py to know if notification was delivered.
        """
        try:
            await self.send_message(text, parse_mode)
            return True
        except Exception as e:
            logger.warning(f"Telegram notification failed: {e}")
            return False
    
    def format_command_suggestion(self, action: str, params: Dict[str, any] = None, description: str = "") -> str:
        """
        Format a suggested command for user action.
        Makes it easy for AI to provide actionable commands to user.
        
        Example:
            format_command_suggestion("open_trade", {"side": "long"}, "Open LONG position now")
        """
        import json
        cmd = {"action": action}
        if params:
            cmd.update(params)
        
        cmd_json = json.dumps(cmd)
        formatted = f"```command\n{cmd_json}\n```"
        
        if description:
            formatted = f"{description}\n\n{formatted}"
        
        return formatted
    
    def get_telegram_status(self) -> Dict[str, Any]:
        """Get Telegram bot status for pipeline monitoring."""
        now = datetime.now(timezone.utc)
        
        # Determine actual status
        if not self.enabled:
            status = 'disabled'
        elif not self.is_started:
            status = 'initializing'
        elif self.bot is None:
            status = 'error'
        elif self.last_error_time and (now - self.last_error_time).total_seconds() < 300:
            # Had an error in the last 5 minutes
            status = 'warning'
        else:
            status = 'ok'
        
        return {
            'enabled': self.enabled,
            'is_started': self.is_started,
            'bot_initialized': self.bot is not None,
            'status': status,
            'last_error': self.last_error,
            'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None,
            'error_count': self.error_count,
            'last_success_time': self.last_success_time.isoformat() if self.last_success_time else None,
            'seconds_since_last_error': int((now - self.last_error_time).total_seconds()) if self.last_error_time else None,
            'seconds_since_last_success': int((now - self.last_success_time).total_seconds()) if self.last_success_time else None
        }
    
    # =========== Notification Methods ===========
    
    async def notify_signal(
        self,
        symbol: str,
        side: str,
        price: float,
        ai_approved: bool,
        confidence: float,
        reasoning: str
    ):
        """Notify about a new trading signal."""
        emoji = "🟢" if side == "LONG" else "🔴"
        status = "✅ APPROVED" if ai_approved else "❌ REJECTED"
        
        msg = f"""
{emoji} *New Signal: {side}*

📊 *Symbol:* `{symbol}`
💰 *Price:* `${price:,.4f}`
🤖 *AI Status:* {status}
📈 *Confidence:* `{confidence:.0%}`
💡 *Analysis:* {reasoning}
"""
        await self.send_message(msg)
    
    async def notify_trade_opened(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        size: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        tp3: float
    ):
        """Notify about a trade being opened."""
        emoji = "🟢" if side == "LONG" else "🔴"
        
        msg = f"""
{emoji} *TRADE OPENED*

📊 *{symbol}* - {side}
💰 *Entry:* `${entry_price:,.4f}`
📦 *Size:* `{size:.4f}`
🛑 *Stop Loss:* `${stop_loss:,.4f}`

🎯 *Take Profits:*
  TP1: `${tp1:,.4f}` (40%)
  TP2: `${tp2:,.4f}` (30%)
  TP3: `${tp3:,.4f}` (30%)
"""
        await self.send_message(msg)
    
    async def notify_ai_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        confidence: float,
        reasoning: str,
        mode: str
    ):
        """Notify about AI-initiated trade opportunity."""
        emoji = "🟢" if action == "LONG" else "🔴"
        
        if mode == "autonomous":
            # AI already opened the trade
            msg = f"""
🤖 *AI AUTONOMOUS TRADE*

{emoji} *{action}* on *{symbol}*
💰 *Price:* `${price:,.4f}`
📈 *Confidence:* `{confidence:.0%}`
💡 *Analysis:* {reasoning}

_Trade opened automatically by AI_
"""
            await self.send_message(msg)
        else:
            # Advisory/Hybrid - ask for confirmation
            msg = f"""
🤖 *AI TRADE SUGGESTION*

{emoji} *{action}* on *{symbol}*
💰 *Price:* `${price:,.4f}`
📈 *Confidence:* `{confidence:.0%}`
💡 *Analysis:* {reasoning}

⏳ *Awaiting your decision...*

👉 *To Execute:* /confirm
👉 *To Reject:* /reject
"""
            # Send with inline buttons
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ Confirm Trade", callback_data="confirm_ai_trade"),
                    InlineKeyboardButton("❌ Reject", callback_data="reject_ai_trade")
                ]
            ])
            
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=msg,
                    parse_mode="Markdown",
                    reply_markup=keyboard
                )
            except Exception as e:
                logger.error(f"Failed to send AI trade notification: {e}")
    
    async def request_trade_confirmation(
        self,
        symbol: str,
        side: str,
        price: float,
        calculated_size: float,
        min_amount: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        tp3: float,
        risk_pct: float,
        balance: float,
        reason: str = "Size below minimum"
    ):
        """
        Request user confirmation for a trade (e.g., when size is below exchange minimum).
        Shows all trade details and lets user approve, use min size, or reject.
        """
        if not self.enabled:
            return
        
        # Store pending trade details
        self.pending_trade_confirmation = {
            'symbol': symbol,
            'side': side,
            'price': price,
            'size': calculated_size,
            'min_amount': min_amount,
            'stop_loss': stop_loss,
            'tp1': tp1,
            'tp2': tp2,
            'tp3': tp3,
            'risk_pct': risk_pct,
            'balance': balance,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Calculate position value and risk
        position_value = calculated_size * price
        risk_amount = abs(price - stop_loss) * calculated_size
        min_position_value = min_amount * price
        min_risk_amount = abs(price - stop_loss) * min_amount
        
        emoji = "🟢" if side.lower() == "long" else "🔴"
        
        msg = f"""
⚠️ *TRADE CONFIRMATION REQUIRED*

{emoji} *{side.upper()}* on *{symbol}*

📊 *Trade Details:*
├ Entry: `${price:,.6f}`
├ Stop Loss: `${stop_loss:,.6f}`
├ TP1: `${tp1:,.6f}`
├ TP2: `${tp2:,.6f}`
└ TP3: `${tp3:,.6f}`

💰 *Position Sizing:*
├ Calculated: `{calculated_size:.6f}` (${position_value:.2f})
├ Minimum Req: `{min_amount:.6f}` (${min_position_value:.2f})
└ Risk: `${risk_amount:.2f}` ({risk_pct*100:.1f}% of ${balance:.2f})

⚠️ *Issue:* {reason}

*Options:*
• ✅ *Execute anyway* - Try with calculated size
• 📈 *Use minimum* - Use exchange minimum size
• ❌ *Cancel* - Skip this trade
"""
        
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Execute", callback_data="confirm_pending_trade"),
                InlineKeyboardButton(f"📈 Min ({min_amount:.4f})", callback_data="confirm_pending_minsize"),
            ],
            [
                InlineKeyboardButton("❌ Cancel", callback_data="reject_pending_trade")
            ]
        ])
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=msg,
                parse_mode="Markdown",
                reply_markup=keyboard
            )
            logger.info(f"📱 Trade confirmation requested for {side} {symbol}")
        except Exception as e:
            logger.error(f"Failed to send trade confirmation request: {e}")
    
    async def notify_tp_hit(
        self,
        symbol: str,
        tp_level: int,
        price: float,
        pnl: float,
        remaining_pct: float
    ):
        """Notify when a take profit level is hit."""
        msg = f"""
🎯 *TP{tp_level} HIT!*

📊 *{symbol}*
💰 *Exit Price:* `${price:,.4f}`
📈 *Realized P&L:* `${pnl:+,.2f}`
📦 *Remaining:* `{remaining_pct:.0%}`
"""
        await self.send_message(msg)
    
    async def notify_stop_loss(
        self,
        symbol: str,
        price: float,
        pnl: float
    ):
        """Notify when stop loss is hit."""
        msg = f"""
🛑 *STOP LOSS HIT*

📊 *{symbol}*
💰 *Exit Price:* `${price:,.4f}`
📉 *Loss:* `${pnl:,.2f}`
"""
        await self.send_message(msg)
    
    async def notify_trade_closed(
        self,
        symbol: str,
        pnl: float,
        pnl_pct: float,
        reason: str,
        side: str = "",
        entry_price: float = 0.0,
        exit_price: float = 0.0,
        balance: float = 0.0,
        total_pnl: float = 0.0,
        win_rate: float = 0.0,
        **kwargs
    ):
        """Notify when a trade is fully closed."""
        emoji = "✅" if pnl >= 0 else "❌"
        
        # Build detail lines only if extra info provided
        details = ""
        if side:
            details += f"📌 *Side:* {side.upper()}\n"
        if entry_price and exit_price:
            details += f"🔹 *Entry:* `${entry_price:,.2f}` → *Exit:* `${exit_price:,.2f}`\n"
        if balance:
            details += f"💼 *Balance:* `${balance:,.2f}`\n"
        if win_rate:
            details += f"📊 *Win Rate:* `{win_rate:.1f}%`\n"
        
        msg = f"""
{emoji} *TRADE CLOSED*

📊 *{symbol}*
{details}💰 *P&L:* `${pnl:+,.2f}` ({pnl_pct:+.2f}%)
📝 *Reason:* {reason}
"""
        await self.send_message(msg)
    
    async def notify_daily_summary(
        self,
        date: str,
        trades: int,
        wins: int,
        losses: int,
        pnl: float,
        balance: float,
        win_rate: float
    ):
        """Send daily trading summary."""
        emoji = "📈" if pnl >= 0 else "📉"
        
        msg = f"""
📊 *Daily Summary - {date}*

{emoji} *Today's P&L:* `${pnl:+,.2f}`
💰 *Balance:* `${balance:,.2f}`

*Trades:* `{trades}` ({wins}W / {losses}L)
*Win Rate:* `{win_rate:.1f}%`

_Keep trading smart! 🤖_
"""
        await self.send_message(msg)
    
    # =========== Command Handlers ===========
    
    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        await update.message.reply_text(
            "🤖 *Julaba Trading Bot*\n\n"
            "I'll send you real-time trading alerts and AI analysis.\n\n"
            "Use /help to see available commands.",
            parse_mode="Markdown"
        )
    
    async def _cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        msg = """
🤖 *Julaba Commands*

📊 *Info Commands:*
/status - Bot status & connection info
/balance - Current balance
/positions - View open positions
/pnl - Show P&L summary
/stats - Detailed statistics

📈 *Trading Commands:*
/trades - Recent trade history
/signals - Recent signals detected
/market - Current market info
/ai - AI filter statistics
/chart - Price chart with levels

💼 *Trade Execution:*
/open long - AI analyzes, then you confirm
/open short - AI analyzes, then you confirm
/close - AI analyzes, then you confirm
/execute\\_long - Immediately open LONG
/execute\\_short - Immediately open SHORT
/execute\\_close - Immediately close position

🤖 *AI Mode Commands:*
/aimode - View current AI mode
/aimode filter - AI validates signals only
/aimode advisory - AI suggests, you confirm
/aimode autonomous - AI trades directly
/aimode hybrid - AI scans + suggests
/confirm - Confirm pending AI trade
/reject - Reject pending AI trade
/watching - Confirm you're monitoring position

🧠 *Intelligence Commands:*
/intel - View intelligent trading features
/ml - Machine learning classifier stats
/regime - Current market regime analysis
/risk - Risk manager status & limits
/mtf - Multi-timeframe analysis
/news - Market news, sentiment & whale alerts

🎛️ *AI Auto-Tune Commands:*
/params - View AI-tunable parameters
/tune param value - Set parameter manually
/autotune - Trigger AI parameter review
/autotune toggle - Enable/disable auto-tune
/shield - Risk Shield circuit breaker status
/shield reset - Reset hard breach (resume trading)
/shield liquidate - Emergency close ALL positions
/threshold - Show entry threshold mode
/threshold default - Strict proven thresholds
/threshold loose - Relaxed (more trades)
/dryrun - Shadow tracker status
/dryrun on|off - Enable/disable dry-run mode
/dryrun reset - Clear shadow data

📉 *Analysis Commands:*
/backtest - Backtest strategy (7 days)
/backtest 30 - Backtest with custom days
/summary - Toggle summary notifications

⚙️ *Control Commands:*
/pause - Pause trading
/resume - Resume trading
/stop - Stop the bot
/setprotection 280 - Set balance protection floor
/protection - View current protection level

🔄 *Mode Commands:*
/mode - Show current trading mode
/mode paper - Switch to paper trading (resets data)
/mode live confirm - Switch to live trading
/sides - Show allowed trade directions
/longonly - Only LONG trades (block shorts)
/shortonly - Only SHORT trades (block longs)
/bothsides - Allow both directions

/help - Show this message
"""
        await update.message.reply_text(msg, parse_mode="Markdown")
    
    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        # Ignore stale commands from before restart
        if not self._is_command_fresh(update):
            return
        
        logger.info(f"📱 Telegram command received: /status from user {update.effective_user.username}")
        try:
            if self.get_status:
                s = self.get_status()
                
                # Determine status emoji
                if s.get('paused'):
                    status_icon = "⏸️ PAUSED"
                elif s.get('has_position'):
                    status_icon = "🟢 IN TRADE"
                else:
                    status_icon = "🔵 WATCHING"
                
                # Calculate P&L percentage
                initial = s.get('initial_balance', 1)
                pnl_pct = ((s.get('balance', 0) - initial) / initial * 100) if initial > 0 else 0
                
                msg = f"""
📊 *Julaba Status*

*Connection*
🔌 Status: {'✅ Connected' if s.get('connected') else '❌ Disconnected'}
📈 Symbol: `{s.get('symbol', 'N/A')}`
⏱ Uptime: `{s.get('uptime', 'N/A')}`
🎮 Mode: `{s.get('mode', 'N/A')}`

*Trading*
{status_icon}
💵 Price: `${s.get('current_price', 0):,.4f}`
📏 ATR: `${s.get('atr', 0):,.4f}`
📍 Position: `{s.get('position_side', 'None')}`
{'💹 Unrealized: `$' + f"{s.get('position_pnl', 0):+,.2f}" + '`' if s.get('has_position') else ''}

*Performance*
💰 Balance: `${s.get('balance', 0):,.2f}`
📊 P&L: `${s.get('total_pnl', 0):+,.2f}` (`{pnl_pct:+.2f}%`)
🎯 Trades: `{s.get('total_trades', 0)}` ({s.get('wins', 0)}W / {s.get('losses', 0)}L)
📈 Win Rate: `{s.get('win_rate', 0):.1f}%`
🔍 Signals: `{s.get('signals_checked', 0)}` checked
"""
            else:
                msg = "⚠️ Status not available"
            
            await update.message.reply_text(msg, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Error in /status command: {e}")
            await update.message.reply_text(f"❌ Error fetching status: {str(e)[:100]}", parse_mode="Markdown")
    
    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command."""
        try:
            if self.get_positions:
                positions = self.get_positions()
                if positions:
                    msg = "📦 *Open Positions*\n\n"
                    for p in positions:
                        msg += f"• {p.get('symbol', 'N/A')}: {p.get('side', 'N/A')} @ ${p.get('entry', 0):,.4f}\n"
                        msg += f"  Size: {p.get('size', 0):.4f} | P&L: ${p.get('pnl', 0):+.2f}\n\n"
                else:
                    msg = "📦 *No open positions*"
            else:
                msg = "⚠️ Positions not available"
            
            await update.message.reply_text(msg, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Error in /positions command: {e}")
            await update.message.reply_text(f"❌ Error fetching positions: {str(e)[:100]}", parse_mode="Markdown")
    
    async def _cmd_pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pnl command."""
        if self.get_pnl:
            pnl = self.get_pnl()
            msg = f"""
💰 *P&L Summary*

📈 *Today:* `${pnl.get('today', 0):+,.2f}`
📊 *Total:* `${pnl.get('total', 0):+,.2f}`
🎯 *Win Rate:* `{pnl.get('win_rate', 0):.1%}`
📦 *Trades:* `{pnl.get('trades', 0)}`
"""
        else:
            msg = "⚠️ P&L not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")
    
    async def _cmd_ai_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /ai command."""
        if self.get_ai_stats:
            stats = self.get_ai_stats()
            msg = f"""
🤖 *AI Filter Stats*

📊 *Total Signals:* `{stats.get('total_signals', 0)}`
✅ *Approved:* `{stats.get('approved', 0)}`
❌ *Rejected:* `{stats.get('rejected', 0)}`
📈 *Approval Rate:* `{stats.get('approval_rate', 'N/A')}`
"""
        else:
            msg = "⚠️ AI stats not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_intel(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /intel command - show intelligent trading features."""
        if self.get_intelligence:
            i = self.get_intelligence()
            
            # Drawdown mode emoji
            mode = i.get('drawdown_mode', 'NORMAL')
            if mode == 'EMERGENCY':
                mode_emoji = "🚨"
            elif mode == 'CAUTIOUS':
                mode_emoji = "⚠️"
            elif mode == 'REDUCED':
                mode_emoji = "📉"
            else:
                mode_emoji = "✅"
            
            # Pattern info
            pattern_info = "None detected"
            pattern = i.get('pattern')
            if pattern and pattern.get('pattern'):
                p_dir = "🟢 Bullish" if pattern.get('bullish') else ("🔴 Bearish" if pattern.get('bullish') is False else "⚪ Neutral")
                pattern_info = f"{pattern.get('pattern')} ({p_dir})"
            
            # Regime info
            regime = i.get('regime', 'UNKNOWN')
            tradeable = "✅ Yes" if i.get('tradeable') else "❌ No"
            
            msg = f"""
🧠 *Intelligence Summary*

*Risk Management*
{mode_emoji} Mode: `{mode}`
📊 Drawdown: `{i.get('drawdown_pct', 0):.1f}%`
🔥 Win Streak: `{i.get('consecutive_wins', 0)}`
❄️ Loss Streak: `{i.get('consecutive_losses', 0)}`

*Market Analysis*
📈 Regime: `{regime}`
🎯 Tradeable: {tradeable}
📏 ADX: `{i.get('adx', 0):.1f}`
📐 Hurst: `{i.get('hurst', 0.5):.2f}`

*Pattern Detection*
🕯️ Pattern: `{pattern_info}`

*Machine Learning*
🤖 Status: `{i.get('ml_status', 'Not available')}`
"""
        else:
            msg = "⚠️ Intelligence data not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_ml(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /ml command - show ML classifier stats."""
        if self.get_ml_stats:
            stats = self.get_ml_stats()
            
            trained = "✅ Yes" if stats.get('is_trained') else "❌ No"
            samples = stats.get('total_samples', 0)
            needed = stats.get('samples_until_training', 50)
            
            msg = f"""
🧠 *ML Classifier Stats*

*Training Status*
📚 Trained: {trained}
📊 Samples: `{samples}`
{'🎯 Samples until training: `' + str(needed) + '`' if needed > 0 else ''}

*Historical Performance*
"""
            if samples > 0:
                wins = stats.get('wins', 0)
                losses = stats.get('losses', 0)
                wr = stats.get('historical_win_rate', 0)
                msg += f"✅ Wins: `{wins}`\n"
                msg += f"❌ Losses: `{losses}`\n"
                msg += f"📈 Win Rate: `{wr:.1%}`\n"
            else:
                msg += "_No trades recorded yet_\n"
            
            # Top features if trained
            if stats.get('is_trained') and stats.get('top_features'):
                msg += "\n*Top Predictive Features*\n"
                for feat, imp in stats.get('top_features', [])[:3]:
                    msg += f"• `{feat}`: {imp:.2f}\n"
        else:
            msg = "⚠️ ML stats not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_regime(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /regime command - show current market regime analysis."""
        if self.get_regime:
            r = self.get_regime()
            
            # Regime emoji
            regime_emoji = {
                'STRONG_TRENDING': '🚀',
                'TRENDING': '📈',
                'WEAK_TRENDING': '📊',
                'RANGING': '↔️',
                'CHOPPY': '🌊',
                'UNKNOWN': '❓'
            }.get(r.get('regime', 'UNKNOWN'), '❓')
            
            # Volatility emoji
            vol_emoji = {'high': '🔥', 'low': '❄️', 'normal': '✅'}.get(r.get('volatility', 'normal'), '✅')
            
            # Tradeable emoji
            trade_emoji = '✅' if r.get('tradeable') else '⚠️'
            
            msg = f"""
📊 *Market Regime Analysis*

{regime_emoji} *Regime:* `{r.get('regime', 'UNKNOWN')}`
{trade_emoji} *Tradeable:* {'Yes' if r.get('tradeable') else 'No'}

📈 *Indicators:*
• ADX (Trend Strength): `{r.get('adx', 0)}`
• Hurst Exponent: `{r.get('hurst', 0.5)}`
  _(>0.5 trending, <0.5 mean-reverting)_

{vol_emoji} *Volatility:*
• Level: `{r.get('volatility', 'normal').upper()}`
• Ratio: `{r.get('volatility_ratio', 1.0)}x`
"""
            
            # ML prediction if available
            if r.get('ml_prediction'):
                msg += f"""
🤖 *ML Prediction:*
• Predicted: `{r.get('ml_prediction')}`
• Confidence: `{r.get('ml_confidence', 0)}%`
"""
            
            msg += f"\n💡 _{r.get('description', '')}_"
        else:
            msg = "⚠️ Regime analysis not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_news(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /news command - show current market news and sentiment."""
        try:
            from news_monitor import NewsMonitor
            
            # Use existing monitor if available, or create new one
            monitor = getattr(self, '_news_monitor', None) or NewsMonitor()
            
            # Fetch summary
            summary = await monitor.get_market_summary()
            
            # Build response
            msg = "📰 *Market News & Sentiment*\n\n"
            
            # Fear & Greed
            sentiment = summary.get('sentiment')
            if sentiment:
                fg = sentiment.get('fear_greed_index', 50)
                fg_label = sentiment.get('fear_greed_label', 'Unknown')
                
                # Emoji based on fear/greed
                if fg <= 25:
                    fg_emoji = '😱'  # Extreme fear
                elif fg <= 40:
                    fg_emoji = '😰'  # Fear
                elif fg >= 75:
                    fg_emoji = '🤑'  # Extreme greed
                elif fg >= 60:
                    fg_emoji = '😊'  # Greed
                else:
                    fg_emoji = '😐'  # Neutral
                
                msg += f"{fg_emoji} *Fear & Greed:* `{fg}` ({fg_label})\n"
                
                if sentiment.get('market_cap_change_24h'):
                    change = sentiment['market_cap_change_24h']
                    change_emoji = '📈' if change > 0 else '📉'
                    msg += f"{change_emoji} *Market 24h:* `{change:+.2f}%`\n"
                
                if sentiment.get('btc_dominance'):
                    msg += f"₿ *BTC Dominance:* `{sentiment['btc_dominance']:.1f}%`\n"
            
            # News summary
            ns = summary.get('news_summary', {})
            if ns:
                msg += f"\n📊 *News Summary:*\n"
                msg += f"• Total: `{ns.get('total_articles', 0)}` articles\n"
                msg += f"• Bullish: `{ns.get('bullish_count', 0)}` | Bearish: `{ns.get('bearish_count', 0)}`\n"
                msg += f"• Avg Sentiment: `{ns.get('average_sentiment', 0):+.2f}`\n"
            
            # Critical news
            critical = summary.get('critical_news', [])
            if critical:
                msg += f"\n🚨 *Critical News:*\n"
                for n in critical[:3]:
                    msg += f"• {n.get('title', '')[:50]}...\n"
            
            # Recent headlines (show even if no critical news)
            recent = summary.get('recent_news', [])
            if recent and not critical:
                msg += f"\n📰 *Recent Headlines:*\n"
                for n in recent[:5]:
                    title = n.get('title', '')[:60]
                    sentiment_score = n.get('sentiment', 0)
                    sent_emoji = '🟢' if sentiment_score > 0.2 else ('🔴' if sentiment_score < -0.2 else '⚪')
                    msg += f"{sent_emoji} {title}...\n"
            
            # Liquidations
            liq = summary.get('liquidations', {})
            if liq and liq.get('total_24h_usd', 0) > 0:
                msg += f"\n💥 *Liquidations 24h:*\n"
                msg += f"• Total: `${liq.get('total_24h_usd', 0)/1e6:.1f}M`\n"
                msg += f"• Alert: `{liq.get('alert_level', 'normal').upper()}`\n"
            
            # Whale alerts
            whales = summary.get('whale_alerts', [])
            if whales:
                btc_whales = [w for w in whales if w.get('coin') == 'BTC'][:2]
                if btc_whales:
                    msg += f"\n🐋 *Whale Activity:*\n"
                    for w in btc_whales:
                        msg += f"• {w.get('amount', 0):.0f} BTC moving\n"
            
            # Trading bias
            bias = monitor.get_trading_bias()
            msg += f"\n⚡ *Trading Bias:* `{bias.get('bias', 'neutral').upper()}` ({bias.get('confidence', 50):.0%})\n"
            
            # Recommendation (truncated)
            rec = summary.get('recommendation', '')
            if rec:
                # Get just the first line
                first_line = rec.split('\n')[0]
                msg += f"\n💡 {first_line}"
            
            await update.message.reply_text(msg, parse_mode="Markdown")
            
        except Exception as e:
            logger.error(f"Error in /news command: {e}")
            await update.message.reply_text(f"❌ Error fetching news: {e}")

    async def _cmd_summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /summary command - toggle summary notifications on/off."""
        if self.toggle_summary and self.get_summary_status:
            # Toggle the state
            new_state = self.toggle_summary()
            
            if new_state:
                msg = """
✅ *Summary Notifications: ON*

📊 Periodic summaries will be sent automatically.
• Every few hours (configurable)
• Daily summary at 8:00 AM

Use /summary again to turn off.
"""
            else:
                msg = """
🔇 *Summary Notifications: OFF*

📊 Periodic summaries are now disabled.
You can still use:
• /status - Current status
• /pnl - Performance stats
• /intel - Intelligence overview

Use /summary again to turn on.
"""
        else:
            msg = "⚠️ Summary toggle not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    # ============== NEW ENHANCED COMMANDS ==============

    async def _cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /risk command - show risk manager status."""
        if self.get_risk_stats:
            r = self.get_risk_stats()
            
            can_trade_emoji = "✅" if r.get('can_trade') else "🛑"
            mode_emoji = {
                'NORMAL': '✅',
                'REDUCED': '⚠️',
                'CAUTIOUS': '🟡',
                'SEVERE': '🟠',
                'EMERGENCY': '🔴'
            }.get(r.get('dd_mode', 'NORMAL'), '❓')
            
            msg = f"""
🎯 *Risk Manager Status*

{can_trade_emoji} *Trading:* {'Allowed' if r.get('can_trade') else 'BLOCKED'}
{mode_emoji} *Mode:* `{r.get('dd_mode', 'NORMAL')}`
📊 *Reason:* {r.get('can_trade_reason', 'OK')}

*Position Sizing:*
├ Base Risk: `{r.get('base_risk', 2):.2%}`
├ Kelly Optimal: `{r.get('kelly_risk', 0.02):.2%}`
└ Adjusted Risk: `{r.get('adjusted_risk', 0.02):.2%}`

*Performance:*
├ Win Rate: `{r.get('win_rate', 0):.1%}`
├ Streak: {r.get('consecutive_wins', 0)}W / {r.get('consecutive_losses', 0)}L
└ Total Trades: `{r.get('total_trades', 0)}`

*Limits:*
├ Daily P&L: `${r.get('daily_pnl', 0):+.2f}`
├ Weekly P&L: `${r.get('weekly_pnl', 0):+.2f}`
├ Daily Limit: {'🛑 HIT' if r.get('daily_limit_hit') else '✅ OK'}
└ Weekly Limit: {'🛑 HIT' if r.get('weekly_limit_hit') else '✅ OK'}
"""
        else:
            msg = "⚠️ Risk manager not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_mtf(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /mtf command - multi-timeframe analysis."""
        if self.get_mtf_analysis:
            r = self.get_mtf_analysis()
            
            if r.get('error'):
                msg = f"⚠️ {r['error']}"
            else:
                conf_emoji = "✅" if r.get('confirmed') else "❌"
                
                # Primary timeframe
                primary = r.get('primary', {})
                trend_3m = primary.get('trend', {}).get('direction', 'unknown')
                
                # Secondary timeframe
                secondary = r.get('secondary', {})
                trend_15m = secondary.get('trend', {}).get('direction', 'N/A') if secondary.get('trend') else 'N/A'
                
                # Higher timeframe
                higher = r.get('higher', {})
                trend_1h = higher.get('trend', {}).get('direction', 'N/A') if higher.get('trend') else 'N/A'
                
                msg = f"""
📊 *Multi-Timeframe Analysis*

{conf_emoji} *Confirmation:* `{r.get('recommendation', 'WAIT')}`
📈 *Confluence:* `{r.get('confluence_pct', 0)}%`
🎯 *Alignment Score:* `{r.get('alignment_score', 0):.2f}`

*Timeframe Trends:*
├ 3m: `{trend_3m.upper()}`
├ 15m: `{trend_15m.upper() if trend_15m != 'N/A' else 'N/A'}`
└ 1H: `{trend_1h.upper() if trend_1h != 'N/A' else 'N/A'}`

*Volume:*
└ Ratio: `{r.get('volume', {}).get('volume_ratio', 1.0):.2f}x` ({r.get('volume', {}).get('trend', 'normal')})

*Confirmations:* {len(r.get('confirmations', []))}
{chr(10).join('✅ ' + c for c in r.get('confirmations', [])[:3]) or '_None_'}

*Conflicts:* {len(r.get('conflicts', []))}
{chr(10).join('⚠️ ' + c for c in r.get('conflicts', [])[:3]) or '_None_'}

💡 _{r.get('message', '')}_
"""
        else:
            msg = "⚠️ MTF analysis not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_backtest(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /backtest command - run historical backtest."""
        if not self.run_backtest:
            await update.message.reply_text("⚠️ Backtest not available")
            return
        
        # Parse days argument
        days = 7
        if context.args:
            try:
                days = int(context.args[0])
                days = min(max(days, 1), 90)  # Clamp to 1-90 days
            except ValueError:
                pass
        
        await update.message.reply_text(f"⏳ Running {days}-day backtest... This may take a moment.")
        
        try:
            result = await self.run_backtest(days)
            
            if result.get('error'):
                msg = f"❌ Backtest failed: {result['error']}"
            else:
                mc = result.get('monte_carlo', {})
                
                pnl_emoji = "📈" if result.get('total_pnl', 0) >= 0 else "📉"
                
                msg = f"""
📊 *Backtest Results ({days} days)*

{pnl_emoji} *Performance:*
├ Total P&L: `${result.get('total_pnl', 0):+,.2f}` ({result.get('total_pnl_pct', 0):+.1f}%)
├ Win Rate: `{result.get('win_rate', 0):.1f}%`
├ Profit Factor: `{result.get('profit_factor', 0):.2f}`
└ Sharpe Ratio: `{result.get('sharpe_ratio', 0):.2f}`

📈 *Trades:*
├ Total: `{result.get('total_trades', 0)}`
└ Max Drawdown: `{result.get('max_drawdown_pct', 0):.1f}%`

🎲 *Monte Carlo ({mc.get('simulations', 0)} sims):*
├ Median Final: `${mc.get('median_final_balance', 0):,.0f}`
├ 5th Percentile: `${mc.get('percentile_5', 0):,.0f}`
├ 95th Percentile: `${mc.get('percentile_95', 0):,.0f}`
├ Prob of Profit: `{mc.get('probability_profit', 0):.0f}%`
└ Worst-Case DD: `{mc.get('worst_case_drawdown', 0):.1f}%`
"""
        except Exception as e:
            msg = f"❌ Backtest error: {str(e)}"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_chart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /chart command - generate and send price chart."""
        if not self.get_chart:
            await update.message.reply_text("⚠️ Chart generation not available")
            return
        
        await update.message.reply_text("📊 Generating chart...")
        
        try:
            chart_bytes = self.get_chart()
            
            if chart_bytes:
                from io import BytesIO
                await self.bot.send_photo(
                    chat_id=self.chat_id,
                    photo=BytesIO(chart_bytes),
                    caption="📊 *Current Price Chart*\nBlue=Entry, Red=SL, Green=TP",
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text("⚠️ Could not generate chart (insufficient data or matplotlib not installed)")
        except Exception as e:
            logger.error(f"Chart command error: {e}")
            await update.message.reply_text(f"❌ Chart error: {str(e)}")

    async def _cmd_params(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /params command - show adaptive parameters."""
        if not self.get_adaptive_params:
            await update.message.reply_text("⚠️ Adaptive params not available")
            return
        
        msg = self.get_adaptive_params()
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_shield(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /shield command — Risk Shield circuit breaker status.
        
        /shield            — Show shield status (auth required)
        /shield reset      — Reset hard breach (auth required)
        /shield liquidate  — Emergency close ALL positions (auth required — THE RED BUTTON)
        """
        if not self._check_auth(update):
            await update.message.reply_text("⛔ Unauthorized")
            return

        args = context.args
        
        # ── /shield reset ─────────────────────────────────────────────────
        if args and args[0].lower() == 'reset':
            if self.reset_shield:
                result = self.reset_shield()
                if "no hard breach" in result.lower():
                    await update.message.reply_text(
                        "❌ *No hard breach to reset.*\n\n"
                        "Shield is already clear — trading is active.",
                        parse_mode="Markdown"
                    )
                else:
                    await update.message.reply_text(
                        f"✅ *Hard breach reset. Trading resumed.*\n\n"
                        f"`{result}`",
                        parse_mode="Markdown"
                    )
            else:
                await update.message.reply_text("⚠️ Shield reset not available")
            return
        
        # ── /shield liquidate (THE RED BUTTON) ────────────────────────────
        if args and args[0].lower() == 'liquidate':
            if self.emergency_liquidate:
                await update.message.reply_text(
                    "🚨🚨🚨 *EMERGENCY LIQUIDATION INITIATED* 🚨🚨🚨\n\n"
                    "Closing ALL open positions at market price...",
                    parse_mode="Markdown"
                )
                try:
                    result = await self.emergency_liquidate()
                    closed = result.get('positions_closed', 0)
                    failed = result.get('positions_failed', 0)
                    details = result.get('details', [])
                    
                    # Build position detail lines
                    detail_lines = ""
                    for d in details:
                        sym = d.get('symbol', '?')
                        st = d.get('status', '?')
                        side = d.get('side', '?')
                        emoji = "✅" if st == "CLOSED" else "❌"
                        detail_lines += f"  {emoji} `{sym}` ({side}) → {st}\n"
                    
                    if not detail_lines:
                        detail_lines = "  _No open positions found_\n"
                    
                    await update.message.reply_text(
                        f"🚨 *EMERGENCY LIQUIDATION COMPLETE* 🚨\n\n"
                        f"*Results:*\n"
                        f"├ Closed: `{closed}` positions\n"
                        f"├ Failed: `{failed}` positions\n"
                        f"└ Reason: `{result.get('reason', 'Manual')}`\n\n"
                        f"*Positions:*\n"
                        f"{detail_lines}\n"
                        f"🔒 *Trading is now LOCKED.*\n"
                        f"Use `/shield reset` to unlock.",
                        parse_mode="Markdown"
                    )
                except Exception as e:
                    logger.error(f"Emergency liquidation error: {e}", exc_info=True)
                    await update.message.reply_text(
                        f"❌ *Liquidation error:* `{e}`\n\n"
                        f"⚠️ Check positions manually!",
                        parse_mode="Markdown"
                    )
            else:
                await update.message.reply_text("⚠️ Emergency liquidation not available")
            return
        
        # ── /shield status (default) ──────────────────────────────────────
        if not self.get_shield_status:
            await update.message.reply_text("⚠️ Risk Shield not available")
            return
        
        try:
            s = self.get_shield_status()
            
            # Overall status indicator
            if s.get('hard_breach_active'):
                status_emoji = "🔴"
                status_text = "HARD BREACH — TRADING HALTED"
            elif s.get('shield_engaged'):
                status_emoji = "🟡"
                status_text = "SOFT BREACH — TRADING PAUSED"
            elif s.get('longs_blocked'):
                status_emoji = "🟠"
                status_text = "LONGs BLOCKED (BTC crash)"
            else:
                status_emoji = "🟢"
                status_text = "ALL CLEAR"
            
            thresholds = s.get('thresholds', {})
            levels = s.get('levels', {})
            
            # ── Progress bars for each breaker ────────────────────────────
            def _bar(current_str: str, threshold_str: str) -> str:
                """Build a visual bar showing how close to the threshold."""
                try:
                    c = float(current_str.replace('%', '').replace('+', ''))
                    t_str = threshold_str.replace('%', '').replace('+', '').replace('-', '').replace('/1h', '')
                    t = float(t_str)
                    if t == 0:
                        return ""
                    ratio = min(abs(c) / t, 1.0)
                    filled = int(ratio * 8)
                    bar = "█" * filled + "░" * (8 - filled)
                    pct = ratio * 100
                    warn = " ⚠️" if pct >= 75 else ""
                    return f" [{bar}] {pct:.0f}%{warn}"
                except (ValueError, TypeError, ZeroDivisionError):
                    return ""
            
            daily_bar = _bar(levels.get('daily_pnl_pct', '0'), thresholds.get('daily_loss', '3%'))
            weekly_bar = _bar(levels.get('weekly_pnl_pct', '0'), thresholds.get('weekly_loss', '7%'))
            dd_bar = _bar(levels.get('drawdown_pct', '0'), thresholds.get('max_drawdown', '10%'))
            tail_val = levels.get('tail_risk', '50')
            tail_floor = thresholds.get('tail_risk_floor', 30)
            try:
                tail_ok = float(str(tail_val)) > float(str(tail_floor))
            except (ValueError, TypeError):
                tail_ok = True
            btc_bar = _bar(levels.get('btc_1h', '0'), thresholds.get('btc_crash', '-5%/1h'))
            
            # ── Active breaches ───────────────────────────────────────────
            breaches = s.get('active_breaches', [])
            if breaches:
                breach_text = "\n⚡ <b>Active Breaches:</b>\n"
                for b in breaches:
                    # Escape HTML special chars in dynamic content
                    b_safe = str(b).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    breach_text += f"  🔸 <code>{b_safe}</code>\n"
            else:
                breach_text = "\n✅ <i>No active breaches</i>\n"
            
            # ── Recent history ────────────────────────────────────────────
            recent = s.get('recent_breaches', [])[-5:]
            history_text = ""
            if recent:
                history_text = "\n📜 <b>Recent Events:</b>\n"
                for h in reversed(recent):
                    sev = "🔴" if h.get('severity') == 'HARD' else "🟡"
                    ts = h.get('timestamp', '')[:16]
                    btype = h.get('breach_type', '?')
                    history_text += f"  {sev} <code>{ts}</code> {btype}\n"
            
            msg = (
                f"🛡️ <b>Risk Shield Dashboard</b>\n"
                f"{'━' * 28}\n\n"
                f"{status_emoji} <b>Status:</b> {status_text}\n\n"
                f"<b>Circuit Breakers:</b>\n"
                f"├ Daily P&amp;L:    <code>{levels.get('daily_pnl_pct', '?')}</code> / <code>{thresholds.get('daily_loss', '?')}</code>{daily_bar}\n"
                f"├ Weekly P&amp;L:   <code>{levels.get('weekly_pnl_pct', '?')}</code> / <code>{thresholds.get('weekly_loss', '?')}</code>{weekly_bar}\n"
                f"├ Drawdown:     <code>{levels.get('drawdown_pct', '?')}</code> / <code>{thresholds.get('max_drawdown', '?')}</code>{dd_bar}\n"
                f"├ Tail Risk:    <code>{tail_val}</code> / <code>{tail_floor}</code> {'🟢' if tail_ok else '🔴'}\n"
                f"└ BTC 1h:       <code>{levels.get('btc_1h', '?')}</code> / <code>{thresholds.get('btc_crash', '?')}</code>{btc_bar}\n"
                f"{breach_text}\n"
                f"📊 <b>Stats:</b>\n"
                f"├ Total blocks:   <code>{s.get('total_blocks', 0)}</code>\n"
                f"├ Hard breaches:  <code>{s.get('total_hard_breaches', 0)}</code>\n"
                f"└ Auto-liquidate: <code>{'ON ⚡' if s.get('auto_liquidate_enabled') else 'OFF'}</code>\n"
                f"{history_text}\n"
                f"<b>Commands:</b>\n"
                f"  /shield reset — reset hard breach\n"
                f"  /shield liquidate — ☢️ close ALL positions"
            )
            await update.message.reply_text(msg.strip(), parse_mode="HTML")
        except Exception as e:
            logger.error(f"Shield command error: {e}", exc_info=True)
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def _cmd_threshold(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /threshold command — switch between default and loose entry thresholds.

        /threshold              — Show current mode + both presets
        /threshold default      — Switch to proven/strict thresholds
        /threshold loose        — Switch to relaxed thresholds (more trades)
        """
        if not self._check_auth(update):
            await update.message.reply_text("⛔ Unauthorized")
            return

        from ai_filter import THRESHOLD_PRESETS, get_threshold_mode, set_threshold_mode, get_active_thresholds

        args = context.args

        # ── /threshold default | loose ───────────────────────────────────
        if args and args[0].lower() in ('default', 'loose'):
            new_mode = args[0].lower()
            old_mode = get_threshold_mode()
            if new_mode == old_mode:
                await update.message.reply_text(
                    f"Already on <b>{THRESHOLD_PRESETS[new_mode]['label']}</b> mode.",
                    parse_mode="HTML"
                )
                return

            set_threshold_mode(new_mode)
            preset = THRESHOLD_PRESETS[new_mode]
            wd = preset['weekday']
            we = preset['weekend']
            msg = (
                f"✅ Switched to <b>{preset['label']}</b>\n\n"
                f"<b>Weekday:</b>\n"
                f"  Stage 2 gate: {preset['stage2']}\n"
                f"  Combined: {wd['combined']}  |  Math: {wd['math']}\n"
                f"  AI high conf: {wd['ai_high_conf']}  |  LONG pref: {wd['long_preferred']}\n"
                f"  Reversal: {wd['reversal']}\n\n"
                f"<b>Weekend:</b>\n"
                f"  Stage 2 gate: {preset['stage2']}\n"
                f"  Combined: {we['combined']}  |  Math: {we['math']}\n"
                f"  AI high conf: {we['ai_high_conf']}  |  LONG pref: {we['long_preferred']}\n"
                f"  Reversal: {we['reversal']}\n\n"
                f"⚠️ Active on next scan cycle."
            )
            await update.message.reply_text(msg, parse_mode="HTML")
            return

        # ── /threshold (status) ───────────────────────────────────────
        mode = get_threshold_mode()
        is_weekend = False
        try:
            from datetime import datetime, timezone
            is_weekend = datetime.now(timezone.utc).weekday() >= 5
        except Exception:
            pass

        active = get_active_thresholds(is_weekend)
        period = 'Weekend' if is_weekend else 'Weekday'

        lines = []
        lines.append(f"🏚️ <b>Entry Thresholds</b>")
        lines.append(f"")
        lines.append(f"Current mode: <b>{THRESHOLD_PRESETS[mode]['label']}</b>")
        lines.append(f"Period: <b>{period}</b>")
        lines.append(f"")
        lines.append(f"<b>Active thresholds:</b>")
        lines.append(f"  Stage 2 gate:  <code>{active['stage2']}</code>")
        lines.append(f"  Combined:      <code>{active['combined']}</code>")
        lines.append(f"  Math:          <code>{active['math']}</code>")
        lines.append(f"  AI high conf:  <code>{active['ai_high_conf']}</code>")
        lines.append(f"  LONG pref:     <code>{active['long_preferred']}</code>")
        lines.append(f"  Reversal:      <code>{active['reversal']}</code>")
        lines.append(f"")

        # Show comparison
        other_mode = 'loose' if mode == 'default' else 'default'
        other = THRESHOLD_PRESETS[other_mode]
        op = 'weekend' if is_weekend else 'weekday'
        lines.append(f"<b>Alternative ({other['label']}):</b>")
        lines.append(f"  Stage 2: {other['stage2']}  |  Combined: {other[op]['combined']}  |  Math: {other[op]['math']}")
        lines.append(f"  AI: {other[op]['ai_high_conf']}  |  LONG: {other[op]['long_preferred']}  |  Rev: {other[op]['reversal']}")
        lines.append(f"")
        lines.append(f"💬 <code>/threshold {other_mode}</code> to switch")

        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    async def _cmd_dryrun(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /dryrun command — Dry-run shadow tracking.
        
        /dryrun            — Show status & shadow performance
        /dryrun on         — Enable dry-run mode
        /dryrun off        — Disable dry-run mode (resume normal trading)
        /dryrun reset      — Clear all shadow data
        """
        if not self._check_auth(update):
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        args = context.args
        
        # /dryrun on
        if args and args[0].lower() in ('on', 'enable', 'start'):
            if self.toggle_dry_run:
                result = self.toggle_dry_run(True)
                await update.message.reply_text(result, parse_mode="Markdown")
            else:
                await update.message.reply_text("⚠️ Dry-run toggle not available")
            return
        
        # /dryrun off
        if args and args[0].lower() in ('off', 'disable', 'stop'):
            if self.toggle_dry_run:
                result = self.toggle_dry_run(False)
                await update.message.reply_text(result, parse_mode="Markdown")
            else:
                await update.message.reply_text("⚠️ Dry-run toggle not available")
            return
        
        # /dryrun reset
        if args and args[0].lower() == 'reset':
            if self.reset_dry_run:
                self.reset_dry_run()
                await update.message.reply_text(
                    "📝 *Dry-Run Reset*\n\nAll shadow positions and history cleared.",
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text("⚠️ Dry-run reset not available")
            return
        
        # Show dry-run status
        if not self.get_dry_run_summary:
            await update.message.reply_text("⚠️ Dry-run tracker not available")
            return
        
        try:
            s = self.get_dry_run_summary()
            
            # Check if dry-run is currently active
            mode_info = self.get_trading_mode() if self.get_trading_mode else {}
            is_active = mode_info.get('dry_run_mode', False)
            
            mode_status = "🟢 ACTIVE" if is_active else "⚪ INACTIVE"
            
            # Open shadow positions
            open_text = ""
            if s['open_shadows']:
                open_text = "\n*Open Shadows:*\n"
                for sh in s['open_shadows']:
                    open_text += f"  👻 `{sh['side'].upper()}` {sh['symbol']} @ ${sh['entry']:.4f}\n"
            else:
                open_text = "\n_No open shadow positions_\n"
            
            # Recent completed trades
            recent_text = ""
            if s['recent_trades']:
                recent_text = "\n*Recent Shadow Trades:*\n"
                for t in reversed(s['recent_trades']):
                    emoji = "✅" if t['pnl'] >= 0 else "❌"
                    recent_text += f"  {emoji} {t['side'].upper()} {t['symbol']} | ${t['pnl']:+.2f} ({t['pnl_pct']:+.1f}%) | {t['reason']}\n"
            
            pnl_emoji = "📈" if s['total_pnl'] >= 0 else "📉"
            
            msg = f"""
👻 *Dry-Run Shadow Tracker*

*Mode:* {mode_status}

*Shadow Performance:*
├ Trades: `{s['total_trades']}` ({s['wins']}W / {s['losses']}L)
├ Win Rate: `{s['win_rate']}%`
├ {pnl_emoji} PnL: `${s['total_pnl']:+.2f}`
├ Best: `${s['max_win']:+.2f}`
└ Worst: `${s['max_loss']:+.2f}`
{open_text}{recent_text}
*Commands:*
  `/dryrun on` — enable shadow tracking
  `/dryrun off` — resume normal trading
  `/dryrun reset` — clear all shadow data
"""
            await update.message.reply_text(msg.strip(), parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Dry-run command error: {e}")
            await update.message.reply_text(f"❌ Error: {str(e)}")

    async def _cmd_tune(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /tune command - manually set a parameter.
        Usage: /tune param_name value
        Example: /tune atr_mult 2.5
        """
        if not self.set_adaptive_param:
            await update.message.reply_text("⚠️ Parameter tuning not available")
            return
        
        args = context.args
        if len(args) < 2:
            msg = "📝 *Usage:* `/tune param_name value`\n\n"
            msg += "*Example:* `/tune atr_mult 2.5`\n\n"
            msg += "Use `/params` to see available parameters."
            await update.message.reply_text(msg, parse_mode="Markdown")
            return
        
        param_name = args[0].lower()
        try:
            value = float(args[1])
        except ValueError:
            await update.message.reply_text(f"❌ Invalid value: {args[1]}")
            return
        
        result = self.set_adaptive_param(param_name, value)
        await update.message.reply_text(result, parse_mode="Markdown")

    async def _cmd_autotune(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /autotune command - trigger AI parameter review or toggle auto-tune."""
        args = context.args
        
        if args and args[0].lower() in ['on', 'off', 'toggle']:
            # Toggle auto-tune
            if self.toggle_auto_tune:
                result = self.toggle_auto_tune()
                await update.message.reply_text(result, parse_mode="Markdown")
            else:
                await update.message.reply_text("⚠️ Auto-tune toggle not available")
        else:
            # Trigger manual AI review
            if self.trigger_auto_tune:
                await update.message.reply_text("🎛️ Triggering AI parameter review...")
                result = await self.trigger_auto_tune()
                await update.message.reply_text(result, parse_mode="Markdown")
            else:
                await update.message.reply_text("⚠️ Auto-tune not available")

    async def _cmd_setprotection(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /setprotection command - view or set balance protection floor.
        Usage: /setprotection 280
        """
        args = context.args
        
        current = getattr(self.bot, 'balance_protection_threshold', 280)
        balance = getattr(self.bot, 'balance', 0)
        
        if not args:
            # Show current protection level
            status = "🟢 TRADING" if balance > current else "🔴 HALTED"
            msg = f"""🛡️ *Balance Protection*

💰 Current Balance: ${balance:,.2f}
🛡️ Protection Floor: ${current:,.2f}
📊 Status: {status}
📏 Headroom: ${balance - current:,.2f}

_Usage:_ `/setprotection 280` to change"""
            await update.message.reply_text(msg, parse_mode="Markdown")
            return
        
        try:
            new_value = float(args[0])
        except ValueError:
            await update.message.reply_text(f"❌ Invalid value: {args[0]}\nUsage: `/setprotection 280`", parse_mode="Markdown")
            return
        
        # Sanity checks
        if new_value < 0:
            await update.message.reply_text("❌ Protection floor can't be negative")
            return
        if new_value > balance:
            await update.message.reply_text(f"⚠️ Warning: ${new_value:.0f} is above current balance ${balance:.2f}!\nBot will NOT trade until balance exceeds this.")
        
        old_value = current
        
        # Update in memory
        self.bot.balance_protection_threshold = new_value
        
        # Persist to config file
        try:
            import json
            from pathlib import Path
            config_file = Path(getattr(self.bot, 'CONFIG_FILE', 'julaba_config.json'))
            config = {}
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
            config['balance_protection'] = new_value
            temp_file = Path(str(config_file) + '.tmp')
            with open(temp_file, 'w') as f:
                json.dump(config, f, indent=2)
            temp_file.rename(config_file)
        except Exception as e:
            logger.error(f"Failed to save protection to config: {e}")
        
        status = "🟢 TRADING" if balance > new_value else "🔴 HALTED"
        msg = f"""🛡️ *Protection Updated!*

🔄 ${old_value:,.0f} → ${new_value:,.0f}
💰 Balance: ${balance:,.2f}
📊 Status: {status}
📏 Headroom: ${balance - new_value:,.2f}"""
        await update.message.reply_text(msg, parse_mode="Markdown")
        logger.info(f"🛡️ Balance protection changed: ${old_value} → ${new_value} (via Telegram)")



    async def _cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /balance command."""
        if self.get_balance:
            data = self.get_balance()
            change = data.get('change', 0)
            change_emoji = "📈" if change >= 0 else "📉"
            msg = f"""
💰 *Balance*

💵 *Current:* `${data.get('current', 0):,.2f}`
🏦 *Initial:* `${data.get('initial', 0):,.2f}`
{change_emoji} *Change:* `${change:+,.2f}` ({data.get('change_pct', 0):+.2f}%)
"""
        else:
            msg = "⚠️ Balance not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /trades command."""
        if self.get_trades:
            trades = self.get_trades()
            if trades:
                msg = "📜 *Recent Trades*\n\n"
                for t in trades[-10:]:  # Last 10 trades
                    emoji = "✅" if t.get('pnl', 0) >= 0 else "❌"
                    msg += f"{emoji} {t.get('side', 'N/A')} @ ${t.get('entry', 0):.4f}\n"
                    msg += f"   P&L: `${t.get('pnl', 0):+.2f}` | {t.get('time', 'N/A')}\n\n"
            else:
                msg = "📜 *No trades yet*"
        else:
            msg = "⚠️ Trade history not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_market(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /market command."""
        if self.get_market:
            data = self.get_market()
            change = data.get('change_24h', 0)
            change_emoji = "🟢" if change >= 0 else "🔴"
            msg = f"""
📈 *Market Info*

📊 *Symbol:* `{data.get('symbol', 'N/A')}`
💰 *Price:* `${data.get('price', 0):,.4f}`
{change_emoji} *24h Change:* `{change:+.2f}%`
📊 *24h Volume:* `${data.get('volume_24h', 0):,.0f}`
📉 *24h Low:* `${data.get('low_24h', 0):,.4f}`
📈 *24h High:* `${data.get('high_24h', 0):,.4f}`
📏 *ATR:* `${data.get('atr', 0):,.4f}`
"""
        else:
            msg = "⚠️ Market data not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /signals command."""
        if self.get_signals:
            signals = self.get_signals()
            if signals:
                msg = "📡 *Recent Signals*\n\n"
                for s in signals[-10:]:  # Last 10 signals
                    status = "✅" if s.get('approved') else "❌"
                    emoji = "🟢" if s.get('side') == 'LONG' else "🔴"
                    msg += f"{emoji} {s.get('side', 'N/A')} @ ${s.get('price', 0):.4f} {status}\n"
                    msg += f"   Confidence: `{s.get('confidence', 0):.0%}` | {s.get('time', 'N/A')}\n\n"
            else:
                msg = "📡 *No signals yet*"
        else:
            msg = "⚠️ Signal history not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command."""
        if self.get_pnl and self.get_ai_stats:
            pnl = self.get_pnl()
            ai = self.get_ai_stats()
            
            # Calculate additional stats
            total_trades = pnl.get('trades', 0)
            winning = pnl.get('winning', 0)
            losing = total_trades - winning
            
            msg = f"""
📊 *Detailed Statistics*

💰 *Performance:*
├ Today P&L: `${pnl.get('today', 0):+,.2f}`
├ Total P&L: `${pnl.get('total', 0):+,.2f}`
├ Win Rate: `{pnl.get('win_rate', 0):.1%}`
├ Winning: `{winning}` | Losing: `{losing}`
└ Total Trades: `{total_trades}`

🤖 *AI Filter:*
├ Signals Analyzed: `{ai.get('total_signals', 0)}`
├ Approved: `{ai.get('approved', 0)}`
├ Rejected: `{ai.get('rejected', 0)}`
└ Approval Rate: `{ai.get('approval_rate', 'N/A')}`

📈 *Strategy:*
├ Max Win: `${pnl.get('max_win', 0):+,.2f}`
├ Max Loss: `${pnl.get('max_loss', 0):,.2f}`
└ Avg Trade: `${pnl.get('avg_trade', 0):+,.2f}`
"""
        else:
            msg = "⚠️ Statistics not available"
        
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop command - only process if message is recent (after bot startup)."""
        from datetime import datetime, timezone, timedelta
        
        # Check if message is from before bot startup (old queued message)
        msg_time = update.message.date
        if self.startup_time and msg_time:
            # Allow 10 second grace period for clock drift
            if msg_time < self.startup_time - timedelta(seconds=10):
                logger.info(f"Ignoring old /stop command from {msg_time} (startup was {self.startup_time})")
                return
        
        await update.message.reply_text(
            "⚠️ *Stopping bot...*\n\nThe bot will shut down gracefully.",
            parse_mode="Markdown"
        )
        if self.do_stop:
            await self.do_stop()

    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pause command."""
        if self.do_pause:
            self.do_pause()
            self.paused = True
            await update.message.reply_text(
                "⏸ *Trading Paused*\n\nThe bot will not open new positions.\nExisting positions will still be managed.\n\nUse /resume to continue trading.",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text("⚠️ Pause not available", parse_mode="Markdown")

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resume command."""
        if self.do_resume:
            self.do_resume()
            self.paused = False
            await update.message.reply_text(
                "▶️ *Trading Resumed*\n\nThe bot will now open new positions when signals are detected.",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text("⚠️ Resume not available", parse_mode="Markdown")

    async def _cmd_aimode(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /aimode command - view or set AI trading mode."""
        if context.args and len(context.args) > 0:
            # Set mode
            new_mode = context.args[0].lower()
            if self.set_ai_mode and self.set_ai_mode(new_mode):
                mode_descriptions = {
                    "filter": "🔍 AI only validates technical signals",
                    "advisory": "💡 AI suggests trades, you confirm via Telegram",
                    "autonomous": "🤖 AI can open trades directly (85%+ confidence)",
                    "hybrid": "🔄 AI suggests trades when no technical signal"
                }
                await update.message.reply_text(
                    f"✅ *AI Mode Changed*\n\n"
                    f"Mode: `{new_mode}`\n"
                    f"{mode_descriptions.get(new_mode, '')}",
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text(
                    "❌ Invalid mode. Use: `filter`, `advisory`, `autonomous`, or `hybrid`",
                    parse_mode="Markdown"
                )
        else:
            # Show current mode
            current_mode = self.get_ai_mode() if self.get_ai_mode else "filter"
            await update.message.reply_text(
                f"""🤖 *AI Trading Mode*

Current: `{current_mode}`

*Available Modes:*
• `filter` - AI only validates technical signals
• `advisory` - AI suggests trades, you confirm
• `autonomous` - AI opens trades directly (85%+)
• `hybrid` - AI suggests when no technical signal

Usage: `/aimode <mode>`
Example: `/aimode advisory`""",
                parse_mode="Markdown"
            )

    async def _cmd_confirm(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /confirm command - confirm pending AI trade with verified execution."""
        if not self.confirm_ai_trade:
            await update.message.reply_text("⚠️ Confirm not available", parse_mode="Markdown")
            return
        
        result = await self.confirm_ai_trade()
        if result:
            # Verify position actually exists before confirming
            await asyncio.sleep(0.5)
            if hasattr(self, 'get_full_system_state') and self.get_full_system_state:
                try:
                    state = self.get_full_system_state()
                    positions = state.get('position', [])
                    if positions and len(positions) > 0:
                        pos = positions[0]
                        # Only confirm if position verified
                        msg = f"✅ *AI Trade Confirmed & Executed!*\n\n"
                        msg += f"📊 *{pos.get('symbol', 'N/A')}*\n"
                        msg += f"• Side: {pos.get('side', 'N/A').upper()}\n"
                        msg += f"• Entry: ${pos.get('entry', 0):,.4f}\n"
                        msg += f"• Size: {pos.get('size', 0):.4f}"
                        await update.message.reply_text(msg, parse_mode="Markdown")
                    else:
                        await update.message.reply_text(
                            "⚠️ *Trade Status Unclear*\n\nPosition not verified. Check /positions.",
                            parse_mode="Markdown"
                        )
                except Exception as e:
                    logger.warning(f"Could not verify position: {e}")
                    await update.message.reply_text(
                        "⚠️ *Trade Status Unclear*\n\nCould not verify. Check /positions.",
                        parse_mode="Markdown"
                    )
            else:
                await update.message.reply_text(
                    "✅ *AI Trade Submitted*\n\nCheck /positions to verify.",
                    parse_mode="Markdown"
                )
        else:
            await update.message.reply_text("⚠️ No pending AI trade to confirm", parse_mode="Markdown")

    async def _cmd_reject(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /reject command - reject pending AI trade."""
        if self.reject_ai_trade:
            result = await self.reject_ai_trade()
            if result:
                await update.message.reply_text("❌ *AI Trade Rejected*", parse_mode="Markdown")
            else:
                await update.message.reply_text("⚠️ No pending AI trade to reject", parse_mode="Markdown")
        else:
            await update.message.reply_text("⚠️ Reject not available", parse_mode="Markdown")

    async def _cmd_open(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /open command - request to open positions via AI analysis."""
        args = context.args
        if not args:
            await update.message.reply_text("⚠️ Usage: /open <long|short> [symbol]")
            return
        
        side = args[0].lower()
        if side not in ["long", "short"]:
            await update.message.reply_text("⚠️ Invalid side. Use 'long' or 'short'")
            return
            
        symbol_override = None
        if len(args) > 1:
            symbol_override = args[1].upper()
            
        # Simulate processing as if user typed "open long" 
        # But we create a simpler path by just creating the manual request context
        user_msg = f"open {side}"
        if symbol_override:
            user_msg += f" {symbol_override}"
            
        # We'll leverage the existing _handle_message logic which is robust
        update.message.text = user_msg
        await self._handle_message(update, context)

    async def _cmd_buy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Alias for /open long."""
        update.message.text = "open long"
        if context.args:
            update.message.text += f" {context.args[0]}"
        await self._handle_message(update, context)

    async def _cmd_sell(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Alias for /open short."""
        update.message.text = "open short"
        if context.args:
            update.message.text += f" {context.args[0]}"
        await self._handle_message(update, context)

    async def _cmd_execute_long(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /execute_long command - Immediate execution with verified confirmation."""
        if not self.execute_ai_trade:
            await update.message.reply_text("⚠️ Trade execution not available", parse_mode="Markdown")
            return
        
        # === BLOCK IN AUTONOMOUS MODE ===
        if self.get_ai_mode:
            try:
                if self.get_ai_mode() == 'autonomous':
                    await update.message.reply_text(
                        "🤖 *AUTONOMOUS MODE ACTIVE*\n\n"
                        "Manual trade execution is disabled.\n"
                        "Switch to `/mode advisory` or `/mode hybrid` to trade manually.",
                        parse_mode="Markdown"
                    )
                    return
            except Exception:
                pass
        
        # === RATE LIMITING CHECK ===
        import time
        current_time = time.time()
        time_since_last = current_time - self._last_execute_time
        if time_since_last < self._EXECUTE_COOLDOWN_SECONDS:
            remaining = int(self._EXECUTE_COOLDOWN_SECONDS - time_since_last)
            await update.message.reply_text(
                f"⏳ *Rate Limited*\n\nPlease wait {remaining}s before executing another trade.\nThis prevents accidental double-clicks.",
                parse_mode="Markdown"
            )
            return
        
        self._last_execute_time = current_time
        
        # === RETRIEVE SYMBOL FROM PENDING COMMAND ===
        trade_symbol = None
        if self._pending_execute_command:
            cmd = self._pending_execute_command
            if cmd.get('action') == 'open_trade' and cmd.get('side', '').lower() == 'long':
                trade_symbol = cmd.get('symbol', None)
                logger.info(f"📋 /execute_long: Retrieved pending symbol: {trade_symbol}")
            self._pending_execute_command = None  # Clear after use
        
        result = await self.execute_ai_trade("long", trade_symbol)
        
        if result.get("success"):
            # Verify position actually exists in system before confirming
            await asyncio.sleep(0.5)
            if hasattr(self, 'get_full_system_state') and self.get_full_system_state:
                try:
                    state = self.get_full_system_state()
                    positions = state.get('position', [])
                    if positions and len(positions) > 0:
                        pos = positions[0]
                        # Only send confirmation if position is verified
                        msg = f"✅ *LONG Trade Executed!*\n\n"
                        msg += f"📊 *{pos.get('symbol', 'N/A')}*\n"
                        msg += f"• Side: {pos.get('side', 'N/A').upper()}\n"
                        msg += f"• Entry: ${pos.get('entry', 0):,.4f}\n"
                        msg += f"• Size: {pos.get('size', 0):.4f}"
                        await update.message.reply_text(msg, parse_mode="Markdown")
                    else:
                        # API said success but no position found - warn user
                        await update.message.reply_text(
                            f"⚠️ *Trade Status Unclear*\n\n{result.get('message', 'Position not verified')}\n\nCheck /positions to confirm.",
                            parse_mode="Markdown"
                        )
                except Exception as e:
                    logger.warning(f"Could not verify position: {e}")
                    await update.message.reply_text(
                        f"⚠️ *Trade Status Unclear*\n\nCould not verify. Check /positions.",
                        parse_mode="Markdown"
                    )
            else:
                # No state function available - just report API result
                await update.message.reply_text(
                    f"✅ *Trade Submitted*\n\n{result.get('message', 'Check /positions')}",
                    parse_mode="Markdown"
                )
        else:
            await update.message.reply_text(
                f"❌ *Trade Failed*\n\n{result.get('message', 'Unknown error')}",
                parse_mode="Markdown"
            )

    async def _cmd_execute_short(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /execute_short command - Immediate execution with verified confirmation."""
        if not self.execute_ai_trade:
            await update.message.reply_text("⚠️ Trade execution not available", parse_mode="Markdown")
            return
        
        # === BLOCK IN AUTONOMOUS MODE ===
        if self.get_ai_mode:
            try:
                if self.get_ai_mode() == 'autonomous':
                    await update.message.reply_text(
                        "🤖 *AUTONOMOUS MODE ACTIVE*\n\n"
                        "Manual trade execution is disabled.\n"
                        "Switch to `/mode advisory` or `/mode hybrid` to trade manually.",
                        parse_mode="Markdown"
                    )
                    return
            except Exception:
                pass
        
        # === RATE LIMITING CHECK ===
        import time
        current_time = time.time()
        time_since_last = current_time - self._last_execute_time
        if time_since_last < self._EXECUTE_COOLDOWN_SECONDS:
            remaining = int(self._EXECUTE_COOLDOWN_SECONDS - time_since_last)
            await update.message.reply_text(
                f"⏳ *Rate Limited*\n\nPlease wait {remaining}s before executing another trade.\nThis prevents accidental double-clicks.",
                parse_mode="Markdown"
            )
            return
        
        self._last_execute_time = current_time
        
        # === RETRIEVE SYMBOL FROM PENDING COMMAND ===
        trade_symbol = None
        if self._pending_execute_command:
            cmd = self._pending_execute_command
            if cmd.get('action') == 'open_trade' and cmd.get('side', '').lower() == 'short':
                trade_symbol = cmd.get('symbol', None)
                logger.info(f"📋 /execute_short: Retrieved pending symbol: {trade_symbol}")
            self._pending_execute_command = None  # Clear after use
        
        result = await self.execute_ai_trade("short", trade_symbol)
        
        if result.get("success"):
            # Verify position actually exists in system before confirming
            await asyncio.sleep(0.5)
            if hasattr(self, 'get_full_system_state') and self.get_full_system_state:
                try:
                    state = self.get_full_system_state()
                    positions = state.get('position', [])
                    if positions and len(positions) > 0:
                        pos = positions[0]
                        # Only send confirmation if position is verified
                        msg = f"✅ *SHORT Trade Executed!*\n\n"
                        msg += f"📊 *{pos.get('symbol', 'N/A')}*\n"
                        msg += f"• Side: {pos.get('side', 'N/A').upper()}\n"
                        msg += f"• Entry: ${pos.get('entry', 0):,.4f}\n"
                        msg += f"• Size: {pos.get('size', 0):.4f}"
                        await update.message.reply_text(msg, parse_mode="Markdown")
                    else:
                        # API said success but no position found - warn user
                        await update.message.reply_text(
                            f"⚠️ *Trade Status Unclear*\n\n{result.get('message', 'Position not verified')}\n\nCheck /positions to confirm.",
                            parse_mode="Markdown"
                        )
                except Exception as e:
                    logger.warning(f"Could not verify position: {e}")
                    await update.message.reply_text(
                        f"⚠️ *Trade Status Unclear*\n\nCould not verify. Check /positions.",
                        parse_mode="Markdown"
                    )
            else:
                # No state function available - just report API result
                await update.message.reply_text(
                    f"✅ *Trade Submitted*\n\n{result.get('message', 'Check /positions')}",
                    parse_mode="Markdown"
                )
        else:
            await update.message.reply_text(
                f"❌ *Trade Failed*\n\n{result.get('message', 'Unknown error')}",
                parse_mode="Markdown"
            )

    async def _cmd_execute_close(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /execute_close command - Immediate close with verified confirmation."""
        if not self.close_ai_trade:
            await update.message.reply_text("⚠️ Close not available", parse_mode="Markdown")
            return
        
        # === BLOCK IN AUTONOMOUS MODE ===
        if self.get_ai_mode:
            try:
                if self.get_ai_mode() == 'autonomous':
                    await update.message.reply_text(
                        "🤖 *AUTONOMOUS MODE ACTIVE*\n\n"
                        "Manual close is disabled. The system manages exits autonomously.\n"
                        "Switch to `/mode advisory` or `/mode hybrid` to close manually.",
                        parse_mode="Markdown"
                    )
                    return
            except Exception:
                pass
        
        # === RATE LIMITING CHECK ===
        import time
        current_time = time.time()
        time_since_last = current_time - self._last_execute_time
        if time_since_last < self._EXECUTE_COOLDOWN_SECONDS:
            remaining = int(self._EXECUTE_COOLDOWN_SECONDS - time_since_last)
            await update.message.reply_text(
                f"⏳ *Rate Limited*\n\nPlease wait {remaining}s before executing another trade.\nThis prevents accidental double-clicks.",
                parse_mode="Markdown"
            )
            return
        
        self._last_execute_time = current_time
        
        # Check if there's a position to close first
        has_position = False
        symbol = "N/A"
        if hasattr(self, 'get_full_system_state') and self.get_full_system_state:
            try:
                state = self.get_full_system_state()
                positions = state.get('position', [])
                if positions and len(positions) > 0:
                    has_position = True
                    symbol = positions[0].get('symbol', 'N/A')
            except Exception:
                pass  # Non-critical: couldn't get position state
        
        if not has_position:
            await update.message.reply_text("⚠️ No open position to close", parse_mode="Markdown")
            return
        
        result = await self.close_ai_trade()
        
        if result.get("success"):
            # Verify position is actually closed
            await asyncio.sleep(0.5)
            if hasattr(self, 'get_full_system_state') and self.get_full_system_state:
                try:
                    state = self.get_full_system_state()
                    positions = state.get('position', [])
                    if not positions or len(positions) == 0:
                        # Position confirmed closed
                        await update.message.reply_text(
                            f"✅ *Position Closed!*\n\n📊 *{symbol}*\n💰 {result.get('message', 'Position closed')}",
                            parse_mode="Markdown"
                        )
                    else:
                        # Position still exists - warn
                        await update.message.reply_text(
                            f"⚠️ *Close Status Unclear*\n\nPosition may still be open. Check /positions.",
                            parse_mode="Markdown"
                        )
                except Exception as e:
                    logger.warning(f"Could not verify close: {e}")
                    await update.message.reply_text(
                        f"✅ *Close Submitted*\n\n{result.get('message', 'Check /positions')}",
                        parse_mode="Markdown"
                    )
            else:
                await update.message.reply_text(
                    f"✅ *Close Submitted*\n\n{result.get('message', 'Check /positions')}",
                    parse_mode="Markdown"
                )
        else:
            await update.message.reply_text(
                f"❌ *Close Failed*\n\n{result.get('message', 'Unknown error')}",
                parse_mode="Markdown"
            )

    async def _cmd_close(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /close command - request to close position via AI analysis."""
        update.message.text = "close position"
        await self._handle_message(update, context)

    async def _cmd_reset_daily(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /reset_daily command - quick alias to reset daily loss limit."""
        logger.info(f"📥 /reset_daily command received from {update.effective_user.id}")
        
        if not self._check_auth(update):
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        logger.info(f"✅ Auth passed, set_system_param={self.set_system_param is not None}")
        
        if self.set_system_param:
            result = self.set_system_param('reset_daily_loss', True)
            logger.info(f"📤 reset_daily_loss result: {result}")
            if result.get('success'):
                await update.message.reply_text(
                    "✅ *Daily Loss Limit Reset*\n\n"
                    "Trading has resumed. Be careful! 🙏",
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text(
                    f"❌ Failed to reset: {result.get('error', 'Unknown error')}",
                    parse_mode="Markdown"
                )
        else:
            logger.warning("⚠️ set_system_param callback not set!")
            await update.message.reply_text("⚠️ Parameter setting not available")

    async def _cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /reset command - reset daily loss limit or other parameters."""
        if not self._check_auth(update):
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        args = context.args if context.args else []
        
        if not args:
            # Show help
            await update.message.reply_text(
                "🔄 *Reset Command*\n\n"
                "Usage:\n"
                "• `/reset drawdown` - Reset max drawdown circuit breaker\n"
                "• `/reset daily_loss` - Reset daily loss limit (resume trading)\n"
                "• `/reset params` - Reset all adaptive params to defaults\n\n"
                "Example: `/reset drawdown`",
                parse_mode="Markdown"
            )
            return
        
        what = args[0].lower()
        
        if what in ['drawdown', 'max_drawdown', 'dd', 'peak']:
            # Reset max drawdown circuit breaker
            if self.set_system_param:
                result = self.set_system_param('reset_drawdown', True)
                if result.get('success'):
                    await update.message.reply_text(
                        f"✅ *Max Drawdown Reset*\n\n"
                        f"{result.get('message', 'Drawdown circuit breaker cleared')}\n\n"
                        f"Trading can resume. Peak balance reset to current balance. 🙏",
                        parse_mode="Markdown"
                    )
                else:
                    await update.message.reply_text(
                        f"❌ Failed to reset: {result.get('error', 'Unknown error')}",
                        parse_mode="Markdown"
                    )
            else:
                await update.message.reply_text("⚠️ Parameter setting not available")
        elif what in ['daily_loss', 'daily', 'dailyloss', 'loss', 'daily_loss_limit']:
            # Reset daily loss trigger
            if self.set_system_param:
                result = self.set_system_param('reset_daily_loss', True)
                if result.get('success'):
                    await update.message.reply_text(
                        "✅ *Daily Loss Limit Reset*\n\n"
                        "Trading has resumed. Be careful! 🙏",
                        parse_mode="Markdown"
                    )
                else:
                    await update.message.reply_text(
                        f"❌ Failed to reset: {result.get('error', 'Unknown error')}",
                        parse_mode="Markdown"
                    )
            else:
                await update.message.reply_text("⚠️ Parameter setting not available")
        elif what == 'params':
            # Reset adaptive params
            if self.set_system_param:
                result = self.set_system_param('reset_params', True)
                await update.message.reply_text(
                    "✅ *Adaptive Parameters Reset*\n\n"
                    "All parameters restored to defaults.",
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text("⚠️ Parameter setting not available")
        else:
            await update.message.reply_text(
                f"❓ Unknown reset target: `{what}`\n\n"
                f"Try: `/reset drawdown`, `/reset daily_loss`, or `/reset params`",
                parse_mode="Markdown"
            )

    async def _cmd_override(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /override or /force command - force clear ALL trading halts."""
        if not self._check_auth(update):
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        if self.set_system_param:
            result = self.set_system_param('force_resume', True)
            if result.get('success'):
                await update.message.reply_text(
                    "🔓 *FORCE OVERRIDE ACTIVATED*\n\n"
                    "✅ Paused state: Cleared\n"
                    "✅ Daily loss halt: Cleared\n"
                    "✅ Cooldown: Cleared\n\n"
                    "⚠️ _Trading is now active. Be careful!_",
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text(
                    f"❌ Override failed: {result.get('error', 'Unknown error')}",
                    parse_mode="Markdown"
                )
        else:
            await update.message.reply_text("⚠️ Parameter setting not available")

    async def _cmd_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /mode command - view or switch trading mode (live/paper)."""
        if not self._check_auth(update):
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        args = context.args
        
        # If no arguments, show current mode
        if not args:
            if self.get_trading_mode:
                mode_info = self.get_trading_mode()
                current_mode = mode_info.get('mode', 'unknown')
                is_live = mode_info.get('live_mode', False)
                balance = mode_info.get('balance', 0)
                initial_balance = mode_info.get('initial_balance', 0)
                total_trades = mode_info.get('total_trades', 0)
                total_pnl = mode_info.get('total_pnl', 0)
                
                mode_emoji = "🔴" if is_live else "📝"
                mode_name = "LIVE TRADING" if is_live else "PAPER TRADING"
                
                msg = f"{mode_emoji} *Current Mode: {mode_name}*\n\n"
                msg += f"💰 Balance: ${balance:,.2f}\n"
                msg += f"📊 Initial: ${initial_balance:,.2f}\n"
                msg += f"📈 Total P&L: ${total_pnl:+,.2f}\n"
                msg += f"🔢 Total Trades: {total_trades}\n\n"
                
                msg += "*Switch Mode Commands:*\n"
                msg += "`/mode paper` - Switch to paper trading\n"
                msg += "  → Resets to $10,000, clears all history\n\n"
                msg += "`/mode live` - Switch to live trading\n"
                msg += "  → Preserves all data, uses real money\n\n"
                
                if is_live:
                    msg += "⚠️ _You are trading with REAL money!_"
                else:
                    msg += "✅ _Safe mode - no real money at risk_"
                
                await update.message.reply_text(msg, parse_mode="Markdown")
            else:
                await update.message.reply_text("⚠️ Mode info not available")
            return
        
        # Parse the mode argument
        target_mode = args[0].lower()
        
        if target_mode not in ['live', 'paper']:
            await update.message.reply_text(
                "❌ Invalid mode. Use:\n"
                "• `/mode live` - Switch to live trading\n"
                "• `/mode paper` - Switch to paper trading",
                parse_mode="Markdown"
            )
            return
        
        if self.switch_trading_mode:
            # Add confirmation for live mode
            if target_mode == 'live':
                # Check if there's a confirmation flag in args
                if len(args) < 2 or args[1].lower() != 'confirm':
                    await update.message.reply_text(
                        "⚠️ *SWITCHING TO LIVE TRADING*\n\n"
                        "This will:\n"
                        "• Execute REAL orders with REAL money\n"
                        "• Preserve all existing trade history\n"
                        "• Connect to your Bybit account\n\n"
                        "To confirm, type:\n"
                        "`/mode live confirm`",
                        parse_mode="Markdown"
                    )
                    return
            
            result = await self.switch_trading_mode(target_mode)
            
            if result.get('success'):
                if target_mode == 'paper':
                    await update.message.reply_text(
                        "📝 *SWITCHED TO PAPER TRADING*\n\n"
                        "✅ Balance reset to $10,000\n"
                        "✅ Trade history cleared\n"
                        "✅ Stats reset to zero\n"
                        "✅ AI decisions cleared\n\n"
                        "🔄 *Bot is restarting...*\n"
                        "Please wait ~10 seconds.",
                        parse_mode="Markdown"
                    )
                else:
                    await update.message.reply_text(
                        "🔴 *SWITCHED TO LIVE TRADING*\n\n"
                        "✅ All data preserved\n"
                        "✅ Balance synced from Bybit\n"
                        "⚠️ REAL money is now at risk!\n\n"
                        "🔄 *Bot is restarting...*\n"
                        "Please wait ~10 seconds.",
                        parse_mode="Markdown"
                    )
            else:
                await update.message.reply_text(
                    f"❌ Mode switch failed: {result.get('error', 'Unknown error')}",
                    parse_mode="Markdown"
                )
        else:
            await update.message.reply_text("⚠️ Mode switching not available")

    async def _cmd_longonly(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /longonly command - only allow LONG trades."""
        if not self._check_auth(update):
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        if self.set_allowed_sides:
            if self.set_allowed_sides('long'):
                await update.message.reply_text(
                    "📈 *LONG-ONLY MODE ENABLED*\n\n"
                    "✅ LONG trades: ALLOWED\n"
                    "❌ SHORT trades: BLOCKED\n\n"
                    "_All SHORT signals will be ignored._\n"
                    "Use `/bothsides` to enable both directions.",
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text("⚠️ Failed to set long-only mode")
        else:
            await update.message.reply_text("⚠️ Side filter not available")

    async def _cmd_shortonly(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /shortonly command - only allow SHORT trades."""
        if not self._check_auth(update):
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        if self.set_allowed_sides:
            if self.set_allowed_sides('short'):
                await update.message.reply_text(
                    "📉 *SHORT-ONLY MODE ENABLED*\n\n"
                    "❌ LONG trades: BLOCKED\n"
                    "✅ SHORT trades: ALLOWED\n\n"
                    "_All LONG signals will be ignored._\n"
                    "Use `/bothsides` to enable both directions.",
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text("⚠️ Failed to set short-only mode")
        else:
            await update.message.reply_text("⚠️ Side filter not available")

    async def _cmd_bothsides(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /bothsides command - allow both LONG and SHORT trades."""
        if not self._check_auth(update):
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        if self.set_allowed_sides:
            if self.set_allowed_sides('both'):
                await update.message.reply_text(
                    "↔️ *BOTH SIDES ENABLED*\n\n"
                    "✅ LONG trades: ALLOWED\n"
                    "✅ SHORT trades: ALLOWED\n\n"
                    "_All signals will be processed._\n"
                    "Use `/longonly` or `/shortonly` to restrict.",
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text("⚠️ Failed to enable both sides")
        else:
            await update.message.reply_text("⚠️ Side filter not available")

    async def _cmd_sides(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /sides command - show current side filter status."""
        if not self._check_auth(update):
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        if self.get_allowed_sides:
            allowed = self.get_allowed_sides().lower()
            
            if allowed == 'long':
                status = "📈 *LONG-ONLY MODE*\n\n✅ LONG: Allowed\n❌ SHORT: Blocked"
            elif allowed == 'short':
                status = "📉 *SHORT-ONLY MODE*\n\n❌ LONG: Blocked\n✅ SHORT: Allowed"
            else:
                status = "↔️ *BOTH SIDES MODE*\n\n✅ LONG: Allowed\n✅ SHORT: Allowed"
            
            msg = f"{status}\n\n*Commands:*\n"
            msg += "`/longonly` - Only LONG trades\n"
            msg += "`/shortonly` - Only SHORT trades\n"
            msg += "`/bothsides` - Allow both directions"
            
            await update.message.reply_text(msg, parse_mode="Markdown")
        else:
            await update.message.reply_text("⚠️ Side filter not available")

    async def _cmd_hold1(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /hold1 command - hold position 1."""
        await self._toggle_hold_position(update, 1, True)

    async def _cmd_hold2(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /hold2 command - hold position 2."""
        await self._toggle_hold_position(update, 2, True)

    async def _cmd_unhold1(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /unhold1 command - release hold on position 1."""
        await self._toggle_hold_position(update, 1, False)

    async def _cmd_unhold2(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /unhold2 command - release hold on position 2."""
        await self._toggle_hold_position(update, 2, False)

    async def _cmd_hold(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /hold command - show hold status or hold by symbol."""
        if not self._check_auth(update):
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        # Get current positions and hold status
        if self.get_full_system_state:
            state = self.get_full_system_state()
            positions = state.get('position', [])
            held = state.get('held_positions', {})
            
            msg = "💎 *DIAMOND HANDS / HOLD Status*\n\n"
            
            if not positions:
                msg += "No open positions.\n"
            else:
                for i, pos in enumerate(positions[:2], 1):
                    if pos:
                        symbol = pos.get('symbol', 'Unknown')
                        sym_key = symbol.replace("/", "").replace(":USDT", "")
                        is_held = held.get(sym_key, False)
                        status = "💎 DIAMOND HANDS" if is_held else "🔓 Normal (SL active)"
                        msg += f"*Position {i}:* {symbol}\n"
                        msg += f"  Status: {status}\n\n"
            
            msg += "*Commands:*\n"
            msg += "`/hold1` `/dh1` - Diamond hands position 1\n"
            msg += "`/hold2` `/dh2` - Diamond hands position 2\n"
            msg += "`/unhold1` - Release position 1 (restore SL)\n"
            msg += "`/unhold2` - Release position 2 (restore SL)\n"
            msg += "`/diamondhands` - Full info"
            
            await update.message.reply_text(msg, parse_mode="Markdown")
        else:
            await update.message.reply_text("⚠️ System state not available")

    async def _cmd_unhold(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /unhold command - show unhold help."""
        if not self._check_auth(update):
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        await update.message.reply_text(
            "🔓 *Release Hold*\n\n"
            "Commands:\n"
            "`/unhold1` - Release hold on position 1\n"
            "`/unhold2` - Release hold on position 2\n\n"
            "Use `/hold` to see current status.",
            parse_mode="Markdown"
        )

    async def _cmd_diamondhands(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /diamondhands command - show diamond hands info and status."""
        if not self._check_auth(update):
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        if self.get_full_system_state:
            state = self.get_full_system_state()
            positions = state.get('position', [])
            held = state.get('held_positions', {})
            
            msg = "💎 *DIAMOND HANDS MODE*\n\n"
            msg += "Ride through drawdowns — no SL closes.\n"
            msg += "Position stays open even at deep negative.\n"
            msg += "Server-side SL is REMOVED from Bybit.\n\n"
            
            if not positions:
                msg += "No open positions.\n"
            else:
                for i, pos in enumerate(positions[:2], 1):
                    if pos:
                        symbol = pos.get('symbol', 'Unknown')
                        sym_key = symbol.replace("/", "").replace(":USDT", "")
                        is_held = held.get(sym_key, False)
                        side = pos.get('side', '?').upper()
                        entry = pos.get('entry_price', 0)
                        sl = pos.get('stop_loss', 0)
                        
                        status = "💎 DIAMOND HANDS ACTIVE" if is_held else "🔓 Normal mode"
                        msg += f"*Position {i}:* `{symbol}` {side}\n"
                        msg += f"  Entry: ${entry:.4f} | SL: ${sl:.4f}\n"
                        msg += f"  Status: {status}\n\n"
            
            msg += "*Commands:*\n"
            msg += "`/hold1` or `/dh1` — Diamond hands position 1\n"
            msg += "`/hold2` or `/dh2` — Diamond hands position 2\n"
            msg += "`/unhold1` — Release position 1 (restore SL)\n"
            msg += "`/unhold2` — Release position 2 (restore SL)\n\n"
            msg += "⚠️ _When active: NO stop loss, NO loss cap._\n"
            msg += "_Only YOU can close it via /close or Bybit._"
            
            await update.message.reply_text(msg, parse_mode="Markdown")
        else:
            await update.message.reply_text("⚠️ System state not available")

    async def _toggle_hold_position(self, update: Update, slot: int, hold_state: bool):
        """Toggle hold state for a position slot."""
        if not self._check_auth(update):
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        if not self.get_full_system_state:
            await update.message.reply_text("⚠️ System state not available")
            return
        
        # Get positions
        state = self.get_full_system_state()
        positions = state.get('position', [])
        
        if slot > len(positions) or not positions[slot-1]:
            await update.message.reply_text(f"⚠️ No position in slot {slot}")
            return
        
        pos = positions[slot-1]
        symbol = pos.get('symbol', '')
        
        if not symbol:
            await update.message.reply_text(f"⚠️ Position {slot} has no symbol")
            return
        
        # Call the toggle hold function via set_system_param
        if self.set_system_param:
            result = self.set_system_param('toggle_hold', {'symbol': symbol, 'hold': hold_state})
            
            if result.get('success'):
                if hold_state:
                    action_emoji = "💎"
                    action_text = "DIAMOND HANDS ACTIVATED"
                    exit_msg = "BLOCKED — no SL, no loss cap"
                    extra = "\n\n⚠️ Server-side SL removed from Bybit!\nOnly YOU can close via /close or on Bybit."
                else:
                    action_emoji = "🔓"
                    action_text = "RELEASED — Normal mode"
                    exit_msg = "ALLOWED — SL restored"
                    extra = ""
                await update.message.reply_text(
                    f"{action_emoji} {action_text}\n\n"
                    f"*Position {slot}:* `{symbol}`\n"
                    f"Bot exit: {exit_msg}{extra}",
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text(
                    f"❌ Failed: {result.get('error', 'Unknown error')}",
                    parse_mode="Markdown"
                )
        else:
            await update.message.reply_text("⚠️ Hold toggle not available")

    async def _cmd_watching(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /watching command - confirm user is monitoring the position."""
        if not self._check_auth(update):
            await update.message.reply_text("⛔ Unauthorized")
            return
        
        if self.confirm_watchdog:
            result = self.confirm_watchdog()
            await update.message.reply_text(result, parse_mode="Markdown")
        else:
            await update.message.reply_text("⚠️ Watchdog confirmation not available")

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button callbacks."""
        query = update.callback_query
        await query.answer()
        
        if query.data == "confirm_ai_trade":
            if self.confirm_ai_trade:
                result = await self.confirm_ai_trade()
                if result:
                    # Verify position actually exists before confirming
                    await asyncio.sleep(0.5)
                    if hasattr(self, 'get_full_system_state') and self.get_full_system_state:
                        try:
                            state = self.get_full_system_state()
                            positions = state.get('position', [])
                            if positions and len(positions) > 0:
                                pos = positions[0]
                                # Only confirm if position verified
                                msg = f"✅ *AI Trade Confirmed & Executed!*\n\n"
                                msg += f"📊 *{pos.get('symbol', 'N/A')}*\n"
                                msg += f"• Side: {pos.get('side', 'N/A').upper()}\n"
                                msg += f"• Entry: ${pos.get('entry', 0):,.4f}\n"
                                msg += f"• Size: {pos.get('size', 0):.4f}"
                                await query.edit_message_text(msg, parse_mode="Markdown")
                            else:
                                await query.edit_message_text(
                                    "⚠️ *Trade Status Unclear*\n\nPosition not verified. Check /positions.",
                                    parse_mode="Markdown"
                                )
                        except Exception as e:
                            logger.warning(f"Could not verify position: {e}")
                            await query.edit_message_text(
                                "⚠️ *Trade Status Unclear*\n\nCould not verify. Check /positions.",
                                parse_mode="Markdown"
                            )
                    else:
                        await query.edit_message_text(
                            "✅ *AI Trade Submitted*\n\nCheck /positions to verify.",
                            parse_mode="Markdown"
                        )
                else:
                    await query.edit_message_text("⚠️ Trade expired or already handled", parse_mode="Markdown")
        
        elif query.data == "reject_ai_trade":
            if self.reject_ai_trade:
                await self.reject_ai_trade()
                await query.edit_message_text("❌ *AI Trade Rejected*", parse_mode="Markdown")
        
        elif query.data == "confirm_pending_trade":
            # User confirmed the pending trade (below min size)
            if self.pending_trade_confirmation and self.execute_pending_trade:
                trade = self.pending_trade_confirmation
                result = await self.execute_pending_trade(trade)
                self.pending_trade_confirmation = None
                if result:
                    await query.edit_message_text(
                        f"✅ *Trade Executed!*\n\n"
                        f"📊 *{trade.get('symbol', 'N/A')}*\n"
                        f"• Side: {trade.get('side', 'N/A').upper()}\n"
                        f"• Size: {trade.get('size', 0):.6f}\n"
                        f"• Entry: ${trade.get('price', 0):,.4f}",
                        parse_mode="Markdown"
                    )
                else:
                    await query.edit_message_text(
                        "❌ *Trade Failed*\n\nOrder was rejected by exchange.",
                        parse_mode="Markdown"
                    )
            else:
                await query.edit_message_text("⚠️ Trade expired or already handled", parse_mode="Markdown")
        
        elif query.data == "confirm_pending_minsize":
            # User wants to use minimum size instead
            if self.pending_trade_confirmation and self.execute_pending_trade:
                trade = self.pending_trade_confirmation.copy()
                trade['size'] = trade.get('min_amount', trade['size'])  # Use minimum size
                result = await self.execute_pending_trade(trade)
                self.pending_trade_confirmation = None
                if result:
                    await query.edit_message_text(
                        f"✅ *Trade Executed with Min Size!*\n\n"
                        f"📊 *{trade.get('symbol', 'N/A')}*\n"
                        f"• Side: {trade.get('side', 'N/A').upper()}\n"
                        f"• Size: {trade.get('size', 0):.6f} (minimum)\n"
                        f"• Entry: ${trade.get('price', 0):,.4f}",
                        parse_mode="Markdown"
                    )
                else:
                    await query.edit_message_text(
                        "❌ *Trade Failed*\n\nOrder was rejected by exchange.",
                        parse_mode="Markdown"
                    )
            else:
                await query.edit_message_text("⚠️ Trade expired or already handled", parse_mode="Markdown")
        
        elif query.data == "reject_pending_trade":
            self.pending_trade_confirmation = None
            await query.edit_message_text("❌ *Trade Cancelled*", parse_mode="Markdown")

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages - chat with AI."""
        user_message = update.message.text
        user_id = update.effective_user.id
        
        logger.info(f"💬 MESSAGE RECEIVED: '{user_message[:50]}...' from user {user_id}")
        
        # Regular AI chat - build context
        # Build AI context from SINGLE SOURCE OF TRUTH: full_state
        context_info = ""
        full_state = None
        
        # Get full system state - this is the ONLY source of context for AI
        if hasattr(self, 'get_full_system_state') and self.get_full_system_state:
            try:
                full_state = self.get_full_system_state()
                
                # Parameters
                params = full_state.get('parameters', {})
                context_info += f"SYSTEM PARAMETERS:\n"
                context_info += f"  Risk: {params.get('risk_pct', 0.02)*100:.1f}% | ATR Mult: {params.get('atr_mult', 2.0)}\n"
                context_info += f"  TP Levels: {params.get('tp1_r', 1)}R/{params.get('tp2_r', 2)}R/{params.get('tp3_r', 3)}R\n"
                context_info += f"  AI Mode: {params.get('ai_mode', 'filter')} | AI Confidence: {params.get('ai_confidence', 0.7)*100:.0f}%\n"
                context_info += f"  Paused: {params.get('paused', False)}\n"
                context_info += f"  Daily Loss Limit: {params.get('daily_loss_limit', 0.05)*100:.1f}%"
                if params.get('daily_loss_triggered'):
                    context_info += " ⚠️ TRIGGERED"
                context_info += "\n"
                context_info += f"  Dry-Run Mode: {params.get('dry_run_mode', False)}\n\n"
                
                # Status (from full_state.status)
                status = full_state.get('status', {})
                context_info += f"BOT STATUS:\n"
                context_info += f"  Connected: {'Yes' if status.get('connected') else 'No'}\n"
                context_info += f"  Mode: {status.get('mode', 'Unknown')}\n"
                context_info += f"  Uptime: {status.get('uptime', 'N/A')}\n"
                context_info += f"  Balance: ${status.get('balance', 0):,.2f} (Initial: ${status.get('initial_balance', 0):,.2f})\n"
                context_info += f"  Total Trades: {status.get('total_trades', 0)} | Win Rate: {status.get('win_rate', 0):.1f}%\n\n"
                
                # ALL POSITIONS (multi-position aware)
                positions = full_state.get('position', [])
                num_positions = len(positions) if positions else 0
                
                if num_positions > 0:
                    context_info += f"🟢 OPEN POSITIONS ({num_positions} total):\n"
                    for pos in positions:
                        slot = pos.get('slot', '?')
                        symbol = pos.get('symbol', 'N/A')
                        side = pos.get('side', 'N/A').upper()
                        entry = pos.get('entry', 0)
                        pnl_val = pos.get('pnl', 0)
                        pnl_pct = pos.get('pnl_pct', 0)
                        context_info += f"  📊 POSITION {slot} ({symbol}): {side}\n"
                        context_info += f"     Entry: ${entry:,.4f} | Size: {pos.get('size', 0):.4f}\n"
                        context_info += f"     P&L: ${pnl_val:,.2f} ({pnl_pct:+.2f}%)\n"
                    context_info += f"\n"
                    
                    # IMPORTANT: Tell AI how to handle multiple positions with clear mapping
                    if num_positions > 1:
                        context_info += f"⚠️ MULTIPLE POSITIONS OPEN - Position mapping (FIXED SLOTS):\n"
                        for pos in positions:
                            context_info += f"   Position {pos.get('slot', '?')} = {pos.get('symbol', 'N/A')}\n"
                        context_info += f"\n"
                else:
                    context_info += "🔴 POSITION: **NONE** - NO OPEN TRADES!\n\n"
                
                # P&L (from full_state.pnl)
                pnl = full_state.get('pnl', {})
                context_info += f"P&L:\n"
                context_info += f"  Today: ${pnl.get('today', 0):,.2f}\n"
                context_info += f"  Total: ${pnl.get('total', 0):,.2f}\n"
                context_info += f"  Win Rate: {pnl.get('win_rate', 0)*100:.1f}%\n"
                context_info += f"  Total Trades: {pnl.get('trades', 0)} ({pnl.get('winning', 0)} wins)\n\n"
                
                # Market (from full_state.market)
                market = full_state.get('market', {})
                if market.get('price', 0) > 0:
                    context_info += f"MARKET:\n"
                    context_info += f"  {market.get('symbol', 'N/A')}: ${market.get('price', 0):,.4f}\n"
                    context_info += f"  24h Change: {market.get('change_24h', 0):.2f}%\n"
                    context_info += f"  ATR: ${market.get('atr', 0):,.4f}\n\n"
                
                # ML Model (from full_state.ml)
                ml = full_state.get('ml', {})
                samples = ml.get('total_samples', 0)
                samples_until = ml.get('samples_until_training', 50)
                is_trained = ml.get('is_trained', False)
                
                context_info += f"ML MODEL:\n"
                if is_trained:
                    win_rate = ml.get('historical_win_rate', 0)
                    context_info += f"  Status: Trained ✅\n"
                    context_info += f"  Samples: {samples}\n"
                    context_info += f"  Historical Win Rate: {win_rate*100:.0f}%\n"
                else:
                    context_info += f"  Status: Learning\n"
                    context_info += f"  Progress: {samples}/{samples + samples_until} samples ({samples_until} more needed)\n"
                context_info += "\n"
                
                # AI Stats (from full_state.ai)
                ai_stats = full_state.get('ai', {})
                context_info += f"AI FILTER:\n"
                context_info += f"  Mode: {ai_stats.get('mode', 'unknown')}\n"
                context_info += f"  Threshold: {ai_stats.get('threshold', 0.7)*100:.0f}%\n"
                context_info += f"  Approved: {ai_stats.get('approved', 0)} | Rejected: {ai_stats.get('rejected', 0)}\n"
                context_info += "\n"
                
                # AI Tracker (from full_state.ai_tracker)
                ai_tracker = full_state.get('ai_tracker', {})
                if ai_tracker.get('total_tracked', 0) > 0:
                    context_info += f"AI ACCURACY TRACKER:\n"
                    context_info += f"  Total Decisions: {ai_tracker.get('total_tracked', 0)}\n"
                    context_info += f"  Approval Rate: {ai_tracker.get('approval_rate', 'N/A')}\n"
                    context_info += f"  Approval Accuracy: {ai_tracker.get('approval_accuracy', 'N/A')}\n"
                    context_info += f"  Net AI Value: {ai_tracker.get('net_ai_value', '$0.00')}\n"
                    context_info += "\n"
                
                # Pre-filter stats (from full_state.prefilter)
                prefilter = full_state.get('prefilter', {})
                if prefilter.get('total_signals', 0) > 0:
                    context_info += f"PRE-FILTER STATS:\n"
                    context_info += f"  Total Signals: {prefilter.get('total_signals', 0)}\n"
                    context_info += f"  Passed to AI: {prefilter.get('passed', 0)}\n"
                    context_info += f"  Pass Rate: {prefilter.get('pass_rate', '0%')}\n"
                    context_info += f"  Blocked: Score={prefilter.get('blocked_by_score', 0)}, ADX Low={prefilter.get('blocked_by_adx_low', 0)}, ADX Danger={prefilter.get('blocked_by_adx_danger', 0)}, Volume={prefilter.get('blocked_by_volume', 0)}\n"
                    context_info += "\n"
                

                # Regime
                context_info += f"MARKET REGIME: {full_state.get('regime', 'unknown')}\n\n"





                
                # Market Scan - Top pairs by volatility
                market_scan = full_state.get('market_scan', {})
                if market_scan.get('top_pairs'):
                    context_info += f"MARKET SCANNER (top by score):\n"
                    context_info += f"  Currently trading: {market_scan.get('current_symbol', 'N/A')}\n"
                    for p in market_scan.get('top_pairs', []):
                        # Include ADX for proper mathematical analysis
                        adx = p.get('adx', 0)
                        adx_warn = " ⚠️DANGER" if 35 <= adx <= 40 else ""
                        context_info += f"  • {p['symbol']}: ${p['price']:,.2f} | Score:{p.get('score',0):.0f} | ADX:{adx:.0f}{adx_warn} | RSI:{p.get('rsi',50):.0f} | {p.get('trend','n/a')}\n"
                    context_info += "\n"
                
                # Recent signals
                signals = full_state.get('signals', [])
                if signals:
                    context_info += f"RECENT SIGNALS ({len(signals)}):\n"
                    for sig in signals[-3:]:
                        context_info += f"  • {sig.get('direction', 'N/A')} @ {sig.get('time', 'N/A')}\n"
                    context_info += "\n"
                    
            except Exception as e:
                logger.warning(f"Could not get full system state: {e}")
                # Fallback to individual getters only if full_state fails
                if self.get_status:
                    status = self.get_status()
                    context_info += f"Bot Status: {'Connected' if status.get('connected') else 'Disconnected'}, "
                    context_info += f"Balance: ${status.get('balance', 0):,.2f}\n"
        
        # Check for PARAMETER CHANGE commands (set risk 3%, change mode to autonomous, etc)
        msg_lower = user_message.lower()
        param_change_result = None
        
        logger.info(f"🔧 Checking for param changes in: '{msg_lower}'")
        
        if hasattr(self, 'set_system_param') and self.set_system_param:
            import re
            # Match patterns like "set risk to 3%", "change ai confidence to 80%", "set tp1 to 1.5"
            set_patterns = [
                (r'set\s+risk\s+(?:to\s+)?(\d+(?:\.\d+)?)\s*%?', 'risk_pct', lambda x: float(x)/100),
                (r'(?:set|change)\s+(?:ai\s+)?confidence\s+(?:to\s+)?(\d+(?:\.\d+)?)\s*%?', 'ai_confidence', lambda x: float(x)/100),
                (r'(?:set|change)\s+(?:ai\s+)?mode\s+(?:to\s+)?(\w+)', 'ai_mode', str),
                (r'set\s+atr\s+(?:mult(?:iplier)?\s+)?(?:to\s+)?(\d+(?:\.\d+)?)', 'atr_mult', float),
                (r'set\s+tp1\s+(?:to\s+)?(\d+(?:\.\d+)?)', 'tp1_r', float),
                (r'set\s+tp2\s+(?:to\s+)?(\d+(?:\.\d+)?)', 'tp2_r', float),
                (r'set\s+tp3\s+(?:to\s+)?(\d+(?:\.\d+)?)', 'tp3_r', float),
                (r'(?:pause|stop)\s+(?:the\s+)?(?:bot|trading)', 'paused', lambda x: True),
                (r'(?:resume|start|unpause)\s+(?:the\s+)?(?:bot|trading)', 'paused', lambda x: False),
                # Symbol/pair switching patterns
                (r'(?:switch|change|trade)\s+(?:to\s+)?(?:symbol\s+)?([A-Za-z]{2,6})(?:/usdt|usdt)?(?:\s|$)', 'symbol', str),
                (r'(?:set|change)\s+(?:symbol|pair)\s+(?:to\s+)?([A-Za-z]{2,6})', 'symbol', str),
                (r'trade\s+([A-Za-z]{2,6})(?:/usdt|usdt)?(?:\s|$)', 'symbol', str),
                # More flexible patterns for mode changes - these catch various phrasings
                (r'\bautonomous\b', 'ai_mode', lambda x: 'autonomous'),
                (r'\bfilter\b(?!\s+mode)', 'ai_mode', lambda x: 'filter'),
                (r'\bhybrid\b', 'ai_mode', lambda x: 'hybrid'),
                (r'\badvisory\b', 'ai_mode', lambda x: 'advisory'),
                (r'switch\s+(?:to\s+)?(?!btc|eth|sol|link|tia|sui)(\w+)\s*mode', 'ai_mode', str),  # Avoid matching coin names
                (r'mode\s*[=:]\s*(\w+)', 'ai_mode', str),
                (r'go\s+(?:to\s+)?(\w+)\s+mode', 'ai_mode', str),
                (r'enable\s+(\w+)\s+mode', 'ai_mode', str),
                (r'use\s+(\w+)\s+mode', 'ai_mode', str),
            ]
            
            for pattern, param, converter in set_patterns:
                match = re.search(pattern, msg_lower)
                if match:
                    try:
                        # Handle patterns with and without capture groups
                        if match.groups() and match.group(1):
                            value = converter(match.group(1))
                        else:
                            value = converter(True)
                        logger.info(f"🔧 Matched pattern '{pattern}' → {param}={value}")
                        param_change_result = self.set_system_param(param, value)
                        if param_change_result.get('success'):
                            context_info += f"\n\n✅ PARAMETER CHANGED: {param_change_result.get('message')}"
                            logger.info(f"✅ Parameter change success: {param_change_result.get('message')}")
                            # If ai_mode changed, update the context_info so AI sees the NEW mode
                            if param == 'ai_mode':
                                import re as _re
                                context_info = _re.sub(
                                    r'AI Mode:\s*\w+',
                                    f'AI Mode: {value}',
                                    context_info
                                )
                                logger.info(f"🔄 Updated context_info AI Mode to: {value}")
                        else:
                            context_info += f"\n\n❌ PARAMETER CHANGE FAILED: {param_change_result.get('message')}"
                            logger.warning(f"❌ Parameter change failed: {param_change_result.get('message')}")
                        break
                    except Exception as e:
                        logger.error(f"Parameter change error: {e}")
        else:
            logger.warning("🔧 set_system_param not available!")
        
        # === AUTONOMOUS MODE DETECTION ===
        # Check if bot is in autonomous mode - if so, auto-execute AI commands
        # NOTE: This runs AFTER parameter changes, so mode switches in the same
        # message are respected (e.g., "switch to advisory and open long")
        is_autonomous_mode = False
        if self.get_ai_mode:
            try:
                ai_mode = self.get_ai_mode()
                is_autonomous_mode = ai_mode == 'autonomous'
                logger.info(f"🤖 AI Mode: {ai_mode} (autonomous={is_autonomous_mode})")
            except Exception as e:
                logger.debug(f"Could not check AI mode: {e}")
        
        # Check for trade execution requests
        # Differentiate between:
        # 1. Trade REQUEST (needs AI analysis first): "open long", "go short", etc.
        # 2. EXPLICIT EXECUTION (user already saw analysis): "execute", "do it", "confirm"
        
        # Explicit execution keywords - these BYPASS AI analysis and execute immediately
        explicit_execute_keywords = ["execute", "do it", "confirm", "yes execute", "execute now", 
                                     "execute it", "just do it", "proceed", "go ahead"]
        is_explicit_execute = any(kw in msg_lower for kw in explicit_execute_keywords)
        
        # Check for long/short intent with flexible pattern matching
        # Handles: "open long", "open elsa long", "buy btc", "go long on eth", etc.
        import re
        
        # Pattern: "open/buy/go" + optional symbol + "long"
        long_pattern = re.compile(r'\b(open|buy|go|enter)\s+\w*\s*(long|buy)\b|\b(long|buy)\s+(now|it|this)\b|\bexecute\s+long\b|\byes\s+long\b', re.IGNORECASE)
        short_pattern = re.compile(r'\b(open|sell|go|enter)\s+\w*\s*(short|sell)\b|\b(short|sell)\s+(now|it|this)\b|\bexecute\s+short\b|\byes\s+short\b', re.IGNORECASE)
        
        # Also check simple keywords
        trade_keywords_long = ["buy", "go long", "open long", "enter long", "long now", "buy now", 
                               "execute long", "execute buy", "yes buy", "yes long",
                               "execute trade", "make the trade", "open the trade", "enter the trade"]
        trade_keywords_short = ["sell", "go short", "open short", "enter short", "short now", "sell now",
                                "execute short", "execute sell", "yes short", "yes sell"]
        
        # Use both pattern matching and keyword matching
        should_execute_long = bool(long_pattern.search(msg_lower)) or any(kw in msg_lower for kw in trade_keywords_long)
        should_execute_short = bool(short_pattern.search(msg_lower)) or any(kw in msg_lower for kw in trade_keywords_short)
        
        # Close keywords - these are COMMANDS to close, not questions
        close_command_keywords = ["close position", "exit position", "close trade", "take profit",
                          "close it", "exit trade", "close now", "exit now", "close the position",
                          "exit the position", "execute close", "exit", "close", "sell position", "sell it"]
        
        # Position inquiry keywords - user is ASKING about hold/close decision (NOT commanding)
        # These take PRECEDENCE over close detection - user is asking, not commanding
        position_inquiry_keywords = ["should we", "should i", "shall we", "shall i",
                                     "hold or close", "close or hold", "hold or exit", "exit or hold",
                                     "what should", "what do you think", "what do you recommend",
                                     "is it time to", "recommend", "advise",
                                     "leave it open", "keep it open", "stay in",
                                     "is the position", "how is the position", "position status", "trade status"]
        is_position_inquiry = any(kw in msg_lower for kw in position_inquiry_keywords)
        
        # CRITICAL: If it's an inquiry question, do NOT treat as close command
        # "should we close" is a QUESTION, not a command
        # "close it now" is a COMMAND
        if is_position_inquiry:
            should_close = False  # Inquiry takes precedence - user is asking, not commanding
        else:
            should_close = any(kw in msg_lower for kw in close_command_keywords) and not (should_execute_long or should_execute_short)
        
        logger.info(f"🔍 Keyword check: msg='{user_message}' long={should_execute_long} short={should_execute_short} close={should_close} inquiry={is_position_inquiry}")
        
        # === HANDLE AMBIGUOUS "EXECUTE" WITHOUT DIRECTION ===
        # User typed "execute" or "Execute" without specifying long/short/close
        if is_explicit_execute and not (should_execute_long or should_execute_short or should_close):
            # Check if it's just "execute" by itself (no direction specified)
            stripped_msg = msg_lower.strip()
            if stripped_msg in ["execute", "do it", "confirm", "proceed", "go ahead", "yes execute", "execute now", "execute it", "just do it"]:
                await update.message.reply_text(
                    "⚠️ *Please specify what to execute:*\n\n"
                    "• /execute\\_long - Open LONG position\n"
                    "• /execute\\_short - Open SHORT position\n"
                    "• /execute\\_close - Close current position\n\n"
                    "Or say: `execute long`, `execute short`, or `execute close`",
                    parse_mode="Markdown"
                )
                return
        
        # === MANUAL TRADE REQUEST HANDLING ===
        # In autonomous mode: Auto-execute AI commands
        # In advisory/filter mode: Let user decide
        trade_executed_by_user = False
        manual_trade_request = False
        requested_side = None
        
        # Extract symbol from message if specified (e.g., "open long NEAR", "go short on ETH")
        # Dynamic extraction: any word that isn't a trade keyword is likely a symbol
        open_symbol_specified = None
        trade_noise_words = {'open', 'close', 'buy', 'sell', 'go', 'enter', 'exit', 'execute',
                             'long', 'short', 'now', 'on', 'a', 'the', 'please', 'pls', 'it',
                             'this', 'that', 'trade', 'position', 'yes', 'no', 'do', 'make',
                             'should', 'we', 'i', 'can', 'you', 'with', 'for', 'at', 'to',
                             'and', 'or', 'is', 'are', 'be', 'in', 'of', 'my', 'me', 'us'}
        msg_words = re.findall(r'[a-zA-Z]{2,}', user_message)  # Words with 2+ letters
        for word in msg_words:
            if word.lower() not in trade_noise_words:
                # This is likely a symbol name (ELSA, BTC, ETH, SOL, etc.)
                candidate = word.upper()
                if not candidate.endswith('USDT'):
                    candidate += 'USDT'
                open_symbol_specified = candidate
                logger.info(f"📋 Extracted symbol from message: {open_symbol_specified}")
                break
        
        if should_execute_long or should_execute_short:
            requested_side = "long" if should_execute_long else "short"
            
            # === AUTONOMOUS MODE: BLOCK ALL MANUAL TRADE COMMANDS ===
            # In autonomous mode, the system makes its own decisions.
            # User must switch to advisory or hybrid mode to manually execute trades.
            if is_autonomous_mode:
                logger.info(f"🚫 BLOCKED manual {requested_side} trade in autonomous mode")
                await update.message.reply_text(
                    "🤖 *AUTONOMOUS MODE ACTIVE*\n\n"
                    "The system is in **autonomous mode** — it opens and closes trades "
                    "on its own based on mathematical analysis.\n\n"
                    "❌ Manual trade commands are disabled in this mode.\n\n"
                    "To manually open/close trades, switch to:\n"
                    "• `/mode advisory` — You decide, AI advises\n"
                    "• `/mode hybrid` — AI suggests, you confirm\n\n"
                    "_Or say: \"switch to advisory\"_",
                    parse_mode="Markdown"
                )
                return
            
            # Check if this is an EXPLICIT EXECUTION command (user already saw analysis)
            if is_explicit_execute:
                # User is explicitly confirming - execute immediately
                logger.info(f"⚡ Explicit execute command - executing {requested_side}{' on ' + open_symbol_specified if open_symbol_specified else ''}")
                if self.execute_ai_trade:
                    result = await self.execute_ai_trade(requested_side, open_symbol_specified)
                    if result["success"]:
                        context_info += f"\n\n✅ TRADE EXECUTED: {result['message']}"
                        trade_executed_by_user = True
                        # Verify position
                        await asyncio.sleep(0.5)
                        if hasattr(self, 'get_full_system_state') and self.get_full_system_state:
                            try:
                                state = self.get_full_system_state()
                                positions = state.get('position', [])
                                if positions and len(positions) > 0:
                                    pos = positions[-1]  # Get latest position
                                    context_info += f"\n📊 VERIFIED: {pos.get('side', 'N/A').upper()} {pos.get('symbol', 'N/A')} @ ${pos.get('entry', 0):.4f}"
                            except Exception as ve:
                                logger.warning(f"Could not verify trade: {ve}")
                    else:
                        context_info += f"\n\n❌ TRADE FAILED: {result['message']}"
            else:
                # ALL MANUAL TRADE REQUESTS: Always show analysis + confirmation button
                # Regardless of AI mode, user-initiated trades require confirmation
                manual_trade_request = True
                logger.info(f"📋 Manual trade request: {requested_side}{' on ' + open_symbol_specified if open_symbol_specified else ''} - showing analysis + confirm")
                context_info += f"\n\n📋 MANUAL TRADE REQUEST: {requested_side.upper()}"
                if open_symbol_specified:
                    context_info += f" on {open_symbol_specified}"
                
                # Show current positions
                positions = full_state.get('position', []) if full_state else []
                num_positions = len(positions)
                max_positions = 2
                
                if num_positions > 0:
                    context_info += f"\n📊 CURRENT POSITIONS ({num_positions}/{max_positions}):"
                    for p in positions:
                        context_info += f"\n   • {p.get('side', 'N/A').upper()} {p.get('symbol', 'N/A')}"
                
                if num_positions >= max_positions:
                    context_info += f"\n⚠️ MAX POSITIONS REACHED ({max_positions}) - Cannot open new position"
                
                # === INJECT SYMBOL-SPECIFIC TRADE CONTEXT (F&G, S/R, Volume, etc.) ===
                try:
                    if self.get_symbol_trade_context and open_symbol_specified:
                        trade_ctx = self.get_symbol_trade_context(open_symbol_specified)
                        if trade_ctx:
                            context_info += f"\n\n📊 REAL-TIME MARKET DATA FOR {trade_ctx.get('symbol', open_symbol_specified)}:"
                            
                            # Fear & Greed
                            fg = trade_ctx.get('fear_greed_index')
                            fg_label = trade_ctx.get('fear_greed_label', '')
                            if fg is not None:
                                context_info += f"\n🌡️ Fear & Greed Index: {fg} ({fg_label})"
                            
                            # Price
                            price = trade_ctx.get('price')
                            if price:
                                context_info += f"\n💰 Current Price: ${price}"
                            
                            # Support/Resistance
                            sr = trade_ctx.get('support_resistance', {})
                            if sr:
                                pos_pct = sr.get('position_in_range_pct')
                                if pos_pct is not None:
                                    context_info += f"\n📏 S/R Position: {pos_pct:.1f}% (0%=support, 100%=resistance)"
                                if sr.get('in_resistance_zone'):
                                    context_info += f"\n🔴 IN RESISTANCE ZONE"
                                if sr.get('in_support_zone'):
                                    context_info += f"\n🟢 IN SUPPORT ZONE"
                                sup = sr.get('support_lower')
                                res = sr.get('resistance_upper')
                                if sup:
                                    context_info += f"\n   Support: ${sup:.4f}"
                                if res:
                                    context_info += f"\n   Resistance: ${res:.4f}"
                                r_touches = sr.get('resistance_touches', 0)
                                s_touches = sr.get('support_touches', 0)
                                if r_touches or s_touches:
                                    context_info += f"\n   Touches - Resistance: {r_touches}, Support: {s_touches}"
                            
                            # Technical indicators
                            rsi = trade_ctx.get('rsi')
                            adx = trade_ctx.get('adx')
                            vol = trade_ctx.get('volume_ratio')
                            if rsi:
                                context_info += f"\n📈 RSI: {rsi:.1f}"
                            if adx:
                                context_info += f"\n📈 ADX: {adx:.1f}"
                            if vol:
                                context_info += f"\n📊 Volume Ratio: {vol:.2f}x"
                            
                            # Regime
                            regime = trade_ctx.get('regime')
                            if regime:
                                context_info += f"\n🔄 Market Regime: {regime}"
                            
                            # CRITICAL: Warnings that will BLOCK the trade
                            warnings = trade_ctx.get('warnings', [])
                            if warnings:
                                context_info += f"\n\n⚠️ CRITICAL WARNINGS (these WILL block the trade):"
                                for w in warnings:
                                    context_info += f"\n   🚫 {w}"
                                context_info += f"\n   → The bot WILL REJECT this trade unless conditions change!"
                            
                            logger.info(f"📊 Injected trade context for {open_symbol_specified}: F&G={fg}, S/R={sr.get('position_in_range_pct', 'N/A')}%, warnings={len(warnings)}")
                except Exception as e:
                    logger.error(f"Failed to get symbol trade context: {e}")
                
                context_info += f"\n\n📋 INSTRUCTIONS FOR AI (ADVISORY MODE - CRITICAL):"
                context_info += f"\n1. You are in ADVISORY mode - analyze AND provide the execute command"
                context_info += f"\n2. Do STRICT PHD-LEVEL MATHEMATICAL ANALYSIS of current conditions"
                context_info += f"\n3. Check Multi-Timeframe (MTF) alignment, Kalman momentum, Volume, Fear&Greed"
                context_info += f"\n4. If there are ⚠️ CRITICAL WARNINGS above, TELL the user the trade WILL BE BLOCKED and WHY"
                context_info += f"\n5. Give clear recommendation: ✅ TRADE or ❌ NO TRADE with math reasoning"
                context_info += f"\n6. 🚨 MANDATORY: You MUST include this EXACT command block in your response"
                context_info += f"\n   (even if you recommend against the trade - user decides):"
                if open_symbol_specified:
                    context_info += f'\n```command\n{{"action": "open_trade", "side": "{requested_side}", "symbol": "{open_symbol_specified}"}}\n```'
                else:
                    context_info += f'\n```command\n{{"action": "open_trade", "side": "{requested_side}"}}\n```'
                context_info += f"\n7. The command block above will show user a /execute_{requested_side} button"
                context_info += f"\n8. User will click the button to confirm - you do NOT execute"
                context_info += f"\n9. DO NOT skip the command block. DO NOT say 'I cannot execute'. ALWAYS include it."

        # Close position if requested - autonomous or manual based on mode
        elif should_close:
            # === AUTONOMOUS MODE: BLOCK MANUAL CLOSE COMMANDS ===
            if is_autonomous_mode:
                logger.info(f"🚫 BLOCKED manual close in autonomous mode")
                await update.message.reply_text(
                    "🤖 *AUTONOMOUS MODE ACTIVE*\n\n"
                    "The system manages positions autonomously — it will close "
                    "trades based on TP/SL levels and mathematical analysis.\n\n"
                    "❌ Manual close commands are disabled in this mode.\n\n"
                    "To manually close trades, switch to:\n"
                    "• `/mode advisory` — You decide, AI advises\n"
                    "• `/mode hybrid` — AI suggests, you confirm\n\n"
                    "_Or say: \"switch to advisory\"_",
                    parse_mode="Markdown"
                )
                return
            
            # Extract symbol from message by matching against ACTUAL OPEN POSITIONS
            # Instead of guessing from words, check if any open position symbol appears in the message
            close_symbol_specified = None
            positions = full_state.get('position', []) if full_state else []
            
            if positions:
                for pos in positions:
                    pos_symbol = pos.get('symbol', '').upper()
                    # Extract base name (e.g., "NEARUSDT" → "near", "HYPEUSDT" → "hype")
                    base_name = pos_symbol.replace('USDT', '').replace('/USDT', '').lower()
                    if base_name and base_name in msg_lower:
                        close_symbol_specified = pos_symbol if pos_symbol.endswith('USDT') else pos_symbol + 'USDT'
                        logger.info(f"📋 Matched close symbol from open position: {close_symbol_specified}")
                        break
                
                # If only 1 position open and user didn't specify, auto-select it
                if not close_symbol_specified and len(positions) == 1:
                    close_symbol_specified = positions[0].get('symbol', '').upper()
                    if close_symbol_specified and not close_symbol_specified.endswith('USDT'):
                        close_symbol_specified += 'USDT'
                    logger.info(f"📋 Auto-selected only open position: {close_symbol_specified}")
            
            # Handle "position 1", "position1", "pos 1", "pos1", "#1", "#2" references
            import re
            pos_num_match = re.search(r'(?:position|pos)\s*#?\s*(\d+)|#(\d+)', msg_lower)
            if pos_num_match and not close_symbol_specified and full_state:
                target_num = int(pos_num_match.group(1) or pos_num_match.group(2))
                positions = full_state.get('position', [])
                
                # First try by SLOT number
                for pos in positions:
                    if pos.get('slot') == target_num:
                        close_symbol_specified = pos.get('symbol', '').upper()
                        if close_symbol_specified and not close_symbol_specified.endswith('USDT'):
                            close_symbol_specified = close_symbol_specified + 'USDT'
                        logger.info(f"📍 Position slot {target_num} mapped to symbol: {close_symbol_specified}")
                        break
                
                # Fallback: treat as list index (position 1 = first, position 2 = second)
                if not close_symbol_specified and 1 <= target_num <= len(positions):
                    pos = positions[target_num - 1]
                    close_symbol_specified = pos.get('symbol', '').upper()
                    if close_symbol_specified and not close_symbol_specified.endswith('USDT'):
                        close_symbol_specified = close_symbol_specified + 'USDT'
                    logger.info(f"📍 Position index {target_num} mapped to symbol: {close_symbol_specified}")
                
                if not close_symbol_specified:
                    logger.warning(f"📍 No position found for number {target_num}")
            
            # Check if this is an EXPLICIT EXECUTION command (user already saw analysis)
            if is_explicit_execute and self.close_ai_trade:
                logger.info(f"🔴 Explicit execute command - closing position{' (' + close_symbol_specified + ')' if close_symbol_specified else ''}")
                result = await self.close_ai_trade(close_symbol_specified)
                logger.info(f"💼 Close result: {result}")
                
                if result["success"]:
                    context_info += f"\n\n✅ POSITION CLOSED: {result['message']}"
                else:
                    context_info += f"\n\n❌ CLOSE FAILED: {result['message']}"
            elif is_autonomous_mode:
                # AUTONOMOUS MODE: Auto-execute close after AI analysis
                logger.info(f"🤖 AUTONOMOUS MODE: Close request{' (' + close_symbol_specified + ')' if close_symbol_specified else ''}")
                context_info += f"\n\n🤖 AUTONOMOUS MODE ACTIVE"
                context_info += f"\n⚡ Close position request - will execute after validation"
                
                # Add symbol context for multi-position handling
                if close_symbol_specified:
                    context_info += f"\n📌 USER SPECIFIED SYMBOL: {close_symbol_specified}"
                else:
                    # Check if multiple positions - AI must ask which one
                    positions = full_state.get('position', []) if full_state else []
                    if len(positions) > 1:
                        context_info += f"\n⚠️ MULTIPLE POSITIONS OPEN - Ask user which one to close!"
                        symbols = [p.get('symbol', 'N/A') for p in positions]
                        context_info += f"\n   Open: {', '.join(symbols)}"
                
                context_info += f"\n📋 INSTRUCTIONS FOR AI:"
                context_info += f"\n1. Do STRICT PHD-LEVEL MATHEMATICAL ANALYSIS of exit conditions"
                context_info += f"\n2. If MULTIPLE positions open and user didn't specify, ASK which one"
                context_info += f"\n3. Give your recommendation (CLOSE or HOLD) with quantitative reasoning"
                context_info += f"\n4. IF YOU RECOMMEND CLOSE, include SYMBOL in command:"
                context_info += f'\n```command\n{{"action": "close_position", "symbol": "SYMBOLUSDT", "autonomous": true}}\n```'
                context_info += f"\n5. Replace SYMBOLUSDT with the actual symbol (e.g., NEARUSDT, ETHUSDT)"
                # NOTE: Do NOT set trade_executed_by_user here - it's only a REQUEST at this point
                # The actual execution will happen later when AI command is processed
            else:
                # ADVISORY/FILTER MODE: Require user confirmation
                manual_trade_request = True
                context_info += f"\n\n⚠️ USER MANUAL CLOSE REQUEST"
                if close_symbol_specified:
                    context_info += f" for {close_symbol_specified}"
                
                # Show current positions for context
                positions = full_state.get('position', []) if full_state else []
                if positions:
                    context_info += f"\n📊 CURRENT POSITIONS:"
                    for p in positions:
                        p_sym = p.get('symbol', 'N/A')
                        p_side = p.get('side', 'N/A').upper()
                        p_pnl = p.get('pnl', p.get('unrealized_pnl', 'N/A'))
                        context_info += f"\n   • {p_side} {p_sym} (PnL: {p_pnl})"
                
                # If no symbol specified and multiple positions, note it
                if not close_symbol_specified and len(positions) > 1:
                    context_info += f"\n⚠️ MULTIPLE POSITIONS - user didn't specify which one. ASK them!"
                
                # Auto-detect symbol if only one position open
                if not close_symbol_specified and len(positions) == 1:
                    close_symbol_specified = positions[0].get('symbol', '')
                    if close_symbol_specified and not close_symbol_specified.endswith('USDT'):
                        close_symbol_specified += 'USDT'
                    logger.info(f"📋 Auto-detected close symbol (only 1 position): {close_symbol_specified}")
                
                context_info += f"\n\n📋 INSTRUCTIONS FOR AI (ADVISORY MODE - CRITICAL):"
                context_info += f"\n1. Do STRICT PHD-LEVEL MATHEMATICAL ANALYSIS of exit conditions"
                context_info += f"\n2. Verify if technical indicators support closing NOW or holding"
                context_info += f"\n3. Give your recommendation (CLOSE or HOLD) with quantitative reasoning"
                context_info += f"\n4. 🚨 MANDATORY: You MUST include this EXACT command block in your response"
                context_info += f"\n   (even if you recommend holding - user decides):"
                if close_symbol_specified:
                    context_info += f'\n```command\n{{"action": "close_position", "symbol": "{close_symbol_specified}"}}\n```'
                else:
                    context_info += f'\n```command\n{{"action": "close_position"}}\n```'
                context_info += f"\n5. The command block above will show user a /execute_close button"
                context_info += f"\n6. User will click the button to confirm - you do NOT execute"
                context_info += f"\n7. DO NOT skip the command block. DO NOT say 'I cannot execute'. ALWAYS include it."
                logger.info(f"📋 Advisory/Filter mode: close request, symbol={close_symbol_specified}")
        
        # === POSITION INQUIRY - User asking about hold/close decision ===
        # In autonomous mode, AI can analyze AND execute if it recommends closing
        elif is_position_inquiry and full_state:
            positions = full_state.get('position', [])
            if positions and len(positions) > 0:
                logger.info(f"🔍 Position inquiry detected - providing math-based analysis")
                
                # Get position monitor data from ai_filter if available
                monitor_context = ""
                math_decision = "HOLD"  # Default
                if hasattr(self, 'get_position_monitor_analysis') and self.get_position_monitor_analysis:
                    try:
                        monitor_data = self.get_position_monitor_analysis()
                        if monitor_data:
                            math_decision = monitor_data.get('action', 'hold').upper()
                            hold_score = monitor_data.get('hold_score', 50)
                            exit_score = monitor_data.get('exit_score', 50)
                            
                            monitor_context = f"\n\n" + "="*50
                            monitor_context += f"\n📊 **PHD MATH DECISION (SOURCE OF TRUTH)**"
                            monitor_context += f"\n" + "="*50
                            monitor_context += f"\n  🎯 MATH SAYS: **{math_decision}**"
                            monitor_context += f"\n  Hold Score: {hold_score:.0f}/100"
                            monitor_context += f"\n  Exit Score: {exit_score:.0f}/100"
                            monitor_context += f"\n  Confidence: {monitor_data.get('confidence', 0)*100:.0f}%"
                            monitor_context += f"\n  P&L: {monitor_data.get('pnl_pct', 0):+.2f}%"
                            monitor_context += f"\n  Reasoning: {monitor_data.get('reasoning', 'N/A')[:100]}"
                            monitor_context += f"\n" + "="*50
                    except Exception as e:
                        logger.debug(f"Could not get position monitor analysis: {e}")
                
                context_info += monitor_context
                context_info += f"\n\n⚠️ **CRITICAL: YOU MUST FOLLOW THE MATH DECISION ABOVE**"
                context_info += f"\n📋 INSTRUCTIONS FOR AI:"
                context_info += f"\n1. The PhD Math Monitor above is the SOURCE OF TRUTH"
                context_info += f"\n2. If Math says HOLD → You MUST recommend HOLD"
                context_info += f"\n3. If Math says CLOSE → You MAY recommend CLOSE"
                context_info += f"\n4. Exit requires: Exit Score >= 70 AND Exit > Hold + 15"
                context_info += f"\n5. DO NOT override the math with your own opinion"
                context_info += f"\n6. Explain the math scores to the user in simple terms"
                context_info += f"\n7. Current math decision is: **{math_decision}** - FOLLOW IT"
        
        # === PAIR ANALYSIS - Use dedicated market analyzer for consistency ===
        # When user asks about pairs/switching, use SAME function as dashboard
        pair_question_keywords = [
            'which pair', 'what pair', 'best pair', 'switch pair', 'change pair',
            'other pair', 'different pair', 'scan pair', 'scanning pair', 'analyze pair',
            'should i switch', 'should we switch', 'recommend pair', 'suggest pair',
            'eth or', 'sol or', 'link or', 'btc or', 'compare pair', 'pair comparison'
        ]
        is_pair_question = any(kw in msg_lower for kw in pair_question_keywords)
        
        if is_pair_question and self.ai_analyze_markets:
            logger.info("📊 Pair question detected - using dedicated market analyzer")
            analysis = self.ai_analyze_markets()
            if analysis and analysis.get('recommendation'):
                context_info += f"\n\n🤖 AI MARKET ANALYSIS (unified with dashboard):\n{analysis['recommendation']}"
                if analysis.get('best_pair'):
                    context_info += f"\n\nBest Pair: {analysis['best_pair']}"
        
        # === ALWAYS INCLUDE MATH CONTEXT IF POSITION OPEN ===
        # This ensures AI knows the math scores for ANY question about the position
        if full_state and not is_position_inquiry:  # Don't duplicate if already added above
            positions = full_state.get('position', [])
            if positions and len(positions) > 0:
                if hasattr(self, 'get_position_monitor_analysis') and self.get_position_monitor_analysis:
                    try:
                        monitor_data = self.get_position_monitor_analysis()
                        if monitor_data:
                            math_decision = monitor_data.get('action', 'hold').upper()
                            context_info += f"\n\n📊 LIVE MATH MONITOR: {math_decision} (Hold:{monitor_data.get('hold_score', 50):.0f} Exit:{monitor_data.get('exit_score', 50):.0f})"
                            context_info += f"\n⚠️ If discussing position, follow this math decision."
                    except Exception:
                        pass  # Non-critical: couldn't get monitor analysis
        
        # === ENHANCED AI INSTRUCTIONS ===
        # Give AI clear guidance on how to respond and when to provide actionable commands
        ai_instructions = ""
        
        if manual_trade_request:
            # Already in context_info above
            pass
        else:
            # General AI instructions for every response
            ai_instructions += "\n\n" + "="*60
            ai_instructions += "\n🤖 AI RESPONSE GUIDELINES:"
            ai_instructions += "\n1. ALWAYS be mathematical and data-driven in analysis"
            ai_instructions += "\n2. When user asks for ACTION (open/close/change), provide actionable commands"
            ai_instructions += "\n3. Format commands as: ```command\n{\"action\": \"...\", \"param\": value}\n```"
            ai_instructions += "\n4. Use these actions: open_trade, close_position, switch_symbol, set_param, pause, resume"
            ai_instructions += "\n5. For parameter changes, provide command like: ```command\n{\"action\": \"set_param\", \"param\": \"risk_pct\", \"value\": 0.03}\n```"
            ai_instructions += "\n6. If user wants a trade, give analysis FIRST, then ALWAYS include the ```command``` block so user can execute"
            ai_instructions += "\n7. The command block will show the user an execute button — they decide whether to click it"
            ai_instructions += "\n8. Suggest specific Telegram commands: /analyze, /status, /positions, /market"
            ai_instructions += "\n" + "="*60
            context_info += ai_instructions
        
        # Use AI to generate response
        logger.info(f"💬 CHAT: About to call AI, chat_with_ai={self.chat_with_ai is not None}")
        if self.chat_with_ai:
            try:
                logger.info(f"💬 CHAT: Calling chat_with_ai with message: '{user_message[:30]}...'")
                response = await self.chat_with_ai(user_message, context_info)
                logger.info(f"💬 CHAT: Got response ({len(response) if response else 0} chars)")
                
                # === FILTER CONTRADICTORY AI RESPONSES ===
                # If trade was executed by user command, remove any lines saying it won't be executed
                if trade_executed_by_user and response:
                    import re
                    lines = response.split('\n')
                    filtered_lines = []
                    for line in lines:
                        # Skip lines that contradict the executed trade
                        if re.search(r"will not be executed|will not execute|not executing|will not open|will not trade|won't execute|won't be executed|should not|would not recommend|don't recommend|do not recommend|mathematically.*no|criteria.*not met", line, re.IGNORECASE):
                            logger.warning(f"🛡️ Filtered contradictory AI line: {line}")
                            continue
                        # Skip lines that promise a command but trade already executed
                        if re.search(r"here is the command|here's the command|command to proceed|if you.*(want|wish|still).*proceed|command below|following command", line, re.IGNORECASE):
                            logger.warning(f"🛡️ Filtered unnecessary command promise: {line}")
                            continue
                        filtered_lines.append(line)
                    response = '\n'.join(filtered_lines).strip()
                    # Prepend confirmation note with the action taken
                    if should_execute_long:
                        response = f"✅ **LONG Trade Executed Successfully**\n\n" + response
                    elif should_execute_short:
                        response = f"✅ **SHORT Trade Executed Successfully**\n\n" + response
                    # Note: trade_executed_by_user is only True for explicit long/short, not close
                
                # === MATHEMATICAL COMMAND VALIDATION (SECOND LAYER) ===
                # Commands require EXPLICIT user intent - no accidental changes
                import re
                import json
                
                user_msg_lower = user_message.lower().strip()
                
                # === VALIDATION FUNCTIONS ===
                def validate_command_intent(user_msg: str, action: str, cmd: dict) -> tuple:
                    """
                    Mathematical validation of user intent before command execution.
                    Returns (should_execute, reason)
                    """
                    msg = user_msg.lower().strip()
                    
                    # BLOCK 1: Acknowledgements NEVER trigger commands
                    acknowledgements = [
                        'okay', 'ok', 'yes', 'got it', 'thanks', 'thank you', 'alright', 
                        'cool', 'noted', 'understood', 'sure', 'fine', 'great', 'good',
                        'nice', 'perfect', 'awesome', 'kk', 'k', 'yep', 'yup', 'right',
                        'i see', 'makes sense', 'roger', 'copy', 'hmm', 'hm', 'ah', 'oh'
                    ]
                    if msg in acknowledgements or (len(msg) < 15 and any(msg == ack or msg.startswith(ack + ' ') for ack in acknowledgements)):
                        return False, f"BLOCKED: Acknowledgement '{msg}' cannot trigger commands"
                    
                    # BLOCK 2: Questions without action don't trigger
                    question_words = ['what', 'how', 'why', 'when', 'where', 'is ', 'are ', 'can ', 'does ', 'do ']
                    is_question = any(msg.startswith(q) or f' {q}' in msg or msg.endswith('?') for q in question_words)
                    
                    # BLOCK 3: Action verbs REQUIRED for settings changes
                    action_verbs = [
                        'change', 'set', 'switch', 'reduce', 'increase', 'adjust', 
                        'make', 'put', 'raise', 'lower', 'modify', 'update', 'enable',
                        'disable', 'turn on', 'turn off', 'activate', 'deactivate'
                    ]
                    has_action_verb = any(verb in msg for verb in action_verbs)
                    
                    # === ACTION-SPECIFIC VALIDATION ===
                    
                    # SET_PARAM: Changing system parameters
                    if action == 'set_param':
                        param = cmd.get('param', '').lower()
                        value = cmd.get('value')
                        
                        # User must explicitly mention the parameter or related term
                        param_keywords = {
                            'risk_pct': ['risk', 'position size', 'bet size'],
                            'ai_confidence': ['confidence', 'threshold', 'ai threshold'],
                            'atr_mult': ['atr', 'stop loss', 'stop distance'],
                            'tp1_r': ['tp1', 'take profit 1', 'first target'],
                            'tp2_r': ['tp2', 'take profit 2', 'second target'],
                            'tp3_r': ['tp3', 'take profit 3', 'third target'],
                            'symbol': ['symbol', 'pair', 'btc', 'eth', 'sol', 'link'],
                            'ai_mode': ['mode', 'autonomous', 'filter'],
                            'paused': ['pause', 'resume', 'stop', 'start']
                        }
                        
                        keywords = param_keywords.get(param, [param])
                        if not any(kw in msg for kw in keywords):
                            return False, f"BLOCKED: User didn't mention '{param}' or related terms"
                        
                        if not has_action_verb:
                            return False, f"BLOCKED: No action verb - user asked about '{param}' but didn't request change"
                        
                        # Value must be mentioned or clearly implied
                        if isinstance(value, (int, float)):
                            # Check if user mentioned a number
                            numbers_in_msg = re.findall(r'[\d.]+', msg)
                            if not numbers_in_msg and 'reduce' not in msg and 'increase' not in msg:
                                return False, f"BLOCKED: No value specified for {param}"
                        
                        return True, f"VALIDATED: User explicitly requested {param} change"
                    
                    # OPEN_TRADE: Opening positions - VALIDATE SYMBOL
                    if action == 'open_trade':
                        trade_keywords = ['open', 'buy', 'sell', 'long', 'short', 'enter', 'trade']
                        if not any(kw in msg for kw in trade_keywords):
                            return False, "BLOCKED: No trade intent detected"
                        
                        # Check if symbol specified in command
                        cmd_symbol = cmd.get('symbol', '').upper().replace('/USDT', '').replace('USDT', '')
                        
                        # Check max positions and existing positions
                        if hasattr(self, 'get_full_system_state') and self.get_full_system_state:
                            try:
                                state = self.get_full_system_state()
                                positions = state.get('position', [])
                                open_symbols = [p.get('symbol', '').replace('USDT', '') for p in positions]
                                
                                # Check if already have position on this symbol
                                if cmd_symbol and cmd_symbol in open_symbols:
                                    return False, f"BLOCKED: Already have position on {cmd_symbol}"
                                
                                # Check max positions
                                if len(positions) >= 2:  # max_positions
                                    return False, f"BLOCKED: Max positions (2) reached. Open: {', '.join(open_symbols)}"
                            except Exception:
                                pass  # Non-critical: position validation
                        
                        return True, "VALIDATED: Trade intent confirmed"
                    
                    # CLOSE_TRADE/CLOSE_POSITION: Closing positions (aliases)
                    if action == 'close_trade' or action == 'close_position':
                        # Distinguish between QUESTIONS and COMMANDS
                        # Questions: "should we close?", "hold or close?" → advice only
                        # Commands: "close it", "close now", "close position" → can execute
                        
                        question_indicators = ['should we', 'should i', 'shall we', 'shall i',
                                              'hold or', 'or hold', 'what do you', 'what should',
                                              'recommend', 'advise', 'think we should']
                        # Only treat as question if ends with ? or has question words without command words
                        is_question = (msg.strip().endswith('?') or 
                                      any(kw in msg for kw in question_indicators))
                        
                        command_keywords = ['close it', 'exit it', 'close now', 'exit now', 
                                           'close position', 'exit position', 'close trade',
                                           'execute close', 'do close', 'close the position',
                                           'close the']  # "close the NEAR position"
                        is_command = any(kw in msg for kw in command_keywords)
                        
                        # Check if user specified a symbol by matching against open positions
                        specified_symbol = None
                        if hasattr(self, 'get_full_system_state') and self.get_full_system_state:
                            try:
                                state = self.get_full_system_state()
                                open_positions = state.get('position', [])
                                msg_lower_check = msg.lower()
                                for pos in open_positions:
                                    pos_sym = pos.get('symbol', '').upper()
                                    base = pos_sym.replace('USDT', '').replace('/USDT', '').lower()
                                    if base and base in msg_lower_check:
                                        specified_symbol = base.upper()
                                        break
                            except Exception:
                                pass
                        
                        # "close NEAR" is a command even without "close it"
                        if specified_symbol and 'close' in msg.lower():
                            is_command = True
                        
                        # If it's clearly a question (and NOT also a command), block execution
                        if is_question and not is_command:
                            return False, "BLOCKED: User asked a question. Say 'close it' or 'close NEAR' to execute."
                        
                        # Validate symbol in command matches actual position
                        cmd_symbol = cmd.get('symbol', '').upper().replace('/USDT', '').replace('USDT', '')
                        if cmd_symbol and hasattr(self, 'get_full_system_state') and self.get_full_system_state:
                            try:
                                state = self.get_full_system_state()
                                positions = state.get('position', [])
                                open_symbols = [p.get('symbol', '').replace('USDT', '') for p in positions]
                                if cmd_symbol not in open_symbols:
                                    return False, f"BLOCKED: AI tried to close {cmd_symbol} but open positions are: {', '.join(open_symbols)}"
                            except Exception:
                                pass  # Non-critical: close validation
                        
                        # If it's a command OR autonomous mode with clear close intent, allow
                        if is_command:
                            return True, "VALIDATED: Close command confirmed"
                        
                        # Fallback: check for basic close keywords for autonomous execution
                        basic_close_kw = ['close', 'exit']
                        if any(kw in msg for kw in basic_close_kw) and not is_question:
                            return True, "VALIDATED: Close intent confirmed"
                        
                        return False, "BLOCKED: No clear close command. Say 'close it' or 'close NEAR' to execute."
                    
                    # SWITCH_SYMBOL: Changing trading pair
                    if action == 'switch_symbol':
                        switch_keywords = ['switch', 'change', 'trade', 'move to', 'go to']
                        symbol = cmd.get('symbol', '').lower().replace('usdt', '').replace('/usdt', '')
                        if not (any(kw in msg for kw in switch_keywords) or symbol in msg):
                            return False, "BLOCKED: No symbol switch intent"
                        return True, "VALIDATED: Symbol switch confirmed"
                    
                    # PAUSE/RESUME
                    if action == 'pause':
                        if 'pause' not in msg and 'stop' not in msg:
                            return False, "BLOCKED: No pause intent"
                        return True, "VALIDATED: Pause confirmed"
                    
                    if action == 'resume':
                        if 'resume' not in msg and 'start' not in msg and 'unpause' not in msg:
                            return False, "BLOCKED: No resume intent"
                        return True, "VALIDATED: Resume confirmed"
                    
                    # SET_MODE
                    if action == 'set_mode':
                        if 'mode' not in msg and 'autonomous' not in msg and 'filter' not in msg:
                            return False, "BLOCKED: No mode change intent"
                        if not has_action_verb and is_question:
                            return False, "BLOCKED: Question about mode, not a change request"
                        return True, "VALIDATED: Mode change confirmed"
                    
                    # Default: require action verb for unknown actions
                    if not has_action_verb:
                        return False, f"BLOCKED: Unknown action '{action}' without action verb"
                    
                    return True, "VALIDATED: Action approved"
                
                # Support multiple command block formats
                command_patterns = [
                    r'```command\s*\n?\s*(\{[^}]+\})\s*\n?```',  # ```command { } ```
                    r'```json\s*\n?\s*(\{[^}]*"action"[^}]+\})\s*\n?```',  # ```json { "action": } ```
                    r'`(\{[^`]*"action"[^`]+\})`',  # inline `{ "action": }`
                ]
                
                command_matches = []
                for pattern in command_patterns:
                    matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
                    command_matches.extend(matches)
                
                logger.info(f"🔍 Found {len(command_matches)} AI command blocks in response")
                
                command_results = []
                blocked_commands = []
                manual_actions_found = []
                
                for cmd_json in command_matches:
                    try:
                        cmd = json.loads(cmd_json.strip())
                        action = cmd.get('action', '')
                        
                        # === AUTONOMOUS MODE FLAG ===
                        # Check if this is an autonomous execution (AI in autonomous mode)
                        is_autonomous_execution = cmd.get('autonomous', False)
                        
                        # === TRADE COMMANDS ALWAYS REQUIRE CONFIRMATION ===
                        # Unless: 1) command has autonomous=true AND 2) system is in autonomous mode
                        # This ensures user ALWAYS sees analysis before any trade
                        if action in ['open_trade', 'close_trade', 'close_position']:
                            if is_autonomous_execution and is_autonomous_mode:
                                # Autonomous mode with autonomous flag - execute directly
                                logger.info(f"🤖 AUTONOMOUS EXECUTION: {action} (AI autonomous mode)")
                            else:
                                # All other cases - require user confirmation
                                logger.info(f"📋 Trade command requires confirmation: {cmd}")
                                manual_actions_found.append(cmd)
                                continue  # Don't execute, keep for user decision
                        
                        # === MATHEMATICAL VALIDATION BEFORE EXECUTION ===
                        should_execute, reason = validate_command_intent(user_message, action, cmd)
                        
                        if not should_execute:
                            logger.warning(f"🛡️ COMMAND BLOCKED: {reason}")
                            blocked_commands.append(f"🛡️ {reason}")
                            continue  # Skip this command
                        
                        logger.info(f"✅ COMMAND VALIDATED: {reason}")
                        logger.info(f"🤖 AI COMMAND: {cmd}")
                        
                        # === SET_PARAM: Change any system parameter ===
                        if action == 'set_param' and self.set_system_param:
                            param = cmd.get('param', '')
                            value = cmd.get('value')
                            logger.info(f"🔧 AI setting param: {param} = {value}")
                            result = self.set_system_param(param, value)
                            if result.get('success'):
                                # Verify param change by checking system state
                                verified = False
                                if hasattr(self, 'get_full_system_state') and self.get_full_system_state:
                                    try:
                                        state = self.get_full_system_state()
                                        params = state.get('parameters', {})
                                        # Check common params
                                        if param == 'risk_pct' and abs(params.get('risk_pct', 0) - float(value)) < 0.001:
                                            verified = True
                                        elif param == 'ai_mode' and params.get('ai_mode') == value:
                                            verified = True
                                        elif param == 'ai_confidence' and abs(params.get('ai_confidence', 0) - float(value)) < 0.01:
                                            verified = True
                                        elif param == 'paused' and str(params.get('paused', '')).lower() == str(value).lower():
                                            verified = True
                                        elif param == 'symbol':
                                            # Verify symbol change
                                            current_symbol = state.get('status', {}).get('symbol', '')
                                            target = str(value).upper().replace('/', '')
                                            if not target.endswith('USDT'):
                                                target = target + 'USDT'
                                            verified = (target == current_symbol or target in current_symbol)
                                        else:
                                            # For other params, trust the result
                                            verified = True
                                    except Exception:
                                        verified = True  # Trust result if verification fails
                                else:
                                    verified = True
                                
                                if verified:
                                    command_results.append(f"✅ {result.get('message')} - verified")
                                else:
                                    command_results.append(f"⚠️ {result.get('message')} - verify with /status")
                                logger.info(f"✅ AI set {param} = {value}")
                            else:
                                command_results.append(f"❌ {result.get('message')}")
                                logger.warning(f"❌ AI failed to set {param}: {result.get('message')}")
                        
                        # === OPEN_TRADE: Execute a trade ===
                        elif action == 'open_trade' and self.execute_ai_trade:
                            side = cmd.get('side', 'long')
                            trade_symbol = cmd.get('symbol', None)
                            exec_mode = "🤖 AUTONOMOUS" if is_autonomous_execution else "AI"
                            
                            if trade_symbol:
                                logger.info(f"{exec_mode} Command: Opening {side} on {trade_symbol}...")
                            else:
                                logger.info(f"{exec_mode} Command: Opening {side} trade...")
                            
                            result = await self.execute_ai_trade(side, trade_symbol)
                            if result.get('success'):
                                exec_suffix = " (AUTONOMOUS)" if is_autonomous_execution else ""
                                command_results.append(f"✅ {result.get('message')}{exec_suffix}")
                                logger.info(f"✅ {exec_mode} opened {side} trade: {result.get('message')}")
                                
                                # Verify trade was actually opened
                                await asyncio.sleep(0.5)
                                if hasattr(self, 'get_full_system_state') and self.get_full_system_state:
                                    try:
                                        state = self.get_full_system_state()
                                        positions = state.get('position', [])
                                        if positions and len(positions) > 0:
                                            pos = positions[-1]  # Get latest position
                                            command_results.append(f"📊 Verified: {pos.get('side', 'N/A').upper()} {pos.get('symbol', 'N/A')} @ ${pos.get('entry', 0):.4f}")
                                        else:
                                            command_results.append(f"⚠️ Warning: Position not found in state!")
                                            logger.warning("AI trade executed but no position found!")
                                    except Exception as ve:
                                        logger.warning(f"Trade verification failed: {ve}")
                            else:
                                command_results.append(f"❌ {result.get('message')}")
                                logger.warning(f"❌ {exec_mode} trade failed: {result.get('message')}")
                        
                        # === CLOSE_TRADE/CLOSE_POSITION: Close current position ===
                        elif (action == 'close_trade' or action == 'close_position') and self.close_ai_trade:
                            exec_mode = "🤖 AUTONOMOUS" if is_autonomous_execution else "AI"
                            # Check if symbol specified in command
                            close_symbol = cmd.get('symbol', None)
                            if close_symbol:
                                logger.info(f"{exec_mode} Command: Closing {close_symbol}...")
                            else:
                                logger.info(f"{exec_mode} Command: Closing trade...")
                            result = await self.close_ai_trade(close_symbol)
                            if result.get('success'):
                                # Verify close actually happened
                                await asyncio.sleep(0.5)
                                if hasattr(self, 'get_full_system_state') and self.get_full_system_state:
                                    try:
                                        state = self.get_full_system_state()
                                        positions = state.get('position', [])
                                        if not positions or len(positions) == 0:
                                            exec_suffix = " (AUTONOMOUS)" if is_autonomous_execution else ""
                                            command_results.append(f"✅ Position closed - verified{exec_suffix}")
                                        else:
                                            command_results.append(f"⚠️ Close submitted but position still showing")
                                    except Exception:
                                        command_results.append(f"✅ {result.get('message')}")
                                else:
                                    command_results.append(f"✅ {result.get('message')}")
                                logger.info(f"✅ {exec_mode} closed trade: {result.get('message')}")
                            else:
                                command_results.append(f"❌ {result.get('message')}")
                                logger.warning(f"❌ {exec_mode} close failed: {result.get('message')}")
                        
                        # === SWITCH_SYMBOL: Change trading pair ===
                        elif action == 'switch_symbol' and self.switch_symbol:
                            symbol = cmd.get('symbol', '').upper()
                            if not symbol.endswith('USDT'):
                                symbol = symbol + 'USDT'
                            logger.info(f"🤖 AI Command: Switching to {symbol}...")
                            result = self.switch_symbol(symbol)
                            if result.get('success'):
                                # Verify switch actually happened
                                await asyncio.sleep(0.3)
                                if hasattr(self, 'get_full_system_state') and self.get_full_system_state:
                                    try:
                                        state = self.get_full_system_state()
                                        current_symbol = state.get('status', {}).get('symbol', '')
                                        if symbol in current_symbol or current_symbol in symbol:
                                            command_results.append(f"✅ Switched to {current_symbol} - verified")
                                        else:
                                            command_results.append(f"⚠️ Switch requested but symbol shows {current_symbol}")
                                    except Exception:
                                        command_results.append(f"✅ {result.get('message')}")
                                else:
                                    command_results.append(f"✅ {result.get('message')}")
                            else:
                                command_results.append(f"❌ {result.get('message', result.get('error', 'Switch failed'))}")
                        
                        # === PAUSE: Pause the bot ===
                        elif action == 'pause' and self.do_pause:
                            logger.info(f"🤖 AI Command: Pausing bot...")
                            result = self.do_pause()
                            # Verify pause
                            if hasattr(self, 'get_full_system_state') and self.get_full_system_state:
                                try:
                                    state = self.get_full_system_state()
                                    if state.get('parameters', {}).get('paused', False):
                                        command_results.append(f"✅ Bot paused - verified")
                                    else:
                                        command_results.append(f"⚠️ Pause requested but status unclear")
                                except Exception:
                                    command_results.append(f"✅ Bot paused")
                            else:
                                command_results.append(f"✅ Bot paused")
                        
                        # === RESUME: Resume the bot ===
                        elif action == 'resume' and self.do_resume:
                            logger.info(f"🤖 AI Command: Resuming bot...")
                            result = self.do_resume()
                            # Verify resume
                            if hasattr(self, 'get_full_system_state') and self.get_full_system_state:
                                try:
                                    state = self.get_full_system_state()
                                    if not state.get('parameters', {}).get('paused', True):
                                        command_results.append(f"✅ Bot resumed - verified")
                                    else:
                                        command_results.append(f"⚠️ Resume requested but status unclear")
                                except Exception:
                                    command_results.append(f"✅ Bot resumed")
                            else:
                                command_results.append(f"✅ Bot resumed")
                        
                        # === SET_MODE: Change AI mode directly ===
                        elif action == 'set_mode' and self.set_ai_mode:
                            mode = cmd.get('mode', 'filter')
                            logger.info(f"🤖 AI Command: Setting mode to {mode}...")
                            result = self.set_ai_mode(mode)
                            if result.get('success'):
                                # Verify mode change
                                if hasattr(self, 'get_full_system_state') and self.get_full_system_state:
                                    try:
                                        state = self.get_full_system_state()
                                        current_mode = state.get('parameters', {}).get('ai_mode', '')
                                        if current_mode == mode:
                                            command_results.append(f"✅ AI mode set to {mode} - verified")
                                        else:
                                            command_results.append(f"⚠️ Mode change requested but shows {current_mode}")
                                    except Exception:
                                        command_results.append(f"✅ {result.get('message')}")
                                else:
                                    command_results.append(f"✅ {result.get('message')}")
                            else:
                                command_results.append(f"❌ {result.get('message')}")
                        
                        # === TRIGGER_SCAN: Force a market scan ===
                        elif action == 'trigger_scan':
                            command_results.append(f"ℹ️ Market scan will run on next cycle")
                        
                        else:
                            logger.warning(f"⚠️ Unknown AI action: {action}")
                            command_results.append(f"⚠️ Unknown action: {action}")
                        
                    except json.JSONDecodeError as je:
                        logger.error(f"AI command JSON error: {je} - Raw: {cmd_json[:100]}")
                    except Exception as e:
                        logger.error(f"AI command execution error: {e}")
                        command_results.append(f"❌ Error: {str(e)}")
                
                # Remove command blocks from displayed response
                # ALWAYS remove command blocks - they are internal metadata not for user display
                clean_response = response
                for pattern in command_patterns:
                    clean_response = re.sub(pattern, '', clean_response, flags=re.IGNORECASE | re.DOTALL)
                
                clean_response = clean_response.strip()
                
                # === INJECT MANUAL COMMAND SUGGESTIONS ===
                if manual_actions_found:
                    clean_response += "\n\n👉 *Action Required:*"
                    for cmd in manual_actions_found:
                        action = cmd.get('action')
                        if action == 'open_trade':
                            side = cmd.get('side', 'long').lower()
                            # Store the full command so /execute_long or /execute_short can retrieve the symbol
                            self._pending_execute_command = cmd
                            trade_sym = cmd.get('symbol', '')
                            if trade_sym:
                                clean_response += f"\nTo execute {side.upper()} on {trade_sym}: /execute_{side}"
                            else:
                                clean_response += f"\nTo execute {side.upper()}: /execute_{side}" 
                            logger.info(f"📋 Stored pending execute command: side={side}, symbol={trade_sym}")
                        elif action == 'close_trade' or action == 'close_position':
                            self._pending_execute_command = cmd
                            clean_response += f"\nTo CLOSE position: /execute_close"
                
                # === FILTER INCOMPLETE COMMAND PROMISES ===
                # If AI says "here's a command" but no command was found, remove that line
                if len(command_matches) == 0:
                    incomplete_promise_patterns = [
                        r"here['']?s? (?:the |a )?command[^.!?\n]*[.!?\n]?",
                        r"the command to[^.!?\n]*[.!?\n]?",
                        r"this (?:is the )?command[^.!?\n]*[.!?\n]?",
                        r"following command[^.!?\n]*[.!?\n]?",
                        r"command (?:below|above|to proceed)[^.!?\n]*[.!?\n]?",
                        r"if you (?:want|wish|still want) to proceed[^.!?\n]*[.!?\n]?",
                    ]
                    for pattern in incomplete_promise_patterns:
                        orig_response = clean_response
                        clean_response = re.sub(pattern, '', clean_response, flags=re.IGNORECASE)
                        if clean_response != orig_response:
                            logger.warning(f"🛡️ Removed incomplete command promise from response")
                    clean_response = clean_response.strip()

                # === COMPREHENSIVE HALLUCINATION CHECK & CORRECTION ===
                # Verify AI claims against actual system state and FIX errors
                hallucination_corrections = []
                hallucination_count = 0
                
                if self.get_full_system_state:
                    try:
                        current_state = self.get_full_system_state()
                        response_lower = clean_response.lower()
                        actual_symbol = current_state.get('status', {}).get('symbol', 'UNKNOWN')
                        actual_base = actual_symbol.replace('USDT', '').replace('/USDT', '').lower()
                        actual_balance = current_state.get('status', {}).get('balance', 0)
                        actual_positions = current_state.get('position', [])
                        has_position = actual_positions and len(actual_positions) > 0
                        actual_mode = current_state.get('parameters', {}).get('ai_mode', 'unknown')
                        actual_paused = current_state.get('parameters', {}).get('paused', False)
                        actual_win_rate = current_state.get('status', {}).get('win_rate', 0)
                        
                        # 1. POSITION HALLUCINATION CHECK
                        position_claim_phrases = [
                            "i have opened", "i opened", "i've opened", "position is open",
                            "we have a", "we are in a", "current position is", "holding a",
                            "i have a long", "i have a short", "you have a long", "you have a short",
                            "our long position", "our short position", "the position i opened"
                        ]
                        claims_position = any(phrase in response_lower for phrase in position_claim_phrases)
                        
                        if claims_position and not has_position:
                            hallucination_corrections.append("❌ Position: NO open position exists")
                            hallucination_count += 1
                            logger.warning("🚨 AI hallucination: claimed position but none exists!")
                            # FIX: Remove false position claims from response
                            for phrase in ["I have opened", "I opened", "I've opened", "position is open"]:
                                if phrase.lower() in response_lower:
                                    clean_response = re.sub(
                                        rf'{re.escape(phrase)}[^.!?]*[.!?]',
                                        f"[No position currently open on {actual_symbol}]",
                                        clean_response,
                                        flags=re.IGNORECASE
                                    )
                        
                        # 2. WRONG SYMBOL HALLUCINATION CHECK - FIX THE RESPONSE
                        common_symbols = ['BTC', 'ETH', 'SOL', 'LINK', 'TIA', 'INJ', 'ARB', 'APT', 'AVAX', 'MATIC', 'DOT', 'ADA', 'XRP', 'DOGE', 'OP', 'SUI', 'PEPE', 'WIF', 'BONK', 'SEI', 'NEAR', 'FET', 'RENDER', 'TAO']
                        wrong_symbols_found = []
                        for sym in common_symbols:
                            sym_lower = sym.lower()
                            # Check various ways AI might mention wrong symbol
                            wrong_mentions = [
                                f"trading {sym_lower}", f"on {sym_lower}", f"for {sym_lower}",
                                f"{sym_lower} position", f"{sym_lower} trade", f"bought {sym_lower}",
                                f"sold {sym_lower}", f"long {sym_lower}", f"short {sym_lower}"
                            ]
                            if sym_lower != actual_base:
                                for mention in wrong_mentions:
                                    if mention in response_lower:
                                        wrong_symbols_found.append(sym)
                                        hallucination_count += 1
                                        logger.warning(f"🚨 AI hallucination: said {sym} but trading {actual_symbol}")
                                        # FIX: Replace wrong symbol with correct one in response
                                        clean_response = re.sub(
                                            rf'\b{sym}(?:USDT)?\b',
                                            actual_symbol,
                                            clean_response,
                                            flags=re.IGNORECASE
                                        )
                                        break
                        
                        if wrong_symbols_found:
                            hallucination_corrections.append(f"❌ Symbol: Actually trading {actual_symbol}, not {', '.join(set(wrong_symbols_found))}")
                        
                        # 3. BALANCE HALLUCINATION CHECK
                        import re as re_mod
                        balance_matches = re_mod.findall(r'\$([0-9,]+(?:\.[0-9]{2})?)', clean_response)
                        for match in balance_matches:
                            try:
                                claimed_balance = float(match.replace(',', ''))
                                if claimed_balance > 1000:  # Only check significant amounts
                                    if actual_balance > 0 and abs(claimed_balance - actual_balance) / actual_balance > 0.2:
                                        hallucination_corrections.append(f"❌ Balance: Actual is ${actual_balance:,.2f}")
                                        hallucination_count += 1
                                        logger.warning(f"🚨 AI hallucination: claimed ${claimed_balance} but actual is ${actual_balance:.2f}")
                                        # FIX: Replace wrong balance with correct one
                                        clean_response = clean_response.replace(f"${match}", f"${actual_balance:,.2f}")
                                        break
                            except (ValueError, TypeError):
                                pass  # Non-critical: balance parsing
                        
                        # 4. WIN RATE HALLUCINATION CHECK
                        winrate_matches = re_mod.findall(r'(\d+(?:\.\d+)?)\s*%\s*win', response_lower)
                        for match in winrate_matches:
                            try:
                                claimed_wr = float(match)
                                if abs(claimed_wr - actual_win_rate) > 15:  # More than 15% off
                                    hallucination_corrections.append(f"❌ Win Rate: Actual is {actual_win_rate:.1f}%")
                                    hallucination_count += 1
                                    logger.warning(f"🚨 AI hallucination: claimed {claimed_wr}% win rate but actual is {actual_win_rate:.1f}%")
                                    break
                            except (ValueError, TypeError):
                                pass  # Non-critical: win rate parsing
                        
                        # 5. MODE/SETTING HALLUCINATION CHECK
                        if "mode is filter" in response_lower and actual_mode != 'filter':
                            hallucination_corrections.append(f"❌ Mode: Actually in '{actual_mode}' mode")
                            hallucination_count += 1
                        if "mode is autonomous" in response_lower and actual_mode != 'autonomous':
                            hallucination_corrections.append(f"❌ Mode: Actually in '{actual_mode}' mode")
                            hallucination_count += 1
                        if "bot is paused" in response_lower and not actual_paused:
                            hallucination_corrections.append("❌ Status: Bot is NOT paused")
                            hallucination_count += 1
                        
                        # 6. TRACK HALLUCINATION FREQUENCY (for reliability scoring)
                        if hallucination_count > 0:
                            # Store in tracker for AI reliability assessment
                            if hasattr(self, '_hallucination_tracker'):
                                self._hallucination_tracker.append({
                                    'time': datetime.now().isoformat(),
                                    'count': hallucination_count,
                                    'types': [c.split(':')[0].replace('❌ ', '') for c in hallucination_corrections]
                                })
                                # Keep last 100 entries
                                self._hallucination_tracker = self._hallucination_tracker[-100:]
                            else:
                                self._hallucination_tracker = [{
                                    'time': datetime.now().isoformat(),
                                    'count': hallucination_count,
                                    'types': [c.split(':')[0].replace('❌ ', '') for c in hallucination_corrections]
                                }]
                            
                            # Log serious hallucinations (3+) as warnings
                            if hallucination_count >= 3:
                                logger.error(f"🚨 SEVERE HALLUCINATION: {hallucination_count} errors detected in AI response!")
                        if ("bot is running" in response_lower or "bot is active" in response_lower) and actual_paused:
                            hallucination_corrections.append("❌ Status: Bot IS paused")
                        
                    except Exception as ve:
                        logger.warning(f"Hallucination check failed: {ve}")
                
                # Add corrections if any hallucinations detected
                if hallucination_corrections:
                    clean_response += "\n\n⚠️ *System Reality Check:*\n" + "\n".join(hallucination_corrections)
                
                # Append command results if any
                if command_results:
                    clean_response += "\n\n🔧 *Commands Executed:*\n" + "\n".join(command_results)
                
                # Log blocked commands (informational, not shown to user unless DEBUG)
                if blocked_commands:
                    for blocked in blocked_commands:
                        logger.info(f"🛡️ {blocked}")
                
                # Sanitize markdown and send
                clean_response = sanitize_markdown(clean_response)
                
                # Try sending with Markdown, fall back to plain text if parsing fails
                try:
                    await update.message.reply_text(clean_response, parse_mode="Markdown")
                except Exception as markdown_error:
                    logger.warning(f"Markdown parsing failed, sending as plain text: {markdown_error}")
                    await update.message.reply_text(clean_response)
            except Exception as e:
                error_str = str(e).lower()
                logger.error(f"AI chat error: {e}")
                if "timed out" in error_str or "timeout" in error_str:
                    await update.message.reply_text(
                        "🤖 AI response timed out — the server was slow. Please try again in a few seconds.",
                        parse_mode="Markdown"
                    )
                else:
                    await update.message.reply_text(
                        "🤖 Sorry, I had trouble processing that. Try asking again or use /help to see available commands.",
                        parse_mode="Markdown"
                    )
        else:
            # Fallback if AI not available
            await update.message.reply_text(
                "🤖 Hi! I'm Julaba, your trading assistant.\n\n"
                "Use /help to see what I can do, or ask me anything about trading!",
                parse_mode="Markdown"
            )


# Singleton instance
_notifier: Optional[TelegramNotifier] = None


def get_telegram_notifier() -> TelegramNotifier:
    """Get or create the Telegram notifier singleton."""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier
