"""
Chart Generator for Julaba
Creates candlestick charts with trade markers for Telegram notifications.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import io

logger = logging.getLogger("Julaba.Charts")

# Chart dependencies (optional)
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not installed. Run: pip install matplotlib")

try:
    import mplfinance as mpf
    MPLFINANCE_AVAILABLE = True
except ImportError:
    MPLFINANCE_AVAILABLE = False
    logger.debug("mplfinance not installed. Using basic charts.")

import pandas as pd
import numpy as np


# Dark theme style for charts
DARK_STYLE = {
    'base_mpl_style': 'dark_background',
    'marketcolors': {
        'candle': {'up': '#00ff88', 'down': '#ff4444'},
        'edge': {'up': '#00ff88', 'down': '#ff4444'},
        'wick': {'up': '#00ff88', 'down': '#ff4444'},
        'ohlc': {'up': '#00ff88', 'down': '#ff4444'},
        'volume': {'up': '#00aa55', 'down': '#aa2222'},
        'vcedge': {'up': '#00aa55', 'down': '#aa2222'},
        'vcdopcod': False,
        'alpha': 0.9
    },
    'mavcolors': ['#00d4ff', '#ffaa00', '#ff00ff'],
    'facecolor': '#1a1a2e',
    'gridcolor': '#333344',
    'gridstyle': '--',
    'y_on_right': True,
    'rc': {
        'axes.labelcolor': '#cccccc',
        'axes.edgecolor': '#444444',
        'axes.titlecolor': '#ffffff',
        'xtick.color': '#888888',
        'ytick.color': '#888888',
        'figure.facecolor': '#1a1a2e',
        'axes.facecolor': '#1a1a2e',
        'savefig.facecolor': '#1a1a2e',
    }
}


class ChartGenerator:
    """
    Generates trading charts for Telegram notifications.
    
    Features:
    - Candlestick charts with volume
    - Trade entry/exit markers
    - Support/resistance levels
    - SMA overlays
    - Dark theme optimized for Telegram
    """
    
    def __init__(self):
        self.output_dir = Path(__file__).parent / "charts"
        self.output_dir.mkdir(exist_ok=True)
        
        if MPLFINANCE_AVAILABLE:
            try:
                self.style = mpf.make_mpf_style(**DARK_STYLE)
            except Exception as e:
                logger.warning(f"Failed to create custom style: {e}")
                self.style = mpf.make_mpf_style(base_mpf_style='nightclouds')
        
        logger.debug(f"ChartGenerator initialized | matplotlib: {MATPLOTLIB_AVAILABLE} | mplfinance: {MPLFINANCE_AVAILABLE}")
    
    def generate_candlestick_chart(
        self,
        df: pd.DataFrame,
        symbol: str = "LINK/USDT",
        title: str = None,
        entry_price: float = None,
        entry_side: str = None,
        stop_loss: float = None,
        take_profits: List[float] = None,
        show_sma: bool = True,
        bars: int = 50
    ) -> Optional[bytes]:
        """
        Generate a candlestick chart as PNG bytes.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair symbol
            title: Chart title (default: auto-generated)
            entry_price: Trade entry price (for marker)
            entry_side: 'long' or 'short'
            stop_loss: Stop loss level
            take_profits: List of take profit levels [TP1, TP2, TP3]
            show_sma: Show SMA lines
            bars: Number of bars to show
        
        Returns:
            PNG image as bytes, or None if failed
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Cannot generate chart - matplotlib not installed")
            return None
        
        if df is None or len(df) < 5:
            logger.warning("Insufficient data for chart")
            return None
        
        try:
            # Prepare data
            df = df.copy().tail(bars)
            
            # Ensure proper column names and index
            df.columns = df.columns.str.lower()
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('timestamp')
            elif not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Ensure required columns exist
            required = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required):
                logger.error(f"Missing required columns. Have: {df.columns.tolist()}")
                return None
            
            if MPLFINANCE_AVAILABLE:
                return self._generate_with_mplfinance(
                    df, symbol, title, entry_price, entry_side,
                    stop_loss, take_profits, show_sma
                )
            else:
                return self._generate_with_matplotlib(
                    df, symbol, title, entry_price, entry_side,
                    stop_loss, take_profits, show_sma
                )
                
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _generate_with_mplfinance(
        self,
        df: pd.DataFrame,
        symbol: str,
        title: str,
        entry_price: float,
        entry_side: str,
        stop_loss: float,
        take_profits: List[float],
        show_sma: bool
    ) -> Optional[bytes]:
        """Generate chart using mplfinance library."""
        # Store original data length for entry marker
        original_len = len(df)
        prediction_bars = 30  # Empty bars for future prediction space
        
        # Build horizontal lines dict for mplfinance
        hlines_dict = None
        hlines_list = []
        hlines_colors = []
        
        if entry_price:
            hlines_list.append(entry_price)
            hlines_colors.append('#00d4ff')
        
        if stop_loss:
            hlines_list.append(stop_loss)
            hlines_colors.append('#ff4444')
        
        if take_profits:
            for tp in take_profits:
                if tp:
                    hlines_list.append(tp)
                    hlines_colors.append('#00ff88')
        
        if hlines_list:
            hlines_dict = dict(hlines=hlines_list, colors=hlines_colors, linestyle='--', linewidths=1.5)
        
        # Create figure
        fig, axes = mpf.plot(
            df,
            type='candle',
            style=self.style,
            volume='volume' in df.columns,
            mav=(15, 40) if show_sma else (),
            title=title or f"{symbol} Chart",
            ylabel='Price',
            ylabel_lower='Volume',
            figsize=(12, 8),
            returnfig=True,
            hlines=hlines_dict
        )
        
        # Add entry marker
        if entry_price and entry_side:
            ax = axes[0]
            marker = '^' if entry_side.lower() == 'long' else 'v'
            color = '#00ff88' if entry_side.lower() == 'long' else '#ff4444'
            
            # Use original data length for entry marker position (last candle)
            x_pos = original_len - 1
            ax.scatter(x_pos, entry_price, marker=marker, s=200, color=color, 
                      edgecolors='white', linewidths=2, zorder=10)
            ax.annotate(
                f"ENTRY\n${entry_price:.4f}",
                (x_pos, entry_price),
                xytext=(10, 20 if entry_side.lower() == 'long' else -20),
                textcoords='offset points',
                fontsize=9,
                color='white',
                ha='left'
            )
        
        # Add prediction space by extending x-axis
        ax = axes[0]
        ax.set_xlim(-0.5, original_len + prediction_bars - 0.5)
        # If there's a volume subplot, also extend its x-axis
        if len(axes) > 1:
            axes[1].set_xlim(-0.5, original_len + prediction_bars - 0.5)
        
        # Save to bytes (don't use bbox_inches='tight' or tight_layout as it crops the prediction space)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150,
                   facecolor='#1a1a2e', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        
        return buf.read()
    
    def _generate_with_matplotlib(
        self,
        df: pd.DataFrame,
        symbol: str,
        title: str,
        entry_price: float,
        entry_side: str,
        stop_loss: float,
        take_profits: List[float],
        show_sma: bool
    ) -> Optional[bytes]:
        """Generate chart using basic matplotlib."""
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        
        # Plot candlesticks manually
        for i, (idx, row) in enumerate(df.iterrows()):
            color = '#00ff88' if row['close'] >= row['open'] else '#ff4444'
            
            # Wick
            ax.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
            
            # Body
            body_bottom = min(row['open'], row['close'])
            body_height = abs(row['close'] - row['open'])
            rect = Rectangle((i - 0.3, body_bottom), 0.6, body_height,
                            facecolor=color, edgecolor=color)
            ax.add_patch(rect)
        
        # Add SMAs
        if show_sma and len(df) >= 40:
            sma15 = df['close'].rolling(15).mean()
            sma40 = df['close'].rolling(40).mean()
            ax.plot(range(len(df)), sma15, color='#00d4ff', linewidth=1.5, label='SMA15')
            ax.plot(range(len(df)), sma40, color='#ffaa00', linewidth=1.5, label='SMA40')
            ax.legend(loc='upper left', facecolor='#1a1a2e', edgecolor='#444')
        
        # Add horizontal lines
        if entry_price:
            ax.axhline(y=entry_price, color='#00d4ff', linestyle='--', linewidth=1.5, label='Entry')
        if stop_loss:
            ax.axhline(y=stop_loss, color='#ff4444', linestyle='--', linewidth=1, label='SL')
        if take_profits:
            for i, tp in enumerate(take_profits):
                if tp:
                    ax.axhline(y=tp, color='#00ff88', linestyle=':', linewidth=1,
                              label=f'TP{i+1}' if i == 0 else '')
        
        # Add entry marker (same as mplfinance version)
        if entry_price and entry_side:
            marker = '^' if entry_side.lower() == 'long' else 'v'
            color = '#00ff88' if entry_side.lower() == 'long' else '#ff4444'
            x_pos = len(df) - 1
            ax.scatter(x_pos, entry_price, marker=marker, s=200, color=color,
                      edgecolors='white', linewidths=2, zorder=10)
            ax.annotate(
                f"ENTRY\n${entry_price:.4f}",
                (x_pos, entry_price),
                xytext=(10, 20 if entry_side.lower() == 'long' else -20),
                textcoords='offset points',
                fontsize=9,
                color='white',
                ha='left'
            )
        
        # Add prediction space (30 empty bars at the end for future predictions)
        prediction_bars = 30
        ax.set_xlim(-0.5, len(df) - 1 + prediction_bars + 0.5)
        
        # Styling
        ax.set_title(title or f"{symbol} Chart", color='white', fontsize=14)
        ax.set_ylabel('Price', color='#888888')
        ax.tick_params(colors='#888888')
        ax.spines['bottom'].set_color('#444444')
        ax.spines['top'].set_color('#444444')
        ax.spines['left'].set_color('#444444')
        ax.spines['right'].set_color('#444444')
        ax.grid(True, linestyle='--', alpha=0.3, color='#444444')
        
        # Save to bytes (don't use bbox_inches='tight' or tight_layout as it crops the prediction space)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150,
                   facecolor='#1a1a2e', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        
        return buf.read()
    
    def generate_equity_chart(
        self,
        equity_curve: List[float],
        title: str = "Equity Curve"
    ) -> Optional[bytes]:
        """Generate an equity curve chart."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if not equity_curve or len(equity_curve) < 2:
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a1a2e')
            ax.set_facecolor('#1a1a2e')
            
            # Determine color based on overall performance
            color = '#00ff88' if equity_curve[-1] >= equity_curve[0] else '#ff4444'
            
            ax.fill_between(range(len(equity_curve)), equity_curve,
                           alpha=0.3, color=color)
            ax.plot(equity_curve, color=color, linewidth=2)
            
            # Add initial balance line
            ax.axhline(y=equity_curve[0], color='#888888', linestyle='--',
                      linewidth=1, alpha=0.5)
            
            ax.set_title(title, color='white', fontsize=14)
            ax.set_ylabel('Balance ($)', color='#888888')
            ax.set_xlabel('Trades', color='#888888')
            ax.tick_params(colors='#888888')
            ax.spines['bottom'].set_color('#444444')
            ax.spines['top'].set_color('#444444')
            ax.spines['left'].set_color('#444444')
            ax.spines['right'].set_color('#444444')
            ax.grid(True, linestyle='--', alpha=0.3, color='#444444')
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                       facecolor='#1a1a2e', edgecolor='none')
            plt.close(fig)
            buf.seek(0)
            
            return buf.read()
            
        except Exception as e:
            logger.error(f"Equity chart generation failed: {e}")
            return None
    
    def generate_pnl_distribution(
        self,
        trade_pnls: List[float],
        title: str = "P&L Distribution"
    ) -> Optional[bytes]:
        """Generate a P&L distribution histogram."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if not trade_pnls or len(trade_pnls) < 5:
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a1a2e')
            ax.set_facecolor('#1a1a2e')
            
            # Split into wins and losses
            wins = [p for p in trade_pnls if p > 0]
            losses = [p for p in trade_pnls if p <= 0]
            
            # Create histogram bins
            all_pnls = np.array(trade_pnls)
            bins = np.linspace(all_pnls.min(), all_pnls.max(), 20)
            
            ax.hist(wins, bins=bins, color='#00ff88', alpha=0.7, label=f'Wins ({len(wins)})')
            ax.hist(losses, bins=bins, color='#ff4444', alpha=0.7, label=f'Losses ({len(losses)})')
            
            ax.axvline(x=0, color='white', linestyle='-', linewidth=1)
            ax.axvline(x=np.mean(trade_pnls), color='#00d4ff', linestyle='--',
                      linewidth=2, label=f'Mean: ${np.mean(trade_pnls):.2f}')
            
            ax.set_title(title, color='white', fontsize=14)
            ax.set_xlabel('P&L ($)', color='#888888')
            ax.set_ylabel('Frequency', color='#888888')
            ax.tick_params(colors='#888888')
            ax.legend(facecolor='#1a1a2e', edgecolor='#444', labelcolor='white')
            ax.spines['bottom'].set_color('#444444')
            ax.spines['top'].set_color('#444444')
            ax.spines['left'].set_color('#444444')
            ax.spines['right'].set_color('#444444')
            ax.grid(True, linestyle='--', alpha=0.3, color='#444444')
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                       facecolor='#1a1a2e', edgecolor='none')
            plt.close(fig)
            buf.seek(0)
            
            return buf.read()
            
        except Exception as e:
            logger.error(f"P&L distribution chart failed: {e}")
            return None


# Singleton instance
_chart_generator: Optional[ChartGenerator] = None


def get_chart_generator() -> ChartGenerator:
    """Get the global chart generator instance."""
    global _chart_generator
    if _chart_generator is None:
        _chart_generator = ChartGenerator()
    return _chart_generator
