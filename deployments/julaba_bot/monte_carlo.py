"""
Monte Carlo Simulation Module for Julaba
Simulates thousands of possible trade sequences to estimate risk of ruin and expected ranges.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger("Julaba.MonteCarlo")


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    n_simulations: int
    n_trades_per_sim: int
    
    # Equity outcomes
    final_balances: List[float]
    median_final_balance: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    
    # Risk metrics
    prob_profit: float
    prob_ruin: float  # Probability of losing 50%+
    prob_drawdown_20: float
    max_drawdown_median: float
    max_drawdown_95: float
    
    # Return distribution
    median_return: float
    expected_return: float
    return_std: float
    worst_case: float
    best_case: float
    
    # Equity curves
    equity_curves: List[List[float]]


class MonteCarloSimulator:
    """
    Monte Carlo simulator for trading strategy risk analysis.
    
    Simulates thousands of possible trade sequences based on historical
    win rate, average win/loss, and distribution statistics.
    
    This answers: "What's the range of realistic outcomes?"
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_per_trade: float = 0.02
    ):
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        
    def _generate_trade_outcome(
        self,
        win_rate: float,
        avg_win_r: float,
        avg_loss_r: float,
        win_std: float = 0.3,
        loss_std: float = 0.2
    ) -> float:
        """
        Generate a single trade outcome using distribution statistics.
        
        Returns R-multiple of the trade.
        """
        # Decide if win or loss
        is_win = np.random.random() < win_rate
        
        if is_win:
            # Sample from win distribution (normal around avg_win)
            r_outcome = np.random.normal(avg_win_r, win_std)
            r_outcome = max(r_outcome, 0.1)  # Wins are at least 0.1R
        else:
            # Sample from loss distribution (normal around -avg_loss)
            r_outcome = -abs(np.random.normal(avg_loss_r, loss_std))
            r_outcome = min(r_outcome, -0.1)  # Losses are at least -0.1R
            r_outcome = max(r_outcome, -1.0)  # Cap at -1R (stop loss)
        
        return r_outcome
    
    def simulate(
        self,
        n_simulations: int,
        n_trades: int,
        win_rate: float,
        avg_win_r: float,
        avg_loss_r: float,
        win_std: float = 0.3,
        loss_std: float = 0.2,
        compounding: bool = True,
        max_drawdown_limit: float = 0.5  # 50% = ruin
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.
        
        Args:
            n_simulations: Number of simulation runs
            n_trades: Number of trades per simulation
            win_rate: Historical win rate (0-1)
            avg_win_r: Average winning R-multiple
            avg_loss_r: Average losing R-multiple (positive)
            win_std: Standard deviation of wins
            loss_std: Standard deviation of losses
            compounding: Whether to compound returns
            max_drawdown_limit: Drawdown threshold for "ruin"
        
        Returns:
            MonteCarloResult with comprehensive statistics
        """
        logger.info(f"Starting Monte Carlo: {n_simulations} sims × {n_trades} trades")
        logger.info(f"Parameters: WR={win_rate:.1%}, Avg Win={avg_win_r:.2f}R, Avg Loss={avg_loss_r:.2f}R")
        
        final_balances = []
        max_drawdowns = []
        equity_curves = []
        ruined_count = 0
        dd20_count = 0
        
        for sim in range(n_simulations):
            balance = self.initial_balance
            peak_balance = balance
            equity_curve = [balance]
            max_dd = 0.0
            
            for trade in range(n_trades):
                # Generate trade outcome
                r_outcome = self._generate_trade_outcome(
                    win_rate, avg_win_r, avg_loss_r, win_std, loss_std
                )
                
                # Calculate P&L
                if compounding:
                    risk_amount = balance * self.risk_per_trade
                else:
                    risk_amount = self.initial_balance * self.risk_per_trade
                
                pnl = risk_amount * r_outcome
                balance += pnl
                
                # Track equity
                equity_curve.append(balance)
                
                # Update drawdown
                if balance > peak_balance:
                    peak_balance = balance
                
                drawdown = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
                max_dd = max(max_dd, drawdown)
                
                # Check for ruin
                if drawdown >= max_drawdown_limit:
                    ruined_count += 1
                    break  # Stop this simulation
                
                # Check for 20% drawdown
                if drawdown >= 0.20:
                    dd20_count += 1
                    # Don't break, just count
            
            final_balances.append(balance)
            max_drawdowns.append(max_dd)
            equity_curves.append(equity_curve)
            
            if (sim + 1) % 100 == 0:
                logger.debug(f"Completed {sim + 1}/{n_simulations} simulations")
        
        # Calculate statistics
        final_balances = np.array(final_balances)
        max_drawdowns = np.array(max_drawdowns)
        
        result = MonteCarloResult(
            n_simulations=n_simulations,
            n_trades_per_sim=n_trades,
            final_balances=final_balances.tolist(),
            median_final_balance=float(np.median(final_balances)),
            percentile_5=float(np.percentile(final_balances, 5)),
            percentile_25=float(np.percentile(final_balances, 25)),
            percentile_75=float(np.percentile(final_balances, 75)),
            percentile_95=float(np.percentile(final_balances, 95)),
            prob_profit=float((final_balances > self.initial_balance).sum() / n_simulations),
            prob_ruin=float(ruined_count / n_simulations),
            prob_drawdown_20=float(dd20_count / n_simulations),
            max_drawdown_median=float(np.median(max_drawdowns)),
            max_drawdown_95=float(np.percentile(max_drawdowns, 95)),
            median_return=float((np.median(final_balances) / self.initial_balance - 1) * 100),
            expected_return=float((final_balances.mean() / self.initial_balance - 1) * 100),
            return_std=float(((final_balances / self.initial_balance - 1) * 100).std()),
            worst_case=float(final_balances.min()),
            best_case=float(final_balances.max()),
            equity_curves=equity_curves
        )
        
        return result
    
    def print_results(self, result: MonteCarloResult):
        """Print formatted Monte Carlo results."""
        print("\n" + "="*70)
        print("🎲 MONTE CARLO SIMULATION RESULTS")
        print("="*70)
        
        print(f"\n📊 Simulation Parameters:")
        print(f"  Simulations:        {result.n_simulations:,}")
        print(f"  Trades per sim:     {result.n_trades_per_sim:,}")
        print(f"  Initial Balance:    ${self.initial_balance:,.2f}")
        
        print(f"\n💰 Final Balance Distribution:")
        print(f"  5th Percentile:     ${result.percentile_5:,.2f}")
        print(f"  25th Percentile:    ${result.percentile_25:,.2f}")
        print(f"  Median (50th):      ${result.median_final_balance:,.2f}")
        print(f"  75th Percentile:    ${result.percentile_75:,.2f}")
        print(f"  95th Percentile:    ${result.percentile_95:,.2f}")
        
        print(f"\n📈 Return Statistics:")
        print(f"  Expected Return:    {result.expected_return:+.2f}%")
        print(f"  Median Return:      {result.median_return:+.2f}%")
        print(f"  Std Deviation:      {result.return_std:.2f}%")
        print(f"  Best Case:          ${result.best_case:,.2f} ({(result.best_case/self.initial_balance-1)*100:+.1f}%)")
        print(f"  Worst Case:         ${result.worst_case:,.2f} ({(result.worst_case/self.initial_balance-1)*100:+.1f}%)")
        
        print(f"\n⚠️ Risk Analysis:")
        print(f"  Probability of Profit:      {result.prob_profit:.1%}")
        print(f"  Probability of Ruin (50%):  {result.prob_ruin:.2%}")
        print(f"  Prob 20%+ Drawdown:         {result.prob_drawdown_20:.1%}")
        print(f"  Median Max Drawdown:        {result.max_drawdown_median:.1%}")
        print(f"  95th Percentile DD:         {result.max_drawdown_95:.1%}")
        
        # Risk assessment
        print(f"\n🎯 Risk Assessment:")
        if result.prob_ruin < 0.01:
            print(f"  ✅ EXCELLENT - Ruin risk < 1%")
        elif result.prob_ruin < 0.05:
            print(f"  ✅ GOOD - Ruin risk < 5%")
        elif result.prob_ruin < 0.10:
            print(f"  ⚠️  MODERATE - Ruin risk < 10%")
        else:
            print(f"  🚨 HIGH RISK - Ruin risk >= 10%")
        
        if result.prob_profit > 0.80:
            print(f"  ✅ High profit probability ({result.prob_profit:.0%})")
        elif result.prob_profit > 0.60:
            print(f"  ⚠️  Moderate profit probability ({result.prob_profit:.0%})")
        else:
            print(f"  🚨 Low profit probability ({result.prob_profit:.0%})")
        
        print("\n" + "="*70 + "\n")
    
    def plot_results(self, result: MonteCarloResult, output_path: Path = None):
        """Generate visualization of Monte Carlo results."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle('Monte Carlo Simulation Results', fontsize=16, fontweight='bold')
            
            # 1. Distribution of final balances (histogram)
            ax1 = axes[0, 0]
            ax1.hist(result.final_balances, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax1.axvline(result.median_final_balance, color='red', linestyle='--', linewidth=2, label='Median')
            ax1.axvline(self.initial_balance, color='green', linestyle='--', linewidth=2, label='Initial')
            ax1.set_xlabel('Final Balance ($)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of Final Balances')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # 2. Sample equity curves
            ax2 = axes[0, 1]
            n_curves = min(100, len(result.equity_curves))
            for i in range(n_curves):
                curve = result.equity_curves[i]
                ax2.plot(curve, alpha=0.1, color='steelblue')
            # Plot median curve
            median_idx = np.argsort(result.final_balances)[len(result.final_balances)//2]
            ax2.plot(result.equity_curves[median_idx], color='red', linewidth=2, label='Median')
            ax2.axhline(self.initial_balance, color='green', linestyle='--', linewidth=1, label='Initial')
            ax2.set_xlabel('Trade Number')
            ax2.set_ylabel('Balance ($)')
            ax2.set_title(f'Sample Equity Curves (n={n_curves})')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            # 3. Return distribution (box plot)
            ax3 = axes[1, 0]
            returns = [(b / self.initial_balance - 1) * 100 for b in result.final_balances]
            ax3.boxplot(returns, vert=True)
            ax3.axhline(0, color='green', linestyle='--', linewidth=1)
            ax3.set_ylabel('Return (%)')
            ax3.set_title('Return Distribution')
            ax3.grid(alpha=0.3)
            
            # 4. Key metrics table
            ax4 = axes[1, 1]
            ax4.axis('off')
            metrics_text = f"""
Monte Carlo Summary
{'='*30}

Simulations:  {result.n_simulations:,}
Trades/Sim:   {result.n_trades_per_sim:,}

Expected Return:   {result.expected_return:+.2f}%
Median Return:     {result.median_return:+.2f}%
Return Std Dev:    {result.return_std:.2f}%

Prob of Profit:    {result.prob_profit:.1%}
Prob of Ruin:      {result.prob_ruin:.2%}
Prob 20%+ DD:      {result.prob_drawdown_20:.1%}

Median Max DD:     {result.max_drawdown_median:.1%}
95th %ile DD:      {result.max_drawdown_95:.1%}

5th Percentile:    ${result.percentile_5:,.0f}
Median:            ${result.median_final_balance:,.0f}
95th Percentile:   ${result.percentile_95:,.0f}
            """
            ax4.text(0.1, 0.9, metrics_text, transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                logger.info(f"Monte Carlo plot saved: {output_path}")
            else:
                output_path = Path("monte_carlo_results.png")
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                logger.info(f"Monte Carlo plot saved: {output_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to generate Monte Carlo plot: {e}")


# Singleton instance
_monte_carlo: MonteCarloSimulator = None

def get_monte_carlo(initial_balance: float = 10000.0, risk_per_trade: float = 0.02) -> MonteCarloSimulator:
    """Get or create Monte Carlo simulator instance."""
    global _monte_carlo
    if _monte_carlo is None:
        _monte_carlo = MonteCarloSimulator(initial_balance, risk_per_trade)
    return _monte_carlo
