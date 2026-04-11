import numpy as np 
import pandas as pd
import math
import src.wasserstein as ws
import src.metrics as mt
import matplotlib.pyplot as plt


def long_strat_unifortho(initial_capital, N_S, S, L, h1, h2, window_size, K=2, metric="CVaR", majority_lookback=7):
    """
    Implements a rolling regime-based trading strategy.
    Input: 
        - weekly_counts : number of weekly data points    
    Logic:
    - Analyzes 'window' size of data to determine the current regime.
    - If the detected regime is 'Bullish', go Long for the NEXT 'window' period.
    - If 'Bearish', stay Flat (0 position).
    """
    epsilon = 1e-6  

    # === FIX: Use percentage returns, equal-weighted ===
    pct_returns = S.pct_change().dropna()
    theta = np.ones((1, S.shape[1])) / S.shape[1]  # Equal weights for each asset
    portfolio_returns = pct_returns.dot(theta.T).sum(axis=1)  # Portfolio returns


    # Arrays to store results
    portfolio_value = [initial_capital]
    cum_pnl = [0.0]
    trade_signals = []
    
    num_steps = math.floor(len(S) / window_size)

    for i in range(num_steps - 1):
        
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        week_data = S.iloc[start_idx:end_idx, :]

        print(f'Analyzing Regime from {S.index[start_idx]} to {S.index[end_idx-1]} with {len(week_data)} data points.')

        if len(week_data) <= h1:
            print(f"Warning: Week {i} has {len(week_data)} points, which is too small for h1={h1}. STOP.")
            return np.array(portfolio_value), trade_signals, cum_pnl
            
        projected_emp, centroids, labels = ws.max_mccd_unifortho_sim(N_S, week_data, K, L, epsilon, h1, h2, metric)
        
        if majority_lookback > len(labels):
            current_regime = np.bincount(labels).argmax()
        else:
            current_regime = np.bincount(labels[-majority_lookback:]).argmax()
        
        if current_regime == 1:
            signal = 1
            print("Bullish ! Week", i+1, "go long")
        else:
            signal = -1
            print("Bearish!, Week", i+1, "go short")
      
        trade_signals.append(signal)

         #TODO Add something to only be able to trade at the open of the next available week day !!! 
        next_week_returns = portfolio_returns.iloc[end_idx : end_idx + window_size]

        # === FIX: Compounding with percentage returns ===
        for ret in next_week_returns:
            period_return = signal * ret
            new_value = portfolio_value[-1] * (1 + period_return)
            portfolio_value.append(new_value)
            cum_pnl.append(new_value - initial_capital)
        print("---" * 10)
        print(f'Portfolio value after week {i+1}: {portfolio_value[-1]}')
        print(f"AND :Cumulative P&L: {cum_pnl[-1]}")
        print("---" * 10)
    return np.array(portfolio_value), trade_signals, cum_pnl

def long_only(S, initial_capital):
    pct_returns = S.pct_change().dropna()
    theta = np.ones((1, S.shape[1])) / S.shape[1]  # Equal weights for each asset
    portfolio_returns = pct_returns.dot(theta.T).sum(axis=1)  # Portfolio returns
    cumulative = (1 + portfolio_returns).cumprod()
    portfolio_value = initial_capital * cumulative
    return portfolio_value, cumulative, portfolio_value.iloc[-1]

def short_only(S, initial_capital):
    #TODO output P&L after holding for the entire time the security S 
    pct_returns = S.pct_change().dropna()
    theta = np.ones((1, S.shape[1])) / S.shape[1]  # Equal weights for each asset
    portfolio_returns = -pct_returns.dot(theta.T).sum(axis=1)  # Short position: negative returns
    cumulative = (1 + portfolio_returns).cumprod()
    portfolio_value = initial_capital * cumulative
    return portfolio_value, cumulative, portfolio_value.iloc[-1]


def long_strat_implied(initial_capital, N_S, S, L, h1, h2, window_size, start_date = None, end_date = None, K=2, metric="CVaR", signal_type="conviction", entry_threshold=0.15, hold_threshold=0.10, lookback=5, use_gradient=False, gradient_weight=0.3, live_plot=False):
    epsilon = 1e-6

    # === FIX 1: Use percentage returns, equal-weighted across assets ===

    if start_date is not None and end_date is not None:
        S = S.loc[start_date:end_date]
    pct_returns = S.pct_change().dropna()
    #portfolio_returns = pct_returns.mean(axis=1)  # equal-weight average
    theta = np.ones((1, S.shape[1])) / S.shape[1] # Equal weights for each asset in the portfolio
    portfolio_returns = pct_returns.dot(theta.T).sum(axis=1) # Portfolio returns as a weighted sum

    # Arrays to store results
    portfolio_value = [initial_capital]
    cum_pnl = [0.0]
    trade_signals = []
    num_steps = math.floor(len(S) / window_size)
    switch_proba_history = []

    # For plotting live 
    if live_plot:
        plt.ion()
        fig, axes = plt.subplots(2, 3, figsize=(18, 9))
        fig.suptitle("Live Strategy Dashboard", fontsize=14, fontweight='bold')
        ax_pnl   = axes[0, 0]  # cumulative P&L
        ax_sp    = axes[0, 1]  # switch probability
        ax_trans = axes[1, 0]  # transition matrix heatmap
        ax_post  = axes[1, 1]  # posterior / proba_matrix
        ax_reg   = axes[1, 2]  # regime (bull/bear)
        ax_long  = axes[0, 2]  # long equal weighted on the currencies (optional)

        regime_history = []
        signal_history = []

        pct_returns_long = S.pct_change().dropna()
        theta_long = np.ones((1, S.shape[1])) / S.shape[1]  # Equal weights for each asset
        portfolio_returns_long = pct_returns_long.dot(theta_long.T).sum(axis=1)  # Portfolio returns
        cumulative_long= (1 + portfolio_returns_long).cumprod()
        portfolio_value_long = initial_capital * cumulative_long
    
    for i in range(num_steps - 1):

        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        week_data = S.iloc[start_idx:end_idx, :]
        
        print(f'Analyzing Regime from {S.index[start_idx]} to {S.index[end_idx-1]} with {len(week_data)} data points.')
        if len(week_data) <= h1:
            print(f"Warning: Week {i} has {len(week_data)} points, which is too small for h1={h1}. STOP.")
            return np.array(portfolio_value), trade_signals, cum_pnl, switch_proba_history

        projected_emp, centroids, labels = ws.max_mccd_unifortho_sim(N_S, week_data, K, L, epsilon, h1, h2, metric)

        proba_matrix, switch_proba, transition_matrix, posterior = ws.compute_implied_proba(projected_emp, centroids, labels, lookback=lookback, use_gradient=use_gradient, gradient_weight=gradient_weight)
        
        if not live_plot:
            print('---' * 10)
            print("Week", i + 1)
            print(f"Switch Probability: {switch_proba:.4f}")
            print(f"Transition Matrix:\n{transition_matrix}")
            print(f"Posterior Probabilities:\n{posterior}")
            print('---' * 10)

        for m in range(start_idx, end_idx):
            switch_proba_history.append(switch_proba)

        current_regime = np.bincount(labels[-h2:]).argmax()

        if signal_type == "continuous":
            signal = posterior[1] - posterior[0]
            dead_zone = 0.1

            if switch_proba > 0.5: 
                #in case the switch proba is extremely high we want to be more agressive
                signal = 1 * np.sign(signal)
            if abs(signal) < dead_zone:
                signal = 0.0
        if signal_type == "hysteresis":
            prev_signal = trade_signals[-1] if trade_signals else 0
            if current_regime == 1:
                if switch_proba >= entry_threshold and prev_signal >= 0:
                    signal = -1
                elif switch_proba < hold_threshold and prev_signal < 0:
                    signal = 1
                else:
                    signal = prev_signal
                    #signal = 0 
            else:
                if switch_proba >= entry_threshold and prev_signal <= 0:
                    signal = 1
                elif switch_proba < hold_threshold and prev_signal > 0:
                    signal = -1
                else:
                    signal = prev_signal
                    #signal = 0
        if signal_type == "conviction":
            regime_direction = 1 if current_regime == 1 else -1

            conviction = 1.0 - 1.5 * switch_proba # more agressive scalling of conviction
            #conviction = 1.0 - 2.0 * switch_proba
            signal = regime_direction * conviction
            if switch_proba > 0.5: 
                #in case the switch proba is extremely high we want to be more agressive
                signal = np.sign(signal) * 1.0

        trade_signals.append(signal)
        if not live_plot:
            print(f"Final signal: {signal}")

        # === FIX 2: Compounding portfolio value with percentage returns ===
        next_week_returns = portfolio_returns.iloc[end_idx: end_idx + window_size]

        for ret in next_week_returns:
            # signal scales exposure: 1.0 = fully long, 0.5 = half, -1 = short
            period_return = signal * ret
            new_value = portfolio_value[-1] * (1 + period_return)
            portfolio_value.append(new_value)
            cum_pnl.append(new_value - initial_capital)
        
        # === Live plot update ===
        if live_plot:
            regime_history.append(current_regime)
            signal_history.append(signal)
            weeks = list(range(1, len(regime_history) + 1))

            # 2) Cumulative P&L
            ax_pnl.clear()
            colors_pnl = ['green' if v >= 0 else 'red' for v in cum_pnl]
            ax_pnl.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                                where=[v >= 0 for v in cum_pnl], color='green', alpha=0.3)
            ax_pnl.fill_between(range(len(cum_pnl)), cum_pnl, 0,
                                where=[v < 0 for v in cum_pnl], color='red', alpha=0.3)
            ax_pnl.plot(cum_pnl, color='black', linewidth=1)
            ax_pnl.set_title("Cumulative P&L")
            ax_pnl.set_xlabel("Step")

            # 3) Switch probability over weeks
            ax_sp.clear()
            sp_per_week = [switch_proba_history[w * window_size] 
                           for w in range(len(regime_history))]
            ax_sp.bar(weeks, sp_per_week, color='orange', alpha=0.7)
            ax_sp.axhline(entry_threshold, color='red', linestyle='--', label='entry')
            ax_sp.axhline(hold_threshold, color='green', linestyle='--', label='hold')
            ax_sp.set_title("Switch Probability")
            ax_sp.set_ylim(0, 1)
            ax_sp.legend(fontsize=8)

            # 4) Transition matrix heatmap
            ax_trans.clear()
            im = ax_trans.imshow(transition_matrix, cmap='YlOrRd', vmin=0, vmax=1)
            for r in range(transition_matrix.shape[0]):
                for c in range(transition_matrix.shape[1]):
                    ax_trans.text(c, r, f"{transition_matrix[r, c]:.2f}",
                                 ha='center', va='center', fontsize=10)
            ax_trans.set_title("Transition Matrix")
            ax_trans.set_xticks(range(K))
            ax_trans.set_yticks(range(K))
            ax_trans.set_xlabel("To")
            ax_trans.set_ylabel("From")

            # 5) Posterior probabilities
            ax_post.clear()
            bars = ax_post.bar(range(len(posterior)), posterior,
                               color=['red', 'green'][:len(posterior)], alpha=0.7)
            ax_post.set_title("Posterior P(regime)")
            ax_post.set_xticks(range(len(posterior)))
            ax_post.set_xticklabels([f"R{k}" for k in range(len(posterior))])
            ax_post.set_ylim(0, 1)

            # 6) Regime history (bull/bear)
            ax_reg.clear()
            colors_reg = ['green' if r == 1 else 'red' for r in regime_history]
            ax_reg.bar(weeks, [1]*len(weeks), color=colors_reg, alpha=0.7)
            ax_reg.set_title("Regime: Green=Bull(1) / Red=Bear(0)")
            ax_reg.set_yticks([])
            # overlay signal as a line
            ax_reg_twin = ax_reg.twinx()
            ax_reg_twin.plot(weeks, signal_history, color='blue', marker='o',
                             markersize=3, linewidth=1, label='signal')
            ax_reg_twin.set_ylabel("Signal", color='blue')
            ax_reg_twin.set_ylim(-1.5, 1.5)

            #7) Plot FX Prices 
            ax_long.clear()
            ax_long.plot(portfolio_value_long[:end_idx+window_size], color='blue', linewidth=1, label='Long Equal-Weighted')
            #differente
            ax_long.set_title("Long Equal-Weighted Portfolio Value")
            ax_long.set_xlabel("Date")
            #rotate x-axis labels for better readability
            ax_long.set_xticks(portfolio_value_long.index[:end_idx+window_size][::max(1, len(portfolio_value_long[:end_idx+window_size])//10)])
            ax_long.set_xticklabels(portfolio_value_long.index[:end_idx+window_size][::max(1, len(portfolio_value_long[:end_idx+window_size])//10)].strftime('%Y-%m-%d'), rotation=45)
            ax_long.legend(fontsize=8)

            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)
        
        if not live_plot:
            print("---" * 10)
            print(f'Portfolio value after week {i+1}: {portfolio_value[-1]}')
            print(f"AND :Cumulative P&L: {cum_pnl[-1]}")
            print("---" * 10)

    if live_plot:
        plt.ioff()
        plt.show()
    return np.array(portfolio_value), trade_signals, cum_pnl, switch_proba_history