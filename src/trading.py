import numpy as np 
import pandas as pd
import math
import src.wasserstein as ws
import src.metrics as mt
import matplotlib.pyplot as plt

debug = False 

def labels_to_signal_series(signal_values, return_index, h1, h2):
    """
    Map per-lifted-window signals to a per-return-date signal series.
 
    Window m covers return positions [m*h2, m*h2 + h1 - 1] and therefore ENDS at
    position m*h2 + h1 - 1. Each signal is stamped on its window's end date and
    forward-filled until the next window end, so the signal at any date t comes
    from a window ending at or before t (staleness <= h2 - 1; == 0 when h2 == 1).
    Dates before the first window end carry NaN (nothing has completed yet).
    """
    signal_values = np.asarray(signal_values, dtype=float)
    end_pos = np.arange(len(signal_values)) * h2 + h1 - 1
    if len(end_pos) and end_pos[-1] >= len(return_index):
        raise ValueError(
            f"label windows overrun the return series: last window ends at "
            f"position {end_pos[-1]} but only {len(return_index)} returns exist "
            f"(check that len(labels) matches the lifting of THIS price series)")
    sig = pd.Series(signal_values, index=return_index[end_pos])
    return sig.reindex(return_index).ffill()
 

def compute_portfolio_returns(S, weighting="inverse_vol", window_size=5, eps=1e-6):
    """Single source of truth for portfolio returns. Date-indexed, no look-ahead.
    - equal:       constant weights, only pct_change warm-up (1 row) lost.
    - inverse_vol: weights known at t-1 (shift(1)); first window_size rows lost to vol estimate.
    Returns a date-indexed Series so callers select next windows by .reindex(dates).
    """
    r = S.pct_change().dropna()
    if weighting == "equal":
        theta = pd.DataFrame(1.0 / S.shape[1], index=r.index, columns=r.columns)
    elif weighting == "inverse_vol":
        vol = r.rolling(window_size).std()
        inv = 1.0 / (vol + eps)
        theta = inv.div(inv.sum(axis=1), axis=0).shift(1)   # causal weights
    else:
        raise ValueError(f"unknown weighting {weighting!r}")
    return (r * theta).sum(axis=1, min_count=1).dropna()
    

def look_ahead_strat_unifortho(initial_capital, N_S, S, L, h1, h2, window_size, K=2, metric="CVaR", weighting = "inverse_vol"):
    """
    Implements a rolling regime-based trading strategy with look-ahead bias (for testing purposes).
    Input: 
        - weekly_counts : number of weekly data points
    """

    epsilon = 1e-6
    
    
    portfolio_returns = compute_portfolio_returns(S, weighting=weighting, window_size=window_size, eps=epsilon)

    value_dates = [S.index[0]]          # initial capital anchored at the first date (pre-trading)
    portfolio_value = [initial_capital]
    signal_series = pd.Series(index = portfolio_returns.index, dtype = float)

    projected_emp, centroids, labels = ws.max_mccd_unifortho_sim(N_S, S, K, L, epsilon, h1, h2, metric)
        
        
    rets = S.pct_change().dropna()
    N = rets.shape[0]

    signal_label = labels_to_signal_series(
            labels,
            rets.index,
            h1,
            h2,
        ) 
        
    common = rets.index.intersection(signal_series.index)

    signal_series.loc[common] = signal_label
            
    signal_series = signal_series.ffill()
    signal_series, pf = signal_series.align(portfolio_returns, join="inner")
    strat_ret = signal_series * pf

    pv_vals = initial_capital * (1 + strat_ret).cumprod()
    pv_series  = pd.Series(
        np.concatenate([[initial_capital], pv_vals.values]),
        index=pd.DatetimeIndex([S.index[0]]).append(strat_ret.index),   # anchor + traded dates
        name="portfolio_value")
    pnl_series = (pv_series - initial_capital).rename("cum_pnl")
    
    return pv_series, signal_series, pnl_series


def long_strat_unifortho(initial_capital, N_S, S, L, h1, h2, window_size, K=2, metric="CVaR", majority_lookback=7, weighting = "inverse_vol", half_life=5):
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
    portfolio_returns = compute_portfolio_returns(S, weighting=weighting, window_size=window_size, eps=epsilon)

    # Arrays to store results
    value_dates = [S.index[0]]
    portfolio_value = [initial_capital]
    cum_pnl = [0.0]
    trade_signals = []
    
    num_steps = math.floor(len(S) / window_size)

    for i in range(num_steps - 1):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        week_data = S.iloc[start_idx:end_idx, :]

        if debug:
            print(f'Analyzing Regime from {S.index[start_idx]} to {S.index[end_idx-1]} with {len(week_data)} data points.')

        if len(week_data) <= h1:
            if debug:
                print(f"Warning: Week {i} has {len(week_data)} points, which is too small for h1={h1}. STOP.")
            return np.array(portfolio_value), trade_signals, cum_pnl
            
        projected_emp, centroids, labels = ws.max_mccd_unifortho_sim(N_S, week_data, K, L, epsilon, h1, h2, metric)
        
        if majority_lookback > len(labels):
            #current_regime = np.bincount(labels).argmax()
            recent_labels = labels
        else:
            #current_regime = np.bincount(labels[-majority_lookback:]).argmax()
            recent_labels = labels[-majority_lookback:]
        weights = np.array([np.exp(-np.log(2) / half_life * (len(recent_labels) - 1 - k)) for k in range(len(recent_labels))])
        weighted_counts = np.bincount(recent_labels, weights=weights, minlength=K)
        current_regime = weighted_counts.argmax()

        if current_regime == 1:
            signal = 1
            if debug:
                print("Bullish ! Week", i+1, "go long")
        else:
            signal = -1
            if debug:
                print("Bearish!, Week", i+1, "go short")
      
        trade_signals.append(signal)

         #TODO Add something to only be able to trade at the open of the next available week day !!! 
        #next_week_returns = portfolio_returns.iloc[end_idx : end_idx + window_size]
        next_dates = S.index[end_idx : end_idx + window_size]
        next_week_returns = portfolio_returns.reindex(next_dates).dropna()

        # === FIX: Compounding with percentage returns ===
        for dt, ret in next_week_returns.items():          # .items() gives (date, return)
            period_return = signal * ret
            new_value = portfolio_value[-1] * (1 + period_return)
            portfolio_value.append(new_value)
            cum_pnl.append(new_value - initial_capital)
            value_dates.append(dt)                          # capture the date that produced this value
        if debug:
            print("---" * 10)
            print(f'Portfolio value after week {i+1}: {portfolio_value[-1]}')
            print(f"AND :Cumulative P&L: {cum_pnl[-1]}")
            print("---" * 10)
    pv_series     = pd.Series(portfolio_value, index=pd.DatetimeIndex(value_dates), name="portfolio_value")
    pnl_series    = pd.Series(cum_pnl,         index=pv_series.index,               name="cum_pnl")
    return pv_series, trade_signals, pnl_series


def long_strat_unifortho_label_data(initial_capital, N_S, S_label, S_trade, L, h1, h2, window_size, K=2, metric="CVaR", majority_lookback=20, weighting = "inverse_vol", half_life=5):
    """
    Similar to long_strat_unifort
    ho but uses S_label for regime detection and S_trade for trading.
    This allows us to test the strategy on one dataset while using another for regime inference.
    """
    epsilon = 1e-6  

    # === FIX: Use percentage returns, equal-weighted ===
    portfolio_returns = compute_portfolio_returns(S_trade, weighting=weighting, window_size=window_size, eps=epsilon)

    # Arrays to store results
    value_dates = [S_trade.index[0]]          # initial capital anchored at the first date (pre-trading)
    portfolio_value = [initial_capital]
    cum_pnl = [0.0]
    trade_signals = []
    
    num_steps = math.floor(len(S_label) / window_size)

    for i in range(num_steps - 1):
        
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        week_data = S_label.iloc[start_idx:end_idx, :]

        if debug:
            print(f'Analyzing Regime from {S_label.index[start_idx]} to {S_label.index[end_idx-1]} with {len(week_data)} data points.')

        if len(week_data) <= h1:
            if debug:
                print(f"Warning: Week {i} has {len(week_data)} points, which is too small for h1={h1}. STOP.")
            return np.array(portfolio_value), trade_signals, cum_pnl
            
        projected_emp, centroids, labels = ws.max_mccd_unifortho_sim(N_S, week_data, K, L, epsilon, h1, h2, metric)
        
        if majority_lookback > len(labels):
            recent_labels = labels
            #current_regime = np.bincount(labels).argmax()
        else:
            recent_labels = labels[-majority_lookback:]
            #current_regime = np.bincount(labels[-majority_lookback:]).argmax()
            
        weights = np.array([np.exp(-np.log(2) / half_life * (len(recent_labels) - 1 - k)) for k in range(len(recent_labels))])
        weighted_counts = np.bincount(recent_labels, weights=weights, minlength=K)
        current_regime = weighted_counts.argmax()

        if current_regime == 1:
            signal = 1
            if debug:
                print("Bullish ! Week", i+1, "go long")
        else:
            signal = -1
            if debug:
                print("Bearish!, Week", i+1, "go short")
        
        trade_signals.append(signal)
        #next_week_returns = portfolio_returns.iloc[end_idx : end_idx + window_size]
        next_dates = S_trade.index[end_idx : end_idx + window_size]
        next_week_returns = portfolio_returns.reindex(next_dates).dropna()
        # === FIX: Compounding with percentage returns ===

        for dt, ret in next_week_returns.items():          # .items() gives (date, return)
            period_return = signal * ret
            new_value = portfolio_value[-1] * (1 + period_return)
            portfolio_value.append(new_value)
            cum_pnl.append(new_value - initial_capital)
            value_dates.append(dt)                          # capture the date that produced this value
        if debug:
            print("---" * 10)
            print(f'Portfolio value after week {i+1}: {portfolio_value[-1]}')
            print(f"AND :Cumulative P&L: {cum_pnl[-1]}")
            print("---" * 10)
    pv_series     = pd.Series(portfolio_value, index=pd.DatetimeIndex(value_dates), name="portfolio_value")
    pnl_series    = pd.Series(cum_pnl,         index=pv_series.index,               name="cum_pnl")
    return pv_series, trade_signals, pnl_series

def long_only(S, initial_capital, weighting = "inverse_vol", window_size=5):
    
    epsilon = 1e-6
    '''
    pct_returns = S.pct_change().dropna()
    if weighting == "equal":
        theta = np.ones((1, S.shape[1])) / S.shape[1]  # Equal weights for each asset
        portfolio_returns = pct_returns.dot(theta.T).sum(axis=1)  # Portfolio returns
    elif weighting == "inverse_vol":
        vol = pct_returns.rolling(window=window_size).std().iloc[window_size-1:]
        inv_vol = 1 / (vol + epsilon)
        theta = inv_vol.div(inv_vol.sum(axis=1), axis=0)  # Normalize to sum to 1
        theta = theta.shift(1)
        portfolio_returns = (pct_returns.iloc[window_size-1:] * theta).sum(axis=1)  # Portfolio returns with inverse volatility weighting
    '''

    portfolio_returns = compute_portfolio_returns(S, weighting=weighting, window_size=window_size, eps=epsilon)


    cumulative = (1 + portfolio_returns).cumprod()
    portfolio_value = initial_capital * cumulative
    cum_pnl = portfolio_value-initial_capital
    return portfolio_value, cum_pnl, portfolio_value.iloc[-1]

def short_only(S, initial_capital, weighting = "inverse_vol", window_size=5):
    #TODO output P&L after holding for the entire time the security S 
    epsilon = 1e-6
    '''
    pct_returns = S.pct_change().dropna()
    if weighting == "equal":
        theta = np.ones((1, S.shape[1])) / S.shape[1]  # Equal weights for each asset
        portfolio_returns = -pct_returns.dot(theta.T).sum(axis=1)  # Short position: negative returns
    elif weighting == "inverse_vol":
        vol = pct_returns.rolling(window=window_size).std().iloc[window_size-1:]
        inv_vol = 1 / (vol + 1e-6)
        theta = inv_vol.div(inv_vol.sum(axis=1), axis=0)  # Normalize to sum to 1
        theta = theta.shift(1)
        portfolio_returns = -(pct_returns.iloc[window_size-1:] * theta).sum(axis=1)  # Short position: negative returns with inverse volatility weighting
    '''
    portfolio_returns = -compute_portfolio_returns(S, weighting=weighting, window_size=window_size, eps=epsilon)
    
    cumulative = (1 + portfolio_returns).cumprod()
    portfolio_value = initial_capital * cumulative
    return portfolio_value, cumulative, portfolio_value.iloc[-1]


def long_strat_implied(initial_capital, N_S, S, L, h1, h2, window_size, K=2, metric="CVaR", signal_type="conviction", majority_lookback= 20, half_life=5.0, entry_threshold=0.11, hold_threshold=0.10, lookback=5, use_gradient=False, gradient_weight=0.3, weighting = "inverse_vol", tau=None, tau_gradient=None, live_plot=False):
   
   
    epsilon = 1e-6

    # === FIX 1: Use percentage returns, equal-weighted across assets ===
    '''
    if weighting == "equal":
        pct_returns = S.pct_change().dropna()
        #portfolio_returns = pct_returns.mean(axis=1)  # equal-weight average
        theta = np.ones((1, S.shape[1])) / S.shape[1] # Equal weights for each asset in the portfolio
        portfolio_returns = pct_returns.dot(theta.T).sum(axis=1) # Portfolio returns as a weighted sum
    elif weighting == "inverse_vol":
        pct_returns = S.pct_change().dropna()
        vol = pct_returns.rolling(window=window_size).std().iloc[window_size-1:]
        inv_vol = 1 / (vol + epsilon)
        theta = inv_vol.div(inv_vol.sum(axis=1), axis=0)  # Normalize to sum to 1
        theta = theta.shift(1)
        portfolio_returns = (pct_returns.iloc[window_size-1:] * theta).sum(axis=1)  # Portfolio returns with inverse volatility weighting
    '''

    portfolio_returns = compute_portfolio_returns(S, weighting=weighting, window_size=window_size, eps=epsilon)

    # Arrays to store results
    portfolio_value = [initial_capital]
    value_dates = [S.index[0]]
    cum_pnl = [0.0]
    trade_signals = []
    num_steps = math.floor(len(S) / window_size)
    switch_proba_history = []


    for i in range(num_steps - 1):

        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        week_data = S.iloc[start_idx:end_idx, :]
        if debug:
            print(f'Analyzing Regime from {S.index[start_idx]} to {S.index[end_idx-1]} with {len(week_data)} data points.')
        if len(week_data) <= h1:
            if debug:
                print(f"Warning: Week {i} has {len(week_data)} points, which is too small for h1={h1}. STOP.")
            return np.array(portfolio_value), trade_signals, cum_pnl, switch_proba_history

        projected_emp, centroids, labels = ws.max_mccd_unifortho_sim(N_S, week_data, K, L, epsilon, h1, h2, metric)

        proba_matrix, switch_proba, transition_matrix, posterior = ws.compute_implied_proba(projected_emp, centroids, labels, lookback=lookback, use_gradient=use_gradient, gradient_weight=gradient_weight, tau=tau, tau_gradient=tau_gradient)
        
        if debug:
            print('---' * 10)
            print("Week", i + 1)
            print(f"Switch Probability: {switch_proba:.4f}")
            print(f"Transition Matrix:\n{transition_matrix}")
            print(f"Posterior Probabilities:\n{posterior}")
            print('---' * 10)

        #current_regime = np.bincount(labels[-h2:]).argmax()
        if majority_lookback > len(labels):
            #current_regime = np.bincount(labels).argmax()
            recent_labels = labels
        else:
            #current_regime = np.bincount(labels[-majority_lookback:]).argmax()
            recent_labels = labels[-majority_lookback:]
        weights = np.array([np.exp(-np.log(2) / half_life * (len(recent_labels) - 1 - k)) for k in range(len(recent_labels))])
        weighted_counts = np.bincount(recent_labels, weights=weights, minlength=K)

        current_regime = weighted_counts.argmax()

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
        if debug:
            print(f"Final signal: {signal}")

        # === FIX 2: Compounding portfolio value with percentage returns ===
        #next_week_returns = portfolio_returns.iloc[end_idx: end_idx + window_size]
        next_dates = S.index[end_idx : end_idx + window_size]
        next_week_returns = portfolio_returns.reindex(next_dates).dropna()
    
        for dt, ret in next_week_returns.items():          # .items() gives (date, return)
            period_return = signal * ret
            new_value = portfolio_value[-1] * (1 + period_return)
            portfolio_value.append(new_value)
            cum_pnl.append(new_value - initial_capital)
            value_dates.append(dt)                          # capture the date that produced this value
            switch_proba_history.append(switch_proba)
                
        if debug:
            print("---" * 10)
            print(f'Portfolio value after week {i+1}: {portfolio_value[-1]}')
            print(f"AND :Cumulative P&L: {cum_pnl[-1]}")
            print("---" * 10)
    pv_series     = pd.Series(portfolio_value, index=pd.DatetimeIndex(value_dates), name="portfolio_value")
    pnl_series    = pd.Series(cum_pnl,         index=pv_series.index,               name="cum_pnl")
    switch_series = switch_proba_history
    return pv_series, trade_signals, pnl_series, switch_series


def long_strat_implied_label_data(initial_capital, N_S, S_label, S_trade, L, h1, h2, window_size, start_date=None, end_date=None, K=2, metric="CVaR", signal_type="conviction", majority_lookback= 20, half_life=5, entry_threshold=0.15, hold_threshold=0.10, lookback=5, use_gradient=False, gradient_weight=0.3, weighting="inverse_vol", tau=None, tau_gradient=None, live_plot=False):
    """
    Similar to long_strat_implied but uses S_label for regime detection and S_trade for trading.
    This allows us to test the strategy on one dataset while using another for regime inference.
    """
    epsilon = 1e-6

    if start_date is not None and end_date is not None:
        S_label = S_label.loc[start_date:end_date]
        S_trade = S_trade.loc[start_date:end_date]


    '''
    if weighting == "equal":
        pct_returns = S_trade.pct_change().dropna()
        theta = np.ones((1, S_trade.shape[1])) / S_trade.shape[1]
        portfolio_returns = pct_returns.dot(theta.T).sum(axis=1)
    elif weighting == "inverse_vol":
        pct_returns = S_trade.pct_change().dropna()
        vol = pct_returns.rolling(window=window_size).std().iloc[window_size - 1:]
        inv_vol = 1 / (vol + epsilon)
        theta = inv_vol.div(inv_vol.sum(axis=1), axis=0)
        theta = theta.shift(1)
        portfolio_returns = (pct_returns.iloc[window_size - 1:] * theta).sum(axis=1)
    '''

    portfolio_returns = compute_portfolio_returns(S_trade, weighting=weighting, window_size=window_size, eps=epsilon)

    # Arrays to store results
    portfolio_value = [initial_capital]
    value_dates = [S.index[0]]
    cum_pnl = [0.0]
    trade_signals = []
    num_steps = math.floor(len(S_label) / window_size)
    switch_proba_history = []

    for i in range(num_steps - 1):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        week_data = S_label.iloc[start_idx:end_idx, :]
        
        if debug:
            print(f'Analyzing Regime from {S_label.index[start_idx]} to {S_label.index[end_idx - 1]} with {len(week_data)} data points.')

        if len(week_data) <= h1:
            if debug:
                print(f"Warning: Week {i} has {len(week_data)} points, which is too small for h1={h1}. STOP.")
            return np.array(portfolio_value), trade_signals, cum_pnl, switch_proba_history

        # Regime detection uses S_label
        projected_emp, centroids, labels = ws.max_mccd_unifortho_sim(N_S, week_data, K, L, epsilon, h1, h2, metric)

        proba_matrix, switch_proba, transition_matrix, posterior = ws.compute_implied_proba(
            projected_emp, centroids, labels,
            lookback=lookback, use_gradient=use_gradient, gradient_weight=gradient_weight, tau=tau, tau_gradient=tau_gradient
        )

        if debug:
            print('---' * 10)
            print("Week", i + 1)
            print(f"Switch Probability: {switch_proba:.4f}")
            print(f"Transition Matrix:\n{transition_matrix}")
            print(f"Posterior Probabilities:\n{posterior}")
            print('---' * 10)

        #for m in range(start_idx, end_idx):
        #    switch_proba_history.append(switch_proba)

        #current_regime = np.bincount(labels[-h2:]).argmax()
        if majority_lookback > len(labels):
            #current_regime = np.bincount(labels).argmax()
            recent_labels = labels
        else:
            #current_regime = np.bincount(labels[-majority_lookback:]).argmax()
            recent_labels = labels[-majority_lookback:]
        weights = np.array([np.exp(-np.log(2) / half_life * (len(recent_labels) - 1 - k)) for k in range(len(recent_labels))])
        weighted_counts = np.bincount(recent_labels, weights=weights, minlength=K)
        current_regime = weighted_counts.argmax()
        
        if signal_type == "continuous":
            signal = posterior[1] - posterior[0]
            dead_zone = 0.1
            if switch_proba > 0.5:
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
            else:
                if switch_proba >= entry_threshold and prev_signal <= 0:
                    signal = 1
                elif switch_proba < hold_threshold and prev_signal > 0:
                    signal = -1
                else:
                    signal = prev_signal

        if signal_type == "conviction":
            regime_direction = 1 if current_regime == 1 else -1
            conviction = 1.0 - 1.5 * switch_proba
            signal = regime_direction * conviction
            if switch_proba > 0.5:
                signal = np.sign(signal) * 1.0

        trade_signals.append(signal)
        if debug:
            print(f"Final signal: {signal}")

        # PnL computed on S_trade returns

        if debug:
            print(f"Applying the signal on week S_trade returns from {S_trade.index[end_idx]} to {S_trade.index[end_idx + window_size - 1]}")
        
        next_dates = S_trade.index[end_idx : end_idx + window_size]
        next_week_returns = portfolio_returns.reindex(next_dates).dropna()
    
        for dt, ret in next_week_returns.items():          # .items() gives (date, return)
            period_return = signal * ret
            new_value = portfolio_value[-1] * (1 + period_return)
            portfolio_value.append(new_value)
            cum_pnl.append(new_value - initial_capital)
            value_dates.append(dt)                          # capture the date that produced this value
            switch_proba_history.append(switch_proba) 

        if debug:
            print("---" * 10)
            print(f'Portfolio value after week {i + 1}: {portfolio_value[-1]}')
            print(f"AND :Cumulative P&L: {cum_pnl[-1]}")
            print("---" * 10)
    # ... at the end, REPLACE the return line with: ...
    pv_series     = pd.Series(portfolio_value, index=pd.DatetimeIndex(value_dates), name="portfolio_value")
    pnl_series    = pd.Series(cum_pnl,         index=pv_series.index,               name="cum_pnl")
    switch_series = switch_proba_history
    return pv_series, trade_signals, pnl_series, switch_series

 

def ensemble_strategy(initial_capital, N_S, S, L, h1, h2, window_size, 
                      K=2, metric="CVaR", majority_lookback=7, 
                      weighting="inverse_vol",
                      ensemble_weights=None,  # None = adaptive
                      lookback=5, use_gradient=False, gradient_weight=0.5,
                      entry_threshold=0.28, hold_threshold=0.31,
                      adaptive_lookback=3,  # how many past windows to evaluate
                      softmax_temperature=10.0, # controls how aggressive weighting is  
                      tau=None, tau_gradient=None, half_life=5):  
    
    epsilon = 1e-6

    '''
    # === Portfolio returns ===
    if weighting == "equal":
        pct_returns = S.pct_change().dropna()
        theta = np.ones((1, S.shape[1])) / S.shape[1]
        portfolio_returns = pct_returns.dot(theta.T).sum(axis=1)
    elif weighting == "inverse_vol":
        pct_returns = S.pct_change().dropna()
        vol = pct_returns.rolling(window=window_size).std().iloc[window_size-1:]
        inv_vol = 1 / (vol + epsilon)
        theta = inv_vol.div(inv_vol.sum(axis=1), axis=0)
        theta = theta.shift(1)
        portfolio_returns = (pct_returns.iloc[window_size-1:] * theta).sum(axis=1)
    '''

    portfolio_returns = compute_portfolio_returns(S, weighting=weighting, window_size=window_size, eps=epsilon)

    # Storage
    portfolio_value = [initial_capital]
    value_dates = [S.index[0]]
    cum_pnl = [0.0]
    trade_signals = []
    signal_details = []
    adaptive_weights_history = []

    # Track per-algo hypothetical cumulative returns
    algo_cum_returns = {
        'unifortho': [],
        'hysteresis': [],
        'continuous': [],
        'conviction': []
    }

    num_steps = math.floor(len(S) / window_size)
    prev_hysteresis_signal = 0
    use_adaptive = ensemble_weights is None

    for i in range(num_steps - 1):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        week_data = S.iloc[start_idx:end_idx, :]


        if debug:
            print(f'Analyzing Regime from {S.index[start_idx]} to {S.index[end_idx-1]}')

        if len(week_data) <= h1:
            if debug:
                print(f"Warning: too small for h1={h1}. STOP.")
            return np.array(portfolio_value), trade_signals, cum_pnl, signal_details

        # === Core regime detection ===
        projected_emp, centroids, labels = ws.max_mccd_unifortho_sim(
            N_S, week_data, K, L, epsilon, h1, h2, metric
        )
        proba_matrix, switch_proba, transition_matrix, posterior = ws.compute_implied_proba(
            projected_emp, centroids, labels,
            lookback=lookback, use_gradient=use_gradient, gradient_weight=gradient_weight, tau=tau, tau_gradient=tau_gradient
        )



        if majority_lookback > len(labels):
            #current_regime = np.bincount(labels).argmax()
            recent_labels = labels
        else:
            #current_regime = np.bincount(labels[-majority_lookback:]).argmax()
            recent_labels = labels[-majority_lookback:]
        weights = np.array([np.exp(-np.log(2) / half_life * (len(recent_labels) - 1 - k)) for k in range(len(recent_labels))])
        weighted_counts = np.bincount(recent_labels, weights=weights, minlength=K)
        current_regime = weighted_counts.argmax()
        
        # === ALGO 1: Unifortho ===
        signal_unifortho = 1.0 if current_regime == 1 else -1.0

        # === ALGO 2: Hysteresis ===
        if current_regime == 1:
            if switch_proba >= entry_threshold and prev_hysteresis_signal >= 0:
                signal_hysteresis = -1.0
            elif switch_proba < hold_threshold and prev_hysteresis_signal < 0:
                signal_hysteresis = 1.0
            else:
                signal_hysteresis = prev_hysteresis_signal
        else:
            if switch_proba >= entry_threshold and prev_hysteresis_signal <= 0:
                signal_hysteresis = 1.0
            elif switch_proba < hold_threshold and prev_hysteresis_signal > 0:
                signal_hysteresis = -1.0
            else:
                signal_hysteresis = prev_hysteresis_signal
        prev_hysteresis_signal = signal_hysteresis

        # === ALGO 3: Continuous ===
        signal_continuous = posterior[1] - posterior[0]
        if switch_proba > 0.5:
            signal_continuous = np.sign(signal_continuous) * 1.0
        if abs(signal_continuous) < 0.1:
            signal_continuous = 0.0

        # === ALGO 4: Conviction ===
        regime_direction = 1.0 if current_regime == 1 else -1.0
        signal_conviction = regime_direction * (1.0 - 1.5 * switch_proba)
        if switch_proba > 0.5:
            signal_conviction = np.sign(signal_conviction) * 1.0

        signals = np.array([signal_unifortho, signal_hysteresis,
                            signal_continuous, signal_conviction])

        # =============================================================
        # ADAPTIVE WEIGHTING: based on realized performance
        # =============================================================
        if use_adaptive:
            if i == 0:
                # First window: equal weights, no history yet
                current_weights = np.array([0.3, 0.1, 0.3, 0.3])
            else:
                # Look back over the last adaptive_lookback windows
                lb = max(0, len(algo_cum_returns['unifortho']) - adaptive_lookback)
                
                # Compute cumulative return per algo over lookback
                perf = np.array([
                    np.prod([1 + r for r in algo_cum_returns['unifortho'][lb:]]) - 1,
                    np.prod([1 + r for r in algo_cum_returns['hysteresis'][lb:]]) - 1,
                    np.prod([1 + r for r in algo_cum_returns['continuous'][lb:]]) - 1,
                    np.prod([1 + r for r in algo_cum_returns['conviction'][lb:]]) - 1,
                ])

                # Softmax with temperature
                # Higher temperature → more equal weights
                # Lower temperature → winner takes more
                scaled = perf * softmax_temperature
                scaled -= np.max(scaled)  # numerical stability
                exp_perf = np.exp(scaled)
                current_weights = exp_perf / np.sum(exp_perf)

            ensemble_signal = np.dot(current_weights, signals)
        else:
            current_weights = np.array(ensemble_weights)
            ensemble_signal = np.dot(current_weights, signals)

        ensemble_signal = np.clip(ensemble_signal, -1.0, 1.0)
        trade_signals.append(ensemble_signal)
        adaptive_weights_history.append(current_weights.copy())

        # === Apply to next window and track per-algo returns ===
        #next_week_returns = portfolio_returns.iloc[end_idx:end_idx + window_size]
        next_dates = S.index[end_idx : end_idx + window_size]
        next_week_returns = portfolio_returns.reindex(next_dates).dropna()
    
        # Per-algo hypothetical return for this window
        window_return_unifortho = np.prod([1 + signal_unifortho * r for r in next_week_returns]) - 1
        window_return_hysteresis = np.prod([1 + signal_hysteresis * r for r in next_week_returns]) - 1
        window_return_continuous = np.prod([1 + signal_continuous * r for r in next_week_returns]) - 1
        window_return_conviction = np.prod([1 + signal_conviction * r for r in next_week_returns]) - 1

        algo_cum_returns['unifortho'].append(window_return_unifortho)
        algo_cum_returns['hysteresis'].append(window_return_hysteresis)
        algo_cum_returns['continuous'].append(window_return_continuous)
        algo_cum_returns['conviction'].append(window_return_conviction)

        # Compound portfolio
        for dt, ret in next_week_returns.items():          # .items() gives (date, return)
            period_return = ensemble_signal * ret
            new_value = portfolio_value[-1] * (1 + period_return)
            portfolio_value.append(new_value)
            cum_pnl.append(new_value - initial_capital)
            value_dates.append(dt)                          # capture the date that produced this value

        signal_details.append({
            'week': i + 1,
            'regime': current_regime,
            'switch_proba': switch_proba,
            'unifortho': signal_unifortho,
            'hysteresis': signal_hysteresis,
            'continuous': signal_continuous,
            'conviction': signal_conviction,
            'ensemble': ensemble_signal,
            'weights': current_weights.tolist(),
            'algo_returns': [window_return_unifortho, window_return_hysteresis,
                             window_return_continuous, window_return_conviction]
        })

        if debug:
            print(f"  Signals → Uni: {signal_unifortho:.2f} | Hyst: {signal_hysteresis:.2f} | "
                f"Cont: {signal_continuous:.2f} | Conv: {signal_conviction:.2f}")
            print(f"  Weights → {current_weights}")
            print(f"  Ensemble: {ensemble_signal:.2f} | Portfolio: {portfolio_value[-1]:.2f}")
            print(f"Cumulative P&L: {cum_pnl[-1]:.2f}")
            print("---" * 10)

    pv_series     = pd.Series(portfolio_value, index=pd.DatetimeIndex(value_dates), name="portfolio_value")
    pnl_series    = pd.Series(cum_pnl,         index=pv_series.index,               name="cum_pnl")
    return pv_series, trade_signals, pnl_series, signal_details


def overlay_strat(initial_capital, N_S, S, L, h1, h2, window_size, 
                      K=2, metric="CVaR", majority_lookback=7, 
                      weighting="inverse_vol",
                      ensemble_weights=None,  # None = adaptive
                      lookback=5, use_gradient=False, gradient_weight=0.5,
                      entry_threshold=0.28, hold_threshold=0.31,
                      adaptive_lookback=3,  # how many past windows to evaluate
                      softmax_temperature=10.0, # controls how aggressive weighting is  
                      tau=None, tau_gradient=None, half_life=5):  
    
    epsilon = 1e-6

    '''
    # === Portfolio returns ===
    if weighting == "equal":
        pct_returns = S.pct_change().dropna()
        theta = np.ones((1, S.shape[1])) / S.shape[1]
        portfolio_returns = pct_returns.dot(theta.T).sum(axis=1)
    elif weighting == "inverse_vol":
        pct_returns = S.pct_change().dropna()
        vol = pct_returns.rolling(window=window_size).std().iloc[window_size-1:]
        inv_vol = 1 / (vol + epsilon)
        theta = inv_vol.div(inv_vol.sum(axis=1), axis=0)
        theta = theta.shift(1)
        portfolio_returns = (pct_returns.iloc[window_size-1:] * theta).sum(axis=1)
    '''

    portfolio_returns = compute_portfolio_returns(S, weighting=weighting, window_size=window_size, eps=epsilon)

    # Storage
    portfolio_value = [initial_capital]
    value_dates = [S.index[0]]
    cum_pnl = [0.0]
    trade_signals = []
    signal_details = []
    adaptive_weights_history = []

    # Track per-algo hypothetical cumulative returns
    algo_cum_returns = {
        'unifortho': [],
        'hysteresis': [],
        'continuous': [],
        'conviction': []
    }

    num_steps = math.floor(len(S) / window_size)
    prev_hysteresis_signal = 0
    use_adaptive = ensemble_weights is None

    for i in range(num_steps - 1):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size
        week_data = S.iloc[start_idx:end_idx, :]


        if debug:
            print(f'Analyzing Regime from {S.index[start_idx]} to {S.index[end_idx-1]}')

        if len(week_data) <= h1:
            if debug:
                print(f"Warning: too small for h1={h1}. STOP.")
            return np.array(portfolio_value), trade_signals, cum_pnl, signal_details

        # === Core regime detection ===
        projected_emp, centroids, labels = ws.max_mccd_unifortho_sim(
            N_S, week_data, K, L, epsilon, h1, h2, metric
        )
        proba_matrix, switch_proba, transition_matrix, posterior = ws.compute_implied_proba(
            projected_emp, centroids, labels,
            lookback=lookback, use_gradient=use_gradient, gradient_weight=gradient_weight, tau=tau, tau_gradient=tau_gradient
        )



        if majority_lookback > len(labels):
            #current_regime = np.bincount(labels).argmax()
            recent_labels = labels
        else:
            #current_regime = np.bincount(labels[-majority_lookback:]).argmax()
            recent_labels = labels[-majority_lookback:]
        weights = np.array([np.exp(-np.log(2) / half_life * (len(recent_labels) - 1 - k)) for k in range(len(recent_labels))])
        weighted_counts = np.bincount(recent_labels, weights=weights, minlength=K)
        current_regime = weighted_counts.argmax()
        
        # === ALGO 1: Unifortho ===
        signal_unifortho = 1.0 if current_regime == 1 else -1.0

        # === ALGO 2: Hysteresis ===
        if current_regime == 1:
            if switch_proba >= entry_threshold and prev_hysteresis_signal >= 0:
                signal_hysteresis = -1.0
            elif switch_proba < hold_threshold and prev_hysteresis_signal < 0:
                signal_hysteresis = 1.0
            else:
                signal_hysteresis = prev_hysteresis_signal
        else:
            if switch_proba >= entry_threshold and prev_hysteresis_signal <= 0:
                signal_hysteresis = 1.0
            elif switch_proba < hold_threshold and prev_hysteresis_signal > 0:
                signal_hysteresis = -1.0
            else:
                signal_hysteresis = prev_hysteresis_signal
        prev_hysteresis_signal = signal_hysteresis

        # === ALGO 3: Continuous ===
        signal_continuous = posterior[1] - posterior[0]
        if switch_proba > 0.5:
            signal_continuous = np.sign(signal_continuous) * 1.0
        if abs(signal_continuous) < 0.1:
            signal_continuous = 0.0

        # === ALGO 4: Conviction ===
        regime_direction = 1.0 if current_regime == 1 else -1.0
        signal_conviction = regime_direction * (1.0 - 1.5 * switch_proba)
        if switch_proba > 0.5:
            signal_conviction = np.sign(signal_conviction) * 1.0

        signals = np.array([signal_unifortho, signal_hysteresis,
                            signal_continuous, signal_conviction])
        signals = np.where(signals>=0, signals, 0)
        # =============================================================
        # ADAPTIVE WEIGHTING: based on realized performance
        # =============================================================
        if use_adaptive:
            if i == 0:
                # First window: equal weights, no history yet
                current_weights = np.array([0.3, 0.1, 0.3, 0.3])
            else:
                # Look back over the last adaptive_lookback windows
                lb = max(0, len(algo_cum_returns['unifortho']) - adaptive_lookback)
                
                # Compute cumulative return per algo over lookback
                perf = np.array([
                    np.prod([1 + r for r in algo_cum_returns['unifortho'][lb:]]) - 1,
                    np.prod([1 + r for r in algo_cum_returns['hysteresis'][lb:]]) - 1,
                    np.prod([1 + r for r in algo_cum_returns['continuous'][lb:]]) - 1,
                    np.prod([1 + r for r in algo_cum_returns['conviction'][lb:]]) - 1,
                ])

                # Softmax with temperature
                # Higher temperature → more equal weights
                # Lower temperature → winner takes more
                scaled = perf * softmax_temperature
                scaled -= np.max(scaled)  # numerical stability
                exp_perf = np.exp(scaled)
                current_weights = exp_perf / np.sum(exp_perf)

            ensemble_signal = np.dot(current_weights, signals)
        else:
            current_weights = np.array(ensemble_weights)
            ensemble_signal = np.dot(current_weights, signals)

        ensemble_signal = np.clip(ensemble_signal, -1.0, 1.0)
        trade_signals.append(ensemble_signal)
        adaptive_weights_history.append(current_weights.copy())

        # === Apply to next window and track per-algo returns ===
        #next_week_returns = portfolio_returns.iloc[end_idx:end_idx + window_size]
        next_dates = S.index[end_idx : end_idx + window_size]
        next_week_returns = portfolio_returns.reindex(next_dates).dropna()
    
        # Per-algo hypothetical return for this window
        window_return_unifortho = np.prod([1 + signal_unifortho * r for r in next_week_returns]) - 1
        window_return_hysteresis = np.prod([1 + signal_hysteresis * r for r in next_week_returns]) - 1
        window_return_continuous = np.prod([1 + signal_continuous * r for r in next_week_returns]) - 1
        window_return_conviction = np.prod([1 + signal_conviction * r for r in next_week_returns]) - 1

        algo_cum_returns['unifortho'].append(window_return_unifortho)
        algo_cum_returns['hysteresis'].append(window_return_hysteresis)
        algo_cum_returns['continuous'].append(window_return_continuous)
        algo_cum_returns['conviction'].append(window_return_conviction)

        # Compound portfolio
        for dt, ret in next_week_returns.items():          # .items() gives (date, return)
            period_return = ensemble_signal * ret
            new_value = portfolio_value[-1] * (1 + period_return)
            portfolio_value.append(new_value)
            cum_pnl.append(new_value - initial_capital)
            value_dates.append(dt)                          # capture the date that produced this value

        signal_details.append({
            'week': i + 1,
            'regime': current_regime,
            'switch_proba': switch_proba,
            'unifortho': signal_unifortho,
            'hysteresis': signal_hysteresis,
            'continuous': signal_continuous,
            'conviction': signal_conviction,
            'ensemble': ensemble_signal,
            'weights': current_weights.tolist(),
            'algo_returns': [window_return_unifortho, window_return_hysteresis,
                             window_return_continuous, window_return_conviction]
        })

        if debug:
            print(f"  Signals → Uni: {signal_unifortho:.2f} | Hyst: {signal_hysteresis:.2f} | "
                f"Cont: {signal_continuous:.2f} | Conv: {signal_conviction:.2f}")
            print(f"  Weights → {current_weights}")
            print(f"  Ensemble: {ensemble_signal:.2f} | Portfolio: {portfolio_value[-1]:.2f}")
            print(f"Cumulative P&L: {cum_pnl[-1]:.2f}")
            print("---" * 10)

    pv_series     = pd.Series(portfolio_value, index=pd.DatetimeIndex(value_dates), name="portfolio_value")
    pnl_series    = pd.Series(cum_pnl,         index=pv_series.index,               name="cum_pnl")
    return pv_series, trade_signals, pnl_series, signal_details


def ensemble_strategy_label_data(initial_capital, N_S, S_label, S_trade, L, h1, h2, window_size,
                                 K=2, metric="CVaR", majority_lookback=7, weighting="inverse_vol",
                                 ensemble_weights=None, lookback=5, use_gradient=False, gradient_weight=0.3,
                                 entry_threshold=0.15, hold_threshold=0.10,
                                 adaptive_lookback=3, softmax_temperature=10.0, tau=None, tau_gradient=None, half_life=5):

    epsilon = 1e-6


    '''
    # === Portfolio returns computed on S_trade ===
    if weighting == "equal":
        pct_returns = S_trade.pct_change().dropna()
        theta = np.ones((1, S_trade.shape[1])) / S_trade.shape[1]
        portfolio_returns = pct_returns.dot(theta.T).sum(axis=1)
    elif weighting == "inverse_vol":
        pct_returns = S_trade.pct_change().dropna()
        vol = pct_returns.rolling(window=window_size).std().iloc[window_size - 1:]
        inv_vol = 1 / (vol + epsilon)
        theta = inv_vol.div(inv_vol.sum(axis=1), axis=0)
        theta = theta.shift(1)

        portfolio_returns = (pct_returns.iloc[window_size - 1:] * theta).sum(axis=1)
    '''

    portfolio_returns = compute_portfolio_returns(S_trade, weighting=weighting, window_size=window_size, eps=epsilon)

    # Storage
    value_dates = [S.index[0]]
    portfolio_value = [initial_capital]
    cum_pnl = [0.0]
    trade_signals = []
    signal_details = []
    adaptive_weights_history = []

    algo_cum_returns = {
        'unifortho': [],
        'hysteresis': [],
        'continuous': [],
        'conviction': []
    }

    num_steps = math.floor(len(S_label) / window_size)
    prev_hysteresis_signal = 0
    use_adaptive = ensemble_weights is None

    for i in range(num_steps - 1):

        start_idx = i * window_size
        end_idx = (i + 1) * window_size

        # === Regime detection on S_label ===
        week_data = S_label.iloc[start_idx:end_idx, :]

        if debug:
            print(f'Analyzing Regime from {S_label.index[start_idx]} to {S_label.index[end_idx - 1]}')

        if len(week_data) <= h1:
            if debug:
                print(f"Warning: too small for h1={h1}. STOP.")
            return np.array(portfolio_value), trade_signals, cum_pnl, signal_details

        projected_emp, centroids, labels = ws.max_mccd_unifortho_sim(
            N_S, week_data, K, L, epsilon, h1, h2, metric
        )
        proba_matrix, switch_proba, transition_matrix, posterior = ws.compute_implied_proba(
            projected_emp, centroids, labels,
            lookback=lookback, use_gradient=use_gradient, gradient_weight=gradient_weight, tau=tau, tau_gradient=tau_gradient
        )

        if majority_lookback > len(labels):
            #current_regime = np.bincount(labels).argmax()
            recent_labels = labels
        else:
            #current_regime = np.bincount(labels[-majority_lookback:]).argmax()
            recent_labels = labels[-majority_lookback:]
        weights = np.array([np.exp(-np.log(2) / half_life * (len(recent_labels) - 1 - k)) for k in range(len(recent_labels))])
        weighted_counts = np.bincount(recent_labels, weights=weights, minlength=K)
        current_regime = weighted_counts.argmax()
        

         # === ALGO 1: Unifortho ===
        signal_unifortho = 1.0 if current_regime == 1 else -1.0

        # === ALGO 2: Hysteresis ===
        if current_regime == 1:
            if switch_proba >= entry_threshold and prev_hysteresis_signal >= 0:
                signal_hysteresis = -1.0
            elif switch_proba < hold_threshold and prev_hysteresis_signal < 0:
                signal_hysteresis = 1.0
            else:
                signal_hysteresis = prev_hysteresis_signal
        else:
            if switch_proba >= entry_threshold and prev_hysteresis_signal <= 0:
                signal_hysteresis = 1.0
            elif switch_proba < hold_threshold and prev_hysteresis_signal > 0:
                signal_hysteresis = -1.0
            else:
                signal_hysteresis = prev_hysteresis_signal
        prev_hysteresis_signal = signal_hysteresis

        # === ALGO 3: Continuous ===
        signal_continuous = posterior[1] - posterior[0]
        if switch_proba > 0.5:
            signal_continuous = np.sign(signal_continuous) * 1.0
        if abs(signal_continuous) < 0.1:
            signal_continuous = 0.0

        # === ALGO 4: Conviction ===
        regime_direction = 1.0 if current_regime == 1 else -1.0
        signal_conviction = regime_direction * (1.0 - 1.5 * switch_proba)
        if switch_proba > 0.5:
            signal_conviction = np.sign(signal_conviction) * 1.0

        signals = np.array([signal_unifortho, signal_hysteresis,
                            signal_continuous, signal_conviction])

        # === Adaptive weighting ===
        if use_adaptive:
            if i == 0:
                current_weights = np.array([0.3, 0.1, 0.3, 0.3])
            else:
                lb = max(0, len(algo_cum_returns['unifortho']) - adaptive_lookback)
                perf = np.array([
                    np.prod([1 + r for r in algo_cum_returns['unifortho'][lb:]]) - 1,
                    np.prod([1 + r for r in algo_cum_returns['hysteresis'][lb:]]) - 1,
                    np.prod([1 + r for r in algo_cum_returns['continuous'][lb:]]) - 1,
                    np.prod([1 + r for r in algo_cum_returns['conviction'][lb:]]) - 1,
                ])
                scaled = perf * softmax_temperature
                scaled -= np.max(scaled)
                exp_perf = np.exp(scaled)
                current_weights = exp_perf / np.sum(exp_perf)

            ensemble_signal = np.dot(current_weights, signals)
        else:
            current_weights = np.array(ensemble_weights)
            ensemble_signal = np.dot(current_weights, signals)

        ensemble_signal = np.clip(ensemble_signal, -1.0, 1.0)
        trade_signals.append(ensemble_signal)
        adaptive_weights_history.append(current_weights.copy())

        # === PnL on S_trade ===
        #next_week_returns = portfolio_returns.iloc[end_idx: end_idx + window_size]
        next_dates = S_trade.index[end_idx : end_idx + window_size]
        next_week_returns = portfolio_returns.reindex(next_dates).dropna()
    

        window_return_unifortho  = np.prod([1 + signal_unifortho  * r for r in next_week_returns]) - 1
        window_return_hysteresis = np.prod([1 + signal_hysteresis * r for r in next_week_returns]) - 1
        window_return_continuous = np.prod([1 + signal_continuous * r for r in next_week_returns]) - 1
        window_return_conviction = np.prod([1 + signal_conviction * r for r in next_week_returns]) - 1

        algo_cum_returns['unifortho'].append(window_return_unifortho)
        algo_cum_returns['hysteresis'].append(window_return_hysteresis)
        algo_cum_returns['continuous'].append(window_return_continuous)
        algo_cum_returns['conviction'].append(window_return_conviction)

        for dt, ret in next_week_returns.items():          # .items() gives (date, return)
            period_return = ensemble_signal * ret
            new_value = portfolio_value[-1] * (1 + period_return)
            portfolio_value.append(new_value)
            cum_pnl.append(new_value - initial_capital)
            value_dates.append(dt)                          # capture the date that produced this value

        signal_details.append({
            'week': i + 1,
            'regime': current_regime,
            'switch_proba': switch_proba,
            'unifortho': signal_unifortho,
            'hysteresis': signal_hysteresis,
            'continuous': signal_continuous,
            'conviction': signal_conviction,
            'ensemble': ensemble_signal,
            'weights': current_weights.tolist(),
            'algo_returns': [window_return_unifortho, window_return_hysteresis,
                             window_return_continuous, window_return_conviction]
        })

        if debug:
            print(f"  Signals → Uni: {signal_unifortho:.2f} | Hyst: {signal_hysteresis:.2f} | "
                  f"Cont: {signal_continuous:.2f} | Conv: {signal_conviction:.2f}")
            print(f"  Weights → {current_weights}")
            print(f"  Ensemble: {ensemble_signal:.2f} | Portfolio: {portfolio_value[-1]:.2f}")
            print(f"  Cumulative P&L: {cum_pnl[-1]:.2f}")
            print("---" * 10)
    pv_series     = pd.Series(portfolio_value, index=pd.DatetimeIndex(value_dates), name="portfolio_value")
    pnl_series    = pd.Series(cum_pnl,         index=pv_series.index,               name="cum_pnl")

    return pv_series, trade_signals, pnl_series, signal_details              

def rolling_sharpe(pv_array, dates, days_lookback=20, rf=0, obs_per_day=1):
    # 1. Setup Series and Frequency
    pv = pd.Series(pv_array, index=dates)
    
    # Observations per day (4 per hour * 24 hours)
    window_size = days_lookback * obs_per_day
    
    # 2. Calculate Returns (15-minute intervals)
    returns = pv.pct_change()
    
    # 3. Calculate Rolling Mean and Std Dev
    # We annualize the mean return and the standard deviation
    ann_factor_mean = obs_per_day * 252
    ann_factor_std = np.sqrt(obs_per_day * 252)
    
    rolling_mean = returns.rolling(window=window_size).mean() * ann_factor_mean
    rolling_std = returns.rolling(window=window_size).std() * ann_factor_std
    
    # 4. Compute Sharpe Ratio
    # We use .clip to prevent division by zero or near-zero volatility
    sharpe = (rolling_mean - rf) / rolling_std.clip(lower=1e-8)
    
    return sharpe

def sharpe_ratio(pv_array, obs_per_year=252, rf=0.0):
    """
    Full-sample annualised Sharpe ratio from a portfolio value series.
    
    Parameters
    ----------
    pv_array    : array-like of portfolio values
    obs_per_year: number of return observations per year
                  (252 for daily, 252*24*4 for 15-min, etc.)
    rf          : annualised risk-free rate (e.g. 0.04 for 4%)
    """
    pv = np.asarray(pv_array, dtype=float)
    returns = np.diff(pv) / pv[:-1]

    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)

    # Annualise both, then take the ratio
    annual_excess = mu * obs_per_year - rf
    annual_vol = sigma * np.sqrt(obs_per_year)

    return annual_excess / annual_vol


def expanding_sharpe(pv_array, dates, rf=0, obs_per_day=1, min_periods=20):
    """
    Expanding-window annualised Sharpe ratio.
    At each time t, uses all returns from inception up to t.

    Parameters
    ----------
    pv_array    : portfolio value series
    dates       : datetime index
    rf          : annualised risk-free rate
    obs_per_day : observations per trading day (1 for daily, 96 for 15-min, etc.)
    min_periods : minimum number of observations before outputting a value
    """
    pv = pd.Series(pv_array, index=dates)
    returns = pv.pct_change()

    ann_factor_mean = obs_per_day * 252
    ann_factor_std = np.sqrt(obs_per_day * 252)

    expanding_mean = returns.expanding(min_periods=min_periods).mean() * ann_factor_mean
    expanding_std = returns.expanding(min_periods=min_periods).std() * ann_factor_std

    sharpe = (expanding_mean - rf) / expanding_std.clip(lower=1e-8)

    return sharpe



def _active_strategy_returns(portfolio_value, trade_signals, window_size,
                             per_period_signals=False):
    """
    Reconstruct per-period strategy returns and the position that produced each,
    then keep only periods where a position was actually held.

    Conventions handled:
      - Walk-forward strategies (long_strat_unifortho, long_strat_implied,
        ensemble_strategy, and their *_label_data variants):
          portfolio_value = [C0, v1, v2, ...]   (leading initial capital)
          trade_signals   = one signal per window -> expanded by window_size
      - Look-ahead strategy (per_period_signals=True):
          portfolio_value = [C0, v1, v2, ...]   (prepend C0 in the strategy, see note)
          trade_signals   = one signal per return period, no expansion
    """
    pv = pd.Series(np.asarray(portfolio_value, dtype=float))
    # pv[0] = C0, so after pct_change the k-th (0-based) return is the return
    # earned during traded period k -> aligns 1:1 with expanded signals. No shift.
    strat_ret = pv.pct_change().dropna().reset_index(drop=True)
    print(strat_ret.shape)
    print(len(trade_signals))

    sig = np.asarray(trade_signals, dtype=float)
    if not per_period_signals:
        sig = np.repeat(sig, window_size)
    trades = pd.Series(sig).fillna(0.0)

    # Front-aligned truncation: only the FINAL window can be shorter than
    # window_size (the loop's iloc[end:end+w] slice at the series tail), so
    # trades may overrun strat_ret; cut both to the common length.
    n = min(len(strat_ret), len(trades))
    strat_ret, trades = strat_ret.iloc[:n], trades.iloc[:n]

    return strat_ret[trades.values != 0]


def compute_hit_ratio(portfolio_value, trade_signals, window_size,
                      per_period_signals=False):
    """Fraction of actively-traded periods with a positive strategy return."""

    active = _active_strategy_returns(portfolio_value, trade_signals,
                                      window_size, per_period_signals)
    if len(active) == 0:
        return 0.0
    return float((active > 0).mean())


def compute_win_loss_ratio(portfolio_value, trade_signals, window_size,
                           per_period_signals=False, normalized=False):
    """
    Average win vs average loss over actively-traded periods.
    normalized=True  -> avg_win / (avg_win + avg_loss), in [0, 1]  (your thesis convention)
    normalized=False -> avg_win / avg_loss, the raw Definition (18) ratio
    """
    active = _active_strategy_returns(portfolio_value, trade_signals,
                                      window_size, per_period_signals)
    if len(active) == 0:
        return 0.5 if normalized else np.nan

    wins, losses = active[active > 0], active[active < 0]
    if len(wins) == 0:
        return 0.0 if normalized else 0.0
    if len(losses) == 0:
        return 1.0 if normalized else np.inf

    avg_win, avg_loss = wins.mean(), -losses.mean()
    return float(avg_win / (avg_win + avg_loss)) if normalized else float(avg_win / avg_loss)

