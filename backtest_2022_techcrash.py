"""
Backtest: 2022 "Tech Wreck" - Bearish Market Analysis
Uses EXISTING trained models to test performance during the 2022 bear market.
This tests how the model would have performed during a major downturn.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# BACKTEST CONFIGURATION - 2022 TECH WRECK
# ==========================================
STOCKS = ['QQQ', 'SPY', 'AAPL', 'NVDA', 'AMD', 'INTC', 'GOOGL', 'TSLA']

# Test Period: 2022 Tech Wreck (Bearish Market)
TEST_START = '2022-01-01'
TEST_END = '2022-12-30'
TEST_NAME = "2022 Tech Wreck (Bearish Market)"

# Training data needed for technical indicators calculation continuity
# We need data from before the test period to calculate indicators properly
LOOKBACK_START = '2021-01-01'  # 1 year before test for indicator calculation

PREDICTION_DAYS = 90
STARTING_CAPITAL = 10000  # $10,000 per stock

# Strategy Parameters (same as main analysis)
CONFIDENCE_THRESHOLD = 0.005  # 0.5% threshold for trading
ATR_STOP_MULTIPLIER = 2.0  # Dynamic stop-loss: Entry - (2 * ATR)
RVOL_THRESHOLD = 0.75  # Volume must be > 0.75
USE_MACRO_FILTER = True  # Use 200-day SMA as macro regime filter
USE_TREND_OVERRIDE = True  # Force entry when trend is strong

# Strategy Mode: 'LONG_ONLY' or 'HYBRID' (allows sniper shorts)
STRATEGY_MODE = 'HYBRID'  # Sniper shorting enabled

# Sniper Short Parameters
SNIPER_ADX_THRESHOLD = 25
SNIPER_RSI_THRESHOLD = 50

# Technical Indicator Parameters
RSI_PERIOD = 21
MFI_PERIOD = 21
MACD_FAST = 24
MACD_SLOW = 52
MACD_SIGNAL = 18

MODEL_DIR = 'saved_models'

# ==========================================
# HELPER FUNCTIONS (copied from main analysis)
# ==========================================

def calculate_rsi(prices, period=RSI_PERIOD):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    """Calculate MACD"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def calculate_obv(close, volume):
    """Calculate On-Balance Volume"""
    obv = np.zeros(len(close))
    obv[0] = volume.iloc[0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv[i] = obv[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv[i] = obv[i-1] - volume.iloc[i]
        else:
            obv[i] = obv[i-1]
    return pd.Series(obv, index=close.index)

def calculate_mfi(high, low, close, volume, period=MFI_PERIOD):
    """Calculate Money Flow Index"""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    delta = typical_price.diff()
    positive_flow = raw_money_flow.where(delta > 0, 0)
    negative_flow = raw_money_flow.where(delta < 0, 0)
    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()
    mfr = positive_sum / negative_sum
    mfi = 100 - (100 / (1 + mfr))
    return mfi

def calculate_vwap(high, low, close, volume):
    """Calculate Volume Weighted Average Price"""
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap

def calculate_rvol(volume, period=20):
    """Calculate Relative Volume"""
    avg_volume = volume.rolling(window=period).mean()
    rvol = volume / avg_volume
    return rvol

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    
    atr = true_range.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()
    return adx

def flatten_columns(df):
    """Flatten MultiIndex columns from yfinance"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

def add_technical_indicators(df):
    """Add technical indicators"""
    df = flatten_columns(df)
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if isinstance(high, pd.DataFrame):
        high = high.iloc[:, 0]
    if isinstance(low, pd.DataFrame):
        low = low.iloc[:, 0]
    if isinstance(volume, pd.DataFrame):
        volume = volume.iloc[:, 0]
    
    df['MA_10'] = close.rolling(window=10).mean()
    df['MA_20'] = close.rolling(window=20).mean()
    df['MA_50'] = close.rolling(window=50).mean()
    df['MA_200'] = close.rolling(window=200).mean()
    df['RSI'] = calculate_rsi(close, period=RSI_PERIOD)
    
    macd, macd_signal, macd_hist = calculate_macd(close)
    df['MACD'] = macd.values if hasattr(macd, 'values') else macd
    df['MACD_Signal'] = macd_signal.values if hasattr(macd_signal, 'values') else macd_signal
    df['MACD_Hist'] = macd_hist.values if hasattr(macd_hist, 'values') else macd_hist
    
    bb_middle = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    df['BB_Middle'] = bb_middle.values if hasattr(bb_middle, 'values') else bb_middle
    df['BB_Upper'] = (bb_middle + (bb_std * 2)).values
    df['BB_Lower'] = (bb_middle - (bb_std * 2)).values
    
    df['Momentum'] = close.pct_change(periods=10).values
    df['Volatility'] = close.rolling(window=20).std().values
    df['OBV'] = calculate_obv(close, volume).values
    df['MFI'] = calculate_mfi(high, low, close, volume, period=MFI_PERIOD).values
    df['VWAP'] = calculate_vwap(high, low, close, volume).values
    df['RVOL'] = calculate_rvol(volume, period=20).values
    df['ATR'] = calculate_atr(high, low, close, period=14).values
    df['ADX'] = calculate_adx(high, low, close, period=14).values
    df['OBV_Norm'] = df['OBV'].pct_change().rolling(window=5).mean() * 100
    df['Above_MA200'] = (close > df['MA_200']).astype(float)
    
    df = df.dropna()
    return df

def load_saved_model(symbol):
    """Load a previously saved model and model info"""
    model_path = os.path.join(MODEL_DIR, f"{symbol}_lstm_model.keras")
    model_info_path = os.path.join(MODEL_DIR, f"{symbol}_model_info.pkl")
    
    if os.path.exists(model_path) and os.path.exists(model_info_path):
        model = load_model(model_path)
        with open(model_info_path, 'rb') as f:
            model_info = pickle.load(f)
        return model, model_info
    return None, None

def prepare_test_data_with_features(lookback_df, test_df, scaler, prediction_days, feature_columns):
    """Prepare test sequences with technical indicators"""
    combined_df = pd.concat([lookback_df, test_df], axis=0)
    combined_df = add_technical_indicators(combined_df.copy())
    
    start_idx = len(combined_df) - len(test_df) - prediction_days
    if start_idx < 0:
        start_idx = 0
    model_inputs_df = combined_df.iloc[start_idx:]
    
    data = model_inputs_df[feature_columns].values
    scaled_data = scaler.transform(data)
    
    x_test = []
    for i in range(prediction_days, len(scaled_data)):
        x_test.append(scaled_data[i-prediction_days:i])
    
    x_test = np.array(x_test)
    return x_test

def calculate_directional_accuracy(actual_prices, predicted_prices):
    """Calculate percentage of correct direction predictions"""
    actual_direction = np.diff(actual_prices.flatten()) > 0
    predicted_direction = np.diff(predicted_prices.flatten()) > 0
    correct = np.sum(actual_direction == predicted_direction)
    return (correct / len(actual_direction)) * 100

def calculate_mse(actual_prices, predicted_prices):
    return np.mean((actual_prices.flatten() - predicted_prices.flatten()) ** 2)

def calculate_rmse(actual_prices, predicted_prices):
    return np.sqrt(calculate_mse(actual_prices, predicted_prices))

def calculate_mape(actual_prices, predicted_prices):
    return np.mean(np.abs((actual_prices.flatten() - predicted_prices.flatten()) / actual_prices.flatten())) * 100

def simulate_investment(actual_prices, predicted_prices, starting_capital, 
                       ma_200=None, ma_50=None, ma_20=None, rvol=None, low_prices=None, high_prices=None,
                       atr=None, atr_avg=None, rsi=None, adx=None,
                       threshold=CONFIDENCE_THRESHOLD):
    """
    Simulate investment strategy with Regime-Based Logic & Volatility Circuit Breaker.
    
    PANIC SWITCH (Volatility Circuit Breaker):
    - If Current_ATR > 2.0 * ATR_Avg: HALT trading, force close all positions
    
    BULL REGIME (Price > 200 SMA):
    - Shorts: Force cover immediately
    - Long Entry: Buy if ML_Predicts_Up AND RVOL > 0.75
    - Long Exit: Sell if ML_Predicts_Down, UNLESS ADX > 30 (strong trend = hold)
    
    BEAR REGIME (Price < 200 SMA):
    - Longs: Force sell UNLESS Price > 20 SMA (Reversal Override)
    - Long Entry (Reversal): Buy ONLY if Price > 20 SMA AND ML_Predicts_Up AND RVOL > 0.75
    - Short Entry (Balanced Sniper): ML_Predicts_Down AND ADX > 15 AND 50 < RSI < 70
    - Short Exit: Cover if ML_Predicts_Up OR RSI < 30 (take profit when oversold)
    """
    min_len = min(len(actual_prices), len(predicted_prices))
    actual_prices = actual_prices[:min_len]
    predicted_prices = predicted_prices[:min_len]
    
    if ma_200 is not None:
        ma_200 = ma_200[:min_len]
    if ma_50 is not None:
        ma_50 = ma_50[:min_len]
    if ma_20 is not None:
        ma_20 = ma_20[:min_len]
    if rvol is not None:
        rvol = rvol[:min_len]
    if low_prices is not None:
        low_prices = low_prices[:min_len]
    if high_prices is not None:
        high_prices = high_prices[:min_len]
    if atr is not None:
        atr = atr[:min_len]
    if atr_avg is not None:
        atr_avg = atr_avg[:min_len]
    if rsi is not None:
        rsi = rsi[:min_len]
    if adx is not None:
        adx = adx[:min_len]
    
    if min_len < 2:
        return 0.0, [], [], 0.0
    
    capital = starting_capital
    position = 0
    position_type = None
    position_entry_price = 0
    position_stop_price = 0
    entry_day = 0
    daily_returns = []
    trade_history = []
    
    peak_capital = starting_capital
    max_drawdown = 0.0
    capital_history = [starting_capital]
    
    for i in range(min_len - 1):
        current_price = float(actual_prices[i])
        next_actual_price = float(actual_prices[i + 1])
        current_low = float(low_prices[i]) if low_prices is not None else current_price
        current_high = float(high_prices[i]) if high_prices is not None else current_price
        
        current_ma200 = float(ma_200[i]) if ma_200 is not None else 0
        current_ma50 = float(ma_50[i]) if ma_50 is not None else 0
        current_ma20 = float(ma_20[i]) if ma_20 is not None else 0
        current_rvol = float(rvol[i]) if rvol is not None else 1.0
        current_atr = float(atr[i]) if atr is not None else current_price * 0.02
        current_atr_avg = float(atr_avg[i]) if atr_avg is not None else current_atr
        current_rsi = float(rsi[i]) if rsi is not None else 50.0
        current_adx = float(adx[i]) if adx is not None else 20.0
        
        if i < min_len - 1:
            predicted_next_price = float(predicted_prices[i + 1])
        else:
            predicted_next_price = float(predicted_prices[i])
        
        pred_change_pct = (predicted_next_price - current_price) / current_price if current_price != 0 else 0
        ml_predicts_up = pred_change_pct > threshold
        ml_predicts_down = pred_change_pct < -threshold
        
        # STEP 0: PANIC SWITCH
        panic_mode = (current_atr_avg > 0) and (current_atr > 2.0 * current_atr_avg)
        
        if panic_mode:
            if position > 0:
                pnl = position * (current_price - position_entry_price)
                capital += pnl + (position * position_entry_price)
                trade_history.append({
                    'trade_num': len(trade_history) + 1, 'type': 'LONG', 'entry_day': entry_day, 'exit_day': i,
                    'entry_price': position_entry_price, 'exit_price': current_price, 'shares': abs(position),
                    'pnl': pnl, 'pnl_percent': (pnl / (abs(position) * position_entry_price)) * 100, 'exit_reason': 'PANIC_SWITCH'
                })
                position = 0; position_type = None; position_entry_price = 0; position_stop_price = 0
            elif position < 0:
                pnl = abs(position) * (position_entry_price - current_price)
                capital += pnl + (abs(position) * position_entry_price)
                trade_history.append({
                    'trade_num': len(trade_history) + 1, 'type': 'SHORT', 'entry_day': entry_day, 'exit_day': i,
                    'entry_price': position_entry_price, 'exit_price': current_price, 'shares': abs(position),
                    'pnl': pnl, 'pnl_percent': (pnl / (abs(position) * position_entry_price)) * 100, 'exit_reason': 'PANIC_SWITCH'
                })
                position = 0; position_type = None; position_entry_price = 0; position_stop_price = 0
            daily_returns.append(0.0); capital_history.append(capital); continue
        
        # STEP 1: ATR Stop-Loss
        stop_loss_triggered = False
        if position > 0 and position_type == 'LONG':
            if current_low <= position_stop_price:
                pnl = position * (position_stop_price - position_entry_price)
                capital += pnl + (position * position_entry_price)
                stop_loss_pct = ((position_stop_price - position_entry_price) / position_entry_price) * 100
                trade_history.append({
                    'trade_num': len(trade_history) + 1, 'type': 'LONG', 'entry_day': entry_day, 'exit_day': i,
                    'entry_price': position_entry_price, 'exit_price': position_stop_price, 'shares': abs(position),
                    'pnl': pnl, 'pnl_percent': (pnl / (abs(position) * position_entry_price)) * 100, 'exit_reason': f'ATR-STOP ({abs(stop_loss_pct):.1f}%)'
                })
                position = 0; position_type = None; position_entry_price = 0; position_stop_price = 0; stop_loss_triggered = True
        elif position < 0 and position_type == 'SHORT':
            if current_high >= position_stop_price:
                pnl = abs(position) * (position_entry_price - position_stop_price)
                capital += pnl + (abs(position) * position_entry_price)
                stop_loss_pct = ((position_stop_price - position_entry_price) / position_entry_price) * 100
                trade_history.append({
                    'trade_num': len(trade_history) + 1, 'type': 'SHORT', 'entry_day': entry_day, 'exit_day': i,
                    'entry_price': position_entry_price, 'exit_price': position_stop_price, 'shares': abs(position),
                    'pnl': pnl, 'pnl_percent': (pnl / (abs(position) * position_entry_price)) * 100, 'exit_reason': f'ATR-STOP ({abs(stop_loss_pct):.1f}%)'
                })
                position = 0; position_type = None; position_entry_price = 0; position_stop_price = 0; stop_loss_triggered = True
        
        if stop_loss_triggered:
            daily_returns.append(0.0); capital_history.append(capital); continue
        
        # STEP 2: Regime Detection
        bull_regime = current_price > current_ma200 if current_ma200 > 0 else True
        bear_regime = not bull_regime
        reversal_override = current_price > current_ma20 if current_ma20 > 0 else False
        volume_ok = current_rvol > RVOL_THRESHOLD
        strong_trend = current_adx > 30
        
        # STEP 3: Trading Logic
        action = None; exit_reason = None; entry_reason = None
        
        if bull_regime:
            if position < 0: action = 'COVER'; exit_reason = 'BULL_REGIME'
            elif position == 0:
                if ml_predicts_up and volume_ok: action = 'BUY'; entry_reason = 'ML_SIGNAL'
            elif position > 0:
                if ml_predicts_down and not strong_trend: action = 'SELL'; exit_reason = 'SIGNAL_SELL'
        elif bear_regime:
            if position > 0:
                if reversal_override:
                    if ml_predicts_down and not strong_trend: action = 'SELL'; exit_reason = 'SIGNAL_SELL'
                else: action = 'SELL'; exit_reason = 'BEAR_REGIME'
            elif position < 0:
                if ml_predicts_up or current_rsi < 30: action = 'COVER'; exit_reason = 'SIGNAL_COVER' if ml_predicts_up else 'RSI_OVERSOLD'
            elif position == 0:
                if reversal_override and ml_predicts_up and volume_ok: action = 'BUY'; entry_reason = 'REVERSAL_LONG'
                elif STRATEGY_MODE == 'HYBRID':
                    if ml_predicts_down and current_adx > 15 and 50 < current_rsi < 70 and volume_ok: action = 'SHORT'; entry_reason = 'BALANCED_SNIPER'
        
        # STEP 4: Execute Trade
        if action == 'SELL' and position > 0:
            pnl = position * (current_price - position_entry_price)
            capital += pnl + (position * position_entry_price)
            trade_history.append({'trade_num': len(trade_history) + 1, 'type': 'LONG', 'entry_day': entry_day, 'exit_day': i,
                'entry_price': position_entry_price, 'exit_price': current_price, 'shares': abs(position),
                'pnl': pnl, 'pnl_percent': (pnl / (abs(position) * position_entry_price)) * 100, 'exit_reason': exit_reason})
            position = 0; position_type = None; position_entry_price = 0; position_stop_price = 0
        elif action == 'BUY' and position == 0:
            position = capital / current_price; position_type = 'LONG'; position_entry_price = current_price
            position_stop_price = current_price - (ATR_STOP_MULTIPLIER * current_atr); entry_day = i; capital = 0
        elif action == 'SHORT' and position == 0:
            position = -(capital / current_price); position_type = 'SHORT'; position_entry_price = current_price
            position_stop_price = current_price + (ATR_STOP_MULTIPLIER * current_atr); entry_day = i; capital = 0
        elif action == 'COVER' and position < 0:
            pnl = abs(position) * (position_entry_price - current_price)
            capital += pnl + (abs(position) * position_entry_price)
            trade_history.append({'trade_num': len(trade_history) + 1, 'type': 'SHORT', 'entry_day': entry_day, 'exit_day': i,
                'entry_price': position_entry_price, 'exit_price': current_price, 'shares': abs(position),
                'pnl': pnl, 'pnl_percent': (pnl / (abs(position) * position_entry_price)) * 100, 'exit_reason': exit_reason})
            position = 0; position_type = None; position_entry_price = 0; position_stop_price = 0
        
        # Calculate daily P&L
        if position > 0: daily_pnl = position * (next_actual_price - current_price); current_value = position * next_actual_price
        elif position < 0: daily_pnl = abs(position) * (current_price - next_actual_price); current_value = abs(position) * position_entry_price + daily_pnl
        else: daily_pnl = 0.0; current_value = capital
        daily_returns.append(float(daily_pnl)); capital_history.append(current_value if position != 0 else capital)
        if capital_history[-1] > peak_capital: peak_capital = capital_history[-1]
        drawdown = (peak_capital - capital_history[-1]) / peak_capital if peak_capital > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    
    # Close final position
    final_price = float(actual_prices[-1])
    if position > 0:
        pnl = position * (final_price - position_entry_price); capital = position * final_price
        trade_history.append({'trade_num': len(trade_history) + 1, 'type': 'LONG', 'entry_day': entry_day, 'exit_day': min_len - 1,
            'entry_price': position_entry_price, 'exit_price': final_price, 'shares': abs(position),
            'pnl': pnl, 'pnl_percent': (pnl / (abs(position) * position_entry_price)) * 100, 'exit_reason': 'END_OF_PERIOD'})
    elif position < 0:
        pnl = abs(position) * (position_entry_price - final_price); capital = abs(position) * position_entry_price + pnl
        trade_history.append({'trade_num': len(trade_history) + 1, 'type': 'SHORT', 'entry_day': entry_day, 'exit_day': min_len - 1,
            'entry_price': position_entry_price, 'exit_price': final_price, 'shares': abs(position),
            'pnl': pnl, 'pnl_percent': (pnl / (abs(position) * position_entry_price)) * 100, 'exit_reason': 'END_OF_PERIOD'})
    
    total_pnl = capital - starting_capital
    return float(total_pnl), daily_returns, trade_history, float(max_drawdown * 100)

# ==========================================
# BACKTEST ANALYSIS
# ==========================================

def backtest_stock(symbol):
    """Run backtest for a single stock using existing model"""
    print(f"\n{'='*50}")
    print(f"Backtesting {symbol} on {TEST_NAME}...")
    print(f"{'='*50}")
    
    try:
        # Load existing model
        print(f"  Loading saved model...")
        model, model_info = load_saved_model(symbol)
        
        if model is None or model_info is None:
            print(f"  ERROR: No saved model found for {symbol}")
            print(f"  Please run monte_carlo_analysis.py first to train models.")
            return None
        
        scaler = model_info.get('scaler')
        price_scaler = model_info.get('price_scaler')
        feature_columns = model_info.get('feature_columns')
        use_features = model_info.get('use_features', False)
        
        print(f"  Model loaded (trained through {model_info.get('training_end_date', 'unknown')})")
        print(f"  Uses technical indicators: {use_features}")
        
        # Download lookback data for indicator calculation
        print(f"  Downloading lookback data ({LOOKBACK_START} to {TEST_START})...")
        lookback_data = yf.download(symbol, start=LOOKBACK_START, end=TEST_START, progress=False)
        
        # Download test data (2022)
        print(f"  Downloading test data ({TEST_START} to {TEST_END})...")
        test_data = yf.download(symbol, start=TEST_START, end=TEST_END, progress=False)
        
        if len(test_data) < 10:
            print(f"  ERROR: Not enough test data for {symbol}")
            return None
        
        actual_prices = test_data['Close'].values
        
        # Prepare test data
        if use_features and feature_columns is not None:
            x_test = prepare_test_data_with_features(lookback_data, test_data, scaler, PREDICTION_DAYS, feature_columns)
        else:
            print(f"  ERROR: Model requires features but none found")
            return None
        
        # Make predictions
        print(f"  Making predictions...")
        predicted_prices = model.predict(x_test, verbose=0)
        predicted_prices = price_scaler.inverse_transform(predicted_prices.reshape(-1, 1))
        
        # Align arrays
        min_len = min(len(actual_prices), len(predicted_prices))
        actual_prices = actual_prices[:min_len]
        predicted_prices = predicted_prices[:min_len].flatten()
        
        # Calculate metrics
        print(f"  Calculating metrics...")
        directional_accuracy = calculate_directional_accuracy(actual_prices, predicted_prices)
        mse = calculate_mse(actual_prices, predicted_prices)
        rmse = calculate_rmse(actual_prices, predicted_prices)
        mape = calculate_mape(actual_prices, predicted_prices)
        
        # Get indicators for simulation
        combined_data = pd.concat([lookback_data, test_data], axis=0)
        combined_data = flatten_columns(combined_data)
        
        close_series = combined_data['Close']
        if isinstance(close_series, pd.DataFrame):
            close_series = close_series.iloc[:, 0]
        
        high_series = combined_data['High']
        if isinstance(high_series, pd.DataFrame):
            high_series = high_series.iloc[:, 0]
        
        volume_series = combined_data['Volume']
        if isinstance(volume_series, pd.DataFrame):
            volume_series = volume_series.iloc[:, 0]
        
        low_series = combined_data['Low']
        if isinstance(low_series, pd.DataFrame):
            low_series = low_series.iloc[:, 0]
        
        # Calculate indicators
        ma_200_full = close_series.rolling(window=200).mean()
        ma_50_full = close_series.rolling(window=50).mean()
        ma_20_full = close_series.rolling(window=20).mean()  # NEW: 20-day SMA for reversal override
        rvol_full = calculate_rvol(volume_series, period=20)
        atr_full = calculate_atr(high_series, low_series, close_series, period=14)
        atr_avg_full = atr_full.rolling(window=30).mean()  # NEW: 30-day ATR average for panic switch
        rsi_full = calculate_rsi(close_series, period=RSI_PERIOD)
        adx_full = calculate_adx(high_series, low_series, close_series, period=14)
        
        # Extract test period values
        test_start_idx = len(combined_data) - len(test_data)
        ma_200_test = ma_200_full.iloc[test_start_idx:test_start_idx + min_len].values
        ma_50_test = ma_50_full.iloc[test_start_idx:test_start_idx + min_len].values
        ma_20_test = ma_20_full.iloc[test_start_idx:test_start_idx + min_len].values  # NEW
        rvol_test = rvol_full.iloc[test_start_idx:test_start_idx + min_len].values
        atr_test = atr_full.iloc[test_start_idx:test_start_idx + min_len].values
        atr_avg_test = atr_avg_full.iloc[test_start_idx:test_start_idx + min_len].values  # NEW
        rsi_test = rsi_full.iloc[test_start_idx:test_start_idx + min_len].values
        adx_test = adx_full.iloc[test_start_idx:test_start_idx + min_len].values
        low_test = low_series.iloc[test_start_idx:test_start_idx + min_len].values
        high_test = high_series.iloc[test_start_idx:test_start_idx + min_len].values
        
        # Simulate investment (Regime-Based with Panic Switch)
        strategy_desc = "LONG-ONLY" if STRATEGY_MODE == 'LONG_ONLY' else "HYBRID (Balanced Sniper)"
        print(f"  Simulating {strategy_desc} strategy...")
        total_pnl, daily_returns, trade_history, max_drawdown = simulate_investment(
            actual_prices, predicted_prices, STARTING_CAPITAL,
            ma_200=ma_200_test, ma_50=ma_50_test, ma_20=ma_20_test, rvol=rvol_test, 
            low_prices=low_test, high_prices=high_test, 
            atr=atr_test, atr_avg=atr_avg_test, rsi=rsi_test, adx=adx_test,
            threshold=CONFIDENCE_THRESHOLD
        )
        pnl_percent = (total_pnl / STARTING_CAPITAL) * 100
        
        # Buy & Hold comparison
        buy_hold_return = ((actual_prices[-1] - actual_prices[0]) / actual_prices[0]) * 100
        buy_hold_pnl = STARTING_CAPITAL * (buy_hold_return / 100)
        
        # Win rate
        winning_trades = [t for t in trade_history if t['pnl'] > 0]
        total_trades = len(trade_history)
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        
        # Count trade types
        long_trades = [t for t in trade_history if t.get('type') == 'LONG']
        short_trades = [t for t in trade_history if t.get('type') == 'SHORT']
        
        results = {
            'symbol': symbol,
            'start_price': float(actual_prices[0]),
            'end_price': float(actual_prices[-1]),
            'directional_accuracy': float(directional_accuracy),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'starting_capital': STARTING_CAPITAL,
            'total_pnl': float(total_pnl),
            'pnl_percent': float(pnl_percent),
            'final_capital': float(STARTING_CAPITAL + total_pnl),
            'status': 'GAIN' if float(total_pnl) > 0 else 'LOSS',
            'test_days': len(actual_prices),
            'trade_history': trade_history,
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'total_trades': total_trades,
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'buy_hold_return': float(buy_hold_return),
            'buy_hold_pnl': float(buy_hold_pnl)
        }
        
        print(f"  ✓ Backtest complete for {symbol}")
        return results
        
    except Exception as e:
        import traceback
        print(f"  ERROR backtesting {symbol}: {str(e)}")
        traceback.print_exc()
        return None

def generate_backtest_report(results):
    """Generate markdown report for backtest"""
    report = []
    
    strategy_title = "Long-Only Strategy" if STRATEGY_MODE == 'LONG_ONLY' else "Hybrid Strategy (Sniper Shorts)"
    report.append(f"# Backtest Report: {TEST_NAME}")
    report.append(f"\n## {strategy_title}")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Test Period:** {TEST_START} to {TEST_END}")
    report.append(f"\n**Starting Capital per Stock:** ${STARTING_CAPITAL:,.2f}")
    report.append(f"\n**Note:** Using models trained on recent data to test on historical bearish period")
    
    # Market Context
    report.append("\n\n## Market Context: 2022 Tech Wreck\n")
    report.append("The 2022 bear market was characterized by:")
    report.append("- **Fed Rate Hikes:** Aggressive interest rate increases to combat inflation")
    report.append("- **Tech Selloff:** High-growth tech stocks hit particularly hard")
    report.append("- **QQQ:** Dropped ~33% from peak to trough")
    report.append("- **NASDAQ:** Worst performance since 2008 financial crisis")
    report.append("- **Key Theme:** Flight from growth stocks to value/defensive sectors")
    
    # Summary Table
    report.append("\n\n## Summary Results\n")
    report.append("| Stock | Strategy Return | Buy & Hold | Outperform | Win Rate | Max DD | Trades (L/S) | Status |")
    report.append("|-------|-----------------|------------|------------|----------|--------|--------------|--------|")
    
    total_pnl = 0
    total_buy_hold_pnl = 0
    valid_results = [r for r in results if r is not None]
    
    if len(valid_results) == 0:
        report.append("\nNo valid results to report.\n")
        return "\n".join(report)
    
    for r in valid_results:
        status_emoji = "✅" if r['status'] == 'GAIN' else "❌"
        outperform_val = r['pnl_percent'] - r['buy_hold_return']
        outperform = f"+{outperform_val:.1f}% 📈" if outperform_val > 0 else f"{outperform_val:.1f}% 📉"
        report.append(
            f"| {r['symbol']} | {r['pnl_percent']:+.1f}% (${r['total_pnl']:+,.0f}) | "
            f"{r['buy_hold_return']:+.1f}% | {outperform} | {r['win_rate']:.0f}% | "
            f"{r['max_drawdown']:.1f}% | {r['total_trades']} ({r['long_trades']}L/{r['short_trades']}S) | {status_emoji} {r['status']} |"
        )
        total_pnl += r['total_pnl']
        total_buy_hold_pnl += r['buy_hold_pnl']
    
    # Portfolio Summary
    report.append("\n\n## Portfolio Summary\n")
    total_invested = STARTING_CAPITAL * len(valid_results)
    portfolio_return = (total_pnl / total_invested) * 100 if total_invested > 0 else 0
    buy_hold_portfolio_return = (total_buy_hold_pnl / total_invested) * 100 if total_invested > 0 else 0
    
    report.append(f"| Metric | Strategy | Buy & Hold |")
    report.append(f"|--------|----------|------------|")
    report.append(f"| Total Invested | ${total_invested:,.2f} | ${total_invested:,.2f} |")
    report.append(f"| Total P&L | ${total_pnl:+,.2f} | ${total_buy_hold_pnl:+,.2f} |")
    report.append(f"| Portfolio Return | {portfolio_return:+.2f}% | {buy_hold_portfolio_return:+.2f}% |")
    report.append(f"| Final Value | ${total_invested + total_pnl:,.2f} | ${total_invested + total_buy_hold_pnl:,.2f} |")
    
    outperformance = portfolio_return - buy_hold_portfolio_return
    report.append(f"\n**Strategy vs Buy & Hold:** {outperformance:+.2f}% {'(Outperformed ✅)' if outperformance > 0 else '(Underperformed ❌)'}")
    
    if outperformance > 0:
        report.append(f"\n🎯 **The strategy PROTECTED capital during the 2022 bear market!**")
        report.append(f"   While buy & hold lost ${abs(total_buy_hold_pnl):,.2f}, the strategy only lost ${abs(total_pnl):,.2f} (or gained if positive).")
    else:
        report.append(f"\n⚠️ Note: Strategy underperformed during this specific period.")
    
    # Detailed Results
    report.append("\n\n## Detailed Stock Analysis\n")
    
    for r in valid_results:
        report.append(f"\n### {r['symbol']}\n")
        
        # Price movement
        price_change_pct = ((r['end_price'] - r['start_price']) / r['start_price']) * 100
        report.append(f"**Stock Movement:** ${r['start_price']:.2f} → ${r['end_price']:.2f} ({price_change_pct:+.1f}%)\n")
        
        report.append(f"| Metric | Value |")
        report.append(f"|--------|-------|")
        report.append(f"| Directional Accuracy | {r['directional_accuracy']:.2f}% |")
        report.append(f"| RMSE | ${r['rmse']:.2f} |")
        report.append(f"| MAPE | {r['mape']:.2f}% |")
        report.append(f"| Starting Capital | ${r['starting_capital']:,.2f} |")
        report.append(f"| Final Capital | ${r['final_capital']:,.2f} |")
        report.append(f"| **Strategy Return** | **{r['pnl_percent']:+.2f}%** |")
        report.append(f"| Buy & Hold Return | {r['buy_hold_return']:+.2f}% |")
        report.append(f"| Max Drawdown | {r['max_drawdown']:.2f}% |")
        report.append(f"| Win Rate | {r['win_rate']:.1f}% |")
        report.append(f"| Total Trades | {r['total_trades']} ({r['long_trades']} Long, {r['short_trades']} Short) |")
        report.append(f"| Trading Days | {r['test_days']} |")
        
        # Trade History
        if r.get('trade_history') and len(r['trade_history']) > 0:
            report.append(f"\n#### Trade History\n")
            report.append("| # | Type | Entry Day | Exit Day | Entry $ | Exit $ | P&L | Return | Exit Reason |")
            report.append("|---|------|-----------|----------|---------|--------|-----|--------|-------------|")
            
            for t in r['trade_history']:
                status_icon = "+" if t['pnl'] >= 0 else ""
                type_emoji = "📈" if t['type'] == 'LONG' else "📉"
                report.append(
                    f"| {t['trade_num']} | {type_emoji} {t['type']} | Day {t['entry_day']} | Day {t['exit_day']} | "
                    f"${t['entry_price']:.2f} | ${t['exit_price']:.2f} | "
                    f"${status_icon}{t['pnl']:.2f} | {status_icon}{t['pnl_percent']:.2f}% | {t.get('exit_reason', 'N/A')} |"
                )
    
    # Key Insights
    report.append("\n\n## Key Insights\n")
    
    # Count stocks that beat buy & hold
    beat_buy_hold = sum(1 for r in valid_results if r['pnl_percent'] > r['buy_hold_return'])
    report.append(f"- **Stocks that beat Buy & Hold:** {beat_buy_hold} of {len(valid_results)}")
    
    # Best and worst performers
    if valid_results:
        best = max(valid_results, key=lambda x: x['pnl_percent'])
        worst = min(valid_results, key=lambda x: x['pnl_percent'])
        report.append(f"- **Best Performer:** {best['symbol']} ({best['pnl_percent']:+.1f}%)")
        report.append(f"- **Worst Performer:** {worst['symbol']} ({worst['pnl_percent']:+.1f}%)")
        
        # Count total shorts
        total_shorts = sum(r['short_trades'] for r in valid_results)
        if total_shorts > 0:
            report.append(f"- **Sniper Short Trades:** {total_shorts} total across all stocks")
    
    report.append("\n\n## Conclusion\n")
    if outperformance > 0:
        report.append(f"✅ **The strategy successfully protected capital during the 2022 bear market.**")
        report.append(f"\nBy using the macro filter (200-day SMA) and sniper shorts, the strategy avoided ")
        report.append(f"extended exposure during the downturn. This demonstrates the value of:")
        report.append(f"1. Staying out of the market when macro conditions are bearish")
        report.append(f"2. Using ATR-based stop-losses to limit drawdowns")
        report.append(f"3. Sniper shorting opportunities during confirmed downtrends")
    else:
        report.append(f"⚠️ The strategy underperformed during this specific period.")
        report.append(f"\nPotential improvements to consider:")
        report.append(f"- Adjust macro filter sensitivity")
        report.append(f"- Tighten stop-losses during volatile periods")
        report.append(f"- Consider more aggressive shorting during confirmed bear markets")
    
    return "\n".join(report)

# ==========================================
# RUN BACKTEST
# ==========================================

if __name__ == "__main__":
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    REPORTS_DIR = 'reports'
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
    REPORT_FILENAME = os.path.join(REPORTS_DIR, f"Backtest_2022_TechWreck_{timestamp}.md")
    
    strategy_title = "LONG-ONLY" if STRATEGY_MODE == 'LONG_ONLY' else "HYBRID (Sniper Shorts)"
    
    print("=" * 70)
    print(f"BACKTEST: {TEST_NAME}")
    print(f"Strategy: {strategy_title}")
    print("=" * 70)
    print(f"Stocks: {', '.join(STOCKS)}")
    print(f"Test Period: {TEST_START} to {TEST_END}")
    print(f"Starting Capital: ${STARTING_CAPITAL:,} per stock")
    print(f"Using existing trained models (not retraining)")
    print(f"Report will be saved as: {REPORT_FILENAME}")
    print("=" * 70)
    
    results = []
    for symbol in STOCKS:
        result = backtest_stock(symbol)
        results.append(result)
    
    # Generate report
    print("\n" + "=" * 70)
    print("GENERATING BACKTEST REPORT...")
    print("=" * 70)
    
    report = generate_backtest_report(results)
    
    with open(REPORT_FILENAME, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ Report saved to: {REPORT_FILENAME}")
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"BACKTEST SUMMARY: {TEST_NAME}")
    print("=" * 70)
    
    valid_results = [r for r in results if r is not None]
    
    if len(valid_results) == 0:
        print("No valid results to display.")
    else:
        print(f"{'Stock':6} | {'Strategy':>12} | {'Buy&Hold':>10} | {'Outperf':>10} | {'MaxDD':>6} | {'Trades':>8} | Status")
        print("-" * 80)
        for r in valid_results:
            status = "✅ GAIN" if r['status'] == 'GAIN' else "❌ LOSS"
            outperf = r['pnl_percent'] - r['buy_hold_return']
            outperf_icon = "📈" if outperf > 0 else "📉"
            print(f"{r['symbol']:6} | {r['pnl_percent']:+10.1f}% | {r['buy_hold_return']:+8.1f}% | {outperf:+8.1f}% {outperf_icon} | "
                  f"{r['max_drawdown']:5.1f}% | {r['long_trades']}L/{r['short_trades']}S   | {status}")
        
        total_pnl = sum(r['total_pnl'] for r in valid_results)
        total_buy_hold = sum(r['buy_hold_pnl'] for r in valid_results)
        total_invested = STARTING_CAPITAL * len(valid_results)
        
        print("-" * 80)
        strategy_return = (total_pnl / total_invested) * 100
        buy_hold_return = (total_buy_hold / total_invested) * 100
        outperformance = strategy_return - buy_hold_return
        print(f"TOTAL  | Strategy: {strategy_return:+.1f}% | Buy&Hold: {buy_hold_return:+.1f}%")
        print(f"       | P&L: ${total_pnl:+,.2f} vs ${total_buy_hold:+,.2f}")
        print(f"       | Outperformance: {outperformance:+.2f}% {'✅' if outperformance > 0 else '❌'}")
        
        if outperformance > 0:
            print(f"\n🎯 Strategy outperformed Buy & Hold by {outperformance:.1f}% during the 2022 bear market!")
    print("=" * 70)
