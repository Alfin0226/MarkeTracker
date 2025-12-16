"""
Monte Carlo Stock Analysis with LSTM Model (Long-Only Strategy)
Analyzes: QQQ, SPY, AAPL, NVDA, AMD, INTC, GOOGL, TSLA
Features: Volume-weighted indicators, Macro regime filter, Stop-loss protection
Generates: Directional Accuracy, MSE, Investment Simulation with Exit Reasons
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
STOCKS = ['QQQ', 'SPY', 'AAPL', 'NVDA', 'AMD', 'INTC', 'GOOGL', 'TSLA']

# Rolling Window Training (2 years training, predict next period)
ROLLING_WINDOW_YEARS = 2
TEST_DAYS = 90  # Past 2 months for testing
TEST_START = (datetime.now() - timedelta(days=TEST_DAYS)).strftime('%Y-%m-%d')
TEST_END = datetime.now().strftime('%Y-%m-%d')
# Training window: 2 years ending today (rolling window with most recent data)
TRAINING_START = (datetime.now() - timedelta(days=ROLLING_WINDOW_YEARS * 365)).strftime('%Y-%m-%d')
TRAINING_END = TEST_END  # Training includes data up to today

PREDICTION_DAYS = 90
STARTING_CAPITAL = 10000  # $10,000 per stock
MONTE_CARLO_SIMULATIONS = 10  # Reduced for speed
EPOCHS = 100
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 20

# Strategy Parameters
CONFIDENCE_THRESHOLD = 0.005  # 0.5% threshold for trading
ATR_STOP_MULTIPLIER = 2.0  # Dynamic stop-loss: Entry - (2 * ATR)
RVOL_THRESHOLD = 0.75  # Relaxed: volume must be > 0.75 (avoid dead volume only)
USE_MACRO_FILTER = True  # Use 200-day SMA as macro regime filter
USE_TREND_OVERRIDE = True  # Force entry when trend is strong (Price > 200 SMA AND > 50 SMA)

# Strategy Mode: 'LONG_ONLY' or 'HYBRID' (allows sniper shorts)
STRATEGY_MODE = 'HYBRID'  # Sniper shorting enabled

# Sniper Short Parameters (only used in HYBRID mode)
# Short only when: Price < 200 SMA AND ADX > 25 AND RSI < 50
SNIPER_ADX_THRESHOLD = 25  # Trend strength threshold for shorts
SNIPER_RSI_THRESHOLD = 50  # RSI must be below this to short

# Hyperparameters (Tuned for less noise)
RSI_PERIOD = 21  # Increased from 14
MFI_PERIOD = 21  # Money Flow Index period
MACD_FAST = 24   # Increased from 12
MACD_SLOW = 52   # Increased from 26
MACD_SIGNAL = 18 # Increased from 9

USE_TECHNICAL_INDICATORS = True
SAVE_MODELS = True
MODEL_DIR = 'saved_models'
LOAD_EXISTING_MODELS = True  # Try to load existing models
AUTO_UPDATE_MODELS = True  # Retrain if model is outdated (training_end != current TRAINING_END)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def load_saved_model(symbol, expected_training_end=None):
    """Load a previously saved model and model info for a symbol.
    
    Args:
        symbol: Stock ticker symbol
        expected_training_end: Expected training end date (YYYY-MM-DD string).
                              If provided, checks if model is up-to-date.
    
    Returns:
        (model, model_info, is_up_to_date) tuple.
        is_up_to_date is True if model's training_end matches expected_training_end.
    """
    from tensorflow.keras.models import load_model
    
    model_path = os.path.join(MODEL_DIR, f"{symbol}_lstm_model.keras")
    model_info_path = os.path.join(MODEL_DIR, f"{symbol}_model_info.pkl")
    
    if os.path.exists(model_path) and os.path.exists(model_info_path):
        model = load_model(model_path)
        with open(model_info_path, 'rb') as f:
            model_info = pickle.load(f)
        
        # Check if model is up-to-date (saved date must be >= expected date)
        is_up_to_date = True
        if expected_training_end:
            saved_training_end = model_info.get('training_end_date')
            if saved_training_end:
                # Compare dates chronologically - model is up-to-date if trained through same or later date
                is_up_to_date = (saved_training_end >= expected_training_end)
            else:
                # Old model without date info - consider outdated
                is_up_to_date = False
        
        return model, model_info, is_up_to_date
    return None, None, False

def calculate_rsi(prices, period=RSI_PERIOD):
    """Calculate Relative Strength Index with configurable period"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    """Calculate MACD with tuned parameters for less noise"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def calculate_obv(close, volume):
    """Calculate On-Balance Volume to detect price/volume divergence"""
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
    """Calculate Money Flow Index (volume-weighted RSI)"""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    
    # Calculate positive and negative money flow
    delta = typical_price.diff()
    positive_flow = raw_money_flow.where(delta > 0, 0)
    negative_flow = raw_money_flow.where(delta < 0, 0)
    
    # Sum over period
    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()
    
    # Money Flow Ratio and MFI
    mfr = positive_sum / negative_sum
    mfi = 100 - (100 / (1 + mfr))
    
    return mfi

def calculate_vwap(high, low, close, volume):
    """Calculate Volume Weighted Average Price (rolling daily reset)"""
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap

def calculate_rvol(volume, period=20):
    """Calculate Relative Volume (current vs 20-day average)"""
    avg_volume = volume.rolling(window=period).mean()
    rvol = volume / avg_volume
    return rvol

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range for dynamic stop-loss"""
    # True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX) for trend strength.
    ADX > 25 indicates a strong trend (up or down).
    ADX < 20 indicates a weak/ranging market.
    """
    # True Range
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # When +DM > -DM, use +DM; else 0
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    atr = true_range.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    
    # ADX = smoothed(|+DI - -DI| / (+DI + -DI))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()
    
    return adx

def flatten_columns(df):
    """Flatten MultiIndex columns from yfinance"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

def add_technical_indicators(df):
    """Add technical indicators including volume-weighted metrics"""
    # Flatten MultiIndex columns if present (yfinance issue)
    df = flatten_columns(df)
    
    # Ensure columns are Series, not DataFrame
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
    
    # Moving Averages
    df['MA_10'] = close.rolling(window=10).mean()
    df['MA_20'] = close.rolling(window=20).mean()
    df['MA_50'] = close.rolling(window=50).mean()
    df['MA_200'] = close.rolling(window=200).mean()  # Macro regime filter
    
    # RSI with tuned period
    df['RSI'] = calculate_rsi(close, period=RSI_PERIOD)
    
    # MACD with tuned parameters
    macd, macd_signal, macd_hist = calculate_macd(close)
    df['MACD'] = macd.values if hasattr(macd, 'values') else macd
    df['MACD_Signal'] = macd_signal.values if hasattr(macd_signal, 'values') else macd_signal
    df['MACD_Hist'] = macd_hist.values if hasattr(macd_hist, 'values') else macd_hist
    
    # Bollinger Bands
    bb_middle = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std()
    df['BB_Middle'] = bb_middle.values if hasattr(bb_middle, 'values') else bb_middle
    df['BB_Upper'] = (bb_middle + (bb_std * 2)).values
    df['BB_Lower'] = (bb_middle - (bb_std * 2)).values
    
    # Price momentum
    df['Momentum'] = close.pct_change(periods=10).values
    
    # Volatility
    df['Volatility'] = close.rolling(window=20).std().values
    
    # NEW: Volume-Weighted Indicators
    df['OBV'] = calculate_obv(close, volume).values  # On-Balance Volume
    df['MFI'] = calculate_mfi(high, low, close, volume, period=MFI_PERIOD).values  # Money Flow Index
    df['VWAP'] = calculate_vwap(high, low, close, volume).values  # VWAP
    df['RVOL'] = calculate_rvol(volume, period=20).values  # Relative Volume
    df['ATR'] = calculate_atr(high, low, close, period=14).values  # Average True Range
    df['ADX'] = calculate_adx(high, low, close, period=14).values  # Average Directional Index (trend strength)
    
    # Normalize OBV for scaling (use percent change)
    df['OBV_Norm'] = df['OBV'].pct_change().rolling(window=5).mean() * 100
    
    # Above/Below 200-day SMA (binary for regime)
    df['Above_MA200'] = (close > df['MA_200']).astype(float)
    
    # Drop NaN rows
    df = df.dropna()
    
    return df

def build_lstm_model(input_shape):
    """Build LSTM model with improved architecture"""
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=25, activation='relu'))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def get_early_stopping():
    """Create early stopping callback"""
    return EarlyStopping(
        monitor='loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

def prepare_training_data_with_features(df, prediction_days):
    """Prepare training sequences with technical indicators"""
    # Add technical indicators
    df = add_technical_indicators(df.copy())
    
    # Updated feature columns with volume-weighted indicators
    feature_columns = ['Close', 'MA_10', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal', 
                       'MACD_Hist', 'BB_Upper', 'BB_Lower', 'Momentum', 'Volatility',
                       'OBV_Norm', 'MFI', 'RVOL', 'Above_MA200']
    
    data = df[feature_columns].values
    
    # Scale all features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create price-only scaler for inverse transform later
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    price_scaler.fit_transform(df[['Close']].values)
    
    x_train, y_train = [], []
    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i-prediction_days:i])  # All features
        y_train.append(scaled_data[i, 0])  # Only Close price as target
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    
    return x_train, y_train, scaler, price_scaler, len(feature_columns), feature_columns

def prepare_training_data(data, prediction_days):
    """Prepare training sequences from price data (single feature)"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    x_train, y_train = [], []
    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i-prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    return x_train, y_train, scaler

def prepare_test_data_with_features(train_df, test_df, scaler, prediction_days, feature_columns):
    """Prepare test sequences with technical indicators"""
    # Combine train and test for indicator calculation continuity
    combined_df = pd.concat([train_df, test_df], axis=0)
    combined_df = add_technical_indicators(combined_df.copy())
    
    # Get only the test portion plus lookback
    start_idx = len(combined_df) - len(test_df) - prediction_days
    model_inputs_df = combined_df.iloc[start_idx:]
    
    data = model_inputs_df[feature_columns].values
    scaled_data = scaler.transform(data)
    
    x_test = []
    for i in range(prediction_days, len(scaled_data)):
        x_test.append(scaled_data[i-prediction_days:i])
    
    x_test = np.array(x_test)
    return x_test

def prepare_test_data(train_data, test_data, scaler, prediction_days):
    """Prepare test sequences (single feature)"""
    total_dataset = pd.concat((train_data, test_data), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)
    
    x_test = []
    for i in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[i-prediction_days:i, 0])
    
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    return x_test

def calculate_directional_accuracy(actual_prices, predicted_prices):
    """Calculate percentage of correct direction predictions"""
    actual_direction = np.diff(actual_prices.flatten()) > 0
    predicted_direction = np.diff(predicted_prices.flatten()) > 0
    correct = np.sum(actual_direction == predicted_direction)
    return (correct / len(actual_direction)) * 100

def calculate_mse(actual_prices, predicted_prices):
    """Calculate Mean Squared Error"""
    return np.mean((actual_prices.flatten() - predicted_prices.flatten()) ** 2)

def calculate_rmse(actual_prices, predicted_prices):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(calculate_mse(actual_prices, predicted_prices))

def calculate_mape(actual_prices, predicted_prices):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((actual_prices.flatten() - predicted_prices.flatten()) / actual_prices.flatten())) * 100

def simulate_investment(actual_prices, predicted_prices, starting_capital, 
                       ma_200=None, ma_50=None, rvol=None, low_prices=None, high_prices=None,
                       atr=None, rsi=None, adx=None,
                       threshold=CONFIDENCE_THRESHOLD):
    """
    Simulate investment strategy (LONG_ONLY or HYBRID with Sniper Shorts).
    
    LONG_ONLY Mode:
    - Only buy when macro is bullish (Close > 200 SMA)
    - Volume confirmation (RVOL > 0.75)
    - Dynamic ATR-based stop-loss
    
    HYBRID Mode (adds Sniper Shorts):
    - All LONG_ONLY rules, PLUS...
    - SHORT only when: Price < 200 SMA AND ADX > 25 AND RSI < 50
    - Shorts get their own ATR stop (Entry + 2*ATR for shorts)
    
    Returns total PnL, daily returns, detailed trade history, and max drawdown
    """
    # Ensure arrays are same length
    min_len = min(len(actual_prices), len(predicted_prices))
    actual_prices = actual_prices[:min_len]
    predicted_prices = predicted_prices[:min_len]
    
    if ma_200 is not None:
        ma_200 = ma_200[:min_len]
    if ma_50 is not None:
        ma_50 = ma_50[:min_len]
    if rvol is not None:
        rvol = rvol[:min_len]
    if low_prices is not None:
        low_prices = low_prices[:min_len]
    if high_prices is not None:
        high_prices = high_prices[:min_len]
    if atr is not None:
        atr = atr[:min_len]
    if rsi is not None:
        rsi = rsi[:min_len]
    if adx is not None:
        adx = adx[:min_len]
    
    if min_len < 2:
        return 0.0, [], [], 0.0
    
    capital = starting_capital
    position = 0  # Number of shares (0 = CASH, >0 = LONG, <0 = SHORT)
    position_type = None  # 'LONG' or 'SHORT'
    position_entry_price = 0
    position_stop_price = 0  # Dynamic stop based on ATR at entry
    entry_day = 0
    daily_returns = []
    trade_history = []
    
    # Track for max drawdown calculation
    peak_capital = starting_capital
    max_drawdown = 0.0
    capital_history = [starting_capital]
    
    for i in range(min_len - 1):
        current_price = float(actual_prices[i])
        next_actual_price = float(actual_prices[i + 1])
        current_low = float(low_prices[i]) if low_prices is not None else current_price
        current_high = float(high_prices[i]) if high_prices is not None else current_price
        
        # Get current indicators
        current_ma200 = float(ma_200[i]) if ma_200 is not None else 0
        current_ma50 = float(ma_50[i]) if ma_50 is not None else 0
        current_rvol = float(rvol[i]) if rvol is not None else 1.0
        current_atr = float(atr[i]) if atr is not None else current_price * 0.02
        current_rsi = float(rsi[i]) if rsi is not None else 50.0
        current_adx = float(adx[i]) if adx is not None else 20.0
        
        # Get prediction for next day
        if i < min_len - 1:
            predicted_next_price = float(predicted_prices[i + 1])
        else:
            predicted_next_price = float(predicted_prices[i])
        
        # Calculate predicted percent change
        pred_change_pct = (predicted_next_price - current_price) / current_price if current_price != 0 else 0
        
        # ==========================================
        # STEP 1: Check Dynamic ATR Stop-Loss (if in position)
        # ==========================================
        stop_loss_triggered = False
        
        # LONG stop-loss
        if position > 0 and position_type == 'LONG':
            if current_low <= position_stop_price:
                pnl = position * (position_stop_price - position_entry_price)
                capital += pnl + (position * position_entry_price)
                stop_loss_pct = ((position_stop_price - position_entry_price) / position_entry_price) * 100
                
                trade_history.append({
                    'trade_num': len(trade_history) + 1,
                    'type': 'LONG',
                    'entry_day': entry_day,
                    'exit_day': i,
                    'entry_price': position_entry_price,
                    'exit_price': position_stop_price,
                    'shares': abs(position),
                    'pnl': pnl,
                    'pnl_percent': (pnl / (abs(position) * position_entry_price)) * 100,
                    'exit_reason': f'ATR-STOP ({abs(stop_loss_pct):.1f}%)'
                })
                
                position = 0
                position_type = None
                position_entry_price = 0
                position_stop_price = 0
                stop_loss_triggered = True
        
        # SHORT stop-loss (stop is ABOVE entry for shorts)
        elif position < 0 and position_type == 'SHORT':
            if current_high >= position_stop_price:
                # Short was stopped out at loss
                pnl = abs(position) * (position_entry_price - position_stop_price)
                capital += pnl + (abs(position) * position_entry_price)
                stop_loss_pct = ((position_stop_price - position_entry_price) / position_entry_price) * 100
                
                trade_history.append({
                    'trade_num': len(trade_history) + 1,
                    'type': 'SHORT',
                    'entry_day': entry_day,
                    'exit_day': i,
                    'entry_price': position_entry_price,
                    'exit_price': position_stop_price,
                    'shares': abs(position),
                    'pnl': pnl,
                    'pnl_percent': (pnl / (abs(position) * position_entry_price)) * 100,
                    'exit_reason': f'ATR-STOP ({abs(stop_loss_pct):.1f}%)'
                })
                
                position = 0
                position_type = None
                position_entry_price = 0
                position_stop_price = 0
                stop_loss_triggered = True
        
        if stop_loss_triggered:
            daily_returns.append(0.0)
            capital_history.append(capital)
            continue
        
        # ==========================================
        # STEP 2: Macro & Trend Analysis
        # ==========================================
        macro_bullish = True
        macro_bearish = False
        if USE_MACRO_FILTER and ma_200 is not None:
            macro_bullish = current_price > current_ma200
            macro_bearish = current_price < current_ma200
        
        # ==========================================
        # STEP 3: Trading Logic
        # ==========================================
        action = None
        exit_reason = None
        entry_reason = None
        
        # --- LONG LOGIC ---
        if macro_bullish:
            ml_buy_signal = pred_change_pct > threshold
            volume_ok = current_rvol > RVOL_THRESHOLD
            strong_trend = (current_price > current_ma200) and (current_price > current_ma50 if current_ma50 > 0 else True)
            
            if position == 0:
                if ml_buy_signal and volume_ok:
                    action = 'BUY'
                    entry_reason = 'ML_SIGNAL'
                elif USE_TREND_OVERRIDE and strong_trend and volume_ok:
                    action = 'BUY'
                    entry_reason = 'TREND_OVERRIDE'
            elif position > 0:
                if pred_change_pct < -threshold:
                    action = 'SELL'
                    exit_reason = 'SIGNAL_SELL'
            elif position < 0:
                # Close short if macro turns bullish
                action = 'COVER'
                exit_reason = 'MACRO_FILTER'
        
        # --- SHORT LOGIC (SNIPER MODE - Only in HYBRID) ---
        elif macro_bearish and STRATEGY_MODE == 'HYBRID':
            # Sniper Short Conditions:
            # 1. Price < 200 SMA (stock is weak) - already checked via macro_bearish
            # 2. ADX > 25 (trend is strong)
            # 3. RSI < 50 (momentum is bearish)
            # 4. ML predicts down
            sniper_conditions_met = (
                current_adx > SNIPER_ADX_THRESHOLD and
                current_rsi < SNIPER_RSI_THRESHOLD and
                current_rvol > RVOL_THRESHOLD
            )
            ml_sell_signal = pred_change_pct < -threshold
            
            if position == 0:
                if sniper_conditions_met and ml_sell_signal:
                    action = 'SHORT'
                    entry_reason = 'SNIPER_SHORT'
            elif position < 0:
                # Cover short if: ML turns bullish OR momentum weakens
                if pred_change_pct > threshold or current_rsi > 50:
                    action = 'COVER'
                    exit_reason = 'SIGNAL_COVER'
            elif position > 0:
                # Close long if macro turns bearish
                action = 'SELL'
                exit_reason = 'MACRO_FILTER'
        
        # --- NEUTRAL (Long-Only mode with bearish macro) ---
        elif macro_bearish and STRATEGY_MODE == 'LONG_ONLY':
            if position > 0:
                action = 'SELL'
                exit_reason = 'MACRO_FILTER'
        
        # ==========================================
        # STEP 4: Execute Trade
        # ==========================================
        if action == 'SELL' and position > 0:
            pnl = position * (current_price - position_entry_price)
            capital += pnl + (position * position_entry_price)
            
            trade_history.append({
                'trade_num': len(trade_history) + 1,
                'type': 'LONG',
                'entry_day': entry_day,
                'exit_day': i,
                'entry_price': position_entry_price,
                'exit_price': current_price,
                'shares': abs(position),
                'pnl': pnl,
                'pnl_percent': (pnl / (abs(position) * position_entry_price)) * 100,
                'exit_reason': exit_reason
            })
            
            position = 0
            position_type = None
            position_entry_price = 0
            position_stop_price = 0
            
        elif action == 'BUY' and position == 0:
            position = capital / current_price
            position_type = 'LONG'
            position_entry_price = current_price
            position_stop_price = current_price - (ATR_STOP_MULTIPLIER * current_atr)
            entry_day = i
            capital = 0
            
        elif action == 'SHORT' and position == 0:
            position = -(capital / current_price)  # Negative for short
            position_type = 'SHORT'
            position_entry_price = current_price
            # Short stop is ABOVE entry: Entry + (2 * ATR)
            position_stop_price = current_price + (ATR_STOP_MULTIPLIER * current_atr)
            entry_day = i
            capital = 0
            
        elif action == 'COVER' and position < 0:
            # Close short position: profit = (entry - exit) * shares
            pnl = abs(position) * (position_entry_price - current_price)
            capital += pnl + (abs(position) * position_entry_price)
            
            trade_history.append({
                'trade_num': len(trade_history) + 1,
                'type': 'SHORT',
                'entry_day': entry_day,
                'exit_day': i,
                'entry_price': position_entry_price,
                'exit_price': current_price,
                'shares': abs(position),
                'pnl': pnl,
                'pnl_percent': (pnl / (abs(position) * position_entry_price)) * 100,
                'exit_reason': exit_reason
            })
            
            position = 0
            position_type = None
            position_entry_price = 0
            position_stop_price = 0
        
        # ==========================================
        # STEP 5: Calculate Daily P&L
        # ==========================================
        if position > 0:
            price_change = next_actual_price - current_price
            daily_pnl = position * price_change
            current_value = position * next_actual_price
        elif position < 0:
            price_change = current_price - next_actual_price  # Shorts profit when price drops
            daily_pnl = abs(position) * price_change
            current_value = abs(position) * position_entry_price + daily_pnl
        else:
            daily_pnl = 0.0
            current_value = capital
        
        daily_returns.append(float(daily_pnl))
        capital_history.append(current_value if position != 0 else capital)
        
        # Update max drawdown
        if capital_history[-1] > peak_capital:
            peak_capital = capital_history[-1]
        drawdown = (peak_capital - capital_history[-1]) / peak_capital if peak_capital > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    
    # ==========================================
    # Close final position
    # ==========================================
    final_price = float(actual_prices[-1])
    if position > 0:
        pnl = position * (final_price - position_entry_price)
        capital = position * final_price
        
        trade_history.append({
            'trade_num': len(trade_history) + 1,
            'type': 'LONG',
            'entry_day': entry_day,
            'exit_day': min_len - 1,
            'entry_price': position_entry_price,
            'exit_price': final_price,
            'shares': abs(position),
            'pnl': pnl,
            'pnl_percent': (pnl / (abs(position) * position_entry_price)) * 100,
            'exit_reason': 'END_OF_PERIOD'
        })
    elif position < 0:
        pnl = abs(position) * (position_entry_price - final_price)
        capital = abs(position) * position_entry_price + pnl
        
        trade_history.append({
            'trade_num': len(trade_history) + 1,
            'type': 'SHORT',
            'entry_day': entry_day,
            'exit_day': min_len - 1,
            'entry_price': position_entry_price,
            'exit_price': final_price,
            'shares': abs(position),
            'pnl': pnl,
            'pnl_percent': (pnl / (abs(position) * position_entry_price)) * 100,
            'exit_reason': 'END_OF_PERIOD'
        })
    
    total_pnl = capital - starting_capital
    return float(total_pnl), daily_returns, trade_history, float(max_drawdown * 100)

def monte_carlo_simulation(model, last_sequence, price_scaler, n_simulations=1000, n_days=14, n_features=1):
    """Run Monte Carlo simulation for future predictions"""
    predictions = []
    
    for _ in range(n_simulations):
        current_sequence = last_sequence.copy()
        sim_predictions = []
        
        for _ in range(n_days):
            # Handle both single and multi-feature models
            if n_features > 1:
                pred = model.predict(current_sequence.reshape(1, current_sequence.shape[0], n_features), verbose=0)[0, 0]
            else:
                pred = model.predict(current_sequence.reshape(1, -1, 1), verbose=0)[0, 0]
            
            # Add random noise based on model uncertainty
            noise = np.random.normal(0, 0.02)  # 2% standard deviation
            pred_with_noise = pred * (1 + noise)
            sim_predictions.append(pred_with_noise)
            
            # Update sequence - for multi-feature, only update close price column
            if n_features > 1:
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1, 0] = pred_with_noise  # Update close price
            else:
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = pred_with_noise
        
        predictions.append(sim_predictions)
    
    predictions = np.array(predictions)
    # Inverse transform predictions using price scaler
    mean_pred = np.mean(predictions, axis=0).reshape(-1, 1)
    return price_scaler.inverse_transform(mean_pred)

# ==========================================
# MAIN ANALYSIS
# ==========================================

def analyze_stock(symbol):
    """Run complete analysis for a single stock"""
    print(f"\n{'='*50}")
    print(f"Analyzing {symbol}...")
    print(f"{'='*50}")
    
    try:
        model = None
        model_info = None
        scaler = None
        price_scaler = None
        feature_columns = None
        use_features = False
        
        # Try to load existing model if enabled
        if LOAD_EXISTING_MODELS:
            print(f"  Checking for saved model...")
            model, model_info, is_up_to_date = load_saved_model(symbol, expected_training_end=TEST_END)
            
            if model is not None and model_info is not None:
                saved_date = model_info.get('training_end_date', 'unknown')
                
                if is_up_to_date:
                    print(f"  ✓ Model is UP-TO-DATE (trained through {saved_date})")
                    scaler = model_info.get('scaler')
                    price_scaler = model_info.get('price_scaler')
                    feature_columns = model_info.get('feature_columns')
                    use_features = model_info.get('use_features', False)
                    print(f"  Model uses technical indicators: {use_features}")
                elif AUTO_UPDATE_MODELS:
                    print(f"  ⚠ Model is OUTDATED (trained through {saved_date}, need {TEST_END})")
                    print(f"  Retraining model with updated data...")
                    model = None  # Force retrain
                    model_info = None
                else:
                    print(f"  ⚠ Model is OUTDATED but AUTO_UPDATE_MODELS=False, using anyway")
                    scaler = model_info.get('scaler')
                    price_scaler = model_info.get('price_scaler')
                    feature_columns = model_info.get('feature_columns')
                    use_features = model_info.get('use_features', False)
        
        # Download training data (needed for scaler if not loaded, or for training)
        print(f"  Downloading training data ({TRAINING_START} to {TRAINING_END})...")
        train_data = yf.download(symbol, start=TRAINING_START, end=TRAINING_END, progress=False)
        
        if len(train_data) < PREDICTION_DAYS + 100:
            print(f"  ERROR: Not enough training data for {symbol}")
            return None
        
        if model is None or scaler is None:
            # Use technical indicators if enabled
            if USE_TECHNICAL_INDICATORS:
                print(f"  Adding technical indicators (MA, RSI, MACD, Volume-Weighted)...")
                x_train, y_train, scaler, price_scaler, n_features, feature_columns = prepare_training_data_with_features(
                    train_data, PREDICTION_DAYS
                )
                use_features = True
                
                # Build and train model with early stopping
                print(f"  Building and training LSTM model (with early stopping)...")
                model = build_lstm_model((x_train.shape[1], n_features))
                early_stopping = get_early_stopping()
                model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                          callbacks=[early_stopping], verbose=1)
                
                # Save additional info for later
                model_info = {
                    'scaler': scaler,
                    'price_scaler': price_scaler,
                    'feature_columns': feature_columns,
                    'use_features': True,
                    'training_end_date': TRAINING_END,
                    'training_start_date': TRAINING_START
                }
            else:
                train_close = train_data[['Close']].values
                print(f"  Preparing training data...")
                x_train, y_train, scaler = prepare_training_data(train_close, PREDICTION_DAYS)
                price_scaler = scaler
                feature_columns = None
                use_features = False
                
                # Build and train model with early stopping
                print(f"  Building and training LSTM model (with early stopping)...")
                model = build_lstm_model((x_train.shape[1], 1))
                early_stopping = get_early_stopping()
                model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                          callbacks=[early_stopping], verbose=1)
                
                model_info = {
                    'scaler': scaler,
                    'price_scaler': scaler,
                    'feature_columns': None,
                    'use_features': False,
                    'training_end_date': TRAINING_END,
                    'training_start_date': TRAINING_START
                }
            
            # Save the model if enabled
            if SAVE_MODELS:
                if not os.path.exists(MODEL_DIR):
                    os.makedirs(MODEL_DIR)
                model_path = os.path.join(MODEL_DIR, f"{symbol}_lstm_model.keras")
                info_path = os.path.join(MODEL_DIR, f"{symbol}_model_info.pkl")
                model.save(model_path)
                with open(info_path, 'wb') as f:
                    pickle.dump(model_info, f)
                print(f"  Model saved to: {model_path}")
        
        # Download test data (past 2 months)
        print(f"  Downloading test data ({TEST_START} to {TEST_END})...")
        test_data = yf.download(symbol, start=TEST_START, end=TEST_END, progress=False)
        
        if len(test_data) < 10:
            print(f"  ERROR: Not enough test data for {symbol}")
            return None
        
        actual_prices = test_data['Close'].values
        
        # Prepare test data based on whether the model uses features
        if use_features and feature_columns is not None:
            x_test = prepare_test_data_with_features(train_data, test_data, scaler, PREDICTION_DAYS, feature_columns)
        else:
            test_close = test_data['Close']
            train_close_series = train_data['Close']
            x_test = prepare_test_data(train_close_series, test_close, scaler, PREDICTION_DAYS)
        
        # Make predictions
        print(f"  Making predictions...")
        predicted_prices = model.predict(x_test, verbose=0)
        predicted_prices = price_scaler.inverse_transform(predicted_prices.reshape(-1, 1))
        
        # Ensure arrays are same length for metrics
        min_len = min(len(actual_prices), len(predicted_prices))
        actual_prices = actual_prices[:min_len]
        predicted_prices = predicted_prices[:min_len].flatten()
        
        # Calculate metrics
        print(f"  Calculating metrics...")
        directional_accuracy = calculate_directional_accuracy(actual_prices, predicted_prices)
        mse = calculate_mse(actual_prices, predicted_prices)
        rmse = calculate_rmse(actual_prices, predicted_prices)
        mape = calculate_mape(actual_prices, predicted_prices)
        
        # Get additional data for simulation (MA-200, MA-50, RVOL, Low prices, ATR)
        # Need to recalculate on combined data to get proper values for test period
        combined_data = pd.concat([train_data, test_data], axis=0)
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
        
        # Calculate indicators for test period
        ma_200_full = close_series.rolling(window=200).mean()
        ma_50_full = close_series.rolling(window=50).mean()
        rvol_full = calculate_rvol(volume_series, period=20)
        atr_full = calculate_atr(high_series, low_series, close_series, period=14)
        rsi_full = calculate_rsi(close_series, period=RSI_PERIOD)
        adx_full = calculate_adx(high_series, low_series, close_series, period=14)
        
        # Extract test period values
        test_start_idx = len(combined_data) - len(test_data)
        ma_200_test = ma_200_full.iloc[test_start_idx:test_start_idx + min_len].values
        ma_50_test = ma_50_full.iloc[test_start_idx:test_start_idx + min_len].values
        rvol_test = rvol_full.iloc[test_start_idx:test_start_idx + min_len].values
        atr_test = atr_full.iloc[test_start_idx:test_start_idx + min_len].values
        rsi_test = rsi_full.iloc[test_start_idx:test_start_idx + min_len].values
        adx_test = adx_full.iloc[test_start_idx:test_start_idx + min_len].values
        low_test = low_series.iloc[test_start_idx:test_start_idx + min_len].values
        high_test = high_series.iloc[test_start_idx:test_start_idx + min_len].values
        
        # Simulate investment with strategy (Long-Only or Hybrid with Sniper Shorts)
        strategy_desc = "LONG-ONLY" if STRATEGY_MODE == 'LONG_ONLY' else "HYBRID (Sniper Shorts)"
        print(f"  Simulating {strategy_desc} strategy (ATR Stop: {ATR_STOP_MULTIPLIER}x, RVOL: {RVOL_THRESHOLD})...")
        total_pnl, daily_returns, trade_history, max_drawdown = simulate_investment(
            actual_prices, predicted_prices, STARTING_CAPITAL,
            ma_200=ma_200_test, ma_50=ma_50_test, rvol=rvol_test, 
            low_prices=low_test, high_prices=high_test, 
            atr=atr_test, rsi=rsi_test, adx=adx_test,
            threshold=CONFIDENCE_THRESHOLD
        )
        pnl_percent = (total_pnl / STARTING_CAPITAL) * 100
        
        # Calculate Buy & Hold comparison
        buy_hold_return = ((actual_prices[-1] - actual_prices[0]) / actual_prices[0]) * 100
        buy_hold_pnl = STARTING_CAPITAL * (buy_hold_return / 100)
        
        # Calculate win rate
        winning_trades = [t for t in trade_history if t['pnl'] > 0]
        total_trades = len(trade_history)
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
        
        # Monte Carlo simulation
        print(f"  Running Monte Carlo simulation ({MONTE_CARLO_SIMULATIONS} paths)...")
        last_sequence = x_test[-1]
        n_features = last_sequence.shape[-1] if len(last_sequence.shape) > 1 else 1
        mc_predictions = monte_carlo_simulation(model, last_sequence, price_scaler, MONTE_CARLO_SIMULATIONS, n_features=n_features)
        
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
            'mc_14day_prediction': float(mc_predictions[-1][0]) if len(mc_predictions) > 0 else None,
            'trade_history': trade_history,
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'total_trades': total_trades,
            'buy_hold_return': float(buy_hold_return),
            'buy_hold_pnl': float(buy_hold_pnl)
        }
        
        print(f"  ✓ Analysis complete for {symbol}")
        return results
        
    except Exception as e:
        import traceback
        print(f"  ERROR analyzing {symbol}: {str(e)}")
        traceback.print_exc()
        return None

def generate_report(results):
    """Generate markdown report with enhanced metrics"""
    report = []
    
    # Dynamic title based on strategy mode
    strategy_title = "Long-Only Strategy" if STRATEGY_MODE == 'LONG_ONLY' else "Hybrid Strategy (Sniper Shorts)"
    report.append(f"# Monte Carlo Stock Analysis Report ({strategy_title})")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Analysis Period:** {TEST_START} to {TEST_END} (Past {TEST_DAYS} Days)")
    report.append(f"\n**Training Period:** {TRAINING_START} to {TRAINING_END} (Rolling {ROLLING_WINDOW_YEARS}-Year Window)")
    report.append(f"\n**Monte Carlo Simulations:** {MONTE_CARLO_SIMULATIONS}")
    report.append(f"\n**Starting Capital per Stock:** ${STARTING_CAPITAL:,.2f}")
    report.append(f"\n**Training Epochs:** {EPOCHS} (with early stopping, patience={EARLY_STOPPING_PATIENCE})")
    
    # Strategy Parameters
    report.append("\n\n## Strategy Configuration\n")
    report.append(f"| Parameter | Value |")
    report.append(f"|-----------|-------|")
    report.append(f"| Strategy Mode | **{STRATEGY_MODE}** |")
    report.append(f"| Confidence Threshold | {CONFIDENCE_THRESHOLD*100}% |")
    report.append(f"| Stop-Loss Type | Dynamic ATR ({ATR_STOP_MULTIPLIER}x ATR) |")
    report.append(f"| RVOL Threshold | {RVOL_THRESHOLD} (relaxed to avoid dead volume only) |")
    report.append(f"| Macro Filter (200-SMA) | {'Enabled' if USE_MACRO_FILTER else 'Disabled'} |")
    report.append(f"| Trend Override | {'Enabled (Buy when Price > 200 SMA AND > 50 SMA)' if USE_TREND_OVERRIDE else 'Disabled'} |")
    
    # Sniper Short parameters (only show if HYBRID mode)
    if STRATEGY_MODE == 'HYBRID':
        report.append(f"| **Sniper Short Enabled** | **YES** |")
        report.append(f"| Short ADX Threshold | > {SNIPER_ADX_THRESHOLD} (strong trend) |")
        report.append(f"| Short RSI Threshold | < {SNIPER_RSI_THRESHOLD} (bearish momentum) |")
        report.append(f"| Short Conditions | Price < 200 SMA + ADX > {SNIPER_ADX_THRESHOLD} + RSI < {SNIPER_RSI_THRESHOLD} |")
    else:
        report.append(f"| Sniper Short | Disabled (Long-Only Mode) |")
    
    report.append(f"| RSI Period | {RSI_PERIOD} |")
    report.append(f"| MACD Parameters | ({MACD_FAST}, {MACD_SLOW}, {MACD_SIGNAL}) |")
    report.append(f"| MFI Period | {MFI_PERIOD} |")
    report.append(f"| Volume Indicators | OBV, MFI, VWAP, RVOL, ATR, ADX |")
    
    # Summary Table
    report.append("\n\n## Summary Results\n")
    report.append("| Stock | Strategy Return | Buy & Hold | Win Rate | Max DD | Trades | Status |")
    report.append("|-------|-----------------|------------|----------|--------|--------|--------|")
    
    total_pnl = 0
    total_buy_hold_pnl = 0
    valid_results = [r for r in results if r is not None]
    
    if len(valid_results) == 0:
        report.append("\nNo valid results to report.\n")
        return "\n".join(report)
    
    for r in valid_results:
        status_emoji = "✅" if r['status'] == 'GAIN' else "❌"
        outperform = "📈" if r['pnl_percent'] > r['buy_hold_return'] else "📉"
        report.append(
            f"| {r['symbol']} | {r['pnl_percent']:+.1f}% (${r['total_pnl']:+,.0f}) | "
            f"{r['buy_hold_return']:+.1f}% {outperform} | {r['win_rate']:.0f}% | "
            f"{r['max_drawdown']:.1f}% | {r['total_trades']} | {status_emoji} {r['status']} |"
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
    
    # Detailed Results
    report.append("\n\n## Detailed Stock Analysis\n")
    
    for r in valid_results:
        report.append(f"\n### {r['symbol']}\n")
        report.append(f"| Metric | Value |")
        report.append(f"|--------|-------|")
        report.append(f"| Start Price | ${r['start_price']:.2f} |")
        report.append(f"| End Price | ${r['end_price']:.2f} |")
        report.append(f"| Directional Accuracy | {r['directional_accuracy']:.2f}% |")
        report.append(f"| Mean Squared Error | {r['mse']:.4f} |")
        report.append(f"| Root Mean Squared Error | ${r['rmse']:.2f} |")
        report.append(f"| Mean Absolute Percentage Error | {r['mape']:.2f}% |")
        report.append(f"| Starting Capital | ${r['starting_capital']:,.2f} |")
        report.append(f"| Final Capital | ${r['final_capital']:,.2f} |")
        report.append(f"| **Strategy Return** | **{r['pnl_percent']:+.2f}%** |")
        report.append(f"| Buy & Hold Return | {r['buy_hold_return']:+.2f}% |")
        report.append(f"| Max Drawdown | {r['max_drawdown']:.2f}% |")
        report.append(f"| Win Rate | {r['win_rate']:.1f}% |")
        report.append(f"| Total Trades | {r['total_trades']} |")
        report.append(f"| Trading Days | {r['test_days']} |")
        if r['mc_14day_prediction']:
            report.append(f"| MC 14-Day Prediction | ${r['mc_14day_prediction']:.2f} |")
        
        # Trade History Breakdown
        if r.get('trade_history') and len(r['trade_history']) > 0:
            report.append(f"\n#### Trade History for {r['symbol']}\n")
            report.append("| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |")
            report.append("|---|-----------|----------|---------|--------|--------|-----|--------|-------------|")
            
            for t in r['trade_history']:
                status_icon = "+" if t['pnl'] >= 0 else ""
                exit_reason = t.get('exit_reason', 'N/A')
                # Add emoji for exit reason
                reason_emoji = {
                    'STOP-LOSS': '🛑',
                    'SIGNAL_SELL': '📊',
                    'MACRO_FILTER': '📉',
                    'END_OF_PERIOD': '⏰'
                }.get(exit_reason, '')
                report.append(
                    f"| {t['trade_num']} | Day {t['entry_day']} | Day {t['exit_day']} | "
                    f"${t['entry_price']:.2f} | ${t['exit_price']:.2f} | {t['shares']:.2f} | "
                    f"${status_icon}{t['pnl']:.2f} | {status_icon}{t['pnl_percent']:.2f}% | {reason_emoji} {exit_reason} |"
                )
            
            # Trade summary with exit reason breakdown
            winning_trades = [t for t in r['trade_history'] if t['pnl'] > 0]
            losing_trades = [t for t in r['trade_history'] if t['pnl'] <= 0]
            stop_loss_trades = [t for t in r['trade_history'] if t.get('exit_reason') == 'STOP-LOSS']
            signal_sells = [t for t in r['trade_history'] if t.get('exit_reason') == 'SIGNAL_SELL']
            macro_exits = [t for t in r['trade_history'] if t.get('exit_reason') == 'MACRO_FILTER']
            total_trades = len(r['trade_history'])
            
            report.append(f"\n**Trade Summary:**")
            report.append(f"- Total Trades: {total_trades}")
            report.append(f"- Winning Trades: {len(winning_trades)} ({len(winning_trades)/total_trades*100:.1f}%)")
            report.append(f"- Losing Trades: {len(losing_trades)} ({len(losing_trades)/total_trades*100:.1f}%)")
            if winning_trades:
                report.append(f"- Total Gains: ${sum(t['pnl'] for t in winning_trades):,.2f}")
            if losing_trades:
                report.append(f"- Total Losses: ${sum(t['pnl'] for t in losing_trades):,.2f}")
            
            report.append(f"\n**Exit Reason Breakdown:**")
            report.append(f"- 🛑 Stop-Loss Hits: {len(stop_loss_trades)}")
            report.append(f"- 📊 Signal Sells: {len(signal_sells)}")
            report.append(f"- 📉 Macro Filter Exits: {len(macro_exits)}")
            
            # Count sniper short trades
            short_trades = [t for t in r['trade_history'] if t.get('type') == 'SHORT']
            if short_trades:
                report.append(f"- 📉 Sniper Short Trades: {len(short_trades)}")
    
    # Metrics Explanation
    report.append("\n\n## Strategy Explanation\n")
    report.append("### Decision Logic")
    report.append("1. **Macro Filter:** Only go LONG when Close > 200-day SMA (bullish regime)")
    report.append("2. **Buy Signal (ML):** Model predicts price increase > threshold AND RVOL > 0.75")
    report.append("3. **Buy Signal (Trend Override):** Price > 200 SMA AND > 50 SMA AND RVOL > 0.75 (trust the trend)")
    report.append("4. **Sell Signal:** Model predicts significant price decrease (< -threshold)")
    report.append("5. **Stop-Loss:** Dynamic ATR-based exit (Entry - 2×ATR for longs, Entry + 2×ATR for shorts)")
    
    if STRATEGY_MODE == 'HYBRID':
        report.append("")
        report.append("### Sniper Short Logic (HYBRID Mode)")
        report.append(f"Shorts are ONLY allowed when ALL conditions are met:")
        report.append(f"1. **Price < 200 SMA:** Stock is in a technical downtrend")
        report.append(f"2. **ADX > {SNIPER_ADX_THRESHOLD}:** Trend is strong (not ranging)")
        report.append(f"3. **RSI < {SNIPER_RSI_THRESHOLD}:** Momentum is bearish")
        report.append(f"4. **ML predicts down:** Model confirms downward movement")
        report.append(f"5. **RVOL > {RVOL_THRESHOLD}:** Volume confirms the move")
        report.append("")
        report.append("**Short Exit Conditions:**")
        report.append("- RSI rises above 50 (momentum shift)")
        report.append("- ML predicts upward movement")
        report.append("- ATR stop-loss hit (Entry + 2×ATR)")
    
    report.append("")
    report.append("### Dynamic Stop-Loss (ATR)")
    report.append("- **Volatile stocks (NVDA, TSLA):** Stop widens to 4-5% automatically")
    report.append("- **Stable stocks (SPY):** Stop tightens to 1-2% automatically")
    report.append("- **Long Formula:** Stop Price = Entry Price - (2 × ATR)")
    report.append("- **Short Formula:** Stop Price = Entry Price + (2 × ATR)")
    report.append("")
    report.append("### Volume-Weighted Indicators")
    report.append("- **OBV (On-Balance Volume):** Detects divergence between price and volume")
    report.append("- **MFI (Money Flow Index):** Volume-weighted RSI with 21-day period")
    report.append("- **VWAP:** Identifies institutional support/resistance levels")
    report.append("- **RVOL:** Current volume vs 20-day average (>0.75 = acceptable volume)")
    report.append("- **ATR:** Average True Range for dynamic stop-loss calculation")
    report.append("- **ADX:** Average Directional Index for trend strength (>25 = strong trend)")
    report.append("")
    report.append("### Exit Reasons")
    report.append("- 🛑 **ATR-STOP:** Dynamic stop-loss hit (adapts to stock volatility)")
    report.append("- 📊 **SIGNAL_SELL:** Model prediction turned bearish (long exit)")
    report.append("- 📊 **SIGNAL_COVER:** Model prediction turned bullish (short exit)")
    report.append("- 📉 **MACRO_FILTER:** Price crossed 200-day SMA (regime change)")
    report.append("- ⏰ **END_OF_PERIOD:** Position held until analysis period ended")
    
    return "\n".join(report)

# ==========================================
# RUN ANALYSIS
# ==========================================

if __name__ == "__main__":
    # Generate timestamped report filename in reports folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    REPORTS_DIR = 'reports'
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
    REPORT_FILENAME = os.path.join(REPORTS_DIR, f"Stock_Analysis_Report_{timestamp}.md")
    
    strategy_title = "LONG-ONLY" if STRATEGY_MODE == 'LONG_ONLY' else "HYBRID (Sniper Shorts)"
    
    print("=" * 70)
    print(f"MONTE CARLO STOCK ANALYSIS - {strategy_title}")
    print("=" * 70)
    print(f"Stocks: {', '.join(STOCKS)}")
    print(f"Training Period: {TRAINING_START} to {TRAINING_END} ({ROLLING_WINDOW_YEARS}-Year Rolling Window)")
    print(f"Test Period: {TEST_START} to {TEST_END}")
    print(f"Starting Capital: ${STARTING_CAPITAL:,} per stock")
    print(f"Stop-Loss: {ATR_STOP_MULTIPLIER}x ATR (Dynamic) | Threshold: {CONFIDENCE_THRESHOLD*100}% | RVOL Min: {RVOL_THRESHOLD}")
    print(f"Trend Override: {'Enabled' if USE_TREND_OVERRIDE else 'Disabled'}")
    if STRATEGY_MODE == 'HYBRID':
        print(f"Sniper Shorts: Enabled (ADX > {SNIPER_ADX_THRESHOLD}, RSI < {SNIPER_RSI_THRESHOLD})")
    print(f"Report will be saved as: {REPORT_FILENAME}")
    print("=" * 70)
    
    results = []
    for symbol in STOCKS:
        result = analyze_stock(symbol)
        results.append(result)
    
    # Generate report
    print("\n" + "=" * 70)
    print("GENERATING REPORT...")
    print("=" * 70)
    
    report = generate_report(results)
    
    # Save report with timestamped filename
    with open(REPORT_FILENAME, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ Report saved to: {REPORT_FILENAME}")
    
    # Print summary to console
    print("\n" + "=" * 70)
    print("QUICK SUMMARY (Strategy vs Buy & Hold)")
    print("=" * 70)
    
    valid_results = [r for r in results if r is not None]
    
    if len(valid_results) == 0:
        print("No valid results to display.")
    else:
        print(f"{'Stock':6} | {'Strategy':>12} | {'Buy&Hold':>10} | {'MaxDD':>6} | {'WinRate':>7} | {'Trades':>6} | Status")
        print("-" * 70)
        for r in valid_results:
            status = "✅ GAIN" if r['status'] == 'GAIN' else "❌ LOSS"
            outperform = "📈" if r['pnl_percent'] > r['buy_hold_return'] else "📉"
            print(f"{r['symbol']:6} | {r['pnl_percent']:+10.1f}% | {r['buy_hold_return']:+8.1f}% {outperform} | "
                  f"{r['max_drawdown']:5.1f}% | {r['win_rate']:5.0f}%  | {r['total_trades']:>6} | {status}")
        
        total_pnl = sum(r['total_pnl'] for r in valid_results)
        total_buy_hold = sum(r['buy_hold_pnl'] for r in valid_results)
        total_invested = STARTING_CAPITAL * len(valid_results)
        
        print("-" * 70)
        strategy_return = (total_pnl / total_invested) * 100
        buy_hold_return = (total_buy_hold / total_invested) * 100
        print(f"TOTAL  | Strategy: {strategy_return:+.1f}% | Buy&Hold: {buy_hold_return:+.1f}%")
        print(f"       | P&L: ${total_pnl:+,.2f} vs ${total_buy_hold:+,.2f}")
        outperformance = strategy_return - buy_hold_return
        print(f"       | Outperformance: {outperformance:+.2f}% {'✅' if outperformance > 0 else '❌'}")
    print("=" * 70)
