"""
Monte Carlo Stock Analysis with LSTM Model
Analyzes: QQQ, SPY, AAPL, NVDA, AMD, INTC, GOOGL, TSLA
Generates: Directional Accuracy, MSE, and Investment Simulation Report
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
TRAINING_START = '2020-01-01'
TRAINING_END = datetime.now().strftime('%Y-%m-%d')  # Up to current system date
TEST_START = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')  # Past 2 months
TEST_END = datetime.now().strftime('%Y-%m-%d')  # Current system date
PREDICTION_DAYS = 60
STARTING_CAPITAL = 10000  # $10,000 per stock
MONTE_CARLO_SIMULATIONS = 10  # Reduced for speed
EPOCHS = 100  # Increased for better accuracy
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 10  # Stop if no improvement for 10 epochs
USE_TECHNICAL_INDICATORS = True  # Add MA, RSI, MACD as features
SAVE_MODELS = True  # Save trained LSTM models
MODEL_DIR = 'saved_models'  # Directory to save models
LOAD_EXISTING_MODELS = False  # Set to True to load pre-trained models instead of training new ones

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def load_saved_model(symbol):
    """Load a previously saved model and scaler for a symbol"""
    from tensorflow.keras.models import load_model
    
    model_path = os.path.join(MODEL_DIR, f"{symbol}_lstm_model.keras")
    scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.pkl")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    return None, None

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD and Signal line"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

def flatten_columns(df):
    """Flatten MultiIndex columns from yfinance"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

def add_technical_indicators(df):
    """Add technical indicators to dataframe"""
    # Flatten MultiIndex columns if present (yfinance issue)
    df = flatten_columns(df)
    
    # Ensure Close is a Series, not DataFrame
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    
    # Moving Averages
    df['MA_10'] = close.rolling(window=10).mean()
    df['MA_20'] = close.rolling(window=20).mean()
    df['MA_50'] = close.rolling(window=50).mean()
    
    # RSI
    df['RSI'] = calculate_rsi(close)
    
    # MACD
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
    
    # Select features for training
    feature_columns = ['Close', 'MA_10', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal', 
                       'MACD_Hist', 'BB_Upper', 'BB_Lower', 'Momentum', 'Volatility']
    
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
    
    return x_train, y_train, scaler, price_scaler, len(feature_columns)

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

def simulate_investment(actual_prices, predicted_prices, starting_capital):
    """
    Simulate investment strategy:
    - Long when model predicts price increase
    - Short when model predicts price decrease
    Returns total PnL, daily returns, and detailed trade history
    """
    # Ensure arrays are same length
    min_len = min(len(actual_prices), len(predicted_prices))
    actual_prices = actual_prices[:min_len]
    predicted_prices = predicted_prices[:min_len]
    
    if min_len < 2:
        return 0.0, [], []
    
    capital = starting_capital
    position = 0  # Number of shares
    is_long = None
    daily_returns = []
    position_entry_price = 0
    entry_day = 0
    trade_history = []  # Track all trades
    
    for i in range(min_len - 1):
        current_price = float(actual_prices[i])
        next_actual_price = float(actual_prices[i + 1])
        
        if i < min_len - 1:
            predicted_next_price = float(predicted_prices[i + 1])
        else:
            predicted_next_price = float(predicted_prices[i])
        
        # Determine if model predicts up or down
        predicted_up = predicted_next_price > float(predicted_prices[i])
        
        if i == 0:
            # Initial position
            shares = capital / current_price
            position_entry_price = current_price
            entry_day = i
            if predicted_up:
                position = shares  # Long
                is_long = True
            else:
                position = shares  # For short, track as positive but treat as short
                is_long = False
            capital = 0
        else:
            # Check if we need to switch position
            if predicted_up and not is_long:
                # Close short position and record trade
                pnl = position * (position_entry_price - current_price)
                trade_history.append({
                    'trade_num': len(trade_history) + 1,
                    'type': 'SHORT',
                    'entry_day': entry_day,
                    'exit_day': i,
                    'entry_price': position_entry_price,
                    'exit_price': current_price,
                    'shares': position,
                    'pnl': pnl,
                    'pnl_percent': (pnl / (position * position_entry_price)) * 100
                })
                # Switch from short to long
                capital = starting_capital + sum(t['pnl'] for t in trade_history)
                shares = capital / current_price
                position = shares
                position_entry_price = current_price
                entry_day = i
                is_long = True
                capital = 0
            elif not predicted_up and is_long:
                # Close long position and record trade
                pnl = position * (current_price - position_entry_price)
                trade_history.append({
                    'trade_num': len(trade_history) + 1,
                    'type': 'LONG',
                    'entry_day': entry_day,
                    'exit_day': i,
                    'entry_price': position_entry_price,
                    'exit_price': current_price,
                    'shares': position,
                    'pnl': pnl,
                    'pnl_percent': (pnl / (position * position_entry_price)) * 100
                })
                # Switch from long to short
                capital = starting_capital + sum(t['pnl'] for t in trade_history)
                shares = capital / current_price
                position = shares
                position_entry_price = current_price
                entry_day = i
                is_long = False
                capital = 0
        
        # Calculate daily return
        price_change = next_actual_price - current_price
        if is_long:
            daily_pnl = position * price_change
        else:
            daily_pnl = position * (-price_change)  # Short profits when price falls
        
        daily_returns.append(float(daily_pnl))
    
    # Close final position and record last trade
    final_price = float(actual_prices[-1])
    if is_long:
        pnl = position * (final_price - position_entry_price)
        trade_history.append({
            'trade_num': len(trade_history) + 1,
            'type': 'LONG',
            'entry_day': entry_day,
            'exit_day': min_len - 1,
            'entry_price': position_entry_price,
            'exit_price': final_price,
            'shares': position,
            'pnl': pnl,
            'pnl_percent': (pnl / (position * position_entry_price)) * 100
        })
        final_capital = position * final_price
    else:
        pnl = position * (position_entry_price - final_price)
        trade_history.append({
            'trade_num': len(trade_history) + 1,
            'type': 'SHORT',
            'entry_day': entry_day,
            'exit_day': min_len - 1,
            'entry_price': position_entry_price,
            'exit_price': final_price,
            'shares': position,
            'pnl': pnl,
            'pnl_percent': (pnl / (position * position_entry_price)) * 100
        })
        final_capital = starting_capital + sum(t['pnl'] for t in trade_history)
    
    total_pnl = sum(t['pnl'] for t in trade_history)
    return float(total_pnl), daily_returns, trade_history

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
        scaler = None
        
        # Try to load existing model if enabled
        if LOAD_EXISTING_MODELS:
            print(f"  Checking for saved model...")
            model, scaler = load_saved_model(symbol)
            if model is not None:
                print(f"  Loaded existing model for {symbol}")
        
        # Download training data (needed for scaler if not loaded, or for training)
        if model is None or scaler is None:
            print(f"  Downloading training data ({TRAINING_START} to {TRAINING_END})...")
            train_data = yf.download(symbol, start=TRAINING_START, end=TRAINING_END, progress=False)
            
            if len(train_data) < PREDICTION_DAYS + 100:
                print(f"  ERROR: Not enough training data for {symbol}")
                return None
            
            # Use technical indicators if enabled
            if USE_TECHNICAL_INDICATORS:
                print(f"  Adding technical indicators (MA, RSI, MACD, Bollinger Bands)...")
                feature_columns = ['Close', 'MA_10', 'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal', 
                                   'MACD_Hist', 'BB_Upper', 'BB_Lower', 'Momentum', 'Volatility']
                x_train, y_train, scaler, price_scaler, n_features = prepare_training_data_with_features(
                    train_data, PREDICTION_DAYS
                )
                
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
                    'use_features': True
                }
            else:
                train_close = train_data[['Close']].values
                print(f"  Preparing training data...")
                x_train, y_train, scaler = prepare_training_data(train_close, PREDICTION_DAYS)
                price_scaler = scaler
                feature_columns = None
                
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
                    'use_features': False
                }
            
            # Save the model if enabled
            if SAVE_MODELS:
                if not os.path.exists(MODEL_DIR):
                    os.makedirs(MODEL_DIR)
                model_path = os.path.join(MODEL_DIR, f"{symbol}_lstm_model.keras")
                scaler_path = os.path.join(MODEL_DIR, f"{symbol}_model_info.pkl")
                model.save(model_path)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(model_info, f)
                print(f"  Model saved to: {model_path}")
        else:
            # Still need train_data for test data preparation
            print(f"  Downloading training data for context...")
            train_data = yf.download(symbol, start=TRAINING_START, end=TRAINING_END, progress=False)
            if len(train_data) < PREDICTION_DAYS + 100:
                print(f"  ERROR: Not enough training data for {symbol}")
                return None
            price_scaler = scaler
            feature_columns = None
        
        # Download test data (past 2 months)
        print(f"  Downloading test data ({TEST_START} to {TEST_END})...")
        test_data = yf.download(symbol, start=TEST_START, end=TEST_END, progress=False)
        
        if len(test_data) < 10:
            print(f"  ERROR: Not enough test data for {symbol}")
            return None
        
        actual_prices = test_data['Close'].values
        
        # Prepare test data based on whether we're using features
        if USE_TECHNICAL_INDICATORS and feature_columns is not None:
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
        
        # Simulate investment
        print(f"  Simulating investment strategy...")
        total_pnl, daily_returns, trade_history = simulate_investment(actual_prices, predicted_prices, STARTING_CAPITAL)
        pnl_percent = (total_pnl / STARTING_CAPITAL) * 100
        
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
            'trade_history': trade_history
        }
        
        print(f"  ✓ Analysis complete for {symbol}")
        return results
        
    except Exception as e:
        import traceback
        print(f"  ERROR analyzing {symbol}: {str(e)}")
        traceback.print_exc()
        return None

def generate_report(results):
    """Generate markdown report"""
    report = []
    report.append("# Monte Carlo Stock Analysis Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**Analysis Period:** {TEST_START} to {TEST_END} (Past 2 Months)")
    report.append(f"\n**Training Period:** {TRAINING_START} to {TRAINING_END}")
    report.append(f"\n**Monte Carlo Simulations:** {MONTE_CARLO_SIMULATIONS}")
    report.append(f"\n**Starting Capital per Stock:** ${STARTING_CAPITAL:,.2f}")
    report.append(f"\n**Training Epochs:** {EPOCHS} (with early stopping, patience={EARLY_STOPPING_PATIENCE})")
    report.append(f"\n**Technical Indicators:** {'Enabled (MA, RSI, MACD, Bollinger Bands, Momentum, Volatility)' if USE_TECHNICAL_INDICATORS else 'Disabled'}")
    
    # Summary Table
    report.append("\n\n## Summary Results\n")
    report.append("| Stock | Dir. Accuracy | MSE | RMSE | MAPE | Investment Result | Status |")
    report.append("|-------|--------------|-----|------|------|-------------------|--------|")
    
    total_pnl = 0
    valid_results = [r for r in results if r is not None]
    
    if len(valid_results) == 0:
        report.append("\nNo valid results to report.\n")
        return "\n".join(report)
    
    for r in valid_results:
        status_emoji = "✅" if r['status'] == 'GAIN' else "❌"
        report.append(
            f"| {r['symbol']} | {r['directional_accuracy']:.1f}% | {r['mse']:.2f} | "
            f"${r['rmse']:.2f} | {r['mape']:.2f}% | ${r['total_pnl']:+,.2f} ({r['pnl_percent']:+.1f}%) | "
            f"{status_emoji} {r['status']} |"
        )
        total_pnl += r['total_pnl']
    
    # Portfolio Summary
    report.append("\n\n## Portfolio Summary\n")
    total_invested = STARTING_CAPITAL * len(valid_results)
    portfolio_return = (total_pnl / total_invested) * 100 if total_invested > 0 else 0
    
    report.append(f"- **Total Invested:** ${total_invested:,.2f}")
    report.append(f"- **Total P&L:** ${total_pnl:+,.2f}")
    report.append(f"- **Portfolio Return:** {portfolio_return:+.2f}%")
    report.append(f"- **Final Portfolio Value:** ${total_invested + total_pnl:,.2f}")
    
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
        report.append(f"| Total P&L | ${r['total_pnl']:+,.2f} |")
        report.append(f"| Return | {r['pnl_percent']:+.2f}% |")
        report.append(f"| Trading Days | {r['test_days']} |")
        if r['mc_14day_prediction']:
            report.append(f"| MC 14-Day Prediction | ${r['mc_14day_prediction']:.2f} |")
        
        # Trade History Breakdown
        if r.get('trade_history') and len(r['trade_history']) > 0:
            report.append(f"\n#### Trade History for {r['symbol']}\n")
            report.append("| # | Type | Entry Day | Exit Day | Entry Price | Exit Price | Shares | P&L | Return |")
            report.append("|---|------|-----------|----------|-------------|------------|--------|-----|--------|")
            
            for t in r['trade_history']:
                status_icon = "+" if t['pnl'] >= 0 else ""
                report.append(
                    f"| {t['trade_num']} | {t['type']} | Day {t['entry_day']} | Day {t['exit_day']} | "
                    f"${t['entry_price']:.2f} | ${t['exit_price']:.2f} | {t['shares']:.2f} | "
                    f"${status_icon}{t['pnl']:.2f} | {status_icon}{t['pnl_percent']:.2f}% |"
                )
            
            # Trade summary
            winning_trades = [t for t in r['trade_history'] if t['pnl'] > 0]
            losing_trades = [t for t in r['trade_history'] if t['pnl'] <= 0]
            total_trades = len(r['trade_history'])
            
            report.append(f"\n**Trade Summary:**")
            report.append(f"- Total Trades: {total_trades}")
            report.append(f"- Winning Trades: {len(winning_trades)} ({len(winning_trades)/total_trades*100:.1f}%)")
            report.append(f"- Losing Trades: {len(losing_trades)} ({len(losing_trades)/total_trades*100:.1f}%)")
            if winning_trades:
                report.append(f"- Total Gains: ${sum(t['pnl'] for t in winning_trades):,.2f}")
            if losing_trades:
                report.append(f"- Total Losses: ${sum(t['pnl'] for t in losing_trades):,.2f}")
    
    # Metrics Explanation
    report.append("\n\n## Metrics Explanation\n")
    report.append("- **Directional Accuracy:** Percentage of correct up/down movement predictions")
    report.append("- **MSE (Mean Squared Error):** Average of squared differences between predicted and actual prices")
    report.append("- **RMSE (Root Mean Squared Error):** Square root of MSE, in dollar terms")
    report.append("- **MAPE (Mean Absolute Percentage Error):** Average percentage error between predictions and actuals")
    report.append("- **Investment Strategy:** Long when model predicts price increase, Short when model predicts decrease")
    
    return "\n".join(report)

# ==========================================
# RUN ANALYSIS
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("MONTE CARLO STOCK ANALYSIS")
    print("=" * 60)
    print(f"Stocks: {', '.join(STOCKS)}")
    print(f"Analysis Period: {TEST_START} to {TEST_END}")
    print(f"Starting Capital: ${STARTING_CAPITAL:,} per stock")
    print("=" * 60)
    
    results = []
    for symbol in STOCKS:
        result = analyze_stock(symbol)
        results.append(result)
    
    # Generate report
    print("\n" + "=" * 60)
    print("GENERATING REPORT...")
    print("=" * 60)
    
    report = generate_report(results)
    
    # Save report
    report_path = "monte_carlo_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✓ Report saved to: {report_path}")
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    
    valid_results = [r for r in results if r is not None]
    
    if len(valid_results) == 0:
        print("No valid results to display.")
    else:
        for r in valid_results:
            status = "✅ GAIN" if r['status'] == 'GAIN' else "❌ LOSS"
            print(f"{r['symbol']:6} | Dir.Acc: {r['directional_accuracy']:5.1f}% | "
                  f"P&L: ${r['total_pnl']:+10,.2f} ({r['pnl_percent']:+6.1f}%) | {status}")
        
        total_pnl = sum(r['total_pnl'] for r in valid_results)
        total_invested = STARTING_CAPITAL * len(valid_results)
        
        print("-" * 60)
        print(f"TOTAL  | Invested: ${total_invested:,.2f} | "
              f"P&L: ${total_pnl:+,.2f} ({(total_pnl/total_invested)*100:+.1f}%)")
    print("=" * 60)
