import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==========================================
# 1. DATA COLLECTION
# ==========================================
# Download data for Apple (AAPL) from Jan 2020 to today
stock_symbol = 'AAPL'
start_date = '2020-01-01'
end_date = '2024-01-01'

data = yf.download(stock_symbol, start=start_date, end=end_date)
print(f"Downloaded {len(data)} days of data")

# We only need the 'Close' price for this simple predictor
dataset = data[['Close']].values

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
# Neural networks work better with small numbers (0 to 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Define how many past days the model should look at to predict the next day
prediction_days = 60

x_train = []
y_train = []

# Create sequences: e.g., Day 0-59 predict Day 60
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape data for LSTM [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# ==========================================
# 3. BUILD THE LSTM MODEL
# ==========================================
model = Sequential()

# Layer 1: LSTM with 50 units
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2)) # Drop 20% of units to prevent overfitting

# Layer 2: LSTM with 50 units
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

# Layer 3: LSTM with 50 units
model.add(LSTM(units=50))
model.add(Dropout(0.2))

# Output Layer: Prediction of the next closing price
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# ==========================================
# 4. TRAINING THE MODEL
# ==========================================
print("Training model... this may take a few minutes.")
model.fit(x_train, y_train, epochs=25, batch_size=32)

# ==========================================
# 5. TESTING THE PREDICTION
# ==========================================
# Get test data (prices that were not used in training)
test_start = '2024-01-01'
test_end = '2025-01-01' # Current date
test_data = yf.download(stock_symbol, start=test_start, end=test_end)
actual_prices = test_data['Close'].values

# We need the last 60 days of the training data to predict the first day of test data
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make Predictions
predicted_prices = model.predict(x_test)
# Reverse the scaling (back to normal prices)
predicted_prices = scaler.inverse_transform(predicted_prices)

# ==========================================
# 6. VISUALIZATION
# ==========================================
plt.figure(figsize=(10,6))
plt.plot(actual_prices, color="black", label=f"Actual {stock_symbol} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {stock_symbol} Price")
plt.title(f"{stock_symbol} Share Price Prediction")
plt.xlabel("Time")
plt.ylabel("Share Price")
plt.legend()
plt.show()