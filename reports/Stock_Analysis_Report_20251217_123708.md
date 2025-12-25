# Monte Carlo Stock Analysis Report (Hybrid Strategy (Sniper Shorts))

**Generated:** 2025-12-17 12:45:09

**Analysis Period:** 2025-09-18 to 2025-12-17 (Past 90 Days)

**Training Period:** 2020-12-18 to 2025-12-17 (Rolling 5-Year Window)

**Monte Carlo Simulations:** 50

**Starting Capital per Stock:** $10,000.00

**Training Epochs:** 100 (with early stopping, patience=20)


## Strategy Configuration

| Parameter | Value |
|-----------|-------|
| Strategy Mode | **HYBRID** |
| Confidence Threshold | 0.5% |
| Stop-Loss Type | Dynamic ATR (2.0x ATR) |
| RVOL Threshold | 0.75 (relaxed to avoid dead volume only) |
| Macro Filter (200-SMA) | Enabled |
| Trend Override | Enabled (Buy when Price > 200 SMA AND > 50 SMA) |
| **Sniper Short Enabled** | **YES** |
| Short ADX Threshold | > 25 (strong trend) |
| Short RSI Threshold | < 50 (bearish momentum) |
| Short Conditions | Price < 200 SMA + ADX > 25 + RSI < 50 |
| RSI Period | 21 |
| MACD Parameters | (24, 52, 18) |
| MFI Period | 21 |
| Volume Indicators | OBV, MFI, VWAP, RVOL, ATR, ADX |


## Summary Results

| Stock | Strategy Return | Buy & Hold | Win Rate | Max DD | Trades | Status |
|-------|-----------------|------------|----------|--------|--------|--------|
| QQQ | +4.4% ($+435) | +2.9% 📈 | 100% | 2.4% | 3 | ✅ GAIN |
| SPY | +0.0% ($+0) | +2.8% 📉 | 0% | 0.0% | 0 | ❌ LOSS |
| AAPL | +8.0% ($+798) | +15.6% 📉 | 100% | 1.5% | 1 | ✅ GAIN |
| NVDA | +0.0% ($+0) | +0.8% 📉 | 0% | 0.0% | 0 | ❌ LOSS |
| AMD | +54.8% ($+5,476) | +32.5% 📈 | 67% | 11.1% | 3 | ✅ GAIN |
| INTC | +11.0% ($+1,102) | +22.0% 📉 | 100% | 5.9% | 4 | ✅ GAIN |
| GOOGL | +2.0% ($+204) | +21.7% 📉 | 100% | 7.1% | 1 | ✅ GAIN |
| TSLA | +21.2% ($+2,122) | +17.5% 📈 | 86% | 12.2% | 7 | ✅ GAIN |


## Portfolio Summary

| Metric | Strategy | Buy & Hold |
|--------|----------|------------|
| Total Invested | $80,000.00 | $80,000.00 |
| Total P&L | $+10,137.03 | $+11,580.93 |
| Portfolio Return | +12.67% | +14.48% |
| Final Value | $90,137.03 | $91,580.93 |

**Strategy vs Buy & Hold:** -1.80% (Underperformed ❌)


## Detailed Stock Analysis


### QQQ

| Metric | Value |
|--------|-------|
| Start Price | $594.63 |
| End Price | $611.75 |
| Directional Accuracy | 46.77% |
| Mean Squared Error | 151.8315 |
| Root Mean Squared Error | $12.32 |
| Mean Absolute Percentage Error | 1.56% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,435.22 |
| **Strategy Return** | **+4.35%** |
| Buy & Hold Return | +2.88% |
| Max Drawdown | 2.37% |
| Win Rate | 100.0% |
| Total Trades | 3 |
| Trading Days | 63 |
| MC 14-Day Prediction | $587.67 |

#### Trade History for QQQ

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 7 | $594.63 | $598.73 | 16.82 | $+68.94 | +0.69% | 📊 SIGNAL_SELL |
| 2 | Day 16 | Day 17 | $589.50 | $602.01 | 17.08 | $+213.68 | +2.12% | 📊 SIGNAL_SELL |
| 3 | Day 43 | Day 47 | $596.31 | $605.16 | 17.24 | $+152.61 | +1.48% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 3
- Winning Trades: 3 (100.0%)
- Losing Trades: 0 (0.0%)
- Total Gains: $435.22

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 3
- 📉 Macro Filter Exits: 0

### SPY

| Metric | Value |
|--------|-------|
| Start Price | $660.43 |
| End Price | $678.87 |
| Directional Accuracy | 35.48% |
| Mean Squared Error | 2879.8232 |
| Root Mean Squared Error | $53.66 |
| Mean Absolute Percentage Error | 7.39% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,000.00 |
| **Strategy Return** | **+0.00%** |
| Buy & Hold Return | +2.79% |
| Max Drawdown | 0.00% |
| Win Rate | 0.0% |
| Total Trades | 0 |
| Trading Days | 63 |
| MC 14-Day Prediction | $600.92 |

### AAPL

| Metric | Value |
|--------|-------|
| Start Price | $237.65 |
| End Price | $274.61 |
| Directional Accuracy | 54.84% |
| Mean Squared Error | 540.3470 |
| Root Mean Squared Error | $23.25 |
| Mean Absolute Percentage Error | 7.57% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,798.30 |
| **Strategy Return** | **+7.98%** |
| Buy & Hold Return | +15.55% |
| Max Drawdown | 1.47% |
| Win Rate | 100.0% |
| Total Trades | 1 |
| Trading Days | 63 |
| MC 14-Day Prediction | $257.78 |

#### Trade History for AAPL

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 5 | $237.65 | $256.62 | 42.08 | $+798.30 | +7.98% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 1
- Winning Trades: 1 (100.0%)
- Losing Trades: 0 (0.0%)
- Total Gains: $798.30

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 1
- 📉 Macro Filter Exits: 0

### NVDA

| Metric | Value |
|--------|-------|
| Start Price | $176.23 |
| End Price | $177.72 |
| Directional Accuracy | 48.39% |
| Mean Squared Error | 143.8920 |
| Root Mean Squared Error | $12.00 |
| Mean Absolute Percentage Error | 5.83% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,000.00 |
| **Strategy Return** | **+0.00%** |
| Buy & Hold Return | +0.85% |
| Max Drawdown | 0.00% |
| Win Rate | 0.0% |
| Total Trades | 0 |
| Trading Days | 63 |
| MC 14-Day Prediction | $166.63 |

### AMD

| Metric | Value |
|--------|-------|
| Start Price | $157.92 |
| End Price | $209.17 |
| Directional Accuracy | 50.00% |
| Mean Squared Error | 223.3146 |
| Root Mean Squared Error | $14.94 |
| Mean Absolute Percentage Error | 5.97% |
| Starting Capital | $10,000.00 |
| Final Capital | $15,475.68 |
| **Strategy Return** | **+54.76%** |
| Buy & Hold Return | +32.45% |
| Max Drawdown | 11.05% |
| Win Rate | 66.7% |
| Total Trades | 3 |
| Trading Days | 63 |
| MC 14-Day Prediction | $207.14 |

#### Trade History for AMD

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 1 | $157.92 | $157.39 | 63.32 | $-33.56 | -0.34% |  BEAR_REGIME |
| 2 | Day 2 | Day 35 | $159.79 | $237.70 | 62.37 | $+4859.41 | +48.76% | 📊 SIGNAL_SELL |
| 3 | Day 45 | Day 47 | $206.02 | $215.05 | 71.96 | $+649.83 | +4.38% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 3
- Winning Trades: 2 (66.7%)
- Losing Trades: 1 (33.3%)
- Total Gains: $5,509.24
- Total Losses: $-33.56

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 2
- 📉 Macro Filter Exits: 0

### INTC

| Metric | Value |
|--------|-------|
| Start Price | $30.57 |
| End Price | $37.31 |
| Directional Accuracy | 48.39% |
| Mean Squared Error | 3.2864 |
| Root Mean Squared Error | $1.81 |
| Mean Absolute Percentage Error | 3.80% |
| Starting Capital | $10,000.00 |
| Final Capital | $11,101.66 |
| **Strategy Return** | **+11.02%** |
| Buy & Hold Return | +22.05% |
| Max Drawdown | 5.92% |
| Win Rate | 100.0% |
| Total Trades | 4 |
| Trading Days | 63 |
| MC 14-Day Prediction | $34.19 |

#### Trade History for INTC

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 4 | $30.57 | $31.22 | 327.12 | $+212.63 | +2.13% | 📊 SIGNAL_SELL |
| 2 | Day 7 | Day 9 | $34.48 | $35.94 | 296.19 | $+432.44 | +4.23% | 📊 SIGNAL_SELL |
| 3 | Day 11 | Day 14 | $36.83 | $37.43 | 289.03 | $+173.42 | +1.63% | 📊 SIGNAL_SELL |
| 4 | Day 45 | Day 46 | $33.62 | $34.50 | 321.79 | $+283.17 | +2.62% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 4
- Winning Trades: 4 (100.0%)
- Losing Trades: 0 (0.0%)
- Total Gains: $1,101.66

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 4
- 📉 Macro Filter Exits: 0

### GOOGL

| Metric | Value |
|--------|-------|
| Start Price | $251.87 |
| End Price | $306.57 |
| Directional Accuracy | 48.39% |
| Mean Squared Error | 375.3831 |
| Root Mean Squared Error | $19.37 |
| Mean Absolute Percentage Error | 6.01% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,204.15 |
| **Strategy Return** | **+2.04%** |
| Buy & Hold Return | +21.72% |
| Max Drawdown | 7.13% |
| Win Rate | 100.0% |
| Total Trades | 1 |
| Trading Days | 63 |
| MC 14-Day Prediction | $307.08 |

#### Trade History for GOOGL

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 1 | Day 26 | $254.55 | $259.75 | 39.28 | $+204.15 | +2.04% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 1
- Winning Trades: 1 (100.0%)
- Losing Trades: 0 (0.0%)
- Total Gains: $204.15

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 1
- 📉 Macro Filter Exits: 0

### TSLA

| Metric | Value |
|--------|-------|
| Start Price | $416.85 |
| End Price | $489.88 |
| Directional Accuracy | 37.70% |
| Mean Squared Error | 309.6263 |
| Root Mean Squared Error | $17.60 |
| Mean Absolute Percentage Error | 3.38% |
| Starting Capital | $10,000.00 |
| Final Capital | $12,122.03 |
| **Strategy Return** | **+21.22%** |
| Buy & Hold Return | +17.52% |
| Max Drawdown | 12.23% |
| Win Rate | 85.7% |
| Total Trades | 7 |
| Trading Days | 62 |
| MC 14-Day Prediction | $413.41 |

#### Trade History for TSLA

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 4 | $416.85 | $442.79 | 23.99 | $+622.29 | +6.22% | 📊 SIGNAL_SELL |
| 2 | Day 16 | Day 17 | $413.49 | $435.90 | 25.69 | $+575.70 | +5.42% | 📊 SIGNAL_SELL |
| 3 | Day 26 | Day 27 | $433.72 | $452.42 | 25.82 | $+482.81 | +4.31% | 📊 SIGNAL_SELL |
| 4 | Day 30 | Day 32 | $440.10 | $468.37 | 26.54 | $+750.32 | +6.42% | 📊 SIGNAL_SELL |
| 5 | Day 33 | Day 34 | $444.26 | $462.07 | 27.98 | $+498.35 | +4.01% | 📊 SIGNAL_SELL |
| 6 | Day 35 | Day 40 | $445.91 | $404.62 | 29.00 | $-1197.11 | -9.26% |  ATR-STOP (9.3%) |
| 7 | Day 41 | Day 47 | $404.35 | $417.78 | 29.02 | $+389.68 | +3.32% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 7
- Winning Trades: 6 (85.7%)
- Losing Trades: 1 (14.3%)
- Total Gains: $3,319.14
- Total Losses: $-1,197.11

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 6
- 📉 Macro Filter Exits: 0


## Strategy Explanation

### Decision Logic
1. **Macro Filter:** Only go LONG when Close > 200-day SMA (bullish regime)
2. **Buy Signal (ML):** Model predicts price increase > threshold AND RVOL > 0.75
3. **Buy Signal (Trend Override):** Price > 200 SMA AND > 50 SMA AND RVOL > 0.75 (trust the trend)
4. **Sell Signal:** Model predicts significant price decrease (< -threshold)
5. **Stop-Loss:** Dynamic ATR-based exit (Entry - 2×ATR for longs, Entry + 2×ATR for shorts)

### Sniper Short Logic (HYBRID Mode)
Shorts are ONLY allowed when ALL conditions are met:
1. **Price < 200 SMA:** Stock is in a technical downtrend
2. **ADX > 25:** Trend is strong (not ranging)
3. **RSI < 50:** Momentum is bearish
4. **ML predicts down:** Model confirms downward movement
5. **RVOL > 0.75:** Volume confirms the move

**Short Exit Conditions:**
- RSI rises above 50 (momentum shift)
- ML predicts upward movement
- ATR stop-loss hit (Entry + 2×ATR)

### Dynamic Stop-Loss (ATR)
- **Volatile stocks (NVDA, TSLA):** Stop widens to 4-5% automatically
- **Stable stocks (SPY):** Stop tightens to 1-2% automatically
- **Long Formula:** Stop Price = Entry Price - (2 × ATR)
- **Short Formula:** Stop Price = Entry Price + (2 × ATR)

### Volume-Weighted Indicators
- **OBV (On-Balance Volume):** Detects divergence between price and volume
- **MFI (Money Flow Index):** Volume-weighted RSI with 21-day period
- **VWAP:** Identifies institutional support/resistance levels
- **RVOL:** Current volume vs 20-day average (>0.75 = acceptable volume)
- **ATR:** Average True Range for dynamic stop-loss calculation
- **ADX:** Average Directional Index for trend strength (>25 = strong trend)

### Exit Reasons
- 🛑 **ATR-STOP:** Dynamic stop-loss hit (adapts to stock volatility)
- 📊 **SIGNAL_SELL:** Model prediction turned bearish (long exit)
- 📊 **SIGNAL_COVER:** Model prediction turned bullish (short exit)
- 📉 **MACRO_FILTER:** Price crossed 200-day SMA (regime change)
- ⏰ **END_OF_PERIOD:** Position held until analysis period ended