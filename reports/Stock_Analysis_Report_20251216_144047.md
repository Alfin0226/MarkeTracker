# Monte Carlo Stock Analysis Report (Hybrid Strategy (Sniper Shorts))

**Generated:** 2025-12-16 14:52:33

**Analysis Period:** 2025-09-17 to 2025-12-16 (Past 90 Days)

**Training Period:** 2023-12-17 to 2025-12-16 (Rolling 2-Year Window)

**Monte Carlo Simulations:** 10

**Starting Capital per Stock:** $10,000.00

**Training Epochs:** 100 (with early stopping, patience=10)


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
| QQQ | +7.1% ($+712) | +3.6% 📈 | 50% | 3.0% | 18 | ✅ GAIN |
| SPY | +2.7% ($+270) | +3.6% 📉 | 64% | 4.3% | 11 | ✅ GAIN |
| AAPL | +6.7% ($+671) | +14.8% 📉 | 62% | 5.0% | 16 | ✅ GAIN |
| NVDA | -0.3% ($-34) | +3.5% 📉 | 50% | 9.8% | 12 | ❌ LOSS |
| AMD | +35.7% ($+3,568) | +30.4% 📈 | 33% | 20.9% | 6 | ✅ GAIN |
| INTC | +41.6% ($+4,165) | +50.6% 📉 | 58% | 13.4% | 12 | ✅ GAIN |
| GOOGL | +24.5% ($+2,448) | +23.6% 📈 | 83% | 7.1% | 12 | ✅ GAIN |
| TSLA | +7.3% ($+727) | +11.6% 📉 | 60% | 17.2% | 10 | ✅ GAIN |


## Portfolio Summary

| Metric | Strategy | Buy & Hold |
|--------|----------|------------|
| Total Invested | $80,000.00 | $80,000.00 |
| Total P&L | $+12,527.15 | $+14,176.99 |
| Portfolio Return | +15.66% | +17.72% |
| Final Value | $92,527.15 | $94,176.99 |

**Strategy vs Buy & Hold:** -2.06% (Underperformed ❌)


## Detailed Stock Analysis


### QQQ

| Metric | Value |
|--------|-------|
| Start Price | $589.32 |
| End Price | $610.54 |
| Directional Accuracy | 61.29% |
| Mean Squared Error | 165.3993 |
| Root Mean Squared Error | $12.86 |
| Mean Absolute Percentage Error | 1.70% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,712.32 |
| **Strategy Return** | **+7.12%** |
| Buy & Hold Return | +3.60% |
| Max Drawdown | 3.01% |
| Win Rate | 50.0% |
| Total Trades | 18 |
| Trading Days | 63 |
| MC 14-Day Prediction | $601.50 |

#### Trade History for QQQ

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 11 | $589.32 | $605.73 | 16.97 | $+278.51 | +2.79% | 📊 SIGNAL_SELL |
| 2 | Day 15 | Day 16 | $611.44 | $610.70 | 16.81 | $-12.44 | -0.12% | 📊 SIGNAL_SELL |
| 3 | Day 17 | Day 20 | $589.50 | $602.22 | 17.41 | $+221.52 | +2.16% | 📊 SIGNAL_SELL |
| 4 | Day 23 | Day 24 | $611.54 | $611.38 | 17.15 | $-2.74 | -0.03% | 📊 SIGNAL_SELL |
| 5 | Day 26 | Day 27 | $610.58 | $617.10 | 17.17 | $+111.96 | +1.07% | 📊 SIGNAL_SELL |
| 6 | Day 28 | Day 29 | $628.09 | $632.92 | 16.87 | $+81.49 | +0.77% | 📊 SIGNAL_SELL |
| 7 | Day 30 | Day 31 | $635.77 | $626.05 | 16.80 | $-163.26 | -1.53% | 📊 SIGNAL_SELL |
| 8 | Day 32 | Day 33 | $629.07 | $632.08 | 16.72 | $+50.31 | +0.48% | 📊 SIGNAL_SELL |
| 9 | Day 34 | Day 35 | $619.25 | $623.28 | 17.06 | $+68.76 | +0.65% | 📊 SIGNAL_SELL |
| 10 | Day 36 | Day 37 | $611.67 | $609.74 | 17.39 | $-33.55 | -0.32% | 📊 SIGNAL_SELL |
| 11 | Day 38 | Day 39 | $623.23 | $621.57 | 17.01 | $-28.23 | -0.27% | 📊 SIGNAL_SELL |
| 12 | Day 40 | Day 41 | $621.08 | $608.40 | 17.02 | $-215.84 | -2.04% | 📊 SIGNAL_SELL |
| 13 | Day 46 | Day 48 | $585.67 | $605.16 | 17.68 | $+344.64 | +3.33% | 📊 SIGNAL_SELL |
| 14 | Day 49 | Day 50 | $608.89 | $614.27 | 17.57 | $+94.55 | +0.88% | 📊 SIGNAL_SELL |
| 15 | Day 53 | Day 54 | $622.00 | $623.52 | 17.36 | $+26.38 | +0.24% | 📊 SIGNAL_SELL |
| 16 | Day 56 | Day 57 | $625.48 | $624.28 | 17.30 | $-20.76 | -0.19% | 📊 SIGNAL_SELL |
| 17 | Day 59 | Day 60 | $627.61 | $625.58 | 17.21 | $-34.94 | -0.32% | 📊 SIGNAL_SELL |
| 18 | Day 61 | Day 62 | $613.62 | $610.54 | 17.55 | $-54.04 | -0.50% | ⏰ END_OF_PERIOD |

**Trade Summary:**
- Total Trades: 18
- Winning Trades: 9 (50.0%)
- Losing Trades: 9 (50.0%)
- Total Gains: $1,278.13
- Total Losses: $-565.81

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 17
- 📉 Macro Filter Exits: 0

### SPY

| Metric | Value |
|--------|-------|
| Start Price | $657.36 |
| End Price | $680.73 |
| Directional Accuracy | 50.00% |
| Mean Squared Error | 95.2397 |
| Root Mean Squared Error | $9.76 |
| Mean Absolute Percentage Error | 1.25% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,270.17 |
| **Strategy Return** | **+2.70%** |
| Buy & Hold Return | +3.56% |
| Max Drawdown | 4.34% |
| Win Rate | 63.6% |
| Total Trades | 11 |
| Trading Days | 63 |
| MC 14-Day Prediction | $671.90 |

#### Trade History for SPY

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 27 | $657.36 | $677.25 | 15.21 | $+302.61 | +3.03% | 📊 SIGNAL_SELL |
| 2 | Day 28 | Day 29 | $685.24 | $687.06 | 15.04 | $+27.36 | +0.27% | 📊 SIGNAL_SELL |
| 3 | Day 30 | Day 31 | $687.39 | $679.83 | 15.03 | $-113.61 | -1.10% | 📊 SIGNAL_SELL |
| 4 | Day 32 | Day 33 | $682.06 | $683.34 | 14.98 | $+19.17 | +0.19% | 📊 SIGNAL_SELL |
| 5 | Day 34 | Day 35 | $675.24 | $677.58 | 15.16 | $+35.47 | +0.35% | 📊 SIGNAL_SELL |
| 6 | Day 38 | Day 39 | $681.44 | $683.00 | 15.07 | $+23.51 | +0.23% | 📊 SIGNAL_SELL |
| 7 | Day 40 | Day 42 | $683.38 | $669.18 | 15.06 | $-213.93 | -2.08% |  ATR-STOP (2.1%) |
| 8 | Day 43 | Day 50 | $665.67 | $679.68 | 15.14 | $+212.16 | +2.10% | 📊 SIGNAL_SELL |
| 9 | Day 56 | Day 57 | $685.69 | $683.63 | 15.01 | $-30.92 | -0.30% | 📊 SIGNAL_SELL |
| 10 | Day 59 | Day 60 | $687.57 | $689.17 | 14.92 | $+23.88 | +0.23% | 📊 SIGNAL_SELL |
| 11 | Day 61 | Day 62 | $681.76 | $680.73 | 15.09 | $-15.54 | -0.15% | ⏰ END_OF_PERIOD |

**Trade Summary:**
- Total Trades: 11
- Winning Trades: 7 (63.6%)
- Losing Trades: 4 (36.4%)
- Total Gains: $644.17
- Total Losses: $-374.00

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 9
- 📉 Macro Filter Exits: 0

### AAPL

| Metric | Value |
|--------|-------|
| Start Price | $238.76 |
| End Price | $274.11 |
| Directional Accuracy | 43.55% |
| Mean Squared Error | 1229.9343 |
| Root Mean Squared Error | $35.07 |
| Mean Absolute Percentage Error | 11.32% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,671.26 |
| **Strategy Return** | **+6.71%** |
| Buy & Hold Return | +14.81% |
| Max Drawdown | 4.96% |
| Win Rate | 62.5% |
| Total Trades | 16 |
| Trading Days | 63 |
| MC 14-Day Prediction | $262.05 |

#### Trade History for AAPL

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 20 | $238.76 | $249.10 | 41.88 | $+433.07 | +4.33% | 📊 SIGNAL_SELL |
| 2 | Day 28 | Day 29 | $268.55 | $268.74 | 38.85 | $+7.37 | +0.07% | 📊 SIGNAL_SELL |
| 3 | Day 30 | Day 31 | $269.44 | $271.14 | 38.75 | $+65.81 | +0.63% | 📊 SIGNAL_SELL |
| 4 | Day 32 | Day 33 | $270.11 | $268.79 | 38.90 | $-51.29 | -0.49% | 📊 SIGNAL_SELL |
| 5 | Day 34 | Day 35 | $269.78 | $269.88 | 38.75 | $+3.87 | +0.04% | 📊 SIGNAL_SELL |
| 6 | Day 36 | Day 37 | $269.51 | $268.21 | 38.81 | $-50.40 | -0.48% | 📊 SIGNAL_SELL |
| 7 | Day 38 | Day 39 | $269.43 | $275.25 | 38.63 | $+224.83 | +2.16% | 📊 SIGNAL_SELL |
| 8 | Day 40 | Day 41 | $273.47 | $272.95 | 38.88 | $-20.22 | -0.19% | 📊 SIGNAL_SELL |
| 9 | Day 42 | Day 43 | $272.41 | $267.46 | 38.96 | $-192.85 | -1.82% | 📊 SIGNAL_SELL |
| 10 | Day 44 | Day 45 | $267.44 | $268.56 | 38.96 | $+43.64 | +0.42% | 📊 SIGNAL_SELL |
| 11 | Day 46 | Day 47 | $266.25 | $271.49 | 39.30 | $+205.94 | +1.97% | 📊 SIGNAL_SELL |
| 12 | Day 48 | Day 49 | $275.92 | $276.97 | 38.67 | $+40.60 | +0.38% | 📊 SIGNAL_SELL |
| 13 | Day 52 | Day 53 | $283.10 | $286.19 | 37.83 | $+116.90 | +1.09% | 📊 SIGNAL_SELL |
| 14 | Day 54 | Day 55 | $284.15 | $280.70 | 38.10 | $-131.46 | -1.21% | 📊 SIGNAL_SELL |
| 15 | Day 56 | Day 57 | $278.78 | $277.89 | 38.37 | $-34.15 | -0.32% | 📊 SIGNAL_SELL |
| 16 | Day 60 | Day 61 | $278.03 | $278.28 | 38.35 | $+9.59 | +0.09% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 16
- Winning Trades: 10 (62.5%)
- Losing Trades: 6 (37.5%)
- Total Gains: $1,151.63
- Total Losses: $-480.37

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 16
- 📉 Macro Filter Exits: 0

### NVDA

| Metric | Value |
|--------|-------|
| Start Price | $170.28 |
| End Price | $176.29 |
| Directional Accuracy | 54.84% |
| Mean Squared Error | 115.8045 |
| Root Mean Squared Error | $10.76 |
| Mean Absolute Percentage Error | 4.74% |
| Starting Capital | $10,000.00 |
| Final Capital | $9,966.49 |
| **Strategy Return** | **-0.34%** |
| Buy & Hold Return | +3.53% |
| Max Drawdown | 9.77% |
| Win Rate | 50.0% |
| Total Trades | 12 |
| Trading Days | 63 |
| MC 14-Day Prediction | $174.24 |

#### Trade History for NVDA

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 1 | $170.28 | $176.23 | 58.73 | $+349.40 | +3.49% | 📊 SIGNAL_SELL |
| 2 | Day 9 | Day 10 | $186.57 | $187.23 | 55.47 | $+36.61 | +0.35% | 📊 SIGNAL_SELL |
| 3 | Day 16 | Day 18 | $192.56 | $188.31 | 53.94 | $-229.22 | -2.21% | 📊 SIGNAL_SELL |
| 4 | Day 19 | Day 27 | $180.02 | $186.25 | 56.42 | $+351.48 | +3.46% | 📊 SIGNAL_SELL |
| 5 | Day 28 | Day 29 | $191.48 | $201.02 | 54.88 | $+523.52 | +4.98% | 📊 SIGNAL_SELL |
| 6 | Day 30 | Day 31 | $207.03 | $202.88 | 53.29 | $-221.13 | -2.00% | 📊 SIGNAL_SELL |
| 7 | Day 32 | Day 33 | $202.48 | $206.87 | 53.39 | $+234.38 | +2.17% | 📊 SIGNAL_SELL |
| 8 | Day 34 | Day 35 | $198.68 | $195.20 | 55.59 | $-193.45 | -1.75% | 📊 SIGNAL_SELL |
| 9 | Day 36 | Day 37 | $188.07 | $188.14 | 57.70 | $+4.04 | +0.04% | 📊 SIGNAL_SELL |
| 10 | Day 38 | Day 39 | $199.04 | $193.15 | 54.54 | $-321.22 | -2.96% | 📊 SIGNAL_SELL |
| 11 | Day 40 | Day 41 | $193.79 | $186.85 | 54.36 | $-377.24 | -3.58% | 📊 SIGNAL_SELL |
| 12 | Day 42 | Day 43 | $190.16 | $186.59 | 53.41 | $-190.68 | -1.88% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 12
- Winning Trades: 6 (50.0%)
- Losing Trades: 6 (50.0%)
- Total Gains: $1,499.43
- Total Losses: $-1,532.93

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 12
- 📉 Macro Filter Exits: 0

### AMD

| Metric | Value |
|--------|-------|
| Start Price | $159.16 |
| End Price | $207.58 |
| Directional Accuracy | 53.23% |
| Mean Squared Error | 1042.2322 |
| Root Mean Squared Error | $32.28 |
| Mean Absolute Percentage Error | 12.85% |
| Starting Capital | $10,000.00 |
| Final Capital | $13,567.55 |
| **Strategy Return** | **+35.68%** |
| Buy & Hold Return | +30.42% |
| Max Drawdown | 20.86% |
| Win Rate | 33.3% |
| Total Trades | 6 |
| Trading Days | 63 |
| MC 14-Day Prediction | $234.25 |

#### Trade History for AMD

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 30 | $159.16 | $264.33 | 62.83 | $+6607.81 | +66.08% | 📊 SIGNAL_SELL |
| 2 | Day 34 | Day 37 | $250.05 | $229.57 | 66.42 | $-1360.43 | -8.19% |  ATR-STOP (8.2%) |
| 3 | Day 38 | Day 40 | $243.98 | $258.89 | 62.49 | $+931.79 | +6.11% | 📊 SIGNAL_SELL |
| 4 | Day 41 | Day 42 | $247.96 | $246.81 | 65.25 | $-75.04 | -0.46% | 📊 SIGNAL_SELL |
| 5 | Day 44 | Day 47 | $230.29 | $201.00 | 69.93 | $-2048.34 | -12.72% |  ATR-STOP (12.7%) |
| 6 | Day 48 | Day 62 | $215.05 | $207.58 | 65.36 | $-488.24 | -3.47% | ⏰ END_OF_PERIOD |

**Trade Summary:**
- Total Trades: 6
- Winning Trades: 2 (33.3%)
- Losing Trades: 4 (66.7%)
- Total Gains: $7,539.61
- Total Losses: $-3,972.06

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 3
- 📉 Macro Filter Exits: 0

### INTC

| Metric | Value |
|--------|-------|
| Start Price | $24.90 |
| End Price | $37.51 |
| Directional Accuracy | 46.77% |
| Mean Squared Error | 12.0925 |
| Root Mean Squared Error | $3.48 |
| Mean Absolute Percentage Error | 7.25% |
| Starting Capital | $10,000.00 |
| Final Capital | $14,164.58 |
| **Strategy Return** | **+41.65%** |
| Buy & Hold Return | +50.64% |
| Max Drawdown | 13.43% |
| Win Rate | 58.3% |
| Total Trades | 12 |
| Trading Days | 63 |
| MC 14-Day Prediction | $37.00 |

#### Trade History for INTC

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 1 | Day 23 | $30.57 | $38.10 | 327.12 | $+2463.20 | +24.63% | 📊 SIGNAL_SELL |
| 2 | Day 26 | Day 27 | $38.16 | $38.28 | 326.60 | $+39.19 | +0.31% | 📊 SIGNAL_SELL |
| 3 | Day 28 | Day 29 | $39.54 | $41.53 | 316.20 | $+629.23 | +5.03% | 📊 SIGNAL_SELL |
| 4 | Day 30 | Day 31 | $41.34 | $40.16 | 317.65 | $-374.83 | -2.85% | 📊 SIGNAL_SELL |
| 5 | Day 34 | Day 35 | $37.03 | $38.38 | 344.50 | $+465.07 | +3.65% | 📊 SIGNAL_SELL |
| 6 | Day 37 | Day 38 | $38.13 | $38.45 | 346.76 | $+110.96 | +0.84% | 📊 SIGNAL_SELL |
| 7 | Day 41 | Day 51 | $35.91 | $40.56 | 371.28 | $+1726.47 | +12.95% | 📊 SIGNAL_SELL |
| 8 | Day 52 | Day 53 | $40.01 | $43.47 | 376.39 | $+1302.31 | +8.65% | 📊 SIGNAL_SELL |
| 9 | Day 54 | Day 55 | $43.76 | $40.50 | 373.89 | $-1218.89 | -7.45% | 📊 SIGNAL_SELL |
| 10 | Day 56 | Day 57 | $41.41 | $40.30 | 365.68 | $-405.90 | -2.68% | 📊 SIGNAL_SELL |
| 11 | Day 59 | Day 60 | $40.78 | $39.51 | 361.37 | $-458.94 | -3.11% | 📊 SIGNAL_SELL |
| 12 | Day 61 | Day 62 | $37.81 | $37.51 | 377.62 | $-113.29 | -0.79% | ⏰ END_OF_PERIOD |

**Trade Summary:**
- Total Trades: 12
- Winning Trades: 7 (58.3%)
- Losing Trades: 5 (41.7%)
- Total Gains: $6,736.44
- Total Losses: $-2,571.86

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 11
- 📉 Macro Filter Exits: 0

### GOOGL

| Metric | Value |
|--------|-------|
| Start Price | $249.37 |
| End Price | $308.22 |
| Directional Accuracy | 54.84% |
| Mean Squared Error | 783.6775 |
| Root Mean Squared Error | $27.99 |
| Mean Absolute Percentage Error | 8.08% |
| Starting Capital | $10,000.00 |
| Final Capital | $12,447.85 |
| **Strategy Return** | **+24.48%** |
| Buy & Hold Return | +23.60% |
| Max Drawdown | 7.13% |
| Win Rate | 83.3% |
| Total Trades | 12 |
| Trading Days | 63 |
| MC 14-Day Prediction | $305.86 |

#### Trade History for GOOGL

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 2 | Day 28 | $254.55 | $269.09 | 39.28 | $+571.21 | +5.71% | 📊 SIGNAL_SELL |
| 2 | Day 30 | Day 31 | $274.39 | $281.30 | 38.53 | $+266.04 | +2.52% | 📊 SIGNAL_SELL |
| 3 | Day 32 | Day 33 | $281.01 | $283.53 | 38.57 | $+97.51 | +0.90% | 📊 SIGNAL_SELL |
| 4 | Day 34 | Day 35 | $277.36 | $284.12 | 39.42 | $+266.73 | +2.44% | 📊 SIGNAL_SELL |
| 5 | Day 36 | Day 38 | $284.56 | $289.91 | 39.36 | $+210.46 | +1.88% | 📊 SIGNAL_SELL |
| 6 | Day 41 | Day 45 | $278.39 | $292.62 | 40.99 | $+583.36 | +5.11% | 📊 SIGNAL_SELL |
| 7 | Day 46 | Day 47 | $289.26 | $299.46 | 41.47 | $+423.12 | +3.53% | 📊 SIGNAL_SELL |
| 8 | Day 48 | Day 49 | $318.37 | $323.23 | 39.01 | $+189.45 | +1.53% | 📊 SIGNAL_SELL |
| 9 | Day 50 | Day 51 | $319.74 | $319.97 | 39.43 | $+9.06 | +0.07% | 📊 SIGNAL_SELL |
| 10 | Day 52 | Day 53 | $314.68 | $315.60 | 40.09 | $+36.86 | +0.29% | 📊 SIGNAL_SELL |
| 11 | Day 54 | Day 55 | $319.42 | $317.41 | 39.61 | $-79.57 | -0.63% | 📊 SIGNAL_SELL |
| 12 | Day 60 | Day 61 | $312.43 | $309.29 | 40.25 | $-126.37 | -1.01% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 12
- Winning Trades: 10 (83.3%)
- Losing Trades: 2 (16.7%)
- Total Gains: $2,653.80
- Total Losses: $-205.95

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 12
- 📉 Macro Filter Exits: 0

### TSLA

| Metric | Value |
|--------|-------|
| Start Price | $425.86 |
| End Price | $475.31 |
| Directional Accuracy | 39.34% |
| Mean Squared Error | 693.2933 |
| Root Mean Squared Error | $26.33 |
| Mean Absolute Percentage Error | 4.60% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,726.93 |
| **Strategy Return** | **+7.27%** |
| Buy & Hold Return | +11.61% |
| Max Drawdown | 17.16% |
| Win Rate | 60.0% |
| Total Trades | 10 |
| Trading Days | 62 |
| MC 14-Day Prediction | $469.11 |

#### Trade History for TSLA

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 9 | $425.86 | $444.72 | 23.48 | $+442.87 | +4.43% | 📊 SIGNAL_SELL |
| 2 | Day 10 | Day 12 | $459.46 | $417.21 | 22.73 | $-960.35 | -9.20% |  ATR-STOP (9.2%) |
| 3 | Day 13 | Day 22 | $453.25 | $439.31 | 20.92 | $-291.64 | -3.08% | 📊 SIGNAL_SELL |
| 4 | Day 25 | Day 26 | $438.97 | $448.98 | 20.94 | $+209.58 | +2.28% | 📊 SIGNAL_SELL |
| 5 | Day 27 | Day 28 | $433.72 | $452.42 | 21.67 | $+405.30 | +4.31% | 📊 SIGNAL_SELL |
| 6 | Day 29 | Day 30 | $460.55 | $461.51 | 21.29 | $+20.44 | +0.21% | 📊 SIGNAL_SELL |
| 7 | Day 31 | Day 32 | $440.10 | $456.56 | 22.33 | $+367.51 | +3.74% | 📊 SIGNAL_SELL |
| 8 | Day 33 | Day 35 | $468.37 | $462.07 | 21.76 | $-137.11 | -1.35% | 📊 SIGNAL_SELL |
| 9 | Day 36 | Day 41 | $445.91 | $404.62 | 22.55 | $-931.12 | -9.26% |  ATR-STOP (9.3%) |
| 10 | Day 42 | Day 61 | $404.35 | $475.31 | 22.57 | $+1601.44 | +17.55% | ⏰ END_OF_PERIOD |

**Trade Summary:**
- Total Trades: 10
- Winning Trades: 6 (60.0%)
- Losing Trades: 4 (40.0%)
- Total Gains: $3,047.15
- Total Losses: $-2,320.22

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 7
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