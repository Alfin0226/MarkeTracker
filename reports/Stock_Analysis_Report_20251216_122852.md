# Monte Carlo Stock Analysis Report (Hybrid Strategy (Sniper Shorts))

**Generated:** 2025-12-16 12:35:01

**Analysis Period:** 2025-09-17 to 2025-12-16 (Past 90 Days)

**Training Period:** 2023-09-18 to 2025-09-17 (Rolling 2-Year Window)

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
| QQQ | +2.7% ($+274) | +3.6% 📉 | 54% | 4.7% | 28 | ✅ GAIN |
| SPY | +3.2% ($+317) | +3.6% 📉 | 69% | 4.0% | 26 | ✅ GAIN |
| AAPL | +9.0% ($+899) | +14.8% 📉 | 54% | 1.7% | 26 | ✅ GAIN |
| NVDA | +11.9% ($+1,187) | +3.5% 📈 | 62% | 9.5% | 24 | ✅ GAIN |
| AMD | +1.5% ($+151) | +30.4% 📉 | 43% | 13.3% | 14 | ✅ GAIN |
| INTC | +46.7% ($+4,668) | +50.6% 📉 | 65% | 3.1% | 20 | ✅ GAIN |
| GOOGL | +3.6% ($+357) | +23.6% 📉 | 44% | 7.7% | 18 | ✅ GAIN |
| TSLA | +29.0% ($+2,897) | +11.6% 📈 | 76% | 8.5% | 21 | ✅ GAIN |


## Portfolio Summary

| Metric | Strategy | Buy & Hold |
|--------|----------|------------|
| Total Invested | $80,000.00 | $80,000.00 |
| Total P&L | $+10,749.88 | $+14,176.99 |
| Portfolio Return | +13.44% | +17.72% |
| Final Value | $90,749.88 | $94,176.99 |

**Strategy vs Buy & Hold:** -4.28% (Underperformed ❌)


## Detailed Stock Analysis


### QQQ

| Metric | Value |
|--------|-------|
| Start Price | $589.32 |
| End Price | $610.54 |
| Directional Accuracy | 46.77% |
| Mean Squared Error | 741.6563 |
| Root Mean Squared Error | $27.23 |
| Mean Absolute Percentage Error | 4.00% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,273.60 |
| **Strategy Return** | **+2.74%** |
| Buy & Hold Return | +3.60% |
| Max Drawdown | 4.68% |
| Win Rate | 53.6% |
| Total Trades | 28 |
| Trading Days | 63 |
| MC 14-Day Prediction | $576.81 |

#### Trade History for QQQ

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 1 | $589.32 | $594.63 | 16.97 | $+90.17 | +0.90% | 📊 SIGNAL_SELL |
| 2 | Day 2 | Day 3 | $598.66 | $602.20 | 16.85 | $+59.73 | +0.59% | 📊 SIGNAL_SELL |
| 3 | Day 4 | Day 5 | $598.20 | $596.10 | 16.97 | $-35.63 | -0.35% | 📊 SIGNAL_SELL |
| 4 | Day 6 | Day 7 | $593.53 | $595.97 | 17.04 | $+41.58 | +0.41% | 📊 SIGNAL_SELL |
| 5 | Day 8 | Day 9 | $598.73 | $600.37 | 16.96 | $+27.82 | +0.27% | 📊 SIGNAL_SELL |
| 6 | Day 10 | Day 11 | $603.25 | $605.73 | 16.88 | $+41.87 | +0.41% | 📊 SIGNAL_SELL |
| 7 | Day 12 | Day 13 | $603.18 | $607.71 | 16.95 | $+76.80 | +0.75% | 📊 SIGNAL_SELL |
| 8 | Day 14 | Day 15 | $604.51 | $611.44 | 17.04 | $+118.10 | +1.15% | 📊 SIGNAL_SELL |
| 9 | Day 16 | Day 17 | $610.70 | $598.89 | 17.06 | $-201.49 | -1.93% |  ATR-STOP (1.9%) |
| 10 | Day 18 | Day 19 | $602.01 | $598.00 | 16.97 | $-68.07 | -0.67% | 📊 SIGNAL_SELL |
| 11 | Day 20 | Day 21 | $602.22 | $599.99 | 16.86 | $-37.59 | -0.37% | 📊 SIGNAL_SELL |
| 12 | Day 22 | Day 23 | $603.93 | $611.54 | 16.75 | $+127.44 | +1.26% | 📊 SIGNAL_SELL |
| 13 | Day 24 | Day 25 | $611.38 | $605.49 | 16.75 | $-98.66 | -0.96% | 📊 SIGNAL_SELL |
| 14 | Day 26 | Day 27 | $610.58 | $617.10 | 16.61 | $+108.30 | +1.07% | 📊 SIGNAL_SELL |
| 15 | Day 28 | Day 29 | $628.09 | $632.92 | 16.32 | $+78.82 | +0.77% | 📊 SIGNAL_SELL |
| 16 | Day 30 | Day 31 | $635.77 | $626.05 | 16.25 | $-157.92 | -1.53% | 📊 SIGNAL_SELL |
| 17 | Day 32 | Day 33 | $629.07 | $632.08 | 16.17 | $+48.67 | +0.48% | 📊 SIGNAL_SELL |
| 18 | Day 34 | Day 35 | $619.25 | $623.28 | 16.50 | $+66.51 | +0.65% | 📊 SIGNAL_SELL |
| 19 | Day 36 | Day 37 | $611.67 | $609.74 | 16.82 | $-32.46 | -0.32% | 📊 SIGNAL_SELL |
| 20 | Day 38 | Day 39 | $623.23 | $621.57 | 16.45 | $-27.31 | -0.27% | 📊 SIGNAL_SELL |
| 21 | Day 40 | Day 41 | $621.08 | $608.40 | 16.47 | $-208.79 | -2.04% | 📊 SIGNAL_SELL |
| 22 | Day 42 | Day 43 | $608.86 | $603.66 | 16.45 | $-85.56 | -0.85% | 📊 SIGNAL_SELL |
| 23 | Day 46 | Day 48 | $585.67 | $605.16 | 16.96 | $+330.53 | +3.33% | 📊 SIGNAL_SELL |
| 24 | Day 49 | Day 50 | $608.89 | $614.27 | 16.86 | $+90.68 | +0.88% | 📊 SIGNAL_SELL |
| 25 | Day 53 | Day 54 | $622.00 | $623.52 | 16.65 | $+25.30 | +0.24% | 📊 SIGNAL_SELL |
| 26 | Day 56 | Day 57 | $625.48 | $624.28 | 16.59 | $-19.91 | -0.19% | 📊 SIGNAL_SELL |
| 27 | Day 59 | Day 60 | $627.61 | $625.58 | 16.51 | $-33.51 | -0.32% | 📊 SIGNAL_SELL |
| 28 | Day 61 | Day 62 | $613.62 | $610.54 | 16.83 | $-51.83 | -0.50% | ⏰ END_OF_PERIOD |

**Trade Summary:**
- Total Trades: 28
- Winning Trades: 15 (53.6%)
- Losing Trades: 13 (46.4%)
- Total Gains: $1,332.32
- Total Losses: $-1,058.72

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 26
- 📉 Macro Filter Exits: 0

### SPY

| Metric | Value |
|--------|-------|
| Start Price | $657.36 |
| End Price | $680.73 |
| Directional Accuracy | 54.84% |
| Mean Squared Error | 368.8488 |
| Root Mean Squared Error | $19.21 |
| Mean Absolute Percentage Error | 2.62% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,317.39 |
| **Strategy Return** | **+3.17%** |
| Buy & Hold Return | +3.56% |
| Max Drawdown | 3.95% |
| Win Rate | 69.2% |
| Total Trades | 26 |
| Trading Days | 63 |
| MC 14-Day Prediction | $645.25 |

#### Trade History for SPY

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 1 | $657.36 | $660.43 | 15.21 | $+46.72 | +0.47% | 📊 SIGNAL_SELL |
| 2 | Day 2 | Day 3 | $663.70 | $666.84 | 15.14 | $+47.53 | +0.47% | 📊 SIGNAL_SELL |
| 3 | Day 4 | Day 5 | $663.21 | $661.10 | 15.22 | $-32.12 | -0.32% | 📊 SIGNAL_SELL |
| 4 | Day 6 | Day 7 | $658.05 | $661.82 | 15.29 | $+57.65 | +0.57% | 📊 SIGNAL_SELL |
| 5 | Day 8 | Day 9 | $663.68 | $666.18 | 15.25 | $+38.12 | +0.38% | 📊 SIGNAL_SELL |
| 6 | Day 10 | Day 11 | $668.45 | $669.22 | 15.20 | $+11.70 | +0.12% | 📊 SIGNAL_SELL |
| 7 | Day 12 | Day 13 | $669.21 | $671.61 | 15.20 | $+36.47 | +0.36% | 📊 SIGNAL_SELL |
| 8 | Day 14 | Day 15 | $669.12 | $673.11 | 15.25 | $+60.86 | +0.60% | 📊 SIGNAL_SELL |
| 9 | Day 16 | Day 17 | $671.16 | $661.57 | 15.30 | $-146.77 | -1.43% |  ATR-STOP (1.4%) |
| 10 | Day 18 | Day 19 | $663.04 | $662.23 | 15.26 | $-12.36 | -0.12% | 📊 SIGNAL_SELL |
| 11 | Day 20 | Day 21 | $665.17 | $660.64 | 15.20 | $-68.84 | -0.68% | 📊 SIGNAL_SELL |
| 12 | Day 22 | Day 23 | $664.39 | $671.30 | 15.11 | $+104.41 | +1.04% | 📊 SIGNAL_SELL |
| 13 | Day 25 | Day 26 | $667.80 | $671.76 | 15.19 | $+60.15 | +0.59% | 📊 SIGNAL_SELL |
| 14 | Day 27 | Day 28 | $677.25 | $685.24 | 15.07 | $+120.38 | +1.18% | 📊 SIGNAL_SELL |
| 15 | Day 29 | Day 30 | $687.06 | $687.39 | 15.03 | $+4.96 | +0.05% | 📊 SIGNAL_SELL |
| 16 | Day 31 | Day 32 | $679.83 | $682.06 | 15.19 | $+33.88 | +0.33% | 📊 SIGNAL_SELL |
| 17 | Day 34 | Day 35 | $675.24 | $677.58 | 15.35 | $+35.91 | +0.35% | 📊 SIGNAL_SELL |
| 18 | Day 36 | Day 37 | $670.31 | $670.97 | 15.51 | $+10.24 | +0.10% | 📊 SIGNAL_SELL |
| 19 | Day 38 | Day 39 | $681.44 | $683.00 | 15.27 | $+23.83 | +0.23% | 📊 SIGNAL_SELL |
| 20 | Day 40 | Day 41 | $683.38 | $672.04 | 15.27 | $-173.12 | -1.66% | 📊 SIGNAL_SELL |
| 21 | Day 42 | Day 44 | $671.93 | $656.26 | 15.27 | $-239.20 | -2.33% |  ATR-STOP (2.3%) |
| 22 | Day 46 | Day 48 | $652.53 | $668.73 | 15.36 | $+248.77 | +2.48% | 📊 SIGNAL_SELL |
| 23 | Day 49 | Day 50 | $675.02 | $679.68 | 15.21 | $+70.89 | +0.69% | 📊 SIGNAL_SELL |
| 24 | Day 56 | Day 57 | $685.69 | $683.63 | 15.08 | $-31.06 | -0.30% | 📊 SIGNAL_SELL |
| 25 | Day 59 | Day 60 | $687.57 | $689.17 | 14.99 | $+23.99 | +0.23% | 📊 SIGNAL_SELL |
| 26 | Day 61 | Day 62 | $681.76 | $680.73 | 15.16 | $-15.61 | -0.15% | ⏰ END_OF_PERIOD |

**Trade Summary:**
- Total Trades: 26
- Winning Trades: 18 (69.2%)
- Losing Trades: 8 (30.8%)
- Total Gains: $1,036.46
- Total Losses: $-719.08

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 23
- 📉 Macro Filter Exits: 0

### AAPL

| Metric | Value |
|--------|-------|
| Start Price | $238.76 |
| End Price | $274.11 |
| Directional Accuracy | 38.71% |
| Mean Squared Error | 843.8695 |
| Root Mean Squared Error | $29.05 |
| Mean Absolute Percentage Error | 9.96% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,899.29 |
| **Strategy Return** | **+8.99%** |
| Buy & Hold Return | +14.81% |
| Max Drawdown | 1.70% |
| Win Rate | 53.8% |
| Total Trades | 26 |
| Trading Days | 63 |
| MC 14-Day Prediction | $244.38 |

#### Trade History for AAPL

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 1 | $238.76 | $237.65 | 41.88 | $-46.45 | -0.46% | 📊 SIGNAL_SELL |
| 2 | Day 2 | Day 3 | $245.26 | $255.83 | 40.58 | $+428.96 | +4.31% | 📊 SIGNAL_SELL |
| 3 | Day 4 | Day 5 | $254.18 | $252.07 | 40.85 | $-86.51 | -0.83% | 📊 SIGNAL_SELL |
| 4 | Day 6 | Day 7 | $256.62 | $255.21 | 40.12 | $-56.52 | -0.55% | 📊 SIGNAL_SELL |
| 5 | Day 10 | Day 11 | $255.20 | $256.88 | 40.12 | $+67.34 | +0.66% | 📊 SIGNAL_SELL |
| 6 | Day 12 | Day 13 | $257.77 | $256.44 | 39.98 | $-53.13 | -0.52% | 📊 SIGNAL_SELL |
| 7 | Day 17 | Day 18 | $245.03 | $247.42 | 41.85 | $+99.92 | +0.97% | 📊 SIGNAL_SELL |
| 8 | Day 21 | Day 22 | $247.21 | $252.05 | 41.88 | $+202.51 | +1.96% | 📊 SIGNAL_SELL |
| 9 | Day 23 | Day 24 | $261.99 | $262.52 | 40.29 | $+21.33 | +0.20% | 📊 SIGNAL_SELL |
| 10 | Day 25 | Day 26 | $258.20 | $259.33 | 40.97 | $+46.25 | +0.44% | 📊 SIGNAL_SELL |
| 11 | Day 27 | Day 28 | $262.57 | $268.55 | 40.46 | $+242.13 | +2.28% | 📊 SIGNAL_SELL |
| 12 | Day 29 | Day 30 | $268.74 | $269.44 | 40.43 | $+28.28 | +0.26% | 📊 SIGNAL_SELL |
| 13 | Day 31 | Day 32 | $271.14 | $270.11 | 40.18 | $-41.34 | -0.38% | 📊 SIGNAL_SELL |
| 14 | Day 33 | Day 34 | $268.79 | $269.78 | 40.38 | $+39.93 | +0.37% | 📊 SIGNAL_SELL |
| 15 | Day 35 | Day 36 | $269.88 | $269.51 | 40.36 | $-14.92 | -0.14% | 📊 SIGNAL_SELL |
| 16 | Day 37 | Day 38 | $268.21 | $269.43 | 40.56 | $+49.48 | +0.45% | 📊 SIGNAL_SELL |
| 17 | Day 39 | Day 40 | $275.25 | $273.47 | 39.70 | $-70.66 | -0.65% | 📊 SIGNAL_SELL |
| 18 | Day 41 | Day 42 | $272.95 | $272.41 | 39.78 | $-21.48 | -0.20% | 📊 SIGNAL_SELL |
| 19 | Day 43 | Day 44 | $267.46 | $267.44 | 40.51 | $-0.81 | -0.01% | 📊 SIGNAL_SELL |
| 20 | Day 45 | Day 46 | $268.56 | $266.25 | 40.34 | $-93.19 | -0.86% | 📊 SIGNAL_SELL |
| 21 | Day 47 | Day 48 | $271.49 | $275.92 | 39.56 | $+175.27 | +1.63% | 📊 SIGNAL_SELL |
| 22 | Day 49 | Day 50 | $276.97 | $277.55 | 39.41 | $+22.86 | +0.21% | 📊 SIGNAL_SELL |
| 23 | Day 52 | Day 53 | $283.10 | $286.19 | 38.64 | $+119.40 | +1.09% | 📊 SIGNAL_SELL |
| 24 | Day 54 | Day 55 | $284.15 | $280.70 | 38.92 | $-134.27 | -1.21% | 📊 SIGNAL_SELL |
| 25 | Day 56 | Day 57 | $278.78 | $277.89 | 39.19 | $-34.88 | -0.32% | 📊 SIGNAL_SELL |
| 26 | Day 60 | Day 61 | $278.03 | $278.28 | 39.17 | $+9.79 | +0.09% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 26
- Winning Trades: 14 (53.8%)
- Losing Trades: 12 (46.2%)
- Total Gains: $1,553.44
- Total Losses: $-654.15

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 26
- 📉 Macro Filter Exits: 0

### NVDA

| Metric | Value |
|--------|-------|
| Start Price | $170.28 |
| End Price | $176.29 |
| Directional Accuracy | 43.55% |
| Mean Squared Error | 128.3343 |
| Root Mean Squared Error | $11.33 |
| Mean Absolute Percentage Error | 4.60% |
| Starting Capital | $10,000.00 |
| Final Capital | $11,186.82 |
| **Strategy Return** | **+11.87%** |
| Buy & Hold Return | +3.53% |
| Max Drawdown | 9.46% |
| Win Rate | 62.5% |
| Total Trades | 24 |
| Trading Days | 63 |
| MC 14-Day Prediction | $179.61 |

#### Trade History for NVDA

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 1 | $170.28 | $176.23 | 58.73 | $+349.40 | +3.49% | 📊 SIGNAL_SELL |
| 2 | Day 2 | Day 3 | $176.66 | $183.60 | 58.58 | $+406.55 | +3.93% | 📊 SIGNAL_SELL |
| 3 | Day 4 | Day 5 | $178.42 | $176.96 | 60.28 | $-88.01 | -0.82% | 📊 SIGNAL_SELL |
| 4 | Day 6 | Day 7 | $177.68 | $178.18 | 60.04 | $+30.02 | +0.28% | 📊 SIGNAL_SELL |
| 5 | Day 8 | Day 9 | $181.84 | $186.57 | 58.83 | $+278.26 | +2.60% | 📊 SIGNAL_SELL |
| 6 | Day 10 | Day 11 | $187.23 | $188.88 | 58.62 | $+96.72 | +0.88% | 📊 SIGNAL_SELL |
| 7 | Day 12 | Day 13 | $187.61 | $185.53 | 59.02 | $-122.76 | -1.11% | 📊 SIGNAL_SELL |
| 8 | Day 14 | Day 15 | $185.03 | $189.10 | 59.18 | $+240.85 | +2.20% | 📊 SIGNAL_SELL |
| 9 | Day 16 | Day 17 | $192.56 | $183.15 | 58.12 | $-546.85 | -4.89% | 📊 SIGNAL_SELL |
| 10 | Day 18 | Day 19 | $188.31 | $180.02 | 56.52 | $-468.57 | -4.40% | 📊 SIGNAL_SELL |
| 11 | Day 20 | Day 21 | $179.82 | $181.80 | 56.59 | $+112.04 | +1.10% | 📊 SIGNAL_SELL |
| 12 | Day 22 | Day 23 | $183.21 | $182.63 | 56.15 | $-32.57 | -0.32% | 📊 SIGNAL_SELL |
| 13 | Day 25 | Day 26 | $180.27 | $182.15 | 56.89 | $+106.94 | +1.04% | 📊 SIGNAL_SELL |
| 14 | Day 27 | Day 28 | $186.25 | $191.48 | 55.64 | $+290.96 | +2.81% | 📊 SIGNAL_SELL |
| 15 | Day 29 | Day 30 | $201.02 | $207.03 | 52.99 | $+318.48 | +2.99% | 📊 SIGNAL_SELL |
| 16 | Day 31 | Day 32 | $202.88 | $202.48 | 54.08 | $-21.63 | -0.20% | 📊 SIGNAL_SELL |
| 17 | Day 33 | Day 34 | $206.87 | $198.68 | 52.93 | $-433.48 | -3.96% | 📊 SIGNAL_SELL |
| 18 | Day 35 | Day 36 | $195.20 | $188.07 | 53.88 | $-384.11 | -3.65% | 📊 SIGNAL_SELL |
| 19 | Day 37 | Day 38 | $188.14 | $199.04 | 53.85 | $+586.99 | +5.79% | 📊 SIGNAL_SELL |
| 20 | Day 39 | Day 40 | $193.15 | $193.79 | 55.50 | $+35.52 | +0.33% | 📊 SIGNAL_SELL |
| 21 | Day 41 | Day 42 | $186.85 | $190.16 | 57.56 | $+190.51 | +1.77% | 📊 SIGNAL_SELL |
| 22 | Day 43 | Day 45 | $186.59 | $186.51 | 58.66 | $-4.69 | -0.04% | 📊 SIGNAL_SELL |
| 23 | Day 46 | Day 48 | $180.63 | $182.54 | 60.57 | $+115.68 | +1.06% | 📊 SIGNAL_SELL |
| 24 | Day 49 | Day 52 | $177.81 | $179.91 | 62.18 | $+130.57 | +1.18% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 24
- Winning Trades: 15 (62.5%)
- Losing Trades: 9 (37.5%)
- Total Gains: $3,289.49
- Total Losses: $-2,102.67

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 24
- 📉 Macro Filter Exits: 0

### AMD

| Metric | Value |
|--------|-------|
| Start Price | $159.16 |
| End Price | $207.58 |
| Directional Accuracy | 53.23% |
| Mean Squared Error | 2902.3722 |
| Root Mean Squared Error | $53.87 |
| Mean Absolute Percentage Error | 20.49% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,150.83 |
| **Strategy Return** | **+1.51%** |
| Buy & Hold Return | +30.42% |
| Max Drawdown | 13.34% |
| Win Rate | 42.9% |
| Total Trades | 14 |
| Trading Days | 63 |
| MC 14-Day Prediction | $165.23 |

#### Trade History for AMD

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 11 | Day 12 | $169.73 | $164.67 | 58.92 | $-298.12 | -2.98% | 📊 SIGNAL_SELL |
| 2 | Day 13 | Day 14 | $203.71 | $211.51 | 47.63 | $+371.48 | +3.83% | 📊 SIGNAL_SELL |
| 3 | Day 15 | Day 16 | $235.56 | $232.89 | 42.76 | $-114.18 | -1.13% | 📊 SIGNAL_SELL |
| 4 | Day 17 | Day 18 | $214.90 | $216.42 | 46.34 | $+70.44 | +0.71% | 📊 SIGNAL_SELL |
| 5 | Day 19 | Day 20 | $218.09 | $238.60 | 45.99 | $+943.22 | +9.40% | 📊 SIGNAL_SELL |
| 6 | Day 21 | Day 22 | $234.56 | $233.08 | 46.78 | $-69.24 | -0.63% | 📊 SIGNAL_SELL |
| 7 | Day 25 | Day 26 | $230.23 | $234.99 | 47.36 | $+225.43 | +2.07% | 📊 SIGNAL_SELL |
| 8 | Day 27 | Day 28 | $252.92 | $259.67 | 44.00 | $+297.02 | +2.67% | 📊 SIGNAL_SELL |
| 9 | Day 34 | Day 35 | $250.05 | $256.33 | 45.70 | $+286.96 | +2.51% | 📊 SIGNAL_SELL |
| 10 | Day 36 | Day 37 | $237.70 | $233.54 | 49.28 | $-204.99 | -1.75% | 📊 SIGNAL_SELL |
| 11 | Day 38 | Day 39 | $243.98 | $237.52 | 47.17 | $-304.70 | -2.65% | 📊 SIGNAL_SELL |
| 12 | Day 40 | Day 41 | $258.89 | $247.96 | 43.27 | $-472.99 | -4.22% | 📊 SIGNAL_SELL |
| 13 | Day 42 | Day 43 | $246.81 | $240.52 | 43.48 | $-273.46 | -2.55% | 📊 SIGNAL_SELL |
| 14 | Day 44 | Day 45 | $230.29 | $223.55 | 45.41 | $-306.05 | -2.93% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 14
- Winning Trades: 6 (42.9%)
- Losing Trades: 8 (57.1%)
- Total Gains: $2,194.56
- Total Losses: $-2,043.73

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 14
- 📉 Macro Filter Exits: 0

### INTC

| Metric | Value |
|--------|-------|
| Start Price | $24.90 |
| End Price | $37.51 |
| Directional Accuracy | 50.00% |
| Mean Squared Error | 206.3824 |
| Root Mean Squared Error | $14.37 |
| Mean Absolute Percentage Error | 36.76% |
| Starting Capital | $10,000.00 |
| Final Capital | $14,667.67 |
| **Strategy Return** | **+46.68%** |
| Buy & Hold Return | +50.64% |
| Max Drawdown | 3.11% |
| Win Rate | 65.0% |
| Total Trades | 20 |
| Trading Days | 63 |
| MC 14-Day Prediction | $21.37 |

#### Trade History for INTC

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 1 | $24.90 | $30.57 | 401.61 | $+2277.11 | +22.77% | 📊 SIGNAL_SELL |
| 2 | Day 2 | Day 3 | $29.58 | $28.76 | 415.05 | $-340.34 | -2.77% | 📊 SIGNAL_SELL |
| 3 | Day 4 | Day 5 | $29.34 | $31.22 | 406.84 | $+764.86 | +6.41% | 📊 SIGNAL_SELL |
| 4 | Day 6 | Day 7 | $33.99 | $35.50 | 373.69 | $+564.27 | +4.44% | 📊 SIGNAL_SELL |
| 5 | Day 8 | Day 9 | $34.48 | $33.55 | 384.74 | $-357.81 | -2.70% | 📊 SIGNAL_SELL |
| 6 | Day 10 | Day 11 | $35.94 | $37.30 | 359.16 | $+488.45 | +3.78% | 📊 SIGNAL_SELL |
| 7 | Day 12 | Day 13 | $36.83 | $36.59 | 363.74 | $-87.30 | -0.65% | 📊 SIGNAL_SELL |
| 8 | Day 17 | Day 18 | $36.37 | $37.22 | 365.94 | $+311.05 | +2.34% | 📊 SIGNAL_SELL |
| 9 | Day 22 | Day 23 | $37.01 | $38.10 | 368.02 | $+401.14 | +2.95% | 📊 SIGNAL_SELL |
| 10 | Day 26 | Day 27 | $38.16 | $38.28 | 367.44 | $+44.09 | +0.31% | 📊 SIGNAL_SELL |
| 11 | Day 28 | Day 29 | $39.54 | $41.53 | 355.73 | $+707.90 | +5.03% | 📊 SIGNAL_SELL |
| 12 | Day 30 | Day 31 | $41.34 | $40.16 | 357.36 | $-421.69 | -2.85% | 📊 SIGNAL_SELL |
| 13 | Day 34 | Day 35 | $37.03 | $38.38 | 387.57 | $+523.22 | +3.65% | 📊 SIGNAL_SELL |
| 14 | Day 37 | Day 38 | $38.13 | $38.45 | 390.11 | $+124.84 | +0.84% | 📊 SIGNAL_SELL |
| 15 | Day 41 | Day 42 | $35.91 | $35.52 | 417.71 | $-162.90 | -1.09% | 📊 SIGNAL_SELL |
| 16 | Day 51 | Day 52 | $40.56 | $40.01 | 365.80 | $-201.19 | -1.36% | 📊 SIGNAL_SELL |
| 17 | Day 53 | Day 54 | $43.47 | $43.76 | 336.68 | $+97.64 | +0.67% | 📊 SIGNAL_SELL |
| 18 | Day 55 | Day 56 | $40.50 | $41.41 | 363.79 | $+331.05 | +2.25% | 📊 SIGNAL_SELL |
| 19 | Day 57 | Day 58 | $40.30 | $40.50 | 373.81 | $+74.76 | +0.50% | 📊 SIGNAL_SELL |
| 20 | Day 59 | Day 60 | $40.78 | $39.51 | 371.24 | $-471.47 | -3.11% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 20
- Winning Trades: 13 (65.0%)
- Losing Trades: 7 (35.0%)
- Total Gains: $6,710.37
- Total Losses: $-2,042.71

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 20
- 📉 Macro Filter Exits: 0

### GOOGL

| Metric | Value |
|--------|-------|
| Start Price | $249.37 |
| End Price | $308.22 |
| Directional Accuracy | 48.39% |
| Mean Squared Error | 2293.4202 |
| Root Mean Squared Error | $47.89 |
| Mean Absolute Percentage Error | 15.04% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,357.12 |
| **Strategy Return** | **+3.57%** |
| Buy & Hold Return | +23.60% |
| Max Drawdown | 7.70% |
| Win Rate | 44.4% |
| Total Trades | 18 |
| Trading Days | 63 |
| MC 14-Day Prediction | $222.27 |

#### Trade History for GOOGL

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 17 | $249.37 | $236.15 | 40.10 | $-530.19 | -5.30% |  ATR-STOP (5.3%) |
| 2 | Day 18 | Day 22 | $243.99 | $253.13 | 38.81 | $+354.90 | +3.75% | 📊 SIGNAL_SELL |
| 3 | Day 23 | Day 24 | $256.38 | $244.53 | 38.32 | $-454.35 | -4.62% |  ATR-STOP (4.6%) |
| 4 | Day 25 | Day 26 | $251.53 | $252.91 | 37.25 | $+51.75 | +0.55% | 📊 SIGNAL_SELL |
| 5 | Day 27 | Day 28 | $259.75 | $269.09 | 36.27 | $+338.94 | +3.60% | 📊 SIGNAL_SELL |
| 6 | Day 29 | Day 30 | $267.30 | $274.39 | 36.52 | $+259.11 | +2.65% | 📊 SIGNAL_SELL |
| 7 | Day 31 | Day 32 | $281.30 | $281.01 | 35.62 | $-10.32 | -0.10% | 📊 SIGNAL_SELL |
| 8 | Day 33 | Day 34 | $283.53 | $277.36 | 35.30 | $-218.03 | -2.18% | 📊 SIGNAL_SELL |
| 9 | Day 35 | Day 36 | $284.12 | $284.56 | 34.46 | $+15.15 | +0.15% | 📊 SIGNAL_SELL |
| 10 | Day 37 | Day 38 | $278.65 | $289.91 | 35.19 | $+396.39 | +4.04% | 📊 SIGNAL_SELL |
| 11 | Day 41 | Day 42 | $278.39 | $276.23 | 36.65 | $-79.12 | -0.78% | 📊 SIGNAL_SELL |
| 12 | Day 43 | Day 44 | $284.83 | $284.09 | 35.54 | $-26.29 | -0.26% | 📊 SIGNAL_SELL |
| 13 | Day 45 | Day 46 | $292.62 | $289.26 | 34.51 | $-115.87 | -1.15% | 📊 SIGNAL_SELL |
| 14 | Day 47 | Day 48 | $299.46 | $318.37 | 33.33 | $+630.25 | +6.31% | 📊 SIGNAL_SELL |
| 15 | Day 49 | Day 50 | $323.23 | $319.74 | 32.83 | $-114.51 | -1.08% | 📊 SIGNAL_SELL |
| 16 | Day 52 | Day 53 | $314.68 | $315.60 | 33.36 | $+30.67 | +0.29% | 📊 SIGNAL_SELL |
| 17 | Day 54 | Day 55 | $319.42 | $317.41 | 32.96 | $-66.21 | -0.63% | 📊 SIGNAL_SELL |
| 18 | Day 60 | Day 61 | $312.43 | $309.29 | 33.49 | $-105.15 | -1.01% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 18
- Winning Trades: 8 (44.4%)
- Losing Trades: 10 (55.6%)
- Total Gains: $2,077.15
- Total Losses: $-1,720.04

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 16
- 📉 Macro Filter Exits: 0

### TSLA

| Metric | Value |
|--------|-------|
| Start Price | $425.86 |
| End Price | $475.31 |
| Directional Accuracy | 45.90% |
| Mean Squared Error | 5648.1934 |
| Root Mean Squared Error | $75.15 |
| Mean Absolute Percentage Error | 15.47% |
| Starting Capital | $10,000.00 |
| Final Capital | $12,897.17 |
| **Strategy Return** | **+28.97%** |
| Buy & Hold Return | +11.61% |
| Max Drawdown | 8.46% |
| Win Rate | 76.2% |
| Total Trades | 21 |
| Trading Days | 62 |
| MC 14-Day Prediction | $334.68 |

#### Trade History for TSLA

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 1 | $425.86 | $416.85 | 23.48 | $-211.57 | -2.12% | 📊 SIGNAL_SELL |
| 2 | Day 2 | Day 3 | $426.07 | $434.21 | 22.97 | $+187.01 | +1.91% | 📊 SIGNAL_SELL |
| 3 | Day 4 | Day 5 | $425.85 | $442.79 | 23.42 | $+396.82 | +3.98% | 📊 SIGNAL_SELL |
| 4 | Day 6 | Day 7 | $423.39 | $440.40 | 24.50 | $+416.71 | +4.02% | 📊 SIGNAL_SELL |
| 5 | Day 8 | Day 9 | $443.21 | $444.72 | 24.34 | $+36.76 | +0.34% | 📊 SIGNAL_SELL |
| 6 | Day 10 | Day 11 | $459.46 | $436.00 | 23.56 | $-552.76 | -5.11% | 📊 SIGNAL_SELL |
| 7 | Day 12 | Day 13 | $429.83 | $453.25 | 23.90 | $+559.74 | +5.45% | 📊 SIGNAL_SELL |
| 8 | Day 14 | Day 15 | $433.09 | $438.69 | 25.01 | $+140.07 | +1.29% | 📊 SIGNAL_SELL |
| 9 | Day 17 | Day 18 | $413.49 | $435.90 | 26.54 | $+594.69 | +5.42% | 📊 SIGNAL_SELL |
| 10 | Day 19 | Day 20 | $429.24 | $435.15 | 26.95 | $+159.27 | +1.38% | 📊 SIGNAL_SELL |
| 11 | Day 21 | Day 22 | $428.75 | $439.31 | 27.35 | $+288.83 | +2.46% | 📊 SIGNAL_SELL |
| 12 | Day 25 | Day 26 | $438.97 | $448.98 | 27.37 | $+274.00 | +2.28% | 📊 SIGNAL_SELL |
| 13 | Day 27 | Day 28 | $433.72 | $452.42 | 28.34 | $+529.87 | +4.31% | 📊 SIGNAL_SELL |
| 14 | Day 29 | Day 30 | $460.55 | $461.51 | 27.84 | $+26.72 | +0.21% | 📊 SIGNAL_SELL |
| 15 | Day 31 | Day 32 | $440.10 | $456.56 | 29.19 | $+480.45 | +3.74% | 📊 SIGNAL_SELL |
| 16 | Day 33 | Day 34 | $468.37 | $444.26 | 28.45 | $-686.00 | -5.15% | 📊 SIGNAL_SELL |
| 17 | Day 35 | Day 36 | $462.07 | $445.91 | 27.36 | $-442.08 | -3.50% | 📊 SIGNAL_SELL |
| 18 | Day 37 | Day 38 | $429.52 | $445.23 | 28.40 | $+446.17 | +3.66% | 📊 SIGNAL_SELL |
| 19 | Day 54 | Day 55 | $446.74 | $454.53 | 28.30 | $+220.49 | +1.74% | 📊 SIGNAL_SELL |
| 20 | Day 57 | Day 58 | $439.58 | $445.17 | 29.27 | $+163.60 | +1.27% | 📊 SIGNAL_SELL |
| 21 | Day 59 | Day 60 | $451.45 | $446.89 | 28.86 | $-131.60 | -1.01% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 21
- Winning Trades: 16 (76.2%)
- Losing Trades: 5 (23.8%)
- Total Gains: $4,921.19
- Total Losses: $-2,024.02

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 21
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