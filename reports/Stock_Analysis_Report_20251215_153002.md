# Monte Carlo Stock Analysis Report (Hybrid Strategy (Sniper Shorts))

**Generated:** 2025-12-15 15:32:02

**Analysis Period:** 2025-10-16 to 2025-12-15 (Past 60 Days)

**Training Period:** 2023-10-17 to 2025-10-16 (Rolling 2-Year Window)

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
| QQQ | +1.8% ($+184) | +4.3% 📉 | 56% | 6.3% | 16 | ✅ GAIN |
| SPY | +3.9% ($+391) | +4.3% 📉 | 73% | 2.6% | 15 | ✅ GAIN |
| AAPL | +5.2% ($+518) | +12.5% 📉 | 56% | 1.7% | 18 | ✅ GAIN |
| NVDA | +10.9% ($+1,087) | -0.5% 📈 | 69% | 7.6% | 13 | ✅ GAIN |
| AMD | -4.5% ($-452) | -5.6% 📈 | 44% | 13.3% | 9 | ❌ LOSS |
| INTC | +1.6% ($+161) | +7.2% 📉 | 50% | 13.5% | 10 | ✅ GAIN |
| GOOGL | +7.3% ($+727) | +24.3% 📉 | 50% | 4.1% | 16 | ✅ GAIN |
| TSLA | +10.0% ($+998) | +4.2% 📈 | 73% | 8.5% | 11 | ✅ GAIN |


## Portfolio Summary

| Metric | Strategy | Buy & Hold |
|--------|----------|------------|
| Total Invested | $80,000.00 | $80,000.00 |
| Total P&L | $+3,614.03 | $+5,078.05 |
| Portfolio Return | +4.52% | +6.35% |
| Final Value | $83,614.03 | $85,078.05 |

**Strategy vs Buy & Hold:** -1.83% (Underperformed ❌)


## Detailed Stock Analysis


### QQQ

| Metric | Value |
|--------|-------|
| Start Price | $599.99 |
| End Price | $625.58 |
| Directional Accuracy | 48.72% |
| Mean Squared Error | 658.8739 |
| Root Mean Squared Error | $25.67 |
| Mean Absolute Percentage Error | 3.69% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,183.87 |
| **Strategy Return** | **+1.84%** |
| Buy & Hold Return | +4.27% |
| Max Drawdown | 6.32% |
| Win Rate | 56.2% |
| Total Trades | 16 |
| Trading Days | 40 |
| MC 14-Day Prediction | $601.17 |

#### Trade History for QQQ

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 1 | $599.99 | $603.93 | 16.67 | $+65.67 | +0.66% | 📊 SIGNAL_SELL |
| 2 | Day 2 | Day 3 | $611.54 | $611.38 | 16.46 | $-2.63 | -0.03% | 📊 SIGNAL_SELL |
| 3 | Day 4 | Day 5 | $605.49 | $610.58 | 16.62 | $+84.59 | +0.84% | 📊 SIGNAL_SELL |
| 4 | Day 6 | Day 7 | $617.10 | $628.09 | 16.44 | $+180.72 | +1.78% | 📊 SIGNAL_SELL |
| 5 | Day 8 | Day 9 | $632.92 | $635.77 | 16.32 | $+46.51 | +0.45% | 📊 SIGNAL_SELL |
| 6 | Day 10 | Day 11 | $626.05 | $629.07 | 16.57 | $+50.05 | +0.48% | 📊 SIGNAL_SELL |
| 7 | Day 13 | Day 14 | $619.25 | $623.28 | 16.83 | $+67.84 | +0.65% | 📊 SIGNAL_SELL |
| 8 | Day 15 | Day 16 | $611.67 | $609.74 | 17.15 | $-33.11 | -0.32% | 📊 SIGNAL_SELL |
| 9 | Day 17 | Day 18 | $623.23 | $621.57 | 16.78 | $-27.86 | -0.27% | 📊 SIGNAL_SELL |
| 10 | Day 19 | Day 20 | $621.08 | $608.40 | 16.80 | $-212.98 | -2.04% | 📊 SIGNAL_SELL |
| 11 | Day 21 | Day 25 | $608.86 | $588.06 | 16.78 | $-349.05 | -3.42% |  ATR-STOP (3.4%) |
| 12 | Day 26 | Day 27 | $590.07 | $605.16 | 16.73 | $+252.40 | +2.56% | 📊 SIGNAL_SELL |
| 13 | Day 28 | Day 29 | $608.89 | $614.27 | 16.62 | $+89.44 | +0.88% | 📊 SIGNAL_SELL |
| 14 | Day 32 | Day 33 | $622.00 | $623.52 | 16.42 | $+24.95 | +0.24% | 📊 SIGNAL_SELL |
| 15 | Day 35 | Day 36 | $625.48 | $624.28 | 16.37 | $-19.64 | -0.19% | 📊 SIGNAL_SELL |
| 16 | Day 38 | Day 39 | $627.61 | $625.58 | 16.28 | $-33.05 | -0.32% | ⏰ END_OF_PERIOD |

**Trade Summary:**
- Total Trades: 16
- Winning Trades: 9 (56.2%)
- Losing Trades: 7 (43.8%)
- Total Gains: $862.18
- Total Losses: $-678.31

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 14
- 📉 Macro Filter Exits: 0

### SPY

| Metric | Value |
|--------|-------|
| Start Price | $660.64 |
| End Price | $689.17 |
| Directional Accuracy | 51.28% |
| Mean Squared Error | 540.0665 |
| Root Mean Squared Error | $23.24 |
| Mean Absolute Percentage Error | 3.12% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,391.43 |
| **Strategy Return** | **+3.91%** |
| Buy & Hold Return | +4.32% |
| Max Drawdown | 2.58% |
| Win Rate | 73.3% |
| Total Trades | 15 |
| Trading Days | 40 |
| MC 14-Day Prediction | $661.06 |

#### Trade History for SPY

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 1 | $660.64 | $664.39 | 15.14 | $+56.76 | +0.57% | 📊 SIGNAL_SELL |
| 2 | Day 2 | Day 3 | $671.30 | $671.29 | 14.98 | $-0.15 | -0.00% | 📊 SIGNAL_SELL |
| 3 | Day 4 | Day 5 | $667.80 | $671.76 | 15.06 | $+59.64 | +0.59% | 📊 SIGNAL_SELL |
| 4 | Day 6 | Day 7 | $677.25 | $685.24 | 14.94 | $+119.35 | +1.18% | 📊 SIGNAL_SELL |
| 5 | Day 8 | Day 9 | $687.06 | $687.39 | 14.90 | $+4.92 | +0.05% | 📊 SIGNAL_SELL |
| 6 | Day 10 | Day 11 | $679.83 | $682.06 | 15.06 | $+33.59 | +0.33% | 📊 SIGNAL_SELL |
| 7 | Day 13 | Day 14 | $675.24 | $677.58 | 15.22 | $+35.60 | +0.35% | 📊 SIGNAL_SELL |
| 8 | Day 15 | Day 16 | $670.31 | $670.97 | 15.38 | $+10.15 | +0.10% | 📊 SIGNAL_SELL |
| 9 | Day 17 | Day 18 | $681.44 | $683.00 | 15.14 | $+23.62 | +0.23% | 📊 SIGNAL_SELL |
| 10 | Day 19 | Day 20 | $683.38 | $672.04 | 15.14 | $-171.64 | -1.66% | 📊 SIGNAL_SELL |
| 11 | Day 21 | Day 22 | $671.93 | $665.67 | 15.14 | $-94.77 | -0.93% | 📊 SIGNAL_SELL |
| 12 | Day 25 | Day 27 | $652.53 | $668.73 | 15.44 | $+250.18 | +2.48% | 📊 SIGNAL_SELL |
| 13 | Day 28 | Day 29 | $675.02 | $679.68 | 15.30 | $+71.29 | +0.69% | 📊 SIGNAL_SELL |
| 14 | Day 35 | Day 36 | $685.69 | $683.63 | 15.17 | $-31.24 | -0.30% | 📊 SIGNAL_SELL |
| 15 | Day 38 | Day 39 | $687.57 | $689.17 | 15.08 | $+24.12 | +0.23% | ⏰ END_OF_PERIOD |

**Trade Summary:**
- Total Trades: 15
- Winning Trades: 11 (73.3%)
- Losing Trades: 4 (26.7%)
- Total Gains: $689.23
- Total Losses: $-297.80

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 14
- 📉 Macro Filter Exits: 0

### AAPL

| Metric | Value |
|--------|-------|
| Start Price | $247.21 |
| End Price | $278.03 |
| Directional Accuracy | 43.59% |
| Mean Squared Error | 1192.4854 |
| Root Mean Squared Error | $34.53 |
| Mean Absolute Percentage Error | 12.18% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,517.58 |
| **Strategy Return** | **+5.18%** |
| Buy & Hold Return | +12.47% |
| Max Drawdown | 1.70% |
| Win Rate | 55.6% |
| Total Trades | 18 |
| Trading Days | 40 |
| MC 14-Day Prediction | $239.04 |

#### Trade History for AAPL

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 1 | $247.21 | $252.05 | 40.45 | $+195.59 | +1.96% | 📊 SIGNAL_SELL |
| 2 | Day 2 | Day 3 | $261.99 | $262.52 | 38.92 | $+20.61 | +0.20% | 📊 SIGNAL_SELL |
| 3 | Day 4 | Day 5 | $258.20 | $259.33 | 39.57 | $+44.67 | +0.44% | 📊 SIGNAL_SELL |
| 4 | Day 6 | Day 7 | $262.57 | $268.55 | 39.08 | $+233.86 | +2.28% | 📊 SIGNAL_SELL |
| 5 | Day 8 | Day 9 | $268.74 | $269.44 | 39.05 | $+27.31 | +0.26% | 📊 SIGNAL_SELL |
| 6 | Day 10 | Day 11 | $271.14 | $270.11 | 38.81 | $-39.93 | -0.38% | 📊 SIGNAL_SELL |
| 7 | Day 12 | Day 13 | $268.79 | $269.78 | 39.00 | $+38.57 | +0.37% | 📊 SIGNAL_SELL |
| 8 | Day 14 | Day 15 | $269.88 | $269.51 | 38.98 | $-14.41 | -0.14% | 📊 SIGNAL_SELL |
| 9 | Day 16 | Day 17 | $268.21 | $269.43 | 39.17 | $+47.79 | +0.45% | 📊 SIGNAL_SELL |
| 10 | Day 18 | Day 19 | $275.25 | $273.47 | 38.34 | $-68.25 | -0.65% | 📊 SIGNAL_SELL |
| 11 | Day 20 | Day 21 | $272.95 | $272.41 | 38.42 | $-20.75 | -0.20% | 📊 SIGNAL_SELL |
| 12 | Day 22 | Day 23 | $267.46 | $267.44 | 39.13 | $-0.78 | -0.01% | 📊 SIGNAL_SELL |
| 13 | Day 24 | Day 25 | $268.56 | $266.25 | 38.96 | $-90.01 | -0.86% | 📊 SIGNAL_SELL |
| 14 | Day 26 | Day 27 | $271.49 | $275.92 | 38.21 | $+169.28 | +1.63% | 📊 SIGNAL_SELL |
| 15 | Day 28 | Day 29 | $276.97 | $277.55 | 38.07 | $+22.08 | +0.21% | 📊 SIGNAL_SELL |
| 16 | Day 31 | Day 32 | $283.10 | $286.19 | 37.32 | $+115.32 | +1.09% | 📊 SIGNAL_SELL |
| 17 | Day 33 | Day 34 | $284.15 | $280.70 | 37.59 | $-129.68 | -1.21% | 📊 SIGNAL_SELL |
| 18 | Day 35 | Day 36 | $278.78 | $277.89 | 37.85 | $-33.68 | -0.32% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 18
- Winning Trades: 10 (55.6%)
- Losing Trades: 8 (44.4%)
- Total Gains: $915.08
- Total Losses: $-397.49

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 18
- 📉 Macro Filter Exits: 0

### NVDA

| Metric | Value |
|--------|-------|
| Start Price | $181.80 |
| End Price | $180.93 |
| Directional Accuracy | 38.46% |
| Mean Squared Error | 117.7999 |
| Root Mean Squared Error | $10.85 |
| Mean Absolute Percentage Error | 3.78% |
| Starting Capital | $10,000.00 |
| Final Capital | $11,087.25 |
| **Strategy Return** | **+10.87%** |
| Buy & Hold Return | -0.48% |
| Max Drawdown | 7.65% |
| Win Rate | 69.2% |
| Total Trades | 13 |
| Trading Days | 40 |
| MC 14-Day Prediction | $180.86 |

#### Trade History for NVDA

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 1 | $181.80 | $183.21 | 55.01 | $+77.55 | +0.78% | 📊 SIGNAL_SELL |
| 2 | Day 4 | Day 5 | $180.27 | $182.15 | 55.90 | $+105.09 | +1.04% | 📊 SIGNAL_SELL |
| 3 | Day 6 | Day 7 | $186.25 | $191.48 | 54.67 | $+285.92 | +2.81% | 📊 SIGNAL_SELL |
| 4 | Day 8 | Day 9 | $201.02 | $207.03 | 52.08 | $+312.97 | +2.99% | 📊 SIGNAL_SELL |
| 5 | Day 10 | Day 11 | $202.88 | $202.48 | 53.14 | $-21.26 | -0.20% | 📊 SIGNAL_SELL |
| 6 | Day 12 | Day 13 | $206.87 | $198.68 | 52.02 | $-425.98 | -3.96% | 📊 SIGNAL_SELL |
| 7 | Day 14 | Day 15 | $195.20 | $188.07 | 52.94 | $-377.46 | -3.65% | 📊 SIGNAL_SELL |
| 8 | Day 16 | Day 17 | $188.14 | $199.04 | 52.92 | $+576.82 | +5.79% | 📊 SIGNAL_SELL |
| 9 | Day 18 | Day 19 | $193.15 | $193.79 | 54.54 | $+34.90 | +0.33% | 📊 SIGNAL_SELL |
| 10 | Day 20 | Day 21 | $186.85 | $190.16 | 56.56 | $+187.21 | +1.77% | 📊 SIGNAL_SELL |
| 11 | Day 22 | Day 24 | $186.59 | $186.51 | 57.64 | $-4.61 | -0.04% | 📊 SIGNAL_SELL |
| 12 | Day 25 | Day 27 | $180.63 | $182.54 | 59.52 | $+113.68 | +1.06% | 📊 SIGNAL_SELL |
| 13 | Day 28 | Day 32 | $177.81 | $181.45 | 61.10 | $+222.41 | +2.05% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 13
- Winning Trades: 9 (69.2%)
- Losing Trades: 4 (30.8%)
- Total Gains: $1,916.55
- Total Losses: $-829.31

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 13
- 📉 Macro Filter Exits: 0

### AMD

| Metric | Value |
|--------|-------|
| Start Price | $234.56 |
| End Price | $221.43 |
| Directional Accuracy | 51.28% |
| Mean Squared Error | 2306.8395 |
| Root Mean Squared Error | $48.03 |
| Mean Absolute Percentage Error | 18.75% |
| Starting Capital | $10,000.00 |
| Final Capital | $9,547.74 |
| **Strategy Return** | **-4.52%** |
| Buy & Hold Return | -5.60% |
| Max Drawdown | 13.34% |
| Win Rate | 44.4% |
| Total Trades | 9 |
| Trading Days | 40 |
| MC 14-Day Prediction | $196.58 |

#### Trade History for AMD

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 2 | $234.56 | $240.56 | 42.63 | $+255.80 | +2.56% | 📊 SIGNAL_SELL |
| 2 | Day 4 | Day 5 | $230.23 | $234.99 | 44.55 | $+212.04 | +2.07% | 📊 SIGNAL_SELL |
| 3 | Day 6 | Day 7 | $252.92 | $259.67 | 41.39 | $+279.37 | +2.67% | 📊 SIGNAL_SELL |
| 4 | Day 13 | Day 14 | $250.05 | $256.33 | 42.98 | $+269.92 | +2.51% | 📊 SIGNAL_SELL |
| 5 | Day 15 | Day 16 | $237.70 | $233.54 | 46.35 | $-192.81 | -1.75% | 📊 SIGNAL_SELL |
| 6 | Day 17 | Day 18 | $243.98 | $237.52 | 44.37 | $-286.60 | -2.65% | 📊 SIGNAL_SELL |
| 7 | Day 19 | Day 20 | $258.89 | $247.96 | 40.70 | $-444.89 | -4.22% | 📊 SIGNAL_SELL |
| 8 | Day 21 | Day 22 | $246.81 | $240.52 | 40.89 | $-257.22 | -2.55% | 📊 SIGNAL_SELL |
| 9 | Day 23 | Day 24 | $230.29 | $223.55 | 42.71 | $-287.86 | -2.93% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 9
- Winning Trades: 4 (44.4%)
- Losing Trades: 5 (55.6%)
- Total Gains: $1,017.12
- Total Losses: $-1,469.38

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 9
- 📉 Macro Filter Exits: 0

### INTC

| Metric | Value |
|--------|-------|
| Start Price | $36.84 |
| End Price | $39.51 |
| Directional Accuracy | 41.03% |
| Mean Squared Error | 6.4675 |
| Root Mean Squared Error | $2.54 |
| Mean Absolute Percentage Error | 5.47% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,161.21 |
| **Strategy Return** | **+1.61%** |
| Buy & Hold Return | +7.25% |
| Max Drawdown | 13.47% |
| Win Rate | 50.0% |
| Total Trades | 10 |
| Trading Days | 40 |
| MC 14-Day Prediction | $39.47 |

#### Trade History for INTC

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 1 | Day 2 | $37.01 | $38.10 | 270.20 | $+294.52 | +2.95% | 📊 SIGNAL_SELL |
| 2 | Day 5 | Day 6 | $38.16 | $38.28 | 269.77 | $+32.37 | +0.31% | 📊 SIGNAL_SELL |
| 3 | Day 7 | Day 8 | $39.54 | $41.53 | 261.18 | $+519.74 | +5.03% | 📊 SIGNAL_SELL |
| 4 | Day 9 | Day 10 | $41.34 | $40.16 | 262.38 | $-309.60 | -2.85% | 📊 SIGNAL_SELL |
| 5 | Day 13 | Day 26 | $37.03 | $33.23 | 284.55 | $-1080.08 | -10.25% |  ATR-STOP (10.3%) |
| 6 | Day 27 | Day 30 | $35.79 | $40.56 | 264.23 | $+1260.40 | +13.33% | 📊 SIGNAL_SELL |
| 7 | Day 31 | Day 32 | $40.01 | $43.47 | 267.87 | $+926.82 | +8.65% | 📊 SIGNAL_SELL |
| 8 | Day 33 | Day 34 | $43.76 | $40.50 | 266.09 | $-867.46 | -7.45% | 📊 SIGNAL_SELL |
| 9 | Day 35 | Day 36 | $41.41 | $40.30 | 260.24 | $-288.87 | -2.68% | 📊 SIGNAL_SELL |
| 10 | Day 38 | Day 39 | $40.78 | $39.51 | 257.18 | $-326.62 | -3.11% | ⏰ END_OF_PERIOD |

**Trade Summary:**
- Total Trades: 10
- Winning Trades: 5 (50.0%)
- Losing Trades: 5 (50.0%)
- Total Gains: $3,033.84
- Total Losses: $-2,872.64

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 8
- 📉 Macro Filter Exits: 0

### GOOGL

| Metric | Value |
|--------|-------|
| Start Price | $251.30 |
| End Price | $312.43 |
| Directional Accuracy | 56.41% |
| Mean Squared Error | 2796.3741 |
| Root Mean Squared Error | $52.88 |
| Mean Absolute Percentage Error | 15.34% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,726.85 |
| **Strategy Return** | **+7.27%** |
| Buy & Hold Return | +24.33% |
| Max Drawdown | 4.10% |
| Win Rate | 50.0% |
| Total Trades | 16 |
| Trading Days | 40 |
| MC 14-Day Prediction | $195.15 |

#### Trade History for GOOGL

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 1 | $251.30 | $253.13 | 39.79 | $+73.17 | +0.73% | 📊 SIGNAL_SELL |
| 2 | Day 2 | Day 3 | $256.38 | $244.53 | 39.29 | $-465.84 | -4.62% |  ATR-STOP (4.6%) |
| 3 | Day 4 | Day 5 | $251.53 | $252.91 | 38.20 | $+53.06 | +0.55% | 📊 SIGNAL_SELL |
| 4 | Day 6 | Day 7 | $259.75 | $269.09 | 37.19 | $+347.51 | +3.60% | 📊 SIGNAL_SELL |
| 5 | Day 8 | Day 9 | $267.30 | $274.39 | 37.44 | $+265.66 | +2.65% | 📊 SIGNAL_SELL |
| 6 | Day 10 | Day 11 | $281.30 | $281.01 | 36.52 | $-10.59 | -0.10% | 📊 SIGNAL_SELL |
| 7 | Day 12 | Day 13 | $283.53 | $277.36 | 36.20 | $-223.55 | -2.18% | 📊 SIGNAL_SELL |
| 8 | Day 14 | Day 15 | $284.12 | $284.56 | 35.33 | $+15.54 | +0.15% | 📊 SIGNAL_SELL |
| 9 | Day 16 | Day 17 | $278.65 | $289.91 | 36.08 | $+406.41 | +4.04% | 📊 SIGNAL_SELL |
| 10 | Day 20 | Day 21 | $278.39 | $276.23 | 37.58 | $-81.12 | -0.78% | 📊 SIGNAL_SELL |
| 11 | Day 22 | Day 23 | $284.83 | $284.09 | 36.44 | $-26.95 | -0.26% | 📊 SIGNAL_SELL |
| 12 | Day 24 | Day 25 | $292.62 | $289.26 | 35.38 | $-118.80 | -1.15% | 📊 SIGNAL_SELL |
| 13 | Day 26 | Day 27 | $299.46 | $318.37 | 34.18 | $+646.19 | +6.31% | 📊 SIGNAL_SELL |
| 14 | Day 28 | Day 29 | $323.23 | $319.74 | 33.66 | $-117.40 | -1.08% | 📊 SIGNAL_SELL |
| 15 | Day 31 | Day 32 | $314.68 | $315.60 | 34.20 | $+31.45 | +0.29% | 📊 SIGNAL_SELL |
| 16 | Day 33 | Day 34 | $319.42 | $317.41 | 33.79 | $-67.88 | -0.63% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 16
- Winning Trades: 8 (50.0%)
- Losing Trades: 8 (50.0%)
- Total Gains: $1,838.98
- Total Losses: $-1,112.13

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 15
- 📉 Macro Filter Exits: 0

### TSLA

| Metric | Value |
|--------|-------|
| Start Price | $428.75 |
| End Price | $446.89 |
| Directional Accuracy | 38.46% |
| Mean Squared Error | 3035.9578 |
| Root Mean Squared Error | $55.10 |
| Mean Absolute Percentage Error | 10.04% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,998.10 |
| **Strategy Return** | **+9.98%** |
| Buy & Hold Return | +4.23% |
| Max Drawdown | 8.46% |
| Win Rate | 72.7% |
| Total Trades | 11 |
| Trading Days | 40 |
| MC 14-Day Prediction | $394.06 |

#### Trade History for TSLA

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 1 | $428.75 | $439.31 | 23.32 | $+246.30 | +2.46% | 📊 SIGNAL_SELL |
| 2 | Day 4 | Day 5 | $438.97 | $448.98 | 23.34 | $+233.65 | +2.28% | 📊 SIGNAL_SELL |
| 3 | Day 6 | Day 7 | $433.72 | $452.42 | 24.16 | $+451.85 | +4.31% | 📊 SIGNAL_SELL |
| 4 | Day 8 | Day 9 | $460.55 | $461.51 | 23.74 | $+22.79 | +0.21% | 📊 SIGNAL_SELL |
| 5 | Day 10 | Day 11 | $440.10 | $456.56 | 24.89 | $+409.71 | +3.74% | 📊 SIGNAL_SELL |
| 6 | Day 12 | Day 13 | $468.37 | $444.26 | 24.26 | $-584.99 | -5.15% | 📊 SIGNAL_SELL |
| 7 | Day 14 | Day 15 | $462.07 | $445.91 | 23.33 | $-376.99 | -3.50% | 📊 SIGNAL_SELL |
| 8 | Day 16 | Day 17 | $429.52 | $445.23 | 24.22 | $+380.47 | +3.66% | 📊 SIGNAL_SELL |
| 9 | Day 33 | Day 34 | $446.74 | $454.53 | 24.14 | $+188.02 | +1.74% | 📊 SIGNAL_SELL |
| 10 | Day 36 | Day 37 | $439.58 | $445.17 | 24.96 | $+139.51 | +1.27% | 📊 SIGNAL_SELL |
| 11 | Day 38 | Day 39 | $451.45 | $446.89 | 24.61 | $-112.22 | -1.01% | ⏰ END_OF_PERIOD |

**Trade Summary:**
- Total Trades: 11
- Winning Trades: 8 (72.7%)
- Losing Trades: 3 (27.3%)
- Total Gains: $2,072.30
- Total Losses: $-1,074.20

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 10
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