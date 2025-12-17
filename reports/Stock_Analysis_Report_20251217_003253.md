# Monte Carlo Stock Analysis Report (Hybrid Strategy (Sniper Shorts))

**Generated:** 2025-12-17 00:57:01

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
| QQQ | +2.6% ($+259) | +2.9% 📉 | 56% | 5.5% | 16 | ✅ GAIN |
| SPY | -0.5% ($-47) | +2.8% 📉 | 64% | 2.0% | 11 | ❌ LOSS |
| AAPL | +10.6% ($+1,065) | +15.6% 📉 | 65% | 2.0% | 17 | ✅ GAIN |
| NVDA | -4.3% ($-426) | +0.8% 📉 | 50% | 11.7% | 12 | ❌ LOSS |
| AMD | +37.9% ($+3,794) | +32.5% 📈 | 53% | 14.3% | 15 | ✅ GAIN |
| INTC | +15.1% ($+1,515) | +22.0% 📉 | 79% | 5.9% | 14 | ✅ GAIN |
| GOOGL | +2.6% ($+256) | +21.7% 📉 | 36% | 4.6% | 14 | ✅ GAIN |
| TSLA | +32.4% ($+3,244) | +17.5% 📈 | 78% | 12.2% | 18 | ✅ GAIN |


## Portfolio Summary

| Metric | Strategy | Buy & Hold |
|--------|----------|------------|
| Total Invested | $80,000.00 | $80,000.00 |
| Total P&L | $+9,659.36 | $+11,580.93 |
| Portfolio Return | +12.07% | +14.48% |
| Final Value | $89,659.36 | $91,580.93 |

**Strategy vs Buy & Hold:** -2.40% (Underperformed ❌)


## Detailed Stock Analysis


### QQQ

| Metric | Value |
|--------|-------|
| Start Price | $594.63 |
| End Price | $611.75 |
| Directional Accuracy | 46.77% |
| Mean Squared Error | 151.8824 |
| Root Mean Squared Error | $12.32 |
| Mean Absolute Percentage Error | 1.56% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,259.32 |
| **Strategy Return** | **+2.59%** |
| Buy & Hold Return | +2.88% |
| Max Drawdown | 5.45% |
| Win Rate | 56.2% |
| Total Trades | 16 |
| Trading Days | 63 |
| MC 14-Day Prediction | $588.10 |

#### Trade History for QQQ

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 7 | $594.63 | $598.73 | 16.82 | $+68.94 | +0.69% | 📊 SIGNAL_SELL |
| 2 | Day 16 | Day 17 | $589.50 | $602.01 | 17.08 | $+213.68 | +2.12% | 📊 SIGNAL_SELL |
| 3 | Day 22 | Day 23 | $611.54 | $611.38 | 16.81 | $-2.69 | -0.03% | 📊 SIGNAL_SELL |
| 4 | Day 25 | Day 26 | $610.58 | $617.10 | 16.84 | $+109.77 | +1.07% | 📊 SIGNAL_SELL |
| 5 | Day 27 | Day 28 | $628.09 | $632.92 | 16.54 | $+79.90 | +0.77% | 📊 SIGNAL_SELL |
| 6 | Day 29 | Day 30 | $635.77 | $626.05 | 16.47 | $-160.07 | -1.53% | 📊 SIGNAL_SELL |
| 7 | Day 31 | Day 32 | $629.07 | $632.08 | 16.39 | $+49.33 | +0.48% | 📊 SIGNAL_SELL |
| 8 | Day 33 | Day 34 | $619.25 | $623.28 | 16.73 | $+67.41 | +0.65% | 📊 SIGNAL_SELL |
| 9 | Day 37 | Day 38 | $623.23 | $621.57 | 16.73 | $-27.77 | -0.27% | 📊 SIGNAL_SELL |
| 10 | Day 39 | Day 41 | $621.08 | $601.98 | 16.74 | $-319.76 | -3.08% |  ATR-STOP (3.1%) |
| 11 | Day 43 | Day 47 | $596.31 | $605.16 | 16.90 | $+149.58 | +1.48% | 📊 SIGNAL_SELL |
| 12 | Day 48 | Day 49 | $608.89 | $614.27 | 16.80 | $+90.37 | +0.88% | 📊 SIGNAL_SELL |
| 13 | Day 52 | Day 53 | $622.00 | $623.52 | 16.59 | $+25.22 | +0.24% | 📊 SIGNAL_SELL |
| 14 | Day 55 | Day 56 | $625.48 | $624.28 | 16.54 | $-19.84 | -0.19% | 📊 SIGNAL_SELL |
| 15 | Day 58 | Day 59 | $627.61 | $625.58 | 16.45 | $-33.39 | -0.32% | 📊 SIGNAL_SELL |
| 16 | Day 60 | Day 62 | $613.62 | $611.75 | 16.77 | $-31.36 | -0.30% | ⏰ END_OF_PERIOD |

**Trade Summary:**
- Total Trades: 16
- Winning Trades: 9 (56.2%)
- Losing Trades: 7 (43.8%)
- Total Gains: $854.20
- Total Losses: $-594.88

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 14
- 📉 Macro Filter Exits: 0

### SPY

| Metric | Value |
|--------|-------|
| Start Price | $660.43 |
| End Price | $678.87 |
| Directional Accuracy | 37.10% |
| Mean Squared Error | 2902.6205 |
| Root Mean Squared Error | $53.88 |
| Mean Absolute Percentage Error | 7.41% |
| Starting Capital | $10,000.00 |
| Final Capital | $9,953.15 |
| **Strategy Return** | **-0.47%** |
| Buy & Hold Return | +2.79% |
| Max Drawdown | 2.00% |
| Win Rate | 63.6% |
| Total Trades | 11 |
| Trading Days | 63 |
| MC 14-Day Prediction | $601.89 |

#### Trade History for SPY

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 25 | Day 26 | $671.76 | $677.25 | 14.89 | $+81.73 | +0.82% | 📊 SIGNAL_SELL |
| 2 | Day 27 | Day 28 | $685.24 | $687.06 | 14.71 | $+26.78 | +0.27% | 📊 SIGNAL_SELL |
| 3 | Day 29 | Day 30 | $687.39 | $679.83 | 14.71 | $-111.17 | -1.10% | 📊 SIGNAL_SELL |
| 4 | Day 31 | Day 32 | $682.06 | $683.34 | 14.66 | $+18.76 | +0.19% | 📊 SIGNAL_SELL |
| 5 | Day 33 | Day 34 | $675.24 | $677.58 | 14.83 | $+34.71 | +0.35% | 📊 SIGNAL_SELL |
| 6 | Day 37 | Day 38 | $681.44 | $683.00 | 14.75 | $+23.01 | +0.23% | 📊 SIGNAL_SELL |
| 7 | Day 39 | Day 40 | $683.38 | $672.04 | 14.74 | $-167.17 | -1.66% | 📊 SIGNAL_SELL |
| 8 | Day 48 | Day 49 | $675.02 | $679.68 | 14.68 | $+68.39 | +0.69% | 📊 SIGNAL_SELL |
| 9 | Day 55 | Day 56 | $685.69 | $683.63 | 14.55 | $-29.97 | -0.30% | 📊 SIGNAL_SELL |
| 10 | Day 58 | Day 59 | $687.57 | $689.17 | 14.46 | $+23.14 | +0.23% | 📊 SIGNAL_SELL |
| 11 | Day 60 | Day 61 | $681.76 | $680.73 | 14.62 | $-15.06 | -0.15% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 11
- Winning Trades: 7 (63.6%)
- Losing Trades: 4 (36.4%)
- Total Gains: $276.52
- Total Losses: $-323.37

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 11
- 📉 Macro Filter Exits: 0

### AAPL

| Metric | Value |
|--------|-------|
| Start Price | $237.65 |
| End Price | $274.61 |
| Directional Accuracy | 54.84% |
| Mean Squared Error | 540.3692 |
| Root Mean Squared Error | $23.25 |
| Mean Absolute Percentage Error | 7.57% |
| Starting Capital | $10,000.00 |
| Final Capital | $11,064.97 |
| **Strategy Return** | **+10.65%** |
| Buy & Hold Return | +15.55% |
| Max Drawdown | 2.00% |
| Win Rate | 64.7% |
| Total Trades | 17 |
| Trading Days | 63 |
| MC 14-Day Prediction | $257.18 |

#### Trade History for AAPL

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 5 | $237.65 | $256.62 | 42.08 | $+798.30 | +7.98% | 📊 SIGNAL_SELL |
| 2 | Day 27 | Day 28 | $268.55 | $268.74 | 40.21 | $+7.63 | +0.07% | 📊 SIGNAL_SELL |
| 3 | Day 29 | Day 30 | $269.44 | $271.14 | 40.11 | $+68.11 | +0.63% | 📊 SIGNAL_SELL |
| 4 | Day 31 | Day 32 | $270.11 | $268.79 | 40.26 | $-53.09 | -0.49% | 📊 SIGNAL_SELL |
| 5 | Day 33 | Day 34 | $269.78 | $269.88 | 40.11 | $+4.01 | +0.04% | 📊 SIGNAL_SELL |
| 6 | Day 35 | Day 36 | $269.51 | $268.21 | 40.17 | $-52.16 | -0.48% | 📊 SIGNAL_SELL |
| 7 | Day 37 | Day 38 | $269.43 | $275.25 | 39.98 | $+232.71 | +2.16% | 📊 SIGNAL_SELL |
| 8 | Day 39 | Day 40 | $273.47 | $272.95 | 40.24 | $-20.93 | -0.19% | 📊 SIGNAL_SELL |
| 9 | Day 41 | Day 42 | $272.41 | $267.46 | 40.32 | $-199.60 | -1.82% | 📊 SIGNAL_SELL |
| 10 | Day 43 | Day 44 | $267.44 | $268.56 | 40.33 | $+45.17 | +0.42% | 📊 SIGNAL_SELL |
| 11 | Day 45 | Day 46 | $266.25 | $271.49 | 40.68 | $+213.14 | +1.97% | 📊 SIGNAL_SELL |
| 12 | Day 47 | Day 48 | $275.92 | $276.97 | 40.02 | $+42.02 | +0.38% | 📊 SIGNAL_SELL |
| 13 | Day 51 | Day 52 | $283.10 | $286.19 | 39.16 | $+120.99 | +1.09% | 📊 SIGNAL_SELL |
| 14 | Day 53 | Day 54 | $284.15 | $280.70 | 39.44 | $-136.06 | -1.21% | 📊 SIGNAL_SELL |
| 15 | Day 55 | Day 56 | $278.78 | $277.89 | 39.71 | $-35.34 | -0.32% | 📊 SIGNAL_SELL |
| 16 | Day 59 | Day 60 | $278.03 | $278.28 | 39.69 | $+9.92 | +0.09% | 📊 SIGNAL_SELL |
| 17 | Day 61 | Day 62 | $274.11 | $274.61 | 40.29 | $+20.15 | +0.18% | ⏰ END_OF_PERIOD |

**Trade Summary:**
- Total Trades: 17
- Winning Trades: 11 (64.7%)
- Losing Trades: 6 (35.3%)
- Total Gains: $1,562.16
- Total Losses: $-497.18

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 16
- 📉 Macro Filter Exits: 0

### NVDA

| Metric | Value |
|--------|-------|
| Start Price | $176.23 |
| End Price | $177.72 |
| Directional Accuracy | 50.00% |
| Mean Squared Error | 143.8951 |
| Root Mean Squared Error | $12.00 |
| Mean Absolute Percentage Error | 5.83% |
| Starting Capital | $10,000.00 |
| Final Capital | $9,573.60 |
| **Strategy Return** | **-4.26%** |
| Buy & Hold Return | +0.85% |
| Max Drawdown | 11.69% |
| Win Rate | 50.0% |
| Total Trades | 12 |
| Trading Days | 63 |
| MC 14-Day Prediction | $167.59 |

#### Trade History for NVDA

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 8 | Day 9 | $186.57 | $187.23 | 53.60 | $+35.37 | +0.35% | 📊 SIGNAL_SELL |
| 2 | Day 10 | Day 11 | $188.88 | $187.61 | 53.13 | $-67.47 | -0.67% | 📊 SIGNAL_SELL |
| 3 | Day 15 | Day 16 | $192.56 | $183.15 | 51.77 | $-487.08 | -4.89% | 📊 SIGNAL_SELL |
| 4 | Day 17 | Day 18 | $188.31 | $180.02 | 50.35 | $-417.35 | -4.40% | 📊 SIGNAL_SELL |
| 5 | Day 26 | Day 27 | $186.25 | $191.48 | 48.66 | $+254.49 | +2.81% | 📊 SIGNAL_SELL |
| 6 | Day 28 | Day 29 | $201.02 | $207.03 | 46.35 | $+278.57 | +2.99% | 📊 SIGNAL_SELL |
| 7 | Day 30 | Day 31 | $202.88 | $202.48 | 47.30 | $-18.92 | -0.20% | 📊 SIGNAL_SELL |
| 8 | Day 32 | Day 33 | $206.87 | $198.68 | 46.30 | $-379.16 | -3.96% | 📊 SIGNAL_SELL |
| 9 | Day 34 | Day 35 | $195.20 | $188.07 | 47.12 | $-335.97 | -3.65% | 📊 SIGNAL_SELL |
| 10 | Day 36 | Day 37 | $188.14 | $199.04 | 47.11 | $+513.43 | +5.79% | 📊 SIGNAL_SELL |
| 11 | Day 38 | Day 39 | $193.15 | $193.79 | 48.54 | $+31.07 | +0.33% | 📊 SIGNAL_SELL |
| 12 | Day 40 | Day 41 | $186.85 | $190.16 | 50.35 | $+166.63 | +1.77% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 12
- Winning Trades: 6 (50.0%)
- Losing Trades: 6 (50.0%)
- Total Gains: $1,279.56
- Total Losses: $-1,705.96

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 12
- 📉 Macro Filter Exits: 0

### AMD

| Metric | Value |
|--------|-------|
| Start Price | $157.92 |
| End Price | $209.17 |
| Directional Accuracy | 50.00% |
| Mean Squared Error | 223.2862 |
| Root Mean Squared Error | $14.94 |
| Mean Absolute Percentage Error | 5.97% |
| Starting Capital | $10,000.00 |
| Final Capital | $13,793.63 |
| **Strategy Return** | **+37.94%** |
| Buy & Hold Return | +32.45% |
| Max Drawdown | 14.28% |
| Win Rate | 53.3% |
| Total Trades | 15 |
| Trading Days | 63 |
| MC 14-Day Prediction | $208.66 |

#### Trade History for AMD

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 1 | $157.92 | $157.39 | 63.32 | $-33.56 | -0.34% | 📉 MACRO_FILTER |
| 2 | Day 2 | Day 12 | $159.79 | $203.71 | 62.37 | $+2739.38 | +27.49% | 📊 SIGNAL_SELL |
| 3 | Day 14 | Day 15 | $235.56 | $232.89 | 53.94 | $-144.02 | -1.13% | 📊 SIGNAL_SELL |
| 4 | Day 16 | Day 17 | $214.90 | $216.42 | 58.45 | $+88.85 | +0.71% | 📊 SIGNAL_SELL |
| 5 | Day 18 | Day 19 | $218.09 | $238.60 | 58.01 | $+1189.72 | +9.40% | 📊 SIGNAL_SELL |
| 6 | Day 20 | Day 22 | $234.56 | $240.56 | 59.01 | $+354.03 | +2.56% | 📊 SIGNAL_SELL |
| 7 | Day 24 | Day 25 | $230.23 | $234.99 | 61.65 | $+293.47 | +2.07% | 📊 SIGNAL_SELL |
| 8 | Day 26 | Day 27 | $252.92 | $259.67 | 57.28 | $+386.66 | +2.67% | 📊 SIGNAL_SELL |
| 9 | Day 33 | Day 34 | $250.05 | $256.33 | 59.49 | $+373.57 | +2.51% | 📊 SIGNAL_SELL |
| 10 | Day 35 | Day 36 | $237.70 | $233.54 | 64.15 | $-266.86 | -1.75% | 📊 SIGNAL_SELL |
| 11 | Day 37 | Day 38 | $243.98 | $237.52 | 61.40 | $-396.67 | -2.65% | 📊 SIGNAL_SELL |
| 12 | Day 39 | Day 40 | $258.89 | $247.96 | 56.34 | $-615.74 | -4.22% | 📊 SIGNAL_SELL |
| 13 | Day 41 | Day 42 | $246.81 | $240.52 | 56.60 | $-356.00 | -2.55% | 📊 SIGNAL_SELL |
| 14 | Day 43 | Day 44 | $230.29 | $223.55 | 59.11 | $-398.41 | -2.93% | 📊 SIGNAL_SELL |
| 15 | Day 45 | Day 47 | $206.02 | $215.05 | 64.14 | $+579.20 | +4.38% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 15
- Winning Trades: 8 (53.3%)
- Losing Trades: 7 (46.7%)
- Total Gains: $6,004.88
- Total Losses: $-2,211.26

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 14
- 📉 Macro Filter Exits: 1

### INTC

| Metric | Value |
|--------|-------|
| Start Price | $30.57 |
| End Price | $37.31 |
| Directional Accuracy | 48.39% |
| Mean Squared Error | 3.2866 |
| Root Mean Squared Error | $1.81 |
| Mean Absolute Percentage Error | 3.80% |
| Starting Capital | $10,000.00 |
| Final Capital | $11,514.53 |
| **Strategy Return** | **+15.15%** |
| Buy & Hold Return | +22.05% |
| Max Drawdown | 5.92% |
| Win Rate | 78.6% |
| Total Trades | 14 |
| Trading Days | 63 |
| MC 14-Day Prediction | $34.19 |

#### Trade History for INTC

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 4 | $30.57 | $31.22 | 327.12 | $+212.63 | +2.13% | 📊 SIGNAL_SELL |
| 2 | Day 7 | Day 9 | $34.48 | $35.94 | 296.19 | $+432.44 | +4.23% | 📊 SIGNAL_SELL |
| 3 | Day 10 | Day 14 | $37.30 | $37.43 | 285.39 | $+37.10 | +0.35% | 📊 SIGNAL_SELL |
| 4 | Day 21 | Day 22 | $37.01 | $38.10 | 288.63 | $+314.61 | +2.95% | 📊 SIGNAL_SELL |
| 5 | Day 25 | Day 26 | $38.16 | $38.28 | 288.18 | $+34.58 | +0.31% | 📊 SIGNAL_SELL |
| 6 | Day 27 | Day 28 | $39.54 | $41.53 | 278.99 | $+555.19 | +5.03% | 📊 SIGNAL_SELL |
| 7 | Day 29 | Day 30 | $41.34 | $40.16 | 280.27 | $-330.72 | -2.85% | 📊 SIGNAL_SELL |
| 8 | Day 36 | Day 37 | $38.13 | $38.45 | 295.20 | $+94.46 | +0.84% | 📊 SIGNAL_SELL |
| 9 | Day 45 | Day 46 | $33.62 | $34.50 | 337.61 | $+297.09 | +2.62% | 📊 SIGNAL_SELL |
| 10 | Day 50 | Day 51 | $40.56 | $40.01 | 287.16 | $-157.94 | -1.36% | 📊 SIGNAL_SELL |
| 11 | Day 52 | Day 53 | $43.47 | $43.76 | 264.31 | $+76.65 | +0.67% | 📊 SIGNAL_SELL |
| 12 | Day 54 | Day 55 | $40.50 | $41.41 | 285.58 | $+259.88 | +2.25% | 📊 SIGNAL_SELL |
| 13 | Day 56 | Day 57 | $40.30 | $40.50 | 293.45 | $+58.69 | +0.50% | 📊 SIGNAL_SELL |
| 14 | Day 58 | Day 59 | $40.78 | $39.51 | 291.43 | $-370.12 | -3.11% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 14
- Winning Trades: 11 (78.6%)
- Losing Trades: 3 (21.4%)
- Total Gains: $2,373.32
- Total Losses: $-858.79

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 14
- 📉 Macro Filter Exits: 0

### GOOGL

| Metric | Value |
|--------|-------|
| Start Price | $251.87 |
| End Price | $306.57 |
| Directional Accuracy | 48.39% |
| Mean Squared Error | 375.3716 |
| Root Mean Squared Error | $19.37 |
| Mean Absolute Percentage Error | 6.01% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,256.22 |
| **Strategy Return** | **+2.56%** |
| Buy & Hold Return | +21.72% |
| Max Drawdown | 4.56% |
| Win Rate | 35.7% |
| Total Trades | 14 |
| Trading Days | 63 |
| MC 14-Day Prediction | $306.10 |

#### Trade History for GOOGL

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 1 | Day 9 | $254.55 | $244.74 | 39.28 | $-385.52 | -3.86% | 📊 SIGNAL_SELL |
| 2 | Day 16 | Day 17 | $236.42 | $243.99 | 40.67 | $+308.06 | +3.20% | 📊 SIGNAL_SELL |
| 3 | Day 30 | Day 31 | $281.30 | $281.01 | 35.27 | $-10.22 | -0.10% | 📊 SIGNAL_SELL |
| 4 | Day 32 | Day 33 | $283.53 | $277.36 | 34.96 | $-215.91 | -2.18% | 📊 SIGNAL_SELL |
| 5 | Day 34 | Day 35 | $284.12 | $284.56 | 34.13 | $+15.01 | +0.15% | 📊 SIGNAL_SELL |
| 6 | Day 36 | Day 37 | $278.65 | $289.91 | 34.85 | $+392.53 | +4.04% | 📊 SIGNAL_SELL |
| 7 | Day 40 | Day 41 | $278.39 | $276.23 | 36.29 | $-78.35 | -0.78% | 📊 SIGNAL_SELL |
| 8 | Day 42 | Day 43 | $284.83 | $284.09 | 35.20 | $-26.03 | -0.26% | 📊 SIGNAL_SELL |
| 9 | Day 44 | Day 45 | $292.62 | $289.26 | 34.17 | $-114.74 | -1.15% | 📊 SIGNAL_SELL |
| 10 | Day 46 | Day 47 | $299.46 | $318.37 | 33.01 | $+624.11 | +6.31% | 📊 SIGNAL_SELL |
| 11 | Day 48 | Day 49 | $323.23 | $319.74 | 32.51 | $-113.39 | -1.08% | 📊 SIGNAL_SELL |
| 12 | Day 51 | Day 52 | $314.68 | $315.60 | 33.03 | $+30.37 | +0.29% | 📊 SIGNAL_SELL |
| 13 | Day 53 | Day 54 | $319.42 | $317.41 | 32.64 | $-65.56 | -0.63% | 📊 SIGNAL_SELL |
| 14 | Day 59 | Day 60 | $312.43 | $309.29 | 33.16 | $-104.12 | -1.01% | 📊 SIGNAL_SELL |

**Trade Summary:**
- Total Trades: 14
- Winning Trades: 5 (35.7%)
- Losing Trades: 9 (64.3%)
- Total Gains: $1,370.07
- Total Losses: $-1,113.86

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 14
- 📉 Macro Filter Exits: 0

### TSLA

| Metric | Value |
|--------|-------|
| Start Price | $416.85 |
| End Price | $489.88 |
| Directional Accuracy | 37.70% |
| Mean Squared Error | 309.4862 |
| Root Mean Squared Error | $17.59 |
| Mean Absolute Percentage Error | 3.38% |
| Starting Capital | $10,000.00 |
| Final Capital | $13,243.94 |
| **Strategy Return** | **+32.44%** |
| Buy & Hold Return | +17.52% |
| Max Drawdown | 12.23% |
| Win Rate | 77.8% |
| Total Trades | 18 |
| Trading Days | 62 |
| MC 14-Day Prediction | $412.07 |

#### Trade History for TSLA

| # | Entry Day | Exit Day | Entry $ | Exit $ | Shares | P&L | Return | Exit Reason |
|---|-----------|----------|---------|--------|--------|-----|--------|-------------|
| 1 | Day 0 | Day 4 | $416.85 | $442.79 | 23.99 | $+622.29 | +6.22% | 📊 SIGNAL_SELL |
| 2 | Day 6 | Day 7 | $440.40 | $443.21 | 24.12 | $+67.78 | +0.64% | 📊 SIGNAL_SELL |
| 3 | Day 8 | Day 9 | $444.72 | $459.46 | 24.04 | $+354.32 | +3.31% | 📊 SIGNAL_SELL |
| 4 | Day 12 | Day 13 | $453.25 | $433.09 | 24.37 | $-491.24 | -4.45% | 📊 SIGNAL_SELL |
| 5 | Day 14 | Day 15 | $438.69 | $435.54 | 24.06 | $-75.78 | -0.72% | 📊 SIGNAL_SELL |
| 6 | Day 16 | Day 17 | $413.49 | $435.90 | 25.34 | $+567.84 | +5.42% | 📊 SIGNAL_SELL |
| 7 | Day 19 | Day 20 | $435.15 | $428.75 | 25.38 | $-162.45 | -1.47% | 📊 SIGNAL_SELL |
| 8 | Day 21 | Day 22 | $439.31 | $447.43 | 24.77 | $+201.15 | +1.85% | 📊 SIGNAL_SELL |
| 9 | Day 24 | Day 25 | $438.97 | $448.98 | 25.25 | $+252.75 | +2.28% | 📊 SIGNAL_SELL |
| 10 | Day 26 | Day 27 | $433.72 | $452.42 | 26.14 | $+488.78 | +4.31% | 📊 SIGNAL_SELL |
| 11 | Day 28 | Day 29 | $460.55 | $461.51 | 25.68 | $+24.65 | +0.21% | 📊 SIGNAL_SELL |
| 12 | Day 30 | Day 32 | $440.10 | $468.37 | 26.93 | $+761.20 | +6.42% | 📊 SIGNAL_SELL |
| 13 | Day 33 | Day 34 | $444.26 | $462.07 | 28.39 | $+505.58 | +4.01% | 📊 SIGNAL_SELL |
| 14 | Day 35 | Day 40 | $445.91 | $404.62 | 29.42 | $-1214.46 | -9.26% |  ATR-STOP (9.3%) |
| 15 | Day 41 | Day 47 | $404.35 | $417.78 | 29.44 | $+395.32 | +3.32% | 📊 SIGNAL_SELL |
| 16 | Day 53 | Day 54 | $446.74 | $454.53 | 27.53 | $+214.44 | +1.74% | 📊 SIGNAL_SELL |
| 17 | Day 56 | Day 58 | $439.58 | $451.45 | 28.46 | $+337.87 | +2.70% | 📊 SIGNAL_SELL |
| 18 | Day 60 | Day 61 | $475.31 | $489.88 | 27.04 | $+393.90 | +3.07% | ⏰ END_OF_PERIOD |

**Trade Summary:**
- Total Trades: 18
- Winning Trades: 14 (77.8%)
- Losing Trades: 4 (22.2%)
- Total Gains: $5,187.86
- Total Losses: $-1,943.92

**Exit Reason Breakdown:**
- 🛑 Stop-Loss Hits: 0
- 📊 Signal Sells: 16
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