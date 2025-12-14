# Monte Carlo Stock Analysis Report

**Generated:** 2025-12-12 12:10:31

**Analysis Period:** 2025-10-13 to 2025-12-12 (Past 2 Months)

**Training Period:** 2020-01-01 to 2025-12-12

**Monte Carlo Simulations:** 10

**Starting Capital per Stock:** $10,000.00

**Training Epochs:** 100 (with early stopping, patience=10)

**Technical Indicators:** Enabled (MA, RSI, MACD, Bollinger Bands, Momentum, Volatility)


## Summary Results

| Stock | Dir. Accuracy | MSE | RMSE | MAPE | Investment Result | Status |
|-------|--------------|-----|------|------|-------------------|--------|
| QQQ | 47.6% | 246.30 | $15.69 | 2.25% | $-92.31 (-0.9%) | ❌ LOSS |
| SPY | 54.8% | 601.83 | $24.53 | 3.35% | $-171.69 (-1.7%) | ❌ LOSS |
| AAPL | 52.4% | 101.47 | $10.07 | 3.41% | $+639.72 (+6.4%) | ✅ GAIN |
| NVDA | 45.2% | 100.26 | $10.01 | 3.73% | $-160.17 (-1.6%) | ❌ LOSS |
| AMD | 47.6% | 461.48 | $21.48 | 8.14% | $-2,245.03 (-22.5%) | ❌ LOSS |
| INTC | 50.0% | 2.96 | $1.72 | 3.58% | $+412.14 (+4.1%) | ✅ GAIN |
| GOOGL | 33.3% | 422.41 | $20.55 | 5.96% | $-1,215.48 (-12.2%) | ❌ LOSS |
| TSLA | 45.2% | 476.81 | $21.84 | 4.23% | $-1,534.66 (-15.3%) | ❌ LOSS |


## Portfolio Summary

- **Total Invested:** $80,000.00
- **Total P&L:** $-4,367.49
- **Portfolio Return:** -5.46%
- **Final Portfolio Value:** $75,632.51


## Detailed Stock Analysis


### QQQ

| Metric | Value |
|--------|-------|
| Start Price | $602.01 |
| End Price | $625.58 |
| Directional Accuracy | 47.62% |
| Mean Squared Error | 246.2987 |
| Root Mean Squared Error | $15.69 |
| Mean Absolute Percentage Error | 2.25% |
| Starting Capital | $10,000.00 |
| Final Capital | $9,907.69 |
| Total P&L | $-92.31 |
| Return | -0.92% |
| Trading Days | 43 |
| MC 14-Day Prediction | $603.06 |

#### Trade History for QQQ

| # | Type | Entry Day | Exit Day | Entry Price | Exit Price | Shares | P&L | Return |
|---|------|-----------|----------|-------------|------------|--------|-----|--------|
| 1 | SHORT | Day 0 | Day 5 | $602.01 | $611.54 | 16.61 | $-158.30 | -1.58% |
| 2 | LONG | Day 5 | Day 13 | $611.54 | $626.05 | 16.09 | $+233.51 | +2.37% |
| 3 | SHORT | Day 13 | Day 14 | $626.05 | $629.07 | 16.09 | $-48.60 | -0.48% |
| 4 | LONG | Day 14 | Day 16 | $629.07 | $619.25 | 15.94 | $-156.52 | -1.56% |
| 5 | SHORT | Day 16 | Day 20 | $619.25 | $623.23 | 15.94 | $-63.44 | -0.64% |
| 6 | LONG | Day 20 | Day 23 | $623.23 | $608.40 | 15.74 | $-233.35 | -2.38% |
| 7 | SHORT | Day 23 | Day 30 | $608.40 | $605.16 | 15.74 | $+50.98 | +0.53% |
| 8 | LONG | Day 30 | Day 39 | $605.16 | $624.28 | 15.90 | $+304.08 | +3.16% |
| 9 | SHORT | Day 39 | Day 42 | $624.28 | $625.58 | 15.90 | $-20.67 | -0.21% |

**Trade Summary:**
- Total Trades: 9
- Winning Trades: 3 (33.3%)
- Losing Trades: 6 (66.7%)
- Total Gains: $588.58
- Total Losses: $-680.89

### SPY

| Metric | Value |
|--------|-------|
| Start Price | $663.04 |
| End Price | $689.17 |
| Directional Accuracy | 54.76% |
| Mean Squared Error | 601.8264 |
| Root Mean Squared Error | $24.53 |
| Mean Absolute Percentage Error | 3.35% |
| Starting Capital | $10,000.00 |
| Final Capital | $9,828.31 |
| Total P&L | $-171.69 |
| Return | -1.72% |
| Trading Days | 43 |
| MC 14-Day Prediction | $654.58 |

#### Trade History for SPY

| # | Type | Entry Day | Exit Day | Entry Price | Exit Price | Shares | P&L | Return |
|---|------|-----------|----------|-------------|------------|--------|-----|--------|
| 1 | SHORT | Day 0 | Day 6 | $663.04 | $671.29 | 15.08 | $-124.43 | -1.24% |
| 2 | LONG | Day 6 | Day 16 | $671.29 | $675.24 | 14.71 | $+58.11 | +0.59% |
| 3 | SHORT | Day 16 | Day 21 | $675.24 | $683.00 | 14.71 | $-114.16 | -1.15% |
| 4 | LONG | Day 21 | Day 23 | $683.00 | $672.04 | 14.38 | $-157.57 | -1.60% |
| 5 | SHORT | Day 23 | Day 30 | $672.04 | $668.73 | 14.38 | $+47.59 | +0.49% |
| 6 | LONG | Day 30 | Day 40 | $668.73 | $683.04 | 14.52 | $+207.77 | +2.14% |
| 7 | SHORT | Day 40 | Day 42 | $683.04 | $689.17 | 14.52 | $-89.00 | -0.90% |

**Trade Summary:**
- Total Trades: 7
- Winning Trades: 3 (42.9%)
- Losing Trades: 4 (57.1%)
- Total Gains: $313.47
- Total Losses: $-485.16

### AAPL

| Metric | Value |
|--------|-------|
| Start Price | $247.42 |
| End Price | $278.03 |
| Directional Accuracy | 52.38% |
| Mean Squared Error | 101.4749 |
| Root Mean Squared Error | $10.07 |
| Mean Absolute Percentage Error | 3.41% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,639.72 |
| Total P&L | $+639.72 |
| Return | +6.40% |
| Trading Days | 43 |
| MC 14-Day Prediction | $274.14 |

#### Trade History for AAPL

| # | Type | Entry Day | Exit Day | Entry Price | Exit Price | Shares | P&L | Return |
|---|------|-----------|----------|-------------|------------|--------|-----|--------|
| 1 | SHORT | Day 0 | Day 4 | $247.42 | $252.05 | 40.42 | $-186.95 | -1.87% |
| 2 | LONG | Day 4 | Day 7 | $252.05 | $258.20 | 38.93 | $+239.60 | +2.44% |
| 3 | SHORT | Day 7 | Day 9 | $258.20 | $262.57 | 38.93 | $-169.98 | -1.69% |
| 4 | LONG | Day 9 | Day 12 | $262.57 | $269.44 | 37.64 | $+258.70 | +2.62% |
| 5 | SHORT | Day 12 | Day 14 | $269.44 | $270.11 | 37.64 | $-25.19 | -0.25% |
| 6 | LONG | Day 14 | Day 15 | $270.11 | $268.79 | 37.45 | $-49.39 | -0.49% |
| 7 | SHORT | Day 15 | Day 17 | $268.79 | $269.88 | 37.45 | $-40.78 | -0.41% |
| 8 | LONG | Day 17 | Day 18 | $269.88 | $269.51 | 37.15 | $-13.73 | -0.14% |
| 9 | SHORT | Day 18 | Day 20 | $269.51 | $269.43 | 37.15 | $+2.93 | +0.03% |
| 10 | LONG | Day 20 | Day 22 | $269.43 | $273.47 | 37.17 | $+150.17 | +1.50% |
| 11 | SHORT | Day 22 | Day 27 | $273.47 | $268.56 | 37.17 | $+182.51 | +1.80% |
| 12 | LONG | Day 27 | Day 28 | $268.56 | $266.25 | 38.53 | $-89.01 | -0.86% |
| 13 | SHORT | Day 28 | Day 29 | $266.25 | $271.49 | 38.53 | $-201.90 | -1.97% |
| 14 | LONG | Day 29 | Day 36 | $271.49 | $284.15 | 37.04 | $+468.97 | +4.66% |
| 15 | SHORT | Day 36 | Day 39 | $284.15 | $277.89 | 37.04 | $+231.89 | +2.20% |
| 16 | LONG | Day 39 | Day 40 | $277.89 | $277.18 | 38.71 | $-27.49 | -0.26% |
| 17 | SHORT | Day 40 | Day 41 | $277.18 | $278.78 | 38.71 | $-61.94 | -0.58% |
| 18 | LONG | Day 41 | Day 42 | $278.78 | $278.03 | 38.27 | $-28.70 | -0.27% |

**Trade Summary:**
- Total Trades: 18
- Winning Trades: 7 (38.9%)
- Losing Trades: 11 (61.1%)
- Total Gains: $1,534.78
- Total Losses: $-895.06

### NVDA

| Metric | Value |
|--------|-------|
| Start Price | $188.31 |
| End Price | $180.93 |
| Directional Accuracy | 45.24% |
| Mean Squared Error | 100.2607 |
| Root Mean Squared Error | $10.01 |
| Mean Absolute Percentage Error | 3.73% |
| Starting Capital | $10,000.00 |
| Final Capital | $9,839.83 |
| Total P&L | $-160.17 |
| Return | -1.60% |
| Trading Days | 43 |
| MC 14-Day Prediction | $173.43 |

#### Trade History for NVDA

| # | Type | Entry Day | Exit Day | Entry Price | Exit Price | Shares | P&L | Return |
|---|------|-----------|----------|-------------|------------|--------|-----|--------|
| 1 | SHORT | Day 0 | Day 9 | $188.31 | $186.25 | 53.10 | $+109.39 | +1.09% |
| 2 | LONG | Day 9 | Day 16 | $186.25 | $198.68 | 54.28 | $+674.65 | +6.67% |
| 3 | SHORT | Day 16 | Day 21 | $198.68 | $193.15 | 54.28 | $+300.14 | +2.78% |
| 4 | LONG | Day 21 | Day 25 | $193.15 | $186.59 | 57.39 | $-376.44 | -3.40% |
| 5 | SHORT | Day 25 | Day 27 | $186.59 | $186.51 | 57.39 | $+4.59 | +0.04% |
| 6 | LONG | Day 27 | Day 29 | $186.51 | $178.87 | 57.44 | $-438.79 | -4.10% |
| 7 | SHORT | Day 29 | Day 30 | $178.87 | $182.54 | 57.44 | $-210.78 | -2.05% |
| 8 | LONG | Day 30 | Day 31 | $182.54 | $177.81 | 55.13 | $-260.73 | -2.59% |
| 9 | SHORT | Day 31 | Day 32 | $177.81 | $180.25 | 55.13 | $-134.50 | -1.37% |
| 10 | LONG | Day 32 | Day 34 | $180.25 | $179.91 | 53.63 | $-18.23 | -0.19% |
| 11 | SHORT | Day 34 | Day 35 | $179.91 | $181.45 | 53.63 | $-82.59 | -0.86% |
| 12 | LONG | Day 35 | Day 41 | $181.45 | $183.78 | 52.72 | $+122.85 | +1.28% |
| 13 | SHORT | Day 41 | Day 42 | $183.78 | $180.93 | 52.72 | $+150.26 | +1.55% |

**Trade Summary:**
- Total Trades: 13
- Winning Trades: 6 (46.2%)
- Losing Trades: 7 (53.8%)
- Total Gains: $1,361.89
- Total Losses: $-1,522.06

### AMD

| Metric | Value |
|--------|-------|
| Start Price | $216.42 |
| End Price | $221.43 |
| Directional Accuracy | 47.62% |
| Mean Squared Error | 461.4761 |
| Root Mean Squared Error | $21.48 |
| Mean Absolute Percentage Error | 8.14% |
| Starting Capital | $10,000.00 |
| Final Capital | $7,754.97 |
| Total P&L | $-2,245.03 |
| Return | -22.45% |
| Trading Days | 43 |
| MC 14-Day Prediction | $194.97 |

#### Trade History for AMD

| # | Type | Entry Day | Exit Day | Entry Price | Exit Price | Shares | P&L | Return |
|---|------|-----------|----------|-------------|------------|--------|-----|--------|
| 1 | SHORT | Day 0 | Day 2 | $216.42 | $238.60 | 46.21 | $-1024.86 | -10.25% |
| 2 | LONG | Day 2 | Day 3 | $238.60 | $234.56 | 37.62 | $-151.97 | -1.69% |
| 3 | SHORT | Day 3 | Day 5 | $234.56 | $240.56 | 37.62 | $-225.70 | -2.56% |
| 4 | LONG | Day 5 | Day 7 | $240.56 | $230.23 | 35.74 | $-369.19 | -4.29% |
| 5 | SHORT | Day 7 | Day 8 | $230.23 | $234.99 | 35.74 | $-170.12 | -2.07% |
| 6 | LONG | Day 8 | Day 11 | $234.99 | $258.01 | 34.29 | $+789.39 | +9.80% |
| 7 | SHORT | Day 11 | Day 12 | $258.01 | $264.33 | 34.29 | $-216.72 | -2.45% |
| 8 | LONG | Day 12 | Day 13 | $264.33 | $254.84 | 32.65 | $-309.86 | -3.59% |
| 9 | SHORT | Day 13 | Day 14 | $254.84 | $256.12 | 32.65 | $-41.79 | -0.50% |
| 10 | LONG | Day 14 | Day 16 | $256.12 | $250.05 | 32.33 | $-196.21 | -2.37% |
| 11 | SHORT | Day 16 | Day 17 | $250.05 | $256.33 | 32.33 | $-203.00 | -2.51% |
| 12 | LONG | Day 17 | Day 18 | $256.33 | $237.70 | 30.74 | $-572.71 | -7.27% |
| 13 | SHORT | Day 18 | Day 20 | $237.70 | $243.98 | 30.74 | $-193.06 | -2.64% |
| 14 | LONG | Day 20 | Day 23 | $243.98 | $247.96 | 29.16 | $+116.05 | +1.63% |
| 15 | SHORT | Day 23 | Day 30 | $247.96 | $215.05 | 29.16 | $+959.62 | +13.27% |
| 16 | LONG | Day 30 | Day 31 | $215.05 | $206.13 | 38.08 | $-339.71 | -4.15% |
| 17 | SHORT | Day 31 | Day 32 | $206.13 | $214.24 | 38.08 | $-308.86 | -3.93% |
| 18 | LONG | Day 32 | Day 35 | $214.24 | $215.24 | 35.20 | $+35.20 | +0.47% |
| 19 | SHORT | Day 35 | Day 37 | $215.24 | $215.98 | 35.20 | $-26.05 | -0.34% |
| 20 | LONG | Day 37 | Day 40 | $215.98 | $221.62 | 34.96 | $+197.17 | +2.61% |
| 21 | SHORT | Day 40 | Day 41 | $221.62 | $221.42 | 34.96 | $+6.99 | +0.09% |
| 22 | LONG | Day 41 | Day 42 | $221.42 | $221.43 | 35.02 | $+0.35 | +0.00% |

**Trade Summary:**
- Total Trades: 22
- Winning Trades: 7 (31.8%)
- Losing Trades: 15 (68.2%)
- Total Gains: $2,104.78
- Total Losses: $-4,349.81

### INTC

| Metric | Value |
|--------|-------|
| Start Price | $37.22 |
| End Price | $39.51 |
| Directional Accuracy | 50.00% |
| Mean Squared Error | 2.9600 |
| Root Mean Squared Error | $1.72 |
| Mean Absolute Percentage Error | 3.58% |
| Starting Capital | $10,000.00 |
| Final Capital | $10,412.14 |
| Total P&L | $+412.14 |
| Return | +4.12% |
| Trading Days | 43 |
| MC 14-Day Prediction | $39.09 |

#### Trade History for INTC

| # | Type | Entry Day | Exit Day | Entry Price | Exit Price | Shares | P&L | Return |
|---|------|-----------|----------|-------------|------------|--------|-----|--------|
| 1 | SHORT | Day 0 | Day 2 | $37.22 | $37.15 | 268.67 | $+18.81 | +0.19% |
| 2 | LONG | Day 2 | Day 7 | $37.15 | $36.92 | 269.69 | $-62.03 | -0.62% |
| 3 | SHORT | Day 7 | Day 8 | $36.92 | $38.16 | 269.69 | $-334.41 | -3.36% |
| 4 | LONG | Day 8 | Day 12 | $38.16 | $41.34 | 252.16 | $+801.86 | +8.33% |
| 5 | SHORT | Day 12 | Day 17 | $41.34 | $38.38 | 252.16 | $+746.39 | +7.16% |
| 6 | LONG | Day 17 | Day 18 | $38.38 | $37.24 | 291.05 | $-331.80 | -2.97% |
| 7 | SHORT | Day 18 | Day 19 | $37.24 | $38.13 | 291.05 | $-259.04 | -2.39% |
| 8 | LONG | Day 19 | Day 21 | $38.13 | $37.88 | 277.47 | $-69.37 | -0.66% |
| 9 | SHORT | Day 21 | Day 27 | $37.88 | $35.11 | 277.47 | $+768.58 | +7.31% |
| 10 | LONG | Day 27 | Day 28 | $35.11 | $33.62 | 321.25 | $-478.66 | -4.24% |
| 11 | SHORT | Day 28 | Day 30 | $33.62 | $35.79 | 321.25 | $-697.11 | -6.45% |
| 12 | LONG | Day 30 | Day 37 | $35.79 | $40.50 | 282.29 | $+1329.60 | +13.16% |
| 13 | SHORT | Day 37 | Day 38 | $40.50 | $41.41 | 282.29 | $-256.89 | -2.25% |
| 14 | LONG | Day 38 | Day 39 | $41.41 | $40.30 | 269.89 | $-299.57 | -2.68% |
| 15 | SHORT | Day 39 | Day 41 | $40.30 | $40.78 | 269.89 | $-129.54 | -1.19% |
| 16 | LONG | Day 41 | Day 42 | $40.78 | $39.51 | 263.53 | $-334.69 | -3.11% |

**Trade Summary:**
- Total Trades: 16
- Winning Trades: 5 (31.2%)
- Losing Trades: 11 (68.8%)
- Total Gains: $3,665.24
- Total Losses: $-3,253.10

### GOOGL

| Metric | Value |
|--------|-------|
| Start Price | $243.99 |
| End Price | $312.43 |
| Directional Accuracy | 33.33% |
| Mean Squared Error | 422.4111 |
| Root Mean Squared Error | $20.55 |
| Mean Absolute Percentage Error | 5.96% |
| Starting Capital | $10,000.00 |
| Final Capital | $8,784.52 |
| Total P&L | $-1,215.48 |
| Return | -12.15% |
| Trading Days | 43 |
| MC 14-Day Prediction | $320.67 |

#### Trade History for GOOGL

| # | Type | Entry Day | Exit Day | Entry Price | Exit Price | Shares | P&L | Return |
|---|------|-----------|----------|-------------|------------|--------|-----|--------|
| 1 | SHORT | Day 0 | Day 9 | $243.99 | $259.75 | 40.99 | $-645.92 | -6.46% |
| 2 | LONG | Day 9 | Day 16 | $259.75 | $277.36 | 36.01 | $+634.11 | +6.78% |
| 3 | SHORT | Day 16 | Day 17 | $277.36 | $284.12 | 36.01 | $-243.64 | -2.44% |
| 4 | LONG | Day 17 | Day 19 | $284.12 | $278.65 | 34.30 | $-187.82 | -1.93% |
| 5 | SHORT | Day 19 | Day 20 | $278.65 | $289.91 | 34.30 | $-386.27 | -4.04% |
| 6 | LONG | Day 20 | Day 23 | $289.91 | $278.39 | 31.63 | $-364.48 | -3.97% |
| 7 | SHORT | Day 23 | Day 25 | $278.39 | $284.83 | 31.63 | $-203.89 | -2.32% |
| 8 | LONG | Day 25 | Day 34 | $284.83 | $314.68 | 30.20 | $+901.50 | +10.48% |
| 9 | SHORT | Day 34 | Day 36 | $314.68 | $319.42 | 30.20 | $-143.06 | -1.51% |
| 10 | LONG | Day 36 | Day 39 | $319.42 | $313.72 | 29.30 | $-167.07 | -1.78% |
| 11 | SHORT | Day 39 | Day 41 | $313.72 | $320.21 | 29.30 | $-190.19 | -2.07% |
| 12 | LONG | Day 41 | Day 42 | $320.21 | $312.43 | 28.12 | $-218.75 | -2.43% |

**Trade Summary:**
- Total Trades: 12
- Winning Trades: 2 (16.7%)
- Losing Trades: 10 (83.3%)
- Total Gains: $1,535.61
- Total Losses: $-2,751.09

### TSLA

| Metric | Value |
|--------|-------|
| Start Price | $435.90 |
| End Price | $446.89 |
| Directional Accuracy | 45.24% |
| Mean Squared Error | 476.8080 |
| Root Mean Squared Error | $21.84 |
| Mean Absolute Percentage Error | 4.23% |
| Starting Capital | $10,000.00 |
| Final Capital | $8,465.34 |
| Total P&L | $-1,534.66 |
| Return | -15.35% |
| Trading Days | 43 |
| MC 14-Day Prediction | $419.73 |

#### Trade History for TSLA

| # | Type | Entry Day | Exit Day | Entry Price | Exit Price | Shares | P&L | Return |
|---|------|-----------|----------|-------------|------------|--------|-----|--------|
| 1 | SHORT | Day 0 | Day 2 | $435.90 | $435.15 | 22.94 | $+17.21 | +0.17% |
| 2 | LONG | Day 2 | Day 3 | $435.15 | $428.75 | 23.02 | $-147.33 | -1.47% |
| 3 | SHORT | Day 3 | Day 4 | $428.75 | $439.31 | 23.02 | $-243.09 | -2.46% |
| 4 | LONG | Day 4 | Day 6 | $439.31 | $442.60 | 21.91 | $+72.10 | +0.75% |
| 5 | SHORT | Day 6 | Day 8 | $442.60 | $448.98 | 21.91 | $-139.81 | -1.44% |
| 6 | LONG | Day 8 | Day 9 | $448.98 | $433.72 | 21.29 | $-324.90 | -3.40% |
| 7 | SHORT | Day 9 | Day 10 | $433.72 | $452.42 | 21.29 | $-398.14 | -4.31% |
| 8 | LONG | Day 10 | Day 12 | $452.42 | $461.51 | 19.53 | $+177.53 | +2.01% |
| 9 | SHORT | Day 12 | Day 14 | $461.51 | $456.56 | 19.53 | $+96.68 | +1.07% |
| 10 | LONG | Day 14 | Day 16 | $456.56 | $444.26 | 19.95 | $-245.44 | -2.69% |
| 11 | SHORT | Day 16 | Day 17 | $444.26 | $462.07 | 19.95 | $-355.38 | -4.01% |
| 12 | LONG | Day 17 | Day 18 | $462.07 | $445.91 | 18.42 | $-297.60 | -3.50% |
| 13 | SHORT | Day 18 | Day 20 | $445.91 | $445.23 | 18.42 | $+12.52 | +0.15% |
| 14 | LONG | Day 20 | Day 21 | $445.23 | $439.62 | 18.47 | $-103.63 | -1.26% |
| 15 | SHORT | Day 21 | Day 25 | $439.62 | $408.92 | 18.47 | $+567.09 | +6.98% |
| 16 | LONG | Day 25 | Day 26 | $408.92 | $401.25 | 21.25 | $-162.96 | -1.88% |
| 17 | SHORT | Day 26 | Day 27 | $401.25 | $403.99 | 21.25 | $-58.21 | -0.68% |
| 18 | LONG | Day 27 | Day 28 | $403.99 | $395.23 | 20.96 | $-183.59 | -2.17% |
| 19 | SHORT | Day 28 | Day 30 | $395.23 | $417.78 | 20.96 | $-472.59 | -5.71% |
| 20 | LONG | Day 30 | Day 31 | $417.78 | $419.40 | 18.70 | $+30.29 | +0.39% |
| 21 | SHORT | Day 31 | Day 32 | $419.40 | $426.58 | 18.70 | $-134.23 | -1.71% |
| 22 | LONG | Day 32 | Day 34 | $426.58 | $430.14 | 18.07 | $+64.31 | +0.83% |
| 23 | SHORT | Day 34 | Day 35 | $430.14 | $429.24 | 18.07 | $+16.26 | +0.21% |
| 24 | LONG | Day 35 | Day 38 | $429.24 | $455.00 | 18.14 | $+467.33 | +6.00% |
| 25 | SHORT | Day 38 | Day 40 | $455.00 | $445.17 | 18.14 | $+178.33 | +2.16% |
| 26 | LONG | Day 40 | Day 42 | $445.17 | $446.89 | 18.94 | $+32.58 | +0.39% |

**Trade Summary:**
- Total Trades: 26
- Winning Trades: 12 (46.2%)
- Losing Trades: 14 (53.8%)
- Total Gains: $1,732.23
- Total Losses: $-3,266.89


## Metrics Explanation

- **Directional Accuracy:** Percentage of correct up/down movement predictions
- **MSE (Mean Squared Error):** Average of squared differences between predicted and actual prices
- **RMSE (Root Mean Squared Error):** Square root of MSE, in dollar terms
- **MAPE (Mean Absolute Percentage Error):** Average percentage error between predictions and actuals
- **Investment Strategy:** Long when model predicts price increase, Short when model predicts decrease