import React, { useEffect, useState, useCallback, useRef } from 'react';
import { createChart, ColorType, CandlestickSeries, LineSeries, HistogramSeries } from 'lightweight-charts';
import '../styles/Dashboard.css';
import { useParams, useNavigate } from 'react-router-dom';
import { fetchDashboardData as apiFetchDashboardData, fetchComparisonData as apiFetchComparisonData, searchSymbols, addToWatchlist, removeFromWatchlist, fetchWatchlist } from '../utils/api';

const Dashboard = () => {
  const { symbol } = useParams();
  const navigate = useNavigate();
  const [dashboardData, setDashboardData] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);
  const [period, setPeriod] = useState('1y');
  const [showSP500, setShowSP500] = useState(false);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const chartContainerRef = useRef(null);
  const chartInstance = useRef(null);
  const searchTimerRef = useRef(null);
  const pollingRef = useRef(null);
  const [isPolling, setIsPolling] = useState(false);
  const [isWatchlisted, setIsWatchlisted] = useState(false);

  const fetchAllData = useCallback(async (sym, per) => {
    setError(null);
    setDashboardData(null);
    setComparisonData(null);
    try {
      const [dashData, compData] = await Promise.all([
        apiFetchDashboardData(sym),
        apiFetchComparisonData(sym, per)
      ]);
      setDashboardData(dashData);
      setComparisonData(compData);
    } catch (err) {
      setError('Failed to load dashboard data. Please try again.');
    }
  }, []);

  useEffect(() => {
    if (symbol) {
      fetchAllData(symbol, period);
    }
  }, [symbol, period, fetchAllData]);

  // Real-time polling: update comparison data every 30s
  useEffect(() => {
    if (!symbol) return;

    const pollData = async () => {
      try {
        const compData = await apiFetchComparisonData(symbol, period);
        setComparisonData(compData);
      } catch {
        // Silent fail on poll
      }
    };

    setIsPolling(true);
    pollingRef.current = setInterval(pollData, 30000);

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
      setIsPolling(false);
    };
  }, [symbol, period]);

  // Check if symbol is in watchlist
  useEffect(() => {
    const checkWatchlist = async () => {
      try {
        const data = await fetchWatchlist();
        const symbols = data.watchlist.map(w => w.symbol);
        setIsWatchlisted(symbols.includes(symbol));
      } catch {
        // Ignore
      }
    };
    if (symbol) checkWatchlist();
  }, [symbol]);

  const handleWatchlistToggle = async () => {
    try {
      if (isWatchlisted) {
        await removeFromWatchlist(symbol);
        setIsWatchlisted(false);
      } else {
        await addToWatchlist(symbol);
        setIsWatchlisted(true);
      }
    } catch {
      // Ignore
    }
  };

  // ===========================
  // LIGHTWEIGHT-CHARTS RENDERING
  // ===========================
  useEffect(() => {
    if (!comparisonData || !chartContainerRef.current) return;

    // Clean up previous chart instance
    if (chartInstance.current) {
      chartInstance.current.remove();
      chartInstance.current = null;
    }

    const container = chartContainerRef.current;

    // Create the chart with dark theme styling
    const chart = createChart(container, {
      width: container.clientWidth,
      height: 420,
      layout: {
        background: { type: ColorType.Solid, color: '#1a1d2e' },
        textColor: '#9aa0b0',
        fontFamily: 'Inter, sans-serif',
      },
      grid: {
        vertLines: { color: 'rgba(255, 255, 255, 0.04)' },
        horzLines: { color: 'rgba(255, 255, 255, 0.04)' },
      },
      crosshair: {
        mode: 0, // Normal crosshair
        vertLine: {
          color: 'rgba(108, 92, 231, 0.4)',
          width: 1,
          style: 2, // Dashed
          labelBackgroundColor: '#6c5ce7',
        },
        horzLine: {
          color: 'rgba(108, 92, 231, 0.4)',
          width: 1,
          style: 2,
          labelBackgroundColor: '#6c5ce7',
        },
      },
      rightPriceScale: {
        borderColor: 'rgba(255, 255, 255, 0.06)',
        scaleMargins: { top: 0.1, bottom: 0.25 },
      },
      timeScale: {
        borderColor: 'rgba(255, 255, 255, 0.06)',
        timeVisible: period === '1d' || period === '5d',
        secondsVisible: false,
      },
      handleScroll: { vertTouchDrag: false },
    });

    chartInstance.current = chart;

    // --- Candlestick Series or Fallback Line Series ---
    if (comparisonData.ohlc_data && comparisonData.ohlc_data.length > 0) {
      // Full OHLC candlestick mode (requires updated backend-datahandle)
      const candlestickSeries = chart.addSeries(CandlestickSeries, {
        upColor: '#00d68f',
        downColor: '#ff6b6b',
        borderDownColor: '#ff6b6b',
        borderUpColor: '#00d68f',
        wickDownColor: '#ff6b6b',
        wickUpColor: '#00d68f',
        priceScaleId: 'right',
      });
      candlestickSeries.setData(comparisonData.ohlc_data);

      // --- Volume Histogram ---
      if (comparisonData.volume_data && comparisonData.volume_data.length > 0) {
        const volumeSeries = chart.addSeries(HistogramSeries, {
          priceFormat: { type: 'volume' },
          priceScaleId: 'volume',
        });
        volumeSeries.priceScale().applyOptions({
          scaleMargins: { top: 0.85, bottom: 0 },
        });
        volumeSeries.setData(comparisonData.volume_data);
      }
    } else if (comparisonData.stock_prices && comparisonData.dates) {
      // Fallback: build a line series from the legacy dates + stock_prices arrays
      const lineData = comparisonData.dates.map((dateStr, i) => {
        // Parse date strings into Unix timestamps
        let ts;
        if (dateStr.includes(':')) {
          // Intraday format "MM-DD HH:MM" — assume current year
          const year = new Date().getFullYear();
          ts = Math.floor(new Date(`${year}-${dateStr.replace(' ', 'T')}:00`).getTime() / 1000);
        } else {
          // Daily format "YYYY-MM-DD"
          ts = Math.floor(new Date(dateStr).getTime() / 1000);
        }
        return { time: ts, value: comparisonData.stock_prices[i] };
      }).filter(d => !isNaN(d.time) && d.value != null);

      if (lineData.length > 0) {
        const lineSeries = chart.addSeries(LineSeries, {
          color: '#6c5ce7',
          lineWidth: 2,
          priceScaleId: 'right',
          crosshairMarkerVisible: true,
          crosshairMarkerRadius: 4,
          lastValueVisible: true,
          priceLineVisible: true,
        });
        lineSeries.setData(lineData);
      }
    }

    // --- S&P 500 Overlay (optional) ---
    if (showSP500) {
      let sp500Data = null;
      let sp500Title = 'S&P 500';

      if (comparisonData.sp500_line_data && comparisonData.sp500_line_data.length > 0) {
        // Use new Unix-timestamp-based data
        sp500Data = comparisonData.sp500_line_data;
      } else if (comparisonData.sp500_performance && comparisonData.dates) {
        // Fallback: build from legacy dates + sp500_performance (% change)
        sp500Title = 'S&P 500 (%)';
        sp500Data = comparisonData.dates.map((dateStr, i) => {
          let ts;
          if (dateStr.includes(':')) {
            const year = new Date().getFullYear();
            ts = Math.floor(new Date(`${year}-${dateStr.replace(' ', 'T')}:00`).getTime() / 1000);
          } else {
            ts = Math.floor(new Date(dateStr).getTime() / 1000);
          }
          return { time: ts, value: comparisonData.sp500_performance[i] };
        }).filter(d => !isNaN(d.time) && d.value != null);
      }

      if (sp500Data && sp500Data.length > 0) {
        const sp500Series = chart.addSeries(LineSeries, {
          color: '#ff6384',
          lineWidth: 2,
          priceScaleId: 'left',
          lineStyle: 0,
          crosshairMarkerVisible: true,
          crosshairMarkerRadius: 4,
          title: sp500Title,
        });
        sp500Series.setData(sp500Data);

        chart.applyOptions({
          leftPriceScale: {
            visible: true,
            borderColor: 'rgba(255, 255, 255, 0.06)',
            scaleMargins: { top: 0.1, bottom: 0.25 },
          },
        });
      } else {
        chart.applyOptions({
          leftPriceScale: { visible: false },
        });
      }
    } else {
      chart.applyOptions({
        leftPriceScale: { visible: false },
      });
    }

    // Fit all data into the visible range
    chart.timeScale().fitContent();

    // --- Responsive resize with ResizeObserver ---
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width } = entry.contentRect;
        if (width > 0) {
          chart.applyOptions({ width });
        }
      }
    });
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      if (chartInstance.current) {
        chartInstance.current.remove();
        chartInstance.current = null;
      }
    };
  }, [comparisonData, showSP500, period]);

  const handleSearchChange = (e) => {
    const val = e.target.value.toUpperCase();
    setSearch(val);

    // Clear any pending search timer
    if (searchTimerRef.current) {
      clearTimeout(searchTimerRef.current);
    }

    if (val.length > 1) {
      // Debounce: wait 300ms after last keystroke before firing API call
      searchTimerRef.current = setTimeout(async () => {
        try {
          const data = await searchSymbols(val);
          setSuggestions(data);
        } catch {
          setSuggestions([]);
        }
      }, 300);
    } else {
      setSuggestions([]);
    }
  };

  const handleSuggestionClick = (sym) => {
    navigate(`/dashboard/${sym}`);
    setSearch('');
    setSuggestions([]);
  };

  const handlePeriodChange = (newPeriod) => {
    setPeriod(newPeriod);
  };

  // Show price and price change for selected period from comparisonData
  const renderPeriodPriceChange = () => {
    if (!comparisonData) return null;
    const isPositive = comparisonData.price_change >= 0;
    return (
      <div className={`price-change ${isPositive ? 'positive' : 'negative'}`}>
        {isPositive ? '+' : ''}{comparisonData.price_change} ({isPositive ? '+' : ''}{comparisonData.price_change_percent}%)
      </div>
    );
  };

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <h1 className="logo">MarkeTracker</h1>
        <div className="search-container">
          <input
            type="text"
            className="search-input"
            placeholder="Search for a stock symbol..."
            value={search}
            onChange={handleSearchChange}
          />
          {suggestions.length > 0 && (
            <ul className="suggestions-list">
              {suggestions.map((s) => (
                <li key={s.symbol} onClick={() => handleSuggestionClick(s.symbol)}>
                  <strong>{s.symbol}</strong> - {s.name}
                </li>
              ))}
            </ul>
          )}
        </div>
      </header>

      {error && <div className="error">{error}</div>}
      {!dashboardData && !error && <div className="loading">Loading Dashboard...</div>}

      {dashboardData && (
        <>
          <main className="main-content">
            <div className="stock-header">
              <div>
                <h1 className="stock-name">{dashboardData.longName}({symbol})</h1>
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginTop: '0.5rem' }}>
                  <button
                    onClick={handleWatchlistToggle}
                    style={{
                      padding: '0.35rem 0.75rem',
                      borderRadius: 'var(--radius-sm)',
                      border: `1px solid ${isWatchlisted ? 'var(--color-warning)' : 'var(--border-color)'}`,
                      background: isWatchlisted ? 'var(--color-warning-bg)' : 'var(--bg-input)',
                      color: isWatchlisted ? 'var(--color-warning)' : 'var(--text-secondary)',
                      cursor: 'pointer',
                      fontSize: '0.8rem',
                      fontWeight: 600,
                      fontFamily: 'Inter, sans-serif',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    {isWatchlisted ? '★ Watchlisted' : '☆ Add to Watchlist'}
                  </button>
                  {isPolling && (
                    <span style={{
                      padding: '0.25rem 0.6rem',
                      borderRadius: 'var(--radius-sm)',
                      background: 'var(--color-success-bg)',
                      color: 'var(--color-success)',
                      fontSize: '0.7rem',
                      fontWeight: 700,
                      border: '1px solid rgba(0,214,143,0.3)',
                      letterSpacing: '0.5px',
                    }}>
                      ● LIVE
                    </span>
                  )}
                </div>
              </div>
              <div className="price-info">
                <div className="current-price">
                  {comparisonData && typeof comparisonData.end_price === 'number'
                    ? `$${comparisonData.end_price}`
                    : 'N/A'}
                </div>
                {renderPeriodPriceChange()}
              </div>
            </div>

            <div className="chart-container">
              <div className="chart-header">
                <div className="chart-title">
                  {showSP500 ? 'Price Chart + S&P 500 Overlay' : 'Candlestick Chart'}
                </div>
                <div className="chart-controls">
                  <div className="period-buttons">
                    {['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', 'max'].map(p => (
                      <button
                        key={p}
                        className={`period-btn ${period === p ? 'active' : ''}`}
                        onClick={() => handlePeriodChange(p)}
                      >
                        {p.toUpperCase()}
                      </button>
                    ))}
                  </div>
                  <button
                    className={`toggle-btn ${showSP500 ? 'active' : ''}`}
                    onClick={() => setShowSP500(!showSP500)}
                  >
                    {showSP500 ? 'Hide S&P 500' : 'Compare S&P 500'}
                  </button>
                </div>
              </div>
              <div
                ref={chartContainerRef}
                className="lightweight-chart-wrapper"
                style={{ width: '100%', height: '420px' }}
              />
              {showSP500 && (
                <div className="chart-legend">
                  <span className="legend-item">
                    <span className="legend-color" style={{ background: '#00d68f' }}></span>
                    {comparisonData?.stock_symbol || symbol} (Candlestick)
                  </span>
                  <span className="legend-item">
                    <span className="legend-color" style={{ background: '#ff6384' }}></span>
                    S&P 500 (Line — Left Axis)
                  </span>
                </div>
              )}
            </div>

            <div className="dashboard-metrics-section">
              <h3 className="section-title">Key Metrics</h3>
              <div className="metrics-grid">
                <div className="metric"><span className="label">Sector:</span><span className="value">{dashboardData.sector || 'N/A'}</span></div>
                <div className="metric"><span className="label">Industry:</span><span className="value">{dashboardData.industry || 'N/A'}</span></div>
                <div className="metric"><span className="label">Market Cap:</span><span className="value">{dashboardData.marketCap || 'N/A'}</span></div>
                <div className="metric"><span className="label">Day High / Low:</span><span className="value">${dashboardData.regularMarketDayHigh?.toFixed(2) || 'N/A'} / ${dashboardData.regularMarketDayLow?.toFixed(2) || 'N/A'}</span></div>
                <div className="metric"><span className="label">52-Week High / Low:</span><span className="value">${dashboardData.fiftyTwoWeekHigh?.toFixed(2) || 'N/A'} / ${dashboardData.fiftyTwoWeekLow?.toFixed(2) || 'N/A'}</span></div>
                <div className="metric"><span className="label">Website:</span><span className="value"><a href={dashboardData.website} target="_blank" rel="noopener noreferrer">Visit Official Website</a></span></div>
              </div>

              <h3 className="section-title">Valuation & Rating</h3>
              <div className="metrics-grid">
                <div className="metric"><span className="label">P/E Ratio (Trailing)</span><span className="value">{dashboardData.trailingPE?.toFixed(2) || 'N/A'}</span></div>
                <div className="metric"><span className="label">EPS (Trailing)</span><span className="value">{dashboardData.trailingEps?.toFixed(2) || 'N/A'}</span></div>
                <div className="metric"><span className="label">Dividend Yield</span><span className="value">{dashboardData.dividendYield ? `${(dashboardData.dividendYield).toFixed(2)}%` : 'N/A'}</span></div>
                <div className="metric"><span className="label">Mean Target Price</span><span className="value">${dashboardData.targetMeanPrice?.toFixed(2) || 'N/A'}</span></div>
                <div className="metric"><span className="label">Analyst Rating</span><span className="value">{dashboardData.averageAnalystRating || 'N/A'}</span></div>
                <div className="metric metric-forecast">
                  <span className="label">ML Price Forecast (Educational)</span>
                  <span className="value">{dashboardData.forecast_price ? `$${dashboardData.forecast_price.toFixed(2)}` : 'Not Available'}</span>
                </div>
              </div>

              <h3 className="section-title">Business Summary</h3>
              <div className="business-summary">
                <p>{dashboardData.longBusinessSummary || 'No summary available.'}</p>
              </div>

              <h3 className="section-title">Quarterly Income Statement</h3>
              <div className="income-statement-grid">
                {dashboardData.income_grid_items && dashboardData.income_grid_items.length > 0 ? (
                  dashboardData.income_grid_items.map((item, idx) => (
                    <div key={idx} className={`income-item ${item.css_class}`}>
                      <span className="item-label">{item.label}</span>
                      <span className="item-value">{item.value}</span>
                    </div>
                  ))
                ) : (
                  <p>Income statement data is not available.</p>
                )}
              </div>
            </div>
          </main>
        </>
      )}
    </div>
  );
};

export default Dashboard;
