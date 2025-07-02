import React, { useEffect, useState, useCallback } from 'react';
import Chart from 'chart.js/auto';
import '../styles/Dashboard.css';
import { fetchDashboardData as apiFetchDashboardData, fetchComparisonData as apiFetchComparisonData, searchSymbols } from '../utils/api';

const Dashboard = ({ symbol: initialSymbol }) => {
  const [symbol, setSymbol] = useState(initialSymbol || 'AAPL');
  const [dashboardData, setDashboardData] = useState(null);
  const [comparisonData, setComparisonData] = useState(null);
  const [period, setPeriod] = useState('1y');
  const [showSP500, setShowSP500] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const chartRef = React.useRef(null);
  const chartInstance = React.useRef(null);

  const fetchDashboardData = useCallback(async (sym) => {
    setError(null);
    setDashboardData(null);
    try {
      const data = await apiFetchDashboardData(sym);
      setDashboardData(data);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  const fetchComparisonData = useCallback(async (sym, per) => {
    setComparisonData(null);
    try {
      const data = await apiFetchComparisonData(sym, per);
      setComparisonData(data);
    } catch (err) {
      setComparisonData(null); // Clear data on error
      setError('Failed to load comparison data.');
    }
  }, []);


  useEffect(() => {
    fetchDashboardData(symbol);
    fetchComparisonData(symbol, period);
  }, [symbol, period, fetchDashboardData, fetchComparisonData]);

  useEffect(() => {
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }
    if (comparisonData && chartRef.current) {
      chartInstance.current = new Chart(chartRef.current, {
        type: 'line',
        data: {
          labels: comparisonData.dates,
          datasets: [
            {
              label: symbol,
              data: comparisonData.stock_performance,
              borderColor: '#007bff',
              fill: false,
            },
            showSP500 && {
              label: 'S&P 500',
              data: comparisonData.sp500_performance,
              borderColor: '#28a745',
              fill: false,
            },
          ].filter(Boolean),
        },
        options: {
          responsive: true,
          plugins: {
            legend: { display: true },
            title: {
              display: true,
              text: `Performance vs S&P 500 (${period})`,
            },
          },
        },
      });
    }
    
    return () => {
        if (chartInstance.current) {
            chartInstance.current.destroy();
        }
    };
  }, [comparisonData, showSP500, symbol, period]);

  const handleSearchChange = async (e) => {
    const val = e.target.value;
    setSearch(val);
    if (val.length > 1) {
      try {
        const data = await searchSymbols(val);
        setSuggestions(data);
      } catch (error) {
        setSuggestions([]);
      }
    } else {
      setSuggestions([]);
    }
  };

  const handleSuggestionClick = (sym) => {
    setSymbol(sym);
    setSearch('');
    setSuggestions([]);
  };

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <div className="search-bar">
          <input
            type="text"
            placeholder="Search company or symbol..."
            value={search}
            onChange={handleSearchChange}
          />
          {suggestions.length > 0 && (
            <ul className="suggestions-list">
              {suggestions.map((s) => (
                <li key={s.symbol} onClick={() => handleSuggestionClick(s.symbol)}>
                  {s.symbol} - {s.name}
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="period-select">
          <label>Period: </label>
          <select value={period} onChange={e => setPeriod(e.target.value)}>
            <option value="1y">1 Year</option>
            <option value="6mo">6 Months</option>
            <option value="3mo">3 Months</option>
            <option value="1mo">1 Month</option>
          </select>
        </div>
        <div className="sp500-toggle">
          <label>
            <input
              type="checkbox"
              checked={showSP500}
              onChange={() => setShowSP500(v => !v)}
            />
            Show S&P 500
          </label>
        </div>
      </div>
      {error && <div className="error">{error}</div>}
      {dashboardData ? (
        <div className="dashboard-main">
          <div className="dashboard-info">
            <h2>{dashboardData.longname || symbol}</h2>
            <div className="info-row">
              <span>Sector: {dashboardData.sector}</span>
              <span>Industry: {dashboardData.industry}</span>
              <span>Market Cap: {dashboardData.marketCap}</span>
            </div>
            <div className="info-row">
              <span>PE Ratio: {dashboardData.PE_ratio}</span>
              <span>EPS: {dashboardData.Eps}</span>
              <span>Dividend Yield: {dashboardData.dividend}</span>
            </div>
            <div className="info-row">
              <span>Target Price: {dashboardData.target}</span>
              <span>Analyst Rating: {dashboardData.rating}</span>
              <span>Forecast Price: {dashboardData.forecast_price ? dashboardData.forecast_price.toFixed(2) : 'N/A'}</span>
            </div>
            <div className="info-row">
              <a href={dashboardData.website} target="_blank" rel="noopener noreferrer">Company Website</a>
            </div>
          </div>
          <div className="dashboard-chart">
            <canvas ref={chartRef} />
          </div>
          <div className="dashboard-income">
            <h3>Quarterly Income Statement</h3>
            <table>
              <thead>
                <tr>
                  <th>Item</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {dashboardData.income_grid_items && dashboardData.income_grid_items.map((item, idx) => (
                  <tr key={idx} className={item.css_class}>
                    <td>{item.label}</td>
                    <td>{item.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="loading">Loading dashboard...</div>
      )}
    </div>
  );
};

export default Dashboard;
