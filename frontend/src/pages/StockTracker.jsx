import React, { useState, useEffect, useRef } from 'react';
import { fetchStockData, searchSymbols } from '../utils/api';
import StockChart from '../components/StockChart';
import { Link } from 'react-router-dom';

const periodMap = {
  '1d': { period: '1d', interval: '5m' },
  '1w': { period: '5d', interval: '15m' },
  '1m': { period: '1mo', interval: '1d' },
  '6m': { period: '6mo', interval: '1d' },
  '1y': { period: '1y', interval: '1d' },
  '5y': { period: '5y', interval: '1wk' },
  '10y': { period: '10y', interval: '1mo' },
  'max': { period: 'max', interval: '1mo' }
};

function StockTracker() {
  const [symbol, setSymbol] = useState('');
  const [stockData, setStockData] = useState(null);
  const [period, setPeriod] = useState('1d');
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const searchTimerRef = useRef(null);

  const loadStockData = async () => {
    if (!symbol) return;

    setLoading(true);
    try {
      const { period: p, interval } = periodMap[period];
      const data = await fetchStockData(symbol, p, interval);
      setStockData(data);
    } catch (error) {
      // Error handled silently
    }
    setLoading(false);
  };

  useEffect(() => {
    if (symbol) {
      loadStockData();
    }
  }, [symbol, period]);

  const handleSearchChange = (e) => {
    const val = e.target.value.toUpperCase();
    setSearch(val);

    if (searchTimerRef.current) {
      clearTimeout(searchTimerRef.current);
    }

    if (val.length > 1) {
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
    setSymbol(sym);
    setSearch(sym);
    setSuggestions([]);
  };

  return (
    <div style={{ maxWidth: '900px', margin: '0 auto', padding: '2rem 1.5rem' }}>
      <h2 style={{
        fontSize: '1.8rem', fontWeight: 800, marginBottom: '1.5rem',
        background: 'var(--gradient-primary)', WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent', backgroundClip: 'text'
      }}>
        Stock Tracker
      </h2>

      <div style={{ position: 'relative', marginBottom: '1.5rem' }}>
        <input
          type="text"
          className="form-control"
          placeholder="Search by symbol or company name (e.g., AAPL)"
          value={search}
          onChange={handleSearchChange}
          autoComplete="off"
        />
        {suggestions.length > 0 && (
          <ul className="suggestions-list" style={{
            position: 'absolute', width: '100%', background: 'var(--bg-elevated)',
            border: '1px solid var(--border-color)', borderTop: 'none',
            borderRadius: '0 0 var(--radius-sm) var(--radius-sm)',
            listStyle: 'none', margin: 0, padding: 0, zIndex: 1000,
            boxShadow: 'var(--shadow-md)', maxHeight: '250px', overflowY: 'auto'
          }}>
            {suggestions.map((s) => (
              <li key={s.symbol} onClick={() => handleSuggestionClick(s.symbol)}
                style={{
                  padding: '0.75rem 1rem', cursor: 'pointer',
                  color: 'var(--text-primary)', fontSize: '0.9rem',
                  borderBottom: '1px solid var(--border-subtle)',
                  transition: 'background-color 0.15s ease'
                }}
                onMouseOver={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-card-hover)'}
                onMouseOut={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
              >
                <strong style={{ color: 'var(--accent-primary)' }}>{s.symbol}</strong> — {s.name}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div style={{ display: 'flex', gap: '0.3rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
        {Object.keys(periodMap).map((p) => (
          <button
            key={p}
            className={`period-btn ${period === p ? 'active' : ''}`}
            onClick={() => setPeriod(p)}
            style={{
              padding: '0.4rem 0.75rem',
              border: `1px solid ${period === p ? 'var(--accent-primary)' : 'var(--border-color)'}`,
              background: period === p ? 'var(--accent-primary)' : 'var(--bg-input)',
              color: period === p ? 'white' : 'var(--text-secondary)',
              borderRadius: 'var(--radius-sm)',
              cursor: 'pointer',
              fontWeight: 500,
              fontSize: '0.8rem',
              fontFamily: 'Inter, sans-serif',
              transition: 'all 0.15s ease',
            }}
          >
            {p}
          </button>
        ))}
      </div>

      {loading && (
        <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-secondary)' }}>
          Loading...
        </div>
      )}

      {stockData && stockData.info && (
        <div style={{
          background: 'var(--bg-card)', border: '1px solid var(--border-color)',
          borderRadius: 'var(--radius-lg)', padding: '1.5rem'
        }}>
          <div style={{ marginBottom: '1rem' }}>
            <h3 style={{ color: 'var(--text-heading)', marginBottom: '0.3rem' }}>
              {stockData.info.longName || symbol}
            </h3>
            <p style={{ color: 'var(--text-heading)', fontSize: '1.8rem', fontWeight: 700, margin: 0 }}>
              ${stockData.info.currentPrice || stockData.info.regularMarketPrice || stockData.prices[stockData.prices.length - 1]?.toFixed(2) || 'N/A'}
            </p>
          </div>
          <StockChart stockData={stockData} symbol={symbol} />
        </div>
      )}

      <div style={{ marginTop: '2rem', textAlign: 'center' }}>
        <Link to="/register" className="btn-hero-primary" style={{ textDecoration: 'none' }}>
          Sign Up for Full Dashboard Access
        </Link>
      </div>
    </div>
  );
}

export default StockTracker;