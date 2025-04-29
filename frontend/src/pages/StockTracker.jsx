import React, { useState, useEffect } from 'react';
import { fetchStockData } from '../utils/api';
import StockChart from '../components/StockChart';

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

  const loadStockData = async () => {
    if (!symbol) return;
    
    setLoading(true);
    try {
      const { period: p, interval } = periodMap[period];
      const data = await fetchStockData(symbol, p, interval);
      setStockData(data);
    } catch (error) {
      console.error('Error fetching stock data:', error);
    }
    setLoading(false);
  };

  useEffect(() => {
    if (symbol) {
      loadStockData();
    }
  }, [symbol, period]);

  return (
    <div className="container mt-5">
      <h2>Stock Tracker</h2>
      <div className="mb-3">
        <input
          type="text"
          className="form-control"
          placeholder="Enter stock symbol (e.g., AAPL)"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value.toUpperCase())}
        />
      </div>
      
      <div className="btn-group mb-3">
        {Object.keys(periodMap).map((p) => (
          <button
            key={p}
            className={`btn btn-outline-primary ${period === p ? 'active' : ''}`}
            onClick={() => setPeriod(p)}
          >
            {p}
          </button>
        ))}
      </div>

      {loading && <div>Loading...</div>}
      
      {stockData && stockData.info && (
        <div>
          <div className="mb-3">
            <h3>{stockData.info.longName || symbol}</h3>
            <p>Current Price: ${stockData.info.currentPrice || stockData.info.regularMarketPrice || stockData.prices[stockData.prices.length - 1]?.toFixed(2) || 'N/A'}</p>
          </div>
          <StockChart stockData={stockData} symbol={symbol} />
        </div>
      )}
    </div>
  );
}

export default StockTracker;