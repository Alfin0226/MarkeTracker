import React, { useState, useEffect } from 'react';
import { fetchPortfolio } from '../utils/api';
import PortfolioSummary from '../components/PortfolioSummary';
import PortfolioTable from '../components/PortfolioTable';
import TradeForm from '../components/TradeForm';
import '../styles/Portfolio.css';

function Portfolio() {
  const [portfolio, setPortfolio] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const initialInvestment = 1000000;

  const loadPortfolio = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchPortfolio();
      setPortfolio(data);
    } catch (error) {
      setError(error.message || 'Failed to load portfolio');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPortfolio();
  }, []);

  return (
    <div className="portfolio-container">
      <h2>Portfolio</h2>

      {error && <div className="alert alert-danger">{error}</div>}

      {loading ? (
        <div className="portfolio-summary">
          <div className="summary-card"><div className="loading-skeleton" style={{ height: '80px' }}></div></div>
          <div className="summary-card"><div className="loading-skeleton" style={{ height: '80px' }}></div></div>
          <div className="summary-card"><div className="loading-skeleton" style={{ height: '80px' }}></div></div>
        </div>
      ) : (
        <>
          <PortfolioSummary portfolio={portfolio} initialInvestment={initialInvestment} />

          <div className="trade-section">
            <TradeForm onTradeComplete={loadPortfolio} />
          </div>

          <div className="holdings-section">
            <h3>Holdings</h3>
            <PortfolioTable portfolio={portfolio} />
          </div>
        </>
      )}

      <button className="portfolio-refresh-btn" onClick={loadPortfolio} disabled={loading}>
        {loading ? 'Loading...' : '↻ Refresh Portfolio'}
      </button>
    </div>
  );
}

export default Portfolio;