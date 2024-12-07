import React, { useState, useEffect } from 'react';
import { fetchPortfolio } from '../utils/api';
import PortfolioSummary from '../components/PortfolioSummary';
import PortfolioTable from '../components/PortfolioTable';
import TradeForm from '../components/TradeForm';

function Portfolio() {
  const [initialInvestment] = useState(1000000);
  const [portfolio, setPortfolio] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadPortfolio = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchPortfolio();
      console.log('Portfolio data received:', data);
      setPortfolio(data);
    } catch (error) {
      console.error('Portfolio error:', error);
      setError(error.message || 'Failed to load portfolio');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPortfolio();
  }, []);

  if (loading) {
    return (
      <div className="container mt-5">
        <div className="alert alert-info">Loading portfolio data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mt-5">
        <div className="alert alert-danger">{error}</div>
      </div>
    );
  }

  if (!portfolio) {
    return (
      <div className="container mt-5">
        <div className="alert alert-warning">No portfolio data available</div>
      </div>
    );
  }

  return (
    <div className="container mt-5">
      <h2>Portfolio</h2>
      
      <div className="row mb-4">
        <div className="col">
          <PortfolioSummary 
            portfolio={{
              total_value: portfolio.total_value || 0,
              cash_balance: portfolio.cash_balance || 0
            }} 
            initialInvestment={initialInvestment} 
          />
        </div>
      </div>

      <div className="row mb-4">
        <div className="col">
          <TradeForm onTradeComplete={loadPortfolio} />
        </div>
      </div>

      <div className="row">
        <div className="col">
          <h3>Holdings</h3>
          <PortfolioTable portfolio={portfolio} />
        </div>
      </div>

      <button 
        className="btn btn-secondary mt-3" 
        onClick={loadPortfolio}
      >
        Refresh Portfolio
      </button>
    </div>
  );
}

export default Portfolio;