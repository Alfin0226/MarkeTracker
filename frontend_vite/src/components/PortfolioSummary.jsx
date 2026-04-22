import React from 'react';

function PortfolioSummary({ portfolio, initialInvestment }) {
  if (!portfolio) return null;

  const gainLoss = portfolio.total_value - initialInvestment;
  const gainLossPercent = ((gainLoss) / initialInvestment * 100).toFixed(2);
  const isPositive = gainLoss >= 0;

  return (
    <div className="portfolio-summary">
      <div className="summary-card">
        <div className="summary-label">Total Portfolio Value</div>
        <div className="summary-value">
          ${portfolio.total_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </div>
      </div>
      <div className="summary-card">
        <div className="summary-label">Cash Balance</div>
        <div className="summary-value">
          ${portfolio.cash_balance.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </div>
      </div>
      <div className="summary-card">
        <div className="summary-label">Total Gain/Loss</div>
        <div className={`summary-value ${isPositive ? 'positive' : 'negative'}`}>
          {isPositive ? '+' : ''}${gainLoss.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          <span style={{ fontSize: '0.85rem', marginLeft: '0.5rem', opacity: 0.8 }}>
            ({isPositive ? '+' : ''}{gainLossPercent}%)
          </span>
        </div>
      </div>
    </div>
  );
}

export default PortfolioSummary;