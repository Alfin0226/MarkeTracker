import React from 'react';

function PortfolioSummary({ portfolio, initialInvestment }) {
  if (!portfolio) return null;
  
  return (
    <div className="card">
      <div className="card-body">
        <h2 className="card-title">Account Summary</h2>
        <div className="Account Summary">
          <div>Total Value: ${portfolio.total_value.toFixed(2)}</div>
          <div>Cash Balance: {portfolio.cash_balance !== undefined && portfolio.cash_balance !== null ? portfolio.cash_balance.toFixed(2) : 'N/A'}</div>
          <div style={{ color: portfolio.total_value - initialInvestment >= 0 ? 'green' : 'red' }}>
            Total Gain/Loss: ${(portfolio.total_value - initialInvestment).toFixed(2)} 
            ({((portfolio.total_value - initialInvestment)/initialInvestment * 100).toFixed(2)}%)
          </div>
        </div>
      </div>
    </div>
  );
}

export default PortfolioSummary;