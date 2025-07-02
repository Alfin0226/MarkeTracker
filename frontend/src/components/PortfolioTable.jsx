import React from 'react';

function PortfolioTable({ portfolio }) {
  return (
    <table className="table">
      <thead>
        <tr className="table-primary">
          <th>Symbol</th>
          <th>Shares</th>
          <th>Avg Buy Price</th>
          <th>Current Price</th>
          <th>Value</th>
          <th>Gain/Loss</th>
        </tr>
      </thead>
      <tbody>
        {portfolio?.portfolio?.length > 0 ? (
          portfolio.portfolio.map((position) => (
            <tr key={position.symbol}>
              <td>{position.symbol}</td>
              <td>{position.shares}</td>
              <td>{typeof position.avg_price === 'number' ? `$${position.avg_price.toFixed(2)}` : 'N/A'}</td>
              <td>{typeof position.current_price === 'number' ? `$${position.current_price.toFixed(2)}` : 'N/A'}</td>
              <td>{typeof position.value === 'number' ? `$${position.value.toFixed(2)}` : 'N/A'}</td>
              <td className={typeof position.gain_loss === 'number' ? (position.gain_loss >= 0 ? 'text-success' : 'text-danger') : ''}>
                {typeof position.gain_loss === 'number' ? (
                  <>
                    {`$${position.gain_loss.toFixed(2)} `}
                    ({((position.gain_loss / (position.avg_price * position.shares)) * 100).toFixed(2)}%)
                  </>
                ) : (
                  'N/A'
                )}
              </td>
            </tr>
          ))
        ) : (
          <tr>
            <td colSpan="6" className="text-center">No holdings yet</td>
          </tr>
        )}
      </tbody>
    </table>
  );
}

export default PortfolioTable;