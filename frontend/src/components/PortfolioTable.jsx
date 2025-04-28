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
          portfolio.portfolio.map((position) => {
            // Defensive checks for all number fields
            const avgPrice =
              position.avg_price !== undefined && position.avg_price !== null
                ? position.avg_price.toFixed(2)
                : 'N/A';
            const currentPrice =
              position.current_price !== undefined && position.current_price !== null
                ? position.current_price.toFixed(2)
                : position.error
                  ? `Error: ${position.error}`
                  : 'Unavailable';
            const value =
              position.current_price !== undefined && position.current_price !== null &&
              position.shares !== undefined && position.shares !== null
                ? (position.current_price * position.shares).toFixed(2)
                : 'N/A';
            const gainLoss =
              position.current_price !== undefined && position.current_price !== null &&
              position.avg_price !== undefined && position.avg_price !== null &&
              position.shares !== undefined && position.shares !== null
                ? ((position.current_price - position.avg_price) * position.shares).toFixed(2)
                : 'N/A';
            const gainLossPct =
              position.current_price !== undefined && position.current_price !== null &&
              position.avg_price !== undefined && position.avg_price !== null &&
              position.shares !== undefined && position.shares !== null && position.avg_price * position.shares !== 0
                ? (((position.current_price - position.avg_price) / position.avg_price) * 100).toFixed(2)
                : 'N/A';
            return (
              <tr key={position.symbol}>
                <td>{position.symbol}</td>
                <td>{position.shares}</td>
                <td>{avgPrice}</td>
                <td>{currentPrice}</td>
                <td>{value}</td>
                <td className={gainLoss !== 'N/A' && parseFloat(gainLoss) >= 0 ? 'text-success' : 'text-danger'}>
                  {gainLoss}
                  {gainLossPct !== 'N/A' && gainLoss !== 'N/A' ? ` (${gainLossPct}%)` : ''}
                </td>
              </tr>
            );
          })
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