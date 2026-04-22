import React from 'react';

function PortfolioTable({ portfolio }) {
  return (
    <table className="table">
      <thead>
        <tr>
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
            const glPercent = typeof position.gain_loss === 'number'
              ? ((position.gain_loss / (position.avg_price * position.shares)) * 100).toFixed(2)
              : null;

            return (
              <tr key={position.symbol}>
                <td style={{ fontWeight: 600, color: 'var(--accent-primary)' }}>
                  {position.symbol}
                </td>
                <td>{position.shares}</td>
                <td>{typeof position.avg_price === 'number' ? `$${position.avg_price.toFixed(2)}` : 'N/A'}</td>
                <td>{typeof position.current_price === 'number' ? `$${position.current_price.toFixed(2)}` : 'N/A'}</td>
                <td>{typeof position.value === 'number' ? `$${position.value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : 'N/A'}</td>
                <td>
                  {typeof position.gain_loss === 'number' ? (
                    <span style={{ color: position.gain_loss >= 0 ? 'var(--color-success)' : 'var(--color-danger)' }}>
                      {position.gain_loss >= 0 ? '+' : ''}${position.gain_loss.toFixed(2)}
                      <span style={{ opacity: 0.7, marginLeft: '0.3rem', fontSize: '0.85em' }}>
                        ({position.gain_loss >= 0 ? '+' : ''}{glPercent}%)
                      </span>
                    </span>
                  ) : (
                    'N/A'
                  )}
                </td>
              </tr>
            );
          })
        ) : (
          <tr>
            <td colSpan="6" style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem' }}>
              No holdings yet — execute a trade above to get started
            </td>
          </tr>
        )}
      </tbody>
    </table>
  );
}

export default PortfolioTable;