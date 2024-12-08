import React, { useState } from 'react';
import { executeTrade } from '../utils/api';

function TradeForm({ onTradeComplete }) {
  const [tradeSymbol, setTradeSymbol] = useState('');
  const [tradeShares, setTradeShares] = useState('');
  const [tradeAction, setTradeAction] = useState('buy');
  const [tradeError, setTradeError] = useState('');

  const handleTrade = async (e) => {
    e.preventDefault();
    setTradeError('');
    
    try {
      if (!tradeSymbol || !tradeShares || tradeShares <= 0) {
        setTradeError('Please enter valid symbol and number of shares');
        return;
      }

      const response = await executeTrade({
        symbol: tradeSymbol.toUpperCase(),
        shares: parseInt(tradeShares),
        action: tradeAction
      });
      
      console.log('Trade response:', response);
      
      if (response.transaction) {
        setTradeSymbol('');
        setTradeShares('');
        const action = response.transaction.action || tradeAction;
        const message = `${action.toUpperCase()}: ${response.transaction.shares} shares of ${response.transaction.symbol} at $${response.transaction.price.toFixed(2)}\nNew Balance: $${response.transaction.new_balance.toFixed(2)}`;
        alert('Trade executed successfully!\n' + message);
        
        if (onTradeComplete) {
          onTradeComplete();
        }
      } else {
        setTradeError('Invalid response from server');
        console.error('Invalid trade response:', response);
      }
    } catch (error) {
      console.error('Trade error:', error);
      const errorMessage = error.response?.data?.error || 
                         (error.response?.data?.details) ||
                         error.message ||
                         'An unexpected error occurred while executing the trade';
      setTradeError(errorMessage);
    }
  };

  return (
    <div className="col">
      <h3>Execute Trade</h3>
      {tradeError && (
        <div className="alert alert-danger">
          <strong>Error: </strong>{tradeError}
        </div>
      )}
      <form onSubmit={handleTrade}>
        <div className="mb-3">
          <input
            type="text"
            className="form-control"
            placeholder="Stock Symbol"
            value={tradeSymbol}
            onChange={(e) => setTradeSymbol(e.target.value.toUpperCase())}
            required
          />
        </div>
        <div className="mb-3">
          <input
            type="number"
            className="form-control"
            placeholder="Number of Shares"
            value={tradeShares}
            onChange={(e) => setTradeShares(e.target.value)}
            required
          />
        </div>
        <div className="mb-3">
          <select
            className="form-control"
            value={tradeAction}
            onChange={(e) => setTradeAction(e.target.value)}
          >
            <option value="buy">Buy</option>
            <option value="sell">Sell</option>
          </select>
        </div>
        <button type="submit" className="btn btn-primary">
          Execute Trade
        </button>
      </form>
    </div>
  );
}

export default TradeForm;