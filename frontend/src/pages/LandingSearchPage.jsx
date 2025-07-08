import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/LandingSearchPage.css';
import { searchSymbols } from '../utils/api';

const LandingSearchPage = () => {
  const [search, setSearch] = useState('');
  const [suggestions, setSuggestions] = useState([]);
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    if (search.trim()) {
      navigate(`/dashboard/${search.trim().toUpperCase()}`);
    }
  };

  const handleSearchChange = async (e) => {
    const val = e.target.value.toUpperCase();
    setSearch(val);
    if (val.length > 1) {
      try {
        const data = await searchSymbols(val);
        setSuggestions(data);
      } catch (error) {
        setSuggestions([]);
      }
    } else {
      setSuggestions([]);
    }
  };

  const handleSuggestionClick = (sym) => {
    navigate(`/dashboard/${sym}`);
    setSearch('');
    setSuggestions([]);
  };

  return (
    <div className="landing-search-container">
      <div className="landing-search-card">
        <h1>S&amp;P 500 Company Data Search</h1>
        <p>Search by name or stock symbol.</p>
        <form onSubmit={handleSubmit} style={{ position: 'relative' }}>
          <input
            type="text"
            placeholder="e.g., 'Apple' or 'AAPL'"
            value={search}
            onChange={handleSearchChange}
            className="landing-search-input"
            autoComplete="off"
          />
          {suggestions.length > 0 && (
            <ul className="suggestions-list">
              {suggestions.map((s) => (
                <li key={s.symbol} onClick={() => handleSuggestionClick(s.symbol)}>
                  <strong>{s.symbol}</strong> - {s.name}
                </li>
              ))}
            </ul>
          )}
          <button type="submit" className="landing-search-btn">Search</button>
        </form>
      </div>
    </div>
  );
};

export default LandingSearchPage;