import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/LandingSearchPage.css';

const LandingSearchPage = () => {
  const [search, setSearch] = useState('');
  const navigate = useNavigate();

  const handleSubmit = (e) => {
    e.preventDefault();
    if (search.trim()) {
      navigate(`/dashboard/${search.trim().toUpperCase()}`);
    }
  };

  return (
    <div className="landing-search-container">
      <div className="landing-search-card">
        <h1>S&amp;P 500 Company Data Search</h1>
        <p>Search by name or stock symbol.</p>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            placeholder="e.g., 'Apple' or 'AAPL'"
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="landing-search-input"
          />
          <button type="submit" className="landing-search-btn">Search</button>
        </form>
      </div>
    </div>
  );
};

export default LandingSearchPage;