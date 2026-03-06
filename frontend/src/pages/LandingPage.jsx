import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/LandingPage.css';

const LandingPage = () => {
  return (
    <div className="landing-container">
      <section className="hero-section">
        <div className="hero-glow"></div>
        <div className="hero-content">
          <span className="hero-badge">📈 Virtual Trading Platform</span>
          <h1 className="hero-title">
            Track. Analyze.
            <span className="hero-gradient-text"> Outperform.</span>
          </h1>
          <p className="hero-subtitle">
            Real-time stock data, S&P 500 comparisons, and a $1M virtual portfolio to practice your trading strategy.
          </p>
          <div className="hero-cta">
            <Link to="/register" className="btn-hero-primary">Get Started Free</Link>
          </div>
        </div>
      </section>

      <section className="features-section">
        <div className="features-grid">
          <div className="feature-card">
            <div className="feature-icon">📊</div>
            <h3>Real-time Data</h3>
            <p>Live stock prices, charts, and key financial metrics from NASDAQ & NYSE</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">⚡</div>
            <h3>Performance Tracking</h3>
            <p>Compare any stock against the S&P 500 across multiple time periods</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">💼</div>
            <h3>Virtual Portfolio</h3>
            <p>Practice with $1M virtual capital — track P&L, win rate, and total returns</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon">🔍</div>
            <h3>Smart Analytics</h3>
            <p>Income statements, analyst ratings, and ML-powered price forecasts</p>
          </div>
        </div>
      </section>

      <section className="cta-section">
        <h2>Ready to start tracking?</h2>
        <p>Join MarkeTracker and level up your investment skills.</p>
        <div className="cta-buttons">
          <Link to="/login" className="btn-hero-primary">Login</Link>
          <Link to="/register" className="btn-hero-secondary">Sign Up</Link>
        </div>
      </section>
    </div>
  );
};

export default LandingPage;
