import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/LandingPage.css';

const LandingPage = () => {
  return (
    <div className="landing-container">
      <header className="landing-header">
        <h1>MarkeTracker</h1>
        <p>Track your investments. Make smarter decisions.</p>
      </header>
      <section className="features">
        <div className="feature">
          <h3>Real-time Stock Data</h3>
          <p>Get up-to-date information on your favorite stocks</p>
        </div>
        <div className="feature">
          <h3>Portfolio Tracking</h3>
          <p>Monitor your investments in one place</p>
        </div>
        <div className="feature">
          <h3>Smart Analytics</h3>
          <p>Insights to help you make better investment choices</p>
        </div>
      </section>
      <div className="cta-container">
        <Link to="/login" className="cta-button">Login</Link>
        <Link to="/register" className="cta-button secondary">Sign Up</Link>
        <Link to ="/dashboard" className="cta-button demo">Demo</Link>
      </div>
    </div>
  );
};

export default LandingPage;
