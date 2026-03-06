import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { register } from '../utils/api';
import '../styles/Auth.css';

function Register() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    try {
      await register({ email, password });
      navigate('/login');
    } catch (err) {
      const errorMessage = err.response?.data?.error || 'Registration failed. Please try again.';
      if (errorMessage === 'Email already exists') {
        setError('This email is already registered. Please try logging in instead.');
      } else {
        setError(errorMessage);
      }
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <h2>Create Account</h2>
        <p className="auth-subtitle">Start trading with $1M virtual capital</p>
        {error && (
          <div className="alert alert-danger">
            {error}
            {error.includes('already registered') && (
              <div className="mt-2">
                <button
                  className="auth-link-btn"
                  onClick={() => navigate('/login')}
                  style={{
                    color: 'var(--accent-primary)',
                    background: 'none',
                    border: 'none',
                    cursor: 'pointer',
                    fontWeight: 600,
                    padding: 0
                  }}
                >
                  Go to Login →
                </button>
              </div>
            )}
          </div>
        )}
        <form onSubmit={handleSubmit}>
          <div className="mb-3">
            <label className="form-label">Email</label>
            <input
              type="email"
              className="form-control"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>
          <div className="mb-3">
            <label className="form-label">Password</label>
            <input
              type="password"
              className="form-control"
              placeholder="Choose a strong password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
          </div>
          <button type="submit" className="btn btn-primary">
            Create Account
          </button>
        </form>
        <div className="auth-link">
          Already have an account? <Link to="/login">Sign In</Link>
        </div>
      </div>
    </div>
  );
}

export default Register;