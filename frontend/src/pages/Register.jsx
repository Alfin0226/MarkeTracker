import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { register } from '../utils/api';

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
      // On successful registration, navigate to login
      navigate('/login');
    } catch (err) {
      // On error, stay on registration page and show error
      const errorMessage = err.response?.data?.error || 'Registration failed. Please try again.';
      setError(errorMessage);
      console.error('Registration error:', err.response?.data || err);
      
      // If email exists, suggest logging in
      if (errorMessage === 'Email already exists') {
        setError('This email is already registered. Please try logging in instead.');
      }
    }
  };

  return (
    <div className="container mt-5">
      <h2>Register</h2>
      {error && (
        <div className="alert alert-danger">
          {error}
          {error.includes('already registered') && (
            <div className="mt-2">
              <button 
                className="btn btn-link p-0" 
                onClick={() => navigate('/login')}
              >
                Go to Login
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
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        <button type="submit" className="btn btn-primary">
          Register
        </button>
      </form>
    </div>
  );
}

export default Register;