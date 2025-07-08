import React, { useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Navbar from './components/Navbar';
import Login from './pages/Login';
import Register from './pages/Register';
import StockTracker from './pages/StockTracker';
import Portfolio from './pages/Portfolio';
import PrivateRoute from './components/PrivateRoute';
import {Analytics} from '@vercel/analytics/react';
import LandingPage from './pages/LandingPage';
import Dashboard from './pages/Dashboard';

function App() {
  useEffect(() => {
    // Clear any existing tokens on app start
    localStorage.removeItem('token');
  }, []);

  return ( <>
    <Router>
      <div className="App">
        <Navbar />
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route
            path="/dashboard"
            element={
              <PrivateRoute>
                <Dashboard />
              </PrivateRoute>
            }
          />
          <Route
            path="/portfolio"
            element={
              <PrivateRoute>
                <Portfolio />
              </PrivateRoute>
            }
          />
          <Route path="/" element={<LandingPage />} />
          <Route path="/stocks" element={<StockTracker />} />
        </Routes>
      </div>
    </Router>
    <Analytics />
    </>
  );
}

export default App;