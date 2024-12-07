import React, { useEffect } from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Navbar from './components/Navbar';
import Login from './pages/Login';
import Register from './pages/Register';
import StockTracker from './pages/StockTracker';
import Portfolio from './pages/Portfolio';
import PrivateRoute from './components/PrivateRoute';

function App() {
  useEffect(() => {
    // Clear any existing tokens on app start
    localStorage.removeItem('token');
  }, []);

  return (
    <Router>
      <div className="App">
        <Navbar />
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route
            path="/stocks"
            element={
              <PrivateRoute>
                <StockTracker />
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
          <Route path="/" element={<Login />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;