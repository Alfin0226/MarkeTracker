import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Navbar from './components/Navbar';
import Login from './pages/Login';
import Register from './pages/Register';
import StockTracker from './pages/StockTracker';
import Portfolio from './pages/Portfolio';
import PrivateRoute from './components/PrivateRoute';
import ErrorBoundary from './components/ErrorBoundary';
import { Analytics } from '@vercel/analytics/react';
import LandingPage from './pages/LandingPage';
import LandingSearchPage from './pages/LandingSearchPage';
import Dashboard from './pages/Dashboard';

function App() {
  return (<>
    <Router>
      <div className="App">
        <Navbar />
        <ErrorBoundary>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route
              path="/dashboard/:symbol"
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
            <Route path="/dashboard" element={<LandingSearchPage />} />
            <Route path="/stocks" element={<StockTracker />} />
          </Routes>
        </ErrorBoundary>
      </div>
    </Router>
    <Analytics />
  </>
  );
}

export default App;