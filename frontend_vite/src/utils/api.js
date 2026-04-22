import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json'
  }
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
}, (error) => {
  return Promise.reject(error);
});

// Handle token expiration and invalid tokens
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 422) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);


export const fetchPortfolio = async () => {
  const response = await api.get('/api/portfolio');
  return response.data;
};

export const executeTrade = async (tradeData) => {
  const response = await api.post('/api/trade', tradeData);
  return response.data;
};

export const fetchDashboardData = async (symbol) => {
  const response = await api.get(`/api/dashboard/${symbol}`);
  return response.data;
};

export const fetchComparisonData = async (symbol, period) => {
  const response = await api.get(`/api/comparison/${symbol}?period=${period}`);
  return response.data;
};

export const searchSymbols = async (query) => {
  const response = await api.get(`/api/search?q=${query}`);
  return response.data;
};

export const login = async (credentials) => {
  const response = await api.post('/api/login', credentials);
  const token = response.data.token || response.data.access_token;
  if (token) {
    localStorage.setItem('token', token);
  }
  return response.data;
};

export const register = async (userData) => {
  const response = await api.post('/api/register', userData);
  return response.data;
};

// ===========================
// TRANSACTION HISTORY
// ===========================

export const fetchTransactions = async (page = 1, perPage = 50) => {
  const response = await api.get(`/api/transactions?page=${page}&per_page=${perPage}`);
  return response.data;
};

// ===========================
// WATCHLIST
// ===========================

export const fetchWatchlist = async () => {
  const response = await api.get('/api/watchlist');
  return response.data;
};

export const addToWatchlist = async (symbol) => {
  const response = await api.post('/api/watchlist', { symbol });
  return response.data;
};

export const removeFromWatchlist = async (symbol) => {
  const response = await api.delete(`/api/watchlist/${symbol}`);
  return response.data;
};

// ===========================
// PORTFOLIO HISTORY
// ===========================

export const fetchPortfolioHistory = async () => {
  const response = await api.get('/api/portfolio/history');
  return response.data;
};