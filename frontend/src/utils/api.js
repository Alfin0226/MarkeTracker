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
      // Clear token and redirect to login for token-related errors
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const fetchStockData = async (symbol, period, interval) => {
  const response = await api.get(`/api/stock/${symbol}?period=${period}&interval=${interval}`);
  return response.data;
};

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
  // Handle both token and access_token formats
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