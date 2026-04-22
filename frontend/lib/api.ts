import axios from 'axios';

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json'
  }
});

api.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    const token = sessionStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
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
      if (typeof window !== 'undefined') {
        sessionStorage.removeItem('token');
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);


export const fetchPortfolio = async () => {
  const response = await api.get('/api/portfolio');
  return response.data;
};

export const executeTrade = async (tradeData: {
  symbol: string;
  shares: number;
  action: string;
}) => {
  const response = await api.post('/api/trade', tradeData);
  return response.data;
};

export const fetchDashboardData = async (symbol: string) => {
  const response = await api.get(`/api/dashboard/${symbol}`);
  return response.data;
};

export const fetchComparisonData = async (symbol: string, period: string) => {
  const response = await api.get(`/api/comparison/${symbol}?period=${period}`);
  return response.data;
};

export const searchSymbols = async (query: string) => {
  const response = await api.get(`/api/search?q=${query}`);
  return response.data;
};

export const login = async (credentials: { email: string; password: string }) => {
  const response = await api.post('/api/login', credentials);
  const token = response.data.token || response.data.access_token;
  if (token && typeof window !== 'undefined') {
    sessionStorage.setItem('token', token);
  }
  return response.data;
};

export const register = async (userData: { email: string; password: string }) => {
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

export const addToWatchlist = async (symbol: string) => {
  const response = await api.post('/api/watchlist', { symbol });
  return response.data;
};

export const removeFromWatchlist = async (symbol: string) => {
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
