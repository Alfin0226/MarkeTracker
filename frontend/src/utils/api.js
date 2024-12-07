import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  console.log('Request URL:', config.url);
  console.log('Token present:', !!token);
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
    console.log('Authorization header set:', config.headers.Authorization);
  }
  return config;
});

// Add response interceptor to handle token expiration and invalid tokens
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 422) {
      console.log('Token error:', error.response?.data);
      // Clear token and redirect to login for any token-related errors
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const fetchStockData = async (symbol, period, interval) => {
  try {
    const response = await api.get(`/stock/${symbol}?period=${period}&interval=${interval}`);
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const fetchPortfolio = async () => {
  try {
    const response = await api.get('/api/portfolio');
    console.log('Portfolio API response:', response.data);
    return response.data;
  } catch (error) {
    console.error('Portfolio API error:', error.response?.data || error);
    throw error;
  }
};

export const executeTrade = async (tradeData) => {
  try {
    const response = await api.post('/trade', tradeData);
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const login = async (credentials) => {
  try {
    const response = await api.post('/api/login', credentials);
    return response.data;
  } catch (error) {
    throw error;
  }
};

export const register = async (userData) => {
  try {
    const response = await api.post('/api/register', userData);
    return response.data;
  } catch (error) {
    throw error;
  }
};