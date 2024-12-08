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
  console.log('Request URL:', config.url);
  console.log('Token present:', !!token);
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
    console.log('Authorization header set:', config.headers.Authorization);
  }
  return config;
}, (error) => {
  console.error('Request interceptor error:', error);
  return Promise.reject(error);
});

// Add response interceptor to handle token expiration and invalid tokens
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', {
      status: error.response?.status,
      data: error.response?.data,
      config: error.config
    });
    
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
    const response = await api.get(`/api/stock/${symbol}?period=${period}&interval=${interval}`);
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
    const response = await api.post('/api/trade', tradeData);
    return response.data;
  } catch (error) {
    console.error('Trade error:', error.response?.data || error);
    throw error;
  }
};

export const login = async (credentials) => {
  try {
    console.log('Attempting login with:', credentials.email);
    const response = await api.post('/api/login', credentials);
    console.log('Login response:', response.data);
    
    // Handle both token and access_token formats
    const token = response.data.token || response.data.access_token;
    if (token) {
      localStorage.setItem('token', token);
      console.log('Token stored successfully');
    } else {
      console.error('No token in response:', response.data);
    }
    return response.data;
  } catch (error) {
    console.error('Login error:', error.response?.data || error);
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