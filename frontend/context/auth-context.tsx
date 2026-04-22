"use client";

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { jwtDecode } from 'jwt-decode';

interface User {
  email: string;
}

interface AuthContextType {
  token: string | null;
  user: User | null;
  isAuthenticated: boolean;
  login: (newToken: string, userData?: User) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<string | null>(() => {
    if (typeof window !== 'undefined') {
      return sessionStorage.getItem('token');
    }
    return null;
  });
  const [user, setUser] = useState<User | null>(() => {
    if (typeof window !== 'undefined') {
      const stored = sessionStorage.getItem('token');
      if (stored) {
        try {
          const decoded = jwtDecode<{ sub: string; exp: number }>(stored);
          if (decoded.exp * 1000 > Date.now()) return { email: decoded.sub };
        } catch {}
      }
    }
    return null;
  });
  const [isAuthenticated, setIsAuthenticated] = useState(() => {
    if (typeof window !== 'undefined') {
      const stored = sessionStorage.getItem('token');
      if (stored) {
        try {
          const decoded = jwtDecode<{ sub: string; exp: number }>(stored);
          if (decoded.exp * 1000 > Date.now()) return true;
        } catch {}
      }
    }
    return false;
  });

  // Parse user info from JWT and check expiry
  const processToken = useCallback((jwt: string | null) => {
    if (!jwt) {
      setUser(null);
      setIsAuthenticated(false);
      return false;
    }

    try {
      const decoded = jwtDecode<{ sub: string; exp: number }>(jwt);
      const currentTime = Date.now() / 1000;

      if (decoded.exp < currentTime) {
        // Token expired
      sessionStorage.removeItem('token');
      setToken(null);
      setUser(null);
      setIsAuthenticated(false);
      return false;
    }

    setUser({ email: decoded.sub });
    setIsAuthenticated(true);
    return true;
  } catch {
    sessionStorage.removeItem('token');
      setToken(null);
      setUser(null);
      setIsAuthenticated(false);
      return false;
    }
  }, []);

  // On mount, validate any existing token
  useEffect(() => {
    processToken(token);
  }, [token, processToken]);

  const login = useCallback((newToken: string, userData?: User) => {
    sessionStorage.setItem('token', newToken);
    setToken(newToken);
    if (userData) {
      setUser(userData);
    }
    setIsAuthenticated(true);
  }, []);

  const logout = useCallback(() => {
    sessionStorage.removeItem('token');
    setToken(null);
    setUser(null);
    setIsAuthenticated(false);
  }, []);

  const value: AuthContextType = {
    token,
    user,
    isAuthenticated,
    login,
    logout,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

export default AuthContext;
