
"use client";

import React, { createContext, useContext, useState, useEffect, useCallback } from "react";
import { jwtDecode } from "jwt-decode";

interface User {
  email: string;
}

interface AuthContextType {
  token: string | null;
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (newToken: string, userData?: User) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

const SESSION_KEY = "token";

function getStoredToken(): string | null {
  if (typeof window === "undefined") return null;
  const sessionToken = sessionStorage.getItem(SESSION_KEY);
  if (sessionToken) return sessionToken;
  return null;
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<User | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

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
        sessionStorage.removeItem("token");
        setToken(null);
        setUser(null);
        setIsAuthenticated(false);
        return false;
      }

      setUser({ email: decoded.sub });
      setIsAuthenticated(true);
      return true;
    } catch {
      sessionStorage.removeItem("token");
      setToken(null);
      setUser(null);
      setIsAuthenticated(false);
      return false;
    }
  }, []);

  useEffect(() => {
    const stored = getStoredToken();
    processToken(stored);
    setIsLoading(false);
  }, [processToken]);

  const login = useCallback((newToken: string, userData?: User) => {
    sessionStorage.removeItem(SESSION_KEY);
    sessionStorage.setItem(SESSION_KEY, newToken);

    setToken(newToken);
    if (userData) {
      setUser(userData);
    }
    setIsAuthenticated(true);
  }, []);

  const logout = useCallback(() => {
    sessionStorage.removeItem(SESSION_KEY);
    setToken(null);
    setUser(null);
    setIsAuthenticated(false);
  }, []);

  const value: AuthContextType = {
    token,
    user,
    isAuthenticated,
    isLoading,
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
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}

export default AuthContext;

