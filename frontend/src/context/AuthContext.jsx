import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { jwtDecode } from 'jwt-decode';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
    const [token, setToken] = useState(() => localStorage.getItem('token'));
    const [user, setUser] = useState(null);
    const [isAuthenticated, setIsAuthenticated] = useState(false);

    // Parse user info from JWT and check expiry
    const processToken = useCallback((jwt) => {
        if (!jwt) {
            setUser(null);
            setIsAuthenticated(false);
            return false;
        }

        try {
            const decoded = jwtDecode(jwt);
            const currentTime = Date.now() / 1000;

            if (decoded.exp < currentTime) {
                // Token expired
                localStorage.removeItem('token');
                setToken(null);
                setUser(null);
                setIsAuthenticated(false);
                return false;
            }

            setUser({ email: decoded.sub });
            setIsAuthenticated(true);
            return true;
        } catch {
            // Invalid token
            localStorage.removeItem('token');
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

    const login = useCallback((newToken, userData) => {
        localStorage.setItem('token', newToken);
        setToken(newToken);
        if (userData) {
            setUser(userData);
        }
        setIsAuthenticated(true);
    }, []);

    const logout = useCallback(() => {
        localStorage.removeItem('token');
        setToken(null);
        setUser(null);
        setIsAuthenticated(false);
    }, []);

    const value = {
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
