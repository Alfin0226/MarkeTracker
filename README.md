# MarketTracker

## Link
https://marketracker.vercel.app/

MarketTracker is a beginner-friendly web-based investment portfolio tracking platform. It helps users monitor and manage investments across stocks, cryptocurrencies, and indices in real-time. The platform also provides a simulated trading environment, allowing users to learn and practice investing without any financial risk.

---

## Features

### 🔑 Account Management
- Secure user authentication (JWT-based) for creating and managing accounts.
- Personalized dashboards for tracking portfolio performance.

### 📈 Real-Time Market Data
- Fetches live stock, cryptocurrency, and index data using Yahoo Finance and other APIs.
- Provides historical and intraday data for analyzing trends.
- S&P 500 comparison for any stock, with dynamic chart switching between price and percent change.

### 🔍 Symbol Search & Suggestions
- Landing search page with instant symbol suggestions and autocomplete.
- In-dashboard search bar for quick symbol switching.

### 📊 Portfolio Tracking and Visualization
- Interactive Chart.js charts for tracking individual assets and overall portfolio performance.
- Dynamic period selection (1d, 5d, 1mo, etc.) for all charts.
- Separate pages for easy navigation between stocks, crypto, and indices.

### 💰 Simulated Trading Environment
- Users are provided with a $1 million virtual balance to simulate trades.
- Buy and sell stocks or crypto assets risk-free and track simulated portfolio performance.

### 🤖 Machine Learning Analysis (Educational)
- An scikit-learn model that predicts the company closing price based on different datasets

---

## Project Structure

1. **Search Page**: Lets users search for a stock symbol and jump to the dashboard (Only Support companies that are listed in NYSE & NASDAQ).
2. **Dashboard**: Displays an overview of the user's portfolio and detailed company/asset info, with dynamic charts and S&P 500 comparison.
3. **Demo Tracking Pages**:
   - Real-time price tracking that support stocks,indices, and cryptocurrencies 
4. **Simulation Portfolio Page**: 
   - Interface for performing simulated trades.
   - Tracks and calculates the performance of virtual investments.

---

## Tech Stack

### **Frontend**
- **Framework**: React.js + Vite
- **Design**: Custom CSS (with some Bootstrap utility classes)
- **Charting**: Chart.js for interactive and dynamic charts

### **Backend**
- **Framework**: Python Flask
- **APIs**: Yahoo Finance (yfinance), with plans for CoinGecko and others
- **Database**: PostgreSQL  for storing user data and trade history

### **Hosting and Deployment**
- Hosted on Vercel (frontend and backend)
- Backend API endpoints deployed as serverless functions

---
