# MarketTracker

## Link
https://marketracker.vercel.app/register

MarketTracker is a beginner-friendly web-based investment portfolio tracking platform. It helps users monitor and manage investments across stocks, cryptocurrencies, and indices in real-time. The platform also provides a simulated trading environment, allowing users to learn and practice investing without any financial risk.

---

## Features

### 🔑 Account Management
- Secure user authentication for creating and managing accounts.
- Personalized dashboards for tracking portfolio performance.

### 📈 Real-Time Market Data
- Fetches live stock, cryptocurrency, and index data using APIs like Alpha Vantage and CoinGecko.
- Provides historical data for analyzing trends.

### 📊 Portfolio Tracking and Visualization
- Interactive charts for tracking individual assets and overall portfolio performance.
- Separate pages for easy navigation between stocks, crypto, and indices.

### 💰 Simulated Trading Environment
- Users are provided with a $1 million virtual balance to simulate trades.
- Buy and sell stocks or crypto assets risk-free and track simulated portfolio performance.

### 🤖 AI-Powered Analysis (Planned)
- An AI chatbot that provides recommendations and analysis based on asset data.

---

## Project Structure

1. **Homepage**: Highlights platform features with a call-to-action for signup and login.
2. **Dashboard**: Displays an overview of the user's portfolio and links to individual asset pages.
3. **Stocks Pages (later indics and crypto pages)**:
   - Real-time price tracking.
   - Interactive charts for performance analysis.
4. **Simulation Market Page**: 
   - Interface for performing simulated trades.
   - Tracks and calculates the performance of virtual investments.

---

## Tech Stack

### **Frontend**
- **Framework**: React.js + Vite
- **Design**: Tailwind CSS or Bootstrap for a responsive UI.

### **Backend**
- **Framework**: Python Flask.
- **APIs**: Yahoo Finance for live market data.
- **Database**: PostgreSQL for storing user data and trade history.

### **Hosting and Deployment**
- Hosted on Vercel.
