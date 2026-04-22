# MarketTracker

## Link

https://marketracker.vercel.app/

MarketTracker is a web-based investment portfolio tracking platform. It allows users to monitor stocks, ETFs, and indices with real-time market data, execute simulated trades using a virtual balance, and analyze company financials -- all without any real financial risk.

---

## Features

### Account Management
- JWT-based authentication with secure registration and login.
- Personalized user accounts with virtual balance tracking.

### Symbol Search
- Landing page with instant search and autocomplete for stock symbols.
- In-dashboard search bar for quick symbol switching.
- Supports companies listed on NYSE and NASDAQ.

### Dashboard
- Detailed company overview including sector, industry, market cap, P/E ratio, EPS, dividend yield, and analyst ratings.
- Quarterly income statement breakdown displayed in a structured grid.
- Dynamic chart comparing any stock's performance against the S&P 500, with switchable views between price and percent change.
- Selectable time periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, and max.

### Machine Learning Price Forecast
- A scikit-learn linear regression model trained on two years of historical data (Open, High, Low, Close, Volume) to predict the next-day closing price.
- Forecast results are displayed directly on the dashboard.

### Simulated Trading
- Each user starts with a $1,000,000 virtual balance.
- Buy and sell stocks at live market prices.
- Weighted average cost tracking for portfolio positions.
- Full transaction history with pagination.

### Portfolio
- Real-time portfolio valuation using current market prices.
- Per-position breakdown showing shares, average cost, current price, market value, and gain/loss.
- Portfolio history computed from transaction records to chart total value over time.

### Watchlist
- Add and remove symbols to a personal watchlist.
- Live price display for all watched symbols.

---

## Project Structure

```text
MarkeTracker/
  frontend/           Next.js 16 App Router application
  backend/            Flask API server (auth, portfolio, trading, watchlist, proxy)
  backend-datahandle/ Flask data service (market data, comparison charts, ML forecast)
```

### Frontend Pages

| Route                | Page               | Description                                      |
|----------------------|--------------------|--------------------------------------------------|
| `/`                  | Landing Page       | Introduction and call to action                  |
| `/dashboard`         | Search Page        | Symbol search with autocomplete                  |
| `/dashboard/[symbol]`| Dashboard          | Company info, charts, income grid, trade form    |
| `/portfolio`         | Portfolio           | Holdings overview with summary and trade form    |
| `/transactions`      | Transaction History | Paginated list of all executed trades             |
| `/watchlist`         | Watchlist           | Tracked symbols with live prices                 |
| `/login`             | Login               | User authentication                              |
| `/register`          | Register            | Account creation                                 |

### Backend API Endpoints (backend/)

| Method | Endpoint                     | Auth | Description                          |
|--------|------------------------------|------|--------------------------------------|
| POST   | `/api/register`              | No   | Create a new user account            |
| POST   | `/api/login`                 | No   | Authenticate and receive JWT         |
| GET    | `/api/search?q=`             | No   | Search companies by symbol or name   |
| GET    | `/api/portfolio`             | Yes  | Retrieve current portfolio           |
| POST   | `/api/trade`                 | Yes  | Execute a buy or sell trade          |
| GET    | `/api/transactions`          | Yes  | Paginated transaction history        |
| GET    | `/api/watchlist`             | Yes  | Get watchlist items                  |
| POST   | `/api/watchlist`             | Yes  | Add symbol to watchlist              |
| DELETE | `/api/watchlist/<symbol>`    | Yes  | Remove symbol from watchlist         |
| GET    | `/api/portfolio/history`     | Yes  | Daily portfolio value snapshots      |
| GET    | `/api/dashboard/<symbol>`    | No   | Proxied to backend-datahandle        |
| GET    | `/api/comparison/<symbol>`   | No   | Proxied to backend-datahandle        |

### Data Service Endpoints (backend-datahandle/)

| Method | Endpoint                     | Description                                      |
|--------|------------------------------|--------------------------------------------------|
| GET    | `/api/dashboard/<symbol>`    | Company info, income grid, and ML price forecast  |
| GET    | `/api/comparison/<symbol>`   | Stock vs S&P 500 performance comparison           |

---

## Tech Stack

### Frontend
- **Framework**: Next.js 16 (React 19)
- **Routing**: Next.js App Router
- **HTTP Client**: Axios
- **Charting**: TradingView Lightweight Charts & Chart.js
- **Styling**: Tailwind CSS & shadcn/ui
- **Icons**: Lucide React
- **Analytics**: Google Analytics

### Backend
- **Framework**: Python Flask
- **Authentication**: Flask-JWT-Extended with bcrypt password hashing
- **Database**: PostgreSQL via Flask-SQLAlchemy (psycopg driver)
- **Market Data**: yfinance
- **Retry Logic**: Exponential backoff for data service requests

### Data Service (backend-datahandle)
- **Framework**: Python Flask with Flask-Caching
- **Data Processing**: pandas
- **Machine Learning**: scikit-learn (LinearRegression)
- **Market Data**: yfinance

### Hosting and Deployment
- **Frontend**: Vercel
- **Backend**: Vercel / Koyeb

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
