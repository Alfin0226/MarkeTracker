from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import yfinance as yf
from datetime import datetime, timedelta, timezone
import bcrypt
from database import db
from models import User, Portfolio, Transaction, Company, Watchlist
from sqlalchemy import or_, case
import os
from dotenv import load_dotenv
import requests
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Define allowed origins
origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "https://marketracker.vercel.app",
    "https://backend-theta-roan-61.vercel.app",
    r"^https://marketracker.*\.vercel\.app$",
]

# Add DATAHANDLE_URL to origins if it exists
datahandle_url = os.getenv('DATAHANDLE_URL')
if datahandle_url:
    origins.append(datahandle_url)

# Configure CORS
CORS(app, resources={
    r"/api/*": {
        "origins": origins,
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Configuration
database_url = os.getenv('POSTGRES_URL')

if not database_url:
    database_url = os.getenv('DATABASE_URI')
    if not database_url:
        raise RuntimeError('No database connection string found. Set either POSTGRES_URL or DATABASE_URI environment variable')

# Update database URL format
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql+psycopg://', 1)
elif not database_url.startswith('postgresql+psycopg://'):
    if database_url.startswith('sqlite:'):
        pass  # Keep sqlite URLs as is
    else:
        database_url = f"postgresql+psycopg://{database_url.split('://', 1)[1]}"

# Configure database connection
engine_options = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
    'pool_size': 20,
    'max_overflow': 5,
}

# Add SSL settings only for production/Vercel environment
if 'POSTGRES_URL' in os.environ:
    engine_options['connect_args'] = {
        'sslmode': 'require'
    }

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = engine_options

app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')

if not app.config['JWT_SECRET_KEY']:
    raise RuntimeError('JWT_SECRET_KEY environment variable is not set')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(minutes=30)

# Initialize extensions
db.init_app(app)
jwt = JWTManager(app)

# Create database tables within the application context.
with app.app_context():
    db.create_all()

logger.info("Application initialized successfully")

# JWT error handlers
@jwt.invalid_token_loader
def invalid_token_callback(error):
    return jsonify({
        'error': 'Invalid token',
        'message': str(error)
    }), 422

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({
        'error': 'Token has expired',
        'message': 'Please log in again'
    }), 422

@jwt.unauthorized_loader
def unauthorized_callback(error):
    return jsonify({
        'error': 'Missing token',
        'message': str(error)
    }), 422

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    
    if not data or 'email' not in data or 'password' not in data:
        return jsonify({'error': 'Email and password are required'}), 400
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 400
    
    try:
        hashed_password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())
        
        new_user = User(
            email=data['email'],
            password=hashed_password,
            virtual_balance=1000000  # Starting balance of $1M
        )
        
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'message': 'User created successfully'}), 201
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Failed to create user'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    
    user = User.query.filter_by(email=data['email']).first()
    
    if user:
        try:
            password_matches = bcrypt.checkpw(data['password'].encode('utf-8'), user.password)
        except Exception as e:
            logger.error(f"Error checking password: {str(e)}")
            return jsonify({'error': 'Server error during login'}), 500
            
        if password_matches:
            wake_up_datahandle()
            
            access_token = create_access_token(
                identity=str(user.email)
            )
            return jsonify({
                'access_token': access_token,
                'user': {
                    'email': user.email,
                    'virtual_balance': user.virtual_balance
                }
            }), 200
    
    return jsonify({'error': 'Invalid credentials'}), 401

def wake_up_datahandle():
    try:
        if DATAHANDLE_URL:
            requests.get(f"{DATAHANDLE_URL}/wakeup", timeout=5)
    except Exception as e:
        logger.warning(f"Could not wake datahandle service: {e}")


@app.route('/api/portfolio', methods=['GET'])
@jwt_required()
def get_portfolio():
    try:
        email = get_jwt_identity()
        
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        portfolio_items = Portfolio.query.filter_by(user_id=user.id).all()
        
        portfolio_data = []
        total_value = user.virtual_balance
        
        for item in portfolio_items:
            try:
                current_price = get_stock_price(item.symbol)
                
                value = item.shares * current_price
                total_value += value
                gain_loss = value - (item.average_price * item.shares)
                
                portfolio_data.append({
                    'symbol': item.symbol,
                    'shares': item.shares,
                    'avg_price': float(item.average_price),
                    'current_price': float(current_price),
                    'value': float(value),
                    'gain_loss': float(gain_loss)
                })
                
            except ValueError:
                portfolio_data.append({
                    'symbol': item.symbol,
                    'shares': item.shares,
                    'avg_price': float(item.average_price),
                    'current_price': 'N/A',
                    'value': 'N/A',
                    'gain_loss': 'N/A'
                })
            except Exception as e:
                logger.error(f"Error processing portfolio item {item.symbol}: {str(e)}")
                continue
        
        response_data = {
            'portfolio': portfolio_data,
            'total_value': float(total_value),
            'cash_balance': float(user.virtual_balance),
            'created_at': user.created_at.isoformat() if user.created_at else None
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Portfolio error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search_companies():
    query = request.args.get('q', '')
    if not query:
        return jsonify([])

    starts_with_term = f"{query}%"
    contains_term = f"%{query}%"

    # Prioritize matches that start with the query
    relevance_ordering = case(
        (Company.symbol.ilike(starts_with_term), 1),
        (Company.name.ilike(starts_with_term), 2),
        (Company.symbol.ilike(contains_term), 3),
        (Company.name.ilike(contains_term), 4),
        else_=5 
    )

    results = Company.query.filter(
        or_(
            Company.symbol.ilike(contains_term),
            Company.name.ilike(contains_term)
        )
    ).order_by(relevance_ordering, Company.name).limit(10).all()
    
    companies = [{'symbol': c.symbol, 'name': c.name} for c in results]
    
    return jsonify(companies)

def get_stock_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        
        price = stock.info.get('regularMarketPrice')
        if price:
            return price
            
        price = stock.info.get('currentPrice')
        if price:
            return price
            
        hist = stock.history(period='1d')
        if not hist.empty and 'Close' in hist.columns:
            return float(hist['Close'].iloc[-1])
            
        fast_info = stock.fast_info
        if hasattr(fast_info, 'last_price') and fast_info.last_price:
            return float(fast_info.last_price)
            
        raise ValueError(f"Could not fetch price for {symbol} using any method")
        
    except Exception as e:
        raise ValueError(f"Failed to fetch price for {symbol}: {str(e)}")

@app.route('/api/trade', methods=['POST'])
@jwt_required()
def trade():
    try:
        email = get_jwt_identity()
        
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        if 'symbol' not in data or 'action' not in data or 'shares' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
            
        try:
            shares = int(data['shares'])
            if shares <= 0:
                return jsonify({'error': 'Number of shares must be positive'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid number of shares'}), 400
        
        try:
            current_price = get_stock_price(data['symbol'])
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception:
            return jsonify({'error': 'Failed to fetch stock price'}), 500
            
        total_cost = current_price * shares
        
        # Start transaction
        db.session.begin_nested()
        
        try:
            if data['action'] == 'buy':
                if total_cost > user.virtual_balance:
                    db.session.rollback()
                    return jsonify({
                        'error': 'Insufficient funds',
                        'required': total_cost,
                        'available': user.virtual_balance
                    }), 400
                
                portfolio_item = Portfolio.query.filter_by(
                    user_id=user.id, symbol=data['symbol']).first()
                
                if portfolio_item:
                    new_shares = portfolio_item.shares + shares
                    new_avg_price = ((portfolio_item.shares * portfolio_item.average_price) + 
                                   (shares * current_price)) / new_shares
                    portfolio_item.shares = new_shares
                    portfolio_item.average_price = new_avg_price
                else:
                    portfolio_item = Portfolio(
                        user_id=user.id,
                        symbol=data['symbol'],
                        shares=shares,
                        average_price=current_price
                    )
                    db.session.add(portfolio_item)
                
                user.virtual_balance -= total_cost
                
            elif data['action'] == 'sell':
                portfolio_item = Portfolio.query.filter_by(
                    user_id=user.id, symbol=data['symbol']).first()
                
                if not portfolio_item:
                    db.session.rollback()
                    return jsonify({'error': 'You do not own this stock'}), 400
                    
                if shares > portfolio_item.shares:
                    db.session.rollback()
                    return jsonify({'error': 'Not enough shares to sell'}), 400
                    
                user.virtual_balance += total_cost
                
                if shares == portfolio_item.shares:
                    db.session.delete(portfolio_item)
                else:
                    portfolio_item.shares -= shares
            
            # Create transaction record
            transaction = Transaction(
                user_id=user.id,
                symbol=data['symbol'],
                shares=shares,
                price=current_price,
                action=data['action'],
                timestamp=datetime.now(timezone.utc)
            )
            
            db.session.add(transaction)
            
            try:
                db.session.commit()
                
                return jsonify({
                    'message': f'Successfully executed {data["action"]} order',
                    'transaction': {
                        'symbol': data['symbol'],
                        'shares': shares,
                        'price': current_price,
                        'total': total_cost,
                        'new_balance': user.virtual_balance
                    }
                })
                
            except Exception as e:
                db.session.rollback()
                logger.error(f"Error committing transaction: {str(e)}")
                return jsonify({'error': 'Database error while executing trade'}), 500
                
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error during trade execution: {str(e)}")
            return jsonify({'error': 'Failed to execute trade'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error in trade endpoint: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/api/test-auth', methods=['GET'])
@jwt_required()
def test_auth():
    user_id = get_jwt_identity()
    return jsonify({
        'message': 'Authentication successful',
        'user_id': user_id
    })


# Set this to your deployed backend-datahandle URL
DATAHANDLE_URL = os.getenv('DATAHANDLE_URL')

import time

def retry_request(url, params=None, max_retries=3, timeout=10):
    """Make a GET request with retry logic and exponential backoff."""
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            return response.json(), response.status_code
        except requests.exceptions.Timeout:
            last_error = "Request timed out"
            logger.warning(f"Datahandle request timed out (attempt {attempt + 1}/{max_retries}): {url}")
        except requests.exceptions.ConnectionError:
            last_error = "Could not connect to data service"
            logger.warning(f"Datahandle connection error (attempt {attempt + 1}/{max_retries}): {url}")
        except Exception as e:
            last_error = str(e)
            logger.error(f"Datahandle unexpected error (attempt {attempt + 1}/{max_retries}): {e}")
        
        # Exponential backoff: 1s, 2s between retries
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    
    return {"error": f"Data service unavailable: {last_error}"}, 503

def get_comparison_data(symbol, period="1y"):
    url = f"{DATAHANDLE_URL}/api/comparison/{symbol}"
    params = {"period": period}
    return retry_request(url, params=params)
    
def get_dashboard_data(symbol):
    url = f"{DATAHANDLE_URL}/api/dashboard/{symbol}"
    return retry_request(url)

@app.route('/api/comparison/<symbol>', methods=['GET'])
def proxy_comparison(symbol):
    period = request.args.get('period', '1y')
    data, status = get_comparison_data(symbol, period)
    return jsonify(data), status

@app.route('/api/dashboard/<symbol>', methods=['GET'])
def proxy_dashboard(symbol):
    data, status = get_dashboard_data(symbol)
    return jsonify(data), status

# ===========================
# TRANSACTION HISTORY
# ===========================

@app.route('/api/transactions', methods=['GET'])
@jwt_required()
def get_transactions():
    try:
        email = get_jwt_identity()
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        per_page = min(per_page, 100)  # Cap at 100

        transactions = Transaction.query.filter_by(user_id=user.id) \
            .order_by(Transaction.timestamp.desc()) \
            .paginate(page=page, per_page=per_page, error_out=False)

        return jsonify({
            'transactions': [{
                'id': t.id,
                'symbol': t.symbol,
                'shares': t.shares,
                'price': float(t.price),
                'action': t.action,
                'total': float(t.price * t.shares),
                'timestamp': t.timestamp.isoformat() if t.timestamp else None
            } for t in transactions.items],
            'total': transactions.total,
            'pages': transactions.pages,
            'current_page': transactions.page
        })
    except Exception as e:
        logger.error(f"Error fetching transactions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/indices', methods=['GET'])
def get_indices():
    indices = {
        'Dow Jones': '^DJI',
        'Nasdaq 100': 'NQ=F',
        'S&P 500': '^GSPC'
    }
    result = []
    
    try:
        for name, symbol in indices.items():
            stock = yf.Ticker(symbol)
            info = stock.fast_info
            
            try:
                hist = stock.history(period='1d', interval='15m')
                if len(hist) >= 2:
                    current_price = float(hist['Close'].iloc[-1])
                    
                    try:
                        previous_close = float(info.previous_close)
                    except Exception:
                        previous_close = float(hist['Close'].iloc[0])
                        
                    change = current_price - previous_close
                    change_percent = (change / previous_close) * 100 if previous_close else 0
                    history = [
                        {
                            'time': int(index.timestamp()),
                            'open': float(row['Open']),
                            'high': float(row['High']),
                            'low': float(row['Low']),
                            'close': float(row['Close'])
                        }
                        for index, row in hist.iterrows() if not __import__('pandas').isna(row['Close'])
                    ]
                elif not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                    try:
                        previous_close = float(info.previous_close)
                    except Exception:
                        previous_close = current_price
                        
                    change = current_price - previous_close
                    change_percent = (change / previous_close) * 100 if previous_close else 0
                    history = [
                        {
                            'time': int(index.timestamp()),
                            'open': float(row['Open']),
                            'high': float(row['High']),
                            'low': float(row['Low']),
                            'close': float(row['Close'])
                        }
                        for index, row in hist.iterrows() if not __import__('pandas').isna(row['Close'])
                    ]
                else:
                    raise Exception("Empty history")
            except Exception:
                try:
                    info = stock.fast_info
                    current_price = float(info.last_price)
                    previous_close = float(info.previous_close)
                    change = current_price - previous_close
                    change_percent = (change / previous_close) * 100 if previous_close else 0
                    history = []
                except Exception:
                    # Fallback if fast_info fails
                    current_price = float(stock.info.get('regularMarketPrice') or stock.history(period='1d')['Close'].iloc[-1])
                    previous_close = float(stock.info.get('previousClose') or current_price)
                    change = current_price - previous_close
                    change_percent = (change / previous_close) * 100 if previous_close else 0
                    history = []
            
            result.append({
                'name': name,
                'symbol': symbol,
                'price': current_price,
                'change': change,
                'changePercent': change_percent,
                'history': history
            })
            
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error fetching indices: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ===========================
# WATCHLIST
# ===========================

@app.route('/api/watchlist', methods=['GET'])
@jwt_required()
def get_watchlist():
    try:
        email = get_jwt_identity()
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        items = Watchlist.query.filter_by(user_id=user.id) \
            .order_by(Watchlist.added_at.desc()).all()

        watchlist_data = []
        for item in items:
            entry = {
                'symbol': item.symbol,
                'added_at': item.added_at.isoformat() if item.added_at else None,
            }
            try:
                company = Company.query.filter_by(symbol=item.symbol).first()
                if company:
                    entry['name'] = company.name
                    
                stock = yf.Ticker(item.symbol)
                info = stock.fast_info
                
                try:
                    hist = stock.history(period='5d')
                    if len(hist) >= 2:
                        current_price = float(hist['Close'].iloc[-1])
                        previous_close = float(hist['Close'].iloc[-2])
                        change = current_price - previous_close
                        change_percent = (change / previous_close) * 100
                    elif not hist.empty:
                        current_price = float(hist['Close'].iloc[-1])
                        previous_close = current_price
                        change = 0
                        change_percent = 0
                    else:
                        raise Exception("Empty history")
                except Exception:
                    try:
                        info = stock.fast_info
                        current_price = float(info.last_price)
                        previous_close = float(info.previous_close)
                        change = current_price - previous_close
                        change_percent = (change / previous_close) * 100 if previous_close else 0
                    except Exception:
                        current_price = float(get_stock_price(item.symbol))
                        previous_close = current_price
                        change = 0
                        change_percent = 0

                entry['current_price'] = current_price
                entry['change'] = change
                entry['change_percent'] = change_percent
                
                if 'name' not in entry:
                    try:
                        entry['name'] = stock.info.get('shortName', item.symbol)
                    except Exception:
                        entry['name'] = item.symbol
                        
            except Exception as e:
                logger.warning(f"Error fetching extended data for {item.symbol}: {e}")
                entry['current_price'] = None
                entry['change'] = 0
                entry['change_percent'] = 0
                if 'name' not in entry:
                    entry['name'] = item.symbol
                    
            watchlist_data.append(entry)

        return jsonify({'watchlist': watchlist_data})
    except Exception as e:
        logger.error(f"Error fetching watchlist: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/watchlist', methods=['POST'])
@jwt_required()
def add_to_watchlist():
    try:
        email = get_jwt_identity()
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        data = request.get_json()
        symbol = data.get('symbol', '').upper().strip()
        if not symbol:
            return jsonify({'error': 'Symbol is required'}), 400

        existing = Watchlist.query.filter_by(user_id=user.id, symbol=symbol).first()
        if existing:
            return jsonify({'error': 'Already in watchlist'}), 400

        entry = Watchlist(user_id=user.id, symbol=symbol)
        db.session.add(entry)
        db.session.commit()

        return jsonify({'message': f'{symbol} added to watchlist'}), 201
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error adding to watchlist: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/watchlist/<symbol>', methods=['DELETE'])
@jwt_required()
def remove_from_watchlist(symbol):
    try:
        email = get_jwt_identity()
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        entry = Watchlist.query.filter_by(user_id=user.id, symbol=symbol.upper()).first()
        if not entry:
            return jsonify({'error': 'Not in watchlist'}), 404

        db.session.delete(entry)
        db.session.commit()

        return jsonify({'message': f'{symbol} removed from watchlist'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error removing from watchlist: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ===========================
# PORTFOLIO HISTORY
# ===========================

@app.route('/api/portfolio/history', methods=['GET'])
@jwt_required()
def get_portfolio_history():
    """Compute approximate weekly portfolio value from transaction history and yfinance weekly close."""
    try:
        email = get_jwt_identity()
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404

        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        
        import datetime
        from collections import defaultdict
        import yfinance as yf
        import pandas as pd

        today = datetime.date.today()
        if end_date_str:
            end_date = datetime.datetime.fromisoformat(end_date_str.split('T')[0]).date()
        else:
            end_date = today

        if start_date_str:
            start_date = datetime.datetime.fromisoformat(start_date_str.split('T')[0]).date()
        else:
            start_date = end_date - datetime.timedelta(days=90)

        def get_week_start(d):
            return d - datetime.timedelta(days=d.weekday())

        start_week = get_week_start(start_date)
        end_week = get_week_start(end_date)
        
        weeks = []
        current_w = start_week
        while current_w <= end_week:
            weeks.append(current_w)
            current_w += datetime.timedelta(days=7)

        transactions = Transaction.query.filter_by(user_id=user.id) \
            .order_by(Transaction.timestamp.asc()).all()

        unique_symbols = list(set([t.symbol for t in transactions]))
        
        # Download daily prices up to end_week + 7 days
        download_end = end_week + datetime.timedelta(days=8)
        download_start = start_week - datetime.timedelta(days=7) 
        
        closes_df = pd.DataFrame()
        if unique_symbols:
            try:
                if len(unique_symbols) == 1:
                    data = yf.download(unique_symbols[0], start=download_start, end=download_end, interval="1d", progress=False)
                    if not data.empty and 'Close' in data.columns:
                        closes_df = data[['Close']].copy()
                        closes_df.columns = [unique_symbols[0]]
                else:
                    data = yf.download(unique_symbols, start=download_start, end=download_end, interval="1d", progress=False)
                    if not data.empty and 'Close' in data.columns:
                        closes_df = data['Close'].copy()
                        if isinstance(closes_df, pd.Series):
                            closes_df = closes_df.to_frame()
            except Exception as e:
                logger.error(f"Error downloading yf data: {e}")

        # ensure datetime index is naive date
        if not closes_df.empty:
            closes_df.index = closes_df.index.tz_localize(None).date

        holdings = defaultdict(int)  
        cash = 1000000.0  
        tx_idx = 0
        num_tx = len(transactions)

        weekly_snapshots = []
        if not weeks:
            return jsonify({'history': []})

        for w in weeks:
            # Friday as last trading day of the week, or up to Sunday
            w_end_date = w + datetime.timedelta(days=6) # Sunday
            w_tz_aware_end = datetime.datetime.combine(w_end_date + datetime.timedelta(days=1), datetime.time.min).replace(tzinfo=datetime.timezone.utc)
            
            while tx_idx < num_tx:
                t = transactions[tx_idx]
                t_stamp = t.timestamp.replace(tzinfo=datetime.timezone.utc) if t.timestamp.tzinfo is None else t.timestamp
                if t_stamp >= w_tz_aware_end:
                    break
                
                if t.action == 'buy':
                    holdings[t.symbol] += t.shares
                    cash -= t.price * t.shares
                elif t.action == 'sell':
                    holdings[t.symbol] -= t.shares
                    cash += t.price * t.shares
                    if holdings[t.symbol] <= 0:
                        del holdings[t.symbol]
                tx_idx += 1

            stock_value = 0
            # Get latest available prices for this week
            if not closes_df.empty:
                valid_dates = closes_df.index[closes_df.index <= w_end_date]
                if len(valid_dates) > 0:
                    last_trading_day = valid_dates.max()
                    day_prices = closes_df.loc[last_trading_day]
                    
                    for sym, shares in holdings.items():
                        if shares > 0 and sym in day_prices and pd.notna(day_prices[sym]):
                            stock_value += day_prices[sym] * shares
                else:
                    # fallback to cached prices if yfinance misses data
                    for sym, shares in holdings.items():
                        if shares > 0:
                            try:
                                stock_value += get_stock_price(sym) * shares
                            except:
                                pass

            weekly_snapshots.append({
                'date': w_end_date.isoformat(),
                'cash': round(cash, 2),
                'stock_value': round(stock_value, 2),
                'total_value': round(cash + stock_value, 2)
            })

        return jsonify({'history': weekly_snapshots})
    except Exception as e:
        logger.error(f"Error computing portfolio history: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
