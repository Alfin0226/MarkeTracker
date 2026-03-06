from flask import Flask, request, jsonify, g
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import yfinance as yf
from datetime import datetime, timedelta, timezone
import bcrypt
from database import db
from models import User, Portfolio, Transaction, Company
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
            access_token = create_access_token(identity=str(user.email))
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

@app.route('/api/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    try:
        period = request.args.get('period', '1d')
        interval = request.args.get('interval', '5m')
        stock = yf.Ticker(symbol)
        
        # Get historical data with interval
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            return jsonify({'error': 'No data available for this period'}), 404
        
        # Format data for frontend
        data = {
            'prices': hist['Close'].tolist(),
            'dates': hist.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'info': stock.info
        }
        
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
            'cash_balance': float(user.virtual_balance)
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
    """Get current stock price with multiple fallback methods"""
    try:
        stock = yf.Ticker(symbol)
        
        # Method 1: Try regular market price
        price = stock.info.get('regularMarketPrice')
        if price:
            return price
            
        # Method 2: Try current price
        price = stock.info.get('currentPrice')
        if price:
            return price
            
        # Method 3: Try last close price from history
        hist = stock.history(period='1d')
        if not hist.empty and 'Close' in hist.columns:
            return float(hist['Close'].iloc[-1])
            
        # Method 4: Try fast info
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

@app.route("/testingbackend")
def testingbackend():
    return("<h1>Testing Backend</h1>")

# Set this to your deployed backend-datahandle URL
DATAHANDLE_URL = os.getenv('DATAHANDLE_URL')

def get_comparison_data(symbol, period="1y"):
    url = f"{DATAHANDLE_URL}/api/comparison/{symbol}"
    params = {"period": period}
    try:
        response = requests.get(url, params=params, timeout=10)
        return response.json(), response.status_code
    except Exception as e:
        logger.error(f"Error contacting datahandle service: {e}")
        return {"error": "Data service unavailable"}, 503
    
def get_dashboard_data(symbol):
    url = f"{DATAHANDLE_URL}/api/dashboard/{symbol}"
    try:
        response = requests.get(url, timeout=10)
        return response.json(), response.status_code
    except Exception as e:
        logger.error(f"Error contacting datahandle service: {e}")
        return {"error": "Data service unavailable"}, 503

@app.route('/api/comparison/<symbol>', methods=['GET'])
def proxy_comparison(symbol):
    period = request.args.get('period', '1y')
    data, status = get_comparison_data(symbol, period)
    return jsonify(data), status

@app.route('/api/dashboard/<symbol>', methods=['GET'])
def proxy_dashboard(symbol):
    data, status = get_dashboard_data(symbol)
    return jsonify(data), status

if __name__ == '__main__':
    app.run(debug=True)
