from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import yfinance as yf
from datetime import datetime, timedelta, timezone
import bcrypt
from database import db
from models import User, Portfolio, Transaction
from sqlalchemy import text
import os
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/*": {  # All routes
        "origins": [
            "http://localhost:3000",
            "http://localhost:5173",
            "https://marketracker.vercel.app",
            "https://backend-theta-roan-61.vercel.app"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Authorization"],
        "supports_credentials": True,
        "max_age": 600  # Cache preflight requests for 10 minutes
    }
})

@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    allowed_origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "https://marketracker.vercel.app",
        "https://backend-theta-roan-61.vercel.app"
    ]
    
    if origin in allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Max-Age"] = "600"  # Cache preflight requests
        
        # Handle preflight requests
        if request.method == 'OPTIONS':
            return response
    return response

# Configuration
print("Starting application configuration...")
database_url = os.getenv('POSTGRES_URL')
print(f"POSTGRES_URL: {'set' if database_url else 'not set'}")

if not database_url:
    # Fallback to DATABASE_URI if POSTGRES_URL is not set
    database_url = os.getenv('DATABASE_URI')
    print(f"DATABASE_URI: {'set' if database_url else 'not set'}")
    if not database_url:
        raise RuntimeError('No database connection string found. Set either POSTGRES_URL or DATABASE_URI environment variable')

print(f"Database URL (masked): {database_url[:15]}...{database_url[-15:]}")

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
print(f"JWT_SECRET_KEY: {'set' if app.config['JWT_SECRET_KEY'] else 'not set'}")

if not app.config['JWT_SECRET_KEY']:
    raise RuntimeError('JWT_SECRET_KEY environment variable is not set')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(minutes=30)

print("Initializing database...")
try:
    # Initialize extensions
    db.init_app(app)
    print("Database initialization successful")
    
    # Test database connection
    with app.app_context():
        # Try to connect and create tables
        print("Testing database connection...")
        db.engine.connect()
        print("Database connection successful")
        
        print("Creating database tables...")
        db.create_all()
        print("Database tables created successfully")
        
        # Verify tables exist
        print("Verifying database tables...")
        inspector = db.inspect(db.engine)
        tables = inspector.get_table_names()
        print(f"Found tables: {tables}")
        
except Exception as e:
    print(f"Database Error: {str(e)}")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Args: {getattr(e, 'args', '')}")
    raise

# Initialize JWT manager
jwt = JWTManager(app)

# Add JWT error handlers
@jwt.invalid_token_loader
def invalid_token_callback(error):
    print(f"Invalid token error: {error}")
    return jsonify({
        'error': 'Invalid token',
        'message': str(error)
    }), 422

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    print(f"Expired token. Header: {jwt_header}, Payload: {jwt_payload}")
    return jsonify({
        'error': 'Token has expired',
        'message': 'Please log in again'
    }), 422

@jwt.unauthorized_loader
def unauthorized_callback(error):
    print(f"Missing token error: {error}")
    return jsonify({
        'error': 'Missing token',
        'message': str(error)
    }), 422

# Ensure database tables exist
def init_db():
    """Initialize the database and sequences"""
    with app.app_context():
        try:
            db.create_all()
            
            # Initialize sequences
            with db.engine.connect() as conn:
                # Initialize portfolio sequence
                result = conn.execute(text('SELECT MAX(id) FROM portfolio'))
                max_id = result.scalar() or 0
                conn.execute(text('ALTER TABLE portfolio ALTER COLUMN id DROP DEFAULT'))
                conn.execute(text('DROP SEQUENCE IF EXISTS portfolio_id_seq'))
                conn.execute(text(f'CREATE SEQUENCE portfolio_id_seq START WITH {max_id + 1}'))
                conn.execute(text('ALTER TABLE portfolio ALTER COLUMN id SET DEFAULT nextval(\'portfolio_id_seq\')'))
                print(f"Initialized portfolio sequence starting at {max_id + 1}")
                
                # Initialize transaction sequence
                result = conn.execute(text('SELECT MAX(id) FROM transaction'))
                max_id = result.scalar() or 0
                conn.execute(text('ALTER TABLE transaction ALTER COLUMN id DROP DEFAULT'))
                conn.execute(text('DROP SEQUENCE IF EXISTS transaction_id_seq'))
                conn.execute(text(f'CREATE SEQUENCE transaction_id_seq START WITH {max_id + 1}'))
                conn.execute(text('ALTER TABLE transaction ALTER COLUMN id SET DEFAULT nextval(\'transaction_id_seq\')'))
                print(f"Initialized transaction sequence starting at {max_id + 1}")
                
        except Exception as e:
            print(f"Error during database initialization: {str(e)}")
            if 'relation "transaction" does not exist' in str(e):
                # If table doesn't exist yet, create it first
                db.create_all()
                init_db()  # Try again after creating tables

# Initialize database and sequences
init_db()

@app.route('/api/register', methods=['POST'])
def register():
    print("Received registration request")
    data = request.get_json()
    print("Request data:", data)
    
    if not data or 'email' not in data or 'password' not in data:
        print("Missing email or password in request")
        return jsonify({'error': 'Email and password are required'}), 400
    
    if User.query.filter_by(email=data['email']).first():
        print(f"Email {data['email']} already exists")
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
        print(f"Successfully created user with email {data['email']}")
        return jsonify({'message': 'User created successfully'}), 201
    except Exception as e:
        print(f"Error creating user: {str(e)}")
        db.session.rollback()
        return jsonify({'error': 'Failed to create user'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    print(f"Login attempt for email: {data.get('email')}")
    
    user = User.query.filter_by(email=data['email']).first()
    print(f"User found: {user is not None}")
    
    if user:
        try:
            password_matches = bcrypt.checkpw(data['password'].encode('utf-8'), user.password)
            print(f"Password match: {password_matches}")
            
        except Exception as e:
            print(f"Error checking password: {str(e)}")
            return jsonify({'error': 'Server error during login'}), 500
            
        if password_matches:
            access_token = create_access_token(identity=str(user.email))
            return jsonify({
                'access_token': access_token,
                'user': {
                    'email': user.email,
                    'virtual_balance': user.virtual_balance
                }
            }), 200
    
    return jsonify({'error': 'Invalid credentials'}), 401

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
        print(f"Error fetching stock data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio', methods=['GET'])
@jwt_required()
def get_portfolio():
    try:
        # Get email from JWT token
        email = get_jwt_identity()
        print(f"Getting portfolio for email: {email}")
        
        # Find user by email
        user = User.query.filter_by(email=email).first()
        if not user:
            print(f"No user found for email: {email}")
            return jsonify({'error': 'User not found'}), 404
            
        print(f"Found user ID: {user.id}")
        
        # Get portfolio items
        portfolio_items = Portfolio.query.filter_by(user_id=user.id).all()
        print(f"Found {len(portfolio_items)} portfolio items")
        
        # Initialize response data
        portfolio_data = []
        total_value = user.virtual_balance
        
        # Process each portfolio item
        for item in portfolio_items:
            try:
                print(f"Processing {item.symbol}")
                stock = yf.Ticker(item.symbol)
                
                # Get current price
                current_price = None
                try:
                    hist = stock.history(period='1d')
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        print(f"Got price from history: {current_price}")
                except Exception as e:
                    print(f"Error getting price from history: {e}")
                    
                if current_price is None:
                    try:
                        current_price = stock.info.get('regularMarketPrice')
                        print(f"Got price from info: {current_price}")
                    except Exception as e:
                        print(f"Error getting price from info: {e}")
                
                if current_price is None:
                    print(f"Could not get price for {item.symbol}")
                    continue
                
                # Calculate values
                value = item.shares * current_price
                total_value += value
                gain_loss = value - (item.average_price * item.shares)
                
                # Add to portfolio data
                portfolio_data.append({
                    'symbol': item.symbol,
                    'shares': item.shares,
                    'avg_price': float(item.average_price),
                    'current_price': float(current_price),
                    'value': float(value),
                    'gain_loss': float(gain_loss)
                })
                print(f"Added {item.symbol} to portfolio data")
                
            except Exception as e:
                print(f"Error processing {item.symbol}: {str(e)}")
                continue
        
        # Prepare response
        response_data = {
            'portfolio': portfolio_data,
            'total_value': float(total_value),
            'cash_balance': float(user.virtual_balance)
        }
        
        print("Final response data:", response_data)
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Portfolio error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

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
        print(f"Error fetching price for {symbol}: {str(e)}")
        raise ValueError(f"Failed to fetch price for {symbol}: {str(e)}")

@app.route('/api/trade', methods=['POST'])
@jwt_required()
def trade():
    try:
        # Get email from JWT token
        email = get_jwt_identity()
        print(f"Processing trade for email: {email}")
        
        # Find user by email
        user = User.query.filter_by(email=email).first()
        if not user:
            print(f"No user found for email: {email}")
            return jsonify({'error': 'User not found'}), 404
            
        print(f"Found user ID: {user.id}")
        
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
            print(f"Fetched price for {data['symbol']}: ${current_price}")
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            print(f"Unexpected error fetching price: {str(e)}")
            return jsonify({'error': 'Failed to fetch stock price'}), 500
            
        total_cost = current_price * shares
        print(f"Calculated total cost: ${total_cost}")
        
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
                print(f"Updated user balance: ${user.virtual_balance}")
                
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
                print(f"Updated user balance: ${user.virtual_balance}")
                
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
                print(f"Transaction committed successfully. ID: {transaction.id}")
                
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
                print(f"Error committing transaction: {str(e)}")
                return jsonify({'error': 'Database error while executing trade'}), 500
                
        except Exception as e:
            db.session.rollback()
            print(f"Error during trade execution: {str(e)}")
            return jsonify({'error': 'Failed to execute trade'}), 500
            
    except Exception as e:
        print(f"Unexpected error in trade endpoint: {str(e)}")
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

if __name__ == '__main__':
    app.run(debug=True) 
