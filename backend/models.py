from database import db
from datetime import datetime
from sqlalchemy import Sequence, event
from sqlalchemy.sql import text

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.LargeBinary, nullable=False)
    virtual_balance = db.Column(db.Float, default=1000000.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    shares = db.Column(db.Integer, nullable=False)
    average_price = db.Column(db.Float, nullable=False)

class Transaction(db.Model):
    __tablename__ = 'transaction'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    shares = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)
    action = db.Column(db.String(4), nullable=False)  # 'buy' or 'sell'
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@event.listens_for(Transaction.__table__, 'after_create')
def create_transaction_sequence(target, connection, **kw):
    # Get the current maximum ID
    result = connection.execute(text('SELECT MAX(id) FROM transaction'))
    max_id = result.scalar() or 0
    
    # Create a new sequence starting after the max ID
    sequence_name = 'transaction_id_seq'
    connection.execute(text(f'DROP SEQUENCE IF EXISTS {sequence_name}'))
    connection.execute(text(f'CREATE SEQUENCE {sequence_name} START WITH {max_id + 1}'))
    connection.execute(text(f'ALTER TABLE transaction ALTER COLUMN id SET DEFAULT nextval(\'{sequence_name}\')'))