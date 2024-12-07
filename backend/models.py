from database import db
from datetime import datetime

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
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    shares = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)
    action = db.Column(db.String(4), nullable=False)  # 'buy' or 'sell'
    timestamp = db.Column(db.DateTime, default=datetime.utcnow) 