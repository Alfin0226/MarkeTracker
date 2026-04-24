import yfinance as yf
from datetime import datetime, timezone, timedelta
from app import app
from models import db, Transaction, BenchmarkHistory

def fetch_and_store_benchmarks():
    symbols = ['QQQ', 'VOO', 'TQQQ']
    
    with app.app_context():
        # Ensure the table is created
        db.create_all()

        # Find the earliest transaction date
        earliest_tx = db.session.query(db.func.min(Transaction.timestamp)).scalar()
        
        if not earliest_tx:
            print("No transactions found. Cannot determine start date.")
            start_date = datetime.now(timezone.utc) - timedelta(days=365) # fallback to 1 year
        else:
            # Ensure it is timezone-aware
            if earliest_tx.tzinfo is None:
                start_date = earliest_tx.replace(tzinfo=timezone.utc)
            else:
                start_date = earliest_tx

        print(f"Fetching data starting from: {start_date.strftime('%Y-%m-%d')}")
        
        for symbol in symbols:
            print(f"Fetching weekly data for {symbol}...")
            ticker = yf.Ticker(symbol)
            # yf history accepts dates as string in 'YYYY-MM-DD'
            history = ticker.history(start=start_date.strftime('%Y-%m-%d'), interval='1wk')
            
            records_added = 0
            for date, row in history.iterrows():
                if date.tzinfo is None:
                    db_date = date.replace(tzinfo=timezone.utc)
                else:
                    db_date = date.astimezone(timezone.utc)
                    
                price = row['Close']
                
                # Check if this precise timestamp exists
                existing = db.session.query(BenchmarkHistory).filter_by(
                    symbol=symbol,
                    timestamp=db_date
                ).first()
                
                if not existing:
                    new_record = BenchmarkHistory(
                        symbol=symbol,
                        price=float(price),
                        timestamp=db_date
                    )
                    db.session.add(new_record)
                    records_added += 1
                    
            db.session.commit()
            print(f"Successfully added {records_added} weekly record(s) for {symbol}.")

if __name__ == '__main__':
    fetch_and_store_benchmarks()
