import subprocess
import sys
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta

# Function to install missing packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required libraries
required_packages = ["yfinance", "numpy", "pandas", "scikit-learn", "matplotlib"]

# Install missing packages
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        install(package)

# Function to fetch stock data
def get_stock_data(ticker, period='5y'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[['Close']].reset_index()
    df['Days'] = np.arange(len(df))
    return df

# Train Linear Regression model
def train_model(df):
    X = df[['Days']]
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f'Mean Absolute Error: {mae:.2f}')
    return model

# Predict future stock prices
def predict_future(model, df, days_ahead=30):
    last_day = df['Days'].max()
    future_days = np.arange(last_day + 1, last_day + days_ahead + 1).reshape(-1, 1)
    future_prices = model.predict(future_days)
    future_dates = [datetime.today() + timedelta(days=i) for i in range(1, days_ahead + 1)]
    return future_dates, future_prices

# Plot future predictions only
def plot_future_predictions(future_dates, future_prices):
    plt.figure(figsize=(10,5))
    plt.plot(future_dates, future_prices, color='green', linestyle='dashed', marker='o', label='Future Prediction')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.ylabel('Stock Price')
    plt.legend()
    plt.title('Next 30 Days Stock Price Prediction')
    plt.show()

if __name__ == "__main__":
    ticker = 'AAPL'  # Change this to any stock ticker you prefer
    df = get_stock_data(ticker)
    df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date column is in datetime format
    model = train_model(df)
    
    # Predict future prices
    future_dates, future_prices = predict_future(model, df)
    
    # Plot only future predictions
    plot_future_predictions(future_dates, future_prices)
    
    # Print future stock prices
    print("\n📈 Next 30 Days Stock Price Predictions:")
    for date, price in zip(future_dates, future_prices):
        print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")
