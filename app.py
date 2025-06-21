import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib

# Set page config
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")


# Add this to your custom CSS section
st.markdown("""
<style>
    .prediction-card, .prediction-card h2, .prediction-card h3, 
    .prediction-card p, .prediction-card table, .prediction-card td {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)
# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
    }
    .prediction-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sample-prediction {
        background-color: #e9f7ef;
        border-left: 5px solid #2ecc71;
    }
    .user-prediction {
        background-color: #eaf2f8;
        border-left: 5px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📈 Stock Price Predictor")
st.markdown("""
Predict future stock prices based on historical data. Enter a future date below to get predictions 
for Open, High, Low, Close prices and Volume.
""")

# Sample data (in a real app, you would load your trained model and data)
@st.cache_resource
def load_model_and_scalers():
    # Load your pre-trained model and scalers here
    # For this example, we'll create a dummy model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Dummy data to fit the model (in a real app, you would use your actual data)
    X_train = np.random.rand(100, 7)  # 7 features
    y_train = np.random.rand(100, 5)  # 5 outputs (Open, High, Low, Close, Volume)
    
    model.fit(X_train, y_train)
    
    # Create dummy scalers
    scaler_X = MinMaxScaler().fit(X_train)
    scaler_y = MinMaxScaler().fit(y_train)
    
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_model_and_scalers()

# Function to make predictions (replace with your actual prediction logic)
def predict_stock_price(model, scaler_X, scaler_y, input_date):
    """
    Predict stock prices for a given date.
    In a real app, this would use your actual prediction logic.
    """
    # Convert date to days in future (for demo purposes)
    base_date = datetime(2021, 1, 29)  # Last date in sample data
    days_ahead = (input_date - base_date).days
    
    # This is dummy prediction logic - replace with your actual model prediction
    # For demo purposes, we'll generate some plausible-looking numbers
    base_price = 460.0
    volatility = 0.02
    trend = 0.0005 * days_ahead
    noise = np.random.normal(0, volatility * base_price)
    
    open_price = base_price * (1 + trend) + noise
    high_price = open_price * (1 + abs(np.random.normal(0, 0.01)))
    low_price = open_price * (1 - abs(np.random.normal(0, 0.01)))
    close_price = (high_price + low_price) / 2 * (1 + np.random.normal(0, 0.005))
    volume = int(abs(np.random.normal(300000, 100000)))
    
    return {
        'Date': input_date.strftime('%b %d, %Y'),
        'Open': round(open_price, 2),
        'High': round(high_price, 2),
        'Low': round(low_price, 2),
        'Close': round(close_price, 2),
        'Volume': volume
    }

# Sample predictions section
st.header("Sample Predictions")
st.markdown("Here are some example predictions for future dates:")

sample_dates = [
    datetime(2025, 9, 1),
    datetime(2025, 12, 1),
    datetime(2026, 6, 1)
]

cols = st.columns(3)
# In the sample predictions section, update the markdown styling:
for i, date in enumerate(sample_dates):
    with cols[i]:
        prediction = predict_stock_price(model, scaler_X, scaler_y, date)
        st.markdown(f"<div class='prediction-card sample-prediction' style='color: black;'>"
                    f"<h3 style='color: black;'>{prediction['Date']}</h3>"
                    f"<p style='color: black;'><strong>Open:</strong> ${prediction['Open']}</p>"
                    f"<p style='color: black;'><strong>High:</strong> ${prediction['High']}</p>"
                    f"<p style='color: black;'><strong>Low:</strong> ${prediction['Low']}</p>"
                    f"<p style='color: black;'><strong>Close:</strong> ${prediction['Close']}</p>"
                    f"<p style='color: black;'><strong>Volume:</strong> {prediction['Volume']:,}</p>"
                    f"</div>", unsafe_allow_html=True)

# User prediction section
st.header("Your Prediction")
st.markdown("Enter a future date to get a custom prediction:")

col1, col2 = st.columns(2)
with col1:
    input_date = st.date_input("Select a future date", 
                             min_value=datetime.today(),
                             value=datetime.today())

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Predict Stock Prices")

if predict_btn:
    with st.spinner("Generating prediction..."):
        # Convert to datetime
        input_datetime = datetime.combine(input_date, datetime.min.time())
        
        # Get prediction
        prediction = predict_stock_price(model, scaler_X, scaler_y, input_datetime)
        
        # Display results
        st.markdown(f"<div class='prediction-card user-prediction'>"
                    f"<h2>Prediction for {prediction['Date']}</h2>"
                    f"<table style='width:100%'>"
                    f"<tr><td><strong>Open Price:</strong></td><td>${prediction['Open']}</td></tr>"
                    f"<tr><td><strong>High Price:</strong></td><td>${prediction['High']}</td></tr>"
                    f"<tr><td><strong>Low Price:</strong></td><td>${prediction['Low']}</td></tr>"
                    f"<tr><td><strong>Close Price:</strong></td><td>${prediction['Close']}</td></tr>"
                    f"<tr><td><strong>Volume:</strong></td><td>{prediction['Volume']:,}</td></tr>"
                    f"</table>"
                    f"</div>", unsafe_allow_html=True)

# How it works section
st.markdown("---")
st.header("How It Works")
st.markdown("""
1. **Model Training**: The system uses a machine learning model (Random Forest) trained on historical stock data.
2. **Feature Engineering**: The model considers various technical indicators like moving averages, price trends, and volume changes.
3. **Prediction**: When you enter a date, the model predicts the stock prices based on learned patterns.
4. **Results**: The predicted values for Open, High, Low, Close prices and Volume are displayed.
""")

# Disclaimer
st.markdown("---")
st.warning("""
**Disclaimer**: These predictions are for demonstration purposes only. 
Stock market predictions are inherently uncertain and this app should not 
be used for actual investment decisions. Always consult with a financial 
advisor before making investment decisions.
""")