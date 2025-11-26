import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import date, timedelta

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 AI Stock Price Predictor")
st.markdown("This app uses a **Long Short-Term Memory (LSTM)** neural network to predict stock prices. It recalibrates daily with the latest market data.")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Configuration")
stock_ticker = st.sidebar.text_input("Stock Ticker (Yahoo Finance)", value="NESCO.NS")
lookback_years = st.sidebar.slider("History to Load (Years)", 1, 10, 5)
epochs = st.sidebar.slider("Training Epochs", 5, 50, 25)

# --- 1. DATA LOADING ---
@st.cache_data
def load_data(ticker, years):
    """Fetches data from Yahoo Finance."""
    try:
        start_date = date.today() - timedelta(days=years*365)
        end_date = date.today()
        data = yf.download(ticker, start=start_date, end=end_date)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data_load_state = st.text('Loading data...')
df = load_data(stock_ticker, lookback_years)
data_load_state.text('Data loaded successfully!')

if df is not None and not df.empty:
    # --- 2. RAW DATA VISUALIZATION ---
    st.subheader(f"Historical Close Price: {stock_ticker}")
    
    # Interactive Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price'))
    fig.layout.update(xaxis_rangeslider_visible=True, xaxis_title="Date", yaxis_title="Price (INR)")
    st.plotly_chart(fig, use_container_width=True)

    # --- 3. DATA PREPROCESSING ---
    data = df[['Close']]
    dataset = data.values
    
    # Scale data to 0-1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    prediction_days = 60

    # --- 4. MODEL TRAINING ---
    @st.cache_resource
    def train_lstm_model(scaled_data, prediction_days, epochs):
        """Trains the LSTM model on the provided data."""
        x_train = []
        y_train = []

        # Use 100% of data for training since we predict the unknown future
        train_len = len(scaled_data)
        
        for i in range(prediction_days, train_len):
            x_train.append(scaled_data[i-prediction_days:i, 0])
            y_train.append(scaled_data[i, 0])
            
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build Model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=0)
        
        return model

    st.write("---")
    st.write(f"🧠 **Training Model on {len(df)} days of data...**")
    
    # Progress bar just for visual feedback during computation
    progress_bar = st.progress(0)
    model = train_lstm_model(scaled_data, prediction_days, epochs)
    progress_bar.progress(100)
    
    # --- 5. PREDICT NEXT DAY ---
    # Take the last 60 days from the actual data
    last_60_days = scaled_data[-prediction_days:]
    X_test = []
    X_test.append(last_60_days)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    pred_price_scaled = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price_scaled)
    
    current_price = df['Close'].iloc[-1]
    predicted_price = pred_price[0][0]
    
    # Display Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Closing Price", f"₹ {current_price:.2f}")
    col2.metric("Predicted Next Price", f"₹ {predicted_price:.2f}", f"{predicted_price - current_price:.2f}")
    
    # --- 6. BACKTESTING / VALIDATION GRAPH ---
    st.subheader("Model Accuracy (Backtest on recent data)")
    
    # Validate on the last 100 days to show user how the model tracks
    val_days = 100
    if len(scaled_data) > val_days + prediction_days:
        x_val = []
        y_val = dataset[-val_days:]
        
        for i in range(len(scaled_data) - val_days, len(scaled_data)):
            x_val.append(scaled_data[i-prediction_days:i, 0])
            
        x_val = np.array(x_val)
        x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
        
        val_predictions = model.predict(x_val)
        val_predictions = scaler.inverse_transform(val_predictions)
        
        # Create comparison dataframe
        valid_df = pd.DataFrame(data={'Actual': y_val.flatten(), 'Predicted': val_predictions.flatten()})
        
        # Plot Validation
        fig_val = go.Figure()
        fig_val.add_trace(go.Scatter(y=valid_df['Actual'], mode='lines', name='Actual Price', line=dict(color='blue')))
        fig_val.add_trace(go.Scatter(y=valid_df['Predicted'], mode='lines', name='Predicted Price', line=dict(color='red')))
        fig_val.update_layout(xaxis_title="Days (Last 100)", yaxis_title="Price (INR)")
        st.plotly_chart(fig_val, use_container_width=True)
        
    else:
        st.warning("Not enough data to perform validation.")

else:
    st.warning("No data found. Please check the Ticker symbol.")
