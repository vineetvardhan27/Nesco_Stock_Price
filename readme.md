# 📈 AI Stock Price Predictor

A Python-based web application that uses **Deep Learning (LSTM)** to forecast stock prices.  
The app pulls live data from Yahoo Finance, trains a model on the fly, and predicts the closing price for the **next trading day**.

---

## 🚀 Live Demo & Features

- **Live Data**: Fetches the latest stock history using *yfinance*
- **Auto-Recalibration**: Model retrains with most recent data on every prediction
- **Interactive Charts**: Zoom & hover analysis with *Plotly*
- **Performance Metrics**: RMSE, MAE & MAPE calculated on last 100 days

---

## 📊 Understanding the Metrics

### 1️⃣ RMSE — Root Mean Squared Error
- Measures the standard deviation of prediction errors  
- High penalty for large mistakes  
- Example: RMSE = ₹45 ⇒ Typical deviation is about ₹45  
- **Lower = better**

### 2️⃣ MAE — Mean Absolute Error
- Average difference between prediction & actual price  
- Most human-friendly accuracy measure  
- Example: MAE = ₹34 ⇒ AI is usually ₹34 off from the real price  

### 3️⃣ MAPE — Mean Absolute Percentage Error
- Error expressed in percentage → stock-to-stock comparable  
- Guideline:
  - **< 5% = Very Good**
  - **< 10% = Good**
  - **> 20% = Risky / Volatile**
- Example: MAPE = 3.24% ⇒ Model ≈ **96.76% accurate**

---
