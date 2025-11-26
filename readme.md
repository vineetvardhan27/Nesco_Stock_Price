📈 AI Stock Price Predictor

A Python-based web application that uses Deep Learning (LSTM) to forecast stock prices. The app pulls live data from Yahoo Finance, trains a model on the fly, and predicts the closing price for the next trading day.

🚀 Live Demo & Features

Live Data: Fetches the latest stock history automatically using yfinance.

Auto-Recalibration: Retrains the model every time new data is loaded to ensure predictions are based on the most recent trends.

Interactive Charts: Zoom and hover over price history using Plotly.

Performance Metrics: Calculates accuracy (RMSE, MAE, MAPE) on the last 100 days of data.

📊 Understanding the Metrics

This app provides three key metrics to help you judge the reliability of the prediction:

1. RMSE (Root Mean Squared Error)

What it is: Measures the "standard deviation" of the prediction errors.

Why it matters: It gives higher weight to large errors. If the RMSE is ₹45, it means the prediction typically deviates by about ₹45, but occasionally significantly more. Lower is better.

2. MAE (Mean Absolute Error)

What it is: The average difference between the Predicted Price and the Actual Price.

Why it matters: This is the most "human-readable" error. If MAE is ₹34, it means on an average day, the AI's guess is ₹34 away from the real closing price.

3. MAPE (Mean Absolute Percentage Error)

What it is: The error expressed as a percentage.

Why it matters: This allows you to compare accuracy across different stocks.

< 5%: Very Good

< 10%: Good

> 20%: Risky/Volatile

Example: A MAPE of 3.24% means the model is usually 96.76% accurate.

🛠️ Installation & Running Locally

Clone the repository:

git clone [https://github.com/your-username/stock-predictor.git](https://github.com/your-username/stock-predictor.git)
cd stock-predictor


Install requirements:

pip install -r requirements.txt


Run the app:

streamlit run app.py


🧰 Tech Stack

Frontend: Streamlit

Data Source: yfinance

Machine Learning: TensorFlow (Keras)

Preprocessing: Scikit-Learn

Visualization: Plotly

⚠️ Disclaimer

This tool is for educational purposes only. Stock market predictions involve significant risk. Do not use this tool as the sole basis for financial investment decisions.
