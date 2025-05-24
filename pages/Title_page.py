import streamlit as st

st.set_page_config(page_title="About", layout="wide")
st.title("Stock Analysis and Forecasting with Deep Learning Models")

st.markdown("""
## Overview
            
This application demonstrates interactive stock analysis and prediction capabilities built with Streamlit & Python powered by libraries such as tensorflow, sklearn, yfinance, ta-lib, and plotly.
            
- **Stock Charts**: Visualize historical price data with a variety of technical indicators. Following indicators are utilized in the application:

| Indicator        | Description                          |
|------------------|--------------------------------------|
| SMA              | Simple Moving Average                |
| EMA              | Exponential Moving Average           |
| Bollinger Bands  | Upper and Lower Bands                |
| VWAP             | Volume Weighted Average Price        |
| MACD             | Moving Average Convergence Divergence|
| RSI              | Relative Strength Index              |

- **Future Predictions**: Forecast future stock prices using multiple deep learning models. Following models are utilized in the application:
            
| Model            | Description                          |
|------------------|--------------------------------------|
| LSTM             | Long Short-Term Memory               |
| GRU              | Gated Recurrent Unit                 |
| DNN              | Dense Neural Network                 |
| XGBoost          | Extreme Gradient Boosting            |

- **Model Comparison**: Train, compare, and select the best model for your prediction needs.
            MSE (Mean Squared Error) is used as the evaluation metric for model comparison.
            
## Instructions
1. **Select a Stock**: Enter the stock ticker symbol in the sidebar.
2. **Choose Date Range**: Specify the start and end dates for the analysis.
3. **Select Indicators**: Choose the technical indicators you want to visualize.
4. **Train Models**: Click on the "Train Models" button to train all the models on the historical data.
5. **Model Comparison**: Compare the performance of different models on evaluation metric and select the best model from the dropdown menu.
4. **View Predictions**: Click on the "Predict" button to see the forecasted prices up to maximum 10 days into the future.




---
**Disclaimer**  
This demo is intended for educational and illustrative purposes, showcasing the application of modern machine learning and deep learning techniques applied to financial time series data within an interactive dashboard.
The content presented is not a financial advice and should not be used for actual trading or investment decisions. Users are strongly encourage to conduct their own research and consult with qulified financial professional before undertaking any investment activities.
""")