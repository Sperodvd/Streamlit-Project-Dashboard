import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import os
import ta
import xgboost as xgb
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

st.set_page_config(layout="wide", page_title="Stock Analysis and Forecasting with Deep Learning Models")
st.title("Stock Analysis and Forecasting Dashboard")
ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input('Start Date', datetime(2024, 1,1))
end_date = st.sidebar.date_input('End Date', datetime.today())

st.sidebar.subheader("Technical Indicators")
indicators = st.sidebar.multiselect(
    "Select Indicators:",
    ["50-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP", "MACD", "RSI"], default=["MACD"]
)

@st.cache_data(ttl=3600)
def get_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end, auto_adjust=False)

def analyze_ticker(ticker, data):
    # Determine which subplots are needed
    show_macd = "MACD" in indicators
    show_rsi = "RSI" in indicators
    n_rows = 1 + int(show_macd) + int(show_rsi)
    subplot_titles = [f"{ticker} Candlestick Chart"]
    if show_macd:
        subplot_titles.append("MACD")
    if show_rsi:
        subplot_titles.append("RSI")
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.7] + [0.25]*(n_rows-1),
                        subplot_titles=tuple(subplot_titles))
    # Main candlestick chart always in row 1
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name="Candlestick"), row=1, col=1)
    def add_indicator(indicator):
        if indicator == "50-Day SMA":
            sma = data['Close'].rolling(window=50).mean()
            fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (50)'), row=1, col=1)
        elif indicator == "20-Day EMA":
            ema = data['Close'].ewm(span=20).mean()
            fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'), row=1, col=1)
        elif indicator == "20-Day Bollinger Bands":
            sma = data['Close'].rolling(window=20).mean()
            std = data['Close'].rolling(window=20).std()
            bb_upper = sma + 2*std
            bb_lower = sma - 2*std
            fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'), row=1, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'), row=1, col=1)
        elif indicator == "VWAP":
            data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
            fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'), row=1, col=1)
        elif indicator == "MACD":
            import ta
            macd_object = ta.trend.MACD(data['Close'])
            data['MACD'] = macd_object.macd()
            data['MACD_Signal'] = macd_object.macd_signal()
            data['MACD_Diff'] = macd_object.macd_diff()
            macd_row = 2 if show_macd and not show_rsi else (2 if show_macd and show_rsi else 1)
            fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD Line', line=dict(color='skyblue')), row=macd_row, col=1)
            fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], mode='lines', name='Signal Line', line=dict(color='lightcoral')), row=macd_row, col=1)
            fig.add_trace(go.Bar(x=data.index, y=data['MACD_Diff'], name='Histogram', marker_color='grey', opacity=0.5), row=macd_row, col=1)
        elif indicator == "RSI":
            def calculate_rsi(data, period=14):
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
            rsi = calculate_rsi(data)
            rsi_row = n_rows if show_rsi else 2
            fig.add_trace(go.Scatter(x=data.index, y=rsi, mode='lines', name='RSI (14)', line=dict(color='orange')), row=rsi_row, col=1)
    for ind in indicators:
        add_indicator(ind)
    fig.update_layout(xaxis_rangeslider_visible=False, showlegend=True)
    return fig

def train_lstm_model(data, model_path='lstm_model.h5', scaler_path='scaler.gz', look_back=20, test_split=0.2):
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    feature_data = data[features].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(feature_data)    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, :])
        y.append(scaled_data[i, features.index('Close')])  # Predicting normalized 'Close'
    X, y = np.array(X), np.array(y)

    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(look_back, len(features))))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss',factor=0.5, patience=5, verbose=1) 
    ]

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks, verbose=0)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    loss = model.evaluate(X_test, y_test, verbose=0)
    st.info(f"Test Loss (MSE): {loss:.6f}")

    return model_path, scaler_path, loss

from tensorflow.keras.models import load_model

def predict_lstm(data, n_days=7, model_path='lstm_model.h5', scaler_path='scaler.gz', look_back=20):
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        st.error("Model and scaler not found. Please train the model first.")
        return None
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    feature_data = data[features].values
    scaled_data = scaler.transform(feature_data)
    last_seq = scaled_data[-look_back:]
    preds = []
    for _ in range(n_days):
        # For future prediction, use last known values for Open/High/Low/Volume, only update Close
        input_seq = last_seq.copy()
        pred = model.predict(input_seq.reshape(1, look_back, len(features)), verbose=0)
        # Create next input: shift, append predicted close, keep other features same as last known
        next_row = last_seq[-1].copy()
        next_row[features.index('Close')] = pred[0, 0]
        last_seq = np.vstack([last_seq[1:], next_row])
        preds.append(pred[0, 0])
    # Inverse transform only the 'Close' column
    dummy = np.zeros((len(preds), len(features)))
    dummy[:, features.index('Close')] = preds
    inv_preds = scaler.inverse_transform(dummy)[:, features.index('Close')]
    return inv_preds

def add_features(data):
    df = data.copy()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['BB_upper'] = df['Close'].rolling(window=20).mean() + 2*df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['Close'].rolling(window=20).mean() - 2*df['Close'].rolling(window=20).std()
    df['Return_1'] = df['Close'].pct_change(1)
    df['Return_5'] = df['Close'].pct_change(5)
    df = df.dropna()
    return df

def train_gru_model(data, model_path='gru_model.h5', scaler_path='gru_scaler.gz', look_back=20, test_split=0.2):
    from tensorflow.keras.layers import GRU
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_20', 'BB_upper', 'BB_lower', 'Return_1', 'Return_5']
    feature_data = data[features].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(feature_data)
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, :])
        y.append(scaled_data[i, features.index('Close')])
    X, y = np.array(X), np.array(y)
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    model = Sequential()
    model.add(GRU(100, return_sequences=True, input_shape=(look_back, len(features))))
    model.add(Dropout(0.2))
    model.add(GRU(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss',factor=0.5, patience=5, verbose=1)
    ]
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks, verbose=0)
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    loss = model.evaluate(X_test, y_test, verbose=0)
    return model_path, scaler_path, loss

def predict_gru(data, n_days=7, model_path='gru_model.h5', scaler_path='gru_scaler.gz', look_back=20):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_20', 'BB_upper', 'BB_lower', 'Return_1', 'Return_5']
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        st.error("GRU model and scaler not found. Please train the model first.")
        return None
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    feature_data = data[features].values
    scaled_data = scaler.transform(feature_data)
    last_seq = scaled_data[-look_back:]
    preds = []
    for _ in range(n_days):
        input_seq = last_seq.copy()
        pred = model.predict(input_seq.reshape(1, look_back, len(features)), verbose=0)
        next_row = last_seq[-1].copy()
        next_row[features.index('Close')] = pred[0, 0]
        last_seq = np.vstack([last_seq[1:], next_row])
        preds.append(pred[0, 0])
    dummy = np.zeros((len(preds), len(features)))
    dummy[:, features.index('Close')] = preds
    inv_preds = scaler.inverse_transform(dummy)[:, features.index('Close')]
    return inv_preds

def train_dense_model(data, scaler_path='dense_scaler.gz', test_split=0.2):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_20', 'BB_upper', 'BB_lower', 'Return_1', 'Return_5']
    X = data[features].values
    y = data['Close'].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    split_idx = int(len(X_scaled) * (1 - test_split))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'dense_model.pkl')
    joblib.dump(scaler, scaler_path)
    preds = model.predict(X_test)
    loss = mean_squared_error(y_test, preds)
    return 'dense_model.pkl', scaler_path, loss

def predict_dense(data, n_days=7, scaler_path='dense_scaler.gz', look_back=20):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_20', 'BB_upper', 'BB_lower', 'Return_1', 'Return_5']
    model = joblib.load('dense_model.pkl')
    scaler = joblib.load(scaler_path)
    X = data[features].values
    X_scaled = scaler.transform(X)
    last_row = X_scaled[-1]
    preds = []
    for _ in range(n_days):
        pred = model.predict(last_row.reshape(1, -1))[0]
        last_row[features.index('Close')] = pred
        preds.append(pred)
    return np.array(preds)

def train_xgb_model(data, scaler_path='xgb_scaler.gz', test_split=0.2):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_20', 'BB_upper', 'BB_lower', 'Return_1', 'Return_5']
    X = data[features].values
    y = data['Close'].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    split_idx = int(len(X_scaled) * (1 - test_split))
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'xgb_model.pkl')
    joblib.dump(scaler, scaler_path)
    preds = model.predict(X_test)
    loss = mean_squared_error(y_test, preds)
    return 'xgb_model.pkl', scaler_path, loss

def predict_xgb(data, n_days=7, scaler_path='xgb_scaler.gz', look_back=20):
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'EMA_20', 'BB_upper', 'BB_lower', 'Return_1', 'Return_5']
    model = joblib.load('xgb_model.pkl')
    scaler = joblib.load(scaler_path)
    X = data[features].values
    X_scaled = scaler.transform(X)
    last_row = X_scaled[-1]
    preds = []
    for _ in range(n_days):
        pred = model.predict(last_row.reshape(1, -1))[0]
        last_row[features.index('Close')] = pred
        preds.append(pred)
    return np.array(preds)

if ticker and start_date < end_date:
    st.write(f"Fetching data for **{ticker}** from **{start_date}** to **{end_date}**")
    data = get_data(ticker, start_date, end_date)
    data = add_features(data)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not data.empty and required_cols.issubset(data.columns):
        if len(data) < 20:
            st.warning('Not enough data for 20-day indicators. Please select a wider date range.')
        fig = analyze_ticker(ticker, data)
        st.plotly_chart(fig)
        st.subheader("Model Training and Prediction")
        model_losses = {}
        if st.button("Train All Models"):
            with st.spinner("Training all models..."):
                _, _, lstm_loss = train_lstm_model(data)
                _, _, gru_loss = train_gru_model(data)
                _, _, dense_loss = train_dense_model(data)
                _, _, xgb_loss = train_xgb_model(data)
                model_losses = {'LSTM': lstm_loss, 'GRU': gru_loss, 'Dense': dense_loss, 'XGBoost': xgb_loss}
                st.session_state['model_losses'] = model_losses
                st.success("All models trained!")
        if 'model_losses' in st.session_state:
            st.write("Model Losses (MSE):", st.session_state['model_losses'])
            model_choice = st.selectbox("Choose model for prediction", list(st.session_state['model_losses'].keys()))
            n_days = st.number_input("Days to predict into the future", min_value=1, max_value=10, value=7)
            if st.button("Predict with Selected Model"):
                with st.spinner("Predicting future prices..."):
                    if model_choice == 'LSTM':
                        preds = predict_lstm(data, n_days=n_days)
                    elif model_choice == 'GRU':
                        preds = predict_gru(data, n_days=n_days)
                    elif model_choice == 'Dense':
                        preds = predict_dense(data, n_days=n_days)
                    elif model_choice == 'XGBoost':
                        preds = predict_xgb(data, n_days=n_days)
                    if preds is not None:
                        future_dates = pd.date_range(data.index[-1], periods=n_days+1, freq='B')[1:]
                        hist_df = pd.DataFrame({'Date': data.index, 'Close': data['Close']})
                        pred_df = pd.DataFrame({'Date': future_dates, 'Close': preds})
                        combined_df = pd.concat([hist_df, pred_df], ignore_index=True)
                        combined_df.set_index('Date', inplace=True)
                        fig_pred = go.Figure()
                        fig_pred.add_trace(go.Scatter(x=combined_df.index, y=combined_df['Close'], mode='lines+markers', name='Historical + Predicted Close', line=dict(color='royalblue')))
                        fig_pred.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Close'], mode='markers', name='Predicted Close', marker=dict(color='deepskyblue', size=8)))
                        fig_pred.update_layout(title=f"{ticker} {model_choice} {n_days}-Day Prediction", xaxis_title="Date", yaxis_title="Price")
                        st.plotly_chart(fig_pred)
    else:
        st.warning('No data found for this ticker and date range, or required columns are missing.')
else:
    st.info('Enter a valid ticker and date range to begin.')
