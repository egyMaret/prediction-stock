import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from keras import load_model
import pickle

# Load the saved model, scaler, and DataFrame
model = load_model('model_lstm.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
df = pd.read_pickle('dataframe.pkl')

# Define the prediction function
def predict_future_close(date_to_predict, model, scaler, df, look_back=60):
    date_to_predict = datetime.strptime(date_to_predict, '%Y-%m-%d')
    last_date = df.index[-1]
    date_range = pd.date_range(start=last_date, end=date_to_predict, freq='B')
    last_close = df['Close'].values[-look_back:]
    last_close_scaled = scaler.transform(last_close.reshape(-1, 1))
    predictions = []
    for _ in date_range[1:]:
        input_data = np.array(last_close_scaled[-look_back:]).reshape(1, look_back, 1)
        predicted_close_scaled = model.predict(input_data)
        predicted_close = scaler.inverse_transform(predicted_close_scaled)
        predictions.append(predicted_close[0][0])
        last_close_scaled = np.append(last_close_scaled, predicted_close_scaled, axis=0)
    return predictions[-1], date_range[-1]

# Streamlit app interface
st.title('Stock Price Prediction')
st.write('Enter a date to predict the stock closing price.')

date_to_predict = st.text_input('Date (YYYY-MM-DD)', '2024-05-22')

if st.button('Predict'):
    predicted_price, predicted_date = predict_future_close(date_to_predict, model, scaler, df)
    st.write(f"Predicted closing price for {predicted_date.date()} is {predicted_price:.2f}")