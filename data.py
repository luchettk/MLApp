import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.title("Time Series Model Comparisons")

@st.cache_resource
def load_data():
    return pd.read_csv("weekly_lyme_disease_cases.csv")

tick_data = load_data()
tick_data['Date'] = pd.to_datetime(tick_data['Year'].astype(str) + tick_data['MMWR Week'].astype(str) + '0', format='%Y%U%w')
tick_data = tick_data.drop(columns = ["Year"])
lyme_disease = tick_data.drop(columns = ["MMWR Week"])
lyme_disease_rate = lyme_disease[['Date', 'Cases']]
lyme_disease_rate.set_index('Date', inplace = True)
lyme_diease_rate = lyme_disease_rate.dropna()

train_size = int(.8 * len(lyme_disease_rate))
train_data = lyme_disease_rate[:train_size]
test_data = lyme_disease_rate[train_size:]

# put the y feature in arrays
y_train = np.array(train_data['Cases']).reshape(-1,1)
y_test = np.array(test_data['Cases']).reshape(-1,1)

@st.cache_resource
def fit_ar_model(y_train, train_data_len, test_data_len):
    model = AutoReg(y_train, lags=35, trend='n').fit()
    return model.predict(start=train_data_len, end=train_data_len + test_data_len - 1)

@st.cache_resource
def fit_ma_model(y_train, train_data_len, test_data_len):
    model = ARIMA(y_train, order=(0, 0, 35)).fit()
    return model.predict(start=train_data_len, end=train_data_len + test_data_len - 1)

@st.cache_resource
def fit_arma_model(y_train, train_data_len, test_data_len):
    model = ARIMA(y_train, order=(14, 0, 14)).fit()
    return model.predict(start=train_data_len, end=train_data_len + test_data_len - 1)


ar_pred = fit_ar_model(y_train, len(train_data), len(test_data))
ma_pred = fit_ma_model(y_train, len(train_data), len(test_data))
arma_pred = fit_arma_model(y_train, len(train_data), len(test_data))



all_models = ["AR", "MA", "ARMA"]
with st.container(border=True):
    selected_models = st.multiselect("Time Series Models", all_models, default=["AR"])


model_predictions = {
    "AR": pd.Series(ar_pred.flatten(), index=test_data.index),
    "MA": pd.Series(ma_pred.flatten(), index=test_data.index),
    "ARMA": pd.Series(arma_pred.flatten(), index=test_data.index)
}

tab1, = st.tabs(["Chart"])

with tab1: 
    fig, ax = plt.subplots()
    ax.plot(test_data.index, test_data["Cases"], label="Actual Close")

    for model in selected_models:
        if model in model_predictions:
            ax.plot(test_data.index, model_predictions[model], label=f"{model} Prediction", linestyle='--')


    ax.set_xlabel("Week")
    ax.set_ylabel("Case")
    ax.set_title(f"{', '.join(selected_models)} Model Prediction(s)")
    ax.legend()
    st.pyplot(fig)

