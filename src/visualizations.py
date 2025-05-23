from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import streamlit as st

def vis_starting_ts(ts):
    plt.figure(figsize=(8, 4))
    ts.plot(title='Daily M01AB Sales')
    plt.ylabel('Sales')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()

def vis_starting_ts_streamlit(ts):
    fig, ax = plt.subplots(figsize=(8, 4))
    ts.plot(ax=ax, title='Daily M01AB Sales')
    ax.set_ylabel('Sales')
    ax.set_xlabel('Date')
    plt.tight_layout()
    st.pyplot(fig)

def decompose_time_series(ts, period=365, model='additive'):
    result = seasonal_decompose(ts, model=model, period=period)
    return result.trend, result.seasonal, result.resid


def plot_forecast(model, forecast):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig = model.plot(forecast, ax=ax)
    plt.tight_layout()
    plt.show()
    return fig
