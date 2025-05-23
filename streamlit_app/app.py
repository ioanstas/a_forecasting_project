import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time


import streamlit as st
from src.data_loader import load_data
from src.preprocess_data import single_ts_choice, reset_indexer, split_time_series
from models.prophet_model.prophet_model import train_prophet_model, do_forecast
from src.visualizations import vis_starting_ts, vis_starting_ts_streamlit, decompose_time_series, plot_forecast
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly, plot_components_plotly


st.title("📈 Facebook Prophet Forecast on Drug sales")

path = "data/raw/extracted/salesdaily.csv"
df = load_data(path)
ts = single_ts_choice(df=df)
prophet_ts = reset_indexer(ts)
ts_to_train, ts_to_test = split_time_series(prophet_ts)

# vis
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(prophet_ts.ds, prophet_ts.y, color='#283593')
ax.set_title('Starting Time Series')
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.grid(True)
st.pyplot(fig)

#split visualization
st.write("Splitting into basic training and test set like below")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(ts_to_train.ds, ts_to_train.y, color='#283593', label='~Training')
ax.plot(ts_to_test.ds, ts_to_test.y, color='#117a65', label='~Test')
ax.set_title('Starting Time Series')
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# button

if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.forecast = None

if st.button("Train Prophet Model", type="primary"):
    
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.02)  # Simulate work
        progress_bar.progress(percent_complete + 1)

    st.session_state.prophet_model = train_prophet_model(ts_to_train)
    st.session_state.forecast = st.session_state.prophet_model.predict(ts_to_test)

    progress_bar.empty()
    st.success("Training complete!")
    st.session_state.model_trained = True

component = st.selectbox(
    "Choose a component to display:",
    options=["None", "Trend", "Yearly", "Weekly", "All"],
    index=0
)

if st.session_state.model_trained and st.session_state.forecast is not None and component != "None":
    if component == "All":
        fig = st.session_state.prophet_model.plot_components(st.session_state.forecast)
    else:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(st.session_state.forecast['ds'], st.session_state.forecast[component.lower()])
        ax.set_title(f"{component} Component")
        ax.grid(True)
    
    st.pyplot(fig)

with st.expander("🔎 Advanced Interactive Visualizations", expanded=False):
    if st.session_state.model_trained:
        st.markdown("### Interactive Forecast ft.Facebook Prophet")
        fig_forecast = plot_plotly(st.session_state.prophet_model, st.session_state.forecast)
        
        fig_forecast.update_layout(
            plot_bgcolor='rgba(40,40,40,1)',
            paper_bgcolor='rgba(0,0,0,0.7)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        st.markdown("### Interactive Components")
        fig_components = plot_components_plotly(st.session_state.prophet_model, st.session_state.forecast)
        
        # Dark mode example
        fig_components.update_layout(
            plot_bgcolor='rgba(30,30,30,1)',
            paper_bgcolor='rgba(0,0,0,0.7)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_components, use_container_width=True)

