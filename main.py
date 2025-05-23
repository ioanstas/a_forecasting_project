from src.data_loader import load_data
from src.preprocess_data import single_ts_choice, reset_indexer, split_time_series
from models.prophet_model.prophet_model import train_prophet_model, do_forecast
from src.visualizations import vis_starting_ts, vis_starting_ts_streamlit, decompose_time_series, plot_forecast
import matplotlib.pyplot as plt
# import matplotli


path = "data/raw/extracted/salesdaily.csv"


def main():
    # loading data
    df = load_data(path)
    print(df.head())

    # picking one time series
    ts = single_ts_choice(df=df)
    vis_starting_ts(ts)
    # vis_starting_ts_streamlit(ts)
    # decomposition to other components for visualization purposes
    trend, seasonality, residual = decompose_time_series(ts)

    # resetting index for prophet feeding
    prophet_ts = reset_indexer(ts)

    # splitting data for evaluation (mostly)
    ts_to_train, ts_to_test = split_time_series(prophet_ts)
    print(len(ts_to_train), len(ts_to_test))
    print(ts_to_train.head())
    print(type(ts_to_train))

    # instantiating and training model
    prophet_model = train_prophet_model(ts_to_train)
    forecast = prophet_model.predict(ts_to_test)

    # vis forecast
    fig = plot_forecast(prophet_model, forecast)
    


    trend, seasonality, residual = decompose_time_series(ts)
    prophet_model.plot_components(forecast)
    plt.show()
    



if __name__ == "__main__":
    main()
