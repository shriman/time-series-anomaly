import typer
import configparser
from pathlib import Path
import pickle
import pandas as pd
import os
from src.ml_core.prophet_forecasting import prophet_train, prophet_forecast

app = typer.Typer(help="Sales Forecast ML Model CLI")

config = configparser.ConfigParser()
config.read("config.ini")
CleanedDataPath = config["DEFAULT"]["CleanedDataPath"]

@app.command()
def train(
        model_type: str = typer.Option(..., help="Type of model to train. Possible values: (PROPHET)"),
        cleaned_data_path: str = typer.Option(CleanedDataPath, help="Path to cleaned training data"),
):
    supported_models = ["PROPHET"]
    if model_type not in supported_models:
        print("Model type: ", model_type, "not yet supported. Please try with one of the following:", supported_models)

    data_file = Path(cleaned_data_path)
    if not data_file.exists():
        print("Model file not found. Please try with one of the following:", supported_models)
        return None

    df = pd.read_csv(cleaned_data_path)

    if model_type == "PROPHET":
        model_output_path = config["PROPHET"]["ProphetModelPath"]
        best_model = prophet_train(data_path=cleaned_data_path,
                                   model_output_path=model_output_path)

        # write model to path
        print("Best model: ", best_model)
        print("Saving model to ", model_output_path)
        with open(model_output_path, 'wb') as f:
            pickle.dump(best_model, f)

    return None


@app.command()
def forecast(
        model_type: str = typer.Option(..., help="Type of model. Possible values: (SARIMAX, PROPHET)"),
):
    print("This is Forecast method")
    supported_models = ["PROPHET"]
    if model_type not in supported_models:
        print("Model type: ", model_type, "not yet supported. Please try with one of the following:", supported_models)

    if model_type == "PROPHET":
        model_path = config["PROPHET"]["ProphetModelPath"]
        forecast_output_path = config["PROPHET"]["ProphetForecastPath"]
        model_file = Path(model_path)
        if not model_file.exists():
            print("Model file not found. Please try with one of the following:", supported_models)
            return None

        sales_forecast = prophet_forecast(model_path=model_path,
                                          forecast_horizon=30)

        # save the dataframe
        print("Saving model to ", forecast_output_path)
        sales_forecast.to_pickle(forecast_output_path)

    return None


if __name__ == "__main__":
    app()
