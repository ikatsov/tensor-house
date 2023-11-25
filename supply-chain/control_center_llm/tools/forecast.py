import datetime

import numpy as np
import pandas as pd
from streamlit.logger import get_logger


def get_forecast(sku: str, location: str) -> pd.DataFrame:
    max_units = 200
    horizon_weeks = 52

    get_logger(__name__).info("forecast")

    today = datetime.date.today()
    nextMonday = today - datetime.timedelta(days=today.weekday()) + + datetime.timedelta(days=7)
    oneWeek = datetime.timedelta(days=7)
    x = [nextMonday + oneWeek*i for i in range(horizon_weeks)]

    sku_hash = hash(sku) % max_units
    y = (np.sin(np.linspace(0, 10, horizon_weeks)) * sku_hash + max_units).astype(int)

    forecast_df = pd.DataFrame({'date': x, 'demand': y})
    forecast_df = forecast_df.rename(columns={'date': 'index'}).set_index('index')

    get_logger(__name__).info(f"Forecast SKU [{sku}] in [{location}]: {forecast_df}")

    return forecast_df
