import numpy as np
import pandas as pd
from scipy import stats


def load_uci_airquality(path="data/AirQualityUCI.csv", n=500):
    """Load and clean the UCI Air Quality dataset (Italy)."""
    df = pd.read_csv(path, sep=";", decimal=",",
                     usecols=["NO2(GT)", "PT08.S4(NO2)"])
    df.columns = ["no2_ref", "no2_sensor"]
    df.replace(-200, np.nan, inplace=True)
    df.dropna(inplace=True)
    return df.iloc[:n].reset_index(drop=True)


def calibrate_sensor(sensor, reference):
    """Linear calibration: map raw sensor units → μg/m³."""
    slope, intercept, *_ = stats.linregress(sensor, reference)
    return slope * sensor + intercept
