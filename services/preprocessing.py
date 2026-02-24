import pandas as pd


def preprocess_input(data):

    df = pd.DataFrame([data])

    expected_features = [
        "temperature",
        "humidity",
        "soil_moisture",
        "rainfall"
    ]

    df = df[expected_features]

    return df