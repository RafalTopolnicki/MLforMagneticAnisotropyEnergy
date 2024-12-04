import pandas as pd


def categorical_features(df: pd.DataFrame) -> list[str]:
    cat_features = [var for var in df.columns if df[var].dtype == "O"]
    return cat_features
