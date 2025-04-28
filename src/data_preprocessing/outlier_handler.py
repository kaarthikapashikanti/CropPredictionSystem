import numpy as np
import pandas as pd


def handle_outliers(df, outliers, target_column, method="replace", threshold=3):
    print(f"Handling the outliers step started")
    df_features = df.drop(columns=[target_column])  # Exclude target column

    if method == "replace":
        df_features = df_features.mask(
            outliers, df_features.median(), axis=1
        )  # Replace with median
    elif method == "cap":
        Q1 = df_features.quantile(0.25)
        Q3 = df_features.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df_features = df_features.clip(
            lower=lower_bound, upper=upper_bound, axis=1
        )  # Clip outliers
    elif method == "remove":
        df_features = df_features[~outliers.any(axis=1)]  # Remove rows with outliers
        df = df.loc[df_features.index]  # Keep only corresponding target column values
    else:
        raise ValueError("Method must be 'replace', 'cap', or 'remove'")

    df_cleaned = pd.concat(
        [df_features, df[target_column]], axis=1
    )  # Reattach target column
    return df_cleaned
