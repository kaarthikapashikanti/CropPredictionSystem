from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


# Abstract Base Class for Feature Engineering Strategy
# ----------------------------------------------------
# This class defines a common interface for different feature engineering strategies
# SubClass must implement the apply_transfoemation method.
class FeatureEngineeringStrategy(ABC):
    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to apply the feature engineering transformation to the DataFrame

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform

        Returns:
        pd.DataFrae: A dataframe with the appiled transformation
        """
        pass


class LogTransformation(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the LogTransformation with the specific features to transform

        Parameters:
        features (list): The list of features to apply the log transformation to.
        """
        self.features = features

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Appiles a log transformation to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with log-transformed features.
        """
        print(f"Applying log transformation to features: {self.features}")
        df_transformed = df.copy()
        for feature in self.features:
            df_transformed[feature] = np.log1p(
                df[feature]
            )  # log1p handles log(0) by calculating log(1+x)
        return df_transformed


# Concrete Strategy for Standard Scaling
# --------------------------------------
# This strategy appiles standard scaling (z-score normalization) to features, centering them around zero with unit variance.
class StandardScaling(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the StandardScaling with the specific features to scale.

        Parameters:
        features (lists): The list of features to apply the standardScaling.
        """
        self.features = features
        self.scaler = StandardScaler()

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Appiles standard scaling to the specified features in the DataFrame.

        Paramters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with scaled features.
        """
        print(f"Appyling standard scaling to features: {self.features}")
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        return df_transformed


# concrete strategy for Min-Max Scaling
# -------------------------------------
# This Strategy appiles Min-Max scaling to features , scaling them to a specified range, typically [0.1].
class MinMaxScaling(FeatureEngineeringStrategy):
    def __init__(self, features, feature_range=(0, 1)):
        """
        Initializes the MinMaxScaling with the specific features to scale and the target range.

        Parameters:
        features (list): The list of features to apply the Min-Max Scaling to.
        feature_range (tuple): The target range for scaling, default is (0,1).

        """
        self.features = features
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Appiles Min-Max Scaling to the specified features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with Min-Max scaled features.
        """
        print(
            f"Applying Min-MAx Scaling to features: {self.features} with range {self.scaler.feature_range}"
        )
        df_transformed = df.copy()
        df_transformed[self.features] = self.scaler.fit_transform(df[self.features])
        return df_transformed


# Concrete Strategy for One-Hot Encoding
# --------------------------------------
# This strategy appiles one-hot encoding to categorical features, converting them into binary  vectors.
class OneHotEncoding(FeatureEngineeringStrategy):
    def __init__(self, features):
        """
        Initializes the OneHotEncoding with the specific features to encode.

        Parameters:
        features (list): The list of categorical features to apply the one-hot encoding to.
        """
        self.features = features
        self.encoder = OneHotEncoder(sparse=False, drop="first")

    def apply_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Appiles one-hot encoding to the specified categorical features in the DataFrame.

        Parameters:
        df (pd.DataFrame): The dataframe containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with one-hot encoded features.
        """
        print(f"Applying one-hot encoding to features : {self.features}")
        df_transformed = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self.features]),
            columns=self.encoder.get_feature_names_out(self.features),
        )
        df_transformed = df_transformed.drop(columns=self.features).reset_index(
            drop=True
        )
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        return df_transformed


# Context Class for Feature Engineering
# -------------------------------------
# This class uses a FeatureEnginneringStrategy to apply transformations to a dataset.
class FeatureEnginner:
    def __init__(self, strategy: FeatureEngineeringStrategy):
        """
        Initializes the FeatureEngineer with a specific feature engineering strategy.

        Parameters:
        strategy (FeatureEnginneringStrategy): The Strategy to be used for feature enginnering.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureEngineeringStrategy):
        """
        Sets a new strategy for the FeatureEngineer.

        Parameters:
        Strategy (FeatureEnginneringStrategy): The new Strategy to be used for feature engineering.
        """
        print("Switching feature engineering startegy")
        self._strategy = strategy

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Exceutes the feature engineering transformation using the current strategy.

        Parameters:
        df (pd.DAtaFrame): The DataFrame containing features to transform.

        Returns:
        pd.DataFrame: The dataframe with appiled feature engineering transformations.
        """
        print("Applying feature engineering strategy.")
        return self._strategy.apply_transformation(df)
