from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.stats import zscore


class OutlierDetectionTemplate(ABC):

    @abstractmethod
    def detect_outliers(self, df_features: pd.DataFrame, threshold: int):
        pass


class ZScoreOutlierDetection(OutlierDetectionTemplate):
    def detect_outliers(self, df_features: pd.DataFrame, threshold=3):
        z_scores = np.abs(zscore(df_features))
        outliers = z_scores > threshold
        return outliers


class IQROutlierDetection(OutlierDetectionTemplate):
    def detect_outliers(self, df_features: pd.DataFrame, threshold=1.5):
        Q1 = df_features.quantile(0.25)
        Q3 = df_features.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (df_features < (Q1 - threshold * IQR)) | (
            df_features > (Q3 + threshold * IQR)
        )
        return outliers


class OutlierDetector:
    def __init__(self, strategy: OutlierDetectionTemplate):
        self._strategy = strategy

    def set_Strategy(self, strategy: OutlierDetectionTemplate):
        self._strategy = strategy

    def analyze_outliers(self, df: pd.DataFrame, target_column: str):
        print("Outliers Detection step started")
        df_features = df.drop(columns=[target_column])
        return self._strategy.detect_outliers(df_features)
