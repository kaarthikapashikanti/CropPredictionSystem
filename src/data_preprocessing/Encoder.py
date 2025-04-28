from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder


class EncoderTemplate(ABC):
    @abstractmethod
    def apply_encoding(self, df: pd.DataFrame, feature: str) -> str:
        pass


class LabelEncoder(EncoderTemplate):
    def apply_encoding(self, df: pd.DataFrame, feature: str) -> str:
        print(f"Encoding the {feature} with technique LabelEncoding")
        label_mapping = {label: idx for idx, label in enumerate(df[feature].unique())}
        encoder = SklearnLabelEncoder()  # ✅ Use renamed import
        col_name = feature + "_Encoded"
        df[col_name] = encoder.fit_transform(df[feature])
        df.drop(columns=[feature], inplace=True)
        return label_mapping


class Encoder:
    def __init__(self, strategy: EncoderTemplate):
        self._strategy = strategy

    def set_strategy(self, strategy: EncoderTemplate):
        print(f"Changing the {self._strategy} strategy to {strategy} Strategy")
        self._strategy = strategy

    def execute(self, df: pd.DataFrame, feature: str) -> str:
        return self._strategy.apply_encoding(df, feature)
