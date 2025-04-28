from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from sklearn.svm import SVC
from sklearn import metrics


class Score:
    def calculate_score(y_test, predicted_values):
        x = metrics.accuracy_score(y_test, predicted_values)
        return x


class ModelBuildingStrategy(ABC):
    @abstractmethod
    def build_train_test_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ):
        pass

    def get_model(self):
        pass


class DecisionTreeModel(ModelBuildingStrategy):
    def __init__(self):
        self._model = DecisionTreeClassifier(
            criterion="entropy", random_state=2, max_depth=5
        )

    def build_train_test_model(self, X_train, y_train, X_test, y_test):
        self._model.fit(X_train, y_train)
        predicted_values = self._model.predict(X_test)
        x = Score.calculate_score(y_test, predicted_values)
        return x

    def get_model(self):
        return self._model


class NaiveBayesModel(ModelBuildingStrategy):
    def __init__(self):
        self._model = GaussianNB()

    def build_train_test_model(self, X_train, y_train, X_test, y_test):

        self._model.fit(X_train, y_train)

        predicted_values = self._model.predict(X_test)
        x = Score.calculate_score(y_test, predicted_values)
        return x

    def get_model(self):
        return self._model


class SVMModel(ModelBuildingStrategy):
    def __init__(self):
        self._model = SVC(kernel="poly", degree=3, C=1)

    def build_train_test_model(self, X_train, y_train, X_test, y_test):
        self._model.fit(X_train, y_train)
        predicted_values = self._model.predict(X_test)
        x = Score.calculate_score(y_test, predicted_values)
        return x

    def get_model(self):
        return self._model


class LogisticRegressionModel(ModelBuildingStrategy):
    def __init__(self):
        self._model = LogisticRegression(random_state=2)

    def build_train_test_model(self, X_train, y_train, X_test, y_test):
        self._model.fit(X_train, y_train)

        predicted_values = self._model.predict(X_test)

        x = Score.calculate_score(y_test, predicted_values)
        return x

    def get_model(self):
        return self._model


class RandomForestModel(ModelBuildingStrategy):
    def __init__(self):
        self._model = RandomForestClassifier(n_estimators=20, random_state=0)

    def build_train_test_model(self, X_train, y_train, X_test, y_test):
        self._model.fit(X_train, y_train)

        predicted_values = self._model.predict(X_test)

        x = Score.calculate_score(y_test, predicted_values)
        return x

    def get_model(self):
        return self._model


class XGBoostModel(ModelBuildingStrategy):
    def __init__(self):
        self._model = xgb.XGBClassifier()

    def build_train_test_model(self, X_train, y_train, X_test, y_test):
        self._model.fit(X_train, y_train)

        predicted_values = self._model.predict(X_test)

        x = Score.calculate_score(y_test, predicted_values)
        return x

    def get_model(self):
        return self._model


class Model:
    def __init__(self, model: ModelBuildingStrategy):
        self._model = model

    def set_model(self, model: ModelBuildingStrategy):
        self._model = model

    def model_developing(self, X_train, y_train, X_test, y_test):
        print(f"model {self._model} training and training started")
        return self._model.build_train_test_model(X_train, y_train, X_test, y_test)
