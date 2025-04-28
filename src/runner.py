import pandas as pd
import os
import sys
import joblib
from scipy.stats import zscore
import numpy as np
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from .data_preprocessing.Encoder import Encoder, LabelEncoder
from .data_preprocessing.ingest_data import ingest_df
from .data_preprocessing.feature_engineering import (
    FeatureEnginner,
    OneHotEncoding,
    MinMaxScaling,
    StandardScaling,
    LogTransformation,
)
from .data_preprocessing.outlier_detection import (
    OutlierDetector,
    ZScoreOutlierDetection,
    IQROutlierDetection,
)
from .data_preprocessing.outlier_handler import handle_outliers
from .data_preprocessing.data_splitter import DataSplitter, SimpleTrainTestSplitStrategy

from ml.model import (
    DecisionTreeModel,
    NaiveBayesModel,
    SVMModel,
    LogisticRegressionModel,
    RandomForestModel,
    XGBoostModel,
    Model,
)


class PreProcessing:

    def run_encoding(self, df):
        encoding = Encoder(LabelEncoder())
        label_mappings = {}
        for col in df.select_dtypes(include=["object"]).columns:
            label_mappings[col] = encoding.execute(df, col)
            print(
                f"Encoding of {col} feature completed the {col} mapping are {label_mappings[col]}"
            )
        return label_mappings

    def run_preprocessing(self):
        try:
            df = ingest_df(
                "D:\Projects\CropPredictionSystem\data\crop_recommendation.csv"
            )
            # ------------------------------------------------------------------------------
            if len(df) > 0:
                print(f"Extracted the dataframe successfully")
                label_mappings = self.run_encoding(df)
                with open("ml/saved_model/label_mappings.json", "w") as f:
                    json.dump(label_mappings, f)

                features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
                target_column = "label_Encoded"
                feature_engineering = FeatureEnginner(StandardScaling(features))
                transformed_df = feature_engineering.apply_feature_engineering(df)
                print("Feature Engineering step completed")

                # -----------------------------------------------------------------------
                outliers_detection = OutlierDetector(IQROutlierDetection())
                outliers = outliers_detection.analyze_outliers(
                    transformed_df, target_column
                )
                print(
                    f"Outlier Detection step completed. There are {outliers.sum().sum()} outliers"
                )
                print(f"Outliers column wise count")
                print(outliers.sum())
                cleaned_df = handle_outliers(
                    transformed_df, outliers, target_column, "cap", 1.5
                )
                outliersAfterCleaning = outliers_detection.analyze_outliers(
                    cleaned_df, target_column
                )
                print(
                    f"Handling the outliers step completed.There are {outliersAfterCleaning.sum().sum()} outliers "
                )
                # ----------------------------------------------------------------------------------------------------------
                split_data = DataSplitter(SimpleTrainTestSplitStrategy())
                X_train, X_test, y_train, y_test = split_data.split(
                    cleaned_df, target_column
                )
                print(f"Splitting the data step completed")
                print(
                    f"X_train : {X_train.shape}  X_test : {X_test.shape},  y_train : {y_train.shape},  y_test : {y_test.shape}"
                )
                # ------------------------------------------------------------------
                models = {
                    "DecisionTree": DecisionTreeModel,
                    "NaiveBayes": NaiveBayesModel,
                    "SVM": SVMModel,
                    "LogisticRegression": LogisticRegressionModel,
                    "RandomForest": RandomForestModel,
                    "XGBoost": XGBoostModel,
                }
                maximumAccuracyModel = None
                maximumAccuracyScore = float("-inf")
                best_model_instance = None
                m = Model(DecisionTreeModel)
                for name, model in models.items():
                    model_instance = model()
                    m.set_model(model_instance)
                    score = m.model_developing(X_train, y_train, X_test, y_test)
                    if score > maximumAccuracyScore:
                        maximumAccuracyModel = name
                        maximumAccuracyScore = score
                        best_model_instance = model_instance
                print(
                    f"Maximum Accuracy Model is {maximumAccuracyModel} with score {maximumAccuracyScore}"
                )
                print("Model Training and testing completed")
                # --------------------------------------------------------------
                print(f"Saving the Navie Bayes model")
                model_path = os.path.abspath(
                    os.path.join("ml", "saved_model", maximumAccuracyModel + ".pkl")
                )
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                joblib.dump(best_model_instance.get_model(), model_path)
                print(f"{maximumAccuracyModel} Model is saved at : {model_path}")
        except FileNotFoundError:
            print(f"File Not Found")

        except Exception as e:
            print(f"Error in preprocessing data ingestion step {e}")
            raise e
