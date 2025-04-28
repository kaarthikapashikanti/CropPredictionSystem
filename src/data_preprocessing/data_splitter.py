from abc import ABC, abstractmethod
import pandas as pd
from sklearn.model_selection import train_test_split


# Abstract Base Class for Data Splitting Startegy
# ------------------------------------------------
# This class defines a common interface for different data splitting strategies
# Subclasses must implement the split_data method.
class DataSplittingStartegy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str):
        """
        Abstract method to split the data into training and testing sets.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column(str): The name of the target column.


        Returns:
        X_train, X_test, y_train, y_test: The training and testing splits  for features and target.
        """
        pass


# Concrete Strategy for Simple Train-Test Split
# ---------------------------------------------
# This strategy implements a simple train-test split.
class SimpleTrainTestSplitStrategy(DataSplittingStartegy):
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initializes the SimpleTrainTestSplitStrategy with specific parameters.

        Parameters:
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used by the random number generator.
        """
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame, target_column: str):
        """
        Splits the data into training and testing sets using a simple train-test split.

        Parameters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the target column.

        Returns:
        X_train,X_test,y_train,y_test: The training and testing splits for features and target.
        """
        print("Peforming simple train-test split.")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        print("Train-test split completed")
        return X_train, X_test, y_train, y_test


# Context class for Data Splitting
# -------------------------------
# This class uses a DataSplittingStrategy to split the data
class DataSplitter:
    def __init__(self, strategy: DataSplittingStartegy):
        """
        Initializes the DataSplitter with a specific data splitting strategy.

        Parameters:
        Strategy (DataSplittingStrategy): The Strategy to be used for data splitting.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataSplittingStartegy):
        """
        Sets a new Strategy for the DataSplitter.

        Parameters:
        Strategy (DataSplittingstrategy): The new strategy to be used for splitting the data.
        """
        print("Switiching the data Splitting strategy")
        self._strategy = strategy

    def split(self, df: pd.DataFrame, target_column: str):
        """
        Executes the data splitting using the current strategy.

        Paramters:
        df (pd.DataFrame): The input DataFrame to be split.
        target_column (str): The name of the traget column.

        Returns:
        X_train,X_test,y_train,y_test: The training and testing splits for features and target.
        """
        print("Splitting data using the selected strategy")
        return self._strategy.split_data(df, target_column)
