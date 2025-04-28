from abc import ABC, abstractmethod
import pandas as pd


# Abstract Base Class for Data Inspection Strategies
# --------------------------------------------------
# This class defines a common interface for data inspection stratergies.
# SubClasses must implement the inspect method
class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Perform a specific type of data inspection.

        Parameters:
        df (pd.DataFrame): The dataframe on which the inspection is to be perform

        Returns:
        None: This method prints the inspection results directly
        """
        pass


# Concrete Strategy for Data Types Inspection
# -------------------------------------------
# This strategy inspects the data types of each column and count
class DataTypesInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspects and prints the data types and non-null counts of the dataframe column

        Paramters:
        df (pd.DataFrame): The dataframe to be inspected

        Returns:
        None: Prints the data types and non-null counts to the console
        """
        print("\nData Types and Non-null counts:")
        print(df.info())


# Concrete Strategy for Summary Statistics Inspection
# ---------------------------------------------------
# This strategy provides summary statisttics for both numerical and categorical features
class SummaryInspectionStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Prints summary statistics for numerical and categorical features

        Parameters:
        df (pd.DataFrame): The dataframe to be inspected

        Returns:
        None: Prints the summary statistics to the console
        """
        numerical_columns = df.select_dtypes(include=["number"]).columns
        if len(numerical_columns) > 0:
            print("\nSummary Statistics (Numerical Features): ")
            print(numerical_columns)
            print(df[numerical_columns].describe())
        else:
            print("No Numerical columns in the dataset")

        categorical_columns = df.select_dtypes(include=["object"]).columns
        if len(categorical_columns) > 0:
            print("\nSummary Statistics (Categorical Features):")
            print(df.describe(include=["object"]))
        else:
            print("No categorical columns in the dataset")


# Context Class that uses a DataInspectionStrategy
# ------------------------------------------------
# This class allows you to switch between different data inspection strategy
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        """
        Initializies the DataInspector with a specific inspection startegy

        Paramters:
        Strategy (DataInspectionStrategy): The strategy to be used for data Inspection

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Sets a new strategy for the DataInspector.

        Parameters:
        strategy (DataInspectorStrategy): The new strategy to be used for data inspection

        Returns:
        None
        """
        self._strategy = strategy

    def exceute_inspection(self, df: pd.DataFrame):
        """
        Executes the inspection using the current strategy

        Paramters:
        df (pd.DataFrame): The dataframe to be inspected

        Returns:
        None:Exceutes the strategy's inspection method
        """
        self._strategy.inspect(df)
