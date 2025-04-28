from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Abstract Base Class for Bivariate Analysis Strategy
# ---------------------------------------------------
# This class defines a common interface for bivariate analysis
# SubClass must implement the analyze methos


class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Perform bivariate analysis on two features of the dataframe

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1(str): The name of the first feature/column to be analyzed
        feature2(str): The name of the second feature/column to be analyzed

        Returns:
        None: This method analysis the relationship between the two features
        """
        pass


# Concrete Strategy for Numerical Vs Numerical Analysis
# -----------------------------------------------------
class NumericalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between two numerical features of the dataframe

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1(str): The name of the first numerical feature/column to be analyzed
        feature2(str): The name of the second numerical feature/column to be analyzed

        Returns:
        None: Displays a scatter plot showing the relationship between the two numerical features
        """
        plt.figure(figsize=(20, 10))
        # sns.scatterplot(x=feature1, y=feature2, data=df)
        sns.histplot(df, x=feature1, hue=feature2, kde=True, bins=30, alpha=0.6)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel("count")
        plt.show()


# Concrete Strategy for Numerical Vs Categorical Analysis
# -----------------------------------------------------
class CategoricalVsNumericalAnalysis(BivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Plots the relationship between  numerical and categorical features of the dataframe

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1(str): The name of the categorical feature/column to be analyzed
        feature2(str): The name of the numerical feature/column to be analyzed

        Returns:
        None: Displays a box plot showing the relationship between the  numerical and categorical features
        """
        plt.figure(figsize=(20, 10))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()


# Context Class that uses a BivariateAnalysisStrategy
# ------------------------------------------------
# This class allows you to switch between different Biivariate analysis strategies.
class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy):
        """
        Initializes the bivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (BivariateAnalysisStrategy): The strategy to be used for Bivariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy):
        """
        Sets a new strategy for the BivariateAnalyzer.

        Parameters:
        strategy (BivariateAnalysisStrategy): The new strategy to be used for Bivariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_analysis(self, df: pd.DataFrame, feature1: str, feature2: str):
        """
        Executes the Bivariate analysis using the current strategy.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature1 (str): The name of the first feature/column to be analyzed.
        feature2 (str): The name of the second feature/column to be analyzed.

        Returns:
        None: Executes the strategy's analysis method and visualizes the results.
        """
        self._strategy.analyze(df, feature1, feature2)
