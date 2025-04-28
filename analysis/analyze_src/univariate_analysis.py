from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Abstract Base class for Univariate Strategy
# ------------------------------------------
# This class defines a common interface for univariate analysis startegies.
# SubClass must implement the analyze method.
class UnivariateAnalaysisStrategy:
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Perform univariate analysis on a specific feature of the dataframe.

        Paramters:
        df (pd.DataFrame): The dataframe containing the data.
        feature(str): The name of the feature/column to be analyzed.

        Returns:
        None: This method visualizes the distribution of the feature
        """
        pass


# Concrete Strategy for Numerical Features
# ----------------------------------------
# This strategy analyzes numerical features by plotting their ditribution.
class NumericalUnivariateAnalysis(UnivariateAnalaysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a numerrical feature using a histogram and KDE
        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature(str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a histogram with  a KDE plot.
        """
        plt.figure(figsize=(20, 10))
        sns.histplot(df[feature], kde=True, bins=30)
        plt.title(f"Ditsribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()


# concrete strategy for Catrgorical Features
# ------------------------------------------
# This Startegy analyzes categorical features by plotting their frequency distribution
class CategoricalUnivariateAnalysis(UnivariateAnalaysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a categroical feature using a bar plot.

        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature(str): The name  of the categorical feature/column to be analyzed.

        Returns:
        None: Displays a bar plot showing the frequency of each category.
        """
        plt.figure(figsize=(20, 6))
        sns.countplot(x=feature, data=df, palette="muted")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()


class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalaysisStrategy):
        """
        Initializes the UnivariateAnalyzer with a specific analysis strategy.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The strategy to be used for univariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalaysisStrategy):
        """
        Sets a new strategy for the UnivariateAnalyzer.

        Parameters:
        strategy (UnivariateAnalysisStrategy): The new strategy to be used for univariate analysis.

        Returns:
        None
        """
        self._strategy = strategy

    def exceute_analysis(self, df: pd.DataFrame, feature: str):
        """
        Plots the distribution of a numerrical feature using a histogram and KDE
        Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        feature(str): The name of the numerical feature/column to be analyzed.

        Returns:
        None: Displays a histogram with  a KDE plot.
        """
        print(feature)
        self._strategy.analyze(df, feature)
