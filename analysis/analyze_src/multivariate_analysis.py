from abc import ABC, abstractmethod

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Abstract Base Class for MultiVariate Analysis
# ---------------------------------------------
# This class defines a template for performing multivariate analsyis
# Subclass can override specific steps like correlation heatamap
class MultiVariateAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Performs a comprehensive multivaraite analysis by generating a correlation and pairplot

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data to be analyzed

        Returns:
        None: This method orchestartes the multivariate analysis
        """
        self.generate_correlation_heatmap(df)
        self.generate_pairplot(df)

    @abstractmethod
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generate and displays a heatmap of the correlations between features/columns of dataFrame

        Parameters:
        df (pd.DataFrame): The dataframe containing the data that to be analyzed

        Returns:
        None: Generate and displays a correlation heatmap
        """
        pass

    @abstractmethod
    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generate and displays a pairplot of the selected features/columns of dataFrame

        Parameters:
        df (pd.DataFrame): The dataframe containing the data that to be analyzed

        Returns:
        None: Generate and displays a pairplot
        """
        pass


# Concrete Class for Multivariate Analysis with Correlation HeatMap and pairplot
# -------------------------------------------------------------------------------
# This class implements the methods to generate a correlation heatmap and pairplot
class SimpleMultiVariateAnalysis(MultiVariateAnalysisTemplate):
    def generate_correlation_heatmap(self, df: pd.DataFrame):
        """
        Generate and displays a heatmap of the correlations between features/columns of dataFrame

        Parameters:
        df (pd.DataFrame): The dataframe containing the data that to be analyzed

        Returns:
        None: Generate and displays a correlation heatmap between multiple features
        """
        plt.figure(figsize=(20, 10))
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidth=0.5)
        plt.title("Correlation Heatmap")
        plt.show()

    def generate_pairplot(self, df: pd.DataFrame):
        """
        Generate and displays a pairplot of the selected features/columns of dataFrame

        Parameters:
        df (pd.DataFrame): The dataframe containing the data that to be analyzed

        Returns:
        None: Generate and displays a pairplot for the selected features
        """
        sns.pairplot(df)
        plt.suptitle("Pair plot of selected Features", y=1.02)
        plt.show()
