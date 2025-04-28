from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Abstract Base class for Missing Values analysis
# -----------------------------------------------
# This class defines a template for missing values analysis.
# Subclasses must implement the methods to identify and visualize the patterns
class MissingValuesAnalysisTemplate(ABC):
    def analyze(self, df: pd.DataFrame):
        """
        Performs a complete missing values analysis by indentifying and visualizing the missing values

        Parameters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        None: This methods performs the analysis and visualizes the missing values
        """
        self.identify_missing_values(df)
        self.visualize_missing_values(df)

    @abstractmethod
    def identify_missing_values(self, df: pd.DataFrame) -> bool:
        """
        Identifies the missing values in the dataframe

        Paramters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        bool: This method print the count of missing values in the dataframe if no missing values returns False
        """
        pass

    @abstractmethod
    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Visualize the missing values in the dataframe

        Paramters:
        df (pd.DataFrame): The dataframe to be Visualized.

        Returns:
        None: This method create a visualization.

        """
        pass


# Concrete Class for Missing Values Identification
# ------------------------------------------------
# This class implements methods to identify and visualize missing values
class SimpleMissingValuesAnalysis(MissingValuesAnalysisTemplate):
    def identify_missing_values(self, df: pd.DataFrame) -> bool:
        """
        Identifies the missing values in the dataframe

        Paramters:
        df (pd.DataFrame): The dataframe to be analyzed.

        Returns:
        bool: This method print the count of missing values in the dataframe if no missing values returns False
        """
        missing_values = df.isnull().sum()
        if missing_values.sum() == 0:
            print("No Null values in the Dataset")
            return False
        print("\nMissing values count by column: ")
        print(missing_values[missing_values > 0])
        return True

    def visualize_missing_values(self, df: pd.DataFrame):
        """
        Visualize the missing values in the dataframe

        Paramters:
        df (pd.DataFrame): The dataframe to be Visualized.

        Returns:
        None: This method create a visualization.

        """
        print("\nVisualizing Missing Values....")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing values Heatmap")
        plt.show()
