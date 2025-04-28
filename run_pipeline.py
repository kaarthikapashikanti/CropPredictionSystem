import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.runner import PreProcessing


def run_pipeline():
    print(f"starting the pipeline")
    print(f"Starting the preprocessing")
    process = PreProcessing()
    process.run_preprocessing()


if __name__ == "__main__":
    run_pipeline()
