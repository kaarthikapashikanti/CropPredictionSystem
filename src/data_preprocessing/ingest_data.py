import pandas as pd


class IngestData:
    def __init__(self, path: str) -> None:
        self.path = path

    def get_data(self):
        print(f"Ingesting the data from the .csv file")
        return pd.read_csv(self.path)


def ingest_df(data_path: str) -> pd.DataFrame:
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        print(f"Error while ingesting the data {e}")
