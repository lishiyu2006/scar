import pandas as pd

def prepare_columns_for_loading(file_path: str, sep: str = ",") -> tuple:
    """
    Only read column names of expression matrix without loading data.
    Returns index name and data column list.
    """
    try:
        # Only read column names, do not load data
        df = pd.read_csv(file_path, sep=sep, nrows=0)
        # Convert column names to strings, use default value for NaN
        columns = df.columns.tolist() 

        index_name = columns[0]         # First column is index
        data_cols = columns[1:]

        if not data_cols:
            raise ValueError("No data columns detected")

        return index_name, data_cols

    except Exception as e:
        raise ValueError(f"Unable to read file column information: {e}")
