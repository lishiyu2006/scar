import pandas as pd

def prepare_columns_for_loading(file_path: str, sep: str = ",") -> tuple:
    """
    Read only the column names of the expression matrix without loading data.
    Return the index name and the list of data columns.
    """
    try:
        # Read column names only, do not load data
        df = pd.read_csv(file_path, sep=sep, nrows=0)
        # Convert column names to list of strings
        columns = df.columns.tolist() 

        index_name = columns[0]         # First column is index
        data_cols = columns[1:]

        if not data_cols:
            raise ValueError("No data columns detected")

        return index_name, data_cols

    except Exception as e:
        raise ValueError(f"Failed to read file column info: {e}")
