import pandas as pd
from typing import List, Tuple
from fedml_databricks import DbConnection
from sklearn.preprocessing import LabelEncoder


def get_data(view_name: str, config_path, csv_path="") -> Tuple[pd.DataFrame, LabelEncoder, List]:
    """
    Given a view name, return the associated dataset from DSP as a dataframe
    
    Must also pass in the path to the DBConnection
    
    If a csv path is passed it, load data from the csv instead of DSP
    """
    if len(csv_path) == 0:
        db = DbConnection(url=config_path)

        query = f"select * from \"SCE\".\"{view_name}\""
        result = db.execute_query(query)

        rows = result[0]
        headers = result[1]

        df = pd.DataFrame(rows, columns=headers)
    else:
        df = pd.read_csv(csv_path)

    # identify object columns
    object_columns = df.select_dtypes(include='object').columns

    # convert object columns to numeric if possible
    for col in object_columns:
        # it's fine if we can't convert a column, since non-numeric types will be encoded
        # thus we fail silently
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    # encode non-numeric object columns
    encoded_cols = list()
    for col in object_columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoded_cols.append(col)

    return df, le, encoded_cols