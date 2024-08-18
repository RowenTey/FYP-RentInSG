import pandas as pd
from io import BytesIO

def parquet(df: pd.DataFrame) -> BytesIO:
    # Code from https://github.com/pandas-dev/pandas/issues/51140
    data: BytesIO = BytesIO()

    # Monkey patch the close method to prevent the file from being closed
    orig_close = data.close
    data.close = lambda: None
    try:
        df.to_parquet(data, compression="gzip", index=False)
    finally:
        data.close = orig_close
    
    # Reset the buffer position to the start
    data.seek(0) 

    return data
