import pandas as pd
import numpy as np 


def load_csv(data_path:str = "data/farms_test.csv") -> pd.DataFrame: 
    df:pd.DataFrame = pd.read_csv(data_path)

    return df



if __name__ == "__main__":
    print(load_csv())