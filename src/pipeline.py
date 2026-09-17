import pandas as pd
import numpy as np


VARIABLES = ["TOF", "AGE", "R7", "R8", "R17", "R22", "R32"]


def load_csv(data_path: str = "data/farms_test.csv") -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv(data_path)
    return df


def create_z_score(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std()


def create_min_max_score(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.min()) / (df.max() - df.min())


def pipeline(
    test_path: str = "data/farms_test.csv",
    train_path: str = "data/farms_train.csv",
    variables=VARIABLES,
    z_score: bool = False,
    min_max: bool = False,
round = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    df_train: pd.DataFrame = load_csv(train_path)
    df_test: pd.DataFrame = load_csv(test_path)

    df_train = df_train.dropna()
    df_test = df_test.dropna()

    x_train = df_train[variables]
    x_test = df_test[variables]

    if z_score:
        x_train = create_z_score(x_train)
        x_test = create_z_score(x_test)
    elif min_max:
        x_train = create_min_max_score(x_train)
        x_test = create_min_max_score(x_test)

    x_train = np.round(x_train.to_numpy(),decimals= round)
    x_test = np.round(x_test.to_numpy(),decimals=round)
    y_train = df_train["DIFF"].to_numpy()

    return x_train, y_train, x_test


if __name__ == "__main__":

    print(f"test pipeline x_train:\n {pipeline(z_score=True)[0]}")
    print(f"test pipeline y_train:\n {pipeline()[1]}")
    print(f"test pipeline x_test:\n {pipeline()[2]}")
    

