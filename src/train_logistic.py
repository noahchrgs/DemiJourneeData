import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from src.pipeline import pipeline, VARIABLES
from tqdm import tqdm

def test_models(
    train_path: str = "data/farms_train.csv",
    test_path: str = "data/farms_test.csv",
    variables: list = VARIABLES,
    alphas: list = None,
    z_score: bool = True,
    min_max: bool = False,
    cv_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Teste plusieurs LogisticRegression en faisant varier :
      - alpha (force de régularisation, via C = 1/alpha)
      - le nombre de variables utilisées (les n premières de `variables`)

    Retourne un DataFrame avec l'AUC moyenne (et std) pour chaque combinaison,
    trié par AUC décroissante.
    """

    if alphas is None:
        alphas = np.logspace(-3, 3, 13)

    results = []

    n_vars_range = range(1, len(variables) + 1)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    pbar_vars = tqdm(n_vars_range, desc="n_vars")
    for n_vars in pbar_vars:
        pbar_vars.set_description(f"training pour n_vars={n_vars}")
        subset_vars = variables[:n_vars]

        x_train, y_train, _ = pipeline(
            train_path=train_path,
            test_path=test_path,
            variables=subset_vars,
            z_score=z_score,
            min_max=min_max,
        )

        pbar_alphas = tqdm(alphas, desc="alpha", leave=False)
        for alpha in pbar_alphas:
            pbar_alphas.set_description(f"training pour alpha={alpha:.4f}")
            C = 1.0 / alpha

            model = LogisticRegression(C=C, max_iter=1000)

            try:
                scores = cross_val_score(
                    model, x_train, y_train, cv=skf, scoring="roc_auc"
                )
                auc_mean = scores.mean()
                auc_std = scores.std()
            except Exception as e:
                auc_mean, auc_std = np.nan, np.nan
                print(f"[WARN] échec pour n_vars={n_vars}, alpha={alpha}: {e}")

            results.append(
                {
                    "n_vars": n_vars,
                    "variables": subset_vars,
                    "alpha": alpha,
                    "C": C,
                    "auc_mean": auc_mean,
                    "auc_std": auc_std,
                }
            )

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("auc_mean", ascending=False).reset_index(drop=True)

    return df_results

if __name__ == "__main__":

    df_results = test_models()

    print("Top 10 des meilleures combinaisons (n_vars, alpha) par AUC :\n")
    print(df_results.head(10).to_string(index=False))

    best = df_results.iloc[0]
    print(f"\nMeilleure config : n_vars={best['n_vars']}, alpha={best['alpha']:.4f} "
          f"-> AUC moyenne = {best['auc_mean']:.4f} (+/- {best['auc_std']:.4f})")