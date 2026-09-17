import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from src.pipeline import pipeline, VARIABLES
from tqdm import tqdm


def test_models(
    train_path: str = "data/farms_train.csv",
    test_path: str = "data/farms_test.csv",
    variables: list = VARIABLES,
    alphas: list = None,
    thresholds: list = None,
    z_score: bool = True,
    min_max: bool = False,
    cv_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:

    if alphas is None:
        alphas = np.logspace(-3, 3, 13)

    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)

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
                proba = cross_val_predict(
                    model, x_train, y_train, cv=skf, method="predict_proba"
                )[:, 1]
                auc = roc_auc_score(y_train, proba)
            except Exception as e:
                proba = None
                auc = np.nan
                print(f"[WARN] échec pour n_vars={n_vars}, alpha={alpha}: {e}")

            pbar_thresholds = tqdm(thresholds, desc="threshold", leave=False)
            for threshold in pbar_thresholds:
                pbar_thresholds.set_description(f"threshold={threshold:.2f}")

                if proba is not None:
                    y_pred = (proba >= threshold).astype(int)
                    f1 = f1_score(y_train, y_pred, zero_division=0)
                    accuracy = accuracy_score(y_train, y_pred)
                else:
                    f1 = accuracy = np.nan

                results.append(
                    {
                        "n_vars": n_vars,
                        "variables": subset_vars,
                        "alpha": alpha,
                        "C": C,
                        "threshold": round(threshold, 2),
                        "auc": auc,
                        "f1": f1,
                        "accuracy": accuracy,
                    }
                )

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(
        ["auc", "f1"], ascending=[False, False]
    ).reset_index(drop=True)

    print("Top 10 des meilleures combinaisons (n_vars, alpha, threshold) par AUC puis F1 :\n")
    print(
        df_results[["n_vars", "alpha", "threshold", "auc", "f1", "accuracy"]]
        .head(10)
        .to_string(index=False)
    )

    best_auc = df_results.iloc[0]
    print(
        f"\nMeilleure config (AUC) : n_vars={best_auc['n_vars']}, alpha={best_auc['alpha']:.4f}, "
        f"threshold={best_auc['threshold']} -> AUC = {best_auc['auc']:.4f}, F1 = {best_auc['f1']:.4f}"
    )

    best_f1 = df_results.sort_values("f1", ascending=False).iloc[0]
    print(
        f"Meilleure config (F1)  : n_vars={best_f1['n_vars']}, alpha={best_f1['alpha']:.4f}, "
        f"threshold={best_f1['threshold']} -> AUC = {best_f1['auc']:.4f}, F1 = {best_f1['f1']:.4f}"
    )

    return df_results

def get_best_model(
    train_path: str = "data/farms_train.csv",
    test_path: str = "data/farms_test.csv",
    variables: list = VARIABLES,
    alphas: list = None,
    thresholds: list = None,
    z_score: bool = True,
    min_max: bool = False,
    cv_folds: int = 5,
    random_state: int = 42,
    select_by: str = "auc",
) -> dict:
    """
    Cherche la meilleure combinaison (n_vars, alpha, threshold) via test_models,
    puis réentraîne un modèle final sur l'ensemble des données d'entraînement
    (sans découpage cross-validation, cette fois) avec ces paramètres.
 
    select_by : "auc" ou "f1" -> critère utilisé pour choisir la meilleure ligne.
 
    Retourne un dict contenant :
      - "model"     : le LogisticRegression entraîné, prêt à l'usage
      - "variables" : la liste des variables utilisées (à réutiliser pour predict)
      - "alpha"     : alpha choisi
      - "threshold" : seuil de décision choisi
      - "auc"       : AUC obtenue en cross-validation avec cette config
      - "f1"        : F1 obtenu en cross-validation avec cette config
    """
 
    df_results = test_models(
        train_path=train_path,
        test_path=test_path,
        variables=variables,
        alphas=alphas,
        thresholds=thresholds,
        z_score=z_score,
        min_max=min_max,
        cv_folds=cv_folds,
        random_state=random_state,
    )
 
    if select_by == "auc":
        df_sorted = df_results.sort_values(["auc", "f1"], ascending=[False, False])
    elif select_by == "f1":
        df_sorted = df_results.sort_values(["f1", "auc"], ascending=[False, False])
    else:
        raise ValueError("select_by doit valoir 'auc' ou 'f1'")
 
    best = df_sorted.iloc[0]
 
    best_variables = best["variables"]
    best_alpha = best["alpha"]
    best_threshold = best["threshold"]
 
    x_train, y_train, _ = pipeline(
        train_path=train_path,
        test_path=test_path,
        variables=best_variables,
        z_score=z_score,
        min_max=min_max,
    )
 
    C = 1.0 / best_alpha
 
    final_model = LogisticRegression(C=C, max_iter=1000)
    final_model.fit(x_train, y_train)
 
    print(
        f"Meilleur modèle sélectionné par '{select_by}' :\n"
        f"  variables  = {best_variables}\n"
        f"  alpha      = {best_alpha:.4f}\n"
        f"  threshold  = {best_threshold}\n"
        f"  AUC (cv)   = {best['auc']:.4f}\n"
        f"  F1 (cv)    = {best['f1']:.4f}"
    )
 
    return {
        "model": final_model,
        "variables": best_variables,
        "alpha": best_alpha,
        "threshold": best_threshold,
        "auc": best["auc"],
        "f1": best["f1"],
    }
 
def predict_new_labels(model,variables,threshold,train_path: str = "data/farms_train.csv",
    test_path: str = "data/farms_test.csv",z_score = False,min_max = False):
    _,_,df_test = pipeline(
            train_path=train_path,
            test_path=test_path,
            variables=variables,
            z_score=z_score,
            min_max=min_max,
        )

    res = (model.predict(df_test) >= threshold).astype(int)
    pd.DataFrame({"ID": [i for i in range(len(res))],"DIFF": res}).set_index("ID").to_csv("res/resultat_logistic.csv")

if __name__ == "__main__":
    df_results = test_models()
    test = get_best_model()

    predict_new_labels(test["model"],test["variables"],test["threshold"])
