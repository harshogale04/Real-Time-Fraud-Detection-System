import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from config import MODEL_PATH, RANDOM_STATE
from preprocess import load_data, preprocess, balance_data


def train_models(X_train, y_train, X_test, y_test):
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    best_model, best_score, best_name = None, -1, None

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        score = average_precision_score(y_test, probs)  # PR-AUC — right metric for imbalance
        print(f"{name} PR-AUC: {score:.4f}")

        if score > best_score:
            best_model, best_score, best_name = model, score, name

    print(f"\nBest model: {best_name} (PR-AUC: {best_score:.4f})")
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved to {MODEL_PATH}")
    return best_model


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    X_res, y_res = balance_data(X_train, y_train)
    train_models(X_res, y_res, X_test, y_test)