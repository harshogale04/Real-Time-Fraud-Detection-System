import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

from config import DATA_PATH, SCALER_PATH, TEST_DATA_PATH, RANDOM_STATE, TEST_SIZE


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.4f}%)")
    return df


def preprocess(df: pd.DataFrame):
    df = df.copy()

    # Scale 'Amount' and 'Time' — the rest (V1-V28) are already PCA-scaled
    scaler = StandardScaler()
    df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Save scaler for reuse in API / stream simulator
    joblib.dump(scaler, SCALER_PATH)

    # Save raw (unscaled features + label) test set for the stream simulator
    test_df = X_test.copy()
    test_df["Class"] = y_test.values
    test_df.to_csv(TEST_DATA_PATH, index=False)

    return X_train, X_test, y_train, y_test


def balance_data(X_train, y_train):
    print("Before SMOTE:", y_train.value_counts().to_dict())
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print("After SMOTE:", y_res.value_counts().to_dict())
    return X_res, y_res


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess(df)
    X_res, y_res = balance_data(X_train, y_train)
    print("Preprocessing complete.")