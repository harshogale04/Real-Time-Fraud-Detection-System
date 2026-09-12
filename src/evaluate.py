import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    ConfusionMatrixDisplay,
)

from config import MODEL_PATH
from preprocess import load_data, preprocess


def evaluate():
    df = load_data()
    _, X_test, _, y_test = preprocess(df)

    model = joblib.load(MODEL_PATH)
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    print("Classification Report:\n", classification_report(y_test, preds, digits=4))
    print("ROC-AUC:", roc_auc_score(y_test, probs))
    print("PR-AUC:", average_precision_score(y_test, probs))

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, probs)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig("precision_recall_curve.png")
    plt.close()

    print("Saved confusion_matrix.png and precision_recall_curve.png")


if __name__ == "__main__":
    evaluate()