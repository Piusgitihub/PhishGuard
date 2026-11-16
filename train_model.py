import pandas as pd
import numpy as np
import re
import ipaddress
import joblib
import time

from urllib.parse import urlparse

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

print("PHISHING URL DETECTION - MODEL TRAINING")
print("=" * 50)

# 1. LOAD DATASET
print("\nLoading dataset...")

dataset = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")

dataset = dataset.dropna(subset=["URL", "label"]).copy()
dataset = dataset.drop_duplicates(subset="URL")
dataset = dataset.reset_index(drop=True)
print("Dataset records:", len(dataset))


# 2. EXTRACT URL-ONLY FEATURES
def extract_url_features(url):

    url = str(url).strip()

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path
    query = parsed.query

    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    suspicious_words = [
        "login", "verify", "update", "secure",
        "account", "bank", "password", "signin",
        "confirm", "wallet"
    ]

    try:
        has_port = int(parsed.port is not None)
    except ValueError:
        has_port = 0

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "at_count": url.count("@"),
        "question_count": url.count("?"),
        "equal_count": url.count("="),
        "ampersand_count": url.count("&"),
        "digit_count": sum(c.isdigit() for c in url),
        "subdomain_count": max(0, len(hostname.split(".")) - 2),
        "has_ip_address": has_ip,
        "uses_https": int(parsed.scheme == "https"),
        "suspicious_word_count": sum(
            word in url.lower() for word in suspicious_words
        ),
        "has_port": has_port
    }


print("\nExtracting URL features...")

features_df = pd.DataFrame(
    dataset["URL"].apply(extract_url_features).tolist()
)

# Original labels: 1 = legitimate, 0 = phishing
# Model labels: 1 = phishing, 0 = legitimate
X = features_df
y = (dataset["label"].astype(int) == 0).astype(int)

print("URL features:", len(X.columns))
print("Phishing records:", int(y.sum()))
print("Legitimate records:", int((y == 0).sum()))


# 3. SPLIT DATA: 70% TRAIN, 15% VALIDATION, 15% TEST
print("\nSplitting dataset...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))


# 4. DEFINE THREE MACHINE-LEARNING MODELS
models = {

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=100,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(
            n_estimators=60,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ))
    ])
}


# 5. TRAIN AND COMPARE MODELS ON VALIDATION DATA
results = {}
best_model = None
best_model_name = None
best_f1 = -1

for name, model in models.items():

    print("\n" + "=" * 50)
    print("Training:", name)

    start_time = time.time()

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)

    val_f1 = f1_score(
        y_val, val_predictions,
        zero_division=0
    )

    val_accuracy = accuracy_score(
        y_val, val_predictions
    )

    elapsed = time.time() - start_time

    results[name] = {
        "validation_accuracy": val_accuracy,
        "validation_f1": val_f1
    }

    print("Validation accuracy:", round(val_accuracy, 4))
    print("Validation F1-score:", round(val_f1, 4))
    print("Training time:", round(elapsed, 2), "seconds")

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_model = model
        best_model_name = name


# 6. FINAL TEST EVALUATION
print("\n" + "=" * 50)
print("FINAL TEST EVALUATION")
print("=" * 50)

test_predictions = best_model.predict(X_test)

accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(
    y_test, test_predictions, zero_division=0
)
recall = recall_score(
    y_test, test_predictions, zero_division=0
)
f1 = f1_score(
    y_test, test_predictions, zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    y_test, test_predictions, labels=[0, 1]
).ravel()

false_positive_rate = fp / (fp + tn) if (fp + tn) else 0

print("Best model:", best_model_name)
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
print("False positive rate:", round(false_positive_rate, 4))

test_probabilities = best_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, test_probabilities)
print("ROC-AUC:", round(roc_auc, 4))
print("\nClassification report:")
print(classification_report(
    y_test,
    test_predictions,
    target_names=["Legitimate", "Phishing"],
    zero_division=0
))


# 7. SAVE TRAINED MODEL
joblib.dump(
    {
        "model": best_model,
        "feature_names": list(X.columns),
        "model_name": best_model_name,
        "positive_class": "Phishing"
    },
    "phishing_model.joblib"
)

print("\nTrained model saved as phishing_model.joblib")

print("\nMODEL TRAINING COMPLETED")
print("=" * 50)