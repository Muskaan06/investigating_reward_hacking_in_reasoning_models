import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ============================================================
# Configuration
# ============================================================

CLASS_0_FILE = "final_token_activations_free.npy"
CLASS_1_FILE = "final_token_activations_modulu_2.npy"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================
# Load activations
# ============================================================

class_0 = np.load(CLASS_0_FILE)
class_1 = np.load(CLASS_1_FILE)

print("Class 0 shape:", class_0.shape)
print("Class 1 shape:", class_1.shape)

# ============================================================
# Create labels
# ============================================================

y_0 = np.zeros(len(class_0), dtype=np.int64)
y_1 = np.ones(len(class_1), dtype=np.int64)

# ============================================================
# Combine the two classes
# ============================================================

X = np.concatenate([class_0, class_1], axis=0)
y = np.concatenate([y_0, y_1], axis=0)

print("Combined X shape:", X.shape)
print("Combined y shape:", y.shape)

# ============================================================
# Train / test split
#
# stratify=y ensures that the class proportions are preserved
# in both the training and testing sets.
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
    stratify=y
)

print("\nTrain samples:", len(X_train))
print("Test samples:", len(X_test))

print("\nTraining class distribution:")
print("Class 0:", np.sum(y_train == 0))
print("Class 1:", np.sum(y_train == 1))

print("\nTest class distribution:")
print("Class 0:", np.sum(y_test == 0))
print("Class 1:", np.sum(y_test == 1))

# ============================================================
# Train Logistic Regression
# ============================================================

clf = LogisticRegression(
    max_iter=5000,
    solver="lbfgs"
)

clf.fit(X_train, y_train)

print("\nLogistic regression trained.")

# ============================================================
# Predictions
# ============================================================

y_pred = clf.predict(X_test)

# Probability of class 1
y_prob = clf.predict_proba(X_test)[:, 1]

# ============================================================
# Evaluation
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n==============================")
print("        TEST RESULTS")
print("==============================")

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC:  {auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Class 0", "Class 1"]
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ============================================================
# Optional: save the trained classifier
# ============================================================

import joblib

joblib.dump(clf, "logistic_regression_classifier.pkl")

print("\nClassifier saved to:")
print("logistic_regression_classifier.pkl")