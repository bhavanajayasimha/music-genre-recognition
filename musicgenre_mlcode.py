!pip install catboost imblearn

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

print("🚀 Loading dataset...")
df = pd.read_csv("dataset.csv")

# Fix '?' values
df = df.replace('?', np.nan)

# Fill numeric NaN with median
for col in df.select_dtypes(include=['float64','int64']).columns:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical NaN with mode
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Rows after cleaning:", len(df))

# Encode categorical
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

target = df.columns[-1]
X = df.drop(columns=[target])
y = df[target]

# Fix imbalance
X, y = SMOTE().fit_resample(X, y)

print("Rows after SMOTE:", len(X))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== FAST CATBOOST =====
model = CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.05,
    loss_function="MultiClass",
    verbose=False,
    random_seed=42
)

print("🔥 Training...")
model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=100)

# Predict
y_pred = model.predict(X_test)

print("\n🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))