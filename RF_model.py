# -----------------------------
# Core Imports
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import shap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# -----------------------------
# Load Training Data
# -----------------------------
train = pd.read_csv("train.csv")
X = train.drop(["id", "species"], axis=1)
y = train["species"]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Number of classes:", len(le.classes_))

# -----------------------------
# Feature Groups
# -----------------------------
shape_features = [c for c in X.columns if "shape" in c]
margin_features = [c for c in X.columns if "margin" in c]
texture_features = [c for c in X.columns if "texture" in c]

X_shape = X[shape_features]
X_margin = X[margin_features]
X_texture = X[texture_features]
X_all = X

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# -----------------------------
# Train Model Function
# -----------------------------
def train_model(X_tr, X_te, y_tr, y_te, model_name="RandomForest"):
    if model_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    else:
        model = XGBClassifier(n_estimators=300, random_state=42, 
                              use_label_encoder=False, eval_metric='mlogloss')
    
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    print(f"{model_name} Accuracy: {acc:.4f}")
    return model, preds

# -----------------------------
# Train Models
# -----------------------------
rf_shape, pred_shape = train_model(X_train[shape_features], X_test[shape_features], y_train, y_test, "RF - Shape")
rf_margin, pred_margin = train_model(X_train[margin_features], X_test[margin_features], y_train, y_test, "RF - Margin")
rf_texture, pred_texture = train_model(X_train[texture_features], X_test[texture_features], y_train, y_test, "RF - Texture")
rf_all, pred_all = train_model(X_train, X_test, y_train, y_test, "RF - All Features")

print("\n🔬 Ablation Study Results")
print(f"Shape Only Accuracy:   {accuracy_score(y_test, pred_shape):.4f}")
print(f"Margin Only Accuracy:  {accuracy_score(y_test, pred_margin):.4f}")
print(f"Texture Only Accuracy: {accuracy_score(y_test, pred_texture):.4f}")
print(f"All Features Accuracy: {accuracy_score(y_test, pred_all):.4f}")

# -----------------------------
# Actual vs Predicted Comparison (for test split)
# -----------------------------
comparison = pd.DataFrame({
    "id": train.loc[X_test.index, "id"],  # original ids
    "Actual": le.inverse_transform(y_test),
    "Predicted": le.inverse_transform(pred_all)
})
comparison["Correct"] = comparison["Actual"] == comparison["Predicted"]

print("\nActual vs Predicted (first 20 rows):")
print(comparison.head(20))

# Optional: save comparison
comparison.to_csv("actual_vs_predicted.csv", index=False)

# -----------------------------
# SHAP Feature Importance
# -----------------------------
explainer = shap.TreeExplainer(rf_all)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=X_all.columns)

def group_shap_importance(model, X, groups):
    importances = model.feature_importances_
    group_scores = {}
    for name, cols in groups.items():
        idxs = [X.columns.get_loc(c) for c in cols]
        group_scores[name] = importances[idxs].sum()
    return group_scores

groups = {"Shape": shape_features, "Margin": margin_features, "Texture": texture_features}
print(group_shap_importance(rf_all, X_all, groups))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, pred_all)
plt.figure(figsize=(12,8))
sns.heatmap(cm, cmap="Blues", annot=False)
plt.title("Confusion Matrix - All Features")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# Dimensionality Reduction
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

X_pca = PCA(n_components=2).fit_transform(X_scaled)
X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_scaled)
X_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(X_scaled)

def plot_2d(X_red, y, title):
    plt.figure(figsize=(12,8))
    plt.scatter(X_red[:,0], X_red[:,1], c=y, cmap="tab20", s=15)
    plt.title(title)
    plt.show()

plot_2d(X_pca, y_encoded, "PCA - Leaf Feature Clusters")
plot_2d(X_tsne, y_encoded, "t-SNE - Leaf Feature Clusters")
plot_2d(X_umap, y_encoded, "UMAP - Leaf Feature Clusters")

# -----------------------------
# Cross-Validation
# -----------------------------
def cross_val_scores(X, y, model_name="RandomForest"):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        if model_name == "RandomForest":
            model = RandomForestClassifier(n_estimators=300, random_state=42)
        else:
            model = XGBClassifier(n_estimators=300, random_state=42,
                                  use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_tr, y_tr)
        scores.append(model.score(X_te, y_te))
    return scores

shape_scores = cross_val_scores(X_shape.values, y_encoded)
margin_scores = cross_val_scores(X_margin.values, y_encoded)
texture_scores = cross_val_scores(X_texture.values, y_encoded)

plt.boxplot([shape_scores, margin_scores, texture_scores], labels=["Shape", "Margin", "Texture"])
plt.title("Cross-Validation Accuracy Distribution")
plt.ylabel("Accuracy")
plt.show()

# -----------------------------
# Predict on test.csv and create submission
# -----------------------------
test = pd.read_csv("test.csv")
test_ids = test["id"]
X_test_data = test.drop(["id"], axis=1)

# Ensure columns match training features
X_test_data = X_test_data[X_all.columns]

pred_test_encoded = rf_all.predict(X_test_data)
pred_test_species = le.inverse_transform(pred_test_encoded)

submission = pd.DataFrame({
    "id": test_ids,
    "species": pred_test_species
})

submission.to_csv("leaf_species_predictions.csv", index=False)
print("Submission file created: leaf_species_predictions.csv")
submission.head()

import joblib
joblib.dump(rf_all, "rf_leaf_model.pkl")
print("✅ RF model saved!")
