import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

# 1. Load data
df = pd.read_csv("telco.csv")

# 2. Clean
df = df.drop("customerID", axis=1)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# 3. Encode categorical columns
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols.remove("Churn")
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop("Churn", axis=1)
y = df["Churn"]

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Scale (CRITICAL for KNN - distance based algorithm)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Define the three distance metrics to compare
metrics = {
    "Euclidean": {"metric": "minkowski", "p": 2},
    "Manhattan": {"metric": "minkowski", "p": 1},
    "Minkowski (p=3)": {"metric": "minkowski", "p": 3},
}

k_range = range(1, 31)
results = {}  # metric_name -> list of accuracies per k
best_k_per_metric = {}

for name, params in metrics.items():
    accs = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric=params["metric"], p=params["p"])
        knn.fit(X_train_scaled, y_train)
        pred = knn.predict(X_test_scaled)
        accs.append(accuracy_score(y_test, pred))
    results[name] = accs
    best_k = k_range[np.argmax(accs)]
    best_k_per_metric[name] = (best_k, max(accs))

# 7. Print best K summary
print("Best K per distance metric:")
for name, (k, acc) in best_k_per_metric.items():
    print(f"  {name}: best K = {k}, accuracy = {acc:.4f}")

# 8. Accuracy vs K plot for all three metrics
plt.figure(figsize=(9, 6))
for name, accs in results.items():
    plt.plot(list(k_range), accs, marker='o', markersize=3, label=name)
plt.xlabel("K (number of neighbors)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K for Different Distance Metrics (Telco Churn)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("accuracy_vs_k.png", dpi=150)
plt.close()

# 9. Confusion matrices for the BEST K of each metric
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, params) in zip(axes, metrics.items()):
    best_k = best_k_per_metric[name][0]
    knn = KNeighborsClassifier(n_neighbors=best_k, metric=params["metric"], p=params["p"])
    knn.fit(X_train_scaled, y_train)
    pred = knn.predict(X_test_scaled)
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"{name}\nBest K={best_k}, Acc={best_k_per_metric[name][1]:.3f}")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.close()

print("\nDone. Plots saved.")
