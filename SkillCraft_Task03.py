# Task 03 - Decision Tree Classifier
# Predict whether a customer will purchase a product/service

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# --- User interaction ---
dataset_path = input("Enter dataset path (e.g., bank.csv): ").strip()
test_size = float(input("Enter test size (e.g., 0.3 for 30%): ").strip())
max_depth = int(input("Enter max depth for decision tree (e.g., 5): ").strip())
show_tree = input("Do you want to visualize the tree? (yes/no): ").strip().lower()

# 1. Load dataset
data = pd.read_csv(dataset_path, sep=";")

# 2. Encode categorical variables
data_encoded = pd.get_dummies(data, drop_first=True)

# 3. Separate features and target
X = data_encoded.drop("y_yes", axis=1)
y = data_encoded["y_yes"]

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# 5. Build and train decision tree
model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate model
y_pred = model.predict(X_test)
print("\nModel Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Visualize decision tree (optional)
if show_tree == "yes":
    plt.figure(figsize=(20,10))
    plot_tree(model, filled=True, feature_names=X.columns, class_names=["No", "Yes"])
    plt.show()
