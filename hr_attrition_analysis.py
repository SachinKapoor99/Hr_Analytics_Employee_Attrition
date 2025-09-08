# ================================
# HR Analytics - Predict Employee Attrition
# Tools: Python (Pandas, Seaborn, Sklearn, Matplotlib)
# ================================

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ================================
# 1. Load Data
# ================================
# Use raw string (r"...") to avoid escape sequence error
df = pd.read_excel(r"D:\hr analytics data 1.xlsx", sheet_name="Employee_Performance_Dataset")

print("Dataset Shape:", df.shape)
print("Columns:", df.columns)

# ================================
# 2. Data Preprocessing
# ================================
# Drop ID, Name (not useful for prediction)
df = df.drop(["Employee ID", "Name"], axis=1)

# Encode categorical variables
le = LabelEncoder()
df["Department"] = le.fit_transform(df["Department"])
df["Job Role"] = le.fit_transform(df["Job Role"])
df["Promotion Eligibility"] = le.fit_transform(df["Promotion Eligibility"])
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Features & Target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ================================
# 3. EDA - Visualizations
# ================================
# Attrition Count
sns.countplot(x="Attrition", data=df)
plt.title("Attrition Count")
plt.show()

# Department-wise Attrition
sns.countplot(x="Department", hue="Attrition", data=df)
plt.title("Department-wise Attrition")
plt.show()

# Job Role-wise Attrition
sns.countplot(y="Job Role", hue="Attrition", data=df)
plt.title("Job Role-wise Attrition")
plt.show()

# Attendance vs Attrition
sns.boxplot(x="Attrition", y="Attendance (%)", data=df)
plt.title("Attendance vs Attrition")
plt.show()

# Training Hours vs Attrition
sns.boxplot(x="Attrition", y="Training Hours", data=df)
plt.title("Training Hours vs Attrition")
plt.show()

# Work Hours vs Attrition
sns.boxplot(x="Attrition", y="Work Hours Logged", data=df)
plt.title("Work Hours vs Attrition")
plt.show()

# Peer Rating vs Attrition
sns.boxplot(x="Attrition", y="Peer Rating", data=df)
plt.title("Peer Rating vs Attrition")
plt.show()

# Manager Feedback vs Attrition
sns.boxplot(x="Attrition", y="Manager Feedback", data=df)
plt.title("Manager Feedback vs Attrition")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ================================
# 4. Machine Learning Models
# ================================

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("\n🔹 Logistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))

# Decision Tree
tree_model = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

print("\n🔹 Decision Tree Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))
print("Classification Report:\n", classification_report(y_test, y_pred_tree))

# Plot Decision Tree
plt.figure(figsize=(15, 8))
plot_tree(tree_model, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.show()

# ================================
# 5. Export Cleaned Data for Tableau
# ================================
df.to_csv("cleaned_hr_dataset.csv", index=False)
print("\n✅ Cleaned dataset exported as 'cleaned_hr_dataset.csv' for Tableau.")