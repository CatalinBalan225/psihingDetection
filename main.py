import kagglehub
import numpy as np
import pandas as pd
import sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns


#path = kagglehub.dataset_download("ethancratchley/email-phishing-dataset")

#print("Path to dataset files:", path)
df = pd.read_csv('email_phishing_data.csv')
#print(df.head())

df_sampled = df.sample(frac=0.4, random_state=42)

x = df_sampled.drop(columns=['label'])
y = df_sampled['label']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

count = df_sampled['label'].value_counts()



plt.figure(figsize=(10, 8))
sns.heatmap(df_sampled.drop('label', axis=1).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig("heatmap.png", dpi=300)
plt.show()
plt.close()

pca = PCA(n_components=5)
x_pca = pca.fit_transform(x_scaled)
explained_variance = pca.explained_variance_ratio_
print("Variatie explicata de fiecare componenta")
for i, var in enumerate(explained_variance, 1):
    print(f"Componenta {i}: {var:.2f}")

print(f"\n Total variatie pastrata: {explained_variance.sum():.4f}")
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)

lr = LogisticRegression(class_weight='balanced')
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)

print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr, cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_lr.png", dpi=300)
plt.show()
plt.close()
#Decision Tree

dt = DecisionTreeClassifier(class_weight='balanced')
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)

print("\n Decision Tree")
print("Accuracy: ", accuracy_score(y_test, y_pred_dt))
print("Precision", precision_score(y_test, y_pred_dt))
print("Recall: ", recall_score(y_test, y_pred_dt))
print("F1 Score: ", f1_score(y_test, y_pred_dt))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr, cmap='Greens')
plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=300)
plt.show()
plt.close()

#Random Forest

rf= RandomForestClassifier(class_weight='balanced')
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

print("\n Random Forest")
print("Accuracy: ", accuracy_score(y_test, y_pred_rf))
print("Precision", precision_score(y_test, y_pred_rf))
print("Recall: ", recall_score(y_test, y_pred_rf))
print("F1 Score: ", f1_score(y_test, y_pred_rf))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rf, cmap='Purples')
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig("random_forest.png", dpi=300)
plt.show()
plt.close()

# Linear SVM
svm = LinearSVC(class_weight='balanced', max_iter=10000, random_state=42)
svm.fit(x_train, y_train)
y_pred_svm = svm.predict(x_test)

print("\n Linear SVM")
print("Accuracy: ", accuracy_score(y_test, y_pred_svm))
print("Precision", precision_score(y_test, y_pred_svm))
print("Recall: ", recall_score(y_test, y_pred_svm))
print("F1 Score: ", f1_score(y_test, y_pred_svm))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm, cmap='Oranges')
plt.title("Linear SVM Confusion Matrix")
plt.tight_layout()
plt.savefig("svm_linear.png", dpi=300)
plt.show()
plt.close()
#KNN

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
print("\n KNN")
print("Accuracy: ", accuracy_score(y_test, y_pred_knn))
print("Precision", precision_score(y_test, y_pred_knn))
print("Recall: ", recall_score(y_test, y_pred_knn))
print("F1 Score: ", f1_score(y_test, y_pred_knn))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_knn, cmap='Reds')
plt.title("KNN Confusion Matrix")
plt.tight_layout()
plt.savefig("knn.png", dpi=300)
plt.show()
plt.close()



models = ['LR', 'DT', 'RF', 'LinearSVM', 'KNN']
accuracy = [0.348, 0.974, 0.983, 0.344, 0.986]
precision = [0.015, 0.111, 0.179, 0.015, 0.264]
recall = [0.808, 0.148, 0.087, 0.808, 0.026]
f1 = [0.030, 0.127, 0.117, 0.030, 0.048]

x = range(len(models))

plt.figure(figsize=(12, 6))
plt.bar(x, accuracy, width=0.2, label='Accuracy', align='center')
plt.bar([i + 0.2 for i in x], precision, width=0.2, label='Precision', align='center')
plt.bar([i + 0.4 for i in x], recall, width=0.2, label='Recall', align='center')
plt.bar([i + 0.6 for i in x], f1, width=0.2, label='F1 Score', align='center')

plt.xticks([i + 0.3 for i in x], models)
plt.ylabel('Score')
plt.title('Comparatie Performanta Modele')
plt.legend()
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300)
plt.show()
plt.close()


