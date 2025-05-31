# phishingDetection

This project applies machine learning techniques in Python to detect phishing emails based on various structural, textual, and quality-related features. The aim is to train and evaluate multiple classification models that are capable of distinguishing between phishing and legitimate emails.

---

## Dataset Overview

The dataset, downloaded from Kaggle, contains labeled email records enriched with several numerical features. Among these, we find indicators such as the total number of words, unique word count, stopword count, number of links, domain diversity, and spelling errors. The label column denotes whether the email is phishing (`1`) or not (`0`).

One key challenge of this dataset is class imbalance: phishing emails make up only around 20% of the total, which significantly impacts the evaluation of model performance. This issue is further addressed in later steps.

---

## Step 1 – Data Preprocessing

To reduce computation time and allow for more efficient experimentation, only 40% of the full dataset was sampled. Standardization was applied using `StandardScaler` to normalize the feature values.

```python
df_sampled = df.sample(frac=0.4, random_state=42)

x = df_sampled.drop(columns=['label'])
y = df_sampled['label']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

print(df_sampled['label'].value_counts())
```

Given the high number of features, a correlation heatmap was created to visually inspect feature relationships before applying dimensionality reduction. Highly correlated variables were identified and then reduced via Principal Component Analysis (PCA).

```python
plt.figure(figsize=(10, 8))
sns.heatmap(df_sampled.drop('label', axis=1).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig("heatmap.png", dpi=300)
plt.close()
```

---

## Step 2 – Dimensionality Reduction with PCA

PCA was used to reduce the original 8 features down to 5 components, retaining as much information as possible. The resulting components explained about 95% of the original variance, which was deemed sufficient for training purposes.

```python
pca = PCA(n_components=5)
x_pca = pca.fit_transform(x_scaled)
explained_variance = pca.explained_variance_ratio_

for i, var in enumerate(explained_variance, 1):
    print(f"Component {i}: {var:.2f}")
print(f"Total variance retained: {explained_variance.sum():.4f}")
```

The output showed the following explained variance per component:

- Component 1: 0.50  
- Component 2: 0.16  
- Component 3: 0.13  
- Component 4: 0.09  
- Component 5: 0.07  

Total variance retained: 94.78%

This dimensionality reduction step helped simplify the problem space and accelerated model training.

---

## Step 3 – Model Training and Evaluation

With the PCA-transformed data, the following machine learning models were trained individually:

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Linear Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  

Each model was evaluated based on four metrics: **accuracy**, **precision**, **recall**, and **F1 score**.

```markdown
| Model              | Accuracy | Precision | Recall | F1 Score |
|-------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.348    | 0.015     | 0.808  | 0.030    |
| Decision Tree       | 0.974    | 0.111     | 0.148  | 0.127    |
| Random Forest       | 0.983    | 0.179     | 0.087  | 0.117    |
| Linear SVM          | 0.344    | 0.015     | 0.808  | 0.030    |
| KNN                 | 0.986    | 0.264     | 0.026  | 0.048    |
```

Despite the high accuracy seen in models like Random Forest or KNN, their precision and recall remain low, particularly due to the dataset imbalance. This is why F1 Score is considered a more balanced metric for evaluating their performance in this context.

---

## Step 4 – Visual Analysis and Comparison

A bar chart was created to provide a comparative view of the models' performance across all four metrics. This allows for easier identification of trade-offs between precision and recall.

```python
plt.figure(figsize=(12, 6))
plt.bar(x, accuracy, width=0.2, label='Accuracy', align='center')
plt.bar([i + 0.2 for i in x], precision, width=0.2, label='Precision', align='center')
plt.bar([i + 0.4 for i in x], recall, width=0.2, label='Recall', align='center')
plt.bar([i + 0.6 for i in x], f1, width=0.2, label='F1 Score', align='center')
plt.xticks([i + 0.3 for i in x], models)
plt.ylabel('Score')
plt.title('Comparative Performance of Models')
plt.legend()
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300)
plt.close()
```


Additionally, confusion matrices were generated for each classifier to visually inspect true positives, false positives, and other classification outcomes. For each model, the following code snippet was used:

```python
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_model, cmap='Purples')
plt.title("Confusion Matrix - ModelName")
plt.tight_layout()
plt.savefig("confusion_matrix_modelname.png", dpi=300)
plt.close()
```
---

## Conclusion

This project demonstrated that while basic classifiers like Logistic Regression or Linear SVM suffer heavily due to class imbalance, tree-based models and instance-based learners perform much better in terms of overall accuracy. However, their ability to correctly identify phishing emails remains limited.

Given the high recall but low precision seen in some models, it becomes clear that the evaluation metric must be chosen carefully. F1 score proved to be the most reliable metric here due to the imbalanced nature of the data.

---

## Project Assets

All generated visualizations are included in the repository:
- `heatmap.png`
- `model_comparison.png`
- `confusion_matrix_randomforest.png`
- `confusion_matrix_logistic.png`
- `confusion_matrix_svm.png`
- `confusion_matrix_knn.png`
- `confusion_matrix_tree.png`

---

