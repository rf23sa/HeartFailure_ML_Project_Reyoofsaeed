import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                            roc_auc_score, roc_curve, precision_recall_curve, auc,
                            precision_score, recall_score, f1_score)

# Set random seed for complete reproducibility
np.random.seed(42)

# Load the dataset
data = pd.read_excel('original data/heart_failure_clinical_records_dataset.xlsx')

# --- Data Properties ---
print("===== Dataset Overview =====")
print(f"Number of samples: {data.shape[0]}")
print(f"Number of attributes: {data.shape[1]}")
print("\nFirst 5 rows:")
print(data.head())
print("\nData types:")
print(data.dtypes)
print("\nSummary statistics:")
print(data.describe())
print("\nClass distribution (DEATH_EVENT):")
print(data['DEATH_EVENT'].value_counts())

# Feature Distributions
plt.figure(figsize=(12, 8))
for i, column in enumerate(data.columns[:-1]):  # Exclude target
    plt.subplot(3, 4, i+1)
    sns.histplot(data[column], kde=True, color='skyblue')
    plt.title(column)
plt.tight_layout()
plt.savefig('simple_feature_distributions.png')
plt.show()

# Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='DEATH_EVENT', data=data, palette='pastel')
plt.title('Distribution of Death Events')
plt.savefig('simple_target_distribution.png')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation')
plt.savefig('simple_correlation.png')
plt.show()

# Boxplots for Key Features
plt.figure(figsize=(12, 6))
features_to_plot = ['age', 'ejection_fraction', 'serum_creatinine']
for i, feature in enumerate(features_to_plot):
    plt.subplot(1, 3, i+1)
    sns.boxplot(x='DEATH_EVENT', y=feature, data=data, palette='pastel')
    plt.title(f'{feature} by Death Event')
plt.tight_layout()
plt.savefig('simple_boxplots.png')
plt.show()

# --- Preprocessing ---
X = data.drop(columns=['DEATH_EVENT'])
y = data['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save preprocessed data
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('preprocessed/X.csv', index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('preprocessed/X_test.csv', index=False)
y_train.to_csv('preprocessed/Y.csv', index=False)
y_test.to_csv('preprocessed/Y_test.csv', index=False)

# --- Model Training & Evaluation ---
models = {
    'LogisticRegression': LogisticRegression(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'KNeighbors': KNeighborsClassifier(),
    'NaiveBayes': GaussianNB(),
    'ANN': MLPClassifier(max_iter=1000, random_state=42)
}

results = {
    'Model': [],
    'Accuracy': [],
    'ROC AUC': [],
    'Precision': [],
    'Recall': [],
    'F1': []
}

# Train and evaluate each model
plt.figure(figsize=(12, 10))
for model_name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else [0] * len(y_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Save predictions
    pd.DataFrame(y_pred, columns=['Prediction']).to_csv(f'results/prediction_{model_name}.csv', index=False)
    
    # Store results
    results['Model'].append(model_name)
    results['Accuracy'].append(accuracy)
    results['ROC AUC'].append(roc_auc)
    results['Precision'].append(precision)
    results['Recall'].append(recall)
    results['F1'].append(f1)
    
    # Print classification report 
    print(f"\n===== {model_name} =====")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.subplot(2, 1, 1)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    # Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    plt.subplot(2, 1, 2)
    plt.plot(recall_curve, precision_curve, label=f'{model_name}')

# Format ROC Curve plot
plt.subplot(2, 1, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')

# Format Precision-Recall plot
plt.subplot(2, 1, 2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('advanced_model_performance.png')
plt.show()

# --- Model Comparison (Bar Plot) ---
results_df = pd.DataFrame(results)
plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x='Model', y='Accuracy', palette='viridis')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.savefig('model_accuracy_barplot.png')
plt.show()

# --- Confusion Matrices for All Models ---
plt.figure(figsize=(18, 12))
plt.suptitle('Confusion Matrices for All Models', fontsize=16, y=1.02)
n_rows = 3
n_cols = 3

for idx, (model_name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.subplot(n_rows, n_cols, idx+1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Survived', 'Died'], 
                yticklabels=['Survived', 'Died'])
    plt.title(f"{model_name}\nAccuracy: {results_df.loc[results_df['Model']==model_name, 'Accuracy'].values[0]:.2%}",
              fontsize=10)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

plt.tight_layout()
plt.savefig('Models_confusion_matrices.png', bbox_inches='tight', dpi=300)
plt.show()

# --- Classification Report Heatmaps ---
plt.figure(figsize=(18, 12))
plt.suptitle('Classification Heatmaps', fontsize=16, y=1.02)

n_models = len(models)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols

for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test_scaled)
    plt.subplot(n_rows, n_cols, i+1)
    
    report = classification_report(y_test, y_pred, 
                                 output_dict=True,
                                 target_names=['Survived', 'Died'])
    
    report_df = pd.DataFrame(report).iloc[:-1, :3].T
    
    sns.heatmap(report_df, annot=True, cmap='YlOrRd', 
                vmin=0, vmax=1, 
                annot_kws={'size': 10},
                fmt='.2f',
                linewidths=0.5)
    
    plt.title(f'{name}', pad=10)
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('Classification Heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Save Complete Results ---
results_df.to_csv('model_performance_summary.csv', index=False)
print("\n=== Model Performance Summary ===")
print(results_df.sort_values('Accuracy', ascending=False))