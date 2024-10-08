import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

df = pd.read_csv('BankChurnPrediction/data.csv')

df['BalanceSalaryRatio'] = df['Balance'] / df['EstimatedSalary']
df['TenureByAge'] = df['Tenure'] / df['Age']
df['CreditScorePerAge'] = df['CreditScore'] / df['Age']

df.drop(['RowNumber', 'CustomerId', 'Surname', 'Geography'], axis = 1, inplace = True)

cat_cols = ['Gender', 'HasCrCard', 'IsActiveMember']
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio', 'TenureByAge', 'CreditScorePerAge']

for col in num_cols:
    df[col] = df[col].replace('[^0-9.]', '', regex = True)
    df[col] = pd.to_numeric(df[col], errors = 'coerce')

df.fillna(0, inplace = True)

crit_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'BalanceSalaryRatio', 'TenureByAge', 'CreditScorePerAge', 'Gender']

X = df[crit_features]
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

preprocessor = ColumnTransformer(
    transformers = [
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop = 'first'), cat_cols)
    ]
)

clf1 = LogisticRegression(random_state = 42, class_weight = 'balanced')
clf2 = RandomForestClassifier(random_state = 42, class_weight = 'balanced', n_estimators = 100)
clf3 = AdaBoostClassifier(estimator = DecisionTreeClassifier(max_depth = 1, random_state = 42), n_estimators = 100, random_state = 42)
clf4 = LGBMClassifier(random_state = 42, class_weight = 'balanced', n_estimators = 100)

ensemble = VotingClassifier(
    estimators = [
        ('lr', clf1),
        ('rfc', clf2),
        ('abc', clf3),
        ('lgb', clf4),
    ],
    voting = 'soft'
)

pipeline = Pipeline(
    steps = [
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state = 42)),
        ('ensemble', ensemble)
    ]
)

pipeline.fit(X_train, y_train)
y_prob = pipeline.predict_proba(X_test)[:, 1]

y_prob = y_prob.astype(np.float64)
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

plt.figure(figsize = (8, 6))
plt.plot(recall, precision, marker = '.', label = 'Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

plt.figure(figsize = (8, 6))
plt.plot(thresholds, precision[:-1], 'b--', label = 'Precision')
plt.plot(thresholds, recall[:-1], 'g-', label = 'Recall')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.title('Precision-Recall vs Threshold')
plt.legend(loc = 'best')
plt.show()

threshold = thresholds[np.argmax(2 * (precision * recall) / (precision + recall))]
y_pred = (y_prob >= threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC AUC Score: {roc_auc}')

plt.figure(figsize = (8, 6))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

with open('model.pkl', 'wb') as f:
    pickle.dump({'model': pipeline, 'threshold': threshold}, f)