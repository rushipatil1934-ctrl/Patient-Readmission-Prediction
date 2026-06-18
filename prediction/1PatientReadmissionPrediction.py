import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,LabelBinarizer,OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

# data setup
d=pd.read_csv('/Users/adinathpatil/Downloads/archive (1)/diabetic_data.csv',na_values=['?'])


#EDA
missing = d.isnull().sum()
missing_pct = (missing / len(d)) * 100
print(missing_pct[missing_pct > 0].sort_values(ascending=False))

# data preprocessing
d['readmitted_30'] = (d['readmitted'] == '<30').astype(int)
age_order = [['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)',
              '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']]

aoe = OrdinalEncoder(categories=age_order)
d.drop(columns=['encounter_id','patient_nbr','A1Cresult','max_glu_serum','weight','medical_specialty','payer_code','readmitted'], inplace=True)
oe=OrdinalEncoder()
imputer = IterativeImputer()
ohe=OneHotEncoder(drop='first', handle_unknown='ignore')
ore = OrdinalEncoder()
orre=OrdinalEncoder()
d['diag_3'] = pd.to_numeric(d['diag_3'], errors='coerce')
d['diag_2'] = pd.to_numeric(d['diag_2'], errors='coerce')
d['diag_1'] = pd.to_numeric(d['diag_1'], errors='coerce')

drug_cols = [
    'metformin','repaglinide','nateglinide','chlorpropamide','glimepiride',
    'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone',
    'rosiglitazone','acarbose','miglitol','troglitazone','tolazamide',
    'examide','citoglipton','insulin','glyburide-metformin',
    'glipizide-metformin','glimepiride-pioglitazone',
    'metformin-rosiglitazone','metformin-pioglitazone',
    'change','diabetesMed'
]


ct=ColumnTransformer(
    transformers=[
        ('gen',ohe,['gender']),

        ('age',aoe,['age']),

        ('race', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), ['race']),

        ('drugs', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), drug_cols),
        
    
        ('imputer', SimpleImputer(strategy='most_frequent'), ['diag_1', 'diag_2', 'diag_3']),
        ('admit', OneHotEncoder(handle_unknown='ignore'), 
         ['admission_type_id', 'discharge_disposition_id', 'admission_source_id']),

    ],remainder='passthrough'
)



x = d.drop(columns=['readmitted_30'])
y = d['readmitted_30']

xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=42)

xtr_t = ct.fit_transform(xtr)
xte_t = ct.transform(xte)

#MODEL
from sklearn.metrics import (classification_report, roc_auc_score,
                              average_precision_score, ConfusionMatrixDisplay)
import xgboost as xgb
neg = (ytr == 0).sum()
pos = (ytr== 1).sum()
scale = neg / pos 
model = xgb.XGBClassifier(
    scale_pos_weight=scale,
    eval_metric='aucpr',
    random_state=42,
    subsample=.7,
    n_estimators=600,
    learning_rate=.01,
    gamma=.1,
    max_depth=6,
    min_child_weight=5,
    colsample_bytree=1,
    n_jobs=-1,


)
model.fit(xtr_t, ytr)
y_pred = model.predict(xte_t)
y_prob = model.predict_proba(xte_t)[:, 1]

print(classification_report(yte, y_pred))
print("AUC-ROC:", round(roc_auc_score(yte, y_prob), 4))
print("Avg Precision (AP):", round(average_precision_score(yte, y_prob), 4))
print("accur",accuracy_score(yte,y_pred))

ConfusionMatrixDisplay.from_predictions(yte, y_pred)
plt.title('Confusion matrix')
plt.show()
import shap

explainer = shap.Explainer(model)
shap_values = explainer(xte_t)
shap.plots.bar(shap_values)
shap.plots.beeswarm(shap_values)
shap.plots.waterfall(shap_values[0])
# from sklearn.model_selection import RandomizedSearchCV

# params = {
#     'n_estimators': [200, 400, 600],
#     'max_depth': [4, 6, 8],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'subsample': [0.7, 0.8, 1.0],
#     'colsample_bytree': [0.7, 0.8, 1.0],
#     'min_child_weight': [1, 3, 5],
#     'gamma': [0, 0.1, 0.3]
# }

# search = RandomizedSearchCV(
#     XGBClassifier(scale_pos_weight=scale, eval_metric='aucpr', random_state=42),
#     param_distributions=params,
#     n_iter=30,           # tries 30 random combos
#     scoring='roc_auc',
#     cv=3,
#     verbose=1,
#     n_jobs=-1,
#     random_state=42
# )
# search.fit(xtr_t, ytr)
# print("Best params:", search.best_params_)
# best_model = search.best_estimator_
# print(search.best_score_)





# print(d.isnull().sum())
# target_counts = d['readmitted'].value_counts()
# target_counts.plot(kind='bar', color=['#4C72B0', '#DD8452', '#55A868'])
# plt.title('Readmission class distribution')
# plt.xlabel('Readmitted')
# plt.ylabel('Count')
# plt.xticks(rotation=0)
# plt.tight_layout()
# plt.show()

# numeric_cols = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
#                 'num_medications', 'number_diagnoses', 'number_outpatient',
#                 'number_emergency', 'number_inpatient']

# d[numeric_cols].hist(bins=30, figsize=(14, 8), color='steelblue', edgecolor='white')
# plt.suptitle('Numeric feature distributions')
# plt.tight_layout()
# plt.show()

# cat_cols = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id']

# fig, axes = plt.subplots(2, 3, figsize=(16, 8))
# for ax, col in zip(axes.flat, cat_cols):
#     d[col].value_counts().plot(kind='bar', ax=ax, color='steelblue')
#     ax.set_title(col)
#     ax.tick_params(axis='x', rotation=45)
# plt.tight_layout()
# plt.show()



# # By age group
# d.groupby('age')['readmitted_30'].mean().sort_index().plot(
#     kind='bar', color='steelblue', figsize=(10, 4))
# plt.title('Readmission rate by age group')
# plt.ylabel('Rate (<30 days)')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # By number of inpatient visits
# d.groupby('number_inpatient')['readmitted_30'].mean().plot(
#     kind='bar', color='coral', figsize=(10, 4))
# plt.title('Readmission rate by prior inpatient visits')
# plt.tight_layout()
# plt.show()