# Re-import necessary libraries and re-establish context after system interruption
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd
import optuna
import imblearn

# Reload the data
file_path = 'data/depression/depression_data.csv'
data = pd.read_csv(file_path)

# data.drop('Name', axis=1, inplace=True)

data['History of Mental Illness'].value_counts(normalize=True) # unbalanced dataset so will employ SMOTE

# Encode categorical variables using LabelEncoder
label_encoders = {}
for column in data.select_dtypes(include='object').columns:
    if column != 'History of Mental Illness':  # Target variable handled separately
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

# Encode the target variable (History of Mental Illness)
target_encoder = LabelEncoder()
data['History of Mental Illness'] = target_encoder.fit_transform(data['History of Mental Illness'])

# Separate features (X) and target variable (y)
X = data.drop(['History of Mental Illness', 'Name'], axis=1)  # Drop Name as it's not a feature
y = data['History of Mental Illness']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardise the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply SMOTE to balance the training data
smote = imblearn.over_sampling.SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# check the class distribution after applying SMOTE
print("Class Distribution after SMOTE:")
print(y_train.value_counts(normalize=True))

# Train a Random Forest Classifier
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)
#
# # Make predictions
# y_pred = model.predict_proba(X_test)
#
# # Evaluate the model using ROC AUC score
# roc_auc = roc_auc_score(y_test, y_pred[:, 1])
# print(f"ROC AUC Score: {roc_auc}") # 0.56

# implement Bayesian optimization for hyperparameter tuning

# split training data into training and validation sets

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Define the objective function

def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)
    roc_auc = roc_auc_score(y_val, y_pred[:, 1])

    return roc_auc


# Perform hyperparameter optimization using Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params = study.best_params
best_score = study.best_value

print(f"Best ROC AUC Score: {best_score}")
print(f"Optimized Hyperparameters: {best_params}")

# Train a new model using the best hyperparameters
best_model = RandomForestClassifier(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# Make predictions with the best model

y_pred_best = best_model.predict_proba(X_test)

# Evaluate the best model using ROC AUC score
roc_auc_best = roc_auc_score(y_test, y_pred_best[:, 1])

print(f"ROC AUC Score (Best Model): {roc_auc_best}")

'''
Best ROC AUC Score: 0.6028017945076724
Optimized Hyperparameters: {'n_estimators': 996, 'max_depth': 4, 'min_samples_split': 16, 'min_samples_leaf': 4}
ROC AUC Score (Best Model): 0.5980090075131111
'''

# get the feature importances
feature_importances = best_model.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort the features by importance
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Display the feature importances
print("Feature Importances:")
print(feature_importance_df)


