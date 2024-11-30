from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
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
# print("Class Distribution after SMOTE:")
# print(y_train.value_counts(normalize=True))

# Train a Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict_proba(X_test)

# Evaluate the model using ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred[:, 1])
print(f"ROC AUC Score: {roc_auc}") # 0.59

# implement Bayesian optimization for hyperparameter tuning

# Define the objective function for optimization
def objective(trial):
    C = trial.suggest_loguniform('C', 1e-5, 1e5)
    Penalty = trial.suggest_categorical('Penalty', ['l1', 'l2'])
    max_iter = trial.suggest_int('max_iter', 100, 1000)

    model = LogisticRegression(C=C, max_iter=max_iter, random_state=42,
                               penalty=Penalty, solver='saga')
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, y_pred[:, 1])

    return roc_auc

# Create a study object and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Get the best trial
best_trial = study.best_trial
print("Best Trial:")
print(f"Value: {best_trial.value}")


# Print the best parameters
print("Best Parameters:")
for key, value in best_trial.params.items():
    print(f"{key}: {value}")

# get the coefficients of the best model
C = best_trial.params['C']
penalty = best_trial.params['Penalty']
max_iter = best_trial.params['max_iter']

best_model = LogisticRegression(C=C, max_iter=max_iter, random_state=42,
                                penalty=penalty, solver='saga')
best_model.fit(X_train, y_train)

# Get the feature names
feature_names = X.columns

# Get the feature importance
feature_importance = best_model.coef_[0]

# Create a summary table of feature names and their importance
feature_table = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
print("Feature Importance:")
print(feature_table)

# print the ROC AUC score of the best model on the test set
y_pred_best = best_model.predict_proba(X_test)
roc_auc_best = roc_auc_score(y_test, y_pred_best[:, 1])

print(f"ROC AUC Score (Best Model): {roc_auc_best}") # 0.59



